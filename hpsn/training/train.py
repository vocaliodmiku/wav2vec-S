"""HPSN training entry point. Uses HuggingFace Accelerate for mixed-precision + multi-GPU."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup

from ..config import HPSNConfig
from ..model.backbone import FrozenWav2VecS
from ..model.hpsn import HPSNMinimal
from .data import build_dataloader
from .loss import HPSNLoss


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    defaults = HPSNConfig()
    for f in dataclasses.fields(HPSNConfig):
        t = f.type if not isinstance(f.type, str) else type(getattr(defaults, f.name))
        if t is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in {"1", "true", "yes"}, default=getattr(defaults, f.name))
        else:
            p.add_argument(f"--{f.name}", type=t, default=getattr(defaults, f.name))
    return p


def main():
    args = _build_argparser().parse_args()
    config = HPSNConfig(**vars(args))
    set_seed(config.seed)

    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

    # Feature extractor (processor.tokenizer isn't needed here).
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.backbone_model)

    # Frozen backbone (bf16 weights, no grad).
    backbone = FrozenWav2VecS(
        config.backbone_model,
        main_context=config.main_context,
        right_context=config.right_context,
        dtype=torch.bfloat16,
    )
    backbone.to(device)

    # Align config.hidden_dim to the actual backbone.
    if config.hidden_dim != backbone.hidden_size:
        if accelerator.is_main_process:
            accelerator.print(
                f"[info] hidden_dim {config.hidden_dim} != backbone {backbone.hidden_size}; using backbone size."
            )
        config.hidden_dim = backbone.hidden_size

    # Validate tap ranges against backbone depth (hidden_states length = N + 1).
    max_layer_idx = backbone.num_hidden_layers
    for name, end in [
        ("acoustic", config.tap_acoustic_end),
        ("lexical", config.tap_lexical_end),
    ]:
        if end > max_layer_idx:
            raise ValueError(f"tap_{name}_end={end} exceeds backbone layers={max_layer_idx}")

    model = HPSNMinimal(config)
    loss_fn = HPSNLoss(config.lambda1, config.lambda2, config.loss_type)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    dataloader = build_dataloader(
        root=config.libri_root,
        feature_extractor=feature_extractor,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
        max_audio_seconds=config.max_audio_seconds,
        max_samples=config.max_train_samples,
        shuffle=True,
        sample_rate=config.sample_rate,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"[info] HPSN trainable parameters: {n_params/1e6:.2f}M")
        accelerator.print(f"[info] dataset size: {len(dataloader.dataset)} utterances")

    model.train()
    step = 0
    last_log_time = time.time()
    running = {"total": 0.0, "recon1": 0.0, "recon2": 0.0, "count": 0}
    done = False

    while not done:
        for batch in dataloader:
            with accelerator.accumulate(model):
                input_values = batch["input_values"]
                attention_mask = batch["attention_mask"]

                with torch.no_grad():
                    with accelerator.autocast():
                        hidden_states = backbone(input_values, attention_mask=attention_mask)
                # Cast to fp32 for HPSN training.
                hidden_states = tuple(h.float() for h in hidden_states)

                outputs = model(hidden_states, attention_mask=attention_mask)
                losses = loss_fn(outputs)
                loss = losses["total"]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Accumulate for logging.
            running["total"] += loss.detach().float().item()
            running["recon1"] += losses["recon1"].float().item()
            running["recon2"] += losses["recon2"].float().item()
            running["count"] += 1

            if accelerator.sync_gradients:
                step += 1

                if step % config.logging_steps == 0 and accelerator.is_main_process:
                    n = max(running["count"], 1)
                    dt = time.time() - last_log_time
                    lr = scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"step {step:>7d} | lr {lr:.2e} | total {running['total']/n:.4f} "
                        f"| recon1 {running['recon1']/n:.4f} | recon2 {running['recon2']/n:.4f} "
                        f"| {n/max(dt,1e-6):.2f} it/s"
                    )
                    running = {"total": 0.0, "recon1": 0.0, "recon2": 0.0, "count": 0}
                    last_log_time = time.time()

                if step % config.save_steps == 0:
                    ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        accelerator.print(f"[info] saved checkpoint to {ckpt_dir}")

                if step >= config.max_steps:
                    done = True
                    break

    # Final save.
    final_dir = os.path.join(config.output_dir, "final")
    accelerator.save_state(final_dir)
    if accelerator.is_main_process:
        accelerator.print(f"[info] training complete. final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
