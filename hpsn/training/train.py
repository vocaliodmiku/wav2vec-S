"""HPSN training entry point. Uses HuggingFace Accelerate for mixed-precision + multi-GPU."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup

from ..config import HPSNConfig
from ..model.backbone import FrozenWav2VecS
from ..model.hpsn import HPSNMinimal
from .data import build_dataloader
from .loss import HPSNLoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_autocast_dtype(s: str):
    s = s.lower()
    if s in {"fp32", "float32", "none", ""}:
        return None
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"hpsn_dtype must be one of fp32/bf16/fp16, got '{s}'")


class RegionTimer:
    """Sync + wall-clock timing grouped by region. Safe for CUDA; sync per region."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self.device = device
        self.is_cuda = enabled and device.type == "cuda"
        self.totals: "OrderedDict[str, float]" = OrderedDict()
        self.counts: "OrderedDict[str, int]" = OrderedDict()

    @contextmanager
    def region(self, name: str):
        if not self.enabled:
            yield
            return
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.is_cuda:
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, n_steps: int) -> str:
        if not self.totals:
            return "(no profile data)"
        total = sum(self.totals.values())
        lines = [
            f"=== profile over {n_steps} optimizer steps ({total:.2f}s cumulative instrumented time) ===",
            f"{'region':<24} {'total_s':>10} {'per_call_ms':>14} {'calls':>8} {'%':>7}",
        ]
        for name, tot in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            calls = self.counts[name]
            lines.append(
                f"{name:<24} {tot:>10.3f} {1000*tot/calls:>14.2f} {calls:>8d} {100*tot/total:>6.1f}%"
            )
        return "\n".join(lines)


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
                f"[{_timestamp()}] [info] hidden_dim {config.hidden_dim} != backbone {backbone.hidden_size}; using backbone size."
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
        accelerator.print(f"[{_timestamp()}] [info] HPSN trainable parameters: {n_params/1e6:.2f}M")
        accelerator.print(f"[{_timestamp()}] [info] dataset size: {len(dataloader.dataset)} utterances")
        if config.profile:
            accelerator.print(
                f"[{_timestamp()}] [info] profile mode ON — will stop after {config.profile_steps} optimizer steps."
            )

    profiler = RegionTimer(enabled=config.profile, device=accelerator.device)
    # Expose profiler to HPSN submodules for inner timings.
    accelerator.unwrap_model(model)._profiler = profiler if config.profile else None

    hpsn_compute_dtype = _parse_autocast_dtype(config.hpsn_dtype)
    if hpsn_compute_dtype is not None and accelerator.is_main_process:
        accelerator.print(
            f"[{_timestamp()}] [info] HPSN head autocast dtype: {hpsn_compute_dtype} (params remain fp32, loss in fp32)."
        )

    model.train()
    step = 0
    last_log_time = time.time()
    running = {"total": 0.0, "recon1": 0.0, "recon2": 0.0, "count": 0}
    done = False

    data_iter_t0 = time.perf_counter() if config.profile else None

    while not done:
        for batch in dataloader:
            if config.profile:
                # Time spent waiting for the dataloader since the last micro-batch finished.
                dt_data = time.perf_counter() - data_iter_t0
                profiler.totals["data_wait"] = profiler.totals.get("data_wait", 0.0) + dt_data
                profiler.counts["data_wait"] = profiler.counts.get("data_wait", 0) + 1

            with accelerator.accumulate(model):
                input_values = batch["input_values"]
                attention_mask = batch["attention_mask"]

                with profiler.region("backbone"):
                    with torch.no_grad():
                        with accelerator.autocast():
                            hidden_states = backbone(input_values, attention_mask=attention_mask)
                    # Keep backbone states in bf16; LayerTap casts only the tapped bands to fp32.

                with profiler.region("hpsn.forward"):
                    if hpsn_compute_dtype is not None:
                        with torch.autocast(
                            device_type=accelerator.device.type, dtype=hpsn_compute_dtype
                        ):
                            outputs = model(hidden_states, attention_mask=attention_mask)
                    else:
                        outputs = model(hidden_states, attention_mask=attention_mask)
                with profiler.region("loss"):
                    # Loss runs outside autocast; _recon_loss casts to fp32 internally.
                    losses = loss_fn(outputs)
                    loss = losses["total"]

                with profiler.region("backward"):
                    accelerator.backward(loss)
                with profiler.region("optim"):
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
                        f"[{_timestamp()}] step {step:>7d} | lr {lr:.2e} | total {running['total']/n:.4f} "
                        f"| recon1 {running['recon1']/n:.4f} | recon2 {running['recon2']/n:.4f} "
                        f"| {n/max(dt,1e-6):.2f} it/s"
                    )
                    running = {"total": 0.0, "recon1": 0.0, "recon2": 0.0, "count": 0}
                    last_log_time = time.time()

                if config.profile and step >= config.profile_steps:
                    if accelerator.is_main_process:
                        accelerator.print(profiler.report(step))
                    done = True
                    break

                if step % config.save_steps == 0:
                    ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
                    accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        accelerator.print(f"[{_timestamp()}] [info] saved checkpoint to {ckpt_dir}")

                if step >= config.max_steps:
                    done = True
                    break

            if config.profile:
                data_iter_t0 = time.perf_counter()

    # Final save (skipped in profile mode — nothing trained).
    if not config.profile:
        final_dir = os.path.join(config.output_dir, "final")
        accelerator.save_state(final_dir)
        if accelerator.is_main_process:
            accelerator.print(f"[{_timestamp()}] [info] training complete. final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
