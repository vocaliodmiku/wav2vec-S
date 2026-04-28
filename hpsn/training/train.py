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
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup

from ..config import HPSNConfig
from ..model.backbone import FrozenWav2VecS
from ..model.hpsn import HPSN
from .data import build_dataloader, build_targets_dataloader
from .loss import HPSNLoss, HPSNV2Loss

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


def _parse_tuple_int(s: str) -> tuple[int, ...]:
    """Parse '1,2,3,4' or '1 2 3 4' into a tuple of ints."""
    parts = s.replace(",", " ").split()
    if not parts:
        raise argparse.ArgumentTypeError("tap layer list cannot be empty")
    return tuple(int(p) for p in parts)


def _format_tap_weights(w: list[float]) -> str:
    return "[" + ",".join(f"{x:.2f}" for x in w) + "]"


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
        default = getattr(defaults, f.name)
        if isinstance(default, tuple):
            p.add_argument(f"--{f.name}", type=_parse_tuple_int, default=default)
        elif isinstance(default, bool):
            p.add_argument(
                f"--{f.name}",
                type=lambda s: s.lower() in {"1", "true", "yes"},
                default=default,
            )
        else:
            t = f.type if not isinstance(f.type, str) else type(default)
            p.add_argument(f"--{f.name}", type=t, default=default)
    return p


def _validate_tap_layers(config: HPSNConfig, max_layer_idx: int) -> None:
    """Each tap index must be in [1, max_layer_idx]; bands must be disjoint."""
    bands = [
        ("level1", config.level1_tap_layers),
        ("level2", config.level2_tap_layers),
        ("level3", config.level3_tap_layers),
    ]
    seen: dict[int, str] = {}
    for name, taps in bands:
        if len(taps) == 0:
            raise ValueError(f"{name}_tap_layers is empty")
        for idx in taps:
            if not (1 <= idx <= max_layer_idx):
                raise ValueError(
                    f"{name}_tap_layers contains index {idx} outside "
                    f"[1, {max_layer_idx}] for backbone {config.backbone_model}"
                )
            if idx in seen:
                raise ValueError(
                    f"{name}_tap_layers index {idx} overlaps with {seen[idx]}_tap_layers"
                )
            seen[idx] = name


@torch.no_grad()
def _compute_monitors(unwrapped_model: HPSN, outputs: dict) -> dict:
    """Per-log-window monitors computed from the most recent batch.

    Returns a dict containing:
      tap1_w, tap2_w, tap3_w  — list[float], softmax-normalized tap weights
      inhibition_alpha        — float, learned inhibition scale at L2
      recon_cos               — dict[int, float], cos(recon_i, target_i) on masked positions
    """
    tap1_w = F.softmax(unwrapped_model.tap1.weights, dim=0).cpu().tolist()
    tap2_w = F.softmax(unwrapped_model.tap2.weights, dim=0).cpu().tolist()
    tap3_w = F.softmax(unwrapped_model.tap3.weights, dim=0).cpu().tolist()

    # Inhibition gates: L1 (phoneme cohort) and L2 (lexical cohort).
    cb1 = unwrapped_model.level1.inhib_gate
    cb2 = unwrapped_model.level2.inhib_gate
    alpha1 = float(cb1.alpha)
    alpha2 = float(cb2.alpha)

    # Codebook utilization on the most recent batch — number of distinct prototypes
    # selected by the top-k cohort. If only a handful are ever used, cohort
    # competition has collapsed.
    n_codes_total_1 = cb1.codebook.num_embeddings
    n_codes_total_2 = cb2.codebook.num_embeddings
    last_idx_1 = getattr(cb1, "last_topk_idx", None)
    last_idx_2 = getattr(cb2, "last_topk_idx", None)
    n_codes_used_1 = int(last_idx_1.unique().numel()) if last_idx_1 is not None else 0
    n_codes_used_2 = int(last_idx_2.unique().numel()) if last_idx_2 is not None else 0

    recon_cos: dict[int, float] = {}
    for i in (1, 2, 3):
        mask = outputs[f"mask{i}"]
        recon = outputs.get(f"recon{i}")
        target = outputs.get(f"target{i}")
        # In v2 mode the v1 tap-recon heads aren't built (recon=None) and the
        # cos(recon, tap) monitor is meaningless — log NaN so per-target losses
        # in `running` remain the source of truth.
        if recon is None or target is None or mask.sum() == 0:
            recon_cos[i] = float("nan")
            continue
        r = recon[mask].float()
        t = target[mask].float()
        recon_cos[i] = F.cosine_similarity(r, t, dim=-1).mean().item()

    # Top-down prediction cosine: does the higher level's prediction μ_i point
    # in the same direction as the lower level's representation? Near zero ⇒
    # cross-attention pathway is being ignored (top-down doing no real work).
    td_cos: dict[int, float] = {}
    for i, repr_key in ((1, "level1_repr"), (2, "level2_repr")):
        mu = outputs[f"mu{i}"].float()
        rep = outputs[repr_key].float()
        td_cos[i] = F.cosine_similarity(mu, rep, dim=-1).mean().item()

    return {
        "tap1_w": tap1_w,
        "tap2_w": tap2_w,
        "tap3_w": tap3_w,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "n_codes_used_1": n_codes_used_1,
        "n_codes_total_1": n_codes_total_1,
        "n_codes_used_2": n_codes_used_2,
        "n_codes_total_2": n_codes_total_2,
        "recon_cos": recon_cos,
        "td_cos": td_cos,
    }


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
                f"[{_timestamp()}] [info] hidden_dim {config.hidden_dim} != backbone "
                f"{backbone.hidden_size}; using backbone size."
            )
        config.hidden_dim = backbone.hidden_size

    _validate_tap_layers(config, backbone.num_hidden_layers)
    if accelerator.is_main_process:
        accelerator.print(
            f"[{_timestamp()}] [info] tap bands "
            f"L1={config.level1_tap_layers} L2={config.level2_tap_layers} "
            f"L3={config.level3_tap_layers} (backbone depth {backbone.num_hidden_layers})"
        )

    model = HPSN(config)

    # Frozen-tap invariant: if config requested a one-hot tap, verify the
    # constructed LayerTap honored it. Catches the v2-era bug where
    # ``level{N}_frozen_tap`` was added to the dataclass but the run config
    # left it at -1, silently keeping the learnable softmax tap and letting
    # tap weights drift to the worst-in-band layer per §17 of the
    # construction doc.
    def _assert_frozen_tap(tap_module, layers, frozen_idx, name):
        if frozen_idx is None or int(frozen_idx) < 0:
            return
        from torch import nn as _nn
        if isinstance(tap_module.weights, _nn.Parameter):
            raise RuntimeError(
                f"{name}: config asked for frozen_layer={frozen_idx} but "
                f"tap.weights is still an nn.Parameter (learnable). "
                f"LayerTap was not constructed with the frozen flag."
            )
        layers_t = tuple(layers)
        try:
            pos = layers_t.index(int(frozen_idx))
        except ValueError:
            raise RuntimeError(
                f"{name}: frozen_layer={frozen_idx} not in layers={layers_t}"
            )
        w = tap_module.weights.detach().cpu()
        expected = torch.zeros(len(layers_t))
        expected[pos] = 1.0
        if not torch.allclose(w, expected, atol=1e-6):
            raise RuntimeError(
                f"{name}: tap.weights={w.tolist()} != one-hot at layer "
                f"{frozen_idx} (idx {pos}) over layers {layers_t}"
            )

    _assert_frozen_tap(model.tap1, config.level1_tap_layers, config.level1_frozen_tap, "tap1")
    _assert_frozen_tap(model.tap2, config.level2_tap_layers, config.level2_frozen_tap, "tap2")
    _assert_frozen_tap(model.tap3, config.level3_tap_layers, config.level3_frozen_tap, "tap3")
    if accelerator.is_main_process:
        accelerator.print(
            f"[{_timestamp()}] [info] frozen taps: "
            f"L1={config.level1_frozen_tap if config.level1_frozen_tap >= 0 else 'learnable'}, "
            f"L2={config.level2_frozen_tap if config.level2_frozen_tap >= 0 else 'learnable'}, "
            f"L3={config.level3_frozen_tap if config.level3_frozen_tap >= 0 else 'learnable'}"
        )

    if config.use_v2_loss:
        loss_fn = HPSNV2Loss(
            lambda_log_mel=config.lambda_log_mel,
            lambda_phonol=config.lambda_phonol,
            lambda_phone_id=config.lambda_phone_id,
            lambda_gpt2_l4=config.lambda_gpt2_l4,
            lambda_gpt2_l8=config.lambda_gpt2_l8,
            lambda_td=config.lambda_td,
            lambda_restore=config.lambda_restore,
        )
        if accelerator.is_main_process:
            accelerator.print(
                f"[{_timestamp()}] [info] using HPSN-v2 loss "
                f"(λ log_mel={config.lambda_log_mel}, phonol={config.lambda_phonol}, "
                f"phone_id={config.lambda_phone_id}, gpt2_l4={config.lambda_gpt2_l4}, "
                f"gpt2_l8={config.lambda_gpt2_l8}, td={config.lambda_td}, "
                f"restore={config.lambda_restore} @ p={config.restore_prob})"
            )
    else:
        loss_fn = HPSNLoss(
            config.lambda1,
            config.lambda2,
            config.lambda3,
            config.lambda_td,
            config.loss_type,
        )

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

    if config.use_v2_loss:
        if not config.targets_manifest or not config.targets_h5:
            raise SystemExit(
                "use_v2_loss=True requires --targets_manifest and --targets_h5"
            )
        dataloader = build_targets_dataloader(
            manifest_path=config.targets_manifest,
            targets_h5_path=config.targets_h5,
            feature_extractor=feature_extractor,
            batch_size=config.per_device_train_batch_size,
            num_workers=config.num_workers,
            max_audio_seconds=config.max_audio_seconds,
            max_samples=config.max_train_samples,
            shuffle=True,
            stats_path=config.target_stats or None,
            restore_prob=config.restore_prob,
        )
    else:
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
    if config.use_v2_loss:
        running = {
            "total": 0.0,
            "log_mel": 0.0, "phonol": 0.0, "phone_id": 0.0,
            "gpt2_l4": 0.0, "gpt2_l8": 0.0,
            "td_2to1": 0.0, "td_3to2": 0.0,
            "restore": 0.0,
            "count": 0,
        }
    else:
        running = {
            "total": 0.0,
            "recon1": 0.0, "recon2": 0.0, "recon3": 0.0,
            "td_2to1": 0.0, "td_3to2": 0.0,
            "count": 0,
        }
    last_grad_norm = float("nan")  # pre-clip global grad norm of the most recent optimizer step
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

                # v2-only inputs (None in v1)
                phone_id = batch.get("targets", {}).get("phone_id") if config.use_v2_loss else None
                word_id = batch.get("targets", {}).get("word_id") if config.use_v2_loss else None
                frame_mask = batch.get("frame_mask") if config.use_v2_loss else None

                with profiler.region("hpsn.forward"):
                    if hpsn_compute_dtype is not None:
                        with torch.autocast(
                            device_type=accelerator.device.type, dtype=hpsn_compute_dtype
                        ):
                            outputs = model(
                                hidden_states, attention_mask=attention_mask,
                                phone_id=phone_id, word_id=word_id,
                            )
                    else:
                        outputs = model(
                            hidden_states, attention_mask=attention_mask,
                            phone_id=phone_id, word_id=word_id,
                        )
                with profiler.region("loss"):
                    # Loss runs outside autocast; recon casts to fp32 internally.
                    if config.use_v2_loss:
                        restore_mask = batch.get("restore_mask")
                        losses = loss_fn(
                            outputs, batch["targets"], frame_mask,
                            restore_mask=restore_mask,
                        )
                    else:
                        losses = loss_fn(outputs)
                    loss = losses["total"]

                with profiler.region("backward"):
                    accelerator.backward(loss)
                with profiler.region("optim"):
                    if accelerator.sync_gradients:
                        gn = accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
                        # gn is a tensor under accelerate; cast safely to float for logging.
                        last_grad_norm = float(gn) if gn is not None else float("nan")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Accumulate for logging.
            running["total"] += loss.detach().float().item()
            if config.use_v2_loss:
                for k in ("log_mel", "phonol", "phone_id", "gpt2_l4", "gpt2_l8",
                          "td_2to1", "td_3to2", "restore"):
                    running[k] += losses[k].float().item()
            else:
                for k in ("recon1", "recon2", "recon3", "td_2to1", "td_3to2"):
                    running[k] += losses[k].float().item()
            running["count"] += 1

            if accelerator.sync_gradients:
                step += 1

                if step % config.logging_steps == 0 and accelerator.is_main_process:
                    n = max(running["count"], 1)
                    dt = time.time() - last_log_time
                    lr = scheduler.get_last_lr()[0]
                    mon = _compute_monitors(accelerator.unwrap_model(model), outputs)
                    rc = mon["recon_cos"]
                    tdc = mon["td_cos"]
                    if config.use_v2_loss:
                        accelerator.print(
                            f"[{_timestamp()}] step {step:>7d} | lr {lr:.2e} | total {running['total']/n:.4f} "
                            f"| logmel {running['log_mel']/n:.4f} phonol {running['phonol']/n:.4f} "
                            f"phone {running['phone_id']/n:.4f} gpt2l4 {running['gpt2_l4']/n:.4f} "
                            f"gpt2l8 {running['gpt2_l8']/n:.4f} restore {running['restore']/n:.4f} "
                            f"| td2to1/3to2 {running['td_2to1']/n:.3f}/{running['td_3to2']/n:.3f} "
                            f"| gnorm {last_grad_norm:.2f} | {n/max(dt,1e-6):.2f} it/s"
                        )
                    else:
                        accelerator.print(
                            f"[{_timestamp()}] step {step:>7d} | lr {lr:.2e} | total {running['total']/n:.4f} "
                            f"| recon1/2/3 {running['recon1']/n:.4f}/{running['recon2']/n:.4f}/{running['recon3']/n:.4f} "
                            f"| td2to1/3to2 {running['td_2to1']/n:.3f}/{running['td_3to2']/n:.3f} "
                            f"| gnorm {last_grad_norm:.2f} | {n/max(dt,1e-6):.2f} it/s"
                        )
                    accelerator.print(
                        f"[{_timestamp()}]         "
                        f"| alpha1/2 {mon['alpha1']:+.3f}/{mon['alpha2']:+.3f} "
                        f"| codes1 {mon['n_codes_used_1']:>3d}/{mon['n_codes_total_1']} "
                        f"codes2 {mon['n_codes_used_2']:>3d}/{mon['n_codes_total_2']} "
                        f"| rcos1/2/3 {rc[1]:+.3f}/{rc[2]:+.3f}/{rc[3]:+.3f} "
                        f"| tdcos1/2 {tdc[1]:+.3f}/{tdc[2]:+.3f} "
                        f"| tap1 {_format_tap_weights(mon['tap1_w'])} "
                        f"tap2 {_format_tap_weights(mon['tap2_w'])} "
                        f"tap3 {_format_tap_weights(mon['tap3_w'])}"
                    )
                    if config.use_v2_loss:
                        running = {
                            "total": 0.0,
                            "log_mel": 0.0, "phonol": 0.0, "phone_id": 0.0,
                            "gpt2_l4": 0.0, "gpt2_l8": 0.0,
                            "td_2to1": 0.0, "td_3to2": 0.0,
                            "restore": 0.0,
                            "count": 0,
                        }
                    else:
                        running = {
                            "total": 0.0,
                            "recon1": 0.0, "recon2": 0.0, "recon3": 0.0,
                            "td_2to1": 0.0, "td_3to2": 0.0,
                            "count": 0,
                        }
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
