"""HPSN-v2 end-to-end smoke test (Phase 2D + 2E + 2F).

Runs ONE real batch through the full v2 pipeline:

    LibriSpeech audio + targets HDF5
        → TargetsCollator
        → wav2vec-S backbone (frozen)
        → HPSN with use_span_masking=True, n_iterations=2, use_v2_loss=True
        → HPSNV2Loss
        → backward()

Asserts the loss is finite, the v2 head outputs have the expected shapes,
the span maskers actually mask whole-phoneme/whole-word spans, and gradients
flow into the new ``error_proj_*``, ``v2_heads.*`` and the level-2 phone
classifier.

Usage
-----
    python -m hpsn.data_prep.check_v2_e2e \\
        --backbone biaofu-xmu/wav2vec-S-Base-ft-960h \\
        --manifest /scratch/.../targets_v2/manifest.csv \\
        --targets_h5 /scratch/.../targets_v2/targets_train-clean-100.h5 \\
        --stats_path /scratch/.../targets_v2/target_stats.npz
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import Wav2Vec2FeatureExtractor

from hpsn.config import HPSNConfig
from hpsn.model.backbone import FrozenWav2VecS
from hpsn.model.hpsn import HPSN
from hpsn.training.data import build_targets_dataloader
from hpsn.training.loss import HPSNV2Loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backbone", default="biaofu-xmu/wav2vec-S-Base-ft-960h")
    p.add_argument("--manifest", required=True)
    p.add_argument("--targets_h5", required=True)
    p.add_argument("--stats_path", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_audio_seconds", type=float, default=10.0)
    p.add_argument("--max_samples", type=int, default=8)
    return p.parse_args()


def assert_true(c, m):
    if not c:
        raise AssertionError(m)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"backbone: {args.backbone}  device: {device}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.backbone)
    backbone = FrozenWav2VecS(args.backbone, dtype=torch.float32).to(device)

    loader = build_targets_dataloader(
        manifest_path=args.manifest,
        targets_h5_path=args.targets_h5,
        feature_extractor=fe,
        batch_size=args.batch_size,
        num_workers=0,
        max_audio_seconds=args.max_audio_seconds,
        max_samples=args.max_samples,
        shuffle=False,
        stats_path=args.stats_path,
    )
    batch = next(iter(loader))
    iv = batch["input_values"].to(device)
    am = batch["attention_mask"].to(device)
    fm = batch["frame_mask"].to(device)
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    B, S = iv.shape
    T = fm.shape[1]
    print(f"batch: B={B} S={S} T={T}  utts={batch['utt_ids']}")

    cfg = HPSNConfig(
        n_iterations=2,
        use_span_masking=True,
        use_v2_loss=True,
        # Use v1 mask-prob defaults; that's the fraction of phoneme/word spans masked.
        level1_mask_prob=0.25,
        level2_mask_prob=0.20,
        level3_mask_prob=0.15,
    )
    model = HPSN(cfg).to(device).train()
    loss_fn = HPSNV2Loss(
        cfg.lambda_log_mel, cfg.lambda_phonol, cfg.lambda_phone_id,
        cfg.lambda_gpt2_l4, cfg.lambda_gpt2_l8, cfg.lambda_td,
    )

    with torch.no_grad():
        hs = backbone(iv, attention_mask=am)
    print(f"backbone returned {len(hs)} hidden states")

    out = model(
        hs, attention_mask=am,
        phone_id=targets["phone_id"], word_id=targets["word_id"],
    )

    # ── shape sanity on v2 head outputs (T_backbone, not collator T) ────────
    rv2 = out["recon_v2"]
    T_b = rv2["l1_log_mel"].shape[1]
    expected = {
        "l1_log_mel": (B, T_b, cfg.n_log_mel),
        "l1_phonol": (B, T_b, cfg.n_phonol_features),
        "l2_phone_logits": (B, T_b, cfg.n_phones),
        "l2_gpt2": (B, T_b, cfg.gpt2_hidden_dim),
        "l3_gpt2": (B, T_b, cfg.gpt2_hidden_dim),
    }
    for k, want in expected.items():
        got = tuple(rv2[k].shape)
        assert_true(got == want, f"recon_v2[{k!r}] shape={got}, expected {want}")
    print(f"OK  v2 head shapes (T_backbone={T_b}): "
          + ", ".join(f"{k}={tuple(v.shape)}" for k, v in rv2.items()))

    # ── span-mask invariant: each phoneme/word run is all-or-nothing masked ─
    # (silence runs, id == 0, must never be masked).
    from hpsn.model.hpsn import resample_ids

    def _check_atomic_runs(mask_row, ids_row, kind: str) -> tuple[int, int, int]:
        n_runs_masked = 0
        n_partial = 0
        n_silence_masked = 0
        i, T = 0, len(ids_row)
        while i < T:
            j = i
            while j < T and int(ids_row[j]) == int(ids_row[i]):
                j += 1
            run_id = int(ids_row[i])
            sub = mask_row[i:j]
            any_, all_ = bool(sub.any()), bool(sub.all())
            if run_id == 0:
                if any_:
                    n_silence_masked += 1
            else:
                if any_:
                    n_runs_masked += 1
                if any_ and not all_:
                    n_partial += 1
            i = j
        return n_runs_masked, n_partial, n_silence_masked

    T_b = out["mask1"].shape[1]
    pid_b = resample_ids(targets["phone_id"], T_b)
    wid_b = resample_ids(targets["word_id"], T_b)
    print(f"  T_target={targets['phone_id'].shape[1]}  T_backbone={T_b}")

    m1_np = out["mask1"][0].cpu().numpy()
    n_masked, n_partial, n_sil = _check_atomic_runs(m1_np, pid_b[0].cpu().numpy(), "phoneme")
    assert_true(n_partial == 0, f"PhonemeSpanMasker partially masked {n_partial} phoneme runs")
    assert_true(n_sil == 0, f"PhonemeSpanMasker masked {n_sil} silence runs")
    print(f"OK  PhonemeSpanMasker (sample 0): {n_masked} whole phoneme runs masked, "
          f"0 partial, 0 silence; total {int(m1_np.sum())}/{T_b} frames")

    m2_np = out["mask2"][0].cpu().numpy()
    n_masked, n_partial, n_sil = _check_atomic_runs(m2_np, wid_b[0].cpu().numpy(), "word")
    assert_true(n_partial == 0, f"WordSpanMasker partially masked {n_partial} word runs")
    assert_true(n_sil == 0, f"WordSpanMasker masked {n_sil} silence runs")
    print(f"OK  WordSpanMasker (sample 0): {n_masked} whole word runs masked, "
          f"0 partial, 0 silence; total {int(m2_np.sum())}/{T_b} frames")

    # ── loss: finite + backprop ──────────────────────────────────────────────
    losses = loss_fn(out, targets, fm)
    print(
        "  loss components:  "
        + "  ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
    )
    total = losses["total"]
    assert_true(torch.isfinite(total), f"v2 total loss is not finite: {float(total)}")

    total.backward()

    # Check gradient flow into the new modules.
    grads = {
        "v2_heads.l1_log_mel.weight": model.v2_heads.l1_log_mel.weight.grad,
        "v2_heads.l2_phone_logits.weight": model.v2_heads.l2_phone_logits.weight.grad,
        "v2_heads.l3_gpt2.weight": model.v2_heads.l3_gpt2.weight.grad,
        "error_proj_2.weight": model.error_proj_2.weight.grad,
    }
    for name, g in grads.items():
        assert_true(g is not None, f"no grad on {name}")
        assert_true(torch.isfinite(g).all(), f"non-finite grad on {name}")
        assert_true(g.abs().sum().item() > 0, f"zero grad on {name} (training pressure missing)")
    print("OK  gradients flow into v2_heads + error_proj_2")

    print("\nAll Phase 2 (v2) end-to-end checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
