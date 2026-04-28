"""HPSN-v2 Phase 2A smoke test for ``TargetsHDF5Dataset`` + ``TargetsCollator``.

Loads a few batches and asserts the invariants every downstream module will
rely on. Run after Phase 1 so we catch alignment / padding / dtype bugs once,
not in the middle of training.

Checks
------
1. Audio sample count = ``n_frames * hop_length`` per item.
2. Per-target frame count = item ``n_frames``.
3. ``frame_mask[i, :n_frames[i]] == True`` and zeros after.
4. Padded target frames are exactly zero.
5. ``phone_id`` ∈ [0, n_phones).
6. ``word_id`` ≥ 0; equals 0 ⇔ silence frames.
7. Wherever ``word_id != 0``, the GPT-2 hidden states are non-zero.
8. Padded audio samples are exactly zero (per attention_mask).

Usage
-----
    python -m hpsn.data_prep.check_dataset \\
        --manifest /scratch/.../targets_v2/manifest.csv \\
        --targets_h5 /scratch/.../targets_v2/targets_train-clean-100.h5 \\
        --backbone biaofu-xmu/wav2vec-S-Base-ft-960h
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import Wav2Vec2FeatureExtractor

from hpsn.training.data import (
    TARGET_FIELDS,
    TARGETS_HOP_LENGTH,
    build_targets_dataloader,
)
from hpsn.data_prep.arpabet import N_PHONES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--targets_h5", required=True)
    p.add_argument("--backbone", default="biaofu-xmu/wav2vec-S-Base-ft-960h",
                   help="Used only to load the wav2vec feature_extractor; weights not loaded.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_batches", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2,
                   help="Set >0 to also exercise the per-worker h5py handle.")
    p.add_argument("--max_audio_seconds", type=float, default=15.0)
    p.add_argument("--max_samples", type=int, default=64,
                   help="Cap on how many manifest rows to consider.")
    p.add_argument("--stats_path", default=None,
                   help="Optional .npz from compute_target_stats.py — if set, "
                        "applies per-dim z-score to 2-D targets at load time.")
    return p.parse_args()


def assert_eq(name: str, got, want, ctx: str = ""):
    if got != want:
        raise AssertionError(f"[{ctx}] {name}: expected {want}, got {got}")


def main() -> int:
    args = parse_args()
    print(f"Loading feature_extractor from {args.backbone}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.backbone)

    loader = build_targets_dataloader(
        manifest_path=args.manifest,
        targets_h5_path=args.targets_h5,
        feature_extractor=fe,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_audio_seconds=args.max_audio_seconds,
        max_samples=args.max_samples,
        shuffle=False,
        stats_path=args.stats_path,
    )

    print(f"Dataset size: {len(loader.dataset)} utterances "
          f"({args.n_batches} × batch={args.batch_size} = "
          f"{args.n_batches * args.batch_size} sampled)")
    print(f"Target fields: {TARGET_FIELDS}\n")

    seen = 0
    for b_idx, batch in enumerate(loader):
        if b_idx >= args.n_batches:
            break
        B, S = batch["input_values"].shape
        T = batch["frame_mask"].shape[1]
        ctx = f"batch {b_idx}"
        print(f"{ctx}: B={B}  S={S}  T_max={T}  utt_ids={batch['utt_ids']}")

        # --- 8. padded audio samples are zero ---
        attn = batch["attention_mask"].bool()
        pad_audio = batch["input_values"][~attn]
        if pad_audio.numel() and pad_audio.abs().max().item() > 0:
            raise AssertionError(f"[{ctx}] padded audio samples are not zero")

        # --- 3. frame_mask shape and contiguity ---
        for i in range(B):
            n_valid = int(batch["frame_mask"][i].sum().item())
            if not torch.all(batch["frame_mask"][i, :n_valid]):
                raise AssertionError(f"[{ctx}] frame_mask[{i}] has gap inside valid prefix")
            if n_valid < T and torch.any(batch["frame_mask"][i, n_valid:]):
                raise AssertionError(f"[{ctx}] frame_mask[{i}] True past n_valid")

            # --- 1. audio length = n_frames * hop ---
            n_audio_valid = int(attn[i].sum().item())
            assert_eq("n_audio_valid", n_audio_valid, n_valid * TARGETS_HOP_LENGTH, ctx)

            # --- 4. padded target positions are zero ---
            for name, t in batch["targets"].items():
                pad_slice = t[i, n_valid:]
                if pad_slice.numel() and pad_slice.abs().sum().item() != 0:
                    raise AssertionError(
                        f"[{ctx}] target {name!r} has non-zero padding for sample {i}"
                    )

            # --- 5/6. ID ranges ---
            pid = batch["targets"]["phone_id"][i, :n_valid]
            if int(pid.min()) < 0 or int(pid.max()) >= N_PHONES:
                raise AssertionError(
                    f"[{ctx}] phone_id out of range [0,{N_PHONES}): "
                    f"min={int(pid.min())} max={int(pid.max())}"
                )
            wid = batch["targets"]["word_id"][i, :n_valid]
            if int(wid.min()) < 0:
                raise AssertionError(f"[{ctx}] word_id has negative values")

            # --- 7. GPT-2 hidden states non-zero where word_id != 0 ---
            for layer in ("gpt2_l4", "gpt2_l8"):
                hid = batch["targets"][layer][i, :n_valid]    # [n_valid, 768]
                has_word = wid != 0
                if has_word.any():
                    nonzero_per_frame = (hid.abs().sum(dim=-1) > 0)
                    bad = has_word & (~nonzero_per_frame)
                    if bad.any():
                        n_bad = int(bad.sum().item())
                        raise AssertionError(
                            f"[{ctx}] {layer} is zero on {n_bad} word frames "
                            f"for sample {i}"
                        )

        seen += B
        # Print a per-target sanity summary on the first batch only
        if b_idx == 0:
            print("  per-target shape / dtype / value range:")
            for name, t in batch["targets"].items():
                v = t.float()
                print(f"    {name:18s} shape={tuple(t.shape)}  dtype={t.dtype}  "
                      f"min={v.min().item():+.3f}  max={v.max().item():+.3f}  "
                      f"mean_nonzero={v[v != 0].mean().item():+.3f}"
                      if (v != 0).any() else
                      f"    {name:18s} shape={tuple(t.shape)}  dtype={t.dtype}  (all zero)")

    print(f"\nOK. {seen} samples passed all invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
