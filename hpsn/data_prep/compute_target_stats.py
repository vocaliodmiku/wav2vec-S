"""HPSN-v2 Phase 2A.5 — per-dimension target normalization stats.

GPT-2 hidden states have ~5 outlier dimensions whose magnitude is 100–1000×
the rest of the channels (Dettmers et al. 2022). Direct MSE on raw GPT-2
hidden states is dominated by those dims, which masks learning on the other
763 dims. The standard fix is per-dim z-score normalization computed on a
sample of the training data and applied at load time.

This script samples ``--n_sample`` utterances from the targets HDF5,
computes per-dim mean / std for the requested fields, and saves them to a
single ``.npz``. ``TargetsHDF5Dataset(stats_path=...)`` then applies them
online so the model trains on a clean target space and the HDF5 stays
untouched.

Stats computed per field
------------------------
* per-dim ``mean`` and ``std`` (population), excluding silence frames
  (``phone_id == 0``) so leading/trailing zero pads don't bias the moments.
* ``std`` is floored at 1e-3 to prevent divide-by-zero on degenerate dims.

Usage
-----
    python -m hpsn.data_prep.compute_target_stats \\
        --targets_h5 /scratch/.../targets_v2/targets_train-clean-100.h5 \\
        --out        /scratch/.../targets_v2/target_stats.npz \\
        --fields gpt2_l4,gpt2_l8,log_mel \\
        --n_sample 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--targets_h5", required=True)
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument(
        "--fields", default="gpt2_l4,gpt2_l8,log_mel",
        help="Comma-separated target field names to compute stats for. "
             "phone_id and word_id are integer fields and are skipped.",
    )
    p.add_argument(
        "--n_sample", type=int, default=1000,
        help="Number of utterances to sample (default: 1000; ~1.5M frames).",
    )
    p.add_argument(
        "--exclude_silence", action="store_true", default=True,
        help="Drop frames with phone_id==0 from the moment computation.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        print("ERROR: --fields is empty", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.targets_h5, "r") as h5:
        keys = list(h5.keys())
        n = min(args.n_sample, len(keys))
        idx = rng.choice(len(keys), size=n, replace=False)
        sample = [keys[i] for i in idx]
        print(f"HDF5: {args.targets_h5}  ({len(keys)} utts)")
        print(f"Sampling {n} utts for moment computation")
        print(f"Fields:  {fields}")
        if args.exclude_silence:
            print("Excluding frames with phone_id==0")

        # Streaming Welford-style accumulators (per-dim).
        # Init lazily once we see the first frame and know D for each field.
        sums: dict[str, np.ndarray] = {}
        sumsq: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}

        for utt_id in tqdm(sample, desc="stats", unit="utt"):
            grp = h5[utt_id]
            if args.exclude_silence:
                pid = grp["phone_id"][:]
                keep = pid != 0
                if not keep.any():
                    continue
            else:
                keep = None

            for fname in fields:
                if fname not in grp:
                    raise KeyError(f"{utt_id}: missing field {fname!r}")
                arr = grp[fname][:]
                if arr.dtype == np.float16:
                    arr = arr.astype(np.float32)
                if arr.ndim == 1:
                    # 1-D fields (phone_id, word_id) — skip; they are integer IDs
                    # and don't get z-scored.
                    continue
                if keep is not None:
                    arr = arr[keep]
                if fname not in sums:
                    sums[fname] = np.zeros(arr.shape[1], dtype=np.float64)
                    sumsq[fname] = np.zeros(arr.shape[1], dtype=np.float64)
                    counts[fname] = 0
                sums[fname] += arr.sum(axis=0)
                sumsq[fname] += (arr.astype(np.float64) ** 2).sum(axis=0)
                counts[fname] += arr.shape[0]

    out: dict[str, np.ndarray] = {}
    print()
    for fname in fields:
        if fname not in sums:
            print(f"  {fname}: skipped (not a 2-D field)")
            continue
        n_frames = counts[fname]
        mean = (sums[fname] / n_frames).astype(np.float32)
        var = (sumsq[fname] / n_frames - (sums[fname] / n_frames) ** 2)
        var = np.maximum(var, 0.0)  # numerical safety
        std = np.sqrt(var).astype(np.float32)
        std_clamped = np.maximum(std, 1e-3)
        out[f"{fname}_mean"] = mean
        out[f"{fname}_std"] = std_clamped

        # Diagnostics — show how many "outlier" dims live in this field.
        n_outlier = int((std > 5.0).sum())
        print(f"  {fname}:  n_frames={n_frames}  D={mean.size}")
        print(f"    mean  range [{mean.min():+.3f}, {mean.max():+.3f}]  |median| {np.median(np.abs(mean)):.3f}")
        print(f"    std   range [{std.min():+.3f}, {std.max():+.3f}]   |median| {np.median(std):.3f}")
        print(f"    outlier-dim count (std > 5): {n_outlier}/{mean.size}")

    if not out:
        print("ERROR: no 2-D fields had stats computed", file=sys.stderr)
        return 2

    np.savez(out_path, **out)
    print(f"\nWrote {out_path}  ({len(out) // 2} fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
