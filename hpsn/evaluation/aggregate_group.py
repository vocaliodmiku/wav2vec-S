"""Group-level aggregation and statistics for HPSN × MEG-MASC ridge results.

Reads per-subject pkls written by ``proc_5_hpsn_ridge.py`` and produces:

* per-subject mean-r table for two conditions (``cond_a`` vs ``cond_b``);
* Wilcoxon signed-rank test (``cond_a > cond_b``);
* optional per-lag Δr plot if both conditions were run with ``--per_lag``.

Usage
-----
    python -m hpsn.evaluation.aggregate_group \
        --results_dir /scratch/.../meg_results \
        --cond_a hpsn_l1 --cond_b baseline_low \
        --space sensor --resample_opt MEG --ses 0 \
        --subjects 01,02,03,04

    # Per-lag curve
    python -m hpsn.evaluation.aggregate_group \
        --results_dir /scratch/.../meg_results \
        --cond_a hpsn_l1 --cond_b baseline_low \
        --space sensor --resample_opt MEG --ses 0 \
        --subjects 01,02,03,04 --per_lag --plot_out delta_curve.png
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--cond_a", required=True, help="Condition expected to be higher.")
    p.add_argument("--cond_b", required=True, help="Comparison baseline.")
    p.add_argument("--space", choices=("sensor", "roi"), default="sensor")
    p.add_argument("--resample_opt", choices=("MEG", "REPR"), default="MEG")
    p.add_argument("--ses", type=int, required=True)
    p.add_argument("--subjects", required=True,
                   help="Comma-separated list, e.g. 01,02,03.")
    p.add_argument("--per_lag", action="store_true")
    p.add_argument("--plot_out", default=None,
                   help="If set with --per_lag, save Δr-vs-lag plot.")
    return p.parse_args()


def _result_path(results_dir: Path, subj: str, ses: int, feat: str,
                 space: str, resample_opt: str) -> Path:
    return (
        results_dir / f"trf_ridge_sub-{subj}" /
        f"trf_ridge_{feat}_{space}_{resample_opt}_sub-{subj}_ses-{ses}.pkl"
    )


def load_subject(results_dir, subj, ses, feat, space, resample_opt):
    path = _result_path(results_dir, subj, ses, feat, space, resample_opt)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    if args.per_lag:
        aggregate_per_lag(args, results_dir, subjects)
    else:
        aggregate_full_lags(args, results_dir, subjects)


def aggregate_full_lags(args, results_dir, subjects):
    a_scores, b_scores, used = [], [], []
    for subj in subjects:
        ra = load_subject(results_dir, subj, args.ses, args.cond_a,
                          args.space, args.resample_opt)
        rb = load_subject(results_dir, subj, args.ses, args.cond_b,
                          args.space, args.resample_opt)
        if ra is None or rb is None:
            print(f"[skip sub-{subj}]  cond_a found: {ra is not None}, "
                  f"cond_b found: {rb is not None}")
            continue
        a = float(np.nanmean(ra["r"]))
        b = float(np.nanmean(rb["r"]))
        a_scores.append(a)
        b_scores.append(b)
        used.append(subj)
        print(f"  sub-{subj}: {args.cond_a}={a:.4f}  {args.cond_b}={b:.4f}  Δ={a-b:+.4f}")

    if len(used) < 2:
        print("Need ≥2 subjects with both conditions to run Wilcoxon.", file=sys.stderr)
        sys.exit(1)

    a_arr = np.array(a_scores)
    b_arr = np.array(b_scores)
    delta = a_arr - b_arr
    stat, p = wilcoxon(a_arr, b_arr, alternative="greater")

    print("\n=== Group-level summary ===")
    print(f"  subjects        : {len(used)}  ({','.join(used)})")
    print(f"  mean r {args.cond_a:>15s} : {a_arr.mean():.4f}")
    print(f"  mean r {args.cond_b:>15s} : {b_arr.mean():.4f}")
    print(f"  median Δr       : {np.median(delta):+.4f}")
    print(f"  mean   Δr       : {delta.mean():+.4f}")
    print(f"  Wilcoxon W      : {stat:.2f}")
    print(f"  p (greater)     : {p:.4g}")


def aggregate_per_lag(args, results_dir, subjects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_delta = []
    lag_values = None
    used = []
    for subj in subjects:
        ra = load_subject(results_dir, subj, args.ses, args.cond_a,
                          args.space, args.resample_opt)
        rb = load_subject(results_dir, subj, args.ses, args.cond_b,
                          args.space, args.resample_opt)
        if ra is None or rb is None or ra.get("mode") != "per_lag":
            print(f"[skip sub-{subj}]")
            continue
        if lag_values is None:
            lag_values = ra["lag_values_ms"]
        # mean across sensors per lag
        da = np.nanmean(ra["r_by_lag"], axis=1)
        db = np.nanmean(rb["r_by_lag"], axis=1)
        all_delta.append(da - db)
        used.append(subj)

    if not all_delta:
        print("No per-lag runs found.", file=sys.stderr)
        sys.exit(1)

    delta = np.stack(all_delta, axis=0)                   # [n_subj, n_lags]
    mean = delta.mean(0)
    sem = delta.std(0, ddof=1) / np.sqrt(delta.shape[0])

    print(f"subjects: {len(used)}  ({','.join(used)})")
    for i, lag in enumerate(lag_values):
        stat, p = wilcoxon(delta[:, i], alternative="greater") if delta.shape[0] > 1 \
            else (np.nan, np.nan)
        print(f"  lag {lag:+6.1f} ms : Δr = {mean[i]:+.4f} ± {sem[i]:.4f}  "
              f"(W={stat:.2f}, p={p:.3g})")

    if args.plot_out:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(0, lw=0.8, color="gray")
        ax.errorbar(lag_values, mean, yerr=sem, marker="o")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel(f"Δr ({args.cond_a} − {args.cond_b})")
        ax.set_title(f"n={len(used)}  {args.space}/{args.resample_opt}")
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=150)
        print(f"Saved plot → {args.plot_out}")


if __name__ == "__main__":
    main()
