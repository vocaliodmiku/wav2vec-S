"""ROI-restricted HPSN preflight report.

HPSN preflight pkls (from ``meg_hpsn.py``) carry a per-condition ``r``
vector but do NOT embed channel positions, so this script borrows them
from a baseline pkl produced by ``baseline_meg_masc.py``. Both pipelines
operate on the same MEG-MASC sensor layout, so any baseline pkl with
``ch_positions`` is a valid source.

Outputs (under ``--out_dir``, defaults to ``<results_dir>/hpsn_roi_report``):
  - ``per_subject_roi_summary.csv`` — one row per (subject, condition,
    ROI) for E1 / E2 / E3 and Δ12 / Δ23 / Δ13.
  - ``figs/violin_roi_{stat}.png`` — per-stat figure with one subplot per
    ROI.
  - ``report.md`` — table + figure pointers.

Usage
-----
    python -m hpsn.evaluation.roi_hpsn \\
        --results_dir /path/to/hpsn/meg_results

    python -m hpsn.evaluation.roi_hpsn \\
        --results_dir ... \\
        --baseline_results_dir /path/to/meg_results_defossez \\
        --rois frontal,temporal,parietal
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline_meg_masc import RESULTS_DIR as BASELINE_RESULTS_DIR
from .hpsn_ridge import RESULTS_DIR
from .report_baseline_meg_masc import fisher_z_mean
from .report_hpsn import (
    DELTA_CONDS,
    DELTA_PAIRS,
    LEVEL_CONDS,
    PKL_RE,
    STATS,
)
from .sensor_rois import ALL_ROIS, assign_rois, summarize_in_roi


def find_baseline_positions(baseline_results_dir: Path) -> np.ndarray | None:
    """Return ``ch_positions`` from the first baseline pkl that has them."""
    for p in baseline_results_dir.glob("trf_ridge_sub-*/trf_ridge_*.pkl"):
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        pos = d.get("ch_positions")
        if pos is not None:
            return np.asarray(pos, dtype=np.float64)
    return None


def discover_hpsn_pkls(results_dir: Path) -> list[dict]:
    out = []
    for pkl in results_dir.glob("trf_ridge_sub-*/preflight_hpsn_*.pkl"):
        m = PKL_RE.match(pkl.name)
        if not m:
            continue
        out.append(dict(
            path=pkl, subj=m.group("subj"), ses=int(m.group("ses")),
            space=m.group("space"), resample=m.group("resample"),
        ))
    return out


def _violin_grid(
    df: pd.DataFrame, conds_order: list[str], rois: list[str],
    stat: str, out_path: Path, title: str,
) -> None:
    n_rois = len(rois)
    ncols = min(3, n_rois)
    nrows = int(np.ceil(n_rois / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows + 0.5),
        squeeze=False, sharey=True,
    )
    axes_flat = axes.flatten()
    rng = np.random.default_rng(0)
    for i, roi in enumerate(rois):
        ax = axes_flat[i]
        sub = df[df["roi"] == roi]
        data, labels = [], []
        for c in conds_order:
            v = sub[sub["condition"] == c][stat].dropna().to_numpy()
            if v.size == 0:
                continue
            data.append(v)
            labels.append(c)
        if not data:
            ax.set_visible(False)
            continue
        ax.violinplot(data, showmeans=True, showmedians=True)
        for j, d in enumerate(data, start=1):
            jitter = rng.uniform(-0.05, 0.05, size=len(d))
            ax.scatter(np.full(len(d), j) + jitter, d, s=10, alpha=0.5,
                       color="black", zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_title(roi)
        if i % ncols == 0:
            ax.set_ylabel(stat)
    for k in range(n_rois, nrows * ncols):
        axes_flat[k].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="ROI-restricted HPSN preflight report.")
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--out_dir", default=None,
        help="Default: <results_dir>/hpsn_roi_report.",
    )
    p.add_argument("--space", choices=("sensor", "roi"), default="sensor")
    p.add_argument("--resample_opt", choices=("MEG", "REPR"), default=None)
    p.add_argument(
        "--baseline_results_dir", default=str(BASELINE_RESULTS_DIR),
        help="Directory containing baseline pkls with ch_positions; used to "
             "look up the MEG sensor positions HPSN pkls do not embed.",
    )
    p.add_argument(
        "--rois", default=",".join(ALL_ROIS),
        help=f"Comma-separated ROIs. Choices: {ALL_ROIS}.",
    )
    args = p.parse_args()

    rois = [r.strip() for r in args.rois.split(",") if r.strip()]
    bad = [r for r in rois if r not in ALL_ROIS]
    if bad:
        raise SystemExit(f"Unknown ROIs: {bad}; choices: {ALL_ROIS}")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "hpsn_roi_report"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    pos = find_baseline_positions(Path(args.baseline_results_dir))
    if pos is None:
        raise SystemExit(
            f"No ch_positions found under --baseline_results_dir "
            f"{args.baseline_results_dir!r}. Pass a directory containing "
            f"baseline pkls produced by the current baseline_meg_masc.py."
        )
    masks = assign_rois(pos)
    print(
        "ROI sensor counts: "
        + ", ".join(f"{r}={int(masks[r].sum())}" for r in rois)
    )

    entries = discover_hpsn_pkls(results_dir)
    entries = [e for e in entries if e["space"] == args.space]
    if args.resample_opt:
        entries = [e for e in entries if e["resample"] == args.resample_opt]
    if not entries:
        raise SystemExit(f"No HPSN pkls under {results_dir}")
    print(f"Discovered {len(entries)} HPSN pkls")

    subj_sess_r: dict = {ek: {} for ek in LEVEL_CONDS}
    skipped_per_lag = skipped_no_r = 0
    for e in entries:
        with open(e["path"], "rb") as f:
            d = pickle.load(f)
        conds = d.get("conditions", {})
        for ek in LEVEL_CONDS:
            cd = conds.get(ek)
            if cd is None:
                continue
            if cd.get("mode") != "full_lags":
                skipped_per_lag += 1
                continue
            r = cd.get("r")
            if r is None:
                skipped_no_r += 1
                continue
            subj_sess_r[ek].setdefault(e["subj"], []).append(
                np.asarray(r, dtype=np.float64)
            )
    if skipped_per_lag:
        print(f"(skipped {skipped_per_lag} per-lag conditions)")
    if skipped_no_r:
        print(f"(skipped {skipped_no_r} conditions with no 'r')")

    subj_r: dict = {ek: {} for ek in LEVEL_CONDS}
    for ek in LEVEL_CONDS:
        for subj, rs in subj_sess_r[ek].items():
            stack = np.stack(rs, axis=0)
            subj_r[ek][subj] = fisher_z_mean(stack, axis=0)

    rows = []
    for ek in LEVEL_CONDS:
        for subj, r in subj_r[ek].items():
            if r.size != pos.shape[0]:
                print(
                    f"WARN: sub-{subj} {ek} has {r.size} sensors but baseline "
                    f"positions have {pos.shape[0]}; skipping."
                )
                continue
            for roi in rois:
                s = summarize_in_roi(r, masks[roi])
                rows.append(dict(
                    subj=subj, condition=ek, roi=roi,
                    n_sensors=s["n_sensors"],
                    mean_r=s["mean_r"],
                    top20_mean=s["top20_mean"],
                    top10pct_mean=s["top10pct_mean"],
                ))

    for delta in DELTA_CONDS:
        a, b = DELTA_PAIRS[delta]
        common = sorted(set(subj_r[a]) & set(subj_r[b]))
        for subj in common:
            ra, rb = subj_r[a][subj], subj_r[b][subj]
            if ra.shape != rb.shape or ra.size != pos.shape[0]:
                continue
            d = rb - ra
            for roi in rois:
                s = summarize_in_roi(d, masks[roi])
                rows.append(dict(
                    subj=subj, condition=delta, roi=roi,
                    n_sensors=s["n_sensors"],
                    mean_r=s["mean_r"],
                    top20_mean=s["top20_mean"],
                    top10pct_mean=s["top10pct_mean"],
                ))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_subject_roi_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    conds_order = list(LEVEL_CONDS) + list(DELTA_CONDS)
    violin_paths: dict = {}
    for stat in STATS:
        pth = figs_dir / f"violin_roi_{stat}.png"
        _violin_grid(df, conds_order, rois, stat, pth,
                     title=f"{stat} per ROI (HPSN E1/E2/E3 + Δ)")
        violin_paths[stat] = pth
        print(f"Wrote {pth}")

    n_subj = df["subj"].nunique()
    lines = [
        "# HPSN preflight ROI report",
        "",
        f"- results dir: `{results_dir}`",
        f"- subjects: **{n_subj}**",
        f"- ROIs: {', '.join(rois)}",
        f"- channel positions sourced from: `{args.baseline_results_dir}`",
        "- aggregation: Fisher-z mean across sessions; per-subject within-ROI stats",
        "",
        "## Per-condition × ROI summary (mean ± s.d. across subjects)",
        "",
        "| Condition | ROI | n_sensors | mean_r | top20_mean | top10pct_mean |",
        "|---|---|---|---|---|---|",
    ]
    for cond in conds_order:
        for roi in rois:
            sub = df[(df["condition"] == cond) & (df["roi"] == roi)]
            if sub.empty:
                continue
            n_med = int(sub["n_sensors"].median())
            row = f"| {cond} | {roi} | {n_med} "
            for s in ("mean_r", "top20_mean", "top10pct_mean"):
                row += f"| {sub[s].mean():.4f} ± {sub[s].std():.4f} "
            row += "|"
            lines.append(row)
    lines.append("")
    lines += ["## Violin plots (per ROI)", ""]
    for stat in STATS:
        pth = violin_paths.get(stat)
        if pth is not None:
            lines += [f"### {stat}",
                      f"![{stat}]({pth.relative_to(out_dir)})", ""]
    lines += ["## CSV", f"`{csv_path.relative_to(out_dir)}`"]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
