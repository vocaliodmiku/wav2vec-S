"""ROI-restricted baseline MEG-MASC report.

Mirrors ``report_baseline_meg_masc.py`` but restricts every scalar
statistic to an anatomical sensor group (frontal / temporal_L /
temporal_R / central / parietal / temporal / all) using
``ch_positions`` embedded in the baseline pkls.

Outputs (under ``--out_dir``, defaults to ``<results_dir>/baseline_roi_report``):
  - ``per_subject_roi_summary.csv`` — one row per (subject, condition, ROI)
    with ``n_sensors``, ``mean_r``, ``top20_mean``, ``top10pct_mean``.
  - ``figs/violin_roi_{stat}.png`` — per-stat figure with one subplot per
    ROI; columns inside each subplot are conditions (and Δ rows when
    ``--baseline_feat`` is set).
  - ``figs/layer_trajectory_roi_{stat}.png`` — when w2v2 layer feats are
    present, per-stat trajectory subplot grid (one ROI per panel).
  - ``report.md`` — table + figure pointers.

Usage
-----
    python -m hpsn.evaluation.roi_baseline_meg_masc \\
        --results_dir /path/to/meg_results_defossez

    python -m hpsn.evaluation.roi_baseline_meg_masc \\
        --baseline_feat acoustic --rois frontal,temporal,parietal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline_meg_masc import RESULTS_DIR
from .report_baseline_meg_masc import (
    STATS,
    _feat_sort_key,
    _load_pkl,
    _w2v2_layer_info,
    discover_pkls,
    fisher_z_mean,
)
from .sensor_rois import ALL_ROIS, assign_rois, summarize_in_roi


def _violin_grid(
    df: pd.DataFrame, conds_order: list[str], rois: list[str],
    stat: str, out_path: Path, title: str,
) -> None:
    n_rois = len(rois)
    ncols = min(3, n_rois)
    nrows = int(np.ceil(n_rois / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.8 * ncols, 3.4 * nrows + 0.5),
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
            ax.scatter(np.full(len(d), j) + jitter, d, s=8, alpha=0.5,
                       color="black", zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
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


def _trajectory_grid(
    df: pd.DataFrame, feats: list[str], rois: list[str],
    stat: str, out_path: Path, baseline_feat: str | None,
) -> bool:
    n_rois = len(rois)
    ncols = min(3, n_rois)
    nrows = int(np.ceil(n_rois / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows + 0.5),
        squeeze=False, sharey=True,
    )
    axes_flat = axes.flatten()
    any_drawn = False
    for i, roi in enumerate(rois):
        ax = axes_flat[i]
        sub = df[df["roi"] == roi]
        families: dict[str, list[tuple[int, float, float]]] = {}
        for cond in feats:
            info = _w2v2_layer_info(cond)
            if info is None:
                continue
            family, layer = info
            v = sub[sub["condition"] == cond][stat].dropna().to_numpy()
            if v.size == 0:
                continue
            sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
            families.setdefault(family, []).append((layer, float(v.mean()), sem))
        if not families:
            ax.set_visible(False)
            continue
        any_drawn = True
        for family in sorted(families):
            pts = sorted(families[family], key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=family)
        if baseline_feat is not None:
            bsub = sub[sub["condition"] == baseline_feat][stat].dropna().to_numpy()
            if bsub.size > 0:
                ax.axhline(
                    float(bsub.mean()), ls="--", color="gray", lw=1.0,
                    label=f"{baseline_feat} mean",
                )
        ax.axhline(0, color="black", lw=0.4)
        ax.set_xlabel("w2v2 layer")
        if i % ncols == 0:
            ax.set_ylabel(stat)
        ax.set_title(roi)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for k in range(n_rois, nrows * ncols):
        axes_flat[k].set_visible(False)
    if any_drawn:
        fig.suptitle(f"{stat} vs w2v2 layer per ROI (per-subject mean ± SEM)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return any_drawn


def main():
    p = argparse.ArgumentParser(
        description="ROI-restricted baseline MEG-MASC report.",
    )
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--out_dir", default=None,
        help="Default: <results_dir>/baseline_roi_report.",
    )
    p.add_argument("--space", choices=("sensor", "roi"), default="sensor")
    p.add_argument(
        "--resample_opt", choices=("MEG", "REPR"), default=None,
        help="Filter pkls by resample_opt. Default: include all.",
    )
    p.add_argument("--feats", default=None,
                   help="Comma-separated feat labels to include.")
    p.add_argument(
        "--rois", default=",".join(ALL_ROIS),
        help=f"Comma-separated ROIs. Choices: {ALL_ROIS}.",
    )
    p.add_argument(
        "--baseline_feat", default=None,
        help="If set, compute Δ(feat − baseline_feat) per subject per ROI.",
    )
    args = p.parse_args()

    rois = [r.strip() for r in args.rois.split(",") if r.strip()]
    bad = [r for r in rois if r not in ALL_ROIS]
    if bad:
        raise SystemExit(f"Unknown ROIs: {bad}; choices: {ALL_ROIS}")

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "baseline_roi_report"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_pkls(results_dir)
    entries = [e for e in entries if e["space"] == args.space]
    if args.resample_opt:
        entries = [e for e in entries if e["resample"] == args.resample_opt]
    if args.feats:
        want = {f.strip() for f in args.feats.split(",") if f.strip()}
        entries = [e for e in entries if e["feat"] in want]
    if not entries:
        raise SystemExit(f"No pkls match filters under {results_dir}")
    print(f"Discovered {len(entries)} pkls under {results_dir}")

    feats = sorted({e["feat"] for e in entries}, key=_feat_sort_key)
    print(f"Features: {feats}")
    if args.baseline_feat and args.baseline_feat not in feats:
        print(
            f"WARN: --baseline_feat {args.baseline_feat!r} not in discovered "
            f"feats {feats}; Δ will be skipped."
        )

    subj_sess_r: dict = {f: {} for f in feats}
    subj_pos: dict = {}
    skipped_per_lag = skipped_no_r = skipped_no_pos = 0
    for e in entries:
        pkl = _load_pkl(e["path"])
        if pkl.get("mode") != "full_lags":
            skipped_per_lag += 1
            continue
        r = pkl.get("r")
        if r is None:
            skipped_no_r += 1
            continue
        pos = pkl.get("ch_positions")
        if pos is None:
            skipped_no_pos += 1
            continue
        subj_pos.setdefault(e["subj"], np.asarray(pos, dtype=np.float64))
        subj_sess_r[e["feat"]].setdefault(e["subj"], []).append(
            np.asarray(r, dtype=np.float64)
        )
    if skipped_per_lag:
        print(f"(skipped {skipped_per_lag} per-lag pkls)")
    if skipped_no_r:
        print(f"(skipped {skipped_no_r} pkls with no 'r' field)")
    if skipped_no_pos:
        print(f"(skipped {skipped_no_pos} pkls with no 'ch_positions' field — "
              f"these were probably written by an older baseline_meg_masc.py)")
    if not subj_pos:
        raise SystemExit(
            "No pkl had 'ch_positions'. Re-run baseline_meg_masc.py with the "
            "current code so channel meta is embedded."
        )

    subj_r: dict = {f: {} for f in feats}
    for f in feats:
        for subj, rs_list in subj_sess_r[f].items():
            stack = np.stack(rs_list, axis=0)
            subj_r[f][subj] = fisher_z_mean(stack, axis=0)

    rows = []
    for f in feats:
        for subj, r in subj_r[f].items():
            pos = subj_pos.get(subj)
            if pos is None or pos.shape[0] != r.size:
                continue
            masks = assign_rois(pos)
            for roi in rois:
                s = summarize_in_roi(r, masks[roi])
                rows.append(dict(
                    subj=subj, condition=f, roi=roi,
                    n_sensors=s["n_sensors"],
                    mean_r=s["mean_r"],
                    top20_mean=s["top20_mean"],
                    top10pct_mean=s["top10pct_mean"],
                ))

    delta_feats: list[str] = []
    if args.baseline_feat and args.baseline_feat in feats:
        base = args.baseline_feat
        for f in feats:
            if f == base:
                continue
            common = sorted(set(subj_r[f]) & set(subj_r[base]))
            if not common:
                continue
            wrote_any = False
            for subj in common:
                r1, r2 = subj_r[base][subj], subj_r[f][subj]
                if r1.shape != r2.shape:
                    continue
                pos = subj_pos.get(subj)
                if pos is None or pos.shape[0] != r1.size:
                    continue
                masks = assign_rois(pos)
                d = r2 - r1
                for roi in rois:
                    s = summarize_in_roi(d, masks[roi])
                    rows.append(dict(
                        subj=subj, condition=f"Δ[{f}−{base}]", roi=roi,
                        n_sensors=s["n_sensors"],
                        mean_r=s["mean_r"],
                        top20_mean=s["top20_mean"],
                        top10pct_mean=s["top10pct_mean"],
                    ))
                wrote_any = True
            if wrote_any:
                delta_feats.append(f)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_subject_roi_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    conds_order = list(feats) + [
        f"Δ[{f}−{args.baseline_feat}]" for f in delta_feats
    ]

    violin_paths: dict = {}
    for stat in STATS:
        pth = figs_dir / f"violin_roi_{stat}.png"
        _violin_grid(df, conds_order, rois, stat, pth,
                     title=f"{stat} per ROI per subject")
        violin_paths[stat] = pth
        print(f"Wrote {pth}")

    trajectory_paths: dict = {}
    if any(_w2v2_layer_info(f) is not None for f in feats):
        for stat in STATS:
            pth = figs_dir / f"layer_trajectory_roi_{stat}.png"
            if _trajectory_grid(df, feats, rois, stat, pth, args.baseline_feat):
                trajectory_paths[stat] = pth
                print(f"Wrote {pth}")

    n_subj = df["subj"].nunique()
    lines = [
        "# Baseline MEG-MASC ROI report",
        "",
        f"- results dir: `{results_dir}`",
        f"- subjects: **{n_subj}**",
        f"- features: {', '.join(feats)}",
        f"- ROIs: {', '.join(rois)}",
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
    if trajectory_paths:
        lines += ["## w2v2 layer trajectories per ROI", ""]
        for stat in STATS:
            pth = trajectory_paths.get(stat)
            if pth is not None:
                lines += [f"### {stat}",
                          f"![{stat}]({pth.relative_to(out_dir)})", ""]
    lines += ["## CSV", f"`{csv_path.relative_to(out_dir)}`"]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
