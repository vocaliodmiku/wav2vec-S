"""Report cached Défossez-style baseline MEG-MASC ridge-TRF results.

Auto-globs ``trf_ridge_{feat}_{space}_{resample}_sub-*_ses-*.pkl`` under
``--results_dir`` (one pkl per subject × session × feature; produced by
``baseline_meg_masc.py``) and emits:
  - ``per_subject_summary.csv`` — per-subject scalar stats (mean_r, top-20
    channel mean, top-10 %% channel mean) per feature.
  - ``figs/topomap_{feat}.png`` — grand-average topomaps
    (Fisher-z mean across subjects; sessions averaged within subject first).
  - ``figs/violin_{mean_r,top20_mean,top10pct_mean}.png`` — per-subject
    distributions per feature.
  - ``figs/layer_trajectory_{stat}.png`` — when the run includes a
    wav2vec2 layer sweep (feats matching ``w2v2_lN`` / ``w2v2rand_lN``),
    a per-stat trajectory across layer index, with separate lines for
    pretrained vs random-init families and an optional baseline_feat
    horizontal reference.
  - ``report.md`` — pointer to the CSV + figures, with a summary table.

w2v2 feat naming (set by ``baseline_meg_masc.py``):
    ``w2v2_lN``       — pretrained wav2vec2-large hidden state at tap N
    ``w2v2rand_lN``   — random-init control at tap N
    (N = 0 means post-CNN/pre-transformer; N ∈ [1, num_hidden_layers] are
    transformer outputs.)

If ``--baseline_feat`` is set, Δ(feat − baseline_feat) per-subject stats,
topomaps, and (for w2v2 layer sweeps) Δ-trajectory plots are emitted
alongside each non-baseline feat, mirroring the HPSN preflight report's
E2 − E1 contrast.

Only pkls fit in ``full_lags`` mode are processed (per-lag pkls are skipped).

Usage
-----
    python -m hpsn.evaluation.report_baseline_meg_masc \\
        --results_dir /path/to/meg_results_defossez

    python -m hpsn.evaluation.report_baseline_meg_masc \\
        --baseline_feat acoustic
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .baseline_meg_masc import RESULTS_DIR

PKL_RE = re.compile(
    r"^trf_ridge_(?P<feat>.+?)_(?P<space>sensor|roi)_(?P<resample>MEG|REPR)_"
    r"sub-(?P<subj>[^_]+)_ses-(?P<ses>\d+)\.pkl$"
)
TOPK_ABS = 20
TOPK_FRAC = 0.10
STATS = ("mean_r", "top20_mean", "top10pct_mean")


# ─────────────────────────────────────────────────────────────────────────────
# Fisher-z helpers
# ─────────────────────────────────────────────────────────────────────────────

def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def fisher_z_inv(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def fisher_z_mean(r_stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Element-wise Fisher-z average of correlations."""
    return fisher_z_inv(fisher_z(r_stack).mean(axis=axis))


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics on a per-sensor r vector
# ─────────────────────────────────────────────────────────────────────────────

def summarize_r(r: np.ndarray) -> dict:
    r_sorted_desc = np.sort(r)[::-1]
    k_frac = max(1, int(round(TOPK_FRAC * r.size)))
    return dict(
        n_sensors=r.size,
        mean_r=float(r.mean()),
        mean_r2=float((r ** 2).mean()),
        top20_mean=float(r_sorted_desc[:TOPK_ABS].mean()),
        top10pct_mean=float(r_sorted_desc[:k_frac].mean()),
    )


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


W2V2_LAYER_RE = re.compile(r"^(w2v2(?:rand)?)_l(\d+)$")
W2V2_LAYER_DELTA_RE = re.compile(r"^Δ\[(w2v2(?:rand)?)_l(\d+)−.+\]$")


def _w2v2_layer_info(feat: str) -> tuple[str, int] | None:
    """``(family, layer_idx)`` for w2v2 layer feats / Δ-labels, else None."""
    m = W2V2_LAYER_RE.match(feat) or W2V2_LAYER_DELTA_RE.match(feat)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _feat_sort_key(feat: str):
    """Order non-w2v2 feats first (alphabetical), then w2v2 by family + layer."""
    info = _w2v2_layer_info(feat)
    if info is None:
        return (0, feat, 0)
    family, layer_idx = info
    return (1, family, layer_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Pkl discovery & loading
# ─────────────────────────────────────────────────────────────────────────────

def discover_pkls(results_dir: Path) -> list[dict]:
    out = []
    for pkl in results_dir.glob("trf_ridge_sub-*/trf_ridge_*.pkl"):
        m = PKL_RE.match(pkl.name)
        if not m:
            continue
        out.append(dict(
            path=pkl,
            subj=m.group("subj"),
            ses=int(m.group("ses")),
            space=m.group("space"),
            resample=m.group("resample"),
            feat=m.group("feat"),
        ))
    return out


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sensor info (for topomaps)
# ─────────────────────────────────────────────────────────────────────────────

def load_sensor_info_from_cache(results_dir: Path, subj: str):
    """Try the Défossez baseline cache first, then legacy HPSN cache."""
    for name in (f"meg_cache_defossez_sub-{subj}", f"meg_cache_sub-{subj}"):
        cache_dir = results_dir / name
        if not cache_dir.exists():
            continue
        for fif in sorted(cache_dir.glob("*.fif")):
            raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
            raw.pick_types(meg=True)
            return raw.info
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_topomap(r_per_sensor, info, title, out_path, *, symmetric=False):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    if symmetric:
        v = float(np.nanmax(np.abs(r_per_sensor)))
        vlim = (-v, v)
    else:
        vlim = (None, None)
    im, _ = mne.viz.plot_topomap(
        r_per_sensor, info, axes=ax, show=False, cmap="RdBu_r",
        vlim=vlim, contours=0, sensors=True,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_trajectory(
    df: pd.DataFrame,
    conds: list[str],
    stat: str,
    out_path: Path,
    *,
    baseline_feat: str | None = None,
    title_suffix: str = "",
) -> bool:
    """Plot ``stat`` (mean ± SEM across subjects) vs w2v2 layer index.

    One line per family present in ``conds`` (``w2v2``, ``w2v2rand``).
    Returns True iff at least one family had data and the figure was written.
    """
    families: dict[str, list[tuple[int, float, float]]] = {}
    for cond in conds:
        info = _w2v2_layer_info(cond)
        if info is None:
            continue
        family, layer_idx = info
        sub = df[df["condition"] == cond][stat].dropna().to_numpy()
        if sub.size == 0:
            continue
        sem = float(sub.std(ddof=1) / np.sqrt(sub.size)) if sub.size > 1 else 0.0
        families.setdefault(family, []).append((layer_idx, float(sub.mean()), sem))

    if not families:
        return False

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for family in sorted(families):
        pts = sorted(families[family], key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=family)

    if baseline_feat is not None:
        sub = df[df["condition"] == baseline_feat][stat].dropna().to_numpy()
        if sub.size > 0:
            ax.axhline(
                float(sub.mean()), ls="--", color="gray", lw=1.0,
                label=f"{baseline_feat} mean", zorder=1,
            )

    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("wav2vec2 layer (0 = post-CNN, ≥1 = transformer)")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} vs wav2vec2 layer{title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_violin(per_cond_data: dict, stat_name: str, out_path: Path):
    labels = list(per_cond_data.keys())
    data = [per_cond_data[k] for k in labels]
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 1.2, 4))
    ax.violinplot(data, showmeans=True, showmedians=True)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        jitter = rng.uniform(-0.05, 0.05, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=12, alpha=0.6,
                   color="black", zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(stat_name)
    ax.set_title(f"{stat_name} per subject")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Report cached baseline MEG-MASC ridge results."
    )
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--out_dir", default=None,
        help="Report output dir. Defaults to <results_dir>/baseline_report.",
    )
    p.add_argument(
        "--resample_opt", choices=("MEG", "REPR"), default=None,
        help="Filter pkls by resample_opt. Default: include all (warn on mix).",
    )
    p.add_argument("--space", choices=("sensor", "roi"), default="sensor")
    p.add_argument(
        "--feats", default=None,
        help="Comma-separated feat labels to include (default: all discovered).",
    )
    p.add_argument(
        "--baseline_feat", default=None,
        help="If set, compute Δ(feat − baseline_feat) for each non-baseline feat.",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "baseline_report"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. discover pkls and apply CLI filters
    entries = discover_pkls(results_dir)
    entries = [e for e in entries if e["space"] == args.space]
    if args.resample_opt:
        entries = [e for e in entries if e["resample"] == args.resample_opt]
    if args.feats:
        want = {f.strip() for f in args.feats.split(",") if f.strip()}
        entries = [e for e in entries if e["feat"] in want]
    if not entries:
        print(f"No trf_ridge_*.pkl matching filters under {results_dir}")
        return
    print(f"Discovered {len(entries)} pkls under {results_dir}")

    resamples = sorted({e["resample"] for e in entries})
    if len(resamples) > 1:
        print(
            f"WARN: mixed resample_opt in results: {resamples}. "
            "Pass --resample_opt to filter."
        )

    feats = sorted({e["feat"] for e in entries}, key=_feat_sort_key)
    print(f"Features: {feats}")
    if args.baseline_feat and args.baseline_feat not in feats:
        print(
            f"WARN: --baseline_feat {args.baseline_feat!r} not in discovered "
            f"feats {feats}; Δ will be skipped."
        )

    # 2. collect per-(subj, ses) r arrays (full_lags only), grouped by feat
    # subj_sess_r[feat][subj] = list of r arrays (one per session)
    subj_sess_r: dict = {f: {} for f in feats}
    skipped_per_lag = 0
    skipped_no_r = 0
    for e in entries:
        pkl = _load_pkl(e["path"])
        if pkl.get("mode") != "full_lags":
            skipped_per_lag += 1
            continue
        r = pkl.get("r")
        if r is None:
            skipped_no_r += 1
            continue
        subj_sess_r[e["feat"]].setdefault(e["subj"], []).append(
            np.asarray(r, dtype=np.float64)
        )
    if skipped_per_lag:
        print(f"(skipped {skipped_per_lag} per-lag pkls)")
    if skipped_no_r:
        print(f"(skipped {skipped_no_r} pkls with no 'r' field)")
    if not any(subj_sess_r[f] for f in feats):
        print("No full_lags results found; nothing to report.")
        return

    # 3. per-subject r (Fisher-z mean across sessions)
    subj_r: dict = {f: {} for f in feats}
    for f in feats:
        for subj, rs_list in subj_sess_r[f].items():
            stack = np.stack(rs_list, axis=0)  # [n_ses, n_sensors]
            subj_r[f][subj] = fisher_z_mean(stack, axis=0)

    for f in feats:
        sizes = {r.size for r in subj_r[f].values()}
        if len(sizes) > 1:
            print(f"WARN: {f} has mixed n_sensors across subjects: {sizes}")

    # 4. per-subject scalar stats (feeds violins + CSV)
    rows = []
    for f in feats:
        for subj, r in subj_r[f].items():
            s = summarize_r(r)
            rows.append(dict(
                subj=subj, condition=f, n_sensors=s["n_sensors"],
                mean_r=s["mean_r"], mean_r2=s["mean_r2"],
                top20_mean=s["top20_mean"],
                top10pct_mean=s["top10pct_mean"],
            ))

    # Δ = per-subject (feat − baseline), then summarize
    delta_feats: list[str] = []
    if args.baseline_feat and args.baseline_feat in feats:
        base = args.baseline_feat
        for f in feats:
            if f == base:
                continue
            common_subj = sorted(set(subj_r[f]) & set(subj_r[base]))
            if not common_subj:
                continue
            any_delta_for_f = False
            for subj in common_subj:
                r1, r2 = subj_r[base][subj], subj_r[f][subj]
                if r1.shape != r2.shape:
                    continue
                d = r2 - r1
                d_r2 = r2 ** 2 - r1 ** 2
                d_sorted_desc = np.sort(d)[::-1]
                k_frac = max(1, int(round(TOPK_FRAC * d.size)))
                rows.append(dict(
                    subj=subj, condition=f"Δ[{f}−{base}]", n_sensors=d.size,
                    mean_r=float(d.mean()),
                    mean_r2=float(d_r2.mean()),
                    top20_mean=float(d_sorted_desc[:TOPK_ABS].mean()),
                    top10pct_mean=float(d_sorted_desc[:k_frac].mean()),
                ))
                any_delta_for_f = True
            if any_delta_for_f:
                delta_feats.append(f)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_subject_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # 5. grand per-sensor r (Fisher-z mean across subjects) + Δ
    grand: dict = {}
    for f in feats:
        stacks = list(subj_r[f].values())
        sizes = {r.size for r in stacks}
        if not stacks or len(sizes) != 1:
            if stacks:
                print(f"WARN: skipping grand {f} topomap (mixed n_sensors {sizes}).")
            continue
        grand[f] = fisher_z_mean(np.stack(stacks, axis=0), axis=0)

    if args.baseline_feat and args.baseline_feat in grand:
        base = args.baseline_feat
        for f in delta_feats:
            if f in grand and grand[f].shape == grand[base].shape:
                grand[f"Δ[{f}−{base}]"] = grand[f] - grand[base]

    # 6. topomaps
    any_feat_with_subj = next((f for f in feats if subj_r[f]), None)
    any_subj = (
        next(iter(subj_r[any_feat_with_subj])) if any_feat_with_subj else None
    )
    info = load_sensor_info_from_cache(results_dir, any_subj) if any_subj else None
    topomap_paths: dict = {}
    if info is None:
        print("WARN: no MEG cache found; topomaps skipped.")
    else:
        n_info = len(info["chs"])
        for cond, r in grand.items():
            if r.size != n_info:
                print(
                    f"WARN: {cond} has {r.size} sensors but info has {n_info}; "
                    "skipping topomap."
                )
                continue
            pth = figs_dir / f"topomap_{_slug(cond)}.png"
            is_delta = cond.startswith("Δ[")
            title = (
                f"Grand r — {cond}"
                + ("" if is_delta else " (Fisher-z mean)")
            )
            plot_topomap(r, info, title=title, out_path=pth, symmetric=is_delta)
            topomap_paths[cond] = pth
            print(f"Wrote {pth}")

    # 7. violin plots (one stat per figure, one column per condition)
    conds_order = list(feats) + [
        f"Δ[{f}−{args.baseline_feat}]" for f in delta_feats
    ]
    violin_paths: dict = {}
    for stat in STATS:
        data_by_cond = {
            c: df[df["condition"] == c][stat].dropna().to_numpy()
            for c in conds_order
            if (df["condition"] == c).any()
        }
        pth = figs_dir / f"violin_{stat}.png"
        plot_violin(data_by_cond, stat, pth)
        violin_paths[stat] = pth
        print(f"Wrote {pth}")

    # 7b. wav2vec2 layer-trajectory plots (only when w2v2 layer feats present).
    # Stat-vs-layer with a line per family (w2v2 vs w2v2rand) and an optional
    # baseline_feat horizontal reference. A second set of trajectories is
    # emitted on the Δ[w2v2_lN−base] rows when baseline_feat is set.
    trajectory_paths: dict = {}
    has_w2v2_layers = any(_w2v2_layer_info(f) is not None for f in feats)
    if has_w2v2_layers:
        for stat in STATS:
            pth = figs_dir / f"layer_trajectory_{stat}.png"
            if plot_layer_trajectory(
                df, list(feats), stat, pth,
                baseline_feat=args.baseline_feat,
            ):
                trajectory_paths[stat] = pth
                print(f"Wrote {pth}")
        if delta_feats:
            delta_conds = [f"Δ[{f}−{args.baseline_feat}]" for f in delta_feats]
            for stat in STATS:
                pth = figs_dir / f"layer_trajectory_delta_{stat}.png"
                if plot_layer_trajectory(
                    df, delta_conds, stat, pth,
                    baseline_feat=None,
                    title_suffix=f"  (Δ vs {args.baseline_feat})",
                ):
                    trajectory_paths[f"delta_{stat}"] = pth
                    print(f"Wrote {pth}")

    # 8. markdown report
    n_subj = df["subj"].nunique()
    header_line = f"- space: `{args.space}`"
    if args.resample_opt:
        header_line += f", resample_opt: `{args.resample_opt}`"
    elif resamples:
        header_line += f", resample_opt: {resamples}"
    feats_line = f"- features: {', '.join(feats)}"
    if args.baseline_feat:
        feats_line += f" | baseline = `{args.baseline_feat}`"
    lines = [
        "# Baseline MEG-MASC report",
        "",
        f"- results dir: `{results_dir}`",
        header_line,
        f"- subjects: **{n_subj}**",
        feats_line,
        f"- condition rows: **{len(df)}**",
        "- aggregation: Fisher-z mean across subjects "
        "(sessions averaged within subject first)",
        "",
        "## Per-condition summary (mean ± s.d. across subjects)",
        "",
        "| Condition | mean_r | mean_r2 | top20_mean | top10pct_mean |",
        "|-----------|--------|---------|------------|---------------|",
    ]
    table_stats = ("mean_r", "mean_r2", "top20_mean", "top10pct_mean")
    for cond in conds_order:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        row = "| {} ".format(cond)
        for stat in table_stats:
            row += "| {:.4f} ± {:.4f} ".format(
                float(sub[stat].mean()), float(sub[stat].std())
            )
        row += "|"
        lines.append(row)
    lines.append("")
    if topomap_paths:
        lines.append("## Topomaps (grand-average)")
        lines.append("")
        for cond in conds_order:
            pth = topomap_paths.get(cond)
            if pth is None:
                continue
            rel = pth.relative_to(out_dir)
            lines.append(f"### {cond}")
            lines.append(f"![{cond}]({rel})")
            lines.append("")
    lines.append("## Violin plots (per-subject distributions)")
    lines.append("")
    for stat in STATS:
        pth = violin_paths.get(stat)
        if pth is None:
            continue
        rel = pth.relative_to(out_dir)
        lines.append(f"### {stat}")
        lines.append(f"![{stat}]({rel})")
        lines.append("")
    if trajectory_paths:
        lines.append("## wav2vec2 layer trajectories")
        lines.append("")
        lines.append(
            "Per-subject mean ± SEM at each transformer tap; layer 0 is "
            "post-CNN/pre-transformer."
        )
        lines.append("")
        for stat in STATS:
            pth = trajectory_paths.get(stat)
            if pth is None:
                continue
            rel = pth.relative_to(out_dir)
            lines.append(f"### {stat}")
            lines.append(f"![{stat}]({rel})")
            lines.append("")
        if any(f"delta_{s}" in trajectory_paths for s in STATS):
            lines.append(f"### Δ vs `{args.baseline_feat}`")
            lines.append("")
            for stat in STATS:
                pth = trajectory_paths.get(f"delta_{stat}")
                if pth is None:
                    continue
                rel = pth.relative_to(out_dir)
                lines.append(f"#### Δ {stat}")
                lines.append(f"![Δ {stat}]({rel})")
                lines.append("")
    lines.append("## CSV")
    lines.append(f"`{csv_path.relative_to(out_dir)}`")
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
