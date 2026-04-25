"""Report cached HPSN preflight results (E1 / E2 / Δ = E2 − E1).

Auto-globs ``preflight_hpsn_*.pkl`` under ``--results_dir`` and emits:
  - ``per_subject_summary.csv`` — per-subject scalar stats (mean_r, top-20
    channel mean, top-10 %% channel mean) for E1, E2, and Δ.
  - ``figs/topomap_{E1,E2,Delta}.png`` — grand-average topomaps
    (Fisher-z mean across subjects; sessions averaged within subject first).
  - ``figs/violin_{mean_r,top20_mean,top10pct_mean}.png`` — per-subject
    distributions per condition.
  - ``report.md`` — pointer to the CSV + figures, with a summary table.

Only pkls fit in ``full_lags`` mode are processed (per-lag pkls are skipped).

Usage
-----
    python -m hpsn.evaluation.report_preflight_hpsn \\
        --results_dir /path/to/results [--out_dir /path/for/report]
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

from .proc_5_hpsn_ridge import RESULTS_DIR

PKL_RE = re.compile(
    r"preflight_hpsn_(?P<space>sensor|roi)_(?P<resample>MEG|REPR)_"
    r"sub-(?P<subj>[^_]+)_ses-(?P<ses>\d+)\.pkl"
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


# ─────────────────────────────────────────────────────────────────────────────
# Pkl discovery & loading
# ─────────────────────────────────────────────────────────────────────────────

def discover_pkls(results_dir: Path) -> list[dict]:
    out = []
    for pkl in results_dir.glob("trf_ridge_sub-*/preflight_hpsn_*.pkl"):
        m = PKL_RE.match(pkl.name)
        if not m:
            continue
        out.append(dict(
            path=pkl,
            subj=m.group("subj"),
            ses=int(m.group("ses")),
            space=m.group("space"),
            resample=m.group("resample"),
        ))
    return out


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sensor info (for topomaps)
# ─────────────────────────────────────────────────────────────────────────────

def load_sensor_info_from_cache(results_dir: Path, subj: str):
    """Find any cached MEG .fif for ``subj`` and return its ``info``.

    Tries the Défossez-preprocessed cache first, then falls back to the legacy
    bandpass-only cache (for reports over older pkls).
    """
    for name in (
        f"meg_cache_defossez_sub-{subj}",
        f"meg_cache_sub-{subj}",
    ):
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
        cmap = "RdBu_r"
    else:
        vlim = (None, None)
        cmap = "RdBu_r"
    im, _ = mne.viz.plot_topomap(
        r_per_sensor, info, axes=ax, show=False, cmap=cmap,
        vlim=vlim, contours=0, sensors=True,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_violin(per_cond_data: dict, stat_name: str, out_path: Path):
    labels = [k for k in ("E1", "E2", "Delta") if k in per_cond_data]
    data = [per_cond_data[k] for k in labels]
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 1.2, 4))
    ax.violinplot(data, showmeans=True, showmedians=True)
    for i, d in enumerate(data, start=1):
        jitter = (np.random.default_rng(0).uniform(-0.05, 0.05, size=len(d)))
        ax.scatter(np.full(len(d), i) + jitter, d, s=12, alpha=0.6,
                   color="black", zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
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
    p = argparse.ArgumentParser(description="Report cached HPSN preflight results.")
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument("--out_dir", default=None,
                   help="Report output dir. Defaults to <results_dir>/preflight_report.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "preflight_report"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. discover pkls
    entries = discover_pkls(results_dir)
    if not entries:
        print(f"No preflight_hpsn_*.pkl found under {results_dir}")
        return
    print(f"Discovered {len(entries)} pkls under {results_dir}")

    # 2. collect per-(subj, ses) r arrays (full_lags only)
    # subj_sess_r[cond][subj] = list of r arrays (one per session)
    subj_sess_r: dict = {"E1": {}, "E2": {}}
    skipped_per_lag = 0
    for e in entries:
        pkl = _load_pkl(e["path"])
        for ck in ("E1", "E2"):
            res = pkl.get("conditions", {}).get(ck)
            if res is None:
                continue
            if res.get("mode") != "full_lags":
                skipped_per_lag += 1
                continue
            r = np.asarray(res["r"], dtype=np.float64)
            subj_sess_r[ck].setdefault(e["subj"], []).append(r)
    if skipped_per_lag:
        print(f"(skipped {skipped_per_lag} per-lag condition blocks)")
    if not any(subj_sess_r[ck] for ck in ("E1", "E2")):
        print("No full_lags results found; nothing to report.")
        return

    # 3. per-subject r (Fisher-z mean across sessions)
    subj_r: dict = {"E1": {}, "E2": {}}
    for ck in ("E1", "E2"):
        for subj, rs_list in subj_sess_r[ck].items():
            stack = np.stack(rs_list, axis=0)  # [n_ses, n_sensors]
            subj_r[ck][subj] = fisher_z_mean(stack, axis=0)

    # Subjects with matching sensor counts across E1 and E2 (needed for Δ)
    def _n_sensors(ck):
        sizes = {r.size for r in subj_r[ck].values()}
        return sizes

    for ck in ("E1", "E2"):
        if len(_n_sensors(ck)) > 1:
            print(f"WARN: {ck} has mixed n_sensors across subjects: {_n_sensors(ck)}")

    # 4. per-subject scalar stats (feeds violins + CSV)
    rows = []
    for ck in ("E1", "E2"):
        for subj, r in subj_r[ck].items():
            s = summarize_r(r)
            rows.append(dict(subj=subj, condition=ck, n_sensors=s["n_sensors"],
                             mean_r=s["mean_r"], mean_r2=s["mean_r2"],
                             top20_mean=s["top20_mean"],
                             top10pct_mean=s["top10pct_mean"]))
    # Δ = per-subject E2 - E1 (per-sensor), then summarize
    common_subj = sorted(set(subj_r["E1"]) & set(subj_r["E2"]))
    for subj in common_subj:
        r1, r2 = subj_r["E1"][subj], subj_r["E2"][subj]
        if r1.shape != r2.shape:
            continue
        d = r2 - r1
        d_r2 = r2 ** 2 - r1 ** 2
        d_sorted_desc = np.sort(d)[::-1]
        k_frac = max(1, int(round(TOPK_FRAC * d.size)))
        rows.append(dict(
            subj=subj, condition="Delta", n_sensors=d.size,
            mean_r=float(d.mean()),
            mean_r2=float(d_r2.mean()),
            top20_mean=float(d_sorted_desc[:TOPK_ABS].mean()),
            top10pct_mean=float(d_sorted_desc[:k_frac].mean()),
        ))
    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_subject_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # 5. grand per-sensor r (Fisher-z mean across subjects) + Δ
    grand: dict = {}
    for ck in ("E1", "E2"):
        stacks = [r for r in subj_r[ck].values()]
        sizes = {r.size for r in stacks}
        if not stacks or len(sizes) != 1:
            if stacks:
                print(f"WARN: skipping grand {ck} topomap (mixed n_sensors {sizes}).")
            continue
        grand[ck] = fisher_z_mean(np.stack(stacks, axis=0), axis=0)
    if "E1" in grand and "E2" in grand and grand["E1"].shape == grand["E2"].shape:
        grand["Delta"] = grand["E2"] - grand["E1"]

    # 6. topomaps
    info = None
    any_subj = next(iter(subj_r["E1"] or subj_r["E2"]))
    info = load_sensor_info_from_cache(results_dir, any_subj)
    topomap_paths: dict = {}
    if info is None:
        print(f"WARN: no MEG cache for sub-{any_subj}; topomaps skipped.")
    else:
        n_info = len(info["chs"])
        for ck, r in grand.items():
            if r.size != n_info:
                print(f"WARN: {ck} has {r.size} sensors but info has {n_info}; skipping topomap.")
                continue
            pth = figs_dir / f"topomap_{ck}.png"
            title = f"Grand r — {ck}" + ("" if ck == "Delta" else " (Fisher-z mean)")
            plot_topomap(r, info, title=title, out_path=pth,
                         symmetric=(ck == "Delta"))
            topomap_paths[ck] = pth
            print(f"Wrote {pth}")

    # 7. violin plots
    violin_paths: dict = {}
    for stat in STATS:
        data_by_cond = {
            ck: df[df["condition"] == ck][stat].dropna().to_numpy()
            for ck in ("E1", "E2", "Delta")
            if (df["condition"] == ck).any()
        }
        pth = figs_dir / f"violin_{stat}.png"
        plot_violin(data_by_cond, stat, pth)
        violin_paths[stat] = pth
        print(f"Wrote {pth}")

    # 8. markdown report
    n_subj = df["subj"].nunique()
    lines = [
        "# HPSN preflight report",
        "",
        f"- results dir: `{results_dir}`",
        f"- subjects: **{n_subj}**",
        f"- condition rows: **{len(df)}**",
        f"- aggregation: Fisher-z mean across subjects "
        "(sessions averaged within subject first)",
        "",
        "## Per-condition summary (mean ± s.d. across subjects)",
        "",
        "| Condition | mean_r | mean_r2 | top20_mean | top10pct_mean |",
        "|-----------|--------|---------|------------|---------------|",
    ]
    table_stats = ("mean_r", "mean_r2", "top20_mean", "top10pct_mean")
    for ck in ("E1", "E2", "Delta"):
        sub = df[df["condition"] == ck]
        if sub.empty:
            continue
        row = "| {} ".format(ck)
        for stat in table_stats:
            row += "| {:.4f} ± {:.4f} ".format(float(sub[stat].mean()),
                                                float(sub[stat].std()))
        row += "|"
        lines.append(row)
    lines.append("")
    if topomap_paths:
        lines.append("## Topomaps (grand-average)")
        lines.append("")
        for ck in ("E1", "E2", "Delta"):
            pth = topomap_paths.get(ck)
            if pth is None:
                continue
            rel = pth.relative_to(out_dir)
            lines.append(f"### {ck}")
            lines.append(f"![{ck}]({rel})")
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
    lines.append("## CSV")
    lines.append(f"`{csv_path.relative_to(out_dir)}`")
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
