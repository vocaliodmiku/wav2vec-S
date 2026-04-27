"""Sensor-ROI assignment for MEG-MASC encoding-result reports.

The MEG-MASC dataset uses a KIT 208-channel axial-gradiometer system. There
is no built-in atlas, so ROIs are derived geometrically from the per-sensor
``ch_positions`` (head-frame XYZ in metres).

Convention (matches MNE / MNE-BIDS head coords): +x = right, +y = anterior,
+z = up. Cuts are percentile-based on (x, y), so the same logic works
across subjects without hand-tuned thresholds.

Disjoint labels (cover every sensor exactly once):
    frontal      — y in top 30 %
    parietal     — y in bottom 30 %
    temporal_L   — mid-AP (30..70 %), x in bottom 30 %
    temporal_R   — mid-AP, x in top 30 %
    central      — mid-AP, x in middle 40 %

Derived (overlap with the disjoint labels):
    temporal     — temporal_L ∪ temporal_R   (bilateral auditory belt)
    all          — every sensor              (no restriction; sanity check)
"""
from __future__ import annotations

import numpy as np

ROI_LABELS: tuple[str, ...] = (
    "frontal", "temporal_L", "temporal_R", "central", "parietal",
)
DERIVED_ROIS: tuple[str, ...] = ("temporal", "all")
ALL_ROIS: tuple[str, ...] = ROI_LABELS + DERIVED_ROIS


def assign_rois(positions: np.ndarray) -> dict[str, np.ndarray]:
    """Map ``(N, 3)`` head-frame positions to ``{roi: bool_mask of shape (N,)}``.

    Percentile cuts make the labelling robust to subject-specific scaling
    of the head coordinate frame.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(f"positions must be (N, >=2); got {pos.shape}")
    x, y = pos[:, 0], pos[:, 1]

    y_lo, y_hi = np.quantile(y, [0.30, 0.70])
    x_lo, x_hi = np.quantile(x, [0.30, 0.70])

    frontal = y >= y_hi
    parietal = y <= y_lo
    mid_ap = (~frontal) & (~parietal)
    temp_L = mid_ap & (x <= x_lo)
    temp_R = mid_ap & (x >= x_hi)
    central = mid_ap & (~temp_L) & (~temp_R)

    masks: dict[str, np.ndarray] = {
        "frontal": frontal,
        "temporal_L": temp_L,
        "temporal_R": temp_R,
        "central": central,
        "parietal": parietal,
    }
    masks["temporal"] = temp_L | temp_R
    masks["all"] = np.ones_like(frontal, dtype=bool)
    return masks


def summarize_in_roi(r: np.ndarray, mask: np.ndarray) -> dict:
    """Within-ROI scalar summary of a per-sensor correlation vector.

    ``top20_mean`` clamps the cap to ``min(20, n_in_roi)`` so small ROIs
    don't silently degrade to "mean of fewer than 20"; ``top10pct_mean``
    is computed on the in-ROI count, not the full sensor array.
    """
    r_in = np.asarray(r)[mask]
    n = int(r_in.size)
    if n == 0:
        return dict(
            n_sensors=0,
            mean_r=float("nan"),
            top20_mean=float("nan"),
            top10pct_mean=float("nan"),
        )
    sorted_desc = np.sort(r_in)[::-1]
    k20 = min(20, n)
    k_frac = max(1, int(round(0.10 * n)))
    return dict(
        n_sensors=n,
        mean_r=float(r_in.mean()),
        top20_mean=float(sorted_desc[:k20].mean()),
        top10pct_mean=float(sorted_desc[:k_frac].mean()),
    )
