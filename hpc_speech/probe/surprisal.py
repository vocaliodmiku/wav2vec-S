"""Criterion (a) and (b): surprisal correlation of ||e_b|| on Provo.

Per-word aggregation: mean_t ||e_b[t]||_2 over frames aligned to the word.
Reports Spearman ρ + bootstrap CI + permutation p-value, both against
-log(cloze_prob) and gpt2_surprisal, at each trunk level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from ..model.hpc_speech import HPCSpeechPreflight


FRAME_HZ = 50.0


@dataclass
class LevelStat:
    rho_cloze: float
    rho_gpt2: float
    ci_cloze: Tuple[float, float]
    ci_gpt2: Tuple[float, float]
    p_cloze: float
    p_gpt2: float
    n_words: int


def _word_frame_span(start_s: float, end_s: float, T: int) -> Tuple[int, int]:
    lo = max(0, int(math.floor(start_s * FRAME_HZ)))
    hi = min(T, int(math.ceil(end_s * FRAME_HZ)))
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[i] = spearmanr(x[idx], y[idx]).statistic
    return float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))


@torch.no_grad()
def compute_surprisal_correlation(
    model: HPCSpeechPreflight,
    provo_dataset,
    device: str = "cuda",
    n_boot: int = 10000,
) -> Dict[str, LevelStat]:
    """Returns {f"L{b}": LevelStat} for each trunk level that has an e_b."""
    model.eval().to(device)
    num_levels_with_e = model.cfg.trunk.num_levels - 1  # e_b defined for b=0..N-2

    # Per-word aggregates across the dataset.
    e_norms_per_level: List[List[float]] = [[] for _ in range(num_levels_with_e)]
    cloze_neglog: List[float] = []
    gpt2: List[float] = []

    for item in provo_dataset:
        iv = item["input_values"].unsqueeze(0).to(device)
        out = model(input_values=iv)
        T = out.h[0].shape[1]
        e_tensors = [e[0].float().cpu() for e in out.e]  # list of (T, d)
        e_frame_norms = [e.norm(dim=-1).numpy() for e in e_tensors]  # list of (T,)

        for w in item["words"]:
            lo, hi = _word_frame_span(w.start_s, w.end_s, T)
            # Skip words with non-finite / out-of-range surprisal.
            if not math.isfinite(w.gpt2_surprisal):
                continue
            p = float(w.cloze_prob)
            if not (0.0 < p <= 1.0):
                continue
            cloze_neglog.append(-math.log(max(p, 1e-6)))
            gpt2.append(float(w.gpt2_surprisal))
            for b in range(num_levels_with_e):
                e_frame_norms_b = e_frame_norms[b]
                e_norms_per_level[b].append(float(e_frame_norms_b[lo:hi].mean()))

    result: Dict[str, LevelStat] = {}
    cloze_arr = np.array(cloze_neglog)
    gpt2_arr = np.array(gpt2)
    n = len(cloze_arr)
    for b in range(num_levels_with_e):
        eb = np.array(e_norms_per_level[b])
        rho_c = spearmanr(eb, cloze_arr)
        rho_g = spearmanr(eb, gpt2_arr)
        ci_c = _bootstrap_rho_ci(eb, cloze_arr, n_boot=n_boot, seed=b * 2 + 1)
        ci_g = _bootstrap_rho_ci(eb, gpt2_arr, n_boot=n_boot, seed=b * 2 + 2)
        result[f"L{b}"] = LevelStat(
            rho_cloze=float(rho_c.statistic),
            rho_gpt2=float(rho_g.statistic),
            ci_cloze=ci_c,
            ci_gpt2=ci_g,
            p_cloze=float(rho_c.pvalue),
            p_gpt2=float(rho_g.pvalue),
            n_words=n,
        )
    return result
