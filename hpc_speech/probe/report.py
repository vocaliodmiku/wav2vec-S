"""Assemble preflight_report.json.

Reports both the strict (proposal §5) and lenient (plan_pre_flight.md)
versions of the go/no-go criteria. No hard gate in code; decision is human.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict

from .shuffle_eb import ShuffleResult
from .surprisal import LevelStat


STRICT = {"rho_min": 0.35, "p_max": 1e-3, "depth_min": 0.15, "shuffle_min_rel": 0.10}
LENIENT = {"rho_min": 0.20, "p_max": 1e-2, "depth_min": 0.10, "shuffle_min_rel": 0.05}


def _evaluate(surprisal: Dict[str, LevelStat], shuffle: ShuffleResult, thr: Dict) -> Dict:
    deep = ["L3", "L4"]
    # criterion (a): rho >= thr at L3 and L4 (both cloze and gpt2 reported).
    a_cloze = all(
        surprisal[lvl].rho_cloze >= thr["rho_min"] and surprisal[lvl].p_cloze < thr["p_max"]
        for lvl in deep if lvl in surprisal
    )
    a_gpt2 = all(
        surprisal[lvl].rho_gpt2 >= thr["rho_min"] and surprisal[lvl].p_gpt2 < thr["p_max"]
        for lvl in deep if lvl in surprisal
    )
    # criterion (b): rho_L4 - rho_L0 >= thr (use cloze as primary).
    if "L0" in surprisal and "L4" in surprisal:
        b_cloze = (surprisal["L4"].rho_cloze - surprisal["L0"].rho_cloze) >= thr["depth_min"]
        b_gpt2 = (surprisal["L4"].rho_gpt2 - surprisal["L0"].rho_gpt2) >= thr["depth_min"]
    else:
        b_cloze = b_gpt2 = False
    # criterion (c): shuffle-e_b degrades CTC by >= thr; check L3 specifically.
    c_any = any(v >= thr["shuffle_min_rel"] for v in shuffle.rel_delta_e.values())
    c_L3 = shuffle.rel_delta_e.get("L3", 0.0) >= thr["shuffle_min_rel"]
    return {
        "a_cloze": a_cloze,
        "a_gpt2": a_gpt2,
        "b_cloze": b_cloze,
        "b_gpt2": b_gpt2,
        "c_any_level": c_any,
        "c_L3": c_L3,
        "thresholds": thr,
    }


def build_report(
    surprisal: Dict[str, LevelStat],
    shuffle: ShuffleResult,
    out_path: str,
) -> Dict:
    warnings = []
    # Flag cloze vs gpt2 disagreement at the deep levels.
    for lvl in ("L3", "L4"):
        if lvl in surprisal:
            s = surprisal[lvl]
            sign_c = s.rho_cloze > 0
            sign_g = s.rho_gpt2 > 0
            if sign_c != sign_g:
                warnings.append(f"cloze and gpt2 disagree on sign at {lvl}")
            if abs(s.rho_cloze - s.rho_gpt2) > 0.2:
                warnings.append(f"cloze vs gpt2 rho differ by >0.2 at {lvl}")

    report = {
        "criteria_strict": _evaluate(surprisal, shuffle, STRICT),
        "criteria_lenient": _evaluate(surprisal, shuffle, LENIENT),
        "surprisal_per_level": {k: asdict(v) for k, v in surprisal.items()},
        "shuffle": {
            "baseline_loss": shuffle.baseline_loss,
            "shuffle_e_losses": shuffle.shuffle_e_losses,
            "shuffle_s_losses": shuffle.shuffle_s_losses,
            "rel_delta_e": shuffle.rel_delta_e,
            "rel_delta_s": shuffle.rel_delta_s,
        },
        "warnings": warnings,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
