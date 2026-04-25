"""HPSN preflight ridge-TRF encoding (E1 = hpsn_l1, E2 = hpsn_l2).

Thin orchestrator around the existing HPSN evaluation machinery in
``proc_5_hpsn_ridge.py``. For each (subject, session) in the requested sweep,
fits ridge encoding for BOTH conditions E1 and E2 in one pass, reusing the
HPSN feature HDF5 and MEG cache. Emits one "ladder" pkl per (subj, ses)
holding both conditions side-by-side, so you can eyeball HPSN-Level-1 vs
HPSN-Level-2 encoding performance before running the full ablation grid.

See ``minimal-2.md`` §9.3 for the ladder naming (E1 / E2).

Usage
-----
    # one subject, one session
    python -m hpsn.evaluation.preflight_hpsn \\
        --subj 01 --ses 0 --ckpt /path/to/hpsn_ckpt_dir

    # multi-subject / multi-session; reuse pkls that already exist
    python -m hpsn.evaluation.preflight_hpsn \\
        --subj 01,02,03 --ses 0,1 --ckpt <path>

    # full sweep in per-lag mode
    python -m hpsn.evaluation.preflight_hpsn \\
        --subj all --ses all --ckpt <path> --per_lag
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import sys
import warnings
from pathlib import Path

import h5py
import mne
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level("WARNING")

from ..config import HPSNConfig
from .features import (
    HPSNFeatureExtractor,
    NATIVE_FRAME_RATE,
    _find_state_dict_file,
    _load_state_dict,
    extract_to_hdf5,
)
from .proc_5_hpsn_ridge import (
    BIDS_ROOT,
    DEFAULT_TRF_TMAX,
    DEFAULT_TRF_TMIN,
    PCA_COMPONENTS,
    PREPROCESSING_TAG,
    RESULTS_DIR,
    SOURCE_DATA_DIR,
    SR,
    build_condition_matrix,
    build_lagged_matrix,
    build_single_lag_matrix,
    collect_unique_stimuli,
    fit_ridge_cv_full_lags,
    fit_ridge_cv_single_lag,
    get_subjects,
    load_roi_run,
    load_sensor_run,
    parse_events_tsv,
)


# Preflight condition map — minimal-2.md §9.3 naming:
#   E1 = HPSN Level 1 representation  (hpsn_l1)
#   E2 = HPSN Level 2 representation  (hpsn_l2)
PREFLIGHT_CONDS = {
    "E1": "hpsn_l1",
    "E2": "hpsn_l2",
}

MEG_MASC_SESSIONS = (0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model-size presets
# ─────────────────────────────────────────────────────────────────────────────
#
# The HPSN training pipeline (hpsn/training/train.py) rescales the default
# Large tap ranges when it sees a shallower backbone. A Base backbone has
# 12 transformer layers vs Large's 24, so the scale factor is 0.5 and the
# default taps (1-8, 13-20) become (1-4, 6-10). We replicate that here so
# that evaluating a Base-trained checkpoint doesn't require six override
# flags.
#
# Fields left as None are filled from the HPSNConfig dataclass defaults.

MODEL_SIZE_PRESETS = {
    "large": {
        "hidden_dim": 1024,
        "tap_acoustic_start": 1, "tap_acoustic_end": 8,
        "tap_lexical_start": 13, "tap_lexical_end": 20,
        "backbone_default": "biaofu-xmu/wav2vec-S-Large-ft-960h",
    },
    "base": {
        "hidden_dim": 768,
        "tap_acoustic_start": 1, "tap_acoustic_end": 4,
        "tap_lexical_start": 6, "tap_lexical_end": 10,
        "backbone_default": "biaofu-xmu/wav2vec-S-Base-ft-960h"
    },
}


def log_stage(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] === {msg} ===", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / resolvers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HPSN preflight ridge-TRF (E1 = hpsn_l1, E2 = hpsn_l2).",
    )
    p.add_argument(
        "--subj", required=True,
        help="Subject ID(s): comma-separated list or 'all' "
             "(e.g., '01', '01,02,03', 'all').",
    )
    p.add_argument(
        "--ses", required=True,
        help="Session(s): comma-separated list or 'all' "
             "(e.g., '0', '0,1', 'all').",
    )
    p.add_argument(
        "--ckpt", required=True,
        help="Trained HPSN checkpoint (file or accelerate dir).",
    )
    p.add_argument(
        "--model_size", choices=("base", "large"), default="large",
        help="Preset for backbone size. 'base' → hidden_dim=768, taps 1-4 / 6-10 "
             "(matches the train-time rescaling for 12-layer backbones). "
             "'large' → hidden_dim=1024, taps 1-8 / 13-20. Individual --hidden_dim "
             "and --tap_* flags override this preset.",
    )
    p.add_argument(
        "--backbone_model", default=None,
        help="HuggingFace id or local path for the wav2vec-S backbone. "
             "Defaults to the preset's default (Large → biaofu-xmu/wav2vec-S-Large-ft-960h; "
             "Base → must be supplied).",
    )
    p.add_argument("--hidden_dim", type=int, default=None,
                   help="Override backbone hidden size (preset default applies if unset).")
    p.add_argument("--tap_acoustic_start", type=int, default=None)
    p.add_argument("--tap_acoustic_end", type=int, default=None)
    p.add_argument("--tap_lexical_start", type=int, default=None)
    p.add_argument("--tap_lexical_end", type=int, default=None)
    p.add_argument("--space", choices=("sensor", "roi"), default="sensor")
    p.add_argument(
        "--resample_opt", choices=("MEG", "REPR"), default="MEG",
        help="MEG = native 50 Hz, REPR = 100 Hz (features ×2-upsampled).",
    )
    p.add_argument("--trf_tmin", type=float, default=DEFAULT_TRF_TMIN)
    p.add_argument("--trf_tmax", type=float, default=DEFAULT_TRF_TMAX)
    p.add_argument(
        "--per_lag", action="store_true",
        help="Fit a single-lag ridge per lag in --lag_list_ms.",
    )
    p.add_argument(
        "--lag_list_ms", default="0,40,80,120,160,200,300,400,500",
        help="Lags (ms) used when --per_lag is set.",
    )
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument("--source_data_dir", default=str(SOURCE_DATA_DIR))
    p.add_argument(
        "--hdf5_path", default=None,
        help="Override HPSN feature HDF5 location.",
    )
    p.add_argument("--force_recompute_features", action="store_true")
    p.add_argument(
        "--force_refit", action="store_true",
        help="Refit ridge even if the ladder pkl already exists.",
    )
    p.add_argument(
        "--save_coef", action="store_true",
        help="Include ridge coefficients in the ladder pkl (large; off by default).",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_hpsn_config(args) -> HPSNConfig:
    """Assemble an HPSNConfig from the model-size preset and any per-field overrides."""
    preset = MODEL_SIZE_PRESETS[args.model_size]
    backbone = args.backbone_model or preset["backbone_default"]
    if backbone is None:
        print(
            f"ERROR: --backbone_model is required when --model_size={args.model_size} "
            f"(no default). Pass the HuggingFace id or local path of the backbone.",
            file=sys.stderr,
        )
        sys.exit(2)
    cfg = HPSNConfig(backbone_model=backbone)
    cfg.hidden_dim = args.hidden_dim if args.hidden_dim is not None else preset["hidden_dim"]
    cfg.tap_acoustic_start = (
        args.tap_acoustic_start if args.tap_acoustic_start is not None
        else preset["tap_acoustic_start"]
    )
    cfg.tap_acoustic_end = (
        args.tap_acoustic_end if args.tap_acoustic_end is not None
        else preset["tap_acoustic_end"]
    )
    cfg.tap_lexical_start = (
        args.tap_lexical_start if args.tap_lexical_start is not None
        else preset["tap_lexical_start"]
    )
    cfg.tap_lexical_end = (
        args.tap_lexical_end if args.tap_lexical_end is not None
        else preset["tap_lexical_end"]
    )
    if cfg.tap_acoustic_end >= cfg.tap_lexical_start:
        print(
            f"ERROR: tap_acoustic_end ({cfg.tap_acoustic_end}) must be strictly less than "
            f"tap_lexical_start ({cfg.tap_lexical_start}).",
            file=sys.stderr,
        )
        sys.exit(2)
    return cfg


def check_checkpoint_matches_config(ckpt_path, cfg: HPSNConfig) -> None:
    """Read the ckpt state_dict and fail loudly on shape mismatch before we try to load."""
    sd_file = _find_state_dict_file(Path(ckpt_path))
    sd = _load_state_dict(sd_file)

    def _shape(key):
        t = sd.get(key, None)
        return tuple(t.shape) if t is not None else None

    tap_ac = _shape("tap_acoustic.weights")
    tap_lex = _shape("tap_lexical.weights")
    l1_in = _shape("level1.input_proj.weight")          # [lstm_dim, hidden_dim]
    l1_recon = _shape("level1.recon_head.weight")       # [hidden_dim, lstm_dim]

    expected_ac = cfg.tap_acoustic_end - cfg.tap_acoustic_start + 1
    expected_lex = cfg.tap_lexical_end - cfg.tap_lexical_start + 1

    problems = []
    if tap_ac is not None and tap_ac[0] != expected_ac:
        problems.append(
            f"  tap_acoustic length: checkpoint={tap_ac[0]}, configured={expected_ac} "
            f"(range {cfg.tap_acoustic_start}-{cfg.tap_acoustic_end})"
        )
    if tap_lex is not None and tap_lex[0] != expected_lex:
        problems.append(
            f"  tap_lexical length:  checkpoint={tap_lex[0]}, configured={expected_lex} "
            f"(range {cfg.tap_lexical_start}-{cfg.tap_lexical_end})"
        )
    if l1_in is not None and l1_in[1] != cfg.hidden_dim:
        problems.append(
            f"  hidden_dim:          checkpoint={l1_in[1]}, configured={cfg.hidden_dim}"
        )
    if l1_recon is not None and l1_recon[0] != cfg.hidden_dim:
        problems.append(
            f"  recon_head hidden:   checkpoint={l1_recon[0]}, configured={cfg.hidden_dim}"
        )

    if problems:
        hint_base = (
            "try --model_size base "
            "(presets hidden_dim=768, taps 1-4 / 6-10)."
        )
        hint_overrides = (
            "You can also override per-field: --hidden_dim, "
            "--tap_acoustic_start/end, --tap_lexical_start/end."
        )
        print(
            "ERROR: HPSN checkpoint shape does not match the configured model.\n"
            + "\n".join(problems)
            + f"\nHint: {hint_base}\n{hint_overrides}",
            file=sys.stderr,
        )
        sys.exit(2)


def resolve_subjects(spec: str) -> list[str]:
    s = spec.strip()
    if s.lower() == "all":
        return sorted(get_subjects(BIDS_ROOT))
    return [x.strip() for x in s.split(",") if x.strip()]


def resolve_sessions(spec: str) -> list[int]:
    s = spec.strip()
    if s.lower() == "all":
        return list(MEG_MASC_SESSIONS)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Per-(subject, session) pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load_meg_runs(subj, ses, space, source_data_dir, meg_cache_dir, meg_rate):
    """Collect the list of run-dicts (with Y attached) for all tasks in (subj, ses)."""
    runs = []
    print(f"  sub-{subj} ses-{ses}:", end="")
    for task in range(4):
        evfile = (BIDS_ROOT / f"sub-{subj}" / f"ses-{ses}" / "meg" /
                  f"sub-{subj}_ses-{ses}_task-{task}_events.tsv")
        if not evfile.exists():
            continue

        if space == "sensor":
            Y = load_sensor_run(
                BIDS_ROOT, subj, ses, task, meg_rate, meg_cache_dir,
            )
            if Y is None:
                continue
        else:
            Y_roi, sf_roi = load_roi_run(source_data_dir, subj, ses, task)
            if Y_roi is None:
                print(f"[skip task-{task}: no ROI npy]", end="")
                continue
            if int(sf_roi) != meg_rate:
                Y = resample_poly(
                    Y_roi, int(meg_rate), int(sf_roi), axis=1
                ).astype(np.float64)
            else:
                Y = Y_roi

        ev = parse_events_tsv(evfile)
        sound_ev = ev[ev["kind"] == "sound"].reset_index(drop=True)
        runs.append(dict(
            subject=subj, session=ses, task=task,
            Y=Y, events_df=ev, sound_events=sound_ev,
        ))
        print(".", end="")
    print()
    return runs


def _build_condition_XY(runs, condition, h5_file, meg_rate):
    """Place per-stim features for ``condition`` across all runs; return concat (X, Y)."""
    Y_parts, X_parts = [], []
    for rd in runs:
        T_run = rd["Y"].shape[1]
        X_run, n_placed = build_condition_matrix(
            rd, condition, h5_file, T_run, meg_rate,
        )
        if n_placed == 0:
            continue
        Y_parts.append(rd["Y"])
        X_parts.append(X_run)
    if not Y_parts:
        return None, None
    return np.concatenate(X_parts, axis=0), np.concatenate(Y_parts, axis=1)


def _maybe_pca(X_all, cond_label):
    """Identity if D ≤ PCA_COMPONENTS; otherwise fit PCA on active frames."""
    if X_all.shape[1] <= PCA_COMPONENTS:
        return X_all
    active_mask = np.abs(X_all).sum(1) > 0
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(X_all[active_mask])
    X = pca.transform(X_all).astype(np.float64)
    print(f"    PCA on {cond_label}: "
          f"{pca.explained_variance_ratio_.sum():.2%} var → D={X.shape[1]}")
    return X


def _fit_one_condition(args, X_all, Y_all, meg_rate, cond_label, save_coef):
    n_sensors = Y_all.shape[0]
    if args.per_lag:
        lag_list_ms = [float(s) for s in args.lag_list_ms.split(",") if s.strip()]
        print(f"    per-lag: {len(lag_list_ms)} lags → {lag_list_ms}")
        r_by_lag = np.zeros((len(lag_list_ms), n_sensors), dtype=np.float64)
        for i, lag_ms in enumerate(lag_list_ms):
            X_lag, lag_frames = build_single_lag_matrix(X_all, meg_rate, lag_ms)
            r_by_lag[i] = fit_ridge_cv_single_lag(X_lag, Y_all)
            print(f"      lag {lag_ms:+6.1f} ms ({lag_frames:+d}f): "
                  f"mean r = {r_by_lag[i].mean():.4f}")
        return dict(
            mode="per_lag",
            r_by_lag=r_by_lag,
            lag_values_ms=np.array(lag_list_ms, dtype=np.float64),
            n_predictors=X_all.shape[1],
        )
    X_lagged, lags = build_lagged_matrix(
        X_all, meg_rate, args.trf_tmin, args.trf_tmax,
    )
    print(f"    lagged matrix: {X_lagged.shape}")
    r_values, coef_full, fold_alphas, full_alpha = fit_ridge_cv_full_lags(
        X_lagged, Y_all,
    )
    print(f"    {cond_label}: mean r = {r_values.mean():.4f}   "
          f"max r = {r_values.max():.4f}")
    result = dict(
        mode="full_lags",
        r=r_values,
        lags=lags,
        trf_tmin=args.trf_tmin, trf_tmax=args.trf_tmax,
        fold_alphas=fold_alphas, full_alpha=full_alpha,
        n_predictors=X_all.shape[1],
    )
    if save_coef:
        result["coef"] = coef_full.reshape(n_sensors, len(lags), X_all.shape[1])
    return result


def preflight_one_subject_session(
    args, subj, ses, *,
    cfg: HPSNConfig,
    h5_file, results_dir, source_data_dir, meg_rate, resample_opt,
) -> str:
    """Run E1 + E2 ridge for one (subj, ses). Returns 'done', 'cached', or 'no_runs'."""
    save_dir = results_dir / f"trf_ridge_sub-{subj}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = (
        f"preflight_hpsn_{args.space}_{resample_opt}_"
        f"sub-{subj}_ses-{ses}.pkl"
    )
    save_path = save_dir / save_name

    if save_path.exists() and not args.force_refit:
        log_stage(
            f"sub-{subj} ses-{ses}: cached → {save_name} "
            f"(pass --force_refit to redo)"
        )
        return "cached"

    log_stage(f"sub-{subj} ses-{ses}: loading MEG runs")
    meg_cache_dir = results_dir / f"meg_cache_defossez_sub-{subj}"
    runs = _load_meg_runs(
        subj, ses, args.space, source_data_dir, meg_cache_dir, meg_rate,
    )
    if not runs:
        log_stage(f"sub-{subj} ses-{ses}: no runs — skipping")
        return "no_runs"

    ladder = {}
    for E_key, cond_name in PREFLIGHT_CONDS.items():
        log_stage(f"sub-{subj} ses-{ses}: {E_key} ({cond_name}) — building X/Y")
        X_all, Y_all = _build_condition_XY(runs, cond_name, h5_file, meg_rate)
        if X_all is None:
            print(f"    {E_key}: no features placed — skipping.")
            continue
        print(f"    X={X_all.shape}, Y={Y_all.shape}")
        X_all = _maybe_pca(X_all, cond_name)
        log_stage(f"sub-{subj} ses-{ses}: {E_key} — ridge fit")
        res = _fit_one_condition(
            args, X_all, Y_all, meg_rate,
            cond_label=E_key, save_coef=args.save_coef,
        )
        res["feat"] = cond_name
        res["E_key"] = E_key
        ladder[E_key] = res

    if not ladder:
        log_stage(f"sub-{subj} ses-{ses}: no conditions produced results")
        return "no_runs"

    out = dict(
        conditions=ladder,
        meta=dict(
            subject=subj, session=ses,
            space=args.space, resample_opt=resample_opt, frame_rate=meg_rate,
            trf_tmin=args.trf_tmin, trf_tmax=args.trf_tmax,
            per_lag=args.per_lag,
            ckpt=str(args.ckpt),
            model_size=args.model_size,
            backbone_model=cfg.backbone_model,
            hidden_dim=cfg.hidden_dim,
            tap_acoustic=(cfg.tap_acoustic_start, cfg.tap_acoustic_end),
            tap_lexical=(cfg.tap_lexical_start, cfg.tap_lexical_end),
            hdf5_path=str(h5_file.filename) if h5_file is not None else None,
            preprocessing=PREPROCESSING_TAG,
            created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    with open(save_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved → {save_path}")
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    RESAMPLE_OPT = args.resample_opt
    MEG_RATE = 100 if RESAMPLE_OPT == "REPR" else NATIVE_FRAME_RATE

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir = Path(args.source_data_dir)
    hdf5_path = Path(args.hdf5_path) if args.hdf5_path else (
        results_dir / f"hpsn_features_{RESAMPLE_OPT}.h5"
    )

    subjects = resolve_subjects(args.subj)
    sessions = resolve_sessions(args.ses)
    if not subjects:
        print("ERROR: no subjects resolved from --subj.", file=sys.stderr)
        sys.exit(2)
    if not sessions:
        print("ERROR: no sessions resolved from --ses.", file=sys.stderr)
        sys.exit(2)

    cfg = build_hpsn_config(args)
    check_checkpoint_matches_config(args.ckpt, cfg)

    log_stage("Stage 1/4: Configuration")
    print(f"Subjects      : {subjects}")
    print(f"Sessions      : {sessions}")
    print(f"Conditions    : E1→hpsn_l1, E2→hpsn_l2")
    print(f"Model size    : {args.model_size}")
    print(f"Backbone      : {cfg.backbone_model}")
    print(f"Hidden dim    : {cfg.hidden_dim}")
    print(f"Tap acoustic  : {cfg.tap_acoustic_start}-{cfg.tap_acoustic_end} "
          f"(len {cfg.tap_acoustic_end - cfg.tap_acoustic_start + 1})")
    print(f"Tap lexical   : {cfg.tap_lexical_start}-{cfg.tap_lexical_end} "
          f"(len {cfg.tap_lexical_end - cfg.tap_lexical_start + 1})")
    print(f"Space         : {args.space}")
    print(f"Resample opt  : {RESAMPLE_OPT}  (MEG rate = {MEG_RATE} Hz)")
    print(f"Checkpoint    : {args.ckpt}")
    print(f"HDF5 features : {hdf5_path}")
    print(f"Results dir   : {results_dir}")
    print(f"Per-lag mode  : {args.per_lag}")
    print(f"Force refit   : {args.force_refit}")

    log_stage("Stage 2/4: Collecting unique stimuli")
    stim_map = collect_unique_stimuli([BIDS_ROOT])
    print(f"Found {len(stim_map)} unique stimuli")

    log_stage("Stage 3/4: Building HPSN features HDF5 (if needed)")
    if (not hdf5_path.exists()) or args.force_recompute_features:
        device = torch.device(args.device) if args.device else None
        extractor = HPSNFeatureExtractor(
            cfg, ckpt_path=args.ckpt,
            resample_opt=RESAMPLE_OPT, device=device,
        )
        extract_to_hdf5(
            stim_map, extractor, hdf5_path,
            sample_rate=SR, force=args.force_recompute_features,
        )
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        with h5py.File(hdf5_path, "r") as h5:
            fr = h5.attrs.get("frame_rate", None)
            ro = h5.attrs.get("resample_opt", None)
        if fr is not None and int(fr) != MEG_RATE:
            print(
                f"ERROR: HDF5 frame_rate={fr} ≠ requested MEG rate "
                f"{MEG_RATE}. Delete {hdf5_path} or pass "
                f"--force_recompute_features.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"Reusing HPSN features at {hdf5_path} "
              f"(frame_rate={fr}, resample_opt={ro})")

    h5_file = h5py.File(hdf5_path, "r")

    total = len(subjects) * len(sessions)
    log_stage(
        f"Stage 4/4: Fitting {len(subjects)} × {len(sessions)} = {total} "
        f"(subject, session) combos"
    )
    status = {"done": 0, "cached": 0, "no_runs": 0}
    try:
        for i, subj in enumerate(subjects):
            for j, ses in enumerate(sessions):
                idx = i * len(sessions) + j + 1
                log_stage(f"[{idx}/{total}] sub-{subj} ses-{ses}")
                s = preflight_one_subject_session(
                    args, subj, ses,
                    cfg=cfg,
                    h5_file=h5_file, results_dir=results_dir,
                    source_data_dir=source_data_dir,
                    meg_rate=MEG_RATE, resample_opt=RESAMPLE_OPT,
                )
                status[s] += 1
    finally:
        h5_file.close()

    log_stage("Done")
    print(f"  fit              : {status['done']}")
    print(f"  cached (skipped) : {status['cached']}")
    print(f"  no runs found    : {status['no_runs']}")


if __name__ == "__main__":
    main()
