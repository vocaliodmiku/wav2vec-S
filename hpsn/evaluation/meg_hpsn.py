"""HPSN preflight ridge-TRF encoding (E1 = hpsn_l1, E2 = hpsn_l2, E3 = hpsn_l3).

Thin orchestrator around the existing HPSN evaluation machinery in
``hpsn_ridge.py``. For each (subject, session) in the requested sweep,
fits ridge encoding for ALL three conditions E1, E2, E3 in one pass,
reusing the HPSN feature HDF5 and MEG cache. Emits one "ladder" pkl per
(subj, ses) holding all conditions side-by-side, so you can eyeball
HPSN-Level-{1,2,3} encoding performance before running the full ablation
grid.

See ``minimal-2.md`` §9.3 for the ladder naming (E1 / E2 / E3).

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
import dataclasses
import datetime
import json
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
from .hpsn_ridge import (
    BIDS_ROOT,
    DEFAULT_TRF_TMAX,
    DEFAULT_TRF_TMIN,
    PCA_COMPONENTS,
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
    parse_events_tsv,
)
from .baseline_meg_masc import PREPROCESSING_TAG, load_sensor_run


# Preflight condition map — minimal-2.md §9.3 naming, extended for the
# 3-level full model:
#   E1 = HPSN Level 1 representation  (hpsn_l1)
#   E2 = HPSN Level 2 representation  (hpsn_l2)
#   E3 = HPSN Level 3 representation  (hpsn_l3)
PREFLIGHT_CONDS = {
    "E1": "hpsn_l1",
    "E2": "hpsn_l2",
    "E3": "hpsn_l3",
    # Concat readout — L1⊕L2⊕L3 (synthesized in build_condition_matrix from
    # the three per-level groups in the same H5; no re-extraction needed).
    "Econcat": "hpsn_concat",
}

MEG_MASC_SESSIONS = (0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model-size presets
# ─────────────────────────────────────────────────────────────────────────────
#
# Three-level HPSN: each level has its own contiguous layer band tapped
# from the frozen wav2vec-S backbone. ``base`` mirrors the run.sh launch
# config (L1=1-4, L2=5-8, L3=9-12). ``large`` keeps the legacy 8-layer
# acoustic/lexical bands; the semantic band is left unset and must be
# supplied via --tap_semantic_start/--tap_semantic_end if you ever evaluate
# a Large checkpoint.
#
# Fields left as None are filled from the HPSNConfig dataclass defaults
# (or, for tap_semantic_* under "large", flagged as a hard error in
# build_hpsn_config until the user provides them).

MODEL_SIZE_PRESETS = {
    "large": {
        "hidden_dim": 1024,
        "tap_acoustic_start": 1, "tap_acoustic_end": 8,
        "tap_lexical_start": 13, "tap_lexical_end": 20,
        "tap_semantic_start": None, "tap_semantic_end": None,
        "backbone_default": "biaofu-xmu/wav2vec-S-Large-ft-960h",
    },
    "base": {
        "hidden_dim": 768,
        "tap_acoustic_start": 1, "tap_acoustic_end": 4,
        "tap_lexical_start": 5, "tap_lexical_end": 8,
        "tap_semantic_start": 9, "tap_semantic_end": 12,
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
        "--config_json", default=None,
        help="Path to the training-time config.json (written by train.py). "
             "Defaults to <ckpt_dir_parent>/config.json if it exists. "
             "Pass an empty string to disable auto-loading and rely solely on "
             "CLI flags.",
    )
    p.add_argument(
        "--model_size", choices=("base", "large"), default="large",
        help="Preset for backbone size. 'base' → hidden_dim=768, taps "
             "1-4 / 5-8 / 9-12 (matches run.sh full-model training). "
             "'large' → hidden_dim=1024, taps 1-8 / 13-20 / (semantic unset; "
             "supply via --tap_semantic_start/_end). Individual --hidden_dim "
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
    p.add_argument("--tap_acoustic_start", type=int, default=None,
                   help="L1 tap band start (inclusive, 1-indexed).")
    p.add_argument("--tap_acoustic_end", type=int, default=None,
                   help="L1 tap band end (inclusive).")
    p.add_argument("--tap_lexical_start", type=int, default=None,
                   help="L2 tap band start (inclusive). Must be > tap_acoustic_end.")
    p.add_argument("--tap_lexical_end", type=int, default=None,
                   help="L2 tap band end (inclusive).")
    p.add_argument("--tap_semantic_start", type=int, default=None,
                   help="L3 tap band start (inclusive). Must be > tap_lexical_end.")
    p.add_argument("--tap_semantic_end", type=int, default=None,
                   help="L3 tap band end (inclusive).")
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


def _resolve_config_json_path(args) -> Path | None:
    """Decide where to read the training-time config.json from.

    * ``--config_json <path>`` → use that path verbatim (must exist).
    * ``--config_json ""``     → auto-load disabled.
    * unset                    → look for ``<ckpt_dir>/config.json`` and the
                                  parent directory (since accelerate
                                  ``checkpoint-N/`` lives one level below the
                                  output_dir where train.py writes config.json).
    """
    if args.config_json == "":
        return None
    if args.config_json is not None:
        p = Path(args.config_json)
        if not p.is_file():
            print(f"ERROR: --config_json {args.config_json!r} not a file",
                  file=sys.stderr)
            sys.exit(2)
        return p
    ckpt = Path(args.ckpt)
    candidates = [
        ckpt / "config.json",
        ckpt.parent / "config.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_config_json(path: Path) -> HPSNConfig:
    """Hydrate an HPSNConfig from the training-time config.json.

    Tolerates extra keys (older or newer fields), tuples written as JSON
    arrays, and missing fields (the dataclass default fills in)."""
    with path.open() as f:
        data = json.load(f)
    field_names = {f.name for f in dataclasses.fields(HPSNConfig)}
    kept = {}
    skipped = []
    for k, v in data.items():
        if k not in field_names:
            skipped.append(k)
            continue
        if isinstance(v, list):
            v = tuple(v)
        kept[k] = v
    cfg = HPSNConfig(**kept)
    if skipped:
        print(f"[config] WARN: ignoring unknown config fields: {sorted(skipped)}")
    return cfg


def build_hpsn_config(args) -> HPSNConfig:
    """Assemble an HPSNConfig from the model-size preset and any per-field overrides.

    If a training-time ``config.json`` is reachable (auto-detected next to
    --ckpt or supplied via --config_json), its values take precedence: this
    is the only way to recover v2 fields like ``use_v2_loss`` /
    ``use_span_masking`` / ``n_iterations`` / ``level{N}_frozen_tap`` reliably.
    CLI ``--tap_*`` and ``--hidden_dim`` flags can still override individual
    fields after JSON load.
    """
    cfg_json_path = _resolve_config_json_path(args)
    if cfg_json_path is not None:
        cfg = _load_config_json(cfg_json_path)
        print(f"[config] loaded {cfg_json_path}")
        if args.backbone_model is not None:
            cfg.backbone_model = args.backbone_model
        if args.hidden_dim is not None:
            cfg.hidden_dim = args.hidden_dim

        # Allow per-tap CLI overrides on top of the JSON config (rare).
        def _override_band(band: str, attr_layers: str):
            s = getattr(args, f"tap_{band}_start", None)
            e = getattr(args, f"tap_{band}_end", None)
            if s is not None and e is not None:
                setattr(cfg, attr_layers, tuple(range(s, e + 1)))

        _override_band("acoustic", "level1_tap_layers")
        _override_band("lexical", "level2_tap_layers")
        _override_band("semantic", "level3_tap_layers")

        # Recover start/end attrs for downstream metadata (matches the CLI path).
        cfg.tap_acoustic_start = cfg.level1_tap_layers[0]
        cfg.tap_acoustic_end = cfg.level1_tap_layers[-1]
        cfg.tap_lexical_start = cfg.level2_tap_layers[0]
        cfg.tap_lexical_end = cfg.level2_tap_layers[-1]
        cfg.tap_semantic_start = cfg.level3_tap_layers[0]
        cfg.tap_semantic_end = cfg.level3_tap_layers[-1]
        return cfg

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

    def _resolve(name):
        cli = getattr(args, name)
        return cli if cli is not None else preset[name]

    bands = {
        "acoustic": (_resolve("tap_acoustic_start"), _resolve("tap_acoustic_end")),
        "lexical":  (_resolve("tap_lexical_start"),  _resolve("tap_lexical_end")),
        "semantic": (_resolve("tap_semantic_start"), _resolve("tap_semantic_end")),
    }
    for band, (s, e) in bands.items():
        if s is None or e is None:
            print(
                f"ERROR: tap_{band}_start/end is unset. The {args.model_size} preset "
                f"does not provide a default for the {band} (L3) band — supply "
                f"--tap_{band}_start and --tap_{band}_end explicitly.",
                file=sys.stderr,
            )
            sys.exit(2)
        if s > e:
            print(
                f"ERROR: tap_{band}_start ({s}) > tap_{band}_end ({e}).",
                file=sys.stderr,
            )
            sys.exit(2)

    ac_s, ac_e = bands["acoustic"]
    lx_s, lx_e = bands["lexical"]
    sm_s, sm_e = bands["semantic"]
    if ac_e >= lx_s:
        print(
            f"ERROR: tap_acoustic_end ({ac_e}) must be strictly less than "
            f"tap_lexical_start ({lx_s}).",
            file=sys.stderr,
        )
        sys.exit(2)
    if lx_e >= sm_s:
        print(
            f"ERROR: tap_lexical_end ({lx_e}) must be strictly less than "
            f"tap_semantic_start ({sm_s}).",
            file=sys.stderr,
        )
        sys.exit(2)

    cfg.level1_tap_layers = tuple(range(ac_s, ac_e + 1))
    cfg.level2_tap_layers = tuple(range(lx_s, lx_e + 1))
    cfg.level3_tap_layers = tuple(range(sm_s, sm_e + 1))

    # Preserve the start/end fields on the config object for metadata logging.
    # These attributes don't exist on HPSNConfig itself (the dataclass uses
    # tuples) but Python will accept them via __dict__ as long as the
    # dataclass isn't frozen, which it isn't.
    cfg.tap_acoustic_start, cfg.tap_acoustic_end = ac_s, ac_e
    cfg.tap_lexical_start,  cfg.tap_lexical_end  = lx_s, lx_e
    cfg.tap_semantic_start, cfg.tap_semantic_end = sm_s, sm_e
    return cfg


def check_checkpoint_matches_config(ckpt_path, cfg: HPSNConfig) -> None:
    """Read the ckpt state_dict and fail loudly on shape mismatch before we try to load.

    Keys for the 3-level model:
      - tap1.weights, tap2.weights, tap3.weights   (LayerTap learnable mixes)
      - level{1,2,3}.input_proj.weight             [lstm_dim, hidden_dim]
      - level{1,2,3}.recon_head.weight             [hidden_dim, lstm_dim]
    """
    sd_file = _find_state_dict_file(Path(ckpt_path))
    sd = _load_state_dict(sd_file)

    def _shape(key):
        t = sd.get(key, None)
        return tuple(t.shape) if t is not None else None

    tap1 = _shape("tap1.weights")
    tap2 = _shape("tap2.weights")
    tap3 = _shape("tap3.weights")
    l1_in    = _shape("level1.input_proj.weight")     # [lstm_dim, hidden_dim]
    # v1 recon_heads are absent in v2 checkpoints (use_v2_loss=True drops them).
    l1_recon = _shape("level1.recon_head.weight")
    l2_recon = _shape("level2.recon_head.weight")
    l3_recon = _shape("level3.recon_head.weight")
    is_v2 = bool(getattr(cfg, "use_v2_loss", False))

    expected_l1 = cfg.tap_acoustic_end - cfg.tap_acoustic_start + 1
    expected_l2 = cfg.tap_lexical_end  - cfg.tap_lexical_start  + 1
    expected_l3 = cfg.tap_semantic_end - cfg.tap_semantic_start + 1

    problems = []
    if tap1 is not None and tap1[0] != expected_l1:
        problems.append(
            f"  tap1 length: checkpoint={tap1[0]}, configured={expected_l1} "
            f"(range {cfg.tap_acoustic_start}-{cfg.tap_acoustic_end})"
        )
    if tap2 is not None and tap2[0] != expected_l2:
        problems.append(
            f"  tap2 length: checkpoint={tap2[0]}, configured={expected_l2} "
            f"(range {cfg.tap_lexical_start}-{cfg.tap_lexical_end})"
        )
    if tap3 is not None and tap3[0] != expected_l3:
        problems.append(
            f"  tap3 length: checkpoint={tap3[0]}, configured={expected_l3} "
            f"(range {cfg.tap_semantic_start}-{cfg.tap_semantic_end})"
        )
    if l1_in is not None and l1_in[1] != cfg.hidden_dim:
        problems.append(
            f"  hidden_dim: checkpoint={l1_in[1]}, configured={cfg.hidden_dim}"
        )
    if not is_v2:
        for name, sh in (("level1", l1_recon), ("level2", l2_recon), ("level3", l3_recon)):
            if sh is not None and sh[0] != cfg.hidden_dim:
                problems.append(
                    f"  {name}.recon_head hidden: checkpoint={sh[0]}, "
                    f"configured={cfg.hidden_dim}"
                )

    # Hard error if the 3-level checkpoint is missing L3 keys entirely
    # (i.e. someone pointed --ckpt at a legacy 2-level checkpoint).
    legacy_keys = ("tap_acoustic.weights", "tap_lexical.weights")
    if any(k in sd for k in legacy_keys) and tap3 is None:
        problems.append(
            "  This appears to be a legacy 2-level (HPSNMinimal) checkpoint "
            "(found tap_acoustic / tap_lexical, no tap3). Re-train with the "
            "3-level model in run.sh, or point --ckpt at a full-model checkpoint."
        )

    if problems:
        hint_base = (
            "try --model_size base "
            "(presets hidden_dim=768, taps 1-4 / 5-8 / 9-12)."
        )
        hint_overrides = (
            "You can also override per-field: --hidden_dim, "
            "--tap_acoustic_start/end, --tap_lexical_start/end, "
            "--tap_semantic_start/end."
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
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized",
              random_state=0)
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
            tap_semantic=(cfg.tap_semantic_start, cfg.tap_semantic_end),
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
    print(f"Conditions    : E1→hpsn_l1, E2→hpsn_l2, E3→hpsn_l3")
    print(f"Model size    : {args.model_size}")
    print(f"Backbone      : {cfg.backbone_model}")
    print(f"Hidden dim    : {cfg.hidden_dim}")
    print(f"Tap acoustic  : {cfg.tap_acoustic_start}-{cfg.tap_acoustic_end} "
          f"(len {cfg.tap_acoustic_end - cfg.tap_acoustic_start + 1})")
    print(f"Tap lexical   : {cfg.tap_lexical_start}-{cfg.tap_lexical_end} "
          f"(len {cfg.tap_lexical_end - cfg.tap_lexical_start + 1})")
    print(f"Tap semantic  : {cfg.tap_semantic_start}-{cfg.tap_semantic_end} "
          f"(len {cfg.tap_semantic_end - cfg.tap_semantic_start + 1})")
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
