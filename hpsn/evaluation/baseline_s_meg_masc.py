"""Baseline MEG-MASC ridge-TRF encoding for the wav2vec-S block-causal model.

Sibling of ``baseline_meg_masc.py``. Same Défossez preprocessing, same ridge-
TRF fit, same HDF5 / pkl conventions — only the feature extractor and cache
key change. Saves into a separate results dir so wav2vec-S pkls cannot collide
with wav2vec2 pkls during aggregation.

Wav2vec-S (Fu et al., 2024; default checkpoint ``biaofu-xmu/wav2vec-S-Base``)
is a block-causal variant of wav2vec2: self-attention is restricted to a
per-block window with ``main_context`` (M) frames per block and
``right_context`` (R) frames of lookahead. Both M and R are runtime-tunable
via ``model.encoder.{main_context,right_context}``; we expose them as CLI
flags so the cache and pkl filenames record which (M, R) was used.

Two-phase CLI mirrors ``baseline_meg_masc``:
    --compute-feature-only   build the per-stim HDF5 cache and exit
    --fitting-only           assume the cache exists; only fit ridge
    (neither)                run both stages back-to-back

Output naming
-------------
Feature label / pkl tag :  ``w2vS_M{M}R{R}_l{layer_idx}``
HDF5 cache              :  ``baseline_features_w2vS_M{M}R{R}_{REPR|MEG}.h5``
Results dir (default)   :  ``.../meg_results_defossez_wav2vS``

Usage
-----
    # 1. precompute features once (GPU node):
    python -m hpsn.evaluation.baseline_s_meg_masc \\
        --subj 01 --ses 0 --feat w2v2 --resample_opt REPR \\
        --w2v2_model biaofu-xmu/wav2vec-S-Base \\
        --main_context 8 --right_context 2 \\
        --compute-feature-only --device cuda

    # 2. fit ridge per subject (slurm array, no GPU needed):
    python -m hpsn.evaluation.baseline_s_meg_masc \\
        --subj $SUBJ --ses all --feat w2v2 --resample_opt REPR \\
        --w2v2_model biaofu-xmu/wav2vec-S-Base \\
        --main_context 8 --right_context 2 \\
        --w2v2_layers all --fitting-only
"""
from __future__ import annotations

import argparse
import datetime
import os
import pickle
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Make the local wav2vec-S package importable. Layout assumed:
#     /home/.../wav2vec-L/hpsn/evaluation/baseline_s_meg_masc.py   ← this file
#     /home/.../wav2vec-L/wav2vec-S-hf/wav2vec_s/                  ← target
_REPO_ROOT = Path(__file__).resolve().parents[2]
_W2V_S_PKG_PARENT = _REPO_ROOT / "wav2vec-S-hf"
if _W2V_S_PKG_PARENT.is_dir() and str(_W2V_S_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_W2V_S_PKG_PARENT))

# Re-use everything that does not depend on the model architecture.
from .baseline_meg_masc import (
    NATIVE_W2V2_RATE,
    PREPROCESSING_TAG,
    _build_X_from_runs,
    _linear_upsample,
    _load_runs_with_meta,
    _w2v2_layer_key,
    _w2v2_n_taps_from_h5,
    _w2v2_required_keys,
    build_feat_matrix_from_hdf5,
    expand_w2v2_layers,
    load_wav,
    log_stage,
    parse_w2v2_layers,
    resolve_sessions,
    resolve_subjects,
)
from .hpsn_ridge import (
    BIDS_ROOT,
    DEFAULT_TRF_TMAX,
    DEFAULT_TRF_TMIN,
    PCA_COMPONENTS,
    build_lagged_matrix,
    build_single_lag_matrix,
    collect_unique_stimuli,
    fit_ridge_cv_full_lags,
    fit_ridge_cv_single_lag,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

W2V_S_DEFAULT_MODEL = "biaofu-xmu/wav2vec-S-Base"
W2V_S_DEFAULT_MAIN_CONTEXT = 8
W2V_S_DEFAULT_RIGHT_CONTEXT = 2

RESULTS_DIR = Path(
    "/scratch/jsm04005/fie24002/DATA/HPSN/EvalResults/meg_results_defossez_wav2vS"
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Défossez-style MEG-MASC baseline (ridge-TRF encoding) using the "
            "block-causal wav2vec-S model."
        ),
    )
    p.add_argument(
        "--subj", required=True,
        help="Subject ID(s): comma-separated list or 'all' "
             "(e.g., '01', '01,02,03', 'all').",
    )
    p.add_argument(
        "--ses", required=True,
        help="Session(s): comma-separated list or 'all' (e.g., '0', '0,1', 'all').",
    )
    p.add_argument(
        "--feat", default="w2v2", choices=("w2v2",),
        help="Kept for CLI symmetry with baseline_meg_masc; only 'w2v2' "
             "(meaning wav2vec-S features) is supported here.",
    )
    p.add_argument(
        "--w2v2_model", default=W2V_S_DEFAULT_MODEL,
        help=f"HuggingFace id or local path for the wav2vec-S checkpoint "
             f"(default: {W2V_S_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--main_context", type=int, default=W2V_S_DEFAULT_MAIN_CONTEXT,
        help="Frames per block (M). Default 8 (= 160 ms at 50 Hz).",
    )
    p.add_argument(
        "--right_context", type=int, default=W2V_S_DEFAULT_RIGHT_CONTEXT,
        help="Lookahead frames per block (R). Default 2 (= 40 ms at 50 Hz). "
             "Constraint: R <= M / 2.",
    )
    p.add_argument(
        "--w2v2_layers", default="all",
        help=(
            "Transformer taps to fit. 'all' = layers 0..n_taps-1 "
            "(0 = post-CNN/pre-transformer, 1..N = transformer outputs). "
            "Or a comma-separated list, e.g. '0,4,8,12'."
        ),
    )
    p.add_argument(
        "--resample_opt", choices=("MEG", "REPR"), default="REPR",
        help="MEG = 50 Hz, REPR = 100 Hz (w2v-S features are ×2 upsampled to match).",
    )
    p.add_argument("--trf_tmin", type=float, default=DEFAULT_TRF_TMIN)
    p.add_argument("--trf_tmax", type=float, default=DEFAULT_TRF_TMAX)
    p.add_argument("--per_lag", action="store_true")
    p.add_argument(
        "--lag_list_ms", default="0,40,80,120,160,200,300,400,500",
        help="Comma-separated lags (ms) used when --per_lag is set.",
    )
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--hdf5_path", default=None,
        help="Override feature HDF5 location.",
    )
    p.add_argument("--force_recompute_features", action="store_true")
    p.add_argument(
        "--force_refit", action="store_true",
        help="Refit ridge even if the output pkl already exists.",
    )
    p.add_argument("--device", default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--compute-feature-only", action="store_true", dest="compute_feature_only",
        help="Build the feature cache and exit.",
    )
    mode.add_argument(
        "--fitting-only", action="store_true", dest="fitting_only",
        help="Skip feature extraction; assume the cache is already built.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# wav2vec-S feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class W2VSExtractor:
    """Multi-layer wav2vec-S feature extractor.

    A single forward pass returns ``n_taps = num_hidden_layers + 1`` hidden
    states (layer 0 = post-CNN/pre-transformer, layers 1..N = transformer
    outputs). For wav2vec-S-Base that's 12 + 1 = 13 taps at 50 Hz.

    When ``right_context > 0``, the wav2vec-S encoder appends ``block_num * R``
    right-context frames to its time axis before running attention, and only
    trims the *final* ``last_hidden_state`` back. Intermediate hidden states
    therefore arrive padded; we trim each to the true frame count
    ``T = last_hidden_state.shape[1]`` here.
    """

    def __init__(
        self,
        model_name: str = W2V_S_DEFAULT_MODEL,
        main_context: int = W2V_S_DEFAULT_MAIN_CONTEXT,
        right_context: int = W2V_S_DEFAULT_RIGHT_CONTEXT,
        device: torch.device | str | None = None,
    ):
        # Imported lazily so that ``--fitting-only`` runs (which never need the
        # extractor) don't pay the import cost or require the package to be on
        # path.
        from wav2vec_s.modeling_wav2vec_s import Wav2VecSModel

        if right_context > main_context // 2:
            raise ValueError(
                f"right_context={right_context} must satisfy R <= M/2 "
                f"(got M={main_context})."
            )

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.main_context = int(main_context)
        self.right_context = int(right_context)

        # ``use_safetensors=True`` mirrors baseline_meg_masc — bypasses the
        # transformers CVE-2025-32434 torch.load gate. If the checkpoint only
        # ships pytorch_model.bin, drop this flag in your local checkout.
        try:
            self.model = Wav2VecSModel.from_pretrained(
                model_name, use_safetensors=True,
            )
        except (OSError, ValueError):
            # Fall back if the repo has no safetensors weights.
            self.model = Wav2VecSModel.from_pretrained(model_name)

        # The (M, R) baked into the released config may differ from what we
        # want at inference (e.g., released as M=8/R=2 but you want R=0).
        # Override after load — encoder reads these directly each forward.
        self.model.encoder.main_context = self.main_context
        self.model.encoder.right_context = self.right_context

        self.n_taps = int(self.model.config.num_hidden_layers) + 1
        self.model = self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_all(self, wav: np.ndarray) -> list[np.ndarray]:
        """Return all ``n_taps`` hidden states as ``[T, H]`` float32 arrays
        at the native 50 Hz wav2vec2 frame rate."""
        x = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        out = self.model(x, output_hidden_states=True, return_dict=True)
        # last_hidden_state is already trimmed to the un-padded length T;
        # intermediate hidden_states are not when right_context > 0.
        T = int(out.last_hidden_state.shape[1])
        return [
            hs[:, :T, :].squeeze(0).cpu().numpy().astype(np.float32)
            for hs in out.hidden_states
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Feature HDF5 cache
# ─────────────────────────────────────────────────────────────────────────────

def _validate_w2vS_h5_attrs(
    h5: h5py.File, frame_rate: int, model_name: str,
    main_context: int, right_context: int, hdf5_path: Path,
) -> None:
    """Hard-fail if HDF5 attrs disagree with the requested run."""
    fr_stored = h5.attrs.get("frame_rate", None)
    if fr_stored is not None and int(fr_stored) != frame_rate:
        print(
            f"HDF5 frame_rate {fr_stored} ≠ requested {frame_rate}. "
            f"Pass --force_recompute_features or delete {hdf5_path}.",
            file=sys.stderr,
        )
        sys.exit(2)
    model_stored = h5.attrs.get("w2v_s_model", None)
    if model_stored is not None and str(model_stored) != model_name:
        print(
            f"HDF5 w2v_s_model '{model_stored}' ≠ requested '{model_name}'. "
            f"Pass --force_recompute_features or delete {hdf5_path}.",
            file=sys.stderr,
        )
        sys.exit(2)
    mc_stored = h5.attrs.get("main_context", None)
    if mc_stored is not None and int(mc_stored) != main_context:
        print(
            f"HDF5 main_context={mc_stored} ≠ requested {main_context}. "
            f"Use a different cache file or delete {hdf5_path}.",
            file=sys.stderr,
        )
        sys.exit(2)
    rc_stored = h5.attrs.get("right_context", None)
    if rc_stored is not None and int(rc_stored) != right_context:
        print(
            f"HDF5 right_context={rc_stored} ≠ requested {right_context}. "
            f"Use a different cache file or delete {hdf5_path}.",
            file=sys.stderr,
        )
        sys.exit(2)


def build_feature_hdf5(
    stim_map: dict, frame_rate: int, hdf5_path: Path, force: bool,
    device: torch.device | str | None,
    *,
    model_name: str,
    main_context: int,
    right_context: int,
):
    """Write per-stim wav2vec-S features to ``hdf5_path``.

    Stores one ``[stim_id]/layer_NN`` dataset per layer NN ∈ [00..n_taps-1]
    per stim, where ``n_taps`` is read from the model config (base = 13).
    """
    extractor = W2VSExtractor(
        model_name=model_name,
        main_context=main_context, right_context=right_context,
        device=device,
    )
    n_taps = extractor.n_taps
    print(
        f"  wav2vec-S model has {n_taps} taps "
        f"(layer 0..{n_taps - 1}; num_hidden_layers={n_taps - 1}); "
        f"M={main_context}, R={right_context}"
    )
    required_keys = _w2v2_required_keys(n_taps)

    if hdf5_path.exists() and not force:
        with h5py.File(hdf5_path, "r") as h5:
            _validate_w2vS_h5_attrs(
                h5, frame_rate, model_name, main_context, right_context,
                hdf5_path,
            )
            stored_n_taps = _w2v2_n_taps_from_h5(h5)
            if stored_n_taps and stored_n_taps != n_taps:
                print(
                    f"HDF5 has {stored_n_taps} taps but the model "
                    f"({model_name}) has {n_taps}. "
                    f"Pass --force_recompute_features or delete {hdf5_path}.",
                    file=sys.stderr,
                )
                sys.exit(2)
            missing = [
                k for k in stim_map
                if (k not in h5)
                or any(rk not in h5[k] for rk in required_keys)
            ]
        if not missing:
            print(f"Features up-to-date at {hdf5_path}")
            return
        print(f"Appending {len(missing)} new stims to {hdf5_path}")
        items = [(k, stim_map[k]) for k in missing]
        mode = "a"
    else:
        if hdf5_path.exists():
            os.remove(hdf5_path)
        items = sorted(stim_map.items())
        mode = "w"

    with h5py.File(hdf5_path, mode) as h5:
        h5.attrs["feat_type"] = "w2v_s"
        h5.attrs["frame_rate"] = frame_rate
        h5.attrs["w2v_s_model"] = model_name
        h5.attrs["main_context"] = int(main_context)
        h5.attrs["right_context"] = int(right_context)
        h5.attrs["w2v2_layers_present"] = list(range(n_taps))

        for stim_id, wav_path in tqdm(items, desc="w2v-S features"):
            wav = load_wav(wav_path)
            if stim_id in h5:
                grp = h5[stim_id]
            else:
                grp = h5.create_group(stim_id)
                grp.attrs["wav_path"] = str(wav_path)

            hidden_states = extractor.extract_all(wav)
            assert len(hidden_states) == n_taps, (
                f"expected {n_taps} hidden states, got {len(hidden_states)}"
            )
            for layer_idx, arr in enumerate(hidden_states):
                if frame_rate == 100:
                    arr = _linear_upsample(arr, 2)
                elif frame_rate != NATIVE_W2V2_RATE:
                    raise ValueError(
                        f"w2v-S frame_rate must be 50 or 100 Hz, got {frame_rate}"
                    )
                key = _w2v2_layer_key(layer_idx)
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=arr, compression="gzip")


# ─────────────────────────────────────────────────────────────────────────────
# Per-(subject, session) pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _feature_label(layer_idx: int, main_context: int, right_context: int) -> str:
    """Canonical feature_tag used in pkl filenames and report aggregation."""
    return f"w2vS_M{main_context}R{right_context}_l{layer_idx}"


def _fit_one_condition(
    args, *, Y_all, X_all, feat_label, RESAMPLE_OPT, MEG_RATE,
    subj, ses, save_path, channel_meta,
) -> None:
    """PCA → ridge → save pkl for a single (subj, ses, feat_label) combo."""
    n_sensors, T_total = Y_all.shape
    print(
        f"  Concat ({feat_label}): T={T_total}, n_sensors={n_sensors}, "
        f"D={X_all.shape[1]}"
    )

    if X_all.shape[1] > PCA_COMPONENTS:
        log_stage(f"sub-{subj} ses-{ses} {feat_label}: PCA → {PCA_COMPONENTS}")
        active_mask = np.abs(X_all).sum(1) > 0
        pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized",
                  random_state=0)
        pca.fit(X_all[active_mask])
        X_all = pca.transform(X_all).astype(np.float64)
        print(f"  PCA on {feat_label}: {pca.explained_variance_ratio_.sum():.2%} var")

    log_stage(f"sub-{subj} ses-{ses} {feat_label}: Ridge fit")

    if args.per_lag:
        lag_list_ms = [float(s) for s in args.lag_list_ms.split(",") if s.strip()]
        print(f"  Per-lag mode: {len(lag_list_ms)} lags → {lag_list_ms}")
        r_by_lag = np.zeros((len(lag_list_ms), n_sensors), dtype=np.float64)
        for i, lag_ms in enumerate(lag_list_ms):
            X_lag, lag_frames = build_single_lag_matrix(X_all, MEG_RATE, lag_ms)
            print(f"    lag {lag_ms:+6.1f} ms ({lag_frames:+d} frames) ...")
            r_by_lag[i] = fit_ridge_cv_single_lag(X_lag, Y_all)
            print(f"      mean r = {r_by_lag[i].mean():.4f}")
        result = dict(
            mode="per_lag",
            r_by_lag=r_by_lag,
            lag_values_ms=np.array(lag_list_ms, dtype=np.float64),
            feat=feat_label, space="sensor", resample_opt=RESAMPLE_OPT,
            subject=subj, session=ses,
            frame_rate=MEG_RATE, n_predictors=X_all.shape[1],
            preprocessing=PREPROCESSING_TAG,
            main_context=int(args.main_context),
            right_context=int(args.right_context),
        )
    else:
        X_lagged, lags = build_lagged_matrix(
            X_all, MEG_RATE, args.trf_tmin, args.trf_tmax,
        )
        print(
            f"  Lagged matrix: {X_lagged.shape}  "
            f"({X_all.shape[1]} feats × {len(lags)} lags)"
        )
        r_values, coef_full, fold_alphas, full_alpha = fit_ridge_cv_full_lags(
            X_lagged, Y_all,
        )
        result = dict(
            mode="full_lags",
            r=r_values,
            coef=coef_full.reshape(n_sensors, len(lags), X_all.shape[1]),
            lags=lags,
            trf_tmin=args.trf_tmin, trf_tmax=args.trf_tmax,
            fold_alphas=fold_alphas, full_alpha=full_alpha,
            feat=feat_label, space="sensor", resample_opt=RESAMPLE_OPT,
            subject=subj, session=ses,
            frame_rate=MEG_RATE, n_predictors=X_all.shape[1],
            preprocessing=PREPROCESSING_TAG,
            main_context=int(args.main_context),
            right_context=int(args.right_context),
        )
        print(f"    mean r = {r_values.mean():.4f}   max r = {r_values.max():.4f}")

    if channel_meta is not None:
        result["ch_names"] = channel_meta["ch_names"]
        result["ch_types"] = channel_meta["ch_types"]
        result["ch_positions"] = channel_meta["ch_positions"]

    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved → {save_path}")


def fit_one_subject_session(
    args, subj: str, ses: int, *,
    h5f, MEG_RATE: int, RESAMPLE_OPT: str,
    results_dir: Path,
    w2v2_layers: list[int],
) -> dict[str, int]:
    """Run the MEG-align → concat → PCA → ridge → save pipeline.

    Loops over every requested layer (each producing its own pkl); MEG runs
    are loaded once per (subj, ses) and reused across layers.
    """
    save_dir = results_dir / f"trf_ridge_sub-{subj}"
    save_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "cached": 0, "no_runs": 0, "skip": 0}

    conditions = [
        (
            _feature_label(layer_idx, args.main_context, args.right_context),
            _w2v2_layer_key(layer_idx),
        )
        for layer_idx in w2v2_layers
    ]

    needs_fit = []
    for feat_label, dataset_name in conditions:
        save_name = (
            f"trf_ridge_{feat_label}_sensor_{RESAMPLE_OPT}_"
            f"sub-{subj}_ses-{ses}.pkl"
        )
        save_path = save_dir / save_name
        if save_path.exists() and not args.force_refit:
            log_stage(
                f"sub-{subj} ses-{ses}: cached → {save_name} "
                f"(pass --force_refit to redo)"
            )
            counts["cached"] += 1
        else:
            needs_fit.append((feat_label, dataset_name, save_path))

    if not needs_fit:
        return counts

    log_stage(f"sub-{subj} ses-{ses}: loading MEG runs")
    meg_cache_dir = results_dir / f"meg_cache_defossez_sub-{subj}"
    runs, channel_meta = _load_runs_with_meta(subj, ses, MEG_RATE, meg_cache_dir)
    if not runs:
        log_stage(f"sub-{subj} ses-{ses}: no runs — skipping")
        counts["no_runs"] += len(needs_fit)
        return counts

    for feat_label, dataset_name, save_path in needs_fit:
        log_stage(f"sub-{subj} ses-{ses} {feat_label}: building X")
        built = _build_X_from_runs(
            runs, FEAT="w2v2", MEG_RATE=MEG_RATE,
            h5f=h5f, acoustic_feats=None,
            dataset_name=dataset_name,
        )
        if built is None:
            log_stage(f"sub-{subj} ses-{ses} {feat_label}: empty X — skipping")
            counts["skip"] += 1
            continue
        Y_all, X_all = built
        _fit_one_condition(
            args, Y_all=Y_all, X_all=X_all,
            feat_label=feat_label, RESAMPLE_OPT=RESAMPLE_OPT,
            MEG_RATE=MEG_RATE, subj=subj, ses=ses,
            save_path=save_path, channel_meta=channel_meta,
        )
        counts["done"] += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_hdf5_path(
    args, results_dir: Path, RESAMPLE_OPT: str,
) -> Path:
    if args.hdf5_path:
        return Path(args.hdf5_path)
    tag = f"w2vS_M{args.main_context}R{args.right_context}"
    return results_dir / f"baseline_features_{tag}_{RESAMPLE_OPT}.h5"


def run_compute_features(
    args, *, MEG_RATE: int, hdf5_path: Path, stim_map: dict, device,
) -> None:
    log_stage("Stage feature: building HDF5 features (wav2vec-S)")
    build_feature_hdf5(
        stim_map,
        frame_rate=MEG_RATE,
        hdf5_path=hdf5_path,
        force=args.force_recompute_features,
        device=device,
        model_name=args.w2v2_model,
        main_context=args.main_context,
        right_context=args.right_context,
    )


def load_existing_features(args, *, MEG_RATE: int, hdf5_path: Path) -> None:
    """For --fitting-only: open the cache read-only and validate."""
    if not hdf5_path.exists():
        print(
            f"--fitting-only: HDF5 cache missing at {hdf5_path}. "
            f"Run --compute-feature-only first.",
            file=sys.stderr,
        )
        sys.exit(2)
    with h5py.File(hdf5_path, "r") as h5:
        _validate_w2vS_h5_attrs(
            h5, MEG_RATE, args.w2v2_model,
            args.main_context, args.right_context, hdf5_path,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    RESAMPLE_OPT = args.resample_opt
    MEG_RATE = 100 if RESAMPLE_OPT == "REPR" else 50

    if args.right_context > args.main_context // 2:
        print(
            f"ERROR: --right_context ({args.right_context}) must satisfy "
            f"R <= M/2 (M = --main_context = {args.main_context}).",
            file=sys.stderr,
        )
        sys.exit(2)

    w2v2_layers = parse_w2v2_layers(args.w2v2_layers)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = _resolve_hdf5_path(args, results_dir, RESAMPLE_OPT)

    subjects = resolve_subjects(args.subj)
    sessions = resolve_sessions(args.ses)
    if not subjects:
        print("ERROR: no subjects resolved from --subj.", file=sys.stderr)
        sys.exit(2)
    if not sessions:
        print("ERROR: no sessions resolved from --ses.", file=sys.stderr)
        sys.exit(2)

    if args.compute_feature_only:
        run_mode = "compute-feature-only"
    elif args.fitting_only:
        run_mode = "fitting-only"
    else:
        run_mode = "compute+fit"

    log_stage("Stage 1/4: Configuration")
    print(f"Run mode      : {run_mode}")
    print(f"Subjects      : {subjects}")
    print(f"Sessions      : {sessions}")
    print(f"Feature       : wav2vec-S")
    print(f"  layers      : {w2v2_layers}")
    print(f"  model       : {args.w2v2_model}")
    print(f"  M (main)    : {args.main_context}")
    print(f"  R (right)   : {args.right_context}")
    print(f"Resample opt  : {RESAMPLE_OPT}  (MEG rate = {MEG_RATE} Hz)")
    print(f"HDF5 features : {hdf5_path}")
    print(f"Results dir   : {results_dir}")
    print(f"Per-lag mode  : {args.per_lag}")
    print(f"Force refit   : {args.force_refit}")

    log_stage("Stage 2/4: Collecting unique stimuli")
    stim_map = collect_unique_stimuli([BIDS_ROOT])
    print(f"Found {len(stim_map)} unique stimuli")

    device = torch.device(args.device) if args.device else None

    log_stage(f"Stage 3/4: Feature cache ({run_mode})")
    if run_mode in ("compute-feature-only", "compute+fit"):
        run_compute_features(
            args, MEG_RATE=MEG_RATE, hdf5_path=hdf5_path,
            stim_map=stim_map, device=device,
        )
    else:  # fitting-only
        load_existing_features(args, MEG_RATE=MEG_RATE, hdf5_path=hdf5_path)

    if run_mode == "compute-feature-only":
        log_stage("Done (feature cache built; --compute-feature-only).")
        return

    # Resolve --w2v2_layers against the actual tap count of the cached model.
    with h5py.File(hdf5_path, "r") as h5:
        n_taps = _w2v2_n_taps_from_h5(h5)
    if n_taps == 0:
        print(
            f"ERROR: HDF5 {hdf5_path} has no 'w2v2_layers_present' attr. "
            f"Re-run --compute-feature-only.",
            file=sys.stderr,
        )
        sys.exit(2)
    w2v2_layers = expand_w2v2_layers(w2v2_layers, n_taps)
    print(f"  resolved layers ({n_taps} taps in cache): {w2v2_layers}")

    h5f = h5py.File(hdf5_path, "r")

    total = len(subjects) * len(sessions)
    log_stage(
        f"Stage 4/4: Fitting {len(subjects)} × {len(sessions)} = {total} "
        f"(subject, session) combos"
    )
    grand_counts = {"done": 0, "cached": 0, "no_runs": 0, "skip": 0}
    try:
        for i, subj in enumerate(subjects):
            for j, ses in enumerate(sessions):
                idx = i * len(sessions) + j + 1
                log_stage(f"[{idx}/{total}] sub-{subj} ses-{ses}")
                counts = fit_one_subject_session(
                    args, subj, ses,
                    h5f=h5f,
                    MEG_RATE=MEG_RATE, RESAMPLE_OPT=RESAMPLE_OPT,
                    results_dir=results_dir,
                    w2v2_layers=w2v2_layers,
                )
                for k, v in counts.items():
                    grand_counts[k] += v
    finally:
        h5f.close()

    log_stage("Done")
    print(f"  fit              : {grand_counts['done']}")
    print(f"  cached (skipped) : {grand_counts['cached']}")
    print(f"  no runs found    : {grand_counts['no_runs']}")
    print(f"  empty/skipped    : {grand_counts['skip']}")


if __name__ == "__main__":
    main()
