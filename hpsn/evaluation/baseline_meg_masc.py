"""Défossez-2023-style MEG-MASC baseline ridge-TRF encoding pipeline.

Standalone baseline for anchoring HPSN results against the published
literature. Produces per-subject pkl files compatible with
``aggregate_group.py``.

Conditions (``--feat``)
    acoustic      — gammatone-8 + onset-8 + word-onset (Gwilliams/Brodbeck, 17 feats)
    melspec80     — 80-bin log-mel at the target MEG rate (low-level acoustic)
    w2v2          — facebook/wav2vec2-large-960h hidden state from a single
                    transformer layer (non-causal anchor). Layer selected with
                    ``--w2v2_layer`` (default 18 for wav2vec2-large).

MEG preprocessing (differs from ``proc_5_hpsn_ridge.py``):
    * bandpass 0.5–30 Hz
    * resample to target rate (100 Hz with --resample_opt REPR, 50 Hz with MEG)
    * per-channel RobustScaler (subtract median, divide by IQR)
    * clip at ±20 (on the robust-scaled data)
    * MEG cache is a SEPARATE directory tagged ``defossez`` so legacy pkls
      using the old preprocessing cannot be silently mixed with these.

Output pkls use the same filename scheme as ``proc_5_hpsn_ridge.py``, so
``aggregate_group.py`` can compare any baseline against any HPSN condition
run with matching preprocessing. Default results dir is separated from the
legacy dir for the same reason.

Usage
-----
    python -m hpsn.evaluation.baseline_meg_masc \
        --subj 01 --ses 0 --feat melspec80 --resample_opt REPR

    python -m hpsn.evaluation.baseline_meg_masc \
        --subj 01 --ses 0 --feat w2v2 --w2v2_layer 18 --resample_opt REPR

    python -m hpsn.evaluation.baseline_meg_masc \
        --subj 01 --ses 0 --feat acoustic --resample_opt REPR
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
import librosa
import mne
import mne_bids
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="Unable to map the following column", category=RuntimeWarning,
)
mne.set_log_level("WARNING")

from .proc_5_hpsn_ridge import (
    BIDS_ROOT,
    DEFAULT_TRF_TMAX,
    DEFAULT_TRF_TMIN,
    N_ACOUSTIC_PREDICTORS,
    PCA_COMPONENTS,
    build_acoustic_feature_matrix,
    build_lagged_matrix,
    build_single_lag_matrix,
    collect_unique_stimuli,
    compute_acoustic_predictors,
    fit_ridge_cv_full_lags,
    fit_ridge_cv_single_lag,
    get_subjects,
    parse_events_tsv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SR = 16_000
NATIVE_W2V2_RATE = 50  # Hz — hop = 320 samples at 16 kHz

W2V2_MODEL = "facebook/wav2vec2-large-960h"
W2V2_N_LAYERS = 24                     # wav2vec2-large transformer depth
DEFAULT_W2V2_LAYER = 18

MEL_N_MELS = 80
MEL_N_FFT = 400
MEL_F_MIN = 0
MEL_F_MAX = 8000

ROBUST_CLIP = 20.0                     # Défossez: clip at ±20 after robust scaling

RESULTS_DIR = Path(
    "/scratch/jsm04005/fie24002/DATA/HPSN/EvalResults/meg_results_defossez"
)


def log_stage(msg: str) -> None:
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] === {msg} ===", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

FEAT_CHOICES = ("acoustic", "melspec80", "w2v2")


def parse_args():
    p = argparse.ArgumentParser(
        description="Défossez-style MEG-MASC baseline (ridge-TRF encoding).",
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
    p.add_argument("--feat", required=True, choices=FEAT_CHOICES)
    p.add_argument(
        "--w2v2_layer", type=int, default=DEFAULT_W2V2_LAYER,
        help=f"Transformer layer index for w2v2 (1..{W2V2_N_LAYERS}).",
    )
    p.add_argument(
        "--w2v2_model", default=W2V2_MODEL,
        help="HuggingFace id or local path for the wav2vec2 model "
             f"(default: {W2V2_MODEL}).",
    )
    p.add_argument(
        "--resample_opt", choices=("MEG", "REPR"), default="REPR",
        help="MEG = 50 Hz, REPR = 100 Hz (w2v2 features are ×2 upsampled to match).",
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
    p.add_argument("--force_recompute_acoustic", action="store_true")
    p.add_argument(
        "--force_refit", action="store_true",
        help="Refit ridge even if the output pkl already exists.",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Subject / session resolution
# ─────────────────────────────────────────────────────────────────────────────

MEG_MASC_SESSIONS = (0, 1)  # MEG-MASC ships with ses-0 and ses-1


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
# Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    """Read a mono 16 kHz float32 waveform from ``path``."""
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        t = torchaudio.transforms.Resample(sr, SR)(t)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32)


def extract_melspec(wav: np.ndarray, frame_rate: int) -> np.ndarray:
    """Return ``[T, MEL_N_MELS]`` log-mel at ``frame_rate`` Hz."""
    hop = SR // frame_rate
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_mels=MEL_N_MELS,
        n_fft=MEL_N_FFT, hop_length=hop,
        fmin=MEL_F_MIN, fmax=MEL_F_MAX,
    )
    return np.log(mel + 1e-6).T.astype(np.float32)


class W2V2Extractor:
    """Non-causal wav2vec2 feature extractor (single-layer tap)."""

    def __init__(self, model_name: str = W2V2_MODEL,
                 device: torch.device | str | None = None):
        from transformers import Wav2Vec2Model
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model_name = model_name

    @torch.no_grad()
    def extract(self, wav: np.ndarray, layer_idx: int) -> np.ndarray:
        """Return ``hidden_states[layer_idx]`` as ``[T, H]`` float32, native 50 Hz."""
        x = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        out = self.model(x, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[layer_idx]                              # [1, T, H]
        return hs.squeeze(0).cpu().numpy().astype(np.float32)


def _linear_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation by ``factor`` along the time axis. ``[T, D] → [factor*T, D]``."""
    T, D = x.shape
    t = torch.from_numpy(x.astype(np.float32)).T.unsqueeze(0)          # [1, D, T]
    t = F.interpolate(t, size=T * factor, mode="linear", align_corners=False)
    return t.squeeze(0).T.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature HDF5 cache
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_hdf5(
    stim_map: dict, feat_type: str, frame_rate: int,
    hdf5_path: Path, force: bool, device: torch.device | str | None,
    w2v2_layer: int, w2v2_model: str = W2V2_MODEL,
):
    """Write one dataset per stim (key ``feat``) at ``frame_rate`` Hz."""
    if hdf5_path.exists() and not force:
        with h5py.File(hdf5_path, "r") as h5:
            fr_stored = h5.attrs.get("frame_rate", None)
            if fr_stored is not None and int(fr_stored) != frame_rate:
                print(
                    f"HDF5 rate {fr_stored} ≠ requested {frame_rate}. "
                    f"Pass --force_recompute_features or delete {hdf5_path}.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if feat_type == "w2v2":
                layer_stored = h5.attrs.get("w2v2_layer", None)
                if layer_stored is not None and int(layer_stored) != w2v2_layer:
                    print(
                        f"HDF5 w2v2_layer {layer_stored} ≠ requested {w2v2_layer}. "
                        f"Pass --force_recompute_features or delete {hdf5_path}.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                model_stored = h5.attrs.get("w2v2_model", None)
                if model_stored is not None and str(model_stored) != w2v2_model:
                    print(
                        f"HDF5 w2v2_model '{model_stored}' ≠ requested "
                        f"'{w2v2_model}'. Pass --force_recompute_features "
                        f"or delete {hdf5_path}.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            missing = [k for k in stim_map if k not in h5]
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

    w2v2 = None
    if feat_type == "w2v2":
        w2v2 = W2V2Extractor(model_name=w2v2_model, device=device)

    with h5py.File(hdf5_path, mode) as h5:
        h5.attrs["feat_type"] = feat_type
        h5.attrs["frame_rate"] = frame_rate
        if feat_type == "w2v2":
            h5.attrs["w2v2_model"] = w2v2_model
            h5.attrs["w2v2_layer"] = w2v2_layer

        for stim_id, wav_path in tqdm(items, desc=f"{feat_type} features"):
            wav = load_wav(wav_path)
            if feat_type == "melspec80":
                arr = extract_melspec(wav, frame_rate)
            elif feat_type == "w2v2":
                arr = w2v2.extract(wav, w2v2_layer)
                if frame_rate == 100:
                    arr = _linear_upsample(arr, 2)
                elif frame_rate != NATIVE_W2V2_RATE:
                    raise ValueError(
                        f"w2v2 frame_rate must be 50 or 100 Hz, got {frame_rate}"
                    )
            else:
                raise ValueError(feat_type)
            grp = h5.create_group(stim_id)
            grp.attrs["wav_path"] = str(wav_path)
            grp.create_dataset("feat", data=arr, compression="gzip")


def build_feat_matrix_from_hdf5(
    run, hdf5: h5py.File, T_target: int, frame_rate: int,
) -> tuple[np.ndarray, int]:
    """Place per-stim features (HDF5 key ``feat``) on the MEG timeline."""
    example = list(hdf5.keys())[0]
    D = hdf5[example]["feat"].shape[1]
    X = np.zeros((T_target, D), dtype=np.float64)
    n_placed = 0
    for _, sev in run["sound_events"].iterrows():
        stim_id = sev.get("stim_id", "")
        if stim_id not in hdf5:
            continue
        feat = hdf5[stim_id]["feat"][()]
        sf_ = int(sev["meg_onset"] * frame_rate)
        ef = min(sf_ + feat.shape[0], T_target)
        t_kept = ef - sf_
        if t_kept <= 0:
            continue
        X[sf_:ef] = feat[:t_kept]
        n_placed += 1
    return X, n_placed


# ─────────────────────────────────────────────────────────────────────────────
# MEG loader — Défossez preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def robust_scale_per_channel(data: np.ndarray, clip: float = ROBUST_CLIP) -> np.ndarray:
    """Per-channel median/IQR scaling, then clip at ±``clip``.

    ``data``: ``[n_channels, T]`` → ``[n_channels, T]`` float64.
    This is the same scheme used in Défossez et al. 2023 (brainmagick repo).
    """
    med = np.median(data, axis=1, keepdims=True)
    q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-8)
    out = (data - med) / iqr
    return np.clip(out, -clip, clip).astype(np.float64)


def load_sensor_run(
    bids_root: Path, subj: str, ses: int, task: int,
    target_rate: int, cache_dir: Path,
) -> np.ndarray | None:
    """Return ``[n_channels, T]`` at ``target_rate`` Hz with Défossez preprocessing.

    Cache stores filtered + resampled raw (scaling is re-applied cheaply on load
    so ``ROBUST_CLIP`` can be re-tuned without invalidating the cache).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fif = cache_dir / (
        f"sub-{subj}_ses-{ses}_task-{task}_defossez-{target_rate}Hz.fif"
    )
    if cache_fif.exists():
        raw = mne.io.read_raw_fif(cache_fif, preload=True, verbose=False)
    else:
        bids_path = mne_bids.BIDSPath(
            subject=subj, session=str(ses), task=str(task),
            datatype="meg", root=bids_root,
        )
        try:
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
        except FileNotFoundError:
            return None
        raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)
        raw.load_data(verbose=False).filter(0.5, 30.0, n_jobs=1, verbose=False)
        raw.resample(target_rate, verbose=False)
        raw.save(cache_fif, overwrite=True, verbose=False)
    data = raw.get_data().astype(np.float64)
    return robust_scale_per_channel(data)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(subject, session) pipeline
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESSING_TAG = "defossez-bp0.5-30-robust-clip20"


def fit_one_subject_session(
    args, subj: str, ses: int, *,
    h5f, acoustic_feats, stim_map,
    feat_label: str, FEAT: str, MEG_RATE: int, RESAMPLE_OPT: str,
    results_dir: Path,
) -> str:
    """Run the MEG-align → concat → PCA → ridge → save pipeline for one
    (subject, session). Returns a status string: 'done', 'cached', 'no_runs'."""
    save_dir = results_dir / f"trf_ridge_sub-{subj}"
    save_dir.mkdir(parents=True, exist_ok=True)
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
        return "cached"

    log_stage(f"sub-{subj} ses-{ses}: loading MEG runs")
    meg_cache_dir = results_dir / f"meg_cache_defossez_sub-{subj}"

    Y_parts, X_parts = [], []
    print(f"  sub-{subj} ses-{ses}:", end="")
    for task in range(4):
        evfile = (BIDS_ROOT / f"sub-{subj}" / f"ses-{ses}" / "meg" /
                  f"sub-{subj}_ses-{ses}_task-{task}_events.tsv")
        if not evfile.exists():
            continue

        Y = load_sensor_run(BIDS_ROOT, subj, ses, task, MEG_RATE, meg_cache_dir)
        if Y is None:
            continue
        T_run = Y.shape[1]

        ev = parse_events_tsv(evfile)
        sound_ev = ev[ev["kind"] == "sound"].reset_index(drop=True)
        run = dict(
            subject=subj, session=ses, task=task,
            events_df=ev, sound_events=sound_ev,
        )

        if FEAT == "acoustic":
            X_run, _ = build_acoustic_feature_matrix(
                run, T_run, MEG_RATE, acoustic_feats,
            )
            n_active = int((np.abs(X_run).sum(1) > 0).sum())
            if n_active < 200:
                print(f"[skip task-{task}: {n_active} active]", end="")
                continue
        else:
            X_run, n_placed = build_feat_matrix_from_hdf5(
                run, h5f, T_run, MEG_RATE,
            )
            if n_placed == 0:
                print(f"[skip task-{task}: no features placed]", end="")
                continue

        Y_parts.append(Y)
        X_parts.append(X_run)
        print(".", end="")
    print()

    if not Y_parts:
        log_stage(f"sub-{subj} ses-{ses}: no runs — skipping")
        return "no_runs"

    log_stage(f"sub-{subj} ses-{ses}: concatenating runs")
    Y_all = np.concatenate(Y_parts, axis=1)
    X_all = np.concatenate(X_parts, axis=0)
    n_sensors, T_total = Y_all.shape
    print(f"Concatenated: T={T_total}, n_sensors={n_sensors}, D={X_all.shape[1]}")

    if X_all.shape[1] > PCA_COMPONENTS:
        log_stage(f"sub-{subj} ses-{ses}: PCA → {PCA_COMPONENTS} components")
        active_mask = np.abs(X_all).sum(1) > 0
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(X_all[active_mask])
        X_all = pca.transform(X_all).astype(np.float64)
        print(f"PCA on {feat_label}: {pca.explained_variance_ratio_.sum():.2%} var")

    log_stage(f"sub-{subj} ses-{ses}: Ridge fit")

    if args.per_lag:
        lag_list_ms = [float(s) for s in args.lag_list_ms.split(",") if s.strip()]
        print(f"Per-lag mode: {len(lag_list_ms)} lags → {lag_list_ms}")
        r_by_lag = np.zeros((len(lag_list_ms), n_sensors), dtype=np.float64)
        for i, lag_ms in enumerate(lag_list_ms):
            X_lag, lag_frames = build_single_lag_matrix(X_all, MEG_RATE, lag_ms)
            print(f"  lag {lag_ms:+6.1f} ms ({lag_frames:+d} frames) ...")
            r_by_lag[i] = fit_ridge_cv_single_lag(X_lag, Y_all)
            print(f"    mean r = {r_by_lag[i].mean():.4f}")
        result = dict(
            mode="per_lag",
            r_by_lag=r_by_lag,
            lag_values_ms=np.array(lag_list_ms, dtype=np.float64),
            feat=feat_label, space="sensor", resample_opt=RESAMPLE_OPT,
            subject=subj, session=ses,
            frame_rate=MEG_RATE, n_predictors=X_all.shape[1],
            preprocessing=PREPROCESSING_TAG,
        )
    else:
        X_lagged, lags = build_lagged_matrix(
            X_all, MEG_RATE, args.trf_tmin, args.trf_tmax,
        )
        print(
            f"Lagged matrix: {X_lagged.shape}  "
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
        )
        print(f"  mean r = {r_values.mean():.4f}   max r = {r_values.max():.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved → {save_path}")
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    FEAT = args.feat
    RESAMPLE_OPT = args.resample_opt
    MEG_RATE = 100 if RESAMPLE_OPT == "REPR" else 50

    if FEAT == "w2v2":
        if not (1 <= args.w2v2_layer <= W2V2_N_LAYERS):
            print(
                f"ERROR: --w2v2_layer must be in [1, {W2V2_N_LAYERS}], "
                f"got {args.w2v2_layer}.",
                file=sys.stderr,
            )
            sys.exit(2)
        feat_label = f"w2v2_l{args.w2v2_layer}"
    else:
        feat_label = FEAT

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if FEAT == "acoustic":
        hdf5_path = None
    else:
        hdf5_path = Path(args.hdf5_path) if args.hdf5_path else (
            results_dir / f"baseline_features_{feat_label}_{RESAMPLE_OPT}.h5"
        )

    subjects = resolve_subjects(args.subj)
    sessions = resolve_sessions(args.ses)
    if not subjects:
        print("ERROR: no subjects resolved from --subj.", file=sys.stderr)
        sys.exit(2)
    if not sessions:
        print("ERROR: no sessions resolved from --ses.", file=sys.stderr)
        sys.exit(2)

    log_stage("Stage 1/4: Configuration")
    print(f"Subjects      : {subjects}")
    print(f"Sessions      : {sessions}")
    print(f"Feature       : {feat_label}")
    print(f"Resample opt  : {RESAMPLE_OPT}  (MEG rate = {MEG_RATE} Hz)")
    print(f"HDF5 features : {hdf5_path}")
    print(f"Results dir   : {results_dir}")
    print(f"Per-lag mode  : {args.per_lag}")
    print(f"Force refit   : {args.force_refit}")

    log_stage("Stage 2/4: Collecting unique stimuli")
    stim_map = collect_unique_stimuli([BIDS_ROOT])
    print(f"Found {len(stim_map)} unique stimuli")

    device = torch.device(args.device) if args.device else None
    acoustic_feats = None
    if FEAT in ("melspec80", "w2v2"):
        log_stage(f"Stage 3/4: Building HDF5 features ({FEAT})")
        build_feature_hdf5(
            stim_map,
            feat_type=FEAT,
            frame_rate=MEG_RATE,
            hdf5_path=hdf5_path,
            force=args.force_recompute_features,
            device=device,
            w2v2_layer=args.w2v2_layer,
            w2v2_model=args.w2v2_model,
        )
    elif FEAT == "acoustic":
        log_stage("Stage 3/4: Building acoustic feature cache")
        aco_cache = results_dir / f"acoustic_features_{MEG_RATE}Hz.npz"
        if aco_cache.exists() and not args.force_recompute_acoustic:
            data = np.load(aco_cache, allow_pickle=True)
            acoustic_feats = {k: data[k] for k in data.files}
            print(f"Loaded acoustic features: {len(acoustic_feats)} stims")
        else:
            acoustic_feats = {}
            for stim_id, wav_path in tqdm(
                sorted(stim_map.items()), desc="Acoustic features",
            ):
                acoustic_feats[stim_id] = compute_acoustic_predictors(wav_path, MEG_RATE)
            np.savez_compressed(aco_cache, **acoustic_feats)
            print(f"Saved acoustic features → {aco_cache}")

    h5f = h5py.File(hdf5_path, "r") if hdf5_path is not None else None

    total = len(subjects) * len(sessions)
    log_stage(
        f"Stage 4/4: Fitting {len(subjects)} × {len(sessions)} = {total} "
        f"(subject, session) combos"
    )
    status_counts = {"done": 0, "cached": 0, "no_runs": 0}
    try:
        for i, subj in enumerate(subjects):
            for j, ses in enumerate(sessions):
                idx = i * len(sessions) + j + 1
                log_stage(f"[{idx}/{total}] sub-{subj} ses-{ses}")
                status = fit_one_subject_session(
                    args, subj, ses,
                    h5f=h5f, acoustic_feats=acoustic_feats, stim_map=stim_map,
                    feat_label=feat_label, FEAT=FEAT,
                    MEG_RATE=MEG_RATE, RESAMPLE_OPT=RESAMPLE_OPT,
                    results_dir=results_dir,
                )
                status_counts[status] += 1
    finally:
        if h5f is not None:
            h5f.close()

    log_stage("Done")
    print(f"  fit              : {status_counts['done']}")
    print(f"  cached (skipped) : {status_counts['cached']}")
    print(f"  no runs found    : {status_counts['no_runs']}")


if __name__ == "__main__":
    main()
