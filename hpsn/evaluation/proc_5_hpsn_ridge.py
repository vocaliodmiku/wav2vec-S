"""HPSN × MEG-MASC ridge-regression encoding-model pipeline.

Rewrite of ``proc_5_meg_masc_ridge.py`` (EarShot) adapted to the trained
wav2vec-S → HPSN stack.

Usage
-----
    # Sensor-level, native 50 Hz, HPSN Level-1 features
    python -m hpsn.evaluation.proc_5_hpsn_ridge \
        --subj 01 --ses 0 --feat hpsn_l1 --space sensor --resample_opt MEG \
        --ckpt /path/to/hpsn/checkpoint

    # ROI-level, 100 Hz upsampled features, acoustic baseline
    python -m hpsn.evaluation.proc_5_hpsn_ridge \
        --subj 01 --ses 0 --feat acoustic --space roi --resample_opt REPR

    # Per-lag time-resolved analysis
    python -m hpsn.evaluation.proc_5_hpsn_ridge \
        --subj 01 --ses 0 --feat hpsn_l1 --space sensor --resample_opt MEG \
        --ckpt /path/to/hpsn/checkpoint \
        --per_lag --lag_list_ms 0,40,80,120,160,200,300,400,500

Conditions (``--feat``)
    acoustic                  — gammatone-8 + onset-8 + word-onset (17 feats)
    baseline_low              — wav2vec-S acoustic-band tap  (H=1024)
    baseline_mid              — wav2vec-S lexical-band tap   (H=1024)
    hpsn_l1                   — HPSN Level-1 representation  (D=512)
    hpsn_l2                   — HPSN Level-2 representation  (D=512)
    combined_baseline_low     — acoustic + PCA(baseline_low)
    combined_baseline_mid     — acoustic + PCA(baseline_mid)
    combined_hpsn_l1          — acoustic + PCA(hpsn_l1)
    combined_hpsn_l2          — acoustic + PCA(hpsn_l2)
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
import pandas as pd
import torch
from gammatone.filters import centre_freqs as _gt_centre_freqs
from gammatone.gtgram import gtgram
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="Unable to map the following column",
    category=RuntimeWarning,
)
mne.set_log_level("WARNING")

from ..config import HPSNConfig
from .features import (
    CONDITIONS,
    HPSNFeatureExtractor,
    extract_to_hdf5,
    NATIVE_FRAME_RATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path(
    "/scratch/jsm04005/fie24002/DATA/meg-masc/meg_masc_tmp/Part-1/osfstorage"
)
BIDS_ROOT_P2 = Path(
    "/scratch/jsm04005/fie24002/DATA/meg-masc/meg_masc_tmp/Part-2/osfstorage"
)
RESULTS_DIR = Path(
    "/scratch/jsm04005/fie24002/DATA/HPSN/EvalResults/meg_results"
)
SOURCE_DATA_DIR = Path(
    "/scratch/jsm04005/fie24002/DATA/L360-Word20000-5L-512-CNN/"
    "EvalResults/meg_results/source_data"
)

SR = 16_000

# Acoustic predictor constants (unchanged from prior pipeline)
ACOUSTIC_N_BANDS = 256
ACOUSTIC_F_MIN = 20
ACOUSTIC_F_MAX = 5000
ACOUSTIC_N_BINS = 8
ACOUSTIC_INTERNAL_SR = 1000
N_ACOUSTIC_PREDICTORS = 17  # 8 gammatone + 8 onset + 1 word onset

# Ridge
RIDGE_ALPHAS = np.logspace(-2, 8, 10)
N_FOLDS = 5
PCA_COMPONENTS = 200

# Default TRF window (used when --per_lag is NOT set)
DEFAULT_TRF_TMIN = -0.100
DEFAULT_TRF_TMAX = 1.000

BASELINE_CONDS = ("baseline_low", "baseline_mid", "hpsn_l1", "hpsn_l2")
COMBINED_CONDS = tuple(f"combined_{c}" for c in BASELINE_CONDS)
ALL_FEATS = ("acoustic",) + BASELINE_CONDS + COMBINED_CONDS


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HPSN × MEG-MASC ridge encoding.")
    p.add_argument("--subj", required=True)
    p.add_argument("--ses", required=True, type=int)
    p.add_argument("--feat", required=True, choices=ALL_FEATS)
    p.add_argument("--space", choices=("sensor", "roi"), default="sensor")
    p.add_argument("--resample_opt", choices=("MEG", "REPR"), default="MEG",
                   help="MEG: resample MEG to wav2vec-S native 50 Hz. "
                        "REPR: upsample features to 100 Hz and resample MEG to 100 Hz.")
    p.add_argument("--ckpt", default=None,
                   help="Path to HPSN accelerate checkpoint dir (or state-dict file). "
                        "Required unless --feat=acoustic.")
    p.add_argument("--backbone_model", default=HPSNConfig.backbone_model)
    p.add_argument("--hdf5_path", default=None,
                   help="Override HPSN feature HDF5 location.")
    p.add_argument("--force_recompute_features", action="store_true")
    p.add_argument("--force_recompute_acoustic", action="store_true")

    p.add_argument("--trf_tmin", type=float, default=DEFAULT_TRF_TMIN)
    p.add_argument("--trf_tmax", type=float, default=DEFAULT_TRF_TMAX)

    p.add_argument("--per_lag", action="store_true",
                   help="Fit a SEPARATE single-lag ridge per lag in --lag_list_ms.")
    p.add_argument("--lag_list_ms", default="0,40,80,120,160,200,300,400,500",
                   help="Comma-separated lags (ms) used when --per_lag is set.")

    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    p.add_argument("--source_data_dir", default=str(SOURCE_DATA_DIR))
    p.add_argument("--device", default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic features (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _freq_to_erb(f):
    return 21.4 * np.log10(0.00437 * f + 1)


def _erb_to_freq(erb):
    return (10 ** (erb / 21.4) - 1) / 0.00437


def _downsample_freq_erb(spec, cfs, n_out=8):
    erb_edges = np.linspace(_freq_to_erb(cfs.min()), _freq_to_erb(cfs.max()), n_out + 1)
    f_edges = _erb_to_freq(erb_edges)
    out = np.zeros((n_out, spec.shape[1]), dtype=np.float32)
    for b in range(n_out):
        mask = (cfs >= f_edges[b]) & (cfs < f_edges[b + 1])
        if not mask.any():
            mid = 0.5 * (f_edges[b] + f_edges[b + 1])
            mask[np.argmin(np.abs(cfs - mid))] = True
        out[b] = spec[mask].mean(axis=0)
    return out


def _onset_fishbach(spec):
    d = np.diff(spec, axis=1, prepend=spec[:, :1])
    return np.maximum(d, 0.0)


def compute_acoustic_predictors(wav_path: Path, frame_rate: int) -> np.ndarray:
    """Return [T, 16] gammatone-8 + onset-8 at ``frame_rate`` Hz."""
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    win_t = hop_t = 1.0 / ACOUSTIC_INTERNAL_SR
    try:
        gt = gtgram(y, SR, win_t, hop_t, ACOUSTIC_N_BANDS, ACOUSTIC_F_MIN, ACOUSTIC_F_MAX)
    except TypeError:
        gt = gtgram(y, SR, win_t, hop_t, ACOUSTIC_N_BANDS, ACOUSTIC_F_MIN)
    try:
        cfs = _gt_centre_freqs(SR, ACOUSTIC_N_BANDS, ACOUSTIC_F_MIN, ACOUSTIC_F_MAX)
    except TypeError:
        cfs = _gt_centre_freqs(SR, ACOUSTIC_N_BANDS, ACOUSTIC_F_MIN)
    sort_idx = np.argsort(cfs)
    cfs = cfs[sort_idx]
    gt = gt[sort_idx]
    gt_log = np.log(gt + 1e-12).astype(np.float32)
    ons = _onset_fishbach(gt_log)
    gt_8 = _downsample_freq_erb(gt_log, cfs, ACOUSTIC_N_BINS)
    ons_8 = _downsample_freq_erb(ons, cfs, ACOUSTIC_N_BINS)
    ds = ACOUSTIC_INTERNAL_SR // frame_rate
    T_out = gt_8.shape[1] // ds
    gt_8 = gt_8[:, : T_out * ds].reshape(ACOUSTIC_N_BINS, T_out, ds).mean(2)
    ons_8 = ons_8[:, : T_out * ds].reshape(ACOUSTIC_N_BINS, T_out, ds).mean(2)
    return np.concatenate([gt_8, ons_8], axis=0).T.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BIDS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_stim_id(sound_field: str) -> str:
    stem = Path(sound_field).stem
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def collect_unique_stimuli(bids_roots):
    stim_map = {}
    for broot in bids_roots:
        for evfile in sorted(broot.rglob("*_events.tsv")):
            df = pd.read_csv(evfile, sep="\t")
            for _, row in df.iterrows():
                try:
                    meta = eval(row["trial_type"])
                except Exception:
                    continue
                if meta.get("kind") != "sound":
                    continue
                sound_rel = meta["sound"]
                stim_id = _normalize_stim_id(sound_rel)
                norm_name = stim_id + ".wav"
                for br in bids_roots:
                    candidate = br / "stimuli" / "audio" / norm_name
                    if not candidate.exists():
                        candidate = br / sound_rel
                    if candidate.exists():
                        stim_map[stim_id] = candidate
                        break
    return stim_map


def parse_events_tsv(evfile: Path) -> pd.DataFrame:
    df = pd.read_csv(evfile, sep="\t")
    rows = []
    for _, row in df.iterrows():
        try:
            meta = eval(row["trial_type"])
        except Exception:
            continue
        meta["meg_onset"] = float(row["onset"])
        rows.append(meta)
    ev = pd.DataFrame(rows)
    if "sound" in ev.columns:
        ev["stim_id"] = ev["sound"].apply(_normalize_stim_id)
    return ev


def get_subjects(bids_root: Path):
    ptfile = bids_root / "participants.tsv"
    if not ptfile.exists():
        return []
    df = pd.read_csv(ptfile, sep="\t")
    return df["participant_id"].apply(lambda x: x.split("-")[1]).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Feature-matrix builders
# ─────────────────────────────────────────────────────────────────────────────

def build_acoustic_feature_matrix(run, T_target, frame_rate, acoustic_feats):
    X = np.zeros((T_target, N_ACOUSTIC_PREDICTORS), dtype=np.float64)
    n_placed = 0
    for _, sev in run["sound_events"].iterrows():
        stim_id = sev.get("stim_id", "")
        if stim_id not in acoustic_feats:
            continue
        feat = acoustic_feats[stim_id]
        sf_ = int(sev["meg_onset"] * frame_rate)
        ef = min(sf_ + feat.shape[0], T_target)
        t_kept = ef - sf_
        if t_kept <= 0:
            continue
        X[sf_:ef, :16] = feat[:t_kept]
        n_placed += 1
    word_ev = run["events_df"][run["events_df"]["kind"] == "word"]
    for _, wev in word_ev.iterrows():
        f_idx = int(wev["meg_onset"] * frame_rate)
        if 0 <= f_idx < T_target:
            X[f_idx, 16] = 1.0
    return X, n_placed


def build_condition_matrix(run, condition, h5, T_target, frame_rate):
    """Place per-stim HPSN features (one condition) on the MEG timeline."""
    D = h5[list(h5.keys())[0]][condition].shape[1]
    X = np.zeros((T_target, D), dtype=np.float64)
    n_placed = 0
    for _, sev in run["sound_events"].iterrows():
        stim_id = sev.get("stim_id", "")
        if stim_id not in h5 or condition not in h5[stim_id]:
            continue
        feat = h5[stim_id][condition][()]
        sf_ = int(sev["meg_onset"] * frame_rate)
        ef = min(sf_ + feat.shape[0], T_target)
        t_kept = ef - sf_
        if t_kept <= 0:
            continue
        X[sf_:ef] = feat[:t_kept]
        n_placed += 1
    return X, n_placed


def build_lagged_matrix(X, frame_rate, tmin_s, tmax_s):
    n_samples, n_features = X.shape
    lag_min = int(np.round(tmin_s * frame_rate))
    lag_max = int(np.round(tmax_s * frame_rate))
    lags = np.arange(lag_min, lag_max + 1)
    n_lags = len(lags)
    X_lagged = np.zeros((n_samples, n_features * n_lags), dtype=X.dtype)
    for li, lag in enumerate(lags):
        s, e = li * n_features, (li + 1) * n_features
        if lag >= 0:
            X_lagged[lag:, s:e] = X[: n_samples - lag]
        else:
            X_lagged[: n_samples + lag, s:e] = X[-lag:]
    return X_lagged, lags


def build_single_lag_matrix(X, frame_rate, lag_ms):
    """Shift X by exactly ``lag_ms`` milliseconds (positive = brain lags features)."""
    lag = int(np.round(lag_ms / 1000.0 * frame_rate))
    n = X.shape[0]
    Xl = np.zeros_like(X)
    if lag >= 0:
        Xl[lag:] = X[: n - lag] if lag < n else 0
    else:
        Xl[: n + lag] = X[-lag:] if -lag < n else 0
    return Xl, lag


# ─────────────────────────────────────────────────────────────────────────────
# MEG / ROI loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sensor_run(bids_root, subj, ses, task, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_fif = cache_dir / f"sub-{subj}_ses-{ses}_task-{task}_meg-filt.fif"
    if cache_fif.exists():
        return mne.io.read_raw_fif(cache_fif, preload=True, verbose=False)
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
    raw.save(cache_fif, overwrite=True, verbose=False)
    return raw


def load_roi_run(source_data_dir: Path, subj: str, ses: int, task: int):
    """Load ROI time-series saved at 100 Hz by proc_5_ROI.ipynb.  Returns
    (data [n_roi, T], sfreq)."""
    roi_path = source_data_dir / subj / f"roi_ses-{ses}_task-{task}.npy"
    if not roi_path.exists():
        return None, None
    data = np.load(roi_path)
    return data.astype(np.float64), 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Ridge fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_ridge_cv_full_lags(X_lagged, Y_all, n_folds=N_FOLDS):
    """Return r_values [n_sensors], coef_full [n_sensors, n_lag_feats], fold_alphas."""
    T_total, _ = X_lagged.shape
    n_sensors = Y_all.shape[0]
    Y_pred = np.zeros((T_total, n_sensors), dtype=np.float64)
    fold_size = T_total // n_folds
    fold_alphas = []
    for fold in range(n_folds):
        t0 = fold * fold_size
        t1 = (fold + 1) * fold_size if fold < n_folds - 1 else T_total
        train_mask = np.ones(T_total, dtype=bool)
        train_mask[t0:t1] = False
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(X_lagged[train_mask], Y_all[:, train_mask].T)
        Y_pred[t0:t1] = ridge.predict(X_lagged[t0:t1])
        fold_alphas.append(float(ridge.alpha_))
        print(f"    Fold {fold+1}/{n_folds}: alpha = {ridge.alpha_:.2e}")
    r_values = np.array([
        np.corrcoef(Y_all[i], Y_pred[:, i])[0, 1] for i in range(n_sensors)
    ])

    ridge_full = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge_full.fit(X_lagged, Y_all.T)
    return r_values, ridge_full.coef_, fold_alphas, float(ridge_full.alpha_)


def fit_ridge_cv_single_lag(X_lag, Y_all, n_folds=N_FOLDS):
    """Cross-validated Pearson r for a single-lag design matrix."""
    T_total = X_lag.shape[0]
    n_sensors = Y_all.shape[0]
    Y_pred = np.zeros((T_total, n_sensors), dtype=np.float64)
    fold_size = T_total // n_folds
    for fold in range(n_folds):
        t0 = fold * fold_size
        t1 = (fold + 1) * fold_size if fold < n_folds - 1 else T_total
        train_mask = np.ones(T_total, dtype=bool)
        train_mask[t0:t1] = False
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(X_lag[train_mask], Y_all[:, train_mask].T)
        Y_pred[t0:t1] = ridge.predict(X_lag[t0:t1])
    return np.array([
        np.corrcoef(Y_all[i], Y_pred[:, i])[0, 1] for i in range(n_sensors)
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    SUBJECT = args.subj
    SESSION = args.ses
    FEAT = args.feat
    SPACE = args.space
    RESAMPLE_OPT = args.resample_opt
    MEG_RATE = 100 if RESAMPLE_OPT == "REPR" else NATIVE_FRAME_RATE

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir = Path(args.source_data_dir)
    hdf5_path = Path(args.hdf5_path) if args.hdf5_path else (
        results_dir / f"hpsn_features_{RESAMPLE_OPT}.h5"
    )

    needs_hpsn = FEAT in BASELINE_CONDS or FEAT in COMBINED_CONDS
    if needs_hpsn and args.ckpt is None:
        print("ERROR: --ckpt is required for HPSN conditions.", file=sys.stderr)
        sys.exit(2)

    print(f"Subject        : {SUBJECT}")
    print(f"Session        : {SESSION}")
    print(f"Feature        : {FEAT}")
    print(f"Space          : {SPACE}")
    print(f"Resample opt   : {RESAMPLE_OPT}  (MEG rate = {MEG_RATE} Hz)")
    print(f"HDF5 features  : {hdf5_path}")
    print(f"Checkpoint     : {args.ckpt}")
    print(f"Per-lag mode   : {args.per_lag}")

    # ── Stimulus map ─────────────────────────────────────────────────────
    stim_map = collect_unique_stimuli([BIDS_ROOT, BIDS_ROOT_P2])
    print(f"Found {len(stim_map)} unique stimuli")

    # ── Step 1: HPSN features HDF5 ───────────────────────────────────────
    if needs_hpsn:
        if not hdf5_path.exists() or args.force_recompute_features:
            cfg = HPSNConfig(backbone_model=args.backbone_model)
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
                print(f"ERROR: HDF5 frame_rate={fr} ≠ requested MEG rate "
                      f"{MEG_RATE}. Delete {hdf5_path} or pass "
                      f"--force_recompute_features.", file=sys.stderr)
                sys.exit(2)
            print(f"Reusing HPSN features at {hdf5_path} "
                  f"(frame_rate={fr}, resample_opt={ro})")

    # ── Step 2: collect subject runs ─────────────────────────────────────
    all_bids = [(BIDS_ROOT, get_subjects(BIDS_ROOT)),
                (BIDS_ROOT_P2, get_subjects(BIDS_ROOT_P2))]
    seen, subject_roots = set(), []
    for broot, subs in all_bids:
        for s in subs:
            if s not in seen:
                seen.add(s)
                subject_roots.append((broot, s))
    subject_roots = [(br, s) for br, s in subject_roots if s == SUBJECT]
    if not subject_roots:
        raise ValueError(f"Subject '{SUBJECT}' not found in BIDS.")

    MEG_CACHE_DIR = results_dir / f"meg_cache_sub-{SUBJECT}"

    h5_file = h5py.File(hdf5_path, "r") if needs_hpsn else None
    run_data = []
    for bids_root, subj in subject_roots:
        print(f"  sub-{subj} ses-{SESSION}:", end="")
        for task in range(4):
            evfile = (bids_root / f"sub-{subj}" / f"ses-{SESSION}" / "meg" /
                      f"sub-{subj}_ses-{SESSION}_task-{task}_events.tsv")
            if not evfile.exists():
                continue

            # Y: sensor or ROI
            if SPACE == "sensor":
                raw = load_sensor_run(bids_root, subj, SESSION, task, MEG_CACHE_DIR)
                if raw is None:
                    continue
                raw_ds = raw.copy().resample(MEG_RATE, verbose=False)
                Y = raw_ds.get_data().astype(np.float64)  # [n_sensors, T]
                del raw, raw_ds
            else:
                Y_roi, sf_roi = load_roi_run(source_data_dir, subj, SESSION, task)
                if Y_roi is None:
                    print(f"[skip task-{task}: no ROI npy]", end="")
                    continue
                if int(sf_roi) != MEG_RATE:
                    # Resample along time axis.
                    # Y_roi: [n_roi, T] at sf_roi → MEG_RATE.
                    up, down = int(MEG_RATE), int(sf_roi)
                    Y = resample_poly(Y_roi, up, down, axis=1).astype(np.float64)
                else:
                    Y = Y_roi

            ev = parse_events_tsv(evfile)
            sound_ev = ev[ev["kind"] == "sound"].reset_index(drop=True)
            run_data.append(dict(
                subject=subj, session=SESSION, task=task,
                Y=Y, events_df=ev, sound_events=sound_ev,
            ))
            print(".", end="")
    print()

    if not run_data:
        print("No runs with data — aborting.")
        if h5_file is not None:
            h5_file.close()
        return

    # ── Step 3: acoustic predictor cache ─────────────────────────────────
    needs_acoustic = (FEAT == "acoustic") or (FEAT in COMBINED_CONDS)
    acoustic_feats = {}
    if needs_acoustic:
        aco_cache = results_dir / f"acoustic_features_{MEG_RATE}Hz.npz"
        if aco_cache.exists() and not args.force_recompute_acoustic:
            data = np.load(aco_cache, allow_pickle=True)
            acoustic_feats = {k: data[k] for k in data.files}
            print(f"Loaded cached acoustic features: {len(acoustic_feats)} stims")
        else:
            for stim_id, wav_path in tqdm(
                sorted(stim_map.items()), desc="Acoustic features"
            ):
                acoustic_feats[stim_id] = compute_acoustic_predictors(wav_path, MEG_RATE)
            np.savez_compressed(aco_cache, **acoustic_feats)
            print(f"Saved acoustic features to {aco_cache}")

    # ── Step 4: build design matrix per run, concatenate ─────────────────
    Y_parts = []
    X_parts = []  # list of [T, D] per run
    for rd in run_data:
        Y_run = rd["Y"]
        T_run = Y_run.shape[1]

        # Validation: need some aligned sound events.
        if needs_acoustic:
            X_aco, n_placed_aco = build_acoustic_feature_matrix(
                rd, T_run, MEG_RATE, acoustic_feats
            )
            n_active = int((np.abs(X_aco).sum(1) > 0).sum())
            if n_active < 200:
                print(f"  task-{rd['task']}: only {n_active} active frames — skipping.")
                continue
        else:
            X_aco = None

        if FEAT == "acoustic":
            X_run = X_aco
        elif FEAT in BASELINE_CONDS:
            X_run, _ = build_condition_matrix(rd, FEAT, h5_file, T_run, MEG_RATE)
        elif FEAT in COMBINED_CONDS:
            cond = FEAT[len("combined_"):]
            X_cond, _ = build_condition_matrix(rd, cond, h5_file, T_run, MEG_RATE)
            X_run = np.concatenate([X_aco, X_cond], axis=1)   # PCA applied post-concat below
        else:
            raise ValueError(FEAT)

        Y_parts.append(Y_run)
        X_parts.append(X_run)

    if not Y_parts:
        print("No runs survived filtering — aborting.")
        if h5_file is not None:
            h5_file.close()
        return

    Y_all = np.concatenate(Y_parts, axis=1)          # [n_sensors, T_total]
    X_all = np.concatenate(X_parts, axis=0)          # [T_total, D_all]
    n_sensors, T_total = Y_all.shape
    print(f"Concatenated: T={T_total}, n_channels={n_sensors}, D={X_all.shape[1]}")

    # ── PCA for high-dim conditions ──────────────────────────────────────
    # Apply PCA when combined_* or when baseline has D > PCA_COMPONENTS.
    if FEAT in COMBINED_CONDS:
        cond_dim = X_all.shape[1] - N_ACOUSTIC_PREDICTORS
        X_aco_cat = X_all[:, :N_ACOUSTIC_PREDICTORS]
        X_cond = X_all[:, N_ACOUSTIC_PREDICTORS:]
        if cond_dim > PCA_COMPONENTS:
            active_mask = np.abs(X_cond).sum(1) > 0
            pca = PCA(n_components=PCA_COMPONENTS)
            pca.fit(X_cond[active_mask])
            X_cond = pca.transform(X_cond).astype(np.float64)
            print(f"PCA on {FEAT[len('combined_'):]}: "
                  f"{pca.explained_variance_ratio_.sum():.2%} var")
        X_all = np.concatenate([X_aco_cat, X_cond], axis=1)
    elif FEAT in BASELINE_CONDS and X_all.shape[1] > PCA_COMPONENTS:
        active_mask = np.abs(X_all).sum(1) > 0
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(X_all[active_mask])
        X_all = pca.transform(X_all).astype(np.float64)
        print(f"PCA on {FEAT}: {pca.explained_variance_ratio_.sum():.2%} var")

    # ── Step 5: ridge fit ────────────────────────────────────────────────
    print(f"\nRidge fit at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    save_dir = results_dir / f"trf_ridge_sub-{SUBJECT}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = (f"trf_ridge_{FEAT}_{SPACE}_{RESAMPLE_OPT}_"
                 f"sub-{SUBJECT}_ses-{SESSION}.pkl")
    save_path = save_dir / save_name

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
            feat=FEAT, space=SPACE, resample_opt=RESAMPLE_OPT,
            subject=SUBJECT, session=SESSION,
            frame_rate=MEG_RATE, n_predictors=X_all.shape[1],
        )
    else:
        X_lagged, lags = build_lagged_matrix(X_all, MEG_RATE, args.trf_tmin, args.trf_tmax)
        print(f"Lagged matrix: {X_lagged.shape}  ({X_all.shape[1]} feats × {len(lags)} lags)")
        r_values, coef_full, fold_alphas, full_alpha = fit_ridge_cv_full_lags(
            X_lagged, Y_all
        )
        result = dict(
            mode="full_lags",
            r=r_values,
            coef=coef_full.reshape(n_sensors, len(lags), X_all.shape[1]),
            lags=lags,
            trf_tmin=args.trf_tmin, trf_tmax=args.trf_tmax,
            fold_alphas=fold_alphas, full_alpha=full_alpha,
            feat=FEAT, space=SPACE, resample_opt=RESAMPLE_OPT,
            subject=SUBJECT, session=SESSION,
            frame_rate=MEG_RATE, n_predictors=X_all.shape[1],
        )
        print(f"  mean r = {r_values.mean():.4f}   max r = {r_values.max():.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved → {save_path}")

    if h5_file is not None:
        h5_file.close()


if __name__ == "__main__":
    main()
