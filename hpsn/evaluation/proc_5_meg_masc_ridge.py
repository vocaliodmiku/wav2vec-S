"""
Layer-Wise MEG-MASC Analysis — TRF-Tools Pipeline

Tests whether EarShot's LSTM encodes a processing hierarchy that mirrors the
brain, using the TRF-Tools pipeline (Brodbeck et al.) for mTRF estimation.

Usage
-----
    python proc_5_meg_masc_acoustic_gpu.py --subj 01 --ses 0 --feat acoustic
    python proc_5_meg_masc_acoustic_gpu.py --subj 01 --ses 0 --feat layer
    python proc_5_meg_masc_acoustic_gpu.py --subj 01 --ses 0 --feat combined

Flags
-----
--subj   Subject ID (e.g. '01').  Must match the BIDS participant_id.
--ses    Session index (e.g. 0 or 1).
--feat   Feature type for the TRF: 'acoustic', 'layer', or 'combined'.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import torch
import torch.nn as nn
import librosa
import mne
import mne_bids
import pandas as pd
import h5py
import datetime
from collections import defaultdict as _ddict
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Unable to map the following column",
                        category=RuntimeWarning)
mne.set_log_level("WARNING")

# ── TRF-Tools Pipeline (Brodbeck et al.) ──────────────────────────────────────
import eelbrain
from eelbrain import (NDVar, UTS, Scalar, Sensor,
                      save as eel_save, load as eel_load)
from trftools.pipeline import TRFExperiment, FilePredictor

from earshot.models.base_model import EarShotModel
from earshot.dataset.audio import spectrogram as earshot_spectrogram
from gammatone.gtgram import gtgram
from gammatone.filters import centre_freqs as _gt_centre_freqs
from wordfreq import zipf_frequency


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer-wise MEG-MASC acoustic TRF / RSA analysis."
    )
    parser.add_argument(
        "--subj",
        required=True,
        help="Subject ID to process (e.g. '01').  Must match the BIDS participant_id.",
    )
    parser.add_argument(
        "--ses",
        required=True,
        type=int,
        help="Session index to process (e.g. 0 or 1).",
    )
    parser.add_argument(
        "--feat",
        required=True,
        choices=["acoustic", "layer", "combined"],
        help="Feature type: 'acoustic' (baseline), 'layer' (EarShot only), "
             "or 'combined' (acoustic + EarShot).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path("/scratch/jsm04005/fie24002/DATA/L360-Word20000-5L-512-CNN/Checkpoint/Checkpoint-48.pt")
BIDS_ROOT       = Path("/scratch/jsm04005/fie24002/DATA/meg-masc/meg_masc_tmp/Part-1/osfstorage")
BIDS_ROOT_P2    = Path("/scratch/jsm04005/fie24002/DATA/meg-masc/meg_masc_tmp/Part-2/osfstorage")
STIMULI_DIR     = BIDS_ROOT / "stimuli" / "audio"
RESULTS_DIR     = Path("/scratch/jsm04005/fie24002/DATA/L360-Word20000-5L-512-CNN/EvalResults/meg_results")
HDF5_PATH       = RESULTS_DIR / "layer_features.h5"
PREDICTOR_DIR   = RESULTS_DIR / "predictors"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

# Model hyper-parameters
FEATURE_DIM  = 256
N_HIDDEN     = 512
TARGET_DIM   = 300
HIDDEN_TYPE  = "LSTM"
NUM_LAYERS   = 5
TARGET_MODE  = "Word2Vec"
HIDDEN_RESET = True
ONSET_DELAY  = 0
FRAME_RATE   = 100       # Hz  (hop_length=160 @ 16 kHz)
SR           = 16_000
HOP_LENGTH   = 160
N_FFT        = 511
N_MELS       = 256

# TRF-Tools Pipeline Parameters
TRF_TMIN       = -0.100
TRF_TMAX       =  1.000
EPOCH_TMIN     = -0.200
EPOCH_TMAX     =  0.600
DECIM          = 10
PCA_COMPONENTS = 200
N_PERMUTATIONS = 200

# Ridge regression parameters for TRF fitting
RIDGE_ALPHAS = np.logspace(-2, 8, 10)
N_FOLDS      = 5

LAYER_LABELS = ["lstm_l4"]
N_LAYERS     = len(LAYER_LABELS)

# Acoustic predictor constants
ACOUSTIC_N_BANDS      = 256
ACOUSTIC_F_MIN        = 20
ACOUSTIC_F_MAX        = 5000
ACOUSTIC_N_BINS       = 8
ACOUSTIC_INTERNAL_SR  = 1000
N_ACOUSTIC_PREDICTORS = 17   # 8 gammatone + 8 onset + 1 word onset



# ─────────────────────────────────────────────────────────────────────────────
# Model: LayerWiseExtractor
# ─────────────────────────────────────────────────────────────────────────────

class LayerWiseExtractor(nn.Module):
    """
    Wraps EarShotModel's CNN and decomposes its multi-layer LSTM into sequential
    single-layer LSTMs so that every layer's full per-timestep output is accessible.

    forward_all_layers returns a dict of {layer_label: ndarray(T, H)}.
    """

    def __init__(self, base_model: EarShotModel):
        super().__init__()
        self.convnet     = base_model.convnet
        self.semantic_fc = base_model.semantic_fc
        self.n_hidden    = base_model.n_hidden
        self.feature_dim = base_model.feature_dim

        src_rnn  = base_model.rnn
        n_layers = src_rnn.num_layers

        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            in_size = self.feature_dim if i == 0 else self.n_hidden
            cell = nn.LSTM(in_size, self.n_hidden, num_layers=1, batch_first=True)
            with torch.no_grad():
                cell.weight_ih_l0.copy_(getattr(src_rnn, f"weight_ih_l{i}"))
                cell.weight_hh_l0.copy_(getattr(src_rnn, f"weight_hh_l{i}"))
                cell.bias_ih_l0  .copy_(getattr(src_rnn, f"bias_ih_l{i}"))
                cell.bias_hh_l0  .copy_(getattr(src_rnn, f"bias_hh_l{i}"))
            self.lstm_layers.append(cell)

    @torch.no_grad()
    def forward_all_layers(self, spec: torch.Tensor) -> dict:
        features = {}

        x_cnn   = self.convnet(spec.unsqueeze(0))           # (1, 256, T)
        cnn_out = x_cnn.squeeze(0).transpose(0, 1)          # (T, 256)
        features["cnn"] = cnn_out.cpu().numpy()

        x_rnn = cnn_out.unsqueeze(0)                        # (1, T, 256)
        for i, cell in enumerate(self.lstm_layers):
            out, _ = cell(x_rnn)                            # (1, T, H)
            x_rnn  = out
            features[f"lstm_l{i}"] = out.squeeze(0).cpu().numpy()

        fc_out = self.semantic_fc(out)                      # (1, T, 300)
        features["fc"] = fc_out.squeeze(0).cpu().numpy()

        return features


# ─────────────────────────────────────────────────────────────────────────────
# Audio / spectrogram helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mel_spectrogram(wav_path: Path) -> np.ndarray:
    y, _ = librosa.core.load(str(wav_path), sr=SR, mono=True)
    feat = earshot_spectrogram(
        y,
        frame_shift_ms=10,
        frame_length_ms=10,
        sample_rate=SR,
    ).astype(np.float32)
    pad = np.zeros((N_MELS, ONSET_DELAY), dtype=np.float32)
    return np.concatenate([pad, feat], axis=1)


def _normalize_stim_id(sound_field: str) -> str:
    stem  = Path(sound_field).stem
    parts = stem.rsplit('.', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def collect_unique_stimuli(bids_roots: list) -> dict:
    stim_map = {}
    for bids_root in bids_roots:
        for evfile in sorted(bids_root.rglob("*_events.tsv")):
            df = pd.read_csv(evfile, sep="\t")
            for _, row in df.iterrows():
                try:
                    meta = eval(row["trial_type"])
                except Exception:
                    continue
                if meta.get("kind") != "sound":
                    continue
                sound_rel = meta["sound"]
                stim_id   = _normalize_stim_id(sound_rel)
                norm_name = stim_id + ".wav"
                for broot in bids_roots:
                    candidate = broot / "stimuli" / "audio" / norm_name
                    if not candidate.exists():
                        candidate = broot / sound_rel
                    if candidate.exists():
                        stim_map[stim_id] = candidate
                        break
    return stim_map


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def freq_to_erb(f):
    return 21.4 * np.log10(0.00437 * f + 1)

def erb_to_freq(erb):
    return (10 ** (erb / 21.4) - 1) / 0.00437

def downsample_freq_erb(spec, cfs, n_out=8):
    erb_edges = np.linspace(freq_to_erb(cfs.min()), freq_to_erb(cfs.max()), n_out + 1)
    f_edges   = erb_to_freq(erb_edges)
    out = np.zeros((n_out, spec.shape[1]), dtype=np.float32)
    for b in range(n_out):
        mask = (cfs >= f_edges[b]) & (cfs < f_edges[b + 1])
        if not mask.any():
            mid = 0.5 * (f_edges[b] + f_edges[b + 1])
            mask[np.argmin(np.abs(cfs - mid))] = True
        out[b] = spec[mask].mean(axis=0)
    return out

def onset_spectrogram_fishbach(spec):
    d = np.diff(spec, axis=1, prepend=spec[:, :1])
    return np.maximum(d, 0.0)

def compute_acoustic_predictors(wav_path: Path) -> np.ndarray:
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
    gt  = gt[sort_idx]
    gt_log = np.log(gt + 1e-12).astype(np.float32)
    ons    = onset_spectrogram_fishbach(gt_log)
    gt_8   = downsample_freq_erb(gt_log, cfs, ACOUSTIC_N_BINS)
    ons_8  = downsample_freq_erb(ons,    cfs, ACOUSTIC_N_BINS)
    ds     = ACOUSTIC_INTERNAL_SR // FRAME_RATE
    T_out  = gt_8.shape[1] // ds
    gt_8   = gt_8[:,  :T_out * ds].reshape(ACOUSTIC_N_BINS, T_out, ds).mean(2)
    ons_8  = ons_8[:, :T_out * ds].reshape(ACOUSTIC_N_BINS, T_out, ds).mean(2)
    return np.concatenate([gt_8, ons_8], axis=0).T.astype(np.float32)  # (T, 16)


# ─────────────────────────────────────────────────────────────────────────────
# BIDS / MEG helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_events_tsv(evfile: Path) -> pd.DataFrame:
    df   = pd.read_csv(evfile, sep="\t")
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


def get_subjects(bids_root: Path) -> list:
    ptfile = bids_root / "participants.tsv"
    if not ptfile.exists():
        return []
    df = pd.read_csv(ptfile, sep="\t")
    return df["participant_id"].apply(lambda x: x.split("-")[1]).tolist()


def build_acoustic_feature_matrix(run, T_meg, sfreq_model, acoustic_feats):
    X = np.zeros((T_meg, N_ACOUSTIC_PREDICTORS), dtype=np.float64)
    n_placed = 0
    for _, sev in run["sound_events"].iterrows():
        stim_id = sev.get("stim_id", "")
        if stim_id not in acoustic_feats:
            continue
        feat = acoustic_feats[stim_id]
        sf   = int(sev["meg_onset"] * sfreq_model)
        ef   = min(sf + feat.shape[0], T_meg)
        t_kept = ef - sf
        if t_kept <= 0:
            continue
        X[sf:ef, :16] = feat[:t_kept]
        n_placed += 1
    word_ev = run["events_df"][run["events_df"]["kind"] == "word"]
    for _, wev in word_ev.iterrows():
        f_idx = int(wev["meg_onset"] * sfreq_model)
        if 0 <= f_idx < T_meg:
            X[f_idx, 16] = 1.0
    return X, n_placed


def build_feature_matrix(run, layer_label, h5, T_meg, sfreq_model):
    D = h5[list(h5.keys())[0]][layer_label].shape[1]
    X = np.zeros((T_meg, D), dtype=np.float64)
    n_placed = 0
    for _, sev in run["sound_events"].iterrows():
        stim_id = sev.get("stim_id", "")
        if stim_id not in h5 or layer_label not in h5[stim_id]:
            continue
        feat    = h5[stim_id][layer_label][()][ONSET_DELAY:]
        start_f = int(sev["meg_onset"] * sfreq_model)
        end_f   = min(start_f + feat.shape[0], T_meg)
        t_kept  = end_f - start_f
        if t_kept <= 0:
            continue
        X[start_f:end_f] = feat[:t_kept]
        n_placed += 1
    return X, n_placed


# ─────────────────────────────────────────────────────────────────────────────
# Ridge Regression TRF Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_lagged_matrix(X, sfreq, tmin, tmax):
    """Build time-lagged design matrix for TRF: y(t) = sum_tau h(tau) * x(t-tau)."""
    n_samples, n_features = X.shape
    lag_min = int(np.round(tmin * sfreq))
    lag_max = int(np.round(tmax * sfreq))
    lags = np.arange(lag_min, lag_max + 1)
    n_lags = len(lags)

    X_lagged = np.zeros((n_samples, n_features * n_lags), dtype=X.dtype)
    for li, lag in enumerate(lags):
        col_s = li * n_features
        col_e = col_s + n_features
        if lag >= 0:
            X_lagged[lag:, col_s:col_e] = X[:n_samples - lag]
        else:
            X_lagged[:n_samples + lag, col_s:col_e] = X[-lag:]
    return X_lagged, lags


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    SUBJECT   = args.subj
    SESSION   = args.ses
    FEAT_TYPE = args.feat

    print(f"Subject          : {SUBJECT}")
    print(f"Session          : {SESSION}")
    print(f"Feature type     : {FEAT_TYPE}")
    print(f"Layer labels ({N_LAYERS}): {LAYER_LABELS}")
    print(f"BIDS root exists : {BIDS_ROOT.exists()}")
    print(f"Checkpoint exists: {CHECKPOINT_PATH.exists()}")
    print(f"eelbrain version : {eelbrain.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")

    # ── Step 1: Load EarShot model ────────────────────────────────────────────
    model = EarShotModel(
        feature_dim  = FEATURE_DIM,
        n_hidden     = N_HIDDEN,
        target_dim   = TARGET_DIM,
        hidden_type  = HIDDEN_TYPE,
        hidden_reset = HIDDEN_RESET,
        extract_dir  = str(RESULTS_DIR),
        target_mode  = TARGET_MODE,
    )
    ckpt       = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    extractor = LayerWiseExtractor(model).eval().to(device)

    # Sanity check
    top_lstm_label = f"lstm_l{NUM_LAYERS - 1}"
    with torch.no_grad():
        dummy_spec = torch.randn(FEATURE_DIM, 50, device=device)
        feats      = extractor.forward_all_layers(dummy_spec)
        cnn_out    = model.convnet(dummy_spec.unsqueeze(0))
        rnn_in     = cnn_out.transpose(1, 2)
        length     = torch.tensor([50], dtype=torch.long)
        ref_out    = model.get_rnn_output(rnn_in, length).squeeze(0).cpu().numpy()
    top_layer_out = feats[top_lstm_label]
    max_diff = np.abs(top_layer_out - ref_out[:len(top_layer_out)]).max()
    print(f"Sanity check — max |decomposed {top_lstm_label} - original|: {max_diff:.2e}")
    assert max_diff < 1e-4, "Decomposition mismatch! Check weight copying."
    print("Sanity check PASSED ✓")

    # ── Step 2: Collect stimuli ───────────────────────────────────────────────
    stim_map = collect_unique_stimuli([BIDS_ROOT, BIDS_ROOT_P2])
    print(f"Found {len(stim_map)} unique stimuli")

    # ── Step 3: Extract per-layer features to HDF5 ───────────────────────────
    FORCE_RECOMPUTE = False
    if HDF5_PATH.exists() and not FORCE_RECOMPUTE:
        print(f"HDF5 already exists at {HDF5_PATH} — skipping extraction.")
    else:
        if HDF5_PATH.exists():
            os.remove(HDF5_PATH)
        print(f"Extracting features for {len(stim_map)} stimuli → {HDF5_PATH}")
        with h5py.File(HDF5_PATH, "w") as h5:
            h5.attrs["frame_rate"]   = FRAME_RATE
            h5.attrs["onset_delay"]  = ONSET_DELAY
            h5.attrs["layer_labels"] = LAYER_LABELS
            for stim_id, wav_path in tqdm(sorted(stim_map.items())):
                spec   = load_mel_spectrogram(wav_path)
                spec_t = torch.from_numpy(spec).to(device)
                feats  = extractor.forward_all_layers(spec_t)
                grp    = h5.create_group(stim_id)
                grp.attrs["wav_path"] = str(wav_path)
                for label, arr in feats.items():
                    grp.create_dataset(label, data=arr, compression="gzip")
        print("Done.")

    with h5py.File(HDF5_PATH, "r") as h5:
        stim_ids_in_h5 = list(h5.keys())
        example = stim_ids_in_h5[0]
        print(f"HDF5 contains {len(stim_ids_in_h5)} stimuli. Example '{example}' shapes:")
        for lbl in LAYER_LABELS:
            if lbl in h5[example]:
                print(f"  {lbl:10s}: {h5[example][lbl].shape}")

    # ── Step 4: Load MEG data for the selected subject ────────────────────────
    all_bids = [(BIDS_ROOT, get_subjects(BIDS_ROOT)),
                (BIDS_ROOT_P2, get_subjects(BIDS_ROOT_P2))]

    seen_subjects = set()
    subject_roots = []
    for broot, subs in all_bids:
        for sub in subs:
            if sub not in seen_subjects:
                seen_subjects.add(sub)
                subject_roots.append((broot, sub))

    # Filter to requested subject
    subject_roots = [(br, s) for br, s in subject_roots if s == SUBJECT]
    if not subject_roots:
        raise ValueError(
            f"Subject '{SUBJECT}' not found in BIDS data.  "
            f"Available subjects: {sorted(seen_subjects)}"
        )
    print(f"Processing subject: {SUBJECT}")

    # Per-subject results dir for caching
    MEG_CACHE_DIR = RESULTS_DIR / f"meg_cache_sub-{SUBJECT}"
    MEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    run_data      = []
    all_word_rows = []
    h5_file       = h5py.File(HDF5_PATH, "r")

    for bids_root, subj in subject_roots:
        print(f"  Subject {subj} session {SESSION}:", end="")
        for task in range(4):
            evfile = (bids_root / f"sub-{subj}" / f"ses-{SESSION}" / "meg" /
                      f"sub-{subj}_ses-{SESSION}_task-{task}_events.tsv")
            if not evfile.exists():
                continue

            cache_fif = MEG_CACHE_DIR / f"sub-{subj}_ses-{SESSION}_task-{task}_meg-filt.fif"
            if cache_fif.exists():
                raw = mne.io.read_raw_fif(cache_fif, preload=True, verbose=False)
            else:
                bids_path = mne_bids.BIDSPath(
                    subject=subj, session=str(SESSION), task=str(task),
                    datatype="meg", root=bids_root,
                )
                try:
                    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
                except FileNotFoundError:
                    continue
                raw = raw.pick_types(meg=True, misc=False, eeg=False,
                                     eog=False, ecg=False)
                raw.load_data(verbose=False).filter(0.5, 30.0, n_jobs=1,
                                                    verbose=False)
                raw.save(cache_fif, overwrite=True, verbose=False)

            ev       = parse_events_tsv(evfile)
            sound_ev = ev[ev["kind"] == "sound"].reset_index(drop=True)
            word_ev  = ev[ev["kind"] == "word"].reset_index(drop=True)

            run_data.append(dict(
                subject=subj, session=SESSION, task=task,
                raw=raw, events_df=ev, sound_events=sound_ev,
            ))

            for _, wrow in word_ev.iterrows():
                stim_id = wrow.get("stim_id", "")
                if stim_id not in h5_file:
                    continue
                word_start_s = float(wrow.get("start", 0))
                frame_idx    = int(word_start_s * FRAME_RATE) + ONSET_DELAY
                feat_row     = dict(
                    subject   = subj,
                    session   = SESSION,
                    task      = task,
                    stim_id   = stim_id,
                    word      = wrow.get("word", ""),
                    meg_onset = wrow["meg_onset"],
                    frame_idx = frame_idx,
                )
                feat_row["wordfreq"] = zipf_frequency(str(wrow.get("word", "")), "en")
                all_word_rows.append(feat_row)
            print(".", end="")

    h5_file.close()
    print()

    word_df = pd.DataFrame(all_word_rows)
    print(f"Total word tokens collected: {len(word_df)}")

    # ── Step 5: Acoustic features + predictor files ───────────────────────────
    ACOUSTIC_CACHE_PATH       = RESULTS_DIR / "acoustic_features.npz"
    FORCE_RECOMPUTE_ACOUSTIC  = False

    if ACOUSTIC_CACHE_PATH.exists() and not FORCE_RECOMPUTE_ACOUSTIC:
        _cache       = np.load(ACOUSTIC_CACHE_PATH, allow_pickle=True)
        acoustic_feats = {k: _cache[k] for k in _cache.files}
        print(f"Loaded cached acoustic features for {len(acoustic_feats)} stimuli")
    else:
        acoustic_feats = {}
        for stim_id, wav_path in tqdm(sorted(stim_map.items()),
                                      desc="Acoustic features"):
            acoustic_feats[stim_id] = compute_acoustic_predictors(wav_path)
        np.savez_compressed(ACOUSTIC_CACHE_PATH, **acoustic_feats)
        print(f"Computed & saved acoustic features → {ACOUSTIC_CACHE_PATH}")

    tstep    = 1.0 / FRAME_RATE
    freq_dim = Scalar('frequency', np.arange(ACOUSTIC_N_BINS))
    n_saved  = 0

    for stim_id in tqdm(sorted(acoustic_feats), desc="Saving acoustic predictor files"):
        feat     = acoustic_feats[stim_id]
        T_frames = feat.shape[0]
        time_dim = UTS(0, tstep, T_frames)
        gt_ndvar = NDVar(feat[:, :8],  (time_dim, freq_dim), name='gammatone')
        on_ndvar = NDVar(feat[:, 8:16],(time_dim, freq_dim), name='onset')
        eel_save.pickle(gt_ndvar, PREDICTOR_DIR / f"{stim_id}~gammatone-8.pickle")
        eel_save.pickle(on_ndvar, PREDICTOR_DIR / f"{stim_id}~onset-8.pickle")
        n_saved += 1

    with h5py.File(HDF5_PATH, "r") as h5:
        for stim_id in tqdm(sorted(h5.keys()), desc="Saving EarShot predictor files"):
            for layer in LAYER_LABELS:
                if layer not in h5[stim_id]:
                    continue
                feat_raw = h5[stim_id][layer][()][ONSET_DELAY:]
                T_frames = feat_raw.shape[0]
                time_dim = UTS(0, tstep, T_frames)
                comp_dim = Scalar('component', np.arange(feat_raw.shape[1]))
                ndvar    = NDVar(feat_raw.astype(np.float32),
                                 (time_dim, comp_dim), name=f'earshot-{layer}')
                eel_save.pickle(ndvar,
                                PREDICTOR_DIR / f"{stim_id}~earshot-{layer}.pickle")

    n_pred_files = len(list(PREDICTOR_DIR.glob("*.pickle")))
    print(f"Predictor files: {n_pred_files}  (in {PREDICTOR_DIR})")

    # ── Step 6: TRF Fitting (Ridge Regression) ─────────────────────────────────
    print(f"\nTRF fitting started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Feature type: {FEAT_TYPE}")
    print(f"Method: Ridge regression (sklearn RidgeCV)")

    MEG_MODEL_RATE = FRAME_RATE
    TRF_RESULTS_DIR = RESULTS_DIR / f"trf_ridge_sub-{SUBJECT}"
    TRF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RERUN_TRF = True

    need_layer = FEAT_TYPE in ("layer", "combined")

    # ── Build MEG response matrix and predictor matrices ──────────────────────
    h5 = h5py.File(HDF5_PATH, "r") if need_layer else None

    Y_parts, X_acoustic_parts = [], []
    X_layer_parts = {lbl: [] for lbl in LAYER_LABELS} if need_layer else {}

    for rd in run_data:
        raw_ds = rd["raw"].copy().resample(MEG_MODEL_RATE, verbose=False)
        Y_run  = raw_ds.get_data().astype(np.float64)
        T_run  = Y_run.shape[1]
        del raw_ds

        X_aco_run, _ = build_acoustic_feature_matrix(
            rd, T_run, MEG_MODEL_RATE, acoustic_feats)
        n_active = int((np.abs(X_aco_run).sum(1) > 0).sum())
        print(f"  sub-{SUBJECT} ses-{SESSION} task-{rd['task']}: "
              f"active {n_active}/{T_run} ({n_active/T_run*100:.1f}%)")
        if n_active < 200:
            continue
        Y_parts.append(Y_run)
        X_acoustic_parts.append(X_aco_run)
        if need_layer:
            for lbl in LAYER_LABELS:
                X_run, _ = build_feature_matrix(rd, lbl, h5, T_run, MEG_MODEL_RATE)
                X_layer_parts[lbl].append(X_run)

    if not Y_parts:
        print(f"  sub-{SUBJECT} ses-{SESSION}: no valid runs — nothing to fit.")
        if h5 is not None:
            h5.close()
        print("Done.")
        return

    Y_all     = np.concatenate(Y_parts, axis=1)
    X_aco_all = np.concatenate(X_acoustic_parts, axis=0)
    n_sensors, T_total = Y_all.shape
    print(f"  Concatenated: T={T_total}, sensors={n_sensors}")

    # ── Fit TRF based on feature type ─────────────────────────────────────────
    if FEAT_TYPE == "acoustic":
        feat_label = "acoustic"
        n_predictors = N_ACOUSTIC_PREDICTORS
        X_all = X_aco_all
        print(f"  Using acoustic predictors ({n_predictors} features)")

        # Build lagged design matrix
        X_lagged, trf_lags = build_lagged_matrix(X_all, MEG_MODEL_RATE, TRF_TMIN, TRF_TMAX)
        n_lags = len(trf_lags)
        print(f"  Lagged matrix: {X_lagged.shape}  ({n_predictors} features × {n_lags} lags = {X_lagged.shape[1]} columns)")

        save_path = TRF_RESULTS_DIR / f"trf_ridge_{feat_label}_sub-{SUBJECT}_ses-{SESSION}.pkl"

        if save_path.exists() and not RERUN_TRF:
            with open(save_path, "rb") as f:
                trf_result = pickle.load(f)
            result_r = trf_result["r"]
            print(f"  Loaded cached ridge TRF result from {save_path}")
        else:
            Y_pred_all = np.zeros((T_total, n_sensors), dtype=np.float64)
            fold_alphas = []
            fold_size = T_total // N_FOLDS

            print(f"  Fitting {N_FOLDS}-fold cross-validated Ridge TRF...")
            for fold in range(N_FOLDS):
                t0 = fold * fold_size
                t1 = (fold + 1) * fold_size if fold < N_FOLDS - 1 else T_total

                train_mask = np.ones(T_total, dtype=bool)
                train_mask[t0:t1] = False

                ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge.fit(X_lagged[train_mask], Y_all[:, train_mask].T)
                Y_pred_all[t0:t1] = ridge.predict(X_lagged[t0:t1])
                fold_alphas.append(float(ridge.alpha_))
                print(f"    Fold {fold+1}/{N_FOLDS}: best alpha = {ridge.alpha_:.2e}")

            # Per-sensor Pearson r on held-out predictions
            result_r = np.array([
                np.corrcoef(Y_all[i], Y_pred_all[:, i])[0, 1]
                for i in range(n_sensors)
            ])

            # Final fit on all data for TRF kernel extraction
            ridge_full = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge_full.fit(X_lagged, Y_all.T)
            result_h = ridge_full.coef_.reshape(n_sensors, n_lags, n_predictors)

            trf_result = dict(
                r=result_r, h=result_h, lags=trf_lags,
                fold_alphas=fold_alphas, full_alpha=float(ridge_full.alpha_),
                subject=SUBJECT, session=SESSION, feat_type=FEAT_TYPE,
                n_predictors=n_predictors,
            )
            with open(save_path, "wb") as f:
                pickle.dump(trf_result, f)

        r_mean = float(result_r.mean())
        r_max  = float(result_r.max())
        print(f"  Ridge TRF ({feat_label}) — mean r: {r_mean:.4f}, max r: {r_max:.4f}")
        print(f"  Saved → {save_path}")

    elif FEAT_TYPE == "layer":
        for layer in LAYER_LABELS:
            X_layer_all = np.concatenate(X_layer_parts[layer], axis=0)
            n_nonzero   = int((np.abs(X_layer_all).sum(1) > 0).sum())
            print(f"  Layer {layer}: features {X_layer_all.shape}, non-zero: {n_nonzero}")
            if n_nonzero < 500:
                print(f"    ERROR: nearly all-zero features — skipping {layer}.")
                continue

            if X_layer_all.shape[1] > PCA_COMPONENTS:
                active_mask = np.abs(X_layer_all).sum(1) > 0
                pca = PCA(n_components=PCA_COMPONENTS)
                pca.fit(X_layer_all[active_mask])
                X_pca = pca.transform(X_layer_all).astype(np.float64)
                explained = pca.explained_variance_ratio_.sum()
                print(f"  PCA explained variance: {explained:.2%}")
            else:
                X_pca = X_layer_all.astype(np.float64)

            n_predictors = X_pca.shape[1]
            feat_label = layer

            # Build lagged design matrix
            X_lagged, trf_lags = build_lagged_matrix(X_pca, MEG_MODEL_RATE, TRF_TMIN, TRF_TMAX)
            n_lags = len(trf_lags)
            print(f"  Lagged matrix: {X_lagged.shape}  ({n_predictors} features × {n_lags} lags = {X_lagged.shape[1]} columns)")

            save_path = TRF_RESULTS_DIR / f"trf_ridge_{feat_label}_sub-{SUBJECT}_ses-{SESSION}.pkl"

            if save_path.exists() and not RERUN_TRF:
                with open(save_path, "rb") as f:
                    trf_result = pickle.load(f)
                result_r = trf_result["r"]
                print(f"  Loaded cached layer-only TRF for {layer}")
            else:
                Y_pred_all = np.zeros((T_total, n_sensors), dtype=np.float64)
                fold_alphas = []
                fold_size = T_total // N_FOLDS

                print(f"  Fitting {N_FOLDS}-fold cross-validated Ridge TRF ({layer})...")
                for fold in range(N_FOLDS):
                    t0 = fold * fold_size
                    t1 = (fold + 1) * fold_size if fold < N_FOLDS - 1 else T_total

                    train_mask = np.ones(T_total, dtype=bool)
                    train_mask[t0:t1] = False

                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(X_lagged[train_mask], Y_all[:, train_mask].T)
                    Y_pred_all[t0:t1] = ridge.predict(X_lagged[t0:t1])
                    fold_alphas.append(float(ridge.alpha_))
                    print(f"    Fold {fold+1}/{N_FOLDS}: best alpha = {ridge.alpha_:.2e}")

                result_r = np.array([
                    np.corrcoef(Y_all[i], Y_pred_all[:, i])[0, 1]
                    for i in range(n_sensors)
                ])

                ridge_full = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge_full.fit(X_lagged, Y_all.T)
                result_h = ridge_full.coef_.reshape(n_sensors, n_lags, n_predictors)

                trf_result = dict(
                    r=result_r, h=result_h, lags=trf_lags,
                    fold_alphas=fold_alphas, full_alpha=float(ridge_full.alpha_),
                    subject=SUBJECT, session=SESSION, feat_type=layer,
                    n_predictors=n_predictors,
                )
                with open(save_path, "wb") as f:
                    pickle.dump(trf_result, f)

            r_mean = float(result_r.mean())
            r_max  = float(result_r.max())
            print(f"  Layer-only Ridge TRF ({layer}) — mean r: {r_mean:.4f}, max r: {r_max:.4f}")
            print(f"  Saved → {save_path}")

    elif FEAT_TYPE == "combined":
        for layer in LAYER_LABELS:
            X_layer_all = np.concatenate(X_layer_parts[layer], axis=0)
            n_nonzero   = int((np.abs(X_layer_all).sum(1) > 0).sum())
            print(f"  Layer {layer}: features {X_layer_all.shape}, non-zero: {n_nonzero}")
            if n_nonzero < 500:
                print(f"    ERROR: nearly all-zero features — skipping {layer}.")
                continue

            if X_layer_all.shape[1] > PCA_COMPONENTS:
                active_mask = np.abs(X_layer_all).sum(1) > 0
                pca = PCA(n_components=PCA_COMPONENTS)
                pca.fit(X_layer_all[active_mask])
                X_pca = pca.transform(X_layer_all).astype(np.float64)
                explained = pca.explained_variance_ratio_.sum()
                print(f"  PCA explained variance: {explained:.2%}")
            else:
                X_pca = X_layer_all.astype(np.float64)

            # Combine acoustic and layer features
            X_combined = np.concatenate([X_aco_all, X_pca], axis=1)
            n_predictors = X_combined.shape[1]
            feat_label = f"combined_{layer}"

            # Build lagged design matrix
            X_lagged, trf_lags = build_lagged_matrix(X_combined, MEG_MODEL_RATE, TRF_TMIN, TRF_TMAX)
            n_lags = len(trf_lags)
            print(f"  Lagged matrix: {X_lagged.shape}  ({n_predictors} features × {n_lags} lags = {X_lagged.shape[1]} columns)")

            save_path = TRF_RESULTS_DIR / f"trf_ridge_{feat_label}_sub-{SUBJECT}_ses-{SESSION}.pkl"

            if save_path.exists() and not RERUN_TRF:
                with open(save_path, "rb") as f:
                    trf_result = pickle.load(f)
                result_r = trf_result["r"]
                print(f"  Loaded cached combined TRF for {layer}")
            else:
                Y_pred_all = np.zeros((T_total, n_sensors), dtype=np.float64)
                fold_alphas = []
                fold_size = T_total // N_FOLDS

                print(f"  Fitting {N_FOLDS}-fold cross-validated Ridge TRF (acoustic + {layer})...")
                for fold in range(N_FOLDS):
                    t0 = fold * fold_size
                    t1 = (fold + 1) * fold_size if fold < N_FOLDS - 1 else T_total

                    train_mask = np.ones(T_total, dtype=bool)
                    train_mask[t0:t1] = False

                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(X_lagged[train_mask], Y_all[:, train_mask].T)
                    Y_pred_all[t0:t1] = ridge.predict(X_lagged[t0:t1])
                    fold_alphas.append(float(ridge.alpha_))
                    print(f"    Fold {fold+1}/{N_FOLDS}: best alpha = {ridge.alpha_:.2e}")

                result_r = np.array([
                    np.corrcoef(Y_all[i], Y_pred_all[:, i])[0, 1]
                    for i in range(n_sensors)
                ])

                ridge_full = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge_full.fit(X_lagged, Y_all.T)
                result_h = ridge_full.coef_.reshape(n_sensors, n_lags, n_predictors)

                trf_result = dict(
                    r=result_r, h=result_h, lags=trf_lags,
                    fold_alphas=fold_alphas, full_alpha=float(ridge_full.alpha_),
                    subject=SUBJECT, session=SESSION, feat_type=f"combined_{layer}",
                    n_predictors=n_predictors,
                )
                with open(save_path, "wb") as f:
                    pickle.dump(trf_result, f)

            r_mean = float(result_r.mean())
            r_max  = float(result_r.max())
            print(f"  Combined Ridge TRF ({layer}) — mean r: {r_mean:.4f}, max r: {r_max:.4f}")
            print(f"  Saved → {save_path}")

    if h5 is not None:
        h5.close()

    print("Done.")


if __name__ == "__main__":
    main()
