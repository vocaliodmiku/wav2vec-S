"""Défossez-2023-style MEG-MASC baseline ridge-TRF encoding pipeline.

Standalone baseline for anchoring HPSN results against the published
literature. Produces per-subject pkl files compatible with
``aggregate_group.py``.

Conditions (``--feat``)
    acoustic      — gammatone-8 + onset-8 + word-onset (Gwilliams/Brodbeck, 17 feats)
    melspec80     — 80-bin log-mel at the target MEG rate (low-level acoustic)
    w2v2          — facebook/wav2vec2-large-960h hidden states. Sweep over a list
                    of transformer taps with ``--w2v2_layers``. Add
                    ``--random-init-w2v`` to replace the pretrained weights with
                    a freshly random-initialised model (Vaidya/Millet-style
                    untrained control). One ridge fit / pkl per (subject,
                    session, layer).

Layer indexing (HuggingFace ``output_hidden_states`` convention):
    layer 0      — post-CNN, pre-transformer (feature projection output)
    layer 1..24  — outputs of the 24 transformer layers (wav2vec2-large)

Two-phase CLI:
    --compute-feature-only   build the per-stim feature HDF5 (or .npz for
                             ``acoustic``) and exit. Run this once on a node
                             with GPU before sharding fits across subjects.
    --fitting-only           assume the cache is already built; only fit ridge.
                             Safe to run as a slurm array (one subject per task).
    (neither)                run both stages back-to-back.

MEG preprocessing (Défossez 2023):
    * bandpass 0.5–30 Hz
    * resample to target rate (100 Hz with --resample_opt REPR, 50 Hz with MEG)
    * per-channel RobustScaler (subtract median, divide by IQR)
    * clip at ±20 (on the robust-scaled data)
    * MEG cache is a SEPARATE directory tagged ``defossez`` so legacy pkls
      using the old preprocessing cannot be silently mixed with these.

Saved pkls now also include the MEG channel metadata (``ch_names``,
``ch_types``, ``ch_positions``) so downstream ROI grouping does not need to
re-load the BIDS data.

Usage
-----
    # 1. precompute features once (GPU node):
    python -m hpsn.evaluation.baseline_meg_masc \
        --subj 01 --ses 0 --feat w2v2 --resample_opt REPR \
        --compute-feature-only

    # 2. fit ridge per subject (slurm array, no GPU needed):
    python -m hpsn.evaluation.baseline_meg_masc \
        --subj $SUBJ --ses all --feat w2v2 --resample_opt REPR \
        --w2v2_layers all --fitting-only

    # random-init control (separate cache + separate pkls):
    python -m hpsn.evaluation.baseline_meg_masc \
        --subj 01 --ses 0 --feat w2v2 --resample_opt REPR \
        --random-init-w2v --compute-feature-only
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

from .hpsn_ridge import (
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
W2V2_DEFAULT_RANDOM_SEED = 0
# Tap count = config.num_hidden_layers + 1 (layer 0 = post-CNN/pre-transformer).
# Determined from the model config at runtime, so base (12+1=13) and large
# (24+1=25) both work without code changes.

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


def parse_w2v2_layers(spec: str) -> list[int] | str:
    """Parse ``--w2v2_layers`` value.

    The upper bound depends on the model (base = 12, large = 24), which we
    don't know at CLI-parse time. So this returns either the literal string
    ``"all"`` or a sorted list of non-negative ints; ``expand_w2v2_layers``
    later validates against the actual tap count.

    Accepted forms:
        "all"        → "all"  (deferred expansion)
        "0,6,12,24"  → [0, 6, 12, 24]
    """
    s = spec.strip()
    if not s:
        raise argparse.ArgumentTypeError("--w2v2_layers cannot be empty")
    if s.lower() == "all":
        return "all"
    try:
        layers = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--w2v2_layers must be 'all' or a comma-separated list of "
            f"non-negative integers; got {spec!r}"
        ) from e
    bad = [n for n in layers if n < 0]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--w2v2_layers values must be non-negative; got {bad}"
        )
    return layers


def expand_w2v2_layers(raw, n_taps: int) -> list[int]:
    """Resolve the parsed --w2v2_layers value against the actual model size.

    ``raw`` is whatever ``parse_w2v2_layers`` returned ("all" or list[int]).
    ``n_taps`` is ``model.config.num_hidden_layers + 1`` for the loaded model
    (or ``len(h5.attrs['w2v2_layers_present'])`` in --fitting-only mode).
    """
    max_layer = n_taps - 1
    if raw == "all":
        return list(range(n_taps))
    out_of_range = [n for n in raw if n > max_layer]
    if out_of_range:
        print(
            f"--w2v2_layers values out of range for this model "
            f"(max layer = {max_layer}, n_taps = {n_taps}): {out_of_range}",
            file=sys.stderr,
        )
        sys.exit(2)
    return list(raw)


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
        "--w2v2_layers", default="all",
        help=(
            "wav2vec2 taps to fit. 'all' = layers 0..24 "
            "(0 = post-CNN/pre-transformer, 1..24 = transformer outputs). "
            "Or a comma-separated list, e.g. '0,6,12,18,24'. "
            "Has no effect unless --feat w2v2."
        ),
    )
    p.add_argument(
        "--w2v2_model", default=W2V2_MODEL,
        help="HuggingFace id or local path for the wav2vec2 architecture "
             f"(default: {W2V2_MODEL}). For --random-init-w2v this is used "
             "only to load the config; weights are reinitialised.",
    )
    p.add_argument(
        "--random-init-w2v", action="store_true", dest="random_init_w2v",
        help="Replace the pretrained wav2vec2 weights with a random-init "
             "model from the same config (Vaidya/Millet-style untrained "
             "control). Only valid with --feat w2v2.",
    )
    p.add_argument(
        "--w2v2_random_seed", type=int, default=W2V2_DEFAULT_RANDOM_SEED,
        help="Seed used for --random-init-w2v. Stored in HDF5 attrs; "
             "running with a different seed against an existing cache exits.",
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

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--compute-feature-only", action="store_true", dest="compute_feature_only",
        help="Build the feature cache (HDF5 / npz) and exit. Run once before "
             "sharding fits across subjects.",
    )
    mode.add_argument(
        "--fitting-only", action="store_true", dest="fitting_only",
        help="Skip feature extraction; assume the cache is already built and "
             "fail if it is missing or mismatched.",
    )
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
    """Multi-layer wav2vec2 feature extractor.

    A single forward pass returns ``n_taps = num_hidden_layers + 1`` hidden
    states (layer 0 = post-CNN/pre-transformer, layers 1..num_hidden_layers =
    transformer outputs). For wav2vec2-base that's 13 taps; for -large, 25.
    Caching every forward pass at this granularity is cheap because the GPU
    forward is the dominant cost; storing all layers amortises across reruns
    that sweep different layer subsets.
    """

    def __init__(
        self,
        model_name: str = W2V2_MODEL,
        device: torch.device | str | None = None,
        random_init: bool = False,
        random_seed: int = W2V2_DEFAULT_RANDOM_SEED,
    ):
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.random_init = bool(random_init)
        self.random_seed = int(random_seed)

        if self.random_init:
            config = Wav2Vec2Config.from_pretrained(model_name)
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            self.model = Wav2Vec2Model(config)
        else:
            # Force safetensors to bypass the transformers CVE-2025-32434
            # torch.load gate (which fires on legacy pytorch_model.bin even
            # when the env has torch < 2.6).
            self.model = Wav2Vec2Model.from_pretrained(
                model_name, use_safetensors=True,
            )

        self.n_taps = int(self.model.config.num_hidden_layers) + 1
        self.model = self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_all(self, wav: np.ndarray) -> list[np.ndarray]:
        """Return all ``n_taps`` hidden states as a list of ``[T, H]`` float32
        arrays at the native 50 Hz wav2vec2 frame rate."""
        x = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        out = self.model(x, output_hidden_states=True, return_dict=True)
        return [
            hs.squeeze(0).cpu().numpy().astype(np.float32)
            for hs in out.hidden_states
        ]


def _linear_upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation by ``factor`` along the time axis. ``[T, D] → [factor*T, D]``."""
    T, D = x.shape
    t = torch.from_numpy(x.astype(np.float32)).T.unsqueeze(0)          # [1, D, T]
    t = F.interpolate(t, size=T * factor, mode="linear", align_corners=False)
    return t.squeeze(0).T.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature HDF5 cache
# ─────────────────────────────────────────────────────────────────────────────

def _w2v2_layer_key(layer_idx: int) -> str:
    return f"layer_{layer_idx:02d}"


def _w2v2_required_keys(n_taps: int) -> list[str]:
    return [_w2v2_layer_key(i) for i in range(n_taps)]


def _w2v2_n_taps_from_h5(h5: h5py.File) -> int:
    """Read tap count from HDF5 attrs (used in --fitting-only)."""
    present = list(h5.attrs.get("w2v2_layers_present", []))
    if not present:
        return 0
    return int(max(present)) + 1


def _validate_w2v2_h5_attrs(
    h5: h5py.File, frame_rate: int, w2v2_model: str,
    random_init: bool, random_seed: int, hdf5_path: Path,
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
    model_stored = h5.attrs.get("w2v2_model", None)
    if model_stored is not None and str(model_stored) != w2v2_model:
        print(
            f"HDF5 w2v2_model '{model_stored}' ≠ requested '{w2v2_model}'. "
            f"Pass --force_recompute_features or delete {hdf5_path}.",
            file=sys.stderr,
        )
        sys.exit(2)
    rand_stored = h5.attrs.get("random_init", None)
    if rand_stored is not None and bool(rand_stored) != random_init:
        print(
            f"HDF5 random_init={bool(rand_stored)} ≠ requested {random_init}. "
            f"Use a different cache file or delete {hdf5_path}.",
            file=sys.stderr,
        )
        sys.exit(2)
    if random_init:
        seed_stored = h5.attrs.get("random_seed", None)
        if seed_stored is not None and int(seed_stored) != random_seed:
            print(
                f"HDF5 random_seed={seed_stored} ≠ requested {random_seed}. "
                f"Pass --force_recompute_features or delete {hdf5_path}.",
                file=sys.stderr,
            )
            sys.exit(2)


def build_feature_hdf5(
    stim_map: dict, feat_type: str, frame_rate: int,
    hdf5_path: Path, force: bool, device: torch.device | str | None,
    *,
    w2v2_model: str = W2V2_MODEL,
    random_init: bool = False,
    random_seed: int = W2V2_DEFAULT_RANDOM_SEED,
):
    """Write per-stim features to ``hdf5_path``.

    For ``feat_type == 'melspec80'``:  one ``[stim_id]/feat`` dataset per stim.
    For ``feat_type == 'w2v2'``:       one ``[stim_id]/layer_NN`` dataset per
                                       layer NN ∈ [00..n_taps-1] per stim,
                                       where ``n_taps`` is read from the model
                                       config (base = 13, large = 25).
    """
    is_w2v2 = (feat_type == "w2v2")

    # For w2v2 we need to know n_taps before we can decide what's "missing".
    # Instantiate the extractor up front so the tap count is authoritative.
    w2v2 = None
    n_taps = None
    if is_w2v2:
        w2v2 = W2V2Extractor(
            model_name=w2v2_model, device=device,
            random_init=random_init, random_seed=random_seed,
        )
        n_taps = w2v2.n_taps
        print(f"  wav2vec2 model has {n_taps} taps "
              f"(layer 0..{n_taps - 1}; "
              f"num_hidden_layers={n_taps - 1})")

    required_keys = _w2v2_required_keys(n_taps) if is_w2v2 else ["feat"]

    if hdf5_path.exists() and not force:
        with h5py.File(hdf5_path, "r") as h5:
            if is_w2v2:
                _validate_w2v2_h5_attrs(
                    h5, frame_rate, w2v2_model, random_init, random_seed,
                    hdf5_path,
                )
                stored_n_taps = _w2v2_n_taps_from_h5(h5)
                if stored_n_taps and stored_n_taps != n_taps:
                    print(
                        f"HDF5 has {stored_n_taps} taps but the model "
                        f"({w2v2_model}) has {n_taps}. "
                        f"Pass --force_recompute_features or delete "
                        f"{hdf5_path}.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            else:
                fr_stored = h5.attrs.get("frame_rate", None)
                if fr_stored is not None and int(fr_stored) != frame_rate:
                    print(
                        f"HDF5 rate {fr_stored} ≠ requested {frame_rate}. "
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
        h5.attrs["feat_type"] = feat_type
        h5.attrs["frame_rate"] = frame_rate
        if is_w2v2:
            h5.attrs["w2v2_model"] = w2v2_model
            h5.attrs["random_init"] = bool(random_init)
            h5.attrs["random_seed"] = int(random_seed) if random_init else -1
            h5.attrs["w2v2_layers_present"] = list(range(n_taps))

        for stim_id, wav_path in tqdm(items, desc=f"{feat_type} features"):
            wav = load_wav(wav_path)
            if stim_id in h5:
                grp = h5[stim_id]
            else:
                grp = h5.create_group(stim_id)
                grp.attrs["wav_path"] = str(wav_path)

            if feat_type == "melspec80":
                arr = extract_melspec(wav, frame_rate)
                if "feat" in grp:
                    del grp["feat"]
                grp.create_dataset("feat", data=arr, compression="gzip")

            elif feat_type == "w2v2":
                hidden_states = w2v2.extract_all(wav)
                assert len(hidden_states) == n_taps, (
                    f"expected {n_taps} hidden states, got {len(hidden_states)}"
                )
                for layer_idx, arr in enumerate(hidden_states):
                    if frame_rate == 100:
                        arr = _linear_upsample(arr, 2)
                    elif frame_rate != NATIVE_W2V2_RATE:
                        raise ValueError(
                            f"w2v2 frame_rate must be 50 or 100 Hz, got {frame_rate}"
                        )
                    key = _w2v2_layer_key(layer_idx)
                    if key in grp:
                        del grp[key]
                    grp.create_dataset(key, data=arr, compression="gzip")

            else:
                raise ValueError(feat_type)


def build_feat_matrix_from_hdf5(
    run, hdf5: h5py.File, T_target: int, frame_rate: int,
    *, dataset_name: str = "feat",
) -> tuple[np.ndarray, int]:
    """Place per-stim features on the MEG timeline.

    ``dataset_name`` selects which dataset under each stim group to read —
    ``"feat"`` for melspec80, ``"layer_NN"`` for w2v2.
    """
    example = list(hdf5.keys())[0]
    D = hdf5[example][dataset_name].shape[1]
    X = np.zeros((T_target, D), dtype=np.float64)
    n_placed = 0
    for _, sev in run["sound_events"].iterrows():
        stim_id = sev.get("stim_id", "")
        if stim_id not in hdf5 or dataset_name not in hdf5[stim_id]:
            continue
        feat = hdf5[stim_id][dataset_name][()]
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


def _channel_meta_from_raw(raw: mne.io.BaseRaw) -> dict:
    ch_positions = np.array(
        [ch["loc"][:3] for ch in raw.info["chs"]], dtype=np.float64,
    )
    return dict(
        ch_names=list(raw.info["ch_names"]),
        ch_types=list(raw.get_channel_types()),
        ch_positions=ch_positions,
    )


def load_sensor_run(
    bids_root: Path, subj: str, ses: int, task: int,
    target_rate: int, cache_dir: Path,
    *, return_info: bool = False,
):
    """Return ``[n_channels, T]`` at ``target_rate`` Hz with Défossez preprocessing.

    With ``return_info=True`` returns ``(data, channel_meta_dict)``; otherwise
    just ``data`` (back-compat with callers like ``meg_hpsn``).

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
    scaled = robust_scale_per_channel(data)
    if return_info:
        return scaled, _channel_meta_from_raw(raw)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# Per-(subject, session) pipeline
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESSING_TAG = "defossez-bp0.5-30-robust-clip20"


def _feature_label(feat: str, layer_idx: int | None, random_init: bool) -> str:
    """Canonical feature_tag used in pkl filenames and report aggregation."""
    if feat != "w2v2":
        return feat
    prefix = "w2v2rand" if random_init else "w2v2"
    return f"{prefix}_l{layer_idx}"


def _load_runs_with_meta(
    subj: str, ses: int, MEG_RATE: int, meg_cache_dir: Path,
):
    """Load all task runs for one (subj, ses): MEG arrays + events + channel meta.

    Returns ``(runs, channel_meta_or_None)`` where ``runs`` is a list of
    ``(Y, run_dict, T_run)`` and ``channel_meta`` comes from the first
    successfully loaded task (consistent across tasks within a session).
    """
    runs = []
    channel_meta = None
    print(f"  sub-{subj} ses-{ses}:", end="")
    for task in range(4):
        evfile = (BIDS_ROOT / f"sub-{subj}" / f"ses-{ses}" / "meg" /
                  f"sub-{subj}_ses-{ses}_task-{task}_events.tsv")
        if not evfile.exists():
            continue

        loaded = load_sensor_run(
            BIDS_ROOT, subj, ses, task, MEG_RATE, meg_cache_dir,
            return_info=True,
        )
        if loaded is None:
            continue
        Y, ch_meta = loaded
        if channel_meta is None:
            channel_meta = ch_meta
        T_run = Y.shape[1]

        ev = parse_events_tsv(evfile)
        sound_ev = ev[ev["kind"] == "sound"].reset_index(drop=True)
        run = dict(
            subject=subj, session=ses, task=task,
            events_df=ev, sound_events=sound_ev,
        )
        runs.append((Y, run, T_run))
        print(".", end="")
    print()
    return runs, channel_meta


def _build_X_from_runs(
    runs, FEAT: str, MEG_RATE: int, *,
    h5f: h5py.File | None, acoustic_feats: dict | None,
    dataset_name: str | None,
):
    """Build the concatenated (Y, X) for one feature condition."""
    Y_parts, X_parts = [], []
    for Y, run, T_run in runs:
        if FEAT == "acoustic":
            X_run, _ = build_acoustic_feature_matrix(
                run, T_run, MEG_RATE, acoustic_feats,
            )
            n_active = int((np.abs(X_run).sum(1) > 0).sum())
            if n_active < 200:
                print(f"  [skip task-{run['task']}: {n_active} active]")
                continue
        else:
            X_run, n_placed = build_feat_matrix_from_hdf5(
                run, h5f, T_run, MEG_RATE, dataset_name=dataset_name,
            )
            if n_placed == 0:
                print(f"  [skip task-{run['task']}: no features placed]")
                continue
        Y_parts.append(Y)
        X_parts.append(X_run)

    if not Y_parts:
        return None
    Y_all = np.concatenate(Y_parts, axis=1)
    X_all = np.concatenate(X_parts, axis=0)
    return Y_all, X_all


def _fit_one_condition(
    args, *, Y_all, X_all, feat_label, RESAMPLE_OPT, MEG_RATE,
    subj, ses, save_path, channel_meta,
) -> str:
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
        )
        print(f"    mean r = {r_values.mean():.4f}   max r = {r_values.max():.4f}")

    if channel_meta is not None:
        result["ch_names"] = channel_meta["ch_names"]
        result["ch_types"] = channel_meta["ch_types"]
        result["ch_positions"] = channel_meta["ch_positions"]

    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved → {save_path}")
    return "done"


def fit_one_subject_session(
    args, subj: str, ses: int, *,
    h5f, acoustic_feats, stim_map,
    FEAT: str, MEG_RATE: int, RESAMPLE_OPT: str,
    results_dir: Path,
    w2v2_layers: list[int] | None,
    random_init: bool,
) -> dict[str, int]:
    """Run the MEG-align → concat → PCA → ridge → save pipeline.

    For w2v2 this loops over every requested layer (each producing its own
    pkl); MEG runs are loaded once per (subj, ses) and reused across layers.

    Returns a dict counting per-condition statuses.
    """
    save_dir = results_dir / f"trf_ridge_sub-{subj}"
    save_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "cached": 0, "no_runs": 0, "skip": 0}

    if FEAT == "w2v2":
        conditions = [
            (
                _feature_label(FEAT, layer_idx, random_init),
                _w2v2_layer_key(layer_idx),
            )
            for layer_idx in w2v2_layers
        ]
    else:
        conditions = [(_feature_label(FEAT, None, False), "feat")]

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
            runs, FEAT, MEG_RATE,
            h5f=h5f, acoustic_feats=acoustic_feats,
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

def _resolve_hdf5_path(args, results_dir: Path, FEAT: str, RESAMPLE_OPT: str) -> Path | None:
    if FEAT == "acoustic":
        return None
    if args.hdf5_path:
        return Path(args.hdf5_path)
    if FEAT == "w2v2":
        tag = "w2v2rand" if args.random_init_w2v else "w2v2"
        return results_dir / f"baseline_features_{tag}_{RESAMPLE_OPT}.h5"
    # melspec80
    return results_dir / f"baseline_features_{FEAT}_{RESAMPLE_OPT}.h5"


def run_compute_features(
    args, *, FEAT: str, MEG_RATE: int, RESAMPLE_OPT: str,
    results_dir: Path, hdf5_path: Path | None,
    stim_map: dict, device,
) -> dict | None:
    """Build the feature cache. Returns the in-memory acoustic_feats dict
    (when applicable) or None."""
    if FEAT in ("melspec80", "w2v2"):
        log_stage(f"Stage feature: building HDF5 features ({FEAT})")
        build_feature_hdf5(
            stim_map,
            feat_type=FEAT,
            frame_rate=MEG_RATE,
            hdf5_path=hdf5_path,
            force=args.force_recompute_features,
            device=device,
            w2v2_model=args.w2v2_model,
            random_init=args.random_init_w2v,
            random_seed=args.w2v2_random_seed,
        )
        return None

    # acoustic — npz cache
    log_stage("Stage feature: building acoustic feature cache")
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
    return acoustic_feats


def load_existing_features(
    args, *, FEAT: str, MEG_RATE: int, RESAMPLE_OPT: str,
    results_dir: Path, hdf5_path: Path | None,
) -> dict | None:
    """For --fitting-only: open the cache read-only, validate, and return
    the in-memory acoustic_feats dict (or None for HDF5-cached feats).

    Per-layer presence is validated downstream by ``expand_w2v2_layers``
    (which has access to the n_taps stored in HDF5 attrs).
    """
    if FEAT == "acoustic":
        aco_cache = results_dir / f"acoustic_features_{MEG_RATE}Hz.npz"
        if not aco_cache.exists():
            print(
                f"--fitting-only: acoustic cache missing at {aco_cache}. "
                f"Run --compute-feature-only first.",
                file=sys.stderr,
            )
            sys.exit(2)
        data = np.load(aco_cache, allow_pickle=True)
        return {k: data[k] for k in data.files}

    if hdf5_path is None or not hdf5_path.exists():
        print(
            f"--fitting-only: HDF5 cache missing at {hdf5_path}. "
            f"Run --compute-feature-only first.",
            file=sys.stderr,
        )
        sys.exit(2)

    with h5py.File(hdf5_path, "r") as h5:
        if FEAT == "w2v2":
            _validate_w2v2_h5_attrs(
                h5, MEG_RATE, args.w2v2_model,
                args.random_init_w2v, args.w2v2_random_seed, hdf5_path,
            )
        else:
            fr_stored = h5.attrs.get("frame_rate", None)
            if fr_stored is not None and int(fr_stored) != MEG_RATE:
                print(
                    f"--fitting-only: HDF5 frame_rate={fr_stored} ≠ "
                    f"requested {MEG_RATE}. Re-run --compute-feature-only.",
                    file=sys.stderr,
                )
                sys.exit(2)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    FEAT = args.feat
    RESAMPLE_OPT = args.resample_opt
    MEG_RATE = 100 if RESAMPLE_OPT == "REPR" else 50

    if args.random_init_w2v and FEAT != "w2v2":
        print(
            "ERROR: --random-init-w2v is only valid with --feat w2v2.",
            file=sys.stderr,
        )
        sys.exit(2)

    if FEAT == "w2v2":
        w2v2_layers = parse_w2v2_layers(args.w2v2_layers)
    else:
        w2v2_layers = None

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = _resolve_hdf5_path(args, results_dir, FEAT, RESAMPLE_OPT)

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
    print(f"Feature       : {FEAT}")
    if FEAT == "w2v2":
        print(f"  layers      : {w2v2_layers}")
        print(f"  model       : {args.w2v2_model}")
        print(f"  random init : {args.random_init_w2v} "
              f"(seed={args.w2v2_random_seed})")
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

    log_stage(f"Stage 3/4: Feature cache ({run_mode})")
    if run_mode in ("compute-feature-only", "compute+fit"):
        acoustic_feats = run_compute_features(
            args,
            FEAT=FEAT, MEG_RATE=MEG_RATE, RESAMPLE_OPT=RESAMPLE_OPT,
            results_dir=results_dir, hdf5_path=hdf5_path,
            stim_map=stim_map, device=device,
        )
    else:  # fitting-only
        acoustic_feats = load_existing_features(
            args,
            FEAT=FEAT, MEG_RATE=MEG_RATE, RESAMPLE_OPT=RESAMPLE_OPT,
            results_dir=results_dir, hdf5_path=hdf5_path,
        )

    if run_mode == "compute-feature-only":
        log_stage("Done (feature cache built; --compute-feature-only).")
        return

    # Now that the HDF5 exists (either freshly built or pre-existing), expand
    # --w2v2_layers against the actual tap count of the cached model.
    if FEAT == "w2v2":
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

    h5f = h5py.File(hdf5_path, "r") if (hdf5_path is not None) else None

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
                    h5f=h5f, acoustic_feats=acoustic_feats, stim_map=stim_map,
                    FEAT=FEAT,
                    MEG_RATE=MEG_RATE, RESAMPLE_OPT=RESAMPLE_OPT,
                    results_dir=results_dir,
                    w2v2_layers=w2v2_layers,
                    random_init=args.random_init_w2v,
                )
                for k, v in counts.items():
                    grand_counts[k] += v
    finally:
        if h5f is not None:
            h5f.close()

    log_stage("Done")
    print(f"  fit              : {grand_counts['done']}")
    print(f"  cached (skipped) : {grand_counts['cached']}")
    print(f"  no runs found    : {grand_counts['no_runs']}")
    print(f"  empty/skipped    : {grand_counts['skip']}")


if __name__ == "__main__":
    main()
