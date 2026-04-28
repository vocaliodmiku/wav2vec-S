"""LibriSpeech folder dataset + feature-extractor collate (audio only)."""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

# Frame rate / hop assumed by the HPSN-v2 targets HDF5 (set by Stage 1
# extract_targets.py). Kept here so data.py is self-contained and not coupled
# to the data_prep package, which has its own runtime dependencies.
TARGETS_FRAME_RATE = 50
TARGETS_HOP_LENGTH = 320  # = 16000 // 50

TARGET_FIELDS = (
    "log_mel",            # (T, 80)  float16
    "phonol_features",    # (T, 14)  float16
    "phone_id",           # (T,)     uint8
    "word_id",            # (T,)     int32
    "gpt2_l4",            # (T, 768) float16
    "gpt2_l8",            # (T, 768) float16
)


class LibriSpeechFolder(Dataset):
    """Walks a LibriSpeech split (expects ``*/*/*.trans.txt``).  Returns raw waveforms only."""

    def __init__(
        self,
        root: str,
        sample_rate: int = 16000,
        max_audio_seconds: float = 15.0,
        max_samples: int = 0,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.max_len = int(max_audio_seconds * sample_rate)

        self.paths: List[str] = []
        trans_files = sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt")))
        if not trans_files:
            raise FileNotFoundError(f"No *.trans.txt found under {root}")
        for tf in trans_files:
            chapter_dir = os.path.dirname(tf)
            with open(tf) as f:
                for line in f:
                    utt_id = line.strip().split(" ", 1)[0]
                    self.paths.append(os.path.join(chapter_dir, utt_id + ".flac"))
        if max_samples and max_samples > 0:
            self.paths = self.paths[:max_samples]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        waveform, sr = sf.read(self.paths[idx], dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz, got {sr} at {self.paths[idx]}")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if len(waveform) > self.max_len:
            waveform = waveform[: self.max_len]
        return waveform


@dataclass
class Collator:
    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, batch):
        proc = self.feature_extractor(
            list(batch),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return {
            "input_values": proc["input_values"],
            "attention_mask": proc["attention_mask"],
        }


def build_dataloader(
    root: str,
    feature_extractor: Wav2Vec2FeatureExtractor,
    batch_size: int,
    num_workers: int = 4,
    max_audio_seconds: float = 15.0,
    max_samples: int = 0,
    shuffle: bool = True,
    sample_rate: int = 16000,
):
    dataset = LibriSpeechFolder(
        root=root,
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
        max_samples=max_samples,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collator(feature_extractor),
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HPSN-v2 targets-aware dataset
# ─────────────────────────────────────────────────────────────────────────────

class TargetsHDF5Dataset(Dataset):
    """Per-utterance loader yielding ``(waveform, targets, transcript, utt_id)``.

    Indexes utterances from the manifest CSV produced by Phase 0
    ``build_manifest.py``; reads the per-utterance target stack from the keyed
    HDF5 produced by Phase 1 ``extract_targets.py``. Audio and target frames
    are cropped to a common length so frame ``i`` of every target tensor
    aligns with samples ``[i*hop, (i+1)*hop)`` of the waveform.

    Notes on parallel loading
    -------------------------
    h5py file handles are not picklable across DataLoader workers, so the
    handle is opened lazily inside ``__getitem__`` per worker (cached on the
    instance after first call). Each worker therefore holds one open file.
    """

    def __init__(
        self,
        manifest_path: str,
        targets_h5_path: str,
        sample_rate: int = 16000,
        max_audio_seconds: float = 15.0,
        max_samples: int = 0,
        target_fields: tuple[str, ...] = TARGET_FIELDS,
        require_transcript: bool = True,
        stats_path: str | None = None,
        restore_prob: float = 0.0,
    ):
        if sample_rate != 16000:
            raise ValueError(f"sample_rate must be 16000 (HDF5 was built at 16kHz); got {sample_rate}")
        self.manifest_path = manifest_path
        self.targets_h5_path = targets_h5_path
        self.sample_rate = sample_rate
        self.hop = TARGETS_HOP_LENGTH
        self.frame_rate = TARGETS_FRAME_RATE
        # Round max audio length down to a frame boundary so audio-vs-frame
        # cropping never has to round.
        self.max_frames = int(max_audio_seconds * self.frame_rate)
        self.max_samples_audio = self.max_frames * self.hop
        self.target_fields = tuple(target_fields)
        self.require_transcript = bool(require_transcript)
        self.restore_prob = float(restore_prob)
        if self.restore_prob > 0 and "phone_id" not in self.target_fields:
            raise ValueError(
                "restore_prob > 0 requires 'phone_id' in target_fields "
                "(needed to locate phoneme spans for noise replacement)"
            )

        # Index from manifest. Validate target HDF5 once on the main process so
        # we fail fast (workers will re-open it lazily).
        self.rows: list[dict] = []
        with open(manifest_path) as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise RuntimeError(f"manifest {manifest_path} is empty")

        with h5py.File(targets_h5_path, "r") as h5:
            attrs = dict(h5.attrs)
            for need in ("frame_rate", "hop_length", "sr"):
                if need not in attrs:
                    raise RuntimeError(f"{targets_h5_path} missing file attr {need!r}")
            if int(attrs["frame_rate"]) != self.frame_rate:
                raise RuntimeError(
                    f"HDF5 frame_rate={int(attrs['frame_rate'])} ≠ expected {self.frame_rate}"
                )
            if int(attrs["hop_length"]) != self.hop:
                raise RuntimeError(
                    f"HDF5 hop_length={int(attrs['hop_length'])} ≠ expected {self.hop}"
                )
            if int(attrs["sr"]) != self.sample_rate:
                raise RuntimeError(
                    f"HDF5 sr={int(attrs['sr'])} ≠ expected {self.sample_rate}"
                )
            # Spot-check that the first manifest row exists in the HDF5.
            first_id = self.rows[0]["utt_id"]
            if first_id not in h5:
                raise RuntimeError(
                    f"manifest references utt_id {first_id!r} not present in {targets_h5_path}"
                )

        # Optional per-dim normalization (z-score) for any 2-D target field.
        # Loaded once on the main process — small numpy arrays, cheap to share
        # across workers via fork/spawn.
        self._norm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if stats_path is not None:
            stats = np.load(stats_path)
            for fname in self.target_fields:
                k_mean, k_std = f"{fname}_mean", f"{fname}_std"
                if k_mean in stats.files and k_std in stats.files:
                    self._norm[fname] = (
                        stats[k_mean].astype(np.float32),
                        stats[k_std].astype(np.float32),
                    )

        # Per-worker handle; created lazily in __getitem__.
        self._h5 = None

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return len(self.rows)

    def _open_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.targets_h5_path, "r", swmr=False)
        return self._h5

    def _crop_lengths(self, n_frames_avail: int, n_samples_avail: int) -> tuple[int, int]:
        """Common-length crop honoring max_audio_seconds and consistency."""
        n_frames = min(n_frames_avail, self.max_frames)
        # Audio may be slightly longer than n_frames * hop because of librosa's
        # right-padding. Take the conservative min so we never read past EOF.
        n_samples = min(n_samples_avail, n_frames * self.hop)
        # Re-derive frames from the actual sample count (may shrink by 1 frame
        # if audio is shorter than the targets reported).
        n_frames = n_samples // self.hop
        return n_frames, n_frames * self.hop

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        utt_id = row["utt_id"]

        # 1. Audio
        wav, sr = sf.read(row["wav_path"], dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"{row['wav_path']}: expected {self.sample_rate} Hz, got {sr}")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)

        # 2. Targets
        h5 = self._open_h5()
        if utt_id not in h5:
            raise KeyError(f"utt_id {utt_id!r} not in {self.targets_h5_path}")
        grp = h5[utt_id]
        n_frames_avail = int(grp.attrs["n_frames"])

        # 3. Common crop
        n_frames, n_samples = self._crop_lengths(n_frames_avail, wav.shape[0])
        if n_frames <= 0:
            raise RuntimeError(
                f"{utt_id}: zero usable frames "
                f"(n_frames_avail={n_frames_avail}, n_samples_wav={wav.shape[0]})"
            )
        wav = wav[:n_samples]

        targets: dict[str, np.ndarray] = {}
        for name in self.target_fields:
            if name not in grp:
                raise KeyError(f"{utt_id}: missing target {name!r} in HDF5")
            arr = grp[name][:n_frames]
            # Cast f16 → f32 here so downstream model code never has to.
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            # Apply per-dim z-score on 2-D fields when stats are loaded; only
            # normalize frames whose target row is non-zero so silence/padding
            # frames stay at zero (which is what the loss treats as "ignore").
            if name in self._norm and arr.ndim == 2:
                mean, std = self._norm[name]
                nonzero = np.any(arr != 0, axis=1)
                if nonzero.any():
                    arr_norm = (arr[nonzero] - mean) / std
                    arr = np.zeros_like(arr)
                    arr[nonzero] = arr_norm.astype(np.float32)
                else:
                    arr = np.zeros_like(arr)
            targets[name] = arr

        # L_restore corruption (full_hpsn_construction.md §19.3). Picks one
        # non-silence phoneme span uniformly at random; replaces those audio
        # samples with RMS-matched white noise. The clean log-mel target on
        # those frames is left untouched (it was precomputed in the H5),
        # so the loss can ask L1 to predict the original phoneme from a
        # waveform that no longer contains it — solvable only via L2→L1
        # cross-attention.
        restore_mask = np.zeros(int(n_frames), dtype=bool)
        if self.restore_prob > 0.0 and np.random.rand() < self.restore_prob:
            pid = targets.get("phone_id")
            if pid is not None and pid.shape[0] > 0:
                pid_int = pid.astype(np.int64, copy=False)
                # Run boundaries: True at the first frame of each new run.
                prev = np.concatenate([[pid_int[0] - 1], pid_int[:-1]])
                boundary = pid_int != prev
                starts = np.flatnonzero(boundary)
                ends = np.concatenate([starts[1:], [pid_int.shape[0]]])
                run_ids = pid_int[starts]
                cand = np.flatnonzero(run_ids > 0)  # skip silence (id == 0)
                if cand.size > 0:
                    pick = int(np.random.choice(cand))
                    s_frame = int(starts[pick])
                    e_frame = int(ends[pick])
                    s_sample = s_frame * self.hop
                    e_sample = min(e_frame * self.hop, wav.shape[0])
                    if e_sample > s_sample:
                        seg = wav[s_sample:e_sample]
                        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
                        # Copy first — sf.read may return a memmap view.
                        wav = wav.copy()
                        wav[s_sample:e_sample] = (
                            np.random.randn(e_sample - s_sample).astype(np.float32) * rms
                        )
                        restore_mask[s_frame:e_frame] = True

        out = {
            "utt_id": utt_id,
            "waveform": wav.astype(np.float32, copy=False),
            "n_frames": int(n_frames),
            "n_samples": int(n_samples),
            "targets": targets,
            "restore_mask": restore_mask,
        }
        if self.require_transcript:
            out["transcript"] = str(grp.attrs.get("transcript", ""))
        return out


@dataclass
class TargetsCollator:
    """Pad-to-batch-max collator for ``TargetsHDF5Dataset`` outputs.

    Pads waveforms with zeros to the longest audio in the batch and target
    arrays with zeros to the longest frame count. Returns:

      * ``input_values``     [B, S_max]            float32
      * ``attention_mask``   [B, S_max]            int64 (1=valid sample)
      * ``frame_mask``       [B, T_max]            bool  (True=valid frame)
      * ``targets``          dict[str → [B, T_max, ...] tensor]
      * ``utt_ids``          list[str]
      * ``transcripts``      list[str]   (only if dataset.require_transcript)
    """

    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, batch: list[dict]) -> dict:
        wavs = [b["waveform"] for b in batch]
        n_frames = [b["n_frames"] for b in batch]
        T_max = max(n_frames)

        proc = self.feature_extractor(
            wavs,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        # Frame-level mask (True for valid frames, False for padding).
        frame_mask = torch.zeros(len(batch), T_max, dtype=torch.bool)
        for i, t in enumerate(n_frames):
            frame_mask[i, :t] = True

        # Target tensors padded to T_max with zeros.
        target_fields = list(batch[0]["targets"].keys())
        targets_out: dict[str, torch.Tensor] = {}
        for name in target_fields:
            sample = batch[0]["targets"][name]
            if sample.ndim == 1:
                arr = np.zeros((len(batch), T_max), dtype=sample.dtype)
                for i, b in enumerate(batch):
                    a = b["targets"][name]
                    arr[i, : a.shape[0]] = a
            elif sample.ndim == 2:
                D = sample.shape[1]
                arr = np.zeros((len(batch), T_max, D), dtype=sample.dtype)
                for i, b in enumerate(batch):
                    a = b["targets"][name]
                    arr[i, : a.shape[0]] = a
            else:
                raise ValueError(f"target {name!r} has unsupported ndim={sample.ndim}")
            targets_out[name] = torch.from_numpy(arr)

        out = {
            "input_values": proc["input_values"],
            "attention_mask": proc["attention_mask"],
            "frame_mask": frame_mask,
            "targets": targets_out,
            "utt_ids": [b["utt_id"] for b in batch],
        }
        if "restore_mask" in batch[0]:
            restore_mask = torch.zeros(len(batch), T_max, dtype=torch.bool)
            for i, b in enumerate(batch):
                rm = b["restore_mask"]
                restore_mask[i, : rm.shape[0]] = torch.from_numpy(rm)
            out["restore_mask"] = restore_mask
        if "transcript" in batch[0]:
            out["transcripts"] = [b["transcript"] for b in batch]
        return out


def build_targets_dataloader(
    manifest_path: str,
    targets_h5_path: str,
    feature_extractor: Wav2Vec2FeatureExtractor,
    batch_size: int,
    num_workers: int = 4,
    max_audio_seconds: float = 15.0,
    max_samples: int = 0,
    shuffle: bool = True,
    target_fields: tuple[str, ...] = TARGET_FIELDS,
    stats_path: str | None = None,
    restore_prob: float = 0.0,
):
    dataset = TargetsHDF5Dataset(
        manifest_path=manifest_path,
        targets_h5_path=targets_h5_path,
        max_audio_seconds=max_audio_seconds,
        max_samples=max_samples,
        target_fields=target_fields,
        stats_path=stats_path,
        restore_prob=restore_prob,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TargetsCollator(feature_extractor),
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
