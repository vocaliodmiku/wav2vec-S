"""HPSN feature extractor for MEG-MASC evaluation.

Runs each stimulus through the frozen wav2vec-S backbone and the trained
3-level HPSN module (eval mode, masking bypassed) and emits six conditions
per stimulus:

    baseline_low  : [T, H]   — L1-band tap (acoustic, e.g. layers 1-4)
    baseline_mid  : [T, H]   — L2-band tap (lexical,  e.g. layers 5-8)
    baseline_high : [T, H]   — L3-band tap (semantic, e.g. layers 9-12)
    hpsn_l1       : [T, D]   — Level-1 representation  (D = lstm_dim, 512)
    hpsn_l2       : [T, D]   — Level-2 representation
    hpsn_l3       : [T, D]   — Level-3 representation

Native wav2vec-S frame rate is 50 Hz (hop = 320 samples at 16 kHz).  When
``resample_opt="REPR"`` the features are upsampled 2× via linear
interpolation to 100 Hz to match the MEG-MASC pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import HPSNConfig
from ..model.backbone import FrozenWav2VecS
from ..model.hpsn import HPSN


SR_DEFAULT = 16_000
NATIVE_FRAME_RATE = 50  # Hz


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_state_dict_file(ckpt_path: Path) -> Path:
    """Locate HPSN state_dict file inside an ``accelerate.save_state`` dir."""
    if ckpt_path.is_file():
        return ckpt_path
    # Accelerate typically writes ``pytorch_model.bin`` or ``model.safetensors``.
    candidates = [
        ckpt_path / "pytorch_model.bin",
        ckpt_path / "model.safetensors",
        ckpt_path / "hpsn.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: any .bin / .safetensors in the directory.
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        hits = sorted(ckpt_path.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No HPSN state_dict file found under {ckpt_path} "
        f"(looked for pytorch_model.bin, model.safetensors, hpsn.pt)"
    )


def _load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return obj


def load_hpsn_from_checkpoint(
    ckpt_path: str | os.PathLike,
    config: HPSNConfig,
    device: torch.device | str = "cpu",
) -> HPSN:
    """Instantiate ``HPSN`` and load weights from ``ckpt_path``."""
    sd_file = _find_state_dict_file(Path(ckpt_path))
    state_dict = _load_state_dict(sd_file)

    hpsn = HPSN(config)
    missing, unexpected = hpsn.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[features] WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[features] WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    hpsn.eval().to(device)
    return hpsn


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class HPSNFeatureExtractor:
    """Emit per-stim HPSN condition features for MEG-MASC.

    Parameters
    ----------
    config         : HPSNConfig already populated with the correct backbone
                     path, tap ranges, etc.
    ckpt_path      : directory (accelerate checkpoint) or file with an HPSN
                     state_dict.
    resample_opt   : 'MEG'  → keep native 50 Hz.
                     'REPR' → linear-upsample to 100 Hz.
    device         : torch device.
    """

    def __init__(
        self,
        config: HPSNConfig,
        ckpt_path: str | os.PathLike,
        resample_opt: str = "MEG",
        device: Optional[torch.device | str] = None,
    ):
        assert resample_opt in ("MEG", "REPR"), f"Bad resample_opt: {resample_opt}"
        self.config = config
        self.resample_opt = resample_opt
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.backbone = FrozenWav2VecS(
            config.backbone_model,
            main_context=config.main_context,
            right_context=config.right_context,
            dtype=torch.bfloat16,
        ).to(self.device)

        self.hpsn = load_hpsn_from_checkpoint(ckpt_path, config, self.device)
        self.hpsn.eval()
        for p in self.hpsn.parameters():
            p.requires_grad = False

        # frame_rate reported by this extractor
        self.frame_rate = 100 if resample_opt == "REPR" else NATIVE_FRAME_RATE

    # ── core forward ────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """Run a mono 16 kHz waveform through backbone + HPSN, return 6 conditions.

        Returns
        -------
        dict with keys {baseline_low, baseline_mid, baseline_high,
        hpsn_l1, hpsn_l2, hpsn_l3}, each ``np.float32`` of shape
        ``[T, D_cond]``.
        """
        if waveform.ndim != 1:
            waveform = waveform.squeeze()
            assert waveform.ndim == 1, f"Expected mono, got shape {waveform.shape}"

        x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(self.device)

        hidden_states = self.backbone(x)  # tuple, each [1, T, H], bf16

        # Cast to fp32 for HPSN (trained fp32)
        hs_fp32 = tuple(h.float() for h in hidden_states)

        # Per-band taps (baseline conditions) — learned ELMo-style scalar mixes.
        # The 3-level model exposes them as tap1 (L1 band), tap2 (L2 band),
        # tap3 (L3 band).
        baseline_low = self.hpsn.tap1(hs_fp32)   # [1, T, H]
        baseline_mid = self.hpsn.tap2(hs_fp32)   # [1, T, H]
        baseline_high = self.hpsn.tap3(hs_fp32)  # [1, T, H]

        # Full HPSN forward (masking bypassed because .eval())
        out = self.hpsn(hs_fp32, attention_mask=None)
        hpsn_l1 = out["level1_repr"]   # [1, T, D]
        hpsn_l2 = out["level2_repr"]   # [1, T, D]
        hpsn_l3 = out["level3_repr"]   # [1, T, D]

        feats = {
            "baseline_low": baseline_low,
            "baseline_mid": baseline_mid,
            "baseline_high": baseline_high,
            "hpsn_l1": hpsn_l1,
            "hpsn_l2": hpsn_l2,
            "hpsn_l3": hpsn_l3,
        }

        if self.resample_opt == "REPR":
            feats = {k: _linear_upsample_2x(v) for k, v in feats.items()}

        # [1, T, D] → [T, D] np.float32
        return {k: v.squeeze(0).cpu().numpy().astype(np.float32) for k, v in feats.items()}


def _linear_upsample_2x(x: torch.Tensor) -> torch.Tensor:
    """Linear interpolation 2× along the time axis.  Shape [B, T, D] → [B, 2T, D]."""
    B, T, D = x.shape
    y = x.transpose(1, 2)                                 # [B, D, T]
    y = F.interpolate(y, size=2 * T, mode="linear", align_corners=False)
    return y.transpose(1, 2)                              # [B, 2T, D]


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = (
    "baseline_low", "baseline_mid", "baseline_high",
    "hpsn_l1", "hpsn_l2", "hpsn_l3",
)


def extract_to_hdf5(
    stim_map: Dict[str, Path],
    extractor: HPSNFeatureExtractor,
    hdf5_path: str | os.PathLike,
    sample_rate: int = SR_DEFAULT,
    force: bool = False,
    progress: bool = True,
) -> None:
    """Write features for every (stim_id → wav_path) into a single HDF5 file."""
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if hdf5_path.exists() and not force:
        # Check for completeness: missing stims will be appended.
        with h5py.File(hdf5_path, "r") as h5:
            existing = set(h5.keys())
        missing = {k: v for k, v in stim_map.items() if k not in existing}
        if not missing:
            print(f"[features] HDF5 up-to-date at {hdf5_path} "
                  f"({len(existing)} stimuli) — skipping.")
            return
        print(f"[features] Appending {len(missing)}/{len(stim_map)} missing stimuli "
              f"to {hdf5_path}")
        stim_map = missing
        mode = "a"
    else:
        if hdf5_path.exists():
            os.remove(hdf5_path)
        mode = "w"

    items = sorted(stim_map.items())
    iterator = tqdm(items, desc="HPSN features") if progress else items

    with h5py.File(hdf5_path, mode) as h5:
        h5.attrs["frame_rate"] = extractor.frame_rate
        h5.attrs["resample_opt"] = extractor.resample_opt
        h5.attrs["backbone_model"] = extractor.config.backbone_model
        h5.attrs["hidden_dim"] = extractor.config.hidden_dim
        h5.attrs["lstm_dim"] = extractor.config.lstm_dim
        h5.attrs["conditions"] = list(CONDITIONS)

        for stim_id, wav_path in iterator:
            try:
                wav, sr = sf.read(str(wav_path))
            except Exception as e:
                print(f"[features] failed to read {wav_path}: {e}")
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != sample_rate:
                # Lightweight resample via torchaudio.
                import torchaudio
                wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
                wav_t = torchaudio.transforms.Resample(sr, sample_rate)(wav_t)
                wav = wav_t.squeeze(0).numpy()

            feats = extractor.extract(wav)
            grp = h5.create_group(stim_id)
            grp.attrs["wav_path"] = str(wav_path)
            for name, arr in feats.items():
                grp.create_dataset(name, data=arr, compression="gzip")
