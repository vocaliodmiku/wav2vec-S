"""Wav2Vec-S frontend wrapper with multi-layer taps.

Loads `biaofu-xmu/wav2vec-S-Large-ft-960h`, pins (main_context, right_context)
to (8, 4), freezes all parameters by default, and exposes hidden states at
pre-configured transformer layer indices (1-indexed per proposal §4.2).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..config import FrontendConfig

# Make wav2vec-S-hf importable without installing it as a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _REPO_ROOT / "wav2vec-S-hf"
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

from wav2vec_s.modeling_wav2vec_s import Wav2VecSModel  # noqa: E402


class Wav2VecSFrontend(nn.Module):
    """Frozen wav2vec-S encoder exposing taps at configured layers.

    Returns a dict mapping `f"tap{layer}"` -> tensor of shape (B, T, hidden_size).
    Under `m=8, r=4`, per-frame lookahead varies in [4, 11] frames (80-220 ms);
    this is a known property of wav2vec-S and handled downstream by MEG TRF
    covariates, not here.
    """

    def __init__(self, cfg: FrontendConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Wav2VecSModel.from_pretrained(cfg.model_name)
        # Pin streaming context to the minimum in-distribution configuration.
        self.model.encoder.main_context = cfg.main_context
        self.model.encoder.right_context = cfg.right_context
        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        # Keep frozen frontend in eval mode regardless of outer .train() calls
        # so dropout/BN on the frontend don't activate.
        super().train(mode)
        if self.cfg.freeze:
            self.model.eval()
        return self

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        with torch.set_grad_enabled(not self.cfg.freeze):
            out = self.model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        # `hidden_states` is a tuple of length num_layers + 1: index 0 is the
        # embedding output, indices 1..N are the N transformer layers.
        taps: Dict[str, torch.Tensor] = {}
        for layer_idx in self.cfg.tap_layers:
            taps[f"tap{layer_idx}"] = out.hidden_states[layer_idx]
        return taps
