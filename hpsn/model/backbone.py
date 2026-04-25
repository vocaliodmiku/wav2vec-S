"""Frozen wav2vec-S backbone wrapper + learnable layer-band tap."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure the local wav2vec-S-hf package is importable.
_WAV2VEC_S_HF = Path(__file__).resolve().parents[2] / "wav2vec-S-hf"
if str(_WAV2VEC_S_HF) not in sys.path:
    sys.path.insert(0, str(_WAV2VEC_S_HF))

from wav2vec_s.modeling_wav2vec_s import Wav2VecSModel  # noqa: E402


class FrozenWav2VecS(nn.Module):
    """Wrap a pretrained wav2vec-S model; freeze all parameters; expose hidden states."""

    def __init__(
        self,
        model_name_or_path: str,
        main_context: int = 8,
        right_context: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        model = Wav2VecSModel.from_pretrained(model_name_or_path)
        model.encoder.main_context = main_context
        model.encoder.right_context = right_context
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(dtype)
        self.dtype = dtype
        self.hidden_size = model.config.hidden_size
        self.num_hidden_layers = model.config.num_hidden_layers

    def train(self, mode: bool = True):
        # Always keep the backbone in eval mode (dropout off, no param update).
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Return tuple of hidden states (len = num_hidden_layers + 1)."""
        outputs = self.model(
            input_values.to(self.dtype),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states


class LayerTap(nn.Module):
    """ELMo-style learnable scalar mix over an arbitrary list of layer indices."""

    def __init__(self, layers: Sequence[int]):
        super().__init__()
        layers = tuple(layers)
        assert len(layers) > 0, "LayerTap needs at least one layer index"
        self.layers = layers
        self.weights = nn.Parameter(torch.zeros(len(layers)))  # softmax → uniform at init

    def forward(self, all_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        selected = torch.stack(
            [all_hidden_states[i].float() for i in self.layers],
            dim=0,
        )  # [n_layers, B, T, D]
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (selected * w).sum(dim=0)
