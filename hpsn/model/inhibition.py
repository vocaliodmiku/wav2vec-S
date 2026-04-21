"""Lateral inhibition gate (simplified soft-winners form, per minimal.md §5)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralInhibitionGate(nn.Module):
    """Vocab-space subtractive inhibition with learnable strength alpha."""

    def __init__(self, hidden_dim: int, vocab_size: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.to_vocab = nn.Linear(hidden_dim, vocab_size)
        self.from_vocab = nn.Linear(vocab_size, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = F.relu(self.to_vocab(x))                    # [B, T, V]
        soft_winners = F.softmax(activations / self.temperature, dim=-1)
        inhibited = activations * soft_winners                    # peaks amplified
        suppression = activations - inhibited                     # what was removed
        result = activations - self.alpha * suppression           # suppress losers
        return x + self.from_vocab(result)
