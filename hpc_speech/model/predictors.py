"""Top-down predictors and state-summary projections.

Per proposal §4.3 / §4.4:
    ŷ_b[t] = U_b · tanh(V_b · h_{b+1}[t - k])       (low-rank bottleneck, lag k)
    e_b[t] = h_b[t] - ŷ_b[t]
    s_b    = W_b^(s) · h_b                          (low-rank state summary)

The lag k shifts h_{b+1} to the right by k frames; positions t < k receive a
zero higher-level input (no future leakage, strictly causal).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopDownPredictor(nn.Module):
    """g_b : h_{b+1}[t - k] -> ŷ_b[t], low-rank through `rank`."""

    def __init__(self, d: int, rank: int, lag_k: int, dropout: float = 0.1):
        super().__init__()
        self.lag_k = lag_k
        self.V = nn.Linear(d, rank, bias=True)
        self.U = nn.Linear(rank, d, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_upper: torch.Tensor) -> torch.Tensor:
        # h_upper: (B, T, d)
        B, T, D = h_upper.shape
        if self.lag_k > 0:
            pad = torch.zeros(B, self.lag_k, D, dtype=h_upper.dtype, device=h_upper.device)
            lagged = torch.cat([pad, h_upper[:, : T - self.lag_k]], dim=1)
        else:
            lagged = h_upper
        z = torch.tanh(self.V(lagged))
        z = self.drop(z)
        return self.U(z)


class StateSummary(nn.Module):
    """s_b = W_b^(s) · h_b, low-rank projection to `rank`."""

    def __init__(self, d: int, rank: int):
        super().__init__()
        self.W = nn.Linear(d, rank, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.W(h)
