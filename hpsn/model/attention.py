"""Causal cross-layer attention between two HPSN levels."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayerAttention(nn.Module):
    """Multi-head causal cross-attention. Query at t attends to key/value at <= t (+ optional lookahead)."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1, lookahead: int = 0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.lookahead = lookahead

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, T, D = query.shape
        Q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal mask: True = disallowed.  diagonal = lookahead + 1 permits `lookahead` frames ahead.
        causal_mask = torch.triu(
            torch.ones(T, T, device=query.device, dtype=torch.bool),
            diagonal=1 + self.lookahead,
        )

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
