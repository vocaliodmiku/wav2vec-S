"""Causal cross-layer attention between two HPSN levels (SDPA-backed)."""
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
        self.lookahead = lookahead
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, T, D = query.shape
        Q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0

        if self.lookahead == 0:
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=dropout_p)
        else:
            # Causal with a right-side lookahead of `self.lookahead` frames.
            disallow = torch.triu(
                torch.ones(T, T, device=query.device, dtype=torch.bool),
                diagonal=1 + self.lookahead,
            )
            attn_mask = torch.zeros(T, T, device=query.device, dtype=Q.dtype)
            attn_mask.masked_fill_(disallow, float("-inf"))
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
