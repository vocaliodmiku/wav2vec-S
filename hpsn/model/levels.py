"""HPSN Level 1 (phonemic) and Level 2 (lexical) — Transformer variant.

Each level stacks ``TransformerBlock`` modules that interleave causal
self-attention (temporal modeling within the level) with causal
cross-attention to the other level. This integrates top-down / bottom-up
signals at *every* block instead of a single post-hoc mixing step.

The kwargs ``lstm_dim`` and ``n_lstm_layers`` are kept for call-site
compatibility with :mod:`hpsn.model.hpsn` and :mod:`hpsn.config`; they map
to ``level_dim`` and ``n_blocks`` respectively.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossLayerAttention
from .inhibition import LateralInhibitionGate


class TransformerBlock(nn.Module):
    """Causal self-attn + (optional) causal cross-attn + FFN.

    Cross-attention is skipped when ``cross_kv`` is None, so the same block
    can run a pass without a top-down / bottom-up signal (e.g. Level 2's
    first forward before any Level 1 error exists).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        lookahead: int = 0,
    ):
        super().__init__()
        self.self_attn = CrossLayerAttention(dim, n_heads, dropout, lookahead)
        self.cross_attn = CrossLayerAttention(dim, n_heads, dropout, lookahead)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, cross_kv: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(query=h, key_value=h)
        if cross_kv is not None:
            x = x + self.cross_attn(query=self.norm2(x), key_value=cross_kv)
        x = x + self.ffn(self.norm3(x))
        return x


class HPSNLevel1(nn.Module):
    """Phonemic level. Stack of Transformer blocks with causal self-attention
    and causal cross-attention from Level 2's top-down signal."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        lstm_dim: int = 512,
        n_lstm_layers: int = 2,
        n_attn_heads: int = 8,
        dropout: float = 0.1,
        causal_lookahead: int = 0,
    ):
        super().__init__()
        level_dim, n_blocks = lstm_dim, n_lstm_layers
        self.input_proj = nn.Linear(hidden_dim, level_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(level_dim, n_attn_heads, dropout, causal_lookahead)
                for _ in range(n_blocks)
            ]
        )
        self.recon_head = nn.Linear(level_dim, hidden_dim)

    def forward(
        self, masked_input: torch.Tensor, top_down_signal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(masked_input)
        for block in self.blocks:
            x = block(x, cross_kv=top_down_signal)
        recon = self.recon_head(x)
        return x, recon


class HPSNLevel2(nn.Module):
    """Lexical level. Stack of Transformer blocks with causal self-attention
    and causal cross-attention from Level 1's bottom-up error (``None`` on
    the first pass). The lateral inhibition gate runs *after* all blocks —
    temporal integration and top-down mixing happen in the stack, then
    inhibition resolves lexical competition on the final representation."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        lstm_dim: int = 512,
        n_lstm_layers: int = 2,
        vocab_size: int = 32000,
        n_attn_heads: int = 8,
        dropout: float = 0.1,
        causal_lookahead: int = 0,
        inhib_temperature: float = 1.0,
        inhib_top_k: int = 64,
    ):
        super().__init__()
        level_dim, n_blocks = lstm_dim, n_lstm_layers
        self.input_proj = nn.Linear(hidden_dim, level_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(level_dim, n_attn_heads, dropout, causal_lookahead)
                for _ in range(n_blocks)
            ]
        )
        self.inhib_gate = LateralInhibitionGate(
            level_dim, vocab_size, temperature=inhib_temperature, top_k=inhib_top_k,
        )
        self.td_predictor = nn.Sequential(
            nn.Linear(level_dim, level_dim),
            nn.GELU(),
            nn.Linear(level_dim, level_dim),
        )
        self.recon_head = nn.Linear(level_dim, hidden_dim)

    def forward(
        self, masked_input: torch.Tensor, bu_error: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(masked_input)
        for block in self.blocks:
            x = block(x, cross_kv=bu_error)
        x = self.inhib_gate(x)
        top_down = self.td_predictor(x)
        recon = self.recon_head(x)
        return x, top_down, recon
