"""Causal Conformer block.

Strict frame-causality is non-negotiable: output at frame t may depend on
inputs at frames <= t only. Verified by tests/test_causality.py.

Causal ingredients:
- Depthwise 1D conv with left-only padding (padding=(kernel-1, 0)).
- Self-attention with a lower-triangular mask.
- FFN/GLU are pointwise (trivially causal).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.silu(self.fc1(y))
        y = self.drop(y)
        y = self.fc2(y)
        return self.drop(y)


class _CausalSelfAttention(nn.Module):
    def __init__(self, d: int, num_heads: int, dropout: float):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, d * 3, bias=True)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, d)
        B, T, D = x.shape
        y = self.norm(x)
        qkv = self.qkv(y).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, hd)
        q = q.transpose(1, 2)  # (B, H, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Causal lower-triangular mask; combined with optional key-padding mask.
        # Use scaled_dot_product_attention with is_causal=True for correctness
        # and speed; attn_mask here is an additive mask for padding.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.out(out))


class _CausalDepthwiseConv(nn.Module):
    """Conformer convolution module, causal variant.

    Structure: LN -> pointwise conv -> GLU -> causal depthwise conv -> BN ->
    SiLU -> pointwise conv -> dropout.
    """

    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "odd kernel simplifies left-padding bookkeeping"
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, kernel_size=1)
        self.left_pad = kernel - 1
        self.dw = nn.Conv1d(d, d, kernel_size=kernel, groups=d, padding=0)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        y = self.norm(x).transpose(1, 2)  # (B, d, T)
        y = self.pw1(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)  # GLU
        y = F.pad(y, (self.left_pad, 0))  # causal left-pad only
        y = self.dw(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pw2(y)
        y = self.drop(y)
        return y.transpose(1, 2)


class CausalConformerBlock(nn.Module):
    """Conformer block with strict frame-causality.

    Macaron FFN structure: 0.5 * FFN -> MHSA (causal) -> Conv (causal) ->
    0.5 * FFN -> LN.
    """

    def __init__(
        self,
        d: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = _FeedForward(d, ffn_mult, dropout)
        self.mhsa = _CausalSelfAttention(d, num_heads, dropout)
        self.conv = _CausalDepthwiseConv(d, conv_kernel, dropout)
        self.ffn2 = _FeedForward(d, ffn_mult, dropout)
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # key_padding_mask: (B, T) bool, True = padded; converted to additive mask
        attn_mask = None
        if key_padding_mask is not None:
            # additive mask of shape (B, 1, 1, T)
            neg_inf = torch.finfo(x.dtype).min
            attn_mask = torch.zeros_like(key_padding_mask, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask, neg_inf)
            attn_mask = attn_mask[:, None, None, :]
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, attn_mask=attn_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)
