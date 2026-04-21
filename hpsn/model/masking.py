"""Masking modules. Disabled in eval mode."""
from __future__ import annotations

import torch
import torch.nn as nn


class ChunkMasker(nn.Module):
    """Mask contiguous spans of frames (set to zero). Fully vectorized — no GPU→CPU sync."""

    def __init__(self, mask_prob: float = 0.25, min_span: int = 2, max_span: int = 5):
        super().__init__()
        self.mask_prob = mask_prob
        self.min_span = min_span
        self.max_span = max_span

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if not self.training or self.mask_prob <= 0.0 or T <= self.min_span:
            return x, torch.zeros(B, T, dtype=torch.bool, device=x.device)

        avg_span = (self.min_span + self.max_span) / 2.0
        # Slight oversample to offset overlap between random spans and hit ~mask_prob coverage.
        num_spans = max(1, int(round(T * self.mask_prob / avg_span * 1.2)))

        starts = torch.randint(
            0, max(1, T - self.max_span), (B, num_spans), device=x.device
        )
        span_lens = torch.randint(
            self.min_span, self.max_span + 1, (B, num_spans), device=x.device
        )

        t_idx = torch.arange(T, device=x.device).view(1, 1, T)
        starts_e = starts.unsqueeze(-1)
        ends_e = (starts + span_lens).unsqueeze(-1)
        span_mask = (t_idx >= starts_e) & (t_idx < ends_e)  # [B, num_spans, T]
        mask = span_mask.any(dim=1)  # [B, T]

        masked_x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return masked_x, mask


class FrameMasker(nn.Module):
    """Randomly mask individual frames (Bernoulli)."""

    def __init__(self, mask_prob: float = 0.15):
        super().__init__()
        self.mask_prob = mask_prob

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if not self.training or self.mask_prob <= 0.0:
            return x, torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask = torch.bernoulli(torch.full((B, T), self.mask_prob, device=x.device)).bool()
        masked_x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return masked_x, mask
