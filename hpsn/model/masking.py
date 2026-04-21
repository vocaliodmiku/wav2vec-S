"""Masking modules. Disabled in eval mode."""
from __future__ import annotations

import torch
import torch.nn as nn


class ChunkMasker(nn.Module):
    """Mask contiguous spans of frames (set to zero)."""

    def __init__(self, mask_prob: float = 0.25, min_span: int = 2, max_span: int = 5):
        super().__init__()
        self.mask_prob = mask_prob
        self.min_span = min_span
        self.max_span = max_span

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        if not self.training or self.mask_prob <= 0.0 or T <= self.min_span:
            return x, torch.zeros(B, T, dtype=torch.bool, device=x.device)

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        num_to_mask = int(T * self.mask_prob)
        for b in range(B):
            masked_so_far = 0
            guard = 0
            while masked_so_far < num_to_mask and guard < 4 * num_to_mask:
                span_len = int(torch.randint(self.min_span, self.max_span + 1, (1,)).item())
                start = int(torch.randint(0, max(1, T - span_len), (1,)).item())
                before = mask[b].sum().item()
                mask[b, start : start + span_len] = True
                masked_so_far += int(mask[b].sum().item() - before)
                guard += 1
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
