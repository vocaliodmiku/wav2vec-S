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


class _SpanMasker(nn.Module):
    """Mask whole spans defined by an integer ID array; ID==0 → silence (skip).

    The ID array (``[B, T]``) marks contiguous frames belonging to the same
    phoneme/word/etc. (this exactly matches the layout produced by Phase 1
    ``extract_targets.py``). A span = a maximal run of equal non-zero IDs.
    On each batch, ``mask_prob`` fraction of the candidate spans (per sample)
    is fully masked.

    Whole-span masking is the lever that makes masked reconstruction
    *intrinsically hierarchical*: random-frame masks can be solved by local
    interpolation, but recovering an entire phoneme requires the model to
    use lexical context (and recovering a whole word requires syntax).
    """

    def __init__(self, mask_prob: float = 0.20):
        super().__init__()
        self.mask_prob = float(mask_prob)

    def _mask_one(
        self, ids: torch.Tensor, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Build a frame-mask for one sample's ID vector ``[T]``."""
        T = ids.shape[0]
        out = torch.zeros(T, dtype=torch.bool, device=ids.device)
        # Boundary mask: True at the first frame of every new run.
        prev = torch.cat([ids[:1] - 1, ids[:-1]])  # ensure ids[0] is a boundary
        boundary = ids != prev
        starts = torch.nonzero(boundary, as_tuple=False).squeeze(-1)  # [n_runs]
        if starts.numel() == 0:
            return out
        ends = torch.cat([starts[1:], torch.tensor([T], device=ids.device)])
        run_ids = ids[starts]
        # Candidate spans = non-silence runs (id != 0).
        cand = (run_ids != 0).nonzero(as_tuple=False).squeeze(-1)
        if cand.numel() == 0:
            return out
        n_to_mask = max(1, int(round(cand.numel() * self.mask_prob)))
        if generator is not None:
            perm = torch.randperm(cand.numel(), generator=generator, device=ids.device)
        else:
            perm = torch.randperm(cand.numel(), device=ids.device)
        chosen = cand[perm[:n_to_mask]]
        for i in chosen.tolist():
            out[int(starts[i].item()): int(ends[i].item())] = True
        return out

    def forward(
        self, x: torch.Tensor, ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x: [B, T, D]``, ``ids: [B, T]`` → ``(masked_x, mask)``."""
        B, T, _ = x.shape
        if not self.training or self.mask_prob <= 0.0:
            return x, torch.zeros(B, T, dtype=torch.bool, device=x.device)
        if ids.shape != (B, T):
            raise ValueError(f"ids shape {tuple(ids.shape)} != x [B,T]={B, T}")
        # Per-sample loop (small B); inner ops vectorised.
        mask = torch.stack([self._mask_one(ids[b]) for b in range(B)], dim=0)
        masked_x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return masked_x, mask


class PhonemeSpanMasker(_SpanMasker):
    """Mask whole-phoneme spans (use ``phone_id`` from the targets HDF5)."""


class WordSpanMasker(_SpanMasker):
    """Mask whole-word spans (use ``word_id`` from the targets HDF5)."""
