"""Full 3-level HPSN module (acoustic / lexical / semantic)."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ..config import HPSNConfig
from .backbone import LayerTap
from .levels import HPSNLevel1, HPSNLevel2, HPSNLevel3
from .masking import ChunkMasker, FrameMasker


@contextmanager
def _null_region(_name: str):
    yield


class HPSN(nn.Module):
    """Three-level Hierarchical Predictive Speech Network.

    Forward order is strictly top-down (single sweep, no iterative refinement):
      L3(tap3, cross_kv=None)            → l3_repr, μ₂, recon3
      L2(tap2, cross_kv=μ₂)              → l2_repr, μ₁, recon2  (with lateral inhibition)
      L1(tap1, cross_kv=μ₁)              → l1_repr,     recon1

    Each level reconstructs its own tapped backbone band (masked positions only).
    """

    def __init__(self, config: HPSNConfig):
        super().__init__()
        self.config = config
        H, D, C = config.hidden_dim, config.lstm_dim, config.inhib_num_codes

        self.tap1 = LayerTap(config.level1_tap_layers)
        self.tap2 = LayerTap(config.level2_tap_layers)
        self.tap3 = LayerTap(config.level3_tap_layers)

        self.masker1 = ChunkMasker(
            mask_prob=config.level1_mask_prob,
            min_span=config.chunk_min_span,
            max_span=config.chunk_max_span,
        )
        self.masker2 = FrameMasker(mask_prob=config.level2_mask_prob)
        self.masker3 = FrameMasker(mask_prob=config.level3_mask_prob)

        self.level3 = HPSNLevel3(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
        )
        self.level2 = HPSNLevel2(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            num_codes=C,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
            inhib_temperature=config.inhib_temperature,
            inhib_top_k=config.inhib_top_k,
        )
        self.level1 = HPSNLevel1(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
        )

    def forward(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        prof = getattr(self, "_profiler", None)
        _region = prof.region if prof is not None else _null_region

        with _region("hpsn.tap"):
            tap1 = self.tap1(all_hidden_states)
            tap2 = self.tap2(all_hidden_states)
            tap3 = self.tap3(all_hidden_states)

        with _region("hpsn.mask"):
            masked1, mask1 = self.masker1(tap1)
            masked2, mask2 = self.masker2(tap2)
            masked3, mask3 = self.masker3(tap3)

        # Top-down sweep: L3 → L2 → L1.
        with _region("hpsn.level3"):
            l3_repr, mu2, recon3 = self.level3(masked3, cross_kv=None)
        with _region("hpsn.level2"):
            l2_repr, mu1, recon2 = self.level2(masked2, cross_kv=mu2)
        with _region("hpsn.level1"):
            l1_repr, recon1 = self.level1(masked1, cross_kv=mu1)

        # Intersect reconstruction masks with attention_mask (valid frames) if provided.
        if attention_mask is not None:
            valid = attention_mask.bool()
            T = mask1.shape[1]
            if valid.shape[1] != T:
                valid = _downsample_mask(valid, T)
            mask1 = mask1 & valid
            mask2 = mask2 & valid
            mask3 = mask3 & valid

        return {
            "recon1": recon1,
            "recon2": recon2,
            "recon3": recon3,
            "target1": tap1,
            "target2": tap2,
            "target3": tap3,
            "mask1": mask1,
            "mask2": mask2,
            "mask3": mask3,
            "level1_repr": l1_repr,
            "level2_repr": l2_repr,
            "level3_repr": l3_repr,
        }


def _downsample_mask(valid: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample a waveform-resolution bool mask to frame resolution by uniform binning (floor)."""
    B, L = valid.shape
    idx = torch.linspace(0, L - 1, target_len, device=valid.device).long()
    return valid[:, idx]
