"""Full 2-level HPSN module wiring the components together."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import HPSNConfig
from .backbone import LayerTap
from .levels import HPSNLevel1, HPSNLevel2
from .masking import ChunkMasker, FrameMasker


class HPSNMinimal(nn.Module):
    def __init__(self, config: HPSNConfig):
        super().__init__()
        self.config = config
        H, D, V = config.hidden_dim, config.lstm_dim, config.vocab_size

        self.tap_acoustic = LayerTap(config.tap_acoustic_start, config.tap_acoustic_end)
        self.tap_lexical = LayerTap(config.tap_lexical_start, config.tap_lexical_end)

        self.masker_acoustic = ChunkMasker(
            mask_prob=config.mask_prob_acoustic,
            min_span=config.chunk_min_span,
            max_span=config.chunk_max_span,
        )
        self.masker_lexical = FrameMasker(mask_prob=config.mask_prob_lexical)

        self.level2 = HPSNLevel2(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            vocab_size=V,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
            inhib_temperature=config.inhib_temperature,
        )
        self.level1 = HPSNLevel1(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
        )

        self.error_proj = nn.Linear(D, D)

    def forward(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        acoustic_features = self.tap_acoustic(all_hidden_states)
        lexical_features = self.tap_lexical(all_hidden_states)

        masked_acoustic, mask1 = self.masker_acoustic(acoustic_features)
        masked_lexical, mask2 = self.masker_lexical(lexical_features)

        # Pass 1: Level 2 without bottom-up error.
        l2_output, top_down_mu1, recon2 = self.level2(masked_lexical, bu_error=None)

        # Level 1 uses top-down prediction from Level 2.
        l1_output, recon1 = self.level1(masked_acoustic, top_down_signal=top_down_mu1)

        if self.config.iterative_refine:
            # Error signal feeds Level 2 on a second pass.
            error_signal = self.error_proj(l1_output.detach()) - top_down_mu1
            l2_output, top_down_mu1, recon2 = self.level2(masked_lexical, bu_error=error_signal)
            l1_output, recon1 = self.level1(masked_acoustic, top_down_signal=top_down_mu1)

        # Intersect reconstruction masks with attention_mask (valid frames) if provided.
        if attention_mask is not None:
            valid = attention_mask.bool()
            # attention_mask is at waveform resolution; downsample to frame resolution.
            T = mask1.shape[1]
            if valid.shape[1] != T:
                valid = _downsample_mask(valid, T)
            mask1 = mask1 & valid
            mask2 = mask2 & valid

        return {
            "recon1": recon1,
            "recon2": recon2,
            "target1": acoustic_features,
            "target2": lexical_features,
            "mask1": mask1,
            "mask2": mask2,
            "level1_repr": l1_output,
            "level2_repr": l2_output,
        }


def _downsample_mask(valid: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample a waveform-resolution bool mask to frame resolution by uniform binning (floor)."""
    B, L = valid.shape
    # Each frame covers roughly L / target_len samples.  Compute true-ness by the frame's start index.
    idx = torch.linspace(0, L - 1, target_len, device=valid.device).long()
    return valid[:, idx]
