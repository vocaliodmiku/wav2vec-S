"""Hierarchical predictive trunk.

Five levels at 50 Hz; each level is `blocks_per_level` CausalConformerBlocks
at width d. Frontend taps are injected at L0..L3 (L4 has no tap per §4.2).
Level b (b >= 1) additionally receives [e_{b-1}, s_{b-1}] from the level
below, linearly projected back to d.

Top-down predictors g_b produce ŷ_b[t] from h_{b+1}[t-k] for b in {0..3}.
Pre-flight uses n_iterations=1: a single bottom-up pass, then top-down
predictions and errors are computed once the full stack of h_b is available.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..config import TrunkConfig
from .conformer import CausalConformerBlock
from .predictors import StateSummary, TopDownPredictor


class _Level(nn.Module):
    """One trunk level: input projection + stacked causal Conformer blocks."""

    def __init__(self, cfg: TrunkConfig, in_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cfg.d) if in_dim != cfg.d else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                CausalConformerBlock(
                    d=cfg.d,
                    num_heads=cfg.num_heads,
                    ffn_mult=cfg.ffn_mult,
                    conv_kernel=cfg.conv_kernel,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.blocks_per_level)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x


class HierarchicalTrunk(nn.Module):
    """5-level causal trunk with top-down predictors and error forwarding.

    Inputs:
        taps: dict mapping the frontend tap key -> tensor (B, T, tap_dim) for
              each level that consumes a tap (per cfg.tap_injection).
        frontend_dim: hidden size of the frontend taps.

    Outputs:
        dict with:
            h:        list of (B, T, d) per level
            y_hat:    list of (B, T, d) per level b in {0..num_levels-2}
            e:        list of (B, T, d) per level b in {0..num_levels-2}
            s:        list of (B, T, state_rank) per level b in {0..num_levels-2}
    """

    def __init__(self, cfg: TrunkConfig, frontend_dim: int, tap_keys: List[str]):
        super().__init__()
        self.cfg = cfg
        self.tap_keys = tap_keys  # ordered per trunk level; "" means no tap

        # Input dim for each level is:
        #   tap feature size (if any) + lower-level forward ([e, s]) size (if any)
        lower_forward_dim = cfg.d + cfg.state_rank  # [e_{b-1}, s_{b-1}]
        self.levels = nn.ModuleList()
        for b in range(cfg.num_levels):
            has_tap = cfg.tap_injection[b] >= 0
            in_dim = 0
            if has_tap:
                in_dim += frontend_dim
            if b > 0:
                in_dim += lower_forward_dim
            assert in_dim > 0, f"Level {b} has no input"
            self.levels.append(_Level(cfg, in_dim=in_dim))

        # One top-down predictor per adjacent pair (b, b+1) for b in 0..num_levels-2
        self.predictors = nn.ModuleList(
            [
                TopDownPredictor(d=cfg.d, rank=cfg.predictor_rank, lag_k=cfg.predictor_lag_k, dropout=cfg.dropout)
                for _ in range(cfg.num_levels - 1)
            ]
        )
        self.state_summaries = nn.ModuleList(
            [StateSummary(d=cfg.d, rank=cfg.state_rank) for _ in range(cfg.num_levels - 1)]
        )

    def forward(
        self,
        taps: Dict[str, torch.Tensor],
        key_padding_mask: torch.Tensor | None = None,
    ) -> Dict[str, List[torch.Tensor]]:
        h_list: List[torch.Tensor] = []
        e_list: List[torch.Tensor] = []
        s_list: List[torch.Tensor] = []
        y_hat_list: List[torch.Tensor] = []

        # --- Bottom-up pass ---
        for b in range(self.cfg.num_levels):
            inputs: List[torch.Tensor] = []
            if self.cfg.tap_injection[b] >= 0:
                key = self.tap_keys[b]
                inputs.append(taps[key])
            if b > 0:
                # Forward [e_{b-1}, s_{b-1}] (computed below once h_b is known for b-1).
                inputs.append(torch.cat([e_list[b - 1], s_list[b - 1]], dim=-1))
            x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
            h_b = self.levels[b](x, key_padding_mask=key_padding_mask)
            h_list.append(h_b)

            # After computing h_b, if b >= 1 we now have h_{b-1} and h_b, so we
            # can compute ŷ_{b-1} = g_{b-1}(h_b[t-k]), e_{b-1}, s_{b-1}. These
            # are needed as input to level b+1 on the next iteration of this
            # loop (i.e. this keeps the bottom-up pass single-sweep).
            if b >= 1:
                lower = b - 1
                y_hat = self.predictors[lower](h_b)
                e = h_list[lower] - y_hat
                s = self.state_summaries[lower](h_list[lower])
                y_hat_list.append(y_hat)
                e_list.append(e)
                s_list.append(s)

        return {"h": h_list, "y_hat": y_hat_list, "e": e_list, "s": s_list}
