"""HPC-Speech pre-flight model.

Ties together:
    frozen wav2vec-S frontend (m=8, r=4) -> 4-layer taps ->
    5-level causal Conformer trunk with top-down predictors and [e, s] forwarding
    -> per-level CPC heads + CTC head on L4.

No inhibition, no iterative refinement (pre-flight subset; proposal §5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PreflightConfig
from .cpc import PerLevelCPC
from .frontend import Wav2VecSFrontend
from .trunk import HierarchicalTrunk


@dataclass
class HPCSpeechOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    ctc_loss: Optional[torch.Tensor]
    pred_losses: List[torch.Tensor]   # per level b in 0..num_levels-2
    cpc_losses: List[torch.Tensor]    # per level b in 0..num_levels-1
    e_norms: List[torch.Tensor]       # mean ||e_b|| per level (for logging)
    h: List[torch.Tensor]
    e: List[torch.Tensor]
    s: List[torch.Tensor]


class HPCSpeechPreflight(nn.Module):
    def __init__(self, cfg: PreflightConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = Wav2VecSFrontend(cfg.frontend)
        tap_keys: List[str] = []
        for b in range(cfg.trunk.num_levels):
            idx = cfg.trunk.tap_injection[b]
            tap_keys.append(f"tap{idx}" if idx >= 0 else "")
        self.trunk = HierarchicalTrunk(
            cfg.trunk,
            frontend_dim=cfg.frontend.hidden_size,
            tap_keys=tap_keys,
        )
        self.cpc_heads = nn.ModuleList(
            [
                PerLevelCPC(
                    d=cfg.trunk.d,
                    hidden=cfg.cpc.hidden,
                    horizon_k=cfg.cpc.horizon_k,
                    num_negatives=cfg.cpc.num_negatives,
                )
                for _ in range(cfg.trunk.num_levels)
            ]
        )
        self.ctc_head = nn.Linear(cfg.trunk.d, cfg.vocab_size)
        self.dropout = nn.Dropout(cfg.trunk.dropout)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> HPCSpeechOutput:
        taps = self.frontend(input_values, attention_mask=attention_mask)
        # Derive a frame-level key-padding mask from attention_mask. wav2vec-S
        # strides ~320 samples per frame (16 kHz, 20 ms). Downstream we use the
        # T dimension of the tap output directly to build the mask.
        sample = next(iter(taps.values()))
        B, T, _ = sample.shape
        if attention_mask is not None:
            # Collapse attention_mask (B, n_samples) -> (B, T) by nearest-neighbor
            # ratio. We avoid assumptions about exact stride here.
            ratio = attention_mask.shape[1] / T
            idx = (torch.arange(T, device=attention_mask.device) * ratio).long()
            idx = idx.clamp_max(attention_mask.shape[1] - 1)
            frame_mask = attention_mask[:, idx]  # (B, T)
            key_padding_mask = frame_mask == 0
        else:
            key_padding_mask = None

        out = self.trunk(taps, key_padding_mask=key_padding_mask)
        h_list = out["h"]
        e_list = out["e"]
        s_list = out["s"]

        # CPC per level.
        cpc_losses: List[torch.Tensor] = []
        for b, head in enumerate(self.cpc_heads):
            cpc_losses.append(head(h_list[b]))

        # Predictive loss per (b, b+1) pair.
        pred_losses: List[torch.Tensor] = []
        e_norms: List[torch.Tensor] = []
        for b, e in enumerate(e_list):
            if key_padding_mask is not None:
                valid = (~key_padding_mask).unsqueeze(-1).to(e.dtype)
                mse = ((e ** 2) * valid).sum() / valid.sum().clamp_min(1.0) / e.shape[-1]
                norm = ((e.norm(dim=-1) * valid.squeeze(-1)).sum() / valid.squeeze(-1).sum().clamp_min(1.0))
            else:
                mse = (e ** 2).mean()
                norm = e.norm(dim=-1).mean()
            pred_losses.append(mse)
            e_norms.append(norm.detach())

        # CTC head on L4.
        logits = self.ctc_head(self.dropout(h_list[-1]))  # (B, T, V)

        total_loss: Optional[torch.Tensor] = None
        ctc_loss: Optional[torch.Tensor] = None
        if labels is not None and label_lengths is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
            if key_padding_mask is not None:
                input_lengths = (~key_padding_mask).sum(dim=-1).long()
            else:
                input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
            ctc_loss = F.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
                blank=0,
                zero_infinity=True,
                reduction="mean",
            )
            alpha = self.cfg.loss.alpha_pred
            eta = self.cfg.loss.eta_cpc
            w = self.cfg.loss.pred_level_weights
            pred_term = sum(w[b] * pred_losses[b] for b in range(len(pred_losses)))
            cpc_term = sum(cpc_losses)
            total_loss = ctc_loss + alpha * pred_term + eta * cpc_term

        return HPCSpeechOutput(
            loss=total_loss,
            logits=logits,
            ctc_loss=ctc_loss,
            pred_losses=pred_losses,
            cpc_losses=cpc_losses,
            e_norms=e_norms,
            h=h_list,
            e=e_list,
            s=s_list,
        )
