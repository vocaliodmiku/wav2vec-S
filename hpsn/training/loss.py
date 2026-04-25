"""Masked-reconstruction loss (masked frames only) for the 3-level HPSN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPSNLoss(nn.Module):
    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        loss_type: str = "l1",
    ):
        super().__init__()
        assert loss_type in {"l1", "mse", "cosine"}
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.loss_type = loss_type

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return recon.sum() * 0.0  # keeps graph; zero loss contribution
        recon_m = recon[mask].float()
        target_m = target[mask].float()
        if self.loss_type == "l1":
            return F.l1_loss(recon_m, target_m)
        if self.loss_type == "mse":
            return F.mse_loss(recon_m, target_m)
        cos_sim = F.cosine_similarity(recon_m, target_m, dim=-1)
        return (1.0 - cos_sim).mean()

    def forward(self, outputs: dict) -> dict:
        loss1 = self._recon_loss(outputs["recon1"], outputs["target1"], outputs["mask1"])
        loss2 = self._recon_loss(outputs["recon2"], outputs["target2"], outputs["mask2"])
        loss3 = self._recon_loss(outputs["recon3"], outputs["target3"], outputs["mask3"])
        total = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3
        return {
            "total": total,
            "recon1": loss1.detach(),
            "recon2": loss2.detach(),
            "recon3": loss3.detach(),
        }
