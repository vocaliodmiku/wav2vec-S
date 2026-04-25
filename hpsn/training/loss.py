"""HPSN loss: per-band masked reconstruction + top-down prediction alignment.

L_total = Σ λᵢ · L_recon_i  +  λ_td · (L_td_2→1 + L_td_3→2)

L_recon_i is the masked-frame reconstruction loss for band i (current behavior).

L_td_j→i = 1 − cos(μᵢ, level_repr_i.detach())   averaged over valid frames

The detach is the safety pin: it lets the upper level's td_predictor learn to
align with the lower level's representation, while the lower level remains free
to optimize its own reconstruction objective. Cosine (not L2) decouples scale
— μ can have its own magnitude as long as it points the right way.
"""
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
        lambda_td: float = 0.2,
        loss_type: str = "l1",
    ):
        super().__init__()
        assert loss_type in {"l1", "mse", "cosine"}
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_td = lambda_td
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

    @staticmethod
    def _td_align_loss(
        mu: torch.Tensor,
        level_repr: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """1 − mean cosine(μ, level_repr.detach()) over valid frames."""
        if valid_mask is not None:
            if valid_mask.sum() == 0:
                return mu.sum() * 0.0
            mu_v = mu[valid_mask].float()
            rep_v = level_repr[valid_mask].detach().float()
        else:
            mu_v = mu.reshape(-1, mu.shape[-1]).float()
            rep_v = level_repr.detach().reshape(-1, level_repr.shape[-1]).float()
        cos = F.cosine_similarity(mu_v, rep_v, dim=-1)
        return 1.0 - cos.mean()

    def forward(self, outputs: dict) -> dict:
        loss1 = self._recon_loss(outputs["recon1"], outputs["target1"], outputs["mask1"])
        loss2 = self._recon_loss(outputs["recon2"], outputs["target2"], outputs["mask2"])
        loss3 = self._recon_loss(outputs["recon3"], outputs["target3"], outputs["mask3"])

        valid = outputs.get("valid_mask")
        loss_td_2to1 = self._td_align_loss(outputs["mu1"], outputs["level1_repr"], valid)
        loss_td_3to2 = self._td_align_loss(outputs["mu2"], outputs["level2_repr"], valid)
        loss_td = loss_td_2to1 + loss_td_3to2

        total = (
            self.lambda1 * loss1
            + self.lambda2 * loss2
            + self.lambda3 * loss3
            + self.lambda_td * loss_td
        )
        return {
            "total": total,
            "recon1": loss1.detach(),
            "recon2": loss2.detach(),
            "recon3": loss3.detach(),
            "td_2to1": loss_td_2to1.detach(),
            "td_3to2": loss_td_3to2.detach(),
        }
