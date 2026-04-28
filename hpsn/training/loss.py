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


# ─────────────────────────────────────────────────────────────────────────────
# HPSN-v2 multi-target loss
# ─────────────────────────────────────────────────────────────────────────────

class HPSNV2Loss(nn.Module):
    """Per-region brain-relevant target reconstruction.

    Targets (from :class:`hpsn.training.data.TargetsHDF5Dataset`):

      L1 (≈ A1 / pSTG)  : log-mel (MSE), phonological features (MSE)
      L2 (≈ MTG)        : phone IDs (CrossEntropy), GPT-2 layer-4 hidden (MSE)
      L3 (≈ aSTG / IFG) : GPT-2 layer-8 hidden (MSE)

    All recon terms are computed on **masked positions only**, intersected
    with the per-frame valid mask from the collator. The top-down alignment
    term (``L_td``) is reused from :class:`HPSNLoss` — μ-vs-level cosine,
    detached at the level side.

    Targets that have not been z-normalized at the dataset level (notably
    GPT-2 hidden states) are dominated by ~5 outlier dims; pass
    ``stats_path`` to ``TargetsHDF5Dataset`` to fix that, or be aware the
    GPT-2 MSE will report values 100–1000× the others if you forget.
    """

    TARGET_KEYS: tuple[str, ...] = (
        "log_mel", "phonol_features", "phone_id", "gpt2_l4", "gpt2_l8",
    )

    def __init__(
        self,
        lambda_log_mel: float = 1.0,
        lambda_phonol: float = 0.5,
        lambda_phone_id: float = 1.0,
        lambda_gpt2_l4: float = 1.0,
        lambda_gpt2_l8: float = 1.0,
        lambda_td: float = 0.2,
        lambda_restore: float = 0.0,
    ):
        super().__init__()
        self.lambda_log_mel = lambda_log_mel
        self.lambda_phonol = lambda_phonol
        self.lambda_phone_id = lambda_phone_id
        self.lambda_gpt2_l4 = lambda_gpt2_l4
        self.lambda_gpt2_l8 = lambda_gpt2_l8
        self.lambda_td = lambda_td
        self.lambda_restore = lambda_restore

    @staticmethod
    def _mse_masked(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask].float(), target[mask].float())

    @staticmethod
    def _ce_masked(
        logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return logits.sum() * 0.0
        l = logits[mask].float()
        t = target[mask].long()
        return F.cross_entropy(l, t)

    def forward(
        self,
        outputs: dict,
        targets: dict[str, torch.Tensor],
        frame_mask: torch.Tensor | None = None,
        restore_mask: torch.Tensor | None = None,
    ) -> dict:
        # Lazy import to avoid a circular dep with the model package.
        from ..model.hpsn import resample_features, resample_ids

        rv2 = outputs["recon_v2"]
        m1 = outputs["mask1"].bool()
        m2 = outputs["mask2"].bool()
        m3 = outputs["mask3"].bool()
        T_b = m1.shape[1]
        fm_aligned = None
        if frame_mask is not None:
            fm = frame_mask.bool().to(m1.device)
            if fm.shape[1] != T_b:
                # Targets/collator are at 50 Hz; backbone may differ. Align
                # the validity mask to the backbone's frame rate (NN).
                fm = resample_ids(fm.long(), T_b).bool()
            fm_aligned = fm
            m1, m2, m3 = m1 & fm, m2 & fm, m3 & fm

        # Align all targets to the backbone's frame rate. Continuous targets
        # use linear interpolation; integer IDs use nearest-neighbor.
        def _to_b(name: str, kind: str) -> torch.Tensor:
            x = targets[name]
            if x.shape[1] == T_b:
                return x
            return resample_ids(x, T_b) if kind == "id" else resample_features(x, T_b)

        log_mel = _to_b("log_mel", "feat")
        phonol = _to_b("phonol_features", "feat")
        phone_id = _to_b("phone_id", "id")
        gpt2_l4 = _to_b("gpt2_l4", "feat")
        gpt2_l8 = _to_b("gpt2_l8", "feat")

        # L1 — log-mel + phonol (regression)
        l_log_mel = self._mse_masked(rv2["l1_log_mel"], log_mel, m1)
        l_phonol = self._mse_masked(rv2["l1_phonol"], phonol, m1)

        # L2 — phone classification + GPT-2 layer-4
        l_phone = self._ce_masked(rv2["l2_phone_logits"], phone_id, m2)
        l_gpt2_l4 = self._mse_masked(rv2["l2_gpt2"], gpt2_l4, m2)

        # L3 — GPT-2 layer-8
        l_gpt2_l8 = self._mse_masked(rv2["l3_gpt2"], gpt2_l8, m3)

        # Top-down alignment (same as v1)
        valid = outputs.get("valid_mask")
        loss_td_2to1 = HPSNLoss._td_align_loss(outputs["mu1"], outputs["level1_repr"], valid)
        loss_td_3to2 = HPSNLoss._td_align_loss(outputs["mu2"], outputs["level2_repr"], valid)
        loss_td = loss_td_2to1 + loss_td_3to2

        # L_restore — predict the *clean* log-mel target on frames where the
        # waveform was replaced by broadband noise (collator did the
        # corruption). Active only when restore_mask is provided AND
        # lambda_restore > 0; otherwise contributes a 0-loss zero-grad term.
        l_restore = rv2["l1_log_mel"].sum() * 0.0
        if restore_mask is not None and self.lambda_restore > 0.0:
            rm = restore_mask.bool().to(m1.device)
            if rm.shape[1] != T_b:
                rm = resample_ids(rm.long(), T_b).bool()
            if fm_aligned is not None:
                rm = rm & fm_aligned
            if rm.any():
                l_restore = F.mse_loss(
                    rv2["l1_log_mel"][rm].float(),
                    log_mel[rm].float(),
                )

        total = (
            self.lambda_log_mel * l_log_mel
            + self.lambda_phonol * l_phonol
            + self.lambda_phone_id * l_phone
            + self.lambda_gpt2_l4 * l_gpt2_l4
            + self.lambda_gpt2_l8 * l_gpt2_l8
            + self.lambda_td * loss_td
            + self.lambda_restore * l_restore
        )
        return {
            "total": total,
            "log_mel": l_log_mel.detach(),
            "phonol": l_phonol.detach(),
            "phone_id": l_phone.detach(),
            "gpt2_l4": l_gpt2_l4.detach(),
            "gpt2_l8": l_gpt2_l8.detach(),
            "td_2to1": loss_td_2to1.detach(),
            "td_3to2": loss_td_3to2.detach(),
            "restore": l_restore.detach(),
        }
