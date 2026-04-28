"""Full 3-level HPSN module (acoustic / lexical / semantic)."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HPSNConfig
from .backbone import LayerTap
from .levels import HPSNLevel1, HPSNLevel2, HPSNLevel3
from .masking import ChunkMasker, FrameMasker, PhonemeSpanMasker, WordSpanMasker


class HPSNV2Heads(nn.Module):
    """Multi-target reconstruction heads for HPSN-v2.

    Mounted in :class:`HPSN` and applied to the per-level representations
    after the (optional) refinement loop. Predictions live alongside the
    v1 ``recon{1,2,3}`` (which still target the tapped backbone band) so
    both training paths coexist; the active loss decides which heads
    contribute gradient.

    Per-level outputs (each shape ``[B, T, D_out]``):

      L1: ``log_mel`` (D_out=80), ``phonol`` (14)
      L2: ``phone_logits`` (40), ``gpt2_l4`` (768)
      L3: ``gpt2_l8`` (768)
    """

    def __init__(
        self,
        level_dim: int,
        n_log_mel: int = 80,
        n_phonol: int = 14,
        n_phones: int = 40,
        gpt2_dim: int = 768,
    ):
        super().__init__()
        self.l1_log_mel = nn.Linear(level_dim, n_log_mel)
        self.l1_phonol = nn.Linear(level_dim, n_phonol)
        self.l2_phone_logits = nn.Linear(level_dim, n_phones)
        self.l2_gpt2 = nn.Linear(level_dim, gpt2_dim)
        self.l3_gpt2 = nn.Linear(level_dim, gpt2_dim)

    def forward(
        self, l1_repr: torch.Tensor, l2_repr: torch.Tensor, l3_repr: torch.Tensor,
    ) -> dict:
        return {
            "l1_log_mel": self.l1_log_mel(l1_repr),
            "l1_phonol": self.l1_phonol(l1_repr),
            "l2_phone_logits": self.l2_phone_logits(l2_repr),
            "l2_gpt2": self.l2_gpt2(l2_repr),
            "l3_gpt2": self.l3_gpt2(l3_repr),
        }


@contextmanager
def _null_region(_name: str):
    yield


class HPSN(nn.Module):
    """Three-level Hierarchical Predictive Speech Network.

    Default forward (``n_iterations == 1``) is the strict top-down sweep used
    by HPSN-v1::

      L3(tap3, cross_kv=None)            → l3_repr, μ₂, recon3
      L2(tap2, cross_kv=μ₂)              → l2_repr, μ₁, recon2  (with lateral inhibition)
      L1(tap1, cross_kv=μ₁)              → l1_repr,     recon1

    With ``n_iterations >= 2`` (HPSN-v2), each extra pass injects the
    bottom-up prediction error ``eps2 = error_proj_2(l2_repr.detach()) - μ₂``
    into L3 via cross-attention, then re-runs the top-down sweep so the
    representations converge per the construction-doc §5.4 predictive-coding
    protocol. Each level reconstructs its own tapped backbone band on the
    final iteration's outputs.
    """

    def __init__(self, config: HPSNConfig):
        super().__init__()
        self.config = config
        H, D, C = config.hidden_dim, config.lstm_dim, config.inhib_num_codes
        self.n_iterations = max(1, int(config.n_iterations))

        def _resolve_frozen(v: int) -> int | None:
            return None if v is None or int(v) < 0 else int(v)

        self.tap1 = LayerTap(
            config.level1_tap_layers, frozen_layer=_resolve_frozen(config.level1_frozen_tap),
        )
        self.tap2 = LayerTap(
            config.level2_tap_layers, frozen_layer=_resolve_frozen(config.level2_frozen_tap),
        )
        self.tap3 = LayerTap(
            config.level3_tap_layers, frozen_layer=_resolve_frozen(config.level3_frozen_tap),
        )

        self.use_span_masking = bool(getattr(config, "use_span_masking", False))
        if self.use_span_masking:
            # v2 — phoneme/word spans (require phone_id / word_id at forward time)
            self.masker1 = PhonemeSpanMasker(mask_prob=config.level1_mask_prob)
            self.masker2 = WordSpanMasker(mask_prob=config.level2_mask_prob)
            self.masker3 = WordSpanMasker(mask_prob=config.level3_mask_prob)
        else:
            # v1 — random chunk / frame masks
            self.masker1 = ChunkMasker(
                mask_prob=config.level1_mask_prob,
                min_span=config.chunk_min_span,
                max_span=config.chunk_max_span,
            )
            self.masker2 = FrameMasker(mask_prob=config.level2_mask_prob)
            self.masker3 = FrameMasker(mask_prob=config.level3_mask_prob)

        # Per-level v1 tap-reconstruction heads are skipped in v2 mode (the
        # v2 multi-target heads on HPSNV2Heads cover reconstruction instead).
        use_recon_head = not bool(getattr(config, "use_v2_loss", False))

        self.level3 = HPSNLevel3(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
            use_recon_head=use_recon_head,
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
            use_recon_head=use_recon_head,
        )
        self.level1 = HPSNLevel1(
            hidden_dim=H,
            lstm_dim=D,
            n_lstm_layers=config.n_lstm_layers,
            num_codes=config.inhib_l1_num_codes,
            n_attn_heads=config.n_attn_heads,
            dropout=config.dropout,
            causal_lookahead=config.causal_lookahead,
            inhib_temperature=config.inhib_temperature,
            inhib_top_k=config.inhib_l1_top_k,
            use_recon_head=use_recon_head,
        )

        # Bottom-up L2-error projection used in the iterative-refinement loop
        # (n_iterations >= 2). The construction-doc dual-cross-attention path
        # for an L1-error projection is deferred; not built.
        self.error_proj_2 = nn.Linear(D, D)

        # v2 multi-target heads. Always built so checkpoints round-trip when
        # the use_v2_loss flag is toggled.
        self.v2_heads = HPSNV2Heads(
            level_dim=D,
            n_log_mel=getattr(config, "n_log_mel", 80),
            n_phonol=getattr(config, "n_phonol_features", 14),
            n_phones=getattr(config, "n_phones", 40),
            gpt2_dim=getattr(config, "gpt2_hidden_dim", 768),
        )

    def forward(
        self,
        all_hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor | None = None,
        phone_id: torch.Tensor | None = None,
        word_id: torch.Tensor | None = None,
    ) -> dict:
        prof = getattr(self, "_profiler", None)
        _region = prof.region if prof is not None else _null_region

        with _region("hpsn.tap"):
            tap1 = self.tap1(all_hidden_states)
            tap2 = self.tap2(all_hidden_states)
            tap3 = self.tap3(all_hidden_states)

        with _region("hpsn.mask"):
            if self.use_span_masking:
                if self.training:
                    if phone_id is None or word_id is None:
                        raise RuntimeError(
                            "use_span_masking=True requires phone_id and word_id "
                            "to be passed to HPSN.forward in training mode."
                        )
                    # Backbone frame rate may differ from the targets' 50 Hz
                    # (chunk-causal padding shifts T_out). Align IDs to T_backbone.
                    T_b = tap1.shape[1]
                    pid_b = resample_ids(phone_id, T_b)
                    wid_b = resample_ids(word_id, T_b)
                    masked1, mask1 = self.masker1(tap1, pid_b)
                    masked2, mask2 = self.masker2(tap2, wid_b)
                    masked3, mask3 = self.masker3(tap3, wid_b)
                else:
                    # Eval mode: span maskers no-op (they early-return when not
                    # self.training). Skip the resample + ID requirement so
                    # downstream feature extraction can run without targets.
                    B, T_b = tap1.shape[:2]
                    zm = torch.zeros(B, T_b, dtype=torch.bool, device=tap1.device)
                    masked1, mask1 = tap1, zm
                    masked2, mask2 = tap2, zm
                    masked3, mask3 = tap3, zm
            else:
                masked1, mask1 = self.masker1(tap1)
                masked2, mask2 = self.masker2(tap2)
                masked3, mask3 = self.masker3(tap3)

        # Pass 1 — strict top-down sweep: L3 → L2 → L1.
        with _region("hpsn.level3"):
            l3_repr, mu2, recon3 = self.level3(masked3, cross_kv=None)
        with _region("hpsn.level2"):
            l2_repr, mu1, recon2 = self.level2(masked2, cross_kv=mu2)
        with _region("hpsn.level1"):
            l1_repr, recon1 = self.level1(masked1, cross_kv=mu1)

        # Pass 2+ — predictive-coding refinement (HPSN-v2). Each extra pass
        # routes the L2 prediction error eps2 = error_proj_2(l2.detach()) - mu2
        # into L3 as cross-attention K/V, then re-runs the top-down sweep with
        # the refreshed mu2'. ``.detach()`` on l2_repr keeps each level's local
        # objective intact (lower levels learn from their own recon loss only;
        # the upper level uses the lower's residual as a *signal*, not a
        # gradient pathway).
        for _it in range(self.n_iterations - 1):
            with _region("hpsn.refine.eps"):
                eps2 = self.error_proj_2(l2_repr.detach()) - mu2
            with _region("hpsn.level3"):
                l3_repr, mu2, recon3 = self.level3(masked3, cross_kv=eps2)
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
        else:
            valid = None

        # v2 multi-target heads (read from final-iteration level reprs).
        with _region("hpsn.v2_heads"):
            recon_v2 = self.v2_heads(l1_repr, l2_repr, l3_repr)

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
            # Top-down predictions (D-space) — supervised against level_repr
            # via the L_td term, and used as cross-attn K/V for the lower level.
            "mu1": mu1,
            "mu2": mu2,
            # Frame-resolution valid mask (downsampled attention_mask), used by
            # L_td to skip padding positions. None if no attention_mask was given.
            "valid_mask": valid,
            # v2 multi-target reconstructions (read by HPSNV2Loss).
            "recon_v2": recon_v2,
        }


def _downsample_mask(valid: torch.Tensor, target_len: int) -> torch.Tensor:
    """Downsample a waveform-resolution bool mask to frame resolution by uniform binning (floor)."""
    B, L = valid.shape
    idx = torch.linspace(0, L - 1, target_len, device=valid.device).long()
    return valid[:, idx]


def resample_ids(ids: torch.Tensor, target_T: int) -> torch.Tensor:
    """Nearest-neighbor resample integer-ID arrays ``[B, T_src]`` → ``[B, T_dst]``.

    Used to align Phase-1 targets (50 Hz, librosa hop) to the backbone's
    native frame rate (which differs slightly because of chunk-causal
    padding inside wav2vec-S).
    """
    B, T_src = ids.shape
    if T_src == target_T:
        return ids
    idx = torch.linspace(0, T_src - 1, target_T, device=ids.device).long()
    return ids.index_select(1, idx)


def resample_features(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linear-interpolate a continuous feature tensor along the time axis.

    Accepts ``[B, T, D]`` or ``[B, T]``; returns the same shape with
    ``T → target_T``. For integer tensors, use ``resample_ids`` instead.
    """
    if x.shape[1] == target_T:
        return x
    if x.dim() == 3:
        t = x.float().transpose(1, 2)  # [B, D, T]
        t = F.interpolate(t, size=target_T, mode="linear", align_corners=False)
        return t.transpose(1, 2).to(x.dtype)
    if x.dim() == 2:
        t = x.float().unsqueeze(1)  # [B, 1, T]
        t = F.interpolate(t, size=target_T, mode="linear", align_corners=False)
        return t.squeeze(1).to(x.dtype)
    raise ValueError(f"resample_features: unsupported ndim={x.dim()}")
