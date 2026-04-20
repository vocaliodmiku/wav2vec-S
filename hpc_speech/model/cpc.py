"""Per-level CPC auxiliary head (proposal §4.7).

For each trunk level b:
- A small causal GRU aggregates h_b[<t] into a context c_b[t].
- A linear predictor W^{(k')}_b projects c_b[t] to a target for h_b[t+k'].
- InfoNCE loss against in-batch negatives at the same level.

No cross-level conditioning; trunk-level CPC is complementary to wav2vec-S's
frontend contrastive loss (different representations, autoregressive context,
longer horizon).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLevelCPC(nn.Module):
    def __init__(self, d: int, hidden: int, horizon_k: int, num_negatives: int):
        super().__init__()
        self.horizon_k = horizon_k
        self.num_negatives = num_negatives
        self.gru = nn.GRU(input_size=d, hidden_size=hidden, batch_first=True)
        self.predictor = nn.Linear(hidden, d)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # h: (B, T, d); mask: (B, T) bool, True where valid frame (optional).
        B, T, D = h.shape
        k = self.horizon_k
        if T <= k + 1:
            return h.new_zeros(())

        c, _ = self.gru(h)  # (B, T, hidden)
        pred = self.predictor(c[:, : T - k])  # (B, T-k, d)
        target = h[:, k:]  # (B, T-k, d) -- positive: h[t+k]

        # Positive score: dot(pred[t], target[t]).
        pos = (pred * target).sum(dim=-1)  # (B, T-k)

        # Negatives: sample num_negatives random time indices per position from
        # the same minibatch (flattened over B x T-k).
        Bt = pred.shape[0] * pred.shape[1]
        flat_pred = pred.reshape(Bt, D)
        flat_tgt = target.reshape(Bt, D)
        # Sample negative indices uniformly from the valid pool.
        neg_idx = torch.randint(0, Bt, (Bt, self.num_negatives), device=h.device)
        # Avoid collisions with the positive index (rough; a collision just
        # makes one negative equal to the positive, which slightly softens
        # the loss — acceptable for an auxiliary objective).
        neg = flat_tgt[neg_idx]  # (Bt, num_neg, D)
        neg_scores = torch.einsum("bd,bnd->bn", flat_pred, neg)  # (Bt, num_neg)

        logits = torch.cat([pos.reshape(Bt, 1), neg_scores], dim=1)  # (Bt, 1+num_neg)
        labels = torch.zeros(Bt, dtype=torch.long, device=h.device)
        loss = F.cross_entropy(logits, labels)
        return loss
