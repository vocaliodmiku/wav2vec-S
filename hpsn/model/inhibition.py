"""Codebook-style lateral inhibition gate.

Hidden states are matched to a small set of learned prototype vectors
(codebook) via cosine similarity. The top-K most similar prototypes form
a "cohort" that competes via pairwise prototype similarity — winners
suppress losers via a similarity-weighted softmax. The result is a
sparse, prototype-weighted correction added back to the input.

Compared to the original (D, V=32000) linear-projection gate, the
codebook variant has ~100× fewer parameters (C × D ≈ 164k for C=320,
D=512) and each prototype is interpretable as a learned "word-like
attractor" — analogous to the wav2vec2 codebook entries.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralInhibitionGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_codes: int,
        temperature: float = 1.0,
        top_k: int = 64,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = min(top_k, num_codes)
        # Single shared codebook plays both roles: matching (input → activation)
        # and reconstruction (cohort → output residual).
        self.codebook = nn.Embedding(num_codes, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.top_k

        # Cosine similarity to all codebook entries.
        x_n = F.normalize(x, dim=-1)                              # [B, T, D]
        cb_n = F.normalize(self.codebook.weight, dim=-1)          # [C, D]
        sims = x_n @ cb_n.t()                                      # [B, T, C] in [-1, 1]

        activations = F.relu(sims)                                 # keep only positive evidence
        topk_vals, topk_idx = activations.topk(K, dim=-1)          # [B, T, K]

        # Pairwise similarity among the K winners' prototypes (cohort competition).
        protos = self.codebook.weight[topk_idx]                    # [B, T, K, D]
        protos_n = F.normalize(protos, dim=-1)
        sim = torch.matmul(protos_n, protos_n.transpose(-1, -2))   # [B, T, K, K]
        sim = sim - torch.eye(K, device=x.device, dtype=sim.dtype)
        sim = F.relu(sim)                                          # only similar pairs compete

        # Cohort softmax → similarity-weighted suppression.
        w = F.softmax(topk_vals / self.temperature, dim=-1)        # [B, T, K]
        inhibition = torch.matmul(sim, w.unsqueeze(-1)).squeeze(-1)  # [B, T, K]
        result_topk = F.relu(topk_vals * (1.0 - self.alpha * inhibition))

        # Reconstruct via weighted sum of the cohort's codebook vectors.
        out = (result_topk.unsqueeze(-2) @ protos).squeeze(-2)     # [B, T, D]
        return x + out
