"""Lateral inhibition gate — top-k pairwise competition in vocab space.

Only the top-k most active candidates per frame enter the "cohort" and inhibit
each other via pairwise cosine similarity of their vocab prototypes; losers are
suppressed, everything outside the top-k stays at zero. Motivated by cohort
models of word recognition (active cohort ~20-100).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralInhibitionGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        temperature: float = 1.0,
        top_k: int = 64,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = min(top_k, vocab_size)
        self.to_vocab = nn.Linear(hidden_dim, vocab_size)
        self.from_vocab = nn.Linear(vocab_size, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.top_k

        activations = F.relu(self.to_vocab(x))                # [B, T, V]
        topk_vals, topk_idx = activations.topk(K, dim=-1)      # [B, T, K]

        # Pairwise similarity among the K winners, via their vocab prototypes.
        protos = self.to_vocab.weight[topk_idx]                # [B, T, K, H]
        protos_n = F.normalize(protos, dim=-1)
        sim = torch.matmul(protos_n, protos_n.transpose(-1, -2))  # [B, T, K, K]
        sim = sim - torch.eye(K, device=x.device, dtype=sim.dtype)
        sim = F.relu(sim)                                       # only similar pairs compete

        # Softmax over the cohort, then each candidate is inhibited by a
        # similarity-weighted sum of its neighbours' activations.
        w = F.softmax(topk_vals / self.temperature, dim=-1)     # [B, T, K]
        inhibition = torch.matmul(sim, w.unsqueeze(-1)).squeeze(-1)   # [B, T, K]

        result_topk = F.relu(topk_vals * (1.0 - self.alpha * inhibition))

        # Sparse from_vocab: gather the K relevant columns of W, weighted sum.
        # Skips materializing the [B, T, V] sparse tensor.
        W_sel = self.from_vocab.weight.t()[topk_idx]            # [B, T, K, H]
        out = (result_topk.unsqueeze(-2) @ W_sel).squeeze(-2)   # [B, T, H]
        if self.from_vocab.bias is not None:
            out = out + self.from_vocab.bias

        return x + out
