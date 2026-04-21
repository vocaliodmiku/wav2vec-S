"""HPSN Level 1 (phonemic) and Level 2 (lexical) modules."""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossLayerAttention
from .inhibition import LateralInhibitionGate


class HPSNLevel1(nn.Module):
    """Phonemic level. Bottom-up LSTM + top-down cross-attention from Level 2."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        lstm_dim: int = 512,
        n_lstm_layers: int = 2,
        n_attn_heads: int = 8,
        dropout: float = 0.1,
        causal_lookahead: int = 0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, lstm_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.td_cross_attn = CrossLayerAttention(
            lstm_dim, n_heads=n_attn_heads, dropout=dropout, lookahead=causal_lookahead
        )
        self.combine = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.GELU(),
            nn.Linear(lstm_dim, lstm_dim),
        )
        self.recon_head = nn.Linear(lstm_dim, hidden_dim)

    def forward(
        self, masked_input: torch.Tensor, top_down_signal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(masked_input)
        bu_out, _ = self.lstm(x)
        td_out = self.td_cross_attn(query=bu_out, key_value=top_down_signal)
        combined = self.combine(torch.cat([bu_out, td_out], dim=-1))
        recon = self.recon_head(combined)
        return combined, recon


class HPSNLevel2(nn.Module):
    """Lexical level. LSTM + lateral inhibition + top-down predictor to Level 1."""

    def __init__(
        self,
        hidden_dim: int = 1024,
        lstm_dim: int = 512,
        n_lstm_layers: int = 2,
        vocab_size: int = 32000,
        n_attn_heads: int = 8,
        dropout: float = 0.1,
        causal_lookahead: int = 0,
        inhib_temperature: float = 1.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, lstm_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.inhib_gate = LateralInhibitionGate(lstm_dim, vocab_size, temperature=inhib_temperature)
        self.bu_cross_attn = CrossLayerAttention(
            lstm_dim, n_heads=n_attn_heads, dropout=dropout, lookahead=causal_lookahead
        )
        self.combine = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.GELU(),
            nn.Linear(lstm_dim, lstm_dim),
        )
        self.td_predictor = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.GELU(),
            nn.Linear(lstm_dim, lstm_dim),
        )
        self.recon_head = nn.Linear(lstm_dim, hidden_dim)

    def forward(
        self, masked_input: torch.Tensor, bu_error: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(masked_input)
        lstm_out, _ = self.lstm(x)
        if bu_error is not None:
            bu_signal = self.bu_cross_attn(query=lstm_out, key_value=bu_error)
            output = self.combine(torch.cat([lstm_out, bu_signal], dim=-1))
        else:
            output = lstm_out
        output = self.inhib_gate(output)
        top_down = self.td_predictor(output)
        recon = self.recon_head(output)
        return output, top_down, recon
