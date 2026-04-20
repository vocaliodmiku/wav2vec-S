"""Model-wide and training-wide config dataclasses.

Single source of truth for dimensions, context sizes, and hyperparameters
used by the pre-flight build. Keep in sync with configs/preflight.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FrontendConfig:
    model_name: str = "biaofu-xmu/wav2vec-S-Large-ft-960h"
    main_context: int = 8
    right_context: int = 4
    tap_layers: Tuple[int, ...] = (6, 12, 18, 24)  # 1-indexed wav2vec-S layers
    freeze: bool = True
    hidden_size: int = 1024  # wav2vec-S-Large


@dataclass
class TrunkConfig:
    num_levels: int = 5
    d: int = 1024
    num_heads: int = 8
    ffn_mult: int = 4
    conv_kernel: int = 15
    blocks_per_level: int = 2
    dropout: float = 0.1
    # predictor g_b
    predictor_rank: int = 128
    predictor_lag_k: int = 4
    # state summary s_b
    state_rank: int = 64
    # which trunk levels receive a frontend tap (len == num_levels; -1 = none)
    tap_injection: Tuple[int, ...] = (0, 1, 2, 3, -1)  # L0..L3 get taps, L4 none


@dataclass
class CPCConfig:
    hidden: int = 256
    horizon_k: int = 12
    num_negatives: int = 32


@dataclass
class LossConfig:
    alpha_pred: float = 0.1
    eta_cpc: float = 0.1
    pred_level_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)  # w_b for b=0..3


@dataclass
class PreflightConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    cpc: CPCConfig = field(default_factory=CPCConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    vocab_size: int = 32  # wav2vec-S CTC char vocab; verified at model init
