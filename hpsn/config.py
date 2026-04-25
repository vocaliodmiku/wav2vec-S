"""HPSN configuration dataclass (full 3-level model)."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HPSNConfig:
    # Backbone / architecture
    backbone_model: str = "biaofu-xmu/wav2vec-S-Base-ft-960h"
    hidden_dim: int = 768           # wav2vec-S-Base hidden size
    lstm_dim: int = 512             # internal level dim D (shared by L1/L2/L3)
    n_lstm_layers: int = 2          # transformer blocks per level (shared)
    n_attn_heads: int = 8
    dropout: float = 0.1

    # Layer tap bands (explicit tuples; index 0 = CNN output, 1..N = transformer layers)
    level1_tap_layers: Tuple[int, ...] = (1, 2, 3, 4)
    level2_tap_layers: Tuple[int, ...] = (5, 6, 7, 8)
    level3_tap_layers: Tuple[int, ...] = (9, 10, 11, 12)

    # Masking
    level1_mask_prob: float = 0.25  # ChunkMasker
    level2_mask_prob: float = 0.15  # FrameMasker
    level3_mask_prob: float = 0.10  # FrameMasker
    chunk_min_span: int = 2
    chunk_max_span: int = 5

    # Causal attention lookahead (frames). 0 = strictly causal.
    causal_lookahead: int = 0

    # Backbone chunk-causal context (frames of 20ms each).
    main_context: int = 8
    right_context: int = 2

    # Inhibition (L2 only) — codebook-style cohort competition
    inhib_num_codes: int = 320      # learned prototype vectors (~wav2vec2 codebook size)
    inhib_temperature: float = 1.0
    inhib_top_k: int = 64

    # Loss
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    loss_type: str = "l1"           # 'l1', 'mse', 'cosine'

    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 50000
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_audio_seconds: float = 15.0
    grad_clip: float = 1.0

    # Data
    libri_root: str = "/scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech/train-clean-100"
    sample_rate: int = 16000
    num_workers: int = 4
    max_train_samples: int = 0      # 0 = use full dataset

    # Logging / checkpointing
    output_dir: str = "./hpsn_output"
    logging_steps: int = 50
    save_steps: int = 5000
    seed: int = 42

    # Profiling (when True, measure per-region time and stop after `profile_steps`
    # optimizer steps).
    profile: bool = False
    profile_steps: int = 20

    # HPSN head compute precision. Parameters stay fp32 (for the optimizer);
    # forward/backward run in this dtype via torch.autocast. Loss is kept in fp32.
    # Options: 'fp32', 'bf16', 'fp16'.
    hpsn_dtype: str = "fp32"
