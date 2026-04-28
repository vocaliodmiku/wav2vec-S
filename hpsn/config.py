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

    # Frozen-one-hot tap selection (HPSN-v2). -1 = learnable (v1 default).
    # Non-negative = pin the tap to that backbone layer index (must be in the
    # corresponding level{N}_tap_layers tuple). Set per band once the wav2vec-S
    # layer-baseline identifies the brain-best layer per band.
    level1_frozen_tap: int = -1
    level2_frozen_tap: int = -1
    level3_frozen_tap: int = -1

    # Iterative predictive-coding refinement.
    # 1 = single top-down sweep (v1 behavior, no error pathway).
    # >=2 = on each extra iteration, compute bottom-up error
    #       eps2 = error_proj_2(l2_repr.detach()) - mu2 and re-run L3 with
    #       cross_kv=eps2, then L2 with the refreshed mu2', then L1. The
    #       construction doc §5.4 protocol; matters only when n_iterations >= 2.
    n_iterations: int = 1

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

    # Inhibition — codebook-style cohort competition.
    # L2 (lexical/sublexical): ~wav2vec2 codebook size.
    inhib_num_codes: int = 320
    inhib_top_k: int = 64
    # L1 (phoneme-level): smaller codebook, narrower cohort — categorical perception.
    inhib_l1_num_codes: int = 50
    inhib_l1_top_k: int = 8
    # Shared temperature for cohort softmax (both levels).
    inhib_temperature: float = 1.0

    # Loss
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    lambda_td: float = 0.2          # weight on top-down prediction alignment loss
    loss_type: str = "l1"           # 'l1', 'mse', 'cosine'  (recon term only)

    # ── HPSN-v2 — multi-target reconstruction ──
    # Switch the loss + masker stack to the v2 design (per-region brain-relevant
    # targets + phoneme/word span masking). Defaults preserve v1 behavior.
    use_v2_loss: bool = False
    use_span_masking: bool = False

    # Target dimensions (must match Phase 1 extract_targets.py output).
    n_log_mel: int = 80
    n_phonol_features: int = 14
    n_phones: int = 40
    gpt2_hidden_dim: int = 768

    # v2 per-target loss weights.
    lambda_log_mel: float = 1.0
    lambda_phonol: float = 0.5
    lambda_phone_id: float = 1.0
    lambda_gpt2_l4: float = 1.0
    lambda_gpt2_l8: float = 1.0

    # L_restore — phonemic-restoration auxiliary (full_hpsn_construction.md §19.3).
    # On `restore_prob` fraction of utterances the dataset replaces one whole
    # non-silence phoneme span in the waveform with RMS-matched broadband
    # noise, then asks L1 to predict the *clean* log-mel target on those
    # frames. Forces the L2→L1 cross-attention pathway to carry signal
    # because the bottom-up acoustic input no longer contains the answer.
    # Default 0 keeps v2 behavior unchanged.
    lambda_restore: float = 0.0
    restore_prob: float = 0.0

    # v2 dataset (consumed only when use_v2_loss is True).
    targets_manifest: str = ""
    targets_h5: str = ""
    target_stats: str = ""          # optional .npz with per-dim mean/std

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
