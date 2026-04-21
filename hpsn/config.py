"""HPSN configuration dataclass."""
from dataclasses import dataclass


@dataclass
class HPSNConfig:
    # Backbone / architecture
    backbone_model: str = "biaofu-xmu/wav2vec-S-Large-ft-960h"
    hidden_dim: int = 1024          # wav2vec-S-Large hidden size
    lstm_dim: int = 512
    n_lstm_layers: int = 2
    vocab_size: int = 32000         # for inhibition gate (not the CTC vocab)
    n_attn_heads: int = 8
    dropout: float = 0.1

    # Layer tap bands (inclusive; index 0 = CNN output, 1..N = transformer layers)
    tap_acoustic_start: int = 1
    tap_acoustic_end: int = 8
    tap_lexical_start: int = 13
    tap_lexical_end: int = 20

    # Masking
    mask_prob_acoustic: float = 0.25
    mask_prob_lexical: float = 0.15
    chunk_min_span: int = 2
    chunk_max_span: int = 5

    # Causal attention lookahead (frames).  0 = strictly causal; 9 matches backbone.
    causal_lookahead: int = 0

    # Backbone chunk-causal context (frames of 20ms each)
    main_context: int = 8
    right_context: int = 2

    # Iterative refinement (run Level 2 a second time with bottom-up error)
    iterative_refine: bool = False

    # Inhibition
    inhib_temperature: float = 1.0

    # Loss
    lambda1: float = 1.0
    lambda2: float = 1.0
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
    output_dir: str = "./preflight_output"
    logging_steps: int = 50
    save_steps: int = 5000
    seed: int = 42
