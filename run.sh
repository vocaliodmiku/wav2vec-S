#!/bin/bash
# HPSN-v2 launch script.
# Override any CLI flag via `./run.sh --flag value ...` (forwarded by "$@").
# set -euo pipefail

source ~/miniconda3/bin/activate neurospeech
export PYTHONPATH="$(pwd):$(pwd)/wav2vec-S-hf:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1,2

TARGETS_DIR=/scratch/jsm04005/fie24002/DATA/HPSN/targets_v2
MANIFEST=$TARGETS_DIR/manifest.csv
TARGETS_H5=$TARGETS_DIR/targets_train-clean-100.h5
STATS_NPZ=$TARGETS_DIR/target_stats.npz
BACKBONE=biaofu-xmu/wav2vec-S-Base

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — manifest (TextGrid ∩ WAV; validates words+phones tier structure).
# Output: $MANIFEST  +  manifest_dropped.csv
# Already done; uncomment to re-run.
# ─────────────────────────────────────────────────────────────────────────────
# python -m hpsn.data_prep.build_manifest \
#     --textgrid_root /scratch/jsm04005/fie24002/DATA/LibriSpeech/LibriSpeech \
#     --wav_root      /scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech \
#     --splits train-clean-100 \
#     --out_dir "$TARGETS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — extract per-utterance target stack to one keyed HDF5.
# Targets: log-mel(80) + phonological-features(14) + phone-id + word-id +
#          GPT-2 layer-4 hidden(768) + GPT-2 layer-8 hidden(768).
# Resumable: re-running skips utts already in the HDF5 (use --force to redo).
# Per-utt failures go to "$TARGETS_H5".errors.txt.
# Smoke-test first by adding `--limit 200`.
# ─────────────────────────────────────────────────────────────────────────────
# python -m hpsn.data_prep.extract_targets \
#     --manifest   "$MANIFEST" \
#     --out_h5     "$TARGETS_H5" \
#     --gpt2_model gpt2 \
#     --device     cuda

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2.5 — per-dim z-score stats for 2-D targets (gpt2_l4/l8 + log_mel).
# GPT-2 hidden states have ~5 outlier dims (Dettmers et al. 2022) whose
# magnitude is 100–1000× the rest; raw MSE would be dominated by them.
# This script computes per-dim mean/std on a 1k-utt sample (~10s) and saves
# them; HPSNV2Loss applies them at load time via TargetsHDF5Dataset(stats_path=).
# ─────────────────────────────────────────────────────────────────────────────
# python -m hpsn.data_prep.compute_target_stats \
#     --targets_h5 "$TARGETS_H5" \
#     --out        "$STATS_NPZ" \
#     --fields gpt2_l4,gpt2_l8,log_mel \
#     --n_sample 1000

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — optional v2 end-to-end smoke (one batch, real backbone, real
# targets, real loss + backward). Run this once after Stages 1/2/2.5 to
# verify everything wires up before committing to a long training run.
# ─────────────────────────────────────────────────────────────────────────────
# python -m hpsn.data_prep.check_v2_e2e \
#     --backbone   "$BACKBONE" \
#     --manifest   "$MANIFEST" \
#     --targets_h5 "$TARGETS_H5" \
#     --stats_path "$STATS_NPZ"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — HPSN-v3 training (v2 + frozen taps + L_restore).
# Differences from v2:
#   --level{1,2,3}_frozen_tap   → pin tap to per-band MEG-best wav2vec-S
#                                 layer (★1=2, ★2=6, ★3=12 from
#                                 baseline_roi_report_S temporal ROI).
#                                 Eliminates §17.1 tap-collapse failure.
#   --restore_prob 0.10         → on 10% of utts, replace one whole non-
#                                 silence phoneme span with RMS-matched
#                                 broadband noise.
#   --lambda_restore 0.3        → L1 must predict CLEAN log-mel on those
#                                 frames; only solvable via L2→L1 cross-
#                                 attention. Forces top-down to be
#                                 functional (full_hpsn_construction.md §19.3).
# ─────────────────────────────────────────────────────────────────────────────
# ACCELERATE_MIXED_PRECISION=bf16 python -m hpsn.training.train \
accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    -m hpsn.training.train \
    --backbone_model "$BACKBONE" \
    --use_v2_loss      true \
    --use_span_masking true \
    --n_iterations     2 \
    --targets_manifest "$MANIFEST" \
    --targets_h5       "$TARGETS_H5" \
    --target_stats     "$STATS_NPZ" \
    --output_dir ./hpsn_v3_output \
    --hpsn_dtype bf16 \
    --max_steps     50000 \
    --save_steps     3000 \
    --logging_steps    50 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --warmup_steps  2000 \
    --level1_tap_layers 1,2,3,4 \
    --level2_tap_layers 5,6,7,8 \
    --level3_tap_layers 9,10,11,12 \
    --level1_frozen_tap 2 \
    --level2_frozen_tap 6 \
    --level3_frozen_tap 12 \
    --level1_mask_prob 0.25 \
    --level2_mask_prob 0.20 \
    --level3_mask_prob 0.15 \
    --lambda_log_mel  1.0 \
    --lambda_phonol   0.5 \
    --lambda_phone_id 1.0 \
    --lambda_gpt2_l4  1.0 \
    --lambda_gpt2_l8  1.0 \
    --lambda_td       0.2 \
    --restore_prob    0.10 \
    --lambda_restore  0.3 \
    --main_context 8 --right_context 2 \
    "$@"

# ─────────────────────────────────────────────────────────────────────────────
# v1 training (kept for reference; not invoked).
# ─────────────────────────────────────────────────────────────────────────────
# ACCELERATE_MIXED_PRECISION=bf16 python -m hpsn.training.train \
#     --backbone_model "$BACKBONE" \
#     --libri_root /scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech/train-clean-100 \
#     --output_dir ./hpsn_output \
#     --hpsn_dtype fp32 \
#     --max_steps 50000 --save_steps 3000 --logging_steps 50 \
#     --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
#     --learning_rate 5e-4 --warmup_steps 2000 \
#     --level1_tap_layers 1,2,3,4 --level2_tap_layers 5,6,7,8 --level3_tap_layers 9,10,11,12 \
#     --level1_mask_prob 0.25 --level2_mask_prob 0.15 --level3_mask_prob 0.10 \
#     --inhib_num_codes 320 --inhib_top_k 64 \
#     --inhib_l1_num_codes 50 --inhib_l1_top_k 8 \
#     --lambda1 1.0 --lambda2 1.0 --lambda3 1.0 \
#     --lambda_td 0.2 \
#     --main_context 8 --right_context 2

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation snippets (kept for reference).
# ─────────────────────────────────────────────────────────────────────────────
# # Per-subject per-condition ridge (sensor space, MEG-native 50 Hz)
# python -m hpsn.evaluation.proc_5_hpsn_ridge \
#   --subj 01 --ses 0 --feat hpsn_l2 --space sensor \
#   --resample_opt MEG \
#   --ckpt ./checkpoints/hpsn_run1/final

# # ROI space
# python -m hpsn.evaluation.proc_5_hpsn_ridge \
#   --subj 01 --ses 0 --feat hpsn_l2 --space roi \
#   --ckpt ./checkpoints/hpsn_run1/final

# # Time-resolved
# python -m hpsn.evaluation.proc_5_hpsn_ridge \
#   --subj 01 --ses 0 --feat hpsn_l2 --space sensor \
#   --ckpt ./checkpoints/hpsn_run1/final --per_lag

# # Group stats: HPSN L2 vs. baseline_mid
# python -m hpsn.evaluation.aggregate_group \
#   --feat_a hpsn_l2 --feat_b baseline_mid --space sensor \
#   --subjects 01 02 03 04 05
