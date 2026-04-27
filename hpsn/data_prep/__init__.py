"""HPSN-v2 offline data preparation.

Phase 0: build_manifest — walks the TextGrid + WAV trees, intersects, validates.
Phase 1: extract_targets — extracts the v2 target stack (log-mel + phonological
         features + phone IDs + GPT-2 hidden states) into a single keyed HDF5.

Usage::

    python -m hpsn.data_prep.build_manifest \\
        --textgrid_root /scratch/jsm04005/fie24002/DATA/LibriSpeech/LibriSpeech \\
        --wav_root      /scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech \\
        --splits train-clean-100 \\
        --out /scratch/jsm04005/fie24002/DATA/HPSN/targets_v2/manifest.csv
"""
