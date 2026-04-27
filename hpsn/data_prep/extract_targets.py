"""HPSN-v2 Phase 1 — extract per-utterance target stack into one keyed HDF5.

Inputs
------
* ``manifest.csv`` from ``build_manifest.py`` (one row per utterance).
* TextGrids (words + phones tiers) at ``row['textgrid_path']``.
* FLAC audio at ``row['wav_path']`` (16 kHz mono assumed; resampled if not).

Per-utterance output (HDF5 group keyed by ``utt_id``)
-----------------------------------------------------
* ``log_mel``         (T, 80)  float16 — 80-band log-mel, 50 Hz frame rate.
* ``phone_id``        (T,)     uint8   — ARPAbet phone class (0 = SIL, see
                                          ``arpabet.PHONE_VOCAB``).
* ``phonol_features`` (T, 14)  float16 — 14-d articulatory feature vector.
* ``word_id``         (T,)     int32   — utterance-local word index
                                          (0 = silence/no-word, 1..n_words).
* ``gpt2_l4``         (T, 768) float16 — GPT-2-small layer-4 hidden state of
                                          the aligned word, broadcast within
                                          the word's frame range.
* ``gpt2_l8``         (T, 768) float16 — same, layer 8.

Group attrs: ``duration_audio``, ``n_frames``, ``transcript``.

File-level attrs: ``frame_rate``, ``sr``, ``n_mels``, ``n_fft``, ``hop_length``,
``gpt2_model``, ``manifest_path``, ``built_at``.

Resumable
---------
If the output HDF5 already exists, utterances already present are skipped
unless ``--force`` is passed. Failed utterances go to a sidecar log
``<out_h5>.errors.txt`` and are retried on the next run.

Usage
-----
    python -m hpsn.data_prep.extract_targets \\
        --manifest /scratch/.../targets_v2/manifest.csv \\
        --out_h5   /scratch/.../targets_v2/targets_train-clean-100.h5 \\
        --gpt2_model gpt2 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
import warnings
from pathlib import Path

import h5py
import librosa
import numpy as np
import soundfile as sf
import textgrid as tg
import torch
from tqdm import tqdm

from .arpabet import N_FEATURES, N_PHONES, phone_to_features, phone_to_id

warnings.filterwarnings("ignore", category=UserWarning)


# Audio / framing constants
SR = 16_000
FRAME_RATE = 50
HOP_LENGTH = SR // FRAME_RATE  # 320 samples = 20 ms
N_FFT = 400
N_MELS = 80
F_MIN = 0
F_MAX = 8000

# GPT-2 layer indices to extract (small has 12 transformer layers + emb;
# hidden_states[i] for i in [0..12]; pick 4 = lexical-ish, 8 = contextual).
GPT2_L_LEXICAL = 4
GPT2_L_SEMANTIC = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True, help="CSV from build_manifest.py")
    p.add_argument("--out_h5", required=True, help="Output keyed HDF5 path")
    p.add_argument("--gpt2_model", default="gpt2",
                   help="HuggingFace model id (default: gpt2 / 124M)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (smoke test).")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if utt_id already exists in HDF5.")
    p.add_argument("--compression", default="gzip",
                   help="HDF5 compression: 'gzip', 'lzf', or '' (off).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Audio + log-mel
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Read FLAC/WAV → 16 kHz mono float32."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        # Rare on LibriSpeech; resample if needed.
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32, copy=False)


def compute_log_mel(wav: np.ndarray) -> np.ndarray:
    """Return ``[T, N_MELS]`` log-mel at ``FRAME_RATE`` Hz."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=F_MIN, fmax=F_MAX,
    )
    return np.log(mel + 1e-6).T.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TextGrid → frame-aligned phone fields
# ─────────────────────────────────────────────────────────────────────────────

def build_phone_arrays(
    phones_tier, n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(phone_id [T], phonol_features [T, 14])``.

    Frames not covered by any phone interval (or covered by silence labels)
    map to id 0 / zero-vector by default.
    """
    phone_id = np.zeros(n_frames, dtype=np.uint8)
    phonol = np.zeros((n_frames, N_FEATURES), dtype=np.float32)
    for iv in phones_tier.intervals:
        f0 = max(0, int(round(iv.minTime * FRAME_RATE)))
        f1 = min(n_frames, int(round(iv.maxTime * FRAME_RATE)))
        if f1 <= f0:
            continue
        pid = phone_to_id(iv.mark)
        if pid == 0:
            continue  # silence — leave default zeros
        phone_id[f0:f1] = pid
        phonol[f0:f1] = phone_to_features(iv.mark)
    return phone_id, phonol


# ─────────────────────────────────────────────────────────────────────────────
# Words tier → transcript + GPT-2-aligned hidden states
# ─────────────────────────────────────────────────────────────────────────────

def words_from_tier(words_tier) -> list[tuple[float, float, str]]:
    """Drop empty intervals; return ``[(t_start, t_end, word_lower)]``."""
    out = []
    for iv in words_tier.intervals:
        w = (iv.mark or "").strip()
        if not w:
            continue
        out.append((float(iv.minTime), float(iv.maxTime), w.lower()))
    return out


def gpt2_align_to_frames(
    word_intervals: list[tuple[float, float, str]],
    n_frames: int,
    tokenizer,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Run GPT-2 on the joined transcript, mean-pool per word, broadcast to frames.

    Returns ``(word_id [T], gpt2_l4 [T, H], gpt2_l8 [T, H], transcript)``.
    Frames outside any word interval get id 0 and zero hidden state.
    """
    word_id = np.zeros(n_frames, dtype=np.int32)
    h_dim = model.config.n_embd
    gpt2_l4 = np.zeros((n_frames, h_dim), dtype=np.float32)
    gpt2_l8 = np.zeros((n_frames, h_dim), dtype=np.float32)

    if not word_intervals:
        return word_id, gpt2_l4, gpt2_l8, ""

    transcript = " ".join(w for _, _, w in word_intervals)

    # Tokenize once with offset mapping so we can map BPE tokens back to words
    # via character spans.
    enc = tokenizer(transcript, return_offsets_mapping=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].cpu().numpy()  # (n_tokens, 2)

    if input_ids.shape[1] == 0:
        return word_id, gpt2_l4, gpt2_l8, transcript

    # GPT-2 max position embedding = 1024. LibriSpeech utterances rarely exceed
    # ~120 tokens; truncate defensively.
    if input_ids.shape[1] > 1024:
        input_ids = input_ids[:, :1024]
        offsets = offsets[:1024]

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    h4 = out.hidden_states[GPT2_L_LEXICAL][0].cpu().numpy().astype(np.float32)
    h8 = out.hidden_states[GPT2_L_SEMANTIC][0].cpu().numpy().astype(np.float32)

    # Walk the transcript and find each word's character span, then collect
    # tokens whose offset overlaps that span.
    n_tokens = offsets.shape[0]
    char_cursor = 0
    tok_cursor = 0
    for w_idx, (t_start, t_end, word_text) in enumerate(word_intervals):
        word_start = char_cursor
        word_end = char_cursor + len(word_text)
        char_cursor = word_end + 1  # +1 for the joining space

        # Advance tok_cursor past tokens that ended before this word
        while tok_cursor < n_tokens and offsets[tok_cursor, 1] <= word_start:
            tok_cursor += 1
        # Collect tokens whose span overlaps [word_start, word_end)
        tok_lo = tok_cursor
        tok_hi = tok_lo
        while tok_hi < n_tokens and offsets[tok_hi, 0] < word_end:
            tok_hi += 1

        f0 = max(0, int(round(t_start * FRAME_RATE)))
        f1 = min(n_frames, int(round(t_end * FRAME_RATE)))
        if f1 <= f0:
            continue
        word_id[f0:f1] = w_idx + 1  # 1-based; 0 reserved for "no word"
        if tok_hi > tok_lo:
            gpt2_l4[f0:f1] = h4[tok_lo:tok_hi].mean(axis=0)
            gpt2_l8[f0:f1] = h8[tok_lo:tok_hi].mean(axis=0)

    return word_id, gpt2_l4, gpt2_l8, transcript


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_utterance(
    row: dict,
    h5: h5py.File,
    tokenizer,
    model,
    device: torch.device,
    *,
    force: bool,
    compression: str | None,
) -> str:
    utt_id = row["utt_id"]
    if utt_id in h5 and not force:
        return "cached"
    if utt_id in h5 and force:
        del h5[utt_id]

    # 1. Audio + log-mel
    wav = load_audio(row["wav_path"])
    log_mel = compute_log_mel(wav)
    n_frames = int(log_mel.shape[0])
    duration_audio = float(wav.size / SR)

    # 2. TextGrid
    grid = tg.TextGrid.fromFile(row["textgrid_path"])
    words_tier = next(t for t in grid.tiers if t.name == "words")
    phones_tier = next(t for t in grid.tiers if t.name == "phones")

    # 3. Phone-aligned arrays
    phone_id, phonol = build_phone_arrays(phones_tier, n_frames)

    # 4. Word-aligned + GPT-2 hidden states
    word_intervals = words_from_tier(words_tier)
    word_id, gpt2_l4, gpt2_l8, transcript = gpt2_align_to_frames(
        word_intervals, n_frames, tokenizer, model, device,
    )

    # 5. Persist
    grp = h5.create_group(utt_id)
    kw = dict(compression=compression) if compression else {}
    grp.create_dataset("log_mel", data=log_mel.astype(np.float16), **kw)
    grp.create_dataset("phone_id", data=phone_id, **kw)
    grp.create_dataset("phonol_features", data=phonol.astype(np.float16), **kw)
    grp.create_dataset("word_id", data=word_id, **kw)
    grp.create_dataset("gpt2_l4", data=gpt2_l4.astype(np.float16), **kw)
    grp.create_dataset("gpt2_l8", data=gpt2_l8.astype(np.float16), **kw)
    grp.attrs["duration_audio"] = duration_audio
    grp.attrs["n_frames"] = n_frames
    grp.attrs["n_words"] = len(word_intervals)
    grp.attrs["transcript"] = transcript
    return "done"


def load_manifest(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    manifest_path = Path(args.manifest).resolve()
    out_h5 = Path(args.out_h5).resolve()
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    err_log = out_h5.with_suffix(out_h5.suffix + ".errors.txt")

    rows = load_manifest(manifest_path, args.limit)
    print(f"Manifest: {manifest_path}  ({len(rows)} rows)")
    print(f"Output:   {out_h5}")
    print(f"Device:   {device}")

    # Lazy import — keeps the script importable on CPU-only nodes.
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"Loading GPT-2 model: {args.gpt2_model}")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model)
    model = GPT2LMHeadModel.from_pretrained(
        args.gpt2_model, output_hidden_states=True,
    ).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    compression = args.compression or None
    counts = {"done": 0, "cached": 0, "error": 0}

    with h5py.File(out_h5, "a") as h5, err_log.open("a") as elog:
        # File-level attrs (idempotent: overwrite each run)
        h5.attrs["frame_rate"] = FRAME_RATE
        h5.attrs["sr"] = SR
        h5.attrs["n_mels"] = N_MELS
        h5.attrs["n_fft"] = N_FFT
        h5.attrs["hop_length"] = HOP_LENGTH
        h5.attrs["gpt2_model"] = args.gpt2_model
        h5.attrs["gpt2_layers"] = (GPT2_L_LEXICAL, GPT2_L_SEMANTIC)
        h5.attrs["n_phones"] = N_PHONES
        h5.attrs["n_phonol_features"] = N_FEATURES
        h5.attrs["manifest_path"] = str(manifest_path)
        h5.attrs["built_at"] = datetime.datetime.now().isoformat(timespec="seconds")

        pbar = tqdm(rows, desc="extract", unit="utt", smoothing=0.05)
        for row in pbar:
            try:
                status = process_utterance(
                    row, h5, tokenizer, model, device,
                    force=args.force, compression=compression,
                )
                counts[status] += 1
            except Exception as e:
                counts["error"] += 1
                elog.write(f"{row['utt_id']}\t{type(e).__name__}: {e}\n")
                elog.flush()
            pbar.set_postfix(counts)

    print()
    print("Done.")
    print(f"  fit:    {counts['done']}")
    print(f"  cached: {counts['cached']}")
    print(f"  errors: {counts['error']}  (see {err_log})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
