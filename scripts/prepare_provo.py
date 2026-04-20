"""Prepare Provo corpus for the pre-flight surprisal probe.

Inputs (expected on disk):
    --provo_audio_dir   directory with Provo paragraph .wav files (16 kHz)
    --provo_cloze_csv   Provo cloze norms (per-word cloze probability; from OSF)
    --mfa_dict          MFA pronunciation dictionary (english_us_arpa or similar)
    --mfa_acoustic      MFA acoustic model (english_us_arpa)

Output:
    --out               JSONL where each line is one paragraph with aligned words,
                        cloze probabilities, and GPT-2 surprisal.

Pipeline:
    1. Force-align each paragraph's audio to its transcript via MFA.
    2. Join alignments to Provo cloze table on (paragraph_id, word_idx).
    3. Score each word's GPT-2 small surprisal using the running paragraph
       context (sum of token surprisals over the word's BPE span).
    4. Emit JSONL.

This script is a stub: it wires the interfaces but assumes the caller has
MFA and gpt2-small available. Spot-check 20 random alignments after running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import soundfile as sf
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def run_mfa(audio_dir: str, transcripts_dir: str, mfa_dict: str, mfa_acoustic: str, out_dir: str) -> None:
    """Invoke Montreal Forced Aligner. Assumes `mfa` CLI is available."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["mfa", "align", audio_dir, mfa_dict, mfa_acoustic, out_dir, "--clean", "--overwrite"],
        check=True,
    )


def parse_textgrid(path: str) -> List[Tuple[float, float, str]]:
    """Minimal TextGrid parser returning (start_s, end_s, label) for the 'words' tier."""
    # Use praatio if available; fall back to manual parsing.
    try:
        from praatio import textgrid as tg  # type: ignore
    except ImportError as e:
        raise ImportError("Install praatio: pip install praatio") from e
    tier = tg.openTextgrid(path, includeEmptyIntervals=False).getTier("words")
    return [(iv.start, iv.end, iv.label) for iv in tier.entries if iv.label.strip()]


def gpt2_surprisal_per_word(
    paragraph_tokens: List[str], model: GPT2LMHeadModel, tok: GPT2TokenizerFast
) -> List[float]:
    """Return per-word surprisal (nats) summed over the word's BPE span."""
    text = " ".join(paragraph_tokens)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (T, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    # Token t's log-prob of the actual next token is log_probs[t, input_ids[0, t+1]].
    token_surprisals = []
    ids = input_ids[0].tolist()
    for t in range(len(ids) - 1):
        token_surprisals.append(-float(log_probs[t, ids[t + 1]]))

    # Map each word to the BPE span that covers its characters.
    word_starts = [0]
    for w in paragraph_tokens[:-1]:
        word_starts.append(word_starts[-1] + len(w) + 1)  # +1 for space
    word_ends = [s + len(w) for s, w in zip(word_starts, paragraph_tokens)]

    out: List[float] = []
    for ws, we in zip(word_starts, word_ends):
        # Sum surprisal over BPE tokens whose character span overlaps [ws, we).
        total = 0.0
        for t, (bs, be) in enumerate(offsets[1:], start=0):  # align with token_surprisals
            if t >= len(token_surprisals):
                break
            if be <= ws or bs >= we:
                continue
            total += token_surprisals[t]
        out.append(total if total > 0 else math.nan)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provo_audio_dir", required=True)
    p.add_argument("--provo_cloze_csv", required=True)
    p.add_argument("--transcripts_dir", required=True,
                   help="Dir with one .lab/.txt per audio file containing the transcript")
    p.add_argument("--mfa_dict", required=True)
    p.add_argument("--mfa_acoustic", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        run_mfa(args.provo_audio_dir, args.transcripts_dir, args.mfa_dict, args.mfa_acoustic, td)
        cloze = pd.read_csv(args.provo_cloze_csv)
        # Expected cloze columns: paragraph_id, word_idx, word, cloze_prob.

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        lm = GPT2LMHeadModel.from_pretrained("gpt2").eval()

        records: List[Dict] = []
        for tg_path in sorted(Path(td).rglob("*.TextGrid")):
            utt_id = tg_path.stem
            audio_path = str(Path(args.provo_audio_dir) / (utt_id + ".wav"))
            if not os.path.exists(audio_path):
                continue
            words = parse_textgrid(str(tg_path))
            word_strings = [w for _, _, w in words]
            surprisals = gpt2_surprisal_per_word(word_strings, lm, tok)
            cloze_for_utt = cloze[cloze["paragraph_id"] == utt_id].sort_values("word_idx")
            cloze_probs = cloze_for_utt["cloze_prob"].tolist()
            # Length may mismatch if MFA drops tokens; use per-word min.
            n = min(len(words), len(surprisals), len(cloze_probs))
            out_words = [
                {
                    "start_s": words[i][0],
                    "end_s": words[i][1],
                    "word": words[i][2],
                    "cloze_prob": float(cloze_probs[i]),
                    "gpt2_surprisal": float(surprisals[i]) if math.isfinite(surprisals[i]) else None,
                }
                for i in range(n)
            ]
            sr = sf.info(audio_path).samplerate
            records.append({
                "utt_id": utt_id,
                "audio_path": audio_path,
                "sampling_rate": sr,
                "words": out_words,
            })

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} utterances to {args.out}")


if __name__ == "__main__":
    main()
