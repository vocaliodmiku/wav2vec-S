"""HPSN-v2 Phase 0 — build a (TextGrid, WAV) manifest for a LibriSpeech split.

The TextGrid root and the WAV root may not be the same directory, and either
side may be missing files for some utterances. This script walks the TextGrid
tree, intersects with the WAV tree, validates that each TextGrid has the
expected ``words`` + ``phones`` tier structure, and writes a CSV manifest of
every utterance that survives.

CSV columns
-----------
    utt_id          e.g. "19-227-0001"
    speaker         e.g. "19"
    chapter         e.g. "227"
    split           e.g. "train-clean-100"
    wav_path        absolute path to the .flac
    textgrid_path   absolute path to the .TextGrid
    duration        seconds, from the TextGrid's xmax
    n_words         # word intervals
    n_phones        # phone intervals

A second CSV ``manifest_dropped.csv`` records dropped utterances + the reason
(missing WAV / TG parse error / unexpected tier structure / etc.).

Usage
-----
    python -m hpsn.data_prep.build_manifest \\
        --textgrid_root /scratch/jsm04005/fie24002/DATA/LibriSpeech/LibriSpeech \\
        --wav_root      /scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech \\
        --splits train-clean-100 \\
        --out_dir /scratch/jsm04005/fie24002/DATA/HPSN/targets_v2
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import textgrid as tg


REQUIRED_TIERS: tuple[str, ...] = ("words", "phones")


@dataclass
class ManifestRow:
    utt_id: str
    speaker: str
    chapter: str
    split: str
    wav_path: str
    textgrid_path: str
    duration: float
    n_words: int
    n_phones: int


@dataclass
class DropRow:
    utt_id: str
    split: str
    textgrid_path: str
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--textgrid_root", required=True,
        help="Root containing {split}/{spkr}/{chapter}/{utt}.TextGrid.",
    )
    p.add_argument(
        "--wav_root", required=True,
        help="Root containing {split}/{spkr}/{chapter}/{utt}.flac.",
    )
    p.add_argument(
        "--splits", default="train-clean-100",
        help="Comma-separated split names (default: train-clean-100).",
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Directory to write manifest.csv + manifest_dropped.csv.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="If set, stop after this many TextGrids (for smoke tests).",
    )
    return p.parse_args()


def discover_textgrids(textgrid_root: Path, split: str) -> list[Path]:
    """All ``*.TextGrid`` files under ``{textgrid_root}/{split}/``."""
    split_dir = textgrid_root / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.rglob("*.TextGrid"))


def utt_id_from_path(p: Path) -> str:
    """``19-227-0001.TextGrid`` → ``19-227-0001``."""
    return p.stem


def speaker_chapter(utt_id: str) -> tuple[str, str]:
    parts = utt_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"unexpected utt_id format: {utt_id!r}")
    return parts[0], parts[1]


def expected_wav_path(wav_root: Path, split: str, utt_id: str) -> Path:
    spkr, chapter = speaker_chapter(utt_id)
    return wav_root / split / spkr / chapter / f"{utt_id}.flac"


def validate_textgrid(path: Path) -> tuple[ManifestRow | None, str | None]:
    """Parse a TG and either return a manifest row or a reason for dropping."""
    try:
        tier_grid = tg.TextGrid.fromFile(str(path))
    except Exception as e:  # malformed TG, encoding issue, etc.
        return None, f"textgrid_parse_error: {type(e).__name__}: {e}"

    tier_names = [t.name for t in tier_grid.tiers]
    missing = [name for name in REQUIRED_TIERS if name not in tier_names]
    if missing:
        return None, f"missing_tiers: {missing} (have: {tier_names})"

    words_tier = next(t for t in tier_grid.tiers if t.name == "words")
    phones_tier = next(t for t in tier_grid.tiers if t.name == "phones")

    duration = float(tier_grid.maxTime)
    if duration <= 0.0:
        return None, f"non_positive_duration: {duration}"

    return (
        # filled in by caller (needs split + utt_id)
        ManifestRow(
            utt_id="", speaker="", chapter="", split="",
            wav_path="", textgrid_path="",
            duration=duration,
            n_words=len(words_tier.intervals),
            n_phones=len(phones_tier.intervals),
        ),
        None,
    )


def main() -> int:
    args = parse_args()
    textgrid_root = Path(args.textgrid_root).resolve()
    wav_root = Path(args.wav_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("ERROR: --splits is empty", file=sys.stderr)
        return 2

    rows: list[ManifestRow] = []
    drops: list[DropRow] = []
    drop_counter: Counter[str] = Counter()

    for split in splits:
        tg_paths = discover_textgrids(textgrid_root, split)
        print(f"[{split}] discovered {len(tg_paths)} TextGrids")
        if args.limit is not None:
            tg_paths = tg_paths[: args.limit]
            print(f"[{split}] limited to {len(tg_paths)} (--limit)")

        for i, tg_path in enumerate(tg_paths, 1):
            if i % 2000 == 0:
                print(f"[{split}]   {i}/{len(tg_paths)} processed")

            utt_id = utt_id_from_path(tg_path)
            try:
                spkr, chapter = speaker_chapter(utt_id)
            except ValueError as e:
                drops.append(DropRow(utt_id, split, str(tg_path), f"bad_utt_id: {e}"))
                drop_counter["bad_utt_id"] += 1
                continue

            wav_path = expected_wav_path(wav_root, split, utt_id)
            if not wav_path.is_file():
                drops.append(DropRow(utt_id, split, str(tg_path), "wav_missing"))
                drop_counter["wav_missing"] += 1
                continue

            row, reason = validate_textgrid(tg_path)
            if row is None:
                drops.append(DropRow(utt_id, split, str(tg_path), reason or "unknown"))
                drop_counter[(reason or "unknown").split(":", 1)[0]] += 1
                continue

            row.utt_id = utt_id
            row.speaker = spkr
            row.chapter = chapter
            row.split = split
            row.wav_path = str(wav_path)
            row.textgrid_path = str(tg_path)
            rows.append(row)

    # Write manifest
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "utt_id", "speaker", "chapter", "split",
            "wav_path", "textgrid_path",
            "duration", "n_words", "n_phones",
        ])
        for r in rows:
            w.writerow([
                r.utt_id, r.speaker, r.chapter, r.split,
                r.wav_path, r.textgrid_path,
                f"{r.duration:.6f}", r.n_words, r.n_phones,
            ])

    drops_path = out_dir / "manifest_dropped.csv"
    with drops_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "split", "textgrid_path", "reason"])
        for d in drops:
            w.writerow([d.utt_id, d.split, d.textgrid_path, d.reason])

    total_dur_h = sum(r.duration for r in rows) / 3600.0
    print()
    print(f"Wrote {manifest_path}")
    print(f"  kept:    {len(rows)} utterances ({total_dur_h:.2f} h total)")
    print(f"Wrote {drops_path}")
    print(f"  dropped: {len(drops)}")
    if drop_counter:
        print("  drop reasons:")
        for reason, n in drop_counter.most_common():
            print(f"    {reason:30s}  {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
