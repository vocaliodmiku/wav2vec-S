"""LibriSpeech folder dataset + feature-extractor collate (audio only)."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor


class LibriSpeechFolder(Dataset):
    """Walks a LibriSpeech split (expects ``*/*/*.trans.txt``).  Returns raw waveforms only."""

    def __init__(
        self,
        root: str,
        sample_rate: int = 16000,
        max_audio_seconds: float = 15.0,
        max_samples: int = 0,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.max_len = int(max_audio_seconds * sample_rate)

        self.paths: List[str] = []
        trans_files = sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt")))
        if not trans_files:
            raise FileNotFoundError(f"No *.trans.txt found under {root}")
        for tf in trans_files:
            chapter_dir = os.path.dirname(tf)
            with open(tf) as f:
                for line in f:
                    utt_id = line.strip().split(" ", 1)[0]
                    self.paths.append(os.path.join(chapter_dir, utt_id + ".flac"))
        if max_samples and max_samples > 0:
            self.paths = self.paths[:max_samples]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        waveform, sr = sf.read(self.paths[idx], dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz, got {sr} at {self.paths[idx]}")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if len(waveform) > self.max_len:
            waveform = waveform[: self.max_len]
        return waveform


@dataclass
class Collator:
    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, batch):
        proc = self.feature_extractor(
            list(batch),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        return {
            "input_values": proc["input_values"],
            "attention_mask": proc["attention_mask"],
        }


def build_dataloader(
    root: str,
    feature_extractor: Wav2Vec2FeatureExtractor,
    batch_size: int,
    num_workers: int = 4,
    max_audio_seconds: float = 15.0,
    max_samples: int = 0,
    shuffle: bool = True,
    sample_rate: int = 16000,
):
    dataset = LibriSpeechFolder(
        root=root,
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
        max_samples=max_samples,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collator(feature_extractor),
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
