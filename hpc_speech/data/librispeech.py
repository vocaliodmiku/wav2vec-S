"""LibriSpeech dataset for CTC training and dev-clean evaluation.

Uses the Wav2Vec2Processor from the wav2vec-S checkpoint to tokenize
transcripts to character IDs (blank = 0 by convention).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor


LIBRISPEECH_ROOT_DEFAULT = "/scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech"


@dataclass
class LibriSample:
    audio_path: str
    text: str
    utt_id: str


def _scan_split(root: str, split: str) -> List[LibriSample]:
    split_dir = os.path.join(root, split)
    out: List[LibriSample] = []
    for trans_file in glob.glob(os.path.join(split_dir, "*", "*", "*.trans.txt")):
        chapter_dir = os.path.dirname(trans_file)
        with open(trans_file) as f:
            for line in f:
                utt_id, text = line.strip().split(" ", 1)
                audio = os.path.join(chapter_dir, utt_id + ".flac")
                out.append(LibriSample(audio_path=audio, text=text, utt_id=utt_id))
    return out


class LibriSpeechCTC(Dataset):
    """Character-level CTC dataset over a LibriSpeech split."""

    def __init__(
        self,
        split: str = "train-clean-100",
        root: str = LIBRISPEECH_ROOT_DEFAULT,
        processor_name: str = "biaofu-xmu/wav2vec-S-Large-ft-960h",
        max_seconds: float = 16.0,
        sampling_rate: int = 16000,
    ):
        self.samples = _scan_split(root, split)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        self.max_samples = int(max_seconds * sampling_rate)
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        waveform, sr = sf.read(s.audio_path)
        assert sr == self.sampling_rate, f"expected {self.sampling_rate} Hz, got {sr}"
        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]
        feats = self.processor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        labels = self.processor(text=s.text, return_tensors="pt").input_ids[0]
        return {
            "input_values": feats.input_values[0],
            "labels": labels,
            "utt_id": s.utt_id,
            "text": s.text,
        }
