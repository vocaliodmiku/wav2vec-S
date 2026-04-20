"""Provo corpus loader for the pre-flight surprisal probe.

Expects a JSONL produced by scripts/prepare_provo.py with entries:
    {
      "utt_id": str,
      "audio_path": str,
      "sampling_rate": int,
      "words": [
        {"start_s": float, "end_s": float, "word": str,
         "cloze_prob": float, "gpt2_surprisal": float},
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor


@dataclass
class ProvoWord:
    start_s: float
    end_s: float
    word: str
    cloze_prob: float
    gpt2_surprisal: float


class ProvoDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor_name: str = "biaofu-xmu/wav2vec-S-Large-ft-960h",
        sampling_rate: int = 16000,
    ):
        self.entries: List[Dict] = []
        with open(jsonl_path) as f:
            for line in f:
                self.entries.append(json.loads(line))
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        e = self.entries[idx]
        waveform, sr = sf.read(e["audio_path"])
        assert sr == self.sampling_rate
        feats = self.processor(waveform, sampling_rate=sr, return_tensors="pt")
        return {
            "utt_id": e["utt_id"],
            "input_values": feats.input_values[0],
            "words": [ProvoWord(**w) for w in e["words"]],
        }
