"""Dynamic padding collator for CTC training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class CTCCollator:
    pad_value: float = 0.0
    label_pad: int = -100

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_values = [b["input_values"] for b in batch]
        labels = [b["labels"] for b in batch]
        B = len(batch)
        max_T = max(x.shape[0] for x in input_values)
        max_L = max(l.shape[0] for l in labels)

        iv = torch.full((B, max_T), self.pad_value, dtype=input_values[0].dtype)
        am = torch.zeros((B, max_T), dtype=torch.long)
        lab = torch.full((B, max_L), self.label_pad, dtype=torch.long)
        lab_len = torch.zeros((B,), dtype=torch.long)

        for i, (x, l) in enumerate(zip(input_values, labels)):
            iv[i, : x.shape[0]] = x
            am[i, : x.shape[0]] = 1
            lab[i, : l.shape[0]] = l
            lab_len[i] = l.shape[0]

        return {
            "input_values": iv,
            "attention_mask": am,
            "labels": lab,
            "label_lengths": lab_len,
        }
