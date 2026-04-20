"""Run all pre-flight probes against a trained checkpoint and emit a report.

Usage:
    python scripts/run_probes.py \
        --checkpoint /path/to/checkpoint \
        --provo_jsonl data/provo_aligned.jsonl \
        --dev_split dev-clean \
        --out preflight_report.json
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from hpc_speech.config import PreflightConfig
from hpc_speech.data.librispeech import LibriSpeechCTC
from hpc_speech.data.provo import ProvoDataset
from hpc_speech.model.hpc_speech import HPCSpeechPreflight
from hpc_speech.probe.report import build_report
from hpc_speech.probe.shuffle_eb import shuffle_eb_probe
from hpc_speech.probe.surprisal import compute_surprisal_correlation
from hpc_speech.training.collator import CTCCollator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--provo_jsonl", required=True)
    p.add_argument("--librispeech_root", default="/scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech")
    p.add_argument("--dev_split", default="dev-clean")
    p.add_argument("--out", default="preflight_report.json")
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    cfg = PreflightConfig()
    model = HPCSpeechPreflight(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    # Accept both raw state_dict and HF Trainer checkpoints.
    if "model" in state:
        state = state["model"]
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    provo = ProvoDataset(args.provo_jsonl, processor_name=cfg.frontend.model_name)
    dev = LibriSpeechCTC(split=args.dev_split, root=args.librispeech_root,
                        processor_name=cfg.frontend.model_name)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=CTCCollator(),
                            shuffle=False, num_workers=2)

    surprisal = compute_surprisal_correlation(model, provo)
    shuffle = shuffle_eb_probe(model, dev_loader)

    report = build_report(surprisal, shuffle, args.out)
    print(f"wrote {args.out}")
    print(f"strict criteria: {report['criteria_strict']}")
    print(f"lenient criteria: {report['criteria_lenient']}")
    if report["warnings"]:
        print("warnings:", report["warnings"])


if __name__ == "__main__":
    main()
