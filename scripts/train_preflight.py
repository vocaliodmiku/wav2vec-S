"""Pre-flight training entrypoint.

Wraps HPCSpeechPreflight with an HF Trainer. Only trainable parameters are
the trunk, predictors, CPC heads, and CTC head. The wav2vec-S frontend is
frozen (see FrontendConfig.freeze).
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from transformers import Trainer, TrainingArguments

from hpc_speech.config import PreflightConfig
from hpc_speech.data.librispeech import LibriSpeechCTC
from hpc_speech.model.hpc_speech import HPCSpeechPreflight
from hpc_speech.training.collator import CTCCollator


class HPCSpeechTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            label_lengths=inputs["label_lengths"],
        )
        # Log auxiliary terms via the trainer's logging hook.
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            logs: Dict[str, float] = {}
            if out.ctc_loss is not None:
                logs["train/ctc_loss"] = float(out.ctc_loss.detach())
            for b, (pl, cl, en) in enumerate(zip(out.pred_losses, out.cpc_losses, out.e_norms)):
                # pred_losses has num_levels-1 entries; e_norms matches.
                logs[f"train/pred_loss_L{b}"] = float(pl.detach())
                logs[f"train/e_norm_L{b}"] = float(en)
            for b, cl in enumerate(out.cpc_losses):
                logs[f"train/cpc_loss_L{b}"] = float(cl.detach())
            self.log(logs)
        return (out.loss, out) if return_outputs else out.loss


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--librispeech_root", default="/scratch/jsm04005/fie24002/LibriSpeech/LibriSpeech")
    p.add_argument("--train_split", default="train-clean-100")
    p.add_argument("--eval_split", default="dev-clean")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--eval_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--bf16", action="store_true", default=True)
    return p


def main():
    args = build_parser().parse_args()
    cfg = PreflightConfig()

    train_ds = LibriSpeechCTC(split=args.train_split, root=args.librispeech_root,
                              processor_name=cfg.frontend.model_name)
    eval_ds = LibriSpeechCTC(split=args.eval_split, root=args.librispeech_root,
                             processor_name=cfg.frontend.model_name)
    # Lock vocab_size to the tokenizer of the actual processor.
    cfg.vocab_size = train_ds.processor.tokenizer.vocab_size

    model = HPCSpeechPreflight(cfg)
    # Sanity: only the trunk + heads are trainable.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable / 1e6:.1f}M / total: {total / 1e6:.1f}M")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        eval_strategy="steps",
        save_total_limit=3,
        report_to=["tensorboard"],
    )

    trainer = HPCSpeechTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CTCCollator(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
