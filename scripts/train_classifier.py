#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-0.5B-Instruct on generated sufficiency examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "huggingface"))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(ROOT))
from classifier import DEFAULT_BASE_MODEL, LABELS, format_prompt  # noqa: E402

DATA_DIR = ROOT / "data" / "classifier"
OUTPUT_DIR = ROOT / "models" / "sufficiency-qwen-lora"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_classifier_data.py first")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in LABELS or not row.get("question"):
                raise ValueError(f"{path}:{line_number}: invalid classifier example")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


class SufficiencyDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        for row in rows:
            prompt = format_prompt(
                row["question"], row.get("contexts", []),
                target_paper_id=row.get("target_paper_id", row.get("paper_id")),
            )
            target = f" {row['label']}{tokenizer.eos_token}"
            # Tokenize prompt separately so training loss applies only to the
            # classification label and EOS, never to copying the evidence text.
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if len(target_ids) >= max_length:
                raise ValueError("max_length is too small to fit the target label")
            prompt_ids = prompt_ids[: max_length - len(target_ids)]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        input_ids, labels = self.items[index]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class CompletionOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        pad = self.tokenizer.pad_token_id
        input_ids, labels, masks = [], [], []
        for item in features:
            count = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad] * count)
            labels.append(item["labels"] + [-100] * count)
            masks.append(item["attention_mask"] + [0] * count)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    p.add_argument("--validation", type=Path, default=DATA_DIR / "validation.jsonl")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "mps":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    elif args.device == "auto" and not torch.cuda.is_available() and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("No accelerator detected; training on CPU will be slow.")

    train_rows = load_jsonl(args.train)
    validation_rows = load_jsonl(args.validation)
    print("Train label counts:", {label: sum(r["label"] == label for r in train_rows) for label in LABELS})
    print("Validation label counts:", {label: sum(r["label"] == label for r in validation_rows) for label in LABELS})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SufficiencyDataset(train_rows, tokenizer, args.max_length),
        eval_dataset=SufficiencyDataset(validation_rows, tokenizer, args.max_length),
        data_collator=CompletionOnlyCollator(tokenizer),
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "classifier_config.json").write_text(
        json.dumps({"base_model": args.base_model, "labels": LABELS,
                    "max_length": args.max_length, "seed": args.seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved LoRA classifier adapter and tokenizer to {args.output}")


if __name__ == "__main__":
    main()
