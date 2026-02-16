from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ft.config import TrainConfig
from ft.data import load_jsonl, format_chat_example

def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path("data/sft_train.jsonl"))
    valid_rows = load_jsonl(Path("data/sft_valid.jsonl"))
    if train_rows and len(train_rows) < cfg.min_train_rows:
        mult = math.ceil(cfg.min_train_rows / len(train_rows))
        train_rows = (train_rows * mult)[: cfg.min_train_rows]
        print(f"Upsampled training rows to {len(train_rows)} for stable demo training.")

    train_prompts, train_targets = [], []
    for row in train_rows:
        prompt, target = format_chat_example(row)
        train_prompts.append(prompt)
        train_targets.append(target)

    valid_prompts, valid_targets = [], []
    for row in valid_rows:
        prompt, target = format_chat_example(row)
        valid_prompts.append(prompt)
        valid_targets.append(target)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(batch):
        model_inputs = tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=cfg.max_source_len,
        )
        labels = tokenizer(
            text_target=batch["target"],
            truncation=True,
            max_length=cfg.max_target_len,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_dict({"prompt": train_prompts, "target": train_targets}).map(
        tok, batched=True, remove_columns=["prompt", "target"]
    )
    valid_ds = Dataset.from_dict({"prompt": valid_prompts, "target": valid_targets}).map(
        tok, batched=True, remove_columns=["prompt", "target"]
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model)
    # model.resize_token_embeddings(len(tokenizer))

    # LoRA: parameter-efficient fine-tuning (train small adapters instead of whole model).
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Some older HF versions don't expose evaluation/save strategy params; filter for compatibility.
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    num_batches = math.ceil(len(train_rows) / cfg.train_batch_size)
    grad_accum = min(cfg.grad_accum, max(1, num_batches))
    if grad_accum != cfg.grad_accum:
        print(
            f"Adjusted gradient_accumulation_steps from {cfg.grad_accum} to {grad_accum} "
            f"(batches/epoch={num_batches})."
        )
    base_args = {
        "output_dir": str(cfg.out_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.train_batch_size,
        "per_device_eval_batch_size": cfg.train_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": cfg.lr,
        "logging_steps": 10,
        "report_to": "none",
        "fp16": False,  # CPU-friendly default
        "bf16": use_bf16,
        "dataloader_pin_memory": use_cuda,
    }

    valid_keys = TrainingArguments.__init__.__code__.co_varnames
    args_kwargs = {k: v for k, v in base_args.items() if k in valid_keys}
    if "evaluation_strategy" in valid_keys:
        args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in valid_keys:
        args_kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in valid_keys:
        args_kwargs["save_strategy"] = "epoch"

    args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(str(cfg.out_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(cfg.out_dir / "tokenizer"))

    # Save a mini training manifest (enterprise habit)
    manifest = {
        "base_model": cfg.base_model,
        "lora": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": list(cfg.lora_target_modules),
        },
        "max_source_len": cfg.max_source_len,
        "max_target_len": cfg.max_target_len,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "train_samples": len(train_rows),
        "valid_samples": len(valid_rows),
    }
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
