"""Supervised fine-tuning on chosen completions (SFT = reference for KL / DPO)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from .config import GlobalConfig
from .paths import sft_dir, ensure_dirs


def _load_prompts_chosen() -> Dataset:
    from .paths import data_dir

    root = data_dir() / "hf_dataset"
    if not root.exists():
        raise FileNotFoundError("Run the data stage first: python -m rlhf_pipeline.main data")
    dsd = load_from_disk(str(root))["train"]
    rows = []
    for ex in dsd:
        text = ex["prompt"] + ex["chosen"]
        rows.append({"text": text})
    return Dataset.from_list(rows)


def _gpt2_lora_targets() -> list:
    return ["c_attn", "c_fc", "c_proj"]


def run_sft(cfg: GlobalConfig) -> Path:
    ensure_dirs()
    out = sft_dir()
    out.mkdir(parents=True, exist_ok=True)
    ds = _load_prompts_chosen()
    tok = AutoTokenizer.from_pretrained(cfg.model.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype=torch.float16 if cfg.sft.fp16 and torch.cuda.is_available() else torch.float32,
    )
    if cfg.model.use_lora:
        tmods = _gpt2_lora_targets() if "gpt2" in cfg.model.base_model else ["q_proj", "v_proj", "k_proj", "o_proj"]
        pe = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=tmods,
        )
        model = get_peft_model(model, pe)

    def tokenize(examples):
        # Pad to max_length so every row has the same token length; otherwise batches
        # can mix e.g. 96 and 183 and collate may fail to stack.
        out = tok(
            examples["text"],
            truncation=True,
            max_length=cfg.model.max_length,
            padding="max_length",
        )
        # Causal LM loss: mask padding in labels (Trainer collator also handles padding,
        # but pre-computed labels keep behavior explicit across tokenizer versions).
        labels = []
        for ids, mask in zip(out["input_ids"], out["attention_mask"]):
            row = [tid if m else -100 for tid, m in zip(ids, mask)]
            labels.append(row)
        out["labels"] = labels
        return out

    tds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    targs = TrainingArguments(
        output_dir=str(out / "sft_trainer"),
        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        fp16=cfg.sft.fp16 and torch.cuda.is_available(),
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tds,
        data_collator=data_collator,
    )
    trainer.train()
    mdir = out / "model"
    trainer.save_model(str(mdir))
    tok.save_pretrained(str(mdir))
    (out / "meta.json").write_text(
        json.dumps({"base": cfg.model.base_model, "lora": cfg.model.use_lora}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved SFT to {mdir}")
    return mdir
