"""Direct Preference Optimization (DPO) on the same preference pairs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GlobalConfig
from .dpo_data import align_row_for_dpo
from .paths import data_dir, dpo_dir, sft_dir, ensure_dirs

try:
    from trl import DPOConfig, DPOTrainer
except Exception as e:  # noqa: BLE001
    DPOConfig = None
    DPOTrainer = None
    _TRL_ERR = e


def _dpo_device_supports_fp16() -> bool:
    return bool(torch.cuda.is_available())


def run_dpo(cfg: GlobalConfig) -> Path:
    if DPOTrainer is None:
        raise RuntimeError(f"Install trl for DPO: pip install trl ({_TRL_ERR})")
    ensure_dirs()
    out = dpo_dir()
    out.mkdir(parents=True, exist_ok=True)
    sft_p = sft_dir() / "model"
    if not sft_p.exists():
        raise FileNotFoundError("Run sft first.")
    dsd = load_from_disk(str(data_dir() / "hf_dataset"))["train"]
    rows = [
        {"prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"]}
        for x in dsd
    ]
    tok = AutoTokenizer.from_pretrained(sft_p, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # TRL: tokenize(prompt) must be a token-prefix of tokenize(prompt+completion); BPE can break
    # string splits — align so [RANK0] mismatch warnings and bad chosen_ids go away.
    rows = [align_row_for_dpo(tok, r) for r in rows]
    tds = Dataset.from_list(rows)

    # TRL/Transformers DPOConfig currently treats non-CUDA local runs most reliably as CPU.
    use_cpu = (
        os.environ.get("RLHF_DPO_USE_CPU", "").lower() in ("1", "true", "yes")
        or not torch.cuda.is_available()
    )
    if use_cpu:
        use_half = False
    else:
        use_half = bool(cfg.dpo.fp16 and _dpo_device_supports_fp16())
    dtype = torch.float16 if use_half else torch.float32
    # With precompute_ref_log_probs, TRL keeps a single SFT-initialized model and bakes in ref logp.
    # If you disable that in config, a second (frozen) ref model is loaded.
    pol = AutoModelForCausalLM.from_pretrained(sft_p, torch_dtype=dtype)
    pol.gradient_checkpointing_enable()
    pol.config.use_cache = False
    ref_model: AutoModelForCausalLM | None = None
    if not cfg.dpo.precompute_ref_log_probs:
        ref = AutoModelForCausalLM.from_pretrained(sft_p, torch_dtype=dtype)
        for p in ref.parameters():
            p.requires_grad = False
        ref.gradient_checkpointing_enable()
        ref.config.use_cache = False
        ref_model = ref

    common = dict(
        output_dir=str(out / "dpo_trainer"),
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        learning_rate=cfg.dpo.learning_rate,
        num_train_epochs=cfg.dpo.num_train_epochs,
        # CUDA: use AMP. MPS: models are already float16; Trainer keeps fp16/bf16 off to avoid
        # version quirks—memory savings come from float16 weights + gradient checkpointing.
        fp16=use_half and torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        beta=cfg.dpo.beta,
    )
    if use_cpu:
        common["use_cpu"] = True
    # TRL >= 1.2: DPOConfig has max_length (full sequence), not max_prompt_length (old API).
    dco_extra: dict = {
        "max_length": cfg.dpo.max_length,
    }
    if cfg.dpo.precompute_ref_log_probs:
        dco_extra["precompute_ref_log_probs"] = True
        dco_extra["precompute_ref_batch_size"] = cfg.dpo.precompute_ref_batch_size
    dargs = DPOConfig(**common, **dco_extra)
    kwargs = {
        "model": pol,
        "ref_model": ref_model,
        "args": dargs,
        "train_dataset": tds,
    }
    try:
        trainer = DPOTrainer(**kwargs, processing_class=tok)
    except TypeError:
        trainer = DPOTrainer(**kwargs, tokenizer=tok)
    trainer.train()
    savep = out / "model"
    trainer.model.save_pretrained(str(savep))
    tok.save_pretrained(str(savep))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "beta": cfg.dpo.beta,
                "precompute_ref_log_probs": cfg.dpo.precompute_ref_log_probs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved DPO model to {savep}")
    return savep
