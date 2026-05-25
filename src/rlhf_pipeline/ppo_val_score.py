"""
Validation-time proxy: mean R_phi on one rollout per val prompt (for PPO KL-β selection, §5.3).
Uses RM + temperature scaling from rm/calibration.json if present.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import List

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GlobalConfig
from .models_rm import CausalScalarRewardModel
from .paths import data_dir, rm_dir
from .rm_temperature import load_rm_temperature


@torch.inference_mode()
def mean_r_phi_on_val_rollouts(
    cfg: GlobalConfig,
    policy_dir: Path,
    *,
    max_prompts: int = 64,
) -> float:
    """
    Sample up to `max_prompts` prompts from the validation split, generate one continuation per
    prompt from the policy, return mean R_phi (calibrated) on full strings.
    """
    root = data_dir() / "hf_dataset"
    if not root.is_dir():
        return float("nan")
    dsd = load_from_disk(str(root))["validation"]
    rows: List[dict] = [dict(x) for x in dsd]
    if not rows:
        return float("nan")
    n = min(max_prompts, len(rows), max(1, int(cfg.ppo.n_rl_prompts) // 16 + 1))
    prompts = [r["prompt"] for r in rows[:n]]

    mps = getattr(torch.backends, "mps", None)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pdt = torch.float16
    elif mps and mps.is_available():
        device = torch.device("mps")
        pdt = torch.float32
    else:
        device = torch.device("cpu")
        pdt = torch.float32
    pol = AutoModelForCausalLM.from_pretrained(
        str(policy_dir),
        torch_dtype=pdt,
    ).to(device)
    pol.eval()
    tok = AutoTokenizer.from_pretrained(policy_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    meta = json.loads((rm_dir() / "meta.json").read_text(encoding="utf-8"))
    rm = CausalScalarRewardModel(
        meta["load_path"],
        dropout=cfg.model.reward_dropout_for_mc,
    ).to(device)
    rm.load_state_dict(torch.load(rm_dir() / "reward_model.pt", map_location=device)["state_dict"])
    rm_tok = AutoTokenizer.from_pretrained(rm_dir() / "tok", use_fast=True)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm.eval()
    if device.type == "cuda":
        rm = rm.half()
    if device.type == "mps":
        pol = pol.to(torch.float32)
        rm = rm.to(torch.float32)
    t_scale = load_rm_temperature()

    gkw = {
        "max_new_tokens": min(64, cfg.ppo.max_new_tokens),
        "pad_token_id": tok.pad_token_id,
    }
    gen_cfg = getattr(cfg.ppo, "generation_kwargs", {}) or {}
    gkw["do_sample"] = bool(gen_cfg.get("do_sample", True))
    if gkw["do_sample"]:
        gkw["top_p"] = float(gen_cfg.get("top_p", 0.95))
        gkw["temperature"] = float(gen_cfg.get("temperature", 0.75))
    acc = 0.0
    for ptxt in prompts:
        enc = tok(
            [ptxt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.model.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = pol.generate(**enc, **gkw)
        p_len = enc["input_ids"].shape[1]
        full = ptxt + tok.decode(gen[0, p_len:], skip_special_tokens=True)
        s = float(rm.score_texts(rm_tok, [full], cfg.model.max_length, device)[0].item())
        acc += s / t_scale
    del pol, rm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return acc / len(prompts)
