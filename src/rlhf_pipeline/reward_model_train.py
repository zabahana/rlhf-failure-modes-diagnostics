"""Pairwise (Bradley--Terry style) training for the scalar reward model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from .config import GlobalConfig
from .models_rm import CausalScalarRewardModel
from .paths import data_dir, rm_dir, ensure_dirs, sft_dir


@torch.inference_mode()
def _pairwise_preference_deltas(
    model: CausalScalarRewardModel,
    tok: AutoTokenizer,
    rows: List[dict],
    device: torch.device,
    max_length: int,
    batch_size: int = 4,
) -> torch.Tensor:
    chs = [r["prompt"] + r["chosen"] for r in rows]
    rjs = [r["prompt"] + r["rejected"] for r in rows]
    dlist: List[torch.Tensor] = []
    for i in range(0, len(chs), batch_size):
        a = tok(
            chs[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        b = tok(
            rjs[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        a = {k: v.to(device) for k, v in a.items()}
        b = {k: v.to(device) for k, v in b.items()}
        sc = model(a["input_ids"], a["attention_mask"])
        sr = model(b["input_ids"], b["attention_mask"])
        dlist.append(sc - sr)
    return torch.cat(dlist, dim=0)


def fit_bt_temperature(
    deltas: torch.Tensor,
) -> Tuple[float, float]:
    """
    Minimize NLL of σ(Δ/T) on validation Bradley--Terry pairs. Returns (T*, mean_nll at T*).
    """
    if deltas.numel() < 2:
        return 1.0, float("nan")
    d = deltas.detach().float().cpu()
    best_t, best_nll = 1.0, float("inf")
    for t in np.logspace(-1.0, 1.0, 48):
        if t < 1e-6:
            continue
        nll = -F.logsigmoid(d / t).mean().item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, best_nll


def run_rm(cfg: GlobalConfig) -> Path:
    ensure_dirs()
    out = rm_dir()
    out.mkdir(parents=True, exist_ok=True)
    root = data_dir() / "hf_dataset"
    if not root.exists():
        raise FileNotFoundError("Run `data` first.")
    dsd = load_from_disk(str(root))["train"]
    rows: List[dict] = [dict(x) for x in dsd]
    if not rows:
        raise ValueError("No training rows after data stage")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_p = sft_dir() / "model"
    load_path = str(sft_p) if sft_p.exists() else cfg.model.base_model
    print(f"RM backbone: {load_path}")
    dtype = torch.float16 if cfg.rm.fp16 and torch.cuda.is_available() else torch.float32
    model = CausalScalarRewardModel(
        load_path, dropout=cfg.model.reward_dropout_for_mc, torch_dtype=dtype if load_path == cfg.model.base_model else None
    )
    if load_path != cfg.model.base_model and torch.cuda.is_available() and cfg.rm.fp16:
        model = model.half()
    model = model.to(device)
    model.train()
    tok = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.rm.learning_rate)

    accum = max(1, int(cfg.rm.gradient_accumulation_steps))
    step = 0
    for epoch in range(cfg.rm.num_train_epochs):
        random_order = list(range(len(rows)))
        # simple shuffle
        g = __import__("random").Random(cfg.seed + epoch)
        g.shuffle(random_order)
        it = 0
        tloss = 0.0
        pbar = tqdm(random_order, desc=f"rm epoch {epoch+1}")
        for j in pbar:
            r = rows[j]
            pc = r["prompt"] + r["chosen"]
            pr = r["prompt"] + r["rejected"]
            a = tok(pc, return_tensors="pt", truncation=True, max_length=cfg.model.max_length)
            b = tok(pr, return_tensors="pt", truncation=True, max_length=cfg.model.max_length)
            a = {k: v.to(device) for k, v in a.items()}
            b = {k: v.to(device) for k, v in b.items()}
            if step % accum == 0:
                opt.zero_grad()
            sc = model(a["input_ids"], a["attention_mask"])
            sr = model(b["input_ids"], b["attention_mask"])
            loss = -F.logsigmoid(sc - sr).mean() / accum
            loss.backward()
            tloss += loss.item() * accum
            it += 1
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            step += 1
        if step % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        print(f"epoch {epoch+1} mean nll~ {-tloss / max(1, it)}")

    temp_record: dict = {}
    if cfg.rm.calibrate_temperature:
        vroot = data_dir() / "hf_dataset"
        if vroot.is_dir():
            vds = load_from_disk(str(vroot))["validation"]
            vrows = [dict(x) for x in vds][: int(cfg.rm.temperature_val_cap)]
            if len(vrows) >= 4:
                model.eval()
                deltas = _pairwise_preference_deltas(
                    model, tok, vrows, device, cfg.model.max_length, batch_size=4
                )
                t_best, nll_v = fit_bt_temperature(deltas)
                temp_record = {
                    "temperature": t_best,
                    "val_mean_bt_nll": nll_v,
                    "n_val_pairs": len(vrows),
                }
                (out / "calibration.json").write_text(
                    json.dumps(
                        {
                            **temp_record,
                            "method": "min NLL of σ((R_φ^+−R_φ^−)/T) on val pairs (§4.4)",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"RM temperature scaling: T={t_best:.4f} (val BT NLL≈{nll_v:.4f}, n={len(vrows)})",
                    flush=True,
                )
            else:
                print("RM temperature scaling: skipped (need ≥4 val pairs).", flush=True)
        else:
            print("RM temperature scaling: skipped (no hf_dataset).", flush=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "load_path": load_path,
        },
        out / "reward_model.pt",
    )
    tok.save_pretrained(str(out / "tok"))
    meta_out = {"load_path": load_path, "dropout": cfg.model.reward_dropout_for_mc}
    if temp_record:
        meta_out["temperature"] = temp_record["temperature"]
        meta_out["calibration"] = "calibration.json"
    (out / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Saved reward model to {out}")
    return out
