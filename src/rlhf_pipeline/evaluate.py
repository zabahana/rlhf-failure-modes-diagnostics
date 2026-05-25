"""Lightweight eval: proxy reward, optional mock or Anthropic judge, Δ_hack-style gap, mean u."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from .config import GlobalConfig
from .judge_mock import mock_judge_score
from .judge_resolution import (
    resolve_primary_judge,
    resolve_second_judge,
    sleep_after_judge_call,
)
from .load_env import load_code_dotenv
from .models_rm import CausalScalarRewardModel
from .paths import data_dir, eval_dir, rm_dir, ensure_dirs
from .rm_temperature import apply_temperature_scalar, load_rm_temperature


def _resolve_judge() -> Tuple[Callable[[str], float], str]:
    """Back-compat: delegates to :mod:`judge_resolution`."""
    return resolve_primary_judge()


def run_eval(cfg: GlobalConfig) -> Path:
    load_code_dotenv()
    judge_fn, judge_name = resolve_primary_judge()
    j2_fn, j2_name = resolve_second_judge(judge_name)
    if j2_name:
        print(
            f"Eval judge: {judge_name} (primary); second: {j2_name}. "
            "Set RLHF_JUDGE_2=none to disable. See .env.example.",
            flush=True,
        )
    else:
        print(
            f"Eval judge: {judge_name}. "
            "OpenAI/Anthropic: set API keys; RLHF_JUDGE=openai|anthropic|mock; "
            "second judge: RLHF_JUDGE_2=openai|anthropic.",
            flush=True,
        )

    ensure_dirs()
    out = eval_dir()
    out.mkdir(parents=True, exist_ok=True)
    dsd = load_from_disk(str(data_dir() / "hf_dataset"))["test"]
    n = max(1, int(cfg.eval_s.n_examples))
    rows: List[dict] = [dict(x) for x in dsd][:n]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpath = rm_dir() / "reward_model.pt"
    meta = json.loads((rm_dir() / "meta.json").read_text(encoding="utf-8"))
    rm = CausalScalarRewardModel(meta["load_path"], dropout=cfg.model.reward_dropout_for_mc)
    rm.load_state_dict(torch.load(mpath, map_location="cpu")["state_dict"])
    rm = rm.to(device)
    if device.type == "cuda":
        rm = rm.half()
    tok = AutoTokenizer.from_pretrained(rm_dir() / "tok", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm.eval()
    t_rm = load_rm_temperature()

    results = []
    for r in rows:
        full = r["prompt"] + r["chosen"]
        with torch.no_grad():
            s = float(
                rm.score_texts(
                    tok,
                    [full],
                    cfg.model.max_length,
                    device,
                )[0]
            )
        with torch.no_grad():
            _, uu = rm.score_texts_with_dropout_std(
                tok, [full], cfg.model.max_length, device, k=cfg.ppo.num_mc
            )
            uu = float(uu[0])
        s = apply_temperature_scalar(s, t_rm)
        uu = apply_temperature_scalar(uu, t_rm)
        jd = float(judge_fn(full))
        sleep_after_judge_call(judge_name)
        jd2: float | None = None
        if j2_fn is not None:
            jd2 = float(j2_fn(full))
            sleep_after_judge_call(j2_name)
        row: dict = {
            "R_phi": s,
            "u": uu,
            "R_dagger": jd,
            "judge": judge_name,
            "delta_slice": s - jd,
        }
        if jd2 is not None:
            row["R_dagger_2"] = jd2
            row["judge_2"] = j2_name
            row["judge_disagreement_abs"] = abs(float(jd) - float(jd2))
        results.append(row)
    mean_r = sum(x["R_phi"] for x in results) / len(results)
    mean_j = sum(x["R_dagger"] for x in results) / len(results)
    mean_u = sum(x["u"] for x in results) / len(results)
    dhack = mean_r - mean_j
    note = (
        f"Primary R_dagger: judge={judge_name}. RM temp T={t_rm:.4f} (rm/calibration.json; no file => 1). "
        "Set RLHF_JUDGE_2=openai for a second 1–10 score (costs another API per example)."
    )
    summary: dict = {
        "mean_R_phi": mean_r,
        "mean_R_dagger": mean_j,
        "judge": judge_name,
        "Delta_hack_proxy": dhack,
        "mean_u": mean_u,
        "n": len(results),
        "note": note,
    }
    if any("R_dagger_2" in r for r in results):
        j2s = [float(r["R_dagger_2"]) for r in results if "R_dagger_2" in r]
        dabs = [float(r["judge_disagreement_abs"]) for r in results if "judge_disagreement_abs" in r]
        if j2s:
            m2 = sum(j2s) / len(j2s)
            summary["judge_2"] = (results[0].get("judge_2") or "")
            summary["mean_R_dagger_2"] = m2
            summary["Delta_hack_proxy_judge2"] = mean_r - m2
        if dabs:
            summary["mean_judge_disagreement_abs"] = sum(dabs) / len(dabs)
    if (
        len(results) > 2
        and getattr(cfg.eval_s, "compute_spearman", True)
    ):
        from scipy.stats import spearmanr

        uu2 = [float(x["u"]) for x in results]
        ee2 = [abs(float(x["R_phi"]) - float(x["R_dagger"])) for x in results]
        try:
            rho, _p = spearmanr(uu2, ee2, nan_policy="omit")
        except TypeError:
            rho, _p = spearmanr(uu2, ee2)
        if not (rho != rho):
            summary["spearman_rho_u_abs_err"] = float(rho)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "per_example.jsonl").write_text(
        "\n".join(json.dumps(x) for x in results), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return out
