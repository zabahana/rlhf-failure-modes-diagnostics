"""
Policy-level eval: generate from SFT / PPO / DPO on test prompts, then RM + judges.

Writes tagged artifacts such as eval/rollout_summary_up_lambda_0p5.json when evaluating
different PPO checkpoints. Also refreshes the legacy `rollout_summary.json` latest-run files
for plotting/backward compatibility.
"""
from __future__ import annotations

import gc
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import GlobalConfig, RolloutEvalConfig
from .judge_resolution import (
    resolve_primary_judge,
    resolve_second_judge,
    sleep_after_judge_call,
)
from .load_env import load_code_dotenv
from .models_rm import CausalScalarRewardModel
from .paths import data_dir, dpo_dir, eval_dir, ppo_dir, ppo_policy_path, rm_dir, sft_dir, ensure_dirs
from .rm_temperature import apply_temperature_scalar, load_rm_temperature


def _seq_logp_sum(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start: int,
) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logp = F.log_softmax(out.logits[:, :-1], dim=-1)
    target = input_ids[:, 1:]
    gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    gmask = mask.clone()
    gmask[:, :start] = 0.0
    return (gathered * gmask).sum(dim=1)


def _model_paths(cfg: GlobalConfig) -> List[Tuple[str, Path]]:
    want = {k.strip().lower() for k in cfg.rollout.model_labels}
    out: List[Tuple[str, Path]] = []
    ppo_rel = (cfg.rollout.ppo_policy_relative or "policy_after_pilot").strip() or "policy_after_pilot"
    m = {
        "sft": sft_dir() / "model",
        "ppo": ppo_policy_path(ppo_rel),
        "dpo": dpo_dir() / "model",
    }
    for label in ("sft", "ppo", "dpo"):
        if label not in want:
            continue
        p = m[label]
        if (p / "config.json").is_file():
            out.append((label, p))
        else:
            print(f"[eval_rollout] skip {label}: no checkpoint at {p}", flush=True)
    return out


def _safe_tag(raw: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    return tag.strip("._-") or "default"


def _rollout_tag(cfg: GlobalConfig) -> str:
    """
    Stable output tag so repeated evals over different PPO checkpoints do not overwrite.
    Override with RLHF_ROLLOUT_TAG. Otherwise derive from RLHF_PPO_POLICY_RELATIVE.
    """
    if (manual := os.environ.get("RLHF_ROLLOUT_TAG", "").strip()):
        return _safe_tag(manual)
    rel = (cfg.rollout.ppo_policy_relative or "policy_after_pilot").strip() or "policy_after_pilot"
    parts = Path(rel).parts
    if parts and parts[-1] == "policy_after_pilot" and len(parts) > 1:
        return _safe_tag(parts[-2])
    return _safe_tag(rel)


def _tagged_path(out: Path, stem: str, suffix: str, tag: str) -> Path:
    return out / f"{stem}_{tag}{suffix}"


def _copy_latest(src: Path, latest: Path) -> None:
    """Refresh legacy latest-run artifacts without losing the tagged artifact."""
    if src == latest:
        return
    shutil.copyfile(src, latest)


def _gen_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gen_dtype(dev: torch.device, rc: RolloutEvalConfig) -> torch.dtype:
    if dev.type == "cuda":
        return torch.float16
    if dev.type == "mps" and rc.use_fp32_on_mps:
        return torch.float32
    if dev.type == "mps":
        return torch.float16
    return torch.float32


def _run_generate(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    enc: dict,
    p_len: int,
    device: torch.device,
    rc: RolloutEvalConfig,
    *,
    do_sample: bool,
) -> tuple[str, torch.Tensor]:
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=rc.max_new_tokens,
            do_sample=do_sample,
            top_p=rc.top_p if do_sample else 1.0,
            temperature=rc.temperature if do_sample else 1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_ids = out[0, p_len:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip() or ""
    return text, out


def _mean_kl_on_generated_ids(
    pol: torch.nn.Module,
    ref: torch.nn.Module,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    p_len: int,
) -> float:
    """Mean (log π_pol − log π_ref) per **generated** token, using the **same** ids as `generate`."""
    p_len = int(p_len)
    lpol = _seq_logp_sum(pol, full_ids, attention_mask, p_len)
    lref = _seq_logp_sum(ref, full_ids, attention_mask, p_len)
    ngen = (attention_mask.float().sum(dim=1) - float(p_len)).clamp(min=1.0)
    return float(((lpol - lref) / ngen).item())


def _generate_one(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    prompt: str,
    device: torch.device,
    rc: RolloutEvalConfig,
    gen_seed: int,
) -> tuple[str, torch.Tensor, int]:
    """Returns (generated_text, full_token_ids[1, T], p_len) for exact KL(π || π_ref)."""
    model.eval()
    torch.manual_seed(int(gen_seed) % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(gen_seed) % (2**32))
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=rc.truncate_prompt_tokens)
    enc = {k: v.to(device) for k, v in enc.items()}
    p_len = int(enc["input_ids"].shape[1])
    if rc.do_sample:
        try:
            gen, out = _run_generate(model, tok, enc, p_len, device, rc, do_sample=True)
        except (RuntimeError, ValueError) as e:
            err = str(e).lower()
            if "probability" in err or "inf" in err or "nan" in err:
                gen, out = _run_generate(model, tok, enc, p_len, device, rc, do_sample=False)
            else:
                raise
    else:
        gen, out = _run_generate(model, tok, enc, p_len, device, rc, do_sample=False)
    return gen, out, p_len


def _md_table(rows: List[Dict[str, Any]], title: str) -> str:
    if not rows:
        return f"## {title}\n\n_(no rows)_\n\n"
    keys = list(rows[0].keys())
    line = "| " + " | ".join(str(k) for k in keys) + " |"
    sep = "|" + "|".join("---" for _ in keys) + "|"
    body = "\n".join(
        "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows
    )
    return f"## {title}\n\n{line}\n{sep}\n{body}\n\n"


def run_eval_rollout(cfg: GlobalConfig) -> Path:
    load_code_dotenv()
    judge_fn, judge_name = resolve_primary_judge()
    j2_fn, j2_name = resolve_second_judge(judge_name)
    print(
        f"eval_rollout: judge={judge_name}, judge_2={j2_name or 'none'} "
        "(generates, then RM + judge on full text; second judge = extra API per example)",
        flush=True,
    )
    ensure_dirs()
    out = eval_dir()
    out.mkdir(parents=True, exist_ok=True)
    rc = cfg.rollout
    tag = _rollout_tag(cfg)
    dsd = load_from_disk(str(data_dir() / "hf_dataset"))["test"]
    rows: List[dict] = [dict(x) for x in dsd][: int(rc.n_examples)]
    if not rows:
        raise ValueError("No test rows; run the data stage first.")
    pols = _model_paths(cfg)
    if not pols:
        raise FileNotFoundError("No policy checkpoints (sft/model, ppo/policy_after_pilot, dpo/model).")

    dev_rm = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpath = rm_dir() / "reward_model.pt"
    meta = json.loads((rm_dir() / "meta.json").read_text(encoding="utf-8"))
    rm = CausalScalarRewardModel(meta["load_path"], dropout=cfg.model.reward_dropout_for_mc)
    rm.load_state_dict(torch.load(mpath, map_location="cpu")["state_dict"])
    rm = rm.to(dev_rm)
    if dev_rm.type == "cuda":
        rm = rm.half()
    rm_tok = AutoTokenizer.from_pretrained(rm_dir() / "tok", use_fast=True)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm.eval()
    t_rm = load_rm_temperature()

    gdev = _gen_device()
    gdtype = _gen_dtype(gdev, rc)
    all_rows: List[dict] = []
    by_model: Dict[str, Dict[str, float]] = {}

    sft_ref: torch.nn.Module | None = None
    if rc.compute_mean_kl and (sft_dir() / "model" / "config.json").is_file():
        sft_p = sft_dir() / "model"
        print("eval_rollout: loading SFT as π_ref for KL(π || π_0)", flush=True)
        sft_ref = AutoModelForCausalLM.from_pretrained(sft_p, torch_dtype=gdtype).to(gdev)
        sft_ref.eval()

    for model_label, pdir in pols:
        pol_tok = AutoTokenizer.from_pretrained(pdir, use_fast=True)
        if pol_tok.pad_token is None:
            pol_tok.pad_token = pol_tok.eos_token
        print(f"eval_rollout: loading {model_label} from {pdir}", flush=True)
        mdl = AutoModelForCausalLM.from_pretrained(pdir, torch_dtype=gdtype)
        mdl = mdl.to(gdev)
        mdl.eval()
        r_acc: List[dict] = []
        for i, r in enumerate(rows):
            prompt = r["prompt"]
            seed = cfg.seed + i * 10_000_001 + (sum(ord(c) for c in model_label) * 97) % 2_000_000_000
            out_ids: torch.Tensor | None = None
            p_len_gen = 0
            try:
                gen, out_ids, p_len_gen = _generate_one(mdl, pol_tok, prompt, gdev, rc, gen_seed=seed)
            except Exception as e:  # noqa: BLE001
                print(f"[{model_label}] i={i} gen failed: {e}", flush=True)
                gen = ""
                out_ids = None
                p_len_gen = 0
            full = prompt + gen
            with torch.no_grad():
                s = float(
                    rm.score_texts(rm_tok, [full], cfg.model.max_length, dev_rm)[0]
                )
            with torch.no_grad():
                _, uu = rm.score_texts_with_dropout_std(
                    rm_tok, [full], cfg.model.max_length, dev_rm, k=cfg.ppo.num_mc
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
            mkl: float | None = None
            if sft_ref is not None and rc.compute_mean_kl:
                if model_label == "sft":
                    mkl = 0.0
                elif out_ids is not None:
                    attn_kl = (out_ids != pol_tok.pad_token_id).long()
                    mkl = _mean_kl_on_generated_ids(mdl, sft_ref, out_ids, attn_kl, p_len_gen)
            rec: Dict[str, Any] = {
                "i": i,
                "model": model_label,
                "prompt": prompt,
                "generated": gen,
                "R_phi": s,
                "u": uu,
                "R_dagger": jd,
                "judge": judge_name,
            }
            if jd2 is not None:
                rec["R_dagger_2"] = jd2
                rec["judge_2"] = j2_name
                rec["judge_disagreement_abs"] = abs(float(jd) - float(jd2))
            if mkl is not None:
                rec["mean_kl_sft"] = mkl
            r_acc.append(rec)
            all_rows.append(rec)
        n = max(1, len(r_acc))
        agg: Dict[str, Any] = {
            "mean_R_phi": sum(x["R_phi"] for x in r_acc) / n,
            "mean_R_dagger": sum(x["R_dagger"] for x in r_acc) / n,
            "mean_u": sum(x["u"] for x in r_acc) / n,
            "Delta_hack_proxy": (sum(x["R_phi"] for x in r_acc) - sum(x["R_dagger"] for x in r_acc)) / n,
            "n": len(r_acc),
        }
        if rc.compute_spearman and len(r_acc) > 2:
            uu2 = [float(x["u"]) for x in r_acc]
            ee2 = [abs(float(x["R_phi"]) - float(x["R_dagger"])) for x in r_acc]
            try:
                rho, _p = spearmanr(uu2, ee2, nan_policy="omit")
            except TypeError:
                rho, _p = spearmanr(uu2, ee2)
            if not (rho != rho):  # not NaN
                agg["spearman_rho_u_abs_err"] = float(rho)
        if j2_name and any("R_dagger_2" in x for x in r_acc):
            r2s = [float(x["R_dagger_2"]) for x in r_acc if "R_dagger_2" in x]
            dabs2 = [float(x["judge_disagreement_abs"]) for x in r_acc if "judge_disagreement_abs" in x]
            if r2s:
                agg["mean_R_dagger_2"] = sum(r2s) / len(r2s)
                agg["Delta_hack_proxy_judge2"] = (sum(x["R_phi"] for x in r_acc) - sum(r2s)) / n
            if dabs2:
                agg["mean_judge_disagreement_abs"] = sum(dabs2) / len(dabs2)
        if sft_ref is not None and rc.compute_mean_kl and any("mean_kl_sft" in x for x in r_acc):
            mks = [float(x["mean_kl_sft"]) for x in r_acc if "mean_kl_sft" in x]
            if mks:
                agg["mean_kl_sft"] = sum(mks) / len(mks)
        by_model[model_label] = agg
        del mdl
        if gdev.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif gdev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    paired: List[dict] = []
    by_i: Dict[int, Dict[str, dict]] = {}
    for r in all_rows:
        by_i.setdefault(int(r["i"]), {})[str(r["model"])] = r
    pmean: float | None = None
    for i, mrec in by_i.items():
        row: Dict[str, Any] = {"i": i}
        for k, v in mrec.items():
            row[f"R_dagger_{k}"] = v["R_dagger"]
            row[f"R_phi_{k}"] = v["R_phi"]
        if "dpo" in mrec and "sft" in mrec:
            row["dpo_minus_sft_R_dagger"] = float(mrec["dpo"]["R_dagger"]) - float(
                mrec["sft"]["R_dagger"]
            )
        paired.append(row)
    dpo_sft: List[float] = [
        float(p["dpo_minus_sft_R_dagger"])
        for p in paired
        if "dpo_minus_sft_R_dagger" in p
    ]
    if dpo_sft:
        pmean = sum(dpo_sft) / len(dpo_sft)

    summary: Dict[str, Any] = {
        "judge": judge_name,
        "rollout_tag": tag,
        "ppo_policy_relative": rc.ppo_policy_relative,
        "n_examples": len(rows),
        "models": list(by_model.keys()),
        "by_model": by_model,
        "mean_paired_dpo_minus_sft_R_dagger": pmean,
        "n_paired_dpo_sft": len(dpo_sft),
        "note": "Policy rollouts: scores are on model-generated text (not human chosen).",
    }

    if sft_ref is not None:
        del sft_ref
        if gdev.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif gdev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    summary_path = _tagged_path(out, "rollout_summary", ".json", tag)
    examples_path = _tagged_path(out, "rollout_examples", ".jsonl", tag)
    paired_path = _tagged_path(out, "rollout_paired_deltas", ".json", tag)
    comparison_path = _tagged_path(out, "rollout_comparison", ".md", tag)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    examples_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in all_rows), encoding="utf-8"
    )
    paired_path.write_text(
        json.dumps(
            {
                "judge": judge_name,
                "rollout_tag": tag,
                "ppo_policy_relative": rc.ppo_policy_relative,
                "n_pairs": len(paired),
                "mean_paired_dpo_minus_sft_R_dagger": pmean,
                "rows": paired,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    table_rows = [
        {
            "Model": m,
            "mean_R_phi": f"{v['mean_R_phi']:.4f}",
            "mean_R_dagger": f"{v['mean_R_dagger']:.4f}",
            "mean_u": f"{v['mean_u']:.4f}",
            "n": v["n"],
        }
        for m, v in by_model.items()
    ]
    md = _md_table(table_rows, "Policy rollout (generated text)")
    if pmean is not None:
        md += f"**Mean paired (same prompts) DPO − SFT on judge R_dagger** (`{judge_name}`): **{pmean:.4f}** (n={len(dpo_sft)})\n\n"
    md += (
        f"Artifacts: `{summary_path.name}`, `{examples_path.name}`, `{paired_path.name}`.\n"
        "Legacy latest-run copies are also refreshed as `rollout_summary.json`, "
        "`rollout_examples.jsonl`, and `rollout_paired_deltas.json`.\n"
    )
    comparison_path.write_text(md, encoding="utf-8")

    _copy_latest(summary_path, out / "rollout_summary.json")
    _copy_latest(examples_path, out / "rollout_examples.jsonl")
    _copy_latest(paired_path, out / "rollout_paired_deltas.json")
    _copy_latest(comparison_path, out / "rollout_comparison.md")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {summary_path} and {comparison_path}", flush=True)
    print(f"Refreshed latest-run copies under {out}", flush=True)
    return out
