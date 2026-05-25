"""
RL stage: PPO-style **clipped** surrogate on fixed rollouts + KL(π || π_ref), with optional
**UP-PPO** shaped reward (R_φ - λ u), matching RESEARCH_PROPOSAL_AND_METHODOLOGY §4.3–A.5.

This is a lightweight trainer (not TRL's full PPO+GAE) but includes ratio clipping and
logs `training_log.csv` for Δ_hack / KL / R_φ / u over steps (§5.4, RQ1).
For frontier-scale LMs, plug TRL PPO or scale `total_steps` / `n_rl_prompts` via env.
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import List, TextIO

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GlobalConfig
from .models_rm import CausalScalarRewardModel
from .paths import data_dir, ppo_dir, rm_dir, sft_dir, ensure_dirs
from .rm_temperature import load_rm_temperature


def _seq_logp(
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


def _ppo_run_root(cfg: GlobalConfig) -> Path:
    r = ppo_dir()
    if cfg.ppo.out_subdir:
        r = r / cfg.ppo.out_subdir
    r.mkdir(parents=True, exist_ok=True)
    return r


def _generate_for_ppo(
    pol: torch.nn.Module,
    enc: dict,
    gkw: dict,
    *,
    step: int,
) -> torch.Tensor:
    """Sample a rollout; if sampling hits invalid probabilities, retry greedily."""
    pol.eval()
    with torch.no_grad():
        try:
            return pol.generate(**enc, **gkw)
        except (RuntimeError, ValueError) as e:
            msg = str(e).lower()
            if "probability" not in msg and "inf" not in msg and "nan" not in msg:
                raise
            print(
                f"[ppo] step={step}: sampled generate failed with invalid probabilities; retrying greedy.",
                flush=True,
            )
            greedy = dict(gkw)
            greedy["do_sample"] = False
            greedy["top_p"] = 1.0
            greedy["temperature"] = 1.0
            return pol.generate(**enc, **greedy)


def run_ppo(cfg: GlobalConfig) -> Path:
    ensure_dirs()
    out = _ppo_run_root(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_p = sft_dir() / "model"
    rmp = rm_dir() / "reward_model.pt"
    if not sft_p.exists() or not rmp.exists():
        raise FileNotFoundError("Run sft and rm stages first.")
    dsd = load_from_disk(str(data_dir() / "hf_dataset"))["train"]
    cap = min(int(cfg.ppo.n_rl_prompts), len(dsd))
    prompts: List[str] = [x["prompt"] for x in dsd][:cap]

    tok = AutoTokenizer.from_pretrained(sft_p, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pol = AutoModelForCausalLM.from_pretrained(
        sft_p,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    ref = AutoModelForCausalLM.from_pretrained(
        sft_p,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    # Keep dropout disabled for policy log-prob calculations; gradients still flow in eval mode.
    # Otherwise π and π_ref differ at step 0 purely because the policy has active dropout.
    pol.eval()
    with open(rm_dir() / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    rm = CausalScalarRewardModel(
        meta["load_path"],
        dropout=cfg.model.reward_dropout_for_mc,
    ).to(device)
    ck = torch.load(rm_dir() / "reward_model.pt", map_location=device)
    rm.load_state_dict(ck["state_dict"])
    rm_tok = AutoTokenizer.from_pretrained(rm_dir() / "tok", use_fast=True)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm.eval()
    t_rm = float(load_rm_temperature())

    opt = torch.optim.AdamW(pol.parameters(), lr=cfg.ppo.learning_rate)
    gkw = dict(getattr(cfg.ppo, "generation_kwargs", {}) or {})
    gkw["max_new_tokens"] = int(cfg.ppo.max_new_tokens)
    gkw.setdefault("do_sample", True)
    gkw.setdefault("top_p", 0.95)
    gkw.setdefault("temperature", 0.75)
    gkw["pad_token_id"] = tok.pad_token_id
    if not gkw.get("do_sample", True):
        gkw.pop("top_p", None)
        gkw.pop("temperature", None)
    beta = float(cfg.ppo.kl_penalty_beta)
    eps = float(cfg.ppo.clip_epsilon)
    lam = float(cfg.ppo.up_lambda)
    n_mc = int(cfg.ppo.num_mc)
    inner = max(1, int(cfg.ppo.ppo_inner_epochs))
    ckpt_every = max(0, int(getattr(cfg.ppo, "checkpoint_every_steps", 0)))
    adaptive_kl = bool(getattr(cfg.ppo, "adaptive_kl", False))
    target_kl = float(getattr(cfg.ppo, "target_kl", 0.05))
    kl_hard_stop = float(getattr(cfg.ppo, "kl_hard_stop", 2.0))
    patience = max(1, int(getattr(cfg.ppo, "early_stop_patience", 5)))
    min_r_phi = float(getattr(cfg.ppo, "min_r_phi", -3.0))
    eval_checkpoints = bool(getattr(cfg.ppo, "eval_checkpoints", False))
    ckpt_eval_prompts = max(1, int(getattr(cfg.ppo, "checkpoint_eval_prompts", 16)))
    use_reward_baseline = bool(getattr(cfg.ppo, "use_reward_baseline", True))
    baseline_momentum = float(getattr(cfg.ppo, "reward_baseline_momentum", 0.95))
    advantage_clip = float(getattr(cfg.ppo, "advantage_clip", 1.0))
    reward_baseline: float | None = None
    bad_steps = 0
    stop_reason = ""
    checkpoint_scores: list[dict] = []

    log_path = out / cfg.ppo.training_log_name
    log_fieldnames = [
        "step",
        "R_phi",
        "u",
        "kl_seq",
        "kl_abs",
        "loss",
        "kl_beta",
        "up_lambda",
        "ppo_inner",
        "advantage",
        "reward_baseline",
    ]
    fcsv: TextIO | None = None
    w: csv.DictWriter | None = None
    if cfg.ppo.write_training_log:
        fcsv = open(log_path, "w", encoding="utf-8", newline="")
        w = csv.DictWriter(fcsv, fieldnames=log_fieldnames)
        w.writeheader()

    try:
        for step in tqdm(range(cfg.ppo.total_steps), desc="up-ppo" if lam > 0 else "rlhf-rl"):
            if not prompts:
                break
            ptxt = [prompts[step % len(prompts)]]
            enc = tok(
                ptxt, return_tensors="pt", padding=True, truncation=True, max_length=cfg.model.max_length
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = _generate_for_ppo(pol, enc, gkw, step=step)
            full = gen
            attn = (full != tok.pad_token_id).long()
            p_len = enc["input_ids"].shape[1]
            lref = _seq_logp(ref, full, attn, p_len)
            texts = [ptxt[0] + tok.decode(g[p_len:], skip_special_tokens=True) for g in full]
            with torch.no_grad():
                s = rm.score_texts(rm_tok, texts, cfg.model.max_length, device) / t_rm
                _, uu = rm.score_texts_with_dropout_std(
                    rm_tok, texts, cfg.model.max_length, device, k=n_mc
                )
                uu = uu / t_rm
            s0 = float(s[0].item())
            u0 = float(uu[0].item())
            R = s - float(lam) * uu
            R = R.detach()
            if not torch.isfinite(R).all() or not torch.isfinite(lref).all():
                print(f"[ppo] step={step}: non-finite reward/reference logp; skipping update.", flush=True)
                continue
            r_mean = float(R.mean().item())
            if use_reward_baseline:
                if reward_baseline is None:
                    reward_baseline = r_mean
                adv = (R - reward_baseline).clamp(-advantage_clip, advantage_clip)
                reward_baseline = baseline_momentum * reward_baseline + (1.0 - baseline_momentum) * r_mean
            else:
                adv = R.clamp(-advantage_clip, advantage_clip)
                reward_baseline = 0.0
            adv0 = float(adv[0].item())

            old_logp = _seq_logp(pol, full, attn, p_len).detach()
            if not torch.isfinite(old_logp).all():
                print(f"[ppo] step={step}: non-finite old policy logp; skipping update.", flush=True)
                continue
            for inner_i in range(inner):
                # Gradients are enabled in eval mode, but dropout stays disabled for stable KL estimates.
                pol.eval()
                lpol = _seq_logp(pol, full, attn, p_len)
                ratio = (lpol - old_logp).exp().clamp(0, 1e4)
                uncl = ratio * adv
                clp = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
                ppo_obj = -torch.min(uncl, clp).mean()
                kl = lpol - lref
                # Sequence log-prob differences can be negative; the KL control term must
                # still penalize drift rather than reward movement in the negative direction.
                kl_monitor = kl.abs().mean()
                loss = ppo_obj + beta * kl_monitor
                if not torch.isfinite(loss):
                    print(
                        f"[ppo] step={step} inner={inner_i}: non-finite loss; skipping update.",
                        flush=True,
                    )
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    print(
                        f"[ppo] step={step} inner={inner_i}: non-finite grad norm; skipping optimizer step.",
                        flush=True,
                    )
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                l_mean = float(kl.mean().item())
                kl_abs = float(kl_monitor.item())
                if w is not None and fcsv is not None and inner_i == inner - 1:
                    w.writerow(
                        {
                            "step": step,
                            "R_phi": s0,
                            "u": u0,
                            "kl_seq": l_mean,
                            "kl_abs": kl_abs,
                            "loss": float(loss.item()),
                            "kl_beta": beta,
                            "up_lambda": lam,
                            "ppo_inner": inner,
                            "advantage": adv0,
                            "reward_baseline": reward_baseline,
                        }
                    )
                    fcsv.flush()
                if inner_i == inner - 1:
                    if adaptive_kl and target_kl > 0:
                        if kl_abs > 1.5 * target_kl:
                            beta *= 1.5
                        elif kl_abs < 0.5 * target_kl:
                            beta = max(beta / 1.25, 1e-6)
                    if kl_abs > kl_hard_stop or s0 < min_r_phi:
                        bad_steps += 1
                    else:
                        bad_steps = 0
                    if bad_steps >= patience:
                        stop_reason = (
                            f"early stop: {bad_steps} consecutive unstable steps "
                            f"(last kl_abs={kl_abs:.4f}, R_phi={s0:.4f})"
                        )
                        print(f"[ppo] {stop_reason}", flush=True)
                        break
            if stop_reason:
                break
            if ckpt_every and (step + 1) % ckpt_every == 0:
                ckpt_path = out / f"policy_step_{step + 1}"
                pol.save_pretrained(str(ckpt_path))
                tok.save_pretrained(str(ckpt_path))
                print(f"[ppo] checkpoint saved to {ckpt_path}", flush=True)
                if eval_checkpoints:
                    from .ppo_val_score import mean_r_phi_on_val_rollouts

                    vscore = mean_r_phi_on_val_rollouts(
                        cfg,
                        ckpt_path,
                        max_prompts=ckpt_eval_prompts,
                    )
                    checkpoint_scores.append(
                        {
                            "step": step + 1,
                            "policy_dir": str(ckpt_path),
                            "val_mean_R_phi": vscore,
                            "kl_beta": beta,
                        }
                    )
                    (out / "checkpoint_val_scores.json").write_text(
                        json.dumps(checkpoint_scores, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[ppo] checkpoint val mean R_phi={vscore:.4f} ({ckpt_eval_prompts} prompts)",
                        flush=True,
                    )
    finally:
        if fcsv is not None:
            fcsv.close()

    save_path = out / "policy_after_pilot"
    pol.save_pretrained(str(save_path))
    tok.save_pretrained(str(save_path))
    (out / "note.json").write_text(
        json.dumps(
            {
                "method": "PPO-style clip + KL to π₀; optional UP-PPO (R - λu).",
                "up_lambda": cfg.ppo.up_lambda,
                "kl_beta": cfg.ppo.kl_penalty_beta,
                "clip_epsilon": cfg.ppo.clip_epsilon,
                "ppo_inner_epochs": cfg.ppo.ppo_inner_epochs,
                "n_rl_prompts": cap,
                "total_steps": cfg.ppo.total_steps,
                "checkpoint_every_steps": ckpt_every,
                "learning_rate": cfg.ppo.learning_rate,
                "adaptive_kl": adaptive_kl,
                "target_kl": target_kl,
                "kl_hard_stop": kl_hard_stop,
                "early_stop_patience": patience,
                "min_r_phi": min_r_phi,
                "use_reward_baseline": use_reward_baseline,
                "reward_baseline_momentum": baseline_momentum,
                "advantage_clip": advantage_clip,
                "final_kl_beta": beta,
                "stop_reason": stop_reason,
                "generation_kwargs": gkw,
                "eval_checkpoints": eval_checkpoints,
                "checkpoint_eval_prompts": ckpt_eval_prompts,
                "out_subdir": cfg.ppo.out_subdir,
                "training_log": str(cfg.ppo.training_log_name) if cfg.ppo.write_training_log else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved policy to {save_path}", flush=True)
    if cfg.ppo.write_training_log:
        print(f"Wrote {log_path}", flush=True)
    return save_path


def run_ppo_sweep(cfg: GlobalConfig) -> List[Path]:
    """Run one policy per λ in `cfg.proposal_lambda_sweep` (§5.3)."""
    saved: List[Path] = []
    for lam in cfg.proposal_lambda_sweep:
        ppo = replace(
            cfg.ppo,
            up_lambda=lam,
            out_subdir=f"up_lambda_{str(lam).replace('.', 'p')}",
        )
        c = replace(cfg, ppo=ppo)
        print(f"=== ppo sweep λ={lam} -> {ppo.out_subdir} ===", flush=True)
        saved.append(run_ppo(c))
    return saved


def run_ppo_stability_exercise(cfg: GlobalConfig) -> Path:
    """
    Conservative PPO debug run: tiny prompt set, low LR, stronger/adaptive KL,
    short generations, frequent checkpoints, and val R_phi scoring.
    This intentionally writes to ppo/stability_tiny so it does not overwrite headline runs.
    """
    ppo = replace(
        cfg.ppo,
        learning_rate=1e-7,
        kl_penalty_beta=0.5,
        adaptive_kl=True,
        target_kl=0.10,
        kl_hard_stop=0.75,
        early_stop_patience=3,
        min_r_phi=-2.5,
        use_reward_baseline=True,
        reward_baseline_momentum=0.9,
        advantage_clip=0.5,
        max_new_tokens=24,
        total_steps=100,
        n_rl_prompts=128,
        ppo_inner_epochs=1,
        clip_epsilon=0.1,
        checkpoint_every_steps=10,
        eval_checkpoints=True,
        checkpoint_eval_prompts=16,
        out_subdir="stability_tiny",
        training_log_name="training_log_stability.csv",
        generation_kwargs={
            "max_new_tokens": 24,
            "do_sample": False,
        },
    )
    tuned = replace(cfg, ppo=ppo)
    print("=== PPO stability exercise -> ppo/stability_tiny ===", flush=True)
    print(
        json.dumps(
            {
                "learning_rate": ppo.learning_rate,
                "kl_penalty_beta": ppo.kl_penalty_beta,
                "adaptive_kl": ppo.adaptive_kl,
                "target_kl": ppo.target_kl,
                "kl_hard_stop": ppo.kl_hard_stop,
                "use_reward_baseline": ppo.use_reward_baseline,
                "advantage_clip": ppo.advantage_clip,
                "total_steps": ppo.total_steps,
                "n_rl_prompts": ppo.n_rl_prompts,
                "max_new_tokens": ppo.max_new_tokens,
                "checkpoint_every_steps": ppo.checkpoint_every_steps,
                "generation_kwargs": ppo.generation_kwargs,
            },
            indent=2,
        ),
        flush=True,
    )
    return run_ppo(tuned)


def run_ppo_reward_hacking_search(cfg: GlobalConfig) -> Path:
    """
    Corrected-PPO overoptimization search.

    Keeps the stability fixes from `ppo_stability` (dropout-free log-probs,
    nonnegative KL penalty, reward baseline), but increases optimization pressure
    so checkpoints can be evaluated for the classic reward-hacking pattern:
    R_phi rises while external judge scores fall.
    """
    ppo = replace(
        cfg.ppo,
        learning_rate=5e-7,
        kl_penalty_beta=0.03,
        adaptive_kl=False,
        target_kl=0.10,
        kl_hard_stop=1.25,
        early_stop_patience=5,
        min_r_phi=-3.0,
        use_reward_baseline=True,
        reward_baseline_momentum=0.9,
        advantage_clip=0.75,
        max_new_tokens=32,
        total_steps=400,
        n_rl_prompts=512,
        ppo_inner_epochs=1,
        clip_epsilon=0.12,
        checkpoint_every_steps=50,
        eval_checkpoints=True,
        checkpoint_eval_prompts=32,
        out_subdir="reward_hack_search_beta0p03",
        training_log_name="training_log_reward_hack_search.csv",
        generation_kwargs={
            "max_new_tokens": 32,
            "do_sample": False,
        },
    )
    tuned = replace(cfg, ppo=ppo)
    print("=== Corrected PPO reward-hacking search -> ppo/reward_hack_search_beta0p03 ===", flush=True)
    print(
        json.dumps(
            {
                "learning_rate": ppo.learning_rate,
                "kl_penalty_beta": ppo.kl_penalty_beta,
                "adaptive_kl": ppo.adaptive_kl,
                "kl_hard_stop": ppo.kl_hard_stop,
                "use_reward_baseline": ppo.use_reward_baseline,
                "advantage_clip": ppo.advantage_clip,
                "total_steps": ppo.total_steps,
                "n_rl_prompts": ppo.n_rl_prompts,
                "max_new_tokens": ppo.max_new_tokens,
                "checkpoint_every_steps": ppo.checkpoint_every_steps,
                "checkpoint_eval_prompts": ppo.checkpoint_eval_prompts,
                "generation_kwargs": ppo.generation_kwargs,
            },
            indent=2,
        ),
        flush=True,
    )
    return run_ppo(tuned)


def run_ppo_kl_beta_sweep(cfg: GlobalConfig) -> Path:
    """
    §5.3: train PPO once per `cfg.proposal_kl_beta_sweep` (default 0.01, 0.05, 0.1),
    score each on val prompts (mean R_φ on one rollout per prompt), keep the best.
    Writes `ppo/kl_beta_sweep.json` and copies the winning `policy_after_pilot` to `ppo/policy_kl_val_best/`.
    """
    from .ppo_val_score import mean_r_phi_on_val_rollouts

    betas = list(cfg.proposal_kl_beta_sweep)
    if not betas:
        raise ValueError("cfg.proposal_kl_beta_sweep is empty; set RLHF_PPO_KL_BETA_SWEEP")
    results: list[dict] = []
    best_beta: float | None = None
    best_score = float("-inf")
    best_policy: Path | None = None
    for beta in betas:
        sub = f"kl_beta_{str(beta).replace('.', 'p')}"
        ppo = replace(
            cfg.ppo,
            kl_penalty_beta=float(beta),
            out_subdir=sub,
        )
        c = replace(cfg, ppo=ppo)
        print(f"=== PPO KL-β sweep β={beta} -> {sub} ===", flush=True)
        pol_path = run_ppo(c)
        vscore = mean_r_phi_on_val_rollouts(c, pol_path)
        results.append(
            {
                "kl_penalty_beta": beta,
                "out_subdir": sub,
                "val_mean_R_phi": vscore,
                "policy_dir": str(pol_path),
            }
        )
        if vscore == vscore and vscore > best_score:
            best_score = vscore
            best_beta = float(beta)
            best_policy = pol_path
    out_json = ppo_dir() / "kl_beta_sweep.json"
    summary: dict = {
        "candidates": results,
        "best_kl_penalty_beta": best_beta,
        "best_val_mean_R_phi": best_score if best_score > float("-inf") else None,
        "best_policy_dir": str(best_policy) if best_policy else None,
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if best_policy is not None and best_policy.is_dir():
        dest = ppo_dir() / "policy_kl_val_best"
        if dest.is_dir():
            shutil.rmtree(dest)
        shutil.copytree(best_policy, dest)
        (ppo_dir() / "policy_kl_val_best_note.json").write_text(
            json.dumps(
                {"source": str(best_policy), "kl_penalty_beta": best_beta},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Copied best val policy to {dest}", flush=True)
    return out_json
