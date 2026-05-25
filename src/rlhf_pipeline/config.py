"""
Hyperparameters aligned with RESEARCH_PROPOSAL_AND_METHODOLOGY.

Defaults now use the proposal minimums for data and evaluation size rather than the
old pilot caps. Use env overrides for larger runs, e.g. `RLHF_BASE_MODEL`,
`RLHF_PPO_TOTAL_STEPS`, `RLHF_PPO_LAMBDA`, `RLHF_SEED`, `RLHF_ROLLOUT_N`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# §5.3: λ grid for UP-PPO; §5.3 seeds: 3 runs for comparisons
_PROPOSAL_DEFAULT_LAMBDA_SWEEP = [0.0, 0.1, 0.5, 1.0]
_PROPOSAL_SEEDS = [42, 43, 44]


@dataclass
class DataConfig:
    # HuggingFace: public, no Anthropic API key. See README for API vs dataset.
    dataset_name: str = "Anthropic/hh-rlhf"
    # Hub currently exposes a single "default" config (older docs used "helpful-base")
    dataset_config: str = "default"
    n_train: int = 10_000  # proposal §5.2 minimum for headline RLHF runs
    n_val: int = 512
    n_test: int = 512
    seed: int = 42
    # After parsing HH rows, drop duplicate prompts (keeps first). §5.2 disjoint splits.
    dedupe_by_prompt: bool = True


@dataclass
class ModelConfig:
    # Small defaults for T4/Colab; switch to e.g. Mistral-7B for headline runs
    base_model: str = "openai-community/gpt2"  # 124M; use gpt2-medium (355M) if memory allows
    use_lora: bool = False  # set True if peft + compatible target_modules for your base model
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_length: int = 512
    # RM head
    reward_dropout_for_mc: float = 0.1  # >0 so MC spread is non-trivial in train mode


@dataclass
class SFTConfig:
    learning_rate: float = 3e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    fp16: bool = True


@dataclass
class RMConfig:
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    # §4.4: fit T on val pairs s.t. p = σ((R_φ^+ - R_φ^-) / T) minimizes NLL; inference uses s/T.
    calibrate_temperature: bool = True
    # Cap val pairs for temperature grid search (full val used if smaller)
    temperature_val_cap: int = 2_048


@dataclass
class PPOConfigLike:
    learning_rate: float = 3e-6
    batch_size: int = 4
    mini_batch_size: int = 1
    ppo_epochs: int = 2
    gradient_accumulation_steps: int = 1
    kl_penalty_beta: float = 0.05
    # §4.3 UP-PPO: β grid in {0.01,0.05,0.1} for KL term — main knob is kl_penalty_beta here
    up_lambda: float = 0.0  # 0 = vanilla; >0 = UP-PPO (R_φ - λ u)
    num_mc: int = 4  # Monte Carlo dropouts for u
    max_new_tokens: int = 64
    total_steps: int = 500
    # §5.2: at least 10k prompts for headline RL
    n_rl_prompts: int = 10_000
    # PPO-style clipped objective on a fixed rollout (proposal §4, A.5)
    ppo_inner_epochs: int = 2
    clip_epsilon: float = 0.2
    # Trainers log metrics for RQ1 (Δ_hack vs implicit KL) — see ppo/training_log.csv
    write_training_log: bool = True
    training_log_name: str = "training_log.csv"
    # Save policy under ppo/{out_subdir}/policy_after_pilot; default "" = ppo/
    out_subdir: str = ""
    # Save a recoverable checkpoint every N outer steps. 0 disables periodic checkpoints.
    checkpoint_every_steps: int = 100
    # Stability controls for conservative PPO tuning/debugging.
    adaptive_kl: bool = False
    target_kl: float = 0.05
    kl_hard_stop: float = 2.0
    early_stop_patience: int = 5
    min_r_phi: float = -3.0
    use_reward_baseline: bool = True
    reward_baseline_momentum: float = 0.95
    advantage_clip: float = 1.0
    eval_checkpoints: bool = False
    checkpoint_eval_prompts: int = 16
    generation_kwargs: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 64,
            "top_p": 0.95,
            "do_sample": True,
            "temperature": 0.75,
        }
    )


@dataclass
class DPOConfigLike:
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    # DPO: TRL's DPOConfig only takes max_length (total tokenized length). max_prompt_length is reserved for
    # optional custom preprocessing; the pipeline currently only passes max_length into DPOConfig.
    max_length: int = 512
    max_prompt_length: int = 256
    fp16: bool = True
    # One SFT in memory: precompute ref logp once, then train policy only (big MPS/VRAM win)
    precompute_ref_log_probs: bool = True
    precompute_ref_batch_size: int = 1


@dataclass
class EvalSetConfig:
    """RM+judge on **human** test `chosen` (no generation). §5.4 metrics."""
    n_examples: int = 512
    # §5.4: Spearman ρ(u, e) on-policy in rollout; for static eval, optional diagnostic
    compute_spearman: bool = True


@dataclass
class RolloutEvalConfig:
    """Policy-level eval: generate on test `prompt` from each existing checkpoint, then score."""
    # Capped at len(test) from the data stage
    n_examples: int = 512
    # PPO checkpoint under artifacts/ppo/ — e.g. policy_after_pilot or up_lambda_0.5/policy_after_pilot
    ppo_policy_relative: str = "policy_after_pilot"
    # Per-model mean KL(π || π_ref) on generated continuations (§5.4)
    compute_mean_kl: bool = True
    # §3 RQ2: Spearman between u and |R_φ - R†| on rollouts
    compute_spearman: bool = True
    max_new_tokens: int = 64
    truncate_prompt_tokens: int = 512
    # Sampling (ignored when do_sample is False; greedy is more stable, sampling more diverse)
    do_sample: bool = True
    top_p: float = 0.95
    temperature: float = 0.75
    # MPS+fp16+sampled generate often breaks with invalid prob tensors; float32 is safer (also helps some DPO heads).
    use_fp32_on_mps: bool = True
    # Subset of: sft, ppo, dpo (missing checkpoints are skipped)
    model_labels: List[str] = field(
        default_factory=lambda: ["sft", "ppo", "dpo"]
    )


@dataclass
class GlobalConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rm: RMConfig = field(default_factory=RMConfig)
    ppo: PPOConfigLike = field(default_factory=PPOConfigLike)
    dpo: DPOConfigLike = field(default_factory=DPOConfigLike)
    eval_s: EvalSetConfig = field(default_factory=EvalSetConfig)
    rollout: RolloutEvalConfig = field(default_factory=RolloutEvalConfig)
    seed: int = 42
    # §5.3: report 3-seed runs (re-invoke with RLHF_SEED=43,44 or a shell loop)
    proposal_seeds: List[int] = field(default_factory=lambda: list(_PROPOSAL_SEEDS))
    proposal_lambda_sweep: List[float] = field(default_factory=lambda: list(_PROPOSAL_DEFAULT_LAMBDA_SWEEP))
    # §5.3 PPO KL penalty β grid; `ppo_kl_sweep` stage trains each and keeps best on val (mean R_φ on val rollouts)
    proposal_kl_beta_sweep: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    # Which stages to run in run_all
    stages: List[str] = field(
        default_factory=lambda: ["data", "sft", "rm", "ppo", "dpo", "eval", "eval_rollout"]
    )


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    return float(v)


def _env_list_float(name: str) -> List[float] | None:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def _apply_proposal_profile(cfg: GlobalConfig) -> None:
    if os.environ.get("RLHF_PROFILE", "").strip().lower() != "proposal":
        return
    # These match the defaults; keep this profile as an explicit "proposal minimum" switch.
    cfg.data.n_train = 10_000
    cfg.data.n_val = 512
    cfg.data.n_test = 512
    cfg.ppo.n_rl_prompts = 10_000
    cfg.ppo.total_steps = 500
    cfg.model.max_length = 512
    # Open default remains gpt2; set RLHF_BASE_MODEL for Mistral-7B etc.
    if os.environ.get("RLHF_BASE_MODEL", "").strip():
        cfg.model.base_model = os.environ["RLHF_BASE_MODEL"].strip()
    # Example headline swap (uncomment in your env, not here):
    # cfg.model.base_model = "meta-llama/Llama-3.2-3B-Instruct"
    # cfg.model.max_length = 1024


def build_global_config() -> GlobalConfig:
    """
    Build config with optional env / profile (call after setting os.environ in tests).
    """
    cfg = GlobalConfig()
    _apply_proposal_profile(cfg)

    if (s := os.environ.get("RLHF_SEED", "").strip()):
        cfg.seed = int(s)
    if (s := _env_list_float("RLHF_PROPOSAL_SEEDS")) is not None:
        cfg.proposal_seeds = s

    cfg.data.n_train = _env_int("RLHF_N_TRAIN", cfg.data.n_train)
    cfg.data.n_val = _env_int("RLHF_N_VAL", cfg.data.n_val)
    cfg.data.n_test = _env_int("RLHF_N_TEST", cfg.data.n_test)
    if (bm := os.environ.get("RLHF_BASE_MODEL", "").strip()):
        cfg.model.base_model = bm
    if (ml := os.environ.get("RLHF_MAX_LENGTH", "").strip()):
        cfg.model.max_length = int(ml)

    cfg.ppo.kl_penalty_beta = _env_float("RLHF_PPO_KL_BETA", cfg.ppo.kl_penalty_beta)
    cfg.ppo.learning_rate = _env_float("RLHF_PPO_LR", cfg.ppo.learning_rate)
    cfg.ppo.up_lambda = _env_float("RLHF_PPO_LAMBDA", cfg.ppo.up_lambda)
    cfg.ppo.total_steps = _env_int("RLHF_PPO_TOTAL_STEPS", cfg.ppo.total_steps)
    cfg.ppo.n_rl_prompts = _env_int("RLHF_PPO_N_RL_PROMPTS", cfg.ppo.n_rl_prompts)
    cfg.ppo.max_new_tokens = _env_int("RLHF_PPO_MAX_NEW_TOKENS", cfg.ppo.max_new_tokens)
    cfg.ppo.checkpoint_every_steps = _env_int(
        "RLHF_PPO_CHECKPOINT_EVERY_STEPS", cfg.ppo.checkpoint_every_steps
    )
    if (sw := _env_list_float("RLHF_PPO_LAMBDA_SWEEP")) is not None:
        cfg.proposal_lambda_sweep = sw
    if (kb := _env_list_float("RLHF_PPO_KL_BETA_SWEEP")) is not None:
        cfg.proposal_kl_beta_sweep = kb
    if (ie := os.environ.get("RLHF_PPO_INNER_EPOCHS", "").strip()):
        cfg.ppo.ppo_inner_epochs = int(ie)
    if (ce := os.environ.get("RLHF_PPO_CLIP", "").strip()):
        cfg.ppo.clip_epsilon = float(ce)
    if (osub := os.environ.get("RLHF_PPO_OUT_SUBDIR", "").strip()):
        cfg.ppo.out_subdir = osub
    if os.environ.get("RLHF_PPO_ADAPTIVE_KL", "").strip().lower() in ("1", "true", "yes", "on"):
        cfg.ppo.adaptive_kl = True
    cfg.ppo.target_kl = _env_float("RLHF_PPO_TARGET_KL", cfg.ppo.target_kl)
    cfg.ppo.kl_hard_stop = _env_float("RLHF_PPO_KL_HARD_STOP", cfg.ppo.kl_hard_stop)
    cfg.ppo.early_stop_patience = _env_int("RLHF_PPO_EARLY_STOP_PATIENCE", cfg.ppo.early_stop_patience)
    cfg.ppo.min_r_phi = _env_float("RLHF_PPO_MIN_R_PHI", cfg.ppo.min_r_phi)
    if os.environ.get("RLHF_PPO_NO_REWARD_BASELINE", "").strip().lower() in ("1", "true", "yes", "on"):
        cfg.ppo.use_reward_baseline = False
    cfg.ppo.reward_baseline_momentum = _env_float(
        "RLHF_PPO_REWARD_BASELINE_MOMENTUM", cfg.ppo.reward_baseline_momentum
    )
    cfg.ppo.advantage_clip = _env_float("RLHF_PPO_ADVANTAGE_CLIP", cfg.ppo.advantage_clip)
    cfg.ppo.checkpoint_eval_prompts = _env_int(
        "RLHF_PPO_CHECKPOINT_EVAL_PROMPTS", cfg.ppo.checkpoint_eval_prompts
    )
    if os.environ.get("RLHF_PPO_EVAL_CHECKPOINTS", "").strip().lower() in ("1", "true", "yes", "on"):
        cfg.ppo.eval_checkpoints = True
    if os.environ.get("RLHF_PPO_GREEDY", "").strip().lower() in ("1", "true", "yes", "on"):
        cfg.ppo.generation_kwargs["do_sample"] = False
    if (top_p := os.environ.get("RLHF_PPO_TOP_P", "").strip()):
        cfg.ppo.generation_kwargs["top_p"] = float(top_p)
    if (temp := os.environ.get("RLHF_PPO_TEMPERATURE", "").strip()):
        cfg.ppo.generation_kwargs["temperature"] = float(temp)

    if os.environ.get("RLHF_DEDUPE_PROMPTS", "").strip() in ("0", "false", "False", "no"):
        cfg.data.dedupe_by_prompt = False
    if os.environ.get("RLHF_RM_NO_CALIB", "").strip() in ("1", "true", "True", "yes"):
        cfg.rm.calibrate_temperature = False
    if (v := os.environ.get("RLHF_RM_TEMP_VAL_CAP", "").strip()):
        cfg.rm.temperature_val_cap = int(v)

    if (n := os.environ.get("RLHF_ROLLOUT_N", "").strip()):
        cfg.rollout.n_examples = int(n)
    if (n := os.environ.get("RLHF_EVAL_N", "").strip()):
        cfg.eval_s.n_examples = int(n)
    if (rel := os.environ.get("RLHF_PPO_POLICY_RELATIVE", "").strip()):
        cfg.rollout.ppo_policy_relative = rel
    if os.environ.get("RLHF_NO_SPEARMAN", "").strip() in ("1", "true", "True"):
        cfg.rollout.compute_spearman = False
        cfg.eval_s.compute_spearman = False
    if os.environ.get("RLHF_NO_ROLLOUT_KL", "").strip() in ("1", "true", "True"):
        cfg.rollout.compute_mean_kl = False

    return cfg
