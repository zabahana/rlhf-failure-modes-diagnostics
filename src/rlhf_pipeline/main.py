"""
CLI: run stages in order, each step reads/writes under artifacts/ (see paths.py).

  python -m rlhf_pipeline.main data
  python -m rlhf_pipeline.main sft
  python -m rlhf_pipeline.main rm
  python -m rlhf_pipeline.main ppo
  python -m rlhf_pipeline.main ppo_stability
  python -m rlhf_pipeline.main ppo_reward_hack_search
  python -m rlhf_pipeline.main ppo_sweep
  python -m rlhf_pipeline.main ppo_kl_sweep
  python -m rlhf_pipeline.main dpo
  python -m rlhf_pipeline.main eval
  python -m rlhf_pipeline.main eval_rollout
  python -m rlhf_pipeline.main all
  python -m rlhf_pipeline.main plot_proposal
  python -m rlhf_pipeline.main paper_artifacts
  python -m rlhf_pipeline.main replicates
  python -m rlhf_pipeline.main aggregate_seeds
"""
from __future__ import annotations

import argparse
import sys

from .load_env import load_code_dotenv


def main(argv: list[str] | None = None) -> int:
    load_code_dotenv()
    from .paths import ensure_dirs

    ensure_dirs()
    p = argparse.ArgumentParser(description="RLHF / UP-PPO pipeline stages")
    p.add_argument(
        "stage",
        choices=[
            "data",
            "sft",
            "rm",
            "ppo",
            "ppo_stability",
            "ppo_reward_hack_search",
            "ppo_sweep",
            "ppo_kl_sweep",
            "dpo",
            "eval",
            "eval_rollout",
            "plot_proposal",
            "paper_artifacts",
            "replicates",
            "aggregate_seeds",
            "all",
        ],
        help="Pipeline stage; 'all' runs in recommended order (skips optional heavy steps on failure is NOT implemented—runs sequentially).",
    )
    args = p.parse_args(argv)
    from .config import build_global_config
    from .data_hh import run_data_stage
    from .sft_train import run_sft
    from .reward_model_train import run_rm
    from .ppo_train import (
        run_ppo,
        run_ppo_kl_beta_sweep,
        run_ppo_reward_hacking_search,
        run_ppo_stability_exercise,
        run_ppo_sweep,
    )
    from .dpo_train import run_dpo
    from .evaluate import run_eval
    from .eval_rollout import run_eval_rollout
    from .paper_artifacts import run_paper_artifacts
    from .plot_proposal_metrics import run_plot_proposal
    from .replicate_seeds import run_aggregate_seeds, run_replicates

    cfg = build_global_config()

    def _replicates() -> None:
        code = run_replicates()
        if code:
            raise SystemExit(code)

    order = {
        "data": lambda: run_data_stage(cfg.data, cfg.seed),
        "sft": lambda: run_sft(cfg),
        "rm": lambda: run_rm(cfg),
        "ppo": lambda: run_ppo(cfg),
        "ppo_stability": lambda: run_ppo_stability_exercise(cfg),
        "ppo_reward_hack_search": lambda: run_ppo_reward_hacking_search(cfg),
        "ppo_sweep": lambda: run_ppo_sweep(cfg),
        "ppo_kl_sweep": lambda: run_ppo_kl_beta_sweep(cfg),
        "dpo": lambda: run_dpo(cfg),
        "eval": lambda: run_eval(cfg),
        "eval_rollout": lambda: run_eval_rollout(cfg),
        "plot_proposal": lambda: run_plot_proposal(cfg),
        "paper_artifacts": run_paper_artifacts,
        "replicates": _replicates,
        "aggregate_seeds": run_aggregate_seeds,
    }
    if args.stage == "all":
        for k in ["data", "sft", "rm", "ppo", "dpo", "eval", "eval_rollout"]:
            print("===", k, "===")
            try:
                order[k]()
            except Exception as e:  # noqa: BLE001
                print(f"Stage {k} failed: {e}", file=sys.stderr)
                if k in ("ppo", "dpo", "eval_rollout"):
                    print("(Continuing — run this stage separately if needed.)", file=sys.stderr)
                else:
                    raise
        return 0
    order[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
