"""Shared artifact layout (each stage reads/writes the same tree). Colab: set RLHF_ARTIFACTS before import."""
import os
from pathlib import Path

_default = Path(__file__).resolve().parent.parent / "artifacts"


def _use_seed_subdir() -> bool:
    """If True, use $RLHF_ARTIFACTS/seed_{RLHF_SEED}/ (unless path already has seed_* in the last segment)."""
    v = os.environ.get("RLHF_ARTIFACTS_SEED_SUBDIR", "").strip().lower()
    return v in ("1", "true", "yes")


def artifacts_root() -> Path:
    p = os.environ.get("RLHF_ARTIFACTS", str(_default))
    r = Path(p).expanduser().resolve()
    if _use_seed_subdir() and "seed_" not in r.name:
        seed = (os.environ.get("RLHF_SEED", "42") or "42").strip() or "42"
        r = r / f"seed_{seed}"
    r.mkdir(parents=True, exist_ok=True)
    return r


# Back-compat: some modules use ARTIFACTS at import time
ARTIFACTS = Path(os.environ.get("RLHF_ARTIFACTS", str(_default))).expanduser().resolve()


def ensure_dirs() -> None:
    root = artifacts_root()
    for sub in ("data", "sft", "rm", "ppo", "dpo", "eval", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def data_dir() -> Path:
    return artifacts_root() / "data"


def sft_dir() -> Path:
    return artifacts_root() / "sft"


def rm_dir() -> Path:
    return artifacts_root() / "rm"


def ppo_dir() -> Path:
    return artifacts_root() / "ppo"


def ppo_policy_path(relative: str = "policy_after_pilot") -> Path:
    """Trained policy checkpoint, e.g. `policy_after_pilot` or `up_lambda_0.5/policy_after_pilot`."""
    return ppo_dir() / relative


def dpo_dir() -> Path:
    return artifacts_root() / "dpo"


def eval_dir() -> Path:
    return artifacts_root() / "eval"
