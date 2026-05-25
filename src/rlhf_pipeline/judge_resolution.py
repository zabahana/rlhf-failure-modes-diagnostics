"""
Which judge to call for R^†: mock, Anthropic, or OpenAI; optional second judge for disagreement.
Env-driven (no extra config flags required).
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

from .judge_anthropic import anthropic_helpfulness_1_to_10, use_anthropic_judge_enabled
from .judge_openai import openai_helpfulness_1_to_10, use_openai_judge_as_primary
from .load_env import load_code_dotenv

JudgeFn = Callable[[str], float]


def _has_key(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def resolve_primary_judge() -> Tuple[JudgeFn, str]:
    """
    Primary R^†. Priority (first match):
      - RLHF_JUDGE=mock  -> length heuristic
      - RLHF_JUDGE=openai + OPENAI_API_KEY
      - RLHF_JUDGE=anthropic + ANTHROPIC_API_KEY
      - RLHF_USE_OPENAI_JUDGE=1 + OPENAI (OpenAI as primary; proposal “Judge 1” = GPT is OK)
      - RLHF_USE_ANTHROPIC_JUDGE=1 + ANTHROPIC (Claude, legacy default)
      - else mock
    """
    load_code_dotenv()
    ex = os.environ.get("RLHF_JUDGE", "").strip().lower()
    if ex == "mock":
        from .judge_mock import mock_judge_score

        return mock_judge_score, "mock"
    if ex == "openai" and _has_key("OPENAI_API_KEY"):
        return openai_helpfulness_1_to_10, "openai"
    if ex == "anthropic" and _has_key("ANTHROPIC_API_KEY"):
        return anthropic_helpfulness_1_to_10, "anthropic"
    if use_openai_judge_as_primary() and _has_key("OPENAI_API_KEY"):
        return openai_helpfulness_1_to_10, "openai"
    if use_anthropic_judge_enabled() and _has_key("ANTHROPIC_API_KEY"):
        return anthropic_helpfulness_1_to_10, "anthropic"
    from .judge_mock import mock_judge_score

    return mock_judge_score, "mock"


def resolve_second_judge(primary: str) -> Tuple[Optional[JudgeFn], str]:
    """
    Optional second judge (proposal: report disagreement). Env:
      RLHF_JUDGE_2=openai|anthropic|none
    Skipped if same as primary or no API key.
    """
    load_code_dotenv()
    ex = os.environ.get("RLHF_JUDGE_2", "").strip().lower()
    if ex in ("", "none", "0", "false", "off"):
        return None, ""
    if ex == "openai" and primary != "openai" and _has_key("OPENAI_API_KEY"):
        return openai_helpfulness_1_to_10, "openai"
    if ex == "anthropic" and primary != "anthropic" and _has_key("ANTHROPIC_API_KEY"):
        return anthropic_helpfulness_1_to_10, "anthropic"
    return None, ""


def sleep_after_judge_call(label: str) -> None:
    """Throttle API usage; seconds from env, label-specific or generic."""
    import time

    if label == "anthropic" and (d := float(os.environ.get("RLHF_ANTHROPIC_JUDGE_DELAY_S", "0.25") or 0.0)) > 0:
        time.sleep(d)
    if label == "openai" and (d2 := float(os.environ.get("RLHF_OPENAI_JUDGE_DELAY_S", "0.1") or 0.0)) > 0:
        time.sleep(d2)
