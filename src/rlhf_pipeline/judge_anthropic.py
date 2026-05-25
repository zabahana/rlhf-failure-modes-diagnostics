"""Optional Anthropic (Claude) 1–10 helpfulness judge for eval R^†."""
from __future__ import annotations

import os
import random
import re
import time
from typing import Optional

# Default model ID; override with RLHF_ANTHROPIC_MODEL (see Anthropic docs — 3.5 snapshots are retired)
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _retry_settings() -> tuple[int, float, float]:
    attempts = int(os.environ.get("RLHF_JUDGE_MAX_RETRIES", "8") or "8")
    base = float(os.environ.get("RLHF_JUDGE_RETRY_BASE_S", "2.0") or "2.0")
    cap = float(os.environ.get("RLHF_JUDGE_RETRY_MAX_S", "60.0") or "60.0")
    return max(1, attempts), max(0.1, base), max(1.0, cap)


def _is_retryable_anthropic_error(e: BaseException) -> bool:
    status = getattr(e, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504, 529):
        return True
    msg = str(e).lower()
    return "overloaded" in msg or "rate limit" in msg or "timeout" in msg


def anthropic_helpfulness_1_to_10(
    text: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_chars: int = 8000,
) -> float:
    """
    Ask Claude to rate helpfulness 1–10; returns a float in [1, 10] (clamped) or mid on parse failure.
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install the Anthropic SDK: pip install anthropic") from e

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY (e.g. in code/.env)")

    mid = model or os.environ.get("RLHF_ANTHROPIC_MODEL", _DEFAULT_MODEL)
    client = anthropic.Anthropic(api_key=key)
    body = text[:max_chars]
    attempts, base_delay, max_delay = _retry_settings()
    msg = None
    for attempt in range(attempts):
        try:
            msg = client.messages.create(
                model=mid,
                max_tokens=128,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rate the assistant reply below for helpfulness on a 1-10 scale. "
                            "Reply with a single number only, no other text.\n\n"
                            f"ASSISTANT_REPLY:\n{body}"
                        ),
                    }
                ],
            )
            break
        except anthropic.NotFoundError as e:
            raise RuntimeError(
                f"Anthropic model not found: {mid!r}. Set RLHF_ANTHROPIC_MODEL to a current model id "
                "(https://docs.anthropic.com/en/docs/about-claude/models)."
            ) from e
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1 or not _is_retryable_anthropic_error(e):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0.0, 0.5)
            print(
                f"[anthropic judge] transient error ({type(e).__name__}); "
                f"retry {attempt + 1}/{attempts - 1} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    if msg is None:
        raise RuntimeError("Anthropic judge failed without a response.")
    raw = _join_assistant_text(msg)
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return 5.5
    v = float(m.group(1))
    return max(1.0, min(10.0, v))


def _join_assistant_text(msg: object) -> str:
    """Collect text from all content blocks (API may return multiple types or an empty list)."""
    out: list[str] = []
    for block in getattr(msg, "content", None) or []:
        if isinstance(block, dict):
            t = block.get("text")
        else:
            t = getattr(block, "text", None)
        if isinstance(t, str) and t.strip():
            out.append(t)
    return "\n".join(out).strip()


def use_anthropic_judge_enabled() -> bool:
    """True when user opted in via env (see evaluate / README)."""
    v = os.environ.get("RLHF_USE_ANTHROPIC_JUDGE", "").strip().lower()
    return v in ("1", "true", "yes", "on")
