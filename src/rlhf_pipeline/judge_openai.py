"""Optional OpenAI 1–10 helpfulness judge for eval R^† (second judge or primary)."""
from __future__ import annotations

import os
import random
import re
import time
from typing import Optional

# Cheap default; override with RLHF_OPENAI_MODEL
_DEFAULT_MODEL = "gpt-4o-mini"


def _retry_settings() -> tuple[int, float, float]:
    attempts = int(os.environ.get("RLHF_JUDGE_MAX_RETRIES", "8") or "8")
    base = float(os.environ.get("RLHF_JUDGE_RETRY_BASE_S", "2.0") or "2.0")
    cap = float(os.environ.get("RLHF_JUDGE_RETRY_MAX_S", "60.0") or "60.0")
    return max(1, attempts), max(0.1, base), max(1.0, cap)


def _is_retryable_openai_error(e: BaseException) -> bool:
    status = getattr(e, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504, 529):
        return True
    msg = str(e).lower()
    return (
        "connection error" in msg
        or "connecterror" in msg
        or "nodename nor servname" in msg
        or "rate limit" in msg
        or "timeout" in msg
        or "temporarily unavailable" in msg
    )


def openai_helpfulness_1_to_10(
    text: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_chars: int = 8000,
) -> float:
    """
    Ask an OpenAI chat model to rate helpfulness 1–10; returns a float in [1, 10] (clamped)
    or mid on parse failure.
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install the OpenAI SDK: pip install openai") from e

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY (e.g. in code/.env)")

    mid = (model or os.environ.get("RLHF_OPENAI_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()
    client = OpenAI(api_key=key)
    body = text[:max_chars]
    attempts, base_delay, max_delay = _retry_settings()
    resp = None
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=mid,
                max_tokens=32,
                temperature=0.2,
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
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1 or not _is_retryable_openai_error(e):
                raise RuntimeError(
                    f"OpenAI chat.completions failed (model={mid!r}). Check RLHF_OPENAI_MODEL and billing."
                ) from e
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0.0, 0.5)
            print(
                f"[openai judge] transient error ({type(e).__name__}); "
                f"retry {attempt + 1}/{attempts - 1} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    if resp is None:
        raise RuntimeError("OpenAI judge failed without a response.")
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return 5.5
    v = float(m.group(1))
    return max(1.0, min(10.0, v))


def use_openai_judge_as_primary() -> bool:
    """User wants OpenAI as the primary R^† (see judge_resolution)."""
    v = os.environ.get("RLHF_USE_OPENAI_JUDGE", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("RLHF_JUDGE", "").strip().lower() == "openai":
        return True
    return False
