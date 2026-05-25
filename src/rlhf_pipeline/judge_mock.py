"""Length-based mock judge (no API)."""
from __future__ import annotations


def mock_judge_score(text: str) -> float:
    """
    Placeholder R†; favors moderate length. Replace with API judges in production.
    """
    l = len(text)
    return float(max(0.0, 8.0 - abs(l - 500) / 200.0))
