"""Load optional RM temperature T from calibration (Bradley--Terry scaling on val pairs). Scores are used as s/T in inference (§4.4)."""
from __future__ import annotations

import json
from pathlib import Path

from .paths import rm_dir


def load_rm_temperature(rm_path: Path | None = None) -> float:
    p = (rm_path or rm_dir()) / "calibration.json"
    if not p.is_file():
        return 1.0
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        t = float(d.get("temperature", 1.0))
        return t if t > 1e-8 else 1.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return 1.0


def apply_temperature_scalar(score: float, t: float) -> float:
    if t <= 0:
        return score
    return float(score) / t
