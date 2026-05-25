"""Load `code/.env` into os.environ (optional python-dotenv)."""
from __future__ import annotations

import os
from pathlib import Path


def load_code_dotenv() -> None:
    """Resolves the `code/` root (parent of this package) and loads `.env` if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    code_root = Path(__file__).resolve().parent.parent
    p = code_root / ".env"
    if p.is_file():
        load_dotenv(p, override=False)
