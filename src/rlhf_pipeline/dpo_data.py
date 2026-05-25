"""
Helpers for TRL DPO: BPE tokenizers do not guarantee tokenize(A)+tokenize(B) == tokenize(A+B),
so TRL's check that tokenize(prompt) is a prefix of tokenize(prompt+completion) can fail. We
right-trim the prompt and move those characters to both completions so the full strings are
unchanged and the token-prefix property holds.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["align_preference_for_dpo"]


def _is_tok_prefix(
    tok: "PreTrainedTokenizerBase", prefix: str, full: str
) -> bool:
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    return len(f_ids) >= len(p_ids) and f_ids[: len(p_ids)] == p_ids


def align_preference_for_dpo(
    tok: "PreTrainedTokenizerBase",
    prompt: str,
    chosen: str,
    rejected: str,
    max_trims: int = 10_000,
) -> Tuple[str, str, str]:
    """
    Shrink `prompt` from the right (moving characters onto both completion strings) until
    tokenize(prompt) is a prefix of both tokenize(prompt+chosen) and tokenize(prompt+rejected).
    Invariants: prompt+chosen and prompt+rejected (as full user/assistant text) are unchanged
    in aggregate across the three return fields.
    """
    p, c, r = prompt, chosen, rejected
    for _ in range(max_trims):
        if not p:
            break
        if _is_tok_prefix(tok, p, p + c) and _is_tok_prefix(tok, p, p + r):
            return p, c, r
        p, tail = p[:-1], p[-1]
        c, r = tail + c, tail + r
    return prompt, chosen, rejected


def align_row_for_dpo(
    tok: "PreTrainedTokenizerBase", row: Dict[str, Any]
) -> Dict[str, str]:
    p, c, r = align_preference_for_dpo(
        tok, row["prompt"], row["chosen"], row["rejected"]
    )
    return {"prompt": p, "chosen": c, "rejected": r}
