"""
Load Anthropic HH-RLHF from Hugging Face — public dataset, no API key.
Writes train/val/test JSONL + HF Dataset dict for downstream TRL.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from .config import DataConfig
from .paths import data_dir, ensure_dirs


def _common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def split_hh_row(ex: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    From HH-RLHF row, extract (prompt, chosen_response, rejected_response).
    """
    chosen = ex.get("chosen", "")
    rejected = ex.get("rejected", "")
    if not chosen or not rejected:
        return None
    prefix = _common_prefix(chosen, rejected)
    asst = "\n\nAssistant: "
    if asst not in prefix:
        if asst in chosen:
            p = chosen.rfind(asst)
            pr = chosen[: p + len(asst)]
            cr = chosen[p + len(asst) :].strip()
            rr = rejected[p + len(asst) :].strip() if p < len(rejected) else ""
            if pr and cr and rr:
                return {"prompt": pr, "chosen": cr, "rejected": rr}
        return None
    pr = prefix
    c_rest = chosen[len(pr) :].strip()
    r_rest = rejected[len(pr) :].strip()
    if not c_rest or not r_rest:
        return None
    return {"prompt": pr, "chosen": c_rest, "rejected": r_rest}


def build_examples(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ex in raw_rows:
        sp = split_hh_row(ex)
        if not sp:
            continue
        out.append(
            {
                "prompt": sp["prompt"],
                "chosen": sp["chosen"],
                "rejected": sp["rejected"],
            }
        )
    return out


def load_hh_anthropic_raw(cfg: DataConfig):
    return load_dataset(cfg.dataset_name, cfg.dataset_config)


def _dedupe_by_prompt(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the first row per `prompt` so train/val/test splits are disjoint in prompt space (§5.2)."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for e in examples:
        p = e.get("prompt", "")
        if p in seen:
            continue
        seen.add(p)
        out.append(e)
    return out


def run_data_stage(cfg: DataConfig, seed: int) -> Path:
    ensure_dirs()
    out_root = data_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    print("Loading dataset (first download may take a few minutes)...")
    raw = load_hh_anthropic_raw(cfg)
    # train + optional test; merge test into pool for our split
    train_name = "train" if "train" in raw else list(raw.keys())[0]
    pool = [dict(x) for x in raw[train_name]]
    if "test" in raw:
        pool.extend([dict(x) for x in raw["test"]])
    random.Random(seed).shuffle(pool)
    examples = build_examples(pool)
    n_before = len(examples)
    if cfg.dedupe_by_prompt:
        examples = _dedupe_by_prompt(examples)
    n_post = len(examples)
    n = min(n_post, cfg.n_train + cfg.n_val + cfg.n_test)
    examples = examples[:n]
    n_train = min(cfg.n_train, len(examples) - cfg.n_val - cfg.n_test)
    n_val = min(cfg.n_val, len(examples) - n_train - 1)
    n_test = min(cfg.n_test, len(examples) - n_train - n_val)
    tr = examples[:n_train]
    va = examples[n_train : n_train + n_val]
    te = examples[n_train + n_val : n_train + n_val + n_test]

    def wjsonl(path: Path, rows: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    wjsonl(out_root / "train.jsonl", tr)
    wjsonl(out_root / "val.jsonl", va)
    wjsonl(out_root / "test.jsonl", te)
    dsd = DatasetDict(
        {
            "train": Dataset.from_list(tr),
            "validation": Dataset.from_list(va),
            "test": Dataset.from_list(te),
        }
    )
    dsd.save_to_disk(str(out_root / "hf_dataset"))
    meta: Dict[str, Any] = {
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "source": f"{cfg.dataset_name}:{cfg.dataset_config}",
        "raw_text_character_truncation": False,
    }
    if cfg.dedupe_by_prompt:
        meta["dedupe_by_prompt"] = True
        meta["n_examples_parsed_pre_dedup"] = n_before
        meta["n_examples_post_dedup"] = n_post
    else:
        meta["dedupe_by_prompt"] = False
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_root} (train={len(tr)}, val={len(va)}, test={len(te)})")
    return out_root
