"""
§5.3: run the same stage list for each seed in `cfg.proposal_seeds`, then aggregate `eval` metrics.

- `replicates` — one subprocess per (seed, stage) with `RLHF_ARTIFACTS` = base/seed_{seed}/.
- `aggregate_seeds` — mean/std for numeric fields in `eval/summary.json` and top-level `rollout_summary.json`.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import build_global_config
from .load_env import load_code_dotenv


def _replicate_base() -> Path:
    b = os.environ.get("RLHF_REPLICATE_BASE", "").strip()
    if b:
        return Path(b).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "artifacts").resolve()


def _seeds_from_cfg() -> List[int]:
    load_code_dotenv()
    cfg = build_global_config()
    return [int(float(x)) for x in cfg.proposal_seeds]


def _stages_from_env() -> List[str]:
    raw = os.environ.get(
        "RLHF_REPLICATE_STAGES",
        "data,sft,rm,ppo,dpo,eval,eval_rollout",
    )
    return [s.strip() for s in raw.split(",") if s.strip()]


def _seed_dir(base: Path, seed: int) -> Path:
    return base / f"seed_{seed}"


def run_replicates() -> int:
    load_code_dotenv()
    base = _replicate_base()
    base.mkdir(parents=True, exist_ok=True)
    seeds = _seeds_from_cfg()
    stages = _stages_from_env()
    code_root = Path(__file__).resolve().parent.parent
    for seed in seeds:
        sd = _seed_dir(base, seed)
        sd.mkdir(parents=True, exist_ok=True)
        for st in stages:
            env = os.environ.copy()
            env["RLHF_SEED"] = str(seed)
            env["RLHF_ARTIFACTS"] = str(sd)
            print(f"=== replicate seed={seed} stage={st} -> {sd} ===", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "rlhf_pipeline.main", st],
                cwd=str(code_root),
                env=env,
            )
            if r.returncode != 0:
                print(
                    f"replicate: seed {seed} stage {st} failed (code {r.returncode})",
                    file=sys.stderr,
                )
                return r.returncode
    print(f"replicates: done. Artifacts under {base}/seed_*/", flush=True)
    return 0


def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
            return False
        return True
    return False


def _collect_json_numbers(obj: Any, prefix: str = "") -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.extend(_collect_json_numbers(v, p))
    elif _is_number(obj) and prefix:
        out.append((prefix, float(obj)))
    return out


def run_aggregate_seeds() -> Path:
    load_code_dotenv()
    base = _replicate_base()
    if not base.is_dir():
        raise FileNotFoundError(f"RLHF_REPLICATE_BASE or default not found: {base}")
    pattern = re.compile(r"^seed_(\d+)$")
    seed_dirs: List[Tuple[int, Path]] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            seed_dirs.append((int(m.group(1)), p))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* under {base}")

    by_metric: Dict[str, List[float]] = {}
    for _seed, d in seed_dirs:
        sfile = d / "eval" / "summary.json"
        if sfile.is_file():
            dct = json.loads(sfile.read_text(encoding="utf-8"))
            for key, val in _collect_json_numbers(dct):
                if key == "n":
                    continue
                by_metric.setdefault(key, []).append(val)
        rfile = d / "eval" / "rollout_summary.json"
        if rfile.is_file():
            dct = json.loads(rfile.read_text(encoding="utf-8"))
            for key, val in _collect_json_numbers(dct):
                if "by_model" in key or key in (
                    "n_examples",
                    "n_paired_dpo_sft",
                ):
                    continue
                if key in ("judge", "note", "models"):
                    continue
                by_metric.setdefault(f"rollout.{key}", []).append(val)

    out_stats: Dict[str, Any] = {"n_seeds": len(seed_dirs), "seeds": [s for s, _ in seed_dirs]}
    for k, vals in sorted(by_metric.items()):
        if not vals:
            continue
        if len(vals) < 2:
            out_stats[k] = {
                "mean": statistics.mean(vals),
                "stdev": 0.0,
                "n": len(vals),
                "values": vals,
            }
        else:
            out_stats[k] = {
                "mean": statistics.mean(vals),
                "stdev": statistics.pstdev(vals),
                "n": len(vals),
                "values": vals,
            }
    out_path = base / "aggregated_seeds.json"
    out_path.write_text(json.dumps(out_stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out_stats, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return out_path
