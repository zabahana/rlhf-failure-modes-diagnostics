"""
Figures for RESEARCH_PROPOSAL_AND_METHODOLOGY §5.5: training KL vs step, and optional
Pareto-style scatter from rollout_summary.json.
Requires `matplotlib` (install if missing: pip install matplotlib).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import GlobalConfig
from .paths import eval_dir, ppo_dir, ensure_dirs


def _read_csv_log(path: Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    steps: List[int] = []
    kls: List[float] = []
    rs: List[float] = []
    us: List[float] = []
    if not path.is_file():
        return steps, kls, rs, us
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            kls.append(float(row.get("kl_seq", 0) or 0))
            rs.append(float(row.get("R_phi", 0) or 0))
            us.append(float(row.get("u", 0) or 0))
    return steps, kls, rs, us


def run_plot_proposal(cfg: GlobalConfig) -> Path:
    ensure_dirs()
    out = eval_dir() / "plots"
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install matplotlib to use plot_proposal: pip install matplotlib") from e

    pd = ppo_dir()
    candidates: List[Tuple[Path, str]] = [(pd, "root")]
    candidates += [(p, p.name) for p in sorted(pd.glob("up_lambda_*")) if p.is_dir()]
    for base, title in candidates:
        logf = base / (cfg.ppo.training_log_name or "training_log.csv")
        steps, kls, rs, us = _read_csv_log(logf)
        if not steps:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax2 = ax.twinx()
        ax.plot(steps, kls, color="C0", label="seq. KL (mean log π- log π_ref)")
        ax2.plot(steps, rs, color="C1", alpha=0.7, label="R_φ", linestyle="--")
        ax2.plot(steps, us, color="C2", alpha=0.7, label="u (MC std)", linestyle=":")
        ax.set_xlabel("outer step")
        ax.set_ylabel("KL (approx.)", color="C0")
        ax2.set_ylabel("R_φ, u", color="C1")
        fig.suptitle(f"RL training log — {title}")
        fig.tight_layout()
        safe = title.replace(os.sep, "_")
        out_png = out / f"training_kl_r_u_{safe}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}", flush=True)

    roll = eval_dir() / "rollout_summary.json"
    if roll.is_file():
        dat: Dict[str, Any] = json.loads(roll.read_text(encoding="utf-8"))
        bm = dat.get("by_model") or {}
        xs, ys, labels = [], [], []
        for m, v in bm.items():
            if "mean_R_phi" in v and "mean_R_dagger" in v and "mean_kl_sft" in v:
                xs.append(v["mean_R_phi"])
                ys.append(v["mean_R_dagger"])
                labels.append(f"{m}\nKL={v['mean_kl_sft']:.3f}")
            elif "mean_R_phi" in v and "mean_R_dagger" in v:
                xs.append(v["mean_R_phi"])
                ys.append(v["mean_R_dagger"])
                labels.append(m)
        if len(xs) >= 2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(xs, ys, s=64)
            for i, t in enumerate(labels):
                ax.annotate(
                    t.split("\n")[0], (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4)
                )
            ax.set_xlabel("mean R_φ (proxy)")
            ax.set_ylabel("mean R† (judge)")
            ax.set_title("Pareto-style: judge vs proxy (rollout_summary)")
            fig.tight_layout()
            ppath = out / "pareto_judge_vs_proxy.png"
            fig.savefig(ppath, dpi=150)
            plt.close(fig)
            print(f"Wrote {ppath}", flush=True)

    return out
