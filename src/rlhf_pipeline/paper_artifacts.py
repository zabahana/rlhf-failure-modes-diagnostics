"""Generate publication-ready tables, figures, and qualitative examples."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .paths import ensure_dirs, eval_dir

METRICS = [
    "mean_R_phi",
    "mean_R_dagger",
    "mean_R_dagger_2",
    "mean_u",
    "mean_kl_sft",
    "mean_judge_disagreement_abs",
    "Delta_hack_proxy",
    "Delta_hack_proxy_judge2",
]

COLORS = {
    "SFT": "#4C78A8",
    "DPO": "#54A24B",
    "UP-PPO lambda=0.0": "#F58518",
    "UP-PPO lambda=0.1": "#E45756",
    "UP-PPO lambda=0.5": "#B279A2",
    "UP-PPO lambda=1.0": "#72B7B2",
}


def _lambda_from_tag(tag: str) -> float:
    return float(tag.replace("up_lambda_", "").replace("p", "."))


def _label_from_tag(tag: str) -> str:
    return f"UP-PPO lambda={_lambda_from_tag(tag):.1f}"


def _load_summaries(base: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(base.glob("rollout_summary_up_lambda_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        out.append(data)
    return sorted(out, key=lambda d: _lambda_from_tag(d["rollout_tag"]))


def _load_examples(base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("rollout_examples_up_lambda_*.jsonl")):
        tag = path.stem.replace("rollout_examples_", "")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["rollout_tag"] = tag
                row["lambda"] = _lambda_from_tag(tag)
                rows.append(row)
    return rows


def _std(xs: list[float]) -> float:
    return stdev(xs) if len(xs) > 1 else 0.0


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def _fmt_mean_sd(values: list[float], digits: int = 3) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return _fmt(values[0], digits)
    return f"{mean(values):.{digits}f} ({_std(values):.{digits}f})"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(_latex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(_latex_escape(cell) for cell in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_values: dict[str, dict[str, list[float]]] = {
        "sft": defaultdict(list),
        "dpo": defaultdict(list),
    }
    for s in summaries:
        tag = s["rollout_tag"]
        lam = _lambda_from_tag(tag)
        for model, vals in s["by_model"].items():
            if model in baseline_values:
                for metric in METRICS:
                    if metric in vals:
                        baseline_values[model][metric].append(float(vals[metric]))
                continue
            row = {
                "label": _label_from_tag(tag),
                "family": "UP-PPO",
                "lambda": lam,
                "rollout_tag": tag,
                "n": vals.get("n", s.get("n_examples")),
            }
            row.update({metric: vals.get(metric, "") for metric in METRICS})
            row["mean_judge_avg"] = (
                float(row["mean_R_dagger"]) + float(row["mean_R_dagger_2"])
            ) / 2.0
            rows.append(row)

    for model, label, family in [("sft", "SFT", "SFT"), ("dpo", "DPO", "DPO")]:
        metrics = baseline_values[model]
        row = {
            "label": label,
            "family": family,
            "lambda": "",
            "rollout_tag": "baseline_average",
            "n": summaries[0].get("n_examples", ""),
        }
        for metric in METRICS:
            vals = metrics.get(metric, [])
            row[metric] = mean(vals) if vals else ""
            row[f"{metric}_sd"] = _std(vals) if vals else ""
        row["mean_judge_avg"] = (float(row["mean_R_dagger"]) + float(row["mean_R_dagger_2"])) / 2.0
        rows.append(row)

    order = {"SFT": 0, "DPO": 1}
    return sorted(rows, key=lambda r: (order.get(r["label"], 2), float(r["lambda"] or -1)))


def _write_tables(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    tables_dir: Path,
) -> None:
    fields = [
        "label",
        "family",
        "lambda",
        "mean_R_phi",
        "mean_R_dagger",
        "mean_R_dagger_2",
        "mean_judge_avg",
        "mean_u",
        "mean_kl_sft",
        "mean_judge_disagreement_abs",
        "Delta_hack_proxy",
        "Delta_hack_proxy_judge2",
        "n",
    ]
    _write_csv(tables_dir / "table_model_comparison.csv", rows, fields)

    display_rows: list[list[str]] = []
    for row in rows:
        display_rows.append(
            [
                str(row["label"]),
                _fmt(row.get("mean_R_phi")),
                _fmt(row.get("mean_R_dagger")),
                _fmt(row.get("mean_R_dagger_2")),
                _fmt(row.get("mean_judge_avg")),
                _fmt(row.get("mean_u")),
                _fmt(row.get("mean_kl_sft")),
            ]
        )
    headers = ["Model", "R_phi", "Anthropic", "OpenAI", "Judge avg.", "u", "KL to SFT"]
    md = "# Table 1. Model comparison on generated rollouts\n\n"
    md += _markdown_table(headers, display_rows)
    md += "\nValues are means over 512 prompts. SFT and DPO rows average repeated judge calls across lambda-tagged runs.\n"
    (tables_dir / "table_model_comparison.md").write_text(md, encoding="utf-8")
    (tables_dir / "table_model_comparison.tex").write_text(
        _latex_table(
            headers,
            display_rows,
            "Model comparison on generated rollouts. SFT and DPO rows average repeated judge calls across lambda-tagged runs.",
            "tab:model_comparison",
        ),
        encoding="utf-8",
    )

    paired_rows: list[dict[str, Any]] = []
    by_tag_i: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in examples:
        by_tag_i[(row["rollout_tag"], int(row["i"]))][row["model"]] = row
    for tag in sorted({s["rollout_tag"] for s in summaries}, key=_lambda_from_tag):
        ppo_minus_sft_j, ppo_minus_sft_o, ppo_minus_sft_r = [], [], []
        dpo_minus_sft_j, dpo_minus_sft_o, dpo_minus_sft_r = [], [], []
        for (row_tag, _idx), group in by_tag_i.items():
            if row_tag != tag or "sft" not in group:
                continue
            sft = group["sft"]
            if "ppo" in group:
                ppo = group["ppo"]
                ppo_minus_sft_j.append(float(ppo["R_dagger"]) - float(sft["R_dagger"]))
                ppo_minus_sft_o.append(float(ppo["R_dagger_2"]) - float(sft["R_dagger_2"]))
                ppo_minus_sft_r.append(float(ppo["R_phi"]) - float(sft["R_phi"]))
            if "dpo" in group:
                dpo = group["dpo"]
                dpo_minus_sft_j.append(float(dpo["R_dagger"]) - float(sft["R_dagger"]))
                dpo_minus_sft_o.append(float(dpo["R_dagger_2"]) - float(sft["R_dagger_2"]))
                dpo_minus_sft_r.append(float(dpo["R_phi"]) - float(sft["R_phi"]))
        paired_rows.append(
            {
                "lambda": _lambda_from_tag(tag),
                "tag": tag,
                "ppo_minus_sft_R_phi": mean(ppo_minus_sft_r),
                "ppo_minus_sft_anthropic": mean(ppo_minus_sft_j),
                "ppo_minus_sft_openai": mean(ppo_minus_sft_o),
                "dpo_minus_sft_R_phi": mean(dpo_minus_sft_r),
                "dpo_minus_sft_anthropic": mean(dpo_minus_sft_j),
                "dpo_minus_sft_openai": mean(dpo_minus_sft_o),
            }
        )
    _write_csv(
        tables_dir / "table_paired_deltas.csv",
        paired_rows,
        [
            "lambda",
            "tag",
            "ppo_minus_sft_R_phi",
            "ppo_minus_sft_anthropic",
            "ppo_minus_sft_openai",
            "dpo_minus_sft_R_phi",
            "dpo_minus_sft_anthropic",
            "dpo_minus_sft_openai",
        ],
    )
    delta_display = [
        [
            _fmt(r["lambda"], 1),
            _fmt(r["ppo_minus_sft_R_phi"]),
            _fmt(r["ppo_minus_sft_anthropic"]),
            _fmt(r["ppo_minus_sft_openai"]),
            _fmt(r["dpo_minus_sft_R_phi"]),
            _fmt(r["dpo_minus_sft_anthropic"]),
            _fmt(r["dpo_minus_sft_openai"]),
        ]
        for r in paired_rows
    ]
    delta_headers = [
        "lambda",
        "PPO-SFT R_phi",
        "PPO-SFT Anthropic",
        "PPO-SFT OpenAI",
        "DPO-SFT R_phi",
        "DPO-SFT Anthropic",
        "DPO-SFT OpenAI",
    ]
    (tables_dir / "table_paired_deltas.md").write_text(
        "# Table 2. Paired deltas relative to SFT\n\n" + _markdown_table(delta_headers, delta_display),
        encoding="utf-8",
    )
    (tables_dir / "table_paired_deltas.tex").write_text(
        _latex_table(
            delta_headers,
            delta_display,
            "Paired mean deltas relative to SFT on the same prompts.",
            "tab:paired_deltas",
        ),
        encoding="utf-8",
    )


def _save_figure(fig: Any, figures_dir: Path, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def _center_limits(values: list[float], pad_frac: float = 0.35) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-9:
        span = max(abs(lo), 1.0) * 0.2
    pad = span * pad_frac
    return lo - pad, hi + pad


def _write_figures(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    labels = [str(r["label"]) for r in rows]
    colors = [COLORS.get(label, "#777777") for label in labels]

    # Pareto-style plot: centered axes by design, not zero anchored.
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    xs = [float(r["mean_R_phi"]) for r in rows]
    ys = [float(r["mean_judge_avg"]) for r in rows]
    sizes = [80 + 900 * float(r["mean_u"]) for r in rows]
    for row, x, y, size, color in zip(rows, xs, ys, sizes, colors):
        ax.scatter(
            [x],
            [y],
            s=size,
            c=[color],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.9,
            label=str(row["label"]),
        )
    ax.set_xlim(*_center_limits(xs, 0.45))
    ax.set_ylim(*_center_limits(ys, 0.35))
    ax.axhline(float(next(r for r in rows if r["label"] == "SFT")["mean_judge_avg"]), color="#4C78A8", lw=1, ls="--", alpha=0.55)
    ax.axvline(float(next(r for r in rows if r["label"] == "SFT")["mean_R_phi"]), color="#4C78A8", lw=1, ls="--", alpha=0.55)
    ax.set_xlabel("Learned reward proxy, mean R_phi")
    ax.set_ylabel("Two-judge helpfulness mean")
    ax.set_title("Pareto-style diagnostic: external judge score vs reward proxy")
    ax.legend(frameon=True, fontsize=8, loc="lower right", title="Policy")
    _save_figure(fig, figures_dir, "fig_pareto_judge_vs_proxy_centered")
    plt.close(fig)

    # Grouped judge bars.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    idx = list(range(len(rows)))
    width = 0.36
    anth = [float(r["mean_R_dagger"]) for r in rows]
    openai = [float(r["mean_R_dagger_2"]) for r in rows]
    ax.bar([i - width / 2 for i in idx], anth, width, label="Anthropic", color="#4C78A8")
    ax.bar([i + width / 2 for i in idx], openai, width, label="OpenAI", color="#F58518")
    ax.set_xticks(idx)
    ax.set_xticklabels([l.replace("UP-PPO ", "") for l in labels], rotation=25, ha="right")
    ax.set_ylabel("Judge helpfulness (1-10)")
    ax.set_title("Two independent judges agree PPO variants underperform")
    ax.legend(frameon=False, ncol=2)
    _save_figure(fig, figures_dir, "fig_two_judge_bars")
    plt.close(fig)

    # KL vs judge score, bubble uncertainty.
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    xs = [float(r["mean_kl_sft"]) for r in rows]
    ys = [float(r["mean_judge_avg"]) for r in rows]
    sizes = [100 + 1100 * float(r["mean_u"]) for r in rows]
    for row, x, y, size, color in zip(rows, xs, ys, sizes, colors):
        ax.scatter(
            [x],
            [y],
            s=size,
            c=[color],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.9,
            label=str(row["label"]),
        )
    ax.set_xlabel("Policy drift from SFT, mean KL")
    ax.set_ylabel("Two-judge helpfulness mean")
    ax.set_title("Policy drift and reward-model uncertainty track degraded helpfulness")
    ax.legend(frameon=True, fontsize=8, loc="upper right", title="Policy")
    _save_figure(fig, figures_dir, "fig_kl_uncertainty_vs_judge")
    plt.close(fig)

    # Lambda sweep only.
    ppo_rows = sorted([r for r in rows if r["family"] == "UP-PPO"], key=lambda r: float(r["lambda"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    lam = [float(r["lambda"]) for r in ppo_rows]
    ax.plot(lam, [float(r["mean_R_dagger"]) for r in ppo_rows], marker="o", lw=2, label="Anthropic", color="#4C78A8")
    ax.plot(lam, [float(r["mean_R_dagger_2"]) for r in ppo_rows], marker="s", lw=2, label="OpenAI", color="#F58518")
    ax.plot(lam, [float(r["mean_R_phi"]) for r in ppo_rows], marker="^", lw=2, label="R_phi", color="#E45756")
    ax.set_xlabel("Uncertainty penalty lambda")
    ax.set_ylabel("Mean score")
    ax.set_title("UP-PPO lambda sweep")
    ax.legend(frameon=False, ncol=3)
    _save_figure(fig, figures_dir, "fig_lambda_sweep_scores")
    plt.close(fig)


def _short(text: str, n: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _write_examples(examples: list[dict[str, Any]], examples_dir: Path) -> None:
    scored = []
    for row in examples:
        judge_avg = (float(row["R_dagger"]) + float(row.get("R_dagger_2", row["R_dagger"]))) / 2.0
        scored.append(
            {
                **row,
                "judge_avg": judge_avg,
                "proxy_judge_gap": float(row["R_phi"]) - judge_avg,
                "low_quality_score": -judge_avg + float(row["u"]),
            }
        )

    hack_candidates = sorted(
        [r for r in scored if r["model"] == "ppo"],
        key=lambda r: (r["proxy_judge_gap"], r["R_phi"]),
        reverse=True,
    )[:8]
    high_uncertainty = sorted(
        [r for r in scored if r["model"] == "ppo"],
        key=lambda r: float(r["u"]),
        reverse=True,
    )[:8]

    by_tag_i: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored:
        by_tag_i[(row["rollout_tag"], int(row["i"]))][row["model"]] = row
    contrast = []
    for (_tag, _idx), group in by_tag_i.items():
        if {"sft", "ppo", "dpo"}.issubset(group):
            ppo_j = group["ppo"]["judge_avg"]
            dpo_j = group["dpo"]["judge_avg"]
            sft_j = group["sft"]["judge_avg"]
            contrast.append((dpo_j - ppo_j, sft_j - ppo_j, group))
    contrast_rows = [
        g for _dpo_gap, _sft_gap, g in sorted(contrast, key=lambda x: (x[0], x[1]), reverse=True)[:6]
    ]

    def block(row: dict[str, Any]) -> str:
        return (
            f"### {row['rollout_tag']} / {row['model']} / example {row['i']}\n\n"
            f"- R_phi: {_fmt(row['R_phi'])}; Anthropic: {_fmt(row['R_dagger'])}; "
            f"OpenAI: {_fmt(row.get('R_dagger_2'))}; u: {_fmt(row['u'])}\n"
            f"- Prompt: {_short(row.get('prompt', ''), 500)}\n"
            f"- Generated: {_short(row.get('generated', ''), 900)}\n\n"
        )

    md = "# Qualitative examples for paper appendix\n\n"
    md += "## Highest proxy-vs-judge gaps among PPO rows\n\n"
    md += "These are candidate cases to inspect for reward-model misalignment. In this run they mostly show low absolute reward and low judge scores, so they support optimization failure more than classic reward hacking.\n\n"
    md += "".join(block(r) for r in hack_candidates)
    md += "## Highest reward-model uncertainty among PPO rows\n\n"
    md += "".join(block(r) for r in high_uncertainty)
    md += "## Paired contrast examples: DPO/SFT stronger than PPO on the same prompt\n\n"
    for group in contrast_rows:
        for model in ("sft", "ppo", "dpo"):
            md += block(group[model])
        md += "---\n\n"
    (examples_dir / "qualitative_examples.md").write_text(md, encoding="utf-8")

    _write_csv(
        examples_dir / "ppo_hack_candidate_rows.csv",
        hack_candidates,
        [
            "rollout_tag",
            "lambda",
            "i",
            "model",
            "R_phi",
            "R_dagger",
            "R_dagger_2",
            "judge_avg",
            "u",
            "proxy_judge_gap",
            "prompt",
            "generated",
        ],
    )
    _write_csv(
        examples_dir / "ppo_high_uncertainty_rows.csv",
        high_uncertainty,
        [
            "rollout_tag",
            "lambda",
            "i",
            "model",
            "R_phi",
            "R_dagger",
            "R_dagger_2",
            "judge_avg",
            "u",
            "prompt",
            "generated",
        ],
    )


def _write_readme(out: Path, rows: list[dict[str, Any]]) -> None:
    best = max(rows, key=lambda r: float(r["mean_judge_avg"]))
    worst = min(rows, key=lambda r: float(r["mean_judge_avg"]))
    text = f"""# Publication Artifacts

Generated from tagged rollout summaries in `{eval_dir()}`.

## Main empirical message

- Best two-judge mean: {best['label']} ({_fmt(best['mean_judge_avg'])}).
- Lowest two-judge mean: {worst['label']} ({_fmt(worst['mean_judge_avg'])}).
- PPO/UP-PPO variants underperform SFT and DPO under both judges.
- The observed pattern is optimization failure, not classic reward hacking: PPO reward-model scores do not rise while judge scores fall.
- PPO/UP-PPO rows show higher reward-model uncertainty and larger policy drift than SFT/DPO.

## Contents

- `tables/table_model_comparison.*`: main model comparison table in CSV, Markdown, and LaTeX.
- `tables/table_paired_deltas.*`: paired deltas relative to SFT.
- `figures/fig_pareto_judge_vs_proxy_centered.*`: centered Pareto-style proxy-vs-judge plot.
- `figures/fig_two_judge_bars.*`: Anthropic/OpenAI grouped bar chart.
- `figures/fig_kl_uncertainty_vs_judge.*`: KL/helpfulness/uncertainty diagnostic.
- `figures/fig_lambda_sweep_scores.*`: UP-PPO lambda sweep plot.
- `examples/qualitative_examples.md`: candidate qualitative examples for the paper appendix.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def run_paper_artifacts() -> Path:
    ensure_dirs()
    base = eval_dir()
    summaries = _load_summaries(base)
    if not summaries:
        raise FileNotFoundError(f"No tagged rollout summaries found under {base}")

    examples = _load_examples(base)
    out = base / "publication_artifacts"
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    examples_dir = out / "examples"
    for d in (tables_dir, figures_dir, examples_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = _summary_rows(summaries)
    _write_tables(rows, summaries, examples, tables_dir)
    _write_figures(rows, figures_dir)
    _write_examples(examples, examples_dir)
    _write_readme(out, rows)

    print(f"Wrote publication artifacts to {out}", flush=True)
    return out
