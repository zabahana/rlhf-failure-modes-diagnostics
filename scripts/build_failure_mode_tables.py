#!/usr/bin/env python
"""Build failure-mode diagnostic tables from existing RLHF rollout artifacts.

This script is artifact-driven: it does not call external judges or retrain
policies. It consolidates rollout summaries and examples, computes checkpoint
and row-level deltas, and applies the failure-mode taxonomy used by the new
paper scaffold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rlhf_pipeline.failure_modes import TransitionDeltas, classify_transition


DEFAULT_SOURCE = Path("/Users/zelalemabahana/reward-hacking-RLHF/code/artifacts_proposal_min/eval")
DEFAULT_OUT = Path("/Users/zelalemabahana/rlhf-failure-modes-diagnostics/tables")


@dataclass(frozen=True)
class ArtifactIdentity:
    tag: str
    family: str
    step: int | None
    stage: str


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _distinct_ngram_ratio(text: str, n: int) -> float:
    words = _tokens(text)
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return _safe_div(len(set(grams)), len(grams))


def _max_ngram_repetition(text: str, n: int) -> int:
    words = _tokens(text)
    if len(words) < n:
        return 0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return max(Counter(grams).values(), default=0)


def _identity_from_tag(tag: str) -> ArtifactIdentity:
    step_match = re.search(r"_policy_step_(\d+)$", tag)
    step = int(step_match.group(1)) if step_match else None
    stage = "checkpoint" if step is not None else "aggregate_or_final"

    if step_match:
        family = tag[: step_match.start()]
    elif tag.endswith("_final_two_judges"):
        family = tag[: -len("_final_two_judges")]
        step = 2000 if "long_2000" in tag else None
        stage = "final_two_judges"
    elif tag.endswith("_final_openai"):
        family = tag[: -len("_final_openai")]
        step = 2000 if "long_2000" in tag else None
        stage = "final_openai"
    else:
        family = tag

    return ArtifactIdentity(tag=tag, family=family, step=step, stage=stage)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summary_rows(source: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("rollout_summary_*.json")):
        payload = _read_json(path)
        tag = str(payload.get("rollout_tag") or path.stem.removeprefix("rollout_summary_"))
        identity = _identity_from_tag(tag)
        by_model = payload.get("by_model", {})
        for model, metrics in by_model.items():
            rows.append(
                {
                    "tag": tag,
                    "family": identity.family,
                    "step": identity.step,
                    "stage": identity.stage,
                    "model": model,
                    "judge": payload.get("judge"),
                    "n_examples": payload.get("n_examples"),
                    "summary_file": str(path),
                    "ppo_policy_relative": payload.get("ppo_policy_relative"),
                    "mean_R_phi": _as_float(metrics.get("mean_R_phi")),
                    "mean_R_dagger": _as_float(metrics.get("mean_R_dagger")),
                    "mean_R_dagger_2": _as_float(metrics.get("mean_R_dagger_2")),
                    "mean_u": _as_float(metrics.get("mean_u")),
                    "mean_kl_sft": _as_float(metrics.get("mean_kl_sft")),
                    "mean_judge_disagreement_abs": _as_float(metrics.get("mean_judge_disagreement_abs")),
                    "spearman_rho_u_abs_err": _as_float(metrics.get("spearman_rho_u_abs_err")),
                    "Delta_hack_proxy": _as_float(metrics.get("Delta_hack_proxy")),
                    "Delta_hack_proxy_judge2": _as_float(metrics.get("Delta_hack_proxy_judge2")),
                    "n": metrics.get("n"),
                }
            )
    return pd.DataFrame(rows)


def _example_rows(source: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("rollout_examples_*.jsonl")):
        tag = path.stem.removeprefix("rollout_examples_")
        identity = _identity_from_tag(tag)
        for row in _read_jsonl(path):
            generated = str(row.get("generated") or "")
            rows.append(
                {
                    "tag": tag,
                    "family": identity.family,
                    "step": identity.step,
                    "stage": identity.stage,
                    "model": row.get("model"),
                    "i": row.get("i"),
                    "prompt": row.get("prompt"),
                    "generated": generated,
                    "R_phi": _as_float(row.get("R_phi")),
                    "R_dagger": _as_float(row.get("R_dagger")),
                    "R_dagger_2": _as_float(row.get("R_dagger_2")),
                    "u": _as_float(row.get("u")),
                    "mean_kl_sft": _as_float(row.get("mean_kl_sft")),
                    "judge_disagreement_abs": _as_float(row.get("judge_disagreement_abs")),
                    "generated_word_count": len(_tokens(generated)),
                    "distinct_1_ratio": _distinct_ngram_ratio(generated, 1),
                    "distinct_2_ratio": _distinct_ngram_ratio(generated, 2),
                    "max_3gram_repetition": _max_ngram_repetition(generated, 3),
                    "examples_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def _checkpoint_transitions(metrics: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checkpoint_metrics = metrics.dropna(subset=["step"]).copy()
    for (family, model), group in checkpoint_metrics.groupby(["family", "model"], dropna=False):
        group = group.sort_values("step")
        records = group.to_dict("records")
        for prev, cur in zip(records, records[1:]):
            delta_r_phi = float(cur["mean_R_phi"] - prev["mean_R_phi"])
            delta_judge = float(cur["mean_R_dagger"] - prev["mean_R_dagger"])
            prev_j2 = _as_float(prev.get("mean_R_dagger_2"))
            cur_j2 = _as_float(cur.get("mean_R_dagger_2"))
            delta_j2 = None if prev_j2 is None or cur_j2 is None else float(cur_j2 - prev_j2)
            prev_kl = _as_float(prev.get("mean_kl_sft"))
            cur_kl = _as_float(cur.get("mean_kl_sft"))
            delta_kl = None if prev_kl is None or cur_kl is None else float(cur_kl - prev_kl)
            prev_u = _as_float(prev.get("mean_u"))
            cur_u = _as_float(cur.get("mean_u"))
            delta_u = None if prev_u is None or cur_u is None else float(cur_u - prev_u)
            label = classify_transition(
                TransitionDeltas(
                    delta_r_phi=delta_r_phi,
                    delta_judge=delta_judge,
                    delta_judge_2=delta_j2,
                    delta_kl=delta_kl,
                    delta_uncertainty=delta_u,
                    epsilon=epsilon,
                )
            )
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "from_tag": prev["tag"],
                    "to_tag": cur["tag"],
                    "from_step": int(prev["step"]),
                    "to_step": int(cur["step"]),
                    "delta_R_phi": delta_r_phi,
                    "delta_R_dagger": delta_judge,
                    "delta_R_dagger_2": delta_j2,
                    "delta_kl": delta_kl,
                    "delta_u": delta_u,
                    **label,
                }
            )
    return pd.DataFrame(rows)


def _row_level_transitions(examples: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checkpoint_examples = examples.dropna(subset=["step"]).copy()
    for (family, model), group in checkpoint_examples.groupby(["family", "model"], dropna=False):
        step_values = sorted(group["step"].dropna().unique())
        for prev_step, cur_step in zip(step_values, step_values[1:]):
            prev = group[group["step"] == prev_step].set_index("i")
            cur = group[group["step"] == cur_step].set_index("i")
            for i in sorted(prev.index.intersection(cur.index)):
                p = prev.loc[i]
                c = cur.loc[i]
                delta_r_phi = float(c["R_phi"] - p["R_phi"])
                delta_judge = float(c["R_dagger"] - p["R_dagger"])
                p_j2 = _as_float(p.get("R_dagger_2"))
                c_j2 = _as_float(c.get("R_dagger_2"))
                delta_j2 = None if p_j2 is None or c_j2 is None else float(c_j2 - p_j2)
                p_kl = _as_float(p.get("mean_kl_sft"))
                c_kl = _as_float(c.get("mean_kl_sft"))
                delta_kl = None if p_kl is None or c_kl is None else float(c_kl - p_kl)
                p_u = _as_float(p.get("u"))
                c_u = _as_float(c.get("u"))
                delta_u = None if p_u is None or c_u is None else float(c_u - p_u)
                label = classify_transition(
                    TransitionDeltas(
                        delta_r_phi=delta_r_phi,
                        delta_judge=delta_judge,
                        delta_judge_2=delta_j2,
                        delta_kl=delta_kl,
                        delta_uncertainty=delta_u,
                        epsilon=epsilon,
                    )
                )
                rows.append(
                    {
                        "family": family,
                        "model": model,
                        "i": i,
                        "from_step": int(prev_step),
                        "to_step": int(cur_step),
                        "delta_R_phi": delta_r_phi,
                        "delta_R_dagger": delta_judge,
                        "delta_R_dagger_2": delta_j2,
                        "delta_kl": delta_kl,
                        "delta_u": delta_u,
                        "to_generated_word_count": c["generated_word_count"],
                        "to_distinct_1_ratio": c["distinct_1_ratio"],
                        "to_distinct_2_ratio": c["distinct_2_ratio"],
                        "to_max_3gram_repetition": c["max_3gram_repetition"],
                        **label,
                    }
                )
    return pd.DataFrame(rows)


def _failure_mode_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    grouped = transitions.groupby(["family", "failure_mode"], dropna=False).size().reset_index(name="n_transitions")
    totals = transitions.groupby("family", dropna=False).size().rename("family_total")
    out = grouped.merge(totals, on="family")
    out["share"] = out["n_transitions"] / out["family_total"]
    return out.sort_values(["family", "n_transitions"], ascending=[True, False])


def _early_warning_candidates(row_transitions: pd.DataFrame) -> pd.DataFrame:
    if row_transitions.empty:
        return pd.DataFrame()
    out = row_transitions.copy()
    out["warning_score"] = (
        out["delta_u"].fillna(0).clip(lower=0)
        + out["delta_kl"].fillna(0).clip(lower=0)
        + out["delta_R_phi"].fillna(0).clip(lower=0)
        + (-out["delta_R_dagger"].fillna(0)).clip(lower=0)
    )
    return out.sort_values("warning_score", ascending=False).head(100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"source artifact directory not found: {args.source}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _summary_rows(args.source)
    examples = _example_rows(args.source)
    transitions = _checkpoint_transitions(metrics, args.epsilon)
    row_transitions = _row_level_transitions(examples, args.epsilon)
    summary = _failure_mode_summary(transitions)
    warnings = _early_warning_candidates(row_transitions)

    metrics.to_csv(args.out_dir / "checkpoint_metrics.csv", index=False)
    examples.to_csv(args.out_dir / "rollout_example_metrics.csv", index=False)
    transitions.to_csv(args.out_dir / "checkpoint_failure_modes.csv", index=False)
    row_transitions.to_csv(args.out_dir / "row_level_failure_modes.csv", index=False)
    summary.to_csv(args.out_dir / "failure_mode_summary.csv", index=False)
    warnings.to_csv(args.out_dir / "early_warning_candidates.csv", index=False)

    manifest = {
        "source": str(args.source),
        "out_dir": str(args.out_dir),
        "n_checkpoint_rows": int(len(metrics)),
        "n_example_rows": int(len(examples)),
        "n_checkpoint_transitions": int(len(transitions)),
        "n_row_level_transitions": int(len(row_transitions)),
        "outputs": [
            "checkpoint_metrics.csv",
            "rollout_example_metrics.csv",
            "checkpoint_failure_modes.csv",
            "row_level_failure_modes.csv",
            "failure_mode_summary.csv",
            "early_warning_candidates.csv",
        ],
    }
    (args.out_dir / "failure_mode_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
