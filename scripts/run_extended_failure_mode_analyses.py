#!/usr/bin/env python
"""Run extended analyses for the RLHF failure-mode paper.

The analyses are intentionally artifact-driven: they consume the CSVs produced
by build_failure_mode_tables.py and do not call judges or retrain models.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional analysis dependency
    XGBClassifier = None


FAILURE_MODE_ORDER = [
    "stable_alignment",
    "reward_hacking",
    "optimization_collapse",
    "proxy_under_alignment",
    "conservative_stagnation",
    "mixed_or_ambiguous",
]

COLORS = {
    "stable_alignment": "#2ca02c",
    "reward_hacking": "#d62728",
    "optimization_collapse": "#9467bd",
    "proxy_under_alignment": "#ff7f0e",
    "conservative_stagnation": "#7f7f7f",
    "mixed_or_ambiguous": "#1f77b4",
}


def _short_family(name: str) -> str:
    out = name.removeprefix("reward_hack_search_")
    out = re.sub(r"beta(\d+)p(\d+)", r"beta=\1.\2", out)
    out = re.sub(r"lambda(\d+)p(\d+)", r"lambda=\1.\2", out)
    out = out.replace("_sampled", " sampled")
    out = out.replace("_aggressive", " aggressive")
    out = out.replace("_up_", " UP ")
    out = out.replace("_long_2000", " long")
    return out


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    df.to_latex(out_dir / f"{name}.tex", index=False, escape=False, float_format="%.3f")


def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_inputs(tables_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "checkpoint": pd.read_csv(tables_dir / "checkpoint_failure_modes.csv"),
        "row": pd.read_csv(tables_dir / "row_level_failure_modes.csv"),
        "examples": pd.read_csv(tables_dir / "rollout_example_metrics.csv"),
    }


def early_warning_modeling(data: dict[str, pd.DataFrame], table_dir: Path, fig_dir: Path) -> dict[str, Any]:
    row = data["row"].copy()
    examples = data["examples"].copy()

    row["from_step"] = row["from_step"].astype(float)
    examples["step"] = examples["step"].astype(float)

    prev_features = examples[
        [
            "family",
            "model",
            "i",
            "step",
            "R_phi",
            "R_dagger",
            "R_dagger_2",
            "u",
            "mean_kl_sft",
            "judge_disagreement_abs",
            "generated_word_count",
            "distinct_1_ratio",
            "distinct_2_ratio",
            "max_3gram_repetition",
        ]
    ].rename(
        columns={
            "step": "from_step",
            "R_phi": "prev_R_phi",
            "R_dagger": "prev_R_dagger",
            "R_dagger_2": "prev_R_dagger_2",
            "u": "prev_u",
            "mean_kl_sft": "prev_mean_kl_sft",
            "judge_disagreement_abs": "prev_judge_disagreement_abs",
            "generated_word_count": "prev_generated_word_count",
            "distinct_1_ratio": "prev_distinct_1_ratio",
            "distinct_2_ratio": "prev_distinct_2_ratio",
            "max_3gram_repetition": "prev_max_3gram_repetition",
        }
    )

    model_df = row.merge(prev_features, on=["family", "model", "i", "from_step"], how="left")
    model_df["target_reward_hacking"] = (model_df["failure_mode"] == "reward_hacking").astype(int)

    feature_sets = {
        "pre_state_only": [
            "prev_R_phi",
            "prev_R_dagger",
            "prev_R_dagger_2",
            "prev_u",
            "prev_mean_kl_sft",
            "prev_judge_disagreement_abs",
            "prev_generated_word_count",
            "prev_distinct_1_ratio",
            "prev_distinct_2_ratio",
            "prev_max_3gram_repetition",
        ],
        "transition_diagnostics": [
            "delta_R_phi",
            "delta_R_dagger",
            "delta_R_dagger_2",
            "delta_kl",
            "delta_u",
            "to_generated_word_count",
            "to_distinct_1_ratio",
            "to_distinct_2_ratio",
            "to_max_3gram_repetition",
        ],
    }

    y = model_df["target_reward_hacking"]
    metrics_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for feature_set_name, features in feature_sets.items():
        X = model_df[features]
        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=7, stratify=stratify
        )

        estimators = {
            "logistic_regression": Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=7,
                        ),
                    ),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=7,
                        ),
                    ),
                ]
            ),
        }
        if XGBClassifier is not None:
            pos = int(y_train.sum())
            neg = int(len(y_train) - pos)
            estimators["xgboost"] = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=3,
                            learning_rate=0.03,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            min_child_weight=3,
                            reg_lambda=1.0,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            scale_pos_weight=neg / max(pos, 1),
                            random_state=7,
                            n_jobs=4,
                        ),
                    ),
                ]
            )

        for model_name, estimator in estimators.items():
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            score = estimator.predict_proba(X_test)[:, 1]
            metrics_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "prevalence_test": float(y_test.mean()),
                    "roc_auc": float(roc_auc_score(y_test, score)),
                    "average_precision": float(average_precision_score(y_test, score)),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "precision": float(precision_score(y_test, pred, zero_division=0)),
                    "recall": float(recall_score(y_test, pred, zero_division=0)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                }
            )
            if model_name == "logistic_regression":
                coefs = estimator.named_steps["clf"].coef_[0]
                for feature, coef in zip(features, coefs):
                    importance_rows.append(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "feature": feature,
                            "importance": float(coef),
                            "abs_importance": float(abs(coef)),
                        }
                    )
            else:
                importances = estimator.named_steps["clf"].feature_importances_
                for feature, importance in zip(features, importances):
                    importance_rows.append(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "feature": feature,
                            "importance": float(importance),
                            "abs_importance": float(abs(importance)),
                        }
                    )

    metrics = pd.DataFrame(metrics_rows)
    importances = pd.DataFrame(importance_rows).sort_values(
        ["feature_set", "model", "abs_importance"], ascending=[True, True, False]
    )
    _write_table(metrics, table_dir, "analysis_early_warning_model_metrics")
    _write_table(importances, table_dir, "analysis_early_warning_feature_importance")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plot_df = metrics.copy()
    labels = plot_df["feature_set"] + "\n" + plot_df["model"]
    x = np.arange(len(plot_df))
    ax.bar(x - 0.18, plot_df["roc_auc"], width=0.36, label="ROC-AUC", color="#1f77b4")
    ax.bar(x + 0.18, plot_df["average_precision"], width=0.36, label="Average precision", color="#ff7f0e")
    ax.axhline(plot_df["prevalence_test"].mean(), color="black", linestyle="--", linewidth=1, label="Mean prevalence")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Early-Warning Prediction of Row-Level Reward Hacking")
    ax.legend(fontsize=8)
    _save_fig(fig, fig_dir, "analysis_early_warning_model_scores")

    return {"n_rows": int(len(model_df)), "n_reward_hacking": int(y.sum())}


def prompt_localization(data: dict[str, pd.DataFrame], table_dir: Path, fig_dir: Path) -> dict[str, Any]:
    row = data["row"].copy()
    examples = data["examples"].copy()

    prompt_text = (
        examples.sort_values(["family", "model", "step"])
        .groupby("i", as_index=False)
        .agg(prompt=("prompt", "first"))
    )

    localized = (
        row.assign(is_reward_hacking=row["failure_mode"].eq("reward_hacking"))
        .groupby("i", as_index=False)
        .agg(
            reward_hacking_count=("is_reward_hacking", "sum"),
            total_transitions=("failure_mode", "size"),
            affected_families=("family", lambda x: x[row.loc[x.index, "failure_mode"].eq("reward_hacking")].nunique()),
            affected_models=("model", lambda x: x[row.loc[x.index, "failure_mode"].eq("reward_hacking")].nunique()),
            evaluator_gaming_count=("evaluator_gaming", "sum"),
        )
    )
    localized["reward_hacking_share"] = localized["reward_hacking_count"] / localized["total_transitions"]
    localized = localized.merge(prompt_text, on="i", how="left").sort_values(
        ["reward_hacking_count", "reward_hacking_share", "evaluator_gaming_count"], ascending=False
    )
    _write_table(localized.head(25), table_dir, "analysis_prompt_localization_top25")

    repeated = localized[localized["reward_hacking_count"] >= 2]

    top = localized.head(15).sort_values("reward_hacking_count")
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.barh(top["i"].astype(str), top["reward_hacking_count"], color="#d62728")
    ax.set_xlabel("Reward-hacking transitions")
    ax.set_ylabel("Prompt ID")
    ax.set_title("Prompt-Localized Reward Hacking")
    _save_fig(fig, fig_dir, "analysis_prompt_localization_top")

    return {
        "n_prompts": int(localized["i"].nunique()),
        "n_repeated_reward_hacking_prompts": int(len(repeated)),
        "max_reward_hacking_count": int(localized["reward_hacking_count"].max()),
    }


def mitigation_comparison(data: dict[str, pd.DataFrame], table_dir: Path, fig_dir: Path) -> dict[str, Any]:
    row = data["row"].copy()
    checkpoint = data["checkpoint"].copy()
    selected = [
        "reward_hack_search_beta0p0_aggressive",
        "reward_hack_search_beta0p0_aggressive_up_lambda0p1",
        "reward_hack_search_beta0p0_aggressive_up_lambda0p5",
    ]

    def summarize(df: pd.DataFrame, unit_name: str) -> pd.DataFrame:
        subset = df[df["family"].isin(selected)].copy()
        out = (
            subset.groupby(["family", "model"], as_index=False)
            .agg(
                total=("failure_mode", "size"),
                reward_hacking=("failure_mode", lambda x: int((x == "reward_hacking").sum())),
                stable_alignment=("failure_mode", lambda x: int((x == "stable_alignment").sum())),
                optimization_collapse=("failure_mode", lambda x: int((x == "optimization_collapse").sum())),
                proxy_under_alignment=("failure_mode", lambda x: int((x == "proxy_under_alignment").sum())),
                evaluator_gaming=("evaluator_gaming", "sum"),
            )
        )
        out.insert(0, "unit", unit_name)
        out.insert(1, "setting", out["family"].map(_short_family) + " / " + out["model"])
        out["reward_hacking_share"] = out["reward_hacking"] / out["total"]
        out["stable_alignment_share"] = out["stable_alignment"] / out["total"]
        out["evaluator_gaming_share"] = out["evaluator_gaming"] / out["total"]
        baseline_share = float(
            out.loc[out["family"].eq("reward_hack_search_beta0p0_aggressive"), "reward_hacking_share"].iloc[0]
        )
        out["reward_hacking_relative_change_vs_aggressive"] = (
            out["reward_hacking_share"] - baseline_share
        ) / baseline_share
        return out

    mitigation = pd.concat([summarize(row, "row_level"), summarize(checkpoint, "checkpoint")], ignore_index=True)
    _write_table(mitigation, table_dir, "analysis_mitigation_comparison")

    plot_df = mitigation[mitigation["unit"].eq("row_level")].copy()
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - 0.25, plot_df["reward_hacking_share"], width=0.25, label="Reward hacking", color="#d62728")
    ax.bar(x, plot_df["stable_alignment_share"], width=0.25, label="Stable alignment", color="#2ca02c")
    ax.bar(x + 0.25, plot_df["evaluator_gaming_share"], width=0.25, label="Evaluator gaming", color="#17becf")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["setting"], rotation=20, ha="right")
    ax.set_ylabel("Share of row-level transitions")
    ax.set_title("Aggressive PPO vs UP-PPO Failure-Mode Shares")
    ax.legend(fontsize=8)
    _save_fig(fig, fig_dir, "analysis_mitigation_comparison")

    return {
        "row_level_baseline_reward_hacking_share": float(
            plot_df.loc[plot_df["family"].eq("reward_hack_search_beta0p0_aggressive"), "reward_hacking_share"].iloc[0]
        )
    }


def judge_disagreement_analysis(data: dict[str, pd.DataFrame], table_dir: Path, fig_dir: Path) -> dict[str, Any]:
    row = data["row"].copy()
    checkpoint = data["checkpoint"].copy()

    row["judge_delta_gap"] = (row["delta_R_dagger"] - row["delta_R_dagger_2"]).abs()
    checkpoint["judge_delta_gap"] = (checkpoint["delta_R_dagger"] - checkpoint["delta_R_dagger_2"]).abs()

    rows = []
    for unit_name, df in [("row_level", row), ("checkpoint", checkpoint)]:
        grouped = (
            df.groupby("failure_mode", as_index=False)
            .agg(
                total=("failure_mode", "size"),
                evaluator_gaming=("evaluator_gaming", "sum"),
                mean_judge_delta_gap=("judge_delta_gap", "mean"),
                median_judge_delta_gap=("judge_delta_gap", "median"),
            )
        )
        grouped.insert(0, "unit", unit_name)
        grouped["evaluator_gaming_share"] = grouped["evaluator_gaming"] / grouped["total"]
        rows.append(grouped)
    disagreement = pd.concat(rows, ignore_index=True).sort_values(["unit", "evaluator_gaming_share"], ascending=False)
    _write_table(disagreement, table_dir, "analysis_judge_disagreement_by_failure_mode")

    top_gaming = row[row["evaluator_gaming"]].sort_values("judge_delta_gap", ascending=False).head(25)
    _write_table(top_gaming, table_dir, "analysis_judge_disagreement_top25_rows")

    plot_df = disagreement[disagreement["unit"].eq("row_level")].set_index("failure_mode").reindex(FAILURE_MODE_ORDER)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(
        plot_df.index,
        plot_df["evaluator_gaming_share"].fillna(0),
        color=[COLORS[mode] for mode in plot_df.index],
    )
    ax.set_ylabel("Evaluator-gaming share")
    ax.set_title("Judge Disagreement Concentrates by Failure Mode")
    ax.tick_params(axis="x", rotation=35)
    _save_fig(fig, fig_dir, "analysis_judge_disagreement_by_mode")

    return {
        "checkpoint_evaluator_gaming_share": float(checkpoint["evaluator_gaming"].mean()),
        "row_level_evaluator_gaming_share": float(row["evaluator_gaming"].mean()),
    }


def ablation_table(data: dict[str, pd.DataFrame], table_dir: Path, fig_dir: Path) -> dict[str, Any]:
    row = data["row"].copy()
    checkpoint = data["checkpoint"].copy()

    ck = (
        checkpoint.groupby(["family", "model"], as_index=False)
        .agg(
            checkpoint_total=("failure_mode", "size"),
            checkpoint_reward_hacking=("failure_mode", lambda x: int((x == "reward_hacking").sum())),
            checkpoint_evaluator_gaming=("evaluator_gaming", "sum"),
        )
    )
    ck["checkpoint_reward_hacking_share"] = ck["checkpoint_reward_hacking"] / ck["checkpoint_total"]

    rw = (
        row.groupby(["family", "model"], as_index=False)
        .agg(
            row_total=("failure_mode", "size"),
            row_reward_hacking=("failure_mode", lambda x: int((x == "reward_hacking").sum())),
            row_evaluator_gaming=("evaluator_gaming", "sum"),
        )
    )
    rw["row_reward_hacking_share"] = rw["row_reward_hacking"] / rw["row_total"]

    ablation = ck.merge(rw, on=["family", "model"], how="outer").fillna(0)
    ablation.insert(0, "setting", ablation["family"].map(_short_family) + " / " + ablation["model"])
    ablation["hidden_localized_reward_hacking"] = (
        (ablation["checkpoint_reward_hacking"] == 0) & (ablation["row_reward_hacking"] > 0)
    )
    _write_table(ablation, table_dir, "analysis_checkpoint_vs_row_ablation")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    plot_df = ablation.sort_values("row_reward_hacking_share", ascending=True)
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["checkpoint_reward_hacking_share"], height=0.36, label="Checkpoint share", color="#1f77b4")
    ax.barh(y + 0.18, plot_df["row_reward_hacking_share"], height=0.36, label="Row-level share", color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["setting"])
    ax.set_xlabel("Reward-hacking share")
    ax.set_title("Ablation: Aggregate Checkpoints vs Row-Level Diagnostics")
    ax.legend(fontsize=8)
    _save_fig(fig, fig_dir, "analysis_checkpoint_vs_row_ablation")

    return {
        "settings_with_hidden_localized_reward_hacking": int(ablation["hidden_localized_reward_hacking"].sum()),
        "total_settings": int(len(ablation)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five extended failure-mode analyses.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    data = _load_inputs(args.tables_dir)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "early_warning_modeling": early_warning_modeling(data, args.analysis_dir, args.figures_dir),
        "prompt_localization": prompt_localization(data, args.analysis_dir, args.figures_dir),
        "mitigation_comparison": mitigation_comparison(data, args.analysis_dir, args.figures_dir),
        "judge_disagreement": judge_disagreement_analysis(data, args.analysis_dir, args.figures_dir),
        "checkpoint_vs_row_ablation": ablation_table(data, args.analysis_dir, args.figures_dir),
    }
    (args.analysis_dir / "analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
