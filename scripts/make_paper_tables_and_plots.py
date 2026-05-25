#!/usr/bin/env python
"""Create paper-ready tables and figures from failure-mode diagnostics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FAILURE_MODE_ORDER = [
    "stable_alignment",
    "reward_hacking",
    "optimization_collapse",
    "proxy_under_alignment",
    "conservative_stagnation",
    "mixed_or_ambiguous",
]

FAILURE_MODE_COLORS = {
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


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _ensure_failure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for mode in FAILURE_MODE_ORDER:
        if mode not in df.columns:
            df[mode] = 0
    return df[FAILURE_MODE_ORDER]


def build_tables(tables_dir: Path, paper_tables_dir: Path) -> dict[str, pd.DataFrame]:
    checkpoint = pd.read_csv(tables_dir / "checkpoint_failure_modes.csv")
    row_level = pd.read_csv(tables_dir / "row_level_failure_modes.csv")
    warnings = pd.read_csv(tables_dir / "early_warning_candidates.csv")

    checkpoint_counts = (
        checkpoint.groupby(["family", "model", "failure_mode"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    checkpoint_modes = _ensure_failure_columns(checkpoint_counts.set_index(["family", "model"])).reset_index()
    checkpoint_modes.insert(0, "setting", checkpoint_modes["family"].map(_short_family) + " / " + checkpoint_modes["model"])
    checkpoint_modes["total_transitions"] = checkpoint_modes[FAILURE_MODE_ORDER].sum(axis=1)
    checkpoint_modes["evaluator_gaming"] = (
        checkpoint.groupby(["family", "model"], dropna=False)["evaluator_gaming"].sum().astype(int).values
    )

    row_summary = (
        row_level.groupby(["family", "model", "failure_mode"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    row_summary = _ensure_failure_columns(row_summary.set_index(["family", "model"])).reset_index()
    row_summary.insert(0, "setting", row_summary["family"].map(_short_family) + " / " + row_summary["model"])
    row_summary["total_row_transitions"] = row_summary[FAILURE_MODE_ORDER].sum(axis=1)
    row_summary["reward_hacking_share"] = row_summary["reward_hacking"] / row_summary["total_row_transitions"]
    row_summary["evaluator_gaming_rows"] = (
        row_level.groupby(["family", "model"], dropna=False)["evaluator_gaming"].sum().astype(int).values
    )

    transition_table = checkpoint[
        [
            "family",
            "model",
            "from_step",
            "to_step",
            "delta_R_phi",
            "delta_R_dagger",
            "delta_R_dagger_2",
            "delta_kl",
            "delta_u",
            "failure_mode",
            "evaluator_gaming",
        ]
    ].copy()
    transition_table.insert(0, "setting", transition_table["family"].map(_short_family) + " / " + transition_table["model"])

    early_warning_top20 = warnings[
        [
            "family",
            "model",
            "i",
            "from_step",
            "to_step",
            "warning_score",
            "delta_R_phi",
            "delta_R_dagger",
            "delta_kl",
            "delta_u",
            "failure_mode",
            "evaluator_gaming",
        ]
    ].head(20).copy()
    early_warning_top20.insert(
        0, "setting", early_warning_top20["family"].map(_short_family) + " / " + early_warning_top20["model"]
    )

    outputs = {
        "paper_checkpoint_failure_mode_counts": checkpoint_modes,
        "paper_row_level_failure_mode_counts": row_summary,
        "paper_checkpoint_transition_table": transition_table,
        "paper_early_warning_top20": early_warning_top20,
    }

    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(paper_tables_dir / f"{name}.csv", index=False)
        df.to_latex(paper_tables_dir / f"{name}.tex", index=False, escape=False, float_format="%.3f")

    return outputs


def build_figures(tables_dir: Path, figures_dir: Path) -> None:
    checkpoint = pd.read_csv(tables_dir / "checkpoint_failure_modes.csv")
    row_level = pd.read_csv(tables_dir / "row_level_failure_modes.csv")
    warnings = pd.read_csv(tables_dir / "early_warning_candidates.csv")

    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_counts = (
        checkpoint.assign(setting=checkpoint["family"].map(_short_family) + " / " + checkpoint["model"])
        .groupby(["setting", "failure_mode"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    checkpoint_counts = _ensure_failure_columns(checkpoint_counts)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bottom = pd.Series(0, index=checkpoint_counts.index)
    for mode in FAILURE_MODE_ORDER:
        ax.barh(checkpoint_counts.index, checkpoint_counts[mode], left=bottom, color=FAILURE_MODE_COLORS[mode], label=mode)
        bottom += checkpoint_counts[mode]
    ax.set_xlabel("Number of checkpoint transitions")
    ax.set_ylabel("")
    ax.set_title("Checkpoint-Level Failure-Mode Taxonomy")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    _save_figure(fig, figures_dir, "fig_checkpoint_failure_modes")

    row_counts = (
        row_level.assign(setting=row_level["family"].map(_short_family) + " / " + row_level["model"])
        .groupby(["setting", "failure_mode"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    row_counts = _ensure_failure_columns(row_counts)
    row_share = row_counts.div(row_counts.sum(axis=1), axis=0).sort_values("reward_hacking", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bottom = pd.Series(0.0, index=row_share.index)
    for mode in FAILURE_MODE_ORDER:
        ax.barh(row_share.index, row_share[mode], left=bottom, color=FAILURE_MODE_COLORS[mode], label=mode)
        bottom += row_share[mode]
    ax.set_xlabel("Share of row-level transitions")
    ax.set_ylabel("")
    ax.set_title("Row-Level Failure Modes Reveal Localized Failures")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    _save_figure(fig, figures_dir, "fig_row_level_failure_mode_shares")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for mode in FAILURE_MODE_ORDER:
        subset = row_level[row_level["failure_mode"] == mode]
        if subset.empty:
            continue
        ax.scatter(
            subset["delta_R_phi"],
            subset["delta_R_dagger"],
            s=18,
            alpha=0.55,
            color=FAILURE_MODE_COLORS[mode],
            label=mode,
            edgecolors="none",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$\Delta R_\phi$")
    ax.set_ylabel(r"$\Delta R^\dagger$")
    ax.set_title("Proxy-Judge Directional Mismatch")
    ax.legend(fontsize=7, loc="best")
    _save_figure(fig, figures_dir, "fig_proxy_judge_delta_scatter")

    warning_counts = warnings["failure_mode"].value_counts().reindex(FAILURE_MODE_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(
        warning_counts.index,
        warning_counts.values,
        color=[FAILURE_MODE_COLORS[mode] for mode in warning_counts.index],
    )
    ax.set_ylabel("Top-100 warning cases")
    ax.set_title("Composition of Highest Warning-Score Cases")
    ax.tick_params(axis="x", rotation=35)
    _save_figure(fig, figures_dir, "fig_early_warning_composition")

    gaming_share = (
        checkpoint.assign(setting=checkpoint["family"].map(_short_family) + " / " + checkpoint["model"])
        .groupby("setting", dropna=False)["evaluator_gaming"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.barh(gaming_share.index, gaming_share.values, color="#17becf")
    ax.set_xlabel("Share of checkpoint transitions")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.set_title("Evaluator-Disagreement Events by Setting")
    _save_figure(fig, figures_dir, "fig_evaluator_gaming_share")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper tables and plots for RLHF failure-mode diagnostics.")
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--paper-tables-dir", type=Path, default=Path("tables/paper"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    build_tables(args.tables_dir, args.paper_tables_dir)
    build_figures(args.tables_dir, args.figures_dir)
    print(f"Wrote paper tables to {args.paper_tables_dir}")
    print(f"Wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
