#!/usr/bin/env python
"""Classify checkpoint or row-level RLHF transitions from a CSV file.

Input CSV columns:
- delta_R_phi
- delta_R_dagger
Optional:
- delta_R_dagger_2
- delta_kl
- delta_u
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rlhf_pipeline.failure_modes import TransitionDeltas, classify_transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    rows = []
    for _, row in df.iterrows():
        result = classify_transition(
            TransitionDeltas(
                delta_r_phi=float(row["delta_R_phi"]),
                delta_judge=float(row["delta_R_dagger"]),
                delta_judge_2=None if "delta_R_dagger_2" not in row or pd.isna(row.get("delta_R_dagger_2")) else float(row["delta_R_dagger_2"]),
                delta_kl=None if "delta_kl" not in row or pd.isna(row.get("delta_kl")) else float(row["delta_kl"]),
                delta_uncertainty=None if "delta_u" not in row or pd.isna(row.get("delta_u")) else float(row["delta_u"]),
                epsilon=args.epsilon,
            )
        )
        rows.append(result)
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
