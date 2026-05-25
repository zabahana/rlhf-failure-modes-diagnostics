# When RLHF Fails: Reward Hacking, Optimization Collapse, and Evaluator Gaming

This repository is a separate research project built from the existing compact RLHF pipeline. It studies RLHF failure modes as training dynamics rather than only final-checkpoint performance.

## Research Goal

The project asks whether RLHF failures can be separated into measurable regimes by tracking learned reward, external judge quality, reward uncertainty, KL drift, and generation behavior over training time.

## Core Failure Modes

- Stable alignment: proxy reward improves and external quality improves.
- Reward hacking: proxy reward improves while external quality declines.
- Optimization collapse: proxy reward and external quality both decline.
- Proxy under-alignment: proxy reward declines while external quality improves.
- Evaluator gaming: one external judge improves while another degrades.
- Conservative stagnation: reward, judge quality, and policy drift remain mostly flat.

## Repository Structure

```text
paper/      Main manuscript source for the new study
arxiv/      arXiv-ready source skeleton
src/        Reused RLHF pipeline plus failure-mode analysis utilities
configs/    Experiment configuration templates
scripts/    Convenience scripts for analysis/table generation
web_ui/     Local Gradio app for comparing model responses
docs/       Experimental protocol and artifact notes
tables/     Generated result tables for the new study
figures/    Generated figures for the new study
examples/   Curated qualitative examples for the new study
```

## Reused Pipeline Components

The copied pipeline supports SFT, reward-model training, PPO, DPO, UP-PPO, rollout evaluation, external judges, reward uncertainty, and KL diagnostics. New utilities in `src/rlhf_pipeline/failure_modes.py` classify checkpoint transitions and prompt-level rows into the taxonomy used by the paper.

## Minimal Study

A first empirical version should run PPO with multiple KL settings, UP-PPO, and DPO; evaluate every checkpoint with two judges; compute uncertainty, KL, repetition, entropy, and lexical diversity; then classify failures at checkpoint and prompt level.

See `docs/EXPERIMENT_PROTOCOL.md` for the planned run sequence.

## Web UI

Run a local response-comparison app with:

```bash
pip install -r requirements.txt
PYTHONPATH=src python web_ui/app.py
```

The app compares PPO, DPO, and UP-PPO responses from the existing trained checkpoints and can optionally score each response with the trained reward model. See `docs/WEB_UI.md` for paths and options.
