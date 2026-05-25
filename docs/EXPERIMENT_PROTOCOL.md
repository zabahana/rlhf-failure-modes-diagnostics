# Experiment Protocol

## Objective

Build a mechanistic taxonomy of RLHF failure modes by tracking proxy reward, judge quality, uncertainty, policy drift, and generation behavior across training checkpoints and prompts.

## Methods To Run

- SFT reference policy
- Reward model trained on Anthropic HH-RLHF preferences
- PPO with no KL penalty
- PPO with low, medium, and high KL penalties
- Adaptive-KL PPO, if available
- UP-PPO with at least one moderate uncertainty penalty
- DPO baseline

## Checkpoints

Save and evaluate checkpoints at fixed intervals, for example:

| Step | Save | Evaluate |
| ---: | :--- | :------- |
| 200 | yes | yes |
| 400 | yes | yes |
| 600 | yes | yes |
| 800 | yes | yes |
| 1000 | yes | yes |
| 1200 | yes | yes |

## Evaluation Metrics

For each checkpoint and prompt row, collect:

- reward-model score `R_phi`
- primary judge score `R_dagger`
- secondary judge score `R_2_dagger`
- two-judge average `bar_R_dagger`
- absolute judge disagreement
- MC-dropout uncertainty `u`
- approximate KL to SFT
- response length
- repetition metrics
- lexical diversity metrics
- entropy, if available
- safety/toxicity score, if available

## Transition Classification

For consecutive checkpoints, classify each aggregate transition and row-level transition:

| Delta proxy | Delta judge | Class |
| ----------: | ----------: | ----- |
| positive | positive | stable_alignment |
| positive | negative | reward_hacking |
| negative | negative | optimization_collapse |
| negative | positive | proxy_under_alignment |
| near zero | near zero | conservative_stagnation |

Evaluator gaming is detected separately when one judge improves while the other judge declines.

## Early-Warning Analysis

For each row at checkpoint `t`, predict whether it enters reward hacking at checkpoint `t+1` using:

- KL drift
- uncertainty
- entropy
- repetition
- judge disagreement
- reward acceleration
- recent change in `R_phi`

A minimal model can use logistic regression or a threshold score.

## Output Artifacts

Expected generated outputs:

- `tables/checkpoint_failure_modes.csv`
- `tables/row_level_failure_modes.csv`
- `tables/early_warning_metrics.csv`
- `figures/trajectory_plot.pdf`
- `figures/failure_phase_diagram.pdf`
- `figures/prompt_failure_heatmap.pdf`
- `figures/early_warning_plot.pdf`
- `figures/evaluator_gaming_plot.pdf`
