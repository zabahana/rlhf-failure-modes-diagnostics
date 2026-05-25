# RLHF Web UI

This project includes a local Gradio app for comparing responses from the trained SFT, PPO, DPO, and UP-PPO checkpoints.

## Run

From the project root:

```bash
pip install -r requirements.txt
PYTHONPATH=src python web_ui/app.py
```

Then open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.

If your Python environment cannot write to the normal package directory, install Gradio into the project-local dependency folder:

```bash
python -m pip install --target .deps gradio
PYTHONPATH=src python web_ui/app.py
```

## Default Model Paths

By default, the app loads artifacts from:

```text
/Users/zelalemabahana/reward-hacking-RLHF/code/artifacts_proposal_min
```

The default model checkpoints are:

```text
SFT:    sft/model
PPO:    ppo/reward_hack_search_beta0p0_aggressive/policy_step_1200
DPO:    dpo/model
UP-PPO: ppo/reward_hack_search_beta0p0_aggressive_up_lambda0p1/policy_step_600
```

To point the app to a different artifact directory:

```bash
PYTHONPATH=src python web_ui/app.py --artifacts /path/to/artifacts
```

or:

```bash
export RLHF_ARTIFACTS=/path/to/artifacts
PYTHONPATH=src python web_ui/app.py
```

## Features

- Generate side-by-side responses from PPO, DPO, and UP-PPO.
- Include SFT as a reference model when desired.
- Adjust max tokens, sampling, temperature, top-p, and seed.
- Optionally score responses with the trained reward model and MC-dropout uncertainty.

## Notes

The UI loads each model lazily the first time it is selected. On CPU this can take a little time, especially when scoring with the reward model.
