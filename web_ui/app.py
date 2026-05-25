#!/usr/bin/env python
"""Local Gradio UI for comparing SFT, PPO, DPO, and UP-PPO responses.

Run from the repository root:

    PYTHONPATH=src python web_ui/app.py

The default artifact root points to the existing trained models from the reward-hacking study.
Override with RLHF_ARTIFACTS or --artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.is_dir() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise SystemExit(
        "Gradio is not installed. Run: pip install gradio\n"
        "Then start the app with: PYTHONPATH=src python web_ui/app.py"
    ) from exc

from rlhf_pipeline.models_rm import CausalScalarRewardModel
from rlhf_pipeline.rm_temperature import apply_temperature_scalar

DEFAULT_ARTIFACTS = Path(
    os.environ.get(
        "RLHF_ARTIFACTS",
        "/Users/zelalemabahana/reward-hacking-RLHF/code/artifacts_proposal_min",
    )
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    relative_path: str
    description: str


MODEL_SPECS = {
    "SFT": ModelSpec("SFT", "sft/model", "Supervised fine-tuned reference policy."),
    "PPO": ModelSpec(
        "PPO",
        "ppo/reward_hack_search_beta0p0_aggressive/policy_step_1200",
        "Aggressive no-KL PPO reward-hacking checkpoint.",
    ),
    "DPO": ModelSpec("DPO", "dpo/model", "Direct Preference Optimization baseline."),
    "UP-PPO": ModelSpec(
        "UP-PPO",
        "ppo/reward_hack_search_beta0p0_aggressive_up_lambda0p1/policy_step_600",
        "Uncertainty-penalized PPO, lambda=0.1, strongest judged checkpoint.",
    ),
}


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelCache:
    def __init__(self, artifacts: Path, device: torch.device) -> None:
        self.artifacts = artifacts.expanduser().resolve()
        self.device = device
        self.models: dict[str, tuple[Any, Any]] = {}
        self.reward_model: tuple[CausalScalarRewardModel, Any, float] | None = None

    def model_path(self, label: str) -> Path:
        return self.artifacts / MODEL_SPECS[label].relative_path

    def load_lm(self, label: str) -> tuple[Any, Any]:
        if label in self.models:
            return self.models[label]
        path = self.model_path(label)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint at {path}")
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        model.to(self.device)
        model.eval()
        self.models[label] = (model, tok)
        return model, tok

    def load_reward_model(self) -> tuple[CausalScalarRewardModel, Any, float]:
        if self.reward_model is not None:
            return self.reward_model
        rm_dir = self.artifacts / "rm"
        ckpt = rm_dir / "reward_model.pt"
        meta_path = rm_dir / "meta.json"
        tok_path = rm_dir / "tok"
        if not ckpt.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing reward model artifacts under {rm_dir}")
        meta = json.loads(meta_path.read_text())
        rm = CausalScalarRewardModel(meta["load_path"], dropout=float(meta.get("dropout", 0.1)))
        rm.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        rm.to(self.device)
        rm.eval()
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        temperature = 1.0
        cal = rm_dir / "calibration.json"
        if cal.is_file():
            temperature = float(json.loads(cal.read_text()).get("temperature", 1.0) or 1.0)
        self.reward_model = (rm, tok, temperature)
        return self.reward_model


def generate_response(
    model: Any,
    tok: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: int,
) -> str:
    torch.manual_seed(int(seed) % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) % (2**32))
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else 1.0,
            top_p=float(top_p) if do_sample else 1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_ids = out[0, enc["input_ids"].shape[1] :]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def format_result(label: str, generated: str, score: dict[str, float] | None) -> str:
    lines = [f"### {label}", generated or "_(empty generation)_"]
    if score is not None:
        lines.append("")
        lines.append(
            f"Reward model: `{score['R_phi']:.3f}` | calibrated: `{score['R_phi_calibrated']:.3f}` | uncertainty: `{score['u']:.3f}`"
        )
    return "\n".join(lines)


def make_app(cache: ModelCache) -> gr.Blocks:
    def run(
        prompt: str,
        selected_models: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int,
        score_with_rm: bool,
    ) -> tuple[str, str, str, str, str]:
        if not prompt.strip():
            return "Please enter a prompt.", "", "", "", ""
        if not selected_models:
            return "Select at least one model.", "", "", "", ""
        outputs: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for label in selected_models:
            model, tok = cache.load_lm(label)
            outputs[label] = generate_response(
                model,
                tok,
                prompt,
                cache.device,
                max_new_tokens,
                temperature,
                top_p,
                do_sample,
                seed,
            )
        if score_with_rm:
            rm, rm_tok, temp = cache.load_reward_model()
            texts = [prompt + outputs[label] for label in selected_models]
            with torch.no_grad():
                means, stds = rm.score_texts_with_dropout_std(
                    rm_tok, texts, max_length=512, device=cache.device, k=4
                )
            for label, mean, std in zip(selected_models, means.tolist(), stds.tolist()):
                scores[label] = {
                    "R_phi": float(mean),
                    "R_phi_calibrated": apply_temperature_scalar(float(mean), temp),
                    "u": float(std),
                }
        status = f"Loaded artifacts from `{cache.artifacts}` on `{cache.device}`."
        rendered = {
            label: format_result(label, outputs[label], scores.get(label))
            for label in selected_models
        }
        return (
            status,
            rendered.get("SFT", ""),
            rendered.get("PPO", ""),
            rendered.get("DPO", ""),
            rendered.get("UP-PPO", ""),
        )

    with gr.Blocks(title="RLHF Model Response Comparator") as demo:
        gr.Markdown(
            "# RLHF Response Comparator\n"
            "Generate side-by-side responses from SFT, PPO, DPO, and UP-PPO checkpoints. "
            "Use this to inspect reward hacking, DPO behavior, and uncertainty-penalized PPO outputs."
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt / dialogue context",
                lines=10,
                value="\n\nHuman: How can I improve my study habits?\n\nAssistant:",
            )
            with gr.Column():
                selected = gr.CheckboxGroup(
                    label="Models",
                    choices=list(MODEL_SPECS.keys()),
                    value=["PPO", "DPO", "UP-PPO"],
                )
                max_new = gr.Slider(16, 192, value=96, step=8, label="Max new tokens")
                temp = gr.Slider(0.1, 1.5, value=0.75, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                sample = gr.Checkbox(value=True, label="Sample")
                seed = gr.Number(value=42, precision=0, label="Seed")
                score = gr.Checkbox(value=True, label="Score with reward model + MC uncertainty")
                btn = gr.Button("Generate", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            sft_out = gr.Markdown(label="SFT")
            ppo_out = gr.Markdown(label="PPO")
        with gr.Row():
            dpo_out = gr.Markdown(label="DPO")
            up_out = gr.Markdown(label="UP-PPO")
        btn.click(
            run,
            inputs=[prompt, selected, max_new, temp, top_p, sample, seed, score],
            outputs=[status, sft_out, ppo_out, dpo_out, up_out],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cache = ModelCache(args.artifacts, default_device())
    app = make_app(cache)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
