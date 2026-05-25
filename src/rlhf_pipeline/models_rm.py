"""Causal LM with a scalar value head for reward and MC-dropout uncertainty."""
from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class CausalScalarRewardModel(nn.Module):
    """
    Scores a full string (prompt + response) with a single scalar;
    use last non-padding position for the reward logit.
    """

    def __init__(self, name_or_path: str, dropout: float = 0.1, torch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        kw = {}
        if torch_dtype is not None:
            kw["torch_dtype"] = torch_dtype
        self.core = AutoModelForCausalLM.from_pretrained(name_or_path, **kw)
        hidden = self.core.config.n_embd if hasattr(self.core.config, "n_embd") else self.core.config.hidden_size
        self.head = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        for m in [self.head]:
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def _last_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.core(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        h = out.hidden_states[-1]  # [B, T, D]
        idx = attention_mask.sum(dim=1) - 1
        b = torch.arange(h.size(0), device=h.device, dtype=torch.long)
        return h[b, idx, :]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last = self._last_hidden(input_ids, attention_mask)
        last = self.drop(last)
        return self.head(last).squeeze(-1)  # [B]

    @torch.inference_mode()
    def score_texts(
        self,
        tok: AutoTokenizer,
        texts: List[str],
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        return self(enc["input_ids"], enc["attention_mask"])

    @torch.inference_mode()
    def score_texts_with_dropout_std(
        self,
        tok: AutoTokenizer,
        texts: List[str],
        max_length: int,
        device: torch.device,
        k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean reward (K passes in train mode) and std across passes as u."""
        scores = []
        was_training = self.training
        self.train()  # enable dropout
        for _ in range(k):
            scores.append(
                self.score_texts(tok, texts, max_length, device).detach()
            )
        if not was_training:
            self.eval()
        st = torch.stack(scores, dim=0)  # [K, B]
        return st.mean(dim=0), st.std(dim=0)


@contextlib.contextmanager
def torch_dtype_for_model(device: torch.device):
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", enabled=True):
            yield
    else:
        yield
