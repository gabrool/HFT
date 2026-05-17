"""Stage 1 linear CMSSL-compatible model scaffolding.

This module intentionally contains only lightweight constant-prior utilities for
smoke-testing the linear pipeline against CMSSL's existing dataset/eval path.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from CMSSL17 import NUM_HORIZONS


LINEAR_CHECKPOINT_SCHEMA = "linear_taker_stage1_prior_v1"
LINEAR_MODEL_ARCH_SCHEMA = "linear_stage1_constant_prior_v1"


def safe_logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)


class LinearConstantPriorModel(nn.Module):
    """CMSSL-output-compatible model that emits train-label constant priors."""

    def __init__(
        self,
        dir_logit_prior: np.ndarray,
        mag_up_sqrt_prior: np.ndarray,
        mag_down_sqrt_prior: np.ndarray,
    ):
        super().__init__()
        dir_logit = self._validate_prior("dir_logit_prior", dir_logit_prior)
        mag_up = np.maximum(self._validate_prior("mag_up_sqrt_prior", mag_up_sqrt_prior), 1e-4)
        mag_down = np.maximum(self._validate_prior("mag_down_sqrt_prior", mag_down_sqrt_prior), 1e-4)

        self.register_buffer("dir_logit_prior", torch.as_tensor(dir_logit, dtype=torch.float32))
        self.register_buffer("mag_up_sqrt_prior", torch.as_tensor(mag_up, dtype=torch.float32))
        self.register_buffer("mag_down_sqrt_prior", torch.as_tensor(mag_down, dtype=torch.float32))

    @staticmethod
    def _validate_prior(name: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (NUM_HORIZONS,):
            raise ValueError(f"{name} must have shape [{NUM_HORIZONS}], got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim < 1:
            raise ValueError(f"x must include a batch dimension, got shape={tuple(x.shape)}")
        batch_size = int(x.shape[0])
        dir_logits = self.dir_logit_prior.to(device=x.device).view(1, NUM_HORIZONS).expand(batch_size, -1)
        mag_up_sqrt = self.mag_up_sqrt_prior.to(device=x.device).clamp_min(1e-4).view(1, NUM_HORIZONS).expand(batch_size, -1)
        mag_down_sqrt = self.mag_down_sqrt_prior.to(device=x.device).clamp_min(1e-4).view(1, NUM_HORIZONS).expand(batch_size, -1)
        return {
            "dir_logits": dir_logits,
            "mag_up_sqrt": mag_up_sqrt,
            "mag_down_sqrt": mag_down_sqrt,
        }


def build_constant_priors_from_train_labels(
    y_train: np.ndarray,
    stats: Dict[str, np.ndarray],
    mag_up_sqrt_prior: np.ndarray,
    mag_down_sqrt_prior: np.ndarray,
) -> Dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != NUM_HORIZONS:
        raise ValueError(f"y_train must have shape [N, {NUM_HORIZONS}], got {y.shape}")

    pos_lo = np.asarray(stats["pos_lo_raw_bps"], dtype=np.float32).reshape(1, -1)
    pos_hi = np.asarray(stats["pos_hi_raw_bps"], dtype=np.float32).reshape(1, -1)
    neg_lo = np.asarray(stats["neg_lo_abs_bps"], dtype=np.float32).reshape(1, -1)
    neg_hi = np.asarray(stats["neg_hi_abs_bps"], dtype=np.float32).reshape(1, -1)
    for name, arr in (
        ("pos_lo_raw_bps", pos_lo),
        ("pos_hi_raw_bps", pos_hi),
        ("neg_lo_abs_bps", neg_lo),
        ("neg_hi_abs_bps", neg_hi),
    ):
        if arr.shape != (1, NUM_HORIZONS):
            raise ValueError(f"stats[{name!r}] must have shape [{NUM_HORIZONS}], got {arr.reshape(-1).shape}")

    pos = y > 0.0
    neg = y < 0.0
    neg_mag = (-y).clip(min=0.0)
    keep_pos = pos & (y >= pos_lo) & (y <= pos_hi)
    keep_neg = neg & (neg_mag >= neg_lo) & (neg_mag <= neg_hi)
    keep_signed = keep_pos | keep_neg

    signed_counts = keep_signed.sum(axis=0).astype(np.float64)
    if np.any(signed_counts <= 0.0):
        bad = [int(i) for i, c in enumerate(signed_counts) if c <= 0.0]
        raise ValueError(f"Cannot build direction priors; zero kept signed train rows for horizons indices={bad}")

    p_up = (keep_pos.sum(axis=0).astype(np.float64) / signed_counts).astype(np.float32)
    dir_logit_prior = safe_logit_np(p_up)
    mag_up = LinearConstantPriorModel._validate_prior("mag_up_sqrt_prior", mag_up_sqrt_prior)
    mag_down = LinearConstantPriorModel._validate_prior("mag_down_sqrt_prior", mag_down_sqrt_prior)

    return {
        "dir_logit_prior": dir_logit_prior,
        "mag_up_sqrt_prior": mag_up,
        "mag_down_sqrt_prior": mag_down,
        "p_up_prior": p_up,
    }


def linear_model_summary(model: LinearConstantPriorModel) -> Dict[str, Any]:
    dir_logit = model.dir_logit_prior.detach().float().cpu().numpy().astype(np.float32)
    mag_up = model.mag_up_sqrt_prior.detach().float().cpu().numpy().astype(np.float32)
    mag_down = model.mag_down_sqrt_prior.detach().float().cpu().numpy().astype(np.float32)
    dir_prob = 1.0 / (1.0 + np.exp(-dir_logit.astype(np.float64)))
    return {
        "type": "constant_prior",
        "num_horizons": NUM_HORIZONS,
        "dir_logit_prior": [float(x) for x in dir_logit],
        "dir_prob_prior": [float(x) for x in dir_prob],
        "mag_up_sqrt_prior": [float(x) for x in mag_up],
        "mag_down_sqrt_prior": [float(x) for x in mag_down],
    }
