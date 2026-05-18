"""Stage 1 linear CMSSL-compatible model scaffolding.

This module intentionally contains only lightweight constant-prior utilities for
smoke-testing the linear pipeline against CMSSL's existing dataset/eval path.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from CMSSL17 import NUM_HORIZONS


LINEAR_CHECKPOINT_SCHEMA = "linear_taker_stage1_prior_v1"
LINEAR_MODEL_ARCH_SCHEMA = "linear_stage1_constant_prior_v1"
LINEAR_EXTRACTOR_SCHEMA = "linear_extractor_stage2_v1"
LINEAR_RAW_LINEAR_SCHEMA = "raw_linear_lag_bank_stats_v1"
LINEAR_AEON_EXTRACTOR_SCHEMA = "aeon_rocket_hydra_stage2_v1"


def side_log_mag_targets_np(
    y: np.ndarray,
    *,
    up_scale_bps: np.ndarray,
    down_scale_bps: np.ndarray,
    target_clip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != int(NUM_HORIZONS):
        raise ValueError(f"y must have shape [N, {NUM_HORIZONS}], got {y.shape}")
    up_scale = np.asarray(up_scale_bps, dtype=np.float32).reshape(1, -1)
    down_scale = np.asarray(down_scale_bps, dtype=np.float32).reshape(1, -1)
    if up_scale.shape[1] != int(NUM_HORIZONS) or down_scale.shape[1] != int(NUM_HORIZONS):
        raise ValueError("scale vectors must have NUM_HORIZONS entries")
    up = np.maximum(y, 0.0)
    down = np.maximum(-y, 0.0)
    up_log = np.log1p(up / np.maximum(up_scale, 1e-12))
    down_log = np.log1p(down / np.maximum(down_scale, 1e-12))
    if target_clip > 0.0:
        up_log = np.clip(up_log, 0.0, float(target_clip))
        down_log = np.clip(down_log, 0.0, float(target_clip))
    return up_log.astype(np.float32, copy=False), down_log.astype(np.float32, copy=False)


def inverse_side_log_mag_np(
    up_log: np.ndarray,
    down_log: np.ndarray,
    *,
    up_scale_bps: np.ndarray,
    down_scale_bps: np.ndarray,
    mag_floor_bps: float = 0.0,
    pred_log_clip: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    up_log = np.asarray(up_log, dtype=np.float32)
    down_log = np.asarray(down_log, dtype=np.float32)
    up_scale = np.asarray(up_scale_bps, dtype=np.float32).reshape(1, -1)
    down_scale = np.asarray(down_scale_bps, dtype=np.float32).reshape(1, -1)
    up_bps = np.maximum(np.expm1(np.clip(np.maximum(up_log, 0.0), 0.0, float(pred_log_clip))) * up_scale, float(mag_floor_bps))
    down_bps = np.maximum(np.expm1(np.clip(np.maximum(down_log, 0.0), 0.0, float(pred_log_clip))) * down_scale, float(mag_floor_bps))
    return up_bps.astype(np.float32, copy=False), down_bps.astype(np.float32, copy=False)


@dataclass
class LinearPreprocessBundle:
    schema: str
    config: Dict[str, Any]
    original_dim: int
    kept_dim: int
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    keep_mask: np.ndarray
    fit_summary: Dict[str, Any]

    def transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim != 2:
            raise ValueError(f"Z must have shape [N, D], got {Z.shape}")
        if Z.shape[1] != int(self.original_dim):
            raise ValueError(f"Z width {Z.shape[1]} != original_dim {self.original_dim}")

        nonfinite_policy = str(self.config.get("nonfinite_policy", "raise")).strip().lower()
        if nonfinite_policy == "raise":
            if not np.isfinite(Z).all():
                raise ValueError("Linear preprocessor input contains non-finite values")
        elif nonfinite_policy == "warn_zero":
            if not np.isfinite(Z).all():
                print("[linear-preprocess-warn] replacing non-finite input values with zero", flush=True)
                Z = np.where(np.isfinite(Z), Z, 0.0).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported nonfinite_policy {nonfinite_policy!r}")

        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        keep_mask = np.asarray(self.keep_mask, dtype=bool)
        for name, arr in (("lower", lower), ("upper", upper), ("mean", mean), ("std", std), ("keep_mask", keep_mask)):
            if arr.shape != (int(self.original_dim),):
                raise ValueError(f"{name} must have shape [{self.original_dim}], got {arr.shape}")
        if int(keep_mask.sum()) != int(self.kept_dim):
            raise ValueError(f"keep_mask sum {int(keep_mask.sum())} != kept_dim {self.kept_dim}")

        std_eps = float(self.config.get("std_eps", 1e-6))
        post_clip_abs = float(self.config.get("post_clip_abs", 0.0))
        Zc = np.minimum(np.maximum(Z, lower), upper)
        Zn = (Zc - mean) / np.maximum(std, std_eps)
        Zn = Zn[:, keep_mask]
        if post_clip_abs > 0.0:
            Zn = np.clip(Zn, -post_clip_abs, post_clip_abs)
        return Zn.astype(np.float32, copy=False)


@dataclass
class LinearSklearnTakerBundle:
    schema: str
    config: Dict[str, Any]
    horizons_ms: list[int]
    direction_models: list[Any]
    mag_up_models: list[Any]
    mag_down_models: list[Any]
    mag_floor: float
    fit_summary: Dict[str, Any]
    mag_mode: str = "side_all_log"
    mag_up_scale_bps: Optional[np.ndarray] = None
    mag_down_scale_bps: Optional[np.ndarray] = None

    def predict_dict_np(self, Z: np.ndarray) -> Dict[str, np.ndarray]:
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim != 2:
            raise ValueError(f"Z must have shape [N, D_preprocessed], got {Z.shape}")
        n_h = int(NUM_HORIZONS)
        if len(self.direction_models) != n_h or len(self.mag_up_models) != n_h or len(self.mag_down_models) != n_h:
            raise ValueError(
                "LinearSklearnTakerBundle must contain one direction/up/down model per horizon; "
                f"got {len(self.direction_models)}, {len(self.mag_up_models)}, {len(self.mag_down_models)}"
            )

        dir_cols = []
        up_cols = []
        down_cols = []
        for h in range(n_h):
            model = self.direction_models[h]
            if hasattr(model, "decision_function"):
                logit = model.decision_function(Z)
            elif hasattr(model, "predict_proba"):
                p = model.predict_proba(Z)[:, 1]
                logit = safe_logit_np(p)
            else:
                raise ValueError(f"Direction model for horizon index {h} has no decision_function or predict_proba")
            logit = np.asarray(logit, dtype=np.float32).reshape(-1)
            if logit.shape[0] != Z.shape[0]:
                raise ValueError(f"Direction model horizon {h} returned shape {logit.shape}, expected [{Z.shape[0]}]")
            dir_cols.append(logit)

            up = np.asarray(self.mag_up_models[h].predict(Z), dtype=np.float32).reshape(-1)
            down = np.asarray(self.mag_down_models[h].predict(Z), dtype=np.float32).reshape(-1)
            if up.shape[0] != Z.shape[0] or down.shape[0] != Z.shape[0]:
                raise ValueError(f"Magnitude model horizon {h} returned invalid row count")
            up_cols.append(up.astype(np.float32, copy=False))
            down_cols.append(down.astype(np.float32, copy=False))

        up_log = np.stack(up_cols, axis=1).astype(np.float32, copy=False)
        down_log = np.stack(down_cols, axis=1).astype(np.float32, copy=False)
        up_scale_raw = np.ones(n_h, dtype=np.float32) if self.mag_up_scale_bps is None else np.asarray(self.mag_up_scale_bps, dtype=np.float32)
        down_scale_raw = np.ones(n_h, dtype=np.float32) if self.mag_down_scale_bps is None else np.asarray(self.mag_down_scale_bps, dtype=np.float32)
        pred_log_clip = float(self.config.get("mag_log_pred_clip", 20.0))
        up_bps, down_bps = inverse_side_log_mag_np(
            up_log,
            down_log,
            up_scale_bps=up_scale_raw,
            down_scale_bps=down_scale_raw,
            mag_floor_bps=float(self.mag_floor),
            pred_log_clip=pred_log_clip,
        )
        return {
            "dir_logits": np.stack(dir_cols, axis=1).astype(np.float32, copy=False),
            "mag_up_log": up_log,
            "mag_down_log": down_log,
            "mag_up_bps": up_bps.astype(np.float32, copy=False),
            "mag_down_bps": down_bps.astype(np.float32, copy=False),
            "mag_up_sqrt": np.sqrt(np.maximum(up_bps, 0.0)).astype(np.float32, copy=False),
            "mag_down_sqrt": np.sqrt(np.maximum(down_bps, 0.0)).astype(np.float32, copy=False),
        }


class LinearSklearnTorchWrapper(nn.Module):
    def __init__(
        self,
        bundle: LinearSklearnTakerBundle,
        *,
        cmssl_schema_only: bool = False,
    ):
        super().__init__()
        self.bundle = bundle
        self.cmssl_schema_only = bool(cmssl_schema_only)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = x.device
        Z = x.detach().cpu().numpy().astype(np.float32, copy=False)
        pred = self.bundle.predict_dict_np(Z)

        if self.cmssl_schema_only:
            pred = {
                "dir_logits": pred["dir_logits"],
                "mag_up_sqrt": pred["mag_up_sqrt"],
                "mag_down_sqrt": pred["mag_down_sqrt"],
            }

        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in pred.items()
        }


def save_linear_sklearn_bundle(bundle: LinearSklearnTakerBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)


def load_linear_sklearn_bundle(path: Path) -> LinearSklearnTakerBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, LinearSklearnTakerBundle):
        raise ValueError(f"Expected LinearSklearnTakerBundle in {path}, got {type(obj).__name__}")
    return obj


def save_linear_preprocess_bundle(bundle: LinearPreprocessBundle, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.array(str(bundle.schema)),
        config_json=np.array(json.dumps(bundle.config, sort_keys=True)),
        original_dim=np.array([bundle.original_dim], dtype=np.int64),
        kept_dim=np.array([bundle.kept_dim], dtype=np.int64),
        lower=np.asarray(bundle.lower, dtype=np.float32),
        upper=np.asarray(bundle.upper, dtype=np.float32),
        mean=np.asarray(bundle.mean, dtype=np.float32),
        std=np.asarray(bundle.std, dtype=np.float32),
        keep_mask=np.asarray(bundle.keep_mask, dtype=bool),
        fit_summary_json=np.array(json.dumps(bundle.fit_summary, sort_keys=True, allow_nan=True)),
    )


def load_linear_preprocess_bundle(path: Path) -> LinearPreprocessBundle:
    with np.load(Path(path), allow_pickle=False) as arr:
        schema = str(arr["schema"].item())
        config = json.loads(str(arr["config_json"].item()))
        original_dim = int(np.asarray(arr["original_dim"]).reshape(-1)[0])
        kept_dim = int(np.asarray(arr["kept_dim"]).reshape(-1)[0])
        fit_summary = json.loads(str(arr["fit_summary_json"].item()))
        bundle = LinearPreprocessBundle(
            schema=schema,
            config=config,
            original_dim=original_dim,
            kept_dim=kept_dim,
            lower=np.asarray(arr["lower"], dtype=np.float32),
            upper=np.asarray(arr["upper"], dtype=np.float32),
            mean=np.asarray(arr["mean"], dtype=np.float32),
            std=np.asarray(arr["std"], dtype=np.float32),
            keep_mask=np.asarray(arr["keep_mask"], dtype=bool),
            fit_summary=fit_summary,
        )
    if bundle.lower.shape != (bundle.original_dim,) or bundle.keep_mask.shape != (bundle.original_dim,):
        raise ValueError(f"Invalid linear preprocess bundle shapes in {path}")
    if int(bundle.keep_mask.sum()) != bundle.kept_dim:
        raise ValueError(f"Invalid linear preprocess bundle kept_dim in {path}")
    return bundle


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
    keep_pos: Optional[np.ndarray] = None,
    keep_neg: Optional[np.ndarray] = None,
    keep_signed: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != NUM_HORIZONS:
        raise ValueError(f"y_train must have shape [N, {NUM_HORIZONS}], got {y.shape}")

    if keep_pos is None or keep_neg is None or keep_signed is None:
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
    else:
        keep_pos = np.asarray(keep_pos, dtype=bool)
        keep_neg = np.asarray(keep_neg, dtype=bool)
        keep_signed = np.asarray(keep_signed, dtype=bool)
        for name, arr in (
            ("keep_pos", keep_pos),
            ("keep_neg", keep_neg),
            ("keep_signed", keep_signed),
        ):
            if arr.shape != y.shape:
                raise ValueError(f"{name} must have shape {y.shape}, got {arr.shape}")
        if not np.array_equal(keep_signed, keep_pos | keep_neg):
            raise ValueError("keep_signed must equal keep_pos | keep_neg")

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


class LinearExtractorBase:
    name: str
    is_fitted: bool

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LinearExtractorBase":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)

    @property
    def output_dim(self) -> Optional[int]:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        raise NotImplementedError


def validate_window_array(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"{name} must have shape [N, T, F], got {X.shape}")
    if X.shape[0] <= 0:
        raise ValueError(f"{name} has zero rows")
    if X.shape[1] <= 0 or X.shape[2] <= 0:
        raise ValueError(f"{name} has invalid time/features shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains non-finite values before extraction")
    return X.astype(np.float32, copy=False)


def cmssl_windows_to_aeon(X: np.ndarray) -> np.ndarray:
    # CMSSL: [N, T, F]
    # aeon collection transformers generally consume [N, C, T]
    X = validate_window_array(X)
    return np.ascontiguousarray(np.transpose(X, (0, 2, 1)), dtype=np.float32)


def require_aeon_for_extractor(name: str) -> None:
    import importlib.util

    if importlib.util.find_spec("aeon") is None:
        raise ImportError(
            f"BYBIT_LINEAR_EXTRACTOR={name!r} requires aeon. "
            "Install aeon in the training environment, or use BYBIT_LINEAR_EXTRACTOR=raw_linear."
        )


class RawLinearExtractor(LinearExtractorBase):
    def __init__(
        self,
        mode: str = "lag_bank_stats",
        lags: Sequence[int] = (1, 2, 5, 10, 20, 50),
        windows: Sequence[int] = (5, 10, 20, 50),
        include_std: bool = True,
        include_slope: bool = False,
    ):
        mode = str(mode).strip().lower()
        if mode not in {"last", "lag_bank", "lag_bank_stats"}:
            raise ValueError(f"Unsupported raw linear mode {mode!r}; expected last, lag_bank, or lag_bank_stats")
        self.name = "raw_linear"
        self.mode = mode
        self.lags = tuple(int(x) for x in lags)
        self.windows = tuple(int(x) for x in windows)
        if any(lag <= 0 for lag in self.lags):
            raise ValueError(f"lags must be positive integers, got {self.lags}")
        if any(w <= 0 for w in self.windows):
            raise ValueError(f"windows must be positive integers, got {self.windows}")
        self.include_std = bool(include_std)
        self.include_slope = bool(include_slope)
        self.is_fitted = False
        self._output_dim: Optional[int] = None
        self._blocks: list[str] = []
        self._skipped_lags: list[int] = []
        self._skipped_windows: list[int] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RawLinearExtractor":
        validate_window_array(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = validate_window_array(X)
        _n, T, _f = X.shape
        blocks = [X[:, -1, :]]
        block_names = ["last"]
        skipped_lags: list[int] = []
        skipped_windows: list[int] = []

        if self.mode in {"lag_bank", "lag_bank_stats"}:
            for lag in self.lags:
                if 1 <= lag < T:
                    blocks.append(X[:, -1, :] - X[:, -1 - lag, :])
                    block_names.append(f"delta_lag_{lag}")
                else:
                    skipped_lags.append(int(lag))

        if self.mode == "lag_bank_stats":
            for w in self.windows:
                if 1 <= w <= T:
                    window = X[:, -w:, :]
                    blocks.append(window.mean(axis=1, dtype=np.float32))
                    block_names.append(f"mean_w_{w}")
                    if self.include_std:
                        blocks.append(window.std(axis=1, dtype=np.float32))
                        block_names.append(f"std_w_{w}")
                    if self.include_slope:
                        t = np.arange(w, dtype=np.float32)
                        t = t - t.mean()
                        den = np.sum(t * t)
                        slope = np.einsum("t,ntf->nf", t, window) / den if den > 0.0 else np.zeros_like(blocks[-1])
                        blocks.append(slope.astype(np.float32, copy=False))
                        block_names.append(f"slope_w_{w}")
                else:
                    skipped_windows.append(int(w))

        Z = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        self._output_dim = int(Z.shape[1])
        self._blocks = block_names
        self._skipped_lags = skipped_lags
        self._skipped_windows = skipped_windows
        self.is_fitted = True
        return Z

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim

    def summary(self) -> Dict[str, Any]:
        return {
            "name": "raw_linear",
            "schema": LINEAR_RAW_LINEAR_SCHEMA,
            "mode": self.mode,
            "lags": [int(x) for x in self.lags],
            "windows": [int(x) for x in self.windows],
            "include_std": bool(self.include_std),
            "include_slope": bool(self.include_slope),
            "output_dim": None if self._output_dim is None else int(self._output_dim),
            "blocks": list(self._blocks),
            "skipped_lags": [int(x) for x in self._skipped_lags],
            "skipped_windows": [int(x) for x in self._skipped_windows],
        }


class CombinedLinearExtractor(LinearExtractorBase):
    def __init__(self, name: str, extractors: Sequence[LinearExtractorBase]):
        if not extractors:
            raise ValueError("CombinedLinearExtractor requires at least one child extractor")
        self.name = str(name)
        self.extractors = list(extractors)
        self.is_fitted = False
        self._output_dim: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CombinedLinearExtractor":
        validate_window_array(X)
        for child in self.extractors:
            child.fit(X, y=y)
        self.is_fitted = True
        self._output_dim = sum(int(child.output_dim or 0) for child in self.extractors) or None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        validate_window_array(X)
        Z = np.concatenate([child.transform(X) for child in self.extractors], axis=1).astype(np.float32, copy=False)
        self._output_dim = int(Z.shape[1])
        return Z

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "schema": LINEAR_AEON_EXTRACTOR_SCHEMA,
            "output_dim": None if self._output_dim is None else int(self._output_dim),
            "children": [child.summary() for child in self.extractors],
        }


class AeonRocketExtractor(LinearExtractorBase):
    ALLOWED_NAMES = {"minirocket", "multirocket", "hydra", "multirocket_hydra"}

    def __init__(
        self,
        name: str,
        n_kernels: int = 10000,
        n_groups: int = 64,
        hydra_n_kernels: int = 8,
        n_jobs: int = 1,
        random_state: int = 17,
    ):
        name = str(name).strip().lower()
        if name not in self.ALLOWED_NAMES:
            raise ValueError(f"Unsupported aeon extractor {name!r}; expected one of {sorted(self.ALLOWED_NAMES)}")
        require_aeon_for_extractor(name)
        self.name = name
        self.n_kernels = int(n_kernels)
        self.n_groups = int(n_groups)
        self.hydra_n_kernels = int(hydra_n_kernels)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.is_fitted = False
        self._output_dim: Optional[int] = None
        self.transformer: Optional[Any] = None

    def _new_minirocket_multivariate(self) -> Any:
        import importlib
        import inspect

        module = importlib.import_module("aeon.transformations.collection.convolution_based")
        if not hasattr(module, "MiniRocketMultivariate"):
            raise ImportError("MiniRocket multivariate transformer unavailable in this aeon version")

        cls = module.MiniRocketMultivariate
        sig = inspect.signature(cls)
        kwargs: Dict[str, Any] = {
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }
        if "n_kernels" in sig.parameters:
            kwargs["n_kernels"] = self.n_kernels
        elif "num_kernels" in sig.parameters:
            kwargs["num_kernels"] = self.n_kernels
        else:
            raise TypeError("MiniRocketMultivariate has no n_kernels or num_kernels parameter")
        return cls(**kwargs)

    def _new_transformer(self, class_name: Optional[str] = None) -> Any:
        import importlib

        module = importlib.import_module("aeon.transformations.collection.convolution_based")
        name = class_name or self.name
        if name == "minirocket":
            return module.MiniRocket(n_kernels=self.n_kernels, n_jobs=self.n_jobs, random_state=self.random_state)
        if name == "MiniRocketMultivariate":
            return self._new_minirocket_multivariate()
        if name == "multirocket":
            return module.MultiRocket(n_kernels=self.n_kernels, n_jobs=self.n_jobs, random_state=self.random_state)
        if name == "hydra":
            return module.HydraTransformer(
                n_kernels=self.hydra_n_kernels,
                n_groups=self.n_groups,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                output_type="numpy",
            )
        raise ValueError(f"Cannot build transformer for {name!r}")

    def _build_transformer(self, X_probe: Optional[np.ndarray] = None) -> Any:
        if self.name == "multirocket_hydra":
            return CombinedLinearExtractor(
                "multirocket_hydra",
                [
                    AeonRocketExtractor(
                        "multirocket",
                        n_kernels=self.n_kernels,
                        n_groups=self.n_groups,
                        hydra_n_kernels=self.hydra_n_kernels,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    ),
                    AeonRocketExtractor(
                        "hydra",
                        n_kernels=self.n_kernels,
                        n_groups=self.n_groups,
                        hydra_n_kernels=self.hydra_n_kernels,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                    ),
                ],
            )
        if self.name != "minirocket":
            return self._new_transformer()

        if X_probe is None:
            return self._new_transformer()
        try:
            probe_transformer = self._new_transformer("minirocket")
            probe_transformer.fit(X_probe)
            return self._new_transformer("minirocket")
        except Exception as mini_exc:
            import importlib

            module = importlib.import_module("aeon.transformations.collection.convolution_based")
            if not hasattr(module, "MiniRocketMultivariate"):
                raise ImportError("MiniRocket multivariate transformer unavailable in this aeon version") from mini_exc
            return self._new_transformer("MiniRocketMultivariate")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AeonRocketExtractor":
        X = validate_window_array(X)
        if self.name == "multirocket_hydra":
            self.transformer = self._build_transformer()
            self.transformer.fit(X, y=y)
            self.is_fitted = True
            self._output_dim = self.transformer.output_dim
            return self
        X_aeon = cmssl_windows_to_aeon(X)
        probe = X_aeon[: min(2, X_aeon.shape[0])]
        self.transformer = self._build_transformer(probe)
        self.transformer.fit(X_aeon)
        self.is_fitted = True
        self._output_dim = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = validate_window_array(X)
        if self.transformer is None:
            raise RuntimeError(f"Extractor {self.name!r} must be fit before transform")
        if isinstance(self.transformer, CombinedLinearExtractor):
            Z = self.transformer.transform(X)
        else:
            Z = self.transformer.transform(cmssl_windows_to_aeon(X))
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim != 2:
            raise ValueError(f"Aeon extractor {self.name!r} produced non-2D output shape {Z.shape}")
        if not np.isfinite(Z).all():
            raise ValueError(f"Aeon extractor {self.name!r} produced non-finite values")
        self._output_dim = int(Z.shape[1])
        return Z

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim

    def summary(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "schema": LINEAR_AEON_EXTRACTOR_SCHEMA,
            "n_kernels": int(self.n_kernels),
            "hydra_n_kernels": int(self.hydra_n_kernels),
            "n_groups": int(self.n_groups),
            "n_jobs": int(self.n_jobs),
            "random_state": int(self.random_state),
            "output_dim": None if self._output_dim is None else int(self._output_dim),
            "transformer_class": None if self.transformer is None else self.transformer.__class__.__name__,
        }
        if isinstance(self.transformer, CombinedLinearExtractor):
            payload["combined"] = self.transformer.summary()
        return payload


def build_linear_extractor_from_config(config: Dict[str, Any]) -> LinearExtractorBase:
    name = str(config["extractor"]).strip().lower()
    if name == "raw_linear":
        return RawLinearExtractor(
            mode=str(config.get("raw_mode", "lag_bank_stats")),
            lags=config.get("raw_lags", (1, 2, 5, 10, 20, 50)),
            windows=config.get("raw_windows", (5, 10, 20, 50)),
            include_std=bool(config.get("raw_include_std", True)),
            include_slope=bool(config.get("raw_include_slope", False)),
        )
    if name in AeonRocketExtractor.ALLOWED_NAMES:
        return AeonRocketExtractor(
            name=name,
            n_kernels=int(config.get("n_kernels", 10000)),
            hydra_n_kernels=int(config.get("hydra_n_kernels", 8)),
            n_groups=int(config.get("n_groups", 64)),
            n_jobs=int(config.get("n_jobs", 1)),
            random_state=int(config.get("random_state", 17)),
        )
    raise ValueError(f"Unsupported linear extractor {name!r}")
