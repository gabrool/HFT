import json
import os
import time
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import numba
    HAS_NUMBA = True
except Exception:
    numba = None
    HAS_NUMBA = False
import torch
import torch.nn as nn
import torch.optim as optim

# Configure CUDA allocator only when this script is run directly, so
# importing RL_exec as a module does not mutate global environment state.
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from CMSSL17 import (
    SAMBA,
    ModelArgs,
    DMODEL,
    MAMBA_LAYERS,
    LOOKBACK,
)
from offline_tokens import iter_week_chunks, load_global_meta

# Reference snapshot cadence in milliseconds (used for runtime scaling only).
RAW_SNAPSHOT_EXPECTED_STEP_MS = 100
RAW_SNAPSHOT_FEATURE_COLUMNS = [
    "best_bid",
    "best_ask",
    "best_bid_size",
    "best_ask_size",
    "time_since_last_ob_update_ms",
    "imbalance",
    "mid",
    "spread_bps",
    "mid_ret_1",
    "vol_short",
    "vol_long",
]
FEATURE_EXTRA_DIM = 5
ENV_OBS_EXTRA_STATE_DIM = 14
SHORT_VOL_WINDOW = 50
LONG_VOL_WINDOW = 200
# CMSSL market-making horizon contract is fixed: exactly [250, 500, 1000] ms.
DEFAULT_MM_HORIZONS_MS = [250, 500, 1000]
DEFAULT_MM_BASE_HALF_SPREAD_BPS = 0.0
DEFAULT_MM_ALPHA_CENTER_SCALE = 1.0
DEFAULT_MM_INVENTORY_CENTER_SCALE = 0.0
DEFAULT_MM_VOL_WIDTH_SCALE = 1.0
DEFAULT_MM_UNCERTAINTY_WIDTH_SCALE = 0.0
DEFAULT_MM_INVENTORY_SIDE_WIDEN_SCALE = 0.0
DEFAULT_MM_OBS_SPREAD_ANCHOR_FRAC = 0.5
DEFAULT_MM_SPREAD_FLOOR_BPS = 0.0
DEFAULT_MM_SPREAD_CAP_BPS = 10_000.0
DEFAULT_MM_INV_REF_NOTIONAL = 1.0
DEFAULT_MM_P250_WEIGHT = 0.0
DEFAULT_MM_P500_WEIGHT = 0.0
DEFAULT_MM_P1000_WEIGHT = 1.0
# PPO training epochs environment variable (used across entrypoint/config helpers).
PPO_EPOCHS_ENV = "BYBIT_MM_PPO_EPOCHS"
# Scaling factors for market-making observation extra-state features.
# Inventory notional and cash scales are denominated in quote currency.
# Time-since-fill scale is in environment steps (1 step per snapshot).
DEFAULT_MM_INVENTORY_NOTIONAL_SCALE = 1e4
DEFAULT_MM_CASH_SCALE = 1e4
DEFAULT_MM_TIME_SINCE_FILL_SCALE = 1000.0
DEFAULT_MM_FILL_NOTIONAL_SCALE = 1e4
DEFAULT_MM_PNL_NOTIONAL_SCALE = 1e4
DEFAULT_MM_MARKOUT_NOTIONAL_SCALE = 1e4
DEFAULT_MM_FILL_EMA_WINDOW_STEPS = 3
DEFAULT_MM_INITIAL_CASH = 1_000_000.0
DEFAULT_MM_TAKER_FEE_BPS = 1.7
DEFAULT_MM_TAKER_THRESHOLD = 0.25
# Inventory risk thresholds are denominated in quote notional (USD).

def require(condition: bool, msg: str, exc_type: type[Exception] = ValueError) -> None:
    """Raise a typed exception when a runtime precondition fails."""
    if not condition:
        raise exc_type(msg)


def _empty_obs_norm_state() -> Dict[str, Any]:
    return {"count": 0, "mean": None, "m2": None, "continuous_mask": None}


def _obs_norm_state_is_ready(state: Dict[str, Any]) -> bool:
    if not isinstance(state, dict):
        return False
    try:
        count = int(state.get("count", 0))
    except (TypeError, ValueError):
        return False
    return count >= 2 and state.get("mean") is not None and state.get("m2") is not None


def _require_event_time_decision_meta(meta: Dict[str, Any]) -> None:
    contract_error = (
        "Dataset is missing event-time decision metadata. "
        "Regenerate canonical snapshots and tokens so decisions use "
        "order-book event timestamps (decision_time_basis='ob_event_time')."
    )
    if meta.get("decision_time_basis") != "ob_event_time":
        raise ValueError(contract_error)
    decision_policy = meta.get("decision_policy")
    if decision_policy is not None and decision_policy != "ob_event_time":
        raise ValueError(contract_error)


def load_cmssl(out_root: str, ckpt_path: str, device: str = "cuda"):
    t0 = time.perf_counter()
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    _require_event_time_decision_meta(meta)
    feat_dim = int(meta["feature_dim_total"])  # includes AUX_DIM already

    args = ModelArgs(DMODEL, MAMBA_LAYERS, feat_dim, LOOKBACK)
    model = SAMBA(args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    require(isinstance(state, dict), "CMSSL checkpoint state_dict must be a mapping")

    raw_keys = {
        key[7:] if isinstance(key, str) and key.startswith("module.") else key
        for key in state.keys()
    }
    legacy_prefixes = ("return_head.", "volatility_head.")
    require(
        not any(any(key.startswith(prefix) for prefix in legacy_prefixes) for key in raw_keys),
        "Legacy three-head CMSSL checkpoints are incompatible with the direction-only runtime; retrain/export a direction-only checkpoint."
    )
    if isinstance(ckpt, dict):
        ckpt_args = ckpt.get("args")
        if isinstance(ckpt_args, dict) and ckpt_args.get("checkpoint_schema") not in (None, "cmssl17-direction-only-v1"):
            require(
                False,
                f"Unsupported CMSSL checkpoint schema {ckpt_args.get('checkpoint_schema')!r}; expected 'cmssl17-direction-only-v1'."
            )

    model_state = model.state_dict()
    missing_model_keys = [k for k in model_state.keys() if k not in raw_keys]
    unexpected_model_keys = [k for k in raw_keys if k not in model_state]
    require(
        not missing_model_keys,
        "CMSSL checkpoint missing model keys: " + ", ".join(missing_model_keys[:10]) + (" ..." if len(missing_model_keys) > 10 else "")
    )
    require(
        not unexpected_model_keys,
        "CMSSL checkpoint has unexpected model keys: " + ", ".join(unexpected_model_keys[:10]) + (" ..." if len(unexpected_model_keys) > 10 else "")
    )

    normalized_state = {
        key[7:] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(normalized_state, strict=True)

    loaded_keys = set(normalized_state.keys())
    required_prefixes = (
        "depatch_proj_encoder.",
        "mamba.",
        "direction_head.",
    )
    missing_components = [
        prefix for prefix in required_prefixes
        if not any(k.startswith(prefix) for k in loaded_keys)
    ]
    require(
        not missing_components,
        "CMSSL checkpoint is incompatible; required components not loaded: "
        + ", ".join(missing_components),
    )
    model.eval()
    model = _maybe_compile_module(
        model,
        enabled=_env_bool("BYBIT_MM_COMPILE_CMSSL", False),
        label="cmssl",
    )
    _timing_log(f"load_cmssl secs={time.perf_counter() - t0:.4f}")
    return model, meta


@torch.inference_mode()
def cmssl_predict(model, x_core, x_aux, meta, device: str = "cuda"):
    # x_core: [B, L, F_core]  x_aux: [B, L, AUX_DIM]
    x_core = torch.as_tensor(x_core, device=device)
    x_aux = torch.as_tensor(x_aux, device=device)
    x = torch.cat([x_core, x_aux], dim=-1)
    dir_logits = model(x)
    require(torch.is_tensor(dir_logits), "CMSSL model(x) must return a tensor of direction logits")
    horizons = meta.get("horizons_ms", [])
    expected_h = len(horizons)
    require(expected_h > 0, "meta['horizons_ms'] must be non-empty")
    expected_shape = (x.shape[0], expected_h)
    require(
        tuple(dir_logits.shape) == expected_shape,
        f"CMSSL model(x) must return shape {expected_shape}; got {tuple(dir_logits.shape)}",
    )
    return dir_logits


def iter_chunk_batches(out_root: str):
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    for week, week_meta, week_dir in iter_week_chunks(out_root, meta=meta):
        for entry in week_meta.get("chunks", []):
            files = entry.get("files", {})
            x_core = np.load(week_dir / files["core"])
            x_aux = np.load(week_dir / files["aux"])
            y = np.load(week_dir / files["y"])
            ts_path = files.get("ts")
            ts = np.load(week_dir / ts_path) if ts_path else None
            yield week, int(entry.get("chunk", 0)), ts, x_core, x_aux, y



def bps_to_px(mid: float, bps: float) -> float:
    return mid * bps * 1e-4


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int_list(name: str, default: List[int]) -> List[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [int(item) for item in raw.split(",") if item.strip()]


def _resolve_cmssl_batch_size(default: int = 2048) -> int:
    return _env_int("BYBIT_MM_CMSSL_BATCH_SIZE", default)


def _resolve_rollout_storage(default: str = "gpu") -> str:
    storage = os.environ.get("BYBIT_MM_ROLLOUT_STORAGE", default).strip().lower()
    allowed = {"gpu", "cpu"}
    if storage not in allowed:
        raise ValueError(
            f"Invalid BYBIT_MM_ROLLOUT_STORAGE='{storage}'. Allowed values: {sorted(allowed)}"
        )
    return storage


def _configure_tf32_from_env() -> bool:
    enabled = _env_bool("BYBIT_MM_ENABLE_TF32", False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
    return enabled


def _timing_enabled() -> bool:
    return _env_bool("BYBIT_MM_ENABLE_TIMING", False)


def _timing_log(message: str) -> None:
    if _timing_enabled():
        print(f"[timing] {message}")


LOG_2PI = float(np.log(2.0 * np.pi))
_SQUASH_EPS = 1e-6


def _diag_gaussian_sample(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn(
        mean.shape,
        generator=generator,
        device=mean.device,
        dtype=mean.dtype,
    )
    action = mean + std * eps
    return action, eps


def _diag_gaussian_logprob(x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    z = (x - mean) / torch.exp(log_std)
    return (-0.5 * (z * z + 2.0 * log_std + LOG_2PI)).sum(dim=-1)


def _diag_gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return (log_std + 0.5 * (1.0 + LOG_2PI)).sum(dim=-1)


def _resolve_market_action_dim(allow_taker: bool) -> int:
    return 3 if bool(allow_taker) else 2


def _ppo_action_bounds(
    env: Optional["MarketMakingEnv"],
    action_dim: int,
    device: torch.device | str,
    delta_scale: float,
    taker_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    low = torch.full((action_dim,), -1.0, device=device, dtype=torch.float32)
    high = torch.full((action_dim,), 1.0, device=device, dtype=torch.float32)
    if action_dim >= 1:
        delta_limit = abs(float(delta_scale))
        if env is not None:
            delta_limit = min(delta_limit, float(env.delta_bps_limit))
        low[0] = -delta_limit
        high[0] = delta_limit
    if action_dim >= 2:
        low[1] = low[0]
        high[1] = high[0]
    if action_dim >= 3:
        taker_limit = abs(float(taker_scale)) if env is None or env.allow_taker else 0.0
        low[2] = -taker_limit
        high[2] = taker_limit
    return low, high


def _squashed_gaussian_log_prob(
    latent_action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    base_log_prob = _diag_gaussian_logprob(latent_action, mean, log_std)
    squashed = torch.tanh(latent_action)
    half_range = 0.5 * (action_high - action_low)
    log_det = torch.log(half_range.clamp_min(_SQUASH_EPS)) + torch.log1p(
        -(squashed * squashed).clamp(max=1.0 - _SQUASH_EPS)
    )
    return base_log_prob - log_det.sum(dim=-1)


def _sample_bounded_ppo_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    std = log_std.exp()
    latent_action, _eps = _diag_gaussian_sample(mean, std, generator=generator)
    action_env = _postprocess_bounded_env_action(latent_action, action_low, action_high)
    logp = _squashed_gaussian_log_prob(latent_action, mean, log_std, action_low, action_high)
    return action_env, logp, latent_action


def _postprocess_bounded_env_action(
    latent_action: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    squashed_action = torch.tanh(latent_action)
    action_mid = 0.5 * (action_high + action_low)
    action_half_range = 0.5 * (action_high - action_low)
    return action_mid + action_half_range * squashed_action


def _bounded_ppo_mean_action(
    mean: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    return _postprocess_bounded_env_action(mean, action_low, action_high)


def _bounded_mean_action_penalty(
    mean: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    power: float = 2.0,
) -> torch.Tensor:
    bounded_mean = _bounded_ppo_mean_action(mean, action_low, action_high)
    bound_mag = torch.maximum(action_low.abs(), action_high.abs())
    normalized_mag = bounded_mean / torch.clamp(bound_mag, min=1e-12)
    return (normalized_mag.abs() ** power).mean()


def _bounded_ppo_latent_action(
    action_env: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    action_mid = 0.5 * (action_high + action_low)
    action_half_range = 0.5 * (action_high - action_low)
    squashed = (action_env - action_mid) / action_half_range.clamp_min(_SQUASH_EPS)
    squashed = squashed.clamp(min=-1.0 + _SQUASH_EPS, max=1.0 - _SQUASH_EPS)
    return torch.atanh(squashed)


def _maybe_compile_module(module: torch.nn.Module, *, enabled: bool, label: str) -> torch.nn.Module:
    if not enabled:
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        warnings.warn(
            f"torch.compile unavailable; skipping compile for {label}.",
            RuntimeWarning,
        )
        return module
    mode = os.environ.get("BYBIT_MM_COMPILE_MODE", "reduce-overhead")
    try:
        return compile_fn(module, mode=mode, fullgraph=False)
    except Exception as exc:
        warnings.warn(
            f"torch.compile failed for {label}: {exc}. Falling back to eager module.",
            RuntimeWarning,
        )
        return module


def _resolve_run_mode(default: str = "train") -> str:
    """Resolve run mode: train, eval, train_eval, or baseline-only evaluation."""
    accepted_modes = {"baseline", "train", "eval", "train_eval"}
    mode = os.environ.get("BYBIT_MM_RUN_MODE", default).strip().lower()
    if mode not in accepted_modes:
        accepted = ", ".join(sorted(accepted_modes))
        raise ValueError(f"Invalid BYBIT_MM_RUN_MODE='{mode}'. Accepted values: {accepted}")
    return mode


def _resolve_baseline_eval_split(default: str = "val") -> str:
    accepted_splits = {"val", "test"}
    split_name = os.environ.get("BYBIT_MM_BASELINE_EVAL_SPLIT", default).strip().lower()
    if split_name not in accepted_splits:
        accepted = ", ".join(sorted(accepted_splits))
        raise ValueError(
            f"Invalid BYBIT_MM_BASELINE_EVAL_SPLIT='{split_name}'. Accepted values: {accepted}"
        )
    return split_name


def _select_baseline_env(
    split_name: str,
    mm_val_env: "MarketMakingEnv",
    mm_test_env: "MarketMakingEnv",
) -> Tuple[str, "MarketMakingEnv"]:
    if split_name == "val":
        return "val", mm_val_env
    if split_name == "test":
        return "test", mm_test_env
    raise ValueError(f"Unsupported baseline split '{split_name}'")


def resolve_market_env_common_kwargs_from_env() -> Dict[str, Any]:
    maker_rebate_bps = float(os.environ.get("BYBIT_MM_MAKER_REBATE_BPS", "0.0"))
    inventory_penalty = float(os.environ.get("BYBIT_MM_INVENTORY_PENALTY", "0.0"))
    inv_soft_notional_str = os.environ.get("BYBIT_MM_INV_SOFT_NOTIONAL", "").strip()
    if not inv_soft_notional_str:
        raise ValueError(
            "Missing required env var BYBIT_MM_INV_SOFT_NOTIONAL (quote notional, USD)."
        )
    inv_soft_notional = float(inv_soft_notional_str)
    lambda_inv = float(os.environ.get("BYBIT_MM_LAMBDA_INV", "0.0"))
    lambda_turn = float(os.environ.get("BYBIT_MM_LAMBDA_TURN", "0.0"))
    max_inventory_notional_str = os.environ.get("BYBIT_MM_MAX_INV_NOTIONAL", "").strip()
    if not max_inventory_notional_str:
        raise ValueError(
            "Missing required env var BYBIT_MM_MAX_INV_NOTIONAL (quote notional, USD)."
        )
    max_inventory_notional = float(max_inventory_notional_str)
    hard_max_inventory_notional_str = os.environ.get("BYBIT_MM_HARD_MAX_INV_NOTIONAL", "").strip()
    hard_max_inventory_notional = (
        float(hard_max_inventory_notional_str)
        if hard_max_inventory_notional_str
        else float(max_inventory_notional)
    )
    fill_size = float(os.environ.get("BYBIT_MM_FILL_SIZE", "1.0"))
    fill_tolerance = float(os.environ.get("BYBIT_MM_FILL_TOLERANCE", "1e-6"))
    taker_fee_bps = float(os.environ.get("BYBIT_MM_TAKER_FEE_BPS", str(DEFAULT_MM_TAKER_FEE_BPS)))
    taker_threshold = float(os.environ.get("BYBIT_MM_TAKER_THRESHOLD", str(DEFAULT_MM_TAKER_THRESHOLD)))
    delta_bps_limit_str = os.environ.get("BYBIT_MM_DELTA_BPS_LIMIT", "").strip()
    if not delta_bps_limit_str:
        raise ValueError(
            "Missing required env var BYBIT_MM_DELTA_BPS_LIMIT (basis points, bps)."
        )
    try:
        delta_bps_limit = float(delta_bps_limit_str)
    except ValueError as exc:
        raise ValueError(
            "BYBIT_MM_DELTA_BPS_LIMIT must be a finite float in basis points (bps)."
        ) from exc
    if not np.isfinite(delta_bps_limit) or delta_bps_limit <= 0.0:
        raise ValueError(
            "BYBIT_MM_DELTA_BPS_LIMIT must be finite and > 0 in basis points (bps)."
        )
    if inv_soft_notional <= 0.0:
        raise ValueError("BYBIT_MM_INV_SOFT_NOTIONAL must be > 0 (quote notional, USD).")
    if max_inventory_notional <= 0.0:
        raise ValueError("BYBIT_MM_MAX_INV_NOTIONAL must be > 0 (quote notional, USD).")
    if max_inventory_notional < inv_soft_notional:
        raise ValueError("BYBIT_MM_MAX_INV_NOTIONAL must be >= BYBIT_MM_INV_SOFT_NOTIONAL.")
    if not np.isfinite(hard_max_inventory_notional) or hard_max_inventory_notional <= 0.0:
        raise ValueError(
            "BYBIT_MM_HARD_MAX_INV_NOTIONAL must be finite and > 0 (quote notional, USD)."
        )
    if hard_max_inventory_notional < max_inventory_notional:
        raise ValueError(
            "BYBIT_MM_HARD_MAX_INV_NOTIONAL must be >= BYBIT_MM_MAX_INV_NOTIONAL."
        )
    if lambda_inv > 0.0 and inv_soft_notional > 0.0:
        print(
            "[mm config warning]",
            "Inventory penalties now use USD notional units; retune",
            "BYBIT_MM_INVENTORY_PENALTY/BYBIT_MM_LAMBDA_INV if needed.",
            f"inv_soft_notional={inv_soft_notional}",
            f"lambda_inv={lambda_inv}",
        )
    return {
        "maker_rebate_bps": maker_rebate_bps,
        "taker_fee_bps": taker_fee_bps,
        "taker_threshold": taker_threshold,
        "inventory_penalty": inventory_penalty,
        "inv_soft_notional": inv_soft_notional,
        "lambda_inv": lambda_inv,
        "lambda_turn": lambda_turn,
        "max_inventory_notional": max_inventory_notional,
        "hard_max_inventory_notional": hard_max_inventory_notional,
        "fill_size": fill_size,
        "fill_tolerance": fill_tolerance,
        "delta_bps_limit": delta_bps_limit,
    }


def _resolve_ppo_epochs(default: int) -> int:
    return _env_int(PPO_EPOCHS_ENV, default)


def _torch_load_trusted_checkpoint(path, map_location):
    """
    Load a trusted project checkpoint with full pickle semantics.

    PyTorch 2.6+ defaults torch.load to weights_only=True, which breaks rich
    PPO checkpoints that intentionally include non-weight metadata.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


@dataclass(frozen=True)
class EvalCheckpointResolution:
    resolved_eval_ckpt: Optional[str]
    checkpoint_origin: str
    external_ckpt_explicit: bool
    checkpoint_payload: Optional[Dict[str, Any]]


def _resolve_eval_checkpoint(
    run_mode: str,
    mm_best_ckpt: Path,
    external_rl_ckpt_raw: str,
    require_rl_ckpt: bool,
) -> EvalCheckpointResolution:
    external_rl_ckpt = external_rl_ckpt_raw.strip()
    external_ckpt_explicit = bool(external_rl_ckpt)
    resolved_external_rl_ckpt = (
        str(Path(external_rl_ckpt).expanduser().resolve()) if external_rl_ckpt else None
    )
    mm_best_ckpt_resolved = str(mm_best_ckpt.expanduser().resolve())

    if run_mode == "eval":
        if not external_rl_ckpt:
            raise SystemExit(
                "BYBIT_MM_RL_CKPT must be set to a non-empty checkpoint path when run_mode=eval."
            )
        resolved_eval_ckpt = resolved_external_rl_ckpt
        if resolved_eval_ckpt is None:
            raise SystemExit(
                "Unable to resolve BYBIT_MM_RL_CKPT when run_mode=eval; provide a valid file path."
            )
        if not Path(resolved_eval_ckpt).exists():
            raise FileNotFoundError(
                f"BYBIT_MM_RL_CKPT does not exist for run_mode=eval: {resolved_eval_ckpt}"
            )
        return EvalCheckpointResolution(
            resolved_eval_ckpt=resolved_eval_ckpt,
            checkpoint_origin="external",
            external_ckpt_explicit=external_ckpt_explicit,
            checkpoint_payload=_torch_load_trusted_checkpoint(
                Path(resolved_eval_ckpt),
                map_location="cpu",
            ),
        )

    if run_mode == "train_eval":
        if require_rl_ckpt and not external_ckpt_explicit:
            raise SystemExit(
                "BYBIT_MM_REQUIRE_RL_CKPT=true requires explicit BYBIT_MM_RL_CKPT when run_mode=train_eval."
            )
        # In train_eval, explicit BYBIT_MM_RL_CKPT is treated as user intent;
        # missing path is fatal, no baseline fallback.
        if external_ckpt_explicit:
            resolved_eval_ckpt = resolved_external_rl_ckpt
            if resolved_eval_ckpt is None or not Path(resolved_eval_ckpt).exists():
                raise FileNotFoundError(
                    f"explicit external checkpoint missing: {resolved_eval_ckpt}"
                )
            return EvalCheckpointResolution(
                resolved_eval_ckpt=resolved_eval_ckpt,
                checkpoint_origin="external",
                external_ckpt_explicit=external_ckpt_explicit,
                checkpoint_payload=_torch_load_trusted_checkpoint(
                    Path(resolved_eval_ckpt),
                    map_location="cpu",
                ),
            )
        return EvalCheckpointResolution(
            resolved_eval_ckpt=mm_best_ckpt_resolved,
            checkpoint_origin="fresh_train",
            external_ckpt_explicit=external_ckpt_explicit,
            checkpoint_payload=None,
        )

    return EvalCheckpointResolution(
        resolved_eval_ckpt=(
            resolved_external_rl_ckpt
            if resolved_external_rl_ckpt is not None
            else mm_best_ckpt_resolved
        ),
        checkpoint_origin="external" if resolved_external_rl_ckpt is not None else "none",
        external_ckpt_explicit=external_ckpt_explicit,
        checkpoint_payload=None,
    )


def _set_seed_from_env(env_name: str = "BYBIT_SEED") -> Optional[int]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    seed = int(raw)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("[seed]", f"{env_name}={seed}")
    return seed


@dataclass(frozen=True)
class BaselineQuoteConfig:
    base_half_spread_bps: float
    alpha_center_scale: float
    inventory_center_scale: float
    vol_width_scale: float
    uncertainty_width_scale: float
    inventory_side_widen_scale: float
    obs_spread_anchor_frac: float
    spread_floor_bps: float
    spread_cap_bps: float
    inv_ref_notional: float
    horizons_ms: List[int]
    p250_weight: float
    p500_weight: float
    p1000_weight: float


@dataclass(frozen=True)
class BaselineAlphaCalibration:
    score_to_bps_slope_by_horizon: np.ndarray
    winsor_lower_bps_by_horizon: np.ndarray
    winsor_upper_bps_by_horizon: np.ndarray
    fit_count_by_horizon: np.ndarray
    horizons_ms: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class PreparedFastBaselineBatch:
    split_name: str
    rows: int
    initial_cash: float
    fill_size: float
    maker_rebate_bps: float
    taker_fee_bps: float
    fill_tolerance: float
    delta_bps_limit: float
    inventory_penalty: float
    inv_soft_notional: float
    lambda_inv: float
    lambda_turn: float
    max_inventory_notional: float
    hard_max_inventory_notional: float
    fill_ema_alpha: float
    best_bid: np.ndarray
    best_ask: np.ndarray
    best_bid_next: np.ndarray
    best_ask_next: np.ndarray
    best_bid_prev: np.ndarray
    best_ask_prev: np.ndarray
    mid: np.ndarray
    mid_next: np.ndarray
    observed_spread_bps: np.ndarray
    decision_ts: Optional[np.ndarray]
    dir_logits_by_horizon: np.ndarray
    p_up_by_horizon: np.ndarray
    direction_confidence_by_horizon: np.ndarray
    future_ret_by_horizon: np.ndarray
    p250: np.ndarray
    p500: np.ndarray
    p1000: np.ndarray
    vol_short: np.ndarray
    vol_long: np.ndarray


@dataclass(frozen=True)
class PreparedBaselineContext:
    out_root: str
    ckpt_path: str
    device: str
    meta: Dict[str, Any]
    cmssl_test_split: Dict[str, Any]
    rl_val_split: Dict[str, Any]
    rl_test_split: Dict[str, Any]
    eval_full_split: Dict[str, Any]
    cmssl_test_metrics: Dict[str, Any]
    joined_rl_rows: int
    joined_eval_rows: int
    mm_rl_val_batch: "MarketMakingBatch"
    mm_rl_test_batch: "MarketMakingBatch"
    fast_rl_val_batch: Optional[PreparedFastBaselineBatch]
    fast_rl_test_batch: Optional[PreparedFastBaselineBatch]
    fast_rl_train_batch: PreparedFastBaselineBatch
    fast_eval_full_batch: PreparedFastBaselineBatch
    baseline_alpha_calibration: BaselineAlphaCalibration
    env_kwargs_common: Dict[str, Any]
    run_config: Dict[str, Any]
    cmssl_batch_size: int


def load_baseline_quote_config() -> BaselineQuoteConfig:
    return BaselineQuoteConfig(
        base_half_spread_bps=_env_float("BYBIT_MM_BASE_HALF_SPREAD_BPS", DEFAULT_MM_BASE_HALF_SPREAD_BPS),
        alpha_center_scale=_env_float("BYBIT_MM_ALPHA_CENTER_SCALE", DEFAULT_MM_ALPHA_CENTER_SCALE),
        inventory_center_scale=_env_float("BYBIT_MM_INVENTORY_CENTER_SCALE", DEFAULT_MM_INVENTORY_CENTER_SCALE),
        vol_width_scale=_env_float("BYBIT_MM_VOL_WIDTH_SCALE", DEFAULT_MM_VOL_WIDTH_SCALE),
        uncertainty_width_scale=_env_float("BYBIT_MM_UNCERTAINTY_WIDTH_SCALE", DEFAULT_MM_UNCERTAINTY_WIDTH_SCALE),
        inventory_side_widen_scale=_env_float("BYBIT_MM_INVENTORY_SIDE_WIDEN_SCALE", DEFAULT_MM_INVENTORY_SIDE_WIDEN_SCALE),
        obs_spread_anchor_frac=_env_float("BYBIT_MM_OBS_SPREAD_ANCHOR_FRAC", DEFAULT_MM_OBS_SPREAD_ANCHOR_FRAC),
        spread_floor_bps=_env_float("BYBIT_MM_SPREAD_FLOOR_BPS", DEFAULT_MM_SPREAD_FLOOR_BPS),
        spread_cap_bps=_env_float("BYBIT_MM_SPREAD_CAP_BPS", DEFAULT_MM_SPREAD_CAP_BPS),
        inv_ref_notional=_env_float("BYBIT_MM_INV_REF_NOTIONAL", DEFAULT_MM_INV_REF_NOTIONAL),
        # CMSSL MM contract is fixed to [250, 500, 1000]; any deviation is a hard error.
        horizons_ms=_env_int_list("BYBIT_MM_HORIZONS_MS", DEFAULT_MM_HORIZONS_MS),
        p250_weight=_env_float("BYBIT_MM_P250_WEIGHT", DEFAULT_MM_P250_WEIGHT),
        p500_weight=_env_float("BYBIT_MM_P500_WEIGHT", DEFAULT_MM_P500_WEIGHT),
        p1000_weight=_env_float("BYBIT_MM_P1000_WEIGHT", DEFAULT_MM_P1000_WEIGHT),
    )


def _validate_baseline_quote_config(cfg: BaselineQuoteConfig, *, tol: float = 1e-6) -> BaselineQuoteConfig:
    horizons = _validate_fixed_cmssl_horizons(_normalize_horizons(len(cfg.horizons_ms), cfg.horizons_ms))
    weight_sum = float(cfg.p250_weight + cfg.p500_weight + cfg.p1000_weight)
    if abs(weight_sum - 1.0) > tol:
        raise ValueError(f"Baseline horizon weights must sum to 1.0; got {weight_sum:.12f}")
    if not np.isfinite(cfg.base_half_spread_bps) or cfg.base_half_spread_bps < 0.0:
        raise ValueError("base_half_spread_bps must be finite and >= 0.0")
    if not np.isfinite(cfg.alpha_center_scale):
        raise ValueError("alpha_center_scale must be finite")
    if not np.isfinite(cfg.inventory_center_scale):
        raise ValueError("inventory_center_scale must be finite")
    if not np.isfinite(cfg.vol_width_scale) or cfg.vol_width_scale < 0.0:
        raise ValueError("vol_width_scale must be finite and >= 0.0")
    if not np.isfinite(cfg.uncertainty_width_scale) or cfg.uncertainty_width_scale < 0.0:
        raise ValueError("uncertainty_width_scale must be finite and >= 0.0")
    if not np.isfinite(cfg.inventory_side_widen_scale) or cfg.inventory_side_widen_scale < 0.0:
        raise ValueError("inventory_side_widen_scale must be finite and >= 0.0")
    if cfg.inv_ref_notional <= 0.0:
        raise ValueError("inv_ref_notional must be > 0")
    if not np.isfinite(cfg.obs_spread_anchor_frac) or cfg.obs_spread_anchor_frac < 0.0:
        raise ValueError(
            "obs_spread_anchor_frac must be finite and >= 0.0 "
            "(applies a floor using the observed spread before width scaling)"
        )
    if cfg.spread_cap_bps < cfg.spread_floor_bps:
        raise ValueError("spread_cap_bps must be >= spread_floor_bps")
    return BaselineQuoteConfig(**{**cfg.__dict__, "horizons_ms": horizons})


def resolve_baseline_quote_config_from_mapping(mapping: Dict[str, Any]) -> BaselineQuoteConfig:
    cfg = BaselineQuoteConfig(
        base_half_spread_bps=float(mapping.get("base_half_spread_bps", DEFAULT_MM_BASE_HALF_SPREAD_BPS)),
        alpha_center_scale=float(mapping.get("alpha_center_scale", DEFAULT_MM_ALPHA_CENTER_SCALE)),
        inventory_center_scale=float(mapping.get("inventory_center_scale", DEFAULT_MM_INVENTORY_CENTER_SCALE)),
        vol_width_scale=float(mapping.get("vol_width_scale", DEFAULT_MM_VOL_WIDTH_SCALE)),
        uncertainty_width_scale=float(mapping.get("uncertainty_width_scale", DEFAULT_MM_UNCERTAINTY_WIDTH_SCALE)),
        inventory_side_widen_scale=float(mapping.get("inventory_side_widen_scale", DEFAULT_MM_INVENTORY_SIDE_WIDEN_SCALE)),
        obs_spread_anchor_frac=float(mapping.get("obs_spread_anchor_frac", DEFAULT_MM_OBS_SPREAD_ANCHOR_FRAC)),
        spread_floor_bps=float(mapping.get("spread_floor_bps", DEFAULT_MM_SPREAD_FLOOR_BPS)),
        spread_cap_bps=float(mapping.get("spread_cap_bps", DEFAULT_MM_SPREAD_CAP_BPS)),
        inv_ref_notional=float(mapping.get("inv_ref_notional", DEFAULT_MM_INV_REF_NOTIONAL)),
        horizons_ms=[int(h) for h in mapping.get("horizons_ms", DEFAULT_MM_HORIZONS_MS)],
        p250_weight=float(mapping.get("p250_weight", DEFAULT_MM_P250_WEIGHT)),
        p500_weight=float(mapping.get("p500_weight", DEFAULT_MM_P500_WEIGHT)),
        p1000_weight=float(mapping.get("p1000_weight", DEFAULT_MM_P1000_WEIGHT)),
    )
    return _validate_baseline_quote_config(cfg)


def baseline_quote_config_to_dict(cfg: BaselineQuoteConfig) -> Dict[str, Any]:
    validated = _validate_baseline_quote_config(cfg)
    return {
        "base_half_spread_bps": float(validated.base_half_spread_bps),
        "alpha_center_scale": float(validated.alpha_center_scale),
        "inventory_center_scale": float(validated.inventory_center_scale),
        "vol_width_scale": float(validated.vol_width_scale),
        "uncertainty_width_scale": float(validated.uncertainty_width_scale),
        "inventory_side_widen_scale": float(validated.inventory_side_widen_scale),
        "obs_spread_anchor_frac": float(validated.obs_spread_anchor_frac),
        "spread_floor_bps": float(validated.spread_floor_bps),
        "spread_cap_bps": float(validated.spread_cap_bps),
        "inv_ref_notional": float(validated.inv_ref_notional),
        "horizons_ms": [int(h) for h in validated.horizons_ms],
        "p250_weight": float(validated.p250_weight),
        "p500_weight": float(validated.p500_weight),
        "p1000_weight": float(validated.p1000_weight),
    }


def _resolve_baseline_quote_config(
    baseline_config: Optional[Dict[str, Any] | BaselineQuoteConfig],
) -> Optional[BaselineQuoteConfig]:
    if baseline_config is None:
        return None
    if isinstance(baseline_config, BaselineQuoteConfig):
        return _validate_baseline_quote_config(baseline_config)
    return resolve_baseline_quote_config_from_mapping(baseline_config)


def resolve_baseline_vol_bucket_edges_bps() -> List[float]:
    raw_value = os.environ.get("BYBIT_MM_BASELINE_VOL_BUCKET_BPS", "0.5,1.0,2.0,4.0,8.0")
    edges = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not edges:
        raise ValueError("BYBIT_MM_BASELINE_VOL_BUCKET_BPS must contain at least one positive edge")
    prev = 0.0
    for edge in edges:
        if not np.isfinite(edge) or edge <= 0.0:
            raise ValueError("Baseline vol bucket edges must be finite and > 0")
        if edge <= prev:
            raise ValueError("Baseline vol bucket edges must be strictly increasing")
        prev = edge
    return edges


def build_vol_bucket_report(
    *,
    sigma_bps_selected: np.ndarray,
    delta_equity_per_step: np.ndarray,
    reward_per_step: np.ndarray,
    maker_buy_per_step: np.ndarray,
    maker_sell_per_step: np.ndarray,
    turnover_notional_per_step: np.ndarray,
    maker_buy_markout_per_step: Optional[np.ndarray] = None,
    maker_sell_markout_per_step: Optional[np.ndarray] = None,
    initial_equity: float,
    bucket_edges_bps: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    sigma_arr = np.asarray(sigma_bps_selected, dtype=np.float64)
    delta_arr = np.asarray(delta_equity_per_step, dtype=np.float64)
    reward_arr = np.asarray(reward_per_step, dtype=np.float64)
    maker_buy_arr = np.asarray(maker_buy_per_step, dtype=np.float64)
    maker_sell_arr = np.asarray(maker_sell_per_step, dtype=np.float64)
    turnover_arr = np.asarray(turnover_notional_per_step, dtype=np.float64)
    if bucket_edges_bps is None:
        bucket_edges = resolve_baseline_vol_bucket_edges_bps()
    else:
        bucket_edges = [float(edge) for edge in bucket_edges_bps]
    total_steps = int(sigma_arr.size)
    safe_sigma = np.where(np.isfinite(sigma_arr), sigma_arr, np.inf)
    bucket_indices = np.searchsorted(np.asarray(bucket_edges, dtype=np.float64), safe_sigma, side="right")
    report_rows: List[Dict[str, Any]] = []
    maker_buy_markout_arr = None if maker_buy_markout_per_step is None else np.asarray(maker_buy_markout_per_step, dtype=np.float64)
    maker_sell_markout_arr = None if maker_sell_markout_per_step is None else np.asarray(maker_sell_markout_per_step, dtype=np.float64)
    for bucket_index in range(len(bucket_edges) + 1):
        mask = bucket_indices == bucket_index
        step_count = int(np.count_nonzero(mask))
        maker_buy_fills = int(np.count_nonzero(maker_buy_arr[mask] > 0.0))
        maker_sell_fills = int(np.count_nonzero(maker_sell_arr[mask] > 0.0))
        maker_fill_count = maker_buy_fills + maker_sell_fills
        maker_opportunities = 2 * step_count
        lower = 0.0 if bucket_index == 0 else float(bucket_edges[bucket_index - 1])
        upper = None if bucket_index == len(bucket_edges) else float(bucket_edges[bucket_index])
        bucket_row: Dict[str, Any] = {
            "bucket_index": bucket_index,
            "bucket_lower_bps": lower,
            "bucket_upper_bps": upper,
            "bucket_label": f"[{lower:.4f}, inf)" if upper is None else f"[{lower:.4f}, {upper:.4f})",
            "step_count": step_count,
            "step_fraction": float(step_count / total_steps) if total_steps > 0 else 0.0,
            "maker_fill_count": maker_fill_count,
            "maker_buy_fills": maker_buy_fills,
            "maker_sell_fills": maker_sell_fills,
            "maker_opportunities": maker_opportunities,
            "maker_fill_rate": float(maker_fill_count / maker_opportunities) if maker_opportunities > 0 else 0.0,
            "turnover_notional": float(np.sum(turnover_arr[mask])) if step_count > 0 else 0.0,
            "delta_equity": float(np.sum(delta_arr[mask])) if step_count > 0 else 0.0,
            "reward": float(np.sum(reward_arr[mask])) if step_count > 0 else 0.0,
            "net_pnl_pct_contrib": float(np.sum(delta_arr[mask]) / max(initial_equity, 1e-12)) if step_count > 0 else 0.0,
        }
        if maker_buy_markout_arr is not None:
            bucket_row["maker_buy_markout"] = float(np.sum(maker_buy_markout_arr[mask])) if step_count > 0 else 0.0
        if maker_sell_markout_arr is not None:
            bucket_row["maker_sell_markout"] = float(np.sum(maker_sell_markout_arr[mask])) if step_count > 0 else 0.0
        report_rows.append(bucket_row)
    return {
        "vol_bucket_edges_bps": bucket_edges,
        "vol_bucket_report": report_rows,
    }


def _infer_num_horizons(feature_dim: int) -> int:
    base_dim = feature_dim - len(RAW_SNAPSHOT_FEATURE_COLUMNS) - FEATURE_EXTRA_DIM
    if base_dim <= 0 or base_dim % 2 != 0:
        raise ValueError(
            "Feature dimension does not align with expected horizon layout: "
            f"feature_dim={feature_dim} base_dim={base_dim}"
        )
    return base_dim // 2


def _joined_feature_layout(num_horizons: int, snapshot_dim: int) -> Dict[str, slice]:
    """Schema for join_features() tensor layout (excluding env extra state).

    Layout order:
      [dir_logits(h), p_up(h), confidence/alignment scalars(5), snapshots(snapshot_dim)]
    """
    offset = 0
    layout = {
        "dir_logits": slice(offset, offset + num_horizons)
    }
    offset += num_horizons
    layout["p_up"] = slice(offset, offset + num_horizons)
    offset += num_horizons
    layout["align_all"] = slice(offset, offset + 1)
    offset += 1
    layout["diff_short_long"] = slice(offset, offset + 1)
    offset += 1
    layout["diff_mid_long"] = slice(offset, offset + 1)
    offset += 1
    layout["conf_long"] = slice(offset, offset + 1)
    offset += 1
    layout["conf_min"] = slice(offset, offset + 1)
    offset += 1
    layout["snapshots"] = slice(offset, offset + snapshot_dim)
    return layout


def _normalize_horizons(
    num_h: int,
    horizons: List[int],
) -> List[int]:
    if len(horizons) == num_h:
        return list(horizons)
    raise ValueError(
        "Horizon count mismatch between model outputs and configured horizons: "
        f"num_model_horizons={num_h} configured_horizons={horizons} configured_count={len(horizons)}. "
        "Set BYBIT_MM_HORIZONS_MS to exactly match model horizons; mismatches are hard errors."
    )


def _validate_fixed_cmssl_horizons(horizons: List[int]) -> List[int]:
    expected_horizons = [250, 500, 1000]
    unique_sorted = sorted(set(horizons))
    if unique_sorted != expected_horizons:
        raise ValueError(
            "CMSSL horizon contract violation: configured horizons must be exactly "
            f"{expected_horizons}, got configured_horizons={horizons} "
            f"(unique_sorted={unique_sorted})."
        )
    return expected_horizons


def _resolve_horizon_index(
    target_ms: int,
    horizons: List[int],
    *,
    label: str,
) -> int:
    if target_ms in horizons:
        return horizons.index(target_ms)
    raise ValueError(
        "Required horizon is not available in configured horizon mapping: "
        f"label={label} target_ms={target_ms} configured_horizons={horizons}."
    )


def _resolve_split_range(range_value: Any, *, label: str) -> Tuple[int, int]:
    if not isinstance(range_value, dict):
        raise KeyError(f"meta['splits']['{label}'] must include decision_ts_range with start/end.")
    try:
        start = int(range_value["start"])
        end = int(range_value["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"meta['splits']['{label}']['decision_ts_range'] must contain integer start/end."
        ) from exc
    if start >= end:
        raise ValueError(f"meta['splits']['{label}']['decision_ts_range'] must satisfy start < end.")
    return start, end


def _resolve_meta_full_week_range(meta: Dict[str, Any], week_key: str, *, label: str) -> Tuple[int, int]:
    week_ranges = meta.get("week_decision_ts_ranges")
    if isinstance(week_ranges, dict):
        wk = week_ranges.get(week_key)
        if isinstance(wk, dict):
            if "start" in wk and "end" in wk:
                return _resolve_split_range(wk, label=label)
            if "min" in wk and "max" in wk:
                return int(wk["min"]), int(wk["max"])
    weeks_meta = meta.get("weeks_meta")
    if isinstance(weeks_meta, dict):
        wk_meta = weeks_meta.get(week_key)
        if isinstance(wk_meta, dict):
            decision_ts_range = wk_meta.get("decision_ts_range")
            if isinstance(decision_ts_range, dict):
                if "start" in decision_ts_range and "end" in decision_ts_range:
                    return _resolve_split_range(decision_ts_range, label=label)
                if "min" in decision_ts_range and "max" in decision_ts_range:
                    start = int(decision_ts_range["min"])
                    end = int(decision_ts_range["max"])
                    if start >= end:
                        raise ValueError(f"meta week range for {label} must satisfy start < end.")
                    return start, end
    raise KeyError(
        f"meta is missing inline decision_ts_range metadata for full-week split '{label}' ({week_key})."
    )


def _normalize_pipeline_split_entry(meta: Dict[str, Any], split_entry: Any, *, label: str, require_range: bool) -> Dict[str, Any]:
    if not isinstance(split_entry, dict):
        raise KeyError(f"meta['splits']['{label}'] must be a dict.")
    week_value = split_entry.get("week", split_entry.get("weeks"))
    if isinstance(week_value, str) and week_value:
        weeks = [week_value]
    elif isinstance(week_value, list) and week_value and all(isinstance(w, str) and w for w in week_value):
        weeks = list(week_value)
    else:
        raise KeyError(f"meta['splits']['{label}'] must include non-empty 'week' or 'weeks'.")
    known_weeks = meta.get("weeks_in_order")
    if not isinstance(known_weeks, list) or len(known_weeks) != 4:
        raise KeyError("meta['weeks_in_order'] must be a list[str] with exactly 4 entries.")
    missing_weeks = [wk for wk in weeks if wk not in set(known_weeks)]
    if missing_weeks:
        raise KeyError(f"meta['splits']['{label}'] references unknown week(s): {missing_weeks}")
    if require_range:
        start, end = _resolve_split_range(split_entry.get("decision_ts_range"), label=label)
    else:
        start, end = _resolve_meta_full_week_range(meta, weeks[0], label=label)
    return {"weeks": weeks, "start": start, "end": end}


def require_four_week_pipeline_splits(meta: Dict[str, Any]) -> Dict[str, Any]:
    _require_event_time_decision_meta(meta)
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        raise KeyError("meta['splits'] must be a dict.")
    weeks_in_order = meta.get("weeks_in_order")
    if not (isinstance(weeks_in_order, list) and len(weeks_in_order) == 4 and all(isinstance(w, str) and w for w in weeks_in_order)):
        raise KeyError("meta['weeks_in_order'] must be a list[str] with exactly 4 entries.")
    if splits.get("protocol") != "four_week_cmssl_val_test_rl_eval_v2":
        raise ValueError("meta['splits']['protocol'] must be 'four_week_cmssl_val_test_rl_eval_v2'.")
    normalized = {"protocol": splits["protocol"]}
    for section in ("cmssl", "rl", "eval"):
        if not isinstance(splits.get(section), dict):
            raise KeyError(f"meta['splits']['{section}'] must be a dict.")
        normalized[section] = {}
    required_entries = {
        "cmssl.train": ("cmssl", "train", False),
        "cmssl.val": ("cmssl", "val", False),
        "cmssl.test": ("cmssl", "test", False),
        "rl.train": ("rl", "train", True),
        "rl.val": ("rl", "val", True),
        "rl.test": ("rl", "test", True),
        "eval.full": ("eval", "full", False),
    }
    for label, (section, name, require_range) in required_entries.items():
        normalized[section][name] = _normalize_pipeline_split_entry(meta, splits[section].get(name), label=label, require_range=require_range)
    week1, week2, week3, week4 = weeks_in_order
    require(normalized["cmssl"]["train"]["weeks"] == [week1], "meta['splits']['cmssl']['train'] must reference weeks_in_order[0].")
    require(normalized["cmssl"]["val"]["weeks"] == [week2], "meta['splits']['cmssl']['val'] must reference weeks_in_order[1].")
    require(normalized["cmssl"]["test"]["weeks"] == [week3], "meta['splits']['cmssl']['test'] must reference weeks_in_order[2].")
    for split_name in ("train", "val", "test"):
        require(normalized["rl"][split_name]["weeks"] == [week3], f"meta['splits']['rl']['{split_name}'] must reference weeks_in_order[2].")
    require(normalized["eval"]["full"]["weeks"] == [week4], "meta['splits']['eval']['full'] must reference weeks_in_order[3].")
    cmssl_test = normalized["cmssl"]["test"]
    week3_start, week3_end = _resolve_meta_full_week_range(meta, week3, label="weeks_in_order[2]")
    require(
        cmssl_test["start"] == week3_start and cmssl_test["end"] == week3_end,
        "meta['splits']['cmssl']['test'] must cover the full CMSSL week-3 range from metadata."
    )
    rl_train = normalized["rl"]["train"]
    rl_val = normalized["rl"]["val"]
    rl_test = normalized["rl"]["test"]
    require(
        rl_train["start"] >= week3_start and rl_train["end"] <= week3_end
        and rl_val["start"] >= week3_start and rl_val["end"] <= week3_end
        and rl_test["start"] >= week3_start and rl_test["end"] <= week3_end,
        "meta['splits']['rl'] train/val/test must stay within CMSSL test week-3 boundaries."
    )
    require(rl_train["end"] <= rl_val["start"] < rl_val["end"] <= rl_test["start"] < rl_test["end"], "meta['splits']['rl'] train/val/test must be strictly ordered and non-overlapping.")
    require(
        normalized["rl"]["train"]["weeks"] == normalized["cmssl"]["test"]["weeks"],
        "meta['splits']['rl'] and meta['splits']['cmssl']['test'] must reference the same week (week-3)."
    )
    return normalized


def resolve_cmssl_train_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["cmssl"]["train"])


def resolve_cmssl_val_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["cmssl"]["val"])


def resolve_cmssl_test_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["cmssl"]["test"])


def resolve_rl_train_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["rl"]["train"])


def resolve_rl_val_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["rl"]["val"])


def resolve_rl_test_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["rl"]["test"])


def resolve_eval_full_split(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta)["eval"]["full"])


def _split_weeks(split: Dict[str, Any]) -> list[str]:
    weeks = split.get("weeks")
    if isinstance(weeks, list) and len(weeks) > 0:
        return list(weeks)
    raise KeyError("split must contain non-empty 'weeks' list")


def load_split_arrays(out_root: str, split: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CMSSL tensors for a split.

    Args:
        out_root: Output root containing CMSSL chunk artifacts.
        split: Split config with ``weeks`` (non-empty list of week keys),
            plus ``start``/``end`` timestamp bounds.
    """
    weeks = _split_weeks(split)
    x_core_list: List[np.ndarray] = []
    x_aux_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    for week, chunk_idx, ts, x_core, x_aux, y in iter_chunk_batches(out_root):
        if week not in weeks:
            continue
        n_rows = x_core.shape[0]
        if ts is None:
            raise ValueError(
                f"Missing decision timestamps for {week}/chunk{chunk_idx:03d}. "
                "Ensure meta_week.json includes ts_*.npy entries."
            )
        if ts.ndim != 1 or ts.shape[0] != n_rows:
            raise ValueError(
                f"{week}/chunk{chunk_idx:03d} timestamps length mismatch: "
                f"expected {n_rows}, got {ts.shape}"
            )
        _ensure_monotonic(ts, f"{week}/chunk{chunk_idx:03d}")
        mask = (ts >= split["start"]) & (ts < split["end"])
        if not np.any(mask):
            continue
        x_core_list.append(x_core[mask])
        x_aux_list.append(x_aux[mask])
        y_list.append(y[mask])
        ts_list.append(ts[mask])
    if not x_core_list:
        raise ValueError(f"No data found for split {split}")
    x_core_all = np.concatenate(x_core_list, axis=0)
    x_aux_all = np.concatenate(x_aux_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    ts_all = np.concatenate(ts_list, axis=0)
    order = np.argsort(ts_all)
    return x_core_all[order], x_aux_all[order], y_all[order], ts_all[order]



def _build_windowed_inputs(
    x_core: np.ndarray,
    x_aux: np.ndarray,
    ts: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_core.ndim == 3:
        if ts.shape[0] != x_core.shape[0]:
            raise ValueError("Timestamp length does not match windowed inputs.")
        return x_core, x_aux, ts
    if x_core.ndim != 2:
        raise ValueError(f"Expected x_core to be 2D or 3D, got {x_core.ndim}D")
    if ts.ndim != 1 or ts.shape[0] != x_core.shape[0]:
        raise ValueError("Raw token timestamps must be 1D and match x_core rows.")
    n, feat_dim = x_core.shape
    if n < lookback:
        return (
            np.empty((0, lookback, feat_dim), dtype=x_core.dtype),
            np.empty((0, lookback, x_aux.shape[1]), dtype=x_aux.dtype),
            np.empty((0,), dtype=ts.dtype),
        )
    window_count = n - lookback + 1
    x_core_win = np.empty((window_count, lookback, feat_dim), dtype=x_core.dtype)
    x_aux_win = np.empty((window_count, lookback, x_aux.shape[1]), dtype=x_aux.dtype)
    for i in range(lookback - 1, n):
        start = i - lookback + 1
        x_core_win[start] = x_core[start:i + 1]
        x_aux_win[start] = x_aux[start:i + 1]
    return x_core_win, x_aux_win, ts[lookback - 1:]


def load_cmssl_test_windowed_inputs(
    out_root: str,
    meta: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = resolve_cmssl_test_split(meta)
    x_core, x_aux, _y, ts = load_split_arrays(out_root, split)
    return _build_windowed_inputs(x_core, x_aux, ts, lookback=LOOKBACK)


def run_cmssl_test_window_inference(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run CMSSL inference over test windowed inputs for offline diagnostics."""
    model, meta = load_cmssl(out_root, ckpt_path, device=device)
    x_core, x_aux, ts = load_cmssl_test_windowed_inputs(out_root, meta)
    resolved_batch_size = _resolve_cmssl_batch_size() if batch_size is None else int(batch_size)
    cmssl_out = run_cmssl_inference(
        model,
        meta,
        x_core,
        x_aux,
        batch_size=resolved_batch_size,
        device=device,
    )
    horizons = meta.get("horizons_ms", [])
    output: Dict[str, Dict[int, np.ndarray]] = {
        "horizons_ms": horizons,
        "dir_logits": {},
    }
    for idx, ts_val in enumerate(ts):
        ts_key = int(ts_val)
        output["dir_logits"][ts_key] = cmssl_out["dir_logits"][idx]
    return output


def run_cmssl_inference(
    model,
    meta: dict,
    x_core: np.ndarray,
    x_aux: np.ndarray,
    batch_size: int = 2048,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Run CMSSL inference for batched inputs; empty batches are valid."""
    t0 = time.perf_counter()
    num_h = len(meta["horizons_ms"])
    n = x_core.shape[0]
    if n == 0:
        empty = np.empty((0, num_h), dtype=np.float32)
        return {
            "dir_logits": empty.copy(),
        }
    logits_out = np.empty((n, num_h), dtype=np.float32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        xc = x_core[i:j]
        xa = x_aux[i:j]
        dir_logits = cmssl_predict(model, xc, xa, meta, device=device)
        logits_out[i:j] = dir_logits.detach().cpu().numpy().astype(np.float32, copy=False)
    elapsed = time.perf_counter() - t0
    if elapsed > 0.0:
        _timing_log(
            f"cmssl_inference rows={n} batch_size={batch_size} secs={elapsed:.4f} rows_per_sec={n / elapsed:.2f}"
        )
    else:
        _timing_log(f"cmssl_inference rows={n} batch_size={batch_size} secs={elapsed:.4f}")
    return {
        "dir_logits": logits_out,
    }


def _find_week_dir(out_root: Path, week_key: str) -> Path:
    meta = load_global_meta(out_root)
    for wk, _wmeta, wk_dir in iter_week_chunks(out_root, meta=meta):
        if wk == week_key:
            return wk_dir
    raise ValueError(f"Unable to locate week directory for {week_key}")


def _ffill_1d(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if out.size == 0:
        return out
    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        out[:] = 0.0
        return out
    idx = np.where(finite_mask, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    out[~np.isfinite(out)] = 0.0
    return out


def _rolling_std_ignore_nan(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(x)
    x_finite = np.where(finite, x, 0.0)
    x_finite_sq = x_finite * x_finite

    csum = np.cumsum(x_finite)
    csum2 = np.cumsum(x_finite_sq)
    count_csum = np.cumsum(finite.astype(np.int64))

    count = count_csum.copy()
    sums = csum.copy()
    sumsq = csum2.copy()
    if window < n:
        count[window:] -= count_csum[:-window]
        sums[window:] -= csum[:-window]
        sumsq[window:] -= csum2[:-window]

    out = np.full(n, np.nan, dtype=np.float64)
    valid = count > 1
    var_num = sumsq[valid] - (sums[valid] * sums[valid]) / count[valid]
    out[valid] = np.sqrt(np.maximum(var_num / (count[valid] - 1), 0.0))
    return out


def _sanitize_snapshot_features(arr: np.ndarray) -> np.ndarray:
    target_cols = [
        "best_bid_size",
        "best_ask_size",
        "time_since_last_ob_update_ms",
        "imbalance",
        "mid_ret_1",
        "vol_short",
        "vol_long",
        "spread_bps",
    ]
    arr = np.asarray(arr).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D snapshot feature array, got shape={arr.shape}.")
    col_idx = {
        name: RAW_SNAPSHOT_FEATURE_COLUMNS.index(name)
        for name in target_cols
        if name in RAW_SNAPSHOT_FEATURE_COLUMNS and RAW_SNAPSHOT_FEATURE_COLUMNS.index(name) < arr.shape[1]
    }
    for idx in col_idx.values():
        col = arr[:, idx].astype(np.float64, copy=False)
        col[~np.isfinite(col)] = np.nan
        col = _ffill_1d(col)
        col[~np.isfinite(col)] = 0.0
        col = col.astype(arr.dtype, copy=False)
        arr[:, idx] = col
    return arr


def _compute_snapshot_feature_matrix(
    snapshot_ts: np.ndarray,
    snapshots: np.ndarray,
    time_since_last_ob_update_ms: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
    snapshots = np.asarray(snapshots)
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D.")
    if snapshots.ndim != 2 or snapshots.shape[1] != 4:
        raise ValueError("Snapshots must be [N,4] with bid/ask and sizes. Rebuild snapshots.")
    order = np.argsort(snapshot_ts)
    snapshot_ts = snapshot_ts[order]
    snapshots = snapshots[order]
    best_bid = snapshots[:, 0].astype(np.float64)
    best_ask = snapshots[:, 1].astype(np.float64)
    best_bid_size = np.maximum(snapshots[:, 2].astype(np.float64), 0.0)
    best_ask_size = np.maximum(snapshots[:, 3].astype(np.float64), 0.0)
    if time_since_last_ob_update_ms is None:
        time_since_last_ob_update_ms = np.zeros_like(best_bid, dtype=np.float64)
    else:
        time_since_last_ob_update_ms = np.asarray(time_since_last_ob_update_ms, dtype=np.float64)
        if time_since_last_ob_update_ms.ndim != 1 or time_since_last_ob_update_ms.shape[0] != snapshot_ts.shape[0]:
            raise ValueError("time_since_last_ob_update_ms must be shape [N].")
        time_since_last_ob_update_ms = np.maximum(time_since_last_ob_update_ms[order], 0.0)
    mid = (best_bid + best_ask) / 2.0
    eps = 1e-9
    imbalance = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size + eps)
    spread_bps = (best_ask - best_bid) / mid * 1e4
    mid_ret_1 = np.log(mid)
    mid_ret_1 = np.concatenate([[np.nan], np.diff(mid_ret_1)])
    vol_short = _rolling_std_ignore_nan(mid_ret_1, SHORT_VOL_WINDOW)
    vol_long = _rolling_std_ignore_nan(mid_ret_1, LONG_VOL_WINDOW)
    features = np.column_stack(
        [
            best_bid,
            best_ask,
            best_bid_size,
            best_ask_size,
            time_since_last_ob_update_ms,
            imbalance,
            mid,
            spread_bps,
            mid_ret_1,
            vol_short,
            vol_long,
        ]
    )
    return snapshot_ts, _sanitize_snapshot_features(features)


def load_raw_snapshots(out_root: str, week_key: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load canonical raw snapshots only (no alternate ingestion fallbacks)."""
    week_dir = _find_week_dir(Path(out_root), week_key)
    canonical_path = week_dir / "snapshots.npz"
    if not canonical_path.exists():
        raise FileNotFoundError("Run offline_snapshots.py first.")

    data = np.load(canonical_path)
    if not {"ts", "snapshots"}.issubset(data.files):
        raise ValueError(
            f"Expected canonical snapshots at {out_root}/{week_key}/snapshots.npz "
            "with fields ts and snapshots. Generate them with offline_snapshots.py."
        )
    snapshots = data["snapshots"]
    if snapshots.ndim != 2 or snapshots.shape[1] != 4:
        raise ValueError(
            f"{canonical_path} has snapshots shape {snapshots.shape}; expected [N,4] (bid, ask, bid_size, ask_size). "
            "Re-run offline_snapshots (1).py to regenerate."
        )
    stale_ms = data["time_since_last_ob_update_ms"] if "time_since_last_ob_update_ms" in data.files else None
    return data["ts"], snapshots, stale_ms


def _format_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _format_duration_ms(duration_ms: int) -> str:
    seconds = duration_ms / 1000.0
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    return f"{duration_ms}ms (~{days:.2f}d)"


def _ensure_monotonic(ts: np.ndarray, label: str) -> None:
    if ts.size and np.any(np.diff(ts) < 0):
        raise ValueError(f"{label} timestamps must be monotonically non-decreasing.")


def report_cmssl_test_diagnostics(out_root: str, meta: dict) -> None:
    """Report diagnostics for CMSSL week-3 out-of-sample/downstream-development split."""
    test_split = resolve_cmssl_test_split(meta)
    split_weeks = _split_weeks(test_split)
    if not split_weeks:
        raise ValueError("Test split contains no weeks.")
    split_weeks_label = ",".join(split_weeks)
    start_ms = int(test_split["start"])
    end_ms = int(test_split["end"])
    duration_ms = end_ms - start_ms
    print(
        "[cmssl split:test week3_oos_downstream_dev]",
        f"weeks={split_weeks_label}",
        f"start={_format_ts(start_ms)}",
        f"end={_format_ts(end_ms)}",
        f"duration={_format_duration_ms(duration_ms)}",
    )
    expected_week_ms = 7 * 24 * 60 * 60 * 1000
    tolerance_ms = 60 * 60 * 1000
    require(abs(duration_ms - expected_week_ms) <= tolerance_ms, (
        f"CMSSL test split duration {duration_ms}ms not ~7 days (week-3 full range)."
    ))

    canonical_snapshot_ts_parts: List[np.ndarray] = []
    for week in split_weeks:
        week_snapshot_ts, _snapshots, _stale_ms = load_raw_snapshots(out_root, week)
        canonical_snapshot_ts_parts.append(np.asarray(week_snapshot_ts, dtype=np.int64))
    canonical_snapshot_ts = np.concatenate(canonical_snapshot_ts_parts, axis=0)
    canonical_snapshot_ts = np.asarray(canonical_snapshot_ts, dtype=np.int64)
    canonical_snapshot_ts = np.sort(canonical_snapshot_ts)
    filtered = canonical_snapshot_ts[(canonical_snapshot_ts >= start_ms) & (canonical_snapshot_ts < end_ms)]
    if filtered.size == 0:
        raise ValueError("No canonical raw snapshots found inside the CMSSL test split range.")
    _ensure_monotonic(filtered, "Raw snapshot (filtered)")
    print(
        "[raw snapshots:test]",
        f"count={filtered.size}",
        f"start={_format_ts(int(filtered[0]))}",
        f"end={_format_ts(int(filtered[-1]))}",
    )


def align_snapshots_to_decisions(
    snapshot_ts: np.ndarray,
    decision_ts: np.ndarray,
) -> np.ndarray:
    """Return exact snapshot indices for each decision timestamp."""
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D")
    if decision_ts.ndim != 1:
        raise ValueError("decision_ts must be 1D")
    if snapshot_ts.size and np.any(np.diff(snapshot_ts) <= 0):
        raise ValueError("snapshot_ts must be strictly increasing (np.diff(snapshot_ts) > 0)")
    if decision_ts.size and np.any(np.diff(decision_ts) < 0):
        raise ValueError("decision_ts must be monotonically non-decreasing (np.diff(decision_ts) >= 0)")

    aligned_idx = np.searchsorted(snapshot_ts, decision_ts, side="left")
    in_bounds = aligned_idx < snapshot_ts.shape[0]
    exact_match = np.zeros(decision_ts.shape[0], dtype=bool)
    exact_match[in_bounds] = snapshot_ts[aligned_idx[in_bounds]] == decision_ts[in_bounds]
    missing_mask = ~exact_match

    if np.any(missing_mask):
        missing = decision_ts[missing_mask]
        sample_count = min(5, int(missing.shape[0]))
        raise ValueError(
            "Snapshot alignment failed; exact timestamp matches missing. "
            f"missing={missing.shape[0]} total={decision_ts.size} "
            f"samples={missing[:sample_count].tolist()}. "
            "Regenerate canonical snapshots/tokens so decision timestamps are "
            "derived from order-book event time "
            "(decision_time_basis='ob_event_time', decision_policy='ob_event_time')."
        )
    return aligned_idx


def _resolve_horizon_indices(meta: dict, targets: Iterable[int]) -> Dict[int, int]:
    horizons = [int(h) for h in meta.get("horizons_ms", [])]
    if not horizons:
        raise ValueError("meta['horizons_ms'] must be non-empty")
    index_map = {h: idx for idx, h in enumerate(horizons)}
    missing = [h for h in targets if h not in index_map]
    if missing:
        raise ValueError(f"Requested horizons not in meta: {missing}")
    return {h: index_map[h] for h in targets}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def join_features(
    decision_ts: np.ndarray,
    y: np.ndarray,
    cmssl_out: Dict[str, np.ndarray],
    snapshots: np.ndarray,
    meta: dict,
) -> Dict[str, np.ndarray]:
    """Join model outputs and snapshot state into a single feature tensor.

    Per-row layout (excluding environment-only extra state):
      1) dir_logits[h]
      2) p_up[h]
      3) confidence/alignment scalars:
         - align_all
         - diff_short_long
         - diff_mid_long
         - conf_long
         - conf_min
      4) snapshot features from RAW_SNAPSHOT_FEATURE_COLUMNS
    """
    dir_logits = cmssl_out["dir_logits"]
    p_up = _sigmoid(dir_logits)
    horizons = [int(h) for h in meta.get("horizons_ms", [])]
    if not horizons:
        raise ValueError("meta['horizons_ms'] must be non-empty")
    sorted_horizons = sorted(set(horizons))
    short_h = sorted_horizons[0]
    long_h = sorted_horizons[-1]
    mid_h = sorted_horizons[len(sorted_horizons) // 2]
    horizon_idx = _resolve_horizon_indices(meta, targets=[short_h, mid_h, long_h])
    idx_short = horizon_idx[short_h]
    idx_mid = horizon_idx[mid_h]
    idx_long = horizon_idx[long_h]
    conf = np.abs(p_up - 0.5) * 2.0
    align_all = np.logical_or(
        np.all(p_up >= 0.5, axis=1),
        np.all(p_up <= 0.5, axis=1),
    ).astype(np.float32)
    diff_short_long = p_up[:, idx_short] - p_up[:, idx_long]
    diff_mid_long = p_up[:, idx_mid] - p_up[:, idx_long]
    conf_long = conf[:, idx_long]
    conf_min = np.min(conf, axis=1)
    layout = _joined_feature_layout(dir_logits.shape[1], snapshots.shape[1])
    snapshot_spread_col = RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")
    spread_bps = snapshots[:, snapshot_spread_col]  # use aligned snapshot spread
    if _env_bool("BYBIT_MM_PREALLOCATE_JOIN_FEATURES", False):
        n_rows = int(dir_logits.shape[0])
        expected_feature_dim = layout["snapshots"].stop
        features = np.empty((n_rows, expected_feature_dim), dtype=np.float32)
        cursor = 0

        d = dir_logits.shape[1]
        features[:, cursor:cursor + d] = dir_logits
        cursor += d
        d = p_up.shape[1]
        features[:, cursor:cursor + d] = p_up
        cursor += d

        features[:, cursor] = align_all
        cursor += 1
        features[:, cursor] = diff_short_long
        cursor += 1
        features[:, cursor] = diff_mid_long
        cursor += 1
        features[:, cursor] = conf_long
        cursor += 1
        features[:, cursor] = conf_min
        cursor += 1

        d = snapshots.shape[1]
        features[:, cursor:cursor + d] = snapshots
    else:
        features = np.concatenate(
            [
                dir_logits,
                p_up,
                align_all[:, None],
                diff_short_long[:, None],
                diff_mid_long[:, None],
                conf_long[:, None],
                conf_min[:, None],
                snapshots,
            ],
            axis=-1,
        )
    expected_feature_dim = layout["snapshots"].stop
    if features.shape[1] != expected_feature_dim:
        raise ValueError(
            "join_features layout mismatch: "
            f"features_shape={features.shape} expected_dim={expected_feature_dim}"
        )
    if not np.all(np.isfinite(features)):
        bad_rows = np.where(~np.isfinite(features).all(axis=1))[0]
        sample_rows = bad_rows[:5].tolist()
        raise ValueError(
            "join_features produced non-finite values in feature tensor. "
            f"bad_row_count={len(bad_rows)} sample_rows={sample_rows} features_shape={features.shape}"
        )
    output = {
        "ts": decision_ts,
        "features": features.astype(np.float32),
        "y": y.astype(np.float32),
        "spread_bps": spread_bps.astype(np.float32),
        "snapshots": snapshots.astype(np.float32),
    }
    return output


def build_joined_split(
    out_root: str,
    split: Dict[str, Any],
    model,
    meta: dict,
    device: str,
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    t0 = time.perf_counter()
    week_outputs: List[Dict[str, np.ndarray]] = []
    for wk in _split_weeks(split):
        wk_split = {"weeks": [wk], "start": split["start"], "end": split["end"]}
        try:
            x_core, x_aux, y, ts = load_split_arrays(out_root, wk_split)
        except ValueError as exc:
            if str(exc).startswith("No data found for split"):
                continue
            raise

        if ts.shape[0] == 0:
            continue

        cmssl_out = run_cmssl_inference(
            model,
            meta,
            x_core,
            x_aux,
            batch_size=batch_size,
            device=device,
        )

        # Canonical snapshot flow: load_raw_snapshots(...) ->
        # _compute_snapshot_feature_matrix(...).
        week_snapshot_ts, week_raw_snapshots, week_stale_ms = load_raw_snapshots(out_root, wk)
        snapshot_ts, snapshots = _compute_snapshot_feature_matrix(
            np.asarray(week_snapshot_ts, dtype=np.int64),
            np.asarray(week_raw_snapshots),
            None if week_stale_ms is None else np.asarray(week_stale_ms, dtype=np.float64),
        )
        snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
        snapshots = np.asarray(snapshots, dtype=np.float32)

        window_start = int(split["start"])
        window_end = int(split["end"])
        effective_mask = (snapshot_ts >= window_start) & (snapshot_ts < window_end)
        if np.any(effective_mask):
            snapshot_ts = snapshot_ts[effective_mask]
            snapshots = snapshots[effective_mask]

        # Perform exact-match decision/snapshot alignment week-by-week so split
        # boundaries follow the authoritative `weeks` ordering without cross-week
        # ambiguity at week edges.
        snap_idx = align_snapshots_to_decisions(snapshot_ts, ts)
        assert snapshot_ts[snap_idx].shape == ts.shape
        assert np.all(snapshot_ts[snap_idx] == ts)
        aligned_snapshots = snapshots[snap_idx]
        week_outputs.append(join_features(ts, y, cmssl_out, aligned_snapshots, meta))

    if not week_outputs:
        raise ValueError(f"No data found for split {split}")

    if _env_bool("BYBIT_MM_PREALLOCATE_JOIN_FEATURES", False):
        total_rows = int(sum(wk["ts"].shape[0] for wk in week_outputs))
        feature_dim = int(week_outputs[0]["features"].shape[1])
        y_dim = int(week_outputs[0]["y"].shape[1])
        snapshot_dim = int(week_outputs[0]["snapshots"].shape[1])

        out = {
            "ts": np.empty((total_rows,), dtype=week_outputs[0]["ts"].dtype),
            "features": np.empty((total_rows, feature_dim), dtype=week_outputs[0]["features"].dtype),
            "y": np.empty((total_rows, y_dim), dtype=week_outputs[0]["y"].dtype),
            "spread_bps": np.empty((total_rows,), dtype=week_outputs[0]["spread_bps"].dtype),
            "snapshots": np.empty((total_rows, snapshot_dim), dtype=week_outputs[0]["snapshots"].dtype),
        }
        cursor = 0
        for wk in week_outputs:
            rows = int(wk["ts"].shape[0])
            end = cursor + rows
            out["ts"][cursor:end] = wk["ts"]
            out["features"][cursor:end] = wk["features"]
            out["y"][cursor:end] = wk["y"]
            out["spread_bps"][cursor:end] = wk["spread_bps"]
            out["snapshots"][cursor:end] = wk["snapshots"]
            cursor = end
    else:
        out = {
            "ts": np.concatenate([wk["ts"] for wk in week_outputs], axis=0),
            "features": np.concatenate([wk["features"] for wk in week_outputs], axis=0),
            "y": np.concatenate([wk["y"] for wk in week_outputs], axis=0),
            "spread_bps": np.concatenate([wk["spread_bps"] for wk in week_outputs], axis=0),
            "snapshots": np.vstack([wk["snapshots"] for wk in week_outputs]),
        }

    expected_rows = out["ts"].shape[0]
    for key, value in out.items():
        if value.shape[0] != expected_rows:
            raise ValueError(
                "build_joined_split row-count mismatch after weekly concatenation: "
                f"ts_rows={expected_rows} {key}_rows={value.shape[0]}"
            )

    ts_all = out["ts"]
    ts_diff = np.diff(ts_all)
    bad_idx = np.where(ts_diff <= 0)[0]
    if bad_idx.size > 0:
        first_bad = int(bad_idx[0])
        raise ValueError(
            "build_joined_split requires strictly increasing concatenated timestamps "
            "(weeks order is preserved; outputs are not resorted). "
            f"first_bad_index={first_bad} ts_prev={int(ts_all[first_bad])} "
            f"ts_next={int(ts_all[first_bad + 1])} diff={int(ts_diff[first_bad])}"
        )

    _timing_log(f"build_joined_split rows={out['ts'].shape[0]} secs={time.perf_counter() - t0:.4f}")
    return out


def slice_joined_by_split(data: Dict[str, np.ndarray], split_def: Dict[str, Any]) -> Dict[str, np.ndarray]:
    start = int(split_def["start"])
    end = int(split_def["end"])
    ts = np.asarray(data["ts"], dtype=np.int64)
    mask = (ts >= start) & (ts < end)
    if not np.any(mask):
        raise ValueError(f"No joined rows found for split range [{start}, {end}).")
    sliced = {key: value[mask] for key, value in data.items()}
    sliced_ts = np.asarray(sliced["ts"], dtype=np.int64)
    if sliced_ts[0] < start or sliced_ts[-1] >= end:
        raise ValueError("slice_joined_by_split produced rows outside requested timestamp bounds.")
    return sliced


@dataclass
class MarketMakingBatch:
    features: np.ndarray
    spread_bps: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    future_ret_by_horizon: Optional[np.ndarray] = None
    decision_ts: Optional[np.ndarray] = None


class MarketMakingEnv:
    def __init__(
        self,
        batch: MarketMakingBatch,
        *,
        maker_rebate_bps: float = 0.0,
        taker_fee_bps: float = DEFAULT_MM_TAKER_FEE_BPS,
        allow_taker: bool = True,
        taker_threshold: float = DEFAULT_MM_TAKER_THRESHOLD,
        inventory_penalty: float = 0.0,
        inv_soft_notional: float,
        lambda_inv: float = 0.0,
        lambda_turn: float = 0.0,
        max_inventory_notional: float,
        hard_max_inventory_notional: Optional[float] = None,
        fill_size: float = 1.0,
        fill_tolerance: float = 1e-6,
        delta_bps_limit: float = 0.0,
        initial_cash: Optional[float] = None,
        obs_norm_state: Optional[dict] = None,
        freeze_obs_norm: bool = False,
        baseline_alpha_calibration: Optional[BaselineAlphaCalibration] = None,
        baseline_quote_config: Optional[BaselineQuoteConfig] = None,
    ):
        self.features = batch.features
        self.spread_bps = batch.spread_bps
        self.best_bid = batch.best_bid
        self.best_ask = batch.best_ask
        self.decision_ts = batch.decision_ts
        self.maker_rebate_bps = maker_rebate_bps
        self.taker_fee_bps = taker_fee_bps
        self.allow_taker = allow_taker
        self.taker_threshold = taker_threshold
        self.inventory_penalty = inventory_penalty
        if inv_soft_notional <= 0.0:
            raise ValueError("inv_soft_notional must be > 0 in quote notional (USD).")
        if max_inventory_notional <= 0.0:
            raise ValueError("max_inventory_notional must be > 0 in quote notional (USD).")
        if max_inventory_notional < inv_soft_notional:
            raise ValueError("max_inventory_notional must be >= inv_soft_notional.")
        self.inv_soft_notional = inv_soft_notional
        self.lambda_inv = lambda_inv
        self.lambda_turn = lambda_turn
        self.stack_inventory_penalties = _env_bool("BYBIT_MM_STACK_INVENTORY_PENALTIES", False)
        self.max_inventory_notional = max_inventory_notional
        # Hard cap (if explicitly set) must be >= soft maker/taker control cap.
        self.hard_max_inventory_notional = (
            float(hard_max_inventory_notional)
            if hard_max_inventory_notional is not None
            else float(max_inventory_notional)
        )
        if (
            not np.isfinite(self.hard_max_inventory_notional)
            or self.hard_max_inventory_notional <= 0.0
        ):
            raise ValueError(
                "hard_max_inventory_notional must be finite and > 0 in quote notional (USD)."
            )
        if self.hard_max_inventory_notional < self.max_inventory_notional:
            raise ValueError(
                "hard_max_inventory_notional must be >= max_inventory_notional."
            )
        self.fill_size = fill_size
        self.fill_tolerance = fill_tolerance
        self.delta_bps_limit = float(delta_bps_limit)
        if not np.isfinite(self.delta_bps_limit) or self.delta_bps_limit <= 0.0:
            raise ValueError(
                "delta_bps_limit must be finite and > 0 in basis points (bps)."
            )
        self.initial_cash = (
            float(initial_cash)
            if initial_cash is not None
            else _env_float("BYBIT_MM_INITIAL_CASH", DEFAULT_MM_INITIAL_CASH)
        )
        inventory_notional_scale_raw = os.environ.get(
            "BYBIT_MM_INVENTORY_NOTIONAL_SCALE",
            "",
        ).strip()
        if not inventory_notional_scale_raw:
            raise ValueError(
                "Missing required env var BYBIT_MM_INVENTORY_NOTIONAL_SCALE "
                "(quote notional, USD)."
            )
        try:
            self.inventory_notional_scale = float(inventory_notional_scale_raw)
        except ValueError as exc:
            raise ValueError(
                "BYBIT_MM_INVENTORY_NOTIONAL_SCALE must be a finite float "
                "in quote notional (USD)."
            ) from exc
        if not np.isfinite(self.inventory_notional_scale) or self.inventory_notional_scale <= 0.0:
            raise ValueError(
                "BYBIT_MM_INVENTORY_NOTIONAL_SCALE must be finite and > 0 "
                "in quote notional (USD)."
            )
        self.cash_scale = _env_float("BYBIT_MM_CASH_SCALE", DEFAULT_MM_CASH_SCALE)
        self.time_since_fill_scale = _env_float(
            "BYBIT_MM_TIME_SINCE_FILL_SCALE",
            DEFAULT_MM_TIME_SINCE_FILL_SCALE,
        )
        self.fill_notional_scale = _env_float(
            "BYBIT_MM_FILL_NOTIONAL_SCALE",
            DEFAULT_MM_FILL_NOTIONAL_SCALE,
        )
        self.pnl_notional_scale = _env_float(
            "BYBIT_MM_PNL_NOTIONAL_SCALE",
            DEFAULT_MM_PNL_NOTIONAL_SCALE,
        )
        self.markout_notional_scale = _env_float(
            "BYBIT_MM_MARKOUT_NOTIONAL_SCALE",
            DEFAULT_MM_MARKOUT_NOTIONAL_SCALE,
        )
        self.fill_ema_window_steps = max(
            1,
            _env_int("BYBIT_MM_FILL_EMA_WINDOW_STEPS", DEFAULT_MM_FILL_EMA_WINDOW_STEPS),
        )
        self.fill_ema_alpha = 2.0 / (float(self.fill_ema_window_steps) + 1.0)
        self.baseline_quote_config = _validate_baseline_quote_config(
            load_baseline_quote_config() if baseline_quote_config is None else baseline_quote_config
        )
        self._num_h = _infer_num_horizons(self.features.shape[-1])
        self._horizons_ms = _normalize_horizons(
            self._num_h,
            self.baseline_quote_config.horizons_ms,
        )
        self._horizons_ms = _validate_fixed_cmssl_horizons(self._horizons_ms)
        self._p250_idx = _resolve_horizon_index(
            250,
            self._horizons_ms,
            label="p250",
        )
        self._p500_idx = _resolve_horizon_index(
            500,
            self._horizons_ms,
            label="p500",
        )
        self._p1000_idx = _resolve_horizon_index(
            1000,
            self._horizons_ms,
            label="p1000",
        )
        print(
            "[mm horizons]",
            f"resolved_horizons_ms={self._horizons_ms}",
            f"p250_idx={self._p250_idx}",
            f"p500_idx={self._p500_idx}",
            f"p1000_idx={self._p1000_idx}",
        )
        self._baseline_alpha_calibration = baseline_alpha_calibration
        if self._baseline_alpha_calibration is None:
            raise ValueError("baseline_alpha_calibration must be provided")
        self._score_to_bps_slope_250 = float(
            self._baseline_alpha_calibration.score_to_bps_slope_by_horizon[self._p250_idx]
        )
        self._score_to_bps_slope_500 = float(
            self._baseline_alpha_calibration.score_to_bps_slope_by_horizon[self._p500_idx]
        )
        self._score_to_bps_slope_1000 = float(
            self._baseline_alpha_calibration.score_to_bps_slope_by_horizon[self._p1000_idx]
        )
        self._feature_layout = _joined_feature_layout(self._num_h, len(RAW_SNAPSHOT_FEATURE_COLUMNS))
        self._validate_feature_layout()

        self.n = len(self.spread_bps)
        self.idx = 0
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.total_reward = 0.0
        self.prev_equity = self.initial_cash
        self.time_since_last_fill = self._initial_time_since_last_fill()
        self.avg_entry_price = 0.0
        self.last_maker_buy_notional = 0.0
        self.last_maker_sell_notional = 0.0
        self.last_taker_buy_notional = 0.0
        self.last_taker_sell_notional = 0.0
        self.last_net_fill_notional = 0.0
        self.last_gross_fill_notional = 0.0
        self.ema_net_fill_notional = 0.0
        self.ema_gross_fill_notional = 0.0
        self.ema_maker_buy_markout = 0.0
        self.ema_maker_sell_markout = 0.0
        self.last_maker_buy_clipped = 0.0
        self.last_maker_sell_clipped = 0.0
        self.last_taker_buy_clipped = 0.0
        self.last_taker_sell_clipped = 0.0
        self._obs_count = 0
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_m2: Optional[np.ndarray] = None
        self._obs_continuous_mask: Optional[np.ndarray] = None
        self.freeze_obs_norm = bool(freeze_obs_norm)
        if obs_norm_state is not None:
            self.set_obs_norm_state(obs_norm_state, freeze=freeze_obs_norm)

    def reset(self, start_idx: int = 0) -> np.ndarray:
        max_start = max(0, self.n - 2)
        if start_idx < 0 or start_idx > max_start:
            raise ValueError(
                f"start_idx out of bounds: start_idx={start_idx} valid=[0, {max_start}]"
            )
        self.idx = int(start_idx)
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.total_reward = 0.0
        # Episode startup semantics: no prior fill is represented by a large sentinel
        # so the feature is distinct from "just filled" (0.0).
        self.time_since_last_fill = self._initial_time_since_last_fill()
        self.avg_entry_price = 0.0
        self.last_maker_buy_notional = 0.0
        self.last_maker_sell_notional = 0.0
        self.last_taker_buy_notional = 0.0
        self.last_taker_sell_notional = 0.0
        self.last_net_fill_notional = 0.0
        self.last_gross_fill_notional = 0.0
        self.ema_net_fill_notional = 0.0
        self.ema_gross_fill_notional = 0.0
        self.ema_maker_buy_markout = 0.0
        self.ema_maker_sell_markout = 0.0
        self.last_maker_buy_clipped = 0.0
        self.last_maker_sell_clipped = 0.0
        self.last_taker_buy_clipped = 0.0
        self.last_taker_sell_clipped = 0.0
        mid = self._mid_price(self.idx)
        self.prev_equity = self.cash + self.inventory * mid
        return self._build_observation(self.idx)

    def _mid_price(self, idx: int) -> float:
        return float((self.best_bid[idx] + self.best_ask[idx]) / 2.0)

    def _initial_time_since_last_fill(self) -> float:
        # Prefer a startup value near 1.0 after scaling. This signals "no fill yet"
        # at episode start while keeping 0.0 reserved for a fresh fill event.
        if self.time_since_fill_scale > 0.0:
            return float(self.time_since_fill_scale)
        return 1.0

    def _build_observation(self, idx: int) -> np.ndarray:
        mid = self._mid_price(idx)
        inventory_notional_scaled = (
            (self.inventory * mid) / self.inventory_notional_scale if self.inventory_notional_scale else 0.0
        )
        cash_scaled = self.cash / self.cash_scale if self.cash_scale else 0.0
        time_since_last_fill_scaled = (
            self.time_since_last_fill / self.time_since_fill_scale if self.time_since_fill_scale else 0.0
        )
        unrealized_pnl_notional = (
            self.inventory * (mid - self.avg_entry_price) if self.inventory != 0.0 else 0.0
        )
        unrealized_pnl_scaled = (
            unrealized_pnl_notional / self.pnl_notional_scale if self.pnl_notional_scale else 0.0
        )
        # Fill-notional `last_*` fields capture the last non-zero fill aggregates.
        # At reset, `time_since_last_fill` starts at a sentinel for "no prior fill"
        # (scaled value ~1.0). A real fill sets it to 0.0. On no-fill steps, it is
        # incremented by (decision_ts[next_idx] - decision_ts[idx]) / RAW_SNAPSHOT_EXPECTED_STEP_MS,
        # i.e., accumulated in RAW_SNAPSHOT_EXPECTED_STEP_MS-equivalent units rather
        # than fixed "1 snapshot == 1 step" units. Under jitter this keeps intent
        # explicit: ~100ms gaps contribute ~1.0, ~300ms gaps contribute ~3.0.
        # `last_*` values persist on no-fill steps.
        extra = np.array(
            [
                inventory_notional_scaled,
                cash_scaled,
                time_since_last_fill_scaled,
                self.last_maker_buy_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_maker_sell_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_taker_buy_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_taker_sell_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_net_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_gross_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.ema_net_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.ema_gross_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                unrealized_pnl_scaled,
                self.ema_maker_buy_markout / self.markout_notional_scale if self.markout_notional_scale else 0.0,
                self.ema_maker_sell_markout / self.markout_notional_scale if self.markout_notional_scale else 0.0,
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([self.features[idx].astype(np.float32), extra], axis=0)
        return self._normalize_observation(obs)

    def _validate_feature_layout(self) -> None:
        expected_feature_dim = self._feature_layout["snapshots"].stop
        actual_feature_dim = int(self.features.shape[-1])
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                "MarketMakingEnv feature layout drift detected: "
                f"actual_feature_dim={actual_feature_dim} expected_feature_dim={expected_feature_dim} "
                f"num_horizons={self._num_h} snapshot_dim={len(RAW_SNAPSHOT_FEATURE_COLUMNS)}"
            )

    def get_observation_scaling_config(self) -> Dict[str, float]:
        return {
            "inventory_notional_scale": float(self.inventory_notional_scale),
            "cash_scale": float(self.cash_scale),
            "time_since_fill_scale": float(self.time_since_fill_scale),
            "fill_notional_scale": float(self.fill_notional_scale),
            "pnl_notional_scale": float(self.pnl_notional_scale),
            "markout_notional_scale": float(self.markout_notional_scale),
            "fill_ema_window_steps": int(self.fill_ema_window_steps),
        }

    def _continuous_mask(self, obs_dim: int) -> np.ndarray:
        expected_obs_dim = self._feature_layout["snapshots"].stop + ENV_OBS_EXTRA_STATE_DIM
        if obs_dim != expected_obs_dim:
            raise ValueError(
                "Observation dimension mismatch for normalization mask: "
                f"obs_dim={obs_dim} expected_obs_dim={expected_obs_dim}"
            )
        mask = np.ones(obs_dim, dtype=bool)
        bounded_feature_keys = (
            "p_up",
            "align_all",
            "conf_long",
            "conf_min",
        )
        for key in bounded_feature_keys:
            feature_slice = self._feature_layout[key]
            mask[feature_slice] = False
        return mask

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        if self._obs_mean is None or self._obs_m2 is None:
            self._obs_mean = np.zeros_like(obs, dtype=np.float64)
            self._obs_m2 = np.zeros_like(obs, dtype=np.float64)
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_m2 += delta * delta2

    def get_obs_norm_state(self) -> Dict[str, Any]:
        return {
            "count": int(self._obs_count),
            "mean": self._obs_mean.astype(np.float64).tolist() if self._obs_mean is not None else None,
            "m2": self._obs_m2.astype(np.float64).tolist() if self._obs_m2 is not None else None,
            "continuous_mask": (
                self._obs_continuous_mask.astype(bool).tolist()
                if self._obs_continuous_mask is not None
                else None
            ),
        }

    def has_obs_norm_state(self) -> bool:
        return _obs_norm_state_is_ready(self.get_obs_norm_state())

    def set_obs_norm_state(self, state: Dict[str, Any], freeze: bool = True) -> None:
        if not isinstance(state, dict):
            raise ValueError("obs normalization state must be a dictionary.")
        count = int(state.get("count", 0))
        mean_raw = state.get("mean")
        m2_raw = state.get("m2")
        mask_raw = state.get("continuous_mask")
        mean = None if mean_raw is None else np.asarray(mean_raw, dtype=np.float64)
        m2 = None if m2_raw is None else np.asarray(m2_raw, dtype=np.float64)
        mask = None if mask_raw is None else np.asarray(mask_raw, dtype=bool)
        if count < 0:
            raise ValueError(f"obs normalization count must be non-negative, got {count}")
        if (mean is None) ^ (m2 is None):
            raise ValueError("obs normalization state must provide both mean and m2 or neither.")
        if mean is not None and mean.shape != m2.shape:
            raise ValueError(
                f"obs normalization mean/m2 shape mismatch: mean={mean.shape} m2={m2.shape}"
            )
        if mask is not None and mean is not None and mask.shape != mean.shape:
            raise ValueError(
                "obs normalization continuous_mask shape mismatch: "
                f"mask={mask.shape} mean={mean.shape}"
            )
        if count == 0:
            mean = None
            m2 = None
        self._obs_count = count
        self._obs_mean = mean
        self._obs_m2 = m2
        self._obs_continuous_mask = mask
        self.freeze_obs_norm = bool(freeze)

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_continuous_mask is None:
            self._obs_continuous_mask = self._continuous_mask(obs.shape[0])
        normalized = obs.copy()
        if self._obs_count >= 2 and self._obs_mean is not None and self._obs_m2 is not None:
            var = self._obs_m2 / max(self._obs_count - 1, 1)
            std = np.sqrt(np.maximum(var, 1e-6))
            mask = self._obs_continuous_mask
            normalized[mask] = (obs[mask] - self._obs_mean[mask]) / std[mask]
        if not self.freeze_obs_norm:
            self._update_obs_stats(obs)
        return normalized

    def _parse_action(self, action: Any) -> Tuple[float, float, float]:
        """Parse an action into (bid_delta_bps, ask_delta_bps, taker_signal).

        Accepted action formats:
        - Scalar: applies the same delta to bid and ask, with no taker signal.
        - Length-2 sequence: interpreted as (bid_delta_bps, ask_delta_bps), taker=0.
        - Length-3 sequence: interpreted as (bid_delta_bps, ask_delta_bps, taker_signal).
        """
        bid_delta_bps: float
        ask_delta_bps: float
        taker_signal: float

        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) == 3:
                bid_delta_bps = float(action[0])
                ask_delta_bps = float(action[1])
                taker_signal = float(action[2])
            elif len(action) == 2:
                bid_delta_bps = float(action[0])
                ask_delta_bps = float(action[1])
                taker_signal = 0.0
            else:
                raise ValueError(
                    "Action sequence must be length 2 or 3: "
                    "(bid_delta_bps, ask_delta_bps[, taker_signal])."
                )
        elif np.isscalar(action):
            bid_delta_bps = float(action)
            ask_delta_bps = float(action)
            taker_signal = 0.0
        else:
            raise ValueError(
                "Action must be a scalar or (bid_delta_bps, ask_delta_bps[, taker_signal])."
            )

        if not np.all(np.isfinite([bid_delta_bps, ask_delta_bps, taker_signal])):
            raise ValueError(
                "Action components must be finite: "
                f"bid_delta_bps={bid_delta_bps}, ask_delta_bps={ask_delta_bps}, "
                f"taker_signal={taker_signal}"
            )
        return bid_delta_bps, ask_delta_bps, taker_signal

    def _feature_slice(self, idx: int, start: int, end: int) -> np.ndarray:
        return self.features[idx, start:end]

    @staticmethod
    def _width_from_market_state(
        cfg: BaselineQuoteConfig,
        observed_spread_bps: float,
        vol_short: float,
        vol_long: float,
        p250: float,
        p500: float,
        p1000: float,
    ) -> float:
        obs_anchor_bps = 0.0
        if np.isfinite(observed_spread_bps) and observed_spread_bps > 0.0:
            obs_anchor_bps = max(0.0, cfg.obs_spread_anchor_frac * observed_spread_bps)
        sigma_bps = 1e4 * max(0.0, vol_short, vol_long)
        disagreement = max(abs(p250 - p500), abs(p500 - p1000), abs(p250 - p1000))
        uncertainty_ref_bps = max(cfg.base_half_spread_bps, obs_anchor_bps, sigma_bps, cfg.spread_floor_bps)
        base_component_bps = max(cfg.base_half_spread_bps, obs_anchor_bps, cfg.spread_floor_bps)
        vol_width_bps = cfg.vol_width_scale * sigma_bps
        uncertainty_width_bps = cfg.uncertainty_width_scale * disagreement * uncertainty_ref_bps
        half_spread_bps_raw = base_component_bps + vol_width_bps + uncertainty_width_bps
        return float(np.clip(half_spread_bps_raw, cfg.spread_floor_bps, cfg.spread_cap_bps))

    def _baseline_quotes(self, idx: int) -> Tuple[float, float, float]:
        cfg = self.baseline_quote_config
        mid = self._mid_price(idx)
        p_up = self._feature_slice(idx, self._feature_layout["p_up"].start, self._feature_layout["p_up"].stop)
        p250 = float(p_up[self._p250_idx])
        p500 = float(p_up[self._p500_idx])
        p1000 = float(p_up[self._p1000_idx])
        score_250 = 2.0 * p250 - 1.0
        score_500 = 2.0 * p500 - 1.0
        score_1000 = 2.0 * p1000 - 1.0
        alpha_center_bps = cfg.alpha_center_scale * (
            cfg.p250_weight * self._score_to_bps_slope_250 * score_250
            + cfg.p500_weight * self._score_to_bps_slope_500 * score_500
            + cfg.p1000_weight * self._score_to_bps_slope_1000 * score_1000
        )
        inv_notional = self.inventory * mid
        inventory_center_bps = cfg.inventory_center_scale * (inv_notional / cfg.inv_ref_notional)
        reservation_bps = alpha_center_bps - inventory_center_bps
        snapshot_row = self.features[idx, self._feature_layout["snapshots"]]
        observed_spread_bps = float(snapshot_row[RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")])
        vol_short = float(snapshot_row[RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_short")])
        vol_long = float(snapshot_row[RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_long")])
        half_spread_bps = self._width_from_market_state(cfg, observed_spread_bps, vol_short, vol_long, p250, p500, p1000)
        eps = 1e-12
        soft_ratio = abs(inv_notional) / max(self.inv_soft_notional, eps)
        overload = max(0.0, soft_ratio - 1.0)
        inventory_side_extra_bps = (
            cfg.inventory_side_widen_scale * overload * max(half_spread_bps, 1.0)
        )
        bid_half_spread_bps = half_spread_bps + (inventory_side_extra_bps if inv_notional > 0.0 else 0.0)
        ask_half_spread_bps = half_spread_bps + (inventory_side_extra_bps if inv_notional < 0.0 else 0.0)
        bid_enabled = inv_notional < self.max_inventory_notional
        ask_enabled = inv_notional > -self.max_inventory_notional
        bid = mid + bps_to_px(mid, reservation_bps - bid_half_spread_bps) if bid_enabled else np.nan
        ask = mid + bps_to_px(mid, reservation_bps + ask_half_spread_bps) if ask_enabled else np.nan
        return bid, ask, mid

    def _apply_deltas(
        self, bid: float, ask: float, mid: float, bid_delta_bps: float, ask_delta_bps: float
    ) -> Tuple[float, float, float, float]:
        bid_delta_bps = float(np.clip(bid_delta_bps, -self.delta_bps_limit, self.delta_bps_limit))
        ask_delta_bps = float(np.clip(ask_delta_bps, -self.delta_bps_limit, self.delta_bps_limit))
        if np.isfinite(bid):
            bid += mid * bid_delta_bps * 1e-4
        if np.isfinite(ask):
            ask += mid * ask_delta_bps * 1e-4
        return bid, ask, bid_delta_bps, ask_delta_bps

    def _enforce_passive(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        mid = 0.5 * (best_bid + best_ask)
        eps = max(1e-8, mid * 1e-6)
        if np.isfinite(bid):
            bid = min(bid, best_ask - eps)
        if np.isfinite(ask):
            ask = max(ask, best_bid + eps)
        if np.isfinite(bid) and np.isfinite(ask) and bid >= ask:
            bid = best_bid
            ask = best_ask
        return bid, ask

    def _inventory_cap_qty(self, mid: float) -> float:
        # Threshold contract: hard_max_inventory_notional is only for execution-time
        # fill clipping; _compute_penalty uses max_inventory_notional as the soft trigger.
        return self.hard_max_inventory_notional / max(mid, 1e-12)

    def _remaining_inventory_room(self, side: int, mid: float) -> float:
        cap_qty = self._inventory_cap_qty(mid)
        if side > 0:
            return max(0.0, cap_qty - self.inventory)
        if side < 0:
            return max(0.0, cap_qty + self.inventory)
        raise ValueError("side must be +1 (buy) or -1 (sell).")

    def _clip_fill_qty(self, side: int, requested_qty: float, mid: float) -> float:
        if requested_qty <= 0.0:
            return 0.0
        room_qty = self._remaining_inventory_room(side, mid)
        return float(min(requested_qty, room_qty))

    def _apply_signed_fill(self, side: int, qty: float, price: float) -> float:
        if qty <= 0.0:
            return 0.0
        signed_qty = float(side) * float(qty)
        self.cash -= price * signed_qty
        self.inventory += signed_qty
        return float(qty)

    def _apply_fills(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        touch_epsilon = 1e-9
        best_bid_next = float(self.best_bid[idx])
        best_ask_next = float(self.best_ask[idx])
        best_bid_prev = float(self.best_bid[idx - 1]) if idx > 0 else best_bid_next
        best_ask_prev = float(self.best_ask[idx - 1]) if idx > 0 else best_ask_next
        buy_fill = 0.0
        sell_fill = 0.0
        self.last_maker_buy_clipped = 0.0
        self.last_maker_sell_clipped = 0.0
        # Hard inventory cap is defined in quote notional and converted to base qty using midpoint for symmetric long/short treatment.
        mid_for_cap = self._mid_price(idx)
        bid_enabled = np.isfinite(bid)
        ask_enabled = np.isfinite(ask)
        # Evaluate fills against the next snapshot's opposite side.
        if bid_enabled and best_ask_next <= bid + self.fill_tolerance:
            requested_buy = self.fill_size
            clipped_buy = self._clip_fill_qty(1, requested_buy, mid_for_cap)
            self.last_maker_buy_clipped = requested_buy - clipped_buy
            buy_fill = self._apply_signed_fill(1, clipped_buy, bid)
        # Keep deterministic buy-then-sell processing; second fill sees updated inventory.
        if ask_enabled and best_bid_next >= ask - self.fill_tolerance:
            requested_sell = self.fill_size
            clipped_sell = self._clip_fill_qty(-1, requested_sell, mid_for_cap)
            self.last_maker_sell_clipped = requested_sell - clipped_sell
            sell_fill = self._apply_signed_fill(-1, clipped_sell, ask)
        # Heuristic: if we're at the touch and the next best moves away, we got hit.
        touch_tolerance = max(self.fill_tolerance, touch_epsilon)
        if bid_enabled and buy_fill == 0.0 and abs(bid - best_bid_prev) <= touch_tolerance:
            if best_bid_next < best_bid_prev - touch_epsilon:
                requested_buy = self.fill_size
                clipped_buy = self._clip_fill_qty(1, requested_buy, mid_for_cap)
                self.last_maker_buy_clipped = requested_buy - clipped_buy
                buy_fill = self._apply_signed_fill(1, clipped_buy, bid)
        if ask_enabled and sell_fill == 0.0 and abs(ask - best_ask_prev) <= touch_tolerance:
            if best_ask_next > best_ask_prev + touch_epsilon:
                requested_sell = self.fill_size
                clipped_sell = self._clip_fill_qty(-1, requested_sell, mid_for_cap)
                self.last_maker_sell_clipped = requested_sell - clipped_sell
                sell_fill = self._apply_signed_fill(-1, clipped_sell, ask)
        return buy_fill, sell_fill

    def _apply_taker(self, idx: int, taker_signal: float) -> Tuple[float, float]:
        self.last_taker_buy_clipped = 0.0
        self.last_taker_sell_clipped = 0.0
        if not self.allow_taker:
            return 0.0, 0.0
        if abs(taker_signal) < self.taker_threshold:
            return 0.0, 0.0
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        # Hard inventory cap is defined in quote notional and converted to base qty using midpoint for symmetric long/short treatment.
        mid_for_cap = self._mid_price(idx)
        buy_fill = 0.0
        sell_fill = 0.0
        if taker_signal > 0.0:
            requested_buy = self.fill_size
            clipped_buy = self._clip_fill_qty(1, requested_buy, mid_for_cap)
            self.last_taker_buy_clipped = requested_buy - clipped_buy
            buy_fill = self._apply_signed_fill(1, clipped_buy, best_ask)
        elif taker_signal < 0.0:
            requested_sell = self.fill_size
            clipped_sell = self._clip_fill_qty(-1, requested_sell, mid_for_cap)
            self.last_taker_sell_clipped = requested_sell - clipped_sell
            sell_fill = self._apply_signed_fill(-1, clipped_sell, best_bid)
        return buy_fill, sell_fill


    def _next_avg_entry_price(
        self,
        prev_inv: float,
        prev_avg: float,
        side: int,
        qty: float,
        price: float,
    ) -> Tuple[float, float]:
        if qty <= 0.0:
            return prev_avg, prev_inv
        signed_qty = float(side) * qty
        new_inv = prev_inv + signed_qty
        if prev_inv == 0.0 or np.sign(prev_inv) == np.sign(signed_qty):
            base_qty = abs(prev_inv)
            total_qty = base_qty + qty
            next_avg = (prev_avg * base_qty + price * qty) / total_qty if total_qty > 0.0 else 0.0
            return float(next_avg), float(new_inv)
        # Trade is against existing position.
        if abs(signed_qty) < abs(prev_inv):
            return float(prev_avg), float(new_inv)
        if abs(signed_qty) == abs(prev_inv):
            return 0.0, 0.0
        return float(price), float(new_inv)

    def _ema_update(self, prev: float, value: float) -> float:
        return (1.0 - self.fill_ema_alpha) * prev + self.fill_ema_alpha * value

    def _compute_penalty(self, mid: float) -> float:
        # Linear inventory penalty trigger uses max_inventory_notional (soft/penalty
        # threshold, quote notional USD), not the hard execution clipping cap.
        inv_notional = abs(self.inventory * mid)
        penalty = 0.0
        if inv_notional > self.max_inventory_notional:
            penalty += self.inventory_penalty * (inv_notional - self.max_inventory_notional)
        return penalty

    def _combine_inventory_penalties(self, linear_penalty: float, quadratic_penalty: float) -> float:
        # Both terms penalize inventory risk. Default to non-stacking behavior to avoid
        # double-charging the same exposure unless explicitly enabled.
        if self.stack_inventory_penalties:
            return linear_penalty + quadratic_penalty
        return max(linear_penalty, quadratic_penalty)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        # Execution convention: both maker and taker fills are priced using the next snapshot
        # (next_idx). We quote on self.idx, then advance state after applying fills at next_idx.
        next_idx = self.idx + 1
        if next_idx >= self.n:
            mid = self._mid_price(self.idx)
            hard_cap_qty = self._inventory_cap_qty(mid)
            pre_hard_cap_qty = hard_cap_qty
            pre_buy_room_qty = self._remaining_inventory_room(1, mid)
            pre_sell_room_qty = self._remaining_inventory_room(-1, mid)
            post_hard_cap_qty = hard_cap_qty
            post_buy_room_qty = pre_buy_room_qty
            post_sell_room_qty = pre_sell_room_qty
            equity = self.cash + self.inventory * mid
            info = {
                "reward": 0.0,
                "total_reward": float(self.total_reward),
                "cash": float(self.cash),
                "inventory": float(self.inventory),
                "inventory_notional": float(abs(self.inventory * mid)),
                "equity": float(equity),
                "delta_equity": 0.0,
                "rebate": 0.0,
                "penalty": 0.0,
                "inv_penalty": 0.0,
                "inventory_excess_notional": 0.0,
                "turnover_penalty": 0.0,
                "mid": float(mid),
                "hard_max_inventory_notional": float(self.hard_max_inventory_notional),
                "pre_hard_cap_qty": float(pre_hard_cap_qty),
                "pre_buy_room_qty": float(pre_buy_room_qty),
                "pre_sell_room_qty": float(pre_sell_room_qty),
                "post_hard_cap_qty": float(post_hard_cap_qty),
                "post_buy_room_qty": float(post_buy_room_qty),
                "post_sell_room_qty": float(post_sell_room_qty),
                "bid": 0.0,
                "ask": 0.0,
                "maker_buy": 0.0,
                "maker_sell": 0.0,
                "taker_buy": 0.0,
                "taker_sell": 0.0,
                "taker_fee": 0.0,
                "maker_buy_notional": float(self.last_maker_buy_notional),
                "maker_sell_notional": float(self.last_maker_sell_notional),
                "taker_buy_notional": float(self.last_taker_buy_notional),
                "taker_sell_notional": float(self.last_taker_sell_notional),
                "net_fill_notional": float(self.last_net_fill_notional),
                "gross_fill_notional": float(self.last_gross_fill_notional),
                "ema_net_fill_notional": float(self.ema_net_fill_notional),
                "ema_gross_fill_notional": float(self.ema_gross_fill_notional),
                "avg_entry_price": float(self.avg_entry_price),
                "unrealized_pnl_notional": float(self.inventory * (mid - self.avg_entry_price) if self.inventory != 0.0 else 0.0),
                "maker_buy_markout": 0.0,
                "maker_sell_markout": 0.0,
                "ema_maker_buy_markout": float(self.ema_maker_buy_markout),
                "ema_maker_sell_markout": float(self.ema_maker_sell_markout),
                "maker_buy_clipped": float(self.last_maker_buy_clipped),
                "maker_sell_clipped": float(self.last_maker_sell_clipped),
                "taker_buy_clipped": float(self.last_taker_buy_clipped),
                "taker_sell_clipped": float(self.last_taker_sell_clipped),
            }
            return self._build_observation(self.idx), 0.0, True, info
        bid_delta_bps, ask_delta_bps, taker_signal = self._parse_action(action)
        bid, ask, mid = self._baseline_quotes(self.idx)
        bid, ask, bid_delta_bps, ask_delta_bps = self._apply_deltas(
            bid, ask, mid, bid_delta_bps, ask_delta_bps
        )
        bid, ask = self._enforce_passive(bid, ask, self.idx)
        inv_prev = self.inventory
        mid_for_cap = self._mid_price(next_idx)
        pre_hard_cap_qty = self._inventory_cap_qty(mid_for_cap)
        pre_buy_room_qty = self._remaining_inventory_room(1, mid_for_cap)
        pre_sell_room_qty = self._remaining_inventory_room(-1, mid_for_cap)
        # Clipping is evaluated per fill attempt, so maker/taker clipped amounts reflect
        # evolving inventory after each in-step fill is applied.
        maker_buy, maker_sell = self._apply_fills(bid, ask, next_idx)
        taker_buy, taker_sell = self._apply_taker(next_idx, taker_signal)
        best_ask_next = float(self.best_ask[next_idx])
        best_bid_next = float(self.best_bid[next_idx])
        avg_tracker = float(self.avg_entry_price)
        inv_tracker = float(inv_prev)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, 1, maker_buy, bid)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, -1, maker_sell, ask)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, 1, taker_buy, best_ask_next)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, -1, taker_sell, best_bid_next)
        self.avg_entry_price = avg_tracker if inv_tracker != 0.0 else 0.0
        inv_new = self.inventory
        inv_change = inv_new - inv_prev
        had_fill = maker_buy > 0.0 or maker_sell > 0.0 or taker_buy > 0.0 or taker_sell > 0.0
        if had_fill:
            self.time_since_last_fill = 0.0
        else:
            dt_ms = 1
            if self.decision_ts is not None:
                dt_ms = int(self.decision_ts[next_idx]) - int(self.decision_ts[self.idx])
            dt_ms = max(1, dt_ms)
            self.time_since_last_fill += float(dt_ms) / float(RAW_SNAPSHOT_EXPECTED_STEP_MS)

        mid_next = self._mid_price(next_idx)
        maker_rebate_notional = maker_buy * bid + maker_sell * ask
        rebate = maker_rebate_notional * self.maker_rebate_bps * 1e-4
        taker_notional = taker_buy * best_ask_next + taker_sell * best_bid_next
        taker_fee = taker_notional * self.taker_fee_bps * 1e-4
        self.cash += rebate - taker_fee

        maker_buy_notional = maker_buy * bid
        maker_sell_notional = maker_sell * ask
        taker_buy_notional = taker_buy * best_ask_next
        taker_sell_notional = taker_sell * best_bid_next
        buy_notional_total = maker_buy_notional + taker_buy_notional
        sell_notional_total = maker_sell_notional + taker_sell_notional
        net_fill_notional = buy_notional_total - sell_notional_total
        gross_fill_notional = buy_notional_total + sell_notional_total
        maker_buy_markout = (mid_next - bid) * maker_buy if maker_buy > 0.0 else 0.0
        maker_sell_markout = (ask - mid_next) * maker_sell if maker_sell > 0.0 else 0.0

        if maker_buy > 0.0:
            self.last_maker_buy_notional = maker_buy_notional
        if maker_sell > 0.0:
            self.last_maker_sell_notional = maker_sell_notional
        if taker_buy > 0.0:
            self.last_taker_buy_notional = taker_buy_notional
        if taker_sell > 0.0:
            self.last_taker_sell_notional = taker_sell_notional
        # Channel-specific last_* tracks the last non-zero event for that channel,
        # while net/gross track the last step with any fill.
        if had_fill:
            self.last_net_fill_notional = net_fill_notional
            self.last_gross_fill_notional = gross_fill_notional
        self.ema_net_fill_notional = self._ema_update(self.ema_net_fill_notional, net_fill_notional)
        self.ema_gross_fill_notional = self._ema_update(self.ema_gross_fill_notional, gross_fill_notional)
        # EMA of conditional markout given maker fill.
        # Interpretable as adverse-selection quality (not maker-fill activity intensity).
        if maker_buy > 0.0:
            self.ema_maker_buy_markout = self._ema_update(self.ema_maker_buy_markout, maker_buy_markout)
        if maker_sell > 0.0:
            self.ema_maker_sell_markout = self._ema_update(self.ema_maker_sell_markout, maker_sell_markout)
        equity = self.cash + self.inventory * mid_next
        delta_equity = equity - self.prev_equity
        penalty = self._compute_penalty(mid_next)
        # Quadratic regularizer uses quote notional inventory with a dead-zone.
        inv_notional = abs(inv_new * mid_next)
        excess_notional = max(0.0, inv_notional - self.inv_soft_notional)
        inv_penalty = (
            self.lambda_inv * (excess_notional / self.inv_soft_notional) ** 2
            if self.inv_soft_notional > 0.0
            else 0.0
        )
        inventory_penalty_total = self._combine_inventory_penalties(penalty, inv_penalty)
        turnover_notional = maker_rebate_notional + taker_notional
        turnover_penalty = self.lambda_turn * turnover_notional
        reward = delta_equity - inventory_penalty_total - turnover_penalty

        self.prev_equity = equity
        self.total_reward += reward
        self.idx = next_idx
        done = self.idx >= self.n - 1
        next_obs = self._build_observation(self.idx)
        post_hard_cap_qty = self._inventory_cap_qty(mid_next)
        post_buy_room_qty = self._remaining_inventory_room(1, mid_next)
        post_sell_room_qty = self._remaining_inventory_room(-1, mid_next)
        info = {
            "reward": float(reward),
            "total_reward": float(self.total_reward),
            "cash": float(self.cash),
            "inventory": float(self.inventory),
            "inventory_notional": float(inv_notional),
            "equity": float(equity),
            "delta_equity": float(delta_equity),
            "rebate": float(rebate),
            "taker_fee": float(taker_fee),
            "penalty": float(penalty),
            "inv_penalty": float(inv_penalty),
            "inventory_excess_notional": float(excess_notional),
            "inventory_penalty_total": float(inventory_penalty_total),
            "turnover_penalty": float(turnover_penalty),
            "mid": float(mid_next),
            "hard_max_inventory_notional": float(self.hard_max_inventory_notional),
            "pre_hard_cap_qty": float(pre_hard_cap_qty),
            "pre_buy_room_qty": float(pre_buy_room_qty),
            "pre_sell_room_qty": float(pre_sell_room_qty),
            "post_hard_cap_qty": float(post_hard_cap_qty),
            "post_buy_room_qty": float(post_buy_room_qty),
            "post_sell_room_qty": float(post_sell_room_qty),
            "bid": float(bid),
            "ask": float(ask),
            "bid_delta_bps": float(bid_delta_bps),
            "ask_delta_bps": float(ask_delta_bps),
            "inv_change": float(inv_change),
            "maker_buy": float(maker_buy),
            "maker_sell": float(maker_sell),
            "taker_buy": float(taker_buy),
            "taker_sell": float(taker_sell),
            "maker_buy_notional": float(maker_buy_notional),
            "maker_sell_notional": float(maker_sell_notional),
            "taker_buy_notional": float(taker_buy_notional),
            "taker_sell_notional": float(taker_sell_notional),
            "net_fill_notional": float(net_fill_notional),
            "gross_fill_notional": float(gross_fill_notional),
            "ema_net_fill_notional": float(self.ema_net_fill_notional),
            "ema_gross_fill_notional": float(self.ema_gross_fill_notional),
            "avg_entry_price": float(self.avg_entry_price),
            "unrealized_pnl_notional": float(self.inventory * (mid_next - self.avg_entry_price) if self.inventory != 0.0 else 0.0),
            "maker_buy_markout": float(maker_buy_markout),
            "maker_sell_markout": float(maker_sell_markout),
            "ema_maker_buy_markout": float(self.ema_maker_buy_markout),
            "ema_maker_sell_markout": float(self.ema_maker_sell_markout),
            "maker_buy_clipped": float(self.last_maker_buy_clipped),
            "maker_sell_clipped": float(self.last_maker_sell_clipped),
            "taker_buy_clipped": float(self.last_taker_buy_clipped),
            "taker_sell_clipped": float(self.last_taker_sell_clipped),
        }
        return next_obs, float(reward), done, info


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (128, 128), action_dim: int = 3):
        super().__init__()
        # MarketPolicyNet wraps its MLP under .net for compatibility with checkpoints.
        self.net = MLP(input_dim, hidden_dims, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        policy_hidden: Iterable[int] = (128, 128),
        value_hidden: Iterable[int] = (128, 128),
        action_dim: int = 3,
        init_log_std: float = -3.0,
    ):
        super().__init__()
        self.policy_net = MarketPolicyNet(input_dim, hidden_dims=policy_hidden, action_dim=action_dim)
        self.value_net = MLP(input_dim, value_hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)
        log_std = torch.clamp(self.log_std, min=-6.0, max=2.0)
        return mean, log_std, value


def _find_final_policy_linear_layer(model: MarketPolicyValueNet) -> nn.Linear:
    final_policy_linear: Optional[nn.Linear] = None
    for module in model.policy_net.net.net:
        if isinstance(module, nn.Linear):
            final_policy_linear = module
    if final_policy_linear is None:
        raise RuntimeError("Could not find final policy linear layer")
    return final_policy_linear


def _init_zero_residual_policy(model: MarketPolicyValueNet, init_log_std: float) -> None:
    final_policy_linear = _find_final_policy_linear_layer(model)
    with torch.no_grad():
        final_policy_linear.weight.zero_()
        if final_policy_linear.bias is not None:
            final_policy_linear.bias.zero_()
        model.log_std.fill_(init_log_std)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 32768
    entropy_coef: float = 0.01
    action_mag_coef: float = 0.0
    action_mag_power: float = 2.0
    value_coef: float = 0.5
    policy_hidden: Tuple[int, ...] = (128, 128)
    value_hidden: Tuple[int, ...] = (128, 128)
    val_every: int = 1
    max_drawdown_guard: Optional[float] = None
    rollout_horizon: int = 32768
    rollouts_per_epoch: int = 16
    randomize_rollout_start: bool = True
    zero_residual_init: bool = True
    init_log_std: float = -3.0


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminals: torch.Tensor,
    gamma: float,
    lam: float,
):
    # `terminals` only marks true environment terminations. Rollout truncation
    # (e.g., horizon cutoff) still bootstraps from `next_values`.
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - terminals[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_market_rollout(
    env: MarketMakingEnv,
    model: MarketPolicyValueNet,
    device: str,
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
    horizon: int = 32768,
    rollouts_per_epoch: int = 16,
    randomize_start: bool = True,
    rollout_storage: str = "gpu",
    pin_memory: bool = True,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    # Canonical PPO action space is the bounded env-facing action tensor.
    # We sample in latent Gaussian space, squash with tanh, then affinely map
    # into env units so rollout actions/log-probs match executed actions.
    t0 = time.perf_counter()

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if rollouts_per_epoch <= 0:
        raise ValueError(f"rollouts_per_epoch must be positive, got {rollouts_per_epoch}")

    max_steps = horizon * rollouts_per_epoch
    storage = rollout_storage.strip().lower()
    if storage not in {"gpu", "cpu"}:
        raise ValueError(f"rollout_storage must be one of ['gpu', 'cpu'], got {rollout_storage}")

    target_device = torch.device(device)
    use_gpu_storage = storage == "gpu"
    if use_gpu_storage and target_device.type != "cuda":
        use_gpu_storage = False
        storage = "cpu"

    use_pinned = bool(pin_memory and storage == "cpu" and target_device.type == "cuda")
    storage_device = target_device if use_gpu_storage else torch.device("cpu")

    obs_buf: Optional[torch.Tensor] = None
    next_obs_buf: Optional[torch.Tensor] = None
    actions_buf: Optional[torch.Tensor] = None
    alloc_kwargs = {"device": storage_device}
    if use_pinned:
        alloc_kwargs["pin_memory"] = True
    logp_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    values_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    rewards_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    terminated_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    truncated_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    dones_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    cursor = 0

    action_dim = int(model.log_std.shape[0])
    action_low, action_high = _ppo_action_bounds(
        env,
        action_dim,
        target_device,
        delta_scale,
        taker_scale,
    )

    max_start = max(0, env.n - 2)
    for _ in range(rollouts_per_epoch):
        start_idx = int(np.random.randint(0, max_start + 1)) if randomize_start else 0
        obs = env.reset(start_idx=start_idx)
        done = False
        steps = 0
        while not done and steps < horizon:
            obs_cpu = torch.from_numpy(obs).float()
            if obs_buf is None:
                obs_dim = int(obs_cpu.shape[0])
                obs_buf = torch.empty((max_steps, obs_dim), dtype=torch.float32, **alloc_kwargs)
                next_obs_buf = torch.empty((max_steps, obs_dim), dtype=torch.float32, **alloc_kwargs)
                actions_buf = torch.empty((max_steps, action_dim), dtype=torch.float32, **alloc_kwargs)
            obs_t = obs_cpu.to(target_device, non_blocking=non_blocking)
            with torch.no_grad():
                mean, log_std, value = model(obs_t.unsqueeze(0))
                action_env, logp_env, _latent_action = _sample_bounded_ppo_action(
                    mean,
                    log_std,
                    action_low,
                    action_high,
                )

            env_action = _market_env_action_tuple(action_env.squeeze(0).detach().cpu().numpy())
            next_obs, reward, env_done, _info = env.step(env_action)
            steps += 1
            terminated = bool(env_done)
            # Truncation means the rollout horizon ended; it is not a true
            # environment terminal state and should continue to bootstrap.
            truncated = (not terminated) and (steps >= horizon)
            done = terminated or truncated

            idx = cursor
            cursor += 1
            require(
                obs_buf is not None and next_obs_buf is not None and actions_buf is not None,
                "rollout buffers not initialized",
            )
            next_obs_cpu = torch.from_numpy(next_obs).float()
            if use_gpu_storage:
                obs_buf[idx].copy_(obs_t)
                next_obs_buf[idx].copy_(next_obs_cpu.to(target_device, non_blocking=non_blocking))
                actions_buf[idx].copy_(action_env.squeeze(0).detach())
                logp_buf[idx] = logp_env.squeeze(0).detach()
                values_buf[idx] = value.squeeze(0).detach()
            else:
                obs_buf[idx].copy_(obs_cpu)
                next_obs_buf[idx].copy_(next_obs_cpu)
                actions_buf[idx].copy_(action_env.squeeze(0).detach().cpu())
                logp_buf[idx] = logp_env.squeeze(0).detach().cpu()
                values_buf[idx] = value.squeeze(0).detach().cpu()
            rewards_buf[idx] = float(reward)
            terminated_buf[idx] = float(terminated)
            truncated_buf[idx] = float(truncated)
            dones_buf[idx] = float(done)
            obs = next_obs

    if obs_buf is None or next_obs_buf is None or actions_buf is None:
        raise RuntimeError("No rollout transitions collected.")

    next_values_buf = torch.zeros((cursor,), dtype=torch.float32, **alloc_kwargs)
    bootstrap_mask = terminated_buf[:cursor] == 0.0
    bootstrap_batches = 0
    if torch.any(bootstrap_mask):
        boot_indices = torch.nonzero(bootstrap_mask, as_tuple=False).squeeze(-1)
        infer_bs = 4096
        with torch.no_grad():
            for start in range(0, int(boot_indices.shape[0]), infer_bs):
                bootstrap_batches += 1
                idx = boot_indices[start:start + infer_bs]
                batch_next_obs = next_obs_buf[idx].to(target_device, non_blocking=non_blocking)
                _next_mean, _next_log_std, next_value = model(batch_next_obs)
                if use_gpu_storage:
                    next_values_buf[idx] = next_value.detach()
                else:
                    next_values_buf[idx] = next_value.detach().cpu()

    _timing_log(
        f"rollout storage={storage} steps={cursor} bootstrap_batches={bootstrap_batches} secs={time.perf_counter() - t0:.4f}"
    )

    return {
        "obs": obs_buf[:cursor],
        "actions": actions_buf[:cursor],
        "logp": logp_buf[:cursor],
        "values": values_buf[:cursor],
        "next_values": next_values_buf,
        "rewards": rewards_buf[:cursor],
        "terminated": terminated_buf[:cursor],
        "truncated": truncated_buf[:cursor],
        "dones": dones_buf[:cursor],
    }


def ppo_update_market(
    model: MarketPolicyValueNet,
    optimizer: optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: PPOConfig,
    device: str,
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
    non_blocking: bool = True,
    env: Optional[MarketMakingEnv] = None,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    # PPO loss is recomputed from the same bounded env-facing action
    # parameterization used during rollout collection.
    obs = rollout["obs"]
    actions = rollout["actions"]
    old_logp = rollout["logp"].detach()
    values = rollout["values"].detach()
    next_values = rollout["next_values"].detach()
    rewards = rollout["rewards"]
    terminals = rollout["terminated"]

    advantages, returns = compute_gae(rewards, values, next_values, terminals, config.gamma, config.gae_lambda)
    action_dim = int(actions.shape[-1])

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    target_device = torch.device(device)
    if obs.device.type != target_device.type:
        same_device = False
    elif target_device.type != "cuda":
        same_device = True
    else:
        # torch.device("cuda") leaves index unspecified; treat any CUDA tensor on
        # the current target GPU as already on-device in this fast path.
        same_device = (target_device.index is None) or (obs.device.index == target_device.index)
    action_low, action_high = _ppo_action_bounds(
        env,
        action_dim,
        target_device,
        delta_scale,
        taker_scale,
    )
    indices = torch.arange(n, device=obs.device)
    action_mag_loss_total = 0.0
    action_mag_loss_batches = 0
    for _ in range(config.update_epochs):
        perm = indices[torch.randperm(n, device=obs.device)]
        for start in range(0, n, config.batch_size):
            mb_idx = perm[start:start + config.batch_size]
            if same_device:
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
            else:
                mb_idx_cpu = mb_idx.cpu() if mb_idx.device.type != "cpu" else mb_idx
                mb_obs = obs[mb_idx_cpu].to(target_device, non_blocking=non_blocking)
                mb_actions = actions[mb_idx_cpu].to(target_device, non_blocking=non_blocking)
                mb_old_logp = old_logp[mb_idx_cpu].to(target_device, non_blocking=non_blocking)
                mb_advantages = advantages[mb_idx_cpu].to(target_device, non_blocking=non_blocking)
                mb_returns = returns[mb_idx_cpu].to(target_device, non_blocking=non_blocking)

            mean, log_std, value = model(mb_obs)
            action_mag_loss = _bounded_mean_action_penalty(
                mean,
                action_low,
                action_high,
                power=config.action_mag_power,
            )
            latent_actions = _bounded_ppo_latent_action(mb_actions, action_low, action_high)
            logp = _squashed_gaussian_log_prob(
                latent_actions,
                mean,
                log_std,
                action_low,
                action_high,
            )
            ratio = torch.exp(logp - mb_old_logp)
            clip_adv = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * mb_advantages
            policy_loss = -(torch.min(ratio * mb_advantages, clip_adv)).mean()
            value_loss = nn.functional.mse_loss(value, mb_returns)
            entropy_loss = _diag_gaussian_entropy(log_std).mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_loss
                + config.action_mag_coef * action_mag_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            action_mag_loss_total += action_mag_loss.detach().item()
            action_mag_loss_batches += 1

    storage = "gpu" if obs.device.type == target_device.type else "cpu"
    _timing_log(
        f"ppo_update storage={storage} on_device={same_device} manual_gaussian=true secs={time.perf_counter() - t0:.4f}"
    )
    avg_action_mag_loss = (
        action_mag_loss_total / action_mag_loss_batches
        if action_mag_loss_batches > 0
        else 0.0
    )
    return {"action_mag_loss": avg_action_mag_loss}

def _steps_per_year_from_snapshot_ms(step_ms: float) -> float:
    if step_ms <= 0:
        return 0.0
    steps_per_second = 1000.0 / step_ms
    return steps_per_second * 60.0 * 60.0 * 24.0 * 365.0


def compute_sharpe(returns: np.ndarray, steps_per_year: float) -> float:
    """Compute annualized Sharpe for per-step percentage returns."""
    if returns.size == 0 or steps_per_year <= 0:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(mean / std * np.sqrt(steps_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute max drawdown as the largest peak-to-trough equity decline."""
    if equity_curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = np.divide(
        peak - equity_curve,
        peak,
        out=np.zeros_like(equity_curve),
        where=peak != 0,
    )
    return float(drawdown.max(initial=0.0))


def compute_capital_returns(equity_curve: np.ndarray, initial_equity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fixed-capital per-step PnL and returns from an equity curve.

    Returns are normalized by initial equity (with a small floor) rather than
    prior equity, making them robust when running equity gets near zero or
    temporarily negative.
    """
    if equity_curve.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty
    prev = np.concatenate([[initial_equity], equity_curve[:-1]])
    pnl_changes = equity_curve - prev
    capital_base = max(float(initial_equity), 1e-12)
    capital_returns = pnl_changes / capital_base
    return pnl_changes, capital_returns


def aggregate_returns_by_time(ts_ms: np.ndarray, returns: np.ndarray, bucket_ms: int) -> np.ndarray:
    """Aggregate per-step returns into fixed time buckets by summation."""
    if ts_ms.size != returns.size:
        raise ValueError("ts_ms and returns must have equal length")
    if ts_ms.size == 0:
        return np.array([], dtype=np.float32)
    if bucket_ms <= 0:
        raise ValueError("bucket_ms must be positive")

    ts_arr = np.asarray(ts_ms, dtype=np.int64)
    ret_arr = np.asarray(returns, dtype=np.float32)
    bucket_idx = (ts_arr - ts_arr[0]) // int(bucket_ms)
    _, inverse = np.unique(bucket_idx, return_inverse=True)
    agg = np.zeros(int(inverse.max()) + 1, dtype=np.float32)
    np.add.at(agg, inverse, ret_arr)
    return agg


def compute_sortino(returns: np.ndarray, periods_per_year: float) -> float:
    """Compute annualized Sortino ratio from periodic returns."""
    if returns.size < 2 or periods_per_year <= 0:
        return 0.0
    mean = float(np.mean(returns))
    downside = returns[returns < 0]
    if downside.size == 0:
        return 0.0
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else float(np.std(downside, ddof=0))
    if downside_std <= 0:
        return 0.0
    return float(mean / downside_std * np.sqrt(periods_per_year))


def evaluate_market_policy(
    env: MarketMakingEnv,
    policy: MarketPolicyNet,
    device: str = "cuda",
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
) -> Dict[str, Any]:
    def _policy_fn(obs: np.ndarray) -> Tuple[float, float, float]:
        return _policy_action_from_obs_numpy(
            obs,
            policy,
            device,
            delta_scale,
            taker_scale,
            env=env,
        )

    return evaluate_market_making(env, _policy_fn)


def _market_env_action_tuple(action: np.ndarray | torch.Tensor | Sequence[float]) -> Tuple[float, float, float]:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    require(action_arr.shape[0] >= 2, f"Expected at least 2 action dimensions, got shape={action_arr.shape}")
    taker_delta = float(action_arr[2]) if action_arr.shape[0] >= 3 else 0.0
    return float(action_arr[0]), float(action_arr[1]), taker_delta


def evaluate_market_policy_ppo(
    env: MarketMakingEnv,
    model: MarketPolicyValueNet,
    *,
    stochastic: bool,
    device: str = "cuda",
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    def _policy_fn(obs: np.ndarray) -> Tuple[float, float, float]:
        action = _ppo_action_from_obs_numpy(
            model,
            obs,
            stochastic=stochastic,
            generator=generator,
            device=device,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
            env=env,
        )
        return _market_env_action_tuple(action)

    return evaluate_market_making(env, _policy_fn)


def _deterministic_env_action_from_model_output(
    raw_action: torch.Tensor,
    *,
    env: Optional[MarketMakingEnv],
    delta_scale: float,
    taker_scale: float,
) -> torch.Tensor:
    action_low, action_high = _ppo_action_bounds(
        env,
        int(raw_action.shape[-1]),
        raw_action.device,
        delta_scale,
        taker_scale,
    )
    return _postprocess_bounded_env_action(raw_action, action_low, action_high)


def _policy_action_from_obs_numpy(
    obs: np.ndarray,
    policy: torch.nn.Module,
    device: str,
    delta_scale: float,
    taker_scale: float,
    *,
    env: Optional[MarketMakingEnv] = None,
) -> Tuple[float, float, float]:
    obs_t = torch.from_numpy(obs).float().to(device)
    with torch.no_grad():
        raw_action = policy(obs_t.unsqueeze(0)).squeeze(0)
        action_env = _deterministic_env_action_from_model_output(
            raw_action,
            env=env,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
        )
    return _market_env_action_tuple(action_env.cpu().numpy())


def _ppo_action_from_obs_numpy(
    model: MarketPolicyValueNet,
    obs_np: np.ndarray,
    stochastic: bool,
    generator: Optional[torch.Generator] = None,
    *,
    device: str = "cuda",
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
    env: Optional[MarketMakingEnv] = None,
) -> np.ndarray:
    obs_t = torch.from_numpy(obs_np).float().to(device)
    action_low, action_high = _ppo_action_bounds(
        env,
        int(model.log_std.shape[0]),
        obs_t.device,
        delta_scale,
        taker_scale,
    )
    with torch.no_grad():
        mean, log_std, _value = model(obs_t.unsqueeze(0))
        if not stochastic:
            action_env = _bounded_ppo_mean_action(mean, action_low, action_high)
        else:
            action_env, _logp, _latent_action = _sample_bounded_ppo_action(
                mean,
                log_std,
                action_low,
                action_high,
                generator=generator,
            )
    return action_env.squeeze(0).cpu().numpy()


def _safe_metric_for_tiebreak(x: Any) -> float:
    x_float = float(x)
    return x_float if np.isfinite(x_float) else float("-inf")


def _checkpoint_selection_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    initial_equity = float(report.get("initial_equity", 0.0))
    final_equity = float(report.get("final_equity", initial_equity))
    net_fee_cost = float(report.get("net_fee_cost", 0.0))

    denom = max(initial_equity, 1e-12)
    net_pnl = final_equity - initial_equity
    net_pnl_pct = net_pnl / denom
    gross_pnl = net_pnl + net_fee_cost
    gross_pnl_pct = gross_pnl / denom

    return {
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "gross_pnl": gross_pnl,
        "gross_pnl_pct": gross_pnl_pct,
        "max_drawdown": float(report.get("max_drawdown", float("inf"))),
        "sharpe_1h": _safe_metric_for_tiebreak(report.get("sharpe_1h", float("-inf"))),
        "sortino_1h": _safe_metric_for_tiebreak(report.get("sortino_1h", float("-inf"))),
    }


def _checkpoint_survives_filters(sel: Dict[str, float], dd_cap: Optional[float]) -> Tuple[bool, str]:
    if not np.isfinite(sel["net_pnl_pct"]):
        return False, "non_finite_net_pnl_pct"
    if sel["net_pnl_pct"] <= 0.0:
        return False, "non_positive_net_pnl_pct"
    if not np.isfinite(sel["max_drawdown"]):
        return False, "non_finite_max_drawdown"
    if dd_cap is not None and sel["max_drawdown"] > dd_cap:
        return False, "drawdown_cap_exceeded"
    return True, "ok"


def _checkpoint_key(sel: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(sel["net_pnl_pct"]),
        float(sel["sharpe_1h"]),
        float(sel["sortino_1h"]),
    )


def _strip_large_report_fields(report: Dict[str, Any]) -> Dict[str, Any]:
    drop_keys = {
        "equity_curve",
        "pnl_curve",
    }
    return {k: v for k, v in report.items() if k not in drop_keys}


def _build_best_validation_summary(
    best_report: Dict[str, Any],
    *,
    best_report_mode: str,
    checkpoint_metric_mode: str,
    selection_metrics: Dict[str, float],
    selection_epoch: int,
) -> Dict[str, Any]:
    summary = _strip_large_report_fields(best_report)
    summary.update({
        "best_report_mode": str(best_report_mode),
        "checkpoint_metric_mode": str(checkpoint_metric_mode),
        "selection_metrics": dict(selection_metrics),
        "selection_epoch": int(selection_epoch),
    })
    return summary


def _resolve_checkpoint_metric_mode() -> str:
    mode = os.environ.get("BYBIT_MM_PPO_CHECKPOINT_METRIC_MODE", "deterministic").strip().lower()
    allowed = {"deterministic", "stochastic"}
    if mode not in allowed:
        accepted = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid BYBIT_MM_PPO_CHECKPOINT_METRIC_MODE='{mode}'. Accepted values: {accepted}"
        )
    return mode


def prefit_market_obs_norm(train_env: MarketMakingEnv) -> Dict[str, Any]:
    """Fit observation normalization on the train split before PPO consumes observations."""
    train_env.set_obs_norm_state(_empty_obs_norm_state(), freeze=False)
    _ = train_env.reset(start_idx=0)
    done = False
    while not done:
        _, _, done, _ = train_env.step((0.0, 0.0, 0.0))
    state = train_env.get_obs_norm_state()
    if not _obs_norm_state_is_ready(state):
        raise RuntimeError(
            "Train-only observation-normalization prefit did not produce a ready state; "
            "ensure the train env has at least two observations."
        )
    mean = np.asarray(state["mean"], dtype=np.float64)
    m2 = np.asarray(state["m2"], dtype=np.float64)
    mask_raw = state.get("continuous_mask")
    mask = None if mask_raw is None else np.asarray(mask_raw, dtype=bool)
    if mean.shape != m2.shape:
        raise RuntimeError(
            f"Prefitted obs normalization mean/m2 shape mismatch: mean={mean.shape} m2={m2.shape}"
        )
    if mask is not None and mask.shape != mean.shape:
        raise RuntimeError(
            "Prefitted obs normalization continuous_mask shape mismatch: "
            f"mask={mask.shape} mean={mean.shape}"
        )
    return {
        "count": int(state["count"]),
        "mean": mean.tolist(),
        "m2": m2.tolist(),
        "continuous_mask": None if mask is None else mask.tolist(),
    }


def _build_market_probe_obs_batch(
    env: MarketMakingEnv,
    *,
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    if env.n <= 0:
        raise ValueError("Cannot build PPO probe batch from an empty market-making env")

    max_start = max(0, env.n - 2)
    probe_count = max(1, min(int(batch_size), max_start + 1))
    probe_indices = np.linspace(0, max_start, num=probe_count, dtype=int)
    probe_obs = [
        env.reset(start_idx=int(idx)).astype(np.float32, copy=True)
        for idx in probe_indices
    ]
    return torch.as_tensor(np.stack(probe_obs, axis=0), device=device)


def save_market_ppo_checkpoint(
    ckpt_path: Path,
    model: MarketPolicyValueNet,
    *,
    policy_hidden_dims: Iterable[int],
    value_hidden_dims: Iterable[int],
    obs_dim: int,
    action_dim: int,
    val_report: Dict[str, Any],
    val_report_mode: str,
    obs_norm_state: Dict[str, Any],
    selection_epoch: int,
    selection_metrics: Dict[str, float],
    selection_key: Tuple[float, float, float],
    checkpoint_metric_mode: str,
    selection_policy: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "policy_hidden_dims": tuple(int(x) for x in policy_hidden_dims),
        "value_hidden_dims": tuple(int(x) for x in value_hidden_dims),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "val_report": _strip_large_report_fields(val_report),
        "val_report_mode": str(val_report_mode),
        "best_report_mode": str(val_report_mode),
        "obs_norm_state": obs_norm_state,
        "selection_metrics": dict(selection_metrics),
        "selection_key": list(selection_key),
        "selection_epoch": int(selection_epoch),
        "checkpoint_metric_mode": str(checkpoint_metric_mode),
        "selection_policy": selection_policy or {},
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save(payload, ckpt_path)


def _canonical_market_ppo_arch_field(ckpt: Dict[str, Any], field_name: str) -> Tuple[int, ...]:
    value = ckpt.get(field_name)
    canonical_error = (
        "Only canonical PPO checkpoints are supported; checkpoint must include "
        f"'{field_name}' and be re-exported or retrained if missing/malformed."
    )
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(canonical_error)
    try:
        dims = tuple(int(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(canonical_error) from exc
    if any(dim <= 0 for dim in dims):
        raise ValueError(canonical_error)
    return dims


def _canonical_market_ppo_action_dim(ckpt: Dict[str, Any]) -> int:
    canonical_error = (
        "Only canonical PPO checkpoints are supported; checkpoint must include "
        "'action_dim' and be re-exported or retrained if missing/malformed."
    )
    try:
        action_dim = int(ckpt["action_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(canonical_error) from exc
    if action_dim <= 0:
        raise ValueError(canonical_error)
    return action_dim


def load_market_ppo_model(
    input_dim: int,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
    require_checkpoint: bool = False,
    checkpoint_data: Optional[Any] = None,
) -> Optional[MarketPolicyValueNet]:
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    if not path.exists():
        if require_checkpoint:
            raise FileNotFoundError(f"Market PPO checkpoint not found: {ckpt_path}")
        warnings.warn(
            f"Market PPO checkpoint not found: {ckpt_path}. Falling back to baseline policy.",
            RuntimeWarning,
        )
        return None
    ckpt = (
        checkpoint_data
        if checkpoint_data is not None
        else _torch_load_trusted_checkpoint(path, map_location=device)
    )
    if not isinstance(ckpt, dict):
        raise ValueError(
            "Unsupported PPO checkpoint payload type; expected a mapping for market PPO loading."
        )

    state = ckpt.get("model_state_dict")
    canonical_metadata_fields = ("policy_hidden_dims", "value_hidden_dims", "action_dim")
    has_any_canonical_metadata = any(field in ckpt for field in canonical_metadata_fields)
    if not isinstance(state, dict):
        if not has_any_canonical_metadata:
            raise ValueError(
                "Unsupported RL checkpoint format. Only canonical full PPO checkpoints are supported. "
                "Re-export or retrain under the PPO checkpoint format with model_state_dict, "
                "policy_hidden_dims, value_hidden_dims, and action_dim."
            )
        raise ValueError(
            "Malformed canonical market PPO checkpoint: model_state_dict is missing or not a mapping."
        )

    policy_hidden_dims = _canonical_market_ppo_arch_field(ckpt, "policy_hidden_dims")
    value_hidden_dims = _canonical_market_ppo_arch_field(ckpt, "value_hidden_dims")
    action_dim = _canonical_market_ppo_action_dim(ckpt)

    model = MarketPolicyValueNet(
        input_dim,
        policy_hidden=policy_hidden_dims,
        value_hidden=value_hidden_dims,
        action_dim=action_dim,
    ).to(device)

    model.load_state_dict(state, strict=True)

    model.eval()
    model = _maybe_compile_module(
        model,
        enabled=_env_bool("BYBIT_MM_COMPILE_PPO", False),
        label="ppo_eval_model",
    )
    return model


def train_market_ppo(
    train_env: MarketMakingEnv,
    val_env: MarketMakingEnv,
    input_dim: int,
    device: str = "cuda",
    epochs: int = 10,
    config: Optional[PPOConfig] = None,
    ckpt_path: Optional[Path] = None,
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
    rollout_storage: str = "gpu",
    pin_rollout_memory: bool = True,
    non_blocking_transfers: bool = True,
) -> Tuple[MarketPolicyValueNet, Dict[str, Any], bool]:
    config = config or PPOConfig()
    checkpoint_metric_mode = _resolve_checkpoint_metric_mode()
    validation_modes = ("deterministic", "stochastic")
    stochastic_val_seed = _env_int("BYBIT_MM_PPO_VAL_SEED", 0)
    compile_enabled = _env_bool("BYBIT_MM_COMPILE_PPO", False)
    compile_mode = os.environ.get("BYBIT_MM_COMPILE_MODE", "reduce-overhead")
    action_dim = _resolve_market_action_dim(train_env.allow_taker)
    model = MarketPolicyValueNet(
        input_dim,
        policy_hidden=config.policy_hidden,
        value_hidden=config.value_hidden,
        action_dim=action_dim,
        init_log_std=config.init_log_std,
    ).to(device)
    if config.zero_residual_init:
        _init_zero_residual_policy(model, config.init_log_std)
    print(
        "[mm ppo compile] "
        f"enabled={compile_enabled} "
        f"mode={compile_mode}"
    )
    model = _maybe_compile_module(
        model,
        enabled=compile_enabled,
        label="ppo_train",
    )
    print(
        "[mm ppo init] "
        f"rollout_horizon={config.rollout_horizon} "
        f"rollouts_per_epoch={config.rollouts_per_epoch} "
        f"steps_per_epoch={config.rollout_horizon * config.rollouts_per_epoch} "
        f"zero_residual_init={config.zero_residual_init} "
        f"init_log_std={config.init_log_std:.4f} "
        f"action_mag_coef={config.action_mag_coef:.6f} "
        f"action_mag_power={config.action_mag_power:.2f}"
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    train_obs_norm_state = train_env.get_obs_norm_state()
    if not _obs_norm_state_is_ready(train_obs_norm_state) or not train_env.freeze_obs_norm:
        raise RuntimeError(
            "PPO requires prefitted frozen observation normalization; call "
            "prefit_market_obs_norm() before train_market_ppo()."
        )
    val_obs_norm_state = val_env.get_obs_norm_state()
    if not _obs_norm_state_is_ready(val_obs_norm_state) or not val_env.freeze_obs_norm:
        raise RuntimeError(
            "Validation env must share the prefitted frozen observation normalization before "
            "train_market_ppo() probe construction."
        )
    if train_obs_norm_state != val_obs_norm_state:
        raise RuntimeError(
            "Train/validation observation normalization states differ; install the same prefitted "
            "frozen state on both envs before train_market_ppo()."
        )
    prefitted_obs_count = int(train_obs_norm_state["count"])
    print(
        "[mm ppo obs norm] "
        f"count={prefitted_obs_count} "
        f"train_frozen={train_env.freeze_obs_norm} "
        f"val_frozen={val_env.freeze_obs_norm}"
    )
    assert prefitted_obs_count >= 2, "prefitted observation normalization must have count >= 2"
    probe_obs = _build_market_probe_obs_batch(val_env, batch_size=8, device=device)
    action_low, action_high = _ppo_action_bounds(
        train_env,
        int(model.log_std.shape[0]),
        device,
        delta_scale,
        taker_scale,
    )
    with torch.no_grad():
        bounded_probe_action = _bounded_ppo_mean_action(
            model.policy_net(probe_obs[:1]),
            action_low,
            action_high,
        ).squeeze(0).detach().cpu().numpy()
    bounds_low_np = action_low.detach().cpu().numpy()
    bounds_high_np = action_high.detach().cpu().numpy()
    within_bounds = bool(
        np.all(bounded_probe_action >= bounds_low_np - 1e-6)
        and np.all(bounded_probe_action <= bounds_high_np + 1e-6)
    )
    print(
        "[mm ppo bounds] "
        f"action_dim={action_dim} "
        f"low={np.array2string(bounds_low_np, precision=4, floatmode='fixed')} "
        f"high={np.array2string(bounds_high_np, precision=4, floatmode='fixed')} "
        f"env_delta_bps_limit={train_env.delta_bps_limit:.4f} "
        f"allow_taker={train_env.allow_taker} "
        f"mean_probe_action={np.array2string(bounded_probe_action, precision=4, floatmode='fixed')} "
        f"env_action={_market_env_action_tuple(bounded_probe_action)} "
        f"bounded_before_step={within_bounds}"
    )
    print(
        "[mm ppo probe] "
        f"batch={int(probe_obs.shape[0])} "
        f"source=val_env "
        f"checkpoint_metric_mode={checkpoint_metric_mode} "
        f"validation_modes={list(validation_modes)} "
        f"stochastic_val_seed={stochastic_val_seed}"
    )
    best_report: Optional[Dict[str, Any]] = None
    best_report_mode: Optional[str] = None
    best_selection_metrics: Optional[Dict[str, float]] = None
    best_selection_epoch: Optional[int] = None
    best_selection_key: Optional[Tuple[float, float, float]] = None
    saved_new_ckpt_this_run = False

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        obs_count_before_rollout = int(train_env.get_obs_norm_state()["count"])
        rollout = collect_market_rollout(
            train_env,
            model,
            device,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
            horizon=config.rollout_horizon,
            rollouts_per_epoch=config.rollouts_per_epoch,
            randomize_start=config.randomize_rollout_start,
            rollout_storage=rollout_storage,
            pin_memory=pin_rollout_memory,
            non_blocking=non_blocking_transfers,
        )
        ppo_update_summary = ppo_update_market(
            model,
            optimizer,
            rollout,
            config,
            device,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
            non_blocking=non_blocking_transfers,
            env=train_env,
        )
        obs_count_after_rollout = int(train_env.get_obs_norm_state()["count"])
        assert train_env.freeze_obs_norm is True, "train env obs normalization must stay frozen during PPO"
        assert obs_count_after_rollout == obs_count_before_rollout, (
            "train env obs normalization count drifted during PPO despite frozen normalization"
        )
        final_policy_linear = _find_final_policy_linear_layer(model)
        with torch.no_grad():
            probe_mean = model.policy_net(probe_obs)
            bounded_probe_actions = _bounded_ppo_mean_action(probe_mean, action_low, action_high)
            probe_mean_abs = probe_mean.abs().mean(dim=0).detach().cpu().numpy()
            bounded_probe_mean_action_mag = bounded_probe_actions.abs().mean(dim=0).detach().cpu().numpy()
            action_bound_magnitude = torch.maximum(action_low.abs(), action_high.abs())
            saturation_fraction = (
                bounded_probe_actions.abs() >= (0.95 * action_bound_magnitude)
            ).float().mean(dim=0).detach().cpu().numpy()
            log_std_values = model.log_std.detach().cpu().numpy()
            policy_weight_l2 = float(final_policy_linear.weight.detach().norm(2).item())
            policy_bias_l2 = (
                float(final_policy_linear.bias.detach().norm(2).item())
                if final_policy_linear.bias is not None
                else 0.0
            )
        print(
            "[mm ppo stats] "
            f"epoch={epoch + 1} "
            f"action_mag_loss={ppo_update_summary['action_mag_loss']:.6f} "
            f"log_std={np.array2string(log_std_values, precision=4, floatmode='fixed')} "
            f"policy_head_weight_l2={policy_weight_l2:.6f} "
            f"policy_head_bias_l2={policy_bias_l2:.6f} "
            f"probe_mean_abs={np.array2string(probe_mean_abs, precision=6, floatmode='fixed')} "
            f"bounded_probe_mean_action_mag={np.array2string(bounded_probe_mean_action_mag, precision=6, floatmode='fixed')} "
            f"saturation_fraction={np.array2string(saturation_fraction, precision=6, floatmode='fixed')}"
        )
        if (epoch + 1) % config.val_every == 0:
            assert train_env.freeze_obs_norm is True, "train env obs normalization must stay frozen during PPO"
            assert val_env.freeze_obs_norm is True, "val env obs normalization must stay frozen during PPO"
            deterministic_report = evaluate_market_policy_ppo(
                val_env,
                model,
                stochastic=False,
                device=device,
                delta_scale=delta_scale,
                taker_scale=taker_scale,
            )
            stochastic_generator = torch.Generator(device=torch.device(device).type)
            stochastic_generator.manual_seed(stochastic_val_seed)
            stochastic_report = evaluate_market_policy_ppo(
                val_env,
                model,
                stochastic=True,
                device=device,
                delta_scale=delta_scale,
                taker_scale=taker_scale,
                generator=stochastic_generator,
            )
            deterministic_sel = _checkpoint_selection_metrics(deterministic_report)
            stochastic_sel = _checkpoint_selection_metrics(stochastic_report)
            guard = config.max_drawdown_guard
            det_candidate_ok, det_candidate_reason = _checkpoint_survives_filters(
                deterministic_sel, guard
            )
            stoch_candidate_ok, stoch_candidate_reason = _checkpoint_survives_filters(
                stochastic_sel, guard
            )
            selected_mode = checkpoint_metric_mode
            print(
                "[mm val deterministic] "
                f"epoch={epoch + 1} "
                f"net_pnl_pct={deterministic_sel['net_pnl_pct']:.6f} "
                f"sharpe_1h={deterministic_sel['sharpe_1h']:.4f} "
                f"sortino_1h={deterministic_sel['sortino_1h']:.4f} "
                f"max_dd={deterministic_sel['max_drawdown']:.4f} "
                "policy=mean "
                f"candidate_mode={selected_mode} "
                f"candidate={det_candidate_ok} "
                f"reason={det_candidate_reason}"
            )
            print(
                "[mm val stochastic] "
                f"epoch={epoch + 1} "
                f"net_pnl_pct={stochastic_sel['net_pnl_pct']:.6f} "
                f"sharpe_1h={stochastic_sel['sharpe_1h']:.4f} "
                f"sortino_1h={stochastic_sel['sortino_1h']:.4f} "
                f"max_dd={stochastic_sel['max_drawdown']:.4f} "
                f"seed={stochastic_val_seed} "
                f"candidate_mode={selected_mode} "
                f"candidate={stoch_candidate_ok} "
                f"reason={stoch_candidate_reason}"
            )
            selected_report = (
                deterministic_report if selected_mode == "deterministic" else stochastic_report
            )
            selected_sel = (
                deterministic_sel if selected_mode == "deterministic" else stochastic_sel
            )
            candidate_ok, candidate_reason = _checkpoint_survives_filters(selected_sel, guard)
            if candidate_ok:
                candidate_key = _checkpoint_key(selected_sel)
                if best_selection_key is None or candidate_key > best_selection_key:
                    best_selection_key = candidate_key
                    best_report = selected_report
                    best_report_mode = selected_mode
                    best_selection_metrics = dict(selected_sel)
                    best_selection_epoch = epoch + 1
                    print(
                        "[mm ckpt] "
                        f"epoch={epoch + 1} "
                        "selected=true "
                        f"metric_mode={selected_mode} "
                        f"net_pnl_pct={selected_sel['net_pnl_pct']:.6f} "
                        f"sharpe_1h={selected_sel['sharpe_1h']:.4f} "
                        f"sortino_1h={selected_sel['sortino_1h']:.4f} "
                        f"max_dd={selected_sel['max_drawdown']:.4f}"
                    )
                    if ckpt_path:
                        save_market_ppo_checkpoint(
                            ckpt_path,
                            model,
                            policy_hidden_dims=config.policy_hidden,
                            value_hidden_dims=config.value_hidden,
                            obs_dim=input_dim,
                            action_dim=int(model.log_std.shape[0]),
                            val_report=selected_report,
                            val_report_mode=selected_mode,
                            obs_norm_state=train_env.get_obs_norm_state(),
                            selection_epoch=epoch + 1,
                            selection_metrics=selected_sel,
                            selection_key=candidate_key,
                            checkpoint_metric_mode=selected_mode,
                            selection_policy={
                                "primary_metric": "net_pnl_pct",
                                "filters": {
                                    "net_pnl_pct_positive": True,
                                    "max_drawdown_le": guard,
                                },
                                "tie_breakers": ["sharpe_1h", "sortino_1h"],
                            },
                            extra_metadata={
                                "config": config.__dict__,
                                "validation_metadata": {
                                    "deterministic_report": _strip_large_report_fields(deterministic_report),
                                    "stochastic_report": _strip_large_report_fields(stochastic_report),
                                    "deterministic_selection_metrics": dict(deterministic_sel),
                                    "stochastic_selection_metrics": dict(stochastic_sel),
                                    "checkpoint_metric_mode": selected_mode,
                                    "best_report_mode": selected_mode,
                                    "ppo_validation_modes": list(validation_modes),
                                    "stochastic_val_seed": stochastic_val_seed,
                                },
                            },
                        )
                        saved_new_ckpt_this_run = True
        _timing_log(f"epoch={epoch + 1} total_secs={time.perf_counter() - epoch_t0:.4f}")
    if best_report is None:
        print("[mm ckpt] no validation checkpoint satisfied selection filters; no new PPO checkpoint saved.")
        return model, {}, saved_new_ckpt_this_run
    require(best_report_mode is not None, "best_report_mode missing for selected PPO checkpoint")
    require(best_selection_metrics is not None, "selection metrics missing for selected PPO checkpoint")
    require(best_selection_epoch is not None, "selection epoch missing for selected PPO checkpoint")
    best_summary = _build_best_validation_summary(
        best_report,
        best_report_mode=best_report_mode,
        checkpoint_metric_mode=checkpoint_metric_mode,
        selection_metrics=best_selection_metrics,
        selection_epoch=best_selection_epoch,
    )
    return model, best_summary, saved_new_ckpt_this_run


def report_cmssl_metrics(y_true: np.ndarray, cmssl_out: Dict[str, np.ndarray]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    dir_logits = np.asarray(cmssl_out["dir_logits"], dtype=np.float64)
    require(y_true.ndim == 2, f"y_true must be 2D, got shape={y_true.shape}")
    require(dir_logits.ndim == 2, f"cmssl_out['dir_logits'] must be 2D, got shape={dir_logits.shape}")
    require(
        y_true.shape == dir_logits.shape,
        f"y_true shape {y_true.shape} must match dir_logits shape {dir_logits.shape}",
    )
    p_up = _sigmoid(dir_logits)
    y_up = (y_true > 0.0).astype(np.float64)
    p_up = np.clip(p_up, 1e-6, 1.0 - 1e-6)
    bce = float(np.mean(-(y_up * np.log(p_up) + (1.0 - y_up) * np.log(1.0 - p_up))))
    accuracy = float(np.mean((p_up >= 0.5) == (y_up >= 0.5)))
    return {
        "direction_bce": bce,
        "direction_accuracy": accuracy,
    }


def _fill_forward(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = arr.copy()
    isnan = np.isnan(out)
    if not np.any(isnan):
        return out
    idx = np.where(~isnan, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    if np.isnan(out[0]):
        first_valid = np.flatnonzero(~isnan)
        if first_valid.size:
            out[: first_valid[0]] = out[first_valid[0]]
    return out


def build_market_batch(split: Dict[str, np.ndarray]) -> MarketMakingBatch:
    snapshots = split.get("snapshots")
    if snapshots is None:
        raise ValueError("snapshots missing from split data; ensure join_features stores snapshots.")
    if snapshots.ndim != 2 or snapshots.shape[1] < 2:
        raise ValueError(f"Expected snapshots with >=2 columns, got {snapshots.shape}")
    best_bid = snapshots[:, 0].astype(np.float32)
    best_ask = snapshots[:, 1].astype(np.float32)
    best_bid = _fill_forward(best_bid)
    best_ask = _fill_forward(best_ask)
    decision_ts = split.get("ts")
    if decision_ts is not None:
        decision_ts = np.asarray(decision_ts, dtype=np.int64)
        if decision_ts.ndim != 1:
            raise ValueError("split['ts'] must be a 1D array.")
        if decision_ts.shape[0] != split["features"].shape[0]:
            raise ValueError(
                "split['ts'] length mismatch: "
                f"expected {split['features'].shape[0]}, got {decision_ts.shape[0]}"
            )
        _ensure_monotonic(decision_ts, "split")
    future_ret_by_horizon = split.get("y")
    if future_ret_by_horizon is not None:
        future_ret_by_horizon = np.asarray(future_ret_by_horizon, dtype=np.float32)
        if future_ret_by_horizon.ndim != 2:
            raise ValueError("split['y'] must be a 2D horizon target matrix.")
        if future_ret_by_horizon.shape[0] != split["features"].shape[0]:
            raise ValueError(
                "split['y'] row mismatch: "
                f"expected {split['features'].shape[0]}, got {future_ret_by_horizon.shape[0]}"
            )
    return MarketMakingBatch(
        features=split["features"],
        spread_bps=split["spread_bps"],
        best_bid=best_bid,
        best_ask=best_ask,
        future_ret_by_horizon=future_ret_by_horizon,
        decision_ts=decision_ts,
    )


def _resolve_eval_step_ms(env: MarketMakingEnv, steps: int) -> Dict[str, Any]:
    fallback_step_ms = _env_float("BYBIT_MM_SNAPSHOT_STEP_MS", RAW_SNAPSHOT_EXPECTED_STEP_MS)
    decision_ts = env.decision_ts
    if decision_ts is None or decision_ts.size < 2 or steps <= 0:
        return {
            "step_ms": float(fallback_step_ms),
            "source": "env_var_fallback",
            "diff_count": 0,
        }

    eval_count = min(int(steps) + 1, int(decision_ts.size))
    diffs = np.diff(decision_ts[:eval_count])
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return {
            "step_ms": float(fallback_step_ms),
            "source": "env_var_fallback",
            "diff_count": 0,
        }

    return {
        "step_ms": float(np.median(positive_diffs)),
        "source": "decision_ts_median_diff",
        "diff_count": int(positive_diffs.size),
    }


def _inventory_distribution(inventory: np.ndarray) -> Dict[str, float]:
    if inventory.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(inventory)),
        "max": float(np.max(inventory)),
        "mean": float(np.mean(inventory)),
        "std": float(np.std(inventory)),
        "p05": float(np.quantile(inventory, 0.05)),
        "p50": float(np.quantile(inventory, 0.50)),
        "p95": float(np.quantile(inventory, 0.95)),
    }


def evaluate_market_making(
    env: MarketMakingEnv,
    policy_fn,
    *,
    collect_vol_bucket_report: bool = False,
    baseline_cfg: Optional[BaselineQuoteConfig] = None,
) -> Dict[str, Any]:
    bucket_cfg = None
    sigma_bps_selected_steps: List[float] = []
    delta_equity_steps: List[float] = []
    reward_steps: List[float] = []
    maker_buy_steps: List[float] = []
    maker_sell_steps: List[float] = []
    turnover_notional_steps: List[float] = []
    maker_buy_markout_steps: List[float] = []
    maker_sell_markout_steps: List[float] = []
    if collect_vol_bucket_report:
        bucket_cfg = _validate_baseline_quote_config(load_baseline_quote_config() if baseline_cfg is None else baseline_cfg)
    obs = env.reset()
    equity_curve: List[float] = []
    inventory_curve: List[float] = []
    turnover_qty = 0.0
    turnover_notional = 0.0
    taker_notional = 0.0
    taker_fee_total = 0.0
    maker_rebate_total = 0.0
    maker_fill_count = 0
    maker_opps = 0
    taker_steps = 0
    steps = 0
    total_reward = 0.0
    total_delta_equity = 0.0
    inventory_penalty_total = 0.0
    total_turnover_penalty = 0.0
    total_maker_buy_markout = 0.0
    total_maker_sell_markout = 0.0
    maker_buy_clipped_steps = 0
    maker_sell_clipped_steps = 0
    taker_buy_clipped_steps = 0
    taker_sell_clipped_steps = 0
    last_mid = 0.0
    initial_equity = env.prev_equity

    done = False
    while not done:
        idx_before_step = env.idx
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        equity_curve.append(info["equity"])
        inventory_curve.append(info["inventory"])
        steps += 1
        total_reward += float(reward)
        total_delta_equity += float(info.get("delta_equity", 0.0))
        step_inventory_penalty_total = info.get("inventory_penalty_total")
        if step_inventory_penalty_total is None:
            step_inventory_penalty_total = info.get("inventory_penalty", 0.0)
        inventory_penalty_total += float(step_inventory_penalty_total)
        total_turnover_penalty += float(info.get("turnover_penalty", 0.0))
        total_maker_buy_markout += float(info.get("maker_buy_markout", 0.0))
        total_maker_sell_markout += float(info.get("maker_sell_markout", 0.0))
        maker_buy_clipped_steps += int(bool(info.get("maker_buy_clipped", False)))
        maker_sell_clipped_steps += int(bool(info.get("maker_sell_clipped", False)))
        taker_buy_clipped_steps += int(bool(info.get("taker_buy_clipped", False)))
        taker_sell_clipped_steps += int(bool(info.get("taker_sell_clipped", False)))
        last_mid = float(info.get("mid", 0.0))

        maker_buy = abs(float(info["maker_buy"]))
        maker_sell = abs(float(info["maker_sell"]))
        taker_buy = abs(float(info["taker_buy"]))
        taker_sell = abs(float(info["taker_sell"]))
        step_qty = maker_buy + maker_sell + taker_buy + taker_sell
        turnover_qty += step_qty
        maker_notional = maker_buy * float(info.get("bid", 0.0)) + maker_sell * float(info.get("ask", 0.0))
        step_taker_notional = taker_buy * float(env.best_ask[env.idx]) + taker_sell * float(env.best_bid[env.idx])
        step_notional = maker_notional + step_taker_notional
        turnover_notional += step_notional
        taker_notional += step_taker_notional
        taker_fee_total += float(info.get("taker_fee", 0.0))
        maker_rebate_total += float(info.get("rebate", 0.0))
        maker_fill_count += int(info["maker_buy"] > 0.0) + int(info["maker_sell"] > 0.0)
        maker_opps += 2
        taker_steps += int(info["taker_buy"] > 0.0 or info["taker_sell"] > 0.0)
        if collect_vol_bucket_report and bucket_cfg is not None:
            snapshot_row = env.features[idx_before_step, env._feature_layout["snapshots"]]
            vol_short = float(snapshot_row[RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_short")])
            vol_long = float(snapshot_row[RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_long")])
            sigma_bps_selected_steps.append(1e4 * max(0.0, vol_short, vol_long))
            delta_equity_steps.append(float(info.get("delta_equity", 0.0)))
            reward_steps.append(float(reward))
            maker_buy_steps.append(maker_buy)
            maker_sell_steps.append(maker_sell)
            turnover_notional_steps.append(float(step_notional))
            maker_buy_markout_steps.append(float(info.get("maker_buy_markout", 0.0)))
            maker_sell_markout_steps.append(float(info.get("maker_sell_markout", 0.0)))

    equity_arr = np.array(equity_curve, dtype=np.float32)
    # Per-snapshot percentage returns; annualization uses the snapshot cadence.
    prev_equity = np.concatenate([[initial_equity], equity_arr[:-1]])
    returns = np.divide(
        equity_arr,
        prev_equity,
        out=np.zeros_like(equity_arr),
        where=prev_equity != 0,
    ) - 1.0
    cadence = _resolve_eval_step_ms(env, steps)
    step_ms = float(cadence["step_ms"])
    steps_per_year = _steps_per_year_from_snapshot_ms(step_ms)
    sharpe = compute_sharpe(returns, steps_per_year)
    pnl_changes, capital_returns = compute_capital_returns(equity_arr, initial_equity)

    ts_source = "env.decision_ts"
    ts_raw = getattr(env, "decision_ts", None)
    if ts_raw is None:
        ts_source = "synthetic_from_cadence"
        ts_ms = np.arange(steps, dtype=np.int64) * int(max(step_ms, 1.0))
    else:
        ts_arr = np.asarray(ts_raw, dtype=np.int64)
        # Equity points are post-step (next_idx), so decision_ts must be shifted by +1 for bucketed metrics.
        ts_shifted = ts_arr[1:steps + 1]
        if ts_shifted.size == steps:
            ts_ms = ts_shifted
        else:
            ts_source = "synthetic_from_cadence"
            ts_ms = np.arange(steps, dtype=np.int64) * int(max(step_ms, 1.0))

    capital_returns_5m = aggregate_returns_by_time(ts_ms, capital_returns, 5 * 60 * 1000)
    capital_returns_1h = aggregate_returns_by_time(ts_ms, capital_returns, 60 * 60 * 1000)
    sharpe_5m = compute_sharpe(capital_returns_5m, 365.0 * 24.0 * 12.0)
    sharpe_1h = compute_sharpe(capital_returns_1h, 365.0 * 24.0)
    sortino_5m = compute_sortino(capital_returns_5m, 365.0 * 24.0 * 12.0)
    sortino_1h = compute_sortino(capital_returns_1h, 365.0 * 24.0)
    max_drawdown = compute_max_drawdown(equity_arr)
    final_equity = float(equity_arr[-1]) if equity_arr.size > 0 else float(initial_equity)
    pnl_curve = equity_arr - float(initial_equity)
    maker_fill_rate = float(maker_fill_count / maker_opps) if maker_opps > 0 else 0.0
    taker_usage_frequency = float(taker_steps / steps) if steps > 0 else 0.0
    taker_volume_share = float(taker_notional / turnover_notional) if turnover_notional > 0 else 0.0
    gross_taker_fees_paid = float(taker_fee_total)
    gross_maker_rebates_earned = float(maker_rebate_total)
    net_fee_cost = float(taker_fee_total - maker_rebate_total)
    # Compatibility: fee_drag now reflects net fees (taker fees minus maker rebates).
    fee_drag = float(net_fee_cost / turnover_notional) if turnover_notional > 0 else 0.0
    net_fee_bps_on_turnover = float(1e4 * net_fee_cost / turnover_notional) if turnover_notional > 0 else 0.0
    net_fee_pct_initial_equity = float(net_fee_cost / max(initial_equity, 1e-12))
    inventory_arr = np.array(inventory_curve, dtype=np.float32)
    denom = max(float(initial_equity), 1e-12)
    net_pnl = float(final_equity - float(initial_equity))
    net_pnl_pct = float(net_pnl / denom)
    gross_pnl = float(net_pnl + net_fee_cost)
    gross_pnl_pct = float(gross_pnl / denom)
    ending_inventory_qty = float(inventory_arr[-1]) if inventory_arr.size > 0 else 0.0
    ending_inventory_notional = float(abs(ending_inventory_qty * last_mid))
    maker_turnover_notional = float(turnover_notional - taker_notional)
    maker_turnover_share = float(maker_turnover_notional / turnover_notional) if turnover_notional > 0 else 0.0

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "gross_pnl": gross_pnl,
        "gross_pnl_pct": gross_pnl_pct,
        "equity_curve": equity_arr,
        "pnl_curve": pnl_curve,
        "sharpe": sharpe,
        "sharpe_5m": sharpe_5m,
        "sharpe_1h": sharpe_1h,
        "sortino_5m": sortino_5m,
        "sortino_1h": sortino_1h,
        "max_drawdown": max_drawdown,
        "turnover_qty": float(turnover_qty),
        "turnover_notional": float(turnover_notional),
        "maker_turnover_notional": maker_turnover_notional,
        "maker_turnover_share": maker_turnover_share,
        "maker_fill_rate": maker_fill_rate,
        "maker_fill_count": int(maker_fill_count),
        "maker_opportunities": int(maker_opps),
        "taker_usage_frequency": taker_usage_frequency,
        "taker_volume_share": taker_volume_share,
        "taker_steps": int(taker_steps),
        "steps": int(steps),
        "gross_taker_fees_paid": gross_taker_fees_paid,
        "gross_maker_rebates_earned": gross_maker_rebates_earned,
        "net_fee_cost": net_fee_cost,
        "net_fee_bps_on_turnover": net_fee_bps_on_turnover,
        "net_fee_pct_initial_equity": net_fee_pct_initial_equity,
        "fee_drag": fee_drag,
        "ending_inventory_qty": ending_inventory_qty,
        "ending_inventory_notional": ending_inventory_notional,
        "total_reward": float(total_reward),
        "total_delta_equity": float(total_delta_equity),
        "inventory_penalty_total": float(inventory_penalty_total),
        "total_turnover_penalty": float(total_turnover_penalty),
        "total_maker_buy_markout": float(total_maker_buy_markout),
        "total_maker_sell_markout": float(total_maker_sell_markout),
        "maker_buy_clipped_steps": int(maker_buy_clipped_steps),
        "maker_sell_clipped_steps": int(maker_sell_clipped_steps),
        "taker_buy_clipped_steps": int(taker_buy_clipped_steps),
        "taker_sell_clipped_steps": int(taker_sell_clipped_steps),
        "inventory_distribution": _inventory_distribution(inventory_arr),
        "cadence": {
            "step_ms": step_ms,
            "steps_per_year": float(steps_per_year),
            "source": cadence["source"],
            "diff_count": cadence["diff_count"],
            "timestamp_source": ts_source,
        },
    }
    if collect_vol_bucket_report and bucket_cfg is not None:
        metrics.update(
            build_vol_bucket_report(
                sigma_bps_selected=np.asarray(sigma_bps_selected_steps, dtype=np.float64),
                delta_equity_per_step=np.asarray(delta_equity_steps, dtype=np.float64),
                reward_per_step=np.asarray(reward_steps, dtype=np.float64),
                maker_buy_per_step=np.asarray(maker_buy_steps, dtype=np.float64),
                maker_sell_per_step=np.asarray(maker_sell_steps, dtype=np.float64),
                turnover_notional_per_step=np.asarray(turnover_notional_steps, dtype=np.float64),
                maker_buy_markout_per_step=np.asarray(maker_buy_markout_steps, dtype=np.float64),
                maker_sell_markout_per_step=np.asarray(maker_sell_markout_steps, dtype=np.float64),
                initial_equity=float(initial_equity),
            )
        )
    return metrics


def _format_mm_summary(label: str, metrics: Dict[str, Any]) -> str:
    inv = metrics.get("inventory_distribution") or {}
    return (
        f"{label}: final_equity={float(metrics.get('final_equity', 0.0)):.4f} "
        f"net_pnl={float(metrics.get('net_pnl', 0.0)):.4f} "
        f"net_pnl_pct={float(metrics.get('net_pnl_pct', 0.0)):.6f} "
        f"gross_pnl={float(metrics.get('gross_pnl', 0.0)):.4f} "
        f"gross_pnl_pct={float(metrics.get('gross_pnl_pct', 0.0)):.6f} "
        f"sharpe={float(metrics.get('sharpe', 0.0)):.4f} "
        f"sharpe_5m={float(metrics.get('sharpe_5m', 0.0)):.4f} "
        f"sharpe_1h={float(metrics.get('sharpe_1h', 0.0)):.4f} "
        f"sortino_5m={float(metrics.get('sortino_5m', 0.0)):.4f} "
        f"sortino_1h={float(metrics.get('sortino_1h', 0.0)):.4f} "
        f"max_dd={float(metrics.get('max_drawdown', 0.0)):.4f} "
        f"turnover_notional={float(metrics.get('turnover_notional', 0.0)):.4f} "
        f"turnover_qty={float(metrics.get('turnover_qty', 0.0)):.4f} "
        f"maker_fill_rate={float(metrics.get('maker_fill_rate', 0.0)):.4f} "
        f"taker_usage_freq={float(metrics.get('taker_usage_frequency', 0.0)):.4f} "
        f"taker_volume_share={float(metrics.get('taker_volume_share', 0.0)):.4f} "
        f"gross_taker_fees_paid={float(metrics.get('gross_taker_fees_paid', 0.0)):.4f} "
        f"gross_maker_rebates_earned={float(metrics.get('gross_maker_rebates_earned', 0.0)):.4f} "
        f"net_fee_cost={float(metrics.get('net_fee_cost', 0.0)):.4f} "
        f"fee_drag={float(metrics.get('fee_drag', 0.0)):.6f} "
        f"net_fee_bps_on_turnover={float(metrics.get('net_fee_bps_on_turnover', 0.0)):.4f} "
        f"inventory_penalty_total={float(metrics.get('inventory_penalty_total', 0.0)):.4f} "
        f"inv[min={float(inv.get('min', 0.0)):.2f}, p50={float(inv.get('p50', 0.0)):.2f}, max={float(inv.get('max', 0.0)):.2f}]"
    )


def _summarize_array_for_log(arr: np.ndarray) -> Dict[str, Any]:
    arr_np = np.asarray(arr)
    summary: Dict[str, Any] = {
        "type": "ndarray",
        "dtype": str(arr_np.dtype),
        "shape": list(arr_np.shape),
    }
    if arr_np.size > 0 and np.issubdtype(arr_np.dtype, np.number):
        summary.update(
            {
                "min": float(np.min(arr_np)),
                "max": float(np.max(arr_np)),
            }
        )
    else:
        summary["size"] = int(arr_np.size)
    return summary


def _summarize_for_log(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _summarize_array_for_log(value)
    if isinstance(value, dict):
        return {k: _summarize_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_for_log(v) for v in value]
    return value


def load_market_policy(
    input_dim: int,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
    require_checkpoint: bool = False,
    checkpoint_data: Optional[Any] = None,
) -> Optional[MarketPolicyNet]:
    """Load a deterministic market policy (mean-action inference only)."""
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    if not path.exists():
        if require_checkpoint:
            raise FileNotFoundError(f"Market policy checkpoint not found: {ckpt_path}")
        warnings.warn(
            f"Market policy checkpoint not found: {ckpt_path}. Falling back to baseline policy.",
            RuntimeWarning,
        )
        return None
    ckpt = (
        checkpoint_data
        if checkpoint_data is not None
        else _torch_load_trusted_checkpoint(path, map_location=device)
    )
    ppo_model = load_market_ppo_model(
        input_dim,
        device=device,
        ckpt_path=ckpt_path,
        require_checkpoint=require_checkpoint,
        checkpoint_data=ckpt,
    )
    if ppo_model is None:
        return None
    setattr(ppo_model.policy_net, "checkpoint_path", str(path.expanduser().resolve()))
    print(
        "[mm deterministic policy] "
        f"path={path.expanduser().resolve()} "
        "action_semantics=bounded_harmonized source=code"
    )
    return ppo_model.policy_net


def _fit_baseline_alpha_calibration(
    prepared_batch: PreparedFastBaselineBatch,
    horizons_ms: Sequence[int],
    *,
    eps: float = 1e-12,
) -> BaselineAlphaCalibration:
    score_by_horizon = np.asarray(2.0 * prepared_batch.p_up_by_horizon - 1.0, dtype=np.float64)
    target_bps_by_horizon = np.asarray(prepared_batch.future_ret_by_horizon, dtype=np.float64) * 1e4
    if score_by_horizon.shape != target_bps_by_horizon.shape:
        raise ValueError(
            "Calibration inputs must have matching score/target shapes: "
            f"score_shape={score_by_horizon.shape} target_shape={target_bps_by_horizon.shape}"
        )
    num_h = score_by_horizon.shape[1]
    slope_by_horizon = np.zeros(num_h, dtype=np.float64)
    winsor_lower = np.zeros(num_h, dtype=np.float64)
    winsor_upper = np.zeros(num_h, dtype=np.float64)
    fit_count = np.zeros(num_h, dtype=np.int64)
    score_mean = np.zeros(num_h, dtype=np.float64)
    target_mean_bps = np.zeros(num_h, dtype=np.float64)
    target_std_bps = np.zeros(num_h, dtype=np.float64)
    winsorized_fraction = np.zeros(num_h, dtype=np.float64)
    for h in range(num_h):
        score_h = score_by_horizon[:, h]
        target_h = target_bps_by_horizon[:, h]
        valid = np.isfinite(score_h) & np.isfinite(target_h)
        fit_count[h] = int(np.sum(valid))
        if fit_count[h] == 0:
            continue
        score_valid = score_h[valid]
        target_valid = target_h[valid]
        lower = float(np.quantile(target_valid, 0.01))
        upper = float(np.quantile(target_valid, 0.99))
        winsor_lower[h] = lower
        winsor_upper[h] = upper
        target_winsor = np.clip(target_valid, lower, upper)
        denom = float(np.sum(score_valid * score_valid))
        slope_by_horizon[h] = float(np.sum(score_valid * target_winsor) / max(denom, eps))
        score_mean[h] = float(np.mean(score_valid))
        target_mean_bps[h] = float(np.mean(target_winsor))
        target_std_bps[h] = float(np.std(target_winsor))
        winsorized_fraction[h] = float(np.mean(target_valid != target_winsor))
    return BaselineAlphaCalibration(
        score_to_bps_slope_by_horizon=slope_by_horizon,
        winsor_lower_bps_by_horizon=winsor_lower,
        winsor_upper_bps_by_horizon=winsor_upper,
        fit_count_by_horizon=fit_count,
        horizons_ms=np.asarray(horizons_ms, dtype=np.int64),
        diagnostics={
            "score_mean_by_horizon": score_mean,
            "target_mean_bps_by_horizon": target_mean_bps,
            "target_std_bps_by_horizon": target_std_bps,
            "winsorized_fraction_by_horizon": winsorized_fraction,
            "eps": float(eps),
            "fit_split": prepared_batch.split_name,
        },
    )


def prepare_fast_baseline_batch(batch: MarketMakingBatch, meta: Dict[str, Any], *, env_kwargs_common: Dict[str, Any], split_name: str) -> PreparedFastBaselineBatch:
    """Precompute baseline-only arrays once so each trial only runs quote math + fills."""
    num_h = _infer_num_horizons(batch.features.shape[-1])
    horizons_ms = _validate_fixed_cmssl_horizons(_normalize_horizons(num_h, meta.get("horizons_ms", DEFAULT_MM_HORIZONS_MS)))
    layout = _joined_feature_layout(num_h, len(RAW_SNAPSHOT_FEATURE_COLUMNS))
    dir_logits_by_horizon = np.asarray(batch.features[:, layout["dir_logits"]], dtype=np.float64)
    p_up_by_horizon = np.asarray(batch.features[:, layout["p_up"]], dtype=np.float64)
    direction_confidence_by_horizon = np.abs(2.0 * p_up_by_horizon - 1.0)
    if batch.future_ret_by_horizon is None:
        raise ValueError("MarketMakingBatch.future_ret_by_horizon is required for baseline calibration.")
    future_ret_by_horizon = np.asarray(batch.future_ret_by_horizon, dtype=np.float64)
    if future_ret_by_horizon.shape != p_up_by_horizon.shape:
        raise ValueError(
            "Future-return horizon target matrix must align with model horizon outputs: "
            f"targets_shape={future_ret_by_horizon.shape} p_up_shape={p_up_by_horizon.shape}"
        )
    snapshot_block = np.asarray(batch.features[:, layout["snapshots"]], dtype=np.float64)
    spread_idx = RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")
    observed_spread_bps = snapshot_block[:, spread_idx]
    vol_short = snapshot_block[:, RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_short")]
    vol_long = snapshot_block[:, RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_long")]
    best_bid = np.asarray(batch.best_bid, dtype=np.float64)
    best_ask = np.asarray(batch.best_ask, dtype=np.float64)
    mid = 0.5 * (best_bid + best_ask)
    best_bid_next = np.concatenate([best_bid[1:], best_bid[-1:]])
    best_ask_next = np.concatenate([best_ask[1:], best_ask[-1:]])
    best_bid_prev = np.concatenate([best_bid[:1], best_bid[:-1]])
    best_ask_prev = np.concatenate([best_ask[:1], best_ask[:-1]])
    mid_next = np.concatenate([mid[1:], mid[-1:]])
    return PreparedFastBaselineBatch(
        split_name=split_name,
        rows=int(batch.features.shape[0]),
        initial_cash=float(_env_float("BYBIT_MM_INITIAL_CASH", DEFAULT_MM_INITIAL_CASH)),
        fill_size=float(env_kwargs_common["fill_size"]),
        maker_rebate_bps=float(env_kwargs_common["maker_rebate_bps"]),
        taker_fee_bps=float(env_kwargs_common["taker_fee_bps"]),
        fill_tolerance=float(env_kwargs_common["fill_tolerance"]),
        delta_bps_limit=float(env_kwargs_common["delta_bps_limit"]),
        inventory_penalty=float(env_kwargs_common["inventory_penalty"]),
        inv_soft_notional=float(env_kwargs_common["inv_soft_notional"]),
        lambda_inv=float(env_kwargs_common["lambda_inv"]),
        lambda_turn=float(env_kwargs_common["lambda_turn"]),
        max_inventory_notional=float(env_kwargs_common["max_inventory_notional"]),
        hard_max_inventory_notional=float(env_kwargs_common["hard_max_inventory_notional"]),
        fill_ema_alpha=2.0 / float(max(1, _env_int("BYBIT_MM_FILL_EMA_WINDOW_STEPS", DEFAULT_MM_FILL_EMA_WINDOW_STEPS)) + 1.0),
        best_bid=best_bid,
        best_ask=best_ask,
        best_bid_next=best_bid_next,
        best_ask_next=best_ask_next,
        best_bid_prev=best_bid_prev,
        best_ask_prev=best_ask_prev,
        mid=mid,
        mid_next=mid_next,
        observed_spread_bps=observed_spread_bps,
        decision_ts=None if batch.decision_ts is None else np.asarray(batch.decision_ts, dtype=np.int64),
        dir_logits_by_horizon=dir_logits_by_horizon,
        p_up_by_horizon=p_up_by_horizon,
        direction_confidence_by_horizon=direction_confidence_by_horizon,
        future_ret_by_horizon=future_ret_by_horizon,
        p250=p_up_by_horizon[:, _resolve_horizon_index(250, horizons_ms, label="p250")],
        p500=p_up_by_horizon[:, _resolve_horizon_index(500, horizons_ms, label="p500")],
        p1000=p_up_by_horizon[:, _resolve_horizon_index(1000, horizons_ms, label="p1000")],
        vol_short=vol_short,
        vol_long=vol_long,
    )

def _fast_kernel_impl(best_bid, best_ask, best_bid_next, best_ask_next, best_bid_prev, best_ask_prev, mid, mid_next, observed_spread_bps, vol_short, vol_long, p250, p500, p1000, decision_ts, initial_cash, fill_size, maker_rebate_bps, fill_tolerance, inventory_penalty, inv_soft_notional, lambda_inv, lambda_turn, max_inventory_notional, hard_max_inventory_notional, base_half_spread_bps, obs_spread_anchor_frac, alpha_center_scale, inventory_center_scale, vol_width_scale, uncertainty_width_scale, inventory_side_widen_scale, spread_floor_bps, spread_cap_bps, inv_ref_notional, p250_weight, p500_weight, p1000_weight, score_to_bps_slope_250, score_to_bps_slope_500, score_to_bps_slope_1000):
    steps = len(mid)
    equity_curve = np.zeros(steps, dtype=np.float64)
    inventory_curve = np.zeros(steps, dtype=np.float64)
    delta_equity_curve = np.zeros(steps, dtype=np.float64)
    reward_curve = np.zeros(steps, dtype=np.float64)
    maker_buy_curve = np.zeros(steps, dtype=np.float64)
    maker_sell_curve = np.zeros(steps, dtype=np.float64)
    turnover_notional_curve = np.zeros(steps, dtype=np.float64)
    maker_buy_markout_curve = np.zeros(steps, dtype=np.float64)
    maker_sell_markout_curve = np.zeros(steps, dtype=np.float64)
    cash = initial_cash
    inventory = 0.0
    prev_equity = initial_cash
    total_reward = 0.0
    total_delta_equity = 0.0
    inventory_penalty_total = 0.0
    total_turnover_penalty = 0.0
    total_maker_buy_markout = 0.0
    total_maker_sell_markout = 0.0
    turnover_qty = 0.0
    turnover_notional = 0.0
    maker_rebate_total = 0.0
    maker_fill_count = 0
    maker_buy_fills = 0
    maker_sell_fills = 0
    maker_buy_clipped_steps = 0
    maker_sell_clipped_steps = 0
    inventory_abs_sum = 0.0
    inventory_abs_max = 0.0
    for idx in range(steps):
        mid_i = mid[idx]
        score_250 = 2.0 * p250[idx] - 1.0
        score_500 = 2.0 * p500[idx] - 1.0
        score_1000 = 2.0 * p1000[idx] - 1.0
        alpha_center_bps = alpha_center_scale * (
            p250_weight * score_to_bps_slope_250 * score_250
            + p500_weight * score_to_bps_slope_500 * score_500
            + p1000_weight * score_to_bps_slope_1000 * score_1000
        )
        observed_spread_bps_i = observed_spread_bps[idx]
        obs_anchor_bps = 0.0
        if np.isfinite(observed_spread_bps_i) and observed_spread_bps_i > 0.0:
            obs_anchor_bps = max(0.0, obs_spread_anchor_frac * observed_spread_bps_i)
        sigma_bps = 1e4 * max(0.0, vol_short[idx], vol_long[idx])
        disagreement = max(abs(p250[idx] - p500[idx]), abs(p500[idx] - p1000[idx]), abs(p250[idx] - p1000[idx]))
        uncertainty_ref_bps = max(base_half_spread_bps, obs_anchor_bps, sigma_bps, spread_floor_bps)
        base_component_bps = max(base_half_spread_bps, obs_anchor_bps, spread_floor_bps)
        vol_width_bps = vol_width_scale * sigma_bps
        uncertainty_width_bps = uncertainty_width_scale * disagreement * uncertainty_ref_bps
        half_spread_bps = base_component_bps + vol_width_bps + uncertainty_width_bps
        if half_spread_bps < spread_floor_bps:
            half_spread_bps = spread_floor_bps
        if half_spread_bps > spread_cap_bps:
            half_spread_bps = spread_cap_bps
        inv_notional = inventory * mid_i
        inventory_center_bps = inventory_center_scale * (inv_notional / inv_ref_notional)
        reservation_bps = alpha_center_bps - inventory_center_bps
        inv_soft_notional_eps = 1e-12
        soft_ratio = abs(inv_notional) / max(inv_soft_notional, inv_soft_notional_eps)
        overload = max(0.0, soft_ratio - 1.0)
        inventory_side_extra_bps = (
            inventory_side_widen_scale * overload * max(half_spread_bps, 1.0)
        )
        bid_half_spread_bps = half_spread_bps + (inventory_side_extra_bps if inv_notional > 0.0 else 0.0)
        ask_half_spread_bps = half_spread_bps + (inventory_side_extra_bps if inv_notional < 0.0 else 0.0)
        bid_enabled = inv_notional < max_inventory_notional
        ask_enabled = inv_notional > -max_inventory_notional
        bid = np.nan
        ask = np.nan
        if bid_enabled:
            bid = mid_i + mid_i * (reservation_bps - bid_half_spread_bps) * 1e-4
        if ask_enabled:
            ask = mid_i + mid_i * (reservation_bps + ask_half_spread_bps) * 1e-4
        eps = max(1e-8, mid_i * 1e-6)
        curr_best_bid = best_bid[idx]
        curr_best_ask = best_ask[idx]
        if bid_enabled and bid > curr_best_ask - eps:
            bid = curr_best_ask - eps
        if ask_enabled and ask < curr_best_bid + eps:
            ask = curr_best_bid + eps
        if bid_enabled and ask_enabled and bid >= ask:
            bid = curr_best_bid
            ask = curr_best_ask
        mid_cap = mid_next[idx]
        cap_qty = hard_max_inventory_notional / max(mid_cap, 1e-12)
        buy_fill = 0.0
        sell_fill = 0.0
        buy_clip = 0.0
        sell_clip = 0.0
        if bid_enabled and best_ask_next[idx] <= bid + fill_tolerance:
            room = max(0.0, cap_qty - inventory)
            buy_fill = min(fill_size, room)
            buy_clip = fill_size - buy_fill
            cash -= bid * buy_fill
            inventory += buy_fill
        if ask_enabled and best_bid_next[idx] >= ask - fill_tolerance:
            room = max(0.0, cap_qty + inventory)
            sell_fill = min(fill_size, room)
            sell_clip = fill_size - sell_fill
            cash += ask * sell_fill
            inventory -= sell_fill
        touch_tolerance = max(fill_tolerance, 1e-9)
        if bid_enabled and buy_fill == 0.0 and abs(bid - best_bid_prev[idx]) <= touch_tolerance and best_bid_next[idx] < best_bid_prev[idx] - 1e-9:
            room = max(0.0, cap_qty - inventory)
            buy_fill = min(fill_size, room)
            buy_clip = fill_size - buy_fill
            cash -= bid * buy_fill
            inventory += buy_fill
        if ask_enabled and sell_fill == 0.0 and abs(ask - best_ask_prev[idx]) <= touch_tolerance and best_ask_next[idx] > best_ask_prev[idx] + 1e-9:
            room = max(0.0, cap_qty + inventory)
            sell_fill = min(fill_size, room)
            sell_clip = fill_size - sell_fill
            cash += ask * sell_fill
            inventory -= sell_fill
        maker_notional = buy_fill * bid + sell_fill * ask
        rebate = maker_notional * maker_rebate_bps * 1e-4
        cash += rebate
        equity = cash + inventory * mid_next[idx]
        delta_equity = equity - prev_equity
        inv_abs_notional = abs(inventory * mid_next[idx])
        penalty = 0.0
        if inv_abs_notional > max_inventory_notional:
            penalty += inventory_penalty * (inv_abs_notional - max_inventory_notional)
        excess_notional = max(0.0, inv_abs_notional - inv_soft_notional)
        inv_penalty = lambda_inv * (excess_notional / inv_soft_notional) ** 2 if inv_soft_notional > 0.0 else 0.0
        turnover_penalty = lambda_turn * (buy_fill + sell_fill)
        reward = delta_equity - max(penalty, inv_penalty) - turnover_penalty
        equity_curve[idx] = equity
        inventory_curve[idx] = inventory
        delta_equity_curve[idx] = delta_equity
        reward_curve[idx] = reward
        maker_buy_curve[idx] = buy_fill
        maker_sell_curve[idx] = sell_fill
        step_turnover_notional = buy_fill * bid + sell_fill * ask
        turnover_notional_curve[idx] = step_turnover_notional
        maker_buy_markout = (mid_next[idx] - bid) * buy_fill if buy_fill > 0.0 else 0.0
        maker_sell_markout = (ask - mid_next[idx]) * sell_fill if sell_fill > 0.0 else 0.0
        maker_buy_markout_curve[idx] = maker_buy_markout
        maker_sell_markout_curve[idx] = maker_sell_markout
        turnover_qty += buy_fill + sell_fill
        turnover_notional += step_turnover_notional
        maker_rebate_total += rebate
        maker_fill_count += (1 if buy_fill > 0.0 else 0) + (1 if sell_fill > 0.0 else 0)
        maker_buy_fills += 1 if buy_fill > 0.0 else 0
        maker_sell_fills += 1 if sell_fill > 0.0 else 0
        maker_buy_clipped_steps += 1 if buy_clip > 0.0 else 0
        maker_sell_clipped_steps += 1 if sell_clip > 0.0 else 0
        total_reward += reward
        total_delta_equity += delta_equity
        inventory_penalty_total += max(penalty, inv_penalty)
        total_turnover_penalty += turnover_penalty
        total_maker_buy_markout += maker_buy_markout
        total_maker_sell_markout += maker_sell_markout
        inventory_abs_sum += inv_abs_notional
        if inv_abs_notional > inventory_abs_max:
            inventory_abs_max = inv_abs_notional
        prev_equity = equity
    step_ms = float(RAW_SNAPSHOT_EXPECTED_STEP_MS)
    diff_count = 0
    if decision_ts is not None and len(decision_ts) >= 2:
        diffs = np.diff(decision_ts)
        positive = diffs[diffs > 0]
        if len(positive) > 0:
            step_ms = float(np.median(positive))
            diff_count = int(len(positive))
    return (equity_curve, inventory_curve, delta_equity_curve, reward_curve, maker_buy_curve, maker_sell_curve, turnover_notional_curve, maker_buy_markout_curve, maker_sell_markout_curve, turnover_qty, turnover_notional, maker_rebate_total, maker_fill_count, maker_buy_fills, maker_sell_fills, total_reward, total_delta_equity, inventory_penalty_total, total_turnover_penalty, total_maker_buy_markout, total_maker_sell_markout, maker_buy_clipped_steps, maker_sell_clipped_steps, inventory_abs_sum, inventory_abs_max, step_ms, diff_count)

_evaluate_prepared_baseline_fast_kernel_py = _fast_kernel_impl
if HAS_NUMBA:
    _evaluate_prepared_baseline_fast_kernel_numba = numba.njit(cache=True)(_fast_kernel_impl)
else:
    _evaluate_prepared_baseline_fast_kernel_numba = None


def evaluate_prepared_baseline_fast(prepared_batch: PreparedFastBaselineBatch, quote_cfg: BaselineQuoteConfig, baseline_alpha_calibration: BaselineAlphaCalibration, *, use_numba: Optional[bool] = None) -> Dict[str, Any]:
    quote_cfg = _validate_baseline_quote_config(quote_cfg)
    kernel = _evaluate_prepared_baseline_fast_kernel_py
    backend = "python"
    allow_numba = _env_bool("BYBIT_MM_BASELINE_FAST_NUMBA", HAS_NUMBA) if use_numba is None else bool(use_numba)
    if allow_numba and _evaluate_prepared_baseline_fast_kernel_numba is not None:
        kernel = _evaluate_prepared_baseline_fast_kernel_numba
        backend = "numba"
    print(f"[baseline fast] backend={backend} split={prepared_batch.split_name}")
    # Pass the precomputed previous-book arrays directly; shifted current arrays break touch/move-away parity.
    result = kernel(
        prepared_batch.best_bid[:-1], prepared_batch.best_ask[:-1], prepared_batch.best_bid_next[:-1], prepared_batch.best_ask_next[:-1], prepared_batch.best_bid_prev[:-1], prepared_batch.best_ask_prev[:-1], prepared_batch.mid[:-1], prepared_batch.mid_next[:-1], prepared_batch.observed_spread_bps[:-1], prepared_batch.vol_short[:-1], prepared_batch.vol_long[:-1], prepared_batch.p250[:-1], prepared_batch.p500[:-1], prepared_batch.p1000[:-1], prepared_batch.decision_ts, prepared_batch.initial_cash, prepared_batch.fill_size, prepared_batch.maker_rebate_bps, prepared_batch.fill_tolerance, prepared_batch.inventory_penalty, prepared_batch.inv_soft_notional, prepared_batch.lambda_inv, prepared_batch.lambda_turn, prepared_batch.max_inventory_notional, prepared_batch.hard_max_inventory_notional, quote_cfg.base_half_spread_bps, quote_cfg.obs_spread_anchor_frac, quote_cfg.alpha_center_scale, quote_cfg.inventory_center_scale, quote_cfg.vol_width_scale, quote_cfg.uncertainty_width_scale, quote_cfg.inventory_side_widen_scale, quote_cfg.spread_floor_bps, quote_cfg.spread_cap_bps, quote_cfg.inv_ref_notional, quote_cfg.p250_weight, quote_cfg.p500_weight, quote_cfg.p1000_weight, baseline_alpha_calibration.score_to_bps_slope_by_horizon[_resolve_horizon_index(250, baseline_alpha_calibration.horizons_ms, label="p250")], baseline_alpha_calibration.score_to_bps_slope_by_horizon[_resolve_horizon_index(500, baseline_alpha_calibration.horizons_ms, label="p500")], baseline_alpha_calibration.score_to_bps_slope_by_horizon[_resolve_horizon_index(1000, baseline_alpha_calibration.horizons_ms, label="p1000")],
    )
    (equity_curve, inventory_curve, delta_equity_curve, reward_curve, maker_buy_curve, maker_sell_curve, turnover_notional_curve, maker_buy_markout_curve, maker_sell_markout_curve, turnover_qty, turnover_notional, maker_rebate_total, maker_fill_count, maker_buy_fills, maker_sell_fills, total_reward, total_delta_equity, inventory_penalty_total, total_turnover_penalty, total_maker_buy_markout, total_maker_sell_markout, maker_buy_clipped_steps, maker_sell_clipped_steps, inventory_abs_sum, inventory_abs_max, step_ms, diff_count) = result
    initial_equity = prepared_batch.initial_cash
    prev_equity = np.concatenate([[initial_equity], equity_curve[:-1]]) if equity_curve.size else np.empty(0, dtype=np.float64)
    returns = np.divide(equity_curve, prev_equity, out=np.zeros_like(equity_curve), where=prev_equity != 0) - 1.0 if equity_curve.size else np.empty(0, dtype=np.float64)
    steps = int(equity_curve.size)
    steps_per_year = _steps_per_year_from_snapshot_ms(step_ms)
    sharpe = compute_sharpe(returns, steps_per_year)
    ts_ms = prepared_batch.decision_ts[1: steps + 1] if prepared_batch.decision_ts is not None and prepared_batch.decision_ts.size >= steps + 1 else np.arange(steps, dtype=np.int64) * int(max(step_ms, 1.0))
    capital_returns_5m = aggregate_returns_by_time(ts_ms, returns, 5 * 60 * 1000)
    capital_returns_1h = aggregate_returns_by_time(ts_ms, returns, 60 * 60 * 1000)
    sharpe_5m = compute_sharpe(capital_returns_5m, 365.0 * 24.0 * 12.0)
    sharpe_1h = compute_sharpe(capital_returns_1h, 365.0 * 24.0)
    sortino_5m = compute_sortino(capital_returns_5m, 365.0 * 24.0 * 12.0)
    sortino_1h = compute_sortino(capital_returns_1h, 365.0 * 24.0)
    max_drawdown = compute_max_drawdown(equity_curve)
    final_equity = float(equity_curve[-1]) if equity_curve.size else float(initial_equity)
    net_pnl = final_equity - initial_equity
    net_pnl_pct = net_pnl / max(initial_equity, 1e-12)
    gross_pnl = net_pnl - maker_rebate_total
    inventory_dist = _inventory_distribution(inventory_curve.astype(np.float32))
    maker_opportunities = 2 * steps
    maker_fill_rate = float(maker_fill_count / maker_opportunities) if maker_opportunities > 0 else 0.0
    ending_inventory_qty = float(inventory_curve[-1]) if inventory_curve.size else 0.0
    ending_inventory_notional = float(abs(ending_inventory_qty * (prepared_batch.mid_next[-2] if steps > 0 else 0.0)))
    metrics = {
        "initial_equity": float(initial_equity), "final_equity": float(final_equity), "net_pnl": float(net_pnl), "net_pnl_pct": float(net_pnl_pct),
        "gross_pnl": float(gross_pnl), "gross_pnl_pct": float(gross_pnl / max(initial_equity, 1e-12)), "equity_curve": equity_curve.astype(np.float32), "pnl_curve": (equity_curve - initial_equity).astype(np.float32),
        "sharpe": float(sharpe), "sharpe_5m": float(sharpe_5m), "sharpe_1h": float(sharpe_1h), "sortino_5m": float(sortino_5m), "sortino_1h": float(sortino_1h),
        "max_drawdown": float(max_drawdown), "max_dd": float(max_drawdown), "turnover_qty": float(turnover_qty), "turnover_notional": float(turnover_notional), "maker_turnover_notional": float(turnover_notional), "maker_turnover_share": 1.0 if turnover_notional > 0 else 0.0,
        "maker_fill_rate": maker_fill_rate, "maker_fill_count": int(maker_fill_count), "maker_opportunities": int(maker_opportunities), "maker_buy_fills": int(maker_buy_fills), "maker_sell_fills": int(maker_sell_fills),
        "taker_usage_frequency": 0.0, "taker_volume_share": 0.0, "taker_steps": 0, "steps": steps,
        "gross_taker_fees_paid": 0.0, "gross_maker_rebates_earned": float(maker_rebate_total), "net_fee_cost": float(-maker_rebate_total), "net_fee_bps_on_turnover": float(-1e4 * maker_rebate_total / turnover_notional) if turnover_notional > 0 else 0.0, "net_fee_pct_initial_equity": float(-maker_rebate_total / max(initial_equity, 1e-12)), "fee_drag": float(-maker_rebate_total / turnover_notional) if turnover_notional > 0 else 0.0,
        "ending_inventory_qty": ending_inventory_qty, "ending_inventory_notional": ending_inventory_notional, "total_reward": float(total_reward), "total_delta_equity": float(total_delta_equity), "inventory_penalty_total": float(inventory_penalty_total), "total_turnover_penalty": float(total_turnover_penalty),
        "total_maker_buy_markout": float(total_maker_buy_markout), "total_maker_sell_markout": float(total_maker_sell_markout), "maker_buy_clipped_steps": int(maker_buy_clipped_steps), "maker_sell_clipped_steps": int(maker_sell_clipped_steps), "taker_buy_clipped_steps": 0, "taker_sell_clipped_steps": 0,
        "inventory_mean_abs_notional": float(inventory_abs_sum / steps) if steps > 0 else 0.0, "inventory_max_abs_notional": float(inventory_abs_max), "inventory_distribution": inventory_dist,
        "cadence": {"step_ms": float(step_ms), "steps_per_year": float(steps_per_year), "source": "decision_ts_median_diff" if diff_count > 0 else "env_var_fallback", "diff_count": int(diff_count), "timestamp_source": "decision_ts" if prepared_batch.decision_ts is not None else "synthetic_from_cadence"},
    }
    metrics.update(
        build_vol_bucket_report(
            sigma_bps_selected=np.asarray(1e4 * np.maximum(0.0, np.maximum(prepared_batch.vol_short[:-1], prepared_batch.vol_long[:-1])), dtype=np.float64),
            delta_equity_per_step=np.asarray(delta_equity_curve, dtype=np.float64),
            reward_per_step=np.asarray(reward_curve, dtype=np.float64),
            maker_buy_per_step=np.asarray(maker_buy_curve, dtype=np.float64),
            maker_sell_per_step=np.asarray(maker_sell_curve, dtype=np.float64),
            turnover_notional_per_step=np.asarray(turnover_notional_curve, dtype=np.float64),
            maker_buy_markout_per_step=np.asarray(maker_buy_markout_curve, dtype=np.float64),
            maker_sell_markout_per_step=np.asarray(maker_sell_markout_curve, dtype=np.float64),
            initial_equity=float(initial_equity),
        )
    )
    return metrics


def compare_fast_vs_env_baseline(context: PreparedBaselineContext, eval_split: str, quote_cfg: BaselineQuoteConfig, *, atol: float = 1e-9, rtol: float = 1e-6) -> None:
    if eval_split not in {"val", "test"}:
        raise ValueError(f"Unsupported baseline split '{eval_split}'")
    fast_batch = context.fast_rl_val_batch if eval_split == "val" else context.fast_rl_test_batch
    batch = context.mm_rl_val_batch if eval_split == "val" else context.mm_rl_test_batch
    if fast_batch is None:
        raise ValueError("Fast baseline batch unavailable for parity check")
    quote_cfg = _validate_baseline_quote_config(quote_cfg)
    env_metrics = evaluate_market_making(
        build_baseline_eval_env(batch, context.env_kwargs_common, quote_cfg, context.baseline_alpha_calibration),
        lambda _obs: (0.0, 0.0, 0.0),
        collect_vol_bucket_report=True,
        baseline_cfg=quote_cfg,
    )
    fast_metrics = evaluate_prepared_baseline_fast(fast_batch, quote_cfg, context.baseline_alpha_calibration)
    for key in ("net_pnl_pct", "maker_fill_rate", "maker_fill_count", "maker_buy_fills", "maker_sell_fills", "turnover_notional", "inventory_mean_abs_notional", "inventory_max_abs_notional"):
        lhs = env_metrics.get(key, env_metrics.get("max_drawdown") if key == "max_dd" else None)
        rhs = fast_metrics.get(key)
        if isinstance(lhs, (int, np.integer)) and lhs != rhs:
            raise AssertionError(f"Baseline parity mismatch {key}: env={lhs} fast={rhs}")
        if lhs is not None and rhs is not None and not np.isclose(float(lhs), float(rhs), atol=atol, rtol=rtol):
            raise AssertionError(f"Baseline parity mismatch {key}: env={lhs} fast={rhs}")


def build_baseline_eval_env(
    batch: MarketMakingBatch,
    env_kwargs_common: Dict[str, Any],
    baseline_quote_config: BaselineQuoteConfig,
    baseline_alpha_calibration: BaselineAlphaCalibration,
) -> MarketMakingEnv:
    return MarketMakingEnv(
        batch,
        allow_taker=False,
        baseline_alpha_calibration=baseline_alpha_calibration,
        baseline_quote_config=baseline_quote_config,
        **env_kwargs_common,
    )


def prepare_baseline_context(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
) -> PreparedBaselineContext:
    print("[baseline prep] building shared CMSSL/RL/eval context")
    meta = dict(load_global_meta(Path(out_root)))
    cmssl_test_split = resolve_cmssl_test_split(meta)
    rl_train_split = resolve_rl_train_split(meta)
    rl_val_split = resolve_rl_val_split(meta)
    rl_test_split = resolve_rl_test_split(meta)
    eval_full_split = resolve_eval_full_split(meta)
    report_cmssl_test_diagnostics(out_root, meta)
    model, _meta = load_cmssl(out_root, ckpt_path, device=device)
    cmssl_batch_size = _resolve_cmssl_batch_size()
    preallocate_join_features = _env_bool("BYBIT_MM_PREALLOCATE_JOIN_FEATURES", False)
    rollout_storage = _resolve_rollout_storage("gpu")
    run_config = {"cmssl_batch_size": cmssl_batch_size, "rollout_storage": rollout_storage, "compile_cmssl": _env_bool("BYBIT_MM_COMPILE_CMSSL", False), "compile_ppo": _env_bool("BYBIT_MM_COMPILE_PPO", False), "tf32": _env_bool("BYBIT_MM_ENABLE_TF32", False), "preallocate_join_features": preallocate_join_features}
    joined_cmssl_test = build_joined_split(out_root, cmssl_test_split, model, meta, device, batch_size=cmssl_batch_size)
    num_h = len(meta.get("horizons_ms", []))
    cmssl_test_metrics = report_cmssl_metrics(joined_cmssl_test["y"], {"dir_logits": joined_cmssl_test["features"][:, :num_h]})
    week3_full_split = {"weeks": rl_train_split["weeks"], "start": rl_train_split["start"], "end": rl_test_split["end"]}
    joined_rl_full = build_joined_split(out_root, week3_full_split, model, meta, device, batch_size=cmssl_batch_size)
    joined_rl_train = slice_joined_by_split(joined_rl_full, rl_train_split)
    joined_rl_val = slice_joined_by_split(joined_rl_full, rl_val_split)
    joined_rl_test = slice_joined_by_split(joined_rl_full, rl_test_split)
    joined_eval_full = build_joined_split(out_root, eval_full_split, model, meta, device, batch_size=cmssl_batch_size)
    env_kwargs_common = resolve_market_env_common_kwargs_from_env()
    mm_rl_train_batch = build_market_batch(joined_rl_train)
    mm_rl_val_batch = build_market_batch(joined_rl_val)
    mm_rl_test_batch = build_market_batch(joined_rl_test)
    mm_eval_full_batch = build_market_batch(joined_eval_full)
    fast_rl_train_batch = prepare_fast_baseline_batch(mm_rl_train_batch, meta, env_kwargs_common=env_kwargs_common, split_name="rl_train")
    fast_rl_val_batch = prepare_fast_baseline_batch(mm_rl_val_batch, meta, env_kwargs_common=env_kwargs_common, split_name="rl_val")
    fast_rl_test_batch = prepare_fast_baseline_batch(mm_rl_test_batch, meta, env_kwargs_common=env_kwargs_common, split_name="rl_test")
    fast_eval_full_batch = prepare_fast_baseline_batch(mm_eval_full_batch, meta, env_kwargs_common=env_kwargs_common, split_name="eval_full")
    baseline_alpha_calibration = _fit_baseline_alpha_calibration(fast_rl_train_batch, meta.get("horizons_ms", DEFAULT_MM_HORIZONS_MS))
    meta["baseline_alpha_calibration"] = {"horizons_ms": baseline_alpha_calibration.horizons_ms.astype(int).tolist(), "score_to_bps_slope_by_horizon": baseline_alpha_calibration.score_to_bps_slope_by_horizon.tolist(), "winsor_lower_bps_by_horizon": baseline_alpha_calibration.winsor_lower_bps_by_horizon.tolist(), "winsor_upper_bps_by_horizon": baseline_alpha_calibration.winsor_upper_bps_by_horizon.tolist(), "fit_count_by_horizon": baseline_alpha_calibration.fit_count_by_horizon.astype(int).tolist(), "diagnostics": _summarize_for_log(baseline_alpha_calibration.diagnostics)}
    return PreparedBaselineContext(out_root=str(out_root), ckpt_path=str(ckpt_path), device=str(device), meta=meta, cmssl_test_split=cmssl_test_split, rl_val_split=rl_val_split, rl_test_split=rl_test_split, eval_full_split=eval_full_split, cmssl_test_metrics=cmssl_test_metrics, joined_rl_rows=int(joined_rl_full["features"].shape[0]), joined_eval_rows=int(joined_eval_full["features"].shape[0]), mm_rl_val_batch=mm_rl_val_batch, mm_rl_test_batch=mm_rl_test_batch, fast_rl_val_batch=fast_rl_val_batch, fast_rl_test_batch=fast_rl_test_batch, fast_rl_train_batch=fast_rl_train_batch, fast_eval_full_batch=fast_eval_full_batch, baseline_alpha_calibration=baseline_alpha_calibration, env_kwargs_common=env_kwargs_common, run_config=run_config, cmssl_batch_size=cmssl_batch_size)


def evaluate_prepared_baseline(
    context: PreparedBaselineContext,
    *,
    eval_split: str,
    baseline_config: Optional[Dict[str, Any] | BaselineQuoteConfig] = None,
    fast_mode: Optional[bool] = None,
    use_numba: Optional[bool] = None,
) -> Dict[str, Any]:
    if eval_split not in {"val", "test"}:
        raise ValueError(f"Unsupported baseline split '{eval_split}'")
    batch = context.mm_rl_val_batch if eval_split == "val" else context.mm_rl_test_batch
    fast_batch = context.fast_rl_val_batch if eval_split == "val" else context.fast_rl_test_batch
    quote_cfg = _resolve_baseline_quote_config(baseline_config)
    quote_cfg = load_baseline_quote_config() if quote_cfg is None else quote_cfg
    fast_enabled = _env_bool("BYBIT_MM_BASELINE_FAST_MODE", True) if fast_mode is None else bool(fast_mode)
    baseline_metrics = evaluate_prepared_baseline_fast(fast_batch, quote_cfg, context.baseline_alpha_calibration, use_numba=use_numba) if fast_enabled and fast_batch is not None else evaluate_market_making(build_baseline_eval_env(batch, context.env_kwargs_common, quote_cfg, context.baseline_alpha_calibration), lambda _obs: (0.0, 0.0, 0.0), collect_vol_bucket_report=True, baseline_cfg=quote_cfg)
    return {"cmssl_test": context.cmssl_test_metrics, "mm_baseline": baseline_metrics, "mm_rl": None, "mm_run_context": {"run_mode": "baseline", "baseline_eval_split": eval_split, "baseline_eval_semantics": f"rl_week3_{eval_split}", "prepared_context_reuse": True, "joined_rl_rows": context.joined_rl_rows, "joined_eval_rows": context.joined_eval_rows, "cmssl_batch_size": context.cmssl_batch_size, "fast_baseline_mode": bool(fast_enabled and fast_batch is not None)}}


def run_pipeline(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    ppo_epochs: int = 10,
    run_mode: str = "train",
) -> Dict[str, Any]:
    print(f"[mm run mode] {run_mode}")
    if run_mode == "baseline":
        prepared_context = prepare_baseline_context(out_root, ckpt_path, device=device)
        return evaluate_prepared_baseline(
            prepared_context,
            eval_split=_resolve_baseline_eval_split(),
            baseline_config=None,
            fast_mode=_env_bool("BYBIT_MM_BASELINE_FAST_MODE", True),
            use_numba=_env_bool("BYBIT_MM_BASELINE_FAST_NUMBA", HAS_NUMBA),
        )
    meta = load_global_meta(Path(out_root))
    cmssl_test_split = resolve_cmssl_test_split(meta)
    rl_train_split = resolve_rl_train_split(meta)
    rl_val_split = resolve_rl_val_split(meta)
    rl_test_split = resolve_rl_test_split(meta)
    eval_full_split = resolve_eval_full_split(meta)

    report_cmssl_test_diagnostics(out_root, meta)

    model, _meta = load_cmssl(out_root, ckpt_path, device=device)

    cmssl_batch_size = _resolve_cmssl_batch_size()
    rollout_storage = _resolve_rollout_storage("gpu")
    pin_rollout_memory = _env_bool("BYBIT_MM_PIN_ROLLOUT_MEMORY", True)
    non_blocking_transfers = _env_bool("BYBIT_MM_NONBLOCKING_TRANSFERS", True)
    preallocate_join_features = _env_bool("BYBIT_MM_PREALLOCATE_JOIN_FEATURES", False)
    _timing_log(
        "run_config "
        f"cmssl_batch_size={cmssl_batch_size} "
        f"rollout_storage={rollout_storage} "
        f"compile_cmssl={_env_bool('BYBIT_MM_COMPILE_CMSSL', False)} "
        f"compile_ppo={_env_bool('BYBIT_MM_COMPILE_PPO', False)} "
        f"tf32={_env_bool('BYBIT_MM_ENABLE_TF32', False)} "
        f"preallocate_join_features={preallocate_join_features}"
    )
    joined_cmssl_test = build_joined_split(
        out_root,
        cmssl_test_split,
        model,
        meta,
        device,
        batch_size=cmssl_batch_size,
    )

    num_h = len(meta.get("horizons_ms", []))
    # CMSSL diagnostics are computed on CMSSL week-3 out-of-sample test data.
    # This split is also used for downstream RL development; it is not final untouched evaluation.
    cmssl_report = report_cmssl_metrics(
        joined_cmssl_test["y"],
        {"dir_logits": joined_cmssl_test["features"][:, :num_h]},
    )

    rl_week3_full_split = {"weeks": rl_train_split["weeks"], "start": rl_train_split["start"], "end": rl_test_split["end"]}
    joined_rl_full = build_joined_split(out_root, rl_week3_full_split, model, meta, device, batch_size=cmssl_batch_size)
    joined_rl_train = slice_joined_by_split(joined_rl_full, rl_train_split)
    joined_rl_val = slice_joined_by_split(joined_rl_full, rl_val_split)
    joined_rl_test = slice_joined_by_split(joined_rl_full, rl_test_split)
    joined_eval_full = build_joined_split(out_root, eval_full_split, model, meta, device, batch_size=cmssl_batch_size)

    mm_train_batch = build_market_batch(joined_rl_train)
    mm_val_batch = build_market_batch(joined_rl_val)
    mm_test_batch = build_market_batch(joined_rl_test)
    mm_eval_full_batch = build_market_batch(joined_eval_full)
    env_kwargs_common = resolve_market_env_common_kwargs_from_env()
    fast_train_batch = prepare_fast_baseline_batch(
        mm_train_batch,
        meta,
        env_kwargs_common=env_kwargs_common,
        split_name="rl_train",
    )
    fast_val_batch = prepare_fast_baseline_batch(
        mm_val_batch,
        meta,
        env_kwargs_common=env_kwargs_common,
        split_name="rl_val",
    )
    fast_test_batch = prepare_fast_baseline_batch(
        mm_test_batch,
        meta,
        env_kwargs_common=env_kwargs_common,
        split_name="rl_test",
    )
    fast_eval_full_batch = prepare_fast_baseline_batch(
        mm_eval_full_batch,
        meta,
        env_kwargs_common=env_kwargs_common,
        split_name="eval_full",
    )
    baseline_alpha_calibration = _fit_baseline_alpha_calibration(
        fast_train_batch,
        meta.get("horizons_ms", DEFAULT_MM_HORIZONS_MS),
    )
    calibration_summary = {
        "fit_split": baseline_alpha_calibration.diagnostics["fit_split"],
        "horizons_ms": baseline_alpha_calibration.horizons_ms.tolist(),
        "slope_bps_by_unit_score": np.round(
            baseline_alpha_calibration.score_to_bps_slope_by_horizon, 6
        ).tolist(),
        "fit_count_by_horizon": baseline_alpha_calibration.fit_count_by_horizon.astype(int).tolist(),
    }
    print(f"[mm runtime] alpha calibration {json.dumps(calibration_summary, sort_keys=True)}")
    delta_scale = float(os.environ.get("BYBIT_MM_DELTA_SCALE", "10.0"))
    taker_scale = float(os.environ.get("BYBIT_MM_TAKER_SCALE", "1.0"))
    allow_taker = os.environ.get("BYBIT_MM_ALLOW_TAKER", "true").strip().lower() in {"1", "true", "yes", "y"}

    mm_train_env = MarketMakingEnv(
        mm_train_batch,
        allow_taker=allow_taker,
        baseline_alpha_calibration=baseline_alpha_calibration,
        **env_kwargs_common,
    )
    print("[mm obs scaling]", json.dumps(mm_train_env.get_observation_scaling_config(), sort_keys=True))
    mm_val_env = MarketMakingEnv(
        mm_val_batch,
        allow_taker=allow_taker,
        baseline_alpha_calibration=baseline_alpha_calibration,
        **env_kwargs_common,
    )
    mm_test_env = MarketMakingEnv(
        mm_test_batch,
        allow_taker=allow_taker,
        baseline_alpha_calibration=baseline_alpha_calibration,
        **env_kwargs_common,
    )
    mm_final_env = MarketMakingEnv(
        mm_eval_full_batch,
        allow_taker=allow_taker,
        baseline_alpha_calibration=baseline_alpha_calibration,
        **env_kwargs_common,
    )
    default_baseline_quote_cfg = load_baseline_quote_config()
    baseline_val_env = build_baseline_eval_env(mm_val_batch, env_kwargs_common, default_baseline_quote_cfg, baseline_alpha_calibration)
    baseline_test_env = build_baseline_eval_env(mm_test_batch, env_kwargs_common, default_baseline_quote_cfg, baseline_alpha_calibration)
    baseline_final_env = build_baseline_eval_env(mm_eval_full_batch, env_kwargs_common, default_baseline_quote_cfg, baseline_alpha_calibration)
    prefitted_obs_norm_state = prefit_market_obs_norm(mm_train_env)
    mm_train_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_val_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_test_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_final_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    baseline_val_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    baseline_test_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    baseline_final_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_obs = mm_train_env.reset()
    mm_obs_dim = mm_obs.shape[0]
    print(
        "[mm obs norm] "
        f"source=train_prefit count={int(prefitted_obs_norm_state['count'])} "
        f"obs_dim={mm_obs_dim} frozen={mm_train_env.freeze_obs_norm}"
    )

    baseline_eval_split = _resolve_baseline_eval_split()
    baseline_split_name, baseline_eval_env = _select_baseline_env(
        baseline_eval_split,
        baseline_val_env,
        baseline_test_env,
    )

    mm_ppo_config = PPOConfig(
        lr=float(os.environ.get("BYBIT_MM_PPO_LR", "3e-4")),
        update_epochs=int(os.environ.get("BYBIT_MM_PPO_UPDATE_EPOCHS", "4")),
        batch_size=int(os.environ.get("BYBIT_MM_PPO_BATCH_SIZE", "32768")),
        clip_ratio=float(os.environ.get("BYBIT_MM_PPO_CLIP_RATIO", "0.2")),
        gamma=float(os.environ.get("BYBIT_MM_PPO_GAMMA", "0.99")),
        gae_lambda=float(os.environ.get("BYBIT_MM_PPO_GAE_LAMBDA", "0.95")),
        entropy_coef=float(os.environ.get("BYBIT_MM_PPO_ENTROPY_COEF", "0.0")),
        action_mag_coef=_env_float("BYBIT_MM_PPO_ACTION_MAG_COEF", 0.0),
        action_mag_power=_env_float("BYBIT_MM_PPO_ACTION_MAG_POWER", 2.0),
        value_coef=float(os.environ.get("BYBIT_MM_PPO_VALUE_COEF", "0.5")),
        policy_hidden=tuple(int(x) for x in os.environ.get("BYBIT_MM_PPO_POLICY_HIDDEN", "128,128").split(",")),
        value_hidden=tuple(int(x) for x in os.environ.get("BYBIT_MM_PPO_VALUE_HIDDEN", "128,128").split(",")),
        val_every=_env_int("BYBIT_MM_PPO_VAL_EVERY", 1),
        max_drawdown_guard=_env_float("BYBIT_MM_PPO_MAX_DRAWDOWN", float("nan")),
        rollout_horizon=_env_int("BYBIT_MM_PPO_ROLLOUT_HORIZON", 32768),
        rollouts_per_epoch=_env_int("BYBIT_MM_PPO_ROLLOUTS_PER_EPOCH", 16),
        randomize_rollout_start=_env_bool("BYBIT_MM_PPO_RANDOMIZE_START", True),
        zero_residual_init=_env_bool("BYBIT_MM_PPO_ZERO_RESIDUAL_INIT", True),
        init_log_std=_env_float("BYBIT_MM_PPO_INIT_LOG_STD", -1.5),
    )
    if np.isnan(mm_ppo_config.max_drawdown_guard):
        mm_ppo_config.max_drawdown_guard = None
    mm_best_ckpt = Path(os.environ.get("BYBIT_MM_PPO_BEST_CKPT", Path(out_root) / "mm_ppo_best.pt"))
    require_rl_ckpt = _env_bool("BYBIT_MM_REQUIRE_RL_CKPT", False)
    external_rl_ckpt = os.environ.get("BYBIT_MM_RL_CKPT", "")
    external_ckpt_explicit = bool(external_rl_ckpt.strip())

    trained_this_run = False
    if run_mode == "train":
        resolved_eval_ckpt = None
        rl_checkpoint_origin = "none"
        eval_ckpt_payload = None
        use_external_eval_ckpt = False
        rl_eval_performed = False
    else:
        eval_ckpt_resolution = _resolve_eval_checkpoint(
            run_mode=run_mode,
            mm_best_ckpt=mm_best_ckpt,
            external_rl_ckpt_raw=external_rl_ckpt,
            require_rl_ckpt=require_rl_ckpt,
        )
        resolved_eval_ckpt = eval_ckpt_resolution.resolved_eval_ckpt
        rl_checkpoint_origin = eval_ckpt_resolution.checkpoint_origin
        external_ckpt_explicit = eval_ckpt_resolution.external_ckpt_explicit
        eval_ckpt_payload = eval_ckpt_resolution.checkpoint_payload
        use_external_eval_ckpt = rl_checkpoint_origin == "external"

    rl_policy_loaded = False
    rl_policy_reason = "not evaluated"
    rl_policy_eval_mode = "deterministic_mean"
    obs_norm_source = "env_default"
    saved_new_ckpt_this_run = False

    if run_mode in {"train", "train_eval"}:
        print(f"[mm train] starting PPO training (run_mode={run_mode})")
        _trained_model, best_val_report, saved_new_ckpt_this_run = train_market_ppo(
            mm_train_env,
            mm_val_env,
            mm_obs_dim,
            device=device,
            epochs=_resolve_ppo_epochs(ppo_epochs),
            config=mm_ppo_config,
            ckpt_path=mm_best_ckpt,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
            rollout_storage=rollout_storage,
            pin_rollout_memory=pin_rollout_memory,
            non_blocking_transfers=non_blocking_transfers,
        )
        trained_this_run = True
        train_obs_norm_state = mm_train_env.get_obs_norm_state()
        obs_norm_source = "train_prefit"
        if best_val_report:
            best_val_mode = str(best_val_report.get("best_report_mode", "unknown"))
            print(_format_mm_summary(f"[mm train] best_val mode={best_val_mode}", best_val_report))
        else:
            print("[mm train] no validation checkpoint satisfied selection filters.")
            if (
                run_mode == "train_eval"
                and not external_ckpt_explicit
                and not saved_new_ckpt_this_run
            ):
                resolved_eval_ckpt = None
                rl_checkpoint_origin = "none"
                print(
                    "[mm eval] no new training checkpoint saved this run; "
                    "ignoring stale default best-checkpoint path."
                )
        print(
            "[mm train] completed PPO training; best checkpoint path="
            f"{mm_best_ckpt.expanduser().resolve()}"
        )
    else:
        train_obs_norm_state = None

    eval_obs_norm_state = train_obs_norm_state
    if use_external_eval_ckpt:
        if not isinstance(eval_ckpt_payload, dict) or "obs_norm_state" not in eval_ckpt_payload:
            raise RuntimeError(
                "External RL checkpoint evaluation requires obs_norm_state; the checkpoint is incomplete or malformed. "
                "Re-save the checkpoint from the new codepath or provide one with valid prefitted frozen observation normalization."
            )
        eval_obs_norm_state = eval_ckpt_payload["obs_norm_state"]
        if not _obs_norm_state_is_ready(eval_obs_norm_state):
            raise RuntimeError(
                "External RL checkpoint contains an unusable obs_norm_state (count < 2 or missing mean/m2). "
                "Evaluation requires a valid prefitted frozen observation normalization state."
            )
        mm_test_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
        mm_final_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
        baseline_val_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
        baseline_test_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
        baseline_final_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
        obs_norm_source = "checkpoint"

    if eval_obs_norm_state is not None:
        baseline_eval_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
        baseline_final_env.set_obs_norm_state(eval_obs_norm_state, freeze=True)
    baseline_dev_t0 = time.perf_counter()
    baseline_dev_metrics = evaluate_prepared_baseline_fast(fast_val_batch if baseline_eval_split == "val" else fast_test_batch, default_baseline_quote_cfg, baseline_alpha_calibration)
    # Final untouched evaluation remains week-4 (eval.full) for both mm_baseline and mm_rl.
    baseline_metrics = evaluate_prepared_baseline_fast(fast_eval_full_batch, default_baseline_quote_cfg, baseline_alpha_calibration)
    _timing_log(
        "evaluate_market_making baseline "
        f"split=eval.full secs={time.perf_counter() - baseline_dev_t0:.4f}"
    )

    rl_metrics = None
    rl_dev_metrics = None
    ppo_eval_stochastic = _env_bool("BYBIT_MM_PPO_EVAL_STOCHASTIC", False)
    ppo_eval_seed = _env_int("BYBIT_MM_PPO_EVAL_SEED", 0)

    eval_action = "skipped" if run_mode == "train" else "performed"
    print(
        "[mm eval] "
        f"mode={run_mode} "
        f"checkpoint_origin={rl_checkpoint_origin} "
        f"resolved_path={resolved_eval_ckpt if resolved_eval_ckpt is not None else 'none'} "
        f"eval_action={eval_action}"
    )

    if run_mode == "train":
        rl_policy_loaded = False
        rl_policy_reason = "skipped because BYBIT_MM_RUN_MODE=train"
        print("[mm eval] baseline only; RL skipped because run_mode=train.")
    else:
        mm_ppo_model: Optional[MarketPolicyValueNet] = None
        mm_policy: Optional[MarketPolicyNet] = None
        if run_mode == "eval":
            mm_ppo_model = load_market_ppo_model(
                mm_obs_dim,
                device=device,
                ckpt_path=resolved_eval_ckpt,
                require_checkpoint=True,
                checkpoint_data=eval_ckpt_payload,
            )
            require(mm_ppo_model is not None, "Failed to load eval PPO checkpoint")
            rl_policy_reason = "loaded"
        elif resolved_eval_ckpt is None:
            rl_policy_reason = "no path provided"
        elif not Path(resolved_eval_ckpt).exists():
            missing_msg = f"[mm eval] no checkpoint saved/found at {resolved_eval_ckpt}; using baseline deltas for RL run."
            if require_rl_ckpt:
                raise FileNotFoundError(missing_msg)
            warnings.warn(missing_msg, RuntimeWarning)
            rl_policy_reason = "missing checkpoint"
        else:
            policy_require_checkpoint = require_rl_ckpt or (
                run_mode == "train_eval" and external_ckpt_explicit
            )
            mm_ppo_model = load_market_ppo_model(
                mm_obs_dim,
                device=device,
                ckpt_path=resolved_eval_ckpt,
                require_checkpoint=policy_require_checkpoint,
                checkpoint_data=eval_ckpt_payload,
            )
            if mm_ppo_model is None:
                mm_policy = load_market_policy(
                    mm_obs_dim,
                    device=device,
                    ckpt_path=resolved_eval_ckpt,
                    require_checkpoint=policy_require_checkpoint,
                    checkpoint_data=eval_ckpt_payload,
                )
            rl_policy_reason = "loaded" if (mm_ppo_model is not None or mm_policy is not None) else "missing checkpoint"

        if mm_ppo_model is not None:
            rl_policy_loaded = True
            rl_policy_eval_mode = "stochastic_sample" if ppo_eval_stochastic else "deterministic_mean"
            stochastic_generator = None
            if ppo_eval_stochastic:
                stochastic_generator = torch.Generator(device=torch.device(device).type)
                stochastic_generator.manual_seed(ppo_eval_seed)
            rl_eval_t0 = time.perf_counter()
            rl_dev_metrics = evaluate_market_policy_ppo(
                mm_test_env,
                mm_ppo_model,
                stochastic=ppo_eval_stochastic,
                device=device,
                delta_scale=delta_scale,
                taker_scale=taker_scale,
                generator=stochastic_generator,
            )
            _timing_log(f"evaluate_market_making rl secs={time.perf_counter() - rl_eval_t0:.4f}")
            rl_eval_performed = True
            rl_metrics = evaluate_market_policy_ppo(
                mm_final_env,
                mm_ppo_model,
                stochastic=ppo_eval_stochastic,
                device=device,
                delta_scale=delta_scale,
                taker_scale=taker_scale,
                generator=stochastic_generator,
            )
        else:
            if mm_policy is None:
                if rl_policy_reason == "no path provided":
                    print("[mm eval] no policy path provided; using baseline deltas for RL run.")
                rl_policy_fn = lambda _obs: (0.0, 0.0, 0.0)
                rl_policy_loaded = False
            else:
                rl_policy_loaded = True
                rl_policy_eval_mode = "deterministic_mean"
                print("[mm eval] deterministic policy action semantics=bounded_harmonized source=code")

                def rl_policy_fn(obs: np.ndarray) -> Tuple[float, float, float]:
                    return _policy_action_from_obs_numpy(
                        obs,
                        mm_policy,
                        device,
                        delta_scale,
                        taker_scale,
                        env=mm_test_env,
                    )

            rl_eval_t0 = time.perf_counter()
            rl_dev_metrics = evaluate_market_making(mm_test_env, rl_policy_fn)
            rl_metrics = evaluate_market_making(mm_final_env, rl_policy_fn)
            _timing_log(f"evaluate_market_making rl secs={time.perf_counter() - rl_eval_t0:.4f}")
            rl_eval_performed = True

    # Return shape:
    # - cmssl_test: CMSSL week-3 out-of-sample/downstream-development diagnostics.
    # - mm_baseline/mm_rl: final untouched week-4 (eval.full) metrics.
    # - *_dev_* fields: week-3 development splits.
    return {
        "cmssl_test": cmssl_report,
        "mm_obs_scaling": mm_train_env.get_observation_scaling_config(),
        "mm_baseline": baseline_metrics,
        "mm_rl": rl_metrics,
        "mm_baseline_dev_test": baseline_dev_metrics if baseline_eval_split == "test" else None,
        "mm_baseline_dev_val": baseline_dev_metrics if baseline_eval_split == "val" else None,
        "mm_rl_dev_test": rl_dev_metrics,
        # Fatal failures are surfaced via exceptions rather than persisted in provenance state.
        "mm_run_context": {
            "run_mode": run_mode,
            "baseline_eval_split": baseline_split_name,
            "baseline_eval_semantics": f"rl_week3_{baseline_split_name}",
            "ppo_trained_this_run": trained_this_run,
            "ppo_best_ckpt_path": str(mm_best_ckpt.expanduser().resolve()),
            "rl_eval_performed": rl_eval_performed,
            "rl_checkpoint_origin": rl_checkpoint_origin,
            # Canonical evaluated policy checkpoint path consumed by downstream readers.
            "rl_checkpoint_path": resolved_eval_ckpt,
            # Preserve user provenance (raw request) without duplicating resolved path state.
            "external_rl_ckpt_requested": external_rl_ckpt.strip() or None,
            "external_rl_ckpt_explicit": external_ckpt_explicit,
            "obs_norm_source": obs_norm_source,
            "require_rl_ckpt": require_rl_ckpt,
        },
        "mm_rl_policy_loaded": {
            "loaded": rl_policy_loaded,
            "reason": rl_policy_reason,
            "path": resolved_eval_ckpt,
            "require_checkpoint": require_rl_ckpt,
            "eval_mode": rl_policy_eval_mode,
            "ppo_eval_stochastic": ppo_eval_stochastic if run_mode != "train" else False,
            "ppo_eval_seed": ppo_eval_seed if run_mode != "train" else None,
        },
    }


if __name__ == "__main__":
    out_root = os.environ.get("BYBIT_OUT_ROOT", "").strip()
    ckpt_path = os.environ.get("BYBIT_CMSSL_CKPT", "").strip()
    device = os.environ.get("BYBIT_DEVICE", "cuda")
    ppo_epochs = _resolve_ppo_epochs(10)
    run_mode = _resolve_run_mode("train")
    run_cmssl_test_window = os.environ.get("BYBIT_RUN_CMSSL_TEST_WINDOW", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    verbose_reports = _env_bool("BYBIT_MM_VERBOSE_REPORTS", False)

    if not out_root or not ckpt_path:
        raise SystemExit("Set BYBIT_OUT_ROOT and BYBIT_CMSSL_CKPT before running.")

    _set_seed_from_env()
    tf32_enabled = _configure_tf32_from_env()

    print(
        "[rl exec config]",
        json.dumps(
            {
                "out_root": out_root,
                "ckpt_path": ckpt_path,
                "device": device,
                "ppo_epochs": ppo_epochs,
                "ppo_epochs_env": PPO_EPOCHS_ENV,
                "run_mode": run_mode,
                "tf32_enabled": tf32_enabled,
            },
            sort_keys=True,
        ),
    )
    report = run_pipeline(
        out_root,
        ckpt_path,
        device=device,
        ppo_epochs=ppo_epochs,
        run_mode=run_mode,
    )
    print("[cmssl test]", report["cmssl_test"])
    print("[mm obs scaling]", report["mm_obs_scaling"])
    # Ownership: __main__ prints MM summaries once; run_pipeline() only returns metrics.
    # Keep default logs compact so routine runs stay readable; full reports are opt-in.
    print("[mm eval]", _format_mm_summary("baseline", report["mm_baseline"]))
    if report["mm_rl"] is None:
        print("[mm rl] skipped (mm_rl is None)")
    else:
        print("[mm eval]", _format_mm_summary("baseline+rl", report["mm_rl"]))
    if verbose_reports:
        print("[mm baseline verbose]", _summarize_for_log(report["mm_baseline"]))
        if report["mm_rl"] is None:
            print("[mm rl verbose] skipped (mm_rl is None)")
        else:
            print("[mm rl verbose]", _summarize_for_log(report["mm_rl"]))
    if run_cmssl_test_window:
        print("[cmssl test window] running windowed inference for diagnostics.")
        test_window_report = run_cmssl_test_window_inference(out_root, ckpt_path, device=device)
        print("[cmssl test window] completed", json.dumps({"horizons_ms": test_window_report["horizons_ms"]}))
