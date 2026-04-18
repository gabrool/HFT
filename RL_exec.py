import json
import os
import hashlib
import time
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from offline_tokens import iter_week_chunks, load_global_meta, read_json

# Reference snapshot cadence in milliseconds (used for runtime scaling only).
RAW_SNAPSHOT_EXPECTED_STEP_MS = 100
DECISION_SNAPSHOTS_SCHEMA_VERSION = 1
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
FEATURE_EXTRA_DIM = 7
ENV_OBS_EXTRA_STATE_DIM = 14
DEFAULT_MM_OBS_SPREAD_ANCHOR_FRAC = 0.5
DEFAULT_MM_QUOTE_HALF_SPREAD_FLOOR_BPS = 0.0050
DEFAULT_MM_SPREAD_CAP_BPS = 10_000.0
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
DEFAULT_MM_TAKER_THRESHOLD = 0.1
DEFAULT_MM_TAKER_SIGNAL_LIMIT = 1.0
MM_PPO_CHECKPOINT_SCHEMA = "mm-ppo-direct-quote-v7"
MM_PPO_ACTION_DIM = 4
MM_PPO_ACTION_SEMANTICS = (
    "center_control",
    "width_control",
    "skew_control",
    "taker_signal",
)
# Inventory risk thresholds are denominated in quote notional (USD).
JOINED_CACHE_SCHEMA_VERSION = 1
JOINED_FEATURE_SCHEMA_VERSION = 2

class JoinedCacheError(RuntimeError):
    """Raised when joined cache artifacts are stale, corrupt, or invalid."""


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
    dataset_trade_history_enabled = meta.get("trade_history_enabled")
    dataset_has_event_stream_mode = "event_stream_mode" in meta
    dataset_event_stream_mode = meta.get("event_stream_mode") if dataset_has_event_stream_mode else None
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
        require(
            isinstance(ckpt_args, dict),
            "CMSSL checkpoint is missing args metadata; retrain/re-export with updated CMSSL17_offline.py so compatibility fields are embedded.",
            exc_type=RuntimeError,
        )
        require(
            "trade_history_enabled" in ckpt_args,
            "CMSSL checkpoint args missing 'trade_history_enabled'; retrain/re-export with updated CMSSL17_offline.py.",
            exc_type=RuntimeError,
        )
        ckpt_trade_history_enabled = ckpt_args.get("trade_history_enabled")
        require(
            ckpt_trade_history_enabled == dataset_trade_history_enabled,
            "CMSSL dataset/checkpoint compatibility mismatch for trade_history_enabled: "
            f"dataset={dataset_trade_history_enabled!r}, checkpoint={ckpt_trade_history_enabled!r}. "
            "Use a checkpoint trained on this dataset mode or retrain/re-export with updated CMSSL17_offline.py.",
        )
        if dataset_has_event_stream_mode:
            require(
                "event_stream_mode" in ckpt_args,
                "CMSSL checkpoint args missing 'event_stream_mode' for an event-stream-mode dataset; "
                "retrain/re-export with updated CMSSL17_offline.py.",
                exc_type=RuntimeError,
            )
            ckpt_event_stream_mode = ckpt_args.get("event_stream_mode")
            require(
                ckpt_event_stream_mode == dataset_event_stream_mode,
                "CMSSL dataset/checkpoint compatibility mismatch for event_stream_mode: "
                f"dataset={dataset_event_stream_mode!r}, checkpoint={ckpt_event_stream_mode!r}. "
                "Use a checkpoint trained on this dataset mode or retrain/re-export with updated CMSSL17_offline.py.",
            )
        elif isinstance(ckpt_args, dict) and "event_stream_mode" in ckpt_args:
            ckpt_event_stream_mode = ckpt_args.get("event_stream_mode")
            if ckpt_event_stream_mode is not None:
                require(
                    False,
                    "CMSSL checkpoint carries event_stream_mode metadata but dataset metadata does not; "
                    "regenerate/re-export artifacts with updated CMSSL17_offline.py so modes are aligned.",
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


def _fail_on_removed_env_vars(removed: Sequence[str]) -> None:
    present = [name for name in removed if os.environ.get(name, "").strip()]
    if present:
        raise ValueError(
            "Removed env vars are set and no longer supported: "
            f"{', '.join(sorted(present))}"
        )


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


def _resolve_market_action_dim(_allow_taker: Optional[bool] = None) -> int:
    return MM_PPO_ACTION_DIM


def _ppo_action_bounds(
    env: Optional["MarketMakingEnv"],
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    action_dim = _resolve_market_action_dim()
    direct_cfg = env.direct_quote_config if env is not None else load_direct_quote_config()
    taker_signal_limit = float(direct_cfg.taker_signal_limit)
    low = torch.tensor(
        [-1.0, 0.0, -1.0, -taker_signal_limit],
        device=device,
        dtype=torch.float32,
    )
    high = torch.tensor(
        [1.0, 1.0, 1.0, taker_signal_limit],
        device=device,
        dtype=torch.float32,
    )
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
    """Resolve run mode: train, eval, or train_eval."""
    accepted_modes = {"train", "eval", "train_eval"}
    mode = os.environ.get("BYBIT_MM_RUN_MODE", default).strip().lower()
    if mode not in accepted_modes:
        accepted = ", ".join(sorted(accepted_modes))
        raise ValueError(f"Invalid BYBIT_MM_RUN_MODE='{mode}'. Accepted values: {accepted}")
    return mode






def resolve_market_env_common_kwargs_from_env() -> Dict[str, Any]:
    _fail_on_removed_env_vars(("BYBIT_MM_SPREAD_FLOOR_BPS", "BYBIT_MM_SKEW_MAX_FRAC"))
    direct_quote_config = load_direct_quote_config()
    continuous_maker_fill_config = load_continuous_maker_fill_config()
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
        "direct_quote_config": direct_quote_config,
        "continuous_maker_fill_config": continuous_maker_fill_config,
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
        # missing path is fatal.
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
class DirectQuoteConfig:
    quote_half_spread_floor_bps: float
    spread_cap_bps: float
    obs_spread_anchor_frac: float
    touch_halfspread_mult: float
    wide_halfspread_mult: float
    taker_signal_limit: float
    inventory_center_weight: float
    alpha_center_weight: float
    asymmetry_residual_frac: float
    directional_response_center_weight: float
    directional_response_asym_weight: float


@dataclass(frozen=True)
class DirectionalSignalConfig:
    horizon_logit_weights: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    training_reward_horizon_ms: int = 1000


@dataclass(frozen=True)
class RolloutStartSamplingConfig:
    enabled: bool = False
    weighted_mix: float = 0.8
    score_power: float = 1.0
    score_epsilon: float = 1e-6
    lead_steps: int = 512
    start_exclusion_window: Optional[int] = None


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = True
    logit_tanh_scale: float = 12.0
    skew_coef: float = 0.01


@dataclass(frozen=True)
class ContinuousMakerFillConfig:
    activity_min: float = 0.03
    activity_max: float = 0.20
    tau_touch: float = 0.12
    tau_cross: float = 0.09
    touch_event_boost: float = 0.15
    touch_event_distance_frac: float = 0.10
    price_epsilon_px: float = 1e-9


@dataclass(frozen=True)
class ContinuousMakerFillCalibration:
    vol_p50_bps: float
    vol_p90_bps: float
    vol_mean_bps: float
    vol_p99_bps: float
    sample_count: int


def _fit_continuous_maker_fill_calibration_from_snapshots(
    train_snapshots: np.ndarray,
) -> ContinuousMakerFillCalibration:
    vol_short_idx = RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_short")
    vol_long_idx = RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_long")
    snapshots = np.asarray(train_snapshots, dtype=np.float64)
    if snapshots.ndim != 2:
        raise ValueError(f"train_snapshots must be rank-2, got shape={snapshots.shape}")
    if snapshots.shape[1] <= max(vol_short_idx, vol_long_idx):
        raise ValueError(
            "train_snapshots does not include required volatility columns: "
            f"shape={snapshots.shape} vol_short_idx={vol_short_idx} vol_long_idx={vol_long_idx}"
        )
    vol_short = snapshots[:, vol_short_idx]
    vol_long = snapshots[:, vol_long_idx]
    sigma_recent = np.maximum(vol_short, vol_long).astype(np.float64, copy=False)
    sigma_bps = 1e4 * sigma_recent
    valid_mask = np.isfinite(sigma_bps) & (sigma_bps >= 0.0)
    sigma_bps_valid = sigma_bps[valid_mask]
    if sigma_bps_valid.size < 1024:
        raise ValueError(
            "Insufficient valid train-split volatility rows for maker fill calibration: "
            f"sample_count={sigma_bps_valid.size} required_min=1024"
        )
    p50 = float(np.percentile(sigma_bps_valid, 50.0))
    p90 = float(np.percentile(sigma_bps_valid, 90.0))
    p99 = float(np.percentile(sigma_bps_valid, 99.0))
    mean = float(np.mean(sigma_bps_valid))
    if not np.isfinite(p50) or not np.isfinite(p90):
        raise ValueError(
            "Invalid maker fill calibration quantiles: "
            f"vol_p50_bps={p50} vol_p90_bps={p90}"
        )
    if p90 <= p50:
        raise ValueError(
            "Maker fill calibration requires vol_p90_bps > vol_p50_bps: "
            f"vol_p50_bps={p50} vol_p90_bps={p90}"
        )
    return ContinuousMakerFillCalibration(
        vol_p50_bps=p50,
        vol_p90_bps=p90,
        vol_mean_bps=mean,
        vol_p99_bps=p99,
        sample_count=int(sigma_bps_valid.size),
    )


def load_continuous_maker_fill_config() -> ContinuousMakerFillConfig:
    cfg = ContinuousMakerFillConfig(
        activity_min=_env_float("BYBIT_MM_FILL_ACTIVITY_MIN", 0.03),
        activity_max=_env_float("BYBIT_MM_FILL_ACTIVITY_MAX", 0.20),
        tau_touch=_env_float("BYBIT_MM_FILL_TAU_TOUCH", 0.12),
        tau_cross=_env_float("BYBIT_MM_FILL_TAU_CROSS", 0.09),
        touch_event_boost=_env_float("BYBIT_MM_FILL_TOUCH_EVENT_BOOST", 0.15),
        touch_event_distance_frac=_env_float("BYBIT_MM_FILL_TOUCH_EVENT_DISTANCE_FRAC", 0.10),
        price_epsilon_px=_env_float("BYBIT_MM_FILL_PRICE_EPSILON_PX", 1e-9),
    )
    if not np.isfinite(cfg.activity_min) or not np.isfinite(cfg.activity_max):
        raise ValueError("Continuous maker-fill activity bounds must be finite.")
    if cfg.activity_min < 0.0 or cfg.activity_max < cfg.activity_min:
        raise ValueError("Continuous maker-fill activity bounds must satisfy 0 <= min <= max.")
    if not np.isfinite(cfg.tau_touch) or cfg.tau_touch <= 0.0:
        raise ValueError("BYBIT_MM_FILL_TAU_TOUCH must be finite and > 0.")
    if not np.isfinite(cfg.tau_cross) or cfg.tau_cross <= 0.0:
        raise ValueError("BYBIT_MM_FILL_TAU_CROSS must be finite and > 0.")
    if not np.isfinite(cfg.touch_event_boost) or cfg.touch_event_boost < 0.0 or cfg.touch_event_boost > 1.0:
        raise ValueError("BYBIT_MM_FILL_TOUCH_EVENT_BOOST must be finite and in [0, 1].")
    if (
        not np.isfinite(cfg.touch_event_distance_frac)
        or cfg.touch_event_distance_frac < 0.0
        or cfg.touch_event_distance_frac > 1.0
    ):
        raise ValueError("BYBIT_MM_FILL_TOUCH_EVENT_DISTANCE_FRAC must be finite and in [0, 1].")
    if not np.isfinite(cfg.price_epsilon_px) or cfg.price_epsilon_px <= 0.0:
        raise ValueError("BYBIT_MM_FILL_PRICE_EPSILON_PX must be finite and > 0.")
    return cfg


def _env_float_tuple(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(float(v) for v in default)
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one float value.")
    return values


def _resolve_fixed_horizon_logit_weights(name: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    values = _env_float_tuple(name, default)
    if len(values) != 3:
        raise ValueError(f"{name} must provide exactly 3 comma-separated floats for horizons [250,500,1000].")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_directional_signal_config(meta: Dict[str, Any]) -> DirectionalSignalConfig:
    horizons = [int(h) for h in meta.get("horizons_ms", [])]
    _validate_fixed_cmssl_horizons(horizons)
    weights = _resolve_fixed_horizon_logit_weights(
        "BYBIT_MM_SIGNAL_HORIZON_LOGIT_WEIGHTS",
        (0.0, 0.0, 1.0),
    )
    training_reward_horizon_ms = _env_int("BYBIT_MM_TRAIN_REWARD_HORIZON_MS", 1000)
    if training_reward_horizon_ms not in horizons:
        raise ValueError(
            "BYBIT_MM_TRAIN_REWARD_HORIZON_MS must be one of meta['horizons_ms']: "
            f"got={training_reward_horizon_ms} horizons={horizons}"
        )
    return DirectionalSignalConfig(
        horizon_logit_weights=weights,
        training_reward_horizon_ms=int(training_reward_horizon_ms),
    )


def load_rollout_start_sampling_config(*, rollout_horizon: int) -> RolloutStartSamplingConfig:
    safe_rollout_horizon = max(1, int(rollout_horizon))
    raw_start_exclusion_window = os.environ.get("BYBIT_MM_START_SAMPLING_EXCLUSION_WINDOW", "").strip()
    if raw_start_exclusion_window:
        resolved_start_exclusion_window = _env_int(
            "BYBIT_MM_START_SAMPLING_EXCLUSION_WINDOW",
            safe_rollout_horizon,
        )
    else:
        resolved_start_exclusion_window = safe_rollout_horizon
    cfg = RolloutStartSamplingConfig(
        enabled=_env_bool("BYBIT_MM_START_SAMPLING_ENABLE", False),
        weighted_mix=_env_float("BYBIT_MM_START_SAMPLING_WEIGHTED_MIX", 0.8),
        score_power=_env_float("BYBIT_MM_START_SAMPLING_SCORE_POWER", 1.0),
        score_epsilon=_env_float("BYBIT_MM_START_SAMPLING_SCORE_EPS", 1e-6),
        lead_steps=_env_int("BYBIT_MM_START_SAMPLING_LEAD_STEPS", 512),
        start_exclusion_window=resolved_start_exclusion_window,
    )
    if not np.isfinite(cfg.weighted_mix) or cfg.weighted_mix < 0.0 or cfg.weighted_mix > 1.0:
        raise ValueError("BYBIT_MM_START_SAMPLING_WEIGHTED_MIX must be in [0, 1].")
    if not np.isfinite(cfg.score_power) or cfg.score_power <= 0.0:
        raise ValueError("BYBIT_MM_START_SAMPLING_SCORE_POWER must be finite and > 0.")
    if not np.isfinite(cfg.score_epsilon) or cfg.score_epsilon <= 0.0:
        raise ValueError("BYBIT_MM_START_SAMPLING_SCORE_EPS must be finite and > 0.")
    if cfg.lead_steps < 0:
        raise ValueError("BYBIT_MM_START_SAMPLING_LEAD_STEPS must be >= 0.")
    resolved_start_exclusion_window_int = int(cfg.start_exclusion_window)
    if resolved_start_exclusion_window_int < 0:
        raise ValueError("BYBIT_MM_START_SAMPLING_EXCLUSION_WINDOW must be >= 0.")
    cfg = RolloutStartSamplingConfig(
        enabled=cfg.enabled,
        weighted_mix=cfg.weighted_mix,
        score_power=cfg.score_power,
        score_epsilon=cfg.score_epsilon,
        lead_steps=cfg.lead_steps,
        start_exclusion_window=resolved_start_exclusion_window_int,
    )
    return cfg


def load_reward_shaping_config() -> RewardShapingConfig:
    removed_width_envs = (
        "BYBIT_MM_REWARD_SHAPING_WIDTH_COEF",
        "BYBIT_MM_REWARD_SHAPING_WIDTH_TARGET_BASE",
        "BYBIT_MM_REWARD_SHAPING_WIDTH_TARGET_ALPHA_WEIGHT",
        "BYBIT_MM_REWARD_SHAPING_WIDTH_TARGET_VOL_WEIGHT",
        "BYBIT_MM_REWARD_SHAPING_WIDTH_TARGET_MAX",
    )
    set_removed = [name for name in removed_width_envs if os.environ.get(name, "").strip()]
    if set_removed:
        raise ValueError(
            "Width shaping was removed from reward shaping; the following env vars are no longer supported: "
            + ", ".join(set_removed)
        )
    cfg = RewardShapingConfig(
        enabled=_env_bool("BYBIT_MM_REWARD_SHAPING_ENABLE", True),
        logit_tanh_scale=_env_float("BYBIT_MM_REWARD_SHAPING_LOGIT_TANH_SCALE", 12.0),
        skew_coef=_env_float("BYBIT_MM_REWARD_SHAPING_SKEW_COEF", 0.01),
    )
    if not np.isfinite(cfg.logit_tanh_scale) or cfg.logit_tanh_scale <= 0.0:
        raise ValueError("BYBIT_MM_REWARD_SHAPING_LOGIT_TANH_SCALE must be finite and > 0.")
    if not np.isfinite(cfg.skew_coef) or cfg.skew_coef < 0.0:
        raise ValueError("BYBIT_MM_REWARD_SHAPING_SKEW_COEF must be finite and >= 0.")
    return cfg


def load_direct_quote_config() -> DirectQuoteConfig:
    return _validate_direct_quote_config(
        DirectQuoteConfig(
            quote_half_spread_floor_bps=_env_float(
                "BYBIT_MM_QUOTE_HALF_SPREAD_FLOOR_BPS",
                DEFAULT_MM_QUOTE_HALF_SPREAD_FLOOR_BPS,
            ),
            spread_cap_bps=_env_float("BYBIT_MM_SPREAD_CAP_BPS", DEFAULT_MM_SPREAD_CAP_BPS),
            obs_spread_anchor_frac=_env_float("BYBIT_MM_OBS_SPREAD_ANCHOR_FRAC", DEFAULT_MM_OBS_SPREAD_ANCHOR_FRAC),
            touch_halfspread_mult=_env_float("BYBIT_MM_TOUCH_HALFSPREAD_MULT", 1.0),
            wide_halfspread_mult=_env_float("BYBIT_MM_WIDE_HALFSPREAD_MULT", 1.75),
            taker_signal_limit=_env_float("BYBIT_MM_TAKER_SIGNAL_LIMIT", DEFAULT_MM_TAKER_SIGNAL_LIMIT),
            inventory_center_weight=_env_float("BYBIT_MM_INVENTORY_CENTER_WEIGHT", 0.25),
            alpha_center_weight=_env_float("BYBIT_MM_ALPHA_CENTER_WEIGHT", 1.00),
            asymmetry_residual_frac=_env_float("BYBIT_MM_ASYMMETRY_RESIDUAL_FRAC", 0.15),
            directional_response_center_weight=_env_float("BYBIT_MM_DIRECTIONAL_RESPONSE_CENTER_WEIGHT", 0.90),
            directional_response_asym_weight=_env_float("BYBIT_MM_DIRECTIONAL_RESPONSE_ASYM_WEIGHT", 0.10),
        )
    )


def _validate_direct_quote_config(cfg: DirectQuoteConfig) -> DirectQuoteConfig:
    if not np.isfinite(cfg.quote_half_spread_floor_bps) or cfg.quote_half_spread_floor_bps <= 0.0:
        raise ValueError("quote_half_spread_floor_bps must be finite and > 0.0")
    if not np.isfinite(cfg.spread_cap_bps) or cfg.spread_cap_bps < cfg.quote_half_spread_floor_bps:
        raise ValueError("spread_cap_bps must be finite and >= quote_half_spread_floor_bps")
    if (
        not np.isfinite(cfg.obs_spread_anchor_frac)
        or cfg.obs_spread_anchor_frac < 0.0
        or cfg.obs_spread_anchor_frac > 1.0
    ):
        raise ValueError("obs_spread_anchor_frac must be finite and in [0.0, 1.0]")
    if not np.isfinite(cfg.touch_halfspread_mult) or cfg.touch_halfspread_mult <= 0.0:
        raise ValueError("touch_halfspread_mult must be finite and > 0.")
    if not np.isfinite(cfg.wide_halfspread_mult) or cfg.wide_halfspread_mult < cfg.touch_halfspread_mult:
        raise ValueError("wide_halfspread_mult must be finite and >= touch_halfspread_mult.")
    if not np.isfinite(cfg.taker_signal_limit) or cfg.taker_signal_limit <= 0.0:
        raise ValueError("taker_signal_limit must be finite and > 0.0")
    if not np.isfinite(cfg.inventory_center_weight) or cfg.inventory_center_weight < 0.0 or cfg.inventory_center_weight > 1.0:
        raise ValueError("inventory_center_weight must be finite and in [0.0, 1.0].")
    if not np.isfinite(cfg.alpha_center_weight) or cfg.alpha_center_weight <= 0.0 or cfg.alpha_center_weight > 1.0:
        raise ValueError("alpha_center_weight must be finite and in (0.0, 1.0].")
    if (
        not np.isfinite(cfg.asymmetry_residual_frac)
        or cfg.asymmetry_residual_frac < 0.0
        or cfg.asymmetry_residual_frac > 0.50
    ):
        raise ValueError("asymmetry_residual_frac must be finite and in [0.0, 0.50].")
    if not np.isfinite(cfg.directional_response_center_weight) or cfg.directional_response_center_weight < 0.0:
        raise ValueError("directional_response_center_weight must be finite and >= 0.")
    if not np.isfinite(cfg.directional_response_asym_weight) or cfg.directional_response_asym_weight < 0.0:
        raise ValueError("directional_response_asym_weight must be finite and >= 0.")
    directional_weight_sum = cfg.directional_response_center_weight + cfg.directional_response_asym_weight
    if abs(directional_weight_sum - 1.0) > 1e-8:
        raise ValueError("directional response weights must sum to 1.0 within tolerance 1e-8.")
    if cfg.directional_response_center_weight <= cfg.directional_response_asym_weight:
        raise ValueError("directional_response_center_weight must be strictly greater than directional_response_asym_weight.")
    return cfg


def resolve_vol_bucket_edges_bps(
    *,
    env_var_name: str = "BYBIT_MM_VOL_BUCKET_BPS",
    default_value: str = "0.5,1.0,2.0,4.0,8.0",
) -> List[float]:
    raw_value = os.environ.get(env_var_name, default_value)
    edges = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not edges:
        raise ValueError(f"{env_var_name} must contain at least one positive edge")
    prev = 0.0
    for edge in edges:
        if not np.isfinite(edge) or edge <= 0.0:
            raise ValueError(f"{env_var_name} edges must be finite and > 0")
        if edge <= prev:
            raise ValueError(f"{env_var_name} edges must be strictly increasing")
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
    maker_buy_fill_frac_per_step: Optional[np.ndarray] = None,
    maker_sell_fill_frac_per_step: Optional[np.ndarray] = None,
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
        bucket_edges = resolve_vol_bucket_edges_bps()
    else:
        bucket_edges = [float(edge) for edge in bucket_edges_bps]
    total_steps = int(sigma_arr.size)
    safe_sigma = np.where(np.isfinite(sigma_arr), sigma_arr, np.inf)
    bucket_indices = np.searchsorted(np.asarray(bucket_edges, dtype=np.float64), safe_sigma, side="right")
    report_rows: List[Dict[str, Any]] = []
    maker_buy_markout_arr = None if maker_buy_markout_per_step is None else np.asarray(maker_buy_markout_per_step, dtype=np.float64)
    maker_sell_markout_arr = None if maker_sell_markout_per_step is None else np.asarray(maker_sell_markout_per_step, dtype=np.float64)
    maker_buy_fill_frac_arr = None if maker_buy_fill_frac_per_step is None else np.asarray(maker_buy_fill_frac_per_step, dtype=np.float64)
    maker_sell_fill_frac_arr = None if maker_sell_fill_frac_per_step is None else np.asarray(maker_sell_fill_frac_per_step, dtype=np.float64)
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
            "maker_side_hit_rate": float(maker_fill_count / maker_opportunities) if maker_opportunities > 0 else 0.0,
            "turnover_notional": float(np.sum(turnover_arr[mask])) if step_count > 0 else 0.0,
            "delta_equity": float(np.sum(delta_arr[mask])) if step_count > 0 else 0.0,
            "reward": float(np.sum(reward_arr[mask])) if step_count > 0 else 0.0,
            "net_pnl_pct_contrib": float(np.sum(delta_arr[mask]) / max(initial_equity, 1e-12)) if step_count > 0 else 0.0,
        }
        if maker_buy_markout_arr is not None:
            bucket_row["maker_buy_markout"] = float(np.sum(maker_buy_markout_arr[mask])) if step_count > 0 else 0.0
        if maker_sell_markout_arr is not None:
            bucket_row["maker_sell_markout"] = float(np.sum(maker_sell_markout_arr[mask])) if step_count > 0 else 0.0
        if maker_buy_fill_frac_arr is not None:
            bucket_row["maker_buy_fill_frac_mean"] = float(np.mean(maker_buy_fill_frac_arr[mask])) if step_count > 0 else 0.0
        if maker_sell_fill_frac_arr is not None:
            bucket_row["maker_sell_fill_frac_mean"] = float(np.mean(maker_sell_fill_frac_arr[mask])) if step_count > 0 else 0.0
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
      [dir_logits(h), p_up(h), confidence/alignment scalars(5),
       weighted_cmssl_logit(1), abs_weighted_cmssl_logit(1), snapshots(snapshot_dim)]
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
    layout["weighted_cmssl_logit"] = slice(offset, offset + 1)
    offset += 1
    layout["abs_weighted_cmssl_logit"] = slice(offset, offset + 1)
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


def _build_rollout_start_sampler(
    env: "MarketMakingEnv",
    config: RolloutStartSamplingConfig,
    *,
    rollout_horizon: int,
) -> Optional[Dict[str, Any]]:
    max_start = max(0, env.n - 2)
    if not config.enabled or max_start <= 0:
        return None
    safe_rollout_horizon = max(1, int(rollout_horizon))
    min_remaining_steps = int(safe_rollout_horizon)
    effective_max_start = int(np.clip(env.n - min_remaining_steps - 1, 0, max_start))
    if effective_max_start < 0:
        return None
    candidate_starts = np.arange(0, effective_max_start + 1, dtype=np.int64)
    if candidate_starts.size == 0:
        return None
    focus_idx = np.minimum(candidate_starts + int(config.lead_steps), env.n - 2).astype(np.int64, copy=False)
    weighted_slice = env._feature_layout["weighted_cmssl_logit"]
    z = env.features[focus_idx, weighted_slice.start].astype(np.float64, copy=False)
    raw_score = np.abs(z)
    weighted_score = float(config.score_epsilon) + np.power(raw_score, float(config.score_power))
    weighted_mass = weighted_score / max(float(np.sum(weighted_score)), 1e-12)
    uniform_mass = np.full(candidate_starts.shape[0], 1.0 / float(candidate_starts.shape[0]), dtype=np.float64)
    mixed_mass = (1.0 - float(config.weighted_mix)) * uniform_mass + float(config.weighted_mix) * weighted_mass
    mixed_mass = mixed_mass / max(float(np.sum(mixed_mass)), 1e-12)
    top_k = min(5, candidate_starts.shape[0])
    top_idx = np.argsort(weighted_score)[-top_k:][::-1]
    top_focus = []
    for i in top_idx:
        fidx = int(focus_idx[i])
        ts_val = None
        if env.decision_ts is not None and 0 <= fidx < env.decision_ts.shape[0]:
            ts_val = int(env.decision_ts[fidx])
        top_focus.append(
            {
                "start_idx": int(candidate_starts[i]),
                "focus_idx": fidx,
                "focus_ts": ts_val,
                "score": float(weighted_score[i]),
                "abs_logit": float(raw_score[i]),
            }
        )
    if config.start_exclusion_window is None:
        effective_start_exclusion_window = int(safe_rollout_horizon)
    else:
        effective_start_exclusion_window = int(max(0, int(config.start_exclusion_window)))
    return {
        "candidate_starts": candidate_starts,
        "focus_idx": focus_idx,
        "weighted_score": weighted_score.astype(np.float64, copy=False),
        "mixed_mass": mixed_mass.astype(np.float64, copy=False),
        "abs_focus_logit": raw_score.astype(np.float64, copy=False),
        "effective_max_start": int(effective_max_start),
        "min_remaining_steps": int(min_remaining_steps),
        "start_exclusion_window": int(effective_start_exclusion_window),
        "top_focus": top_focus,
        "config": config,
    }


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


def _resolve_meta_full_week_range(out_root: Path, meta: Dict[str, Any], week_key: str, *, label: str) -> Tuple[int, int]:
    weeks_meta_map = meta.get("weeks_meta")
    if not isinstance(weeks_meta_map, dict) or not weeks_meta_map:
        raise KeyError("meta.json missing required non-empty key 'weeks_meta'. Rerun offline_ingest.")
    rel_path = weeks_meta_map.get(week_key)
    if not isinstance(rel_path, str) or not rel_path:
        raise KeyError(f"meta['weeks_meta'] missing path for week '{week_key}' referenced by {label}.")
    week_meta = read_json(out_root / rel_path)
    decision_range = week_meta.get("decision_ts_range")
    if not isinstance(decision_range, dict) or "min" not in decision_range or "max" not in decision_range:
        raise KeyError(f"Week metadata for {label} must include decision_ts_range min/max.")
    start = int(decision_range["min"])
    end = int(decision_range["max"]) + 1
    if start >= end:
        raise ValueError(f"Week metadata for {label} has invalid decision_ts_range: start={start} end={end}.")
    return start, end


def _normalize_pipeline_split_entry(
    out_root: Path,
    meta: Dict[str, Any],
    split_entry: Any,
    *,
    label: str,
    require_range: bool,
) -> Dict[str, Any]:
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
        explicit_range = split_entry.get("decision_ts_range")
        if isinstance(explicit_range, dict) and "start" in explicit_range and "end" in explicit_range:
            start, end = _resolve_split_range(explicit_range, label=label)
        elif "start" in split_entry and "end" in split_entry:
            start, end = _resolve_split_range(split_entry, label=label)
        else:
            start, end = _resolve_meta_full_week_range(out_root, meta, weeks[0], label=label)
    return {"weeks": weeks, "start": start, "end": end}


def require_four_week_pipeline_splits(meta: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    _require_event_time_decision_meta(meta)
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        raise KeyError("meta['splits'] must be a dict.")
    weeks_in_order = meta.get("weeks_in_order")
    if not (isinstance(weeks_in_order, list) and len(weeks_in_order) == 4 and all(isinstance(w, str) and w for w in weeks_in_order)):
        raise KeyError("meta['weeks_in_order'] must be a list[str] with exactly 4 entries.")
    weeks_meta_map = meta.get("weeks_meta")
    if not isinstance(weeks_meta_map, dict) or not weeks_meta_map:
        raise KeyError("meta.json missing required non-empty key 'weeks_meta'. Rerun offline_ingest.")
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
        normalized[section][name] = _normalize_pipeline_split_entry(
            out_root,
            meta,
            splits[section].get(name),
            label=label,
            require_range=require_range,
        )
    week1, week2, week3, week4 = weeks_in_order
    require(normalized["cmssl"]["train"]["weeks"] == [week1], "meta['splits']['cmssl']['train'] must reference weeks_in_order[0].")
    require(normalized["cmssl"]["val"]["weeks"] == [week2], "meta['splits']['cmssl']['val'] must reference weeks_in_order[1].")
    require(normalized["cmssl"]["test"]["weeks"] == [week3], "meta['splits']['cmssl']['test'] must reference weeks_in_order[2].")
    for split_name in ("train", "val", "test"):
        require(normalized["rl"][split_name]["weeks"] == [week3], f"meta['splits']['rl']['{split_name}'] must reference weeks_in_order[2].")
    require(normalized["eval"]["full"]["weeks"] == [week4], "meta['splits']['eval']['full'] must reference weeks_in_order[3].")
    cmssl_test = normalized["cmssl"]["test"]
    week3_start, week3_end = _resolve_meta_full_week_range(out_root, meta, week3, label="weeks_in_order[2]")
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


def resolve_cmssl_train_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["cmssl"]["train"])


def resolve_cmssl_val_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["cmssl"]["val"])


def resolve_cmssl_test_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["cmssl"]["test"])


def resolve_rl_train_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["rl"]["train"])


def resolve_rl_val_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["rl"]["val"])


def resolve_rl_test_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["rl"]["test"])


def resolve_eval_full_split(out_root: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(require_four_week_pipeline_splits(meta, Path(out_root))["eval"]["full"])


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
    split = resolve_cmssl_test_split(out_root, meta)
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


def load_decision_snapshots(out_root: str, week_key: str) -> Tuple[np.ndarray, np.ndarray]:
    week_dir: Optional[Path] = None
    meta = load_global_meta(Path(out_root))
    for wk, _wmeta, wk_dir in iter_week_chunks(Path(out_root), meta=meta):
        if wk == week_key:
            week_dir = wk_dir
            break
    if week_dir is None:
        raise ValueError(f"Unable to locate week directory for {week_key}")
    decision_path = week_dir / "decision_snapshots.npz"
    rebuild_msg = (
        "Decision snapshot artifact is missing/stale/malformed. "
        "Expected keys={ts,snapshots,schema_version,feature_columns_json}, "
        "ts to be 1D monotonically non-decreasing, snapshots to be finite with "
        "the expected feature columns/schema, and row counts to match. "
        f"Expected {decision_path}. Please rerun offline_snapshots.py."
    )
    if not decision_path.exists():
        raise FileNotFoundError(rebuild_msg)
    data = np.load(decision_path)
    if not {"ts", "snapshots", "schema_version", "feature_columns_json"}.issubset(data.files):
        raise ValueError(rebuild_msg)
    ts = np.asarray(data["ts"], dtype=np.int64)
    snapshots = np.asarray(data["snapshots"], dtype=np.float32)
    if ts.ndim != 1:
        raise ValueError(rebuild_msg)
    if snapshots.ndim != 2 or snapshots.shape[1] != len(RAW_SNAPSHOT_FEATURE_COLUMNS):
        raise ValueError(rebuild_msg)
    schema_version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
    if schema_version != DECISION_SNAPSHOTS_SCHEMA_VERSION:
        raise ValueError(rebuild_msg)
    feature_columns_raw = np.asarray(data["feature_columns_json"]).reshape(-1)[0]
    if isinstance(feature_columns_raw, np.bytes_):
        feature_columns_raw = feature_columns_raw.decode("utf-8")
    try:
        feature_columns = json.loads(str(feature_columns_raw))
    except Exception as exc:
        raise ValueError(rebuild_msg) from exc
    if feature_columns != RAW_SNAPSHOT_FEATURE_COLUMNS:
        raise ValueError(rebuild_msg)
    if ts.size and np.any(np.diff(ts) < 0):
        raise ValueError(rebuild_msg)
    if not np.all(np.isfinite(snapshots)):
        raise ValueError(rebuild_msg)
    if snapshots.shape[0] != ts.shape[0]:
        raise ValueError(rebuild_msg)
    return ts, snapshots


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
    test_split = resolve_cmssl_test_split(out_root, meta)
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

    split_token_ts_parts: List[np.ndarray] = []
    decision_snapshot_ts_parts: List[np.ndarray] = []
    for week in split_weeks:
        wk_split = {"weeks": [week], "start": start_ms, "end": end_ms}
        try:
            _x_core, _x_aux, _y, token_ts = load_split_arrays(out_root, wk_split)
        except ValueError as exc:
            if str(exc).startswith("No data found for split"):
                continue
            raise
        split_token_ts_parts.append(np.asarray(token_ts, dtype=np.int64))
        wk_snapshot_ts, _snapshots = load_decision_snapshots(out_root, week)
        mask = (wk_snapshot_ts >= start_ms) & (wk_snapshot_ts < end_ms)
        decision_snapshot_ts_parts.append(np.asarray(wk_snapshot_ts[mask], dtype=np.int64))
    if not split_token_ts_parts:
        raise ValueError("No token decision timestamps found inside the CMSSL test split range.")
    split_token_ts = np.concatenate(split_token_ts_parts, axis=0).astype(np.int64, copy=False)
    decision_snapshot_ts = np.concatenate(decision_snapshot_ts_parts, axis=0).astype(np.int64, copy=False)
    _ensure_monotonic(split_token_ts, "Token decision (filtered)")
    _ensure_monotonic(decision_snapshot_ts, "Decision snapshot (filtered)")
    if split_token_ts.shape != decision_snapshot_ts.shape or not np.array_equal(split_token_ts, decision_snapshot_ts):
        raise ValueError(
            "CMSSL test diagnostics timestamp contract failed: token decision timestamps "
            "do not exactly match decision_snapshots.npz timestamps. Please rerun offline_snapshots.py."
        )
    print(
        "[tokens:test]",
        f"count={split_token_ts.size}",
        f"start={_format_ts(int(split_token_ts[0]))}",
        f"end={_format_ts(int(split_token_ts[-1]))}",
    )
    print(
        "[decision snapshots:test]",
        f"count={decision_snapshot_ts.size}",
        f"start={_format_ts(int(decision_snapshot_ts[0]))}",
        f"end={_format_ts(int(decision_snapshot_ts[-1]))}",
    )


def _resolve_horizon_indices(meta: dict, targets: Iterable[int]) -> Dict[int, int]:
    horizons = [int(h) for h in meta.get("horizons_ms", [])]
    if not horizons:
        raise ValueError("meta['horizons_ms'] must be non-empty")
    index_map = {h: idx for idx, h in enumerate(horizons)}
    missing = [h for h in targets if h not in index_map]
    if missing:
        raise ValueError(f"Requested horizons not in meta: {missing}")
    return {h: index_map[h] for h in targets}


def _sigmoid(x):
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def join_features(
    decision_ts: np.ndarray,
    y: np.ndarray,
    cmssl_out: Dict[str, np.ndarray],
    snapshots: np.ndarray,
    meta: dict,
    directional_signal_config: DirectionalSignalConfig,
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
      4) weighted_cmssl_logit, abs_weighted_cmssl_logit
      5) snapshot features from RAW_SNAPSHOT_FEATURE_COLUMNS
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
    weighted_cmssl_logit = np.sum(
        dir_logits.astype(np.float64, copy=False)
        * np.asarray(directional_signal_config.horizon_logit_weights, dtype=np.float64).reshape(1, 3),
        axis=1,
    ).astype(np.float32, copy=False)
    abs_weighted_cmssl_logit = np.abs(weighted_cmssl_logit).astype(np.float32, copy=False)
    layout = _joined_feature_layout(dir_logits.shape[1], snapshots.shape[1])
    snapshot_spread_col = RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")
    spread_bps = snapshots[:, snapshot_spread_col]  # use aligned snapshot spread
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
    features[:, cursor] = weighted_cmssl_logit
    cursor += 1
    features[:, cursor] = abs_weighted_cmssl_logit
    cursor += 1

    d = snapshots.shape[1]
    features[:, cursor:cursor + d] = snapshots
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
        "features": features,
        "y": y,
        "spread_bps": spread_bps,
        "snapshots": snapshots,
    }
    return output


def _build_joined_split_uncached(
    out_root: str,
    split: Dict[str, Any],
    model,
    meta: dict,
    device: str,
    split_label: str,
    directional_signal_config: DirectionalSignalConfig,
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

        decision_snapshot_ts, aligned_snapshots = load_decision_snapshots(out_root, wk)
        decision_snapshot_ts = np.asarray(decision_snapshot_ts, dtype=np.int64)
        aligned_snapshots = np.asarray(aligned_snapshots, dtype=np.float32)

        window_start = int(split["start"])
        window_end = int(split["end"])
        effective_mask = (decision_snapshot_ts >= window_start) & (decision_snapshot_ts < window_end)
        if np.any(effective_mask):
            decision_snapshot_ts = decision_snapshot_ts[effective_mask]
            aligned_snapshots = aligned_snapshots[effective_mask]
        if decision_snapshot_ts.shape != ts.shape or not np.array_equal(decision_snapshot_ts, ts):
            raise ValueError(
                f"Decision snapshot timestamps do not exactly match token timestamps for week={wk}. "
                "Please rerun offline_snapshots.py to rebuild decision_snapshots.npz."
            )
        week_outputs.append(join_features(ts, y, cmssl_out, aligned_snapshots, meta, directional_signal_config))

    if not week_outputs:
        raise ValueError(f"No data found for split {split}")

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

    expected_rows = out["ts"].shape[0]
    for key, value in out.items():
        if value.shape[0] != expected_rows:
            raise ValueError(
                "build_joined_split row-count mismatch after weekly concatenation: "
                f"ts_rows={expected_rows} {key}_rows={value.shape[0]}"
            )

    ts_all = out["ts"]
    ts_diff = np.diff(ts_all)
    bad_idx = np.where(ts_diff < 0)[0]
    if bad_idx.size > 0:
        first_bad = int(bad_idx[0])
        raise ValueError(
            "build_joined_split requires monotonically non-decreasing concatenated timestamps "
            "(weeks order is preserved; outputs are not resorted). "
            f"first_decreasing_index={first_bad} ts_prev={int(ts_all[first_bad])} "
            f"ts_next={int(ts_all[first_bad + 1])} diff={int(ts_diff[first_bad])}"
        )

    _timing_log(
        f"build_joined_split_uncached label={split_label} rows={out['ts'].shape[0]} "
        f"secs={time.perf_counter() - t0:.4f}"
    )
    return out


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: Any) -> str:
    digest = hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()
    return digest[:20]


def _safe_file_identity(path: Path | str, *, required: bool = True) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    exists = resolved.exists()
    if required and not exists:
        raise FileNotFoundError(f"Required file is missing: {resolved}")
    out: Dict[str, Any] = {
        "path": str(resolved),
        "exists": bool(exists),
        "size": None,
        "mtime_ns": None,
    }
    if exists:
        stat = resolved.stat()
        out["size"] = int(stat.st_size)
        out["mtime_ns"] = int(stat.st_mtime_ns)
    return out


def _joined_cache_paths(out_root: str, split_label: str, fingerprint: str) -> Tuple[Path, Path]:
    cache_dir = Path(out_root).resolve() / "rl_exec_cache" / "joined"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(split_label).strip().lower())
    sanitized_label = sanitized_label or "split"
    stem = f"{sanitized_label}-{fingerprint}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def _build_joined_cache_identity(
    out_root: str,
    split_label: str,
    split: Dict[str, Any],
    meta: Dict[str, Any],
    ckpt_path: str,
    directional_signal_config: DirectionalSignalConfig,
) -> Dict[str, Any]:
    out_root_path = Path(out_root).resolve()
    weeks = [str(w) for w in _split_weeks(split)]
    weeks_meta = meta.get("weeks_meta")
    if not isinstance(weeks_meta, dict):
        raise KeyError("meta['weeks_meta'] must be a dict for joined cache identity.")
    week_refs: Dict[str, Any] = {}
    for week in weeks:
        rel_meta_path = weeks_meta.get(week)
        if not isinstance(rel_meta_path, str) or not rel_meta_path.strip():
            raise KeyError(f"meta['weeks_meta'] missing path for week '{week}'")
        week_meta_path = (out_root_path / rel_meta_path).resolve()
        week_dir = week_meta_path.parent
        week_refs[week] = {
            "week_meta": _safe_file_identity(week_meta_path, required=True),
            "decision_snapshots_npz": _safe_file_identity(week_dir / "decision_snapshots.npz", required=True),
        }
    split_protocol = None
    if isinstance(meta.get("splits"), dict):
        split_protocol = meta["splits"].get("protocol")
    directional_signal_payload = {
        "horizon_logit_weights": [float(v) for v in directional_signal_config.horizon_logit_weights],
        "training_reward_horizon_ms": int(directional_signal_config.training_reward_horizon_ms),
    }
    return {
        "joined_cache_schema_version": JOINED_CACHE_SCHEMA_VERSION,
        "joined_feature_schema_version": JOINED_FEATURE_SCHEMA_VERSION,
        "split_label": str(split_label),
        "split": {
            "weeks": weeks,
            "start": int(split["start"]),
            "end": int(split["end"]),
        },
        "checkpoint": _safe_file_identity(ckpt_path, required=True),
        "dataset_meta_contract": {
            "decision_time_basis": meta.get("decision_time_basis"),
            "decision_policy": meta.get("decision_policy"),
            "trade_history_enabled": meta.get("trade_history_enabled"),
            "event_stream_mode": meta.get("event_stream_mode"),
            "feature_dim_total": int(meta.get("feature_dim_total", -1)),
            "label_dim": int(meta["label_dim"]) if meta.get("label_dim") is not None else None,
            "horizons_ms": [int(v) for v in meta.get("horizons_ms", [])],
            "weeks_in_order": [str(v) for v in meta.get("weeks_in_order", [])] if isinstance(meta.get("weeks_in_order"), list) else None,
            "weeks_meta": {str(k): str(v) for k, v in weeks_meta.items()},
            "splits.protocol": split_protocol,
            "joined_ts_ordering": "nondecreasing",
        },
        "directional_signal_config": directional_signal_payload,
        "referenced_weeks": week_refs,
    }


def _joined_ts_ordering_mode(meta: Optional[Dict[str, Any]]) -> str:
    if not isinstance(meta, dict):
        return "nondecreasing"
    mode = meta.get("joined_ts_ordering")
    if mode in {"strictly_increasing", "nondecreasing"}:
        return str(mode)
    return "nondecreasing"


def _validate_joined_payload(payload: Dict[str, np.ndarray], *, meta: Optional[Dict[str, Any]]) -> None:
    expected_keys = {"ts", "features", "y", "spread_bps", "snapshots"}
    if set(payload.keys()) != expected_keys:
        raise JoinedCacheError(
            f"Joined cache keys mismatch: got={sorted(payload.keys())} expected={sorted(expected_keys)}"
        )
    ts_all = np.asarray(payload["ts"])
    features = np.asarray(payload["features"])
    y = np.asarray(payload["y"])
    spread_bps = np.asarray(payload["spread_bps"])
    snapshots = np.asarray(payload["snapshots"])

    if ts_all.ndim != 1:
        raise JoinedCacheError(f"Joined cache ts must be 1D, got shape={ts_all.shape}")
    if features.ndim != 2:
        raise JoinedCacheError(f"Joined cache features must be 2D, got shape={features.shape}")
    if y.ndim != 2:
        raise JoinedCacheError(f"Joined cache y must be 2D, got shape={y.shape}")
    if spread_bps.ndim != 1:
        raise JoinedCacheError(f"Joined cache spread_bps must be 1D, got shape={spread_bps.shape}")
    if snapshots.ndim != 2:
        raise JoinedCacheError(f"Joined cache snapshots must be 2D, got shape={snapshots.shape}")

    expected_rows = int(ts_all.shape[0])
    for key in ("features", "y", "spread_bps", "snapshots"):
        if int(payload[key].shape[0]) != expected_rows:
            raise JoinedCacheError(
                f"Joined cache row-count mismatch: ts_rows={expected_rows} {key}_rows={int(payload[key].shape[0])}"
            )

    numeric_arrays = {
        "ts": np.asarray(ts_all, dtype=np.int64),
        "features": features,
        "y": y,
        "spread_bps": spread_bps,
        "snapshots": snapshots,
    }
    for key, arr in numeric_arrays.items():
        if not np.issubdtype(arr.dtype, np.number):
            raise JoinedCacheError(f"Joined cache {key} must be numeric dtype, got dtype={arr.dtype}")
        if not np.all(np.isfinite(arr)):
            raise JoinedCacheError(f"Joined cache {key} contains non-finite values.")

    ts_all = numeric_arrays["ts"]
    ordering_mode = _joined_ts_ordering_mode(meta)
    ts_diff = np.diff(ts_all)
    bad_idx = np.where(ts_diff <= 0)[0] if ordering_mode == "strictly_increasing" else np.where(ts_diff < 0)[0]
    if bad_idx.size > 0:
        first_bad = int(bad_idx[0])
        ordering_desc = "non-increasing" if ordering_mode == "strictly_increasing" else "decreasing"
        raise JoinedCacheError(
            f"Joined cache has {ordering_desc} timestamps: "
            f"first_decreasing_index={first_bad} ts_prev={int(ts_all[first_bad])} "
            f"ts_next={int(ts_all[first_bad + 1])} diff={int(ts_diff[first_bad])}"
        )

    if isinstance(meta, dict):
        horizons = [int(h) for h in meta.get("horizons_ms", [])]
        if horizons:
            expected_feature_dim = _joined_feature_layout(len(horizons), snapshots.shape[1])["snapshots"].stop
            if int(features.shape[1]) != int(expected_feature_dim):
                raise JoinedCacheError(
                    "Joined cache feature-dim mismatch against layout contract: "
                    f"features_dim={int(features.shape[1])} expected_dim={int(expected_feature_dim)} "
                    f"num_horizons={len(horizons)} snapshot_dim={int(snapshots.shape[1])}"
                )


def _load_joined_cache(npz_path: Path, json_path: Path, *, meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    expected_payload = dict(meta.get("identity_payload", {}))
    if not (npz_path.exists() and json_path.exists()):
        raise JoinedCacheError(
            f"Joined cache artifact missing: npz_exists={npz_path.exists()} json_exists={json_path.exists()}"
        )
    try:
        sidecar = read_json(json_path)
    except Exception as exc:
        raise JoinedCacheError(f"Failed reading joined cache sidecar: {exc}") from exc
    try:
        if not isinstance(sidecar, dict):
            raise JoinedCacheError("Joined cache sidecar must be a JSON object.")
        if int(sidecar.get("joined_cache_schema_version", -1)) != JOINED_CACHE_SCHEMA_VERSION:
            raise JoinedCacheError(
                "Joined cache schema mismatch: "
                f"got={sidecar.get('joined_cache_schema_version')} expected={JOINED_CACHE_SCHEMA_VERSION}"
            )
        if int(sidecar.get("joined_feature_schema_version", -1)) != JOINED_FEATURE_SCHEMA_VERSION:
            raise JoinedCacheError(
                "Joined feature schema mismatch: "
                f"got={sidecar.get('joined_feature_schema_version')} expected={JOINED_FEATURE_SCHEMA_VERSION}"
            )
        if sidecar.get("identity_payload") != expected_payload:
            raise JoinedCacheError("Joined cache identity payload mismatch (stale cache).")
    except JoinedCacheError:
        raise
    except Exception as exc:
        raise JoinedCacheError(f"Invalid joined cache sidecar contents: {exc}") from exc
    try:
        with np.load(npz_path, allow_pickle=False) as arr:
            out = {key: np.asarray(arr[key]) for key in ("ts", "features", "y", "spread_bps", "snapshots")}
    except Exception as exc:
        raise JoinedCacheError(f"Failed reading joined cache npz: {exc}") from exc
    _validate_joined_payload(out, meta=meta.get("runtime_meta"))
    return out


def _save_joined_cache(
    npz_path: Path,
    json_path: Path,
    payload: Dict[str, np.ndarray],
    sidecar: Dict[str, Any],
) -> None:
    npz_tmp = npz_path.with_name(npz_path.name + f".tmp-{os.getpid()}.npz")
    sidecar_tmp = json_path.with_name(json_path.name + f".tmp-{os.getpid()}")
    with npz_tmp.open("wb") as fh:
        np.savez(
            fh,
            ts=payload["ts"],
            features=payload["features"],
            y=payload["y"],
            spread_bps=payload["spread_bps"],
            snapshots=payload["snapshots"],
        )
    sidecar_tmp.write_text(json.dumps(sidecar, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(npz_tmp, npz_path)
    os.replace(sidecar_tmp, json_path)


def build_joined_split(
    out_root: str,
    split: Dict[str, Any],
    model,
    meta: dict,
    device: str,
    *,
    ckpt_path: str,
    split_label: str,
    directional_signal_config: DirectionalSignalConfig,
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    cache_t0 = time.perf_counter()
    directional_signal_payload = {
        "horizon_logit_weights": [float(v) for v in directional_signal_config.horizon_logit_weights],
        "training_reward_horizon_ms": int(directional_signal_config.training_reward_horizon_ms),
    }
    signal_weights_summary = "[" + ",".join(f"{v:.6g}" for v in directional_signal_payload["horizon_logit_weights"]) + "]"
    train_reward_horizon_ms = directional_signal_payload["training_reward_horizon_ms"]
    payload = _build_joined_cache_identity(
        out_root,
        split_label,
        split,
        meta,
        ckpt_path,
        directional_signal_config,
    )
    fingerprint = _hash_payload(payload)
    npz_path, sidecar_path = _joined_cache_paths(out_root, split_label, fingerprint)

    cache_meta = {"identity_payload": payload, "runtime_meta": meta}
    try:
        cached = _load_joined_cache(npz_path, sidecar_path, meta=cache_meta)
        _timing_log(
            f"[joined-cache] hit label={split_label} fingerprint={fingerprint} "
            f"path={npz_path} rows={cached['ts'].shape[0]} signal_weights={signal_weights_summary} "
            f"train_reward_horizon_ms={train_reward_horizon_ms} secs={time.perf_counter() - cache_t0:.4f}"
        )
        return cached
    except JoinedCacheError as exc:
        reason = str(exc).replace("\n", " ").strip()
        if "artifact missing" in reason:
            _timing_log(
                f"[joined-cache] miss label={split_label} fingerprint={fingerprint} "
                f"path={npz_path} sidecar={sidecar_path} signal_weights={signal_weights_summary} "
                f"train_reward_horizon_ms={train_reward_horizon_ms}"
            )
        else:
            _timing_log(
                f"[joined-cache] stale label={split_label} fingerprint={fingerprint} "
                f"path={npz_path} sidecar={sidecar_path} signal_weights={signal_weights_summary} "
                f"train_reward_horizon_ms={train_reward_horizon_ms} reason={reason} rebuilding"
            )

    out = _build_joined_split_uncached(
        out_root,
        split,
        model,
        meta,
        device,
        split_label=split_label,
        directional_signal_config=directional_signal_config,
        batch_size=batch_size,
    )
    _validate_joined_payload(out, meta=meta)
    sidecar = {
        "joined_cache_schema_version": JOINED_CACHE_SCHEMA_VERSION,
        "joined_feature_schema_version": JOINED_FEATURE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "identity_payload": payload,
        "directional_signal_config": directional_signal_payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_label": str(split_label),
        "row_count": int(out["ts"].shape[0]),
        "feature_shape": [int(v) for v in out["features"].shape],
        "y_shape": [int(v) for v in out["y"].shape],
        "spread_bps_shape": [int(v) for v in out["spread_bps"].shape],
        "snapshots_shape": [int(v) for v in out["snapshots"].shape],
    }
    save_t0 = time.perf_counter()
    _save_joined_cache(npz_path, sidecar_path, out, sidecar)
    _timing_log(
        f"[joined-cache] saved label={split_label} fingerprint={fingerprint} "
        f"path={npz_path} sidecar={sidecar_path} rows={out['ts'].shape[0]} "
        f"feature_dim={out['features'].shape[1]} signal_weights={signal_weights_summary} "
        f"train_reward_horizon_ms={train_reward_horizon_ms} secs={time.perf_counter() - save_t0:.4f}"
    )
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
        continuous_maker_fill_config: ContinuousMakerFillConfig,
        continuous_maker_fill_calibration: ContinuousMakerFillCalibration,
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
        initial_cash: Optional[float] = None,
        obs_norm_state: Optional[dict] = None,
        freeze_obs_norm: bool = False,
        direct_quote_config: Optional[DirectQuoteConfig] = None,
        reward_shaping_config: Optional[RewardShapingConfig] = None,
        directional_signal_config: DirectionalSignalConfig = DirectionalSignalConfig(),
    ):
        self.features = np.ascontiguousarray(np.asarray(batch.features, dtype=np.float32))
        self.spread_bps = np.ascontiguousarray(np.asarray(batch.spread_bps, dtype=np.float32))
        self.best_bid = np.ascontiguousarray(np.asarray(batch.best_bid, dtype=np.float32))
        self.best_ask = np.ascontiguousarray(np.asarray(batch.best_ask, dtype=np.float32))
        self.future_ret_by_horizon = (
            None
            if batch.future_ret_by_horizon is None
            else np.ascontiguousarray(np.asarray(batch.future_ret_by_horizon, dtype=np.float32))
        )
        self.decision_ts = (
            None
            if batch.decision_ts is None
            else np.ascontiguousarray(np.asarray(batch.decision_ts, dtype=np.int64))
        )
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
        self._inv_inventory_notional_scale = (
            1.0 / self.inventory_notional_scale if self.inventory_notional_scale != 0.0 else 0.0
        )
        self._inv_cash_scale = 1.0 / self.cash_scale if self.cash_scale != 0.0 else 0.0
        self._inv_time_since_fill_scale = (
            1.0 / self.time_since_fill_scale if self.time_since_fill_scale != 0.0 else 0.0
        )
        self._inv_fill_notional_scale = (
            1.0 / self.fill_notional_scale if self.fill_notional_scale != 0.0 else 0.0
        )
        self._inv_pnl_notional_scale = (
            1.0 / self.pnl_notional_scale if self.pnl_notional_scale != 0.0 else 0.0
        )
        self._inv_markout_notional_scale = (
            1.0 / self.markout_notional_scale if self.markout_notional_scale != 0.0 else 0.0
        )
        self.direct_quote_config = _validate_direct_quote_config(
            load_direct_quote_config() if direct_quote_config is None else direct_quote_config
        )
        self.reward_shaping_config = (
            load_reward_shaping_config() if reward_shaping_config is None else reward_shaping_config
        )
        self.directional_signal_config = directional_signal_config
        self.continuous_maker_fill_config = continuous_maker_fill_config
        self.continuous_maker_fill_calibration = continuous_maker_fill_calibration
        if self.continuous_maker_fill_calibration.sample_count < 1024:
            raise ValueError(
                "Continuous maker-fill calibration sample_count must be >= 1024, got "
                f"{self.continuous_maker_fill_calibration.sample_count}"
            )
        if self.continuous_maker_fill_calibration.vol_p90_bps <= self.continuous_maker_fill_calibration.vol_p50_bps:
            raise ValueError(
                "Continuous maker-fill calibration requires vol_p90_bps > vol_p50_bps, got "
                f"vol_p50_bps={self.continuous_maker_fill_calibration.vol_p50_bps} "
                f"vol_p90_bps={self.continuous_maker_fill_calibration.vol_p90_bps}"
            )
        print("[mm fill config]", json.dumps(dict(self.continuous_maker_fill_config.__dict__), sort_keys=True))
        print(
            "[mm fill config] "
            f"calibrated=True "
            f"vol_p50_bps={self.continuous_maker_fill_calibration.vol_p50_bps:.6f} "
            f"vol_p90_bps={self.continuous_maker_fill_calibration.vol_p90_bps:.6f}"
        )
        raw_spreads_px = np.asarray(self.best_ask - self.best_bid, dtype=np.float64)
        positive_raw_spreads_px = raw_spreads_px[
            np.isfinite(raw_spreads_px) & (raw_spreads_px > 0.0)
        ]
        if positive_raw_spreads_px.size:
            spread_q05 = float(np.quantile(positive_raw_spreads_px, 0.05))
            self.fill_norm_spread_px_floor = float(max(spread_q05, 1e-6))
        else:
            self.fill_norm_spread_px_floor = 1e-6
        self.mid_px = 0.5 * (self.best_bid.astype(np.float64) + self.best_ask.astype(np.float64))
        mids_px = self.mid_px
        finite_positive_mids = mids_px[np.isfinite(mids_px) & (mids_px > 0.0)]
        self.quote_mid_ref_px = (
            float(np.median(finite_positive_mids)) if finite_positive_mids.size else 1.0
        )
        self.fill_norm_spread_floor_full_bps = float(
            (self.fill_norm_spread_px_floor / max(self.quote_mid_ref_px, 1e-12)) * 1e4
        )
        self.quote_half_spread_floor_bps = float(self.direct_quote_config.quote_half_spread_floor_bps)
        if (
            not np.isfinite(self.quote_half_spread_floor_bps)
            or self.quote_half_spread_floor_bps <= 0.0
        ):
            raise ValueError(
                "quote_half_spread_floor_bps must be finite and > 0.0; "
                f"got {self.quote_half_spread_floor_bps}"
            )
        self.positive_raw_spread_frac = float(
            np.mean(np.isfinite(raw_spreads_px) & (raw_spreads_px > 0.0))
        )
        spread_bps_arr = self.spread_bps.astype(np.float64)
        self.positive_snapshot_spread_frac = float(
            np.mean(np.isfinite(spread_bps_arr) & (spread_bps_arr > 0.0))
        )
        print(
            "[mm quote floor] "
            f"positive_raw_spread_frac={self.positive_raw_spread_frac:.6f} "
            f"positive_snapshot_spread_frac={self.positive_snapshot_spread_frac:.6f} "
            f"fill_norm_spread_px_floor={self.fill_norm_spread_px_floor:.10f} "
            f"quote_mid_ref_px={self.quote_mid_ref_px:.10f} "
            f"fill_norm_spread_floor_full_bps={self.fill_norm_spread_floor_full_bps:.10f} "
            f"quote_half_spread_floor_bps={self.quote_half_spread_floor_bps:.10f}"
        )
        if self.positive_raw_spread_frac < 0.01 or self.positive_snapshot_spread_frac < 0.01:
            print(
                "[mm quote floor warning] "
                "low-positive-spread-fraction detected; quote floor remains active "
                f"positive_raw_spread_frac={self.positive_raw_spread_frac:.6f} "
                f"positive_snapshot_spread_frac={self.positive_snapshot_spread_frac:.6f}"
            )
        self.taker_signal_limit = float(self.direct_quote_config.taker_signal_limit)
        self._num_h = _infer_num_horizons(self.features.shape[-1])
        self._feature_layout = _joined_feature_layout(self._num_h, len(RAW_SNAPSHOT_FEATURE_COLUMNS))
        self._validate_feature_layout()
        vol_short_idx = RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_short")
        vol_long_idx = RAW_SNAPSHOT_FEATURE_COLUMNS.index("vol_long")
        snapshot_features = self.features[:, self._feature_layout["snapshots"]].astype(np.float64, copy=False)
        sigma_recent = np.maximum(snapshot_features[:, vol_short_idx], snapshot_features[:, vol_long_idx])
        if not np.all(np.isfinite(sigma_recent)):
            bad_count = int(np.size(sigma_recent) - np.count_nonzero(np.isfinite(sigma_recent)))
            raise ValueError(
                "Non-finite snapshot volatility encountered in MarketMakingEnv; "
                f"bad_rows={bad_count}"
            )
        self.snapshot_sigma_bps = np.ascontiguousarray((1e4 * sigma_recent).astype(np.float64, copy=False))
        self.last_activity_sigma_bps = 0.0
        self._feature_dim = int(self.features.shape[-1])
        self._obs_dim = self._feature_dim + ENV_OBS_EXTRA_STATE_DIM
        self._obs_feature_slice = slice(0, self._feature_dim)
        self._obs_extra_slice = slice(self._feature_dim, self._obs_dim)
        self._obs_raw_buf_a = np.empty(self._obs_dim, dtype=np.float32)
        self._obs_raw_buf_b = np.empty(self._obs_dim, dtype=np.float32)
        self._obs_out_buf_a = np.empty(self._obs_dim, dtype=np.float32)
        self._obs_out_buf_b = np.empty(self._obs_dim, dtype=np.float32)
        self._obs_ping_pong_idx = 0

        self.n = len(self.spread_bps)
        self.training_reward_horizon_ms = int(self.directional_signal_config.training_reward_horizon_ms)
        if self.decision_ts is None:
            raise ValueError(
                "decision_ts is required for training reward horizon alignment; missing in MarketMakingBatch."
            )
        if self.decision_ts.shape[0] != self.n:
            raise ValueError(
                "decision_ts length mismatch with environment rows: "
                f"decision_ts={self.decision_ts.shape[0]} rows={self.n}"
            )
        future_idx = np.empty((self.n,), dtype=np.int64)
        for i in range(self.n):
            target_ts = int(self.decision_ts[i]) + int(self.training_reward_horizon_ms)
            j = int(np.searchsorted(self.decision_ts, target_ts, side="left"))
            j = int(np.clip(j, i + 1, self.n - 1))
            future_idx[i] = j
        self.training_reward_future_idx = future_idx
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
        self.last_maker_buy_fill_frac = 0.0
        self.last_maker_sell_fill_frac = 0.0
        self.last_activity_score = 0.0
        self.last_touch_dist_buy = 0.0
        self.last_touch_dist_sell = 0.0
        self.last_touch_event_boost_buy = 0.0
        self.last_touch_event_boost_sell = 0.0
        self.last_fill_interaction_buy = 0.0
        self.last_fill_interaction_sell = 0.0
        self.last_resting_quality_buy = 0.0
        self.last_resting_quality_sell = 0.0
        self.last_cross_confirmation_buy = 0.0
        self.last_cross_confirmation_sell = 0.0
        self.last_raw_spread_px = 0.0
        self.last_norm_spread_px = float(self.fill_norm_spread_px_floor)
        self.last_used_norm_spread_floor = 0.0
        self.last_taker_buy_clipped = 0.0
        self.last_taker_sell_clipped = 0.0
        self.last_weighted_cmssl_logit = 0.0
        self.last_abs_weighted_cmssl_logit = 0.0
        self.last_cmssl_alpha = 0.0
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
        self.last_maker_buy_fill_frac = 0.0
        self.last_maker_sell_fill_frac = 0.0
        self.last_activity_score = 0.0
        self.last_touch_dist_buy = 0.0
        self.last_touch_dist_sell = 0.0
        self.last_touch_event_boost_buy = 0.0
        self.last_touch_event_boost_sell = 0.0
        self.last_fill_interaction_buy = 0.0
        self.last_fill_interaction_sell = 0.0
        self.last_resting_quality_buy = 0.0
        self.last_resting_quality_sell = 0.0
        self.last_cross_confirmation_buy = 0.0
        self.last_cross_confirmation_sell = 0.0
        self.last_activity_sigma_bps = 0.0
        self.last_raw_spread_px = 0.0
        self.last_norm_spread_px = float(self.fill_norm_spread_px_floor)
        self.last_used_norm_spread_floor = 0.0
        self.last_taker_buy_clipped = 0.0
        self.last_taker_sell_clipped = 0.0
        self.last_weighted_cmssl_logit = 0.0
        self.last_abs_weighted_cmssl_logit = 0.0
        self.last_cmssl_alpha = 0.0
        self._obs_ping_pong_idx = 0
        mid = self._mid_price(self.idx)
        self.prev_equity = self.cash + self.inventory * mid
        return self._build_observation(self.idx)

    def _mid_price(self, idx: int) -> float:
        return float(self.mid_px[idx])

    def _initial_time_since_last_fill(self) -> float:
        # Prefer a startup value near 1.0 after scaling. This signals "no fill yet"
        # at episode start while keeping 0.0 reserved for a fresh fill event.
        if self.time_since_fill_scale > 0.0:
            return float(self.time_since_fill_scale)
        return 1.0

    def _build_observation(self, idx: int) -> np.ndarray:
        mid = self._mid_price(idx)
        raw = self._obs_raw_buf_a if self._obs_ping_pong_idx == 0 else self._obs_raw_buf_b
        out = self._obs_out_buf_a if self._obs_ping_pong_idx == 0 else self._obs_out_buf_b
        feature_row = self.features[idx]
        inventory_notional_scaled = (self.inventory * mid) * self._inv_inventory_notional_scale
        cash_scaled = self.cash * self._inv_cash_scale
        time_since_last_fill_scaled = self.time_since_last_fill * self._inv_time_since_fill_scale
        unrealized_pnl_notional = (
            self.inventory * (mid - self.avg_entry_price) if self.inventory != 0.0 else 0.0
        )
        unrealized_pnl_scaled = unrealized_pnl_notional * self._inv_pnl_notional_scale
        # Fill-notional `last_*` fields capture the last non-zero fill aggregates.
        # At reset, `time_since_last_fill` starts at a sentinel for "no prior fill"
        # (scaled value ~1.0). A real fill sets it to 0.0. On no-fill steps, it is
        # incremented by (decision_ts[next_idx] - decision_ts[idx]) / RAW_SNAPSHOT_EXPECTED_STEP_MS,
        # i.e., accumulated in RAW_SNAPSHOT_EXPECTED_STEP_MS-equivalent units rather
        # than fixed "1 snapshot == 1 step" units. Under jitter this keeps intent
        # explicit: ~100ms gaps contribute ~1.0, ~300ms gaps contribute ~3.0.
        # `last_*` values persist on no-fill steps.
        raw[self._obs_feature_slice] = feature_row
        extra = raw[self._obs_extra_slice]
        extra[0] = inventory_notional_scaled
        extra[1] = cash_scaled
        extra[2] = time_since_last_fill_scaled
        extra[3] = self.last_maker_buy_notional * self._inv_fill_notional_scale
        extra[4] = self.last_maker_sell_notional * self._inv_fill_notional_scale
        extra[5] = self.last_taker_buy_notional * self._inv_fill_notional_scale
        extra[6] = self.last_taker_sell_notional * self._inv_fill_notional_scale
        extra[7] = self.last_net_fill_notional * self._inv_fill_notional_scale
        extra[8] = self.last_gross_fill_notional * self._inv_fill_notional_scale
        extra[9] = self.ema_net_fill_notional * self._inv_fill_notional_scale
        extra[10] = self.ema_gross_fill_notional * self._inv_fill_notional_scale
        extra[11] = unrealized_pnl_scaled
        extra[12] = self.ema_maker_buy_markout * self._inv_markout_notional_scale
        extra[13] = self.ema_maker_sell_markout * self._inv_markout_notional_scale
        self._normalize_observation_into(raw, out)
        self._obs_ping_pong_idx ^= 1
        return out

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
        expected_obs_dim = self._obs_dim
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
        mask[self._obs_extra_slice] = False
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

    def _normalize_observation_into(self, obs_raw: np.ndarray, out: np.ndarray) -> np.ndarray:
        norm_active = self._obs_count >= 2 and self._obs_mean is not None and self._obs_m2 is not None
        if not norm_active:
            out[:] = obs_raw
        else:
            if self._obs_continuous_mask is None:
                self._obs_continuous_mask = self._continuous_mask(obs_raw.shape[0])
            var = self._obs_m2 / max(self._obs_count - 1, 1)
            std = np.sqrt(np.maximum(var, 1e-6))
            mask = self._obs_continuous_mask
            out[:] = obs_raw
            out[mask] = (obs_raw[mask] - self._obs_mean[mask]) / std[mask]
        if not self.freeze_obs_norm:
            self._update_obs_stats(obs_raw)
        return out

    def _parse_action(self, action: Any) -> Tuple[float, float, float, float]:
        """Hot-path parser for canonical float32 action arrays."""
        if isinstance(action, np.ndarray) and action.dtype == np.float32 and action.shape == (4,):
            return float(action[0]), float(action[1]), float(action[2]), float(action[3])
        # Minimal debug conversion/fallback for non-canonical callers outside rollout hot paths.
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        require(
            action_arr.shape == (4,),
            f"Expected action shape {(4,)}, got shape={action_arr.shape}",
        )
        if not np.all(np.isfinite(action_arr)):
            raise ValueError(f"Action components must be finite, got {action_arr}")
        return float(action_arr[0]), float(action_arr[1]), float(action_arr[2]), float(action_arr[3])

    def _feature_slice(self, idx: int, start: int, end: int) -> np.ndarray:
        return self.features[idx, start:end]

    def _policy_quotes(
        self,
        idx: int,
        center_control: float,
        width_control: float,
        skew_control: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        cfg = self.direct_quote_config
        mid = self._mid_price(idx)
        snapshot_row = self.features[idx, self._feature_layout["snapshots"]]
        observed_spread_bps = float(snapshot_row[RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")])
        observed_anchor_half_spread_bps = 0.0
        if np.isfinite(observed_spread_bps) and observed_spread_bps > 0.0:
            observed_anchor_half_spread_bps = cfg.obs_spread_anchor_frac * observed_spread_bps
        anchor_half_spread_bps = float(
            max(
                self.quote_half_spread_floor_bps,
                observed_anchor_half_spread_bps,
            )
        )
        anchor_half_spread_bps = float(
            np.clip(
                anchor_half_spread_bps,
                self.quote_half_spread_floor_bps,
                cfg.spread_cap_bps,
            )
        )
        wc = float(np.clip(width_control, 0.0, 1.0))
        width_mult = cfg.touch_halfspread_mult + wc * (cfg.wide_halfspread_mult - cfg.touch_halfspread_mult)
        base_half_spread_bps = anchor_half_spread_bps * width_mult
        base_half_spread_bps = float(
            np.clip(
                base_half_spread_bps,
                self.quote_half_spread_floor_bps,
                cfg.spread_cap_bps,
            )
        )

        skew_control_clipped = float(np.clip(skew_control, -1.0, 1.0))
        half_spread_floor_bps = self.quote_half_spread_floor_bps
        total_half_spread_bps = 2.0 * base_half_spread_bps
        asymmetry_cap_bps = max(total_half_spread_bps - 2.0 * half_spread_floor_bps, 0.0)
        asymmetry_target_bps = cfg.asymmetry_residual_frac * skew_control_clipped * asymmetry_cap_bps

        desired_bid_half_spread_bps = 0.5 * (total_half_spread_bps - asymmetry_target_bps)
        bid_half_lower_bps = max(half_spread_floor_bps, total_half_spread_bps - cfg.spread_cap_bps)
        bid_half_upper_bps = min(cfg.spread_cap_bps, total_half_spread_bps - half_spread_floor_bps)
        bid_half_spread_bps = float(np.clip(desired_bid_half_spread_bps, bid_half_lower_bps, bid_half_upper_bps))
        ask_half_spread_bps = float(total_half_spread_bps - bid_half_spread_bps)
        if not (np.isfinite(bid_half_spread_bps) and np.isfinite(ask_half_spread_bps)):
            raise RuntimeError(
                "Non-finite projected half spreads: "
                f"idx={idx} bid_half_spread_bps={bid_half_spread_bps} ask_half_spread_bps={ask_half_spread_bps}"
            )
        if not (half_spread_floor_bps <= bid_half_spread_bps <= cfg.spread_cap_bps):
            raise RuntimeError(
                "Bid half spread outside feasible bounds after projection: "
                f"idx={idx} bid_half_spread_bps={bid_half_spread_bps} "
                f"floor={half_spread_floor_bps} cap={cfg.spread_cap_bps}"
            )
        if not (half_spread_floor_bps <= ask_half_spread_bps <= cfg.spread_cap_bps):
            raise RuntimeError(
                "Ask half spread outside feasible bounds after projection: "
                f"idx={idx} ask_half_spread_bps={ask_half_spread_bps} "
                f"floor={half_spread_floor_bps} cap={cfg.spread_cap_bps}"
            )
        effective_half_spread_bps = 0.5 * (bid_half_spread_bps + ask_half_spread_bps)
        bid_half_spread_px = bps_to_px(mid, bid_half_spread_bps)
        ask_half_spread_px = bps_to_px(mid, ask_half_spread_bps)

        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        eps_px = max(1e-8, mid * 1e-6)
        center_shift_min_px = (best_bid + eps_px) - mid - ask_half_spread_px
        center_shift_max_px = (best_ask - eps_px) - mid + bid_half_spread_px
        if center_shift_min_px > center_shift_max_px + 1e-12:
            raise RuntimeError(
                "Passive-feasible center interval invalid: "
                f"idx={idx} best_bid={best_bid:.12f} best_ask={best_ask:.12f} mid={mid:.12f} "
                f"bid_half_spread_px={bid_half_spread_px:.12f} ask_half_spread_px={ask_half_spread_px:.12f} "
                f"center_shift_min_px={center_shift_min_px:.12f} center_shift_max_px={center_shift_max_px:.12f}"
            )

        center_control_clipped = float(np.clip(center_control, -1.0, 1.0))
        center_shift_mid_px = 0.5 * (center_shift_min_px + center_shift_max_px)
        center_shift_half_range_px = 0.5 * (center_shift_max_px - center_shift_min_px)

        inventory_center_shift_px_raw = (
            cfg.inventory_center_weight * center_control_clipped * center_shift_half_range_px
        )
        inventory_only_center_shift_px = float(
            np.clip(
                center_shift_mid_px + inventory_center_shift_px_raw,
                center_shift_min_px,
                center_shift_max_px,
            )
        )

        nominal_alpha_center_capacity_px = cfg.alpha_center_weight * center_shift_half_range_px
        positive_alpha_capacity_px = min(
            nominal_alpha_center_capacity_px,
            max(0.0, center_shift_max_px - inventory_only_center_shift_px),
        )
        negative_alpha_capacity_px = min(
            nominal_alpha_center_capacity_px,
            max(0.0, inventory_only_center_shift_px - center_shift_min_px),
        )
        if skew_control_clipped >= 0.0:
            alpha_center_shift_px_raw = skew_control_clipped * positive_alpha_capacity_px
        else:
            alpha_center_shift_px_raw = skew_control_clipped * negative_alpha_capacity_px

        center_shift_px = float(
            np.clip(
                inventory_only_center_shift_px + alpha_center_shift_px_raw,
                center_shift_min_px,
                center_shift_max_px,
            )
        )
        actual_inventory_center_shift_px = inventory_only_center_shift_px - center_shift_mid_px
        actual_alpha_center_shift_px = center_shift_px - inventory_only_center_shift_px

        if actual_alpha_center_shift_px > 0.0:
            effective_alpha_center_capacity_px = positive_alpha_capacity_px
        elif actual_alpha_center_shift_px < 0.0:
            effective_alpha_center_capacity_px = negative_alpha_capacity_px
        else:
            effective_alpha_center_capacity_px = max(positive_alpha_capacity_px, negative_alpha_capacity_px)

        center_shift_bps = center_shift_px / max(mid, 1e-12) * 1e4
        inventory_center_shift_bps = actual_inventory_center_shift_px / max(mid, 1e-12) * 1e4
        alpha_center_shift_bps = actual_alpha_center_shift_px / max(mid, 1e-12) * 1e4
        center_shift_scale_bps = center_shift_half_range_px / max(mid, 1e-12) * 1e4
        center_shift_min_bps = center_shift_min_px / max(mid, 1e-12) * 1e4
        center_shift_max_bps = center_shift_max_px / max(mid, 1e-12) * 1e4
        nominal_alpha_center_capacity_bps = nominal_alpha_center_capacity_px / max(mid, 1e-12) * 1e4
        positive_alpha_capacity_bps = positive_alpha_capacity_px / max(mid, 1e-12) * 1e4
        negative_alpha_capacity_bps = negative_alpha_capacity_px / max(mid, 1e-12) * 1e4
        effective_alpha_center_capacity_bps = effective_alpha_center_capacity_px / max(mid, 1e-12) * 1e4

        bid = mid + center_shift_px - bid_half_spread_px
        ask = mid + center_shift_px + ask_half_spread_px
        skew_bps = 0.5 * (ask_half_spread_bps - bid_half_spread_bps)
        directional_center_response = float(
            np.clip(actual_alpha_center_shift_px / max(effective_alpha_center_capacity_px, 1e-12), -1.0, 1.0)
        )
        directional_asym_response = (ask_half_spread_bps - bid_half_spread_bps) / max(
            ask_half_spread_bps + bid_half_spread_bps,
            1e-12,
        )
        directional_response = (
            cfg.directional_response_center_weight * directional_center_response
            + cfg.directional_response_asym_weight * directional_asym_response
        )
        bid_at_floor = bid_half_spread_bps <= (half_spread_floor_bps + 1e-12)
        ask_at_floor = ask_half_spread_bps <= (half_spread_floor_bps + 1e-12)
        quote_metrics = {
            "observed_spread_bps": float(observed_spread_bps),
            "observed_anchor_half_spread_bps": float(observed_anchor_half_spread_bps),
            "fill_norm_spread_floor_full_bps": float(self.fill_norm_spread_floor_full_bps),
            "quote_half_spread_floor_bps": float(self.quote_half_spread_floor_bps),
            "inventory_center_weight": float(cfg.inventory_center_weight),
            "alpha_center_weight": float(cfg.alpha_center_weight),
            "asymmetry_residual_frac": float(cfg.asymmetry_residual_frac),
            "directional_response_center_weight": float(cfg.directional_response_center_weight),
            "directional_response_asym_weight": float(cfg.directional_response_asym_weight),
            "anchor_half_spread_bps": float(anchor_half_spread_bps),
            "width_control": float(wc),
            "width_mult": float(width_mult),
            "base_half_spread_bps": float(base_half_spread_bps),
            "half_spread_bps": float(effective_half_spread_bps),
            "bid_half_spread_bps": float(bid_half_spread_bps),
            "ask_half_spread_bps": float(ask_half_spread_bps),
            "center_shift_scale_bps": float(center_shift_scale_bps),
            "center_shift_min_bps": float(center_shift_min_bps),
            "center_shift_max_bps": float(center_shift_max_bps),
            "nominal_alpha_center_capacity_bps": float(nominal_alpha_center_capacity_bps),
            "positive_alpha_capacity_bps": float(positive_alpha_capacity_bps),
            "negative_alpha_capacity_bps": float(negative_alpha_capacity_bps),
            "effective_alpha_center_capacity_bps": float(effective_alpha_center_capacity_bps),
            "center_control": float(center_control_clipped),
            "center_shift_bps": float(center_shift_bps),
            "inventory_center_shift_bps": float(inventory_center_shift_bps),
            "alpha_center_shift_bps": float(alpha_center_shift_bps),
            "skew_control": float(skew_control_clipped),
            "skew_bps": float(skew_bps),
            "directional_center_response": float(directional_center_response),
            "directional_asym_response": float(directional_asym_response),
            "directional_response": float(directional_response),
            "bid_at_floor": bool(bid_at_floor),
            "ask_at_floor": bool(ask_at_floor),
        }
        return bid, ask, quote_metrics

    def _assert_passive_quotes(self, bid: float, ask: float, idx: int, quote_metrics: Mapping[str, Any]) -> None:
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        mid = 0.5 * (best_bid + best_ask)
        eps_px = max(1e-8, mid * 1e-6)
        violations: List[str] = []
        if not (bid <= best_ask - eps_px + 1e-12):
            violations.append("bid<=best_ask-eps")
        if not (ask >= best_bid + eps_px - 1e-12):
            violations.append("ask>=best_bid+eps")
        if not (bid < ask):
            violations.append("bid<ask")
        if violations:
            raise RuntimeError(
                "Passive quote invariant violation: "
                f"idx={idx} bid={bid:.12f} ask={ask:.12f} best_bid={best_bid:.12f} best_ask={best_ask:.12f} "
                f"eps_px={eps_px:.12f} violations={violations} quote_metrics={quote_metrics}"
            )

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

    def _volatility_activity_score(self, decision_idx: int) -> float:
        sigma_bps = float(self.snapshot_sigma_bps[decision_idx])
        if not np.isfinite(sigma_bps):
            raise ValueError(
                f"Non-finite snapshot sigma_bps at decision_idx={decision_idx}: sigma_bps={sigma_bps}"
            )
        cfg = self.continuous_maker_fill_config
        cal = self.continuous_maker_fill_calibration
        u = (sigma_bps - cal.vol_p50_bps) / (cal.vol_p90_bps - cal.vol_p50_bps)
        u = float(np.clip(u, 0.0, 1.0))
        self.last_activity_sigma_bps = sigma_bps
        return float(cfg.activity_min + (cfg.activity_max - cfg.activity_min) * u)

    def _weighted_cmssl_logit(self, decision_idx: int) -> float:
        weighted_slice = self._feature_layout["weighted_cmssl_logit"]
        return float(self.features[decision_idx, weighted_slice.start])

    def _abs_weighted_cmssl_logit(self, decision_idx: int) -> float:
        abs_weighted_slice = self._feature_layout["abs_weighted_cmssl_logit"]
        return float(self.features[decision_idx, abs_weighted_slice.start])

    def _compute_maker_fill_hard(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        touch_epsilon = self.continuous_maker_fill_config.price_epsilon_px
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

    def _compute_maker_fill_continuous(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        decision_idx = max(0, idx - 1)
        best_bid_t = float(self.best_bid[decision_idx])
        best_ask_t = float(self.best_ask[decision_idx])
        best_bid_next = float(self.best_bid[idx])
        best_ask_next = float(self.best_ask[idx])
        cfg = self.continuous_maker_fill_config
        raw_spread_px = max(best_ask_t - best_bid_t, 0.0)
        norm_spread_px = max(raw_spread_px, self.fill_norm_spread_px_floor)
        activity = self._volatility_activity_score(decision_idx)
        self.last_activity_score = activity
        bid_enabled = np.isfinite(bid)
        ask_enabled = np.isfinite(ask)

        buy_fill_frac = 0.0
        sell_fill_frac = 0.0
        touch_dist_buy = 0.0
        touch_dist_sell = 0.0

        touch_boost_buy = 0.0
        touch_boost_sell = 0.0
        resting_quality_buy = 0.0
        resting_quality_sell = 0.0
        cross_confirmation_buy = 0.0
        cross_confirmation_sell = 0.0
        touch_dist_threshold = max(0.0, cfg.touch_event_distance_frac)
        touch_epsilon = max(self.fill_tolerance, cfg.price_epsilon_px)
        best_bid_prev = float(self.best_bid[idx - 1]) if idx > 0 else best_bid_t
        best_ask_prev = float(self.best_ask[idx - 1]) if idx > 0 else best_ask_t
        if bid_enabled:
            touch_dist_buy = float(max(0.0, best_bid_t - bid) / norm_spread_px)
            cross_gap_buy = float((best_ask_next - bid) / norm_spread_px)
            resting_quality_buy = float(np.exp(-((touch_dist_buy / cfg.tau_touch) ** 2)))
            cross_confirmation_buy = float(_sigmoid(-cross_gap_buy / cfg.tau_cross))
            at_touch_buy = (
                touch_dist_buy <= touch_dist_threshold
                and abs(bid - best_bid_prev) <= touch_epsilon
                and best_bid_next < best_bid_prev - touch_epsilon
            )
            touch_boost_buy = cfg.touch_event_boost if at_touch_buy else 0.0
            buy_fill_base = activity * resting_quality_buy * cross_confirmation_buy
            buy_fill_frac = float(np.clip(buy_fill_base * (1.0 + touch_boost_buy), 0.0, 1.0))
        if ask_enabled:
            touch_dist_sell = float(max(0.0, ask - best_ask_t) / norm_spread_px)
            cross_gap_sell = float((ask - best_bid_next) / norm_spread_px)
            resting_quality_sell = float(np.exp(-((touch_dist_sell / cfg.tau_touch) ** 2)))
            cross_confirmation_sell = float(_sigmoid(-cross_gap_sell / cfg.tau_cross))
            at_touch_sell = (
                touch_dist_sell <= touch_dist_threshold
                and abs(ask - best_ask_prev) <= touch_epsilon
                and best_ask_next > best_ask_prev + touch_epsilon
            )
            touch_boost_sell = cfg.touch_event_boost if at_touch_sell else 0.0
            sell_fill_base = activity * resting_quality_sell * cross_confirmation_sell
            sell_fill_frac = float(np.clip(sell_fill_base * (1.0 + touch_boost_sell), 0.0, 1.0))

        self.last_touch_dist_buy = float(touch_dist_buy)
        self.last_touch_dist_sell = float(touch_dist_sell)
        self.last_touch_event_boost_buy = float(touch_boost_buy)
        self.last_touch_event_boost_sell = float(touch_boost_sell)
        self.last_resting_quality_buy = float(resting_quality_buy)
        self.last_resting_quality_sell = float(resting_quality_sell)
        self.last_cross_confirmation_buy = float(cross_confirmation_buy)
        self.last_cross_confirmation_sell = float(cross_confirmation_sell)
        self.last_fill_interaction_buy = float(resting_quality_buy * cross_confirmation_buy)
        self.last_fill_interaction_sell = float(resting_quality_sell * cross_confirmation_sell)
        self.last_raw_spread_px = float(raw_spread_px)
        self.last_norm_spread_px = float(norm_spread_px)
        self.last_used_norm_spread_floor = float(raw_spread_px < self.fill_norm_spread_px_floor)
        self.last_maker_buy_fill_frac = float(buy_fill_frac)
        self.last_maker_sell_fill_frac = float(sell_fill_frac)

        mid_for_cap = self._mid_price(idx)
        requested_buy = self.fill_size * buy_fill_frac
        requested_sell = self.fill_size * sell_fill_frac
        clipped_buy = self._clip_fill_qty(1, requested_buy, mid_for_cap)
        self.last_maker_buy_clipped = max(0.0, requested_buy - clipped_buy)
        buy_fill = self._apply_signed_fill(1, clipped_buy, bid) if bid_enabled else 0.0
        clipped_sell = self._clip_fill_qty(-1, requested_sell, mid_for_cap)
        self.last_maker_sell_clipped = max(0.0, requested_sell - clipped_sell)
        sell_fill = self._apply_signed_fill(-1, clipped_sell, ask) if ask_enabled else 0.0
        return float(buy_fill), float(sell_fill)

    def _apply_taker(self, idx: int, taker_signal: float) -> Tuple[float, float]:
        self.last_taker_buy_clipped = 0.0
        self.last_taker_sell_clipped = 0.0
        if not self.allow_taker:
            return 0.0, 0.0
        taker_signal = float(np.clip(taker_signal, -self.taker_signal_limit, self.taker_signal_limit))
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

    def _step_from_action_components(
        self,
        center_control: float,
        width_control: float,
        skew_control: float,
        taker_signal: float,
        *,
        emit_info: bool = False,
    ) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, float]]]:
        return self._step_from_action_components_with_fill(
            center_control,
            width_control,
            skew_control,
            taker_signal,
            emit_info=emit_info,
            maker_fill_fn=self._compute_maker_fill_continuous,
            maker_fill_postprocess_fn=None,
        )

    def _step_from_action_components_hard_diagnostic(
        self,
        center_control: float,
        width_control: float,
        skew_control: float,
        taker_signal: float,
        *,
        emit_info: bool = False,
    ) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, float]]]:
        return self._step_from_action_components_with_fill(
            center_control,
            width_control,
            skew_control,
            taker_signal,
            emit_info=emit_info,
            maker_fill_fn=self._compute_maker_fill_hard,
            maker_fill_postprocess_fn=self._postprocess_hard_maker_fill,
        )

    def _postprocess_hard_maker_fill(self, maker_buy: float, maker_sell: float) -> None:
        self.last_maker_buy_fill_frac = 1.0 if maker_buy > 0.0 else 0.0
        self.last_maker_sell_fill_frac = 1.0 if maker_sell > 0.0 else 0.0
        self.last_activity_score = 0.0
        self.last_touch_dist_buy = 0.0
        self.last_touch_dist_sell = 0.0
        self.last_raw_spread_px = 0.0
        self.last_norm_spread_px = float(self.fill_norm_spread_px_floor)
        self.last_used_norm_spread_floor = 1.0

    def _step_from_action_components_with_fill(
        self,
        center_control: float,
        width_control: float,
        skew_control: float,
        taker_signal: float,
        *,
        emit_info: bool,
        maker_fill_fn,
        maker_fill_postprocess_fn=None,
    ) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, float]]]:
        # Execution convention: both maker and taker fills are priced using the next snapshot
        # (next_idx). We quote on self.idx, then advance state after applying fills at next_idx.
        next_idx = self.idx + 1
        if next_idx >= self.n:
            mid = self._mid_price(self.idx)
            if not emit_info:
                return self._build_observation(self.idx), 0.0, True, None

            hard_cap_qty = self._inventory_cap_qty(mid)
            pre_buy_room_qty = self._remaining_inventory_room(1, mid)
            pre_sell_room_qty = self._remaining_inventory_room(-1, mid)
            equity = self.cash + self.inventory * mid
            info = {
                "reward": 0.0,
                "reward_true": 0.0,
                "reward_train": 0.0,
                "reward_shape_skew": 0.0,
                "reward_shape_total": 0.0,
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
                "pre_hard_cap_qty": float(hard_cap_qty),
                "pre_buy_room_qty": float(pre_buy_room_qty),
                "pre_sell_room_qty": float(pre_sell_room_qty),
                "post_hard_cap_qty": float(hard_cap_qty),
                "post_buy_room_qty": float(pre_buy_room_qty),
                "post_sell_room_qty": float(pre_sell_room_qty),
                "bid": 0.0,
                "ask": 0.0,
                "center_control": 0.0,
                "center_shift_bps": 0.0,
                "center_shift_scale_bps": 0.0,
                "center_shift_min_bps": 0.0,
                "center_shift_max_bps": 0.0,
                "nominal_alpha_center_capacity_bps": 0.0,
                "positive_alpha_capacity_bps": 0.0,
                "negative_alpha_capacity_bps": 0.0,
                "effective_alpha_center_capacity_bps": 0.0,
                "observed_spread_bps": 0.0,
                "observed_anchor_half_spread_bps": 0.0,
                "fill_norm_spread_floor_full_bps": float(self.fill_norm_spread_floor_full_bps),
                "quote_half_spread_floor_bps": float(self.quote_half_spread_floor_bps),
                "inventory_center_weight": float(self.direct_quote_config.inventory_center_weight),
                "alpha_center_weight": float(self.direct_quote_config.alpha_center_weight),
                "asymmetry_residual_frac": float(self.direct_quote_config.asymmetry_residual_frac),
                "directional_response_center_weight": float(self.direct_quote_config.directional_response_center_weight),
                "directional_response_asym_weight": float(self.direct_quote_config.directional_response_asym_weight),
                "width_control": 0.0,
                "width_mult": 0.0,
                "skew_control": 0.0,
                "anchor_half_spread_bps": 0.0,
                "base_half_spread_bps": 0.0,
                "half_spread_bps": 0.0,
                "bid_half_spread_bps": float(self.quote_half_spread_floor_bps),
                "ask_half_spread_bps": float(self.quote_half_spread_floor_bps),
                "skew_bps": 0.0,
                "inventory_center_shift_bps": 0.0,
                "alpha_center_shift_bps": 0.0,
                "directional_center_response": 0.0,
                "directional_asym_response": 0.0,
                "directional_response": 0.0,
                "bid_at_floor": 1.0,
                "ask_at_floor": 1.0,
                "weighted_cmssl_logit": float(self.last_weighted_cmssl_logit),
                "abs_weighted_cmssl_logit": float(self.last_abs_weighted_cmssl_logit),
                "cmssl_alpha": float(self.last_cmssl_alpha),
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
                "maker_buy_fill_frac": float(self.last_maker_buy_fill_frac),
                "maker_sell_fill_frac": float(self.last_maker_sell_fill_frac),
                "maker_buy_exec_qty": 0.0,
                "maker_sell_exec_qty": 0.0,
                "activity_score": float(self.last_activity_score),
                "activity_sigma_bps": float(self.last_activity_sigma_bps),
                "touch_dist_buy": float(self.last_touch_dist_buy),
                "touch_dist_sell": float(self.last_touch_dist_sell),
                "touch_event_boost_buy": float(self.last_touch_event_boost_buy),
                "touch_event_boost_sell": float(self.last_touch_event_boost_sell),
                "resting_quality_buy": float(self.last_resting_quality_buy),
                "resting_quality_sell": float(self.last_resting_quality_sell),
                "cross_confirmation_buy": float(self.last_cross_confirmation_buy),
                "cross_confirmation_sell": float(self.last_cross_confirmation_sell),
                "fill_interaction_buy": float(self.last_fill_interaction_buy),
                "fill_interaction_sell": float(self.last_fill_interaction_sell),
                "raw_spread_px": float(self.last_raw_spread_px),
                "norm_spread_px": float(self.last_norm_spread_px),
                "used_norm_spread_floor": float(self.last_used_norm_spread_floor),
                "taker_buy_clipped": float(self.last_taker_buy_clipped),
                "taker_sell_clipped": float(self.last_taker_sell_clipped),
            }
            return self._build_observation(self.idx), 0.0, True, info
        bid, ask, quote_metrics = self._policy_quotes(self.idx, center_control, width_control, skew_control)
        self._assert_passive_quotes(bid, ask, self.idx, quote_metrics)
        self.last_weighted_cmssl_logit = self._weighted_cmssl_logit(self.idx)
        self.last_abs_weighted_cmssl_logit = self._abs_weighted_cmssl_logit(self.idx)
        cmssl_alpha = float(np.tanh(self.last_weighted_cmssl_logit / self.reward_shaping_config.logit_tanh_scale))
        self.last_cmssl_alpha = cmssl_alpha
        inv_prev = self.inventory
        mid_for_cap = self._mid_price(next_idx)
        pre_hard_cap_qty = self._inventory_cap_qty(mid_for_cap) if emit_info else 0.0
        pre_buy_room_qty = self._remaining_inventory_room(1, mid_for_cap) if emit_info else 0.0
        pre_sell_room_qty = self._remaining_inventory_room(-1, mid_for_cap) if emit_info else 0.0
        # Clipping is evaluated per fill attempt, so maker/taker clipped amounts reflect
        # evolving inventory after each in-step fill is applied.
        maker_buy, maker_sell = maker_fill_fn(bid, ask, next_idx)
        if maker_fill_postprocess_fn is not None:
            maker_fill_postprocess_fn(maker_buy, maker_sell)
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
        maker_buy_notional = maker_buy * bid if maker_buy > 0.0 and np.isfinite(bid) else 0.0
        maker_sell_notional = maker_sell * ask if maker_sell > 0.0 and np.isfinite(ask) else 0.0
        maker_rebate_notional = maker_buy_notional + maker_sell_notional
        rebate = maker_rebate_notional * self.maker_rebate_bps * 1e-4
        taker_buy_notional = taker_buy * best_ask_next if taker_buy > 0.0 and np.isfinite(best_ask_next) else 0.0
        taker_sell_notional = taker_sell * best_bid_next if taker_sell > 0.0 and np.isfinite(best_bid_next) else 0.0
        taker_notional = taker_buy_notional + taker_sell_notional
        taker_fee = taker_notional * self.taker_fee_bps * 1e-4
        self.cash += rebate - taker_fee

        buy_notional_total = maker_buy_notional + taker_buy_notional
        sell_notional_total = maker_sell_notional + taker_sell_notional
        net_fill_notional = buy_notional_total - sell_notional_total
        gross_fill_notional = buy_notional_total + sell_notional_total
        maker_buy_markout = (mid_next - bid) * maker_buy if maker_buy > 0.0 and np.isfinite(bid) else 0.0
        maker_sell_markout = (ask - mid_next) * maker_sell if maker_sell > 0.0 and np.isfinite(ask) else 0.0

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
        reward_true = delta_equity - inventory_penalty_total - turnover_penalty
        future_idx = int(self.training_reward_future_idx[self.idx])
        mid_future = float(self.mid_px[future_idx])
        equity_future = self.cash + self.inventory * mid_future
        delta_equity_future = equity_future - self.prev_equity
        penalty_future = self._compute_penalty(mid_future)
        inv_notional_future = abs(inv_new * mid_future)
        excess_notional_future = max(0.0, inv_notional_future - self.inv_soft_notional)
        inv_penalty_future = (
            self.lambda_inv * (excess_notional_future / self.inv_soft_notional) ** 2
            if self.inv_soft_notional > 0.0
            else 0.0
        )
        inventory_penalty_total_future = self._combine_inventory_penalties(penalty_future, inv_penalty_future)
        reward_true_future = delta_equity_future - inventory_penalty_total_future - turnover_penalty
        reward_future_bonus = reward_true_future - reward_true
        reward_train_econ = reward_true + reward_future_bonus
        reward_shape_skew = (
            self.reward_shaping_config.skew_coef
            * cmssl_alpha
            * float(quote_metrics["directional_response"])
        )
        reward_shape_total = reward_shape_skew
        reward_train = reward_train_econ + reward_shape_total

        self.prev_equity = equity
        self.total_reward += reward_train
        self.idx = next_idx
        done = self.idx >= self.n - 1
        next_obs = self._build_observation(self.idx)
        if not np.all(np.isfinite(next_obs)):
            raise RuntimeError(
                f"Non-finite observation at idx={self.idx}: "
                f"bid={bid}, ask={ask}, maker_buy={maker_buy}, maker_sell={maker_sell}, "
                f"taker_buy={taker_buy}, taker_sell={taker_sell}"
            )
        if not emit_info:
            return next_obs, float(reward_train), done, None

        post_hard_cap_qty = self._inventory_cap_qty(mid_next)
        post_buy_room_qty = self._remaining_inventory_room(1, mid_next)
        post_sell_room_qty = self._remaining_inventory_room(-1, mid_next)
        info = {
            "reward": float(reward_train),
            "reward_true": float(reward_true),
            "reward_true_future": float(reward_true_future),
            "reward_future_bonus": float(reward_future_bonus),
            "reward_train_econ": float(reward_train_econ),
            "reward_train": float(reward_train),
            "reward_shape_skew": float(reward_shape_skew),
            "reward_shape_total": float(reward_shape_total),
            "total_reward": float(self.total_reward),
            "cash": float(self.cash),
            "inventory": float(self.inventory),
            "inventory_notional": float(inv_notional),
            "equity": float(equity),
            "delta_equity": float(delta_equity),
            "delta_equity_future": float(delta_equity_future),
            "rebate": float(rebate),
            "taker_fee": float(taker_fee),
            "penalty": float(penalty),
            "inv_penalty": float(inv_penalty),
            "inventory_excess_notional": float(excess_notional),
            "inventory_penalty_total": float(inventory_penalty_total),
            "turnover_penalty": float(turnover_penalty),
            "mid": float(mid_next),
            "mid_future": float(mid_future),
            "training_reward_future_idx": int(future_idx),
            "training_reward_actual_horizon_ms": int(self.decision_ts[future_idx] - self.decision_ts[self.idx]),
            "hard_max_inventory_notional": float(self.hard_max_inventory_notional),
            "pre_hard_cap_qty": float(pre_hard_cap_qty),
            "pre_buy_room_qty": float(pre_buy_room_qty),
            "pre_sell_room_qty": float(pre_sell_room_qty),
            "post_hard_cap_qty": float(post_hard_cap_qty),
            "post_buy_room_qty": float(post_buy_room_qty),
            "post_sell_room_qty": float(post_sell_room_qty),
            "bid": float(bid),
            "ask": float(ask),
            "center_control": float(quote_metrics["center_control"]),
            "center_shift_bps": float(quote_metrics["center_shift_bps"]),
            "center_shift_scale_bps": float(quote_metrics["center_shift_scale_bps"]),
            "center_shift_min_bps": float(quote_metrics["center_shift_min_bps"]),
            "center_shift_max_bps": float(quote_metrics["center_shift_max_bps"]),
            "nominal_alpha_center_capacity_bps": float(quote_metrics["nominal_alpha_center_capacity_bps"]),
            "positive_alpha_capacity_bps": float(quote_metrics["positive_alpha_capacity_bps"]),
            "negative_alpha_capacity_bps": float(quote_metrics["negative_alpha_capacity_bps"]),
            "effective_alpha_center_capacity_bps": float(quote_metrics["effective_alpha_center_capacity_bps"]),
            "observed_spread_bps": float(quote_metrics["observed_spread_bps"]),
            "observed_anchor_half_spread_bps": float(quote_metrics["observed_anchor_half_spread_bps"]),
            "fill_norm_spread_floor_full_bps": float(quote_metrics["fill_norm_spread_floor_full_bps"]),
            "quote_half_spread_floor_bps": float(quote_metrics["quote_half_spread_floor_bps"]),
            "inventory_center_weight": float(quote_metrics["inventory_center_weight"]),
            "alpha_center_weight": float(quote_metrics["alpha_center_weight"]),
            "asymmetry_residual_frac": float(quote_metrics["asymmetry_residual_frac"]),
            "directional_response_center_weight": float(quote_metrics["directional_response_center_weight"]),
            "directional_response_asym_weight": float(quote_metrics["directional_response_asym_weight"]),
            "width_control": float(quote_metrics["width_control"]),
            "width_mult": float(quote_metrics["width_mult"]),
            "skew_control": float(quote_metrics["skew_control"]),
            "anchor_half_spread_bps": float(quote_metrics["anchor_half_spread_bps"]),
            "base_half_spread_bps": float(quote_metrics["base_half_spread_bps"]),
            "half_spread_bps": float(quote_metrics["half_spread_bps"]),
            "bid_half_spread_bps": float(quote_metrics["bid_half_spread_bps"]),
            "ask_half_spread_bps": float(quote_metrics["ask_half_spread_bps"]),
            "skew_bps": float(quote_metrics["skew_bps"]),
            "inventory_center_shift_bps": float(quote_metrics["inventory_center_shift_bps"]),
            "alpha_center_shift_bps": float(quote_metrics["alpha_center_shift_bps"]),
            "directional_center_response": float(quote_metrics["directional_center_response"]),
            "directional_asym_response": float(quote_metrics["directional_asym_response"]),
            "directional_response": float(quote_metrics["directional_response"]),
            "bid_at_floor": float(quote_metrics["bid_at_floor"]),
            "ask_at_floor": float(quote_metrics["ask_at_floor"]),
            "weighted_cmssl_logit": float(self.last_weighted_cmssl_logit),
            "abs_weighted_cmssl_logit": float(self.last_abs_weighted_cmssl_logit),
            "cmssl_alpha": float(cmssl_alpha),
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
            "maker_buy_fill_frac": float(self.last_maker_buy_fill_frac),
            "maker_sell_fill_frac": float(self.last_maker_sell_fill_frac),
            "maker_buy_exec_qty": float(maker_buy),
            "maker_sell_exec_qty": float(maker_sell),
            "activity_score": float(self.last_activity_score),
            "activity_sigma_bps": float(self.last_activity_sigma_bps),
            "touch_dist_buy": float(self.last_touch_dist_buy),
            "touch_dist_sell": float(self.last_touch_dist_sell),
            "touch_event_boost_buy": float(self.last_touch_event_boost_buy),
            "touch_event_boost_sell": float(self.last_touch_event_boost_sell),
            "resting_quality_buy": float(self.last_resting_quality_buy),
            "resting_quality_sell": float(self.last_resting_quality_sell),
            "cross_confirmation_buy": float(self.last_cross_confirmation_buy),
            "cross_confirmation_sell": float(self.last_cross_confirmation_sell),
            "fill_interaction_buy": float(self.last_fill_interaction_buy),
            "fill_interaction_sell": float(self.last_fill_interaction_sell),
            "raw_spread_px": float(self.last_raw_spread_px),
            "norm_spread_px": float(self.last_norm_spread_px),
            "used_norm_spread_floor": float(self.last_used_norm_spread_floor),
            "taker_buy_clipped": float(self.last_taker_buy_clipped),
            "taker_sell_clipped": float(self.last_taker_sell_clipped),
        }
        return next_obs, float(reward_train), done, info

    def step_canonical_action_array(
        self, action_arr: np.ndarray, *, emit_info: bool = False
    ) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, float]]]:
        if not isinstance(action_arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray action_arr, got {type(action_arr)!r}")
        if action_arr.dtype != np.float32:
            action_arr = action_arr.astype(np.float32, copy=False)
        require(
            action_arr.shape == (4,),
            f"Expected action shape {(4,)}, got shape={action_arr.shape}",
        )
        if not np.all(np.isfinite(action_arr)):
            raise ValueError(f"Action components must be finite, got {action_arr}")
        center_control = float(action_arr[0])
        width_control = float(action_arr[1])
        skew_control = float(action_arr[2])
        taker_signal = float(action_arr[3])
        return self._step_from_action_components(
            center_control,
            width_control,
            skew_control,
            taker_signal,
            emit_info=emit_info,
        )

    def step(self, action: Any, emit_info: bool = False) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, float]]]:
        center_control, width_control, skew_control, taker_signal = self._parse_action(action)
        return self._step_from_action_components(
            center_control,
            width_control,
            skew_control,
            taker_signal,
            emit_info=emit_info,
        )

    def step_hard_diagnostic(
        self,
        action: Any,
        emit_info: bool = False,
    ) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, float]]]:
        center_control, width_control, skew_control, taker_signal = self._parse_action(action)
        return self._step_from_action_components_hard_diagnostic(
            center_control,
            width_control,
            skew_control,
            taker_signal,
            emit_info=emit_info,
        )


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
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (128, 128), action_dim: int = 4):
        super().__init__()
        self.mlp_mean = MLP(input_dim, hidden_dims, action_dim)
        self.linear_skip = nn.Linear(input_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_mean(x) + self.linear_skip(x)


class MarketPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        policy_hidden: Iterable[int] = (128, 128),
        value_hidden: Iterable[int] = (128, 128),
        action_dim: int = 4,
        init_log_std: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.policy_net = MarketPolicyNet(input_dim, hidden_dims=policy_hidden, action_dim=action_dim)
        self.value_net = MLP(input_dim, value_hidden, 1)
        if init_log_std is None:
            init_log_std_vec = torch.full((action_dim,), 0.0, dtype=torch.float32)
        else:
            init_log_std_vec = torch.as_tensor(init_log_std, dtype=torch.float32).reshape(-1)
            if init_log_std_vec.numel() != action_dim:
                raise ValueError(
                    f"init_log_std must have length {action_dim}, got {init_log_std_vec.numel()}"
                )
        self.log_std = nn.Parameter(init_log_std_vec.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)
        log_std = torch.clamp(self.log_std, min=-6.0, max=2.0)
        return mean, log_std, value


def _find_final_policy_linear_layer(model: MarketPolicyValueNet) -> nn.Linear:
    final_policy_linear: Optional[nn.Linear] = None
    for module in model.policy_net.mlp_mean.net:
        if isinstance(module, nn.Linear):
            final_policy_linear = module
    if final_policy_linear is None:
        raise RuntimeError("Could not find final policy linear layer")
    return final_policy_linear


def _init_market_policy_mean_head(model: MarketPolicyValueNet, env: MarketMakingEnv) -> None:
    width_action_init = _env_float("BYBIT_MM_PPO_WIDTH_ACTION_INIT", 0.08)
    if not np.isfinite(width_action_init) or not (0.0 < width_action_init < 1.0):
        raise ValueError(
            "BYBIT_MM_PPO_WIDTH_ACTION_INIT must be finite and satisfy 0.0 < value < 1.0, "
            f"got {width_action_init}"
        )
    width_latent_bias = float(np.arctanh(2.0 * width_action_init - 1.0))
    with torch.no_grad():
        for module in model.policy_net.mlp_mean.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.zero_()
        model.policy_net.linear_skip.weight.zero_()
        if model.policy_net.linear_skip.bias is not None:
            model.policy_net.linear_skip.bias.zero_()
    final_policy_linear = _find_final_policy_linear_layer(model)
    with torch.no_grad():
        if final_policy_linear.bias is not None:
            final_policy_linear.bias[0] = 0.0
            final_policy_linear.bias[1] = float(width_latent_bias)
            final_policy_linear.bias[2] = 0.0
            final_policy_linear.bias[3] = 0.0
        feature_layout = env._feature_layout
        skip_weight = model.policy_net.linear_skip.weight
        skip_bias = model.policy_net.linear_skip.bias
        dir_slice = feature_layout["dir_logits"]
        p_up_slice = feature_layout["p_up"]
        short_idx = 0
        mid_idx = env._num_h // 2
        long_idx = env._num_h - 1
        skip_weight[2, dir_slice.start + short_idx] = 0.04
        skip_weight[2, dir_slice.start + mid_idx] = 0.08
        skip_weight[2, dir_slice.start + long_idx] = 0.22
        skip_weight[2, p_up_slice.start + short_idx] = 0.02
        skip_weight[2, p_up_slice.start + mid_idx] = 0.04
        skip_weight[2, p_up_slice.start + long_idx] = 0.08
        skip_bias[2] = -0.07
        skip_weight[0, env._obs_extra_slice.start + 0] = -0.25
        skip_bias[0] = 0.0


@dataclass
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.99
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 8
    batch_size: int = 65536
    entropy_coef: float = 0.0075
    value_coef: float = 0.5
    policy_hidden: Tuple[int, ...] = (128, 128)
    value_hidden: Tuple[int, ...] = (128, 128)
    val_every: int = 10
    max_drawdown_guard: Optional[float] = None
    rollout_horizon: int = 8192
    rollouts_per_epoch: int = 32
    randomize_rollout_start: bool = True
    init_log_std_center: float = -0.20
    init_log_std_width: float = -1.00
    init_log_std_skew: float = 0.00
    init_log_std_taker: float = -1.0


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
    horizon: int = 8192,
    rollouts_per_epoch: int = 16,
    randomize_start: bool = True,
    rollout_storage: str = "gpu",
    pin_memory: bool = True,
    non_blocking: bool = True,
    start_sampler: Optional[Dict[str, Any]] = None,
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
    start_idx_buf = torch.empty((max_steps,), dtype=torch.int64, **alloc_kwargs)
    focus_logit_abs_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    reward_true_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    reward_true_future_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    reward_future_bonus_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    reward_train_econ_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    reward_shape_skew_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    reward_shape_total_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    training_reward_actual_horizon_ms_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    mid_future_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    cmssl_alpha_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    touch_dist_buy_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    touch_dist_sell_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    maker_buy_fill_frac_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    maker_sell_fill_frac_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    activity_score_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    activity_sigma_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    touch_event_boost_buy_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    touch_event_boost_sell_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    resting_quality_buy_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    resting_quality_sell_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    cross_confirmation_buy_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    cross_confirmation_sell_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    fill_interaction_buy_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    fill_interaction_sell_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    raw_spread_px_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    norm_spread_px_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    used_norm_spread_floor_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    width_control_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    width_mult_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    base_half_spread_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    half_spread_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    center_control_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    center_shift_scale_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    center_shift_min_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    center_shift_max_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    center_shift_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    inventory_center_shift_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    alpha_center_shift_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    skew_control_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    skew_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    directional_center_response_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    directional_asym_response_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    directional_response_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    effective_alpha_center_capacity_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    bid_at_floor_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    ask_at_floor_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    weighted_cmssl_logit_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    observed_spread_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    observed_anchor_half_spread_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    fill_norm_spread_floor_full_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    quote_half_spread_floor_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    bid_half_spread_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    ask_half_spread_bps_buf = torch.empty((max_steps,), dtype=torch.float32, **alloc_kwargs)
    cursor = 0

    action_dim = _resolve_market_action_dim()
    action_cpu_buf = np.empty((action_dim,), dtype=np.float32)
    action_cpu_stage_t: Optional[torch.Tensor] = None
    if target_device.type == "cuda":
        stage_kwargs: Dict[str, Any] = {}
        if use_pinned:
            stage_kwargs["pin_memory"] = True
        action_cpu_stage_t = torch.empty((action_dim,), dtype=torch.float32, device="cpu", **stage_kwargs)
    action_low, action_high = _ppo_action_bounds(
        env,
        target_device,
    )

    max_start = max(0, env.n - 2)
    sampler_starts = None if start_sampler is None else start_sampler.get("candidate_starts")
    sampler_mass = None if start_sampler is None else start_sampler.get("mixed_mass")
    sampler_abs_focus = None if start_sampler is None else start_sampler.get("abs_focus_logit")
    sampler_start_exclusion_window = 0
    available_mask: Optional[np.ndarray] = None
    sampler_reset_count = 0
    sampler_reset_warned = False
    if sampler_starts is not None and sampler_mass is not None:
        sampler_start_exclusion_window = int(
            max(
                0,
                int(
                    start_sampler.get(
                        "start_exclusion_window",
                        getattr(start_sampler.get("config"), "start_exclusion_window", horizon),
                    )
                ),
            )
        )
        available_mask = np.ones(sampler_starts.shape[0], dtype=bool)
    rollout_start_indices: List[int] = []
    obs_device_buf: Optional[torch.Tensor] = None
    for _ in range(rollouts_per_epoch):
        if randomize_start:
            if sampler_starts is not None and sampler_mass is not None:
                require(available_mask is not None, "start-sampler availability mask not initialized")
                if not bool(np.any(available_mask)):
                    available_mask[:] = True
                    sampler_reset_count += 1
                    if not sampler_reset_warned:
                        print(
                            "[mm rollout sampler] "
                            "availability_exhausted=true action=reset_available_mask"
                        )
                        sampler_reset_warned = True
                available_slots = np.flatnonzero(available_mask)
                current_mass = np.asarray(sampler_mass[available_slots], dtype=np.float64)
                mass_total = float(np.sum(current_mass))
                if mass_total <= 0.0:
                    current_mass = np.full(
                        available_slots.shape[0],
                        1.0 / float(max(1, available_slots.shape[0])),
                        dtype=np.float64,
                    )
                else:
                    current_mass = current_mass / mass_total
                sampled_slot = int(np.random.choice(available_slots, p=current_mass))
                start_idx = int(sampler_starts[sampled_slot])
                sampled_focus_abs = float(sampler_abs_focus[sampled_slot]) if sampler_abs_focus is not None else 0.0
                lower = start_idx - sampler_start_exclusion_window
                upper = start_idx + sampler_start_exclusion_window
                available_mask[(sampler_starts >= lower) & (sampler_starts <= upper)] = False
            else:
                start_idx = int(np.random.randint(0, max_start + 1))
                sampled_focus_abs = 0.0
        else:
            start_idx = 0
            sampled_focus_abs = 0.0
        rollout_start_indices.append(start_idx)
        obs = env.reset(start_idx=start_idx)
        done = False
        steps = 0
        while not done and steps < horizon:
            # 1) Read obs
            # 2) Create zero-copy CPU tensor view
            obs_cpu = torch.from_numpy(obs)
            if obs_buf is None:
                obs_dim = int(obs_cpu.shape[0])
                obs_buf = torch.empty((max_steps, obs_dim), dtype=torch.float32, **alloc_kwargs)
                next_obs_buf = torch.empty((max_steps, obs_dim), dtype=torch.float32, **alloc_kwargs)
                actions_buf = torch.empty((max_steps, action_dim), dtype=torch.float32, **alloc_kwargs)
                if target_device.type == "cuda":
                    # Reusable single-row GPU inference buffer to avoid
                    # per-step tensor allocation/churn for model inputs.
                    obs_device_buf = torch.empty((1, obs_dim), dtype=torch.float32, device=target_device)
            require(
                obs_buf is not None and next_obs_buf is not None and actions_buf is not None,
                "rollout buffers not initialized",
            )
            idx = cursor
            cursor += 1
            if target_device.type == "cuda":
                require(obs_device_buf is not None, "obs_device_buf not initialized")
                # 3) Copy CPU view into model input buffer
                obs_device_buf[0].copy_(obs_cpu, non_blocking=non_blocking)
                obs_t = obs_device_buf[0]
            else:
                obs_t = obs_cpu.to(target_device, non_blocking=non_blocking)
            # 3) Copy CPU view into rollout storage buffer
            if use_gpu_storage:
                obs_buf[idx].copy_(obs_t, non_blocking=non_blocking)
            else:
                obs_buf[idx].copy_(obs_cpu)
            with torch.no_grad():
                mean, log_std, value = model(obs_t.unsqueeze(0))
                action_env, logp_env, _latent_action = _sample_bounded_ppo_action(
                    mean,
                    log_std,
                    action_low,
                    action_high,
                )

            sampled_action_env = action_env.squeeze(0).detach()
            if sampled_action_env.device.type == "cpu":
                np.copyto(action_cpu_buf, sampled_action_env.numpy(), casting="no")
            else:
                require(action_cpu_stage_t is not None, "action_cpu_stage_t not initialized")
                action_cpu_stage_t.copy_(sampled_action_env, non_blocking=non_blocking)
                np.copyto(action_cpu_buf, action_cpu_stage_t.numpy(), casting="no")
            # 4) Step env only after obs has been ingested into model/storage buffers.
            next_obs, reward, env_done, info = env.step_canonical_action_array(action_cpu_buf, emit_info=True)
            steps += 1
            terminated = bool(env_done)
            # Truncation means the rollout horizon ended; it is not a true
            # environment terminal state and should continue to bootstrap.
            truncated = (not terminated) and (steps >= horizon)
            done = terminated or truncated

            next_obs_cpu = torch.from_numpy(next_obs)
            if use_gpu_storage:
                next_obs_buf[idx].copy_(next_obs_cpu, non_blocking=non_blocking)
                actions_buf[idx].copy_(action_env.squeeze(0).detach())
                logp_buf[idx] = logp_env.squeeze(0).detach()
                values_buf[idx] = value.squeeze(0).detach()
            else:
                next_obs_buf[idx].copy_(next_obs_cpu)
                actions_buf[idx].copy_(action_env.squeeze(0).detach().cpu())
                logp_buf[idx] = logp_env.squeeze(0).detach().cpu()
                values_buf[idx] = value.squeeze(0).detach().cpu()
            rewards_buf[idx] = float(reward)
            info = info or {}
            reward_true_buf[idx] = float(info.get("reward_true", reward))
            reward_true_future_buf[idx] = float(info.get("reward_true_future", info.get("reward_true", reward)))
            reward_future_bonus_buf[idx] = float(info.get("reward_future_bonus", 0.0))
            reward_train_econ_buf[idx] = float(info.get("reward_train_econ", info.get("reward_true", reward)))
            reward_shape_skew_buf[idx] = float(info.get("reward_shape_skew", 0.0))
            reward_shape_total_buf[idx] = float(info.get("reward_shape_total", 0.0))
            training_reward_actual_horizon_ms_buf[idx] = float(info.get("training_reward_actual_horizon_ms", 0.0))
            mid_future_buf[idx] = float(info.get("mid_future", 0.0))
            cmssl_alpha_buf[idx] = float(info.get("cmssl_alpha", 0.0))
            touch_dist_buy_buf[idx] = float(info.get("touch_dist_buy", 0.0))
            touch_dist_sell_buf[idx] = float(info.get("touch_dist_sell", 0.0))
            maker_buy_fill_frac_buf[idx] = float(info.get("maker_buy_fill_frac", 0.0))
            maker_sell_fill_frac_buf[idx] = float(info.get("maker_sell_fill_frac", 0.0))
            activity_score_buf[idx] = float(info.get("activity_score", 0.0))
            activity_sigma_bps_buf[idx] = float(info.get("activity_sigma_bps", 0.0))
            touch_event_boost_buy_buf[idx] = float(info.get("touch_event_boost_buy", 0.0))
            touch_event_boost_sell_buf[idx] = float(info.get("touch_event_boost_sell", 0.0))
            resting_quality_buy_buf[idx] = float(info.get("resting_quality_buy", 0.0))
            resting_quality_sell_buf[idx] = float(info.get("resting_quality_sell", 0.0))
            cross_confirmation_buy_buf[idx] = float(info.get("cross_confirmation_buy", 0.0))
            cross_confirmation_sell_buf[idx] = float(info.get("cross_confirmation_sell", 0.0))
            fill_interaction_buy_buf[idx] = float(info.get("fill_interaction_buy", 0.0))
            fill_interaction_sell_buf[idx] = float(info.get("fill_interaction_sell", 0.0))
            raw_spread_px_buf[idx] = float(info.get("raw_spread_px", 0.0))
            norm_spread_px_buf[idx] = float(info.get("norm_spread_px", 0.0))
            used_norm_spread_floor_buf[idx] = float(info.get("used_norm_spread_floor", 0.0))
            width_control_buf[idx] = float(info.get("width_control", 0.0))
            width_mult_buf[idx] = float(info.get("width_mult", 0.0))
            base_half_spread_bps_buf[idx] = float(info.get("base_half_spread_bps", 0.0))
            half_spread_bps_buf[idx] = float(info.get("half_spread_bps", 0.0))
            center_control_buf[idx] = float(info.get("center_control", 0.0))
            center_shift_scale_bps_buf[idx] = float(info.get("center_shift_scale_bps", 0.0))
            center_shift_min_bps_buf[idx] = float(info.get("center_shift_min_bps", 0.0))
            center_shift_max_bps_buf[idx] = float(info.get("center_shift_max_bps", 0.0))
            center_shift_bps_buf[idx] = float(info.get("center_shift_bps", 0.0))
            inventory_center_shift_bps_buf[idx] = float(info.get("inventory_center_shift_bps", 0.0))
            alpha_center_shift_bps_buf[idx] = float(info.get("alpha_center_shift_bps", 0.0))
            skew_control_buf[idx] = float(info.get("skew_control", 0.0))
            skew_bps_buf[idx] = float(info.get("skew_bps", 0.0))
            directional_center_response_buf[idx] = float(info.get("directional_center_response", 0.0))
            directional_asym_response_buf[idx] = float(info.get("directional_asym_response", 0.0))
            directional_response_buf[idx] = float(info.get("directional_response", 0.0))
            effective_alpha_center_capacity_bps_buf[idx] = float(info.get("effective_alpha_center_capacity_bps", 0.0))
            bid_at_floor_buf[idx] = float(info.get("bid_at_floor", 0.0))
            ask_at_floor_buf[idx] = float(info.get("ask_at_floor", 0.0))
            weighted_cmssl_logit_buf[idx] = float(info.get("weighted_cmssl_logit", 0.0))
            observed_spread_bps_buf[idx] = float(info.get("observed_spread_bps", 0.0))
            observed_anchor_half_spread_bps_buf[idx] = float(info.get("observed_anchor_half_spread_bps", 0.0))
            fill_norm_spread_floor_full_bps_buf[idx] = float(info.get("fill_norm_spread_floor_full_bps", 0.0))
            quote_half_spread_floor_bps_buf[idx] = float(info.get("quote_half_spread_floor_bps", 0.0))
            bid_half_spread_bps_buf[idx] = float(info.get("bid_half_spread_bps", 0.0))
            ask_half_spread_bps_buf[idx] = float(info.get("ask_half_spread_bps", 0.0))
            terminated_buf[idx] = float(terminated)
            truncated_buf[idx] = float(truncated)
            dones_buf[idx] = float(done)
            start_idx_buf[idx] = int(start_idx)
            focus_logit_abs_buf[idx] = float(sampled_focus_abs)
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
        "start_idx": start_idx_buf[:cursor],
        "focus_logit_abs": focus_logit_abs_buf[:cursor],
        "reward_true": reward_true_buf[:cursor],
        "reward_true_future": reward_true_future_buf[:cursor],
        "reward_future_bonus": reward_future_bonus_buf[:cursor],
        "reward_train_econ": reward_train_econ_buf[:cursor],
        "reward_shape_skew": reward_shape_skew_buf[:cursor],
        "reward_shape_total": reward_shape_total_buf[:cursor],
        "training_reward_actual_horizon_ms": training_reward_actual_horizon_ms_buf[:cursor],
        "mid_future": mid_future_buf[:cursor],
        "cmssl_alpha": cmssl_alpha_buf[:cursor],
        "touch_dist_buy": touch_dist_buy_buf[:cursor],
        "touch_dist_sell": touch_dist_sell_buf[:cursor],
        "maker_buy_fill_frac": maker_buy_fill_frac_buf[:cursor],
        "maker_sell_fill_frac": maker_sell_fill_frac_buf[:cursor],
        "activity_score": activity_score_buf[:cursor],
        "activity_sigma_bps": activity_sigma_bps_buf[:cursor],
        "touch_event_boost_buy": touch_event_boost_buy_buf[:cursor],
        "touch_event_boost_sell": touch_event_boost_sell_buf[:cursor],
        "resting_quality_buy": resting_quality_buy_buf[:cursor],
        "resting_quality_sell": resting_quality_sell_buf[:cursor],
        "cross_confirmation_buy": cross_confirmation_buy_buf[:cursor],
        "cross_confirmation_sell": cross_confirmation_sell_buf[:cursor],
        "fill_interaction_buy": fill_interaction_buy_buf[:cursor],
        "fill_interaction_sell": fill_interaction_sell_buf[:cursor],
        "raw_spread_px": raw_spread_px_buf[:cursor],
        "norm_spread_px": norm_spread_px_buf[:cursor],
        "used_norm_spread_floor": used_norm_spread_floor_buf[:cursor],
        "width_control": width_control_buf[:cursor],
        "width_mult": width_mult_buf[:cursor],
        "base_half_spread_bps": base_half_spread_bps_buf[:cursor],
        "half_spread_bps": half_spread_bps_buf[:cursor],
        "center_control": center_control_buf[:cursor],
        "center_shift_scale_bps": center_shift_scale_bps_buf[:cursor],
        "center_shift_min_bps": center_shift_min_bps_buf[:cursor],
        "center_shift_max_bps": center_shift_max_bps_buf[:cursor],
        "center_shift_bps": center_shift_bps_buf[:cursor],
        "inventory_center_shift_bps": inventory_center_shift_bps_buf[:cursor],
        "alpha_center_shift_bps": alpha_center_shift_bps_buf[:cursor],
        "skew_control": skew_control_buf[:cursor],
        "skew_bps": skew_bps_buf[:cursor],
        "directional_center_response": directional_center_response_buf[:cursor],
        "directional_asym_response": directional_asym_response_buf[:cursor],
        "directional_response": directional_response_buf[:cursor],
        "effective_alpha_center_capacity_bps": effective_alpha_center_capacity_bps_buf[:cursor],
        "bid_at_floor": bid_at_floor_buf[:cursor],
        "ask_at_floor": ask_at_floor_buf[:cursor],
        "weighted_cmssl_logit": weighted_cmssl_logit_buf[:cursor],
        "observed_spread_bps": observed_spread_bps_buf[:cursor],
        "observed_anchor_half_spread_bps": observed_anchor_half_spread_bps_buf[:cursor],
        "fill_norm_spread_floor_full_bps": fill_norm_spread_floor_full_bps_buf[:cursor],
        "quote_half_spread_floor_bps": quote_half_spread_floor_bps_buf[:cursor],
        "bid_half_spread_bps": bid_half_spread_bps_buf[:cursor],
        "ask_half_spread_bps": ask_half_spread_bps_buf[:cursor],
        "rollout_start_indices": np.asarray(rollout_start_indices, dtype=np.int64),
        "sampler_availability_resets": int(sampler_reset_count),
    }


def ppo_update_market(
    model: MarketPolicyValueNet,
    optimizer: optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: PPOConfig,
    device: str,
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
        target_device,
    )
    indices = torch.arange(n, device=obs.device)
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
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    storage = "gpu" if obs.device.type == target_device.type else "cpu"
    _timing_log(
        f"ppo_update storage={storage} on_device={same_device} manual_gaussian=true secs={time.perf_counter() - t0:.4f}"
    )
    return {}

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


def _canonical_market_action_array(
    action: np.ndarray | torch.Tensor | Sequence[float],
) -> np.ndarray:
    expected_dim = _resolve_market_action_dim()
    if isinstance(action, np.ndarray) and action.dtype == np.float32 and action.ndim == 1:
        action_arr = action
    else:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    require(
        action_arr.shape == (expected_dim,),
        f"Expected action shape {(expected_dim,)}, got shape={action_arr.shape}",
    )
    if not np.all(np.isfinite(action_arr)):
        raise ValueError(f"Action components must be finite, got {action_arr}")
    return action_arr


def _market_env_action_tuple(action: np.ndarray | torch.Tensor | Sequence[float]) -> Tuple[float, float, float, float]:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    require(
        action_arr.shape == (_resolve_market_action_dim(),),
        f"Expected canonical action shape {(4,)}, got shape={action_arr.shape}",
    )
    return float(action_arr[0]), float(action_arr[1]), float(action_arr[2]), float(action_arr[3])


def evaluate_market_policy_ppo(
    env: MarketMakingEnv,
    model: MarketPolicyValueNet,
    *,
    stochastic: bool,
    device: str = "cuda",
    generator: Optional[torch.Generator] = None,
    use_hard_maker_fill: bool = False,
) -> Dict[str, Any]:
    def _policy_fn(obs: np.ndarray) -> np.ndarray:
        action = _ppo_action_from_obs_numpy(
            model,
            obs,
            stochastic=stochastic,
            generator=generator,
            device=device,
            env=env,
        )
        return _canonical_market_action_array(action)

    return evaluate_market_making(env, _policy_fn, use_hard_maker_fill=use_hard_maker_fill)


def _ppo_action_from_obs_numpy(
    model: MarketPolicyValueNet,
    obs_np: np.ndarray,
    stochastic: bool,
    generator: Optional[torch.Generator] = None,
    *,
    device: str = "cuda",
    env: Optional[MarketMakingEnv] = None,
) -> np.ndarray:
    obs_t = torch.from_numpy(obs_np).to(device)
    action_low, action_high = _ppo_action_bounds(
        env,
        obs_t.device,
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
    pre_fee_pnl = net_pnl - net_fee_cost
    pre_fee_pnl_pct = pre_fee_pnl / denom

    return {
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "pre_fee_pnl": pre_fee_pnl,
        "pre_fee_pnl_pct": pre_fee_pnl_pct,
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
    """Fit train-split obs normalization from exogenous feature rows only."""
    feature_rows = np.asarray(train_env.features, dtype=np.float64)
    if feature_rows.ndim != 2 or feature_rows.shape[0] < 2:
        raise RuntimeError(
            "Train-only observation-normalization prefit requires at least two training feature rows."
        )
    obs_dim = train_env._obs_dim
    feature_dim = train_env._feature_dim
    count = int(feature_rows.shape[0])
    mean = np.zeros((obs_dim,), dtype=np.float64)
    m2 = np.zeros((obs_dim,), dtype=np.float64)
    mean[:feature_dim] = np.mean(feature_rows, axis=0)
    centered = feature_rows - mean[:feature_dim]
    m2[:feature_dim] = np.sum(centered * centered, axis=0)
    mask = train_env._continuous_mask(obs_dim).astype(bool, copy=False)
    if mean.shape != (obs_dim,) or m2.shape != (obs_dim,):
        raise RuntimeError(
            f"Prefitted obs normalization shape mismatch: mean={mean.shape} m2={m2.shape} obs_dim={obs_dim}"
        )
    if mask.shape != mean.shape:
        raise RuntimeError(
            "Prefitted obs normalization continuous_mask shape mismatch: "
            f"mask={mask.shape} mean={mean.shape}"
        )
    if bool(np.any(mask[feature_dim:])):
        raise RuntimeError("Execution-state extras must be excluded from observation z-scoring.")
    return {
        "count": count,
        "mean": mean.tolist(),
        "m2": m2.tolist(),
        "continuous_mask": mask.tolist(),
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
    first_obs = env.reset(start_idx=int(probe_indices[0]))
    probe_obs_batch = np.empty((probe_count, first_obs.shape[0]), dtype=first_obs.dtype)
    probe_obs_batch[0] = first_obs
    for row_idx, idx in enumerate(probe_indices[1:], start=1):
        probe_obs_batch[row_idx] = env.reset(start_idx=int(idx))
    return torch.as_tensor(probe_obs_batch, device=device)


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
        "checkpoint_schema": MM_PPO_CHECKPOINT_SCHEMA,
        "model_state_dict": _market_ppo_model_state_dict_for_ckpt(model),
        "policy_hidden_dims": tuple(int(x) for x in policy_hidden_dims),
        "value_hidden_dims": tuple(int(x) for x in value_hidden_dims),
        "obs_dim": int(obs_dim),
        "action_dim": MM_PPO_ACTION_DIM,
        "action_semantics": list(MM_PPO_ACTION_SEMANTICS),
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


def _market_ppo_model_state_dict_for_ckpt(model: MarketPolicyValueNet) -> Dict[str, torch.Tensor]:
    source_module = model._orig_mod if hasattr(model, "_orig_mod") else model
    return dict(source_module.state_dict())


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
    if action_dim != _resolve_market_action_dim():
        raise ValueError(canonical_error)
    return action_dim


def _canonical_market_ppo_schema(ckpt: Dict[str, Any]) -> str:
    schema = ckpt.get("checkpoint_schema")
    if schema != MM_PPO_CHECKPOINT_SCHEMA:
        legacy_note = (
            " Direct-quote v5/v6 checkpoints are incompatible with v7 because policy architecture and quote-geometry semantics "
            "changed to MLP + linear skip and mean-head warm-start semantics changed; retraining is required."
            if schema in {"mm-ppo-direct-quote-v5", "mm-ppo-direct-quote-v6"}
            else ""
        )
        raise ValueError(
            "Unsupported market PPO checkpoint schema. "
            f"Expected checkpoint_schema={MM_PPO_CHECKPOINT_SCHEMA!r}, got {schema!r}. "
            f"{legacy_note}".strip()
        )
    return str(schema)


def _canonical_market_ppo_action_semantics(ckpt: Dict[str, Any]) -> Tuple[str, ...]:
    semantics = ckpt.get("action_semantics")
    canonical_error = (
        "Unsupported market PPO checkpoint action semantics. "
        f"Expected {list(MM_PPO_ACTION_SEMANTICS)!r}; retrain or re-export a direct quote policy PPO checkpoint with center/width/skew/taker controls."
    )
    if isinstance(semantics, tuple):
        semantics = list(semantics)
    if not isinstance(semantics, list):
        raise ValueError(canonical_error)
    parsed = tuple(str(x) for x in semantics)
    if parsed != MM_PPO_ACTION_SEMANTICS:
        raise ValueError(canonical_error)
    return parsed


def load_market_ppo_model(
    input_dim: int,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
    checkpoint_data: Optional[Any] = None,
) -> Optional[MarketPolicyValueNet]:
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Market PPO checkpoint not found: {ckpt_path}")
    ckpt = (
        checkpoint_data
        if checkpoint_data is not None
        else _torch_load_trusted_checkpoint(path, map_location=device)
    )
    if not isinstance(ckpt, dict):
        raise ValueError(
            "Unsupported PPO checkpoint payload type; expected a mapping for market PPO loading."
        )
    _canonical_market_ppo_schema(ckpt)

    state = ckpt.get("model_state_dict")
    canonical_metadata_fields = ("policy_hidden_dims", "value_hidden_dims", "action_dim", "action_semantics")
    has_any_canonical_metadata = any(field in ckpt for field in canonical_metadata_fields)
    if not isinstance(state, dict):
        if not has_any_canonical_metadata:
            raise ValueError(
                "Unsupported RL checkpoint format. Only canonical full PPO checkpoints are supported. "
                "Re-export or retrain under the PPO checkpoint format with model_state_dict, "
                "policy_hidden_dims, value_hidden_dims, action_dim, action_semantics, and checkpoint_schema."
            )
        raise ValueError(
            "Malformed canonical market PPO checkpoint: model_state_dict is missing or not a mapping."
        )

    policy_hidden_dims = _canonical_market_ppo_arch_field(ckpt, "policy_hidden_dims")
    value_hidden_dims = _canonical_market_ppo_arch_field(ckpt, "value_hidden_dims")
    action_dim = _canonical_market_ppo_action_dim(ckpt)
    _canonical_market_ppo_action_semantics(ckpt)

    model = MarketPolicyValueNet(
        input_dim,
        policy_hidden=policy_hidden_dims,
        value_hidden=value_hidden_dims,
        action_dim=action_dim,
    ).to(device)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Malformed canonical market PPO checkpoint: model_state_dict does not match "
            "the current canonical MarketPolicyValueNet parameter layout. "
            "Only canonical PPO checkpoints saved from the current codepath are supported; "
            "checkpoints with _orig_mod.-prefixed keys or other non-canonical naming are unsupported. "
            "Re-save from the current codepath or retrain."
        ) from exc

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
    epochs: int = 50,
    config: Optional[PPOConfig] = None,
    ckpt_path: Optional[Path] = None,
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
    action_dim = _resolve_market_action_dim()
    model = MarketPolicyValueNet(
        input_dim,
        policy_hidden=config.policy_hidden,
        value_hidden=config.value_hidden,
        action_dim=action_dim,
        init_log_std=(
            config.init_log_std_center,
            config.init_log_std_width,
            config.init_log_std_skew,
            config.init_log_std_taker,
        ),
    ).to(device)
    _init_market_policy_mean_head(model, train_env)
    mean_head_width_action_init = _env_float("BYBIT_MM_PPO_WIDTH_ACTION_INIT", 0.08)
    if not np.isfinite(mean_head_width_action_init) or not (0.0 < mean_head_width_action_init < 1.0):
        raise ValueError(
            "BYBIT_MM_PPO_WIDTH_ACTION_INIT must be finite and satisfy 0.0 < value < 1.0, "
            f"got {mean_head_width_action_init}"
        )
    mean_head_width_bias_latent = float(np.arctanh(2.0 * mean_head_width_action_init - 1.0))
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
        f"gamma={config.gamma:.6f} "
        f"gae_lambda={config.gae_lambda:.6f} "
        f"entropy_coef={config.entropy_coef:.6f} "
        f"update_epochs={config.update_epochs} "
        f"batch_size={config.batch_size} "
        f"steps_per_epoch={config.rollout_horizon * config.rollouts_per_epoch} "
        "mean_head_init=small_normal_mlp_plus_cmssl_skip_warm_start "
        f"width_action_init={mean_head_width_action_init:.2f} "
        "width_action_init_source=BYBIT_MM_PPO_WIDTH_ACTION_INIT(default=0.08) "
        f"width_bias_latent={mean_head_width_bias_latent:.6f} "
        f"inventory_center_weight={train_env.direct_quote_config.inventory_center_weight:.2f} "
        f"alpha_center_weight={train_env.direct_quote_config.alpha_center_weight:.2f} "
        f"asymmetry_residual_frac={train_env.direct_quote_config.asymmetry_residual_frac:.2f} "
        f"directional_response_center_weight={train_env.direct_quote_config.directional_response_center_weight:.2f} "
        f"directional_response_asym_weight={train_env.direct_quote_config.directional_response_asym_weight:.2f} "
        f"quote_half_spread_floor_bps={train_env.direct_quote_config.quote_half_spread_floor_bps:.4f} "
        "action_controls=center/width/skew/taker controls "
        f"init_log_std_center={config.init_log_std_center:.4f} "
        f"init_log_std_width={config.init_log_std_width:.4f} "
        f"init_log_std_skew={config.init_log_std_skew:.4f} "
        f"init_log_std_taker={config.init_log_std_taker:.4f}"
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
    prefitted_obs_count = int(train_obs_norm_state["count"])
    train_mask = np.asarray(train_obs_norm_state.get("continuous_mask"), dtype=bool)
    require(
        train_mask.shape == (input_dim,),
        f"Train obs normalization mask shape mismatch: {train_mask.shape} vs {(input_dim,)}",
    )
    require(
        not bool(np.any(train_mask[-ENV_OBS_EXTRA_STATE_DIM:])),
        "Execution-state extras must be excluded from z-scoring (continuous_mask tail must be all False).",
    )
    print(
        "[mm ppo obs norm] "
        f"count={prefitted_obs_count} "
        f"train_frozen={train_env.freeze_obs_norm} "
        f"val_frozen={val_env.freeze_obs_norm} "
        f"feature_mask_true={int(np.sum(train_mask[:-ENV_OBS_EXTRA_STATE_DIM]))} "
        f"extras_normalized={bool(np.any(train_mask[-ENV_OBS_EXTRA_STATE_DIM:]))}"
    )
    assert prefitted_obs_count >= 2, "prefitted observation normalization must have count >= 2"
    probe_obs = _build_market_probe_obs_batch(val_env, batch_size=8, device=device)
    action_low, action_high = _ppo_action_bounds(
        train_env,
        device,
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
    probe_action_labels = {
        "center_control": float(bounded_probe_action[0]),
        "width_control": float(bounded_probe_action[1]),
        "skew_control": float(bounded_probe_action[2]),
        "taker_signal": float(bounded_probe_action[3]),
    }
    print(
        "[mm ppo bounds] "
        f"action_dim={action_dim} "
        f"low={np.array2string(bounds_low_np, precision=4, floatmode='fixed')} "
        f"high={np.array2string(bounds_high_np, precision=4, floatmode='fixed')} "
        f"env_quote_half_spread_floor_bps={train_env.quote_half_spread_floor_bps:.6f} "
        f"taker_signal_limit={train_env.taker_signal_limit:.4f} "
        f"mean_probe_action={np.array2string(bounded_probe_action, precision=4, floatmode='fixed')} "
        f"mean_probe_action_labeled={probe_action_labels} "
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
    start_sampling_cfg = load_rollout_start_sampling_config(rollout_horizon=config.rollout_horizon)
    start_sampler = _build_rollout_start_sampler(
        train_env,
        start_sampling_cfg,
        rollout_horizon=config.rollout_horizon,
    )
    if start_sampler is None:
        print(
            "[mm rollout sampler] "
            f"enabled={start_sampling_cfg.enabled} "
            "active=false reason=disabled_or_insufficient_candidates"
        )
    else:
        score_arr = np.asarray(start_sampler["weighted_score"], dtype=np.float64)
        print(
            "[mm rollout sampler] "
            f"enabled={start_sampling_cfg.enabled} "
            f"active=true "
            f"horizon_logit_weights={train_env.directional_signal_config.horizon_logit_weights} "
            f"weighted_mix={start_sampling_cfg.weighted_mix:.4f} "
            f"score_power={start_sampling_cfg.score_power:.4f} "
            f"score_epsilon={start_sampling_cfg.score_epsilon:.6g} "
            f"lead_steps={start_sampling_cfg.lead_steps} "
            f"start_exclusion_window={int(start_sampler['start_exclusion_window'])} "
            f"candidate_count={int(start_sampler['candidate_starts'].shape[0])} "
            f"effective_max_start={int(start_sampler['effective_max_start'])} "
            f"min_remaining_steps={int(start_sampler['min_remaining_steps'])} "
            f"score_min={float(np.min(score_arr)):.6g} "
            f"score_max={float(np.max(score_arr)):.6g} "
            f"score_mean={float(np.mean(score_arr)):.6g} "
            f"top_focus={start_sampler['top_focus']}"
        )

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        obs_count_before_rollout = int(train_env.get_obs_norm_state()["count"])
        rollout = collect_market_rollout(
            train_env,
            model,
            device,
            horizon=config.rollout_horizon,
            rollouts_per_epoch=config.rollouts_per_epoch,
            randomize_start=config.randomize_rollout_start,
            rollout_storage=rollout_storage,
            pin_memory=pin_rollout_memory,
            non_blocking=non_blocking_transfers,
            start_sampler=start_sampler if config.randomize_rollout_start else None,
        )
        ppo_update_market(
            model,
            optimizer,
            rollout,
            config,
            device,
            non_blocking=non_blocking_transfers,
            env=train_env,
        )
        obs_count_after_rollout = int(train_env.get_obs_norm_state()["count"])
        assert train_env.freeze_obs_norm is True, "train env obs normalization must stay frozen during PPO"
        assert obs_count_after_rollout == obs_count_before_rollout, (
            "train env obs normalization count drifted during PPO despite frozen normalization"
        )
        final_policy_linear = _find_final_policy_linear_layer(model)
        policy_skip_linear = model.policy_net.linear_skip
        with torch.no_grad():
            probe_mean = model.policy_net(probe_obs)
            bounded_probe_actions = _bounded_ppo_mean_action(probe_mean, action_low, action_high)
            probe_mean_abs = probe_mean.abs().mean(dim=0).detach().cpu().numpy()
            bounded_probe_mean_action_abs = bounded_probe_actions.abs().mean(dim=0).detach().cpu().numpy()
            action_bound_magnitude = torch.maximum(action_low.abs(), action_high.abs())
            saturation_fraction = (
                bounded_probe_actions.abs() >= (0.95 * action_bound_magnitude)
            ).float().mean(dim=0).detach().cpu().numpy()
            log_std_values = model.log_std.detach().cpu().numpy()
            policy_mlp_final_weight_l2 = float(final_policy_linear.weight.detach().norm(2).item())
            policy_skip_weight_l2 = float(policy_skip_linear.weight.detach().norm(2).item())
            policy_total_head_weight_l2 = float(
                np.sqrt(policy_mlp_final_weight_l2 ** 2 + policy_skip_weight_l2 ** 2)
            )
            policy_head_bias_l2 = 0.0
            if final_policy_linear.bias is not None:
                policy_head_bias_l2 += float(final_policy_linear.bias.detach().norm(2).item()) ** 2
            if policy_skip_linear.bias is not None:
                policy_head_bias_l2 += float(policy_skip_linear.bias.detach().norm(2).item()) ** 2
            policy_head_bias_l2 = float(np.sqrt(policy_head_bias_l2))
        print(
            "[mm ppo stats] "
            f"epoch={epoch + 1} "
            f"log_std={np.array2string(log_std_values, precision=4, floatmode='fixed')} "
            f"policy_mlp_final_weight_l2={policy_mlp_final_weight_l2:.6f} "
            f"policy_skip_weight_l2={policy_skip_weight_l2:.6f} "
            f"policy_total_head_weight_l2={policy_total_head_weight_l2:.6f} "
            f"policy_head_bias_l2={policy_head_bias_l2:.6f} "
            f"probe_mean_abs={np.array2string(probe_mean_abs, precision=6, floatmode='fixed')} "
            f"bounded_probe_mean_action_abs={np.array2string(bounded_probe_mean_action_abs, precision=6, floatmode='fixed')} "
            f"saturation_fraction={np.array2string(saturation_fraction, precision=6, floatmode='fixed')}"
        )
        start_idx_np = rollout["start_idx"].detach().cpu().numpy().astype(np.float64)
        focus_abs_np = rollout["focus_logit_abs"].detach().cpu().numpy().astype(np.float64)
        rollout_start_indices_np = np.asarray(rollout.get("rollout_start_indices", np.asarray([], dtype=np.int64)), dtype=np.int64)
        unique_starts = int(np.unique(rollout_start_indices_np).shape[0]) if rollout_start_indices_np.size else 0
        min_start_distance = (
            int(np.min(np.diff(np.sort(np.unique(rollout_start_indices_np)))))
            if unique_starts > 1
            else None
        )
        sampler_resets = int(rollout.get("sampler_availability_resets", 0))
        reward_true_np = rollout["reward_true"].detach().cpu().numpy().astype(np.float64)
        reward_true_future_np = rollout["reward_true_future"].detach().cpu().numpy().astype(np.float64)
        reward_future_bonus_np = rollout["reward_future_bonus"].detach().cpu().numpy().astype(np.float64)
        reward_train_econ_np = rollout["reward_train_econ"].detach().cpu().numpy().astype(np.float64)
        shape_skew_np = rollout["reward_shape_skew"].detach().cpu().numpy().astype(np.float64)
        shape_total_np = rollout["reward_shape_total"].detach().cpu().numpy().astype(np.float64)
        training_reward_actual_horizon_ms_np = rollout["training_reward_actual_horizon_ms"].detach().cpu().numpy().astype(np.float64)
        touch_dist_buy_np = rollout["touch_dist_buy"].detach().cpu().numpy().astype(np.float64)
        touch_dist_sell_np = rollout["touch_dist_sell"].detach().cpu().numpy().astype(np.float64)
        maker_buy_fill_frac_np = rollout["maker_buy_fill_frac"].detach().cpu().numpy().astype(np.float64)
        maker_sell_fill_frac_np = rollout["maker_sell_fill_frac"].detach().cpu().numpy().astype(np.float64)
        activity_score_np = rollout["activity_score"].detach().cpu().numpy().astype(np.float64)
        activity_sigma_bps_np = rollout["activity_sigma_bps"].detach().cpu().numpy().astype(np.float64)
        touch_event_boost_buy_np = rollout["touch_event_boost_buy"].detach().cpu().numpy().astype(np.float64)
        touch_event_boost_sell_np = rollout["touch_event_boost_sell"].detach().cpu().numpy().astype(np.float64)
        resting_quality_buy_np = rollout["resting_quality_buy"].detach().cpu().numpy().astype(np.float64)
        resting_quality_sell_np = rollout["resting_quality_sell"].detach().cpu().numpy().astype(np.float64)
        cross_confirmation_buy_np = rollout["cross_confirmation_buy"].detach().cpu().numpy().astype(np.float64)
        cross_confirmation_sell_np = rollout["cross_confirmation_sell"].detach().cpu().numpy().astype(np.float64)
        fill_interaction_buy_np = rollout["fill_interaction_buy"].detach().cpu().numpy().astype(np.float64)
        fill_interaction_sell_np = rollout["fill_interaction_sell"].detach().cpu().numpy().astype(np.float64)
        raw_spread_px_np = rollout["raw_spread_px"].detach().cpu().numpy().astype(np.float64)
        norm_spread_px_np = rollout["norm_spread_px"].detach().cpu().numpy().astype(np.float64)
        used_norm_spread_floor_np = rollout["used_norm_spread_floor"].detach().cpu().numpy().astype(np.float64)
        width_control_np = rollout["width_control"].detach().cpu().numpy().astype(np.float64)
        width_mult_np = rollout["width_mult"].detach().cpu().numpy().astype(np.float64)
        base_half_spread_bps_np = rollout["base_half_spread_bps"].detach().cpu().numpy().astype(np.float64)
        half_spread_bps_np = rollout["half_spread_bps"].detach().cpu().numpy().astype(np.float64)
        center_control_np = rollout["center_control"].detach().cpu().numpy().astype(np.float64)
        center_shift_scale_bps_np = rollout["center_shift_scale_bps"].detach().cpu().numpy().astype(np.float64)
        center_shift_min_bps_np = rollout["center_shift_min_bps"].detach().cpu().numpy().astype(np.float64)
        center_shift_max_bps_np = rollout["center_shift_max_bps"].detach().cpu().numpy().astype(np.float64)
        center_shift_bps_np = rollout["center_shift_bps"].detach().cpu().numpy().astype(np.float64)
        inventory_center_shift_bps_np = rollout["inventory_center_shift_bps"].detach().cpu().numpy().astype(np.float64)
        alpha_center_shift_bps_np = rollout["alpha_center_shift_bps"].detach().cpu().numpy().astype(np.float64)
        skew_control_np = rollout["skew_control"].detach().cpu().numpy().astype(np.float64)
        skew_bps_np = rollout["skew_bps"].detach().cpu().numpy().astype(np.float64)
        directional_center_response_np = rollout["directional_center_response"].detach().cpu().numpy().astype(np.float64)
        directional_asym_response_np = rollout["directional_asym_response"].detach().cpu().numpy().astype(np.float64)
        directional_response_np = rollout["directional_response"].detach().cpu().numpy().astype(np.float64)
        effective_alpha_center_capacity_bps_np = rollout["effective_alpha_center_capacity_bps"].detach().cpu().numpy().astype(np.float64)
        bid_at_floor_np = rollout["bid_at_floor"].detach().cpu().numpy().astype(np.float64)
        ask_at_floor_np = rollout["ask_at_floor"].detach().cpu().numpy().astype(np.float64)
        weighted_cmssl_logit_np = rollout["weighted_cmssl_logit"].detach().cpu().numpy().astype(np.float64)
        cmssl_alpha_np = rollout["cmssl_alpha"].detach().cpu().numpy().astype(np.float64)
        observed_spread_bps_np = rollout["observed_spread_bps"].detach().cpu().numpy().astype(np.float64)
        observed_anchor_half_spread_bps_np = rollout["observed_anchor_half_spread_bps"].detach().cpu().numpy().astype(np.float64)
        fill_norm_spread_floor_full_bps_np = rollout["fill_norm_spread_floor_full_bps"].detach().cpu().numpy().astype(np.float64)
        quote_half_spread_floor_bps_np = rollout["quote_half_spread_floor_bps"].detach().cpu().numpy().astype(np.float64)
        bid_half_spread_bps_np = rollout["bid_half_spread_bps"].detach().cpu().numpy().astype(np.float64)
        ask_half_spread_bps_np = rollout["ask_half_spread_bps"].detach().cpu().numpy().astype(np.float64)
        true_abs_mean = float(np.mean(np.abs(reward_train_econ_np))) if reward_train_econ_np.size else 0.0
        shape_abs_mean = float(np.mean(np.abs(shape_total_np))) if shape_total_np.size else 0.0
        shaping_ratio = shape_abs_mean / max(true_abs_mean, 1e-8)
        avg_touch_dist = float(np.mean(0.5 * (touch_dist_buy_np + touch_dist_sell_np))) if touch_dist_buy_np.size else 0.0
        p50_touch_dist = float(np.percentile(0.5 * (touch_dist_buy_np + touch_dist_sell_np), 50.0)) if touch_dist_buy_np.size else 0.0
        p90_touch_dist = float(np.percentile(0.5 * (touch_dist_buy_np + touch_dist_sell_np), 90.0)) if touch_dist_buy_np.size else 0.0
        at_touch_frac = float(np.mean((touch_dist_buy_np <= 0.10) & (touch_dist_sell_np <= 0.10))) if touch_dist_buy_np.size else 0.0
        off_touch_frac = float(np.mean((touch_dist_buy_np >= 0.50) & (touch_dist_sell_np >= 0.50))) if touch_dist_buy_np.size else 0.0
        bonus_mass = touch_event_boost_buy_np + touch_event_boost_sell_np
        smooth_mass = fill_interaction_buy_np + fill_interaction_sell_np
        bonus_frac = float(np.mean(bonus_mass / np.maximum(bonus_mass + smooth_mass, 1e-8))) if bonus_mass.size else 0.0
        fill_mean = float(np.mean(0.5 * (maker_buy_fill_frac_np + maker_sell_fill_frac_np))) if maker_buy_fill_frac_np.size else 0.0
        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            if a.size == 0 or b.size == 0 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        logit_abs = np.abs(weighted_cmssl_logit_np)
        logit_cut = float(np.median(logit_abs)) if logit_abs.size else 0.0
        high_logit_mask = logit_abs >= logit_cut
        low_logit_mask = ~high_logit_mask
        sign_agreement = float(
            np.mean(np.sign(directional_response_np) == np.sign(weighted_cmssl_logit_np))
        ) if directional_response_np.size else 0.0
        print(
            "[mm ppo sampler/shaping] "
            f"epoch={epoch + 1} "
            f"start_idx_mean={float(np.mean(start_idx_np)):.2f} "
            f"focus_abs_logit_mean={float(np.mean(focus_abs_np)):.6f} "
            f"reward_true_mean={float(np.mean(reward_true_np)):.6f} "
            f"reward_true_future_mean={float(np.mean(reward_true_future_np)):.6f} "
            f"reward_future_bonus_mean={float(np.mean(reward_future_bonus_np)):.6f} "
            f"reward_train_econ_mean={float(np.mean(reward_train_econ_np)):.6f} "
            f"reward_shape_skew_mean={float(np.mean(shape_skew_np)):.6f} "
            f"reward_shape_total_mean={float(np.mean(shape_total_np)):.6f} "
            f"cmssl_alpha_mean={float(np.mean(cmssl_alpha_np)):.6f} "
            f"shape_abs_ratio={shaping_ratio:.6f} "
            f"training_reward_actual_horizon_ms_mean={float(np.mean(training_reward_actual_horizon_ms_np)):.2f} "
            f"training_reward_actual_horizon_ms_p50={float(np.percentile(training_reward_actual_horizon_ms_np, 50.0)):.2f} "
            f"training_reward_actual_horizon_ms_p90={float(np.percentile(training_reward_actual_horizon_ms_np, 90.0)):.2f} "
            f"sampled_unique_starts={unique_starts} "
            f"sampled_min_start_distance={min_start_distance} "
            f"sampler_availability_reset={sampler_resets > 0}"
        )
        print(
            "[mm ppo quote geometry] "
            f"epoch={epoch + 1} "
            f"width_control_mean={float(np.mean(width_control_np)):.6f} "
            f"width_control_p50={float(np.percentile(width_control_np, 50.0)):.6f} "
            f"width_control_p90={float(np.percentile(width_control_np, 90.0)):.6f} "
            f"width_mult_mean={float(np.mean(width_mult_np)):.6f} "
            f"width_mult_p50={float(np.percentile(width_mult_np, 50.0)):.6f} "
            f"width_mult_p90={float(np.percentile(width_mult_np, 90.0)):.6f} "
            f"base_half_spread_bps_mean={float(np.mean(base_half_spread_bps_np)):.6f} "
            f"base_half_spread_bps_p50={float(np.percentile(base_half_spread_bps_np, 50.0)):.6f} "
            f"base_half_spread_bps_p90={float(np.percentile(base_half_spread_bps_np, 90.0)):.6f} "
            f"half_spread_bps_mean={float(np.mean(half_spread_bps_np)):.6f} "
            f"half_spread_bps_p50={float(np.percentile(half_spread_bps_np, 50.0)):.6f} "
            f"half_spread_bps_p90={float(np.percentile(half_spread_bps_np, 90.0)):.6f} "
            f"center_control_mean={float(np.mean(center_control_np)):.6f} "
            f"center_control_p50={float(np.percentile(center_control_np, 50.0)):.6f} "
            f"center_control_p90={float(np.percentile(center_control_np, 90.0)):.6f} "
            f"center_shift_scale_bps_mean={float(np.mean(center_shift_scale_bps_np)):.6f} "
            f"center_shift_scale_bps_p50={float(np.percentile(center_shift_scale_bps_np, 50.0)):.6f} "
            f"center_shift_scale_bps_p90={float(np.percentile(center_shift_scale_bps_np, 90.0)):.6f} "
            f"center_shift_min_bps_mean={float(np.mean(center_shift_min_bps_np)):.6f} "
            f"center_shift_max_bps_mean={float(np.mean(center_shift_max_bps_np)):.6f} "
            f"center_shift_bps_mean={float(np.mean(center_shift_bps_np)):.6f} "
            f"center_shift_bps_p50={float(np.percentile(center_shift_bps_np, 50.0)):.6f} "
            f"center_shift_bps_abs_p90={float(np.percentile(np.abs(center_shift_bps_np), 90.0)):.6f} "
            f"observed_spread_bps_mean={float(np.mean(observed_spread_bps_np)):.6f} "
            f"observed_spread_bps_p50={float(np.percentile(observed_spread_bps_np, 50.0)):.6f} "
            f"observed_spread_bps_p90={float(np.percentile(observed_spread_bps_np, 90.0)):.6f} "
            f"observed_anchor_half_spread_bps_mean={float(np.mean(observed_anchor_half_spread_bps_np)):.6f} "
            f"observed_anchor_half_spread_bps_p50={float(np.percentile(observed_anchor_half_spread_bps_np, 50.0)):.6f} "
            f"observed_anchor_half_spread_bps_p90={float(np.percentile(observed_anchor_half_spread_bps_np, 90.0)):.6f} "
            f"fill_norm_spread_floor_full_bps_mean={float(np.mean(fill_norm_spread_floor_full_bps_np)):.6f} "
            f"fill_norm_spread_floor_full_bps_p50={float(np.percentile(fill_norm_spread_floor_full_bps_np, 50.0)):.6f} "
            f"fill_norm_spread_floor_full_bps_p90={float(np.percentile(fill_norm_spread_floor_full_bps_np, 90.0)):.6f} "
            f"quote_half_spread_floor_bps_mean={float(np.mean(quote_half_spread_floor_bps_np)):.6f} "
            f"quote_half_spread_floor_bps_p50={float(np.percentile(quote_half_spread_floor_bps_np, 50.0)):.6f} "
            f"quote_half_spread_floor_bps_p90={float(np.percentile(quote_half_spread_floor_bps_np, 90.0)):.6f} "
            f"skew_bps_mean={float(np.mean(skew_bps_np)):.6f} "
            f"skew_bps_p50={float(np.percentile(skew_bps_np, 50.0)):.6f} "
            f"skew_bps_abs_p90={float(np.percentile(np.abs(skew_bps_np), 90.0)):.6f} "
            f"directional_response_mean={float(np.mean(directional_response_np)):.6f} "
            f"directional_response_abs_p90={float(np.percentile(np.abs(directional_response_np), 90.0)):.6f} "
            f"inventory_center_shift_bps_mean={float(np.mean(inventory_center_shift_bps_np)):.6f} "
            f"inventory_center_shift_bps_abs_p90={float(np.percentile(np.abs(inventory_center_shift_bps_np), 90.0)):.6f} "
            f"alpha_center_shift_bps_mean={float(np.mean(alpha_center_shift_bps_np)):.6f} "
            f"alpha_center_shift_bps_abs_p90={float(np.percentile(np.abs(alpha_center_shift_bps_np), 90.0)):.6f} "
            f"directional_center_response_mean={float(np.mean(directional_center_response_np)):.6f} "
            f"directional_center_response_abs_p90={float(np.percentile(np.abs(directional_center_response_np), 90.0)):.6f} "
            f"directional_asym_response_mean={float(np.mean(directional_asym_response_np)):.6f} "
            f"directional_asym_response_abs_p90={float(np.percentile(np.abs(directional_asym_response_np), 90.0)):.6f} "
            f"effective_alpha_center_capacity_bps_mean={float(np.mean(effective_alpha_center_capacity_bps_np)):.6f} "
            f"effective_alpha_center_capacity_bps_p50={float(np.percentile(effective_alpha_center_capacity_bps_np, 50.0)):.6f} "
            f"effective_alpha_center_capacity_bps_p90={float(np.percentile(effective_alpha_center_capacity_bps_np, 90.0)):.6f} "
            f"bid_half_spread_bps_mean={float(np.mean(bid_half_spread_bps_np)):.6f} "
            f"bid_half_spread_bps_p50={float(np.percentile(bid_half_spread_bps_np, 50.0)):.6f} "
            f"bid_half_spread_bps_p90={float(np.percentile(bid_half_spread_bps_np, 90.0)):.6f} "
            f"ask_half_spread_bps_mean={float(np.mean(ask_half_spread_bps_np)):.6f} "
            f"ask_half_spread_bps_p50={float(np.percentile(ask_half_spread_bps_np, 50.0)):.6f} "
            f"ask_half_spread_bps_p90={float(np.percentile(ask_half_spread_bps_np, 90.0)):.6f} "
            f"raw_spread_px_p50={float(np.percentile(raw_spread_px_np, 50.0)):.8f} "
            f"raw_spread_px_p90={float(np.percentile(raw_spread_px_np, 90.0)):.8f} "
            f"norm_spread_px_p50={float(np.percentile(norm_spread_px_np, 50.0)):.8f} "
            f"norm_spread_px_p90={float(np.percentile(norm_spread_px_np, 90.0)):.8f} "
            f"spread_floor_usage_frac={float(np.mean(used_norm_spread_floor_np)):.6f}"
        )
        print(
            "[mm ppo regime] "
            f"epoch={epoch + 1} "
            f"quote_touch_dist_avg={avg_touch_dist:.6f} "
            f"quote_touch_dist_p50={p50_touch_dist:.6f} "
            f"quote_touch_dist_p90={p90_touch_dist:.6f} "
            f"quote_at_touch_frac={at_touch_frac:.6f} "
            f"quote_off_touch_frac={off_touch_frac:.6f} "
            f"maker_fill_frac_mean={fill_mean:.6f} "
            f"activity_score_mean={float(np.mean(activity_score_np)):.6f} "
            f"touch_event_boost_mass_frac={bonus_frac:.6f} "
            f"reward_true_std={float(np.std(reward_true_np)):.6f} "
            f"reward_train_std={float(np.std((reward_train_econ_np + shape_total_np))):.6f}"
        )
        print(
            "[mm ppo fill path] "
            f"epoch={epoch + 1} "
            f"activity_sigma_bps_mean={float(np.mean(activity_sigma_bps_np)):.6f} "
            f"activity_sigma_bps_p50={float(np.percentile(activity_sigma_bps_np, 50.0)):.6f} "
            f"activity_sigma_bps_p90={float(np.percentile(activity_sigma_bps_np, 90.0)):.6f} "
            f"activity_score_mean={float(np.mean(activity_score_np)):.6f} "
            f"activity_score_p50={float(np.percentile(activity_score_np, 50.0)):.6f} "
            f"activity_score_p90={float(np.percentile(activity_score_np, 90.0)):.6f} "
            f"resting_quality_buy_mean={float(np.mean(resting_quality_buy_np)):.6f} "
            f"resting_quality_sell_mean={float(np.mean(resting_quality_sell_np)):.6f} "
            f"cross_confirmation_buy_mean={float(np.mean(cross_confirmation_buy_np)):.6f} "
            f"cross_confirmation_sell_mean={float(np.mean(cross_confirmation_sell_np)):.6f} "
            f"fill_interaction_buy_mean={float(np.mean(fill_interaction_buy_np)):.6f} "
            f"fill_interaction_sell_mean={float(np.mean(fill_interaction_sell_np)):.6f}"
        )
        print(
            "[mm ppo signal usage] "
            f"epoch={epoch + 1} "
            f"skew_bps_mean={float(np.mean(skew_bps_np)):.6f} "
            f"skew_bps_p50={float(np.percentile(skew_bps_np, 50.0)):.6f} "
            f"skew_bps_abs_p90={float(np.percentile(np.abs(skew_bps_np), 90.0)):.6f} "
            f"directional_response_mean={float(np.mean(directional_response_np)):.6f} "
            f"directional_response_abs_p90={float(np.percentile(np.abs(directional_response_np), 90.0)):.6f} "
            f"center_shift_bps_mean={float(np.mean(center_shift_bps_np)):.6f} "
            f"center_shift_bps_p50={float(np.percentile(center_shift_bps_np, 50.0)):.6f} "
            f"center_shift_bps_abs_p90={float(np.percentile(np.abs(center_shift_bps_np), 90.0)):.6f} "
            f"cmssl_skew_control_corr={_safe_corr(weighted_cmssl_logit_np, skew_control_np):.6f} "
            f"cmssl_skew_bps_corr={_safe_corr(weighted_cmssl_logit_np, skew_bps_np):.6f} "
            f"cmssl_alpha_center_shift_corr={_safe_corr(weighted_cmssl_logit_np, alpha_center_shift_bps_np):.6f} "
            f"cmssl_directional_center_response_corr={_safe_corr(weighted_cmssl_logit_np, directional_center_response_np):.6f} "
            f"cmssl_sign_agreement={sign_agreement:.6f} "
            f"abs_directional_response_high_abs_logit_mean={float(np.mean(np.abs(directional_response_np[high_logit_mask])) if np.any(high_logit_mask) else 0.0):.6f} "
            f"abs_directional_response_low_abs_logit_mean={float(np.mean(np.abs(directional_response_np[low_logit_mask])) if np.any(low_logit_mask) else 0.0):.6f} "
            f"abs_alpha_center_shift_high_abs_logit_mean={float(np.mean(np.abs(alpha_center_shift_bps_np[high_logit_mask])) if np.any(high_logit_mask) else 0.0):.6f} "
            f"abs_alpha_center_shift_low_abs_logit_mean={float(np.mean(np.abs(alpha_center_shift_bps_np[low_logit_mask])) if np.any(low_logit_mask) else 0.0):.6f} "
            f"bid_floor_fraction={float(np.mean(bid_at_floor_np)):.6f} "
            f"ask_floor_fraction={float(np.mean(ask_at_floor_np)):.6f}"
        )
        if shape_abs_mean > 0.0 and shaping_ratio > 0.10:
            print(
                "[mm ppo shaping warning] "
                f"epoch={epoch + 1} shape_abs_ratio={shaping_ratio:.6f} "
                "threshold=0.10 shaping may be dominating economic reward."
            )
        if (epoch + 1) % config.val_every == 0:
            assert train_env.freeze_obs_norm is True, "train env obs normalization must stay frozen during PPO"
            assert val_env.freeze_obs_norm is True, "val env obs normalization must stay frozen during PPO"
            deterministic_report = evaluate_market_policy_ppo(
                val_env,
                model,
                stochastic=False,
                device=device,
            )
            stochastic_generator = torch.Generator(device=torch.device(device).type)
            stochastic_generator.manual_seed(stochastic_val_seed)
            stochastic_report = evaluate_market_policy_ppo(
                val_env,
                model,
                stochastic=True,
                device=device,
                generator=stochastic_generator,
            )
            val_touch_mean = float(stochastic_report.get("mean_touch_dist", 0.0))
            val_touch_at = float(stochastic_report.get("touch_quote_frac", 0.0))
            val_touch_off = float(stochastic_report.get("off_touch_quote_frac", 0.0))
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
                f"reason={stoch_candidate_reason} "
                f"touch_dist_avg={val_touch_mean:.6f} "
                f"touch_at_frac={val_touch_at:.6f} "
                f"touch_off_frac={val_touch_off:.6f}"
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
                                "runtime_fill_config": {
                                    "width_action_init": float(mean_head_width_action_init),
                                    "fill_activity_min": float(train_env.continuous_maker_fill_config.activity_min),
                                    "fill_activity_max": float(train_env.continuous_maker_fill_config.activity_max),
                                    "fill_tau_touch": float(train_env.continuous_maker_fill_config.tau_touch),
                                    "fill_tau_cross": float(train_env.continuous_maker_fill_config.tau_cross),
                                    "fill_touch_event_boost": float(train_env.continuous_maker_fill_config.touch_event_boost),
                                    "fill_touch_event_distance_frac": float(train_env.continuous_maker_fill_config.touch_event_distance_frac),
                                    "fill_vol_p50_bps": float(train_env.continuous_maker_fill_calibration.vol_p50_bps),
                                    "fill_vol_p90_bps": float(train_env.continuous_maker_fill_calibration.vol_p90_bps),
                                },
                                "direct_quote_config": dict(train_env.direct_quote_config.__dict__),
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
    use_hard_maker_fill: bool = False,
) -> Dict[str, Any]:
    sigma_bps_selected_steps: List[float] = []
    delta_equity_steps: List[float] = []
    reward_steps: List[float] = []
    maker_buy_steps: List[float] = []
    maker_sell_steps: List[float] = []
    turnover_notional_steps: List[float] = []
    maker_buy_markout_steps: List[float] = []
    maker_sell_markout_steps: List[float] = []
    maker_buy_fill_frac_steps: List[float] = []
    maker_sell_fill_frac_steps: List[float] = []
    maker_exec_qty_total_steps: List[float] = []
    maker_buy_steps_signed: List[float] = []
    maker_sell_steps_signed: List[float] = []
    touch_dist_buy_steps: List[float] = []
    touch_dist_sell_steps: List[float] = []
    skew_control_steps: List[float] = []
    skew_bps_steps: List[float] = []
    directional_response_steps: List[float] = []
    effective_alpha_center_capacity_bps_steps: List[float] = []
    directional_center_response_steps: List[float] = []
    directional_asym_response_steps: List[float] = []
    bid_at_floor_steps: List[float] = []
    ask_at_floor_steps: List[float] = []
    center_shift_bps_steps: List[float] = []
    inventory_center_shift_bps_steps: List[float] = []
    alpha_center_shift_bps_steps: List[float] = []
    weighted_cmssl_logit_steps: List[float] = []
    obs = env.reset()
    equity_curve: List[float] = []
    inventory_curve: List[float] = []
    turnover_qty = 0.0
    turnover_notional = 0.0
    taker_notional = 0.0
    taker_fee_total = 0.0
    maker_rebate_total = 0.0
    maker_fill_count = 0
    maker_step_any_fill_count = 0
    maker_step_both_fill_count = 0
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
        if use_hard_maker_fill:
            obs, reward, done, info = env.step_hard_diagnostic(action, emit_info=True)
        else:
            obs, reward, done, info = env.step(action, emit_info=True)
        require(info is not None, "MarketMakingEnv.step(..., emit_info=True) must return diagnostics info")
        equity_curve.append(info["equity"])
        inventory_curve.append(info["inventory"])
        steps += 1
        total_reward += float(reward)
        total_delta_equity += float(info.get("delta_equity", 0.0))
        inventory_penalty_total += float(info.get("inventory_penalty_total", 0.0))
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
        maker_step_any_fill_count += int(info["maker_buy"] > 0.0 or info["maker_sell"] > 0.0)
        maker_step_both_fill_count += int(info["maker_buy"] > 0.0 and info["maker_sell"] > 0.0)
        maker_opps += 2
        maker_buy_fill_frac_steps.append(float(info.get("maker_buy_fill_frac", 0.0)))
        maker_sell_fill_frac_steps.append(float(info.get("maker_sell_fill_frac", 0.0)))
        maker_exec_qty_total_steps.append(float(info.get("maker_buy_exec_qty", maker_buy)) + float(info.get("maker_sell_exec_qty", maker_sell)))
        touch_dist_buy_steps.append(float(info.get("touch_dist_buy", 0.0)))
        touch_dist_sell_steps.append(float(info.get("touch_dist_sell", 0.0)))
        maker_buy_steps_signed.append(float(info.get("maker_buy", 0.0)))
        maker_sell_steps_signed.append(float(info.get("maker_sell", 0.0)))
        skew_control_steps.append(float(info.get("skew_control", 0.0)))
        skew_bps_steps.append(float(info.get("skew_bps", 0.0)))
        directional_center_response_steps.append(float(info.get("directional_center_response", 0.0)))
        directional_asym_response_steps.append(float(info.get("directional_asym_response", 0.0)))
        directional_response_steps.append(float(info.get("directional_response", 0.0)))
        effective_alpha_center_capacity_bps_steps.append(float(info.get("effective_alpha_center_capacity_bps", 0.0)))
        bid_at_floor_steps.append(float(info.get("bid_at_floor", 0.0)))
        ask_at_floor_steps.append(float(info.get("ask_at_floor", 0.0)))
        center_shift_bps_steps.append(float(info.get("center_shift_bps", 0.0)))
        inventory_center_shift_bps_steps.append(float(info.get("inventory_center_shift_bps", 0.0)))
        alpha_center_shift_bps_steps.append(float(info.get("alpha_center_shift_bps", 0.0)))
        weighted_cmssl_logit_steps.append(float(info.get("weighted_cmssl_logit", 0.0)))
        taker_steps += int(info["taker_buy"] > 0.0 or info["taker_sell"] > 0.0)
        if collect_vol_bucket_report:
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
    maker_side_hit_rate = float(maker_fill_count / maker_opps) if maker_opps > 0 else 0.0
    maker_step_any_fill_rate = float(maker_step_any_fill_count / steps) if steps > 0 else 0.0
    maker_step_both_fill_rate = float(maker_step_both_fill_count / steps) if steps > 0 else 0.0
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
    pre_fee_pnl = float(net_pnl - net_fee_cost)
    pre_fee_pnl_pct = float(pre_fee_pnl / denom)
    pnl_identity_residual = float(abs(net_pnl - (pre_fee_pnl + net_fee_cost)))
    if pnl_identity_residual > 1e-8:
        raise RuntimeError(
            f"PnL identity violated: net_pnl={net_pnl:.10f} pre_fee_pnl={pre_fee_pnl:.10f} "
            f"net_fee_cost={net_fee_cost:.10f} residual={pnl_identity_residual:.10e}"
        )
    ending_inventory_qty = float(inventory_arr[-1]) if inventory_arr.size > 0 else 0.0
    ending_inventory_notional = float(abs(ending_inventory_qty * last_mid))
    maker_turnover_notional = float(turnover_notional - taker_notional)
    maker_turnover_share = float(maker_turnover_notional / turnover_notional) if turnover_notional > 0 else 0.0
    maker_buy_fill_frac_arr = np.asarray(maker_buy_fill_frac_steps, dtype=np.float64)
    maker_sell_fill_frac_arr = np.asarray(maker_sell_fill_frac_steps, dtype=np.float64)
    maker_exec_qty_total_arr = np.asarray(maker_exec_qty_total_steps, dtype=np.float64)
    touch_dist_buy_arr = np.asarray(touch_dist_buy_steps, dtype=np.float64)
    touch_dist_sell_arr = np.asarray(touch_dist_sell_steps, dtype=np.float64)
    skew_control_arr = np.asarray(skew_control_steps, dtype=np.float64)
    skew_bps_arr = np.asarray(skew_bps_steps, dtype=np.float64)
    directional_center_response_arr = np.asarray(directional_center_response_steps, dtype=np.float64)
    directional_asym_response_arr = np.asarray(directional_asym_response_steps, dtype=np.float64)
    directional_response_arr = np.asarray(directional_response_steps, dtype=np.float64)
    effective_alpha_center_capacity_bps_arr = np.asarray(effective_alpha_center_capacity_bps_steps, dtype=np.float64)
    bid_at_floor_arr = np.asarray(bid_at_floor_steps, dtype=np.float64)
    ask_at_floor_arr = np.asarray(ask_at_floor_steps, dtype=np.float64)
    center_shift_bps_arr = np.asarray(center_shift_bps_steps, dtype=np.float64)
    inventory_center_shift_bps_arr = np.asarray(inventory_center_shift_bps_steps, dtype=np.float64)
    alpha_center_shift_bps_arr = np.asarray(alpha_center_shift_bps_steps, dtype=np.float64)
    weighted_cmssl_logit_arr = np.asarray(weighted_cmssl_logit_steps, dtype=np.float64)
    maker_buy_signed_arr = np.asarray(maker_buy_steps_signed, dtype=np.float64)
    maker_sell_signed_arr = np.asarray(maker_sell_steps_signed, dtype=np.float64)
    abs_logit_arr = np.abs(weighted_cmssl_logit_arr)
    logit_med = float(np.median(abs_logit_arr)) if abs_logit_arr.size else 0.0
    high_mask = abs_logit_arr >= logit_med
    low_mask = ~high_mask

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "pre_fee_pnl": pre_fee_pnl,
        "pre_fee_pnl_pct": pre_fee_pnl_pct,
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
        "maker_side_hit_rate": maker_side_hit_rate,
        "maker_step_any_fill_rate": maker_step_any_fill_rate,
        "maker_step_both_fill_rate": maker_step_both_fill_rate,
        "maker_buy_fill_frac_mean": float(np.mean(maker_buy_fill_frac_arr)) if maker_buy_fill_frac_arr.size > 0 else 0.0,
        "maker_sell_fill_frac_mean": float(np.mean(maker_sell_fill_frac_arr)) if maker_sell_fill_frac_arr.size > 0 else 0.0,
        "maker_exec_qty_total_per_step_mean": float(np.mean(maker_exec_qty_total_arr)) if maker_exec_qty_total_arr.size > 0 else 0.0,
        "mean_touch_dist": float(np.mean(0.5 * (touch_dist_buy_arr + touch_dist_sell_arr))) if touch_dist_buy_arr.size > 0 else 0.0,
        "touch_quote_frac": float(np.mean((touch_dist_buy_arr <= 0.10) & (touch_dist_sell_arr <= 0.10))) if touch_dist_buy_arr.size > 0 else 0.0,
        "off_touch_quote_frac": float(np.mean((touch_dist_buy_arr >= 0.50) & (touch_dist_sell_arr >= 0.50))) if touch_dist_buy_arr.size > 0 else 0.0,
        "maker_fill_count": int(maker_fill_count),
        "maker_opportunities": int(maker_opps),
        "maker_step_any_fill_count": int(maker_step_any_fill_count),
        "maker_step_both_fill_count": int(maker_step_both_fill_count),
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
        "pnl_identity_residual": pnl_identity_residual,
        "bid_floor_fraction": float(np.mean(bid_at_floor_arr)) if bid_at_floor_arr.size else 0.0,
        "ask_floor_fraction": float(np.mean(ask_at_floor_arr)) if ask_at_floor_arr.size else 0.0,
        "maker_buy_inventory_clip_fraction": float(maker_buy_clipped_steps / steps) if steps > 0 else 0.0,
        "maker_sell_inventory_clip_fraction": float(maker_sell_clipped_steps / steps) if steps > 0 else 0.0,
        "skew_bps_mean": float(np.mean(skew_bps_arr)) if skew_bps_arr.size else 0.0,
        "skew_bps_p50": float(np.percentile(skew_bps_arr, 50.0)) if skew_bps_arr.size else 0.0,
        "skew_bps_abs_p90": float(np.percentile(np.abs(skew_bps_arr), 90.0)) if skew_bps_arr.size else 0.0,
        "directional_response_mean": float(np.mean(directional_response_arr)) if directional_response_arr.size else 0.0,
        "directional_response_abs_p90": float(np.percentile(np.abs(directional_response_arr), 90.0)) if directional_response_arr.size else 0.0,
        "effective_alpha_center_capacity_bps_mean": float(np.mean(effective_alpha_center_capacity_bps_arr)) if effective_alpha_center_capacity_bps_arr.size else 0.0,
        "effective_alpha_center_capacity_bps_p90": float(np.percentile(effective_alpha_center_capacity_bps_arr, 90.0)) if effective_alpha_center_capacity_bps_arr.size else 0.0,
        "directional_center_response_mean": float(np.mean(directional_center_response_arr)) if directional_center_response_arr.size else 0.0,
        "directional_center_response_abs_p90": float(np.percentile(np.abs(directional_center_response_arr), 90.0)) if directional_center_response_arr.size else 0.0,
        "directional_asym_response_mean": float(np.mean(directional_asym_response_arr)) if directional_asym_response_arr.size else 0.0,
        "directional_asym_response_abs_p90": float(np.percentile(np.abs(directional_asym_response_arr), 90.0)) if directional_asym_response_arr.size else 0.0,
        "center_shift_bps_mean": float(np.mean(center_shift_bps_arr)) if center_shift_bps_arr.size else 0.0,
        "center_shift_bps_p50": float(np.percentile(center_shift_bps_arr, 50.0)) if center_shift_bps_arr.size else 0.0,
        "center_shift_bps_abs_p90": float(np.percentile(np.abs(center_shift_bps_arr), 90.0)) if center_shift_bps_arr.size else 0.0,
        "inventory_center_shift_bps_mean": float(np.mean(inventory_center_shift_bps_arr)) if inventory_center_shift_bps_arr.size else 0.0,
        "inventory_center_shift_bps_abs_p90": float(np.percentile(np.abs(inventory_center_shift_bps_arr), 90.0)) if inventory_center_shift_bps_arr.size else 0.0,
        "alpha_center_shift_bps_mean": float(np.mean(alpha_center_shift_bps_arr)) if alpha_center_shift_bps_arr.size else 0.0,
        "alpha_center_shift_bps_abs_p90": float(np.percentile(np.abs(alpha_center_shift_bps_arr), 90.0)) if alpha_center_shift_bps_arr.size else 0.0,
        "cmssl_skew_control_corr": _safe_corr(weighted_cmssl_logit_arr, skew_control_arr),
        "cmssl_skew_bps_corr": _safe_corr(weighted_cmssl_logit_arr, skew_bps_arr),
        "cmssl_alpha_center_shift_corr": _safe_corr(weighted_cmssl_logit_arr, alpha_center_shift_bps_arr),
        "cmssl_directional_center_response_corr": _safe_corr(weighted_cmssl_logit_arr, directional_center_response_arr),
        "cmssl_sign_agreement_rate": float(np.mean(np.sign(directional_response_arr) == np.sign(weighted_cmssl_logit_arr))) if directional_response_arr.size else 0.0,
        "abs_directional_response_high_abs_logit_mean": float(np.mean(np.abs(directional_response_arr[high_mask]))) if directional_response_arr.size and np.any(high_mask) else 0.0,
        "abs_directional_response_low_abs_logit_mean": float(np.mean(np.abs(directional_response_arr[low_mask]))) if directional_response_arr.size and np.any(low_mask) else 0.0,
        "abs_alpha_center_shift_high_abs_logit_mean": float(np.mean(np.abs(alpha_center_shift_bps_arr[high_mask]))) if alpha_center_shift_bps_arr.size and np.any(high_mask) else 0.0,
        "abs_alpha_center_shift_low_abs_logit_mean": float(np.mean(np.abs(alpha_center_shift_bps_arr[low_mask]))) if alpha_center_shift_bps_arr.size and np.any(low_mask) else 0.0,
        "maker_buy_volume_when_signal_pos": float(np.sum(maker_buy_signed_arr[weighted_cmssl_logit_arr > 0.0])) if maker_buy_signed_arr.size else 0.0,
        "maker_sell_volume_when_signal_pos": float(np.sum(maker_sell_signed_arr[weighted_cmssl_logit_arr > 0.0])) if maker_sell_signed_arr.size else 0.0,
        "maker_buy_volume_when_signal_neg": float(np.sum(maker_buy_signed_arr[weighted_cmssl_logit_arr < 0.0])) if maker_buy_signed_arr.size else 0.0,
        "maker_sell_volume_when_signal_neg": float(np.sum(maker_sell_signed_arr[weighted_cmssl_logit_arr < 0.0])) if maker_sell_signed_arr.size else 0.0,
        "cadence": {
            "step_ms": step_ms,
            "steps_per_year": float(steps_per_year),
            "source": cadence["source"],
            "diff_count": cadence["diff_count"],
            "timestamp_source": ts_source,
        },
    }
    if collect_vol_bucket_report:
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
                maker_buy_fill_frac_per_step=np.asarray(maker_buy_fill_frac_steps, dtype=np.float64),
                maker_sell_fill_frac_per_step=np.asarray(maker_sell_fill_frac_steps, dtype=np.float64),
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
        f"pre_fee_pnl={float(metrics.get('pre_fee_pnl', 0.0)):.4f} "
        f"pre_fee_pnl_pct={float(metrics.get('pre_fee_pnl_pct', 0.0)):.6f} "
        f"net_fee_pct_initial_equity={float(metrics.get('net_fee_pct_initial_equity', 0.0)):.6f} "
        f"sharpe={float(metrics.get('sharpe', 0.0)):.4f} "
        f"sharpe_5m={float(metrics.get('sharpe_5m', 0.0)):.4f} "
        f"sharpe_1h={float(metrics.get('sharpe_1h', 0.0)):.4f} "
        f"sortino_5m={float(metrics.get('sortino_5m', 0.0)):.4f} "
        f"sortino_1h={float(metrics.get('sortino_1h', 0.0)):.4f} "
        f"max_dd={float(metrics.get('max_drawdown', 0.0)):.4f} "
        f"turnover_notional={float(metrics.get('turnover_notional', 0.0)):.4f} "
        f"turnover_qty={float(metrics.get('turnover_qty', 0.0)):.4f} "
        f"maker_side_hit_rate={float(metrics.get('maker_side_hit_rate', 0.0)):.4f} "
        f"maker_step_any_fill_rate={float(metrics.get('maker_step_any_fill_rate', 0.0)):.4f} "
        f"maker_step_both_fill_rate={float(metrics.get('maker_step_both_fill_rate', 0.0)):.4f} "
        f"bid_floor_fraction={float(metrics.get('bid_floor_fraction', 0.0)):.4f} "
        f"ask_floor_fraction={float(metrics.get('ask_floor_fraction', 0.0)):.4f} "
        f"cmssl_skew_bps_corr={float(metrics.get('cmssl_skew_bps_corr', 0.0)):.4f} "
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




def run_pipeline(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    ppo_epochs: int = 50,
    run_mode: str = "train",
) -> Dict[str, Any]:
    print(f"[mm run mode] {run_mode}")
    _fail_on_removed_env_vars(
        (
            "BYBIT_MM_CENTER_ANCHOR_FRAC",
            "BYBIT_MM_SKEW_ANCHOR_FRAC",
            "BYBIT_MM_FILL_TOUCH_EVENT_BONUS",
            "BYBIT_MM_WIDTH_EXPAND_MULT",
            "BYBIT_MM_WIDTH_TIGHTEN_FRAC",
            "BYBIT_MM_FILL_EPS_PX",
            "BYBIT_MM_CENTER_LIMIT_BPS",
            "BYBIT_MM_SKEW_LIMIT_BPS",
            "BYBIT_MM_REWARD_SHAPING_SKEW_TARGET_SCALE",
            "BYBIT_MM_REWARD_SHAPING_WIDTH_ALPHA_COEF",
            "BYBIT_MM_REWARD_SHAPING_WIDTH_VOL_COEF",
            "BYBIT_MM_REWARD_SHAPING_WIDTH_TARGET_MIN",
            "BYBIT_MM_REWARD_SHAPING_WIDTH_TARGET_MAX",
            "BYBIT_MM_REWARD_SHAPING_HORIZON_LOGIT_WEIGHTS",
            "BYBIT_MM_START_SAMPLING_HORIZON_LOGIT_WEIGHTS",
        )
    )
    meta = load_global_meta(Path(out_root))
    directional_signal_cfg = load_directional_signal_config(meta)
    cmssl_test_split = resolve_cmssl_test_split(out_root, meta)
    rl_train_split = resolve_rl_train_split(out_root, meta)
    rl_val_split = resolve_rl_val_split(out_root, meta)
    rl_test_split = resolve_rl_test_split(out_root, meta)
    eval_full_split = resolve_eval_full_split(out_root, meta)

    report_cmssl_test_diagnostics(out_root, meta)

    model, _meta = load_cmssl(out_root, ckpt_path, device=device)
    _validate_fixed_cmssl_horizons([int(h) for h in meta.get("horizons_ms", [])])

    cmssl_batch_size = _resolve_cmssl_batch_size()
    rollout_storage = _resolve_rollout_storage("gpu")
    pin_rollout_memory = _env_bool("BYBIT_MM_PIN_ROLLOUT_MEMORY", True)
    non_blocking_transfers = _env_bool("BYBIT_MM_NONBLOCKING_TRANSFERS", True)
    _timing_log(
        "run_config "
        f"cmssl_batch_size={cmssl_batch_size} "
        f"rollout_storage={rollout_storage} "
        f"compile_cmssl={_env_bool('BYBIT_MM_COMPILE_CMSSL', False)} "
        f"compile_ppo={_env_bool('BYBIT_MM_COMPILE_PPO', False)} "
        f"tf32={_env_bool('BYBIT_MM_ENABLE_TF32', False)}"
    )
    joined_cmssl_test = build_joined_split(
        out_root,
        cmssl_test_split,
        model,
        meta,
        device,
        ckpt_path=ckpt_path,
        split_label="cmssl_test",
        directional_signal_config=directional_signal_cfg,
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
    joined_rl_full = build_joined_split(
        out_root,
        rl_week3_full_split,
        model,
        meta,
        device,
        ckpt_path=ckpt_path,
        split_label="rl_week3_full",
        directional_signal_config=directional_signal_cfg,
        batch_size=cmssl_batch_size,
    )
    joined_rl_train = slice_joined_by_split(joined_rl_full, rl_train_split)
    joined_rl_val = slice_joined_by_split(joined_rl_full, rl_val_split)
    joined_rl_test = slice_joined_by_split(joined_rl_full, rl_test_split)
    joined_eval_full = build_joined_split(
        out_root,
        eval_full_split,
        model,
        meta,
        device,
        ckpt_path=ckpt_path,
        split_label="eval_full",
        directional_signal_config=directional_signal_cfg,
        batch_size=cmssl_batch_size,
    )

    mm_train_batch = build_market_batch(joined_rl_train)
    mm_val_batch = build_market_batch(joined_rl_val)
    mm_test_batch = build_market_batch(joined_rl_test)
    mm_eval_full_batch = build_market_batch(joined_eval_full)
    env_kwargs_common = resolve_market_env_common_kwargs_from_env()
    allow_taker = os.environ.get("BYBIT_MM_ALLOW_TAKER", "true").strip().lower() in {"1", "true", "yes", "y"}
    reward_shaping_cfg = load_reward_shaping_config()
    quote_cfg = env_kwargs_common["direct_quote_config"]
    fill_cfg = env_kwargs_common["continuous_maker_fill_config"]
    fill_calibration = _fit_continuous_maker_fill_calibration_from_snapshots(joined_rl_train["snapshots"])
    print(
        "[mm fill calibration] "
        f"source=rl_train_split "
        f"sample_count={fill_calibration.sample_count} "
        f"vol_p50_bps={fill_calibration.vol_p50_bps:.6f} "
        f"vol_p90_bps={fill_calibration.vol_p90_bps:.6f} "
        f"vol_mean_bps={fill_calibration.vol_mean_bps:.6f} "
        f"vol_p99_bps={fill_calibration.vol_p99_bps:.6f}"
    )
    print("[mm quote config]", json.dumps(dict(quote_cfg.__dict__), sort_keys=True))
    print("[mm fill config]", json.dumps(dict(fill_cfg.__dict__), sort_keys=True))
    print(
        "[mm fill config] "
        f"calibrated=True "
        f"vol_p50_bps={fill_calibration.vol_p50_bps:.6f} "
        f"vol_p90_bps={fill_calibration.vol_p90_bps:.6f}"
    )

    mm_train_env = MarketMakingEnv(
        mm_train_batch,
        allow_taker=allow_taker,
        reward_shaping_config=reward_shaping_cfg,
        directional_signal_config=directional_signal_cfg,
        continuous_maker_fill_calibration=fill_calibration,
        **env_kwargs_common,
    )
    print("[mm obs scaling]", json.dumps(mm_train_env.get_observation_scaling_config(), sort_keys=True))
    mm_val_env = MarketMakingEnv(
        mm_val_batch,
        allow_taker=allow_taker,
        reward_shaping_config=reward_shaping_cfg,
        directional_signal_config=directional_signal_cfg,
        continuous_maker_fill_calibration=fill_calibration,
        **env_kwargs_common,
    )
    mm_test_env = MarketMakingEnv(
        mm_test_batch,
        allow_taker=allow_taker,
        reward_shaping_config=reward_shaping_cfg,
        directional_signal_config=directional_signal_cfg,
        continuous_maker_fill_calibration=fill_calibration,
        **env_kwargs_common,
    )
    mm_final_env = MarketMakingEnv(
        mm_eval_full_batch,
        allow_taker=allow_taker,
        reward_shaping_config=reward_shaping_cfg,
        directional_signal_config=directional_signal_cfg,
        continuous_maker_fill_calibration=fill_calibration,
        **env_kwargs_common,
    )
    prefitted_obs_norm_state = prefit_market_obs_norm(mm_train_env)
    mm_train_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_val_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_test_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_final_env.set_obs_norm_state(prefitted_obs_norm_state, freeze=True)
    mm_obs = mm_train_env.reset()
    mm_obs_dim = mm_obs.shape[0]
    print(
        "[mm obs norm] "
        f"source=train_prefit count={int(prefitted_obs_norm_state['count'])} "
        f"obs_dim={mm_obs_dim} frozen={mm_train_env.freeze_obs_norm}"
    )
    prefitted_mask = np.asarray(prefitted_obs_norm_state["continuous_mask"], dtype=bool)
    extras_mask = prefitted_mask[-ENV_OBS_EXTRA_STATE_DIM:]
    print(
        "[mm obs norm extras] "
        f"extra_dim={ENV_OBS_EXTRA_STATE_DIM} "
        f"extras_normalized={bool(np.any(extras_mask))} "
        f"feature_mask_true={int(np.sum(prefitted_mask[:-ENV_OBS_EXTRA_STATE_DIM]))}"
    )

    mm_ppo_config = PPOConfig(
        lr=float(os.environ.get("BYBIT_MM_PPO_LR", "3e-4")),
        update_epochs=int(os.environ.get("BYBIT_MM_PPO_UPDATE_EPOCHS", "8")),
        batch_size=int(os.environ.get("BYBIT_MM_PPO_BATCH_SIZE", "65536")),
        clip_ratio=float(os.environ.get("BYBIT_MM_PPO_CLIP_RATIO", "0.2")),
        gamma=float(os.environ.get("BYBIT_MM_PPO_GAMMA", "0.999")),
        gae_lambda=float(os.environ.get("BYBIT_MM_PPO_GAE_LAMBDA", "0.99")),
        entropy_coef=float(os.environ.get("BYBIT_MM_PPO_ENTROPY_COEF", "0.0075")),
        value_coef=float(os.environ.get("BYBIT_MM_PPO_VALUE_COEF", "0.5")),
        policy_hidden=tuple(int(x) for x in os.environ.get("BYBIT_MM_PPO_POLICY_HIDDEN", "128,128").split(",")),
        value_hidden=tuple(int(x) for x in os.environ.get("BYBIT_MM_PPO_VALUE_HIDDEN", "128,128").split(",")),
        val_every=_env_int("BYBIT_MM_PPO_VAL_EVERY", 10),
        max_drawdown_guard=_env_float("BYBIT_MM_PPO_MAX_DRAWDOWN", float("nan")),
        rollout_horizon=_env_int("BYBIT_MM_PPO_ROLLOUT_HORIZON", 8192),
        rollouts_per_epoch=_env_int("BYBIT_MM_PPO_ROLLOUTS_PER_EPOCH", 32),
        randomize_rollout_start=_env_bool("BYBIT_MM_PPO_RANDOMIZE_START", True),
        init_log_std_center=_env_float("BYBIT_MM_PPO_INIT_LOG_STD_CENTER", -0.20),
        init_log_std_width=_env_float("BYBIT_MM_PPO_INIT_LOG_STD_WIDTH", -1.00),
        init_log_std_skew=_env_float("BYBIT_MM_PPO_INIT_LOG_STD_SKEW", 0.00),
        init_log_std_taker=_env_float("BYBIT_MM_PPO_INIT_LOG_STD_TAKER", -1.0),
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
        obs_norm_source = "checkpoint"

    rl_metrics = None
    rl_dev_metrics = None
    rl_dev_hard_diag_metrics = None
    rl_final_hard_diag_metrics = None
    ppo_eval_stochastic = _env_bool("BYBIT_MM_PPO_EVAL_STOCHASTIC", False)
    ppo_eval_seed = _env_int("BYBIT_MM_PPO_EVAL_SEED", 0)
    run_hard_fill_diagnostics = _env_bool("BYBIT_MM_RUN_HARD_FILL_DIAGNOSTICS", True)

    eval_action = "skipped" if run_mode == "train" else "performed"
    print(
        "[mm eval] "
        f"mode={run_mode} "
        f"checkpoint_origin={rl_checkpoint_origin} "
        f"resolved_path={resolved_eval_ckpt if resolved_eval_ckpt is not None else 'none'} "
        f"eval_action={eval_action}"
    )

    if run_mode != "train":
        if resolved_eval_ckpt is None:
            raise RuntimeError(
                "run_mode in {'eval','train_eval'} requires a PPO checkpoint; none was resolved."
            )
        mm_ppo_model = load_market_ppo_model(
            mm_obs_dim,
            device=device,
            ckpt_path=resolved_eval_ckpt,
            checkpoint_data=eval_ckpt_payload,
        )
        require(mm_ppo_model is not None, "Failed to load eval PPO checkpoint")
        rl_policy_loaded = True
        rl_policy_reason = "loaded"
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
            generator=stochastic_generator,
        )
        if run_hard_fill_diagnostics:
            rl_dev_hard_diag_metrics = evaluate_market_policy_ppo(
                mm_test_env,
                mm_ppo_model,
                stochastic=ppo_eval_stochastic,
                device=device,
                generator=stochastic_generator,
                use_hard_maker_fill=True,
            )
        _timing_log(f"evaluate_market_making rl secs={time.perf_counter() - rl_eval_t0:.4f}")
        rl_eval_performed = True
        rl_metrics = evaluate_market_policy_ppo(
            mm_final_env,
            mm_ppo_model,
            stochastic=ppo_eval_stochastic,
            device=device,
            generator=stochastic_generator,
        )
        if run_hard_fill_diagnostics:
            rl_final_hard_diag_metrics = evaluate_market_policy_ppo(
                mm_final_env,
                mm_ppo_model,
                stochastic=ppo_eval_stochastic,
                device=device,
                generator=stochastic_generator,
                use_hard_maker_fill=True,
            )
    else:
        rl_policy_loaded = False
        rl_policy_reason = "skipped because BYBIT_MM_RUN_MODE=train"

    return {
        "cmssl_test": cmssl_report,
        "mm_obs_scaling": mm_train_env.get_observation_scaling_config(),
        "mm_rl": rl_metrics,
        "mm_rl_dev_test": rl_dev_metrics,
        "mm_rl_test_hard_diag": rl_dev_hard_diag_metrics,
        "mm_rl_final_hard_diag": rl_final_hard_diag_metrics,
        # Fatal failures are surfaced via exceptions rather than persisted in provenance state.
        "mm_run_context": {
            "run_mode": run_mode,
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
    ppo_epochs = _resolve_ppo_epochs(50)
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
    rl_report = report["mm_rl"]
    if rl_report is None:
        print("[mm rl] skipped (mm_rl is None)")
    else:
        print("[mm eval]", _format_mm_summary("rl", rl_report))
    if verbose_reports:
        if rl_report is None:
            print("[mm rl verbose] skipped (mm_rl is None)")
        else:
            print("[mm rl verbose]", _summarize_for_log(rl_report))
    if run_cmssl_test_window:
        print("[cmssl test window] running windowed inference for diagnostics.")
        test_window_report = run_cmssl_test_window_inference(out_root, ckpt_path, device=device)
        print("[cmssl test window] completed", json.dumps({"horizons_ms": test_window_report["horizons_ms"]}))
