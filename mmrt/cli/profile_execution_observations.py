"""Profile raw and normalized execution PPO observation values."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from mmrt.execution.adverse_signal import AdverseSelectionSignalArtifact, load_adverse_selection_signals
from mmrt.execution.contracts import QueueModelMode
from mmrt.execution.decision_grid import load_decision_grid, validate_decision_grid_for_execution_tape
from mmrt.execution.env import ExecutionEnv
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.linear_signal import (
    load_linear_signal_artifact_npz,
    linear_signal_artifact_summary,
)
from mmrt.execution.obs_schema import ObservationSchema, observation_field_groups
from mmrt.execution.quote_geometry import QuoteAction
from mmrt.execution.split_contract import load_execution_split_contract, ranges_for_split, split_contracts_equal
from mmrt.cli.execution_defaults import (
    DEFAULT_CANCEL_GUARD_TICKS,
    DEFAULT_CANCEL_LATENCY_US,
    DEFAULT_DECISION_COMPUTE_LATENCY_US,
    DEFAULT_DEFAULT_ORDER_QTY,
    DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS,
    DEFAULT_L2_DECREASE_WEIGHT,
    DEFAULT_MAKER_FEE_BPS,
    DEFAULT_MAX_DISTANCE_TICKS,
    DEFAULT_MAX_ORDER_QTY,
    DEFAULT_ORDER_ENTRY_LATENCY_US,
    DEFAULT_POST_ONLY_GAP_TICKS,
    DEFAULT_QUEUE_MODE,
    DEFAULT_TRADE_AT_LEVEL_WEIGHT,
    DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY,
)
from mmrt.cli.execution_env_config import ExecutionEnvConfigBuildInput, build_execution_env_config_from_input
from mmrt.cli.evaluate_execution_policy import (
    _adverse_queue_config_compatibility,
    _env_config_from_training_cli_config,
    _load_checkpoint,
    _load_observation_normalizer_from_checkpoint,
    _load_policy_from_checkpoint,
)
from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from mmrt.cli.output import STDOUT_MODES, compact_json_line, validate_stdout_mode, write_json_atomic
from mmrt.rl.device import resolve_torch_device, torch_device_summary
from mmrt.rl.normalization import ObservationNormalizer, ObservationNormalizerConfig
from mmrt.rl.rollout import TrainWindowSampler


SAMPLE_POLICIES = (
    "no_quote",
    "two_sided",
    "alternate_bid_ask",
    "checkpoint_deterministic",
    "checkpoint_stochastic",
)

_ENV_DEFAULTS = {
    "cancel_guard_ticks": DEFAULT_CANCEL_GUARD_TICKS,
    "max_episode_steps": None,
    "max_distance_ticks": DEFAULT_MAX_DISTANCE_TICKS,
    "max_order_qty": DEFAULT_MAX_ORDER_QTY,
    "post_only_gap_ticks": DEFAULT_POST_ONLY_GAP_TICKS,
    "default_order_qty": DEFAULT_DEFAULT_ORDER_QTY,
    "queue_mode": DEFAULT_QUEUE_MODE,
    "l2_decrease_weight": DEFAULT_L2_DECREASE_WEIGHT,
    "trade_at_level_weight": DEFAULT_TRADE_AT_LEVEL_WEIGHT,
    "unknown_level_queue_ahead_qty": DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY,
    "dedupe_l2_decrease_with_trade_prints": DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS,
    "maker_fee_bps": DEFAULT_MAKER_FEE_BPS,
    "edge_min_executable_edge_bps": 0.0,
    "edge_latency_buffer_bps": 0.0,
    "edge_inventory_skew_bps_per_unit": 0.0,
    "decision_compute_latency_us": DEFAULT_DECISION_COMPUTE_LATENCY_US,
    "order_entry_latency_us": DEFAULT_ORDER_ENTRY_LATENCY_US,
    "cancel_latency_us": DEFAULT_CANCEL_LATENCY_US,
    "inventory_penalty_bps": 0.0,
    "turnover_penalty_bps": 0.0,
    "cancel_penalty": 0.0,
    "drawdown_penalty_rate": 0.0,
    "terminal_inventory_penalty_bps": 0.0,
    "reward_scale": 1.0,
}

_ENV_OVERRIDE_KEYS = tuple(_ENV_DEFAULTS.keys())

__all__ = [
    "ExecutionObservationProfileConfig",
    "run_execution_observation_profile",
    "build_arg_parser",
    "main",
]


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty str")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite float")
    return out


def _optional_finite_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, name)


def _coerce_queue_mode(value: QueueModelMode | str | None) -> QueueModelMode | None:
    if value is None:
        return None
    if isinstance(value, QueueModelMode):
        return value
    if isinstance(value, str):
        try:
            return QueueModelMode(value)
        except ValueError as exc:
            raise ValueError(f"queue_mode has invalid value {value!r}") from exc
    raise ValueError("queue_mode must be QueueModelMode, str, or None")


def _coerce_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value in (torch.float32, torch.float64):
            return value
        raise ValueError("dtype must be float32 or float64")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("float32", "fp32", "torch.float32"):
            return torch.float32
        if normalized in ("float64", "fp64", "torch.float64"):
            return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionObservationProfileConfig:
    tape_root: str
    decision_grid_path: str
    split_source_dataset_root: str
    split: str
    linear_signals_npz: str
    output_json: str

    adverse_signals_npz: str | None = None
    checkpoint_path: str | None = None
    sample_rows: int = 100_000
    num_envs: int = 4
    seed: int = 123
    sample_policy: str = "no_quote"
    device: str | None = "auto"
    dtype: torch.dtype | str = torch.float32
    stdout_mode: str = "summary"
    overwrite: bool = False

    cancel_guard_ticks: int | None = None
    max_episode_steps: int | None = None
    max_distance_ticks: int | None = None
    max_order_qty: float | None = None
    post_only_gap_ticks: int | None = None
    default_order_qty: float | None = None
    queue_mode: QueueModelMode | str | None = None
    l2_decrease_weight: float | None = None
    trade_at_level_weight: float | None = None
    unknown_level_queue_ahead_qty: float | None = None
    dedupe_l2_decrease_with_trade_prints: bool | None = None
    maker_fee_bps: float | None = None
    edge_min_executable_edge_bps: float | None = None
    edge_latency_buffer_bps: float | None = None
    edge_inventory_skew_bps_per_unit: float | None = None
    decision_compute_latency_us: int | None = None
    order_entry_latency_us: int | None = None
    cancel_latency_us: int | None = None
    inventory_penalty_bps: float | None = None
    turnover_penalty_bps: float | None = None
    cancel_penalty: float | None = None
    drawdown_penalty_rate: float | None = None
    terminal_inventory_penalty_bps: float | None = None
    reward_scale: float | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.tape_root, "tape_root")
        _require_nonempty_str(self.decision_grid_path, "decision_grid_path")
        _require_nonempty_str(self.split_source_dataset_root, "split_source_dataset_root")
        if self.split not in ("train", "val", "test"):
            raise ValueError('split must be one of "train", "val", or "test"')
        _require_nonempty_str(self.linear_signals_npz, "linear_signals_npz")
        _require_nonempty_str(self.output_json, "output_json")
        if self.adverse_signals_npz is not None:
            _require_nonempty_str(self.adverse_signals_npz, "adverse_signals_npz")
        if self.checkpoint_path is not None:
            _require_nonempty_str(self.checkpoint_path, "checkpoint_path")
        _require_positive_int(self.sample_rows, "sample_rows")
        _require_positive_int(self.num_envs, "num_envs")
        _optional_nonnegative_int(self.seed, "seed")
        if self.sample_policy not in SAMPLE_POLICIES:
            raise ValueError(f"sample_policy must be one of {SAMPLE_POLICIES}")
        if self.sample_policy.startswith("checkpoint_") and self.checkpoint_path is None:
            raise ValueError("checkpoint sample policies require checkpoint_path")
        if self.device is not None:
            _require_nonempty_str(self.device, "device")
        object.__setattr__(self, "dtype", _coerce_dtype(self.dtype))
        object.__setattr__(self, "stdout_mode", validate_stdout_mode(self.stdout_mode))
        _require_bool(self.overwrite, "overwrite")
        for key in ("cancel_guard_ticks", "max_distance_ticks", "post_only_gap_ticks"):
            value = getattr(self, key)
            if value is not None:
                _require_positive_int(value, key)
        _optional_positive_int(self.max_episode_steps, "max_episode_steps")
        for key in (
            "max_order_qty",
            "default_order_qty",
            "l2_decrease_weight",
            "trade_at_level_weight",
            "unknown_level_queue_ahead_qty",
            "maker_fee_bps",
            "edge_min_executable_edge_bps",
            "edge_latency_buffer_bps",
            "edge_inventory_skew_bps_per_unit",
            "inventory_penalty_bps",
            "turnover_penalty_bps",
            "cancel_penalty",
            "drawdown_penalty_rate",
            "terminal_inventory_penalty_bps",
            "reward_scale",
        ):
            _optional_finite_float(getattr(self, key), key)
        for key in ("decision_compute_latency_us", "order_entry_latency_us", "cancel_latency_us"):
            _optional_nonnegative_int(getattr(self, key), key)
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        if self.dedupe_l2_decrease_with_trade_prints is not None:
            _require_bool(self.dedupe_l2_decrease_with_trade_prints, "dedupe_l2_decrease_with_trade_prints")


def _summary_config(config: ExecutionObservationProfileConfig, env_raw: Mapping[str, object]) -> dict[str, object]:
    return {
        "tape_root": config.tape_root,
        "decision_grid_path": config.decision_grid_path,
        "split_source_dataset_root": config.split_source_dataset_root,
        "split": config.split,
        "linear_signals_npz": config.linear_signals_npz,
        "adverse_signals_npz": config.adverse_signals_npz,
        "checkpoint_path": config.checkpoint_path,
        "sample_rows": config.sample_rows,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "sample_policy": config.sample_policy,
        "device": config.device,
        "dtype": str(config.dtype),
        "stdout_mode": config.stdout_mode,
        "overwrite": config.overwrite,
        **{key: _json_safe_env_value(value) for key, value in env_raw.items() if key in _ENV_OVERRIDE_KEYS},
    }


def _json_safe_env_value(value: object) -> object:
    if isinstance(value, QueueModelMode):
        return value.value
    return value


def _env_override_raw(config: ExecutionObservationProfileConfig) -> dict[str, object]:
    raw: dict[str, object] = {}
    for key in _ENV_OVERRIDE_KEYS:
        value = getattr(config, key)
        if value is not None:
            raw[key] = value.value if isinstance(value, QueueModelMode) else value
    return raw


def _env_raw_from_config_defaults(config: ExecutionObservationProfileConfig) -> dict[str, object]:
    raw = dict(_ENV_DEFAULTS)
    raw.update(_env_override_raw(config))
    raw["adverse_signals_npz"] = config.adverse_signals_npz
    return raw


def _build_env_config_from_raw(raw: Mapping[str, object], *, adverse_signals_enabled: bool) -> ExecutionEnvConfig:
    params = ExecutionEnvConfigBuildInput(
        cancel_guard_ticks=int(raw.get("cancel_guard_ticks", _ENV_DEFAULTS["cancel_guard_ticks"])),
        max_distance_ticks=int(raw.get("max_distance_ticks", _ENV_DEFAULTS["max_distance_ticks"])),
        max_order_qty=float(raw.get("max_order_qty", _ENV_DEFAULTS["max_order_qty"])),
        post_only_gap_ticks=int(raw.get("post_only_gap_ticks", _ENV_DEFAULTS["post_only_gap_ticks"])),
        default_order_qty=float(raw.get("default_order_qty", _ENV_DEFAULTS["default_order_qty"])),
        queue_mode=raw.get("queue_mode", _ENV_DEFAULTS["queue_mode"]),
        l2_decrease_weight=float(raw.get("l2_decrease_weight", _ENV_DEFAULTS["l2_decrease_weight"])),
        trade_at_level_weight=float(raw.get("trade_at_level_weight", _ENV_DEFAULTS["trade_at_level_weight"])),
        unknown_level_queue_ahead_qty=float(raw.get("unknown_level_queue_ahead_qty", _ENV_DEFAULTS["unknown_level_queue_ahead_qty"])),
        dedupe_l2_decrease_with_trade_prints=bool(raw.get("dedupe_l2_decrease_with_trade_prints", _ENV_DEFAULTS["dedupe_l2_decrease_with_trade_prints"])),
        maker_fee_bps=float(raw.get("maker_fee_bps", _ENV_DEFAULTS["maker_fee_bps"])),
        edge_min_executable_edge_bps=float(raw.get("edge_min_executable_edge_bps", _ENV_DEFAULTS["edge_min_executable_edge_bps"])),
        edge_latency_buffer_bps=float(raw.get("edge_latency_buffer_bps", _ENV_DEFAULTS["edge_latency_buffer_bps"])),
        edge_inventory_skew_bps_per_unit=float(raw.get("edge_inventory_skew_bps_per_unit", _ENV_DEFAULTS["edge_inventory_skew_bps_per_unit"])),
        decision_compute_latency_us=int(raw.get("decision_compute_latency_us", _ENV_DEFAULTS["decision_compute_latency_us"])),
        order_entry_latency_us=int(raw.get("order_entry_latency_us", _ENV_DEFAULTS["order_entry_latency_us"])),
        cancel_latency_us=int(raw.get("cancel_latency_us", _ENV_DEFAULTS["cancel_latency_us"])),
        inventory_penalty_bps=float(raw.get("inventory_penalty_bps", _ENV_DEFAULTS["inventory_penalty_bps"])),
        turnover_penalty_bps=float(raw.get("turnover_penalty_bps", _ENV_DEFAULTS["turnover_penalty_bps"])),
        cancel_penalty=float(raw.get("cancel_penalty", _ENV_DEFAULTS["cancel_penalty"])),
        drawdown_penalty_rate=float(raw.get("drawdown_penalty_rate", _ENV_DEFAULTS["drawdown_penalty_rate"])),
        terminal_inventory_penalty_bps=float(raw.get("terminal_inventory_penalty_bps", _ENV_DEFAULTS["terminal_inventory_penalty_bps"])),
        reward_scale=float(raw.get("reward_scale", _ENV_DEFAULTS["reward_scale"])),
        max_episode_steps=raw.get("max_episode_steps"),
        adverse_signals_enabled=adverse_signals_enabled,
    )
    return build_execution_env_config_from_input(params)


def _checkpoint_adverse_path(checkpoint: Mapping[str, object] | None, config: ExecutionObservationProfileConfig) -> str | None:
    if config.adverse_signals_npz is not None:
        return config.adverse_signals_npz
    if checkpoint is None:
        return None
    cli_config = checkpoint.get("cli_config")
    if isinstance(cli_config, Mapping):
        value = cli_config.get("adverse_signals_npz")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _env_config_for_profile(
    *,
    config: ExecutionObservationProfileConfig,
    checkpoint: Mapping[str, object] | None,
    adverse_signals_path: str | None,
) -> tuple[ExecutionEnvConfig, dict[str, object], str]:
    overrides = _env_override_raw(config)
    if checkpoint is not None:
        checkpoint_cli = _require_mapping(checkpoint.get("cli_config"), "checkpoint cli_config")
        env_raw = dict(checkpoint_cli)
        env_raw.update(overrides)
        env_raw["adverse_signals_npz"] = adverse_signals_path
        env_config = _env_config_from_training_cli_config(env_raw)
        source = "checkpoint_cli_config"
    else:
        env_raw = _env_raw_from_config_defaults(replace(config, adverse_signals_npz=adverse_signals_path))
        env_config = _build_env_config_from_raw(env_raw, adverse_signals_enabled=adverse_signals_path is not None)
        source = "profile_cli_config"
    if checkpoint is not None:
        schema_payload = _require_mapping(checkpoint.get("observation_schema"), "checkpoint observation_schema")
        env_config = replace(env_config, observation_schema=ObservationSchema.from_dict(dict(schema_payload)))
    return env_config, env_raw, source


def _adverse_signal_summary(artifact: AdverseSelectionSignalArtifact | None, path: str | None) -> dict[str, object] | None:
    if artifact is None:
        return None
    return {
        "path": path,
        "schema": artifact.schema,
        "decision_grid_schema": artifact.decision_grid_schema,
        "decision_grid_hash": artifact.decision_grid_hash,
        "decision_grid_n_rows": artifact.decision_grid_n_rows,
        "decision_schedule": dict(artifact.decision_schedule),
        "target_names": list(artifact.target_names),
        "n_rows": int(artifact.decision_grid_n_rows),
        "adverse_label_config": dict(artifact.adverse_label_config),
    }


def _split_contract_summary(contract: Mapping[str, object], split: str) -> dict[str, object]:
    return {
        "schema": contract["schema"],
        "decision_grid_hash": contract["decision_grid_hash"],
        "decision_grid_n_rows": contract["decision_grid_n_rows"],
        "row_counts_by_split": contract["row_counts_by_split"],
        "selected_split": split,
    }


def _policy_action(policy: str, step_index: int, *, action_size_raw: float) -> QuoteAction:
    if policy == "no_quote":
        bid_enabled, ask_enabled = False, False
    elif policy == "two_sided":
        bid_enabled, ask_enabled = True, True
    elif policy == "alternate_bid_ask":
        bid_enabled, ask_enabled = (True, False) if step_index % 2 == 0 else (False, True)
    else:
        raise ValueError("fixed policy expected")
    return QuoteAction(
        bid_enabled=bid_enabled,
        ask_enabled=ask_enabled,
        bid_price_raw=0.0,
        ask_price_raw=0.0,
        bid_size_raw=action_size_raw,
        ask_size_raw=action_size_raw,
    )


def _tensor_from_obs(obs: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(obs, device=device, dtype=dtype).reshape(1, -1)


def _collect_raw_observations(
    *,
    envs: Sequence[ExecutionEnv],
    ranges,
    config: ExecutionObservationProfileConfig,
    policy,
    checkpoint_normalizer: ObservationNormalizer | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, dict[str, object]]:
    sampler = TrainWindowSampler(ranges, mode="stratified_random", seed=config.seed, num_envs=config.num_envs)
    target_rows = min(config.sample_rows, int(sum(item.rollout_step_capacity for item in ranges)))
    if target_rows <= 0:
        raise ValueError("selected split has no sampleable rows")
    observations: list[np.ndarray] = []
    current_obs: list[np.ndarray] = []
    current_ranges = []
    for env_index, env in enumerate(envs):
        row, split_range = sampler.sample(env_index)
        reset = env.reset(start_decision_row=row)
        current_obs.append(np.array(reset.observation, copy=True))
        current_ranges.append(split_range)

    step_index = 0
    while len(observations) < target_rows:
        for env_index, env in enumerate(envs):
            if len(observations) >= target_rows:
                break
            obs = current_obs[env_index]
            observations.append(np.array(obs, copy=True))
            if len(observations) >= target_rows:
                break
            if config.sample_policy.startswith("checkpoint_"):
                assert policy is not None
                obs_tensor = _tensor_from_obs(obs, device=device, dtype=dtype)
                if checkpoint_normalizer is not None:
                    obs_tensor = checkpoint_normalizer.normalize(obs_tensor)
                with torch.no_grad():
                    policy_action = policy.sample_action(
                        obs_tensor,
                        deterministic=config.sample_policy == "checkpoint_deterministic",
                    )
                action = policy_action.action.detach().cpu().numpy()[0]
            else:
                action = _policy_action(
                    config.sample_policy,
                    step_index,
                    action_size_raw=100.0,
                )
            step = env.step(action)
            current_obs[env_index] = np.array(step.observation, copy=True)
            state = env._state
            split_range = current_ranges[env_index]
            exhausted_range = state is None or state.decision_row_index + 1 >= split_range.end_decision_row
            if step.done or step.truncated or exhausted_range:
                row, split_range = sampler.sample(env_index)
                reset = env.reset(start_decision_row=row)
                current_obs[env_index] = np.array(reset.observation, copy=True)
                current_ranges[env_index] = split_range
            step_index += 1
    stats = sampler.stats_since(0)
    start_rows = stats.get("sampled_start_decision_row_min"), stats.get("sampled_start_decision_row_max")
    sample_summary = {
        "split": config.split,
        "sample_rows_requested": config.sample_rows,
        "sample_rows_collected": len(observations),
        "sample_policy": config.sample_policy,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "start_row_min": start_rows[0],
        "start_row_max": start_rows[1],
        "range_count_visited": stats.get("unique_train_ranges_visited", 0),
    }
    return np.asarray(observations, dtype=envs[0].config.observation_schema.np_dtype), sample_summary


def _fit_sample_normalizer(raw_observations: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> ObservationNormalizer:
    normalizer = ObservationNormalizer(
        obs_shape=raw_observations.shape[1],
        config=ObservationNormalizerConfig(enabled=True, update=False, clip=10.0),
    )
    obs_tensor = torch.as_tensor(raw_observations, device=device, dtype=dtype)
    normalizer.to(device=device, dtype=dtype)
    normalizer.update(obs_tensor)
    return normalizer


def _normalized_observations(
    raw_observations: np.ndarray,
    *,
    normalizer: ObservationNormalizer,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(raw_observations, device=device, dtype=dtype)
        normalized = normalizer.normalize(obs_tensor)
    return normalized.detach().cpu().numpy()


def _finite_stats(values: np.ndarray, *, thresholds: tuple[float, ...]) -> dict[str, object]:
    values64 = np.asarray(values, dtype=np.float64)
    count = int(values64.size)
    finite = values64[np.isfinite(values64)]
    finite_count = int(finite.size)
    out: dict[str, object] = {
        "count": count,
        "finite_count": finite_count,
        "nonfinite_count": count - finite_count,
        "nan_count": int(np.count_nonzero(np.isnan(values64))),
        "posinf_count": int(np.count_nonzero(np.isposinf(values64))),
        "neginf_count": int(np.count_nonzero(np.isneginf(values64))),
    }
    if finite_count:
        percentiles = np.percentile(finite, [1, 5, 10, 50, 90, 95, 99])
        out.update(
            {
                "min": float(np.min(finite)),
                "p01": float(percentiles[0]),
                "p05": float(percentiles[1]),
                "p10": float(percentiles[2]),
                "p50": float(percentiles[3]),
                "p90": float(percentiles[4]),
                "p95": float(percentiles[5]),
                "p99": float(percentiles[6]),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "zero_fraction": float(np.count_nonzero(finite == 0.0) / count),
                "abs_p99": float(np.percentile(np.abs(finite), 99)),
            }
        )
        for threshold in thresholds:
            out[f"abs_gt_{int(threshold)}_fraction"] = float(np.count_nonzero(np.abs(finite) > threshold) / count)
    else:
        for key in ("min", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max", "mean", "std", "zero_fraction", "abs_p99"):
            out[key] = None
        for threshold in thresholds:
            out[f"abs_gt_{int(threshold)}_fraction"] = None
    return out


def _normalizer_arrays(normalizer: ObservationNormalizer) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    running = normalizer.running
    mean = running.mean.detach().cpu().numpy().reshape(-1).astype(np.float64)
    var = running.var.detach().cpu().numpy().reshape(-1).astype(np.float64)
    std = np.sqrt(var + float(normalizer.config.epsilon))
    count = float(running.count.detach().cpu().item())
    return mean, var, std, count


def _field_group_lookup(field_names: Sequence[str]) -> dict[str, str]:
    groups = observation_field_groups()
    lookup: dict[str, str] = {}
    for group, names in groups.items():
        for name in names:
            lookup.setdefault(name, group)
    return {name: lookup.get(name, "custom") for name in field_names}


def _field_statistics(
    *,
    raw_observations: np.ndarray,
    normalized_observations: np.ndarray,
    schema: ObservationSchema,
    normalizer: ObservationNormalizer,
    normalization_source: str,
) -> list[dict[str, object]]:
    mean, var, std, count = _normalizer_arrays(normalizer)
    groups = _field_group_lookup(schema.field_names)
    clip = normalizer.config.clip
    field_stats: list[dict[str, object]] = []
    for index, name in enumerate(schema.field_names):
        raw_stats = _finite_stats(raw_observations[:, index], thresholds=(1.0, 10.0, 100.0))
        norm_stats = _finite_stats(normalized_observations[:, index], thresholds=(5.0, 10.0))
        if clip is not None:
            values = normalized_observations[:, index]
            finite = values[np.isfinite(values)]
            norm_stats["clip_saturation_fraction"] = (
                float(np.count_nonzero(np.abs(finite) >= (clip - 1e-6)) / max(values.size, 1))
                if finite.size
                else 0.0
            )
        else:
            norm_stats["clip_saturation_fraction"] = 0.0
        norm_stats["enabled"] = True
        norm_stats["source"] = normalization_source
        warnings = _field_warnings(raw_stats, norm_stats)
        field_stats.append(
            {
                "name": name,
                "index": index,
                "group": groups[name],
                "raw": raw_stats,
                "normalized": norm_stats,
                "normalizer": {
                    "mean": float(mean[index]),
                    "std": float(std[index]),
                    "var": float(var[index]),
                    "count": count,
                },
                "warnings": warnings,
            }
        )
    return field_stats


def _field_warnings(raw_stats: Mapping[str, object], norm_stats: Mapping[str, object]) -> list[str]:
    warnings: list[str] = []
    raw_nonfinite = int(raw_stats.get("nonfinite_count") or 0)
    norm_nonfinite = int(norm_stats.get("nonfinite_count") or 0)
    raw_std = raw_stats.get("std")
    zero_fraction = float(raw_stats.get("zero_fraction") or 0.0)
    abs_gt_100 = float(raw_stats.get("abs_gt_100_fraction") or 0.0)
    norm_std = norm_stats.get("std")
    clip_fraction = float(norm_stats.get("clip_saturation_fraction") or 0.0)
    mostly_zero = zero_fraction >= 0.999
    if raw_nonfinite > 0:
        warnings.append("raw_nonfinite")
    if norm_nonfinite > 0:
        warnings.append("normalized_nonfinite")
    if raw_std is not None and float(raw_std) <= 1e-12:
        warnings.append("near_constant")
    if mostly_zero:
        warnings.append("mostly_zero")
    if abs_gt_100 >= 0.01:
        warnings.append("high_raw_magnitude")
    if clip_fraction >= 0.01:
        warnings.append("high_normalized_clip")
    if norm_std is not None and not mostly_zero and (float(norm_std) < 0.05 or float(norm_std) > 20.0):
        warnings.append("normalized_scale_suspicious")
    return warnings


def _group_summary(field_stats: Sequence[Mapping[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    groups = sorted(set(str(item["group"]) for item in field_stats))
    for group in groups:
        rows = [item for item in field_stats if item["group"] == group]
        out[group] = {
            "field_count": len(rows),
            "raw_nonfinite_count": int(sum(int(_map(item, "raw").get("nonfinite_count") or 0) for item in rows)),
            "max_abs_raw_p99": _max_float(_map(item, "raw").get("abs_p99") for item in rows),
            "max_normalized_abs_p99": _max_float(_map(item, "normalized").get("abs_p99") for item in rows),
            "max_clip_saturation_fraction": _max_float(_map(item, "normalized").get("clip_saturation_fraction") for item in rows),
            "near_constant_field_count": int(sum("near_constant" in item.get("warnings", []) for item in rows)),
            "warning_field_count": int(sum(bool(item.get("warnings")) for item in rows)),
            "top_warning_fields": _top_warning_fields(rows),
        }
    return out


def _map(item: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = item.get(key)
    return value if isinstance(value, Mapping) else {}


def _max_float(values) -> float:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else 0.0


def _top_warning_fields(rows: Sequence[Mapping[str, object]], *, limit: int = 5) -> list[str]:
    out: list[str] = []
    for item in rows:
        warnings = item.get("warnings")
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) and warnings:
            out.append(f"{item['name']}[{','.join(str(w) for w in warnings)}]")
    return out[:limit]


def _overall_summary(field_stats: Sequence[Mapping[str, object]], *, sample_rows: int, normalization_source: str) -> dict[str, object]:
    warning_rows = [item for item in field_stats if item.get("warnings")]
    clip_sorted = sorted(
        field_stats,
        key=lambda item: float(_map(item, "normalized").get("clip_saturation_fraction") or 0.0),
        reverse=True,
    )
    magnitude_sorted = sorted(
        field_stats,
        key=lambda item: float(_map(item, "raw").get("abs_p99") or 0.0),
        reverse=True,
    )
    return {
        "sample_rows": sample_rows,
        "field_count": len(field_stats),
        "normalization_source": normalization_source,
        "raw_nonfinite_count": int(sum(int(_map(item, "raw").get("nonfinite_count") or 0) for item in field_stats)),
        "normalized_nonfinite_count": int(sum(int(_map(item, "normalized").get("nonfinite_count") or 0) for item in field_stats)),
        "max_clip_saturation_fraction": _max_float(_map(item, "normalized").get("clip_saturation_fraction") for item in field_stats),
        "near_constant_field_count": int(sum("near_constant" in item.get("warnings", []) for item in field_stats)),
        "warning_field_count": len(warning_rows),
        "top_warning_fields": _top_warning_fields(warning_rows, limit=10),
        "top_clip_fields": [
            f"{item['name']}={float(_map(item, 'normalized').get('clip_saturation_fraction') or 0.0):.6g}"
            for item in clip_sorted[:5]
        ],
        "top_raw_magnitude_fields": [
            f"{item['name']}={float(_map(item, 'raw').get('abs_p99') or 0.0):.6g}"
            for item in magnitude_sorted[:5]
        ],
    }


def run_execution_observation_profile(config: ExecutionObservationProfileConfig) -> dict[str, object]:
    if not isinstance(config, ExecutionObservationProfileConfig):
        raise ValueError("config must be ExecutionObservationProfileConfig")

    output_json = Path(config.output_json)
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(str(output_json))

    device = resolve_torch_device(config.device)
    dtype = config.dtype
    checkpoint = _load_checkpoint(config.checkpoint_path, device=device) if config.checkpoint_path is not None else None
    adverse_signals_path = _checkpoint_adverse_path(checkpoint, config)

    tape = load_execution_tape(
        config.tape_root,
        mmap_mode="r",
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    decision_grid = load_decision_grid(config.decision_grid_path)
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    linear_signals = load_linear_signal_artifact_npz(config.linear_signals_npz)
    split_contract = load_execution_split_contract(config.split_source_dataset_root, decision_grid).as_dict()
    if checkpoint is not None:
        checkpoint_split_contract = _require_mapping(checkpoint.get("split_contract"), "checkpoint split_contract")
        if not split_contracts_equal(checkpoint_split_contract, split_contract):
            raise ValueError("checkpoint split_contract does not match profile split source")
    ranges = tuple(entry for entry in ranges_for_split(split_contract, config.split) if entry.rollout_step_capacity > 0)
    if not ranges:
        raise ValueError(f"{config.split} split must contain at least one range with two decision rows")
    requested_start_event_index = int(decision_grid.decision_event_index[ranges[0].start_decision_row])
    decision_grid_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_grid=decision_grid,
        requested_start_event_index=requested_start_event_index,
        min_rows=2,
    )
    adverse_signals = load_adverse_selection_signals(adverse_signals_path) if adverse_signals_path is not None else None
    env_config, env_raw, env_config_source = _env_config_for_profile(
        config=config,
        checkpoint=checkpoint,
        adverse_signals_path=adverse_signals_path,
    )
    adverse_queue_config = _adverse_queue_config_compatibility(
        adverse_signals,
        env_config=env_config,
        allow_mismatch=False,
    )
    envs = tuple(
        ExecutionEnv(
            tape,
            config=env_config,
            decision_grid=decision_grid,
            linear_signals=linear_signals,
            adverse_signals=adverse_signals,
        )
        for _ in range(config.num_envs)
    )
    schema = envs[0].config.observation_schema
    if checkpoint is not None and checkpoint.get("observation_schema") != schema.as_dict():
        raise ValueError("checkpoint observation_schema does not match profile env observation_schema")
    policy = None
    checkpoint_normalizer = None
    if checkpoint is not None and config.sample_policy.startswith("checkpoint_"):
        policy = _load_policy_from_checkpoint(checkpoint, obs_dim=schema.dim, device=device, dtype=dtype)
        checkpoint_normalizer = _load_observation_normalizer_from_checkpoint(
            checkpoint,
            obs_dim=schema.dim,
            device=device,
            dtype=dtype,
        )

    raw_observations, sample_summary = _collect_raw_observations(
        envs=envs,
        ranges=ranges,
        config=config,
        policy=policy,
        checkpoint_normalizer=checkpoint_normalizer,
        device=device,
        dtype=dtype,
    )
    if checkpoint is not None:
        normalizer = _load_observation_normalizer_from_checkpoint(
            checkpoint,
            obs_dim=schema.dim,
            device=device,
            dtype=dtype,
        )
        normalization_source = "checkpoint" if normalizer is not None else "sample_fit"
    else:
        normalizer = None
        normalization_source = "sample_fit"
    if normalizer is None:
        normalizer = _fit_sample_normalizer(raw_observations, device=device, dtype=dtype)
    normalized = _normalized_observations(raw_observations, normalizer=normalizer, device=device, dtype=dtype)
    field_stats = _field_statistics(
        raw_observations=raw_observations,
        normalized_observations=normalized,
        schema=schema,
        normalizer=normalizer,
        normalization_source=normalization_source,
    )
    groups = _group_summary(field_stats)
    compact = _overall_summary(field_stats, sample_rows=int(raw_observations.shape[0]), normalization_source=normalization_source)
    normalizer_count = float(normalizer.running.count.detach().cpu().item())
    clip = normalizer.config.clip
    observation_groups = observation_field_groups()
    payload: dict[str, object] = {
        "status": "ok",
        "run_type": "profile_execution_observations",
        "compact_summary": compact,
        "config": _summary_config(config, env_raw),
        "env_config_source": env_config_source,
        "device": torch_device_summary(requested_device=config.device, resolved_device=device),
        "lineage": {
            "decision_grid": {
                "schema": decision_grid.metadata.schema,
                "hash": decision_grid.decision_grid_hash,
                "n_rows": decision_grid.n_rows,
                "schedule": decision_grid.decision_schedule,
                "start": decision_grid_start.as_dict(),
            },
            "linear_signals": linear_signal_artifact_summary(linear_signals, path=config.linear_signals_npz),
            "adverse_signals": _adverse_signal_summary(adverse_signals, adverse_signals_path),
            "split_contract": _split_contract_summary(split_contract, config.split),
            "checkpoint": {
                "path": config.checkpoint_path,
                "schema": None if checkpoint is None else checkpoint.get("schema"),
                "updates_completed": None if checkpoint is None else checkpoint.get("updates_completed"),
                "has_observation_normalizer": checkpoint is not None and checkpoint.get("observation_normalizer_state_dict") is not None,
            },
        },
        "adverse_signal_queue_config": adverse_queue_config,
        "observation_schema": {
            "dtype": schema.dtype,
            "field_count": schema.dim,
            "field_names": list(schema.field_names),
            "field_groups": {
                group: [name for name in names if name in schema.field_names]
                for group, names in observation_groups.items()
            },
        },
        "normalization": {
            "enabled": True,
            "source": normalization_source,
            "clip": clip,
            "normalizer_count": normalizer_count,
        },
        "sample": sample_summary,
        "field_stats": field_stats,
        "group_summary": groups,
        "warnings": list(compact["top_warning_fields"]),
    }
    write_json_atomic(output_json, payload)
    _print_stdout(config.stdout_mode, payload, output_json)
    return payload


def _print_stdout(stdout_mode: str, payload: Mapping[str, object], output_json: Path) -> None:
    if stdout_mode == "none":
        return
    compact = _require_mapping(payload.get("compact_summary"), "compact_summary")
    sample = _require_mapping(payload.get("sample"), "sample")
    normalization = _require_mapping(payload.get("normalization"), "normalization")
    if stdout_mode == "json":
        print(compact_json_line({
            "status": payload.get("status"),
            "run_type": payload.get("run_type"),
            "compact_summary": compact,
            "output_json": str(output_json),
        }))
        return
    print("profile_execution_observations: ok")
    print(
        f"split={sample.get('split')} sample_rows={sample.get('sample_rows_collected')} "
        f"fields={compact.get('field_count')} policy={sample.get('sample_policy')}"
    )
    print(
        f"normalization={normalization.get('source')} clip={normalization.get('clip')} "
        f"raw_nonfinite={compact.get('raw_nonfinite_count')} normalized_nonfinite={compact.get('normalized_nonfinite_count')}"
    )
    print(
        f"warnings: fields={compact.get('warning_field_count')} "
        f"near_constant={compact.get('near_constant_field_count')} "
        f"max_clip={compact.get('max_clip_saturation_fraction')}"
    )
    print("top_clip: " + " ".join(compact.get("top_clip_fields", [])))
    print("top_warnings: " + " ".join(compact.get("top_warning_fields", [])))
    print(f"output_json={output_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile raw and normalized execution PPO observations.")
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--decision-grid", dest="decision_grid_path", required=True)
    parser.add_argument("--split-source-dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--linear-signals-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--adverse-signals-npz")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--sample-rows", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-policy", choices=SAMPLE_POLICIES, default="no_quote")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64", "fp32", "fp64"), default="float32")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--queue-mode", choices=("conservative", "balanced"))
    parser.add_argument("--maker-fee-bps", type=float)
    parser.add_argument("--max-distance-ticks", type=int)
    parser.add_argument("--max-order-qty", type=float)
    parser.add_argument("--default-order-qty", type=float)
    parser.add_argument("--post-only-gap-ticks", type=int)
    parser.add_argument("--decision-compute-latency-us", type=int)
    parser.add_argument("--order-entry-latency-us", type=int)
    parser.add_argument("--cancel-latency-us", type=int)
    parser.add_argument("--l2-decrease-weight", type=float)
    parser.add_argument("--trade-at-level-weight", type=float)
    parser.add_argument("--unknown-level-queue-ahead-qty", type=float)
    parser.add_argument("--stdout-mode", choices=STDOUT_MODES, default="summary")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> ExecutionObservationProfileConfig:
    return ExecutionObservationProfileConfig(
        tape_root=args.tape_root,
        decision_grid_path=args.decision_grid_path,
        split_source_dataset_root=args.split_source_dataset_root,
        split=args.split,
        linear_signals_npz=args.linear_signals_npz,
        output_json=args.output_json,
        adverse_signals_npz=args.adverse_signals_npz,
        checkpoint_path=args.checkpoint_path,
        sample_rows=args.sample_rows,
        num_envs=args.num_envs,
        seed=args.seed,
        sample_policy=args.sample_policy,
        device=args.device,
        dtype=args.dtype,
        stdout_mode=args.stdout_mode,
        overwrite=args.overwrite,
        max_episode_steps=args.max_episode_steps,
        queue_mode=args.queue_mode,
        maker_fee_bps=args.maker_fee_bps,
        max_distance_ticks=args.max_distance_ticks,
        max_order_qty=args.max_order_qty,
        default_order_qty=args.default_order_qty,
        post_only_gap_ticks=args.post_only_gap_ticks,
        decision_compute_latency_us=args.decision_compute_latency_us,
        order_entry_latency_us=args.order_entry_latency_us,
        cancel_latency_us=args.cancel_latency_us,
        l2_decrease_weight=args.l2_decrease_weight,
        trade_at_level_weight=args.trade_at_level_weight,
        unknown_level_queue_ahead_qty=args.unknown_level_queue_ahead_qty,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    run_execution_observation_profile(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
