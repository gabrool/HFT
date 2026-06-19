"""Evaluate an execution PPO checkpoint on an explicit val/test split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mmrt.execution.contracts import QueueModelMode
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.horizon_diagnostics import (
    DEFAULT_HORIZONS_US,
    HorizonDiagnosticsAccumulator,
    HorizonDiagnosticsConfig,
    parse_horizon_diagnostics_us,
)
from mmrt.execution.adverse_signal import AdverseSelectionSignalArtifact, load_adverse_selection_signals
from mmrt.execution.decision_grid import load_decision_grid, validate_decision_grid_for_execution_tape
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.linear_signal import (
    LINEAR_SIGNAL_ARTIFACT_SCHEMA,
    LINEAR_SIGNALS_FILENAME,
    load_linear_signal_artifact_npz,
    linear_signal_artifact_summary,
)
from mmrt.execution.split_contract import (
    load_execution_split_contract,
    ranges_for_split,
    split_contracts_equal,
    validate_split_contract_payload,
)
from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from mmrt.cli.execution_env_config import (
    ExecutionEnvConfigBuildInput,
    build_execution_env_config_from_attrs,
    build_execution_env_config_from_input,
)
from mmrt.cli.output import (
    STDOUT_MODES,
    compact_eval_summary,
    compact_json_line,
    print_human_summary,
    validate_stdout_mode,
    write_json_atomic,
)
from mmrt.rl.device import cuda_memory_summary, resolve_torch_device, torch_device_summary
from mmrt.rl.evaluate import PolicyEvaluationConfig, evaluate_policy
from mmrt.rl.normalization import ObservationNormalizer, ObservationNormalizerConfig
from mmrt.rl.train import PPO_CHECKPOINT_SCHEMA
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork

__all__ = [
    "ExecutionPolicyEvaluationCLIConfig",
    "run_execution_policy_evaluation",
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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite float")
    return out


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_probability(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _coerce_queue_mode(value: QueueModelMode | str) -> QueueModelMode:
    if isinstance(value, QueueModelMode):
        return value
    if isinstance(value, str):
        try:
            return QueueModelMode(value)
        except ValueError as exc:
            raise ValueError(f"queue_mode has invalid value {value!r}") from exc
    raise ValueError("queue_mode must be QueueModelMode or str")


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
    raise ValueError("dtype must be torch.dtype or str")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _mapping_get_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in mapping:
        raise ValueError(f"{key} is required")
    return _require_mapping(mapping[key], key)


def _mapping_get_optional_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _require_mapping(value, key)


@dataclass(frozen=True, slots=True)
class ExecutionPolicyEvaluationCLIConfig:
    tape_root: str
    decision_grid_path: str
    checkpoint_path: str
    split_source_dataset_root: str
    eval_split: str
    output_json: str | None = None
    debug_output_json: str | None = None
    horizon_debug_json: str | None = None
    linear_signals_npz: str | None = None
    adverse_signals_npz: str | None = None
    allow_adverse_queue_config_mismatch: bool = False
    overwrite: bool = False
    mmap_mode: str | None = "r"

    cancel_guard_ticks: int = 2
    max_episode_steps: int | None = None

    max_distance_ticks: int = 1
    max_order_qty: float = 0.001
    post_only_gap_ticks: int = 1
    default_order_qty: float = 0.001

    queue_mode: QueueModelMode | str = QueueModelMode.CONSERVATIVE
    l2_decrease_weight: float = 0.25
    trade_at_level_weight: float = 0.5
    unknown_level_queue_ahead_qty: float = 1_000_000_000.0
    dedupe_l2_decrease_with_trade_prints: bool = True

    maker_fee_bps: float = 0.0
    edge_min_executable_edge_bps: float = 0.0
    edge_latency_buffer_bps: float = 0.0
    edge_inventory_skew_bps_per_unit: float = 0.0

    decision_compute_latency_us: int = 50
    order_entry_latency_us: int = 500
    cancel_latency_us: int = 500

    inventory_penalty_bps: float = 0.0
    turnover_penalty_bps: float = 0.0
    cancel_penalty: float = 0.0
    drawdown_penalty_rate: float = 0.0
    terminal_inventory_penalty_bps: float = 0.0
    reward_scale: float = 1.0

    max_steps: int | None = None
    deterministic: bool = True
    device: str | None = "auto"
    dtype: torch.dtype | str = torch.float32
    include_diagnostics: bool = True
    horizon_diagnostics_enabled: bool = True
    horizon_diagnostics_us: tuple[int, ...] = DEFAULT_HORIZONS_US
    stdout_mode: str = "summary"

    override_env_config: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_str(self.tape_root, "tape_root")
        _require_nonempty_str(self.decision_grid_path, "decision_grid_path")
        _require_nonempty_str(self.checkpoint_path, "checkpoint_path")
        _require_nonempty_str(self.split_source_dataset_root, "split_source_dataset_root")
        if self.eval_split not in ("val", "test"):
            raise ValueError('eval_split must be "val" or "test"')
        if self.output_json is not None:
            _require_nonempty_str(self.output_json, "output_json")
        if self.debug_output_json is not None:
            _require_nonempty_str(self.debug_output_json, "debug_output_json")
        if self.horizon_debug_json is not None:
            _require_nonempty_str(self.horizon_debug_json, "horizon_debug_json")
        if self.linear_signals_npz is not None:
            _require_nonempty_str(self.linear_signals_npz, "linear_signals_npz")
        if self.adverse_signals_npz is not None:
            _require_nonempty_str(self.adverse_signals_npz, "adverse_signals_npz")
        _require_bool(self.allow_adverse_queue_config_mismatch, "allow_adverse_queue_config_mismatch")
        _require_bool(self.overwrite, "overwrite")
        if self.mmap_mode not in (None, "r"):
            raise ValueError('mmap_mode must be None or "r"')

        _require_positive_int(self.cancel_guard_ticks, "cancel_guard_ticks")
        _optional_positive_int(self.max_episode_steps, "max_episode_steps")
        _require_positive_int(self.max_distance_ticks, "max_distance_ticks")
        _require_positive_float(self.max_order_qty, "max_order_qty")
        _require_positive_int(self.post_only_gap_ticks, "post_only_gap_ticks")
        _require_positive_float(self.default_order_qty, "default_order_qty")
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        _require_probability(self.l2_decrease_weight, "l2_decrease_weight")
        _require_probability(self.trade_at_level_weight, "trade_at_level_weight")
        _require_nonnegative_float(
            self.unknown_level_queue_ahead_qty,
            "unknown_level_queue_ahead_qty",
        )
        _require_bool(self.dedupe_l2_decrease_with_trade_prints, "dedupe_l2_decrease_with_trade_prints")
        _require_finite_float(self.maker_fee_bps, "maker_fee_bps")
        _require_finite_float(self.edge_min_executable_edge_bps, "edge_min_executable_edge_bps")
        _require_nonnegative_float(self.edge_latency_buffer_bps, "edge_latency_buffer_bps")
        _require_finite_float(self.edge_inventory_skew_bps_per_unit, "edge_inventory_skew_bps_per_unit")
        _require_nonnegative_int(self.decision_compute_latency_us, "decision_compute_latency_us")
        _require_nonnegative_int(self.order_entry_latency_us, "order_entry_latency_us")
        _require_nonnegative_int(self.cancel_latency_us, "cancel_latency_us")
        _require_nonnegative_float(self.inventory_penalty_bps, "inventory_penalty_bps")
        _require_nonnegative_float(self.turnover_penalty_bps, "turnover_penalty_bps")
        _require_nonnegative_float(self.cancel_penalty, "cancel_penalty")
        _require_nonnegative_float(self.drawdown_penalty_rate, "drawdown_penalty_rate")
        _require_nonnegative_float(
            self.terminal_inventory_penalty_bps,
            "terminal_inventory_penalty_bps",
        )
        _require_positive_float(self.reward_scale, "reward_scale")

        _optional_positive_int(self.max_steps, "max_steps")
        _require_bool(self.deterministic, "deterministic")
        if self.device is not None:
            _require_nonempty_str(self.device, "device")
        object.__setattr__(self, "dtype", _coerce_dtype(self.dtype))
        _require_bool(self.include_diagnostics, "include_diagnostics")
        _require_bool(self.horizon_diagnostics_enabled, "horizon_diagnostics_enabled")
        object.__setattr__(
            self,
            "horizon_diagnostics_us",
            parse_horizon_diagnostics_us(tuple(self.horizon_diagnostics_us)),
        )
        object.__setattr__(self, "stdout_mode", validate_stdout_mode(self.stdout_mode))
        _require_bool(self.override_env_config, "override_env_config")


def _summary_config(config: ExecutionPolicyEvaluationCLIConfig) -> dict[str, object]:
    return {
        "tape_root": config.tape_root,
        "decision_grid_path": config.decision_grid_path,
        "checkpoint_path": config.checkpoint_path,
        "split_source_dataset_root": config.split_source_dataset_root,
        "eval_split": config.eval_split,
        "output_json": config.output_json,
        "debug_output_json": config.debug_output_json,
        "horizon_debug_json": config.horizon_debug_json,
        "linear_signals_npz": config.linear_signals_npz,
        "adverse_signals_npz": config.adverse_signals_npz,
        "allow_adverse_queue_config_mismatch": config.allow_adverse_queue_config_mismatch,
        "overwrite": config.overwrite,
        "mmap_mode": config.mmap_mode,
        "cancel_guard_ticks": config.cancel_guard_ticks,
        "max_episode_steps": config.max_episode_steps,
        "max_distance_ticks": config.max_distance_ticks,
        "max_order_qty": config.max_order_qty,
        "post_only_gap_ticks": config.post_only_gap_ticks,
        "default_order_qty": config.default_order_qty,
        "queue_mode": config.queue_mode.value,
        "l2_decrease_weight": config.l2_decrease_weight,
        "trade_at_level_weight": config.trade_at_level_weight,
        "unknown_level_queue_ahead_qty": config.unknown_level_queue_ahead_qty,
        "dedupe_l2_decrease_with_trade_prints": config.dedupe_l2_decrease_with_trade_prints,
        "maker_fee_bps": config.maker_fee_bps,
        "edge_min_executable_edge_bps": config.edge_min_executable_edge_bps,
        "edge_latency_buffer_bps": config.edge_latency_buffer_bps,
        "edge_inventory_skew_bps_per_unit": config.edge_inventory_skew_bps_per_unit,
        "decision_compute_latency_us": config.decision_compute_latency_us,
        "order_entry_latency_us": config.order_entry_latency_us,
        "cancel_latency_us": config.cancel_latency_us,
        "inventory_penalty_bps": config.inventory_penalty_bps,
        "turnover_penalty_bps": config.turnover_penalty_bps,
        "cancel_penalty": config.cancel_penalty,
        "drawdown_penalty_rate": config.drawdown_penalty_rate,
        "terminal_inventory_penalty_bps": config.terminal_inventory_penalty_bps,
        "reward_scale": config.reward_scale,
        "max_steps": config.max_steps,
        "deterministic": config.deterministic,
        "device": config.device,
        "dtype": str(config.dtype),
        "include_diagnostics": config.include_diagnostics,
        "horizon_diagnostics_enabled": config.horizon_diagnostics_enabled,
        "horizon_diagnostics_us": list(config.horizon_diagnostics_us),
        "stdout_mode": config.stdout_mode,
        "override_env_config": config.override_env_config,
    }


def _env_config_summary_from_raw(raw: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "cancel_guard_ticks",
        "max_episode_steps",
        "max_distance_ticks",
        "max_order_qty",
        "post_only_gap_ticks",
        "default_order_qty",
        "queue_mode",
        "l2_decrease_weight",
        "trade_at_level_weight",
        "unknown_level_queue_ahead_qty",
        "dedupe_l2_decrease_with_trade_prints",
        "maker_fee_bps",
        "edge_min_executable_edge_bps",
        "edge_latency_buffer_bps",
        "edge_inventory_skew_bps_per_unit",
        "decision_compute_latency_us",
        "order_entry_latency_us",
        "cancel_latency_us",
        "inventory_penalty_bps",
        "turnover_penalty_bps",
        "cancel_penalty",
        "drawdown_penalty_rate",
        "terminal_inventory_penalty_bps",
        "reward_scale",
        "adverse_signals_npz",
    )
    return {key: raw.get(key) for key in keys}


def _env_config_summary_from_config(config: ExecutionPolicyEvaluationCLIConfig) -> dict[str, object]:
    raw = _summary_config(config)
    raw["adverse_signals_npz"] = config.adverse_signals_npz
    return _env_config_summary_from_raw(raw)


def _env_config_diff(checkpoint_raw: Mapping[str, object], override: ExecutionPolicyEvaluationCLIConfig) -> dict[str, dict[str, object]]:
    checkpoint_summary = _env_config_summary_from_raw(checkpoint_raw)
    override_summary = _env_config_summary_from_config(override)
    diff: dict[str, dict[str, object]] = {}
    for key, override_value in override_summary.items():
        checkpoint_value = checkpoint_summary.get(key)
        if checkpoint_value != override_value:
            diff[key] = {"checkpoint": checkpoint_value, "override": override_value}
    return diff


def _env_config_from_cli_config(
    config: ExecutionPolicyEvaluationCLIConfig,
) -> ExecutionEnvConfig:
    return build_execution_env_config_from_attrs(
        config,
        adverse_signals_enabled=config.adverse_signals_npz is not None,
    )


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


_ADVERSE_ENV_COMPATIBILITY_KEYS = (
    "queue_mode",
    "l2_decrease_weight",
    "trade_at_level_weight",
    "dedupe_l2_decrease_with_trade_prints",
    "unknown_level_queue_ahead_qty",
    "qty_epsilon",
    "order_entry_latency_us",
    "decision_compute_latency_us",
    "post_only_gap_ticks",
    "order_qty",
)


def _expected_adverse_label_config_from_env(env_config: ExecutionEnvConfig) -> dict[str, object]:
    queue = env_config.fill_simulator_config.queue_model
    return {
        "queue_mode": queue.mode.value,
        "l2_decrease_weight": queue.l2_decrease_weight,
        "trade_at_level_weight": queue.trade_at_level_weight,
        "dedupe_l2_decrease_with_trade_prints": queue.dedupe_l2_decrease_with_trade_prints,
        "unknown_level_queue_ahead_qty": queue.unknown_level_queue_ahead_qty,
        "qty_epsilon": queue.qty_epsilon,
        "order_entry_latency_us": env_config.latency_config.order_entry_latency_us,
        "decision_compute_latency_us": env_config.latency_config.decision_compute_latency_us,
        "post_only_gap_ticks": env_config.quote_geometry_config.post_only_gap_ticks,
        "order_qty": env_config.quote_geometry_config.default_order_qty,
    }


def _adverse_queue_config_compatibility(
    artifact: AdverseSelectionSignalArtifact | None,
    *,
    env_config: ExecutionEnvConfig,
    allow_mismatch: bool,
) -> dict[str, object] | None:
    if artifact is None:
        return None
    expected = _expected_adverse_label_config_from_env(env_config)
    actual = dict(artifact.adverse_label_config)
    mismatches = {
        key: {"expected": expected[key], "actual": actual.get(key)}
        for key in _ADVERSE_ENV_COMPATIBILITY_KEYS
        if actual.get(key) != expected[key]
    }
    status = "match" if not mismatches else "mismatch_allowed" if allow_mismatch else "mismatch"
    result: dict[str, object] = {
        "status": status,
        "allow_mismatch": allow_mismatch,
        "compared_keys": list(_ADVERSE_ENV_COMPATIBILITY_KEYS),
        "label_only_keys": ["fill_horizon_us", "adverse_horizon_us"],
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }
    if mismatches and not allow_mismatch:
        raise ValueError(f"adverse signal queue config mismatch: {mismatches}")
    return result


def _env_config_from_training_cli_config(raw: Mapping[str, object]) -> ExecutionEnvConfig:
    cancel_guard_ticks = _require_positive_int(
        raw.get("cancel_guard_ticks", 2),
        "cancel_guard_ticks",
    )
    max_episode_steps = _optional_positive_int(
        raw.get("max_episode_steps"),
        "max_episode_steps",
    )
    max_distance_ticks = _require_positive_int(
        raw.get("max_distance_ticks", 1),
        "max_distance_ticks",
    )
    max_order_qty = _require_positive_float(raw.get("max_order_qty", 0.001), "max_order_qty")
    post_only_gap_ticks = _require_positive_int(
        raw.get("post_only_gap_ticks", 1),
        "post_only_gap_ticks",
    )
    default_order_qty = _require_positive_float(
        raw.get("default_order_qty", 0.001),
        "default_order_qty",
    )
    queue_mode = _coerce_queue_mode(raw.get("queue_mode", QueueModelMode.CONSERVATIVE))
    l2_decrease_weight = _require_probability(
        raw.get("l2_decrease_weight", 0.25),
        "l2_decrease_weight",
    )
    trade_at_level_weight = _require_probability(
        raw.get("trade_at_level_weight", 0.5),
        "trade_at_level_weight",
    )
    unknown_level_queue_ahead_qty = _require_nonnegative_float(
        raw.get("unknown_level_queue_ahead_qty", 1_000_000_000.0),
        "unknown_level_queue_ahead_qty",
    )
    dedupe_l2_decrease_with_trade_prints = _require_bool(
        raw.get("dedupe_l2_decrease_with_trade_prints", True),
        "dedupe_l2_decrease_with_trade_prints",
    )
    maker_fee_bps = _require_finite_float(raw.get("maker_fee_bps", 0.0), "maker_fee_bps")
    inventory_penalty_bps = _require_nonnegative_float(
        raw.get("inventory_penalty_bps", 0.0),
        "inventory_penalty_bps",
    )
    turnover_penalty_bps = _require_nonnegative_float(
        raw.get("turnover_penalty_bps", 0.0),
        "turnover_penalty_bps",
    )
    cancel_penalty = _require_nonnegative_float(raw.get("cancel_penalty", 0.0), "cancel_penalty")
    drawdown_penalty_rate = _require_nonnegative_float(
        raw.get("drawdown_penalty_rate", 0.0),
        "drawdown_penalty_rate",
    )
    terminal_inventory_penalty_bps = _require_nonnegative_float(
        raw.get("terminal_inventory_penalty_bps", 0.0),
        "terminal_inventory_penalty_bps",
    )
    reward_scale = _require_positive_float(raw.get("reward_scale", 1.0), "reward_scale")
    decision_compute_latency_us = _require_nonnegative_int(
        raw.get("decision_compute_latency_us", 50),
        "decision_compute_latency_us",
    )
    order_entry_latency_us = _require_nonnegative_int(
        raw.get("order_entry_latency_us", 500),
        "order_entry_latency_us",
    )
    cancel_latency_us = _require_nonnegative_int(
        raw.get("cancel_latency_us", 500),
        "cancel_latency_us",
    )

    params = ExecutionEnvConfigBuildInput(
        cancel_guard_ticks=cancel_guard_ticks,
        max_distance_ticks=max_distance_ticks,
        max_order_qty=max_order_qty,
        post_only_gap_ticks=post_only_gap_ticks,
        default_order_qty=default_order_qty,
        queue_mode=queue_mode,
        l2_decrease_weight=l2_decrease_weight,
        trade_at_level_weight=trade_at_level_weight,
        unknown_level_queue_ahead_qty=unknown_level_queue_ahead_qty,
        dedupe_l2_decrease_with_trade_prints=dedupe_l2_decrease_with_trade_prints,
        maker_fee_bps=maker_fee_bps,
        edge_min_executable_edge_bps=_require_finite_float(raw.get("edge_min_executable_edge_bps", 0.0), "edge_min_executable_edge_bps"),
        edge_latency_buffer_bps=_require_nonnegative_float(raw.get("edge_latency_buffer_bps", 0.0), "edge_latency_buffer_bps"),
        edge_inventory_skew_bps_per_unit=_require_finite_float(raw.get("edge_inventory_skew_bps_per_unit", 0.0), "edge_inventory_skew_bps_per_unit"),
        decision_compute_latency_us=decision_compute_latency_us,
        order_entry_latency_us=order_entry_latency_us,
        cancel_latency_us=cancel_latency_us,
        inventory_penalty_bps=inventory_penalty_bps,
        turnover_penalty_bps=turnover_penalty_bps,
        cancel_penalty=cancel_penalty,
        drawdown_penalty_rate=drawdown_penalty_rate,
        terminal_inventory_penalty_bps=terminal_inventory_penalty_bps,
        reward_scale=reward_scale,
        max_episode_steps=max_episode_steps,
        adverse_signals_enabled=raw.get("adverse_signals_npz") is not None,
    )
    return build_execution_env_config_from_input(params)


def _load_checkpoint(path: str | Path, *, device: torch.device) -> Mapping[str, object]:
    payload = torch.load(path, map_location=device)
    payload = _require_mapping(payload, "checkpoint payload")
    if payload.get("schema") != PPO_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    _mapping_get_mapping(payload, "config")
    if "policy_state_dict" not in payload:
        raise ValueError("policy_state_dict is required")
    if "split_contract" not in payload:
        raise ValueError("checkpoint missing split_contract")
    validate_split_contract_payload(_mapping_get_mapping(payload, "split_contract"))
    if payload.get("train_split") != "train":
        raise ValueError("checkpoint train_split must be train")
    return payload


def _actor_critic_config_from_checkpoint(
    training_config: Mapping[str, object],
) -> ActorCriticConfig:
    raw = _mapping_get_mapping(training_config, "network_config")
    stale_keys = ("policy_log_std_init",)
    present_stale_keys = [key for key in stale_keys if key in raw]
    if present_stale_keys:
        raise ValueError(f"unsupported stale network_config keys: {present_stale_keys}")
    hidden_sizes_value = raw.get("hidden_sizes", (128, 128))
    if not isinstance(hidden_sizes_value, Sequence) or isinstance(hidden_sizes_value, str):
        raise ValueError("hidden_sizes must be a sequence of positive ints")
    hidden_sizes = tuple(
        _require_positive_int(value, "hidden_sizes item") for value in hidden_sizes_value
    )
    required = (
        "enable_threshold",
        "enable_logit_bias_init",
        "continuous_log_std_init",
        "continuous_log_std_min",
        "continuous_log_std_max",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"checkpoint network_config missing required keys: {missing}")
    return ActorCriticConfig(
        hidden_sizes=hidden_sizes,
        activation=_require_nonempty_str(raw.get("activation", "tanh"), "activation"),
        layer_norm=_require_bool(raw.get("layer_norm", False), "layer_norm"),
        orthogonal_init=_require_bool(
            raw.get("orthogonal_init", True),
            "orthogonal_init",
        ),
        enable_threshold=_require_probability(raw["enable_threshold"], "enable_threshold"),
        enable_logit_bias_init=_require_finite_float(raw["enable_logit_bias_init"], "enable_logit_bias_init"),
        continuous_log_std_init=_require_finite_float(raw["continuous_log_std_init"], "continuous_log_std_init"),
        continuous_log_std_min=_require_finite_float(raw["continuous_log_std_min"], "continuous_log_std_min"),
        continuous_log_std_max=_require_finite_float(raw["continuous_log_std_max"], "continuous_log_std_max"),
        policy_head_gain=_require_positive_float(
            raw.get("policy_head_gain", 0.01),
            "policy_head_gain",
        ),
        value_head_gain=_require_positive_float(
            raw.get("value_head_gain", 1.0),
            "value_head_gain",
        ),
    )


def _observation_normalizer_config_from_checkpoint(
    training_config: Mapping[str, object],
) -> ObservationNormalizerConfig:
    raw = _mapping_get_optional_mapping(training_config, "observation_normalizer_config")
    if raw is None:
        return ObservationNormalizerConfig()
    clip_value = raw.get("clip", 10.0)
    return ObservationNormalizerConfig(
        enabled=_require_bool(raw.get("enabled", True), "enabled"),
        update=_require_bool(raw.get("update", True), "update"),
        epsilon=_require_positive_float(raw.get("epsilon", 1e-8), "epsilon"),
        clip=None if clip_value is None else _require_positive_float(clip_value, "clip"),
        rms_epsilon=_require_positive_float(
            raw.get("rms_epsilon", 1e-4),
            "rms_epsilon",
        ),
    )


def _load_policy_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    obs_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ActorCriticNetwork:
    training_config = _mapping_get_mapping(checkpoint, "config")
    network_config = _actor_critic_config_from_checkpoint(training_config)
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=network_config)
    policy.to(device=device, dtype=dtype)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


def _load_observation_normalizer_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    obs_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ObservationNormalizer | None:
    state_dict = checkpoint.get("observation_normalizer_state_dict")
    if state_dict is None:
        return None
    training_config = _mapping_get_mapping(checkpoint, "config")
    obs_norm_config = _observation_normalizer_config_from_checkpoint(training_config)
    normalizer = ObservationNormalizer(obs_shape=obs_dim, config=obs_norm_config)
    normalizer.load_state_dict(state_dict)
    normalizer.to(device=device, dtype=dtype)
    return normalizer


def _default_output_json(tape_root: str) -> Path:
    return Path(tape_root) / "evaluate_execution_policy_summary.json"


def _default_linear_signals_npz(tape_root: str) -> Path:
    return Path(tape_root) / LINEAR_SIGNALS_FILENAME


def _observation_schema_summary(schema: Mapping[str, object]) -> dict[str, object]:
    field_names = schema.get("field_names")
    field_count = len(field_names) if isinstance(field_names, Sequence) and not isinstance(field_names, (str, bytes)) else None
    return {
        "dtype": schema.get("dtype"),
        "field_count": field_count,
    }


def _split_contract_summary(contract: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": contract["schema"],
        "version": contract["version"],
        "split_source_dataset_root": contract["split_source_dataset_root"],
        "split_source_dataset_id": contract["split_source_dataset_id"],
        "split_source_manifest_hash": contract["split_source_manifest_hash"],
        "decision_grid_schema": contract["decision_grid_schema"],
        "decision_grid_hash": contract["decision_grid_hash"],
        "decision_grid_n_rows": contract["decision_grid_n_rows"],
        "decision_schedule": contract["decision_schedule"],
        "row_counts_by_split": contract["row_counts_by_split"],
    }


def _linear_signal_summary_compact(summary: Mapping[str, object]) -> dict[str, object]:
    metadata = summary.get("metadata")
    metadata_summary: dict[str, object] = {}
    if isinstance(metadata, Mapping):
        metadata_summary = {
            "decision_grid_schema": metadata.get("decision_grid_schema"),
            "decision_grid_hash": metadata.get("decision_grid_hash"),
            "decision_grid_n_rows": metadata.get("decision_grid_n_rows"),
            "decision_schedule": metadata.get("decision_schedule"),
        }
    return {
        "schema": summary.get("schema"),
        "path": summary.get("path"),
        "n_rows": summary.get("n_rows"),
        "dtype": summary.get("dtype"),
        "fields": summary.get("fields"),
        "first_decision_event_index": summary.get("first_decision_event_index"),
        "last_decision_event_index": summary.get("last_decision_event_index"),
        "first_decision_local_ts_us": summary.get("first_decision_local_ts_us"),
        "last_decision_local_ts_us": summary.get("last_decision_local_ts_us"),
        "lineage": metadata_summary,
    }


def _split_summary(contract: Mapping[str, object], split: str, *, include_ranges: bool = False) -> dict[str, object]:
    ranges = [entry.as_dict() for entry in ranges_for_split(contract, split)]
    payload: dict[str, object] = {
        "schema": contract["schema"],
        "version": contract["version"],
        "split_source_dataset_root": contract["split_source_dataset_root"],
        "split_source_dataset_id": contract["split_source_dataset_id"],
        "split_source_manifest_hash": contract["split_source_manifest_hash"],
        "decision_grid_schema": contract["decision_grid_schema"],
        "decision_grid_hash": contract["decision_grid_hash"],
        "decision_grid_n_rows": contract["decision_grid_n_rows"],
        "decision_schedule": contract["decision_schedule"],
        "row_counts_by_split": contract["row_counts_by_split"],
        "eval_split": split,
        "eval_range_count": len(ranges),
        "eval_row_count": int(contract["row_counts_by_split"][split]),  # type: ignore[index]
    }
    if include_ranges:
        payload["ranges_by_split"] = contract["ranges_by_split"]
        payload["eval_ranges"] = ranges
    return payload


def run_execution_policy_evaluation(
    config: ExecutionPolicyEvaluationCLIConfig,
) -> dict[str, object]:
    if not isinstance(config, ExecutionPolicyEvaluationCLIConfig):
        raise ValueError("config must be ExecutionPolicyEvaluationCLIConfig")

    output_json = Path(config.output_json) if config.output_json else _default_output_json(config.tape_root)
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(str(output_json))

    device = resolve_torch_device(config.device)
    dtype = _coerce_dtype(config.dtype)
    checkpoint = _load_checkpoint(config.checkpoint_path, device=device)
    checkpoint_split_contract = validate_split_contract_payload(_mapping_get_mapping(checkpoint, "split_contract"))
    tape = load_execution_tape(
        config.tape_root,
        mmap_mode=config.mmap_mode,
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    linear_signals_path = (
        Path(config.linear_signals_npz)
        if config.linear_signals_npz is not None
        else _default_linear_signals_npz(config.tape_root)
    )
    linear_signals = load_linear_signal_artifact_npz(linear_signals_path)
    decision_grid = load_decision_grid(config.decision_grid_path)
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    split_contract = load_execution_split_contract(config.split_source_dataset_root, decision_grid).as_dict()
    if not split_contracts_equal(checkpoint_split_contract, split_contract):
        raise ValueError("checkpoint split_contract does not match evaluation split source")
    eval_ranges = ranges_for_split(split_contract, config.eval_split)
    adverse_signals = load_adverse_selection_signals(config.adverse_signals_npz) if config.adverse_signals_npz is not None else None

    checkpoint_cli_config: Mapping[str, object] = _mapping_get_mapping(checkpoint, "cli_config")
    if config.override_env_config:
        env_config = _env_config_from_cli_config(config)
        env_config_source = "evaluation_cli_config"
        env_config_diff = _env_config_diff(checkpoint_cli_config, config)
    else:
        env_config = _env_config_from_training_cli_config(checkpoint_cli_config)
        env_config_source = "checkpoint_cli_config"
        env_config_diff = {}
    adverse_queue_config = _adverse_queue_config_compatibility(
        adverse_signals,
        env_config=env_config,
        allow_mismatch=config.allow_adverse_queue_config_mismatch,
    )

    requested_start_event_index = int(decision_grid.decision_event_index[eval_ranges[0].start_decision_row])
    decision_grid_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_grid=decision_grid,
        requested_start_event_index=requested_start_event_index,
        min_rows=2,
    )

    env = ExecutionEnv(tape, config=env_config, decision_grid=decision_grid, linear_signals=linear_signals, adverse_signals=adverse_signals)
    checkpoint_schema = checkpoint.get("observation_schema")
    if checkpoint_schema is None:
        raise ValueError("checkpoint missing observation_schema")
    if checkpoint_schema != env.config.observation_schema.as_dict():
        raise ValueError("checkpoint observation_schema does not match evaluation env observation_schema")
    checkpoint_linear_schema = checkpoint.get("linear_signals")
    if checkpoint_linear_schema is None:
        raise ValueError("checkpoint missing linear_signals metadata")
    checkpoint_linear_schema = _require_mapping(checkpoint_linear_schema, "checkpoint linear_signals")
    if checkpoint_linear_schema.get("schema") != LINEAR_SIGNAL_ARTIFACT_SCHEMA:
        raise ValueError("checkpoint linear signal schema mismatch")
    checkpoint_linear_metadata = checkpoint_linear_schema.get("metadata")
    if checkpoint_linear_metadata is None:
        raise ValueError("checkpoint missing linear signal metadata")
    if checkpoint_linear_schema.get("fields") != linear_signal_artifact_summary(linear_signals)["fields"]:
        raise ValueError("checkpoint linear signal fields mismatch")
    if dict(checkpoint_linear_metadata) != linear_signals.metadata.as_dict():
        raise ValueError("checkpoint linear signal metadata mismatch with loaded artifact")
    checkpoint_grid = _mapping_get_mapping(checkpoint, "decision_grid")
    if checkpoint_grid.get("hash") != decision_grid.decision_grid_hash:
        raise ValueError("checkpoint decision grid hash mismatch")
    if checkpoint_grid.get("schema") != decision_grid.metadata.schema or int(checkpoint_grid.get("n_rows", -1)) != decision_grid.n_rows:
        raise ValueError("checkpoint decision grid metadata mismatch")
    obs_dim = int(env.config.observation_schema.dim)
    policy = _load_policy_from_checkpoint(
        checkpoint,
        obs_dim=obs_dim,
        device=device,
        dtype=dtype,
    )
    observation_normalizer = _load_observation_normalizer_from_checkpoint(
        checkpoint,
        obs_dim=obs_dim,
        device=device,
        dtype=dtype,
    )
    eval_config = PolicyEvaluationConfig(
        max_steps=config.max_steps,
        decision_row_ranges=eval_ranges,
        deterministic=config.deterministic,
        reset_env=True,
        device=device,
        dtype=dtype,
        include_diagnostics=config.include_diagnostics,
    )
    horizon_config = HorizonDiagnosticsConfig(
        enabled=config.horizon_diagnostics_enabled,
        horizons_us=config.horizon_diagnostics_us,
    )
    horizon_accumulator = (
        HorizonDiagnosticsAccumulator.from_execution(
            decision_grid=decision_grid,
            tape=tape,
            linear_signals=linear_signals,
            config=horizon_config,
        )
        if horizon_config.enabled
        else None
    )
    result = evaluate_policy(
        env,
        policy,
        config=eval_config,
        observation_normalizer=observation_normalizer,
        horizon_diagnostics=horizon_accumulator,
    )

    split_lineage = _split_summary(split_contract, config.eval_split)
    full_split_lineage = _split_summary(split_contract, config.eval_split, include_ranges=True)
    evaluation_payload = result.as_dict()
    evaluation_config = evaluation_payload["config"]  # type: ignore[index]
    evaluated_ranges = evaluation_config["evaluated_decision_row_ranges"]  # type: ignore[index]
    horizon_payload = (
        horizon_accumulator.as_dict(include_records=False)
        if horizon_accumulator is not None
        else {
            "enabled": False,
            "horizons_us": list(config.horizon_diagnostics_us),
            "decision_level": {},
            "fill_markouts": {},
            "signal_alignment": {},
            "warnings": [],
        }
    )
    if config.horizon_debug_json is not None and horizon_accumulator is not None:
        write_json_atomic(
            config.horizon_debug_json,
            horizon_accumulator.as_dict(include_records=True),
        )
    linear_summary = linear_signal_artifact_summary(linear_signals, path=str(linear_signals_path))
    observation_schema_full = env.config.observation_schema.as_dict()
    evaluation_payload_primary = dict(evaluation_payload)
    evaluation_payload_primary["config"] = {
        key: value
        for key, value in dict(evaluation_config).items()
        if key not in ("decision_row_ranges", "evaluated_decision_row_ranges")
    }
    evaluation_payload_primary.pop("telemetry", None)
    debug_output_path = Path(config.debug_output_json) if config.debug_output_json is not None else None
    summary = {
        "status": result.status,
        "run_type": "evaluate_execution_policy",
        "compact_summary": {},
        "metrics": evaluation_payload["metrics"],
        "horizon_diagnostics": horizon_payload,
        "eval_split": config.eval_split,
        "tape_root": str(Path(config.tape_root)),
        "decision_grid_path": str(Path(config.decision_grid_path)),
        "split_source_dataset_root": str(Path(config.split_source_dataset_root)),
        "checkpoint_path": str(Path(config.checkpoint_path)),
        "output_json": str(output_json),
        "debug_output_json": None if debug_output_path is None else str(debug_output_path),
        "horizon_debug_json": config.horizon_debug_json,
        "config": _summary_config(config),
        "device": torch_device_summary(requested_device=config.device, resolved_device=device),
        "cuda_memory": cuda_memory_summary(device),
        "env_config_source": env_config_source,
        "env_config_diff": env_config_diff,
        "checkpoint_split_source_matches_current": True,
        "checkpoint": {
            "schema": checkpoint.get("schema"),
            "updates_completed": checkpoint.get("updates_completed"),
            "train_split": checkpoint.get("train_split"),
            "has_observation_normalizer": checkpoint.get("observation_normalizer_state_dict") is not None,
        },
        "tape": {
            "schema": tape.manifest.schema,
            "exchange": tape.manifest.exchange,
            "symbol": tape.manifest.symbol,
            "num_events": tape.manifest.num_events,
            "num_l2_batches": tape.manifest.num_l2_batches,
            "num_trades": tape.manifest.num_trades,
            "start_local_ts_us": tape.manifest.start_local_ts_us,
            "end_local_ts_us": tape.manifest.end_local_ts_us,
            "book_depth": (
                tape.manifest.notes.get("book_depth")
                if tape.manifest.notes is not None
                else None
            ),
        },
        "observation_schema": _observation_schema_summary(observation_schema_full),
        "linear_signals": _linear_signal_summary_compact(linear_summary),
        "adverse_signals": _adverse_signal_summary(adverse_signals, config.adverse_signals_npz),
        "adverse_signal_queue_config": adverse_queue_config,
        "decision_grid_start": decision_grid_start.as_dict(),
        "decision_grid": {
            "schema": decision_grid.metadata.schema,
            "hash": decision_grid.decision_grid_hash,
            "n_rows": decision_grid.n_rows,
            "schedule": decision_grid.decision_schedule,
        },
        "lineage": {
            "decision_grid": {
                "schema": decision_grid.metadata.schema,
                "hash": decision_grid.decision_grid_hash,
                "n_rows": decision_grid.n_rows,
                "schedule": decision_grid.decision_schedule,
            },
            "linear_signals": _linear_signal_summary_compact(linear_summary),
            "adverse_signals": _adverse_signal_summary(adverse_signals, config.adverse_signals_npz),
            "split_contract": _split_contract_summary(split_contract),
        },
        "split_contract": _split_contract_summary(split_contract),
        "checkpoint_split_contract": _split_contract_summary(checkpoint_split_contract),
        "split_lineage": split_lineage,
        "eval_requested_row_count": evaluation_config["eval_requested_row_count"],  # type: ignore[index]
        "eval_covered_row_count": evaluation_config["eval_covered_row_count"],  # type: ignore[index]
        "eval_coverage_fraction": evaluation_config["eval_coverage_fraction"],  # type: ignore[index]
        "evaluated_decision_row_range_count": len(evaluated_ranges),
        "episode_count": evaluation_config["episode_count"],  # type: ignore[index]
        "truncation_counts": evaluation_config["truncation_counts"],  # type: ignore[index]
        "evaluated_start_decision_row": evaluation_config["evaluated_start_decision_row"],  # type: ignore[index]
        "evaluated_end_decision_row": evaluation_config["evaluated_end_decision_row"],  # type: ignore[index]
        "evaluation": evaluation_payload_primary,
        "policy_action_telemetry": evaluation_payload.get("telemetry"),
        "debug": {
            "debug_output_json": None if debug_output_path is None else str(debug_output_path),
            "horizon_debug_json": config.horizon_debug_json,
        },
    }
    summary["compact_summary"] = compact_eval_summary(summary)
    if debug_output_path is not None:
        write_json_atomic(
            debug_output_path,
            {
                "status": result.status,
                "run_type": "evaluate_execution_policy_debug",
                "primary_output_json": str(output_json),
                "config": _summary_config(config),
                "observation_schema": observation_schema_full,
                "linear_signals": linear_summary,
                "adverse_signals": _adverse_signal_summary(adverse_signals, config.adverse_signals_npz),
                "split_contract": split_contract,
                "checkpoint_split_contract": checkpoint_split_contract,
                "split_lineage": full_split_lineage,
                "evaluation": evaluation_payload,
                "policy_action_telemetry": evaluation_payload.get("telemetry"),
            },
        )
    write_json_atomic(output_json, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved execution PPO policy checkpoint on an explicit val/test split."
    )
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--decision-grid", dest="decision_grid_path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--split-source-dataset-root", required=True)
    parser.add_argument("--eval-split", required=True, choices=("val", "test"))

    parser.add_argument("--output-json")
    parser.add_argument("--debug-output-json")
    parser.add_argument("--horizon-debug-json")
    parser.add_argument(
        "--linear-signals-npz",
        help="Canonical no-move-gated linear signal NPZ. Defaults to <tape-root>/linear_signals.npz. Required; missing file is an error.",
    )
    parser.add_argument("--adverse-signals-npz")
    parser.add_argument(
        "--allow-adverse-queue-config-mismatch",
        action="store_true",
        help="Allow adverse signal queue/fill config to differ from the resolved execution env config; mismatches are errors by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")

    parser.add_argument("--cancel-guard-ticks", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--max-distance-ticks", type=int, default=1)
    parser.add_argument("--max-order-qty", type=float, default=0.001)
    parser.add_argument("--post-only-gap-ticks", type=int, default=1)
    parser.add_argument("--default-order-qty", type=float, default=0.001)
    parser.add_argument("--queue-mode", choices=("conservative", "balanced"), default="conservative")
    parser.add_argument("--l2-decrease-weight", type=float, default=0.25)
    parser.add_argument("--trade-at-level-weight", type=float, default=0.5)
    parser.add_argument("--unknown-level-queue-ahead-qty", type=float, default=1000000000.0)
    parser.add_argument(
        "--no-dedupe-l2-decrease-with-trade-prints",
        action="store_true",
        help="Disable de-duplication of L2 visible decreases already explained by same-level trade prints.",
    )
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--edge-min-executable-edge-bps", type=float, default=0.0)
    parser.add_argument("--edge-latency-buffer-bps", type=float, default=0.0)
    parser.add_argument("--edge-inventory-skew-bps-per-unit", type=float, default=0.0)
    parser.add_argument("--decision-compute-latency-us", type=int, default=50)
    parser.add_argument("--order-entry-latency-us", type=int, default=500)
    parser.add_argument("--cancel-latency-us", type=int, default=500)
    parser.add_argument("--inventory-penalty-bps", type=float, default=0.0)
    parser.add_argument("--turnover-penalty-bps", type=float, default=0.0)
    parser.add_argument("--cancel-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty-rate", type=float, default=0.0)
    parser.add_argument("--terminal-inventory-penalty-bps", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)

    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64", "fp32", "fp64"), default="float32")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--horizon-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable compact future-horizon reward/markout diagnostics.",
    )
    parser.add_argument(
        "--horizon-diagnostics-us",
        default="250000,500000,1000000",
        help="Comma-separated positive future horizons in microseconds.",
    )
    parser.add_argument("--stdout-mode", choices=STDOUT_MODES, default="summary")
    parser.add_argument("--override-env-config", action="store_true")

    return parser


def _config_from_args(args: argparse.Namespace) -> ExecutionPolicyEvaluationCLIConfig:
    return ExecutionPolicyEvaluationCLIConfig(
        tape_root=args.tape_root,
        decision_grid_path=args.decision_grid_path,
        checkpoint_path=args.checkpoint_path,
        split_source_dataset_root=args.split_source_dataset_root,
        eval_split=args.eval_split,
        output_json=args.output_json,
        debug_output_json=args.debug_output_json,
        horizon_debug_json=args.horizon_debug_json,
        linear_signals_npz=args.linear_signals_npz,
        adverse_signals_npz=args.adverse_signals_npz,
        allow_adverse_queue_config_mismatch=args.allow_adverse_queue_config_mismatch,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
        cancel_guard_ticks=args.cancel_guard_ticks,
        max_episode_steps=args.max_episode_steps,
        max_distance_ticks=args.max_distance_ticks,
        max_order_qty=args.max_order_qty,
        post_only_gap_ticks=args.post_only_gap_ticks,
        default_order_qty=args.default_order_qty,
        queue_mode=args.queue_mode,
        l2_decrease_weight=args.l2_decrease_weight,
        trade_at_level_weight=args.trade_at_level_weight,
        unknown_level_queue_ahead_qty=args.unknown_level_queue_ahead_qty,
        dedupe_l2_decrease_with_trade_prints=not args.no_dedupe_l2_decrease_with_trade_prints,
        maker_fee_bps=args.maker_fee_bps,
        edge_min_executable_edge_bps=args.edge_min_executable_edge_bps,
        edge_latency_buffer_bps=args.edge_latency_buffer_bps,
        edge_inventory_skew_bps_per_unit=args.edge_inventory_skew_bps_per_unit,
        decision_compute_latency_us=args.decision_compute_latency_us,
        order_entry_latency_us=args.order_entry_latency_us,
        cancel_latency_us=args.cancel_latency_us,
        inventory_penalty_bps=args.inventory_penalty_bps,
        turnover_penalty_bps=args.turnover_penalty_bps,
        cancel_penalty=args.cancel_penalty,
        drawdown_penalty_rate=args.drawdown_penalty_rate,
        terminal_inventory_penalty_bps=args.terminal_inventory_penalty_bps,
        reward_scale=args.reward_scale,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        device=args.device,
        dtype=args.dtype,
        include_diagnostics=not args.no_diagnostics,
        horizon_diagnostics_enabled=args.horizon_diagnostics,
        horizon_diagnostics_us=parse_horizon_diagnostics_us(args.horizon_diagnostics_us),
        stdout_mode=args.stdout_mode,
        override_env_config=args.override_env_config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    summary = run_execution_policy_evaluation(config)
    if config.stdout_mode == "summary":
        print_human_summary("evaluate_execution_policy", summary)
    elif config.stdout_mode == "json":
        print(compact_json_line(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
