"""Evaluate a saved execution PPO policy checkpoint on an execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mmrt.execution.contracts import ActionSpec, PositionState, QueueModelMode
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.execution_tape import load_execution_tape
from mmrt.execution.fill_sim import FillSimulatorConfig
from mmrt.execution.linear_signal import (
    LINEAR_SIGNAL_ARTIFACT_SCHEMA_VERSION,
    LINEAR_SIGNALS_FILENAME,
    load_linear_signal_artifact_npz,
    linear_signal_artifact_summary,
    validate_linear_signal_artifact_metadata,
)
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.execution.quote_geometry import QuoteGeometryConfig
from mmrt.execution.reward import RewardConfig
from mmrt.rl.evaluate import PolicyEvaluationConfig, evaluate_policy
from mmrt.rl.normalization import ObservationNormalizer, ObservationNormalizerConfig
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


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, name)


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
    checkpoint_path: str
    output_json: str | None = None
    linear_signals_npz: str | None = None
    overwrite: bool = False
    mmap_mode: str | None = "r"

    decision_interval_us: int = 500_000
    max_episode_steps: int | None = None

    max_distance_ticks: int = 1
    max_order_qty: float = 0.001
    min_distance_ticks: int = 1
    default_order_qty: float = 0.001

    queue_mode: QueueModelMode | str = QueueModelMode.BALANCED
    l2_decrease_weight: float = 1.0
    trade_at_level_weight: float = 1.0
    unknown_level_queue_ahead_qty: float = 0.0

    maker_fee_bps: float = 0.0

    inventory_penalty_bps: float = 0.0
    turnover_penalty_bps: float = 0.0
    cancel_penalty: float = 0.0
    drawdown_penalty_rate: float = 0.0
    terminal_inventory_penalty_bps: float = 0.0

    max_steps: int | None = None
    start_event_index: int | None = None
    deterministic: bool = True
    device: str | None = "cpu"
    dtype: torch.dtype | str = torch.float32
    include_diagnostics: bool = True

    use_checkpoint_cli_env_config: bool = True

    def __post_init__(self) -> None:
        _require_nonempty_str(self.tape_root, "tape_root")
        _require_nonempty_str(self.checkpoint_path, "checkpoint_path")
        if self.output_json is not None:
            _require_nonempty_str(self.output_json, "output_json")
        if self.linear_signals_npz is not None:
            _require_nonempty_str(self.linear_signals_npz, "linear_signals_npz")
        _require_bool(self.overwrite, "overwrite")
        if self.mmap_mode not in (None, "r"):
            raise ValueError('mmap_mode must be None or "r"')

        _require_positive_int(self.decision_interval_us, "decision_interval_us")
        _optional_positive_int(self.max_episode_steps, "max_episode_steps")
        _require_positive_int(self.max_distance_ticks, "max_distance_ticks")
        _require_positive_float(self.max_order_qty, "max_order_qty")
        _require_positive_int(self.min_distance_ticks, "min_distance_ticks")
        _require_positive_float(self.default_order_qty, "default_order_qty")
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        _require_probability(self.l2_decrease_weight, "l2_decrease_weight")
        _require_probability(self.trade_at_level_weight, "trade_at_level_weight")
        _require_nonnegative_float(
            self.unknown_level_queue_ahead_qty,
            "unknown_level_queue_ahead_qty",
        )
        _require_nonnegative_float(self.maker_fee_bps, "maker_fee_bps")
        _require_nonnegative_float(self.inventory_penalty_bps, "inventory_penalty_bps")
        _require_nonnegative_float(self.turnover_penalty_bps, "turnover_penalty_bps")
        _require_nonnegative_float(self.cancel_penalty, "cancel_penalty")
        _require_nonnegative_float(self.drawdown_penalty_rate, "drawdown_penalty_rate")
        _require_nonnegative_float(
            self.terminal_inventory_penalty_bps,
            "terminal_inventory_penalty_bps",
        )

        _optional_positive_int(self.max_steps, "max_steps")
        _optional_nonnegative_int(self.start_event_index, "start_event_index")
        _require_bool(self.deterministic, "deterministic")
        if self.device is not None:
            _require_nonempty_str(self.device, "device")
        object.__setattr__(self, "dtype", _coerce_dtype(self.dtype))
        _require_bool(self.include_diagnostics, "include_diagnostics")
        _require_bool(
            self.use_checkpoint_cli_env_config,
            "use_checkpoint_cli_env_config",
        )


def _summary_config(config: ExecutionPolicyEvaluationCLIConfig) -> dict[str, object]:
    return {
        "tape_root": config.tape_root,
        "checkpoint_path": config.checkpoint_path,
        "output_json": config.output_json,
        "linear_signals_npz": config.linear_signals_npz,
        "overwrite": config.overwrite,
        "mmap_mode": config.mmap_mode,
        "decision_interval_us": config.decision_interval_us,
        "max_episode_steps": config.max_episode_steps,
        "max_distance_ticks": config.max_distance_ticks,
        "max_order_qty": config.max_order_qty,
        "min_distance_ticks": config.min_distance_ticks,
        "default_order_qty": config.default_order_qty,
        "queue_mode": config.queue_mode.value,
        "l2_decrease_weight": config.l2_decrease_weight,
        "trade_at_level_weight": config.trade_at_level_weight,
        "unknown_level_queue_ahead_qty": config.unknown_level_queue_ahead_qty,
        "maker_fee_bps": config.maker_fee_bps,
        "inventory_penalty_bps": config.inventory_penalty_bps,
        "turnover_penalty_bps": config.turnover_penalty_bps,
        "cancel_penalty": config.cancel_penalty,
        "drawdown_penalty_rate": config.drawdown_penalty_rate,
        "terminal_inventory_penalty_bps": config.terminal_inventory_penalty_bps,
        "max_steps": config.max_steps,
        "start_event_index": config.start_event_index,
        "deterministic": config.deterministic,
        "device": config.device,
        "dtype": str(config.dtype),
        "include_diagnostics": config.include_diagnostics,
        "use_checkpoint_cli_env_config": config.use_checkpoint_cli_env_config,
    }


def _env_config_from_cli_config(
    config: ExecutionPolicyEvaluationCLIConfig,
) -> ExecutionEnvConfig:
    return ExecutionEnvConfig(
        decision_interval_us=config.decision_interval_us,
        action_spec=ActionSpec(
            max_distance_ticks=config.max_distance_ticks,
            max_order_qty=config.max_order_qty,
        ),
        quote_geometry_config=QuoteGeometryConfig(
            min_distance_ticks=config.min_distance_ticks,
            default_order_qty=config.default_order_qty,
        ),
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=config.queue_mode,
                l2_decrease_weight=config.l2_decrease_weight,
                trade_at_level_weight=config.trade_at_level_weight,
                unknown_level_queue_ahead_qty=config.unknown_level_queue_ahead_qty,
            ),
            maker_fee_bps=config.maker_fee_bps,
        ),
        reward_config=RewardConfig(
            inventory_penalty_bps=config.inventory_penalty_bps,
            turnover_penalty_bps=config.turnover_penalty_bps,
            cancel_penalty=config.cancel_penalty,
            drawdown_penalty_rate=config.drawdown_penalty_rate,
            terminal_inventory_penalty_bps=config.terminal_inventory_penalty_bps,
        ),
        initial_position=PositionState(),
        max_episode_steps=config.max_episode_steps,
    )


def _env_config_from_training_cli_config(raw: Mapping[str, object]) -> ExecutionEnvConfig:
    decision_interval_us = _require_positive_int(
        raw.get("decision_interval_us", 500_000),
        "decision_interval_us",
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
    min_distance_ticks = _require_positive_int(
        raw.get("min_distance_ticks", 1),
        "min_distance_ticks",
    )
    default_order_qty = _require_positive_float(
        raw.get("default_order_qty", 0.001),
        "default_order_qty",
    )
    queue_mode = _coerce_queue_mode(raw.get("queue_mode", QueueModelMode.BALANCED))
    l2_decrease_weight = _require_probability(
        raw.get("l2_decrease_weight", 1.0),
        "l2_decrease_weight",
    )
    trade_at_level_weight = _require_probability(
        raw.get("trade_at_level_weight", 1.0),
        "trade_at_level_weight",
    )
    unknown_level_queue_ahead_qty = _require_nonnegative_float(
        raw.get("unknown_level_queue_ahead_qty", 0.0),
        "unknown_level_queue_ahead_qty",
    )
    maker_fee_bps = _require_nonnegative_float(raw.get("maker_fee_bps", 0.0), "maker_fee_bps")
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

    return ExecutionEnvConfig(
        decision_interval_us=decision_interval_us,
        action_spec=ActionSpec(
            max_distance_ticks=max_distance_ticks,
            max_order_qty=max_order_qty,
        ),
        quote_geometry_config=QuoteGeometryConfig(
            min_distance_ticks=min_distance_ticks,
            default_order_qty=default_order_qty,
        ),
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=queue_mode,
                l2_decrease_weight=l2_decrease_weight,
                trade_at_level_weight=trade_at_level_weight,
                unknown_level_queue_ahead_qty=unknown_level_queue_ahead_qty,
            ),
            maker_fee_bps=maker_fee_bps,
        ),
        reward_config=RewardConfig(
            inventory_penalty_bps=inventory_penalty_bps,
            turnover_penalty_bps=turnover_penalty_bps,
            cancel_penalty=cancel_penalty,
            drawdown_penalty_rate=drawdown_penalty_rate,
            terminal_inventory_penalty_bps=terminal_inventory_penalty_bps,
        ),
        initial_position=PositionState(),
        max_episode_steps=max_episode_steps,
    )


def _load_checkpoint(path: str | Path, *, device: torch.device) -> Mapping[str, object]:
    payload = torch.load(path, map_location=device)
    payload = _require_mapping(payload, "checkpoint payload")
    if payload.get("schema_version") != "mmrt_execution_ppo_checkpoint_v2_required_linear_signals":
        raise ValueError("checkpoint schema_version is not mmrt_execution_ppo_checkpoint_v2_required_linear_signals")
    _mapping_get_mapping(payload, "config")
    if "policy_state_dict" not in payload:
        raise ValueError("policy_state_dict is required")
    return payload


def _actor_critic_config_from_checkpoint(
    training_config: Mapping[str, object],
) -> ActorCriticConfig:
    raw = _mapping_get_mapping(training_config, "network_config")
    hidden_sizes_value = raw.get("hidden_sizes", (128, 128))
    if not isinstance(hidden_sizes_value, Sequence) or isinstance(hidden_sizes_value, str):
        raise ValueError("hidden_sizes must be a sequence of positive ints")
    hidden_sizes = tuple(
        _require_positive_int(value, "hidden_sizes item") for value in hidden_sizes_value
    )
    return ActorCriticConfig(
        hidden_sizes=hidden_sizes,
        activation=_require_nonempty_str(raw.get("activation", "tanh"), "activation"),
        layer_norm=_require_bool(raw.get("layer_norm", False), "layer_norm"),
        orthogonal_init=_require_bool(
            raw.get("orthogonal_init", True),
            "orthogonal_init",
        ),
        policy_log_std_init=_require_finite_float(
            raw.get("policy_log_std_init", -0.5),
            "policy_log_std_init",
        ),
        policy_log_std_min=_require_finite_float(
            raw.get("policy_log_std_min", -5.0),
            "policy_log_std_min",
        ),
        policy_log_std_max=_require_finite_float(
            raw.get("policy_log_std_max", 2.0),
            "policy_log_std_max",
        ),
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


def _effective_start_event_index(value: int | None) -> int:
    return 0 if value is None else value


def _resolve_evaluation_start_event_index(
    *,
    config_start_event_index: int | None,
    checkpoint_cli_config: Mapping[str, object] | None,
) -> int | None:
    if config_start_event_index is not None:
        return _require_nonnegative_int(config_start_event_index, "start_event_index")
    if checkpoint_cli_config is None:
        return None
    return _optional_nonnegative_int(
        checkpoint_cli_config.get("start_event_index"),
        "checkpoint cli_config start_event_index",
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_execution_policy_evaluation(
    config: ExecutionPolicyEvaluationCLIConfig,
) -> dict[str, object]:
    if not isinstance(config, ExecutionPolicyEvaluationCLIConfig):
        raise ValueError("config must be ExecutionPolicyEvaluationCLIConfig")

    output_json = Path(config.output_json) if config.output_json else _default_output_json(config.tape_root)
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(str(output_json))

    device = torch.device(config.device) if config.device is not None else torch.device("cpu")
    dtype = _coerce_dtype(config.dtype)
    checkpoint = _load_checkpoint(config.checkpoint_path, device=device)
    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode)
    linear_signals_path = (
        Path(config.linear_signals_npz)
        if config.linear_signals_npz is not None
        else _default_linear_signals_npz(config.tape_root)
    )
    linear_signals = load_linear_signal_artifact_npz(linear_signals_path)

    checkpoint_cli_config: Mapping[str, object] | None = None

    if config.use_checkpoint_cli_env_config:
        checkpoint_cli_config = _mapping_get_mapping(checkpoint, "cli_config")
        env_config = _env_config_from_training_cli_config(checkpoint_cli_config)
        env_config_source = "checkpoint_cli_config"
    else:
        env_config = _env_config_from_cli_config(config)
        env_config_source = "evaluation_cli_config"

    effective_start_event_index = _resolve_evaluation_start_event_index(
        config_start_event_index=config.start_event_index,
        checkpoint_cli_config=checkpoint_cli_config,
    )

    validate_linear_signal_artifact_metadata(
        linear_signals,
        tape_schema_version=tape.manifest.schema_version,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_interval_us=env_config.decision_interval_us,
        start_event_index=_effective_start_event_index(effective_start_event_index),
        min_rows=(env_config.max_episode_steps + 1) if env_config.max_episode_steps is not None else None,
    )

    env = ExecutionEnv(tape, config=env_config, linear_signals=linear_signals)
    checkpoint_schema = checkpoint.get("observation_schema")
    if checkpoint_schema is None:
        raise ValueError("checkpoint missing observation_schema")
    if checkpoint_schema != env.config.observation_schema.as_dict():
        raise ValueError("checkpoint observation_schema does not match evaluation env observation_schema")
    checkpoint_linear_schema = checkpoint.get("linear_signals")
    if checkpoint_linear_schema is None:
        raise ValueError("checkpoint missing linear_signals metadata")
    checkpoint_linear_schema = _require_mapping(checkpoint_linear_schema, "checkpoint linear_signals")
    if checkpoint_linear_schema.get("schema_version") != LINEAR_SIGNAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("checkpoint linear signal schema mismatch")
    checkpoint_linear_metadata = checkpoint_linear_schema.get("metadata")
    if checkpoint_linear_metadata is None:
        raise ValueError("checkpoint missing linear signal metadata")
    if checkpoint_linear_schema.get("fields") != linear_signal_artifact_summary(linear_signals)["fields"]:
        raise ValueError("checkpoint linear signal fields mismatch")
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
        start_event_index=effective_start_event_index,
        deterministic=config.deterministic,
        reset_env=True,
        device=device,
        dtype=dtype,
        include_diagnostics=config.include_diagnostics,
    )
    result = evaluate_policy(
        env,
        policy,
        config=eval_config,
        observation_normalizer=observation_normalizer,
    )

    summary = {
        "status": result.status,
        "run_type": "evaluate_execution_policy",
        "tape_root": str(Path(config.tape_root)),
        "checkpoint_path": str(Path(config.checkpoint_path)),
        "output_json": str(output_json),
        "config": _summary_config(config),
        "env_config_source": env_config_source,
        "effective_start_event_index": effective_start_event_index,
        "checkpoint": {
            "schema_version": checkpoint.get("schema_version"),
            "updates_completed": checkpoint.get("updates_completed"),
            "has_observation_normalizer": checkpoint.get("observation_normalizer_state_dict")
            is not None,
        },
        "tape": {
            "schema_version": tape.manifest.schema_version,
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
        "observation_schema": env.config.observation_schema.as_dict(),
        "linear_signals": linear_signal_artifact_summary(
            linear_signals, path=str(linear_signals_path)
        ),
        "evaluation": result.as_dict(),
    }
    _write_json_atomic(output_json, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved execution PPO policy checkpoint on an execution tape."
    )
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)

    parser.add_argument("--output-json")
    parser.add_argument(
        "--linear-signals-npz",
        help="Canonical no-move-gated linear signal NPZ. Defaults to <tape-root>/linear_signals.npz. Required; missing file is an error.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")

    parser.add_argument("--decision-interval-us", type=int, default=500_000)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--max-distance-ticks", type=int, default=1)
    parser.add_argument("--max-order-qty", type=float, default=0.001)
    parser.add_argument("--min-distance-ticks", type=int, default=1)
    parser.add_argument("--default-order-qty", type=float, default=0.001)
    parser.add_argument("--queue-mode", choices=("conservative", "balanced"), default="balanced")
    parser.add_argument("--l2-decrease-weight", type=float, default=1.0)
    parser.add_argument("--trade-at-level-weight", type=float, default=1.0)
    parser.add_argument("--unknown-level-queue-ahead-qty", type=float, default=0.0)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--inventory-penalty-bps", type=float, default=0.0)
    parser.add_argument("--turnover-penalty-bps", type=float, default=0.0)
    parser.add_argument("--cancel-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty-rate", type=float, default=0.0)
    parser.add_argument("--terminal-inventory-penalty-bps", type=float, default=0.0)

    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "fp32", "fp64"), default="float32")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--no-use-checkpoint-cli-env-config", action="store_true")

    return parser


def _config_from_args(args: argparse.Namespace) -> ExecutionPolicyEvaluationCLIConfig:
    return ExecutionPolicyEvaluationCLIConfig(
        tape_root=args.tape_root,
        checkpoint_path=args.checkpoint_path,
        output_json=args.output_json,
        linear_signals_npz=args.linear_signals_npz,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
        decision_interval_us=args.decision_interval_us,
        max_episode_steps=args.max_episode_steps,
        max_distance_ticks=args.max_distance_ticks,
        max_order_qty=args.max_order_qty,
        min_distance_ticks=args.min_distance_ticks,
        default_order_qty=args.default_order_qty,
        queue_mode=args.queue_mode,
        l2_decrease_weight=args.l2_decrease_weight,
        trade_at_level_weight=args.trade_at_level_weight,
        unknown_level_queue_ahead_qty=args.unknown_level_queue_ahead_qty,
        maker_fee_bps=args.maker_fee_bps,
        inventory_penalty_bps=args.inventory_penalty_bps,
        turnover_penalty_bps=args.turnover_penalty_bps,
        cancel_penalty=args.cancel_penalty,
        drawdown_penalty_rate=args.drawdown_penalty_rate,
        terminal_inventory_penalty_bps=args.terminal_inventory_penalty_bps,
        max_steps=args.max_steps,
        start_event_index=args.start_event_index,
        deterministic=not args.stochastic,
        device=args.device,
        dtype=args.dtype,
        include_diagnostics=not args.no_diagnostics,
        use_checkpoint_cli_env_config=not args.no_use_checkpoint_cli_env_config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    summary = run_execution_policy_evaluation(config)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
