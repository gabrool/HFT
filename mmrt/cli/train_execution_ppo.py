"""Train an execution PPO policy from an existing execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from mmrt.execution.contracts import QueueModelMode
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.adverse_signal import load_adverse_selection_signals
from mmrt.execution.execution_tape import load_execution_tape
from mmrt.execution.linear_signal import (
    LINEAR_SIGNALS_FILENAME,
    load_linear_signal_artifact_npz,
    linear_signal_artifact_summary,
)
from mmrt.cli.execution_env_config import build_execution_env_config_from_attrs
from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from mmrt.rl.normalization import ObservationNormalizerConfig
from mmrt.rl.ppo import PPOConfig
from mmrt.rl.rollout import RolloutConfig
from mmrt.rl.torch_networks import ActorCriticConfig
from mmrt.rl.train import (
    PPOTrainingConfig,
    make_training_checkpoint_payload,
    train_ppo_policy,
)

__all__ = [
    "ExecutionPPOTrainCLIConfig",
    "run_execution_ppo_training",
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


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, name)


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


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


def _parse_hidden_sizes(value: str | Sequence[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = value.strip()
        if raw == "" or raw.lower() == "none":
            return ()
        parts = [part.strip() for part in raw.split(",")]
        if any(part == "" for part in parts):
            raise ValueError("hidden_sizes must be a comma-separated list of positive ints")
        try:
            values = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("hidden_sizes must be a comma-separated list of positive ints") from exc
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise ValueError("hidden_sizes must be str or Sequence[int]")

    return tuple(_require_positive_int(size, "hidden_sizes item") for size in values)


@dataclass(frozen=True, slots=True)
class ExecutionPPOTrainCLIConfig:
    tape_root: str
    output_json: str | None = None
    checkpoint_path: str | None = None
    linear_signals_npz: str | None = None
    adverse_signals_npz: str | None = None
    overwrite: bool = False
    save_checkpoint: bool = True
    mmap_mode: str | None = "r"

    decision_interval_us: int = 500_000
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

    maker_fee_bps: float = -0.5
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

    num_updates: int = 10
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    weight_decay: float = 0.0
    start_event_index: int | None = None
    seed: int | None = None

    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    deterministic: bool = False
    reset_on_terminal: bool = True
    device: str | None = "cpu"
    dtype: torch.dtype | str = torch.float32

    hidden_sizes: tuple[int, ...] | str = (128, 128)
    activation: str = "tanh"
    layer_norm: bool = False
    orthogonal_init: bool = True
    enable_threshold: float = 0.5
    enable_logit_bias_init: float = 0.0
    continuous_log_std_init: float = -0.5
    continuous_log_std_min: float = -5.0
    continuous_log_std_max: float = 2.0
    policy_head_gain: float = 0.01
    value_head_gain: float = 1.0

    update_epochs: int = 4
    minibatch_size: int = 256
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    clip_value_loss: bool = True
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float | None = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None

    use_observation_normalizer: bool = True
    observation_normalizer_enabled: bool = True
    observation_normalizer_update: bool = True
    observation_normalizer_epsilon: float = 1e-8
    observation_normalizer_clip: float | None = 10.0
    observation_normalizer_rms_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        _require_nonempty_str(self.tape_root, "tape_root")
        if self.output_json is not None:
            _require_nonempty_str(self.output_json, "output_json")
        if self.checkpoint_path is not None:
            _require_nonempty_str(self.checkpoint_path, "checkpoint_path")
        if self.linear_signals_npz is not None:
            _require_nonempty_str(self.linear_signals_npz, "linear_signals_npz")
        if self.adverse_signals_npz is not None:
            _require_nonempty_str(self.adverse_signals_npz, "adverse_signals_npz")
        _require_bool(self.overwrite, "overwrite")
        _require_bool(self.save_checkpoint, "save_checkpoint")
        if self.mmap_mode not in (None, "r"):
            raise ValueError('mmap_mode must be None or "r"')

        _require_positive_int(self.decision_interval_us, "decision_interval_us")
        _optional_positive_int(self.max_episode_steps, "max_episode_steps")
        _require_positive_int(self.max_distance_ticks, "max_distance_ticks")
        _require_positive_float(self.max_order_qty, "max_order_qty")
        _require_positive_int(self.post_only_gap_ticks, "post_only_gap_ticks")
        _require_positive_float(self.default_order_qty, "default_order_qty")
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        _require_probability(self.l2_decrease_weight, "l2_decrease_weight")
        _require_probability(self.trade_at_level_weight, "trade_at_level_weight")
        _require_nonnegative_float(self.unknown_level_queue_ahead_qty, "unknown_level_queue_ahead_qty")
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
        _require_nonnegative_float(self.terminal_inventory_penalty_bps, "terminal_inventory_penalty_bps")
        _require_positive_float(self.reward_scale, "reward_scale")

        _require_positive_int(self.num_updates, "num_updates")
        _require_positive_float(self.learning_rate, "learning_rate")
        _require_positive_float(self.adam_eps, "adam_eps")
        _require_nonnegative_float(self.weight_decay, "weight_decay")
        _optional_nonnegative_int(self.start_event_index, "start_event_index")
        _optional_nonnegative_int(self.seed, "seed")

        _require_positive_int(self.rollout_steps, "rollout_steps")
        _require_positive_float(self.gamma, "gamma")
        _require_positive_float(self.gae_lambda, "gae_lambda")
        _require_bool(self.deterministic, "deterministic")
        _require_bool(self.reset_on_terminal, "reset_on_terminal")
        if self.device is not None:
            _require_nonempty_str(self.device, "device")
        object.__setattr__(self, "dtype", _coerce_dtype(self.dtype))

        object.__setattr__(self, "hidden_sizes", _parse_hidden_sizes(self.hidden_sizes))
        if self.activation not in ("tanh", "relu", "silu"):
            raise ValueError('activation must be one of "tanh", "relu", or "silu"')
        _require_bool(self.layer_norm, "layer_norm")
        _require_bool(self.orthogonal_init, "orthogonal_init")
        _require_probability(self.enable_threshold, "enable_threshold")
        _require_finite_float(self.enable_logit_bias_init, "enable_logit_bias_init")
        _require_finite_float(self.continuous_log_std_init, "continuous_log_std_init")
        continuous_log_std_min = _require_finite_float(self.continuous_log_std_min, "continuous_log_std_min")
        continuous_log_std_max = _require_finite_float(self.continuous_log_std_max, "continuous_log_std_max")
        if continuous_log_std_min >= continuous_log_std_max:
            raise ValueError("continuous_log_std_min must be less than continuous_log_std_max")
        _require_positive_float(self.policy_head_gain, "policy_head_gain")
        _require_positive_float(self.value_head_gain, "value_head_gain")

        _require_positive_int(self.update_epochs, "update_epochs")
        _require_positive_int(self.minibatch_size, "minibatch_size")
        _require_positive_float(self.clip_range, "clip_range")
        _require_positive_float(self.value_clip_range, "value_clip_range")
        _require_bool(self.clip_value_loss, "clip_value_loss")
        _require_nonnegative_float(self.value_loss_coef, "value_loss_coef")
        _require_nonnegative_float(self.entropy_coef, "entropy_coef")
        if self.max_grad_norm is not None:
            _require_positive_float(self.max_grad_norm, "max_grad_norm")
        _require_bool(self.normalize_advantages, "normalize_advantages")
        if self.target_kl is not None:
            _require_positive_float(self.target_kl, "target_kl")

        _require_bool(self.use_observation_normalizer, "use_observation_normalizer")
        _require_bool(self.observation_normalizer_enabled, "observation_normalizer_enabled")
        _require_bool(self.observation_normalizer_update, "observation_normalizer_update")
        _require_positive_float(self.observation_normalizer_epsilon, "observation_normalizer_epsilon")
        if self.observation_normalizer_clip is not None:
            _require_positive_float(self.observation_normalizer_clip, "observation_normalizer_clip")
        _require_positive_float(self.observation_normalizer_rms_epsilon, "observation_normalizer_rms_epsilon")


def _summary_config(config: ExecutionPPOTrainCLIConfig) -> dict[str, object]:
    return {
        "tape_root": config.tape_root,
        "output_json": config.output_json,
        "checkpoint_path": config.checkpoint_path,
        "linear_signals_npz": config.linear_signals_npz,
        "adverse_signals_npz": config.adverse_signals_npz,
        "overwrite": config.overwrite,
        "save_checkpoint": config.save_checkpoint,
        "mmap_mode": config.mmap_mode,
        "decision_interval_us": config.decision_interval_us,
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
        "num_updates": config.num_updates,
        "learning_rate": config.learning_rate,
        "adam_eps": config.adam_eps,
        "weight_decay": config.weight_decay,
        "start_event_index": config.start_event_index,
        "seed": config.seed,
        "rollout_steps": config.rollout_steps,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "deterministic": config.deterministic,
        "reset_on_terminal": config.reset_on_terminal,
        "device": config.device,
        "dtype": str(config.dtype),
        "hidden_sizes": list(config.hidden_sizes),
        "activation": config.activation,
        "layer_norm": config.layer_norm,
        "orthogonal_init": config.orthogonal_init,
        "enable_threshold": config.enable_threshold,
        "enable_logit_bias_init": config.enable_logit_bias_init,
        "continuous_log_std_init": config.continuous_log_std_init,
        "continuous_log_std_min": config.continuous_log_std_min,
        "continuous_log_std_max": config.continuous_log_std_max,
        "policy_head_gain": config.policy_head_gain,
        "value_head_gain": config.value_head_gain,
        "update_epochs": config.update_epochs,
        "minibatch_size": config.minibatch_size,
        "clip_range": config.clip_range,
        "value_clip_range": config.value_clip_range,
        "clip_value_loss": config.clip_value_loss,
        "value_loss_coef": config.value_loss_coef,
        "entropy_coef": config.entropy_coef,
        "max_grad_norm": config.max_grad_norm,
        "normalize_advantages": config.normalize_advantages,
        "target_kl": config.target_kl,
        "use_observation_normalizer": config.use_observation_normalizer,
        "observation_normalizer_enabled": config.observation_normalizer_enabled,
        "observation_normalizer_update": config.observation_normalizer_update,
        "observation_normalizer_epsilon": config.observation_normalizer_epsilon,
        "observation_normalizer_clip": config.observation_normalizer_clip,
        "observation_normalizer_rms_epsilon": config.observation_normalizer_rms_epsilon,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def _default_output_json(tape_root: str) -> Path:
    return Path(tape_root) / "train_execution_ppo_summary.json"


def _default_checkpoint_path(tape_root: str) -> Path:
    return Path(tape_root) / "execution_ppo_checkpoint.pt"


def _default_linear_signals_npz(tape_root: str) -> Path:
    return Path(tape_root) / LINEAR_SIGNALS_FILENAME



def _build_training_config(config: ExecutionPPOTrainCLIConfig) -> PPOTrainingConfig:
    network_config = ActorCriticConfig(
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        layer_norm=config.layer_norm,
        orthogonal_init=config.orthogonal_init,
        enable_threshold=config.enable_threshold,
        enable_logit_bias_init=config.enable_logit_bias_init,
        continuous_log_std_init=config.continuous_log_std_init,
        continuous_log_std_min=config.continuous_log_std_min,
        continuous_log_std_max=config.continuous_log_std_max,
        policy_head_gain=config.policy_head_gain,
        value_head_gain=config.value_head_gain,
    )
    rollout_config = RolloutConfig(
        rollout_steps=config.rollout_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        deterministic=config.deterministic,
        reset_on_terminal=config.reset_on_terminal,
        device=config.device,
        dtype=config.dtype,
    )
    ppo_config = PPOConfig(
        update_epochs=config.update_epochs,
        minibatch_size=config.minibatch_size,
        clip_range=config.clip_range,
        value_clip_range=config.value_clip_range,
        clip_value_loss=config.clip_value_loss,
        value_loss_coef=config.value_loss_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        normalize_advantages=config.normalize_advantages,
        target_kl=config.target_kl,
    )
    normalizer_config = ObservationNormalizerConfig(
        enabled=config.observation_normalizer_enabled,
        update=config.observation_normalizer_update,
        epsilon=config.observation_normalizer_epsilon,
        clip=config.observation_normalizer_clip,
        rms_epsilon=config.observation_normalizer_rms_epsilon,
    )
    return PPOTrainingConfig(
        num_updates=config.num_updates,
        learning_rate=config.learning_rate,
        adam_eps=config.adam_eps,
        weight_decay=config.weight_decay,
        start_event_index=config.start_event_index,
        seed=config.seed,
        use_observation_normalizer=config.use_observation_normalizer,
        network_config=network_config,
        rollout_config=rollout_config,
        ppo_config=ppo_config,
        observation_normalizer_config=normalizer_config,
    )

def _build_env_config(config: ExecutionPPOTrainCLIConfig) -> ExecutionEnvConfig:
    return build_execution_env_config_from_attrs(
        config,
        adverse_signals_enabled=config.adverse_signals_npz is not None,
    )


def run_execution_ppo_training(config: ExecutionPPOTrainCLIConfig) -> dict[str, object]:
    if not isinstance(config, ExecutionPPOTrainCLIConfig):
        raise ValueError("config must be ExecutionPPOTrainCLIConfig")

    output_json = Path(config.output_json) if config.output_json is not None else _default_output_json(config.tape_root)
    checkpoint_path = (
        Path(config.checkpoint_path)
        if config.checkpoint_path is not None
        else _default_checkpoint_path(config.tape_root)
    )

    if output_json.exists() and not config.overwrite:
        raise FileExistsError(str(output_json))

    if config.save_checkpoint and checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(str(checkpoint_path))

    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode)
    linear_signals_path = (
        Path(config.linear_signals_npz)
        if config.linear_signals_npz is not None
        else _default_linear_signals_npz(config.tape_root)
    )
    linear_signals = load_linear_signal_artifact_npz(linear_signals_path)
    adverse_signals = load_adverse_selection_signals(config.adverse_signals_npz) if config.adverse_signals_npz is not None else None
    env_config = _build_env_config(config)
    linear_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_interval_us=config.decision_interval_us,
        requested_start_event_index=config.start_event_index,
        min_rows=(config.max_episode_steps + 1) if config.max_episode_steps is not None else None,
    )
    env = ExecutionEnv(tape, config=env_config, linear_signals=linear_signals, adverse_signals=adverse_signals)
    training_config = _build_training_config(config)
    result = train_ppo_policy(env, config=training_config)

    summary: dict[str, object] = {
        "status": "ok",
        "run_type": "train_execution_ppo",
        "tape_root": str(Path(config.tape_root)),
        "output_json": str(output_json),
        "checkpoint_path": None if not config.save_checkpoint else str(checkpoint_path),
        "config": _summary_config(config),
        "tape": {
            "schema": tape.manifest.schema,
            "exchange": tape.manifest.exchange,
            "symbol": tape.manifest.symbol,
            "num_events": tape.manifest.num_events,
            "num_l2_batches": tape.manifest.num_l2_batches,
            "num_trades": tape.manifest.num_trades,
            "start_local_ts_us": tape.manifest.start_local_ts_us,
            "end_local_ts_us": tape.manifest.end_local_ts_us,
            "book_depth": tape.manifest.notes.get("book_depth") if tape.manifest.notes is not None else None,
        },
        "training": result.summary_dict(),
        "observation_schema": env.config.observation_schema.as_dict(),
        "linear_signals": linear_signal_artifact_summary(linear_signals, path=str(linear_signals_path)),
        "linear_signal_start": linear_start.as_dict(),
    }

    if config.save_checkpoint:
        checkpoint_payload = make_training_checkpoint_payload(result)
        checkpoint_payload["cli_config"] = _summary_config(config)
        checkpoint_payload["tape"] = summary["tape"]
        checkpoint_payload["observation_schema"] = env.config.observation_schema.as_dict()
        checkpoint_payload["linear_signals"] = linear_signal_artifact_summary(
            linear_signals, path=str(linear_signals_path)
        )
        checkpoint_payload["linear_signal_start"] = linear_start.as_dict()
        _save_checkpoint_atomic(checkpoint_path, checkpoint_payload)
        summary["checkpoint_saved"] = True
    else:
        summary["checkpoint_saved"] = False

    _write_json_atomic(output_json, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an execution PPO policy from an existing execution tape.")
    parser.add_argument("--tape-root", required=True)

    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--linear-signals-npz",
        help="Canonical no-move-gated linear signal NPZ. Defaults to <tape-root>/linear_signals.npz. Required; missing file is an error.",
    )
    parser.add_argument("--adverse-signals-npz")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")

    parser.add_argument("--decision-interval-us", type=int, default=500_000)
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
    parser.add_argument("--maker-fee-bps", type=float, default=-0.5)
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

    parser.add_argument("--num-updates", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-reset-on-terminal", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "fp32", "fp64"), default="float32")

    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--activation", choices=("tanh", "relu", "silu"), default="tanh")
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--no-orthogonal-init", action="store_true")
    parser.add_argument("--enable-threshold", type=float, default=0.5)
    parser.add_argument("--enable-logit-bias-init", type=float, default=0.0)
    parser.add_argument("--continuous-log-std-init", type=float, default=-0.5)
    parser.add_argument("--continuous-log-std-min", type=float, default=-5.0)
    parser.add_argument("--continuous-log-std-max", type=float, default=2.0)
    parser.add_argument("--policy-head-gain", type=float, default=0.01)
    parser.add_argument("--value-head-gain", type=float, default=1.0)

    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-clip-range", type=float, default=0.2)
    parser.add_argument("--no-clip-value-loss", action="store_true")
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--no-grad-clip", action="store_true")
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--target-kl", type=float)

    parser.add_argument("--no-observation-normalizer", action="store_true")
    parser.add_argument("--disable-observation-normalizer", action="store_true")
    parser.add_argument("--freeze-observation-normalizer", action="store_true")
    parser.add_argument("--observation-normalizer-epsilon", type=float, default=1e-8)
    parser.add_argument("--observation-normalizer-clip", type=float, default=10.0)
    parser.add_argument("--no-observation-normalizer-clip", action="store_true")
    parser.add_argument("--observation-normalizer-rms-epsilon", type=float, default=1e-4)

    return parser


def _config_from_args(args: argparse.Namespace) -> ExecutionPPOTrainCLIConfig:
    return ExecutionPPOTrainCLIConfig(
        tape_root=args.tape_root,
        output_json=args.output_json,
        checkpoint_path=args.checkpoint_path,
        linear_signals_npz=args.linear_signals_npz,
        adverse_signals_npz=args.adverse_signals_npz,
        overwrite=args.overwrite,
        save_checkpoint=not args.no_checkpoint,
        mmap_mode=None if args.no_mmap else "r",
        decision_interval_us=args.decision_interval_us,
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
        num_updates=args.num_updates,
        learning_rate=args.learning_rate,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        start_event_index=args.start_event_index,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        deterministic=args.deterministic,
        reset_on_terminal=not args.no_reset_on_terminal,
        device=args.device,
        dtype=args.dtype,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        layer_norm=args.layer_norm,
        orthogonal_init=not args.no_orthogonal_init,
        enable_threshold=args.enable_threshold,
        enable_logit_bias_init=args.enable_logit_bias_init,
        continuous_log_std_init=args.continuous_log_std_init,
        continuous_log_std_min=args.continuous_log_std_min,
        continuous_log_std_max=args.continuous_log_std_max,
        policy_head_gain=args.policy_head_gain,
        value_head_gain=args.value_head_gain,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_range=args.clip_range,
        value_clip_range=args.value_clip_range,
        clip_value_loss=not args.no_clip_value_loss,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=None if args.no_grad_clip else args.max_grad_norm,
        normalize_advantages=not args.no_normalize_advantages,
        target_kl=args.target_kl,
        use_observation_normalizer=not args.no_observation_normalizer,
        observation_normalizer_enabled=not args.disable_observation_normalizer,
        observation_normalizer_update=not args.freeze_observation_normalizer,
        observation_normalizer_epsilon=args.observation_normalizer_epsilon,
        observation_normalizer_clip=(
            None if args.no_observation_normalizer_clip else args.observation_normalizer_clip
        ),
        observation_normalizer_rms_epsilon=args.observation_normalizer_rms_epsilon,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    summary = run_execution_ppo_training(config)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
