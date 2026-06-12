"""Shared CLI-layer construction of :class:`ExecutionEnvConfig`."""

from __future__ import annotations

from dataclasses import dataclass
import math

from mmrt.execution.adverse_runtime import AdverseRuntimeConfig
from mmrt.execution.contracts import ActionSpec, LatencyConfig, PositionState, QueueModelMode
from mmrt.execution.env import ExecutionEnvConfig
from mmrt.execution.executable_edge import ExecutableEdgeConfig
from mmrt.execution.fill_sim import FillSimulatorConfig
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.execution.quote_geometry import QuoteGeometryConfig
from mmrt.execution.reward import RewardConfig


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


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite float")
    return out


def _require_positive_float(value: float, name: str) -> float:
    out = _require_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return out


def _require_nonnegative_float(value: float, name: str) -> float:
    out = _require_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return out


def _require_probability(value: float, name: str) -> float:
    out = _require_finite_float(value, name)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return out


def _coerce_queue_mode(value: QueueModelMode | str) -> QueueModelMode:
    if isinstance(value, QueueModelMode):
        return value
    if isinstance(value, str):
        try:
            return QueueModelMode(value)
        except ValueError as exc:
            raise ValueError(f"queue_mode has invalid value {value!r}") from exc
    raise ValueError("queue_mode must be QueueModelMode or str")


@dataclass(frozen=True, slots=True)
class ExecutionEnvConfigBuildInput:
    cancel_guard_ticks: int = 2

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

    max_episode_steps: int | None = None
    adverse_signals_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cancel_guard_ticks", _require_positive_int(self.cancel_guard_ticks, "cancel_guard_ticks"))
        object.__setattr__(self, "max_distance_ticks", _require_positive_int(self.max_distance_ticks, "max_distance_ticks"))
        object.__setattr__(self, "post_only_gap_ticks", _require_nonnegative_int(self.post_only_gap_ticks, "post_only_gap_ticks"))
        object.__setattr__(self, "max_order_qty", _require_positive_float(self.max_order_qty, "max_order_qty"))
        object.__setattr__(self, "default_order_qty", _require_positive_float(self.default_order_qty, "default_order_qty"))
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        object.__setattr__(self, "l2_decrease_weight", _require_probability(self.l2_decrease_weight, "l2_decrease_weight"))
        object.__setattr__(self, "trade_at_level_weight", _require_probability(self.trade_at_level_weight, "trade_at_level_weight"))
        object.__setattr__(self, "unknown_level_queue_ahead_qty", _require_nonnegative_float(self.unknown_level_queue_ahead_qty, "unknown_level_queue_ahead_qty"))
        object.__setattr__(self, "dedupe_l2_decrease_with_trade_prints", _require_bool(self.dedupe_l2_decrease_with_trade_prints, "dedupe_l2_decrease_with_trade_prints"))
        object.__setattr__(self, "maker_fee_bps", _require_finite_float(self.maker_fee_bps, "maker_fee_bps"))
        object.__setattr__(self, "edge_min_executable_edge_bps", _require_finite_float(self.edge_min_executable_edge_bps, "edge_min_executable_edge_bps"))
        object.__setattr__(self, "edge_latency_buffer_bps", _require_nonnegative_float(self.edge_latency_buffer_bps, "edge_latency_buffer_bps"))
        object.__setattr__(self, "edge_inventory_skew_bps_per_unit", _require_finite_float(self.edge_inventory_skew_bps_per_unit, "edge_inventory_skew_bps_per_unit"))
        object.__setattr__(self, "decision_compute_latency_us", _require_nonnegative_int(self.decision_compute_latency_us, "decision_compute_latency_us"))
        object.__setattr__(self, "order_entry_latency_us", _require_nonnegative_int(self.order_entry_latency_us, "order_entry_latency_us"))
        object.__setattr__(self, "cancel_latency_us", _require_nonnegative_int(self.cancel_latency_us, "cancel_latency_us"))
        object.__setattr__(self, "inventory_penalty_bps", _require_nonnegative_float(self.inventory_penalty_bps, "inventory_penalty_bps"))
        object.__setattr__(self, "turnover_penalty_bps", _require_nonnegative_float(self.turnover_penalty_bps, "turnover_penalty_bps"))
        object.__setattr__(self, "cancel_penalty", _require_nonnegative_float(self.cancel_penalty, "cancel_penalty"))
        object.__setattr__(self, "drawdown_penalty_rate", _require_nonnegative_float(self.drawdown_penalty_rate, "drawdown_penalty_rate"))
        object.__setattr__(self, "terminal_inventory_penalty_bps", _require_nonnegative_float(self.terminal_inventory_penalty_bps, "terminal_inventory_penalty_bps"))
        object.__setattr__(self, "reward_scale", _require_positive_float(self.reward_scale, "reward_scale"))
        object.__setattr__(self, "max_episode_steps", _optional_positive_int(self.max_episode_steps, "max_episode_steps"))
        object.__setattr__(self, "adverse_signals_enabled", _require_bool(self.adverse_signals_enabled, "adverse_signals_enabled"))


def build_execution_env_config_from_input(params: ExecutionEnvConfigBuildInput) -> ExecutionEnvConfig:
    if not isinstance(params, ExecutionEnvConfigBuildInput):
        raise ValueError("params must be ExecutionEnvConfigBuildInput")
    return ExecutionEnvConfig(
        cancel_guard_ticks=params.cancel_guard_ticks,
        action_spec=ActionSpec(max_distance_ticks=params.max_distance_ticks, max_order_qty=params.max_order_qty),
        quote_geometry_config=QuoteGeometryConfig(post_only_gap_ticks=params.post_only_gap_ticks, default_order_qty=params.default_order_qty),
        latency_config=LatencyConfig(
            decision_compute_latency_us=params.decision_compute_latency_us,
            order_entry_latency_us=params.order_entry_latency_us,
            cancel_latency_us=params.cancel_latency_us,
        ),
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=params.queue_mode,
                l2_decrease_weight=params.l2_decrease_weight,
                trade_at_level_weight=params.trade_at_level_weight,
                unknown_level_queue_ahead_qty=params.unknown_level_queue_ahead_qty,
                dedupe_l2_decrease_with_trade_prints=params.dedupe_l2_decrease_with_trade_prints,
            ),
            maker_fee_bps=params.maker_fee_bps,
        ),
        adverse_runtime_config=(
            AdverseRuntimeConfig(
                post_only_gap_ticks=params.post_only_gap_ticks,
                executable_edge=ExecutableEdgeConfig(
                    maker_fee_bps=params.maker_fee_bps,
                    min_executable_edge_bps=params.edge_min_executable_edge_bps,
                    latency_buffer_bps=params.edge_latency_buffer_bps,
                    inventory_skew_bps_per_unit=params.edge_inventory_skew_bps_per_unit,
                ),
            )
            if params.adverse_signals_enabled
            else None
        ),
        reward_config=RewardConfig(
            inventory_penalty_bps=params.inventory_penalty_bps,
            turnover_penalty_bps=params.turnover_penalty_bps,
            cancel_penalty=params.cancel_penalty,
            drawdown_penalty_rate=params.drawdown_penalty_rate,
            terminal_inventory_penalty_bps=params.terminal_inventory_penalty_bps,
            reward_scale=params.reward_scale,
        ),
        initial_position=PositionState(),
        max_episode_steps=params.max_episode_steps,
    )


def build_execution_env_config_from_attrs(obj: object, *, adverse_signals_enabled: bool) -> ExecutionEnvConfig:
    max_episode_steps = getattr(obj, "max_episode_steps", getattr(obj, "max_steps", None))
    params = ExecutionEnvConfigBuildInput(
        cancel_guard_ticks=obj.cancel_guard_ticks,
        max_distance_ticks=obj.max_distance_ticks,
        max_order_qty=obj.max_order_qty,
        post_only_gap_ticks=obj.post_only_gap_ticks,
        default_order_qty=obj.default_order_qty,
        queue_mode=obj.queue_mode,
        l2_decrease_weight=obj.l2_decrease_weight,
        trade_at_level_weight=obj.trade_at_level_weight,
        unknown_level_queue_ahead_qty=obj.unknown_level_queue_ahead_qty,
        dedupe_l2_decrease_with_trade_prints=obj.dedupe_l2_decrease_with_trade_prints,
        maker_fee_bps=obj.maker_fee_bps,
        edge_min_executable_edge_bps=obj.edge_min_executable_edge_bps,
        edge_latency_buffer_bps=obj.edge_latency_buffer_bps,
        edge_inventory_skew_bps_per_unit=obj.edge_inventory_skew_bps_per_unit,
        decision_compute_latency_us=obj.decision_compute_latency_us,
        order_entry_latency_us=obj.order_entry_latency_us,
        cancel_latency_us=obj.cancel_latency_us,
        inventory_penalty_bps=obj.inventory_penalty_bps,
        turnover_penalty_bps=obj.turnover_penalty_bps,
        cancel_penalty=obj.cancel_penalty,
        drawdown_penalty_rate=obj.drawdown_penalty_rate,
        terminal_inventory_penalty_bps=obj.terminal_inventory_penalty_bps,
        reward_scale=obj.reward_scale,
        max_episode_steps=max_episode_steps,
        adverse_signals_enabled=adverse_signals_enabled,
    )
    return build_execution_env_config_from_input(params)


__all__ = [
    "ExecutionEnvConfigBuildInput",
    "build_execution_env_config_from_input",
    "build_execution_env_config_from_attrs",
]
