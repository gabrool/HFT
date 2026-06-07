"""Executable-edge helpers that combine linear alpha and adverse-selection signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mmrt.execution.contracts import LinearSignal, OrderSide


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class ExecutableEdgeConfig:
    maker_fee_bps: float = -0.5
    min_executable_edge_bps: float = 0.0
    latency_buffer_bps: float = 0.0
    inventory_skew_bps_per_unit: float = 0.0
    probability_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "maker_fee_bps", _finite(self.maker_fee_bps, "maker_fee_bps"))
        object.__setattr__(self, "min_executable_edge_bps", _finite(self.min_executable_edge_bps, "min_executable_edge_bps"))
        object.__setattr__(self, "latency_buffer_bps", max(_finite(self.latency_buffer_bps, "latency_buffer_bps"), 0.0))
        object.__setattr__(self, "inventory_skew_bps_per_unit", _finite(self.inventory_skew_bps_per_unit, "inventory_skew_bps_per_unit"))
        eps = _finite(self.probability_epsilon, "probability_epsilon")
        if eps <= 0.0:
            raise ValueError("probability_epsilon must be > 0")
        object.__setattr__(self, "probability_epsilon", eps)


@dataclass(frozen=True, slots=True)
class SideExecutableEdge:
    candidate_name: str
    side: OrderSide
    fill_prob: float
    spread_capture_bps: float
    maker_rebate_bps: float
    alpha_bps: float
    adverse_cost_bps_uncond: float
    adverse_cost_bps_cond: float
    edge_attempt_bps: float
    edge_cond_fill_bps: float
    quote_allowed: bool


def compute_side_executable_edge(
    *,
    candidate_name: str,
    side: OrderSide,
    mid_tick: float,
    price_tick: int,
    linear_signal: LinearSignal,
    adverse_predictions: Mapping[str, float],
    inventory_qty: float = 0.0,
    config: ExecutableEdgeConfig = ExecutableEdgeConfig(),
) -> SideExecutableEdge:
    if not isinstance(side, OrderSide):
        raise ValueError("side must be OrderSide")
    if not isinstance(linear_signal, LinearSignal):
        raise ValueError("linear_signal must be LinearSignal")
    mid_tick = _finite(mid_tick, "mid_tick")
    if mid_tick <= 0.0 or int(price_tick) <= 0:
        raise ValueError("mid_tick and price_tick must be positive")
    if not isinstance(config, ExecutableEdgeConfig):
        raise ValueError("config must be ExecutableEdgeConfig")
    prefix = "bid" if side == OrderSide.BUY else "ask"
    fill_target = f"{prefix}_{candidate_name}_filled"
    cost_target = f"{prefix}_{candidate_name}_toxic_cost_bps"
    if fill_target not in adverse_predictions or cost_target not in adverse_predictions:
        raise ValueError(f"missing adverse-selection predictions required for executable edge: {[n for n in (fill_target, cost_target) if n not in adverse_predictions]}")
    fill_prob = min(max(_finite(adverse_predictions[fill_target], fill_target), 0.0), 1.0)
    adverse_uncond = max(_finite(adverse_predictions[cost_target], cost_target), 0.0)
    if side == OrderSide.BUY:
        alpha_bps = linear_signal.expected_return_bps
        spread_capture_bps = max(mid_tick - price_tick, 0.0) / mid_tick * 10_000.0
        inventory_penalty = inventory_qty * config.inventory_skew_bps_per_unit
    else:
        alpha_bps = -linear_signal.expected_return_bps
        spread_capture_bps = max(price_tick - mid_tick, 0.0) / mid_tick * 10_000.0
        inventory_penalty = -inventory_qty * config.inventory_skew_bps_per_unit
    maker_rebate_bps = -config.maker_fee_bps
    adverse_cond = adverse_uncond / max(fill_prob, config.probability_epsilon)
    edge_attempt = fill_prob * (spread_capture_bps + maker_rebate_bps + alpha_bps) - adverse_uncond - config.latency_buffer_bps - inventory_penalty
    edge_cond = spread_capture_bps + maker_rebate_bps + alpha_bps - adverse_cond
    return SideExecutableEdge(
        candidate_name=candidate_name,
        side=side,
        fill_prob=fill_prob,
        spread_capture_bps=spread_capture_bps,
        maker_rebate_bps=maker_rebate_bps,
        alpha_bps=alpha_bps,
        adverse_cost_bps_uncond=adverse_uncond,
        adverse_cost_bps_cond=adverse_cond,
        edge_attempt_bps=edge_attempt,
        edge_cond_fill_bps=edge_cond,
        quote_allowed=edge_attempt > config.min_executable_edge_bps,
    )
