"""Runtime adverse-selection observation feature helpers.

This module converts already-aligned adverse-selection signal rows into flat
observation maps. It intentionally does not import RL or environment modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from mmrt.execution.adverse_selection import (
    DEFAULT_QUOTE_CANDIDATE_NAMES,
    QuoteCandidateConfig,
    quote_candidate_configs_from_names,
    candidate_price_tick,
)
from mmrt.execution.adverse_signal import AdverseSelectionSignalArtifact
from mmrt.execution.contracts import LinearSignal, OrderSide
from mmrt.execution.executable_edge import ExecutableEdgeConfig, compute_side_executable_edge


@dataclass(frozen=True, slots=True)
class AdverseRuntimeConfig:
    candidate_names: tuple[str, ...] = DEFAULT_QUOTE_CANDIDATE_NAMES
    post_only_gap_ticks: int = 1
    executable_edge: ExecutableEdgeConfig = ExecutableEdgeConfig()
    candidate_configs: tuple[QuoteCandidateConfig, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        candidates = quote_candidate_configs_from_names(self.candidate_names)
        object.__setattr__(self, "candidate_configs", candidates)
        object.__setattr__(self, "candidate_names", tuple(candidate.name for candidate in candidates))

        if (
            isinstance(self.post_only_gap_ticks, bool)
            or not isinstance(self.post_only_gap_ticks, int)
            or self.post_only_gap_ticks < 0
        ):
            raise ValueError("post_only_gap_ticks must be a nonnegative int")
        if not isinstance(self.executable_edge, ExecutableEdgeConfig):
            raise ValueError("executable_edge must be ExecutableEdgeConfig")


def adverse_predictions_for_row(signals: AdverseSelectionSignalArtifact, row: int) -> dict[str, float]:
    if not isinstance(signals, AdverseSelectionSignalArtifact):
        raise ValueError("signals must be AdverseSelectionSignalArtifact")
    if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row >= signals.decision_local_ts_us.shape[0]:
        raise ValueError("row out of range")
    return {target: float(signals.predictions[target][row]) for target in signals.target_names}


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def build_adverse_observation_features(
    *,
    predictions: Mapping[str, float],
    config: AdverseRuntimeConfig,
) -> dict[str, float]:
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must be a mapping")
    if not isinstance(config, AdverseRuntimeConfig):
        raise ValueError("config must be AdverseRuntimeConfig")
    out: dict[str, float] = {}
    for candidate in config.candidate_configs:
        c = candidate.name
        for side in ("bid", "ask"):
            fill_target = f"{side}_{c}_filled"
            cost_target = f"{side}_{c}_toxic_cost_bps"
            prefix = f"adverse_{side}_{c}"
            if fill_target in predictions and cost_target in predictions:
                out[f"{prefix}_fill_prob"] = min(max(_finite(predictions[fill_target], fill_target), 0.0), 1.0)
                out[f"{prefix}_cost_bps"] = max(_finite(predictions[cost_target], cost_target), 0.0)
                out[f"{prefix}_valid"] = 1.0
            else:
                out[f"{prefix}_fill_prob"] = 0.0
                out[f"{prefix}_cost_bps"] = 0.0
                out[f"{prefix}_valid"] = 0.0
    return out


def build_executable_edge_observation_features(
    *,
    predictions: Mapping[str, float],
    best_bid_tick: int,
    best_ask_tick: int,
    linear_signal: LinearSignal,
    inventory_qty: float,
    config: AdverseRuntimeConfig,
) -> dict[str, float]:
    if not isinstance(linear_signal, LinearSignal):
        raise ValueError("linear_signal must be LinearSignal")
    if not isinstance(config, AdverseRuntimeConfig):
        raise ValueError("config must be AdverseRuntimeConfig")
    mid_tick = (int(best_bid_tick) + int(best_ask_tick)) * 0.5
    out: dict[str, float] = {}
    for candidate in config.candidate_configs:
        c = candidate.name
        for side_name, side in (("bid", OrderSide.BUY), ("ask", OrderSide.SELL)):
            prefix = f"edge_{side_name}_{c}"
            price_tick = candidate_price_tick(
                candidate=candidate,
                side=side,
                best_bid=int(best_bid_tick),
                best_ask=int(best_ask_tick),
                post_only_gap_ticks=config.post_only_gap_ticks,
            )
            if price_tick is None:
                out[f"{prefix}_attempt_bps"] = 0.0
                out[f"{prefix}_cond_fill_bps"] = 0.0
                out[f"{prefix}_allowed"] = 0.0
                out[f"{prefix}_valid"] = 0.0
                continue
            edge = compute_side_executable_edge(
                candidate_name=c,
                side=side,
                mid_tick=mid_tick,
                price_tick=price_tick,
                linear_signal=linear_signal,
                adverse_predictions=predictions,
                inventory_qty=inventory_qty,
                config=config.executable_edge,
            )
            out[f"{prefix}_attempt_bps"] = float(edge.edge_attempt_bps)
            out[f"{prefix}_cond_fill_bps"] = float(edge.edge_cond_fill_bps)
            out[f"{prefix}_allowed"] = 1.0 if edge.quote_allowed else 0.0
            out[f"{prefix}_valid"] = 1.0
    return out


__all__ = [
    "AdverseRuntimeConfig",
    "adverse_predictions_for_row",
    "build_adverse_observation_features",
    "build_executable_edge_observation_features",
]
