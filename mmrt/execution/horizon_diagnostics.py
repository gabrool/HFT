"""Compact future-horizon diagnostics for execution policy audits/evals.

This module is read-only with respect to the execution environment.  It records
the minimum step/fill facts needed to compare immediate rewards with later
book/equity outcomes, then emits aggregate diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.contracts import ExecutionStepResult, Fill, OrderSide

DEFAULT_HORIZONS_US = (250_000, 500_000, 1_000_000)
QUOTE_MODE_NAMES = ("no_quote", "bid_only", "ask_only", "two_sided")
SIGNAL_BUCKET_NAMES = ("negative", "neutral", "positive")

__all__ = [
    "DEFAULT_HORIZONS_US",
    "HorizonDiagnosticsConfig",
    "HorizonDiagnosticsAccumulator",
    "parse_horizon_diagnostics_us",
    "quote_mode_from_bools",
]


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _optional_finite_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, name)


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return int(value)


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int or None")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return int(value)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sum(values: Sequence[float]) -> float:
    return float(sum(values))


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def parse_horizon_diagnostics_us(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse and validate comma-separated horizon durations in microseconds."""

    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            raise ValueError("horizon diagnostics list must not be empty")
        parts = [part.strip() for part in raw.split(",")]
        if any(part == "" for part in parts):
            raise ValueError("horizon diagnostics must be comma-separated positive ints")
        try:
            horizons = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("horizon diagnostics must be comma-separated positive ints") from exc
    else:
        horizons = tuple(value)
    return _validate_horizons(horizons, enabled=True)


def _validate_horizons(values: Sequence[int], *, enabled: bool) -> tuple[int, ...]:
    horizons = tuple(_require_positive_int(int(item), "horizons_us item") for item in values)
    if enabled and not horizons:
        raise ValueError("horizons_us must not be empty when horizon diagnostics are enabled")
    if tuple(sorted(horizons)) != horizons:
        raise ValueError("horizons_us must be sorted ascending")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons_us must be unique")
    return horizons


def quote_mode_from_bools(bid_enabled: bool, ask_enabled: bool) -> str:
    bid_enabled = bool(bid_enabled)
    ask_enabled = bool(ask_enabled)
    if bid_enabled and ask_enabled:
        return "two_sided"
    if bid_enabled:
        return "bid_only"
    if ask_enabled:
        return "ask_only"
    return "no_quote"


@dataclass(frozen=True, slots=True)
class HorizonDiagnosticsConfig:
    enabled: bool = True
    horizons_us: tuple[int, ...] = DEFAULT_HORIZONS_US
    max_records: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _require_bool(self.enabled, "enabled"))
        object.__setattr__(
            self,
            "horizons_us",
            _validate_horizons(tuple(self.horizons_us), enabled=self.enabled),
        )
        object.__setattr__(self, "max_records", _optional_nonnegative_int(self.max_records, "max_records"))

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "horizons_us": [int(item) for item in self.horizons_us],
            "max_records": self.max_records,
        }


@dataclass(frozen=True, slots=True)
class _DecisionRecord:
    episode_id: int
    step_index: int
    decision_row: int
    decision_local_ts_us: int
    next_decision_row: int
    next_decision_local_ts_us: int | None
    previous_equity: float
    current_equity: float
    immediate_reward: float
    previous_mid_price: float
    current_mid_price: float | None
    cash_after_step: float
    inventory_after_step: float
    effective_quote_mode: str
    requested_quote_mode: str
    quote_bid_enabled: bool
    quote_ask_enabled: bool
    fill_count: int
    buy_fill_qty: float
    sell_fill_qty: float
    net_fill_qty: float
    linear_expected_return_bps: float | None
    linear_confidence: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": int(self.episode_id),
            "step_index": int(self.step_index),
            "decision_row": int(self.decision_row),
            "decision_local_ts_us": int(self.decision_local_ts_us),
            "next_decision_row": int(self.next_decision_row),
            "next_decision_local_ts_us": self.next_decision_local_ts_us,
            "previous_equity": self.previous_equity,
            "current_equity": self.current_equity,
            "immediate_reward": self.immediate_reward,
            "previous_mid_price": self.previous_mid_price,
            "current_mid_price": self.current_mid_price,
            "cash_after_step": self.cash_after_step,
            "inventory_after_step": self.inventory_after_step,
            "effective_quote_mode": self.effective_quote_mode,
            "requested_quote_mode": self.requested_quote_mode,
            "quote_bid_enabled": self.quote_bid_enabled,
            "quote_ask_enabled": self.quote_ask_enabled,
            "fill_count": int(self.fill_count),
            "buy_fill_qty": self.buy_fill_qty,
            "sell_fill_qty": self.sell_fill_qty,
            "net_fill_qty": self.net_fill_qty,
            "linear_expected_return_bps": self.linear_expected_return_bps,
            "linear_confidence": self.linear_confidence,
        }


@dataclass(frozen=True, slots=True)
class _FillRecord:
    episode_id: int
    decision_row: int
    fill_side: str
    fill_reason: str
    fill_local_ts_us: int
    fill_price: float
    fill_qty: float
    fill_fee: float
    effective_quote_mode: str

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": int(self.episode_id),
            "decision_row": int(self.decision_row),
            "fill_side": self.fill_side,
            "fill_reason": self.fill_reason,
            "fill_local_ts_us": int(self.fill_local_ts_us),
            "fill_price": self.fill_price,
            "fill_qty": self.fill_qty,
            "fill_fee": self.fill_fee,
            "effective_quote_mode": self.effective_quote_mode,
        }


@dataclass(slots=True)
class _ValueSeries:
    values: list[float]

    def add(self, value: float | None) -> None:
        if value is None:
            return
        out = _safe_float(value)
        if out is not None:
            self.values.append(out)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float | None:
        return _mean(self.values)

    @property
    def total(self) -> float:
        return _sum(self.values)

    def percentiles(self) -> dict[str, float | None]:
        return {
            "p10": _percentile(self.values, 10),
            "p50": _percentile(self.values, 50),
            "p90": _percentile(self.values, 90),
        }


@dataclass(slots=True)
class _DecisionHorizonStats:
    count: int = 0
    unavailable_count: int = 0
    immediate_reward: _ValueSeries = None  # type: ignore[assignment]
    actual_path_equity_delta: _ValueSeries = None  # type: ignore[assignment]
    carry_mark_equity_delta: _ValueSeries = None  # type: ignore[assignment]
    carry_mark_increment_after_step: _ValueSeries = None  # type: ignore[assignment]
    future_mid_return_bps: _ValueSeries = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.immediate_reward is None:
            self.immediate_reward = _ValueSeries([])
        if self.actual_path_equity_delta is None:
            self.actual_path_equity_delta = _ValueSeries([])
        if self.carry_mark_equity_delta is None:
            self.carry_mark_equity_delta = _ValueSeries([])
        if self.carry_mark_increment_after_step is None:
            self.carry_mark_increment_after_step = _ValueSeries([])
        if self.future_mid_return_bps is None:
            self.future_mid_return_bps = _ValueSeries([])

    def add_anchor(self, immediate_reward: float) -> None:
        self.count += 1
        self.immediate_reward.add(immediate_reward)

    def add_unavailable(self) -> None:
        self.unavailable_count += 1

    def add_available(
        self,
        *,
        actual_path_equity_delta: float,
        carry_mark_equity_delta: float,
        carry_mark_increment_after_step: float,
        future_mid_return_bps: float,
    ) -> None:
        self.actual_path_equity_delta.add(actual_path_equity_delta)
        self.carry_mark_equity_delta.add(carry_mark_equity_delta)
        self.carry_mark_increment_after_step.add(carry_mark_increment_after_step)
        self.future_mid_return_bps.add(future_mid_return_bps)

    def as_dict(self) -> dict[str, object]:
        mid = self.future_mid_return_bps.percentiles()
        return {
            "count": int(self.count),
            "available_count": int(self.actual_path_equity_delta.count),
            "unavailable_count": int(self.unavailable_count),
            "immediate_reward_mean": self.immediate_reward.mean,
            "immediate_reward_sum": self.immediate_reward.total,
            "actual_path_equity_delta_mean": self.actual_path_equity_delta.mean,
            "actual_path_equity_delta_sum": self.actual_path_equity_delta.total,
            "carry_mark_equity_delta_mean": self.carry_mark_equity_delta.mean,
            "carry_mark_equity_delta_sum": self.carry_mark_equity_delta.total,
            "carry_mark_increment_after_step_mean": self.carry_mark_increment_after_step.mean,
            "future_mid_return_bps_mean": self.future_mid_return_bps.mean,
            "future_mid_return_bps_p10": mid["p10"],
            "future_mid_return_bps_p50": mid["p50"],
            "future_mid_return_bps_p90": mid["p90"],
        }


@dataclass(slots=True)
class _FillHorizonStats:
    fill_count: int = 0
    unavailable_count: int = 0
    gross_markout_bps: _ValueSeries = None  # type: ignore[assignment]
    net_markout_bps: _ValueSeries = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gross_markout_bps is None:
            self.gross_markout_bps = _ValueSeries([])
        if self.net_markout_bps is None:
            self.net_markout_bps = _ValueSeries([])

    def add_fill(self) -> None:
        self.fill_count += 1

    def add_unavailable(self) -> None:
        self.unavailable_count += 1

    def add_available(self, *, gross_markout_bps: float, net_markout_bps: float) -> None:
        self.gross_markout_bps.add(gross_markout_bps)
        self.net_markout_bps.add(net_markout_bps)

    def as_dict(self) -> dict[str, object]:
        gross = self.gross_markout_bps.percentiles()
        net = self.net_markout_bps.percentiles()
        return {
            "fill_count": int(self.fill_count),
            "available_count": int(self.gross_markout_bps.count),
            "unavailable_count": int(self.unavailable_count),
            "gross_markout_bps_mean": self.gross_markout_bps.mean,
            "gross_markout_bps_p10": gross["p10"],
            "gross_markout_bps_p50": gross["p50"],
            "gross_markout_bps_p90": gross["p90"],
            "net_markout_bps_mean": self.net_markout_bps.mean,
            "net_markout_bps_p10": net["p10"],
            "net_markout_bps_p50": net["p50"],
            "net_markout_bps_p90": net["p90"],
        }


@dataclass(slots=True)
class _BucketStats:
    count: int = 0
    bid_enabled_count: int = 0
    ask_enabled_count: int = 0
    mode_counts: dict[str, int] = None  # type: ignore[assignment]
    reward: _ValueSeries = None  # type: ignore[assignment]
    actual_path_equity_delta: _ValueSeries = None  # type: ignore[assignment]
    carry_mark_equity_delta: _ValueSeries = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mode_counts is None:
            self.mode_counts = {mode: 0 for mode in QUOTE_MODE_NAMES}
        if self.reward is None:
            self.reward = _ValueSeries([])
        if self.actual_path_equity_delta is None:
            self.actual_path_equity_delta = _ValueSeries([])
        if self.carry_mark_equity_delta is None:
            self.carry_mark_equity_delta = _ValueSeries([])

    def add_record(self, record: _DecisionRecord) -> None:
        self.count += 1
        self.bid_enabled_count += int(record.quote_bid_enabled)
        self.ask_enabled_count += int(record.quote_ask_enabled)
        self.mode_counts[record.effective_quote_mode] = self.mode_counts.get(record.effective_quote_mode, 0) + 1
        self.reward.add(record.immediate_reward)

    def add_horizon(self, *, actual_path_equity_delta: float, carry_mark_equity_delta: float) -> None:
        self.actual_path_equity_delta.add(actual_path_equity_delta)
        self.carry_mark_equity_delta.add(carry_mark_equity_delta)

    def as_dict(self) -> dict[str, object]:
        denom = max(self.count, 1)
        return {
            "count": int(self.count),
            "bid_enabled_rate": float(self.bid_enabled_count / denom),
            "ask_enabled_rate": float(self.ask_enabled_count / denom),
            "no_quote_rate": float(self.mode_counts.get("no_quote", 0) / denom),
            "bid_only_rate": float(self.mode_counts.get("bid_only", 0) / denom),
            "ask_only_rate": float(self.mode_counts.get("ask_only", 0) / denom),
            "two_sided_rate": float(self.mode_counts.get("two_sided", 0) / denom),
            "reward_mean": self.reward.mean,
            "actual_path_equity_delta_mean": self.actual_path_equity_delta.mean,
            "carry_mark_equity_delta_mean": self.carry_mark_equity_delta.mean,
        }


class HorizonDiagnosticsAccumulator:
    """Accumulate aggregate horizon diagnostics from execution steps."""

    def __init__(
        self,
        *,
        decision_local_ts_us: Sequence[int] | np.ndarray,
        mid_prices: Sequence[float] | np.ndarray,
        tick_size: float = 1.0,
        contract_size: float = 1.0,
        linear_expected_return_bps: Sequence[float] | np.ndarray | None = None,
        linear_confidence: Sequence[float] | np.ndarray | None = None,
        config: HorizonDiagnosticsConfig = HorizonDiagnosticsConfig(),
    ) -> None:
        if not isinstance(config, HorizonDiagnosticsConfig):
            raise ValueError("config must be HorizonDiagnosticsConfig")
        self.config = config
        self.decision_local_ts_us = np.asarray(decision_local_ts_us, dtype=np.int64)
        self.mid_prices = np.asarray(mid_prices, dtype=np.float64)
        if self.decision_local_ts_us.ndim != 1 or self.mid_prices.ndim != 1:
            raise ValueError("decision_local_ts_us and mid_prices must be 1D")
        if self.decision_local_ts_us.shape[0] != self.mid_prices.shape[0]:
            raise ValueError("decision_local_ts_us and mid_prices must have matching lengths")
        if self.decision_local_ts_us.shape[0] == 0:
            raise ValueError("decision arrays must not be empty")
        if (self.decision_local_ts_us <= 0).any():
            raise ValueError("decision_local_ts_us must be positive")
        if (np.diff(self.decision_local_ts_us) < 0).any():
            raise ValueError("decision_local_ts_us must be sorted ascending")
        if not np.isfinite(self.mid_prices).all() or (self.mid_prices <= 0.0).any():
            raise ValueError("mid_prices must be finite and positive")
        self.tick_size = _require_finite_float(tick_size, "tick_size")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be > 0")
        self.contract_size = _require_finite_float(contract_size, "contract_size")
        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")
        self.linear_expected_return_bps = self._optional_signal_array(
            linear_expected_return_bps,
            "linear_expected_return_bps",
        )
        self.linear_confidence = self._optional_signal_array(linear_confidence, "linear_confidence")
        self._records: list[_DecisionRecord] = []
        self._fill_records: list[_FillRecord] = []
        self._equity_by_episode_row: dict[int, dict[int, float]] = {}
        self._current_episode_id = -1
        self._warnings: list[str] = []

    @classmethod
    def from_execution(
        cls,
        *,
        decision_grid: object,
        tape: object,
        linear_signals: object | None = None,
        config: HorizonDiagnosticsConfig = HorizonDiagnosticsConfig(),
    ) -> "HorizonDiagnosticsAccumulator":
        book_ptr = np.asarray(getattr(decision_grid, "book_ptr"), dtype=np.int64)
        l2_events = getattr(getattr(tape, "arrays"), "l2_events")
        bid_ticks = np.asarray(l2_events["best_bid_tick"][book_ptr], dtype=np.float64)
        ask_ticks = np.asarray(l2_events["best_ask_tick"][book_ptr], dtype=np.float64)
        symbol_spec = getattr(getattr(tape, "manifest"), "symbol_spec")
        tick_size = float(symbol_spec.tick_size)
        mid_prices = (bid_ticks + ask_ticks) * tick_size * 0.5
        linear_expected_return_bps = None
        linear_confidence = None
        if linear_signals is not None:
            arrays = getattr(linear_signals, "arrays")
            linear_expected_return_bps = getattr(arrays, "expected_return_bps", None)
            linear_confidence = getattr(arrays, "confidence", None)
        return cls(
            decision_local_ts_us=getattr(decision_grid, "decision_local_ts_us"),
            mid_prices=mid_prices,
            tick_size=tick_size,
            contract_size=float(symbol_spec.contract_size),
            linear_expected_return_bps=linear_expected_return_bps,
            linear_confidence=linear_confidence,
            config=config,
        )

    def _optional_signal_array(
        self,
        values: Sequence[float] | np.ndarray | None,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D")
        if arr.shape[0] != self.decision_local_ts_us.shape[0]:
            raise ValueError(f"{name} length must match decision rows")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        return arr

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start_episode(self) -> int:
        self._current_episode_id += 1
        self._equity_by_episode_row.setdefault(self._current_episode_id, {})
        return self._current_episode_id

    def _episode_id(self, episode_id: int | None) -> int:
        if episode_id is not None:
            self._equity_by_episode_row.setdefault(int(episode_id), {})
            return int(episode_id)
        if self._current_episode_id < 0:
            return self.start_episode()
        return self._current_episode_id

    def record_step(
        self,
        step: object,
        *,
        requested_bid_enabled: bool | None = None,
        requested_ask_enabled: bool | None = None,
        episode_id: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        execution = getattr(step, "execution", step)
        if not isinstance(execution, ExecutionStepResult):
            raise ValueError("step must be ExecutionEnvStep or ExecutionStepResult")
        info = execution.info or {}
        decision_row = int(info["decision_grid_row_index"])
        next_decision_row = int(info["next_decision_grid_row_index"])
        bid_enabled = bool(info.get("quote_bid_enabled", False))
        ask_enabled = bool(info.get("quote_ask_enabled", False))
        requested_bid = bid_enabled if requested_bid_enabled is None else bool(requested_bid_enabled)
        requested_ask = ask_enabled if requested_ask_enabled is None else bool(requested_ask_enabled)
        reward_value = getattr(step, "reward", None)
        immediate_reward = (
            _require_finite_float(reward_value, "step.reward")
            if reward_value is not None
            else float(execution.reward.total_reward)
        )
        buy_fill_qty = 0.0
        sell_fill_qty = 0.0
        for fill in execution.fills:
            if fill.side == OrderSide.BUY:
                buy_fill_qty += fill.qty
            elif fill.side == OrderSide.SELL:
                sell_fill_qty += fill.qty
        record = self.add_decision_record(
            step_index=int(info.get("step_index", len(self._records))),
            decision_row=decision_row,
            next_decision_row=next_decision_row,
            previous_equity=_require_finite_float(info["previous_equity"], "previous_equity"),
            current_equity=_require_finite_float(info["current_equity"], "current_equity"),
            immediate_reward=immediate_reward,
            cash_after_step=execution.position.cash,
            inventory_after_step=execution.position.inventory_qty,
            effective_quote_mode=quote_mode_from_bools(bid_enabled, ask_enabled),
            requested_quote_mode=quote_mode_from_bools(requested_bid, requested_ask),
            quote_bid_enabled=bid_enabled,
            quote_ask_enabled=ask_enabled,
            fill_count=len(execution.fills),
            buy_fill_qty=buy_fill_qty,
            sell_fill_qty=sell_fill_qty,
            episode_id=episode_id,
        )
        for fill in execution.fills:
            self.add_fill_record(fill, decision_record=record)

    def add_decision_record(
        self,
        *,
        step_index: int,
        decision_row: int,
        next_decision_row: int,
        previous_equity: float,
        current_equity: float,
        immediate_reward: float,
        cash_after_step: float,
        inventory_after_step: float,
        effective_quote_mode: str,
        requested_quote_mode: str,
        quote_bid_enabled: bool,
        quote_ask_enabled: bool,
        fill_count: int,
        buy_fill_qty: float,
        sell_fill_qty: float,
        linear_expected_return_bps: float | None = None,
        linear_confidence: float | None = None,
        episode_id: int | None = None,
    ) -> _DecisionRecord:
        if not self.config.enabled:
            raise ValueError("cannot add records when horizon diagnostics are disabled")
        episode = self._episode_id(episode_id)
        decision_row = self._validate_row(decision_row, "decision_row")
        if next_decision_row < 0 or next_decision_row >= self.decision_local_ts_us.shape[0]:
            next_ts = None
            current_mid = None
        else:
            next_ts = int(self.decision_local_ts_us[next_decision_row])
            current_mid = float(self.mid_prices[next_decision_row])
        if effective_quote_mode not in QUOTE_MODE_NAMES:
            raise ValueError("effective_quote_mode is invalid")
        if requested_quote_mode not in QUOTE_MODE_NAMES:
            raise ValueError("requested_quote_mode is invalid")
        if linear_expected_return_bps is None and self.linear_expected_return_bps is not None:
            linear_expected_return_bps = float(self.linear_expected_return_bps[decision_row])
        if linear_confidence is None and self.linear_confidence is not None:
            linear_confidence = float(self.linear_confidence[decision_row])
        buy_fill_qty = _require_finite_float(buy_fill_qty, "buy_fill_qty")
        sell_fill_qty = _require_finite_float(sell_fill_qty, "sell_fill_qty")
        record = _DecisionRecord(
            episode_id=episode,
            step_index=int(step_index),
            decision_row=decision_row,
            decision_local_ts_us=int(self.decision_local_ts_us[decision_row]),
            next_decision_row=int(next_decision_row),
            next_decision_local_ts_us=next_ts,
            previous_equity=_require_finite_float(previous_equity, "previous_equity"),
            current_equity=_require_finite_float(current_equity, "current_equity"),
            immediate_reward=_require_finite_float(immediate_reward, "immediate_reward"),
            previous_mid_price=float(self.mid_prices[decision_row]),
            current_mid_price=current_mid,
            cash_after_step=_require_finite_float(cash_after_step, "cash_after_step"),
            inventory_after_step=_require_finite_float(inventory_after_step, "inventory_after_step"),
            effective_quote_mode=effective_quote_mode,
            requested_quote_mode=requested_quote_mode,
            quote_bid_enabled=bool(quote_bid_enabled),
            quote_ask_enabled=bool(quote_ask_enabled),
            fill_count=int(fill_count),
            buy_fill_qty=buy_fill_qty,
            sell_fill_qty=sell_fill_qty,
            net_fill_qty=buy_fill_qty - sell_fill_qty,
            linear_expected_return_bps=_optional_finite_float(linear_expected_return_bps, "linear_expected_return_bps"),
            linear_confidence=_optional_finite_float(linear_confidence, "linear_confidence"),
        )
        self._records.append(record)
        equity_by_row = self._equity_by_episode_row.setdefault(episode, {})
        equity_by_row.setdefault(decision_row, record.previous_equity)
        if 0 <= next_decision_row < self.decision_local_ts_us.shape[0]:
            equity_by_row[int(next_decision_row)] = record.current_equity
        return record

    def add_fill_record(
        self,
        fill: Fill,
        *,
        decision_record: _DecisionRecord,
    ) -> None:
        if not isinstance(fill, Fill):
            raise ValueError("fill must be Fill")
        price = float(fill.price_tick * self.tick_size)
        self._fill_records.append(
            _FillRecord(
                episode_id=decision_record.episode_id,
                decision_row=decision_record.decision_row,
                fill_side=fill.side.value,
                fill_reason=fill.reason.value,
                fill_local_ts_us=int(fill.local_ts_us),
                fill_price=price,
                fill_qty=float(fill.qty),
                fill_fee=float(fill.fee),
                effective_quote_mode=decision_record.effective_quote_mode,
            )
        )

    def _validate_row(self, row: int, name: str) -> int:
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError(f"{name} must be int")
        if row < 0 or row >= self.decision_local_ts_us.shape[0]:
            raise ValueError(f"{name} out of range")
        return int(row)

    def _future_row(self, *, anchor_local_ts_us: int, horizon_us: int, episode_id: int) -> int | None:
        target = int(anchor_local_ts_us) + int(horizon_us)
        row = int(np.searchsorted(self.decision_local_ts_us, target, side="left"))
        if row >= self.decision_local_ts_us.shape[0]:
            return None
        if row not in self._equity_by_episode_row.get(episode_id, {}):
            return None
        return row

    def _future_decision_values(
        self,
        record: _DecisionRecord,
        horizon_us: int,
    ) -> tuple[float, float, float, float] | None:
        future_row = self._future_row(
            anchor_local_ts_us=record.decision_local_ts_us,
            horizon_us=horizon_us,
            episode_id=record.episode_id,
        )
        if future_row is None:
            return None
        future_mid = float(self.mid_prices[future_row])
        future_equity = self._equity_by_episode_row[record.episode_id][future_row]
        actual_path_equity_delta = future_equity - record.previous_equity
        carry_mark_equity = (
            record.cash_after_step
            + record.inventory_after_step * future_mid * self.contract_size
        )
        carry_mark_equity_delta = carry_mark_equity - record.previous_equity
        carry_mark_increment_after_step = carry_mark_equity - record.current_equity
        future_mid_return_bps = (future_mid - record.previous_mid_price) / record.previous_mid_price * 10_000.0
        return (
            actual_path_equity_delta,
            carry_mark_equity_delta,
            carry_mark_increment_after_step,
            future_mid_return_bps,
        )

    def _fill_markout_values(
        self,
        record: _FillRecord,
        horizon_us: int,
    ) -> tuple[float, float] | None:
        future_row = self._future_row(
            anchor_local_ts_us=record.fill_local_ts_us,
            horizon_us=horizon_us,
            episode_id=record.episode_id,
        )
        if future_row is None:
            return None
        future_mid = float(self.mid_prices[future_row])
        if record.fill_side == OrderSide.BUY.value:
            gross = (future_mid - record.fill_price) / record.fill_price * 10_000.0
        elif record.fill_side == OrderSide.SELL.value:
            gross = (record.fill_price - future_mid) / record.fill_price * 10_000.0
        else:
            return None
        notional = record.fill_price * record.fill_qty * self.contract_size
        fee_bps = record.fill_fee / notional * 10_000.0 if notional > 0.0 else 0.0
        return gross, gross - fee_bps

    def _decision_groups(self) -> dict[str, object]:
        by_horizon: dict[str, object] = {}
        for horizon_us in self.config.horizons_us:
            groups: dict[str, _DecisionHorizonStats] = {}
            for record in self._records:
                values = self._future_decision_values(record, horizon_us)
                for group_name in ("all", record.effective_quote_mode):
                    stats = groups.setdefault(group_name, _DecisionHorizonStats())
                    stats.add_anchor(record.immediate_reward)
                    if values is None:
                        stats.add_unavailable()
                    else:
                        stats.add_available(
                            actual_path_equity_delta=values[0],
                            carry_mark_equity_delta=values[1],
                            carry_mark_increment_after_step=values[2],
                            future_mid_return_bps=values[3],
                        )
            by_horizon[str(horizon_us)] = {
                name: stats.as_dict() for name, stats in sorted(groups.items())
            }
        return {"by_horizon": by_horizon}

    def _fill_groups(self) -> dict[str, object]:
        by_horizon: dict[str, object] = {}
        for horizon_us in self.config.horizons_us:
            groups: dict[str, _FillHorizonStats] = {}
            for record in self._fill_records:
                values = self._fill_markout_values(record, horizon_us)
                group_names = (
                    "all",
                    f"side:{record.fill_side}",
                    f"reason:{record.fill_reason}",
                    f"mode:{record.effective_quote_mode}",
                    f"side_reason:{record.fill_side}:{record.fill_reason}",
                )
                for group_name in group_names:
                    stats = groups.setdefault(group_name, _FillHorizonStats())
                    stats.add_fill()
                    if values is None:
                        stats.add_unavailable()
                    else:
                        stats.add_available(
                            gross_markout_bps=values[0],
                            net_markout_bps=values[1],
                        )
            by_horizon[str(horizon_us)] = {
                name: stats.as_dict() for name, stats in sorted(groups.items())
            }
        return {"by_horizon": by_horizon}

    def _signal_bucket(self, expected_return_bps: float | None) -> str | None:
        if expected_return_bps is None:
            return None
        if expected_return_bps < 0.0:
            return "negative"
        if expected_return_bps > 0.0:
            return "positive"
        return "neutral"

    def _signal_alignment(self) -> dict[str, object]:
        by_horizon: dict[str, object] = {}
        for horizon_us in self.config.horizons_us:
            predicted: list[float] = []
            realized: list[float] = []
            for record in self._records:
                if record.linear_expected_return_bps is None:
                    continue
                values = self._future_decision_values(record, horizon_us)
                if values is None:
                    continue
                predicted.append(record.linear_expected_return_bps)
                realized.append(values[3])
            by_horizon[str(horizon_us)] = self._alignment_stats(predicted, realized)

        action_horizon = 1_000_000 if 1_000_000 in self.config.horizons_us else (
            self.config.horizons_us[-1] if self.config.horizons_us else None
        )
        buckets = {name: _BucketStats() for name in SIGNAL_BUCKET_NAMES}
        if action_horizon is not None:
            for record in self._records:
                bucket_name = self._signal_bucket(record.linear_expected_return_bps)
                if bucket_name is None:
                    continue
                bucket = buckets[bucket_name]
                bucket.add_record(record)
                values = self._future_decision_values(record, action_horizon)
                if values is not None:
                    bucket.add_horizon(
                        actual_path_equity_delta=values[0],
                        carry_mark_equity_delta=values[1],
                    )
        return {
            "signal_alignment_by_horizon": by_horizon,
            "action_by_signal_bucket": {
                "horizon_us": action_horizon,
                "buckets": {name: buckets[name].as_dict() for name in SIGNAL_BUCKET_NAMES},
            },
        }

    def _alignment_stats(self, predicted: Sequence[float], realized: Sequence[float]) -> dict[str, object]:
        count = len(predicted)
        pearson: float | None = None
        if count >= 2:
            pred_arr = np.asarray(predicted, dtype=np.float64)
            real_arr = np.asarray(realized, dtype=np.float64)
            pred_std = float(pred_arr.std())
            real_std = float(real_arr.std())
            if pred_std > 0.0 and real_std > 0.0:
                pearson = float(np.corrcoef(pred_arr, real_arr)[0, 1])
        sign_total = 0
        sign_correct = 0
        positive_realized: list[float] = []
        negative_realized: list[float] = []
        for pred, real in zip(predicted, realized):
            if pred > 0.0:
                positive_realized.append(real)
            elif pred < 0.0:
                negative_realized.append(real)
            if pred == 0.0 or real == 0.0:
                continue
            sign_total += 1
            sign_correct += int((pred > 0.0) == (real > 0.0))
        return {
            "count": int(count),
            "pearson_correlation": pearson,
            "sign_accuracy_excluding_zero": (
                None if sign_total == 0 else float(sign_correct / sign_total)
            ),
            "sign_accuracy_count": int(sign_total),
            "mean_predicted_return_bps": _mean(predicted),
            "mean_realized_return_bps": _mean(realized),
            "mean_realized_when_pred_positive_bps": _mean(positive_realized),
            "mean_realized_when_pred_negative_bps": _mean(negative_realized),
        }

    def as_dict(self, *, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "enabled": bool(self.config.enabled),
            "horizons_us": [int(item) for item in self.config.horizons_us],
            "decision_level": self._decision_groups() if self.config.enabled else {},
            "fill_markouts": self._fill_groups() if self.config.enabled else {},
            "signal_alignment": self._signal_alignment() if self.config.enabled else {},
            "warnings": list(self._warnings),
        }
        if include_records:
            max_records = self.config.max_records
            decision_records = self._records if max_records is None else self._records[:max_records]
            fill_records = self._fill_records if max_records is None else self._fill_records[:max_records]
            payload["debug_records"] = {
                "decision_records": [record.as_dict() for record in decision_records],
                "fill_records": [record.as_dict() for record in fill_records],
                "truncated": (
                    max_records is not None
                    and (len(self._records) > max_records or len(self._fill_records) > max_records)
                ),
                "total_decision_records": len(self._records),
                "total_fill_records": len(self._fill_records),
            }
        return payload

    def compact_horizon(self, horizon_us: int = 1_000_000) -> dict[str, object]:
        payload = self.as_dict(include_records=False)
        return compact_horizon_summary(payload, horizon_us=horizon_us)


def compact_horizon_summary(
    horizon_diagnostics: Mapping[str, object] | None,
    *,
    horizon_us: int = 1_000_000,
) -> dict[str, object]:
    if not horizon_diagnostics or not horizon_diagnostics.get("enabled"):
        return {
            "actual_path_equity_delta_mean": None,
            "carry_mark_equity_delta_mean": None,
            "fill_net_markout_bps_mean": None,
        }
    horizon_key = str(horizon_us)
    decision = horizon_diagnostics.get("decision_level")
    fill = horizon_diagnostics.get("fill_markouts")
    decision_all = None
    fill_all = None
    if isinstance(decision, Mapping):
        by_horizon = decision.get("by_horizon")
        if isinstance(by_horizon, Mapping):
            groups = by_horizon.get(horizon_key)
            if groups is None and by_horizon:
                groups = by_horizon.get(str(max(int(key) for key in by_horizon.keys())))
            if isinstance(groups, Mapping):
                decision_all = groups.get("all")
    if isinstance(fill, Mapping):
        by_horizon = fill.get("by_horizon")
        if isinstance(by_horizon, Mapping):
            groups = by_horizon.get(horizon_key)
            if groups is None and by_horizon:
                groups = by_horizon.get(str(max(int(key) for key in by_horizon.keys())))
            if isinstance(groups, Mapping):
                fill_all = groups.get("all")
    return {
        "actual_path_equity_delta_mean": (
            decision_all.get("actual_path_equity_delta_mean") if isinstance(decision_all, Mapping) else None
        ),
        "carry_mark_equity_delta_mean": (
            decision_all.get("carry_mark_equity_delta_mean") if isinstance(decision_all, Mapping) else None
        ),
        "fill_net_markout_bps_mean": (
            fill_all.get("net_markout_bps_mean") if isinstance(fill_all, Mapping) else None
        ),
    }
