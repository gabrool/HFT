"""Rolling non-alpha control features for execution RL observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Sequence

import numpy as np

from mmrt.contracts import AggressorSide
from mmrt.execution.obs_schema import CONTROL_FIELDS


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float >= 0")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite float >= 0")
    return out


def _require_positive_float(value: float, name: str) -> float:
    out = _require_nonnegative_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return out


def _require_positive_int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive ints")
    out = tuple(values)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return tuple(_require_positive_int(int(item), f"{name} item") for item in out)


def control_observation_fields() -> tuple[str, ...]:
    return CONTROL_FIELDS


def depth_shape_features(
    *,
    book_ptr: int,
    book_bid_sizes: np.ndarray,
    book_ask_sizes: np.ndarray,
    size_epsilon: float = 1e-12,
    levels: int = 5,
) -> dict[str, float]:
    book_ptr = _require_positive_or_zero_int(book_ptr, "book_ptr")
    levels = _require_positive_int(levels, "levels")
    size_epsilon = _require_positive_float(size_epsilon, "size_epsilon")
    if not isinstance(book_bid_sizes, np.ndarray) or not isinstance(book_ask_sizes, np.ndarray):
        raise ValueError("book size arrays must be NumPy arrays")
    if book_bid_sizes.ndim != 2 or book_ask_sizes.ndim != 2:
        raise ValueError("book size arrays must be rank-2")
    if book_bid_sizes.shape[0] != book_ask_sizes.shape[0]:
        raise ValueError("book size arrays must have matching row counts")
    if book_ptr >= book_bid_sizes.shape[0]:
        raise ValueError("book_ptr outside book size arrays")

    n_bid = min(levels, int(book_bid_sizes.shape[1]))
    n_ask = min(levels, int(book_ask_sizes.shape[1]))
    bid_qty = float(np.sum(book_bid_sizes[book_ptr, :n_bid], dtype=np.float64))
    ask_qty = float(np.sum(book_ask_sizes[book_ptr, :n_ask], dtype=np.float64))
    denom = max(bid_qty + ask_qty, size_epsilon)
    return {
        "depth_bid_qty_5": bid_qty,
        "depth_ask_qty_5": ask_qty,
        "depth_imbalance_5": (bid_qty - ask_qty) / denom,
    }


def _require_positive_or_zero_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


@dataclass(frozen=True, slots=True)
class ControlFeatureTrackerConfig:
    flow_windows_us: tuple[int, ...] = (200_000, 500_000, 1_000_000)
    touch_window_us: int = 1_000_000
    mid_return_windows_us: tuple[int, ...] = (200_000, 1_000_000)
    size_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "flow_windows_us", _require_positive_int_tuple(self.flow_windows_us, "flow_windows_us"))
        object.__setattr__(self, "touch_window_us", _require_positive_int(self.touch_window_us, "touch_window_us"))
        object.__setattr__(
            self,
            "mid_return_windows_us",
            _require_positive_int_tuple(self.mid_return_windows_us, "mid_return_windows_us"),
        )
        object.__setattr__(self, "size_epsilon", _require_positive_float(self.size_epsilon, "size_epsilon"))

    @property
    def max_lookback_us(self) -> int:
        return max((*self.flow_windows_us, self.touch_window_us, *self.mid_return_windows_us))


@dataclass(slots=True)
class _WindowTradeState:
    rows: Deque[tuple[int, float, float]] = field(default_factory=deque)
    signed_sum: float = 0.0
    abs_sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.rows.clear()
        self.signed_sum = 0.0
        self.abs_sum = 0.0
        self.count = 0

    def append(self, local_ts_us: int, signed_qty: float, abs_qty: float) -> None:
        self.rows.append((local_ts_us, signed_qty, abs_qty))
        self.signed_sum += signed_qty
        self.abs_sum += abs_qty
        self.count += 1

    def prune(self, current_local_ts_us: int, window_us: int) -> None:
        cutoff = current_local_ts_us - window_us
        while self.rows and self.rows[0][0] < cutoff:
            _, signed_qty, abs_qty = self.rows.popleft()
            self.signed_sum -= signed_qty
            self.abs_sum -= abs_qty
            self.count -= 1


@dataclass(slots=True)
class _WindowQtyState:
    rows: Deque[tuple[int, float]] = field(default_factory=deque)
    qty_sum: float = 0.0

    def reset(self) -> None:
        self.rows.clear()
        self.qty_sum = 0.0

    def append(self, local_ts_us: int, qty: float) -> None:
        if qty <= 0.0:
            return
        self.rows.append((local_ts_us, qty))
        self.qty_sum += qty

    def prune(self, current_local_ts_us: int, window_us: int) -> None:
        cutoff = current_local_ts_us - window_us
        while self.rows and self.rows[0][0] < cutoff:
            _, qty = self.rows.popleft()
            self.qty_sum -= qty


class ControlFeatureTracker:
    def __init__(self, config: ControlFeatureTrackerConfig = ControlFeatureTrackerConfig()) -> None:
        if not isinstance(config, ControlFeatureTrackerConfig):
            raise ValueError("config must be ControlFeatureTrackerConfig")
        self.config = config
        self._trade_windows = {window: _WindowTradeState() for window in config.flow_windows_us}
        self._touch_windows = {
            "bid_depletion": _WindowQtyState(),
            "bid_replenishment": _WindowQtyState(),
            "ask_depletion": _WindowQtyState(),
            "ask_replenishment": _WindowQtyState(),
        }
        self._mid_history: Deque[tuple[int, float]] = deque()
        self._last_bid_tick: int | None = None
        self._last_bid_size: float = 0.0
        self._last_ask_tick: int | None = None
        self._last_ask_size: float = 0.0

    @property
    def max_lookback_us(self) -> int:
        return self.config.max_lookback_us

    def reset(self) -> None:
        for state in self._trade_windows.values():
            state.reset()
        for state in self._touch_windows.values():
            state.reset()
        self._mid_history.clear()
        self._last_bid_tick = None
        self._last_bid_size = 0.0
        self._last_ask_tick = None
        self._last_ask_size = 0.0

    def record_trade(self, *, local_ts_us: int, side: AggressorSide | int, qty: float) -> None:
        local_ts_us = _require_positive_int(int(local_ts_us), "local_ts_us")
        qty = _require_nonnegative_float(qty, "qty")
        if qty <= 0.0:
            return
        signed_qty = _signed_trade_qty(side, qty)
        for window, state in self._trade_windows.items():
            state.append(local_ts_us, signed_qty, qty)
            state.prune(local_ts_us, window)

    def record_l2_top(
        self,
        *,
        local_ts_us: int,
        best_bid_tick: int,
        best_bid_size: float,
        best_ask_tick: int,
        best_ask_size: float,
    ) -> None:
        local_ts_us = _require_positive_int(int(local_ts_us), "local_ts_us")
        best_bid_tick = _require_positive_int(int(best_bid_tick), "best_bid_tick")
        best_ask_tick = _require_positive_int(int(best_ask_tick), "best_ask_tick")
        best_bid_size = _require_nonnegative_float(best_bid_size, "best_bid_size")
        best_ask_size = _require_nonnegative_float(best_ask_size, "best_ask_size")
        if best_bid_tick >= best_ask_tick:
            return

        if self._last_bid_tick is not None and self._last_ask_tick is not None:
            self._record_bid_touch_delta(local_ts_us, best_bid_tick, best_bid_size)
            self._record_ask_touch_delta(local_ts_us, best_ask_tick, best_ask_size)

        self._last_bid_tick = best_bid_tick
        self._last_bid_size = best_bid_size
        self._last_ask_tick = best_ask_tick
        self._last_ask_size = best_ask_size
        self._record_mid(local_ts_us, (best_bid_tick + best_ask_tick) * 0.5)
        self._prune_touch(local_ts_us)
        self._prune_mid(local_ts_us)

    def snapshot(self, current_local_ts_us: int) -> dict[str, float]:
        current_local_ts_us = _require_positive_int(int(current_local_ts_us), "current_local_ts_us")
        out: dict[str, float] = {}
        for window in self.config.flow_windows_us:
            state = self._trade_windows[window]
            state.prune(current_local_ts_us, window)
            suffix = f"{int(window / 1000)}ms"
            out[f"flow_signed_qty_{suffix}"] = float(state.signed_sum)
            out[f"flow_abs_qty_{suffix}"] = float(state.abs_sum)
            out[f"flow_trade_count_{suffix}"] = float(state.count)
            denom = max(state.abs_sum, self.config.size_epsilon)
            out[f"flow_imbalance_ratio_{suffix}"] = float(np.clip(state.signed_sum / denom, -1.0, 1.0))

        self._prune_touch(current_local_ts_us)
        bid_dep = self._touch_windows["bid_depletion"].qty_sum
        bid_rep = self._touch_windows["bid_replenishment"].qty_sum
        ask_dep = self._touch_windows["ask_depletion"].qty_sum
        ask_rep = self._touch_windows["ask_replenishment"].qty_sum
        out["bid_touch_depletion_ratio_1000ms"] = _ratio_or_zero(bid_dep, bid_dep + bid_rep, self.config.size_epsilon)
        out["bid_touch_replenishment_ratio_1000ms"] = _ratio_or_zero(bid_rep, bid_dep + bid_rep, self.config.size_epsilon)
        out["ask_touch_depletion_ratio_1000ms"] = _ratio_or_zero(ask_dep, ask_dep + ask_rep, self.config.size_epsilon)
        out["ask_touch_replenishment_ratio_1000ms"] = _ratio_or_zero(ask_rep, ask_dep + ask_rep, self.config.size_epsilon)

        self._prune_mid(current_local_ts_us)
        current_mid = self._latest_mid_at_or_before(current_local_ts_us)
        for window in self.config.mid_return_windows_us:
            suffix = f"{int(window / 1000)}ms"
            reference_mid = self._latest_mid_at_or_before(current_local_ts_us - window)
            value = 0.0
            if current_mid is not None and reference_mid is not None and current_mid > 0.0 and reference_mid > 0.0:
                value = (current_mid - reference_mid) / current_mid * 10_000.0
            out[f"recent_mid_return_bps_{suffix}"] = float(value)
        return out

    def _record_bid_touch_delta(self, local_ts_us: int, bid_tick: int, bid_size: float) -> None:
        assert self._last_bid_tick is not None
        if bid_tick == self._last_bid_tick:
            if bid_size < self._last_bid_size:
                self._touch_windows["bid_depletion"].append(local_ts_us, self._last_bid_size - bid_size)
            elif bid_size > self._last_bid_size:
                self._touch_windows["bid_replenishment"].append(local_ts_us, bid_size - self._last_bid_size)
        elif bid_tick < self._last_bid_tick:
            self._touch_windows["bid_depletion"].append(local_ts_us, self._last_bid_size)
        else:
            self._touch_windows["bid_replenishment"].append(local_ts_us, bid_size)

    def _record_ask_touch_delta(self, local_ts_us: int, ask_tick: int, ask_size: float) -> None:
        assert self._last_ask_tick is not None
        if ask_tick == self._last_ask_tick:
            if ask_size < self._last_ask_size:
                self._touch_windows["ask_depletion"].append(local_ts_us, self._last_ask_size - ask_size)
            elif ask_size > self._last_ask_size:
                self._touch_windows["ask_replenishment"].append(local_ts_us, ask_size - self._last_ask_size)
        elif ask_tick > self._last_ask_tick:
            self._touch_windows["ask_depletion"].append(local_ts_us, self._last_ask_size)
        else:
            self._touch_windows["ask_replenishment"].append(local_ts_us, ask_size)

    def _prune_touch(self, current_local_ts_us: int) -> None:
        for state in self._touch_windows.values():
            state.prune(current_local_ts_us, self.config.touch_window_us)

    def _record_mid(self, local_ts_us: int, mid_tick: float) -> None:
        if not math.isfinite(mid_tick) or mid_tick <= 0.0:
            return
        if self._mid_history and local_ts_us < self._mid_history[-1][0]:
            raise ValueError("L2 updates must be recorded in chronological order")
        if self._mid_history and local_ts_us == self._mid_history[-1][0]:
            self._mid_history[-1] = (local_ts_us, float(mid_tick))
        else:
            self._mid_history.append((local_ts_us, float(mid_tick)))

    def _prune_mid(self, current_local_ts_us: int) -> None:
        cutoff = current_local_ts_us - self.config.max_lookback_us
        while len(self._mid_history) >= 2 and self._mid_history[1][0] <= cutoff:
            self._mid_history.popleft()

    def _latest_mid_at_or_before(self, local_ts_us: int) -> float | None:
        for ts, mid in reversed(self._mid_history):
            if ts <= local_ts_us:
                return mid
        return None


def _ratio_or_zero(numer: float, denom: float, epsilon: float) -> float:
    if denom <= epsilon:
        return 0.0
    return float(numer / denom)


def _signed_trade_qty(side: AggressorSide | int, qty: float) -> float:
    if side == AggressorSide.BUY or side == 1:
        return qty
    if side == AggressorSide.SELL or side == -1:
        return -qty
    return 0.0


__all__ = [
    "ControlFeatureTrackerConfig",
    "ControlFeatureTracker",
    "control_observation_fields",
    "depth_shape_features",
]
