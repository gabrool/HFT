"""Trade-state feature computation for the MMRT feature pipeline.

This module consumes normalized trade events and maintains causal trade-only
state. It computes only TRADE-owned features from specs.py and exposes trade
summaries for later engine-level cross features. It does not parse market data
files, compute book features, build labels, apply transforms, or write storage
artifacts.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from mmrt.features import kernels as k
from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSource,
    feature_indices_by_source,
    feature_spec_by_name,
)

BUY_SIDE_CODE = 1
SELL_SIDE_CODE = -1
UNKNOWN_SIDE_CODE = 0

WINDOW_200MS_US = 200_000
WINDOW_500MS_US = 500_000
WINDOW_1000MS_US = 1_000_000
WINDOW_3000MS_US = 3_000_000

TRADE_WINDOWS_US = (WINDOW_200MS_US, WINDOW_500MS_US, WINDOW_1000MS_US, WINDOW_3000MS_US)
DEFAULT_HISTORY_CAPACITY = 65_536
FLOAT_EPS = 1e-12

TRADE_FEATURE_INDICES = feature_indices_by_source(FeatureSource.TRADE)
TRADE_FEATURE_NAMES = tuple(FEATURE_NAMES[i] for i in TRADE_FEATURE_INDICES)
TRADE_FEATURE_NAME_SET = frozenset(TRADE_FEATURE_NAMES)
ACTIVE_TRADE_FEATURES = {
    "trade_count_per_second_200000us",
    "trade_imbalance_notional_500000us",
    "trade_count_per_second_500000us",
    "zero_tick_fraction_1000000us",
    "trade_count_per_second_1000000us",
    "time_since_last_buy_trade_us",
    "time_since_last_sell_trade_us",
    "max_signed_trade_notional_usd_1000000us",
    "same_side_trade_cluster_notional_1000000us",
    "max_trade_silence_gap_3000000us",
    "trade_sign_entropy_3000000us",
}
assert TRADE_FEATURE_NAME_SET == ACTIVE_TRADE_FEATURES

assert TRADE_FEATURE_INDICES
assert all(FEATURE_SPECS[i].source == FeatureSource.TRADE for i in TRADE_FEATURE_INDICES)
assert len(TRADE_FEATURE_NAMES) == len(set(TRADE_FEATURE_NAMES))
assert all(win in TRADE_WINDOWS_US for spec in FEATURE_SPECS if spec.source == FeatureSource.TRADE for win in spec.windows_us)


def _require_int(value: int, name: str, *, positive: bool = False, allow_minus_one: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if allow_minus_one and value == -1:
        return value
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if not positive and value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_positive_capacity(capacity: int) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an int")
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    return capacity


def _require_finite_positive_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return out


def _coerce_side_code(side_code: int) -> int:
    if isinstance(side_code, bool) or not isinstance(side_code, int):
        raise TypeError("side_code must be an int")
    if side_code not in (SELL_SIDE_CODE, UNKNOWN_SIDE_CODE, BUY_SIDE_CODE):
        raise ValueError("side_code must be -1, 0, or 1")
    return side_code


def _finite(value: float) -> float:
    out = float(value)
    return out if math.isfinite(out) else 0.0


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) <= FLOAT_EPS:
        return default
    return _finite(num / den)


def _safe_log2_prob(p: float) -> float:
    if p <= 0.0:
        return 0.0
    return -p * math.log2(p)


@dataclass(frozen=True, slots=True)
class TradeInput:
    local_ts_us: int
    ts_us: int
    price: float
    amount: float
    side_code: int
    event_seq: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_ts_us", _require_int(self.local_ts_us, "local_ts_us", positive=True))
        object.__setattr__(self, "ts_us", _require_int(self.ts_us, "ts_us", positive=True))
        object.__setattr__(self, "price", _require_finite_positive_float(self.price, "price"))
        object.__setattr__(self, "amount", _require_finite_positive_float(self.amount, "amount"))
        object.__setattr__(self, "side_code", _coerce_side_code(self.side_code))
        object.__setattr__(self, "event_seq", _require_int(self.event_seq, "event_seq", allow_minus_one=True))


@dataclass(frozen=True, slots=True)
class TradeSummary:
    local_ts_us: int
    ts_us: int
    event_seq: int
    price: float
    amount: float
    notional: float
    side_code: int
    tick_sign: int
    last_trade_side_sign: int
    last_tick_sign: int
    trade_count: int
    buy_trade_count: int
    sell_trade_count: int
    unknown_trade_count: int


class TradeHistory:
    FIELDS = (
        "ts_us",
        "notional",
        "signed_notional",
        "side_code",
        "tick_sign",
        "buy_notional",
        "sell_notional",
    )

    def __init__(self, capacity: int = DEFAULT_HISTORY_CAPACITY):
        self.capacity = _require_positive_capacity(capacity)
        self.size = 0
        self._head = 0
        self._data = {
            "ts_us": np.zeros(self.capacity, dtype=np.int64),
            "side_code": np.zeros(self.capacity, dtype=np.int64),
            "tick_sign": np.zeros(self.capacity, dtype=np.int64),
            "notional": np.zeros(self.capacity, dtype=np.float64),
            "signed_notional": np.zeros(self.capacity, dtype=np.float64),
            "buy_notional": np.zeros(self.capacity, dtype=np.float64),
            "sell_notional": np.zeros(self.capacity, dtype=np.float64),
        }

    def append(self, **kwargs: float | int) -> None:
        for f in self.FIELDS:
            if f not in kwargs:
                raise KeyError(f"missing field {f}")
        idx = self._head
        for f in self.FIELDS:
            self._data[f][idx] = kwargs[f]
        self._head = (self._head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ordered_slice(self, field_name: str) -> np.ndarray:
        data = self._data[field_name]
        if self.size == 0:
            return data[:0]
        if self.size < self.capacity:
            return data[:self.size]
        return np.concatenate((data[self._head:], data[:self._head]))

    def ordered_ts(self) -> np.ndarray:
        return self.ordered_slice("ts_us")

    def values_in_window(self, field_name: str, now_us: int, window_us: int) -> np.ndarray:
        ts = self.ordered_ts()
        if ts.size == 0:
            return self.ordered_slice(field_name)
        lo = now_us - window_us
        start = int(np.searchsorted(ts, lo, side="left"))
        end = int(np.searchsorted(ts, now_us, side="right"))
        return self.ordered_slice(field_name)[start:end]

    def ts_in_window(self, now_us: int, window_us: int) -> np.ndarray:
        ts = self.ordered_ts()
        if ts.size == 0:
            return ts
        lo = now_us - window_us
        start = int(np.searchsorted(ts, lo, side="left"))
        end = int(np.searchsorted(ts, now_us, side="right"))
        return ts[start:end]

    def asof_value(self, field_name: str, query_ts_us: int, default: float = 0.0) -> float:
        ts = self.ordered_ts()
        if ts.size == 0 or query_ts_us <= 0:
            return float(default)
        idx = k.asof_index_right(ts.astype(np.int64, copy=False), int(query_ts_us))
        if idx < 0:
            return float(default)
        val = self.ordered_slice(field_name)[idx]
        return float(val) if np.isfinite(val) else float(default)


class TradeState:
    def __init__(self, history_capacity: int = DEFAULT_HISTORY_CAPACITY):
        self._history_capacity = _require_positive_capacity(history_capacity)
        self.history = TradeHistory(self._history_capacity)
        self.reset()

    def reset(self) -> None:
        self.history = TradeHistory(self._history_capacity)
        self.trade_count = 0
        self.buy_trade_count = 0
        self.sell_trade_count = 0
        self.unknown_trade_count = 0
        self.last_local_ts_us = None
        self.last_trade = None
        self.last_price = None
        self.last_side_code = UNKNOWN_SIDE_CODE
        self.last_tick_sign = 0
        self.last_buy_trade_ts_us = None
        self.last_sell_trade_ts_us = None

    def has_trades(self) -> bool:
        return self.trade_count > 0

    def _tick_sign(self, price: float) -> int:
        if self.last_price is None:
            return 0
        if price > self.last_price:
            return 1
        if price < self.last_price:
            return -1
        return 0

    def apply_trade(self, trade: TradeInput) -> TradeSummary:
        if not isinstance(trade, TradeInput):
            raise TypeError("trade must be TradeInput")
        if self.last_local_ts_us is not None and trade.local_ts_us < self.last_local_ts_us:
            raise ValueError("local_ts_us must be nondecreasing")
        notional = trade.price * trade.amount
        tick_sign = self._tick_sign(trade.price)
        signed_notional = trade.side_code * notional
        self.trade_count += 1
        if trade.side_code == BUY_SIDE_CODE:
            self.buy_trade_count += 1
            self.last_buy_trade_ts_us = trade.local_ts_us
        elif trade.side_code == SELL_SIDE_CODE:
            self.sell_trade_count += 1
            self.last_sell_trade_ts_us = trade.local_ts_us
        else:
            self.unknown_trade_count += 1
        self.history.append(
            ts_us=trade.local_ts_us,
            notional=notional,
            signed_notional=signed_notional,
            side_code=trade.side_code,
            tick_sign=tick_sign,
            buy_notional=notional if trade.side_code == BUY_SIDE_CODE else 0.0,
            sell_notional=notional if trade.side_code == SELL_SIDE_CODE else 0.0,
        )
        self.last_local_ts_us = trade.local_ts_us
        self.last_trade = trade
        self.last_price = trade.price
        self.last_side_code = trade.side_code
        self.last_tick_sign = tick_sign
        return self.current_summary()

    def current_summary(self) -> TradeSummary:
        if not self.has_trades() or self.last_trade is None:
            raise ValueError("no trades")
        t = self.last_trade
        return TradeSummary(
            t.local_ts_us,
            t.ts_us,
            t.event_seq,
            t.price,
            t.amount,
            t.price * t.amount,
            t.side_code,
            self.last_tick_sign,
            self.last_side_code,
            self.last_tick_sign,
            self.trade_count,
            self.buy_trade_count,
            self.sell_trade_count,
            self.unknown_trade_count,
        )

    def _window_values(self, field_name: str, window_us: int, now_us: int | None = None) -> np.ndarray:
        now = self.last_local_ts_us if now_us is None else now_us
        return self.history.values_in_window(field_name, now, window_us)

    def _window_ts(self, window_us: int, now_us: int | None = None) -> np.ndarray:
        now = self.last_local_ts_us if now_us is None else now_us
        return self.history.ts_in_window(now, window_us)

    def _trade_count_per_second(self, window_us: int, now_us: int) -> float:
        return _safe_div(float(self._window_ts(window_us, now_us).size), window_us / 1e6)

    def _trade_imbalance_notional(self, window_us: int, now_us: int) -> float:
        buy = float(np.sum(self._window_values("buy_notional", window_us, now_us)))
        sell = float(np.sum(self._window_values("sell_notional", window_us, now_us)))
        return _safe_div(buy - sell, buy + sell)

    def _zero_tick_fraction(self, window_us: int, now_us: int) -> float:
        tick = self._window_values("tick_sign", window_us, now_us)
        return _safe_div(float(np.sum(tick == 0)), float(tick.size))

    def _max_signed_trade_notional(self, window_us: int, now_us: int) -> float:
        signed = self._window_values("signed_notional", window_us, now_us)
        if signed.size == 0:
            return 0.0
        idx = int(np.argmax(np.abs(signed)))
        return float(signed[idx])

    def _max_trade_silence_gap(self, window_us: int, now_us: int) -> float:
        ts = self._window_ts(window_us, now_us)
        if ts.size < 2:
            return 0.0
        return float(np.max(np.diff(ts)))

    def _trade_sign_entropy(self, window_us: int, now_us: int) -> float:
        side = self._window_values("side_code", window_us, now_us)
        n = side.size
        if n == 0:
            return 0.0
        counts = (np.sum(side == BUY_SIDE_CODE), np.sum(side == SELL_SIDE_CODE), np.sum(side == UNKNOWN_SIDE_CODE))
        ent = 0.0
        for c in counts:
            ent += _safe_log2_prob(c / n)
        return _safe_div(ent, math.log2(3.0))

    def _same_side_trade_cluster_notional(self, window_us: int, now_us: int) -> float:
        side = self._window_values("side_code", window_us, now_us)
        notional = self._window_values("notional", window_us, now_us)
        best = 0.0
        cur = 0.0
        cur_side = 0
        for s, n in zip(side, notional):
            si = int(s)
            if si == 0:
                cur = 0.0
                cur_side = 0
                continue
            if si == cur_side:
                cur += float(n)
            else:
                cur_side = si
                cur = float(n)
            if cur > best:
                best = cur
        return best

    def fill_trade_features(self, out: np.ndarray, *, as_of_local_ts_us: int | None = None) -> np.ndarray:
        if not self.has_trades():
            raise ValueError("no trades")
        arr = np.asarray(out)
        if arr.shape != (FEATURE_COUNT,):
            raise ValueError("out shape mismatch")
        now = self.last_local_ts_us if as_of_local_ts_us is None else _require_int(as_of_local_ts_us, "as_of_local_ts_us", positive=True)
        if now < self.last_local_ts_us:
            raise ValueError("as_of_local_ts_us cannot be earlier than latest trade")
        assigned: set[str] = set()

        def setf(name: str, value: float) -> None:
            if name not in TRADE_FEATURE_NAME_SET:
                raise ValueError(name)
            arr[feature_spec_by_name(name).index] = _finite(value)
            assigned.add(name)

        setf("trade_count_per_second_200000us", self._trade_count_per_second(WINDOW_200MS_US, now))
        setf("trade_imbalance_notional_500000us", self._trade_imbalance_notional(WINDOW_500MS_US, now))
        setf("trade_count_per_second_500000us", self._trade_count_per_second(WINDOW_500MS_US, now))
        setf("zero_tick_fraction_1000000us", self._zero_tick_fraction(WINDOW_1000MS_US, now))
        setf("trade_count_per_second_1000000us", self._trade_count_per_second(WINDOW_1000MS_US, now))
        setf("time_since_last_buy_trade_us", float(0 if self.last_buy_trade_ts_us is None else now - self.last_buy_trade_ts_us))
        setf("time_since_last_sell_trade_us", float(0 if self.last_sell_trade_ts_us is None else now - self.last_sell_trade_ts_us))
        setf("max_signed_trade_notional_usd_1000000us", self._max_signed_trade_notional(WINDOW_1000MS_US, now))
        setf("same_side_trade_cluster_notional_1000000us", self._same_side_trade_cluster_notional(WINDOW_1000MS_US, now))
        setf("max_trade_silence_gap_3000000us", self._max_trade_silence_gap(WINDOW_3000MS_US, now))
        setf("trade_sign_entropy_3000000us", self._trade_sign_entropy(WINDOW_3000MS_US, now))
        missing = TRADE_FEATURE_NAME_SET - assigned
        extra = assigned - TRADE_FEATURE_NAME_SET
        if missing or extra:
            raise RuntimeError("trade feature assignment mismatch")
        return arr

    def trade_feature_vector(self, *, as_of_local_ts_us: int | None = None) -> np.ndarray:
        out = np.zeros(FEATURE_COUNT, dtype=np.float64)
        return self.fill_trade_features(out, as_of_local_ts_us=as_of_local_ts_us)


def trade_owned_feature_names() -> tuple[str, ...]:
    return TRADE_FEATURE_NAMES


def trade_owned_feature_indices() -> tuple[int, ...]:
    return TRADE_FEATURE_INDICES


__all__ = (
    "BUY_SIDE_CODE", "SELL_SIDE_CODE", "UNKNOWN_SIDE_CODE", "TRADE_WINDOWS_US", "DEFAULT_HISTORY_CAPACITY",
    "TRADE_FEATURE_INDICES", "TRADE_FEATURE_NAMES", "ACTIVE_TRADE_FEATURES", "TradeInput", "TradeSummary", "TradeHistory", "TradeState",
    "trade_owned_feature_names", "trade_owned_feature_indices",
)
