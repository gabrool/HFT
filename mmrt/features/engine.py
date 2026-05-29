"""Feature engine for the MMRT feature pipeline.

This module coordinates causal book and trade state, emits decision-time raw
feature vectors, and computes only ENGINE-owned cross/event-context features.
It consumes already-normalized event objects and does not parse market data,
build labels, apply transforms, or write storage artifacts.
"""

from dataclasses import dataclass
import math

import numpy as np

from mmrt.features import kernels as k
from mmrt.features.book_state import BookSnapshotInput, BookState, BookSummary
from mmrt.features.trade_state import (
    BUY_SIDE_CODE,
    SELL_SIDE_CODE,
    UNKNOWN_SIDE_CODE,
    TradeInput,
    TradeState,
    TradeSummary,
)
from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSource,
    feature_indices_by_source,
    feature_spec_by_name,
)

BOOK_EVENT_CODE = 1
TRADE_EVENT_CODE = 2

DECISION_STRIDE_US = 500_000

WINDOW_100MS_US = 100_000
WINDOW_200MS_US = 200_000
WINDOW_500MS_US = 500_000
WINDOW_1000MS_US = 1_000_000
WINDOW_3000MS_US = 3_000_000

ENGINE_EVENT_WINDOWS_US = (WINDOW_100MS_US, WINDOW_200MS_US, WINDOW_500MS_US, WINDOW_1000MS_US, WINDOW_3000MS_US)
DEFAULT_EVENT_HISTORY_CAPACITY = 131_072
FLOAT_EPS = 1e-12

CROSS_FEATURE_INDICES = feature_indices_by_source(FeatureSource.CROSS)
CROSS_FEATURE_NAMES = tuple(FEATURE_NAMES[i] for i in CROSS_FEATURE_INDICES)
CROSS_FEATURE_NAME_SET = frozenset(CROSS_FEATURE_NAMES)
EVENT_CONTEXT_FEATURE_INDICES = feature_indices_by_source(FeatureSource.EVENT_CONTEXT)
EVENT_CONTEXT_FEATURE_NAMES = tuple(FEATURE_NAMES[i] for i in EVENT_CONTEXT_FEATURE_INDICES)
EVENT_CONTEXT_FEATURE_NAME_SET = frozenset(EVENT_CONTEXT_FEATURE_NAMES)
ENGINE_FEATURE_INDICES = CROSS_FEATURE_INDICES + EVENT_CONTEXT_FEATURE_INDICES
ENGINE_FEATURE_NAMES = CROSS_FEATURE_NAMES + EVENT_CONTEXT_FEATURE_NAMES
ENGINE_FEATURE_NAME_SET = frozenset(ENGINE_FEATURE_NAMES)

assert CROSS_FEATURE_INDICES
assert EVENT_CONTEXT_FEATURE_INDICES
assert all(FEATURE_SPECS[i].source == FeatureSource.CROSS for i in CROSS_FEATURE_INDICES)
assert all(FEATURE_SPECS[i].source == FeatureSource.EVENT_CONTEXT for i in EVENT_CONTEXT_FEATURE_INDICES)
assert len(ENGINE_FEATURE_NAMES) == len(ENGINE_FEATURE_NAME_SET)
for n in (
    "vwap_vs_mid_bps_200000us", "vwap_vs_mid_bps_500000us", "absorption_bid_200000us", "absorption_ask_200000us",
    "absorption_bid_500000us", "absorption_ask_500000us", "absorption_bid_1000000us", "absorption_ask_1000000us",
    "ofi_l1_pressure_over_depth_5bps_200000us", "ofi_l1_pressure_over_realized_vol_200000us",
    "ofi_l1_pressure_over_depth_5bps_500000us", "ofi_l1_pressure_over_realized_vol_500000us",
    "ofi_l1_pressure_over_depth_5bps_1000000us", "ofi_l1_pressure_over_realized_vol_1000000us",
    "post_buy_trade_ask_replenishment_200000us", "post_sell_trade_bid_replenishment_200000us",
    "opposite_side_replenishment_after_depletion_200000us", "same_side_replenishment_after_depletion_200000us",
    "trade_side_quote_response_asymmetry_500000us", "trade_impact_half_life_proxy",
):
    assert n in CROSS_FEATURE_NAME_SET
for n in ("log_dt_decision_us", "log_events_100000us", "log_events_200000us", "log_events_500000us", "log_events_1000000us", "log_events_3000000us"):
    assert n in EVENT_CONTEXT_FEATURE_NAME_SET


def _require_int(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(name)
    if positive and value <= 0:
        raise ValueError(name)
    if not positive and value < 0:
        raise ValueError(name)
    return value


def _require_positive_capacity(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name)
    return value




def _require_positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite float > 0")
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite float > 0")
    return out

def _require_feature_vector(out: np.ndarray) -> np.ndarray:
    arr = np.asarray(out)
    if arr.ndim != 1 or arr.shape[0] != FEATURE_COUNT:
        raise ValueError("feature_vector")
    return arr


def _finite(value: float) -> float:
    out = float(value)
    return out if math.isfinite(out) else 0.0


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) <= FLOAT_EPS:
        return default
    return _finite(num / den)


def _safe_log1p(value: float) -> float:
    if value <= 0:
        return 0.0
    return _finite(math.log1p(value))


def _safe_bps_change(new_value: float, old_value: float) -> float:
    return _finite(k.bps_change(new_value, old_value))


@dataclass(frozen=True, slots=True)
class FeatureEngineConfig:
    decision_stride_us: int = DECISION_STRIDE_US
    event_history_capacity: int = DEFAULT_EVENT_HISTORY_CAPACITY

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_stride_us", _require_int(self.decision_stride_us, "decision_stride_us", positive=True))
        object.__setattr__(self, "event_history_capacity", _require_positive_capacity(self.event_history_capacity, "event_history_capacity"))


@dataclass(frozen=True, slots=True)
class EngineDecision:
    decision_index: int
    local_ts_us: int
    ts_us: int
    event_seq: int
    raw_mid: float
    feature_vector: np.ndarray
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_ts_us", _require_int(self.local_ts_us, "local_ts_us", positive=True))
        object.__setattr__(self, "ts_us", _require_int(self.ts_us, "ts_us", positive=True))
        if self.event_seq != -1:
            object.__setattr__(self, "event_seq", _require_int(self.event_seq, "event_seq"))
        object.__setattr__(self, "raw_mid", _require_positive_finite_float(self.raw_mid, "raw_mid"))
        arr = _require_feature_vector(self.feature_vector)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason")
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64).copy())
        object.__setattr__(self, "feature_vector", arr)


class EventHistory:
    def __init__(self, capacity: int = DEFAULT_EVENT_HISTORY_CAPACITY):
        self.capacity = _require_positive_capacity(capacity, "capacity")
        self.ts_us = np.zeros(self.capacity, dtype=np.int64)
        self.event_kind = np.zeros(self.capacity, dtype=np.int8)
        self.reset()

    def append(self, local_ts_us: int, event_kind: int) -> None:
        _require_int(local_ts_us, "local_ts_us", positive=True)
        _require_int(event_kind, "event_kind", positive=True)
        self.ts_us[self._head] = local_ts_us
        self.event_kind[self._head] = event_kind
        self._head = (self._head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def reset(self) -> None:
        self._head = 0
        self.size = 0

    def ordered_ts(self) -> np.ndarray:
        if self.size == 0:
            return self.ts_us[:0].copy()
        if self.size < self.capacity:
            return self.ts_us[:self.size].copy()
        return np.concatenate((self.ts_us[self._head:], self.ts_us[:self._head])).copy()

    def ordered_kinds(self) -> np.ndarray:
        if self.size == 0:
            return self.event_kind[:0].copy()
        if self.size < self.capacity:
            return self.event_kind[:self.size].copy()
        return np.concatenate((self.event_kind[self._head:], self.event_kind[:self._head])).copy()

    def count_in_window(self, now_us: int, window_us: int) -> int:
        _require_int(now_us, "now_us", positive=True)
        _require_int(window_us, "window_us", positive=True)
        ts = self.ordered_ts()
        if ts.size == 0:
            return 0
        lo = now_us - window_us
        start = int(np.searchsorted(ts, lo, side="left"))
        end = int(np.searchsorted(ts, now_us, side="right"))
        return max(0, end - start)


class FeatureEngine:
    def __init__(self, config: FeatureEngineConfig | None = None):
        self.config = config or FeatureEngineConfig()
        self.book_state = BookState()
        self.trade_state = TradeState()
        self.event_history = EventHistory(self.config.event_history_capacity)
        self.reset()

    def reset(self) -> None:
        self.book_state.reset(); self.trade_state.reset(); self.event_history.reset()
        self.decision_count = 0
        self.last_event_local_ts_us = None
        self.last_decision_local_ts_us = None
        self.next_decision_local_ts_us = None

    def has_book(self) -> bool: return self.book_state.has_book()
    def has_trades(self) -> bool: return self.trade_state.has_trades()
    def is_ready(self) -> bool: return self.has_book() and self.has_trades()

    def on_trade(self, trade: TradeInput) -> None:
        if not isinstance(trade, TradeInput): raise TypeError("trade")
        self._validate_event_time(trade.local_ts_us)
        self.event_history.append(trade.local_ts_us, TRADE_EVENT_CODE)
        self.trade_state.apply_trade(trade)
        self.last_event_local_ts_us = trade.local_ts_us
        return None

    def on_book_snapshot(self, snapshot: BookSnapshotInput) -> EngineDecision | None:
        if not isinstance(snapshot, BookSnapshotInput): raise TypeError("snapshot")
        self._validate_event_time(snapshot.local_ts_us)
        self.event_history.append(snapshot.local_ts_us, BOOK_EVENT_CODE)
        self.book_state.apply_snapshot(snapshot)
        self.last_event_local_ts_us = snapshot.local_ts_us
        if not self.is_ready(): return None
        if self.next_decision_local_ts_us is None:
            self.next_decision_local_ts_us = snapshot.local_ts_us
        if snapshot.local_ts_us < self.next_decision_local_ts_us:
            return None
        decision = self._emit_decision(snapshot.local_ts_us, snapshot.ts_us, snapshot.event_seq)
        while self.next_decision_local_ts_us <= snapshot.local_ts_us:
            self.next_decision_local_ts_us += self.config.decision_stride_us
        return decision

    def _validate_event_time(self, local_ts_us: int) -> None:
        _require_int(local_ts_us, "local_ts_us", positive=True)
        if self.last_event_local_ts_us is not None and local_ts_us < self.last_event_local_ts_us:
            raise ValueError("local_ts_us")

    def _emit_decision(self, local_ts_us: int, ts_us: int, event_seq: int) -> EngineDecision:
        prev = self.last_decision_local_ts_us
        summary = self.book_state.current_summary()
        raw_mid = summary.mid
        fv = self._build_feature_vector_for_decision(local_ts_us, prev)
        d = EngineDecision(self.decision_count, local_ts_us, ts_us, event_seq, raw_mid, fv, "book_stride")
        self.decision_count += 1
        self.last_decision_local_ts_us = local_ts_us
        return d

    def _build_feature_vector_for_decision(self, now_us: int, previous_decision_local_ts_us: int | None) -> np.ndarray:
        out = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self.book_state.fill_book_features(out)
        self.trade_state.fill_trade_features(out, as_of_local_ts_us=now_us)
        self.fill_engine_features(out, as_of_local_ts_us=now_us, previous_decision_local_ts_us=previous_decision_local_ts_us)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("non-finite")
        return out

    def build_feature_vector(self, as_of_local_ts_us: int | None = None) -> np.ndarray:
        if not self.is_ready(): raise ValueError("not_ready")
        now = self.book_state.last_local_ts_us if as_of_local_ts_us is None else _require_int(as_of_local_ts_us, "as_of_local_ts_us", positive=True)
        if now < self.book_state.last_local_ts_us or now < self.trade_state.last_local_ts_us:
            raise ValueError("as_of_local_ts_us")
        if now != self.book_state.last_local_ts_us:
            raise ValueError("as_of_local_ts_us")
        return self._build_feature_vector_for_decision(now, self.last_decision_local_ts_us)

    def _trade_values(self, field_name: str, window_us: int, now_us: int) -> np.ndarray:
        return self.trade_state.history.values_in_window(field_name, now_us, window_us)
    def _trade_sum(self, field_name: str, window_us: int, now_us: int) -> float: return float(np.sum(self._trade_values(field_name, window_us, now_us)))
    def _trade_buy_notional(self, window_us: int, now_us: int) -> float: return self._trade_sum("buy_notional", window_us, now_us)
    def _trade_sell_notional(self, window_us: int, now_us: int) -> float: return self._trade_sum("sell_notional", window_us, now_us)
    def _trade_total_notional(self, window_us: int, now_us: int) -> float: return self._trade_buy_notional(window_us, now_us) + self._trade_sell_notional(window_us, now_us)
    def _trade_vwap(self, window_us: int, now_us: int) -> float:
        amount = self._trade_values("amount", window_us, now_us); notional = self._trade_values("notional", window_us, now_us)
        den = float(np.sum(amount)); return 0.0 if den <= FLOAT_EPS else _safe_div(float(np.sum(notional)), den, 0.0)
    def _book_values(self, field_name: str, window_us: int, now_us: int) -> np.ndarray:
        return self.book_state.history.values_in_window(field_name, now_us, window_us)
    def _book_sum(self, field_name: str, window_us: int, now_us: int) -> float: return float(np.sum(self._book_values(field_name, window_us, now_us)))
    def _book_mean(self, field_name: str, window_us: int, now_us: int) -> float:
        v = self._book_values(field_name, window_us, now_us); return float(np.mean(v)) if v.size else 0.0
    def _book_realized_vol_bps(self, field_name: str, window_us: int, now_us: int) -> float:
        vals = self._book_values(field_name, window_us, now_us)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size < 2: return 0.0
        ret = np.array([_safe_bps_change(vals[i], vals[i - 1]) for i in range(1, vals.size)], dtype=np.float64)
        return float(np.std(ret)) if ret.size else 0.0
    def _book_current_summary(self) -> BookSummary: return self.book_state.current_summary()
    def _replenishment_ratio(self, add_sum: float, rem_sum: float) -> float: return _safe_div(add_sum, max(add_sum + rem_sum, FLOAT_EPS), 0.0)
    def _current_depth_size(self) -> float: return self._book_current_summary().total_depth_5bps_size
    def _current_depth_notional(self) -> float: return self._book_current_summary().total_depth_5bps_notional

    def fill_engine_features(self, out: np.ndarray, *, as_of_local_ts_us: int, previous_decision_local_ts_us: int | None = None) -> np.ndarray:
        if not self.is_ready():
            raise ValueError("not_ready")
        arr = _require_feature_vector(out)
        assigned = set()
        now = _require_int(as_of_local_ts_us, "as_of_local_ts_us", positive=True)

        def setf(name: str, value: float) -> None:
            if name not in ENGINE_FEATURE_NAME_SET:
                raise ValueError(name)
            arr[feature_spec_by_name(name).index] = _finite(value)
            assigned.add(name)

        s = self._book_current_summary()
        mid = s.mid
        vwap_200 = self._trade_vwap(WINDOW_200MS_US, now)
        vwap_500 = self._trade_vwap(WINDOW_500MS_US, now)
        vwap_vs_mid_200 = 0.0 if mid <= 0.0 or vwap_200 <= 0.0 else _safe_bps_change(vwap_200, mid)
        vwap_vs_mid_500 = 0.0 if mid <= 0.0 or vwap_500 <= 0.0 else _safe_bps_change(vwap_500, mid)
        setf("vwap_vs_mid_bps_200000us", vwap_vs_mid_200)
        setf("vwap_vs_mid_bps_500000us", vwap_vs_mid_500)

        for w in (WINDOW_200MS_US, WINDOW_500MS_US, WINDOW_1000MS_US):
            buy_n = self._trade_buy_notional(w, now)
            sell_n = self._trade_sell_notional(w, now)
            total = buy_n + sell_n
            bid_add = self._book_sum("bid_l1_add", w, now)
            bid_rem = self._book_sum("bid_l1_rem", w, now)
            ask_add = self._book_sum("ask_l1_add", w, now)
            ask_rem = self._book_sum("ask_l1_rem", w, now)
            bid_rr = self._replenishment_ratio(bid_add, bid_rem)
            ask_rr = self._replenishment_ratio(ask_add, ask_rem)
            ab = _safe_div(sell_n, max(total, FLOAT_EPS), 0.0) * bid_rr
            aa = _safe_div(buy_n, max(total, FLOAT_EPS), 0.0) * ask_rr
            setf(f"absorption_bid_{w}us", ab)
            setf(f"absorption_ask_{w}us", aa)
            ofi_pressure = self._book_sum("ofi_l1", w, now)
            p_over_d = _safe_div(ofi_pressure, max(s.total_depth_5bps_size, FLOAT_EPS), 0.0)
            setf(f"ofi_l1_pressure_over_depth_5bps_{w}us", p_over_d)
            rv = self._book_realized_vol_bps("microprice", w, now)
            setf(f"ofi_l1_pressure_over_realized_vol_{w}us", 0.0 if rv <= FLOAT_EPS else _safe_div(p_over_d, max(rv, FLOAT_EPS), 0.0))
            if w == WINDOW_500MS_US:
                buy_share = _safe_div(buy_n, max(total, FLOAT_EPS), 0.0)
                sell_share = _safe_div(sell_n, max(total, FLOAT_EPS), 0.0)
                setf("trade_side_quote_response_asymmetry_500000us", buy_share * ask_rr - sell_share * bid_rr)
            if w == WINDOW_200MS_US:
                setf("post_buy_trade_ask_replenishment_200000us", _safe_div(buy_n, max(total, FLOAT_EPS), 0.0) * ask_rr)
                setf("post_sell_trade_bid_replenishment_200000us", _safe_div(sell_n, max(total, FLOAT_EPS), 0.0) * bid_rr)
                depth = max(s.total_depth_5bps_size, FLOAT_EPS)
                setf("same_side_replenishment_after_depletion_200000us", (min(bid_add, bid_rem) + min(ask_add, ask_rem)) / depth)
                setf("opposite_side_replenishment_after_depletion_200000us", (min(bid_add, ask_rem) + min(ask_add, bid_rem)) / depth)

        impact_200 = abs(vwap_vs_mid_200)
        impact_500 = abs(vwap_vs_mid_500)
        setf("trade_impact_half_life_proxy", 0.0 if impact_200 <= FLOAT_EPS else min(max(impact_500 / impact_200, 0.0), 10.0))
        dt = 0 if previous_decision_local_ts_us is None else max(0, now - previous_decision_local_ts_us)
        setf("log_dt_decision_us", _safe_log1p(dt))
        setf("log_events_100000us", _safe_log1p(self.event_history.count_in_window(now, WINDOW_100MS_US)))
        setf("log_events_200000us", _safe_log1p(self.event_history.count_in_window(now, WINDOW_200MS_US)))
        setf("log_events_500000us", _safe_log1p(self.event_history.count_in_window(now, WINDOW_500MS_US)))
        setf("log_events_1000000us", _safe_log1p(self.event_history.count_in_window(now, WINDOW_1000MS_US)))
        setf("log_events_3000000us", _safe_log1p(self.event_history.count_in_window(now, WINDOW_3000MS_US)))

        missing = ENGINE_FEATURE_NAME_SET - assigned
        extra = assigned - ENGINE_FEATURE_NAME_SET
        if missing or extra: raise RuntimeError("feature assignment mismatch")
        return arr


def cross_feature_names() -> tuple[str, ...]: return CROSS_FEATURE_NAMES

def cross_feature_indices() -> tuple[int, ...]: return CROSS_FEATURE_INDICES

def event_context_feature_names() -> tuple[str, ...]: return EVENT_CONTEXT_FEATURE_NAMES

def event_context_feature_indices() -> tuple[int, ...]: return EVENT_CONTEXT_FEATURE_INDICES

def engine_owned_feature_names() -> tuple[str, ...]: return ENGINE_FEATURE_NAMES

def engine_owned_feature_indices() -> tuple[int, ...]: return ENGINE_FEATURE_INDICES


__all__ = [
    "BOOK_EVENT_CODE", "TRADE_EVENT_CODE", "DECISION_STRIDE_US", "ENGINE_EVENT_WINDOWS_US", "DEFAULT_EVENT_HISTORY_CAPACITY",
    "CROSS_FEATURE_INDICES", "CROSS_FEATURE_NAMES", "EVENT_CONTEXT_FEATURE_INDICES", "EVENT_CONTEXT_FEATURE_NAMES",
    "ENGINE_FEATURE_INDICES", "ENGINE_FEATURE_NAMES", "FeatureEngineConfig", "EngineDecision", "EventHistory", "FeatureEngine",
    "cross_feature_names", "cross_feature_indices", "event_context_feature_names", "event_context_feature_indices",
    "engine_owned_feature_names", "engine_owned_feature_indices",
]
