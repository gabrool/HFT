"""Causal fill-aware adverse-selection features and labels for execution tapes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from mmrt.execution.contracts import FillReason, OrderSide, QueueModelMode, SymbolSpec
from mmrt.execution.execution_tape import (
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    ExecutionTape,
)
from mmrt.execution.queue_model import QueueModelConfig


__all__ = [
    "VPINConfig",
    "KyleLambdaConfig",
    "CounterfactualQuoteConfig",
    "AdverseSelectionConfig",
    "CounterfactualFillResult",
    "AdverseSelectionDataset",
    "VPINState",
    "RollingKyleLambdaState",
    "build_adverse_selection_dataset",
    "summarize_adverse_selection_dataset",
    "adverse_selection_feature_names",
    "adverse_selection_label_names",
]

_EPS = 1e-12


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


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_probability(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _positive_int_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive ints")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of positive ints") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_require_positive_int(value, f"{name}[{i}]") for i, value in enumerate(result))


def _positive_float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive floats")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of positive floats") from exc
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_require_positive_float(value, f"{name}[{i}]") for i, value in enumerate(result))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) <= _EPS:
        return default
    return num / den


def _bps_from_ticks(delta_ticks: float, ref_tick: float) -> float:
    return _safe_div(delta_ticks, ref_tick, 0.0) * 10_000.0


@dataclass(frozen=True, slots=True)
class VPINConfig:
    bucket_volume: float = 50.0
    num_buckets: int = 50
    min_completed_buckets: int = 10
    use_notional_volume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket_volume", _require_positive_float(self.bucket_volume, "bucket_volume"))
        object.__setattr__(self, "num_buckets", _require_positive_int(self.num_buckets, "num_buckets"))
        min_completed = _require_positive_int(self.min_completed_buckets, "min_completed_buckets")
        if min_completed > self.num_buckets:
            raise ValueError("min_completed_buckets must be <= num_buckets")
        object.__setattr__(self, "min_completed_buckets", min_completed)
        object.__setattr__(self, "use_notional_volume", _require_bool(self.use_notional_volume, "use_notional_volume"))


@dataclass(frozen=True, slots=True)
class KyleLambdaConfig:
    sample_interval_us: int = 500_000
    response_horizon_us: int = 1_000_000
    windows_us: tuple[int, ...] = (10_000_000, 30_000_000)
    min_samples: int = 5
    use_notional_flow: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_interval_us", _require_positive_int(self.sample_interval_us, "sample_interval_us"))
        object.__setattr__(self, "response_horizon_us", _require_positive_int(self.response_horizon_us, "response_horizon_us"))
        object.__setattr__(self, "windows_us", _positive_int_tuple(self.windows_us, "windows_us"))
        object.__setattr__(self, "min_samples", _require_positive_int(self.min_samples, "min_samples"))
        object.__setattr__(self, "use_notional_flow", _require_bool(self.use_notional_flow, "use_notional_flow"))


@dataclass(frozen=True, slots=True)
class CounterfactualQuoteConfig:
    quote_distance_ticks: int = 0
    order_qty: float = 0.001
    fill_horizon_us: int = 1_000_000
    adverse_horizon_us: int = 1_000_000
    toxic_threshold_bps: float = 0.0
    queue_model: QueueModelConfig = QueueModelConfig(mode=QueueModelMode.BALANCED)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_distance_ticks", _require_nonnegative_int(self.quote_distance_ticks, "quote_distance_ticks"))
        object.__setattr__(self, "order_qty", _require_positive_float(self.order_qty, "order_qty"))
        object.__setattr__(self, "fill_horizon_us", _require_positive_int(self.fill_horizon_us, "fill_horizon_us"))
        object.__setattr__(self, "adverse_horizon_us", _require_positive_int(self.adverse_horizon_us, "adverse_horizon_us"))
        object.__setattr__(self, "toxic_threshold_bps", _require_nonnegative_float(self.toxic_threshold_bps, "toxic_threshold_bps"))
        if not isinstance(self.queue_model, QueueModelConfig):
            raise ValueError("queue_model must be QueueModelConfig")


@dataclass(frozen=True, slots=True)
class AdverseSelectionConfig:
    decision_interval_us: int = 500_000
    start_event_index: int | None = None
    max_decisions: int | None = None
    flow_windows_us: tuple[int, ...] = (200_000, 500_000, 1_000_000)
    vpin: VPINConfig = VPINConfig()
    kyle: KyleLambdaConfig = KyleLambdaConfig()
    quote: CounterfactualQuoteConfig = CounterfactualQuoteConfig()
    drop_incomplete_horizon: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_interval_us", _require_positive_int(self.decision_interval_us, "decision_interval_us"))
        if self.start_event_index is not None:
            object.__setattr__(self, "start_event_index", _require_nonnegative_int(self.start_event_index, "start_event_index"))
        if self.max_decisions is not None:
            object.__setattr__(self, "max_decisions", _require_positive_int(self.max_decisions, "max_decisions"))
        object.__setattr__(self, "flow_windows_us", _positive_int_tuple(self.flow_windows_us, "flow_windows_us"))
        if not isinstance(self.vpin, VPINConfig):
            raise ValueError("vpin must be VPINConfig")
        if not isinstance(self.kyle, KyleLambdaConfig):
            raise ValueError("kyle must be KyleLambdaConfig")
        if not isinstance(self.quote, CounterfactualQuoteConfig):
            raise ValueError("quote must be CounterfactualQuoteConfig")
        object.__setattr__(self, "drop_incomplete_horizon", _require_bool(self.drop_incomplete_horizon, "drop_incomplete_horizon"))


@dataclass(slots=True)
class VPINState:
    config: VPINConfig
    completed_imbalances: deque[float]
    current_buy_volume: float = 0.0
    current_sell_volume: float = 0.0
    current_total_volume: float = 0.0
    completed_bucket_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.config, VPINConfig):
            raise ValueError("config must be VPINConfig")

    def update_trade(self, *, side_code: int, price_tick: int, amount: float, tick_size: float) -> None:
        if side_code == 0 or amount <= 0.0:
            return
        sign = 1 if side_code > 0 else -1
        volume = float(amount)
        if self.config.use_notional_volume:
            volume = float(amount) * float(price_tick) * float(tick_size)
        if volume <= 0.0:
            return
        remaining = volume
        while remaining > _EPS:
            room = self.config.bucket_volume - self.current_total_volume
            take = min(remaining, room)
            if sign > 0:
                self.current_buy_volume += take
            else:
                self.current_sell_volume += take
            self.current_total_volume += take
            remaining -= take
            if self.current_total_volume >= self.config.bucket_volume - _EPS:
                imbalance = abs(self.current_buy_volume - self.current_sell_volume)
                self.completed_imbalances.append(float(imbalance))
                while len(self.completed_imbalances) > self.config.num_buckets:
                    self.completed_imbalances.popleft()
                self.completed_bucket_count += 1
                self.current_buy_volume = 0.0
                self.current_sell_volume = 0.0
                self.current_total_volume = 0.0

    def vpin(self) -> float:
        if len(self.completed_imbalances) < self.config.min_completed_buckets:
            return 0.0
        return float(sum(self.completed_imbalances) / (len(self.completed_imbalances) * self.config.bucket_volume))

    def is_valid(self) -> bool:
        return len(self.completed_imbalances) >= self.config.min_completed_buckets

    def current_bucket_completion(self) -> float:
        return min(max(_safe_div(self.current_total_volume, self.config.bucket_volume), 0.0), 1.0)

    def current_bucket_signed_imbalance(self) -> float:
        return _safe_div(self.current_buy_volume - self.current_sell_volume, self.config.bucket_volume)

    def feature_values(self) -> tuple[float, ...]:
        return (
            self.vpin(),
            1.0 if self.is_valid() else 0.0,
            float(self.completed_bucket_count),
            self.current_bucket_completion(),
            self.current_bucket_signed_imbalance(),
        )


class _KyleSample(NamedTuple):
    end_local_ts_us: int
    x_flow: float
    y_mid_bps: float


class _PendingKyleSample(NamedTuple):
    start_local_ts_us: int
    response_local_ts_us: int
    start_mid_tick: float
    signed_flow: float


@dataclass(slots=True)
class RollingKyleLambdaState:
    config: KyleLambdaConfig
    samples_by_window: tuple[deque[_KyleSample], ...]
    n: np.ndarray
    sum_x: np.ndarray
    sum_y: np.ndarray
    sum_xx: np.ndarray
    sum_xy: np.ndarray
    sum_yy: np.ndarray

    def add_finalized_sample(self, sample: _KyleSample, current_local_ts_us: int) -> None:
        for i in range(len(self.config.windows_us)):
            self.samples_by_window[i].append(sample)
            self.n[i] += 1.0
            self.sum_x[i] += sample.x_flow
            self.sum_y[i] += sample.y_mid_bps
            self.sum_xx[i] += sample.x_flow * sample.x_flow
            self.sum_xy[i] += sample.x_flow * sample.y_mid_bps
            self.sum_yy[i] += sample.y_mid_bps * sample.y_mid_bps
        self.expire_old(current_local_ts_us)

    def expire_old(self, current_local_ts_us: int) -> None:
        for i, window_us in enumerate(self.config.windows_us):
            cutoff = current_local_ts_us - window_us
            q = self.samples_by_window[i]
            while q and q[0].end_local_ts_us < cutoff:
                sample = q.popleft()
                self.n[i] -= 1.0
                self.sum_x[i] -= sample.x_flow
                self.sum_y[i] -= sample.y_mid_bps
                self.sum_xx[i] -= sample.x_flow * sample.x_flow
                self.sum_xy[i] -= sample.x_flow * sample.y_mid_bps
                self.sum_yy[i] -= sample.y_mid_bps * sample.y_mid_bps

    def lambda_for_window(self, window_index: int) -> float:
        n = self.n[window_index]
        var_x = self.sum_xx[window_index] - self.sum_x[window_index] * self.sum_x[window_index] / max(n, 1.0)
        if n < self.config.min_samples or var_x <= _EPS:
            return 0.0
        cov = self.sum_xy[window_index] - self.sum_x[window_index] * self.sum_y[window_index] / n
        return float(cov / var_x)

    def r2_for_window(self, window_index: int) -> float:
        n = self.n[window_index]
        var_x = self.sum_xx[window_index] - self.sum_x[window_index] * self.sum_x[window_index] / max(n, 1.0)
        var_y = self.sum_yy[window_index] - self.sum_y[window_index] * self.sum_y[window_index] / max(n, 1.0)
        if n < self.config.min_samples or var_x <= _EPS or var_y <= _EPS:
            return 0.0
        cov = self.sum_xy[window_index] - self.sum_x[window_index] * self.sum_y[window_index] / n
        return float((cov * cov) / (var_x * var_y))

    def feature_values(self) -> tuple[float, ...]:
        values: list[float] = []
        for i in range(len(self.config.windows_us)):
            valid = 1.0 if self.n[i] >= self.config.min_samples and abs(self.sum_xx[i] - self.sum_x[i] * self.sum_x[i] / max(self.n[i], 1.0)) > _EPS else 0.0
            values.extend((self.lambda_for_window(i), self.r2_for_window(i), float(self.n[i]), valid))
        return tuple(values)


class CounterfactualFillResult(NamedTuple):
    filled: bool
    fill_local_ts_us: int
    fill_price_tick: int
    fill_reason: str | None
    fill_latency_us: int
    queue_ahead_at_start: float
    queue_ahead_before_fill: float


@dataclass(frozen=True, slots=True)
class AdverseSelectionDataset:
    decision_local_ts_us: np.ndarray
    decision_event_index: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray
    label_names: tuple[str, ...]
    labels: np.ndarray
    label_masks: np.ndarray
    config: AdverseSelectionConfig

    def __post_init__(self) -> None:
        if self.decision_local_ts_us.ndim != 1 or self.decision_local_ts_us.dtype != np.int64:
            raise ValueError("decision_local_ts_us must be rank-1 int64")
        if self.decision_event_index.ndim != 1 or self.decision_event_index.dtype != np.int64:
            raise ValueError("decision_event_index must be rank-1 int64")
        if len(self.decision_local_ts_us) != len(self.decision_event_index):
            raise ValueError("decision arrays must have same length")
        if self.features.ndim != 2 or self.features.dtype != np.float32:
            raise ValueError("features must be rank-2 float32")
        if self.labels.ndim != 2 or self.labels.dtype != np.float32:
            raise ValueError("labels must be rank-2 float32")
        if self.label_masks.ndim != 2 or self.label_masks.dtype != np.bool_:
            raise ValueError("label_masks must be rank-2 bool")
        if self.features.shape[0] != len(self.decision_local_ts_us):
            raise ValueError("features row count must match decisions")
        if self.labels.shape[0] != len(self.decision_local_ts_us) or self.label_masks.shape != self.labels.shape:
            raise ValueError("labels and masks must match decisions and each other")
        if len(self.feature_names) != self.features.shape[1]:
            raise ValueError("feature_names length must match features")
        if len(self.label_names) != self.labels.shape[1]:
            raise ValueError("label_names length must match labels")
        if not np.isfinite(self.features).all():
            raise ValueError("all feature values must be finite")
        bad_labels = np.isnan(self.labels) & self.label_masks
        if bad_labels.any():
            raise ValueError("labels may contain NaN only where mask is false")
        if not np.isfinite(self.labels[self.label_masks]).all():
            raise ValueError("masked label values must be finite")
        if not isinstance(self.config, AdverseSelectionConfig):
            raise ValueError("config must be AdverseSelectionConfig")

    @property
    def num_decisions(self) -> int:
        return int(len(self.decision_local_ts_us))

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_labels(self) -> int:
        return int(self.labels.shape[1])

    def as_dict_summary(self) -> dict[str, object]:
        return summarize_adverse_selection_dataset(self)


def adverse_selection_feature_names(config: AdverseSelectionConfig) -> tuple[str, ...]:
    if not isinstance(config, AdverseSelectionConfig):
        raise ValueError("config must be AdverseSelectionConfig")
    names = [
        "spread_ticks",
        "spread_bps",
        "mid_price",
        "top_imbalance",
        "bid_top_size",
        "ask_top_size",
        "book_depth_bid_count",
        "book_depth_ask_count",
        "vpin",
        "vpin_valid",
        "vpin_completed_bucket_count",
        "vpin_current_bucket_completion",
        "vpin_current_bucket_signed_imbalance",
    ]
    for window_us in config.flow_windows_us:
        window_ms = window_us // 1000
        names.extend(
            (
                f"signed_qty_{window_ms}ms",
                f"signed_notional_{window_ms}ms",
                f"abs_qty_{window_ms}ms",
                f"buy_qty_{window_ms}ms",
                f"sell_qty_{window_ms}ms",
                f"trade_count_{window_ms}ms",
                f"flow_imbalance_ratio_{window_ms}ms",
            )
        )
    for window_us in config.kyle.windows_us:
        window_ms = window_us // 1000
        names.extend((f"kyle_lambda_{window_ms}ms", f"kyle_r2_{window_ms}ms", f"kyle_n_{window_ms}ms", f"kyle_valid_{window_ms}ms"))
    names.extend(
        (
            "bid_touch_depletion_ratio",
            "ask_touch_depletion_ratio",
            "bid_touch_replenishment_ratio",
            "ask_touch_replenishment_ratio",
            "microprice_bps",
        )
    )
    return tuple(names)


def adverse_selection_label_names() -> tuple[str, ...]:
    return (
        "bid_filled",
        "ask_filled",
        "bid_fill_latency_us",
        "ask_fill_latency_us",
        "bid_adverse_bps",
        "ask_adverse_bps",
        "bid_toxic_fill",
        "ask_toxic_fill",
        "bid_toxic_cost_bps",
        "ask_toxic_cost_bps",
    )


def _book_qty_at_price(ticks: np.ndarray, sizes: np.ndarray, price_tick: int) -> float | None:
    for tick, size in zip(ticks, sizes):
        tick_int = int(tick)
        if tick_int == 0:
            break
        if tick_int == price_tick:
            return float(size)
    return None


def _book_top_from_l2_row(tape: ExecutionTape, book_ptr: int) -> tuple[int, int, float, float, float, int]:
    l2 = tape.arrays.l2_events[book_ptr]
    best_bid_tick = int(l2["best_bid_tick"])
    best_ask_tick = int(l2["best_ask_tick"])
    best_bid_size = float(l2["best_bid_size"])
    best_ask_size = float(l2["best_ask_size"])
    if best_bid_tick <= 0 or best_ask_tick <= 0 or best_ask_tick <= best_bid_tick or best_bid_size < 0.0 or best_ask_size < 0.0:
        raise ValueError("L2 book must be two-sided and positive-spread")
    return best_bid_tick, best_ask_tick, best_bid_size, best_ask_size, (best_bid_tick + best_ask_tick) * 0.5, best_ask_tick - best_bid_tick


def _future_mid_tick_at_or_after(
    l2_local_ts_us: np.ndarray,
    l2_best_bid_ticks: np.ndarray,
    l2_best_ask_ticks: np.ndarray,
    target_local_ts_us: int,
) -> float | None:
    idx = int(np.searchsorted(l2_local_ts_us, target_local_ts_us, side="left"))
    if idx >= len(l2_local_ts_us):
        return None
    bid = int(l2_best_bid_ticks[idx])
    ask = int(l2_best_ask_ticks[idx])
    if bid <= 0 or ask <= bid:
        return None
    return (bid + ask) * 0.5


def _future_mid_and_ts_at_or_after(
    l2_local_ts_us: np.ndarray,
    l2_best_bid_ticks: np.ndarray,
    l2_best_ask_ticks: np.ndarray,
    target_local_ts_us: int,
) -> tuple[float, int] | None:
    idx = int(np.searchsorted(l2_local_ts_us, target_local_ts_us, side="left"))
    while idx < len(l2_local_ts_us):
        bid = int(l2_best_bid_ticks[idx])
        ask = int(l2_best_ask_ticks[idx])
        if bid > 0 and ask > bid:
            return (bid + ask) * 0.5, int(l2_local_ts_us[idx])
        idx += 1
    return None


def _clean_fill_result(
    *,
    filled: bool,
    fill_local_ts_us: int,
    fill_price_tick: int,
    fill_reason: FillReason | None,
    decision_local_ts_us: int,
    queue_ahead_at_start: float,
    queue_ahead_before_fill: float,
) -> CounterfactualFillResult:
    if not filled:
        return CounterfactualFillResult(False, -1, -1, None, -1, queue_ahead_at_start, queue_ahead_before_fill)
    return CounterfactualFillResult(
        True,
        int(fill_local_ts_us),
        int(fill_price_tick),
        None if fill_reason is None else fill_reason.value,
        int(fill_local_ts_us - decision_local_ts_us),
        float(queue_ahead_at_start),
        float(queue_ahead_before_fill),
    )


def _counterfactual_fill_one_side(
    tape: ExecutionTape,
    *,
    start_event_index: int,
    start_book_ptr: int,
    decision_local_ts_us: int,
    side: OrderSide,
    price_tick: int,
    qty: float,
    queue_ahead_qty: float,
    fill_deadline_us: int,
    config: CounterfactualQuoteConfig,
) -> CounterfactualFillResult:
    remaining_qty = float(qty)
    queue_ahead = max(float(queue_ahead_qty), 0.0)
    queue_at_start = queue_ahead
    prev_book_ptr = int(start_book_ptr)
    qm = config.queue_model
    events = tape.arrays.events
    trades = tape.arrays.trades
    for event_idx in range(start_event_index + 1, len(events)):
        event = events[event_idx]
        local_ts_us = int(event["local_ts_us"])
        if local_ts_us > fill_deadline_us:
            break
        event_type = int(event["event_type_code"])
        if event_type == EVENT_TYPE_CODE_TRADE:
            trade_ptr = int(event["trade_ptr"])
            if trade_ptr < 0:
                continue
            trade = trades[trade_ptr]
            side_code = int(trade["side_code"])
            if side_code == 0:
                continue
            trade_price_tick = int(trade["price_tick"])
            trade_amount = float(trade["amount"])
            if trade_amount <= 0.0:
                continue
            reason: FillReason | None = None
            if side == OrderSide.BUY and side_code == -1:
                if trade_price_tick < price_tick:
                    reason = FillReason.TRADE_THROUGH
                elif trade_price_tick == price_tick:
                    reason = FillReason.TRADE_AT_LEVEL
            elif side == OrderSide.SELL and side_code == 1:
                if trade_price_tick > price_tick:
                    reason = FillReason.TRADE_THROUGH
                elif trade_price_tick == price_tick:
                    reason = FillReason.TRADE_AT_LEVEL
            if reason == FillReason.TRADE_THROUGH:
                return _clean_fill_result(
                    filled=True,
                    fill_local_ts_us=local_ts_us,
                    fill_price_tick=price_tick,
                    fill_reason=FillReason.TRADE_THROUGH,
                    decision_local_ts_us=decision_local_ts_us,
                    queue_ahead_at_start=queue_at_start,
                    queue_ahead_before_fill=queue_ahead,
                )
            if reason == FillReason.TRADE_AT_LEVEL:
                effective_trade_qty = trade_amount * qm.trade_at_level_weight
                queue_before = queue_ahead
                queue_consumed = min(queue_ahead, effective_trade_qty)
                queue_ahead = max(queue_ahead - effective_trade_qty, 0.0)
                leftover = max(effective_trade_qty - queue_consumed, 0.0)
                fillable_qty = min(remaining_qty, leftover)
                if fillable_qty > qm.qty_epsilon:
                    return _clean_fill_result(
                        filled=True,
                        fill_local_ts_us=local_ts_us,
                        fill_price_tick=price_tick,
                        fill_reason=FillReason.TRADE_AT_LEVEL,
                        decision_local_ts_us=decision_local_ts_us,
                        queue_ahead_at_start=queue_at_start,
                        queue_ahead_before_fill=queue_before,
                    )
        elif event_type == EVENT_TYPE_CODE_L2_BATCH:
            curr_book_ptr = int(event["book_ptr"])
            if curr_book_ptr < 0:
                continue
            if qm.mode == QueueModelMode.BALANCED and prev_book_ptr >= 0:
                if side == OrderSide.BUY:
                    prev_qty = _book_qty_at_price(tape.arrays.book_bid_ticks[prev_book_ptr], tape.arrays.book_bid_sizes[prev_book_ptr], price_tick)
                    curr_qty = _book_qty_at_price(tape.arrays.book_bid_ticks[curr_book_ptr], tape.arrays.book_bid_sizes[curr_book_ptr], price_tick)
                else:
                    prev_qty = _book_qty_at_price(tape.arrays.book_ask_ticks[prev_book_ptr], tape.arrays.book_ask_sizes[prev_book_ptr], price_tick)
                    curr_qty = _book_qty_at_price(tape.arrays.book_ask_ticks[curr_book_ptr], tape.arrays.book_ask_sizes[curr_book_ptr], price_tick)
                if prev_qty is not None and curr_qty is not None:
                    visible_decrease = max(prev_qty - curr_qty, 0.0)
                    effective_l2_advance = visible_decrease * qm.l2_decrease_weight
                    queue_before = queue_ahead
                    queue_consumed = min(queue_ahead, effective_l2_advance)
                    queue_ahead = max(queue_ahead - effective_l2_advance, 0.0)
                    leftover = max(effective_l2_advance - queue_consumed, 0.0)
                    fillable_qty = min(remaining_qty, leftover)
                    if fillable_qty > qm.qty_epsilon:
                        return _clean_fill_result(
                            filled=True,
                            fill_local_ts_us=local_ts_us,
                            fill_price_tick=price_tick,
                            fill_reason=FillReason.QUEUE_DEPLETION,
                            decision_local_ts_us=decision_local_ts_us,
                            queue_ahead_at_start=queue_at_start,
                            queue_ahead_before_fill=queue_before,
                        )
            prev_book_ptr = curr_book_ptr
    return _clean_fill_result(
        filled=False,
        fill_local_ts_us=-1,
        fill_price_tick=-1,
        fill_reason=None,
        decision_local_ts_us=decision_local_ts_us,
        queue_ahead_at_start=queue_at_start,
        queue_ahead_before_fill=queue_ahead,
    )


class _TradeWindowState:
    def __init__(self, window_us: int) -> None:
        self.window_us = window_us
        self.items: deque[tuple[int, int, float, float]] = deque()
        self.signed_qty = 0.0
        self.signed_notional = 0.0
        self.abs_qty = 0.0
        self.buy_qty = 0.0
        self.sell_qty = 0.0
        self.trade_count = 0

    def update_trade(self, local_ts_us: int, side_code: int, qty: float, notional: float) -> None:
        sign = 1 if side_code > 0 else -1 if side_code < 0 else 0
        qty = max(float(qty), 0.0)
        notional = max(float(notional), 0.0)
        self.items.append((int(local_ts_us), sign, qty, notional))
        self.signed_qty += sign * qty
        self.signed_notional += sign * notional
        self.abs_qty += qty
        if sign > 0:
            self.buy_qty += qty
        elif sign < 0:
            self.sell_qty += qty
        self.trade_count += 1

    def expire(self, current_local_ts_us: int) -> None:
        cutoff = current_local_ts_us - self.window_us
        while self.items and self.items[0][0] < cutoff:
            _, sign, qty, notional = self.items.popleft()
            self.signed_qty -= sign * qty
            self.signed_notional -= sign * notional
            self.abs_qty -= qty
            if sign > 0:
                self.buy_qty -= qty
            elif sign < 0:
                self.sell_qty -= qty
            self.trade_count -= 1

    def feature_values(self) -> tuple[float, ...]:
        return (
            self.signed_qty,
            self.signed_notional,
            self.abs_qty,
            self.buy_qty,
            self.sell_qty,
            float(self.trade_count),
            _safe_div(self.signed_qty, self.abs_qty),
        )


def _new_kyle_state(config: KyleLambdaConfig) -> RollingKyleLambdaState:
    n_windows = len(config.windows_us)
    return RollingKyleLambdaState(
        config=config,
        samples_by_window=tuple(deque() for _ in config.windows_us),
        n=np.zeros(n_windows, dtype=np.float64),
        sum_x=np.zeros(n_windows, dtype=np.float64),
        sum_y=np.zeros(n_windows, dtype=np.float64),
        sum_xx=np.zeros(n_windows, dtype=np.float64),
        sum_xy=np.zeros(n_windows, dtype=np.float64),
        sum_yy=np.zeros(n_windows, dtype=np.float64),
    )


def _valid_l2_views(tape: ExecutionTape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    l2 = tape.arrays.l2_events
    local_ts = np.asarray(l2["local_ts_us"], dtype=np.int64)
    bid_ticks = np.asarray(l2["best_bid_tick"], dtype=np.int64)
    ask_ticks = np.asarray(l2["best_ask_tick"], dtype=np.int64)
    bid_sizes = np.asarray(l2["best_bid_size"], dtype=np.float64)
    ask_sizes = np.asarray(l2["best_ask_size"], dtype=np.float64)
    valid = (bid_ticks > 0) & (ask_ticks > bid_ticks)
    return local_ts[valid], bid_ticks[valid], ask_ticks[valid], bid_sizes[valid], ask_sizes[valid]


def _precompute_kyle_samples(tape: ExecutionTape, config: KyleLambdaConfig, tick_size: float) -> list[_KyleSample]:
    l2_ts, l2_bid, l2_ask, _, _ = _valid_l2_views(tape)
    if len(l2_ts) == 0:
        return []
    trades = tape.arrays.trades
    trade_ts = np.asarray(trades["local_ts_us"], dtype=np.int64)
    side = np.asarray(trades["side_code"], dtype=np.int8).astype(np.float64)
    amount = np.asarray(trades["amount"], dtype=np.float64)
    price_tick = np.asarray(trades["price_tick"], dtype=np.float64)
    flow = side * amount
    if config.use_notional_flow:
        flow = flow * price_tick * tick_size
    cflow = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(flow, dtype=np.float64)))
    samples: list[_KyleSample] = []
    start = int(l2_ts[0])
    end = int(l2_ts[-1] - config.response_horizon_us)
    while start <= end:
        start_info = _future_mid_and_ts_at_or_after(l2_ts, l2_bid, l2_ask, start)
        response_info = _future_mid_and_ts_at_or_after(l2_ts, l2_bid, l2_ask, start + config.response_horizon_us)
        if start_info is not None and response_info is not None:
            start_mid, start_obs_ts = start_info
            response_mid, response_obs_ts = response_info
            left = int(np.searchsorted(trade_ts, start, side="left"))
            right = int(np.searchsorted(trade_ts, start + config.sample_interval_us, side="right"))
            x_flow = float(cflow[right] - cflow[left])
            y_mid_bps = _bps_from_ticks(response_mid - start_mid, start_mid)
            end_ts = max(int(response_obs_ts), int(start_obs_ts), start + config.sample_interval_us)
            samples.append(_KyleSample(end_ts, x_flow, y_mid_bps))
        start += config.sample_interval_us
    samples.sort(key=lambda sample: sample.end_local_ts_us)
    return samples


def _book_fragility_values(tape: ExecutionTape, latest_book_ptr: int, previous_book_ptr: int, mid_tick: float) -> tuple[float, float, float, float, float]:
    if previous_book_ptr < 0 or latest_book_ptr < 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    _, _, bid_size, ask_size, _, _ = _book_top_from_l2_row(tape, latest_book_ptr)
    _, _, prev_bid_size, prev_ask_size, _, _ = _book_top_from_l2_row(tape, previous_book_ptr)
    bid_depletion = _safe_div(max(prev_bid_size - bid_size, 0.0), prev_bid_size)
    ask_depletion = _safe_div(max(prev_ask_size - ask_size, 0.0), prev_ask_size)
    bid_replenishment = _safe_div(max(bid_size - prev_bid_size, 0.0), prev_bid_size)
    ask_replenishment = _safe_div(max(ask_size - prev_ask_size, 0.0), prev_ask_size)
    best_bid, best_ask, _, _, _, _ = _book_top_from_l2_row(tape, latest_book_ptr)
    denom = bid_size + ask_size
    micro_tick = (best_ask * bid_size + best_bid * ask_size) / denom if denom > _EPS else mid_tick
    microprice_bps = _bps_from_ticks(micro_tick - mid_tick, mid_tick)
    return bid_depletion, ask_depletion, bid_replenishment, ask_replenishment, microprice_bps


def _feature_row(
    tape: ExecutionTape,
    *,
    latest_book_ptr: int,
    previous_book_ptr: int,
    spec: SymbolSpec,
    vpin_state: VPINState,
    flow_states: Sequence[_TradeWindowState],
    kyle_state: RollingKyleLambdaState,
) -> tuple[float, ...]:
    best_bid, best_ask, bid_size, ask_size, mid_tick, spread_ticks = _book_top_from_l2_row(tape, latest_book_ptr)
    mid_price = mid_tick * spec.tick_size
    top_imbalance = _safe_div(bid_size - ask_size, bid_size + ask_size)
    values: list[float] = [
        float(spread_ticks),
        _bps_from_ticks(float(spread_ticks), mid_tick),
        mid_price,
        top_imbalance,
        bid_size,
        ask_size,
        float(np.count_nonzero(tape.arrays.book_bid_ticks[latest_book_ptr] > 0)),
        float(np.count_nonzero(tape.arrays.book_ask_ticks[latest_book_ptr] > 0)),
    ]
    values.extend(vpin_state.feature_values())
    for flow_state in flow_states:
        values.extend(flow_state.feature_values())
    values.extend(kyle_state.feature_values())
    values.extend(_book_fragility_values(tape, latest_book_ptr, previous_book_ptr, mid_tick))
    return tuple(0.0 if not math.isfinite(value) else float(value) for value in values)


def _labels_for_decision(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig,
    decision_event_index: int,
    latest_book_ptr: int,
    decision_ts: int,
    l2_ts: np.ndarray,
    l2_bid: np.ndarray,
    l2_ask: np.ndarray,
) -> tuple[list[float], list[bool]] | None:
    best_bid, best_ask, _, _, _, _ = _book_top_from_l2_row(tape, latest_book_ptr)
    bid_price_tick = best_bid - config.quote.quote_distance_ticks
    ask_price_tick = best_ask + config.quote.quote_distance_ticks
    if bid_price_tick <= 0 or ask_price_tick <= 0:
        return None
    bid_visible_qty = _book_qty_at_price(tape.arrays.book_bid_ticks[latest_book_ptr], tape.arrays.book_bid_sizes[latest_book_ptr], bid_price_tick)
    ask_visible_qty = _book_qty_at_price(tape.arrays.book_ask_ticks[latest_book_ptr], tape.arrays.book_ask_sizes[latest_book_ptr], ask_price_tick)
    bid_queue = config.quote.queue_model.unknown_level_queue_ahead_qty if bid_visible_qty is None else bid_visible_qty
    ask_queue = config.quote.queue_model.unknown_level_queue_ahead_qty if ask_visible_qty is None else ask_visible_qty
    fill_deadline = decision_ts + config.quote.fill_horizon_us
    if config.drop_incomplete_horizon and fill_deadline > int(tape.arrays.events["local_ts_us"][-1]):
        return None
    bid_fill = _counterfactual_fill_one_side(
        tape,
        start_event_index=decision_event_index,
        start_book_ptr=latest_book_ptr,
        decision_local_ts_us=decision_ts,
        side=OrderSide.BUY,
        price_tick=bid_price_tick,
        qty=config.quote.order_qty,
        queue_ahead_qty=bid_queue,
        fill_deadline_us=fill_deadline,
        config=config.quote,
    )
    ask_fill = _counterfactual_fill_one_side(
        tape,
        start_event_index=decision_event_index,
        start_book_ptr=latest_book_ptr,
        decision_local_ts_us=decision_ts,
        side=OrderSide.SELL,
        price_tick=ask_price_tick,
        qty=config.quote.order_qty,
        queue_ahead_qty=ask_queue,
        fill_deadline_us=fill_deadline,
        config=config.quote,
    )
    labels = [math.nan] * 10
    masks = [False] * 10
    if fill_deadline <= int(tape.arrays.events["local_ts_us"][-1]):
        labels[0] = 1.0 if bid_fill.filled else 0.0
        labels[1] = 1.0 if ask_fill.filled else 0.0
        masks[0] = True
        masks[1] = True
    if bid_fill.filled:
        labels[2] = float(bid_fill.fill_latency_us)
        masks[2] = True
    if ask_fill.filled:
        labels[3] = float(ask_fill.fill_latency_us)
        masks[3] = True

    def set_adverse(fill: CounterfactualFillResult, label_idx: int, toxic_idx: int, cost_idx: int, is_bid: bool) -> bool:
        if not fill.filled:
            if masks[0 if is_bid else 1]:
                labels[cost_idx] = 0.0
                masks[cost_idx] = True
            return True
        future_mid = _future_mid_tick_at_or_after(l2_ts, l2_bid, l2_ask, fill.fill_local_ts_us + config.quote.adverse_horizon_us)
        if future_mid is None:
            return False
        if is_bid:
            adverse_bps = max(0.0, fill.fill_price_tick - future_mid) / fill.fill_price_tick * 10_000.0
        else:
            adverse_bps = max(0.0, future_mid - fill.fill_price_tick) / fill.fill_price_tick * 10_000.0
        labels[label_idx] = adverse_bps
        labels[toxic_idx] = 1.0 if adverse_bps > config.quote.toxic_threshold_bps else 0.0
        labels[cost_idx] = adverse_bps
        masks[label_idx] = True
        masks[toxic_idx] = True
        masks[cost_idx] = True
        return True

    bid_ok = set_adverse(bid_fill, 4, 6, 8, True)
    ask_ok = set_adverse(ask_fill, 5, 7, 9, False)
    if config.drop_incomplete_horizon and not (bid_ok and ask_ok):
        return None
    return labels, masks


def build_adverse_selection_dataset(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig = AdverseSelectionConfig(),
) -> AdverseSelectionDataset:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(config, AdverseSelectionConfig):
        raise ValueError("config must be AdverseSelectionConfig")
    spec = tape.manifest.symbol_spec
    if not isinstance(spec, SymbolSpec):
        raise ValueError("tape.manifest.symbol_spec must be SymbolSpec")
    if hasattr(spec, "is_valid_qty") and not spec.is_valid_qty(config.quote.order_qty):
        raise ValueError("quote.order_qty is invalid for tape symbol_spec")
    l2_ts, l2_bid, l2_ask, _, _ = _valid_l2_views(tape)
    if len(l2_ts) == 0:
        raise ValueError("tape must contain at least one valid two-sided L2 book")

    feature_names = adverse_selection_feature_names(config)
    label_names = adverse_selection_label_names()
    vpin_state = VPINState(config.vpin, deque())
    kyle_state = _new_kyle_state(config.kyle)
    kyle_samples = _precompute_kyle_samples(tape, config.kyle, spec.tick_size)
    next_kyle_sample_idx = 0
    flow_states = tuple(_TradeWindowState(window_us) for window_us in config.flow_windows_us)

    decision_ts_values: list[int] = []
    decision_event_indices: list[int] = []
    feature_rows: list[tuple[float, ...]] = []
    label_rows: list[list[float]] = []
    mask_rows: list[list[bool]] = []

    events = tape.arrays.events
    trades = tape.arrays.trades
    latest_book_ptr = -1
    previous_book_ptr = -1
    last_processed_event_idx = -1
    first_valid_ts = int(l2_ts[0])
    if config.start_event_index is not None and config.start_event_index < len(events):
        first_valid_ts = max(first_valid_ts, int(events[config.start_event_index]["local_ts_us"]))
    next_decision_ts = first_valid_ts

    def update_time_states(current_ts: int) -> None:
        nonlocal next_kyle_sample_idx
        for flow_state in flow_states:
            flow_state.expire(current_ts)
        while next_kyle_sample_idx < len(kyle_samples) and kyle_samples[next_kyle_sample_idx].end_local_ts_us <= current_ts:
            kyle_state.add_finalized_sample(kyle_samples[next_kyle_sample_idx], current_ts)
            next_kyle_sample_idx += 1
        kyle_state.expire_old(current_ts)

    def maybe_emit_decisions(up_to_ts: int) -> bool:
        nonlocal next_decision_ts
        while next_decision_ts <= up_to_ts and latest_book_ptr >= 0:
            if next_decision_ts >= first_valid_ts:
                update_time_states(next_decision_ts)
                labels_and_masks = _labels_for_decision(
                    tape,
                    config=config,
                    decision_event_index=last_processed_event_idx,
                    latest_book_ptr=latest_book_ptr,
                    decision_ts=next_decision_ts,
                    l2_ts=l2_ts,
                    l2_bid=l2_bid,
                    l2_ask=l2_ask,
                )
                if labels_and_masks is not None:
                    labels, masks = labels_and_masks
                    decision_ts_values.append(next_decision_ts)
                    decision_event_indices.append(last_processed_event_idx)
                    feature_rows.append(
                        _feature_row(
                            tape,
                            latest_book_ptr=latest_book_ptr,
                            previous_book_ptr=previous_book_ptr,
                            spec=spec,
                            vpin_state=vpin_state,
                            flow_states=flow_states,
                            kyle_state=kyle_state,
                        )
                    )
                    label_rows.append(labels)
                    mask_rows.append(masks)
                    if config.max_decisions is not None and len(decision_ts_values) >= config.max_decisions:
                        return True
            next_decision_ts += config.decision_interval_us
        return False

    i = 0
    while i < len(events):
        group_ts = int(events[i]["local_ts_us"])
        if maybe_emit_decisions(group_ts - 1):
            break
        while i < len(events) and int(events[i]["local_ts_us"]) == group_ts:
            event = events[i]
            event_type = int(event["event_type_code"])
            if event_type == EVENT_TYPE_CODE_L2_BATCH:
                book_ptr = int(event["book_ptr"])
                if book_ptr >= 0:
                    try:
                        _book_top_from_l2_row(tape, book_ptr)
                    except ValueError:
                        pass
                    else:
                        previous_book_ptr = latest_book_ptr
                        latest_book_ptr = book_ptr
            elif event_type == EVENT_TYPE_CODE_TRADE:
                trade_ptr = int(event["trade_ptr"])
                if trade_ptr >= 0:
                    trade = trades[trade_ptr]
                    side_code = int(trade["side_code"])
                    price_tick = int(trade["price_tick"])
                    amount = float(trade["amount"])
                    notional = amount * price_tick * spec.tick_size
                    for flow_state in flow_states:
                        flow_state.update_trade(group_ts, side_code, amount, notional)
                    vpin_state.update_trade(side_code=side_code, price_tick=price_tick, amount=amount, tick_size=spec.tick_size)
            last_processed_event_idx = i
            i += 1
        update_time_states(group_ts)
        if maybe_emit_decisions(group_ts):
            break

    features = np.asarray(feature_rows, dtype=np.float32).reshape((len(feature_rows), len(feature_names)))
    labels = np.asarray(label_rows, dtype=np.float32).reshape((len(label_rows), len(label_names)))
    masks = np.asarray(mask_rows, dtype=np.bool_).reshape((len(mask_rows), len(label_names)))
    return AdverseSelectionDataset(
        decision_local_ts_us=np.asarray(decision_ts_values, dtype=np.int64),
        decision_event_index=np.asarray(decision_event_indices, dtype=np.int64),
        feature_names=feature_names,
        features=features,
        label_names=label_names,
        labels=labels,
        label_masks=masks,
        config=config,
    )


def _masked_mean(values: np.ndarray, masks: np.ndarray) -> float:
    if values.size == 0 or not masks.any():
        return 0.0
    return float(np.mean(values[masks]))


def summarize_adverse_selection_dataset(
    dataset: AdverseSelectionDataset,
) -> dict[str, object]:
    if not isinstance(dataset, AdverseSelectionDataset):
        raise ValueError("dataset must be AdverseSelectionDataset")
    label_index = {name: i for i, name in enumerate(dataset.label_names)}
    feature_index = {name: i for i, name in enumerate(dataset.feature_names)}
    labels_summary = {
        "bid_fill_rate": _masked_mean(dataset.labels[:, label_index["bid_filled"]], dataset.label_masks[:, label_index["bid_filled"]]),
        "ask_fill_rate": _masked_mean(dataset.labels[:, label_index["ask_filled"]], dataset.label_masks[:, label_index["ask_filled"]]),
        "bid_toxic_fill_rate": _masked_mean(dataset.labels[:, label_index["bid_toxic_fill"]], dataset.label_masks[:, label_index["bid_toxic_fill"]]),
        "ask_toxic_fill_rate": _masked_mean(dataset.labels[:, label_index["ask_toxic_fill"]], dataset.label_masks[:, label_index["ask_toxic_fill"]]),
        "bid_adverse_bps_mean_conditional": _masked_mean(dataset.labels[:, label_index["bid_adverse_bps"]], dataset.label_masks[:, label_index["bid_adverse_bps"]]),
        "ask_adverse_bps_mean_conditional": _masked_mean(dataset.labels[:, label_index["ask_adverse_bps"]], dataset.label_masks[:, label_index["ask_adverse_bps"]]),
        "bid_toxic_cost_bps_mean_unconditional": _masked_mean(dataset.labels[:, label_index["bid_toxic_cost_bps"]], dataset.label_masks[:, label_index["bid_toxic_cost_bps"]]),
        "ask_toxic_cost_bps_mean_unconditional": _masked_mean(dataset.labels[:, label_index["ask_toxic_cost_bps"]], dataset.label_masks[:, label_index["ask_toxic_cost_bps"]]),
    }
    features_summary: dict[str, object] = {
        "finite_fraction": float(np.isfinite(dataset.features).mean()) if dataset.features.size else 1.0,
    }
    if "vpin" in feature_index and dataset.num_decisions:
        features_summary["vpin_mean"] = float(np.mean(dataset.features[:, feature_index["vpin"]]))
    for name, idx in feature_index.items():
        if name.startswith("kyle_lambda_") and dataset.num_decisions:
            features_summary[f"{name}_mean"] = float(np.mean(dataset.features[:, idx]))
    return {
        "num_decisions": dataset.num_decisions,
        "num_features": dataset.num_features,
        "num_labels": dataset.num_labels,
        "feature_names": list(dataset.feature_names),
        "label_names": list(dataset.label_names),
        "labels": labels_summary,
        "features": features_summary,
    }
