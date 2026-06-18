"""Causal fill-aware adverse-selection features and labels for execution tapes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Iterator, Mapping, NamedTuple, Protocol, Sequence

import numpy as np

try:
    from numba import njit as _numba_njit
except Exception:
    _numba_njit = None

from mmrt.execution.contracts import (
    AggressorSide,
    ActiveOrder,
    BookTop,
    Fill,
    FillReason,
    LatencyConfig,
    OrderSide,
    OrderStatus,
    QueueModelMode,
    QuoteIntent,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.decision_grid import DecisionGrid, validate_decision_grid_for_execution_tape, validate_decision_key_order
from mmrt.execution.execution_tape import (
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    ExecutionTape,
)
from mmrt.execution.fill_sim import (
    FillSimulatorConfig,
    activate_pending_orders,
    place_orders_from_quote,
    simulate_l2_level_update,
    simulate_trade_event,
)
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.time_key import EventKey, MAX_EVENT_SEQ


__all__ = [
    "VPINConfig",
    "KyleLambdaConfig",
    "QuoteCandidateMode",
    "QuoteCandidateConfig",
    "DEFAULT_QUOTE_CANDIDATES",
    "DEFAULT_QUOTE_CANDIDATE_NAMES",
    "CounterfactualQuoteConfig",
    "AdverseSelectionConfig",
    "CounterfactualFillResult",
    "AdverseSelectionDataset",
    "AdverseSelectionFeatureDataset",
    "VPINState",
    "RollingKyleLambdaState",
    "quote_candidate_configs_from_names",
    "adverse_selection_config_from_training_summary",
    "build_adverse_selection_dataset_to_disk",
    "summarize_adverse_selection_feature_dataset",
    "summarize_disk_adverse_selection_dataset",
    "summarize_adverse_selection_dataset",
    "adverse_selection_feature_names",
    "adverse_selection_label_names",
    "candidate_price_tick",
    "BuildAdverseSelectionDatasetToDiskConfig",
    "profile_adverse_selection_label_generation",
]

_EPS = 1e-12
_LABEL_ENGINE_AUTO = "auto"
_LABEL_ENGINE_NUMBA = "numba"
_LABEL_ENGINE_SCALAR = "scalar"
_LABEL_ENGINES = (_LABEL_ENGINE_AUTO, _LABEL_ENGINE_NUMBA, _LABEL_ENGINE_SCALAR)
_CONSERVATIVE_NUMBA_AVAILABLE = _numba_njit is not None
_AUTO_NUMBA_WARNING_EMITTED = False


def _maybe_njit(func):
    if _numba_njit is None:
        return func
    return _numba_njit(cache=True, fastmath=False)(func)


def _log_stage(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _coerce_label_engine(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("label_engine must be a string")
    value = value.strip().lower()
    if value not in _LABEL_ENGINES:
        raise ValueError("label_engine must be auto, numba, or scalar")
    return value


def _resolve_conservative_label_backend(label_engine: str, *, invalid_quote_policy: str) -> str:
    global _AUTO_NUMBA_WARNING_EMITTED
    label_engine = _coerce_label_engine(label_engine)
    if invalid_quote_policy == "error":
        return _LABEL_ENGINE_SCALAR
    if label_engine == _LABEL_ENGINE_SCALAR:
        return _LABEL_ENGINE_SCALAR
    if label_engine == _LABEL_ENGINE_NUMBA:
        if not _CONSERVATIVE_NUMBA_AVAILABLE:
            raise ValueError("label_engine='numba' requires numba to be installed")
        return _LABEL_ENGINE_NUMBA
    if _CONSERVATIVE_NUMBA_AVAILABLE:
        return _LABEL_ENGINE_NUMBA
    if not _AUTO_NUMBA_WARNING_EMITTED:
        _log_stage("adverse_dataset label_engine=auto numba_unavailable using scalar conservative labels")
        _AUTO_NUMBA_WARNING_EMITTED = True
    return _LABEL_ENGINE_SCALAR


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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


class QuoteCandidateMode(str, Enum):
    TOUCH = "touch"
    INSIDE = "inside"
    AWAY = "away"


def _coerce_quote_candidate_mode(value: QuoteCandidateMode | str) -> QuoteCandidateMode:
    if isinstance(value, QuoteCandidateMode):
        return value
    if isinstance(value, str):
        try:
            return QuoteCandidateMode(value)
        except ValueError as exc:
            raise ValueError(f"quote candidate mode has invalid value {value!r}") from exc
    raise ValueError("quote candidate mode must be QuoteCandidateMode or str")


@dataclass(frozen=True, slots=True)
class QuoteCandidateConfig:
    name: str
    mode: QuoteCandidateMode | str
    offset_ticks: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_nonempty_str(self.name, "name"))
        object.__setattr__(self, "mode", _coerce_quote_candidate_mode(self.mode))
        object.__setattr__(self, "offset_ticks", _require_nonnegative_int(self.offset_ticks, "offset_ticks"))
        if self.mode in (QuoteCandidateMode.INSIDE, QuoteCandidateMode.AWAY) and self.offset_ticks <= 0:
            raise ValueError("inside/away quote candidates require offset_ticks > 0")
        if self.mode == QuoteCandidateMode.TOUCH and self.offset_ticks != 0:
            raise ValueError("touch quote candidate requires offset_ticks == 0")


DEFAULT_QUOTE_CANDIDATES = (
    QuoteCandidateConfig(name="touch", mode=QuoteCandidateMode.TOUCH, offset_ticks=0),
    QuoteCandidateConfig(name="inside_1", mode=QuoteCandidateMode.INSIDE, offset_ticks=1),
    QuoteCandidateConfig(name="away_1", mode=QuoteCandidateMode.AWAY, offset_ticks=1),
)
DEFAULT_QUOTE_CANDIDATE_NAMES = tuple(candidate.name for candidate in DEFAULT_QUOTE_CANDIDATES)


def _quote_candidate_config_from_name(name: str) -> QuoteCandidateConfig:
    token = _require_nonempty_str(name, "quote_candidate_name")
    if token == "touch":
        return QuoteCandidateConfig("touch", QuoteCandidateMode.TOUCH, 0)
    for prefix, mode in (("inside_", QuoteCandidateMode.INSIDE), ("away_", QuoteCandidateMode.AWAY)):
        if token.startswith(prefix):
            suffix = token[len(prefix):]
            try:
                offset = int(suffix)
            except ValueError as exc:
                raise ValueError(f"malformed quote candidate {token!r}") from exc
            try:
                return QuoteCandidateConfig(token, mode, offset)
            except ValueError as exc:
                raise ValueError(f"malformed quote candidate {token!r}") from exc
    raise ValueError(f"malformed quote candidate {token!r}")


def quote_candidate_configs_from_names(names: Sequence[str]) -> tuple[QuoteCandidateConfig, ...]:
    if isinstance(names, (str, bytes)):
        raise ValueError("quote_candidates must be a sequence of non-empty strings")
    try:
        values = tuple(names)
    except TypeError as exc:
        raise ValueError("quote_candidates must be a sequence of non-empty strings") from exc
    if not values:
        raise ValueError("quote_candidates must be non-empty")
    return _quote_candidates_tuple(tuple(_quote_candidate_config_from_name(str(name)) for name in values))


def _required(payload: Mapping[str, object], key: str) -> object:
    if key not in payload:
        raise ValueError(f"missing required training summary field {key!r}")
    return payload[key]


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = _required(payload, key)
    if value is None:
        return None
    return int(value)


def adverse_selection_config_from_training_summary(payload: Mapping[str, object]) -> AdverseSelectionConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("training summary payload must be a mapping")
    try:
        quote_names_obj = _required(payload, "quote_candidates")
        if isinstance(quote_names_obj, (str, bytes)):
            quote_names = [part.strip() for part in str(quote_names_obj).split(",") if part.strip()]
        else:
            quote_names = [str(x) for x in quote_names_obj]  # type: ignore[union-attr]
        return AdverseSelectionConfig(
            flow_windows_us=tuple(int(x) for x in _required(payload, "flow_windows_us")),  # type: ignore[union-attr]
            drop_incomplete_horizon=bool(_required(payload, "drop_incomplete_horizon")),
            vpin=VPINConfig(
                bucket_volume=float(_required(payload, "vpin_bucket_volume")),
                num_buckets=int(_required(payload, "vpin_num_buckets")),
                min_completed_buckets=int(_required(payload, "vpin_min_completed_buckets")),
                use_notional_volume=bool(_required(payload, "vpin_use_notional_volume")),
            ),
            kyle=KyleLambdaConfig(
                sample_interval_us=int(_required(payload, "kyle_sample_interval_us")),
                response_horizon_us=int(_required(payload, "kyle_response_horizon_us")),
                windows_us=tuple(int(x) for x in _required(payload, "kyle_windows_us")),  # type: ignore[union-attr]
                min_samples=int(_required(payload, "kyle_min_samples")),
                use_notional_flow=bool(_required(payload, "kyle_use_notional_flow")),
            ),
            quote=CounterfactualQuoteConfig(
                quote_candidates=quote_candidate_configs_from_names(quote_names),
                post_only_gap_ticks=int(_required(payload, "post_only_gap_ticks")),
                invalid_quote_policy=str(_required(payload, "invalid_quote_policy")),
                order_qty=float(_required(payload, "order_qty")),
                fill_horizon_us=int(_required(payload, "fill_horizon_us")),
                adverse_horizon_us=int(_required(payload, "adverse_horizon_us")),
                toxic_threshold_bps=float(_required(payload, "toxic_threshold_bps")),
                latency_config=LatencyConfig(
                    decision_compute_latency_us=int(payload.get("decision_compute_latency_us", 50)),
                    order_entry_latency_us=int(payload.get("order_entry_latency_us", 500)),
                ),
                queue_model=QueueModelConfig(
                    mode=str(_required(payload, "queue_mode")),
                    l2_decrease_weight=float(_required(payload, "l2_decrease_weight")),
                    trade_at_level_weight=float(_required(payload, "trade_at_level_weight")),
                    unknown_level_queue_ahead_qty=float(_required(payload, "unknown_level_queue_ahead_qty")),
                    dedupe_l2_decrease_with_trade_prints=bool(_required(payload, "dedupe_l2_decrease_with_trade_prints")),
                ),
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("missing required", "malformed", "quote_", "decision_", "flow_", "vpin", "kyle", "queue", "invalid_")):
            raise
        raise ValueError("malformed adverse-selection training summary config") from exc


def _quote_candidates_tuple(values: Sequence[QuoteCandidateConfig]) -> tuple[QuoteCandidateConfig, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("quote_candidates must be a sequence of QuoteCandidateConfig")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("quote_candidates must be a sequence of QuoteCandidateConfig") from exc
    if not result:
        raise ValueError("quote_candidates must be non-empty")
    seen: set[str] = set()
    for idx, candidate in enumerate(result):
        if not isinstance(candidate, QuoteCandidateConfig):
            raise ValueError(f"quote_candidates[{idx}] must be QuoteCandidateConfig")
        if candidate.name in seen:
            raise ValueError(f"duplicate quote candidate name {candidate.name!r}")
        seen.add(candidate.name)
    return result


def candidate_price_tick(*, candidate: QuoteCandidateConfig, side: OrderSide, best_bid: int, best_ask: int, post_only_gap_ticks: int) -> int | None:
    if candidate.mode == QuoteCandidateMode.TOUCH:
        return best_bid if side == OrderSide.BUY else best_ask
    if candidate.mode == QuoteCandidateMode.INSIDE:
        if side == OrderSide.BUY:
            price = best_bid + candidate.offset_ticks
            return price if price <= best_ask - post_only_gap_ticks else None
        price = best_ask - candidate.offset_ticks
        return price if price >= best_bid + post_only_gap_ticks else None
    if candidate.mode == QuoteCandidateMode.AWAY:
        if side == OrderSide.BUY:
            price = best_bid - candidate.offset_ticks
            return price if price > 0 else None
        return best_ask + candidate.offset_ticks
    raise ValueError("unsupported quote candidate mode")


@dataclass(frozen=True, slots=True)
class CounterfactualQuoteConfig:
    quote_candidates: tuple[QuoteCandidateConfig, ...] = DEFAULT_QUOTE_CANDIDATES
    post_only_gap_ticks: int = 1
    invalid_quote_policy: str = "mask"
    order_qty: float = 0.001
    fill_horizon_us: int = 1_000_000
    adverse_horizon_us: int = 1_000_000
    toxic_threshold_bps: float = 0.0
    queue_model: QueueModelConfig = QueueModelConfig()
    latency_config: LatencyConfig = LatencyConfig()
    maker_fee_bps: float = -0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote_candidates", _quote_candidates_tuple(self.quote_candidates))
        object.__setattr__(self, "post_only_gap_ticks", _require_nonnegative_int(self.post_only_gap_ticks, "post_only_gap_ticks"))
        if self.invalid_quote_policy not in ("mask", "error"):
            raise ValueError("invalid_quote_policy must be 'mask' or 'error'")
        object.__setattr__(self, "order_qty", _require_positive_float(self.order_qty, "order_qty"))
        object.__setattr__(self, "fill_horizon_us", _require_positive_int(self.fill_horizon_us, "fill_horizon_us"))
        object.__setattr__(self, "adverse_horizon_us", _require_positive_int(self.adverse_horizon_us, "adverse_horizon_us"))
        object.__setattr__(self, "toxic_threshold_bps", _require_nonnegative_float(self.toxic_threshold_bps, "toxic_threshold_bps"))
        if not isinstance(self.queue_model, QueueModelConfig):
            raise ValueError("queue_model must be QueueModelConfig")
        if not isinstance(self.latency_config, LatencyConfig):
            raise ValueError("latency_config must be LatencyConfig")
        object.__setattr__(self, "maker_fee_bps", _require_finite_float(self.maker_fee_bps, "maker_fee_bps"))


@dataclass(frozen=True, slots=True)
class AdverseSelectionConfig:
    flow_windows_us: tuple[int, ...] = (200_000, 500_000, 1_000_000)
    vpin: VPINConfig = VPINConfig()
    kyle: KyleLambdaConfig = KyleLambdaConfig()
    quote: CounterfactualQuoteConfig = CounterfactualQuoteConfig()
    drop_incomplete_horizon: bool = True

    def __post_init__(self) -> None:
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
    end_key: EventKey
    x_flow: float
    y_mid_bps: float


class _DiskKyleSampleView(Sequence[_KyleSample]):
    def __init__(self, arrays) -> None:
        self._arrays = arrays
        self._count = int(arrays.count)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> _KyleSample:
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        arrays = self._arrays
        return _KyleSample(
            end_key=EventKey(int(arrays.end_local_ts_us[index]), int(arrays.end_event_seq[index])),
            x_flow=float(arrays.x_flow[index]),
            y_mid_bps=float(arrays.y_mid_bps[index]),
        )


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

    def add_finalized_sample(self, sample: _KyleSample, current_key: EventKey) -> None:
        for i in range(len(self.config.windows_us)):
            self.samples_by_window[i].append(sample)
            self.n[i] += 1.0
            self.sum_x[i] += sample.x_flow
            self.sum_y[i] += sample.y_mid_bps
            self.sum_xx[i] += sample.x_flow * sample.x_flow
            self.sum_xy[i] += sample.x_flow * sample.y_mid_bps
            self.sum_yy[i] += sample.y_mid_bps * sample.y_mid_bps
        self.expire_old(current_key)

    def expire_old(self, current_key: EventKey) -> None:
        for i, window_us in enumerate(self.config.windows_us):
            cutoff = current_key.local_ts_us - window_us
            q = self.samples_by_window[i]
            while q and q[0].end_key.local_ts_us < cutoff:
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
    decision_event_seq: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray
    label_names: tuple[str, ...]
    labels: np.ndarray
    label_masks: np.ndarray
    config: AdverseSelectionConfig
    decision_grid_schema: str = ""
    decision_grid_hash: str = ""
    decision_grid_n_rows: int = 0
    decision_schedule: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.decision_local_ts_us.ndim != 1 or self.decision_local_ts_us.dtype != np.int64:
            raise ValueError("decision_local_ts_us must be rank-1 int64")
        if self.decision_event_index.ndim != 1 or self.decision_event_index.dtype != np.int64:
            raise ValueError("decision_event_index must be rank-1 int64")
        if self.decision_event_seq.ndim != 1 or self.decision_event_seq.dtype != np.int64:
            raise ValueError("decision_event_seq must be rank-1 int64")
        if len(self.decision_event_seq) != len(self.decision_local_ts_us):
            raise ValueError("decision_event_seq length must match decisions")
        if len(self.decision_local_ts_us) != len(self.decision_event_index):
            raise ValueError("decision arrays must have same length")
        validate_decision_key_order(
            decision_event_index=self.decision_event_index,
            decision_local_ts_us=self.decision_local_ts_us,
            decision_event_seq=self.decision_event_seq,
        )
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
        if not isinstance(self.decision_grid_schema, str) or not self.decision_grid_schema:
            raise ValueError("decision_grid_schema must be non-empty")
        if len(self.decision_grid_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.decision_grid_hash):
            raise ValueError("decision_grid_hash must be 64 lowercase hex characters")
        if int(self.decision_grid_n_rows) < self.num_decisions:
            raise ValueError("decision_grid_n_rows must cover dataset rows")
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")

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


@dataclass(frozen=True, slots=True)
class AdverseSelectionFeatureDataset:
    decision_local_ts_us: np.ndarray
    decision_event_index: np.ndarray
    decision_event_seq: np.ndarray
    feature_names: tuple[str, ...]
    features: np.ndarray
    config: AdverseSelectionConfig
    decision_grid_schema: str = ""
    decision_grid_hash: str = ""
    decision_grid_n_rows: int = 0
    decision_schedule: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.decision_local_ts_us.ndim != 1 or self.decision_local_ts_us.dtype != np.int64:
            raise ValueError("decision_local_ts_us must be rank-1 int64")
        if self.decision_event_index.ndim != 1 or self.decision_event_index.dtype != np.int64:
            raise ValueError("decision_event_index must be rank-1 int64")
        if self.decision_event_seq.ndim != 1 or self.decision_event_seq.dtype != np.int64:
            raise ValueError("decision_event_seq must be rank-1 int64")
        if len(self.decision_event_seq) != len(self.decision_local_ts_us):
            raise ValueError("decision_event_seq length must match decisions")
        if len(self.decision_local_ts_us) != len(self.decision_event_index):
            raise ValueError("decision arrays must have same length")
        validate_decision_key_order(
            decision_event_index=self.decision_event_index,
            decision_local_ts_us=self.decision_local_ts_us,
            decision_event_seq=self.decision_event_seq,
        )
        if self.features.ndim != 2 or self.features.dtype != np.float32:
            raise ValueError("features must be rank-2 float32")
        if self.features.shape[0] != len(self.decision_local_ts_us):
            raise ValueError("features row count must match decisions")
        if len(self.feature_names) != self.features.shape[1]:
            raise ValueError("feature_names length must match features")
        if not np.isfinite(self.features).all():
            raise ValueError("all feature values must be finite")
        if not isinstance(self.config, AdverseSelectionConfig):
            raise ValueError("config must be AdverseSelectionConfig")
        if not isinstance(self.decision_grid_schema, str) or not self.decision_grid_schema:
            raise ValueError("decision_grid_schema must be non-empty")
        if len(self.decision_grid_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.decision_grid_hash):
            raise ValueError("decision_grid_hash must be 64 lowercase hex characters")
        if int(self.decision_grid_n_rows) != self.num_decisions:
            raise ValueError("decision_grid_n_rows must match feature rows")
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")

    @property
    def num_decisions(self) -> int:
        return int(len(self.decision_local_ts_us))

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    def as_dict_summary(self) -> dict[str, object]:
        return summarize_adverse_selection_feature_dataset(self)


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


def adverse_selection_label_names(config: AdverseSelectionConfig) -> tuple[str, ...]:
    if not isinstance(config, AdverseSelectionConfig):
        raise ValueError("config must be AdverseSelectionConfig")
    names: list[str] = []
    for candidate in config.quote.quote_candidates:
        c = candidate.name
        names.extend((
            f"bid_{c}_filled",
            f"ask_{c}_filled",
            f"bid_{c}_fill_latency_us",
            f"ask_{c}_fill_latency_us",
            f"bid_{c}_adverse_bps",
            f"ask_{c}_adverse_bps",
            f"bid_{c}_toxic_fill",
            f"ask_{c}_toxic_fill",
            f"bid_{c}_toxic_cost_bps",
            f"ask_{c}_toxic_cost_bps",
        ))
    return tuple(names)


@dataclass(frozen=True, slots=True)
class _AdverseLabelLayout:
    label_names: tuple[str, ...]
    label_count: int
    candidate_bases: tuple[int, ...]

    @classmethod
    def from_config(cls, config: AdverseSelectionConfig) -> "_AdverseLabelLayout":
        names = adverse_selection_label_names(config)
        return cls(
            label_names=names,
            label_count=len(names),
            candidate_bases=tuple(i * 10 for i in range(len(config.quote.quote_candidates))),
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


def _book_top_obj_from_l2_row(tape: ExecutionTape, book_ptr: int) -> BookTop:
    best_bid_tick, best_ask_tick, best_bid_size, best_ask_size, _, _ = _book_top_from_l2_row(tape, book_ptr)
    return BookTop(
        local_ts_us=int(tape.arrays.l2_events[book_ptr]["local_ts_us"]),
        best_bid_tick=best_bid_tick,
        best_ask_tick=best_ask_tick,
        best_bid_size=best_bid_size,
        best_ask_size=best_ask_size,
    )


def _level_qty_with_depth_status(tape: ExecutionTape, *, book_ptr: int, side: OrderSide, price_tick: int) -> tuple[float | None, bool]:
    if side == OrderSide.BUY:
        ticks = tape.arrays.book_bid_ticks[book_ptr]; sizes = tape.arrays.book_bid_sizes[book_ptr]
    else:
        ticks = tape.arrays.book_ask_ticks[book_ptr]; sizes = tape.arrays.book_ask_sizes[book_ptr]
    active: list[int] = []
    for tick, size in zip(ticks, sizes):
        tick_int = int(tick)
        if tick_int == 0:
            break
        active.append(tick_int)
        if tick_int == price_tick:
            return float(size), True
    if active and min(active) <= price_tick <= max(active):
        return 0.0, True
    return None, False


def _queue_ahead_for_candidate(tape: ExecutionTape, *, book_ptr: int, side: OrderSide, price_tick: int, queue_model: QueueModelConfig) -> float:
    top = _book_top_obj_from_l2_row(tape, book_ptr)
    if top.best_bid_tick < price_tick < top.best_ask_tick:
        return 0.0
    qty, known = _level_qty_with_depth_status(tape, book_ptr=book_ptr, side=side, price_tick=price_tick)
    if known:
        return 0.0 if qty is None else qty
    return queue_model.unknown_level_queue_ahead_qty


def _activation_key_for_latency(decision_key: EventKey, delay_us: int) -> EventKey:
    target = decision_key.local_ts_us + delay_us
    if target <= decision_key.local_ts_us:
        return decision_key
    return EventKey(target, MAX_EVENT_SEQ)


def _deadline_key(decision_key: EventKey, horizon_us: int) -> EventKey:
    return EventKey(decision_key.local_ts_us + horizon_us, MAX_EVENT_SEQ)


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
        True, int(fill_local_ts_us), int(fill_price_tick), None if fill_reason is None else fill_reason.value,
        int(fill_local_ts_us - decision_local_ts_us), float(queue_ahead_at_start), float(queue_ahead_before_fill),
    )


def _live_orders(orders: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    return tuple(order for order in orders if order.is_live)


def _trade_side_from_code(side_code: int) -> AggressorSide | None:
    if side_code > 0:
        return AggressorSide.BUY
    if side_code < 0:
        return AggressorSide.SELL
    return None


def _has_fillable_order_at_level(orders: Sequence[ActiveOrder], *, side: OrderSide, price_tick: int, event_key: EventKey) -> bool:
    return any(order.side == side and order.price_tick == price_tick and order.is_fillable_at_key(event_key) for order in orders)


@dataclass(frozen=True, slots=True)
class _ConservativeFillIndex:
    """Per-tape arrays for closed-form conservative counterfactual fills.

    With the conservative queue model, visible L2 decreases never advance the
    queue, so a passive order's fate is fully determined by the book at its
    activation moment plus the opposite-side trade prints inside the fill
    window. That reduces each counterfactual replay to binary searches and a
    primitive scan of the window's trades.
    """

    event_local_ts: np.ndarray
    last_l2_event_pos: np.ndarray
    event_book_ptr: np.ndarray
    l2_best_bid_tick: np.ndarray
    l2_best_ask_tick: np.ndarray
    trade_event_pos: np.ndarray
    trade_price_tick: np.ndarray
    trade_amount: np.ndarray
    trade_is_buy: np.ndarray
    buy_trade_event_pos: np.ndarray
    buy_trade_price_tick: np.ndarray
    buy_trade_amount: np.ndarray
    sell_trade_event_pos: np.ndarray
    sell_trade_price_tick: np.ndarray
    sell_trade_amount: np.ndarray


def _build_conservative_fill_index(tape: ExecutionTape) -> _ConservativeFillIndex:
    events = tape.arrays.events
    event_local_ts = np.ascontiguousarray(events["local_ts_us"], dtype=np.int64)
    type_codes = np.asarray(events["event_type_code"])
    book_ptrs = np.ascontiguousarray(events["book_ptr"], dtype=np.int64)
    trade_ptrs = np.asarray(events["trade_ptr"], dtype=np.int64)
    n = len(events)
    positions = np.arange(n, dtype=np.int64)
    l2_mask = (type_codes == EVENT_TYPE_CODE_L2_BATCH) & (book_ptrs >= 0)
    last_l2_event_pos = np.maximum.accumulate(np.where(l2_mask, positions, -1))

    trade_mask = (type_codes == EVENT_TYPE_CODE_TRADE) & (trade_ptrs >= 0)
    trade_event_pos = positions[trade_mask]
    ptrs = trade_ptrs[trade_mask]
    trades = tape.arrays.trades
    side_codes = np.asarray(trades["side_code"], dtype=np.int64)[ptrs]
    amounts = np.asarray(trades["amount"], dtype=np.float64)[ptrs]
    effective = (side_codes != 0) & (amounts > 0.0)
    event_pos = np.ascontiguousarray(trade_event_pos[effective])
    price_tick = np.ascontiguousarray(np.asarray(trades["price_tick"], dtype=np.int64)[ptrs][effective])
    amount = np.ascontiguousarray(amounts[effective])
    is_buy = np.ascontiguousarray(side_codes[effective] > 0)
    buy = is_buy
    sell = ~is_buy
    return _ConservativeFillIndex(
        event_local_ts=event_local_ts,
        last_l2_event_pos=last_l2_event_pos,
        event_book_ptr=book_ptrs,
        l2_best_bid_tick=np.ascontiguousarray(tape.arrays.l2_events["best_bid_tick"], dtype=np.int64),
        l2_best_ask_tick=np.ascontiguousarray(tape.arrays.l2_events["best_ask_tick"], dtype=np.int64),
        trade_event_pos=event_pos,
        trade_price_tick=price_tick,
        trade_amount=amount,
        trade_is_buy=is_buy,
        buy_trade_event_pos=np.ascontiguousarray(event_pos[buy]),
        buy_trade_price_tick=np.ascontiguousarray(price_tick[buy]),
        buy_trade_amount=np.ascontiguousarray(amount[buy]),
        sell_trade_event_pos=np.ascontiguousarray(event_pos[sell]),
        sell_trade_price_tick=np.ascontiguousarray(price_tick[sell]),
        sell_trade_amount=np.ascontiguousarray(amount[sell]),
    )


def _conservative_queue_ahead_from_row(
    tape: ExecutionTape,
    *,
    book_ptr: int,
    side: OrderSide,
    price_tick: int,
    best_bid: int,
    best_ask: int,
    queue_model: QueueModelConfig,
) -> float:
    if best_bid < price_tick < best_ask:
        return 0.0
    if side == OrderSide.BUY:
        ticks = tape.arrays.book_bid_ticks[book_ptr]
        sizes = tape.arrays.book_bid_sizes[book_ptr]
    else:
        ticks = tape.arrays.book_ask_ticks[book_ptr]
        sizes = tape.arrays.book_ask_sizes[book_ptr]
    match = np.nonzero(ticks == price_tick)[0]
    if match.size:
        return float(sizes[int(match[0])])
    active = ticks[ticks > 0]
    if active.size and int(active.min()) <= price_tick <= int(active.max()):
        return 0.0
    return queue_model.unknown_level_queue_ahead_qty


def _conservative_fill_one_side(
    index: _ConservativeFillIndex,
    tape: ExecutionTape,
    *,
    start_event_index: int,
    start_book_ptr: int,
    decision_key: EventKey,
    side: OrderSide,
    price_tick: int,
    end_event_index: int,
    config: CounterfactualQuoteConfig,
) -> CounterfactualFillResult:
    decision_ts = decision_key.local_ts_us
    not_filled_queue = _queue_ahead_for_candidate(
        tape, book_ptr=start_book_ptr, side=side, price_tick=price_tick, queue_model=config.queue_model
    )
    end_event_index = min(end_event_index, len(index.event_local_ts))

    delay_us = config.latency_config.order_activation_delay_us
    if delay_us == 0:
        activation_book_ptr = start_book_ptr
        first_fillable_pos = int(np.searchsorted(index.event_local_ts, decision_ts, side="right"))
    else:
        first_fillable_pos = int(np.searchsorted(index.event_local_ts, decision_ts + delay_us, side="right"))
        if first_fillable_pos >= end_event_index:
            return _clean_fill_result(
                filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None,
                decision_local_ts_us=decision_ts, queue_ahead_at_start=not_filled_queue,
                queue_ahead_before_fill=not_filled_queue,
            )
        last_pos = int(index.last_l2_event_pos[first_fillable_pos - 1]) if first_fillable_pos > 0 else -1
        if last_pos > start_event_index:
            activation_book_ptr = int(index.event_book_ptr[last_pos])
        else:
            activation_book_ptr = start_book_ptr
    if first_fillable_pos >= end_event_index:
        return _clean_fill_result(
            filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None,
            decision_local_ts_us=decision_ts, queue_ahead_at_start=not_filled_queue,
            queue_ahead_before_fill=not_filled_queue,
        )

    best_bid, best_ask, _, _, _, _ = _book_top_from_l2_row(tape, activation_book_ptr)
    queue_ahead = _conservative_queue_ahead_from_row(
        tape, book_ptr=activation_book_ptr, side=side, price_tick=price_tick,
        best_bid=best_bid, best_ask=best_ask, queue_model=config.queue_model,
    )
    if side == OrderSide.BUY:
        post_only_safe = price_tick <= best_ask - config.post_only_gap_ticks
    else:
        post_only_safe = price_tick >= best_bid + config.post_only_gap_ticks
    if not post_only_safe:
        return _clean_fill_result(
            filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None,
            decision_local_ts_us=decision_ts, queue_ahead_at_start=queue_ahead,
            queue_ahead_before_fill=queue_ahead,
        )

    t_lo = int(np.searchsorted(index.trade_event_pos, first_fillable_pos, side="left"))
    t_hi = int(np.searchsorted(index.trade_event_pos, end_event_index, side="left"))
    if t_lo >= t_hi:
        return _clean_fill_result(
            filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None,
            decision_local_ts_us=decision_ts, queue_ahead_at_start=queue_ahead,
            queue_ahead_before_fill=queue_ahead,
        )

    prices = index.trade_price_tick[t_lo:t_hi]
    is_buy = index.trade_is_buy[t_lo:t_hi]
    if side == OrderSide.BUY:
        opp = ~is_buy
        through = opp & (prices < price_tick)
    else:
        opp = is_buy
        through = opp & (prices > price_tick)
    at_level = opp & (prices == price_tick)

    through_pos = int(np.argmax(through)) if through.any() else -1
    fill_pos = -1
    fill_reason: FillReason | None = None
    queue_before_fill = queue_ahead
    weight = config.queue_model.trade_at_level_weight
    if at_level.any() and weight > 0.0:
        at_positions = np.nonzero(at_level)[0]
        cum = np.cumsum(index.trade_amount[t_lo:t_hi][at_positions] * weight)
        k = int(np.searchsorted(cum, queue_ahead + config.queue_model.qty_epsilon, side="right"))
        if k < len(at_positions):
            fill_pos = int(at_positions[k])
            fill_reason = FillReason.TRADE_AT_LEVEL
            queue_before_fill = max(queue_ahead - (float(cum[k - 1]) if k > 0 else 0.0), 0.0)
    if through_pos >= 0 and (fill_pos < 0 or through_pos < fill_pos):
        fill_pos = through_pos
        fill_reason = FillReason.TRADE_THROUGH
        at_before = at_level[:through_pos]
        if at_before.any():
            queue_before_fill = max(
                queue_ahead - float(np.sum(index.trade_amount[t_lo:t_hi][:through_pos][at_before]) * weight),
                0.0,
            )
        else:
            queue_before_fill = queue_ahead
    if fill_pos < 0:
        return _clean_fill_result(
            filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None,
            decision_local_ts_us=decision_ts, queue_ahead_at_start=queue_ahead,
            queue_ahead_before_fill=queue_ahead,
        )

    fill_event_pos = int(index.trade_event_pos[t_lo + fill_pos])
    fill_ts = int(index.event_local_ts[fill_event_pos])
    return _clean_fill_result(
        filled=True,
        fill_local_ts_us=fill_ts,
        fill_price_tick=price_tick,
        fill_reason=fill_reason,
        decision_local_ts_us=decision_ts,
        queue_ahead_at_start=queue_ahead,
        queue_ahead_before_fill=queue_before_fill,
    )


@_maybe_njit
def _lower_bound_int64(arr: np.ndarray, value: int) -> int:
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@_maybe_njit
def _upper_bound_int64(arr: np.ndarray, value: int) -> int:
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@_maybe_njit
def _future_mid_at_or_after_impl(
    local_ts_us: np.ndarray,
    event_seq: np.ndarray,
    mid_tick: np.ndarray,
    key_ts: int,
    key_seq: int,
) -> float:
    count = local_ts_us.shape[0]
    idx = _lower_bound_int64(local_ts_us, key_ts)
    if idx >= count:
        return math.nan
    if local_ts_us[idx] == key_ts:
        j = idx
        best = -1
        while j < count and local_ts_us[j] == key_ts:
            if event_seq[j] <= key_seq:
                best = j
            j += 1
        if best >= 0:
            return float(mid_tick[best])
        idx = j
    if idx >= count:
        return math.nan
    return float(mid_tick[idx])


@_maybe_njit
def _queue_ahead_from_book_arrays_impl(
    bid_ticks: np.ndarray,
    bid_sizes: np.ndarray,
    ask_ticks: np.ndarray,
    ask_sizes: np.ndarray,
    book_ptr: int,
    side_code: int,
    price_tick: int,
    best_bid: int,
    best_ask: int,
    unknown_queue: float,
) -> float:
    if best_bid < price_tick < best_ask:
        return 0.0
    ticks = bid_ticks if side_code > 0 else ask_ticks
    sizes = bid_sizes if side_code > 0 else ask_sizes
    has_active = False
    min_tick = 0
    max_tick = 0
    for level in range(ticks.shape[1]):
        tick = int(ticks[book_ptr, level])
        if tick == price_tick:
            return float(sizes[book_ptr, level])
        if tick > 0:
            if not has_active:
                min_tick = tick
                max_tick = tick
                has_active = True
            else:
                if tick < min_tick:
                    min_tick = tick
                if tick > max_tick:
                    max_tick = tick
    if has_active and min_tick <= price_tick <= max_tick:
        return 0.0
    return unknown_queue


@_maybe_njit
def _conservative_fill_one_side_impl(
    event_local_ts: np.ndarray,
    last_l2_event_pos: np.ndarray,
    event_book_ptr: np.ndarray,
    bid_ticks: np.ndarray,
    bid_sizes: np.ndarray,
    ask_ticks: np.ndarray,
    ask_sizes: np.ndarray,
    l2_best_bid_tick: np.ndarray,
    l2_best_ask_tick: np.ndarray,
    side_trade_event_pos: np.ndarray,
    side_trade_price_tick: np.ndarray,
    side_trade_amount: np.ndarray,
    start_event_index: int,
    start_book_ptr: int,
    decision_ts: int,
    side_code: int,
    price_tick: int,
    end_event_index: int,
    delay_us: int,
    post_only_gap_ticks: int,
    trade_at_level_weight: float,
    unknown_queue: float,
    qty_epsilon: float,
) -> tuple[int, int, int]:
    event_count = event_local_ts.shape[0]
    if end_event_index > event_count:
        end_event_index = event_count
    if delay_us == 0:
        activation_book_ptr = start_book_ptr
        first_fillable_pos = _upper_bound_int64(event_local_ts, decision_ts)
    else:
        first_fillable_pos = _upper_bound_int64(event_local_ts, decision_ts + delay_us)
        if first_fillable_pos >= end_event_index:
            return 0, -1, -1
        last_pos = -1
        if first_fillable_pos > 0:
            last_pos = int(last_l2_event_pos[first_fillable_pos - 1])
        if last_pos > start_event_index:
            activation_book_ptr = int(event_book_ptr[last_pos])
        else:
            activation_book_ptr = start_book_ptr
    if first_fillable_pos >= end_event_index:
        return 0, -1, -1

    best_bid = int(l2_best_bid_tick[activation_book_ptr])
    best_ask = int(l2_best_ask_tick[activation_book_ptr])
    queue_ahead = _queue_ahead_from_book_arrays_impl(
        bid_ticks,
        bid_sizes,
        ask_ticks,
        ask_sizes,
        activation_book_ptr,
        side_code,
        price_tick,
        best_bid,
        best_ask,
        unknown_queue,
    )
    if side_code > 0:
        if price_tick > best_ask - post_only_gap_ticks:
            return 0, -1, -1
    else:
        if price_tick < best_bid + post_only_gap_ticks:
            return 0, -1, -1

    t_lo = _lower_bound_int64(side_trade_event_pos, first_fillable_pos)
    t_hi = _lower_bound_int64(side_trade_event_pos, end_event_index)
    if t_lo >= t_hi:
        return 0, -1, -1

    consumed = 0.0
    threshold = queue_ahead + qty_epsilon
    for trade_idx in range(t_lo, t_hi):
        trade_price = int(side_trade_price_tick[trade_idx])
        if side_code > 0:
            if trade_price < price_tick:
                fill_event_pos = int(side_trade_event_pos[trade_idx])
                fill_ts = int(event_local_ts[fill_event_pos])
                return 1, fill_ts, int(fill_ts - decision_ts)
            at_level = trade_price == price_tick
        else:
            if trade_price > price_tick:
                fill_event_pos = int(side_trade_event_pos[trade_idx])
                fill_ts = int(event_local_ts[fill_event_pos])
                return 1, fill_ts, int(fill_ts - decision_ts)
            at_level = trade_price == price_tick
        if at_level and trade_at_level_weight > 0.0:
            consumed += float(side_trade_amount[trade_idx]) * trade_at_level_weight
            if consumed > threshold:
                fill_event_pos = int(side_trade_event_pos[trade_idx])
                fill_ts = int(event_local_ts[fill_event_pos])
                return 1, fill_ts, int(fill_ts - decision_ts)
    return 0, -1, -1


@_maybe_njit
def _conservative_labels_kernel_impl(
    labels: np.ndarray,
    masks: np.ndarray,
    keep_rows: np.ndarray,
    decision_local_ts_us: np.ndarray,
    decision_event_index: np.ndarray,
    latest_book_ptr: np.ndarray,
    event_local_ts: np.ndarray,
    last_l2_event_pos: np.ndarray,
    event_book_ptr: np.ndarray,
    bid_ticks: np.ndarray,
    bid_sizes: np.ndarray,
    ask_ticks: np.ndarray,
    ask_sizes: np.ndarray,
    l2_best_bid_tick: np.ndarray,
    l2_best_ask_tick: np.ndarray,
    buy_trade_event_pos: np.ndarray,
    buy_trade_price_tick: np.ndarray,
    buy_trade_amount: np.ndarray,
    sell_trade_event_pos: np.ndarray,
    sell_trade_price_tick: np.ndarray,
    sell_trade_amount: np.ndarray,
    valid_l2_local_ts: np.ndarray,
    valid_l2_event_seq: np.ndarray,
    valid_l2_mid_tick: np.ndarray,
    candidate_modes: np.ndarray,
    candidate_offsets: np.ndarray,
    fill_horizon_us: int,
    adverse_horizon_us: int,
    last_event_local_ts_us: int,
    drop_incomplete_horizon: int,
    post_only_gap_ticks: int,
    delay_us: int,
    trade_at_level_weight: float,
    unknown_queue: float,
    qty_epsilon: float,
    toxic_threshold_bps: float,
    order_qty: float,
    tick_size: float,
    contract_size: float,
    min_notional: float,
) -> None:
    row_count = decision_local_ts_us.shape[0]
    candidate_count = candidate_modes.shape[0]
    max_event_seq = int(MAX_EVENT_SEQ)
    for row in range(row_count):
        decision_ts = int(decision_local_ts_us[row])
        fill_deadline_ts = decision_ts + fill_horizon_us
        if drop_incomplete_horizon != 0 and fill_deadline_ts > last_event_local_ts_us:
            continue
        end_event_index = _upper_bound_int64(event_local_ts, fill_deadline_ts)
        book_ptr = int(latest_book_ptr[row])
        best_bid = int(l2_best_bid_tick[book_ptr])
        best_ask = int(l2_best_ask_tick[book_ptr])
        dropped = False
        for candidate_index in range(candidate_count):
            mode = int(candidate_modes[candidate_index])
            offset = int(candidate_offsets[candidate_index])
            base = candidate_index * 10
            for side_slot in range(2):
                side_code = 1 if side_slot == 0 else -1
                if mode == 0:
                    price_tick = best_bid if side_code > 0 else best_ask
                elif mode == 1:
                    if side_code > 0:
                        price_tick = best_bid + offset
                        if price_tick > best_ask - post_only_gap_ticks:
                            continue
                    else:
                        price_tick = best_ask - offset
                        if price_tick < best_bid + post_only_gap_ticks:
                            continue
                else:
                    if side_code > 0:
                        price_tick = best_bid - offset
                        if price_tick <= 0:
                            continue
                    else:
                        price_tick = best_ask + offset
                if price_tick <= 0:
                    continue
                if min_notional > 0.0 and order_qty * float(price_tick) * tick_size * contract_size < min_notional:
                    continue

                filled_idx = base + side_slot
                latency_idx = base + 2 + side_slot
                adverse_idx = base + 4 + side_slot
                toxic_idx = base + 6 + side_slot
                cost_idx = base + 8 + side_slot
                if side_code > 0:
                    filled, fill_ts, latency = _conservative_fill_one_side_impl(
                        event_local_ts,
                        last_l2_event_pos,
                        event_book_ptr,
                        bid_ticks,
                        bid_sizes,
                        ask_ticks,
                        ask_sizes,
                        l2_best_bid_tick,
                        l2_best_ask_tick,
                        sell_trade_event_pos,
                        sell_trade_price_tick,
                        sell_trade_amount,
                        int(decision_event_index[row]),
                        book_ptr,
                        decision_ts,
                        side_code,
                        price_tick,
                        end_event_index,
                        delay_us,
                        post_only_gap_ticks,
                        trade_at_level_weight,
                        unknown_queue,
                        qty_epsilon,
                    )
                else:
                    filled, fill_ts, latency = _conservative_fill_one_side_impl(
                        event_local_ts,
                        last_l2_event_pos,
                        event_book_ptr,
                        bid_ticks,
                        bid_sizes,
                        ask_ticks,
                        ask_sizes,
                        l2_best_bid_tick,
                        l2_best_ask_tick,
                        buy_trade_event_pos,
                        buy_trade_price_tick,
                        buy_trade_amount,
                        int(decision_event_index[row]),
                        book_ptr,
                        decision_ts,
                        side_code,
                        price_tick,
                        end_event_index,
                        delay_us,
                        post_only_gap_ticks,
                        trade_at_level_weight,
                        unknown_queue,
                        qty_epsilon,
                    )
                if fill_deadline_ts <= last_event_local_ts_us:
                    labels[row, filled_idx] = 1.0 if filled != 0 else 0.0
                    masks[row, filled_idx] = True
                if filled == 0:
                    if masks[row, filled_idx]:
                        labels[row, cost_idx] = 0.0
                        masks[row, cost_idx] = True
                    continue
                labels[row, latency_idx] = float(latency)
                masks[row, latency_idx] = True

                future_mid = _future_mid_at_or_after_impl(
                    valid_l2_local_ts,
                    valid_l2_event_seq,
                    valid_l2_mid_tick,
                    fill_ts + adverse_horizon_us,
                    max_event_seq,
                )
                if math.isnan(future_mid):
                    if drop_incomplete_horizon != 0:
                        dropped = True
                        break
                    continue
                if side_code > 0:
                    adverse_bps = max(0.0, float(price_tick) - future_mid) / float(price_tick) * 10000.0
                else:
                    adverse_bps = max(0.0, future_mid - float(price_tick)) / float(price_tick) * 10000.0
                labels[row, adverse_idx] = adverse_bps
                labels[row, toxic_idx] = 1.0 if adverse_bps > toxic_threshold_bps else 0.0
                labels[row, cost_idx] = adverse_bps
                masks[row, adverse_idx] = True
                masks[row, toxic_idx] = True
                masks[row, cost_idx] = True
            if dropped:
                break
        if not dropped:
            keep_rows[row] = True


def _counterfactual_fill_one_side(
    tape: ExecutionTape,
    *,
    start_event_index: int,
    start_book_ptr: int,
    decision_key: EventKey,
    side: OrderSide,
    price_tick: int,
    qty: float,
    fill_deadline_key: EventKey,
    end_event_index: int,
    config: CounterfactualQuoteConfig,
) -> CounterfactualFillResult:
    sim_config = FillSimulatorConfig(queue_model=config.queue_model, maker_fee_bps=config.maker_fee_bps)
    activation_key = _activation_key_for_latency(decision_key, config.latency_config.order_activation_delay_us)
    queue_ahead = _queue_ahead_for_candidate(tape, book_ptr=start_book_ptr, side=side, price_tick=price_tick, queue_model=config.queue_model)
    queue_at_start = queue_ahead
    quote = QuoteIntent(
        bid_enabled=side == OrderSide.BUY, ask_enabled=side == OrderSide.SELL,
        bid_price_tick=price_tick if side == OrderSide.BUY else 0, ask_price_tick=price_tick if side == OrderSide.SELL else 0,
        bid_qty=qty if side == OrderSide.BUY else 0.0, ask_qty=qty if side == OrderSide.SELL else 0.0,
    )
    orders = place_orders_from_quote(
        quote, next_order_id=0, created_key=decision_key, bid_effective_key=activation_key, ask_effective_key=activation_key,
        bid_queue_ahead_qty=queue_ahead if side == OrderSide.BUY else 0.0,
        ask_queue_ahead_qty=queue_ahead if side == OrderSide.SELL else 0.0,
    )
    current_book_ptr = int(start_book_ptr)
    recent_trade_depletion_by_level: dict[tuple[OrderSide, int], float] = {}
    events = tape.arrays.events

    def activate_at(event_key: EventKey, book_ptr: int) -> None:
        nonlocal orders, queue_at_start
        refreshed: list[ActiveOrder] = []
        for order in orders:
            if order.status == OrderStatus.PENDING_NEW and order.effective_key <= event_key:
                refreshed_q = _queue_ahead_for_candidate(tape, book_ptr=book_ptr, side=order.side, price_tick=order.price_tick, queue_model=config.queue_model)
                if order.order_id == 0:
                    queue_at_start = refreshed_q
                order = replace(order, queue_ahead_qty=refreshed_q)
            refreshed.append(order)
        result = activate_pending_orders(refreshed, event_key=event_key, book_top=_book_top_obj_from_l2_row(tape, book_ptr), post_only_gap_ticks=config.post_only_gap_ticks)
        orders = result.orders

    def return_fill(fill: Fill) -> CounterfactualFillResult:
        return _clean_fill_result(
            filled=True, fill_local_ts_us=fill.local_ts_us, fill_price_tick=fill.price_tick, fill_reason=fill.reason,
            decision_local_ts_us=decision_key.local_ts_us, queue_ahead_at_start=queue_at_start, queue_ahead_before_fill=fill.queue_ahead_before,
        )

    if fill_deadline_key.event_seq != MAX_EVENT_SEQ:
        raise ValueError("fill_deadline_key must use MAX_EVENT_SEQ")
    activate_at(decision_key, current_book_ptr)
    limit = min(end_event_index, len(events))
    for event_idx in range(start_event_index + 1, limit):
        event = events[event_idx]
        event_key = EventKey(int(event["local_ts_us"]), int(event["event_seq"]))
        pre_key = EventKey(event_key.local_ts_us, event_key.event_seq - 1) if event_key.event_seq > 0 else (EventKey(event_key.local_ts_us - 1, MAX_EVENT_SEQ) if event_key.local_ts_us > 1 else event_key)
        activate_at(pre_key, current_book_ptr)
        if int(event["event_type_code"]) == EVENT_TYPE_CODE_TRADE:
            trade_ptr = int(event["trade_ptr"])
            if trade_ptr >= 0:
                row = tape.arrays.trades[trade_ptr]
                agg_side = _trade_side_from_code(int(row["side_code"]))
                if agg_side is not None and float(row["amount"]) > 0.0:
                    trade = TradePrint(local_ts_us=int(row["local_ts_us"]), ts_us=int(row["ts_us"]), side=agg_side, price_tick=int(row["price_tick"]), amount=float(row["amount"]), trade_id=str(int(row["source_row"])), source_row=int(row["source_row"]))
                    result = simulate_trade_event(orders, trade, event_key=event_key, symbol_spec=tape.manifest.symbol_spec, config=sim_config)
                    orders = result.orders
                    if result.fills:
                        return return_fill(result.fills[0])
                    dedupe_qty = sum(u.trade_advance_qty + u.fillable_qty for u in result.queue_updates if u.trade_at_level)
                    level_side = OrderSide.BUY if agg_side == AggressorSide.SELL else OrderSide.SELL
                    if dedupe_qty > sim_config.qty_epsilon and config.queue_model.mode == QueueModelMode.BALANCED and config.queue_model.dedupe_l2_decrease_with_trade_prints:
                        key = (level_side, trade.price_tick)
                        recent_trade_depletion_by_level[key] = recent_trade_depletion_by_level.get(key, 0.0) + dedupe_qty
        elif int(event["event_type_code"]) == EVENT_TYPE_CODE_L2_BATCH:
            curr_book_ptr = int(event["book_ptr"])
            if curr_book_ptr >= 0:
                updated_orders = orders
                for order in tuple(updated_orders):
                    if not order.is_live:
                        continue
                    level_key = (order.side, order.price_tick)
                    if not _has_fillable_order_at_level(updated_orders, side=order.side, price_tick=order.price_tick, event_key=event_key):
                        continue
                    prev_qty, prev_known = _level_qty_with_depth_status(tape, book_ptr=current_book_ptr, side=order.side, price_tick=order.price_tick)
                    curr_qty, curr_known = _level_qty_with_depth_status(tape, book_ptr=curr_book_ptr, side=order.side, price_tick=order.price_tick)
                    if not (prev_known and curr_known):
                        continue
                    raw_decrease = max((prev_qty if prev_qty is not None else 0.0) - (curr_qty if curr_qty is not None else 0.0), 0.0)
                    if config.queue_model.dedupe_l2_decrease_with_trade_prints:
                        already = recent_trade_depletion_by_level.get(level_key, 0.0)
                        deduped = min(raw_decrease, already)
                        l2_decrease_qty = raw_decrease - deduped
                        recent_trade_depletion_by_level[level_key] = max(already - deduped, 0.0)
                    else:
                        l2_decrease_qty = raw_decrease
                    result = simulate_l2_level_update(updated_orders, side=order.side, price_tick=order.price_tick, l2_decrease_qty=l2_decrease_qty, event_key=event_key, symbol_spec=tape.manifest.symbol_spec, config=sim_config)
                    updated_orders = result.orders
                    if result.fills:
                        orders = updated_orders
                        return return_fill(result.fills[0])
                orders = updated_orders
                current_book_ptr = curr_book_ptr
        activate_at(event_key, current_book_ptr)
        for order in orders:
            if order.status == OrderStatus.REJECTED:
                return _clean_fill_result(filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None, decision_local_ts_us=decision_key.local_ts_us, queue_ahead_at_start=queue_at_start, queue_ahead_before_fill=order.queue_ahead_qty)
    before = orders[0].queue_ahead_qty if orders else queue_at_start
    return _clean_fill_result(filled=False, fill_local_ts_us=-1, fill_price_tick=-1, fill_reason=None, decision_local_ts_us=decision_key.local_ts_us, queue_ahead_at_start=queue_at_start, queue_ahead_before_fill=before)


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


def _invalid_quote(config: AdverseSelectionConfig, message: str) -> bool:
    if config.quote.invalid_quote_policy == "error":
        raise ValueError(message)
    return False


def _side_valid_for_candidate(tape: ExecutionTape, config: AdverseSelectionConfig, *, side: OrderSide, price_tick: int | None) -> bool:
    if price_tick is None or price_tick <= 0:
        return _invalid_quote(config, "candidate quote price is invalid or post-only unsafe")
    spec = tape.manifest.symbol_spec
    if not spec.is_valid_notional(config.quote.order_qty, price_tick):
        return _invalid_quote(config, "candidate quote notional is invalid for tape symbol_spec")
    return True




class FutureMidLookup(Protocol):
    def future_mid_tick_at_or_after(self, key: EventKey) -> float | None: ...


def _labels_for_decision(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig,
    layout: _AdverseLabelLayout,
    last_event_local_ts_us: int,
    events_local_ts_us: np.ndarray,
    decision_event_index: int,
    latest_book_ptr: int,
    decision_key: EventKey,
    future_mid_lookup: FutureMidLookup,
    fill_index: _ConservativeFillIndex | None = None,
) -> tuple[list[float], list[bool]] | None:
    best_bid, best_ask, _, _, _, _ = _book_top_from_l2_row(tape, latest_book_ptr)
    labels = [math.nan] * layout.label_count
    masks = [False] * layout.label_count
    fill_deadline_key = _deadline_key(decision_key, config.quote.fill_horizon_us)
    if config.drop_incomplete_horizon and fill_deadline_key.local_ts_us > last_event_local_ts_us:
        return None
    end_event_index = int(np.searchsorted(events_local_ts_us, fill_deadline_key.local_ts_us, side="right"))

    for candidate_index, candidate in enumerate(config.quote.quote_candidates):
        base = layout.candidate_bases[candidate_index]
        fills_by_side: dict[bool, CounterfactualFillResult] = {}
        for is_bid, side in ((True, OrderSide.BUY), (False, OrderSide.SELL)):
            price_tick = candidate_price_tick(candidate=candidate, side=side, best_bid=best_bid, best_ask=best_ask, post_only_gap_ticks=config.quote.post_only_gap_ticks)
            filled_idx = base + (0 if is_bid else 1)
            latency_idx = base + (2 if is_bid else 3)
            if not _side_valid_for_candidate(tape, config, side=side, price_tick=price_tick):
                continue
            assert price_tick is not None
            if fill_index is not None:
                fill = _conservative_fill_one_side(
                    fill_index, tape, start_event_index=decision_event_index, start_book_ptr=latest_book_ptr,
                    decision_key=decision_key, side=side, price_tick=price_tick,
                    end_event_index=end_event_index, config=config.quote,
                )
            else:
                fill = _counterfactual_fill_one_side(
                    tape, start_event_index=decision_event_index, start_book_ptr=latest_book_ptr, decision_key=decision_key,
                    side=side, price_tick=price_tick, qty=config.quote.order_qty, fill_deadline_key=fill_deadline_key,
                    end_event_index=end_event_index, config=config.quote,
                )
            fills_by_side[is_bid] = fill
            if fill_deadline_key.local_ts_us <= last_event_local_ts_us:
                labels[filled_idx] = 1.0 if fill.filled else 0.0
                masks[filled_idx] = True
            if fill.filled:
                labels[latency_idx] = float(fill.fill_latency_us)
                masks[latency_idx] = True

        for is_bid, fill in fills_by_side.items():
            filled_idx = base + (0 if is_bid else 1)
            adverse_idx = base + (4 if is_bid else 5)
            toxic_idx = base + (6 if is_bid else 7)
            cost_idx = base + (8 if is_bid else 9)
            if not fill.filled:
                if masks[filled_idx]:
                    labels[cost_idx] = 0.0
                    masks[cost_idx] = True
                continue
            future_key = EventKey(fill.fill_local_ts_us + config.quote.adverse_horizon_us, MAX_EVENT_SEQ)
            future_mid = future_mid_lookup.future_mid_tick_at_or_after(future_key)
            if future_mid is None:
                if config.drop_incomplete_horizon:
                    return None
                continue
            if is_bid:
                adverse_bps = max(0.0, fill.fill_price_tick - future_mid) / fill.fill_price_tick * 10_000.0
            else:
                adverse_bps = max(0.0, future_mid - fill.fill_price_tick) / fill.fill_price_tick * 10_000.0
            labels[adverse_idx] = adverse_bps
            labels[toxic_idx] = 1.0 if adverse_bps > config.quote.toxic_threshold_bps else 0.0
            labels[cost_idx] = adverse_bps
            masks[adverse_idx] = masks[toxic_idx] = masks[cost_idx] = True
    return labels, masks


def _candidate_kernel_arrays(config: AdverseSelectionConfig) -> tuple[np.ndarray, np.ndarray]:
    modes = np.empty(len(config.quote.quote_candidates), dtype=np.int64)
    offsets = np.empty(len(config.quote.quote_candidates), dtype=np.int64)
    for i, candidate in enumerate(config.quote.quote_candidates):
        if candidate.mode == QuoteCandidateMode.TOUCH:
            modes[i] = 0
        elif candidate.mode == QuoteCandidateMode.INSIDE:
            modes[i] = 1
        elif candidate.mode == QuoteCandidateMode.AWAY:
            modes[i] = 2
        else:
            raise ValueError("unsupported quote candidate mode")
        offsets[i] = candidate.offset_ticks
    return modes, offsets


def _empty_label_batch(row_count: int, label_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full((row_count, label_count), np.nan, dtype=np.float32)
    masks = np.zeros((row_count, label_count), dtype=np.bool_)
    keep_rows = np.zeros(row_count, dtype=np.bool_)
    return labels, masks, keep_rows


def _labels_for_decision_batch_scalar(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig,
    layout: _AdverseLabelLayout,
    last_event_local_ts_us: int,
    events_local_ts_us: np.ndarray,
    decision_event_index: np.ndarray,
    latest_book_ptr: np.ndarray,
    decision_local_ts_us: np.ndarray,
    decision_event_seq: np.ndarray,
    future_mid_lookup: FutureMidLookup,
    fill_index: _ConservativeFillIndex | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = int(decision_local_ts_us.shape[0])
    labels, masks, keep_rows = _empty_label_batch(row_count, layout.label_count)
    for row in range(row_count):
        labels_info = _labels_for_decision(
            tape,
            config=config,
            layout=layout,
            last_event_local_ts_us=last_event_local_ts_us,
            events_local_ts_us=events_local_ts_us,
            decision_event_index=int(decision_event_index[row]),
            latest_book_ptr=int(latest_book_ptr[row]),
            decision_key=EventKey(int(decision_local_ts_us[row]), int(decision_event_seq[row])),
            future_mid_lookup=future_mid_lookup,
            fill_index=fill_index,
        )
        if labels_info is None:
            continue
        row_labels, row_masks = labels_info
        labels[row] = np.asarray(row_labels, dtype=np.float32)
        masks[row] = np.asarray(row_masks, dtype=np.bool_)
        keep_rows[row] = True
    return labels, masks, keep_rows


def _labels_for_decision_batch_conservative(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig,
    layout: _AdverseLabelLayout,
    last_event_local_ts_us: int,
    events_local_ts_us: np.ndarray,
    decision_event_index: np.ndarray,
    latest_book_ptr: np.ndarray,
    decision_local_ts_us: np.ndarray,
    decision_event_seq: np.ndarray,
    future_mid_lookup: FutureMidLookup,
    fill_index: _ConservativeFillIndex,
    label_engine: str = _LABEL_ENGINE_AUTO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    backend = _resolve_conservative_label_backend(label_engine, invalid_quote_policy=config.quote.invalid_quote_policy)
    if backend == _LABEL_ENGINE_SCALAR:
        labels, masks, keep_rows = _labels_for_decision_batch_scalar(
            tape,
            config=config,
            layout=layout,
            last_event_local_ts_us=last_event_local_ts_us,
            events_local_ts_us=events_local_ts_us,
            decision_event_index=decision_event_index,
            latest_book_ptr=latest_book_ptr,
            decision_local_ts_us=decision_local_ts_us,
            decision_event_seq=decision_event_seq,
            future_mid_lookup=future_mid_lookup,
            fill_index=fill_index,
        )
        return labels, masks, keep_rows, backend

    labels, masks, keep_rows = _empty_label_batch(int(decision_local_ts_us.shape[0]), layout.label_count)
    candidate_modes, candidate_offsets = _candidate_kernel_arrays(config)
    spec = tape.manifest.symbol_spec
    _conservative_labels_kernel_impl(
        labels,
        masks,
        keep_rows,
        np.ascontiguousarray(decision_local_ts_us, dtype=np.int64),
        np.ascontiguousarray(decision_event_index, dtype=np.int64),
        np.ascontiguousarray(latest_book_ptr, dtype=np.int64),
        fill_index.event_local_ts,
        fill_index.last_l2_event_pos,
        fill_index.event_book_ptr,
        tape.arrays.book_bid_ticks,
        tape.arrays.book_bid_sizes,
        tape.arrays.book_ask_ticks,
        tape.arrays.book_ask_sizes,
        fill_index.l2_best_bid_tick,
        fill_index.l2_best_ask_tick,
        fill_index.buy_trade_event_pos,
        fill_index.buy_trade_price_tick,
        fill_index.buy_trade_amount,
        fill_index.sell_trade_event_pos,
        fill_index.sell_trade_price_tick,
        fill_index.sell_trade_amount,
        future_mid_lookup.local_ts_us,  # type: ignore[attr-defined]
        future_mid_lookup.event_seq,  # type: ignore[attr-defined]
        future_mid_lookup.mid_tick,  # type: ignore[attr-defined]
        candidate_modes,
        candidate_offsets,
        config.quote.fill_horizon_us,
        config.quote.adverse_horizon_us,
        int(last_event_local_ts_us),
        1 if config.drop_incomplete_horizon else 0,
        config.quote.post_only_gap_ticks,
        config.quote.latency_config.order_activation_delay_us,
        config.quote.queue_model.trade_at_level_weight,
        config.quote.queue_model.unknown_level_queue_ahead_qty,
        config.quote.queue_model.qty_epsilon,
        config.quote.toxic_threshold_bps,
        config.quote.order_qty,
        spec.tick_size,
        spec.contract_size,
        spec.min_notional,
    )
    return labels, masks, keep_rows, backend


def _dataset_manifest_matches(
    dataset,
    *,
    metadata: Mapping[str, object],
    feature_names: tuple[str, ...],
    label_names: tuple[str, ...],
) -> bool:
    manifest = dataset.manifest
    def split_source_view(contract: Mapping[str, object]) -> dict[str, object]:
        return {key: contract[key] for key in (
            "schema",
            "version",
            "split_source_dataset_root",
            "split_source_dataset_id",
            "split_source_manifest_hash",
            "ranges",
            "source_row_counts",
            "decision_grid_schema",
            "decision_grid_hash",
            "decision_grid_n_rows",
            "decision_schedule",
        )}

    return (
        manifest.exchange == metadata["exchange"]
        and manifest.symbol == metadata["symbol"]
        and manifest.tape_schema == metadata["tape_schema"]
        and manifest.tape_num_events == metadata["tape_num_events"]
        and manifest.tape_num_l2_batches == metadata["tape_num_l2_batches"]
        and manifest.tape_num_trades == metadata["tape_num_trades"]
        and manifest.tape_start_local_ts_us == metadata["tape_start_local_ts_us"]
        and manifest.tape_end_local_ts_us == metadata["tape_end_local_ts_us"]
        and manifest.decision_grid_schema == metadata["decision_grid_schema"]
        and manifest.decision_grid_hash == metadata["decision_grid_hash"]
        and manifest.decision_grid_n_rows == metadata["decision_grid_n_rows"]
        and dict(manifest.decision_schedule) == dict(metadata["decision_schedule"])  # type: ignore[arg-type]
        and manifest.split_source_dataset_root == metadata["split_source_dataset_root"]
        and manifest.split_source_dataset_id == metadata["split_source_dataset_id"]
        and manifest.split_source_manifest_hash == metadata["split_source_manifest_hash"]
        and split_source_view(manifest.split_contract) == split_source_view(dict(metadata["split_contract"]))  # type: ignore[arg-type]
        and manifest.config_json == metadata["config_json"]
        and manifest.index_schema == metadata["index_schema"]
        and manifest.index_manifest_sha256 == metadata["index_manifest_sha256"]
        and manifest.feature_names == feature_names
        and manifest.label_names == label_names
    )


def _load_reusable_adverse_dataset(
    root_path: Path,
    *,
    metadata: Mapping[str, object],
    feature_names: tuple[str, ...],
    label_names: tuple[str, ...],
):
    from mmrt.execution.adverse_selection_dataset import load_adverse_selection_dataset

    if not root_path.exists():
        return None
    manifest_path = root_path / "manifest.json"
    if not manifest_path.exists():
        return None
    dataset = load_adverse_selection_dataset(root_path, mmap_mode="r")
    if _dataset_manifest_matches(dataset, metadata=metadata, feature_names=feature_names, label_names=label_names):
        return dataset
    return None



class _AdverseFeatureRow(NamedTuple):
    decision_local_ts_us: int
    decision_event_index: int
    decision_event_seq: int
    features: tuple[float, ...]
    latest_book_ptr: int


def _iter_adverse_selection_feature_rows_for_decision_grid(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig,
    decision_grid: DecisionGrid,
    kyle_samples: Sequence[_KyleSample],
) -> Iterator[_AdverseFeatureRow]:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(config, AdverseSelectionConfig):
        raise ValueError("config must be AdverseSelectionConfig")
    if kyle_samples is None:
        raise ValueError("kyle_samples is required")
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    spec = tape.manifest.symbol_spec
    if not isinstance(spec, SymbolSpec):
        raise ValueError("tape.manifest.symbol_spec must be SymbolSpec")
    if hasattr(spec, "is_valid_qty") and not spec.is_valid_qty(config.quote.order_qty):
        raise ValueError("quote.order_qty is invalid for tape symbol_spec")

    vpin_state = VPINState(config.vpin, deque())
    kyle_state = _new_kyle_state(config.kyle)
    next_kyle_sample_idx = 0
    flow_states = tuple(_TradeWindowState(window_us) for window_us in config.flow_windows_us)

    events = tape.arrays.events
    trades = tape.arrays.trades
    latest_book_ptr = -1
    previous_book_ptr = -1
    next_grid_row = 0
    last_grid_event_index = int(decision_grid.decision_event_index[-1])

    def update_time_states(current_key: EventKey) -> None:
        nonlocal next_kyle_sample_idx
        for flow_state in flow_states:
            flow_state.expire(current_key.local_ts_us)
        while next_kyle_sample_idx < len(kyle_samples) and kyle_samples[next_kyle_sample_idx].end_key <= current_key:
            kyle_state.add_finalized_sample(kyle_samples[next_kyle_sample_idx], current_key)
            next_kyle_sample_idx += 1
        kyle_state.expire_old(current_key)

    for event_index, event in enumerate(events):
        event_ts = int(event["local_ts_us"])
        event_seq = int(event["event_seq"])
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
                    flow_state.update_trade(event_ts, side_code, amount, notional)
                vpin_state.update_trade(side_code=side_code, price_tick=price_tick, amount=amount, tick_size=spec.tick_size)

        if next_grid_row < decision_grid.n_rows and event_index == int(decision_grid.decision_event_index[next_grid_row]):
            grid_book_ptr = int(decision_grid.book_ptr[next_grid_row])
            if latest_book_ptr != grid_book_ptr:
                raise ValueError(
                    f"decision grid row {next_grid_row} book_ptr mismatch during adverse replay: "
                    f"grid={grid_book_ptr} replay={latest_book_ptr}"
                )
            decision_key = EventKey(int(decision_grid.decision_local_ts_us[next_grid_row]), int(decision_grid.decision_event_seq[next_grid_row]))
            if decision_key.local_ts_us != event_ts or decision_key.event_seq != event_seq:
                raise ValueError(f"decision grid row {next_grid_row} does not match tape event key")
            update_time_states(decision_key)
            yield _AdverseFeatureRow(
                decision_local_ts_us=decision_key.local_ts_us,
                decision_event_index=event_index,
                decision_event_seq=decision_key.event_seq,
                features=_feature_row(
                    tape,
                    latest_book_ptr=latest_book_ptr,
                    previous_book_ptr=previous_book_ptr,
                    spec=spec,
                    vpin_state=vpin_state,
                    flow_states=flow_states,
                    kyle_state=kyle_state,
                ),
                latest_book_ptr=latest_book_ptr,
            )
            next_grid_row += 1
            if next_grid_row == decision_grid.n_rows:
                return
        if event_index >= last_grid_event_index:
            break
    raise ValueError(f"adverse replay consumed {next_grid_row} decision grid rows, expected {decision_grid.n_rows}")


@dataclass(frozen=True, slots=True)
class BuildAdverseSelectionDatasetToDiskConfig:
    output_root: str
    chunk_rows: int = 100_000
    overwrite: bool = False
    cleanup_chunks: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", _require_nonempty_str(self.output_root, "output_root"))
        object.__setattr__(self, "chunk_rows", _require_positive_int(self.chunk_rows, "chunk_rows"))
        object.__setattr__(self, "overwrite", _require_bool(self.overwrite, "overwrite"))
        object.__setattr__(self, "cleanup_chunks", _require_bool(self.cleanup_chunks, "cleanup_chunks"))


def _adverse_config_summary(config: AdverseSelectionConfig) -> dict[str, object]:
    return {
        "flow_windows_us": list(config.flow_windows_us),
        "drop_incomplete_horizon": config.drop_incomplete_horizon,
        "vpin_bucket_volume": config.vpin.bucket_volume,
        "vpin_num_buckets": config.vpin.num_buckets,
        "vpin_min_completed_buckets": config.vpin.min_completed_buckets,
        "vpin_use_notional_volume": config.vpin.use_notional_volume,
        "kyle_sample_interval_us": config.kyle.sample_interval_us,
        "kyle_response_horizon_us": config.kyle.response_horizon_us,
        "kyle_windows_us": list(config.kyle.windows_us),
        "kyle_min_samples": config.kyle.min_samples,
        "kyle_use_notional_flow": config.kyle.use_notional_flow,
        "quote_candidates": [candidate.name for candidate in config.quote.quote_candidates],
        "post_only_gap_ticks": config.quote.post_only_gap_ticks,
        "decision_compute_latency_us": config.quote.latency_config.decision_compute_latency_us,
        "order_entry_latency_us": config.quote.latency_config.order_entry_latency_us,
        "invalid_quote_policy": config.quote.invalid_quote_policy,
        "order_qty": config.quote.order_qty,
        "fill_horizon_us": config.quote.fill_horizon_us,
        "adverse_horizon_us": config.quote.adverse_horizon_us,
        "toxic_threshold_bps": config.quote.toxic_threshold_bps,
        "queue_mode": config.quote.queue_model.mode.value,
        "l2_decrease_weight": config.quote.queue_model.l2_decrease_weight,
        "trade_at_level_weight": config.quote.queue_model.trade_at_level_weight,
        "unknown_level_queue_ahead_qty": config.quote.queue_model.unknown_level_queue_ahead_qty,
        "dedupe_l2_decrease_with_trade_prints": config.quote.queue_model.dedupe_l2_decrease_with_trade_prints,
    }


def build_adverse_selection_dataset_to_disk(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig = AdverseSelectionConfig(),
    decision_grid: DecisionGrid,
    split_contract: Mapping[str, object],
    output_root: object,
    work_dir: object | None = None,
    chunk_rows: int = 100_000,
    overwrite: bool = False,
    cleanup_chunks: bool = True,
    cleanup_work_dir: bool = True,
    progress_interval: int | None = None,
    label_engine: str = _LABEL_ENGINE_AUTO,
):
    from mmrt.execution.adverse_selection_dataset import AdverseSelectionDatasetWriter, AdverseSelectionDatasetWriterConfig
    from mmrt.execution.adverse_selection_index import AdverseSelectionIndexConfig, adverse_selection_index_manifest_sha256, build_or_load_adverse_selection_index

    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(config, AdverseSelectionConfig):
        raise ValueError("config must be AdverseSelectionConfig")
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    label_engine = _coerce_label_engine(label_engine)
    if progress_interval is not None:
        progress_interval = _require_positive_int(progress_interval, "progress_interval")
    spec = tape.manifest.symbol_spec
    if not isinstance(spec, SymbolSpec):
        raise ValueError("tape.manifest.symbol_spec must be SymbolSpec")
    if hasattr(spec, "is_valid_qty") and not spec.is_valid_qty(config.quote.order_qty):
        raise ValueError("quote.order_qty is invalid for tape symbol_spec")
    root_path = Path(output_root)
    index_root = (Path(work_dir) if work_dir is not None else root_path.parent) / "adverse_selection_work"
    _log_stage(f"adverse_dataset stage=index_build_or_load root={index_root}")
    index = build_or_load_adverse_selection_index(
        tape,
        config=AdverseSelectionIndexConfig(
            output_root=str(index_root),
            kyle=config.kyle,
            use_notional_flow=config.kyle.use_notional_flow,
            tick_size=spec.tick_size,
            chunk_rows=chunk_rows,
            overwrite=overwrite,
            cleanup_chunks=cleanup_chunks,
        ),
    )
    if index.valid_l2.count == 0:
        raise ValueError("tape must contain at least one valid two-sided L2 book")
    _log_stage(
        "adverse_dataset stage=index_ready "
        f"valid_l2={index.manifest.valid_l2_count} trade_flow={index.manifest.trade_flow_count} "
        f"kyle_samples={index.manifest.kyle_sample_count}"
    )

    feature_names = adverse_selection_feature_names(config)
    layout = _AdverseLabelLayout.from_config(config)
    manifest = tape.manifest
    metadata = {
        "exchange": manifest.exchange,
        "symbol": manifest.symbol,
        "tape_schema": manifest.schema,
        "tape_num_events": manifest.num_events,
        "tape_num_l2_batches": manifest.num_l2_batches,
        "tape_num_trades": manifest.num_trades,
        "tape_start_local_ts_us": manifest.start_local_ts_us,
        "tape_end_local_ts_us": manifest.end_local_ts_us,
        "decision_grid_schema": decision_grid.metadata.schema,
        "decision_grid_hash": decision_grid.decision_grid_hash,
        "decision_grid_n_rows": decision_grid.n_rows,
        "decision_schedule": decision_grid.decision_schedule,
        "split_source_dataset_root": str(split_contract["split_source_dataset_root"]),
        "split_source_dataset_id": str(split_contract["split_source_dataset_id"]),
        "split_source_manifest_hash": str(split_contract["split_source_manifest_hash"]),
        "split_contract": dict(split_contract),
        "config_json": json.dumps(_adverse_config_summary(config), sort_keys=True),
        "index_schema": index.manifest.schema,
        "index_manifest_sha256": adverse_selection_index_manifest_sha256(index.root),
        "index_root": str(index.root),
    }
    if root_path.exists() and not overwrite:
        reusable = _load_reusable_adverse_dataset(
            root_path,
            metadata=metadata,
            feature_names=feature_names,
            label_names=layout.label_names,
        )
        if reusable is not None:
            _log_stage(f"adverse_dataset stage=dataset_reuse root={root_path} rows={reusable.num_rows}")
            return reusable
        if not (root_path / "manifest.json").exists():
            raise FileExistsError(f"partial adverse dataset root exists without manifest: {root_path}; pass overwrite=True to rebuild")
        raise FileExistsError(f"adverse dataset root exists with non-matching manifest: {root_path}; pass overwrite=True to rebuild")

    writer = AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(
        output_root=str(output_root), feature_names=feature_names, label_names=layout.label_names,
        manifest_metadata=metadata, chunk_rows=chunk_rows, overwrite=overwrite, cleanup_chunks=cleanup_chunks,
    ))

    fill_index_start = time.perf_counter()
    fill_index = (
        _build_conservative_fill_index(tape)
        if config.quote.queue_model.mode == QueueModelMode.CONSERVATIVE
        else None
    )
    backend_used = "balanced_scalar"
    if fill_index is not None:
        backend_used = _resolve_conservative_label_backend(label_engine, invalid_quote_policy=config.quote.invalid_quote_policy)
        _log_stage(
            "adverse_dataset stage=conservative_fill_index_ready "
            f"seconds={time.perf_counter() - fill_index_start:.3f} backend={backend_used}"
        )
    events = tape.arrays.events
    events_local_ts_us = events["local_ts_us"]
    last_ts = int(events_local_ts_us[-1])
    emitted = 0
    considered = 0
    kyle_samples = _DiskKyleSampleView(index.kyle_samples)
    feature_count = len(feature_names)
    batch_ts = np.empty(chunk_rows, dtype=np.int64)
    batch_event_index = np.empty(chunk_rows, dtype=np.int64)
    batch_event_seq = np.empty(chunk_rows, dtype=np.int64)
    batch_book_ptr = np.empty(chunk_rows, dtype=np.int64)
    batch_features = np.empty((chunk_rows, feature_count), dtype=np.float32)
    batch_used = 0
    started = time.perf_counter()
    last_progress_considered = 0

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_considered
        if progress_interval is None:
            return
        if not force and considered - last_progress_considered < progress_interval:
            return
        elapsed = max(time.perf_counter() - started, _EPS)
        rows_per_sec = considered / elapsed
        remaining = max(decision_grid.n_rows - considered, 0)
        eta = remaining / rows_per_sec if rows_per_sec > 0.0 else math.inf
        _log_stage(
            "adverse_dataset progress "
            f"decision_grid_rows_considered={considered} rows_written={emitted} "
            f"rows_dropped={considered - emitted} elapsed_seconds={elapsed:.3f} "
            f"rows_per_sec={rows_per_sec:.3f} eta_seconds={eta:.3f} "
            f"label_engine={label_engine} backend={backend_used}"
        )
        last_progress_considered = considered

    def flush_batch() -> None:
        nonlocal batch_used, emitted, backend_used
        if batch_used == 0:
            return
        if fill_index is not None:
            labels, masks, keep_rows, backend = _labels_for_decision_batch_conservative(
                tape,
                config=config,
                layout=layout,
                last_event_local_ts_us=last_ts,
                events_local_ts_us=events_local_ts_us,
                decision_event_index=batch_event_index[:batch_used],
                latest_book_ptr=batch_book_ptr[:batch_used],
                decision_local_ts_us=batch_ts[:batch_used],
                decision_event_seq=batch_event_seq[:batch_used],
                future_mid_lookup=index.valid_l2,
                fill_index=fill_index,
                label_engine=label_engine,
            )
            backend_used = backend
        else:
            labels, masks, keep_rows = _labels_for_decision_batch_scalar(
                tape,
                config=config,
                layout=layout,
                last_event_local_ts_us=last_ts,
                events_local_ts_us=events_local_ts_us,
                decision_event_index=batch_event_index[:batch_used],
                latest_book_ptr=batch_book_ptr[:batch_used],
                decision_local_ts_us=batch_ts[:batch_used],
                decision_event_seq=batch_event_seq[:batch_used],
                future_mid_lookup=index.valid_l2,
                fill_index=None,
            )
            backend_used = "balanced_scalar"
        kept = int(np.count_nonzero(keep_rows))
        _log_stage(f"adverse_dataset stage=batch_flush rows_considered={batch_used} rows_written={kept} backend={backend_used}")
        if kept == 0:
            batch_used = 0
            return
        writer.append_many(
            decision_local_ts_us=batch_ts[:batch_used][keep_rows],
            decision_event_index=batch_event_index[:batch_used][keep_rows],
            decision_event_seq=batch_event_seq[:batch_used][keep_rows],
            features=batch_features[:batch_used][keep_rows],
            labels=labels[keep_rows],
            label_masks=masks[keep_rows],
        )
        emitted += kept
        batch_used = 0

    _log_stage(
        "adverse_dataset stage=label_generation_start "
        f"decision_grid_rows={decision_grid.n_rows} chunk_rows={chunk_rows} label_engine={label_engine} backend={backend_used}"
    )
    for row in _iter_adverse_selection_feature_rows_for_decision_grid(
        tape,
        config=config,
        decision_grid=decision_grid,
        kyle_samples=kyle_samples,
    ):
        considered += 1
        batch_ts[batch_used] = row.decision_local_ts_us
        batch_event_index[batch_used] = row.decision_event_index
        batch_event_seq[batch_used] = row.decision_event_seq
        batch_book_ptr[batch_used] = row.latest_book_ptr
        batch_features[batch_used] = row.features
        batch_used += 1
        if batch_used >= chunk_rows:
            flush_batch()
            log_progress()
    flush_batch()
    log_progress(force=True)
    _log_stage(f"adverse_dataset stage=writer_finalize rows_written={emitted}")
    dataset = writer.finalize()
    if cleanup_work_dir:
        _log_stage(f"adverse_dataset stage=cleanup_work_dir root={index_root}")
        shutil.rmtree(index_root, ignore_errors=True)
    return dataset


def profile_adverse_selection_label_generation(
    tape: ExecutionTape,
    *,
    config: AdverseSelectionConfig = AdverseSelectionConfig(),
    decision_grid: DecisionGrid,
    profile_rows: int,
    work_dir: object | None = None,
    chunk_rows: int = 100_000,
    overwrite: bool = False,
    cleanup_chunks: bool = True,
    progress_interval: int | None = None,
    label_engine: str = _LABEL_ENGINE_AUTO,
) -> dict[str, object]:
    from mmrt.execution.adverse_selection_index import AdverseSelectionIndexConfig, build_or_load_adverse_selection_index

    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(config, AdverseSelectionConfig):
        raise ValueError("config must be AdverseSelectionConfig")
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    profile_rows = _require_positive_int(profile_rows, "profile_rows")
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    label_engine = _coerce_label_engine(label_engine)
    if progress_interval is not None:
        progress_interval = _require_positive_int(progress_interval, "progress_interval")
    spec = tape.manifest.symbol_spec
    if not isinstance(spec, SymbolSpec):
        raise ValueError("tape.manifest.symbol_spec must be SymbolSpec")
    if hasattr(spec, "is_valid_qty") and not spec.is_valid_qty(config.quote.order_qty):
        raise ValueError("quote.order_qty is invalid for tape symbol_spec")

    total_start = time.perf_counter()
    profile_limit = min(profile_rows, decision_grid.n_rows)
    index_root = (Path(work_dir) if work_dir is not None else Path(".")) / "adverse_selection_work"
    _log_stage(f"adverse_dataset_profile stage=index_build_or_load root={index_root}")
    index_start = time.perf_counter()
    index = build_or_load_adverse_selection_index(
        tape,
        config=AdverseSelectionIndexConfig(
            output_root=str(index_root),
            kyle=config.kyle,
            use_notional_flow=config.kyle.use_notional_flow,
            tick_size=spec.tick_size,
            chunk_rows=chunk_rows,
            overwrite=overwrite,
            cleanup_chunks=cleanup_chunks,
        ),
    )
    index_seconds = time.perf_counter() - index_start
    if index.valid_l2.count == 0:
        raise ValueError("tape must contain at least one valid two-sided L2 book")

    fill_index = None
    fill_index_seconds = 0.0
    backend_used = "balanced_scalar"
    if config.quote.queue_model.mode == QueueModelMode.CONSERVATIVE:
        fill_start = time.perf_counter()
        fill_index = _build_conservative_fill_index(tape)
        fill_index_seconds = time.perf_counter() - fill_start
        backend_used = _resolve_conservative_label_backend(label_engine, invalid_quote_policy=config.quote.invalid_quote_policy)
        _log_stage(f"adverse_dataset_profile stage=conservative_fill_index_ready seconds={fill_index_seconds:.3f} backend={backend_used}")

    layout = _AdverseLabelLayout.from_config(config)
    events = tape.arrays.events
    events_local_ts_us = events["local_ts_us"]
    last_ts = int(events_local_ts_us[-1])
    kyle_samples = _DiskKyleSampleView(index.kyle_samples)
    feature_count = len(adverse_selection_feature_names(config))
    batch_ts = np.empty(chunk_rows, dtype=np.int64)
    batch_event_index = np.empty(chunk_rows, dtype=np.int64)
    batch_event_seq = np.empty(chunk_rows, dtype=np.int64)
    batch_book_ptr = np.empty(chunk_rows, dtype=np.int64)
    batch_features = np.empty((chunk_rows, feature_count), dtype=np.float32)
    batch_used = 0
    considered = 0
    emitted = 0
    label_seconds = 0.0
    compile_seconds = 0.0
    first_batch = True
    label_start = time.perf_counter()
    last_progress_considered = 0

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_considered
        if progress_interval is None:
            return
        if not force and considered - last_progress_considered < progress_interval:
            return
        elapsed = max(time.perf_counter() - label_start, _EPS)
        rows_per_sec = considered / elapsed
        _log_stage(
            "adverse_dataset_profile progress "
            f"decision_grid_rows_considered={considered} rows_kept={emitted} "
            f"rows_dropped={considered - emitted} elapsed_seconds={elapsed:.3f} "
            f"rows_per_sec={rows_per_sec:.3f} label_engine={label_engine} backend={backend_used}"
        )
        last_progress_considered = considered

    def process_batch() -> None:
        nonlocal batch_used, emitted, label_seconds, compile_seconds, first_batch, backend_used
        if batch_used == 0:
            return
        start = time.perf_counter()
        if fill_index is not None:
            _, _, keep_rows, backend = _labels_for_decision_batch_conservative(
                tape,
                config=config,
                layout=layout,
                last_event_local_ts_us=last_ts,
                events_local_ts_us=events_local_ts_us,
                decision_event_index=batch_event_index[:batch_used],
                latest_book_ptr=batch_book_ptr[:batch_used],
                decision_local_ts_us=batch_ts[:batch_used],
                decision_event_seq=batch_event_seq[:batch_used],
                future_mid_lookup=index.valid_l2,
                fill_index=fill_index,
                label_engine=label_engine,
            )
            backend_used = backend
        else:
            _, _, keep_rows = _labels_for_decision_batch_scalar(
                tape,
                config=config,
                layout=layout,
                last_event_local_ts_us=last_ts,
                events_local_ts_us=events_local_ts_us,
                decision_event_index=batch_event_index[:batch_used],
                latest_book_ptr=batch_book_ptr[:batch_used],
                decision_local_ts_us=batch_ts[:batch_used],
                decision_event_seq=batch_event_seq[:batch_used],
                future_mid_lookup=index.valid_l2,
                fill_index=None,
            )
            backend_used = "balanced_scalar"
        elapsed = time.perf_counter() - start
        if first_batch and backend_used == _LABEL_ENGINE_NUMBA:
            compile_seconds = elapsed
        else:
            label_seconds += elapsed
        first_batch = False
        emitted += int(np.count_nonzero(keep_rows))
        batch_used = 0

    for row in _iter_adverse_selection_feature_rows_for_decision_grid(
        tape,
        config=config,
        decision_grid=decision_grid,
        kyle_samples=kyle_samples,
    ):
        if considered >= profile_limit:
            break
        batch_ts[batch_used] = row.decision_local_ts_us
        batch_event_index[batch_used] = row.decision_event_index
        batch_event_seq[batch_used] = row.decision_event_seq
        batch_book_ptr[batch_used] = row.latest_book_ptr
        batch_features[batch_used] = row.features
        batch_used += 1
        considered += 1
        if batch_used >= chunk_rows:
            process_batch()
            log_progress()
    process_batch()
    log_progress(force=True)
    total_seconds = time.perf_counter() - total_start
    label_wall_seconds = max(time.perf_counter() - label_start, _EPS)
    return {
        "status": "ok",
        "run_type": "profile_adverse_selection_label_generation",
        "decision_grid_rows_requested": int(profile_rows),
        "decision_grid_rows_considered": int(considered),
        "rows_kept": int(emitted),
        "rows_dropped": int(considered - emitted),
        "label_engine": label_engine,
        "backend": backend_used,
        "timing": {
            "total_seconds": total_seconds,
            "index_seconds": index_seconds,
            "fill_index_seconds": fill_index_seconds,
            "compile_seconds": compile_seconds,
            "label_seconds": label_seconds,
            "label_wall_seconds": label_wall_seconds,
            "rows_per_second": considered / label_wall_seconds,
        },
    }

def summarize_disk_adverse_selection_dataset(dataset, *, chunk_rows: int = 100_000) -> dict[str, object]:
    from mmrt.execution.adverse_selection_dataset import DiskBackedAdverseSelectionDataset
    if not isinstance(dataset, DiskBackedAdverseSelectionDataset):
        raise ValueError("dataset must be DiskBackedAdverseSelectionDataset")
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    label_index = {name: i for i, name in enumerate(dataset.label_names)}
    feature_index = {name: i for i, name in enumerate(dataset.feature_names)}
    labels_summary: dict[str, float] = {}
    label_sum = np.zeros(dataset.num_labels, dtype=np.float64)
    label_count = np.zeros(dataset.num_labels, dtype=np.int64)
    finite_count = 0
    feature_count = 0
    selected = [idx for name, idx in feature_index.items() if name == "vpin" or name.startswith("kyle_lambda_")]
    selected_sum = {idx: 0.0 for idx in selected}
    for start in range(0, dataset.num_rows, chunk_rows):
        end = min(start + chunk_rows, dataset.num_rows)
        masks = np.asarray(dataset.arrays.label_masks[start:end], dtype=np.bool_)
        labels = np.asarray(dataset.arrays.labels[start:end], dtype=np.float64)
        label_sum += np.sum(np.where(masks, labels, 0.0), axis=0)
        label_count += np.sum(masks, axis=0)
        feats = np.asarray(dataset.arrays.features[start:end], dtype=np.float64)
        finite_count += int(np.count_nonzero(np.isfinite(feats)))
        feature_count += int(feats.size)
        for idx in selected:
            selected_sum[idx] += float(np.sum(feats[:, idx]))
    for label_name, idx in label_index.items():
        if label_count[idx] > 0:
            labels_summary[label_name] = float(label_sum[idx] / label_count[idx])
    features_summary: dict[str, object] = {"finite_fraction": float(finite_count / feature_count) if feature_count else 1.0}
    n = max(dataset.num_rows, 1)
    if "vpin" in feature_index:
        features_summary["vpin_mean"] = float(selected_sum[feature_index["vpin"]] / n)
    for name, idx in feature_index.items():
        if name.startswith("kyle_lambda_"):
            features_summary[f"{name}_mean"] = float(selected_sum[idx] / n)
    for name, idx in label_index.items():
        if label_count[idx] > 0:
            labels_summary.setdefault(name, float(label_sum[idx] / label_count[idx]))
    return {
        "num_decisions": dataset.num_rows,
        "num_features": dataset.num_features,
        "num_labels": dataset.num_labels,
        "feature_names": list(dataset.feature_names),
        "label_names": list(dataset.label_names),
        "labels": labels_summary,
        "features": features_summary,
    }


def _masked_mean(values: np.ndarray, masks: np.ndarray) -> float:
    if values.size == 0 or not masks.any():
        return 0.0
    return float(np.mean(values[masks]))



def summarize_adverse_selection_feature_dataset(
    dataset: AdverseSelectionFeatureDataset,
) -> dict[str, object]:
    if not isinstance(dataset, AdverseSelectionFeatureDataset):
        raise ValueError("dataset must be AdverseSelectionFeatureDataset")
    feature_index = {name: i for i, name in enumerate(dataset.feature_names)}
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
        "feature_names": list(dataset.feature_names),
        "features": features_summary,
    }


def summarize_adverse_selection_dataset(
    dataset: AdverseSelectionDataset,
) -> dict[str, object]:
    if not isinstance(dataset, AdverseSelectionDataset):
        raise ValueError("dataset must be AdverseSelectionDataset")
    label_index = {name: i for i, name in enumerate(dataset.label_names)}
    feature_index = {name: i for i, name in enumerate(dataset.feature_names)}
    labels_summary: dict[str, float] = {}
    for candidate in dataset.config.quote.quote_candidates:
        c = candidate.name
        keys = {
            f"{c}_bid_fill_rate": f"bid_{c}_filled",
            f"{c}_ask_fill_rate": f"ask_{c}_filled",
            f"{c}_bid_toxic_fill_rate": f"bid_{c}_toxic_fill",
            f"{c}_ask_toxic_fill_rate": f"ask_{c}_toxic_fill",
            f"{c}_bid_adverse_bps_mean_conditional": f"bid_{c}_adverse_bps",
            f"{c}_ask_adverse_bps_mean_conditional": f"ask_{c}_adverse_bps",
            f"{c}_bid_toxic_cost_bps_mean_unconditional": f"bid_{c}_toxic_cost_bps",
            f"{c}_ask_toxic_cost_bps_mean_unconditional": f"ask_{c}_toxic_cost_bps",
        }
        for out_name, label_name in keys.items():
            if label_name in label_index:
                idx = label_index[label_name]
                labels_summary[out_name] = _masked_mean(dataset.labels[:, idx], dataset.label_masks[:, idx])
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
