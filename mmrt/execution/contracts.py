"""Stable typed contracts for the MMRT execution/RL layer.

This module is intentionally IO-free and dependency-light. It defines enums,
validation helpers, and immutable dataclasses shared by L2 reconstruction,
execution-tape building, quote geometry, fill simulation, rewards, and RL
environment orchestration.

It does not parse market data, reconstruct books, simulate fills, compute
rewards, write artifacts, or import ML/dataframe libraries.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from mmrt.contracts import AggressorSide, BookSide, TardisDataType
from mmrt.metadata.rule_compatibility import RuleCompatibilityReport
from mmrt.metadata.symbol_rules import ExchangeSymbolRules
from mmrt.time_key import EventKey


class _StrEnum(str, Enum):
    """String-valued enum base."""


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    return _require_int(value, name, allow_zero=True)


def _require_positive_int(value: int, name: str) -> int:
    return _require_int(value, name)


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _clean_info_value(value: object, name: str) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _require_finite_float(value, name)
    if isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a JSON-safe scalar")


def _coerce_enum(enum_cls: type[Enum], value: Any, name: str):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ValueError(f"{name} has invalid value {value!r}") from exc
    raise ValueError(f"{name} must be {enum_cls.__name__} or str")


def _tuple_of_nonempty_str(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of non-empty strings, not a single string/bytes value")
    try:
        seq = values if isinstance(values, tuple) else tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of non-empty strings") from exc
    return tuple(_require_nonempty_str(v, f"{name}[{i}]") for i, v in enumerate(seq))


def _tuple_of_positive_ints(values: Any, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable, not a single string/bytes value")
    try:
        seq = values if isinstance(values, tuple) else tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of positive ints") from exc
    return tuple(_require_positive_int(v, f"{name}[{i}]") for i, v in enumerate(seq))


def _tuple_of_nonnegative_floats(values: Any, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable, not a single string/bytes value")
    try:
        seq = values if isinstance(values, tuple) else tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of finite floats >= 0") from exc
    return tuple(_require_nonnegative_float(v, f"{name}[{i}]") for i, v in enumerate(seq))


class ExecutionTapeFormat(_StrEnum):
    L2_TRADES_ARRAYS = "l2_trades_arrays"


class ExecutionEventType(_StrEnum):
    L2_BATCH = "l2_batch"
    TRADE = "trade"
    DECISION = "decision"


class OrderSide(_StrEnum):
    BUY = "buy"
    SELL = "sell"


class QuoteSide(_StrEnum):
    BID = "bid"
    ASK = "ask"


class OrderStatus(_StrEnum):
    PENDING_NEW = "pending_new"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    PENDING_CANCEL = "pending_cancel"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class QueueModelMode(_StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"


class FillReason(_StrEnum):
    TRADE_THROUGH = "trade_through"
    TRADE_AT_LEVEL = "trade_at_level"
    QUEUE_DEPLETION = "queue_depletion"


class RewardMode(_StrEnum):
    MARK_TO_MARKET_DELTA = "mark_to_market_delta"


class ActionMode(_StrEnum):
    CONTINUOUS_LATENT = "continuous_latent"


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    decision_compute_latency_us: int = 50
    order_entry_latency_us: int = 500
    cancel_latency_us: int = 500

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.decision_compute_latency_us, "decision_compute_latency_us")
        _require_nonnegative_int(self.order_entry_latency_us, "order_entry_latency_us")
        _require_nonnegative_int(self.cancel_latency_us, "cancel_latency_us")

    @property
    def order_activation_delay_us(self) -> int:
        return self.decision_compute_latency_us + self.order_entry_latency_us

    @property
    def cancel_effective_delay_us(self) -> int:
        return self.decision_compute_latency_us + self.cancel_latency_us


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    exchange: str
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    max_qty: float
    min_notional: float = 0.0
    contract_size: float = 1.0

    def __post_init__(self) -> None:
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")
        object.__setattr__(self, "tick_size", _require_positive_float(self.tick_size, "tick_size"))
        object.__setattr__(self, "step_size", _require_positive_float(self.step_size, "step_size"))
        object.__setattr__(self, "min_qty", _require_nonnegative_float(self.min_qty, "min_qty"))
        object.__setattr__(self, "max_qty", _require_positive_float(self.max_qty, "max_qty"))
        object.__setattr__(self, "min_notional", _require_nonnegative_float(self.min_notional, "min_notional"))
        object.__setattr__(self, "contract_size", _require_positive_float(self.contract_size, "contract_size"))
        if self.max_qty < self.min_qty:
            raise ValueError("max_qty must be >= min_qty")

    def price_to_tick(self, price: float) -> int:
        """Round price to nearest integer tick; validate finite positive price."""
        price = _require_positive_float(price, "price")
        return int(round(price / self.tick_size))

    def tick_to_price(self, price_tick: int) -> float:
        """Convert positive integer tick to float price."""
        return _require_positive_int(price_tick, "price_tick") * self.tick_size

    def qty_to_steps_floor(self, qty: float) -> int:
        """Convert quantity to integer step count by flooring."""
        qty = _require_nonnegative_float(qty, "qty")
        return int(math.floor((qty / self.step_size) + 1e-12))

    def steps_to_qty(self, qty_steps: int) -> float:
        """Convert integer quantity steps to float quantity."""
        return _require_nonnegative_int(qty_steps, "qty_steps") * self.step_size

    def round_qty_down(self, qty: float) -> float:
        """Floor to step-size grid."""
        return self.steps_to_qty(self.qty_to_steps_floor(qty))

    def is_valid_qty(self, qty: float) -> bool:
        """Check step-rounded quantity within min/max."""
        try:
            qty = _require_nonnegative_float(qty, "qty")
        except ValueError:
            return False
        if qty < self.min_qty or qty > self.max_qty:
            return False
        rounded = self.round_qty_down(qty)
        return math.isclose(qty, rounded, rel_tol=0.0, abs_tol=max(1e-12, self.step_size * 1e-9))

    def notional(self, qty: float, price_tick: int) -> float:
        """qty * tick_to_price(price_tick) * contract_size."""
        qty = _require_nonnegative_float(qty, "qty")
        return qty * self.tick_to_price(price_tick) * self.contract_size

    def is_valid_notional(self, qty: float, price_tick: int) -> bool:
        """min_notional == 0 or notional >= min_notional."""
        try:
            value = self.notional(qty, price_tick)
        except ValueError:
            return False
        return self.min_notional == 0 or value >= self.min_notional


@dataclass(frozen=True, slots=True)
class L2Update:
    local_ts_us: int
    ts_us: int
    side: BookSide
    price_tick: int
    amount: float
    is_snapshot: bool
    source_row: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_positive_int(self.ts_us, "ts_us")
        object.__setattr__(self, "side", _coerce_enum(BookSide, self.side, "side"))
        _require_positive_int(self.price_tick, "price_tick")
        object.__setattr__(self, "amount", _require_nonnegative_float(self.amount, "amount"))
        _require_bool(self.is_snapshot, "is_snapshot")
        _require_nonnegative_int(self.source_row, "source_row")


@dataclass(frozen=True, slots=True)
class L2UpdateBatch:
    local_ts_us: int
    min_ts_us: int
    max_ts_us: int
    updates: tuple[L2Update, ...]
    is_snapshot_batch: bool
    batch_seq: int

    def __post_init__(self) -> None:
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_positive_int(self.min_ts_us, "min_ts_us")
        _require_positive_int(self.max_ts_us, "max_ts_us")
        if self.max_ts_us < self.min_ts_us:
            raise ValueError("max_ts_us must be >= min_ts_us")
        updates = tuple(self.updates)
        if not updates:
            raise ValueError("updates must be non-empty")
        for update in updates:
            if not isinstance(update, L2Update):
                raise ValueError("updates must contain L2Update values")
            if update.local_ts_us != self.local_ts_us:
                raise ValueError("all updates must share local_ts_us")
            if update.ts_us < self.min_ts_us or update.ts_us > self.max_ts_us:
                raise ValueError("update.ts_us must be within [min_ts_us, max_ts_us]")
        object.__setattr__(self, "updates", updates)
        _require_bool(self.is_snapshot_batch, "is_snapshot_batch")
        if self.is_snapshot_batch != any(update.is_snapshot for update in updates):
            raise ValueError("is_snapshot_batch must equal any(update.is_snapshot for update in updates)")
        _require_nonnegative_int(self.batch_seq, "batch_seq")


@dataclass(frozen=True, slots=True)
class TradePrint:
    local_ts_us: int
    ts_us: int
    side: AggressorSide
    price_tick: int
    amount: float
    trade_id: str = ""
    source_row: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_positive_int(self.ts_us, "ts_us")
        object.__setattr__(self, "side", _coerce_enum(AggressorSide, self.side, "side"))
        _require_positive_int(self.price_tick, "price_tick")
        object.__setattr__(self, "amount", _require_positive_float(self.amount, "amount"))
        if not isinstance(self.trade_id, str):
            raise ValueError("trade_id must be str")
        _require_nonnegative_int(self.source_row, "source_row")


@dataclass(frozen=True, slots=True)
class BookTop:
    local_ts_us: int
    best_bid_tick: int
    best_ask_tick: int
    best_bid_size: float
    best_ask_size: float

    def __post_init__(self) -> None:
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_positive_int(self.best_bid_tick, "best_bid_tick")
        _require_positive_int(self.best_ask_tick, "best_ask_tick")
        if self.best_bid_tick >= self.best_ask_tick:
            raise ValueError("best_bid_tick must be < best_ask_tick")
        object.__setattr__(self, "best_bid_size", _require_nonnegative_float(self.best_bid_size, "best_bid_size"))
        object.__setattr__(self, "best_ask_size", _require_nonnegative_float(self.best_ask_size, "best_ask_size"))

    @property
    def spread_ticks(self) -> int:
        return self.best_ask_tick - self.best_bid_tick

    @property
    def mid_tick_x2(self) -> int:
        return self.best_bid_tick + self.best_ask_tick


@dataclass(frozen=True, slots=True)
class BookLevelSnapshot:
    local_ts_us: int
    bid_ticks: tuple[int, ...]
    bid_sizes: tuple[float, ...]
    ask_ticks: tuple[int, ...]
    ask_sizes: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_positive_int(self.local_ts_us, "local_ts_us")
        bid_ticks = _tuple_of_positive_ints(self.bid_ticks, "bid_ticks")
        ask_ticks = _tuple_of_positive_ints(self.ask_ticks, "ask_ticks")
        bid_sizes = _tuple_of_nonnegative_floats(self.bid_sizes, "bid_sizes")
        ask_sizes = _tuple_of_nonnegative_floats(self.ask_sizes, "ask_sizes")
        if len(bid_ticks) != len(bid_sizes):
            raise ValueError("bid_ticks length must equal bid_sizes length")
        if len(ask_ticks) != len(ask_sizes):
            raise ValueError("ask_ticks length must equal ask_sizes length")
        for i in range(1, len(bid_ticks)):
            if bid_ticks[i - 1] <= bid_ticks[i]:
                raise ValueError("bid_ticks must be strictly descending")
        for i in range(1, len(ask_ticks)):
            if ask_ticks[i - 1] >= ask_ticks[i]:
                raise ValueError("ask_ticks must be strictly ascending")
        if bid_ticks and ask_ticks and bid_ticks[0] >= ask_ticks[0]:
            raise ValueError("best bid tick must be < best ask tick")
        object.__setattr__(self, "bid_ticks", bid_ticks)
        object.__setattr__(self, "bid_sizes", bid_sizes)
        object.__setattr__(self, "ask_ticks", ask_ticks)
        object.__setattr__(self, "ask_sizes", ask_sizes)


@dataclass(frozen=True, slots=True)
class ExecutionEventRef:
    event_seq: int
    local_ts_us: int
    event_type: ExecutionEventType
    book_ptr: int = -1
    trade_ptr: int = -1
    decision_ptr: int = -1

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.event_seq, "event_seq")
        _require_positive_int(self.local_ts_us, "local_ts_us")
        object.__setattr__(self, "event_type", _coerce_enum(ExecutionEventType, self.event_type, "event_type"))
        for name in ("book_ptr", "trade_ptr", "decision_ptr"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError(f"{name} must be >= -1")
        if self.event_type == ExecutionEventType.L2_BATCH:
            if self.book_ptr < 0 or self.trade_ptr != -1 or self.decision_ptr != -1:
                raise ValueError("L2_BATCH requires book_ptr >= 0 and trade_ptr == decision_ptr == -1")
        elif self.event_type == ExecutionEventType.TRADE:
            if self.trade_ptr < 0 or self.decision_ptr != -1:
                raise ValueError("TRADE requires trade_ptr >= 0 and decision_ptr == -1")
        elif self.event_type == ExecutionEventType.DECISION:
            if self.decision_ptr < 0 or self.book_ptr != -1 or self.trade_ptr != -1:
                raise ValueError("DECISION requires decision_ptr >= 0 and book_ptr == trade_ptr == -1")
        else:
            raise ValueError("unknown event_type")


@dataclass(frozen=True, slots=True)
class DecisionRef:
    decision_index: int
    local_ts_us: int
    event_seq_start: int
    event_seq_end_next: int
    book_ptr: int
    linear_pred_ptr: int = -1

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.decision_index, "decision_index")
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_nonnegative_int(self.event_seq_start, "event_seq_start")
        _require_nonnegative_int(self.event_seq_end_next, "event_seq_end_next")
        if self.event_seq_end_next < self.event_seq_start:
            raise ValueError("event_seq_end_next must be >= event_seq_start")
        _require_nonnegative_int(self.book_ptr, "book_ptr")
        if isinstance(self.linear_pred_ptr, bool) or not isinstance(self.linear_pred_ptr, int) or self.linear_pred_ptr < -1:
            raise ValueError("linear_pred_ptr must be >= -1")


_LINEAR_SIGNAL_CONSISTENCY_EPS = 1e-5


@dataclass(frozen=True, slots=True)
class LinearSignal:
    p_no_move: float
    p_move: float

    p_up_move: float
    p_down_move: float
    signed_move_prob: float

    expected_up_bps: float
    expected_down_bps: float
    expected_return_bps: float
    expected_abs_move_bps: float
    predicted_vol_bps: float

    confidence: float

    def __post_init__(self) -> None:
        for name in ("p_no_move", "p_move", "p_up_move", "p_down_move"):
            value = _require_finite_float(getattr(self, name), name)
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)

        signed_move_prob = _require_finite_float(self.signed_move_prob, "signed_move_prob")
        if signed_move_prob < -1.0 or signed_move_prob > 1.0:
            raise ValueError("signed_move_prob must be in [-1, 1]")
        object.__setattr__(self, "signed_move_prob", signed_move_prob)

        for name in (
            "expected_up_bps",
            "expected_down_bps",
            "expected_abs_move_bps",
            "predicted_vol_bps",
            "confidence",
        ):
            object.__setattr__(self, name, _require_nonnegative_float(getattr(self, name), name))
        object.__setattr__(self, "expected_return_bps", _require_finite_float(self.expected_return_bps, "expected_return_bps"))

        if abs((self.p_no_move + self.p_move) - 1.0) > _LINEAR_SIGNAL_CONSISTENCY_EPS:
            raise ValueError("p_no_move + p_move must be approximately 1")
        if abs((self.p_up_move + self.p_down_move) - self.p_move) > _LINEAR_SIGNAL_CONSISTENCY_EPS:
            raise ValueError("p_up_move + p_down_move must be approximately p_move")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    mode: ActionMode = ActionMode.CONTINUOUS_LATENT
    max_distance_ticks: int = 20
    max_order_qty: float = 0.01
    allow_bid: bool = True
    allow_ask: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _coerce_enum(ActionMode, self.mode, "mode"))
        _require_positive_int(self.max_distance_ticks, "max_distance_ticks")
        object.__setattr__(self, "max_order_qty", _require_positive_float(self.max_order_qty, "max_order_qty"))
        _require_bool(self.allow_bid, "allow_bid")
        _require_bool(self.allow_ask, "allow_ask")
        if not (self.allow_bid or self.allow_ask):
            raise ValueError("at least one of allow_bid or allow_ask must be true")


@dataclass(frozen=True, slots=True)
class QuoteIntent:
    bid_enabled: bool
    ask_enabled: bool
    bid_price_tick: int = 0
    ask_price_tick: int = 0
    bid_qty: float = 0.0
    ask_qty: float = 0.0

    def __post_init__(self) -> None:
        _require_bool(self.bid_enabled, "bid_enabled")
        _require_bool(self.ask_enabled, "ask_enabled")
        object.__setattr__(self, "bid_qty", _require_nonnegative_float(self.bid_qty, "bid_qty"))
        object.__setattr__(self, "ask_qty", _require_nonnegative_float(self.ask_qty, "ask_qty"))
        _require_nonnegative_int(self.bid_price_tick, "bid_price_tick")
        _require_nonnegative_int(self.ask_price_tick, "ask_price_tick")
        if self.bid_enabled:
            if self.bid_price_tick <= 0 or self.bid_qty <= 0:
                raise ValueError("enabled bid requires bid_price_tick > 0 and bid_qty > 0")
        elif self.bid_price_tick != 0 or self.bid_qty != 0:
            raise ValueError("disabled bid requires zero price and quantity")
        if self.ask_enabled:
            if self.ask_price_tick <= 0 or self.ask_qty <= 0:
                raise ValueError("enabled ask requires ask_price_tick > 0 and ask_qty > 0")
        elif self.ask_price_tick != 0 or self.ask_qty != 0:
            raise ValueError("disabled ask requires zero price and quantity")
        if self.bid_enabled and self.ask_enabled and self.bid_price_tick >= self.ask_price_tick:
            raise ValueError("bid_price_tick must be < ask_price_tick")


@dataclass(frozen=True, slots=True)
class ActiveOrder:
    order_id: int
    side: OrderSide
    price_tick: int
    qty: float
    remaining_qty: float
    queue_ahead_qty: float
    status: OrderStatus
    created_local_ts_us: int
    last_update_local_ts_us: int
    created_event_seq: int
    last_update_event_seq: int
    effective_local_ts_us: int = 0
    cancel_requested_local_ts_us: int = 0
    cancel_effective_local_ts_us: int = 0
    effective_event_seq: int = -1
    cancel_requested_event_seq: int = -1
    cancel_effective_event_seq: int = -1

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.order_id, "order_id")
        object.__setattr__(self, "side", _coerce_enum(OrderSide, self.side, "side"))
        _require_positive_int(self.price_tick, "price_tick")
        object.__setattr__(self, "qty", _require_positive_float(self.qty, "qty"))
        object.__setattr__(self, "remaining_qty", _require_nonnegative_float(self.remaining_qty, "remaining_qty"))
        if self.remaining_qty > self.qty:
            raise ValueError("remaining_qty must be <= qty")
        object.__setattr__(self, "queue_ahead_qty", _require_nonnegative_float(self.queue_ahead_qty, "queue_ahead_qty"))
        object.__setattr__(self, "status", _coerce_enum(OrderStatus, self.status, "status"))
        _require_positive_int(self.created_local_ts_us, "created_local_ts_us")
        _require_positive_int(self.last_update_local_ts_us, "last_update_local_ts_us")
        _require_nonnegative_int(self.created_event_seq, "created_event_seq")
        _require_nonnegative_int(self.last_update_event_seq, "last_update_event_seq")
        _require_nonnegative_int(self.effective_local_ts_us, "effective_local_ts_us")
        _require_nonnegative_int(self.cancel_requested_local_ts_us, "cancel_requested_local_ts_us")
        _require_nonnegative_int(self.cancel_effective_local_ts_us, "cancel_effective_local_ts_us")
        for value, name in ((self.effective_event_seq, "effective_event_seq"), (self.cancel_requested_event_seq, "cancel_requested_event_seq"), (self.cancel_effective_event_seq, "cancel_effective_event_seq")):
            if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                raise ValueError(f"{name} must be >= -1")
        if (self.effective_local_ts_us == 0) != (self.effective_event_seq == -1):
            raise ValueError("effective timestamp/key sentinels must agree")
        if (self.cancel_requested_local_ts_us == 0) != (self.cancel_requested_event_seq == -1):
            raise ValueError("cancel requested timestamp/key sentinels must agree")
        if (self.cancel_effective_local_ts_us == 0) != (self.cancel_effective_event_seq == -1):
            raise ValueError("cancel effective timestamp/key sentinels must agree")

        created_key = self.created_key
        if self.last_update_key < created_key:
            raise ValueError("last_update_key must be >= created_key")
        if self.effective_local_ts_us and self.effective_key < created_key:
            raise ValueError("effective_key must be >= created_key")
        if self.cancel_effective_local_ts_us and not self.cancel_requested_local_ts_us:
            raise ValueError("cancel_effective_key requires cancel_requested_key")
        if self.cancel_requested_local_ts_us:
            if self.cancel_requested_key is None or self.cancel_requested_key < created_key:
                raise ValueError("cancel_requested_key must be >= created_key")
            if self.cancel_effective_key is None or self.cancel_effective_key < self.cancel_requested_key:
                raise ValueError("cancel_effective_key must be >= cancel_requested_key")

        if self.status == OrderStatus.PENDING_NEW and self.remaining_qty != self.qty:
            raise ValueError("PENDING_NEW order requires remaining_qty == qty")
        if self.status == OrderStatus.FILLED and self.remaining_qty != 0.0:
            raise ValueError("FILLED order requires remaining_qty == 0")
        if self.status == OrderStatus.PARTIALLY_FILLED:
            if not (0.0 < self.remaining_qty < self.qty):
                raise ValueError("PARTIALLY_FILLED order requires 0 < remaining_qty < qty")
            if self.cancel_requested_key is not None or self.cancel_effective_key is not None:
                raise ValueError("PARTIALLY_FILLED order cannot have pending cancel")
        if self.status == OrderStatus.ACTIVE:
            if self.remaining_qty <= 0.0:
                raise ValueError("ACTIVE order requires remaining_qty > 0")
            if self.cancel_requested_key is not None or self.cancel_effective_key is not None:
                raise ValueError("ACTIVE order cannot have pending cancel")
        if self.status == OrderStatus.PENDING_CANCEL:
            if self.remaining_qty <= 0.0:
                raise ValueError("PENDING_CANCEL order requires remaining_qty > 0")
            if self.cancel_requested_key is None or self.cancel_effective_key is None:
                raise ValueError("PENDING_CANCEL order requires cancel keys")
        if self.status == OrderStatus.REJECTED and self.remaining_qty != self.qty:
            raise ValueError("REJECTED order requires remaining_qty == qty")

    @property
    def filled_qty(self) -> float:
        return self.qty - self.remaining_qty

    @property
    def created_key(self) -> EventKey:
        return EventKey(self.created_local_ts_us, self.created_event_seq)

    @property
    def last_update_key(self) -> EventKey:
        return EventKey(self.last_update_local_ts_us, self.last_update_event_seq)

    @property
    def effective_key(self) -> EventKey:
        if self.effective_local_ts_us:
            return EventKey(self.effective_local_ts_us, self.effective_event_seq)
        return self.created_key

    @property
    def cancel_requested_key(self) -> EventKey | None:
        if not self.cancel_requested_local_ts_us:
            return None
        return EventKey(self.cancel_requested_local_ts_us, self.cancel_requested_event_seq)

    @property
    def cancel_effective_key(self) -> EventKey | None:
        if not self.cancel_effective_local_ts_us:
            return None
        return EventKey(self.cancel_effective_local_ts_us, self.cancel_effective_event_seq)

    @property
    def is_live(self) -> bool:
        return self.status in (OrderStatus.PENDING_NEW, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL)

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    def is_fillable_at_key(self, key: EventKey) -> bool:
        if not isinstance(key, EventKey):
            raise ValueError("key must be EventKey")
        if self.status not in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL):
            return False
        if key < self.effective_key:
            return False
        cancel_key = self.cancel_effective_key
        if cancel_key is not None and key >= cancel_key:
            return False
        return True


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: int
    side: OrderSide
    local_ts_us: int
    event_seq: int
    price_tick: int
    qty: float
    fee: float
    reason: FillReason
    queue_ahead_before: float = 0.0
    queue_ahead_after: float = 0.0

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.order_id, "order_id")
        object.__setattr__(self, "side", _coerce_enum(OrderSide, self.side, "side"))
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_nonnegative_int(self.event_seq, "event_seq")
        EventKey(self.local_ts_us, self.event_seq)
        _require_positive_int(self.price_tick, "price_tick")
        object.__setattr__(self, "qty", _require_positive_float(self.qty, "qty"))
        object.__setattr__(self, "fee", _require_finite_float(self.fee, "fee"))
        object.__setattr__(self, "reason", _coerce_enum(FillReason, self.reason, "reason"))
        object.__setattr__(self, "queue_ahead_before", _require_nonnegative_float(self.queue_ahead_before, "queue_ahead_before"))
        object.__setattr__(self, "queue_ahead_after", _require_nonnegative_float(self.queue_ahead_after, "queue_ahead_after"))


@dataclass(frozen=True, slots=True)
class PositionState:
    cash: float = 0.0
    inventory_qty: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cash", "inventory_qty", "realized_pnl", "fees_paid"):
            object.__setattr__(self, name, _require_finite_float(getattr(self, name), name))

    def mark_to_market(self, mid_price: float, contract_size: float = 1.0) -> float:
        """Return cash + inventory_qty * mid_price * contract_size."""
        mid_price = _require_positive_float(mid_price, "mid_price")
        contract_size = _require_positive_float(contract_size, "contract_size")
        return self.cash + self.inventory_qty * mid_price * contract_size


@dataclass(frozen=True, slots=True)
class RewardComponents:
    raw_equity_delta: float
    inventory_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    turnover_penalty: float = 0.0
    cancel_penalty: float = 0.0
    terminal_penalty: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_equity_delta", _require_finite_float(self.raw_equity_delta, "raw_equity_delta"))
        for name in ("inventory_penalty", "drawdown_penalty", "turnover_penalty", "cancel_penalty", "terminal_penalty"):
            object.__setattr__(self, name, _require_nonnegative_float(getattr(self, name), name))

    @property
    def total_reward(self) -> float:
        return (
            self.raw_equity_delta
            - self.inventory_penalty
            - self.drawdown_penalty
            - self.turnover_penalty
            - self.cancel_penalty
            - self.terminal_penalty
        )


@dataclass(frozen=True, slots=True)
class ExecutionStepResult:
    reward: RewardComponents
    position: PositionState
    fills: tuple[Fill, ...]
    done: bool
    truncated: bool
    info: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reward, RewardComponents):
            raise ValueError("reward must be RewardComponents")
        if not isinstance(self.position, PositionState):
            raise ValueError("position must be PositionState")
        fills = tuple(self.fills)
        if not all(isinstance(fill, Fill) for fill in fills):
            raise ValueError("fills must contain Fill values")
        object.__setattr__(self, "fills", fills)
        _require_bool(self.done, "done")
        _require_bool(self.truncated, "truncated")
        if self.info is not None:
            if not isinstance(self.info, dict):
                raise ValueError("info must be None or dict[str, object]")
            clean: dict[str, object] = {}
            for key, value in self.info.items():
                if not isinstance(key, str):
                    raise ValueError("info keys must be str")
                clean[key] = _clean_info_value(value, f"info[{key!r}]")
            object.__setattr__(self, "info", clean)


@dataclass(frozen=True, slots=True)
class ExecutionTapeManifest:
    schema: str
    tape_format: ExecutionTapeFormat
    exchange: str
    symbol: str
    symbol_spec: SymbolSpec
    symbol_rules: ExchangeSymbolRules
    source_data_types: tuple[TardisDataType, ...]
    array_names: tuple[str, ...]
    num_events: int
    num_l2_batches: int
    num_trades: int
    num_decisions: int
    start_local_ts_us: int
    end_local_ts_us: int
    symbol_rule_compatibility: RuleCompatibilityReport | None = None
    created_at_utc: str = ""
    notes: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.schema, "schema")
        object.__setattr__(self, "tape_format", _coerce_enum(ExecutionTapeFormat, self.tape_format, "tape_format"))
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")
        if not isinstance(self.symbol_spec, SymbolSpec):
            raise ValueError("symbol_spec must be SymbolSpec")
        if self.symbol_spec.exchange != self.exchange or self.symbol_spec.symbol != self.symbol:
            raise ValueError("symbol_spec exchange/symbol must match manifest")
        if not isinstance(self.symbol_rules, ExchangeSymbolRules):
            raise ValueError("symbol_rules must be ExchangeSymbolRules")
        if self.symbol_rules.exchange != self.exchange or self.symbol_rules.symbol != self.symbol:
            raise ValueError("symbol_rules exchange/symbol must match manifest")
        expected_spec = self.symbol_rules.to_symbol_spec()
        if expected_spec.exchange != self.symbol_spec.exchange or expected_spec.symbol != self.symbol_spec.symbol:
            raise ValueError("symbol_spec must match symbol_rules")
        for field in ("tick_size", "step_size", "min_qty", "max_qty", "min_notional", "contract_size"):
            if not math.isclose(getattr(expected_spec, field), getattr(self.symbol_spec, field), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("symbol_spec must equal symbol_rules.to_symbol_spec()")
        if self.symbol_rule_compatibility is not None and not isinstance(self.symbol_rule_compatibility, RuleCompatibilityReport):
            raise ValueError("symbol_rule_compatibility must be None or RuleCompatibilityReport")
        source_data_types = tuple(_coerce_enum(TardisDataType, value, f"source_data_types[{i}]") for i, value in enumerate(self.source_data_types))
        if not source_data_types:
            raise ValueError("source_data_types must be non-empty")
        if len(set(source_data_types)) != len(source_data_types):
            raise ValueError("source_data_types must not contain duplicates")
        if TardisDataType.INCREMENTAL_BOOK_L2 not in source_data_types:
            raise ValueError("source_data_types must include INCREMENTAL_BOOK_L2")
        if TardisDataType.TRADES not in source_data_types:
            raise ValueError("source_data_types must include TRADES")
        object.__setattr__(self, "source_data_types", source_data_types)
        array_names = _tuple_of_nonempty_str(self.array_names, "array_names")
        if not array_names:
            raise ValueError("array_names must be non-empty")
        if len(set(array_names)) != len(array_names):
            raise ValueError("array_names must not contain duplicates")
        object.__setattr__(self, "array_names", array_names)
        for name in ("num_events", "num_l2_batches", "num_trades", "num_decisions"):
            _require_nonnegative_int(getattr(self, name), name)
        if self.num_events < self.num_l2_batches + self.num_trades:
            raise ValueError("num_events must be >= num_l2_batches + num_trades")
        _require_positive_int(self.start_local_ts_us, "start_local_ts_us")
        _require_positive_int(self.end_local_ts_us, "end_local_ts_us")
        if self.end_local_ts_us <= self.start_local_ts_us:
            raise ValueError("end_local_ts_us must be > start_local_ts_us")
        if not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be str")
        if self.notes is not None:
            if not isinstance(self.notes, dict):
                raise ValueError("notes must be None or dict[str, str]")
            clean: dict[str, str] = {}
            for key, value in self.notes.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("notes must be dict[str, str]")
                clean[key] = value
            object.__setattr__(self, "notes", clean)


__all__ = [
    "ExecutionTapeFormat",
    "ExecutionEventType",
    "OrderSide",
    "QuoteSide",
    "OrderStatus",
    "QueueModelMode",
    "FillReason",
    "RewardMode",
    "ActionMode",
    "LatencyConfig",
    "SymbolSpec",
    "L2Update",
    "L2UpdateBatch",
    "TradePrint",
    "BookTop",
    "BookLevelSnapshot",
    "ExecutionEventRef",
    "DecisionRef",
    "LinearSignal",
    "ActionSpec",
    "QuoteIntent",
    "ActiveOrder",
    "Fill",
    "PositionState",
    "RewardComponents",
    "ExecutionStepResult",
    "ExecutionTapeManifest",
]
