"""Stable typed contracts for the microsecond-native Tardis/Binance linear market-making
pipeline.

This module is intentionally IO-free and dependency-light. It defines shared enums,
validation helpers, and immutable dataclasses used across ingestion, feature building,
labeling, and dataset metadata boundaries.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Union


class _StrEnum(str, Enum):
    """Compatibility base for string enums on Python versions without StrEnum."""


class TimeUnit(_StrEnum):
    MICROSECOND = "us"


class TardisDataType(_StrEnum):
    INCREMENTAL_BOOK_L2 = "incremental_book_L2"
    BOOK_SNAPSHOT_25 = "book_snapshot_25"
    BOOK_SNAPSHOT_5 = "book_snapshot_5"
    TRADES = "trades"
    OPTIONS_CHAIN = "options_chain"
    QUOTES = "quotes"
    BOOK_TICKER = "book_ticker"
    DERIVATIVE_TICKER = "derivative_ticker"
    LIQUIDATIONS = "liquidations"


class EventType(_StrEnum):
    BOOK_SNAPSHOT = "book_snapshot"
    BOOK_DELTA = "book_delta"
    TRADE = "trade"
    BOOK_TICKER = "book_ticker"
    DERIVATIVE_TICKER = "derivative_ticker"
    LIQUIDATION = "liquidation"


class BookSide(_StrEnum):
    BID = "bid"
    ASK = "ask"


class AggressorSide(_StrEnum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class DecisionReason(_StrEnum):
    BOOK_EVENT = "book_event"
    SCHEDULED_TIME = "scheduled_time"


class PriceReference(_StrEnum):
    MID = "mid"
    MICROPRICE = "microprice"
    MARK = "mark"


class AsOfPolicy(_StrEnum):
    LAST_OBSERVATION = "last_observation"


class SplitRole(_StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class StorageFormat(_StrEnum):
    FLAT_DECISION_ROWS_US_V1 = "flat_decision_rows_us_v1"


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int_us(value: int, name: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_positive_float(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite float > 0")
    return float(value)


def _require_nonnegative_float(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite float >= 0")
    return float(value)


def _tuple_of_ints(values: Any, name: str, *, positive: bool = True) -> tuple[int, ...]:
    if isinstance(values, tuple):
        seq = values
    else:
        seq = tuple(values)
    out: list[int] = []
    for idx, v in enumerate(seq):
        if positive:
            out.append(_require_int_us(v, f"{name}[{idx}]"))
        else:
            if not isinstance(v, int):
                raise ValueError(f"{name}[{idx}] must be int")
            out.append(v)
    return tuple(out)


def _tuple_of_floats(values: Any, name: str) -> tuple[float, ...]:
    if isinstance(values, tuple):
        seq = values
    else:
        seq = tuple(values)
    out: list[float] = []
    for idx, v in enumerate(seq):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}[{idx}] must be finite float")
        out.append(float(v))
    return tuple(out)


def _coerce_enum(enum_cls: type[Enum], value: Any, name: str):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ValueError(f"{name} has invalid value {value!r}") from exc
    raise ValueError(f"{name} must be {enum_cls.__name__} or str")


@dataclass(frozen=True, slots=True)
class EventMeta:
    exchange: str
    symbol: str
    ts_us: int
    local_ts_us: int
    source_data_type: TardisDataType
    source_row: int
    source_file: str = ""
    sequence_id: str = ""

    def __post_init__(self) -> None:
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")
        _require_int_us(self.ts_us, "ts_us")
        _require_int_us(self.local_ts_us, "local_ts_us")
        _require_nonnegative_int(self.source_row, "source_row")
        object.__setattr__(self, "source_data_type", _coerce_enum(TardisDataType, self.source_data_type, "source_data_type"))
        if not isinstance(self.source_file, str):
            raise ValueError("source_file must be str")
        if not isinstance(self.sequence_id, str):
            raise ValueError("sequence_id must be str")

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (self.local_ts_us, self.ts_us, self.source_row)


@dataclass(frozen=True, slots=True)
class PriceLevel:
    price: float
    amount: float

    def __post_init__(self) -> None:
        _require_positive_float(self.price, "price")
        _require_nonnegative_float(self.amount, "amount")


@dataclass(frozen=True, slots=True)
class BookSnapshotEvent:
    meta: EventMeta
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]
    depth_limit: int = 25

    def __post_init__(self) -> None:
        if self.meta.source_data_type not in (TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.BOOK_SNAPSHOT_5):
            raise ValueError("meta.source_data_type must be BOOK_SNAPSHOT_25 or BOOK_SNAPSHOT_5")
        if self.depth_limit not in (5, 25):
            raise ValueError("depth_limit must be 5 or 25")
        if len(self.bids) > self.depth_limit or len(self.asks) > self.depth_limit:
            raise ValueError("bids/asks exceed depth_limit")
        for lvl in self.bids:
            if lvl.amount <= 0:
                raise ValueError("snapshot bid amount must be > 0")
        for lvl in self.asks:
            if lvl.amount <= 0:
                raise ValueError("snapshot ask amount must be > 0")
        for i in range(1, len(self.bids)):
            if self.bids[i - 1].price <= self.bids[i].price:
                raise ValueError("bids must be strictly descending by price")
        for i in range(1, len(self.asks)):
            if self.asks[i - 1].price >= self.asks[i].price:
                raise ValueError("asks must be strictly ascending by price")
        if self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            raise ValueError("best_bid must be < best_ask")

    @property
    def event_type(self) -> EventType:
        return EventType.BOOK_SNAPSHOT

    @property
    def best_bid(self) -> PriceLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> PriceLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if not (self.best_bid and self.best_ask):
            return None
        return (self.best_bid.price + self.best_ask.price) / 2.0

    @property
    def spread(self) -> float | None:
        if not (self.best_bid and self.best_ask):
            return None
        return self.best_ask.price - self.best_bid.price


@dataclass(frozen=True, slots=True)
class BookDeltaEvent:
    """Incremental L2 delta: amount is absolute at level; amount=0 removes the level.

    This contract does not apply deltas or group updates.
    """

    meta: EventMeta
    side: BookSide
    price: float
    amount: float
    is_snapshot: bool

    def __post_init__(self) -> None:
        if self.meta.source_data_type != TardisDataType.INCREMENTAL_BOOK_L2:
            raise ValueError("meta.source_data_type must be INCREMENTAL_BOOK_L2")
        object.__setattr__(self, "side", _coerce_enum(BookSide, self.side, "side"))
        _require_positive_float(self.price, "price")
        _require_nonnegative_float(self.amount, "amount")

    @property
    def event_type(self) -> EventType:
        return EventType.BOOK_DELTA


@dataclass(frozen=True, slots=True)
class TradeEvent:
    meta: EventMeta
    trade_id: str
    side: AggressorSide
    price: float
    amount: float

    def __post_init__(self) -> None:
        if self.meta.source_data_type != TardisDataType.TRADES:
            raise ValueError("meta.source_data_type must be TRADES")
        if not isinstance(self.trade_id, str):
            raise ValueError("trade_id must be str")
        object.__setattr__(self, "side", _coerce_enum(AggressorSide, self.side, "side"))
        _require_positive_float(self.price, "price")
        _require_positive_float(self.amount, "amount")

    @property
    def event_type(self) -> EventType:
        return EventType.TRADE


@dataclass(frozen=True, slots=True)
class BookTickerEvent:
    meta: EventMeta
    bid_price: float
    bid_amount: float
    ask_price: float
    ask_amount: float

    def __post_init__(self) -> None:
        if self.meta.source_data_type != TardisDataType.BOOK_TICKER:
            raise ValueError("meta.source_data_type must be BOOK_TICKER")
        _require_positive_float(self.bid_price, "bid_price")
        _require_nonnegative_float(self.bid_amount, "bid_amount")
        _require_positive_float(self.ask_price, "ask_price")
        _require_nonnegative_float(self.ask_amount, "ask_amount")
        if self.bid_price >= self.ask_price:
            raise ValueError("bid_price must be < ask_price")

    @property
    def event_type(self) -> EventType:
        return EventType.BOOK_TICKER


@dataclass(frozen=True, slots=True)
class DerivativeTickerEvent:
    meta: EventMeta
    funding_timestamp_us: int | None = None
    funding_rate: float | None = None
    predicted_funding_rate: float | None = None
    open_interest: float | None = None
    last_price: float | None = None
    index_price: float | None = None
    mark_price: float | None = None

    def __post_init__(self) -> None:
        if self.meta.source_data_type != TardisDataType.DERIVATIVE_TICKER:
            raise ValueError("meta.source_data_type must be DERIVATIVE_TICKER")
        if self.funding_timestamp_us is not None:
            _require_int_us(self.funding_timestamp_us, "funding_timestamp_us")
        if self.open_interest is not None:
            _require_nonnegative_float(self.open_interest, "open_interest")
        for fld in ("last_price", "index_price", "mark_price"):
            value = getattr(self, fld)
            if value is not None:
                _require_positive_float(value, fld)
        for fld in ("funding_rate", "predicted_funding_rate"):
            value = getattr(self, fld)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{fld} must be finite if present")

    @property
    def event_type(self) -> EventType:
        return EventType.DERIVATIVE_TICKER


@dataclass(frozen=True, slots=True)
class LiquidationEvent:
    """Tardis liquidation side values are buy/sell/unknown but semantics differ from trades."""

    meta: EventMeta
    liquidation_id: str
    side: AggressorSide
    price: float
    amount: float

    def __post_init__(self) -> None:
        if self.meta.source_data_type != TardisDataType.LIQUIDATIONS:
            raise ValueError("meta.source_data_type must be LIQUIDATIONS")
        if not isinstance(self.liquidation_id, str):
            raise ValueError("liquidation_id must be str")
        object.__setattr__(self, "side", _coerce_enum(AggressorSide, self.side, "side"))
        _require_positive_float(self.price, "price")
        _require_positive_float(self.amount, "amount")

    @property
    def event_type(self) -> EventType:
        return EventType.LIQUIDATION


MarketEvent = Union[BookSnapshotEvent, BookDeltaEvent, TradeEvent, BookTickerEvent, DerivativeTickerEvent, LiquidationEvent]


def event_type(event: MarketEvent) -> EventType:
    return event.event_type


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    meta: EventMeta
    event_type: EventType
    is_decision: bool
    decision_reason: DecisionReason | None
    features: tuple[float, ...]
    raw_mid: float | None
    dt_us: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", _coerce_enum(EventType, self.event_type, "event_type"))
        expected_event_type = {
            TardisDataType.BOOK_SNAPSHOT_25: EventType.BOOK_SNAPSHOT,
            TardisDataType.BOOK_SNAPSHOT_5: EventType.BOOK_SNAPSHOT,
            TardisDataType.INCREMENTAL_BOOK_L2: EventType.BOOK_DELTA,
            TardisDataType.TRADES: EventType.TRADE,
            TardisDataType.BOOK_TICKER: EventType.BOOK_TICKER,
            TardisDataType.DERIVATIVE_TICKER: EventType.DERIVATIVE_TICKER,
            TardisDataType.LIQUIDATIONS: EventType.LIQUIDATION,
        }.get(self.meta.source_data_type)
        if expected_event_type is not None and self.event_type != expected_event_type:
            raise ValueError("event_type must match originating event type")
        _require_int_us(self.dt_us, "dt_us", allow_zero=True)
        feats = _tuple_of_floats(self.features, "features")
        object.__setattr__(self, "features", feats)
        if self.is_decision:
            if self.decision_reason is None:
                raise ValueError("decision_reason is required when is_decision is True")
            object.__setattr__(self, "decision_reason", _coerce_enum(DecisionReason, self.decision_reason, "decision_reason"))
            if not feats:
                raise ValueError("features must be non-empty when is_decision is True")
            if self.raw_mid is None or _require_positive_float(self.raw_mid, "raw_mid") <= 0:
                raise ValueError("raw_mid must be > 0 when is_decision is True")
        else:
            if self.decision_reason is not None:
                raise ValueError("decision_reason must be None when is_decision is False")
            if feats:
                raise ValueError("features must be empty when is_decision is False")
            if self.raw_mid is not None:
                _require_positive_float(self.raw_mid, "raw_mid")


@dataclass(frozen=True, slots=True)
class DecisionRowRef:
    segment_key: str
    row_idx: int
    ts_us: int
    local_ts_us: int

    def __post_init__(self) -> None:
        _require_nonempty_str(self.segment_key, "segment_key")
        _require_nonnegative_int(self.row_idx, "row_idx")
        _require_int_us(self.ts_us, "ts_us")
        _require_int_us(self.local_ts_us, "local_ts_us")


@dataclass(frozen=True, slots=True)
class LabelSpec:
    horizons_us: tuple[int, ...]
    entry_delay_us: int
    price_reference: PriceReference = PriceReference.MID
    asof_policy: AsOfPolicy = AsOfPolicy.LAST_OBSERVATION

    def __post_init__(self) -> None:
        horizons = sorted(set(_tuple_of_ints(self.horizons_us, "horizons_us", positive=True)))
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        object.__setattr__(self, "horizons_us", tuple(horizons))
        _require_int_us(self.entry_delay_us, "entry_delay_us", allow_zero=True)
        object.__setattr__(self, "price_reference", _coerce_enum(PriceReference, self.price_reference, "price_reference"))
        object.__setattr__(self, "asof_policy", _coerce_enum(AsOfPolicy, self.asof_policy, "asof_policy"))

    @property
    def label_context_us(self) -> int:
        return self.entry_delay_us + max(self.horizons_us)


@dataclass(frozen=True, slots=True)
class LabelResult:
    decision_ts_us: int
    entry_ts_us: int
    horizons_us: tuple[int, ...]
    values_bps: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_int_us(self.decision_ts_us, "decision_ts_us")
        _require_int_us(self.entry_ts_us, "entry_ts_us")
        if self.entry_ts_us < self.decision_ts_us:
            raise ValueError("entry_ts_us must be >= decision_ts_us")
        horizons = _tuple_of_ints(self.horizons_us, "horizons_us", positive=True)
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        if tuple(sorted(horizons)) != horizons:
            raise ValueError("horizons_us must be sorted ascending")
        vals = _tuple_of_floats(self.values_bps, "values_bps")
        if len(vals) != len(horizons):
            raise ValueError("values_bps length must match horizons_us")


@dataclass(frozen=True, slots=True)
class TimeRangeUS:
    start_us: int
    end_us: int

    def __post_init__(self) -> None:
        _require_int_us(self.start_us, "start_us")
        _require_int_us(self.end_us, "end_us")
        if self.end_us <= self.start_us:
            raise ValueError("end_us must be > start_us")

    def contains(self, ts_us: int) -> bool:
        _require_int_us(ts_us, "ts_us")
        return self.start_us <= ts_us < self.end_us


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    segment_key: str
    time_range: TimeRangeUS
    source_files: tuple[str, ...]
    row_count: int
    label_count: int

    def __post_init__(self) -> None:
        _require_nonempty_str(self.segment_key, "segment_key")
        if not isinstance(self.source_files, tuple):
            object.__setattr__(self, "source_files", tuple(self.source_files))
        for i, p in enumerate(self.source_files):
            if not isinstance(p, str):
                raise ValueError(f"source_files[{i}] must be str")
        _require_nonnegative_int(self.row_count, "row_count")
        _require_nonnegative_int(self.label_count, "label_count")
        if self.label_count > self.row_count:
            raise ValueError("label_count must be <= row_count")


@dataclass(frozen=True, slots=True)
class SplitEntry:
    role: SplitRole
    segment_key: str
    start_row: int
    end_row: int
    time_range: TimeRangeUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _coerce_enum(SplitRole, self.role, "role"))
        _require_nonempty_str(self.segment_key, "segment_key")
        _require_nonnegative_int(self.start_row, "start_row")
        _require_nonnegative_int(self.end_row, "end_row")
        if self.end_row < self.start_row:
            raise ValueError("end_row must be >= start_row")


@dataclass(frozen=True, slots=True)
class SplitPlan:
    entries: tuple[SplitEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries:
            raise ValueError("entries must be non-empty")
        roles = {e.role for e in self.entries}
        if SplitRole.TRAIN not in roles or SplitRole.VAL not in roles:
            raise ValueError("entries must contain TRAIN and VAL roles")

    def entries_for(self, role: SplitRole) -> tuple[SplitEntry, ...]:
        role = _coerce_enum(SplitRole, role, "role")
        return tuple(e for e in self.entries if e.role == role)

    def roles_present(self) -> tuple[SplitRole, ...]:
        seen: list[SplitRole] = []
        for e in self.entries:
            if e.role not in seen:
                seen.append(e.role)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    storage_format: StorageFormat
    exchange: str
    symbol: str
    time_unit: TimeUnit
    source_data_types: tuple[TardisDataType, ...]
    label_spec: LabelSpec
    lookback_rows: int
    feature_schema_version: str
    feature_names_hash: str
    feature_dim_core: int
    aux_dim: int
    feature_dim_total: int
    segments: tuple[SegmentSpec, ...]
    split_plan: SplitPlan | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.schema_version, "schema_version")
        object.__setattr__(self, "storage_format", _coerce_enum(StorageFormat, self.storage_format, "storage_format"))
        if self.storage_format != StorageFormat.FLAT_DECISION_ROWS_US_V1:
            raise ValueError("storage_format must be FLAT_DECISION_ROWS_US_V1")
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")
        object.__setattr__(self, "time_unit", _coerce_enum(TimeUnit, self.time_unit, "time_unit"))
        if self.time_unit != TimeUnit.MICROSECOND:
            raise ValueError("time_unit must be MICROSECOND")
        if not isinstance(self.source_data_types, tuple):
            object.__setattr__(self, "source_data_types", tuple(self.source_data_types))
        if not self.source_data_types:
            raise ValueError("source_data_types must be non-empty")
        object.__setattr__(self, "source_data_types", tuple(_coerce_enum(TardisDataType, d, "source_data_types") for d in self.source_data_types))
        _require_int_us(self.lookback_rows, "lookback_rows")
        _require_nonempty_str(self.feature_schema_version, "feature_schema_version")
        _require_nonempty_str(self.feature_names_hash, "feature_names_hash")
        _require_int_us(self.feature_dim_core, "feature_dim_core")
        _require_nonnegative_int(self.aux_dim, "aux_dim")
        _require_int_us(self.feature_dim_total, "feature_dim_total")
        if self.feature_dim_total != self.feature_dim_core + self.aux_dim:
            raise ValueError("feature_dim_total must equal feature_dim_core + aux_dim")
        if not isinstance(self.segments, tuple):
            object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError("segments must be non-empty")

    @property
    def total_rows(self) -> int:
        return sum(seg.row_count for seg in self.segments)

    @property
    def total_labels(self) -> int:
        return sum(seg.label_count for seg in self.segments)
