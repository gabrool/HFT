"""Stable typed contracts for the current MMRT pipeline.

This module is intentionally IO-free and dependency-light. It defines only the
shared enums, validation helpers, and immutable dataclasses used by the active
microsecond-native supervised storage ingest and execution tape paths.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class _StrEnum(str, Enum):
    """String-valued enum base."""


class TimeUnit(_StrEnum):
    MICROSECOND = "us"


class TardisDataType(_StrEnum):
    INCREMENTAL_BOOK_L2 = "incremental_book_L2"
    BOOK_SNAPSHOT_25 = "book_snapshot_25"
    TRADES = "trades"


class BookSide(_StrEnum):
    BID = "bid"
    ASK = "ask"


class AggressorSide(_StrEnum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class DecisionReason(_StrEnum):
    BOOK_STRIDE = "book_stride"


class PriceReference(_StrEnum):
    MID = "mid"


class AsOfPolicy(_StrEnum):
    LAST_OBSERVATION = "last_observation"


class SplitRole(_StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class StorageFormat(_StrEnum):
    FLAT_DECISION_ROWS_US = "flat_decision_rows_us"


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int_us(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


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
            if isinstance(v, bool) or not isinstance(v, int):
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
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
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
class LabelSpec:
    horizons_us: tuple[int, ...]
    entry_delay_us: int
    price_reference: PriceReference = PriceReference.MID
    asof_policy: AsOfPolicy = AsOfPolicy.LAST_OBSERVATION

    def __post_init__(self) -> None:
        horizons = _tuple_of_ints(self.horizons_us, "horizons_us", positive=True)
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons_us must not contain duplicates")
        object.__setattr__(self, "horizons_us", tuple(sorted(horizons)))
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
        vals = _tuple_of_floats(self.values_bps, "values_bps")
        object.__setattr__(self, "horizons_us", horizons)
        object.__setattr__(self, "values_bps", vals)
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        if tuple(sorted(horizons)) != horizons:
            raise ValueError("horizons_us must be sorted ascending")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons_us must not contain duplicates")
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


__all__ = [
    "TimeUnit",
    "TardisDataType",
    "BookSide",
    "AggressorSide",
    "DecisionReason",
    "PriceReference",
    "AsOfPolicy",
    "SplitRole",
    "StorageFormat",
    "LabelSpec",
    "LabelResult",
    "TimeRangeUS",
]
