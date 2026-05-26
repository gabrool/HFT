"""Streaming L2 order book reconstruction for normalized Tardis incremental_book_L2 rows.

This module applies Tardis incremental_book_L2 snapshot/update semantics to caller-provided
rows. It does not read files, sort rows, run quality checks, merge event streams, compute
features, compute labels, or make trading decisions.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Iterator, Mapping

from mmrt.contracts import BookSide

TS_US = "ts_us"
LOCAL_TS_US = "local_ts_us"
RAW_SOURCE_ROW = "raw_source_row"

INCREMENTAL_IS_SNAPSHOT = "is_snapshot"
INCREMENTAL_SIDE = "side"
INCREMENTAL_PRICE = "price"
INCREMENTAL_AMOUNT = "amount"

BOOK_SIDE_BID_TEXT = "bid"
BOOK_SIDE_ASK_TEXT = "ask"


class ReconstructedBookStatus(str, Enum):
    WAITING_FOR_SNAPSHOT = "waiting_for_snapshot"
    READY = "ready"


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite float")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return parsed


def _require_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite float")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a non-negative finite float")
    return parsed


def _coerce_book_side(value: Any) -> BookSide:
    if value is BookSide.BID:
        return BookSide.BID
    if value is BookSide.ASK:
        return BookSide.ASK
    if isinstance(value, str):
        text = value.strip().lower()
        if text == BOOK_SIDE_BID_TEXT:
            return BookSide.BID
        if text == BOOK_SIDE_ASK_TEXT:
            return BookSide.ASK
    raise ValueError("side must be 'bid' or 'ask'")


def _mapping_get(row: Mapping[str, Any], name: str) -> Any:
    if name not in row:
        raise ValueError(f"missing required field: {name}")
    return row[name]


@dataclass(frozen=True, slots=True)
class ReconstructedPriceLevel:
    price: float
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _require_positive_float(self.price, "price"))
        object.__setattr__(self, "amount", _require_positive_float(self.amount, "amount"))


@dataclass(frozen=True, slots=True)
class ReconstructedBookSnapshot:
    local_ts_us: int
    ts_us: int | None
    raw_source_row: int | None
    bids: tuple[ReconstructedPriceLevel, ...]
    asks: tuple[ReconstructedPriceLevel, ...]
    is_crossed: bool
    bid_depth: int
    ask_depth: int
    reset_count: int
    skipped_pre_snapshot_rows: int
    applied_update_count: int
    deleted_level_count: int

    def __post_init__(self) -> None:
        _require_positive_int(self.local_ts_us, "local_ts_us")
        if self.ts_us is not None:
            _require_positive_int(self.ts_us, "ts_us")
        if self.raw_source_row is not None:
            _require_nonnegative_int(self.raw_source_row, "raw_source_row")
        if not isinstance(self.bids, tuple) or not all(isinstance(x, ReconstructedPriceLevel) for x in self.bids):
            raise ValueError("bids must be a tuple of ReconstructedPriceLevel")
        if not isinstance(self.asks, tuple) or not all(isinstance(x, ReconstructedPriceLevel) for x in self.asks):
            raise ValueError("asks must be a tuple of ReconstructedPriceLevel")
        for idx in range(1, len(self.bids)):
            if not self.bids[idx - 1].price > self.bids[idx].price:
                raise ValueError("bids must be strictly descending by price")
        for idx in range(1, len(self.asks)):
            if not self.asks[idx - 1].price < self.asks[idx].price:
                raise ValueError("asks must be strictly ascending by price")
        _require_bool(self.is_crossed, "is_crossed")
        _require_nonnegative_int(self.bid_depth, "bid_depth")
        _require_nonnegative_int(self.ask_depth, "ask_depth")
        _require_nonnegative_int(self.reset_count, "reset_count")
        _require_nonnegative_int(self.skipped_pre_snapshot_rows, "skipped_pre_snapshot_rows")
        _require_nonnegative_int(self.applied_update_count, "applied_update_count")
        _require_nonnegative_int(self.deleted_level_count, "deleted_level_count")

    @property
    def best_bid(self) -> ReconstructedPriceLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> ReconstructedPriceLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid.price + self.best_ask.price) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask.price - self.best_bid.price


@dataclass(frozen=True, slots=True)
class BookReconstructionStats:
    row_count: int = 0
    skipped_pre_snapshot_rows: int = 0
    snapshot_reset_count: int = 0
    applied_update_count: int = 0
    deleted_level_count: int = 0
    missing_delete_count: int = 0
    bad_row_count: int = 0
    local_ts_decrease_count: int = 0
    emitted_snapshot_count: int = 0
    crossed_snapshot_count: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class IncrementalBookRow:
    local_ts_us: int
    ts_us: int | None
    raw_source_row: int | None
    is_snapshot: bool
    side: BookSide
    price: float
    amount: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_ts_us", _require_positive_int(self.local_ts_us, "local_ts_us"))
        if self.ts_us is not None:
            object.__setattr__(self, "ts_us", _require_positive_int(self.ts_us, "ts_us"))
        if self.raw_source_row is not None:
            object.__setattr__(self, "raw_source_row", _require_nonnegative_int(self.raw_source_row, "raw_source_row"))
        object.__setattr__(self, "is_snapshot", _require_bool(self.is_snapshot, "is_snapshot"))
        object.__setattr__(self, "side", _coerce_book_side(self.side))
        object.__setattr__(self, "price", _require_positive_float(self.price, "price"))
        object.__setattr__(self, "amount", _require_nonnegative_float(self.amount, "amount"))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "IncrementalBookRow":
        local_ts_us = _require_positive_int(_mapping_get(row, LOCAL_TS_US), LOCAL_TS_US)
        ts_us_raw = row.get(TS_US)
        ts_us = None if ts_us_raw is None else _require_positive_int(ts_us_raw, TS_US)
        raw_source_raw = row.get(RAW_SOURCE_ROW)
        raw_source_row = None if raw_source_raw is None else _require_nonnegative_int(raw_source_raw, RAW_SOURCE_ROW)
        is_snapshot = _require_bool(_mapping_get(row, INCREMENTAL_IS_SNAPSHOT), INCREMENTAL_IS_SNAPSHOT)
        side = _coerce_book_side(_mapping_get(row, INCREMENTAL_SIDE))
        price = _require_positive_float(_mapping_get(row, INCREMENTAL_PRICE), INCREMENTAL_PRICE)
        amount = _require_nonnegative_float(_mapping_get(row, INCREMENTAL_AMOUNT), INCREMENTAL_AMOUNT)
        return cls(
            local_ts_us=local_ts_us,
            ts_us=ts_us,
            raw_source_row=raw_source_row,
            is_snapshot=is_snapshot,
            side=side,
            price=price,
            amount=amount,
        )


class L2BookReconstructor:
    def __init__(self) -> None:
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._seen_snapshot = False
        self._previous_row_was_snapshot = False
        self._last_local_ts_us: int | None = None
        self._stats = BookReconstructionStats()

    @property
    def status(self) -> ReconstructedBookStatus:
        return ReconstructedBookStatus.READY if self._seen_snapshot else ReconstructedBookStatus.WAITING_FOR_SNAPSHOT

    @property
    def stats(self) -> BookReconstructionStats:
        return self._stats

    def reset(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._seen_snapshot = False
        self._previous_row_was_snapshot = False
        self._last_local_ts_us = None
        self._stats = BookReconstructionStats()

    def _bump(self, **increments: int) -> None:
        known = set(self._stats.__dataclass_fields__.keys())
        for key, value in increments.items():
            if key not in known:
                raise ValueError(f"unknown stats field: {key}")
            _require_nonnegative_int(value, key)
        values = {name: getattr(self._stats, name) for name in known}
        for key, value in increments.items():
            values[key] += value
        self._stats = BookReconstructionStats(**values)

    def apply_row(self, row: IncrementalBookRow | Mapping[str, Any]) -> bool:
        try:
            parsed = row if isinstance(row, IncrementalBookRow) else IncrementalBookRow.from_mapping(row)
        except ValueError:
            self._bump(bad_row_count=1)
            raise
        self._bump(row_count=1)

        if self._last_local_ts_us is not None and parsed.local_ts_us < self._last_local_ts_us:
            self._bump(local_ts_decrease_count=1)
            raise ValueError("local_ts_us decreased; input must be caller-provided Tardis order")

        if not self._seen_snapshot and not parsed.is_snapshot:
            self._bump(skipped_pre_snapshot_rows=1)
            self._previous_row_was_snapshot = False
            self._last_local_ts_us = parsed.local_ts_us
            return False

        if parsed.is_snapshot and not self._previous_row_was_snapshot:
            self._bids.clear()
            self._asks.clear()
            self._seen_snapshot = True
            self._bump(snapshot_reset_count=1)

        side_dict = self._bids if parsed.side is BookSide.BID else self._asks
        if parsed.amount == 0.0:
            if parsed.price in side_dict:
                del side_dict[parsed.price]
                self._bump(deleted_level_count=1)
            else:
                self._bump(missing_delete_count=1)
        else:
            side_dict[parsed.price] = parsed.amount
            self._bump(applied_update_count=1)

        self._previous_row_was_snapshot = parsed.is_snapshot
        self._last_local_ts_us = parsed.local_ts_us
        return True

    def snapshot(self, *, depth: int = 25, local_ts_us: int | None = None, ts_us: int | None = None, raw_source_row: int | None = None) -> ReconstructedBookSnapshot:
        depth = _require_positive_int(depth, "depth")
        if not self._seen_snapshot:
            raise ValueError("cannot snapshot before first Tardis snapshot row")

        resolved_local_ts = _require_positive_int(local_ts_us, "local_ts_us") if local_ts_us is not None else self._last_local_ts_us
        if resolved_local_ts is None:
            raise ValueError("local_ts_us is required when no rows were applied")
        resolved_ts = None if ts_us is None else _require_positive_int(ts_us, "ts_us")
        resolved_raw_row = None if raw_source_row is None else _require_nonnegative_int(raw_source_row, "raw_source_row")

        bid_levels = tuple(
            ReconstructedPriceLevel(price=price, amount=amount)
            for price, amount in sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        )
        ask_levels = tuple(
            ReconstructedPriceLevel(price=price, amount=amount)
            for price, amount in sorted(self._asks.items(), key=lambda x: x[0])[:depth]
        )
        is_crossed = bool(bid_levels and ask_levels and bid_levels[0].price >= ask_levels[0].price)
        self._bump(emitted_snapshot_count=1)
        if is_crossed:
            self._bump(crossed_snapshot_count=1)
        return ReconstructedBookSnapshot(
            local_ts_us=resolved_local_ts,
            ts_us=resolved_ts,
            raw_source_row=resolved_raw_row,
            bids=bid_levels,
            asks=ask_levels,
            is_crossed=is_crossed,
            bid_depth=len(self._bids),
            ask_depth=len(self._asks),
            reset_count=self._stats.snapshot_reset_count,
            skipped_pre_snapshot_rows=self._stats.skipped_pre_snapshot_rows,
            applied_update_count=self._stats.applied_update_count,
            deleted_level_count=self._stats.deleted_level_count,
        )


def reconstruct_snapshots_from_rows(
    rows: Iterable[IncrementalBookRow | Mapping[str, Any]],
    *,
    depth: int = 25,
) -> Iterator[ReconstructedBookSnapshot]:
    reconstructor = L2BookReconstructor()
    pending_local_ts_us: int | None = None
    pending_ts_us: int | None = None
    pending_raw_source_row: int | None = None

    for row in rows:
        parsed = row if isinstance(row, IncrementalBookRow) else IncrementalBookRow.from_mapping(row)
        if (
            reconstructor.status is ReconstructedBookStatus.READY
            and pending_local_ts_us is not None
            and parsed.local_ts_us != pending_local_ts_us
        ):
            yield reconstructor.snapshot(
                depth=depth,
                local_ts_us=pending_local_ts_us,
                ts_us=pending_ts_us,
                raw_source_row=pending_raw_source_row,
            )

        applied = reconstructor.apply_row(parsed)
        if applied:
            pending_local_ts_us = parsed.local_ts_us
            pending_ts_us = parsed.ts_us
            pending_raw_source_row = parsed.raw_source_row

    if reconstructor.status is ReconstructedBookStatus.READY and pending_local_ts_us is not None:
        yield reconstructor.snapshot(
            depth=depth,
            local_ts_us=pending_local_ts_us,
            ts_us=pending_ts_us,
            raw_source_row=pending_raw_source_row,
        )


def reconstruct_final_snapshot(
    rows: Iterable[IncrementalBookRow | Mapping[str, Any]],
    *,
    depth: int = 25,
) -> ReconstructedBookSnapshot:
    last: ReconstructedBookSnapshot | None = None
    for snap in reconstruct_snapshots_from_rows(rows, depth=depth):
        last = snap
    if last is None:
        raise ValueError("no reconstructable book snapshot emitted")
    return last


__all__ = [
    "TS_US",
    "LOCAL_TS_US",
    "RAW_SOURCE_ROW",
    "INCREMENTAL_IS_SNAPSHOT",
    "INCREMENTAL_SIDE",
    "INCREMENTAL_PRICE",
    "INCREMENTAL_AMOUNT",
    "BOOK_SIDE_BID_TEXT",
    "BOOK_SIDE_ASK_TEXT",
    "ReconstructedBookStatus",
    "ReconstructedPriceLevel",
    "ReconstructedBookSnapshot",
    "BookReconstructionStats",
    "IncrementalBookRow",
    "L2BookReconstructor",
    "reconstruct_snapshots_from_rows",
    "reconstruct_final_snapshot",
]
