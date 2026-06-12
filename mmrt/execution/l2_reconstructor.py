"""Batch-native, tick-native Tardis L2 order book reconstruction.

This module consumes normalized execution-layer L2 contracts and maintains an
in-memory market-by-price book. It is intentionally IO-free and dependency-light:
it does not parse files, import dataframe libraries, merge event streams, compute
features, or simulate fills.

Tardis incremental_book_L2 rows are flat price-level updates where rows sharing a
local timestamp can belong to one WebSocket message. This reconstructor therefore
applies one grouped :class:`L2UpdateBatch` atomically and exposes at most one
externally consistent event per batch.
"""

from bisect import bisect_left, insort
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator

from mmrt.contracts import BookSide
from mmrt.execution.contracts import (
    BookLevelSnapshot,
    BookTop,
    L2Update,
    L2UpdateBatch,
    SymbolSpec,
)


class L2ReconstructionStatus(str, Enum):
    WAITING_FOR_SNAPSHOT = "waiting_for_snapshot"
    READY = "ready"


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


def _require_positive_depth(value: int, name: str = "depth") -> int:
    return _require_positive_int(value, name)


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


@dataclass(frozen=True, slots=True)
class L2ReconstructionCounters:
    batches_seen: int = 0
    updates_seen: int = 0
    skipped_pre_snapshot_updates: int = 0
    snapshot_reset_count: int = 0
    applied_update_count: int = 0
    deleted_level_count: int = 0
    missing_delete_count: int = 0
    emitted_event_count: int = 0
    crossed_batch_count: int = 0
    crossed_repair_count: int = 0
    crossed_levels_removed: int = 0
    local_ts_decrease_count: int = 0
    max_bid_depth: int = 0
    max_ask_depth: int = 0
    max_batch_size: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ReconstructedL2Event:
    batch_seq: int
    local_ts_us: int
    min_ts_us: int
    max_ts_us: int
    num_updates: int
    is_snapshot_batch: bool
    book_top: BookTop | None
    bid_depth: int
    ask_depth: int
    book_snapshot: BookLevelSnapshot | None = None
    crossed_repaired: bool = False
    crossed_levels_removed: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.batch_seq, "batch_seq")
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_positive_int(self.min_ts_us, "min_ts_us")
        _require_positive_int(self.max_ts_us, "max_ts_us")
        if self.max_ts_us < self.min_ts_us:
            raise ValueError("max_ts_us must be >= min_ts_us")
        _require_positive_int(self.num_updates, "num_updates")
        _require_bool(self.is_snapshot_batch, "is_snapshot_batch")
        if self.book_top is not None and not isinstance(self.book_top, BookTop):
            raise ValueError("book_top must be BookTop or None")
        if self.book_snapshot is not None:
            if not isinstance(self.book_snapshot, BookLevelSnapshot):
                raise ValueError("book_snapshot must be BookLevelSnapshot or None")
            if self.book_snapshot.local_ts_us != self.local_ts_us:
                raise ValueError("book_snapshot.local_ts_us must equal event local_ts_us")
            if len(self.book_snapshot.bid_ticks) != len(self.book_snapshot.bid_sizes):
                raise ValueError("book_snapshot bid ticks/sizes lengths must match")
            if len(self.book_snapshot.ask_ticks) != len(self.book_snapshot.ask_sizes):
                raise ValueError("book_snapshot ask ticks/sizes lengths must match")
        _require_nonnegative_int(self.bid_depth, "bid_depth")
        _require_nonnegative_int(self.ask_depth, "ask_depth")
        _require_bool(self.crossed_repaired, "crossed_repaired")
        _require_nonnegative_int(self.crossed_levels_removed, "crossed_levels_removed")
        if not self.crossed_repaired and self.crossed_levels_removed != 0:
            raise ValueError("crossed_levels_removed must be 0 when crossed_repaired is False")

    @classmethod
    def from_trusted(
        cls,
        *,
        batch_seq: int,
        local_ts_us: int,
        min_ts_us: int,
        max_ts_us: int,
        num_updates: int,
        is_snapshot_batch: bool,
        book_top: BookTop | None,
        bid_depth: int,
        ask_depth: int,
        book_snapshot: BookLevelSnapshot | None,
        crossed_repaired: bool,
        crossed_levels_removed: int,
    ) -> "ReconstructedL2Event":
        """Construct without re-validation for reconstructor-produced events."""
        self = object.__new__(cls)
        object.__setattr__(self, "batch_seq", batch_seq)
        object.__setattr__(self, "local_ts_us", local_ts_us)
        object.__setattr__(self, "min_ts_us", min_ts_us)
        object.__setattr__(self, "max_ts_us", max_ts_us)
        object.__setattr__(self, "num_updates", num_updates)
        object.__setattr__(self, "is_snapshot_batch", is_snapshot_batch)
        object.__setattr__(self, "book_top", book_top)
        object.__setattr__(self, "bid_depth", bid_depth)
        object.__setattr__(self, "ask_depth", ask_depth)
        object.__setattr__(self, "book_snapshot", book_snapshot)
        object.__setattr__(self, "crossed_repaired", crossed_repaired)
        object.__setattr__(self, "crossed_levels_removed", crossed_levels_removed)
        return self


class _SideBook:
    __slots__ = ("side", "_qty_by_tick", "_ticks")

    side: BookSide
    _qty_by_tick: dict[int, float]
    _ticks: list[int]

    def __init__(self, side: BookSide) -> None:
        if side not in (BookSide.BID, BookSide.ASK):
            raise ValueError("side must be BookSide.BID or BookSide.ASK")
        self.side = side
        self._qty_by_tick = {}
        self._ticks = []

    def clear(self) -> None:
        self._qty_by_tick.clear()
        self._ticks.clear()

    def __len__(self) -> int:
        return len(self._ticks)

    def has(self, price_tick: int) -> bool:
        return price_tick in self._qty_by_tick

    def qty_at(self, price_tick: int) -> float:
        return self._qty_by_tick[price_tick]

    def upsert(self, price_tick: int, amount: float) -> None:
        if price_tick not in self._qty_by_tick:
            insort(self._ticks, price_tick)
        self._qty_by_tick[price_tick] = amount

    def delete(self, price_tick: int) -> bool:
        if price_tick not in self._qty_by_tick:
            return False
        del self._qty_by_tick[price_tick]
        idx = bisect_left(self._ticks, price_tick)
        if idx >= len(self._ticks) or self._ticks[idx] != price_tick:
            raise RuntimeError("side book tick index corrupted")
        del self._ticks[idx]
        return True

    def best_tick(self) -> int | None:
        if not self._ticks:
            return None
        if self.side is BookSide.BID:
            return self._ticks[-1]
        return self._ticks[0]

    def best_size(self) -> float | None:
        tick = self.best_tick()
        if tick is None:
            return None
        return self._qty_by_tick[tick]

    def top_ticks_and_sizes(self, depth: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        if self.side is BookSide.BID:
            ticks = self._ticks[-depth:][::-1]
        else:
            ticks = self._ticks[:depth]
        qty_by_tick = self._qty_by_tick
        return tuple(ticks), tuple([qty_by_tick[tick] for tick in ticks])


class L2BookReconstructor:
    __slots__ = (
        "symbol_spec",
        "snapshot_depth",
        "_bids",
        "_asks",
        "_seen_snapshot",
        "_previous_row_was_snapshot",
        "_last_batch_local_ts_us",
        "_batches_seen",
        "_updates_seen",
        "_skipped_pre_snapshot_updates",
        "_snapshot_reset_count",
        "_applied_update_count",
        "_deleted_level_count",
        "_missing_delete_count",
        "_emitted_event_count",
        "_crossed_batch_count",
        "_crossed_repair_count",
        "_crossed_levels_removed",
        "_local_ts_decrease_count",
        "_max_bid_depth",
        "_max_ask_depth",
        "_max_batch_size",
    )

    def __init__(self, symbol_spec: SymbolSpec, *, snapshot_depth: int = 25) -> None:
        if not isinstance(symbol_spec, SymbolSpec):
            raise ValueError("symbol_spec must be SymbolSpec")
        self.symbol_spec = symbol_spec
        self.snapshot_depth = _require_positive_int(snapshot_depth, "snapshot_depth")
        self._bids = _SideBook(BookSide.BID)
        self._asks = _SideBook(BookSide.ASK)
        self._seen_snapshot = False
        self._previous_row_was_snapshot = False
        self._last_batch_local_ts_us: int | None = None
        self._reset_counters()

    @property
    def status(self) -> L2ReconstructionStatus:
        if self._seen_snapshot:
            return L2ReconstructionStatus.READY
        return L2ReconstructionStatus.WAITING_FOR_SNAPSHOT

    @property
    def counters(self) -> L2ReconstructionCounters:
        return L2ReconstructionCounters(
            batches_seen=self._batches_seen,
            updates_seen=self._updates_seen,
            skipped_pre_snapshot_updates=self._skipped_pre_snapshot_updates,
            snapshot_reset_count=self._snapshot_reset_count,
            applied_update_count=self._applied_update_count,
            deleted_level_count=self._deleted_level_count,
            missing_delete_count=self._missing_delete_count,
            emitted_event_count=self._emitted_event_count,
            crossed_batch_count=self._crossed_batch_count,
            crossed_repair_count=self._crossed_repair_count,
            crossed_levels_removed=self._crossed_levels_removed,
            local_ts_decrease_count=self._local_ts_decrease_count,
            max_bid_depth=self._max_bid_depth,
            max_ask_depth=self._max_ask_depth,
            max_batch_size=self._max_batch_size,
        )

    @property
    def is_ready(self) -> bool:
        return self._seen_snapshot

    def _reset_counters(self) -> None:
        self._batches_seen = 0
        self._updates_seen = 0
        self._skipped_pre_snapshot_updates = 0
        self._snapshot_reset_count = 0
        self._applied_update_count = 0
        self._deleted_level_count = 0
        self._missing_delete_count = 0
        self._emitted_event_count = 0
        self._crossed_batch_count = 0
        self._crossed_repair_count = 0
        self._crossed_levels_removed = 0
        self._local_ts_decrease_count = 0
        self._max_bid_depth = 0
        self._max_ask_depth = 0
        self._max_batch_size = 0

    def reset(self, *, reset_counters: bool = True) -> None:
        self._bids.clear()
        self._asks.clear()
        self._seen_snapshot = False
        self._previous_row_was_snapshot = False
        self._last_batch_local_ts_us = None
        if reset_counters:
            self._reset_counters()

    def apply_batch(self, batch: L2UpdateBatch) -> ReconstructedL2Event | None:
        if not isinstance(batch, L2UpdateBatch):
            raise ValueError("batch must be L2UpdateBatch")
        if self._last_batch_local_ts_us is not None and batch.local_ts_us <= self._last_batch_local_ts_us:
            self._local_ts_decrease_count += 1
            raise ValueError("batch.local_ts_us must be strictly greater than previous batch local_ts_us")

        updates = batch.updates
        self._batches_seen += 1
        self._updates_seen += len(updates)
        if len(updates) > self._max_batch_size:
            self._max_batch_size = len(updates)

        any_ready_update = False
        last_positive_update_side: BookSide | None = None
        batch_crossed_repaired = False
        batch_crossed_levels_removed = 0

        for update in updates:
            if update.is_snapshot and not self._previous_row_was_snapshot:
                self._bids.clear()
                self._asks.clear()
                self._seen_snapshot = True
                self._snapshot_reset_count += 1

            if not self._seen_snapshot and not update.is_snapshot:
                self._skipped_pre_snapshot_updates += 1
                self._previous_row_was_snapshot = False
                continue

            side_book = self._bids if update.side is BookSide.BID else self._asks
            if update.amount == 0:
                if side_book.delete(update.price_tick):
                    self._deleted_level_count += 1
                else:
                    self._missing_delete_count += 1
                any_ready_update = True
            else:
                side_book.upsert(update.price_tick, update.amount)
                self._applied_update_count += 1
                any_ready_update = True
                last_positive_update_side = update.side

            self._previous_row_was_snapshot = update.is_snapshot

        if last_positive_update_side is not None:
            crossed_repaired, removed = self._repair_crossed_after_update(last_positive_update_side)
            if crossed_repaired:
                batch_crossed_repaired = True
                batch_crossed_levels_removed += removed

        bid_depth = len(self._bids)
        ask_depth = len(self._asks)
        if bid_depth > self._max_bid_depth:
            self._max_bid_depth = bid_depth
        if ask_depth > self._max_ask_depth:
            self._max_ask_depth = ask_depth
        self._last_batch_local_ts_us = batch.local_ts_us

        if not any_ready_update:
            return None

        if batch_crossed_repaired:
            self._crossed_batch_count += 1

        self._emitted_event_count += 1
        return ReconstructedL2Event.from_trusted(
            batch_seq=batch.batch_seq,
            local_ts_us=batch.local_ts_us,
            min_ts_us=batch.min_ts_us,
            max_ts_us=batch.max_ts_us,
            num_updates=len(updates),
            is_snapshot_batch=batch.is_snapshot_batch,
            book_top=self.book_top(local_ts_us=batch.local_ts_us),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            book_snapshot=self.snapshot(depth=self.snapshot_depth, local_ts_us=batch.local_ts_us),
            crossed_repaired=batch_crossed_repaired,
            crossed_levels_removed=batch_crossed_levels_removed,
        )

    def _repair_crossed_after_update(self, updated_side: BookSide) -> tuple[bool, int]:
        removed = 0
        while True:
            best_bid = self._bids.best_tick()
            best_ask = self._asks.best_tick()
            if best_bid is None or best_ask is None or best_bid < best_ask:
                break
            if updated_side is BookSide.BID:
                deleted = self._asks.delete(best_ask)
            else:
                deleted = self._bids.delete(best_bid)
            if not deleted:
                raise RuntimeError("crossed book repair failed to delete best level")
            removed += 1

        if removed > 0:
            self._crossed_repair_count += 1
            self._crossed_levels_removed += removed
        return removed > 0, removed

    def book_top(self, *, local_ts_us: int | None = None) -> BookTop | None:
        if not self.is_ready:
            raise ValueError("book is not ready; waiting for initial snapshot")
        effective_local_ts_us = self._effective_local_ts_us(local_ts_us)
        best_bid_tick = self._bids.best_tick()
        best_ask_tick = self._asks.best_tick()
        if best_bid_tick is None or best_ask_tick is None:
            return None
        best_bid_size = self._bids.best_size()
        best_ask_size = self._asks.best_size()
        if best_bid_size is None or best_ask_size is None:
            return None
        return BookTop(effective_local_ts_us, best_bid_tick, best_ask_tick, best_bid_size, best_ask_size)

    def snapshot(self, *, depth: int = 25, local_ts_us: int | None = None) -> BookLevelSnapshot:
        if not self.is_ready:
            raise ValueError("book is not ready; waiting for initial snapshot")
        depth = _require_positive_depth(depth)
        effective_local_ts_us = self._effective_local_ts_us(local_ts_us)
        bid_ticks, bid_sizes = self._bids.top_ticks_and_sizes(depth)
        ask_ticks, ask_sizes = self._asks.top_ticks_and_sizes(depth)
        # The side books keep ticks sorted, so the level invariants hold by
        # construction and per-level re-validation is skipped.
        return BookLevelSnapshot.from_trusted(
            local_ts_us=effective_local_ts_us,
            bid_ticks=bid_ticks,
            bid_sizes=bid_sizes,
            ask_ticks=ask_ticks,
            ask_sizes=ask_sizes,
        )

    def _effective_local_ts_us(self, local_ts_us: int | None) -> int:
        if local_ts_us is None:
            if self._last_batch_local_ts_us is None:
                raise ValueError("local_ts_us is unavailable before applying a batch")
            return self._last_batch_local_ts_us
        return _require_positive_int(local_ts_us, "local_ts_us")

    def bid_depth(self) -> int:
        return len(self._bids)

    def ask_depth(self) -> int:
        return len(self._asks)


def iter_l2_update_batches(updates: Iterable[L2Update]) -> Iterator[L2UpdateBatch]:
    current_local_ts_us: int | None = None
    current_updates: list[L2Update] = []
    last_local_ts_us: int | None = None
    batch_seq = 0

    for update in updates:
        if not isinstance(update, L2Update):
            raise ValueError("updates must contain L2Update values")
        if last_local_ts_us is not None and update.local_ts_us < last_local_ts_us:
            raise ValueError("updates must be sorted/grouped by nondecreasing local_ts_us")

        if current_local_ts_us is None:
            current_local_ts_us = update.local_ts_us
            current_updates.append(update)
        elif update.local_ts_us == current_local_ts_us:
            current_updates.append(update)
        else:
            yield _make_batch(current_local_ts_us, current_updates, batch_seq)
            batch_seq += 1
            current_local_ts_us = update.local_ts_us
            current_updates = [update]

        last_local_ts_us = update.local_ts_us

    if current_local_ts_us is not None:
        yield _make_batch(current_local_ts_us, current_updates, batch_seq)


def _make_batch(local_ts_us: int, updates: list[L2Update], batch_seq: int) -> L2UpdateBatch:
    min_ts_us = min(update.ts_us for update in updates)
    max_ts_us = max(update.ts_us for update in updates)
    # Grouping by identical local_ts_us guarantees the batch invariants the
    # validated constructor would re-check per update.
    return L2UpdateBatch.from_trusted(
        local_ts_us=local_ts_us,
        min_ts_us=min_ts_us,
        max_ts_us=max_ts_us,
        updates=tuple(updates),
        is_snapshot_batch=any(update.is_snapshot for update in updates),
        batch_seq=batch_seq,
    )


__all__ = [
    "L2ReconstructionStatus",
    "L2ReconstructionCounters",
    "ReconstructedL2Event",
    "L2BookReconstructor",
    "iter_l2_update_batches",
]
