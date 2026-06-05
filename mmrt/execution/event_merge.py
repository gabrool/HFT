"""Dependency-light merger for execution-layer L2 and trade events.

This module answers only the local-clock ordering question for the execution
simulator: given reconstructed L2 batches and trade prints, emit deterministic
``ExecutionEventRef`` objects with contiguous ``event_seq`` values.

It intentionally performs no IO, storage-row normalization, L2 reconstruction,
decision scheduling, tape writing, fill simulation, or ML/dataframe work.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator

from mmrt.execution.contracts import (
    ExecutionEventRef,
    ExecutionEventType,
    TradePrint,
)
from mmrt.execution.l2_reconstructor import ReconstructedL2Event


class ExecutionMergeTiePolicy(str, Enum):
    L2_BEFORE_TRADE = "l2_before_trade"
    TRADE_BEFORE_L2 = "trade_before_l2"


@dataclass(frozen=True, slots=True)
class ExecutionMergeCounters:
    l2_event_count: int = 0
    trade_count: int = 0
    emitted_event_count: int = 0
    same_local_ts_tie_count: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class MergedExecutionEvent:
    ref: ExecutionEventRef
    local_ts_us: int
    ts_us: int
    l2_event: ReconstructedL2Event | None = None
    trade: TradePrint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ExecutionEventRef):
            raise ValueError("ref must be ExecutionEventRef")
        _require_positive_int(self.local_ts_us, "local_ts_us")
        _require_positive_int(self.ts_us, "ts_us")
        if self.ref.local_ts_us != self.local_ts_us:
            raise ValueError("ref.local_ts_us must equal local_ts_us")

        has_l2 = self.l2_event is not None
        has_trade = self.trade is not None
        if has_l2 == has_trade:
            raise ValueError("exactly one of l2_event/trade must be set")

        if self.l2_event is not None:
            if not isinstance(self.l2_event, ReconstructedL2Event):
                raise ValueError("l2_event must be ReconstructedL2Event or None")
            if self.ref.event_type != ExecutionEventType.L2_BATCH:
                raise ValueError("L2 merged events require L2_BATCH ref")
            if self.ref.book_ptr < 0:
                raise ValueError("L2 merged events require ref.book_ptr >= 0")
            if self.local_ts_us != self.l2_event.local_ts_us:
                raise ValueError("local_ts_us must equal l2_event.local_ts_us")
            if self.ts_us != self.l2_event.max_ts_us:
                raise ValueError("ts_us must equal l2_event.max_ts_us")

        if self.trade is not None:
            if not isinstance(self.trade, TradePrint):
                raise ValueError("trade must be TradePrint or None")
            if self.ref.event_type != ExecutionEventType.TRADE:
                raise ValueError("trade merged events require TRADE ref")
            if self.ref.trade_ptr < 0:
                raise ValueError("trade merged events require ref.trade_ptr >= 0")
            if self.local_ts_us != self.trade.local_ts_us:
                raise ValueError("local_ts_us must equal trade.local_ts_us")
            if self.ts_us != self.trade.ts_us:
                raise ValueError("ts_us must equal trade.ts_us")


@dataclass(frozen=True, slots=True)
class ExecutionMergePlan:
    events: tuple[MergedExecutionEvent, ...]
    counters: ExecutionMergeCounters

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise ValueError("events must be a tuple")
        if not isinstance(self.counters, ExecutionMergeCounters):
            raise ValueError("counters must be ExecutionMergeCounters")

        prev_local_ts_us: int | None = None
        for idx, event in enumerate(self.events):
            if not isinstance(event, MergedExecutionEvent):
                raise ValueError("events must contain MergedExecutionEvent values")
            if event.ref.event_seq != idx:
                raise ValueError("event_seq values must be contiguous from 0")
            if prev_local_ts_us is not None and event.local_ts_us < prev_local_ts_us:
                raise ValueError("events must be nondecreasing by local_ts_us")
            prev_local_ts_us = event.local_ts_us


def iter_merged_execution_events(
    l2_events: Iterable[ReconstructedL2Event],
    trades: Iterable[TradePrint],
    *,
    tie_policy: ExecutionMergeTiePolicy | str = ExecutionMergeTiePolicy.L2_BEFORE_TRADE,
) -> Iterator[MergedExecutionEvent]:
    """Stream a stable local-clock merge of reconstructed L2 events and trades."""
    policy = _coerce_tie_policy(tie_policy)
    yield from _iter_merged_execution_events(l2_events, trades, tie_policy=policy)


def merge_execution_events(
    l2_events: Iterable[ReconstructedL2Event],
    trades: Iterable[TradePrint],
    *,
    tie_policy: ExecutionMergeTiePolicy | str = ExecutionMergeTiePolicy.L2_BEFORE_TRADE,
) -> ExecutionMergePlan:
    """Materialize a small merged plan and diagnostics counters.

    Large tape builders should prefer :func:`iter_merged_execution_events`.
    """
    policy = _coerce_tie_policy(tie_policy)
    tie_counter = [0]

    events = tuple(_iter_merged_execution_events(l2_events, trades, tie_policy=policy, tie_counter=tie_counter))
    l2_event_count = 0
    trade_count = 0
    for event in events:
        if event.l2_event is not None:
            l2_event_count += 1
        else:
            trade_count += 1

    counters = ExecutionMergeCounters(
        l2_event_count=l2_event_count,
        trade_count=trade_count,
        emitted_event_count=len(events),
        same_local_ts_tie_count=tie_counter[0],
    )
    return ExecutionMergePlan(events=events, counters=counters)


def _iter_merged_execution_events(
    l2_events: Iterable[ReconstructedL2Event],
    trades: Iterable[TradePrint],
    *,
    tie_policy: ExecutionMergeTiePolicy,
    tie_counter: list[int] | None = None,
) -> Iterator[MergedExecutionEvent]:
    l2_iter = _iter_l2_with_index(l2_events)
    trade_iter = _iter_trade_with_index(trades)

    l2_item = next(l2_iter, None)
    trade_item = next(trade_iter, None)
    event_seq = 0

    while l2_item is not None or trade_item is not None:
        if l2_item is None:
            use_l2 = False
        elif trade_item is None:
            use_l2 = True
        else:
            use_l2, is_tie = _choose_l2(l2_item[1], trade_item[1], tie_policy)
            if is_tie and tie_counter is not None:
                tie_counter[0] += 1

        if use_l2:
            if l2_item is None:
                raise RuntimeError("merge selected missing L2 event")
            l2_index, l2_event = l2_item
            yield _make_l2_merged_event(event_seq, l2_index, l2_event)
            l2_item = next(l2_iter, None)
        else:
            if trade_item is None:
                raise RuntimeError("merge selected missing trade event")
            trade_index, trade = trade_item
            yield _make_trade_merged_event(event_seq, trade_index, trade)
            trade_item = next(trade_iter, None)
        event_seq += 1


def _make_l2_merged_event(
    event_seq: int,
    l2_index: int,
    l2_event: ReconstructedL2Event,
) -> MergedExecutionEvent:
    return MergedExecutionEvent(
        ref=ExecutionEventRef(
            event_seq=event_seq,
            local_ts_us=l2_event.local_ts_us,
            event_type=ExecutionEventType.L2_BATCH,
            book_ptr=l2_index,
        ),
        local_ts_us=l2_event.local_ts_us,
        ts_us=l2_event.max_ts_us,
        l2_event=l2_event,
    )


def _make_trade_merged_event(
    event_seq: int,
    trade_index: int,
    trade: TradePrint,
) -> MergedExecutionEvent:
    return MergedExecutionEvent(
        ref=ExecutionEventRef(
            event_seq=event_seq,
            local_ts_us=trade.local_ts_us,
            event_type=ExecutionEventType.TRADE,
            trade_ptr=trade_index,
        ),
        local_ts_us=trade.local_ts_us,
        ts_us=trade.ts_us,
        trade=trade,
    )


def _iter_l2_with_index(
    events: Iterable[ReconstructedL2Event],
) -> Iterator[tuple[int, ReconstructedL2Event]]:
    prev_local_ts_us: int | None = None
    for idx, event in enumerate(events):
        if not isinstance(event, ReconstructedL2Event):
            raise ValueError("l2_events must contain ReconstructedL2Event values")
        if prev_local_ts_us is not None and event.local_ts_us < prev_local_ts_us:
            raise ValueError("l2_events must be sorted by nondecreasing local_ts_us")
        prev_local_ts_us = event.local_ts_us
        yield idx, event


def _iter_trade_with_index(trades: Iterable[TradePrint]) -> Iterator[tuple[int, TradePrint]]:
    prev_local_ts_us: int | None = None
    for idx, trade in enumerate(trades):
        if not isinstance(trade, TradePrint):
            raise ValueError("trades must contain TradePrint values")
        if prev_local_ts_us is not None and trade.local_ts_us < prev_local_ts_us:
            raise ValueError("trades must be sorted by nondecreasing local_ts_us")
        prev_local_ts_us = trade.local_ts_us
        yield idx, trade


def _coerce_tie_policy(value: ExecutionMergeTiePolicy | str) -> ExecutionMergeTiePolicy:
    if isinstance(value, ExecutionMergeTiePolicy):
        return value
    if isinstance(value, str):
        try:
            return ExecutionMergeTiePolicy(value)
        except ValueError as exc:
            raise ValueError(f"invalid tie_policy: {value!r}") from exc
    raise ValueError("tie_policy must be ExecutionMergeTiePolicy or str")


def _choose_l2(
    l2_event: ReconstructedL2Event,
    trade: TradePrint,
    tie_policy: ExecutionMergeTiePolicy,
) -> tuple[bool, bool]:
    if l2_event.local_ts_us < trade.local_ts_us:
        return True, False
    if l2_event.local_ts_us > trade.local_ts_us:
        return False, False
    if tie_policy == ExecutionMergeTiePolicy.L2_BEFORE_TRADE:
        return True, True
    if tie_policy == ExecutionMergeTiePolicy.TRADE_BEFORE_L2:
        return False, True
    raise ValueError(f"invalid tie_policy: {tie_policy!r}")


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value



__all__ = [
    "ExecutionMergeTiePolicy",
    "ExecutionMergeCounters",
    "MergedExecutionEvent",
    "ExecutionMergePlan",
    "iter_merged_execution_events",
    "merge_execution_events",
]
