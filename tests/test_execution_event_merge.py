from pathlib import Path

import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import ExecutionEventRef, ExecutionEventType, TradePrint
from mmrt.execution.event_merge import (
    ExecutionMergeCounterAccumulator,
    ExecutionMergeTiePolicy,
    MergedExecutionEvent,
    iter_merged_execution_events,
    merge_execution_events,
)
from mmrt.execution.l2_reconstructor import ReconstructedL2Event


def _l2(local, ts_min=None, ts_max=None, seq=0):
    ts_min = local - 10 if ts_min is None else ts_min
    ts_max = local - 5 if ts_max is None else ts_max
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local,
        min_ts_us=ts_min,
        max_ts_us=ts_max,
        num_updates=1,
        is_snapshot_batch=False,
        book_top=None,
        bid_depth=1,
        ask_depth=1,
    )


def _trade(local, ts=None, idx=0):
    return TradePrint(
        local_ts_us=local,
        ts_us=local - 1 if ts is None else ts,
        side=AggressorSide.BUY,
        price_tick=1000 + idx,
        amount=0.01,
        trade_id=str(idx),
        source_row=idx,
    )


def _ref(event_seq=0, local=100, event_type=ExecutionEventType.L2_BATCH, book_ptr=-1, trade_ptr=-1):
    return ExecutionEventRef(
        event_seq=event_seq,
        local_ts_us=local,
        event_type=event_type,
        book_ptr=book_ptr,
        trade_ptr=trade_ptr,
    )


def test_empty_inputs_produce_empty_plan():
    plan = merge_execution_events([], [])

    assert plan.events == ()
    assert plan.counters.l2_event_count == 0
    assert plan.counters.trade_count == 0
    assert plan.counters.emitted_event_count == 0
    assert plan.counters.same_local_ts_tie_count == 0


def test_merge_by_local_timestamp_preserves_event_seq_and_source_pointers():
    events = list(iter_merged_execution_events([_l2(100), _l2(300)], [_trade(200), _trade(400)]))

    assert [event.ref.event_type for event in events] == [
        ExecutionEventType.L2_BATCH,
        ExecutionEventType.TRADE,
        ExecutionEventType.L2_BATCH,
        ExecutionEventType.TRADE,
    ]
    assert [event.ref.event_seq for event in events] == [0, 1, 2, 3]
    assert [event.ref.book_ptr for event in events if event.l2_event is not None] == [0, 1]
    assert [event.ref.trade_ptr for event in events if event.trade is not None] == [0, 1]
    assert [event.local_ts_us for event in events] == sorted(event.local_ts_us for event in events)


def test_default_tie_policy_is_l2_before_trade():
    plan = merge_execution_events([_l2(100)], [_trade(100)])

    assert [event.ref.event_type for event in plan.events] == [
        ExecutionEventType.L2_BATCH,
        ExecutionEventType.TRADE,
    ]
    assert plan.counters.same_local_ts_tie_count == 1


def test_trade_before_l2_tie_policy_accepts_enum_and_string():
    enum_events = list(
        iter_merged_execution_events(
            [_l2(100)],
            [_trade(100)],
            tie_policy=ExecutionMergeTiePolicy.TRADE_BEFORE_L2,
        )
    )
    string_events = list(iter_merged_execution_events([_l2(100)], [_trade(100)], tie_policy="trade_before_l2"))

    assert enum_events[0].ref.event_type == ExecutionEventType.TRADE
    assert enum_events[1].ref.event_type == ExecutionEventType.L2_BATCH
    assert [event.ref.event_type for event in string_events] == [event.ref.event_type for event in enum_events]


def test_multiple_same_local_ties_are_stable():
    plan = merge_execution_events(
        [_l2(100, seq=0), _l2(200, seq=1)],
        [_trade(100, idx=0), _trade(200, idx=1)],
        tie_policy=ExecutionMergeTiePolicy.TRADE_BEFORE_L2,
    )

    assert plan.counters.same_local_ts_tie_count == 2
    assert [(event.local_ts_us, event.ref.event_type) for event in plan.events] == [
        (100, ExecutionEventType.TRADE),
        (100, ExecutionEventType.L2_BATCH),
        (200, ExecutionEventType.TRADE),
        (200, ExecutionEventType.L2_BATCH),
    ]


def test_rejects_unsorted_l2_stream():
    with pytest.raises(ValueError):
        list(iter_merged_execution_events([_l2(200), _l2(100)], []))


def test_rejects_unsorted_trade_stream():
    with pytest.raises(ValueError):
        list(iter_merged_execution_events([], [_trade(200), _trade(100)]))


def test_rejects_wrong_input_types():
    with pytest.raises(ValueError):
        list(iter_merged_execution_events([object()], []))

    with pytest.raises(ValueError):
        list(iter_merged_execution_events([], [object()]))


def test_merged_execution_event_validation():
    l2 = _l2(100)
    trade = _trade(100)
    l2_ref = _ref(local=100, event_type=ExecutionEventType.L2_BATCH, book_ptr=0)
    trade_ref = _ref(local=100, event_type=ExecutionEventType.TRADE, trade_ptr=0)

    with pytest.raises(ValueError):
        MergedExecutionEvent(ref=l2_ref, local_ts_us=100, ts_us=95, l2_event=l2, trade=trade)

    with pytest.raises(ValueError):
        MergedExecutionEvent(ref=l2_ref, local_ts_us=100, ts_us=95)

    with pytest.raises(ValueError):
        MergedExecutionEvent(ref=trade_ref, local_ts_us=100, ts_us=95, l2_event=l2)

    with pytest.raises(ValueError):
        MergedExecutionEvent(ref=l2_ref, local_ts_us=100, ts_us=99, trade=trade)

    with pytest.raises(ValueError):
        MergedExecutionEvent(ref=l2_ref, local_ts_us=101, ts_us=95, l2_event=l2)

    with pytest.raises(ValueError):
        MergedExecutionEvent(
            ref=ExecutionEventRef(
                event_seq=0,
                local_ts_us=999,
                event_type=ExecutionEventType.L2_BATCH,
                book_ptr=0,
            ),
            local_ts_us=100,
            ts_us=95,
            l2_event=l2,
        )

    with pytest.raises(ValueError):
        MergedExecutionEvent(
            ref=ExecutionEventRef(
                event_seq=0,
                local_ts_us=999,
                event_type=ExecutionEventType.TRADE,
                trade_ptr=0,
            ),
            local_ts_us=100,
            ts_us=99,
            trade=trade,
        )


def test_execution_event_merge_has_no_heavy_imports():
    source = Path("mmrt/execution/event_merge.py").read_text()

    assert "import torch" not in source
    assert "import polars" not in source
    assert "import pandas" not in source
    assert "import pyarrow" not in source
    assert "import numpy" not in source


def test_iter_merged_execution_events_counter_matches_materialized_plan():
    l2_events = [_l2(100, seq=0), _l2(200, seq=1)]
    trades = [_trade(100, idx=0), _trade(250, idx=1)]
    acc = ExecutionMergeCounterAccumulator()
    streamed = tuple(iter_merged_execution_events(l2_events, trades, counter=acc))
    materialized = merge_execution_events(l2_events, trades)
    assert streamed == materialized.events
    assert acc.as_counters() == materialized.counters
    assert acc.as_dict() == {
        "l2_event_count": materialized.counters.l2_event_count,
        "trade_count": materialized.counters.trade_count,
        "emitted_event_count": materialized.counters.emitted_event_count,
        "same_local_ts_tie_count": materialized.counters.same_local_ts_tie_count,
    }
