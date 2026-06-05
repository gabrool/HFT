import json
from pathlib import Path

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookTop, ExecutionEventRef, ExecutionEventType, SymbolSpec, TradePrint
from mmrt.execution.event_merge import MergedExecutionEvent, merge_execution_events
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.execution_tape import (
    EVENT_DTYPE,
    L2_EVENT_DTYPE,
    TRADE_DTYPE,
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    EVENTS_ARRAY_NAME,
    L2_EVENTS_ARRAY_NAME,
    TRADES_ARRAY_NAME,
    ExecutionTape,
    ExecutionTapeArrays,
    build_execution_tape,
    execution_tape_manifest_from_dict,
    execution_tape_manifest_to_dict,
    load_execution_tape,
    save_execution_tape,
)


def _spec():
    return SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=5.0,
    )


def _l2(local, seq=0, top=True):
    book_top = BookTop(local, 1000 + seq, 1002 + seq, 1.0, 2.0) if top else None
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local,
        min_ts_us=local - 10,
        max_ts_us=local - 5,
        num_updates=2,
        is_snapshot_batch=(seq == 0),
        book_top=book_top,
        bid_depth=3,
        ask_depth=4,
    )


def _trade(local, idx=0, side=AggressorSide.BUY):
    return TradePrint(
        local_ts_us=local,
        ts_us=local - 1,
        side=side,
        price_tick=1001 + idx,
        amount=0.01,
        trade_id=str(idx),
        source_row=idx,
    )


def _basic_tape():
    l2 = [_l2(100, seq=0), _l2(300, seq=1)]
    trades = [_trade(200, idx=0), _trade(400, idx=1)]
    plan = merge_execution_events(l2, trades)
    return build_execution_tape(symbol_spec=_spec(), l2_events=l2, trades=trades, merged_events=plan.events)


def test_build_tape_arrays_basic():
    tape = _basic_tape()

    assert isinstance(tape, ExecutionTape)
    assert tape.arrays.events.dtype == EVENT_DTYPE
    assert tape.arrays.l2_events.dtype == L2_EVENT_DTYPE
    assert tape.arrays.trades.dtype == TRADE_DTYPE
    assert len(tape.arrays.events) == 4
    assert tape.arrays.events["event_seq"].tolist() == [0, 1, 2, 3]
    assert tape.arrays.events["event_type_code"].tolist() == [
        EVENT_TYPE_CODE_L2_BATCH,
        EVENT_TYPE_CODE_TRADE,
        EVENT_TYPE_CODE_L2_BATCH,
        EVENT_TYPE_CODE_TRADE,
    ]
    assert tape.arrays.events["book_ptr"].tolist() == [0, -1, 1, -1]
    assert tape.arrays.events["trade_ptr"].tolist() == [-1, 0, -1, 1]
    assert tape.manifest.num_events == len(tape.arrays.events)
    assert tape.manifest.num_l2_batches == len(tape.arrays.l2_events)
    assert tape.manifest.num_trades == len(tape.arrays.trades)
    assert [value.value for value in tape.manifest.source_data_types] == ["incremental_book_L2", "trades"]
    assert tape.manifest.array_names == (EVENTS_ARRAY_NAME, L2_EVENTS_ARRAY_NAME, TRADES_ARRAY_NAME)


def test_l2_book_top_none_uses_sentinel_values():
    l2 = [_l2(100, seq=0, top=False)]
    plan = merge_execution_events(l2, [])
    tape = build_execution_tape(symbol_spec=_spec(), l2_events=l2, trades=[], merged_events=plan.events)

    row = tape.arrays.l2_events[0]
    assert int(row["best_bid_tick"]) == -1
    assert int(row["best_ask_tick"]) == -1
    assert float(row["best_bid_size"]) == 0.0
    assert float(row["best_ask_size"]) == 0.0


def test_trade_side_encoding():
    trades = [
        _trade(100, idx=0, side=AggressorSide.BUY),
        _trade(200, idx=1, side=AggressorSide.SELL),
        _trade(300, idx=2, side=AggressorSide.UNKNOWN),
    ]
    plan = merge_execution_events([], trades)
    tape = build_execution_tape(symbol_spec=_spec(), l2_events=[], trades=trades, merged_events=plan.events)

    assert tape.arrays.trades["side_code"].tolist() == [1, -1, 0]


def test_rejects_empty_tape():
    with pytest.raises(ValueError):
        build_execution_tape(symbol_spec=_spec(), l2_events=[], trades=[], merged_events=[])


def test_rejects_l2_pointer_mismatch():
    l2_a = _l2(100, seq=0)
    l2_b = _l2(100, seq=0)
    plan = merge_execution_events([l2_a], [])

    with pytest.raises(ValueError, match="merged L2 event pointer"):
        build_execution_tape(symbol_spec=_spec(), l2_events=[l2_b], trades=[], merged_events=plan.events)


def test_rejects_trade_pointer_mismatch():
    trade_a = _trade(100, idx=0)
    trade_b = _trade(100, idx=0)
    plan = merge_execution_events([], [trade_a])

    with pytest.raises(ValueError, match="merged trade event pointer"):
        build_execution_tape(symbol_spec=_spec(), l2_events=[], trades=[trade_b], merged_events=plan.events)


def test_rejects_unsorted_source_arrays_or_merged_events():
    with pytest.raises(ValueError, match="l2_events local_ts_us"):
        l2 = [_l2(300, seq=0), _l2(100, seq=1)]
        merged = [
            MergedExecutionEvent(
                ref=ExecutionEventRef(
                    event_seq=0,
                    local_ts_us=l2[0].local_ts_us,
                    event_type=ExecutionEventType.L2_BATCH,
                    book_ptr=0,
                ),
                local_ts_us=l2[0].local_ts_us,
                ts_us=l2[0].max_ts_us,
                l2_event=l2[0],
            )
        ]
        build_execution_tape(symbol_spec=_spec(), l2_events=l2, trades=[], merged_events=merged)

    with pytest.raises(ValueError, match="trades local_ts_us"):
        trades = [_trade(300, idx=0), _trade(100, idx=1)]
        merged = [
            MergedExecutionEvent(
                ref=ExecutionEventRef(
                    event_seq=0,
                    local_ts_us=trades[0].local_ts_us,
                    event_type=ExecutionEventType.TRADE,
                    trade_ptr=0,
                ),
                local_ts_us=trades[0].local_ts_us,
                ts_us=trades[0].ts_us,
                trade=trades[0],
            )
        ]
        build_execution_tape(symbol_spec=_spec(), l2_events=[], trades=trades, merged_events=merged)

    l2 = [_l2(100, seq=0)]
    bad_event = MergedExecutionEvent(
        ref=ExecutionEventRef(
            event_seq=1,
            local_ts_us=l2[0].local_ts_us,
            event_type=ExecutionEventType.L2_BATCH,
            book_ptr=0,
        ),
        local_ts_us=l2[0].local_ts_us,
        ts_us=l2[0].max_ts_us,
        l2_event=l2[0],
    )
    with pytest.raises(ValueError, match="event_seq"):
        build_execution_tape(symbol_spec=_spec(), l2_events=l2, trades=[], merged_events=[bad_event])

    first = _trade(300, idx=0)
    second = _l2(100, seq=0)
    bad_merged = [
        MergedExecutionEvent(
            ref=ExecutionEventRef(
                event_seq=0,
                local_ts_us=first.local_ts_us,
                event_type=ExecutionEventType.TRADE,
                trade_ptr=0,
            ),
            local_ts_us=first.local_ts_us,
            ts_us=first.ts_us,
            trade=first,
        ),
        MergedExecutionEvent(
            ref=ExecutionEventRef(
                event_seq=1,
                local_ts_us=second.local_ts_us,
                event_type=ExecutionEventType.L2_BATCH,
                book_ptr=0,
            ),
            local_ts_us=second.local_ts_us,
            ts_us=second.max_ts_us,
            l2_event=second,
        ),
    ]
    with pytest.raises(ValueError, match="merged_events local_ts_us"):
        build_execution_tape(symbol_spec=_spec(), l2_events=[second], trades=[first], merged_events=bad_merged)


def test_save_and_load_round_trip(tmp_path):
    tape = _basic_tape()
    path = tmp_path / "tape"

    save_execution_tape(tape, path)
    loaded = load_execution_tape(path)

    assert loaded.manifest == tape.manifest
    assert np.array_equal(loaded.arrays.events, tape.arrays.events)
    assert np.array_equal(loaded.arrays.l2_events, tape.arrays.l2_events)
    assert np.array_equal(loaded.arrays.trades, tape.arrays.trades)
    assert (path / "manifest.json").exists()
    assert (path / "arrays" / "events.npy").exists()
    assert (path / "arrays" / "l2_events.npy").exists()
    assert (path / "arrays" / "trades.npy").exists()


def test_overwrite_protection(tmp_path):
    tape = _basic_tape()
    path = tmp_path / "tape"

    save_execution_tape(tape, path)
    with pytest.raises(FileExistsError):
        save_execution_tape(tape, path)
    save_execution_tape(tape, path, overwrite=True)


def test_mmap_load(tmp_path):
    tape = _basic_tape()
    path = tmp_path / "tape"
    save_execution_tape(tape, path)

    loaded = load_execution_tape(path, mmap_mode="r")

    assert isinstance(loaded.arrays.events, np.memmap)


def test_manifest_json_helpers_round_trip():
    tape = _basic_tape()

    payload = execution_tape_manifest_to_dict(tape.manifest)
    restored = execution_tape_manifest_from_dict(payload)

    assert restored == tape.manifest
    assert json.loads(json.dumps(payload))["tape_format"] == "l2_trades_arrays_v1"


def test_execution_tape_arrays_reject_invalid_event_pointers():
    tape = _basic_tape()
    events = tape.arrays.events.copy()
    events[0]["trade_ptr"] = 0

    with pytest.raises(ValueError):
        ExecutionTapeArrays(events=events, l2_events=tape.arrays.l2_events, trades=tape.arrays.trades)


def test_no_forbidden_imports():
    source = Path("mmrt/execution/execution_tape.py").read_text()
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import pyarrow" not in source
    assert "import torch" not in source
    assert "import numba" not in source
