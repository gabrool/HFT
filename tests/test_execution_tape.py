import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, ExecutionEventRef, ExecutionEventType, SymbolSpec, TradePrint
from mmrt.execution.event_merge import MergedExecutionEvent, merge_execution_events
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.metadata.rule_compatibility import RuleCompatibilityMode, RuleCompatibilityReport
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.execution.execution_tape import (
    EVENT_DTYPE,
    L2_EVENT_DTYPE,
    TRADE_DTYPE,
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    EVENTS_ARRAY_NAME,
    L2_EVENTS_ARRAY_NAME,
    TRADES_ARRAY_NAME,
    BOOK_BID_TICKS_ARRAY_NAME,
    BOOK_BID_SIZES_ARRAY_NAME,
    BOOK_ASK_TICKS_ARRAY_NAME,
    BOOK_ASK_SIZES_ARRAY_NAME,
    ExecutionTape,
    ExecutionTapeValidationMode,
    ExecutionTapeArrays,
    build_execution_tape,
    book_snapshot_to_depth_rows,
    book_snapshot_to_depth_rows_into,
    execution_tape_manifest_from_dict,
    execution_tape_manifest_to_dict,
    load_execution_tape,
    save_execution_tape,
)



def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures",
        symbol="BTCUSDT",
        mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_type="PERPETUAL",
        status="TRADING",
        tick_size=Decimal("0.1"),
        min_price=Decimal("0.1"),
        max_price=Decimal("1000000"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100"),
        min_notional=Decimal("5"),
        allowed_order_types=("LIMIT",),
        allowed_time_in_force=("GTC", "GTX"),
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


def _snapshot(local_ts_us=100):
    return BookLevelSnapshot(
        local_ts_us=local_ts_us,
        bid_ticks=(1000, 999),
        bid_sizes=(1.0, 2.0),
        ask_ticks=(1002, 1003),
        ask_sizes=(1.5, 2.5),
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
        book_snapshot=_snapshot(local),
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
    return build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=trades, merged_events=plan.events)


def _compat_report():
    return RuleCompatibilityReport(
        mode=RuleCompatibilityMode.WARN,
        price_count=3,
        price_grid_violation_count=1,
        price_grid_violation_fraction=1 / 3,
        max_abs_price_residual_ticks=0.5,
        qty_count=2,
        qty_grid_violation_count=0,
        qty_grid_violation_fraction=0.0,
        max_abs_qty_residual_steps=0.0,
        min_price_seen=100.0,
        max_price_seen=101.0,
        min_qty_seen=0.001,
        max_qty_seen=0.01,
        examples=({"kind": "price", "value": 100.05, "residual": 0.5, "source": "l2.price"},),
        status="warning",
    )


def _arrays_with(tape, **overrides):
    values = {
        "events": tape.arrays.events,
        "l2_events": tape.arrays.l2_events,
        "trades": tape.arrays.trades,
        "book_bid_ticks": tape.arrays.book_bid_ticks,
        "book_bid_sizes": tape.arrays.book_bid_sizes,
        "book_ask_ticks": tape.arrays.book_ask_ticks,
        "book_ask_sizes": tape.arrays.book_ask_sizes,
    }
    values.update(overrides)
    return ExecutionTapeArrays(**values)


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
    assert tape.manifest.array_names == (
        EVENTS_ARRAY_NAME,
        L2_EVENTS_ARRAY_NAME,
        TRADES_ARRAY_NAME,
        BOOK_BID_TICKS_ARRAY_NAME,
        BOOK_BID_SIZES_ARRAY_NAME,
        BOOK_ASK_TICKS_ARRAY_NAME,
        BOOK_ASK_SIZES_ARRAY_NAME,
    )
    assert tape.arrays.book_bid_ticks.shape == (2, 2)
    assert tape.arrays.book_bid_sizes.shape == (2, 2)
    assert tape.arrays.book_ask_ticks.shape == (2, 2)
    assert tape.arrays.book_ask_sizes.shape == (2, 2)
    assert tape.manifest.notes["book_depth"] == "2"


def test_l2_book_top_none_uses_sentinel_values():
    l2 = [_l2(100, seq=0, top=False)]
    plan = merge_execution_events(l2, [])
    tape = build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=[], merged_events=plan.events)

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
    tape = build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[], trades=trades, merged_events=plan.events, book_depth=2)

    assert tape.arrays.trades["side_code"].tolist() == [1, -1, 0]


def test_rejects_empty_tape():
    with pytest.raises(ValueError):
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[], trades=[], merged_events=[])


def test_rejects_l2_pointer_mismatch():
    l2_a = _l2(100, seq=0)
    l2_b = _l2(100, seq=0)
    plan = merge_execution_events([l2_a], [])

    with pytest.raises(ValueError, match="merged L2 event pointer"):
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[l2_b], trades=[], merged_events=plan.events)


def test_rejects_trade_pointer_mismatch():
    trade_a = _trade(100, idx=0)
    trade_b = _trade(100, idx=0)
    plan = merge_execution_events([], [trade_a])

    with pytest.raises(ValueError, match="merged trade event pointer"):
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[], trades=[trade_b], merged_events=plan.events, book_depth=2)


def test_rejects_unreferenced_l2_event():
    l2_a = _l2(100, seq=0)
    l2_b = _l2(200, seq=1)
    plan = merge_execution_events([l2_a], [])

    with pytest.raises(ValueError, match="reference every l2_event"):
        build_execution_tape(
            symbol_spec=_spec(),
            symbol_rules=_rules(),
            l2_events=[l2_a, l2_b],
            trades=[],
            merged_events=plan.events,
        )


def test_rejects_unreferenced_trade():
    trade_a = _trade(100, idx=0)
    trade_b = _trade(200, idx=1)
    plan = merge_execution_events([], [trade_a])

    with pytest.raises(ValueError, match="reference every trade"):
        build_execution_tape(
            symbol_spec=_spec(),
            symbol_rules=_rules(),
            l2_events=[],
            trades=[trade_a, trade_b],
            merged_events=plan.events,
            book_depth=2,
        )


def test_rejects_duplicate_l2_book_ptr():
    l2 = [_l2(100, seq=0)]

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
        ),
        MergedExecutionEvent(
            ref=ExecutionEventRef(
                event_seq=1,
                local_ts_us=l2[0].local_ts_us,
                event_type=ExecutionEventType.L2_BATCH,
                book_ptr=0,
            ),
            local_ts_us=l2[0].local_ts_us,
            ts_us=l2[0].max_ts_us,
            l2_event=l2[0],
        ),
    ]

    with pytest.raises(ValueError, match="duplicate L2 book_ptr"):
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=[], merged_events=merged)


def test_rejects_duplicate_trade_ptr():
    trades = [_trade(100, idx=0)]

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
        ),
        MergedExecutionEvent(
            ref=ExecutionEventRef(
                event_seq=1,
                local_ts_us=trades[0].local_ts_us,
                event_type=ExecutionEventType.TRADE,
                trade_ptr=0,
            ),
            local_ts_us=trades[0].local_ts_us,
            ts_us=trades[0].ts_us,
            trade=trades[0],
        ),
    ]

    with pytest.raises(ValueError, match="duplicate trade_ptr"):
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[], trades=trades, merged_events=merged, book_depth=2)


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
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=[], merged_events=merged)

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
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[], trades=trades, merged_events=merged, book_depth=2)

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
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=[], merged_events=[bad_event])

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
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=[second], trades=[first], merged_events=bad_merged)


def test_save_and_load_round_trip(tmp_path):
    tape = _basic_tape()
    path = tmp_path / "tape"

    save_execution_tape(tape, path)
    loaded = load_execution_tape(path)

    assert loaded.manifest == tape.manifest
    assert np.array_equal(loaded.arrays.events, tape.arrays.events)
    assert np.array_equal(loaded.arrays.l2_events, tape.arrays.l2_events)
    assert np.array_equal(loaded.arrays.trades, tape.arrays.trades)
    assert np.array_equal(loaded.arrays.book_bid_ticks, tape.arrays.book_bid_ticks)
    assert np.array_equal(loaded.arrays.book_bid_sizes, tape.arrays.book_bid_sizes)
    assert np.array_equal(loaded.arrays.book_ask_ticks, tape.arrays.book_ask_ticks)
    assert np.array_equal(loaded.arrays.book_ask_sizes, tape.arrays.book_ask_sizes)
    assert (path / "manifest.json").exists()
    assert (path / "arrays" / "events.npy").exists()
    assert (path / "arrays" / "l2_events.npy").exists()
    assert (path / "arrays" / "trades.npy").exists()
    assert (path / "arrays" / "book_bid_ticks.npy").exists()
    assert (path / "arrays" / "book_bid_sizes.npy").exists()
    assert (path / "arrays" / "book_ask_ticks.npy").exists()
    assert (path / "arrays" / "book_ask_sizes.npy").exists()


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
    assert json.loads(json.dumps(payload))["tape_format"] == "l2_trades_arrays"



def test_manifest_roundtrip_preserves_symbol_rule_compatibility_report():
    l2 = [_l2(100, seq=0), _l2(300, seq=1)]
    trades = [_trade(200, idx=0)]
    plan = merge_execution_events(l2, trades)
    tape = build_execution_tape(
        symbol_spec=_spec(),
        symbol_rules=_rules(),
        symbol_rule_compatibility=_compat_report(),
        l2_events=l2,
        trades=trades,
        merged_events=plan.events,
    )

    payload = execution_tape_manifest_to_dict(tape.manifest)
    restored = execution_tape_manifest_from_dict(payload)

    assert restored.symbol_rule_compatibility == tape.manifest.symbol_rule_compatibility
    assert payload["symbol_rule_compatibility"]["status"] == "warning"


def test_execution_tape_arrays_reject_invalid_event_pointers():
    tape = _basic_tape()
    events = tape.arrays.events.copy()
    events[0]["trade_ptr"] = 0

    with pytest.raises(ValueError):
        _arrays_with(tape, events=events)


def test_execution_tape_arrays_reject_duplicate_l2_pointer():
    tape = _basic_tape()
    events = tape.arrays.events.copy()

    # Make event 2 point to the same L2 row as event 0.
    events[2]["book_ptr"] = events[0]["book_ptr"]

    with pytest.raises(ValueError, match="duplicate L2 book_ptr"):
        _arrays_with(tape, events=events)


def test_execution_tape_arrays_reject_duplicate_trade_pointer():
    tape = _basic_tape()
    events = tape.arrays.events.copy()

    # Make event 3 point to the same trade row as event 1.
    events[3]["trade_ptr"] = events[1]["trade_ptr"]

    with pytest.raises(ValueError, match="duplicate trade_ptr"):
        _arrays_with(tape, events=events)


def test_execution_tape_arrays_reject_unreferenced_l2_row():
    tape = _basic_tape()
    events = tape.arrays.events.copy()
    l2_events = np.concatenate([tape.arrays.l2_events, tape.arrays.l2_events[:1]])
    book_bid_ticks = np.concatenate([tape.arrays.book_bid_ticks, tape.arrays.book_bid_ticks[:1]])
    book_bid_sizes = np.concatenate([tape.arrays.book_bid_sizes, tape.arrays.book_bid_sizes[:1]])
    book_ask_ticks = np.concatenate([tape.arrays.book_ask_ticks, tape.arrays.book_ask_ticks[:1]])
    book_ask_sizes = np.concatenate([tape.arrays.book_ask_sizes, tape.arrays.book_ask_sizes[:1]])

    with pytest.raises(ValueError, match="reference every l2_event"):
        _arrays_with(
            tape,
            events=events,
            l2_events=l2_events,
            book_bid_ticks=book_bid_ticks,
            book_bid_sizes=book_bid_sizes,
            book_ask_ticks=book_ask_ticks,
            book_ask_sizes=book_ask_sizes,
        )


def test_execution_tape_arrays_reject_unreferenced_trade_row():
    tape = _basic_tape()
    events = tape.arrays.events.copy()
    trades = np.concatenate([tape.arrays.trades, tape.arrays.trades[:1]])

    with pytest.raises(ValueError, match="reference every trade"):
        _arrays_with(tape, events=events, trades=trades)


def test_no_forbidden_imports():
    source = Path("mmrt/execution/execution_tape.py").read_text()
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import pyarrow" not in source
    assert "import torch" not in source
    assert "import numba" not in source


def test_missing_book_snapshot_rejected():
    l2 = [_l2(100, seq=0)]
    bad_l2 = [ReconstructedL2Event(
        batch_seq=l2[0].batch_seq,
        local_ts_us=l2[0].local_ts_us,
        min_ts_us=l2[0].min_ts_us,
        max_ts_us=l2[0].max_ts_us,
        num_updates=l2[0].num_updates,
        is_snapshot_batch=l2[0].is_snapshot_batch,
        book_top=l2[0].book_top,
        bid_depth=l2[0].bid_depth,
        ask_depth=l2[0].ask_depth,
        book_snapshot=None,
    )]
    plan = merge_execution_events(bad_l2, [])

    with pytest.raises(ValueError, match="book_snapshot"):
        build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=bad_l2, trades=[], merged_events=plan.events)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"book_bid_sizes": np.zeros((2, 3), dtype=np.float32)}, "same shape"),
        ({"book_bid_ticks": np.array([[1000, 0, 999], [1000, 999, 0]], dtype=np.int64),
          "book_bid_sizes": np.array([[1.0, 0.0, 2.0], [1.0, 2.0, 0.0]], dtype=np.float32),
          "book_ask_ticks": np.array([[1002, 1003, 0], [1002, 1003, 0]], dtype=np.int64),
          "book_ask_sizes": np.array([[1.5, 2.5, 0.0], [1.5, 2.5, 0.0]], dtype=np.float32)}, "zero padding"),
        ({"book_bid_sizes": np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)}, "exactly when tick"),
        ({"book_bid_ticks": np.array([[1000, 0], [1000, 999]], dtype=np.int64),
          "book_bid_sizes": np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)}, "exactly when tick"),
        ({"book_bid_ticks": np.array([[999, 1000], [1000, 999]], dtype=np.int64)}, "descending"),
        ({"book_ask_ticks": np.array([[1003, 1002], [1002, 1003]], dtype=np.int64)}, "ascending"),
    ],
)
def test_invalid_book_depth_arrays_rejected(override, match):
    tape = _basic_tape()

    with pytest.raises(ValueError, match=match):
        _arrays_with(tape, **override)


def test_explicit_book_depth_pads_snapshots():
    l2 = [_l2(100, seq=0), _l2(200, seq=1)]
    plan = merge_execution_events(l2, [])

    tape = build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=[], merged_events=plan.events, book_depth=3)

    assert tape.arrays.book_bid_ticks.shape == (2, 3)
    assert tape.arrays.book_bid_ticks[0].tolist() == [1000, 999, 0]
    assert tape.arrays.book_bid_sizes[0].tolist() == [1.0, 2.0, 0.0]
    assert tape.manifest.schema == "mmrt_execution_tape_book_depth"
    assert tape.manifest.notes["book_depth"] == "3"


def test_book_snapshot_to_depth_rows_into_matches_allocating_helper():
    event = _l2(100, seq=0)
    expected = book_snapshot_to_depth_rows(event, book_depth=3)
    bid_ticks = np.full((3,), -99, dtype=np.int64)
    bid_sizes = np.full((3,), -99.0, dtype=np.float32)
    ask_ticks = np.full((3,), -99, dtype=np.int64)
    ask_sizes = np.full((3,), -99.0, dtype=np.float32)

    book_snapshot_to_depth_rows_into(
        event,
        book_depth=3,
        bid_ticks=bid_ticks,
        bid_sizes=bid_sizes,
        ask_ticks=ask_ticks,
        ask_sizes=ask_sizes,
    )

    for actual, want in zip((bid_ticks, bid_sizes, ask_ticks, ask_sizes), expected, strict=True):
        np.testing.assert_array_equal(actual, want)


def test_load_execution_tape_defaults_to_full_for_in_memory_load(tmp_path):
    tape = _basic_tape()
    root = tmp_path / "tape"
    save_execution_tape(tape, root)

    loaded = load_execution_tape(root)

    assert loaded.arrays.validation_mode is ExecutionTapeValidationMode.FULL


def test_load_execution_tape_defaults_to_shape_only_for_mmap_load(tmp_path):
    tape = _basic_tape()
    root = tmp_path / "tape"
    save_execution_tape(tape, root)

    loaded = load_execution_tape(root, mmap_mode="r")

    assert loaded.arrays.validation_mode is ExecutionTapeValidationMode.SHAPE_ONLY
    assert isinstance(loaded.arrays.events, np.memmap)


def test_load_execution_tape_shape_only_skips_full_event_scan(tmp_path, monkeypatch):
    tape = _basic_tape()
    root = tmp_path / "tape"
    save_execution_tape(tape, root)

    import mmrt.execution.execution_tape as execution_tape_module

    def fail_full(*args, **kwargs):
        raise AssertionError("full validator should not run")

    monkeypatch.setattr(execution_tape_module, "_validate_events_array_full", fail_full)
    monkeypatch.setattr(execution_tape_module, "_validate_book_depth_arrays_full", fail_full)

    loaded = load_execution_tape(root, mmap_mode="r", validation_mode="shape_only")

    assert loaded.manifest.num_events == tape.manifest.num_events
    assert loaded.arrays.validation_mode is ExecutionTapeValidationMode.SHAPE_ONLY


def test_load_execution_tape_full_calls_full_validators(tmp_path, monkeypatch):
    tape = _basic_tape()
    root = tmp_path / "tape"
    save_execution_tape(tape, root)
    flags = {"events": False, "book": False}

    import mmrt.execution.execution_tape as execution_tape_module

    def events_full(events, num_l2_events, num_trades):
        flags["events"] = True

    def book_full(book_bid_ticks, book_bid_sizes, book_ask_ticks, book_ask_sizes, *, num_l2_events):
        flags["book"] = True

    monkeypatch.setattr(execution_tape_module, "_validate_events_array_full", events_full)
    monkeypatch.setattr(execution_tape_module, "_validate_book_depth_arrays_full", book_full)

    loaded = load_execution_tape(root, mmap_mode="r", validation_mode="full")

    assert loaded.arrays.validation_mode is ExecutionTapeValidationMode.FULL
    assert flags == {"events": True, "book": True}


def test_shape_only_still_rejects_bad_shapes(tmp_path):
    tape = _basic_tape()
    root = tmp_path / "tape"
    save_execution_tape(tape, root)
    np.save(root / "arrays" / "book_bid_sizes.npy", np.zeros((len(tape.arrays.l2_events), 3), dtype=np.float32))

    with pytest.raises(ValueError, match="same shape"):
        load_execution_tape(root, mmap_mode="r", validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY)


def test_shape_only_validation_does_not_call_full_validators_for_largeish_tape(tmp_path, monkeypatch):
    n = 10_000
    depth = 25
    root = tmp_path / "tape"
    arrays_dir = root / "arrays"
    arrays_dir.mkdir(parents=True)
    events = np.empty(n * 2, dtype=EVENT_DTYPE)
    events[0::2]["event_seq"] = np.arange(0, n * 2, 2, dtype=np.int64)
    events[1::2]["event_seq"] = np.arange(1, n * 2, 2, dtype=np.int64)
    events["local_ts_us"] = np.arange(100, 100 + n * 2, dtype=np.int64)
    events["ts_us"] = events["local_ts_us"]
    events[0::2]["event_type_code"] = EVENT_TYPE_CODE_L2_BATCH
    events[0::2]["book_ptr"] = np.arange(n, dtype=np.int64)
    events[0::2]["trade_ptr"] = -1
    events[1::2]["event_type_code"] = EVENT_TYPE_CODE_TRADE
    events[1::2]["book_ptr"] = -1
    events[1::2]["trade_ptr"] = np.arange(n, dtype=np.int64)
    np.save(arrays_dir / "events.npy", events)
    np.save(arrays_dir / "l2_events.npy", np.zeros(n, dtype=L2_EVENT_DTYPE))
    np.save(arrays_dir / "trades.npy", np.zeros(n, dtype=TRADE_DTYPE))
    np.save(arrays_dir / "book_bid_ticks.npy", np.zeros((n, depth), dtype=np.int64))
    np.save(arrays_dir / "book_bid_sizes.npy", np.zeros((n, depth), dtype=np.float32))
    np.save(arrays_dir / "book_ask_ticks.npy", np.zeros((n, depth), dtype=np.int64))
    np.save(arrays_dir / "book_ask_sizes.npy", np.zeros((n, depth), dtype=np.float32))
    manifest = _basic_tape().manifest
    manifest = type(manifest)(
        schema=manifest.schema,
        tape_format=manifest.tape_format,
        exchange=manifest.exchange,
        symbol=manifest.symbol,
        symbol_spec=manifest.symbol_spec,
        symbol_rules=manifest.symbol_rules,
        source_data_types=manifest.source_data_types,
        array_names=manifest.array_names,
        num_events=len(events),
        num_l2_batches=n,
        num_trades=n,
        num_decisions=0,
        start_local_ts_us=int(events["local_ts_us"][0]),
        end_local_ts_us=int(events["local_ts_us"][-1]),
        created_at_utc=manifest.created_at_utc,
        symbol_rule_compatibility=manifest.symbol_rule_compatibility,
        notes={"book_depth": str(depth)},
    )
    (root / "manifest.json").write_text(json.dumps(execution_tape_manifest_to_dict(manifest)), encoding="utf-8")

    import mmrt.execution.execution_tape as execution_tape_module

    def fail_full(*args, **kwargs):
        raise AssertionError("full validator should not run")

    monkeypatch.setattr(execution_tape_module, "_validate_events_array_full", fail_full)
    monkeypatch.setattr(execution_tape_module, "_validate_book_depth_arrays_full", fail_full)

    loaded = load_execution_tape(root, mmap_mode="r", validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY)

    assert loaded.arrays.events.shape == (n * 2,)
    assert loaded.arrays.book_bid_ticks.shape == (n, depth)
