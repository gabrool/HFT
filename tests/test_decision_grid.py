from decimal import Decimal
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
from mmrt.execution.decision_grid import (
    DECISION_GRID_SCHEMA,
    load_decision_grid,
    validate_decision_grid_start_event_index,
    validate_decision_grid_for_execution_tape,
)
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape, save_execution_tape
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.features.schedule import (
    DECISION_REASON_FIRST_VALID_BOOK,
    DECISION_REASON_HEARTBEAT,
    DECISION_REASON_TOP_OF_BOOK_WAKE,
    DECISION_REASON_TRADE_WAKE,
)
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.cli.build_decision_grid import BuildDecisionGridConfig, build_decision_grid_from_config
from mmrt.cli.build_decision_grid import _DecisionGridWriters


def _workspace_tmp(name: str) -> Path:
    root = Path("work") / "test_decision_grid_runtime" / f"{name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


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
    return SymbolSpec("binance-futures", "BTCUSDT", 0.1, 0.001, 0.001, 100.0, 5.0)


def _l2(seq: int, local_ts_us: int, *, bid_size: float = 1.0) -> ReconstructedL2Event:
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us,
        max_ts_us=local_ts_us,
        num_updates=1,
        is_snapshot_batch=(seq == 0),
        book_top=BookTop(local_ts_us, 1000, 1002, bid_size, 1.2),
        bid_depth=2,
        ask_depth=2,
        book_snapshot=BookLevelSnapshot(
            local_ts_us,
            (1000, 999),
            (bid_size, 2.0),
            (1002, 1003),
            (1.2, 2.2),
        ),
    )


def _trade(local_ts_us: int, source_row: int = 0) -> TradePrint:
    return TradePrint(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us,
        side=AggressorSide.BUY,
        price_tick=1002,
        amount=0.01,
        source_row=source_row,
    )


def _tape():
    l2 = [
        _l2(0, 100),
        _l2(1, 200),
        _l2(2, 250, bid_size=2.0),
        _l2(3, 300, bid_size=2.0),
        _l2(4, 800, bid_size=2.0),
    ]
    trades = [_trade(150)]
    merged = merge_execution_events(l2, trades).events
    return build_execution_tape(
        symbol_spec=_spec(),
        symbol_rules=_rules(),
        l2_events=l2,
        trades=trades,
        merged_events=merged,
        book_depth=2,
        created_at_utc="2026-01-01T00:00:00Z",
    )


def test_build_decision_grid_records_real_rows_hash_and_reasons():
    tmp_path = _workspace_tmp("reasons")
    tape = _tape()
    try:
        tape_root = tmp_path / "tape"
        save_execution_tape(tape, tape_root, overwrite=True)

        grid_path = tape_root / "decision_grid"
        summary_json = tape_root / "decision_grid_summary.json"
        summary = build_decision_grid_from_config(
            BuildDecisionGridConfig(
                tape_root=str(tape_root),
                output_grid=str(grid_path),
                output_json=str(summary_json),
                min_decision_interval_us=100,
                max_decision_interval_us=500,
                l1_size_change_fraction=0.25,
                overwrite=True,
            )
        )

        grid = load_decision_grid(grid_path)
        validate_decision_grid_for_execution_tape(grid, tape)
        loaded_again = load_decision_grid(grid_path)

        assert grid.metadata.schema == DECISION_GRID_SCHEMA
        assert (grid_path / "manifest.json").exists()
        assert (grid_path / "arrays" / "decision_event_index.npy").exists()
        assert not (grid_path / "chunks").exists()
        assert grid.decision_grid_hash == loaded_again.decision_grid_hash
        assert grid.n_rows == 4
        assert summary["decision_grid"]["decision_grid_hash"] == grid.decision_grid_hash
        assert set(summary["interval_stats"]) == {
            "elapsed_since_prev_decision_us",
            "events_since_prev_decision",
            "l2_events_since_prev_decision",
            "trade_events_since_prev_decision",
        }
        for stats in summary["interval_stats"].values():
            assert list(stats) == ["count", "mean", "min", "max"]
            assert stats["count"] == grid.n_rows - 1
        assert list(grid.reason_code) == [
            DECISION_REASON_FIRST_VALID_BOOK,
            DECISION_REASON_TRADE_WAKE,
            DECISION_REASON_TOP_OF_BOOK_WAKE,
            DECISION_REASON_HEARTBEAT,
        ]
        assert np.array_equal(grid.decision_event_seq, tape.arrays.events["event_seq"][grid.decision_event_index])
        assert np.all(grid.decision_event_seq < 9_223_372_036_854_775_807)
        start = validate_decision_grid_start_event_index(grid, start_event_index=int(grid.decision_event_index[1]), min_rows=2)
        assert start.as_dict() == {
            "event_index": int(grid.decision_event_index[1]),
            "decision_grid_row_index": 1,
            "rows_available": 3,
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_decision_grid_validation_rejects_tape_mismatch():
    tmp_path = _workspace_tmp("mismatch")
    tape = _tape()
    try:
        tape_root = tmp_path / "tape"
        save_execution_tape(tape, tape_root, overwrite=True)
        build_decision_grid_from_config(
            BuildDecisionGridConfig(
                tape_root=str(tape_root),
                min_decision_interval_us=100,
                max_decision_interval_us=500,
                overwrite=True,
            )
        )
        grid = load_decision_grid(tape_root / "decision_grid")

        bad_tape = _tape()
        bad_tape.arrays.events["event_seq"][int(grid.decision_event_index[0])] += 100
        try:
            validate_decision_grid_for_execution_tape(grid, bad_tape, mode="full")
        except ValueError as exc:
            assert "event_seq" in str(exc)
        else:
            raise AssertionError("expected tape/grid mismatch")
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_decision_grid_writers_buffer_rows_before_chunk_flush(tmp_path):
    writers = _DecisionGridWriters(tmp_path / "grid", chunk_rows=3)
    try:
        for i in range(2):
            writers.append(
                decision_event_index=i,
                decision_local_ts_us=100 + i,
                decision_event_seq=i,
                book_ptr=i,
                reason_code=1,
                reason_flags=1,
                elapsed_since_prev_decision_us=i,
                events_since_prev_decision=i,
                l2_events_since_prev_decision=i,
                trade_events_since_prev_decision=0,
            )

        assert writers.total_rows == 2
        assert writers.writers["decision_event_index"].total_rows == 0
        arrays = writers.finalize_arrays()
        assert writers.writers["decision_event_index"].total_rows == 2
        np.testing.assert_array_equal(arrays["decision_event_index"], np.array([0, 1], dtype=np.int64))
    finally:
        writers.cleanup()
