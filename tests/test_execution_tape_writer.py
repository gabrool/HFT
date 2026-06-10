from decimal import Decimal

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
from mmrt.execution.event_merge import iter_merged_execution_events, merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape, load_execution_tape
from mmrt.execution.execution_tape_writer import NpyChunkWriter, StreamingExecutionTapeWriter, StreamingExecutionTapeWriterConfig
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode


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


def _snapshot(local_ts_us: int, seq: int = 0):
    return BookLevelSnapshot(
        local_ts_us=local_ts_us,
        bid_ticks=(1000 + seq, 999 + seq),
        bid_sizes=(1.0 + seq, 2.0 + seq),
        ask_ticks=(1002 + seq, 1003 + seq),
        ask_sizes=(1.5 + seq, 2.5 + seq),
    )


def _l2(local: int, seq: int = 0):
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local,
        min_ts_us=local - 10,
        max_ts_us=local - 5,
        num_updates=2,
        is_snapshot_batch=(seq == 0),
        book_top=BookTop(local, 1000 + seq, 1002 + seq, 1.0, 2.0),
        bid_depth=2,
        ask_depth=2,
        book_snapshot=_snapshot(local, seq),
    )


def _trade(local: int, idx: int = 0):
    return TradePrint(local, local - 1, AggressorSide.BUY, 1001 + idx, 0.01, str(idx), idx)


def test_npy_chunk_writer_finalizes_exact_array_with_small_chunks(tmp_path):
    writer = NpyChunkWriter("values", np.dtype("<i8"), (), 2, tmp_path / "chunks")
    assert [writer.append(i) for i in range(5)] == [0, 1, 2, 3, 4]
    total = writer.finalize(tmp_path / "values.npy")
    assert total == 5
    np.testing.assert_array_equal(np.load(tmp_path / "values.npy"), np.arange(5, dtype=np.int64))
    assert len(list((tmp_path / "chunks").glob("values_*.npy"))) == 3
    writer.cleanup()
    assert list((tmp_path / "chunks").glob("values_*.npy")) == []


def test_streaming_execution_tape_writer_matches_materialized_tape(tmp_path):
    l2 = [_l2(100, 0), _l2(300, 1)]
    trades = [_trade(200, 0), _trade(400, 1)]
    plan = merge_execution_events(l2, trades)
    materialized = build_execution_tape(
        symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=trades, merged_events=plan.events, book_depth=2
    )
    writer = StreamingExecutionTapeWriter(
        StreamingExecutionTapeWriterConfig(
            output_root=str(tmp_path / "streamed"), symbol_spec=_spec(), symbol_rules=_rules(), book_depth=2, chunk_rows=1
        )
    )
    for event in iter_merged_execution_events(l2, trades):
        writer.append(event)
    streamed = writer.finalize().tape
    np.testing.assert_array_equal(streamed.arrays.events, materialized.arrays.events)
    np.testing.assert_array_equal(streamed.arrays.l2_events, materialized.arrays.l2_events)
    np.testing.assert_array_equal(streamed.arrays.trades, materialized.arrays.trades)
    np.testing.assert_array_equal(streamed.arrays.book_bid_ticks, materialized.arrays.book_bid_ticks)
    np.testing.assert_array_equal(streamed.arrays.book_bid_sizes, materialized.arrays.book_bid_sizes)
    np.testing.assert_array_equal(streamed.arrays.book_ask_ticks, materialized.arrays.book_ask_ticks)
    np.testing.assert_array_equal(streamed.arrays.book_ask_sizes, materialized.arrays.book_ask_sizes)
    assert streamed.manifest.num_events == materialized.manifest.num_events
    assert streamed.manifest.start_local_ts_us == materialized.manifest.start_local_ts_us
    assert streamed.manifest.end_local_ts_us == materialized.manifest.end_local_ts_us


def test_streaming_writer_flushes_multiple_book_chunks(tmp_path):
    l2 = [_l2(100, 0), _l2(200, 1), _l2(300, 2)]
    trades = [_trade(150, 0)]
    writer = StreamingExecutionTapeWriter(
        StreamingExecutionTapeWriterConfig(
            output_root=str(tmp_path / "tape"), symbol_spec=_spec(), symbol_rules=_rules(), book_depth=2, chunk_rows=1
        )
    )
    for event in iter_merged_execution_events(l2, trades):
        writer.append(event)
    tape = writer.finalize().tape
    assert tape.arrays.book_bid_ticks.shape == (3, 2)
    np.testing.assert_array_equal(tape.arrays.book_bid_ticks[:, 0], np.array([1000, 1001, 1002]))


def test_streaming_writer_writes_manifest_last_or_no_manifest_on_failure(tmp_path):
    writer = StreamingExecutionTapeWriter(
        StreamingExecutionTapeWriterConfig(
            output_root=str(tmp_path / "tape"), symbol_spec=_spec(), symbol_rules=_rules(), book_depth=2, chunk_rows=1
        )
    )
    writer.append(next(iter_merged_execution_events([_l2(100)], [])))
    with pytest.raises(ValueError):
        writer.finalize()
    assert not (tmp_path / "tape" / "manifest.json").exists()


def test_streaming_build_output_loads_with_mmap(tmp_path):
    writer = StreamingExecutionTapeWriter(
        StreamingExecutionTapeWriterConfig(
            output_root=str(tmp_path / "tape"), symbol_spec=_spec(), symbol_rules=_rules(), book_depth=2, chunk_rows=1
        )
    )
    for event in iter_merged_execution_events([_l2(100)], [_trade(150)]):
        writer.append(event)
    writer.finalize()
    tape = load_execution_tape(tmp_path / "tape", mmap_mode="r")
    assert isinstance(tape.arrays.events, np.memmap)
