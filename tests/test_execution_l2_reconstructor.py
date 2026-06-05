import pytest

from mmrt.contracts import BookSide
from mmrt.execution.contracts import L2Update, L2UpdateBatch, SymbolSpec
from mmrt.execution.l2_reconstructor import (
    L2BookReconstructor,
    L2ReconstructionStatus,
    iter_l2_update_batches,
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


def _u(local, ts, side, tick, amount, snap=False, row=0):
    return L2Update(
        local_ts_us=local,
        ts_us=ts,
        side=side,
        price_tick=tick,
        amount=amount,
        is_snapshot=snap,
        source_row=row,
    )


def _batch(local, updates, seq=0):
    return L2UpdateBatch(
        local_ts_us=local,
        min_ts_us=min(update.ts_us for update in updates),
        max_ts_us=max(update.ts_us for update in updates),
        updates=tuple(updates),
        is_snapshot_batch=any(update.is_snapshot for update in updates),
        batch_seq=seq,
    )


def _snapshot_batch(local=100, seq=0):
    return _batch(
        local,
        [
            _u(local, 90, BookSide.BID, 1000, 1.0, snap=True, row=1),
            _u(local, 91, BookSide.BID, 999, 2.0, snap=True, row=2),
            _u(local, 92, BookSide.ASK, 1002, 1.5, snap=True, row=3),
            _u(local, 93, BookSide.ASK, 1003, 2.5, snap=True, row=4),
        ],
        seq=seq,
    )


def test_initial_status_and_pre_snapshot_skip():
    reconstructor = L2BookReconstructor(_spec())

    assert reconstructor.status is L2ReconstructionStatus.WAITING_FOR_SNAPSHOT
    event = reconstructor.apply_batch(_batch(100, [_u(100, 90, BookSide.BID, 1000, 1.0)]))

    assert event is None
    assert reconstructor.counters.skipped_pre_snapshot_updates == 1
    assert reconstructor.status is L2ReconstructionStatus.WAITING_FOR_SNAPSHOT


def test_snapshot_batch_initializes_book_atomically():
    reconstructor = L2BookReconstructor(_spec())

    event = reconstructor.apply_batch(_snapshot_batch())

    assert event is not None
    assert reconstructor.status is L2ReconstructionStatus.READY
    assert event.book_top is not None
    assert event.book_top.best_bid_tick == 1000
    assert event.book_top.best_ask_tick == 1002
    assert reconstructor.counters.snapshot_reset_count == 1
    assert reconstructor.counters.applied_update_count == 4
    assert event.bid_depth == 2
    assert event.ask_depth == 2
    snapshot = reconstructor.snapshot(depth=2)
    assert snapshot.bid_ticks == (1000, 999)
    assert snapshot.bid_sizes == (1.0, 2.0)
    assert snapshot.ask_ticks == (1002, 1003)
    assert snapshot.ask_sizes == (1.5, 2.5)


def test_update_amount_is_absolute_not_delta():
    reconstructor = L2BookReconstructor(_spec())
    reconstructor.apply_batch(_snapshot_batch())

    reconstructor.apply_batch(_batch(101, [_u(101, 94, BookSide.BID, 1000, 5.0)], seq=1))

    snapshot = reconstructor.snapshot(depth=2)
    assert snapshot.bid_ticks[0] == 1000
    assert snapshot.bid_sizes[0] == 5.0


def test_amount_zero_deletes_level_and_counts_missing_delete():
    reconstructor = L2BookReconstructor(_spec())
    reconstructor.apply_batch(_snapshot_batch())

    reconstructor.apply_batch(_batch(101, [_u(101, 94, BookSide.BID, 999, 0.0)], seq=1))
    assert 999 not in reconstructor.snapshot(depth=5).bid_ticks
    assert reconstructor.counters.deleted_level_count == 1

    reconstructor.apply_batch(_batch(102, [_u(102, 95, BookSide.BID, 900, 0.0)], seq=2))
    assert reconstructor.counters.missing_delete_count == 1


def test_new_snapshot_after_updates_resets_book():
    reconstructor = L2BookReconstructor(_spec())
    reconstructor.apply_batch(
        _batch(
            100,
            [
                _u(100, 90, BookSide.BID, 1000, 1.0, snap=True),
                _u(100, 91, BookSide.ASK, 1002, 1.0, snap=True),
            ],
        )
    )
    reconstructor.apply_batch(
        _batch(
            101,
            [
                _u(101, 92, BookSide.BID, 999, 1.0),
                _u(101, 93, BookSide.ASK, 1003, 1.0),
            ],
            seq=1,
        )
    )

    reconstructor.apply_batch(
        _batch(
            102,
            [
                _u(102, 94, BookSide.BID, 900, 1.0, snap=True),
                _u(102, 95, BookSide.ASK, 902, 1.0, snap=True),
            ],
            seq=2,
        )
    )

    snapshot = reconstructor.snapshot(depth=10)
    assert snapshot.bid_ticks == (900,)
    assert snapshot.ask_ticks == (902,)
    top = reconstructor.book_top()
    assert top is not None
    assert top.best_bid_tick == 900
    assert top.best_ask_tick == 902


def test_local_timestamps_must_be_strictly_increasing_per_batch():
    reconstructor = L2BookReconstructor(_spec())
    reconstructor.apply_batch(_snapshot_batch(local=100))

    with pytest.raises(ValueError):
        reconstructor.apply_batch(_batch(100, [_u(100, 94, BookSide.BID, 1001, 1.0)], seq=1))
    assert reconstructor.counters.local_ts_decrease_count == 1

    with pytest.raises(ValueError):
        reconstructor.apply_batch(_batch(99, [_u(99, 95, BookSide.BID, 1001, 1.0)], seq=2))
    assert reconstructor.counters.local_ts_decrease_count == 2


def test_crossed_repair_when_bid_update_crosses_ask():
    reconstructor = L2BookReconstructor(_spec())
    reconstructor.apply_batch(
        _batch(
            100,
            [
                _u(100, 90, BookSide.BID, 1000, 1.0, snap=True),
                _u(100, 91, BookSide.ASK, 1002, 1.0, snap=True),
                _u(100, 92, BookSide.ASK, 1004, 1.0, snap=True),
            ],
        )
    )

    event = reconstructor.apply_batch(_batch(101, [_u(101, 93, BookSide.BID, 1003, 1.0)], seq=1))

    assert event is not None
    assert event.book_top is not None
    assert event.book_top.best_bid_tick == 1003
    assert event.book_top.best_ask_tick == 1004
    assert event.crossed_repaired is True
    assert event.crossed_levels_removed == 1
    assert reconstructor.counters.crossed_repair_count == 1
    assert reconstructor.counters.crossed_levels_removed == 1


def test_crossed_repair_when_ask_update_crosses_bid():
    reconstructor = L2BookReconstructor(_spec())
    reconstructor.apply_batch(
        _batch(
            100,
            [
                _u(100, 90, BookSide.BID, 1000, 1.0, snap=True),
                _u(100, 91, BookSide.BID, 998, 1.0, snap=True),
                _u(100, 92, BookSide.ASK, 1002, 1.0, snap=True),
            ],
        )
    )

    event = reconstructor.apply_batch(_batch(101, [_u(101, 93, BookSide.ASK, 999, 1.0)], seq=1))

    assert event is not None
    assert event.book_top is not None
    assert event.book_top.best_bid_tick == 998
    assert event.book_top.best_ask_tick == 999


def test_one_sided_book_top_returns_none_and_snapshot_still_works():
    reconstructor = L2BookReconstructor(_spec())

    event = reconstructor.apply_batch(
        _batch(
            100,
            [
                _u(100, 90, BookSide.BID, 1000, 1.0, snap=True),
                _u(100, 91, BookSide.BID, 999, 2.0, snap=True),
            ],
        )
    )

    assert event is not None
    assert event.book_top is None
    assert reconstructor.book_top() is None
    snapshot = reconstructor.snapshot(depth=5)
    assert snapshot.bid_ticks == (1000, 999)
    assert snapshot.ask_ticks == ()


def test_iterator_groups_updates_by_local_timestamp():
    updates = [
        _u(100, 90, BookSide.BID, 1000, 1.0, snap=True, row=0),
        _u(100, 92, BookSide.ASK, 1002, 1.0, row=1),
        _u(101, 91, BookSide.BID, 999, 2.0, row=2),
        _u(103, 95, BookSide.ASK, 1003, 1.5, row=3),
        _u(103, 94, BookSide.BID, 998, 2.5, row=4),
    ]

    batches = list(iter_l2_update_batches(updates))

    assert len(batches) == 3
    assert [batch.batch_seq for batch in batches] == [0, 1, 2]
    assert batches[0].updates == tuple(updates[:2])
    assert batches[1].updates == (updates[2],)
    assert batches[2].updates == tuple(updates[3:])
    assert batches[0].min_ts_us == 90
    assert batches[0].max_ts_us == 92
    assert batches[2].min_ts_us == 94
    assert batches[2].max_ts_us == 95
    assert batches[0].is_snapshot_batch is True
    assert batches[1].is_snapshot_batch is False


def test_iterator_rejects_unsorted_local_timestamp():
    updates = [
        _u(100, 90, BookSide.BID, 1000, 1.0),
        _u(101, 91, BookSide.BID, 999, 1.0),
        _u(100, 92, BookSide.BID, 998, 1.0),
    ]

    with pytest.raises(ValueError):
        list(iter_l2_update_batches(updates))


def test_l2_reconstructor_has_no_heavy_imports():
    from pathlib import Path

    source = Path("mmrt/execution/l2_reconstructor.py").read_text()

    assert "import torch" not in source
    assert "import polars" not in source
    assert "import pandas" not in source
    assert "import pyarrow" not in source
