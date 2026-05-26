import importlib
import sys

import pytest

from mmrt.contracts import BookSide
from mmrt.data.book_reconstructor import (
    BOOK_SIDE_ASK_TEXT,
    BOOK_SIDE_BID_TEXT,
    INCREMENTAL_AMOUNT,
    INCREMENTAL_IS_SNAPSHOT,
    INCREMENTAL_PRICE,
    INCREMENTAL_SIDE,
    LOCAL_TS_US,
    RAW_SOURCE_ROW,
    TS_US,
    IncrementalBookRow,
    L2BookReconstructor,
    ReconstructedBookSnapshot,
    ReconstructedBookStatus,
    ReconstructedPriceLevel,
    reconstruct_final_snapshot,
    reconstruct_snapshots_from_rows,
)


def _row(local_ts_us: int, *, ts_us: int | None = None, raw_source_row: int | None = None, is_snapshot: bool, side: str, price: float, amount: float) -> dict:
    return {
        LOCAL_TS_US: local_ts_us,
        TS_US: ts_us if ts_us is not None else local_ts_us,
        RAW_SOURCE_ROW: raw_source_row,
        INCREMENTAL_IS_SNAPSHOT: is_snapshot,
        INCREMENTAL_SIDE: side,
        INCREMENTAL_PRICE: price,
        INCREMENTAL_AMOUNT: amount,
    }


def test_incremental_row_from_mapping_validates_normalized_fields():
    bid = IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0))
    ask = IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0))
    assert bid.side is BookSide.BID
    assert ask.side is BookSide.ASK
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping({INCREMENTAL_IS_SNAPSHOT: True, INCREMENTAL_SIDE: "bid", INCREMENTAL_PRICE: 1.0, INCREMENTAL_AMOUNT: 1.0})
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping({"local_timestamp": 1, INCREMENTAL_IS_SNAPSHOT: True, INCREMENTAL_SIDE: "bid", INCREMENTAL_PRICE: 1.0, INCREMENTAL_AMOUNT: 1.0})
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping(_row(1, is_snapshot="true", side="bid", price=100.0, amount=1.0))
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="buy", price=100.0, amount=1.0))
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="bid", price=0.0, amount=1.0))
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="bid", price=1.0, amount=-1.0))
    zero_amt = IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="bid", price=100.0, amount=0.0))
    assert zero_amt.amount == 0.0


def test_price_level_and_snapshot_validation():
    lvl = ReconstructedPriceLevel(price=10.0, amount=2.0)
    assert lvl.price == 10.0
    with pytest.raises(ValueError):
        ReconstructedPriceLevel(price=0.0, amount=1.0)
    with pytest.raises(ValueError):
        ReconstructedPriceLevel(price=1.0, amount=0.0)
    with pytest.raises(ValueError):
        ReconstructedBookSnapshot(local_ts_us=1, ts_us=1, raw_source_row=0, bids=(ReconstructedPriceLevel(99, 1), ReconstructedPriceLevel(100, 1)), asks=(), is_crossed=False, bid_depth=2, ask_depth=0, reset_count=0, skipped_pre_snapshot_rows=0, applied_update_count=0, deleted_level_count=0)
    with pytest.raises(ValueError):
        ReconstructedBookSnapshot(local_ts_us=1, ts_us=1, raw_source_row=0, bids=(), asks=(ReconstructedPriceLevel(102, 1), ReconstructedPriceLevel(101, 1)), is_crossed=False, bid_depth=0, ask_depth=2, reset_count=0, skipped_pre_snapshot_rows=0, applied_update_count=0, deleted_level_count=0)
    crossed = ReconstructedBookSnapshot(local_ts_us=1, ts_us=1, raw_source_row=0, bids=(ReconstructedPriceLevel(101, 1),), asks=(ReconstructedPriceLevel(100, 1),), is_crossed=True, bid_depth=1, ask_depth=1, reset_count=0, skipped_pre_snapshot_rows=0, applied_update_count=0, deleted_level_count=0)
    assert crossed.best_bid is not None
    assert crossed.best_ask is not None
    assert crossed.mid == 100.5
    assert crossed.spread == -1


def test_skips_rows_before_first_snapshot():
    rows = [
        _row(1, is_snapshot=False, side="bid", price=99.0, amount=2.0),
        _row(2, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(2, is_snapshot=True, side="ask", price=101.0, amount=1.0),
    ]
    snaps = list(reconstruct_snapshots_from_rows(rows))
    assert len(snaps) == 1
    assert [b.price for b in snaps[0].bids] == [100.0]
    assert snaps[0].skipped_pre_snapshot_rows == 1


def test_snapshot_batch_resets_state_when_snapshot_starts_after_non_snapshot():
    rows = [
        _row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0),
        _row(2, is_snapshot=False, side="bid", price=99.0, amount=2.0),
        _row(3, is_snapshot=True, side="bid", price=98.0, amount=3.0),
        _row(3, is_snapshot=True, side="ask", price=102.0, amount=4.0),
    ]
    snap = reconstruct_final_snapshot(rows)
    assert [b.price for b in snap.bids] == [98.0]
    assert [a.price for a in snap.asks] == [102.0]
    assert snap.reset_count == 2


def test_amount_zero_deletes_existing_level():
    snap = reconstruct_final_snapshot([
        _row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0),
        _row(2, is_snapshot=False, side="bid", price=100.0, amount=0.0),
    ])
    assert snap.bids == ()
    assert snap.deleted_level_count == 1


def test_amount_zero_missing_delete_count():
    reconstructor = L2BookReconstructor()
    reconstructor.apply_row(_row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0))
    reconstructor.apply_row(_row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0))
    reconstructor.apply_row(_row(2, is_snapshot=False, side="bid", price=99.0, amount=0.0))
    assert reconstructor.stats.missing_delete_count == 1


def test_nonzero_update_adds_or_updates_level():
    snap = reconstruct_final_snapshot([
        _row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0),
        _row(2, is_snapshot=False, side="bid", price=100.0, amount=2.0),
        _row(2, is_snapshot=False, side="bid", price=99.0, amount=3.0),
    ])
    assert [(x.price, x.amount) for x in snap.bids] == [(100.0, 2.0), (99.0, 3.0)]


def test_emits_only_after_local_timestamp_group_complete():
    snaps = list(reconstruct_snapshots_from_rows([
        _row(10, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(10, is_snapshot=True, side="ask", price=101.0, amount=1.0),
        _row(20, is_snapshot=False, side="bid", price=99.0, amount=1.0),
    ]))
    assert len(snaps) == 2
    assert [x.price for x in snaps[0].bids] == [100.0]
    assert [x.price for x in snaps[0].asks] == [101.0]
    assert [x.price for x in snaps[1].bids] == [100.0, 99.0]


def test_local_ts_decrease_raises_without_sorting():
    with pytest.raises(ValueError):
        list(reconstruct_snapshots_from_rows([
            _row(10, is_snapshot=True, side="bid", price=100.0, amount=1.0),
            _row(10, is_snapshot=True, side="ask", price=101.0, amount=1.0),
            _row(9, is_snapshot=False, side="bid", price=99.0, amount=1.0),
        ]))


def test_crossed_book_is_reported_not_rejected():
    snap = reconstruct_final_snapshot([
        _row(1, is_snapshot=True, side="bid", price=101.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=100.0, amount=1.0),
    ])
    assert snap.is_crossed is True


def test_top_depth_limit_and_ordering():
    snap = reconstruct_final_snapshot([
        _row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(1, is_snapshot=True, side="bid", price=99.0, amount=1.0),
        _row(1, is_snapshot=True, side="bid", price=98.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=102.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=103.0, amount=1.0),
    ], depth=2)
    assert [x.price for x in snap.bids] == [100.0, 99.0]
    assert [x.price for x in snap.asks] == [101.0, 102.0]


def test_reconstructor_apply_row_status_and_stats():
    r = L2BookReconstructor()
    assert r.status is ReconstructedBookStatus.WAITING_FOR_SNAPSHOT
    assert r.apply_row(_row(1, is_snapshot=False, side="bid", price=99.0, amount=1.0)) is False
    assert r.status is ReconstructedBookStatus.WAITING_FOR_SNAPSHOT
    assert r.apply_row(_row(2, is_snapshot=True, side="bid", price=100.0, amount=1.0)) is True
    assert r.status is ReconstructedBookStatus.READY
    assert r.stats.row_count == 2
    assert r.stats.skipped_pre_snapshot_rows == 1
    assert r.stats.applied_update_count == 1
    assert r.stats.snapshot_reset_count == 1


def test_reconstructor_snapshot_before_first_snapshot_raises():
    with pytest.raises(ValueError):
        L2BookReconstructor().snapshot()


def test_reconstruct_final_snapshot_raises_when_no_snapshot_emitted():
    with pytest.raises(ValueError):
        reconstruct_final_snapshot([_row(1, is_snapshot=False, side="bid", price=99.0, amount=1.0)])


def test_no_heavy_data_imports():
    importlib.import_module("mmrt.data.book_reconstructor")
    assert ("po" + "lars") not in sys.modules
    assert "mmrt.data.tardis_csv" not in sys.modules
    assert "mmrt.data.event_merge" not in sys.modules
    assert "mmrt.data.quality" not in sys.modules


def test_no_feature_label_or_decision_concepts():
    module = importlib.import_module("mmrt.data.book_reconstructor")
    get_all = getattr(module, "__all__")
    joined = " ".join(get_all).lower()
    for term in ["feature", "label", "decision", "cmssl", "bybit"]:
        assert term not in joined
    assert ("tar" + "get") not in joined


def test_mapping_rows_and_dataclass_rows_both_supported():
    rows = [
        IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0)),
        _row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0),
    ]
    snap = reconstruct_final_snapshot(rows)
    assert [x.price for x in snap.bids] == [100.0]
    assert [x.price for x in snap.asks] == [101.0]


def test_book_can_be_empty_after_deletes():
    snap = reconstruct_final_snapshot([
        _row(1, is_snapshot=True, side="bid", price=100.0, amount=1.0),
        _row(1, is_snapshot=True, side="ask", price=101.0, amount=1.0),
        _row(2, is_snapshot=False, side="bid", price=100.0, amount=0.0),
        _row(2, is_snapshot=False, side="ask", price=101.0, amount=0.0),
    ])
    assert snap.bids == ()
    assert snap.asks == ()
    assert snap.is_crossed is False


def test_unknown_side_rejected():
    with pytest.raises(ValueError):
        IncrementalBookRow.from_mapping(_row(1, is_snapshot=True, side="unknown", price=1.0, amount=1.0))


def test_no_sort_by_exchange_timestamp():
    snaps = list(reconstruct_snapshots_from_rows([
        _row(10, ts_us=300, is_snapshot=True, side=BOOK_SIDE_BID_TEXT, price=100.0, amount=1.0),
        _row(20, ts_us=100, is_snapshot=False, side=BOOK_SIDE_ASK_TEXT, price=101.0, amount=1.0),
    ]))
    assert [s.local_ts_us for s in snaps] == [10, 20]
