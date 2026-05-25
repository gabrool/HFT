import pytest

from mmrt.contracts import (
    AggressorSide,
    BookDeltaEvent,
    BookSide,
    BookSnapshotEvent,
    DatasetManifest,
    DecisionReason,
    EventMeta,
    EventType,
    FeatureBuildResult,
    LabelSpec,
    PriceLevel,
    SegmentSpec,
    SplitEntry,
    SplitPlan,
    SplitRole,
    StorageFormat,
    TardisDataType,
    TimeRangeUS,
    TimeUnit,
    TradeEvent,
)


def _meta(dtype: TardisDataType) -> EventMeta:
    return EventMeta("binance-futures", "BTCUSDT", 100, 110, dtype, 7)


def test_event_meta_order_key_and_validation():
    meta = _meta(TardisDataType.TRADES)
    assert meta.order_key == (110, 100, 7)
    with pytest.raises(ValueError):
        EventMeta("", "BTCUSDT", 1, 1, TardisDataType.TRADES, 0)


def test_book_snapshot_valid_computed_fields():
    event = BookSnapshotEvent(
        meta=_meta(TardisDataType.BOOK_SNAPSHOT_25),
        bids=(PriceLevel(100.0, 1.0), PriceLevel(99.0, 2.0)),
        asks=(PriceLevel(101.0, 1.5), PriceLevel(102.0, 2.5)),
    )
    assert event.best_bid == PriceLevel(100.0, 1.0)
    assert event.best_ask == PriceLevel(101.0, 1.5)
    assert event.mid == 100.5
    assert event.spread == 1.0


def test_book_snapshot_rejects_crossed_book():
    with pytest.raises(ValueError):
        BookSnapshotEvent(
            meta=_meta(TardisDataType.BOOK_SNAPSHOT_25),
            bids=(PriceLevel(101.0, 1.0),),
            asks=(PriceLevel(101.0, 2.0),),
        )


def test_book_snapshot_rejects_nonpositive_amount():
    with pytest.raises(ValueError):
        BookSnapshotEvent(
            meta=_meta(TardisDataType.BOOK_SNAPSHOT_25),
            bids=(PriceLevel(100.0, 0.0),),
            asks=(PriceLevel(101.0, 1.0),),
        )


def test_book_delta_accepts_deletion_amount_zero():
    event = BookDeltaEvent(_meta(TardisDataType.INCREMENTAL_BOOK_L2), BookSide.BID, 100.0, 0.0, False)
    assert event.amount == 0.0


def test_trade_event_side_and_amount_validation():
    for side in (AggressorSide.BUY, AggressorSide.SELL, AggressorSide.UNKNOWN):
        evt = TradeEvent(_meta(TardisDataType.TRADES), "id", side, 100.0, 1.0)
        assert evt.event_type == EventType.TRADE
    with pytest.raises(ValueError):
        TradeEvent(_meta(TardisDataType.TRADES), "id", AggressorSide.BUY, 100.0, 0.0)


def test_label_spec_sorts_horizons_and_context():
    spec = LabelSpec(horizons_us=(300, 100, 200), entry_delay_us=50)
    assert spec.horizons_us == (100, 200, 300)
    assert spec.label_context_us == 350


def test_label_spec_context_specific_example():
    spec = LabelSpec(horizons_us=(200000, 500000, 1000000), entry_delay_us=1000)
    assert spec.label_context_us == 1001000


def test_feature_build_result_rejects_non_decision_with_features():
    with pytest.raises(ValueError):
        FeatureBuildResult(
            meta=_meta(TardisDataType.TRADES),
            event_type=EventType.TRADE,
            is_decision=False,
            decision_reason=None,
            features=(1.0,),
            raw_mid=None,
            dt_us=0,
        )


def test_feature_build_result_rejects_bad_decision_payload():
    with pytest.raises(ValueError):
        FeatureBuildResult(
            meta=_meta(TardisDataType.TRADES),
            event_type=EventType.TRADE,
            is_decision=True,
            decision_reason=DecisionReason.BOOK_EVENT,
            features=(),
            raw_mid=100.0,
            dt_us=1,
        )
    with pytest.raises(ValueError):
        FeatureBuildResult(
            meta=_meta(TardisDataType.TRADES),
            event_type=EventType.TRADE,
            is_decision=True,
            decision_reason=DecisionReason.BOOK_EVENT,
            features=(1.0,),
            raw_mid=None,
            dt_us=1,
        )


def test_dataset_manifest_feature_dim_validation():
    seg = SegmentSpec("seg", TimeRangeUS(1, 2), ("a.csv",), 10, 5)
    with pytest.raises(ValueError):
        DatasetManifest(
            schema_version="v1",
            storage_format=StorageFormat.FLAT_DECISION_ROWS_US_V1,
            exchange="binance-futures",
            symbol="BTCUSDT",
            time_unit=TimeUnit.MICROSECOND,
            source_data_types=(TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES),
            label_spec=LabelSpec((100,), 0),
            lookback_rows=10,
            feature_schema_version="f1",
            feature_names_hash="abc",
            feature_dim_core=4,
            aux_dim=2,
            feature_dim_total=7,
            segments=(seg,),
        )


def test_split_plan_requires_train_and_val():
    entry = SplitEntry(SplitRole.TRAIN, "seg", 0, 1, TimeRangeUS(1, 2))
    with pytest.raises(ValueError):
        SplitPlan((entry,))
