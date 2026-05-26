import pytest

from mmrt.contracts import (
    AggressorSide,
    BookDeltaEvent,
    BookSide,
    BookSnapshotEvent,
    BookTickerEvent,
    DatasetManifest,
    DecisionReason,
    DerivativeTickerEvent,
    EventMeta,
    EventType,
    FeatureBuildResult,
    LabelResult,
    LabelSpec,
    LiquidationEvent,
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


def test_book_snapshot_depth_consistency():
    with pytest.raises(ValueError):
        BookSnapshotEvent(_meta(TardisDataType.BOOK_SNAPSHOT_5), (PriceLevel(100.0, 1.0),), (PriceLevel(101.0, 1.0),), 25)
    with pytest.raises(ValueError):
        BookSnapshotEvent(_meta(TardisDataType.BOOK_SNAPSHOT_25), (PriceLevel(100.0, 1.0),), (PriceLevel(101.0, 1.0),), 5)
    BookSnapshotEvent(_meta(TardisDataType.BOOK_SNAPSHOT_5), (PriceLevel(100.0, 1.0),), (PriceLevel(101.0, 1.0),), 5)
    BookSnapshotEvent(_meta(TardisDataType.BOOK_SNAPSHOT_25), (PriceLevel(100.0, 1.0),), (PriceLevel(101.0, 1.0),), 25)


def test_book_snapshot_rejects_crossed_book():
    with pytest.raises(ValueError):
        BookSnapshotEvent(
            meta=_meta(TardisDataType.BOOK_SNAPSHOT_25),
            bids=(PriceLevel(101.0, 1.0),),
            asks=(PriceLevel(101.0, 2.0),),
        )


def test_book_delta_accepts_deletion_amount_zero():
    event = BookDeltaEvent(_meta(TardisDataType.INCREMENTAL_BOOK_L2), BookSide.BID, 100.0, 0.0, False)
    assert event.amount == 0.0


def test_boolean_field_validation():
    with pytest.raises(ValueError):
        BookDeltaEvent(_meta(TardisDataType.INCREMENTAL_BOOK_L2), BookSide.BID, 100.0, 0.0, "false")
    with pytest.raises(ValueError):
        BookDeltaEvent(_meta(TardisDataType.INCREMENTAL_BOOK_L2), BookSide.BID, 100.0, 0.0, 0)
    with pytest.raises(ValueError):
        FeatureBuildResult(_meta(TardisDataType.TRADES), EventType.TRADE, 1, None, (), None, 0)
    with pytest.raises(ValueError):
        FeatureBuildResult(_meta(TardisDataType.TRADES), EventType.TRADE, "false", None, (), None, 0)


def test_trade_event_side_and_amount_validation():
    for side in (AggressorSide.BUY, AggressorSide.SELL, AggressorSide.UNKNOWN):
        evt = TradeEvent(_meta(TardisDataType.TRADES), "id", side, 100.0, 1.0)
        assert evt.event_type == EventType.TRADE
    with pytest.raises(ValueError):
        TradeEvent(_meta(TardisDataType.TRADES), "id", AggressorSide.BUY, 100.0, 0.0)


def test_label_spec_horizons_rules_and_context():
    spec = LabelSpec(horizons_us=(300, 100, 200), entry_delay_us=50)
    assert spec.horizons_us == (100, 200, 300)
    assert spec.label_context_us == 350
    with pytest.raises(ValueError):
        LabelSpec((100, 100, 200), 0)


def test_label_result_normalization_and_duplicate_rejection():
    result = LabelResult(1, 1, [100, 200], [0.1, -0.2])
    assert result.horizons_us == (100, 200)
    assert result.values_bps == (0.1, -0.2)
    assert isinstance(result.horizons_us, tuple)
    assert isinstance(result.values_bps, tuple)
    with pytest.raises(ValueError):
        LabelResult(decision_ts_us=1, entry_ts_us=1, horizons_us=(100, 100), values_bps=(1.0, 2.0))


def test_feature_build_result_rejects_non_decision_with_features():
    with pytest.raises(ValueError):
        FeatureBuildResult(_meta(TardisDataType.TRADES), EventType.TRADE, False, None, (1.0,), None, 0)




def test_feature_build_result_rejects_unsupported_source_data_types():
    for dtype in (TardisDataType.QUOTES, TardisDataType.OPTIONS_CHAIN):
        with pytest.raises(ValueError):
            FeatureBuildResult(
                meta=_meta(dtype),
                event_type=EventType.TRADE,
                is_decision=False,
                decision_reason=None,
                features=(),
                raw_mid=None,
                dt_us=0,
            )


def test_feature_build_result_accepts_supported_source_type_mappings():
    cases = (
        (TardisDataType.BOOK_SNAPSHOT_25, EventType.BOOK_SNAPSHOT),
        (TardisDataType.BOOK_SNAPSHOT_5, EventType.BOOK_SNAPSHOT),
        (TardisDataType.INCREMENTAL_BOOK_L2, EventType.BOOK_DELTA),
        (TardisDataType.TRADES, EventType.TRADE),
        (TardisDataType.BOOK_TICKER, EventType.BOOK_TICKER),
        (TardisDataType.DERIVATIVE_TICKER, EventType.DERIVATIVE_TICKER),
        (TardisDataType.LIQUIDATIONS, EventType.LIQUIDATION),
    )
    for dtype, event_type in cases:
        result = FeatureBuildResult(
            meta=_meta(dtype),
            event_type=event_type,
            is_decision=False,
            decision_reason=None,
            features=(),
            raw_mid=None,
            dt_us=0,
        )
        assert result.event_type == event_type


def test_feature_build_result_rejects_event_type_mismatch():
    with pytest.raises(ValueError):
        FeatureBuildResult(
            meta=_meta(TardisDataType.TRADES),
            event_type=EventType.BOOK_SNAPSHOT,
            is_decision=False,
            decision_reason=None,
            features=(),
            raw_mid=None,
            dt_us=0,
        )
def test_feature_build_result_rejects_bad_decision_payload():
    with pytest.raises(ValueError):
        FeatureBuildResult(_meta(TardisDataType.TRADES), EventType.TRADE, True, DecisionReason.BOOK_EVENT, (), 100.0, 1)
    with pytest.raises(ValueError):
        FeatureBuildResult(_meta(TardisDataType.TRADES), EventType.TRADE, True, DecisionReason.BOOK_EVENT, (1.0,), None, 1)


def test_bool_rejection_in_numeric_validators():
    with pytest.raises(ValueError):
        EventMeta("binance-futures", "BTCUSDT", True, 110, TardisDataType.TRADES, 0)
    with pytest.raises(ValueError):
        EventMeta("binance-futures", "BTCUSDT", 100, True, TardisDataType.TRADES, 0)
    with pytest.raises(ValueError):
        PriceLevel(True, 1.0)
    with pytest.raises(ValueError):
        PriceLevel(100.0, False)
    with pytest.raises(ValueError):
        TimeRangeUS(True, 2)
    with pytest.raises(ValueError):
        SegmentSpec("seg", TimeRangeUS(1, 2), ("a.csv",), True, 0)


def test_dataset_manifest_positive_and_totals():
    segments = (
        SegmentSpec("seg-a", TimeRangeUS(1, 3), ("a.csv",), 10, 6),
        SegmentSpec("seg-b", TimeRangeUS(3, 5), ("b.csv",), 20, 14),
    )
    manifest = DatasetManifest(
        schema_version="v1",
        storage_format=StorageFormat.FLAT_DECISION_ROWS_US_V1,
        exchange="binance-futures",
        symbol="BTCUSDT",
        time_unit=TimeUnit.MICROSECOND,
        source_data_types=(TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES),
        label_spec=LabelSpec((100, 200), 0),
        lookback_rows=10,
        feature_schema_version="f1",
        feature_names_hash="abc",
        feature_dim=6,
        segments=segments,
        split_plan=None,
    )
    assert manifest.total_rows == 30
    assert manifest.total_labels == 20


def test_dataset_manifest_rejects_invalid_split_plan_type():
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
            feature_dim=6,
            segments=(seg,),
            split_plan="not-a-split-plan",
        )


def test_segment_spec_rejects_invalid_time_range_type():
    with pytest.raises(ValueError):
        SegmentSpec("seg", "not-a-time-range", ("a.csv",), 10, 5)


def test_split_entry_rejects_invalid_time_range_type():
    with pytest.raises(ValueError):
        SplitEntry(SplitRole.TRAIN, "seg", 0, 1, "not-a-time-range")


def test_dataset_manifest_rejects_invalid_label_spec_type():
    seg = SegmentSpec("seg", TimeRangeUS(1, 2), ("a.csv",), 10, 5)
    with pytest.raises(ValueError):
        DatasetManifest(
            schema_version="v1",
            storage_format=StorageFormat.FLAT_DECISION_ROWS_US_V1,
            exchange="binance-futures",
            symbol="BTCUSDT",
            time_unit=TimeUnit.MICROSECOND,
            source_data_types=(TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES),
            label_spec="not-a-label-spec",
            lookback_rows=10,
            feature_schema_version="f1",
            feature_names_hash="abc",
            feature_dim=6,
            segments=(seg,),
            split_plan=None,
        )


def test_dataset_manifest_rejects_invalid_segment_item_type():
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
            feature_dim=6,
            segments=("not-a-segment",),
            split_plan=None,
        )


def test_dataset_manifest_rejects_duplicate_source_data_types():
    seg = SegmentSpec("seg", TimeRangeUS(1, 2), ("a.csv",), 10, 5)
    with pytest.raises(ValueError):
        DatasetManifest(
            schema_version="v1",
            storage_format=StorageFormat.FLAT_DECISION_ROWS_US_V1,
            exchange="binance-futures",
            symbol="BTCUSDT",
            time_unit=TimeUnit.MICROSECOND,
            source_data_types=(TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.BOOK_SNAPSHOT_25),
            label_spec=LabelSpec((100,), 0),
            lookback_rows=10,
            feature_schema_version="f1",
            feature_names_hash="abc",
            feature_dim=6,
            segments=(seg,),
            split_plan=None,
        )


def test_dataset_manifest_accepts_string_enums_for_unique_source_data_types():
    seg = SegmentSpec("seg", TimeRangeUS(1, 2), ("a.csv",), 10, 5)
    manifest = DatasetManifest(
        schema_version="v1",
        storage_format="flat_decision_rows_us_v1",
        exchange="binance-futures",
        symbol="BTCUSDT",
        time_unit="us",
        source_data_types=("book_snapshot_25", "trades"),
        label_spec=LabelSpec((100,), 0),
        lookback_rows=10,
        feature_schema_version="f1",
        feature_names_hash="abc",
        feature_dim=6,
        segments=(seg,),
        split_plan=None,
    )
    assert manifest.storage_format == StorageFormat.FLAT_DECISION_ROWS_US_V1
    assert manifest.time_unit == TimeUnit.MICROSECOND
    assert manifest.source_data_types == (TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES)


def test_split_plan_requires_train_and_val():
    entry = SplitEntry(SplitRole.TRAIN, "seg", 0, 1, TimeRangeUS(1, 2))
    with pytest.raises(ValueError):
        SplitPlan((entry,))


def test_non_core_events_basic_validation():
    BookTickerEvent(_meta(TardisDataType.BOOK_TICKER), 100.0, 1.0, 101.0, 1.2)
    with pytest.raises(ValueError):
        BookTickerEvent(_meta(TardisDataType.BOOK_TICKER), 101.0, 1.0, 101.0, 1.2)

    DerivativeTickerEvent(
        _meta(TardisDataType.DERIVATIVE_TICKER),
        open_interest=10.0,
        mark_price=100.0,
        funding_rate=0.001,
    )
    with pytest.raises(ValueError):
        DerivativeTickerEvent(_meta(TardisDataType.DERIVATIVE_TICKER), open_interest=-1.0)

    LiquidationEvent(_meta(TardisDataType.LIQUIDATIONS), "liq", AggressorSide.BUY, 100.0, 1.0)
    with pytest.raises(ValueError):
        LiquidationEvent(_meta(TardisDataType.LIQUIDATIONS), "liq", AggressorSide.BUY, 100.0, 0.0)
