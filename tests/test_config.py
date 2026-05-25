import pytest

from mmrt.config import (
    DataConfig,
    DecisionConfig,
    FeatureRuntimeConfig,
    LabelConfig,
    PipelineConfig,
    StorageConfig,
    default_config,
    default_label_spec,
)
from mmrt.contracts import (
    AsOfPolicy,
    DecisionReason,
    LabelSpec,
    PriceReference,
    StorageFormat,
    TardisDataType,
    TimeUnit,
)


def test_default_config_core_values() -> None:
    cfg = default_config()
    assert cfg.market.exchange == "binance-futures"
    assert cfg.market.symbol == "BTCUSDT"
    assert cfg.data.source_data_types == (TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES)
    assert cfg.labels.horizons_us == (200_000, 500_000, 1_000_000)
    assert cfg.labels.entry_delay_us == 1_000
    assert cfg.runtime.lookback_rows == 10
    assert cfg.storage.time_unit == TimeUnit.MICROSECOND


def test_default_label_spec() -> None:
    spec = default_label_spec()
    assert isinstance(spec, LabelSpec)
    assert spec.horizons_us == (200_000, 500_000, 1_000_000)
    assert spec.entry_delay_us == 1_000
    assert spec.label_context_us == 1_001_000


def test_label_config_sorting_and_duplicate_rejection() -> None:
    cfg = LabelConfig(horizons_us=(1_000_000, 200_000, 500_000))
    assert cfg.horizons_us == (200_000, 500_000, 1_000_000)
    with pytest.raises(ValueError):
        LabelConfig(horizons_us=(200_000, 200_000))


def test_label_config_rejects_unsupported_price_refs_v1() -> None:
    with pytest.raises(ValueError):
        LabelConfig(price_reference=PriceReference.MICROPRICE)
    with pytest.raises(ValueError):
        LabelConfig(price_reference=PriceReference.MARK)


def test_data_config_validation() -> None:
    cfg = DataConfig(source_data_types=("book_snapshot_25", "trades"))
    assert cfg.source_data_types == (TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES)
    with pytest.raises(ValueError):
        DataConfig(source_data_types=(TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.BOOK_SNAPSHOT_25))
    with pytest.raises(ValueError):
        DataConfig(
            source_data_types=(TardisDataType.BOOK_SNAPSHOT_25,),
            disabled_context_data_types=(TardisDataType.BOOK_SNAPSHOT_25,),
        )
    with pytest.raises(ValueError):
        DataConfig(strict_validation=1)
    with pytest.raises(ValueError):
        DataConfig(allow_equal_local_ts="true")


def test_decision_config_v1_constraints() -> None:
    DecisionConfig()
    with pytest.raises(ValueError):
        DecisionConfig(policy="scheduled_time")
    with pytest.raises(ValueError):
        DecisionConfig(stride_rows=5)
    with pytest.raises(ValueError):
        DecisionConfig(reason=DecisionReason.SCHEDULED_TIME)


def test_feature_runtime_config_v1_constraints() -> None:
    FeatureRuntimeConfig()
    with pytest.raises(ValueError):
        FeatureRuntimeConfig(lookback_rows=0)
    with pytest.raises(ValueError):
        FeatureRuntimeConfig(feature_dtype="float64")
    with pytest.raises(ValueError):
        FeatureRuntimeConfig(timestamp_dtype="float64")


def test_storage_config_v1_constraints() -> None:
    StorageConfig()
    assert StorageConfig(time_unit="us").time_unit == TimeUnit.MICROSECOND
    assert StorageConfig(storage_format="flat_decision_rows_us_v1").storage_format == StorageFormat.FLAT_DECISION_ROWS_US_V1
    with pytest.raises(ValueError):
        StorageConfig(pipeline_schema_version="")


def test_pipeline_config_invariants() -> None:
    cfg = PipelineConfig()
    assert cfg.source_data_type_values == ("book_snapshot_25", "trades")
    assert cfg.label_spec == cfg.labels.to_label_spec()
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.BOOK_SNAPSHOT_5, TardisDataType.TRADES)))
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.QUOTES, TardisDataType.TRADES)))
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.OPTIONS_CHAIN, TardisDataType.TRADES)))


def test_pipeline_config_requires_book_and_trades() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.TRADES,)))
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.BOOK_SNAPSHOT_25,)))


def test_pipeline_config_allows_additional_supported_source_types() -> None:
    cfg = PipelineConfig(
        data=DataConfig(
            source_data_types=(
                TardisDataType.BOOK_SNAPSHOT_25,
                TardisDataType.TRADES,
                TardisDataType.DERIVATIVE_TICKER,
            ),
            disabled_context_data_types=(
                TardisDataType.LIQUIDATIONS,
                TardisDataType.BOOK_TICKER,
            ),
        )
    )
    assert cfg.data.source_data_types == (
        TardisDataType.BOOK_SNAPSHOT_25,
        TardisDataType.TRADES,
        TardisDataType.DERIVATIVE_TICKER,
    )


def test_pipeline_config_rejects_invalid_nested_config_objects() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(market="bad")
    with pytest.raises(ValueError):
        PipelineConfig(data="bad")
    with pytest.raises(ValueError):
        PipelineConfig(decision="bad")
    with pytest.raises(ValueError):
        PipelineConfig(labels="bad")
    with pytest.raises(ValueError):
        PipelineConfig(runtime="bad")
    with pytest.raises(ValueError):
        PipelineConfig(storage="bad")


def test_label_config_v1_asof_policy() -> None:
    assert LabelConfig(asof_policy=AsOfPolicy.LAST_OBSERVATION).asof_policy == AsOfPolicy.LAST_OBSERVATION
