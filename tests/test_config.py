import pytest

import mmrt.config as cfg_module
from mmrt.features import specs
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
    assert cfg.data.source_data_types == (TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES)
    assert cfg.labels.horizons_us == (200_000, 500_000, 1_000_000)
    assert cfg.labels.entry_delay_us == 1_000
    assert cfg.runtime.lookback_rows == 10
    assert cfg.storage.time_unit == TimeUnit.MICROSECOND
    assert cfg.decision.stride_us == 500_000
    assert cfg.storage.feature_schema == specs.FEATURE_SCHEMA


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


def test_label_config_rejects_unsupported_price_refs() -> None:
    with pytest.raises(ValueError):
        LabelConfig(price_reference="microprice")
    with pytest.raises(ValueError):
        LabelConfig(price_reference="mark")


def test_data_config_validation() -> None:
    cfg = DataConfig(source_data_types=("incremental_book_L2", "trades"))
    assert cfg.source_data_types == (TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES)
    with pytest.raises(ValueError):
        DataConfig(source_data_types=(TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.INCREMENTAL_BOOK_L2))
    with pytest.raises(ValueError):
        DataConfig(source_data_types=(TardisDataType.INCREMENTAL_BOOK_L2,))
    with pytest.raises(TypeError):
        DataConfig(**{"drop_duplicate_" + "trades": True})


def test_decision_config_constraints() -> None:
    assert DecisionConfig().stride_us == 500_000
    assert DecisionConfig(stride_us=500_000).stride_us == 500_000
    with pytest.raises(ValueError):
        DecisionConfig(policy="scheduled_time")
    with pytest.raises(ValueError):
        DecisionConfig(reason="scheduled_time")
    with pytest.raises(ValueError):
        DecisionConfig(stride_us=0)
    with pytest.raises(ValueError):
        DecisionConfig(stride_us=True)
    with pytest.raises(ValueError):
        DecisionConfig(stride_us=1)
    with pytest.raises(TypeError):
        DecisionConfig(**{"stride_" + "rows": 1})


def test_feature_runtime_config_constraints() -> None:
    FeatureRuntimeConfig()
    with pytest.raises(ValueError):
        FeatureRuntimeConfig(lookback_rows=0)
    with pytest.raises(ValueError):
        FeatureRuntimeConfig(feature_dtype="float64")
    with pytest.raises(ValueError):
        FeatureRuntimeConfig(timestamp_dtype="float64")


def test_storage_config_constraints() -> None:
    assert StorageConfig().feature_schema == specs.FEATURE_SCHEMA
    assert StorageConfig(time_unit="us").time_unit == TimeUnit.MICROSECOND
    assert StorageConfig(storage_format="flat_decision_rows_us").storage_format == StorageFormat.FLAT_DECISION_ROWS_US
    with pytest.raises(ValueError):
        StorageConfig(pipeline_schema="")


def test_pipeline_config_invariants() -> None:
    cfg = PipelineConfig()
    assert cfg.source_data_type_values == ("incremental_book_L2", "trades")
    assert cfg.decision.stride_us == cfg_module.DEFAULT_DECISION_STRIDE_US
    assert cfg.storage.feature_schema == specs.FEATURE_SCHEMA
    assert cfg.label_spec == cfg.labels.to_label_spec()
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=("book_snapshot_5", TardisDataType.TRADES)))
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=("quotes", TardisDataType.TRADES)))
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=("options_chain", TardisDataType.TRADES)))


def test_pipeline_config_requires_book_and_trades() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.TRADES,)))
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=(TardisDataType.INCREMENTAL_BOOK_L2,)))


def test_pipeline_config_rejects_additional_source_types() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(source_data_types=("incremental_book_L2", "trades", "trades")))


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


def test_label_config_asof_policy() -> None:
    assert LabelConfig(asof_policy=AsOfPolicy.LAST_OBSERVATION).asof_policy == AsOfPolicy.LAST_OBSERVATION


def test_public_api_alignment() -> None:
    assert cfg_module.DEFAULT_DECISION_STRIDE_US == 500_000
    assert cfg_module.DEFAULT_FEATURE_SCHEMA == specs.FEATURE_SCHEMA
    assert "DEFAULT_DECISION_STRIDE_US" in cfg_module.__all__
    assert "DEFAULT_DECISION_STRIDE_" + "ROWS" not in cfg_module.__all__
    assert "DEFAULT_DROP_DUPLICATE_" + "TRADES" not in cfg_module.__all__


def test_retired_surface_removed() -> None:
    assert not hasattr(cfg_module, "DEFAULT_DECISION_STRIDE_" + "ROWS")
    assert not hasattr(DecisionConfig(), "stride_" + "rows")
    assert not hasattr(cfg_module, "DEFAULT_DROP_DUPLICATE_" + "TRADES")
    assert not hasattr(DataConfig(), "drop_duplicate_" + "trades")
    assert not hasattr(DataConfig(), "disabled_" + "context_data_types")


def test_default_config_alignment() -> None:
    c = default_config()
    assert c.decision.stride_us == cfg_module.DEFAULT_DECISION_STRIDE_US
    assert c.storage.feature_schema == specs.FEATURE_SCHEMA
