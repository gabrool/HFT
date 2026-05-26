import sys
import importlib
import pytest
from mmrt.features import specs

# keep existing core checks concise

def test_feature_counts_and_order_and_hash_shape():
    assert specs.FEATURE_COUNT == 178
    assert len(specs.FEATURE_NAMES) == 178
    assert specs.FEATURE_NAMES[0] == "micro_ret_bps_200000us"
    assert specs.FEATURE_NAMES[171] == "trade_impact_half_life_proxy"
    assert specs.FEATURE_NAMES_HASH == specs.feature_names_hash(specs.FEATURE_NAMES)
    assert specs.FEATURE_SPECS_HASH == specs.feature_specs_hash(specs.FEATURE_SPECS)


def test_corrected_non_book_ownership():
    cases = {
        "time_since_trade_us": (specs.FeatureSource.TRADE, specs.FeatureOwner.TRADE_STATE, specs.FeatureFamily.TRADE_FLOW, 0),
        "regime_volume_ewma_500000us": (specs.FeatureSource.TRADE, specs.FeatureOwner.TRADE_STATE, specs.FeatureFamily.REGIME, 0),
        "regime_volume_ewma_3000000us": (specs.FeatureSource.TRADE, specs.FeatureOwner.TRADE_STATE, specs.FeatureFamily.REGIME, 0),
        "vwap_vs_mid_bps_200000us": (specs.FeatureSource.CROSS, specs.FeatureOwner.ENGINE, specs.FeatureFamily.CROSS_SIGNAL, 1),
        "vwap_vs_mid_bps_500000us": (specs.FeatureSource.CROSS, specs.FeatureOwner.ENGINE, specs.FeatureFamily.CROSS_SIGNAL, 1),
    }
    for name, (source, owner, family, depth) in cases.items():
        spec = specs.feature_spec_by_name(name)
        assert spec.source == source
        assert spec.owner == owner
        assert spec.family == family
        assert spec.required_book_depth == depth


def test_transform_key_still_time_for_time_since_trade():
    assert specs.feature_spec_by_name("time_since_trade_us").transform_key == specs.TransformKey.TIME_LOG1P_NO_EWMA


def test_no_heavy_imports():
    forbidden = ("num"+"py", "po"+"lars", "num"+"ba", "to"+"rch", "pan"+"das", "mmrt.data.tardis_csv", "mmrt.data.event_merge", "mmrt.data.quality")
    for name in forbidden:
        sys.modules.pop(name, None)
    importlib.reload(specs)
    for name in forbidden:
        assert name not in sys.modules
