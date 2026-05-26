import subprocess
import sys

import pytest

from mmrt.features import specs
from mmrt.features.specs import FeatureFamily, FeatureOwner, FeatureSource, TransformKey


def test_feature_count_and_order():
    assert specs.FEATURE_COUNT == 178
    assert len(specs.FEATURE_NAMES) == 178
    assert specs.FEATURE_NAMES[0] == "micro_ret_bps_200000us"
    assert specs.FEATURE_NAMES[171] == "trade_impact_half_life_proxy"
    assert specs.FEATURE_NAMES[172:] == (
        "log_dt_decision_us",
        "log_events_100000us",
        "log_events_200000us",
        "log_events_500000us",
        "log_events_1000000us",
        "log_events_3000000us",
    )


def test_no_duplicate_feature_names():
    assert len(specs.FEATURE_NAMES) == len(set(specs.FEATURE_NAMES))


def test_canonical_names_are_microsecond_native():
    for n in specs.FEATURE_NAMES:
        assert "ms" not in n


def test_legacy_to_canonical_roundtrip():
    for legacy in specs.LEGACY_CORE_FEATURE_NAMES + specs.LEGACY_EVENT_CONTEXT_FEATURE_NAMES:
        canonical = specs.legacy_name_to_canonical_name(legacy)
        assert specs.canonical_name_to_legacy_name(canonical) == legacy


def test_context_tail_after_legacy_core():
    assert specs.LEGACY_CORE_FEATURE_COUNT == 172
    assert specs.FEATURE_NAMES[:172] == tuple(specs.legacy_name_to_canonical_name(n) for n in specs.LEGACY_CORE_FEATURE_NAMES)


def test_required_windows_and_depth():
    assert specs.SUPPORTED_WINDOWS_US == (100_000, 200_000, 500_000, 1_000_000, 3_000_000)
    assert specs.REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH == 25
    assert specs.MAX_REQUIRED_BOOK_FEATURE_DEPTH == 20


def test_source_owner_family_examples():
    s = specs.feature_spec_by_name("micro_ret_bps_200000us")
    assert s.source == FeatureSource.BOOK
    assert s.owner == FeatureOwner.BOOK_STATE
    assert s.family == FeatureFamily.PRICE

    s = specs.feature_spec_by_name("signed_notional_flow_usd_200000us")
    assert s.source == FeatureSource.TRADE
    assert s.owner == FeatureOwner.TRADE_STATE
    assert s.family == FeatureFamily.TRADE_FLOW


def test_transform_key_examples():
    assert specs.feature_spec_by_name("spread_bps").transform_key == TransformKey.IDENTITY_EWMA_FAST
    assert specs.feature_spec_by_name("time_since_trade_us").transform_key == TransformKey.TIME_LOG1P_NO_EWMA


def test_lookup_helpers_and_errors():
    idx = specs.feature_index("spread_bps")
    assert isinstance(idx, int)
    assert specs.feature_name(idx) == "spread_bps"
    with pytest.raises(KeyError):
        specs.feature_index("__missing__")
    with pytest.raises(IndexError):
        specs.feature_name(10_000)


def test_feature_spec_record_contents():
    spec = specs.feature_spec_by_name("micro_ret_bps_200000us")
    assert spec.name == "micro_ret_bps_200000us"
    assert spec.index == 0
    assert spec.source == specs.FeatureSource.BOOK
    assert spec.owner == specs.FeatureOwner.BOOK_STATE
    assert spec.legacy_name == "micro_ret_bps_200ms"
    assert spec.required_book_depth >= 1


def test_stable_hashes():
    assert specs.FEATURE_NAMES_HASH == specs.feature_names_hash(specs.FEATURE_NAMES)
    assert specs.FEATURE_SPECS_HASH == specs.feature_specs_hash(specs.FEATURE_SPECS)


def test_public_schema_has_no_aux_core_split():
    schema = specs.schema_record()
    text = str(schema)
    assert "feature" + "_dim_" + "core" not in text
    assert "feature" + "_dim_" + "total" not in text
    assert "AU" + "X_DIM" not in text


def test_corrected_cross_engine_classification():
    s = specs.feature_spec_by_name("trade_side_quote_response_asymmetry_500000us")
    assert s.source == FeatureSource.CROSS
    assert s.owner == FeatureOwner.ENGINE
    assert s.family == FeatureFamily.CROSS_SIGNAL

    s = specs.feature_spec_by_name("trade_impact_half_life_proxy")
    assert s.source == FeatureSource.CROSS
    assert s.owner == FeatureOwner.ENGINE
    assert s.family == FeatureFamily.CROSS_SIGNAL


def test_corrected_non_book_ownership():
    cases = {
        "time_since_trade_us": (FeatureSource.TRADE, FeatureOwner.TRADE_STATE, FeatureFamily.TRADE_FLOW, 0),
        "regime_volume_ewma_500000us": (FeatureSource.TRADE, FeatureOwner.TRADE_STATE, FeatureFamily.REGIME, 0),
        "regime_volume_ewma_3000000us": (FeatureSource.TRADE, FeatureOwner.TRADE_STATE, FeatureFamily.REGIME, 0),
        "vwap_vs_mid_bps_200000us": (FeatureSource.CROSS, FeatureOwner.ENGINE, FeatureFamily.CROSS_SIGNAL, 1),
        "vwap_vs_mid_bps_500000us": (FeatureSource.CROSS, FeatureOwner.ENGINE, FeatureFamily.CROSS_SIGNAL, 1),
    }
    for name, (source, owner, family, depth) in cases.items():
        spec = specs.feature_spec_by_name(name)
        assert spec.source == source
        assert spec.owner == owner
        assert spec.family == family
        assert spec.required_book_depth == depth


def test_no_heavy_imports():
    code = r'''
import sys

before = set(sys.modules)
import mmrt.features.specs  # noqa: F401
after = set(sys.modules) - before

forbidden = (
    "num" + "py",
    "po" + "lars",
    "num" + "ba",
    "to" + "rch",
    "pan" + "das",
    "pya" + "rrow",
    "mmrt.data.tardis_csv",
    "mmrt.data.event_merge",
    "mmrt.data.quality",
)

bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit("forbidden imports loaded by specs: " + repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)
