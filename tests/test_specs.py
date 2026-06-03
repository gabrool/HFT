import re
import subprocess
import sys
from pathlib import Path

import pytest

from mmrt.features import specs
from mmrt.features.specs import FeatureFamily, FeatureOwner, FeatureSource, TransformKey


def test_feature_count_and_order():
    assert specs.CORE_FEATURE_COUNT == 146
    assert specs.EVENT_CONTEXT_FEATURE_COUNT == 6
    assert specs.FEATURE_COUNT == 152
    assert len(specs.FEATURE_SPECS) == 152
    assert len(specs.FEATURE_NAMES) == 152
    assert len(set(specs.FEATURE_NAMES)) == 152
    assert specs.FEATURE_NAMES[0] == "micro_ret_bps_200000us"
    assert specs.FEATURE_NAMES[145] == "trade_impact_half_life_proxy"
    assert specs.FEATURE_NAMES[146:] == (
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
    for legacy in specs.CORE_FEATURE_NAMES + specs.EVENT_CONTEXT_FEATURE_NAMES:
        canonical = specs.legacy_name_to_canonical_name(legacy)
        assert specs.canonical_name_to_legacy_name(canonical) == legacy


def test_context_tail_after_core_features():
    assert specs.CORE_FEATURE_COUNT == 146
    assert specs.FEATURE_NAMES[:146] == tuple(specs.legacy_name_to_canonical_name(n) for n in specs.CORE_FEATURE_NAMES)




def test_pruned_correlated_features_absent_from_feature_registry():
    dropped = {
        "micro_ret_bps_500000us",
        "micro_ret_bps_1000000us",
        "obi_l10",
        "ofi_l1",
        "ofi_l5",
        "ofi_l1_over_depth_5bps",
        "ofi_l5_over_depth_5bps",
        "ofi_l10_over_depth_5bps",
        "ofi_l1_sum_over_depth_200000us",
        "ofi_l1_sum_over_depth_500000us",
        "ofi_l10_sum_over_depth_200000us",
        "ofi_l10_sum_over_depth_500000us",
        "ofi_l5_sum_over_depth_1000000us",
        "ofi_l1_pressure_over_depth_5bps_1000000us",
        "ofi_l1_accel_200000us_minus_500000us",
        "ofi_l5_accel_200000us_minus_500000us",
        "ofi_l10_accel_200000us_minus_500000us",
        "ofi_l1_accel_500000us_minus_1000000us",
        "ofi_l5_accel_500000us_minus_1000000us",
        "ofi_l10_accel_500000us_minus_1000000us",
        "spread_change_count_500000us",
        "cvd_slope_usd_per_sec_500000us",
        "cvd_slope_usd_per_sec_1000000us",
        "post_sell_trade_bid_replenishment_200000us",
        "post_buy_trade_ask_replenishment_200000us",
        "depth_imbalance_5bps_mean_500000us",
    }
    assert specs.CORE_FEATURE_COUNT == 146
    assert specs.EVENT_CONTEXT_FEATURE_COUNT == 6
    assert specs.FEATURE_COUNT == 152
    assert len(specs.FEATURE_SPECS) == 152
    assert len(specs.FEATURE_NAMES) == 152
    assert len(set(specs.FEATURE_NAMES)) == 152
    assert not dropped.intersection(set(specs.FEATURE_NAMES))
    for name in dropped:
        assert name not in specs.FEATURE_NAME_TO_INDEX


def test_retained_correlated_cluster_representatives_still_exist():
    retained = {
        "mid_slope_bps_per_sec_500000us",
        "mid_slope_bps_per_sec_1000000us",
        "obi_l1",
        "ofi_l3",
        "ofi_l3_over_depth_5bps",
        "ofi_l1_pressure_over_depth_5bps_200000us",
        "ofi_l1_pressure_over_depth_5bps_500000us",
        "ofi_l5_sum_over_depth_200000us",
        "ofi_l5_sum_over_depth_500000us",
        "ofi_l10_sum_over_depth_1000000us",
        "ofi_l3_accel_200000us_minus_500000us",
        "ofi_l3_accel_500000us_minus_1000000us",
        "bid_price_change_rate_500000us",
        "cvd_change_usd_500000us",
        "cvd_change_usd_1000000us",
        "absorption_bid_200000us",
        "absorption_ask_200000us",
        "depth_imbalance_5bps_mean_1000000us",
    }
    assert retained.issubset(set(specs.FEATURE_NAMES))


def test_no_pruned_output_assignment_remains():
    dropped = {
        "micro_ret_bps_500000us",
        "micro_ret_bps_1000000us",
        "obi_l10",
        "ofi_l1",
        "ofi_l5",
        "ofi_l1_over_depth_5bps",
        "ofi_l5_over_depth_5bps",
        "ofi_l10_over_depth_5bps",
        "ofi_l1_sum_over_depth_200000us",
        "ofi_l1_sum_over_depth_500000us",
        "ofi_l10_sum_over_depth_200000us",
        "ofi_l10_sum_over_depth_500000us",
        "ofi_l5_sum_over_depth_1000000us",
        "ofi_l1_pressure_over_depth_5bps_1000000us",
        "ofi_l1_accel_200000us_minus_500000us",
        "ofi_l5_accel_200000us_minus_500000us",
        "ofi_l10_accel_200000us_minus_500000us",
        "ofi_l1_accel_500000us_minus_1000000us",
        "ofi_l5_accel_500000us_minus_1000000us",
        "ofi_l10_accel_500000us_minus_1000000us",
        "spread_change_count_500000us",
        "cvd_slope_usd_per_sec_500000us",
        "cvd_slope_usd_per_sec_1000000us",
        "post_sell_trade_bid_replenishment_200000us",
        "post_buy_trade_ask_replenishment_200000us",
        "depth_imbalance_5bps_mean_500000us",
    }
    for relpath in (
        "mmrt/features/book_state.py",
        "mmrt/features/trade_state.py",
        "mmrt/features/engine.py",
    ):
        text = Path(relpath).read_text()
        assigned = set(re.findall(r'setf\(\s*f?["\']([^"\']+)["\']', text))
        assert not dropped.intersection(assigned)


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


def test_feature_schema_version_is_core_not_legacy():
    assert specs.FEATURE_SCHEMA_VERSION == "mmrt_feature_schema_v2_snapshot25_trades_core146_ctx6_us_corr_pruned"
    assert "legacy" not in specs.FEATURE_SCHEMA_VERSION


def test_no_public_legacy_core_names_required_for_new_code():
    legacy_exports = {name for name in specs.__all__ if name.startswith("LEGACY_")}
    assert legacy_exports == set()


def test_transform_keys_include_log_ewma_half_life_classes():
    assert TransformKey.LOG1P_POS_EWMA_FAST.value == "log1p_pos_ewma_fast"
    assert TransformKey.LOG1P_POS_EWMA_MEDIUM.value == "log1p_pos_ewma_medium"
    assert TransformKey.LOG1P_POS_EWMA_SLOW.value == "log1p_pos_ewma_slow"
    assert TransformKey.SIGNED_LOG1P_EWMA_FAST.value == "signed_log1p_ewma_fast"
    assert TransformKey.SIGNED_LOG1P_EWMA_MEDIUM.value == "signed_log1p_ewma_medium"
    assert TransformKey.SIGNED_LOG1P_EWMA_SLOW.value == "signed_log1p_ewma_slow"


def test_event_context_features_are_no_ewma():
    for name in specs.FEATURE_NAMES[-specs.EVENT_CONTEXT_FEATURE_COUNT:]:
        assert specs.feature_spec_by_name(name).transform_key == TransformKey.LOG1P_POS_NO_EWMA


def test_time_features_are_time_log1p_no_ewma():
    for spec in specs.FEATURE_SPECS:
        if spec.unit == specs.FeatureUnit.MICROSECONDS:
            assert spec.transform_key == TransformKey.TIME_LOG1P_NO_EWMA


def test_sign_features_are_sign_no_ewma():
    assert specs.feature_spec_by_name("last_trade_side_sign").transform_key == TransformKey.SIGN_NO_EWMA
    assert specs.feature_spec_by_name("last_tick_sign").transform_key == TransformKey.SIGN_NO_EWMA


def test_bounded_ratio_features_are_ratio_bounded():
    for name in [
        "zero_tick_fraction_200000us",
        "tick_sign_imbalance_200000us",
        "top5_trade_share_notional_3000000us",
        "trade_sign_entropy_3000000us",
        "trade_side_quote_response_asymmetry_500000us",
        "trade_impact_half_life_proxy",
    ]:
        assert specs.feature_spec_by_name(name).transform_key == TransformKey.RATIO_BOUNDED


def test_short_horizon_price_and_ofi_features_are_fast():
    assert specs.feature_spec_by_name("micro_ret_bps_200000us").transform_key == TransformKey.IDENTITY_EWMA_FAST
    assert specs.feature_spec_by_name("spread_bps").transform_key == TransformKey.IDENTITY_EWMA_FAST
    assert specs.feature_spec_by_name("ofi_l3").transform_key == TransformKey.IDENTITY_EWMA_FAST
    assert specs.feature_spec_by_name("ofi_l5_sum_over_depth_200000us").transform_key == TransformKey.RATIO_BOUNDED


def test_short_horizon_trade_flow_features_are_fast_or_bounded():
    assert specs.feature_spec_by_name("signed_notional_flow_usd_200000us").transform_key == TransformKey.SIGNED_LOG1P_EWMA_FAST
    assert specs.feature_spec_by_name("trade_count_per_second_200000us").transform_key == TransformKey.LOG1P_POS_EWMA_FAST
    assert specs.feature_spec_by_name("signed_trade_count_imbalance_200000us").transform_key == TransformKey.RATIO_BOUNDED


def test_scale_depth_features_are_medium():
    assert specs.feature_spec_by_name("bid_depth_notional_5bps").transform_key == TransformKey.LOG1P_POS_EWMA_MEDIUM
    assert specs.feature_spec_by_name("cvd_change_usd_500000us").transform_key == TransformKey.SIGNED_LOG1P_EWMA_MEDIUM


def test_regime_features_are_slow_or_bounded():
    assert specs.feature_spec_by_name("regime_volume_ewma_3000000us").transform_key == TransformKey.LOG1P_POS_EWMA_SLOW
    assert specs.feature_spec_by_name("return_std_bps_200000us").transform_key == TransformKey.IDENTITY_EWMA_SLOW
    assert specs.feature_spec_by_name("down_up_vol_imbalance_500000us").transform_key == TransformKey.RATIO_BOUNDED


def test_every_feature_has_explicit_transform_policy():
    supported = set(TransformKey)
    for spec in specs.FEATURE_SPECS:
        assert spec.transform_key in supported


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
