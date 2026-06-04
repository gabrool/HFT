import re
import subprocess
import sys
from pathlib import Path

from mmrt.features import specs
from mmrt.features.specs import FeatureFamily, FeatureOwner, FeatureSource, TransformKey


def test_feature_count_and_order():
    assert specs.CORE_FEATURE_COUNT == 44
    assert specs.EVENT_CONTEXT_FEATURE_COUNT == 4
    assert specs.FEATURE_COUNT == 48
    assert len(specs.FEATURE_SPECS) == 48
    assert len(specs.FEATURE_NAMES) == 48
    assert len(set(specs.FEATURE_NAMES)) == 48
    assert specs.FEATURE_NAMES[0] == "mid_slope_bps_per_sec_1000000us"
    assert specs.FEATURE_NAMES[-4:] == (
        "log_events_200000us",
        "log_events_500000us",
        "log_events_1000000us",
        "log_events_3000000us",
    )


def test_context_tail_after_core_features():
    assert specs.FEATURE_NAMES[: specs.CORE_FEATURE_COUNT] == tuple(
        specs.legacy_name_to_canonical_name(n) for n in specs.CORE_FEATURE_NAMES
    )
    assert specs.FEATURE_NAMES[-4:] == tuple(
        specs.legacy_name_to_canonical_name(n) for n in specs.EVENT_CONTEXT_FEATURE_NAMES
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


def test_active_registry_matches_current_feature_subset_union():
    from mmrt.linear import head_feature_presets as hp

    subset_cols = set()
    for cols in hp.FEATURE_SUBSET_COLUMNS_BY_HEAD.values():
        subset_cols.update(cols)

    computed_cols = {f"x_{name}" for name in specs.FEATURE_NAMES}
    assert computed_cols == subset_cols


def test_corr90_removed_features_absent_from_registry_and_subsets():
    from mmrt.linear import head_feature_presets as hp

    removed = {
        "x_depth_imbalance_5bps_mean_1000000us",
        "x_ask_depth_notional_5bps",
        "x_bid_depth_notional_5bps",
        "x_micro_minus_mid_bps",
        "x_cvd_change_usd_1000000us",
    }

    computed_cols = {f"x_{name}" for name in specs.FEATURE_NAMES}
    subset_cols = set()
    for cols in hp.FEATURE_SUBSET_COLUMNS_BY_HEAD.values():
        subset_cols.update(cols)

    assert not removed.intersection(computed_cols)
    assert not removed.intersection(subset_cols)


def test_no_pruned_output_assignment_remains():
    dropped = {
        "depth_imbalance_5bps_mean_1000000us",
        "ask_depth_notional_5bps",
        "bid_depth_notional_5bps",
        "micro_minus_mid_bps",
        "cvd_change_usd_1000000us",
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
    cases = {
        "mid_slope_bps_per_sec_1000000us": (FeatureSource.BOOK, FeatureOwner.BOOK_STATE, FeatureFamily.PRICE, 1),
        "trade_count_per_second_200000us": (FeatureSource.TRADE, FeatureOwner.TRADE_STATE, FeatureFamily.TRADE_FLOW, 0),
        "absorption_bid_1000000us": (FeatureSource.CROSS, FeatureOwner.ENGINE, FeatureFamily.ABSORPTION, 20),
        "log_events_200000us": (FeatureSource.EVENT_CONTEXT, FeatureOwner.ENGINE, FeatureFamily.EVENT_CONTEXT, 0),
        "trade_side_quote_response_asymmetry_500000us": (FeatureSource.CROSS, FeatureOwner.ENGINE, FeatureFamily.CROSS_SIGNAL, 20),
    }
    for name, (source, owner, family, depth) in cases.items():
        spec = specs.feature_spec_by_name(name)
        assert spec.source == source
        assert spec.owner == owner
        assert spec.family == family
        assert spec.required_book_depth == depth


def test_feature_schema_version_is_current_active_schema():
    assert specs.FEATURE_SCHEMA_VERSION == "mmrt_feature_schema_v3_snapshot25_trades_active44_ctx4_feature_subset_corr90"
    assert "legacy" not in specs.FEATURE_SCHEMA_VERSION


def test_public_schema_has_no_aux_core_split():
    schema = specs.schema_record()
    text = str(schema)
    assert "feature" + "_dim_" + "core" not in text
    assert "feature" + "_dim_" + "total" not in text
    assert "AU" + "X_DIM" not in text


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


def test_bounded_ratio_features_are_ratio_bounded():
    for name in [
        "zero_tick_fraction_1000000us",
        "trade_sign_entropy_3000000us",
        "trade_side_quote_response_asymmetry_500000us",
        "depth_imbalance_within_1bps",
    ]:
        assert specs.feature_spec_by_name(name).transform_key == TransformKey.RATIO_BOUNDED


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
forbidden = ("numpy", "polars", "numba", "torch", "pandas", "pyarrow")
bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit("forbidden imports loaded by specs: " + repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)
