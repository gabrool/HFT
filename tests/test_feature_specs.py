import pytest

from mmrt.features import specs


def test_feature_counts_and_no_duplicates():
    assert specs.LEGACY_CORE_FEATURE_COUNT == 172
    assert specs.EVENT_CONTEXT_FEATURE_COUNT == 6
    assert specs.FEATURE_COUNT == 178
    assert len(specs.LEGACY_CORE_FEATURE_NAMES) == 172
    assert len(specs.LEGACY_EVENT_CONTEXT_FEATURE_NAMES) == 6
    assert len(specs.FEATURE_NAMES) == 178
    assert len(set(specs.FEATURE_NAMES)) == 178
    assert len(specs.FEATURE_SPECS) == 178
    assert [s.index for s in specs.FEATURE_SPECS] == list(range(178))


def test_no_legacy_aux_core_split():
    for name in specs.__all__:
        assert "AUX" not in name
        assert "CORE_DIM" not in name
        assert "TOTAL_DIM" not in name
        assert "feature_" + "dim_core" not in name
        assert "feature_" + "dim_total" not in name
    record = specs.schema_record()
    for key in ("aux_dim", "aux_names", "feature_" + "dim_core", "feature_" + "dim_total"):
        assert key not in record


def test_canonical_names_are_microsecond_native():
    assert all("ms" not in n for n in specs.FEATURE_NAMES)
    assert "micro_ret_bps_200000us" in specs.FEATURE_NAMES
    assert "micro_ret_bps_200ms" not in specs.FEATURE_NAMES
    assert "time_since_trade_us" in specs.FEATURE_NAMES
    assert "best_bid_size_age_us" in specs.FEATURE_NAMES
    assert "log_dt_decision_us" in specs.FEATURE_NAMES
    assert "log_events_100000us" in specs.FEATURE_NAMES


def test_legacy_mapping_round_trip():
    pairs = {
        "micro_ret_bps_200ms": "micro_ret_bps_200000us",
        "ofi_l1_accel_200_minus_500ms": "ofi_l1_accel_200000us_minus_500000us",
        "time_since_last_buy_trade_ms": "time_since_last_buy_trade_us",
        "best_ask_size_age_ms": "best_ask_size_age_us",
        "log_events_3000ms": "log_events_3000000us",
    }
    for old, new in pairs.items():
        assert specs.canonical_name_for_legacy_feature(old) == new
        assert specs.legacy_name_for_feature(new) == old


def test_feature_order_preserved_with_context_tail():
    assert specs.FEATURE_NAMES[0] == "micro_ret_bps_200000us"
    assert specs.FEATURE_NAMES[1] == "micro_ret_bps_500000us"
    assert specs.FEATURE_NAMES[171] == "trade_impact_half_life_proxy"
    assert specs.FEATURE_NAMES[172:] == (
        "log_dt_decision_us",
        "log_events_100000us",
        "log_events_200000us",
        "log_events_500000us",
        "log_events_1000000us",
        "log_events_3000000us",
    )


def test_removed_legacy_features_not_present():
    removed = (
        "micro_premia",
        "micro_minus_mid_over_spread",
        "obi_l3",
        "obi_l5",
        "micro_l1_minus_micro_l10_bps",
        "ofi_l1_sum_over_depth_1000000us",
        "ofi_l3_sum_over_depth_1000000us",
        "ofi_l3_sum_over_depth_500000us",
        "ofi_l3_sum_over_depth_200000us",
        "ofi_l10",
        "ofi_l1_pressure_ewma_200000us",
        "ofi_l1_pressure_ewma_500000us",
        "ofi_l1_pressure_ewma_1000000us",
        "bid_l1_depletion_over_depth_200000us",
        "regime_volume_ewma_1000000us",
    )
    for name in removed:
        assert name not in specs.FEATURE_NAMES


def test_must_keep_features_present():
    keep = (
        "micro_minus_mid_bps",
        "obi_l1",
        "obi_l10",
        "micro_l10_minus_mid_bps",
        "ofi_l5_sum_over_depth_1000000us",
        "ofi_l5_sum_over_depth_500000us",
        "ofi_l5_sum_over_depth_200000us",
        "ofi_l10_over_depth_5bps",
        "ofi_l1_pressure_over_depth_5bps_200000us",
        "ofi_l1_pressure_over_depth_5bps_500000us",
        "ofi_l1_pressure_over_depth_5bps_1000000us",
        "bid_l1_rem_rate_over_depth_200000us",
        "regime_volume_ewma_500000us",
        "regime_volume_ewma_3000000us",
    )
    for name in keep:
        assert name in specs.FEATURE_NAMES


def test_window_inference():
    assert specs.infer_windows_us_from_legacy_name("micro_ret_bps_200ms") == (200_000,)
    assert specs.infer_windows_us_from_legacy_name("ofi_l1_accel_200_minus_500ms") == (200_000, 500_000)
    assert specs.infer_windows_us_from_legacy_name("time_since_trade_ms") == ()
    assert specs.infer_windows_us_from_legacy_name("ask_depth_centroid_bps_25bps") == ()
    assert specs.required_windows_us() == (100_000, 200_000, 500_000, 1_000_000, 3_000_000)


def test_book_depth_tardis_snapshot25_compatibility():
    assert specs.max_required_book_depth() == 20
    assert specs.max_required_book_depth() <= specs.REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH
    assert specs.REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH == 25
    assert specs.feature_spec_by_name("micro_l10_minus_mid_bps").required_book_depth == 10
    assert specs.feature_spec_by_name("bid_depth_notional_5bps").required_book_depth == 20
    assert specs.feature_spec_by_name("spread_bps").required_book_depth == 1


def test_source_owner_family_examples():
    spread = specs.feature_spec_by_name("spread_bps")
    assert spread.source == specs.FeatureSource.BOOK
    assert spread.owner == specs.FeatureOwner.BOOK_STATE
    signed = specs.feature_spec_by_name("signed_notional_flow_usd_200000us")
    assert signed.source == specs.FeatureSource.TRADE
    assert signed.owner == specs.FeatureOwner.TRADE_STATE
    assert specs.feature_spec_by_name("cvd_change_usd_500000us").family == specs.FeatureFamily.CVD
    absorption = specs.feature_spec_by_name("absorption_bid_200000us")
    assert absorption.source == specs.FeatureSource.CROSS
    assert absorption.owner == specs.FeatureOwner.ENGINE
    assert specs.feature_spec_by_name("post_buy_trade_ask_replenishment_200000us").source == specs.FeatureSource.CROSS
    ctx = specs.feature_spec_by_name("log_events_3000000us")
    assert ctx.source == specs.FeatureSource.EVENT_CONTEXT
    assert ctx.owner == specs.FeatureOwner.ENGINE
    assert ctx.family == specs.FeatureFamily.EVENT_CONTEXT


def test_transform_key_examples():
    assert specs.feature_spec_by_name("log_dt_decision_us").transform_key == specs.TransformKey.LOG1P_POS_NO_EWMA
    assert specs.feature_spec_by_name("time_since_trade_us").transform_key == specs.TransformKey.TIME_LOG1P_NO_EWMA
    assert specs.feature_spec_by_name("last_trade_side_sign").transform_key == specs.TransformKey.SIGN_NO_EWMA
    assert specs.feature_spec_by_name("signed_notional_flow_usd_200000us").transform_key == specs.TransformKey.SIGNED_LOG1P_EWMA
    assert specs.feature_spec_by_name("trade_count_per_second_500000us").transform_key == specs.TransformKey.LOG1P_POS_EWMA
    assert specs.feature_spec_by_name("spread_bps").transform_key == specs.TransformKey.IDENTITY_EWMA_FAST


def test_lookup_helpers():
    assert specs.feature_index(specs.FEATURE_NAMES[10]) == 10
    assert specs.feature_name(10) == specs.FEATURE_NAMES[10]
    assert specs.feature_spec_by_name(specs.FEATURE_NAMES[10]).index == 10
    with pytest.raises(IndexError):
        specs.feature_name(-1)
    with pytest.raises(IndexError):
        specs.feature_name(178)
    with pytest.raises(KeyError):
        specs.feature_index("missing")
    with pytest.raises(KeyError):
        specs.feature_spec_by_name("missing")


def test_hashes_are_stable_shape():
    assert specs.FEATURE_NAMES_HASH == specs.feature_names_hash(specs.FEATURE_NAMES)
    assert specs.FEATURE_SPECS_HASH == specs.feature_specs_hash(specs.FEATURE_SPECS)
    assert len(specs.FEATURE_NAMES_HASH) == 12
    assert len(specs.FEATURE_SPECS_HASH) == 12


def test_schema_record():
    record = specs.schema_record()
    assert "feature_schema_version" in record
    assert record["feature_count"] == 178
    assert "feature_names_hash" in record
    assert "feature_specs_hash" in record
    assert record["feature_dtype"] == "float32"
    assert record["time_unit"] == "us"
    assert record["required_tardis_book_snapshot_depth"] == 25
    assert record["max_required_book_feature_depth"] == 20
    assert "source_counts" in record
    assert "owner_counts" in record
    assert "family_counts" in record
    assert sum(record["source_counts"].values()) == 178


def test_no_heavy_imports():
    forbidden = (
        "num" + "py",
        "po" + "lars",
        "num" + "ba",
        "tor" + "ch",
        "pan" + "das",
        "mmrt.data.tardis_csv",
        "mmrt.data.event_merge",
        "mmrt.data.quality",
    )
    loaded_by_specs = set(getattr(specs, "__dict__", {}))
    assert all(name not in loaded_by_specs for name in forbidden)


def test_specs_have_no_future_leakage_concepts():
    bad = ("future" + "_ret", "future" + "_mid", "label", "target", "lookahead", "peek")
    tokens = list(specs.FEATURE_NAMES)
    tokens.extend(specs.legacy_feature_names())
    tokens.extend([s.formula_group for s in specs.FEATURE_SPECS])
    tokens.extend(specs.__all__)
    lower = [t.lower() for t in tokens]
    for tok in bad:
        assert all(tok not in text for text in lower)


def test_specs_are_not_computation():
    bad = ("compute", "update_state", "apply_row", "transform_engine", "label_builder")
    lower = [name.lower() for name in dir(specs)]
    for tok in bad:
        assert all(tok not in name for name in lower)
