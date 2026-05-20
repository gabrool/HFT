from CMSSL17 import FeatureEngine, AUX_DIM, LOOKBACK

REMOVED = {
    "utc_hour_cos",
    "utc_dow_cos",
    "trade_toxicity_notional_1000ms",
    "depth_5bps_z_1000ms",
    "zero_tick_fraction_500ms",
    "max_abs_return_bps_1000ms",
    "spread_widening_slope_bps_per_sec_3000ms",
    "spread_change_count_1000ms",
    "spread_change_count_200ms",
    "max_abs_return_bps_3000ms",
    "last_trade_notional_usd",
    "last_is_zero_tick",
    "cvd_minus_ema_usd_200ms",
}

RETAINED = {
    "utc_hour_sin",
    "utc_dow_sin",
    "is_weekend",
    "trade_toxicity_notional_200ms",
    "trade_toxicity_notional_500ms",
    "zero_tick_fraction_200ms",
    "zero_tick_fraction_1000ms",
    "cvd_minus_ema_usd_500ms",
    "cvd_minus_ema_usd_1000ms",
    "max_abs_return_bps_500ms",
    "depth_5bps_z_500ms",
    "depth_5bps_z_3000ms",
    "spread_widening_slope_bps_per_sec_500ms",
    "spread_widening_slope_bps_per_sec_1000ms",
    "spread_change_count_500ms",
}

def test_pruned_features_removed_and_dims():
    fe = FeatureEngine()
    names = fe.feature_names()

    assert LOOKBACK == 10
    assert len(names) == len(set(names))
    assert len(names) == fe.core_feature_dim()
    assert fe.feature_dim() == fe.core_feature_dim() + AUX_DIM
    assert fe.core_feature_dim() == 159
    assert fe.feature_dim() == 165

    for name in REMOVED:
        assert name not in names
    for name in RETAINED:
        assert name in names
