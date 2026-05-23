from pathlib import Path

from CMSSL17 import FEATURE_AUX_TAIL, FeatureEngine


REMOVED_FEATURES = {
    "utc_hour_sin",
    "utc_dow_sin",
    "is_weekend",
    "signed_trade_count_imbalance_1000ms",
    "regime_flow_imbalance_3000ms",
    "tick_sign_imbalance_500ms",
    "vwap_vs_mid_bps_1000ms",
    "trade_imbalance_notional_1000ms",
    "micro_l3_minus_mid_bps",
    "gap_a_bps",
    "depth_imbalance_5bps_mean_3000ms",
    "cvd_minus_ema_usd_1000ms",
    "cvd_slope_usd_per_sec_200ms",
    "signed_trade_premium_bps_volume_weighted_200ms",
    "ob_update_rate_1000ms",
}


def deep_snapshot_ob(ts: int, n_levels: int = 60):
    bids = tuple((100.0 - 0.5 * i, 1.0 + 0.01 * i) for i in range(n_levels))
    asks = tuple((101.0 + 0.5 * i, 1.0 + 0.01 * i) for i in range(n_levels))
    return ("ob", ts, 1, 1, bids, asks)


def test_removed_features_not_in_emitted_schema() -> None:
    names = FeatureEngine().feature_names()
    assert len(names) == 153
    assert not (REMOVED_FEATURES & set(names))


def test_core_and_total_feature_dims_are_consistent() -> None:
    fe = FeatureEngine()
    names = fe.feature_names()
    assert fe.core_feature_dim() == len(names) == 153
    assert fe.aux_dim() == 6
    assert fe.feature_dim() == 159
    assert fe.feature_dim() == fe.core_feature_dim() + fe.aux_dim()


def test_decision_event_feature_vector_matches_pruned143_schema() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1000, n_levels=60))
    assert result is not None
    assert result.features.shape == (143,)
    assert result.features.shape[0] == len(fe.feature_names())


def test_removed_features_have_no_source_references() -> None:
    for path in ["offline_ingest.py", "CMSSL17.py"]:
        text = Path(path).read_text(encoding="utf-8")
        for name in REMOVED_FEATURES:
            assert name not in text


def test_no_stale_pruned_schema_references() -> None:
    paths = ["CMSSL17.py", "offline_ingest.py"]
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        assert "pruned153" not in text
        assert "pruned159" not in text
        assert "pruned144" not in text


def test_retained_neighbor_features_still_present() -> None:
    core_names = set(FeatureEngine().feature_names())
    all_names = core_names | set(FEATURE_AUX_TAIL)
    retained = {
        "trade_imbalance_notional_500ms",
        "vwap_vs_mid_bps_200ms",
        "vwap_vs_mid_bps_500ms",
        "micro_l5_minus_mid_bps",
        "micro_l10_minus_mid_bps",
        "gap_b_bps",
        "cvd_minus_ema_usd_500ms",
        "cvd_slope_usd_per_sec_500ms",
        "cvd_slope_usd_per_sec_1000ms",
        "ob_update_rate_200ms",
        "ob_update_rate_500ms",
        "log_events_1000ms",
    }
    missing = retained - all_names
    assert not missing


def test_retained_aux_features_still_present() -> None:
    aux_names = set(FEATURE_AUX_TAIL)
    assert "log_events_1000ms" in aux_names
