from pathlib import Path

from CMSSL17 import FeatureEngine


REMOVED_FEATURES = {
    "utc_hour_sin",
    "utc_dow_sin",
    "is_weekend",
    "signed_trade_count_imbalance_1000ms",
    "regime_flow_imbalance_3000ms",
    "tick_sign_imbalance_500ms",
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
    assert fe.feature_dim() == fe.core_feature_dim() + 6


def test_decision_event_feature_vector_matches_pruned153_schema() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1000, n_levels=60))
    assert result is not None
    assert result.features.shape == (153,)
    assert result.features.shape[0] == len(fe.feature_names())


def test_removed_features_have_no_source_references() -> None:
    for path in ["offline_ingest.py", "CMSSL17.py"]:
        text = Path(path).read_text(encoding="utf-8")
        for name in REMOVED_FEATURES:
            assert name not in text


def test_no_stale_pruned159_schema_references() -> None:
    paths = [
        "CMSSL17.py",
        "CMSSL17_offline.py",
        "offline_ingest.py",
        "linear_offline.py",
        "feature_audit.py",
    ]
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        assert "pruned159" not in text
        assert "feature_transform_spec_v2_pruned159" not in text
