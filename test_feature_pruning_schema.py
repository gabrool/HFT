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


def test_removed_features_not_in_emitted_schema() -> None:
    names = FeatureEngine().feature_names()
    assert len(names) == 153
    assert not (REMOVED_FEATURES & set(names))


def test_removed_features_have_no_source_references() -> None:
    text = Path("offline_ingest.py").read_text(encoding="utf-8")
    for name in REMOVED_FEATURES:
        assert name not in text
