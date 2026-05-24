import inspect
import math

import numpy as np

from feature_event_candidates_round2 import (
    FAMILY_BY_FEATURE,
    ROUND2_REQUESTED_FEATURES,
    NovelMicrostructureCandidatePack,
)


def _feed(pack, events):
    for ev in events:
        pack.on_event(ev)


def test_exact_feature_list_and_metadata_keys():
    pack = NovelMicrostructureCandidatePack()
    assert len(ROUND2_REQUESTED_FEATURES) == 136
    assert len(set(ROUND2_REQUESTED_FEATURES)) == 136
    assert pack.feature_names() == ROUND2_REQUESTED_FEATURES
    assert set(pack.metadata()) == set(ROUND2_REQUESTED_FEATURES)


def test_metadata_family_specificity_and_exact_coverage():
    pack = NovelMicrostructureCandidatePack()
    meta = pack.metadata()
    assert set(FAMILY_BY_FEATURE) == set(ROUND2_REQUESTED_FEATURES)

    required = {
        "trade_size_hhi_1000ms": "trade_concentration",
        "trade_size_max_over_ewma_3000ms": "trade_concentration",
        "obi_realized_vol_1000ms": "realized_vol",
        "event_interarrival_cv_1000ms": "event_timing",
        "trade_sign_entropy_1000ms": "trade_sign",
        "same_side_replenishment_after_depletion_200ms": "book_resilience",
        "post_buy_ask_net_replenishment_over_trade_200ms": "quote_response",
        "depth_slope_bid_1_to_10": "book_shape",
        "no_trade_no_book_change_age_ms": "quiet_state",
        "mid_unchanged_and_depth_stable_ms": "quiet_state",
        "best_bid_price_age_ms": "touch_age",
        "quote_lifetime_cv_3000ms": "quote_lifetime",
        "touch_flicker_score_1000ms": "l1_flicker",
        "post_buy_ask_cancel_over_trade_200ms": "quote_response",
        "last_buy_mid_impact_bps_since_trade": "trade_impact",
        "last_sell_mid_impact_bps_since_trade": "trade_impact",
        "last_trade_mid_impact_signed_bps": "trade_impact",
        "bid_depth_centroid_bps_10bps": "depth_centroid",
        "bid_near_touch_depth_share_10bps": "depth_centroid",
        "far_depth_wall_ratio_10_to_25bps": "depth_centroid",
        "spread_widen_event_count_1000ms": "spread_regime",
        "mid_price_path_efficiency_1000ms": "mid_path",
        "event_interarrival_entropy_3000ms": "event_irregularity",
        "trade_arrival_clumpiness_3000ms": "event_irregularity",
        "thin_book_with_trade_burst_score_500ms": "stress_regime",
        "trade_burst_without_book_replenishment_score_1000ms": "stress_regime",
        "impact_per_notional_high_and_replenishment_low_score_1000ms": "stress_regime",
    }
    for k, fam in required.items():
        assert FAMILY_BY_FEATURE[k] == fam
        assert meta[k]["candidate_family"] == fam


def test_source_guard_no_placeholder_families():
    src = inspect.getsource(NovelMicrostructureCandidatePack.emit)
    for bad in [
        '{n: 0.0 for n in ROUND2_REQUESTED_FEATURES}',
        'for k, v in const.items()',
        'for n in ROUND2_REQUESTED_FEATURES: o[n] = o[n]',
        'setdefault(',
        '.get(k, 0.0)',
        '.get(name, 0.0)',
    ]:
        assert bad not in src


def test_all_features_explicitly_assigned():
    src = inspect.getsource(NovelMicrostructureCandidatePack.emit)
    for n in ROUND2_REQUESTED_FEATURES:
        assert f'o["{n}"]' in src or f"o['{n}']" in src


def test_exact_formula_hhi_entropy_flip():
    p = NovelMicrostructureCandidatePack()
    events = [
        ("ob", 0, 1, 1, [(99.99, 10), (99.98, 8), (99.97, 5), (99.96, 4), (99.95, 3)], [(100.01, 10), (100.02, 8), (100.03, 5), (100.04, 4), (100.05, 3)]),
        ("trade", 100, 2, 100.0, 1.0, 1, 1, 0),
        ("trade", 200, 3, 100.0, 2.0, 1, 1, 0),
        ("trade", 300, 4, 100.0, 3.0, -1, -1, 0),
    ]
    _feed(p, events)
    out = p.emit()
    notionals = np.asarray([100.0, 200.0, 300.0])
    assert math.isclose(out["trade_size_hhi_1000ms"], float((notionals**2).sum() / notionals.sum() ** 2), rel_tol=1e-12)
    assert math.isclose(out["largest_trade_share_notional_1000ms"], 0.5, rel_tol=1e-12)
    assert math.isclose(out["buy_trade_size_hhi_1000ms"], (100**2 + 200**2) / (300**2), rel_tol=1e-12)
    assert math.isclose(out["sell_trade_size_hhi_1000ms"], 1.0, rel_tol=1e-12)
    p_plus = 2 / 3
    p_minus = 1 / 3
    ent = -((p_plus * math.log(p_plus)) + (p_minus * math.log(p_minus))) / math.log(2.0)
    assert math.isclose(out["trade_sign_entropy_1000ms"], ent, rel_tol=1e-12)
    assert math.isclose(out["trade_sign_flip_rate_1000ms"], 0.5, rel_tol=1e-12)
