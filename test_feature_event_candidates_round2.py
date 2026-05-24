import inspect
import math
import re

import numpy as np

from feature_event_candidates_round2 import (
    ROUND2_REQUESTED_FEATURES,
    NovelMicrostructureCandidatePack,
)


def _feed(pack, events):
    for ev in events:
        pack.on_event(ev)


def test_structural_guard_feature_count_uniqueness_metadata_and_source_denylists():
    p = NovelMicrostructureCandidatePack()
    meta = p.metadata()

    assert len(ROUND2_REQUESTED_FEATURES) == 136
    assert len(set(ROUND2_REQUESTED_FEATURES)) == 136
    assert p.feature_names() == ROUND2_REQUESTED_FEATURES
    assert set(meta) == set(ROUND2_REQUESTED_FEATURES)

    for name, m in meta.items():
        assert m["candidate_kind"] == "event_derived"
        assert m["uses_book_state"] is True
        assert m["uses_trade_state"] is True
        assert m["expected_target"] == "all"
        if any(f"_{w}ms" in name for w in (200, 500, 1000, 3000)):
            assert m["candidate_horizon_ms"] in (200, 500, 1000, 3000)
        else:
            assert m["candidate_horizon_ms"] is None

    src = inspect.getsource(NovelMicrostructureCandidatePack.emit)

    forbidden_patterns = [
        "TODO",
        "pass #",
        "NotImplemented",
        "raise NotImplementedError",
    ]
    for pat in forbidden_patterns:
        assert pat not in src

    loop_default_assignment_guard = re.compile(
        r"for\s+[^\n]+:\s*\n(?:\s+.*\n){0,3}?\s*o\[(?:\"|\').+?(?:\"|\')\]\s*=\s*0(?:\.0)?",
        re.MULTILINE,
    )
    assert loop_default_assignment_guard.search(src) is None


def test_emit_explicit_assignments_for_all_requested_names():
    src = inspect.getsource(NovelMicrostructureCandidatePack.emit)
    for name in ROUND2_REQUESTED_FEATURES:
        assert (f'o["{name}"]' in src) or (f"o['{name}']" in src), name


def test_metadata_family_specificity_and_exact_coverage():
    p = NovelMicrostructureCandidatePack()
    meta = p.metadata()

    reps = {
        "trade_size_hhi_1000ms": "trade_concentration",
        "obi_realized_vol_1000ms": "realized_vol",
        "event_interarrival_cv_1000ms": "event_timing",
        "trade_sign_entropy_1000ms": "trade_sign",
        "same_side_replenishment_after_depletion_200ms": "book_resilience",
        "depth_slope_bid_1_to_10": "book_shape",
        "best_bid_price_age_ms": "touch_age",
        "quote_lifetime_cv_3000ms": "quote_lifetime",
        "touch_flicker_score_1000ms": "l1_flicker",
        "post_buy_ask_cancel_over_trade_200ms": "quote_response",
        "last_buy_mid_impact_bps_since_trade": "trade_impact",
        "bid_depth_centroid_bps_10bps": "depth_centroid",
        "spread_widen_event_count_1000ms": "spread_regime",
        "mid_price_path_efficiency_1000ms": "mid_path",
        "event_interarrival_entropy_3000ms": "event_irregularity",
        "thin_book_with_trade_burst_score_500ms": "stress_regime",
    }
    for k, fam in reps.items():
        assert meta[k]["candidate_family"] == fam

    covered = set(reps)
    fallback = {k for k in ROUND2_REQUESTED_FEATURES if k not in covered}
    assert all(meta[k]["candidate_family"] == "event_timing" for k in fallback)


def test_deterministic_synthetic_sequence_broad_nonzero_coverage():
    p = NovelMicrostructureCandidatePack()

    events = [
        ("ob", 0, 1, 1,
         [(100.0, 5.0), (99.5, 4.0), (99.0, 3.0), (98.5, 2.0), (98.0, 1.0)],
         [(101.0, 5.0), (101.5, 4.0), (102.0, 3.0), (102.5, 2.0), (103.0, 1.0)]),
        ("trade", 50, 2, 100.8, 2.0, 1, 1, 0),
        ("ob", 100, 3, 0, [(100.0, 4.0), (99.5, 6.0)], [(101.0, 3.0), (101.5, 5.0)]),
        ("trade", 170, 4, 100.7, 1.0, -1, -1, 0),
        ("ob", 260, 5, 0, [(100.2, 5.0), (99.7, 4.0)], [(101.2, 6.0), (101.7, 2.0)]),
        ("trade", 340, 6, 100.9, 3.0, 1, 1, 0),
        ("ob", 430, 7, 0, [(100.1, 8.0), (99.6, 2.0)], [(101.3, 2.0), (101.8, 7.0)]),
        ("trade", 520, 8, 100.85, 2.5, -1, -1, 0),
        ("ob", 620, 9, 0, [(100.3, 5.0), (99.8, 5.0)], [(101.4, 3.0), (101.9, 6.0)]),
        ("trade", 760, 10, 101.0, 1.5, 1, 1, 0),
        ("ob", 900, 11, 0, [(100.4, 5.5), (99.9, 4.5)], [(101.5, 4.5), (102.0, 5.5)]),
    ]
    _feed(p, events)
    out = p.emit()

    nz = sum(abs(float(v)) > 1e-12 for v in out.values())
    assert nz >= 80
    for name in [
        "trade_size_hhi_3000ms",
        "trade_sign_entropy_1000ms",
        "event_interarrival_cv_1000ms",
        "depth_slope_bid_1_to_10",
        "bid_depth_centroid_bps_10bps",
        "spread_widen_event_count_1000ms",
        "max_event_gap_1000ms",
    ]:
        assert abs(out[name]) > 0.0, name


def test_exact_formula_hhi_largest_share_trade_sign_and_queue_cliff_centroid_checks():
    p = NovelMicrostructureCandidatePack()
    events = [
        ("ob", 0, 1, 1, [(100, 5), (99, 4), (98, 1), (97, 1), (96, 1)], [(101, 5), (102, 4), (103, 1), (104, 1), (105, 1)]),
        ("trade", 100, 2, 100.0, 1.0, 1, 1, 0),
        ("trade", 200, 3, 100.0, 2.0, -1, -1, 0),
        ("trade", 300, 4, 100.0, 3.0, 1, 1, 0),
        ("ob", 400, 5, 0, [(100, 10), (99, 4), (98, 1), (97, 1), (96, 1)], [(101, 8), (102, 4), (103, 1), (104, 1), (105, 1)]),
    ]
    _feed(p, events)
    out = p.emit()

    notionals = np.asarray([100.0, 200.0, 300.0])
    expected_hhi = float((notionals * notionals).sum() / (notionals.sum() ** 2))
    expected_largest = float(notionals.max() / notionals.sum())
    assert math.isclose(out["trade_size_hhi_1000ms"], expected_hhi, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out["largest_trade_share_notional_1000ms"], expected_largest, rel_tol=1e-12, abs_tol=1e-12)

    p_plus = 2 / 3
    p_minus = 1 / 3
    expected_entropy = -(p_plus * math.log(p_plus) + p_minus * math.log(p_minus)) / math.log(2.0)
    expected_flip = 1.0
    assert math.isclose(out["trade_sign_entropy_1000ms"], expected_entropy, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out["trade_sign_flip_rate_1000ms"], expected_flip, rel_tol=1e-12, abs_tol=1e-12)

    assert out["bid_queue_cliff_ratio_l1_l5"] == 0.0
    assert out["ask_queue_cliff_ratio_l1_l5"] == 0.0
    assert out["bid_depth_centroid_bps_10bps"] > 0
    assert out["ask_depth_centroid_bps_10bps"] > 0


def test_causality_prefix_equality_no_future_leakage():
    prefix = [
        ("ob", 0, 1, 1, [(100, 3), (99, 2)], [(101, 3), (102, 2)]),
        ("trade", 100, 2, 100.5, 1, 1, 1, 0),
        ("ob", 200, 3, 0, [(100, 4)], [(101, 2)]),
        ("trade", 300, 4, 100.4, 2, -1, -1, 0),
    ]
    suffix = [
        ("ob", 450, 5, 0, [(100.2, 5)], [(101.2, 4)]),
        ("trade", 550, 6, 100.6, 3, 1, 1, 0),
    ]

    p1 = NovelMicrostructureCandidatePack()
    _feed(p1, prefix)
    o1 = p1.emit()

    p2 = NovelMicrostructureCandidatePack()
    _feed(p2, prefix + suffix)
    _ = p2.emit()

    p3 = NovelMicrostructureCandidatePack()
    _feed(p3, prefix)
    o3 = p3.emit()

    assert o1 == o3


def test_non_mutating_interarrival_queries_and_first_snapshot_no_fake_churn():
    p = NovelMicrostructureCandidatePack()

    p.on_event(("ob", 100, 1, 1, [(100, 2), (99, 1)], [(101, 2), (102, 1)]))
    p.on_event(("ob", 200, 2, 0, [(100, 3)], [(101, 1)]))

    pre = p.event_i.values(1000, p.ts).copy()
    _ = p.event_i.p90_over_p10(1000, p.ts)
    _ = p.event_i.clumpiness(1000, p.ts)
    _ = p.event_i.entropy(1000, p.ts)
    post = p.event_i.values(1000, p.ts)
    assert np.array_equal(pre, post)

    p0 = NovelMicrostructureCandidatePack()
    p0.on_event(("ob", 0, 1, 1, [(100, 5)], [(101, 5)]))
    out0 = p0.emit()
    assert out0["spread_widen_event_count_1000ms"] == 0.0
    assert out0["spread_tighten_event_count_1000ms"] == 0.0
    assert out0["mid_unchanged_and_depth_stable_ms"] == 0.0
