import math
import subprocess
import sys

import numpy as np
import pytest

from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.features import engine as eg
from mmrt.features import kernels as k
from mmrt.features.book_state import BOOK_DEPTH, BookSnapshotInput
from mmrt.features.trade_state import BUY_SIDE_CODE, SELL_SIDE_CODE, UNKNOWN_SIDE_CODE, TradeInput
from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_SPECS,
    FeatureSource,
    feature_spec_by_name,
)


def make_snapshot(local_ts_us=2_000_000, mid=100.0, spread=0.10, bid_sz0=10.0, ask_sz0=12.0, bid_size_offset=0.0, ask_size_offset=0.0):
    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0
    bid_px = best_bid - 0.1 * np.arange(BOOK_DEPTH, dtype=np.float64)
    ask_px = best_ask + 0.1 * np.arange(BOOK_DEPTH, dtype=np.float64)
    bid_sz = bid_sz0 + bid_size_offset + np.arange(BOOK_DEPTH, dtype=np.float64)
    ask_sz = ask_sz0 + ask_size_offset + np.arange(BOOK_DEPTH, dtype=np.float64)
    return BookSnapshotInput(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us,
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
    )


def make_trade(local_ts_us=2_000_000, price=100.0, amount=1.0, side_code=BUY_SIDE_CODE):
    return TradeInput(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us,
        price=price,
        amount=amount,
        side_code=side_code,
    )


def decide_on_snapshot(eng: eg.FeatureEngine, snapshot: BookSnapshotInput) -> eg.EngineDecision:
    eng.observe_book_snapshot(snapshot)
    return eng.force_decision(local_ts_us=snapshot.local_ts_us, ts_us=snapshot.ts_us, event_seq=snapshot.event_seq)


def fv_value(vec, name):
    return vec[feature_spec_by_name(name).index]


def feed_ready_engine():
    eng = eg.FeatureEngine()
    eng.on_trade(make_trade(2_000_000, 100.0, 1.0, BUY_SIDE_CODE))
    dec = decide_on_snapshot(eng, make_snapshot(2_000_000, mid=100.0))
    assert dec is not None
    return eng, dec


def apply_two_books_with_known_l1_changes(eng):
    eng.on_book_snapshot(make_snapshot(2_000_000, mid=100.0, bid_sz0=10.0, ask_sz0=12.0))
    snap2 = make_snapshot(2_100_000, mid=100.0, bid_sz0=13.0, ask_sz0=9.0)
    dec = decide_on_snapshot(eng, snap2)
    return dec


def test_public_api_boundary():
    assert set(eg.__all__) == {
        "L2_EVENT_CODE", "TRADE_EVENT_CODE", "ENGINE_EVENT_WINDOWS_US", "DEFAULT_EVENT_HISTORY_CAPACITY",
        "CROSS_FEATURE_INDICES", "CROSS_FEATURE_NAMES", "EVENT_CONTEXT_FEATURE_INDICES", "EVENT_CONTEXT_FEATURE_NAMES",
        "ENGINE_FEATURE_INDICES", "ENGINE_FEATURE_NAMES", "FeatureEngineConfig", "EngineDecision", "EventHistory", "FeatureEngine",
        "cross_feature_names", "cross_feature_indices", "event_context_feature_names", "event_context_feature_indices",
        "engine_owned_feature_names", "engine_owned_feature_indices",
    }
    for name in eg.__all__:
        assert not name.startswith("_")
        if name.islower():
            low = name.lower()
            for tok in ("target", "future", "lookahead", "peek", "bybit", "cmssl", "aux", "label", "transform"):
                assert tok not in low


def test_engine_event_windows_are_active_only():
    assert eg.ENGINE_EVENT_WINDOWS_US == (
        eg.WINDOW_200MS_US,
        eg.WINDOW_500MS_US,
        eg.WINDOW_1000MS_US,
        eg.WINDOW_3000MS_US,
    )


def test_no_inactive_engine_helpers_remain():
    import inspect

    src = inspect.getsource(eg)
    forbidden = [
        "WINDOW_" + "100MS_US",
        "_trade_total_" + "notional",
        "_trade_" + "vwap",
        "_book_" + "mean",
        "_current_depth_" + "size",
        "_current_depth_" + "notional",
        "log_events_" + "100000us",
        "vwap_vs_" + "mid",
        "trade_impact_" + "half_life",
    ]
    for token in forbidden:
        assert token not in src


def test_no_forbidden_imports():
    code = r'''
import sys
before = set(sys.modules)
import mmrt.features.engine  # noqa: F401
after = set(sys.modules) - before
forbidden = (
    "po" + "lars",
    "pan" + "das",
    "to" + "rch",
    "py" + "arrow",
    "mmrt.features.la" + "bels",
    "mmrt.features.trans" + "forms",
    "CM" + "SSL17",
    "offline_" + "ingest",
)
bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit(repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_validation():
    assert eg.FeatureEngineConfig().schedule == DecisionScheduleConfig()
    with pytest.raises(ValueError):
        eg.FeatureEngineConfig(schedule=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        eg.FeatureEngineConfig(event_history_capacity=0)
    with pytest.raises(ValueError):
        eg.FeatureEngineConfig(schedule=True)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        eg.FeatureEngineConfig(event_history_capacity=True)


def test_engine_decision_validation_and_copy():
    vec = np.ones(FEATURE_COUNT, dtype=np.float64)
    d = eg.EngineDecision(0, 1, 1, -1, 100.0, vec, "ok")
    vec[0] = 99.0
    assert d.feature_vector[0] == 1.0

    d_int = eg.EngineDecision(0, 1, 1, -1, 100, np.ones(FEATURE_COUNT), "ok")
    d_float = eg.EngineDecision(0, 1, 1, -1, 100.5, np.ones(FEATURE_COUNT), "ok")
    d_np = eg.EngineDecision(0, 1, 1, -1, np.float64(101.5), np.ones(FEATURE_COUNT), "ok")
    assert isinstance(d_int.raw_mid, float)
    assert isinstance(d_float.raw_mid, float)
    assert isinstance(d_np.raw_mid, float)

    with pytest.raises(ValueError, match="raw_mid"):
        eg.EngineDecision(0, 1, 1, -1, 0.0, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError, match="raw_mid"):
        eg.EngineDecision(0, 1, 1, -1, -1.0, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError, match="raw_mid"):
        eg.EngineDecision(0, 1, 1, -1, np.nan, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError, match="raw_mid"):
        eg.EngineDecision(0, 1, 1, -1, np.inf, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError, match="raw_mid"):
        eg.EngineDecision(0, 1, 1, -1, True, np.ones(FEATURE_COUNT), "ok")

    with pytest.raises(ValueError):
        eg.EngineDecision(0, 0, 1, -1, 100.0, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError):
        eg.EngineDecision(0, 1, 0, -1, 100.0, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError):
        eg.EngineDecision(0, 1, 1, -1, 100.0, np.ones(FEATURE_COUNT - 1), "ok")
    with pytest.raises(ValueError):
        eg.EngineDecision(0, 1, 1, -1, 100.0, np.ones(FEATURE_COUNT), "")
    eg.EngineDecision(0, 1, 1, 0, 100.0, np.ones(FEATURE_COUNT), "ok")
    with pytest.raises(ValueError):
        eg.EngineDecision(0, 1, 1, -2, 100.0, np.ones(FEATURE_COUNT), "ok")


def test_event_history_counts_inclusive():
    h = eg.EventHistory()
    h.append(2_000_000, 1)
    h.append(2_100_000, 2)
    h.append(2_200_000, 1)
    assert h.count_in_window(2_200_000, 200_000) == 3
    assert h.count_in_window(2_200_000, 100_000) == 2
    h.append(2_200_000, 2)
    assert h.count_in_window(2_200_000, 1) == 2


def test_event_history_capacity():
    h = eg.EventHistory(3)
    for i, kind in [(1, 1), (2, 2), (3, 1), (4, 2), (5, 1)]:
        h.append(i, kind)
    assert np.array_equal(h.ordered_ts(), np.array([3, 4, 5]))
    assert np.array_equal(h.ordered_kinds(), np.array([1, 2, 1]))


def test_no_decision_before_ready():
    e = eg.FeatureEngine()
    assert e.on_book_snapshot(make_snapshot(2_000_000)) is None
    assert e.on_trade(make_trade(2_000_000)) is None
    assert decide_on_snapshot(e, make_snapshot(2_000_000)) is not None


def test_engine_first_decision_at_early_timestamp_does_not_crash_trade_asof():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(1_000_000, price=100.0, amount=2.0, side_code=BUY_SIDE_CODE))
    d = decide_on_snapshot(e, make_snapshot(1_000_000, mid=100.0))

    assert d is not None
    assert d.local_ts_us == 1_000_000
    assert np.all(np.isfinite(d.feature_vector))
    assert fv_value(d.feature_vector, "max_signed_trade_notional_usd_1000000us") == pytest.approx(200.0)


def test_trade_events_never_emit_decisions():
    e = eg.FeatureEngine()
    assert e.on_trade(make_trade(2_000_000)) is None
    assert e.on_trade(make_trade(2_100_000)) is None
    e.on_book_snapshot(make_snapshot(2_100_000))
    assert e.on_trade(make_trade(2_200_000, 100.5, 3.0, SELL_SIDE_CODE)) is None


def test_decision_cadence_book_only():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000))
    assert decide_on_snapshot(e, make_snapshot(2_000_000)) is not None
    assert e.on_book_snapshot(make_snapshot(2_100_000)) is None
    assert e.on_book_snapshot(make_snapshot(2_499_999)) is None
    assert decide_on_snapshot(e, make_snapshot(2_500_000)) is not None
    assert e.on_book_snapshot(make_snapshot(2_500_000)) is None
    assert decide_on_snapshot(e, make_snapshot(3_000_000)) is not None


def test_long_gap_emits_at_most_one_decision():
    e, _ = feed_ready_engine()
    c = e.decision_count
    assert decide_on_snapshot(e, make_snapshot(4_000_000)) is not None
    assert e.decision_count == c + 1
    assert e.last_decision_local_ts_us == 4_000_000



def test_emitted_decision_carries_decision_time_raw_mid():
    e = eg.FeatureEngine(config=eg.FeatureEngineConfig(schedule=DecisionScheduleConfig(min_decision_interval_us=100_000, max_decision_interval_us=100_000)))
    e.on_trade(make_trade(2_000_000, 100.0, 1.0, BUY_SIDE_CODE))

    snap1 = make_snapshot(2_000_000, mid=100.0)
    d1 = decide_on_snapshot(e, snap1)
    assert d1 is not None
    assert d1.raw_mid == pytest.approx(100.0)
    assert d1.local_ts_us == snap1.local_ts_us
    assert d1.ts_us == snap1.ts_us
    assert d1.event_seq == snap1.event_seq

    snap2 = make_snapshot(2_100_000, mid=101.0)
    d2 = decide_on_snapshot(e, snap2)
    assert d2 is not None
    assert d2.raw_mid == pytest.approx(101.0)
    assert d2.local_ts_us == snap2.local_ts_us
    assert d2.ts_us == snap2.ts_us
    assert d2.event_seq == snap2.event_seq


def test_runner_handoff_payload_uses_decision_raw_mid_without_book_lookup():
    e = eg.FeatureEngine(config=eg.FeatureEngineConfig(schedule=DecisionScheduleConfig(min_decision_interval_us=100_000, max_decision_interval_us=100_000)))
    e.on_trade(make_trade(2_000_000, 100.0, 1.0, BUY_SIDE_CODE))
    d = decide_on_snapshot(e, make_snapshot(2_000_000, mid=100.0))
    assert d is not None

    row_payload = {
        "decision_index": d.decision_index,
        "ts_us": d.ts_us,
        "local_ts_us": d.local_ts_us,
        "event_seq": d.event_seq,
        "raw_mid": d.raw_mid,
        "feature_values": tuple(d.feature_vector),
    }
    assert row_payload["raw_mid"] > 0.0

def test_feature_vector_shape_and_all_finite():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000, 100.0, 1.5, BUY_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_000_000, mid=100.0))
    e.on_trade(make_trade(2_100_000, 101.0, 2.0, SELL_SIDE_CODE))
    d = decide_on_snapshot(e, make_snapshot(2_500_000, mid=100.2, bid_sz0=11.0, ask_sz0=9.0))
    assert d is not None
    assert d.feature_vector.shape == (FEATURE_COUNT,)
    assert np.all(np.isfinite(d.feature_vector))


def test_non_engine_features_filled_by_states_and_engine_features_filled_by_engine():
    e, d = feed_ready_engine()
    assert np.all(np.isfinite(d.feature_vector))
    assert math.isfinite(fv_value(d.feature_vector, "obi_l1"))
    assert math.isfinite(fv_value(d.feature_vector, "trade_count_per_second_200000us"))
    assert math.isfinite(fv_value(d.feature_vector, "absorption_bid_1000000us"))
    assert math.isfinite(fv_value(d.feature_vector, "log_events_200000us"))


def test_engine_owned_indices_match_specs():
    expected_cross = tuple(i for i, spec in enumerate(FEATURE_SPECS) if spec.source == FeatureSource.CROSS)
    expected_event = tuple(i for i, spec in enumerate(FEATURE_SPECS) if spec.source == FeatureSource.EVENT_CONTEXT)
    assert eg.cross_feature_indices() == expected_cross
    assert eg.event_context_feature_indices() == expected_event
    assert eg.engine_owned_feature_indices() == expected_cross + expected_event
    assert eg.cross_feature_names() == tuple(spec.name for spec in FEATURE_SPECS if spec.source == FeatureSource.CROSS)
    assert eg.event_context_feature_names() == tuple(spec.name for spec in FEATURE_SPECS if spec.source == FeatureSource.EVENT_CONTEXT)
    assert eg.engine_owned_feature_names() == eg.cross_feature_names() + eg.event_context_feature_names()
    for n in ("obi_l1", "trade_count_per_second_200000us", "max_signed_trade_notional_usd_1000000us"):
        assert n not in set(eg.engine_owned_feature_names())


def test_absorption_formula_manual():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000, 100.0, 2.0, BUY_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_000_000, bid_sz0=10.0, ask_sz0=10.0))
    e.on_trade(make_trade(2_050_000, 100.0, 1.0, SELL_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_100_000, bid_sz0=13.0, ask_sz0=8.0))
    vec = e.build_feature_vector()
    eps = eg.FLOAT_EPS
    bid_add = e._book_sum("bid_l1_add", 1_000_000, 2_100_000)
    bid_rem = e._book_sum("bid_l1_rem", 1_000_000, 2_100_000)
    ask_add = e._book_sum("ask_l1_add", 1_000_000, 2_100_000)
    ask_rem = e._book_sum("ask_l1_rem", 1_000_000, 2_100_000)
    buy_n = e._trade_buy_notional(1_000_000, 2_100_000)
    sell_n = e._trade_sell_notional(1_000_000, 2_100_000)
    total = buy_n + sell_n
    expected_bid = (sell_n / total) * (bid_add / max(bid_add + bid_rem, eps))
    expected_ask = (buy_n / total) * (ask_add / max(ask_add + ask_rem, eps))
    assert fv_value(vec, "absorption_bid_1000000us") == pytest.approx(expected_bid)
    assert fv_value(vec, "absorption_ask_1000000us") == pytest.approx(expected_ask)


def test_ofi_pressure_formulas_manual():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000, 100.0, 1.0, BUY_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_000_000, mid=100.0, bid_sz0=10.0, ask_sz0=12.0))
    e.on_book_snapshot(make_snapshot(2_050_000, mid=100.1, bid_sz0=12.0, ask_sz0=11.0))
    e.on_book_snapshot(make_snapshot(2_100_000, mid=100.2, bid_sz0=11.0, ask_sz0=9.0))
    vec = e.build_feature_vector()
    now = e.book_state.last_local_ts_us
    ofi_pressure = e._book_sum("ofi_l1", 1_000_000, now)
    depth = e.book_state.current_summary().total_depth_5bps_size
    expected_depth = ofi_pressure / depth
    rv = e._book_realized_vol_bps("microprice", 1_000_000, now)
    expected_vol = 0.0 if rv <= eg.FLOAT_EPS else expected_depth / rv
    assert fv_value(vec, "ofi_l1_pressure_over_realized_vol_1000000us") == pytest.approx(expected_vol)


def test_absorption_replenishment_manual():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000, 100.0, 2.0, BUY_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_000_000, bid_sz0=10.0, ask_sz0=10.0))
    e.on_trade(make_trade(2_050_000, 100.0, 1.0, SELL_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_100_000, bid_sz0=13.0, ask_sz0=8.0))
    vec = e.build_feature_vector()
    eps = eg.FLOAT_EPS
    bid_add = e._book_sum("bid_l1_add", 1_000_000, 2_100_000)
    bid_rem = e._book_sum("bid_l1_rem", 1_000_000, 2_100_000)
    ask_add = e._book_sum("ask_l1_add", 1_000_000, 2_100_000)
    ask_rem = e._book_sum("ask_l1_rem", 1_000_000, 2_100_000)
    buy_n = e._trade_buy_notional(1_000_000, 2_100_000)
    sell_n = e._trade_sell_notional(1_000_000, 2_100_000)
    total = buy_n + sell_n
    buy_share = buy_n / total
    sell_share = sell_n / total
    ask_rr = ask_add / max(ask_add + ask_rem, eps)
    bid_rr = bid_add / max(bid_add + bid_rem, eps)
    assert fv_value(vec, "absorption_ask_1000000us") == pytest.approx(buy_share * ask_rr)
    assert fv_value(vec, "absorption_bid_1000000us") == pytest.approx(sell_share * bid_rr)


def test_trade_side_quote_response_asymmetry_manual():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000, 100.0, 2.0, BUY_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_000_000, bid_sz0=10.0, ask_sz0=10.0))
    e.on_trade(make_trade(2_050_000, 100.0, 1.0, SELL_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_100_000, bid_sz0=13.0, ask_sz0=8.0))
    vec = e.build_feature_vector()
    eps = eg.FLOAT_EPS
    bid_add = e._book_sum("bid_l1_add", 500_000, 2_100_000)
    bid_rem = e._book_sum("bid_l1_rem", 500_000, 2_100_000)
    ask_add = e._book_sum("ask_l1_add", 500_000, 2_100_000)
    ask_rem = e._book_sum("ask_l1_rem", 500_000, 2_100_000)
    buy_n = e._trade_buy_notional(500_000, 2_100_000)
    sell_n = e._trade_sell_notional(500_000, 2_100_000)
    total = buy_n + sell_n
    buy_share = buy_n / total
    sell_share = sell_n / total
    ask_rr = ask_add / max(ask_add + ask_rem, eps)
    bid_rr = bid_add / max(bid_add + bid_rem, eps)
    expected = buy_share * ask_rr - sell_share * bid_rr
    assert fv_value(vec, "trade_side_quote_response_asymmetry_500000us") == pytest.approx(expected)


def test_event_context_formulas_manual():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000))
    d1 = decide_on_snapshot(e, make_snapshot(2_000_000))
    e.on_trade(make_trade(2_300_000, side_code=UNKNOWN_SIDE_CODE))
    d2 = decide_on_snapshot(e, make_snapshot(2_500_000))
    assert d1 is not None and d2 is not None
    now = 2_500_000
    for w, name in [
        (200_000, "log_events_200000us"),
        (500_000, "log_events_500000us"),
        (1_000_000, "log_events_1000000us"),
        (3_000_000, "log_events_3000000us"),
    ]:
        c = e.event_history.count_in_window(now, w)
        assert fv_value(d2.feature_vector, name) == pytest.approx(math.log1p(c))


def test_same_timestamp_ordering_is_caller_order():
    t = 2_000_000
    e = eg.FeatureEngine()
    e.on_trade(make_trade(t, 110.0, 3.0, BUY_SIDE_CODE))
    d = decide_on_snapshot(e, make_snapshot(t, mid=100.0))
    assert d is not None
    assert abs(fv_value(d.feature_vector, "max_signed_trade_notional_usd_1000000us")) > 0.0

    e2 = eg.FeatureEngine()
    e2.on_trade(make_trade(t - 500_000, 100.0, 1.0, BUY_SIDE_CODE))
    decide_on_snapshot(e2, make_snapshot(t - 500_000, mid=100.0))
    d2 = decide_on_snapshot(e2, make_snapshot(t, mid=100.0))
    assert d2 is not None
    pre_max = fv_value(d2.feature_vector, "max_signed_trade_notional_usd_1000000us")
    e2.on_trade(make_trade(t, 140.0, 10.0, BUY_SIDE_CODE))
    assert fv_value(d2.feature_vector, "max_signed_trade_notional_usd_1000000us") == pre_max


def test_out_of_order_events_rejected():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000))
    with pytest.raises(ValueError):
        e.on_trade(make_trade(1_999_999))
    with pytest.raises(ValueError):
        e.on_book_snapshot(make_snapshot(1_999_999))
    assert e.on_trade(make_trade(2_000_000, 100.0, 1.0, SELL_SIDE_CODE)) is None


def test_build_feature_vector_rejects_not_ready_or_bad_as_of():
    e = eg.FeatureEngine()
    with pytest.raises(ValueError):
        e.build_feature_vector()
    e.on_trade(make_trade(2_000_000))
    e.on_book_snapshot(make_snapshot(2_000_000))
    e.on_trade(make_trade(2_100_000))
    with pytest.raises(ValueError):
        e.build_feature_vector(as_of_local_ts_us=2_050_000)
    with pytest.raises(ValueError):
        e.build_feature_vector(as_of_local_ts_us=2_100_000)
    e.on_book_snapshot(make_snapshot(2_500_000))
    _ = e.build_feature_vector()


def test_no_label_transform_or_storage_residue():
    for name in eg.__all__:
        low = name.lower()
        for tok in ("label", "transform", "target", "future", "lookahead", "storage"):
            assert tok not in low


def test_no_future_leakage_with_as_of():
    e, _ = feed_ready_engine()
    current = e.book_state.last_local_ts_us
    size_before = e.event_history.size
    decisions_before = e.decision_count
    with pytest.raises(ValueError):
        e.build_feature_vector(as_of_local_ts_us=current + 1)
    assert e.event_history.size == size_before
    assert e.decision_count == decisions_before


def test_all_engine_features_assigned_no_placeholders():
    e = eg.FeatureEngine()
    e.on_trade(make_trade(2_000_000, 101.0, 3.0, BUY_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_000_000, mid=100.0, bid_sz0=10.0, ask_sz0=10.0))
    e.on_trade(make_trade(2_150_000, 99.0, 2.0, SELL_SIDE_CODE))
    e.on_book_snapshot(make_snapshot(2_300_000, mid=100.2, bid_sz0=12.0, ask_sz0=8.0))
    e.on_trade(make_trade(2_450_000, 100.8, 1.0, BUY_SIDE_CODE))
    d = decide_on_snapshot(e, make_snapshot(2_500_000, mid=100.1, bid_sz0=11.0, ask_sz0=9.0))
    assert d is not None
    vec = d.feature_vector
    for idx in eg.CROSS_FEATURE_INDICES + eg.EVENT_CONTEXT_FEATURE_INDICES:
        assert np.isfinite(vec[idx])
    assert (
        abs(fv_value(vec, "absorption_bid_1000000us")) > 0.0
        or abs(fv_value(vec, "absorption_ask_1000000us")) > 0.0
    )
    assert np.isfinite(fv_value(vec, "ofi_l1_pressure_over_realized_vol_1000000us"))
    assert abs(fv_value(vec, "trade_side_quote_response_asymmetry_500000us")) > 0.0
    assert fv_value(vec, "log_events_500000us") > 0.0
