import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import book_state as bs
from mmrt.features import kernels as k
from mmrt.features.specs import FEATURE_COUNT, FEATURE_SPECS, FeatureSource, feature_spec_by_name


def make_snapshot(local_ts_us=1_000_000, mid=100.0, spread=0.10, bid_sz0=10.0, ask_sz0=12.0, bid_size_offset=0.0, ask_size_offset=0.0):
    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0
    bid_px = best_bid - 0.1 * np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    ask_px = best_ask + 0.1 * np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    bid_sz = bid_sz0 + bid_size_offset + np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    ask_sz = ask_sz0 + ask_size_offset + np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    return bs.BookSnapshotInput(local_ts_us=local_ts_us, ts_us=local_ts_us, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)


def fv_value(vec, name):
    return vec[feature_spec_by_name(name).index]


def apply_dynamic_sequence(st: bs.BookState) -> None:
    mids = [100.0, 100.20, 100.10, 100.40, 100.15, 100.60, 100.45, 100.80, 100.55, 100.95, 100.70, 101.10, 100.90, 101.25, 101.05]
    spreads = [0.10, 0.12, 0.09, 0.15, 0.11, 0.14, 0.08, 0.16, 0.10, 0.13, 0.09, 0.15, 0.12, 0.14, 0.10]
    for i, (mid, spread) in enumerate(zip(mids, spreads)):
        st.apply_snapshot(make_snapshot(local_ts_us=1_000_000 + i * 250_000, mid=mid, spread=spread, bid_sz0=10.0 + ((i * 3) % 7), ask_sz0=12.0 + ((i * 5) % 9), bid_size_offset=float(i % 4), ask_size_offset=float((i + 1) % 4)))


def test_public_api_boundary():
    expected = {"BOOK_DEPTH", "MAX_EMITTED_DEPTH", "BID_SIDE_CODE", "ASK_SIDE_CODE", "BOOK_WINDOWS_US", "DEFAULT_HISTORY_CAPACITY", "BOOK_FEATURE_INDICES", "BOOK_FEATURE_NAMES", "BookSnapshotInput", "BookSummary", "BookHistory", "BookState", "book_owned_feature_names", "book_owned_feature_indices"}
    assert set(bs.__all__) == expected
    for name in bs.__all__:
        low = name.lower()
        assert not name.startswith("_")
        for bad in ("target", "future", "lookahead", "peek", "bybit", "cmssl", "aux"):
            assert bad not in low


def test_no_forbidden_imports():
    code = r'''
import sys
before = set(sys.modules)
import mmrt.features.book_state  # noqa: F401
after = set(sys.modules) - before
forbidden = (
    "po" + "lars",
    "pan" + "das",
    "to" + "rch",
    "py" + "arrow",
    "mmrt.data.tardis_csv",
    "mmrt.data.event_merge",
    "mmrt.data.quality",
    "CM" + "SSL17",
    "offline_" + "ingest",
)
bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit(repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_snapshot_input_validation_full():
    make_snapshot()
    with pytest.raises(ValueError):
        s = make_snapshot(); bs.BookSnapshotInput(s.local_ts_us, s.ts_us, np.ones(bs.BOOK_DEPTH - 1), s.bid_sz, s.ask_px, s.ask_sz)
    with pytest.raises(ValueError):
        bs.BookSnapshotInput(1, 1, np.ones(bs.BOOK_DEPTH), -np.ones(bs.BOOK_DEPTH), np.ones(bs.BOOK_DEPTH), np.ones(bs.BOOK_DEPTH))
    with pytest.raises(ValueError):
        a = np.ones(bs.BOOK_DEPTH); a[0] = np.nan; bs.BookSnapshotInput(1, 1, a, np.ones(bs.BOOK_DEPTH), np.ones(bs.BOOK_DEPTH), np.ones(bs.BOOK_DEPTH))
    with pytest.raises(ValueError):
        s = make_snapshot(); s.bid_px[0] = 0.0; bs.BookSnapshotInput(s.local_ts_us, s.ts_us, s.bid_px, s.bid_sz, s.ask_px, s.ask_sz)
    with pytest.raises(ValueError):
        s = make_snapshot(); s.ask_px[0] = 0.0; bs.BookSnapshotInput(s.local_ts_us, s.ts_us, s.bid_px, s.bid_sz, s.ask_px, s.ask_sz)
    with pytest.raises(ValueError):
        s = make_snapshot(); bp = s.bid_px.copy(); bp[1] = bp[0] + 0.01; bs.BookSnapshotInput(s.local_ts_us, s.ts_us, bp, s.bid_sz, s.ask_px, s.ask_sz)
    with pytest.raises(ValueError):
        s = make_snapshot(); ap = s.ask_px.copy(); ap[1] = ap[0] - 0.01; bs.BookSnapshotInput(s.local_ts_us, s.ts_us, s.bid_px, s.bid_sz, ap, s.ask_sz)
    with pytest.raises(ValueError):
        s = make_snapshot(); bp = s.bid_px.copy(); bp[5] = 0.0; bp[6] = 1.0; bs.BookSnapshotInput(s.local_ts_us, s.ts_us, bp, s.bid_sz, s.ask_px, s.ask_sz)
    st = bs.BookState(); c = make_snapshot(mid=100.0, spread=-0.01); st.apply_snapshot(c); assert st.current_summary().is_crossed


def test_apply_snapshot_summary():
    st = bs.BookState()
    snap = make_snapshot()
    summ = st.apply_snapshot(snap)
    best_bid = snap.bid_px[0]
    best_ask = snap.ask_px[0]
    mid = (best_bid + best_ask) / 2.0
    assert summ.best_bid == pytest.approx(best_bid)
    assert summ.best_ask == pytest.approx(best_ask)
    assert summ.mid == pytest.approx(mid)
    assert summ.spread_bps == pytest.approx((best_ask - best_bid) / mid * 10_000.0)
    assert summ.microprice == pytest.approx(k.microprice(best_bid, best_ask, snap.bid_sz[0], snap.ask_sz[0]))
    assert summ.bid_depth_5bps_size > 0 and summ.ask_depth_5bps_size > 0
    assert summ.bid_depth_5bps_notional > 0 and summ.ask_depth_5bps_notional > 0
    assert summ.total_depth_5bps_size == pytest.approx(summ.bid_depth_5bps_size + summ.ask_depth_5bps_size)
    assert summ.total_depth_5bps_notional == pytest.approx(summ.bid_depth_5bps_notional + summ.ask_depth_5bps_notional)
    assert summ.update_count == 1
    assert st.has_book()


def test_monotonic_local_time():
    st = bs.BookState(); st.apply_snapshot(make_snapshot(local_ts_us=1_000_000)); st.apply_snapshot(make_snapshot(local_ts_us=1_000_000))
    with pytest.raises(ValueError):
        st.apply_snapshot(make_snapshot(local_ts_us=999_999))


def test_reset_and_capacity():
    st = bs.BookState(history_capacity=3)
    for i in range(5):
        st.apply_snapshot(make_snapshot(local_ts_us=1_000_000 + i * 100_000))
    assert st.history.size == 3
    assert st.history.ordered_ts().tolist() == [1_200_000, 1_300_000, 1_400_000]
    st.reset()
    assert not st.has_book()
    with pytest.raises(ValueError):
        st.fill_book_features(np.zeros(FEATURE_COUNT))
    with pytest.raises(ValueError):
        st.current_summary()


def test_book_owned_indices_match_specs_after_reclassification():
    expected = tuple(i for i, spec in enumerate(FEATURE_SPECS) if spec.source == FeatureSource.BOOK)
    assert bs.book_owned_feature_indices() == expected
    assert bs.book_owned_feature_names() == tuple(FEATURE_SPECS[i].name for i in expected)
    for n in ("time_since_trade_us", "vwap_vs_mid_bps_200000us", "vwap_vs_mid_bps_500000us", "regime_volume_ewma_500000us", "regime_volume_ewma_3000000us"):
        assert n not in bs.book_owned_feature_names()


def test_feature_vector_shape_and_non_book_indices_untouched():
    st = bs.BookState(); apply_dynamic_sequence(st)
    out = np.full(FEATURE_COUNT, -123.0); st.fill_book_features(out)
    b = set(bs.book_owned_feature_indices())
    assert np.all(np.isfinite(out[list(b)]))
    for i in range(FEATURE_COUNT):
        if i not in b:
            assert out[i] == -123.0


def test_sparse_window_micro_ret_uses_right_asof():
    st = bs.BookState(); st.apply_snapshot(make_snapshot(local_ts_us=1_000_000, mid=100.0)); st.apply_snapshot(make_snapshot(local_ts_us=1_250_000, mid=101.0)); st.apply_snapshot(make_snapshot(local_ts_us=1_500_000, mid=102.0))
    out = np.full(FEATURE_COUNT, -1.0); st.fill_book_features(out)
    exp = k.bps_change(st.history.asof_value("microprice", 1_500_000), st.history.asof_value("microprice", 1_300_000))
    got = fv_value(out, "micro_ret_bps_200000us")
    assert got == pytest.approx(exp)
    assert got != 0.0


def test_all_book_features_assigned_and_dynamic_nonzero():
    st = bs.BookState(); apply_dynamic_sequence(st); out = np.full(FEATURE_COUNT, -1.0); st.fill_book_features(out)
    for i in bs.book_owned_feature_indices():
        assert np.isfinite(out[i])
    assert fv_value(out, "micro_ret_bps_200000us") != 0
    assert fv_value(out, "mid_slope_bps_per_sec_500000us") != 0
    assert fv_value(out, "mid_range_bps_500000us") > 0
    assert fv_value(out, "ofi_l1") != 0
    assert fv_value(out, "ofi_l5_sum_over_depth_200000us") != 0
    assert fv_value(out, "ob_update_rate_500000us") > 0
    assert fv_value(out, "max_abs_return_bps_500000us") > 0
    assert fv_value(out, "microprice_realized_vol_1000000us") > 0
    assert fv_value(out, "down_up_vol_imbalance_500000us") != 0


def test_no_invalid_placeholders_remaining_runtime():
    st = bs.BookState(); apply_dynamic_sequence(st); out = np.full(FEATURE_COUNT, -321.0); st.fill_book_features(out)
    for n in ("time_since_trade_us", "vwap_vs_mid_bps_200000us", "vwap_vs_mid_bps_500000us", "regime_volume_ewma_500000us", "regime_volume_ewma_3000000us"):
        assert fv_value(out, n) == -321.0
    vals = [fv_value(out, "down_up_vol_imbalance_500000us"), fv_value(out, "down_up_vol_imbalance_1000000us"), fv_value(out, "down_up_vol_imbalance_3000000us")]
    assert all(np.isfinite(v) for v in vals)
    assert any(v != 0 for v in vals)


def test_basic_instantaneous_features():
    st = bs.BookState(); snap = make_snapshot(); out = np.zeros(FEATURE_COUNT); st.apply_snapshot(snap); st.fill_book_features(out)
    mid = (snap.bid_px[0] + snap.ask_px[0]) / 2.0
    assert fv_value(out, "spread_bps") == pytest.approx((snap.ask_px[0] - snap.bid_px[0]) / mid * 10_000)
    assert fv_value(out, "bsz1") == pytest.approx(snap.bid_sz[0])
    assert fv_value(out, "asz1") == pytest.approx(snap.ask_sz[0])
    assert fv_value(out, "bid_l1_notional_usd") == pytest.approx(snap.bid_px[0] * snap.bid_sz[0])
    assert fv_value(out, "ask_l1_notional_usd") == pytest.approx(snap.ask_px[0] * snap.ask_sz[0])
    obi = (snap.bid_sz[0] - snap.ask_sz[0]) / (snap.bid_sz[0] + snap.ask_sz[0])
    assert fv_value(out, "obi_l1") == pytest.approx(obi)
    micro = k.microprice(snap.bid_px[0], snap.ask_px[0], snap.bid_sz[0], snap.ask_sz[0])
    assert fv_value(out, "micro_minus_mid_bps") == pytest.approx(k.bps_change(micro, mid))
    assert fv_value(out, "gap_b_bps") == pytest.approx(k.bps_change(snap.bid_px[0], snap.bid_px[1]))


def test_depth_features():
    st = bs.BookState(); st.apply_snapshot(make_snapshot()); out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert fv_value(out, "bid_depth_within_1bps") > 0
    assert fv_value(out, "ask_depth_within_1bps") > 0
    assert fv_value(out, "bid_depth_notional_5bps") + fv_value(out, "ask_depth_notional_5bps") == pytest.approx(fv_value(out, "total_depth_notional_5bps"))
    assert np.isfinite(fv_value(out, "depth_imbalance_within_1bps"))
    assert np.isfinite(fv_value(out, "micro_l5_minus_mid_bps"))
    assert np.isfinite(fv_value(out, "vamp_l10_minus_mid_bps"))
    assert fv_value(out, "bid_liquidity_void_bps") >= 0
    assert fv_value(out, "ask_liquidity_void_bps") >= 0
    assert fv_value(out, "bid_depth_centroid_bps_25bps") >= 0
    assert fv_value(out, "ask_depth_centroid_bps_25bps") >= 0


def test_liquidity_void_distance_from_mid():
    st = bs.BookState(); s = make_snapshot(mid=100.0, spread=0.10)
    s.bid_sz[:20] = np.array([1, 1, 1, 1, 1, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50], dtype=np.float64)
    st.apply_snapshot(s); out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    expected = (st._mid() - s.bid_px[5]) / st._mid() * 10_000.0
    assert fv_value(out, "bid_liquidity_void_bps") == pytest.approx(expected)


def test_ofi_first_snapshot_zero():
    st = bs.BookState(); st.apply_snapshot(make_snapshot()); out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert fv_value(out, "ofi_l1") == 0
    assert fv_value(out, "ofi_l3") == 0
    assert fv_value(out, "ofi_l5") == 0
    assert fv_value(out, "ofi_l1_sum_over_depth_200000us") == 0
    assert fv_value(out, "bid_l1_depletion_200000us") == 0
    assert fv_value(out, "ask_l1_depletion_200000us") == 0


def test_ofi_second_snapshot_size_change():
    st = bs.BookState(); st.apply_snapshot(make_snapshot(bid_sz0=10, ask_sz0=12)); st.apply_snapshot(make_snapshot(local_ts_us=1_200_000, bid_sz0=15, ask_sz0=10))
    out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert fv_value(out, "ofi_l1") == pytest.approx(7)
    assert fv_value(out, "ofi_l1_over_depth_5bps") > 0
    assert fv_value(out, "ofi_l1_sum_over_depth_200000us") > 0


def test_l1_add_rem_are_side_specific():
    st = bs.BookState(); st.apply_snapshot(make_snapshot(bid_sz0=10, ask_sz0=12)); st.apply_snapshot(make_snapshot(local_ts_us=1_200_000, bid_sz0=15, ask_sz0=10))
    out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert fv_value(out, "bid_l1_depletion_200000us") == 0
    assert fv_value(out, "ask_l1_depletion_200000us") > 0
    assert fv_value(out, "bid_l1_add_rate_over_depth_200000us") > 0


def test_return_distribution_features():
    st = bs.BookState(); apply_dynamic_sequence(st); out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert np.isfinite(fv_value(out, "return_std_bps_200000us"))
    assert fv_value(out, "max_abs_return_bps_500000us") > 0
    for n in ("down_up_vol_imbalance_500000us", "down_up_vol_imbalance_1000000us", "down_up_vol_imbalance_3000000us"):
        assert -1 <= fv_value(out, n) <= 1
    returns = st._window_values("mid_return_bps", bs.WINDOW_500MS_US)
    assert fv_value(out, "max_abs_return_bps_500000us") == pytest.approx(float(np.max(np.abs(returns))))


def test_age_features():
    st = bs.BookState(); st.apply_snapshot(make_snapshot(local_ts_us=1_000_000)); st.apply_snapshot(make_snapshot(local_ts_us=1_200_000))
    out = np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert fv_value(out, "best_bid_size_age_us") > 0
    assert fv_value(out, "best_ask_size_age_us") > 0
    st.apply_snapshot(make_snapshot(local_ts_us=1_400_000, bid_sz0=11.0))
    st.fill_book_features(out)
    assert fv_value(out, "best_bid_size_age_us") == 0
    assert fv_value(out, "best_ask_size_age_us") > 0


def test_no_cross_trade_or_event_context_features_written():
    st = bs.BookState(); apply_dynamic_sequence(st)
    out = np.full(FEATURE_COUNT, -77.0); st.fill_book_features(out)
    names = ("time_since_trade_us", "vwap_vs_mid_bps_200000us", "vwap_vs_mid_bps_500000us", "regime_volume_ewma_500000us", "regime_volume_ewma_3000000us", "signed_notional_flow_usd_200000us", "trade_count_per_second_500000us", "cvd_change_usd_500000us", "absorption_bid_200000us", "ofi_l1_pressure_over_depth_5bps_200000us", "log_events_100000us")
    for n in names:
        assert fv_value(out, n) == -77.0


def test_apply_snapshot_rejects_non_snapshot_input():
    with pytest.raises(TypeError):
        bs.BookState().apply_snapshot(object())


def test_no_future_leakage_public_names():
    for n in bs.__all__:
        assert all(x not in n.lower() for x in ("future", "lookahead", "peek", "target"))
    for name in dir(bs):
        if name.startswith("_"):
            continue
        obj = getattr(bs, name)
        if callable(obj):
            assert all(x not in name.lower() for x in ("future", "lookahead", "peek", "target"))
