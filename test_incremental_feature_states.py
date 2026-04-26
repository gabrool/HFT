import math
from collections import deque

from CMSSL17 import (
    CVDWindowState,
    FeatureEngine,
    LARGE_TRADE_CLUSTER_GAP_MS,
    LARGE_TRADE_CLUSTER_THRESHOLD_USD,
    LARGE_TRADE_NOTIONAL_USD,
    LargeTradeWindowState,
    RollingReturnDistributionState,
    TradeBurstWindowState,
)


def _assert_close(a, b, tol=1e-9, msg=""):
    if not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{msg} expected={b} got={a}")


def _mk_engine_stub() -> FeatureEngine:
    return FeatureEngine.__new__(FeatureEngine)


def test_trade_burst_autocorr_and_distribution():
    fe = _mk_engine_stub()
    st = TradeBurstWindowState.create(30_000)
    for ts, sign in [(1000, 1), (2000, 1), (3000, 1)]:
        FeatureEngine._trade_burst_insert(fe, 30_000, ts, sign) if hasattr(fe, "trade_burst_states") else None
    # direct state construction using same insertion path semantics
    st = TradeBurstWindowState.create(30_000)
    fe.trade_burst_states = {30_000: st}
    for ts, sign in [(1000, 1), (2000, 1), (3000, 1)]:
        fe._trade_burst_insert(30_000, ts, sign)
    _assert_close(fe._corr_lag1_from_sign_state(st), 1.0, msg="constant buys autocorr")
    _assert_close(st.buy_count / (st.buy_count + st.sell_count), 1.0)
    _assert_close(0.0, 0.0)
    if st.runs[-1][2] != 3:
        raise AssertionError("expected max buy run length 3")

    st2 = TradeBurstWindowState.create(30_000)
    fe.trade_burst_states = {30_000: st2}
    for ts, sign in [(1000, -1), (2000, -1), (3000, -1)]:
        fe._trade_burst_insert(30_000, ts, sign)
    _assert_close(fe._corr_lag1_from_sign_state(st2), 1.0, msg="constant sells autocorr")

    st3 = TradeBurstWindowState.create(30_000)
    fe.trade_burst_states = {30_000: st3}
    for i, sign in enumerate([1, -1, 1, -1], start=1):
        fe._trade_burst_insert(30_000, i * 1000, sign)
    _assert_close(fe._corr_lag1_from_sign_state(st3), -1.0, tol=1e-9, msg="alternating autocorr")
    p_buy = st3.buy_count / (st3.buy_count + st3.sell_count)
    p_sell = st3.sell_count / (st3.buy_count + st3.sell_count)
    entropy = -(p_buy * math.log2(p_buy) + p_sell * math.log2(p_sell))
    _assert_close(entropy, 1.0, tol=1e-9, msg="alternating entropy")

    st4 = TradeBurstWindowState.create(30_000)
    fe.trade_burst_states = {30_000: st4}
    for i, sign in enumerate([1, 0, 1, 0, -1], start=1):
        fe._trade_burst_insert(30_000, i * 1000, sign)
    assert st4.buy_count + st4.sell_count == 3
    assert st4.buy_count == 2
    assert st4.sell_count == 1

    st5 = TradeBurstWindowState.create(3_000)
    fe.trade_burst_states = {3_000: st5}
    for ts, sign in [(1000, 1), (2000, 1), (3000, -1), (7000, -1)]:
        fe._trade_burst_insert(3_000, ts, sign)
    fe._trade_burst_prune(3_000, 7000)
    assert len(st5.signs) == 1 and st5.signs[0][0] == 7000
    _assert_close(fe._corr_lag1_from_sign_state(st5), 0.0)


def test_cvd_baseline_and_slope_stability():
    st = CVDWindowState.create(10_000)
    st.add(1000, 100)
    st.add(5000, 150)
    st.add(9000, 175)
    st.prune(20_000)
    _assert_close(st.change_usd(20_000, 175), 0.0)
    _assert_close(st.asof_before_window_value, 175.0)

    st2 = CVDWindowState.create(10_000)
    for ts, y in [(5000, 100), (10_000, 130), (12_000, 160)]:
        st2.add(ts, y)
    st2.prune(20_000)
    _assert_close(st2.change_usd(20_000, 160), 30.0)
    assert st2.points and st2.points[0][0] == 10_000
    _assert_close(st2.slope_usd_per_sec(), 0.0)

    st3 = CVDWindowState.create(10_000)
    for ts, y in [(10_000, 100), (10_000, 140), (12_000, 180)]:
        st3.add(ts, y)
    st3.prune(20_000)
    _assert_close(st3.change_usd(20_000, 180), 40.0)

    base = 1_770_000_000_000
    st4 = CVDWindowState.create(10_000)
    st4.add(base + 0, 0)
    st4.add(base + 1000, 100)
    st4.add(base + 2000, 200)
    st4.add(base + 3000, 300)
    st4.prune(base + 3000)
    slope = st4.slope_usd_per_sec()
    assert math.isfinite(slope)
    _assert_close(slope, 100.0, tol=1e-6)


def _reference_large_trade_stats(trades, ms, now_ms):
    cutoff = int(now_ms) - int(ms)
    in_window = [t for t in trades if int(t[0]) >= cutoff]
    out = {}
    for thr in LARGE_TRADE_NOTIONAL_USD:
        buys = [t for t in in_window if t[3] >= thr and t[5] > 0]
        sells = [t for t in in_window if t[3] >= thr and t[5] < 0]
        thr_key = f"{int(thr)}" if float(thr).is_integer() else str(thr).replace('.', 'p')
        bn = sum(t[3] for t in buys)
        sn = sum(t[3] for t in sells)
        out[f"large_buy_count_ge_{thr_key}_{ms}ms"] = float(len(buys))
        out[f"large_sell_count_ge_{thr_key}_{ms}ms"] = float(len(sells))
        out[f"large_buy_notional_ge_{thr_key}_{ms}ms"] = float(bn)
        out[f"large_sell_notional_ge_{thr_key}_{ms}ms"] = float(sn)
        out[f"large_trade_imbalance_ge_{thr_key}_{ms}ms"] = 0.0 if (bn + sn) <= 1e-12 else float((bn - sn) / (bn + sn))
    if in_window:
        largest = max(in_window, key=lambda x: x[3])
        out[f"max_signed_trade_notional_usd_{ms}ms"] = float(largest[5] * largest[3])
        out[f"top5_trade_notional_sum_usd_{ms}ms"] = float(sum(sorted((t[3] for t in in_window), reverse=True)[:5]))
    else:
        out[f"max_signed_trade_notional_usd_{ms}ms"] = 0.0
        out[f"top5_trade_notional_sum_usd_{ms}ms"] = 0.0
    large = sorted((t for t in in_window if t[3] >= LARGE_TRADE_CLUSTER_THRESHOLD_USD), key=lambda x: x[0])
    clusters = 0
    prev = None
    for t in large:
        if prev is None or (t[0] - prev) > LARGE_TRADE_CLUSTER_GAP_MS:
            clusters += 1
        prev = t[0]
    out[f"large_trade_cluster_count_{ms}ms"] = float(clusters)
    return out


def test_large_trade_and_rolling_distribution_equivalence():
    fe = _mk_engine_stub()
    ms = 30_000
    fe.large_trade_states = {ms: LargeTradeWindowState.create(ms)}
    trades = [
        (1000, 1.0, 1.0, 50_000.0, "buy", 1.0, 1.0, 0.0),
        (2000, 1.0, 1.0, 200_000.0, "sell", -1.0, -1.0, 0.0),
        (2500, 1.0, 1.0, 220_000.0, "buy", 1.0, 1.0, 0.0),
        (7000, 1.0, 1.0, 210_000.0, "buy", 1.0, 1.0, 0.0),
        (50_000, 1.0, 1.0, 190_000.0, "sell", -1.0, -1.0, 0.0),
    ]
    for entry in trades:
        fe._large_trade_state_insert(ms, entry)
    now_ms = 50_000
    for entry in trades:
        if entry[0] < now_ms - ms:
            fe._large_trade_state_expire(ms, entry)
    got = fe._large_trade_stats_from_state(ms, now_ms)
    ref = _reference_large_trade_stats(trades, ms, now_ms)
    for k in ref:
        _assert_close(got[k], ref[k], tol=1e-6, msg=f"large trade key={k}")

    state = RollingReturnDistributionState(
        window_ms=3000,
        deq=deque(),
        n=0,
        sum1=0.0,
        sum2=0.0,
        sum3=0.0,
        sum4=0.0,
        up_sumsq=0.0,
        down_sumsq=0.0,
        bipower=0.0,
        max_abs_q=deque(),
        seq=0,
    )
    fe.regime_return_states = {3000: state}
    for ts, r in [(1000, 1.0), (2000, -2.0), (3000, 3.0), (7000, -4.0)]:
        fe._regime_return_add(state, ts, r)
        fe._regime_return_prune(state, ts)
    vals = [r for ts, r in [(1000, 1.0), (2000, -2.0), (3000, 3.0), (7000, -4.0)] if ts >= 4000]
    got_dist = fe._regime_distribution(3000)
    ref_dist = fe._return_distribution_stats(vals)
    for key in ("realized_up_vol_bps", "realized_down_vol_bps", "down_up_vol_ratio", "bipower_variation", "jump_variation", "max_abs_return_bps", "return_skew", "return_kurtosis"):
        _assert_close(got_dist[key], ref_dist[key], tol=1e-6, msg=f"dist key={key}")


def run_all_tests():
    test_trade_burst_autocorr_and_distribution()
    test_cvd_baseline_and_slope_stability()
    test_large_trade_and_rolling_distribution_equivalence()
    print("incremental state tests: PASS")


if __name__ == "__main__":
    run_all_tests()
