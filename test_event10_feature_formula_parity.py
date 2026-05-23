import math
from collections import deque

import numpy as np

from test_feature_event_result_contract import _install_optional_dependency_stubs

_install_optional_dependency_stubs()
from CMSSL17 import FeatureEngine


EPS = 1e-9
PROMOTED = [
    "top5_trade_share_notional_3000ms",
    "depth_imbalance_realized_vol_1000ms",
    "microprice_zero_cross_rate_1000ms",
    "l1_churn_over_depth_1000ms",
    "same_side_trade_cluster_notional_1000ms",
    "ofi_pressure_x_churn_500ms",
    "bid_liquidity_void_bps",
    "ask_liquidity_void_bps",
    "post_buy_trade_ask_replenishment_200ms",
    "post_sell_trade_bid_replenishment_200ms",
]


def _ob(ts, tp, bids, asks):
    return ("ob", ts, 0, tp, tuple(bids), tuple(asks))


def _tr(ts, price, size, side_sign):
    return ("trade", ts, 10, price, size, side_sign, 0, 0)


def _prune(dq, now, window_ms):
    while dq and (now - dq[0][0]) > window_ms:
        dq.popleft()


def _sum_values(dq):
    return float(sum(v for _, v in dq))


def _depth_notional_5bps(levels, mid, side):
    if mid <= 0.0:
        return 0.0
    out = 0.0
    if side == "bid":
        cutoff = mid * (1.0 - 5e-4)
        for px, sz in levels:
            if px >= cutoff:
                out += float(px) * max(float(sz), 0.0)
    else:
        cutoff = mid * (1.0 + 5e-4)
        for px, sz in levels:
            if px <= cutoff:
                out += float(px) * max(float(sz), 0.0)
    return out


def _liquidity_void_bps(levels, mid, side, max_bps=10.0):
    if mid <= 0.0 or len(levels) < 2:
        return 0.0
    top = levels[0][0]
    for px, sz in levels[1:]:
        if float(sz) > EPS:
            gap = (top - px) if side == "bid" else (px - top)
            bps = 1e4 * gap / mid
            return float(min(max(bps, 0.0), max_bps))
    return float(max_bps)


def _zero_cross_rate(points):
    signs = []
    for _, v in points:
        if v > 0.0:
            signs.append(1)
        elif v < 0.0:
            signs.append(-1)
    if len(signs) < 2:
        return 0.0
    crosses = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    return float(crosses / max(len(signs) - 1, 1))


def _reference(events):
    trade_dq = deque()
    churn500, churn1000 = deque(), deque()
    ofi500 = deque()
    ask_add200, bid_add200 = deque(), deque()
    dim_hist = deque()
    micro_hist = deque()
    cluster = deque()

    prev_bid_n, prev_ask_n = None, None
    out = None

    for e in events:
        if e[0] == "trade":
            _, ts, _, px, sz, side_sign, *_ = e
            side = "Buy" if side_sign > 0 else "Sell"
            notional = float(px) * float(sz)
            trade_dq.append((ts, notional, side))
            cluster.append((ts, notional, side_sign))
            _prune(trade_dq, ts, 3000)
            _prune(cluster, ts, 1000)
            continue

        _, ts, _, _, bids, asks = e
        bid1, bsz1 = bids[0]
        ask1, asz1 = asks[0]
        mid = 0.5 * (bid1 + ask1)
        micro = (ask1 * bsz1 + bid1 * asz1) / max(bsz1 + asz1, EPS)
        micro_minus_mid_bps = 1e4 * (micro - mid) / max(mid, EPS)

        bid_n = float(bid1) * max(float(bsz1), 0.0)
        ask_n = float(ask1) * max(float(asz1), 0.0)
        if prev_bid_n is None or prev_ask_n is None:
            bd_notional, ad_notional = 0.0, 0.0
        else:
            bd_notional = bid_n - prev_bid_n
            ad_notional = ask_n - prev_ask_n

        prev_bid_n, prev_ask_n = bid_n, ask_n

        churn_evt = abs(bd_notional) + abs(ad_notional)
        churn500.append((ts, churn_evt))
        churn1000.append((ts, churn_evt))
        ofi500.append((ts, bd_notional - ad_notional))
        ask_add200.append((ts, max(ad_notional, 0.0)))
        bid_add200.append((ts, max(bd_notional, 0.0)))

        for dq, w in ((churn500, 500), (churn1000, 1000), (ofi500, 500), (ask_add200, 200), (bid_add200, 200), (dim_hist, 1000), (micro_hist, 1000), (cluster, 1000), (trade_dq, 3000)):
            _prune(dq, ts, w)

        bid5 = _depth_notional_5bps(bids, mid, "bid")
        ask5 = _depth_notional_5bps(asks, mid, "ask")
        total5 = bid5 + ask5
        dim5 = (bid5 - ask5) / max(total5, EPS)
        dim_hist.append((ts, dim5))
        micro_hist.append((ts, micro_minus_mid_bps))

        top5_share = 0.0
        if trade_dq:
            notionals = sorted([n for _, n, _ in trade_dq if n > 0.0])
            total_ntl = float(sum(notionals))
            if total_ntl > EPS:
                top5_share = float(sum(notionals[-5:]) / total_ntl)

        dim_vals = np.asarray([v for _, v in dim_hist], dtype=np.float64)
        depth_rv = float(np.sqrt(np.sum(np.diff(dim_vals) ** 2))) if dim_vals.size > 1 else 0.0
        micro_cross = _zero_cross_rate(list(micro_hist))

        l1_churn_500 = _sum_values(churn500) / max(total5, EPS)
        l1_churn_1000 = _sum_values(churn1000) / max(total5, EPS)
        ofi_pressure = abs(_sum_values(ofi500) / max(total5, EPS))

        buy_ntl_200 = sum(n for t, n, s in trade_dq if (ts - t) <= 200 and s == "Buy")
        sell_ntl_200 = sum(n for t, n, s in trade_dq if (ts - t) <= 200 and s == "Sell")

        last_side = None
        run_notional = 0.0
        for _, n, sgn in cluster:
            if last_side is None:
                last_side = sgn
                run_notional = n
            elif sgn == last_side:
                run_notional += n
            else:
                last_side = sgn
                run_notional = n

        out = {
            "top5_trade_share_notional_3000ms": top5_share,
            "depth_imbalance_realized_vol_1000ms": depth_rv,
            "microprice_zero_cross_rate_1000ms": micro_cross,
            "l1_churn_over_depth_1000ms": l1_churn_1000,
            "same_side_trade_cluster_notional_1000ms": float(run_notional),
            "ofi_pressure_x_churn_500ms": ofi_pressure * l1_churn_500,
            "bid_liquidity_void_bps": _liquidity_void_bps(bids, mid, "bid"),
            "ask_liquidity_void_bps": _liquidity_void_bps(asks, mid, "ask"),
            "post_buy_trade_ask_replenishment_200ms": _sum_values(ask_add200) / max(float(buy_ntl_200), EPS),
            "post_sell_trade_bid_replenishment_200ms": _sum_values(bid_add200) / max(float(sell_ntl_200), EPS),
        }

    return out


def test_event10_promoted_feature_formula_parity():
    fe = FeatureEngine()
    events = [
        _ob(0, 1, [(100.00, 10.0), (99.95, 4.0), (99.70, 8.0)], [(100.02, 10.0), (100.07, 5.0), (100.30, 8.0)]),
        _tr(100, 100.02, 5.0, 1),
        _ob(150, 2, [(100.01, 12.0), (99.98, 7.0), (99.70, 8.0)], [(100.02, 9.0), (100.04, 7.0), (100.30, 8.0)]),
        _tr(300, 100.03, 8.0, 1),
        _ob(450, 3, [(99.99, 8.0), (99.93, 10.0), (99.70, 8.0)], [(100.03, 13.0), (100.10, 2.0), (100.30, 8.0)]),
        _tr(700, 99.99, 6.0, -1),
        _ob(900, 4, [(100.01, 14.0), (99.98, 8.0), (99.70, 8.0)], [(100.04, 7.0), (100.09, 7.0), (100.30, 8.0)]),
        _tr(1050, 100.04, 4.0, 1),
        _ob(1200, 5, [(100.02, 9.0), (99.96, 11.0), (99.70, 8.0)], [(100.05, 15.0), (100.12, 3.0), (100.30, 8.0)]),
    ]

    last = None
    for event in events:
        last = fe.on_fast_event(event)

    assert last is not None
    assert last.is_decision is True
    assert last.features.shape == (153,)
    assert np.isfinite(last.features).all()

    names = fe.feature_names()
    for name in PROMOTED:
        assert names.count(name) == 1

    refs = _reference(events)
    for name in PROMOTED:
        prod = float(last.features[names.index(name)])
        ref = float(refs[name])
        diff = abs(prod - ref)
        assert np.isclose(prod, ref, rtol=1e-5, atol=1e-6), (
            f"feature={name} prod={prod:.12g} ref={ref:.12g} abs_diff={diff:.12g}"
        )

    prod_ofi = float(last.features[names.index("ofi_pressure_x_churn_500ms")])
    prod_churn = float(last.features[names.index("l1_churn_over_depth_1000ms")])
    prod_buy_repl = float(last.features[names.index("post_buy_trade_ask_replenishment_200ms")])
    prod_sell_repl = float(last.features[names.index("post_sell_trade_bid_replenishment_200ms")])
    prod_depth_rv = float(last.features[names.index("depth_imbalance_realized_vol_1000ms")])
    prod_cross = float(last.features[names.index("microprice_zero_cross_rate_1000ms")])

    assert prod_ofi >= 0.0
    assert prod_churn < 10.0
    assert prod_buy_repl > 1e-3
    assert prod_sell_repl > 1e-3
    assert prod_depth_rv > 0.0
    assert prod_cross > 0.0
