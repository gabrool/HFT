import math
from collections import deque

import numpy as np

import sys
import types


def _install_optional_dependency_stubs():
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name):
            return self

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

    class _Parameter:
        def __init__(self, value=None, *args, **kwargs):
            self.value = value

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _Dummy
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    torch_mod.ones = lambda *args, **kwargs: np.ones(args[0] if args else (), dtype=np.float32)
    torch_mod.tensor = lambda *args, **kwargs: np.asarray(args[0] if args else 0)
    torch_mod.empty = lambda *args, **kwargs: np.empty(args[0] if len(args) == 1 else args, dtype=np.float32)
    torch_mod.randn = lambda *args, **kwargs: np.random.randn(*args)
    torch_mod.exp = np.exp
    torch_mod.log = np.log
    torch_mod.arange = lambda *args, **kwargs: np.arange(*args)
    torch_mod.float32 = np.float32
    torch_mod.no_grad = lambda func=None: (lambda f: f) if func is None else func

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = _Module
    nn_mod.Parameter = _Parameter
    for name in ("Linear", "Conv1d", "SiLU", "ReLU", "GELU", "Dropout", "LayerNorm", "BatchNorm1d", "MultiheadAttention", "Sequential", "ModuleList"):
        setattr(nn_mod, name, type(name, (_Module,), {}))
    functional_mod = types.ModuleType("torch.nn.functional")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = type("Dataset", (_Module,), {})
    data_mod.DataLoader = type("DataLoader", (_Module,), {})
    functorch_mod = types.ModuleType("torch._functorch")
    config_mod = types.ModuleType("torch._functorch.config")
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.Optimizer = type("Optimizer", (_Module,), {"__init__": lambda self, *args, **kwargs: None})

    torch_mod.optim = optim_mod
    torch_mod.nn = nn_mod
    torch_mod.utils = utils_mod
    torch_mod._functorch = functorch_mod

    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = functional_mod
    sys.modules["torch.optim"] = optim_mod
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod
    sys.modules["torch._functorch"] = functorch_mod
    sys.modules["torch._functorch.config"] = config_mod


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


def _ob(ts, change_id, bids, asks):
    _ = change_id
    return ("ob", ts, 0, 1, tuple(bids), tuple(asks))


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

    if side == "bid":
        filtered = [float(px) for px, sz in levels if float(sz) > EPS and float(px) >= mid * (1.0 - max_bps / 1e4)]
    else:
        filtered = [float(px) for px, sz in levels if float(sz) > EPS and float(px) <= mid * (1.0 + max_bps / 1e4)]

    if len(filtered) < 2:
        return 0.0

    gaps = [abs(filtered[i] - filtered[i + 1]) / max(mid, EPS) * 1e4 for i in range(len(filtered) - 1)]
    return float(min(max(max(gaps) if gaps else 0.0, 0.0), max_bps))


def _zero_cross_rate(points):
    vals = np.asarray([v for _, v in points], dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    signs = np.sign(vals)
    last = 0.0
    filled = []
    for s in signs:
        if s == 0:
            filled.append(last)
        else:
            filled.append(s)
            last = s
    f = np.asarray(filled, dtype=np.float64)
    v = f[f != 0]
    return 0.0 if v.size <= 1 else float(np.sum(v[1:] != v[:-1]) / max(len(v) - 1, 1))


def _max_same_side_run_notional(cluster_dq):
    max_run = 0.0
    cur_side = None
    cur_sum = 0.0
    for _, notional, side_sign in cluster_dq:
        if cur_side is None or side_sign != cur_side:
            cur_side = side_sign
            cur_sum = notional
        else:
            cur_sum += notional
        max_run = max(max_run, cur_sum)
    return float(max_run)


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
            notional = float(px) * float(sz)
            trade_dq.append((ts, notional, side_sign))
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

        _prune(churn500, ts, 500)
        _prune(churn1000, ts, 1000)
        _prune(ofi500, ts, 500)
        _prune(ask_add200, ts, 200)
        _prune(bid_add200, ts, 200)
        _prune(cluster, ts, 1000)
        _prune(trade_dq, ts, 3000)

        bid5 = _depth_notional_5bps(bids, mid, "bid")
        ask5 = _depth_notional_5bps(asks, mid, "ask")
        total5 = bid5 + ask5
        dim5 = (bid5 - ask5) / max(total5, EPS)

        dim_hist.append((ts, dim5))
        _prune(dim_hist, ts, 1000)
        micro_hist.append((ts, micro_minus_mid_bps))
        _prune(micro_hist, ts, 1000)

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

        buy_ntl_200 = sum(n for t, n, s in trade_dq if (ts - t) <= 200 and s > 0)
        sell_ntl_200 = sum(n for t, n, s in trade_dq if (ts - t) <= 200 and s < 0)

        out = {
            "top5_trade_share_notional_3000ms": top5_share,
            "depth_imbalance_realized_vol_1000ms": depth_rv,
            "microprice_zero_cross_rate_1000ms": micro_cross,
            "l1_churn_over_depth_1000ms": l1_churn_1000,
            "same_side_trade_cluster_notional_1000ms": _max_same_side_run_notional(cluster),
            "ofi_pressure_x_churn_500ms": ofi_pressure * l1_churn_500,
            "bid_liquidity_void_bps": _liquidity_void_bps(bids, mid, "bid"),
            "ask_liquidity_void_bps": _liquidity_void_bps(asks, mid, "ask"),
            "post_buy_trade_ask_replenishment_200ms": _sum_values(ask_add200) / max(float(buy_ntl_200), EPS),
            "post_sell_trade_bid_replenishment_200ms": _sum_values(bid_add200) / max(float(sell_ntl_200), EPS),
        }

    return out


def test_event10_promoted_feature_formula_parity():
    fe = FeatureEngine()
    fe._transform_features = lambda raw, dt_ms: raw.astype(np.float32)

    events = [
        _ob(0, 1, [(100.00, 10.0), (99.98, 4.0), (99.91, 8.0)], [(100.03, 8.0), (100.05, 5.0), (100.12, 8.0)]),
        _tr(100, 100.03, 5.0, +1),
        _ob(150, 2, [(100.01, 6.0), (99.99, 6.0), (99.92, 8.0)], [(100.04, 13.0), (100.06, 7.0), (100.13, 8.0)]),
        _tr(300, 100.04, 8.0, +1),
        _ob(450, 3, [(99.99, 14.0), (99.95, 10.0), (99.90, 8.0)], [(100.03, 7.0), (100.08, 2.0), (100.13, 8.0)]),
        _tr(700, 99.99, 6.0, -1),
        _ob(900, 4, [(100.01, 8.0), (99.98, 8.0), (99.92, 8.0)], [(100.04, 8.0), (100.09, 7.0), (100.13, 8.0)]),
        _tr(1000, 100.04, 4.0, +1),
        _tr(1050, 100.04, 6.0, +1),
        _tr(1100, 100.02, 5.0, -1),
        _ob(1200, 5, [(100.02, 14.0), (99.99, 9.0), (99.92, 8.0)], [(100.05, 15.0), (100.10, 3.0), (100.14, 8.0)]),
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
    assert 0.0 < prod_churn < 10.0
    assert prod_buy_repl > 1e-3
    assert prod_sell_repl > 1e-3
    assert prod_depth_rv > 0.0
    assert prod_cross > 0.0
