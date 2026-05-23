import math
import numpy as np
from test_feature_event_result_contract import _install_optional_dependency_stubs
_install_optional_dependency_stubs()
from CMSSL17 import FeatureEngine


def _ob(ts, tp, bids, asks):
    return ("ob", ts, 0, tp, tuple(bids), tuple(asks))

def _tr(ts, price, size, side):
    return {"timestamp": ts, "price": price, "size": size, "side": side}


def test_event10_promoted_feature_formula_parity():
    fe = FeatureEngine()
    events = [
        _ob(0, 1, [(100.0,10.0),(99.95,9.0)], [(100.02,12.0),(100.07,11.0)]),
        _tr(100, 100.02, 5.0, "Buy"),
        _ob(150, 2, [(100.0,12.0),(99.95,9.0)], [(100.02,10.0),(100.07,13.0)]),
        _tr(300, 100.02, 8.0, "Buy"),
        _ob(450, 2, [(99.99,11.0),(99.94,8.0)], [(100.03,11.0),(100.08,12.0)]),
        _tr(700, 99.99, 7.0, "Sell"),
        _ob(900, 2, [(100.01,9.0),(99.96,8.0)], [(100.04,14.0),(100.10,3.0)]),
        _ob(1200,2, [(100.01,13.0),(99.97,7.0)], [(100.04,9.0),(100.10,2.0)]),
    ]
    last = None
    for e in events:
        last = fe.on_fast_event(e)
    assert last is not None and last.is_decision
    names = fe.feature_names()
    promoted = [
        "top5_trade_share_notional_3000ms","depth_imbalance_realized_vol_1000ms","microprice_zero_cross_rate_1000ms",
        "l1_churn_over_depth_1000ms","same_side_trade_cluster_notional_1000ms","ofi_pressure_x_churn_500ms",
        "bid_liquidity_void_bps","ask_liquidity_void_bps","post_buy_trade_ask_replenishment_200ms","post_sell_trade_bid_replenishment_200ms"
    ]
    for n in promoted:
        assert names.count(n) == 1
    assert last.features.shape == (153,)
    assert np.isfinite(last.features).all()

    idx={n:names.index(n) for n in promoted}
    vals={n:float(last.features[idx[n]]) for n in promoted}

    assert vals["ofi_pressure_x_churn_500ms"] >= 0.0
    assert abs(vals["l1_churn_over_depth_1000ms"]) < 10.0
    assert vals["post_buy_trade_ask_replenishment_200ms"] >= 0.0
    assert vals["post_sell_trade_bid_replenishment_200ms"] >= 0.0
