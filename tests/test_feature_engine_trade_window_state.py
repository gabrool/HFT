from collections import deque
from pathlib import Path
import ast
import math
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np


def load_feature_engine():
    source_path = Path(__file__).resolve().parent.parent / "CMSSL17.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "coerce_ts_ms":
            wanted.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "FeatureEngine":
            wanted.append(node)

    module = {
        "np": np,
        "math": math,
        "deque": deque,
        "Any": Any,
        "Deque": Deque,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
    }
    ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    exec(compile(ast.Module(body=wanted, type_ignores=[]), filename="CMSSL17.py", mode="exec"), module)
    return module["FeatureEngine"]


FeatureEngine = load_feature_engine()


def _ob_event(ts_ms: int, bid: float = 100.0, ask: float = 100.5):
    return (
        "ob",
        ts_ms,
        0,
        {
            "type": "snapshot",
            "data": {
                "b": [[str(bid), "1.0"]],
                "a": [[str(ask), "1.0"]],
            },
        },
    )


def _trade_event(ts_ms: int, side: str, price: float, size: float):
    return (
        "trade",
        ts_ms,
        0,
        {
            "side": side,
            "price": price,
            "size": size,
        },
    )


def test_trade_window_stats_empty_window():
    fe = FeatureEngine()
    fe.on_event(_ob_event(0))

    stats = fe._compute_trade_window_stats(1_000, mid=100.25)

    assert stats["buy_cnt"] == 0.0
    assert stats["sell_cnt"] == 0.0
    assert stats["buy_vol"] == 0.0
    assert stats["sell_vol"] == 0.0
    assert stats["buy_mean"] == 0.0
    assert stats["sell_mean"] == 0.0
    assert stats["imbalance"] == 0.0
    assert stats["toxicity"] == 0.0
    assert stats["trade_through"] == 0.0


def test_trade_window_stats_single_trade():
    fe = FeatureEngine()
    fe.on_event(_ob_event(0))
    fe.on_event(_trade_event(100, "buy", 101.0, 2.5))

    stats = fe._compute_trade_window_stats(1_000, mid=100.25)

    assert stats["buy_cnt"] == 1.0
    assert stats["sell_cnt"] == 0.0
    assert stats["buy_vol"] == 2.5
    assert stats["sell_vol"] == 0.0
    assert stats["buy_mean"] == 2.5
    assert stats["sell_mean"] == 0.0
    assert stats["buy_max"] == 2.5
    assert stats["sell_max"] == 0.0
    assert stats["imbalance"] == 1.0
    assert stats["toxicity"] == 1.0
    assert np.isclose(stats["trade_through"], (101.0 / 100.25) - 1.0)


def test_trade_window_stats_mixed_sides():
    fe = FeatureEngine()
    fe.on_event(_ob_event(0))
    fe.on_event(_trade_event(100, "buy", 101.0, 2.0))
    fe.on_event(_trade_event(200, "sell", 99.5, 3.0))
    fe.on_event(_trade_event(300, "buy", 100.5, 1.0))

    stats = fe._compute_trade_window_stats(1_000, mid=100.0)

    assert stats["buy_cnt"] == 2.0
    assert stats["sell_cnt"] == 1.0
    assert stats["buy_vol"] == 3.0
    assert stats["sell_vol"] == 3.0
    assert stats["buy_mean"] == 1.5
    assert stats["sell_mean"] == 3.0
    assert stats["buy_max"] == 2.0
    assert stats["sell_max"] == 3.0
    assert stats["net_flow"] == 0.0
    assert stats["imbalance"] == 0.0
    assert stats["toxicity"] == 0.0
    expected_through = ((101.0 / 100.0) - 1.0) - ((99.5 / 100.0) - 1.0) + ((100.5 / 100.0) - 1.0)
    assert np.isclose(stats["trade_through"], expected_through)


def test_trade_window_rapid_burst_then_quiet_prunes_state():
    fe = FeatureEngine()
    fe.on_event(_ob_event(0))

    for i in range(20):
        side = "buy" if i % 2 == 0 else "sell"
        fe.on_event(_trade_event(100 + i, side, 100.0 + (i * 0.01), 1.0 + (i % 3)))

    stats_before = fe._compute_trade_window_stats(1_000, mid=100.25)
    assert stats_before["buy_cnt"] > 0.0
    assert stats_before["sell_cnt"] > 0.0

    fe.on_event(_ob_event(7_000))

    stats_1s = fe._compute_trade_window_stats(1_000, mid=100.25)
    stats_5s = fe._compute_trade_window_stats(5_000, mid=100.25)
    state_1s = fe.trade_window_state[1_000]
    state_5s = fe.trade_window_state[5_000]

    assert stats_1s["buy_cnt"] == 0.0
    assert stats_1s["sell_cnt"] == 0.0
    assert stats_5s["buy_cnt"] == 0.0
    assert stats_5s["sell_cnt"] == 0.0
    assert state_1s["vol_sum"] == 0.0
    assert state_5s["vol_sum"] == 0.0
    assert len(state_1s["buy_max_q"]) == 0
    assert len(state_1s["sell_max_q"]) == 0
    assert len(state_5s["buy_max_q"]) == 0
    assert len(state_5s["sell_max_q"]) == 0
