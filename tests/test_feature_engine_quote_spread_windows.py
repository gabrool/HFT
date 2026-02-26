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


def _ob_event(ts_ms: int, bid: float, ask: float, bid_sz: float = 1.0, ask_sz: float = 1.0):
    return (
        "ob",
        ts_ms,
        0,
        {
            "type": "snapshot",
            "data": {
                "b": [[str(bid), str(bid_sz)]],
                "a": [[str(ask), str(ask_sz)]],
            },
        },
    )


def test_quote_and_spread_change_counts_include_true_5s_windows():
    fe = FeatureEngine()

    events = [
        _ob_event(0, 100.0, 100.5),
        _ob_event(500, 100.0, 100.5),
        _ob_event(1_000, 100.0, 100.5),
        _ob_event(1_500, 100.0, 100.5),
        _ob_event(2_000, 100.0, 100.6),
        _ob_event(2_500, 100.0, 100.6),
        _ob_event(3_000, 100.0, 100.6),
        _ob_event(3_500, 100.0, 100.7),
        _ob_event(4_000, 100.0, 100.7),
        _ob_event(4_500, 100.0, 100.7),
        _ob_event(5_000, 100.0, 100.7),
        _ob_event(5_500, 100.0, 100.7),
        _ob_event(6_000, 100.0, 100.7),
    ]

    for event in events:
        fe.on_event(event)

    assert len(fe.quotes_1s) == 3
    assert len(fe.quotes_5s) == 11
    assert len(fe.quotes_5s) > len(fe.quotes_1s)

    assert len(fe._spread_change_deques[1_000]) == 0
    assert len(fe._spread_change_deques[5_000]) == 2
