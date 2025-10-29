from pathlib import Path
import ast

import pytest


def load_merge_event_time():
    source = Path(__file__).resolve().parent.parent.joinpath("CMSSL15.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "merge_event_time":
            module = {}
            ast.fix_missing_locations(node)
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename="CMSSL15.py", mode="exec"), module)
            return module["merge_event_time"]
    raise AssertionError("merge_event_time not found")


merge_event_time = load_merge_event_time()


def _iter(items):
    for item in items:
        yield item


def test_trade_preferred_when_timestamps_tie():
    ob_events = _iter([(1000, 1, {"type": "ob"})])
    trade_events = _iter([(1000, 1, {"type": "trade"})])

    merged = list(merge_event_time(ob_events, trade_events))

    assert merged[0][0] == "trade"
    assert [event[0] for event in merged] == ["trade", "ob"]
    assert merged[0][1] == merged[1][1] == 1000
