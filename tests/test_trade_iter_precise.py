from pathlib import Path
import ast
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
from typing import Iterable, Tuple, Union
import math


def _load_function(file_name: str, function_name: str, extra_globals: dict):
    source = Path(__file__).resolve().parent.parent.joinpath(file_name).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = dict(extra_globals)
            ast.fix_missing_locations(node)
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    filename=file_name,
                    mode="exec",
                ),
                module,
            )
            return module[function_name]
    raise AssertionError(f"{function_name} not found in {file_name}")


_timestamp_to_ms_half_even = _load_function(
    "CMSSL17.py",
    "timestamp_to_ms_half_even",
    {
        "Decimal": Decimal,
        "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
        "InvalidOperation": InvalidOperation,
        "Union": Union,
        "math": math,
        "_TS_SECONDS_THRESHOLD": Decimal("1e12"),
        "_TS_MILLI_SCALE": Decimal("1000"),
    },
)


_trade_iter_precise = _load_function(
    "offline_ingest.py",
    "_trade_iter_precise",
    {
        "Iterable": Iterable,
        "Tuple": Tuple,
        "timestamp_to_ms_half_even": _timestamp_to_ms_half_even,
    },
)


def _iter(items):
    for item in items:
        yield item


def test_timestamp_to_ms_half_even_boundary_values():
    assert _timestamp_to_ms_half_even("1717800000.1234995") == 1717800000123
    assert _timestamp_to_ms_half_even("1717800000.1235000") == 1717800000124
    assert _timestamp_to_ms_half_even("-1717800000.1235000") == -1717800000124
    assert _timestamp_to_ms_half_even("1717800000123") == 1717800000123


def test_trade_iter_precise_uses_shared_conversion_policy():
    rows = [
        (1111, 1, {"timestamp": "1717800000"}),
        (1222, 2, {"timestamp": "1717800000.0"}),
        (1333, 3, {"timestamp": "1717800000.1234995"}),
        (1444, 4, {"timestamp": "1717800000.1235000"}),
        (1555, 5, {"timestamp": "-1717800000.1235000"}),
        (1666, 6, {"timestamp": "1.717800000123e9"}),
    ]

    got = list(_trade_iter_precise(_iter(rows)))

    # Whole-second values preserve upstream bucket spread timestamps.
    assert got[0][0] == 1111
    assert got[1][0] == 1222
    # Subsecond values are normalized by the shared half-even helper.
    assert got[2][0] == 1717800000123
    assert got[3][0] == 1717800000124
    assert got[4][0] == -1717800000124
    assert got[5][0] == 1717800000123


def test_trade_iter_precise_fallback_when_timestamp_missing_or_invalid():
    rows = [
        (2111, 1, {}),
        (2222, 2, {"timestamp": "not-a-number"}),
    ]

    got = list(_trade_iter_precise(_iter(rows)))

    assert got[0][0] == 2111
    assert got[1][0] == 2222
