from pathlib import Path
import ast
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable, Tuple


def load_trade_iter_precise():
    source = Path(__file__).resolve().parent.parent.joinpath("offline_ingest.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_trade_iter_precise":
            module = {
                "Decimal": Decimal,
                "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
                "Iterable": Iterable,
                "Tuple": Tuple,
                "_DEC_THOUSAND": Decimal("1000"),
            }
            ast.fix_missing_locations(node)
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    filename="offline_ingest.py",
                    mode="exec",
                ),
                module,
            )
            return module["_trade_iter_precise"]
    raise AssertionError("_trade_iter_precise not found")


_trade_iter_precise = load_trade_iter_precise()


def _iter(items):
    for item in items:
        yield item


def test_trade_iter_precise_numeric_override_behavior():
    rows = [
        (1111, 1, {"timestamp": "1717800000"}),
        (1222, 2, {"timestamp": "1717800000.0"}),
        (1333, 3, {"timestamp": "1717800000.123"}),
        (1444, 4, {"timestamp": "1.7178e9"}),
        (1555, 5, {"timestamp": "1.717800000123e9"}),
    ]

    got = list(_trade_iter_precise(_iter(rows)))

    assert got[0][0] == 1111
    assert got[1][0] == 1222
    assert got[2][0] == 1717800000123
    assert got[3][0] == 1444
    assert got[4][0] == 1717800000123


def test_trade_iter_precise_fallback_when_timestamp_missing_or_invalid():
    rows = [
        (2111, 1, {}),
        (2222, 2, {"timestamp": "not-a-number"}),
    ]

    got = list(_trade_iter_precise(_iter(rows)))

    assert got[0][0] == 2111
    assert got[1][0] == 2222
