import ast
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def _load_utc_day_bounds_ms():
    source = Path(__file__).resolve().parent.parent.joinpath("offline_snapshots.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_utc_day_bounds_ms":
            module = {
                "datetime": datetime,
                "timedelta": timedelta,
                "timezone": timezone,
                "ONE_DAY": timedelta(days=1),
                "date": date,
            }
            ast.fix_missing_locations(node)
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), filename="offline_snapshots.py", mode="exec"),
                module,
            )
            return module["_utc_day_bounds_ms"]
    raise AssertionError("_utc_day_bounds_ms not found")


_utc_day_bounds_ms = _load_utc_day_bounds_ms()


def test_utc_day_bounds_ms_is_exact_utc_midnight_epoch_window():
    start_ms, end_ms = _utc_day_bounds_ms(date(2020, 1, 1))

    assert start_ms == 1577836800000
    assert end_ms == 1577923200000
    assert end_ms - start_ms == 86_400_000
