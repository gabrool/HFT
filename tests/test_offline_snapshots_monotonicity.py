import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytest


def load_build_snapshots_from_ob_files():
    source = Path(__file__).resolve().parent.parent.joinpath("offline_snapshots.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_snapshots_from_ob_files":
            module = {
                "List": List,
                "Optional": Optional,
            }

            @dataclass
            class SnapshotSeries:
                ts: list
                best_bid: list
                best_ask: list
                best_bid_size: list
                best_ask_size: list
                time_since_last_ob_update_ms: list

                def append(self, ts_ms, bid, ask, bid_size, ask_size, stale_ms):
                    self.ts.append(ts_ms)
                    self.best_bid.append(bid)
                    self.best_ask.append(ask)
                    self.best_bid_size.append(bid_size)
                    self.best_ask_size.append(ask_size)
                    self.time_since_last_ob_update_ms.append(stale_ms)

            class FeatureEngine:
                def _parse_event(self, raw):
                    return raw["etype"], raw["ts_ms_raw"], raw.get("payload", {})

                def _update_book_from_ob(self, payload):
                    return None

                def _book_best(self):
                    return 100.0, 101.0, 1.0, 2.0

            def quantize_ts_ms(ts_ms_raw, step_ms, guard_ms):
                return ts_ms_raw

            def iter_ob_events_many(ob_paths):
                for stream in ob_paths:
                    yield from stream

            module.update(
                {
                    "SnapshotSeries": SnapshotSeries,
                    "FeatureEngine": FeatureEngine,
                    "quantize_ts_ms": quantize_ts_ms,
                    "iter_ob_events_many": iter_ob_events_many,
                    "TIME_GRID_STEP_MS": 1_000,
                    "TIME_GRID_GUARD_MS": 0,
                }
            )

            ast.fix_missing_locations(node)
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), filename="offline_snapshots.py", mode="exec"),
                module,
            )
            return module["build_snapshots_from_ob_files"]
    raise AssertionError("build_snapshots_from_ob_files not found")


build_snapshots_from_ob_files = load_build_snapshots_from_ob_files()


def _ob(ts_ms_raw: int):
    return {"etype": "ob", "ts_ms_raw": ts_ms_raw, "payload": {}}


def test_build_snapshots_from_ob_files_raises_on_decreasing_quantized_ts():
    day1 = [_ob(2_000), _ob(3_000)]
    day2 = [_ob(1_000), _ob(4_000)]

    with pytest.raises(ValueError, match="Non-decreasing quantized OB timestamps violated"):
        build_snapshots_from_ob_files([day1, day2])


def test_build_snapshots_from_ob_files_allows_equal_quantized_ts():
    day1 = [_ob(2_000), _ob(2_000)]
    day2 = [_ob(2_000), _ob(3_000)]

    series = build_snapshots_from_ob_files([day1, day2])

    assert series.ts
