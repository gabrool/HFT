from pathlib import Path
import ast


def _load_symbols():
    source_path = Path(__file__).resolve().parent.parent / "offline_ingest.py"
    source = source_path.read_text()
    tree = ast.parse(source)

    wanted_assigns = {
        "BYBIT_BAD_EXAMPLES_N",
        "BYBIT_BAD_FRAC_ABORT",
        "BYBIT_BAD_ABS_ABORT",
    }
    wanted_defs = {"DayQuality", "WeekQuality", "_day_bad_abs_and_total"}

    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            if any(name in {"os", "dataclasses", "typing"} for name in names):
                body.append(node)
            continue

        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(name in wanted_assigns for name in targets):
                body.append(node)
            continue

        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted_defs:
            body.append(node)

    module = {}
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, filename=str(source_path), mode="exec"), module)
    return module


M = _load_symbols()
DayQuality = M["DayQuality"]
WeekQuality = M["WeekQuality"]
day_bad_abs_and_total = M["_day_bad_abs_and_total"]


def test_day_bad_abs_and_total_counts_drop_error_bad_and_inputs():
    dq = DayQuality(day="2024-01-01", ob_path="ob.zip", th_path="th.csv.gz")
    dq.increment_counter("ob", "total", 100)
    dq.increment_counter("th", "total", 50)
    dq.increment_counter("ob", "dropped_backstep", 2)
    dq.increment_counter("th", "bad_ts", 3)
    dq.increment_counter("merge", "merge_error", 4)
    dq.increment_counter("chain", "chain_clamped_backstep", 9)

    bad_abs, total = day_bad_abs_and_total(dq)

    assert bad_abs == 9
    assert total == 150


def test_week_quality_serializes_corruption_abort_flag_and_taint():
    wq = WeekQuality(week_key="01-01-2024-to-07-01-2024")
    dq = DayQuality(day="2024-01-01", ob_path="ob.zip", th_path="th.csv.gz")
    dq.set_abort_flag("aborted_due_to_corruption", True)
    wq.add_day(dq)
    wq.append_note("[warn] corruption abort day=2024-01-01")
    wq.recompute_totals()

    payload = wq.to_dict()

    assert payload["tainted"] is True
    assert payload["notes"]
    assert payload["days"][0]["abort_flags"]["aborted_due_to_corruption"] is True
