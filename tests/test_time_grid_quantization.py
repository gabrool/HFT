import ast
from pathlib import Path

import pytest



def load_time_grid_quantizer():
    source = Path(__file__).resolve().parent.parent.joinpath("CMSSL17.py").read_text()
    tree = ast.parse(source)

    module = {}
    required_consts = {"TIME_GRID_STEP_MS", "TIME_GRID_GUARD_MS"}
    loaded_consts = set()
    quantize_node = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in required_consts:
                    ast.fix_missing_locations(node)
                    exec(compile(ast.Module(body=[node], type_ignores=[]), filename="CMSSL17.py", mode="exec"), module)
                    loaded_consts.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name == "quantize_ts_ms":
            quantize_node = node

    if loaded_consts != required_consts or quantize_node is None:
        raise AssertionError("Failed to load time grid constants/function")

    ast.fix_missing_locations(quantize_node)
    exec(compile(ast.Module(body=[quantize_node], type_ignores=[]), filename="CMSSL17.py", mode="exec"), module)
    return module["TIME_GRID_STEP_MS"], module["TIME_GRID_GUARD_MS"], module["quantize_ts_ms"]


TIME_GRID_STEP_MS, TIME_GRID_GUARD_MS, quantize_ts_ms = load_time_grid_quantizer()


def test_time_grid_constants():
    assert TIME_GRID_STEP_MS == 100
    assert TIME_GRID_GUARD_MS == 50


def test_half_step_rounds_to_even_grid_index_when_lower_index_even():
    assert quantize_ts_ms(50) == 0


def test_half_step_rounds_to_even_grid_index_when_lower_index_odd():
    assert quantize_ts_ms(150) == 200


def test_exact_grid_points_unchanged():
    assert quantize_ts_ms(0) == 0
    assert quantize_ts_ms(100) == 100
    assert quantize_ts_ms(200) == 200


def test_max_deviation_exactly_guard_is_accepted():
    assert quantize_ts_ms(50) == 0
    assert quantize_ts_ms(150) == 200


def test_exceeding_guard_is_rejected_when_guard_tightened():
    with pytest.raises(ValueError, match="off-grid by 50ms"):
        quantize_ts_ms(50, guard_ms=49)
