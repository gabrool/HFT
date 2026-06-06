from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


PRODUCTION_ROOT = Path("mmrt")

FORBIDDEN_NUMERIC_ASSIGNMENTS = {
    "DEFAULT_TICK_SIZE": {0.1, 0.10},
    "DEFAULT_STEP_SIZE": {0.001},
    "DEFAULT_MIN_NOTIONAL": {5.0},
}

FORBIDDEN_DERIVED_DEFAULTS = {
    "tick_size": {0.1, 0.10},
    "step_size": {0.001},
    "min_notional": {5.0},
}

FORBIDDEN_CLI_FLAGS = {
    "--tick-size",
    "--step-size",
    "--min-notional",
}


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _literal_number(node: ast.AST) -> float | None:
    try:
        value: Any = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _is_add_argument_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"


def _iter_py_files():
    yield from PRODUCTION_ROOT.rglob("*.py")


def test_production_code_has_no_hardcoded_execution_symbol_rule_defaults():
    offenders: list[str] = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_add_argument_call(node):
                for arg in node.args:
                    flag = _literal_string(arg)
                    if flag in FORBIDDEN_CLI_FLAGS:
                        offenders.append(f"{path}:{node.lineno}: forbidden CLI flag {flag}")
            elif isinstance(node, ast.Assign):
                value = _literal_number(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name) and value in FORBIDDEN_NUMERIC_ASSIGNMENTS.get(target.id, set()):
                        offenders.append(f"{path}:{node.lineno}: forbidden numeric default {target.id}={value:g}")
            elif isinstance(node, ast.AnnAssign):
                value = _literal_number(node.value) if node.value is not None else None
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if value in FORBIDDEN_NUMERIC_ASSIGNMENTS.get(name, set()):
                        offenders.append(f"{path}:{node.lineno}: forbidden numeric default {name}={value:g}")
                    if value in FORBIDDEN_DERIVED_DEFAULTS.get(name, set()):
                        offenders.append(f"{path}:{node.lineno}: forbidden dataclass symbol-rule default {name}={value:g}")
    assert offenders == []


def test_runtime_execution_modules_do_not_parse_exchange_info():
    paths = [
        Path("mmrt/execution/quote_geometry.py"),
        Path("mmrt/execution/fill_sim.py"),
        Path("mmrt/execution/env.py"),
        *Path("mmrt/rl").rglob("*.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "exchangeInfo" not in text, f"exchangeInfo reference found in {path}"
        assert "binance_exchange_info" not in text, f"binance_exchange_info reference found in {path}"
