from dataclasses import dataclass
from pathlib import Path
import ast
from typing import Any, Dict, Optional, Tuple

import pytest

np = pytest.importorskip("numpy")


def _load_symbols(file_name: str, symbol_names: set[str], extra_globals: dict) -> dict:
    source = Path(__file__).resolve().parent.parent.joinpath(file_name).read_text()
    tree = ast.parse(source)
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in symbol_names:
            ast.fix_missing_locations(node)
            selected_nodes.append(node)
    if len(selected_nodes) != len(symbol_names):
        missing = symbol_names - {node.name for node in selected_nodes}
        raise AssertionError(f"Missing symbols in {file_name}: {sorted(missing)}")
    module = dict(extra_globals)
    exec(
        compile(ast.Module(body=selected_nodes, type_ignores=[]), filename=file_name, mode="exec"),
        module,
    )
    return {name: module[name] for name in symbol_names}


symbols = _load_symbols(
    "RL_exec.py",
    {"RolloutStartSamplingConfig", "_build_rollout_start_sampler"},
    {
        "dataclass": dataclass,
        "np": np,
        "Optional": Optional,
        "Tuple": Tuple,
        "Dict": Dict,
        "Any": Any,
    },
)
RolloutStartSamplingConfig = symbols["RolloutStartSamplingConfig"]
_build_rollout_start_sampler = symbols["_build_rollout_start_sampler"]


class _DummyEnv:
    def __init__(self, n: int = 64):
        self.n = n
        self.features = np.zeros((n, 3), dtype=np.float64)
        self._feature_layout = {"dir_logits": slice(0, 3)}
        self.decision_ts = np.arange(n, dtype=np.int64)


def test_rollout_start_sampler_defaults_exclusion_window_to_rollout_horizon():
    env = _DummyEnv(n=80)
    config = RolloutStartSamplingConfig(enabled=True, start_exclusion_window=None)

    sampler = _build_rollout_start_sampler(env, config, rollout_horizon=17)

    assert sampler is not None
    assert sampler["start_exclusion_window"] == 17
    assert isinstance(sampler["start_exclusion_window"], int)
