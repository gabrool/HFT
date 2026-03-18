from pathlib import Path
import ast
from typing import Any, Dict, Tuple

import pytest


def _load_function(file_name: str, function_name: str, extra_globals: dict):
    source = Path(__file__).resolve().parent.parent.joinpath(file_name).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = dict(extra_globals)
            ast.fix_missing_locations(node)
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), filename=file_name, mode="exec"),
                module,
            )
            return module[function_name]
    raise AssertionError(f"{function_name} not found in {file_name}")


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeTorch:
    Tensor = _FakeTensor


_canonical_market_ppo_arch_field = _load_function(
    "RL_exec.py",
    "_canonical_market_ppo_arch_field",
    {"Dict": Dict, "Any": Any, "Tuple": Tuple, "torch": _FakeTorch},
)

_canonical_market_ppo_action_dim = _load_function(
    "RL_exec.py",
    "_canonical_market_ppo_action_dim",
    {"Dict": Dict, "Any": Any},
)


def test_canonical_market_ppo_arch_field_accepts_sequence_and_tensor_payloads():
    assert _canonical_market_ppo_arch_field({"policy_hidden_dims": [16, 8]}, "policy_hidden_dims") == (16, 8)
    assert _canonical_market_ppo_arch_field(
        {"value_hidden_dims": _FakeTensor((12, 6))},
        "value_hidden_dims",
    ) == (12, 6)


@pytest.mark.parametrize("payload", [{}, {"policy_hidden_dims": "16,8"}, {"policy_hidden_dims": []}, {"policy_hidden_dims": [16, 0]}])
def test_canonical_market_ppo_arch_field_rejects_noncanonical_payloads(payload):
    with pytest.raises(ValueError, match="Only canonical PPO checkpoints are supported"):
        _canonical_market_ppo_arch_field(payload, "policy_hidden_dims")


@pytest.mark.parametrize("payload", [{"action_dim": 3}, {"action_dim": True}, {"action_dim": "4"}])
def test_canonical_market_ppo_action_dim_accepts_positive_checkpoint_values(payload):
    assert _canonical_market_ppo_action_dim(payload) == int(payload["action_dim"])


@pytest.mark.parametrize("payload", [{}, {"action_dim": 0}, {"action_dim": -1}, {"action_dim": "abc"}])
def test_canonical_market_ppo_action_dim_rejects_noncanonical_payloads(payload):
    with pytest.raises(ValueError, match="Only canonical PPO checkpoints are supported"):
        _canonical_market_ppo_action_dim(payload)
