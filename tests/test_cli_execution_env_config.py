from pathlib import Path

import pytest

from mmrt.cli.execution_env_config import (
    ExecutionEnvConfigBuildInput,
    build_execution_env_config_from_input,
)


def test_shared_env_config_builder_matches_quote_gap_and_fee():
    params = ExecutionEnvConfigBuildInput(
        post_only_gap_ticks=2,
        maker_fee_bps=-0.25,
        adverse_signals_enabled=True,
    )
    cfg = build_execution_env_config_from_input(params)
    assert cfg.quote_geometry_config.post_only_gap_ticks == 2
    assert cfg.adverse_runtime_config.post_only_gap_ticks == 2
    assert cfg.fill_simulator_config.maker_fee_bps == -0.25
    assert cfg.adverse_runtime_config.executable_edge.maker_fee_bps == -0.25
    assert cfg.observation_builder_config.inventory_qty_reference == cfg.action_spec.max_order_qty


def test_shared_env_config_builder_inventory_reference_tracks_max_order_qty():
    cfg = build_execution_env_config_from_input(
        ExecutionEnvConfigBuildInput(max_order_qty=0.005)
    )
    assert cfg.action_spec.max_order_qty == 0.005
    assert cfg.observation_builder_config.inventory_qty_reference == 0.005


def test_shared_env_config_builder_rejects_queue_weights_above_one():
    with pytest.raises(ValueError, match="l2_decrease_weight"):
        ExecutionEnvConfigBuildInput(l2_decrease_weight=1.01)

    with pytest.raises(ValueError, match="trade_at_level_weight"):
        ExecutionEnvConfigBuildInput(trade_at_level_weight=1.01)


def test_shared_env_config_builder_rejects_bool_queue_weights():
    with pytest.raises(ValueError):
        ExecutionEnvConfigBuildInput(l2_decrease_weight=True)  # type: ignore[arg-type]


def test_execution_env_config_construction_is_centralized_in_clis():
    cli_paths = [
        Path("mmrt/cli/train_execution_ppo.py"),
        Path("mmrt/cli/evaluate_execution_policy.py"),
        Path("mmrt/cli/audit_execution_sim.py"),
    ]
    forbidden = (
        "QuoteGeometryConfig(",
        "FillSimulatorConfig(",
        "AdverseRuntimeConfig(",
        "ExecutableEdgeConfig(",
        "RewardConfig(",
    )

    for path in cli_paths:
        text = path.read_text(encoding="utf-8")
        assert (
            "build_execution_env_config_from_attrs" in text
            or "build_execution_env_config_from_input" in text
        )
        offenders = [token for token in forbidden if token in text]
        assert (
            offenders == []
        ), f"{path} has duplicated env config construction: {offenders}"
