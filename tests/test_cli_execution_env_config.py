from pathlib import Path

from mmrt.cli.execution_env_config import ExecutionEnvConfigBuildInput, build_execution_env_config_from_input


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


def test_execution_env_config_not_duplicated_in_clis():
    for path in (
        "mmrt/cli/train_execution_ppo.py",
        "mmrt/cli/evaluate_execution_policy.py",
        "mmrt/cli/audit_execution_sim.py",
    ):
        source = Path(path).read_text()
        assert "QuoteGeometryConfig(" not in source
        assert "FillSimulatorConfig(" not in source
        assert "AdverseRuntimeConfig(" not in source
