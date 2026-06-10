from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from mmrt.cli.train_execution_ppo import (
    ExecutionPPOTrainCLIConfig,
    build_arg_parser,
    _build_env_config,
    _build_training_config,
    _config_from_args,
    _summary_config,
)
from tests.test_audit_execution_sim import _linear_artifact_for_tape, _l2, _tape


def test_parser_can_disable_l2_trade_dedupe():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape", "--no-dedupe-l2-decrease-with-trade-prints"])
    config = _config_from_args(args)
    env_config = _build_env_config(config)
    assert config.dedupe_l2_decrease_with_trade_prints is False
    assert env_config.fill_simulator_config.queue_model.dedupe_l2_decrease_with_trade_prints is False
    assert _summary_config(config)["dedupe_l2_decrease_with_trade_prints"] is False


def test_parser_dedupe_l2_trade_default_enabled():
    parser = build_arg_parser()
    args = parser.parse_args(["--tape-root", "/tmp/tape"])
    config = _config_from_args(args)
    assert config.dedupe_l2_decrease_with_trade_prints is True


def test_adverse_runtime_config_inherits_post_only_gap_from_ppo_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--tape-root", "/tmp/tape",
        "--adverse-signals-npz", "/tmp/adverse.npz",
        "--post-only-gap-ticks", "2",
    ])
    config = _config_from_args(args)
    env_config = _build_env_config(config)

    assert env_config.adverse_runtime_config is not None
    assert env_config.quote_geometry_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.executable_edge.maker_fee_bps == env_config.fill_simulator_config.maker_fee_bps


def test_train_execution_ppo_accepts_explicit_later_linear_signal_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    linear_signals = _linear_artifact_for_tape(tape, n_rows=3, decision_interval_us=500_000)
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        start_event_index=1,
        max_episode_steps=1,
    )

    linear_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_interval_us=config.decision_interval_us,
        requested_start_event_index=config.start_event_index,
        min_rows=(config.max_episode_steps + 1),
    )
    training_config = _build_training_config(config)

    assert linear_signals.metadata.start_event_index == 0
    assert linear_start.event_index == 1
    assert training_config.start_event_index == 1
