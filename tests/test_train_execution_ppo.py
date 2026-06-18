from pathlib import Path

from mmrt.cli.train_execution_ppo import (
    ExecutionPPOTrainCLIConfig,
    build_arg_parser,
    _build_env_config,
    _build_training_config,
    _config_from_args,
    _debug_start_rows,
    _summary_config,
)
from mmrt.execution.split_contract import DecisionSplitRange


REQUIRED_TRAIN_ARGS = [
    "--tape-root", "/tmp/tape",
    "--decision-grid", "/tmp/tape/decision_grid",
    "--split-source-dataset-root", "/tmp/split-source",
    "--train-split", "train",
    "--maker-fee-bps", "0.0",
]


def test_parser_can_disable_l2_trade_dedupe():
    parser = build_arg_parser()
    args = parser.parse_args([*REQUIRED_TRAIN_ARGS, "--no-dedupe-l2-decrease-with-trade-prints"])
    config = _config_from_args(args)
    env_config = _build_env_config(config)
    assert config.dedupe_l2_decrease_with_trade_prints is False
    assert env_config.fill_simulator_config.queue_model.dedupe_l2_decrease_with_trade_prints is False
    assert _summary_config(config)["dedupe_l2_decrease_with_trade_prints"] is False


def test_parser_dedupe_l2_trade_default_enabled():
    parser = build_arg_parser()
    args = parser.parse_args(REQUIRED_TRAIN_ARGS)
    config = _config_from_args(args)
    assert config.dedupe_l2_decrease_with_trade_prints is True


def test_adverse_runtime_config_inherits_post_only_gap_from_ppo_config():
    parser = build_arg_parser()
    args = parser.parse_args([
        *REQUIRED_TRAIN_ARGS,
        "--adverse-signals-npz", "/tmp/adverse.npz",
        "--post-only-gap-ticks", "2",
    ])
    config = _config_from_args(args)
    env_config = _build_env_config(config)

    assert env_config.adverse_runtime_config is not None
    assert env_config.quote_geometry_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.executable_edge.maker_fee_bps == env_config.fill_simulator_config.maker_fee_bps


def test_parser_requires_current_split_source_and_maker_fee():
    parser = build_arg_parser()
    for missing_args in (
        ["--tape-root", "/tmp/tape", "--decision-grid", "/tmp/tape/decision_grid", "--maker-fee-bps", "0.0"],
        ["--tape-root", "/tmp/tape", "--decision-grid", "/tmp/tape/decision_grid", "--split-source-dataset-root", "/tmp/split"],
    ):
        try:
            parser.parse_args(missing_args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("parser accepted split-free or fee-implicit training args")


def test_train_execution_ppo_debug_start_row_stays_inside_train_split():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
        debug_start_decision_row=1,
        max_episode_steps=1,
        num_envs=2,
        rollout_steps=2,
        minibatch_size=2,
    )
    ranges = (
        DecisionSplitRange(
            role="train",
            segment_key="seg_000",
            start_decision_row=0,
            end_decision_row=3,
            start_local_ts_us=100,
            end_local_ts_us=301,
        ),
    )
    training_config = _build_training_config(config)
    start_rows = _debug_start_rows(config, ranges)

    assert start_rows == (1, 1)
    assert training_config.rollout_config.num_envs == 2
    assert _summary_config(config)["debug_start_decision_row"] == 1
    assert not hasattr(training_config, "start_event_index")


def test_train_execution_ppo_rejects_debug_start_row_outside_train_split():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
        debug_start_decision_row=3,
        rollout_steps=2,
        minibatch_size=2,
    )
    ranges = (
        DecisionSplitRange(
            role="train",
            segment_key="seg_000",
            start_decision_row=0,
            end_decision_row=3,
            start_local_ts_us=100,
            end_local_ts_us=301,
        ),
    )

    try:
        _debug_start_rows(config, ranges)
    except ValueError as exc:
        assert "selected train split" in str(exc)
    else:
        raise AssertionError("debug_start_decision_row escaped the train split")


def test_execution_ppo_source_guard_has_no_split_free_cli_path():
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "mmrt/cli/train_execution_ppo.py",
            "mmrt/cli/evaluate_execution_policy.py",
            "mmrt/rl/train.py",
            "mmrt/rl/rollout.py",
            "mmrt/rl/evaluate.py",
        )
    )
    for forbidden in (
        "--start-event-index",
        "use_checkpoint_cli_env_config",
        "train_fraction",
        "--train-fraction",
        "split_source_dataset_root: str | None",
    ):
        assert forbidden not in source
