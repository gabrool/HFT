from pathlib import Path

import numpy as np
import pytest

from mmrt.cli.train_execution_ppo import (
    ExecutionPPOTrainCLIConfig,
    _adverse_queue_config_compatibility,
    build_arg_parser,
    _build_env_config,
    _build_training_config,
    _config_from_args,
    _debug_start_rows,
    _summary_config,
)
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_SIGNALS_SCHEMA, AdverseSelectionSignalArtifact
from mmrt.execution.split_contract import DecisionSplitRange
from mmrt.rl.rollout import TrainWindowSampler
from tests.grid_helpers import grid_lineage_fields


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
    assert config.train_window_sampling == "stratified_random"


def test_parser_accepts_train_window_sampling_mode():
    parser = build_arg_parser()
    args = parser.parse_args([*REQUIRED_TRAIN_ARGS, "--train-window-sampling", "cyclic_spread"])
    config = _config_from_args(args)

    assert config.train_window_sampling == "cyclic_spread"
    assert _summary_config(config)["train_window_sampling"] == "cyclic_spread"
    assert _build_training_config(config).train_window_sampling == "cyclic_spread"


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


def _adverse_label_config(queue_mode: str) -> dict[str, object]:
    return {
        "queue_mode": queue_mode,
        "l2_decrease_weight": 0.25,
        "trade_at_level_weight": 0.5,
        "dedupe_l2_decrease_with_trade_prints": True,
        "unknown_level_queue_ahead_qty": 1_000_000_000.0,
        "order_entry_latency_us": 500,
        "decision_compute_latency_us": 50,
        "post_only_gap_ticks": 1,
        "order_qty": 0.001,
        "fill_horizon_us": 1_000_000,
        "adverse_horizon_us": 1_000_000,
    }


def _adverse_signals(queue_mode: str) -> AdverseSelectionSignalArtifact:
    return AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=np.array([100], dtype=np.int64),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_event_seq=np.array([0], dtype=np.int64),
        target_names=("bid_touch_filled",),
        predictions={"bid_touch_filled": np.array([0.5], dtype=np.float32)},
        adverse_label_config=_adverse_label_config(queue_mode),
        **grid_lineage_fields(n_rows=1),
    )


def test_ppo_rejects_conservative_adverse_signals_for_balanced_env():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
        queue_mode="balanced",
    )
    env_config = _build_env_config(config)
    with pytest.raises(ValueError, match="adverse signal queue config mismatch"):
        _adverse_queue_config_compatibility(
            _adverse_signals("conservative"),
            env_config=env_config,
            allow_mismatch=False,
        )


def test_ppo_accepts_balanced_adverse_signals_for_balanced_env():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
        queue_mode="balanced",
    )
    env_config = _build_env_config(config)
    result = _adverse_queue_config_compatibility(
        _adverse_signals("balanced"),
        env_config=env_config,
        allow_mismatch=False,
    )
    assert result is not None
    assert result["status"] == "match"


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


def test_stratified_train_window_sampler_uses_only_train_rows_and_is_reproducible():
    ranges = (
        DecisionSplitRange(
            role="train",
            segment_key="seg_000",
            start_decision_row=0,
            end_decision_row=4,
            start_local_ts_us=100,
            end_local_ts_us=401,
        ),
        DecisionSplitRange(
            role="train",
            segment_key="seg_001",
            start_decision_row=10,
            end_decision_row=14,
            start_local_ts_us=1000,
            end_local_ts_us=1401,
        ),
    )

    def _sample_rows() -> list[int]:
        sampler = TrainWindowSampler(ranges, mode="stratified_random", seed=123, num_envs=2)
        return [sampler.sample(env_index)[0] for _round in range(4) for env_index in range(2)]

    rows = _sample_rows()
    assert rows == _sample_rows()
    assert all((0 <= row <= 2) or (10 <= row <= 12) for row in rows)
    assert all(not (3 <= row < 10) and row < 13 for row in rows)


def test_num_envs_stratified_sampler_diversifies_starts_on_multi_range_fixture():
    ranges = (
        DecisionSplitRange(
            role="train",
            segment_key="seg_000",
            start_decision_row=0,
            end_decision_row=5,
            start_local_ts_us=100,
            end_local_ts_us=501,
        ),
        DecisionSplitRange(
            role="train",
            segment_key="seg_001",
            start_decision_row=20,
            end_decision_row=25,
            start_local_ts_us=2000,
            end_local_ts_us=2501,
        ),
    )
    sampler = TrainWindowSampler(ranges, mode="stratified_random", seed=7, num_envs=2)

    starts = [sampler.sample(env_index)[0] for env_index in range(2)]
    stats = sampler.stats_since(0)

    assert len(set(starts)) == 2
    assert stats["unique_train_ranges_visited"] == 2
    assert stats["train_row_count"] == 10


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
