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
    _config_warnings,
    _debug_start_rows,
    _summary_config,
)
from mmrt.cli.execution_defaults import (
    DEFAULT_CANCEL_GUARD_TICKS,
    DEFAULT_CANCEL_LATENCY_US,
    DEFAULT_DECISION_COMPUTE_LATENCY_US,
    DEFAULT_DEFAULT_ORDER_QTY,
    DEFAULT_L2_DECREASE_WEIGHT,
    DEFAULT_MAKER_FEE_BPS,
    DEFAULT_MAX_DISTANCE_TICKS,
    DEFAULT_MAX_ORDER_QTY,
    DEFAULT_ORDER_ENTRY_LATENCY_US,
    DEFAULT_POST_ONLY_GAP_TICKS,
    DEFAULT_QTY_EPSILON,
    DEFAULT_TRADE_AT_LEVEL_WEIGHT,
    DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY,
)
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_SIGNALS_SCHEMA, AdverseSelectionSignalArtifact
from mmrt.execution.contracts import QueueModelMode
from mmrt.execution.split_contract import DecisionSplitRange
from mmrt.rl.reward_modes import TRAINING_REWARD_MODES
from mmrt.rl.rollout import TrainWindowSampler
from tests.grid_helpers import grid_lineage_fields


REQUIRED_TRAIN_ARGS = [
    "--tape-root", "/tmp/tape",
    "--decision-grid", "/tmp/tape/decision_grid",
    "--split-source-dataset-root", "/tmp/split-source",
    "--train-split", "train",
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
    assert config.training_reward_mode == "equity_delta"


def test_parser_default_env_config_is_colocated_balanced():
    parser = build_arg_parser()
    config = _config_from_args(parser.parse_args(REQUIRED_TRAIN_ARGS))
    env_config = _build_env_config(config)
    summary = _summary_config(config)
    queue = env_config.fill_simulator_config.queue_model

    assert config.cancel_guard_ticks == DEFAULT_CANCEL_GUARD_TICKS
    assert config.max_distance_ticks == DEFAULT_MAX_DISTANCE_TICKS
    assert config.max_order_qty == DEFAULT_MAX_ORDER_QTY
    assert config.default_order_qty == DEFAULT_DEFAULT_ORDER_QTY
    assert config.post_only_gap_ticks == DEFAULT_POST_ONLY_GAP_TICKS
    assert config.queue_mode == QueueModelMode.BALANCED
    assert config.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert config.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert config.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert config.maker_fee_bps == DEFAULT_MAKER_FEE_BPS
    assert config.decision_compute_latency_us == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert config.cancel_latency_us == DEFAULT_CANCEL_LATENCY_US
    assert env_config.cancel_guard_ticks == DEFAULT_CANCEL_GUARD_TICKS
    assert env_config.action_spec.max_distance_ticks == DEFAULT_MAX_DISTANCE_TICKS
    assert env_config.action_spec.max_order_qty == DEFAULT_MAX_ORDER_QTY
    assert env_config.observation_builder_config.inventory_qty_reference == DEFAULT_MAX_ORDER_QTY
    assert env_config.quote_geometry_config.default_order_qty == DEFAULT_DEFAULT_ORDER_QTY
    assert env_config.quote_geometry_config.post_only_gap_ticks == DEFAULT_POST_ONLY_GAP_TICKS
    assert queue.mode == QueueModelMode.BALANCED
    assert queue.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert queue.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert queue.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert queue.qty_epsilon == DEFAULT_QTY_EPSILON
    for key in (
        "cancel_guard_ticks",
        "max_distance_ticks",
        "max_order_qty",
        "default_order_qty",
        "post_only_gap_ticks",
        "l2_decrease_weight",
        "trade_at_level_weight",
        "unknown_level_queue_ahead_qty",
        "maker_fee_bps",
        "decision_compute_latency_us",
        "order_entry_latency_us",
        "cancel_latency_us",
    ):
        assert summary[key] == getattr(config, key)
    assert summary["queue_mode"] == "balanced"


def test_parser_accepts_train_window_sampling_mode():
    parser = build_arg_parser()
    args = parser.parse_args([*REQUIRED_TRAIN_ARGS, "--train-window-sampling", "cyclic_spread"])
    config = _config_from_args(args)

    assert config.train_window_sampling == "cyclic_spread"
    assert _summary_config(config)["train_window_sampling"] == "cyclic_spread"
    assert _build_training_config(config).train_window_sampling == "cyclic_spread"


def test_parser_accepts_every_training_reward_mode_and_summarizes_config():
    parser = build_arg_parser()
    for mode in TRAINING_REWARD_MODES:
        args = parser.parse_args([*REQUIRED_TRAIN_ARGS, "--training-reward-mode", mode])
        config = _config_from_args(args)
        training_config = _build_training_config(config)
        summary_config = _summary_config(config)

        assert config.training_reward_mode == mode
        assert training_config.rollout_config.reward_config.mode.value == mode
        assert summary_config["training_reward_mode"] == mode
        assert summary_config["reward_horizon_us"] == 1_000_000
        assert summary_config["horizon_path_weight"] == 1.0
        assert summary_config["fill_markout_weight"] == 0.25
        assert summary_config["horizon_potential_weight"] == 1.0
        assert summary_config["realized_lot_weight"] == 1.0
        assert summary_config["unrealized_horizon_weight"] == 1.0
        assert summary_config["multi_horizon_us"] == [250_000, 500_000, 1_000_000]
        assert summary_config["multi_horizon_weights"] == [0.25, 0.25, 1.0]


def test_parser_training_reward_config_overrides():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            *REQUIRED_TRAIN_ARGS,
            "--training-reward-mode",
            "multi_horizon_path",
            "--reward-horizon-us",
            "500000",
            "--horizon-path-weight",
            "2.0",
            "--fill-markout-weight",
            "0.5",
            "--horizon-potential-weight",
            "3.0",
            "--realized-lot-weight",
            "4.0",
            "--unrealized-horizon-weight",
            "5.0",
            "--multi-horizon-us",
            "100000,200000",
            "--multi-horizon-weights",
            "0.25,0.75",
        ]
    )
    config = _config_from_args(args)
    reward_config = _build_training_config(config).rollout_config.reward_config

    assert config.training_reward_mode == "multi_horizon_path"
    assert reward_config.reward_horizon_us == 500_000
    assert reward_config.horizon_path_weight == 2.0
    assert reward_config.fill_markout_weight == 0.5
    assert reward_config.horizon_potential_weight == 3.0
    assert reward_config.realized_lot_weight == 4.0
    assert reward_config.unrealized_horizon_weight == 5.0
    assert reward_config.multi_horizon_us == (100_000, 200_000)
    assert reward_config.multi_horizon_weights == (0.25, 0.75)


def test_parser_rejects_invalid_training_reward_config():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*REQUIRED_TRAIN_ARGS, "--training-reward-mode", "unknown"])

    cases = (
        ["--reward-horizon-us", "0"],
        ["--horizon-path-weight", "-1.0"],
        [
            "--training-reward-mode",
            "horizon_blend",
            "--horizon-path-weight",
            "0.0",
            "--fill-markout-weight",
            "0.0",
        ],
        [
            "--training-reward-mode",
            "realized_lot_horizon",
            "--realized-lot-weight",
            "0.0",
            "--unrealized-horizon-weight",
            "0.0",
            "--fill-markout-weight",
            "0.0",
        ],
        ["--multi-horizon-us", "500000,250000"],
        ["--multi-horizon-us", "250000,250000"],
        ["--multi-horizon-weights", "0.25,0.25"],
        ["--multi-horizon-weights", "0,0,0"],
    )
    for extra_args in cases:
        args = parser.parse_args([*REQUIRED_TRAIN_ARGS, *extra_args])
        with pytest.raises(ValueError):
            _config_from_args(args)


def test_parser_accepts_discount_mode_and_horizon():
    parser = build_arg_parser()
    args = parser.parse_args([
        *REQUIRED_TRAIN_ARGS,
        "--discount-mode",
        "time",
        "--discount-horizon-us",
        "1000000",
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.95",
    ])
    config = _config_from_args(args)
    training_config = _build_training_config(config)
    summary_config = _summary_config(config)

    assert config.discount_mode == "time"
    assert config.discount_horizon_us == 1_000_000
    assert training_config.rollout_config.discount_mode == "time"
    assert training_config.rollout_config.discount_horizon_us == 1_000_000
    assert summary_config["discount_mode"] == "time"
    assert summary_config["discount_horizon_us"] == 1_000_000


def test_parser_accepts_stdout_mode():
    parser = build_arg_parser()
    args = parser.parse_args([*REQUIRED_TRAIN_ARGS, "--stdout-mode", "none"])
    config = _config_from_args(args)

    assert config.stdout_mode == "none"
    assert _summary_config(config)["stdout_mode"] == "none"


def test_train_execution_ppo_rejects_invalid_discount_config():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*REQUIRED_TRAIN_ARGS, "--discount-mode", "calendar"])

    args = parser.parse_args([*REQUIRED_TRAIN_ARGS, "--discount-horizon-us", "0"])
    with pytest.raises(ValueError, match="discount_horizon_us"):
        _config_from_args(args)


def test_step_discount_warning_for_event_driven_schedule():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
        discount_mode="step",
        maker_fee_bps=0.0,
    )

    warnings = _config_warnings(
        config,
        decision_schedule={
            "min_decision_interval_us": 0,
            "max_decision_interval_us": 500_000,
            "wake_on_trade": True,
            "wake_on_top_of_book": True,
        },
    )

    assert any("physical PPO horizon vary with event density" in item for item in warnings)
    assert _config_warnings(
        config,
        decision_schedule={
            "min_decision_interval_us": 100_000,
            "max_decision_interval_us": 100_000,
            "wake_on_trade": False,
            "wake_on_top_of_book": False,
        },
    ) == []


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


def _adverse_label_config(
    queue_mode: str,
    *,
    qty_epsilon: float = DEFAULT_QTY_EPSILON,
    old_defaults: bool = False,
) -> dict[str, object]:
    if old_defaults:
        l2_decrease_weight = 0.25
        trade_at_level_weight = 0.5
        unknown_level_queue_ahead_qty = 1_000_000_000.0
        order_entry_latency_us = 500
    else:
        l2_decrease_weight = DEFAULT_L2_DECREASE_WEIGHT
        trade_at_level_weight = DEFAULT_TRADE_AT_LEVEL_WEIGHT
        unknown_level_queue_ahead_qty = DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
        order_entry_latency_us = DEFAULT_ORDER_ENTRY_LATENCY_US
    return {
        "queue_mode": queue_mode,
        "l2_decrease_weight": l2_decrease_weight,
        "trade_at_level_weight": trade_at_level_weight,
        "dedupe_l2_decrease_with_trade_prints": True,
        "unknown_level_queue_ahead_qty": unknown_level_queue_ahead_qty,
        "qty_epsilon": qty_epsilon,
        "order_entry_latency_us": order_entry_latency_us,
        "decision_compute_latency_us": DEFAULT_DECISION_COMPUTE_LATENCY_US,
        "post_only_gap_ticks": DEFAULT_POST_ONLY_GAP_TICKS,
        "order_qty": DEFAULT_DEFAULT_ORDER_QTY,
        "fill_horizon_us": 1_000_000,
        "adverse_horizon_us": 1_000_000,
    }


def _adverse_signals(
    queue_mode: str,
    *,
    qty_epsilon: float = DEFAULT_QTY_EPSILON,
    old_defaults: bool = False,
) -> AdverseSelectionSignalArtifact:
    return AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=np.array([100], dtype=np.int64),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_event_seq=np.array([0], dtype=np.int64),
        target_names=("bid_touch_filled",),
        predictions={"bid_touch_filled": np.array([0.5], dtype=np.float32)},
        adverse_label_config=_adverse_label_config(queue_mode, qty_epsilon=qty_epsilon, old_defaults=old_defaults),
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


def test_ppo_rejects_old_balanced_adverse_signals_for_new_default_env():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
    )
    env_config = _build_env_config(config)
    with pytest.raises(ValueError, match="adverse signal queue config mismatch"):
        _adverse_queue_config_compatibility(
            _adverse_signals("balanced", old_defaults=True),
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


def test_ppo_rejects_qty_epsilon_mismatch_unless_explicitly_allowed():
    config = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
        queue_mode="balanced",
    )
    env_config = _build_env_config(config)
    signals = _adverse_signals("balanced", qty_epsilon=1e-9)
    with pytest.raises(ValueError, match="adverse signal queue config mismatch"):
        _adverse_queue_config_compatibility(
            signals,
            env_config=env_config,
            allow_mismatch=False,
        )
    allowed = _adverse_queue_config_compatibility(
        signals,
        env_config=env_config,
        allow_mismatch=True,
    )
    assert allowed is not None
    assert allowed["status"] == "mismatch_allowed"
    assert "qty_epsilon" in allowed["mismatches"]


def test_parser_requires_current_split_source_and_uses_default_maker_fee():
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--tape-root", "/tmp/tape", "--decision-grid", "/tmp/tape/decision_grid"])
    assert excinfo.value.code == 2

    args = parser.parse_args(REQUIRED_TRAIN_ARGS)
    config = _config_from_args(args)
    assert config.maker_fee_bps == DEFAULT_MAKER_FEE_BPS


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
