import pytest

from mmrt.cli.audit_execution_sim import ExecutionSimAuditConfig, build_arg_parser as audit_arg_parser, _summary_config as audit_summary
from mmrt.cli.execution_defaults import (
    DEFAULT_CANCEL_GUARD_TICKS,
    DEFAULT_CANCEL_LATENCY_US,
    DEFAULT_DECISION_COMPUTE_LATENCY_US,
    DEFAULT_DEFAULT_ORDER_QTY,
    DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS,
    DEFAULT_L2_DECREASE_WEIGHT,
    DEFAULT_MAKER_FEE_BPS,
    DEFAULT_MAX_DISTANCE_TICKS,
    DEFAULT_MAX_ORDER_QTY,
    DEFAULT_ORDER_ENTRY_LATENCY_US,
    DEFAULT_POST_ONLY_GAP_TICKS,
    DEFAULT_QTY_EPSILON,
    DEFAULT_QUEUE_MODE,
    DEFAULT_TRADE_AT_LEVEL_WEIGHT,
    DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY,
)
from mmrt.cli.execution_env_config import (
    ExecutionEnvConfigBuildInput,
    build_execution_env_config_from_input,
)
from mmrt.cli.evaluate_execution_policy import (
    ExecutionPolicyEvaluationCLIConfig,
    build_arg_parser as eval_arg_parser,
    _actor_critic_config_from_checkpoint,
    _env_config_from_cli_config,
    _env_config_from_training_cli_config,
    _summary_config as eval_summary,
)
from mmrt.cli.train_execution_ppo import ExecutionPPOTrainCLIConfig, build_arg_parser as train_arg_parser, _build_env_config as train_env_config, _summary_config as train_summary
from mmrt.execution.contracts import LatencyConfig, QueueModelMode


def assert_execution_defaults(config):
    assert config.cancel_guard_ticks == DEFAULT_CANCEL_GUARD_TICKS
    assert config.max_distance_ticks == DEFAULT_MAX_DISTANCE_TICKS
    assert config.max_order_qty == DEFAULT_MAX_ORDER_QTY
    assert config.default_order_qty == DEFAULT_DEFAULT_ORDER_QTY
    assert config.post_only_gap_ticks == DEFAULT_POST_ONLY_GAP_TICKS
    assert config.queue_mode == DEFAULT_QUEUE_MODE
    assert config.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert config.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert config.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert config.dedupe_l2_decrease_with_trade_prints is DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS
    assert config.maker_fee_bps == DEFAULT_MAKER_FEE_BPS
    assert config.decision_compute_latency_us == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert config.cancel_latency_us == DEFAULT_CANCEL_LATENCY_US
    assert config.reward_scale == 1.0


def _assert_default_env(env_config):
    queue_model = env_config.fill_simulator_config.queue_model
    assert env_config.cancel_guard_ticks == DEFAULT_CANCEL_GUARD_TICKS
    assert env_config.action_spec.max_distance_ticks == DEFAULT_MAX_DISTANCE_TICKS
    assert env_config.action_spec.max_order_qty == DEFAULT_MAX_ORDER_QTY
    assert env_config.quote_geometry_config.default_order_qty == DEFAULT_DEFAULT_ORDER_QTY
    assert env_config.quote_geometry_config.post_only_gap_ticks == DEFAULT_POST_ONLY_GAP_TICKS
    assert env_config.latency_config.decision_compute_latency_us == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert env_config.latency_config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert env_config.latency_config.cancel_latency_us == DEFAULT_CANCEL_LATENCY_US
    assert queue_model.mode == QueueModelMode.BALANCED
    assert queue_model.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert queue_model.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert queue_model.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert queue_model.dedupe_l2_decrease_with_trade_prints is True
    assert env_config.fill_simulator_config.qty_epsilon == DEFAULT_QTY_EPSILON
    assert env_config.reward_config.reward_scale == 1.0
    assert env_config.fill_simulator_config.maker_fee_bps == DEFAULT_MAKER_FEE_BPS
    assert env_config.observation_builder_config.inventory_qty_reference == DEFAULT_MAX_ORDER_QTY
    assert (
        env_config.observation_builder_config.inventory_qty_reference
        == env_config.action_spec.max_order_qty
    )


def test_shared_colocated_balanced_defaults_are_exact():
    assert DEFAULT_CANCEL_GUARD_TICKS == 1
    assert DEFAULT_MAX_DISTANCE_TICKS == 2
    assert DEFAULT_DEFAULT_ORDER_QTY == 0.001
    assert DEFAULT_MAX_ORDER_QTY == 0.003
    assert DEFAULT_POST_ONLY_GAP_TICKS == 1
    assert DEFAULT_QUEUE_MODE == QueueModelMode.BALANCED
    assert DEFAULT_L2_DECREASE_WEIGHT == 0.5
    assert DEFAULT_TRADE_AT_LEVEL_WEIGHT == 1.0
    assert DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY == 0.0
    assert DEFAULT_DEDUPE_L2_DECREASE_WITH_TRADE_PRINTS is True
    assert DEFAULT_DECISION_COMPUTE_LATENCY_US == 50
    assert DEFAULT_ORDER_ENTRY_LATENCY_US == 250
    assert DEFAULT_CANCEL_LATENCY_US == 250
    assert DEFAULT_MAKER_FEE_BPS == -0.5
    assert DEFAULT_QTY_EPSILON == 1e-12


def test_execution_env_config_build_input_defaults_are_colocated_balanced():
    env_config = build_execution_env_config_from_input(ExecutionEnvConfigBuildInput())
    _assert_default_env(env_config)


def test_execution_env_config_inventory_reference_tracks_max_order_qty():
    default_env = build_execution_env_config_from_input(ExecutionEnvConfigBuildInput())
    assert default_env.observation_builder_config.inventory_qty_reference == 0.003
    assert default_env.action_spec.max_order_qty == 0.003

    custom_env = build_execution_env_config_from_input(
        ExecutionEnvConfigBuildInput(max_order_qty=0.005)
    )
    assert custom_env.action_spec.max_order_qty == 0.005
    assert custom_env.observation_builder_config.inventory_qty_reference == 0.005


def test_execution_cli_defaults_match_colocated_balanced_latency_reward():
    audit = ExecutionSimAuditConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/tape/decision_grid")
    train = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        split_source_dataset_root="/tmp/split-source",
    )
    evaluate = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        checkpoint_path="/tmp/ckpt.pt",
        split_source_dataset_root="/tmp/split-source",
        eval_split="val",
    )

    assert audit.queue_mode == QueueModelMode.BALANCED
    assert train.queue_mode == QueueModelMode.BALANCED
    assert evaluate.queue_mode == QueueModelMode.BALANCED
    assert_execution_defaults(audit)
    assert_execution_defaults(train)
    assert_execution_defaults(evaluate)
    train_env = train_env_config(train)
    eval_env = _env_config_from_cli_config(evaluate)
    _assert_default_env(train_env)
    _assert_default_env(eval_env)
    assert train_env.latency_config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert train_env.fill_simulator_config.queue_model.mode.value == "balanced"
    assert train_env.reward_config.reward_scale == 1.0

    for summary in (audit_summary(audit), train_summary(train), eval_summary(evaluate)):
        assert summary["cancel_guard_ticks"] == DEFAULT_CANCEL_GUARD_TICKS
        assert summary["max_distance_ticks"] == DEFAULT_MAX_DISTANCE_TICKS
        assert summary["max_order_qty"] == DEFAULT_MAX_ORDER_QTY
        assert summary["default_order_qty"] == DEFAULT_DEFAULT_ORDER_QTY
        assert summary["post_only_gap_ticks"] == DEFAULT_POST_ONLY_GAP_TICKS
        assert summary["queue_mode"] == "balanced"
        assert summary["l2_decrease_weight"] == DEFAULT_L2_DECREASE_WEIGHT
        assert summary["trade_at_level_weight"] == DEFAULT_TRADE_AT_LEVEL_WEIGHT
        assert summary["unknown_level_queue_ahead_qty"] == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
        assert summary["decision_compute_latency_us"] == DEFAULT_DECISION_COMPUTE_LATENCY_US
        assert summary["order_entry_latency_us"] == DEFAULT_ORDER_ENTRY_LATENCY_US
        assert summary["cancel_latency_us"] == DEFAULT_CANCEL_LATENCY_US
        assert summary["reward_scale"] == 1.0
        assert summary["maker_fee_bps"] == DEFAULT_MAKER_FEE_BPS
        assert "min" + "_distance_ticks" not in summary


def test_evaluation_checkpoint_env_reconstruction_preserves_new_fields():
    env_config = _env_config_from_training_cli_config(
        {
            "queue_mode": "balanced",
            "l2_decrease_weight": 1.0,
            "trade_at_level_weight": 1.0,
            "unknown_level_queue_ahead_qty": 0.0,
            "post_only_gap_ticks": 2,
            "maker_fee_bps": -0.5,
            "decision_compute_latency_us": 7,
            "order_entry_latency_us": 11,
            "cancel_latency_us": 13,
            "reward_scale": 2.5,
        }
    )
    assert env_config.quote_geometry_config.post_only_gap_ticks == 2
    assert env_config.latency_config == LatencyConfig(7, 11, 13)
    assert env_config.reward_config.reward_scale == 2.5
    assert env_config.fill_simulator_config.maker_fee_bps == -0.5
    assert env_config.fill_simulator_config.queue_model.mode == QueueModelMode.BALANCED


def test_checkpoint_network_config_rejects_stale_keys_and_accepts_fresh_schema():
    fresh = {
        "network_config": {
            "hidden_sizes": [8],
            "activation": "tanh",
            "layer_norm": False,
            "orthogonal_init": True,
            "enable_threshold": 0.5,
            "enable_logit_bias_init": 0.0,
            "continuous_log_std_init": -0.5,
            "continuous_log_std_min": -5.0,
            "continuous_log_std_max": 2.0,
            "policy_head_gain": 0.01,
            "value_head_gain": 1.0,
        }
    }
    assert _actor_critic_config_from_checkpoint(fresh).enable_threshold == 0.5

    stale = {"network_config": dict(fresh["network_config"], **{"policy" + "_log_std_init": -0.5})}
    with pytest.raises(ValueError, match="unsupported stale"):
        _actor_critic_config_from_checkpoint(stale)


def test_removed_min_distance_ticks_flag_is_rejected_by_execution_parsers():
    train_required = [
        "--decision-grid", "/tmp/tape/decision_grid",
        "--split-source-dataset-root", "/tmp/split-source",
        "--train-split", "train",
    ]
    eval_required = [
        "--decision-grid", "/tmp/tape/decision_grid",
        "--checkpoint-path", "/tmp/c.pt",
        "--split-source-dataset-root", "/tmp/split-source",
        "--eval-split", "val",
    ]
    parser_args = (
        (audit_arg_parser(), ["--tape-root", "/tmp/tape", "--min-distance-ticks", "1"]),
        (train_arg_parser(), ["--tape-root", "/tmp/tape", *train_required, "--min-distance-ticks", "1"]),
        (eval_arg_parser(), ["--tape-root", "/tmp/tape", *eval_required, "--min-distance-ticks", "1"]),
    )
    for parser, args in parser_args:
        with pytest.raises(SystemExit):
            parser.parse_args(args)


def test_post_only_gap_ticks_flag_is_accepted_by_execution_parsers():
    grid_arg = ["--decision-grid", "/tmp/tape/decision_grid"]
    assert audit_arg_parser().parse_args(["--tape-root", "/tmp/tape", *grid_arg, "--post-only-gap-ticks", "2"]).post_only_gap_ticks == 2
    assert train_arg_parser().parse_args([
        "--tape-root", "/tmp/tape",
        *grid_arg,
        "--split-source-dataset-root", "/tmp/split-source",
        "--train-split", "train",
        "--post-only-gap-ticks", "2",
    ]).post_only_gap_ticks == 2
    assert eval_arg_parser().parse_args([
        "--tape-root", "/tmp/tape",
        *grid_arg,
        "--checkpoint-path", "/tmp/c.pt",
        "--split-source-dataset-root", "/tmp/split-source",
        "--eval-split", "val",
        "--post-only-gap-ticks", "2",
    ]).post_only_gap_ticks == 2
