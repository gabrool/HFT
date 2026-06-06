import pytest

from mmrt.cli.audit_execution_sim import ExecutionSimAuditConfig, _summary_config as audit_summary
from mmrt.cli.evaluate_execution_policy import (
    ExecutionPolicyEvaluationCLIConfig,
    _actor_critic_config_from_checkpoint,
    _env_config_from_cli_config,
    _env_config_from_training_cli_config,
    _summary_config as eval_summary,
)
from mmrt.cli.train_execution_ppo import ExecutionPPOTrainCLIConfig, _build_env_config as train_env_config, _summary_config as train_summary
from mmrt.execution.contracts import LatencyConfig, QueueModelMode
from mmrt.execution.queue_model import QueueModelConfig


def _assert_default_env(env_config):
    assert env_config.latency_config == LatencyConfig()
    assert env_config.fill_simulator_config.queue_model == QueueModelConfig()
    assert env_config.quote_geometry_config.post_only_gap_ticks == 1
    assert env_config.reward_config.reward_scale == 1.0
    assert env_config.fill_simulator_config.maker_fee_bps == -0.5


def test_execution_cli_defaults_match_core_conservative_latency_reward():
    audit = ExecutionSimAuditConfig(tape_root="/tmp/tape")
    train = ExecutionPPOTrainCLIConfig(tape_root="/tmp/tape")
    evaluate = ExecutionPolicyEvaluationCLIConfig(tape_root="/tmp/tape", checkpoint_path="/tmp/ckpt.pt")

    assert audit.queue_mode == QueueModelMode.CONSERVATIVE
    assert train.queue_mode == QueueModelMode.CONSERVATIVE
    assert evaluate.queue_mode == QueueModelMode.CONSERVATIVE
    assert audit.decision_compute_latency_us == 50
    assert audit.order_entry_latency_us == 500
    assert audit.cancel_latency_us == 500
    assert audit.reward_scale == 1.0
    assert audit.maker_fee_bps == -0.5
    assert audit.l2_decrease_weight == 0.25
    assert audit.trade_at_level_weight == 0.5
    assert audit.unknown_level_queue_ahead_qty == 1_000_000_000.0
    _assert_default_env(train_env_config(train))
    _assert_default_env(_env_config_from_cli_config(evaluate))

    for summary in (audit_summary(audit), train_summary(train), eval_summary(evaluate)):
        assert summary["post_only_gap_ticks"] == 1
        assert summary["decision_compute_latency_us"] == 50
        assert summary["order_entry_latency_us"] == 500
        assert summary["cancel_latency_us"] == 500
        assert summary["reward_scale"] == 1.0
        assert summary["maker_fee_bps"] == -0.5
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
