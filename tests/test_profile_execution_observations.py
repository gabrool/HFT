import json

import pytest

from mmrt.cli.profile_execution_observations import (
    SAMPLE_POLICIES,
    ExecutionObservationProfileConfig,
    _config_from_args,
    _env_config_for_profile,
    _field_warnings,
    build_arg_parser,
    main,
    run_execution_observation_profile,
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
from mmrt.cli.train_execution_ppo import ExecutionPPOTrainCLIConfig, run_execution_ppo_training
from mmrt.execution.contracts import QueueModelMode
from mmrt.execution.linear_signal import LINEAR_SIGNALS_FILENAME
from mmrt.execution.obs_schema import CONTROL_FIELDS, default_observation_schema
from tests.test_ppo_tiny_env import _tiny_tape_root


def _required_args(tmp_path):
    tape_root = tmp_path / "tape"
    return [
        "--tape-root",
        str(tape_root),
        "--decision-grid",
        str(tape_root / "decision_grid"),
        "--split-source-dataset-root",
        str(tape_root / "split_source"),
        "--split",
        "train",
        "--linear-signals-npz",
        str(tape_root / LINEAR_SIGNALS_FILENAME),
        "--output-json",
        str(tmp_path / "profile.json"),
    ]


def _profile_config(tmp_path, **kwargs):
    tape_root = _tiny_tape_root(tmp_path)
    values = dict(
        tape_root=str(tape_root),
        decision_grid_path=str(tape_root / "decision_grid"),
        split_source_dataset_root=str(tape_root / "split_source"),
        split="train",
        linear_signals_npz=str(tape_root / LINEAR_SIGNALS_FILENAME),
        output_json=str(tmp_path / "profile.json"),
        sample_rows=4,
        num_envs=1,
        stdout_mode="none",
        overwrite=True,
    )
    values.update(kwargs)
    return ExecutionObservationProfileConfig(**values)


def _tiny_checkpoint(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            split_source_dataset_root=str(tape_root / "split_source"),
            train_split="train",
            output_json=str(tmp_path / "train_summary.json"),
            checkpoint_path=str(checkpoint_path),
            overwrite=True,
            num_updates=1,
            num_envs=1,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_sizes=(8,),
            max_episode_steps=4,
            seed=123,
        )
    )
    return tape_root, checkpoint_path


def test_parser_accepts_required_args_and_sample_policies(tmp_path):
    parser = build_arg_parser()
    for policy in SAMPLE_POLICIES:
        args = [*_required_args(tmp_path), "--sample-policy", policy]
        if policy.startswith("checkpoint_"):
            args.extend(["--checkpoint-path", str(tmp_path / "checkpoint.pt")])
        config = _config_from_args(parser.parse_args(args))
        assert config.sample_policy == policy


def test_checkpoint_policies_reject_missing_checkpoint(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args([*_required_args(tmp_path), "--sample-policy", "checkpoint_stochastic"])
    with pytest.raises(ValueError, match="checkpoint"):
        _config_from_args(args)


def test_tiny_env_profile_writes_json_with_sample_fit_stats_and_control_fields(tmp_path):
    summary = run_execution_observation_profile(_profile_config(tmp_path))
    payload = json.loads((tmp_path / "profile.json").read_text())

    assert payload == summary
    assert payload["run_type"] == "profile_execution_observations"
    assert payload["normalization"]["source"] == "sample_fit"
    assert payload["sample"]["sample_rows_collected"] > 0
    assert len(payload["field_stats"]) == payload["observation_schema"]["field_count"]
    assert "group_summary" in payload
    names = {item["name"] for item in payload["field_stats"]}
    assert set(CONTROL_FIELDS).issubset(names)
    for item in payload["field_stats"]:
        assert "raw" in item
        assert "normalized" in item


def test_no_checkpoint_env_defaults_are_colocated_balanced(tmp_path):
    config = _profile_config(tmp_path)
    env_config, env_raw, source = _env_config_for_profile(
        config=config,
        checkpoint=None,
        adverse_signals_path=None,
    )
    queue = env_config.fill_simulator_config.queue_model

    assert source == "profile_cli_config"
    assert env_raw["queue_mode"] == QueueModelMode.BALANCED
    assert env_config.cancel_guard_ticks == DEFAULT_CANCEL_GUARD_TICKS
    assert env_config.action_spec.max_distance_ticks == DEFAULT_MAX_DISTANCE_TICKS
    assert env_config.action_spec.max_order_qty == DEFAULT_MAX_ORDER_QTY
    assert env_config.quote_geometry_config.default_order_qty == DEFAULT_DEFAULT_ORDER_QTY
    assert env_config.quote_geometry_config.post_only_gap_ticks == DEFAULT_POST_ONLY_GAP_TICKS
    assert env_config.latency_config.decision_compute_latency_us == DEFAULT_DECISION_COMPUTE_LATENCY_US
    assert env_config.latency_config.order_entry_latency_us == DEFAULT_ORDER_ENTRY_LATENCY_US
    assert env_config.latency_config.cancel_latency_us == DEFAULT_CANCEL_LATENCY_US
    assert env_config.fill_simulator_config.maker_fee_bps == DEFAULT_MAKER_FEE_BPS
    assert queue.mode == QueueModelMode.BALANCED
    assert queue.l2_decrease_weight == DEFAULT_L2_DECREASE_WEIGHT
    assert queue.trade_at_level_weight == DEFAULT_TRADE_AT_LEVEL_WEIGHT
    assert queue.unknown_level_queue_ahead_qty == DEFAULT_UNKNOWN_LEVEL_QUEUE_AHEAD_QTY
    assert queue.qty_epsilon == DEFAULT_QTY_EPSILON


def test_checkpoint_env_source_uses_checkpoint_values_until_explicit_override(tmp_path):
    config = _profile_config(tmp_path, l2_decrease_weight=0.75)
    checkpoint = {
        "cli_config": {
            "cancel_guard_ticks": 2,
            "max_distance_ticks": 1,
            "max_order_qty": 0.001,
            "post_only_gap_ticks": 1,
            "default_order_qty": 0.001,
            "queue_mode": "conservative",
            "l2_decrease_weight": 0.25,
            "trade_at_level_weight": 0.5,
            "unknown_level_queue_ahead_qty": 1_000_000_000.0,
            "dedupe_l2_decrease_with_trade_prints": True,
            "maker_fee_bps": 0.0,
            "decision_compute_latency_us": 50,
            "order_entry_latency_us": 500,
            "cancel_latency_us": 500,
        },
        "observation_schema": default_observation_schema().as_dict(),
    }

    env_config, env_raw, source = _env_config_for_profile(
        config=config,
        checkpoint=checkpoint,
        adverse_signals_path=None,
    )
    queue = env_config.fill_simulator_config.queue_model

    assert source == "checkpoint_cli_config"
    assert env_raw["queue_mode"] == "conservative"
    assert env_raw["l2_decrease_weight"] == 0.75
    assert env_config.cancel_guard_ticks == 2
    assert env_config.action_spec.max_distance_ticks == 1
    assert env_config.action_spec.max_order_qty == 0.001
    assert env_config.latency_config.order_entry_latency_us == 500
    assert queue.mode == QueueModelMode.CONSERVATIVE
    assert queue.l2_decrease_weight == 0.75
    assert queue.trade_at_level_weight == 0.5
    assert queue.unknown_level_queue_ahead_qty == 1_000_000_000.0


def test_checkpoint_normalization_source_works_with_tiny_checkpoint(tmp_path):
    tape_root, checkpoint_path = _tiny_checkpoint(tmp_path)
    config = ExecutionObservationProfileConfig(
        tape_root=str(tape_root),
        decision_grid_path=str(tape_root / "decision_grid"),
        split_source_dataset_root=str(tape_root / "split_source"),
        split="train",
        linear_signals_npz=str(tape_root / LINEAR_SIGNALS_FILENAME),
        output_json=str(tmp_path / "checkpoint_profile.json"),
        checkpoint_path=str(checkpoint_path),
        sample_policy="checkpoint_deterministic",
        sample_rows=4,
        num_envs=1,
        stdout_mode="none",
        overwrite=True,
    )

    summary = run_execution_observation_profile(config)

    assert summary["normalization"]["source"] == "checkpoint"
    assert summary["lineage"]["checkpoint"]["has_observation_normalizer"] is True


def test_stdout_summary_is_concise_and_omits_large_payloads(tmp_path, capsys):
    tape_root = _tiny_tape_root(tmp_path)
    rc = main(
        [
            "--tape-root",
            str(tape_root),
            "--decision-grid",
            str(tape_root / "decision_grid"),
            "--split-source-dataset-root",
            str(tape_root / "split_source"),
            "--split",
            "train",
            "--linear-signals-npz",
            str(tape_root / LINEAR_SIGNALS_FILENAME),
            "--output-json",
            str(tmp_path / "profile_stdout.json"),
            "--sample-rows",
            "3",
            "--num-envs",
            "1",
            "--overwrite",
        ]
    )

    stdout = capsys.readouterr().out
    assert rc == 0
    assert "profile_execution_observations: ok" in stdout
    assert len(stdout.splitlines()) < 50
    for forbidden in (
        "field_stats",
        "policy_state_dict",
        "optimizer_state_dict",
        "ranges_by_split",
        "raw_observations",
        "normalized_observations",
    ):
        assert forbidden not in stdout


def test_field_warning_logic_catches_core_warning_types():
    warnings = _field_warnings(
        {
            "nonfinite_count": 1,
            "std": 0.0,
            "zero_fraction": 1.0,
            "abs_gt_100_fraction": 0.02,
        },
        {
            "nonfinite_count": 1,
            "std": 1.0,
            "clip_saturation_fraction": 0.02,
        },
    )

    assert "raw_nonfinite" in warnings
    assert "normalized_nonfinite" in warnings
    assert "near_constant" in warnings
    assert "mostly_zero" in warnings
    assert "high_raw_magnitude" in warnings
    assert "high_normalized_clip" in warnings
