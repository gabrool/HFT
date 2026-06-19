import json

import pytest

from mmrt.cli.profile_execution_observations import (
    SAMPLE_POLICIES,
    ExecutionObservationProfileConfig,
    _config_from_args,
    _field_warnings,
    build_arg_parser,
    main,
    run_execution_observation_profile,
)
from mmrt.cli.train_execution_ppo import ExecutionPPOTrainCLIConfig, run_execution_ppo_training
from mmrt.execution.linear_signal import LINEAR_SIGNALS_FILENAME
from mmrt.execution.obs_schema import CONTROL_FIELDS
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
