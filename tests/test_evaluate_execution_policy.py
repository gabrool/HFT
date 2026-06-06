import inspect
import json

import numpy as np

import pytest
import torch

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, LatencyConfig, SymbolSpec, TradePrint
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape, load_execution_tape, save_execution_tape
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    LINEAR_SIGNALS_FILENAME,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    predictions_to_signal_arrays,
    save_linear_signal_artifact_npz,
)
from mmrt.cli.train_execution_ppo import ExecutionPPOTrainCLIConfig, run_execution_ppo_training
from mmrt.cli.evaluate_execution_policy import (
    ExecutionPolicyEvaluationCLIConfig,
    main,
    run_execution_policy_evaluation,
)
from mmrt.rl.evaluate import PolicyEvaluationConfig, evaluate_policy
from mmrt.rl.normalization import ObservationNormalizer
from mmrt.rl.ppo import PPOConfig
from mmrt.rl.rollout import RolloutConfig
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork
from mmrt.rl.train import PPOTrainingConfig, train_ppo_policy


def _spec() -> SymbolSpec:
    return SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=0.0,
    )


def _l2(
    *,
    seq: int,
    local_ts_us: int,
    bid_ticks=(1000, 999),
    bid_sizes=(1.0, 2.0),
    ask_ticks=(1002, 1003),
    ask_sizes=(1.0, 2.0),
) -> ReconstructedL2Event:
    top = BookTop(
        local_ts_us=local_ts_us,
        best_bid_tick=bid_ticks[0],
        best_ask_tick=ask_ticks[0],
        best_bid_size=bid_sizes[0],
        best_ask_size=ask_sizes[0],
    )
    snapshot = BookLevelSnapshot(
        local_ts_us=local_ts_us,
        bid_ticks=tuple(bid_ticks),
        bid_sizes=tuple(bid_sizes),
        ask_ticks=tuple(ask_ticks),
        ask_sizes=tuple(ask_sizes),
    )
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us - 10,
        max_ts_us=local_ts_us - 5,
        num_updates=1,
        is_snapshot_batch=(seq == 0),
        book_top=top,
        bid_depth=len(bid_ticks),
        ask_depth=len(ask_ticks),
        book_snapshot=snapshot,
    )


def _trade(
    *,
    local_ts_us: int,
    side: AggressorSide,
    price_tick: int,
    amount: float,
    source_row: int,
) -> TradePrint:
    return TradePrint(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us - 1,
        side=side,
        price_tick=price_tick,
        amount=amount,
        trade_id=str(source_row),
        source_row=source_row,
    )


def _tape(l2_events, trades):
    plan = merge_execution_events(l2_events, trades)
    return build_execution_tape(
        symbol_spec=_spec(),
        l2_events=l2_events,
        trades=trades,
        merged_events=plan.events,
        book_depth=2,
    )


def _save_tape(tmp_path, tape):
    root = tmp_path / "execution_tape"
    save_execution_tape(tape, root, overwrite=True)
    return root


def _tiny_events():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
        _l2(seq=3, local_ts_us=400),
        _l2(seq=4, local_ts_us=500),
    ]
    trades = [
        _trade(
            local_ts_us=250,
            side=AggressorSide.SELL,
            price_tick=1000,
            amount=1.0,
            source_row=0,
        )
    ]
    return l2_events, trades


def _signal_arrays(n_rows: int = 16):
    return predictions_to_signal_arrays({
        NO_MOVE_PROBA_KEY: np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (n_rows, 1)),
        DIRECTION_PROBA_KEY: np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (n_rows, 1)),
        MAGNITUDE_UP_KEY: np.full(n_rows, np.log1p(10.0), dtype=np.float32),
        MAGNITUDE_DOWN_KEY: np.full(n_rows, np.log1p(5.0), dtype=np.float32),
    })


def _linear_artifact_for_tape(tape, n_rows: int = 16, *, decision_interval_us: int = 50, start_event_index: int = 0):
    arrays = _signal_arrays(n_rows)
    pairs = []
    for event_index, event in enumerate(tape.arrays.events):
        if event_index < start_event_index:
            continue
        if int(event["event_type_code"]) != 1:
            continue
        book_ptr = int(event["book_ptr"])
        if book_ptr >= 0:
            pairs.append((event_index, int(tape.arrays.l2_events[book_ptr]["local_ts_us"])))
    if not pairs:
        pairs.append((start_event_index, int(tape.manifest.start_local_ts_us)))
    decision_event_index = [pair[0] for pair in pairs[:n_rows]]
    decision_local_ts_us = [pair[1] for pair in pairs[:n_rows]]
    while len(decision_event_index) < n_rows:
        decision_event_index.append(decision_event_index[-1] + 1)
        decision_local_ts_us.append(decision_local_ts_us[-1] + decision_interval_us)
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_interval_us=decision_interval_us,
        start_event_index=start_event_index,
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.asarray(decision_event_index, dtype=np.int64),
        decision_local_ts_us=np.asarray(decision_local_ts_us, dtype=np.int64),
    )


def _save_linear_signals(root, n_rows: int = 16, *, decision_interval_us: int = 50, start_event_index: int = 0):
    tape = load_execution_tape(root)
    artifact = _linear_artifact_for_tape(
        tape, n_rows=n_rows, decision_interval_us=decision_interval_us, start_event_index=start_event_index
    )
    path = root / LINEAR_SIGNALS_FILENAME
    save_linear_signal_artifact_npz(path, artifact, overwrite=True)
    return path


def _tiny_tape():
    l2_events, trades = _tiny_events()
    return _tape(l2_events, trades)


def _tiny_env(max_episode_steps: int | None = 4) -> ExecutionEnv:
    tape = _tiny_tape()
    return ExecutionEnv(
        tape,
        linear_signals=_linear_artifact_for_tape(tape, decision_interval_us=50),
        config=ExecutionEnvConfig(
            decision_interval_us=50,
            latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0, cancel_latency_us=0),
            max_episode_steps=max_episode_steps,
        ),
    )


def _tiny_tape_root(tmp_path):
    root = _save_tape(tmp_path, _tiny_tape())
    _save_linear_signals(root)
    return root


def _train_tiny_checkpoint(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            output_json=str(tmp_path / "train_summary.json"),
            checkpoint_path=str(checkpoint_path),
            overwrite=True,
            num_updates=1,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_sizes=(8,),
            decision_interval_us=50,
            max_episode_steps=4,
            seed=123,
        )
    )
    return tape_root, checkpoint_path


def test_evaluate_policy_runs_tiny_env_and_preserves_policy_mode():
    env = _tiny_env(max_episode_steps=4)
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))
    policy.train()

    result = evaluate_policy(
        env,
        policy,
        config=PolicyEvaluationConfig(max_steps=4, deterministic=True),
    )

    assert result.steps > 0
    assert result.steps <= 4
    assert result.status in ("ok", "warning", "error")
    assert result.metrics["steps"]["count"] == result.steps
    assert result.diagnostics is not None
    assert policy.training is True
    payload = result.as_dict()
    assert payload["steps"] == result.steps
    assert "metrics" in payload
    assert "diagnostics" in payload


def test_evaluate_policy_uses_observation_normalizer_read_only():
    env = _tiny_env(max_episode_steps=4)
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))
    normalizer = ObservationNormalizer(obs_shape=obs_dim)
    before = normalizer.running.count.clone()

    evaluate_policy(
        env,
        policy,
        config=PolicyEvaluationConfig(max_steps=2),
        observation_normalizer=normalizer,
    )

    after = normalizer.running.count
    assert torch.equal(before, after)


def test_run_execution_policy_evaluation_from_checkpoint(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    train_summary = run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            output_json=str(tmp_path / "train_summary.json"),
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            overwrite=True,
            num_updates=1,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_sizes=(8,),
            decision_interval_us=50,
            max_episode_steps=4,
            seed=123,
        )
    )
    assert train_summary["checkpoint_saved"] is True

    output_json = tmp_path / "eval_summary.json"
    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            output_json=str(output_json),
            overwrite=True,
            max_steps=4,
            start_event_index=0,
        )
    )

    assert output_json.exists()
    assert json.loads(output_json.read_text()) == summary
    assert summary["status"] in ("ok", "warning", "error")
    assert summary["run_type"] == "evaluate_execution_policy"
    assert (
        summary["checkpoint"]["schema"]
        == "mmrt_execution_ppo_checkpoint"
    )
    assert summary["checkpoint"]["updates_completed"] == 1
    assert summary["checkpoint"]["has_observation_normalizer"] is True
    assert summary["env_config_source"] == "checkpoint_cli_config"
    assert summary["effective_start_event_index"] == 0
    assert summary["evaluation"]["steps"] > 0
    assert summary["evaluation"]["steps"] <= 4
    assert summary["evaluation"]["metrics"]["steps"]["count"] == summary["evaluation"]["steps"]
    assert summary["tape"]["symbol"] == "BTCUSDT"
    assert (
        summary["linear_signals"]["schema"]
        == "mmrt_execution_linear_signals_aligned"
    )
    assert summary["linear_signals"]["n_rows"] >= 1
    assert summary["linear_signals"]["fields"] == train_summary["linear_signals"]["fields"]
    assert summary["observation_schema"] == train_summary["observation_schema"]


def test_evaluate_execution_policy_requires_linear_signals_file(tmp_path):
    _, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    eval_root = _save_tape(tmp_path / "eval", _tiny_tape())
    with pytest.raises(FileNotFoundError):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(eval_root),
                checkpoint_path=str(checkpoint_path),
                output_json=str(tmp_path / "eval_summary.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_rejects_observation_schema_mismatch(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["observation_schema"] = dict(checkpoint["observation_schema"])
    checkpoint["observation_schema"]["field_names"] = checkpoint["observation_schema"][
        "field_names"
    ][:-1]
    bad_checkpoint = tmp_path / "bad_checkpoint.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="observation_schema"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(bad_checkpoint),
                output_json=str(tmp_path / "eval_bad_summary.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_rejects_checkpoint_missing_linear_signal_metadata(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("linear_signals", None)

    bad_checkpoint = tmp_path / "missing_linear_metadata.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="linear_signals"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(bad_checkpoint),
                output_json=str(tmp_path / "eval_missing_linear.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_rejects_linear_signal_field_mismatch(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["linear_signals"] = dict(checkpoint["linear_signals"])
    checkpoint["linear_signals"]["fields"] = checkpoint["linear_signals"]["fields"][:-1]

    bad_checkpoint = tmp_path / "bad_linear_fields.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="linear signal fields"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(bad_checkpoint),
                output_json=str(tmp_path / "eval_bad_linear_fields.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_rejects_misaligned_linear_signal_metadata(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    tape = load_execution_tape(tape_root)
    artifact = _linear_artifact_for_tape(tape, decision_interval_us=999)
    save_linear_signal_artifact_npz(tape_root / LINEAR_SIGNALS_FILENAME, artifact, overwrite=True)

    with pytest.raises(ValueError, match="linear signal metadata mismatch"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(checkpoint_path),
                output_json=str(tmp_path / "eval_mismatch.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_requires_checkpoint_cli_config_by_default(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("cli_config", None)
    bad_checkpoint = tmp_path / "missing_cli_config.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="cli_config"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(bad_checkpoint),
                output_json=str(tmp_path / "eval_missing_cli_config.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_can_explicitly_ignore_missing_checkpoint_cli_config(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint.pop("cli_config", None)
    bad_checkpoint = tmp_path / "missing_cli_config_allowed.pt"
    torch.save(checkpoint, bad_checkpoint)

    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            checkpoint_path=str(bad_checkpoint),
            output_json=str(tmp_path / "eval_cli_config.json"),
            overwrite=True,
            max_steps=4,
            decision_interval_us=50,
            max_episode_steps=4,
            use_checkpoint_cli_env_config=False,
        )
    )
    assert summary["env_config_source"] == "evaluation_cli_config"
    assert summary["effective_start_event_index"] is None


def test_evaluate_execution_policy_inherits_checkpoint_start_event_index_when_unset(tmp_path):
    tape = _tape([_l2(seq=seq, local_ts_us=100 + seq * 100) for seq in range(8)], [])
    tape_root = _save_tape(tmp_path, tape)

    _save_linear_signals(
        tape_root,
        decision_interval_us=50,
        start_event_index=1,
    )

    checkpoint_path = tmp_path / "checkpoint_start_1.pt"
    run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            output_json=str(tmp_path / "train_start_1_summary.json"),
            checkpoint_path=str(checkpoint_path),
            overwrite=True,
            num_updates=1,
            rollout_steps=3,
            update_epochs=1,
            minibatch_size=1,
            hidden_sizes=(8,),
            decision_interval_us=50,
            max_episode_steps=4,
            start_event_index=1,
            seed=123,
        )
    )

    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            checkpoint_path=str(checkpoint_path),
            output_json=str(tmp_path / "eval_start_1_summary.json"),
            overwrite=True,
            max_steps=3,
        )
    )

    assert summary["env_config_source"] == "checkpoint_cli_config"
    assert summary["effective_start_event_index"] == 1
    assert summary["linear_signals"]["metadata"]["start_event_index"] == 1
    assert summary["evaluation"]["steps"] > 0


def test_evaluate_execution_policy_explicit_start_event_index_override_must_match_signal_metadata(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="linear signal metadata mismatch"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(checkpoint_path),
                output_json=str(tmp_path / "eval_bad_start_override.json"),
                overwrite=True,
                max_steps=3,
                start_event_index=1,
            )
        )


def test_evaluate_execution_policy_main_writes_summary_and_prints_json(tmp_path, capsys):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    output_json = tmp_path / "eval_summary.json"

    rc = main(
        [
            "--tape-root",
            str(tape_root),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-json",
            str(output_json),
            "--max-steps",
            "4",
            "--start-event-index",
            "0",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert output_json.exists()
    stdout_payload = json.loads(capsys.readouterr().out)
    disk_payload = json.loads(output_json.read_text())
    assert stdout_payload == disk_payload
    assert stdout_payload["run_type"] == "evaluate_execution_policy"
    assert stdout_payload["evaluation"]["steps"] > 0
    assert (
        stdout_payload["linear_signals"]["schema"]
        == "mmrt_execution_linear_signals_aligned"
    )
    assert "observation_schema" in stdout_payload


def test_evaluate_execution_policy_refuses_overwrite_without_flag(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    output_json = tmp_path / "eval_summary.json"
    output_json.write_text("{}")

    with pytest.raises(FileExistsError):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                checkpoint_path=str(checkpoint_path),
                output_json=str(output_json),
            )
        )


def test_evaluate_execution_policy_config_parses_dtype_and_queue_mode():
    cfg = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        checkpoint_path="/tmp/checkpoint.pt",
        dtype="float64",
        queue_mode="balanced",
    )
    assert cfg.dtype is torch.float64
    assert cfg.queue_mode.value == "balanced"

    with pytest.raises(ValueError):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            checkpoint_path="/tmp/checkpoint.pt",
            dtype="float16",
        )


def test_evaluate_execution_policy_accepts_zero_queue_weights():
    cfg = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        checkpoint_path="/tmp/checkpoint.pt",
        l2_decrease_weight=0.0,
        trade_at_level_weight=0.0,
    )

    assert cfg.l2_decrease_weight == 0.0
    assert cfg.trade_at_level_weight == 0.0


def test_evaluate_execution_policy_rejects_queue_weights_above_one():
    with pytest.raises(ValueError):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            checkpoint_path="/tmp/checkpoint.pt",
            l2_decrease_weight=1.1,
        )

    with pytest.raises(ValueError):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            checkpoint_path="/tmp/checkpoint.pt",
            trade_at_level_weight=1.1,
        )


def test_evaluate_execution_policy_can_use_cli_env_config_instead_of_checkpoint(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)

    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            checkpoint_path=str(checkpoint_path),
            output_json=str(tmp_path / "eval_cli_env.json"),
            overwrite=True,
            use_checkpoint_cli_env_config=False,
            decision_interval_us=50,
            max_episode_steps=4,
            max_steps=4,
        )
    )

    assert summary["env_config_source"] == "evaluation_cli_config"
    assert summary["effective_start_event_index"] is None


def test_evaluate_modules_do_not_import_forbidden_layers():
    import mmrt.rl.evaluate as evaluate
    import mmrt.cli.evaluate_execution_policy as cli

    eval_source = inspect.getsource(evaluate)
    cli_source = inspect.getsource(cli)

    for text in (
        "argparse",
        "json",
        "pathlib",
        "load_execution_tape",
        "torch.load",
        "torch.save",
        "pandas",
        "polars",
        "pyarrow",
        "sklearn",
        "gym",
        "gymnasium",
        "mmrt.storage",
        "mmrt.linear",
    ):
        assert text not in eval_source

    for text in (
        "pandas",
        "polars",
        "pyarrow",
        "sklearn",
        "gym",
        "gymnasium",
        "mmrt.storage",
        "mmrt.linear",
        "build_execution_tape",
        "train_ppo_policy",
        "update_ppo",
    ):
        assert text not in cli_source
