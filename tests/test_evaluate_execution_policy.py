import inspect
import json

import pytest
import torch

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape, save_execution_tape
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
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


def _tiny_tape():
    l2_events, trades = _tiny_events()
    return _tape(l2_events, trades)


def _tiny_env(max_episode_steps: int | None = 4) -> ExecutionEnv:
    return ExecutionEnv(
        _tiny_tape(),
        config=ExecutionEnvConfig(
            decision_interval_us=50,
            max_episode_steps=max_episode_steps,
        ),
    )


def _tiny_tape_root(tmp_path):
    return _save_tape(tmp_path, _tiny_tape())


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
    assert summary["checkpoint"]["schema_version"] == "mmrt_execution_ppo_checkpoint_v1"
    assert summary["checkpoint"]["updates_completed"] == 1
    assert summary["checkpoint"]["has_observation_normalizer"] is True
    assert summary["env_config_source"] == "checkpoint_cli_config"
    assert summary["evaluation"]["steps"] > 0
    assert summary["evaluation"]["steps"] <= 4
    assert summary["evaluation"]["metrics"]["steps"]["count"] == summary["evaluation"]["steps"]
    assert summary["tape"]["symbol"] == "BTCUSDT"


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
