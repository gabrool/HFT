import inspect
import json
from decimal import Decimal
from dataclasses import replace

import numpy as np

import pytest
import torch

from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, LatencyConfig, SymbolSpec, TradePrint
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.event_merge import merge_execution_events
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.execution.execution_tape import build_execution_tape, load_execution_tape, save_execution_tape
from mmrt.execution.decision_grid import save_decision_grid
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
from mmrt.execution.split_contract import DecisionSplitRange
from mmrt.cli.train_execution_ppo import ExecutionPPOTrainCLIConfig, run_execution_ppo_training
from mmrt.cli.evaluate_execution_policy import (
    ExecutionPolicyEvaluationCLIConfig,
    _adverse_queue_config_compatibility,
    _env_config_from_cli_config,
    main,
    run_execution_policy_evaluation,
)
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_SIGNALS_SCHEMA, AdverseSelectionSignalArtifact
from mmrt.rl.evaluate import PolicyEvaluationConfig, evaluate_policy
from mmrt.rl.normalization import ObservationNormalizer
from mmrt.rl.ppo import PPOConfig
from mmrt.rl.rollout import RolloutConfig
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork
from mmrt.rl.train import PPOTrainingConfig, train_ppo_policy
from tests.grid_helpers import decision_grid_for_tape, grid_lineage_fields, write_split_source_manifest



def _fixed_schedule_payload(stride_us: int) -> dict:
    return DecisionScheduleConfig(min_decision_interval_us=stride_us, max_decision_interval_us=stride_us).as_dict()


def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures", symbol="BTCUSDT", mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC", quote_asset="USDT", margin_asset="USDT", contract_type="PERPETUAL", status="TRADING",
        tick_size=Decimal("0.1"), min_price=Decimal("0.1"), max_price=Decimal("1000000"),
        step_size=Decimal("0.001"), min_qty=Decimal("0.001"), max_qty=Decimal("100"), min_notional=Decimal("0"),
        allowed_order_types=("LIMIT",), allowed_time_in_force=("GTC", "GTX"),
    )

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
        symbol_rules=_rules(),
        l2_events=l2_events,
        trades=trades,
        merged_events=plan.events,
        book_depth=2,
    )


def _save_tape(tmp_path, tape):
    root = tmp_path / "execution_tape"
    save_execution_tape(tape, root, overwrite=True)
    save_decision_grid(root / "decision_grid", decision_grid_for_tape(tape), overwrite=True)
    return root


def _tiny_events():
    l2_events = [
        _l2(seq=seq, local_ts_us=100 + seq * 100)
        for seq in range(8)
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


def _linear_artifact_for_tape(tape, n_rows: int = 16, *, decision_interval_us: int = 50, start_event_index: int = 0, decision_grid=None):
    if decision_grid is None:
        decision_grid = decision_grid_for_tape(tape)
    n_rows = decision_grid.n_rows
    arrays = _signal_arrays(n_rows)
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_grid_schema=decision_grid.metadata.schema,
        decision_grid_hash=decision_grid.decision_grid_hash,
        decision_grid_n_rows=decision_grid.n_rows,
        decision_schedule=decision_grid.decision_schedule,
        start_event_index=int(decision_grid.decision_event_index[0]),
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=decision_grid.decision_event_index.copy(),
        decision_local_ts_us=decision_grid.decision_local_ts_us.copy(),
        decision_event_seq=decision_grid.decision_event_seq.copy(),
    )


def _save_linear_signals(root, n_rows: int = 16, *, decision_interval_us: int = 50, start_event_index: int = 0):
    tape = load_execution_tape(root)
    decision_grid = decision_grid_for_tape(tape)
    artifact = _linear_artifact_for_tape(
        tape,
        n_rows=n_rows,
        decision_interval_us=decision_interval_us,
        start_event_index=start_event_index,
        decision_grid=decision_grid,
    )
    path = root / LINEAR_SIGNALS_FILENAME
    save_linear_signal_artifact_npz(path, artifact, overwrite=True)
    return path


def _tiny_tape():
    l2_events, trades = _tiny_events()
    return _tape(l2_events, trades)


def _tiny_env(max_episode_steps: int | None = 4) -> ExecutionEnv:
    tape = _tiny_tape()
    decision_grid = decision_grid_for_tape(tape)
    return ExecutionEnv(
        tape,
        decision_grid=decision_grid,
        linear_signals=_linear_artifact_for_tape(tape, decision_interval_us=50, decision_grid=decision_grid),
        config=ExecutionEnvConfig(
            latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0, cancel_latency_us=0),
            max_episode_steps=max_episode_steps,
        ),
    )


def _tiny_tape_root(tmp_path):
    tape = _tiny_tape()
    root = _save_tape(tmp_path, tape)
    _save_linear_signals(root)
    write_split_source_manifest(root / "split_source", decision_grid_for_tape(tape))
    return root


def _split_source_root(tape_root):
    return tape_root / "split_source"


def _train_tiny_checkpoint(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            split_source_dataset_root=str(_split_source_root(tape_root)),
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
    telemetry = payload["telemetry"]
    assert telemetry["sample_count"] == result.steps
    assert telemetry["policy_distribution"]["deterministic"] is True
    assert telemetry["policy_distribution"]["enable_prob"]["bid_enabled"]["count"] == result.steps
    assert "enable_logit" in telemetry["policy_distribution"]
    effective = telemetry["effective_quotes"]
    assert (
        effective["quote_no_quote_rate"]
        + effective["quote_bid_only_rate"]
        + effective["quote_ask_only_rate"]
        + effective["quote_two_sided_rate"]
    ) == pytest.approx(1.0)


def test_evaluate_policy_stochastic_includes_sampled_requested_telemetry_on_cpu():
    env = _tiny_env(max_episode_steps=4)
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))

    result = evaluate_policy(
        env,
        policy,
        config=PolicyEvaluationConfig(max_steps=4, deterministic=False, device="cpu"),
    )

    telemetry = result.as_dict()["telemetry"]
    requested = telemetry["requested_actions"]
    effective = telemetry["effective_quotes"]
    assert telemetry["sample_count"] == result.steps
    assert telemetry["policy_distribution"]["deterministic"] is False
    assert 0.0 <= requested["requested_bid_enabled_rate"] <= 1.0
    assert 0.0 <= requested["requested_ask_enabled_rate"] <= 1.0
    assert (
        effective["quote_no_quote_rate"]
        + effective["quote_bid_only_rate"]
        + effective["quote_ask_only_rate"]
        + effective["quote_two_sided_rate"]
    ) == pytest.approx(1.0)


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


def test_evaluate_policy_continues_after_max_episode_truncation_inside_split():
    env = _tiny_env(max_episode_steps=2)
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))
    result = evaluate_policy(
        env,
        policy,
        config=PolicyEvaluationConfig(
            decision_row_ranges=(
                DecisionSplitRange(
                    role="val",
                    segment_key="seg_000",
                    start_decision_row=2,
                    end_decision_row=7,
                    start_local_ts_us=300,
                    end_local_ts_us=701,
                ),
            ),
            deterministic=True,
        ),
    )

    payload = result.as_dict()
    eval_config = payload["config"]
    assert result.steps == 4
    assert eval_config["episode_count"] == 2
    assert eval_config["eval_requested_row_count"] == 4
    assert eval_config["eval_covered_row_count"] == 4
    assert eval_config["eval_coverage_fraction"] == 1.0
    assert eval_config["truncation_counts"]["max_episode_steps"] >= 1
    assert eval_config["evaluated_decision_row_ranges"][0]["start_decision_row"] == 2
    assert eval_config["evaluated_decision_row_ranges"][-1]["end_decision_row"] > 4


def test_evaluate_policy_does_not_cross_selected_split_boundary():
    env = _tiny_env(max_episode_steps=None)
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))
    result = evaluate_policy(
        env,
        policy,
        config=PolicyEvaluationConfig(
            decision_row_ranges=(
                DecisionSplitRange(
                    role="test",
                    segment_key="seg_000",
                    start_decision_row=4,
                    end_decision_row=6,
                    start_local_ts_us=500,
                    end_local_ts_us=601,
                ),
            ),
            deterministic=True,
        ),
    )

    eval_config = result.as_dict()["config"]
    assert result.steps == 1
    assert eval_config["eval_covered_row_count"] == 1
    assert eval_config["truncation_counts"]["split_exhausted"] == 1
    assert all(
        entry["start_decision_row"] >= 4 and entry["end_decision_row"] <= 6
        for entry in eval_config["evaluated_decision_row_ranges"]
    )


def test_evaluate_policy_max_steps_caps_split_coverage():
    env = _tiny_env(max_episode_steps=None)
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))
    result = evaluate_policy(
        env,
        policy,
        config=PolicyEvaluationConfig(
            max_steps=2,
            decision_row_ranges=(
                DecisionSplitRange(
                    role="val",
                    segment_key="seg_000",
                    start_decision_row=1,
                    end_decision_row=7,
                    start_local_ts_us=200,
                    end_local_ts_us=701,
                ),
            ),
            deterministic=True,
        ),
    )

    eval_config = result.as_dict()["config"]
    assert result.steps == 2
    assert eval_config["eval_requested_row_count"] == 5
    assert eval_config["eval_covered_row_count"] == 2
    assert eval_config["eval_coverage_fraction"] < 1.0
    assert eval_config["truncation_counts"]["max_steps"] == 1


def test_run_execution_policy_evaluation_from_checkpoint(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    train_summary = run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            split_source_dataset_root=str(_split_source_root(tape_root)),
            train_split="train",
            output_json=str(tmp_path / "train_summary.json"),
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
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
    assert train_summary["checkpoint_saved"] is True

    output_json = tmp_path / "eval_summary.json"
    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            split_source_dataset_root=str(_split_source_root(tape_root)),
            eval_split="val",
            output_json=str(output_json),
            overwrite=True,
            max_steps=4,
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
    assert summary["eval_split"] == "val"
    assert summary["checkpoint_split_source_matches_current"] is True
    assert summary["evaluated_start_decision_row"] >= 0
    assert summary["evaluation"]["steps"] > 0
    assert summary["evaluation"]["steps"] <= 4
    assert summary["evaluation"]["metrics"]["steps"]["count"] == summary["evaluation"]["steps"]
    assert summary["evaluation"]["telemetry"]["sample_count"] == summary["evaluation"]["steps"]
    assert summary["policy_action_telemetry"] == summary["evaluation"]["telemetry"]
    assert summary["tape"]["symbol"] == "BTCUSDT"
    assert (
        summary["linear_signals"]["schema"]
        == "mmrt_execution_linear_signals_grid_v1"
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
                decision_grid_path=str(eval_root / "decision_grid"),
                checkpoint_path=str(checkpoint_path),
                split_source_dataset_root=str(_split_source_root(eval_root)),
                eval_split="val",
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
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(bad_checkpoint),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
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
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(bad_checkpoint),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
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
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(bad_checkpoint),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
                output_json=str(tmp_path / "eval_bad_linear_fields.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_rejects_misaligned_linear_signal_metadata(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    tape = load_execution_tape(tape_root)
    artifact = _linear_artifact_for_tape(tape)
    bad_metadata = replace(artifact.metadata, decision_grid_hash="2" * 64)
    artifact = LinearSignalArtifact(
        arrays=artifact.arrays,
        metadata=bad_metadata,
        decision_event_index=artifact.decision_event_index,
        decision_local_ts_us=artifact.decision_local_ts_us,
        decision_event_seq=artifact.decision_event_seq,
    )
    save_linear_signal_artifact_npz(tape_root / LINEAR_SIGNALS_FILENAME, artifact, overwrite=True)

    with pytest.raises(ValueError, match="linear signal metadata mismatch"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(checkpoint_path),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
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
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(bad_checkpoint),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
                output_json=str(tmp_path / "eval_missing_cli_config.json"),
                overwrite=True,
                max_steps=4,
            )
        )


def test_evaluate_execution_policy_override_env_config_records_diff(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            checkpoint_path=str(checkpoint_path),
            split_source_dataset_root=str(_split_source_root(tape_root)),
            eval_split="val",
            output_json=str(tmp_path / "eval_cli_config.json"),
            overwrite=True,
            max_steps=4,
            max_episode_steps=3,
            override_env_config=True,
        )
    )
    assert summary["env_config_source"] == "evaluation_cli_config"
    assert summary["env_config_diff"]["max_episode_steps"]["override"] == 3


def test_evaluate_execution_policy_runs_explicit_test_split(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)

    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            checkpoint_path=str(checkpoint_path),
            split_source_dataset_root=str(_split_source_root(tape_root)),
            eval_split="test",
            output_json=str(tmp_path / "eval_test_summary.json"),
            overwrite=True,
            max_steps=3,
        )
    )

    assert summary["env_config_source"] == "checkpoint_cli_config"
    assert summary["eval_split"] == "test"
    assert summary["split_lineage"]["eval_split"] == "test"
    assert all(entry["role"] == "test" for entry in summary["evaluated_decision_row_ranges"])
    assert summary["evaluation"]["steps"] > 0


def test_evaluate_execution_policy_rejects_checkpoint_split_contract_mismatch(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["split_contract"] = dict(checkpoint["split_contract"])
    checkpoint["split_contract"]["split_source_dataset_id"] = "other-split-source"
    bad_checkpoint = tmp_path / "bad_split_checkpoint.pt"
    torch.save(checkpoint, bad_checkpoint)

    with pytest.raises(ValueError, match="checkpoint split_contract"):
        run_execution_policy_evaluation(
            ExecutionPolicyEvaluationCLIConfig(
                tape_root=str(tape_root),
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(bad_checkpoint),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
                output_json=str(tmp_path / "eval_bad_split.json"),
                overwrite=True,
                max_steps=3,
            )
        )


def test_evaluate_execution_policy_main_writes_summary_and_prints_json(tmp_path, capsys):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)
    output_json = tmp_path / "eval_summary.json"

    rc = main(
        [
            "--tape-root",
            str(tape_root),
            "--decision-grid",
            str(tape_root / "decision_grid"),
            "--checkpoint-path",
            str(checkpoint_path),
            "--split-source-dataset-root",
            str(_split_source_root(tape_root)),
            "--eval-split",
            "val",
            "--output-json",
            str(output_json),
            "--max-steps",
            "4",
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
        == "mmrt_execution_linear_signals_grid_v1"
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
                decision_grid_path=str(tape_root / "decision_grid"),
                checkpoint_path=str(checkpoint_path),
                split_source_dataset_root=str(_split_source_root(tape_root)),
                eval_split="val",
                output_json=str(output_json),
            )
        )


def test_evaluate_execution_policy_config_parses_dtype_and_queue_mode():
    cfg = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        checkpoint_path="/tmp/checkpoint.pt",
        split_source_dataset_root="/tmp/split-source",
        eval_split="val",
        dtype="float64",
        queue_mode="balanced",
    )
    assert cfg.dtype is torch.float64
    assert cfg.queue_mode.value == "balanced"

    with pytest.raises(ValueError):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            decision_grid_path="/tmp/tape/decision_grid",
            checkpoint_path="/tmp/checkpoint.pt",
            split_source_dataset_root="/tmp/split-source",
            eval_split="val",
            dtype="float16",
        )

    with pytest.raises(ValueError, match="eval_split"):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            decision_grid_path="/tmp/tape/decision_grid",
            checkpoint_path="/tmp/checkpoint.pt",
            split_source_dataset_root="/tmp/split-source",
            eval_split="train",
        )


def test_evaluate_execution_policy_accepts_zero_queue_weights():
    cfg = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        checkpoint_path="/tmp/checkpoint.pt",
        split_source_dataset_root="/tmp/split-source",
        eval_split="val",
        l2_decrease_weight=0.0,
        trade_at_level_weight=0.0,
    )

    assert cfg.l2_decrease_weight == 0.0
    assert cfg.trade_at_level_weight == 0.0


def test_evaluate_execution_policy_rejects_queue_weights_above_one():
    with pytest.raises(ValueError):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            decision_grid_path="/tmp/tape/decision_grid",
            checkpoint_path="/tmp/checkpoint.pt",
            split_source_dataset_root="/tmp/split-source",
            eval_split="val",
            l2_decrease_weight=1.1,
        )

    with pytest.raises(ValueError):
        ExecutionPolicyEvaluationCLIConfig(
            tape_root="/tmp/tape",
            decision_grid_path="/tmp/tape/decision_grid",
            checkpoint_path="/tmp/checkpoint.pt",
            split_source_dataset_root="/tmp/split-source",
            eval_split="val",
            trade_at_level_weight=1.1,
        )


def test_evaluate_execution_policy_can_use_cli_env_config_instead_of_checkpoint(tmp_path):
    tape_root, checkpoint_path = _train_tiny_checkpoint(tmp_path)

    summary = run_execution_policy_evaluation(
        ExecutionPolicyEvaluationCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            checkpoint_path=str(checkpoint_path),
            split_source_dataset_root=str(_split_source_root(tape_root)),
            eval_split="val",
            output_json=str(tmp_path / "eval_cli_env.json"),
            overwrite=True,
            override_env_config=True,
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


def test_parser_can_disable_l2_trade_dedupe():
    from mmrt.cli.evaluate_execution_policy import build_arg_parser, _env_config_from_cli_config, _config_from_args

    parser = build_arg_parser()
    args = parser.parse_args([
        "--tape-root", "/tmp/tape",
        "--decision-grid", "/tmp/tape/decision_grid",
        "--checkpoint-path", "/tmp/ckpt.pt",
        "--split-source-dataset-root", "/tmp/split-source",
        "--eval-split", "val",
        "--no-dedupe-l2-decrease-with-trade-prints",
    ])
    config = _config_from_args(args)
    env_config = _env_config_from_cli_config(config)
    assert config.dedupe_l2_decrease_with_trade_prints is False
    assert env_config.fill_simulator_config.queue_model.dedupe_l2_decrease_with_trade_prints is False


def test_parser_dedupe_l2_trade_default_enabled():
    from mmrt.cli.evaluate_execution_policy import build_arg_parser, _config_from_args

    parser = build_arg_parser()
    args = parser.parse_args([
        "--tape-root", "/tmp/tape",
        "--decision-grid", "/tmp/tape/decision_grid",
        "--checkpoint-path", "/tmp/ckpt.pt",
        "--split-source-dataset-root", "/tmp/split-source",
        "--eval-split", "val",
    ])
    config = _config_from_args(args)
    assert config.dedupe_l2_decrease_with_trade_prints is True


def _eval_adverse_label_config(queue_mode: str) -> dict[str, object]:
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


def _eval_adverse_signals(queue_mode: str) -> AdverseSelectionSignalArtifact:
    return AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=np.array([100], dtype=np.int64),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_event_seq=np.array([0], dtype=np.int64),
        target_names=("bid_touch_filled",),
        predictions={"bid_touch_filled": np.array([0.5], dtype=np.float32)},
        adverse_label_config=_eval_adverse_label_config(queue_mode),
        **grid_lineage_fields(n_rows=1),
    )


def test_evaluation_adverse_signal_queue_guard_allows_only_explicit_mismatch():
    config = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        checkpoint_path="/tmp/checkpoint.pt",
        split_source_dataset_root="/tmp/split-source",
        eval_split="val",
        queue_mode="balanced",
        override_env_config=True,
    )
    env_config = _env_config_from_cli_config(config)
    with pytest.raises(ValueError, match="adverse signal queue config mismatch"):
        _adverse_queue_config_compatibility(
            _eval_adverse_signals("conservative"),
            env_config=env_config,
            allow_mismatch=False,
        )
    allowed = _adverse_queue_config_compatibility(
        _eval_adverse_signals("conservative"),
        env_config=env_config,
        allow_mismatch=True,
    )
    assert allowed is not None
    assert allowed["status"] == "mismatch_allowed"


def test_evaluation_cli_env_config_adverse_runtime_uses_post_only_gap():
    from mmrt.cli.evaluate_execution_policy import _env_config_from_cli_config

    config = ExecutionPolicyEvaluationCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        checkpoint_path="/tmp/checkpoint.pt",
        split_source_dataset_root="/tmp/split-source",
        eval_split="val",
        adverse_signals_npz="/tmp/adverse.npz",
        post_only_gap_ticks=2,
        override_env_config=True,
    )
    env_config = _env_config_from_cli_config(config)

    assert env_config.adverse_runtime_config is not None
    assert env_config.quote_geometry_config.post_only_gap_ticks == 2
    assert env_config.adverse_runtime_config.post_only_gap_ticks == 2


def test_evaluation_checkpoint_env_config_adverse_runtime_uses_checkpoint_post_only_gap():
    from mmrt.cli.evaluate_execution_policy import _env_config_from_training_cli_config

    raw = {
        "adverse_signals_npz": "/tmp/adverse.npz",
        "post_only_gap_ticks": 3,
    }
    env_config = _env_config_from_training_cli_config(raw)

    assert env_config.quote_geometry_config.post_only_gap_ticks == 3
    assert env_config.adverse_runtime_config is not None
    assert env_config.adverse_runtime_config.post_only_gap_ticks == 3
