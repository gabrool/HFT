import inspect
import json
from decimal import Decimal

import numpy as np

import pytest
import torch

from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
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
from mmrt.cli.train_execution_ppo import (
    ExecutionPPOTrainCLIConfig,
    main,
    run_execution_ppo_training,
)
from mmrt.rl.normalization import ObservationNormalizer
from mmrt.rl.rollout import RolloutCollector, RolloutConfig
from mmrt.rl.torch_networks import ActorCriticConfig, ActorCriticNetwork
from mmrt.rl.ppo import PPOConfig
from mmrt.rl.train import PPOTrainingConfig, train_ppo_policy, make_training_checkpoint_payload
from tests.grid_helpers import decision_grid_for_tape



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


def _tiny_tape_root(tmp_path):
    root = _save_tape(tmp_path, _tiny_tape())
    _save_linear_signals(root)
    return root


def _tiny_env() -> ExecutionEnv:
    tape = _tiny_tape()
    decision_grid = decision_grid_for_tape(tape)
    return ExecutionEnv(
        tape,
        decision_grid=decision_grid,
        linear_signals=_linear_artifact_for_tape(tape, decision_interval_us=50, decision_grid=decision_grid),
        config=ExecutionEnvConfig(
            max_episode_steps=4,
        ),
    )


def test_rollout_collector_collects_tiny_env_batch():
    env = _tiny_env()
    obs_dim = env.config.observation_schema.dim
    policy = ActorCriticNetwork(
        obs_dim=obs_dim,
        config=ActorCriticConfig(hidden_sizes=(8,)),
    )
    normalizer = ObservationNormalizer(obs_shape=obs_dim)
    collector = RolloutCollector(
        env,
        policy,
        config=RolloutConfig(
            rollout_steps=4,
            gamma=0.99,
            gae_lambda=0.95,
            device="cpu",
        ),
        observation_normalizer=normalizer,
    )
    batch = collector.collect()

    assert batch.num_steps == 4
    assert batch.observations.shape == (4, obs_dim)
    assert batch.actions.shape == (4, 8)
    assert batch.log_probs.shape == (4,)
    assert batch.values.shape == (4,)
    assert batch.advantages.shape == (4,)
    assert batch.returns.shape == (4,)
    assert torch.isfinite(batch.observations).all()
    assert torch.isfinite(batch.actions).all()
    assert torch.isfinite(batch.log_probs).all()
    assert torch.isfinite(batch.values).all()
    assert torch.isfinite(batch.advantages).all()
    assert torch.isfinite(batch.returns).all()
    assert normalizer.running.count > normalizer.running.initial_epsilon


def test_train_ppo_policy_runs_one_update_on_tiny_env():
    env = _tiny_env()
    config = PPOTrainingConfig(
        num_updates=1,
        learning_rate=1e-3,
        seed=123,
        network_config=ActorCriticConfig(hidden_sizes=(8,)),
        rollout_config=RolloutConfig(
            rollout_steps=4,
            device="cpu",
            dtype=torch.float32,
        ),
        ppo_config=PPOConfig(
            update_epochs=1,
            minibatch_size=2,
        ),
    )

    result = train_ppo_policy(env, config=config)

    assert result.updates_completed == 1
    assert len(result.history) == 1
    assert result.history[0].ppo.minibatches_processed == 2
    assert result.observation_normalizer is not None
    summary = result.summary_dict()
    assert summary["status"] == "ok"
    assert summary["updates_completed"] == 1

    payload = make_training_checkpoint_payload(result)
    assert payload["schema"] == "mmrt_execution_ppo_checkpoint"
    assert "policy_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert payload["observation_normalizer_state_dict"] is not None


def test_run_execution_ppo_training_writes_summary_and_checkpoint(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    output_json = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    summary = run_execution_ppo_training(
        ExecutionPPOTrainCLIConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            output_json=str(output_json),
            checkpoint_path=str(checkpoint_path),
            overwrite=True,
            num_updates=1,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_sizes=(8,),
            max_episode_steps=4,
            seed=123,
        )
    )

    assert output_json.exists()
    assert checkpoint_path.exists()
    assert json.loads(output_json.read_text()) == summary
    assert summary["status"] == "ok"
    assert summary["run_type"] == "train_execution_ppo"
    assert summary["checkpoint_saved"] is True
    assert summary["training"]["updates_completed"] == 1
    assert summary["training"]["final"]["ppo"]["minibatches_processed"] == 2
    assert summary["linear_signals"]["schema"] == "mmrt_execution_linear_signals_grid_v1"
    assert summary["linear_signals"]["n_rows"] >= 1

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert ckpt["schema"] == "mmrt_execution_ppo_checkpoint"
    assert ckpt["updates_completed"] == 1
    assert "policy_state_dict" in ckpt
    assert ckpt["tape"]["symbol"] == "BTCUSDT"
    assert ckpt["observation_schema"] == summary["observation_schema"]
    assert ckpt["linear_signals"]["schema"] == "mmrt_execution_linear_signals_grid_v1"


def test_train_execution_ppo_main_writes_summary_and_prints_json(tmp_path, capsys):
    tape_root = _tiny_tape_root(tmp_path)
    output_json = tmp_path / "summary.json"

    rc = main(
        [
            "--tape-root",
            str(tape_root),
            "--decision-grid",
            str(tape_root / "decision_grid"),
            "--output-json",
            str(output_json),
            "--num-updates",
            "1",
            "--rollout-steps",
            "4",
            "--update-epochs",
            "1",
            "--minibatch-size",
            "2",
            "--hidden-sizes",
            "8",
            "--max-episode-steps",
            "4",
            "--seed",
            "123",
            "--no-checkpoint",
            "--overwrite",
        ]
    )

    assert rc == 0
    assert output_json.exists()
    stdout_payload = json.loads(capsys.readouterr().out)
    disk_payload = json.loads(output_json.read_text())
    assert stdout_payload == disk_payload
    assert stdout_payload["checkpoint_saved"] is False
    assert stdout_payload["training"]["updates_completed"] == 1



def test_train_execution_ppo_requires_linear_signals_file(tmp_path):
    tape_root = _save_tape(tmp_path, _tiny_tape())
    with pytest.raises(FileNotFoundError):
        run_execution_ppo_training(
            ExecutionPPOTrainCLIConfig(
                tape_root=str(tape_root),
                decision_grid_path=str(tape_root / "decision_grid"),
                output_json=str(tmp_path / "summary.json"),
                save_checkpoint=False,
                overwrite=True,
                num_updates=1,
                rollout_steps=2,
                update_epochs=1,
                minibatch_size=2,
                hidden_sizes=(8,),
                max_episode_steps=2,
            )
        )

def test_train_execution_ppo_refuses_overwrite_without_flag(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    output_json = tmp_path / "summary.json"
    output_json.write_text("{}")

    with pytest.raises(FileExistsError):
        run_execution_ppo_training(
            ExecutionPPOTrainCLIConfig(
                tape_root=str(tape_root),
                decision_grid_path=str(tape_root / "decision_grid"),
                output_json=str(output_json),
                save_checkpoint=False,
                num_updates=1,
                rollout_steps=2,
                update_epochs=1,
                minibatch_size=2,
                hidden_sizes=(8,),
            )
        )

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        run_execution_ppo_training(
            ExecutionPPOTrainCLIConfig(
                tape_root=str(tape_root),
                decision_grid_path=str(tape_root / "decision_grid"),
                output_json=str(tmp_path / "fresh.json"),
                checkpoint_path=str(checkpoint_path),
                save_checkpoint=True,
                num_updates=1,
                rollout_steps=2,
                update_epochs=1,
                minibatch_size=2,
                hidden_sizes=(8,),
            )
        )


def test_train_execution_ppo_cli_config_parses_hidden_sizes_and_dtype():
    cfg = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        hidden_sizes="8,4",
        dtype="float64",
        queue_mode="balanced",
    )
    assert cfg.hidden_sizes == (8, 4)
    assert cfg.dtype is torch.float64
    assert cfg.queue_mode.value == "balanced"

    cfg = ExecutionPPOTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/tape/decision_grid", hidden_sizes="none")
    assert cfg.hidden_sizes == ()


def test_train_execution_ppo_accepts_zero_queue_weights():
    cfg = ExecutionPPOTrainCLIConfig(
        tape_root="/tmp/tape",
        decision_grid_path="/tmp/tape/decision_grid",
        l2_decrease_weight=0.0,
        trade_at_level_weight=0.0,
    )

    assert cfg.l2_decrease_weight == 0.0
    assert cfg.trade_at_level_weight == 0.0


def test_train_execution_ppo_rejects_queue_weights_above_one():
    with pytest.raises(ValueError):
        ExecutionPPOTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/tape/decision_grid", l2_decrease_weight=1.1)

    with pytest.raises(ValueError):
        ExecutionPPOTrainCLIConfig(tape_root="/tmp/tape", decision_grid_path="/tmp/tape/decision_grid", trade_at_level_weight=1.1)


def test_train_execution_ppo_cli_does_not_import_forbidden_modules():
    import mmrt.cli.train_execution_ppo as cli

    source = inspect.getsource(cli)
    for text in (
        "pandas",
        "polars",
        "pyarrow",
        "sklearn",
        "gym",
        "gymnasium",
        "mmrt.linear",
        "mmrt.storage",
        "build_execution_tape",
    ):
        assert text not in source
