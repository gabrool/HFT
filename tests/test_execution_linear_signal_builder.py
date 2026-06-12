from decimal import Decimal
import json

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.execution_tape import build_execution_tape, save_execution_tape
from mmrt.execution.env import ExecutionEnv, ExecutionEnvConfig
from mmrt.execution.linear_signal import load_linear_signal_artifact_npz, save_linear_signal_artifact_npz
from mmrt.execution.linear_signal_builder import (
    ExecutionLinearFeatureDataset,
    build_execution_linear_feature_dataset,
    build_linear_signal_artifact_from_execution_features,
    execution_linear_feature_names,
    predict_linear_heads_for_execution_features,
)
from mmrt.linear import models as lm
from mmrt.linear import preprocess as pp
from mmrt.linear import train as tr
from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.features.transforms import TransformConfig
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode


def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures", symbol="BTCUSDT", mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC", quote_asset="USDT", margin_asset="USDT", contract_type="PERPETUAL", status="TRADING",
        tick_size=Decimal("0.1"), min_price=Decimal("0.1"), max_price=Decimal("1000000"),
        step_size=Decimal("0.001"), min_qty=Decimal("0.001"), max_qty=Decimal("100"), min_notional=Decimal("5"),
        allowed_order_types=("LIMIT",), allowed_time_in_force=("GTC", "GTX"),
    )


def _spec():
    return SymbolSpec("binance-futures", "BTCUSDT", 0.1, 0.001, 0.001, 100.0, 5.0)


def _l2(seq, local_ts_us, bid=1000, ask=1002):
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us,
        max_ts_us=local_ts_us,
        num_updates=2,
        is_snapshot_batch=(seq == 0),
        book_top=BookTop(local_ts_us, bid, ask, 1.0, 1.2),
        bid_depth=3,
        ask_depth=3,
        book_snapshot=BookLevelSnapshot(local_ts_us, (bid, bid - 1, bid - 2), (1.0, 2.0, 3.0), (ask, ask + 1, ask + 2), (1.2, 2.2, 3.2)),
    )


def _trade(local_ts_us, side=AggressorSide.BUY, price_tick=1001, amount=0.02, source_row=0):
    return TradePrint(local_ts_us, local_ts_us, side, price_tick, amount, source_row=source_row)


def _tiny_tape():
    l2 = [_l2(0, 100), _l2(1, 200), _l2(2, 300, bid=1001, ask=1003), _l2(3, 400, bid=1002, ask=1004)]
    trades = [_trade(150, source_row=0), _trade(250, side=AggressorSide.SELL, source_row=1)]
    return build_execution_tape(symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=trades, merged_events=merge_execution_events(l2, trades).events, book_depth=3, created_at_utc="2026-01-01T00:00:00Z")


_SCHED50 = DecisionScheduleConfig(min_decision_interval_us=50, max_decision_interval_us=50)


def _preprocess_state(cols):
    n = len(cols)
    return pp.LinearPreprocessState(
        feature_columns=tuple(cols),
        n_rows_fit=2,
        mean=np.zeros(n),
        variance=np.ones(n),
        scale=np.ones(n),
        active_mask=np.ones(n, dtype=bool),
        config=pp.LinearPreprocessConfig(),
    )


def _train_result(feature_columns_by_head):
    bundle = lm.make_linear_model_bundle(feature_columns_by_head)
    states = {head: _preprocess_state(cols) for head, cols in feature_columns_by_head.items()}
    return tr.LinearTrainResult(
        schema=tr.LINEAR_TRAINING_RESULT_SCHEMA,
        dataset_id="dataset",
        manifest_hash="hash",
        config={},
        decision_schedule=_SCHED50.as_dict(),
        transform_config=TransformConfig().as_dict(),
        preprocess_state={"schema": "mmrt_linear_preprocess", "states_by_head": {h: states[h].as_dict() for h in lm.MODEL_HEADS}},
        model_bundle_state=bundle.as_dict(),
        splits={
            "train": tr.SplitEvaluation("train", 1, {}, {}),
            "val": tr.SplitEvaluation("val", 1, {}, {}),
        },
        selection_summary={},
    )


def test_build_execution_linear_feature_dataset_is_causal_and_aligned():
    dataset = build_execution_linear_feature_dataset(_tiny_tape(), schedule_config=_SCHED50)
    assert dataset.num_decisions > 1
    assert np.all(np.diff(dataset.decision_event_index) > 0)
    assert np.all(np.diff(dataset.decision_local_ts_us) > 0)
    assert np.isfinite(dataset.features).all()


def test_execution_linear_feature_dataset_start_event_index_is_first_decision_not_replay_start():
    dataset = build_execution_linear_feature_dataset(_tiny_tape(), schedule_config=_SCHED50, start_event_index=0)

    assert dataset.replay_start_event_index == 0
    assert dataset.start_event_index == int(dataset.decision_event_index[0])
    assert dataset.start_event_index > dataset.replay_start_event_index


def test_execution_linear_feature_names_match_storage_linear_prefix_contract():
    dataset = build_execution_linear_feature_dataset(_tiny_tape(), schedule_config=_SCHED50)
    assert dataset.feature_names == execution_linear_feature_names()
    assert all(name.startswith("x_") for name in dataset.feature_names)
    assert "x_mid_slope_bps_per_sec_1000000us" in dataset.feature_names


def test_build_execution_linear_feature_dataset_respects_start_and_max_decisions():
    limited = build_execution_linear_feature_dataset(_tiny_tape(), schedule_config=_SCHED50, start_event_index=2, max_decisions=2)
    assert limited.num_decisions <= 2
    assert int(limited.decision_event_index[0]) >= 2


def test_build_linear_signal_artifact_from_execution_features_roundtrip(tmp_path):
    tape = _tiny_tape()
    dataset = build_execution_linear_feature_dataset(tape, schedule_config=_SCHED50)
    cols = tuple(dataset.feature_names[:3])
    result = _train_result({head: cols for head in lm.MODEL_HEADS})
    artifact = build_linear_signal_artifact_from_execution_features(tape=tape, feature_dataset=dataset, linear_train_result=result)
    path = tmp_path / "signals.npz"
    save_linear_signal_artifact_npz(path, artifact)
    loaded = load_linear_signal_artifact_npz(path)
    assert loaded.metadata.n_rows == dataset.num_decisions
    assert np.isfinite(loaded.arrays.expected_return_bps).all()


def test_linear_signal_artifact_metadata_start_is_first_signal_row():
    tape = _tiny_tape()
    dataset = build_execution_linear_feature_dataset(tape, schedule_config=_SCHED50)
    cols = tuple(dataset.feature_names[:3])
    result = _train_result({head: cols for head in lm.MODEL_HEADS})

    artifact = build_linear_signal_artifact_from_execution_features(tape=tape, feature_dataset=dataset, linear_train_result=result)

    assert artifact.metadata.start_event_index == int(artifact.decision_event_index[0])


def test_generated_linear_signals_reset_execution_env_without_alignment_error():
    tape = _tiny_tape()
    dataset = build_execution_linear_feature_dataset(tape, schedule_config=_SCHED50)
    cols = tuple(dataset.feature_names[:3])
    result = _train_result({head: cols for head in lm.MODEL_HEADS})
    artifact = build_linear_signal_artifact_from_execution_features(tape=tape, feature_dataset=dataset, linear_train_result=result)

    reset = ExecutionEnv(tape, linear_signals=artifact).reset()

    assert reset.info["event_index"] == int(artifact.decision_event_index[0])
    assert reset.info["signal_row_index"] == 0


def test_generated_linear_signals_env_can_step_once():
    tape = _tiny_tape()
    dataset = build_execution_linear_feature_dataset(tape, schedule_config=_SCHED50)
    cols = tuple(dataset.feature_names[:3])
    result = _train_result({head: cols for head in lm.MODEL_HEADS})
    artifact = build_linear_signal_artifact_from_execution_features(tape=tape, feature_dataset=dataset, linear_train_result=result)
    env = ExecutionEnv(tape, linear_signals=artifact, config=ExecutionEnvConfig())

    env.reset()
    step = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    assert isinstance(step.reward, float)


def test_linear_signal_builder_rejects_missing_feature_columns():
    tape = _tiny_tape()
    dataset = build_execution_linear_feature_dataset(tape, schedule_config=_SCHED50)
    result = _train_result({head: ("x_not_produced",) for head in lm.MODEL_HEADS})
    with pytest.raises(ValueError, match="x_not_produced"):
        build_linear_signal_artifact_from_execution_features(tape=tape, feature_dataset=dataset, linear_train_result=result)


def test_linear_signal_builder_supports_per_head_feature_sets():
    dataset = build_execution_linear_feature_dataset(_tiny_tape(), schedule_config=_SCHED50)
    feature_sets = {
        lm.NO_MOVE_HEAD: tuple(dataset.feature_names[:2]),
        lm.DIRECTION_HEAD: tuple(dataset.feature_names[1:4]),
        lm.MAGNITUDE_UP_HEAD: tuple(dataset.feature_names[4:6]),
        lm.MAGNITUDE_DOWN_HEAD: tuple(dataset.feature_names[6:9]),
    }
    result = _train_result(feature_sets)
    preds = predict_linear_heads_for_execution_features(
        feature_dataset=dataset,
        model_bundle=tr.linear_model_bundle_from_train_result(result),
        preprocess_states_by_head=tr.linear_preprocess_states_from_train_result(result),
    )
    assert preds["no_move_proba"].shape[0] == dataset.num_decisions


def test_linear_signal_builder_rejects_preprocess_model_feature_mismatch():
    dataset = build_execution_linear_feature_dataset(_tiny_tape(), schedule_config=_SCHED50)
    cols = tuple(dataset.feature_names[:2])
    bundle = lm.make_linear_model_bundle({head: cols for head in lm.MODEL_HEADS})
    bad_state = _preprocess_state(tuple(dataset.feature_names[1:3])).as_dict()
    result = tr.LinearTrainResult(
        schema=tr.LINEAR_TRAINING_RESULT_SCHEMA,
        dataset_id="dataset",
        manifest_hash="hash",
        config={},
        decision_schedule=DecisionScheduleConfig().as_dict(),
        transform_config=TransformConfig().as_dict(),
        preprocess_state={"schema": "mmrt_linear_preprocess", "states_by_head": {h: bad_state for h in lm.MODEL_HEADS}},
        model_bundle_state=bundle.as_dict(),
        splits={"train": tr.SplitEvaluation("train", 1, {}, {}), "val": tr.SplitEvaluation("val", 1, {}, {})},
        selection_summary={},
    )
    with pytest.raises(ValueError, match="feature_columns"):
        tr.linear_preprocess_states_from_train_result(result)
