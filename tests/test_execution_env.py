from pathlib import Path

from decimal import Decimal

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActionSpec,
    LatencyConfig,
    BookLevelSnapshot,
    BookTop,
    OrderSide,
    PositionState,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.event_merge import merge_execution_events
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.execution.execution_tape import build_execution_tape
from mmrt.execution.env import (
    ExecutionEnv,
    ExecutionEnvConfig,
    ExecutionEnvReset,
    ExecutionEnvStep,
    action_array_to_continuous_action,
)
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_SIGNALS_SCHEMA, AdverseSelectionSignalArtifact
from mmrt.execution.fill_sim import FillSimulatorConfig
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    predictions_to_signal_arrays,
)
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.queue_model import QueueModelConfig, QueueModelMode
from mmrt.execution.quote_geometry import QuoteAction, QuoteGeometryConfig
from mmrt.execution.reward import RewardConfig
from mmrt.time_key import EventKey, MAX_EVENT_SEQ



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


def _env_config(**kwargs) -> ExecutionEnvConfig:
    base = dict(
        decision_interval_us=100,
        action_spec=ActionSpec(max_distance_ticks=1, max_order_qty=1.0),
        quote_geometry_config=QuoteGeometryConfig(
            post_only_gap_ticks=1,
            default_order_qty=1.0,
        ),
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=QueueModelMode.BALANCED,
                l2_decrease_weight=1.0,
                trade_at_level_weight=1.0,
            ),
            maker_fee_bps=0.0,
        ),
        latency_config=LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0, cancel_latency_us=0),
        reward_config=RewardConfig(),
    )
    base.update(kwargs)
    return ExecutionEnvConfig(**base)


def _signal_arrays(n_rows: int = 16):
    prediction = {
        NO_MOVE_PROBA_KEY: np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (n_rows, 1)),
        DIRECTION_PROBA_KEY: np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (n_rows, 1)),
        MAGNITUDE_UP_KEY: np.full(n_rows, np.log1p(10.0), dtype=np.float32),
        MAGNITUDE_DOWN_KEY: np.full(n_rows, np.log1p(5.0), dtype=np.float32),
    }
    return predictions_to_signal_arrays(prediction)


def _linear_signals(tape, n_rows: int = 16, *, start_event_index: int = 0) -> LinearSignalArtifact:
    arrays = _signal_arrays(n_rows)
    valid_pairs = []
    for event_index, event in enumerate(tape.arrays.events):
        if event_index < start_event_index:
            continue
        if int(event["event_type_code"]) != 1:
            continue
        book_ptr = int(event["book_ptr"])
        if book_ptr >= 0:
            valid_pairs.append((event_index, int(tape.arrays.l2_events[book_ptr]["local_ts_us"])))
    if not valid_pairs:
        valid_pairs.append((start_event_index, int(tape.manifest.start_local_ts_us)))
    decision_event_index = [pair[0] for pair in valid_pairs[:n_rows]]
    decision_local_ts_us = [pair[1] for pair in valid_pairs[:n_rows]]
    while len(decision_event_index) < n_rows:
        decision_event_index.append(decision_event_index[-1] + 1)
        decision_local_ts_us.append(decision_local_ts_us[-1] + 100)
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_interval_us=100,
        start_event_index=start_event_index,
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.asarray(decision_event_index, dtype=np.int64),
        decision_local_ts_us=np.asarray(decision_local_ts_us, dtype=np.int64),
    )


def _linear_signals_for_decisions(tape, decision_event_index, decision_local_ts_us):
    arrays = _signal_arrays(max(len(decision_event_index), 2))
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_interval_us=100,
        start_event_index=0,
        n_rows=len(decision_event_index),
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.asarray(decision_event_index, dtype=np.int64),
        decision_local_ts_us=np.asarray(decision_local_ts_us, dtype=np.int64),
    )


def _bid_only_action() -> QuoteAction:
    return QuoteAction(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_raw=0.0,
        ask_price_raw=0.0,
        bid_size_raw=100.0,
        ask_size_raw=0.0,
    )


def _ask_only_action() -> QuoteAction:
    return QuoteAction(
        bid_enabled=False,
        ask_enabled=True,
        bid_price_raw=0.0,
        ask_price_raw=0.0,
        bid_size_raw=0.0,
        ask_size_raw=100.0,
    )


def _disabled_action() -> QuoteAction:
    return QuoteAction(
        bid_enabled=False,
        ask_enabled=False,
        bid_price_raw=0.0,
        ask_price_raw=0.0,
        bid_size_raw=0.0,
        ask_size_raw=0.0,
    )


def test_action_array_to_continuous_action():
    action = action_array_to_continuous_action(np.array([1, -1, 0, 0, 2, -2], dtype=np.float32))

    assert isinstance(action, QuoteAction)
    assert action.bid_enabled is True
    assert action.ask_enabled is False

    with pytest.raises(ValueError):
        action_array_to_continuous_action([1.0, 2.0])

    with pytest.raises(ValueError):
        action_array_to_continuous_action([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")])


def test_reset_returns_initial_observation():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())

    reset = env.reset()

    assert isinstance(reset, ExecutionEnvReset)
    assert reset.observation.shape == (env.config.observation_schema.dim,)
    assert reset.observation.dtype == np.float32
    assert reset.info["event_index"] == 0
    assert reset.info["current_book_ptr"] == 0


def test_disabled_action_advances_without_orders_or_fills():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())
    env.reset()

    step = env.step(_disabled_action())

    assert isinstance(step, ExecutionEnvStep)
    assert step.reward == pytest.approx(0.0)
    assert step.execution.fills == ()
    assert step.info["quote_bid_enabled"] is False
    assert step.info["quote_ask_enabled"] is False


def test_bid_trade_fill_updates_position_and_reward():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=300),
    ]
    trades = [
        _trade(
            local_ts_us=150,
            side=AggressorSide.SELL,
            price_tick=1000,
            amount=2.0,
            source_row=0,
        )
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())
    env.reset()

    step = env.step(_bid_only_action())

    assert len(step.fills) == 1
    fill = step.fills[0]
    assert fill.side == OrderSide.BUY
    assert fill.price_tick == 1001
    assert fill.qty == pytest.approx(1.0)
    assert step.position.inventory_qty == pytest.approx(1.0)
    assert step.position.cash == pytest.approx(-100.1)

    assert step.execution.reward.raw_equity_delta == pytest.approx(0.0)
    assert step.reward == pytest.approx(0.0)


def test_ask_trade_fill_updates_short_position():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=300),
    ]
    trades = [
        _trade(
            local_ts_us=150,
            side=AggressorSide.BUY,
            price_tick=1002,
            amount=2.0,
            source_row=0,
        )
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())
    env.reset()

    step = env.step(_ask_only_action())

    assert len(step.fills) == 1
    assert step.fills[0].side == OrderSide.SELL
    assert step.position.inventory_qty == pytest.approx(-1.0)
    assert step.position.cash == pytest.approx(100.1)


def test_l2_queue_decrease_advances_queue_without_artificial_fill():
    l2_events = [
        _l2(
            seq=0,
            local_ts_us=100,
            bid_ticks=(1000, 999),
            bid_sizes=(1.0, 2.0),
        ),
        _l2(
            seq=1,
            local_ts_us=150,
            bid_ticks=(999,),
            bid_sizes=(2.0,),
        ),
        _l2(
            seq=2,
            local_ts_us=300,
            bid_ticks=(999,),
            bid_sizes=(2.0,),
        ),
    ]
    tape = _tape(l2_events, [])
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert step.position == PositionState()
    assert len(env._state.live_orders) == 1
    order = env._state.live_orders[0]
    assert order.side == OrderSide.BUY
    assert order.price_tick == 1001
    assert order.queue_ahead_qty == pytest.approx(0.0)
    assert order.remaining_qty == pytest.approx(1.0)


def test_repeated_decision_cancels_previous_live_order():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
    ]
    tape = _tape(l2_events, [])
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config(decision_interval_us=50))
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["cancel_count"] == 0

    second = env.step(_ask_only_action())
    assert second.info["cancel_count"] == 1


def test_max_episode_steps_truncates():
    tape = _tape(
        [
            _l2(seq=0, local_ts_us=100),
            _l2(seq=1, local_ts_us=200),
            _l2(seq=2, local_ts_us=300),
        ],
        [],
    )
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config(max_episode_steps=1))
    env.reset()

    step = env.step(_disabled_action())

    assert step.truncated is True
    assert step.done is False


def test_step_after_terminal_rejected():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config(max_episode_steps=1))
    env.reset()
    env.step(_disabled_action())

    with pytest.raises(RuntimeError):
        env.step(_disabled_action())


def test_linear_signal_rows_used_by_step_index():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    signals = _linear_signals(tape, n_rows=2)
    env = ExecutionEnv(tape, linear_signals=signals, config=_env_config())

    reset = env.reset()
    idx = env.config.observation_schema.index("linear_p_no_move")
    assert reset.observation[idx] == pytest.approx(0.2)

    step = env.step(_disabled_action())
    assert step.observation[idx] == pytest.approx(0.2)


def test_execution_env_requires_linear_signals():
    tape = _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], [])
    with pytest.raises((TypeError, ValueError)):
        ExecutionEnv(tape, config=_env_config(), linear_signals=None)


def test_execution_env_rejects_insufficient_linear_signal_rows():
    tape = _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], [])
    with pytest.raises(ValueError, match="linear_signals"):
        ExecutionEnv(tape, linear_signals=_linear_signals(tape, n_rows=4), config=_env_config(max_episode_steps=4))


def test_execution_env_rejects_signal_event_index_mismatch():
    tape = _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], [])
    artifact = _linear_signals(tape, n_rows=2)
    bad = LinearSignalArtifact(
        arrays=artifact.arrays,
        metadata=artifact.metadata,
        decision_event_index=np.asarray([1, 2], dtype=np.int64),
        decision_local_ts_us=artifact.decision_local_ts_us,
    )
    env = ExecutionEnv(tape, linear_signals=bad, config=_env_config())
    with pytest.raises(ValueError, match="decision_event_index mismatch"):
        env.reset()


def test_execution_env_rejects_signal_local_ts_mismatch():
    tape = _tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)], [])
    artifact = _linear_signals(tape, n_rows=2)
    bad_ts = artifact.decision_local_ts_us.copy()
    bad_ts[0] = 101
    bad = LinearSignalArtifact(
        arrays=artifact.arrays,
        metadata=artifact.metadata,
        decision_event_index=artifact.decision_event_index,
        decision_local_ts_us=bad_ts,
    )
    env = ExecutionEnv(tape, linear_signals=bad, config=_env_config())
    with pytest.raises(ValueError, match="decision_local_ts_us mismatch"):
        env.reset()


def test_execution_env_source_has_no_linear_fallback():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")
    assert "neutral_linear_signal" not in source
    assert "linear_signals is None" not in source


def test_reset_rejects_tape_without_valid_two_sided_book():
    local_ts_us = 100
    snapshot = BookLevelSnapshot(
        local_ts_us=local_ts_us,
        bid_ticks=(1000,),
        bid_sizes=(1.0,),
        ask_ticks=(),
        ask_sizes=(),
    )
    l2_event = ReconstructedL2Event(
        batch_seq=0,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us - 10,
        max_ts_us=local_ts_us - 5,
        num_updates=1,
        is_snapshot_batch=True,
        book_top=None,
        bid_depth=1,
        ask_depth=0,
        book_snapshot=snapshot,
    )
    tape = _tape([l2_event], [])
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())

    with pytest.raises(ValueError, match="valid two-sided"):
        env.reset()


def test_env_step_info_is_json_safe_and_validated_by_contract():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())
    env.reset()

    step = env.step(_disabled_action())

    assert isinstance(step.info["quote_bid_enabled"], bool)
    assert isinstance(step.info["quote_bid_disabled_reason"], str)
    assert isinstance(step.info["events_processed"], int)
    assert isinstance(step.info["previous_equity"], float)
    assert step.execution.info is step.info or step.execution.info == step.info


def test_env_has_no_forbidden_imports():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import gym" not in source
    assert "import gymnasium" not in source
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import sklearn" not in source
    assert "import pyarrow" not in source
    assert "mmrt.storage" not in source
    assert "mmrt.linear.models" not in source
    assert "mmrt.rl" not in source


def test_env_does_not_bypass_execution_step_result_info_validation():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")

    assert 'object.__setattr__(execution, "info"' not in source
    assert "object.__setattr__(execution, 'info'" not in source


def test_reset_with_start_event_index_starts_later_in_tape():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape, start_event_index=1), config=_env_config())

    reset = env.reset(start_event_index=1)

    assert reset.info["event_index"] == 1
    assert reset.info["current_book_ptr"] == 1


def test_unknown_trade_side_does_not_fill():
    l2_events = [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=300)]
    trades = [
        _trade(
            local_ts_us=150,
            side=AggressorSide.UNKNOWN,
            price_tick=1000,
            amount=2.0,
            source_row=0,
        )
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=_env_config())
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert step.position == PositionState()


def test_post_only_safe_at_decision_but_marketable_at_activation_is_rejected():
    l2_events = [
        _l2(seq=0, local_ts_us=100, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
        _l2(seq=1, local_ts_us=150, bid_ticks=(999,), bid_sizes=(1.0,), ask_ticks=(1001,), ask_sizes=(1.0,)),
        _l2(seq=2, local_ts_us=300, bid_ticks=(999,), bid_sizes=(1.0,), ask_ticks=(1001,), ask_sizes=(1.0,)),
    ]
    trades = [
        _trade(local_ts_us=200, side=AggressorSide.SELL, price_tick=1001, amount=10.0, source_row=0),
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 2], [100, 150]),
        config=_env_config(
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=50,
                cancel_latency_us=0,
            )
        ),
    )
    env.reset()

    step = env.step(_bid_only_action())

    assert step.info["post_only_reject_count"] == 1
    assert step.fills == ()
    assert step.info["orders_live_count"] == 0


def test_order_activation_between_events_can_fill_next_trade():
    l2_events = [
        _l2(seq=0, local_ts_us=100, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
        _l2(seq=1, local_ts_us=300, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
    ]
    trades = [
        _trade(local_ts_us=150, side=AggressorSide.SELL, price_tick=1001, amount=10.0, source_row=0),
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 2], [100, 300]),
        config=_env_config(
            decision_interval_us=300,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=20,
                cancel_latency_us=0,
            ),
        ),
    )
    env.reset()

    step = env.step(_bid_only_action())

    assert len(step.fills) == 1
    assert step.fills[0].local_ts_us == 150


def test_cancel_between_events_blocks_next_trade():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
    ]
    trades = [
        _trade(local_ts_us=250, side=AggressorSide.SELL, price_tick=1001, amount=10.0, source_row=0),
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 1, 3], [100, 200, 300]),
        config=_env_config(
            decision_interval_us=50,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=0,
                cancel_latency_us=20,
            ),
        ),
    )
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["orders_live_count"] >= 1

    second = env.step(_disabled_action())

    assert second.info["effective_cancel_count"] >= 1
    assert second.fills == ()


def test_pending_order_queue_ahead_is_refreshed_at_activation():
    l2_events = [
        _l2(
            seq=0,
            local_ts_us=100,
            bid_ticks=(1000,),
            bid_sizes=(1.0,),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=1,
            local_ts_us=150,
            bid_ticks=(1001, 1000),
            bid_sizes=(3.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=2,
            local_ts_us=300,
            bid_ticks=(1001, 1000),
            bid_sizes=(3.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
    ]
    trades = [
        _trade(local_ts_us=200, side=AggressorSide.SELL, price_tick=1001, amount=2.0, source_row=0),
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 3], [100, 300]),
        config=_env_config(
            decision_interval_us=300,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=80,
                cancel_latency_us=0,
            ),
            fill_simulator_config=FillSimulatorConfig(
                queue_model=QueueModelConfig(
                    mode=QueueModelMode.BALANCED,
                    trade_at_level_weight=1.0,
                    l2_decrease_weight=1.0,
                ),
                maker_fee_bps=0.0,
            ),
        ),
    )
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert len(env._state.live_orders) == 1
    order = env._state.live_orders[0]
    assert order.price_tick == 1001
    assert order.queue_ahead_qty == pytest.approx(1.0)


def test_env_no_next_event_latency_snapping_helper():
    source = Path("mmrt/execution/env.py").read_text(encoding="utf-8")
    assert "_effective_event_key_for_latency" not in source
    assert "_order_activation_key_for_latency" in source
    assert "_cancel_effective_key_for_latency" in source


def test_activation_at_exact_event_key_does_not_fill_on_that_event():
    l2_events = [
        _l2(seq=0, local_ts_us=100, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
        _l2(seq=1, local_ts_us=160, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
    ]
    trades = [
        _trade(local_ts_us=150, side=AggressorSide.SELL, price_tick=1001, amount=10.0, source_row=0),
        _trade(local_ts_us=170, side=AggressorSide.SELL, price_tick=1001, amount=10.0, source_row=1),
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 3], [100, 160]),
        config=_env_config(
            decision_interval_us=200,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=50,
                cancel_latency_us=0,
            ),
        ),
    )
    env.reset()

    step = env.step(_bid_only_action())

    assert len(step.fills) == 1
    assert step.fills[0].local_ts_us == 170
    assert step.fills[0].event_seq > 0


def test_cancel_exact_event_key_blocks_fill_on_that_event():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
    ]
    trades = [
        _trade(local_ts_us=250, side=AggressorSide.SELL, price_tick=1001, amount=10.0, source_row=0),
    ]
    tape = _tape(l2_events, trades)
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals(tape),
        config=_env_config(
            decision_interval_us=50,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=0,
                cancel_latency_us=50,
            ),
        ),
    )
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["orders_live_count"] >= 1

    second = env.step(_disabled_action())

    assert second.info["effective_cancel_count"] >= 1
    assert second.fills == ()


def test_repeated_same_quote_before_activation_preserves_pending_new_order():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=150),
        _l2(seq=2, local_ts_us=200),
        _l2(seq=3, local_ts_us=500),
    ]
    tape = _tape(l2_events, [])
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals(tape),
        config=_env_config(
            decision_interval_us=25,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=300,
                cancel_latency_us=0,
            ),
        ),
    )
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["cancel_count"] == 0
    first_orders = tuple(env._state.live_orders)
    assert len(first_orders) == 1
    order_id = first_orders[0].order_id
    effective_key = first_orders[0].effective_key

    second = env.step(_bid_only_action())
    assert second.info["cancel_count"] == 0
    assert len(env._state.live_orders) == 1
    assert env._state.live_orders[0].order_id == order_id
    assert env._state.live_orders[0].effective_key == effective_key


def test_trade_before_activation_does_not_dedupe_later_l2_decrease():
    l2_events = [
        # Decision book: bid 1000 / ask 1002, bid 1001 is inside spread.
        _l2(
            seq=0,
            local_ts_us=100,
            bid_ticks=(1000,),
            bid_sizes=(1.0,),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        # Before activation, visible queue appears at our future price.
        _l2(
            seq=1,
            local_ts_us=120,
            bid_ticks=(1001, 1000),
            bid_sizes=(3.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        # After activation, visible queue drops by 1.0.
        _l2(
            seq=2,
            local_ts_us=200,
            bid_ticks=(1001, 1000),
            bid_sizes=(2.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=3,
            local_ts_us=300,
            bid_ticks=(1001, 1000),
            bid_sizes=(2.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
    ]
    trades = [
        # This trade is before our order activation. It should not create a
        # de-dup ledger entry because our order is not fillable yet.
        _trade(
            local_ts_us=150,
            side=AggressorSide.SELL,
            price_tick=1001,
            amount=1.0,
            source_row=0,
        ),
    ]
    tape = _tape(l2_events, trades)

    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 4], [100, 300]),
        config=_env_config(
            decision_interval_us=300,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=80,  # activation target = 180
                cancel_latency_us=0,
            ),
            fill_simulator_config=FillSimulatorConfig(
                queue_model=QueueModelConfig(
                    mode=QueueModelMode.BALANCED,
                    trade_at_level_weight=1.0,
                    l2_decrease_weight=1.0,
                    dedupe_l2_decrease_with_trade_prints=True,
                ),
                maker_fee_bps=0.0,
            ),
        ),
    )
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert step.info["l2_trade_dedupe_qty"] == pytest.approx(0.0)
    assert step.info["l2_raw_decrease_qty"] == pytest.approx(1.0)
    assert step.info["l2_effective_decrease_qty"] == pytest.approx(1.0)

    assert len(env._state.live_orders) == 1
    order = env._state.live_orders[0]
    assert order.price_tick == 1001
    assert order.queue_ahead_qty == pytest.approx(2.0)


def test_balanced_mode_trade_l2_dedupe_enabled_vs_disabled():
    l2_events = [
        _l2(
            seq=0,
            local_ts_us=100,
            bid_ticks=(1001,),
            bid_sizes=(1.0,),
            ask_ticks=(1003,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=1,
            local_ts_us=160,
            bid_ticks=(1001,),
            bid_sizes=(0.5,),
            ask_ticks=(1003,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=2,
            local_ts_us=300,
            bid_ticks=(1001,),
            bid_sizes=(0.5,),
            ask_ticks=(1003,),
            ask_sizes=(1.0,),
        ),
    ]
    trades = [
        _trade(local_ts_us=150, side=AggressorSide.SELL, price_tick=1001, amount=0.5, source_row=0),
    ]
    tape = _tape(l2_events, trades)

    enabled_config = _env_config(
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=QueueModelMode.BALANCED,
                l2_decrease_weight=1.0,
                trade_at_level_weight=1.0,
                dedupe_l2_decrease_with_trade_prints=True,
            ),
            maker_fee_bps=0.0,
        )
    )
    enabled_env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=enabled_config)
    enabled_env.reset()
    enabled_step = enabled_env.step(QuoteAction(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_raw=-100.0,
        ask_price_raw=0.0,
        bid_size_raw=100.0,
        ask_size_raw=0.0,
    ))

    disabled_config = _env_config(
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=QueueModelMode.BALANCED,
                l2_decrease_weight=1.0,
                trade_at_level_weight=1.0,
                dedupe_l2_decrease_with_trade_prints=False,
            ),
            maker_fee_bps=0.0,
        )
    )
    disabled_env = ExecutionEnv(tape, linear_signals=_linear_signals(tape), config=disabled_config)
    disabled_env.reset()
    disabled_step = disabled_env.step(QuoteAction(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_raw=-100.0,
        ask_price_raw=0.0,
        bid_size_raw=100.0,
        ask_size_raw=0.0,
    ))

    assert enabled_step.info["queue_trade_advance_qty"] > 0.0
    assert enabled_step.info["l2_trade_dedupe_qty"] == pytest.approx(0.5)
    assert enabled_step.info["l2_effective_decrease_qty"] == pytest.approx(0.0)
    assert disabled_step.info["l2_trade_dedupe_qty"] == pytest.approx(0.0)
    assert disabled_step.info["l2_effective_decrease_qty"] == pytest.approx(0.5)


def test_same_side_replacement_after_cancel_does_not_fill_later_same_timestamp_event():
    l2_events = [
        _l2(seq=0, local_ts_us=100, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
        _l2(seq=1, local_ts_us=200, bid_ticks=(1000,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
        _l2(seq=2, local_ts_us=500, bid_ticks=(999,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
        _l2(seq=3, local_ts_us=600, bid_ticks=(999,), bid_sizes=(1.0,), ask_ticks=(1002,), ask_sizes=(1.0,)),
    ]
    trades = [
        _trade(
            local_ts_us=500,
            side=AggressorSide.SELL,
            price_tick=999,
            amount=10.0,
            source_row=0,
        ),
    ]
    tape = _tape(l2_events, trades)

    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 1, 2, 4], [100, 200, 500, 600]),
        config=_env_config(
            decision_interval_us=100,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=20,
                cancel_latency_us=300,
            ),
        ),
    )
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["orders_active_count"] == 1

    replacement_bid = QuoteAction(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_raw=-100.0,
        ask_price_raw=0.0,
        bid_size_raw=200.0,
        ask_size_raw=0.0,
    )

    second = env.step(replacement_bid)
    assert second.fills == ()

    third = env.step(replacement_bid)

    assert all(fill.local_ts_us != 500 for fill in third.fills)


def test_same_side_replacement_does_not_overlap_when_entry_latency_shorter_than_cancel_latency():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
        _l2(seq=3, local_ts_us=400),
        _l2(seq=4, local_ts_us=600),
    ]
    tape = _tape(l2_events, [])
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals(tape),
        config=_env_config(
            decision_interval_us=50,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=20,
                cancel_latency_us=300,
            ),
        ),
    )
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["orders_active_count"] == 1

    larger_bid = QuoteAction(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_raw=-100.0,
        ask_price_raw=0.0,
        bid_size_raw=200.0,
        ask_size_raw=0.0,
    )
    env.step(larger_bid)

    live = tuple(env._state.live_orders)
    bid_orders = [order for order in live if order.side == OrderSide.BUY]
    assert len(bid_orders) == 2

    old = next(order for order in bid_orders if order.status.name == "PENDING_CANCEL")
    new = next(order for order in bid_orders if order.status.name == "PENDING_NEW")

    assert new.effective_key == EventKey(old.cancel_effective_key.local_ts_us, MAX_EVENT_SEQ)

    for key in (EventKey(250, 0), EventKey(300, 0), EventKey(399, MAX_EVENT_SEQ)):
        assert sum(order.is_fillable_at_key(key) for order in bid_orders) <= 1


def test_l2_decrease_for_pending_new_only_level_not_counted_in_diagnostics():
    l2_events = [
        _l2(
            seq=0,
            local_ts_us=100,
            bid_ticks=(1000,),
            bid_sizes=(1.0,),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=1,
            local_ts_us=150,
            bid_ticks=(1001, 1000),
            bid_sizes=(3.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=2,
            local_ts_us=160,
            bid_ticks=(1001, 1000),
            bid_sizes=(2.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
        _l2(
            seq=3,
            local_ts_us=300,
            bid_ticks=(1001, 1000),
            bid_sizes=(2.0, 1.0),
            ask_ticks=(1002,),
            ask_sizes=(1.0,),
        ),
    ]
    tape = _tape(l2_events, [])
    env = ExecutionEnv(
        tape,
        linear_signals=_linear_signals_for_decisions(tape, [0, 3], [100, 300]),
        config=_env_config(
            decision_interval_us=300,
            latency_config=LatencyConfig(
                decision_compute_latency_us=0,
                order_entry_latency_us=120,
                cancel_latency_us=0,
            ),
            fill_simulator_config=FillSimulatorConfig(
                queue_model=QueueModelConfig(
                    mode=QueueModelMode.BALANCED,
                    l2_decrease_weight=1.0,
                    trade_at_level_weight=1.0,
                ),
                maker_fee_bps=0.0,
            ),
        ),
    )
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert step.info["l2_raw_decrease_qty"] == pytest.approx(0.0)
    assert step.info["l2_effective_decrease_qty"] == pytest.approx(0.0)


def _aligned_adverse_signals(linear: LinearSignalArtifact, candidate_names=("touch", "inside_1", "away_1")) -> AdverseSelectionSignalArtifact:
    target_names = []
    predictions = {}
    n_rows = linear.n_rows
    for candidate in candidate_names:
        for side in ("bid", "ask"):
            fill_name = f"{side}_{candidate}_filled"
            cost_name = f"{side}_{candidate}_toxic_cost_bps"
            target_names.extend([fill_name, cost_name])
            predictions[fill_name] = np.full(n_rows, 0.5, dtype=np.float32)
            predictions[cost_name] = np.zeros(n_rows, dtype=np.float32)
    return AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=linear.decision_local_ts_us.copy(),
        decision_event_index=linear.decision_event_index.copy(),
        decision_event_seq=np.full(n_rows, MAX_EVENT_SEQ, dtype=np.int64),
        target_names=tuple(target_names),
        predictions=predictions,
    )


def test_env_default_adverse_runtime_config_inherits_quote_geometry_post_only_gap():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    linear = _linear_signals(tape, n_rows=3)
    adverse = _aligned_adverse_signals(linear)
    config = _env_config(
        quote_geometry_config=QuoteGeometryConfig(post_only_gap_ticks=2, default_order_qty=1.0),
        fill_simulator_config=FillSimulatorConfig(
            queue_model=QueueModelConfig(
                mode=QueueModelMode.BALANCED,
                l2_decrease_weight=1.0,
                trade_at_level_weight=1.0,
            ),
            maker_fee_bps=-0.25,
        ),
    )

    env = ExecutionEnv(tape, linear_signals=linear, adverse_signals=adverse, config=config)

    assert env.config.adverse_runtime_config is not None
    assert env.config.adverse_runtime_config.post_only_gap_ticks == 2
    assert env.config.adverse_runtime_config.executable_edge.maker_fee_bps == env.config.fill_simulator_config.maker_fee_bps

    env.reset()
    step = env.step(_disabled_action())
    assert step.info["adverse_runtime_post_only_gap_ticks"] == 2


def test_adverse_signals_do_not_change_fills_rewards_or_position_for_same_actions():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
        _l2(seq=3, local_ts_us=400),
    ]
    trades = [
        _trade(local_ts_us=150, side=AggressorSide.SELL, price_tick=1000, amount=2.0, source_row=0),
        _trade(local_ts_us=250, side=AggressorSide.BUY, price_tick=1002, amount=2.0, source_row=1),
    ]
    tape = _tape(l2_events, trades)
    linear = _linear_signals(tape, n_rows=4)
    adverse = _aligned_adverse_signals(linear)

    config = _env_config(max_episode_steps=3)
    env_plain = ExecutionEnv(tape, linear_signals=linear, config=config)
    env_adv = ExecutionEnv(tape, linear_signals=linear, adverse_signals=adverse, config=config)

    env_plain.reset()
    env_adv.reset()

    actions = [_bid_only_action(), _ask_only_action(), _disabled_action()]
    for action in actions:
        plain_step = env_plain.step(action)
        adv_step = env_adv.step(action)

        assert plain_step.reward == pytest.approx(adv_step.reward)
        assert plain_step.position == adv_step.position
        assert [(f.side, f.local_ts_us, f.price_tick, f.qty) for f in plain_step.fills] == [
            (f.side, f.local_ts_us, f.price_tick, f.qty) for f in adv_step.fills
        ]
        assert adv_step.info["adverse_signal_available"] is True
