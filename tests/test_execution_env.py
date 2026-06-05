from pathlib import Path

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActionSpec,
    BookLevelSnapshot,
    BookTop,
    OrderSide,
    PositionState,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape
from mmrt.execution.env import (
    ExecutionEnv,
    ExecutionEnvConfig,
    ExecutionEnvReset,
    ExecutionEnvStep,
    action_array_to_continuous_action,
)
from mmrt.execution.fill_sim import FillSimulatorConfig
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.execution.queue_model import QueueModelConfig, QueueModelMode
from mmrt.execution.quote_geometry import ContinuousQuoteAction, QuoteGeometryConfig
from mmrt.execution.reward import RewardConfig


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


def _env_config(**kwargs) -> ExecutionEnvConfig:
    base = dict(
        decision_interval_us=100,
        action_spec=ActionSpec(max_distance_ticks=1, max_order_qty=1.0),
        quote_geometry_config=QuoteGeometryConfig(
            min_distance_ticks=1,
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
        reward_config=RewardConfig(),
    )
    base.update(kwargs)
    return ExecutionEnvConfig(**base)


def _bid_only_action() -> ContinuousQuoteAction:
    return ContinuousQuoteAction(
        bid_enable_logit=1.0,
        ask_enable_logit=-1.0,
        bid_distance_raw=0.0,
        ask_distance_raw=0.0,
        bid_size_raw=100.0,
        ask_size_raw=0.0,
    )


def _ask_only_action() -> ContinuousQuoteAction:
    return ContinuousQuoteAction(
        bid_enable_logit=-1.0,
        ask_enable_logit=1.0,
        bid_distance_raw=0.0,
        ask_distance_raw=0.0,
        bid_size_raw=0.0,
        ask_size_raw=100.0,
    )


def _disabled_action() -> ContinuousQuoteAction:
    return ContinuousQuoteAction(
        bid_enable_logit=-1.0,
        ask_enable_logit=-1.0,
        bid_distance_raw=0.0,
        ask_distance_raw=0.0,
        bid_size_raw=0.0,
        ask_size_raw=0.0,
    )


def test_action_array_to_continuous_action():
    action = action_array_to_continuous_action(np.array([1, -1, 0, 0, 2, -2], dtype=np.float32))

    assert isinstance(action, ContinuousQuoteAction)
    assert action.bid_enable_logit == pytest.approx(1.0)
    assert action.ask_enable_logit == pytest.approx(-1.0)

    with pytest.raises(ValueError):
        action_array_to_continuous_action([1.0, 2.0])

    with pytest.raises(ValueError):
        action_array_to_continuous_action([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")])


def test_reset_returns_initial_observation():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, config=_env_config())

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
    env = ExecutionEnv(tape, config=_env_config())
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
    env = ExecutionEnv(tape, config=_env_config())
    env.reset()

    step = env.step(_bid_only_action())

    assert len(step.fills) == 1
    fill = step.fills[0]
    assert fill.side == OrderSide.BUY
    assert fill.price_tick == 1000
    assert fill.qty == pytest.approx(1.0)
    assert step.position.inventory_qty == pytest.approx(1.0)
    assert step.position.cash == pytest.approx(-100.0)

    assert step.execution.reward.raw_equity_delta == pytest.approx(0.1)
    assert step.reward == pytest.approx(0.1)


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
    env = ExecutionEnv(tape, config=_env_config())
    env.reset()

    step = env.step(_ask_only_action())

    assert len(step.fills) == 1
    assert step.fills[0].side == OrderSide.SELL
    assert step.position.inventory_qty == pytest.approx(-1.0)
    assert step.position.cash == pytest.approx(100.2)


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
    env = ExecutionEnv(tape, config=_env_config())
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert step.position == PositionState()
    assert len(env._state.live_orders) == 1
    order = env._state.live_orders[0]
    assert order.side == OrderSide.BUY
    assert order.price_tick == 1000
    assert order.queue_ahead_qty == pytest.approx(0.0)
    assert order.remaining_qty == pytest.approx(1.0)


def test_repeated_decision_cancels_previous_live_order():
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=300),
    ]
    tape = _tape(l2_events, [])
    env = ExecutionEnv(tape, config=_env_config(decision_interval_us=50))
    env.reset()

    first = env.step(_bid_only_action())
    assert first.info["cancel_count"] == 0

    second = env.step(_bid_only_action())
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
    env = ExecutionEnv(tape, config=_env_config(max_episode_steps=1))
    env.reset()

    step = env.step(_disabled_action())

    assert step.truncated is True
    assert step.done is False


def test_step_after_terminal_rejected():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, config=_env_config(max_episode_steps=1))
    env.reset()
    env.step(_disabled_action())

    with pytest.raises(RuntimeError):
        env.step(_disabled_action())


def test_linear_signal_rows_used_by_step_index():
    from mmrt.execution.linear_signal import LinearSignalArrays

    signals = LinearSignalArrays(
        p_no_move=np.array([0.1, 0.2], dtype=np.float32),
        p_up=np.array([0.6, 0.7], dtype=np.float32),
        mag_up_bps=np.array([1.0, 2.0], dtype=np.float32),
        mag_down_bps=np.array([3.0, 4.0], dtype=np.float32),
        expected_return_bps=np.array([5.0, 6.0], dtype=np.float32),
        confidence=np.array([0.11, 0.22], dtype=np.float32),
    )
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, config=_env_config(), linear_signals=signals)

    reset = env.reset()
    idx = env.config.observation_schema.index("linear_p_no_move")
    assert reset.observation[idx] == pytest.approx(0.1)

    step = env.step(_disabled_action())
    assert step.observation[idx] == pytest.approx(0.2)


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
    env = ExecutionEnv(tape, config=_env_config())

    with pytest.raises(ValueError, match="valid two-sided"):
        env.reset()


def test_env_step_info_is_json_safe_and_validated_by_contract():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200)],
        [],
    )
    env = ExecutionEnv(tape, config=_env_config())
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
    env = ExecutionEnv(tape, config=_env_config())

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
    env = ExecutionEnv(tape, config=_env_config())
    env.reset()

    step = env.step(_bid_only_action())

    assert step.fills == ()
    assert step.position == PositionState()
