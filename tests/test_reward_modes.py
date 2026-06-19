import numpy as np
import pytest

from mmrt.execution.contracts import Fill, FillReason, OrderSide
from mmrt.rl.reward_modes import (
    FifoLotLedger,
    HorizonRewardProjector,
    RewardAnchor,
    TrainingRewardConfig,
    TrainingRewardMode,
    future_row_for_ts,
)


def _fill(
    *,
    side: OrderSide,
    local_ts_us: int,
    price_tick: int,
    qty: float = 1.0,
    fee: float = 0.0,
) -> Fill:
    return Fill(
        order_id=1,
        side=side,
        local_ts_us=local_ts_us,
        event_seq=0,
        price_tick=price_tick,
        qty=qty,
        fee=fee,
        reason=FillReason.TRADE_THROUGH,
    )


def _anchor(
    *,
    t_index: int = 0,
    decision_row: int = 0,
    next_decision_row: int = 1,
    decision_local_ts_us: int = 1,
    next_decision_local_ts_us: int = 250_001,
    range_end_row: int = 5,
    previous_equity: float = 0.0,
    current_equity: float = 0.0,
    env_reward: float = 0.0,
    fills: tuple[Fill, ...] = (),
    inventory_after_step: float = 0.0,
    current_mid_after_step: float = 100.0,
    realized_lot_pnl: float = 0.0,
) -> RewardAnchor:
    return RewardAnchor(
        t_index=t_index,
        env_index=0,
        episode_id=0,
        decision_row=decision_row,
        next_decision_row=next_decision_row,
        decision_local_ts_us=decision_local_ts_us,
        next_decision_local_ts_us=next_decision_local_ts_us,
        range_end_row=range_end_row,
        previous_equity=previous_equity,
        current_equity=current_equity,
        env_reward=env_reward,
        fills=fills,
        inventory_after_step=inventory_after_step,
        current_mid_after_step=current_mid_after_step,
        realized_lot_pnl=realized_lot_pnl,
    )


def _projector(mode: str | TrainingRewardMode, **kwargs) -> HorizonRewardProjector:
    config = TrainingRewardConfig(training_reward_mode=mode, **kwargs)
    return HorizonRewardProjector(
        decision_local_ts_us=np.array([1, 250_001, 500_001, 1_000_001, 1_500_001], dtype=np.int64),
        mid_prices=np.array([100.0, 100.5, 101.0, 102.0, 99.0], dtype=np.float64),
        tick_size=1.0,
        contract_size=1.0,
        config=config,
    )


def test_training_reward_config_validation():
    assert TrainingRewardConfig().mode is TrainingRewardMode.EQUITY_DELTA
    assert TrainingRewardConfig(training_reward_mode="horizon_path_equity").mode is TrainingRewardMode.HORIZON_PATH_EQUITY

    with pytest.raises(ValueError, match="reward_horizon_us"):
        TrainingRewardConfig(reward_horizon_us=0)
    with pytest.raises(ValueError, match="horizon_path_weight"):
        TrainingRewardConfig(horizon_path_weight=-1.0)
    with pytest.raises(ValueError, match="horizon_path_weight \\+ fill_markout_weight"):
        TrainingRewardConfig(
            training_reward_mode="horizon_blend",
            horizon_path_weight=0.0,
            fill_markout_weight=0.0,
        )
    with pytest.raises(ValueError, match="realized_lot_weight"):
        TrainingRewardConfig(
            training_reward_mode="realized_lot_horizon",
            realized_lot_weight=0.0,
            unrealized_horizon_weight=0.0,
            fill_markout_weight=0.0,
        )
    with pytest.raises(ValueError, match="sorted"):
        TrainingRewardConfig(multi_horizon_us=(500_000, 250_000))
    with pytest.raises(ValueError, match="unique"):
        TrainingRewardConfig(multi_horizon_us=(250_000, 250_000))
    with pytest.raises(ValueError, match="length"):
        TrainingRewardConfig(multi_horizon_weights=(1.0,))
    with pytest.raises(ValueError, match="sum"):
        TrainingRewardConfig(multi_horizon_weights=(0.0, 0.0, 0.0))


def test_future_row_for_ts_uses_left_search_and_range_end():
    ts = np.array([0, 500_000, 1_000_000, 1_000_000, 1_500_000], dtype=np.int64)

    assert future_row_for_ts(ts, 0, 1_000_000, 5) == 2
    assert future_row_for_ts(ts, 1, 1_000_000, 5) == 4
    assert future_row_for_ts(ts, 500_000, 1_000_000, 4) is None
    assert future_row_for_ts(ts, 1_000_000, 1_000_000, 5) is None


def test_horizon_path_equity_projects_actual_path_delta():
    projector = _projector("horizon_path_equity", reward_horizon_us=1_000_000)
    anchors = (
        _anchor(
            t_index=0,
            decision_row=0,
            next_decision_row=1,
            decision_local_ts_us=1,
            next_decision_local_ts_us=500_001,
            previous_equity=0.0,
            current_equity=1.0,
        ),
        _anchor(
            t_index=1,
            decision_row=2,
            next_decision_row=3,
            decision_local_ts_us=500_001,
            next_decision_local_ts_us=1_000_001,
            previous_equity=1.0,
            current_equity=3.0,
        ),
        _anchor(
            t_index=2,
            decision_row=3,
            next_decision_row=4,
            decision_local_ts_us=1_000_001,
            next_decision_local_ts_us=1_500_001,
            range_end_row=4,
            previous_equity=3.0,
            current_equity=6.0,
        ),
    )
    equity = {(0, 0): {0: 0.0, 1: 1.0, 3: 3.0, 4: 6.0}}

    result = projector.project(anchors, equity, shape=(3, 1))

    assert result.valid_mask[:, 0].tolist() == [True, True, False]
    assert result.projected_rewards[0, 0] == pytest.approx(3.0)
    assert result.projected_rewards[1, 0] == pytest.approx(5.0)
    assert result.components["path_equity_delta_H"][0, 0] == pytest.approx(3.0)


def test_fill_markout_horizon_scores_fills_and_preserves_no_fill_zero():
    projector = _projector("fill_markout_horizon", reward_horizon_us=1_000_000)
    anchors = (
        _anchor(
            t_index=0,
            fills=(_fill(side=OrderSide.BUY, local_ts_us=1, price_tick=100, fee=0.1),),
        ),
        _anchor(
            t_index=1,
            decision_row=2,
            next_decision_row=3,
            decision_local_ts_us=500_001,
            next_decision_local_ts_us=1_000_001,
            fills=(_fill(side=OrderSide.SELL, local_ts_us=500_001, price_tick=100, fee=-0.05),),
        ),
        _anchor(t_index=2, decision_row=3, next_decision_row=4, decision_local_ts_us=1_000_001),
    )

    result = projector.project(anchors, {(0, 0): {3: 0.0, 4: 0.0}}, shape=(3, 1))

    assert result.valid_mask[:, 0].tolist() == [True, True, True]
    assert result.projected_rewards[0, 0] == pytest.approx(1.9)
    assert result.projected_rewards[1, 0] == pytest.approx(1.05)
    assert result.projected_rewards[2, 0] == pytest.approx(0.0)
    assert result.components["fill_count_scored"][0, 0] == pytest.approx(1.0)

    unavailable = projector.project(
        (
            _anchor(
                t_index=0,
                range_end_row=3,
                fills=(_fill(side=OrderSide.BUY, local_ts_us=500_001, price_tick=100),),
            ),
        ),
        {(0, 0): {}},
        shape=(1, 1),
    )
    assert not bool(unavailable.valid_mask[0, 0])
    assert unavailable.stats["unavailable_reason_counts"] == {"fill_future_row_unavailable": 1}


def test_fill_markout_horizon_requires_same_episode_future_row():
    projector = _projector("fill_markout_horizon", reward_horizon_us=1_000_000)
    anchor = _anchor(
        fills=(_fill(side=OrderSide.BUY, local_ts_us=1, price_tick=100, fee=0.1),),
    )

    result = projector.project((anchor,), {(0, 0): {}}, shape=(1, 1))

    assert not bool(result.valid_mask[0, 0])
    assert result.projected_rewards[0, 0] == pytest.approx(0.0)
    assert result.components["fill_count_scored"][0, 0] == pytest.approx(0.0)
    assert result.components["fill_count_unavailable"][0, 0] == pytest.approx(1.0)
    assert result.stats["unavailable_reason_counts"] == {"fill_future_episode_row_unavailable": 1}


def test_fill_markout_horizon_no_fill_remains_valid_without_future_row():
    projector = _projector("fill_markout_horizon", reward_horizon_us=1_000_000)
    anchor = _anchor(fills=())

    result = projector.project((anchor,), {(0, 0): {}}, shape=(1, 1))

    assert bool(result.valid_mask[0, 0])
    assert result.projected_rewards[0, 0] == pytest.approx(0.0)
    assert result.components["fill_count_scored"][0, 0] == pytest.approx(0.0)
    assert result.components["fill_count_unavailable"][0, 0] == pytest.approx(0.0)


def test_horizon_blend_weights_path_and_fill_once():
    projector = _projector(
        "horizon_blend",
        reward_horizon_us=1_000_000,
        reward_scale=3.0,
        horizon_path_weight=2.0,
        fill_markout_weight=0.5,
    )
    anchor = _anchor(
        fills=(_fill(side=OrderSide.BUY, local_ts_us=1, price_tick=100, fee=0.1),),
        previous_equity=0.0,
    )
    equity = {(0, 0): {0: 0.0, 3: 3.0}}

    result = projector.project((anchor,), equity, shape=(1, 1))

    assert result.valid_mask[0, 0]
    assert result.components["path_equity_delta_H"][0, 0] == pytest.approx(3.0)
    assert result.components["fill_markout_pnl_H"][0, 0] == pytest.approx(1.9)
    assert result.components["blended_reward_raw"][0, 0] == pytest.approx(6.95)
    assert result.projected_rewards[0, 0] == pytest.approx(20.85)


def test_horizon_blend_invalidates_missing_same_episode_fill_horizon():
    projector = _projector(
        "horizon_blend",
        reward_horizon_us=1_000_000,
        horizon_path_weight=1.0,
        fill_markout_weight=1.0,
    )
    anchor = _anchor(
        fills=(_fill(side=OrderSide.BUY, local_ts_us=250_001, price_tick=100, fee=0.1),),
        previous_equity=0.0,
    )

    result = projector.project((anchor,), {(0, 0): {3: 3.0}}, shape=(1, 1))

    assert not bool(result.valid_mask[0, 0])
    assert result.projected_rewards[0, 0] == pytest.approx(0.0)
    assert "blended_reward_raw" not in result.components
    assert result.stats["unavailable_reason_counts"] == {"fill_future_episode_row_unavailable": 1}


def test_horizon_potential_shaped_uses_phi_delta_and_invalidates_missing_phi():
    projector = _projector("horizon_potential_shaped", reward_horizon_us=1_000_000)
    anchor = _anchor(
        env_reward=0.2,
        previous_equity=0.0,
        current_equity=0.5,
        decision_local_ts_us=1,
        next_decision_local_ts_us=500_001,
    )
    equity = {(0, 0): {0: 0.0, 1: 0.5, 3: 1.0, 4: 1.9}}

    result = projector.project((anchor,), equity, shape=(1, 1))

    assert result.valid_mask[0, 0]
    assert result.components["horizon_phi_t"][0, 0] == pytest.approx(1.0)
    assert result.components["horizon_phi_next"][0, 0] == pytest.approx(1.4)
    assert result.projected_rewards[0, 0] == pytest.approx(0.6)

    invalid = projector.project((anchor,), {(0, 0): {0: 0.0, 1: 0.5, 3: 1.0}}, shape=(1, 1))
    assert not bool(invalid.valid_mask[0, 0])


def test_fifo_lot_ledger_and_realized_lot_projection():
    ledger = FifoLotLedger(tick_size=1.0, contract_size=1.0)
    assert ledger.apply_fill(_fill(side=OrderSide.BUY, local_ts_us=1, price_tick=100, qty=2.0, fee=0.2)) == 0.0
    realized = ledger.apply_fill(_fill(side=OrderSide.SELL, local_ts_us=2, price_tick=101, qty=1.0, fee=0.1))
    assert realized == pytest.approx(0.8)
    assert ledger.lots[0][0] == "long"
    assert ledger.lots[0][1:] == pytest.approx((1.0, 100.0, 0.1))

    rebate_realized = ledger.apply_fill(_fill(side=OrderSide.SELL, local_ts_us=3, price_tick=101, qty=1.0, fee=-0.1))
    assert rebate_realized == pytest.approx(1.0)
    assert ledger.lots == ()

    projector = _projector(
        "realized_lot_horizon",
        reward_horizon_us=1_000_000,
        realized_lot_weight=1.0,
        unrealized_horizon_weight=1.0,
        fill_markout_weight=0.5,
    )
    anchor = _anchor(
        decision_row=1,
        next_decision_row=2,
        decision_local_ts_us=500_001,
        next_decision_local_ts_us=1_000_001,
        fills=(_fill(side=OrderSide.SELL, local_ts_us=500_001, price_tick=101, qty=1.0, fee=0.1),),
        inventory_after_step=1.0,
        current_mid_after_step=101.0,
        realized_lot_pnl=0.8,
    )

    result = projector.project((anchor,), {(0, 0): {4: 0.0}}, shape=(1, 1))

    assert result.valid_mask[0, 0]
    assert result.components["realized_lot_pnl"][0, 0] == pytest.approx(0.8)
    assert result.components["unrealized_horizon_carry_pnl"][0, 0] == pytest.approx(-2.0)
    assert result.components["fill_markout_pnl_H"][0, 0] == pytest.approx(1.9)
    assert result.projected_rewards[0, 0] == pytest.approx(-0.25)


def test_realized_lot_horizon_requires_anchor_future_row():
    projector = _projector("realized_lot_horizon", reward_horizon_us=1_000_000)
    anchor = _anchor(fills=())

    result = projector.project((anchor,), {(0, 0): {}}, shape=(1, 1))

    assert not bool(result.valid_mask[0, 0])
    assert result.stats["unavailable_reason_counts"] == {
        "realized_lot_future_episode_row_unavailable": 1
    }


def test_realized_lot_horizon_requires_fill_future_row_when_fills_exist():
    projector = _projector("realized_lot_horizon", reward_horizon_us=1_000_000)
    anchor = _anchor(
        fills=(_fill(side=OrderSide.SELL, local_ts_us=250_001, price_tick=101, qty=1.0, fee=0.1),),
    )

    result = projector.project((anchor,), {(0, 0): {3: 0.0}}, shape=(1, 1))

    assert not bool(result.valid_mask[0, 0])
    assert result.components["fill_count_scored"][0, 0] == pytest.approx(0.0)
    assert result.components["fill_count_unavailable"][0, 0] == pytest.approx(1.0)
    assert result.stats["unavailable_reason_counts"] == {"fill_future_episode_row_unavailable": 1}


def test_realized_lot_horizon_no_fill_still_requires_anchor_future_row():
    projector = _projector("realized_lot_horizon", reward_horizon_us=1_000_000)
    anchor = _anchor(
        fills=(),
        inventory_after_step=1.0,
        current_mid_after_step=101.0,
        realized_lot_pnl=0.5,
    )

    valid = projector.project((anchor,), {(0, 0): {3: 0.0}}, shape=(1, 1))
    invalid = projector.project((anchor,), {(0, 0): {}}, shape=(1, 1))

    assert bool(valid.valid_mask[0, 0])
    assert valid.projected_rewards[0, 0] == pytest.approx(1.5)
    assert not bool(invalid.valid_mask[0, 0])


def test_multi_horizon_path_weighted_sum_and_missing_horizon_invalid():
    projector = _projector(
        "multi_horizon_path",
        multi_horizon_us=(250_000, 500_000, 1_000_000),
        multi_horizon_weights=(0.25, 0.25, 1.0),
    )
    anchor = _anchor(previous_equity=0.0)
    equity = {(0, 0): {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}}

    result = projector.project((anchor,), equity, shape=(1, 1))

    assert result.valid_mask[0, 0]
    assert result.components["path_equity_delta_250000"][0, 0] == pytest.approx(1.0)
    assert result.components["path_equity_delta_500000"][0, 0] == pytest.approx(2.0)
    assert result.components["path_equity_delta_1000000"][0, 0] == pytest.approx(4.0)
    assert result.components["multi_horizon_path_reward_raw"][0, 0] == pytest.approx(4.75)
    assert result.projected_rewards[0, 0] == pytest.approx(4.75)

    invalid = projector.project((anchor,), {(0, 0): {0: 0.0, 1: 1.0, 2: 2.0}}, shape=(1, 1))
    assert not bool(invalid.valid_mask[0, 0])
