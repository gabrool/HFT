from pathlib import Path

import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import ActiveOrder, FillReason, OrderSide, OrderStatus, QueueModelMode, TradePrint
from mmrt.time_key import EventKey
from mmrt.execution.queue_model import (
    QueueModelConfig,
    QueueModelUpdate,
    classify_trade_against_order,
    estimate_initial_queue_ahead,
    is_trade_relevant_to_order,
    update_queue_position,
)


def _order(
    *,
    side: OrderSide = OrderSide.BUY,
    price_tick: int = 1000,
    qty: float = 1.0,
    remaining_qty: float = 1.0,
    queue_ahead_qty: float = 2.0,
    status: OrderStatus = OrderStatus.ACTIVE,
) -> ActiveOrder:
    return ActiveOrder(
        order_id=1,
        side=side,
        price_tick=price_tick,
        qty=qty,
        remaining_qty=remaining_qty,
        queue_ahead_qty=queue_ahead_qty,
        status=status,
        created_local_ts_us=100,
        created_event_seq=0,
        last_update_local_ts_us=100,
        last_update_event_seq=0,
    )


def _trade(
    *,
    side: AggressorSide = AggressorSide.SELL,
    price_tick: int = 1000,
    amount: float = 1.0,
) -> TradePrint:
    return TradePrint(
        local_ts_us=200,
        ts_us=190,
        side=side,
        price_tick=price_tick,
        amount=amount,
        trade_id="t",
        source_row=0,
    )


def test_queue_model_config_validation():
    assert QueueModelConfig(mode="balanced").mode == QueueModelMode.BALANCED

    with pytest.raises(ValueError):
        QueueModelConfig(l2_decrease_weight=-0.1)

    with pytest.raises(ValueError):
        QueueModelConfig(l2_decrease_weight=1.1)

    with pytest.raises(ValueError):
        QueueModelConfig(trade_at_level_weight=1.1)

    with pytest.raises(ValueError):
        QueueModelConfig(unknown_level_queue_ahead_qty=-1.0)

    with pytest.raises(ValueError):
        QueueModelConfig(qty_epsilon=0.0)


def test_estimate_initial_queue_ahead():
    assert estimate_initial_queue_ahead(2.5) == 2.5
    assert estimate_initial_queue_ahead(None) == 1_000_000_000.0
    assert estimate_initial_queue_ahead(None, config=QueueModelConfig(unknown_level_queue_ahead_qty=3.0)) == 3.0

    with pytest.raises(ValueError):
        estimate_initial_queue_ahead(-1.0)


def test_trade_relevance_by_side():
    bid = _order(side=OrderSide.BUY)
    ask = _order(side=OrderSide.SELL)

    assert is_trade_relevant_to_order(bid, _trade(side=AggressorSide.SELL))
    assert not is_trade_relevant_to_order(bid, _trade(side=AggressorSide.BUY))

    assert is_trade_relevant_to_order(ask, _trade(side=AggressorSide.BUY))
    assert not is_trade_relevant_to_order(ask, _trade(side=AggressorSide.SELL))


def test_classify_trade_against_bid_order():
    bid = _order(side=OrderSide.BUY, price_tick=1000)

    assert classify_trade_against_order(bid, _trade(side=AggressorSide.SELL, price_tick=999)) == FillReason.TRADE_THROUGH
    assert classify_trade_against_order(bid, _trade(side=AggressorSide.SELL, price_tick=1000)) == FillReason.TRADE_AT_LEVEL
    assert classify_trade_against_order(bid, _trade(side=AggressorSide.SELL, price_tick=1001)) is None
    assert classify_trade_against_order(bid, _trade(side=AggressorSide.BUY, price_tick=999)) is None


def test_classify_trade_against_ask_order():
    ask = _order(side=OrderSide.SELL, price_tick=1002)

    assert classify_trade_against_order(ask, _trade(side=AggressorSide.BUY, price_tick=1003)) == FillReason.TRADE_THROUGH
    assert classify_trade_against_order(ask, _trade(side=AggressorSide.BUY, price_tick=1002)) == FillReason.TRADE_AT_LEVEL
    assert classify_trade_against_order(ask, _trade(side=AggressorSide.BUY, price_tick=1001)) is None
    assert classify_trade_against_order(ask, _trade(side=AggressorSide.SELL, price_tick=1003)) is None


def test_trade_at_level_advances_queue_without_fill_when_queue_remains():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0, remaining_qty=1.0),
        config=QueueModelConfig(mode=QueueModelMode.BALANCED, trade_at_level_weight=1.0),
        trade=_trade(side=AggressorSide.SELL, price_tick=1000, amount=0.5),
    )

    assert isinstance(update, QueueModelUpdate)
    assert update.queue_ahead_before == 2.0
    assert update.queue_ahead_after == 1.5
    assert update.trade_advance_qty == 0.5
    assert update.l2_advance_qty == 0.0
    assert update.fillable_qty == 0.0
    assert update.fill_reason is None
    assert update.trade_at_level is True


def test_trade_at_level_leftover_becomes_fillable():
    update = update_queue_position(
        _order(queue_ahead_qty=0.5, remaining_qty=1.0),
        config=QueueModelConfig(mode=QueueModelMode.BALANCED, trade_at_level_weight=1.0),
        trade=_trade(side=AggressorSide.SELL, price_tick=1000, amount=0.75),
    )

    assert update.queue_ahead_after == 0.0
    assert update.trade_advance_qty == 0.5
    assert update.fillable_qty == 0.25
    assert update.fill_reason == FillReason.TRADE_AT_LEVEL


def test_trade_through_makes_remaining_qty_fillable():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0, remaining_qty=0.7),
        trade=_trade(side=AggressorSide.SELL, price_tick=999, amount=0.01),
    )

    assert update.trade_through is True
    assert update.queue_ahead_after == 0.0
    assert update.trade_advance_qty == 2.0
    assert update.fillable_qty == 0.7
    assert update.fill_reason == FillReason.TRADE_THROUGH


def test_irrelevant_trade_leaves_queue_unchanged():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0),
        trade=_trade(side=AggressorSide.BUY, price_tick=1000),
    )

    assert update.queue_ahead_after == 2.0
    assert update.advanced_qty == 0.0
    assert update.fillable_qty == 0.0
    assert update.fill_reason is None


def test_conservative_mode_ignores_l2_decreases():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0),
        config=QueueModelConfig(mode=QueueModelMode.CONSERVATIVE),
        l2_decrease_qty=2.0,
    )

    assert update.queue_ahead_after == 2.0
    assert update.l2_advance_qty == 0.0
    assert update.fillable_qty == 0.0


def test_balanced_mode_l2_decrease_advances_queue():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0, remaining_qty=1.0),
        config=QueueModelConfig(mode=QueueModelMode.BALANCED, l2_decrease_weight=0.5),
        l2_decrease_qty=1.0,
    )

    assert update.queue_ahead_after == 1.5
    assert update.l2_advance_qty == 0.5
    assert update.fillable_qty == 0.0


def test_balanced_l2_decrease_beyond_queue_creates_fillable_signal():
    update = update_queue_position(
        _order(queue_ahead_qty=0.5, remaining_qty=1.0),
        config=QueueModelConfig(mode=QueueModelMode.BALANCED, l2_decrease_weight=1.0, trade_at_level_weight=1.0),
        l2_decrease_qty=5.0,
    )

    assert update.queue_ahead_after == 0.0
    assert update.l2_advance_qty == 0.5
    assert update.fillable_qty == 1.0
    assert update.fill_reason == FillReason.QUEUE_DEPLETION


def test_l2_increase_does_not_worsen_queue():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0),
        config=QueueModelConfig(mode=QueueModelMode.BALANCED),
        l2_decrease_qty=0.0,
    )

    assert update.queue_ahead_after == 2.0
    assert update.advanced_qty == 0.0


def test_combined_trade_and_l2_applies_trade_first():
    update = update_queue_position(
        _order(queue_ahead_qty=2.0, remaining_qty=1.0),
        config=QueueModelConfig(mode=QueueModelMode.BALANCED, l2_decrease_weight=1.0, trade_at_level_weight=1.0),
        trade=_trade(side=AggressorSide.SELL, price_tick=1000, amount=1.0),
        l2_decrease_qty=5.0,
    )

    assert update.trade_advance_qty == 1.0
    assert update.l2_advance_qty == 1.0
    assert update.queue_ahead_after == 0.0
    assert update.fillable_qty == 1.0


def test_rejects_non_live_orders():
    with pytest.raises(ValueError):
        update_queue_position(_order(status=OrderStatus.FILLED, remaining_qty=0.0))

    with pytest.raises(ValueError):
        update_queue_position(_order(status=OrderStatus.CANCELLED))

    with pytest.raises(ValueError):
        update_queue_position(_order(status=OrderStatus.PENDING_NEW))


def test_update_queue_position_does_not_mutate_order():
    order = _order(queue_ahead_qty=2.0)
    update_queue_position(order, trade=_trade(side=AggressorSide.SELL, price_tick=1000, amount=1.0))

    assert order.queue_ahead_qty == 2.0
    assert order.remaining_qty == 1.0


def test_queue_model_has_no_forbidden_imports():
    source = Path("mmrt/execution/queue_model.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import numpy" not in source
    assert "import sklearn" not in source
