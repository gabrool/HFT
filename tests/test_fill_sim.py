from pathlib import Path

import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActiveOrder,
    FillReason,
    OrderSide,
    OrderStatus,
    QuoteIntent,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.fill_sim import (
    FillSimulatorConfig,
    FillSimulationResult,
    apply_fill_to_order,
    request_cancel_live_orders,
    live_orders,
    place_orders_from_quote,
    sync_orders_to_quote,
    simulate_l2_level_update,
    simulate_trade_event,
)
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.time_key import EventKey


def _spec() -> SymbolSpec:
    return SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=5.0,
    )


def _order(
    *,
    order_id: int = 1,
    side: OrderSide = OrderSide.BUY,
    price_tick: int = 1000,
    qty: float = 1.0,
    remaining_qty: float = 1.0,
    queue_ahead_qty: float = 2.0,
    status: OrderStatus = OrderStatus.ACTIVE,
    created_local_ts_us: int = 100,
    last_update_local_ts_us: int = 100,
    effective_local_ts_us: int = 0,
    cancel_requested_local_ts_us: int = 0,
    cancel_effective_local_ts_us: int = 0,
) -> ActiveOrder:
    return ActiveOrder(
        order_id=order_id,
        side=side,
        price_tick=price_tick,
        qty=qty,
        remaining_qty=remaining_qty,
        queue_ahead_qty=queue_ahead_qty,
        status=status,
        created_local_ts_us=created_local_ts_us,
        created_event_seq=0,
        last_update_local_ts_us=last_update_local_ts_us,
        last_update_event_seq=0,
        effective_local_ts_us=effective_local_ts_us,
        effective_event_seq=0 if effective_local_ts_us else -1,
        cancel_requested_local_ts_us=cancel_requested_local_ts_us,
        cancel_requested_event_seq=0 if cancel_requested_local_ts_us else -1,
        cancel_effective_local_ts_us=cancel_effective_local_ts_us,
        cancel_effective_event_seq=0 if cancel_effective_local_ts_us else -1,
    )


def _trade(
    *,
    side: AggressorSide = AggressorSide.SELL,
    price_tick: int = 1000,
    amount: float = 1.0,
    local_ts_us: int = 200,
) -> TradePrint:
    return TradePrint(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us - 10,
        side=side,
        price_tick=price_tick,
        amount=amount,
        trade_id="t",
        source_row=0,
    )


def test_place_orders_from_two_sided_quote():
    quote = QuoteIntent(
        bid_enabled=True,
        ask_enabled=True,
        bid_price_tick=1000,
        ask_price_tick=1002,
        bid_qty=0.01,
        ask_qty=0.02,
    )

    orders = place_orders_from_quote(
        quote,
        next_order_id=10,
        created_key=EventKey(100, 0), effective_key=EventKey(100, 0),
        bid_queue_ahead_qty=1.5,
        ask_queue_ahead_qty=2.5,
    )

    assert len(orders) == 2
    bid, ask = orders
    assert bid.order_id == 10
    assert bid.side == OrderSide.BUY
    assert bid.price_tick == 1000
    assert bid.qty == 0.01
    assert bid.remaining_qty == 0.01
    assert bid.queue_ahead_qty == 1.5
    assert bid.status == OrderStatus.PENDING_NEW

    assert ask.order_id == 11
    assert ask.side == OrderSide.SELL
    assert ask.price_tick == 1002
    assert ask.qty == 0.02
    assert ask.queue_ahead_qty == 2.5


def test_place_orders_skips_disabled_sides():
    quote = QuoteIntent(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_tick=1000,
        bid_qty=0.01,
    )

    orders = place_orders_from_quote(quote, next_order_id=5, created_key=EventKey(100, 0), effective_key=EventKey(100, 0))

    assert len(orders) == 1
    assert orders[0].order_id == 5
    assert orders[0].side == OrderSide.BUY


def test_request_cancel_live_orders_only():
    active = _order(order_id=1, status=OrderStatus.ACTIVE)
    partial = _order(order_id=2, qty=1.0, remaining_qty=0.5, status=OrderStatus.PARTIALLY_FILLED)
    filled = _order(order_id=3, qty=1.0, remaining_qty=0.0, status=OrderStatus.FILLED)

    out = request_cancel_live_orders([active, partial, filled], request_key=EventKey(200, 0), cancel_effective_key=EventKey(200, 0))

    assert out[0].status == OrderStatus.PENDING_CANCEL
    assert out[1].status == OrderStatus.PENDING_CANCEL
    assert out[0].cancel_effective_local_ts_us == 200
    assert out[1].cancel_effective_local_ts_us == 200
    assert out[2] == filled
    assert out[0].last_update_local_ts_us == 200
    assert out[1].last_update_local_ts_us == 200


def test_sync_orders_to_quote_cancels_live_and_appends_new():
    old = _order(order_id=1, side=OrderSide.BUY)
    quote = QuoteIntent(
        bid_enabled=True,
        ask_enabled=True,
        bid_price_tick=999,
        ask_price_tick=1002,
        bid_qty=0.01,
        ask_qty=0.01,
    )

    out, cancel_count = sync_orders_to_quote(
        [old],
        quote,
        next_order_id=10,
        decision_key=EventKey(200, 0),
        order_effective_key=EventKey(200, 0),
        cancel_effective_key=EventKey(200, 0),
    )

    assert cancel_count == 1
    assert out[0].status == OrderStatus.PENDING_CANCEL
    assert out[0].cancel_effective_local_ts_us == 200
    assert len(out) == 3
    assert out[1].order_id == 10
    assert out[2].order_id == 11


def test_sync_same_price_pending_cancel_places_replacement_without_double_counting_cancel():
    pending_cancel = _order(
        order_id=1,
        side=OrderSide.BUY,
        price_tick=1000,
        created_local_ts_us=100,
        last_update_local_ts_us=150,
        cancel_requested_local_ts_us=150,
        cancel_effective_local_ts_us=250,
    )
    quote = QuoteIntent(
        bid_enabled=True,
        ask_enabled=False,
        bid_price_tick=1000,
        bid_qty=0.01,
    )

    out, cancel_count = sync_orders_to_quote(
        [pending_cancel],
        quote,
        next_order_id=10,
        decision_key=EventKey(200, 0),
        order_effective_key=EventKey(260, 0),
        cancel_effective_key=EventKey(300, 0),
    )

    assert cancel_count == 0
    assert len(out) == 2
    old, replacement = out
    assert old.order_id == 1
    assert old.cancel_effective_local_ts_us == 250
    assert replacement.order_id == 10
    assert replacement.side == OrderSide.BUY
    assert replacement.price_tick == 1000
    assert replacement.effective_local_ts_us == 260


def test_pending_cancel_and_future_replacement_same_price_not_duplicate_before_overlap():
    old = _order(
        order_id=1,
        side=OrderSide.BUY,
        price_tick=1000,
        cancel_requested_local_ts_us=150,
        cancel_effective_local_ts_us=250,
        last_update_local_ts_us=150,
    )
    replacement = _order(
        order_id=2,
        side=OrderSide.BUY,
        price_tick=1000,
        created_local_ts_us=200,
        last_update_local_ts_us=200,
        effective_local_ts_us=260,
    )

    simulate_trade_event([old, replacement], _trade(local_ts_us=220), event_key=EventKey(220, 0), symbol_spec=_spec())
    simulate_trade_event([old, replacement], _trade(local_ts_us=255), event_key=EventKey(255, 0), symbol_spec=_spec())
    simulate_trade_event([old, replacement], _trade(local_ts_us=270), event_key=EventKey(270, 0), symbol_spec=_spec())


def test_live_orders_filters_correctly():
    active = _order(order_id=1, status=OrderStatus.ACTIVE)
    partial = _order(order_id=2, qty=1.0, remaining_qty=0.5, status=OrderStatus.PARTIALLY_FILLED)
    cancelled = _order(order_id=3, status=OrderStatus.CANCELLED)

    assert live_orders([active, partial, cancelled]) == (active, partial)


def test_apply_partial_fill_to_order():
    order = _order(qty=1.0, remaining_qty=1.0, queue_ahead_qty=0.0)

    updated = apply_fill_to_order(
        order,
        0.25,
        queue_ahead_after=0.0,
        event_key=EventKey(200, 0),
    )

    assert updated.remaining_qty == 0.75
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.queue_ahead_qty == 0.0
    assert updated.last_update_local_ts_us == 200


def test_apply_full_fill_to_order():
    order = _order(qty=1.0, remaining_qty=0.25, queue_ahead_qty=0.0, status=OrderStatus.PARTIALLY_FILLED)

    updated = apply_fill_to_order(
        order,
        0.25,
        queue_ahead_after=0.0,
        event_key=EventKey(200, 0),
    )

    assert updated.remaining_qty == 0.0
    assert updated.status == OrderStatus.FILLED
    assert updated.queue_ahead_qty == 0.0


def test_trade_event_irrelevant_trade_preserves_order():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=2.0)
    result = simulate_trade_event(
        [order],
        _trade(side=AggressorSide.BUY, price_tick=1000, amount=10.0),
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
    )

    assert result.orders == (order,)
    assert result.fills == ()


def test_trade_at_level_advances_queue_without_fill():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=2.0)
    result = simulate_trade_event(
        [order],
        _trade(side=AggressorSide.SELL, price_tick=1000, amount=0.5),
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(queue_model=QueueModelConfig(mode="balanced", trade_at_level_weight=1.0)),
    )

    updated = result.orders[0]
    assert result.fills == ()
    assert updated.queue_ahead_qty == 1.5
    assert updated.status == OrderStatus.ACTIVE
    assert updated.last_update_local_ts_us == 200


def test_trade_at_level_leftover_creates_fill_and_updates_order():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=0.5, qty=1.0, remaining_qty=1.0)
    result = simulate_trade_event(
        [order],
        _trade(side=AggressorSide.SELL, price_tick=1000, amount=0.75),
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(queue_model=QueueModelConfig(mode="balanced", trade_at_level_weight=1.0)),
    )

    assert len(result.fills) == 1
    fill = result.fills[0]
    updated = result.orders[0]

    assert fill.order_id == order.order_id
    assert fill.side == OrderSide.BUY
    assert fill.local_ts_us == 200
    assert fill.price_tick == order.price_tick
    assert fill.qty == 0.25
    assert fill.reason == FillReason.TRADE_AT_LEVEL
    assert fill.queue_ahead_before == 0.5
    assert fill.queue_ahead_after == 0.0

    assert updated.remaining_qty == 0.75
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.queue_ahead_qty == 0.0


def test_trade_through_fills_full_remaining():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=10.0, qty=1.0, remaining_qty=0.4)
    result = simulate_trade_event(
        [order],
        _trade(side=AggressorSide.SELL, price_tick=999, amount=0.01),
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
    )

    fill = result.fills[0]
    updated = result.orders[0]

    assert fill.qty == 0.4
    assert fill.reason == FillReason.TRADE_THROUGH
    assert fill.queue_ahead_before == 10.0
    assert fill.queue_ahead_after == 0.0
    assert updated.status == OrderStatus.FILLED
    assert updated.remaining_qty == 0.0


def test_maker_fee_bps_applied_to_fill():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=0.0, qty=1.0, remaining_qty=1.0)
    result = simulate_trade_event(
        [order],
        _trade(side=AggressorSide.SELL, price_tick=1000, amount=0.1),
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(maker_fee_bps=1.0, queue_model=QueueModelConfig(mode="balanced", trade_at_level_weight=1.0)),
    )

    fill = result.fills[0]
    expected_notional = _spec().tick_to_price(1000) * 0.1 * _spec().contract_size
    assert fill.fee == pytest.approx(expected_notional * 1.0 / 10_000.0)

    rebate_result = simulate_trade_event(
        [order],
        _trade(side=AggressorSide.SELL, price_tick=1000, amount=0.1),
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(maker_fee_bps=-0.5, queue_model=QueueModelConfig(mode="balanced", trade_at_level_weight=1.0)),
    )
    assert rebate_result.fills[0].fee < 0


def test_l2_update_balanced_advances_queue_without_fill():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=2.0)
    result = simulate_l2_level_update(
        [order],
        side=OrderSide.BUY,
        price_tick=1000,
        l2_decrease_qty=2.0,
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(queue_model=QueueModelConfig(mode="balanced", l2_decrease_weight=0.5)),
    )

    assert result.fills == ()
    assert result.orders[0].queue_ahead_qty == 1.0
    assert result.orders[0].status == OrderStatus.ACTIVE


def test_l2_update_balanced_depletion_creates_fill():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=0.5, qty=1.0, remaining_qty=1.0)
    result = simulate_l2_level_update(
        [order],
        side=OrderSide.BUY,
        price_tick=1000,
        l2_decrease_qty=1.0,
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(queue_model=QueueModelConfig(mode="balanced", l2_decrease_weight=1.0)),
    )

    assert len(result.fills) == 1
    assert result.fills[0].qty == 0.5
    assert result.fills[0].reason == FillReason.QUEUE_DEPLETION
    assert result.orders[0].remaining_qty == 0.5


def test_l2_update_conservative_ignored():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=0.5)
    result = simulate_l2_level_update(
        [order],
        side=OrderSide.BUY,
        price_tick=1000,
        l2_decrease_qty=5.0,
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(queue_model=QueueModelConfig(mode="conservative")),
    )

    assert result.orders == (order,)
    assert result.fills == ()


def test_l2_update_nonmatching_level_ignored():
    order = _order(side=OrderSide.BUY, price_tick=1000, queue_ahead_qty=0.5)
    result = simulate_l2_level_update(
        [order],
        side=OrderSide.SELL,
        price_tick=1000,
        l2_decrease_qty=5.0,
        event_key=EventKey(200, 0),
        symbol_spec=_spec(),
        config=FillSimulatorConfig(queue_model=QueueModelConfig(mode="balanced")),
    )

    assert result.orders == (order,)
    assert result.fills == ()


def test_two_simultaneously_fillable_same_side_price_orders_rejected():
    order1 = _order(order_id=1, side=OrderSide.BUY, price_tick=1000)
    order2 = _order(order_id=2, side=OrderSide.BUY, price_tick=1000)

    with pytest.raises(ValueError, match="duplicate fillable orders"):
        simulate_trade_event([order1, order2], _trade(side=AggressorSide.SELL, price_tick=1000), event_key=EventKey(200, 0), symbol_spec=_spec())

    with pytest.raises(ValueError, match="duplicate fillable orders"):
        simulate_l2_level_update(
            [order1, order2],
            side=OrderSide.BUY,
            price_tick=1000,
            l2_decrease_qty=1.0,
            event_key=EventKey(200, 0),
            symbol_spec=_spec(),
        )


def test_local_timestamp_cannot_decrease():
    order = _order(last_update_local_ts_us=300)

    with pytest.raises(ValueError):
        request_cancel_live_orders([order], request_key=EventKey(200, 0), cancel_effective_key=EventKey(200, 0))

    with pytest.raises(ValueError):
        simulate_l2_level_update(
            [order],
            side=OrderSide.BUY,
            price_tick=1000,
            l2_decrease_qty=1.0,
            event_key=EventKey(200, 0),
            symbol_spec=_spec(),
        )

    with pytest.raises(ValueError):
        simulate_trade_event([order], _trade(local_ts_us=200), event_key=EventKey(200, 0), symbol_spec=_spec())


def test_fill_simulation_result_rejects_duplicate_order_ids():
    order1 = _order(order_id=1)
    order2 = _order(order_id=1, price_tick=999)

    with pytest.raises(ValueError):
        FillSimulationResult(orders=(order1, order2), fills=())


def test_fill_sim_has_no_forbidden_imports():
    source = Path("mmrt/execution/fill_sim.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import numpy" not in source
    assert "import sklearn" not in source
