"""Pure functional fill simulation for passive execution orders.

This module owns ActiveOrder state transitions, fill construction, and applying
queue-model outputs to orders. It intentionally does not update cash,
inventory, rewards, observations, execution tapes, or environments.
"""

from dataclasses import dataclass, replace
from typing import Any, Sequence

from mmrt.execution.contracts import (
    ActiveOrder,
    Fill,
    FillReason,
    OrderSide,
    OrderStatus,
    QuoteIntent,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.queue_model import QueueModelConfig, QueueModelUpdate, update_queue_position


_INF = float("inf")
DEFAULT_MAKER_FEE_BPS = -0.5


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    value = float(value)
    if value != value or value in (_INF, -_INF):
        raise ValueError(f"{name} must be a finite float")
    return value


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


@dataclass(frozen=True, slots=True)
class FillSimulatorConfig:
    queue_model: QueueModelConfig = QueueModelConfig()
    maker_fee_bps: float = -0.5
    qty_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.queue_model, QueueModelConfig):
            raise ValueError("queue_model must be QueueModelConfig")
        object.__setattr__(self, "maker_fee_bps", _require_finite_float(self.maker_fee_bps, "maker_fee_bps"))
        object.__setattr__(self, "qty_epsilon", _require_positive_float(self.qty_epsilon, "qty_epsilon"))


@dataclass(frozen=True, slots=True)
class FillSimulationResult:
    orders: tuple[ActiveOrder, ...]
    fills: tuple[Fill, ...]

    def __post_init__(self) -> None:
        orders = _orders_tuple(self.orders)
        fills = _fills_tuple(self.fills)
        seen: set[int] = set()
        for order in orders:
            if order.order_id in seen:
                raise ValueError("order ids must be unique")
            seen.add(order.order_id)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "fills", fills)


def place_orders_from_quote(
    quote: QuoteIntent,
    *,
    next_order_id: int,
    local_ts_us: int,
    bid_queue_ahead_qty: float = 0.0,
    ask_queue_ahead_qty: float = 0.0,
    effective_local_ts_us: int = 0,
) -> tuple[ActiveOrder, ...]:
    quote = _require_quote(quote)
    next_order_id = _require_nonnegative_int(next_order_id, "next_order_id")
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    bid_queue_ahead_qty = _require_nonnegative_float(bid_queue_ahead_qty, "bid_queue_ahead_qty")
    ask_queue_ahead_qty = _require_nonnegative_float(ask_queue_ahead_qty, "ask_queue_ahead_qty")
    effective_local_ts_us = _require_nonnegative_int(effective_local_ts_us, "effective_local_ts_us")
    if effective_local_ts_us and effective_local_ts_us < local_ts_us:
        raise ValueError("effective_local_ts_us must be >= local_ts_us")

    orders: list[ActiveOrder] = []
    if quote.bid_enabled:
        orders.append(
            ActiveOrder(
                order_id=next_order_id,
                side=OrderSide.BUY,
                price_tick=quote.bid_price_tick,
                qty=quote.bid_qty,
                remaining_qty=quote.bid_qty,
                queue_ahead_qty=bid_queue_ahead_qty,
                status=OrderStatus.ACTIVE,
                created_local_ts_us=local_ts_us,
                last_update_local_ts_us=local_ts_us,
                effective_local_ts_us=effective_local_ts_us,
            )
        )
    if quote.ask_enabled:
        orders.append(
            ActiveOrder(
                order_id=next_order_id + len(orders),
                side=OrderSide.SELL,
                price_tick=quote.ask_price_tick,
                qty=quote.ask_qty,
                remaining_qty=quote.ask_qty,
                queue_ahead_qty=ask_queue_ahead_qty,
                status=OrderStatus.ACTIVE,
                created_local_ts_us=local_ts_us,
                last_update_local_ts_us=local_ts_us,
                effective_local_ts_us=effective_local_ts_us,
            )
        )
    return tuple(orders)


def request_cancel_live_orders(
    orders: Sequence[ActiveOrder],
    *,
    request_local_ts_us: int,
    cancel_effective_local_ts_us: int,
) -> tuple[ActiveOrder, ...]:
    orders_tuple = _orders_tuple(orders)
    request_local_ts_us = _require_positive_int(request_local_ts_us, "request_local_ts_us")
    cancel_effective_local_ts_us = _require_positive_int(cancel_effective_local_ts_us, "cancel_effective_local_ts_us")
    if cancel_effective_local_ts_us < request_local_ts_us:
        raise ValueError("cancel_effective_local_ts_us must be >= request_local_ts_us")
    out: list[ActiveOrder] = []
    for order in orders_tuple:
        if not order.is_live:
            out.append(order)
            continue
        _validate_local_ts_for_order(order, request_local_ts_us)
        effective = order.cancel_effective_local_ts_us
        if effective and effective <= cancel_effective_local_ts_us:
            out.append(order)
            continue
        out.append(replace(
            order,
            cancel_requested_local_ts_us=request_local_ts_us,
            cancel_effective_local_ts_us=cancel_effective_local_ts_us,
            last_update_local_ts_us=request_local_ts_us,
        ))
    return tuple(out)


def finalize_effective_cancels(
    orders: Sequence[ActiveOrder],
    *,
    local_ts_us: int,
) -> tuple[ActiveOrder, ...]:
    orders_tuple = _orders_tuple(orders)
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    out: list[ActiveOrder] = []
    for order in orders_tuple:
        if order.is_live and order.cancel_effective_local_ts_us and order.cancel_effective_local_ts_us <= local_ts_us:
            _validate_local_ts_for_order(order, local_ts_us)
            out.append(replace(order, status=OrderStatus.CANCELLED, last_update_local_ts_us=local_ts_us))
        else:
            out.append(order)
    return tuple(out)


def live_orders(orders: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    return tuple(order for order in _orders_tuple(orders) if order.is_live)


def apply_fill_to_order(
    order: ActiveOrder,
    fill_qty: float,
    *,
    queue_ahead_after: float,
    local_ts_us: int,
    qty_epsilon: float = 1e-12,
) -> ActiveOrder:
    order = _require_order(order)
    if not order.is_live:
        raise ValueError("order must be live")
    fill_qty = _require_positive_float(fill_qty, "fill_qty")
    queue_ahead_after = _require_nonnegative_float(queue_ahead_after, "queue_ahead_after")
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    qty_epsilon = _require_positive_float(qty_epsilon, "qty_epsilon")
    _validate_local_ts_for_order(order, local_ts_us)
    if fill_qty > order.remaining_qty + qty_epsilon:
        raise ValueError("fill_qty must be <= order.remaining_qty + qty_epsilon")

    fill_qty = min(fill_qty, order.remaining_qty)
    remaining_qty = _clean_qty(order.remaining_qty - fill_qty, qty_epsilon)
    if remaining_qty <= qty_epsilon:
        return replace(
            order,
            remaining_qty=0.0,
            status=OrderStatus.FILLED,
            queue_ahead_qty=0.0,
            last_update_local_ts_us=local_ts_us,
        )

    return replace(
        order,
        remaining_qty=remaining_qty,
        status=OrderStatus.PARTIALLY_FILLED,
        queue_ahead_qty=_clean_qty(queue_ahead_after, qty_epsilon),
        last_update_local_ts_us=local_ts_us,
    )


def simulate_trade_event(
    orders: Sequence[ActiveOrder],
    trade: TradePrint,
    *,
    symbol_spec: SymbolSpec,
    config: FillSimulatorConfig = FillSimulatorConfig(),
) -> FillSimulationResult:
    orders_tuple = _orders_tuple(orders)
    trade = _require_trade(trade)
    symbol_spec = _require_symbol_spec(symbol_spec)
    config = _require_config(config)
    _assert_no_duplicate_fillable_side_price(orders_tuple, local_ts_us=trade.local_ts_us)

    updated_orders: list[ActiveOrder] = []
    fills: list[Fill] = []
    for order in orders_tuple:
        if not order.is_fillable_at(trade.local_ts_us):
            updated_orders.append(order)
            continue
        _validate_local_ts_for_order(order, trade.local_ts_us)
        queue_update = update_queue_position(order, config=config.queue_model, trade=trade)
        updated_order, fill = _apply_queue_update_to_order(
            order,
            queue_update,
            local_ts_us=trade.local_ts_us,
            symbol_spec=symbol_spec,
            config=config,
        )
        updated_orders.append(updated_order)
        if fill is not None:
            fills.append(fill)
    return FillSimulationResult(orders=tuple(updated_orders), fills=tuple(fills))


def simulate_l2_level_update(
    orders: Sequence[ActiveOrder],
    *,
    side: OrderSide,
    price_tick: int,
    prev_level_qty: float,
    curr_level_qty: float,
    local_ts_us: int,
    symbol_spec: SymbolSpec,
    config: FillSimulatorConfig = FillSimulatorConfig(),
) -> FillSimulationResult:
    orders_tuple = _orders_tuple(orders)
    if not isinstance(side, OrderSide):
        raise ValueError("side must be OrderSide")
    price_tick = _require_positive_int(price_tick, "price_tick")
    prev_level_qty = _require_nonnegative_float(prev_level_qty, "prev_level_qty")
    curr_level_qty = _require_nonnegative_float(curr_level_qty, "curr_level_qty")
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    symbol_spec = _require_symbol_spec(symbol_spec)
    config = _require_config(config)
    _assert_no_duplicate_fillable_side_price(orders_tuple, local_ts_us=local_ts_us)

    updated_orders: list[ActiveOrder] = []
    fills: list[Fill] = []
    for order in orders_tuple:
        if not order.is_fillable_at(local_ts_us) or order.side != side or order.price_tick != price_tick:
            updated_orders.append(order)
            continue
        _validate_local_ts_for_order(order, local_ts_us)
        queue_update = update_queue_position(
            order,
            config=config.queue_model,
            prev_level_qty=prev_level_qty,
            curr_level_qty=curr_level_qty,
        )
        updated_order, fill = _apply_queue_update_to_order(
            order,
            queue_update,
            local_ts_us=local_ts_us,
            symbol_spec=symbol_spec,
            config=config,
        )
        updated_orders.append(updated_order)
        if fill is not None:
            fills.append(fill)
    return FillSimulationResult(orders=tuple(updated_orders), fills=tuple(fills))


def sync_orders_to_quote(
    existing_orders: Sequence[ActiveOrder],
    quote: QuoteIntent,
    *,
    next_order_id: int,
    decision_local_ts_us: int,
    order_effective_local_ts_us: int,
    cancel_effective_local_ts_us: int,
    bid_queue_ahead_qty: float = 0.0,
    ask_queue_ahead_qty: float = 0.0,
) -> tuple[tuple[ActiveOrder, ...], int]:
    existing = _orders_tuple(existing_orders)
    quote = _require_quote(quote)
    next_order_id = _require_nonnegative_int(next_order_id, "next_order_id")
    decision_local_ts_us = _require_positive_int(decision_local_ts_us, "decision_local_ts_us")
    order_effective_local_ts_us = _require_positive_int(order_effective_local_ts_us, "order_effective_local_ts_us")
    cancel_effective_local_ts_us = _require_positive_int(cancel_effective_local_ts_us, "cancel_effective_local_ts_us")
    if order_effective_local_ts_us < decision_local_ts_us or cancel_effective_local_ts_us < decision_local_ts_us:
        raise ValueError("effective timestamps must be >= decision_local_ts_us")

    target = {
        OrderSide.BUY: quote.bid_price_tick if quote.bid_enabled else 0,
        OrderSide.SELL: quote.ask_price_tick if quote.ask_enabled else 0,
    }
    updated: list[ActiveOrder] = []
    cancel_count = 0
    preserved: set[OrderSide] = set()
    for order in existing:
        if not order.is_live:
            updated.append(order)
            continue
        desired_price = target[order.side]
        if desired_price == order.price_tick and order.cancel_effective_local_ts_us == 0:
            preserved.add(order.side)
            updated.append(order)
        else:
            cancel_count += 1 if order.cancel_effective_local_ts_us == 0 else 0
            updated.append(request_cancel_live_orders(
                (order,),
                request_local_ts_us=decision_local_ts_us,
                cancel_effective_local_ts_us=cancel_effective_local_ts_us,
            )[0])

    new_quote = QuoteIntent(
        bid_enabled=quote.bid_enabled and OrderSide.BUY not in preserved,
        ask_enabled=quote.ask_enabled and OrderSide.SELL not in preserved,
        bid_price_tick=quote.bid_price_tick if quote.bid_enabled and OrderSide.BUY not in preserved else 0,
        ask_price_tick=quote.ask_price_tick if quote.ask_enabled and OrderSide.SELL not in preserved else 0,
        bid_qty=quote.bid_qty if quote.bid_enabled and OrderSide.BUY not in preserved else 0.0,
        ask_qty=quote.ask_qty if quote.ask_enabled and OrderSide.SELL not in preserved else 0.0,
    )
    new_orders = place_orders_from_quote(
        new_quote,
        next_order_id=next_order_id,
        local_ts_us=decision_local_ts_us,
        bid_queue_ahead_qty=bid_queue_ahead_qty,
        ask_queue_ahead_qty=ask_queue_ahead_qty,
        effective_local_ts_us=order_effective_local_ts_us,
    )
    return tuple(updated) + new_orders, cancel_count


def _apply_queue_update_to_order(
    order: ActiveOrder,
    queue_update: QueueModelUpdate,
    *,
    local_ts_us: int,
    symbol_spec: SymbolSpec,
    config: FillSimulatorConfig,
) -> tuple[ActiveOrder, Fill | None]:
    if not isinstance(queue_update, QueueModelUpdate):
        raise ValueError("queue_update must be QueueModelUpdate")
    if queue_update.fillable_qty > 0.0:
        fill_qty = min(order.remaining_qty, queue_update.fillable_qty)
        fill = _make_fill(
            order,
            local_ts_us=local_ts_us,
            fill_qty=fill_qty,
            reason=_require_fill_reason(queue_update.fill_reason),
            queue_ahead_before=queue_update.queue_ahead_before,
            queue_ahead_after=queue_update.queue_ahead_after,
            symbol_spec=symbol_spec,
            maker_fee_bps=config.maker_fee_bps,
        )
        updated_order = apply_fill_to_order(
            order,
            fill_qty,
            queue_ahead_after=queue_update.queue_ahead_after,
            local_ts_us=local_ts_us,
            qty_epsilon=config.qty_epsilon,
        )
        return updated_order, fill

    if abs(queue_update.queue_ahead_after - order.queue_ahead_qty) > config.qty_epsilon:
        return (
            _replace_order_queue_ahead(
                order,
                queue_ahead_qty=queue_update.queue_ahead_after,
                local_ts_us=local_ts_us,
                qty_epsilon=config.qty_epsilon,
            ),
            None,
        )

    return order, None


def _make_fill(
    order: ActiveOrder,
    *,
    local_ts_us: int,
    fill_qty: float,
    reason: FillReason,
    queue_ahead_before: float,
    queue_ahead_after: float,
    symbol_spec: SymbolSpec,
    maker_fee_bps: float,
) -> Fill:
    order = _require_order(order)
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    fill_qty = _require_positive_float(fill_qty, "fill_qty")
    reason = _require_fill_reason(reason)
    queue_ahead_before = _require_nonnegative_float(queue_ahead_before, "queue_ahead_before")
    queue_ahead_after = _require_nonnegative_float(queue_ahead_after, "queue_ahead_after")
    symbol_spec = _require_symbol_spec(symbol_spec)
    maker_fee_bps = _require_finite_float(maker_fee_bps, "maker_fee_bps")
    price = symbol_spec.tick_to_price(order.price_tick)
    notional = price * fill_qty * symbol_spec.contract_size
    fee = notional * maker_fee_bps / 10_000.0
    return Fill(
        order_id=order.order_id,
        side=order.side,
        local_ts_us=local_ts_us,
        price_tick=order.price_tick,
        qty=fill_qty,
        fee=fee,
        reason=reason,
        queue_ahead_before=queue_ahead_before,
        queue_ahead_after=queue_ahead_after,
    )


def _replace_order_queue_ahead(
    order: ActiveOrder,
    *,
    queue_ahead_qty: float,
    local_ts_us: int,
    qty_epsilon: float,
) -> ActiveOrder:
    order = _require_order(order)
    if not order.is_live:
        return order
    queue_ahead_qty = _require_nonnegative_float(queue_ahead_qty, "queue_ahead_qty")
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    qty_epsilon = _require_positive_float(qty_epsilon, "qty_epsilon")
    _validate_local_ts_for_order(order, local_ts_us)
    queue_ahead_qty = _clean_qty(queue_ahead_qty, qty_epsilon)
    if abs(queue_ahead_qty - order.queue_ahead_qty) <= qty_epsilon:
        return order
    return replace(order, queue_ahead_qty=queue_ahead_qty, last_update_local_ts_us=local_ts_us)


def _require_symbol_spec(value: Any) -> SymbolSpec:
    if not isinstance(value, SymbolSpec):
        raise ValueError("symbol_spec must be SymbolSpec")
    return value


def _require_config(value: Any) -> FillSimulatorConfig:
    if not isinstance(value, FillSimulatorConfig):
        raise ValueError("config must be FillSimulatorConfig")
    return value


def _require_quote(value: Any) -> QuoteIntent:
    if not isinstance(value, QuoteIntent):
        raise ValueError("quote must be QuoteIntent")
    return value


def _require_order(value: Any) -> ActiveOrder:
    if not isinstance(value, ActiveOrder):
        raise ValueError("order must be ActiveOrder")
    return value


def _orders_tuple(values: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("orders must be a sequence of ActiveOrder")
    try:
        orders = tuple(values)
    except TypeError as exc:
        raise ValueError("orders must be a sequence of ActiveOrder") from exc
    for idx, order in enumerate(orders):
        if not isinstance(order, ActiveOrder):
            raise ValueError(f"orders[{idx}] must be ActiveOrder")
    return orders


def _fills_tuple(values: Sequence[Fill]) -> tuple[Fill, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("fills must be a sequence of Fill")
    try:
        fills = tuple(values)
    except TypeError as exc:
        raise ValueError("fills must be a sequence of Fill") from exc
    for idx, fill in enumerate(fills):
        if not isinstance(fill, Fill):
            raise ValueError(f"fills[{idx}] must be Fill")
    return fills


def _require_trade(value: Any) -> TradePrint:
    if not isinstance(value, TradePrint):
        raise ValueError("trade must be TradePrint")
    return value


def _require_fill_reason(value: Any) -> FillReason:
    if not isinstance(value, FillReason):
        raise ValueError("fill_reason must be FillReason")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _clean_qty(value: float, eps: float) -> float:
    value = _require_finite_float(value, "value")
    eps = _require_positive_float(eps, "eps")
    return 0.0 if abs(value) <= eps else max(value, 0.0)


def _validate_local_ts_for_order(order: ActiveOrder, local_ts_us: int) -> None:
    order = _require_order(order)
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    if local_ts_us < order.last_update_local_ts_us:
        raise ValueError("local_ts_us must be >= order.last_update_local_ts_us")


def _assert_no_duplicate_fillable_side_price(orders: tuple[ActiveOrder, ...], *, local_ts_us: int) -> None:
    local_ts_us = _require_positive_int(local_ts_us, "local_ts_us")
    seen: set[tuple[OrderSide, int]] = set()
    for order in orders:
        if not order.is_fillable_at(local_ts_us):
            continue
        key = (order.side, order.price_tick)
        if key in seen:
            raise ValueError("duplicate fillable orders at same side/price are not supported")
        seen.add(key)


__all__ = [
    "DEFAULT_MAKER_FEE_BPS",
    "FillSimulatorConfig",
    "FillSimulationResult",
    "place_orders_from_quote",
    "request_cancel_live_orders",
    "finalize_effective_cancels",
    "sync_orders_to_quote",
    "live_orders",
    "apply_fill_to_order",
    "simulate_trade_event",
    "simulate_l2_level_update",
]
