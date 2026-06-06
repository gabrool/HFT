"""Pure functional fill simulation for passive execution orders.

This module owns ActiveOrder state transitions, fill construction, and applying
queue-model outputs to orders. It intentionally does not update cash,
inventory, rewards, observations, execution tapes, or environments.
"""

from dataclasses import dataclass, replace
from typing import Any, Sequence

from mmrt.execution.contracts import (
    ActiveOrder,
    BookTop,
    Fill,
    FillReason,
    OrderSide,
    OrderStatus,
    QuoteIntent,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.queue_model import QueueModelConfig, QueueModelUpdate, update_queue_position
from mmrt.time_key import EventKey


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
class OrderActivationResult:
    orders: tuple[ActiveOrder, ...]
    activated_count: int = 0
    post_only_reject_count: int = 0
    already_cancelled_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", _unique_orders_tuple(self.orders))
        _require_nonnegative_int(self.activated_count, "activated_count")
        _require_nonnegative_int(self.post_only_reject_count, "post_only_reject_count")
        _require_nonnegative_int(self.already_cancelled_count, "already_cancelled_count")


@dataclass(frozen=True, slots=True)
class CancelFinalizationResult:
    orders: tuple[ActiveOrder, ...]
    cancelled_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", _unique_orders_tuple(self.orders))
        _require_nonnegative_int(self.cancelled_count, "cancelled_count")


@dataclass(frozen=True, slots=True)
class FillSimulationResult:
    orders: tuple[ActiveOrder, ...]
    fills: tuple[Fill, ...]
    queue_updates: tuple[QueueModelUpdate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", _unique_orders_tuple(self.orders))
        object.__setattr__(self, "fills", _fills_tuple(self.fills))
        updates = tuple(self.queue_updates)
        if not all(isinstance(update, QueueModelUpdate) for update in updates):
            raise ValueError("queue_updates must contain QueueModelUpdate values")
        object.__setattr__(self, "queue_updates", updates)


def place_orders_from_quote(
    quote: QuoteIntent,
    *,
    next_order_id: int,
    created_key: EventKey,
    effective_key: EventKey,
    bid_queue_ahead_qty: float = 0.0,
    ask_queue_ahead_qty: float = 0.0,
) -> tuple[ActiveOrder, ...]:
    quote = _require_quote(quote)
    next_order_id = _require_nonnegative_int(next_order_id, "next_order_id")
    created_key = _require_event_key(created_key, "created_key")
    effective_key = _require_event_key(effective_key, "effective_key")
    if effective_key < created_key:
        raise ValueError("effective_key must be >= created_key")
    bid_queue_ahead_qty = _require_nonnegative_float(bid_queue_ahead_qty, "bid_queue_ahead_qty")
    ask_queue_ahead_qty = _require_nonnegative_float(ask_queue_ahead_qty, "ask_queue_ahead_qty")

    orders: list[ActiveOrder] = []
    if quote.bid_enabled:
        orders.append(_new_order(next_order_id, OrderSide.BUY, quote.bid_price_tick, quote.bid_qty, bid_queue_ahead_qty, created_key, effective_key))
    if quote.ask_enabled:
        orders.append(_new_order(next_order_id + len(orders), OrderSide.SELL, quote.ask_price_tick, quote.ask_qty, ask_queue_ahead_qty, created_key, effective_key))
    return tuple(orders)


def _new_order(order_id: int, side: OrderSide, price_tick: int, qty: float, queue_ahead_qty: float, created_key: EventKey, effective_key: EventKey) -> ActiveOrder:
    return ActiveOrder(
        order_id=order_id,
        side=side,
        price_tick=price_tick,
        qty=qty,
        remaining_qty=qty,
        queue_ahead_qty=queue_ahead_qty,
        status=OrderStatus.PENDING_NEW,
        created_local_ts_us=created_key.local_ts_us,
        created_event_seq=created_key.event_seq,
        last_update_local_ts_us=created_key.local_ts_us,
        last_update_event_seq=created_key.event_seq,
        effective_local_ts_us=effective_key.local_ts_us,
        effective_event_seq=effective_key.event_seq,
    )


def activate_pending_orders(orders: Sequence[ActiveOrder], *, event_key: EventKey, book_top: BookTop | None, post_only_gap_ticks: int) -> OrderActivationResult:
    orders_tuple = _orders_tuple(orders)
    event_key = _require_event_key(event_key, "event_key")
    post_only_gap_ticks = _require_nonnegative_int(post_only_gap_ticks, "post_only_gap_ticks")
    activated = rejected = already_cancelled = 0
    out: list[ActiveOrder] = []
    for order in orders_tuple:
        if order.status != OrderStatus.PENDING_NEW or order.effective_key > event_key:
            out.append(order)
            continue
        cancel_key = order.cancel_effective_key
        if cancel_key is not None and cancel_key <= event_key:
            out.append(_replace_status(order, OrderStatus.CANCELLED, event_key))
            already_cancelled += 1
            continue
        if not _is_post_only_safe(order, book_top, post_only_gap_ticks):
            out.append(_replace_status(order, OrderStatus.REJECTED, event_key, remaining_qty=order.qty))
            rejected += 1
            continue
        status = OrderStatus.PENDING_CANCEL if cancel_key is not None else OrderStatus.ACTIVE
        out.append(_replace_status(order, status, event_key))
        activated += 1
    return OrderActivationResult(tuple(out), activated, rejected, already_cancelled)


def _is_post_only_safe(order: ActiveOrder, book_top: BookTop | None, post_only_gap_ticks: int) -> bool:
    if book_top is None:
        return False
    if order.side == OrderSide.BUY:
        return order.price_tick <= book_top.best_ask_tick - post_only_gap_ticks
    if order.side == OrderSide.SELL:
        return order.price_tick >= book_top.best_bid_tick + post_only_gap_ticks
    return False


def request_cancel_live_orders(orders: Sequence[ActiveOrder], *, request_key: EventKey, cancel_effective_key: EventKey) -> tuple[ActiveOrder, ...]:
    orders_tuple = _orders_tuple(orders)
    request_key = _require_event_key(request_key, "request_key")
    cancel_effective_key = _require_event_key(cancel_effective_key, "cancel_effective_key")
    if cancel_effective_key < request_key:
        raise ValueError("cancel_effective_key must be >= request_key")
    out: list[ActiveOrder] = []
    for order in orders_tuple:
        if order.is_terminal:
            out.append(order)
            continue
        _validate_event_key_for_order(order, request_key)
        existing = order.cancel_effective_key
        if existing is not None and existing <= cancel_effective_key:
            out.append(order)
            continue
        status = order.status
        if status in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL):
            status = OrderStatus.PENDING_CANCEL
        out.append(replace(
            order,
            status=status,
            cancel_requested_local_ts_us=request_key.local_ts_us,
            cancel_requested_event_seq=request_key.event_seq,
            cancel_effective_local_ts_us=cancel_effective_key.local_ts_us,
            cancel_effective_event_seq=cancel_effective_key.event_seq,
            last_update_local_ts_us=request_key.local_ts_us,
            last_update_event_seq=request_key.event_seq,
        ))
    return tuple(out)


def finalize_effective_cancels(orders: Sequence[ActiveOrder], *, event_key: EventKey) -> CancelFinalizationResult:
    orders_tuple = _orders_tuple(orders)
    event_key = _require_event_key(event_key, "event_key")
    out: list[ActiveOrder] = []
    cancelled = 0
    for order in orders_tuple:
        cancel_key = order.cancel_effective_key
        if order.status in (OrderStatus.PENDING_CANCEL, OrderStatus.PENDING_NEW) and cancel_key is not None and cancel_key <= event_key:
            _validate_event_key_for_order(order, event_key)
            out.append(_replace_status(order, OrderStatus.CANCELLED, event_key))
            cancelled += 1
        else:
            out.append(order)
    return CancelFinalizationResult(tuple(out), cancelled)


def live_orders(orders: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    return tuple(order for order in _orders_tuple(orders) if order.is_live)


def apply_fill_to_order(order: ActiveOrder, fill_qty: float, *, queue_ahead_after: float, event_key: EventKey, qty_epsilon: float = 1e-12) -> ActiveOrder:
    order = _require_order(order)
    if not order.status in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL):
        raise ValueError("order must be fillable-live")
    fill_qty = _require_positive_float(fill_qty, "fill_qty")
    queue_ahead_after = _require_nonnegative_float(queue_ahead_after, "queue_ahead_after")
    event_key = _require_event_key(event_key, "event_key")
    qty_epsilon = _require_positive_float(qty_epsilon, "qty_epsilon")
    _validate_event_key_for_order(order, event_key)
    if fill_qty > order.remaining_qty + qty_epsilon:
        raise ValueError("fill_qty must be <= order.remaining_qty + qty_epsilon")

    fill_qty = min(fill_qty, order.remaining_qty)
    remaining_qty = _clean_qty(order.remaining_qty - fill_qty, qty_epsilon)
    if remaining_qty <= qty_epsilon:
        return replace(order, remaining_qty=0.0, status=OrderStatus.FILLED, queue_ahead_qty=0.0, last_update_local_ts_us=event_key.local_ts_us, last_update_event_seq=event_key.event_seq)

    status = OrderStatus.PENDING_CANCEL if order.status == OrderStatus.PENDING_CANCEL or order.cancel_effective_key is not None else OrderStatus.PARTIALLY_FILLED
    return replace(order, remaining_qty=remaining_qty, status=status, queue_ahead_qty=_clean_qty(queue_ahead_after, qty_epsilon), last_update_local_ts_us=event_key.local_ts_us, last_update_event_seq=event_key.event_seq)


def simulate_trade_event(orders: Sequence[ActiveOrder], trade: TradePrint, *, event_key: EventKey, symbol_spec: SymbolSpec, config: FillSimulatorConfig = FillSimulatorConfig()) -> FillSimulationResult:
    orders_tuple = _orders_tuple(orders)
    trade = _require_trade(trade)
    event_key = _require_event_key(event_key, "event_key")
    if trade.local_ts_us != event_key.local_ts_us:
        raise ValueError("trade.local_ts_us must equal event_key.local_ts_us")
    symbol_spec = _require_symbol_spec(symbol_spec)
    config = _require_config(config)
    _assert_no_duplicate_fillable_side_price(orders_tuple, event_key=event_key)

    updated_orders: list[ActiveOrder] = []
    fills: list[Fill] = []
    queue_updates: list[QueueModelUpdate] = []
    for order in orders_tuple:
        if not order.is_fillable_at_key(event_key):
            updated_orders.append(order)
            continue
        _validate_event_key_for_order(order, event_key)
        queue_update = update_queue_position(order, config=config.queue_model, trade=trade)
        if _queue_update_touched(queue_update):
            queue_updates.append(queue_update)
        updated_order, fill = _apply_queue_update_to_order(order, queue_update, event_key=event_key, symbol_spec=symbol_spec, config=config)
        updated_orders.append(updated_order)
        if fill is not None:
            fills.append(fill)
    return FillSimulationResult(tuple(updated_orders), tuple(fills), tuple(queue_updates))


def simulate_l2_level_update(orders: Sequence[ActiveOrder], *, side: OrderSide, price_tick: int, l2_decrease_qty: float, event_key: EventKey, symbol_spec: SymbolSpec, config: FillSimulatorConfig = FillSimulatorConfig()) -> FillSimulationResult:
    orders_tuple = _orders_tuple(orders)
    if not isinstance(side, OrderSide):
        raise ValueError("side must be OrderSide")
    price_tick = _require_positive_int(price_tick, "price_tick")
    l2_decrease_qty = _require_nonnegative_float(l2_decrease_qty, "l2_decrease_qty")
    event_key = _require_event_key(event_key, "event_key")
    symbol_spec = _require_symbol_spec(symbol_spec)
    config = _require_config(config)
    _assert_no_duplicate_fillable_side_price(orders_tuple, event_key=event_key)

    updated_orders: list[ActiveOrder] = []
    fills: list[Fill] = []
    queue_updates: list[QueueModelUpdate] = []
    for order in orders_tuple:
        if not order.is_fillable_at_key(event_key) or order.side != side or order.price_tick != price_tick:
            updated_orders.append(order)
            continue
        _validate_event_key_for_order(order, event_key)
        queue_update = update_queue_position(order, config=config.queue_model, l2_decrease_qty=l2_decrease_qty)
        if _queue_update_touched(queue_update):
            queue_updates.append(queue_update)
        updated_order, fill = _apply_queue_update_to_order(order, queue_update, event_key=event_key, symbol_spec=symbol_spec, config=config)
        updated_orders.append(updated_order)
        if fill is not None:
            fills.append(fill)
    return FillSimulationResult(tuple(updated_orders), tuple(fills), tuple(queue_updates))


def sync_orders_to_quote(
    existing_orders: Sequence[ActiveOrder],
    quote: QuoteIntent,
    *,
    next_order_id: int,
    decision_key: EventKey,
    order_effective_key: EventKey,
    cancel_effective_key: EventKey,
    bid_queue_ahead_qty: float = 0.0,
    ask_queue_ahead_qty: float = 0.0,
    qty_epsilon: float = 1e-12,
) -> tuple[tuple[ActiveOrder, ...], int]:
    existing = _orders_tuple(existing_orders)
    quote = _require_quote(quote)
    next_order_id = _require_nonnegative_int(next_order_id, "next_order_id")
    decision_key = _require_event_key(decision_key, "decision_key")
    order_effective_key = _require_event_key(order_effective_key, "order_effective_key")
    cancel_effective_key = _require_event_key(cancel_effective_key, "cancel_effective_key")
    qty_epsilon = _require_positive_float(qty_epsilon, "qty_epsilon")
    if order_effective_key < decision_key or cancel_effective_key < decision_key:
        raise ValueError("effective keys must be >= decision_key")

    target = {
        OrderSide.BUY: (quote.bid_enabled, quote.bid_price_tick, quote.bid_qty),
        OrderSide.SELL: (quote.ask_enabled, quote.ask_price_tick, quote.ask_qty),
    }
    updated: list[ActiveOrder] = []
    cancel_count = 0
    preserved: set[OrderSide] = set()
    for order in existing:
        if not order.is_live:
            updated.append(order)
            continue
        enabled, desired_price, desired_qty = target[order.side]
        preservable_statuses = (
            OrderStatus.PENDING_NEW,
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
        )
        preserve = (
            order.status in preservable_statuses
            and enabled
            and order.price_tick == desired_price
            and abs(order.remaining_qty - desired_qty) <= qty_epsilon
            and order.cancel_effective_key is None
        )
        if preserve:
            preserved.add(order.side)
            updated.append(order)
        else:
            if order.cancel_effective_key is None:
                cancel_count += 1
            updated.append(request_cancel_live_orders((order,), request_key=decision_key, cancel_effective_key=cancel_effective_key)[0])

    new_quote = QuoteIntent(
        bid_enabled=quote.bid_enabled and OrderSide.BUY not in preserved,
        ask_enabled=quote.ask_enabled and OrderSide.SELL not in preserved,
        bid_price_tick=quote.bid_price_tick if quote.bid_enabled and OrderSide.BUY not in preserved else 0,
        ask_price_tick=quote.ask_price_tick if quote.ask_enabled and OrderSide.SELL not in preserved else 0,
        bid_qty=quote.bid_qty if quote.bid_enabled and OrderSide.BUY not in preserved else 0.0,
        ask_qty=quote.ask_qty if quote.ask_enabled and OrderSide.SELL not in preserved else 0.0,
    )
    new_orders = place_orders_from_quote(new_quote, next_order_id=next_order_id, created_key=decision_key, effective_key=order_effective_key, bid_queue_ahead_qty=bid_queue_ahead_qty, ask_queue_ahead_qty=ask_queue_ahead_qty)
    return tuple(updated) + new_orders, cancel_count


def _queue_update_touched(update: QueueModelUpdate) -> bool:
    return (
        update.advanced_qty > 0.0
        or update.fillable_qty > 0.0
        or update.trade_advance_qty > 0.0
        or update.l2_advance_qty > 0.0
        or update.trade_through
        or update.trade_at_level
    )


def _apply_queue_update_to_order(order: ActiveOrder, queue_update: QueueModelUpdate, *, event_key: EventKey, symbol_spec: SymbolSpec, config: FillSimulatorConfig) -> tuple[ActiveOrder, Fill | None]:
    if not isinstance(queue_update, QueueModelUpdate):
        raise ValueError("queue_update must be QueueModelUpdate")
    if queue_update.fillable_qty > 0.0:
        fill_qty = min(order.remaining_qty, queue_update.fillable_qty)
        fill = _make_fill(order, event_key=event_key, fill_qty=fill_qty, reason=_require_fill_reason(queue_update.fill_reason), queue_ahead_before=queue_update.queue_ahead_before, queue_ahead_after=queue_update.queue_ahead_after, symbol_spec=symbol_spec, maker_fee_bps=config.maker_fee_bps)
        updated_order = apply_fill_to_order(order, fill_qty, queue_ahead_after=queue_update.queue_ahead_after, event_key=event_key, qty_epsilon=config.qty_epsilon)
        return updated_order, fill
    if abs(queue_update.queue_ahead_after - order.queue_ahead_qty) > config.qty_epsilon:
        return (_replace_order_queue_ahead(order, queue_ahead_qty=queue_update.queue_ahead_after, event_key=event_key, qty_epsilon=config.qty_epsilon), None)
    return order, None


def _make_fill(order: ActiveOrder, *, event_key: EventKey, fill_qty: float, reason: FillReason, queue_ahead_before: float, queue_ahead_after: float, symbol_spec: SymbolSpec, maker_fee_bps: float) -> Fill:
    order = _require_order(order)
    event_key = _require_event_key(event_key, "event_key")
    fill_qty = _require_positive_float(fill_qty, "fill_qty")
    reason = _require_fill_reason(reason)
    queue_ahead_before = _require_nonnegative_float(queue_ahead_before, "queue_ahead_before")
    queue_ahead_after = _require_nonnegative_float(queue_ahead_after, "queue_ahead_after")
    symbol_spec = _require_symbol_spec(symbol_spec)
    maker_fee_bps = _require_finite_float(maker_fee_bps, "maker_fee_bps")
    price = symbol_spec.tick_to_price(order.price_tick)
    notional = price * fill_qty * symbol_spec.contract_size
    fee = notional * maker_fee_bps / 10_000.0
    return Fill(order_id=order.order_id, side=order.side, local_ts_us=event_key.local_ts_us, event_seq=event_key.event_seq, price_tick=order.price_tick, qty=fill_qty, fee=fee, reason=reason, queue_ahead_before=queue_ahead_before, queue_ahead_after=queue_ahead_after)


def _replace_order_queue_ahead(order: ActiveOrder, *, queue_ahead_qty: float, event_key: EventKey, qty_epsilon: float) -> ActiveOrder:
    order = _require_order(order)
    if not order.status in (OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL):
        return order
    queue_ahead_qty = _require_nonnegative_float(queue_ahead_qty, "queue_ahead_qty")
    event_key = _require_event_key(event_key, "event_key")
    qty_epsilon = _require_positive_float(qty_epsilon, "qty_epsilon")
    _validate_event_key_for_order(order, event_key)
    queue_ahead_qty = _clean_qty(queue_ahead_qty, qty_epsilon)
    if abs(queue_ahead_qty - order.queue_ahead_qty) <= qty_epsilon:
        return order
    return replace(order, queue_ahead_qty=queue_ahead_qty, last_update_local_ts_us=event_key.local_ts_us, last_update_event_seq=event_key.event_seq)


def _replace_status(order: ActiveOrder, status: OrderStatus, event_key: EventKey, *, remaining_qty: float | None = None) -> ActiveOrder:
    return replace(order, status=status, remaining_qty=order.remaining_qty if remaining_qty is None else remaining_qty, last_update_local_ts_us=event_key.local_ts_us, last_update_event_seq=event_key.event_seq)


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


def _unique_orders_tuple(values: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    orders = _orders_tuple(values)
    seen: set[int] = set()
    for order in orders:
        if order.order_id in seen:
            raise ValueError("order ids must be unique")
        seen.add(order.order_id)
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


def _require_event_key(value: Any, name: str) -> EventKey:
    if not isinstance(value, EventKey):
        raise ValueError(f"{name} must be EventKey")
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


def _validate_event_key_for_order(order: ActiveOrder, event_key: EventKey) -> None:
    order = _require_order(order)
    event_key = _require_event_key(event_key, "event_key")
    if event_key < order.last_update_key:
        raise ValueError("event_key must be >= order.last_update_key")


def _assert_no_duplicate_fillable_side_price(orders: tuple[ActiveOrder, ...], *, event_key: EventKey) -> None:
    seen: set[tuple[OrderSide, int]] = set()
    for order in orders:
        if not order.is_fillable_at_key(event_key):
            continue
        key = (order.side, order.price_tick)
        if key in seen:
            raise ValueError("duplicate fillable orders at same side/price are not supported")
        seen.add(key)


__all__ = [
    "DEFAULT_MAKER_FEE_BPS",
    "FillSimulatorConfig",
    "OrderActivationResult",
    "CancelFinalizationResult",
    "FillSimulationResult",
    "place_orders_from_quote",
    "activate_pending_orders",
    "request_cancel_live_orders",
    "finalize_effective_cancels",
    "sync_orders_to_quote",
    "live_orders",
    "apply_fill_to_order",
    "simulate_trade_event",
    "simulate_l2_level_update",
]
