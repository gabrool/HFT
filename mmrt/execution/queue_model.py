"""Pure queue-position primitives for passive execution simulation.

The queue model estimates how much displayed quantity ahead of a resting order
is consumed by trades and, in balanced mode, by visible L2 size decreases. It
returns fillability signals for a later fill simulator; it does not mutate
orders or construct fills.
"""

from dataclasses import dataclass
from typing import Any

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActiveOrder,
    FillReason,
    OrderSide,
    QueueModelMode,
    TradePrint,
)


_INF = float("inf")


def _clean_qty(value: float, eps: float) -> float:
    return 0.0 if abs(value) <= eps else max(value, 0.0)


def _coerce_mode(value: QueueModelMode | str) -> QueueModelMode:
    if isinstance(value, QueueModelMode):
        return value
    if isinstance(value, str):
        try:
            return QueueModelMode(value)
        except ValueError as exc:
            raise ValueError(f"mode has invalid value {value!r}") from exc
    raise ValueError("mode must be QueueModelMode or str")


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    value = float(value)
    if value != value or value in (_INF, -_INF):
        raise ValueError(f"{name} must be a finite float")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_probability(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_live_order(order: ActiveOrder) -> ActiveOrder:
    order = _require_order(order)
    if not order.is_live:
        raise ValueError("order must be live")
    return order


def _optional_nonnegative_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _require_nonnegative_float(value, name)


def _require_config(config: Any) -> Any:
    if not isinstance(config, QueueModelConfig):
        raise ValueError("config must be QueueModelConfig")
    return config


def _require_order(order: Any) -> ActiveOrder:
    if not isinstance(order, ActiveOrder):
        raise ValueError("order must be ActiveOrder")
    return order


def _require_trade(trade: Any) -> TradePrint:
    if not isinstance(trade, TradePrint):
        raise ValueError("trade must be TradePrint")
    return trade


@dataclass(frozen=True, slots=True)
class QueueModelConfig:
    mode: QueueModelMode = QueueModelMode.CONSERVATIVE
    l2_decrease_weight: float = 0.5
    trade_at_level_weight: float = 1.0
    unknown_level_queue_ahead_qty: float = 0.0
    qty_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _coerce_mode(self.mode))
        object.__setattr__(self, "l2_decrease_weight", _require_probability(self.l2_decrease_weight, "l2_decrease_weight"))
        object.__setattr__(self, "trade_at_level_weight", _require_probability(self.trade_at_level_weight, "trade_at_level_weight"))
        object.__setattr__(
            self,
            "unknown_level_queue_ahead_qty",
            _require_nonnegative_float(self.unknown_level_queue_ahead_qty, "unknown_level_queue_ahead_qty"),
        )
        object.__setattr__(self, "qty_epsilon", _require_positive_float(self.qty_epsilon, "qty_epsilon"))


@dataclass(frozen=True, slots=True)
class QueueModelUpdate:
    queue_ahead_before: float
    queue_ahead_after: float
    advanced_qty: float
    fillable_qty: float
    trade_advance_qty: float = 0.0
    l2_advance_qty: float = 0.0
    trade_through: bool = False
    trade_at_level: bool = False
    fill_reason: FillReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue_ahead_before", _require_nonnegative_float(self.queue_ahead_before, "queue_ahead_before"))
        object.__setattr__(self, "queue_ahead_after", _require_nonnegative_float(self.queue_ahead_after, "queue_ahead_after"))
        object.__setattr__(self, "advanced_qty", _require_nonnegative_float(self.advanced_qty, "advanced_qty"))
        fillable_qty = _require_nonnegative_float(self.fillable_qty, "fillable_qty")
        object.__setattr__(self, "fillable_qty", fillable_qty)
        object.__setattr__(self, "trade_advance_qty", _require_nonnegative_float(self.trade_advance_qty, "trade_advance_qty"))
        object.__setattr__(self, "l2_advance_qty", _require_nonnegative_float(self.l2_advance_qty, "l2_advance_qty"))
        object.__setattr__(self, "trade_through", _require_bool(self.trade_through, "trade_through"))
        object.__setattr__(self, "trade_at_level", _require_bool(self.trade_at_level, "trade_at_level"))
        if self.fill_reason is not None and not isinstance(self.fill_reason, FillReason):
            raise ValueError("fill_reason must be FillReason or None")
        if fillable_qty > 0.0 and self.fill_reason is None:
            raise ValueError("fillable_qty > 0 requires fill_reason")


def estimate_initial_queue_ahead(
    visible_level_qty: float | None,
    *,
    config: QueueModelConfig = QueueModelConfig(),
) -> float:
    config = _require_config(config)
    if visible_level_qty is None:
        return config.unknown_level_queue_ahead_qty
    return _require_nonnegative_float(visible_level_qty, "visible_level_qty")


def is_trade_relevant_to_order(order: ActiveOrder, trade: TradePrint) -> bool:
    _require_order(order)
    _require_trade(trade)
    if order.side == OrderSide.BUY:
        return trade.side == AggressorSide.SELL
    if order.side == OrderSide.SELL:
        return trade.side == AggressorSide.BUY
    return False


def classify_trade_against_order(order: ActiveOrder, trade: TradePrint) -> FillReason | None:
    _require_order(order)
    _require_trade(trade)
    if not is_trade_relevant_to_order(order, trade):
        return None

    if order.side == OrderSide.BUY:
        if trade.price_tick < order.price_tick:
            return FillReason.TRADE_THROUGH
        if trade.price_tick == order.price_tick:
            return FillReason.TRADE_AT_LEVEL
        return None

    if trade.price_tick > order.price_tick:
        return FillReason.TRADE_THROUGH
    if trade.price_tick == order.price_tick:
        return FillReason.TRADE_AT_LEVEL
    return None


def update_queue_position(
    order: ActiveOrder,
    *,
    config: QueueModelConfig = QueueModelConfig(),
    trade: TradePrint | None = None,
    prev_level_qty: float | None = None,
    curr_level_qty: float | None = None,
) -> QueueModelUpdate:
    order = _require_live_order(order)
    config = _require_config(config)

    queue_before = max(order.queue_ahead_qty, 0.0)
    queue_after = queue_before
    fillable_qty = 0.0
    trade_advance_qty = 0.0
    l2_advance_qty = 0.0
    fill_reason: FillReason | None = None
    trade_through = False
    trade_at_level = False

    if trade is not None:
        _require_trade(trade)
        classification = classify_trade_against_order(order, trade)
        if classification == FillReason.TRADE_THROUGH:
            trade_through = True
            trade_advance_qty = queue_after
            queue_after = 0.0
            fillable_qty = order.remaining_qty
            fill_reason = FillReason.TRADE_THROUGH
        elif classification == FillReason.TRADE_AT_LEVEL:
            trade_at_level = True
            effective_trade_qty = trade.amount * config.trade_at_level_weight
            queue_consumed = min(queue_after, effective_trade_qty)
            queue_after = max(queue_after - effective_trade_qty, 0.0)
            leftover = max(effective_trade_qty - queue_consumed, 0.0)
            fillable_qty = min(order.remaining_qty, leftover)
            trade_advance_qty = queue_consumed
            if fillable_qty > 0.0:
                fill_reason = FillReason.TRADE_AT_LEVEL

    if (
        config.mode == QueueModelMode.BALANCED
        and prev_level_qty is not None
        and curr_level_qty is not None
        and fillable_qty < order.remaining_qty
    ):
        prev_qty = _require_nonnegative_float(prev_level_qty, "prev_level_qty")
        curr_qty = _require_nonnegative_float(curr_level_qty, "curr_level_qty")
        visible_decrease = max(prev_qty - curr_qty, 0.0)
        effective_l2_advance = visible_decrease * config.l2_decrease_weight
        queue_consumed = min(queue_after, effective_l2_advance)
        queue_after = max(queue_after - effective_l2_advance, 0.0)
        leftover = max(effective_l2_advance - queue_consumed, 0.0)
        additional_fillable = min(order.remaining_qty - fillable_qty, leftover)
        fillable_qty += additional_fillable
        l2_advance_qty = queue_consumed
        if additional_fillable > 0.0 and fill_reason is None:
            fill_reason = FillReason.QUEUE_DEPLETION

    queue_after = _clean_qty(queue_after, config.qty_epsilon)
    trade_advance_qty = _clean_qty(trade_advance_qty, config.qty_epsilon)
    l2_advance_qty = _clean_qty(l2_advance_qty, config.qty_epsilon)
    advanced_qty = _clean_qty(min(queue_before, trade_advance_qty + l2_advance_qty), config.qty_epsilon)
    fillable_qty = _clean_qty(min(fillable_qty, order.remaining_qty), config.qty_epsilon)
    if fillable_qty <= 0.0:
        fill_reason = None

    return QueueModelUpdate(
        queue_ahead_before=queue_before,
        queue_ahead_after=queue_after,
        advanced_qty=advanced_qty,
        fillable_qty=fillable_qty,
        trade_advance_qty=trade_advance_qty,
        l2_advance_qty=l2_advance_qty,
        trade_through=trade_through,
        trade_at_level=trade_at_level,
        fill_reason=fill_reason,
    )


__all__ = [
    "QueueModelConfig",
    "QueueModelUpdate",
    "estimate_initial_queue_ahead",
    "is_trade_relevant_to_order",
    "classify_trade_against_order",
    "update_queue_position",
]
