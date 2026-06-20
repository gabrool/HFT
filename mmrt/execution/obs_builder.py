"""Build fixed-size execution observation vectors from current simulator state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mmrt.time_key import EventKey, MAX_EVENT_SEQ
from mmrt.execution.contracts import (
    ActiveOrder,
    BookTop,
    Fill,
    LinearSignal,
    OrderSide,
    PositionState,
    SymbolSpec,
)
from mmrt.execution.obs_schema import (
    ObservationSchema,
    default_observation_schema,
    validate_observation_vector,
)


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


@dataclass(frozen=True, slots=True)
class ObservationContext:
    current_local_ts_us: int
    episode_start_local_ts_us: int
    current_event_index: int = 0
    total_events: int = 1
    previous_event_local_ts_us: int | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.current_local_ts_us, "current_local_ts_us")
        _require_positive_int(self.episode_start_local_ts_us, "episode_start_local_ts_us")
        if self.current_local_ts_us < self.episode_start_local_ts_us:
            raise ValueError("current_local_ts_us must be >= episode_start_local_ts_us")
        _require_nonnegative_int(self.current_event_index, "current_event_index")
        _require_positive_int(self.total_events, "total_events")
        previous = _optional_positive_int(self.previous_event_local_ts_us, "previous_event_local_ts_us")
        if previous is not None and previous > self.current_local_ts_us:
            raise ValueError("previous_event_local_ts_us must be <= current_local_ts_us")


@dataclass(frozen=True, slots=True)
class ObservationBuilderConfig:
    inventory_qty_reference: float = 0.003
    size_epsilon: float = 1e-12
    max_abs_observation: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inventory_qty_reference",
            _require_positive_float(self.inventory_qty_reference, "inventory_qty_reference"),
        )
        object.__setattr__(self, "size_epsilon", _require_positive_float(self.size_epsilon, "size_epsilon"))
        if self.max_abs_observation is not None:
            object.__setattr__(
                self,
                "max_abs_observation",
                _require_positive_float(self.max_abs_observation, "max_abs_observation"),
            )


@dataclass(frozen=True, slots=True)
class ObservationInput:
    symbol_spec: SymbolSpec
    book_top: BookTop
    bid_depth: int
    ask_depth: int
    linear_signal: LinearSignal
    control_features: Mapping[str, float] = field(default_factory=dict)
    adverse_features: Mapping[str, float] = field(default_factory=dict)
    executable_edge_features: Mapping[str, float] = field(default_factory=dict)

    position: PositionState = PositionState()
    live_orders: tuple[ActiveOrder, ...] = ()
    recent_fills: tuple[Fill, ...] = ()
    context: ObservationContext | None = None

    def __post_init__(self) -> None:
        _require_symbol_spec(self.symbol_spec)
        _require_book_top(self.book_top)
        _require_nonnegative_int(self.bid_depth, "bid_depth")
        _require_nonnegative_int(self.ask_depth, "ask_depth")
        _require_position(self.position)
        object.__setattr__(self, "live_orders", _orders_tuple(self.live_orders))
        object.__setattr__(self, "recent_fills", _fills_tuple(self.recent_fills))
        if not isinstance(self.linear_signal, LinearSignal):
            raise ValueError("linear_signal must be LinearSignal")
        object.__setattr__(self, "control_features", _feature_map(self.control_features, "control_features"))
        object.__setattr__(self, "adverse_features", _feature_map(self.adverse_features, "adverse_features"))
        object.__setattr__(self, "executable_edge_features", _feature_map(self.executable_edge_features, "executable_edge_features"))
        if self.context is not None and not isinstance(self.context, ObservationContext):
            raise ValueError("context must be None or ObservationContext")


@dataclass(frozen=True, slots=True)
class ObservationBuilder:
    schema: ObservationSchema = default_observation_schema()
    config: ObservationBuilderConfig = ObservationBuilderConfig()
    _index_by_name: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ObservationSchema):
            raise ValueError("schema must be ObservationSchema")
        if not isinstance(self.config, ObservationBuilderConfig):
            raise ValueError("config must be ObservationBuilderConfig")
        object.__setattr__(self, "_index_by_name", {name: i for i, name in enumerate(self.schema.field_names)})

    def build(self, inputs: ObservationInput, out: np.ndarray | None = None) -> np.ndarray:
        if not isinstance(inputs, ObservationInput):
            raise ValueError("inputs must be ObservationInput")
        if out is None:
            obs = np.zeros(self.schema.dim, dtype=self.schema.np_dtype)
        else:
            obs = _validate_out_buffer(out, self.schema)
            obs.fill(0.0)

        index_by_name = self._index_by_name

        def set_field(name: str, value: float) -> None:
            idx = index_by_name.get(name)
            if idx is not None:
                obs[idx] = value

        symbol_spec = inputs.symbol_spec
        book_top = inputs.book_top
        config = self.config
        context = inputs.context or ObservationContext(
            current_local_ts_us=book_top.local_ts_us,
            episode_start_local_ts_us=book_top.local_ts_us,
            current_event_index=0,
            total_events=1,
            previous_event_local_ts_us=None,
        )
        if context.current_local_ts_us < book_top.local_ts_us:
            raise ValueError("context.current_local_ts_us must be >= book_top.local_ts_us")

        best_bid_tick = book_top.best_bid_tick
        best_ask_tick = book_top.best_ask_tick
        spread_ticks = best_ask_tick - best_bid_tick
        mid_tick = book_top.mid_tick_x2 * 0.5
        mid_price = mid_tick * symbol_spec.tick_size
        if mid_price <= 0.0 or not math.isfinite(mid_price):
            raise ValueError("mid_price must be positive and finite")
        spread_price = spread_ticks * symbol_spec.tick_size
        spread_bps = spread_price / mid_price * 10_000.0
        bid_top_size = book_top.best_bid_size
        ask_top_size = book_top.best_ask_size
        size_denom = bid_top_size + ask_top_size
        if size_denom > config.size_epsilon:
            top_imbalance = (bid_top_size - ask_top_size) / size_denom
            best_bid_price = best_bid_tick * symbol_spec.tick_size
            best_ask_price = best_ask_tick * symbol_spec.tick_size
            microprice = (best_bid_price * ask_top_size + best_ask_price * bid_top_size) / size_denom
            microprice_bps = (microprice - mid_price) / mid_price * 10_000.0
        else:
            top_imbalance = 0.0
            microprice_bps = 0.0

        set_field("spread_ticks", float(spread_ticks))
        set_field("spread_bps", spread_bps)
        set_field("mid_price", mid_price)
        set_field("microprice_bps", microprice_bps)
        set_field("top_imbalance", top_imbalance)
        set_field("bid_depth_count", float(inputs.bid_depth))
        set_field("ask_depth_count", float(inputs.ask_depth))
        set_field("bid_top_size", bid_top_size)
        set_field("ask_top_size", ask_top_size)

        for name, value in inputs.control_features.items():
            set_field(name, value)

        signal = inputs.linear_signal
        set_field("linear_p_no_move", signal.p_no_move)
        set_field("linear_p_move", signal.p_move)
        set_field("linear_p_up_move", signal.p_up_move)
        set_field("linear_p_down_move", signal.p_down_move)
        set_field("linear_signed_move_prob", signal.signed_move_prob)
        set_field("linear_expected_up_bps", signal.expected_up_bps)
        set_field("linear_expected_down_bps", signal.expected_down_bps)
        set_field("linear_expected_return_bps", signal.expected_return_bps)
        set_field("linear_expected_abs_move_bps", signal.expected_abs_move_bps)
        set_field("linear_predicted_vol_bps", signal.predicted_vol_bps)
        set_field("linear_confidence", signal.confidence)

        position = inputs.position
        inventory_notional = position.inventory_qty * mid_price * symbol_spec.contract_size
        inventory_abs_notional = abs(inventory_notional)
        equity = position.mark_to_market(mid_price, symbol_spec.contract_size)
        inventory_order_units = position.inventory_qty / max(config.inventory_qty_reference, config.size_epsilon)
        set_field("cash", position.cash)
        set_field("inventory_qty", position.inventory_qty)
        set_field("inventory_notional", inventory_notional)
        set_field("inventory_order_units", inventory_order_units)
        set_field("equity", equity)
        set_field("inventory_abs_notional", inventory_abs_notional)
        set_field("fees_paid", position.fees_paid)

        bid_order, ask_order = _live_orders_by_side(inputs.live_orders, local_ts_us=context.current_local_ts_us)
        if bid_order is not None:
            bid_distance_ticks = max(book_top.best_bid_tick - bid_order.price_tick, 0)
            set_field("has_live_bid", 1.0)
            set_field("bid_distance_ticks", float(bid_distance_ticks))
            set_field("bid_distance_bps", bid_distance_ticks * symbol_spec.tick_size / mid_price * 10_000.0)
            set_field("bid_qty", bid_order.qty)
            set_field("bid_remaining_qty", bid_order.remaining_qty)
            set_field("bid_queue_ahead_qty", bid_order.queue_ahead_qty)
            set_field("bid_age_ms", max(context.current_local_ts_us - bid_order.created_local_ts_us, 0) / 1000.0)
        if ask_order is not None:
            ask_distance_ticks = max(ask_order.price_tick - book_top.best_ask_tick, 0)
            set_field("has_live_ask", 1.0)
            set_field("ask_distance_ticks", float(ask_distance_ticks))
            set_field("ask_distance_bps", ask_distance_ticks * symbol_spec.tick_size / mid_price * 10_000.0)
            set_field("ask_qty", ask_order.qty)
            set_field("ask_remaining_qty", ask_order.remaining_qty)
            set_field("ask_queue_ahead_qty", ask_order.queue_ahead_qty)
            set_field("ask_age_ms", max(context.current_local_ts_us - ask_order.created_local_ts_us, 0) / 1000.0)

        recent_fills = inputs.recent_fills
        if recent_fills:
            step_fill_notional = 0.0
            step_buy_fill_qty = 0.0
            step_sell_fill_qty = 0.0
            for fill in recent_fills:
                if fill.local_ts_us > context.current_local_ts_us:
                    raise ValueError("fill local_ts_us must be <= current_local_ts_us")
                notional = _fill_notional(fill, symbol_spec)
                step_fill_notional += notional
                if fill.side == OrderSide.BUY:
                    step_buy_fill_qty += fill.qty
                elif fill.side == OrderSide.SELL:
                    step_sell_fill_qty += fill.qty
                else:
                    raise ValueError(f"unsupported fill side: {fill.side!r}")

            last_fill = recent_fills[-1]
            last_fill_notional = _fill_notional(last_fill, symbol_spec)
            last_fill_side = 1.0 if last_fill.side == OrderSide.BUY else -1.0
            set_field("last_fill_side", last_fill_side)
            set_field("last_fill_qty", last_fill.qty)
            set_field("last_fill_notional", last_fill_notional)
            set_field("last_fill_fee", last_fill.fee)
            set_field("last_fill_age_ms", max(context.current_local_ts_us - last_fill.local_ts_us, 0) / 1000.0)
            set_field("step_fill_count", float(len(recent_fills)))
            set_field("step_buy_fill_qty", step_buy_fill_qty)
            set_field("step_sell_fill_qty", step_sell_fill_qty)
            set_field("step_fill_notional", step_fill_notional)

        local_time_since_start_s = (context.current_local_ts_us - context.episode_start_local_ts_us) / 1_000_000.0
        if context.previous_event_local_ts_us is None:
            time_since_last_event_ms = 0.0
        else:
            time_since_last_event_ms = (context.current_local_ts_us - context.previous_event_local_ts_us) / 1000.0
        set_field("local_time_since_start_s", local_time_since_start_s)
        set_field("time_since_last_event_ms", time_since_last_event_ms)

        for name, value in inputs.adverse_features.items():
            set_field(name, value)
        for name, value in inputs.executable_edge_features.items():
            set_field(name, value)

        if config.max_abs_observation is not None:
            np.clip(obs, -config.max_abs_observation, config.max_abs_observation, out=obs)
        return validate_observation_vector(obs, schema=self.schema)


def _validate_out_buffer(out: np.ndarray, schema: ObservationSchema) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise ValueError("out must be a NumPy array")
    if out.shape != (schema.dim,):
        raise ValueError(f"out shape must be ({schema.dim},)")
    if np.dtype(out.dtype) != schema.np_dtype:
        raise ValueError(f"out dtype must be {schema.np_dtype}")
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    return out


def build_observation(
    inputs: ObservationInput,
    *,
    schema: ObservationSchema = default_observation_schema(),
    config: ObservationBuilderConfig = ObservationBuilderConfig(),
    out: np.ndarray | None = None,
) -> np.ndarray:
    return ObservationBuilder(schema=schema, config=config).build(inputs, out=out)



def _feature_map(values: Mapping[str, float], name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    out: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        out[key] = _require_finite_float(value, f"{name}[{key!r}]")
    return out

def _require_symbol_spec(value: Any) -> SymbolSpec:
    if not isinstance(value, SymbolSpec):
        raise ValueError("symbol_spec must be SymbolSpec")
    return value


def _require_book_top(value: Any) -> BookTop:
    if not isinstance(value, BookTop):
        raise ValueError("book_top must be BookTop")
    return value


def _require_position(value: Any) -> PositionState:
    if not isinstance(value, PositionState):
        raise ValueError("position must be PositionState")
    return value


def _orders_tuple(values: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("live_orders must be a sequence of ActiveOrder")
    try:
        seq = tuple(values)
    except TypeError as exc:
        raise ValueError("live_orders must be a sequence of ActiveOrder") from exc
    for i, order in enumerate(seq):
        if not isinstance(order, ActiveOrder):
            raise ValueError(f"live_orders[{i}] must be ActiveOrder")
    return seq


def _fills_tuple(values: Sequence[Fill]) -> tuple[Fill, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("recent_fills must be a sequence of Fill")
    try:
        seq = tuple(values)
    except TypeError as exc:
        raise ValueError("recent_fills must be a sequence of Fill") from exc
    for i, fill in enumerate(seq):
        if not isinstance(fill, Fill):
            raise ValueError(f"recent_fills[{i}] must be Fill")
    return seq


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _fill_notional(fill: Fill, symbol_spec: SymbolSpec) -> float:
    return symbol_spec.tick_to_price(fill.price_tick) * fill.qty * symbol_spec.contract_size


def _live_orders_by_side(orders: tuple[ActiveOrder, ...], *, local_ts_us: int) -> tuple[ActiveOrder | None, ActiveOrder | None]:
    bid_order: ActiveOrder | None = None
    ask_order: ActiveOrder | None = None
    for order in orders:
        if not order.is_fillable_at_key(EventKey(local_ts_us, MAX_EVENT_SEQ)):
            continue
        if order.side == OrderSide.BUY:
            if bid_order is not None:
                raise ValueError("observation builder supports at most one live order per side")
            bid_order = order
        elif order.side == OrderSide.SELL:
            if ask_order is not None:
                raise ValueError("observation builder supports at most one live order per side")
            ask_order = order
        else:
            raise ValueError(f"unsupported order side: {order.side!r}")
    return bid_order, ask_order


__all__ = [
    "ObservationContext",
    "ObservationBuilderConfig",
    "ObservationInput",
    "ObservationBuilder",
    "build_observation",
]
