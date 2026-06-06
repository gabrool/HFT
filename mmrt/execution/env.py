"""Plain Python execution replay environment for an :class:`ExecutionTape`.

The environment owns only simulator state: it replays already-built tape arrays,
turns one continuous action per decision interval into maker-safe quotes,
simulates fills, computes rewards, and builds fixed observation vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActionSpec,
    ActiveOrder,
    BookTop,
    ExecutionStepResult,
    Fill,
    OrderSide,
    PositionState,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.execution_tape import (
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    ExecutionTape,
)
from mmrt.execution.fill_sim import (
    FillSimulatorConfig,
    replace_orders_from_quote,
    simulate_l2_level_update,
    simulate_trade_event,
)
from mmrt.execution.linear_signal import LinearSignalArrays, linear_signal_at
from mmrt.execution.obs_builder import (
    ObservationBuilder,
    ObservationBuilderConfig,
    ObservationContext,
    ObservationInput,
)
from mmrt.execution.obs_schema import ObservationSchema, default_observation_schema
from mmrt.execution.queue_model import QueueModelConfig, QueueModelMode, estimate_initial_queue_ahead
from mmrt.execution.quote_geometry import (
    ContinuousQuoteAction,
    QuoteGeometryConfig,
    continuous_action_to_quote,
)
from mmrt.execution.reward import RewardConfig, compute_reward_step



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


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _require_position(value: Any) -> PositionState:
    if not isinstance(value, PositionState):
        raise ValueError("position must be PositionState")
    return value

__all__ = [
    "ExecutionEnvConfig",
    "ExecutionEnvReset",
    "ExecutionEnvStep",
    "ExecutionEnv",
    "action_array_to_continuous_action",
]


@dataclass(frozen=True, slots=True)
class ExecutionEnvConfig:
    decision_interval_us: int = 500_000
    action_spec: ActionSpec = field(default_factory=ActionSpec)
    quote_geometry_config: QuoteGeometryConfig = field(default_factory=QuoteGeometryConfig)
    fill_simulator_config: FillSimulatorConfig = field(
        default_factory=lambda: FillSimulatorConfig(queue_model=QueueModelConfig(mode=QueueModelMode.BALANCED))
    )
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    observation_schema: ObservationSchema = field(default_factory=default_observation_schema)
    observation_builder_config: ObservationBuilderConfig = field(default_factory=ObservationBuilderConfig)
    initial_position: PositionState = field(default_factory=PositionState)
    max_episode_steps: int | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.decision_interval_us, "decision_interval_us")
        if not isinstance(self.action_spec, ActionSpec):
            raise ValueError("action_spec must be ActionSpec")
        if not isinstance(self.quote_geometry_config, QuoteGeometryConfig):
            raise ValueError("quote_geometry_config must be QuoteGeometryConfig")
        if not isinstance(self.fill_simulator_config, FillSimulatorConfig):
            raise ValueError("fill_simulator_config must be FillSimulatorConfig")
        if not isinstance(self.reward_config, RewardConfig):
            raise ValueError("reward_config must be RewardConfig")
        if not isinstance(self.observation_schema, ObservationSchema):
            raise ValueError("observation_schema must be ObservationSchema")
        if not isinstance(self.observation_builder_config, ObservationBuilderConfig):
            raise ValueError("observation_builder_config must be ObservationBuilderConfig")
        _require_position(self.initial_position)
        object.__setattr__(self, "max_episode_steps", _optional_positive_int(self.max_episode_steps, "max_episode_steps"))


@dataclass(frozen=True, slots=True)
class ExecutionEnvReset:
    observation: np.ndarray
    info: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.observation, np.ndarray):
            raise ValueError("observation must be a NumPy array")
        if not isinstance(self.info, dict):
            raise ValueError("info must be dict")


@dataclass(frozen=True, slots=True)
class ExecutionEnvStep:
    observation: np.ndarray
    reward: float
    execution: ExecutionStepResult

    def __post_init__(self) -> None:
        if not isinstance(self.observation, np.ndarray):
            raise ValueError("observation must be a NumPy array")
        object.__setattr__(self, "reward", _require_finite_float(self.reward, "reward"))
        if not isinstance(self.execution, ExecutionStepResult):
            raise ValueError("execution must be ExecutionStepResult")

    @property
    def done(self) -> bool:
        return self.execution.done

    @property
    def truncated(self) -> bool:
        return self.execution.truncated

    @property
    def info(self) -> dict[str, object]:
        return self.execution.info or {}

    @property
    def fills(self) -> tuple[Fill, ...]:
        return self.execution.fills

    @property
    def position(self) -> PositionState:
        return self.execution.position


@dataclass(slots=True)
class _EnvState:
    event_index: int
    current_book_ptr: int
    previous_event_local_ts_us: int | None
    step_index: int
    next_order_id: int
    position: PositionState
    live_orders: tuple[ActiveOrder, ...]
    done: bool
    truncated: bool


def action_array_to_continuous_action(values: Sequence[float] | np.ndarray) -> ContinuousQuoteAction:
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be a sequence or ndarray of six finite numeric values")
    try:
        seq = tuple(values)
    except TypeError as exc:
        raise ValueError("values must be a sequence or ndarray of six finite numeric values") from exc
    if len(seq) != 6:
        raise ValueError("values must contain exactly 6 values")
    cleaned = tuple(_require_finite_float(value, f"values[{i}]") for i, value in enumerate(seq))
    return ContinuousQuoteAction(
        bid_enable_logit=cleaned[0],
        ask_enable_logit=cleaned[1],
        bid_distance_raw=cleaned[2],
        ask_distance_raw=cleaned[3],
        bid_size_raw=cleaned[4],
        ask_size_raw=cleaned[5],
    )


class ExecutionEnv:
    def __init__(
        self,
        tape: ExecutionTape,
        *,
        linear_signals: LinearSignalArrays,
        config: ExecutionEnvConfig = ExecutionEnvConfig(),
    ) -> None:
        self.tape = _require_tape(tape)
        self.config = _require_config(config)
        if not isinstance(linear_signals, LinearSignalArrays):
            raise ValueError("linear_signals must be LinearSignalArrays")
        if linear_signals.n_rows <= 0:
            raise ValueError("linear_signals must contain at least one row")
        if config.max_episode_steps is not None and linear_signals.n_rows < config.max_episode_steps + 1:
            raise ValueError("linear_signals must contain at least max_episode_steps + 1 rows")
        self.linear_signals = linear_signals
        self.observation_builder = ObservationBuilder(
            schema=config.observation_schema,
            config=config.observation_builder_config,
        )
        self._obs_buffer = np.zeros(config.observation_schema.dim, dtype=config.observation_schema.np_dtype)
        self._state: _EnvState | None = None
        self._episode_start_local_ts_us = 0
        self._last_step_fills: tuple[Fill, ...] = ()
        self._peak_equity: float | None = None

    def reset(
        self,
        *,
        start_event_index: int | None = None,
        initial_position: PositionState | None = None,
    ) -> ExecutionEnvReset:
        events = self.tape.arrays.events
        if start_event_index is None:
            start = 0
        else:
            start = _require_nonnegative_int(start_event_index, "start_event_index")
            if start >= len(events):
                raise ValueError("start_event_index must be < len(tape.arrays.events)")

        found_event_index: int | None = None
        found_book_ptr: int | None = None
        for event_index in range(start, len(events)):
            event = events[event_index]
            if int(event["event_type_code"]) != EVENT_TYPE_CODE_L2_BATCH:
                continue
            book_ptr = int(event["book_ptr"])
            if book_ptr >= 0 and self._book_top_from_ptr(book_ptr) is not None:
                found_event_index = event_index
                found_book_ptr = book_ptr
                break
        if found_event_index is None or found_book_ptr is None:
            raise ValueError("execution tape contains no valid two-sided L2 book event")

        position = self.config.initial_position if initial_position is None else _require_position(initial_position)
        self._state = _EnvState(
            event_index=found_event_index,
            current_book_ptr=found_book_ptr,
            previous_event_local_ts_us=None,
            step_index=0,
            next_order_id=0,
            position=position,
            live_orders=(),
            done=False,
            truncated=False,
        )
        self._episode_start_local_ts_us = int(self.tape.arrays.l2_events[found_book_ptr]["local_ts_us"])
        self._last_step_fills = ()
        self._peak_equity = None
        observation = self._build_observation()
        return ExecutionEnvReset(
            observation=observation,
            info={
                "event_index": found_event_index,
                "current_book_ptr": found_book_ptr,
                "step_index": 0,
            },
        )

    def step(self, action: ContinuousQuoteAction | Sequence[float] | np.ndarray) -> ExecutionEnvStep:
        state = self._require_state()
        if state.done or state.truncated:
            raise RuntimeError("environment is done; call reset() before step()")

        events = self.tape.arrays.events
        num_events = len(events)
        if state.event_index + 1 >= num_events:
            raise RuntimeError("cannot step: no future tape events remain")

        action = _coerce_action(action)
        symbol_spec = self.tape.manifest.symbol_spec
        previous_position = state.position
        previous_book_top = self._current_book_top()
        quote_result = continuous_action_to_quote(
            action=action,
            book_top=previous_book_top,
            symbol_spec=symbol_spec,
            action_spec=self.config.action_spec,
            config=self.config.quote_geometry_config,
            inventory_qty=state.position.inventory_qty,
        )
        quote = quote_result.quote

        bid_queue_ahead_qty = 0.0
        if quote.bid_enabled:
            bid_queue_ahead_qty = estimate_initial_queue_ahead(
                self._level_qty(book_ptr=state.current_book_ptr, side=OrderSide.BUY, price_tick=quote.bid_price_tick),
                config=self.config.fill_simulator_config.queue_model,
            )
        ask_queue_ahead_qty = 0.0
        if quote.ask_enabled:
            ask_queue_ahead_qty = estimate_initial_queue_ahead(
                self._level_qty(book_ptr=state.current_book_ptr, side=OrderSide.SELL, price_tick=quote.ask_price_tick),
                config=self.config.fill_simulator_config.queue_model,
            )

        cancel_count = len(state.live_orders)
        replacement_orders = replace_orders_from_quote(
            state.live_orders,
            quote,
            next_order_id=state.next_order_id,
            local_ts_us=previous_book_top.local_ts_us,
            bid_queue_ahead_qty=bid_queue_ahead_qty,
            ask_queue_ahead_qty=ask_queue_ahead_qty,
        )
        state.live_orders = _live_orders_tuple(replacement_orders)
        state.next_order_id = _next_order_id_after(replacement_orders, state.next_order_id)

        decision_end_local_ts_us = previous_book_top.local_ts_us + self.config.decision_interval_us
        next_event_index = state.event_index + 1
        processed_any = False
        processed_valid_l2 = False
        events_processed = 0
        fills: list[Fill] = []
        truncated = False

        while next_event_index < num_events:
            event_local = int(events[next_event_index]["local_ts_us"])
            if processed_any and processed_valid_l2 and event_local > decision_end_local_ts_us:
                break

            event_code = int(events[next_event_index]["event_type_code"])
            event_book_ptr = int(events[next_event_index]["book_ptr"])
            fills.extend(self._process_event(next_event_index))
            if event_code == EVENT_TYPE_CODE_L2_BATCH and event_book_ptr >= 0:
                processed_valid_l2 = processed_valid_l2 or self._book_top_from_ptr(event_book_ptr) is not None
            processed_any = True
            events_processed += 1
            next_event_index += 1

            if self.config.max_episode_steps is not None and state.step_index + 1 >= self.config.max_episode_steps:
                truncated = True
                break

        done = False if truncated else next_event_index >= num_events
        state.done = done
        state.truncated = truncated

        current_book_top = self._current_book_top()
        step_fills = tuple(fills)
        reward_step = compute_reward_step(
            previous_position=previous_position,
            fills=step_fills,
            previous_book_top=previous_book_top,
            current_book_top=current_book_top,
            symbol_spec=symbol_spec,
            config=self.config.reward_config,
            cancel_count=cancel_count,
            peak_equity=self._peak_equity,
            terminal=done or truncated,
        )
        state.position = reward_step.position
        self._peak_equity = reward_step.peak_equity
        self._last_step_fills = step_fills
        state.step_index += 1

        info: dict[str, object] = {
            "step_index": state.step_index - 1,
            "event_index": state.event_index,
            "current_book_ptr": state.current_book_ptr,
            "events_processed": events_processed,
            "cancel_count": cancel_count,
            "num_fills": len(step_fills),
            "quote_bid_enabled": quote.bid_enabled,
            "quote_ask_enabled": quote.ask_enabled,
            "quote_bid_price_tick": quote.bid_price_tick,
            "quote_ask_price_tick": quote.ask_price_tick,
            "quote_bid_qty": quote.bid_qty,
            "quote_ask_qty": quote.ask_qty,
            "quote_bid_disabled_reason": quote_result.bid_disabled_reason,
            "quote_ask_disabled_reason": quote_result.ask_disabled_reason,
            "previous_equity": reward_step.previous_equity,
            "current_equity": reward_step.current_equity,
            "peak_equity": reward_step.peak_equity,
            "turnover_notional": reward_step.turnover_notional,
        }
        execution = ExecutionStepResult(
            reward=reward_step.reward,
            position=reward_step.position,
            fills=step_fills,
            done=done,
            truncated=truncated,
            info=info,
        )
        observation = self._build_observation()
        return ExecutionEnvStep(
            observation=observation,
            reward=reward_step.reward.total_reward,
            execution=execution,
        )

    def _build_observation(self) -> np.ndarray:
        state = self._require_state()
        book_top = self._current_book_top()
        l2_row = self.tape.arrays.l2_events[state.current_book_ptr]
        current_local_ts_us = book_top.local_ts_us
        if self._last_step_fills:
            current_local_ts_us = max(current_local_ts_us, max(fill.local_ts_us for fill in self._last_step_fills))
        context = ObservationContext(
            current_local_ts_us=current_local_ts_us,
            episode_start_local_ts_us=self._episode_start_local_ts_us,
            current_event_index=state.event_index,
            total_events=len(self.tape.arrays.events),
            previous_event_local_ts_us=state.previous_event_local_ts_us,
        )
        inputs = ObservationInput(
            symbol_spec=self.tape.manifest.symbol_spec,
            book_top=book_top,
            bid_depth=int(l2_row["bid_depth"]),
            ask_depth=int(l2_row["ask_depth"]),
            position=state.position,
            live_orders=state.live_orders,
            recent_fills=self._last_step_fills,
            linear_signal=self._linear_signal_for_step(state.step_index),
            context=context,
        )
        return self.observation_builder.build(inputs, out=self._obs_buffer)

    def _linear_signal_for_step(self, step_index: int):
        if step_index >= self.linear_signals.n_rows:
            raise ValueError("linear_signals does not contain a row for current step_index")
        return linear_signal_at(self.linear_signals, step_index)

    def _process_event(self, event_index: int) -> tuple[Fill, ...]:
        state = self._require_state()
        event = self.tape.arrays.events[event_index]
        code = int(event["event_type_code"])
        old_event_local = int(self.tape.arrays.events[state.event_index]["local_ts_us"])
        fills: tuple[Fill, ...] = ()
        if code == EVENT_TYPE_CODE_L2_BATCH:
            book_ptr = int(event["book_ptr"])
            fills = self._process_l2_event(book_ptr)
            if self._book_top_from_ptr(book_ptr) is not None:
                state.current_book_ptr = book_ptr
        elif code == EVENT_TYPE_CODE_TRADE:
            trade = self._trade_from_ptr(int(event["trade_ptr"]))
            result = simulate_trade_event(
                state.live_orders,
                trade,
                symbol_spec=self.tape.manifest.symbol_spec,
                config=self.config.fill_simulator_config,
            )
            state.live_orders = _live_orders_tuple(result.orders)
            fills = result.fills
        state.previous_event_local_ts_us = old_event_local
        state.event_index = event_index
        return fills

    def _process_l2_event(self, curr_book_ptr: int) -> tuple[Fill, ...]:
        state = self._require_state()
        if curr_book_ptr < 0:
            return ()
        local_ts_us = int(self.tape.arrays.l2_events[curr_book_ptr]["local_ts_us"])
        updated_orders = state.live_orders
        fills: list[Fill] = []
        for order in tuple(updated_orders):
            if not order.is_live:
                continue
            prev_qty = self._level_qty(book_ptr=state.current_book_ptr, side=order.side, price_tick=order.price_tick)
            curr_qty = self._level_qty(book_ptr=curr_book_ptr, side=order.side, price_tick=order.price_tick)
            result = simulate_l2_level_update(
                updated_orders,
                side=order.side,
                price_tick=order.price_tick,
                prev_level_qty=prev_qty if prev_qty is not None else 0.0,
                curr_level_qty=curr_qty if curr_qty is not None else 0.0,
                local_ts_us=local_ts_us,
                symbol_spec=self.tape.manifest.symbol_spec,
                config=self.config.fill_simulator_config,
            )
            updated_orders = _live_orders_tuple(result.orders)
            fills.extend(result.fills)
        state.live_orders = _live_orders_tuple(updated_orders)
        return tuple(fills)

    def _trade_from_ptr(self, trade_ptr: int) -> TradePrint:
        row = self.tape.arrays.trades[trade_ptr]
        source_row = int(row["source_row"])
        return TradePrint(
            local_ts_us=int(row["local_ts_us"]),
            ts_us=int(row["ts_us"]),
            side=_trade_side_from_code(int(row["side_code"])),
            price_tick=int(row["price_tick"]),
            amount=float(row["amount"]),
            trade_id=str(source_row),
            source_row=source_row,
        )

    def _level_qty(self, *, book_ptr: int, side: OrderSide, price_tick: int) -> float | None:
        if side == OrderSide.BUY:
            ticks = self.tape.arrays.book_bid_ticks[book_ptr]
            sizes = self.tape.arrays.book_bid_sizes[book_ptr]
        elif side == OrderSide.SELL:
            ticks = self.tape.arrays.book_ask_ticks[book_ptr]
            sizes = self.tape.arrays.book_ask_sizes[book_ptr]
        else:
            raise ValueError("side must be OrderSide")
        for i in range(ticks.shape[0]):
            tick = int(ticks[i])
            if tick == 0:
                break
            if tick == price_tick:
                return float(sizes[i])
        return None

    def _book_top_from_ptr(self, book_ptr: int) -> BookTop | None:
        row = self.tape.arrays.l2_events[book_ptr]
        best_bid_tick = int(row["best_bid_tick"])
        best_ask_tick = int(row["best_ask_tick"])
        if best_bid_tick <= 0 or best_ask_tick <= 0 or best_bid_tick >= best_ask_tick:
            return None
        return BookTop(
            local_ts_us=int(row["local_ts_us"]),
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            best_bid_size=float(row["best_bid_size"]),
            best_ask_size=float(row["best_ask_size"]),
        )

    def _current_book_top(self) -> BookTop:
        state = self._require_state()
        top = self._book_top_from_ptr(state.current_book_ptr)
        if top is None:
            raise ValueError("current book is not valid two-sided")
        return top

    def _current_book_is_valid(self) -> bool:
        state = self._require_state()
        return self._book_top_from_ptr(state.current_book_ptr) is not None

    def _require_state(self) -> _EnvState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state


def _require_config(value: Any) -> ExecutionEnvConfig:
    if not isinstance(value, ExecutionEnvConfig):
        raise ValueError("config must be ExecutionEnvConfig")
    return value


def _require_tape(value: Any) -> ExecutionTape:
    if not isinstance(value, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    return value


def _coerce_action(value: ContinuousQuoteAction | Sequence[float] | np.ndarray) -> ContinuousQuoteAction:
    if isinstance(value, ContinuousQuoteAction):
        return value
    return action_array_to_continuous_action(value)


def _live_orders_tuple(orders: Sequence[ActiveOrder]) -> tuple[ActiveOrder, ...]:
    if isinstance(orders, (str, bytes)):
        raise ValueError("orders must be a sequence of ActiveOrder")
    try:
        seq = tuple(orders)
    except TypeError as exc:
        raise ValueError("orders must be a sequence of ActiveOrder") from exc
    for i, order in enumerate(seq):
        if not isinstance(order, ActiveOrder):
            raise ValueError(f"orders[{i}] must be ActiveOrder")
    return tuple(order for order in seq if order.is_live)


def _next_order_id_after(orders: Sequence[ActiveOrder], fallback: int) -> int:
    next_id = _require_nonnegative_int(fallback, "fallback")
    for order in orders:
        if not isinstance(order, ActiveOrder):
            raise ValueError("orders must contain ActiveOrder values")
        next_id = max(next_id, order.order_id + 1)
    return next_id


def _trade_side_from_code(code: int) -> AggressorSide:
    if code == 1:
        return AggressorSide.BUY
    if code == -1:
        return AggressorSide.SELL
    return AggressorSide.UNKNOWN
