"""Plain Python execution replay environment for an :class:`ExecutionTape`.

The environment owns only simulator state: it replays already-built tape arrays,
turns one continuous action per decision interval into maker-safe quotes,
simulates fills, computes rewards, and builds fixed observation vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Sequence

import numpy as np

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import (
    ActionSpec,
    ActiveOrder,
    BookTop,
    LatencyConfig,
    ExecutionStepResult,
    Fill,
    FillReason,
    OrderSide,
    OrderStatus,
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
    activate_pending_orders,
    finalize_effective_cancels,
    simulate_l2_level_update,
    simulate_trade_event,
    sync_orders_to_quote,
)
from mmrt.execution.contracts import LinearSignal
from mmrt.execution.linear_signal import LinearSignalArtifact, linear_signal_at
from mmrt.execution.adverse_runtime import (
    AdverseRuntimeConfig,
    adverse_predictions_for_row,
    build_adverse_observation_features,
    build_executable_edge_observation_features,
)
from mmrt.execution.adverse_signal import AdverseSelectionSignalArtifact, validate_adverse_signal_alignment
from mmrt.execution.obs_builder import (
    ObservationBuilder,
    ObservationBuilderConfig,
    ObservationContext,
    ObservationInput,
)
from mmrt.execution.obs_schema import DEFAULT_OBSERVATION_FIELDS, ObservationSchema, default_observation_schema, execution_observation_schema
from mmrt.execution.queue_model import QueueModelConfig, QueueModelMode, estimate_initial_queue_ahead
from mmrt.time_key import EventKey, MAX_EVENT_SEQ
from mmrt.execution.quote_geometry import (
    QuoteAction,
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
    fill_simulator_config: FillSimulatorConfig = field(default_factory=FillSimulatorConfig)
    latency_config: LatencyConfig = field(default_factory=LatencyConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    observation_schema: ObservationSchema = field(default_factory=default_observation_schema)
    observation_builder_config: ObservationBuilderConfig = field(default_factory=ObservationBuilderConfig)
    initial_position: PositionState = field(default_factory=PositionState)
    max_episode_steps: int | None = None
    adverse_runtime_config: AdverseRuntimeConfig | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.decision_interval_us, "decision_interval_us")
        if not isinstance(self.action_spec, ActionSpec):
            raise ValueError("action_spec must be ActionSpec")
        if not isinstance(self.quote_geometry_config, QuoteGeometryConfig):
            raise ValueError("quote_geometry_config must be QuoteGeometryConfig")
        if not isinstance(self.fill_simulator_config, FillSimulatorConfig):
            raise ValueError("fill_simulator_config must be FillSimulatorConfig")
        if not isinstance(self.latency_config, LatencyConfig):
            raise ValueError("latency_config must be LatencyConfig")
        if not isinstance(self.reward_config, RewardConfig):
            raise ValueError("reward_config must be RewardConfig")
        if not isinstance(self.observation_schema, ObservationSchema):
            raise ValueError("observation_schema must be ObservationSchema")
        if not isinstance(self.observation_builder_config, ObservationBuilderConfig):
            raise ValueError("observation_builder_config must be ObservationBuilderConfig")
        _require_position(self.initial_position)
        object.__setattr__(self, "max_episode_steps", _optional_positive_int(self.max_episode_steps, "max_episode_steps"))
        if self.adverse_runtime_config is not None and not isinstance(self.adverse_runtime_config, AdverseRuntimeConfig):
            raise ValueError("adverse_runtime_config must be None or AdverseRuntimeConfig")


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


def action_array_to_continuous_action(values: Sequence[float] | np.ndarray) -> QuoteAction:
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be a sequence or ndarray of six finite numeric values")
    try:
        seq = tuple(values)
    except TypeError as exc:
        raise ValueError("values must be a sequence or ndarray of six finite numeric values") from exc
    if len(seq) != 6:
        raise ValueError("values must contain exactly 6 values")
    cleaned = tuple(_require_finite_float(value, f"values[{i}]") for i, value in enumerate(seq))
    return QuoteAction(
        bid_enabled=cleaned[0] >= 0.5,
        ask_enabled=cleaned[1] >= 0.5,
        bid_price_raw=cleaned[2],
        ask_price_raw=cleaned[3],
        bid_size_raw=cleaned[4],
        ask_size_raw=cleaned[5],
    )


class ExecutionEnv:
    def __init__(
        self,
        tape: ExecutionTape,
        *,
        linear_signals: LinearSignalArtifact,
        adverse_signals: AdverseSelectionSignalArtifact | None = None,
        config: ExecutionEnvConfig = ExecutionEnvConfig(),
    ) -> None:
        self.tape = _require_tape(tape)
        explicit_default_schema = tuple(config.observation_schema.field_names) == tuple(DEFAULT_OBSERVATION_FIELDS)
        if adverse_signals is not None and config.adverse_runtime_config is None:
            runtime_config = AdverseRuntimeConfig()
            config = replace(config, adverse_runtime_config=runtime_config)
        if adverse_signals is not None and explicit_default_schema:
            assert config.adverse_runtime_config is not None
            config = replace(
                config,
                observation_schema=execution_observation_schema(
                    dtype=config.observation_schema.dtype,
                    include_adverse_selection=True,
                    include_executable_edge=True,
                    candidate_names=config.adverse_runtime_config.candidate_names,
                ),
            )
        self.config = _require_config(config)
        if not isinstance(linear_signals, LinearSignalArtifact):
            raise ValueError("linear_signals must be LinearSignalArtifact")
        if linear_signals.n_rows <= 0:
            raise ValueError("linear_signals must contain at least one row")
        if config.max_episode_steps is not None and linear_signals.n_rows < config.max_episode_steps + 1:
            raise ValueError("linear_signals must contain at least max_episode_steps + 1 rows")
        self.linear_signals = linear_signals
        if adverse_signals is not None:
            if not isinstance(adverse_signals, AdverseSelectionSignalArtifact):
                raise ValueError("adverse_signals must be AdverseSelectionSignalArtifact")
            validate_adverse_signal_alignment(
                adverse_signals,
                decision_local_ts_us=linear_signals.decision_local_ts_us,
                decision_event_index=linear_signals.decision_event_index,
                decision_event_seq=np.full(linear_signals.n_rows, MAX_EVENT_SEQ, dtype=np.int64),
                right_name="linear_signals",
            )
        self.adverse_signals = adverse_signals
        self.observation_builder = ObservationBuilder(
            schema=config.observation_schema,
            config=config.observation_builder_config,
        )
        self._obs_buffer = np.zeros(config.observation_schema.dim, dtype=config.observation_schema.np_dtype)
        self._state: _EnvState | None = None
        self._episode_start_local_ts_us = 0
        self._last_step_fills: tuple[Fill, ...] = ()
        self._peak_equity: float | None = None
        self._recent_trade_depletion_by_level: dict[tuple[OrderSide, int], float] = {}
        self._step_diag: dict[str, float | int] = {}

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

    def step(self, action: QuoteAction | Sequence[float] | np.ndarray) -> ExecutionEnvStep:
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

        bid_queue_ahead_qty = self._queue_ahead_for_new_order(OrderSide.BUY, quote.bid_price_tick, state.current_book_ptr) if quote.bid_enabled else 0.0
        ask_queue_ahead_qty = self._queue_ahead_for_new_order(OrderSide.SELL, quote.ask_price_tick, state.current_book_ptr) if quote.ask_enabled else 0.0

        self._recent_trade_depletion_by_level = {}
        self._step_diag = {
            "activated_order_count": 0,
            "post_only_reject_count": 0,
            "effective_cancel_count": 0,
            "queue_trade_advance_qty": 0.0,
            "queue_l2_advance_qty": 0.0,
            "queue_advanced_qty": 0.0,
            "queue_fillable_qty": 0.0,
            "trade_at_level_fill_count": 0,
            "trade_through_fill_count": 0,
            "queue_depletion_fill_count": 0,
            "l2_trade_dedupe_qty": 0.0,
            "l2_raw_decrease_qty": 0.0,
            "l2_effective_decrease_qty": 0.0,
        }
        decision_key = self._event_key_at_index(state.event_index)
        decision_ts = decision_key.local_ts_us
        order_effective_key = self._order_activation_key_for_latency(
            decision_key=decision_key,
            target_local_ts_us=decision_ts + self.config.latency_config.order_activation_delay_us,
        )
        cancel_effective_key = self._cancel_effective_key_for_latency(
            decision_key=decision_key,
            target_local_ts_us=decision_ts + self.config.latency_config.cancel_effective_delay_us,
        )
        replacement_orders, cancel_count = sync_orders_to_quote(
            state.live_orders,
            quote,
            next_order_id=state.next_order_id,
            decision_key=decision_key,
            base_order_effective_key=order_effective_key,
            cancel_effective_key=cancel_effective_key,
            bid_queue_ahead_qty=bid_queue_ahead_qty,
            ask_queue_ahead_qty=ask_queue_ahead_qty,
            qty_epsilon=self.config.fill_simulator_config.qty_epsilon,
        )
        state.live_orders = _live_orders_tuple(replacement_orders)
        self._activate_pending_orders_at(event_key=decision_key, book_ptr=state.current_book_ptr)
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
            event_key = self._event_key_at_index(next_event_index)
            cancels = finalize_effective_cancels(state.live_orders, event_key=event_key)
            state.live_orders = _live_orders_tuple(cancels.orders)
            self._step_diag["effective_cancel_count"] = int(self._step_diag["effective_cancel_count"]) + cancels.cancelled_count
            if event_key.event_seq > 0:
                pre_activation_key = EventKey(event_key.local_ts_us, event_key.event_seq - 1)
            else:
                pre_activation_key = EventKey(max(event_key.local_ts_us - 1, 1), MAX_EVENT_SEQ)
            self._activate_pending_orders_at(
                event_key=pre_activation_key,
                book_ptr=state.current_book_ptr,
            )
            fills.extend(self._process_event(next_event_index))
            self._activate_pending_orders_at(
                event_key=event_key,
                book_ptr=state.current_book_ptr,
            )
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
            "cancel_request_count": cancel_count,
            "num_fills": len(step_fills),
            "orders_live_count": sum(1 for order in state.live_orders if order.is_live),
            "orders_pending_new_count": sum(1 for order in state.live_orders if order.status == OrderStatus.PENDING_NEW),
            "orders_pending_cancel_count": sum(1 for order in state.live_orders if order.status == OrderStatus.PENDING_CANCEL),
            "orders_active_count": sum(1 for order in state.live_orders if order.status == OrderStatus.ACTIVE),
            "orders_partially_filled_count": sum(1 for order in state.live_orders if order.status == OrderStatus.PARTIALLY_FILLED),
            "activated_order_count": int(self._step_diag["activated_order_count"]),
            "post_only_reject_count": int(self._step_diag["post_only_reject_count"]),
            "effective_cancel_count": int(self._step_diag["effective_cancel_count"]),
            "queue_trade_advance_qty": float(self._step_diag["queue_trade_advance_qty"]),
            "queue_l2_advance_qty": float(self._step_diag["queue_l2_advance_qty"]),
            "queue_advanced_qty": float(self._step_diag["queue_advanced_qty"]),
            "queue_fillable_qty": float(self._step_diag["queue_fillable_qty"]),
            "trade_at_level_fill_count": int(self._step_diag["trade_at_level_fill_count"]),
            "trade_through_fill_count": int(self._step_diag["trade_through_fill_count"]),
            "queue_depletion_fill_count": int(self._step_diag["queue_depletion_fill_count"]),
            "l2_trade_dedupe_qty": float(self._step_diag["l2_trade_dedupe_qty"]),
            "l2_raw_decrease_qty": float(self._step_diag["l2_raw_decrease_qty"]),
            "l2_effective_decrease_qty": float(self._step_diag["l2_effective_decrease_qty"]),
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
            "reward_scale": self.config.reward_config.reward_scale,
            "decision_compute_latency_us": self.config.latency_config.decision_compute_latency_us,
            "order_entry_latency_us": self.config.latency_config.order_entry_latency_us,
            "cancel_latency_us": self.config.latency_config.cancel_latency_us,
        }
        if self.adverse_signals is not None:
            row = state.step_index - 1
            runtime_config = self.config.adverse_runtime_config or AdverseRuntimeConfig()
            predictions = adverse_predictions_for_row(self.adverse_signals, row)
            edge_features = build_executable_edge_observation_features(
                predictions=predictions,
                candidate_names=runtime_config.candidate_names,
                best_bid_tick=previous_book_top.best_bid_tick,
                best_ask_tick=previous_book_top.best_ask_tick,
                linear_signal=linear_signal_at(self.linear_signals.arrays, row),
                inventory_qty=previous_position.inventory_qty,
                config=runtime_config,
            )
            for name, value in edge_features.items():
                if name.endswith("_attempt_bps") or name.endswith("_valid"):
                    info[name] = value
            info["adverse_signal_available"] = True
            info["adverse_signal_row"] = row
        else:
            info["adverse_signal_available"] = False

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
            reward=reward_step.reward.total_reward * self.config.reward_config.reward_scale,
            execution=execution,
        )

    def _event_key_at_index(self, event_index: int) -> EventKey:
        row = self.tape.arrays.events[event_index]
        return EventKey(int(row["local_ts_us"]), int(row["event_seq"]))

    def _order_activation_key_for_latency(
        self,
        *,
        decision_key: EventKey,
        target_local_ts_us: int,
    ) -> EventKey:
        target_local_ts_us = _require_positive_int(target_local_ts_us, "target_local_ts_us")
        if target_local_ts_us <= decision_key.local_ts_us:
            return decision_key
        return EventKey(target_local_ts_us, MAX_EVENT_SEQ)

    def _cancel_effective_key_for_latency(
        self,
        *,
        decision_key: EventKey,
        target_local_ts_us: int,
    ) -> EventKey:
        target_local_ts_us = _require_positive_int(target_local_ts_us, "target_local_ts_us")
        if target_local_ts_us <= decision_key.local_ts_us:
            return decision_key
        return EventKey(target_local_ts_us, 0)

    def _refresh_pending_new_queue_ahead_for_activation(
        self,
        orders: tuple[ActiveOrder, ...],
        *,
        event_key: EventKey,
        book_ptr: int,
    ) -> tuple[ActiveOrder, ...]:
        refreshed: list[ActiveOrder] = []
        for order in orders:
            if order.status != OrderStatus.PENDING_NEW:
                refreshed.append(order)
                continue
            if order.effective_key > event_key:
                refreshed.append(order)
                continue
            cancel_key = order.cancel_effective_key
            if cancel_key is not None and cancel_key <= event_key:
                refreshed.append(order)
                continue

            queue_ahead_qty = self._queue_ahead_for_new_order(
                order.side,
                order.price_tick,
                book_ptr,
            )
            if abs(queue_ahead_qty - order.queue_ahead_qty) <= self.config.fill_simulator_config.qty_epsilon:
                refreshed.append(order)
                continue

            refreshed.append(replace(order, queue_ahead_qty=queue_ahead_qty))
        return tuple(refreshed)

    def _activate_pending_orders_at(
        self,
        *,
        event_key: EventKey,
        book_ptr: int,
    ) -> None:
        state = self._require_state()
        refreshed = self._refresh_pending_new_queue_ahead_for_activation(
            state.live_orders,
            event_key=event_key,
            book_ptr=book_ptr,
        )
        activation = activate_pending_orders(
            refreshed,
            event_key=event_key,
            book_top=self._book_top_from_ptr(book_ptr),
            post_only_gap_ticks=self.config.quote_geometry_config.post_only_gap_ticks,
        )
        state.live_orders = _live_orders_tuple(activation.orders)
        self._step_diag["activated_order_count"] = int(self._step_diag["activated_order_count"]) + activation.activated_count
        self._step_diag["post_only_reject_count"] = int(self._step_diag["post_only_reject_count"]) + activation.post_only_reject_count
        self._step_diag["effective_cancel_count"] = int(self._step_diag["effective_cancel_count"]) + activation.already_cancelled_count

    def _record_queue_updates(self, result) -> None:
        for update in result.queue_updates:
            self._step_diag["queue_trade_advance_qty"] = float(self._step_diag["queue_trade_advance_qty"]) + update.trade_advance_qty
            self._step_diag["queue_l2_advance_qty"] = float(self._step_diag["queue_l2_advance_qty"]) + update.l2_advance_qty
            self._step_diag["queue_advanced_qty"] = float(self._step_diag["queue_advanced_qty"]) + update.advanced_qty
            self._step_diag["queue_fillable_qty"] = float(self._step_diag["queue_fillable_qty"]) + update.fillable_qty
        for fill in result.fills:
            if fill.reason == FillReason.TRADE_AT_LEVEL:
                self._step_diag["trade_at_level_fill_count"] = int(self._step_diag["trade_at_level_fill_count"]) + 1
            elif fill.reason == FillReason.TRADE_THROUGH:
                self._step_diag["trade_through_fill_count"] = int(self._step_diag["trade_through_fill_count"]) + 1
            elif fill.reason == FillReason.QUEUE_DEPLETION:
                self._step_diag["queue_depletion_fill_count"] = int(self._step_diag["queue_depletion_fill_count"]) + 1

    def _trade_l2_dedupe_qty_from_queue_updates(self, result) -> float:
        """Return same-level trade quantity already applied to our simulated queue.

        Only TRADE_AT_LEVEL queue updates should feed the L2 de-dup ledger.
        Trade-through fills terminally consume the order and should not create
        same-level L2 de-dup for future/pending orders.
        """
        total = 0.0
        for update in result.queue_updates:
            if update.trade_at_level:
                total += update.trade_advance_qty + update.fillable_qty
        return total

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
            linear_signal=self._linear_signal_for_step(
                state.step_index,
                expected_event_index=state.event_index,
                expected_local_ts_us=book_top.local_ts_us,
            ),
            adverse_features=self._adverse_observation_features_for_step(state.step_index, book_top),
            executable_edge_features=self._edge_observation_features_for_step(state.step_index, book_top),
            context=context,
        )
        return self.observation_builder.build(inputs, out=self._obs_buffer)


    def _adverse_predictions_for_step(self, step_index: int) -> dict[str, float] | None:
        if self.adverse_signals is None:
            return None
        return adverse_predictions_for_row(self.adverse_signals, step_index)

    def _adverse_observation_features_for_step(self, step_index: int, book_top: BookTop) -> dict[str, float]:
        predictions = self._adverse_predictions_for_step(step_index)
        if predictions is None:
            return {}
        runtime_config = self.config.adverse_runtime_config or AdverseRuntimeConfig()
        return build_adverse_observation_features(
            predictions=predictions,
            candidate_names=runtime_config.candidate_names,
        )

    def _edge_observation_features_for_step(self, step_index: int, book_top: BookTop) -> dict[str, float]:
        predictions = self._adverse_predictions_for_step(step_index)
        if predictions is None:
            return {}
        runtime_config = self.config.adverse_runtime_config or AdverseRuntimeConfig()
        linear_signal = self._linear_signal_for_step(
            step_index,
            expected_event_index=self._require_state().event_index,
            expected_local_ts_us=book_top.local_ts_us,
        )
        return build_executable_edge_observation_features(
            predictions=predictions,
            candidate_names=runtime_config.candidate_names,
            best_bid_tick=book_top.best_bid_tick,
            best_ask_tick=book_top.best_ask_tick,
            linear_signal=linear_signal,
            inventory_qty=self._require_state().position.inventory_qty,
            config=runtime_config,
        )

    def _linear_signal_for_step(
        self,
        step_index: int,
        *,
        expected_event_index: int,
        expected_local_ts_us: int,
    ) -> LinearSignal:
        if step_index >= self.linear_signals.n_rows:
            raise ValueError("linear_signals does not contain a row for current step_index")

        actual_event_index = int(self.linear_signals.decision_event_index[step_index])
        if actual_event_index != expected_event_index:
            raise ValueError(
                "linear signal decision_event_index mismatch: "
                f"row={step_index} expected={expected_event_index} actual={actual_event_index}"
            )

        actual_local_ts_us = int(self.linear_signals.decision_local_ts_us[step_index])
        if actual_local_ts_us != expected_local_ts_us:
            raise ValueError(
                "linear signal decision_local_ts_us mismatch: "
                f"row={step_index} expected={expected_local_ts_us} actual={actual_local_ts_us}"
            )

        return linear_signal_at(self.linear_signals.arrays, step_index)

    def _process_event(self, event_index: int) -> tuple[Fill, ...]:
        state = self._require_state()
        event = self.tape.arrays.events[event_index]
        event_key = self._event_key_at_index(event_index)
        code = int(event["event_type_code"])
        old_event_local = int(self.tape.arrays.events[state.event_index]["local_ts_us"])
        fills: tuple[Fill, ...] = ()
        if code == EVENT_TYPE_CODE_L2_BATCH:
            book_ptr = int(event["book_ptr"])
            fills = self._process_l2_event(book_ptr, event_key=event_key)
            if self._book_top_from_ptr(book_ptr) is not None:
                state.current_book_ptr = book_ptr
        elif code == EVENT_TYPE_CODE_TRADE:
            trade = self._trade_from_ptr(int(event["trade_ptr"]))
            result = simulate_trade_event(
                state.live_orders,
                trade,
                event_key=event_key,
                symbol_spec=self.tape.manifest.symbol_spec,
                config=self.config.fill_simulator_config,
            )
            self._record_queue_updates(result)
            state.live_orders = _live_orders_tuple(result.orders)
            fills = result.fills
            queue_config = self.config.fill_simulator_config.queue_model
            dedupe_qty = self._trade_l2_dedupe_qty_from_queue_updates(result)

            level_side = None
            if trade.side == AggressorSide.SELL:
                level_side = OrderSide.BUY
            elif trade.side == AggressorSide.BUY:
                level_side = OrderSide.SELL

            if (
                level_side is not None
                and dedupe_qty > self.config.fill_simulator_config.qty_epsilon
                and queue_config.mode == QueueModelMode.BALANCED
                and queue_config.dedupe_l2_decrease_with_trade_prints
            ):
                key = (level_side, trade.price_tick)
                self._recent_trade_depletion_by_level[key] = self._recent_trade_depletion_by_level.get(key, 0.0) + dedupe_qty
        state.previous_event_local_ts_us = old_event_local
        state.event_index = event_index
        return fills

    def _process_l2_event(self, curr_book_ptr: int, *, event_key: EventKey) -> tuple[Fill, ...]:
        state = self._require_state()
        if curr_book_ptr < 0:
            return ()
        updated_orders = state.live_orders
        fills: list[Fill] = []
        processed_levels: set[tuple[OrderSide, int]] = set()
        for order in tuple(updated_orders):
            if not order.is_live:
                continue
            level_key = (order.side, order.price_tick)
            if level_key in processed_levels:
                continue
            processed_levels.add(level_key)
            if not self._has_fillable_order_at_level(
                tuple(updated_orders),
                side=order.side,
                price_tick=order.price_tick,
                event_key=event_key,
            ):
                continue
            prev_qty, prev_known = self._level_qty_with_depth_status(book_ptr=state.current_book_ptr, side=order.side, price_tick=order.price_tick)
            curr_qty, curr_known = self._level_qty_with_depth_status(book_ptr=curr_book_ptr, side=order.side, price_tick=order.price_tick)
            if not (prev_known and curr_known):
                continue
            raw_decrease = max((prev_qty if prev_qty is not None else 0.0) - (curr_qty if curr_qty is not None else 0.0), 0.0)
            self._step_diag["l2_raw_decrease_qty"] = float(self._step_diag["l2_raw_decrease_qty"]) + raw_decrease
            if self.config.fill_simulator_config.queue_model.dedupe_l2_decrease_with_trade_prints:
                already = self._recent_trade_depletion_by_level.get(level_key, 0.0)
                deduped = min(raw_decrease, already)
                l2_decrease_qty = raw_decrease - deduped
                self._recent_trade_depletion_by_level[level_key] = max(already - deduped, 0.0)
                self._step_diag["l2_trade_dedupe_qty"] = float(self._step_diag["l2_trade_dedupe_qty"]) + deduped
            else:
                l2_decrease_qty = raw_decrease
            self._step_diag["l2_effective_decrease_qty"] = float(self._step_diag["l2_effective_decrease_qty"]) + l2_decrease_qty
            result = simulate_l2_level_update(
                updated_orders,
                side=order.side,
                price_tick=order.price_tick,
                l2_decrease_qty=l2_decrease_qty,
                event_key=event_key,
                symbol_spec=self.tape.manifest.symbol_spec,
                config=self.config.fill_simulator_config,
            )
            self._record_queue_updates(result)
            updated_orders = _live_orders_tuple(result.orders)
            fills.extend(result.fills)
        state.live_orders = _live_orders_tuple(updated_orders)
        return tuple(fills)

    def _has_fillable_order_at_level(
        self,
        orders: tuple[ActiveOrder, ...],
        *,
        side: OrderSide,
        price_tick: int,
        event_key: EventKey,
    ) -> bool:
        return any(
            order.side == side
            and order.price_tick == price_tick
            and order.is_fillable_at_key(event_key)
            for order in orders
        )

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

    def _queue_ahead_for_new_order(self, side: OrderSide, price_tick: int, book_ptr: int) -> float:
        top = self._book_top_from_ptr(book_ptr)
        if top is not None:
            if side == OrderSide.BUY and top.best_bid_tick < price_tick < top.best_ask_tick:
                return 0.0
            if side == OrderSide.SELL and top.best_bid_tick < price_tick < top.best_ask_tick:
                return 0.0
        level = self._level_qty(book_ptr=book_ptr, side=side, price_tick=price_tick)
        if level is not None:
            return estimate_initial_queue_ahead(level, config=self.config.fill_simulator_config.queue_model)
        if self._price_within_known_depth(book_ptr=book_ptr, side=side, price_tick=price_tick):
            return 0.0
        return self.config.fill_simulator_config.queue_model.unknown_level_queue_ahead_qty

    def _price_within_known_depth(self, *, book_ptr: int, side: OrderSide, price_tick: int) -> bool:
        ticks = self.tape.arrays.book_bid_ticks[book_ptr] if side == OrderSide.BUY else self.tape.arrays.book_ask_ticks[book_ptr]
        active = [int(t) for t in ticks if int(t) > 0]
        if not active:
            return False
        return min(active) <= price_tick <= max(active)

    def _level_qty_with_depth_status(self, *, book_ptr: int, side: OrderSide, price_tick: int) -> tuple[float | None, bool]:
        level = self._level_qty(book_ptr=book_ptr, side=side, price_tick=price_tick)
        if level is not None:
            return level, True
        if self._price_within_known_depth(book_ptr=book_ptr, side=side, price_tick=price_tick):
            return 0.0, True
        return None, False

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


def _coerce_action(value: QuoteAction | Sequence[float] | np.ndarray) -> QuoteAction:
    if isinstance(value, QuoteAction):
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
