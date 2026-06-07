"""Streaming metrics for execution simulation step results."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from mmrt.execution.contracts import ExecutionStepResult, FillReason, OrderSide

__all__ = [
    "ExecutionMetricAccumulator",
    "summarize_execution_steps",
]


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _info_int(info: Mapping[str, object], key: str, default: int = 0) -> int:
    value = info.get(key, default)
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _info_float(info: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = info.get(key, default)
    out = _finite_float_or_none(value)
    return default if out is None else out


def _info_bool(info: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = info.get(key, default)
    return value if isinstance(value, bool) else default


@dataclass(slots=True)
class _RunningStat:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.variance, 0.0))

    def as_dict(self, prefix: str | None = None) -> dict[str, object]:
        values: dict[str, object] = {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value,
        }
        if prefix is None:
            return values
        return {f"{prefix}_{key}": value for key, value in values.items()}


@dataclass(slots=True)
class ExecutionMetricAccumulator:
    step_count: int = 0
    done_count: int = 0
    truncated_count: int = 0
    terminal_count: int = 0
    events_processed_total: int = 0

    fill_count: int = 0
    fill_step_count: int = 0
    buy_fill_count: int = 0
    sell_fill_count: int = 0
    buy_fill_qty: float = 0.0
    sell_fill_qty: float = 0.0
    net_fill_qty: float = 0.0
    fill_fee_total: float = 0.0
    fill_reason_counts: dict[str, int] = field(
        default_factory=lambda: {
            FillReason.TRADE_THROUGH.value: 0,
            FillReason.TRADE_AT_LEVEL.value: 0,
            FillReason.QUEUE_DEPLETION.value: 0,
        }
    )

    quote_bid_enabled_count: int = 0
    quote_ask_enabled_count: int = 0
    two_sided_quote_count: int = 0
    all_quotes_disabled_count: int = 0

    cancel_count_total: int = 0
    post_only_reject_count_total: int = 0
    activated_order_count_total: int = 0
    effective_cancel_count_total: int = 0
    orders_pending_cancel_count_total: int = 0
    orders_pending_cancel_count_max: int = 0
    queue_trade_advance_qty_total: float = 0.0
    queue_l2_advance_qty_total: float = 0.0
    queue_advanced_qty_total: float = 0.0
    queue_fillable_qty_total: float = 0.0
    trade_at_level_fill_count: int = 0
    trade_through_fill_count: int = 0
    queue_depletion_fill_count: int = 0
    l2_trade_dedupe_qty_total: float = 0.0
    l2_raw_decrease_qty_total: float = 0.0
    l2_effective_decrease_qty_total: float = 0.0
    turnover_notional_total: float = 0.0

    raw_equity_delta_total: float = 0.0
    inventory_penalty_total: float = 0.0
    drawdown_penalty_total: float = 0.0
    turnover_penalty_total: float = 0.0
    cancel_penalty_total: float = 0.0
    terminal_penalty_total: float = 0.0
    total_reward_raw: float = 0.0

    final_cash: float = 0.0
    final_inventory_qty: float = 0.0
    final_fees_paid: float = 0.0
    max_abs_inventory_qty: float = 0.0
    equity_initial: float | None = None
    equity_final: float | None = None
    equity_min: float | None = None
    equity_max: float | None = None
    max_drawdown: float = 0.0

    _reward_stat: _RunningStat = field(default_factory=_RunningStat)
    _equity_stat: _RunningStat = field(default_factory=_RunningStat)
    _inventory_abs_stat: _RunningStat = field(default_factory=_RunningStat)
    _events_per_step_stat: _RunningStat = field(default_factory=_RunningStat)
    _peak_equity: float | None = None

    def update(self, step: ExecutionStepResult) -> None:
        if not isinstance(step, ExecutionStepResult):
            raise ValueError("step must be ExecutionStepResult")

        info = step.info or {}
        self.step_count += 1
        self.done_count += int(step.done)
        self.truncated_count += int(step.truncated)
        self.terminal_count += int(step.done or step.truncated)

        events_processed = _info_int(info, "events_processed", 0)
        self.events_processed_total += events_processed
        self._events_per_step_stat.update(float(events_processed))

        if step.fills:
            self.fill_step_count += 1
        for fill in step.fills:
            self.fill_count += 1
            self.fill_fee_total += fill.fee
            self.fill_reason_counts[fill.reason.value] = self.fill_reason_counts.get(fill.reason.value, 0) + 1
            if fill.side == OrderSide.BUY:
                self.buy_fill_count += 1
                self.buy_fill_qty += fill.qty
                self.net_fill_qty += fill.qty
            elif fill.side == OrderSide.SELL:
                self.sell_fill_count += 1
                self.sell_fill_qty += fill.qty
                self.net_fill_qty -= fill.qty

        bid_enabled = _info_bool(info, "quote_bid_enabled", False)
        ask_enabled = _info_bool(info, "quote_ask_enabled", False)
        self.quote_bid_enabled_count += int(bid_enabled)
        self.quote_ask_enabled_count += int(ask_enabled)
        self.two_sided_quote_count += int(bid_enabled and ask_enabled)
        self.all_quotes_disabled_count += int(not bid_enabled and not ask_enabled)

        self.cancel_count_total += _info_int(info, "cancel_count", 0)
        self.post_only_reject_count_total += _info_int(info, "post_only_reject_count", 0)
        self.activated_order_count_total += _info_int(info, "activated_order_count", 0)
        self.effective_cancel_count_total += _info_int(info, "effective_cancel_count", 0)
        orders_pending_cancel_count = _info_int(info, "orders_pending_cancel_count", 0)
        self.orders_pending_cancel_count_total += orders_pending_cancel_count
        self.orders_pending_cancel_count_max = max(
            self.orders_pending_cancel_count_max,
            orders_pending_cancel_count,
        )
        self.queue_trade_advance_qty_total += _info_float(info, "queue_trade_advance_qty", 0.0)
        self.queue_l2_advance_qty_total += _info_float(info, "queue_l2_advance_qty", 0.0)
        self.queue_advanced_qty_total += _info_float(info, "queue_advanced_qty", 0.0)
        self.queue_fillable_qty_total += _info_float(info, "queue_fillable_qty", 0.0)
        self.trade_at_level_fill_count += _info_int(info, "trade_at_level_fill_count", 0)
        self.trade_through_fill_count += _info_int(info, "trade_through_fill_count", 0)
        self.queue_depletion_fill_count += _info_int(info, "queue_depletion_fill_count", 0)
        self.l2_trade_dedupe_qty_total += _info_float(info, "l2_trade_dedupe_qty", 0.0)
        self.l2_raw_decrease_qty_total += _info_float(info, "l2_raw_decrease_qty", 0.0)
        self.l2_effective_decrease_qty_total += _info_float(info, "l2_effective_decrease_qty", 0.0)
        self.turnover_notional_total += _info_float(info, "turnover_notional", 0.0)

        reward = step.reward
        self.raw_equity_delta_total += reward.raw_equity_delta
        self.inventory_penalty_total += reward.inventory_penalty
        self.drawdown_penalty_total += reward.drawdown_penalty
        self.turnover_penalty_total += reward.turnover_penalty
        self.cancel_penalty_total += reward.cancel_penalty
        self.terminal_penalty_total += reward.terminal_penalty
        total_reward = reward.total_reward
        self.total_reward_raw += total_reward
        self._reward_stat.update(total_reward)

        self.final_cash = step.position.cash
        self.final_inventory_qty = step.position.inventory_qty
        self.final_fees_paid = step.position.fees_paid
        abs_inventory = abs(step.position.inventory_qty)
        self.max_abs_inventory_qty = max(self.max_abs_inventory_qty, abs_inventory)
        self._inventory_abs_stat.update(abs_inventory)

        previous_equity = _finite_float_or_none(info.get("previous_equity"))
        current_equity = _finite_float_or_none(info.get("current_equity"))
        if self.equity_initial is None and previous_equity is not None:
            self.equity_initial = previous_equity
        if current_equity is not None:
            self.equity_final = current_equity
            self._equity_stat.update(current_equity)
            self.equity_min = current_equity if self.equity_min is None else min(self.equity_min, current_equity)
            self.equity_max = current_equity if self.equity_max is None else max(self.equity_max, current_equity)
            self._peak_equity = current_equity if self._peak_equity is None else max(self._peak_equity, current_equity)
            self.max_drawdown = max(self.max_drawdown, (self._peak_equity or current_equity) - current_equity)
        if previous_equity is not None:
            self.equity_min = previous_equity if self.equity_min is None else min(self.equity_min, previous_equity)
            self.equity_max = previous_equity if self.equity_max is None else max(self.equity_max, previous_equity)
            self._peak_equity = previous_equity if self._peak_equity is None else max(self._peak_equity, previous_equity)

    def as_dict(self) -> dict[str, object]:
        denominator = max(self.step_count, 1)
        return {
            "steps": {
                "count": self.step_count,
                "done_count": self.done_count,
                "truncated_count": self.truncated_count,
                "terminal_count": self.terminal_count,
                "events_processed_total": self.events_processed_total,
                "events_processed_mean": self._events_per_step_stat.mean,
            },
            "rewards": {
                "total_raw": self.total_reward_raw,
                "mean": self._reward_stat.mean,
                "std": self._reward_stat.std,
                "min": self._reward_stat.min_value,
                "max": self._reward_stat.max_value,
                "components_total": {
                    "raw_equity_delta": self.raw_equity_delta_total,
                    "inventory_penalty": self.inventory_penalty_total,
                    "drawdown_penalty": self.drawdown_penalty_total,
                    "turnover_penalty": self.turnover_penalty_total,
                    "cancel_penalty": self.cancel_penalty_total,
                    "terminal_penalty": self.terminal_penalty_total,
                },
            },
            "equity": {
                "initial": self.equity_initial,
                "final": self.equity_final,
                "min": self.equity_min,
                "max": self.equity_max,
                "max_drawdown": self.max_drawdown,
            },
            "position": {
                "final_cash": self.final_cash,
                "final_inventory_qty": self.final_inventory_qty,
                "final_fees_paid": self.final_fees_paid,
                "max_abs_inventory_qty": self.max_abs_inventory_qty,
                "mean_abs_inventory_qty": self._inventory_abs_stat.mean,
            },
            "fills": {
                "count": self.fill_count,
                "fill_step_count": self.fill_step_count,
                "fill_rate": self.fill_step_count / denominator,
                "buy_count": self.buy_fill_count,
                "sell_count": self.sell_fill_count,
                "buy_qty": self.buy_fill_qty,
                "sell_qty": self.sell_fill_qty,
                "net_qty": self.net_fill_qty,
                "fee_total": self.fill_fee_total,
                "reason_counts": dict(self.fill_reason_counts),
            },
            "orders": {
                "cancel_count_total": self.cancel_count_total,
                "cancel_rate_per_step": self.cancel_count_total / denominator,
                "post_only_reject_count_total": self.post_only_reject_count_total,
                "activated_order_count_total": self.activated_order_count_total,
                "effective_cancel_count_total": self.effective_cancel_count_total,
                "orders_pending_cancel_count_total": self.orders_pending_cancel_count_total,
                "orders_pending_cancel_count_mean": self.orders_pending_cancel_count_total / denominator,
                "orders_pending_cancel_count_max": self.orders_pending_cancel_count_max,
                "quote_bid_enabled_count": self.quote_bid_enabled_count,
                "quote_ask_enabled_count": self.quote_ask_enabled_count,
                "two_sided_quote_count": self.two_sided_quote_count,
                "all_quotes_disabled_count": self.all_quotes_disabled_count,
            },
            "queue": {
                "trade_advance_qty_total": self.queue_trade_advance_qty_total,
                "l2_advance_qty_total": self.queue_l2_advance_qty_total,
                "advanced_qty_total": self.queue_advanced_qty_total,
                "fillable_qty_total": self.queue_fillable_qty_total,
                "trade_at_level_fill_count": self.trade_at_level_fill_count,
                "trade_through_fill_count": self.trade_through_fill_count,
                "queue_depletion_fill_count": self.queue_depletion_fill_count,
                "l2_trade_dedupe_qty_total": self.l2_trade_dedupe_qty_total,
                "l2_raw_decrease_qty_total": self.l2_raw_decrease_qty_total,
                "l2_effective_decrease_qty_total": self.l2_effective_decrease_qty_total,
            },
            "turnover": {
                "notional_total": self.turnover_notional_total,
                "notional_mean_per_step": self.turnover_notional_total / denominator,
            },
        }


def summarize_execution_steps(steps: Iterable[ExecutionStepResult]) -> dict[str, object]:
    acc = ExecutionMetricAccumulator()
    for step in steps:
        acc.update(step)
    return acc.as_dict()
