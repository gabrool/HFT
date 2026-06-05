"""Pure reward and position accounting helpers for execution environments.

This module consumes fills produced elsewhere, updates immutable position state,
and computes mark-to-market reward components. It intentionally avoids owning
quote construction, fill simulation, order lifecycle, observations, or RL logic.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from mmrt.execution.contracts import (
    BookTop,
    Fill,
    OrderSide,
    PositionState,
    RewardComponents,
    SymbolSpec,
)


__all__ = [
    "RewardConfig",
    "RewardStepResult",
    "mark_price_from_book_top",
    "fill_notional",
    "fills_turnover_notional",
    "apply_fill_to_position",
    "apply_fills_to_position",
    "compute_reward_components",
    "compute_reward_step",
]


def _require_config(value: Any) -> RewardConfig:
    if not isinstance(value, RewardConfig):
        raise ValueError("config must be RewardConfig")
    return value


def _require_symbol_spec(value: Any) -> SymbolSpec:
    if not isinstance(value, SymbolSpec):
        raise ValueError("symbol_spec must be SymbolSpec")
    return value


def _require_position(value: Any) -> PositionState:
    if not isinstance(value, PositionState):
        raise ValueError("position must be PositionState")
    return value


def _require_book_top(value: Any) -> BookTop:
    if not isinstance(value, BookTop):
        raise ValueError("book_top must be BookTop")
    return value


def _require_fill(value: Any) -> Fill:
    if not isinstance(value, Fill):
        raise ValueError("fill must be Fill")
    return value


def _fills_tuple(values: Sequence[Fill]) -> tuple[Fill, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("fills must be a sequence of Fill values")
    try:
        fills = values if isinstance(values, tuple) else tuple(values)
    except TypeError as exc:
        raise ValueError("fills must be a sequence of Fill values") from exc
    for fill in fills:
        _require_fill(fill)
    return fills


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a nonnegative int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    value = float(value)
    if value != value or value in (float("inf"), -float("inf")):
        raise ValueError(f"{name} must be a finite float")
    return value


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _bps_to_fraction(value: float) -> float:
    return _require_nonnegative_float(value, "bps") / 10_000.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    inventory_penalty_bps: float = 0.0
    turnover_penalty_bps: float = 0.0
    cancel_penalty: float = 0.0
    drawdown_penalty_rate: float = 0.0
    terminal_inventory_penalty_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "inventory_penalty_bps",
            "turnover_penalty_bps",
            "cancel_penalty",
            "drawdown_penalty_rate",
            "terminal_inventory_penalty_bps",
        ):
            object.__setattr__(self, name, _require_nonnegative_float(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class RewardStepResult:
    position: PositionState
    reward: RewardComponents
    previous_equity: float
    current_equity: float
    peak_equity: float
    turnover_notional: float

    def __post_init__(self) -> None:
        _require_position(self.position)
        if not isinstance(self.reward, RewardComponents):
            raise ValueError("reward must be RewardComponents")
        object.__setattr__(self, "previous_equity", _require_finite_float(self.previous_equity, "previous_equity"))
        object.__setattr__(self, "current_equity", _require_finite_float(self.current_equity, "current_equity"))
        object.__setattr__(self, "peak_equity", _require_finite_float(self.peak_equity, "peak_equity"))
        object.__setattr__(
            self, "turnover_notional", _require_nonnegative_float(self.turnover_notional, "turnover_notional")
        )


def mark_price_from_book_top(book_top: BookTop, symbol_spec: SymbolSpec) -> float:
    book_top = _require_book_top(book_top)
    symbol_spec = _require_symbol_spec(symbol_spec)
    mark_tick = book_top.mid_tick_x2 * 0.5
    mark_price = mark_tick * symbol_spec.tick_size
    return _require_positive_float(mark_price, "mark_price")


def fill_notional(fill: Fill, symbol_spec: SymbolSpec) -> float:
    fill = _require_fill(fill)
    symbol_spec = _require_symbol_spec(symbol_spec)
    price = symbol_spec.tick_to_price(fill.price_tick)
    notional = price * fill.qty * symbol_spec.contract_size
    return _require_nonnegative_float(notional, "fill_notional")


def fills_turnover_notional(
    fills: Sequence[Fill],
    *,
    symbol_spec: SymbolSpec,
) -> float:
    fills_tuple = _fills_tuple(fills)
    symbol_spec = _require_symbol_spec(symbol_spec)
    total = 0.0
    for fill in fills_tuple:
        total += fill_notional(fill, symbol_spec)
    return _require_nonnegative_float(total, "turnover_notional")


def apply_fill_to_position(
    position: PositionState,
    fill: Fill,
    *,
    symbol_spec: SymbolSpec,
) -> PositionState:
    position = _require_position(position)
    fill = _require_fill(fill)
    symbol_spec = _require_symbol_spec(symbol_spec)
    notional = fill_notional(fill, symbol_spec)

    if fill.side == OrderSide.BUY:
        new_inventory = position.inventory_qty + fill.qty
        new_cash = position.cash - notional - fill.fee
    elif fill.side == OrderSide.SELL:
        new_inventory = position.inventory_qty - fill.qty
        new_cash = position.cash + notional - fill.fee
    else:
        raise ValueError("unsupported fill side")

    return PositionState(
        cash=new_cash,
        inventory_qty=new_inventory,
        realized_pnl=position.realized_pnl,
        fees_paid=position.fees_paid + fill.fee,
    )


def apply_fills_to_position(
    position: PositionState,
    fills: Sequence[Fill],
    *,
    symbol_spec: SymbolSpec,
) -> PositionState:
    position = _require_position(position)
    fills_tuple = _fills_tuple(fills)
    symbol_spec = _require_symbol_spec(symbol_spec)

    cash = position.cash
    inventory_qty = position.inventory_qty
    fees_paid = position.fees_paid

    for fill in fills_tuple:
        notional = fill_notional(fill, symbol_spec)
        if fill.side == OrderSide.BUY:
            inventory_qty += fill.qty
            cash -= notional
        elif fill.side == OrderSide.SELL:
            inventory_qty -= fill.qty
            cash += notional
        else:
            raise ValueError("unsupported fill side")
        cash -= fill.fee
        fees_paid += fill.fee

    return PositionState(
        cash=cash,
        inventory_qty=inventory_qty,
        realized_pnl=position.realized_pnl,
        fees_paid=fees_paid,
    )


def compute_reward_components(
    *,
    previous_equity: float,
    current_equity: float,
    current_position: PositionState,
    current_mark_price: float,
    symbol_spec: SymbolSpec,
    turnover_notional: float = 0.0,
    cancel_count: int = 0,
    peak_equity: float | None = None,
    terminal: bool = False,
    config: RewardConfig = RewardConfig(),
) -> RewardComponents:
    previous_equity = _require_finite_float(previous_equity, "previous_equity")
    current_equity = _require_finite_float(current_equity, "current_equity")
    current_position = _require_position(current_position)
    current_mark_price = _require_positive_float(current_mark_price, "current_mark_price")
    symbol_spec = _require_symbol_spec(symbol_spec)
    turnover_notional = _require_nonnegative_float(turnover_notional, "turnover_notional")
    cancel_count = _require_nonnegative_int(cancel_count, "cancel_count")
    if peak_equity is not None:
        peak_equity = _require_finite_float(peak_equity, "peak_equity")
    terminal = _require_bool(terminal, "terminal")
    config = _require_config(config)

    raw_equity_delta = current_equity - previous_equity
    inventory_notional = abs(current_position.inventory_qty) * current_mark_price * symbol_spec.contract_size
    inventory_penalty = inventory_notional * _bps_to_fraction(config.inventory_penalty_bps)
    turnover_penalty = turnover_notional * _bps_to_fraction(config.turnover_penalty_bps)
    cancel_penalty = cancel_count * config.cancel_penalty

    if peak_equity is None:
        drawdown_penalty = 0.0
    else:
        effective_peak = max(peak_equity, previous_equity)
        previous_drawdown = max(effective_peak - previous_equity, 0.0)
        current_drawdown = max(effective_peak - current_equity, 0.0)
        incremental_drawdown = max(current_drawdown - previous_drawdown, 0.0)
        drawdown_penalty = incremental_drawdown * config.drawdown_penalty_rate

    if terminal:
        terminal_penalty = inventory_notional * _bps_to_fraction(config.terminal_inventory_penalty_bps)
    else:
        terminal_penalty = 0.0

    return RewardComponents(
        raw_equity_delta=raw_equity_delta,
        inventory_penalty=inventory_penalty,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
        cancel_penalty=cancel_penalty,
        terminal_penalty=terminal_penalty,
    )


def compute_reward_step(
    *,
    previous_position: PositionState,
    fills: Sequence[Fill],
    previous_book_top: BookTop,
    current_book_top: BookTop,
    symbol_spec: SymbolSpec,
    config: RewardConfig = RewardConfig(),
    cancel_count: int = 0,
    peak_equity: float | None = None,
    terminal: bool = False,
) -> RewardStepResult:
    previous_position = _require_position(previous_position)
    fills_tuple = _fills_tuple(fills)
    previous_book_top = _require_book_top(previous_book_top)
    current_book_top = _require_book_top(current_book_top)
    symbol_spec = _require_symbol_spec(symbol_spec)
    config = _require_config(config)
    cancel_count = _require_nonnegative_int(cancel_count, "cancel_count")
    if peak_equity is not None:
        peak_equity = _require_finite_float(peak_equity, "peak_equity")
    terminal = _require_bool(terminal, "terminal")

    previous_mark_price = mark_price_from_book_top(previous_book_top, symbol_spec)
    current_mark_price = mark_price_from_book_top(current_book_top, symbol_spec)
    previous_equity = previous_position.mark_to_market(previous_mark_price, symbol_spec.contract_size)
    current_position = apply_fills_to_position(previous_position, fills_tuple, symbol_spec=symbol_spec)
    current_equity = current_position.mark_to_market(current_mark_price, symbol_spec.contract_size)
    turnover_notional = fills_turnover_notional(fills_tuple, symbol_spec=symbol_spec)
    reward = compute_reward_components(
        previous_equity=previous_equity,
        current_equity=current_equity,
        current_position=current_position,
        current_mark_price=current_mark_price,
        symbol_spec=symbol_spec,
        turnover_notional=turnover_notional,
        cancel_count=cancel_count,
        peak_equity=peak_equity,
        terminal=terminal,
        config=config,
    )
    if peak_equity is None:
        updated_peak = max(previous_equity, current_equity)
    else:
        updated_peak = max(peak_equity, previous_equity, current_equity)

    return RewardStepResult(
        position=current_position,
        reward=reward,
        previous_equity=previous_equity,
        current_equity=current_equity,
        peak_equity=updated_peak,
        turnover_notional=turnover_notional,
    )
