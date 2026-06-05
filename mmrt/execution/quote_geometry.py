"""Quote construction from continuous policy actions.

This module is intentionally small, pure, and dependency-light. It converts
continuous latent action values into exchange-valid, maker-safe QuoteIntent
objects using the typed execution contracts.
"""

from dataclasses import dataclass
import math

from mmrt.execution.contracts import ActionSpec, BookTop, QuoteIntent, SymbolSpec

__all__ = [
    "QuoteGeometryConfig",
    "ContinuousQuoteAction",
    "QuoteGeometryResult",
    "raw_distance_to_ticks",
    "raw_size_to_qty",
    "continuous_action_to_quote",
]


_REASON_DISABLED_BY_ACTION = "disabled_by_action"
_REASON_DISABLED_BY_ACTION_SPEC = "disabled_by_action_spec"
_REASON_MISSING_BOOK_TOP = "missing_book_top"
_REASON_INVENTORY_LIMIT = "inventory_limit"
_REASON_POSITION_NOTIONAL_LIMIT = "position_notional_limit"
_REASON_QTY_BELOW_MIN = "qty_below_min"
_REASON_NOTIONAL_BELOW_MIN = "notional_below_min"
_REASON_INVALID_GEOMETRY = "invalid_geometry"


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


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _sigmoid(x: float) -> float:
    x = _require_finite_float(x, "x")
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class QuoteGeometryConfig:
    """Configuration for shaping continuous actions into quote intents."""

    min_distance_ticks: int = 1
    default_order_qty: float = 0.001
    max_inventory_abs_qty: float | None = None
    max_position_notional: float | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.min_distance_ticks, "min_distance_ticks")
        object.__setattr__(
            self,
            "default_order_qty",
            _require_positive_float(self.default_order_qty, "default_order_qty"),
        )
        if self.max_inventory_abs_qty is not None:
            object.__setattr__(
                self,
                "max_inventory_abs_qty",
                _require_positive_float(self.max_inventory_abs_qty, "max_inventory_abs_qty"),
            )
        if self.max_position_notional is not None:
            object.__setattr__(
                self,
                "max_position_notional",
                _require_positive_float(self.max_position_notional, "max_position_notional"),
            )


@dataclass(frozen=True, slots=True)
class ContinuousQuoteAction:
    """Continuous latent action emitted by a future policy."""

    bid_enable_logit: float
    ask_enable_logit: float
    bid_distance_raw: float
    ask_distance_raw: float
    bid_size_raw: float
    ask_size_raw: float

    def __post_init__(self) -> None:
        for name in (
            "bid_enable_logit",
            "ask_enable_logit",
            "bid_distance_raw",
            "ask_distance_raw",
            "bid_size_raw",
            "ask_size_raw",
        ):
            object.__setattr__(self, name, _require_finite_float(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class QuoteGeometryResult:
    """Quote intent plus stable debug metadata for quote shaping decisions."""

    quote: QuoteIntent
    bid_distance_ticks: int
    ask_distance_ticks: int
    bid_disabled_reason: str = ""
    ask_disabled_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.quote, QuoteIntent):
            raise ValueError("quote must be QuoteIntent")
        _require_nonnegative_int(self.bid_distance_ticks, "bid_distance_ticks")
        _require_nonnegative_int(self.ask_distance_ticks, "ask_distance_ticks")
        if not isinstance(self.bid_disabled_reason, str):
            raise ValueError("bid_disabled_reason must be str")
        if not isinstance(self.ask_disabled_reason, str):
            raise ValueError("ask_disabled_reason must be str")

def _disabled_result(bid_reason: str, ask_reason: str) -> QuoteGeometryResult:
    return QuoteGeometryResult(
        quote=QuoteIntent(False, False),
        bid_distance_ticks=0,
        ask_distance_ticks=0,
        bid_disabled_reason=bid_reason,
        ask_disabled_reason=ask_reason,
    )


def raw_distance_to_ticks(raw: float, *, max_distance_ticks: int, min_distance_ticks: int = 1) -> int:
    """Map an unconstrained finite raw distance to bounded integer ticks."""

    raw = _require_finite_float(raw, "raw")
    _require_positive_int(max_distance_ticks, "max_distance_ticks")
    _require_positive_int(min_distance_ticks, "min_distance_ticks")
    if min_distance_ticks > max_distance_ticks:
        raise ValueError("min_distance_ticks must be <= max_distance_ticks")

    unit = _sigmoid(raw)
    ticks = min_distance_ticks + int(math.floor(unit * (max_distance_ticks - min_distance_ticks + 1)))
    return min(ticks, max_distance_ticks)


def raw_size_to_qty(
    raw: float,
    *,
    symbol_spec: SymbolSpec,
    max_order_qty: float,
    default_order_qty: float,
) -> float:
    """Map an unconstrained finite raw size to a step-rounded valid quantity."""

    raw = _require_finite_float(raw, "raw")
    if not isinstance(symbol_spec, SymbolSpec):
        raise ValueError("symbol_spec must be SymbolSpec")
    max_order_qty = _require_positive_float(max_order_qty, "max_order_qty")
    default_order_qty = _require_positive_float(default_order_qty, "default_order_qty")

    max_qty = min(max_order_qty, symbol_spec.max_qty)
    if max_qty < symbol_spec.min_qty:
        return 0.0

    lo = max(symbol_spec.min_qty, min(default_order_qty, max_qty))
    hi = max_qty
    unit = _sigmoid(raw)
    target_qty = min(lo + unit * (hi - lo), max_qty)
    qty = symbol_spec.round_qty_down(target_qty)
    if qty > max_qty:
        qty = symbol_spec.round_qty_down(max_qty)
    if qty < symbol_spec.min_qty:
        return 0.0
    return qty


def continuous_action_to_quote(
    *,
    action: ContinuousQuoteAction,
    book_top: BookTop | None,
    symbol_spec: SymbolSpec,
    action_spec: ActionSpec,
    config: QuoteGeometryConfig = QuoteGeometryConfig(),
    inventory_qty: float = 0.0,
) -> QuoteGeometryResult:
    """Convert a continuous quote action into a validated maker-safe QuoteIntent."""

    if not isinstance(action, ContinuousQuoteAction):
        raise ValueError("action must be ContinuousQuoteAction")
    if book_top is not None and not isinstance(book_top, BookTop):
        raise ValueError("book_top must be BookTop or None")
    if not isinstance(symbol_spec, SymbolSpec):
        raise ValueError("symbol_spec must be SymbolSpec")
    if not isinstance(action_spec, ActionSpec):
        raise ValueError("action_spec must be ActionSpec")
    if not isinstance(config, QuoteGeometryConfig):
        raise ValueError("config must be QuoteGeometryConfig")
    inventory_qty = _require_finite_float(inventory_qty, "inventory_qty")

    if book_top is None:
        return _disabled_result(_REASON_MISSING_BOOK_TOP, _REASON_MISSING_BOOK_TOP)

    bid_distance_ticks = raw_distance_to_ticks(
        action.bid_distance_raw,
        max_distance_ticks=action_spec.max_distance_ticks,
        min_distance_ticks=config.min_distance_ticks,
    )
    ask_distance_ticks = raw_distance_to_ticks(
        action.ask_distance_raw,
        max_distance_ticks=action_spec.max_distance_ticks,
        min_distance_ticks=config.min_distance_ticks,
    )

    bid_enabled = action.bid_enable_logit > 0
    ask_enabled = action.ask_enable_logit > 0
    bid_reason = ""
    ask_reason = ""

    if not bid_enabled:
        bid_reason = _REASON_DISABLED_BY_ACTION
    if not ask_enabled:
        ask_reason = _REASON_DISABLED_BY_ACTION

    if bid_enabled and not action_spec.allow_bid:
        bid_enabled = False
        bid_reason = _REASON_DISABLED_BY_ACTION_SPEC
    if ask_enabled and not action_spec.allow_ask:
        ask_enabled = False
        ask_reason = _REASON_DISABLED_BY_ACTION_SPEC

    bid_price_tick = 0
    ask_price_tick = 0
    bid_qty = 0.0
    ask_qty = 0.0

    if bid_enabled:
        bid_qty = raw_size_to_qty(
            action.bid_size_raw,
            symbol_spec=symbol_spec,
            max_order_qty=action_spec.max_order_qty,
            default_order_qty=config.default_order_qty,
        )
        bid_price_tick = book_top.best_bid_tick - bid_distance_ticks + 1
        max_bid_tick = book_top.best_ask_tick - config.min_distance_ticks
        bid_price_tick = min(bid_price_tick, max_bid_tick)
        if bid_price_tick <= 0:
            bid_enabled = False
            bid_reason = _REASON_INVALID_GEOMETRY
        elif bid_qty <= 0:
            bid_enabled = False
            bid_reason = _REASON_QTY_BELOW_MIN
        elif config.max_inventory_abs_qty is not None and inventory_qty >= config.max_inventory_abs_qty:
            bid_enabled = False
            bid_reason = _REASON_INVENTORY_LIMIT
        elif not symbol_spec.is_valid_notional(bid_qty, bid_price_tick):
            bid_enabled = False
            bid_reason = _REASON_NOTIONAL_BELOW_MIN

    if ask_enabled:
        ask_qty = raw_size_to_qty(
            action.ask_size_raw,
            symbol_spec=symbol_spec,
            max_order_qty=action_spec.max_order_qty,
            default_order_qty=config.default_order_qty,
        )
        ask_price_tick = book_top.best_ask_tick + ask_distance_ticks - 1
        min_ask_tick = book_top.best_bid_tick + config.min_distance_ticks
        ask_price_tick = max(ask_price_tick, min_ask_tick)
        if ask_price_tick <= 0:
            ask_enabled = False
            ask_reason = _REASON_INVALID_GEOMETRY
        elif ask_qty <= 0:
            ask_enabled = False
            ask_reason = _REASON_QTY_BELOW_MIN
        elif config.max_inventory_abs_qty is not None and inventory_qty <= -config.max_inventory_abs_qty:
            ask_enabled = False
            ask_reason = _REASON_INVENTORY_LIMIT
        elif not symbol_spec.is_valid_notional(ask_qty, ask_price_tick):
            ask_enabled = False
            ask_reason = _REASON_NOTIONAL_BELOW_MIN

    if config.max_position_notional is not None:
        mid_price = book_top.mid_tick_x2 * 0.5 * symbol_spec.tick_size
        if bid_enabled:
            projected_inventory = inventory_qty + bid_qty
            projected_notional = abs(projected_inventory) * mid_price * symbol_spec.contract_size
            if projected_notional > config.max_position_notional:
                bid_enabled = False
                bid_reason = _REASON_POSITION_NOTIONAL_LIMIT
        if ask_enabled:
            projected_inventory = inventory_qty - ask_qty
            projected_notional = abs(projected_inventory) * mid_price * symbol_spec.contract_size
            if projected_notional > config.max_position_notional:
                ask_enabled = False
                ask_reason = _REASON_POSITION_NOTIONAL_LIMIT

    if bid_enabled and ask_enabled and bid_price_tick >= ask_price_tick:
        bid_enabled = False
        ask_enabled = False
        bid_reason = _REASON_INVALID_GEOMETRY
        ask_reason = _REASON_INVALID_GEOMETRY

    quote = QuoteIntent(
        bid_enabled=bid_enabled,
        ask_enabled=ask_enabled,
        bid_price_tick=bid_price_tick if bid_enabled else 0,
        ask_price_tick=ask_price_tick if ask_enabled else 0,
        bid_qty=bid_qty if bid_enabled else 0.0,
        ask_qty=ask_qty if ask_enabled else 0.0,
    )
    return QuoteGeometryResult(
        quote=quote,
        bid_distance_ticks=bid_distance_ticks,
        ask_distance_ticks=ask_distance_ticks,
        bid_disabled_reason=bid_reason if not bid_enabled else "",
        ask_disabled_reason=ask_reason if not ask_enabled else "",
    )
