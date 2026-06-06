"""Diagnostic grid compatibility checks for market data and symbol rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from typing import Mapping

from mmrt.metadata.symbol_rules import ExchangeSymbolRules


class RuleCompatibilityMode(str, Enum):
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class RuleCompatibilityConfig:
    mode: RuleCompatibilityMode = RuleCompatibilityMode.WARN
    price_tolerance_ticks: float = 1e-6
    qty_tolerance_steps: float = 1e-6
    max_examples: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", self.mode if isinstance(self.mode, RuleCompatibilityMode) else RuleCompatibilityMode(self.mode))
        if self.price_tolerance_ticks < 0 or self.qty_tolerance_steps < 0:
            raise ValueError("compatibility tolerances must be >= 0")
        if self.max_examples < 0:
            raise ValueError("max_examples must be >= 0")


@dataclass(frozen=True, slots=True)
class RuleCompatibilityReport:
    mode: RuleCompatibilityMode
    price_count: int
    price_grid_violation_count: int
    price_grid_violation_fraction: float
    max_abs_price_residual_ticks: float
    qty_count: int
    qty_grid_violation_count: int
    qty_grid_violation_fraction: float
    max_abs_qty_residual_steps: float
    min_price_seen: float | None
    max_price_seen: float | None
    min_qty_seen: float | None
    max_qty_seen: float | None
    examples: tuple[dict[str, object], ...]
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", self.mode if isinstance(self.mode, RuleCompatibilityMode) else RuleCompatibilityMode(self.mode))
        if self.status not in {"ok", "warning", "error"}:
            raise ValueError("status must be ok, warning, or error")
        object.__setattr__(self, "examples", tuple(dict(item) for item in self.examples))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "price_count": self.price_count,
            "price_grid_violation_count": self.price_grid_violation_count,
            "price_grid_violation_fraction": self.price_grid_violation_fraction,
            "max_abs_price_residual_ticks": self.max_abs_price_residual_ticks,
            "qty_count": self.qty_count,
            "qty_grid_violation_count": self.qty_grid_violation_count,
            "qty_grid_violation_fraction": self.qty_grid_violation_fraction,
            "max_abs_qty_residual_steps": self.max_abs_qty_residual_steps,
            "min_price_seen": self.min_price_seen,
            "max_price_seen": self.max_price_seen,
            "min_qty_seen": self.min_qty_seen,
            "max_qty_seen": self.max_qty_seen,
            "examples": [dict(item) for item in self.examples],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuleCompatibilityReport":
        if not isinstance(payload, Mapping):
            raise ValueError("compatibility report must be a mapping")
        return cls(
            mode=RuleCompatibilityMode(payload.get("mode")),
            price_count=int(payload.get("price_count", 0)),
            price_grid_violation_count=int(payload.get("price_grid_violation_count", 0)),
            price_grid_violation_fraction=float(payload.get("price_grid_violation_fraction", 0.0)),
            max_abs_price_residual_ticks=float(payload.get("max_abs_price_residual_ticks", 0.0)),
            qty_count=int(payload.get("qty_count", 0)),
            qty_grid_violation_count=int(payload.get("qty_grid_violation_count", 0)),
            qty_grid_violation_fraction=float(payload.get("qty_grid_violation_fraction", 0.0)),
            max_abs_qty_residual_steps=float(payload.get("max_abs_qty_residual_steps", 0.0)),
            min_price_seen=payload.get("min_price_seen"),  # type: ignore[arg-type]
            max_price_seen=payload.get("max_price_seen"),  # type: ignore[arg-type]
            min_qty_seen=payload.get("min_qty_seen"),  # type: ignore[arg-type]
            max_qty_seen=payload.get("max_qty_seen"),  # type: ignore[arg-type]
            examples=tuple(payload.get("examples", ())),  # type: ignore[arg-type]
            status=str(payload.get("status", "ok")),
        )


def _nearest_integer(value: Decimal) -> Decimal:
    floor = value.to_integral_value(rounding=ROUND_FLOOR)
    ceil = floor + 1
    return floor if abs(value - floor) <= abs(value - ceil) else ceil


class RuleCompatibilityAccumulator:
    def __init__(self, rules: ExchangeSymbolRules, config: RuleCompatibilityConfig):
        if not isinstance(rules, ExchangeSymbolRules):
            raise ValueError("rules must be ExchangeSymbolRules")
        if not isinstance(config, RuleCompatibilityConfig):
            raise ValueError("config must be RuleCompatibilityConfig")
        self.rules = rules
        self.config = config
        self.price_count = self.qty_count = 0
        self.price_violations = self.qty_violations = 0
        self.max_price_residual = Decimal("0")
        self.max_qty_residual = Decimal("0")
        self.min_price: float | None = None
        self.max_price: float | None = None
        self.min_qty: float | None = None
        self.max_qty: float | None = None
        self.examples: list[dict[str, object]] = []

    def observe_price(self, price: float, *, source: str, local_ts_us: int | None = None) -> None:
        if self.config.mode is RuleCompatibilityMode.OFF:
            return
        p = float(price)
        self.price_count += 1
        self.min_price = p if self.min_price is None else min(self.min_price, p)
        self.max_price = p if self.max_price is None else max(self.max_price, p)
        residual = abs((Decimal(str(price)) / self.rules.tick_size) - _nearest_integer(Decimal(str(price)) / self.rules.tick_size))
        self.max_price_residual = max(self.max_price_residual, residual)
        if residual > Decimal(str(self.config.price_tolerance_ticks)):
            self.price_violations += 1
            self._add_example("price", p, residual, source, local_ts_us)

    def observe_qty(self, qty: float, *, source: str, local_ts_us: int | None = None) -> None:
        if self.config.mode is RuleCompatibilityMode.OFF:
            return
        q = float(qty)
        self.qty_count += 1
        self.min_qty = q if self.min_qty is None else min(self.min_qty, q)
        self.max_qty = q if self.max_qty is None else max(self.max_qty, q)
        residual = abs((Decimal(str(qty)) / self.rules.step_size) - _nearest_integer(Decimal(str(qty)) / self.rules.step_size))
        self.max_qty_residual = max(self.max_qty_residual, residual)
        if residual > Decimal(str(self.config.qty_tolerance_steps)):
            self.qty_violations += 1
            self._add_example("qty", q, residual, source, local_ts_us)

    def _add_example(self, kind: str, value: float, residual: Decimal, source: str, local_ts_us: int | None) -> None:
        if len(self.examples) >= self.config.max_examples:
            return
        item: dict[str, object] = {"kind": kind, "value": value, "residual": float(residual), "source": source}
        if local_ts_us is not None:
            item["local_ts_us"] = local_ts_us
        self.examples.append(item)

    def report(self) -> RuleCompatibilityReport:
        status = "ok"
        if self.config.mode is RuleCompatibilityMode.WARN and (self.price_violations or self.qty_violations):
            status = "warning"
        if self.config.mode is RuleCompatibilityMode.STRICT and (self.price_violations or self.qty_violations):
            status = "error"
        report = RuleCompatibilityReport(
            mode=self.config.mode,
            price_count=self.price_count,
            price_grid_violation_count=self.price_violations,
            price_grid_violation_fraction=(self.price_violations / self.price_count) if self.price_count else 0.0,
            max_abs_price_residual_ticks=float(self.max_price_residual),
            qty_count=self.qty_count,
            qty_grid_violation_count=self.qty_violations,
            qty_grid_violation_fraction=(self.qty_violations / self.qty_count) if self.qty_count else 0.0,
            max_abs_qty_residual_steps=float(self.max_qty_residual),
            min_price_seen=self.min_price,
            max_price_seen=self.max_price,
            min_qty_seen=self.min_qty,
            max_qty_seen=self.max_qty,
            examples=tuple(self.examples),
            status=status,
        )
        if self.config.mode is RuleCompatibilityMode.STRICT and status == "error":
            raise ValueError("symbol rule compatibility strict mode failed")
        return report
