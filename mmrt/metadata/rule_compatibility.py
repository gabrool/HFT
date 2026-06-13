"""Diagnostic grid compatibility checks for market data and symbol rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
import math
from typing import Mapping

import numpy as np

from mmrt.metadata.symbol_rules import ExchangeSymbolRules


class RuleCompatibilityMode(str, Enum):
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


def _finite_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative float")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")
    return out


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a nonnegative int")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


@dataclass(frozen=True, slots=True)
class RuleCompatibilityConfig:
    mode: RuleCompatibilityMode = RuleCompatibilityMode.WARN
    price_tolerance_ticks: float = 1e-6
    qty_tolerance_steps: float = 1e-6
    max_examples: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", self.mode if isinstance(self.mode, RuleCompatibilityMode) else RuleCompatibilityMode(self.mode))
        object.__setattr__(
            self,
            "price_tolerance_ticks",
            _finite_nonnegative_float(self.price_tolerance_ticks, "price_tolerance_ticks"),
        )
        object.__setattr__(
            self,
            "qty_tolerance_steps",
            _finite_nonnegative_float(self.qty_tolerance_steps, "qty_tolerance_steps"),
        )
        object.__setattr__(self, "max_examples", _nonnegative_int(self.max_examples, "max_examples"))


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


_RESIDUAL_FLOAT_GUARD = 1e-9


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
        self._tick_size_float = float(rules.tick_size)
        self._step_size_float = float(rules.step_size)
        self._price_tolerance_float = float(config.price_tolerance_ticks)
        self._qty_tolerance_float = float(config.qty_tolerance_steps)
        self._price_tolerance_decimal = Decimal(str(config.price_tolerance_ticks))
        self._qty_tolerance_decimal = Decimal(str(config.qty_tolerance_steps))
        self._max_price_residual_float = 0.0
        self._max_qty_residual_float = 0.0

    def observe_price(self, price: float, *, source: str, local_ts_us: int | None = None) -> None:
        if self.config.mode is RuleCompatibilityMode.OFF:
            return
        p = float(price)
        self.price_count += 1
        if self.min_price is None or p < self.min_price:
            self.min_price = p
        if self.max_price is None or p > self.max_price:
            self.max_price = p
        # Fast float screen: skip the exact decimal residual when the value is
        # comfortably on the grid and cannot raise the running maximum.
        ratio = p / self._tick_size_float
        residual_float = abs(ratio - round(ratio))
        if (
            residual_float + _RESIDUAL_FLOAT_GUARD < self._price_tolerance_float
            and residual_float + _RESIDUAL_FLOAT_GUARD <= self._max_price_residual_float
        ):
            return
        self._escalate_price(p, source, local_ts_us)

    def observe_price_array(self, prices: np.ndarray, *, source: str, local_ts_us: np.ndarray | None = None) -> None:
        """Observe a finite float64 price array; state-identical to per-value calls in order.

        The float screen only skips values that are provably on the grid and
        provably unable to raise the running maximum residual, so screening
        the whole array against the maximum held before the call escalates a
        superset of the values the per-value path escalates; the extra values
        cannot change any counted, reported, or example state.
        """
        if self.config.mode is RuleCompatibilityMode.OFF:
            return
        n = int(prices.size)
        if n == 0:
            return
        self.price_count += n
        lo = float(prices.min())
        hi = float(prices.max())
        if self.min_price is None or lo < self.min_price:
            self.min_price = lo
        if self.max_price is None or hi > self.max_price:
            self.max_price = hi
        ratio = prices / self._tick_size_float
        screened = np.abs(ratio - np.rint(ratio)) + _RESIDUAL_FLOAT_GUARD
        candidates = ~((screened < self._price_tolerance_float) & (screened <= self._max_price_residual_float))
        if not candidates.any():
            return
        candidate_index = np.flatnonzero(candidates)
        # Escalate each distinct value once: equal floats share one exact
        # residual, so violation counts multiply by occurrence and the
        # capped examples replay the violating rows in order.
        uniques, inverse = np.unique(prices[candidate_index], return_inverse=True)
        residuals = [self._exact_price_residual(value) for value in uniques.tolist()]
        best = max(residuals)
        if best > self.max_price_residual:
            self.max_price_residual = best
            self._max_price_residual_float = float(best) - _RESIDUAL_FLOAT_GUARD
        violating = np.fromiter((residual > self._price_tolerance_decimal for residual in residuals), dtype=bool, count=len(residuals))
        if not violating.any():
            return
        violating_positions = np.flatnonzero(violating[inverse])
        self.price_violations += int(violating_positions.size)
        room = self.config.max_examples - len(self.examples)
        for position in violating_positions[: max(room, 0)].tolist():
            row = int(candidate_index[position])
            self._add_example(
                "price",
                float(prices[row]),
                residuals[int(inverse[position])],
                source,
                None if local_ts_us is None else int(local_ts_us[row]),
            )

    def _exact_price_residual(self, price: float) -> Decimal:
        ratio = Decimal(str(price)) / self.rules.tick_size
        return abs(ratio - _nearest_integer(ratio))

    def _escalate_price(self, price: float, source: str, local_ts_us: int | None) -> None:
        residual = self._exact_price_residual(price)
        if residual > self.max_price_residual:
            self.max_price_residual = residual
            self._max_price_residual_float = float(residual) - _RESIDUAL_FLOAT_GUARD
        if residual > self._price_tolerance_decimal:
            self.price_violations += 1
            self._add_example("price", price, residual, source, local_ts_us)

    def observe_qty(self, qty: float, *, source: str, local_ts_us: int | None = None) -> None:
        if self.config.mode is RuleCompatibilityMode.OFF:
            return
        q = float(qty)
        self.qty_count += 1
        if self.min_qty is None or q < self.min_qty:
            self.min_qty = q
        if self.max_qty is None or q > self.max_qty:
            self.max_qty = q
        ratio = q / self._step_size_float
        residual_float = abs(ratio - round(ratio))
        if (
            residual_float + _RESIDUAL_FLOAT_GUARD < self._qty_tolerance_float
            and residual_float + _RESIDUAL_FLOAT_GUARD <= self._max_qty_residual_float
        ):
            return
        self._escalate_qty(q, source, local_ts_us)

    def observe_qty_array(self, qtys: np.ndarray, *, source: str, local_ts_us: np.ndarray | None = None) -> None:
        """Observe a finite float64 qty array; state-identical to per-value calls in order."""
        if self.config.mode is RuleCompatibilityMode.OFF:
            return
        n = int(qtys.size)
        if n == 0:
            return
        self.qty_count += n
        lo = float(qtys.min())
        hi = float(qtys.max())
        if self.min_qty is None or lo < self.min_qty:
            self.min_qty = lo
        if self.max_qty is None or hi > self.max_qty:
            self.max_qty = hi
        ratio = qtys / self._step_size_float
        screened = np.abs(ratio - np.rint(ratio)) + _RESIDUAL_FLOAT_GUARD
        candidates = ~((screened < self._qty_tolerance_float) & (screened <= self._max_qty_residual_float))
        if not candidates.any():
            return
        candidate_index = np.flatnonzero(candidates)
        uniques, inverse = np.unique(qtys[candidate_index], return_inverse=True)
        residuals = [self._exact_qty_residual(value) for value in uniques.tolist()]
        best = max(residuals)
        if best > self.max_qty_residual:
            self.max_qty_residual = best
            self._max_qty_residual_float = float(best) - _RESIDUAL_FLOAT_GUARD
        violating = np.fromiter((residual > self._qty_tolerance_decimal for residual in residuals), dtype=bool, count=len(residuals))
        if not violating.any():
            return
        violating_positions = np.flatnonzero(violating[inverse])
        self.qty_violations += int(violating_positions.size)
        room = self.config.max_examples - len(self.examples)
        for position in violating_positions[: max(room, 0)].tolist():
            row = int(candidate_index[position])
            self._add_example(
                "qty",
                float(qtys[row]),
                residuals[int(inverse[position])],
                source,
                None if local_ts_us is None else int(local_ts_us[row]),
            )

    def _exact_qty_residual(self, qty: float) -> Decimal:
        ratio = Decimal(str(qty)) / self.rules.step_size
        return abs(ratio - _nearest_integer(ratio))

    def _escalate_qty(self, qty: float, source: str, local_ts_us: int | None) -> None:
        residual = self._exact_qty_residual(qty)
        if residual > self.max_qty_residual:
            self.max_qty_residual = residual
            self._max_qty_residual_float = float(residual) - _RESIDUAL_FLOAT_GUARD
        if residual > self._qty_tolerance_decimal:
            self.qty_violations += 1
            self._add_example("qty", qty, residual, source, local_ts_us)

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
