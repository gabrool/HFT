"""Diagnostics for execution simulation metric summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

__all__ = [
    "ExecutionDiagnosticsConfig",
    "ExecutionDiagnosticReport",
    "diagnose_execution_metrics",
]

_REQUIRED_SECTIONS = ("steps", "rewards", "fills", "orders", "turnover", "position", "equity")


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _optional_nonnegative_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite nonnegative float or None")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _optional_finite_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float or None")
    return float(value)


def _number(metrics: Mapping[str, object], section: str, key: str, default: float = 0.0) -> float:
    value = _section(metrics, section).get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = float(value)
    return value if math.isfinite(value) else default


def _section(metrics: Mapping[str, object], section: str) -> Mapping[str, object]:
    value = metrics.get(section, {})
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True)
class ExecutionDiagnosticsConfig:
    warn_if_no_fills: bool = True
    warn_if_all_quotes_disabled: bool = True
    warn_if_no_turnover: bool = True
    min_steps_warn: int = 1
    max_cancel_rate_warn: float | None = 1.0
    max_abs_inventory_qty_warn: float | None = None
    max_drawdown_warn: float | None = None
    min_total_reward_warn: float | None = None

    def __post_init__(self) -> None:
        _require_bool(self.warn_if_no_fills, "warn_if_no_fills")
        _require_bool(self.warn_if_all_quotes_disabled, "warn_if_all_quotes_disabled")
        _require_bool(self.warn_if_no_turnover, "warn_if_no_turnover")
        _require_nonnegative_int(self.min_steps_warn, "min_steps_warn")
        object.__setattr__(self, "max_cancel_rate_warn", _optional_nonnegative_float(self.max_cancel_rate_warn, "max_cancel_rate_warn"))
        object.__setattr__(self, "max_abs_inventory_qty_warn", _optional_nonnegative_float(self.max_abs_inventory_qty_warn, "max_abs_inventory_qty_warn"))
        object.__setattr__(self, "max_drawdown_warn", _optional_nonnegative_float(self.max_drawdown_warn, "max_drawdown_warn"))
        object.__setattr__(self, "min_total_reward_warn", _optional_finite_float(self.min_total_reward_warn, "min_total_reward_warn"))


@dataclass(frozen=True, slots=True)
class ExecutionDiagnosticReport:
    status: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    metrics: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metrics": self.metrics,
        }


def _contains_nonfinite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(v) for v in value)
    return False


def diagnose_execution_metrics(
    metrics: Mapping[str, object],
    *,
    config: ExecutionDiagnosticsConfig = ExecutionDiagnosticsConfig(),
) -> ExecutionDiagnosticReport:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    if not isinstance(config, ExecutionDiagnosticsConfig):
        raise ValueError("config must be ExecutionDiagnosticsConfig")

    errors: list[str] = []
    warnings: list[str] = []

    for section in _REQUIRED_SECTIONS:
        if not isinstance(metrics.get(section), Mapping):
            errors.append("missing_required_metrics_section")
            break
    if _contains_nonfinite(metrics):
        errors.append("nonfinite_metric_value")

    steps_count = int(_number(metrics, "steps", "count", 0.0))
    if steps_count <= 0:
        errors.append("zero_steps")

    if steps_count < config.min_steps_warn:
        warnings.append("low_step_count")
    if _number(metrics, "steps", "events_processed_total", 0.0) <= 0:
        warnings.append("no_events_processed")
    if config.warn_if_no_fills and _number(metrics, "fills", "count", 0.0) == 0:
        warnings.append("no_fills_observed")
    if config.warn_if_no_turnover and _number(metrics, "turnover", "notional_total", 0.0) == 0.0:
        warnings.append("no_turnover_observed")
    if config.warn_if_all_quotes_disabled and steps_count > 0 and _number(metrics, "orders", "all_quotes_disabled_count", 0.0) == steps_count:
        warnings.append("all_quotes_disabled")
    if config.max_cancel_rate_warn is not None and _number(metrics, "orders", "cancel_rate_per_step", 0.0) > config.max_cancel_rate_warn:
        warnings.append("high_cancel_rate")
    if config.max_abs_inventory_qty_warn is not None and _number(metrics, "position", "max_abs_inventory_qty", 0.0) > config.max_abs_inventory_qty_warn:
        warnings.append("max_abs_inventory_qty_exceeded")
    if config.max_drawdown_warn is not None and _number(metrics, "equity", "max_drawdown", 0.0) > config.max_drawdown_warn:
        warnings.append("max_drawdown_exceeded")
    if config.min_total_reward_warn is not None and _number(metrics, "rewards", "total", 0.0) < config.min_total_reward_warn:
        warnings.append("total_reward_below_threshold")
    if _number(metrics, "steps", "terminal_count", 0.0) == 0:
        warnings.append("episode_never_terminal")

    status = "error" if errors else "warning" if warnings else "ok"
    return ExecutionDiagnosticReport(
        status=status,
        warnings=tuple(warnings),
        errors=tuple(errors),
        metrics=dict(metrics),
    )
