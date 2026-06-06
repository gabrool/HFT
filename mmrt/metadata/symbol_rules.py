"""Normalized, reproducible exchange symbol-rule artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


class SymbolRuleMode(str, Enum):
    CURRENT_RULES_REPLAY = "current_rules_replay"
    USER_SUPPLIED_RULES = "user_supplied_rules"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name)


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be Decimal-compatible")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be Decimal-compatible") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _tuple_str(values: object, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of strings")
    try:
        seq = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of strings") from exc
    return tuple(_nonempty(v, f"{name}[{i}]") for i, v in enumerate(seq))


@dataclass(frozen=True, slots=True)
class ExchangeSymbolRules:
    exchange: str
    symbol: str
    mode: SymbolRuleMode

    base_asset: str
    quote_asset: str
    margin_asset: str | None
    contract_type: str
    status: str

    tick_size: Decimal
    min_price: Decimal
    max_price: Decimal

    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal

    min_notional: Decimal
    contract_size: Decimal = Decimal("1")

    allowed_order_types: tuple[str, ...] = ()
    allowed_time_in_force: tuple[str, ...] = ()
    post_only_time_in_force: str = "GTX"

    source: str = ""
    source_sha256: str = ""
    captured_at_utc: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange", _nonempty(self.exchange, "exchange"))
        object.__setattr__(self, "symbol", _nonempty(self.symbol, "symbol"))
        object.__setattr__(self, "mode", self.mode if isinstance(self.mode, SymbolRuleMode) else SymbolRuleMode(self.mode))
        object.__setattr__(self, "base_asset", _nonempty(self.base_asset, "base_asset"))
        object.__setattr__(self, "quote_asset", _nonempty(self.quote_asset, "quote_asset"))
        object.__setattr__(self, "margin_asset", _optional_str(self.margin_asset, "margin_asset"))
        object.__setattr__(self, "contract_type", _nonempty(self.contract_type, "contract_type"))
        object.__setattr__(self, "status", _nonempty(self.status, "status"))
        for field in (
            "tick_size", "min_price", "max_price", "step_size", "min_qty", "max_qty", "min_notional", "contract_size"
        ):
            object.__setattr__(self, field, _decimal(getattr(self, field), field))
        if self.tick_size <= 0:
            raise ValueError("tick_size must be > 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        if self.min_qty < 0:
            raise ValueError("min_qty must be >= 0")
        if self.max_qty <= 0 or self.max_qty < self.min_qty:
            raise ValueError("max_qty must be > 0 and >= min_qty")
        if self.min_notional < 0:
            raise ValueError("min_notional must be >= 0")
        if self.contract_size <= 0:
            raise ValueError("contract_size must be > 0")
        object.__setattr__(self, "allowed_order_types", _tuple_str(self.allowed_order_types, "allowed_order_types"))
        object.__setattr__(self, "allowed_time_in_force", _tuple_str(self.allowed_time_in_force, "allowed_time_in_force"))
        object.__setattr__(self, "post_only_time_in_force", _nonempty(self.post_only_time_in_force, "post_only_time_in_force"))
        if "LIMIT" not in self.allowed_order_types:
            raise ValueError("allowed_order_types must include LIMIT")
        if self.post_only_time_in_force not in self.allowed_time_in_force:
            raise ValueError("post_only_time_in_force must be in allowed_time_in_force")
        if not isinstance(self.source, str):
            raise ValueError("source must be str")
        if not isinstance(self.source_sha256, str):
            raise ValueError("source_sha256 must be str")
        object.__setattr__(self, "captured_at_utc", _optional_str(self.captured_at_utc, "captured_at_utc"))

    def to_dict(self) -> dict[str, object]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "mode": self.mode.value,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "margin_asset": self.margin_asset,
            "contract_type": self.contract_type,
            "status": self.status,
            "tick_size": str(self.tick_size),
            "min_price": str(self.min_price),
            "max_price": str(self.max_price),
            "step_size": str(self.step_size),
            "min_qty": str(self.min_qty),
            "max_qty": str(self.max_qty),
            "min_notional": str(self.min_notional),
            "contract_size": str(self.contract_size),
            "allowed_order_types": list(self.allowed_order_types),
            "allowed_time_in_force": list(self.allowed_time_in_force),
            "post_only_time_in_force": self.post_only_time_in_force,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "captured_at_utc": self.captured_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ExchangeSymbolRules":
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        kwargs = dict(payload)
        for key in ("tick_size", "min_price", "max_price", "step_size", "min_qty", "max_qty", "min_notional", "contract_size"):
            if key in kwargs:
                kwargs[key] = _decimal(kwargs[key], key)
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_symbol_spec(self):
        from mmrt.execution.contracts import SymbolSpec

        return SymbolSpec(
            exchange=self.exchange,
            symbol=self.symbol,
            tick_size=float(self.tick_size),
            step_size=float(self.step_size),
            min_qty=float(self.min_qty),
            max_qty=float(self.max_qty),
            min_notional=float(self.min_notional),
            contract_size=float(self.contract_size),
        )


def canonical_symbol_rules_json(rules: ExchangeSymbolRules) -> str:
    if not isinstance(rules, ExchangeSymbolRules):
        raise ValueError("rules must be ExchangeSymbolRules")
    return json.dumps(rules.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_symbol_rules_json(path: str | Path) -> ExchangeSymbolRules:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("symbol rules JSON must contain an object")
    return ExchangeSymbolRules.from_dict(payload)


def write_symbol_rules_json(path: str | Path, rules: ExchangeSymbolRules, *, overwrite: bool = False) -> None:
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(f"JSON output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_symbol_rules_json(rules) + "\n", encoding="utf-8")
