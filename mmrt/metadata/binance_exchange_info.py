"""Parse local Binance USD-M exchangeInfo snapshots into symbol rules."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Mapping

from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _decimal_str(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(_as_str(value, name)))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal string") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _str_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of strings")
    try:
        seq = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be a list of strings") from exc
    return tuple(_as_str(item, f"{name}[{i}]") for i, item in enumerate(seq))


def parse_binance_usdm_exchange_info_symbol(
    payload: Mapping[str, object],
    *,
    symbol: str,
    exchange: str = "binance-futures",
    mode: SymbolRuleMode = SymbolRuleMode.CURRENT_RULES_REPLAY,
    source: str = "binance_usdm_exchange_info",
    source_sha256: str = "",
    captured_at_utc: str | None = None,
) -> ExchangeSymbolRules:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    symbol = _as_str(symbol, "symbol")
    exchange = _as_str(exchange, "exchange")
    mode = mode if isinstance(mode, SymbolRuleMode) else SymbolRuleMode(mode)
    symbols = payload.get("symbols")
    if isinstance(symbols, (str, bytes)):
        raise ValueError("symbols must be a list")
    try:
        symbol_entries = tuple(symbols)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("symbols must be a list") from exc
    entry = None
    for idx, item in enumerate(symbol_entries):
        item_map = _as_mapping(item, f"symbols[{idx}]")
        if item_map.get("symbol") == symbol:
            entry = item_map
            break
    if entry is None:
        raise ValueError(f"symbol {symbol!r} not found in exchangeInfo")

    filters_raw = entry.get("filters")
    if isinstance(filters_raw, (str, bytes)):
        raise ValueError("filters must be a list")
    try:
        filters_seq = tuple(filters_raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("filters must be a list") from exc
    filters: dict[str, Mapping[str, object]] = {}
    for idx, filt in enumerate(filters_seq):
        filt_map = _as_mapping(filt, f"filters[{idx}]")
        filters[_as_str(filt_map.get("filterType"), f"filters[{idx}].filterType")] = filt_map
    missing = [name for name in ("PRICE_FILTER", "LOT_SIZE", "MIN_NOTIONAL") if name not in filters]
    if missing:
        raise ValueError(f"missing required Binance filters: {', '.join(missing)}")
    price_filter = filters["PRICE_FILTER"]
    lot_size = filters["LOT_SIZE"]
    min_notional = filters["MIN_NOTIONAL"]
    order_types = _str_tuple(entry.get("orderTypes"), "orderTypes")
    tif = _str_tuple(entry.get("timeInForce"), "timeInForce")
    if "LIMIT" not in order_types:
        raise ValueError("Binance symbol must support LIMIT orders")
    if "GTX" not in tif:
        raise ValueError("Binance symbol must support GTX timeInForce")
    contract_type = _as_str(entry.get("contractType"), "contractType")
    if contract_type != "PERPETUAL":
        raise ValueError("Binance USD-M execution tape pipeline requires PERPETUAL contractType")
    status = _as_str(entry.get("status"), "status")
    if mode is SymbolRuleMode.CURRENT_RULES_REPLAY and status != "TRADING":
        raise ValueError("current_rules_replay requires TRADING status")
    return ExchangeSymbolRules(
        exchange=exchange,
        symbol=symbol,
        mode=mode,
        base_asset=_as_str(entry.get("baseAsset"), "baseAsset"),
        quote_asset=_as_str(entry.get("quoteAsset"), "quoteAsset"),
        margin_asset=_as_str(entry.get("marginAsset"), "marginAsset") if entry.get("marginAsset") is not None else None,
        contract_type=contract_type,
        status=status,
        tick_size=_decimal_str(price_filter.get("tickSize"), "PRICE_FILTER.tickSize"),
        min_price=_decimal_str(price_filter.get("minPrice"), "PRICE_FILTER.minPrice"),
        max_price=_decimal_str(price_filter.get("maxPrice"), "PRICE_FILTER.maxPrice"),
        step_size=_decimal_str(lot_size.get("stepSize"), "LOT_SIZE.stepSize"),
        min_qty=_decimal_str(lot_size.get("minQty"), "LOT_SIZE.minQty"),
        max_qty=_decimal_str(lot_size.get("maxQty"), "LOT_SIZE.maxQty"),
        min_notional=_decimal_str(min_notional.get("notional"), "MIN_NOTIONAL.notional"),
        allowed_order_types=order_types,
        allowed_time_in_force=tif,
        post_only_time_in_force="GTX",
        source=source,
        source_sha256=source_sha256,
        captured_at_utc=captured_at_utc,
    )


def load_binance_usdm_exchange_info_symbol_rules(
    path: str | Path,
    *,
    symbol: str,
    exchange: str = "binance-futures",
    mode: SymbolRuleMode = SymbolRuleMode.CURRENT_RULES_REPLAY,
    captured_at_utc: str | None = None,
) -> ExchangeSymbolRules:
    raw = Path(path).read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("exchangeInfo JSON must contain an object")
    return parse_binance_usdm_exchange_info_symbol(
        payload,
        symbol=symbol,
        exchange=exchange,
        mode=mode,
        source="binance_usdm_exchange_info",
        source_sha256=source_sha256,
        captured_at_utc=captured_at_utc,
    )
