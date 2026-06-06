"""Binance USDS-M Futures adapter policy for the MMRT data layer.

This module contains exchange/symbol/data-type policy for normalized Tardis Binance
USDS-M Futures data. It does not read files, normalize rows, merge events, run quality
checks, reconstruct books, compute features, or compute labels.
"""

from dataclasses import dataclass
from typing import Sequence

from mmrt.contracts import TardisDataType
BINANCE_FUTURES_EXCHANGE = "binance-futures"
SIDE_UNKNOWN = 0
SIDE_BUY = 1
SIDE_SELL = -1

BOOK_SIDE_UNKNOWN = 0
BOOK_SIDE_BID = 1
BOOK_SIDE_ASK = -1

BINANCE_FUTURES_SYMBOL = "BTCUSDT"
BINANCE_FUTURES_SYMBOLS = ("BTCUSDT",)

BINANCE_FUTURES_SOURCE_DATA_TYPES = (
    TardisDataType.BOOK_SNAPSHOT_25,
    TardisDataType.TRADES,
)

BINANCE_FUTURES_CONTEXT_DATA_TYPES = (
    TardisDataType.BOOK_TICKER,
    TardisDataType.DERIVATIVE_TICKER,
    TardisDataType.LIQUIDATIONS,
    TardisDataType.INCREMENTAL_BOOK_L2,
    TardisDataType.BOOK_SNAPSHOT_5,
)

BINANCE_FUTURES_UNSUPPORTED_DATA_TYPES = (
    TardisDataType.QUOTES,
    TardisDataType.OPTIONS_CHAIN,
)

BINANCE_FUTURES_ACCEPTED_DATA_TYPES = (
    *BINANCE_FUTURES_SOURCE_DATA_TYPES,
    *BINANCE_FUTURES_CONTEXT_DATA_TYPES,
)

BINANCE_FUTURES_DEFAULT_MERGE_RANKS = {
    TardisDataType.BOOK_SNAPSHOT_25: 0,
    TardisDataType.BOOK_SNAPSHOT_5: 1,
    TardisDataType.INCREMENTAL_BOOK_L2: 2,
    TardisDataType.TRADES: 3,
    TardisDataType.BOOK_TICKER: 4,
    TardisDataType.LIQUIDATIONS: 5,
    TardisDataType.DERIVATIVE_TICKER: 6,
}


def _validate_unique_merge_ranks() -> None:
    ranks = tuple(BINANCE_FUTURES_DEFAULT_MERGE_RANKS.values())
    if len(set(ranks)) != len(ranks):
        raise ValueError("BINANCE_FUTURES_DEFAULT_MERGE_RANKS values must be unique")
    missing = tuple(
        dtype
        for dtype in BINANCE_FUTURES_ACCEPTED_DATA_TYPES
        if dtype not in BINANCE_FUTURES_DEFAULT_MERGE_RANKS
    )
    if missing:
        raise ValueError(f"missing default merge ranks for: {missing!r}")


_validate_unique_merge_ranks()

BINANCE_FUTURES_TRADE_SIDE_TO_CODE = {
    "buy": SIDE_BUY,
    "sell": SIDE_SELL,
    "unknown": SIDE_UNKNOWN,
    "": SIDE_UNKNOWN,
}

BINANCE_FUTURES_BOOK_SIDE_TO_CODE = {
    "bid": BOOK_SIDE_BID,
    "ask": BOOK_SIDE_ASK,
    "": BOOK_SIDE_UNKNOWN,
}


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _coerce_data_type(data_type: TardisDataType | str) -> TardisDataType:
    if isinstance(data_type, TardisDataType):
        return data_type
    if isinstance(data_type, str):
        try:
            return TardisDataType(data_type)
        except ValueError as exc:
            raise ValueError(f"invalid data_type: {data_type!r}") from exc
    raise ValueError("data_type must be TardisDataType or str")


def _tuple_of_data_types(values: Sequence[TardisDataType | str], name: str) -> tuple[TardisDataType, ...]:
    seq = tuple(values)
    if not seq:
        raise ValueError(f"{name} must not be empty")
    out: list[TardisDataType] = []
    seen: set[TardisDataType] = set()
    for idx, value in enumerate(seq):
        dtype = _coerce_data_type(value)
        if dtype in seen:
            raise ValueError(f"{name}[{idx}] duplicates {dtype.value!r}")
        seen.add(dtype)
        out.append(dtype)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class BinanceFuturesMarket:
    exchange: str = BINANCE_FUTURES_EXCHANGE
    symbol: str = BINANCE_FUTURES_SYMBOL

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange", normalize_binance_futures_exchange(self.exchange))
        object.__setattr__(self, "symbol", normalize_binance_futures_symbol(self.symbol))


def normalize_binance_futures_exchange(exchange: str) -> str:
    normalized = _require_nonempty_str(exchange, "exchange")
    if normalized != BINANCE_FUTURES_EXCHANGE:
        raise ValueError(f"exchange must be {BINANCE_FUTURES_EXCHANGE!r}")
    return BINANCE_FUTURES_EXCHANGE


def normalize_binance_futures_symbol(symbol: str) -> str:
    normalized = _require_nonempty_str(symbol, "symbol").upper()
    if normalized not in BINANCE_FUTURES_SYMBOLS:
        raise ValueError(f"unsupported symbol: {symbol!r}")
    return normalized


def validate_binance_futures_market(exchange: str, symbol: str) -> BinanceFuturesMarket:
    return BinanceFuturesMarket(
        exchange=normalize_binance_futures_exchange(exchange),
        symbol=normalize_binance_futures_symbol(symbol),
    )


def is_binance_futures_source_data_type(data_type: TardisDataType | str) -> bool:
    return _coerce_data_type(data_type) in BINANCE_FUTURES_SOURCE_DATA_TYPES


def is_binance_futures_context_data_type(data_type: TardisDataType | str) -> bool:
    return _coerce_data_type(data_type) in BINANCE_FUTURES_CONTEXT_DATA_TYPES


def is_binance_futures_accepted_data_type(data_type: TardisDataType | str) -> bool:
    return _coerce_data_type(data_type) in BINANCE_FUTURES_ACCEPTED_DATA_TYPES


def require_binance_futures_data_type(data_type: TardisDataType | str) -> TardisDataType:
    dtype = _coerce_data_type(data_type)
    if dtype not in BINANCE_FUTURES_ACCEPTED_DATA_TYPES:
        raise ValueError(f"unsupported Binance futures data_type: {dtype.value}")
    return dtype


def normalize_binance_futures_data_types(
    data_types: Sequence[TardisDataType | str],
) -> tuple[TardisDataType, ...]:
    dtypes = _tuple_of_data_types(data_types, "data_types")
    return tuple(require_binance_futures_data_type(dtype) for dtype in dtypes)


def default_binance_futures_source_data_types() -> tuple[TardisDataType, ...]:
    return BINANCE_FUTURES_SOURCE_DATA_TYPES


def default_binance_futures_context_data_types() -> tuple[TardisDataType, ...]:
    return BINANCE_FUTURES_CONTEXT_DATA_TYPES


def default_binance_futures_accepted_data_types() -> tuple[TardisDataType, ...]:
    return BINANCE_FUTURES_ACCEPTED_DATA_TYPES


def binance_futures_default_merge_rank(data_type: TardisDataType | str) -> int:
    dtype = require_binance_futures_data_type(data_type)
    if dtype not in BINANCE_FUTURES_DEFAULT_MERGE_RANKS:
        raise ValueError(f"no default merge rank for data_type: {dtype.value}")
    return BINANCE_FUTURES_DEFAULT_MERGE_RANKS[dtype]


def binance_futures_trade_side_code(side: str | None) -> int:
    if side is None:
        return SIDE_UNKNOWN
    if not isinstance(side, str):
        raise ValueError("side must be str or None")
    normalized = side.strip().lower()
    if normalized in BINANCE_FUTURES_TRADE_SIDE_TO_CODE:
        return BINANCE_FUTURES_TRADE_SIDE_TO_CODE[normalized]
    raise ValueError(f"unsupported trade side: {side!r}")


def binance_futures_book_side_code(side: str | None) -> int:
    if side is None:
        return BOOK_SIDE_UNKNOWN
    if not isinstance(side, str):
        raise ValueError("side must be str or None")
    normalized = side.strip().lower()
    if normalized in BINANCE_FUTURES_BOOK_SIDE_TO_CODE:
        return BINANCE_FUTURES_BOOK_SIDE_TO_CODE[normalized]
    raise ValueError(f"unsupported book side: {side!r}")


def _validate_iso_date(date: str) -> str:
    normalized = _require_nonempty_str(date, "date")
    if len(normalized) != 10 or normalized[4] != "-" or normalized[7] != "-":
        raise ValueError("date must match YYYY-MM-DD")
    if not (normalized[:4].isdigit() and normalized[5:7].isdigit() and normalized[8:10].isdigit()):
        raise ValueError("date must match YYYY-MM-DD")
    return normalized


def normalized_parquet_basename(
    exchange: str,
    symbol: str,
    data_type: TardisDataType | str,
    date: str,
) -> str:
    normalized_exchange = normalize_binance_futures_exchange(exchange)
    normalized_symbol = normalize_binance_futures_symbol(symbol)
    dtype = require_binance_futures_data_type(data_type)
    normalized_date = _validate_iso_date(date)
    return f"{normalized_exchange}_{normalized_symbol}_{dtype.value}_{normalized_date}.parquet"


def merged_parquet_basename(
    exchange: str,
    symbol: str,
    date: str,
) -> str:
    normalized_exchange = normalize_binance_futures_exchange(exchange)
    normalized_symbol = normalize_binance_futures_symbol(symbol)
    normalized_date = _validate_iso_date(date)
    return f"{normalized_exchange}_{normalized_symbol}_merged_events_{normalized_date}.parquet"


__all__ = [
    "BINANCE_FUTURES_EXCHANGE",
    "BINANCE_FUTURES_SYMBOL",
    "BINANCE_FUTURES_SYMBOLS",
    "BINANCE_FUTURES_SOURCE_DATA_TYPES",
    "BINANCE_FUTURES_CONTEXT_DATA_TYPES",
    "BINANCE_FUTURES_UNSUPPORTED_DATA_TYPES",
    "BINANCE_FUTURES_ACCEPTED_DATA_TYPES",
    "BINANCE_FUTURES_DEFAULT_MERGE_RANKS",
    "SIDE_UNKNOWN",
    "SIDE_BUY",
    "SIDE_SELL",
    "BOOK_SIDE_UNKNOWN",
    "BOOK_SIDE_BID",
    "BOOK_SIDE_ASK",
    "BINANCE_FUTURES_TRADE_SIDE_TO_CODE",
    "BINANCE_FUTURES_BOOK_SIDE_TO_CODE",
    "BinanceFuturesMarket",
    "normalize_binance_futures_exchange",
    "normalize_binance_futures_symbol",
    "validate_binance_futures_market",
    "is_binance_futures_source_data_type",
    "is_binance_futures_context_data_type",
    "is_binance_futures_accepted_data_type",
    "require_binance_futures_data_type",
    "normalize_binance_futures_data_types",
    "default_binance_futures_source_data_types",
    "default_binance_futures_context_data_types",
    "default_binance_futures_accepted_data_types",
    "binance_futures_default_merge_rank",
    "binance_futures_trade_side_code",
    "binance_futures_book_side_code",
    "normalized_parquet_basename",
    "merged_parquet_basename",
]
