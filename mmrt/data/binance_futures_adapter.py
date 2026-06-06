"""Binance USDS-M Futures adapter policy for the MMRT data layer.

This module contains exchange/symbol/data-type policy for normalized Tardis Binance
USDS-M Futures data. It does not read files, normalize rows, merge events, run quality
checks, reconstruct books, compute features, or compute labels.
"""

from dataclasses import dataclass

from mmrt.contracts import TardisDataType

BINANCE_FUTURES_EXCHANGE = "binance-futures"
BINANCE_FUTURES_SYMBOL = "BTCUSDT"
BINANCE_FUTURES_SYMBOLS = ("BTCUSDT",)

SIDE_UNKNOWN = 0
SIDE_BUY = 1
SIDE_SELL = -1

BOOK_SIDE_BID = 1
BOOK_SIDE_ASK = -1

BINANCE_FUTURES_TRADE_SIDE_TO_CODE = {
    "buy": SIDE_BUY,
    "sell": SIDE_SELL,
    "unknown": SIDE_UNKNOWN,
}

BINANCE_FUTURES_BOOK_SIDE_TO_CODE = {
    "bid": BOOK_SIDE_BID,
    "ask": BOOK_SIDE_ASK,
}

BINANCE_FUTURES_DEFAULT_MERGE_RANKS = {
    TardisDataType.BOOK_SNAPSHOT_25: 0,
    TardisDataType.INCREMENTAL_BOOK_L2: 1,
    TardisDataType.TRADES: 2,
}


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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


def binance_futures_default_merge_rank(data_type: TardisDataType | str) -> int:
    dtype = TardisDataType(data_type)
    try:
        return BINANCE_FUTURES_DEFAULT_MERGE_RANKS[dtype]
    except KeyError as exc:
        raise ValueError(f"no default merge rank for data_type: {dtype.value}") from exc


def binance_futures_trade_side_code(side: str | None) -> int:
    if side is None:
        raise ValueError("side must be str")
    if not isinstance(side, str):
        raise ValueError("side must be str")
    normalized = side.strip().lower()
    try:
        return BINANCE_FUTURES_TRADE_SIDE_TO_CODE[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported trade side: {side!r}") from exc


def binance_futures_book_side_code(side: str | None) -> int:
    if side is None:
        raise ValueError("side must be str")
    if not isinstance(side, str):
        raise ValueError("side must be str")
    normalized = side.strip().lower()
    try:
        return BINANCE_FUTURES_BOOK_SIDE_TO_CODE[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported book side: {side!r}") from exc


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
    dtype = TardisDataType(data_type)
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
    "BINANCE_FUTURES_DEFAULT_MERGE_RANKS",
    "SIDE_UNKNOWN",
    "SIDE_BUY",
    "SIDE_SELL",
    "BOOK_SIDE_BID",
    "BOOK_SIDE_ASK",
    "BINANCE_FUTURES_TRADE_SIDE_TO_CODE",
    "BINANCE_FUTURES_BOOK_SIDE_TO_CODE",
    "BinanceFuturesMarket",
    "normalize_binance_futures_exchange",
    "normalize_binance_futures_symbol",
    "validate_binance_futures_market",
    "binance_futures_default_merge_rank",
    "binance_futures_trade_side_code",
    "binance_futures_book_side_code",
    "normalized_parquet_basename",
    "merged_parquet_basename",
]
