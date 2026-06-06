import inspect
import subprocess
import sys

import pytest

from mmrt.contracts import TardisDataType
from mmrt.data.binance_futures_adapter import (
    BINANCE_FUTURES_BOOK_SIDE_TO_CODE,
    BINANCE_FUTURES_DEFAULT_MERGE_RANKS,
    BINANCE_FUTURES_EXCHANGE,
    BINANCE_FUTURES_SYMBOL,
    BINANCE_FUTURES_TRADE_SIDE_TO_CODE,
    BOOK_SIDE_ASK,
    BOOK_SIDE_BID,
    SIDE_BUY,
    SIDE_SELL,
    SIDE_UNKNOWN,
    BinanceFuturesMarket,
    binance_futures_book_side_code,
    binance_futures_default_merge_rank,
    binance_futures_trade_side_code,
    merged_parquet_basename,
    normalize_binance_futures_exchange,
    normalize_binance_futures_symbol,
    normalized_parquet_basename,
    validate_binance_futures_market,
)


def test_constants_match_tardis_policy():
    assert BINANCE_FUTURES_EXCHANGE == "binance-futures"
    assert BINANCE_FUTURES_SYMBOL == "BTCUSDT"
    assert BINANCE_FUTURES_DEFAULT_MERGE_RANKS == {
        TardisDataType.BOOK_SNAPSHOT_25: 0,
        TardisDataType.INCREMENTAL_BOOK_L2: 1,
        TardisDataType.TRADES: 2,
    }


def test_local_side_code_constants():
    assert SIDE_BUY == 1
    assert SIDE_SELL == -1
    assert SIDE_UNKNOWN == 0
    assert BOOK_SIDE_BID == 1
    assert BOOK_SIDE_ASK == -1


def test_market_dataclass_validation():
    assert BinanceFuturesMarket() == BinanceFuturesMarket("binance-futures", "BTCUSDT")
    market = BinanceFuturesMarket("binance-futures", "btcusdt")
    assert market.symbol == "BTCUSDT"
    with pytest.raises(ValueError):
        BinanceFuturesMarket("binance-delivery", "BTCUSDT")
    with pytest.raises(ValueError):
        BinanceFuturesMarket("binance-futures", "ETHUSDT")
    with pytest.raises(ValueError):
        BinanceFuturesMarket("binance-futures", "")


def test_exchange_normalization_is_exact():
    assert normalize_binance_futures_exchange("binance-futures") == "binance-futures"
    assert normalize_binance_futures_exchange(" binance-futures ") == "binance-futures"
    with pytest.raises(ValueError):
        normalize_binance_futures_exchange("BINANCE-FUTURES")
    with pytest.raises(ValueError):
        normalize_binance_futures_exchange("binance")
    with pytest.raises(ValueError):
        normalize_binance_futures_exchange("binance-delivery")


def test_symbol_normalization_btcusdt_only():
    assert normalize_binance_futures_symbol("BTCUSDT") == "BTCUSDT"
    assert normalize_binance_futures_symbol("btcusdt") == "BTCUSDT"
    assert normalize_binance_futures_symbol(" BTCUSDT ") == "BTCUSDT"
    with pytest.raises(ValueError):
        normalize_binance_futures_symbol("ETHUSDT")
    with pytest.raises(ValueError):
        normalize_binance_futures_symbol("BTCUSD_PERP")
    with pytest.raises(ValueError):
        normalize_binance_futures_symbol("")
    with pytest.raises(ValueError):
        normalize_binance_futures_symbol(None)  # type: ignore[arg-type]


def test_validate_market():
    market = validate_binance_futures_market("binance-futures", "btcusdt")
    assert market == BinanceFuturesMarket("binance-futures", "BTCUSDT")


def test_default_merge_rank():
    assert binance_futures_default_merge_rank(TardisDataType.BOOK_SNAPSHOT_25) == 0
    assert binance_futures_default_merge_rank(TardisDataType.INCREMENTAL_BOOK_L2) == 1
    assert binance_futures_default_merge_rank(TardisDataType.TRADES) == 2
    with pytest.raises(ValueError):
        binance_futures_default_merge_rank("quotes")
    ranks = [binance_futures_default_merge_rank(dtype) for dtype in BINANCE_FUTURES_DEFAULT_MERGE_RANKS]
    assert len(set(ranks)) == len(ranks)


def test_trade_side_code():
    assert binance_futures_trade_side_code("buy") == SIDE_BUY
    assert binance_futures_trade_side_code("BUY") == SIDE_BUY
    assert binance_futures_trade_side_code(" sell ") == SIDE_SELL
    assert binance_futures_trade_side_code("unknown") == SIDE_UNKNOWN
    with pytest.raises(ValueError):
        binance_futures_trade_side_code("")
    with pytest.raises(ValueError):
        binance_futures_trade_side_code(None)
    with pytest.raises(ValueError):
        binance_futures_trade_side_code("bid")
    with pytest.raises(ValueError):
        binance_futures_trade_side_code("foo")


def test_book_side_code():
    assert binance_futures_book_side_code("bid") == BOOK_SIDE_BID
    assert binance_futures_book_side_code("BID") == BOOK_SIDE_BID
    assert binance_futures_book_side_code(" ask ") == BOOK_SIDE_ASK
    with pytest.raises(ValueError):
        binance_futures_book_side_code("")
    with pytest.raises(ValueError):
        binance_futures_book_side_code(None)
    with pytest.raises(ValueError):
        binance_futures_book_side_code("buy")
    with pytest.raises(ValueError):
        binance_futures_book_side_code("foo")


def test_normalized_parquet_basename():
    assert (
        normalized_parquet_basename("binance-futures", "btcusdt", "trades", "2026-02-22")
        == "binance-futures_BTCUSDT_trades_2026-02-22.parquet"
    )
    with pytest.raises(ValueError):
        normalized_parquet_basename("binance-delivery", "btcusdt", "trades", "2026-02-22")
    with pytest.raises(ValueError):
        normalized_parquet_basename("binance-futures", "ETHUSDT", "trades", "2026-02-22")
    with pytest.raises(ValueError):
        normalized_parquet_basename("binance-futures", "btcusdt", "quotes", "2026-02-22")
    for bad_date in ("", "20260222", "2026-2-22", "2026-02-2x"):
        with pytest.raises(ValueError):
            normalized_parquet_basename("binance-futures", "btcusdt", "trades", bad_date)


def test_merged_parquet_basename():
    assert (
        merged_parquet_basename("binance-futures", "btcusdt", "2026-02-22")
        == "binance-futures_BTCUSDT_merged_events_2026-02-22.parquet"
    )
    with pytest.raises(ValueError):
        merged_parquet_basename("binance-delivery", "btcusdt", "2026-02-22")
    with pytest.raises(ValueError):
        merged_parquet_basename("binance-futures", "ETHUSDT", "2026-02-22")
    with pytest.raises(ValueError):
        merged_parquet_basename("binance-futures", "btcusdt", "2026-2-22")


def test_adapter_import_smoke():
    import mmrt.data.binance_futures_adapter as adapter

    assert adapter.BINANCE_FUTURES_EXCHANGE == "binance-futures"


def test_adapter_does_not_import_heavy_data_modules():
    code = r'''
import sys

before = set(sys.modules)
import mmrt.data.binance_futures_adapter as adapter
after = set(sys.modules)
new = after - before

assert adapter.BINANCE_FUTURES_EXCHANGE == "binance-futures"

forbidden = {
    "polars",
    "pyarrow",
    "pandas",
    "sklearn",
    "scipy",
    "torch",
    "numba",
    "mmrt.data.tardis_csv",
    "mmrt.data.event_merge",
    "mmrt.storage",
    "mmrt.linear",
}

bad = sorted(
    mod for mod in new
    if any(mod == banned or mod.startswith(banned + ".") for banned in forbidden)
)
if bad:
    raise SystemExit("forbidden imports loaded: " + repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_feature_label_or_decision_concepts():
    import mmrt.data.binance_futures_adapter as a

    for name in a.__all__:
        lowered = name.lower()
        assert "feature" not in lowered
        assert "label" not in lowered
        assert "tar" + "get" not in lowered
        assert "decision" not in lowered
        assert "cmssl" not in lowered
        assert "bybit" not in lowered

    assert BINANCE_FUTURES_TRADE_SIDE_TO_CODE["buy"] == SIDE_BUY
    assert BINANCE_FUTURES_BOOK_SIDE_TO_CODE["bid"] == BOOK_SIDE_BID
    assert BINANCE_FUTURES_DEFAULT_MERGE_RANKS[TardisDataType.TRADES] == 2


def test_adapter_has_no_source_context_accepted_helper_bloat():
    import mmrt.data.binance_futures_adapter as adapter

    src = inspect.getsource(adapter)
    forbidden = (
        "SOURCE" + "_DATA_TYPES",
        "CONTEXT" + "_DATA_TYPES",
        "ACCEPTED" + "_DATA_TYPES",
        "is_binance_futures_source" + "_data_type",
        "is_binance_futures_context" + "_data_type",
        "is_binance_futures_accepted" + "_data_type",
        "default_binance_futures_source" + "_data_types",
        "default_binance_futures_context" + "_data_types",
        "default_binance_futures_accepted" + "_data_types",
        "normalize_binance_futures_data" + "_types",
        "require_binance_futures_data" + "_type",
    )
    for token in forbidden:
        assert token not in src
