import pytest

from mmrt.contracts import TardisDataType
from mmrt.data.binance_futures_adapter import (
    BINANCE_FUTURES_BOOK_SIDE_TO_CODE,
    BINANCE_FUTURES_DEFAULT_MERGE_RANKS,
    BINANCE_FUTURES_EXCHANGE,
    BINANCE_FUTURES_TRADE_SIDE_TO_CODE,
    BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES,
    BINANCE_FUTURES_V1_CONTEXT_DATA_TYPES,
    BINANCE_FUTURES_V1_SOURCE_DATA_TYPES,
    BINANCE_FUTURES_V1_SYMBOL,
    BOOK_SIDE_ASK,
    BOOK_SIDE_BID,
    BOOK_SIDE_UNKNOWN,
    SIDE_BUY,
    SIDE_SELL,
    SIDE_UNKNOWN,
    BinanceFuturesMarket,
    binance_futures_book_side_code,
    binance_futures_default_merge_rank,
    binance_futures_trade_side_code,
    default_binance_futures_v1_accepted_data_types,
    default_binance_futures_v1_context_data_types,
    default_binance_futures_v1_source_data_types,
    is_binance_futures_v1_accepted_data_type,
    is_binance_futures_v1_context_data_type,
    is_binance_futures_v1_source_data_type,
    merged_parquet_basename,
    normalize_binance_futures_exchange,
    normalize_binance_futures_symbol,
    normalize_binance_futures_v1_data_types,
    normalized_parquet_basename,
    require_binance_futures_v1_data_type,
    validate_binance_futures_market,
)


def test_constants_match_tardis_policy():
    assert BINANCE_FUTURES_EXCHANGE == "binance-futures"
    assert BINANCE_FUTURES_V1_SYMBOL == "BTCUSDT"
    assert BINANCE_FUTURES_V1_SOURCE_DATA_TYPES == (
        TardisDataType.BOOK_SNAPSHOT_25,
        TardisDataType.TRADES,
    )
    assert TardisDataType.QUOTES not in BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES
    assert TardisDataType.OPTIONS_CHAIN not in BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES
    for dtype in (
        TardisDataType.BOOK_TICKER,
        TardisDataType.DERIVATIVE_TICKER,
        TardisDataType.LIQUIDATIONS,
        TardisDataType.INCREMENTAL_BOOK_L2,
        TardisDataType.BOOK_SNAPSHOT_5,
    ):
        assert dtype in BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES


def test_local_side_code_constants():
    assert SIDE_BUY == 1
    assert SIDE_SELL == -1
    assert SIDE_UNKNOWN == 0
    assert BOOK_SIDE_BID == 1
    assert BOOK_SIDE_ASK == -1
    assert BOOK_SIDE_UNKNOWN == 0


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


def test_data_type_support_predicates():
    assert is_binance_futures_v1_source_data_type(TardisDataType.BOOK_SNAPSHOT_25)
    assert is_binance_futures_v1_source_data_type(TardisDataType.TRADES)
    assert not is_binance_futures_v1_source_data_type(TardisDataType.BOOK_TICKER)
    assert is_binance_futures_v1_context_data_type(TardisDataType.BOOK_TICKER)
    assert is_binance_futures_v1_context_data_type(TardisDataType.DERIVATIVE_TICKER)
    assert is_binance_futures_v1_context_data_type(TardisDataType.LIQUIDATIONS)
    assert is_binance_futures_v1_context_data_type(TardisDataType.INCREMENTAL_BOOK_L2)
    assert is_binance_futures_v1_context_data_type(TardisDataType.BOOK_SNAPSHOT_5)
    assert not is_binance_futures_v1_accepted_data_type(TardisDataType.QUOTES)
    assert not is_binance_futures_v1_accepted_data_type(TardisDataType.OPTIONS_CHAIN)


def test_require_data_type():
    assert require_binance_futures_v1_data_type("trades") == TardisDataType.TRADES
    assert require_binance_futures_v1_data_type(TardisDataType.BOOK_SNAPSHOT_25) == TardisDataType.BOOK_SNAPSHOT_25
    with pytest.raises(ValueError):
        require_binance_futures_v1_data_type(TardisDataType.QUOTES)
    with pytest.raises(ValueError):
        require_binance_futures_v1_data_type(TardisDataType.OPTIONS_CHAIN)
    with pytest.raises(ValueError):
        require_binance_futures_v1_data_type("not-a-type")


def test_normalize_data_types_preserves_order_and_rejects_duplicates():
    got = normalize_binance_futures_v1_data_types(["trades", "book_snapshot_25"])
    assert got == (TardisDataType.TRADES, TardisDataType.BOOK_SNAPSHOT_25)
    with pytest.raises(ValueError):
        normalize_binance_futures_v1_data_types(["trades", "trades"])
    with pytest.raises(ValueError):
        normalize_binance_futures_v1_data_types([])
    with pytest.raises(ValueError):
        normalize_binance_futures_v1_data_types(["quotes"])


def test_default_data_type_functions():
    assert isinstance(default_binance_futures_v1_source_data_types(), tuple)
    assert isinstance(default_binance_futures_v1_context_data_types(), tuple)
    assert isinstance(default_binance_futures_v1_accepted_data_types(), tuple)
    assert default_binance_futures_v1_source_data_types() == BINANCE_FUTURES_V1_SOURCE_DATA_TYPES
    assert default_binance_futures_v1_context_data_types() == BINANCE_FUTURES_V1_CONTEXT_DATA_TYPES
    assert default_binance_futures_v1_accepted_data_types() == BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES


def test_default_merge_rank():
    assert binance_futures_default_merge_rank(TardisDataType.BOOK_SNAPSHOT_25) == 0
    assert binance_futures_default_merge_rank(TardisDataType.BOOK_SNAPSHOT_5) == 1
    assert binance_futures_default_merge_rank(TardisDataType.INCREMENTAL_BOOK_L2) == 2
    assert binance_futures_default_merge_rank(TardisDataType.TRADES) == 3
    assert binance_futures_default_merge_rank(TardisDataType.BOOK_TICKER) == 4
    assert binance_futures_default_merge_rank(TardisDataType.LIQUIDATIONS) == 5
    assert binance_futures_default_merge_rank(TardisDataType.DERIVATIVE_TICKER) == 6
    with pytest.raises(ValueError):
        binance_futures_default_merge_rank(TardisDataType.QUOTES)
    with pytest.raises(ValueError):
        binance_futures_default_merge_rank(TardisDataType.OPTIONS_CHAIN)
    for dtype in BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES:
        assert isinstance(binance_futures_default_merge_rank(dtype), int)
    ranks = [binance_futures_default_merge_rank(dtype) for dtype in BINANCE_FUTURES_V1_ACCEPTED_DATA_TYPES]
    assert len(set(ranks)) == len(ranks)


def test_trade_side_code():
    assert binance_futures_trade_side_code("buy") == SIDE_BUY
    assert binance_futures_trade_side_code("BUY") == SIDE_BUY
    assert binance_futures_trade_side_code(" sell ") == SIDE_SELL
    assert binance_futures_trade_side_code("unknown") == SIDE_UNKNOWN
    assert binance_futures_trade_side_code("") == SIDE_UNKNOWN
    assert binance_futures_trade_side_code(None) == SIDE_UNKNOWN
    with pytest.raises(ValueError):
        binance_futures_trade_side_code("bid")
    with pytest.raises(ValueError):
        binance_futures_trade_side_code("foo")


def test_book_side_code():
    assert binance_futures_book_side_code("bid") == BOOK_SIDE_BID
    assert binance_futures_book_side_code("BID") == BOOK_SIDE_BID
    assert binance_futures_book_side_code(" ask ") == BOOK_SIDE_ASK
    assert binance_futures_book_side_code("") == BOOK_SIDE_UNKNOWN
    assert binance_futures_book_side_code(None) == BOOK_SIDE_UNKNOWN
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
        normalized_parquet_basename("binance-futures", "btcusdt", TardisDataType.QUOTES, "2026-02-22")
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
    import sys
    import mmrt.data.binance_futures_adapter as adapter

    assert adapter.BINANCE_FUTURES_EXCHANGE == "binance-futures"
    assert "po" + "lars" not in sys.modules
    assert "mmrt.data.tardis_csv" not in sys.modules
    assert "mmrt.data.event_merge" not in sys.modules


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
    assert BINANCE_FUTURES_DEFAULT_MERGE_RANKS[TardisDataType.TRADES] == 3
