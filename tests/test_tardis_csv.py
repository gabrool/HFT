import gzip

import polars as pl
import pytest

from mmrt.contracts import TardisDataType
from mmrt.data.tardis_csv import (
    BOOK_SIDE_ASK,
    BOOK_SIDE_BID,
    LOCAL_TS_US,
    RAW_SOURCE_ROW,
    SIDE_BUY,
    SIDE_SELL,
    SIDE_UNKNOWN,
    SOURCE_DATA_TYPE,
    SOURCE_FILE,
    TS_US,
    expected_normalized_columns,
    normalized_column_renames,
    normalized_snapshot_column_name,
    polars_schema_overrides,
    read_tardis_csv_header,
    scan_tardis_csv_normalized,
    validate_normalized_timestamps,
    validate_tardis_csv_header,
    write_normalized_parquet,
)
from mmrt.schemas import BOOK_SNAPSHOT_25_SCHEMA, TRADES_SCHEMA


def test_data_init_has_no_import_side_effects():
    import mmrt.data

    assert mmrt.data.__all__ == []


def test_read_tardis_csv_header_plain(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,1599868800280000,1599868800280001,t1,buy,100.0,0.5\n",
        encoding="utf-8",
    )
    assert read_tardis_csv_header(p) == TRADES_SCHEMA.column_names


def test_read_tardis_csv_header_gzip(tmp_path):
    p = tmp_path / "trades.csv.gz"
    with gzip.open(p, "wt", newline="") as f:
        f.write(
            "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
            "binance-futures,BTCUSDT,1599868800280000,1599868800280001,t1,buy,100.0,0.5\n"
        )
    assert read_tardis_csv_header(p) == TRADES_SCHEMA.column_names


def test_validate_tardis_csv_header_accepts_valid_trades(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,1,2,t1,buy,100.0,0.5\n",
        encoding="utf-8",
    )
    schema = validate_tardis_csv_header(p, TardisDataType.TRADES)
    assert schema.data_type == TardisDataType.TRADES


def test_validate_tardis_csv_header_rejects_wrong_order(tmp_path):
    p = tmp_path / "trades_bad.csv"
    p.write_text(
        "exchange,symbol,local_timestamp,timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,2,1,t1,buy,100.0,0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_tardis_csv_header(p, TardisDataType.TRADES)


def test_polars_schema_overrides_trades():
    schema = polars_schema_overrides(TRADES_SCHEMA)
    assert schema["timestamp"] == pl.Int64
    assert schema["local_timestamp"] == pl.Int64
    assert schema["price"] == pl.Float64
    assert schema["amount"] == pl.Float64
    assert schema["exchange"] == pl.Utf8
    assert schema["symbol"] == pl.Utf8
    assert schema["id"] == pl.Utf8
    assert schema["side"] == pl.Utf8


def test_normalized_snapshot_column_name():
    assert normalized_snapshot_column_name("asks[0].price") == "ask_px_00"
    assert normalized_snapshot_column_name("asks[0].amount") == "ask_sz_00"
    assert normalized_snapshot_column_name("bids[24].price") == "bid_px_24"
    assert normalized_snapshot_column_name("bids[24].amount") == "bid_sz_24"
    with pytest.raises(ValueError):
        normalized_snapshot_column_name("not_snapshot")


def test_normalized_column_renames_snapshot():
    renames = normalized_column_renames(BOOK_SNAPSHOT_25_SCHEMA)
    assert renames["timestamp"] == TS_US
    assert renames["local_timestamp"] == LOCAL_TS_US
    assert renames["asks[0].price"] == "ask_px_00"
    assert renames["bids[24].amount"] == "bid_sz_24"


def test_scan_tardis_csv_normalized_trades(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,300,300,t1,buy,100.0,0.5\n"
        "binance-futures,BTCUSDT,100,100,t2,sell,101.0,0.6\n"
        "binance-futures,BTCUSDT,200,200,t3,unknown,102.0,0.7\n"
        "binance-futures,BTCUSDT,400,400,t4,,103.0,0.8\n",
        encoding="utf-8",
    )
    df = scan_tardis_csv_normalized(p, TardisDataType.TRADES).collect()
    assert df.columns == list(expected_normalized_columns(TardisDataType.TRADES))
    assert df[RAW_SOURCE_ROW].to_list() == [0, 1, 2, 3]
    assert df[TS_US].dtype == pl.Int64
    assert df[LOCAL_TS_US].dtype == pl.Int64
    assert df[SOURCE_FILE].to_list() == [str(p)] * 4
    assert df[SOURCE_DATA_TYPE].to_list() == ["trades"] * 4
    assert df["side_code"].to_list() == [SIDE_BUY, SIDE_SELL, SIDE_UNKNOWN, SIDE_UNKNOWN]
    assert "side" in df.columns
    assert df[TS_US].to_list() == [300, 100, 200, 400]


def test_scan_tardis_csv_normalized_book_snapshot_25(tmp_path):
    p = tmp_path / "book_snapshot_25.csv"
    header = list(BOOK_SNAPSHOT_25_SCHEMA.column_names)
    row = ["binance-futures", "BTCUSDT", "1", "2"]
    for i in range(25):
        row.extend([str(1000 + i), str(1 + i), str(999 - i), str(2 + i)])
    p.write_text(",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf-8")

    df = scan_tardis_csv_normalized(p, TardisDataType.BOOK_SNAPSHOT_25).collect()
    assert df.columns == list(expected_normalized_columns(TardisDataType.BOOK_SNAPSHOT_25))
    for c in ("ask_px_00", "ask_sz_00", "bid_px_00", "bid_sz_00", "ask_px_24", "bid_sz_24"):
        assert c in df.columns
    assert "asks[0].price" not in df.columns
    assert df[RAW_SOURCE_ROW].to_list() == [0]
    assert "mid" not in df.columns
    assert "spread_bps" not in df.columns
    assert "microprice" not in df.columns


def test_scan_tardis_csv_normalized_incremental_book_l2(tmp_path):
    p = tmp_path / "inc.csv"
    p.write_text(
        "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount\n"
        "binance-futures,BTCUSDT,1,2,false,bid,100.0,1.0\n"
        "binance-futures,BTCUSDT,3,4,false,ask,101.0,2.0\n",
        encoding="utf-8",
    )
    df = scan_tardis_csv_normalized(p, TardisDataType.INCREMENTAL_BOOK_L2).collect()
    assert df.columns == list(expected_normalized_columns(TardisDataType.INCREMENTAL_BOOK_L2))
    assert df["book_side_code"].to_list() == [BOOK_SIDE_BID, BOOK_SIDE_ASK]
    assert "side" in df.columns


def test_scan_tardis_csv_normalized_liquidations(tmp_path):
    p = tmp_path / "liq.csv"
    p.write_text(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,1,2,l1,buy,100.0,1.0\n"
        "binance-futures,BTCUSDT,3,4,l2,sell,99.0,2.0\n",
        encoding="utf-8",
    )
    df = scan_tardis_csv_normalized(p, TardisDataType.LIQUIDATIONS).collect()
    assert "side_code" in df.columns
    assert df["side_code"].to_list() == [SIDE_BUY, SIDE_SELL]


def test_validate_normalized_timestamps():
    df = pl.DataFrame({TS_US: [1, 2], LOCAL_TS_US: [3, 4]})
    validate_normalized_timestamps(df)
    with pytest.raises(ValueError):
        validate_normalized_timestamps(pl.DataFrame({TS_US: [0], LOCAL_TS_US: [1]}))
    with pytest.raises(ValueError):
        validate_normalized_timestamps(pl.DataFrame({LOCAL_TS_US: [1]}))


def test_write_normalized_parquet(tmp_path):
    src = tmp_path / "trades.csv"
    dst = tmp_path / "out" / "trades.parquet"
    src.write_text(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,1,2,t1,buy,100.0,0.5\n",
        encoding="utf-8",
    )
    out = write_normalized_parquet(src, dst, TardisDataType.TRADES)
    assert dst.exists()
    assert out.data_type == TardisDataType.TRADES
    assert out.row_count is None
    assert pl.read_parquet(dst).columns == list(expected_normalized_columns(TardisDataType.TRADES))


def test_expected_normalized_columns():
    tcols = expected_normalized_columns(TardisDataType.TRADES)
    assert RAW_SOURCE_ROW in tcols
    assert TS_US in tcols and LOCAL_TS_US in tcols
    assert "side_code" in tcols
    assert SOURCE_FILE in tcols and SOURCE_DATA_TYPE in tcols

    bcols = expected_normalized_columns(TardisDataType.BOOK_SNAPSHOT_25)
    assert "ask_px_00" in bcols and "bid_sz_24" in bcols
    assert all("asks[" not in c and "bids[" not in c for c in bcols)

    icols = expected_normalized_columns(TardisDataType.INCREMENTAL_BOOK_L2)
    assert "book_side_code" in icols

    forbidden = ("mid", "spread", "microprice", "label", "ta" + "rget", "fu" + "ture")
    assert not any(any(f in c for f in forbidden) for c in tcols + bcols + icols)


def test_unsupported_schemas_raise(tmp_path):
    p = tmp_path / "dummy.csv"
    p.write_text("exchange,symbol,timestamp,local_timestamp\nex,sym,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        scan_tardis_csv_normalized(p, TardisDataType.QUOTES, validate_header=False)
    with pytest.raises(ValueError):
        scan_tardis_csv_normalized(p, TardisDataType.OPTIONS_CHAIN, validate_header=False)


def test_no_sorting_row_order_preserved(tmp_path):
    p = tmp_path / "trades_unsorted.csv"
    p.write_text(
        "exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
        "binance-futures,BTCUSDT,300,300,t1,buy,100.0,0.5\n"
        "binance-futures,BTCUSDT,100,100,t2,sell,101.0,0.6\n"
        "binance-futures,BTCUSDT,200,200,t3,buy,102.0,0.7\n",
        encoding="utf-8",
    )
    df = scan_tardis_csv_normalized(p, TardisDataType.TRADES).collect()
    assert df[RAW_SOURCE_ROW].to_list() == [0, 1, 2]
    assert df[TS_US].to_list() == [300, 100, 200]
