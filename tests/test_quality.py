from pathlib import Path

import polars as pl
import pytest

from mmrt.contracts import TardisDataType
from mmrt.data.quality import (
    QualityIssue,
    QualityReport,
    QualitySeverity,
    analyze_normalized_lazyframe,
    analyze_normalized_parquet,
    raise_on_quality_errors,
    scan_normalized_parquet,
)
from mmrt.data.tardis_csv import (
    LOCAL_TS_US,
    RAW_SOURCE_ROW,
    SOURCE_DATA_TYPE,
    SOURCE_FILE,
    TS_US,
    expected_normalized_columns,
)


def _base_row(dt: TardisDataType) -> dict:
    row = {c: None for c in expected_normalized_columns(dt)}
    row.update({RAW_SOURCE_ROW: 0, "exchange": "binance", "symbol": "BTCUSDT", TS_US: 100, LOCAL_TS_US: 110, SOURCE_FILE: "file.csv", SOURCE_DATA_TYPE: dt.value})
    return row


def _good_trades_df() -> pl.DataFrame:
    r0 = _base_row(TardisDataType.TRADES)
    r0.update({"id": "1", "side": "buy", "price": 10.0, "amount": 2.0, "side_code": 1})
    r1 = dict(r0)
    r1.update({RAW_SOURCE_ROW: 1, TS_US: 200, LOCAL_TS_US: 210, "id": "2", "side": "sell", "side_code": -1})
    return pl.DataFrame([r0, r1]).select(list(expected_normalized_columns(TardisDataType.TRADES)))


def _good_liquidations_df() -> pl.DataFrame:
    r0 = _base_row(TardisDataType.LIQUIDATIONS)
    r0.update({"id": "1", "side": "buy", "price": 10.0, "amount": 2.0, "side_code": 1})
    r1 = dict(r0)
    r1.update({RAW_SOURCE_ROW: 1, TS_US: 200, LOCAL_TS_US: 210, "id": "2", "side": "sell", "side_code": -1})
    return pl.DataFrame([r0, r1]).select(list(expected_normalized_columns(TardisDataType.LIQUIDATIONS)))


def _good_incremental_df() -> pl.DataFrame:
    r0 = _base_row(TardisDataType.INCREMENTAL_BOOK_L2)
    r0.update({"is_snapshot": True, "side": "bid", "price": 10.0, "amount": 1.0, "book_side_code": 1})
    r1 = dict(r0)
    r1.update({RAW_SOURCE_ROW: 1, TS_US: 101, LOCAL_TS_US: 111, "is_snapshot": False, "side": "ask", "book_side_code": -1})
    return pl.DataFrame([r0, r1]).select(list(expected_normalized_columns(TardisDataType.INCREMENTAL_BOOK_L2)))


def _good_snapshot_df(depth: int) -> pl.DataFrame:
    dt = TardisDataType.BOOK_SNAPSHOT_25 if depth == 25 else TardisDataType.BOOK_SNAPSHOT_5
    row = _base_row(dt)
    for i in range(depth):
        row[f"ask_px_{i:02d}"] = 101.0 + i
        row[f"ask_sz_{i:02d}"] = 1.0
        row[f"bid_px_{i:02d}"] = 99.0 - i
        row[f"bid_sz_{i:02d}"] = 1.0
    return pl.DataFrame([row]).select(list(expected_normalized_columns(dt)))


def _good_book_ticker_df() -> pl.DataFrame:
    row = _base_row(TardisDataType.BOOK_TICKER)
    row.update({"ask_amount": 1.0, "ask_price": 101.0, "bid_amount": 1.0, "bid_price": 100.0})
    return pl.DataFrame([row]).select(list(expected_normalized_columns(TardisDataType.BOOK_TICKER)))


def _good_derivative_ticker_df() -> pl.DataFrame:
    row = _base_row(TardisDataType.DERIVATIVE_TICKER)
    row.update({"funding_timestamp": 1, "funding_rate": -0.1, "predicted_funding_rate": 0.1, "open_interest": 1.0, "last_price": 1.0, "index_price": 1.0, "mark_price": 1.0})
    return pl.DataFrame([row]).select(list(expected_normalized_columns(TardisDataType.DERIVATIVE_TICKER)))


def _find(report: QualityReport, name: str):
    return next((i for i in report.issues if i.name == name), None)


def test_quality_report_dataclass_validation():
    issue = QualityIssue("n", "error", 1, "m")
    assert issue.severity == QualitySeverity.ERROR
    with pytest.raises(ValueError):
        QualityIssue("n", QualitySeverity.ERROR, True, "m")
    with pytest.raises(ValueError):
        QualityIssue("n", QualitySeverity.ERROR, -1, "m")
    report = QualityReport(TardisDataType.TRADES, 1, None, None, None, None, (issue,))
    assert report.has_errors
    with pytest.raises(ValueError):
        QualityReport(TardisDataType.TRADES, True, None, None, None, None, ())


def test_scan_normalized_parquet(tmp_path: Path):
    df = _good_trades_df()
    p = tmp_path / "a.parquet"
    df.write_parquet(p)
    out = scan_normalized_parquet(p).collect()
    assert out.shape[0] == 2


def test_analyze_good_trades():
    r = analyze_normalized_lazyframe(_good_trades_df().lazy(), TardisDataType.TRADES)
    assert r.row_count == 2
    assert r.min_ts_us == 100 and r.max_ts_us == 200
    assert r.min_local_ts_us == 110 and r.max_local_ts_us == 210
    assert not r.has_errors
    assert _find(r, "missing_columns") is None


def test_empty_data_is_warning_not_error():
    empty = _good_trades_df().head(0)
    r = analyze_normalized_lazyframe(empty.lazy(), TardisDataType.TRADES)
    assert r.row_count == 0
    assert _find(r, "empty_data") is not None
    assert not r.has_errors


def test_missing_columns_error():
    df = _good_trades_df().drop("amount")
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "missing_columns") is not None


def test_unexpected_columns_warning():
    df = _good_trades_df().with_columns(pl.lit(1).alias("debug_extra"))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "unexpected_columns").severity == QualitySeverity.WARNING


def test_column_order_mismatch_error():
    df = _good_trades_df()
    cols = list(df.columns)
    i = cols.index("price")
    j = cols.index("amount")
    cols[i], cols[j] = cols[j], cols[i]
    df = df.select(cols)
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    issue = _find(r, "column_order_mismatch")
    assert issue is not None
    assert issue.severity == QualitySeverity.ERROR
    assert issue.count == 1


def test_timestamp_errors():
    df = _good_trades_df().with_columns([pl.lit(0).alias(TS_US), pl.lit(None).cast(pl.Int64).alias(LOCAL_TS_US)])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "nonpositive_ts_us") is not None
    assert _find(r, "null_local_ts_us") is not None


def test_local_ts_decrease_error_but_ts_decrease_allowed():
    df = _good_trades_df().with_columns([pl.Series(TS_US, [300, 100]), pl.Series(LOCAL_TS_US, [100, 90])])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "local_ts_us_decreases") is not None

    df2 = _good_trades_df().with_columns([pl.Series(TS_US, [300, 100]), pl.Series(LOCAL_TS_US, [100, 101])])
    r2 = analyze_normalized_lazyframe(df2.lazy(), TardisDataType.TRADES)
    assert _find(r2, "local_ts_us_decreases") is None


def test_raw_source_row_contiguity_error():
    df = _good_trades_df().with_columns(pl.Series(RAW_SOURCE_ROW, [0, 2]))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "raw_source_row_not_contiguous") is not None


def test_raw_source_row_null_and_negative_errors():
    df = _good_trades_df().with_columns(pl.Series(RAW_SOURCE_ROW, [None, -1], dtype=pl.Int64))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "null_raw_source_row") is not None
    assert _find(r, "negative_raw_source_row") is not None


def test_duplicate_source_row_error():
    df = _good_trades_df().with_columns(pl.Series(RAW_SOURCE_ROW, [0, 0]))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "duplicate_source_rows") is not None


def test_bad_source_data_type_error():
    df = _good_trades_df().with_columns(pl.lit("book_snapshot_25").alias(SOURCE_DATA_TYPE))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "bad_source_data_type") is not None


def test_trades_value_checks():
    df = _good_trades_df().with_columns([pl.lit(0.0).alias("price"), pl.lit(0.0).alias("amount"), pl.lit(99).alias("side_code")])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    assert _find(r, "invalid_price") and _find(r, "invalid_amount") and _find(r, "invalid_side_code")


def test_trades_null_side_code_is_invalid():
    df = _good_trades_df().with_columns(pl.Series("side_code", [1, None], dtype=pl.Int8))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.TRADES)
    issue = _find(r, "invalid_side_code")
    assert issue is not None
    assert issue.count == 1


def test_liquidations_value_checks():
    df = _good_liquidations_df().with_columns([pl.lit(0.0).alias("price"), pl.lit(0.0).alias("amount"), pl.lit(99).alias("side_code")])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.LIQUIDATIONS)
    assert _find(r, "invalid_price") and _find(r, "invalid_amount") and _find(r, "invalid_side_code")


def test_good_liquidations_have_no_source_type_error():
    r = analyze_normalized_lazyframe(_good_liquidations_df().lazy(), TardisDataType.LIQUIDATIONS)
    assert _find(r, "bad_source_data_type") is None
    assert not r.has_errors


def test_incremental_l2_checks_and_snapshot_info():
    df = _good_incremental_df().with_columns([pl.Series("amount", [1.0, -1.0]), pl.Series("book_side_code", [1, 99])])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.INCREMENTAL_BOOK_L2)
    assert _find(r, "invalid_amount") and _find(r, "invalid_book_side_code")
    assert _find(r, "incremental_snapshot_markers").count == 1


def test_incremental_l2_null_book_side_code_is_invalid():
    df = _good_incremental_df().with_columns(pl.Series("book_side_code", [1, None], dtype=pl.Int8))
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.INCREMENTAL_BOOK_L2)
    issue = _find(r, "invalid_book_side_code")
    assert issue is not None
    assert issue.count == 1


def test_snapshot_25_quality_checks():
    df = _good_snapshot_df(25).with_columns([pl.lit(101.0).alias("bid_px_00"), pl.lit(100.0).alias("ask_px_00"), pl.lit(None).cast(pl.Float64).alias("ask_px_01")])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.BOOK_SNAPSHOT_25)
    assert _find(r, "crossed_l1_book") and _find(r, "missing_snapshot_depth_values")
    df2 = _good_snapshot_df(25).with_columns(pl.lit(None).cast(pl.Float64).alias("bid_px_00"))
    r2 = analyze_normalized_lazyframe(df2.lazy(), TardisDataType.BOOK_SNAPSHOT_25)
    assert _find(r2, "missing_l1_snapshot")


def test_snapshot_5_quality_checks():
    df = _good_snapshot_df(5)
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.BOOK_SNAPSHOT_5)
    assert _find(r, "missing_columns") is None


def test_book_ticker_quality_checks():
    df = _good_book_ticker_df().with_columns([pl.lit(100.0).alias("ask_price"), pl.lit(100.0).alias("bid_price"), pl.lit(-1.0).alias("ask_amount")])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.BOOK_TICKER)
    assert _find(r, "crossed_book_ticker") and _find(r, "invalid_book_ticker_amount")


def test_derivative_ticker_quality_checks():
    df = _good_derivative_ticker_df().with_columns([pl.lit(-1.0).alias("open_interest"), pl.lit(0.0).alias("mark_price"), pl.lit(0).alias("funding_timestamp")])
    r = analyze_normalized_lazyframe(df.lazy(), TardisDataType.DERIVATIVE_TICKER)
    assert _find(r, "invalid_open_interest") and _find(r, "invalid_derivative_price") and _find(r, "invalid_funding_timestamp")


def test_raise_on_quality_errors():
    good = QualityReport(TardisDataType.TRADES, 1, 1, 1, 1, 1, ())
    raise_on_quality_errors(good)
    bad = QualityReport(TardisDataType.TRADES, 1, 1, 1, 1, 1, (QualityIssue("x", QualitySeverity.ERROR, 1, "x"),))
    with pytest.raises(ValueError):
        raise_on_quality_errors(bad)


def test_no_feature_or_label_requirements():
    r = analyze_normalized_lazyframe(_good_trades_df().lazy(), TardisDataType.TRADES)
    assert not r.has_errors


def test_analyze_normalized_parquet_round_trip(tmp_path: Path):
    p = tmp_path / "b.parquet"
    _good_trades_df().write_parquet(p)
    r = analyze_normalized_parquet(p, TardisDataType.TRADES)
    assert r.source_path == p
    assert not r.has_errors
