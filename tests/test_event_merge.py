from pathlib import Path

import polars as pl
import pytest

from mmrt.contracts import EventType, TardisDataType
from mmrt.data.event_merge import (
    EVENT_SEQ,
    EVENT_TYPE,
    EVENT_TYPE_CODE,
    EVENT_TYPE_CODE_BOOK_DELTA,
    EVENT_TYPE_CODE_BOOK_SNAPSHOT,
    EVENT_TYPE_CODE_TRADE,
    MERGE_INPUT_RANK,
    ParquetEventStreamInput,
    event_type_code_for_data_type,
    event_type_for_data_type,
    expected_merged_columns,
    iter_merged_events_streaming,
    parquet_event_stream_input,
    validate_merge_input_schema,
    validate_merged_event_frame,
)
from mmrt.data.tardis_csv import LOCAL_TS_US, RAW_SOURCE_ROW, SOURCE_DATA_TYPE, SOURCE_FILE, TS_US, expected_normalized_columns


def _base_row(dt: TardisDataType) -> dict:
    row = {c: None for c in expected_normalized_columns(dt)}
    row.update({RAW_SOURCE_ROW: 0, "exchange": "binance-futures", "symbol": "BTCUSDT", TS_US: 100, LOCAL_TS_US: 110, SOURCE_FILE: f"{dt.value}.parquet", SOURCE_DATA_TYPE: dt.value})
    return row


def _trades_df(local_ts_values, ts_values=None, raw_rows=None):
    rows = []
    for i, lts in enumerate(local_ts_values):
        row = _base_row(TardisDataType.TRADES)
        row[LOCAL_TS_US] = lts
        row[TS_US] = ts_values[i] if ts_values is not None else 100 + i
        row[RAW_SOURCE_ROW] = raw_rows[i] if raw_rows is not None else i
        row["id"] = f"t{i}"
        row["side"] = "buy"
        row["price"] = 10000.0 + i
        row["amount"] = 1.0
        row["side_code"] = 1
        rows.append(row)
    return pl.DataFrame(rows).select(list(expected_normalized_columns(TardisDataType.TRADES)))


def _snapshot_df(local_ts_values, ts_values=None, raw_rows=None):
    rows = []
    cols = expected_normalized_columns(TardisDataType.BOOK_SNAPSHOT_25)
    ask_cols = [c for c in cols if c.startswith("ask_px_") or c.startswith("ask_sz_")]
    bid_cols = [c for c in cols if c.startswith("bid_px_") or c.startswith("bid_sz_")]
    for i, lts in enumerate(local_ts_values):
        row = _base_row(TardisDataType.BOOK_SNAPSHOT_25)
        row[LOCAL_TS_US] = lts
        row[TS_US] = ts_values[i] if ts_values is not None else 200 + i
        row[RAW_SOURCE_ROW] = raw_rows[i] if raw_rows is not None else i
        for c in ask_cols:
            row[c] = 10100.0 + i
        for c in bid_cols:
            row[c] = 10000.0 + i
        rows.append(row)
    return pl.DataFrame(rows).select(list(cols))


def _incremental_df(local_ts_values, ts_values=None, raw_rows=None):
    rows = []
    for i, lts in enumerate(local_ts_values):
        row = _base_row(TardisDataType.INCREMENTAL_BOOK_L2)
        row[LOCAL_TS_US] = lts
        row[TS_US] = ts_values[i] if ts_values is not None else 300 + i
        row[RAW_SOURCE_ROW] = raw_rows[i] if raw_rows is not None else i
        row["side"] = "bid"
        row["price"] = 9999.0
        row["amount"] = 2.0
        row["book_side_code"] = 1
        rows.append(row)
    return pl.DataFrame(rows).select(list(expected_normalized_columns(TardisDataType.INCREMENTAL_BOOK_L2)))


def _write_norm_parquet(tmp_path: Path, name: str, df: pl.DataFrame) -> Path:
    path = tmp_path / name
    df.write_parquet(path)
    return path


def _stream_df(inputs):
    rows = list(iter_merged_events_streaming(inputs, batch_size=2))
    return pl.DataFrame(rows).select(list(expected_merged_columns([inp.data_type for inp in inputs])))


def test_event_merge_has_no_merged_parquet_writer_api():
    import inspect
    import mmrt.data.event_merge as em

    src = inspect.getsource(em)

    forbidden = [
        "MergedEventFile",
        "write_merged_events_parquet",
        "ParquetWriter",
        "DEFAULT_PARQUET_COMPRESSION",
        "merged/events.parquet",
        "merge_normalized_events",
    ]
    for token in forbidden:
        assert token not in src

    assert "iter_merged_events_streaming" in src


def test_event_type_mappings():
    assert event_type_for_data_type(TardisDataType.BOOK_SNAPSHOT_25) == EventType.BOOK_SNAPSHOT
    assert event_type_for_data_type(TardisDataType.BOOK_SNAPSHOT_5) == EventType.BOOK_SNAPSHOT
    assert event_type_for_data_type(TardisDataType.INCREMENTAL_BOOK_L2) == EventType.BOOK_DELTA
    assert event_type_for_data_type(TardisDataType.TRADES) == EventType.TRADE
    assert event_type_for_data_type(TardisDataType.BOOK_TICKER) == EventType.BOOK_TICKER
    assert event_type_for_data_type(TardisDataType.DERIVATIVE_TICKER) == EventType.DERIVATIVE_TICKER
    assert event_type_for_data_type(TardisDataType.LIQUIDATIONS) == EventType.LIQUIDATION
    with pytest.raises(ValueError):
        event_type_for_data_type(TardisDataType.QUOTES)
    with pytest.raises(ValueError):
        event_type_for_data_type(TardisDataType.OPTIONS_CHAIN)
    assert event_type_code_for_data_type(TardisDataType.BOOK_SNAPSHOT_25) == EVENT_TYPE_CODE_BOOK_SNAPSHOT
    assert event_type_code_for_data_type(TardisDataType.INCREMENTAL_BOOK_L2) == EVENT_TYPE_CODE_BOOK_DELTA
    assert event_type_code_for_data_type(TardisDataType.TRADES) == EVENT_TYPE_CODE_TRADE


def test_parquet_event_stream_input_validation(tmp_path: Path):
    path = _write_norm_parquet(tmp_path, "trades.parquet", _trades_df([10]))
    inp = parquet_event_stream_input(path, TardisDataType.TRADES, 0)
    assert inp.data_type == TardisDataType.TRADES
    assert inp.path == path
    assert inp.input_rank == 0
    assert inp.source_name == str(path)
    with pytest.raises(ValueError):
        ParquetEventStreamInput(TardisDataType.TRADES, path, True, "x")
    with pytest.raises(ValueError):
        ParquetEventStreamInput(TardisDataType.TRADES, path, -1, "x")
    with pytest.raises(ValueError):
        ParquetEventStreamInput(TardisDataType.TRADES, tmp_path / "missing.parquet", 0, "x")
    with pytest.raises(ValueError):
        ParquetEventStreamInput(TardisDataType.TRADES, path, 0, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parquet_event_stream_input(path, TardisDataType.QUOTES, 0)
def test_validate_merge_input_schema_accepts_exact_schema(tmp_path: Path):
    inp = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t.parquet", _trades_df([1])), TardisDataType.TRADES, 0)
    validate_merge_input_schema(inp)


def test_validate_merge_input_schema_rejects_mismatch(tmp_path: Path):
    path = _write_norm_parquet(tmp_path, "bad.parquet", _trades_df([1]).drop("amount"))
    inp = parquet_event_stream_input(path, TardisDataType.TRADES, 0)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_merge_input_schema(inp)


def test_streaming_merge_sorts_by_local_ts_not_ts_us(tmp_path: Path):
    t = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t.parquet", _trades_df([20, 40], ts_values=[1000, 10])), TardisDataType.TRADES, 1)
    s = parquet_event_stream_input(_write_norm_parquet(tmp_path, "s.parquet", _snapshot_df([10, 30], ts_values=[9999, 1])), TardisDataType.BOOK_SNAPSHOT_25, 0)
    df = _stream_df([t, s])
    assert df[LOCAL_TS_US].to_list() == [10, 20, 30, 40]
    assert df[TS_US].to_list() == [9999, 1000, 1, 10]


def test_ties_break_by_input_rank_then_raw_source_row(tmp_path: Path):
    high = parquet_event_stream_input(_write_norm_parquet(tmp_path, "high.parquet", _trades_df([10, 10], raw_rows=[0, 1])), TardisDataType.TRADES, 10)
    low = parquet_event_stream_input(_write_norm_parquet(tmp_path, "low.parquet", _snapshot_df([10, 10], raw_rows=[2, 5])), TardisDataType.BOOK_SNAPSHOT_25, 1)
    df = _stream_df([high, low])
    assert df.select([MERGE_INPUT_RANK, RAW_SOURCE_ROW]).rows() == [(1, 2), (1, 5), (10, 0), (10, 1)]


def test_event_seq_assigned_after_final_merge(tmp_path: Path):
    t = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t.parquet", _trades_df([20, 40])), TardisDataType.TRADES, 1)
    s = parquet_event_stream_input(_write_norm_parquet(tmp_path, "s.parquet", _snapshot_df([10, 30])), TardisDataType.BOOK_SNAPSHOT_25, 0)
    df = _stream_df([t, s])
    assert df[EVENT_SEQ].to_list() == [0, 1, 2, 3]


def test_event_type_and_code_added_correctly(tmp_path: Path):
    t = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t.parquet", _trades_df([20])), TardisDataType.TRADES, 1)
    s = parquet_event_stream_input(_write_norm_parquet(tmp_path, "s.parquet", _snapshot_df([10])), TardisDataType.BOOK_SNAPSHOT_25, 0)
    df = _stream_df([t, s])
    assert df[EVENT_TYPE].to_list() == [EventType.BOOK_SNAPSHOT.value, EventType.TRADE.value]
    assert df[EVENT_TYPE_CODE].to_list() == [EVENT_TYPE_CODE_BOOK_SNAPSHOT, EVENT_TYPE_CODE_TRADE]


def test_output_columns_match_expected_and_validate(tmp_path: Path):
    inputs = [
        parquet_event_stream_input(_write_norm_parquet(tmp_path, "t.parquet", _trades_df([20])), TardisDataType.TRADES, 1),
        parquet_event_stream_input(_write_norm_parquet(tmp_path, "s.parquet", _snapshot_df([10])), TardisDataType.BOOK_SNAPSHOT_25, 0),
    ]
    df = _stream_df(inputs)
    assert df.columns == list(expected_merged_columns([TardisDataType.TRADES, TardisDataType.BOOK_SNAPSHOT_25]))
    validate_merged_event_frame(df, [inp.data_type for inp in inputs])


def test_empty_inputs_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        list(iter_merged_events_streaming([]))


def test_duplicate_ranks_rejected(tmp_path: Path):
    t1 = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t1.parquet", _trades_df([1])), TardisDataType.TRADES, 1)
    t2 = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t2.parquet", _trades_df([2])), TardisDataType.TRADES, 1)
    with pytest.raises(ValueError, match="unique"):
        list(iter_merged_events_streaming([t1, t2]))


def test_rows_are_not_deduplicated(tmp_path: Path):
    df = _trades_df([10, 10], ts_values=[1, 1], raw_rows=[0, 0])
    inp = parquet_event_stream_input(_write_norm_parquet(tmp_path, "dupes.parquet", df), TardisDataType.TRADES, 0)
    out = _stream_df([inp])
    assert out.height == 2


def test_source_file_and_source_data_type_preserved(tmp_path: Path):
    df = _trades_df([10])
    df = df.with_columns(pl.lit("raw.csv").alias(SOURCE_FILE), pl.lit("trades").alias(SOURCE_DATA_TYPE))
    inp = parquet_event_stream_input(_write_norm_parquet(tmp_path, "src.parquet", df), TardisDataType.TRADES, 0)
    out = _stream_df([inp])
    assert out[SOURCE_FILE].to_list() == ["raw.csv"]
    assert out[SOURCE_DATA_TYPE].to_list() == ["trades"]


def test_missing_payload_columns_are_filled_with_none(tmp_path: Path):
    inp = parquet_event_stream_input(_write_norm_parquet(tmp_path, "t.parquet", _trades_df([10])), TardisDataType.TRADES, 0)
    out = _stream_df([inp])
    assert "bid_px_00" not in out.columns
    both = _stream_df([inp, parquet_event_stream_input(_write_norm_parquet(tmp_path, "s.parquet", _snapshot_df([20])), TardisDataType.BOOK_SNAPSHOT_25, 1)])
    trade_row = both.filter(pl.col(EVENT_TYPE_CODE) == EVENT_TYPE_CODE_TRADE).row(0, named=True)
    assert trade_row["bid_px_00"] is None


def test_unsorted_individual_input_parquet_raises(tmp_path: Path):
    inp = parquet_event_stream_input(_write_norm_parquet(tmp_path, "bad.parquet", _trades_df([20, 10], raw_rows=[0, 1])), TardisDataType.TRADES, 0)
    with pytest.raises(ValueError, match="streaming merge input is not sorted"):
        list(iter_merged_events_streaming([inp], batch_size=1))


def test_reference_equivalence_to_small_global_sort(tmp_path: Path):
    frames = [
        (TardisDataType.TRADES, 2, _trades_df([10, 30], raw_rows=[0, 1])),
        (TardisDataType.BOOK_SNAPSHOT_25, 1, _snapshot_df([10, 20], raw_rows=[5, 0])),
        (TardisDataType.TRADES, 3, _trades_df([20, 20], raw_rows=[0, 1])),
    ]
    inputs = [parquet_event_stream_input(_write_norm_parquet(tmp_path, f"{i}.parquet", df), dt, rank) for i, (dt, rank, df) in enumerate(frames)]
    streaming = _stream_df(inputs)

    prepared = []
    for dt, rank, df in frames:
        et = event_type_for_data_type(dt)
        prepared.append(df.with_columns(pl.lit(rank).alias(MERGE_INPUT_RANK), pl.lit(et.value).alias(EVENT_TYPE), pl.lit(event_type_code_for_data_type(dt)).alias(EVENT_TYPE_CODE)).select([EVENT_TYPE_CODE, EVENT_TYPE, MERGE_INPUT_RANK, *expected_normalized_columns(dt)]))
    reference = (
        pl.concat(prepared, how="diagonal_relaxed")
        .sort([LOCAL_TS_US, MERGE_INPUT_RANK, RAW_SOURCE_ROW], maintain_order=True)
        .with_row_index(EVENT_SEQ)
        .with_columns(pl.col(EVENT_SEQ).cast(pl.Int64))
        .select(list(expected_merged_columns([dt for dt, _, _ in frames])))
    )
    assert streaming.to_dicts() == reference.to_dicts()
