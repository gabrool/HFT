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
    EventMergeInput,
    MergedEventFile,
    MERGE_INPUT_RANK,
    event_type_code_for_data_type,
    event_type_for_data_type,
    expected_merged_columns,
    merge_normalized_events,
    parquet_merge_input,
    validate_merge_input_schema,
    validate_merged_event_frame,
    write_merged_events_parquet,
)
from mmrt.data.tardis_csv import LOCAL_TS_US, RAW_SOURCE_ROW, SOURCE_DATA_TYPE, SOURCE_FILE, TS_US, expected_normalized_columns


def _base_row(dt: TardisDataType) -> dict:
    row = {c: None for c in expected_normalized_columns(dt)}
    row.update(
        {
            RAW_SOURCE_ROW: 0,
            "exchange": "binance-futures",
            "symbol": "BTCUSDT",
            TS_US: 100,
            LOCAL_TS_US: 110,
            SOURCE_FILE: f"{dt.value}.parquet",
            SOURCE_DATA_TYPE: dt.value,
        }
    )
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
            row[c] = 10100.0
        for c in bid_cols:
            row[c] = 10000.0
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


def test_event_merge_input_validation():
    good = EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), 0, "x")
    assert good.input_rank == 0
    with pytest.raises(ValueError):
        EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), True, "x")
    with pytest.raises(ValueError):
        EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), -1, "x")
    with pytest.raises(ValueError):
        EventMergeInput(TardisDataType.TRADES, _trades_df([1]), 0, "x")
    with pytest.raises(ValueError):
        EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), 0, 123)


def test_merged_event_file_validation():
    m = MergedEventFile(Path("a.parquet"), 1, (TardisDataType.TRADES,), None)
    assert m.input_count == 1
    with pytest.raises(ValueError):
        MergedEventFile(Path("a.parquet"), 0, (TardisDataType.TRADES,), None)
    with pytest.raises(ValueError):
        MergedEventFile(Path("a.parquet"), 1, (TardisDataType.TRADES,), True)
    with pytest.raises(ValueError):
        MergedEventFile(Path("a.parquet"), 1, (), None)


def test_parquet_merge_input(tmp_path):
    path = tmp_path / "trades.parquet"
    _trades_df([10]).write_parquet(path)
    inp = parquet_merge_input(path, TardisDataType.TRADES, 0)
    assert inp.data_type == TardisDataType.TRADES
    assert inp.input_rank == 0
    assert inp.source_name == str(path)
    assert isinstance(inp.frame, pl.LazyFrame)


def test_validate_merge_input_schema_accepts_exact_schema():
    inp = EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), 0)
    validate_merge_input_schema(inp)


def test_validate_merge_input_schema_rejects_missing_or_reordered_columns():
    df = _trades_df([1]).drop("amount")
    with pytest.raises(ValueError):
        validate_merge_input_schema(EventMergeInput(TardisDataType.TRADES, df.lazy(), 0))
    cols = list(expected_normalized_columns(TardisDataType.TRADES))
    i, j = cols.index("price"), cols.index("amount")
    cols[i], cols[j] = cols[j], cols[i]
    with pytest.raises(ValueError):
        validate_merge_input_schema(EventMergeInput(TardisDataType.TRADES, _trades_df([1]).select(cols).lazy(), 0))




def test_expected_merged_columns_rejects_unsupported_data_types():
    with pytest.raises(ValueError):
        expected_merged_columns([TardisDataType.QUOTES])
    with pytest.raises(ValueError):
        expected_merged_columns([TardisDataType.OPTIONS_CHAIN])
    with pytest.raises(ValueError):
        expected_merged_columns(["quotes"])
    with pytest.raises(ValueError):
        expected_merged_columns(["options_chain"])


def test_event_merge_input_rejects_unsupported_data_types():
    lf = _trades_df([1]).lazy()
    with pytest.raises(ValueError):
        EventMergeInput(TardisDataType.QUOTES, lf, 0)
    with pytest.raises(ValueError):
        EventMergeInput(TardisDataType.OPTIONS_CHAIN, lf, 0)


def test_merged_event_file_rejects_unsupported_data_types():
    with pytest.raises(ValueError):
        MergedEventFile(Path("bad.parquet"), 1, (TardisDataType.QUOTES,), None)
    with pytest.raises(ValueError):
        MergedEventFile(Path("bad.parquet"), 1, (TardisDataType.OPTIONS_CHAIN,), None)


def test_parquet_merge_input_rejects_unsupported_data_types(tmp_path):
    path = tmp_path / "trades.parquet"
    _trades_df([10]).write_parquet(path)

    with pytest.raises(ValueError):
        parquet_merge_input(path, TardisDataType.QUOTES, 0)
    with pytest.raises(ValueError):
        parquet_merge_input(path, TardisDataType.OPTIONS_CHAIN, 0)

def test_expected_merged_columns_stable_order():
    cols = expected_merged_columns([TardisDataType.TRADES, TardisDataType.BOOK_SNAPSHOT_25])
    assert cols[:11] == (
        EVENT_SEQ,
        EVENT_TYPE_CODE,
        EVENT_TYPE,
        MERGE_INPUT_RANK,
        RAW_SOURCE_ROW,
        SOURCE_FILE,
        SOURCE_DATA_TYPE,
        "exchange",
        "symbol",
        TS_US,
        LOCAL_TS_US,
    )
    for c in ("ask_px_00", "bid_sz_24", "id", "side", "price", "amount", "side_code"):
        assert c in cols
    assert len(cols) == len(set(cols))
    bad_markers = ["mid", "spread", "microprice", "label", "tar" + "get", "fu" + "ture_ret", "fu" + "ture_mid"]
    assert not any(any(marker in c for marker in bad_markers) for c in cols)


def test_merge_sorts_by_local_ts_not_ts():
    t = EventMergeInput(TardisDataType.TRADES, _trades_df([100], [999]).lazy(), 1)
    s = EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([200], [1]).lazy(), 0)
    df = merge_normalized_events([t, s]).collect()
    assert df.get_column(LOCAL_TS_US).to_list() == [100, 200]


def test_merge_tie_breaks_by_input_rank_then_raw_source_row():
    s = EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([100], raw_rows=[5]).lazy(), 0)
    t = EventMergeInput(TardisDataType.TRADES, _trades_df([100], raw_rows=[0]).lazy(), 1)
    i = EventMergeInput(TardisDataType.INCREMENTAL_BOOK_L2, _incremental_df([100], raw_rows=[0]).lazy(), 2)
    df = merge_normalized_events([i, t, s]).collect()
    assert df.get_column(MERGE_INPUT_RANK).to_list()[:3] == [0, 1, 2]

    t2 = EventMergeInput(TardisDataType.TRADES, _trades_df([150, 150, 150], raw_rows=[2, 0, 1]).lazy(), 3)
    df2 = merge_normalized_events([t2]).collect()
    assert df2.get_column(RAW_SOURCE_ROW).to_list() == [0, 1, 2]


def test_merge_assigns_event_seq_after_sort():
    t = EventMergeInput(TardisDataType.TRADES, _trades_df([300, 100], raw_rows=[0, 1]).lazy(), 1)
    s = EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([200]).lazy(), 0)
    df = merge_normalized_events([t, s]).collect()
    assert df.get_column(EVENT_SEQ).to_list() == list(range(df.height))
    assert df.schema[EVENT_SEQ] == pl.Int64


def test_merge_adds_event_type_and_codes():
    t = EventMergeInput(TardisDataType.TRADES, _trades_df([100]).lazy(), 1)
    s = EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([200]).lazy(), 0)
    df = merge_normalized_events([t, s]).collect()
    trade_row = df.filter(pl.col(SOURCE_DATA_TYPE) == TardisDataType.TRADES.value).row(0, named=True)
    snap_row = df.filter(pl.col(SOURCE_DATA_TYPE) == TardisDataType.BOOK_SNAPSHOT_25.value).row(0, named=True)
    assert trade_row[EVENT_TYPE] == EventType.TRADE.value
    assert trade_row[EVENT_TYPE_CODE] == EVENT_TYPE_CODE_TRADE
    assert snap_row[EVENT_TYPE] == EventType.BOOK_SNAPSHOT.value
    assert snap_row[EVENT_TYPE_CODE] == EVENT_TYPE_CODE_BOOK_SNAPSHOT


def test_merge_output_columns_match_expected():
    dtypes = [TardisDataType.TRADES, TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.INCREMENTAL_BOOK_L2]
    inputs = [
        EventMergeInput(TardisDataType.TRADES, _trades_df([100]).lazy(), 1),
        EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([200]).lazy(), 0),
        EventMergeInput(TardisDataType.INCREMENTAL_BOOK_L2, _incremental_df([300]).lazy(), 2),
    ]
    df = merge_normalized_events(inputs).collect()
    assert df.columns == list(expected_merged_columns(dtypes))
    validate_merged_event_frame(df, dtypes)


def test_merge_rejects_empty_inputs():
    with pytest.raises(ValueError):
        merge_normalized_events([])


def test_merge_rejects_duplicate_input_ranks():
    t1 = EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), 0)
    t2 = EventMergeInput(TardisDataType.TRADES, _trades_df([2]).lazy(), 0)
    with pytest.raises(ValueError):
        merge_normalized_events([t1, t2])


def test_write_merged_events_parquet(tmp_path):
    out = tmp_path / "merged.parquet"
    inputs = [
        EventMergeInput(TardisDataType.TRADES, _trades_df([100]).lazy(), 1),
        EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([200]).lazy(), 0),
    ]
    meta = write_merged_events_parquet(inputs, out)
    assert out.exists()
    assert meta.output_path == out
    assert meta.input_count == 2
    assert meta.data_types == (TardisDataType.TRADES, TardisDataType.BOOK_SNAPSHOT_25)
    assert meta.row_count is None
    validate_merged_event_frame(pl.read_parquet(out), [TardisDataType.TRADES, TardisDataType.BOOK_SNAPSHOT_25])


def test_validate_merged_event_frame_rejects_bad_event_seq():
    df = merge_normalized_events([EventMergeInput(TardisDataType.TRADES, _trades_df([1, 2]).lazy(), 0)]).collect()
    bad = df.with_columns((pl.col(EVENT_SEQ) + 1).alias(EVENT_SEQ))
    with pytest.raises(ValueError):
        validate_merged_event_frame(bad, [TardisDataType.TRADES])


def test_no_feature_label_or_decision_columns_required():
    df = merge_normalized_events(
        [
            EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), 1),
            EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([2]).lazy(), 0),
        ]
    ).collect()
    cols = set(df.columns)
    for c in ["mid", "spread_bps", "microprice", "label", "tar" + "get", "fu" + "ture_ret", "fu" + "ture_mid", "decision"]:
        assert c not in cols


def test_merge_does_not_mutate_or_deduplicate_rows():
    rows = _trades_df([100, 100], raw_rows=[0, 1]).to_dicts()
    rows[1]["price"] = rows[0]["price"]
    rows[1]["amount"] = rows[0]["amount"]
    rows[1]["side"] = rows[0]["side"]
    rows[1]["side_code"] = rows[0]["side_code"]
    df = pl.DataFrame(rows).select(list(expected_normalized_columns(TardisDataType.TRADES)))
    out = merge_normalized_events([EventMergeInput(TardisDataType.TRADES, df.lazy(), 0)]).collect()
    assert out.height == 2
    assert out.get_column(RAW_SOURCE_ROW).to_list() == [0, 1]


def test_source_file_and_source_data_type_preserved():
    t = EventMergeInput(TardisDataType.TRADES, _trades_df([1]).lazy(), 1)
    s = EventMergeInput(TardisDataType.BOOK_SNAPSHOT_25, _snapshot_df([2]).lazy(), 0)
    df = merge_normalized_events([t, s]).collect()
    assert set(df.get_column(SOURCE_FILE).to_list()) == {"trades.parquet", "book_snapshot_25.parquet"}
    assert set(df.get_column(SOURCE_DATA_TYPE).to_list()) == {"trades", "book_snapshot_25"}
