"""Canonical streaming event merge for normalized Tardis data.

This module merges sorted normalized Parquet files produced by mmrt.data.tardis_csv
into a deterministic local_ts_us-ordered event stream. It does not parse raw CSV,
repair data, reconstruct books, compute features, compute labels, or make trading
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Any, Iterator, Sequence

import polars as pl
import pyarrow.parquet as pq

from mmrt.contracts import EventType, TardisDataType
from mmrt.data.tardis_csv import (
    LOCAL_TS_US,
    RAW_SOURCE_ROW,
    SOURCE_DATA_TYPE,
    SOURCE_FILE,
    TS_US,
    expected_normalized_columns,
)

EVENT_SEQ = "event_seq"
EVENT_TYPE = "event_type"
EVENT_TYPE_CODE = "event_type_code"
MERGE_INPUT_RANK = "merge_input_rank"

EVENT_TYPE_CODE_BOOK_SNAPSHOT = 1
EVENT_TYPE_CODE_BOOK_DELTA = 2
EVENT_TYPE_CODE_TRADE = 3
EVENT_TYPE_CODE_BOOK_TICKER = 4
EVENT_TYPE_CODE_DERIVATIVE_TICKER = 5
EVENT_TYPE_CODE_LIQUIDATION = 6

MERGED_PAYLOAD_TYPE_ORDER = (
    TardisDataType.BOOK_SNAPSHOT_25,
    TardisDataType.BOOK_SNAPSHOT_5,
    TardisDataType.INCREMENTAL_BOOK_L2,
    TardisDataType.TRADES,
    TardisDataType.BOOK_TICKER,
    TardisDataType.DERIVATIVE_TICKER,
    TardisDataType.LIQUIDATIONS,
)


def _coerce_data_type(data_type: TardisDataType | str) -> TardisDataType:
    if isinstance(data_type, TardisDataType):
        return data_type
    if isinstance(data_type, str):
        try:
            return TardisDataType(data_type)
        except ValueError as exc:
            raise ValueError(f"invalid data_type: {data_type!r}") from exc
    raise ValueError("data_type must be TardisDataType or str")


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _tuple_of_data_types(values: Sequence[TardisDataType | str], name: str) -> tuple[TardisDataType, ...]:
    seq = tuple(values)
    if not seq:
        raise ValueError(f"{name} must not be empty")
    return tuple(_require_supported_merge_data_type(_coerce_data_type(v)) for v in seq)


def event_type_for_data_type(data_type: TardisDataType | str) -> EventType:
    dtype = _coerce_data_type(data_type)
    if dtype in (TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.BOOK_SNAPSHOT_5):
        return EventType.BOOK_SNAPSHOT
    if dtype == TardisDataType.INCREMENTAL_BOOK_L2:
        return EventType.BOOK_DELTA
    if dtype == TardisDataType.TRADES:
        return EventType.TRADE
    if dtype == TardisDataType.BOOK_TICKER:
        return EventType.BOOK_TICKER
    if dtype == TardisDataType.DERIVATIVE_TICKER:
        return EventType.DERIVATIVE_TICKER
    if dtype == TardisDataType.LIQUIDATIONS:
        return EventType.LIQUIDATION
    raise ValueError(f"unsupported data_type for event mapping: {dtype.value}")


def _require_supported_merge_data_type(dtype: TardisDataType) -> TardisDataType:
    event_type_for_data_type(dtype)
    return dtype


def event_type_code_for_event_type(event_type: EventType | str) -> int:
    et = EventType(event_type)
    if et == EventType.BOOK_SNAPSHOT:
        return EVENT_TYPE_CODE_BOOK_SNAPSHOT
    if et == EventType.BOOK_DELTA:
        return EVENT_TYPE_CODE_BOOK_DELTA
    if et == EventType.TRADE:
        return EVENT_TYPE_CODE_TRADE
    if et == EventType.BOOK_TICKER:
        return EVENT_TYPE_CODE_BOOK_TICKER
    if et == EventType.DERIVATIVE_TICKER:
        return EVENT_TYPE_CODE_DERIVATIVE_TICKER
    if et == EventType.LIQUIDATION:
        return EVENT_TYPE_CODE_LIQUIDATION
    raise ValueError(f"unsupported event_type: {et.value}")


def event_type_code_for_data_type(data_type: TardisDataType | str) -> int:
    return event_type_code_for_event_type(event_type_for_data_type(data_type))


@dataclass(frozen=True, slots=True)
class ParquetEventStreamInput:
    data_type: TardisDataType
    path: Path
    input_rank: int
    source_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_type", _require_supported_merge_data_type(_coerce_data_type(self.data_type)))
        p = Path(self.path)
        if not p.exists() or not p.is_file():
            raise ValueError(f"path must exist and be a file: {p}")
        object.__setattr__(self, "path", p)
        _require_nonnegative_int(self.input_rank, "input_rank")
        if not isinstance(self.source_name, str):
            raise ValueError("source_name must be str")


def parquet_event_stream_input(path: str | Path, data_type: TardisDataType | str, input_rank: int) -> ParquetEventStreamInput:
    p = Path(path)
    return ParquetEventStreamInput(data_type=data_type, path=p, input_rank=input_rank, source_name=str(p))


def validate_merge_input_schema(inp: ParquetEventStreamInput) -> None:
    expected = expected_normalized_columns(inp.data_type)
    actual = tuple(pq.ParquetFile(inp.path).schema_arrow.names)
    if actual != expected:
        raise ValueError(f"normalized input schema mismatch for {inp.data_type.value}: expected {expected}, got {actual}")


def expected_merged_columns(data_types: Sequence[TardisDataType | str]) -> tuple[str, ...]:
    dtypes = _tuple_of_data_types(data_types, "data_types")
    base = [EVENT_SEQ, EVENT_TYPE_CODE, EVENT_TYPE, MERGE_INPUT_RANK, RAW_SOURCE_ROW, SOURCE_FILE, SOURCE_DATA_TYPE, "exchange", "symbol", TS_US, LOCAL_TS_US]
    excluded = {RAW_SOURCE_ROW, SOURCE_FILE, SOURCE_DATA_TYPE, "exchange", "symbol", TS_US, LOCAL_TS_US}
    present = set(dtypes)
    payload: list[str] = []
    seen = set()
    for dtype in MERGED_PAYLOAD_TYPE_ORDER:
        if dtype not in present:
            continue
        for c in expected_normalized_columns(dtype):
            if c in excluded or c in seen:
                continue
            seen.add(c)
            payload.append(c)
    return tuple(base + payload)


class _BatchRowReader:
    def __init__(self, inp: ParquetEventStreamInput, input_index: int, batch_size: int):
        self.inp = inp
        self.input_index = input_index
        self._batches = pq.ParquetFile(inp.path).iter_batches(batch_size=batch_size)
        self._data: dict[str, list[Any]] = {}
        self._batch_rows = 0
        self._row_idx = 0
        self._prev_key: tuple[int, int] | None = None

    def next_row(self) -> dict[str, Any] | None:
        while self._row_idx >= self._batch_rows:
            try:
                batch = next(self._batches)
            except StopIteration:
                return None
            self._data = batch.to_pydict()
            self._batch_rows = batch.num_rows
            self._row_idx = 0
            if self._batch_rows == 0:
                continue
        row = {k: self._data[k][self._row_idx] for k in self._data}
        self._row_idx += 1
        key = (int(row[LOCAL_TS_US]), int(row[RAW_SOURCE_ROW]))
        if self._prev_key is not None and key < self._prev_key:
            raise ValueError("streaming merge input is not sorted by local_ts_us, raw_source_row")
        self._prev_key = key
        return row


def iter_merged_events_streaming(inputs: Sequence[ParquetEventStreamInput], *, batch_size: int = 65536) -> Iterator[dict[str, Any]]:
    inps = tuple(inputs)
    if not inps:
        raise ValueError("inputs must not be empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive int")
    for idx, inp in enumerate(inps):
        if not isinstance(inp, ParquetEventStreamInput):
            raise ValueError(f"inputs[{idx}] must be ParquetEventStreamInput")
    ranks = [inp.input_rank for inp in inps]
    if len(set(ranks)) != len(ranks):
        raise ValueError("input_rank values must be unique")
    for inp in inps:
        validate_merge_input_schema(inp)

    out_cols = expected_merged_columns(tuple(inp.data_type for inp in inps))
    readers = [_BatchRowReader(inp, idx, batch_size) for idx, inp in enumerate(inps)]
    heap: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []

    for reader in readers:
        row = reader.next_row()
        if row is None:
            continue
        inp = reader.inp
        key = (int(row[LOCAL_TS_US]), int(inp.input_rank), int(row[RAW_SOURCE_ROW]), int(reader.input_index))
        heapq.heappush(heap, (key, reader.input_index, row))

    event_seq = 0
    while heap:
        _, input_index, row = heapq.heappop(heap)
        reader = readers[input_index]
        inp = reader.inp
        event_type = event_type_for_data_type(inp.data_type)
        out = {c: None for c in out_cols}
        out.update(row)
        out[EVENT_SEQ] = event_seq
        out[EVENT_TYPE_CODE] = event_type_code_for_event_type(event_type)
        out[EVENT_TYPE] = event_type.value
        out[MERGE_INPUT_RANK] = inp.input_rank
        yield out
        event_seq += 1

        next_row = reader.next_row()
        if next_row is not None:
            next_key = (int(next_row[LOCAL_TS_US]), int(inp.input_rank), int(next_row[RAW_SOURCE_ROW]), int(input_index))
            heapq.heappush(heap, (next_key, input_index, next_row))


def validate_merged_event_frame(df: pl.DataFrame, data_types: Sequence[TardisDataType | str]) -> None:
    expected_cols = list(expected_merged_columns(data_types))
    if df.columns != expected_cols:
        raise ValueError(f"merged columns mismatch: expected {expected_cols}, got {df.columns}")
    if EVENT_SEQ not in df.columns:
        raise ValueError(f"{EVENT_SEQ} missing")
    if df.schema[EVENT_SEQ] != pl.Int64:
        raise ValueError(f"{EVENT_SEQ} must be Int64")
    seq = df.get_column(EVENT_SEQ).to_list()
    if seq != list(range(len(df))):
        raise ValueError(f"{EVENT_SEQ} must be contiguous starting at 0")
    if EVENT_TYPE_CODE not in df.columns:
        raise ValueError(f"{EVENT_TYPE_CODE} missing")
    local = df.get_column(LOCAL_TS_US).to_list()
    if any(local[i] > local[i + 1] for i in range(len(local) - 1)):
        raise ValueError(f"{LOCAL_TS_US} must be non-decreasing")


__all__ = [
    "EVENT_SEQ",
    "EVENT_TYPE",
    "EVENT_TYPE_CODE",
    "MERGE_INPUT_RANK",
    "EVENT_TYPE_CODE_BOOK_SNAPSHOT",
    "EVENT_TYPE_CODE_BOOK_DELTA",
    "EVENT_TYPE_CODE_TRADE",
    "EVENT_TYPE_CODE_BOOK_TICKER",
    "EVENT_TYPE_CODE_DERIVATIVE_TICKER",
    "EVENT_TYPE_CODE_LIQUIDATION",
    "MERGED_PAYLOAD_TYPE_ORDER",
    "ParquetEventStreamInput",
    "event_type_for_data_type",
    "event_type_code_for_event_type",
    "event_type_code_for_data_type",
    "parquet_event_stream_input",
    "validate_merge_input_schema",
    "expected_merged_columns",
    "iter_merged_events_streaming",
    "validate_merged_event_frame",
]
