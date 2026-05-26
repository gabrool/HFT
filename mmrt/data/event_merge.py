"""Canonical event-stream merging for normalized Tardis data.

This module merges normalized LazyFrames/Parquet files produced by mmrt.data.tardis_csv
into a deterministic local_ts_us-ordered event stream. It does not parse raw CSV,
repair data, reconstruct books, compute features, compute labels, or make trading
decisions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from mmrt.contracts import EventType, TardisDataType
from mmrt.data.tardis_csv import (
    DEFAULT_PARQUET_COMPRESSION,
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


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _tuple_of_data_types(values: Sequence[TardisDataType | str], name: str) -> tuple[TardisDataType, ...]:
    seq = tuple(values)
    if not seq:
        raise ValueError(f"{name} must not be empty")
    return tuple(_coerce_data_type(v) for v in seq)


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
class EventMergeInput:
    data_type: TardisDataType
    frame: pl.LazyFrame
    input_rank: int
    source_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_type", _coerce_data_type(self.data_type))
        if not isinstance(self.frame, pl.LazyFrame):
            raise ValueError("frame must be pl.LazyFrame")
        _require_nonnegative_int(self.input_rank, "input_rank")
        if not isinstance(self.source_name, str):
            raise ValueError("source_name must be str")


@dataclass(frozen=True, slots=True)
class MergedEventFile:
    output_path: Path
    input_count: int
    data_types: tuple[TardisDataType, ...]
    row_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))
        if isinstance(self.input_count, bool) or not isinstance(self.input_count, int) or self.input_count <= 0:
            raise ValueError("input_count must be a positive int")
        object.__setattr__(self, "data_types", _tuple_of_data_types(self.data_types, "data_types"))
        if self.row_count is not None:
            _require_nonnegative_int(self.row_count, "row_count")


def parquet_merge_input(path: str | Path, data_type: TardisDataType | str, input_rank: int) -> EventMergeInput:
    p = Path(path)
    return EventMergeInput(
        data_type=_coerce_data_type(data_type),
        frame=pl.scan_parquet(p),
        input_rank=input_rank,
        source_name=str(p),
    )


def validate_merge_input_schema(inp: EventMergeInput) -> None:
    expected = expected_normalized_columns(inp.data_type)
    actual = tuple(inp.frame.collect_schema().names())
    if actual != expected:
        raise ValueError(
            f"normalized input schema mismatch for {inp.data_type.value}: expected {expected}, got {actual}"
        )


def expected_merged_columns(data_types: Sequence[TardisDataType | str]) -> tuple[str, ...]:
    dtypes = _tuple_of_data_types(data_types, "data_types")
    base = [
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
    ]
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


def _prepare_input_for_merge(inp: EventMergeInput) -> pl.LazyFrame:
    validate_merge_input_schema(inp)
    event_type = event_type_for_data_type(inp.data_type)
    code = event_type_code_for_event_type(event_type)
    return inp.frame.with_columns(
        [
            pl.lit(inp.input_rank).cast(pl.Int64).alias(MERGE_INPUT_RANK),
            pl.lit(event_type.value).cast(pl.Utf8).alias(EVENT_TYPE),
            pl.lit(code).cast(pl.Int8).alias(EVENT_TYPE_CODE),
        ]
    ).select([EVENT_TYPE_CODE, EVENT_TYPE, MERGE_INPUT_RANK, *expected_normalized_columns(inp.data_type)])


def merge_normalized_events(inputs: Sequence[EventMergeInput]) -> pl.LazyFrame:
    inps = tuple(inputs)
    if not inps:
        raise ValueError("inputs must not be empty")
    for idx, inp in enumerate(inps):
        if not isinstance(inp, EventMergeInput):
            raise ValueError(f"inputs[{idx}] must be EventMergeInput")
    ranks = [inp.input_rank for inp in inps]
    if len(set(ranks)) != len(ranks):
        raise ValueError("input_rank values must be unique")
    prepared = [_prepare_input_for_merge(inp) for inp in inps]
    merged = pl.concat(prepared, how="diagonal_relaxed")
    merged = merged.sort([LOCAL_TS_US, MERGE_INPUT_RANK, RAW_SOURCE_ROW], maintain_order=True)
    merged = merged.with_row_index(EVENT_SEQ).with_columns(pl.col(EVENT_SEQ).cast(pl.Int64))
    return merged.select(list(expected_merged_columns(tuple(inp.data_type for inp in inps))))


def write_merged_events_parquet(
    inputs: Sequence[EventMergeInput],
    output_path: str | Path,
    *,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> MergedEventFile:
    out = Path(output_path)
    lf = merge_normalized_events(inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(out, compression=compression)
    return MergedEventFile(
        output_path=out,
        input_count=len(inputs),
        data_types=tuple(inp.data_type for inp in inputs),
        row_count=None,
    )


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
    "EventMergeInput",
    "MergedEventFile",
    "event_type_for_data_type",
    "event_type_code_for_event_type",
    "event_type_code_for_data_type",
    "parquet_merge_input",
    "validate_merge_input_schema",
    "expected_merged_columns",
    "merge_normalized_events",
    "write_merged_events_parquet",
    "validate_merged_event_frame",
]
