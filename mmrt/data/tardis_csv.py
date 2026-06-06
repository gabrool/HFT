"""Polars-based ETL helpers for Tardis normalized CSV files. This module validates Tardis headers and converts raw CSV files into canonical typed LazyFrames/Parquet without computing features, labels, or event merges."""

from dataclasses import dataclass
from pathlib import Path
import csv
import gzip
from typing import Sequence

import polars as pl

from mmrt.schemas import ColumnKind, TardisCSVSchema, tardis_csv_schema
from mmrt.contracts import TardisDataType
from mmrt.time_utils import parse_tardis_ts_us

RAW_SOURCE_ROW = "raw_source_row"
SOURCE_FILE = "source_file"
SOURCE_DATA_TYPE = "source_data_type"
TS_US = "ts_us"
LOCAL_TS_US = "local_ts_us"

SIDE_UNKNOWN = 0
SIDE_BUY = 1
SIDE_SELL = -1

BOOK_SIDE_BID = 1
BOOK_SIDE_ASK = -1

DEFAULT_PARQUET_COMPRESSION = "zstd"


@dataclass(frozen=True, slots=True)
class NormalizedTardisFile:
    data_type: TardisDataType
    input_path: Path
    output_path: Path
    row_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_type", TardisDataType(self.data_type))
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if self.row_count is not None:
            if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 0:
                raise ValueError("row_count must be None or a non-negative int")


def read_tardis_csv_header(path: str | Path) -> tuple[str, ...]:
    csv_path = Path(path)
    opener = gzip.open if csv_path.suffix == ".gz" else open
    mode = "rt" if csv_path.suffix == ".gz" else "r"
    with opener(csv_path, mode, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {csv_path}") from exc
    return tuple(header)


def validate_tardis_csv_header(path: str | Path, data_type: TardisDataType | str) -> TardisCSVSchema:
    schema = tardis_csv_schema(data_type)
    header = read_tardis_csv_header(path)
    schema.validate_header(header, exact=True)
    return schema


def _polars_dtype_for_kind(kind: ColumnKind) -> pl.DataType:
    if kind == ColumnKind.STRING:
        return pl.Utf8
    if kind == ColumnKind.INT_US:
        return pl.Int64
    if kind == ColumnKind.FLOAT:
        return pl.Float64
    if kind == ColumnKind.BOOL:
        return pl.Boolean
    if kind in (ColumnKind.SIDE, ColumnKind.BOOK_SIDE):
        return pl.Utf8
    raise ValueError(f"unsupported column kind: {kind}")


def polars_schema_overrides(schema: TardisCSVSchema) -> dict[str, pl.DataType]:
    return {col.name: _polars_dtype_for_kind(col.kind) for col in schema.columns}


def normalized_snapshot_column_name(raw_name: str) -> str:
    if raw_name.startswith("asks["):
        prefix = "ask"
        rest = raw_name[5:]
    elif raw_name.startswith("bids["):
        prefix = "bid"
        rest = raw_name[5:]
    else:
        raise ValueError(f"not a snapshot level column: {raw_name}")

    bracket_end = rest.find("]")
    if bracket_end <= 0:
        raise ValueError(f"not a snapshot level column: {raw_name}")
    level_text = rest[:bracket_end]
    suffix = rest[bracket_end + 1 :]
    if not level_text.isdigit():
        raise ValueError(f"not a snapshot level column: {raw_name}")
    level = int(level_text)

    if suffix == ".price":
        field = "px"
    elif suffix == ".amount":
        field = "sz"
    else:
        raise ValueError(f"not a snapshot level column: {raw_name}")
    return f"{prefix}_{field}_{level:02d}"


def normalized_column_renames(schema: TardisCSVSchema) -> dict[str, str]:
    renames = {"timestamp": TS_US, "local_timestamp": LOCAL_TS_US}
    if schema.data_type == TardisDataType.BOOK_SNAPSHOT_25:
        for name in schema.column_names:
            if name.startswith("asks[") or name.startswith("bids["):
                renames[name] = normalized_snapshot_column_name(name)
    return renames


def _normalized_str_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def _invalid_trade_side_filter(column: str = "side") -> pl.Expr:
    side = _normalized_str_expr(column)
    return side.is_null() | (side == "") | ~side.is_in(["buy", "sell", "unknown"])


def _invalid_book_side_filter(column: str = "side") -> pl.Expr:
    side = _normalized_str_expr(column)
    return side.is_null() | (side == "") | ~side.is_in(["bid", "ask"])


def _validate_side_values(lf: pl.LazyFrame, schema: TardisCSVSchema) -> None:
    if schema.data_type == TardisDataType.TRADES:
        bad = (
            lf.select([RAW_SOURCE_ROW, "side"])
            .filter(_invalid_trade_side_filter("side"))
            .limit(1)
            .collect()
        )
        if bad.height:
            raise ValueError(f"invalid trade side in {schema.data_type.value}: {bad.to_dicts()[0]!r}")
    elif schema.data_type == TardisDataType.INCREMENTAL_BOOK_L2:
        bad = (
            lf.select([RAW_SOURCE_ROW, "side"])
            .filter(_invalid_book_side_filter("side"))
            .limit(1)
            .collect()
        )
        if bad.height:
            raise ValueError(f"invalid book side in {schema.data_type.value}: {bad.to_dicts()[0]!r}")


def _trade_side_expr(column: str = "side") -> pl.Expr:
    side = _normalized_str_expr(column)
    return (
        pl.when(side == "unknown")
        .then(pl.lit(SIDE_UNKNOWN))
        .when(side == "buy")
        .then(pl.lit(SIDE_BUY))
        .when(side == "sell")
        .then(pl.lit(SIDE_SELL))
        .otherwise(pl.lit(None))
        .cast(pl.Int8)
        .alias("side_code")
    )


def _book_side_expr(column: str = "side") -> pl.Expr:
    side = _normalized_str_expr(column)
    return (
        pl.when(side == "bid")
        .then(pl.lit(BOOK_SIDE_BID))
        .when(side == "ask")
        .then(pl.lit(BOOK_SIDE_ASK))
        .otherwise(pl.lit(None))
        .cast(pl.Int8)
        .alias("book_side_code")
    )


def expected_normalized_columns(data_type: TardisDataType | str) -> tuple[str, ...]:
    schema = tardis_csv_schema(data_type)
    renames = normalized_column_renames(schema)
    cols = [RAW_SOURCE_ROW]
    cols.extend(renames.get(name, name) for name in schema.column_names)
    if schema.data_type == TardisDataType.TRADES:
        cols.append("side_code")
    if schema.data_type == TardisDataType.INCREMENTAL_BOOK_L2:
        cols.append("book_side_code")
    cols.extend((SOURCE_FILE, SOURCE_DATA_TYPE))
    return tuple(cols)


def scan_tardis_csv_normalized(
    path: str | Path,
    data_type: TardisDataType | str,
    *,
    source_file: str | None = None,
    row_index_name: str = RAW_SOURCE_ROW,
    validate_header: bool = True,
) -> pl.LazyFrame:
    csv_path = Path(path)
    schema = validate_tardis_csv_header(csv_path, data_type) if validate_header else tardis_csv_schema(data_type)

    lf = pl.scan_csv(
        csv_path,
        has_header=True,
        schema_overrides=polars_schema_overrides(schema),
        row_index_name=row_index_name,
    )
    lf = lf.select([row_index_name, *schema.column_names])
    lf = lf.rename(normalized_column_renames(schema))

    with_cols: list[pl.Expr] = [
        pl.col(TS_US).cast(pl.Int64),
        pl.col(LOCAL_TS_US).cast(pl.Int64),
        pl.col(row_index_name).cast(pl.Int64),
        pl.lit(source_file if source_file is not None else str(csv_path)).cast(pl.Utf8).alias(SOURCE_FILE),
        pl.lit(schema.data_type.value).cast(pl.Utf8).alias(SOURCE_DATA_TYPE),
    ]

    if schema.data_type == TardisDataType.TRADES:
        with_cols.append(_trade_side_expr())
    if schema.data_type == TardisDataType.INCREMENTAL_BOOK_L2:
        with_cols.append(_book_side_expr())

    if schema.data_type == TardisDataType.BOOK_SNAPSHOT_25:
        renames = normalized_column_renames(schema)
        for original in schema.column_names:
            if original.startswith("asks[") or original.startswith("bids["):
                with_cols.append(pl.col(renames[original]).cast(pl.Float64))

    lf = lf.with_columns(with_cols)
    return lf.select(list(expected_normalized_columns(schema.data_type)))


def validate_normalized_timestamps(df: pl.DataFrame) -> None:
    if TS_US not in df.columns or LOCAL_TS_US not in df.columns:
        raise ValueError(f"DataFrame must contain {TS_US!r} and {LOCAL_TS_US!r} columns")
    for v in df.get_column(TS_US).to_list():
        parse_tardis_ts_us(v)
    for v in df.get_column(LOCAL_TS_US).to_list():
        parse_tardis_ts_us(v)


def write_normalized_parquet(
    input_path: str | Path,
    output_path: str | Path,
    data_type: TardisDataType | str,
    *,
    source_file: str | None = None,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
    validate_header: bool = True,
) -> NormalizedTardisFile:
    in_path = Path(input_path)
    out_path = Path(output_path)
    lf = scan_tardis_csv_normalized(
        in_path,
        data_type,
        source_file=source_file,
        validate_header=validate_header,
    )
    schema = tardis_csv_schema(data_type)
    _validate_side_values(lf, schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf.sort([LOCAL_TS_US, RAW_SOURCE_ROW]).sink_parquet(out_path, compression=compression)
    return NormalizedTardisFile(data_type=schema.data_type, input_path=in_path, output_path=out_path, row_count=None)


__all__ = [
    "RAW_SOURCE_ROW",
    "SOURCE_FILE",
    "SOURCE_DATA_TYPE",
    "TS_US",
    "LOCAL_TS_US",
    "SIDE_UNKNOWN",
    "SIDE_BUY",
    "SIDE_SELL",
    "BOOK_SIDE_BID",
    "BOOK_SIDE_ASK",
    "DEFAULT_PARQUET_COMPRESSION",
    "NormalizedTardisFile",
    "read_tardis_csv_header",
    "validate_tardis_csv_header",
    "polars_schema_overrides",
    "normalized_snapshot_column_name",
    "normalized_column_renames",
    "scan_tardis_csv_normalized",
    "validate_normalized_timestamps",
    "write_normalized_parquet",
    "expected_normalized_columns",
]
