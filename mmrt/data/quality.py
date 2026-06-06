"""Columnar quality checks for normalized Tardis event data.

This module analyzes normalized LazyFrames/Parquet files produced by mmrt.data.tardis_csv.
It is report-only: it does not sort, repair, drop, deduplicate, feature-engineer, label, or merge events.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl

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
)
from mmrt.schemas import tardis_csv_schema


class QualitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _coerce_data_type(data_type: TardisDataType | str) -> TardisDataType:
    if isinstance(data_type, TardisDataType):
        return data_type
    if isinstance(data_type, str):
        return TardisDataType(data_type)
    raise ValueError("data_type must be TardisDataType or str")


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _positive_int_or_none(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be None or positive int")
    return value


def _issue(name: str, severity: QualitySeverity, count: int, message: str) -> "QualityIssue":
    return QualityIssue(name=name, severity=severity, count=count, message=message)


@dataclass(frozen=True, slots=True)
class QualityIssue:
    name: str
    severity: QualitySeverity
    count: int
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_nonempty_str(self.name, "name"))
        object.__setattr__(self, "message", _require_nonempty_str(self.message, "message"))
        if isinstance(self.severity, str):
            object.__setattr__(self, "severity", QualitySeverity(self.severity))
        elif not isinstance(self.severity, QualitySeverity):
            raise ValueError("severity must be QualitySeverity or str")
        object.__setattr__(self, "count", _require_nonnegative_int(self.count, "count"))


@dataclass(frozen=True, slots=True)
class QualityReport:
    data_type: TardisDataType
    row_count: int
    min_ts_us: int | None
    max_ts_us: int | None
    min_local_ts_us: int | None
    max_local_ts_us: int | None
    issues: tuple[QualityIssue, ...]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_type", _coerce_data_type(self.data_type))
        object.__setattr__(self, "row_count", _require_nonnegative_int(self.row_count, "row_count"))
        object.__setattr__(self, "min_ts_us", _positive_int_or_none(self.min_ts_us, "min_ts_us"))
        object.__setattr__(self, "max_ts_us", _positive_int_or_none(self.max_ts_us, "max_ts_us"))
        object.__setattr__(self, "min_local_ts_us", _positive_int_or_none(self.min_local_ts_us, "min_local_ts_us"))
        object.__setattr__(self, "max_local_ts_us", _positive_int_or_none(self.max_local_ts_us, "max_local_ts_us"))
        issues = tuple(self.issues)
        for idx, issue in enumerate(issues):
            if not isinstance(issue, QualityIssue):
                raise ValueError(f"issues[{idx}] must be QualityIssue")
        object.__setattr__(self, "issues", issues)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def error_count(self) -> int:
        return sum(i.count for i in self.issues if i.severity == QualitySeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(i.count for i in self.issues if i.severity == QualitySeverity.WARNING)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        return self.warning_count > 0

    @property
    def issue_names(self) -> tuple[str, ...]:
        return tuple(i.name for i in self.issues)


def scan_normalized_parquet(path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(Path(path))


def analyze_normalized_parquet(path: str | Path, data_type: TardisDataType | str) -> QualityReport:
    p = Path(path)
    return analyze_normalized_lazyframe(scan_normalized_parquet(p), data_type, source_path=p)


def analyze_normalized_lazyframe(
    lf: pl.LazyFrame,
    data_type: TardisDataType | str,
    *,
    source_path: str | Path | None = None,
) -> QualityReport:
    dt = _coerce_data_type(data_type)
    issues: list[QualityIssue] = []

    expected = expected_normalized_columns(dt)
    actual = tuple(lf.collect_schema().names())
    actual_set = set(actual)
    expected_set = set(expected)
    missing = tuple(c for c in expected if c not in actual_set)
    unexpected = tuple(c for c in actual if c not in expected_set)
    if missing:
        issues.append(_issue("missing_columns", QualitySeverity.ERROR, len(missing), f"normalized data is missing expected columns: {', '.join(missing)}"))
    if unexpected:
        issues.append(_issue("unexpected_columns", QualitySeverity.WARNING, len(unexpected), f"normalized data contains unexpected columns: {', '.join(unexpected)}"))
    if not missing and not unexpected and actual != expected:
        issues.append(_issue("column_order_mismatch", QualitySeverity.ERROR, 1, "normalized columns are present but not in expected order"))

    has_ts = TS_US in actual_set
    has_local_ts = LOCAL_TS_US in actual_set
    has_raw_source_row = RAW_SOURCE_ROW in actual_set
    has_source_data_type = SOURCE_DATA_TYPE in actual_set
    has_source_file = SOURCE_FILE in actual_set
    common_metrics_exprs: list[pl.Expr] = [pl.len().alias("row_count")]
    if has_ts:
        common_metrics_exprs += [
            pl.col(TS_US).min().alias("min_ts_us"), pl.col(TS_US).max().alias("max_ts_us"),
            pl.col(TS_US).is_null().sum().alias("null_ts_count"),
            ((pl.col(TS_US) <= 0) & pl.col(TS_US).is_not_null()).sum().alias("nonpositive_ts_count"),
        ]
    if has_local_ts:
        common_metrics_exprs += [
            pl.col(LOCAL_TS_US).min().alias("min_local_ts_us"), pl.col(LOCAL_TS_US).max().alias("max_local_ts_us"),
            pl.col(LOCAL_TS_US).is_null().sum().alias("null_local_ts_count"),
            ((pl.col(LOCAL_TS_US) <= 0) & pl.col(LOCAL_TS_US).is_not_null()).sum().alias("nonpositive_local_ts_count"),
            (
                pl.col(LOCAL_TS_US).diff().is_not_null()
                & (pl.col(LOCAL_TS_US).diff() < 0)
            ).sum().alias("local_ts_us_decreases"),
        ]
    if has_raw_source_row:
        common_metrics_exprs += [
            pl.col(RAW_SOURCE_ROW).is_null().sum().alias("null_raw_source_row_count"),
            ((pl.col(RAW_SOURCE_ROW) < 0) & pl.col(RAW_SOURCE_ROW).is_not_null()).sum().alias("negative_raw_source_row_count"),
            (
                pl.col(RAW_SOURCE_ROW).diff().over(SOURCE_FILE).is_not_null()
                & (pl.col(RAW_SOURCE_ROW).diff().over(SOURCE_FILE) != 1)
            ).sum().alias("raw_source_row_not_contiguous"),
        ]
    if has_source_data_type:
        common_metrics_exprs.append((pl.col(SOURCE_DATA_TYPE).is_null() | (pl.col(SOURCE_DATA_TYPE) != dt.value)).sum().alias("bad_source_data_type_count"))
    if has_source_file:
        common_metrics_exprs.append((pl.col(SOURCE_FILE).is_null() | (pl.col(SOURCE_FILE) == "")).sum().alias("missing_source_file_count"))

    common = lf.select(common_metrics_exprs).collect().row(0, named=True)
    row_count = int(common["row_count"])
    min_ts_us = int(common["min_ts_us"]) if has_ts and common["min_ts_us"] is not None else None
    max_ts_us = int(common["max_ts_us"]) if has_ts and common["max_ts_us"] is not None else None
    min_local_ts_us = int(common["min_local_ts_us"]) if has_local_ts and common["min_local_ts_us"] is not None else None
    max_local_ts_us = int(common["max_local_ts_us"]) if has_local_ts and common["max_local_ts_us"] is not None else None

    if row_count == 0:
        issues.append(_issue("empty_data", QualitySeverity.WARNING, 1, "normalized data contains zero rows"))

    if has_ts and int(common["null_ts_count"]) > 0:
        issues.append(_issue("null_ts_us", QualitySeverity.ERROR, int(common["null_ts_count"]), "ts_us contains null values"))
    if has_local_ts and int(common["null_local_ts_count"]) > 0:
        issues.append(_issue("null_local_ts_us", QualitySeverity.ERROR, int(common["null_local_ts_count"]), "local_ts_us contains null values"))
    if has_ts and int(common["nonpositive_ts_count"]) > 0:
        issues.append(_issue("nonpositive_ts_us", QualitySeverity.ERROR, int(common["nonpositive_ts_count"]), "ts_us must be positive integer microseconds"))
    if has_local_ts and int(common["nonpositive_local_ts_count"]) > 0:
        issues.append(_issue("nonpositive_local_ts_us", QualitySeverity.ERROR, int(common["nonpositive_local_ts_count"]), "local_ts_us must be positive integer microseconds"))
    if has_local_ts and int(common["local_ts_us_decreases"]) > 0:
        issues.append(_issue("local_ts_us_decreases", QualitySeverity.ERROR, int(common["local_ts_us_decreases"]), "local_ts_us must be nondecreasing in stored row order"))
    if has_raw_source_row and int(common["null_raw_source_row_count"]) > 0:
        issues.append(_issue("null_raw_source_row", QualitySeverity.ERROR, int(common["null_raw_source_row_count"]), "raw_source_row contains null values"))
    if has_raw_source_row and int(common["negative_raw_source_row_count"]) > 0:
        issues.append(_issue("negative_raw_source_row", QualitySeverity.ERROR, int(common["negative_raw_source_row_count"]), "raw_source_row must be non-negative"))
    if has_raw_source_row and int(common["raw_source_row_not_contiguous"]) > 0:
        issues.append(_issue("raw_source_row_not_contiguous", QualitySeverity.ERROR, int(common["raw_source_row_not_contiguous"]), "raw_source_row should preserve source-file row order"))

    if SOURCE_FILE in actual_set and RAW_SOURCE_ROW in actual_set:
        count = int(lf.select(pl.struct([SOURCE_FILE, RAW_SOURCE_ROW]).is_duplicated().sum().alias("c")).collect().item())
        if count > 0:
            issues.append(_issue("duplicate_source_rows", QualitySeverity.ERROR, count, "duplicate source_file/raw_source_row pairs found"))

    if has_source_data_type and int(common["bad_source_data_type_count"]) > 0:
        issues.append(_issue("bad_source_data_type", QualitySeverity.ERROR, int(common["bad_source_data_type_count"]), "source_data_type values do not match requested data_type"))

    if has_source_file and int(common["missing_source_file_count"]) > 0:
        issues.append(_issue("missing_source_file", QualitySeverity.WARNING, int(common["missing_source_file_count"]), "source_file is null or empty"))

    if dt == TardisDataType.TRADES and {"price", "amount", "side_code"}.issubset(actual_set):
        m = lf.select([
            (pl.col("price").is_null() | (pl.col("price") <= 0)).sum().alias("invalid_price"),
            (pl.col("amount").is_null() | (pl.col("amount") <= 0)).sum().alias("invalid_amount"),
            (pl.col("side_code").is_null() | ~pl.col("side_code").is_in([SIDE_SELL, SIDE_UNKNOWN, SIDE_BUY])).sum().alias("invalid_side_code"),
        ]).collect().row(0, named=True)
        for k in ("invalid_price", "invalid_amount", "invalid_side_code"):
            if int(m[k]) > 0:
                issues.append(_issue(k, QualitySeverity.ERROR, int(m[k]), f"{k} rows found"))

    elif dt == TardisDataType.INCREMENTAL_BOOK_L2 and {"price", "amount", "book_side_code", "is_snapshot"}.issubset(actual_set):
        m = lf.select([
            (pl.col("price").is_null() | (pl.col("price") <= 0)).sum().alias("invalid_price"),
            (pl.col("amount").is_null() | (pl.col("amount") < 0)).sum().alias("invalid_amount"),
            (pl.col("book_side_code").is_null() | ~pl.col("book_side_code").is_in([BOOK_SIDE_BID, BOOK_SIDE_ASK])).sum().alias("invalid_book_side_code"),
            pl.col("is_snapshot").is_null().sum().alias("null_is_snapshot"),
            (pl.col("is_snapshot") == True).sum().alias("snapshot_marker_count"),
        ]).collect().row(0, named=True)
        for k in ("invalid_price", "invalid_amount", "invalid_book_side_code", "null_is_snapshot"):
            if int(m[k]) > 0:
                nm = "null_is_snapshot" if k == "null_is_snapshot" else k
                issues.append(_issue(nm, QualitySeverity.ERROR, int(m[k]), f"{nm} rows found"))
        issues.append(_issue("incremental_snapshot_markers", QualitySeverity.INFO, int(m["snapshot_marker_count"]), "incremental_book_L2 rows marked as snapshots"))

    elif dt == TardisDataType.BOOK_SNAPSHOT_25 and {"bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00"}.issubset(actual_set):
        depth = tardis_csv_schema(dt).depth_limit
        assert depth is not None
        cols = [f"{side}_{field}_{i:02d}" for i in range(depth) for side in ("bid", "ask") for field in ("px", "sz")]
        deeper_cols = [c for c in cols if not c.endswith("_00")]
        m = lf.select([
            pl.sum_horizontal([pl.col("bid_px_00").is_null(), pl.col("bid_sz_00").is_null(), pl.col("ask_px_00").is_null(), pl.col("ask_sz_00").is_null()]).sum().alias("missing_l1_snapshot"),
            ((pl.col("bid_px_00") <= 0) & pl.col("bid_px_00").is_not_null()).sum().alias("invalid_l1_price_bid"),
            ((pl.col("ask_px_00") <= 0) & pl.col("ask_px_00").is_not_null()).sum().alias("invalid_l1_price_ask"),
            ((pl.col("bid_sz_00") <= 0) & pl.col("bid_sz_00").is_not_null()).sum().alias("invalid_l1_size_bid"),
            ((pl.col("ask_sz_00") <= 0) & pl.col("ask_sz_00").is_not_null()).sum().alias("invalid_l1_size_ask"),
            ((pl.col("bid_px_00") >= pl.col("ask_px_00")) & pl.col("bid_px_00").is_not_null() & pl.col("ask_px_00").is_not_null()).sum().alias("crossed_l1_book"),
            pl.sum_horizontal([pl.col(c).is_null().sum() for c in deeper_cols]).alias("missing_depth"),
        ]).collect().row(0, named=True)
        if int(m["missing_l1_snapshot"]) > 0:
            issues.append(_issue("missing_l1_snapshot", QualitySeverity.ERROR, int(m["missing_l1_snapshot"]), "L1 snapshot fields contain null values"))
        inv_px = int(m["invalid_l1_price_bid"]) + int(m["invalid_l1_price_ask"])
        if inv_px > 0:
            issues.append(_issue("invalid_l1_price", QualitySeverity.ERROR, inv_px, "L1 prices must be > 0 when non-null"))
        inv_sz = int(m["invalid_l1_size_bid"]) + int(m["invalid_l1_size_ask"])
        if inv_sz > 0:
            issues.append(_issue("invalid_l1_size", QualitySeverity.ERROR, inv_sz, "L1 sizes must be > 0 when non-null"))
        if int(m["crossed_l1_book"]) > 0:
            issues.append(_issue("crossed_l1_book", QualitySeverity.ERROR, int(m["crossed_l1_book"]), "snapshot L1 is crossed"))
        if int(m["missing_depth"]) > 0:
            issues.append(_issue("missing_snapshot_depth_values", QualitySeverity.WARNING, int(m["missing_depth"]), "snapshot depth contains null values beyond L1"))

    return QualityReport(
        data_type=dt,
        row_count=row_count,
        min_ts_us=min_ts_us if (min_ts_us is None or min_ts_us > 0) else None,
        max_ts_us=max_ts_us if (max_ts_us is None or max_ts_us > 0) else None,
        min_local_ts_us=min_local_ts_us if (min_local_ts_us is None or min_local_ts_us > 0) else None,
        max_local_ts_us=max_local_ts_us if (max_local_ts_us is None or max_local_ts_us > 0) else None,
        issues=tuple(issues),
        source_path=Path(source_path) if source_path is not None else None,
    )


def raise_on_quality_errors(report: QualityReport) -> None:
    if report.has_errors:
        names = ", ".join(report.issue_names)
        raise ValueError(f"quality report contains errors: {names}")


__all__ = [
    "QualitySeverity",
    "QualityIssue",
    "QualityReport",
    "scan_normalized_parquet",
    "analyze_normalized_parquet",
    "analyze_normalized_lazyframe",
    "raise_on_quality_errors",
]
