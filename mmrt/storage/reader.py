from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from mmrt.contracts import SplitRole, StorageFormat, TimeRangeUS, TimeUnit
from mmrt.storage import manifest as mf

DEFAULT_BATCH_SIZE = 65_536


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _unique_preserve_order(values: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def _ensure_columns_exist(columns: tuple[str, ...], required_columns: tuple[str, ...]) -> None:
    required_set = set(required_columns)
    missing = [c for c in columns if c not in required_set]
    if missing:
        raise ValueError(f"unknown column(s): {missing!r}")


def _column_type_map_for_manifest(manifest: mf.StorageManifest) -> dict[str, pa.DataType]:
    out = {c: pa.float32() for c in manifest.x_columns}
    out.update({c: pa.float32() for c in manifest.y_columns})
    out[mf.ROW_IDX_COLUMN] = pa.int64()
    out[mf.DECISION_INDEX_COLUMN] = pa.int64()
    out[mf.TS_US_COLUMN] = pa.int64()
    out[mf.LOCAL_TS_US_COLUMN] = pa.int64()
    out[mf.EVENT_SEQ_COLUMN] = pa.int64()
    out[mf.LABEL_ENTRY_TS_US_COLUMN] = pa.int64()
    out[mf.RAW_MID_COLUMN] = pa.float64()
    return out


def _validate_arrow_schema(schema: pa.Schema, manifest: mf.StorageManifest) -> None:
    if list(schema.names) != list(manifest.required_columns):
        raise ValueError("parquet columns do not match manifest.required_columns")
    expected = _column_type_map_for_manifest(manifest)
    for name in manifest.required_columns:
        got = schema.field(name).type
        want = expected[name]
        if got != want:
            raise ValueError(f"column {name!r} type mismatch: {got} != {want}")


@dataclass(frozen=True, slots=True)
class ReaderConfig:
    dataset_root: str
    validate_on_open: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_root", _require_nonempty_str(self.dataset_root, "dataset_root"))
        _require_bool(self.validate_on_open, "validate_on_open")
        _require_positive_int(self.batch_size, "batch_size")


@dataclass(frozen=True, slots=True)
class SegmentReadPlan:
    segment_key: str
    parquet_path: str
    absolute_path: str
    first_row_idx: int
    last_row_idx: int
    row_count: int
    local_time_range: TimeRangeUS
    time_range: TimeRangeUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_key", _require_nonempty_str(self.segment_key, "segment_key"))
        object.__setattr__(self, "parquet_path", _require_nonempty_str(self.parquet_path, "parquet_path"))
        object.__setattr__(self, "absolute_path", _require_nonempty_str(self.absolute_path, "absolute_path"))
        if isinstance(self.first_row_idx, bool) or not isinstance(self.first_row_idx, int) or self.first_row_idx < 0:
            raise ValueError("first_row_idx must be non-negative int")
        if isinstance(self.last_row_idx, bool) or not isinstance(self.last_row_idx, int) or self.last_row_idx < self.first_row_idx:
            raise ValueError("last_row_idx invalid")
        _require_positive_int(self.row_count, "row_count")
        if self.last_row_idx - self.first_row_idx + 1 != self.row_count:
            raise ValueError("row_count does not match row idx range")
        if not isinstance(self.local_time_range, TimeRangeUS):
            raise ValueError("local_time_range must be TimeRangeUS")
        if not isinstance(self.time_range, TimeRangeUS):
            raise ValueError("time_range must be TimeRangeUS")


class StorageDatasetReader:
    def __init__(self, config: ReaderConfig) -> None:
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.manifest_path = self.dataset_root / mf.DEFAULT_MANIFEST_FILENAME
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        self.manifest = mf.read_manifest_json(self.manifest_path)
        self.manifest.validate_against_current_code()
        self.segments = tuple(
            SegmentReadPlan(
                segment_key=s.segment_key,
                parquet_path=s.parquet_path,
                absolute_path=str((self.dataset_root / s.parquet_path).resolve()),
                first_row_idx=s.first_row_idx,
                last_row_idx=s.last_row_idx,
                row_count=s.row_count,
                local_time_range=s.local_time_range,
                time_range=s.time_range,
            )
            for s in self.manifest.segments
        )
        if self.config.validate_on_open:
            self.validate_dataset()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.manifest.x_columns

    @property
    def label_columns(self) -> tuple[str, ...]:
        return self.manifest.y_columns

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.manifest.required_columns

    @property
    def base_columns(self) -> tuple[str, ...]:
        return mf.BASE_ROW_COLUMNS

    @property
    def total_rows(self) -> int:
        return self.manifest.total_rows

    @property
    def total_labels(self) -> int:
        return self.manifest.total_labels

    def _segment_plan_by_key(self, segment_key: str) -> SegmentReadPlan:
        for s in self.segments:
            if s.segment_key == segment_key:
                return s
        raise ValueError(f"unknown segment key: {segment_key}")

    def _segment_path(self, segment: mf.StorageSegment | SegmentReadPlan | str) -> Path:
        if isinstance(segment, str):
            plan = self._segment_plan_by_key(segment)
            p = self.dataset_root / plan.parquet_path
        elif isinstance(segment, SegmentReadPlan):
            p = self.dataset_root / segment.parquet_path
        else:
            p = self.dataset_root / segment.parquet_path
        if not p.exists():
            raise FileNotFoundError(f"segment not found: {p}")
        return p

    def dataset(self, *, segments: tuple[str, ...] | None = None) -> ds.Dataset:
        seg_keys = tuple(s.segment_key for s in self.segments) if segments is None else segments
        paths = [str(self._segment_path(k)) for k in seg_keys]
        return ds.dataset(paths, format="parquet")

    def available_columns(self) -> tuple[str, ...]:
        return self.required_columns

    def select_columns(self, *, include_base: bool = True, include_labels: bool = True, include_features: bool = True, feature_columns: tuple[str, ...] | None = None, label_columns: tuple[str, ...] | None = None, extra_columns: tuple[str, ...] = ()) -> tuple[str, ...]:
        cols: list[str] = []
        if include_base:
            cols.extend(mf.BASE_ROW_COLUMNS)
        if label_columns is not None:
            for c in label_columns:
                if c not in self.manifest.y_columns:
                    raise ValueError(f"unknown label column: {c}")
            cols.extend(label_columns)
        elif include_labels:
            cols.extend(self.manifest.y_columns)
        if feature_columns is not None:
            for c in feature_columns:
                if c not in self.manifest.x_columns:
                    raise ValueError(f"unknown feature column: {c}")
            cols.extend(feature_columns)
        elif include_features:
            cols.extend(self.manifest.x_columns)
        for c in extra_columns:
            if c not in self.required_columns:
                raise ValueError(f"unknown extra column: {c}")
            cols.append(c)
        result = _unique_preserve_order(tuple(cols))
        _ensure_columns_exist(result, self.required_columns)
        return result

    def read_table(self, *, columns: tuple[str, ...] | None = None, segments: tuple[str, ...] | None = None) -> pa.Table:
        cols = self.required_columns if columns is None else columns
        _ensure_columns_exist(cols, self.required_columns)
        return self.dataset(segments=segments).to_table(columns=list(cols))

    def iter_batches(self, *, columns: tuple[str, ...] | None = None, segments: tuple[str, ...] | None = None, batch_size: int | None = None) -> Iterator[pa.RecordBatch]:
        cols = self.required_columns if columns is None else columns
        _ensure_columns_exist(cols, self.required_columns)
        bs = self.config.batch_size if batch_size is None else _require_positive_int(batch_size, "batch_size")
        yield from self.dataset(segments=segments).scanner(columns=list(cols), batch_size=bs).to_batches()

    def read_segment_table(self, segment_key: str, columns: tuple[str, ...] | None = None) -> pa.Table:
        return self.read_table(columns=columns, segments=(segment_key,))

    def split_entries(self, role: SplitRole | str) -> tuple[mf.SplitMetadata, ...]:
        role_enum = SplitRole(role)
        return tuple(sp for sp in self.manifest.splits if sp.role == role_enum)

    def read_split_table(self, role: SplitRole | str, columns: tuple[str, ...] | None = None) -> pa.Table:
        entries = self.split_entries(role)
        if not entries:
            raise ValueError("no split entries for role")
        cols = self.required_columns if columns is None else columns
        _ensure_columns_exist(cols, self.required_columns)
        pieces: list[pa.Table] = []
        for sp in entries:
            table = self.read_segment_table(sp.segment_key, columns=cols)
            mask = pc.and_(pc.greater_equal(table[mf.ROW_IDX_COLUMN], sp.start_row), pc.less(table[mf.ROW_IDX_COLUMN], sp.end_row))
            pieces.append(table.filter(mask))
        return pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]

    def iter_split_batches(self, role: SplitRole | str, columns: tuple[str, ...] | None = None, batch_size: int | None = None) -> Iterator[pa.RecordBatch]:
        bs = self.config.batch_size if batch_size is None else _require_positive_int(batch_size, "batch_size")
        table = self.read_split_table(role, columns=columns)
        yield from table.to_batches(max_chunksize=bs)

    def validate_dataset(self) -> None:
        self.manifest.validate_against_current_code()
        manifest_paths = {self._segment_path(s.segment_key).resolve() for s in self.segments}
        all_segment_files = set((self.dataset_root / "segments").glob("*.parquet"))
        if manifest_paths != {p.resolve() for p in all_segment_files}:
            extra = {p.resolve() for p in all_segment_files} - manifest_paths
            if extra:
                raise ValueError(f"unmanifested parquet files found: {sorted(str(x) for x in extra)}")
        prev_last_row = -1
        prev_last_decision_index: int | None = None
        prev_last_local_ts: int | None = None
        for seg, seg_meta in zip(self.segments, self.manifest.segments):
            path = self._segment_path(seg)
            schema = pq.read_schema(path)
            _validate_arrow_schema(schema, self.manifest)
            pf = pq.ParquetFile(path)
            if pf.metadata is None or pf.metadata.num_rows != seg.row_count:
                raise ValueError(f"row_count mismatch for segment {seg.segment_key}")
            min_cols = [mf.ROW_IDX_COLUMN, mf.DECISION_INDEX_COLUMN, mf.LOCAL_TS_US_COLUMN, mf.TS_US_COLUMN]
            table = pq.read_table(path, columns=min_cols)
            row_idx = table[mf.ROW_IDX_COLUMN].to_pylist()
            decision = table[mf.DECISION_INDEX_COLUMN].to_pylist()
            local_ts = table[mf.LOCAL_TS_US_COLUMN].to_pylist()
            ts = table[mf.TS_US_COLUMN].to_pylist()
            expected_rows = list(range(seg.first_row_idx, seg.last_row_idx + 1))
            if row_idx != expected_rows:
                raise ValueError(f"row_idx mismatch in segment {seg.segment_key}")
            if seg.first_row_idx != prev_last_row + 1:
                raise ValueError("cross-segment row_idx discontinuity")
            prev_last_row = seg.last_row_idx
            for i in range(1, len(decision)):
                if decision[i] <= decision[i - 1]:
                    raise ValueError("decision_index must be strictly increasing")
            if prev_last_decision_index is not None and decision and decision[0] <= prev_last_decision_index:
                raise ValueError("decision_index must increase across segments")
            if decision:
                prev_last_decision_index = decision[-1]
            for i in range(1, len(local_ts)):
                if local_ts[i] < local_ts[i - 1]:
                    raise ValueError("local_ts_us must be nondecreasing")
            if prev_last_local_ts is not None and local_ts and local_ts[0] < prev_last_local_ts:
                raise ValueError("local_ts_us must be nondecreasing across segments")
            if local_ts:
                prev_last_local_ts = local_ts[-1]
            if any(x <= 0 for x in ts):
                raise ValueError("ts_us must be positive")
            if any(x <= 0 for x in local_ts):
                raise ValueError("local_ts_us must be positive")
            if any(not (seg_meta.time_range.start_us <= x < seg_meta.time_range.end_us) for x in ts):
                raise ValueError("ts_us out of segment time range")
            if any(not (seg_meta.local_time_range.start_us <= x < seg_meta.local_time_range.end_us) for x in local_ts):
                raise ValueError("local_ts_us out of segment local_time_range")


def open_dataset(dataset_root: str, *, validate_on_open: bool = True, batch_size: int = DEFAULT_BATCH_SIZE) -> StorageDatasetReader:
    return StorageDatasetReader(ReaderConfig(dataset_root=dataset_root, validate_on_open=validate_on_open, batch_size=batch_size))


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "ReaderConfig",
    "SegmentReadPlan",
    "StorageDatasetReader",
    "open_dataset",
]
