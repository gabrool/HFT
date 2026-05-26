from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from mmrt.config import PipelineConfig, default_config
from mmrt.contracts import LabelSpec, StorageFormat, TimeRangeUS, TimeUnit
from mmrt.storage import manifest as mf

DEFAULT_CHUNK_ROWS = 131_072
DEFAULT_ROW_GROUP_ROWS = 131_072


@dataclass(frozen=True, slots=True)
class DecisionRow:
    decision_index: int
    ts_us: int
    local_ts_us: int
    event_seq: int
    raw_mid: float
    label_entry_ts_us: int
    label_values: tuple[float, ...]
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_values", tuple(self.label_values))
        object.__setattr__(self, "feature_values", tuple(self.feature_values))

    @classmethod
    def from_arrays(
        cls,
        *,
        decision_index: int,
        ts_us: int,
        local_ts_us: int,
        event_seq: int,
        raw_mid: float,
        label_entry_ts_us: int,
        label_values,
        feature_values,
    ) -> "DecisionRow":
        return cls(
            decision_index=decision_index,
            ts_us=ts_us,
            local_ts_us=local_ts_us,
            event_seq=event_seq,
            raw_mid=raw_mid,
            label_entry_ts_us=label_entry_ts_us,
            label_values=tuple(label_values),
            feature_values=tuple(feature_values),
        )


@dataclass(frozen=True, slots=True)
class WriterConfig:
    dataset_id: str
    created_at_utc: str
    dataset_root: str
    config: PipelineConfig = field(default_factory=default_config)
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    row_group_rows: int = DEFAULT_ROW_GROUP_ROWS
    transform_config: Mapping[str, Any] | None = None
    transform_diagnostics: Mapping[str, Any] | None = None
    notes: Mapping[str, Any] | None = None
    source_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.strip():
            raise ValueError("created_at_utc must be non-empty")
        if not isinstance(self.dataset_root, str) or not self.dataset_root.strip():
            raise ValueError("dataset_root must be non-empty")
        if not isinstance(self.config, PipelineConfig):
            raise ValueError("config must be PipelineConfig")
        if self.config.storage.storage_format != StorageFormat.FLAT_DECISION_ROWS_US_V1:
            raise ValueError("unsupported storage format")
        if self.config.storage.time_unit != TimeUnit.MICROSECOND:
            raise ValueError("unsupported time unit")
        _require_positive_int(self.chunk_rows, "chunk_rows")
        _require_positive_int(self.row_group_rows, "row_group_rows")
        if self.transform_config is not None and not isinstance(self.transform_config, Mapping):
            raise ValueError("transform_config must be mapping or None")
        if self.transform_diagnostics is not None and not isinstance(self.transform_diagnostics, Mapping):
            raise ValueError("transform_diagnostics must be mapping or None")
        if self.notes is not None and not isinstance(self.notes, Mapping):
            raise ValueError("notes must be mapping or None")
        if isinstance(self.source_files, (str, bytes)):
            raise ValueError("source_files must not be a string")
        object.__setattr__(self, "source_files", tuple(self.source_files))


def arrow_schema(label_spec: LabelSpec) -> pa.Schema:
    names = mf.required_row_columns(label_spec)
    label_cols = set(mf.label_columns(label_spec))
    feature_cols = set(mf.feature_columns())
    fields = []
    for name in names:
        if name == mf.RAW_MID_COLUMN:
            ftype = pa.float64()
        elif name in label_cols or name in feature_cols:
            ftype = pa.float32()
        else:
            ftype = pa.int64()
        fields.append(pa.field(name, ftype, nullable=False))
    md = {
        "manifest_schema_version": mf.MANIFEST_SCHEMA_VERSION,
        "storage_format": StorageFormat.FLAT_DECISION_ROWS_US_V1.value,
        "time_unit": TimeUnit.MICROSECOND.value,
        "feature_schema_version": str(mf.feature_schema_record()["feature_schema_version"]),
        "feature_count": str(len(mf.feature_columns())),
    }
    return pa.schema(fields, metadata={k: v.encode("utf-8") for k, v in md.items()})


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite float")
    return out


def _require_positive_finite_float(value: float, name: str) -> float:
    out = _require_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return out


def _coerce_float_tuple(values, expected_len: int, name: str) -> tuple[float, ...]:
    vals = tuple(values)
    if len(vals) != expected_len:
        raise ValueError(f"{name} length mismatch")
    return tuple(_require_finite_float(v, f"{name}[{i}]") for i, v in enumerate(vals))


class DecisionRowWriter:
    def __init__(self, config: WriterConfig) -> None:
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.segments_dir = self.dataset_root / "segments"
        self.manifest_path = self.dataset_root / mf.DEFAULT_MANIFEST_FILENAME
        self.label_spec = config.config.label_spec
        self.feature_columns = mf.feature_columns()
        self.label_columns = mf.label_columns(self.label_spec)
        self.required_columns = mf.required_row_columns(self.label_spec)
        self.schema = arrow_schema(self.label_spec)
        self._columns: dict[str, list[Any]] = {c: [] for c in self.required_columns}
        self.segments: list[mf.StorageSegment] = []
        self.next_row_idx = 0
        self.next_segment_index = 0
        self.previous_decision_index: int | None = None
        self.previous_local_ts_us: int | None = None
        self.segment_min_ts_us: int | None = None
        self.segment_max_ts_us: int | None = None
        self.segment_min_local_ts_us: int | None = None
        self.segment_max_local_ts_us: int | None = None
        self.segment_first_row_idx: int | None = None
        self.segment_last_row_idx: int | None = None
        self.source_files = tuple(config.source_files)
        self._final_manifest: mf.StorageManifest | None = None

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            raise FileExistsError(f"manifest already exists: {self.manifest_path}")
        if any(self.segments_dir.glob("*.parquet")):
            raise FileExistsError("existing parquet segments found")

    def append_values(self, **kwargs) -> None:
        self.append(DecisionRow.from_arrays(**kwargs))

    def append(self, row: DecisionRow) -> None:
        if self._final_manifest is not None:
            raise RuntimeError("writer already finalized")
        self._validate_row(row)
        row_idx = self.next_row_idx
        self._columns[mf.ROW_IDX_COLUMN].append(row_idx)
        self._columns[mf.DECISION_INDEX_COLUMN].append(row.decision_index)
        self._columns[mf.TS_US_COLUMN].append(row.ts_us)
        self._columns[mf.LOCAL_TS_US_COLUMN].append(row.local_ts_us)
        self._columns[mf.EVENT_SEQ_COLUMN].append(row.event_seq)
        self._columns[mf.RAW_MID_COLUMN].append(row.raw_mid)
        self._columns[mf.LABEL_ENTRY_TS_US_COLUMN].append(row.label_entry_ts_us)
        for i, col in enumerate(self.label_columns):
            self._columns[col].append(row.label_values[i])
        for i, col in enumerate(self.feature_columns):
            self._columns[col].append(row.feature_values[i])

        self.next_row_idx += 1
        self.previous_decision_index = row.decision_index
        self.previous_local_ts_us = row.local_ts_us
        self.segment_min_ts_us = row.ts_us if self.segment_min_ts_us is None else min(self.segment_min_ts_us, row.ts_us)
        self.segment_max_ts_us = row.ts_us if self.segment_max_ts_us is None else max(self.segment_max_ts_us, row.ts_us)
        self.segment_min_local_ts_us = row.local_ts_us if self.segment_min_local_ts_us is None else min(self.segment_min_local_ts_us, row.local_ts_us)
        self.segment_max_local_ts_us = row.local_ts_us if self.segment_max_local_ts_us is None else max(self.segment_max_local_ts_us, row.local_ts_us)
        if self.segment_first_row_idx is None:
            self.segment_first_row_idx = row_idx
        self.segment_last_row_idx = row_idx

        if len(self._columns[mf.ROW_IDX_COLUMN]) >= self.config.chunk_rows:
            self.flush()

    def _validate_row(self, row: DecisionRow) -> None:
        decision_index = _require_nonnegative_int(row.decision_index, "decision_index")
        ts_us = _require_positive_int(row.ts_us, "ts_us")
        local_ts_us = _require_positive_int(row.local_ts_us, "local_ts_us")
        event_seq = row.event_seq
        if isinstance(event_seq, bool) or not isinstance(event_seq, int) or event_seq < -1:
            raise ValueError("event_seq must be int >= -1")
        raw_mid = _require_positive_finite_float(row.raw_mid, "raw_mid")
        label_entry_ts_us = _require_positive_int(row.label_entry_ts_us, "label_entry_ts_us")
        if label_entry_ts_us < local_ts_us:
            raise ValueError("label_entry_ts_us must be >= local_ts_us")
        labels = _coerce_float_tuple(row.label_values, len(self.label_columns), "label_values")
        features = _coerce_float_tuple(row.feature_values, len(self.feature_columns), "feature_values")

        if self.previous_decision_index is not None and decision_index <= self.previous_decision_index:
            raise ValueError("decision_index must be strictly increasing")
        if self.previous_local_ts_us is not None and local_ts_us < self.previous_local_ts_us:
            raise ValueError("local_ts_us must be nondecreasing")

        object.__setattr__(row, "raw_mid", raw_mid)
        object.__setattr__(row, "label_values", labels)
        object.__setattr__(row, "feature_values", features)
        object.__setattr__(row, "event_seq", event_seq)
        object.__setattr__(row, "decision_index", decision_index)
        object.__setattr__(row, "ts_us", ts_us)
        object.__setattr__(row, "local_ts_us", local_ts_us)
        object.__setattr__(row, "label_entry_ts_us", label_entry_ts_us)

    def _buffer_to_table(self) -> pa.Table:
        arrays = []
        for field in self.schema:
            arrays.append(pa.array(self._columns[field.name], type=field.type))
        return pa.Table.from_arrays(arrays, schema=self.schema)

    def flush(self) -> None:
        nrows = len(self._columns[mf.ROW_IDX_COLUMN])
        if nrows == 0:
            return
        table = self._buffer_to_table()
        if table.num_rows != nrows:
            raise RuntimeError("buffer/table row count mismatch")

        segment_key = f"seg_{self.next_segment_index:06d}"
        rel_path = f"segments/{segment_key}.parquet"
        target = self.dataset_root / rel_path
        tmp = target.with_suffix(target.suffix + ".tmp")
        pq.write_table(
            table,
            tmp,
            compression=mf.DEFAULT_COMPRESSION,
            version=mf.DEFAULT_PARQUET_VERSION,
            row_group_size=self.config.row_group_rows,
            write_statistics=True,
        )
        tmp.replace(target)
        self.segments.append(
            mf.StorageSegment(
                segment_key=segment_key,
                parquet_path=rel_path,
                row_count=table.num_rows,
                label_count=table.num_rows,
                time_range=TimeRangeUS(self.segment_min_ts_us, self.segment_max_ts_us),
                local_time_range=TimeRangeUS(self.segment_min_local_ts_us, self.segment_max_local_ts_us),
                first_row_idx=self.segment_first_row_idx,
                last_row_idx=self.segment_last_row_idx,
                source_files=self.source_files,
            )
        )
        for col in self._columns.values():
            col.clear()
        self.segment_min_ts_us = None
        self.segment_max_ts_us = None
        self.segment_min_local_ts_us = None
        self.segment_max_local_ts_us = None
        self.segment_first_row_idx = None
        self.segment_last_row_idx = None
        self.next_segment_index += 1

    def finalize(self) -> mf.StorageManifest:
        if self._final_manifest is not None:
            raise RuntimeError("writer already finalized")
        self.flush()
        if not self.segments:
            raise ValueError("cannot finalize empty dataset")
        manifest = mf.make_manifest(
            dataset_id=self.config.dataset_id,
            created_at_utc=self.config.created_at_utc,
            segments=tuple(self.segments),
            config=self.config.config,
            transform_config=self.config.transform_config or {},
            transform_diagnostics=self.config.transform_diagnostics or {},
            splits=(),
            notes=self.config.notes,
        )
        mf.write_manifest_json(manifest, self.manifest_path)
        self._final_manifest = manifest
        return manifest

    def close(self) -> mf.StorageManifest:
        if self._final_manifest is not None:
            return self._final_manifest
        return self.finalize()

    def __enter__(self) -> "DecisionRowWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None and self._final_manifest is None:
            self.finalize()
        return False


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "DEFAULT_ROW_GROUP_ROWS",
    "DecisionRow",
    "WriterConfig",
    "DecisionRowWriter",
    "arrow_schema",
]
