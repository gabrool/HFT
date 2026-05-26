from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from mmrt.config import PipelineConfig, default_config
from mmrt.contracts import AsOfPolicy, LabelSpec, PriceReference, SplitRole, StorageFormat, TimeRangeUS, TimeUnit
from mmrt.features import specs

MANIFEST_SCHEMA_VERSION = "mmrt_storage_manifest_v1"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
ROW_IDX_COLUMN = "row_idx"
DECISION_INDEX_COLUMN = "decision_index"
TS_US_COLUMN = "ts_us"
LOCAL_TS_US_COLUMN = "local_ts_us"
EVENT_SEQ_COLUMN = "event_seq"
RAW_MID_COLUMN = "raw_mid"
LABEL_ENTRY_TS_US_COLUMN = "label_entry_ts_us"
FEATURE_COLUMN_PREFIX = "x_"
LABEL_COLUMN_PREFIX = "y_ret_bps_"
BASE_ROW_COLUMNS = (
    ROW_IDX_COLUMN,
    DECISION_INDEX_COLUMN,
    TS_US_COLUMN,
    LOCAL_TS_US_COLUMN,
    EVENT_SEQ_COLUMN,
    RAW_MID_COLUMN,
    LABEL_ENTRY_TS_US_COLUMN,
)
DEFAULT_COMPRESSION = "zstd"
DEFAULT_PARQUET_VERSION = "2.6"


def _required(m: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in m:
        raise ValueError(f"{where} missing required key {key!r}")
    return m[key]


def _require_keys(m: Mapping[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in m]
    if missing:
        raise ValueError(f"{where} missing required keys {missing!r}")


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _posix_relative_path(value: str, name: str) -> str:
    value = _require_nonempty_str(value, name)
    if "\\" in value or value.startswith("/") or "//" in value:
        raise ValueError(f"{name} must be a relative POSIX path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        raise ValueError(f"{name} has invalid parts")
    return p.as_posix()


def _json_safe(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if isinstance(value, bool):
            raise ValueError(f"{name} contains bool-as-int")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, f"{name}[]") for v in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{name} has non-string key")
            out[k] = _json_safe(v, f"{name}.{k}")
        return out
    raise ValueError(f"{name} is not JSON-safe")


def _canonical_json_bytes(obj: Mapping[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def feature_columns() -> tuple[str, ...]:
    cols = tuple(f"x_{name}" for name in specs.FEATURE_NAMES)
    if len(cols) != specs.FEATURE_COUNT:
        raise ValueError("feature count mismatch")
    return cols


def label_columns(label_spec: LabelSpec) -> tuple[str, ...]:
    ls = LabelSpec(label_spec.horizons_us, label_spec.entry_delay_us, label_spec.price_reference, label_spec.asof_policy)
    return tuple(f"y_ret_bps_{h}us" for h in ls.horizons_us)


def required_row_columns(label_spec: LabelSpec) -> tuple[str, ...]:
    return BASE_ROW_COLUMNS + label_columns(label_spec) + feature_columns()


def feature_schema_record() -> dict[str, Any]:
    return dict(specs.schema_record())


def default_writer_metadata() -> dict[str, Any]:
    return {
        "storage_format": StorageFormat.FLAT_DECISION_ROWS_US_V1.value,
        "compression": DEFAULT_COMPRESSION,
        "parquet_version": DEFAULT_PARQUET_VERSION,
        "feature_column_prefix": FEATURE_COLUMN_PREFIX,
        "label_column_prefix": LABEL_COLUMN_PREFIX,
        "base_row_columns": list(BASE_ROW_COLUMNS),
    }


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json_bytes(payload))


def label_spec_to_dict(label_spec: LabelSpec) -> dict[str, Any]:
    ls = LabelSpec(label_spec.horizons_us, label_spec.entry_delay_us, label_spec.price_reference, label_spec.asof_policy)
    return {
        "horizons_us": list(ls.horizons_us),
        "entry_delay_us": ls.entry_delay_us,
        "price_reference": ls.price_reference.value,
        "asof_policy": ls.asof_policy.value,
        "label_context_us": ls.label_context_us,
    }


def label_spec_from_dict(d: Mapping[str, Any]) -> LabelSpec:
    m = _require_mapping(d, "label_spec")
    return LabelSpec(
        horizons_us=tuple(_required(m, "horizons_us", "label_spec")),
        entry_delay_us=_required(m, "entry_delay_us", "label_spec"),
        price_reference=PriceReference(_required(m, "price_reference", "label_spec")),
        asof_policy=AsOfPolicy(_required(m, "asof_policy", "label_spec")),
    )


def time_range_to_dict(r: TimeRangeUS) -> dict[str, int]:
    if not isinstance(r, TimeRangeUS):
        raise ValueError("time range must be TimeRangeUS")
    return {"start_us": r.start_us, "end_us": r.end_us}


def time_range_from_dict(d: Mapping[str, Any]) -> TimeRangeUS:
    m = _require_mapping(d, "time_range")
    return TimeRangeUS(_required(m, "start_us", "time_range"), _required(m, "end_us", "time_range"))


def pipeline_config_to_manifest_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "exchange": config.market.exchange,
        "symbol": config.market.symbol,
        "source_data_types": [d.value for d in config.data.source_data_types],
        "disabled_context_data_types": [d.value for d in config.data.disabled_context_data_types],
        "decision_policy": config.decision.policy,
        "decision_reason": config.decision.reason.value,
        "decision_stride_us": config.decision.stride_us,
        "horizons_us": list(config.labels.horizons_us),
        "entry_delay_us": config.labels.entry_delay_us,
        "price_reference": config.labels.price_reference.value,
        "asof_policy": config.labels.asof_policy.value,
        "lookback_rows": config.runtime.lookback_rows,
        "feature_dtype": config.runtime.feature_dtype,
        "label_dtype": config.runtime.label_dtype,
        "timestamp_dtype": config.runtime.timestamp_dtype,
        "storage_format": config.storage.storage_format.value,
        "time_unit": config.storage.time_unit.value,
        "pipeline_schema_version": config.storage.pipeline_schema_version,
        "feature_schema_version": config.storage.feature_schema_version,
    }


@dataclass(frozen=True, slots=True)
class StorageSegment:
    segment_key: str
    parquet_path: str
    row_count: int
    label_count: int
    time_range: TimeRangeUS
    local_time_range: TimeRangeUS
    first_row_idx: int
    last_row_idx: int
    source_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_key", _require_nonempty_str(self.segment_key, "segment_key"))
        object.__setattr__(self, "parquet_path", _posix_relative_path(self.parquet_path, "parquet_path"))
        _require_positive_int(self.row_count, "row_count")
        _require_nonnegative_int(self.label_count, "label_count")
        if self.label_count > self.row_count:
            raise ValueError("label_count must be <= row_count")
        if not isinstance(self.time_range, TimeRangeUS):
            raise ValueError("time_range must be TimeRangeUS")
        if not isinstance(self.local_time_range, TimeRangeUS):
            raise ValueError("local_time_range must be TimeRangeUS")
        _require_nonnegative_int(self.first_row_idx, "first_row_idx")
        _require_nonnegative_int(self.last_row_idx, "last_row_idx")
        if self.last_row_idx < self.first_row_idx or self.last_row_idx - self.first_row_idx + 1 != self.row_count:
            raise ValueError("row index range mismatch")
        if isinstance(self.source_files, (str, bytes)):
            raise ValueError("source_files must be an iterable of relative POSIX paths, not a string")
        source_files = tuple(self.source_files)
        object.__setattr__(
            self,
            "source_files",
            tuple(_posix_relative_path(p, f"source_files[{i}]") for i, p in enumerate(source_files)),
        )

    @property
    def start_local_us(self) -> int:
        return self.local_time_range.start_us

    @property
    def end_local_us(self) -> int:
        return self.local_time_range.end_us

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_key": self.segment_key,
            "parquet_path": self.parquet_path,
            "row_count": self.row_count,
            "label_count": self.label_count,
            "time_range": time_range_to_dict(self.time_range),
            "local_time_range": time_range_to_dict(self.local_time_range),
            "first_row_idx": self.first_row_idx,
            "last_row_idx": self.last_row_idx,
            "source_files": list(self.source_files),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StorageSegment":
        m = _require_mapping(d, "segment")
        _require_keys(
            m,
            (
                "segment_key",
                "parquet_path",
                "row_count",
                "label_count",
                "time_range",
                "local_time_range",
                "first_row_idx",
                "last_row_idx",
            ),
            "segment",
        )
        return cls(
            _required(m, "segment_key", "segment"),
            _required(m, "parquet_path", "segment"),
            _required(m, "row_count", "segment"),
            _required(m, "label_count", "segment"),
            time_range_from_dict(_required(m, "time_range", "segment")),
            time_range_from_dict(_required(m, "local_time_range", "segment")),
            _required(m, "first_row_idx", "segment"),
            _required(m, "last_row_idx", "segment"),
            tuple(m.get("source_files", ())),
        )


@dataclass(frozen=True, slots=True)
class SplitMetadata:
    role: SplitRole
    segment_key: str
    start_row: int
    end_row: int
    local_time_range: TimeRangeUS
    embargo_before_us: int = 0
    embargo_after_us: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", SplitRole(self.role))
        object.__setattr__(self, "segment_key", _require_nonempty_str(self.segment_key, "segment_key"))
        _require_nonnegative_int(self.start_row, "start_row")
        _require_nonnegative_int(self.end_row, "end_row")
        if self.end_row <= self.start_row:
            raise ValueError("end_row must be > start_row")
        if not isinstance(self.local_time_range, TimeRangeUS):
            raise ValueError("local_time_range must be TimeRangeUS")
        _require_nonnegative_int(self.embargo_before_us, "embargo_before_us")
        _require_nonnegative_int(self.embargo_after_us, "embargo_after_us")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "segment_key": self.segment_key,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "local_time_range": time_range_to_dict(self.local_time_range),
            "embargo_before_us": self.embargo_before_us,
            "embargo_after_us": self.embargo_after_us,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SplitMetadata":
        m = _require_mapping(d, "split")
        _require_keys(m, ("role", "segment_key", "start_row", "end_row", "local_time_range"), "split")
        return cls(
            _required(m, "role", "split"),
            _required(m, "segment_key", "split"),
            _required(m, "start_row", "split"),
            _required(m, "end_row", "split"),
            time_range_from_dict(_required(m, "local_time_range", "split")),
            m.get("embargo_before_us", 0),
            m.get("embargo_after_us", 0),
        )


@dataclass(frozen=True, slots=True)
class StorageManifest:
    manifest_schema_version: str
    dataset_id: str
    created_at_utc: str
    pipeline_config: dict[str, Any]
    writer_metadata: dict[str, Any]
    feature_schema: dict[str, Any]
    label_spec: LabelSpec
    transform_config: dict[str, Any]
    transform_diagnostics: dict[str, Any]
    exchange: str
    symbol: str
    storage_format: StorageFormat
    time_unit: TimeUnit
    decision_stride_us: int
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    segments: tuple[StorageSegment, ...]
    splits: tuple[SplitMetadata, ...] = ()
    notes: dict[str, Any] | None = None

    def _validate_pipeline_config_consistency(self) -> None:
        pc = self.pipeline_config
        if "feature_schema_version" in pc and pc["feature_schema_version"] != specs.FEATURE_SCHEMA_VERSION:
            raise ValueError("pipeline_config feature_schema_version drift")
        if "time_unit" in pc and pc["time_unit"] != TimeUnit.MICROSECOND.value:
            raise ValueError("pipeline_config time_unit drift")
        if "storage_format" in pc and pc["storage_format"] != StorageFormat.FLAT_DECISION_ROWS_US_V1.value:
            raise ValueError("pipeline_config storage_format drift")
        if "decision_stride_us" in pc and pc["decision_stride_us"] != self.decision_stride_us:
            raise ValueError("pipeline_config decision_stride_us drift")
        if "exchange" in pc and pc["exchange"] != self.exchange:
            raise ValueError("pipeline_config exchange drift")
        if "symbol" in pc and pc["symbol"] != self.symbol:
            raise ValueError("pipeline_config symbol drift")

    def __post_init__(self) -> None:
        if self.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("invalid manifest schema")
        ds = _require_nonempty_str(self.dataset_id, "dataset_id")
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-:" for c in ds):
            raise ValueError("dataset_id contains unsafe chars")
        if not isinstance(self.label_spec, LabelSpec):
            raise ValueError("label_spec must be LabelSpec")
        object.__setattr__(self, "dataset_id", ds)
        object.__setattr__(self, "created_at_utc", _require_nonempty_str(self.created_at_utc, "created_at_utc"))
        object.__setattr__(self, "pipeline_config", _json_safe(dict(_require_mapping(self.pipeline_config, "pipeline_config")), "pipeline_config"))
        object.__setattr__(self, "writer_metadata", _json_safe(dict(_require_mapping(self.writer_metadata, "writer_metadata")), "writer_metadata"))
        object.__setattr__(self, "feature_schema", _json_safe(dict(_require_mapping(self.feature_schema, "feature_schema")), "feature_schema"))
        object.__setattr__(self, "transform_config", _json_safe(dict(_require_mapping(self.transform_config, "transform_config")), "transform_config"))
        object.__setattr__(self, "transform_diagnostics", _json_safe(dict(_require_mapping(self.transform_diagnostics, "transform_diagnostics")), "transform_diagnostics"))
        object.__setattr__(self, "exchange", _require_nonempty_str(self.exchange, "exchange"))
        object.__setattr__(self, "symbol", _require_nonempty_str(self.symbol, "symbol"))
        object.__setattr__(self, "storage_format", StorageFormat(self.storage_format))
        object.__setattr__(self, "time_unit", TimeUnit(self.time_unit))
        if self.storage_format != StorageFormat.FLAT_DECISION_ROWS_US_V1:
            raise ValueError("storage/time/stride invalid")
        if self.time_unit != TimeUnit.MICROSECOND:
            raise ValueError("storage/time/stride invalid")
        if _require_positive_int(self.decision_stride_us, "decision_stride_us") != 500_000:
            raise ValueError("storage/time/stride invalid")
        self._validate_pipeline_config_consistency()

        feature_cols = tuple(self.feature_columns)
        label_cols = tuple(self.label_columns)
        required_cols = tuple(self.required_columns)

        if feature_cols != feature_columns():
            raise ValueError("column schema drift")
        if label_cols != label_columns(self.label_spec):
            raise ValueError("column schema drift")
        if required_cols != required_row_columns(self.label_spec):
            raise ValueError("column schema drift")

        object.__setattr__(self, "feature_columns", feature_cols)
        object.__setattr__(self, "label_columns", label_cols)
        object.__setattr__(self, "required_columns", required_cols)

        self.validate_against_current_code()

        segs = tuple(self.segments)
        if not segs:
            raise ValueError("segments must be non-empty")
        if len({s.segment_key for s in segs}) != len(segs) or len({s.parquet_path for s in segs}) != len(segs):
            raise ValueError("duplicate segment key/path")
        if segs[0].first_row_idx != 0:
            raise ValueError("first segment must start at 0")
        for i, s in enumerate(segs[1:], 1):
            p = segs[i - 1]
            if s.first_row_idx != p.last_row_idx + 1:
                raise ValueError("non contiguous row ranges")
            if s.local_time_range.start_us < p.local_time_range.start_us:
                raise ValueError("nondecreasing local ranges required")
            if s.local_time_range.end_us < p.local_time_range.end_us:
                raise ValueError("nondecreasing local ranges required")
        object.__setattr__(self, "segments", segs)

        seg_map = {s.segment_key: s for s in segs}
        seen = set()
        for sp in tuple(self.splits):
            seg = seg_map.get(sp.segment_key)
            if seg is None:
                raise ValueError("split missing segment")
            if sp.start_row < seg.first_row_idx or sp.end_row > seg.last_row_idx + 1:
                raise ValueError("split rows out of bounds")
            if sp.local_time_range.start_us < seg.local_time_range.start_us or sp.local_time_range.end_us > seg.local_time_range.end_us:
                raise ValueError("split time out of bounds")
            key = (sp.role, sp.segment_key, sp.start_row, sp.end_row)
            if key in seen:
                raise ValueError("duplicate split")
            seen.add(key)
        object.__setattr__(self, "splits", tuple(self.splits))
        object.__setattr__(self, "notes", None if self.notes is None else _json_safe(dict(_require_mapping(self.notes, "notes")), "notes"))

    @property
    def total_rows(self) -> int:
        return sum(s.row_count for s in self.segments)

    @property
    def total_labels(self) -> int:
        return sum(s.label_count for s in self.segments)

    @property
    def segment_keys(self) -> tuple[str, ...]:
        return tuple(s.segment_key for s in self.segments)

    @property
    def parquet_paths(self) -> tuple[str, ...]:
        return tuple(s.parquet_path for s in self.segments)

    @property
    def x_columns(self) -> tuple[str, ...]:
        return self.feature_columns

    @property
    def y_columns(self) -> tuple[str, ...]:
        return self.label_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "dataset_id": self.dataset_id,
            "created_at_utc": self.created_at_utc,
            "pipeline_config": self.pipeline_config,
            "writer_metadata": self.writer_metadata,
            "feature_schema": self.feature_schema,
            "label_spec": label_spec_to_dict(self.label_spec),
            "transform_config": self.transform_config,
            "transform_diagnostics": self.transform_diagnostics,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "storage_format": self.storage_format.value,
            "time_unit": self.time_unit.value,
            "decision_stride_us": self.decision_stride_us,
            "feature_columns": list(self.feature_columns),
            "label_columns": list(self.label_columns),
            "required_columns": list(self.required_columns),
            "segments": [s.to_dict() for s in self.segments],
            "splits": [s.to_dict() for s in self.splits],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StorageManifest":
        m = _require_mapping(d, "manifest")
        _require_keys(
            m,
            (
                "manifest_schema_version",
                "dataset_id",
                "created_at_utc",
                "pipeline_config",
                "writer_metadata",
                "feature_schema",
                "label_spec",
                "transform_config",
                "transform_diagnostics",
                "exchange",
                "symbol",
                "storage_format",
                "time_unit",
                "decision_stride_us",
                "feature_columns",
                "label_columns",
                "required_columns",
                "segments",
            ),
            "manifest",
        )
        return cls(
            _required(m, "manifest_schema_version", "manifest"),
            _required(m, "dataset_id", "manifest"),
            _required(m, "created_at_utc", "manifest"),
            dict(_required(m, "pipeline_config", "manifest")),
            dict(_required(m, "writer_metadata", "manifest")),
            dict(_required(m, "feature_schema", "manifest")),
            label_spec_from_dict(_required(m, "label_spec", "manifest")),
            dict(_required(m, "transform_config", "manifest")),
            dict(_required(m, "transform_diagnostics", "manifest")),
            _required(m, "exchange", "manifest"),
            _required(m, "symbol", "manifest"),
            _required(m, "storage_format", "manifest"),
            _required(m, "time_unit", "manifest"),
            _required(m, "decision_stride_us", "manifest"),
            tuple(_required(m, "feature_columns", "manifest")),
            tuple(_required(m, "label_columns", "manifest")),
            tuple(_required(m, "required_columns", "manifest")),
            tuple(StorageSegment.from_dict(x) for x in _required(m, "segments", "manifest")),
            tuple(SplitMetadata.from_dict(x) for x in m.get("splits", [])),
            m.get("notes"),
        )

    def content_hash(self) -> str:
        return manifest_sha256(self.to_dict())

    def validate_against_current_code(self) -> None:
        fs = self.feature_schema
        checks = {
            "feature_schema_version": specs.FEATURE_SCHEMA_VERSION,
            "feature_count": specs.FEATURE_COUNT,
            "feature_names_hash": specs.FEATURE_NAMES_HASH,
            "feature_specs_hash": specs.FEATURE_SPECS_HASH,
            "feature_dtype": specs.DEFAULT_FEATURE_DTYPE,
            "time_unit": "us",
        }
        for k, v in checks.items():
            if fs.get(k) != v:
                raise ValueError(f"feature_schema[{k}] drift")
        if tuple(self.feature_columns) != feature_columns():
            raise ValueError("column schema drift")
        if tuple(self.label_columns) != label_columns(self.label_spec):
            raise ValueError("column schema drift")
        if tuple(self.required_columns) != required_row_columns(self.label_spec):
            raise ValueError("column schema drift")
        if self.time_unit != TimeUnit.MICROSECOND:
            raise ValueError("time_unit drift")
        if self.storage_format != StorageFormat.FLAT_DECISION_ROWS_US_V1:
            raise ValueError("storage_format drift")
        if self.decision_stride_us != 500_000:
            raise ValueError("decision_stride_us drift")


def make_manifest(
    *,
    dataset_id: str,
    created_at_utc: str,
    segments: tuple[StorageSegment, ...],
    config: PipelineConfig | None = None,
    transform_config: Mapping[str, Any] | None = None,
    transform_diagnostics: Mapping[str, Any] | None = None,
    splits: tuple[SplitMetadata, ...] = (),
    notes: Mapping[str, Any] | None = None,
) -> StorageManifest:
    cfg = default_config() if config is None else config
    return StorageManifest(
        MANIFEST_SCHEMA_VERSION,
        dataset_id,
        created_at_utc,
        pipeline_config_to_manifest_dict(cfg),
        default_writer_metadata(),
        feature_schema_record(),
        cfg.label_spec,
        dict(transform_config or {}),
        dict(transform_diagnostics or {}),
        cfg.market.exchange,
        cfg.market.symbol,
        cfg.storage.storage_format,
        cfg.storage.time_unit,
        cfg.decision.stride_us,
        feature_columns(),
        label_columns(cfg.label_spec),
        required_row_columns(cfg.label_spec),
        segments,
        splits,
        None if notes is None else dict(notes),
    )


def manifest_to_json_bytes(manifest: StorageManifest) -> bytes:
    return _canonical_json_bytes(manifest.to_dict()) + b"\n"


def manifest_from_json_bytes(data: bytes | str) -> StorageManifest:
    return StorageManifest.from_dict(json.loads(data.decode("utf-8") if isinstance(data, bytes) else data))


def write_manifest_json(manifest: StorageManifest, path: str | object) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(manifest_to_json_bytes(manifest))
    tmp.replace(target)


def read_manifest_json(path: str | object) -> StorageManifest:
    from pathlib import Path

    return manifest_from_json_bytes(Path(path).read_bytes())


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "DEFAULT_MANIFEST_FILENAME",
    "ROW_IDX_COLUMN",
    "DECISION_INDEX_COLUMN",
    "TS_US_COLUMN",
    "LOCAL_TS_US_COLUMN",
    "EVENT_SEQ_COLUMN",
    "RAW_MID_COLUMN",
    "LABEL_ENTRY_TS_US_COLUMN",
    "FEATURE_COLUMN_PREFIX",
    "LABEL_COLUMN_PREFIX",
    "BASE_ROW_COLUMNS",
    "DEFAULT_COMPRESSION",
    "DEFAULT_PARQUET_VERSION",
    "StorageSegment",
    "SplitMetadata",
    "StorageManifest",
    "feature_columns",
    "label_columns",
    "required_row_columns",
    "feature_schema_record",
    "default_writer_metadata",
    "label_spec_to_dict",
    "label_spec_from_dict",
    "time_range_to_dict",
    "time_range_from_dict",
    "pipeline_config_to_manifest_dict",
    "make_manifest",
    "manifest_sha256",
    "manifest_to_json_bytes",
    "manifest_from_json_bytes",
    "write_manifest_json",
    "read_manifest_json",
]
