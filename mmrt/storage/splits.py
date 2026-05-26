from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import bisect

import pyarrow as pa

from mmrt.contracts import SplitRole, TimeRangeUS
from mmrt.storage import manifest as mf
from mmrt.storage.reader import StorageDatasetReader, open_dataset

DEFAULT_SPLIT_BATCH_SIZE = 65_536


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _coerce_window_tuple(windows: Iterable["SplitWindow"]) -> tuple["SplitWindow", ...]:
    if isinstance(windows, SplitWindow):
        return (windows,)
    out: list[SplitWindow] = []
    for i, w in enumerate(tuple(windows)):
        if not isinstance(w, SplitWindow):
            raise ValueError(f"windows[{i}] must be SplitWindow")
        out.append(w)
    return tuple(out)


def _resolve_purge_embargo(reader: StorageDatasetReader, config: "SplitConfig") -> tuple[int, int, int, int]:
    label_context = reader.manifest.label_spec.label_context_us
    purge_before = label_context if config.purge_before_us is None else config.purge_before_us
    purge_after = label_context if config.purge_after_us is None else config.purge_after_us
    embargo_before = purge_before if config.embargo_before_us is None else config.embargo_before_us
    embargo_after = purge_after if config.embargo_after_us is None else config.embargo_after_us
    return purge_before, purge_after, embargo_before, embargo_after


def _assert_nonoverlapping_windows(windows: tuple["SplitWindow", ...]) -> None:
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        cur = windows[i]
        if cur.start_local_ts_us < prev.end_local_ts_us:
            raise ValueError("windows must be chronological and non-overlapping")


def _assert_nonoverlapping_entries(entries: tuple[mf.SplitMetadata, ...]) -> None:
    ordered = sorted(entries, key=lambda e: (e.start_row, e.end_row, e.role.value, e.segment_key))
    for i in range(1, len(ordered)):
        a = ordered[i - 1]
        b = ordered[i]
        if max(a.start_row, b.start_row) < min(a.end_row, b.end_row):
            raise ValueError("split entries overlap in row space")


def _lower_bound(values: list[int], target: int) -> int:
    return bisect.bisect_left(values, target)


def _unique_roles(entries: tuple[mf.SplitMetadata, ...]) -> tuple[SplitRole, ...]:
    out: list[SplitRole] = []
    seen: set[SplitRole] = set()
    for e in entries:
        if e.role not in seen:
            seen.add(e.role)
            out.append(e.role)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class SplitWindow:
    role: SplitRole
    start_local_ts_us: int
    end_local_ts_us: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", SplitRole(self.role))
        _require_positive_int(self.start_local_ts_us, "start_local_ts_us")
        _require_positive_int(self.end_local_ts_us, "end_local_ts_us")
        if self.end_local_ts_us <= self.start_local_ts_us:
            raise ValueError("end_local_ts_us must be > start_local_ts_us")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    windows: tuple[SplitWindow, ...]
    purge_before_us: int | None = None
    purge_after_us: int | None = None
    embargo_before_us: int | None = None
    embargo_after_us: int | None = None
    min_rows_per_split: int = 1
    allow_empty_roles: bool = False
    validate_dataset_on_open: bool = True
    batch_size: int = DEFAULT_SPLIT_BATCH_SIZE

    def __post_init__(self) -> None:
        windows = _coerce_window_tuple(self.windows)
        if not windows:
            raise ValueError("windows must be non-empty")
        _assert_nonoverlapping_windows(windows)
        object.__setattr__(self, "windows", windows)
        if self.purge_before_us is not None:
            _require_nonnegative_int(self.purge_before_us, "purge_before_us")
        if self.purge_after_us is not None:
            _require_nonnegative_int(self.purge_after_us, "purge_after_us")
        if self.embargo_before_us is not None:
            _require_nonnegative_int(self.embargo_before_us, "embargo_before_us")
        if self.embargo_after_us is not None:
            _require_nonnegative_int(self.embargo_after_us, "embargo_after_us")
        _require_positive_int(self.min_rows_per_split, "min_rows_per_split")
        _require_bool(self.allow_empty_roles, "allow_empty_roles")
        _require_bool(self.validate_dataset_on_open, "validate_dataset_on_open")
        _require_positive_int(self.batch_size, "batch_size")


@dataclass(frozen=True, slots=True)
class SplitPlan:
    dataset_id: str
    entries: tuple[mf.SplitMetadata, ...]
    purge_before_us: int
    purge_after_us: int
    embargo_before_us: int
    embargo_after_us: int
    source_windows: tuple[SplitWindow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _require_nonempty_str(self.dataset_id, "dataset_id"))
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "source_windows", _coerce_window_tuple(self.source_windows))
        _require_nonnegative_int(self.purge_before_us, "purge_before_us")
        _require_nonnegative_int(self.purge_after_us, "purge_after_us")
        _require_nonnegative_int(self.embargo_before_us, "embargo_before_us")
        _require_nonnegative_int(self.embargo_after_us, "embargo_after_us")
        for i, entry in enumerate(entries):
            if not isinstance(entry, mf.SplitMetadata):
                raise ValueError(f"entries[{i}] must be SplitMetadata")
            if entry.embargo_before_us != self.embargo_before_us or entry.embargo_after_us != self.embargo_after_us:
                raise ValueError("entry embargo values must match SplitPlan embargo values")
        _assert_nonoverlapping_entries(entries)

    @property
    def roles(self) -> tuple[SplitRole, ...]:
        return _unique_roles(self.entries)

    @property
    def total_rows(self) -> int:
        return sum(e.end_row - e.start_row for e in self.entries)

    def entries_for(self, role: SplitRole | str) -> tuple[mf.SplitMetadata, ...]:
        role_enum = SplitRole(role)
        return tuple(e for e in self.entries if e.role == role_enum)


def chronological_windows(*, train: tuple[int, int], val: tuple[int, int], test: tuple[int, int] | None = None) -> tuple[SplitWindow, ...]:
    windows: list[SplitWindow] = [
        SplitWindow(role=SplitRole.TRAIN, start_local_ts_us=train[0], end_local_ts_us=train[1]),
        SplitWindow(role=SplitRole.VAL, start_local_ts_us=val[0], end_local_ts_us=val[1]),
    ]
    if test is not None:
        windows.append(SplitWindow(role=SplitRole.TEST, start_local_ts_us=test[0], end_local_ts_us=test[1]))
    return tuple(windows)


def build_split_plan(dataset_root: str, config: SplitConfig) -> SplitPlan:
    reader = open_dataset(dataset_root, validate_on_open=config.validate_dataset_on_open, batch_size=config.batch_size)
    purge_before, purge_after, embargo_before, embargo_after = _resolve_purge_embargo(reader, config)
    table = reader.read_table(columns=(mf.ROW_IDX_COLUMN, mf.LOCAL_TS_US_COLUMN))
    if not isinstance(table, pa.Table):
        raise ValueError("reader.read_table must return pyarrow.Table")
    row_idx = table[mf.ROW_IDX_COLUMN].to_pylist()
    local_ts = table[mf.LOCAL_TS_US_COLUMN].to_pylist()
    if len(row_idx) != reader.total_rows or len(local_ts) != reader.total_rows:
        raise ValueError("row count mismatch")
    if row_idx != list(range(reader.total_rows)):
        raise ValueError("row_idx must be contiguous from 0..total_rows-1")
    prev: int | None = None
    for i, ts in enumerate(local_ts):
        _require_positive_int(ts, f"local_ts_us[{i}]")
        if prev is not None and ts < prev:
            raise ValueError("local_ts_us must be nondecreasing")
        prev = ts

    entries: list[mf.SplitMetadata] = []
    for window in config.windows:
        effective_start = window.start_local_ts_us + purge_before + embargo_before
        effective_end = window.end_local_ts_us - purge_after - embargo_after
        if effective_end <= effective_start:
            if not config.allow_empty_roles:
                raise ValueError(f"window {window.role.value} has no rows after purge/embargo")
            continue
        start_pos = _lower_bound(local_ts, effective_start)
        end_pos = _lower_bound(local_ts, effective_end)
        if end_pos <= start_pos:
            if not config.allow_empty_roles:
                raise ValueError(f"window {window.role.value} has no assigned rows")
            continue
        if end_pos - start_pos < config.min_rows_per_split:
            if not config.allow_empty_roles:
                raise ValueError(f"window {window.role.value} has too few rows")
            continue

        for seg in reader.manifest.segments:
            seg_start = max(start_pos, seg.first_row_idx)
            seg_end = min(end_pos, seg.last_row_idx + 1)
            if seg_end <= seg_start:
                continue
            entry_local_start = local_ts[seg_start]
            entry_local_end = local_ts[seg_end - 1] + 1
            if entry_local_start < effective_start or entry_local_end > effective_end:
                raise ValueError("entry local_time_range outside effective window")
            entries.append(
                mf.SplitMetadata(
                    role=window.role,
                    segment_key=seg.segment_key,
                    start_row=seg_start,
                    end_row=seg_end,
                    local_time_range=TimeRangeUS(entry_local_start, entry_local_end),
                    embargo_before_us=embargo_before,
                    embargo_after_us=embargo_after,
                )
            )

    entries_tuple = tuple(entries)
    _assert_nonoverlapping_entries(entries_tuple)
    starts = [e.start_row for e in entries_tuple]
    if starts != sorted(starts):
        raise ValueError("entries must be nondecreasing by start_row")

    if not config.allow_empty_roles:
        for w in config.windows:
            if not any(e.role == w.role for e in entries_tuple):
                raise ValueError(f"role {w.role.value} has no split entries")

    return SplitPlan(
        dataset_id=reader.manifest.dataset_id,
        entries=entries_tuple,
        purge_before_us=purge_before,
        purge_after_us=purge_after,
        embargo_before_us=embargo_before,
        embargo_after_us=embargo_after,
        source_windows=config.windows,
    )


def apply_split_plan(manifest: mf.StorageManifest, plan: SplitPlan, *, replace_existing: bool = True) -> mf.StorageManifest:
    if manifest.dataset_id != plan.dataset_id:
        raise ValueError("dataset_id mismatch")
    _require_bool(replace_existing, "replace_existing")

    if replace_existing:
        new_splits = tuple(plan.entries)
    else:
        new_splits = tuple(manifest.splits) + tuple(plan.entries)

    _assert_nonoverlapping_entries(new_splits)

    return mf.StorageManifest(
        manifest_schema_version=manifest.manifest_schema_version,
        dataset_id=manifest.dataset_id,
        created_at_utc=manifest.created_at_utc,
        pipeline_config=manifest.pipeline_config,
        writer_metadata=manifest.writer_metadata,
        feature_schema=manifest.feature_schema,
        label_spec=manifest.label_spec,
        transform_config=manifest.transform_config,
        transform_diagnostics=manifest.transform_diagnostics,
        exchange=manifest.exchange,
        symbol=manifest.symbol,
        storage_format=manifest.storage_format,
        time_unit=manifest.time_unit,
        decision_stride_us=manifest.decision_stride_us,
        feature_columns=manifest.feature_columns,
        label_columns=manifest.label_columns,
        required_columns=manifest.required_columns,
        segments=manifest.segments,
        splits=new_splits,
        notes=manifest.notes,
    )


def write_split_manifest(dataset_root: str, plan: SplitPlan, *, replace_existing: bool = True) -> mf.StorageManifest:
    root = Path(dataset_root)
    manifest_path = root / mf.DEFAULT_MANIFEST_FILENAME
    manifest = mf.read_manifest_json(manifest_path)
    updated = apply_split_plan(manifest, plan, replace_existing=replace_existing)
    mf.write_manifest_json(updated, manifest_path)
    return updated


def build_and_write_splits(dataset_root: str, config: SplitConfig, *, replace_existing: bool = True) -> mf.StorageManifest:
    plan = build_split_plan(dataset_root, config)
    return write_split_manifest(dataset_root, plan, replace_existing=replace_existing)


__all__ = [
    "DEFAULT_SPLIT_BATCH_SIZE",
    "SplitWindow",
    "SplitConfig",
    "SplitPlan",
    "chronological_windows",
    "build_split_plan",
    "apply_split_plan",
    "write_split_manifest",
    "build_and_write_splits",
]
