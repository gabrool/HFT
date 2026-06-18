"""Disk-backed adverse-selection dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.execution_tape_writer import NpyChunkWriter

ADVERSE_SELECTION_DATASET_SCHEMA = "mmrt_adverse_selection_dataset_grid_v1"
ADVERSE_SPLIT_CONTRACT_SCHEMA = "mmrt_adverse_split_contract_v1"
_SPLIT_ROLES = ("train", "val", "test")


def _nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _hash64(value: str, name: str) -> str:
    out = _nonempty_str(str(value), name)
    if len(out) != 64 or any(ch not in "0123456789abcdef" for ch in out):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return out


def _json_safe(value: object, name: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} contains non-finite float")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} contains non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{name}[]") for item in value]
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} contains non-string key")
            out[key] = _json_safe(item, f"{name}.{key}")
        return out
    raise ValueError(f"{name} is not JSON-safe")


def _split_contract(value: Mapping[str, object], name: str = "split_contract") -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    out = _json_safe(dict(value), name)
    if not isinstance(out, dict):
        raise ValueError(f"{name} must be a mapping")
    required = (
        "schema",
        "version",
        "split_source_dataset_root",
        "split_source_dataset_id",
        "split_source_manifest_hash",
        "ranges",
        "source_row_counts",
        "adverse_row_counts",
        "decision_grid_schema",
        "decision_grid_hash",
        "decision_grid_n_rows",
        "decision_schedule",
    )
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"{name} missing fields: {missing}")
    if out["schema"] != ADVERSE_SPLIT_CONTRACT_SCHEMA:
        raise ValueError("invalid split_contract schema")
    if int(out["version"]) != 1:
        raise ValueError("invalid split_contract version")
    _nonempty_str(str(out["split_source_dataset_root"]), "split_source_dataset_root")
    _nonempty_str(str(out["split_source_dataset_id"]), "split_source_dataset_id")
    _hash64(str(out["split_source_manifest_hash"]), "split_source_manifest_hash")
    _nonempty_str(str(out["decision_grid_schema"]), "decision_grid_schema")
    _hash64(str(out["decision_grid_hash"]), "decision_grid_hash")
    _positive_int(int(out["decision_grid_n_rows"]), "decision_grid_n_rows")
    if not isinstance(out["decision_schedule"], Mapping):
        raise ValueError("split_contract decision_schedule must be a mapping")
    ranges = out["ranges"]
    if not isinstance(ranges, Mapping):
        raise ValueError("split_contract ranges must be a mapping")
    source_counts = out["source_row_counts"]
    adverse_counts = out["adverse_row_counts"]
    if not isinstance(source_counts, Mapping) or not isinstance(adverse_counts, Mapping):
        raise ValueError("split_contract row counts must be mappings")
    for role in _SPLIT_ROLES:
        entries = ranges.get(role)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"split_contract ranges must include {role}")
        _nonnegative_int(int(source_counts.get(role, -1)), f"source_row_counts.{role}")
        _nonnegative_int(int(adverse_counts.get(role, -1)), f"adverse_row_counts.{role}")
        for i, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"split_contract ranges.{role}[{i}] must be a mapping")
            start = _nonnegative_int(int(entry.get("start_local_ts_us", -1)), f"ranges.{role}[{i}].start_local_ts_us")
            end = _nonnegative_int(int(entry.get("end_local_ts_us", -1)), f"ranges.{role}[{i}].end_local_ts_us")
            if end <= start:
                raise ValueError(f"ranges.{role}[{i}] end_local_ts_us must be greater than start_local_ts_us")
            _positive_int(int(entry.get("row_count", 0)), f"ranges.{role}[{i}].row_count")
    _nonnegative_int(int(adverse_counts.get("out_of_split", -1)), "adverse_row_counts.out_of_split")
    return out


def _name_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    out = tuple(str(v) for v in values)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if any(not v for v in out):
        raise ValueError(f"{name} entries must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} entries must be unique")
    return out


@dataclass(frozen=True, slots=True)
class AdverseSelectionDatasetManifest:
    schema: str
    exchange: str
    symbol: str
    tape_schema: str
    tape_num_events: int
    tape_num_l2_batches: int
    tape_num_trades: int
    tape_start_local_ts_us: int
    tape_end_local_ts_us: int
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: Mapping[str, object]
    split_source_dataset_root: str
    split_source_dataset_id: str
    split_source_manifest_hash: str
    split_contract: Mapping[str, object]
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    num_rows: int
    num_features: int
    num_labels: int
    config_json: str
    index_schema: str
    index_manifest_sha256: str
    index_root: str
    created_at_utc: str

    def __post_init__(self) -> None:
        if self.schema != ADVERSE_SELECTION_DATASET_SCHEMA:
            raise ValueError("invalid adverse-selection dataset schema")
        object.__setattr__(self, "exchange", _nonempty_str(self.exchange, "exchange"))
        object.__setattr__(self, "symbol", _nonempty_str(self.symbol, "symbol"))
        object.__setattr__(self, "tape_schema", _nonempty_str(self.tape_schema, "tape_schema"))
        for name in ("tape_num_events", "tape_num_l2_batches", "tape_num_trades", "num_rows"):
            object.__setattr__(self, name, _nonnegative_int(int(getattr(self, name)), name))
        object.__setattr__(self, "decision_grid_schema", _nonempty_str(self.decision_grid_schema, "decision_grid_schema"))
        object.__setattr__(self, "decision_grid_hash", _hash64(self.decision_grid_hash, "decision_grid_hash"))
        object.__setattr__(self, "decision_grid_n_rows", _positive_int(int(self.decision_grid_n_rows), "decision_grid_n_rows"))
        if self.num_rows > self.decision_grid_n_rows:
            raise ValueError("num_rows cannot exceed decision_grid_n_rows")
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")
        object.__setattr__(self, "decision_schedule", dict(self.decision_schedule))
        split_contract = _split_contract(self.split_contract)
        object.__setattr__(self, "split_source_dataset_root", _nonempty_str(self.split_source_dataset_root, "split_source_dataset_root"))
        object.__setattr__(self, "split_source_dataset_id", _nonempty_str(self.split_source_dataset_id, "split_source_dataset_id"))
        object.__setattr__(self, "split_source_manifest_hash", _hash64(self.split_source_manifest_hash, "split_source_manifest_hash"))
        if split_contract["split_source_dataset_root"] != self.split_source_dataset_root:
            raise ValueError("split_source_dataset_root must match split_contract")
        if split_contract["split_source_dataset_id"] != self.split_source_dataset_id:
            raise ValueError("split_source_dataset_id must match split_contract")
        if split_contract["split_source_manifest_hash"] != self.split_source_manifest_hash:
            raise ValueError("split_source_manifest_hash must match split_contract")
        if split_contract["decision_grid_schema"] != self.decision_grid_schema:
            raise ValueError("split_contract decision_grid_schema mismatch")
        if split_contract["decision_grid_hash"] != self.decision_grid_hash:
            raise ValueError("split_contract decision_grid_hash mismatch")
        if int(split_contract["decision_grid_n_rows"]) != self.decision_grid_n_rows:
            raise ValueError("split_contract decision_grid_n_rows mismatch")
        if dict(split_contract["decision_schedule"]) != dict(self.decision_schedule):  # type: ignore[arg-type]
            raise ValueError("split_contract decision_schedule mismatch")
        object.__setattr__(self, "split_contract", split_contract)
        features = _name_tuple(self.feature_names, "feature_names")
        labels = _name_tuple(self.label_names, "label_names")
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "label_names", labels)
        object.__setattr__(self, "num_features", _nonnegative_int(int(self.num_features), "num_features"))
        object.__setattr__(self, "num_labels", _nonnegative_int(int(self.num_labels), "num_labels"))
        if self.num_features != len(features) or self.num_labels != len(labels):
            raise ValueError("manifest feature/label counts must match names")
        object.__setattr__(self, "config_json", str(self.config_json))
        object.__setattr__(self, "index_schema", _nonempty_str(self.index_schema, "index_schema"))
        object.__setattr__(self, "index_manifest_sha256", _nonempty_str(self.index_manifest_sha256, "index_manifest_sha256"))
        object.__setattr__(self, "index_root", _nonempty_str(self.index_root, "index_root"))
        object.__setattr__(self, "created_at_utc", _nonempty_str(self.created_at_utc, "created_at_utc"))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "tape_schema": self.tape_schema,
            "tape_num_events": self.tape_num_events,
            "tape_num_l2_batches": self.tape_num_l2_batches,
            "tape_num_trades": self.tape_num_trades,
            "tape_start_local_ts_us": self.tape_start_local_ts_us,
            "tape_end_local_ts_us": self.tape_end_local_ts_us,
            "decision_grid_schema": self.decision_grid_schema,
            "decision_grid_hash": self.decision_grid_hash,
            "decision_grid_n_rows": self.decision_grid_n_rows,
            "decision_schedule": dict(self.decision_schedule),
            "split_source_dataset_root": self.split_source_dataset_root,
            "split_source_dataset_id": self.split_source_dataset_id,
            "split_source_manifest_hash": self.split_source_manifest_hash,
            "split_contract": dict(self.split_contract),
            "feature_names": list(self.feature_names),
            "label_names": list(self.label_names),
            "num_rows": self.num_rows,
            "num_features": self.num_features,
            "num_labels": self.num_labels,
            "config_json": self.config_json,
            "index_schema": self.index_schema,
            "index_manifest_sha256": self.index_manifest_sha256,
            "index_root": self.index_root,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AdverseSelectionDatasetManifest":
        return cls(
            schema=str(raw["schema"]), exchange=str(raw["exchange"]), symbol=str(raw["symbol"]), tape_schema=str(raw["tape_schema"]),
            tape_num_events=int(raw["tape_num_events"]), tape_num_l2_batches=int(raw["tape_num_l2_batches"]), tape_num_trades=int(raw["tape_num_trades"]),
            tape_start_local_ts_us=int(raw["tape_start_local_ts_us"]), tape_end_local_ts_us=int(raw["tape_end_local_ts_us"]),
            decision_grid_schema=str(raw["decision_grid_schema"]),
            decision_grid_hash=str(raw["decision_grid_hash"]),
            decision_grid_n_rows=int(raw["decision_grid_n_rows"]),
            decision_schedule=dict(raw["decision_schedule"]),  # type: ignore[arg-type]
            split_source_dataset_root=str(raw["split_source_dataset_root"]),
            split_source_dataset_id=str(raw["split_source_dataset_id"]),
            split_source_manifest_hash=str(raw["split_source_manifest_hash"]),
            split_contract=dict(raw["split_contract"]),  # type: ignore[arg-type]
            feature_names=tuple(raw["feature_names"]), label_names=tuple(raw["label_names"]), num_rows=int(raw["num_rows"]),
            num_features=int(raw["num_features"]), num_labels=int(raw["num_labels"]), config_json=str(raw.get("config_json", "{}")),
            index_schema=str(raw["index_schema"]), index_manifest_sha256=str(raw["index_manifest_sha256"]), index_root=str(raw["index_root"]),
            created_at_utc=str(raw["created_at_utc"]),
        )


@dataclass(frozen=True, slots=True)
class AdverseSelectionDatasetArrays:
    decision_local_ts_us: np.ndarray
    decision_event_index: np.ndarray
    decision_event_seq: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    label_masks: np.ndarray


@dataclass(frozen=True, slots=True)
class DiskBackedAdverseSelectionDataset:
    root: Path
    manifest: AdverseSelectionDatasetManifest
    arrays: AdverseSelectionDatasetArrays

    @property
    def num_rows(self) -> int:
        return self.manifest.num_rows

    @property
    def num_decisions(self) -> int:
        return self.manifest.num_rows

    @property
    def num_features(self) -> int:
        return self.manifest.num_features

    @property
    def num_labels(self) -> int:
        return self.manifest.num_labels

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.manifest.feature_names

    @property
    def label_names(self) -> tuple[str, ...]:
        return self.manifest.label_names


def _validate_arrays(manifest: AdverseSelectionDatasetManifest, arrays: AdverseSelectionDatasetArrays) -> None:
    n = manifest.num_rows
    checks = [
        (arrays.decision_local_ts_us, np.dtype("int64"), (n,), "decision_local_ts_us"),
        (arrays.decision_event_index, np.dtype("int64"), (n,), "decision_event_index"),
        (arrays.decision_event_seq, np.dtype("int64"), (n,), "decision_event_seq"),
        (arrays.features, np.dtype("float32"), (n, manifest.num_features), "features"),
        (arrays.labels, np.dtype("float32"), (n, manifest.num_labels), "labels"),
        (arrays.label_masks, np.dtype("bool"), (n, manifest.num_labels), "label_masks"),
    ]
    for arr, dtype, shape, name in checks:
        if arr.dtype != dtype or arr.shape != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")


@dataclass(frozen=True, slots=True)
class AdverseSelectionDatasetWriterConfig:
    output_root: str
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    manifest_metadata: Mapping[str, object]
    chunk_rows: int = 100_000
    overwrite: bool = False
    cleanup_chunks: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", _nonempty_str(self.output_root, "output_root"))
        object.__setattr__(self, "feature_names", _name_tuple(self.feature_names, "feature_names"))
        object.__setattr__(self, "label_names", _name_tuple(self.label_names, "label_names"))
        object.__setattr__(self, "chunk_rows", _positive_int(int(self.chunk_rows), "chunk_rows"))
        if not isinstance(self.overwrite, bool) or not isinstance(self.cleanup_chunks, bool):
            raise ValueError("overwrite and cleanup_chunks must be bool")


class AdverseSelectionDatasetWriter:
    def __init__(self, config: AdverseSelectionDatasetWriterConfig):
        self.config = config
        self.root = Path(config.output_root)
        if self.root.exists() and not config.overwrite:
            raise FileExistsError(f"adverse dataset root exists: {self.root}")
        self.arrays_dir = self.root / "arrays"
        self.chunks_dir = self.root / "chunks"
        if self.root.exists() and config.overwrite:
            import shutil
            shutil.rmtree(self.root)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        nf = len(config.feature_names)
        nl = len(config.label_names)
        self._writers = {
            "decision_local_ts_us": NpyChunkWriter("decision_local_ts_us", np.int64, (), config.chunk_rows, self.chunks_dir),
            "decision_event_index": NpyChunkWriter("decision_event_index", np.int64, (), config.chunk_rows, self.chunks_dir),
            "decision_event_seq": NpyChunkWriter("decision_event_seq", np.int64, (), config.chunk_rows, self.chunks_dir),
            "features": NpyChunkWriter("features", np.float32, (nf,), config.chunk_rows, self.chunks_dir),
            "labels": NpyChunkWriter("labels", np.float32, (nl,), config.chunk_rows, self.chunks_dir),
            "label_masks": NpyChunkWriter("label_masks", np.bool_, (nl,), config.chunk_rows, self.chunks_dir),
        }
        self._rows = 0

    def append(self, *, decision_local_ts_us: int, decision_event_index: int, decision_event_seq: int, features: Sequence[float] | np.ndarray, labels: Sequence[float] | np.ndarray, label_masks: Sequence[bool] | np.ndarray) -> int:
        f = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        m = np.asarray(label_masks, dtype=np.bool_)
        if f.shape != (len(self.config.feature_names),):
            raise ValueError("features width mismatch")
        if y.shape != (len(self.config.label_names),) or m.shape != y.shape:
            raise ValueError("labels/masks width mismatch")
        row_index = self._rows
        self._writers["decision_local_ts_us"].append(int(decision_local_ts_us))
        self._writers["decision_event_index"].append(int(decision_event_index))
        self._writers["decision_event_seq"].append(int(decision_event_seq))
        self._writers["features"].append(f)
        self._writers["labels"].append(y)
        self._writers["label_masks"].append(m)
        self._rows += 1
        return row_index

    def append_many(self, *, decision_local_ts_us, decision_event_index, decision_event_seq, features, labels, label_masks) -> tuple[int, int]:
        ts = np.asarray(decision_local_ts_us, dtype=np.int64)
        idx = np.asarray(decision_event_index, dtype=np.int64)
        seq = np.asarray(decision_event_seq, dtype=np.int64)
        f = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        m = np.asarray(label_masks, dtype=np.bool_)
        if ts.ndim != 1:
            raise ValueError("decision_local_ts_us must be 1D")
        n = int(ts.shape[0])
        if idx.shape != (n,) or seq.shape != (n,):
            raise ValueError("decision arrays length mismatch")
        if f.shape != (n, len(self.config.feature_names)):
            raise ValueError("features width mismatch")
        if y.shape != (n, len(self.config.label_names)) or m.shape != y.shape:
            raise ValueError("labels/masks width mismatch")
        start = self._rows
        self._writers["decision_local_ts_us"].append_many(ts)
        self._writers["decision_event_index"].append_many(idx)
        self._writers["decision_event_seq"].append_many(seq)
        self._writers["features"].append_many(f)
        self._writers["labels"].append_many(y)
        self._writers["label_masks"].append_many(m)
        self._rows += n
        return start, self._rows

    def finalize(self) -> DiskBackedAdverseSelectionDataset:
        manifest_path = self.root / "manifest.json"
        manifest_path.unlink(missing_ok=True)
        n = self._rows
        nf = len(self.config.feature_names)
        nl = len(self.config.label_names)
        try:
            for name, writer in self._writers.items():
                rows = writer.finalize(self.arrays_dir / f"{name}.npy")
                if rows != n:
                    raise RuntimeError(f"chunk row count mismatch for {name}")
            meta = dict(self.config.manifest_metadata)
            manifest = AdverseSelectionDatasetManifest(
                schema=ADVERSE_SELECTION_DATASET_SCHEMA,
                exchange=str(meta["exchange"]), symbol=str(meta["symbol"]), tape_schema=str(meta["tape_schema"]),
                tape_num_events=int(meta["tape_num_events"]), tape_num_l2_batches=int(meta["tape_num_l2_batches"]), tape_num_trades=int(meta["tape_num_trades"]),
                tape_start_local_ts_us=int(meta["tape_start_local_ts_us"]), tape_end_local_ts_us=int(meta["tape_end_local_ts_us"]),
                decision_grid_schema=str(meta["decision_grid_schema"]),
                decision_grid_hash=str(meta["decision_grid_hash"]),
                decision_grid_n_rows=int(meta["decision_grid_n_rows"]),
                decision_schedule=dict(meta["decision_schedule"]),  # type: ignore[arg-type]
                split_source_dataset_root=str(meta["split_source_dataset_root"]),
                split_source_dataset_id=str(meta["split_source_dataset_id"]),
                split_source_manifest_hash=str(meta["split_source_manifest_hash"]),
                split_contract=dict(meta["split_contract"]),  # type: ignore[arg-type]
                feature_names=self.config.feature_names, label_names=self.config.label_names, num_rows=n, num_features=nf, num_labels=nl,
                config_json=str(meta.get("config_json", "{}")), index_schema=str(meta["index_schema"]), index_manifest_sha256=str(meta["index_manifest_sha256"]), index_root=str(meta["index_root"]), created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            tmp = self.root / "manifest.json.tmp"
            tmp.write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
            tmp.replace(manifest_path)
            if self.config.cleanup_chunks:
                for writer in self._writers.values():
                    writer.cleanup()
                import shutil
                shutil.rmtree(self.chunks_dir, ignore_errors=True)
            return load_adverse_selection_dataset(self.root, mmap_mode="r")
        except Exception:
            manifest_path.unlink(missing_ok=True)
            (self.root / "manifest.json.tmp").unlink(missing_ok=True)
            raise

def load_adverse_selection_dataset(root: str | Path, *, mmap_mode: str | None = "r") -> DiskBackedAdverseSelectionDataset:
    if mmap_mode not in (None, "r"):
        raise ValueError("mmap_mode must be None or 'r'")
    root = Path(root)
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest = AdverseSelectionDatasetManifest.from_dict(raw)
    arrays_dir = root / "arrays"
    arrays = AdverseSelectionDatasetArrays(
        decision_local_ts_us=np.load(arrays_dir / "decision_local_ts_us.npy", mmap_mode=mmap_mode),
        decision_event_index=np.load(arrays_dir / "decision_event_index.npy", mmap_mode=mmap_mode),
        decision_event_seq=np.load(arrays_dir / "decision_event_seq.npy", mmap_mode=mmap_mode),
        features=np.load(arrays_dir / "features.npy", mmap_mode=mmap_mode),
        labels=np.load(arrays_dir / "labels.npy", mmap_mode=mmap_mode),
        label_masks=np.load(arrays_dir / "label_masks.npy", mmap_mode=mmap_mode),
    )
    _validate_arrays(manifest, arrays)
    return DiskBackedAdverseSelectionDataset(root=root, manifest=manifest, arrays=arrays)


def write_adverse_selection_dataset_manifest(root: str | Path, manifest: AdverseSelectionDatasetManifest) -> None:
    if not isinstance(manifest, AdverseSelectionDatasetManifest):
        raise ValueError("manifest must be AdverseSelectionDatasetManifest")
    root = Path(root)
    target = root / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)


def adverse_selection_dataset_manifest_with_split_contract(
    manifest: AdverseSelectionDatasetManifest,
    split_contract: Mapping[str, object],
) -> AdverseSelectionDatasetManifest:
    if not isinstance(manifest, AdverseSelectionDatasetManifest):
        raise ValueError("manifest must be AdverseSelectionDatasetManifest")
    contract = _split_contract(split_contract)
    return AdverseSelectionDatasetManifest(
        schema=manifest.schema,
        exchange=manifest.exchange,
        symbol=manifest.symbol,
        tape_schema=manifest.tape_schema,
        tape_num_events=manifest.tape_num_events,
        tape_num_l2_batches=manifest.tape_num_l2_batches,
        tape_num_trades=manifest.tape_num_trades,
        tape_start_local_ts_us=manifest.tape_start_local_ts_us,
        tape_end_local_ts_us=manifest.tape_end_local_ts_us,
        decision_grid_schema=manifest.decision_grid_schema,
        decision_grid_hash=manifest.decision_grid_hash,
        decision_grid_n_rows=manifest.decision_grid_n_rows,
        decision_schedule=manifest.decision_schedule,
        split_source_dataset_root=str(contract["split_source_dataset_root"]),
        split_source_dataset_id=str(contract["split_source_dataset_id"]),
        split_source_manifest_hash=str(contract["split_source_manifest_hash"]),
        split_contract=contract,
        feature_names=manifest.feature_names,
        label_names=manifest.label_names,
        num_rows=manifest.num_rows,
        num_features=manifest.num_features,
        num_labels=manifest.num_labels,
        config_json=manifest.config_json,
        index_schema=manifest.index_schema,
        index_manifest_sha256=manifest.index_manifest_sha256,
        index_root=manifest.index_root,
        created_at_utc=manifest.created_at_utc,
    )


def estimate_adverse_dataset_bytes(*, num_decisions_estimate: int, num_features: int, num_labels: int) -> dict[str, int]:
    n = _nonnegative_int(int(num_decisions_estimate), "num_decisions_estimate")
    nf = _nonnegative_int(int(num_features), "num_features")
    nl = _nonnegative_int(int(num_labels), "num_labels")
    decision_arrays = 3 * n * 8
    features = n * nf * 4
    labels = n * nl * 4
    masks = n * nl
    final = decision_arrays + features + labels + masks
    return {"decision_arrays": decision_arrays, "features": features, "labels": labels, "label_masks": masks, "estimated_final_bytes": final, "estimated_with_temp_overhead_bytes": int(final * 1.25)}
