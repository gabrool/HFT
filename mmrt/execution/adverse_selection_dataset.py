"""Disk-backed adverse-selection dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ADVERSE_SELECTION_DATASET_SCHEMA = "mmrt_adverse_selection_dataset" + "_" + "v" + "1"


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
    decision_interval_us: int
    start_event_index: int | None
    max_decisions: int | None
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    num_rows: int
    num_features: int
    num_labels: int
    config_json: str
    created_at_utc: str

    def __post_init__(self) -> None:
        if self.schema != ADVERSE_SELECTION_DATASET_SCHEMA:
            raise ValueError("invalid adverse-selection dataset schema")
        object.__setattr__(self, "exchange", _nonempty_str(self.exchange, "exchange"))
        object.__setattr__(self, "symbol", _nonempty_str(self.symbol, "symbol"))
        object.__setattr__(self, "tape_schema", _nonempty_str(self.tape_schema, "tape_schema"))
        for name in ("tape_num_events", "tape_num_l2_batches", "tape_num_trades", "num_rows"):
            object.__setattr__(self, name, _nonnegative_int(int(getattr(self, name)), name))
        object.__setattr__(self, "decision_interval_us", _positive_int(int(self.decision_interval_us), "decision_interval_us"))
        if self.start_event_index is not None:
            object.__setattr__(self, "start_event_index", _nonnegative_int(int(self.start_event_index), "start_event_index"))
        if self.max_decisions is not None:
            object.__setattr__(self, "max_decisions", _positive_int(int(self.max_decisions), "max_decisions"))
        features = _name_tuple(self.feature_names, "feature_names")
        labels = _name_tuple(self.label_names, "label_names")
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "label_names", labels)
        object.__setattr__(self, "num_features", _nonnegative_int(int(self.num_features), "num_features"))
        object.__setattr__(self, "num_labels", _nonnegative_int(int(self.num_labels), "num_labels"))
        if self.num_features != len(features) or self.num_labels != len(labels):
            raise ValueError("manifest feature/label counts must match names")
        object.__setattr__(self, "config_json", str(self.config_json))
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
            "decision_interval_us": self.decision_interval_us,
            "start_event_index": self.start_event_index,
            "max_decisions": self.max_decisions,
            "feature_names": list(self.feature_names),
            "label_names": list(self.label_names),
            "num_rows": self.num_rows,
            "num_features": self.num_features,
            "num_labels": self.num_labels,
            "config_json": self.config_json,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AdverseSelectionDatasetManifest":
        return cls(
            schema=str(raw["schema"]), exchange=str(raw["exchange"]), symbol=str(raw["symbol"]), tape_schema=str(raw["tape_schema"]),
            tape_num_events=int(raw["tape_num_events"]), tape_num_l2_batches=int(raw["tape_num_l2_batches"]), tape_num_trades=int(raw["tape_num_trades"]),
            tape_start_local_ts_us=int(raw["tape_start_local_ts_us"]), tape_end_local_ts_us=int(raw["tape_end_local_ts_us"]),
            decision_interval_us=int(raw["decision_interval_us"]), start_event_index=None if raw.get("start_event_index") is None else int(raw["start_event_index"]),
            max_decisions=None if raw.get("max_decisions") is None else int(raw["max_decisions"]),
            feature_names=tuple(raw["feature_names"]), label_names=tuple(raw["label_names"]), num_rows=int(raw["num_rows"]),
            num_features=int(raw["num_features"]), num_labels=int(raw["num_labels"]), config_json=str(raw.get("config_json", "{}")),
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
        self._decision_ts: list[int] = []
        self._decision_idx: list[int] = []
        self._decision_seq: list[int] = []
        self._features: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._chunk_count = 0
        self._rows = 0

    def append(self, *, decision_local_ts_us: int, decision_event_index: int, decision_event_seq: int, features: Sequence[float] | np.ndarray, labels: Sequence[float] | np.ndarray, label_masks: Sequence[bool] | np.ndarray) -> int:
        f = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        m = np.asarray(label_masks, dtype=np.bool_)
        if f.shape != (len(self.config.feature_names),):
            raise ValueError("features width mismatch")
        if y.shape != (len(self.config.label_names),) or m.shape != y.shape:
            raise ValueError("labels/masks width mismatch")
        self._decision_ts.append(int(decision_local_ts_us)); self._decision_idx.append(int(decision_event_index)); self._decision_seq.append(int(decision_event_seq))
        self._features.append(f.copy()); self._labels.append(y.copy()); self._masks.append(m.copy())
        self._rows += 1
        if len(self._features) >= self.config.chunk_rows:
            self._flush()
        return self._rows

    def _flush(self) -> None:
        if not self._features:
            return
        prefix = self.chunks_dir / f"chunk_{self._chunk_count:06d}"
        np.save(str(prefix) + "_decision_local_ts_us.npy", np.asarray(self._decision_ts, dtype=np.int64))
        np.save(str(prefix) + "_decision_event_index.npy", np.asarray(self._decision_idx, dtype=np.int64))
        np.save(str(prefix) + "_decision_event_seq.npy", np.asarray(self._decision_seq, dtype=np.int64))
        np.save(str(prefix) + "_features.npy", np.vstack(self._features).astype(np.float32, copy=False))
        np.save(str(prefix) + "_labels.npy", np.vstack(self._labels).astype(np.float32, copy=False))
        np.save(str(prefix) + "_label_masks.npy", np.vstack(self._masks).astype(np.bool_, copy=False))
        self._decision_ts.clear(); self._decision_idx.clear(); self._decision_seq.clear(); self._features.clear(); self._labels.clear(); self._masks.clear()
        self._chunk_count += 1

    def finalize(self) -> DiskBackedAdverseSelectionDataset:
        self._flush()
        n = self._rows; nf = len(self.config.feature_names); nl = len(self.config.label_names)
        specs = {
            "decision_local_ts_us": (np.int64, (n,)), "decision_event_index": (np.int64, (n,)), "decision_event_seq": (np.int64, (n,)),
            "features": (np.float32, (n, nf)), "labels": (np.float32, (n, nl)), "label_masks": (np.bool_, (n, nl)),
        }
        outs = {name: np.lib.format.open_memmap(self.arrays_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape) for name, (dtype, shape) in specs.items()}
        pos = 0
        for i in range(self._chunk_count):
            prefix = self.chunks_dir / f"chunk_{i:06d}"
            rows = np.load(str(prefix) + "_decision_local_ts_us.npy", mmap_mode="r").shape[0]
            for name in specs:
                outs[name][pos:pos+rows] = np.load(str(prefix) + f"_{name}.npy", mmap_mode="r")
            pos += rows
        for arr in outs.values():
            arr.flush()
        meta = dict(self.config.manifest_metadata)
        manifest = AdverseSelectionDatasetManifest(
            schema=ADVERSE_SELECTION_DATASET_SCHEMA,
            exchange=str(meta["exchange"]), symbol=str(meta["symbol"]), tape_schema=str(meta["tape_schema"]),
            tape_num_events=int(meta["tape_num_events"]), tape_num_l2_batches=int(meta["tape_num_l2_batches"]), tape_num_trades=int(meta["tape_num_trades"]),
            tape_start_local_ts_us=int(meta["tape_start_local_ts_us"]), tape_end_local_ts_us=int(meta["tape_end_local_ts_us"]),
            decision_interval_us=int(meta["decision_interval_us"]), start_event_index=meta.get("start_event_index"), max_decisions=meta.get("max_decisions"),
            feature_names=self.config.feature_names, label_names=self.config.label_names, num_rows=n, num_features=nf, num_labels=nl,
            config_json=str(meta.get("config_json", "{}")), created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        (self.root / "manifest.json").write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if self.config.cleanup_chunks:
            import shutil
            shutil.rmtree(self.chunks_dir, ignore_errors=True)
        return load_adverse_selection_dataset(self.root, mmap_mode="r")


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
