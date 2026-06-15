"""Disk-backed adverse-selection feature datasets for signal generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.adverse_selection import (
    AdverseSelectionConfig,
    _DiskKyleSampleView,
    _adverse_config_summary,
    _iter_adverse_selection_feature_rows_for_decision_grid,
    adverse_selection_feature_names,
)
from mmrt.execution.adverse_selection_index import AdverseSelectionIndexConfig, adverse_selection_index_manifest_sha256, build_or_load_adverse_selection_index
from mmrt.execution.contracts import SymbolSpec
from mmrt.execution.decision_grid import DecisionGrid, validate_decision_grid_for_execution_tape
from mmrt.execution.execution_tape import ExecutionTape
from mmrt.execution.execution_tape_writer import NpyChunkWriter
from mmrt.time_key import MAX_EVENT_SEQ

ADVERSE_SELECTION_FEATURE_DATASET_SCHEMA = "mmrt_adverse_selection_feature_dataset_grid_v1"


@dataclass(frozen=True, slots=True)
class AdverseSelectionFeatureDatasetManifest:
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
    feature_names: tuple[str, ...]
    num_rows: int
    num_features: int
    config_json: str
    index_schema: str
    index_manifest_sha256: str
    index_root: str
    created_at_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "exchange": self.exchange, "symbol": self.symbol, "tape_schema": self.tape_schema,
            "tape_num_events": self.tape_num_events, "tape_num_l2_batches": self.tape_num_l2_batches, "tape_num_trades": self.tape_num_trades,
            "tape_start_local_ts_us": self.tape_start_local_ts_us, "tape_end_local_ts_us": self.tape_end_local_ts_us,
            "decision_grid_schema": self.decision_grid_schema, "decision_grid_hash": self.decision_grid_hash, "decision_grid_n_rows": self.decision_grid_n_rows,
            "decision_schedule": dict(self.decision_schedule),
            "feature_names": list(self.feature_names), "num_rows": self.num_rows, "num_features": self.num_features,
            "config_json": self.config_json, "index_schema": self.index_schema, "index_manifest_sha256": self.index_manifest_sha256, "index_root": self.index_root, "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AdverseSelectionFeatureDatasetManifest":
        if str(raw["schema"]) != ADVERSE_SELECTION_FEATURE_DATASET_SCHEMA:
            raise ValueError("invalid adverse-selection feature dataset schema")
        return cls(
            schema=str(raw["schema"]), exchange=str(raw["exchange"]), symbol=str(raw["symbol"]), tape_schema=str(raw["tape_schema"]),
            tape_num_events=int(raw["tape_num_events"]), tape_num_l2_batches=int(raw["tape_num_l2_batches"]), tape_num_trades=int(raw["tape_num_trades"]),
            tape_start_local_ts_us=int(raw["tape_start_local_ts_us"]), tape_end_local_ts_us=int(raw["tape_end_local_ts_us"]),
            decision_grid_schema=str(raw["decision_grid_schema"]),
            decision_grid_hash=str(raw["decision_grid_hash"]),
            decision_grid_n_rows=int(raw["decision_grid_n_rows"]),
            decision_schedule=dict(raw["decision_schedule"]),  # type: ignore[arg-type]
            feature_names=tuple(str(x) for x in raw["feature_names"]), num_rows=int(raw["num_rows"]), num_features=int(raw["num_features"]),
            config_json=str(raw.get("config_json", "{}")), index_schema=str(raw["index_schema"]), index_manifest_sha256=str(raw["index_manifest_sha256"]), index_root=str(raw["index_root"]), created_at_utc=str(raw["created_at_utc"]),
        )


@dataclass(frozen=True, slots=True)
class DiskBackedAdverseSelectionFeatureDataset:
    root: Path
    manifest: AdverseSelectionFeatureDatasetManifest
    decision_local_ts_us: np.ndarray
    decision_event_index: np.ndarray
    decision_event_seq: np.ndarray
    features: np.ndarray

    @property
    def feature_names(self) -> tuple[str, ...]: return self.manifest.feature_names
    @property
    def num_rows(self) -> int: return self.manifest.num_rows
    @property
    def num_features(self) -> int: return self.manifest.num_features


class _FeatureWriter:
    def __init__(self, root: Path, feature_names: tuple[str, ...], chunk_rows: int, overwrite: bool, cleanup_chunks: bool, manifest_metadata: Mapping[str, object]):
        self.root = root; self.arrays_dir = root / "arrays"; self.chunks_dir = root / "chunks"; self.cleanup_chunks = cleanup_chunks; self.meta = dict(manifest_metadata); self.feature_names = feature_names
        if root.exists() and not overwrite: raise FileExistsError(root)
        if root.exists() and overwrite: shutil.rmtree(root)
        self.arrays_dir.mkdir(parents=True); self.chunks_dir.mkdir(parents=True)
        nf = len(feature_names)
        self.writers = {
            "decision_local_ts_us": NpyChunkWriter("decision_local_ts_us", np.int64, (), chunk_rows, self.chunks_dir),
            "decision_event_index": NpyChunkWriter("decision_event_index", np.int64, (), chunk_rows, self.chunks_dir),
            "decision_event_seq": NpyChunkWriter("decision_event_seq", np.int64, (), chunk_rows, self.chunks_dir),
            "features": NpyChunkWriter("features", np.float32, (nf,), chunk_rows, self.chunks_dir),
        }
        self.rows = 0
    def append(self, ts: int, idx: int, seq: int, features: Sequence[float]) -> None:
        f = np.asarray(features, dtype=np.float32)
        if f.shape != (len(self.feature_names),): raise ValueError("feature width mismatch")
        self.writers["decision_local_ts_us"].append(ts); self.writers["decision_event_index"].append(idx); self.writers["decision_event_seq"].append(seq); self.writers["features"].append(f); self.rows += 1
    def append_many(self, ts, idx, seq, features) -> tuple[int, int]:
        ts_arr = np.asarray(ts, dtype=np.int64)
        idx_arr = np.asarray(idx, dtype=np.int64)
        seq_arr = np.asarray(seq, dtype=np.int64)
        f = np.asarray(features, dtype=np.float32)
        if ts_arr.ndim != 1: raise ValueError("decision_local_ts_us must be 1D")
        n = int(ts_arr.shape[0])
        if idx_arr.shape != (n,) or seq_arr.shape != (n,): raise ValueError("decision arrays length mismatch")
        if f.shape != (n, len(self.feature_names)): raise ValueError("feature width mismatch")
        start = self.rows
        self.writers["decision_local_ts_us"].append_many(ts_arr); self.writers["decision_event_index"].append_many(idx_arr); self.writers["decision_event_seq"].append_many(seq_arr); self.writers["features"].append_many(f)
        self.rows += n
        return start, self.rows
    def finalize(self) -> DiskBackedAdverseSelectionFeatureDataset:
        manifest_path = self.root / "manifest.json"; manifest_path.unlink(missing_ok=True)
        for name, writer in self.writers.items():
            writer.finalize(self.arrays_dir / f"{name}.npy")
        m = self.meta
        manifest = AdverseSelectionFeatureDatasetManifest(
            schema=ADVERSE_SELECTION_FEATURE_DATASET_SCHEMA, exchange=str(m["exchange"]), symbol=str(m["symbol"]), tape_schema=str(m["tape_schema"]),
            tape_num_events=int(m["tape_num_events"]), tape_num_l2_batches=int(m["tape_num_l2_batches"]), tape_num_trades=int(m["tape_num_trades"]),
            tape_start_local_ts_us=int(m["tape_start_local_ts_us"]), tape_end_local_ts_us=int(m["tape_end_local_ts_us"]),
            decision_grid_schema=str(m["decision_grid_schema"]), decision_grid_hash=str(m["decision_grid_hash"]), decision_grid_n_rows=int(m["decision_grid_n_rows"]),
            decision_schedule=dict(m["decision_schedule"]), feature_names=self.feature_names, num_rows=self.rows, num_features=len(self.feature_names),
            config_json=str(m.get("config_json", "{}")), index_schema=str(m["index_schema"]), index_manifest_sha256=str(m["index_manifest_sha256"]), index_root=str(m["index_root"]), created_at_utc=datetime.now(timezone.utc).isoformat())
        tmp = self.root / "manifest.json.tmp"; tmp.write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2)+"\n", encoding="utf-8"); tmp.replace(manifest_path)
        if self.cleanup_chunks:
            for w in self.writers.values(): w.cleanup()
            shutil.rmtree(self.chunks_dir, ignore_errors=True)
        return load_adverse_selection_features(self.root)


def load_adverse_selection_features(root: str | Path, *, mmap_mode: str | None = "r") -> DiskBackedAdverseSelectionFeatureDataset:
    root = Path(root); manifest = AdverseSelectionFeatureDatasetManifest.from_dict(json.loads((root/"manifest.json").read_text()))
    arr = root / "arrays"
    decision_local_ts_us = np.load(arr/"decision_local_ts_us.npy", mmap_mode=mmap_mode)
    decision_event_index = np.load(arr/"decision_event_index.npy", mmap_mode=mmap_mode)
    decision_event_seq = np.load(arr/"decision_event_seq.npy", mmap_mode=mmap_mode)
    features = np.load(arr/"features.npy", mmap_mode=mmap_mode)
    n = manifest.num_rows
    if decision_local_ts_us.dtype != np.int64 or decision_local_ts_us.shape != (n,): raise ValueError("decision_local_ts_us shape/dtype mismatch")
    if decision_event_index.dtype != np.int64 or decision_event_index.shape != (n,): raise ValueError("decision_event_index shape/dtype mismatch")
    if decision_event_seq.dtype != np.int64 or decision_event_seq.shape != (n,): raise ValueError("decision_event_seq shape/dtype mismatch")
    if features.dtype != np.float32 or features.shape != (n, manifest.num_features): raise ValueError("features shape/dtype mismatch")
    return DiskBackedAdverseSelectionFeatureDataset(root, manifest, decision_local_ts_us, decision_event_index, decision_event_seq, features)


def build_adverse_selection_features_to_disk(tape: ExecutionTape, *, config: AdverseSelectionConfig, decision_grid: DecisionGrid, output_root: str | Path, work_dir: str | Path | None = None, chunk_rows: int = 100_000, overwrite: bool = False, cleanup_chunks: bool = True, cleanup_work_dir: bool = True, progress_interval: int | None = None) -> DiskBackedAdverseSelectionFeatureDataset:
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    spec = tape.manifest.symbol_spec
    if not isinstance(spec, SymbolSpec): raise ValueError("tape.manifest.symbol_spec must be SymbolSpec")
    root = Path(output_root); index_root = (Path(work_dir) if work_dir is not None else root.parent) / "adverse_selection_work"
    index = build_or_load_adverse_selection_index(tape, config=AdverseSelectionIndexConfig(str(index_root), config.kyle, config.kyle.use_notional_flow, spec.tick_size, chunk_rows=chunk_rows, overwrite=overwrite, cleanup_chunks=cleanup_chunks))
    if index.valid_l2.count == 0: raise ValueError("tape must contain at least one valid two-sided L2 book")
    feature_names = adverse_selection_feature_names(config)
    manifest = tape.manifest
    writer = _FeatureWriter(root, feature_names, chunk_rows, overwrite, cleanup_chunks, {"exchange": manifest.exchange, "symbol": manifest.symbol, "tape_schema": manifest.schema, "tape_num_events": manifest.num_events, "tape_num_l2_batches": manifest.num_l2_batches, "tape_num_trades": manifest.num_trades, "tape_start_local_ts_us": manifest.start_local_ts_us, "tape_end_local_ts_us": manifest.end_local_ts_us, "decision_grid_schema": decision_grid.metadata.schema, "decision_grid_hash": decision_grid.decision_grid_hash, "decision_grid_n_rows": decision_grid.n_rows, "decision_schedule": decision_grid.decision_schedule, "config_json": json.dumps(_adverse_config_summary(config), sort_keys=True), "index_schema": index.manifest.schema, "index_manifest_sha256": adverse_selection_index_manifest_sha256(index.root), "index_root": str(index.root)})
    kyle_samples = _DiskKyleSampleView(index.kyle_samples)
    batch_ts = np.empty(chunk_rows, dtype=np.int64)
    batch_idx = np.empty(chunk_rows, dtype=np.int64)
    batch_seq = np.empty(chunk_rows, dtype=np.int64)
    batch_features = np.empty((chunk_rows, len(feature_names)), dtype=np.float32)
    batch_used = 0

    def flush_batch() -> None:
        nonlocal batch_used
        if batch_used == 0:
            return
        writer.append_many(
            batch_ts[:batch_used],
            batch_idx[:batch_used],
            batch_seq[:batch_used],
            batch_features[:batch_used],
        )
        batch_used = 0

    for emitted, row in enumerate(_iter_adverse_selection_feature_rows_for_decision_grid(tape, config=config, decision_grid=decision_grid, kyle_samples=kyle_samples), start=1):
        batch_ts[batch_used] = row.decision_local_ts_us
        batch_idx[batch_used] = row.decision_event_index
        batch_seq[batch_used] = row.decision_event_seq
        batch_features[batch_used] = row.features
        batch_used += 1
        if batch_used >= chunk_rows:
            flush_batch()
        if progress_interval and emitted % progress_interval == 0:
            print(f"adverse_features progress rows_written={emitted}/{decision_grid.n_rows}")
    flush_batch()
    ds = writer.finalize()
    if cleanup_work_dir: shutil.rmtree(index_root, ignore_errors=True)
    return ds


def summarize_adverse_selection_feature_store(dataset: DiskBackedAdverseSelectionFeatureDataset) -> dict[str, object]:
    return {"num_decisions": dataset.num_rows, "num_features": dataset.num_features, "feature_names": list(dataset.feature_names), "first_decision_local_ts_us": int(dataset.decision_local_ts_us[0]) if dataset.num_rows else None, "last_decision_local_ts_us": int(dataset.decision_local_ts_us[-1]) if dataset.num_rows else None, "first_decision_event_index": int(dataset.decision_event_index[0]) if dataset.num_rows else None, "last_decision_event_index": int(dataset.decision_event_index[-1]) if dataset.num_rows else None}
