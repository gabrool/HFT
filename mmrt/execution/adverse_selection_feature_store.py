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
    VPINState,
    _KyleSample,
    _TradeWindowState,
    _adverse_config_summary,
    _book_top_from_l2_row,
    _feature_row,
    _new_kyle_state,
    adverse_selection_feature_names,
)
from mmrt.execution.adverse_selection_index import AdverseSelectionIndexConfig, build_or_load_adverse_selection_index
from mmrt.execution.contracts import SymbolSpec
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, EVENT_TYPE_CODE_TRADE, ExecutionTape
from mmrt.execution.execution_tape_writer import NpyChunkWriter
from mmrt.time_key import EventKey, MAX_EVENT_SEQ
from collections import deque

ADVERSE_SELECTION_FEATURE_DATASET_SCHEMA = "mmrt_adverse_selection_feature_dataset" + "_" + "v" + "1"


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
    decision_interval_us: int
    start_event_index: int | None
    max_decisions: int | None
    feature_names: tuple[str, ...]
    num_rows: int
    num_features: int
    config_json: str
    created_at_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "exchange": self.exchange, "symbol": self.symbol, "tape_schema": self.tape_schema,
            "tape_num_events": self.tape_num_events, "tape_num_l2_batches": self.tape_num_l2_batches, "tape_num_trades": self.tape_num_trades,
            "tape_start_local_ts_us": self.tape_start_local_ts_us, "tape_end_local_ts_us": self.tape_end_local_ts_us,
            "decision_interval_us": self.decision_interval_us, "start_event_index": self.start_event_index, "max_decisions": self.max_decisions,
            "feature_names": list(self.feature_names), "num_rows": self.num_rows, "num_features": self.num_features,
            "config_json": self.config_json, "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AdverseSelectionFeatureDatasetManifest":
        return cls(
            schema=str(raw["schema"]), exchange=str(raw["exchange"]), symbol=str(raw["symbol"]), tape_schema=str(raw["tape_schema"]),
            tape_num_events=int(raw["tape_num_events"]), tape_num_l2_batches=int(raw["tape_num_l2_batches"]), tape_num_trades=int(raw["tape_num_trades"]),
            tape_start_local_ts_us=int(raw["tape_start_local_ts_us"]), tape_end_local_ts_us=int(raw["tape_end_local_ts_us"]),
            decision_interval_us=int(raw["decision_interval_us"]), start_event_index=None if raw.get("start_event_index") is None else int(raw["start_event_index"]),
            max_decisions=None if raw.get("max_decisions") is None else int(raw["max_decisions"]),
            feature_names=tuple(str(x) for x in raw["feature_names"]), num_rows=int(raw["num_rows"]), num_features=int(raw["num_features"]),
            config_json=str(raw.get("config_json", "{}")), created_at_utc=str(raw["created_at_utc"]),
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
    def finalize(self) -> DiskBackedAdverseSelectionFeatureDataset:
        manifest_path = self.root / "manifest.json"; manifest_path.unlink(missing_ok=True)
        for name, writer in self.writers.items():
            writer.finalize(self.arrays_dir / f"{name}.npy")
        m = self.meta
        manifest = AdverseSelectionFeatureDatasetManifest(
            schema=ADVERSE_SELECTION_FEATURE_DATASET_SCHEMA, exchange=str(m["exchange"]), symbol=str(m["symbol"]), tape_schema=str(m["tape_schema"]),
            tape_num_events=int(m["tape_num_events"]), tape_num_l2_batches=int(m["tape_num_l2_batches"]), tape_num_trades=int(m["tape_num_trades"]),
            tape_start_local_ts_us=int(m["tape_start_local_ts_us"]), tape_end_local_ts_us=int(m["tape_end_local_ts_us"]), decision_interval_us=int(m["decision_interval_us"]),
            start_event_index=m.get("start_event_index"), max_decisions=m.get("max_decisions"), feature_names=self.feature_names, num_rows=self.rows, num_features=len(self.feature_names),
            config_json=str(m.get("config_json", "{}")), created_at_utc=datetime.now(timezone.utc).isoformat())
        tmp = self.root / "manifest.json.tmp"; tmp.write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2)+"\n", encoding="utf-8"); tmp.replace(manifest_path)
        if self.cleanup_chunks:
            for w in self.writers.values(): w.cleanup()
            shutil.rmtree(self.chunks_dir, ignore_errors=True)
        return load_adverse_selection_features(self.root)


def load_adverse_selection_features(root: str | Path, *, mmap_mode: str | None = "r") -> DiskBackedAdverseSelectionFeatureDataset:
    root = Path(root); manifest = AdverseSelectionFeatureDatasetManifest.from_dict(json.loads((root/"manifest.json").read_text()))
    arr = root / "arrays"
    return DiskBackedAdverseSelectionFeatureDataset(root, manifest, np.load(arr/"decision_local_ts_us.npy", mmap_mode=mmap_mode), np.load(arr/"decision_event_index.npy", mmap_mode=mmap_mode), np.load(arr/"decision_event_seq.npy", mmap_mode=mmap_mode), np.load(arr/"features.npy", mmap_mode=mmap_mode))


def build_adverse_selection_features_to_disk(tape: ExecutionTape, *, config: AdverseSelectionConfig, output_root: str | Path, work_dir: str | Path | None = None, chunk_rows: int = 100_000, overwrite: bool = False, cleanup_chunks: bool = True, cleanup_work_dir: bool = True, progress_interval: int | None = None) -> DiskBackedAdverseSelectionFeatureDataset:
    spec = tape.manifest.symbol_spec
    if not isinstance(spec, SymbolSpec): raise ValueError("tape.manifest.symbol_spec must be SymbolSpec")
    root = Path(output_root); index_root = (Path(work_dir) if work_dir is not None else root.parent) / "adverse_selection_work"
    index = build_or_load_adverse_selection_index(tape, config=AdverseSelectionIndexConfig(str(index_root), config.kyle, config.kyle.use_notional_flow, spec.tick_size, chunk_rows=chunk_rows, overwrite=overwrite, cleanup_chunks=cleanup_chunks))
    if index.valid_l2.count == 0: raise ValueError("tape must contain at least one valid two-sided L2 book")
    feature_names = adverse_selection_feature_names(config)
    manifest = tape.manifest
    writer = _FeatureWriter(root, feature_names, chunk_rows, overwrite, cleanup_chunks, {"exchange": manifest.exchange, "symbol": manifest.symbol, "tape_schema": manifest.schema, "tape_num_events": manifest.num_events, "tape_num_l2_batches": manifest.num_l2_batches, "tape_num_trades": manifest.num_trades, "tape_start_local_ts_us": manifest.start_local_ts_us, "tape_end_local_ts_us": manifest.end_local_ts_us, "decision_interval_us": config.decision_interval_us, "start_event_index": config.start_event_index, "max_decisions": config.max_decisions, "config_json": json.dumps(_adverse_config_summary(config), sort_keys=True)})
    vpin_state = VPINState(config.vpin, deque()); kyle_state = _new_kyle_state(config.kyle); next_sample = 0
    flow_states = tuple(_TradeWindowState(w) for w in config.flow_windows_us)
    events = tape.arrays.events; trades = tape.arrays.trades
    latest_book_ptr = -1; previous_book_ptr = -1; last_event_idx = -1; emitted = 0; considered = 0
    first_valid_ts = int(index.valid_l2.local_ts_us[0])
    if config.start_event_index is not None and config.start_event_index < len(events): first_valid_ts = max(first_valid_ts, int(events[config.start_event_index]["local_ts_us"]))
    next_ts = first_valid_ts
    def update_states(current_key: EventKey) -> None:
        nonlocal next_sample
        for fs in flow_states: fs.expire(current_key.local_ts_us)
        samples = index.kyle_samples
        while next_sample < samples.count:
            sample_key = EventKey(int(samples.end_local_ts_us[next_sample]), int(samples.end_event_seq[next_sample]))
            if sample_key > current_key: break
            kyle_state.add_finalized_sample(_KyleSample(sample_key, float(samples.x_flow[next_sample]), float(samples.y_mid_bps[next_sample])), current_key); next_sample += 1
        kyle_state.expire_old(current_key)
    def emit_until(up_to_ts: int) -> bool:
        nonlocal next_ts, emitted, considered
        while next_ts <= up_to_ts and latest_book_ptr >= 0:
            if next_ts >= first_valid_ts:
                considered += 1; key = EventKey(next_ts, MAX_EVENT_SEQ); update_states(key)
                writer.append(next_ts, last_event_idx, MAX_EVENT_SEQ, _feature_row(tape, latest_book_ptr=latest_book_ptr, previous_book_ptr=previous_book_ptr, spec=spec, vpin_state=vpin_state, flow_states=flow_states, kyle_state=kyle_state)); emitted += 1
                if progress_interval and considered % progress_interval == 0: print(f"adverse_features progress decisions_considered={considered} rows_written={emitted}")
                if config.max_decisions is not None and considered >= config.max_decisions: return True
            next_ts += config.decision_interval_us
        return False
    i = 0
    while i < len(events):
        group_ts = int(events[i]["local_ts_us"])
        if emit_until(group_ts - 1): break
        while i < len(events) and int(events[i]["local_ts_us"]) == group_ts:
            event = events[i]; et = int(event["event_type_code"])
            if et == EVENT_TYPE_CODE_L2_BATCH:
                bp = int(event["book_ptr"])
                if bp >= 0:
                    try: _book_top_from_l2_row(tape, bp)
                    except ValueError: pass
                    else: previous_book_ptr = latest_book_ptr; latest_book_ptr = bp
            elif et == EVENT_TYPE_CODE_TRADE:
                tp = int(event["trade_ptr"])
                if tp >= 0:
                    tr = trades[tp]; side = int(tr["side_code"]); price_tick = int(tr["price_tick"]); amount = float(tr["amount"]); notional = amount * price_tick * spec.tick_size
                    for fs in flow_states: fs.update_trade(group_ts, side, amount, notional)
                    vpin_state.update_trade(side_code=side, price_tick=price_tick, amount=amount, tick_size=spec.tick_size)
            last_event_idx = i; i += 1
        update_states(EventKey(group_ts, MAX_EVENT_SEQ))
        if emit_until(group_ts): break
    ds = writer.finalize()
    if cleanup_work_dir: shutil.rmtree(index_root, ignore_errors=True)
    return ds


def summarize_adverse_selection_feature_store(dataset: DiskBackedAdverseSelectionFeatureDataset) -> dict[str, object]:
    return {"num_decisions": dataset.num_rows, "num_features": dataset.num_features, "feature_names": list(dataset.feature_names), "first_decision_local_ts_us": int(dataset.decision_local_ts_us[0]) if dataset.num_rows else None, "last_decision_local_ts_us": int(dataset.decision_local_ts_us[-1]) if dataset.num_rows else None, "first_decision_event_index": int(dataset.decision_event_index[0]) if dataset.num_rows else None, "last_decision_event_index": int(dataset.decision_event_index[-1]) if dataset.num_rows else None}
