"""Disk-backed helper indexes for adverse-selection training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from mmrt.execution.adverse_selection import KyleLambdaConfig, _bps_from_ticks
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, EVENT_TYPE_CODE_TRADE, ExecutionTape
from mmrt.execution.execution_tape_writer import NpyChunkWriter
from mmrt.time_key import EventKey, MAX_EVENT_SEQ

ADVERSE_SELECTION_INDEX_SCHEMA = "mmrt_adverse_selection_index_grid_v1"
MANIFEST_FILENAME = "index_manifest.json"
ARRAYS_DIRNAME = "arrays"
_CHUNKS_DIRNAME = "chunks"


def adverse_selection_index_manifest_sha256(root: str | Path) -> str:
    """Return the SHA-256 of canonical index manifest JSON bytes."""
    manifest_path = Path(root) / MANIFEST_FILENAME
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _check_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


@dataclass(frozen=True, slots=True)
class AdverseSelectionIndexConfig:
    output_root: str
    kyle: KyleLambdaConfig
    use_notional_flow: bool
    tick_size: float
    chunk_rows: int = 100_000
    overwrite: bool = False
    cleanup_chunks: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be a non-empty str")
        if not isinstance(self.kyle, KyleLambdaConfig):
            raise ValueError("kyle must be KyleLambdaConfig")
        object.__setattr__(self, "use_notional_flow", _check_bool(self.use_notional_flow, "use_notional_flow"))
        tick_size = float(self.tick_size)
        if not np.isfinite(tick_size) or tick_size <= 0.0:
            raise ValueError("tick_size must be positive and finite")
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "chunk_rows", _positive_int(int(self.chunk_rows), "chunk_rows"))
        object.__setattr__(self, "overwrite", _check_bool(self.overwrite, "overwrite"))
        object.__setattr__(self, "cleanup_chunks", _check_bool(self.cleanup_chunks, "cleanup_chunks"))


@dataclass(frozen=True, slots=True)
class AdverseSelectionIndexManifest:
    schema: str
    tape_schema: str
    exchange: str
    symbol: str
    tape_num_events: int
    tape_num_l2_batches: int
    tape_num_trades: int
    tape_start_local_ts_us: int
    tape_end_local_ts_us: int
    kyle_sample_interval_us: int
    kyle_response_horizon_us: int
    kyle_windows_us: tuple[int, ...]
    kyle_min_samples: int
    trade_flow_use_notional_flow: bool
    tick_size: float
    valid_l2_count: int
    trade_flow_count: int
    kyle_sample_count: int
    created_at_utc: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "tape_schema": self.tape_schema,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "tape_num_events": int(self.tape_num_events),
            "tape_num_l2_batches": int(self.tape_num_l2_batches),
            "tape_num_trades": int(self.tape_num_trades),
            "tape_start_local_ts_us": int(self.tape_start_local_ts_us),
            "tape_end_local_ts_us": int(self.tape_end_local_ts_us),
            "kyle": {
                "sample_interval_us": int(self.kyle_sample_interval_us),
                "response_horizon_us": int(self.kyle_response_horizon_us),
                "windows_us": list(self.kyle_windows_us),
                "min_samples": int(self.kyle_min_samples),
            },
            "trade_flow": {"use_notional_flow": bool(self.trade_flow_use_notional_flow)},
            "tick_size": float(self.tick_size),
            "valid_l2_count": int(self.valid_l2_count),
            "trade_flow_count": int(self.trade_flow_count),
            "kyle_sample_count": int(self.kyle_sample_count),
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AdverseSelectionIndexManifest":
        kyle = raw.get("kyle", {})
        flow = raw.get("trade_flow", {})
        if not isinstance(kyle, Mapping) or not isinstance(flow, Mapping):
            raise ValueError("invalid adverse-selection index manifest")
        return cls(
            schema=str(raw["schema"]),
            tape_schema=str(raw["tape_schema"]),
            exchange=str(raw["exchange"]),
            symbol=str(raw["symbol"]),
            tape_num_events=int(raw["tape_num_events"]),
            tape_num_l2_batches=int(raw["tape_num_l2_batches"]),
            tape_num_trades=int(raw["tape_num_trades"]),
            tape_start_local_ts_us=int(raw["tape_start_local_ts_us"]),
            tape_end_local_ts_us=int(raw["tape_end_local_ts_us"]),
            kyle_sample_interval_us=int(kyle["sample_interval_us"]),
            kyle_response_horizon_us=int(kyle["response_horizon_us"]),
            kyle_windows_us=tuple(int(x) for x in kyle["windows_us"]),
            kyle_min_samples=int(kyle["min_samples"]),
            trade_flow_use_notional_flow=bool(flow["use_notional_flow"]),
            tick_size=float(raw["tick_size"]),
            valid_l2_count=int(raw["valid_l2_count"]),
            trade_flow_count=int(raw["trade_flow_count"]),
            kyle_sample_count=int(raw["kyle_sample_count"]),
            created_at_utc=str(raw["created_at_utc"]),
        )

    def matches(self, tape: ExecutionTape, config: AdverseSelectionIndexConfig) -> bool:
        m = tape.manifest
        return (
            self.schema == ADVERSE_SELECTION_INDEX_SCHEMA
            and self.tape_schema == m.schema
            and self.exchange == m.exchange
            and self.symbol == m.symbol
            and self.tape_num_events == m.num_events
            and self.tape_num_l2_batches == m.num_l2_batches
            and self.tape_num_trades == m.num_trades
            and self.tape_start_local_ts_us == m.start_local_ts_us
            and self.tape_end_local_ts_us == m.end_local_ts_us
            and self.kyle_sample_interval_us == config.kyle.sample_interval_us
            and self.kyle_response_horizon_us == config.kyle.response_horizon_us
            and self.kyle_windows_us == tuple(config.kyle.windows_us)
            and self.kyle_min_samples == config.kyle.min_samples
            and self.trade_flow_use_notional_flow == config.use_notional_flow
            and abs(self.tick_size - config.tick_size) <= 1e-18
        )


@dataclass(frozen=True, slots=True)
class ValidL2Index:
    local_ts_us: np.ndarray
    event_seq: np.ndarray
    mid_tick: np.ndarray

    @property
    def count(self) -> int:
        return int(self.local_ts_us.shape[0])

    def future_mid_and_key_at_or_after(self, key: EventKey) -> tuple[float, EventKey] | None:
        idx = int(np.searchsorted(self.local_ts_us, key.local_ts_us, side="left"))
        if idx >= self.count:
            return None

        if int(self.local_ts_us[idx]) == key.local_ts_us:
            j = idx
            best: int | None = None
            while j < self.count and int(self.local_ts_us[j]) == key.local_ts_us:
                if int(self.event_seq[j]) <= key.event_seq:
                    best = j
                j += 1
            if best is not None:
                return (
                    float(self.mid_tick[best]),
                    EventKey(int(self.local_ts_us[best]), int(self.event_seq[best])),
                )
            idx = j

        if idx >= self.count:
            return None
        return (
            float(self.mid_tick[idx]),
            EventKey(int(self.local_ts_us[idx]), int(self.event_seq[idx])),
        )

    def future_mid_tick_at_or_after(self, key: EventKey) -> float | None:
        out = self.future_mid_and_key_at_or_after(key)
        return None if out is None else out[0]


@dataclass(frozen=True, slots=True)
class TradeFlowIndex:
    local_ts_us: np.ndarray
    event_seq: np.ndarray
    cumulative_flow_after: np.ndarray

    @property
    def count(self) -> int:
        return int(self.local_ts_us.shape[0])

    def _upper_bound(self, key: EventKey) -> int:
        lo = 0; hi = self.count
        while lo < hi:
            mid = (lo + hi) // 2
            if (int(self.local_ts_us[mid]), int(self.event_seq[mid])) <= (key.local_ts_us, key.event_seq):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def flow_between_keys(self, start_exclusive: EventKey, end_inclusive: EventKey) -> float:
        if end_inclusive <= start_exclusive or self.count == 0:
            return 0.0
        left = self._upper_bound(start_exclusive)
        right = self._upper_bound(end_inclusive)
        before_left = 0.0 if left == 0 else float(self.cumulative_flow_after[left - 1])
        before_right = 0.0 if right == 0 else float(self.cumulative_flow_after[right - 1])
        return before_right - before_left


@dataclass(frozen=True, slots=True)
class KyleSampleIndex:
    end_local_ts_us: np.ndarray
    end_event_seq: np.ndarray
    x_flow: np.ndarray
    y_mid_bps: np.ndarray

    @property
    def count(self) -> int:
        return int(self.end_local_ts_us.shape[0])


@dataclass(frozen=True, slots=True)
class AdverseSelectionIndex:
    root: Path
    valid_l2: ValidL2Index
    trade_flow: TradeFlowIndex
    kyle_samples: KyleSampleIndex
    manifest: AdverseSelectionIndexManifest


def _array_path(root: Path, name: str) -> Path:
    return root / ARRAYS_DIRNAME / f"{name}.npy"


def _write_manifest(root: Path, manifest: AdverseSelectionIndexManifest) -> None:
    tmp = root / f"{MANIFEST_FILENAME}.tmp"
    tmp.write_text(json.dumps(manifest.as_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(root / MANIFEST_FILENAME)


def _book_mid_from_ptr(tape: ExecutionTape, book_ptr: int) -> float | None:
    if book_ptr < 0:
        return None
    bid = int(tape.arrays.book_bid_ticks[book_ptr][0])
    ask = int(tape.arrays.book_ask_ticks[book_ptr][0])
    if bid <= 0 or ask <= bid:
        return None
    return 0.5 * (bid + ask)


def build_adverse_selection_index(tape: ExecutionTape, *, config: AdverseSelectionIndexConfig) -> AdverseSelectionIndex:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(config, AdverseSelectionIndexConfig):
        raise ValueError("config must be AdverseSelectionIndexConfig")
    root = Path(config.output_root)
    if root.exists():
        if not config.overwrite:
            existing = root / MANIFEST_FILENAME
            if existing.exists():
                loaded = load_adverse_selection_index(root, mmap_mode="r")
                if loaded.manifest.matches(tape, config):
                    return loaded
                raise ValueError(f"stale adverse-selection index manifest at {root}; pass overwrite=True to rebuild")
            raise FileExistsError(f"adverse-selection index root exists: {root}")
        shutil.rmtree(root)
    arrays_dir = root / ARRAYS_DIRNAME
    chunks_dir = root / _CHUNKS_DIRNAME
    arrays_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    writers = {
        "valid_l2_local_ts_us": NpyChunkWriter("valid_l2_local_ts_us", np.int64, (), config.chunk_rows, chunks_dir),
        "valid_l2_event_seq": NpyChunkWriter("valid_l2_event_seq", np.int64, (), config.chunk_rows, chunks_dir),
        "valid_l2_mid_tick": NpyChunkWriter("valid_l2_mid_tick", np.float32, (), config.chunk_rows, chunks_dir),
        "trade_local_ts_us": NpyChunkWriter("trade_local_ts_us", np.int64, (), config.chunk_rows, chunks_dir),
        "trade_event_seq": NpyChunkWriter("trade_event_seq", np.int64, (), config.chunk_rows, chunks_dir),
        "trade_cumulative_flow": NpyChunkWriter("trade_cumulative_flow", np.float64, (), config.chunk_rows, chunks_dir),
    }
    running = 0.0
    events = tape.arrays.events
    trades = tape.arrays.trades
    for event in events:
        etype = int(event["event_type_code"])
        if etype == EVENT_TYPE_CODE_L2_BATCH:
            mid = _book_mid_from_ptr(tape, int(event["book_ptr"]))
            if mid is not None:
                writers["valid_l2_local_ts_us"].append(int(event["local_ts_us"]))
                writers["valid_l2_event_seq"].append(int(event["event_seq"]))
                writers["valid_l2_mid_tick"].append(float(mid))
        elif etype == EVENT_TYPE_CODE_TRADE:
            trade_ptr = int(event["trade_ptr"])
            if trade_ptr < 0:
                continue
            trade = trades[trade_ptr]
            side = int(trade["side_code"])
            amount = float(trade["amount"])
            if side == 0 or amount <= 0.0:
                continue
            signed = (1.0 if side > 0 else -1.0) * amount
            if config.use_notional_flow:
                signed *= int(trade["price_tick"]) * config.tick_size
            running += signed
            writers["trade_local_ts_us"].append(int(event["local_ts_us"]))
            writers["trade_event_seq"].append(int(event["event_seq"]))
            writers["trade_cumulative_flow"].append(running)
    counts: dict[str, int] = {}
    for name, writer in writers.items():
        counts[name] = writer.finalize(_array_path(root, name))
    valid = ValidL2Index(
        np.load(_array_path(root, "valid_l2_local_ts_us"), mmap_mode="r"),
        np.load(_array_path(root, "valid_l2_event_seq"), mmap_mode="r"),
        np.load(_array_path(root, "valid_l2_mid_tick"), mmap_mode="r"),
    )
    flow = TradeFlowIndex(
        np.load(_array_path(root, "trade_local_ts_us"), mmap_mode="r"),
        np.load(_array_path(root, "trade_event_seq"), mmap_mode="r"),
        np.load(_array_path(root, "trade_cumulative_flow"), mmap_mode="r"),
    )
    sample_writers = {
        "kyle_sample_end_local_ts_us": NpyChunkWriter("kyle_sample_end_local_ts_us", np.int64, (), config.chunk_rows, chunks_dir),
        "kyle_sample_end_event_seq": NpyChunkWriter("kyle_sample_end_event_seq", np.int64, (), config.chunk_rows, chunks_dir),
        "kyle_sample_x_flow": NpyChunkWriter("kyle_sample_x_flow", np.float64, (), config.chunk_rows, chunks_dir),
        "kyle_sample_y_mid_bps": NpyChunkWriter("kyle_sample_y_mid_bps", np.float64, (), config.chunk_rows, chunks_dir),
    }
    last_key: EventKey | None = None
    if valid.count:
        start = int(valid.local_ts_us[0])
        end = int(valid.local_ts_us[-1]) - config.kyle.response_horizon_us
        while start <= end:
            start_key = EventKey(start, MAX_EVENT_SEQ)
            flow_end_key = EventKey(start + config.kyle.sample_interval_us, MAX_EVENT_SEQ)
            response_key = EventKey(start + config.kyle.response_horizon_us, MAX_EVENT_SEQ)
            start_info = valid.future_mid_and_key_at_or_after(start_key)
            response_info = valid.future_mid_and_key_at_or_after(response_key)
            if start_info is not None and response_info is not None:
                start_mid, start_obs_key = start_info
                response_mid, response_obs_key = response_info
                end_key = max(start_obs_key, response_obs_key, flow_end_key)
                if last_key is not None and end_key < last_key:
                    raise RuntimeError("Kyle sample end keys are not nondecreasing")
                last_key = end_key
                sample_writers["kyle_sample_end_local_ts_us"].append(end_key.local_ts_us)
                sample_writers["kyle_sample_end_event_seq"].append(end_key.event_seq)
                sample_writers["kyle_sample_x_flow"].append(flow.flow_between_keys(start_key, flow_end_key))
                sample_writers["kyle_sample_y_mid_bps"].append(_bps_from_ticks(float(response_mid - start_mid), float(start_mid)))
            start += config.kyle.sample_interval_us
    sample_count = 0
    for name, writer in sample_writers.items():
        sample_count = writer.finalize(_array_path(root, name))
    manifest = AdverseSelectionIndexManifest(
        schema=ADVERSE_SELECTION_INDEX_SCHEMA,
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        tape_num_events=tape.manifest.num_events,
        tape_num_l2_batches=tape.manifest.num_l2_batches,
        tape_num_trades=tape.manifest.num_trades,
        tape_start_local_ts_us=tape.manifest.start_local_ts_us,
        tape_end_local_ts_us=tape.manifest.end_local_ts_us,
        kyle_sample_interval_us=config.kyle.sample_interval_us,
        kyle_response_horizon_us=config.kyle.response_horizon_us,
        kyle_windows_us=tuple(config.kyle.windows_us),
        kyle_min_samples=config.kyle.min_samples,
        trade_flow_use_notional_flow=config.use_notional_flow,
        tick_size=config.tick_size,
        valid_l2_count=counts["valid_l2_local_ts_us"],
        trade_flow_count=counts["trade_local_ts_us"],
        kyle_sample_count=sample_count,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(root, manifest)
    if config.cleanup_chunks:
        for writer in (*writers.values(), *sample_writers.values()):
            writer.cleanup()
        shutil.rmtree(chunks_dir, ignore_errors=True)
    return load_adverse_selection_index(root, mmap_mode="r")


def load_adverse_selection_index(root: str | Path, *, mmap_mode: str | None = "r") -> AdverseSelectionIndex:
    root = Path(root)
    manifest = AdverseSelectionIndexManifest.from_dict(json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8")))
    if manifest.schema != ADVERSE_SELECTION_INDEX_SCHEMA:
        raise ValueError("invalid adverse-selection index schema")
    def load(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        arr = np.load(_array_path(root, name), mmap_mode=mmap_mode)
        if arr.dtype != np.dtype(dtype) or arr.shape != shape:
            raise ValueError(f"{name} must have dtype {np.dtype(dtype)} and shape {shape}")
        return arr
    valid = ValidL2Index(
        load("valid_l2_local_ts_us", np.int64, (manifest.valid_l2_count,)),
        load("valid_l2_event_seq", np.int64, (manifest.valid_l2_count,)),
        load("valid_l2_mid_tick", np.float32, (manifest.valid_l2_count,)),
    )
    flow = TradeFlowIndex(
        load("trade_local_ts_us", np.int64, (manifest.trade_flow_count,)),
        load("trade_event_seq", np.int64, (manifest.trade_flow_count,)),
        load("trade_cumulative_flow", np.float64, (manifest.trade_flow_count,)),
    )
    kyle = KyleSampleIndex(
        load("kyle_sample_end_local_ts_us", np.int64, (manifest.kyle_sample_count,)),
        load("kyle_sample_end_event_seq", np.int64, (manifest.kyle_sample_count,)),
        load("kyle_sample_x_flow", np.float64, (manifest.kyle_sample_count,)),
        load("kyle_sample_y_mid_bps", np.float64, (manifest.kyle_sample_count,)),
    )
    return AdverseSelectionIndex(root=root, valid_l2=valid, trade_flow=flow, kyle_samples=kyle, manifest=manifest)


def build_or_load_adverse_selection_index(tape: ExecutionTape, *, config: AdverseSelectionIndexConfig) -> AdverseSelectionIndex:
    root = Path(config.output_root)
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = AdverseSelectionIndexManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest.matches(tape, config):
            return load_adverse_selection_index(root, mmap_mode="r")
        if not config.overwrite:
            raise ValueError(f"stale adverse-selection index manifest at {root}; pass overwrite=True to rebuild")
    return build_adverse_selection_index(tape, config=config)
