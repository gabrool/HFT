"""First-class execution decision-grid artifacts.

``DecisionScheduleConfig`` describes the live policy for when to decide.
``decision_grid.npz`` is the deterministic offline realization of that policy
for one execution tape. Downstream training, signal, adverse-selection, and RL
code must align by this grid's hash instead of regenerating decision rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, ExecutionTape
from mmrt.features.schedule import DecisionScheduleConfig, decision_schedule_config_from_dict

DECISION_GRID_SCHEMA = "mmrt_execution_decision_grid_v1"
DECISION_GRID_FILENAME = "decision_grid.npz"
DECISION_GRID_SUMMARY_FILENAME = "decision_grid_summary.json"
SCHEDULER_VERSION = "event_schedule_reason_v1"

DECISION_GRID_ARRAY_ORDER = (
    "decision_event_index",
    "decision_local_ts_us",
    "decision_event_seq",
    "book_ptr",
    "reason_code",
    "reason_flags",
    "elapsed_since_prev_decision_us",
    "events_since_prev_decision",
    "l2_events_since_prev_decision",
    "trade_events_since_prev_decision",
)

_ARRAY_DTYPES = {
    "decision_event_index": np.dtype("int64"),
    "decision_local_ts_us": np.dtype("int64"),
    "decision_event_seq": np.dtype("int64"),
    "book_ptr": np.dtype("int64"),
    "reason_code": np.dtype("int16"),
    "reason_flags": np.dtype("int16"),
    "elapsed_since_prev_decision_us": np.dtype("int64"),
    "events_since_prev_decision": np.dtype("int64"),
    "l2_events_since_prev_decision": np.dtype("int64"),
    "trade_events_since_prev_decision": np.dtype("int64"),
}


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
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json_bytes(obj: Mapping[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _json_safe(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, f"{name}[]") for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, f"{name}.{k}") for k, v in value.items()}
    raise ValueError(f"{name} is not JSON-safe")


def _coerce_grid_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 array")
    dtype = _ARRAY_DTYPES[name]
    if not np.can_cast(arr.dtype, dtype, casting="same_kind") and arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer-compatible")
    return np.ascontiguousarray(arr, dtype=dtype)


def _metadata_without_hash_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(metadata)
    out.pop("created_at_utc", None)
    out.pop("decision_grid_hash", None)
    return _json_safe(out, "metadata")  # type: ignore[return-value]


def compute_decision_grid_hash(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """Hash canonical metadata plus raw arrays in schema order."""

    h = hashlib.sha256()
    h.update(_canonical_json_bytes(_metadata_without_hash_fields(metadata)))
    for name in DECISION_GRID_ARRAY_ORDER:
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype=_ARRAY_DTYPES[name]))
        h.update(name.encode("ascii"))
        h.update(b"\0")
        h.update(arr.dtype.str.encode("ascii"))
        h.update(b"\0")
        h.update(_canonical_json_bytes({"shape": list(arr.shape)}))
        h.update(b"\0")
        h.update(arr.tobytes(order="C"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class DecisionGridMetadata:
    schema: str
    tape_schema: str
    exchange: str
    symbol: str
    tape_num_events: int
    tape_num_l2_batches: int
    tape_num_trades: int
    tape_start_local_ts_us: int
    tape_end_local_ts_us: int
    decision_schedule: dict[str, object]
    n_rows: int
    first_decision_event_index: int
    last_decision_event_index: int
    first_decision_local_ts_us: int
    last_decision_local_ts_us: int
    first_decision_event_seq: int
    last_decision_event_seq: int
    first_book_ptr: int
    last_book_ptr: int
    scheduler_version: str
    created_at_utc: str
    decision_grid_hash: str

    def __post_init__(self) -> None:
        if self.schema != DECISION_GRID_SCHEMA:
            raise ValueError("invalid decision grid schema")
        for name in ("tape_schema", "exchange", "symbol", "scheduler_version", "created_at_utc", "decision_grid_hash"):
            object.__setattr__(self, name, _require_nonempty_str(getattr(self, name), name))
        for name in (
            "tape_num_events",
            "tape_num_l2_batches",
            "tape_start_local_ts_us",
            "tape_end_local_ts_us",
            "n_rows",
        ):
            object.__setattr__(self, name, _require_positive_int(int(getattr(self, name)), name))
        object.__setattr__(self, "tape_num_trades", _require_nonnegative_int(int(self.tape_num_trades), "tape_num_trades"))
        for name in (
            "first_decision_event_index",
            "last_decision_event_index",
            "first_decision_local_ts_us",
            "last_decision_local_ts_us",
            "first_decision_event_seq",
            "last_decision_event_seq",
            "first_book_ptr",
            "last_book_ptr",
        ):
            object.__setattr__(self, name, _require_nonnegative_int(int(getattr(self, name)), name))
        schedule = dict(self.decision_schedule)
        decision_schedule_config_from_dict(schedule)
        object.__setattr__(self, "decision_schedule", schedule)
        if len(self.decision_grid_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.decision_grid_hash):
            raise ValueError("decision_grid_hash must be lowercase sha256 hex")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "tape_schema": self.tape_schema,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "tape_num_events": self.tape_num_events,
            "tape_num_l2_batches": self.tape_num_l2_batches,
            "tape_num_trades": self.tape_num_trades,
            "tape_start_local_ts_us": self.tape_start_local_ts_us,
            "tape_end_local_ts_us": self.tape_end_local_ts_us,
            "decision_schedule": dict(self.decision_schedule),
            "n_rows": self.n_rows,
            "first_decision_event_index": self.first_decision_event_index,
            "last_decision_event_index": self.last_decision_event_index,
            "first_decision_local_ts_us": self.first_decision_local_ts_us,
            "last_decision_local_ts_us": self.last_decision_local_ts_us,
            "first_decision_event_seq": self.first_decision_event_seq,
            "last_decision_event_seq": self.last_decision_event_seq,
            "first_book_ptr": self.first_book_ptr,
            "last_book_ptr": self.last_book_ptr,
            "scheduler_version": self.scheduler_version,
            "created_at_utc": self.created_at_utc,
            "decision_grid_hash": self.decision_grid_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "DecisionGridMetadata":
        if not isinstance(raw, Mapping):
            raise ValueError("metadata must be a mapping")
        required = (
            "schema",
            "tape_schema",
            "exchange",
            "symbol",
            "tape_num_events",
            "tape_num_l2_batches",
            "tape_num_trades",
            "tape_start_local_ts_us",
            "tape_end_local_ts_us",
            "decision_schedule",
            "n_rows",
            "first_decision_event_index",
            "last_decision_event_index",
            "first_decision_local_ts_us",
            "last_decision_local_ts_us",
            "first_decision_event_seq",
            "last_decision_event_seq",
            "first_book_ptr",
            "last_book_ptr",
            "scheduler_version",
            "created_at_utc",
            "decision_grid_hash",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"decision grid metadata missing fields: {missing}")
        return cls(**{key: raw[key] for key in required})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DecisionGrid:
    metadata: DecisionGridMetadata
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    decision_event_seq: np.ndarray
    book_ptr: np.ndarray
    reason_code: np.ndarray
    reason_flags: np.ndarray
    elapsed_since_prev_decision_us: np.ndarray
    events_since_prev_decision: np.ndarray
    l2_events_since_prev_decision: np.ndarray
    trade_events_since_prev_decision: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, DecisionGridMetadata):
            raise ValueError("metadata must be DecisionGridMetadata")
        arrays = self.arrays_dict(copy=False)
        cleaned = {name: _coerce_grid_array(arrays[name], name) for name in DECISION_GRID_ARRAY_ORDER}
        n_rows = self.metadata.n_rows
        for name, arr in cleaned.items():
            if arr.shape != (n_rows,):
                raise ValueError(f"{name} length must equal metadata.n_rows")
        if n_rows <= 0:
            raise ValueError("decision grid must contain at least one row")
        idx = cleaned["decision_event_index"]
        ts = cleaned["decision_local_ts_us"]
        seq = cleaned["decision_event_seq"]
        bp = cleaned["book_ptr"]
        if (idx < 0).any() or (bp < 0).any() or (seq < 0).any():
            raise ValueError("decision grid row pointers must be nonnegative")
        if (ts <= 0).any():
            raise ValueError("decision_local_ts_us must be positive")
        if n_rows > 1:
            if (np.diff(idx) <= 0).any():
                raise ValueError("decision_event_index must be strictly increasing")
            if (np.diff(ts) <= 0).any():
                raise ValueError("decision_local_ts_us must be strictly increasing")
        if int(idx[0]) != self.metadata.first_decision_event_index or int(idx[-1]) != self.metadata.last_decision_event_index:
            raise ValueError("metadata first/last decision_event_index mismatch")
        if int(ts[0]) != self.metadata.first_decision_local_ts_us or int(ts[-1]) != self.metadata.last_decision_local_ts_us:
            raise ValueError("metadata first/last decision_local_ts_us mismatch")
        if int(seq[0]) != self.metadata.first_decision_event_seq or int(seq[-1]) != self.metadata.last_decision_event_seq:
            raise ValueError("metadata first/last decision_event_seq mismatch")
        if int(bp[0]) != self.metadata.first_book_ptr or int(bp[-1]) != self.metadata.last_book_ptr:
            raise ValueError("metadata first/last book_ptr mismatch")
        expected_hash = compute_decision_grid_hash(self.metadata.as_dict(), cleaned)
        if expected_hash != self.metadata.decision_grid_hash:
            raise ValueError("decision_grid_hash mismatch")
        for name, arr in cleaned.items():
            object.__setattr__(self, name, arr)

    @property
    def n_rows(self) -> int:
        return self.metadata.n_rows

    @property
    def decision_schedule(self) -> dict[str, object]:
        return dict(self.metadata.decision_schedule)

    @property
    def decision_grid_hash(self) -> str:
        return self.metadata.decision_grid_hash

    def arrays_dict(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        out = {name: np.asarray(getattr(self, name)) for name in DECISION_GRID_ARRAY_ORDER}
        if copy:
            return {name: np.array(arr, copy=True) for name, arr in out.items()}
        return out


def decision_grid_metadata_from_tape(
    tape: ExecutionTape,
    *,
    schedule_config: DecisionScheduleConfig,
    arrays: Mapping[str, np.ndarray],
    created_at_utc: str | None = None,
    scheduler_version: str = SCHEDULER_VERSION,
) -> DecisionGridMetadata:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(schedule_config, DecisionScheduleConfig):
        raise ValueError("schedule_config must be DecisionScheduleConfig")
    coerced = {name: _coerce_grid_array(arrays[name], name) for name in DECISION_GRID_ARRAY_ORDER}
    n_rows = int(coerced["decision_event_index"].shape[0])
    _require_positive_int(n_rows, "n_rows")
    base = {
        "schema": DECISION_GRID_SCHEMA,
        "tape_schema": tape.manifest.schema,
        "exchange": tape.manifest.exchange,
        "symbol": tape.manifest.symbol,
        "tape_num_events": tape.manifest.num_events,
        "tape_num_l2_batches": tape.manifest.num_l2_batches,
        "tape_num_trades": tape.manifest.num_trades,
        "tape_start_local_ts_us": tape.manifest.start_local_ts_us,
        "tape_end_local_ts_us": tape.manifest.end_local_ts_us,
        "decision_schedule": schedule_config.as_dict(),
        "n_rows": n_rows,
        "first_decision_event_index": int(coerced["decision_event_index"][0]),
        "last_decision_event_index": int(coerced["decision_event_index"][-1]),
        "first_decision_local_ts_us": int(coerced["decision_local_ts_us"][0]),
        "last_decision_local_ts_us": int(coerced["decision_local_ts_us"][-1]),
        "first_decision_event_seq": int(coerced["decision_event_seq"][0]),
        "last_decision_event_seq": int(coerced["decision_event_seq"][-1]),
        "first_book_ptr": int(coerced["book_ptr"][0]),
        "last_book_ptr": int(coerced["book_ptr"][-1]),
        "scheduler_version": scheduler_version,
        "created_at_utc": created_at_utc or _utc_now_iso(),
        "decision_grid_hash": "",
    }
    base["decision_grid_hash"] = compute_decision_grid_hash(base, coerced)
    return DecisionGridMetadata.from_dict(base)


def save_decision_grid_npz(path: str | Path, grid: DecisionGrid, *, overwrite: bool = False) -> None:
    if not isinstance(grid, DecisionGrid):
        raise ValueError("grid must be DecisionGrid")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    payload = grid.arrays_dict(copy=False)
    payload["schema"] = np.array(DECISION_GRID_SCHEMA)
    payload["metadata_json"] = np.array(json.dumps(grid.metadata.as_dict(), sort_keys=True))
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)


def load_decision_grid_npz(path: str | Path, *, mmap_mode: str | None = None) -> DecisionGrid:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, mmap_mode=mmap_mode, allow_pickle=False) as data:
        keys = set(data.files)
        if "schema" not in keys:
            raise ValueError("decision grid NPZ missing schema")
        schema = str(np.asarray(data["schema"]).item())
        if schema != DECISION_GRID_SCHEMA:
            raise ValueError("decision grid schema mismatch")
        required = set(DECISION_GRID_ARRAY_ORDER) | {"metadata_json"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"decision grid NPZ missing required arrays: {missing}")
        metadata = DecisionGridMetadata.from_dict(json.loads(str(np.asarray(data["metadata_json"]).item())))
        arrays = {name: np.array(data[name], copy=True) for name in DECISION_GRID_ARRAY_ORDER}
    return DecisionGrid(metadata=metadata, **arrays)


def decision_grid_summary(grid: DecisionGrid, *, path: str | None = None) -> dict[str, object]:
    if not isinstance(grid, DecisionGrid):
        raise ValueError("grid must be DecisionGrid")
    return {
        "schema": DECISION_GRID_SCHEMA,
        "path": path,
        "decision_grid_hash": grid.decision_grid_hash,
        "n_rows": grid.n_rows,
        "decision_schedule": grid.decision_schedule,
        "scheduler_version": grid.metadata.scheduler_version,
        "first_decision_event_index": int(grid.decision_event_index[0]),
        "last_decision_event_index": int(grid.decision_event_index[-1]),
        "first_decision_local_ts_us": int(grid.decision_local_ts_us[0]),
        "last_decision_local_ts_us": int(grid.decision_local_ts_us[-1]),
        "first_decision_event_seq": int(grid.decision_event_seq[0]),
        "last_decision_event_seq": int(grid.decision_event_seq[-1]),
    }


def decision_grid_lineage(grid: DecisionGrid, *, path: str | None = None) -> dict[str, object]:
    if not isinstance(grid, DecisionGrid):
        raise ValueError("grid must be DecisionGrid")
    return {
        "decision_grid_npz": path,
        "decision_grid_schema": grid.metadata.schema,
        "decision_grid_hash": grid.decision_grid_hash,
        "decision_grid_n_rows": grid.n_rows,
        "decision_schedule": grid.decision_schedule,
        "scheduler_version": grid.metadata.scheduler_version,
    }


def validate_decision_grid_for_execution_tape(grid: DecisionGrid, tape: ExecutionTape) -> None:
    if not isinstance(grid, DecisionGrid):
        raise ValueError("grid must be DecisionGrid")
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    meta = grid.metadata
    expected = {
        "tape_schema": tape.manifest.schema,
        "exchange": tape.manifest.exchange,
        "symbol": tape.manifest.symbol,
        "tape_num_events": tape.manifest.num_events,
        "tape_num_l2_batches": tape.manifest.num_l2_batches,
        "tape_num_trades": tape.manifest.num_trades,
        "tape_start_local_ts_us": tape.manifest.start_local_ts_us,
        "tape_end_local_ts_us": tape.manifest.end_local_ts_us,
    }
    for key, value in expected.items():
        if getattr(meta, key) != value:
            raise ValueError(f"decision grid metadata mismatch for {key}: expected={value!r} actual={getattr(meta, key)!r}")

    events = tape.arrays.events
    l2_events = tape.arrays.l2_events
    if (grid.decision_event_index >= len(events)).any():
        raise ValueError("decision grid contains event indices outside tape")
    if (grid.book_ptr >= len(l2_events)).any():
        raise ValueError("decision grid contains book_ptr outside tape")
    for row in range(grid.n_rows):
        event_index = int(grid.decision_event_index[row])
        event = events[event_index]
        if int(event["event_type_code"]) != EVENT_TYPE_CODE_L2_BATCH:
            raise ValueError("decision grid rows must point to L2 tape events")
        if int(event["book_ptr"]) != int(grid.book_ptr[row]):
            raise ValueError("decision grid book_ptr must match tape event book_ptr")
        if int(event["event_seq"]) != int(grid.decision_event_seq[row]):
            raise ValueError("decision grid decision_event_seq must match tape event_seq")
        if int(event["local_ts_us"]) != int(grid.decision_local_ts_us[row]):
            raise ValueError("decision grid decision_local_ts_us must match tape event local_ts_us")
        book = l2_events[int(grid.book_ptr[row])]
        best_bid_tick = int(book["best_bid_tick"])
        best_ask_tick = int(book["best_ask_tick"])
        if best_bid_tick <= 0 or best_ask_tick <= best_bid_tick:
            raise ValueError("decision grid rows must point to valid two-sided L2 book events")


__all__ = [
    "DECISION_GRID_SCHEMA",
    "DECISION_GRID_FILENAME",
    "DECISION_GRID_SUMMARY_FILENAME",
    "SCHEDULER_VERSION",
    "DECISION_GRID_ARRAY_ORDER",
    "DecisionGridMetadata",
    "DecisionGrid",
    "compute_decision_grid_hash",
    "decision_grid_metadata_from_tape",
    "save_decision_grid_npz",
    "load_decision_grid_npz",
    "decision_grid_summary",
    "decision_grid_lineage",
    "validate_decision_grid_for_execution_tape",
]
