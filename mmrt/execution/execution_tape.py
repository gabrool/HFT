"""Compact NumPy execution-tape arrays for reconstructed L2 and trade events.

This module bridges execution-layer Python objects into fixed-width structured
arrays that can be saved as simple ``.npy`` files and loaded cheaply, including
via NumPy memory maps. It intentionally performs no market-data IO,
reconstruction, merge-policy decisions, fill simulation, observation building,
or ML/RL work.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any, Mapping

import numpy as np

from mmrt.contracts import AggressorSide, TardisDataType
from mmrt.execution.contracts import (
    ExecutionEventType,
    ExecutionTapeFormat,
    ExecutionTapeManifest,
    SymbolSpec,
    TradePrint,
)
from mmrt.execution.event_merge import MergedExecutionEvent
from mmrt.execution.l2_reconstructor import ReconstructedL2Event

EXECUTION_TAPE_SCHEMA_VERSION = "mmrt_execution_tape_v1"
MANIFEST_FILENAME = "manifest.json"
ARRAYS_DIRNAME = "arrays"

EVENTS_ARRAY_NAME = "events"
L2_EVENTS_ARRAY_NAME = "l2_events"
TRADES_ARRAY_NAME = "trades"

EVENT_TYPE_CODE_L2_BATCH = 1
EVENT_TYPE_CODE_TRADE = 2

EVENT_DTYPE = np.dtype(
    [
        ("event_seq", "<i8"),
        ("local_ts_us", "<i8"),
        ("ts_us", "<i8"),
        ("event_type_code", "i1"),
        ("book_ptr", "<i8"),
        ("trade_ptr", "<i8"),
    ]
)

L2_EVENT_DTYPE = np.dtype(
    [
        ("batch_seq", "<i8"),
        ("local_ts_us", "<i8"),
        ("min_ts_us", "<i8"),
        ("max_ts_us", "<i8"),
        ("num_updates", "<i4"),
        ("is_snapshot_batch", "?"),
        ("best_bid_tick", "<i8"),
        ("best_ask_tick", "<i8"),
        ("best_bid_size", "<f4"),
        ("best_ask_size", "<f4"),
        ("bid_depth", "<i4"),
        ("ask_depth", "<i4"),
        ("crossed_repaired", "?"),
        ("crossed_levels_removed", "<i4"),
    ]
)

TRADE_DTYPE = np.dtype(
    [
        ("local_ts_us", "<i8"),
        ("ts_us", "<i8"),
        ("side_code", "i1"),
        ("price_tick", "<i8"),
        ("amount", "<f4"),
        ("source_row", "<i8"),
    ]
)

_EXPECTED_ARRAY_NAMES = (EVENTS_ARRAY_NAME, L2_EVENTS_ARRAY_NAME, TRADES_ARRAY_NAME)
_ALLOWED_MMAP_MODES = (None, "r", "r+")


@dataclass(frozen=True, slots=True)
class ExecutionTapeArrays:
    events: np.ndarray
    l2_events: np.ndarray
    trades: np.ndarray

    def __post_init__(self) -> None:
        _validate_array(self.events, EVENT_DTYPE, "events")
        _validate_array(self.l2_events, L2_EVENT_DTYPE, "l2_events")
        _validate_array(self.trades, TRADE_DTYPE, "trades")
        _validate_events_array(self.events, len(self.l2_events), len(self.trades))


@dataclass(frozen=True, slots=True)
class ExecutionTape:
    manifest: ExecutionTapeManifest
    arrays: ExecutionTapeArrays

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ExecutionTapeManifest):
            raise ValueError("manifest must be ExecutionTapeManifest")
        if not isinstance(self.arrays, ExecutionTapeArrays):
            raise ValueError("arrays must be ExecutionTapeArrays")

        if self.manifest.num_events != len(self.arrays.events):
            raise ValueError("manifest.num_events must match events length")
        if self.manifest.num_l2_batches != len(self.arrays.l2_events):
            raise ValueError("manifest.num_l2_batches must match l2_events length")
        if self.manifest.num_trades != len(self.arrays.trades):
            raise ValueError("manifest.num_trades must match trades length")
        if self.manifest.num_decisions != 0:
            raise ValueError("execution tape v1 requires num_decisions == 0")
        if tuple(self.manifest.array_names) != _EXPECTED_ARRAY_NAMES:
            raise ValueError("manifest.array_names must be events, l2_events, trades")
        if self.manifest.schema_version != EXECUTION_TAPE_SCHEMA_VERSION:
            raise ValueError("unsupported execution tape schema_version")
        if self.manifest.tape_format != ExecutionTapeFormat.L2_TRADES_ARRAYS_V1:
            raise ValueError("unsupported execution tape format")
        if len(self.arrays.events) == 0:
            raise ValueError("execution tape must contain at least one event")

        start_local_ts_us = int(self.arrays.events["local_ts_us"][0])
        end_local_ts_us = int(self.arrays.events["local_ts_us"][-1])
        if end_local_ts_us <= start_local_ts_us:
            end_local_ts_us = start_local_ts_us + 1
        if self.manifest.start_local_ts_us != start_local_ts_us:
            raise ValueError("manifest.start_local_ts_us must match events")
        if self.manifest.end_local_ts_us != end_local_ts_us:
            raise ValueError("manifest.end_local_ts_us must match events")


def execution_tape_manifest_to_dict(manifest: ExecutionTapeManifest) -> dict[str, Any]:
    if not isinstance(manifest, ExecutionTapeManifest):
        raise ValueError("manifest must be ExecutionTapeManifest")
    spec = manifest.symbol_spec
    return {
        "schema_version": manifest.schema_version,
        "tape_format": manifest.tape_format.value,
        "exchange": manifest.exchange,
        "symbol": manifest.symbol,
        "symbol_spec": {
            "exchange": spec.exchange,
            "symbol": spec.symbol,
            "tick_size": spec.tick_size,
            "step_size": spec.step_size,
            "min_qty": spec.min_qty,
            "max_qty": spec.max_qty,
            "min_notional": spec.min_notional,
            "contract_size": spec.contract_size,
        },
        "source_data_types": [value.value for value in manifest.source_data_types],
        "array_names": list(manifest.array_names),
        "num_events": manifest.num_events,
        "num_l2_batches": manifest.num_l2_batches,
        "num_trades": manifest.num_trades,
        "num_decisions": manifest.num_decisions,
        "start_local_ts_us": manifest.start_local_ts_us,
        "end_local_ts_us": manifest.end_local_ts_us,
        "created_at_utc": manifest.created_at_utc,
        "notes": dict(manifest.notes or {}),
    }


def execution_tape_manifest_from_dict(payload: Mapping[str, Any]) -> ExecutionTapeManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    symbol_spec_payload = payload.get("symbol_spec")
    if not isinstance(symbol_spec_payload, Mapping):
        raise ValueError("symbol_spec must be a mapping")
    symbol_spec = SymbolSpec(
        exchange=_require_nonempty_str(symbol_spec_payload.get("exchange"), "symbol_spec.exchange"),
        symbol=_require_nonempty_str(symbol_spec_payload.get("symbol"), "symbol_spec.symbol"),
        tick_size=_require_finite_float(symbol_spec_payload.get("tick_size"), "symbol_spec.tick_size"),
        step_size=_require_finite_float(symbol_spec_payload.get("step_size"), "symbol_spec.step_size"),
        min_qty=_require_finite_float(symbol_spec_payload.get("min_qty"), "symbol_spec.min_qty"),
        max_qty=_require_finite_float(symbol_spec_payload.get("max_qty"), "symbol_spec.max_qty"),
        min_notional=_require_finite_float(symbol_spec_payload.get("min_notional", 0.0), "symbol_spec.min_notional"),
        contract_size=_require_finite_float(symbol_spec_payload.get("contract_size", 1.0), "symbol_spec.contract_size"),
    )
    return ExecutionTapeManifest(
        schema_version=_require_nonempty_str(payload.get("schema_version"), "schema_version"),
        tape_format=ExecutionTapeFormat(payload.get("tape_format")),
        exchange=_require_nonempty_str(payload.get("exchange"), "exchange"),
        symbol=_require_nonempty_str(payload.get("symbol"), "symbol"),
        symbol_spec=symbol_spec,
        source_data_types=tuple(TardisDataType(value) for value in payload.get("source_data_types", ())),
        array_names=tuple(payload.get("array_names", ())),
        num_events=_require_nonnegative_int(payload.get("num_events"), "num_events"),
        num_l2_batches=_require_nonnegative_int(payload.get("num_l2_batches"), "num_l2_batches"),
        num_trades=_require_nonnegative_int(payload.get("num_trades"), "num_trades"),
        num_decisions=_require_nonnegative_int(payload.get("num_decisions"), "num_decisions"),
        start_local_ts_us=_require_positive_int(payload.get("start_local_ts_us"), "start_local_ts_us"),
        end_local_ts_us=_require_positive_int(payload.get("end_local_ts_us"), "end_local_ts_us"),
        created_at_utc=payload.get("created_at_utc", ""),
        notes=dict(payload.get("notes") or {}),
    )


def build_execution_tape(
    *,
    symbol_spec: SymbolSpec,
    l2_events: tuple[ReconstructedL2Event, ...] | list[ReconstructedL2Event],
    trades: tuple[TradePrint, ...] | list[TradePrint],
    merged_events: tuple[MergedExecutionEvent, ...] | list[MergedExecutionEvent],
    created_at_utc: str = "",
    notes: Mapping[str, str] | None = None,
) -> ExecutionTape:
    if not isinstance(symbol_spec, SymbolSpec):
        raise ValueError("symbol_spec must be SymbolSpec")
    if not isinstance(created_at_utc, str):
        raise ValueError("created_at_utc must be str")
    clean_notes = _coerce_notes(notes)

    l2_events_tuple = _coerce_tuple(l2_events, ReconstructedL2Event, "l2_events")
    trades_tuple = _coerce_tuple(trades, TradePrint, "trades")
    merged_events_tuple = _coerce_tuple(merged_events, MergedExecutionEvent, "merged_events")
    if not merged_events_tuple:
        raise ValueError("execution tape must contain at least one event")

    _validate_non_decreasing_local_ts(l2_events_tuple, lambda item: item.local_ts_us, "l2_events")
    _validate_non_decreasing_local_ts(trades_tuple, lambda item: item.local_ts_us, "trades")
    _validate_non_decreasing_local_ts(merged_events_tuple, lambda item: item.local_ts_us, "merged_events")

    l2_arr = _build_l2_events_array(l2_events_tuple)
    trade_arr = _build_trades_array(trades_tuple)
    events_arr = _build_events_array(merged_events_tuple, l2_events_tuple, trades_tuple)

    start_local_ts_us = int(events_arr["local_ts_us"][0])
    end_local_ts_us = int(events_arr["local_ts_us"][-1])
    if end_local_ts_us <= start_local_ts_us:
        end_local_ts_us = start_local_ts_us + 1

    manifest = ExecutionTapeManifest(
        schema_version=EXECUTION_TAPE_SCHEMA_VERSION,
        tape_format=ExecutionTapeFormat.L2_TRADES_ARRAYS_V1,
        exchange=symbol_spec.exchange,
        symbol=symbol_spec.symbol,
        symbol_spec=symbol_spec,
        source_data_types=(TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES),
        array_names=_EXPECTED_ARRAY_NAMES,
        num_events=len(events_arr),
        num_l2_batches=len(l2_arr),
        num_trades=len(trade_arr),
        num_decisions=0,
        start_local_ts_us=start_local_ts_us,
        end_local_ts_us=end_local_ts_us,
        created_at_utc=created_at_utc,
        notes=clean_notes,
    )
    return ExecutionTape(manifest=manifest, arrays=ExecutionTapeArrays(events=events_arr, l2_events=l2_arr, trades=trade_arr))


def save_execution_tape(tape: ExecutionTape, root: str | Path, *, overwrite: bool = False) -> None:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    root_path = Path(root)
    arrays_dir = root_path / ARRAYS_DIRNAME
    manifest_path = root_path / MANIFEST_FILENAME
    paths = (
        manifest_path,
        arrays_dir / f"{EVENTS_ARRAY_NAME}.npy",
        arrays_dir / f"{L2_EVENTS_ARRAY_NAME}.npy",
        arrays_dir / f"{TRADES_ARRAY_NAME}.npy",
    )
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"execution tape files already exist under {root_path}")

    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / f"{EVENTS_ARRAY_NAME}.npy", tape.arrays.events)
    np.save(arrays_dir / f"{L2_EVENTS_ARRAY_NAME}.npy", tape.arrays.l2_events)
    np.save(arrays_dir / f"{TRADES_ARRAY_NAME}.npy", tape.arrays.trades)

    manifest_text = json.dumps(execution_tape_manifest_to_dict(tape.manifest), sort_keys=True, indent=2)
    tmp_path = root_path / f"{MANIFEST_FILENAME}.tmp"
    tmp_path.write_text(manifest_text + "\n", encoding="utf-8")
    tmp_path.replace(manifest_path)


def load_execution_tape(root: str | Path, *, mmap_mode: str | None = None) -> ExecutionTape:
    if mmap_mode not in _ALLOWED_MMAP_MODES:
        raise ValueError("mmap_mode must be None, 'r', or 'r+'")
    root_path = Path(root)
    payload = json.loads((root_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest = execution_tape_manifest_from_dict(payload)
    arrays_dir = root_path / ARRAYS_DIRNAME
    arrays = ExecutionTapeArrays(
        events=np.load(arrays_dir / f"{EVENTS_ARRAY_NAME}.npy", mmap_mode=mmap_mode),
        l2_events=np.load(arrays_dir / f"{L2_EVENTS_ARRAY_NAME}.npy", mmap_mode=mmap_mode),
        trades=np.load(arrays_dir / f"{TRADES_ARRAY_NAME}.npy", mmap_mode=mmap_mode),
    )
    return ExecutionTape(manifest=manifest, arrays=arrays)


def _build_events_array(
    merged_events: tuple[MergedExecutionEvent, ...],
    l2_events: tuple[ReconstructedL2Event, ...],
    trades: tuple[TradePrint, ...],
) -> np.ndarray:
    events_arr = np.empty(len(merged_events), dtype=EVENT_DTYPE)
    prev_local_ts_us: int | None = None
    for i, event in enumerate(merged_events):
        if event.ref.event_seq != i:
            raise ValueError("merged event_seq values must be contiguous from 0")
        if prev_local_ts_us is not None and event.local_ts_us < prev_local_ts_us:
            raise ValueError("merged_events local_ts_us must be nondecreasing")
        prev_local_ts_us = event.local_ts_us

        if event.ref.event_type == ExecutionEventType.L2_BATCH:
            if event.ref.book_ptr < 0 or event.ref.book_ptr >= len(l2_events):
                raise ValueError("merged L2 event book_ptr out of range")
            if event.ref.trade_ptr != -1:
                raise ValueError("merged L2 event trade_ptr must be -1")
            if event.l2_event is not l2_events[event.ref.book_ptr]:
                raise ValueError("merged L2 event pointer does not match l2_events")
            code = EVENT_TYPE_CODE_L2_BATCH
        elif event.ref.event_type == ExecutionEventType.TRADE:
            if event.ref.trade_ptr < 0 or event.ref.trade_ptr >= len(trades):
                raise ValueError("merged trade event trade_ptr out of range")
            if event.ref.book_ptr != -1:
                raise ValueError("merged trade event book_ptr must be -1")
            if event.trade is not trades[event.ref.trade_ptr]:
                raise ValueError("merged trade event pointer does not match trades")
            code = EVENT_TYPE_CODE_TRADE
        else:
            raise ValueError("execution tape v1 supports only L2_BATCH and TRADE events")

        events_arr[i] = (
            event.ref.event_seq,
            event.local_ts_us,
            event.ts_us,
            code,
            event.ref.book_ptr,
            event.ref.trade_ptr,
        )
    return events_arr


def _build_l2_events_array(l2_events: tuple[ReconstructedL2Event, ...]) -> np.ndarray:
    l2_arr = np.empty(len(l2_events), dtype=L2_EVENT_DTYPE)
    for i, event in enumerate(l2_events):
        if event.book_top is None:
            best_bid_tick = -1
            best_ask_tick = -1
            best_bid_size = 0.0
            best_ask_size = 0.0
        else:
            best_bid_tick = event.book_top.best_bid_tick
            best_ask_tick = event.book_top.best_ask_tick
            best_bid_size = _require_finite_float(event.book_top.best_bid_size, "book_top.best_bid_size")
            best_ask_size = _require_finite_float(event.book_top.best_ask_size, "book_top.best_ask_size")
            if best_bid_size < 0.0 or best_ask_size < 0.0:
                raise ValueError("book_top sizes must be nonnegative")
        l2_arr[i] = (
            event.batch_seq,
            event.local_ts_us,
            event.min_ts_us,
            event.max_ts_us,
            event.num_updates,
            event.is_snapshot_batch,
            best_bid_tick,
            best_ask_tick,
            best_bid_size,
            best_ask_size,
            event.bid_depth,
            event.ask_depth,
            event.crossed_repaired,
            event.crossed_levels_removed,
        )
    return l2_arr


def _build_trades_array(trades: tuple[TradePrint, ...]) -> np.ndarray:
    trade_arr = np.empty(len(trades), dtype=TRADE_DTYPE)
    for i, trade in enumerate(trades):
        amount = _require_finite_float(trade.amount, "trade.amount")
        if amount < 0.0:
            raise ValueError("trade amount must be nonnegative")
        trade_arr[i] = (
            trade.local_ts_us,
            trade.ts_us,
            _trade_side_code(trade.side),
            trade.price_tick,
            amount,
            trade.source_row,
        )
    return trade_arr


def _trade_side_code(side: AggressorSide) -> int:
    if side.value == "buy":
        return 1
    if side.value == "sell":
        return -1
    if side.value == "unknown":
        return 0
    raise ValueError(f"unsupported trade side {side!r}")


def _validate_array(array: np.ndarray, dtype: np.dtype, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if array.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if array.dtype.hasobject:
        raise ValueError(f"{name} must not use object dtype")


def _validate_events_array(events: np.ndarray, num_l2_events: int, num_trades: int) -> None:
    if len(events) == 0:
        return
    expected_event_seq = np.arange(len(events), dtype=np.int64)
    if not np.array_equal(events["event_seq"], expected_event_seq):
        raise ValueError("events event_seq values must be contiguous from 0")
    if np.any(events["local_ts_us"][1:] < events["local_ts_us"][:-1]):
        raise ValueError("events local_ts_us must be nondecreasing")

    event_type_codes = events["event_type_code"]
    book_ptrs = events["book_ptr"]
    trade_ptrs = events["trade_ptr"]
    for i in range(len(events)):
        code = int(event_type_codes[i])
        book_ptr = int(book_ptrs[i])
        trade_ptr = int(trade_ptrs[i])
        if code == EVENT_TYPE_CODE_L2_BATCH:
            if book_ptr < 0 or book_ptr >= num_l2_events or trade_ptr != -1:
                raise ValueError("L2 events require valid book_ptr and trade_ptr == -1")
        elif code == EVENT_TYPE_CODE_TRADE:
            if trade_ptr < 0 or trade_ptr >= num_trades or book_ptr != -1:
                raise ValueError("trade events require valid trade_ptr and book_ptr == -1")
        else:
            raise ValueError("unknown event_type_code")


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def _coerce_tuple(values: Any, item_type: type, name: str) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable, not a string/bytes value")
    try:
        seq = values if isinstance(values, tuple) else tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be iterable") from exc
    for idx, item in enumerate(seq):
        if not isinstance(item, item_type):
            raise ValueError(f"{name}[{idx}] must be {item_type.__name__}")
    return seq


def _validate_non_decreasing_local_ts(items: tuple, get_ts, name: str) -> None:
    prev_ts: int | None = None
    for idx, item in enumerate(items):
        ts = get_ts(item)
        if isinstance(ts, bool) or not isinstance(ts, int):
            raise ValueError(f"{name}[{idx}] local_ts_us must be int")
        if prev_ts is not None and ts < prev_ts:
            raise ValueError(f"{name} local_ts_us must be nondecreasing")
        prev_ts = ts


def _coerce_notes(notes: Mapping[str, str] | None) -> dict[str, str]:
    if notes is None:
        return {}
    if not isinstance(notes, Mapping):
        raise ValueError("notes must be a mapping")
    clean: dict[str, str] = {}
    for key, value in notes.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("notes must be Mapping[str, str]")
        clean[key] = value
    return clean


__all__ = [
    "EXECUTION_TAPE_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "ARRAYS_DIRNAME",
    "EVENTS_ARRAY_NAME",
    "L2_EVENTS_ARRAY_NAME",
    "TRADES_ARRAY_NAME",
    "EVENT_TYPE_CODE_L2_BATCH",
    "EVENT_TYPE_CODE_TRADE",
    "EVENT_DTYPE",
    "L2_EVENT_DTYPE",
    "TRADE_DTYPE",
    "ExecutionTapeArrays",
    "ExecutionTape",
    "execution_tape_manifest_to_dict",
    "execution_tape_manifest_from_dict",
    "build_execution_tape",
    "save_execution_tape",
    "load_execution_tape",
]
