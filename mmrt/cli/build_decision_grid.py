"""Build the canonical decision_grid.npz for one execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.decision_grid import (
    DECISION_GRID_ARRAY_ORDER,
    DECISION_GRID_FILENAME,
    DECISION_GRID_SCHEMA,
    DECISION_GRID_SUMMARY_FILENAME,
    DecisionGrid,
    decision_grid_metadata_from_tape,
    decision_grid_summary,
    save_decision_grid_npz,
    validate_decision_grid_for_execution_tape,
)
from mmrt.execution.execution_tape import (
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    ExecutionTapeValidationMode,
    load_execution_tape,
)
from mmrt.execution.execution_tape_writer import NpyChunkWriter
from mmrt.features.schedule import (
    DECISION_REASON_CODE_NAMES,
    DEFAULT_L1_SIZE_CHANGE_FRACTION,
    DEFAULT_MAX_DECISION_INTERVAL_US,
    DEFAULT_MIN_DECISION_INTERVAL_US,
    DecisionSchedule,
    DecisionScheduleConfig,
)

__all__ = [
    "BuildDecisionGridConfig",
    "build_decision_grid_from_config",
    "build_arg_parser",
    "main",
]


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _default_output_npz(tape_root: str) -> Path:
    return Path(tape_root) / DECISION_GRID_FILENAME


def _default_output_json(tape_root: str) -> Path:
    return Path(tape_root) / DECISION_GRID_SUMMARY_FILENAME


def _write_json_atomic(path: Path, payload: Mapping[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output_json already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stats(values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "min": None, "max": None, "p50": None, "p95": None}
    q = np.quantile(arr, [0.50, 0.95])
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(q[0]),
        "p95": float(q[1]),
    }


def _valid_l2_book(tape, book_ptr: int) -> bool:
    if book_ptr < 0:
        return False
    row = tape.arrays.l2_events[book_ptr]
    best_bid_tick = int(row["best_bid_tick"])
    best_ask_tick = int(row["best_ask_tick"])
    return best_bid_tick > 0 and best_ask_tick > best_bid_tick


class _DecisionGridWriters:
    def __init__(self, output_npz: Path, *, chunk_rows: int) -> None:
        self.chunk_dir = output_npz.parent / f".{output_npz.name}.decision_grid_chunks"
        self.arrays_dir = output_npz.parent / f".{output_npz.name}.decision_grid_arrays"
        shutil.rmtree(self.chunk_dir, ignore_errors=True)
        shutil.rmtree(self.arrays_dir, ignore_errors=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.writers = {
            "decision_event_index": NpyChunkWriter("decision_event_index", np.int64, (), chunk_rows, self.chunk_dir),
            "decision_local_ts_us": NpyChunkWriter("decision_local_ts_us", np.int64, (), chunk_rows, self.chunk_dir),
            "decision_event_seq": NpyChunkWriter("decision_event_seq", np.int64, (), chunk_rows, self.chunk_dir),
            "book_ptr": NpyChunkWriter("book_ptr", np.int64, (), chunk_rows, self.chunk_dir),
            "reason_code": NpyChunkWriter("reason_code", np.int16, (), chunk_rows, self.chunk_dir),
            "reason_flags": NpyChunkWriter("reason_flags", np.int16, (), chunk_rows, self.chunk_dir),
            "elapsed_since_prev_decision_us": NpyChunkWriter("elapsed_since_prev_decision_us", np.int64, (), chunk_rows, self.chunk_dir),
            "events_since_prev_decision": NpyChunkWriter("events_since_prev_decision", np.int64, (), chunk_rows, self.chunk_dir),
            "l2_events_since_prev_decision": NpyChunkWriter("l2_events_since_prev_decision", np.int64, (), chunk_rows, self.chunk_dir),
            "trade_events_since_prev_decision": NpyChunkWriter("trade_events_since_prev_decision", np.int64, (), chunk_rows, self.chunk_dir),
        }

    def append(
        self,
        *,
        decision_event_index: int,
        decision_local_ts_us: int,
        decision_event_seq: int,
        book_ptr: int,
        reason_code: int,
        reason_flags: int,
        elapsed_since_prev_decision_us: int,
        events_since_prev_decision: int,
        l2_events_since_prev_decision: int,
        trade_events_since_prev_decision: int,
    ) -> None:
        self.writers["decision_event_index"].append(decision_event_index)
        self.writers["decision_local_ts_us"].append(decision_local_ts_us)
        self.writers["decision_event_seq"].append(decision_event_seq)
        self.writers["book_ptr"].append(book_ptr)
        self.writers["reason_code"].append(reason_code)
        self.writers["reason_flags"].append(reason_flags)
        self.writers["elapsed_since_prev_decision_us"].append(elapsed_since_prev_decision_us)
        self.writers["events_since_prev_decision"].append(events_since_prev_decision)
        self.writers["l2_events_since_prev_decision"].append(l2_events_since_prev_decision)
        self.writers["trade_events_since_prev_decision"].append(trade_events_since_prev_decision)

    @property
    def total_rows(self) -> int:
        return self.writers["decision_event_index"].total_rows

    def finalize(self) -> dict[str, np.ndarray]:
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        row_counts = {name: writer.finalize(self.arrays_dir / f"{name}.npy") for name, writer in self.writers.items()}
        if len(set(row_counts.values())) != 1:
            raise RuntimeError("decision grid chunk row count mismatch")
        return {name: np.load(self.arrays_dir / f"{name}.npy", mmap_mode="r") for name in DECISION_GRID_ARRAY_ORDER}

    def cleanup(self) -> None:
        shutil.rmtree(self.chunk_dir, ignore_errors=True)
        shutil.rmtree(self.arrays_dir, ignore_errors=True)


class _OpenArrays:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = arrays

    def close(self) -> None:
        for arr in self.arrays.values():
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()


@dataclass(frozen=True, slots=True)
class BuildDecisionGridConfig:
    tape_root: str
    output_npz: str | None = None
    output_json: str | None = None
    min_decision_interval_us: int = DEFAULT_MIN_DECISION_INTERVAL_US
    max_decision_interval_us: int = DEFAULT_MAX_DECISION_INTERVAL_US
    wake_on_trade: bool = True
    wake_on_top_of_book: bool = True
    l1_size_change_fraction: float = DEFAULT_L1_SIZE_CHANGE_FRACTION
    start_event_index: int = 0
    max_decisions: int | None = None
    chunk_rows: int = 100_000
    overwrite: bool = False
    mmap_mode: str | None = "r"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_nonempty_str(self.tape_root, "tape_root"))
        if self.output_npz is not None:
            object.__setattr__(self, "output_npz", _require_nonempty_str(self.output_npz, "output_npz"))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", _require_nonempty_str(self.output_json, "output_json"))
        _require_positive_int(self.min_decision_interval_us, "min_decision_interval_us")
        _require_positive_int(self.max_decision_interval_us, "max_decision_interval_us")
        _require_bool(self.wake_on_trade, "wake_on_trade")
        _require_bool(self.wake_on_top_of_book, "wake_on_top_of_book")
        object.__setattr__(self, "start_event_index", _require_nonnegative_int(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "max_decisions", _optional_positive_int(self.max_decisions, "max_decisions"))
        object.__setattr__(self, "chunk_rows", _require_positive_int(self.chunk_rows, "chunk_rows"))
        object.__setattr__(self, "overwrite", _require_bool(self.overwrite, "overwrite"))
        if self.mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'")
        DecisionScheduleConfig(
            min_decision_interval_us=self.min_decision_interval_us,
            max_decision_interval_us=self.max_decision_interval_us,
            wake_on_trade=self.wake_on_trade,
            wake_on_top_of_book=self.wake_on_top_of_book,
            l1_size_change_fraction=float(self.l1_size_change_fraction),
        )

    def schedule_config(self) -> DecisionScheduleConfig:
        return DecisionScheduleConfig(
            min_decision_interval_us=self.min_decision_interval_us,
            max_decision_interval_us=self.max_decision_interval_us,
            wake_on_trade=self.wake_on_trade,
            wake_on_top_of_book=self.wake_on_top_of_book,
            l1_size_change_fraction=float(self.l1_size_change_fraction),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "tape_root": self.tape_root,
            "output_npz": self.output_npz,
            "output_json": self.output_json,
            "min_decision_interval_us": self.min_decision_interval_us,
            "max_decision_interval_us": self.max_decision_interval_us,
            "wake_on_trade": self.wake_on_trade,
            "wake_on_top_of_book": self.wake_on_top_of_book,
            "l1_size_change_fraction": self.l1_size_change_fraction,
            "start_event_index": self.start_event_index,
            "max_decisions": self.max_decisions,
            "chunk_rows": self.chunk_rows,
            "overwrite": self.overwrite,
            "mmap_mode": self.mmap_mode,
        }


def build_decision_grid_from_config(config: BuildDecisionGridConfig) -> dict[str, object]:
    if not isinstance(config, BuildDecisionGridConfig):
        raise ValueError("config must be BuildDecisionGridConfig")
    output_npz = Path(config.output_npz) if config.output_npz is not None else _default_output_npz(config.tape_root)
    output_json = Path(config.output_json) if config.output_json is not None else _default_output_json(config.tape_root)
    if output_npz.exists() and not config.overwrite:
        raise FileExistsError(f"output_npz already exists: {output_npz}")
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output_json already exists: {output_json}")

    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode, validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY)
    events = tape.arrays.events
    if config.start_event_index >= len(events):
        raise ValueError("start_event_index must be < len(tape.arrays.events)")
    schedule_config = config.schedule_config()
    scheduler = DecisionSchedule(schedule_config)
    writers = _DecisionGridWriters(output_npz, chunk_rows=config.chunk_rows)
    arrays_owner: _OpenArrays | None = None
    total_events_seen = 0
    total_l2_events_seen = 0
    total_trade_events_seen = 0
    valid_l2_events_seen = 0
    last_decision_event_index: int | None = None
    last_decision_local_ts_us: int | None = None
    last_decision_l2_count = 0
    last_decision_trade_count = 0
    try:
        for event_index in range(config.start_event_index, len(events)):
            event = events[event_index]
            total_events_seen += 1
            event_type = int(event["event_type_code"])
            local_ts_us = int(event["local_ts_us"])
            if event_type == EVENT_TYPE_CODE_TRADE:
                total_trade_events_seen += 1
                scheduler.observe_trade(local_ts_us)
                continue
            if event_type != EVENT_TYPE_CODE_L2_BATCH:
                raise ValueError(f"unsupported event_type_code: {event_type}")
            total_l2_events_seen += 1
            book_ptr = int(event["book_ptr"])
            if not _valid_l2_book(tape, book_ptr):
                continue
            valid_l2_events_seen += 1
            book = tape.arrays.l2_events[book_ptr]
            scheduler.observe_book(
                local_ts_us,
                best_bid=float(book["best_bid_tick"]),
                best_ask=float(book["best_ask_tick"]),
                bid_l1_size=float(book["best_bid_size"]),
                ask_l1_size=float(book["best_ask_size"]),
            )
            fire = scheduler.fire_reason(local_ts_us)
            if not fire.should_fire:
                continue
            elapsed = 0 if last_decision_local_ts_us is None else local_ts_us - last_decision_local_ts_us
            events_since = 0 if last_decision_event_index is None else event_index - last_decision_event_index
            l2_since = 0 if last_decision_event_index is None else total_l2_events_seen - last_decision_l2_count
            trade_since = 0 if last_decision_event_index is None else total_trade_events_seen - last_decision_trade_count
            writers.append(
                decision_event_index=event_index,
                decision_local_ts_us=local_ts_us,
                decision_event_seq=int(event["event_seq"]),
                book_ptr=book_ptr,
                reason_code=int(fire.reason_code),
                reason_flags=int(fire.reason_flags),
                elapsed_since_prev_decision_us=elapsed,
                events_since_prev_decision=events_since,
                l2_events_since_prev_decision=l2_since,
                trade_events_since_prev_decision=trade_since,
            )
            scheduler.mark_decision(local_ts_us)
            last_decision_event_index = event_index
            last_decision_local_ts_us = local_ts_us
            last_decision_l2_count = total_l2_events_seen
            last_decision_trade_count = total_trade_events_seen
            if config.max_decisions is not None and writers.total_rows >= config.max_decisions:
                break
        if writers.total_rows <= 0:
            raise ValueError("no decision grid rows emitted")
        arrays = writers.finalize()
        arrays_owner = _OpenArrays(arrays)
        metadata = decision_grid_metadata_from_tape(tape, schedule_config=schedule_config, arrays=arrays)
        grid = DecisionGrid(metadata=metadata, **arrays)
        validate_decision_grid_for_execution_tape(grid, tape)
        save_decision_grid_npz(output_npz, grid, overwrite=config.overwrite)
        reason_values, reason_counts_raw = np.unique(np.asarray(grid.reason_code), return_counts=True)
        reason_counts = {
            DECISION_REASON_CODE_NAMES.get(int(code), str(int(code))): int(count)
            for code, count in zip(reason_values, reason_counts_raw)
        }
        summary = {
            "status": "ok",
            "run_type": "build_decision_grid",
            "schema": DECISION_GRID_SCHEMA,
            "tape_root": str(Path(config.tape_root)),
            "output_npz": str(output_npz),
            "output_json": str(output_json),
            "decision_grid": decision_grid_summary(grid, path=str(output_npz)),
            "reason_counts": reason_counts,
            "interval_stats": {
                "elapsed_since_prev_decision_us": _stats(grid.elapsed_since_prev_decision_us[1:]),
                "events_since_prev_decision": _stats(grid.events_since_prev_decision[1:]),
                "l2_events_since_prev_decision": _stats(grid.l2_events_since_prev_decision[1:]),
                "trade_events_since_prev_decision": _stats(grid.trade_events_since_prev_decision[1:]),
            },
            "counters": {
                "tape_events_seen": total_events_seen,
                "l2_events_seen": total_l2_events_seen,
                "trade_events_seen": total_trade_events_seen,
                "valid_l2_events_seen": valid_l2_events_seen,
                "decision_grid_rows": grid.n_rows,
            },
            "config": config.as_dict(),
        }
        _write_json_atomic(output_json, summary, overwrite=config.overwrite)
        return summary
    finally:
        if arrays_owner is not None:
            arrays_owner.close()
        writers.cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--output-npz")
    parser.add_argument("--output-json")
    parser.add_argument("--min-decision-interval-us", type=int, default=DEFAULT_MIN_DECISION_INTERVAL_US)
    parser.add_argument("--max-decision-interval-us", type=int, default=DEFAULT_MAX_DECISION_INTERVAL_US)
    parser.add_argument("--no-wake-on-trade", dest="wake_on_trade", action="store_false", default=True)
    parser.add_argument("--no-wake-on-top-of-book", dest="wake_on_top_of_book", action="store_false", default=True)
    parser.add_argument("--l1-size-change-fraction", type=float, default=DEFAULT_L1_SIZE_CHANGE_FRACTION)
    parser.add_argument("--start-event-index", type=int, default=0)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildDecisionGridConfig:
    return BuildDecisionGridConfig(
        tape_root=args.tape_root,
        output_npz=args.output_npz,
        output_json=args.output_json,
        min_decision_interval_us=args.min_decision_interval_us,
        max_decision_interval_us=args.max_decision_interval_us,
        wake_on_trade=args.wake_on_trade,
        wake_on_top_of_book=args.wake_on_top_of_book,
        l1_size_change_fraction=args.l1_size_change_fraction,
        start_event_index=args.start_event_index,
        max_decisions=args.max_decisions,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_decision_grid_from_config(_config_from_args(args))
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
