"""Chunked writers for canonical execution-tape NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from mmrt.contracts import TardisDataType
from mmrt.execution.contracts import ExecutionEventType, ExecutionTapeFormat, ExecutionTapeManifest, SymbolSpec
from mmrt.execution.event_merge import MergedExecutionEvent
from mmrt.execution.execution_tape import (
    ARRAYS_DIRNAME,
    BOOK_ASK_SIZES_ARRAY_NAME,
    BOOK_ASK_TICKS_ARRAY_NAME,
    BOOK_BID_SIZES_ARRAY_NAME,
    BOOK_BID_TICKS_ARRAY_NAME,
    EVENT_DTYPE,
    EVENTS_ARRAY_NAME,
    EXECUTION_TAPE_SCHEMA,
    L2_EVENTS_ARRAY_NAME,
    L2_EVENT_DTYPE,
    MANIFEST_FILENAME,
    TRADES_ARRAY_NAME,
    TRADE_DTYPE,
    ExecutionTape,
    ExecutionTapeValidationMode,
    _EXPECTED_ARRAY_NAMES,
    _coerce_validation_mode,
    book_snapshot_to_depth_rows,
    execution_tape_manifest_to_dict,
    l2_event_to_array_row,
    load_execution_tape,
    merged_event_to_array_row,
    trade_print_to_array_row,
)
from mmrt.metadata.rule_compatibility import RuleCompatibilityReport
from mmrt.metadata.symbol_rules import ExchangeSymbolRules


@dataclass(slots=True)
class NpyChunkWriter:
    name: str
    dtype: np.dtype
    row_shape: tuple[int, ...]
    chunk_rows: int
    chunk_dir: Path
    _buffer: np.ndarray = field(init=False, repr=False)
    _used: int = field(init=False, default=0, repr=False)
    _total_rows: int = field(init=False, default=0, repr=False)
    _chunk_index: int = field(init=False, default=0, repr=False)
    _chunk_paths: list[Path] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty str")
        self.dtype = np.dtype(self.dtype)
        self.row_shape = tuple(self.row_shape)
        if isinstance(self.chunk_rows, bool) or not isinstance(self.chunk_rows, int) or self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive int")
        self.chunk_dir = Path(self.chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = np.empty((self.chunk_rows, *self.row_shape), dtype=self.dtype)
        self._used = 0
        self._total_rows = 0
        self._chunk_index = 0
        self._chunk_paths: list[Path] = []

    def append(self, row: Any) -> int:
        if self._used >= self.chunk_rows:
            self.flush()
        row_index = self._total_rows
        self._buffer[self._used] = row
        self._used += 1
        self._total_rows += 1
        return row_index

    def flush(self) -> None:
        if self._used == 0:
            return
        path = self.chunk_dir / f"{self.name}_{self._chunk_index:06d}.npy"
        np.save(path, self._buffer[: self._used])
        self._chunk_paths.append(path)
        self._chunk_index += 1
        self._used = 0

    def finalize(self, output_path: Path) -> int:
        self.flush()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=self.dtype,
            shape=(self._total_rows, *self.row_shape),
        )
        offset = 0
        for path in self._chunk_paths:
            chunk = np.load(path, mmap_mode="r")
            rows = int(chunk.shape[0])
            out[offset : offset + rows] = chunk
            offset += rows
        if offset != self._total_rows:
            raise RuntimeError(f"chunk row count mismatch for {self.name}")
        out.flush()
        del out
        return self._total_rows

    def cleanup(self) -> None:
        for path in self._chunk_paths:
            path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class StreamingExecutionTapeWriterConfig:
    output_root: str
    symbol_spec: SymbolSpec
    symbol_rules: ExchangeSymbolRules
    symbol_rule_compatibility: RuleCompatibilityReport | None = None
    book_depth: int = 25
    chunk_rows: int = 250_000
    overwrite: bool = False
    cleanup_chunks: bool = True
    created_at_utc: str = ""
    notes: Mapping[str, str] | None = None
    validation_mode: ExecutionTapeValidationMode | str = ExecutionTapeValidationMode.SHAPE_ONLY

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be a non-empty str")
        if not isinstance(self.symbol_spec, SymbolSpec):
            raise ValueError("symbol_spec must be SymbolSpec")
        if not isinstance(self.symbol_rules, ExchangeSymbolRules):
            raise ValueError("symbol_rules must be ExchangeSymbolRules")
        if self.symbol_rules.to_symbol_spec() != self.symbol_spec:
            raise ValueError("symbol_spec must equal symbol_rules.to_symbol_spec()")
        if self.symbol_rule_compatibility is not None and not isinstance(self.symbol_rule_compatibility, RuleCompatibilityReport):
            raise ValueError("symbol_rule_compatibility must be None or RuleCompatibilityReport")
        if isinstance(self.book_depth, bool) or not isinstance(self.book_depth, int) or self.book_depth <= 0:
            raise ValueError("book_depth must be a positive int")
        if isinstance(self.chunk_rows, bool) or not isinstance(self.chunk_rows, int) or self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive int")
        if not isinstance(self.overwrite, bool):
            raise ValueError("overwrite must be bool")
        if not isinstance(self.cleanup_chunks, bool):
            raise ValueError("cleanup_chunks must be bool")
        if not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be str")
        if self.notes is not None:
            for key, value in self.notes.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("notes must be Mapping[str, str]")
        object.__setattr__(self, "validation_mode", _coerce_validation_mode(self.validation_mode))


@dataclass(frozen=True, slots=True)
class StreamingExecutionTapeWriteResult:
    tape: ExecutionTape
    chunk_summary: dict[str, object]


class StreamingExecutionTapeWriter:
    def __init__(self, config: StreamingExecutionTapeWriterConfig) -> None:
        if not isinstance(config, StreamingExecutionTapeWriterConfig):
            raise ValueError("config must be StreamingExecutionTapeWriterConfig")
        self.config = config
        self.output_root = Path(config.output_root)
        self.arrays_dir = self.output_root / ARRAYS_DIRNAME
        self.chunk_dir = self.output_root / ".execution_tape_chunks"
        self._check_output_paths()
        if config.overwrite and self.chunk_dir.exists():
            shutil.rmtree(self.chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        depth_shape = (config.book_depth,)
        self._writers = {
            EVENTS_ARRAY_NAME: NpyChunkWriter(EVENTS_ARRAY_NAME, EVENT_DTYPE, (), config.chunk_rows, self.chunk_dir),
            L2_EVENTS_ARRAY_NAME: NpyChunkWriter(L2_EVENTS_ARRAY_NAME, L2_EVENT_DTYPE, (), config.chunk_rows, self.chunk_dir),
            TRADES_ARRAY_NAME: NpyChunkWriter(TRADES_ARRAY_NAME, TRADE_DTYPE, (), config.chunk_rows, self.chunk_dir),
            BOOK_BID_TICKS_ARRAY_NAME: NpyChunkWriter(BOOK_BID_TICKS_ARRAY_NAME, np.int64, depth_shape, config.chunk_rows, self.chunk_dir),
            BOOK_BID_SIZES_ARRAY_NAME: NpyChunkWriter(BOOK_BID_SIZES_ARRAY_NAME, np.float32, depth_shape, config.chunk_rows, self.chunk_dir),
            BOOK_ASK_TICKS_ARRAY_NAME: NpyChunkWriter(BOOK_ASK_TICKS_ARRAY_NAME, np.int64, depth_shape, config.chunk_rows, self.chunk_dir),
            BOOK_ASK_SIZES_ARRAY_NAME: NpyChunkWriter(BOOK_ASK_SIZES_ARRAY_NAME, np.float32, depth_shape, config.chunk_rows, self.chunk_dir),
        }
        self.event_count = 0
        self.l2_count = 0
        self.trade_count = 0
        self.first_local_ts_us: int | None = None
        self.last_local_ts_us: int | None = None
        self._finalized = False

    def _check_output_paths(self) -> None:
        paths = [self.output_root / MANIFEST_FILENAME, self.output_root / "build_summary.json"]
        paths.extend((self.arrays_dir / f"{name}.npy") for name in _EXPECTED_ARRAY_NAMES)
        if not self.config.overwrite:
            existing = [path for path in paths if path.exists()]
            if existing:
                raise FileExistsError(f"execution tape files already exist under {self.output_root}")

    def append(self, event: MergedExecutionEvent) -> None:
        if self._finalized:
            raise RuntimeError("cannot append after finalize")
        if not isinstance(event, MergedExecutionEvent):
            raise ValueError("event must be MergedExecutionEvent")
        if event.ref.event_seq != self.event_count:
            raise ValueError("merged event_seq values must be contiguous from 0")
        if self.last_local_ts_us is not None and event.local_ts_us < self.last_local_ts_us:
            raise ValueError("merged events local_ts_us must be nondecreasing")
        if self.first_local_ts_us is None:
            self.first_local_ts_us = event.local_ts_us
        self.last_local_ts_us = event.local_ts_us

        if event.ref.event_type == ExecutionEventType.L2_BATCH:
            if event.l2_event is None:
                raise ValueError("L2 merged event requires l2_event")
            if event.ref.book_ptr != self.l2_count:
                raise ValueError("L2 merged event book_ptr must match next L2 row")
            self._writers[L2_EVENTS_ARRAY_NAME].append(l2_event_to_array_row(event.l2_event))
            bid_ticks, bid_sizes, ask_ticks, ask_sizes = book_snapshot_to_depth_rows(event.l2_event, book_depth=self.config.book_depth)
            self._writers[BOOK_BID_TICKS_ARRAY_NAME].append(bid_ticks)
            self._writers[BOOK_BID_SIZES_ARRAY_NAME].append(bid_sizes)
            self._writers[BOOK_ASK_TICKS_ARRAY_NAME].append(ask_ticks)
            self._writers[BOOK_ASK_SIZES_ARRAY_NAME].append(ask_sizes)
            self._writers[EVENTS_ARRAY_NAME].append(merged_event_to_array_row(event))
            self.l2_count += 1
        elif event.ref.event_type == ExecutionEventType.TRADE:
            if event.trade is None:
                raise ValueError("trade merged event requires trade")
            if event.ref.trade_ptr != self.trade_count:
                raise ValueError("trade merged event trade_ptr must match next trade row")
            self._writers[TRADES_ARRAY_NAME].append(trade_print_to_array_row(event.trade))
            self._writers[EVENTS_ARRAY_NAME].append(merged_event_to_array_row(event))
            self.trade_count += 1
        else:
            raise ValueError("execution tape supports only L2_BATCH and TRADE events")
        self.event_count += 1

    def finalize(self, *, symbol_rule_compatibility: RuleCompatibilityReport | None = None) -> StreamingExecutionTapeWriteResult:
        if self._finalized:
            raise RuntimeError("writer already finalized")
        if symbol_rule_compatibility is None:
            symbol_rule_compatibility = self.config.symbol_rule_compatibility
        if self.event_count <= 0:
            raise ValueError("cannot build execution tape without events")
        if self.l2_count <= 0:
            raise ValueError("cannot build execution tape without reconstructed L2 events")
        if self.trade_count <= 0:
            raise ValueError("cannot build execution tape without trades")
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        row_counts: dict[str, int] = {}
        for name in _EXPECTED_ARRAY_NAMES:
            row_counts[name] = self._writers[name].finalize(self.arrays_dir / f"{name}.npy")
        notes = dict(self.config.notes or {})
        notes["book_depth"] = str(self.config.book_depth)
        start = self.first_local_ts_us
        last = self.last_local_ts_us
        if start is None or last is None:
            raise RuntimeError("missing tape time range")
        manifest = ExecutionTapeManifest(
            schema=EXECUTION_TAPE_SCHEMA,
            tape_format=ExecutionTapeFormat.L2_TRADES_ARRAYS,
            exchange=self.config.symbol_spec.exchange,
            symbol=self.config.symbol_spec.symbol,
            symbol_spec=self.config.symbol_spec,
            symbol_rules=self.config.symbol_rules,
            source_data_types=(TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES),
            array_names=_EXPECTED_ARRAY_NAMES,
            num_events=self.event_count,
            num_l2_batches=self.l2_count,
            num_trades=self.trade_count,
            num_decisions=0,
            start_local_ts_us=start,
            end_local_ts_us=max(last, start + 1),
            created_at_utc=self.config.created_at_utc,
            symbol_rule_compatibility=symbol_rule_compatibility,
            notes=notes,
        )
        manifest_text = json.dumps(execution_tape_manifest_to_dict(manifest), sort_keys=True, indent=2)
        tmp_path = self.output_root / f"{MANIFEST_FILENAME}.tmp"
        tmp_path.write_text(manifest_text + "\n", encoding="utf-8")
        tmp_path.replace(self.output_root / MANIFEST_FILENAME)
        tape = load_execution_tape(
            self.output_root,
            mmap_mode="r",
            validation_mode=self.config.validation_mode,
        )
        chunk_summary: dict[str, object] = {
            "chunk_dir": str(self.chunk_dir),
            "chunk_rows": self.config.chunk_rows,
            "row_counts": row_counts,
            "chunk_files": {name: len(writer._chunk_paths) for name, writer in self._writers.items()},
            "tape_validation_mode": self.config.validation_mode.value,
        }
        self._finalized = True
        if self.config.cleanup_chunks:
            shutil.rmtree(self.chunk_dir, ignore_errors=True)
            chunk_summary["chunks_cleaned"] = True
        else:
            chunk_summary["chunks_cleaned"] = False
        return StreamingExecutionTapeWriteResult(tape=tape, chunk_summary=chunk_summary)


__all__ = [
    "NpyChunkWriter",
    "StreamingExecutionTapeWriterConfig",
    "StreamingExecutionTapeWriteResult",
    "StreamingExecutionTapeWriter",
]
