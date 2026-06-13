"""CLI builder for execution replay tapes from Tardis-style L2/trade inputs.

This module intentionally orchestrates existing execution-layer primitives only:
it decodes input files into typed column chunks, adapts them into contracts,
reconstructs L2 batches, merges L2 and trade events, and delegates
array/manifest creation to ``execution_tape.py``.

CSV and Parquet inputs share one columnar decode path: pyarrow parses each
file into typed record batches and the per-field converters validate whole
columns with numpy before constructing contract objects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from mmrt.contracts import AggressorSide, BookSide
from mmrt.execution.contracts import L2Update, SymbolSpec, TradePrint
from mmrt.execution.event_merge import (
    ExecutionMergeCounterAccumulator,
    ExecutionMergeCounters,
    ExecutionMergeTiePolicy,
    iter_merged_execution_events,
)
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, _coerce_validation_mode
from mmrt.execution.execution_tape_writer import StreamingExecutionTapeWriter, StreamingExecutionTapeWriterConfig
from mmrt.execution.l2_reconstructor import (
    L2BookReconstructor,
    ReconstructedL2Event,
    iter_l2_update_batches,
)
from mmrt.metadata.binance_exchange_info import load_binance_usdm_exchange_info_symbol_rules
from mmrt.metadata.rule_compatibility import (
    RuleCompatibilityAccumulator,
    RuleCompatibilityConfig,
    RuleCompatibilityMode,
)
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode, read_symbol_rules_json


@dataclass(frozen=True, slots=True)
class ExecutionTapeBuildConfig:
    l2_inputs: tuple[str, ...]
    trade_inputs: tuple[str, ...]
    output_root: str

    exchange: str = "binance-futures"
    symbol: str = "BTCUSDT"
    exchange_info_json: str | None = None
    symbol_rules_json: str | None = None
    symbol_rules_mode: SymbolRuleMode | str = SymbolRuleMode.CURRENT_RULES_REPLAY
    symbol_rule_compatibility_mode: RuleCompatibilityMode | str = RuleCompatibilityMode.WARN
    price_grid_tolerance_ticks: float = 1e-6
    qty_grid_tolerance_steps: float = 1e-6

    batch_size: int = 65_536
    book_depth: int = 25
    max_l2_rows: int | None = None
    max_trade_rows: int | None = None
    tie_policy: ExecutionMergeTiePolicy | str = ExecutionMergeTiePolicy.L2_BEFORE_TRADE
    overwrite: bool = False
    chunk_rows: int = 250_000
    cleanup_chunks: bool = True
    created_at_utc: str = ""
    tape_validation_mode: ExecutionTapeValidationMode | str = ExecutionTapeValidationMode.SHAPE_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "l2_inputs", _tuple_of_nonempty_str(self.l2_inputs, "l2_inputs"))
        object.__setattr__(self, "trade_inputs", _tuple_of_nonempty_str(self.trade_inputs, "trade_inputs"))
        if not self.l2_inputs:
            raise ValueError("l2_inputs must be nonempty")
        if not self.trade_inputs:
            raise ValueError("trade_inputs must be nonempty")
        _require_nonempty_str(self.output_root, "output_root")
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")
        if (self.exchange_info_json is None) == (self.symbol_rules_json is None):
            raise ValueError("exactly one of exchange_info_json or symbol_rules_json is required")
        if self.exchange_info_json is not None:
            _require_nonempty_str(self.exchange_info_json, "exchange_info_json")
        if self.symbol_rules_json is not None:
            _require_nonempty_str(self.symbol_rules_json, "symbol_rules_json")
        object.__setattr__(self, "symbol_rules_mode", _coerce_symbol_rules_mode(self.symbol_rules_mode))
        object.__setattr__(self, "symbol_rule_compatibility_mode", _coerce_compatibility_mode(self.symbol_rule_compatibility_mode))
        object.__setattr__(self, "price_grid_tolerance_ticks", _require_nonnegative_float(self.price_grid_tolerance_ticks, "price_grid_tolerance_ticks"))
        object.__setattr__(self, "qty_grid_tolerance_steps", _require_nonnegative_float(self.qty_grid_tolerance_steps, "qty_grid_tolerance_steps"))
        object.__setattr__(self, "batch_size", _require_positive_int(self.batch_size, "batch_size"))
        object.__setattr__(self, "book_depth", _require_positive_int(self.book_depth, "book_depth"))
        object.__setattr__(self, "max_l2_rows", _optional_positive_int(self.max_l2_rows, "max_l2_rows"))
        object.__setattr__(self, "max_trade_rows", _optional_positive_int(self.max_trade_rows, "max_trade_rows"))
        object.__setattr__(self, "tie_policy", _coerce_tie_policy(self.tie_policy))
        if not isinstance(self.overwrite, bool):
            raise ValueError("overwrite must be bool")
        object.__setattr__(self, "chunk_rows", _require_positive_int(self.chunk_rows, "chunk_rows"))
        if not isinstance(self.cleanup_chunks, bool):
            raise ValueError("cleanup_chunks must be bool")
        if not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be str")
        object.__setattr__(self, "tape_validation_mode", _coerce_validation_mode(self.tape_validation_mode))


def build_execution_tape_from_config(config: ExecutionTapeBuildConfig) -> dict[str, object]:
    """Build and save an execution tape from validated file inputs."""
    if not isinstance(config, ExecutionTapeBuildConfig):
        raise ValueError("config must be ExecutionTapeBuildConfig")

    rules = _load_symbol_rules(config)
    symbol_spec = rules.to_symbol_spec()
    compat = RuleCompatibilityAccumulator(
        rules,
        RuleCompatibilityConfig(
            mode=config.symbol_rule_compatibility_mode,
            price_tolerance_ticks=config.price_grid_tolerance_ticks,
            qty_tolerance_steps=config.qty_grid_tolerance_steps,
        ),
    )
    l2_paths = _resolve_input_paths(config.l2_inputs, "l2_inputs")
    trade_paths = _resolve_input_paths(config.trade_inputs, "trade_inputs")
    output_root = Path(config.output_root)
    summary_path = output_root / "build_summary.json"
    if summary_path.exists() and not config.overwrite:
        raise FileExistsError(f"JSON output already exists: {summary_path}")

    l2_stats = L2StreamStatsAccumulator(book_depth=config.book_depth)
    trade_stats = TradeStreamStatsAccumulator()
    merge_counter = ExecutionMergeCounterAccumulator()

    l2_iter = iter_reconstructed_l2_events_streaming(
        l2_paths,
        symbol_spec=symbol_spec,
        batch_size=config.batch_size,
        max_rows=config.max_l2_rows,
        book_depth=config.book_depth,
        compatibility=compat,
        stats=l2_stats,
    )
    trade_iter = iter_trade_prints_streaming(
        trade_paths,
        symbol_spec=symbol_spec,
        batch_size=config.batch_size,
        max_rows=config.max_trade_rows,
        compatibility=compat,
        stats=trade_stats,
    )
    merged_iter = iter_merged_execution_events(
        l2_iter,
        trade_iter,
        tie_policy=config.tie_policy,
        counter=merge_counter,
    )
    writer = StreamingExecutionTapeWriter(
        StreamingExecutionTapeWriterConfig(
            output_root=str(output_root),
            symbol_spec=symbol_spec,
            symbol_rules=rules,
            book_depth=config.book_depth,
            chunk_rows=config.chunk_rows,
            overwrite=config.overwrite,
            cleanup_chunks=config.cleanup_chunks,
            created_at_utc=config.created_at_utc,
            validation_mode=config.tape_validation_mode,
            notes={
                "builder": "mmrt.cli.build_execution_tape",
                "tie_policy": config.tie_policy.value,
                "chunk_rows": str(config.chunk_rows),
                "streaming": "true",
                "tape_validation_mode": config.tape_validation_mode.value,
            },
        )
    )
    for merged in merged_iter:
        writer.append(merged)

    compatibility_report = compat.report() if config.symbol_rule_compatibility_mode is not RuleCompatibilityMode.OFF else None
    result = writer.finalize(symbol_rule_compatibility=compatibility_report)
    tape = result.tape
    l2_stats_dict = l2_stats.as_dict()
    trade_stats_dict = trade_stats.as_dict()
    merge_counters = merge_counter.as_counters()
    if merge_counter.emitted_event_count != tape.manifest.num_events:
        raise RuntimeError("merge event count does not match tape manifest")
    if merge_counter.l2_event_count != tape.manifest.num_l2_batches:
        raise RuntimeError("merge L2 count does not match tape manifest")
    if merge_counter.trade_count != tape.manifest.num_trades:
        raise RuntimeError("merge trade count does not match tape manifest")
    summary = _build_summary(
        config,
        output_root,
        l2_stats_dict,
        trade_stats_dict,
        merge_counters,
        tape,
        chunk_summary=result.chunk_summary,
    )
    _write_json_atomic(summary, summary_path, overwrite=config.overwrite)
    return summary

@dataclass(slots=True)
class L2StreamStatsAccumulator:
    rows_seen: int = 0
    first_local_ts_us: int | None = None
    last_local_ts_us: int | None = None
    scan_limit_hit: bool = False
    updates_converted: int = 0
    status: str = "not_ready"
    is_ready: bool = False
    batches_seen: int = 0
    updates_seen: int = 0
    skipped_pre_snapshot_updates: int = 0
    snapshot_reset_count: int = 0
    applied_update_count: int = 0
    deleted_level_count: int = 0
    missing_delete_count: int = 0
    emitted_event_count: int = 0
    crossed_batch_count: int = 0
    crossed_repair_count: int = 0
    crossed_levels_removed: int = 0
    local_ts_decrease_count: int = 0
    max_bid_depth: int = 0
    max_ask_depth: int = 0
    max_batch_size: int = 0
    book_depth: int = 25

    def update_from_reconstructor(self, reconstructor: L2BookReconstructor) -> None:
        counters = reconstructor.counters
        self.status = reconstructor.status.value
        self.is_ready = reconstructor.is_ready
        self.batches_seen = counters.batches_seen
        self.updates_seen = counters.updates_seen
        self.skipped_pre_snapshot_updates = counters.skipped_pre_snapshot_updates
        self.snapshot_reset_count = counters.snapshot_reset_count
        self.applied_update_count = counters.applied_update_count
        self.deleted_level_count = counters.deleted_level_count
        self.missing_delete_count = counters.missing_delete_count
        self.emitted_event_count = counters.emitted_event_count
        self.crossed_batch_count = counters.crossed_batch_count
        self.crossed_repair_count = counters.crossed_repair_count
        self.crossed_levels_removed = counters.crossed_levels_removed
        self.local_ts_decrease_count = counters.local_ts_decrease_count
        self.max_bid_depth = counters.max_bid_depth
        self.max_ask_depth = counters.max_ask_depth
        self.max_batch_size = counters.max_batch_size

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_seen": self.rows_seen,
            "updates_converted": self.updates_converted,
            "scan_limit_hit": self.scan_limit_hit,
            "first_local_ts_us": self.first_local_ts_us,
            "last_local_ts_us": self.last_local_ts_us,
            "status": self.status,
            "is_ready": self.is_ready,
            "batches_seen": self.batches_seen,
            "updates_seen": self.updates_seen,
            "skipped_pre_snapshot_updates": self.skipped_pre_snapshot_updates,
            "snapshot_reset_count": self.snapshot_reset_count,
            "applied_update_count": self.applied_update_count,
            "deleted_level_count": self.deleted_level_count,
            "missing_delete_count": self.missing_delete_count,
            "emitted_event_count": self.emitted_event_count,
            "crossed_batch_count": self.crossed_batch_count,
            "crossed_repair_count": self.crossed_repair_count,
            "crossed_levels_removed": self.crossed_levels_removed,
            "local_ts_decrease_count": self.local_ts_decrease_count,
            "max_bid_depth": self.max_bid_depth,
            "max_ask_depth": self.max_ask_depth,
            "max_batch_size": self.max_batch_size,
            "book_depth": self.book_depth,
        }


def iter_reconstructed_l2_events_streaming(
    paths: Sequence[Path],
    *,
    symbol_spec: SymbolSpec,
    batch_size: int,
    max_rows: int | None,
    book_depth: int,
    compatibility: RuleCompatibilityAccumulator | None,
    stats: L2StreamStatsAccumulator,
) -> Iterator[ReconstructedL2Event]:
    book_depth = _require_positive_int(book_depth, "book_depth")
    stats.book_depth = book_depth
    reconstructor = L2BookReconstructor(symbol_spec, snapshot_depth=book_depth)

    def iter_updates() -> Iterator[L2Update]:
        for path in paths:
            for chunk in _iter_path_column_chunks(path, batch_size=batch_size, columns=_L2_COLUMNS):
                rows = chunk.num_rows
                if rows == 0:
                    continue
                if max_rows is not None:
                    remaining = max_rows - stats.rows_seen
                    if remaining <= 0:
                        stats.scan_limit_hit = True
                        return
                    if rows > remaining:
                        chunk = chunk.slice(0, remaining)
                        rows = remaining
                        stats.scan_limit_hit = True
                first_fallback_source_row = stats.rows_seen
                stats.rows_seen += rows
                updates = _l2_updates_from_chunk(
                    chunk,
                    symbol_spec=symbol_spec,
                    compatibility=compatibility,
                    first_fallback_source_row=first_fallback_source_row,
                )
                stats.updates_converted += rows
                if stats.first_local_ts_us is None:
                    stats.first_local_ts_us = updates[0].local_ts_us
                stats.last_local_ts_us = updates[-1].local_ts_us
                yield from updates

    try:
        for batch in iter_l2_update_batches(iter_updates()):
            event = reconstructor.apply_batch(batch)
            if event is not None:
                yield event
    finally:
        stats.update_from_reconstructor(reconstructor)


@dataclass(slots=True)
class TradeStreamStatsAccumulator:
    rows_seen: int = 0
    trades_converted: int = 0
    scan_limit_hit: bool = False
    first_local_ts_us: int | None = None
    last_local_ts_us: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_seen": self.rows_seen,
            "trades_converted": self.trades_converted,
            "scan_limit_hit": self.scan_limit_hit,
            "first_local_ts_us": self.first_local_ts_us,
            "last_local_ts_us": self.last_local_ts_us,
        }


def iter_trade_prints_streaming(
    paths: Sequence[Path],
    *,
    symbol_spec: SymbolSpec,
    batch_size: int,
    max_rows: int | None,
    compatibility: RuleCompatibilityAccumulator | None,
    stats: TradeStreamStatsAccumulator,
) -> Iterator[TradePrint]:
    prev_local_ts_us: int | None = None
    for path in paths:
        for chunk in _iter_path_column_chunks(path, batch_size=batch_size, columns=_TRADE_COLUMNS):
            rows = chunk.num_rows
            if rows == 0:
                continue
            if max_rows is not None:
                remaining = max_rows - stats.rows_seen
                if remaining <= 0:
                    stats.scan_limit_hit = True
                    return
                if rows > remaining:
                    chunk = chunk.slice(0, remaining)
                    rows = remaining
                    stats.scan_limit_hit = True
            first_fallback_source_row = stats.rows_seen
            stats.rows_seen += rows
            trades, local_ts = _trade_prints_from_chunk(
                chunk,
                symbol_spec=symbol_spec,
                compatibility=compatibility,
                first_fallback_source_row=first_fallback_source_row,
            )
            unsorted_within = rows > 1 and bool((np.diff(local_ts) < 0).any())
            if unsorted_within or (prev_local_ts_us is not None and int(local_ts[0]) < prev_local_ts_us):
                raise ValueError("trades must be sorted by nondecreasing local_ts_us")
            if stats.first_local_ts_us is None:
                stats.first_local_ts_us = int(local_ts[0])
            stats.last_local_ts_us = int(local_ts[-1])
            prev_local_ts_us = stats.last_local_ts_us
            stats.trades_converted += rows
            yield from trades


def load_reconstructed_l2_events(
    paths: Sequence[Path],
    *,
    symbol_spec: SymbolSpec,
    batch_size: int,
    max_rows: int | None = None,
    book_depth: int = 25,
    compatibility: RuleCompatibilityAccumulator | None = None,
) -> tuple[list[ReconstructedL2Event], dict[str, object]]:
    stats = L2StreamStatsAccumulator(book_depth=book_depth)
    events = list(iter_reconstructed_l2_events_streaming(
        paths,
        symbol_spec=symbol_spec,
        batch_size=batch_size,
        max_rows=max_rows,
        book_depth=book_depth,
        compatibility=compatibility,
        stats=stats,
    ))
    return events, stats.as_dict()


def load_trade_prints(
    paths: Sequence[Path],
    *,
    symbol_spec: SymbolSpec,
    batch_size: int,
    max_rows: int | None = None,
    compatibility: RuleCompatibilityAccumulator | None = None,
) -> tuple[list[TradePrint], dict[str, object]]:
    stats = TradeStreamStatsAccumulator()
    trades = list(iter_trade_prints_streaming(
        paths,
        symbol_spec=symbol_spec,
        batch_size=batch_size,
        max_rows=max_rows,
        compatibility=compatibility,
        stats=stats,
    ))
    return trades, stats.as_dict()

class _ExecutionTapeArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if parsed.l2_input and isinstance(parsed.l2_input[0], list):
            parsed.l2_input = list(_flatten_repeated(parsed.l2_input))
        if parsed.trade_input and isinstance(parsed.trade_input[0], list):
            parsed.trade_input = list(_flatten_repeated(parsed.trade_input))
        return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _ExecutionTapeArgumentParser(description="Build an execution tape from Tardis-style L2 and trade inputs.")
    parser.add_argument("--l2-input", nargs="+", action="append", required=True, dest="l2_input")
    parser.add_argument("--trade-input", nargs="+", action="append", required=True, dest="trade_input")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exchange-info-json")
    group.add_argument("--symbol-rules-json")
    parser.add_argument("--symbol-rules-mode", choices=[mode.value for mode in SymbolRuleMode], default=SymbolRuleMode.CURRENT_RULES_REPLAY.value)
    parser.add_argument("--symbol-rule-compatibility-mode", choices=[mode.value for mode in RuleCompatibilityMode], default=RuleCompatibilityMode.WARN.value)
    parser.add_argument("--price-grid-tolerance-ticks", type=float, default=1e-6)
    parser.add_argument("--qty-grid-tolerance-steps", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--book-depth", type=int, default=25)
    parser.add_argument("--max-l2-rows", type=int)
    parser.add_argument("--max-trade-rows", type=int)
    parser.add_argument(
        "--tie-policy",
        choices=[policy.value for policy in ExecutionMergeTiePolicy],
        default=ExecutionMergeTiePolicy.L2_BEFORE_TRADE.value,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--created-at-utc", default="")
    parser.add_argument(
        "--tape-validation-mode",
        choices=[mode.value for mode in ExecutionTapeValidationMode],
        default=ExecutionTapeValidationMode.SHAPE_ONLY.value,
    )
    return parser



def _flatten_repeated(values: Sequence[Sequence[str]] | Sequence[str]) -> tuple[str, ...]:
    if not values:
        return ()
    first = values[0]
    if isinstance(first, str):
        return tuple(values)  # type: ignore[arg-type]
    return tuple(item for group in values for item in group)  # type: ignore[union-attr]

def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExecutionTapeBuildConfig(
        l2_inputs=_flatten_repeated(args.l2_input),
        trade_inputs=_flatten_repeated(args.trade_input),
        output_root=args.output_root,
        exchange=args.exchange,
        symbol=args.symbol,
        exchange_info_json=args.exchange_info_json,
        symbol_rules_json=args.symbol_rules_json,
        symbol_rules_mode=args.symbol_rules_mode,
        symbol_rule_compatibility_mode=args.symbol_rule_compatibility_mode,
        price_grid_tolerance_ticks=args.price_grid_tolerance_ticks,
        qty_grid_tolerance_steps=args.qty_grid_tolerance_steps,
        batch_size=args.batch_size,
        book_depth=args.book_depth,
        max_l2_rows=args.max_l2_rows,
        max_trade_rows=args.max_trade_rows,
        tie_policy=args.tie_policy,
        overwrite=args.overwrite,
        chunk_rows=args.chunk_rows,
        cleanup_chunks=not args.keep_chunks,
        created_at_utc=args.created_at_utc,
        tape_validation_mode=args.tape_validation_mode,
    )
    summary = build_execution_tape_from_config(config)
    print(json.dumps(_json_safe_summary(summary), sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


# Decompressed bytes handed to the CSV parser per chunk; large enough to keep
# the per-chunk numpy validation passes amortized, small enough for 16GB hosts.
_CSV_BLOCK_BYTES = 16 << 20

# Smallest float strictly above every int64-representable tick/timestamp.
_INT64_BOUND_FLOAT = float(2**63)

# CSV cells are parsed straight into the column types the converters expect.
# Integer-semantic columns use int64 so microsecond timestamps and row ids
# stay exact at any magnitude; enum-like and id columns stay strings so the
# original cell text reaches the existing per-value parsers unchanged.
_CSV_COLUMN_TYPES: dict[str, pa.DataType] = {
    "local_ts_us": pa.int64(),
    "local_timestamp": pa.int64(),
    "ts_us": pa.int64(),
    "timestamp": pa.int64(),
    "price_tick": pa.int64(),
    "price": pa.float64(),
    "amount": pa.float64(),
    "side": pa.string(),
    "is_snapshot": pa.string(),
    "side_code": pa.int64(),
    "trade_id": pa.string(),
    "raw_source_row": pa.int64(),
    "source_row": pa.int64(),
}

_L2_COLUMNS = (
    "local_ts_us",
    "local_timestamp",
    "ts_us",
    "timestamp",
    "price_tick",
    "price",
    "side",
    "amount",
    "is_snapshot",
    "raw_source_row",
    "source_row",
)

_TRADE_COLUMNS = (
    "local_ts_us",
    "local_timestamp",
    "ts_us",
    "timestamp",
    "price_tick",
    "price",
    "side",
    "side_code",
    "amount",
    "trade_id",
    "raw_source_row",
    "source_row",
)


def _iter_path_column_chunks(path: Path, *, batch_size: int, columns: tuple[str, ...]) -> Iterator[pa.RecordBatch]:
    if path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
        stream = pa.input_stream(str(path), compression="detect")
        try:
            reader = pacsv.open_csv(
                stream,
                read_options=pacsv.ReadOptions(block_size=_CSV_BLOCK_BYTES),
                convert_options=pacsv.ConvertOptions(
                    column_types=_CSV_COLUMN_TYPES,
                    include_columns=list(columns),
                    include_missing_columns=True,
                ),
            )
            try:
                for chunk in reader:
                    yield chunk
            finally:
                reader.close()
        finally:
            stream.close()
        return
    if path.suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        wanted = [name for name in columns if name in names] or None
        yield from parquet_file.iter_batches(batch_size=batch_size, columns=wanted)
        return
    raise ValueError(f"unsupported input file type: {path}")


def _chunk_column(chunk: pa.RecordBatch, name: str) -> pa.Array | None:
    index = chunk.schema.get_field_index(name)
    if index < 0:
        return None
    column = chunk.column(index)
    if pa.types.is_null(column.type):
        return None
    return column


def _numeric_column_values(column: pa.Array, *, invalid_type_message: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode a numeric column to (int64-or-float64 values, present mask or None).

    A ``None`` mask means every row is present. Slots that are not present
    hold filler values; callers must resolve or reject those rows before
    validating values.
    """
    kind = column.type
    if pa.types.is_boolean(kind):
        raise ValueError(invalid_type_message)
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        column = pc.if_else(pc.equal(column, ""), pa.scalar(None, kind), column).cast(pa.float64())
        kind = column.type
    if pa.types.is_integer(kind):
        if column.null_count:
            present = ~column.is_null().to_numpy(zero_copy_only=False)
            values = column.fill_null(0).to_numpy(zero_copy_only=False)
        else:
            present = None
            values = column.to_numpy(zero_copy_only=False)
        return values.astype(np.int64, copy=False), present
    if pa.types.is_floating(kind):
        if column.null_count:
            present = ~column.is_null().to_numpy(zero_copy_only=False)
        else:
            present = None
        return column.to_numpy(zero_copy_only=False).astype(np.float64, copy=False), present
    raise ValueError(invalid_type_message)


def _optional_numeric_column(chunk: pa.RecordBatch, name: str, *, invalid_type_message: str) -> tuple[np.ndarray, np.ndarray | None] | None:
    column = _chunk_column(chunk, name)
    if column is None or column.null_count == len(column):
        return None
    return _numeric_column_values(column, invalid_type_message=invalid_type_message)


def _require_positive_int_column(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype.kind == "f":
        ok = np.isfinite(values) & (values > 0.0) & (values == np.floor(values)) & (values < _INT64_BOUND_FLOAT)
        if not bool(ok.all()):
            raise ValueError(f"{name} must be a positive int")
        return values.astype(np.int64)
    if bool((values <= 0).any()):
        raise ValueError(f"{name} must be a positive int")
    return values


def _require_int_column(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype.kind == "f":
        ok = np.isfinite(values) & (values == np.floor(values)) & (np.abs(values) < _INT64_BOUND_FLOAT)
        if not bool(ok.all()):
            raise ValueError(f"{name} must be int")
        return values.astype(np.int64)
    return values


def _require_finite_floats(values: np.ndarray, name: str) -> None:
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must be a finite float")


def _resolve_required_int_alias(chunk: pa.RecordBatch, names: tuple[str, ...], value_name: str) -> np.ndarray:
    """Resolve per-row alias fallthrough across columns, first present wins."""
    invalid_type_message = f"{value_name} must be a positive int"
    resolved: np.ndarray | None = None
    have: np.ndarray | None = None
    for name in names:
        column = _chunk_column(chunk, name)
        if column is None or column.null_count == len(column):
            continue
        values, present = _numeric_column_values(column, invalid_type_message=invalid_type_message)
        if resolved is None:
            resolved = values
            have = present
        else:
            fill = ~have if present is None else (~have & present)
            if fill.any():
                if resolved.dtype != values.dtype:
                    resolved = resolved.astype(np.float64)
                    values = values.astype(np.float64)
                resolved = np.where(fill, values, resolved)
                have = have | fill
        if have is None or bool(have.all()):
            have = None
            break
    if resolved is None or have is not None:
        raise ValueError(f"row is missing required field(s): {', '.join(names)}")
    return _require_positive_int_column(resolved, value_name)


def _required_float_column(chunk: pa.RecordBatch, name: str) -> np.ndarray:
    pair = _optional_numeric_column(chunk, name, invalid_type_message=f"{name} must be a finite float")
    if pair is None:
        raise ValueError(f"row is missing required field(s): {name}")
    values, present = pair
    if present is not None and not bool(present.all()):
        raise ValueError(f"row is missing required field(s): {name}")
    return values.astype(np.float64, copy=False)


def _ticks_from_prices(prices: np.ndarray, symbol_spec: SymbolSpec) -> np.ndarray:
    _require_finite_floats(prices, "price")
    if bool((prices <= 0.0).any()):
        raise ValueError("price must be > 0")
    ticks = np.rint(prices / symbol_spec.tick_size)
    if bool(((ticks < 1.0) | (ticks >= _INT64_BOUND_FLOAT)).any()):
        raise ValueError("price_tick must be a positive int")
    return ticks.astype(np.int64)


def _resolve_price_ticks_and_observe(
    chunk: pa.RecordBatch,
    n: int,
    *,
    symbol_spec: SymbolSpec,
    compatibility: RuleCompatibilityAccumulator | None,
    price_source: str,
    local_ts: np.ndarray,
) -> np.ndarray:
    tick_pair = _optional_numeric_column(chunk, "price_tick", invalid_type_message="price_tick must be a positive int")
    price_pair = _optional_numeric_column(chunk, "price", invalid_type_message="price must be a finite float")

    price_values: np.ndarray | None = None
    price_present: np.ndarray | None = None
    if price_pair is not None:
        price_values = price_pair[0].astype(np.float64, copy=False)
        price_present = price_pair[1]
        if compatibility is not None:
            if price_present is None:
                _require_finite_floats(price_values, "price")
                compatibility.observe_price_array(price_values, source=price_source, local_ts_us=local_ts)
            elif bool(price_present.any()):
                observed = price_values[price_present]
                _require_finite_floats(observed, "price")
                compatibility.observe_price_array(observed, source=price_source, local_ts_us=local_ts[price_present])

    if tick_pair is not None:
        tick_values, tick_present = tick_pair
        if tick_present is None:
            return _require_positive_int_column(tick_values, "price_tick")
        need = ~tick_present
        if price_values is None or (price_present is not None and not bool(price_present[need].all())):
            raise ValueError("row is missing required field(s): price")
        ticks = np.empty(n, dtype=np.int64)
        ticks[tick_present] = _require_positive_int_column(tick_values[tick_present], "price_tick")
        ticks[need] = _ticks_from_prices(price_values[need], symbol_spec)
        return ticks
    if price_values is None or (price_present is not None and not bool(price_present.all())):
        raise ValueError("row is missing required field(s): price")
    return _ticks_from_prices(price_values, symbol_spec)


def _resolve_input_paths(paths: Sequence[str], name: str) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for i, raw_path in enumerate(paths):
        _require_nonempty_str(raw_path, f"{name}[{i}]")
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{name}[{i}] does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{name}[{i}] is not a file: {path}")
        if not _is_supported_input_path(path):
            raise ValueError(f"{name}[{i}] has unsupported suffix: {path}")
        resolved.append(path)
    return tuple(resolved)


def _l2_sides(chunk: pa.RecordBatch) -> list[BookSide]:
    column = _chunk_column(chunk, "side")
    if column is None or column.null_count:
        raise ValueError("row is missing required field(s): side")
    encoded = column.dictionary_encode()
    side_map: list[BookSide] = []
    for value in encoded.dictionary.to_pylist():
        if value == "":
            raise ValueError("row is missing required field(s): side")
        side_map.append(_parse_book_side(value))
    return [side_map[i] for i in encoded.indices.to_numpy(zero_copy_only=False).tolist()]


def _l2_snapshot_flags(chunk: pa.RecordBatch) -> list[bool]:
    column = _chunk_column(chunk, "is_snapshot")
    if column is None or column.null_count:
        raise ValueError("row is missing required field(s): is_snapshot")
    if pa.types.is_boolean(column.type):
        return column.to_pylist()
    encoded = column.dictionary_encode()
    flag_map: list[bool] = []
    for value in encoded.dictionary.to_pylist():
        if value == "":
            raise ValueError("row is missing required field(s): is_snapshot")
        flag_map.append(_parse_bool(value))
    return [flag_map[i] for i in encoded.indices.to_numpy(zero_copy_only=False).tolist()]


def _trade_sides_from_codes(codes: np.ndarray) -> list[AggressorSide]:
    buy = codes == 1
    sell = codes == -1
    unknown = codes == 0
    bad = ~(buy | sell | unknown)
    if bool(bad.any()):
        _parse_trade_side_code(int(codes[int(np.argmax(bad))]))
    out = np.empty(codes.shape[0], dtype=object)
    out[buy] = AggressorSide.BUY
    out[sell] = AggressorSide.SELL
    out[unknown] = AggressorSide.UNKNOWN
    return out.tolist()


def _trade_sides(chunk: pa.RecordBatch, n: int) -> list[AggressorSide]:
    side_column = _chunk_column(chunk, "side")
    sides: list[AggressorSide | None] | None = None
    if side_column is not None and side_column.null_count != len(side_column):
        encoded = side_column.dictionary_encode()
        side_map: list[AggressorSide | None] = []
        for value in encoded.dictionary.to_pylist():
            side_map.append(None if value == "" else _parse_trade_side(value))
        indices = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False)
        sides = [None if i < 0 else side_map[i] for i in indices.tolist()]
        need = np.fromiter((side is None for side in sides), dtype=bool, count=n)
        if not bool(need.any()):
            return sides
    else:
        need = np.ones(n, dtype=bool)

    code_pair = _optional_numeric_column(chunk, "side_code", invalid_type_message="side_code must be int")
    if code_pair is None:
        raise ValueError("trade row missing side or side_code")
    code_values, code_present = code_pair
    if code_present is not None and not bool(code_present[need].all()):
        raise ValueError("trade row missing side or side_code")
    code_sides = _trade_sides_from_codes(_require_int_column(code_values[need], "side_code"))
    if sides is None:
        return code_sides
    for index, side in zip(np.flatnonzero(need).tolist(), code_sides):
        sides[index] = side
    return sides


def _trade_ids(chunk: pa.RecordBatch, n: int) -> list[str]:
    column = _chunk_column(chunk, "trade_id")
    if column is None:
        return [""] * n
    return ["" if value is None else str(value) for value in column.to_pylist()]


def _validated_source_rows(values: np.ndarray) -> np.ndarray:
    out = _require_int_column(values, "source_row")
    if bool((out < 0).any()):
        raise ValueError("source_row must be >= 0")
    return out


def _source_rows(chunk: pa.RecordBatch, n: int, first_fallback_source_row: int) -> np.ndarray:
    raw_pair = _optional_numeric_column(chunk, "raw_source_row", invalid_type_message="source_row must be int")
    if raw_pair is not None and raw_pair[1] is None:
        return _validated_source_rows(raw_pair[0])
    src_pair = _optional_numeric_column(chunk, "source_row", invalid_type_message="source_row must be int")
    if raw_pair is None:
        if src_pair is None:
            return np.arange(first_fallback_source_row, first_fallback_source_row + n, dtype=np.int64)
        if src_pair[1] is None:
            return _validated_source_rows(src_pair[0])
    resolved = np.arange(first_fallback_source_row, first_fallback_source_row + n, dtype=np.int64)
    filled = np.zeros(n, dtype=bool)
    if raw_pair is not None:
        values, present = raw_pair
        resolved[present] = _validated_source_rows(values[present])
        filled |= present
    if src_pair is not None:
        values, present = src_pair
        take = ~filled if present is None else (~filled & present)
        if bool(take.any()):
            resolved[take] = _validated_source_rows(values[take])
    return resolved


def _l2_updates_from_chunk(
    chunk: pa.RecordBatch,
    *,
    symbol_spec: SymbolSpec,
    compatibility: RuleCompatibilityAccumulator | None,
    first_fallback_source_row: int,
) -> list[L2Update]:
    n = chunk.num_rows
    local_ts = _resolve_required_int_alias(chunk, ("local_ts_us", "local_timestamp"), "local_ts_us")
    ts = _resolve_required_int_alias(chunk, ("ts_us", "timestamp"), "ts_us")
    ticks = _resolve_price_ticks_and_observe(
        chunk, n, symbol_spec=symbol_spec, compatibility=compatibility, price_source="l2.price", local_ts=local_ts
    )
    sides = _l2_sides(chunk)
    amounts = _required_float_column(chunk, "amount")
    _require_finite_floats(amounts, "amount")
    if bool((amounts < 0.0).any()):
        raise ValueError("amount must be >= 0")
    if compatibility is not None:
        positive = amounts > 0.0
        if bool(positive.all()):
            compatibility.observe_qty_array(amounts, source="l2.amount", local_ts_us=local_ts)
        else:
            compatibility.observe_qty_array(amounts[positive], source="l2.amount", local_ts_us=local_ts[positive])
    snapshots = _l2_snapshot_flags(chunk)
    source_rows = _source_rows(chunk, n, first_fallback_source_row)
    # Every column is validated above; the trusted constructor skips the
    # redundant dataclass re-validation on this per-row hot path.
    from_trusted = L2Update.from_trusted
    return [
        from_trusted(
            local_ts_us=row_local_ts,
            ts_us=row_ts,
            side=side,
            price_tick=tick,
            amount=amount,
            is_snapshot=is_snapshot,
            source_row=source_row,
        )
        for row_local_ts, row_ts, side, tick, amount, is_snapshot, source_row in zip(
            local_ts.tolist(), ts.tolist(), sides, ticks.tolist(), amounts.tolist(), snapshots, source_rows.tolist()
        )
    ]


def _trade_prints_from_chunk(
    chunk: pa.RecordBatch,
    *,
    symbol_spec: SymbolSpec,
    compatibility: RuleCompatibilityAccumulator | None,
    first_fallback_source_row: int,
) -> tuple[list[TradePrint], np.ndarray]:
    n = chunk.num_rows
    local_ts = _resolve_required_int_alias(chunk, ("local_ts_us", "local_timestamp"), "local_ts_us")
    ts = _resolve_required_int_alias(chunk, ("ts_us", "timestamp"), "ts_us")
    ticks = _resolve_price_ticks_and_observe(
        chunk, n, symbol_spec=symbol_spec, compatibility=compatibility, price_source="trade.price", local_ts=local_ts
    )
    amounts = _required_float_column(chunk, "amount")
    _require_finite_floats(amounts, "amount")
    if bool((amounts <= 0.0).any()):
        raise ValueError("trade amount must be > 0")
    if compatibility is not None:
        compatibility.observe_qty_array(amounts, source="trade.amount", local_ts_us=local_ts)
    sides = _trade_sides(chunk, n)
    trade_ids = _trade_ids(chunk, n)
    source_rows = _source_rows(chunk, n, first_fallback_source_row)
    from_trusted = TradePrint.from_trusted
    trades = [
        from_trusted(
            local_ts_us=row_local_ts,
            ts_us=row_ts,
            side=side,
            price_tick=tick,
            amount=amount,
            trade_id=trade_id,
            source_row=source_row,
        )
        for row_local_ts, row_ts, side, tick, amount, trade_id, source_row in zip(
            local_ts.tolist(), ts.tolist(), sides, ticks.tolist(), amounts.tolist(), trade_ids, source_rows.tolist()
        )
    ]
    return trades, local_ts


def _build_summary(config, output_root, l2_stats, trade_stats, merge_counters: ExecutionMergeCounters, tape, *, chunk_summary: dict[str, object]) -> dict[str, object]:
    warnings: list[str] = []
    if not bool(l2_stats["is_ready"]):
        warnings.append("no_snapshot_seen")
    if int(l2_stats["emitted_event_count"]) == 0:
        warnings.append("no_reconstructed_l2_events")
    if int(trade_stats["trades_converted"]) == 0:
        warnings.append("no_trades")
    if bool(l2_stats["scan_limit_hit"]):
        warnings.append("l2_scan_limit_hit")
    if bool(trade_stats["scan_limit_hit"]):
        warnings.append("trade_scan_limit_hit")
    if int(l2_stats["crossed_repair_count"]) > 0:
        warnings.append("crossed_repairs_observed")
    if int(l2_stats["missing_delete_count"]) > 0:
        warnings.append("missing_deletes_observed")
    if merge_counters.same_local_ts_tie_count > 0:
        warnings.append("same_local_ts_ties_observed")

    return {
        "status": "ok",
        "build_type": "execution_tape",
        "output_root": str(output_root),
        "created_at_utc": config.created_at_utc,
        "inputs": {
            "l2_input_count": len(config.l2_inputs),
            "trade_input_count": len(config.trade_inputs),
            "l2_inputs": list(config.l2_inputs),
            "trade_inputs": list(config.trade_inputs),
            "max_l2_rows": config.max_l2_rows,
            "max_trade_rows": config.max_trade_rows,
            "batch_size": config.batch_size,
            "book_depth": config.book_depth,
        },
        "market": {
            "exchange": config.exchange,
            "symbol": config.symbol,
            "symbol_rules_mode": tape.manifest.symbol_rules.mode.value,
            "symbol_rules_source": tape.manifest.symbol_rules.source,
            "symbol_rules_source_sha256": tape.manifest.symbol_rules.source_sha256,
            "symbol_rules_captured_at_utc": tape.manifest.symbol_rules.captured_at_utc,
            "symbol_rule_compatibility_mode": config.symbol_rule_compatibility_mode.value,
        },
        "symbol_rules": tape.manifest.symbol_rules.to_dict(),
        "symbol_rule_compatibility": (
            None
            if tape.manifest.symbol_rule_compatibility is None
            else tape.manifest.symbol_rule_compatibility.to_dict()
        ),
        "counts": {
            "l2_rows_seen": l2_stats["rows_seen"],
            "l2_updates_converted": l2_stats["updates_converted"],
            "l2_batches_seen": l2_stats["batches_seen"],
            "l2_events_emitted": l2_stats["emitted_event_count"],
            "trade_rows_seen": trade_stats["rows_seen"],
            "trades_converted": trade_stats["trades_converted"],
            "merged_events": merge_counters.emitted_event_count,
            "tape_l2_events": tape.manifest.num_l2_batches,
            "tape_trades": tape.manifest.num_trades,
            "tape_events": tape.manifest.num_events,
        },
        "reconstruction": {
            "status": l2_stats["status"],
            "is_ready": l2_stats["is_ready"],
            "skipped_pre_snapshot_updates": l2_stats["skipped_pre_snapshot_updates"],
            "snapshot_reset_count": l2_stats["snapshot_reset_count"],
            "applied_update_count": l2_stats["applied_update_count"],
            "deleted_level_count": l2_stats["deleted_level_count"],
            "missing_delete_count": l2_stats["missing_delete_count"],
            "crossed_batch_count": l2_stats["crossed_batch_count"],
            "crossed_repair_count": l2_stats["crossed_repair_count"],
            "crossed_levels_removed": l2_stats["crossed_levels_removed"],
            "local_ts_decrease_count": l2_stats["local_ts_decrease_count"],
            "max_bid_depth": l2_stats["max_bid_depth"],
            "max_ask_depth": l2_stats["max_ask_depth"],
            "max_batch_size": l2_stats["max_batch_size"],
            "book_depth": l2_stats["book_depth"],
        },
        "merge": {
            "tie_policy": config.tie_policy.value,
            "l2_event_count": merge_counters.l2_event_count,
            "trade_count": merge_counters.trade_count,
            "emitted_event_count": merge_counters.emitted_event_count,
            "same_local_ts_tie_count": merge_counters.same_local_ts_tie_count,
        },
        "tape": {
            "schema": tape.manifest.schema,
            "tape_format": tape.manifest.tape_format.value,
            "array_names": list(tape.manifest.array_names),
            "num_events": tape.manifest.num_events,
            "num_l2_batches": tape.manifest.num_l2_batches,
            "num_trades": tape.manifest.num_trades,
            "num_decisions": tape.manifest.num_decisions,
            "book_depth": config.book_depth,
            "start_local_ts_us": tape.manifest.start_local_ts_us,
            "end_local_ts_us": tape.manifest.end_local_ts_us,
        },
        "chunking": {
            "enabled": True,
            "chunk_rows": config.chunk_rows,
            "cleanup_chunks": config.cleanup_chunks,
            "chunk_summary": chunk_summary,
            "tape_validation_mode": config.tape_validation_mode.value,
        },
        "warnings": warnings,
    }


def _write_json_atomic(payload: dict[str, object], path: str | Path, *, overwrite: bool = False) -> str:
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("path suffix must be .json")
    if path.exists() and not overwrite:
        raise FileExistsError(f"JSON output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    safe_payload = _json_safe_summary(payload)
    text = json.dumps(safe_payload, sort_keys=True, indent=2, allow_nan=False)
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return str(path)


def _json_safe_summary(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_summary(v) for v in value]
    return value


def _tuple_of_nonempty_str(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of non-empty strings, not a single string/bytes value")
    try:
        seq = values if isinstance(values, tuple) else tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of non-empty strings") from exc
    return tuple(_require_nonempty_str(v, f"{name}[{i}]") for i, v in enumerate(seq))


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _require_positive_float(value: float, name: str) -> float:
    value = _coerce_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _coerce_float(value, name)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _coerce_tie_policy(value: ExecutionMergeTiePolicy | str) -> ExecutionMergeTiePolicy:
    if isinstance(value, ExecutionMergeTiePolicy):
        return value
    if isinstance(value, str):
        try:
            return ExecutionMergeTiePolicy(value)
        except ValueError as exc:
            raise ValueError(f"invalid tie_policy: {value!r}") from exc
    raise ValueError("tie_policy must be ExecutionMergeTiePolicy or str")


def _is_supported_input_path(path: Path) -> bool:
    return path.name.endswith(".csv") or path.name.endswith(".csv.gz") or path.suffix == ".parquet"


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite float")
    return out


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "t", "1", "yes", "y"):
            return True
        if normalized in ("false", "f", "0", "no", "n", ""):
            return False
    raise ValueError(f"cannot parse bool value: {value!r}")


def _parse_book_side(value: Any) -> BookSide:
    if isinstance(value, BookSide):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "bid":
            return BookSide.BID
        if normalized == "ask":
            return BookSide.ASK
    raise ValueError(f"unsupported L2 side: {value!r}")


def _parse_trade_side(value: Any) -> AggressorSide:
    if isinstance(value, AggressorSide):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "buy":
            return AggressorSide.BUY
        if normalized == "sell":
            return AggressorSide.SELL
        if normalized == "unknown":
            return AggressorSide.UNKNOWN
    raise ValueError(f"unsupported trade side: {value!r}")


def _parse_trade_side_code(code: int) -> AggressorSide:
    if code == 1:
        return AggressorSide.BUY
    if code == -1:
        return AggressorSide.SELL
    if code == 0:
        return AggressorSide.UNKNOWN
    raise ValueError(f"unsupported side_code: {code!r}")


def _coerce_symbol_rules_mode(value: SymbolRuleMode | str) -> SymbolRuleMode:
    if isinstance(value, SymbolRuleMode):
        return value
    if isinstance(value, str):
        return SymbolRuleMode(value)
    raise ValueError("symbol_rules_mode must be SymbolRuleMode or str")


def _coerce_compatibility_mode(value: RuleCompatibilityMode | str) -> RuleCompatibilityMode:
    if isinstance(value, RuleCompatibilityMode):
        return value
    if isinstance(value, str):
        return RuleCompatibilityMode(value)
    raise ValueError("symbol_rule_compatibility_mode must be RuleCompatibilityMode or str")


def _load_symbol_rules(config: ExecutionTapeBuildConfig) -> ExchangeSymbolRules:
    if config.exchange_info_json is not None:
        rules = load_binance_usdm_exchange_info_symbol_rules(
            config.exchange_info_json,
            symbol=config.symbol,
            exchange=config.exchange,
            mode=config.symbol_rules_mode,
        )
    elif config.symbol_rules_json is not None:
        rules = read_symbol_rules_json(config.symbol_rules_json)
    else:  # guarded by config validation
        raise ValueError("exactly one of exchange_info_json or symbol_rules_json is required")
    if rules.exchange != config.exchange or rules.symbol != config.symbol:
        raise ValueError("symbol rules exchange/symbol must match build config")
    return rules


__all__ = [
    "ExecutionTapeBuildConfig",
    "build_execution_tape_from_config",
    "build_arg_parser",
    "_flatten_repeated",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
