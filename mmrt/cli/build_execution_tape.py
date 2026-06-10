"""CLI builder for execution replay tapes from Tardis-style L2/trade inputs.

This module intentionally orchestrates existing execution-layer primitives only:
it reads rows, adapts them into contracts, reconstructs L2 batches, merges L2 and
trade events, and delegates array/manifest creation to ``execution_tape.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:  # Optional dependency for Parquet input support.
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - exercised only without pyarrow
    pq = None

from mmrt.contracts import AggressorSide, BookSide
from mmrt.execution.contracts import L2Update, SymbolSpec, TradePrint
from mmrt.execution.event_merge import (
    ExecutionMergeCounterAccumulator,
    ExecutionMergeCounters,
    ExecutionMergeTiePolicy,
    iter_merged_execution_events,
)
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
            notes={
                "builder": "mmrt.cli.build_execution_tape",
                "tie_policy": config.tie_policy.value,
                "chunk_rows": str(config.chunk_rows),
                "streaming": "true",
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
            for row in _iter_path_rows(path, batch_size=batch_size):
                if max_rows is not None and stats.rows_seen >= max_rows:
                    stats.scan_limit_hit = True
                    return
                fallback_source_row = stats.rows_seen
                stats.rows_seen += 1
                if compatibility is not None:
                    _observe_l2_row_compatibility(row, compatibility)
                update = _row_to_l2_update(row, symbol_spec=symbol_spec, fallback_source_row=fallback_source_row)
                stats.updates_converted += 1
                if stats.first_local_ts_us is None:
                    stats.first_local_ts_us = update.local_ts_us
                stats.last_local_ts_us = update.local_ts_us
                yield update

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
        for row in _iter_path_rows(path, batch_size=batch_size):
            if max_rows is not None and stats.rows_seen >= max_rows:
                stats.scan_limit_hit = True
                return
            fallback_source_row = stats.rows_seen
            stats.rows_seen += 1
            if compatibility is not None:
                _observe_trade_row_compatibility(row, compatibility)
            trade = _row_to_trade_print(row, symbol_spec=symbol_spec, fallback_source_row=fallback_source_row)
            if prev_local_ts_us is not None and trade.local_ts_us < prev_local_ts_us:
                raise ValueError("trades must be sorted by nondecreasing local_ts_us")
            if stats.first_local_ts_us is None:
                stats.first_local_ts_us = trade.local_ts_us
            stats.last_local_ts_us = trade.local_ts_us
            prev_local_ts_us = trade.local_ts_us
            stats.trades_converted += 1
            yield trade


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
    )
    summary = build_execution_tape_from_config(config)
    print(json.dumps(_json_safe_summary(summary), sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def _iter_path_rows(path: Path, *, batch_size: int) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
        yield from _iter_csv_rows(path)
        return
    if path.suffix == ".parquet":
        yield from _iter_parquet_rows(path, batch_size=batch_size)
        return
    raise ValueError(f"unsupported input file type: {path}")


def _iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    open_fn = gzip.open if path.name.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _iter_parquet_rows(path: Path, *, batch_size: int) -> Iterator[dict[str, Any]]:
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required to read Parquet input files")
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {col: values[i] for col, values in data.items()}


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


def _row_to_l2_update(row: Mapping[str, Any], *, symbol_spec: SymbolSpec, fallback_source_row: int) -> L2Update:
    local_ts_us = _coerce_positive_int(_first_present(row, ("local_ts_us", "local_timestamp")), "local_ts_us")
    ts_us = _coerce_positive_int(_first_present(row, ("ts_us", "timestamp")), "ts_us")
    if _has_nonempty(row, "price_tick"):
        price_tick = _coerce_positive_int(row["price_tick"], "price_tick")
    else:
        price_tick = symbol_spec.price_to_tick(_coerce_float(_first_present(row, ("price",)), "price"))
    side = _parse_book_side(_first_present(row, ("side",)))
    amount = _coerce_float(_first_present(row, ("amount",)), "amount")
    if amount < 0:
        raise ValueError("amount must be >= 0")
    is_snapshot = _parse_bool(_first_present(row, ("is_snapshot",)))
    source_row = _source_row(row, fallback_source_row)
    return L2Update(
        local_ts_us=local_ts_us,
        ts_us=ts_us,
        side=side,
        price_tick=price_tick,
        amount=amount,
        is_snapshot=is_snapshot,
        source_row=source_row,
    )


def _row_to_trade_print(row: Mapping[str, Any], *, symbol_spec: SymbolSpec, fallback_source_row: int) -> TradePrint:
    local_ts_us = _coerce_positive_int(_first_present(row, ("local_ts_us", "local_timestamp")), "local_ts_us")
    ts_us = _coerce_positive_int(_first_present(row, ("ts_us", "timestamp")), "ts_us")
    if _has_nonempty(row, "price_tick"):
        price_tick = _coerce_positive_int(row["price_tick"], "price_tick")
    else:
        price_tick = symbol_spec.price_to_tick(_coerce_float(_first_present(row, ("price",)), "price"))
    amount = _coerce_float(_first_present(row, ("amount",)), "amount")
    if amount <= 0:
        raise ValueError("trade amount must be > 0")
    side = _parse_trade_side(row)
    trade_id = "" if not _has_nonempty(row, "trade_id") else str(row["trade_id"])
    source_row = _source_row(row, fallback_source_row)
    return TradePrint(
        local_ts_us=local_ts_us,
        ts_us=ts_us,
        side=side,
        price_tick=price_tick,
        amount=amount,
        trade_id=trade_id,
        source_row=source_row,
    )


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


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if _has_nonempty(row, name):
            return row[name]
    raise ValueError(f"row is missing required field(s): {', '.join(names)}")


def _has_nonempty(row: Mapping[str, Any], name: str) -> bool:
    if name not in row:
        return False
    value = row[name]
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


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


def _coerce_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive int")
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError
            out = int(value)
        elif isinstance(value, str):
            if "." in value:
                f = float(value)
                if not f.is_integer():
                    raise ValueError
                out = int(f)
            else:
                out = int(value)
        else:
            out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive int") from exc
    if out <= 0:
        raise ValueError(f"{name} must be a positive int")
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


def _parse_trade_side(row: Mapping[str, Any]) -> AggressorSide:
    if _has_nonempty(row, "side"):
        value = row["side"]
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

    if _has_nonempty(row, "side_code"):
        code = _coerce_int(row["side_code"], "side_code")
        if code == 1:
            return AggressorSide.BUY
        if code == -1:
            return AggressorSide.SELL
        if code == 0:
            return AggressorSide.UNKNOWN
        raise ValueError(f"unsupported side_code: {code!r}")
    raise ValueError("trade row missing side or side_code")


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


def _observe_l2_row_compatibility(row: Mapping[str, Any], compatibility: RuleCompatibilityAccumulator) -> None:
    local_ts_us = _compat_local_ts(row)
    if _has_nonempty(row, "price"):
        compatibility.observe_price(_coerce_float(row["price"], "price"), source="l2.price", local_ts_us=local_ts_us)
    if _has_nonempty(row, "amount"):
        amount = _coerce_float(row["amount"], "amount")
        if amount > 0:
            compatibility.observe_qty(amount, source="l2.amount", local_ts_us=local_ts_us)


def _observe_trade_row_compatibility(row: Mapping[str, Any], compatibility: RuleCompatibilityAccumulator) -> None:
    local_ts_us = _compat_local_ts(row)
    if _has_nonempty(row, "price"):
        compatibility.observe_price(_coerce_float(row["price"], "price"), source="trade.price", local_ts_us=local_ts_us)
    if _has_nonempty(row, "amount"):
        compatibility.observe_qty(_coerce_float(row["amount"], "amount"), source="trade.amount", local_ts_us=local_ts_us)


def _compat_local_ts(row: Mapping[str, Any]) -> int | None:
    if _has_nonempty(row, "local_ts_us"):
        return _coerce_positive_int(row["local_ts_us"], "local_ts_us")
    if _has_nonempty(row, "local_timestamp"):
        return _coerce_positive_int(row["local_timestamp"], "local_timestamp")
    return None

def _coerce_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int")
    try:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError
            return int(value)
        if isinstance(value, str):
            f = float(value)
            if not f.is_integer():
                raise ValueError
            return int(f)
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be int") from exc


def _source_row(row: Mapping[str, Any], fallback_source_row: int) -> int:
    if _has_nonempty(row, "raw_source_row"):
        value = row["raw_source_row"]
    elif _has_nonempty(row, "source_row"):
        value = row["source_row"]
    else:
        return fallback_source_row
    out = _coerce_int(value, "source_row")
    if out < 0:
        raise ValueError("source_row must be >= 0")
    return out


__all__ = [
    "ExecutionTapeBuildConfig",
    "build_execution_tape_from_config",
    "build_arg_parser",
    "_flatten_repeated",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
