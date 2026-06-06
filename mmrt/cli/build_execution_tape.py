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
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:  # Optional dependency for Parquet input support.
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - exercised only without pyarrow
    pq = None

from mmrt.contracts import AggressorSide, BookSide
from mmrt.execution.contracts import L2Update, SymbolSpec, TradePrint
from mmrt.execution.event_merge import ExecutionMergeTiePolicy, merge_execution_events
from mmrt.execution.execution_tape import (
    build_execution_tape as build_execution_tape_object,
    save_execution_tape,
)
from mmrt.execution.l2_reconstructor import (
    L2BookReconstructor,
    ReconstructedL2Event,
    iter_l2_update_batches,
)


@dataclass(frozen=True, slots=True)
class ExecutionTapeBuildConfig:
    l2_inputs: tuple[str, ...]
    trade_inputs: tuple[str, ...]
    output_root: str

    exchange: str = "binance-futures"
    symbol: str = "BTCUSDT"
    tick_size: float = 0.1
    step_size: float = 0.001
    min_qty: float = 0.001
    max_qty: float = 100.0
    min_notional: float = 5.0

    batch_size: int = 65_536
    book_depth: int = 25
    max_l2_rows: int | None = None
    max_trade_rows: int | None = None
    tie_policy: ExecutionMergeTiePolicy | str = ExecutionMergeTiePolicy.L2_BEFORE_TRADE
    overwrite: bool = False
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
        object.__setattr__(self, "tick_size", _require_positive_float(self.tick_size, "tick_size"))
        object.__setattr__(self, "step_size", _require_positive_float(self.step_size, "step_size"))
        object.__setattr__(self, "min_qty", _require_nonnegative_float(self.min_qty, "min_qty"))
        object.__setattr__(self, "max_qty", _require_positive_float(self.max_qty, "max_qty"))
        if self.max_qty < self.min_qty:
            raise ValueError("max_qty must be >= min_qty")
        object.__setattr__(self, "min_notional", _require_nonnegative_float(self.min_notional, "min_notional"))
        object.__setattr__(self, "batch_size", _require_positive_int(self.batch_size, "batch_size"))
        object.__setattr__(self, "book_depth", _require_positive_int(self.book_depth, "book_depth"))
        object.__setattr__(self, "max_l2_rows", _optional_positive_int(self.max_l2_rows, "max_l2_rows"))
        object.__setattr__(self, "max_trade_rows", _optional_positive_int(self.max_trade_rows, "max_trade_rows"))
        object.__setattr__(self, "tie_policy", _coerce_tie_policy(self.tie_policy))
        if not isinstance(self.overwrite, bool):
            raise ValueError("overwrite must be bool")
        if not isinstance(self.created_at_utc, str):
            raise ValueError("created_at_utc must be str")


def build_execution_tape_from_config(config: ExecutionTapeBuildConfig) -> dict[str, object]:
    """Build and save an execution tape from validated file inputs."""
    if not isinstance(config, ExecutionTapeBuildConfig):
        raise ValueError("config must be ExecutionTapeBuildConfig")

    symbol_spec = SymbolSpec(
        exchange=config.exchange,
        symbol=config.symbol,
        tick_size=config.tick_size,
        step_size=config.step_size,
        min_qty=config.min_qty,
        max_qty=config.max_qty,
        min_notional=config.min_notional,
    )
    l2_paths = _resolve_input_paths(config.l2_inputs, "l2_inputs")
    trade_paths = _resolve_input_paths(config.trade_inputs, "trade_inputs")
    output_root = Path(config.output_root)
    summary_path = output_root / "build_summary.json"
    if summary_path.exists() and not config.overwrite:
        raise FileExistsError(f"JSON output already exists: {summary_path}")

    l2_events, l2_stats = load_reconstructed_l2_events(
        l2_paths,
        symbol_spec=symbol_spec,
        batch_size=config.batch_size,
        max_rows=config.max_l2_rows,
        book_depth=config.book_depth,
    )
    trades, trade_stats = load_trade_prints(
        trade_paths,
        symbol_spec=symbol_spec,
        batch_size=config.batch_size,
        max_rows=config.max_trade_rows,
    )

    if not l2_events:
        raise ValueError("cannot build execution tape without reconstructed L2 events")
    if not trades:
        raise ValueError("cannot build execution tape without trades")

    plan = merge_execution_events(l2_events, trades, tie_policy=config.tie_policy)
    tape = build_execution_tape_object(
        symbol_spec=symbol_spec,
        l2_events=l2_events,
        trades=trades,
        merged_events=plan.events,
        book_depth=config.book_depth,
        created_at_utc=config.created_at_utc,
        notes={
            "builder": "mmrt.cli.build_execution_tape",
            "tie_policy": config.tie_policy.value,
        },
    )

    save_execution_tape(tape, output_root, overwrite=config.overwrite)
    summary = _build_summary(config, output_root, l2_stats, trade_stats, plan, tape)
    _write_json_atomic(summary, summary_path, overwrite=config.overwrite)
    return summary


def load_reconstructed_l2_events(
    paths: Sequence[Path],
    *,
    symbol_spec: SymbolSpec,
    batch_size: int,
    max_rows: int | None = None,
    book_depth: int = 25,
) -> tuple[list[ReconstructedL2Event], dict[str, object]]:
    book_depth = _require_positive_int(book_depth, "book_depth")
    reconstructor = L2BookReconstructor(symbol_spec, snapshot_depth=book_depth)
    events: list[ReconstructedL2Event] = []
    rows_seen = 0
    first_local_ts_us: int | None = None
    last_local_ts_us: int | None = None
    scan_limit_hit = False

    def iter_updates() -> Iterator[L2Update]:
        nonlocal rows_seen, first_local_ts_us, last_local_ts_us, scan_limit_hit
        for path in paths:
            for row in _iter_path_rows(path, batch_size=batch_size):
                if max_rows is not None and rows_seen >= max_rows:
                    scan_limit_hit = True
                    return
                fallback_source_row = rows_seen
                rows_seen += 1
                update = _row_to_l2_update(row, symbol_spec=symbol_spec, fallback_source_row=fallback_source_row)
                if first_local_ts_us is None:
                    first_local_ts_us = update.local_ts_us
                last_local_ts_us = update.local_ts_us
                yield update

    for batch in iter_l2_update_batches(iter_updates()):
        event = reconstructor.apply_batch(batch)
        if event is not None:
            events.append(event)

    counters = reconstructor.counters
    stats: dict[str, object] = {
        "rows_seen": rows_seen,
        "updates_converted": rows_seen,
        "scan_limit_hit": scan_limit_hit,
        "first_local_ts_us": first_local_ts_us,
        "last_local_ts_us": last_local_ts_us,
        "status": reconstructor.status.value,
        "is_ready": reconstructor.is_ready,
        "batches_seen": counters.batches_seen,
        "updates_seen": counters.updates_seen,
        "skipped_pre_snapshot_updates": counters.skipped_pre_snapshot_updates,
        "snapshot_reset_count": counters.snapshot_reset_count,
        "applied_update_count": counters.applied_update_count,
        "deleted_level_count": counters.deleted_level_count,
        "missing_delete_count": counters.missing_delete_count,
        "emitted_event_count": counters.emitted_event_count,
        "crossed_batch_count": counters.crossed_batch_count,
        "crossed_repair_count": counters.crossed_repair_count,
        "crossed_levels_removed": counters.crossed_levels_removed,
        "local_ts_decrease_count": counters.local_ts_decrease_count,
        "max_bid_depth": counters.max_bid_depth,
        "max_ask_depth": counters.max_ask_depth,
        "max_batch_size": counters.max_batch_size,
        "book_depth": book_depth,
    }
    return events, stats


def load_trade_prints(
    paths: Sequence[Path],
    *,
    symbol_spec: SymbolSpec,
    batch_size: int,
    max_rows: int | None = None,
) -> tuple[list[TradePrint], dict[str, object]]:
    trades: list[TradePrint] = []
    rows_seen = 0
    first_local_ts_us: int | None = None
    last_local_ts_us: int | None = None
    prev_local_ts_us: int | None = None
    scan_limit_hit = False

    for path in paths:
        for row in _iter_path_rows(path, batch_size=batch_size):
            if max_rows is not None and rows_seen >= max_rows:
                scan_limit_hit = True
                stats = {
                    "rows_seen": rows_seen,
                    "trades_converted": len(trades),
                    "scan_limit_hit": scan_limit_hit,
                    "first_local_ts_us": first_local_ts_us,
                    "last_local_ts_us": last_local_ts_us,
                }
                return trades, stats
            fallback_source_row = rows_seen
            rows_seen += 1
            trade = _row_to_trade_print(row, symbol_spec=symbol_spec, fallback_source_row=fallback_source_row)
            if prev_local_ts_us is not None and trade.local_ts_us < prev_local_ts_us:
                raise ValueError("trades must be sorted by nondecreasing local_ts_us")
            if first_local_ts_us is None:
                first_local_ts_us = trade.local_ts_us
            last_local_ts_us = trade.local_ts_us
            prev_local_ts_us = trade.local_ts_us
            trades.append(trade)

    stats = {
        "rows_seen": rows_seen,
        "trades_converted": len(trades),
        "scan_limit_hit": scan_limit_hit,
        "first_local_ts_us": first_local_ts_us,
        "last_local_ts_us": last_local_ts_us,
    }
    return trades, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an execution tape from Tardis-style L2 and trade inputs.")
    parser.add_argument("--l2-input", nargs="+", required=True, dest="l2_input")
    parser.add_argument("--trade-input", nargs="+", required=True, dest="trade_input")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--min-qty", type=float, default=0.001)
    parser.add_argument("--max-qty", type=float, default=100.0)
    parser.add_argument("--min-notional", type=float, default=5.0)
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
    parser.add_argument("--created-at-utc", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExecutionTapeBuildConfig(
        l2_inputs=tuple(args.l2_input),
        trade_inputs=tuple(args.trade_input),
        output_root=args.output_root,
        exchange=args.exchange,
        symbol=args.symbol,
        tick_size=args.tick_size,
        step_size=args.step_size,
        min_qty=args.min_qty,
        max_qty=args.max_qty,
        min_notional=args.min_notional,
        batch_size=args.batch_size,
        book_depth=args.book_depth,
        max_l2_rows=args.max_l2_rows,
        max_trade_rows=args.max_trade_rows,
        tie_policy=args.tie_policy,
        overwrite=args.overwrite,
        created_at_utc=args.created_at_utc,
    )
    summary = build_execution_tape_from_config(config)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
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


def _build_summary(config, output_root, l2_stats, trade_stats, plan, tape) -> dict[str, object]:
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
    if plan.counters.same_local_ts_tie_count > 0:
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
            "tick_size": config.tick_size,
            "step_size": config.step_size,
            "min_qty": config.min_qty,
            "max_qty": config.max_qty,
            "min_notional": config.min_notional,
        },
        "counts": {
            "l2_rows_seen": l2_stats["rows_seen"],
            "l2_updates_converted": l2_stats["updates_converted"],
            "l2_batches_seen": l2_stats["batches_seen"],
            "l2_events_emitted": l2_stats["emitted_event_count"],
            "trade_rows_seen": trade_stats["rows_seen"],
            "trades_converted": trade_stats["trades_converted"],
            "merged_events": plan.counters.emitted_event_count,
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
            "l2_event_count": plan.counters.l2_event_count,
            "trade_count": plan.counters.trade_count,
            "emitted_event_count": plan.counters.emitted_event_count,
            "same_local_ts_tie_count": plan.counters.same_local_ts_tie_count,
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
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=True)
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return str(path)


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
            if normalized in ("unknown", ""):
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
    return AggressorSide.UNKNOWN


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
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
