"""Standalone audit CLI for incremental L2 book reconstruction.

This module streams normalized or raw CSV/Parquet-compatible Tardis
``incremental_book_L2`` rows, converts them to execution-layer L2 contracts,
groups them with the execution batcher, and feeds them into the execution L2
book reconstructor. It intentionally does not build execution tapes, parse
trades, compute features/labels, simulate fills, or mutate storage datasets.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # optional dependency for Parquet input
    pq = None

from mmrt.contracts import BookSide
from mmrt.execution.contracts import L2Update, SymbolSpec
from mmrt.execution.l2_reconstructor import L2BookReconstructor, ReconstructedL2Event, iter_l2_update_batches


@dataclass(frozen=True, slots=True)
class L2AuditConfig:
    l2_inputs: tuple[str, ...]
    exchange: str = "binance-futures"
    symbol: str = "BTCUSDT"
    tick_size: float = 0.1
    step_size: float = 0.001
    min_qty: float = 0.001
    max_qty: float = 100.0
    min_notional: float = 5.0
    max_rows: int | None = None
    sample_event_limit: int = 10
    batch_size: int = 65_536

    def __post_init__(self) -> None:
        inputs = _tuple_of_nonempty_str(self.l2_inputs, "l2_inputs")
        if not inputs:
            raise ValueError("l2_inputs must be nonempty")
        object.__setattr__(self, "l2_inputs", inputs)
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")
        object.__setattr__(self, "tick_size", _require_positive_float(self.tick_size, "tick_size"))
        object.__setattr__(self, "step_size", _require_positive_float(self.step_size, "step_size"))
        object.__setattr__(self, "min_qty", _require_nonnegative_float(self.min_qty, "min_qty"))
        object.__setattr__(self, "max_qty", _require_positive_float(self.max_qty, "max_qty"))
        object.__setattr__(self, "min_notional", _require_nonnegative_float(self.min_notional, "min_notional"))
        if self.max_qty < self.min_qty:
            raise ValueError("max_qty must be >= min_qty")
        if self.max_rows is not None:
            _require_positive_int(self.max_rows, "max_rows")
        _require_nonnegative_int(self.sample_event_limit, "sample_event_limit")
        _require_positive_int(self.batch_size, "batch_size")


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _tuple_of_nonempty_str(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of non-empty strings, not a single string/bytes value")
    try:
        seq = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of non-empty strings") from exc
    return tuple(_require_nonempty_str(v, f"{name}[{i}]") for i, v in enumerate(seq))


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_finite_float(value: float, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite float")
    return out


def _require_positive_float(value: float, name: str) -> float:
    out = _require_finite_float(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be > 0")
    return out


def _require_nonnegative_float(value: float, name: str) -> float:
    out = _require_finite_float(value, name)
    if out < 0:
        raise ValueError(f"{name} must be >= 0")
    return out


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _required_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    joined = " or ".join(names)
    raise ValueError(f"missing required column/value: {joined}")


def _parse_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive int")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer value")
        parsed = int(value)
    else:
        text = str(value).strip()
        try:
            parsed = int(text)
        except ValueError:
            as_float = float(text)
            if not as_float.is_integer():
                raise ValueError(f"{name} must be an integer value")
            parsed = int(as_float)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _parse_nonnegative_int(value: Any, name: str) -> int:
    parsed = _parse_positive_int(value, name) if value not in (0, "0") else 0
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "t", "yes", "y", "1"):
            return True
        if text in ("false", "f", "no", "n", "0"):
            return False
    raise ValueError(f"{name} must be a clear bool value")


def _parse_side(value: Any) -> BookSide:
    if isinstance(value, BookSide):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == BookSide.BID.value:
            return BookSide.BID
        if text == BookSide.ASK.value:
            return BookSide.ASK
    raise ValueError(f"side has invalid value {value!r}")


def _row_to_l2_update(row: Mapping[str, Any], *, symbol_spec: SymbolSpec, fallback_source_row: int) -> L2Update:
    local_ts_us = _parse_positive_int(_required_value(row, "local_ts_us", "local_timestamp"), "local_ts_us")
    ts_us = _parse_positive_int(_required_value(row, "ts_us", "timestamp"), "ts_us")
    if "price_tick" in row and row["price_tick"] not in (None, ""):
        price_tick = _parse_positive_int(row["price_tick"], "price_tick")
    else:
        price_tick = symbol_spec.price_to_tick(_require_positive_float(_required_value(row, "price"), "price"))
    source_value = row.get("raw_source_row", row.get("source_row", fallback_source_row))
    return L2Update(
        local_ts_us=local_ts_us,
        ts_us=ts_us,
        side=_parse_side(_required_value(row, "side")),
        price_tick=price_tick,
        amount=_require_nonnegative_float(_required_value(row, "amount"), "amount"),
        is_snapshot=_parse_bool(_required_value(row, "is_snapshot"), "is_snapshot"),
        source_row=_parse_nonnegative_int(source_value, "source_row"),
    )


def _iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    open_fn = gzip.open if path.name.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _iter_parquet_rows(path: Path, *, batch_size: int) -> Iterator[dict[str, Any]]:
    if pq is None:
        raise ModuleNotFoundError("pyarrow is required to read Parquet L2 inputs")
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            yield {col: values[i] for col, values in data.items()}


def _iter_path_rows(path: Path, *, batch_size: int) -> Iterator[dict[str, Any]]:
    name = path.name.lower()
    if name.endswith(".parquet"):
        yield from _iter_parquet_rows(path, batch_size=batch_size)
    elif name.endswith(".csv") or name.endswith(".csv.gz"):
        yield from _iter_csv_rows(path)
    else:
        raise ValueError(f"unsupported L2 input suffix for {path}; expected .parquet, .csv, or .csv.gz")


def _iter_l2_rows(paths: Sequence[Path], *, batch_size: int) -> Iterator[tuple[int, Mapping[str, Any]]]:
    row_idx = 0
    for path in paths:
        for row in _iter_path_rows(path, batch_size=batch_size):
            yield row_idx, row
            row_idx += 1


def _event_sample(event: ReconstructedL2Event) -> dict[str, object]:
    top = event.book_top
    return {
        "batch_seq": event.batch_seq,
        "local_ts_us": event.local_ts_us,
        "min_ts_us": event.min_ts_us,
        "max_ts_us": event.max_ts_us,
        "num_updates": event.num_updates,
        "is_snapshot_batch": event.is_snapshot_batch,
        "bid_depth": event.bid_depth,
        "ask_depth": event.ask_depth,
        "crossed_repaired": event.crossed_repaired,
        "crossed_levels_removed": event.crossed_levels_removed,
        "best_bid_tick": top.best_bid_tick if top else None,
        "best_ask_tick": top.best_ask_tick if top else None,
        "spread_ticks": top.spread_ticks if top else None,
    }


def audit_l2_reconstruction(config: L2AuditConfig) -> dict[str, object]:
    if not isinstance(config, L2AuditConfig):
        raise ValueError("config must be L2AuditConfig")
    symbol_spec = SymbolSpec(
        exchange=config.exchange,
        symbol=config.symbol,
        tick_size=config.tick_size,
        step_size=config.step_size,
        min_qty=config.min_qty,
        max_qty=config.max_qty,
        min_notional=config.min_notional,
    )
    reconstructor = L2BookReconstructor(symbol_spec)
    paths = tuple(Path(p) for p in config.l2_inputs)

    rows_seen = 0
    updates_converted = 0
    scan_limit_hit = False
    first_local_ts_us: int | None = None
    last_local_ts_us: int | None = None
    first_ts_us: int | None = None
    last_ts_us: int | None = None

    def updates() -> Iterator[L2Update]:
        nonlocal rows_seen, updates_converted, scan_limit_hit
        nonlocal first_local_ts_us, last_local_ts_us, first_ts_us, last_ts_us
        for row_idx, row in _iter_l2_rows(paths, batch_size=config.batch_size):
            if config.max_rows is not None and rows_seen >= config.max_rows:
                scan_limit_hit = True
                break
            rows_seen += 1
            update = _row_to_l2_update(row, symbol_spec=symbol_spec, fallback_source_row=row_idx)
            updates_converted += 1
            if first_local_ts_us is None:
                first_local_ts_us = update.local_ts_us
            if first_ts_us is None:
                first_ts_us = update.ts_us
            last_local_ts_us = update.local_ts_us
            last_ts_us = update.ts_us
            yield update

    first_events: list[dict[str, object]] = []
    last_events: deque[dict[str, object]] = deque(maxlen=config.sample_event_limit)
    events_with_book_top = 0
    events_without_book_top = 0
    one_sided_or_empty_event_count = 0
    spread_sum = 0
    min_spread_ticks: int | None = None
    max_spread_ticks: int | None = None
    max_crossed_levels_removed_in_event = 0

    for batch in iter_l2_update_batches(updates()):
        event = reconstructor.apply_batch(batch)
        if event is None:
            continue
        sample = _event_sample(event)
        if len(first_events) < config.sample_event_limit:
            first_events.append(sample)
        if config.sample_event_limit > 0:
            last_events.append(sample)
        if event.book_top is None:
            events_without_book_top += 1
            one_sided_or_empty_event_count += 1
        else:
            events_with_book_top += 1
            spread = event.book_top.spread_ticks
            spread_sum += spread
            min_spread_ticks = spread if min_spread_ticks is None else min(min_spread_ticks, spread)
            max_spread_ticks = spread if max_spread_ticks is None else max(max_spread_ticks, spread)
        max_crossed_levels_removed_in_event = max(max_crossed_levels_removed_in_event, event.crossed_levels_removed)

    counters = reconstructor.counters
    warnings = _warnings_from_values(
        rows_seen=rows_seen,
        events_emitted=counters.emitted_event_count,
        skipped_pre_snapshot_updates=counters.skipped_pre_snapshot_updates,
        snapshot_reset_count=counters.snapshot_reset_count,
        crossed_repair_count=counters.crossed_repair_count,
        missing_delete_count=counters.missing_delete_count,
        one_sided_or_empty_event_count=one_sided_or_empty_event_count,
        scan_limit_hit=scan_limit_hit,
    )
    return {
        "status": "ok",
        "audit_type": "l2_reconstruction",
        "inputs": {
            "l2_input_count": len(config.l2_inputs),
            "l2_inputs": list(config.l2_inputs),
            "max_rows": config.max_rows,
            "batch_size": config.batch_size,
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
            "rows_seen": rows_seen,
            "updates_converted": updates_converted,
            "batches_seen": counters.batches_seen,
            "updates_seen_by_reconstructor": counters.updates_seen,
            "events_emitted": counters.emitted_event_count,
        },
        "reconstruction": {
            "status": reconstructor.status.value,
            "is_ready": reconstructor.is_ready,
            "skipped_pre_snapshot_updates": counters.skipped_pre_snapshot_updates,
            "snapshot_reset_count": counters.snapshot_reset_count,
            "applied_update_count": counters.applied_update_count,
            "deleted_level_count": counters.deleted_level_count,
            "missing_delete_count": counters.missing_delete_count,
            "crossed_batch_count": counters.crossed_batch_count,
            "crossed_repair_count": counters.crossed_repair_count,
            "crossed_levels_removed": counters.crossed_levels_removed,
            "local_ts_decrease_count": counters.local_ts_decrease_count,
            "max_bid_depth": counters.max_bid_depth,
            "max_ask_depth": counters.max_ask_depth,
            "max_batch_size": counters.max_batch_size,
        },
        "time_range": {
            "first_local_ts_us": first_local_ts_us,
            "last_local_ts_us": last_local_ts_us,
            "first_ts_us": first_ts_us,
            "last_ts_us": last_ts_us,
        },
        "book_quality": {
            "events_with_book_top": events_with_book_top,
            "events_without_book_top": events_without_book_top,
            "one_sided_or_empty_event_count": one_sided_or_empty_event_count,
            "min_spread_ticks": min_spread_ticks,
            "max_spread_ticks": max_spread_ticks,
            "mean_spread_ticks": (spread_sum / events_with_book_top) if events_with_book_top else None,
            "max_crossed_levels_removed_in_event": max_crossed_levels_removed_in_event,
        },
        "samples": {
            "first_events": first_events,
            "last_events": list(last_events),
        },
        "warnings": warnings,
    }


def _warnings_from_values(
    *,
    rows_seen: int,
    events_emitted: int,
    skipped_pre_snapshot_updates: int,
    snapshot_reset_count: int,
    crossed_repair_count: int,
    missing_delete_count: int,
    one_sided_or_empty_event_count: int,
    scan_limit_hit: bool,
) -> list[str]:
    warnings: list[str] = []
    if snapshot_reset_count == 0:
        warnings.append("no_snapshot_seen")
    if events_emitted == 0:
        warnings.append("no_reconstructed_events")
    if rows_seen > 0 and skipped_pre_snapshot_updates / rows_seen > 0.05:
        warnings.append("high_pre_snapshot_skip_fraction")
    if crossed_repair_count > 0:
        warnings.append("crossed_repairs_observed")
    if missing_delete_count > 0:
        warnings.append("missing_deletes_observed")
    if one_sided_or_empty_event_count > 0:
        warnings.append("one_sided_or_empty_books_observed")
    if scan_limit_hit:
        warnings.append("scan_limit_hit")
    return warnings


def _write_json_atomic(report: dict[str, object], path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    target = Path(path)
    if target.suffix != ".json":
        raise ValueError("output path must end with .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan = True) + "\n"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return str(target)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan = True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrt-audit-l2-reconstruction",
        description="Audit incremental L2 reconstruction using execution-layer semantics.",
    )
    parser.add_argument("--l2-input", nargs="+", required=True, dest="l2_inputs", help="Input .parquet, .csv, or .csv.gz files.")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--min-qty", type=float, default=0.001)
    parser.add_argument("--max-qty", type=float, default=100.0)
    parser.add_argument("--min-notional", type=float, default=5.0)
    parser.add_argument("--max-rows", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=65_536)
    parser.add_argument("--sample-event-limit", type=_nonnegative_int, default=10)
    parser.add_argument("--output-json", default=None, help="Optional path where the full audit JSON report will be written.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = L2AuditConfig(
        l2_inputs=tuple(args.l2_inputs),
        exchange=args.exchange,
        symbol=args.symbol,
        tick_size=args.tick_size,
        step_size=args.step_size,
        min_qty=args.min_qty,
        max_qty=args.max_qty,
        min_notional=args.min_notional,
        max_rows=args.max_rows,
        sample_event_limit=args.sample_event_limit,
        batch_size=args.batch_size,
    )
    report = audit_l2_reconstruction(config)
    if args.output_json is not None:
        report = dict(report)
        output_json = str(args.output_json)
        report["output_json"] = output_json
        _write_json_atomic(report, output_json)
    _print_json(report)
    return 0


__all__ = [
    "L2AuditConfig",
    "audit_l2_reconstruction",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
