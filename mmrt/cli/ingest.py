from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from mmrt import config as cfg
from mmrt.contracts import TardisDataType
from mmrt.data import binance_futures_adapter as bfa
from mmrt.data import event_merge as em
from mmrt.data import tardis_csv as tc
from mmrt.features.book_state import BookSnapshotInput
from mmrt.features.engine import FeatureEngine, FeatureEngineConfig
from mmrt.features.labels import LabelBuilder
from mmrt.features.trade_state import TradeInput
from mmrt.features.transforms import CausalFeatureTransformer, TransformConfig, TransformDiagnostics
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _parse_csv_ints(text: str, name: str) -> tuple[int, ...]:
    s = _require_nonempty_str(text, name)
    out: list[int] = []
    for idx, part in enumerate(s.split(",")):
        p = part.strip()
        if not p:
            raise ValueError(f"{name}[{idx}] is empty")
        try:
            val = int(p)
        except ValueError as exc:
            raise ValueError(f"{name}[{idx}] must be int") from exc
        out.append(_require_positive_int(val, f"{name}[{idx}]"))
    if not out:
        raise ValueError(f"{name} must not be empty")
    return tuple(out)


def _parse_us_range(text: str, name: str) -> tuple[int, int]:
    s = _require_nonempty_str(text, name)
    if s.count(":") != 1:
        raise ValueError(f"{name} must be START_US:END_US")
    a, b = s.split(":")
    try:
        start = int(a)
        end = int(b)
    except ValueError as exc:
        raise ValueError(f"{name} must be START_US:END_US") from exc
    _require_positive_int(start, f"{name}.start")
    _require_positive_int(end, f"{name}.end")
    if end <= start:
        raise ValueError(f"{name} end must be > start")
    return start, end


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_paths(paths: Sequence[str], name: str) -> tuple[Path, ...]:
    seq = tuple(paths)
    if not seq:
        raise ValueError(f"{name} must not be empty")
    out: list[Path] = []
    for i, p in enumerate(seq):
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"{name}[{i}] not found: {path}")
        if not path.is_file():
            raise ValueError(f"{name}[{i}] must be a file: {path}")
        out.append(path)
    return tuple(out)




def _safe_posix_leaf(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "input.csv"


def _safe_manifest_source_files(book_paths: Sequence[Path], trade_paths: Sequence[Path]) -> tuple[str, ...]:
    out: list[str] = []
    for i, p in enumerate(book_paths):
        out.append(f"source/book_snapshot_25/{i:06d}_{_safe_posix_leaf(p.name)}")
    for i, p in enumerate(trade_paths):
        out.append(f"source/trades/{i:06d}_{_safe_posix_leaf(p.name)}")
    return tuple(out)

def _build_pipeline_config(args: argparse.Namespace, *, exchange: str, symbol: str, label_horizons_us: tuple[int, ...]) -> cfg.PipelineConfig:
    base = cfg.default_config()
    return cfg.PipelineConfig(
        market=cfg.MarketConfig(exchange=exchange, symbol=symbol),
        data=cfg.DataConfig(source_data_types=(TardisDataType.BOOK_SNAPSHOT_25, TardisDataType.TRADES), disabled_context_data_types=base.data.disabled_context_data_types),
        decision=cfg.DecisionConfig(policy=base.decision.policy, reason=base.decision.reason, stride_us=args.decision_stride_us),
        labels=cfg.LabelConfig(horizons_us=label_horizons_us, entry_delay_us=args.label_entry_delay_us),
        runtime=base.runtime,
        storage=base.storage,
    )


def _normalize_input_files(book_csv: tuple[Path, ...], trades_csv: tuple[Path, ...], work_dir: Path, exchange: str, symbol: str) -> tuple[tc.NormalizedTardisFile, ...]:
    out: list[tc.NormalizedTardisFile] = []
    bdir = work_dir / "normalized" / "book_snapshot_25"
    tdir = work_dir / "normalized" / "trades"
    bdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(book_csv):
        dst = bdir / f"book_snapshot_25_{i:06d}.parquet"
        nf = tc.write_normalized_parquet(src, dst, TardisDataType.BOOK_SNAPSHOT_25, source_file=str(src))
        _validate_normalized_market(nf, exchange=exchange, symbol=symbol)
        out.append(nf)
    for i, src in enumerate(trades_csv):
        dst = tdir / f"trades_{i:06d}.parquet"
        nf = tc.write_normalized_parquet(src, dst, TardisDataType.TRADES, source_file=str(src))
        _validate_normalized_market(nf, exchange=exchange, symbol=symbol)
        out.append(nf)
    return tuple(out)


def _validate_normalized_market(normalized_file: tc.NormalizedTardisFile, *, exchange: str, symbol: str) -> None:
    bad = (
        pl.scan_parquet(str(normalized_file.output_path))
        .select(["exchange", "symbol"])
        .filter((pl.col("exchange").is_not_null() & (pl.col("exchange") != exchange)) | (pl.col("symbol").is_not_null() & (pl.col("symbol") != symbol)))
        .limit(1)
        .collect()
    )
    if bad.height:
        actual_exchange = bad["exchange"][0]
        actual_symbol = bad["symbol"][0]
        raise ValueError(
            f"normalized file market mismatch: path={normalized_file.output_path} expected=({exchange},{symbol}) actual=({actual_exchange},{actual_symbol})"
        )


def _build_merge_inputs(normalized_files: Sequence[tc.NormalizedTardisFile]) -> tuple[em.EventMergeInput, ...]:
    out: list[em.EventMergeInput] = []
    for i, nf in enumerate(normalized_files):
        base = bfa.binance_futures_default_merge_rank(nf.data_type)
        rank = base * 1_000_000 + i
        out.append(em.parquet_merge_input(nf.output_path, nf.data_type, rank))
    return tuple(out)


def _write_merged_events(normalized_files: Sequence[tc.NormalizedTardisFile], work_dir: Path) -> Path:
    out_path = work_dir / "merged" / "events.parquet"
    em.write_merged_events_parquet(_build_merge_inputs(normalized_files), out_path)
    return out_path


def _iter_record_batch_rows(parquet_path: Path, columns: Sequence[str], batch_size: int):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=list(columns), batch_size=batch_size):
        data = batch.to_pydict()
        names = list(data.keys())
        n = batch.num_rows
        for i in range(n):
            yield {k: data[k][i] for k in names}


def _to_float_or_zero(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not math.isfinite(f) else f


def _book_snapshot_input_from_row(row: Mapping[str, Any]) -> BookSnapshotInput | None:
    bid_px = np.array([_to_float_or_zero(row.get(f"bid_px_{i:02d}")) for i in range(25)], dtype=np.float64)
    bid_sz = np.array([_to_float_or_zero(row.get(f"bid_sz_{i:02d}")) for i in range(25)], dtype=np.float64)
    ask_px = np.array([_to_float_or_zero(row.get(f"ask_px_{i:02d}")) for i in range(25)], dtype=np.float64)
    ask_sz = np.array([_to_float_or_zero(row.get(f"ask_sz_{i:02d}")) for i in range(25)], dtype=np.float64)
    if bid_px[0] <= 0.0 or ask_px[0] <= 0.0:
        return None
    return BookSnapshotInput(local_ts_us=int(row[tc.LOCAL_TS_US]), ts_us=int(row[tc.TS_US]), event_seq=int(row[em.EVENT_SEQ]), bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)


def _trade_input_from_row(row: Mapping[str, Any]) -> TradeInput | None:
    price = _to_float_or_zero(row.get("price"))
    amount = _to_float_or_zero(row.get("amount"))
    if price <= 0.0 or amount <= 0.0:
        return None
    return TradeInput(local_ts_us=int(row[tc.LOCAL_TS_US]), ts_us=int(row[tc.TS_US]), price=price, amount=amount, side_code=int(row["side_code"]), event_seq=int(row[em.EVENT_SEQ]))


@dataclass(slots=True)
class PendingDecision:
    decision_index: int
    ts_us: int
    local_ts_us: int
    event_seq: int
    raw_mid: float
    feature_values: tuple[float, ...]


@dataclass(slots=True)
class IngestCounters:
    input_book_files: int = 0
    input_trade_files: int = 0
    normalized_files: int = 0
    merged_events_seen: int = 0
    book_events_seen: int = 0
    trade_events_seen: int = 0
    skipped_empty_book_events: int = 0
    skipped_bad_trade_events: int = 0
    decisions_emitted: int = 0
    labels_matured: int = 0
    rows_written: int = 0
    pending_decisions_at_eof: int = 0
    transform_rows_seen: int = 0
    output_segments: int = 0
    output_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _write_matured_labels(label_results, pending_decisions: dict[int, PendingDecision], writer: wr.DecisionRowWriter, counters: IngestCounters) -> None:
    for label in label_results:
        key = int(label.decision_ts_us)
        if key not in pending_decisions:
            raise KeyError(f"missing pending decision for {key}")
        p = pending_decisions.pop(key)
        writer.append_values(decision_index=p.decision_index, ts_us=p.ts_us, local_ts_us=p.local_ts_us, event_seq=p.event_seq, raw_mid=p.raw_mid, label_entry_ts_us=int(label.entry_ts_us), label_values=label.values_bps, feature_values=p.feature_values)
        counters.labels_matured += 1
        counters.rows_written += 1


def _run_causal_ingest(merged_path: Path, writer: wr.DecisionRowWriter, pipeline_config: cfg.PipelineConfig, event_batch_size: int, max_events: int | None) -> tuple[IngestCounters, TransformConfig, TransformDiagnostics]:
    counters = IngestCounters()
    engine = FeatureEngine(FeatureEngineConfig(decision_stride_us=pipeline_config.decision.stride_us))
    label_builder = LabelBuilder(pipeline_config.label_spec)
    tcfg = TransformConfig()
    transformer = CausalFeatureTransformer(tcfg)
    pending_decisions: dict[int, PendingDecision] = {}
    cols = [em.EVENT_TYPE_CODE, em.EVENT_SEQ, tc.TS_US, tc.LOCAL_TS_US, "price", "amount", "side_code", *[f"bid_px_{i:02d}" for i in range(25)], *[f"bid_sz_{i:02d}" for i in range(25)], *[f"ask_px_{i:02d}" for i in range(25)], *[f"ask_sz_{i:02d}" for i in range(25)]]

    for row in _iter_record_batch_rows(merged_path, cols, event_batch_size):
        if max_events is not None and counters.merged_events_seen >= max_events:
            break
        counters.merged_events_seen += 1
        code = int(row[em.EVENT_TYPE_CODE])
        if code == em.EVENT_TYPE_CODE_TRADE:
            counters.trade_events_seen += 1
            tr = _trade_input_from_row(row)
            if tr is None:
                counters.skipped_bad_trade_events += 1
                continue
            engine.on_trade(tr)
            continue
        if code == em.EVENT_TYPE_CODE_BOOK_SNAPSHOT:
            counters.book_events_seen += 1
            snap = _book_snapshot_input_from_row(row)
            if snap is None:
                counters.skipped_empty_book_events += 1
                continue
            decision = engine.on_book_snapshot(snap)
            # Causality contract: observe current book price first, mature older labels,
            # then create/store current decision transformed at decision time.
            mid = engine.book_state.current_summary().mid
            _write_matured_labels(label_builder.observe_price_local(snap.local_ts_us, mid), pending_decisions, writer, counters)
            if decision is not None:
                transformed = transformer.transform_one_local(decision.local_ts_us, decision.feature_vector)
                label_builder.on_decision_local(decision.local_ts_us)
                pending_decisions[decision.local_ts_us] = PendingDecision(decision_index=decision.decision_index, ts_us=decision.ts_us, local_ts_us=decision.local_ts_us, event_seq=decision.event_seq, raw_mid=decision.raw_mid, feature_values=tuple(float(x) for x in transformed))
                counters.decisions_emitted += 1

    _write_matured_labels(label_builder.mature_ready(), pending_decisions, writer, counters)
    counters.pending_decisions_at_eof = len(pending_decisions)
    counters.transform_rows_seen = transformer.diagnostics.rows_seen

    if counters.book_events_seen == 0:
        raise ValueError("no book events seen")
    if counters.trade_events_seen == 0:
        raise ValueError("no trade events seen")
    if counters.decisions_emitted == 0:
        raise ValueError("no decisions emitted")
    if counters.rows_written == 0:
        raise ValueError("no matured rows written")
    return counters, tcfg, transformer.diagnostics_snapshot()


def _transform_config_to_dict(config: TransformConfig) -> dict:
    return dict(config.as_dict())


def _transform_diagnostics_to_dict(diag: TransformDiagnostics) -> dict:
    return diag.as_dict()


def _patch_manifest_transform_metadata(dataset_root: Path, transform_config: TransformConfig, transform_diagnostics: TransformDiagnostics, counters: IngestCounters) -> mf.StorageManifest:
    mp = dataset_root / mf.DEFAULT_MANIFEST_FILENAME
    manifest = mf.read_manifest_json(mp)
    notes = dict(manifest.notes)
    notes.update({"ingest_counters": counters.to_dict()})
    updated = replace(manifest, transform_config=_transform_config_to_dict(transform_config), transform_diagnostics=_transform_diagnostics_to_dict(transform_diagnostics), notes=notes)
    mf.write_manifest_json(updated, mp)
    return updated


def _maybe_apply_splits(dataset_root: Path, args: argparse.Namespace):
    if not _split_related_args_supplied(args):
        return None
    windows = sp.chronological_windows(train=_parse_us_range(args.split_train, "split_train"), val=_parse_us_range(args.split_val, "split_val"), test=None if args.split_test is None else _parse_us_range(args.split_test, "split_test"))
    cfgs = sp.SplitConfig(windows=windows, purge_before_us=args.purge_before_us, purge_after_us=args.purge_after_us, embargo_before_us=args.embargo_before_us, embargo_after_us=args.embargo_after_us, min_rows_per_split=args.min_rows_per_split, allow_empty_roles=False, validate_dataset_on_open=True)
    return sp.build_and_write_splits(str(dataset_root), cfgs, replace_existing=True)


def _split_related_args_supplied(args: argparse.Namespace) -> bool:
    return any(getattr(args, n) is not None for n in ("split_train", "split_val", "split_test", "purge_before_us", "purge_after_us", "embargo_before_us", "embargo_after_us"))


def _validate_split_args(args: argparse.Namespace) -> None:
    if not _split_related_args_supplied(args):
        return
    if args.split_train is None or args.split_val is None:
        raise ValueError("if any split args are supplied, both --split-train and --split-val are required")
    _parse_us_range(args.split_train, "split_train")
    _parse_us_range(args.split_val, "split_val")
    if args.split_test is not None:
        _parse_us_range(args.split_test, "split_test")
    for name in ("purge_before_us", "purge_after_us", "embargo_before_us", "embargo_after_us"):
        v = getattr(args, name)
        if v is not None:
            _require_nonnegative_int(v, name)


def _validate_output_dataset(dataset_root: Path) -> dict[str, int]:
    reader = rd.open_dataset(str(dataset_root), validate_on_open=True)
    reader.validate_dataset()
    if reader.total_rows <= 0:
        raise ValueError("dataset has no rows")
    return {"total_rows": reader.total_rows, "total_labels": reader.total_labels}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmrt.cli.ingest")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--book-csv", action="append", required=True)
    p.add_argument("--trades-csv", action="append", required=True)
    p.add_argument("--exchange", default=cfg.DEFAULT_EXCHANGE)
    p.add_argument("--symbol", default=cfg.DEFAULT_SYMBOL)
    p.add_argument("--book-data-type", default="book_snapshot_25")
    p.add_argument("--created-at-utc", default=None)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--event-batch-size", type=int, default=65536)
    p.add_argument("--chunk-rows", type=int, default=wr.DEFAULT_CHUNK_ROWS)
    p.add_argument("--row-group-rows", type=int, default=wr.DEFAULT_ROW_GROUP_ROWS)
    p.add_argument("--decision-stride-us", type=int, default=cfg.DEFAULT_DECISION_STRIDE_US)
    p.add_argument("--label-horizons-us", default=",".join(str(x) for x in cfg.DEFAULT_HORIZONS_US))
    p.add_argument("--label-entry-delay-us", type=int, default=cfg.DEFAULT_ENTRY_DELAY_US)
    p.add_argument("--validate-output", dest="validate_output", action="store_true", default=True)
    p.add_argument("--no-validate-output", dest="validate_output", action="store_false")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--split-train", default=None)
    p.add_argument("--split-val", default=None)
    p.add_argument("--split-test", default=None)
    p.add_argument("--purge-before-us", type=int, default=None)
    p.add_argument("--purge-after-us", type=int, default=None)
    p.add_argument("--embargo-before-us", type=int, default=None)
    p.add_argument("--embargo-after-us", type=int, default=None)
    p.add_argument("--min-rows-per-split", type=int, default=1)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.book_data_type != "book_snapshot_25":
        if args.book_data_type == "incremental_book_L2":
            raise ValueError("cli.ingest v1 supports only book_snapshot_25 book inputs; incremental_book_L2 reconstruction belongs in a later data-layer integration")
        raise ValueError("cli.ingest v1 supports only book_snapshot_25 book inputs")
    _require_positive_int(args.event_batch_size, "event_batch_size")
    _require_positive_int(args.chunk_rows, "chunk_rows")
    if args.decision_stride_us != cfg.DEFAULT_DECISION_STRIDE_US:
        raise ValueError("decision_stride_us must be 500_000 for cli.ingest v1")
    _require_positive_int(args.row_group_rows, "row_group_rows")
    _require_positive_int(args.decision_stride_us, "decision_stride_us")
    _require_nonnegative_int(args.label_entry_delay_us, "label_entry_delay_us")
    _require_positive_int(args.min_rows_per_split, "min_rows_per_split")
    label_horizons_us = _parse_csv_ints(args.label_horizons_us, "label_horizons_us")
    _validate_split_args(args)
    if args.created_at_utc is not None:
        _require_nonempty_str(args.created_at_utc, "created_at_utc")
    if args.max_events is not None:
        _require_positive_int(args.max_events, "max_events")
    market = bfa.validate_binance_futures_market(args.exchange, args.symbol)
    exchange = market.exchange
    symbol = market.symbol

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / mf.DEFAULT_MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    seg_dir = dataset_root / "segments"
    if any(seg_dir.glob("*.parquet")) if seg_dir.exists() else False:
        raise FileExistsError("existing parquet segments found")

    book_paths = _resolve_paths(args.book_csv, "book_csv")
    trade_paths = _resolve_paths(args.trades_csv, "trades_csv")
    work_dir = Path(args.work_dir) if args.work_dir is not None else dataset_root / "_ingest_work"
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"work_dir exists and is not empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = _build_pipeline_config(args, exchange=exchange, symbol=symbol, label_horizons_us=label_horizons_us)

    normalized_files = _normalize_input_files(book_paths, trade_paths, work_dir, exchange, symbol)
    merged_path = _write_merged_events(normalized_files, work_dir)

    writer_cfg = wr.WriterConfig(dataset_id=_require_nonempty_str(args.dataset_id, "dataset_id"), created_at_utc=args.created_at_utc or _utc_now_iso(), dataset_root=str(dataset_root), config=pipeline_config, chunk_rows=args.chunk_rows, row_group_rows=args.row_group_rows, transform_config=_transform_config_to_dict(TransformConfig()), transform_diagnostics={}, source_files=_safe_manifest_source_files(book_paths, trade_paths), notes={"cli": "mmrt.cli.ingest", "book_data_type": "book_snapshot_25", "trade_data_type": "trades"})

    with wr.DecisionRowWriter(writer_cfg) as writer:
        counters, tcfg, tdiag = _run_causal_ingest(merged_path, writer, pipeline_config, args.event_batch_size, args.max_events)
        manifest = writer.finalize()

    counters.input_book_files = len(book_paths)
    counters.input_trade_files = len(trade_paths)
    counters.normalized_files = len(normalized_files)
    counters.output_segments = len(manifest.segments)
    counters.output_rows = manifest.total_rows
    manifest = _patch_manifest_transform_metadata(dataset_root, tcfg, tdiag, counters)
    split_manifest = _maybe_apply_splits(dataset_root, args)
    if split_manifest is not None:
        manifest = split_manifest
    if args.validate_output:
        _validate_output_dataset(dataset_root)

    shutil.rmtree(work_dir)
    summary = {
        "status": "ok", "dataset_root": str(dataset_root), "dataset_id": manifest.dataset_id, "exchange": manifest.pipeline_config.market.exchange,
        "symbol": manifest.pipeline_config.market.symbol, "book_data_type": "book_snapshot_25", "trade_data_type": "trades",
        "manifest_path": str(dataset_root / mf.DEFAULT_MANIFEST_FILENAME), "segments": len(manifest.segments), "rows": manifest.total_rows,
        "decisions_emitted": counters.decisions_emitted, "rows_written": counters.rows_written, "pending_decisions_at_eof": counters.pending_decisions_at_eof,
        "splits_written": bool(manifest.splits), "split_roles": [s.role.value for s in manifest.splits], "work_dir": str(work_dir), "work_dir_removed": True,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
