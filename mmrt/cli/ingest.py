"""Build the supervised decision-row dataset from an execution tape.

This CLI replays an already-built execution tape (incremental_book_L2 +
trades) through the shared decision feature pipeline, matures fixed-horizon
mid-return labels on the tape's local clock, and writes the storage dataset
plus optional chronological splits. Because the same replay path also builds
execution linear signals, training features and serving features are
identical by construction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from mmrt import config as cfg
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.feature_replay import iter_tape_feature_steps
from mmrt.features.labels import LabelBuilder
from mmrt.features.pipeline import DecisionFeaturePipeline, FeaturePipelineConfig
from mmrt.features.transforms import TransformConfig, TransformDiagnostics
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


def _safe_posix_leaf(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "tape"


def _build_pipeline_config(args: argparse.Namespace, *, exchange: str, symbol: str, label_horizons_us: tuple[int, ...]) -> cfg.PipelineConfig:
    base = cfg.default_config()
    return cfg.PipelineConfig(
        market=cfg.MarketConfig(exchange=exchange, symbol=symbol),
        data=cfg.DataConfig(),
        decision=cfg.DecisionConfig(policy=base.decision.policy, reason=base.decision.reason, stride_us=args.decision_stride_us),
        labels=cfg.LabelConfig(horizons_us=label_horizons_us, entry_delay_us=args.label_entry_delay_us),
        runtime=base.runtime,
        storage=base.storage,
    )


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
    tape_events: int = 0
    tape_l2_batches: int = 0
    tape_trades: int = 0
    l2_steps_seen: int = 0
    decisions_emitted: int = 0
    labels_matured: int = 0
    rows_written: int = 0
    pending_decisions_at_eof: int = 0
    transform_rows_seen: int = 0
    output_segments: int = 0
    output_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _write_matured_labels(label_results, pending_decisions: dict[tuple[int, int], PendingDecision], writer: wr.DecisionRowWriter, counters: IngestCounters) -> None:
    for label in label_results:
        key = (int(label.decision_ts_us), int(label.decision_event_seq))
        if key not in pending_decisions:
            raise KeyError(f"missing pending decision for {key}")
        p = pending_decisions.pop(key)
        writer.append_values(decision_index=p.decision_index, ts_us=p.ts_us, local_ts_us=p.local_ts_us, event_seq=p.event_seq, raw_mid=p.raw_mid, label_entry_ts_us=int(label.entry_ts_us), label_values=label.values_bps, feature_values=p.feature_values)
        counters.labels_matured += 1
        counters.rows_written += 1


def _run_tape_ingest(
    tape,
    writer: wr.DecisionRowWriter,
    pipeline_config: cfg.PipelineConfig,
    *,
    start_event_index: int,
    max_events: int | None,
) -> tuple[IngestCounters, TransformConfig, TransformDiagnostics]:
    counters = IngestCounters()
    counters.tape_events = int(tape.manifest.num_events)
    counters.tape_l2_batches = int(tape.manifest.num_l2_batches)
    counters.tape_trades = int(tape.manifest.num_trades)
    feature_pipeline_config = FeaturePipelineConfig(decision_stride_us=pipeline_config.decision.stride_us)
    pipeline = DecisionFeaturePipeline(feature_pipeline_config)
    label_builder = LabelBuilder(pipeline_config.label_spec)
    pending_decisions: dict[tuple[int, int], PendingDecision] = {}

    for step in iter_tape_feature_steps(
        tape,
        pipeline=pipeline,
        start_event_index=start_event_index,
        max_events=max_events,
    ):
        counters.l2_steps_seen += 1
        # Causality contract: observe the current book mid first, mature older
        # labels, then register the decision emitted at this same event.
        _write_matured_labels(label_builder.observe_price_local(step.local_ts_us, step.event_seq, step.mid), pending_decisions, writer, counters)
        decision = step.decision
        if decision is not None:
            label_builder.on_decision_local(decision.local_ts_us, decision.event_seq)
            pending_decisions[(decision.local_ts_us, decision.event_seq)] = PendingDecision(
                decision_index=decision.decision_index,
                ts_us=decision.ts_us,
                local_ts_us=decision.local_ts_us,
                event_seq=decision.event_seq,
                raw_mid=decision.raw_mid,
                feature_values=tuple(float(x) for x in decision.feature_values),
            )
            counters.decisions_emitted += 1

    _write_matured_labels(label_builder.finalize_at_eof(), pending_decisions, writer, counters)
    counters.pending_decisions_at_eof = len(pending_decisions)
    diagnostics = pipeline.transform_diagnostics_snapshot()
    counters.transform_rows_seen = diagnostics.rows_seen

    if counters.l2_steps_seen == 0:
        raise ValueError("no valid two-sided L2 events seen")
    if counters.decisions_emitted == 0:
        raise ValueError("no decisions emitted")
    if counters.rows_written == 0:
        raise ValueError("no matured rows written")
    return counters, feature_pipeline_config.transform, diagnostics


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
    p = argparse.ArgumentParser(prog="mmrt.cli.ingest", description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--tape-root", required=True)
    p.add_argument("--created-at-utc", default=None)
    p.add_argument("--chunk-rows", type=int, default=wr.DEFAULT_CHUNK_ROWS)
    p.add_argument("--row-group-rows", type=int, default=wr.DEFAULT_ROW_GROUP_ROWS)
    p.add_argument("--decision-stride-us", type=int, default=cfg.DEFAULT_DECISION_STRIDE_US)
    p.add_argument("--label-horizons-us", default=",".join(str(x) for x in cfg.DEFAULT_HORIZONS_US))
    p.add_argument("--label-entry-delay-us", type=int, default=cfg.DEFAULT_ENTRY_DELAY_US)
    p.add_argument("--start-event-index", type=int, default=0)
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
    _require_positive_int(args.chunk_rows, "chunk_rows")
    if args.decision_stride_us != cfg.DEFAULT_DECISION_STRIDE_US:
        raise ValueError("decision_stride_us must be 500_000 for cli.ingest")
    _require_positive_int(args.row_group_rows, "row_group_rows")
    _require_positive_int(args.decision_stride_us, "decision_stride_us")
    _require_nonnegative_int(args.label_entry_delay_us, "label_entry_delay_us")
    _require_positive_int(args.min_rows_per_split, "min_rows_per_split")
    _require_nonnegative_int(args.start_event_index, "start_event_index")
    label_horizons_us = _parse_csv_ints(args.label_horizons_us, "label_horizons_us")
    _validate_split_args(args)
    if args.created_at_utc is not None:
        _require_nonempty_str(args.created_at_utc, "created_at_utc")
    if args.max_events is not None:
        _require_positive_int(args.max_events, "max_events")

    tape_root = Path(_require_nonempty_str(args.tape_root, "tape_root"))
    tape = load_execution_tape(
        str(tape_root),
        mmap_mode="r",
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    exchange = tape.manifest.exchange
    symbol = tape.manifest.symbol

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / mf.DEFAULT_MANIFEST_FILENAME
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    seg_dir = dataset_root / "segments"
    if any(seg_dir.glob("*.parquet")) if seg_dir.exists() else False:
        raise FileExistsError("existing parquet segments found")

    pipeline_config = _build_pipeline_config(args, exchange=exchange, symbol=symbol, label_horizons_us=label_horizons_us)

    writer_cfg = wr.WriterConfig(
        dataset_id=_require_nonempty_str(args.dataset_id, "dataset_id"),
        created_at_utc=args.created_at_utc or _utc_now_iso(),
        dataset_root=str(dataset_root),
        config=pipeline_config,
        chunk_rows=args.chunk_rows,
        row_group_rows=args.row_group_rows,
        transform_config=_transform_config_to_dict(TransformConfig()),
        transform_diagnostics={},
        source_files=(f"execution_tape/{_safe_posix_leaf(tape_root.name)}/manifest.json",),
        notes={
            "cli": "mmrt.cli.ingest",
            "source": "execution_tape",
            "tape_root": str(tape_root),
            "tape_schema": tape.manifest.schema,
            "book_data_type": "incremental_book_L2",
            "trade_data_type": "trades",
        },
    )

    with wr.DecisionRowWriter(writer_cfg) as writer:
        counters, tcfg, tdiag = _run_tape_ingest(
            tape,
            writer,
            pipeline_config,
            start_event_index=args.start_event_index,
            max_events=args.max_events,
        )
        manifest = writer.finalize()

    counters.output_segments = len(manifest.segments)
    counters.output_rows = manifest.total_rows
    manifest = _patch_manifest_transform_metadata(dataset_root, tcfg, tdiag, counters)
    split_manifest = _maybe_apply_splits(dataset_root, args)
    if split_manifest is not None:
        manifest = split_manifest
    _validate_output_dataset(dataset_root)

    summary = {
        "status": "ok", "dataset_root": str(dataset_root), "dataset_id": manifest.dataset_id, "exchange": manifest.exchange,
        "symbol": manifest.symbol, "tape_root": str(tape_root), "tape_schema": tape.manifest.schema,
        "book_data_type": "incremental_book_L2", "trade_data_type": "trades",
        "manifest_path": str(dataset_root / mf.DEFAULT_MANIFEST_FILENAME), "segments": len(manifest.segments), "rows": manifest.total_rows,
        "decisions_emitted": counters.decisions_emitted, "rows_written": counters.rows_written, "pending_decisions_at_eof": counters.pending_decisions_at_eof,
        "splits_written": bool(manifest.splits), "split_roles": [s.role.value for s in manifest.splits],
        "start_event_index": args.start_event_index,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main"]
