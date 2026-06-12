"""Audit execution-tape linear features against a trained linear model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mmrt.execution.linear_feature_audit import LinearExecutionFeatureAuditConfig, audit_linear_execution_features_from_config


def _parse_thresholds(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--linear-train-result-json", required=True)
    parser.add_argument("--linear-signals-npz")
    parser.add_argument("--output-json")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--decision-interval-us", type=int, default=500_000)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--z-thresholds", default="3,5,8")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--work-dir")
    parser.add_argument("--cleanup-work-dir", dest="cleanup_work_dir", action="store_true", default=True)
    parser.add_argument("--keep-work-dir", dest="cleanup_work_dir", action="store_false")
    parser.add_argument("--quantile-mode", choices=("exact_memmap", "reservoir"), default="exact_memmap")
    parser.add_argument("--max-quantile-samples", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> LinearExecutionFeatureAuditConfig:
    tape_root = Path(args.tape_root)
    signals = args.linear_signals_npz
    default_signals = tape_root / "linear_signals.npz"
    if signals is None and default_signals.exists():
        signals = str(default_signals)
    output = args.output_json if args.output_json is not None else str(tape_root / "linear_execution_feature_audit.json")
    if Path(output).exists() and not args.overwrite:
        raise FileExistsError(f"output_json already exists: {output}")
    return LinearExecutionFeatureAuditConfig(
        tape_root=args.tape_root,
        linear_train_result_json=args.linear_train_result_json,
        linear_signals_npz=signals,
        output_json=output,
        mmap_mode=None if args.no_mmap else "r",
        decision_interval_us=args.decision_interval_us,
        start_event_index=args.start_event_index,
        max_decisions=args.max_decisions,
        chunk_rows=args.chunk_rows,
        z_thresholds=_parse_thresholds(args.z_thresholds),
        top_k=args.top_k,
        work_dir=args.work_dir,
        cleanup_work_dir=args.cleanup_work_dir,
        quantile_mode=args.quantile_mode,
        max_quantile_samples=args.max_quantile_samples,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = audit_linear_execution_features_from_config(_config_from_args(args))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
