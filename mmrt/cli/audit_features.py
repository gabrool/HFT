"""CLI for storage-backed MMRT feature health and redundancy audits.

This command audits already-written storage feature columns for health,
train-only redundancy/correlation, and train-vs-val/test distribution drift.
It does not ingest raw Tardis CSV, compute features or labels, create splits,
train models, select model features, or mutate storage.
"""

import argparse
from dataclasses import asdict
import json
import math

from mmrt.analysis.feature_audit import (
    DEFAULT_DRIFT_MEAN_Z_THRESHOLD,
    DEFAULT_DRIFT_STD_RATIO_HIGH,
    DEFAULT_DRIFT_STD_RATIO_LOW,
    DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS,
    DEFAULT_HIGH_CORR_THRESHOLD,
    DEFAULT_LOW_VARIANCE_STD_THRESHOLD,
    DEFAULT_MAX_CORR_PAIRS,
    DEFAULT_MIN_CORR_OUTPUT_THRESHOLD,
    FeatureAuditConfig,
    run_feature_audit,
    write_feature_audit_artifacts,
)
from mmrt.linear import extractors as ex
from mmrt.storage import reader as rd


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive int")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative int")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite float")
    return value


def _corr_threshold_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise argparse.ArgumentTypeError("value must be finite and between 0 and 1")
    return value


def _ratio_low_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise argparse.ArgumentTypeError("value must be finite and satisfy 0 < value < 1")
    return value


def _ratio_high_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and > 1")
    return value


def _parse_feature_columns(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    cols = tuple(part.strip() for part in text.split(",") if part.strip())
    if not cols:
        raise argparse.ArgumentTypeError("--feature-columns must contain at least one column")
    if len(set(cols)) != len(cols):
        raise argparse.ArgumentTypeError("--feature-columns must not contain duplicates")
    return cols


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=rd.DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-sample-rows-per-split",
        type=_nonnegative_int,
        default=DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS,
    )
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument(
        "--extractor-dtype",
        choices=ex.ALLOWED_EXTRACTOR_DTYPES,
        default="float32",
    )
    parser.add_argument(
        "--low-variance-std-threshold",
        type=_positive_float,
        default=DEFAULT_LOW_VARIANCE_STD_THRESHOLD,
    )
    parser.add_argument(
        "--high-corr-threshold",
        type=_corr_threshold_float,
        default=DEFAULT_HIGH_CORR_THRESHOLD,
    )
    parser.add_argument(
        "--min-corr-output-threshold",
        type=_corr_threshold_float,
        default=DEFAULT_MIN_CORR_OUTPUT_THRESHOLD,
    )
    parser.add_argument("--max-corr-pairs", type=_positive_int, default=DEFAULT_MAX_CORR_PAIRS)
    parser.add_argument(
        "--drift-mean-z-threshold",
        type=_positive_float,
        default=DEFAULT_DRIFT_MEAN_Z_THRESHOLD,
    )
    parser.add_argument(
        "--drift-std-ratio-low",
        type=_ratio_low_float,
        default=DEFAULT_DRIFT_STD_RATIO_LOW,
    )
    parser.add_argument(
        "--drift-std-ratio-high",
        type=_ratio_high_float,
        default=DEFAULT_DRIFT_STD_RATIO_HIGH,
    )
    parser.add_argument("--no-validate-on-open", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cols = _parse_feature_columns(args.feature_columns)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    config = FeatureAuditConfig(
        batch_size=args.batch_size,
        validate_dataset_on_open=(not args.no_validate_on_open),
        max_sample_rows_per_split=args.max_sample_rows_per_split,
        feature_columns=cols,
        extractor_dtype=args.extractor_dtype,
        low_variance_std_threshold=args.low_variance_std_threshold,
        high_corr_threshold=args.high_corr_threshold,
        min_corr_output_threshold=args.min_corr_output_threshold,
        max_corr_pairs=args.max_corr_pairs,
        drift_mean_z_threshold=args.drift_mean_z_threshold,
        drift_std_ratio_low=args.drift_std_ratio_low,
        drift_std_ratio_high=args.drift_std_ratio_high,
    )
    result = run_feature_audit(args.dataset_root, config=config)
    paths = write_feature_audit_artifacts(result, args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                **paths,
                "warnings": list(result.warnings),
                "splits": {k: asdict(v) for k, v in result.splits.items()},
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
    )
    return 0


__all__ = ["build_arg_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
