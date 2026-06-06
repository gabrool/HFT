"""CLI for storage-backed MMRT preprocessing audits.

This command audits train-only z-score, variance-floor, and clipping behavior
for an already-written storage dataset. It does not ingest raw Tardis CSV,
compute features or labels, create splits, train models, or mutate storage.
"""

import argparse
import json
import math
from dataclasses import asdict

from mmrt.analysis.preprocess_audit import (
    DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS,
    PreprocessAuditConfig,
    _json_safe,
    run_preprocess_audit,
    write_preprocess_audit_artifacts,
)
from mmrt.linear import extractors as ex
from mmrt.linear import preprocess as pp
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
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite float")
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
        default=DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS,
    )
    parser.add_argument("--clip-z", type=_positive_float, default=pp.DEFAULT_CLIP_Z)
    parser.add_argument("--variance-floor", type=_positive_float, default=pp.DEFAULT_VARIANCE_FLOOR)
    parser.add_argument("--extractor-dtype", choices=ex.ALLOWED_EXTRACTOR_DTYPES, default="float32")
    parser.add_argument("--preprocess-dtype", choices=pp.ALLOWED_PREPROCESS_DTYPES, default="float32")
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--no-validate-on-open", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        feature_columns = _parse_feature_columns(args.feature_columns)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    config = PreprocessAuditConfig(
        batch_size=args.batch_size,
        validate_dataset_on_open=(not args.no_validate_on_open),
        max_sample_rows_per_split=args.max_sample_rows_per_split,
        extractor_config=ex.LinearFeatureExtractorConfig(
            feature_columns=feature_columns,
            output_dtype=args.extractor_dtype,
        ),
        preprocess_config=pp.LinearPreprocessConfig(
            variance_floor=args.variance_floor,
            clip_z=args.clip_z,
            output_dtype=args.preprocess_dtype,
        ),
    )

    result = run_preprocess_audit(args.dataset_root, config=config)
    paths = write_preprocess_audit_artifacts(result, args.output_dir)

    payload = {
        "status": "ok",
        "summary_json": paths["summary_json"],
        "features_csv": paths["features_csv"],
        "warnings": list(result.warnings),
        "splits": {key: asdict(value) for key, value in result.splits.items()},
    }
    print(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


__all__ = ["build_arg_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
