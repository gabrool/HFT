"""CLI wrapper for storage-backed MMRT linear training.

This command trains from an existing storage dataset with precomputed
features, labels, and manifest splits. It does not ingest raw market data,
create splits, recompute labels, inspect row timing fields, or mutate the
dataset manifest.
"""

import argparse
import json
import math
from typing import Sequence

import mmrt.linear.diagnostics as dg
import mmrt.linear.head_feature_presets as hp
import mmrt.linear.models as lm
import mmrt.linear.preprocess as pp
import mmrt.linear.targets as tg
import mmrt.linear.train as lt

__all__ = [
    "build_arg_parser",
    "main",
]


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be a positive finite float")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be a nonnegative finite float")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrt-train-linear",
        description="Train MMRT linear models from an existing storage dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to an existing MMRT storage dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the linear training JSON result will be written.",
    )
    parser.add_argument(
        "--result-filename",
        default=lt.DEFAULT_OUTPUT_FILENAME,
        help="Output JSON filename. Must end with .json.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=lt.DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--epochs", type=_positive_int, default=lt.DEFAULT_EPOCHS)
    parser.add_argument(
        "--no-validate-on-open",
        action="store_true",
        help="Skip reader validation when opening the dataset.",
    )
    parser.add_argument(
        "--head-feature-preset",
        choices=hp.AVAILABLE_HEAD_FEATURE_PRESETS,
        default=hp.ALL_FEATURES_PRESET,
        help=(
            "Named per-head feature subset preset. "
            "'all' uses every manifest feature for every head."
        ),
    )

    parser.add_argument("--target-horizon-us", type=_positive_int, default=tg.DEFAULT_TARGET_HORIZON_US)
    parser.add_argument(
        "--move-deadband-bps",
        type=_nonnegative_float,
        default=tg.DEFAULT_MOVE_DEADBAND_BPS,
    )
    parser.add_argument(
        "--target-output-dtype",
        choices=("float32", "float64"),
        default=tg.DEFAULT_TARGET_DTYPE,
    )

    parser.add_argument("--variance-floor", type=_positive_float, default=pp.DEFAULT_VARIANCE_FLOOR)
    parser.add_argument("--clip-z", type=_positive_float, default=pp.DEFAULT_CLIP_Z)
    parser.add_argument(
        "--preprocess-output-dtype",
        choices=("float32", "float64"),
        default=pp.DEFAULT_PREPROCESS_DTYPE,
    )

    parser.add_argument("--learning-rate", type=_positive_float, default=lm.DEFAULT_LEARNING_RATE)
    parser.add_argument("--l2", type=_nonnegative_float, default=lm.DEFAULT_L2)
    parser.add_argument("--max-grad-norm", type=_positive_float, default=lm.DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--model-output-dtype",
        choices=("float32", "float64"),
        default=lm.DEFAULT_MODEL_DTYPE,
    )

    parser.add_argument("--diagnostics-top-k", type=_positive_int, default=dg.DEFAULT_TOP_K)
    parser.add_argument("--diagnostics-num-bins", type=_positive_int, default=dg.DEFAULT_NUM_BINS)
    parser.add_argument("--diagnostics-max-rows", type=_positive_int, default=dg.DEFAULT_MAX_ROWS)
    return parser


def _config_from_args(args: argparse.Namespace) -> lt.LinearTrainConfig:
    return lt.LinearTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        validate_dataset_on_open=not args.no_validate_on_open,
        head_feature_config=hp.head_feature_config_for_preset(args.head_feature_preset),
        target_config=tg.LinearTargetConfig(
            target_horizon_us=args.target_horizon_us,
            move_deadband_bps=args.move_deadband_bps,
            output_dtype=args.target_output_dtype,
        ),
        preprocess_config=pp.LinearPreprocessConfig(
            variance_floor=args.variance_floor,
            clip_z=args.clip_z,
            output_dtype=args.preprocess_output_dtype,
        ),
        model_config=lm.LinearModelConfig(
            learning_rate=args.learning_rate,
            l2=args.l2,
            max_grad_norm=args.max_grad_norm,
            output_dtype=args.model_output_dtype,
        ),
        diagnostics_config=dg.DiagnosticsConfig(
            top_k=args.diagnostics_top_k,
            num_bins=args.diagnostics_num_bins,
            max_rows=args.diagnostics_max_rows,
        ),
    )


def _summary_from_result(
    *,
    result: lt.LinearTrainResult,
    paths: dict[str, str],
    dataset_root: str,
    output_dir: str,
) -> dict[str, object]:
    return {
        "status": "ok",
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "result_json": paths["result_json"],
        "schema_version": result.schema_version,
        "dataset_id": result.dataset_id,
        "manifest_hash": result.manifest_hash,
        "splits": {
            role: {"n_rows": split.n_rows}
            for role, split in result.splits.items()
        },
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    result = lt.train_linear_model(args.dataset_root, config=config)
    paths = lt.write_linear_train_artifacts(
        result,
        args.output_dir,
        filename=args.result_filename,
    )
    summary = _summary_from_result(
        result=result,
        paths=paths,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )
    _print_json(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
