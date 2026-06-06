"""CLI for storage-backed MMRT linear feature importance audits.

This command ranks already-materialized storage features for each trained
linear head using an existing training artifact and validation split only. It
does not ingest raw Tardis CSV, compute features or labels, create splits,
train models, alter head feature subsets, or mutate storage.
"""

import argparse
import json

from mmrt.analysis import feature_importance as fi


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-result-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=fi.DEFAULT_FEATURE_IMPORTANCE_BATCH_SIZE)
    parser.add_argument("--max-sample-rows", type=_nonnegative_int, default=fi.DEFAULT_FEATURE_IMPORTANCE_MAX_SAMPLE_ROWS)
    parser.add_argument("--seed", type=_nonnegative_int, default=fi.DEFAULT_FEATURE_IMPORTANCE_SEED)
    parser.add_argument("--no-validate-on-open", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> fi.FeatureImportanceConfig:
    return fi.FeatureImportanceConfig(
        batch_size=args.batch_size,
        validate_dataset_on_open=not args.no_validate_on_open,
        max_sample_rows=args.max_sample_rows,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    result = fi.run_feature_importance(args.dataset_root, args.train_result_json, config=config)
    paths = fi.write_feature_importance_artifacts(result, args.output_dir)
    print(
        json.dumps(
            fi._json_safe({
                "status": "ok",
                "dataset_root": args.dataset_root,
                "train_result_json": args.train_result_json,
                "output_dir": args.output_dir,
                **paths,
                "dataset_id": result.dataset_id,
                "manifest_hash": result.manifest_hash,
                "selection_split": result.selection_split,
                "n_sample_rows": result.n_sample_rows,
            }),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


__all__ = ["build_arg_parser", "config_from_args", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
