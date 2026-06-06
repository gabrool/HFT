"""CLI for storage-backed MMRT Feature Lab candidate analysis.

This command evaluates externally generated candidate feature Parquet files
against existing storage rows and a trained linear artifact. It writes compact
analysis artifacts only; it does not ingest raw data, compute production
features, create splits, train models, retrain candidates, use held-out rows, or
mutate storage.
"""

import argparse
import json

from mmrt.analysis import feature_lab


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
    parser.add_argument("--candidate-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=feature_lab.DEFAULT_FEATURE_LAB_BATCH_SIZE)
    parser.add_argument("--max-sample-rows-train", type=_nonnegative_int, default=feature_lab.DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN)
    parser.add_argument("--max-sample-rows-val", type=_nonnegative_int, default=feature_lab.DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL)
    parser.add_argument("--seed", type=_nonnegative_int, default=feature_lab.DEFAULT_FEATURE_LAB_SEED)
    parser.add_argument("--no-validate-on-open", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> feature_lab.FeatureLabConfig:
    return feature_lab.FeatureLabConfig(
        batch_size=args.batch_size,
        validate_dataset_on_open=not args.no_validate_on_open,
        max_sample_rows_train=args.max_sample_rows_train,
        max_sample_rows_val=args.max_sample_rows_val,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    result = feature_lab.run_feature_lab(
        args.dataset_root,
        args.train_result_json,
        args.candidate_features,
        config=config,
    )
    paths = feature_lab.write_feature_lab_artifacts(result, args.output_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_root": args.dataset_root,
                "train_result_json": args.train_result_json,
                "candidate_features": args.candidate_features,
                "output_dir": args.output_dir,
                **paths,
                "dataset_id": result.dataset_id,
                "manifest_hash": result.manifest_hash,
                "n_candidates": result.n_candidates,
                "train_sample_rows": result.train_sample_rows,
                "val_sample_rows": result.val_sample_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan = True,
        )
    )
    return 0


__all__ = ["build_arg_parser", "config_from_args", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
