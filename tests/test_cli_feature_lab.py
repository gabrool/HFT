import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmrt.analysis import feature_lab
from mmrt.cli import feature_lab as cli


def test_public_api_boundary():
    assert cli.__all__ == ["build_arg_parser", "config_from_args", "main"]


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args([
        "--dataset-root",
        "ds",
        "--train-result-json",
        "train.json",
        "--candidate-features",
        "c.parquet",
        "--output-dir",
        "out",
    ])
    assert args.batch_size == feature_lab.DEFAULT_FEATURE_LAB_BATCH_SIZE
    assert args.max_sample_rows_train == feature_lab.DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN
    assert args.max_sample_rows_val == feature_lab.DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL
    assert args.seed == feature_lab.DEFAULT_FEATURE_LAB_SEED
    assert args.no_validate_on_open is False


def test_config_from_args():
    args = SimpleNamespace(batch_size=7, max_sample_rows_train=8, max_sample_rows_val=9, seed=3, no_validate_on_open=True)
    cfg = cli.config_from_args(args)
    assert cfg == feature_lab.FeatureLabConfig(batch_size=7, validate_dataset_on_open=False, max_sample_rows_train=8, max_sample_rows_val=9, seed=3)


def test_parser_rejects_bad_numeric_values():
    parser = cli.build_arg_parser()
    base = ["--dataset-root", "d", "--train-result-json", "t", "--candidate-features", "c.parquet", "--output-dir", "o"]
    for extra in [["--batch-size", "0"], ["--max-sample-rows-train", "-1"], ["--max-sample-rows-val", "-1"], ["--seed", "-1"]]:
        with pytest.raises(SystemExit):
            parser.parse_args(base + extra)


def test_main_calls_analysis_and_writer_and_prints_compact_json(monkeypatch, tmp_path: Path, capsys):
    calls = {}
    result = feature_lab.FeatureLabResult(
        schema_version=1,
        dataset_id="d1",
        manifest_hash="h1",
        train_result_path="train.json",
        candidate_features_path="c.parquet",
        n_candidates=3,
        train_sample_rows=11,
        val_sample_rows=12,
        config={},
        health_records=(),
        existing_correlation_records=(),
        redundancy_records=(),
        head_metric_records=(),
        recommendation_records=(),
        summary={"schema_version": 1},
        warnings=(),
    )

    def fake_run(dataset_root, train_result_json, candidate_features_path, *, config=None):
        calls["run"] = (dataset_root, train_result_json, candidate_features_path, config)
        return result

    def fake_write(res, output_dir):
        calls["write"] = (res, output_dir)
        return {
            "summary_json": "s.json",
            "candidate_health_csv": "h.csv",
            "candidate_existing_correlations_csv": "c.csv",
            "candidate_redundancy_summary_csv": "r.csv",
            "candidate_head_metrics_csv": "m.csv",
            "candidate_recommendations_csv": "rec.csv",
        }

    monkeypatch.setattr(cli.feature_lab, "run_feature_lab", fake_run)
    monkeypatch.setattr(cli.feature_lab, "write_feature_lab_artifacts", fake_write)
    rc = cli.main([
        "--dataset-root",
        "ds",
        "--train-result-json",
        "train.json",
        "--candidate-features",
        "cand.parquet",
        "--output-dir",
        str(tmp_path),
        "--batch-size",
        "5",
        "--max-sample-rows-train",
        "6",
        "--max-sample-rows-val",
        "7",
        "--seed",
        "8",
        "--no-validate-on-open",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "ok",
        "dataset_root": "ds",
        "train_result_json": "train.json",
        "candidate_features": "cand.parquet",
        "output_dir": str(tmp_path),
        "summary_json": "s.json",
        "candidate_health_csv": "h.csv",
        "candidate_existing_correlations_csv": "c.csv",
        "candidate_redundancy_summary_csv": "r.csv",
        "candidate_head_metrics_csv": "m.csv",
        "candidate_recommendations_csv": "rec.csv",
        "dataset_id": "d1",
        "manifest_hash": "h1",
        "n_candidates": 3,
        "train_sample_rows": 11,
        "val_sample_rows": 12,
    }
    assert calls["run"] == ("ds", "train.json", "cand.parquet", feature_lab.FeatureLabConfig(batch_size=5, validate_dataset_on_open=False, max_sample_rows_train=6, max_sample_rows_val=7, seed=8))
    assert calls["write"] == (result, str(tmp_path))


def test_no_bad_imports():
    src = inspect.getsource(cli)
    for bad in ["import pan" + "das", "import pol" + "ars", "import sk" + "learn", "import sci" + "py", "import to" + "rch", "import num" + "ba", "import job" + "lib", "multiprocessing"]:
        assert bad not in src


def test_no_raw_data_ingest_or_training_surface():
    src = inspect.getsource(cli).lower()
    for bad in ["read_csv", "tardis", "splitconfig", "feature-columns", "--heads", "--metrics", "--n-jobs"]:
        assert bad not in src


def test_no_test_split_or_retrain_flags():
    parser_text = "\n".join(action.option_strings[0] for action in cli.build_arg_parser()._actions if action.option_strings)
    for bad in ["--split", "--test", "--retrain", "--csv", "--top-k", "--candidate-prefix"]:
        assert bad not in parser_text
