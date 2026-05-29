import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mmrt.cli.audit_feature_importance as cli
from mmrt.analysis import feature_importance as fi


def test_public_api_boundary():
    assert cli.__all__ == ["build_arg_parser", "config_from_args", "main"]


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--dataset-root", "d", "--train-result-json", "t.json", "--output-dir", "o"])
    assert args.batch_size == fi.DEFAULT_FEATURE_IMPORTANCE_BATCH_SIZE
    assert args.max_sample_rows == fi.DEFAULT_FEATURE_IMPORTANCE_MAX_SAMPLE_ROWS
    assert args.seed == fi.DEFAULT_FEATURE_IMPORTANCE_SEED
    assert args.no_validate_on_open is False


def test_config_from_args():
    args = SimpleNamespace(batch_size=7, max_sample_rows=9, seed=3, no_validate_on_open=True)
    cfg = cli.config_from_args(args)
    assert cfg.batch_size == 7
    assert cfg.max_sample_rows == 9
    assert cfg.seed == 3
    assert cfg.validate_dataset_on_open is False


def test_parser_rejects_bad_numeric_values():
    parser = cli.build_arg_parser()
    base = ["--dataset-root", "d", "--train-result-json", "t", "--output-dir", "o"]
    for extra in [["--batch-size", "0"], ["--max-sample-rows", "-1"], ["--seed", "-1"]]:
        with pytest.raises(SystemExit):
            parser.parse_args(base + extra)


def test_main_calls_analysis_and_writer_and_prints_compact_json(monkeypatch, tmp_path: Path, capsys):
    calls = {}
    result = fi.FeatureImportanceResult(
        schema_version=1,
        dataset_id="d1",
        manifest_hash="h1",
        train_result_path="train.json",
        selection_split="val",
        n_sample_rows=12,
        seed=17,
        records=(),
        family_records=(),
        summary={"schema_version": 1},
    )

    def fake_run(dataset_root, train_result_json, *, config=None):
        calls["run"] = (dataset_root, train_result_json, config)
        return result

    def fake_write(res, output_dir):
        calls["write"] = (res, output_dir)
        return {"summary_json": "s.json", "by_head_csv": "b.csv", "family_summary_csv": "f.csv"}

    monkeypatch.setattr(cli.fi, "run_feature_importance", fake_run)
    monkeypatch.setattr(cli.fi, "write_feature_importance_artifacts", fake_write)
    rc = cli.main([
        "--dataset-root",
        "ds",
        "--train-result-json",
        "train.json",
        "--output-dir",
        str(tmp_path),
        "--batch-size",
        "5",
        "--max-sample-rows",
        "6",
        "--seed",
        "7",
        "--no-validate-on-open",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "ok",
        "dataset_root": "ds",
        "train_result_json": "train.json",
        "output_dir": str(tmp_path),
        "summary_json": "s.json",
        "by_head_csv": "b.csv",
        "family_summary_csv": "f.csv",
        "dataset_id": "d1",
        "manifest_hash": "h1",
        "selection_split": "val",
        "n_sample_rows": 12,
    }
    assert calls["run"][2] == fi.FeatureImportanceConfig(batch_size=5, validate_dataset_on_open=False, max_sample_rows=6, seed=7)
    assert calls["write"] == (result, str(tmp_path))


def test_no_bad_imports():
    src = inspect.getsource(cli)
    for bad in ["import pan" + "das", "import sk" + "learn", "import sci" + "py", "import to" + "rch", "multiprocessing"]:
        assert bad not in src


def test_no_raw_data_ingest_or_split_surface():
    src = inspect.getsource(cli).lower()
    for bad in ["read_csv", "splitconfig", "feature-columns", "--split", "--heads", "--n-jobs"]:
        assert bad not in src


def test_no_future_leakage_surface():
    src = inspect.getsource(cli)
    for bad in ["--split", "test_split", "permutation_repeats", "future"]:
        assert bad not in src
