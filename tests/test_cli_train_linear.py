import inspect
import json
import subprocess
import sys

import pytest

import mmrt.cli.train_linear as cli
import mmrt.linear.diagnostics as dg
import mmrt.linear.models as lm
import mmrt.linear.preprocess as pp
import mmrt.linear.targets as tg
import mmrt.linear.train as lt


def test_public_api_boundary() -> None:
    assert cli.__all__ == ["build_arg_parser", "main"]


def test_build_arg_parser_defaults() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--dataset-root", "ds", "--output-dir", "out"])
    assert args.batch_size == lt.DEFAULT_TRAIN_BATCH_SIZE
    assert args.epochs == lt.DEFAULT_EPOCHS
    assert args.result_filename == lt.DEFAULT_OUTPUT_FILENAME
    assert args.no_validate_on_open is False
    assert args.target_horizon_us == tg.DEFAULT_TARGET_HORIZON_US
    assert args.direction_deadband_bps == tg.DEFAULT_DIRECTION_DEADBAND_BPS
    assert args.target_output_dtype == tg.DEFAULT_TARGET_DTYPE
    assert args.variance_floor == pp.DEFAULT_VARIANCE_FLOOR
    assert args.clip_z == pp.DEFAULT_CLIP_Z
    assert args.preprocess_output_dtype == pp.DEFAULT_PREPROCESS_DTYPE
    assert args.learning_rate == lm.DEFAULT_LEARNING_RATE
    assert args.l2 == lm.DEFAULT_L2
    assert args.max_grad_norm == lm.DEFAULT_MAX_GRAD_NORM
    assert args.model_output_dtype == lm.DEFAULT_MODEL_DTYPE
    assert args.diagnostics_top_k == dg.DEFAULT_TOP_K
    assert args.diagnostics_num_bins == dg.DEFAULT_NUM_BINS
    assert args.diagnostics_max_rows == dg.DEFAULT_MAX_ROWS


def test_config_from_args_constructs_frozen_config() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args(
        [
            "--dataset-root", "ds", "--output-dir", "out", "--batch-size", "7", "--epochs", "3", "--no-validate-on-open",
            "--target-horizon-us", "500000", "--direction-deadband-bps", "0.25", "--target-output-dtype", "float64",
            "--variance-floor", "1e-8", "--clip-z", "5.0", "--preprocess-output-dtype", "float64",
            "--learning-rate", "0.01", "--l2", "0.001", "--max-grad-norm", "10.0", "--model-output-dtype", "float64",
            "--diagnostics-top-k", "5", "--diagnostics-num-bins", "4", "--diagnostics-max-rows", "123",
        ]
    )
    cfg = cli._config_from_args(args)
    assert isinstance(cfg, lt.LinearTrainConfig)
    assert cfg.batch_size == 7
    assert cfg.epochs == 3
    assert cfg.validate_dataset_on_open is False
    assert cfg.target_config.target_horizon_us == 500000
    assert cfg.target_config.direction_deadband_bps == 0.25
    assert cfg.target_config.output_dtype == "float64"
    assert cfg.preprocess_config.variance_floor == 1e-8
    assert cfg.preprocess_config.clip_z == 5.0
    assert cfg.preprocess_config.output_dtype == "float64"
    assert cfg.model_config.learning_rate == 0.01
    assert cfg.model_config.l2 == 0.001
    assert cfg.model_config.max_grad_norm == 10.0
    assert cfg.model_config.output_dtype == "float64"
    assert cfg.diagnostics_config.top_k == 5
    assert cfg.diagnostics_config.num_bins == 4
    assert cfg.diagnostics_config.max_rows == 123


def test_parser_rejects_bad_numeric_values() -> None:
    parser = cli.build_arg_parser()
    bad = [
        ["--batch-size", "0"], ["--epochs", "0"], ["--target-horizon-us", "0"],
        ["--direction-deadband-bps", "-0.1"], ["--variance-floor", "-1"], ["--clip-z", "0"],
        ["--learning-rate", "0"], ["--l2", "-1"], ["--max-grad-norm", "0"],
        ["--diagnostics-top-k", "0"], ["--diagnostics-num-bins", "0"], ["--diagnostics-max-rows", "0"],
        ["--learning-rate", "nan"], ["--learning-rate", "inf"],
    ]
    for extra in bad:
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset-root", "ds", "--output-dir", "out", *extra])


def test_main_calls_train_and_writer_and_prints_compact_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    se_train = lt.SplitEvaluation("train", 10, evaluation={}, diagnostics={})
    se_val = lt.SplitEvaluation("val", 4, evaluation={}, diagnostics={})
    result = lt.LinearTrainResult(
        schema_version=lt.TRAIN_RESULT_SCHEMA_VERSION,
        dataset_id="d1",
        manifest_hash="abc",
        config={},
        preprocess_state={},
        model_bundle_state={},
        splits={"train": se_train, "val": se_val},
    )
    calls: dict[str, object] = {}

    def fake_train(dataset_root: str, *, config: lt.LinearTrainConfig) -> lt.LinearTrainResult:
        calls["dataset_root"] = dataset_root
        calls["config"] = config
        return result

    def fake_write(result_obj: lt.LinearTrainResult, output_dir: str, *, filename: str) -> dict[str, str]:
        calls["result_obj"] = result_obj
        calls["output_dir"] = output_dir
        calls["filename"] = filename
        return {"result_json": "/tmp/out/result.json"}

    monkeypatch.setattr(lt, "train_linear_model", fake_train)
    monkeypatch.setattr(lt, "write_linear_train_artifacts", fake_write)

    rc = cli.main(["--dataset-root", "ds", "--output-dir", "out", "--result-filename", "x.json", "--epochs", "2"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["dataset_root"] == "ds"
    assert payload["output_dir"] == "out"
    assert payload["result_json"] == "/tmp/out/result.json"
    assert payload["dataset_id"] == "d1"
    assert payload["manifest_hash"] == "abc"
    assert payload["splits"] == {"train": {"n_rows": 10}, "val": {"n_rows": 4}}
    assert "model_bundle_state" not in payload
    assert "preprocess_state" not in payload
    assert "diagnostics" not in payload


def test_no_bad_imports() -> None:
    code = "import sys; before=set(sys.modules); import mmrt.cli.train_linear; after=set(sys.modules)-before; print('\\n'.join(sorted(after)))"
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    loaded = set(proc.stdout.splitlines())
    forbidden = {
        "pan" + "das", "po" + "lars", "to" + "rch", "sk" + "learn", "scipy", "pyarrow", "mmrt.storage.writer",
        "mmrt.storage.splits", "mmrt.data.tardis_csv", "mmrt.data.event_merge", "mmrt.features.engine",
        "mmrt.features.labels", "mmrt.features.transforms", "CMSSL17", "offline_" + "ingest",
    }
    assert loaded.isdisjoint(forbidden)


def test_no_old_pipeline_residue() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "stage" + "1", "stage" + "2",
        "stage" + "3", "stage" + "4", "stage" + "5", "Mini" + "Rocket", "Multi" + "Rocket",
        "Hy" + "dra", "Ae" + "on", "P" + "CA", "Standard" + "Scaler", "sk" + "learn", "to" + "rch",
        "pan" + "das", "po" + "lars", "GRACE_" + "MS", "global_" + "meta", "week", "tar." + "zst",
    ]
    for token in forbidden:
        assert token not in src


def test_no_raw_data_ingest_or_split_surface() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "tardis_" + "csv", "event_" + "merge", "book_" + "reconstructor", "FeatureEngine", "LabelBuilder",
        "CausalFeatureTransformer", "DecisionRowWriter", "build_split_plan", "write_split_manifest",
        "build_and_write_splits", "SplitMetadata",
    ]
    for token in forbidden:
        assert token not in src


def test_no_future_leakage_surface() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "future_" + "mid", "future_" + "ret", "shu" + "ffle", "sort_" + "values", "rand" + "om",
        "threshold_" + "search", "optimize_" + "threshold", "fit_" + "transform", "ROW_IDX_" + "COLUMN",
        "LOCAL_TS_" + "US_COLUMN", "TS_US_" + "COLUMN", "EVENT_SEQ_" + "COLUMN", "RAW_MID_" + "COLUMN",
    ]
    for token in forbidden:
        assert token not in src


def test_cli_init_is_empty_package_marker() -> None:
    import mmrt.cli as c

    assert c.__all__ == []
    src = inspect.getsource(c)
    forbidden = ["import mmrt.cli.train_linear", "ingest", "audit_dataset", "storage", "linear", "data", "features"]
    for token in forbidden:
        assert token not in src
