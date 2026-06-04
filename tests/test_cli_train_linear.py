import inspect
import json

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

import mmrt.cli.train_linear as cli
import mmrt.linear.diagnostics as dg
import mmrt.linear.head_feature_presets as hp
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
    assert args.head_feature_preset == hp.ALL_FEATURES_PRESET
    assert args.target_horizon_us == tg.DEFAULT_TARGET_HORIZON_US
    assert args.move_deadband_bps == tg.DEFAULT_MOVE_DEADBAND_BPS
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
            "--target-horizon-us", "500000", "--move-deadband-bps", "0.25", "--target-output-dtype", "float64",
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
    assert cfg.target_config.move_deadband_bps == 0.25
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


def test_train_linear_default_head_feature_preset_is_all() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--dataset-root", "/tmp/ds", "--output-dir", "/tmp/out"])
    cfg = cli._config_from_args(args)
    assert cfg.head_feature_config.feature_columns_by_head is None


def test_train_linear_accepts_corr_pruned_head_feature_preset() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args([
        "--dataset-root", "/tmp/ds",
        "--output-dir", "/tmp/out",
        "--head-feature-preset", "corr_pruned152_head_subset_v1",
    ])
    cfg = cli._config_from_args(args)
    assert cfg.head_feature_config.feature_columns_by_head is not None
    assert len(cfg.head_feature_config.feature_columns_by_head["direction"]) == 40
    assert len(cfg.head_feature_config.feature_columns_by_head["no_move"]) == 40
    assert len(cfg.head_feature_config.feature_columns_by_head["magnitude_up"]) == 30
    assert len(cfg.head_feature_config.feature_columns_by_head["magnitude_down"]) == 40


def test_train_linear_accepts_corr_pruned_v2_head_feature_preset() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args([
        "--dataset-root", "/tmp/ds",
        "--output-dir", "/tmp/out",
        "--head-feature-preset", "corr_pruned152_head_subset_v2",
    ])
    cfg = cli._config_from_args(args)
    assert cfg.head_feature_config.feature_columns_by_head is not None
    assert len(cfg.head_feature_config.feature_columns_by_head["direction"]) == 34
    assert len(cfg.head_feature_config.feature_columns_by_head["no_move"]) == 39
    assert len(cfg.head_feature_config.feature_columns_by_head["magnitude_up"]) == 19
    assert len(cfg.head_feature_config.feature_columns_by_head["magnitude_down"]) == 9


def test_train_linear_accepts_corr_pruned_v3_head_feature_preset() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args([
        "--dataset-root", "/tmp/ds",
        "--output-dir", "/tmp/out",
        "--head-feature-preset", "corr_pruned152_head_subset_v3",
    ])
    cfg = cli._config_from_args(args)
    assert cfg.head_feature_config.feature_columns_by_head is not None
    assert len(cfg.head_feature_config.feature_columns_by_head["direction"]) == 25
    assert len(cfg.head_feature_config.feature_columns_by_head["no_move"]) == 38
    assert len(cfg.head_feature_config.feature_columns_by_head["magnitude_up"]) == 15
    assert len(cfg.head_feature_config.feature_columns_by_head["magnitude_down"]) == 6


def test_parser_rejects_bad_head_feature_preset() -> None:
    parser = cli.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--dataset-root", "ds",
            "--output-dir", "out",
            "--head-feature-preset", "__missing__",
        ])


def test_parser_rejects_bad_numeric_values() -> None:
    parser = cli.build_arg_parser()
    bad = [
        ["--batch-size", "0"], ["--epochs", "0"], ["--target-horizon-us", "0"],
        ["--move-deadband-bps", "-0.1"], ["--variance-floor", "-1"], ["--variance-floor", "0"], ["--clip-z", "0"],
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
        selection_summary={"selection_split": "val", "primary_metrics": {}, "guardrails": {}},
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
    src = inspect.getsource(cli)
    forbidden = [
        "from mmrt.data", "import mmrt.data", "from mmrt.features.engine", "from mmrt.features.labels", "from mmrt.features.transforms",
        "import pan" + "das", "from pan" + "das", "import to" + "rch", "from to" + "rch", "import sk" + "learn", "from sk" + "learn",
        "CM" + "SSL", "offline_" + "ingest", "linear_" + "offline",
    ]
    for token in forbidden:
        assert token not in src
    assert "direction_deadband_bps" not in src
    assert "DEFAULT_DIRECTION_DEADBAND_BPS" not in src
    assert "--direction-deadband-bps" not in src


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
        "tardis_" + "csv", "event_" + "merge", "book_" + "reconstructor", "Feature" + "Engine", "Label" + "Builder",
        "CausalFeature" + "Transformer", "DecisionRow" + "Writer", "build_" + "split_plan", "write_" + "split_manifest",
        "build_" + "and_write_splits", "Split" + "Metadata",
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

def test_cli_written_artifact_contains_no_move(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    result_path = tmp_path / "result.json"
    artifact_payload = {
        "model_bundle_state": {"no_move": {}, "feature_columns_by_head": {"no_move": ["x"]}},
        "preprocess_state": {"states_by_head": {"no_move": {}}},
        "config": {"resolved_head_features": {"feature_columns_by_head": {"no_move": ["x"]}}},
        "splits": {"train": {"evaluation": {"no_move": {}}, "diagnostics": {"coefficients": {"no_move": {}}}}},
        "selection_summary": {
            "selection_split": "val",
            "primary_metrics": {
                "no_move": {
                    "metric": "auc",
                    "value": 0.5,
                    "mode": "max",
                    "scope": "all_rows",
                },
                "direction": {
                    "metric": "auc",
                    "value": 0.5,
                    "mode": "max",
                    "scope": "move_mask",
                },
                "magnitude_up": {
                    "metric": "mae",
                    "value": 1.0,
                    "mode": "min",
                    "scope": "up_move_mask",
                },
                "magnitude_down": {
                    "metric": "mae",
                    "value": 1.0,
                    "mode": "min",
                    "scope": "down_move_mask",
                },
            },
            "guardrails": {
                "no_move": {"log_loss": 1.0, "brier": 0.25},
                "direction": {"log_loss": 1.0, "brier": 0.25},
                "magnitude_up": {"spearman": 0.0, "rmse": 1.0},
                "magnitude_down": {"spearman": 0.0, "rmse": 1.0},
            },
        },
    }

    def fake_train(*args, **kwargs):
        se = lt.SplitEvaluation("train", 1, evaluation={}, diagnostics={})
        return lt.LinearTrainResult(schema_version=1, dataset_id="d", manifest_hash="h", config={}, preprocess_state={}, model_bundle_state={}, splits={"train": se, "val": lt.SplitEvaluation("val", 1, evaluation={}, diagnostics={})}, selection_summary={"selection_split": "val", "primary_metrics": {}, "guardrails": {}})

    def fake_write(*args, **kwargs):
        result_path.write_text(json.dumps(artifact_payload))
        return {"result_json": str(result_path)}

    monkeypatch.setattr(lt, "train_linear_model", fake_train)
    monkeypatch.setattr(lt, "write_linear_train_artifacts", fake_write)
    rc = cli.main(["--dataset-root", "ds", "--output-dir", "out"])
    assert rc == 0
    payload = json.loads(result_path.read_text())
    assert "no_move" in payload["model_bundle_state"]
    assert "no_move" in payload["model_bundle_state"]["feature_columns_by_head"]
    assert "no_move" in payload["preprocess_state"]["states_by_head"]
    assert "no_move" in payload["config"]["resolved_head_features"]["feature_columns_by_head"]
    for split in payload["splits"].values():
        assert "no_move" in split["evaluation"]
        assert "no_move" in split["diagnostics"]["coefficients"]
    assert payload["selection_summary"]["primary_metrics"]["direction"]["metric"] == "auc"
    assert payload["selection_summary"]["primary_metrics"]["magnitude_up"]["metric"] == "mae"
