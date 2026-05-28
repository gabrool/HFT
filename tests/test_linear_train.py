import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import writer as wr
from mmrt.storage import splits as sp
from mmrt.linear import train as tr
from mmrt.linear import extractors as ex
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm


def feature_values(row_idx: int) -> tuple[float, ...]:
    x0 = -1.0 if row_idx < 6 else 1.0
    rest = [0.01 * ((row_idx + i) % 3) for i in range(specs.FEATURE_COUNT - 1)]
    return (x0, *rest)


def label_values(ret_bps: float) -> tuple[float, ...]:
    return (float(ret_bps), float(ret_bps), float(ret_bps))


def make_dataset_with_splits(tmp_path: Path, *, with_test: bool = True, with_splits: bool = True, train_only: bool = False, train_zero_rows: bool = False):
    root = tmp_path / "ds"
    cfg = wr.WriterConfig(dataset_id="d1", created_at_utc="2026-01-01T00:00:00Z", dataset_root=str(root), chunk_rows=4)
    writer = wr.DecisionRowWriter(cfg)
    rets = [-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, -1.5, 1.5, 0.5, -0.5, 2.5, -2.5]
    for i, ret in enumerate(rets):
        ts = 1_000_000 + i * 500_000
        writer.append_values(
            decision_index=i + 1,
            ts_us=ts,
            local_ts_us=ts,
            event_seq=i + 10,
            raw_mid=100.0 + i,
            label_entry_ts_us=ts,
            label_values=label_values(ret),
            feature_values=feature_values(i),
        )
    manifest = writer.finalize()
    if with_splits:
        windows = [sp.SplitWindow(SplitRole.TRAIN, 1_000_000, 4_000_000)]
        if train_zero_rows:
            windows = [
                sp.SplitWindow(SplitRole.TRAIN, 1_000_000, 1_500_000),
                sp.SplitWindow(SplitRole.VAL, 4_000_000, 5_500_000),
            ]
        else:
            if not train_only:
                windows.append(sp.SplitWindow(SplitRole.VAL, 4_000_000, 5_500_000))
            if with_test:
                windows.append(sp.SplitWindow(SplitRole.TEST, 5_500_000, 7_000_001))
        split_cfg = sp.SplitConfig(windows=tuple(windows), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0, min_rows_per_split=1, allow_empty_roles=False, validate_dataset_on_open=True)
        manifest = sp.build_and_write_splits(str(root), split_cfg, replace_existing=True)
    return root, manifest


def test_public_api_boundary():
    assert tr.__all__ == [
        "DEFAULT_TRAIN_BATCH_SIZE", "DEFAULT_EPOCHS", "DEFAULT_OUTPUT_FILENAME", "TRAIN_RESULT_SCHEMA_VERSION",
        "LinearTrainConfig", "SplitEvaluation", "LinearTrainResult", "fit_preprocessors_from_train_split",
        "train_model_bundle_from_train_split", "evaluate_model_on_split", "train_linear_model", "write_linear_train_artifacts",
    ]


def test_no_forbidden_imports():
    src = inspect.getsource(tr)
    for bad in [
        "import pan" + "das", "from pan" + "das", "import po" + "lars", "from po" + "lars",
        "import to" + "rch", "from to" + "rch", "import sk" + "learn", "from sk" + "learn",
        "from mmrt.data", "import mmrt.data", "from mmrt.features.engine", "from mmrt.features.labels", "from mmrt.features.transforms",
        "CM" + "SSL", "offline_" + "ingest", "linear_" + "offline", "BY" + "BIT",
        "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on", "P" + "CA", "Standard" + "Scaler",
        "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5",
    ]:
        assert bad not in src


def test_config_validation():
    cfg = tr.LinearTrainConfig()
    assert cfg.batch_size == 8192 and cfg.epochs == 5 and cfg.validate_dataset_on_open is True
    for kwargs in [{"batch_size": 0}, {"batch_size": -1}, {"batch_size": True}, {"epochs": 0}, {"epochs": -1}, {"epochs": True}, {"validate_dataset_on_open": 1}]:
        with pytest.raises(ValueError):
            tr.LinearTrainConfig(**kwargs)
    with pytest.raises(ValueError):
        tr.LinearTrainConfig(extractor_config=object())
    d = cfg.as_dict()
    assert set(d.keys()) >= {"extractor_config", "target_config", "preprocess_config", "model_config", "diagnostics_config"}
    assert d["model_config"]["magnitude_huber_delta"] == 1.0
    for forbidden in ["random_state", "shuffle", "pca_components", "split_config", "output_dir", "early_stopping", "class_weight", "sample_weight"]:
        assert not hasattr(cfg, forbidden)


def test_train_requires_existing_train_and_val_splits(tmp_path: Path):
    root, _ = make_dataset_with_splits(tmp_path / "a", with_splits=False)
    with pytest.raises(ValueError):
        tr.train_linear_model(str(root))
    root2, _ = make_dataset_with_splits(tmp_path / "b", with_splits=True, train_only=True)
    with pytest.raises(ValueError):
        tr.train_linear_model(str(root2))


def test_fit_preprocessors_from_train_split_uses_train_only(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3)
    resolved = hf.resolve_head_feature_sets(manifest)
    st = tr.fit_preprocessors_from_train_split(reader, manifest=manifest, head_features=resolved, config=cfg)
    assert st[lm.DIRECTION_HEAD].n_rows_fit == 6
    assert np.isclose(st[lm.DIRECTION_HEAD].mean[0], np.mean([-1.0] * 6))


def test_train_model_bundle_from_train_split_updates_models_only_from_train(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3, epochs=2)
    resolved = hf.resolve_head_feature_sets(manifest)
    st = tr.fit_preprocessors_from_train_split(reader, manifest=manifest, head_features=resolved, config=cfg)
    bundle = tr.train_model_bundle_from_train_split(reader, manifest=manifest, head_features=resolved, preprocess_states_by_head=st, config=cfg)
    train_rets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, -3.0])
    assert bundle.no_move.n_rows_seen == 12
    assert bundle.direction.n_rows_seen == 10
    assert bundle.magnitude_up.n_rows_seen == 4
    assert bundle.magnitude_down.n_rows_seen == 6
    assert bundle.direction.is_fitted()


def test_evaluate_model_on_split_does_not_update_state(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3, epochs=1)
    resolved = hf.resolve_head_feature_sets(manifest)
    st = tr.fit_preprocessors_from_train_split(reader, manifest=manifest, head_features=resolved, config=cfg)
    bundle = tr.train_model_bundle_from_train_split(reader, manifest=manifest, head_features=resolved, preprocess_states_by_head=st, config=cfg)
    snap = bundle.as_dict()
    out = tr.evaluate_model_on_split(reader, manifest=manifest, role=SplitRole.VAL, head_features=resolved, preprocess_states_by_head=st, model_bundle=bundle, config=cfg)
    assert bundle.as_dict() == snap
    assert out.role == "val" and out.n_rows == 3
    assert isinstance(out.evaluation, dict) and isinstance(out.diagnostics, dict)


def test_train_linear_model_end_to_end(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    cfg = tr.LinearTrainConfig(epochs=2, batch_size=3)
    result = tr.train_linear_model(str(root), config=cfg)
    assert result.schema_version == 1
    assert result.dataset_id == manifest.dataset_id
    assert result.manifest_hash == rd.open_dataset(str(root)).manifest.content_hash()
    assert set(result.splits.keys()) == {"train", "val", "test"}
    assert result.splits["train"].n_rows == 6 and result.splits["val"].n_rows == 3
    assert {"no_move", "direction", "magnitude_up", "magnitude_down"}.issubset(result.model_bundle_state.keys())
    assert result.preprocess_state["schema"] == "per_head_preprocess_v1"
    assert set(result.preprocess_state["states_by_head"].keys()) == set(lm.MODEL_HEADS)
    payload = result.as_dict()
    selection = payload["selection_summary"]
    assert selection["selection_split"] == "val"
    assert set(selection["primary_metrics"]) == {"no_move", "direction", "magnitude_up", "magnitude_down"}
    assert selection["primary_metrics"]["no_move"]["metric"] == "auc"
    assert selection["primary_metrics"]["no_move"]["mode"] == "max"
    assert selection["primary_metrics"]["no_move"]["scope"] == "all_rows"
    assert selection["primary_metrics"]["direction"]["metric"] == "auc"
    assert selection["primary_metrics"]["direction"]["mode"] == "max"
    assert selection["primary_metrics"]["direction"]["scope"] == "move_mask"
    assert selection["primary_metrics"]["magnitude_up"]["metric"] == "mae"
    assert selection["primary_metrics"]["magnitude_up"]["mode"] == "min"
    assert selection["primary_metrics"]["magnitude_up"]["scope"] == "up_move_mask"
    assert selection["primary_metrics"]["magnitude_down"]["metric"] == "mae"
    assert selection["primary_metrics"]["magnitude_down"]["mode"] == "min"
    assert selection["primary_metrics"]["magnitude_down"]["scope"] == "down_move_mask"
    assert set(selection["guardrails"]["no_move"]) == {"log_loss", "brier"}
    assert set(selection["guardrails"]["direction"]) == {"log_loss", "brier"}
    assert set(selection["guardrails"]["magnitude_up"]) == {"spearman", "rmse"}
    assert set(selection["guardrails"]["magnitude_down"]) == {"spearman", "rmse"}
    for head_payload in selection["primary_metrics"].values():
        assert isinstance(head_payload["value"], float)
    for guard_payload in selection["guardrails"].values():
        for value in guard_payload.values():
            assert isinstance(value, float)
    assert "no_move" in payload["model_bundle_state"]
    assert "no_move" in payload["model_bundle_state"]["feature_columns_by_head"]
    assert "no_move" in payload["preprocess_state"]["states_by_head"]
    assert "no_move" in payload["config"]["resolved_head_features"]["feature_columns_by_head"]
    for split in payload["splits"].values():
        assert "no_move" in split["evaluation"]
        assert "gated_signal" in split["evaluation"]
        assert "no_move" in split["diagnostics"]["coefficients"]
        assert "p_no_move" in split["diagnostics"]["predictions"]
        assert "expected_signed_edge_bps" in split["diagnostics"]["predictions"]
    json.dumps(result.as_dict(), allow_nan=True)


def test_train_linear_model_without_test_split(tmp_path: Path):
    root, _ = make_dataset_with_splits(tmp_path, with_test=False)
    result = tr.train_linear_model(str(root))
    assert set(result.splits.keys()) == {"train", "val"}


def test_column_projection_reads_only_features_and_target(tmp_path: Path, monkeypatch):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3, epochs=1)
    captured = []
    orig = tr._split_batches
    def wrapped(reader_, role, columns, batch_size):
        captured.append(tuple(columns))
        yield from orig(reader_, role, columns, batch_size)
    monkeypatch.setattr(tr, "_split_batches", wrapped)
    resolved = hf.resolve_head_feature_sets(manifest)
    st = tr.fit_preprocessors_from_train_split(reader, manifest=manifest, head_features=resolved, config=cfg)
    assert captured[-1] == tuple(manifest.feature_columns)
    _ = tr.train_model_bundle_from_train_split(reader, manifest=manifest, head_features=resolved, preprocess_states_by_head=st, config=cfg)
    assert len(captured[-1]) == len(manifest.feature_columns) + 1


def test_direction_invalid_rows_filtered_for_direction_head(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(epochs=1)
    resolved = hf.resolve_head_feature_sets(manifest)
    st = tr.fit_preprocessors_from_train_split(reader, manifest=manifest, head_features=resolved, config=cfg)
    bundle = tr.train_model_bundle_from_train_split(reader, manifest=manifest, head_features=resolved, preprocess_states_by_head=st, config=cfg)
    assert bundle.no_move.n_rows_seen == 6
    assert bundle.direction.n_rows_seen == 5
    assert bundle.magnitude_up.n_rows_seen == 2
    assert bundle.magnitude_down.n_rows_seen == 3


def test_write_linear_train_artifacts(tmp_path: Path):
    root, _ = make_dataset_with_splits(tmp_path / "data")
    result = tr.train_linear_model(str(root), config=tr.LinearTrainConfig(batch_size=3, epochs=1))
    out = tr.write_linear_train_artifacts(result, str(tmp_path / "out"))
    p = Path(out["result_json"])
    assert p.exists()
    assert json.loads(p.read_text())
    assert not (Path(str(p) + ".tmp")).exists()
    with pytest.raises(ValueError):
        tr.write_linear_train_artifacts(result, "", filename="x.json")
    with pytest.raises(ValueError):
        tr.write_linear_train_artifacts(result, str(tmp_path / "o2"), filename="x.txt")
    with pytest.raises(ValueError):
        tr.write_linear_train_artifacts(object(), str(tmp_path / "o3"))


def test_result_dataclass_validation():
    se = tr.SplitEvaluation(role="train", n_rows=0, evaluation={}, diagnostics={})
    with pytest.raises(ValueError):
        tr.SplitEvaluation(role="bad", n_rows=0, evaluation={}, diagnostics={})
    with pytest.raises(ValueError):
        tr.LinearTrainResult(schema_version=2, dataset_id="d", manifest_hash="h", config={}, preprocess_state={}, model_bundle_state={}, splits={"train": se, "val": se}, selection_summary={})


def test_no_old_pipeline_residue():
    source = inspect.getsource(tr)
    for bad in ["BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "stage" + "1", "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on", "sk" + "learn", "to" + "rch", "pan" + "das", "po" + "lars", "P" + "CA"]:
        assert bad not in source


def test_no_raw_data_or_split_building_surface():
    source = inspect.getsource(tr)
    for bad in ["tardis_csv", "event_merge", "book_reconstructor", "FeatureEngine", "LabelBuilder", "build_split_plan", "write_split_manifest", "build_and_write_splits", "DecisionRowWriter"]:
        assert bad not in source


def test_no_future_leakage_surface():
    source = inspect.getsource(tr)
    for bad in ["future_" + "mid", "future_" + "ret", "shuffle", "sort_" + "values", "rand" + "om", "threshold_search", "optimize_threshold", "fit_transform", "ROW_IDX_" + "COLUMN", "LOCAL_TS_" + "US_COLUMN", "TS_US_COLUMN", "EVENT_SEQ_COLUMN", "RAW_MID_COLUMN"]:
        assert bad not in source


def test_no_row_loop_over_examples():
    source = inspect.getsource(tr)
    for bad in [".iterrows", "to_pandas", "for row in", "for sample in"]:
        assert bad not in source


def test_train_config_rejects_global_extractor_feature_columns(tmp_path: Path):
    _, manifest = make_dataset_with_splits(tmp_path)
    col = manifest.feature_columns[0]

    with pytest.raises(ValueError, match="head_feature_config"):
        tr.LinearTrainConfig(
            extractor_config=ex.LinearFeatureExtractorConfig(feature_columns=(col,))
        )


def test_train_linear_model_respects_per_head_feature_subsets(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    cols = tuple(manifest.feature_columns)

    cfg = tr.LinearTrainConfig(
        epochs=1,
        batch_size=3,
        head_feature_config=hf.HeadFeatureConfig(
            {
                lm.DIRECTION_HEAD: (cols[0], cols[1]),
                lm.MAGNITUDE_UP_HEAD: (cols[1], cols[2]),
                lm.MAGNITUDE_DOWN_HEAD: (cols[2], cols[3]),
            }
        ),
    )

    result = tr.train_linear_model(str(root), config=cfg)

    expected = {
        lm.DIRECTION_HEAD: [cols[0], cols[1]],
        lm.MAGNITUDE_UP_HEAD: [cols[1], cols[2]],
        lm.MAGNITUDE_DOWN_HEAD: [cols[2], cols[3]],
    }

    resolved = result.config["resolved_head_features"]["feature_columns_by_head"]
    assert resolved == expected

    states = result.preprocess_state["states_by_head"]
    for head, expected_cols in expected.items():
        assert states[head]["feature_columns"] == expected_cols
        assert result.model_bundle_state[head]["feature_columns"] == expected_cols

    assert result.model_bundle_state["feature_columns_by_head"] == expected
    assert result.model_bundle_state["feature_counts_by_head"] == {
        head: len(cols_for_head)
        for head, cols_for_head in expected.items()
    }

    for split_eval in result.splits.values():
        assert set(split_eval.evaluation.keys()) == {
            lm.DIRECTION_HEAD,
            lm.MAGNITUDE_UP_HEAD,
            lm.MAGNITUDE_DOWN_HEAD,
        }

    for split_eval in result.splits.values():
        pre_diag = split_eval.diagnostics["preprocess"]
        assert pre_diag["schema"] == "per_head_preprocess_v1"
        assert set(pre_diag["states_by_head"]) == set(lm.MODEL_HEADS)
        assert pre_diag["states_by_head"][lm.DIRECTION_HEAD]["n_features"] == 2
        assert pre_diag["states_by_head"][lm.MAGNITUDE_UP_HEAD]["n_features"] == 2
        assert pre_diag["states_by_head"][lm.MAGNITUDE_DOWN_HEAD]["n_features"] == 2


def test_train_missing_head_feature_entry_defaults_to_all_features(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    cols = tuple(manifest.feature_columns)

    cfg = tr.LinearTrainConfig(
        epochs=1,
        batch_size=3,
        head_feature_config=hf.HeadFeatureConfig(
            {
                lm.DIRECTION_HEAD: (cols[0], cols[1]),
            }
        ),
    )

    result = tr.train_linear_model(str(root), config=cfg)
    resolved = result.config["resolved_head_features"]["feature_columns_by_head"]

    assert resolved[lm.DIRECTION_HEAD] == [cols[0], cols[1]]
    assert resolved[lm.MAGNITUDE_UP_HEAD] == list(cols)
    assert resolved[lm.MAGNITUDE_DOWN_HEAD] == list(cols)

    assert result.preprocess_state["states_by_head"][lm.MAGNITUDE_UP_HEAD]["feature_columns"] == list(cols)
    assert result.model_bundle_state[lm.MAGNITUDE_DOWN_HEAD]["feature_columns"] == list(cols)


def test_train_per_head_feature_order_is_manifest_order(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    cols = tuple(manifest.feature_columns)

    cfg = tr.LinearTrainConfig(
        epochs=1,
        batch_size=3,
        head_feature_config=hf.HeadFeatureConfig(
            {
                lm.DIRECTION_HEAD: (cols[3], cols[1]),
                lm.MAGNITUDE_UP_HEAD: (cols[2], cols[0]),
                lm.MAGNITUDE_DOWN_HEAD: (cols[4], cols[1]),
            }
        ),
    )

    result = tr.train_linear_model(str(root), config=cfg)
    resolved = result.config["resolved_head_features"]["feature_columns_by_head"]

    assert resolved[lm.DIRECTION_HEAD] == [cols[1], cols[3]]
    assert resolved[lm.MAGNITUDE_UP_HEAD] == [cols[0], cols[2]]
    assert resolved[lm.MAGNITUDE_DOWN_HEAD] == [cols[1], cols[4]]


def test_model_bundle_predict_rejects_nonshared_feature_columns():
    bundle = lm.make_linear_model_bundle(
        {
            lm.DIRECTION_HEAD: ("x_a", "x_b"),
            lm.MAGNITUDE_UP_HEAD: ("x_a",),
            lm.MAGNITUDE_DOWN_HEAD: ("x_b",),
        },
        lm.LinearModelConfig(),
    )

    with pytest.raises(ValueError, match="requires all heads to share"):
        bundle.predict(np.zeros((2, 2), dtype=np.float32))


def test_make_linear_model_bundle_requires_exact_mapping_keys():
    with pytest.raises(ValueError):
        lm.make_linear_model_bundle(
            {
                lm.DIRECTION_HEAD: ("x_a",),
                lm.MAGNITUDE_UP_HEAD: ("x_a",),
            }
        )

    with pytest.raises(ValueError):
        lm.make_linear_model_bundle(
            {
                lm.DIRECTION_HEAD: ("x_a",),
                lm.MAGNITUDE_UP_HEAD: ("x_a",),
                lm.MAGNITUDE_DOWN_HEAD: ("x_a",),
                "bad": ("x_a",),
            }
        )
