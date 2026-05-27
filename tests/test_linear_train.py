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
        "LinearTrainConfig", "SplitEvaluation", "LinearTrainResult", "fit_preprocessor_from_train_split",
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
    for forbidden in ["random_state", "shuffle", "pca_components", "split_config", "output_dir", "early_stopping", "class_weight", "sample_weight"]:
        assert not hasattr(cfg, forbidden)


def test_train_requires_existing_train_and_val_splits(tmp_path: Path):
    root, _ = make_dataset_with_splits(tmp_path / "a", with_splits=False)
    with pytest.raises(ValueError):
        tr.train_linear_model(str(root))
    root2, _ = make_dataset_with_splits(tmp_path / "b", with_splits=True, train_only=True)
    with pytest.raises(ValueError):
        tr.train_linear_model(str(root2))


def test_fit_preprocessor_from_train_split_uses_train_only(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3)
    st = tr.fit_preprocessor_from_train_split(reader, manifest=manifest, config=cfg)
    assert st.n_rows_fit == 6
    assert np.isclose(st.mean[0], np.mean([-1.0] * 6))


def test_train_model_bundle_from_train_split_updates_models_only_from_train(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3, epochs=2)
    st = tr.fit_preprocessor_from_train_split(reader, manifest=manifest, config=cfg)
    bundle = tr.train_model_bundle_from_train_split(reader, manifest=manifest, preprocess_state=st, config=cfg)
    train_rets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, -3.0])
    valid = np.sum(train_rets != 0.0)
    assert bundle.direction.n_rows_seen == 2 * int(valid)
    assert bundle.magnitude_up.n_rows_seen == 2 * 6
    assert bundle.magnitude_down.n_rows_seen == 2 * 6
    assert bundle.direction.is_fitted()


def test_evaluate_model_on_split_does_not_update_state(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(batch_size=3, epochs=1)
    st = tr.fit_preprocessor_from_train_split(reader, manifest=manifest, config=cfg)
    bundle = tr.train_model_bundle_from_train_split(reader, manifest=manifest, preprocess_state=st, config=cfg)
    snap = bundle.as_dict()
    out = tr.evaluate_model_on_split(reader, manifest=manifest, role=SplitRole.VAL, preprocess_state=st, model_bundle=bundle, config=cfg)
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
    assert {"direction", "magnitude_up", "magnitude_down"}.issubset(result.model_bundle_state.keys())
    assert {"feature_columns", "mean", "variance", "scale", "active_mask"}.issubset(result.preprocess_state.keys())
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
    st = tr.fit_preprocessor_from_train_split(reader, manifest=manifest, config=cfg)
    assert captured[-1] == tuple(manifest.feature_columns)
    _ = tr.train_model_bundle_from_train_split(reader, manifest=manifest, preprocess_state=st, config=cfg)
    assert len(captured[-1]) == len(manifest.feature_columns) + 1


def test_direction_invalid_rows_filtered_for_direction_head(tmp_path: Path):
    root, manifest = make_dataset_with_splits(tmp_path)
    reader = rd.open_dataset(str(root), validate_on_open=True, batch_size=3)
    cfg = tr.LinearTrainConfig(epochs=1)
    st = tr.fit_preprocessor_from_train_split(reader, manifest=manifest, config=cfg)
    bundle = tr.train_model_bundle_from_train_split(reader, manifest=manifest, preprocess_state=st, config=cfg)
    assert bundle.direction.n_rows_seen == 5
    assert bundle.magnitude_up.n_rows_seen == 6


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
        tr.LinearTrainResult(schema_version=2, dataset_id="d", manifest_hash="h", config={}, preprocess_state={}, model_bundle_state={}, splits={"train": se, "val": se})


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
