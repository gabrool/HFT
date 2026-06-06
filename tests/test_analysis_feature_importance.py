import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

from mmrt.analysis import feature_importance as fi
from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm
from mmrt.linear import train as tr
from mmrt.storage import manifest as mf
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def _label_values(ret_bps: float) -> tuple[float, ...]:
    return (float(ret_bps), float(ret_bps), float(ret_bps))


def _write_predictive_ds(root: Path, *, train_rows: int = 60, val_rows: int = 30, test_rows: int = 20) -> Path:
    cfg = wr.WriterConfig(dataset_id="fi_ds", created_at_utc="2026-01-01T00:00:00Z", dataset_root=str(root), chunk_rows=17)
    writer = wr.DecisionRowWriter(cfg)
    n_rows = train_rows + val_rows + test_rows
    for i in range(n_rows):
        mod = i % 6
        if mod == 0:
            ret = 0.0
        elif mod in (1, 3):
            ret = 3.0 + 0.1 * (i % 5)
        else:
            ret = -3.0 - 0.1 * (i % 5)
        predictive = 0.0 if ret == 0.0 else (10.0 if ret > 0 else -10.0)
        features = [predictive]
        features.extend(0.001 * (((i + j) % 7) - 3) for j in range(specs.FEATURE_COUNT - 1))
        ts = 1_000_000 + i * 1_000
        writer.append_values(
            decision_index=i,
            ts_us=ts,
            local_ts_us=ts,
            event_seq=i,
            raw_mid=100.0 + i,
            label_entry_ts_us=ts,
            label_values=_label_values(ret),
            feature_values=tuple(features),
        )
    writer.finalize()
    windows = [
        sp.SplitWindow(SplitRole.TRAIN, 1_000_000, 1_000_000 + train_rows * 1_000),
        sp.SplitWindow(SplitRole.VAL, 1_000_000 + train_rows * 1_000, 1_000_000 + (train_rows + val_rows) * 1_000),
        sp.SplitWindow(SplitRole.TEST, 1_000_000 + (train_rows + val_rows) * 1_000, 1_000_000 + n_rows * 1_000 + 1),
    ]
    sp.build_and_write_splits(
        str(root),
        sp.SplitConfig(
            windows=tuple(windows),
            purge_before_us=0,
            purge_after_us=0,
            embargo_before_us=0,
            embargo_after_us=0,
            min_rows_per_split=1,
            allow_empty_roles=False,
            validate_dataset_on_open=True,
        ),
        replace_existing=True,
    )
    return root


@pytest.fixture()
def trained_artifact(tmp_path: Path) -> tuple[Path, Path]:
    root = _write_predictive_ds(tmp_path / "ds")
    all_cols = mf.feature_columns()
    cols = (all_cols[0], all_cols[2], all_cols[4], all_cols[6])
    head_cfg = hf.HeadFeatureConfig(feature_columns_by_head={head: cols for head in lm.MODEL_HEADS})
    result = tr.train_linear_model(str(root), config=tr.LinearTrainConfig(batch_size=13, epochs=20, head_feature_config=head_cfg))
    paths = tr.write_linear_train_artifacts(result, str(tmp_path / "train"))
    return root, Path(paths["result_json"])


def test_public_api_boundary():
    assert fi.__all__ == [
        "FEATURE_IMPORTANCE_REPORT_TYPE",
        "DEFAULT_FEATURE_IMPORTANCE_BATCH_SIZE",
        "DEFAULT_FEATURE_IMPORTANCE_MAX_SAMPLE_ROWS",
        "DEFAULT_FEATURE_IMPORTANCE_SEED",
        "DEFAULT_FEATURE_IMPORTANCE_SUMMARY_FILENAME",
        "DEFAULT_FEATURE_IMPORTANCE_BY_HEAD_FILENAME",
        "DEFAULT_FEATURE_IMPORTANCE_FAMILY_SUMMARY_FILENAME",
        "FeatureImportanceConfig",
        "FeatureImportanceRecord",
        "FeatureImportanceFamilyRecord",
        "FeatureImportanceResult",
        "run_feature_importance",
        "write_feature_importance_artifacts",
    ]


def test_config_validation():
    assert fi.FeatureImportanceConfig().seed == 17
    for kwargs in [
        {"batch_size": 0},
        {"batch_size": True},
        {"validate_dataset_on_open": 1},
        {"max_sample_rows": -1},
        {"max_sample_rows": True},
        {"seed": -1},
        {"seed": True},
    ]:
        with pytest.raises(ValueError):
            fi.FeatureImportanceConfig(**kwargs)


def test_run_feature_importance_end_to_end(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact), config=fi.FeatureImportanceConfig(batch_size=11))
    assert out.report_type == fi.FEATURE_IMPORTANCE_REPORT_TYPE
    assert out.selection_split == "val"
    assert set(r.head for r in out.records) == set(lm.MODEL_HEADS)
    assert set(out.summary["heads"].keys()) == set(lm.MODEL_HEADS)


def test_importance_uses_validation_not_train_or_test(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact))
    direction = [r for r in out.records if r.head == lm.DIRECTION_HEAD]
    assert {r.n_eval_rows for r in direction} == {25}
    assert out.n_sample_rows == 30


def test_feature_index_is_canonical_not_head_local(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact))

    for record in out.records:
        canonical = record.feature[len(mf.FEATURE_COLUMN_PREFIX) :]
        expected = specs.feature_spec_by_name(canonical).index
        assert record.feature_index == expected


def test_run_feature_importance_streams_validation_sampling(trained_artifact, monkeypatch):
    root, artifact = trained_artifact

    from mmrt.storage import reader as rd

    def boom(*args, **kwargs):
        raise AssertionError("read_split_table must not be used by feature importance")

    monkeypatch.setattr(rd.StorageDatasetReader, "read_split_table", boom)

    out = fi.run_feature_importance(
        str(root),
        str(artifact),
        config=fi.FeatureImportanceConfig(max_sample_rows=12, batch_size=5),
    )

    assert out.n_sample_rows == 12
    assert set(r.head for r in out.records) == set(lm.MODEL_HEADS)


def test_max_sample_rows_bounds_validation_sample(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(
        str(root),
        str(artifact),
        config=fi.FeatureImportanceConfig(max_sample_rows=7, batch_size=3),
    )
    assert out.n_sample_rows == 7
    assert {r.n_eval_rows for r in out.records if r.head == lm.NO_MOVE_HEAD} == {7}


def test_zero_sample_rows_produces_empty_importance_metrics(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(
        str(root),
        str(artifact),
        config=fi.FeatureImportanceConfig(max_sample_rows=0),
    )
    assert out.n_sample_rows == 0
    assert set(r.head for r in out.records) == set(lm.MODEL_HEADS)
    assert all(r.n_eval_rows == 0 for r in out.records)


def test_rejects_manifest_hash_mismatch(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    data = json.loads(artifact.read_text())
    data["manifest_hash"] = "bad"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest_hash"):
        fi.run_feature_importance(str(root), str(bad))


def test_rejects_non_val_selection_split(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    data = json.loads(artifact.read_text())
    data["selection_summary"]["selection_split"] = "test"
    bad = tmp_path / "bad_split.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selection_split"):
        fi.run_feature_importance(str(root), str(bad))


def test_outputs_all_four_heads(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact))
    assert [head for head in lm.MODEL_HEADS if any(r.head == head for r in out.records)] == list(lm.MODEL_HEADS)


def test_head_scopes_are_correct(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact))
    counts = {head: {r.n_eval_rows for r in out.records if r.head == head} for head in lm.MODEL_HEADS}
    assert counts[lm.NO_MOVE_HEAD] == {30}
    assert counts[lm.DIRECTION_HEAD] == {25}
    assert counts[lm.MAGNITUDE_UP_HEAD] == {10}
    assert counts[lm.MAGNITUDE_DOWN_HEAD] == {15}


def test_permutation_importance_detects_predictive_feature(trained_artifact):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact))
    top_direction = sorted([r for r in out.records if r.head == lm.DIRECTION_HEAD], key=lambda r: r.importance_rank)[0]
    assert top_direction.feature == mf.feature_columns()[0]
    assert top_direction.primary_importance > 0


def test_write_feature_importance_artifacts(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fi.run_feature_importance(str(root), str(artifact), config=fi.FeatureImportanceConfig(max_sample_rows=12))
    paths = fi.write_feature_importance_artifacts(out, str(tmp_path / "out"))
    assert set(paths) == {"summary_json", "by_head_csv", "family_summary_csv"}
    assert Path(paths["summary_json"]).name == fi.DEFAULT_FEATURE_IMPORTANCE_SUMMARY_FILENAME
    header = Path(paths["by_head_csv"]).read_text().splitlines()[0]
    assert header == "head,feature,feature_index,source,owner,family,unit,transform_key,required_book_depth,n_eval_rows,primary_metric,primary_mode,base_primary,permuted_primary,primary_importance,guardrail_metric,base_guardrail,permuted_guardrail,guardrail_delta,coefficient,abs_coefficient,coefficient_rank,importance_rank"


def test_no_forbidden_imports_or_old_pipeline_residue():
    src = inspect.getsource(fi)
    for bad in [
        "import pan" + "das",
        "import po" + "lars",
        "import sk" + "learn",
        "import sci" + "py",
        "import to" + "rch",
        "import num" + "ba",
        "import job" + "lib",
        "BY" + "BIT",
        "CM" + "SSL",
        "offline_" + "ingest",
        "Mini" + "Rocket",
        "Multi" + "Rocket",
        "Hy" + "dra",
        "Ae" + "on",
        "P" + "CA",
        "Standard" + "Scaler",
    ]:
        assert bad not in src


def test_no_raw_data_or_split_building_surface():
    src = inspect.getsource(fi)
    for bad in ["read_csv", "build_and_write_splits", "SplitConfig", "permutation_repeats", "n_jobs"]:
        assert bad not in src.lower()
