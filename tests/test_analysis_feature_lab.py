import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from mmrt.analysis import feature_lab as fl
from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm
from mmrt.linear import train as tr
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def _label_values(ret_bps: float) -> tuple[float, ...]:
    return (float(ret_bps), float(ret_bps), float(ret_bps))


def _write_predictive_ds(root: Path, *, train_rows: int = 60, val_rows: int = 30, test_rows: int = 20) -> Path:
    cfg = wr.WriterConfig(dataset_id="feature_lab_ds", created_at_utc="2026-01-01T00:00:00Z", dataset_root=str(root), chunk_rows=17)
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
        predictive = 0.0 if ret == 0.0 else (10.0 if ret > 0.0 else -10.0)
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


def _candidate_path(path: Path, n_rows: int = 110, **overrides) -> Path:
    di = np.arange(n_rows, dtype=np.int64)
    predictive = np.where(di % 6 == 0, 0.0, np.where(np.isin(di % 6, [1, 3]), 1.0, -1.0))
    duplicate = np.where(di % 6 == 0, 0.0, np.where(np.isin(di % 6, [1, 3]), 10.0, -10.0))
    data = {
        mf.DECISION_INDEX_COLUMN: di,
        "c_predictive_direction": predictive.astype(np.float64),
        "c_duplicate_existing": duplicate.astype(np.float64),
        "c_noise": np.sin(di).astype(np.float64),
    }
    data.update(overrides)
    table = pa.Table.from_pydict(data)
    pq.write_table(table, path)
    return path


def test_public_api_boundary():
    assert fl.__all__ == [
        "FEATURE_LAB_SCHEMA_VERSION",
        "DEFAULT_FEATURE_LAB_BATCH_SIZE",
        "DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN",
        "DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL",
        "DEFAULT_FEATURE_LAB_SEED",
        "DEFAULT_FEATURE_LAB_SUMMARY_FILENAME",
        "DEFAULT_CANDIDATE_HEALTH_FILENAME",
        "DEFAULT_CANDIDATE_EXISTING_CORR_FILENAME",
        "DEFAULT_CANDIDATE_REDUNDANCY_FILENAME",
        "DEFAULT_CANDIDATE_HEAD_METRICS_FILENAME",
        "DEFAULT_CANDIDATE_RECOMMENDATIONS_FILENAME",
        "FeatureLabConfig",
        "CandidateHealthRecord",
        "CandidateExistingCorrelationRecord",
        "CandidateRedundancyRecord",
        "CandidateHeadMetricRecord",
        "CandidateRecommendationRecord",
        "FeatureLabResult",
        "run_feature_lab",
        "write_feature_lab_artifacts",
    ]


def test_config_validation():
    assert fl.FeatureLabConfig().seed == 17
    bad = [
        {"batch_size": 0},
        {"batch_size": True},
        {"validate_dataset_on_open": 1},
        {"max_sample_rows_train": -1},
        {"max_sample_rows_val": True},
        {"seed": -1},
        {"variance_floor": 0.0},
        {"z_clip": float("inf")},
        {"moderate_redundancy_threshold": 0.0},
        {"high_redundancy_threshold": 1.0},
        {"moderate_redundancy_threshold": 0.98, "high_redundancy_threshold": 0.97},
        {"min_scope_rows": 0},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            fl.FeatureLabConfig(**kwargs)


def test_rejects_non_parquet_candidate_file(tmp_path: Path):
    p = tmp_path / "c.csv"
    p.write_text("decision_index,c_a\n0,1\n")
    with pytest.raises(ValueError, match="Parquet"):
        fl._read_candidate_parquet(str(p))


def test_rejects_missing_decision_index(tmp_path: Path):
    p = tmp_path / "c.parquet"
    pq.write_table(pa.Table.from_pydict({"c_a": [1.0]}), p)
    with pytest.raises(ValueError, match="decision_index"):
        fl._read_candidate_parquet(str(p))


def test_rejects_duplicate_decision_index(tmp_path: Path):
    p = tmp_path / "c.parquet"
    pq.write_table(pa.Table.from_pydict({"decision_index": [1, 1], "c_a": [1.0, 2.0]}), p)
    with pytest.raises(ValueError, match="unique"):
        fl._read_candidate_parquet(str(p))


def test_rejects_non_c_prefixed_candidate_columns(tmp_path: Path):
    p = tmp_path / "c.parquet"
    pq.write_table(pa.Table.from_pydict({"decision_index": [1], "bad": [1.0]}), p)
    with pytest.raises(ValueError, match="c_"):
        fl._read_candidate_parquet(str(p))


def test_rejects_non_numeric_candidate_columns(tmp_path: Path):
    p = tmp_path / "c.parquet"
    pq.write_table(pa.Table.from_pydict({"decision_index": [1], "c_bad": ["x"]}), p)
    with pytest.raises(ValueError, match="numeric"):
        fl._read_candidate_parquet(str(p))


def test_rejects_candidate_column_collision(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    p = tmp_path / "c.parquet"
    pq.write_table(pa.Table.from_pydict({"decision_index": [1], mf.feature_columns()[0]: [1.0]}), p)
    with pytest.raises(ValueError, match="c_"):
        fl.run_feature_lab(str(root), str(artifact), str(p))


def test_run_feature_lab_end_to_end(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    cand = _candidate_path(tmp_path / "c.parquet")
    out = fl.run_feature_lab(str(root), str(artifact), str(cand), config=fl.FeatureLabConfig(batch_size=11, min_scope_rows=1))
    assert out.schema_version == 1
    assert out.n_candidates == 3
    assert out.train_sample_rows == 60
    assert out.val_sample_rows == 30
    assert set(out.summary["top_candidates_by_head"].keys()) == set(lm.MODEL_HEADS)


def test_uses_train_and_val_only_not_test(trained_artifact, tmp_path: Path, monkeypatch):
    root, artifact = trained_artifact
    cand = _candidate_path(tmp_path / "c.parquet")
    seen = []
    original = rd.StorageDatasetReader.iter_split_batches

    def wrapped(self, role, *args, **kwargs):
        seen.append(SplitRole(role))
        yield from original(self, role, *args, **kwargs)

    monkeypatch.setattr(rd.StorageDatasetReader, "iter_split_batches", wrapped)
    out = fl.run_feature_lab(str(root), str(artifact), str(cand))
    assert out.train_sample_rows == 60
    assert out.val_sample_rows == 30
    assert SplitRole.TEST not in seen


def test_rejects_manifest_hash_mismatch(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    data = json.loads(artifact.read_text())
    data["manifest_hash"] = "bad"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest_hash"):
        fl.run_feature_lab(str(root), str(bad), str(_candidate_path(tmp_path / "c.parquet")))


def test_rejects_non_val_selection_split(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    data = json.loads(artifact.read_text())
    data["selection_summary"]["selection_split"] = "train"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selection_split"):
        fl.run_feature_lab(str(root), str(bad), str(_candidate_path(tmp_path / "c.parquet")))


def test_candidate_health_records(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    assert {(r.candidate, r.split) for r in out.health_records} == {(c, s) for c in ["c_predictive_direction", "c_duplicate_existing", "c_noise"] for s in ["train", "val"]}
    assert all(r.status in {"ok", "missing_review", "nonfinite_review", "low_variance", "bad_health"} for r in out.health_records)


def test_candidate_existing_correlations_include_all_existing_features(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    assert len(out.existing_correlation_records) == 3 * len(mf.feature_columns())
    assert {r.existing_feature for r in out.existing_correlation_records} == set(mf.feature_columns())


def test_redundancy_summary_detects_duplicate_existing_feature(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    dup = next(r for r in out.redundancy_records if r.candidate == "c_duplicate_existing")
    assert dup.status == "high_redundancy"
    assert dup.max_abs_existing_corr >= 0.99


def test_head_metrics_include_all_heads(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    assert {r.head for r in out.head_metric_records} == set(lm.MODEL_HEADS)
    assert len(out.head_metric_records) == 3 * 4


def test_head_scopes_are_correct(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    counts = {r.head: r.n_val_rows for r in out.head_metric_records if r.candidate == "c_noise"}
    assert counts[lm.NO_MOVE_HEAD] == 30
    assert counts[lm.DIRECTION_HEAD] == 25
    assert counts[lm.MAGNITUDE_UP_HEAD] == 10
    assert counts[lm.MAGNITUDE_DOWN_HEAD] == 15


def test_residual_ranking_detects_predictive_candidate(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    direction = [r for r in out.head_metric_records if r.head == lm.DIRECTION_HEAD]
    pred = next(r for r in direction if r.candidate == "c_predictive_direction")
    assert pred.rank_within_head_by_residual <= 2
    assert pred.target_val_abs_value > 0.45


def test_write_feature_lab_artifacts(trained_artifact, tmp_path: Path):
    root, artifact = trained_artifact
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")))
    paths = fl.write_feature_lab_artifacts(out, str(tmp_path / "out"))
    assert set(paths) == {"summary_json", "candidate_health_csv", "candidate_existing_correlations_csv", "candidate_redundancy_summary_csv", "candidate_head_metrics_csv", "candidate_recommendations_csv"}
    assert json.loads(Path(paths["summary_json"]).read_text())["schema_version"] == 1
    with Path(paths["candidate_head_metrics_csv"]).open() as f:
        assert next(csv.reader(f)) == ["candidate", "head", "scope", "n_train_rows", "n_val_rows", "target_metric_primary", "target_train_value", "target_val_value", "target_val_abs_value", "target_same_sign", "residual_metric_primary", "residual_train_value", "residual_val_value", "residual_val_abs_value", "residual_same_sign", "max_abs_existing_corr", "most_correlated_existing_feature", "missing_rate_train", "missing_rate_val", "finite_rate_train", "finite_rate_val", "zero_rate_train", "zero_rate_val", "health_status", "redundancy_status", "rank_within_head_by_residual"]


def test_no_forbidden_imports_or_old_pipeline_residue():
    src = inspect.getsource(fl)
    for bad in ["import pan" + "das", "import pol" + "ars", "import sk" + "learn", "import sci" + "py", "import to" + "rch", "import num" + "ba", "import job" + "lib"]:
        assert bad not in src
    for bad in ["BY" + "BIT", "CM" + "SSL", "offline" + "_ingest", "Mini" + "Rocket", "Multi" + "Rocket", "Hyd" + "ra", "Ae" + "on", "P" + "CA", "Standard" + "Scaler"]:
        assert bad not in src


def test_does_not_call_read_split_table(trained_artifact, tmp_path: Path, monkeypatch):
    root, artifact = trained_artifact

    def boom(*args, **kwargs):
        raise AssertionError("full split materialization is forbidden")

    monkeypatch.setattr(rd.StorageDatasetReader, "read_split_table", boom)
    out = fl.run_feature_lab(str(root), str(artifact), str(_candidate_path(tmp_path / "c.parquet")), config=fl.FeatureLabConfig(max_sample_rows_train=12, max_sample_rows_val=8, batch_size=5))
    assert out.train_sample_rows == 12
    assert out.val_sample_rows == 8
