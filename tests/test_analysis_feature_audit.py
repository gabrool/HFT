from pathlib import Path
import numpy as np
import pytest

from mmrt.analysis import feature_audit as fa
from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.storage import manifest as mf
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def _write_feature_audit_ds(
    root: Path,
    train_rows: int = 100,
    val_rows: int = 100,
    test_rows: int = 0,
    correlated: bool = False,
    low_variance: bool = False,
    val_shift: float = 0.0,
) -> None:
    writer = wr.DecisionRowWriter(
        wr.WriterConfig(
            dataset_id="d",
            created_at_utc="2026",
            dataset_root=str(root),
            chunk_rows=32,
        )
    )
    n = train_rows + val_rows + test_rows
    for i in range(n):
        ts = 1_000_000 + i * 1_000
        if i < train_rows:
            x0 = float(i)
        elif i < train_rows + val_rows:
            val_i = i - train_rows
            x0 = float(val_i + val_shift)
        else:
            test_i = i - train_rows - val_rows
            x0 = float(test_i)

        x1 = x0 * 2.0 if correlated else float((i * 7) % 11)
        x2 = 1.0 if low_variance else float((i * 3) % 5)
        feats = [x0, x1, x2] + [float((i + j) % 13) for j in range(specs.FEATURE_COUNT - 3)]
        writer.append_values(
            decision_index=i + 1,
            ts_us=ts,
            local_ts_us=ts,
            event_seq=i,
            raw_mid=100.0,
            label_entry_ts_us=ts,
            label_values=(1.0, 1.0, 1.0),
            feature_values=tuple(feats),
        )
    writer.finalize()

    windows = [
        sp.SplitWindow(SplitRole.TRAIN, 1_000_000, 1_000_000 + train_rows * 1_000),
        sp.SplitWindow(
            SplitRole.VAL,
            1_000_000 + train_rows * 1_000,
            1_000_000 + (train_rows + val_rows) * 1_000,
        ),
    ]
    if test_rows:
        windows.append(
            sp.SplitWindow(
                SplitRole.TEST,
                1_000_000 + (train_rows + val_rows) * 1_000,
                1_000_000 + n * 1_000 + 1,
            )
        )

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


def test_feature_audit_runs_and_returns_expected_records(tmp_path: Path):
    root = tmp_path / "ds"
    _write_feature_audit_ds(root)
    out = fa.run_feature_audit(str(root))
    assert "train" in out.splits and "val" in out.splits
    assert len(out.health_records) == specs.FEATURE_COUNT * 2
    assert "feature_records" not in out.as_dict()
    assert "missing_test_split" in out.warnings


def test_train_only_correlation_detection(tmp_path: Path):
    root = tmp_path / "corr"
    _write_feature_audit_ds(root, correlated=True)
    out = fa.run_feature_audit(str(root))
    pair = [p for p in out.correlation_pairs if p.index_a == 0 and p.index_b == 1][0]
    assert pair.abs_corr >= out.config["high_corr_threshold"]


def test_low_variance_detection(tmp_path: Path):
    root = tmp_path / "lv"
    _write_feature_audit_ds(root, low_variance=True)
    out = fa.run_feature_audit(str(root))
    r = [x for x in out.health_records if x.split == "train" and x.feature_index == 2][0]
    assert r.status == "low_variance" and r.low_variance
    assert "low_variance_train" in out.warnings


def test_full_split_stats_not_sample_only(tmp_path: Path):
    root = tmp_path / "sample"
    _write_feature_audit_ds(root)
    out = fa.run_feature_audit(str(root), config=fa.FeatureAuditConfig(max_sample_rows_per_split=1))
    r = [x for x in out.health_records if x.split == "train" and x.feature_index == 0][0]
    assert np.isclose(r.raw_mean, np.mean(np.arange(100)))


def test_feature_subset(tmp_path: Path):
    root = tmp_path / "subset"
    _write_feature_audit_ds(root)
    cols = mf.feature_columns()[:2]
    out = fa.run_feature_audit(str(root), config=fa.FeatureAuditConfig(feature_columns=cols))
    assert {r.feature for r in out.health_records} == set(cols)


def test_single_feature_subset_family_summary_is_valid(tmp_path: Path):
    root = tmp_path / "single_feature"
    _write_feature_audit_ds(root)

    col = mf.feature_columns()[0]
    out = fa.run_feature_audit(
        str(root),
        config=fa.FeatureAuditConfig(feature_columns=(col,)),
    )

    assert {record.feature for record in out.health_records} == {col}
    assert len(out.correlation_pairs) == 0

    train_family_records = [record for record in out.family_records if record.split == "train"]
    assert train_family_records
    for record in train_family_records:
        assert record.train_high_corr_pair_count == 0.0


def test_reversed_feature_columns_are_ordered_by_registry(tmp_path: Path):
    root = tmp_path / "reversed"
    _write_feature_audit_ds(root, correlated=True)

    cols = tuple(reversed(mf.feature_columns()[:2]))
    out = fa.run_feature_audit(
        str(root),
        config=fa.FeatureAuditConfig(feature_columns=cols),
    )

    health_features = [record.feature for record in out.health_records if record.split == "train"]
    assert health_features == list(mf.feature_columns()[:2])

    pair = [record for record in out.correlation_pairs if record.index_a == 0 and record.index_b == 1][0]
    assert pair.feature_a == mf.feature_columns()[0]
    assert pair.feature_b == mf.feature_columns()[1]


def test_train_correlation_stats_rejects_invalid_feature_count():
    with pytest.raises(ValueError):
        fa._StreamingTrainCorrelationStats.empty(0)


def test_streaming_feature_stats_reuses_centered_scratch():
    first = np.array([[1.0, 2.0], [3.0, 8.0]], dtype=np.float64)
    second = np.array([[5.0, 4.0], [7.0, 16.0]], dtype=np.float64)
    stats = fa._StreamingFeatureStats.empty(2)

    stats.update(first)
    scratch = stats._centered_scratch
    assert scratch is not None

    stats.update(second)
    assert stats._centered_scratch is scratch
    expected = np.vstack([first, second])
    np.testing.assert_allclose(stats.mean, expected.mean(axis=0))
    np.testing.assert_allclose(stats.variance(), expected.var(axis=0, ddof=1))


def test_sampling_deterministic(tmp_path: Path):
    root = tmp_path / "deterministic"
    _write_feature_audit_ds(root, train_rows=100, val_rows=100)

    cfg = fa.FeatureAuditConfig(max_sample_rows_per_split=10)
    first = fa.run_feature_audit(str(root), config=cfg)
    second = fa.run_feature_audit(str(root), config=cfg)

    assert first.splits["train"].sample_stride == second.splits["train"].sample_stride
    assert first.as_dict() == second.as_dict()

    first_train = [record for record in first.health_records if record.split == "train" and record.feature_index == 0][0]
    second_train = [record for record in second.health_records if record.split == "train" and record.feature_index == 0][0]
    assert first_train.raw_p50 == second_train.raw_p50


def test_val_drift_does_not_affect_train_correlation(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"

    _write_feature_audit_ds(root_a, correlated=True, val_shift=0.0)
    _write_feature_audit_ds(root_b, correlated=True, val_shift=10_000.0)

    out_a = fa.run_feature_audit(str(root_a))
    out_b = fa.run_feature_audit(str(root_b))

    pairs_a = [(p.index_a, p.index_b, round(p.abs_corr, 12), p.status) for p in out_a.correlation_pairs]
    pairs_b = [(p.index_a, p.index_b, round(p.abs_corr, 12), p.status) for p in out_b.correlation_pairs]
    assert pairs_a == pairs_b
    assert out_a.splits["train"] == out_b.splits["train"]
    assert out_b.splits["val"].drift_count > out_a.splits["val"].drift_count
    assert "distribution_shift:val" in out_b.warnings


def test_val_drift_count_populated(tmp_path: Path):
    root = tmp_path / "drift"
    _write_feature_audit_ds(root, val_shift=10_000.0)

    out = fa.run_feature_audit(str(root))

    drift_records = [record for record in out.drift_records if record.split == "val" and record.status == "distribution_shift"]
    assert out.splits["val"].drift_count == len(drift_records)
    assert out.splits["train"].drift_count == 0


def test_result_config_includes_thresholds(tmp_path: Path):
    root = tmp_path / "config"
    _write_feature_audit_ds(root)

    cfg = fa.FeatureAuditConfig(high_corr_threshold=0.95, min_corr_output_threshold=0.90, drift_mean_z_threshold=2.0)
    out = fa.run_feature_audit(str(root), config=cfg)

    assert out.config["high_corr_threshold"] == 0.95
    assert out.config["min_corr_output_threshold"] == 0.90
    assert out.config["drift_mean_z_threshold"] == 2.0
    assert "drift_std_ratio_low" in out.config
    assert "drift_std_ratio_high" in out.config


def test_dataclass_validation_errors():
    with pytest.raises(ValueError):
        fa.FeatureAuditSplitSummary("train", 10, 10, 11, 1, 2, 0, 0)
    with pytest.raises(ValueError):
        fa.FeatureAuditSplitSummary("train", 10, 11, 10, 1, 2, 0, 0)

    base_split = fa.FeatureAuditSplitSummary("train", 10, 10, 10, 1, 2, 0, 0)
    with pytest.raises(ValueError):
        fa.FeatureAuditResult(
            fa.FEATURE_AUDIT_REPORT_TYPE,
            "a",
            "b",
            "c",
            "d",
            {},
            {"train": "bad", "val": base_split},
            (),
            (),
            (),
            (),
            (),
            {},
            (),
        )
    with pytest.raises(ValueError):
        fa.FeatureAuditResult(
            fa.FEATURE_AUDIT_REPORT_TYPE,
            "a",
            "b",
            "c",
            "d",
            {},
            {"train": base_split, "val": base_split},
            (),
            (),
            (),
            (),
            (),
            {},
            (),
        )
    with pytest.raises(ValueError):
        fa.FeatureHealthRecord(
            "train", "x_a", 0, "s", "o", "f", "u", "t", 0, 1, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, "bad"
        )
    with pytest.raises(ValueError):
        fa.FeatureCorrelationPairRecord("a", "b", 1, 1, "s", "s", "f", "f", 0.1, 0.1, True, True, "moderate_redundancy")


def test_artifact_filename_validation(tmp_path: Path):
    root = tmp_path / "art"
    _write_feature_audit_ds(root)
    out = fa.run_feature_audit(str(root))
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), summary_filename="x.csv")
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), cluster_summary_filename="x.csv")
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), health_filename="x.json")
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), drift_filename="x.json")
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), family_filename="x.json")
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), corr_pairs_filename="x.json")
    with pytest.raises(ValueError):
        fa.write_feature_audit_artifacts(out, str(tmp_path), clusters_filename="x.json")
