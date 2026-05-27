from pathlib import Path

import numpy as np
import pytest

from mmrt.analysis import preprocess_audit as pa
from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def _write_ds(root: Path, train_rows=20, val_rows=20, test_rows=0, shift=0.0, clip=False, constant=False):
    writer = wr.DecisionRowWriter(
        wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(root), chunk_rows=16)
    )
    n_rows = train_rows + val_rows + test_rows
    for i in range(n_rows):
        ts = 1_000_000 + i * 1_000
        x0 = 0.1 * i if i < train_rows else (0.1 * i + shift)
        if clip and i >= train_rows:
            x0 = 1_000.0
        features = [x0] + [float((i + j) % 3) for j in range(specs.FEATURE_COUNT - 1)]
        if constant:
            features[1] = 1.0
        writer.append_values(
            decision_index=i + 1,
            ts_us=ts,
            local_ts_us=ts,
            event_seq=i,
            raw_mid=100.0,
            label_entry_ts_us=ts,
            label_values=(1.0, 1.0, 1.0),
            feature_values=tuple(features),
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
                1_000_000 + n_rows * 1_000 + 1,
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


def test_train_only_fit_no_leakage(tmp_path: Path):
    train_rows = 20
    first = tmp_path / "a"
    _write_ds(first, train_rows=train_rows, shift=100.0)
    out = pa.run_preprocess_audit(str(first))

    assert out.as_dict()["preprocess_state_summary"]["n_rows_fit"] == train_rows
    expected_train_mean = np.mean([0.1 * i for i in range(train_rows)])
    assert np.isclose(out.preprocess_state["mean"][0], expected_train_mean)

    train_feature = [x for x in out.feature_records if x.split == "train" and x.feature_index == 0][0]
    val_feature = [x for x in out.feature_records if x.split == "val" and x.feature_index == 0][0]
    assert abs(train_feature.drift_mean_z) < 1e-6
    assert abs(val_feature.drift_mean_z) > 1.0

    second = tmp_path / "b"
    _write_ds(second, train_rows=train_rows, shift=1_000.0)
    out_shifted = pa.run_preprocess_audit(str(second))
    assert np.isclose(out_shifted.preprocess_state["mean"][0], expected_train_mean)


def test_clip_detection(tmp_path: Path):
    root = tmp_path / "clip"
    _write_ds(root, clip=True)
    out = pa.run_preprocess_audit(str(root))

    v0 = [x for x in out.feature_records if x.split == "val" and x.feature_index == 0][0]
    assert v0.clip_total_rate > 0
    assert v0.status in {"clip_review", "clip_excessive"}
    assert v0.recommendation in {"review_clip_z", "review_clip_z_and_drift"}


def test_inactive_detection(tmp_path: Path):
    root = tmp_path / "inactive"
    _write_ds(root, constant=True)
    out = pa.run_preprocess_audit(str(root))

    rec = [x for x in out.feature_records if x.split == "train" and x.feature_index == 1][0]
    assert rec.active is False
    assert rec.status == "inactive"
    assert rec.recommendation == "review_variance_floor"
    assert out.splits["train"].inactive_count > 0
    assert "inactive_features_present" in out.warnings


def test_sampling_deterministic(tmp_path: Path):
    root = tmp_path / "sampling"
    _write_ds(root, train_rows=100, val_rows=100)
    cfg = pa.PreprocessAuditConfig(max_sample_rows_per_split=10)
    first = pa.run_preprocess_audit(str(root), config=cfg)
    second = pa.run_preprocess_audit(str(root), config=cfg)
    assert first.splits["train"].sample_stride == second.splits["train"].sample_stride
    assert first.as_dict() == second.as_dict()


def test_counts_full_split_not_sampled(tmp_path: Path):
    root = tmp_path / "counts"
    _write_ds(root, train_rows=100, val_rows=100, clip=True)
    cfg = pa.PreprocessAuditConfig(max_sample_rows_per_split=1)
    out = pa.run_preprocess_audit(str(root), config=cfg)
    v0 = [x for x in out.feature_records if x.split == "val" and x.feature_index == 0][0]
    assert v0.clip_total_count == 100


def test_artifact_validation(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_ds(root)
    out = pa.run_preprocess_audit(str(root))

    with pytest.raises(ValueError):
        pa.write_preprocess_audit_artifacts(out, str(tmp_path), summary_filename="bad.txt")

    with pytest.raises(ValueError):
        pa.write_preprocess_audit_artifacts(out, str(tmp_path), features_filename="bad.json")


def test_result_json_excludes_feature_records(tmp_path: Path):
    root = tmp_path / "json"
    _write_ds(root)
    out = pa.run_preprocess_audit(str(root))
    payload = out.as_dict()
    assert "feature_records" not in payload
