import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("BYBIT_FEATURE_STORAGE_DTYPE", "fp32")

from test_feature_event_result_contract import _install_optional_dependency_stubs

_install_optional_dependency_stubs()

sys.modules.setdefault("torch._inductor", types.ModuleType("torch._inductor"))
sys.modules.setdefault("torch._inductor.config", types.ModuleType("torch._inductor.config"))


class FakeDataset:
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def close(self):
        pass


class FakePreprocessBundle:
    original_dim = 4
    kept_dim = 3

    def __init__(self):
        self.fit_summary = {
            "fit_mode": "streaming_full_train_v1",
            "fit_split": "train_full",
            "mean_std_fit_rows": 20,
            "quantile_fit_rows": 8,
        }


def _summary(rows=4, dim=3):
    return {"summary": {"shape": [rows, dim], "finite_frac": 1.0}, "_audit_full_summary": None}


def configure_stage3(monkeypatch, linear_offline, *, audit=False):
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_SPLIT", "train_full")
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT", audit)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_WINSORIZE", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_WINSOR_Q_LO", 0.0)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_WINSOR_Q_HI", 1.0)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_STANDARDIZE", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_VARIANCE_FILTER", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_MIN_STD", 1e-6)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_STD_EPS", 1e-6)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_POST_CLIP_ABS", 0.0)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_NONFINITE_POLICY", "raise")
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_MAX_ROWS", 1000)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB", 64)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS", 200_000)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS", 200_000)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS", 200_000)
    monkeypatch.setattr(linear_offline, "LINEAR_EXTRACT_BATCH_ROWS", 4)
    monkeypatch.setattr(linear_offline, "LINEAR_RUN_TEST", True)
    monkeypatch.setattr(linear_offline, "OUT_ROOT", "/tmp/fake-out-root")


def install_stage3_fakes(monkeypatch, linear_offline, tmp_path: Path, *, has_test=False):
    plan = {"train_split_entries": [{}], "train_week_keys": ["w0"], "has_cmssl_test": has_test}
    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda out_root: plan)
    monkeypatch.setattr(
        linear_offline,
        "load_stage2_extractor_bundle",
        lambda **kwargs: (
            object(),
            {
                "payload_path": str(tmp_path / "stage2.json"),
                "decision_stride_rows": linear_offline.LINEAR_DECISION_STRIDE_ROWS,
                "decision_offset_rows": linear_offline.LINEAR_DECISION_OFFSET_ROWS,
                "decision_row_policy": "linear_every_n_rows_v1",
            },
        ),
    )
    monkeypatch.setattr(linear_offline, "fit_linear_preprocessor_streaming_from_plan", lambda **kwargs: FakePreprocessBundle())
    monkeypatch.setattr(linear_offline, "audit_preprocessing_streaming_train_plan", lambda **kwargs: _summary(rows=10))
    monkeypatch.setattr(linear_offline, "_audit_stream_split", lambda *args, **kwargs: _summary(rows=4))
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: FakeDataset(8))
    monkeypatch.setattr(linear_offline, "build_test_dataset_from_plan", lambda plan: FakeDataset(6) if has_test else None)

    def fake_save(bundle, path):
        del bundle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"bundle")

    monkeypatch.setattr(linear_offline, "save_linear_preprocess_bundle", fake_save)
    return plan


def test_stage3_records_train_full_fit_metadata(monkeypatch, tmp_path):
    import linear_offline

    configure_stage3(monkeypatch, linear_offline)
    install_stage3_fakes(monkeypatch, linear_offline, tmp_path)

    payload = linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")

    assert payload["streaming_features"] is True
    assert payload["persisted_preprocessed_shards"] is False
    assert payload["preprocess_config"]["fit_split"] == "train_full"
    assert payload["fit_summary"]["fit_mode"] == "streaming_full_train_v1"
    assert payload["manifests"] == {}
    assert Path(payload["preprocess_bundle_path"]).exists()


def test_stage3_writes_audit_summary_files(monkeypatch, tmp_path):
    import linear_offline

    configure_stage3(monkeypatch, linear_offline, audit=True)
    install_stage3_fakes(monkeypatch, linear_offline, tmp_path)
    monkeypatch.setattr(linear_offline, "write_preprocess_audit_csv", lambda path, summaries: Path(path).write_text("split\n"))
    monkeypatch.setattr(linear_offline, "write_preprocess_top_features_csv", lambda path, summaries: Path(path).write_text("feature\n"))

    payload = linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")

    assert payload["audit_enabled"] is True
    assert Path(payload["audit_summary_path"]).exists()
    assert Path(payload["audit_csv_path"]).exists()
    assert Path(payload["audit_top_features_csv_path"]).exists()


def test_stage3_audit_uses_bounded_max_rows(monkeypatch, tmp_path):
    import linear_offline

    configure_stage3(monkeypatch, linear_offline, audit=True)
    install_stage3_fakes(monkeypatch, linear_offline, tmp_path, has_test=True)
    seen = {}

    def fake_train_audit(**kwargs):
        seen["train_max_rows"] = kwargs.get("max_rows")
        return _summary(rows=200_000)

    def fake_split_audit(*args, **kwargs):
        split = kwargs.get("split_name", args[3] if len(args) > 3 else "unknown")
        seen[f"{split}_max_rows"] = kwargs.get("max_rows")
        return _summary(rows=200_000)

    monkeypatch.setattr(linear_offline, "audit_preprocessing_streaming_train_plan", fake_train_audit)
    monkeypatch.setattr(linear_offline, "_audit_stream_split", fake_split_audit)
    monkeypatch.setattr(linear_offline, "write_preprocess_audit_csv", lambda path, summaries: Path(path).write_text("split\n"))
    monkeypatch.setattr(linear_offline, "write_preprocess_top_features_csv", lambda path, summaries: Path(path).write_text("feature\n"))

    payload = linear_offline.run_stage3_preprocessing(
        linear_out_dir=tmp_path, extractor_name="raw_linear"
    )

    assert seen == {
        "train_max_rows": 200_000,
        "val_max_rows": 200_000,
        "test_max_rows": 200_000,
    }
    assert payload["audit_max_train_rows"] == 200_000
    assert payload["audit_max_val_rows"] == 200_000
    assert payload["audit_max_test_rows"] == 200_000
    assert payload["preprocess_config"]["audit_max_train_rows"] == 200_000
    assert payload["preprocess_config"]["audit_max_val_rows"] == 200_000
    assert payload["preprocess_config"]["audit_max_test_rows"] == 200_000
    assert payload["audit_summary"]["audit_max_train_rows"] == 200_000


def test_stage3_audit_zero_max_rows_requests_full_audit(monkeypatch, tmp_path):
    import linear_offline

    configure_stage3(monkeypatch, linear_offline, audit=True)
    install_stage3_fakes(monkeypatch, linear_offline, tmp_path, has_test=True)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS", 0)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS", 0)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS", 0)
    seen = {}

    def fake_train_audit(**kwargs):
        seen["train_max_rows"] = kwargs.get("max_rows")
        return _summary(rows=10)

    monkeypatch.setattr(linear_offline, "audit_preprocessing_streaming_train_plan", fake_train_audit)

    def fake_split_audit(*args, **kwargs):
        split = args[3]
        seen[f"{split}_max_rows"] = kwargs.get("max_rows")
        return _summary(rows=4)

    monkeypatch.setattr(linear_offline, "_audit_stream_split", fake_split_audit)
    monkeypatch.setattr(linear_offline, "write_preprocess_audit_csv", lambda path, summaries: Path(path).write_text("split\n"))
    monkeypatch.setattr(linear_offline, "write_preprocess_top_features_csv", lambda path, summaries: Path(path).write_text("feature\n"))

    payload = linear_offline.run_stage3_preprocessing(
        linear_out_dir=tmp_path, extractor_name="raw_linear"
    )

    assert seen["train_max_rows"] == 0
    assert seen["val_max_rows"] == 0
    assert seen["test_max_rows"] == 0
    assert payload["audit_max_train_rows"] == 0
    assert payload["preprocess_config"]["audit_max_val_rows"] == 0


@pytest.mark.parametrize(
    ("attr", "env_name"),
    [
        ("LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS", "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS"),
        ("LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS", "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS"),
        ("LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS", "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS"),
    ],
)
def test_stage3_audit_negative_caps_are_validated(monkeypatch, tmp_path, attr, env_name):
    import linear_offline

    configure_stage3(monkeypatch, linear_offline, audit=True)
    monkeypatch.setattr(linear_offline, attr, -1)

    with pytest.raises(ValueError, match=f"{env_name} must be >= 0"):
        linear_offline.run_stage3_preprocessing(
            linear_out_dir=tmp_path, extractor_name="raw_linear"
        )


def test_stage3_rejects_non_train_full_fit_split(monkeypatch, tmp_path):
    import linear_offline

    configure_stage3(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_SPLIT", "train_sample")

    with pytest.raises(ValueError, match="train_full"):
        linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")


def test_streaming_stats_abs_sample_max_defined():
    import numpy as np
    import linear_offline

    stats = linear_offline._empty_streaming_stats()
    Z = np.asarray([[1.0, -2.0], [3.0, 0.0]], dtype=np.float32)

    linear_offline._update_streaming_stats(stats, Z)
    summary = linear_offline._finalize_streaming_summary(
        stats,
        n_shards=1,
        chunk_rows=2,
        positions_rows=2,
    )

    assert summary["shape"] == [2, 2]
    assert summary["finite_frac"] == 1.0
    assert "abs_p95" in summary
