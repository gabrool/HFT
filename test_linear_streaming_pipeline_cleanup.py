import json
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
    def __init__(self, n):
        self.n = n
        import numpy as np
        self.y = np.linspace(-1.0, 1.0, n * 3, dtype=np.float32).reshape(n, 3)

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


class FakeModelBundle:
    def __init__(self, alpha=1e-4):
        self.config = {"alpha": alpha}
        self.fit_summary = {"train_rows": 10}


def _stage3_payload(tmp_path: Path, **overrides):
    payload = {
        "stage": "stage3",
        "status": "ok",
        "schema": "linear_preprocess_stage3_v1",
        "streaming_features": True,
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "preprocess_bundle_path": str(tmp_path / "bundle.npz"),
    }
    payload.update(overrides)
    return payload


def test_stage4_accepts_train_full_default_streaming(monkeypatch, tmp_path):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_TRAIN_SPLIT", "train_full")
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_ALPHA_VALUES", [1e-4])
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_RUN_TEST", False)
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))
    plan = {"train_split_entries": [{}, {}], "train_week_keys": ["w0", "w1"], "has_cmssl_test": False}
    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda out_root: plan)
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: FakeDataset(8))
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: 5)
    monkeypatch.setattr(linear_offline, "split_decision_row_count_from_plan", lambda plan, split_name, max_rows=0: 2)
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(linear_offline, "load_stage3_payload", lambda *args, **kwargs: _stage3_payload(tmp_path, payload_path="stage3.json"))
    monkeypatch.setattr(linear_offline, "load_linear_preprocess_bundle", lambda path: FakePreprocessBundle())
    monkeypatch.setattr(linear_offline, "load_linear_trim_stats", lambda linear_out_dir: {})
    monkeypatch.setattr(
        linear_offline,
        "train_stage4_candidates_streaming_from_plan",
        lambda **kwargs: [FakeModelBundle(alpha=1e-4)],
    )
    monkeypatch.setattr(
        linear_offline,
        "evaluate_stage4_bundle_streaming",
        lambda **kwargs: {"primary_metric_value": 0.25, "primary_metric_label": "unit", "primary_metric_guard_passed": True},
    )
    monkeypatch.setattr(linear_offline, "compute_primary_metric", lambda metrics: (metrics["primary_metric_value"], metrics["primary_metric_label"]))

    def fake_save(bundle, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"model")

    monkeypatch.setattr(linear_offline, "save_linear_sklearn_bundle", fake_save)

    payload = linear_offline.run_stage4_training(
        linear_out_dir=tmp_path,
        extractor_name="raw_linear",
        preprocess_name="default",
        device=object(),
    )

    assert payload["train_split"] == "train_full"
    assert payload["stage4_config"]["schema"] == linear_offline.LINEAR_STAGE4_SCHEMA
    assert Path(payload["best_model_path"]).exists()


def test_stage4_rejects_stale_stage3_decision_metadata(monkeypatch, tmp_path):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_TRAIN_SPLIT", "train_full")
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))
    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda out_root: {"train_split_entries": [{}], "train_week_keys": ["w0"], "has_cmssl_test": False})
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(linear_offline, "load_stage3_payload", lambda *args, **kwargs: _stage3_payload(tmp_path, decision_stride_rows=1))

    with pytest.raises(ValueError, match="decision-row mismatch"):
        linear_offline.run_stage4_training(
            linear_out_dir=tmp_path,
            extractor_name="raw_linear",
            preprocess_name="default",
            device=object(),
        )


def test_stage3_records_train_full_fit_metadata(monkeypatch, tmp_path):
    import linear_offline

    bundle = FakePreprocessBundle()
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_SPLIT", "train_full")
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_AUDIT", True)
    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda out_root: {"train_split_entries": [{}], "train_week_keys": ["week0"], "has_cmssl_test": False})
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: FakeDataset(8))
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(linear_offline, "fit_linear_preprocessor_streaming_from_plan", lambda **kwargs: bundle)
    monkeypatch.setattr(linear_offline, "save_linear_preprocess_bundle", lambda bundle, path: Path(path).write_bytes(b"bundle"))
    monkeypatch.setattr(
        linear_offline,
        "_audit_stream_split",
        lambda *args, **kwargs: {"summary": {"shape": [5, 3], "finite_frac": 1.0}, "_audit_full_summary": None},
    )
    monkeypatch.setattr(linear_offline, "audit_preprocessing_streaming_train_plan", lambda **kwargs: {"summary": {"shape": [5, 3], "finite_frac": 1.0}, "_audit_full_summary": None})
    monkeypatch.setattr(linear_offline, "write_preprocess_audit_csv", lambda *args, **kwargs: Path(args[0]).write_text(""))
    monkeypatch.setattr(linear_offline, "write_preprocess_top_features_csv", lambda *args, **kwargs: Path(args[0]).write_text(""))

    payload = linear_offline.run_stage3_preprocessing(
        linear_out_dir=tmp_path,
        extractor_name="raw_linear",
    )

    assert payload["preprocess_config"]["fit_split"] == "train_full"
    assert payload["fit_summary"]["fit_split"] == "train_full"
    assert payload["fit_summary"]["fit_mode"] == "streaming_full_train_v1"


def test_main_guard_requires_out_root_not_linear_out_dir():
    text = Path("linear_offline.py").read_text(encoding="utf-8")
    assert 'assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"' in text
    assert "OUT_ROOT or LINEAR_OUT_DIR" not in text


def test_legacy_persisted_shard_env_validation_removed_from_active_stage_blocks():
    text = Path("linear_offline.py").read_text(encoding="utf-8")
    assert "BYBIT_LINEAR_TRANSFORM_SHARD_ROWS must be > 0" not in text
    assert "BYBIT_LINEAR_MAX_X_CHUNK_MB must be > 0" not in text
    assert "BYBIT_LINEAR_MAX_Z_CHUNK_MB must be > 0" not in text
    assert "BYBIT_LINEAR_TRANSFORM_SAVE_FORMAT must be 'npz_shards'" not in text
    stage3_block = text.split('if LINEAR_STAGE == "stage3":', 1)[1].split('if LINEAR_STAGE == "stage4":', 1)[0]
    assert "LINEAR_PREPROCESS_SHARD_ROWS <= 0" not in stage3_block
    assert "LINEAR_PREPROCESS_MAX_Z_CHUNK_MB <= 0" not in stage3_block



def test_collect_train_labels_from_plan_builds_one_week_at_a_time(monkeypatch):
    import linear_offline

    live = {"count": 0, "max": 0}

    class CountedDataset(FakeDataset):
        def __init__(self, n):
            super().__init__(n)
            live["count"] += 1
            live["max"] = max(live["max"], live["count"])

        def close(self):
            live["count"] -= 1

    plan = {"train_split_entries": [{}, {}], "train_week_keys": ["w0", "w1"]}
    monkeypatch.setattr(linear_offline, "build_train_week_dataset", lambda plan, week_index: CountedDataset(10))
    y = linear_offline.collect_train_labels_from_plan(plan)
    assert y.shape[0] == 4
    assert live["max"] <= 1
    assert live["count"] == 0


def test_train_plan_iterator_releases_each_week(monkeypatch):
    import numpy as np
    import linear_offline

    live = {"count": 0, "max": 0}

    class CountedDataset(FakeDataset):
        def __init__(self, n):
            super().__init__(n)
            live["count"] += 1
            live["max"] = max(live["max"], live["count"])

        def close(self):
            live["count"] -= 1

    plan = {"train_split_entries": [{}, {}], "train_week_keys": ["w0", "w1"]}
    monkeypatch.setattr(linear_offline, "build_train_week_dataset", lambda plan, week_index: CountedDataset(10))
    monkeypatch.setattr(linear_offline, "collect_windows_for_positions", lambda ds, pos, batch_rows, split_name: (np.zeros((len(pos), 2, 4), dtype=np.float32), ds.y[pos]))
    rows = 0
    for X, y, pos in linear_offline.iter_train_week_window_batches_from_plan(plan=plan, batch_rows=2, max_rows=0):
        rows += y.shape[0]
    assert rows == 4
    assert live["max"] <= 1
    assert live["count"] == 0
