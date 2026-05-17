import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("BYBIT_FEATURE_STORAGE_DTYPE", "fp32")

from test_feature_event_result_contract import _install_optional_dependency_stubs

_install_optional_dependency_stubs()

sys.modules.setdefault("torch._inductor", types.ModuleType("torch._inductor"))
sys.modules.setdefault("torch._inductor.config", types.ModuleType("torch._inductor.config"))


class FakeDirectionModel:
    def __init__(self, feature_index: int, offset: float = 0.0):
        self.feature_index = int(feature_index)
        self.offset = float(offset)
        self.coef_ = np.zeros((1, 5), dtype=np.float32)
        self.coef_[0, self.feature_index % 5] = 1.0 + self.offset
        self.intercept_ = np.array([self.offset], dtype=np.float32)

    def decision_function(self, Z):
        return Z[:, self.feature_index % Z.shape[1]] + self.offset


class FakeRegressorModel:
    def __init__(self, value: float):
        self.value = float(value)
        self.coef_ = np.full((5,), self.value, dtype=np.float32)
        self.intercept_ = np.array([self.value], dtype=np.float32)

    def predict(self, Z):
        return np.full(Z.shape[0], self.value, dtype=np.float32)


def make_synthetic(n=240, d=5):
    rng = np.random.default_rng(123)
    Z = rng.normal(size=(n, d)).astype(np.float32)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float32)
    Z[:, 0] = signs + 0.05 * rng.normal(size=n).astype(np.float32)
    mags = (0.2 + (np.arange(n, dtype=np.float32) % 50) / 50.0).reshape(-1, 1)
    scales = np.array([1.0, 1.2, 1.5], dtype=np.float32).reshape(1, 3)
    y = signs.reshape(-1, 1) * mags * scales
    return Z, y.astype(np.float32), np.arange(n, dtype=np.int64)


def write_stage3_manifest(tmp_path: Path, extractor: str, split: str, Z: np.ndarray, y: np.ndarray, positions: np.ndarray):
    out_dir = tmp_path / "stage3_preprocess" / extractor / "default"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_idx, start in enumerate(range(0, Z.shape[0], 41)):
        end = min(Z.shape[0], start + 41)
        path = out_dir / f"{split}_preprocessed_shard_{shard_idx:05d}.npz"
        np.savez_compressed(path, Z=Z[start:end], y=y[start:end], positions=positions[start:end])
        shards.append({"shard": shard_idx, "path": str(path), "rows": int(end - start)})
    manifest = {
        "split": split,
        "stage": "stage3",
        "schema": "linear_preprocess_stage3_v1",
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "n_rows": int(Z.shape[0]),
        "kept_dim": int(Z.shape[1]),
        "summary": {"shape": [int(Z.shape[0]), int(Z.shape[1])], "finite_frac": 1.0},
        "shards": shards,
    }
    manifest_path = out_dir / f"{split}_preprocessed_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_stage3_payload(tmp_path: Path, extractor: str, manifests):
    out_dir = tmp_path / "stage3_preprocess" / extractor / "default"
    payload = {
        "stage": "stage3",
        "status": "ok",
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "extractor": extractor,
        "manifests": manifests,
    }
    (out_dir / "linear_stage3_preprocess_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_trim_stats(tmp_path: Path, y_train: np.ndarray):
    from CMSSL17_offline import compute_signed_raw_stats, save_stats_cache

    stats = compute_signed_raw_stats(y_train)
    save_stats_cache(tmp_path / "linear_signed_side_trim_stats_cache.npz", stats, {"unit_test": True})
    return stats


def write_stage4_payload_and_bundle(tmp_path: Path, extractor: str):
    from CMSSL17 import NUM_HORIZONS
    from CMSSL17_linear import LinearSklearnTakerBundle, save_linear_sklearn_bundle

    stage4_dir = tmp_path / "stage4_models" / extractor / "default" / "sgd_l2_huber"
    stage4_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = stage4_dir / "linear_stage4_best_model.pkl"
    bundle = LinearSklearnTakerBundle(
        "linear_target_models_stage4_v1",
        {},
        [200, 500, 1000],
        [FakeDirectionModel(h, 0.1 * h) for h in range(NUM_HORIZONS)],
        [FakeRegressorModel(0.5 + 0.1 * h) for h in range(NUM_HORIZONS)],
        [FakeRegressorModel(0.4 + 0.1 * h) for h in range(NUM_HORIZONS)],
        1e-4,
        {},
    )
    save_linear_sklearn_bundle(bundle, bundle_path)
    payload = {
        "stage": "stage4",
        "status": "ok",
        "schema": "linear_target_models_stage4_v1",
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "extractor": extractor,
        "preprocess_name": "default",
        "train_split": "train_sample",
        "train_rows": 240,
        "best_alpha": 1e-4,
        "best_model_path": str(bundle_path),
        "best_primary_metric": {"label": "dir_auc_kept_1000ms", "value": 0.75, "guard_passed": True},
        "val_metrics": {"dir_auc_kept": [0.7, 0.71, 0.72], "edge_spearman_q50plus": [0.1, 0.2, 0.3], "primary_metric_value": 0.72},
        "test_metrics": None,
    }
    (stage4_dir / "linear_stage4_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def prepare_extractor(tmp_path: Path, extractor: str):
    Z_train, y_train, pos_train = make_synthetic(n=240)
    Z_val, y_val, pos_val = make_synthetic(n=120)
    manifests = {
        "train_sample": write_stage3_manifest(tmp_path, extractor, "train_sample", Z_train, y_train, pos_train),
        "val": write_stage3_manifest(tmp_path, extractor, "val", Z_val, y_val, pos_val),
        "test": None,
    }
    write_stage3_payload(tmp_path, extractor, manifests)
    write_stage4_payload_and_bundle(tmp_path, extractor)
    return y_train


def configure_stage5(monkeypatch, linear_offline, *, save_predictions=False, strict=False):
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_STRICT", strict)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_REEVALUATE", False)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_RUN_TEST", False)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_BATCH_ROWS", 32)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_MAX_VAL_ROWS", 0)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_TOP_COEFS", 3)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_LABEL_SHIFT_VALUES", [-1, 1])
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_LABEL_PERMUTATION", True)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_PERMUTATION_SEED", 7)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_SAVE_PREDICTIONS", save_predictions)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_PREDICTION_MAX_ROWS", 20)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE5_BASELINE_METRICS_JSON", "")


def test_stage5_comparison_discovers_multiple_extractors(tmp_path, monkeypatch):
    import torch
    import linear_offline

    configure_stage5(monkeypatch, linear_offline)
    y_train = prepare_extractor(tmp_path, "raw_linear")
    prepare_extractor(tmp_path, "hydra")
    write_trim_stats(tmp_path, y_train)

    payload = linear_offline.run_stage5_comparison(
        linear_out_dir=tmp_path,
        extractor_names=["raw_linear", "hydra"],
        preprocess_name="default",
        predictor="sgd_l2_huber",
        device=object(),
    )

    stage5_dir = tmp_path / "stage5_comparison" / "default" / "sgd_l2_huber"
    assert payload["stage"] == "stage5"
    assert len(payload["comparison_rows"]) == 2
    assert (stage5_dir / "linear_stage5_comparison.csv").exists()
    assert (stage5_dir / "linear_stage5_comparison.json").exists()
    assert (tmp_path / "linear_stage5_comparison.csv").exists()


def test_stage5_missing_extractor_skip_vs_strict(tmp_path, monkeypatch):
    import torch
    import linear_offline

    y_train = prepare_extractor(tmp_path, "raw_linear")
    write_trim_stats(tmp_path, y_train)
    configure_stage5(monkeypatch, linear_offline, strict=False)
    payload = linear_offline.run_stage5_comparison(
        linear_out_dir=tmp_path,
        extractor_names=["raw_linear", "missing"],
        preprocess_name="default",
        predictor="sgd_l2_huber",
        device=object(),
    )
    assert payload["extractors_completed"] == ["raw_linear"]

    configure_stage5(monkeypatch, linear_offline, strict=True)
    with pytest.raises(FileNotFoundError):
        linear_offline.run_stage5_comparison(
            linear_out_dir=tmp_path,
            extractor_names=["raw_linear", "missing"],
            preprocess_name="default",
            predictor="sgd_l2_huber",
            device=object(),
        )


def test_stage5_diagnostics_prediction_summary_and_sanity_checks(tmp_path, monkeypatch):
    import torch
    import linear_offline

    configure_stage5(monkeypatch, linear_offline)
    y_train = prepare_extractor(tmp_path, "raw_linear")
    write_trim_stats(tmp_path, y_train)
    payload = linear_offline.run_stage5_comparison(
        linear_out_dir=tmp_path,
        extractor_names=["raw_linear"],
        preprocess_name="default",
        predictor="sgd_l2_huber",
        device=object(),
    )
    diag = payload["diagnostics"]["raw_linear"]
    coef_diag = diag["coefficient_diagnostics"]
    assert "direction" in coef_diag
    assert coef_diag["direction"][0]["top_coefficients"][0].keys() >= {"index", "coef", "abs_coef"}
    assert diag["prediction_summary_val"]["n_rows"] > 0
    assert "label_shift_sanity_val" in diag
    assert "label_permutation_sanity_val" in diag


def test_stage5_prediction_dump(tmp_path, monkeypatch):
    import torch
    import linear_offline

    configure_stage5(monkeypatch, linear_offline, save_predictions=True)
    y_train = prepare_extractor(tmp_path, "raw_linear")
    write_trim_stats(tmp_path, y_train)
    linear_offline.run_stage5_comparison(
        linear_out_dir=tmp_path,
        extractor_names=["raw_linear"],
        preprocess_name="default",
        predictor="sgd_l2_huber",
        device=object(),
    )
    dump_path = tmp_path / "stage5_comparison" / "default" / "sgd_l2_huber" / "stage5_diagnostics" / "raw_linear" / "val_predictions.npz"
    assert dump_path.exists()
    with np.load(dump_path) as arr:
        assert set(arr.files) >= {"positions", "y", "dir_logits", "p_up", "mag_up_sqrt", "mag_down_sqrt", "edge_bps"}
        assert arr["y"].shape[0] == 20


def test_stage5_decision_stride_mismatch_skip_vs_strict(tmp_path, monkeypatch):
    import linear_offline

    y_train = prepare_extractor(tmp_path, "raw_linear")
    write_trim_stats(tmp_path, y_train)
    stage4_path = tmp_path / "stage4_models" / "raw_linear" / "default" / "sgd_l2_huber" / "linear_stage4_metrics.json"
    payload = json.loads(stage4_path.read_text())
    payload["decision_stride_rows"] = 1
    stage4_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    configure_stage5(monkeypatch, linear_offline, strict=False)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)
    result = linear_offline.run_stage5_comparison(
        linear_out_dir=tmp_path,
        extractor_names=["raw_linear"],
        preprocess_name="default",
        predictor="sgd_l2_huber",
        device=object(),
    )
    assert result["extractors_completed"] == []

    configure_stage5(monkeypatch, linear_offline, strict=True)
    with pytest.raises(ValueError, match="decision-row mismatch"):
        linear_offline.run_stage5_comparison(
            linear_out_dir=tmp_path,
            extractor_names=["raw_linear"],
            preprocess_name="default",
            predictor="sgd_l2_huber",
            device=object(),
        )
