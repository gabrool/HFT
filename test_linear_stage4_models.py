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


def make_synthetic(n=240, d=5):
    rng = np.random.default_rng(123)
    Z = rng.normal(size=(n, d)).astype(np.float32)
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float32)
    Z[:, 0] = signs + 0.05 * rng.normal(size=n).astype(np.float32)
    mags = (0.2 + (np.arange(n, dtype=np.float32) % 50) / 50.0).reshape(-1, 1)
    scales = np.array([1.0, 1.2, 1.5], dtype=np.float32).reshape(1, 3)
    y = signs.reshape(-1, 1) * mags * scales
    return Z, y.astype(np.float32), np.arange(n, dtype=np.int64)


def write_fake_stage3_manifest(tmp_path: Path, split: str, Z: np.ndarray, y: np.ndarray, positions: np.ndarray, shard_rows: int = 37):
    out_dir = tmp_path / "stage3_preprocess" / "raw_linear" / "default"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_idx, start in enumerate(range(0, Z.shape[0], shard_rows)):
        end = min(Z.shape[0], start + shard_rows)
        path = out_dir / f"{split}_preprocessed_shard_{shard_idx:05d}.npz"
        np.savez_compressed(path, Z=Z[start:end].astype(np.float32), y=y[start:end].astype(np.float32), positions=positions[start:end])
        shards.append({
            "shard": shard_idx,
            "path": str(path),
            "rows": int(end - start),
            "z_shape": [int(end - start), int(Z.shape[1])],
            "y_shape": [int(end - start), int(y.shape[1])],
        })
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
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def write_fake_stage3_payload(tmp_path: Path, manifests):
    out_dir = tmp_path / "stage3_preprocess" / "raw_linear" / "default"
    payload = {
        "stage": "stage3",
        "status": "ok",
        "schema": "linear_preprocess_stage3_v1",
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "extractor": "raw_linear",
        "manifests": manifests,
    }
    (out_dir / "linear_stage3_preprocess_metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def write_trim_stats(tmp_path: Path, y_train: np.ndarray):
    from CMSSL17_offline import compute_signed_raw_stats, save_stats_cache

    stats = compute_signed_raw_stats(y_train)
    save_stats_cache(
        tmp_path / "linear_signed_side_trim_stats_cache.npz",
        stats,
        {
            "unit_test": True,
            "decision_stride_rows": 5,
            "decision_offset_rows": 0,
            "decision_row_policy": "linear_every_n_rows_v1",
        },
    )
    return stats


def configure_stage4(monkeypatch, linear_offline):
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_ALPHA_VALUES", [1e-4])
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_EPOCHS", 1)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_BATCH_ROWS", 32)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_RANDOM_SEED", 7)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_PENALTY", "l2")
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_L1_RATIO", 0.15)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_DIRECTION_WEIGHTING", "tempered")
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING", "none")
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_RUN_TEST", True)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_MAX_VAL_ROWS", 0)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_MAX_TEST_ROWS", 0)


def test_bundle_prediction_schema():
    from CMSSL17 import NUM_HORIZONS
    from CMSSL17_linear import LinearSklearnTakerBundle

    class FakeDirection:
        def __init__(self, offset):
            self.offset = offset

        def decision_function(self, Z):
            return Z[:, 0] + self.offset

    class FakeRegressor:
        def __init__(self, value):
            self.value = value

        def predict(self, Z):
            return np.full(Z.shape[0], self.value, dtype=np.float32)

    Z, _y, _ = make_synthetic(n=24, d=4)
    direction_models = [FakeDirection(float(h)) for h in range(NUM_HORIZONS)]
    mag_up_models = [FakeRegressor(0.2 + h) for h in range(NUM_HORIZONS)]
    mag_down_models = [FakeRegressor(0.3 + h) for h in range(NUM_HORIZONS)]
    bundle = LinearSklearnTakerBundle("linear_target_models_stage4_v1", {}, [1, 2, 3], direction_models, mag_up_models, mag_down_models, 1e-4, {})
    pred = bundle.predict_dict_np(Z[:11])
    assert set(pred) == {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
    assert pred["dir_logits"].shape == (11, NUM_HORIZONS)
    assert np.isfinite(pred["dir_logits"]).all()
    assert (pred["mag_up_sqrt"] > 0).all()
    assert (pred["mag_down_sqrt"] > 0).all()


def test_train_stage4_candidate_fits_all_models(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import linear_offline
    from CMSSL17 import NUM_HORIZONS

    configure_stage4(monkeypatch, linear_offline)
    Z, y, pos = make_synthetic()
    train_manifest = write_fake_stage3_manifest(tmp_path, "train_sample", Z, y, pos)
    stats = write_trim_stats(tmp_path, y)
    config = {
        "schema": linear_offline.LINEAR_STAGE4_SCHEMA,
        "penalty": "l2",
        "l1_ratio": 0.15,
        "epochs": 1,
        "batch_rows": 32,
        "random_state": 7,
        "direction_weighting": "tempered",
        "mag_floor": 1e-4,
    }
    bundle = linear_offline.train_stage4_candidate(train_manifest=train_manifest, stats=stats, alpha=1e-4, config=config)
    assert len(bundle.direction_models) == NUM_HORIZONS
    assert len(bundle.mag_up_models) == NUM_HORIZONS
    assert len(bundle.mag_down_models) == NUM_HORIZONS


def test_stage4_end_to_end_fake_run(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import torch
    if not hasattr(torch, "device"):
        pytest.skip("real torch is not installed")
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    Z_train, y_train, pos_train = make_synthetic(n=240)
    Z_val, y_val, pos_val = make_synthetic(n=120)
    Z_test, y_test, pos_test = make_synthetic(n=120)
    manifests = {
        "train_sample": write_fake_stage3_manifest(tmp_path, "train_sample", Z_train, y_train, pos_train),
        "val": write_fake_stage3_manifest(tmp_path, "val", Z_val, y_val, pos_val),
        "test": write_fake_stage3_manifest(tmp_path, "test", Z_test, y_test, pos_test),
    }
    write_fake_stage3_payload(tmp_path, manifests)
    write_trim_stats(tmp_path, y_train)

    payload = linear_offline.run_stage4_training(
        linear_out_dir=tmp_path,
        extractor_name="raw_linear",
        preprocess_name="default",
        device=torch.device("cpu"),
    )
    assert payload["stage"] == "stage4"
    assert payload["train_split"] == "train_sample"
    assert Path(payload["best_model_path"]).exists()
    assert "val_metrics" in payload
    assert (tmp_path / "linear_stage4_metrics.json").exists()


def test_stage4_training_uses_train_manifest_not_validation(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import torch
    if not hasattr(torch, "device"):
        pytest.skip("real torch is not installed")
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    Z_train, y_train, pos_train = make_synthetic(n=240)
    Z_val, y_val, pos_val = make_synthetic(n=120)
    manifests = {
        "train_sample": write_fake_stage3_manifest(tmp_path, "train_sample", Z_train, y_train, pos_train),
        "val": write_fake_stage3_manifest(tmp_path, "val", Z_val, -y_val, pos_val),
        "test": None,
    }
    write_fake_stage3_payload(tmp_path, manifests)
    write_trim_stats(tmp_path, y_train)

    seen = {}
    real_train = linear_offline.train_stage4_candidate

    def wrapped_train_stage4_candidate(*, train_manifest, stats, alpha, config):
        seen["path"] = train_manifest.get("manifest_path")
        return real_train(train_manifest=train_manifest, stats=stats, alpha=alpha, config=config)

    monkeypatch.setattr(linear_offline, "train_stage4_candidate", wrapped_train_stage4_candidate)
    linear_offline.run_stage4_training(linear_out_dir=tmp_path, extractor_name="raw_linear", preprocess_name="default", device=torch.device("cpu"))
    assert seen["path"].endswith("train_sample_preprocessed_manifest.json")


def test_stage4_missing_trim_stats_fails(tmp_path, monkeypatch):
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    Z_train, y_train, pos_train = make_synthetic(n=240)
    Z_val, y_val, pos_val = make_synthetic(n=120)
    manifests = {
        "train_sample": write_fake_stage3_manifest(tmp_path, "train_sample", Z_train, y_train, pos_train),
        "val": write_fake_stage3_manifest(tmp_path, "val", Z_val, y_val, pos_val),
        "test": None,
    }
    write_fake_stage3_payload(tmp_path, manifests)
    with pytest.raises(FileNotFoundError, match="Missing linear trim stats cache"):
        linear_offline.run_stage4_training(linear_out_dir=tmp_path, extractor_name="raw_linear", preprocess_name="default", device=object())


def test_load_linear_trim_stats_rejects_decision_stride_mismatch(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_offline import compute_signed_raw_stats, save_stats_cache

    y = np.tile(
        np.array(
            [
                [1.0, -1.0, 2.0],
                [-1.0, 1.0, -2.0],
                [2.0, -2.0, 3.0],
                [-2.0, 2.0, -3.0],
            ],
            dtype=np.float32,
        ),
        (60, 1),
    )

    stats = compute_signed_raw_stats(y)

    # Deliberately stale/wrong cache metadata.
    save_stats_cache(
        tmp_path / "linear_signed_side_trim_stats_cache.npz",
        stats,
        {
            "decision_stride_rows": 1,
            "decision_offset_rows": 0,
            "decision_row_policy": "linear_every_n_rows_v1",
        },
    )

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    with pytest.raises(ValueError, match="Trim stats cache decision-row mismatch"):
        linear_offline.load_linear_trim_stats(tmp_path)


def test_load_linear_trim_stats_rejects_decision_offset_mismatch(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_offline import compute_signed_raw_stats, save_stats_cache

    _Z, y, _pos = make_synthetic(n=240)
    stats = compute_signed_raw_stats(y)
    save_stats_cache(
        tmp_path / "linear_signed_side_trim_stats_cache.npz",
        stats,
        {
            "decision_stride_rows": 5,
            "decision_offset_rows": 1,
            "decision_row_policy": "linear_every_n_rows_v1",
        },
    )

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    with pytest.raises(ValueError, match="Trim stats cache decision-row mismatch"):
        linear_offline.load_linear_trim_stats(tmp_path)


def test_load_linear_trim_stats_rejects_decision_row_policy_mismatch(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_offline import compute_signed_raw_stats, save_stats_cache

    _Z, y, _pos = make_synthetic(n=240)
    stats = compute_signed_raw_stats(y)
    save_stats_cache(
        tmp_path / "linear_signed_side_trim_stats_cache.npz",
        stats,
        {
            "decision_stride_rows": 5,
            "decision_offset_rows": 0,
            "decision_row_policy": "legacy_policy",
        },
    )

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    with pytest.raises(ValueError, match="Trim stats cache decision_row_policy mismatch"):
        linear_offline.load_linear_trim_stats(tmp_path)


def test_stage4_rejects_decision_stride_mismatch(tmp_path, monkeypatch):
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)
    Z_train, y_train, pos_train = make_synthetic(n=240)
    Z_val, y_val, pos_val = make_synthetic(n=120)
    train_manifest = write_fake_stage3_manifest(tmp_path, "train_sample", Z_train, y_train, pos_train)
    train_manifest["decision_stride_rows"] = 1
    Path(train_manifest["manifest_path"]).write_text(json.dumps(train_manifest, indent=2))
    manifests = {
        "train_sample": train_manifest,
        "val": write_fake_stage3_manifest(tmp_path, "val", Z_val, y_val, pos_val),
        "test": None,
    }
    write_fake_stage3_payload(tmp_path, manifests)
    write_trim_stats(tmp_path, y_train)

    with pytest.raises(ValueError, match="decision-row mismatch"):
        linear_offline.run_stage4_training(
            linear_out_dir=tmp_path,
            extractor_name="raw_linear",
            preprocess_name="default",
            device=object(),
        )
