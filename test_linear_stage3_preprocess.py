import json
import os
import sys
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("BYBIT_FEATURE_STORAGE_DTYPE", "fp32")

from test_feature_event_result_contract import _install_optional_dependency_stubs

_install_optional_dependency_stubs()

sys.modules.setdefault("torch._inductor", types.ModuleType("torch._inductor"))
sys.modules.setdefault("torch._inductor.config", types.ModuleType("torch._inductor.config"))


def write_fake_stage2_manifest(tmp_path: Path, split: str, Z: np.ndarray, y: np.ndarray, positions: np.ndarray, shard_rows: int = 5):
    out_dir = tmp_path / "stage2_extractors" / "raw_linear"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard_idx, start in enumerate(range(0, Z.shape[0], shard_rows)):
        end = min(Z.shape[0], start + shard_rows)
        path = out_dir / f"{split}_transform_shard_{shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            Z=Z[start:end].astype(np.float32),
            y=y[start:end].astype(np.float32),
            positions=positions[start:end].astype(np.int64),
        )
        shards.append(
            {
                "shard": shard_idx,
                "path": str(path),
                "rows": int(end - start),
                "z_shape": [int(end - start), int(Z.shape[1])],
                "y_shape": list(y[start:end].shape),
                "positions_start": int(positions[start]),
                "positions_end": int(positions[end - 1]),
            }
        )
    summary = {
        "shape": [int(Z.shape[0]), int(Z.shape[1])],
        "dtype": "float32",
        "finite_frac": 1.0,
        "mean": float(Z.mean()),
        "std": float(Z.std()),
        "n_shards": len(shards),
        "chunk_rows": shard_rows,
        "positions_rows": int(Z.shape[0]),
    }
    manifest = {
        "split": split,
        "format": "npz_shards",
        "save_transforms": True,
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "n_rows": int(Z.shape[0]),
        "extractor_output_dim": int(Z.shape[1]),
        "max_z_chunk_mb": 1,
        "processed_chunks": len(shards),
        "n_saved_shards": len(shards),
        "n_shards": len(shards),
        "summary": summary,
        "shards": shards,
    }
    manifest_path = out_dir / f"{split}_transform_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def write_fake_stage2_payload(tmp_path: Path, manifests):
    stage2_dir = tmp_path / "stage2_extractors" / "raw_linear"
    payload = {
        "stage": "stage2",
        "status": "ok",
        "decision_stride_rows": 5,
        "decision_offset_rows": 0,
        "decision_row_policy": "linear_every_n_rows_v1",
        "extractor_output_dim": manifests["train_sample"]["extractor_output_dim"],
        "stage2_dir": str(stage2_dir),
        "manifests": manifests,
    }
    (stage2_dir / "linear_stage2_extractor_metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def configure_stage3(monkeypatch, linear_offline, *, winsorize=True, q_lo=0.0, q_hi=1.0):
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_WINSORIZE", winsorize)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_WINSOR_Q_LO", q_lo)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_WINSOR_Q_HI", q_hi)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_STANDARDIZE", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_VARIANCE_FILTER", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_MIN_STD", 1e-6)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_STD_EPS", 1e-6)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_POST_CLIP_ABS", 0.0)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_NONFINITE_POLICY", "raise")
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_MAX_ROWS", 1000)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB", 64)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_SHARD_ROWS", 4)
    monkeypatch.setattr(linear_offline, "LINEAR_PREPROCESS_MAX_Z_CHUNK_MB", 64)


def test_stage3_preprocessor_standardizes_and_filters_constant_feature(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_linear import load_linear_preprocess_bundle

    configure_stage3(monkeypatch, linear_offline, winsorize=True, q_lo=0.0, q_hi=1.0)
    Z_train = np.array(
        [
            [1.0, 2.0, 3.0, 7.0],
            [2.0, 3.0, 4.0, 7.0],
            [3.0, 4.0, 5.0, 7.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 7.0],
            [6.0, 7.0, 8.0, 7.0],
        ],
        dtype=np.float32,
    )
    Z_val = Z_train[:4] + np.array([10.0, 10.0, 10.0, 0.0], dtype=np.float32)
    y_train = np.arange(Z_train.shape[0] * 3, dtype=np.float32).reshape(Z_train.shape[0], 3)
    y_val = np.arange(Z_val.shape[0] * 3, dtype=np.float32).reshape(Z_val.shape[0], 3)
    manifests = {
        "train_sample": write_fake_stage2_manifest(tmp_path, "train_sample", Z_train, y_train, np.arange(Z_train.shape[0])),
        "val": write_fake_stage2_manifest(tmp_path, "val", Z_val, y_val, np.arange(100, 100 + Z_val.shape[0])),
        "test": None,
    }
    write_fake_stage2_payload(tmp_path, manifests)

    payload = linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")
    bundle = load_linear_preprocess_bundle(Path(payload["preprocess_bundle_path"]))

    assert bundle.original_dim == 4
    assert bundle.kept_dim == 3
    val_shard = payload["manifests"]["val"]["shards"][0]["path"]
    with np.load(val_shard) as arr:
        assert arr["Z"].shape[1] == 3
        assert np.isfinite(arr["Z"]).all()


def test_stage3_winsorization_uses_train_only_caps(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_linear import load_linear_preprocess_bundle

    configure_stage3(monkeypatch, linear_offline, winsorize=True, q_lo=0.0, q_hi=1.0)
    Z_train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32)
    Z_val = np.array([[1000.0, 1000.0], [2000.0, 2000.0]], dtype=np.float32)
    y_train = np.zeros((Z_train.shape[0], 3), dtype=np.float32)
    y_val = np.zeros((Z_val.shape[0], 3), dtype=np.float32)
    manifests = {
        "train_sample": write_fake_stage2_manifest(tmp_path, "train_sample", Z_train, y_train, np.arange(Z_train.shape[0])),
        "val": write_fake_stage2_manifest(tmp_path, "val", Z_val, y_val, np.arange(Z_val.shape[0])),
        "test": None,
    }
    write_fake_stage2_payload(tmp_path, manifests)

    payload = linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")
    bundle = load_linear_preprocess_bundle(Path(payload["preprocess_bundle_path"]))
    np.testing.assert_allclose(bundle.upper, np.array([3.0, 4.0], dtype=np.float32))


def test_stage3_no_validation_leakage_in_mean(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_linear import load_linear_preprocess_bundle

    configure_stage3(monkeypatch, linear_offline, winsorize=True, q_lo=0.0, q_hi=1.0)
    Z_train = np.array([[8.0, 0.0], [10.0, 1.0], [12.0, 2.0]], dtype=np.float32)
    Z_val = np.array([[1000.0, 0.0], [1000.0, 1.0], [1000.0, 2.0]], dtype=np.float32)
    y_train = np.zeros((3, 3), dtype=np.float32)
    y_val = np.zeros((3, 3), dtype=np.float32)
    manifests = {
        "train_sample": write_fake_stage2_manifest(tmp_path, "train_sample", Z_train, y_train, np.arange(3)),
        "val": write_fake_stage2_manifest(tmp_path, "val", Z_val, y_val, np.arange(3)),
        "test": None,
    }
    write_fake_stage2_payload(tmp_path, manifests)

    payload = linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")
    bundle = load_linear_preprocess_bundle(Path(payload["preprocess_bundle_path"]))
    assert abs(float(bundle.mean[0]) - 10.0) < 1e-6


def test_stage3_manifest_row_alignment_and_bundle_reload(tmp_path, monkeypatch):
    import linear_offline
    from CMSSL17_linear import load_linear_preprocess_bundle

    configure_stage3(monkeypatch, linear_offline, winsorize=True, q_lo=0.0, q_hi=1.0)
    Z_train = np.arange(30, dtype=np.float32).reshape(10, 3)
    Z_val = np.arange(18, dtype=np.float32).reshape(6, 3)
    y_train = np.arange(30, dtype=np.float32).reshape(10, 3)
    y_val = np.arange(18, dtype=np.float32).reshape(6, 3)
    manifests = {
        "train_sample": write_fake_stage2_manifest(tmp_path, "train_sample", Z_train, y_train, np.arange(10), shard_rows=3),
        "val": write_fake_stage2_manifest(tmp_path, "val", Z_val, y_val, np.arange(20, 26), shard_rows=4),
        "test": None,
    }
    write_fake_stage2_payload(tmp_path, manifests)

    payload = linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")
    shard_path = payload["manifests"]["val"]["shards"][0]["path"]
    with np.load(shard_path) as arr:
        assert arr["Z"].shape[0] == arr["y"].shape[0] == arr["positions"].shape[0]

    loaded = load_linear_preprocess_bundle(Path(payload["preprocess_bundle_path"]))
    original = linear_offline.fit_linear_preprocessor_from_manifest(manifests["train_sample"], config=payload["preprocess_config"])
    np.testing.assert_allclose(original.transform(Z_val), loaded.transform(Z_val))


def test_stage3_rejects_decision_stride_mismatch(tmp_path, monkeypatch):
    import pytest
    import linear_offline

    configure_stage3(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)
    Z = np.ones((4, 2), dtype=np.float32)
    y = np.zeros((4, 3), dtype=np.float32)
    manifests = {
        "train_sample": write_fake_stage2_manifest(tmp_path, "train_sample", Z, y, np.arange(4)),
        "val": write_fake_stage2_manifest(tmp_path, "val", Z, y, np.arange(4)),
        "test": None,
    }
    manifests["train_sample"]["decision_stride_rows"] = 1
    Path(manifests["train_sample"]["manifest_path"]).write_text(json.dumps(manifests["train_sample"], indent=2))
    write_fake_stage2_payload(tmp_path, manifests)

    with pytest.raises(ValueError, match="decision-row mismatch"):
        linear_offline.run_stage3_preprocessing(linear_out_dir=tmp_path, extractor_name="raw_linear")
