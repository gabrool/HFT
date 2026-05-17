import os
import sys
import types

import numpy as np

os.environ.setdefault("BYBIT_FEATURE_STORAGE_DTYPE", "fp32")

try:
    from test_feature_event_result_contract import _install_optional_dependency_stubs
except Exception:
    _install_optional_dependency_stubs = None

if _install_optional_dependency_stubs is not None:
    _install_optional_dependency_stubs()

sys.modules.setdefault("torch._inductor", types.ModuleType("torch._inductor"))
sys.modules.setdefault("torch._inductor.config", types.ModuleType("torch._inductor.config"))

try:
    from CMSSL17 import NUM_HORIZONS
except Exception:
    NUM_HORIZONS = 3


class FakeFlatDataset:
    def __init__(self, n: int, lookback: int = 12, feature_dim: int = 3, seed: int = 17):
        self.n = n
        self.lookback = lookback
        self.feature_dim_total = feature_dim
        self.stores = [object()]
        self.week_ids = np.zeros(n, dtype=np.int64)
        self.row_idx = np.arange(lookback - 1, lookback - 1 + n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.X = rng.normal(size=(n, lookback, feature_dim)).astype(np.float32)
        self.y = rng.normal(size=(n, NUM_HORIZONS)).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[int(idx)], self.y[int(idx)]


def test_compute_safe_transform_chunk_rows_respects_z_memory():
    import linear_offline

    rows = linear_offline.compute_safe_transform_chunk_rows(
        requested_rows=100_000,
        lookback=100,
        feature_dim=200,
        output_dim=50_000,
        max_x_chunk_mb=2048,
        max_z_chunk_mb=100,
        hard_cap_rows=50_000,
    )

    # 100 MB / (50k * 4 bytes) ≈ 524 rows
    assert 1 <= rows <= 600


def test_compute_safe_transform_chunk_rows_respects_hard_cap():
    import linear_offline

    rows = linear_offline.compute_safe_transform_chunk_rows(
        requested_rows=100_000,
        lookback=100,
        feature_dim=200,
        output_dim=100,
        max_x_chunk_mb=2048,
        max_z_chunk_mb=2048,
        hard_cap_rows=1234,
    )
    assert rows == 1234



def test_dataset_positions_uses_decision_stride(monkeypatch):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    pos = linear_offline._dataset_positions(20, 0)
    np.testing.assert_array_equal(pos, np.array([0, 5, 10, 15], dtype=np.int64))


def test_dataset_positions_uses_decision_offset(monkeypatch):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 1)

    pos = linear_offline._dataset_positions(20, 0)
    np.testing.assert_array_equal(pos, np.array([1, 6, 11, 16], dtype=np.int64))


def test_dataset_positions_samples_after_stride(monkeypatch):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    pos = linear_offline._dataset_positions(100, 4)
    assert len(pos) == 4
    assert np.all(pos % 5 == 0)


def test_collect_train_labels_from_datasets_uses_decision_rows(monkeypatch):
    import linear_offline

    class FakeDataset:
        def __init__(self, n):
            self.y = np.arange(n * 3, dtype=np.float32).reshape(n, 3)

        def __len__(self):
            return len(self.y)

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    ds = FakeDataset(20)
    y = linear_offline.collect_train_labels_from_datasets([ds])
    np.testing.assert_array_equal(y, ds.y[[0, 5, 10, 15]])


def test_run_stage2_extraction_raw_linear_writes_sharded_outputs(tmp_path, monkeypatch):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LOOKBACK", 12)
    monkeypatch.setattr(linear_offline, "LINEAR_EXTRACTOR", "raw_linear")
    monkeypatch.setattr(linear_offline, "LINEAR_EXTRACTOR_FIT_MAX_ROWS", 10)
    monkeypatch.setattr(linear_offline, "LINEAR_TRANSFORM_MAX_ROWS_PER_SPLIT", 0)
    monkeypatch.setattr(linear_offline, "LINEAR_EXTRACT_BATCH_ROWS", 4)
    monkeypatch.setattr(linear_offline, "LINEAR_CHUNKED_TRANSFORMS", True)
    monkeypatch.setattr(linear_offline, "LINEAR_TRANSFORM_SHARD_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_MAX_X_CHUNK_MB", 1)
    monkeypatch.setattr(linear_offline, "LINEAR_MAX_Z_CHUNK_MB", 1)
    monkeypatch.setattr(linear_offline, "LINEAR_SAVE_TRANSFORMS", True)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    ds_train = FakeFlatDataset(12)
    ds_val = FakeFlatDataset(9, seed=18)
    ds_test = FakeFlatDataset(7, seed=19)

    payload = linear_offline.legacy_run_stage2_extraction(
        linear_out_dir=tmp_path,
        ds_train_list=[ds_train],
        ds_val=ds_val,
        ds_test=ds_test,
        has_cmssl_test=True,
        meta={"feature_dim_total": ds_train.feature_dim_total},
        protocol="unit_test",
        train_week_keys=["week0"],
        extractor_config={
            "extractor": "raw_linear",
            "raw_mode": "lag_bank_stats",
            "raw_lags": [1, 2],
            "raw_windows": [3],
            "raw_include_std": True,
            "raw_include_slope": False,
            "n_kernels": 128,
            "hydra_n_kernels": 4,
            "n_groups": 8,
            "n_jobs": 1,
            "random_state": 17,
        },
    )

    stage2_dir = tmp_path / "stage2_extractors" / "raw_linear"
    assert (stage2_dir / "train_sample_transform_manifest.json").exists()
    assert (stage2_dir / "val_transform_manifest.json").exists()
    assert (stage2_dir / "test_transform_manifest.json").exists()
    assert payload["chunked_transforms"] is True
    assert payload["max_z_chunk_mb"] == 1
    assert payload["extractor_output_dim"] > 0
    assert payload["val_summary"]["shape"][0] == 2
    assert payload["val_summary"]["shape"][1] == payload["extractor_output_dim"]
    assert payload["val_summary"]["n_shards"] >= 1
    val_manifest = payload["manifests"]["val"]
    assert val_manifest["extractor_output_dim"] == payload["extractor_output_dim"]
    assert val_manifest["decision_stride_rows"] == 5
    assert val_manifest["decision_offset_rows"] == 0
    assert val_manifest["decision_row_policy"] == "linear_every_n_rows_v1"
    assert val_manifest["max_z_chunk_mb"] == 1
    assert val_manifest["processed_chunks"] >= 1
    assert val_manifest["n_saved_shards"] >= 1

    shards = list(stage2_dir.glob("val_transform_shard_*.npz"))
    assert shards
    with np.load(shards[0]) as arr:
        assert "Z" in arr and "y" in arr and "positions" in arr
        assert arr["Z"].shape[0] == arr["y"].shape[0] == arr["positions"].shape[0]
        assert np.all(arr["positions"] % 5 == 0)
