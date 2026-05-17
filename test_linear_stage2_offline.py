import numpy as np

try:
    from test_feature_event_result_contract import _install_optional_dependency_stubs
except Exception:
    _install_optional_dependency_stubs = None

if _install_optional_dependency_stubs is not None:
    _install_optional_dependency_stubs()

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
    monkeypatch.setattr(linear_offline, "LINEAR_SAVE_TRANSFORMS", True)

    ds_train = FakeFlatDataset(12)
    ds_val = FakeFlatDataset(9, seed=18)
    ds_test = FakeFlatDataset(7, seed=19)

    payload = linear_offline.run_stage2_extraction(
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
    assert payload["val_summary"]["shape"][0] == len(ds_val)
    assert payload["val_summary"]["n_shards"] >= 2

    shards = list(stage2_dir.glob("val_transform_shard_*.npz"))
    assert shards
    with np.load(shards[0]) as arr:
        assert "Z" in arr and "y" in arr and "positions" in arr
        assert arr["Z"].shape[0] == arr["y"].shape[0] == arr["positions"].shape[0]
