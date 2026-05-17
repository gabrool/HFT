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


def test_collect_train_labels_from_plan_uses_decision_rows(monkeypatch):
    import linear_offline

    class FakeDataset:
        def __init__(self, n):
            self.y = np.arange(n * 3, dtype=np.float32).reshape(n, 3)

        def __len__(self):
            return len(self.y)

        def close(self):
            pass

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    ds = FakeDataset(20)
    monkeypatch.setattr(linear_offline, "build_train_week_dataset", lambda plan, week_index: ds)
    plan = {"train_split_entries": [{}], "train_week_keys": ["w0"]}
    y = linear_offline.collect_train_labels_from_plan(plan)
    np.testing.assert_array_equal(y, ds.y[[0, 5, 10, 15]])
