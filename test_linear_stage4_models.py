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
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_TRAIN_SPLIT", "train_full")
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


def test_train_stage4_candidates_streaming_from_plan_fits_all_models(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    import linear_offline
    from CMSSL17 import NUM_HORIZONS

    configure_stage4(monkeypatch, linear_offline)
    Z, y, _pos = make_synthetic()
    stats = write_trim_stats(tmp_path, y)

    def fake_iter(**kwargs):
        del kwargs
        yield Z, y, None

    monkeypatch.setattr(linear_offline, "iter_preprocessed_batches_from_train_plan", fake_iter)
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: int(y.shape[0]))
    monkeypatch.setattr(
        linear_offline,
        "compute_global_direction_weights_from_train_labels_plan",
        lambda **kwargs: [(1.0, 1.0) for _ in range(NUM_HORIZONS)],
    )
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
    bundles = linear_offline.train_stage4_candidates_streaming_from_plan(
        extractor=object(),
        preprocess_bundle=object(),
        plan={"train_split_entries": [{}], "train_week_keys": ["w0"]},
        stats=stats,
        alpha_values=[1e-4],
        config=config,
    )
    bundle = bundles[0]
    assert len(bundle.direction_models) == NUM_HORIZONS
    assert len(bundle.mag_up_models) == NUM_HORIZONS
    assert len(bundle.mag_down_models) == NUM_HORIZONS


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


def test_stable_sigmoid_handles_large_logits_without_warning():
    import warnings
    from CMSSL17_offline import _stable_sigmoid_np

    logits = np.asarray([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prob = _stable_sigmoid_np(logits)

    assert caught == []
    assert np.isfinite(prob).all()
    assert (prob >= 0.0).all()
    assert (prob <= 1.0).all()
    assert prob[0] == 0.0
    assert prob[2] == np.float32(0.5)
    assert prob[-1] == 1.0
