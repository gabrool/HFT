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

import torch

if not hasattr(torch, "as_tensor"):
    torch.as_tensor = lambda value, dtype=None, device=None: np.asarray(value, dtype=dtype)
if not hasattr(torch, "device"):
    torch.device = lambda value: value


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
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_MAG_ALPHA_VALUES", [1e-4])
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
    mag_up_models = [FakeRegressor(np.log1p(0.2 + h)) for h in range(NUM_HORIZONS)]
    mag_down_models = [FakeRegressor(np.log1p(0.3 + h)) for h in range(NUM_HORIZONS)]
    bundle = LinearSklearnTakerBundle("linear_target_models_stage4_v1", {}, [1, 2, 3], direction_models, mag_up_models, mag_down_models, 1e-4, {}, "side_cond_log", np.ones(NUM_HORIZONS, dtype=np.float32), np.ones(NUM_HORIZONS, dtype=np.float32))
    pred = bundle.predict_dict_np(Z[:11])
    assert set(pred) == {"dir_logits", "move_logits", "p_move", "mag_up_log", "mag_down_log", "mag_up_bps", "mag_down_bps", "mag_up_sqrt", "mag_down_sqrt"}
    assert pred["dir_logits"].shape == (11, NUM_HORIZONS)
    assert pred["move_logits"].shape == (11, NUM_HORIZONS)
    assert pred["p_move"].shape == (11, NUM_HORIZONS)
    assert np.isfinite(pred["p_move"]).all()
    assert ((pred["p_move"] >= 0.0) & (pred["p_move"] <= 1.0)).all()
    assert np.isfinite(pred["dir_logits"]).all()
    assert (pred["mag_up_bps"] >= bundle.mag_floor).all()
    assert (pred["mag_down_bps"] >= bundle.mag_floor).all()
    assert np.allclose(pred["mag_up_sqrt"] ** 2, pred["mag_up_bps"], rtol=1e-4, atol=1e-6)


def test_torch_wrapper_cmssl_schema_only_filters_extra_keys():
    import torch
    from CMSSL17 import NUM_HORIZONS
    from CMSSL17_linear import LinearSklearnTakerBundle, LinearSklearnTorchWrapper

    class FakeDirection:
        def decision_function(self, Z):
            return Z[:, 0]

    class FakeRegressor:
        def __init__(self, value):
            self.value = value

        def predict(self, Z):
            return np.full(Z.shape[0], self.value, dtype=np.float32)

    Z = np.ones((8, 4), dtype=np.float32)

    bundle = LinearSklearnTakerBundle(
        "linear_target_models_stage4_v1",
        {"mag_log_pred_clip": 20.0},
        [200, 500, 1000],
        [FakeDirection() for _ in range(NUM_HORIZONS)],
        [FakeRegressor(np.log1p(1.0)) for _ in range(NUM_HORIZONS)],
        [FakeRegressor(np.log1p(1.0)) for _ in range(NUM_HORIZONS)],
        1e-4,
        {},
        "side_cond_log",
        np.ones(NUM_HORIZONS, dtype=np.float32),
        np.ones(NUM_HORIZONS, dtype=np.float32),
    )

    class FakeTensor:
        device = torch.device("cpu")

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return Z

    full = LinearSklearnTorchWrapper(bundle, cmssl_schema_only=False).forward(FakeTensor())
    assert set(full) == {
        "dir_logits",
        "move_logits",
        "p_move",
        "mag_up_log",
        "mag_down_log",
        "mag_up_bps",
        "mag_down_bps",
        "mag_up_sqrt",
        "mag_down_sqrt",
    }
    assert np.allclose(full["p_move"].detach().cpu().numpy(), 1.0)
    assert np.isposinf(full["move_logits"].detach().cpu().numpy()).all()

    strict = LinearSklearnTorchWrapper(bundle, cmssl_schema_only=True).forward(FakeTensor())
    assert set(strict) == {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
    assert strict["mag_up_sqrt"].shape == strict["dir_logits"].shape
    assert strict["mag_down_sqrt"].shape == strict["dir_logits"].shape
    assert np.allclose(
        strict["mag_up_sqrt"].detach().cpu().numpy(),
        strict["mag_down_sqrt"].detach().cpu().numpy(),
    )






def test_collect_predictions_streaming_side_mode_has_only_side_keys(monkeypatch):
    import linear_offline
    from CMSSL17 import NUM_HORIZONS
    from CMSSL17_linear import LinearSklearnTakerBundle

    class FakeDirection:
        def decision_function(self, Z):
            return np.full(Z.shape[0], 0.25, dtype=np.float32)

    class FakeRegressor:
        def __init__(self, value):
            self.value = value

        def predict(self, Z):
            return np.full(Z.shape[0], self.value, dtype=np.float32)

    Z = np.ones((6, 4), dtype=np.float32)
    y = np.zeros((6, 3), dtype=np.float32)
    pos = np.arange(6, dtype=np.int64)
    fake_ds = [None] * len(y)
    monkeypatch.setattr(
        linear_offline,
        "iter_preprocessed_batches_from_dataset",
        lambda **kwargs: iter([(Z, y, pos)]),
    )

    bundle = LinearSklearnTakerBundle(
        "linear_target_models_stage4_v1",
        {"mag_log_pred_clip": 20.0},
        [200, 500, 1000],
        [FakeDirection() for _ in range(NUM_HORIZONS)],
        [FakeRegressor(np.log1p(1.0)) for _ in range(NUM_HORIZONS)],
        [FakeRegressor(np.log1p(1.25)) for _ in range(NUM_HORIZONS)],
        1e-4,
        {},
        "side_cond_log",
        np.ones(NUM_HORIZONS, dtype=np.float32),
        np.ones(NUM_HORIZONS, dtype=np.float32),
    )
    payload = linear_offline.collect_predictions_and_labels_streaming(
        model_bundle=bundle,
        extractor=object(),
        preprocess_bundle=object(),
        ds=fake_ds,
        max_rows=0,
        batch_rows=8,
        split_name="unit_side",
    )
    assert set(payload) == {"dir_logits", "p_up", "move_logits", "p_move", "mag_up_sqrt", "mag_down_sqrt", "mag_up_log", "mag_down_log", "mag_up_bps", "mag_down_bps", "cond_edge_bps", "edge_bps", "y", "positions"}
    assert "mag_abs_log" not in payload
    assert "mag_abs_bps" not in payload
    p_up = 1.0 / (1.0 + np.exp(-np.clip(payload["dir_logits"], -50.0, 50.0)))
    expected_cond_edge = (
        p_up * payload["mag_up_bps"]
        - (1.0 - p_up) * payload["mag_down_bps"]
    )
    expected_edge = payload["p_move"] * expected_cond_edge
    assert np.allclose(payload["cond_edge_bps"], expected_cond_edge, rtol=1e-5, atol=1e-6)
    assert np.allclose(payload["edge_bps"], expected_edge, rtol=1e-5, atol=1e-6)
    assert np.allclose(payload["p_move"], 1.0)
    assert np.allclose(payload["edge_bps"], payload["cond_edge_bps"])


def test_side_cond_log_mag_targets_np():
    from CMSSL17_linear import side_cond_log_mag_targets_np
    y = np.asarray([[0.0, 1.0, -2.0], [3.0, 0.0, -4.0]], dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    up_log, down_log = side_cond_log_mag_targets_np(y, up_scale_bps=scale, down_scale_bps=scale, target_clip=0.0)
    assert up_log[0, 0] == 0.0
    assert np.isclose(up_log[0, 1], np.log1p(1.0))
    assert up_log[0, 2] == 0.0
    assert np.isclose(down_log[0, 2], np.log1p(2.0))
    assert np.isclose(up_log[1, 0], np.log1p(3.0))
    assert np.isclose(down_log[1, 2], np.log1p(4.0))


def test_inverse_side_cond_log_mag_np_roundtrip():
    from CMSSL17_linear import inverse_side_cond_log_mag_np, side_cond_log_mag_targets_np
    y = np.asarray([[0.0, 1.0, -2.0], [3.0, 0.0, -4.0]], dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    up_log, down_log = side_cond_log_mag_targets_np(y, up_scale_bps=scale, down_scale_bps=scale, target_clip=0.0)
    up_bps, down_bps = inverse_side_cond_log_mag_np(up_log, down_log, up_scale_bps=scale, down_scale_bps=scale, mag_floor_bps=0.0, pred_log_clip=20.0)
    assert np.allclose(up_bps, np.maximum(y, 0.0), rtol=1e-5, atol=1e-6)
    assert np.allclose(down_bps, np.maximum(-y, 0.0), rtol=1e-5, atol=1e-6)


def test_build_move_target_from_stats_np():
    import linear_offline
    y = np.asarray([[0.00, 0.01, -0.01], [0.04, 0.10, -0.10], [-0.04, -0.10, 0.10], [5.00, -5.00, 0.00]], dtype=np.float32)
    stats = {"pos_lo_raw_bps": np.asarray([0.03, 0.05, 0.05], dtype=np.float32), "neg_lo_abs_bps": np.asarray([0.03, 0.05, 0.05], dtype=np.float32)}
    tgt = linear_offline.build_move_target_from_stats_np(y, stats)
    assert tgt.shape == y.shape
    assert np.array_equal(tgt[0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(tgt[1], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.array_equal(tgt[2], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert np.array_equal(tgt[3], np.array([1.0, 1.0, 0.0], dtype=np.float32))


def test_train_move_models_streaming_from_plan_uses_all_rows(monkeypatch):
    import linear_offline
    from CMSSL17 import NUM_HORIZONS
    Z = np.ones((8, 4), dtype=np.float32)
    y = np.asarray([[0.0, 0.01, -0.01], [0.04, 0.10, -0.10], [-0.04, -0.10, 0.10], [5.0, -5.0, 0.0], [0.01, 0.01, -0.01], [0.04, 0.0, -0.1], [-0.04, -0.1, 0.1], [0.0, 0.0, 0.0]], dtype=np.float32)
    stats = {"pos_lo_raw_bps": np.asarray([0.03, 0.05, 0.05], dtype=np.float32), "neg_lo_abs_bps": np.asarray([0.03, 0.05, 0.05], dtype=np.float32)}

    class RecordingModel:
        def __init__(self): self.rows = []
        def partial_fit(self, X, t, classes=None): self.rows.append((X.shape[0], np.asarray(t))); return self

    made = []
    monkeypatch.setattr(linear_offline, "make_move_model", lambda **kwargs: made.append(RecordingModel()) or made[-1])
    monkeypatch.setattr(linear_offline, "iter_preprocessed_batches_from_train_plan", lambda **kwargs: iter([(Z, y, None)]))
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: int(y.shape[0]))
    alphas = [1e-3, 1e-2]
    results = linear_offline.train_move_models_streaming_from_plan(extractor=object(), preprocess_bundle=object(), plan={"train_split_entries": [{}]}, stats=stats, move_alpha_values=alphas, config={"epochs": 1, "batch_rows": 64, "random_state": 1, "move_weighting": "balanced"})
    assert len(results) == len(alphas)
    expected_t = linear_offline.build_move_target_from_stats_np(y, stats)
    for result in results:
        assert len(result["move_models"]) == NUM_HORIZONS
        fs = result["fit_summary"]
        assert fs["move_training_rows"] == "all_decision_rows"
        assert fs["move_pos_rows_per_horizon"] == expected_t.sum(axis=0).astype(int).tolist()
        assert fs["move_neg_rows_per_horizon"] == (y.shape[0] - expected_t.sum(axis=0)).astype(int).tolist()
    for m in made:
        assert sum(r for r, _ in m.rows) == y.shape[0]


def test_add_move_head_metrics_outputs_move_and_edge_metrics():
    import linear_offline
    from CMSSL17 import NUM_HORIZONS
    n = 16
    y = np.tile(np.array([[0.0, 0.1, -0.1]], dtype=np.float32), (n, 1))
    pred = {
        "p_move": np.full((n, NUM_HORIZONS), 0.5, dtype=np.float32),
        "move_logits": np.zeros((n, NUM_HORIZONS), dtype=np.float32),
        "cond_edge_bps": np.full((n, NUM_HORIZONS), 0.2, dtype=np.float32),
        "edge_bps": np.full((n, NUM_HORIZONS), 0.1, dtype=np.float32),
    }
    stats = {"pos_lo_raw_bps": np.asarray([0.03, 0.05, 0.05], dtype=np.float32), "neg_lo_abs_bps": np.asarray([0.03, 0.05, 0.05], dtype=np.float32)}
    m = {}
    linear_offline.add_move_head_metrics(m, y=y, pred=pred, stats=stats)
    for k in ["move_auc", "move_bal_acc", "move_bce", "move_pos_frac_true", "move_prob_mean_zero_rows", "move_prob_mean_nonmove_rows", "move_prob_mean_move_rows", "cond_edge_spearman_all", "cond_edge_spearman_kept", "edge_spearman_all", "edge_spearman_kept", "edge_bal_sign_acc_q50plus"]:
        assert k in m
        assert len(m[k]) == NUM_HORIZONS


def test_direction_helper_trains_only_direction(tmp_path, monkeypatch):
    import linear_offline
    Z, y, _ = make_synthetic()
    stats = write_trim_stats(tmp_path, y)
    calls = {"dir": 0, "mag": 0}
    class _M:
        def partial_fit(self, *args, **kwargs):
            return self
    monkeypatch.setattr(linear_offline, "make_direction_model", lambda **kwargs: type("D", (), {"partial_fit": lambda self, *a, **k: calls.__setitem__("dir", calls["dir"] + 1) or self})())
    monkeypatch.setattr(linear_offline, "make_magnitude_model", lambda **kwargs: type("M", (), {"partial_fit": lambda self, *a, **k: calls.__setitem__("mag", calls["mag"] + 1) or self})())
    monkeypatch.setattr(linear_offline, "iter_preprocessed_batches_from_train_plan", lambda **kwargs: iter([(Z, y, None)]))
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: int(y.shape[0]))
    linear_offline.train_direction_models_streaming_from_plan(extractor=object(), preprocess_bundle=object(), plan={"train_split_entries": [{}]}, stats=stats, direction_alpha_values=[1e-3], config={"epochs": 1, "batch_rows": 32, "random_state": 1, "direction_weighting": "none"}, direction_weights=[(1.0, 1.0)] * len(linear_offline.HORIZONS_MS))
    assert calls["dir"] > 0
    assert calls["mag"] == 0


def test_magnitude_helper_trains_side_active_rows_only(tmp_path, monkeypatch):
    import linear_offline
    Z = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    y = np.asarray([
        [ 1.0,  0.0, -2.0],
        [ 0.0,  3.0,  0.0],
        [-1.5,  0.0,  4.0],
        [ 0.0, -2.0,  0.0],
        [ 2.0,  1.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [-3.0, -1.0,  2.0],
        [ 0.5,  0.0,  0.0],
    ], dtype=np.float32)

    class RecordingMagnitudeModel:
        def __init__(self):
            self.n_rows = 0
            self.targets = []
        def partial_fit(self, X, target):
            self.n_rows += int(X.shape[0])
            self.targets.append(np.asarray(target, dtype=np.float32).copy())
            return self

    models = []
    def fake_make_magnitude_model(**kwargs):
        m = RecordingMagnitudeModel()
        models.append(m)
        return m

    monkeypatch.setattr(linear_offline, "make_direction_model", lambda **kwargs: (_ for _ in ()).throw(AssertionError("direction should not be trained")))
    monkeypatch.setattr(linear_offline, "make_magnitude_model", fake_make_magnitude_model)
    monkeypatch.setattr(linear_offline, "iter_preprocessed_batches_from_train_plan", lambda **kwargs: iter([(Z, y, None)]))
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: int(y.shape[0]))

    result = linear_offline.train_magnitude_models_streaming_from_plan(
        extractor=object(), preprocess_bundle=object(), plan={"train_split_entries": [{}]}, mag_alpha_values=[1e-4],
        config={"epochs": 1, "batch_rows": 32, "random_state": 1, "mag_up_scale_bps": [1.0, 1.0, 1.0], "mag_down_scale_bps": [1.0, 1.0, 1.0]}
    )[0]

    up_models = result["mag_up_models"]
    down_models = result["mag_down_models"]
    expected_up = [(y[:, h] > 0).sum() for h in range(y.shape[1])]
    expected_down = [(y[:, h] < 0).sum() for h in range(y.shape[1])]

    assert result["fit_summary"]["mag_training_rows"] == "side_active_rows"
    assert result["fit_summary"]["up_rows_per_horizon"] == [int(x) for x in expected_up]
    assert result["fit_summary"]["down_rows_per_horizon"] == [int(x) for x in expected_down]

    for h in range(y.shape[1]):
        assert up_models[h].n_rows == int(expected_up[h])
        assert down_models[h].n_rows == int(expected_down[h])
        up_targets = np.concatenate(up_models[h].targets) if up_models[h].targets else np.array([])
        down_targets = np.concatenate(down_models[h].targets) if down_models[h].targets else np.array([])
        assert up_targets.size == int(expected_up[h])
        assert down_targets.size == int(expected_down[h])
        assert np.all(up_targets > 0.0)
        assert np.all(down_targets > 0.0)


def test_add_side_cond_log_magnitude_metrics_values_and_no_all_row_keys():
    import linear_offline
    y = np.asarray([
        [ 1.0,  0.0, -2.0],
        [ 0.0,  3.0,  0.0],
        [-2.0,  0.0,  4.0],
        [ 0.0, -1.0,  0.0],
        [ 4.0,  1.0, -1.0],
        [ 0.0,  0.0,  0.0],
    ], dtype=np.float32)
    mag_up_bps = np.maximum(y, 0.0) * 2.0 + 0.1
    mag_down_bps = np.maximum(-y, 0.0) * 0.5 + 0.1
    pred = {"dir_logits": np.zeros_like(y, dtype=np.float32), "mag_up_bps": mag_up_bps, "mag_down_bps": mag_down_bps, "mag_up_log": np.log1p(mag_up_bps), "mag_down_log": np.log1p(mag_down_bps)}
    metrics = {}
    linear_offline.add_side_cond_log_magnitude_metrics(metrics, y=y, pred=pred, scale_up_bps=np.ones(y.shape[1],dtype=np.float32), scale_down_bps=np.ones(y.shape[1],dtype=np.float32))
    h = 0
    up_rows = y[:, h] > 0
    true_up = y[up_rows, h]
    pred_up = mag_up_bps[up_rows, h]
    assert np.isclose(metrics["up_p50_ratio_cond"][h], np.percentile(pred_up, 50) / np.percentile(true_up, 50))
    assert np.isclose(metrics["up_p90_ratio_cond"][h], np.percentile(pred_up, 90) / np.percentile(true_up, 90))
    assert np.isclose(metrics["up_mean_ratio_cond"][h], np.mean(pred_up) / np.mean(true_up))
    assert metrics["up_n_cond"][h] == int(up_rows.sum())
    assert metrics["down_n_cond"][h] == int((y[:, h] < 0).sum())
    for key in ["mean_side_log_huber_cond", "mean_side_spearman_cond", "mean_side_mean_ratio_cond", "mean_side_p50_ratio_cond", "mean_side_p90_ratio_cond", "mean_side_top_bottom_true_mean_lift_cond", "zero_row_mean_pred_abs_bps", "up_inactive_pred_p90_bps", "down_inactive_pred_p90_bps", "zero_row_up_pred_p90_bps", "zero_row_down_pred_p90_bps"]:
        assert key in metrics
    for key in ["mag_expected_abs_spearman_all", "pred_expected_abs_p90_over_true_abs_p90_all", "pred_expected_abs_p95_over_true_abs_p95_all", "pred_expected_abs_p50_over_true_abs_p50_all", "true_abs_bps_p50_all", "pred_expected_abs_bps_p50_all"]:
        assert key not in metrics


def test_add_side_cond_log_magnitude_metrics_deciles_and_lift():
    import linear_offline
    n = 100
    y = np.zeros((n, 3), dtype=np.float32)
    y[:, 0] = np.linspace(0.1, 10.0, n)
    y[:, 1] = -np.linspace(0.1, 10.0, n)
    up = np.tile(np.linspace(0.1, 10.0, n).reshape(-1, 1), (1, 3)).astype(np.float32)
    down = np.tile(np.linspace(0.1, 10.0, n).reshape(-1, 1), (1, 3)).astype(np.float32)
    pred = {"dir_logits": np.zeros_like(y, dtype=np.float32), "mag_up_bps": up, "mag_down_bps": down, "mag_up_log": np.log1p(up), "mag_down_log": np.log1p(down)}
    metrics = {}
    linear_offline.add_side_cond_log_magnitude_metrics(metrics, y=y, pred=pred, scale_up_bps=np.ones(y.shape[1],dtype=np.float32), scale_down_bps=np.ones(y.shape[1],dtype=np.float32))
    assert len(metrics["up_decile_calibration_cond"][0]) == 10
    assert np.isfinite(metrics["up_top_bottom_true_mean_lift_cond"][0])
    assert metrics["up_top_bottom_true_mean_lift_cond"][0] > 1.0
    assert len(metrics["down_decile_calibration_cond"][1]) == 10
    assert np.isfinite(metrics["down_top_bottom_true_mean_lift_cond"][1])
    assert metrics["down_top_bottom_true_mean_lift_cond"][1] > 1.0


def test_stage4_prints_candidate_and_best_summary(capsys, tmp_path, monkeypatch):
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_ALPHA_VALUES", [1e-4])
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))

    fake_plan = {
        "has_cmssl_test": False,
        "train_split_entries": [{}],
        "train_week_keys": ["w0"],
        "val_split_entries": [{}],
    }
    fake_preprocess = types.SimpleNamespace(original_dim=5, kept_dim=5)
    fake_bundle = types.SimpleNamespace(config={"alpha": 1e-4}, fit_summary={"unit_test": True})
    fake_dir_result = {"direction_alpha": 1e-4, "direction_models": [], "fit_summary": {"direction_alpha": 1e-4}}
    fake_mag_result = {"mag_alpha": 1e-4, "mag_up_models": [], "mag_down_models": [], "fit_summary": {"mag_alpha": 1e-4}}

    metric_len = len(linear_offline.HORIZONS_MS)
    fake_metrics = {
        "dir_auc_kept": [0.71] * metric_len,
        "dir_bal_acc_kept": [0.62] * metric_len,
        "val_dir_bce_kept": [0.55] * metric_len,
        "edge_spearman_kept": [0.13] * metric_len,
        "edge_spearman_all": [0.14] * metric_len,
        "mean_side_log_huber_cond": [0.22] * metric_len,
        "mean_side_spearman_cond": [0.44] * metric_len,
        "mean_side_p50_ratio_cond": [0.88] * metric_len,
        "mean_side_p90_ratio_cond": [1.23] * metric_len,
        "mean_side_top_bottom_true_mean_lift_cond": [2.5] * metric_len,
        "zero_row_mean_pred_abs_bps": [0.07] * metric_len,
        "primary_metric_guard_passed": True,
    }

    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda *, out_root: fake_plan)
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(
        linear_offline,
        "load_stage3_payload",
        lambda *args, **kwargs: {"payload_path": "stage3.json", "preprocess_bundle_path": str(tmp_path / "preprocess.npz")},
    )
    monkeypatch.setattr(linear_offline, "_validate_manifest_decision_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "load_linear_preprocess_bundle", lambda path: fake_preprocess)
    monkeypatch.setattr(linear_offline, "load_linear_trim_stats", lambda linear_out_dir: {})
    monkeypatch.setattr(linear_offline, "compute_side_log_mag_scales_from_train_plan", lambda **kwargs: (np.ones(len(linear_offline.HORIZONS_MS), dtype=np.float32), np.ones(len(linear_offline.HORIZONS_MS), dtype=np.float32)))
    monkeypatch.setattr(linear_offline, "compute_global_direction_weights_from_train_labels_plan", lambda **kwargs: [(1.0, 1.0)] * len(linear_offline.HORIZONS_MS))
    monkeypatch.setattr(linear_offline, "train_direction_models_streaming_from_plan", lambda **kwargs: [fake_dir_result])
    monkeypatch.setattr(linear_offline, "train_magnitude_models_streaming_from_plan", lambda **kwargs: [fake_mag_result])
    monkeypatch.setattr(linear_offline, "build_stage4_bundle_from_parts", lambda **kwargs: fake_bundle)
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: object())
    monkeypatch.setattr(linear_offline, "evaluate_stage4_bundle_streaming", lambda **kwargs: dict(fake_metrics))
    monkeypatch.setattr(linear_offline, "close_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "force_gc", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: 10)
    monkeypatch.setattr(linear_offline, "split_decision_row_count_from_plan", lambda plan, split, max_rows=0: 5)

    linear_offline.run_stage4_training(
        linear_out_dir=tmp_path,
        extractor_name="fake_extractor",
        preprocess_name="fake_preprocess",
        device=linear_offline.torch.device("cpu"),
    )

    out = capsys.readouterr().out
    assert out.count("[linear-stage4-dir-candidate]") == 1
    assert out.count("[linear-stage4-mag-candidate]") == 1
    assert out.count("[linear-stage4-best]") == 1
    assert "dir_alpha=0.0001" in out
    assert "mag_alpha=" in out
    assert "auc_1s=" in out
    assert "bal_1s=" in out
    assert "bce_1s=" in out
    assert "edge_sp_1s=" in out
    assert "log_huber_1s=0.22" in out
    assert "sp_1s=0.44" in out
    assert "p50_ratio_1s=0.88" in out
    assert "p90_ratio_1s=1.23" in out
    assert "lift_1s=2.5" in out
    assert "zero_pred_1s=0.07" in out
    assert "mag_huber_1s=0.22" in out
    assert "mag_sp_1s=0.44" in out
    assert "mag_p50_ratio_1s=0.88" in out
    assert "mag_p90_ratio_1s=1.23" in out
    assert "mag_lift_1s=2.5" in out
    assert "mag_abs_sp_1s" not in out
    assert "expected_abs" not in out
    assert "selection_score=" in out


def test_stage4_records_distinct_direction_and_magnitude_alphas(tmp_path, monkeypatch):
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_ALPHA_VALUES", [1e-2])
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_MAG_ALPHA_VALUES", [1e-4])
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))

    fake_plan = {"has_cmssl_test": False, "train_split_entries": [{}], "train_week_keys": ["w0"], "val_split_entries": [{}]}
    fake_preprocess = types.SimpleNamespace(original_dim=5, kept_dim=5)
    fake_direction_models = []
    fake_mag_up_models = []
    fake_mag_down_models = []
    fake_dir_result = {"direction_alpha": 1e-2, "direction_models": fake_direction_models, "fit_summary": {"direction_alpha": 1e-2}}
    fake_mag_result = {"mag_alpha": 1e-4, "mag_up_models": fake_mag_up_models, "mag_down_models": fake_mag_down_models, "fit_summary": {"mag_alpha": 1e-4}}

    metric_len = len(linear_offline.HORIZONS_MS)
    fake_metrics = {"dir_auc_kept": [0.71] * metric_len, "dir_bal_acc_kept": [0.62] * metric_len, "val_dir_bce_kept": [0.55] * metric_len, "edge_spearman_kept": [0.13] * metric_len, "pred_abs_p90_over_true_abs_p90_kept": [0.91] * metric_len, "mean_side_p90_ratio_cond": [1.23] * metric_len, "mean_side_spearman_cond": [0.44] * metric_len, "primary_metric_guard_passed": True}

    saved = {}

    def fake_save_bundle(bundle, path):
        saved["bundle"] = bundle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"bundle")

    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda *, out_root: fake_plan)
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(linear_offline, "load_stage3_payload", lambda *args, **kwargs: {"payload_path": "stage3.json", "preprocess_bundle_path": str(tmp_path / "preprocess.npz")})
    monkeypatch.setattr(linear_offline, "_validate_manifest_decision_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "load_linear_preprocess_bundle", lambda path: fake_preprocess)
    monkeypatch.setattr(linear_offline, "load_linear_trim_stats", lambda linear_out_dir: {})
    monkeypatch.setattr(linear_offline, "compute_side_log_mag_scales_from_train_plan", lambda **kwargs: (np.ones(len(linear_offline.HORIZONS_MS), dtype=np.float32), np.ones(len(linear_offline.HORIZONS_MS), dtype=np.float32)))
    monkeypatch.setattr(linear_offline, "compute_global_direction_weights_from_train_labels_plan", lambda **kwargs: [(1.0, 1.0)] * len(linear_offline.HORIZONS_MS))
    monkeypatch.setattr(linear_offline, "train_direction_models_streaming_from_plan", lambda **kwargs: [fake_dir_result])
    monkeypatch.setattr(linear_offline, "train_magnitude_models_streaming_from_plan", lambda **kwargs: [fake_mag_result for _ in kwargs["mag_alpha_values"]])
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: object())
    monkeypatch.setattr(linear_offline, "evaluate_stage4_bundle_streaming", lambda **kwargs: dict(fake_metrics))
    monkeypatch.setattr(linear_offline, "close_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "force_gc", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: 10)
    monkeypatch.setattr(linear_offline, "split_decision_row_count_from_plan", lambda plan, split, max_rows=0: 5)
    monkeypatch.setattr(linear_offline, "save_linear_sklearn_bundle", fake_save_bundle)

    payload = linear_offline.run_stage4_training(linear_out_dir=tmp_path, extractor_name="fake_extractor", preprocess_name="fake_preprocess", device=linear_offline.torch.device("cpu"))

    assert payload["best_alpha"] == 1e-2
    assert payload["best_direction_alpha"] == 1e-2
    assert payload["best_mag_alpha"] == 1e-4

    bundle = saved["bundle"]
    assert bundle.config["direction_alpha"] == 1e-2
    assert bundle.config["mag_alpha"] == 1e-4
    assert bundle.fit_summary["direction_alpha"] == 1e-2
    assert bundle.fit_summary["mag_alpha"] == 1e-4
    assert "direction_fit_summary" in bundle.fit_summary
    assert "magnitude_fit_summary" in bundle.fit_summary


def test_stage4_reuses_reference_magnitude_alpha_without_retraining(tmp_path, monkeypatch):
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_ALPHA_VALUES", [1e-4])
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_MAG_ALPHA_VALUES", [1e-4, 1e-3, 1e-2])
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))

    fake_plan = {"has_cmssl_test": False, "train_split_entries": [{}], "train_week_keys": ["w0"], "val_split_entries": [{}]}
    fake_preprocess = types.SimpleNamespace(original_dim=5, kept_dim=5)
    fake_dir_result = {"direction_alpha": 1e-4, "direction_models": [], "fit_summary": {"direction_alpha": 1e-4}}
    fake_up, fake_down = [], []
    metric_len = len(linear_offline.HORIZONS_MS)
    fake_metrics = {"dir_auc_kept": [0.71] * metric_len, "dir_bal_acc_kept": [0.62] * metric_len, "val_dir_bce_kept": [0.55] * metric_len, "edge_spearman_kept": [0.13] * metric_len, "pred_abs_p90_over_true_abs_p90_kept": [0.91] * metric_len, "mean_side_p90_ratio_cond": [1.23] * metric_len, "mean_side_spearman_cond": [0.44] * metric_len, "primary_metric_guard_passed": True}
    seen_mag_alpha_batches = []

    def fake_train_mag(**kwargs):
        alphas = list(kwargs["mag_alpha_values"])
        seen_mag_alpha_batches.append(alphas)
        return [{"mag_alpha": float(a), "mag_up_models": fake_up, "mag_down_models": fake_down, "fit_summary": {"mag_alpha": float(a)}} for a in alphas]

    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda *, out_root: fake_plan)
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(linear_offline, "load_stage3_payload", lambda *args, **kwargs: {"payload_path": "stage3.json", "preprocess_bundle_path": str(tmp_path / "preprocess.npz")})
    monkeypatch.setattr(linear_offline, "_validate_manifest_decision_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "load_linear_preprocess_bundle", lambda path: fake_preprocess)
    monkeypatch.setattr(linear_offline, "load_linear_trim_stats", lambda linear_out_dir: {})
    monkeypatch.setattr(linear_offline, "compute_side_log_mag_scales_from_train_plan", lambda **kwargs: (np.ones(len(linear_offline.HORIZONS_MS), dtype=np.float32), np.ones(len(linear_offline.HORIZONS_MS), dtype=np.float32)))
    monkeypatch.setattr(linear_offline, "compute_global_direction_weights_from_train_labels_plan", lambda **kwargs: [(1.0, 1.0)] * len(linear_offline.HORIZONS_MS))
    monkeypatch.setattr(linear_offline, "train_direction_models_streaming_from_plan", lambda **kwargs: [fake_dir_result])
    monkeypatch.setattr(linear_offline, "train_magnitude_models_streaming_from_plan", fake_train_mag)
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: object())
    monkeypatch.setattr(linear_offline, "evaluate_stage4_bundle_streaming", lambda **kwargs: dict(fake_metrics))
    monkeypatch.setattr(linear_offline, "close_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "force_gc", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: 10)
    monkeypatch.setattr(linear_offline, "split_decision_row_count_from_plan", lambda plan, split, max_rows=0: 5)
    monkeypatch.setattr(linear_offline, "save_linear_sklearn_bundle", lambda bundle, path: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"bundle"))

    linear_offline.run_stage4_training(linear_out_dir=tmp_path, extractor_name="fake_extractor", preprocess_name="fake_preprocess", device=linear_offline.torch.device("cpu"))

    assert seen_mag_alpha_batches[0] == [1e-3]
    assert seen_mag_alpha_batches[1] == [1e-4, 1e-2]


def test_stage4_payload_contains_final_move_metrics_and_test_metrics(tmp_path, monkeypatch):
    import linear_offline

    configure_stage4(monkeypatch, linear_offline)
    monkeypatch.setattr(linear_offline, "LINEAR_STAGE4_RUN_TEST", True)
    monkeypatch.setattr(linear_offline, "OUT_ROOT", str(tmp_path / "out_root"))
    fake_plan = {"has_cmssl_test": True, "train_split_entries": [{}], "train_week_keys": ["w0"], "val_split_entries": [{}], "test_split_entries": [{}]}
    fake_preprocess = types.SimpleNamespace(original_dim=5, kept_dim=5)
    metric_len = len(linear_offline.HORIZONS_MS)
    fake_metrics = {"dir_auc_kept": [0.7] * metric_len, "dir_bal_acc_kept": [0.6] * metric_len, "dir_bce_kept": [0.5] * metric_len, "mag_primary_huber": [0.2] * metric_len, "mag_primary_spearman": [0.3] * metric_len, "mag_primary_p50_ratio": [1.0] * metric_len, "mag_primary_p90_ratio": [1.1] * metric_len, "mag_primary_top_bottom_true_mean_lift": [0.4] * metric_len, "move_auc": [0.65] * metric_len, "move_bal_acc": [0.55] * metric_len, "move_bce": [0.45] * metric_len, "move_pos_frac_true": [0.25] * metric_len, "move_prob_mean_zero_rows": [0.1] * metric_len, "move_prob_mean_nonmove_rows": [0.2] * metric_len, "move_prob_mean_move_rows": [0.8] * metric_len, "cond_edge_spearman_all": [0.15] * metric_len, "cond_edge_spearman_kept": [0.2] * metric_len, "edge_spearman_all": [0.12] * metric_len, "edge_spearman_kept": [0.18] * metric_len, "edge_bal_sign_acc_q50plus": [0.52] * metric_len, "primary_metric_guard_passed": True}

    monkeypatch.setattr(linear_offline, "load_linear_split_plan_from_out_root", lambda *, out_root: fake_plan)
    monkeypatch.setattr(linear_offline, "load_stage2_extractor_bundle", lambda **kwargs: (object(), {"payload_path": "stage2.json"}))
    monkeypatch.setattr(linear_offline, "load_stage3_payload", lambda *args, **kwargs: {"payload_path": "stage3.json", "preprocess_bundle_path": str(tmp_path / "preprocess.npz")})
    monkeypatch.setattr(linear_offline, "_validate_manifest_decision_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "load_linear_preprocess_bundle", lambda path: fake_preprocess)
    monkeypatch.setattr(linear_offline, "load_linear_trim_stats", lambda linear_out_dir: {})
    monkeypatch.setattr(linear_offline, "compute_side_log_mag_scales_from_train_plan", lambda **kwargs: (np.ones(metric_len, dtype=np.float32), np.ones(metric_len, dtype=np.float32)))
    monkeypatch.setattr(linear_offline, "compute_global_direction_weights_from_train_labels_plan", lambda **kwargs: [(1.0, 1.0)] * metric_len)
    monkeypatch.setattr(linear_offline, "train_direction_models_streaming_from_plan", lambda **kwargs: [{"direction_alpha": 1e-2, "direction_models": [], "fit_summary": {}}])
    monkeypatch.setattr(linear_offline, "train_magnitude_models_streaming_from_plan", lambda **kwargs: [{"mag_alpha": 1e-3, "mag_up_models": [], "mag_down_models": [], "fit_summary": {}} for _ in kwargs["mag_alpha_values"]])
    monkeypatch.setattr(linear_offline, "train_move_models_streaming_from_plan", lambda **kwargs: [{"move_alpha": 1e-3, "move_models": [], "fit_summary": {}}])
    monkeypatch.setattr(linear_offline, "build_val_dataset_from_plan", lambda plan: object())
    monkeypatch.setattr(linear_offline, "build_test_dataset_from_plan", lambda plan: object())
    monkeypatch.setattr(linear_offline, "evaluate_stage4_bundle_streaming", lambda **kwargs: dict(fake_metrics))
    monkeypatch.setattr(linear_offline, "close_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "force_gc", lambda *args, **kwargs: None)
    monkeypatch.setattr(linear_offline, "save_linear_sklearn_bundle", lambda bundle, path: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"bundle"))

    payload = linear_offline.run_stage4_training(linear_out_dir=tmp_path, extractor_name="fake_extractor", preprocess_name="fake_preprocess", device=linear_offline.torch.device("cpu"))
    assert "best_move_alpha" in payload
    assert "move_candidate_summaries" in payload
    assert "val_metrics" in payload
    assert "test_metrics" in payload
    assert "move" in payload["stage4_summary_metrics"]
    assert "edge" in payload["stage4_summary_metrics"]
    assert payload["stage4_summary_metrics"]["edge"]["schema"] == "p_move_times_conditional_side_edge_v1"
    assert "mag_abs_scale_bps" not in payload
    assert "direction_zero_rows" not in payload["stage4_summary_metrics"]


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
