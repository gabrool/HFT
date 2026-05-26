import inspect
import subprocess
import sys

import numpy as np
import pytest

import mmrt.linear.diagnostics as dg


def test_public_api_boundary():
    expected = [
        "DEFAULT_TOP_K", "DEFAULT_NUM_BINS", "DEFAULT_MAX_ROWS",
        "DIRECTION_DOWN_CLASS", "DIRECTION_UP_CLASS", "DIRECTION_INVALID_CLASS",
        "DiagnosticsConfig", "VectorSummary", "CoefficientRecord", "CoefficientDiagnostics",
        "PreprocessDiagnostics", "CalibrationBin", "CalibrationDiagnostics", "PredictionDiagnostics",
        "summarize_vector", "coefficient_diagnostics", "coefficient_diagnostics_from_head_dict",
        "coefficient_diagnostics_from_bundle_dict", "preprocess_diagnostics_from_state_dict",
        "direction_calibration_diagnostics", "prediction_diagnostics", "build_linear_diagnostics_report",
    ]
    assert dg.__all__ == expected
    forbidden = ("bybit", "cmssl", "stage", "pca", "sklearn", "torch", "pandas", "polars", "reader", "writer", "storage", "extract", "train", "model")
    lowered = [name.lower() for name in dg.__all__]
    for name in lowered:
        assert not any(tok in name for tok in forbidden)


def test_no_forbidden_imports():
    code = "import sys; before=set(sys.modules); import mmrt.linear.diagnostics; after=set(sys.modules)-before; print('\\n'.join(sorted(after)))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    loaded = set(out.splitlines())
    forbidden = {
        "pandas", "polars", "torch", "sklearn", "scipy", "numba", "pyarrow",
        "mmrt.storage.reader", "mmrt.storage.writer", "mmrt.storage.splits",
        "mmrt.linear.extractors", "mmrt.linear.targets", "mmrt.linear.preprocess",
        "mmrt.linear.models", "mmrt.linear.evaluate", "mmrt.features.engine",
        "mmrt.features.labels", "mmrt.features.transforms", "mmrt.data.tardis_csv",
        "mmrt.data.event_merge", "CMSSL17", "offline_ingest",
    }
    assert not (loaded & forbidden)


def test_config_validation():
    cfg = dg.DiagnosticsConfig()
    assert (cfg.top_k, cfg.num_bins, cfg.max_rows) == (25, 10, 200_000)
    for kwargs in ({"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"num_bins": 0}, {"num_bins": -1}, {"num_bins": True}, {"max_rows": 0}, {"max_rows": -1}, {"max_rows": True}):
        with pytest.raises(ValueError):
            dg.DiagnosticsConfig(**kwargs)
    for attr in ("output_path", "plot", "stage", "random_state", "sample_fraction", "pca", "feature_removal_threshold"):
        assert not hasattr(cfg, attr)


def test_summarize_vector_basic_and_empty():
    s = dg.summarize_vector(np.array([1.0, 2.0, 3.0, 4.0]), name="x")
    assert s.n_rows == 4 and s.mean == 2.5 and s.min == 1.0 and s.p50 == 2.5 and s.max == 4.0
    assert np.isclose(s.std, np.std(np.array([1.0, 2.0, 3.0, 4.0]), ddof=0))
    d = s.as_dict()
    assert list(d.keys()) == ["name", "n_rows", "mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max"]
    e = dg.summarize_vector(np.array([], dtype=float), name="empty")
    assert e.n_rows == 0
    for k in ("mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max"):
        assert np.isnan(getattr(e, k))
    with pytest.raises(ValueError):
        dg.summarize_vector(np.array([1.0, np.inf]), name="bad")
    with pytest.raises(ValueError):
        dg.summarize_vector(np.array([[1.0]]), name="bad")


def test_summarize_vector_bounded_deterministic():
    arr = np.arange(1000, dtype=float)
    cfg = dg.DiagnosticsConfig(max_rows=10)
    a = dg.summarize_vector(arr, name="a", config=cfg).as_dict()
    b = dg.summarize_vector(arr, name="a", config=cfg).as_dict()
    assert a == b
    assert a["n_rows"] == 1000


def test_coefficient_diagnostics_basic():
    out = dg.coefficient_diagnostics(head_name="direction", feature_columns=("a", "b", "c", "d"), weights=np.array([0.1, -2.0, 0.5, -0.3]), intercept=1.25, config=dg.DiagnosticsConfig(top_k=2))
    assert np.isclose(out.l1_norm, 2.9)
    assert np.isclose(out.l2_norm, np.sqrt(0.1**2 + 2.0**2 + 0.5**2 + 0.3**2))
    assert np.isclose(out.max_abs, 2.0)
    assert [r.feature for r in out.top_abs] == ["b", "c"]
    assert [r.feature for r in out.top_positive] == ["c", "a"]
    assert [r.feature for r in out.top_negative] == ["b", "d"]
    assert [r.rank for r in out.top_abs] == [1, 2]
    assert isinstance(out.as_dict()["top_abs"], list)


def test_coefficient_diagnostics_all_zero_or_one_sided_weights():
    cfg = dg.DiagnosticsConfig(top_k=2)

    zero = dg.coefficient_diagnostics(
        head_name="direction",
        feature_columns=("a", "b"),
        weights=np.array([0.0, 0.0]),
        intercept=0.0,
        config=cfg,
    )
    assert [r.feature for r in zero.top_abs] == ["a", "b"]
    assert zero.top_positive == ()
    assert zero.top_negative == ()
    assert zero.l1_norm == 0.0
    assert zero.l2_norm == 0.0
    assert zero.max_abs == 0.0

    pos = dg.coefficient_diagnostics(
        head_name="direction",
        feature_columns=("a", "b"),
        weights=np.array([1.0, 2.0]),
        intercept=0.0,
        config=cfg,
    )
    assert [r.feature for r in pos.top_abs] == ["b", "a"]
    assert [r.feature for r in pos.top_positive] == ["b", "a"]
    assert pos.top_negative == ()

    neg = dg.coefficient_diagnostics(
        head_name="direction",
        feature_columns=("a", "b"),
        weights=np.array([-1.0, -2.0]),
        intercept=0.0,
        config=cfg,
    )
    assert [r.feature for r in neg.top_abs] == ["b", "a"]
    assert neg.top_positive == ()
    assert [r.feature for r in neg.top_negative] == ["b", "a"]


def test_coefficient_diagnostics_validation():
    with pytest.raises(ValueError):
        dg.coefficient_diagnostics(head_name="x", feature_columns=("a", "a"), weights=np.array([1.0, 2.0]), intercept=0.0)
    with pytest.raises(ValueError):
        dg.coefficient_diagnostics(head_name="x", feature_columns=("a",), weights=np.array([1.0, 2.0]), intercept=0.0)
    with pytest.raises(ValueError):
        dg.coefficient_diagnostics(head_name="x", feature_columns=("a",), weights=np.array([np.nan]), intercept=0.0)
    with pytest.raises(ValueError):
        dg.coefficient_diagnostics(head_name="x", feature_columns=("a",), weights=np.array([1.0]), intercept=np.inf)
    with pytest.raises(ValueError):
        dg.coefficient_diagnostics(head_name=" ", feature_columns=("a",), weights=np.array([1.0]), intercept=0.0)
    with pytest.raises(ValueError):
        dg.CoefficientRecord(feature="a", coefficient=1.0, abs_coefficient=2.0, rank=1)
    with pytest.raises(ValueError):
        dg.CoefficientRecord(feature="a", coefficient=1.0, abs_coefficient=1.0, rank=0)


def test_coefficient_diagnostics_from_head_and_bundle_dict():
    head = {"head_name": "direction", "feature_columns": ["a", "b"], "weights": [1, -1], "intercept": 0.1}
    assert dg.coefficient_diagnostics_from_head_dict(head).head_name == "direction"
    bundle = {"direction": head, "magnitude_up": {**head, "head_name": "magnitude_up"}, "magnitude_down": {**head, "head_name": "magnitude_down"}}
    out = dg.coefficient_diagnostics_from_bundle_dict(bundle)
    assert set(out) == {"direction", "magnitude_up", "magnitude_down"}
    with pytest.raises(ValueError):
        dg.coefficient_diagnostics_from_head_dict({"head_name": "x"})


def test_preprocess_diagnostics_from_state_dict():
    state = {"feature_columns": ["a", "b", "c"], "variance": [0.0, 1.0, 4.0], "scale": [1e-6, 1.0, 2.0], "active_mask": np.array([False, True, True], dtype=bool)}
    out = dg.preprocess_diagnostics_from_state_dict(state)
    assert out.active_count == 2 and out.inactive_count == 1 and out.inactive_features == ("a",)
    assert isinstance(out.scale_summary, dg.VectorSummary)
    with pytest.raises(ValueError):
        dg.preprocess_diagnostics_from_state_dict({**state, "scale": [1.0, 2.0]})
    with pytest.raises(ValueError):
        dg.preprocess_diagnostics_from_state_dict({**state, "variance": [-1.0, 1.0, 1.0]})
    with pytest.raises(ValueError):
        dg.preprocess_diagnostics_from_state_dict({**state, "scale": [1.0, 0.0, 1.0]})
    with pytest.raises(ValueError):
        dg.preprocess_diagnostics_from_state_dict({**state, "active_mask": np.array([0, 1, 1])})


def test_direction_calibration_diagnostics_basic():
    y = np.array([0, 1, 1, 0, -1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.5])
    out = dg.direction_calibration_diagnostics(y, p, config=dg.DiagnosticsConfig(num_bins=2))
    assert out.valid_count == 4 and out.n_rows == 5
    b0, b1 = out.bins
    assert b0.count == 2 and np.isclose(b0.mean_predicted, 0.15) and np.isclose(b0.empirical_positive_rate, 0.5)
    assert b1.count == 2 and np.isclose(b1.mean_predicted, 0.85) and np.isclose(b1.empirical_positive_rate, 0.5)


def test_direction_calibration_validation_and_empty_bins():
    with pytest.raises(ValueError):
        dg.direction_calibration_diagnostics(np.array([0, 1]), np.array([1.2, 0.1]))
    with pytest.raises(ValueError):
        dg.direction_calibration_diagnostics(np.array([2, 1]), np.array([0.2, 0.1]))
    with pytest.raises(ValueError):
        dg.direction_calibration_diagnostics(np.array([0, -1]), np.array([0.2, 0.1]), direction_mask=np.array([True, True], dtype=bool))
    with pytest.raises(ValueError):
        dg.direction_calibration_diagnostics(np.array([0, -1]), np.array([0.2, 0.1]), direction_mask=np.array([1, 0]))
    out = dg.direction_calibration_diagnostics(np.array([0, 1]), np.array([0.01, 0.99]), config=dg.DiagnosticsConfig(num_bins=3))
    assert any(np.isnan(b.mean_predicted) and b.count == 0 for b in out.bins)


def test_prediction_diagnostics():
    out = dg.prediction_diagnostics(direction_p_up=np.array([0.1, 0.9]), magnitude_up=np.array([1.0, 2.0]), magnitude_down=np.array([0.5, 0.2]))
    assert out.direction_p_up.name == "direction_p_up"
    assert out.magnitude_up.name == "magnitude_up"
    assert out.magnitude_down.name == "magnitude_down"
    with pytest.raises(ValueError):
        dg.prediction_diagnostics(direction_p_up=np.array([0.1]), magnitude_up=np.array([1.0, 2.0]), magnitude_down=np.array([0.5, 0.2]))
    with pytest.raises(ValueError):
        dg.prediction_diagnostics(direction_p_up=np.array([1.1, 0.2]), magnitude_up=np.array([1.0, 2.0]), magnitude_down=np.array([0.5, 0.2]))
    with pytest.raises(ValueError):
        dg.prediction_diagnostics(direction_p_up=np.array([0.1, 0.2]), magnitude_up=np.array([1.0, np.inf]), magnitude_down=np.array([0.5, 0.2]))


def test_build_linear_diagnostics_report():
    head = {"head_name": "direction", "feature_columns": ["a", "b"], "weights": [1.0, -1.0], "intercept": 0.0}
    bundle = {"direction": head, "magnitude_up": {**head, "head_name": "magnitude_up"}, "magnitude_down": {**head, "head_name": "magnitude_down"}}
    prep = {"feature_columns": ["a", "b"], "variance": [1.0, 2.0], "scale": [1.0, 2.0], "active_mask": np.array([True, False], dtype=bool)}
    eval_result = {"some_metric": 1.23}
    report = dg.build_linear_diagnostics_report(model_bundle_state=bundle, preprocess_state=prep, evaluation_result=eval_result, direction_p_up=np.array([0.1, 0.9]), magnitude_up=np.array([1.0, 2.0]), magnitude_down=np.array([0.4, 0.2]), y_direction=np.array([0, 1]))
    assert set(report) == {"diagnostics_version", "config", "coefficients", "preprocess", "predictions", "calibration", "evaluation"}
    assert report["evaluation"] is eval_result


def test_dataclass_validation():
    with pytest.raises(ValueError):
        dg.VectorSummary("x", 1, np.inf, 0, 0, 0, 0, 0, 0, 0, 0)
    rec = dg.CoefficientRecord("a", 1.0, 1.0, 1)
    with pytest.raises(ValueError):
        dg.CoefficientDiagnostics("x", 1, 0.0, 1.0, 1.0, 1.0, ("bad",), (rec,), (rec,))
    with pytest.raises(ValueError):
        dg.CoefficientDiagnostics("x", 1, 0.0, 1.0, 1.0, 1.0, (), (), ())
    dg.CoefficientDiagnostics("x", 1, 0.0, 1.0, 1.0, 1.0, (rec,), (), ())
    vs = dg.summarize_vector(np.array([1.0]), name="x")
    with pytest.raises(ValueError):
        dg.PreprocessDiagnostics(3, 1, 1, vs, vs, ("a",))
    with pytest.raises(ValueError):
        dg.CalibrationBin(0, 0.7, 0.6, 1, 0.1, 0.1)
    with pytest.raises(ValueError):
        dg.CalibrationBin(0, 0.0, 1.0, 1, np.inf, 0.1)
    b = dg.CalibrationBin(0, 0.0, 1.0, 0, np.nan, np.nan)
    with pytest.raises(ValueError):
        dg.CalibrationDiagnostics(1, 1, 2, (b,))
    with pytest.raises(ValueError):
        dg.PredictionDiagnostics(vs, "bad", vs)


def test_no_training_evaluate_or_storage_api():
    for name in ("train", "partial_fit", "fit", "predict", "evaluate_model", "read_split", "transform_table", "save", "load", "write", "plot", "dataframe", "to_csv"):
        assert not hasattr(dg, name)


def test_no_future_leakage_or_timestamp_surface():
    src = inspect.getsource(dg)
    forbidden = [
        "local_" + "ts_us", "ts_" + "us", "event_seq", "raw_mid", "row_idx",
        "future_" + "mid", "future_" + "ret", "timestamp", "shuffle", "sort_values", "random",
        "partial_" + "fit", "fit(", "optimize", "threshold_search",
    ]
    for tok in forbidden:
        assert tok not in src


def test_no_old_pipeline_residue():
    src = inspect.getsource(dg)
    forbidden = [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5",
        "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on", "sklearn", "torch", "pandas", "polars", "pyarrow", "PCA", "StandardScaler",
    ]
    for tok in forbidden:
        assert tok not in src


def test_vectorized_no_pandas_or_row_loop_smoke():
    src = inspect.getsource(dg)
    for tok in (".iterrows", "to_pandas", "for row in", "for sample in"):
        assert tok not in src
