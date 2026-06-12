import inspect
import subprocess
import sys

import numpy as np
import pytest

import mmrt.linear.evaluate as ev


def test_public_api_boundary():
    expected = [
        "DEFAULT_CLASSIFICATION_THRESHOLD",
        "PROB_EPS",
        "DIRECTION_DOWN_CLASS",
        "DIRECTION_UP_CLASS",
        "DIRECTION_INVALID_CLASS",
        "DirectionMetrics",
        "RegressionMetrics",
        "LinearEvaluationResult",
        "derive_gated_signal_predictions",
        "evaluate_direction",
        "evaluate_regression",
        "evaluate_linear_predictions",
        "confusion_counts",
    ]
    assert ev.__all__ == expected
    banned = ["bybit", "cmssl", "stage", "pca", "sklearn", "torch", "pandas", "polars", "reader", "writer", "storage", "extract", "preprocess", "train", "model"]
    for name in ev.__all__:
        lname = name.lower()
        assert all(term not in lname for term in banned)


def test_no_forbidden_imports():
    script = (
        "import sys\n"
        "before=set(sys.modules)\n"
        "import mmrt.linear.evaluate\n"
        "after=set(sys.modules)\n"
        "new=after-before\n"
        "print('\\n'.join(sorted(new)))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    loaded = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    forbidden = {
        "pandas", "polars", "torch", "sklearn", "scipy", "numba", "pyarrow",
        "mmrt.storage.reader", "mmrt.storage.writer", "mmrt.storage.splits",
        "mmrt.linear.extractors", "mmrt.linear.targets", "mmrt.linear.preprocess", "mmrt.linear.models",
        "mmrt.features.engine", "mmrt.features.labels", "mmrt.features.transforms",
        "CMSSL17", "offline_ingest",
    }
    for mod in loaded:
        assert all(not (mod == banned or mod.startswith(f"{banned}.")) for banned in forbidden)


def test_direction_metrics_basic():
    y = np.array([0, 0, 1, 1, -1])
    m = np.array([True, True, True, True, False])
    p = np.array([0.1, 0.4, 0.6, 0.9, 0.5])
    out = ev.evaluate_direction(y, p, direction_mask=m, threshold=0.5)
    assert out.accuracy == pytest.approx(1.0)
    assert out.balanced_accuracy == pytest.approx(1.0)
    assert out.auc == pytest.approx(1.0)
    assert np.isfinite(out.log_loss)
    assert np.isfinite(out.brier)
    assert out.valid_count == 4 and out.positive_count == 2 and out.negative_count == 2
    assert out.has_both_classes is True
    assert set(out.as_dict()) == {"n_rows", "valid_count", "positive_count", "negative_count", "positive_rate", "predicted_positive_rate", "accuracy", "balanced_accuracy", "auc", "log_loss", "brier", "threshold"}


def test_direction_metrics_without_mask_uses_invalid_class():
    y = np.array([0, 0, 1, 1, -1])
    p = np.array([0.1, 0.4, 0.6, 0.9, 0.5])
    m = np.array([True, True, True, True, False])
    out_a = ev.evaluate_direction(y, p)
    out_b = ev.evaluate_direction(y, p, direction_mask=m)
    assert out_a == out_b


def test_direction_single_class_auc_balanced_nan():
    y = np.array([1, 1, 1])
    p = np.array([0.2, 0.7, 0.9])
    out = ev.evaluate_direction(y, p, direction_mask=np.array([True, True, True]))
    assert out.valid_count == 3
    assert out.positive_count == 3
    assert out.negative_count == 0
    assert np.isfinite(out.accuracy)
    assert np.isnan(out.auc)
    assert np.isnan(out.balanced_accuracy)
    assert np.isfinite(out.log_loss)
    assert np.isfinite(out.brier)


def test_direction_all_invalid():
    out = ev.evaluate_direction(np.array([-1, -1]), np.array([0.1, 0.9]))
    assert out.n_rows == 2
    assert out.valid_count == 0
    assert out.positive_count == 0
    assert out.negative_count == 0
    assert np.isnan(out.accuracy)
    assert np.isnan(out.balanced_accuracy)
    assert np.isnan(out.auc)
    assert np.isnan(out.log_loss)
    assert np.isnan(out.brier)


def test_direction_validation():
    with pytest.raises(ValueError):
        ev.evaluate_direction(np.array([0, 1]), np.array([0.2, 1.2]))
    with pytest.raises(ValueError):
        ev.evaluate_direction(np.array([2, 1]), np.array([0.2, 0.3]))
    with pytest.raises(ValueError):
        ev.evaluate_direction(np.array([0, -1]), np.array([0.2, 0.3]), direction_mask=np.array([True, True]))
    with pytest.raises(ValueError):
        ev.evaluate_direction(np.array([0, 1]), np.array([0.2, 0.3]), direction_mask=np.array([1, 1]))
    with pytest.raises(ValueError):
        ev.evaluate_direction(np.array([0, 1]), np.array([0.2]))
    for bad in (-0.1, 1.1, np.nan, True):
        with pytest.raises(ValueError):
            ev.evaluate_direction(np.array([0, 1]), np.array([0.2, 0.3]), threshold=bad)
    with pytest.raises(ValueError):
        ev.evaluate_direction(np.array([0, 1]), np.array([0.2, np.inf]))


def test_confusion_counts():
    y = np.array([0, 0, 1, 1, -1])
    p = np.array([0.6, 0.2, 0.7, 0.4, 0.8])
    m = np.array([True, True, True, True, False])
    out = ev.confusion_counts(y, p, direction_mask=m, threshold=0.5)
    assert out == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_binary_auc_with_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.2, 0.8])
    out = ev.evaluate_direction(y, p, direction_mask=np.array([True, True, True, True]))
    assert out.auc == pytest.approx(0.875)


def test_regression_metrics_basic():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 2.0, 1.0])
    out = ev.evaluate_regression(y_true, y_pred)
    assert out.mae == pytest.approx(2.0 / 3.0)
    assert out.rmse == pytest.approx(np.sqrt(2.0 / 3.0))
    assert out.mean_error == pytest.approx(0.0)
    assert np.isfinite(out.pearson)
    assert np.isfinite(out.spearman)
    assert out.y_true_mean == pytest.approx(1.0)
    assert out.y_pred_mean == pytest.approx(1.0)
    assert set(out.as_dict()) == {"n_rows", "mae", "rmse", "mean_error", "spearman", "pearson", "y_true_mean", "y_pred_mean"}


def test_regression_constant_or_empty_correlations_nan():
    empty = ev.evaluate_regression(np.array([]), np.array([]))
    assert empty.n_rows == 0
    assert np.isnan(empty.mae) and np.isnan(empty.rmse) and np.isnan(empty.mean_error)
    assert np.isnan(empty.pearson) and np.isnan(empty.spearman)

    const = ev.evaluate_regression(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(const.pearson)
    assert np.isnan(const.spearman)
    assert np.isfinite(const.mae)
    assert np.isfinite(const.rmse)


def test_regression_validation():
    with pytest.raises(ValueError):
        ev.evaluate_regression(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        ev.evaluate_regression(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        ev.evaluate_regression(np.array([[1.0]]), np.array([[1.0]]))


def test_evaluate_linear_predictions_bundle():
    out = ev.evaluate_linear_predictions(
        y_return_bps=np.array([-2.0, 0.0, 1.5, -1.0], dtype=np.float64),
        y_no_move=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        y_direction=np.array([0, -1, 1, 0], dtype=np.int8),
        no_move_mask=np.array([False, True, False, False]),
        move_mask=np.array([True, False, True, True]),
        up_move_mask=np.array([False, False, True, False]),
        down_move_mask=np.array([True, False, False, True]),
        p_no_move=np.array([0.1, 0.8, 0.2, 0.3]),
        p_up_given_move=np.array([0.2, 0.5, 0.7, 0.3]),
        pred_magnitude_up=np.log1p(np.array([0.5, 0.5, 1.4, 0.5])),
        pred_magnitude_down=np.log1p(np.array([1.8, 0.2, 0.2, 0.9])),
    )
    bundle = out.as_dict()
    assert set(bundle) == {"no_move", "direction", "magnitude_up", "magnitude_down", "gated_signal"}
    assert bundle["no_move"]["n_rows"] == 4
    assert bundle["direction"]["valid_count"] == 3
    assert bundle["magnitude_up"]["n_rows"] == 1
    assert bundle["magnitude_down"]["n_rows"] == 2
    assert set(bundle["gated_signal"]) == {"signed_edge", "abs_move", "n_rows"}


def test_evaluate_linear_predictions_mask_validation():
    kwargs = dict(
        y_return_bps=np.array([-2.0, 0.0, 1.5, -1.0], dtype=np.float64),
        y_no_move=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        y_direction=np.array([0, -1, 1, 0], dtype=np.int8),
        no_move_mask=np.array([False, True, False, False]),
        move_mask=np.array([True, False, True, True]),
        up_move_mask=np.array([False, False, True, False]),
        down_move_mask=np.array([True, False, False, True]),
        p_no_move=np.array([0.1, 0.8, 0.2, 0.3]),
        p_up_given_move=np.array([0.2, 0.5, 0.7, 0.3]),
        pred_magnitude_up=np.log1p(np.array([0.5, 0.5, 1.4, 0.5])),
        pred_magnitude_down=np.log1p(np.array([1.8, 0.2, 0.2, 0.9])),
    )
    with pytest.raises(ValueError, match="no_move_mask"):
        ev.evaluate_linear_predictions(**{**kwargs, "no_move_mask": np.array([True, True, False, False])})
    with pytest.raises(ValueError, match="pred_magnitude_up"):
        ev.evaluate_linear_predictions(**{**kwargs, "pred_magnitude_up": np.array([0.1, 0.2])})
    with pytest.raises(ValueError, match="pred_magnitude_down"):
        ev.evaluate_linear_predictions(**{**kwargs, "pred_magnitude_down": np.array([0.1, 0.2])})
    with pytest.raises(ValueError, match="up class"):
        ev.evaluate_linear_predictions(**{**kwargs, "y_direction": np.array([0, -1, 0, 0], dtype=np.int8)})
    with pytest.raises(ValueError, match="down class"):
        ev.evaluate_linear_predictions(**{**kwargs, "y_direction": np.array([0, -1, 1, -1], dtype=np.int8)})
    with pytest.raises(ValueError, match="invalid class"):
        ev.evaluate_linear_predictions(**{**kwargs, "y_direction": np.array([0, 2, 1, 0], dtype=np.int8)})


def test_derive_gated_signal_predictions():
    g = ev.derive_gated_signal_predictions(
        p_no_move=np.array([0.2]),
        p_up_given_move=np.array([0.75]),
        pred_magnitude_up=np.array([np.log1p(2.0)]),
        pred_magnitude_down=np.array([np.log1p(1.0)]),
    )
    assert g["p_move"][0] == pytest.approx(0.8)
    assert g["p_up_effective"][0] == pytest.approx(0.6)
    assert g["p_down_effective"][0] == pytest.approx(0.2)
    assert g["expected_up_bps"][0] == pytest.approx(1.2)
    assert g["expected_down_bps"][0] == pytest.approx(0.2)
    assert g["expected_signed_edge_bps"][0] == pytest.approx(1.0)
    assert g["expected_abs_move_bps"][0] == pytest.approx(1.4)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            ev.derive_gated_signal_predictions(
                p_no_move=np.array([bad]), p_up_given_move=np.array([0.5]),
                pred_magnitude_up=np.array([0.0]), pred_magnitude_down=np.array([0.0]),
            )
        with pytest.raises(ValueError):
            ev.derive_gated_signal_predictions(
                p_no_move=np.array([0.5]), p_up_given_move=np.array([bad]),
                pred_magnitude_up=np.array([0.0]), pred_magnitude_down=np.array([0.0]),
            )


def test_metrics_dataclass_validation():
    with pytest.raises(ValueError):
        ev.DirectionMetrics(n_rows=-1, valid_count=0, positive_count=0, negative_count=0, positive_rate=0.0, predicted_positive_rate=0.0, accuracy=0.0, balanced_accuracy=0.0, auc=0.0, log_loss=0.0, brier=0.0, threshold=0.5)
    with pytest.raises(ValueError):
        ev.DirectionMetrics(n_rows=2, valid_count=2, positive_count=1, negative_count=0, positive_rate=0.5, predicted_positive_rate=0.5, accuracy=0.0, balanced_accuracy=0.0, auc=0.0, log_loss=0.0, brier=0.0, threshold=0.5)
    ev.DirectionMetrics(n_rows=1, valid_count=1, positive_count=1, negative_count=0, positive_rate=np.nan, predicted_positive_rate=np.nan, accuracy=np.nan, balanced_accuracy=np.nan, auc=np.nan, log_loss=0.1, brier=0.1, threshold=0.5)
    with pytest.raises(ValueError):
        ev.DirectionMetrics(n_rows=1, valid_count=1, positive_count=1, negative_count=0, positive_rate=np.inf, predicted_positive_rate=0.0, accuracy=0.0, balanced_accuracy=0.0, auc=0.0, log_loss=0.0, brier=0.0, threshold=0.5)
    with pytest.raises(ValueError):
        ev.DirectionMetrics(n_rows=1, valid_count=1, positive_count=1, negative_count=0, positive_rate=0.0, predicted_positive_rate=0.0, accuracy=0.0, balanced_accuracy=0.0, auc=0.0, log_loss=0.0, brier=0.0, threshold=1.2)

    with pytest.raises(ValueError):
        ev.RegressionMetrics(1, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    dm = ev.DirectionMetrics(n_rows=1, valid_count=1, positive_count=1, negative_count=0, positive_rate=1.0, predicted_positive_rate=1.0, accuracy=1.0, balanced_accuracy=np.nan, auc=np.nan, log_loss=0.1, brier=0.1, threshold=0.5)
    rm = ev.RegressionMetrics(1, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0)
    ev.LinearEvaluationResult(no_move=dm, direction=dm, magnitude_up=rm, magnitude_down=rm, gated_signal={"signed_edge": {}, "abs_move": {}, "n_rows": 1})


def test_no_model_training_or_storage_api():
    for name in ["evaluate_model", "train", "partial_fit", "fit", "predict", "read_split", "transform_table", "save", "load", "report", "write"]:
        assert not hasattr(ev, name)


def test_no_future_leakage_or_timestamp_surface():
    source = inspect.getsource(ev)
    forbidden = [
        "local_" + "ts_us", "ts_" + "us", "event_seq", "raw_mid", "row_idx",
        "future_" + "mid", "future_" + "ret", "target_" + "column", "label", "timestamp",
        "shuffle", "sort_values", "partial_" + "fit", "fit(",
    ]
    for token in forbidden:
        assert token not in source


def test_no_old_pipeline_residue():
    source = inspect.getsource(ev)
    forbidden = [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5",
        "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on", "sklearn", "torch", "pandas", "polars", "pyarrow", "PCA", "StandardScaler",
    ]
    for token in forbidden:
        assert token not in source


def test_vectorized_no_pandas_or_row_loop_smoke():
    source = inspect.getsource(ev)
    for token in [".iterrows", "to_pandas", "for row in", "for sample in"]:
        assert token not in source
