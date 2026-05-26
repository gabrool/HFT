"""Evaluation metrics for storage-backed MMRT linear models.

This module computes metrics from explicit target and prediction arrays. It
does not read storage, build features or targets, fit preprocessing, call
model objects, inspect row timing fields, train models, or write reports.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
PROB_EPS = 1e-12

DIRECTION_DOWN_CLASS = 0
DIRECTION_UP_CLASS = 1
DIRECTION_INVALID_CLASS = -1


def _nan() -> float:
    return float("nan")


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float, got bool")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _require_probability_threshold(value: float, name: str = "threshold") -> float:
    out = _require_finite_float(value, name)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return out


def _coerce_1d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _coerce_direction_classes(values: np.ndarray, *, name: str = "y_direction") -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.issubdtype(arr.dtype, np.bool_):
        raise ValueError(f"{name} must not be boolean")
    if np.issubdtype(arr.dtype, np.number):
        if not np.all(np.isfinite(arr.astype(np.float64, copy=False))):
            raise ValueError(f"{name} must contain finite values")
    try:
        arr_f = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    arr_i = arr_f.astype(np.int64)
    if not np.all(arr_f == arr_i):
        raise ValueError(f"{name} must contain integer values")
    allowed = (
        (arr_i == DIRECTION_INVALID_CLASS)
        | (arr_i == DIRECTION_DOWN_CLASS)
        | (arr_i == DIRECTION_UP_CLASS)
    )
    if not np.all(allowed):
        raise ValueError(f"{name} contains invalid class values")
    return np.ascontiguousarray(arr_i, dtype=np.int8)


def _coerce_bool_mask(values: np.ndarray, *, n_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean dtype")
    if arr.shape[0] != n_rows:
        raise ValueError(f"{name} must have shape ({n_rows},)")
    return np.ascontiguousarray(arr)


def _coerce_probability_vector(values: np.ndarray, *, n_rows: int, name: str = "p_up") -> np.ndarray:
    out = _coerce_1d_float(values, name=name)
    if out.shape[0] != n_rows:
        raise ValueError(f"{name} must have shape ({n_rows},)")
    if np.any((out < 0.0) | (out > 1.0)):
        raise ValueError(f"{name} values must be in [0, 1]")
    return out


def _coerce_regression_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = _coerce_1d_float(y_true, name="y_true")
    yp = _coerce_1d_float(y_pred, name="y_pred")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return yt, yp


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return _nan()
    return float(np.mean(x))


def _rank_average(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _binary_auc(y_true_binary: np.ndarray, score: np.ndarray) -> float:
    if y_true_binary.size == 0:
        return _nan()
    if not np.all((y_true_binary == 0) | (y_true_binary == 1)):
        raise ValueError("y_true_binary must contain only 0/1")
    n_pos = int(np.sum(y_true_binary == 1))
    n_neg = int(np.sum(y_true_binary == 0))
    if n_pos == 0 or n_neg == 0:
        return _nan()
    ranks = _rank_average(score)
    sum_ranks_pos = float(np.sum(ranks[y_true_binary == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = y_true.shape[0]
    if n < 2:
        return _nan()
    a = y_true - np.mean(y_true)
    b = y_pred - np.mean(y_pred)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return _nan()
    return float(np.sum(a * b) / denom)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] < 2:
        return _nan()
    return _pearson(_rank_average(y_true), _rank_average(y_pred))


def _require_metric_float(value: float, name: str) -> float:
    out = float(value)
    if np.isinf(out):
        raise ValueError(f"{name} must be finite or NaN")
    return out


@dataclass(frozen=True, slots=True)
class DirectionMetrics:
    n_rows: int
    valid_count: int
    positive_count: int
    negative_count: int
    accuracy: float
    balanced_accuracy: float
    auc: float
    log_loss: float
    brier: float
    threshold: float

    def __post_init__(self) -> None:
        for name in ("n_rows", "valid_count", "positive_count", "negative_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative int")
        if self.valid_count > self.n_rows:
            raise ValueError("valid_count must be <= n_rows")
        if self.positive_count + self.negative_count != self.valid_count:
            raise ValueError("positive_count + negative_count must equal valid_count")
        object.__setattr__(self, "accuracy", _require_metric_float(self.accuracy, "accuracy"))
        object.__setattr__(self, "balanced_accuracy", _require_metric_float(self.balanced_accuracy, "balanced_accuracy"))
        object.__setattr__(self, "auc", _require_metric_float(self.auc, "auc"))
        object.__setattr__(self, "log_loss", _require_metric_float(self.log_loss, "log_loss"))
        object.__setattr__(self, "brier", _require_metric_float(self.brier, "brier"))
        object.__setattr__(self, "threshold", _require_probability_threshold(self.threshold))

    @property
    def has_both_classes(self) -> bool:
        return self.positive_count > 0 and self.negative_count > 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_rows": int(self.n_rows),
            "valid_count": int(self.valid_count),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "auc": float(self.auc),
            "log_loss": float(self.log_loss),
            "brier": float(self.brier),
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    n_rows: int
    mae: float
    rmse: float
    mean_error: float
    spearman: float
    pearson: float
    y_true_mean: float
    y_pred_mean: float

    def __post_init__(self) -> None:
        if not isinstance(self.n_rows, int) or self.n_rows < 0:
            raise ValueError("n_rows must be a nonnegative int")
        for name in ("mae", "rmse", "mean_error", "spearman", "pearson", "y_true_mean", "y_pred_mean"):
            object.__setattr__(self, name, _require_metric_float(getattr(self, name), name))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_rows": int(self.n_rows),
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "mean_error": float(self.mean_error),
            "spearman": float(self.spearman),
            "pearson": float(self.pearson),
            "y_true_mean": float(self.y_true_mean),
            "y_pred_mean": float(self.y_pred_mean),
        }


@dataclass(frozen=True, slots=True)
class LinearEvaluationResult:
    direction: DirectionMetrics
    magnitude_up: RegressionMetrics
    magnitude_down: RegressionMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.direction, DirectionMetrics):
            raise TypeError("direction must be DirectionMetrics")
        if not isinstance(self.magnitude_up, RegressionMetrics):
            raise TypeError("magnitude_up must be RegressionMetrics")
        if not isinstance(self.magnitude_down, RegressionMetrics):
            raise TypeError("magnitude_down must be RegressionMetrics")

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction.as_dict(),
            "magnitude_up": self.magnitude_up.as_dict(),
            "magnitude_down": self.magnitude_down.as_dict(),
        }


def evaluate_direction(
    y_direction: np.ndarray,
    p_up: np.ndarray,
    *,
    direction_mask: np.ndarray | None = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> DirectionMetrics:
    y = _coerce_direction_classes(y_direction)
    n_rows = y.shape[0]
    p = _coerce_probability_vector(p_up, n_rows=n_rows)
    t = _require_probability_threshold(threshold)

    implied_mask = y != DIRECTION_INVALID_CLASS
    if direction_mask is None:
        mask = implied_mask
    else:
        mask = _coerce_bool_mask(direction_mask, n_rows=n_rows, name="direction_mask")
        if not np.array_equal(mask, implied_mask):
            raise ValueError("direction_mask must match y_direction != DIRECTION_INVALID_CLASS")

    yv = y[mask]
    pv = p[mask]
    valid_count = int(yv.shape[0])

    if valid_count == 0:
        return DirectionMetrics(
            n_rows=n_rows,
            valid_count=0,
            positive_count=0,
            negative_count=0,
            accuracy=_nan(),
            balanced_accuracy=_nan(),
            auc=_nan(),
            log_loss=_nan(),
            brier=_nan(),
            threshold=t,
        )

    if not np.all((yv == DIRECTION_DOWN_CLASS) | (yv == DIRECTION_UP_CLASS)):
        raise ValueError("Valid direction classes must be 0/1")

    pred = (pv >= t).astype(np.int8)
    accuracy = float(np.mean(pred == yv))

    positive_count = int(np.sum(yv == DIRECTION_UP_CLASS))
    negative_count = int(np.sum(yv == DIRECTION_DOWN_CLASS))

    if positive_count > 0 and negative_count > 0:
        tp = int(np.sum((pred == 1) & (yv == 1)))
        tn = int(np.sum((pred == 0) & (yv == 0)))
        tpr = tp / positive_count
        tnr = tn / negative_count
        balanced_accuracy = float(0.5 * (tpr + tnr))
        auc = _binary_auc(yv, pv)
    else:
        balanced_accuracy = _nan()
        auc = _nan()

    p_clip = np.clip(pv, PROB_EPS, 1.0 - PROB_EPS)
    log_loss = float(-np.mean(yv * np.log(p_clip) + (1 - yv) * np.log(1.0 - p_clip)))
    brier = float(np.mean((pv - yv) ** 2))

    return DirectionMetrics(
        n_rows=n_rows,
        valid_count=valid_count,
        positive_count=positive_count,
        negative_count=negative_count,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        auc=auc,
        log_loss=log_loss,
        brier=brier,
        threshold=t,
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    yt, yp = _coerce_regression_arrays(y_true, y_pred)
    n_rows = yt.shape[0]
    if n_rows == 0:
        nan = _nan()
        return RegressionMetrics(n_rows=0, mae=nan, rmse=nan, mean_error=nan, spearman=nan, pearson=nan, y_true_mean=nan, y_pred_mean=nan)

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mean_error = float(np.mean(err))
    spearman = _spearman(yt, yp)
    pearson = _pearson(yt, yp)
    y_true_mean = _safe_mean(yt)
    y_pred_mean = _safe_mean(yp)
    return RegressionMetrics(
        n_rows=n_rows,
        mae=mae,
        rmse=rmse,
        mean_error=mean_error,
        spearman=spearman,
        pearson=pearson,
        y_true_mean=y_true_mean,
        y_pred_mean=y_pred_mean,
    )


def evaluate_linear_predictions(
    *,
    y_direction: np.ndarray,
    direction_p_up: np.ndarray,
    y_magnitude_up: np.ndarray,
    pred_magnitude_up: np.ndarray,
    y_magnitude_down: np.ndarray,
    pred_magnitude_down: np.ndarray,
    direction_mask: np.ndarray | None = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> LinearEvaluationResult:
    n_rows = np.asarray(y_direction).shape[0]
    arrays: Sequence[tuple[str, np.ndarray]] = (
        ("direction_p_up", direction_p_up),
        ("y_magnitude_up", y_magnitude_up),
        ("pred_magnitude_up", pred_magnitude_up),
        ("y_magnitude_down", y_magnitude_down),
        ("pred_magnitude_down", pred_magnitude_down),
    )
    for name, arr in arrays:
        if np.asarray(arr).ndim != 1 or np.asarray(arr).shape[0] != n_rows:
            raise ValueError(f"{name} must be 1D with length matching y_direction")
    if direction_mask is not None and np.asarray(direction_mask).shape[0] != n_rows:
        raise ValueError("direction_mask must have length matching y_direction")

    direction = evaluate_direction(y_direction, direction_p_up, direction_mask=direction_mask, threshold=threshold)
    magnitude_up = evaluate_regression(y_magnitude_up, pred_magnitude_up)
    magnitude_down = evaluate_regression(y_magnitude_down, pred_magnitude_down)
    return LinearEvaluationResult(direction=direction, magnitude_up=magnitude_up, magnitude_down=magnitude_down)


def confusion_counts(
    y_direction: np.ndarray,
    p_up: np.ndarray,
    *,
    direction_mask: np.ndarray | None = None,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict[str, int]:
    y = _coerce_direction_classes(y_direction)
    n_rows = y.shape[0]
    p = _coerce_probability_vector(p_up, n_rows=n_rows)
    t = _require_probability_threshold(threshold)
    implied_mask = y != DIRECTION_INVALID_CLASS
    if direction_mask is None:
        mask = implied_mask
    else:
        mask = _coerce_bool_mask(direction_mask, n_rows=n_rows, name="direction_mask")
        if not np.array_equal(mask, implied_mask):
            raise ValueError("direction_mask must match y_direction != DIRECTION_INVALID_CLASS")
    yv = y[mask]
    if yv.size == 0:
        return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    pv = p[mask]
    pred = (pv >= t).astype(np.int8)
    tp = int(np.sum((pred == 1) & (yv == 1)))
    tn = int(np.sum((pred == 0) & (yv == 0)))
    fp = int(np.sum((pred == 1) & (yv == 0)))
    fn = int(np.sum((pred == 0) & (yv == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


__all__ = [
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "PROB_EPS",
    "DIRECTION_DOWN_CLASS",
    "DIRECTION_UP_CLASS",
    "DIRECTION_INVALID_CLASS",
    "DirectionMetrics",
    "RegressionMetrics",
    "LinearEvaluationResult",
    "evaluate_direction",
    "evaluate_regression",
    "evaluate_linear_predictions",
    "confusion_counts",
]
