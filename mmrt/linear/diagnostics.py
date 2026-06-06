"""Bounded diagnostics for storage-backed MMRT linear models.

This module summarizes already-produced arrays and serialized states for
linear training runs. It does not read storage, build features or targets,
fit preprocessing, call model objects, inspect row timing fields, train
models, evaluate scalar metrics, or write reports.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_TOP_K = 25
DEFAULT_NUM_BINS = 10
DEFAULT_MAX_ROWS = 200_000

DIRECTION_DOWN_CLASS = 0
DIRECTION_UP_CLASS = 1
DIRECTION_INVALID_CLASS = -1

NO_MOVE_HEAD = "no_move"
DIRECTION_HEAD = "direction"
MAGNITUDE_UP_HEAD = "magnitude_up"
MAGNITUDE_DOWN_HEAD = "magnitude_down"
MODEL_HEADS = (NO_MOVE_HEAD, DIRECTION_HEAD, MAGNITUDE_UP_HEAD, MAGNITUDE_DOWN_HEAD)

LINEAR_PREPROCESS_SCHEMA = "mmrt_linear_preprocess"


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be non-empty")
    return stripped


def _coerce_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(feature_columns)
    if not columns:
        raise ValueError("feature_columns must be non-empty")
    cleaned: list[str] = []
    seen: set[str] = set()
    for idx, col in enumerate(columns):
        name = _require_non_empty_str(col, f"feature_columns[{idx}]")
        if name in seen:
            raise ValueError("feature_columns contains duplicates")
        seen.add(name)
        cleaned.append(name)
    return tuple(cleaned)


def _coerce_1d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    out = np.ascontiguousarray(arr.astype(np.float64, copy=False))
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _coerce_1d_bool(values: np.ndarray, *, n_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype != np.bool_:
        raise ValueError(f"{name} must be bool dtype")
    if arr.shape[0] != n_rows:
        raise ValueError(f"{name} length mismatch")
    return np.ascontiguousarray(arr)


def _coerce_direction_classes(values: np.ndarray, *, name: str = "y_direction") -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype == np.bool_:
        raise ValueError(f"{name} must be integer class values")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    as_float = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(as_float)):
        raise ValueError(f"{name} must be finite")
    if not np.all(as_float == np.floor(as_float)):
        raise ValueError(f"{name} must contain integer values")
    as_int = as_float.astype(np.int64, copy=False)
    allowed = (as_int == DIRECTION_INVALID_CLASS) | (as_int == DIRECTION_DOWN_CLASS) | (as_int == DIRECTION_UP_CLASS)
    if not np.all(allowed):
        raise ValueError(f"{name} has invalid class values")
    return np.ascontiguousarray(as_int.astype(np.int8, copy=False))


def _coerce_probability_vector(values: np.ndarray, *, n_rows: int, name: str) -> np.ndarray:
    arr = _coerce_1d_float(values, name=name)
    if arr.shape[0] != n_rows:
        raise ValueError(f"{name} length mismatch")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return arr


def _bounded_indices(n_rows: int, max_rows: int) -> np.ndarray:
    rows = _require_nonnegative_int(n_rows, "n_rows")
    max_n = _require_positive_int(max_rows, "max_rows")
    if rows <= max_n:
        return np.arange(rows, dtype=np.int64)
    idx = np.linspace(0, rows - 1, max_n, dtype=np.int64)
    return np.unique(idx)


def _metric_float(value: float) -> float:
    out = float(value)
    if np.isinf(out):
        raise ValueError("metric must not be inf")
    return out


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    top_k: int = DEFAULT_TOP_K
    num_bins: int = DEFAULT_NUM_BINS
    max_rows: int = DEFAULT_MAX_ROWS

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_k", _require_positive_int(self.top_k, "top_k"))
        object.__setattr__(self, "num_bins", _require_positive_int(self.num_bins, "num_bins"))
        object.__setattr__(self, "max_rows", _require_positive_int(self.max_rows, "max_rows"))


@dataclass(frozen=True, slots=True)
class VectorSummary:
    name: str
    n_rows: int
    mean: float
    std: float
    min: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty_str(self.name, "name"))
        object.__setattr__(self, "n_rows", _require_nonnegative_int(self.n_rows, "n_rows"))
        for field in ("mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max"):
            object.__setattr__(self, field, _metric_float(getattr(self, field)))

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "p01": self.p01,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
        }


def summarize_vector(values: np.ndarray, *, name: str, config: DiagnosticsConfig | None = None) -> VectorSummary:
    cfg = config or DiagnosticsConfig()
    vector_name = _require_non_empty_str(name, "name")
    arr = _coerce_1d_float(values, name=vector_name)
    if arr.shape[0] == 0:
        nan = float("nan")
        return VectorSummary(vector_name, 0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    sample = arr[_bounded_indices(arr.shape[0], cfg.max_rows)]
    return VectorSummary(
        name=vector_name,
        n_rows=int(arr.shape[0]),
        mean=float(np.mean(sample)),
        std=float(np.std(sample, ddof=0)),
        min=float(np.min(sample)),
        p01=float(np.quantile(sample, 0.01)),
        p05=float(np.quantile(sample, 0.05)),
        p50=float(np.quantile(sample, 0.50)),
        p95=float(np.quantile(sample, 0.95)),
        p99=float(np.quantile(sample, 0.99)),
        max=float(np.max(sample)),
    )


@dataclass(frozen=True, slots=True)
class CoefficientRecord:
    feature: str
    coefficient: float
    abs_coefficient: float
    rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature", _require_non_empty_str(self.feature, "feature"))
        coef = float(self.coefficient)
        abs_coef = float(self.abs_coefficient)
        if not np.isfinite(coef):
            raise ValueError("coefficient must be finite")
        if not np.isfinite(abs_coef) or abs_coef < 0.0:
            raise ValueError("abs_coefficient must be finite and nonnegative")
        if not np.isclose(abs_coef, abs(coef), rtol=0.0, atol=1e-15):
            raise ValueError("abs_coefficient must match abs(coefficient)")
        object.__setattr__(self, "coefficient", coef)
        object.__setattr__(self, "abs_coefficient", abs_coef)
        object.__setattr__(self, "rank", _require_positive_int(self.rank, "rank"))

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "coefficient": self.coefficient,
            "abs_coefficient": self.abs_coefficient,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class CoefficientDiagnostics:
    head_name: str
    n_features: int
    intercept: float
    l1_norm: float
    l2_norm: float
    max_abs: float
    top_abs: tuple[CoefficientRecord, ...]
    top_positive: tuple[CoefficientRecord, ...]
    top_negative: tuple[CoefficientRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_name", _require_non_empty_str(self.head_name, "head_name"))
        object.__setattr__(self, "n_features", _require_positive_int(self.n_features, "n_features"))
        for field in ("intercept", "l1_norm", "l2_norm", "max_abs"):
            value = float(getattr(self, field))
            if not np.isfinite(value):
                raise ValueError(f"{field} must be finite")
            if field != "intercept" and value < 0.0:
                raise ValueError(f"{field} must be nonnegative")
            object.__setattr__(self, field, value)
        for field in ("top_abs", "top_positive", "top_negative"):
            records = getattr(self, field)
            if not isinstance(records, tuple) or not all(isinstance(r, CoefficientRecord) for r in records):
                raise ValueError(f"{field} must contain CoefficientRecord values")

        if self.n_features > 0 and len(self.top_abs) == 0:
            raise ValueError("top_abs must be non-empty when n_features > 0")

    def as_dict(self) -> dict[str, object]:
        return {
            "head_name": self.head_name,
            "n_features": self.n_features,
            "intercept": self.intercept,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "max_abs": self.max_abs,
            "top_abs": [r.as_dict() for r in self.top_abs],
            "top_positive": [r.as_dict() for r in self.top_positive],
            "top_negative": [r.as_dict() for r in self.top_negative],
        }


def coefficient_diagnostics(*, head_name: str, feature_columns: Sequence[str], weights: np.ndarray, intercept: float, config: DiagnosticsConfig | None = None) -> CoefficientDiagnostics:
    cfg = config or DiagnosticsConfig()
    clean_head = _require_non_empty_str(head_name, "head_name")
    cols = _coerce_feature_columns(feature_columns)
    w = _coerce_1d_float(weights, name="weights")
    if w.shape[0] != len(cols):
        raise ValueError("weights length mismatch")
    intercept_value = float(intercept)
    if not np.isfinite(intercept_value):
        raise ValueError("intercept must be finite")
    top_k = min(cfg.top_k, len(cols))
    abs_w = np.abs(w)

    def _records(indices: np.ndarray) -> tuple[CoefficientRecord, ...]:
        return tuple(
            CoefficientRecord(feature=cols[int(i)], coefficient=float(w[int(i)]), abs_coefficient=float(abs_w[int(i)]), rank=rank)
            for rank, i in enumerate(indices, start=1)
        )

    top_abs_idx = np.argsort(-abs_w, kind="stable")[:top_k]
    pos_idx_all = np.flatnonzero(w > 0.0)
    pos_sorted = pos_idx_all[np.argsort(-w[pos_idx_all], kind="stable")][:top_k]
    neg_idx_all = np.flatnonzero(w < 0.0)
    neg_sorted = neg_idx_all[np.argsort(w[neg_idx_all], kind="stable")][:top_k]

    return CoefficientDiagnostics(
        head_name=clean_head,
        n_features=len(cols),
        intercept=intercept_value,
        l1_norm=float(np.sum(abs_w)),
        l2_norm=float(np.sqrt(np.sum(w * w))),
        max_abs=float(np.max(abs_w)),
        top_abs=_records(top_abs_idx),
        top_positive=_records(pos_sorted),
        top_negative=_records(neg_sorted),
    )


def coefficient_diagnostics_from_head_dict(head_state: dict[str, object], *, config: DiagnosticsConfig | None = None) -> CoefficientDiagnostics:
    required = ("head_name", "feature_columns", "weights", "intercept")
    for key in required:
        if key not in head_state:
            raise ValueError(f"missing required key: {key}")
    return coefficient_diagnostics(
        head_name=head_state["head_name"],
        feature_columns=head_state["feature_columns"],
        weights=head_state["weights"],
        intercept=head_state["intercept"],
        config=config,
    )


def coefficient_diagnostics_from_bundle_dict(bundle_state: dict[str, object], *, config: DiagnosticsConfig | None = None) -> dict[str, dict[str, object]]:
    if not isinstance(bundle_state, dict):
        raise ValueError("bundle_state must be a dict")
    output: dict[str, dict[str, object]] = {}
    for key in MODEL_HEADS:
        if key not in bundle_state:
            raise ValueError(f"missing required key: {key}")
        output[key] = coefficient_diagnostics_from_head_dict(bundle_state[key], config=config).as_dict()
    return output


@dataclass(frozen=True, slots=True)
class PreprocessDiagnostics:
    n_features: int
    active_count: int
    inactive_count: int
    scale_summary: VectorSummary
    variance_summary: VectorSummary
    inactive_features: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_features", _require_nonnegative_int(self.n_features, "n_features"))
        object.__setattr__(self, "active_count", _require_nonnegative_int(self.active_count, "active_count"))
        object.__setattr__(self, "inactive_count", _require_nonnegative_int(self.inactive_count, "inactive_count"))
        if self.active_count + self.inactive_count != self.n_features:
            raise ValueError("active_count + inactive_count must equal n_features")
        if not isinstance(self.scale_summary, VectorSummary) or not isinstance(self.variance_summary, VectorSummary):
            raise ValueError("summaries must be VectorSummary")
        if not isinstance(self.inactive_features, tuple):
            raise ValueError("inactive_features must be tuple")
        if len(self.inactive_features) != self.inactive_count:
            raise ValueError("inactive_features length mismatch")
        for idx, name in enumerate(self.inactive_features):
            _require_non_empty_str(name, f"inactive_features[{idx}]")

    def as_dict(self) -> dict[str, object]:
        return {
            "n_features": self.n_features,
            "active_count": self.active_count,
            "inactive_count": self.inactive_count,
            "scale_summary": self.scale_summary.as_dict(),
            "variance_summary": self.variance_summary.as_dict(),
            "inactive_features": list(self.inactive_features),
        }


def preprocess_diagnostics_from_state_dict(state: dict[str, object], *, config: DiagnosticsConfig | None = None) -> PreprocessDiagnostics:
    cfg = config or DiagnosticsConfig()
    for key in ("feature_columns", "variance", "scale", "active_mask"):
        if key not in state:
            raise ValueError(f"missing required key: {key}")
    cols = _coerce_feature_columns(state["feature_columns"])
    variance = _coerce_1d_float(state["variance"], name="variance")
    scale = _coerce_1d_float(state["scale"], name="scale")
    mask = _coerce_1d_bool(state["active_mask"], n_rows=len(cols), name="active_mask")
    if variance.shape[0] != len(cols) or scale.shape[0] != len(cols):
        raise ValueError("variance/scale length mismatch")
    if np.any(variance < 0.0):
        raise ValueError("variance must be nonnegative")
    if np.any(scale <= 0.0):
        raise ValueError("scale must be positive")
    inactive_features = tuple(cols[i] for i in np.flatnonzero(~mask))
    return PreprocessDiagnostics(
        n_features=len(cols),
        active_count=int(np.sum(mask)),
        inactive_count=int(np.sum(~mask)),
        scale_summary=summarize_vector(scale, name="scale", config=cfg),
        variance_summary=summarize_vector(variance, name="variance", config=cfg),
        inactive_features=inactive_features,
    )


def preprocess_diagnostics_from_train_state_dict(
    state: dict[str, object],
    *,
    config: DiagnosticsConfig | None = None,
) -> dict[str, object]:
    if not isinstance(state, dict):
        raise ValueError("preprocess_state must be a dict")

    if state.get("schema") != LINEAR_PREPROCESS_SCHEMA:
        return preprocess_diagnostics_from_state_dict(state, config=config).as_dict()

    states_by_head = state.get("states_by_head")
    if not isinstance(states_by_head, dict):
        raise ValueError("states_by_head must be a dict")

    if set(states_by_head.keys()) != set(MODEL_HEADS):
        raise ValueError("states_by_head keys must exactly match model heads")

    return {
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "states_by_head": {
            head: preprocess_diagnostics_from_state_dict(
                states_by_head[head],
                config=config,
            ).as_dict()
            for head in MODEL_HEADS
        },
    }


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    bin_index: int
    left: float
    right: float
    count: int
    mean_predicted: float
    empirical_positive_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "bin_index", _require_nonnegative_int(self.bin_index, "bin_index"))
        left = float(self.left)
        right = float(self.right)
        if not np.isfinite(left) or not np.isfinite(right) or not (0.0 <= left <= right <= 1.0):
            raise ValueError("invalid bin bounds")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "count", _require_nonnegative_int(self.count, "count"))
        mp = _metric_float(self.mean_predicted)
        epr = _metric_float(self.empirical_positive_rate)
        if self.count == 0 and (not np.isnan(mp) or not np.isnan(epr)):
            raise ValueError("empty bins must have NaN metrics")
        object.__setattr__(self, "mean_predicted", mp)
        object.__setattr__(self, "empirical_positive_rate", epr)

    def as_dict(self) -> dict[str, object]:
        return {
            "bin_index": self.bin_index,
            "left": self.left,
            "right": self.right,
            "count": self.count,
            "mean_predicted": self.mean_predicted,
            "empirical_positive_rate": self.empirical_positive_rate,
        }


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    n_rows: int
    valid_count: int
    num_bins: int
    bins: tuple[CalibrationBin, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_rows", _require_nonnegative_int(self.n_rows, "n_rows"))
        object.__setattr__(self, "valid_count", _require_nonnegative_int(self.valid_count, "valid_count"))
        if self.valid_count > self.n_rows:
            raise ValueError("valid_count must be <= n_rows")
        object.__setattr__(self, "num_bins", _require_positive_int(self.num_bins, "num_bins"))
        if not isinstance(self.bins, tuple) or len(self.bins) != self.num_bins or not all(isinstance(b, CalibrationBin) for b in self.bins):
            raise ValueError("bins must be tuple[CalibrationBin, ...] matching num_bins")

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "valid_count": self.valid_count,
            "num_bins": self.num_bins,
            "bins": [b.as_dict() for b in self.bins],
        }


def direction_calibration_diagnostics(y_direction: np.ndarray, p_up: np.ndarray, *, direction_mask: np.ndarray | None = None, config: DiagnosticsConfig | None = None) -> CalibrationDiagnostics:
    cfg = config or DiagnosticsConfig()
    y = _coerce_direction_classes(y_direction)
    p = _coerce_probability_vector(p_up, n_rows=y.shape[0], name="p_up")
    implied_mask = y != DIRECTION_INVALID_CLASS
    if direction_mask is None:
        mask = implied_mask
    else:
        mask = _coerce_1d_bool(direction_mask, n_rows=y.shape[0], name="direction_mask")
        if not np.array_equal(mask, implied_mask):
            raise ValueError("direction_mask must equal implied valid-direction mask")
    yv = y[mask]
    pv = p[mask]
    valid_count = int(yv.shape[0])
    idx = _bounded_indices(valid_count, cfg.max_rows)
    y_sample = yv[idx]
    p_sample = pv[idx]
    edges = np.linspace(0.0, 1.0, cfg.num_bins + 1)
    bins: list[CalibrationBin] = []
    for k in range(cfg.num_bins):
        left = float(edges[k])
        right = float(edges[k + 1])
        in_bin = (p_sample >= left) & ((p_sample < right) if k < (cfg.num_bins - 1) else (p_sample <= right))
        count = int(np.sum(in_bin))
        if count == 0:
            mean_pred = float("nan")
            pos_rate = float("nan")
        else:
            p_bin = p_sample[in_bin]
            y_bin = y_sample[in_bin]
            mean_pred = float(np.mean(p_bin))
            pos_rate = float(np.mean(y_bin == DIRECTION_UP_CLASS))
        bins.append(CalibrationBin(k, left, right, count, mean_pred, pos_rate))
    return CalibrationDiagnostics(n_rows=int(y.shape[0]), valid_count=valid_count, num_bins=cfg.num_bins, bins=tuple(bins))


def prediction_diagnostics(*, p_no_move: np.ndarray, p_move: np.ndarray, p_up_given_move: np.ndarray, p_up_effective: np.ndarray, p_down_effective: np.ndarray, magnitude_up: np.ndarray, magnitude_down: np.ndarray, expected_up_bps: np.ndarray, expected_down_bps: np.ndarray, expected_signed_edge_bps: np.ndarray, expected_abs_move_bps: np.ndarray, config: DiagnosticsConfig | None = None) -> dict[str, object]:
    cfg = config or DiagnosticsConfig()
    return {
        "p_no_move": summarize_vector(p_no_move, name="p_no_move", config=cfg).as_dict(),
        "p_move": summarize_vector(p_move, name="p_move", config=cfg).as_dict(),
        "p_up_given_move": summarize_vector(p_up_given_move, name="p_up_given_move", config=cfg).as_dict(),
        "p_up_effective": summarize_vector(p_up_effective, name="p_up_effective", config=cfg).as_dict(),
        "p_down_effective": summarize_vector(p_down_effective, name="p_down_effective", config=cfg).as_dict(),
        "magnitude_up": summarize_vector(magnitude_up, name="magnitude_up", config=cfg).as_dict(),
        "magnitude_down": summarize_vector(magnitude_down, name="magnitude_down", config=cfg).as_dict(),
        "expected_up_bps": summarize_vector(expected_up_bps, name="expected_up_bps", config=cfg).as_dict(),
        "expected_down_bps": summarize_vector(expected_down_bps, name="expected_down_bps", config=cfg).as_dict(),
        "expected_signed_edge_bps": summarize_vector(expected_signed_edge_bps, name="expected_signed_edge_bps", config=cfg).as_dict(),
        "expected_abs_move_bps": summarize_vector(expected_abs_move_bps, name="expected_abs_move_bps", config=cfg).as_dict(),
    }


def build_linear_diagnostics_report(*, model_bundle_state: dict[str, object], preprocess_state: dict[str, object], evaluation_result: dict[str, object], p_no_move: np.ndarray, p_move: np.ndarray, p_up_given_move: np.ndarray, p_up_effective: np.ndarray, p_down_effective: np.ndarray, magnitude_up: np.ndarray, magnitude_down: np.ndarray, expected_up_bps: np.ndarray, expected_down_bps: np.ndarray, expected_signed_edge_bps: np.ndarray, expected_abs_move_bps: np.ndarray, y_no_move: np.ndarray, y_direction: np.ndarray, move_mask: np.ndarray | None = None, config: DiagnosticsConfig | None = None) -> dict[str, object]:
    cfg = config or DiagnosticsConfig()
    if not isinstance(evaluation_result, dict):
        raise ValueError("evaluation_result must be a dict")
    return {
        "diagnostics_version": 1,
        "config": {"top_k": cfg.top_k, "num_bins": cfg.num_bins, "max_rows": cfg.max_rows},
        "coefficients": coefficient_diagnostics_from_bundle_dict(model_bundle_state, config=cfg),
        "preprocess": preprocess_diagnostics_from_train_state_dict(preprocess_state, config=cfg),
        "predictions": prediction_diagnostics(
            p_no_move=p_no_move, p_move=p_move, p_up_given_move=p_up_given_move, p_up_effective=p_up_effective, p_down_effective=p_down_effective, magnitude_up=magnitude_up, magnitude_down=magnitude_down, expected_up_bps=expected_up_bps, expected_down_bps=expected_down_bps, expected_signed_edge_bps=expected_signed_edge_bps, expected_abs_move_bps=expected_abs_move_bps, config=cfg,
        ),
        "calibration": {
            "no_move": direction_calibration_diagnostics(y_no_move.astype(np.int8), p_no_move, direction_mask=np.ones_like(y_no_move, dtype=bool), config=cfg).as_dict(),
            "direction": direction_calibration_diagnostics(y_direction, p_up_given_move, direction_mask=move_mask, config=cfg).as_dict(),
        },
        "evaluation": evaluation_result,
    }


__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_NUM_BINS",
    "DEFAULT_MAX_ROWS",
    "DIRECTION_DOWN_CLASS",
    "DIRECTION_UP_CLASS",
    "DIRECTION_INVALID_CLASS",
    "DiagnosticsConfig",
    "VectorSummary",
    "CoefficientRecord",
    "CoefficientDiagnostics",
    "PreprocessDiagnostics",
    "CalibrationBin",
    "CalibrationDiagnostics",
    "summarize_vector",
    "coefficient_diagnostics",
    "coefficient_diagnostics_from_head_dict",
    "coefficient_diagnostics_from_bundle_dict",
    "preprocess_diagnostics_from_state_dict",
    "preprocess_diagnostics_from_train_state_dict",
    "direction_calibration_diagnostics",
    "prediction_diagnostics",
    "build_linear_diagnostics_report",
]
