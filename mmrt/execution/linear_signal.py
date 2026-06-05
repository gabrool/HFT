"""Adapters from supervised linear model predictions to execution signals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from mmrt.execution.contracts import LinearSignal

MAGNITUDE_INPUT_LOG1P_BPS = "log1p_bps"
MAGNITUDE_INPUT_BPS = "bps"
MAGNITUDE_INPUT_MODES = (
    MAGNITUDE_INPUT_LOG1P_BPS,
    MAGNITUDE_INPUT_BPS,
)

DIRECTION_PROBA_KEY = "direction_proba"
NO_MOVE_PROBA_KEY = "no_move_proba"
MAGNITUDE_UP_KEY = "magnitude_up"
MAGNITUDE_DOWN_KEY = "magnitude_down"


@dataclass(frozen=True, slots=True)
class LinearSignalConfig:
    """Configuration for converting linear-model outputs into signals."""

    magnitude_input: str = MAGNITUDE_INPUT_LOG1P_BPS
    probability_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.magnitude_input not in MAGNITUDE_INPUT_MODES:
            raise ValueError(f"magnitude_input must be one of {MAGNITUDE_INPUT_MODES}")
        if isinstance(self.probability_epsilon, bool):
            raise ValueError("probability_epsilon must be a finite float")
        try:
            eps = float(self.probability_epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError("probability_epsilon must be a finite float") from exc
        if not math.isfinite(eps) or eps < 0.0 or eps >= 0.5:
            raise ValueError("probability_epsilon must be finite, >= 0, and < 0.5")
        object.__setattr__(self, "probability_epsilon", eps)


@dataclass(frozen=True, slots=True)
class LinearSignalArrays:
    """Compact vector representation of precomputed linear signals."""

    p_no_move: np.ndarray
    p_up: np.ndarray
    mag_up_bps: np.ndarray
    mag_down_bps: np.ndarray
    expected_return_bps: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "p_no_move": self.p_no_move,
            "p_up": self.p_up,
            "mag_up_bps": self.mag_up_bps,
            "mag_down_bps": self.mag_down_bps,
            "expected_return_bps": self.expected_return_bps,
            "confidence": self.confidence,
        }
        cleaned: dict[str, np.ndarray] = {}
        n_rows: int | None = None
        dtype: np.dtype | None = None

        for name, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name} must be a NumPy array")
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D")
            arr_dtype = np.dtype(arr.dtype)
            if arr_dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError(f"{name} must have float32 or float64 dtype")
            if dtype is None:
                dtype = arr_dtype
            elif arr_dtype != dtype:
                raise ValueError("all arrays must have the same dtype")
            if n_rows is None:
                n_rows = arr.shape[0]
            elif arr.shape[0] != n_rows:
                raise ValueError("all arrays must have the same length")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            cleaned[name] = np.ascontiguousarray(arr, dtype=arr_dtype)

        if not ((cleaned["p_no_move"] >= 0.0).all() and (cleaned["p_no_move"] <= 1.0).all()):
            raise ValueError("p_no_move must be in [0, 1]")
        if not ((cleaned["p_up"] >= 0.0).all() and (cleaned["p_up"] <= 1.0).all()):
            raise ValueError("p_up must be in [0, 1]")
        for name in ("mag_up_bps", "mag_down_bps", "confidence"):
            if (cleaned[name] < 0.0).any():
                raise ValueError(f"{name} must be nonnegative")

        for name, arr in cleaned.items():
            object.__setattr__(self, name, arr)

    @property
    def n_rows(self) -> int:
        return int(self.p_no_move.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.p_no_move.dtype)


def magnitude_to_bps(
    value: float,
    *,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> float:
    """Convert a scalar magnitude output to basis points."""

    config = _require_config(config)
    value = _require_nonnegative_float(value, "value")
    if config.magnitude_input == MAGNITUDE_INPUT_LOG1P_BPS:
        return float(math.expm1(value))
    if config.magnitude_input == MAGNITUDE_INPUT_BPS:
        return value
    raise ValueError(f"unsupported magnitude_input: {config.magnitude_input}")


def expected_return_bps(
    *,
    p_no_move: float,
    p_up: float,
    mag_up_bps: float,
    mag_down_bps: float,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> float:
    """Compute signed expected return in bps from no-move, direction, and magnitude heads."""

    config = _require_config(config)
    p_no_move = _require_probability(p_no_move, "p_no_move", eps=config.probability_epsilon)
    p_up = _require_probability(p_up, "p_up", eps=config.probability_epsilon)
    mag_up_bps = _require_nonnegative_float(mag_up_bps, "mag_up_bps")
    mag_down_bps = _require_nonnegative_float(mag_down_bps, "mag_down_bps")
    p_move = 1.0 - p_no_move
    return float(p_move * (p_up * mag_up_bps - (1.0 - p_up) * mag_down_bps))


def signal_confidence(
    *,
    p_no_move: float,
    p_up: float,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> float:
    """Compute bounded directional confidence from move probability and direction edge."""

    config = _require_config(config)
    p_no_move = _require_probability(p_no_move, "p_no_move", eps=config.probability_epsilon)
    p_up = _require_probability(p_up, "p_up", eps=config.probability_epsilon)
    p_move = 1.0 - p_no_move
    return float(p_move * abs(2.0 * p_up - 1.0))


def make_linear_signal(
    *,
    p_no_move: float,
    p_up: float,
    magnitude_up: float,
    magnitude_down: float,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> LinearSignal:
    """Build a :class:`LinearSignal` from scalar probability and magnitude outputs."""

    config = _require_config(config)
    p_no_move = _require_probability(p_no_move, "p_no_move", eps=config.probability_epsilon)
    p_up = _require_probability(p_up, "p_up", eps=config.probability_epsilon)
    mag_up_bps = magnitude_to_bps(magnitude_up, config=config)
    mag_down_bps = magnitude_to_bps(magnitude_down, config=config)
    expected = expected_return_bps(
        p_no_move=p_no_move,
        p_up=p_up,
        mag_up_bps=mag_up_bps,
        mag_down_bps=mag_down_bps,
        config=config,
    )
    confidence = signal_confidence(p_no_move=p_no_move, p_up=p_up, config=config)
    return LinearSignal(
        p_no_move=p_no_move,
        p_up=p_up,
        mag_up_bps=mag_up_bps,
        mag_down_bps=mag_down_bps,
        expected_return_bps=expected,
        confidence=confidence,
    )


def neutral_linear_signal() -> LinearSignal:
    """Return a neutral signal for rows without a supervised linear prediction."""

    return LinearSignal(
        p_no_move=1.0,
        p_up=0.5,
        mag_up_bps=0.0,
        mag_down_bps=0.0,
        expected_return_bps=0.0,
        confidence=0.0,
    )


def prediction_row_to_signal(
    prediction: Mapping[str, Any],
    row: int,
    *,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> LinearSignal:
    """Convert one row from a linear model prediction dictionary into a signal."""

    config = _require_config(config)
    row = _require_nonnegative_int(row, "row")
    no_move_proba, direction_proba, magnitude_up, magnitude_down = _required_prediction_arrays(
        prediction,
        dtype=np.dtype("float64"),
    )
    row = _require_row_index(row, no_move_proba.shape[0])
    return make_linear_signal(
        p_no_move=float(no_move_proba[row, 1]),
        p_up=float(direction_proba[row, 1]),
        magnitude_up=float(magnitude_up[row]),
        magnitude_down=float(magnitude_down[row]),
        config=config,
    )


def predictions_to_signal_arrays(
    prediction: Mapping[str, Any],
    *,
    config: LinearSignalConfig = LinearSignalConfig(),
    output_dtype: str = "float32",
) -> LinearSignalArrays:
    """Vectorize a batch of linear model predictions into compact signal arrays."""

    config = _require_config(config)
    dtype = _require_output_dtype(output_dtype)
    no_move_proba, direction_proba, magnitude_up, magnitude_down = _required_prediction_arrays(prediction, dtype=dtype)

    no_move_proba = _clean_probability_array(
        no_move_proba,
        eps=config.probability_epsilon,
        name=NO_MOVE_PROBA_KEY,
    )
    direction_proba = _clean_probability_array(
        direction_proba,
        eps=config.probability_epsilon,
        name=DIRECTION_PROBA_KEY,
    )
    p_no_move = no_move_proba[:, 1]
    p_up = direction_proba[:, 1]
    mag_up_bps = _convert_magnitude_array(magnitude_up, config=config, dtype=dtype, name="magnitude_up")
    mag_down_bps = _convert_magnitude_array(magnitude_down, config=config, dtype=dtype, name="magnitude_down")

    one = np.array(1.0, dtype=dtype)
    two = np.array(2.0, dtype=dtype)
    p_move = one - p_no_move
    expected = p_move * (p_up * mag_up_bps - (one - p_up) * mag_down_bps)
    confidence = p_move * np.abs(two * p_up - one)

    return LinearSignalArrays(
        p_no_move=np.ascontiguousarray(p_no_move, dtype=dtype),
        p_up=np.ascontiguousarray(p_up, dtype=dtype),
        mag_up_bps=np.ascontiguousarray(mag_up_bps, dtype=dtype),
        mag_down_bps=np.ascontiguousarray(mag_down_bps, dtype=dtype),
        expected_return_bps=np.ascontiguousarray(expected, dtype=dtype),
        confidence=np.ascontiguousarray(confidence, dtype=dtype),
    )


def linear_signal_at(arrays: LinearSignalArrays, row: int) -> LinearSignal:
    """Return a scalar :class:`LinearSignal` from compact signal arrays."""

    if not isinstance(arrays, LinearSignalArrays):
        raise ValueError("arrays must be a LinearSignalArrays instance")
    row = _require_row_index(row, arrays.n_rows)
    return LinearSignal(
        p_no_move=float(arrays.p_no_move[row]),
        p_up=float(arrays.p_up[row]),
        mag_up_bps=float(arrays.mag_up_bps[row]),
        mag_down_bps=float(arrays.mag_down_bps[row]),
        expected_return_bps=float(arrays.expected_return_bps[row]),
        confidence=float(arrays.confidence[row]),
    )


def _require_config(value: Any) -> LinearSignalConfig:
    if not isinstance(value, LinearSignalConfig):
        raise ValueError("config must be a LinearSignalConfig")
    return value


def _require_probability(value: float, name: str, *, eps: float) -> float:
    value = _require_finite_float(value, name)
    if value < -eps or value > 1.0 + eps:
        raise ValueError(f"{name} must be within [-epsilon, 1 + epsilon]")
    return float(min(max(value, 0.0), 1.0))


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_row_index(row: int, n_rows: int) -> int:
    row = _require_nonnegative_int(row, "row")
    n_rows = _require_nonnegative_int(n_rows, "n_rows")
    if row >= n_rows:
        raise ValueError("row is out of range")
    return row


def _require_output_dtype(value: str) -> np.dtype:
    if value not in ("float32", "float64"):
        raise ValueError('output_dtype must be "float32" or "float64"')
    return np.dtype(value)


def _coerce_vector(values: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=dtype)


def _coerce_proba_matrix(values: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape [n, 2]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=dtype)


def _required_prediction_arrays(
    prediction: Mapping[str, Any],
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(prediction, Mapping):
        raise ValueError("prediction must be a Mapping")
    required = (NO_MOVE_PROBA_KEY, DIRECTION_PROBA_KEY, MAGNITUDE_UP_KEY, MAGNITUDE_DOWN_KEY)
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"prediction is missing required keys: {missing}")

    no_move_proba = _coerce_proba_matrix(prediction[NO_MOVE_PROBA_KEY], name=NO_MOVE_PROBA_KEY, dtype=dtype)
    direction_proba = _coerce_proba_matrix(prediction[DIRECTION_PROBA_KEY], name=DIRECTION_PROBA_KEY, dtype=dtype)
    magnitude_up = _coerce_vector(prediction[MAGNITUDE_UP_KEY], name=MAGNITUDE_UP_KEY, dtype=dtype)
    magnitude_down = _coerce_vector(prediction[MAGNITUDE_DOWN_KEY], name=MAGNITUDE_DOWN_KEY, dtype=dtype)

    n_rows = no_move_proba.shape[0]
    if direction_proba.shape[0] != n_rows or magnitude_up.shape[0] != n_rows or magnitude_down.shape[0] != n_rows:
        raise ValueError("prediction arrays must have the same number of rows")
    return no_move_proba, direction_proba, magnitude_up, magnitude_down


def _clean_probability_array(values: np.ndarray, *, eps: float, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if (values < -eps).any() or (values > 1.0 + eps).any():
        raise ValueError(f"{name} must be within [-epsilon, 1 + epsilon]")
    out = np.array(values, copy=True)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _convert_magnitude_array(
    values: np.ndarray,
    *,
    config: LinearSignalConfig,
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if (values < 0.0).any():
        raise ValueError(f"{name} must be nonnegative")
    if config.magnitude_input == MAGNITUDE_INPUT_LOG1P_BPS:
        return np.ascontiguousarray(np.expm1(values).astype(dtype, copy=False), dtype=dtype)
    if config.magnitude_input == MAGNITUDE_INPUT_BPS:
        return np.ascontiguousarray(values, dtype=dtype)
    raise ValueError(f"unsupported magnitude_input: {config.magnitude_input}")


__all__ = [
    "DIRECTION_PROBA_KEY",
    "MAGNITUDE_DOWN_KEY",
    "MAGNITUDE_INPUT_BPS",
    "MAGNITUDE_INPUT_LOG1P_BPS",
    "MAGNITUDE_INPUT_MODES",
    "MAGNITUDE_UP_KEY",
    "NO_MOVE_PROBA_KEY",
    "LinearSignalArrays",
    "LinearSignalConfig",
    "expected_return_bps",
    "linear_signal_at",
    "magnitude_to_bps",
    "make_linear_signal",
    "neutral_linear_signal",
    "prediction_row_to_signal",
    "predictions_to_signal_arrays",
    "signal_confidence",
]
