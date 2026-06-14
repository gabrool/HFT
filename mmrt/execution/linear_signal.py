"""Adapters from supervised linear model predictions to no-move-gated execution signals."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mmrt.execution.contracts import LinearSignal
from mmrt.execution.decision_grid import DecisionGrid
from mmrt.features.schedule import decision_schedule_config_from_dict

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

LINEAR_SIGNAL_ARTIFACT_SCHEMA = "mmrt_execution_linear_signals_grid_v1"
LINEAR_SIGNALS_FILENAME = "linear_signals.npz"
_LINEAR_SIGNAL_CONSISTENCY_EPS = 1e-5
_LINEAR_SIGNAL_ARRAY_FIELDS = (
    "p_no_move",
    "p_move",
    "p_up_move",
    "p_down_move",
    "signed_move_prob",
    "expected_up_bps",
    "expected_down_bps",
    "expected_return_bps",
    "expected_abs_move_bps",
    "predicted_vol_bps",
    "confidence",
)


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
    """Validated vector representation of no-move-gated linear execution signals."""

    p_no_move: np.ndarray
    p_move: np.ndarray

    p_up_move: np.ndarray
    p_down_move: np.ndarray
    signed_move_prob: np.ndarray

    expected_up_bps: np.ndarray
    expected_down_bps: np.ndarray
    expected_return_bps: np.ndarray
    expected_abs_move_bps: np.ndarray
    predicted_vol_bps: np.ndarray

    confidence: np.ndarray

    def __post_init__(self) -> None:
        arrays = {name: getattr(self, name) for name in _LINEAR_SIGNAL_ARRAY_FIELDS}
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
                n_rows = int(arr.shape[0])
            elif int(arr.shape[0]) != n_rows:
                raise ValueError("all arrays must have the same length")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
            cleaned[name] = np.ascontiguousarray(arr, dtype=arr_dtype)
        if n_rows is None or n_rows == 0:
            raise ValueError("linear signal arrays must contain at least one row")

        for name in ("p_no_move", "p_move", "p_up_move", "p_down_move"):
            arr = cleaned[name]
            if ((arr < 0.0) | (arr > 1.0)).any():
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.allclose(
            cleaned["p_no_move"] + cleaned["p_move"],
            1.0,
            rtol=0.0,
            atol=_LINEAR_SIGNAL_CONSISTENCY_EPS,
        ):
            raise ValueError("p_no_move + p_move must be approximately 1")
        if not np.allclose(
            cleaned["p_up_move"] + cleaned["p_down_move"],
            cleaned["p_move"],
            rtol=0.0,
            atol=_LINEAR_SIGNAL_CONSISTENCY_EPS,
        ):
            raise ValueError("p_up_move + p_down_move must be approximately p_move")
        if (np.abs(cleaned["signed_move_prob"]) > 1.0 + _LINEAR_SIGNAL_CONSISTENCY_EPS).any():
            raise ValueError("signed_move_prob must have abs <= 1 + tolerance")
        for name in ("expected_up_bps", "expected_down_bps", "expected_abs_move_bps", "predicted_vol_bps", "confidence"):
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


def magnitude_to_bps(value: float, *, config: LinearSignalConfig = LinearSignalConfig()) -> float:
    """Convert a scalar magnitude output to basis points."""

    config = _require_config(config)
    value = _require_nonnegative_float(value, "value")
    if config.magnitude_input == MAGNITUDE_INPUT_LOG1P_BPS:
        return float(math.expm1(value))
    if config.magnitude_input == MAGNITUDE_INPUT_BPS:
        return value
    raise ValueError(f"unsupported magnitude_input: {config.magnitude_input}")


def build_gated_linear_signal(
    *,
    p_no_move: float,
    p_up: float,
    magnitude_up: float,
    magnitude_down: float,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> LinearSignal:
    """Build a scalar no-move-gated execution signal from raw linear-head outputs."""

    config = _require_config(config)
    p_no_move = _require_probability(p_no_move, "p_no_move", eps=config.probability_epsilon)
    p_up = _require_probability(p_up, "p_up", eps=config.probability_epsilon)
    mag_up_bps = magnitude_to_bps(magnitude_up, config=config)
    mag_down_bps = magnitude_to_bps(magnitude_down, config=config)

    p_move = 1.0 - p_no_move
    p_up_move = p_move * p_up
    p_down_move = p_move * (1.0 - p_up)
    signed_move_prob = p_up_move - p_down_move
    expected_up_bps = p_up_move * mag_up_bps
    expected_down_bps = p_down_move * mag_down_bps
    expected_return_bps = expected_up_bps - expected_down_bps
    expected_abs_move_bps = expected_up_bps + expected_down_bps
    second_moment_bps2 = p_up_move * mag_up_bps * mag_up_bps + p_down_move * mag_down_bps * mag_down_bps
    variance_bps2 = max(second_moment_bps2 - expected_return_bps * expected_return_bps, 0.0)
    predicted_vol_bps = math.sqrt(variance_bps2)
    confidence = abs(signed_move_prob)

    return LinearSignal(
        p_no_move=p_no_move,
        p_move=p_move,
        p_up_move=p_up_move,
        p_down_move=p_down_move,
        signed_move_prob=signed_move_prob,
        expected_up_bps=expected_up_bps,
        expected_down_bps=expected_down_bps,
        expected_return_bps=expected_return_bps,
        expected_abs_move_bps=expected_abs_move_bps,
        predicted_vol_bps=predicted_vol_bps,
        confidence=confidence,
    )


def prediction_row_to_signal(
    prediction: Mapping[str, Any],
    row: int,
    *,
    config: LinearSignalConfig = LinearSignalConfig(),
) -> LinearSignal:
    """Convert one row from a linear model prediction dictionary into a gated signal."""

    config = _require_config(config)
    row = _require_nonnegative_int(row, "row")
    no_move_proba, direction_proba, magnitude_up, magnitude_down = _required_prediction_arrays(
        prediction,
        dtype=np.dtype("float64"),
    )
    row = _require_row_index(row, no_move_proba.shape[0])
    return build_gated_linear_signal(
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
    """Vectorize raw linear model predictions into no-move-gated signal arrays."""

    config = _require_config(config)
    dtype = _require_output_dtype(output_dtype)
    no_move_proba, direction_proba, magnitude_up, magnitude_down = _required_prediction_arrays(prediction, dtype=dtype)
    if no_move_proba.shape[0] == 0:
        raise ValueError("prediction arrays must contain at least one row")

    no_move_proba = _clean_probability_array(no_move_proba, eps=config.probability_epsilon, name=NO_MOVE_PROBA_KEY)
    direction_proba = _clean_probability_array(direction_proba, eps=config.probability_epsilon, name=DIRECTION_PROBA_KEY)
    p_no_move = no_move_proba[:, 1]
    p_up = direction_proba[:, 1]
    mag_up_bps = _convert_magnitude_array(magnitude_up, config=config, dtype=dtype, name=MAGNITUDE_UP_KEY)
    mag_down_bps = _convert_magnitude_array(magnitude_down, config=config, dtype=dtype, name=MAGNITUDE_DOWN_KEY)

    one = np.array(1.0, dtype=dtype)
    p_move = one - p_no_move
    p_up_move = p_move * p_up
    p_down_move = p_move * (one - p_up)
    signed_move_prob = p_up_move - p_down_move
    expected_up_bps = p_up_move * mag_up_bps
    expected_down_bps = p_down_move * mag_down_bps
    expected_return_bps = expected_up_bps - expected_down_bps
    expected_abs_move_bps = expected_up_bps + expected_down_bps
    second_moment_bps2 = p_up_move * mag_up_bps * mag_up_bps + p_down_move * mag_down_bps * mag_down_bps
    variance_bps2 = second_moment_bps2 - expected_return_bps * expected_return_bps
    variance_bps2 = np.maximum(variance_bps2, np.array(0.0, dtype=dtype))
    predicted_vol_bps = np.sqrt(variance_bps2).astype(dtype, copy=False)
    confidence = np.abs(signed_move_prob)

    return LinearSignalArrays(
        p_no_move=np.ascontiguousarray(p_no_move, dtype=dtype),
        p_move=np.ascontiguousarray(p_move, dtype=dtype),
        p_up_move=np.ascontiguousarray(p_up_move, dtype=dtype),
        p_down_move=np.ascontiguousarray(p_down_move, dtype=dtype),
        signed_move_prob=np.ascontiguousarray(signed_move_prob, dtype=dtype),
        expected_up_bps=np.ascontiguousarray(expected_up_bps, dtype=dtype),
        expected_down_bps=np.ascontiguousarray(expected_down_bps, dtype=dtype),
        expected_return_bps=np.ascontiguousarray(expected_return_bps, dtype=dtype),
        expected_abs_move_bps=np.ascontiguousarray(expected_abs_move_bps, dtype=dtype),
        predicted_vol_bps=np.ascontiguousarray(predicted_vol_bps, dtype=dtype),
        confidence=np.ascontiguousarray(confidence, dtype=dtype),
    )


def linear_signal_at(arrays: LinearSignalArrays, row: int) -> LinearSignal:
    """Return a scalar :class:`LinearSignal` from compact signal arrays."""

    if not isinstance(arrays, LinearSignalArrays):
        raise ValueError("arrays must be a LinearSignalArrays instance")
    row = _require_row_index(row, arrays.n_rows)
    return LinearSignal(**{name: float(getattr(arrays, name)[row]) for name in _LINEAR_SIGNAL_ARRAY_FIELDS})



@dataclass(frozen=True, slots=True)
class LinearSignalArtifactMetadata:
    tape_schema: str
    exchange: str
    symbol: str
    num_events: int
    num_l2_batches: int
    num_trades: int
    start_local_ts_us: int
    end_local_ts_us: int
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: dict[str, object]
    start_event_index: int
    n_rows: int

    def __post_init__(self) -> None:
        for name in ("tape_schema", "exchange", "symbol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "num_events", _require_positive_int(self.num_events, "num_events"))
        for name in ("num_l2_batches", "num_trades"):
            object.__setattr__(self, name, _require_nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "start_local_ts_us", _require_positive_int(self.start_local_ts_us, "start_local_ts_us"))
        object.__setattr__(self, "end_local_ts_us", _require_positive_int(self.end_local_ts_us, "end_local_ts_us"))
        if self.end_local_ts_us < self.start_local_ts_us:
            raise ValueError("end_local_ts_us must be >= start_local_ts_us")
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")
        schedule_payload = dict(self.decision_schedule)
        decision_schedule_config_from_dict(schedule_payload)
        object.__setattr__(self, "decision_schedule", schedule_payload)
        object.__setattr__(self, "decision_grid_schema", _require_nonempty_str(self.decision_grid_schema, "decision_grid_schema"))
        object.__setattr__(self, "decision_grid_hash", _require_nonempty_str(self.decision_grid_hash, "decision_grid_hash"))
        if len(self.decision_grid_hash) != 64 or any(ch not in "0123456789abcdef" for ch in self.decision_grid_hash):
            raise ValueError("decision_grid_hash must be lowercase sha256 hex")
        object.__setattr__(self, "decision_grid_n_rows", _require_positive_int(self.decision_grid_n_rows, "decision_grid_n_rows"))
        object.__setattr__(self, "start_event_index", _require_nonnegative_int(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "n_rows", _require_positive_int(self.n_rows, "n_rows"))
        if self.decision_grid_n_rows != self.n_rows:
            raise ValueError("decision_grid_n_rows must equal n_rows")

    def as_dict(self) -> dict[str, object]:
        return {
            "tape_schema": self.tape_schema,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "num_events": self.num_events,
            "num_l2_batches": self.num_l2_batches,
            "num_trades": self.num_trades,
            "start_local_ts_us": self.start_local_ts_us,
            "end_local_ts_us": self.end_local_ts_us,
            "decision_grid_schema": self.decision_grid_schema,
            "decision_grid_hash": self.decision_grid_hash,
            "decision_grid_n_rows": self.decision_grid_n_rows,
            "decision_schedule": dict(self.decision_schedule),
            "start_event_index": self.start_event_index,
            "n_rows": self.n_rows,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "LinearSignalArtifactMetadata":
        if not isinstance(raw, Mapping):
            raise ValueError("metadata must be a mapping")
        required = (
            "tape_schema",
            "exchange",
            "symbol",
            "num_events",
            "num_l2_batches",
            "num_trades",
            "start_local_ts_us",
            "end_local_ts_us",
            "decision_grid_schema",
            "decision_grid_hash",
            "decision_grid_n_rows",
            "decision_schedule",
            "start_event_index",
            "n_rows",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"linear signal metadata missing fields: {missing}")
        return cls(**{key: raw[key] for key in required})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LinearSignalArtifact:
    arrays: LinearSignalArrays
    metadata: LinearSignalArtifactMetadata
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    decision_event_seq: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.arrays, LinearSignalArrays):
            raise ValueError("arrays must be LinearSignalArrays")
        if not isinstance(self.metadata, LinearSignalArtifactMetadata):
            raise ValueError("metadata must be LinearSignalArtifactMetadata")
        decision_event_index = _coerce_int64_vector(self.decision_event_index, name="decision_event_index")
        decision_local_ts_us = _coerce_int64_vector(self.decision_local_ts_us, name="decision_local_ts_us")
        decision_event_seq = _coerce_int64_vector(self.decision_event_seq, name="decision_event_seq")
        if (
            decision_event_index.shape[0] != self.arrays.n_rows
            or decision_local_ts_us.shape[0] != self.arrays.n_rows
            or decision_event_seq.shape[0] != self.arrays.n_rows
        ):
            raise ValueError("decision alignment arrays length must equal signal array rows")
        if self.metadata.n_rows != self.arrays.n_rows:
            raise ValueError("metadata.n_rows must equal signal array rows")
        if (decision_event_index < 0).any():
            raise ValueError("decision_event_index must be nonnegative")
        if (decision_event_seq < 0).any():
            raise ValueError("decision_event_seq must be nonnegative")
        if (decision_local_ts_us <= 0).any():
            raise ValueError("decision_local_ts_us must be positive")
        if decision_event_index.shape[0] > 1 and (np.diff(decision_event_index) <= 0).any():
            raise ValueError("decision_event_index must be strictly increasing")
        if decision_local_ts_us.shape[0] > 1 and (np.diff(decision_local_ts_us) <= 0).any():
            raise ValueError("decision_local_ts_us must be strictly increasing")
        if self.metadata.start_event_index != int(decision_event_index[0]):
            raise ValueError("metadata.start_event_index must equal first decision_event_index")
        object.__setattr__(self, "decision_event_index", decision_event_index)
        object.__setattr__(self, "decision_local_ts_us", decision_local_ts_us)
        object.__setattr__(self, "decision_event_seq", decision_event_seq)

    @property
    def n_rows(self) -> int:
        return self.arrays.n_rows


def save_linear_signal_artifact_npz(
    path: str | Path,
    artifact: LinearSignalArtifact,
    *,
    overwrite: bool = False,
) -> None:
    """Save a validated aligned linear signal artifact to the canonical execution NPZ."""

    if not isinstance(artifact, LinearSignalArtifact):
        raise ValueError("artifact must be LinearSignalArtifact")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    payload = {name: getattr(artifact.arrays, name) for name in _LINEAR_SIGNAL_ARRAY_FIELDS}
    payload["schema"] = np.array(LINEAR_SIGNAL_ARTIFACT_SCHEMA)
    payload["metadata_json"] = np.array(json.dumps(artifact.metadata.as_dict(), sort_keys=True))
    payload["decision_event_index"] = artifact.decision_event_index
    payload["decision_local_ts_us"] = artifact.decision_local_ts_us
    payload["decision_event_seq"] = artifact.decision_event_seq
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)


def linear_signal_array_fields() -> tuple[str, ...]:
    return _LINEAR_SIGNAL_ARRAY_FIELDS


def save_linear_signal_artifact_arrays(
    path: str | Path,
    *,
    metadata: LinearSignalArtifactMetadata,
    decision_event_index: np.ndarray,
    decision_local_ts_us: np.ndarray,
    decision_event_seq: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    overwrite: bool = False,
    validate_chunk_rows: int = 100_000,
) -> None:
    """Save a linear signal NPZ from array-like inputs without artifact copies."""

    if not isinstance(metadata, LinearSignalArtifactMetadata):
        raise ValueError("metadata must be LinearSignalArtifactMetadata")
    if isinstance(validate_chunk_rows, bool) or int(validate_chunk_rows) <= 0:
        raise ValueError("validate_chunk_rows must be a positive int")
    validate_chunk_rows = int(validate_chunk_rows)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))

    decision_arrays = {
        "decision_event_index": np.asarray(decision_event_index),
        "decision_local_ts_us": np.asarray(decision_local_ts_us),
        "decision_event_seq": np.asarray(decision_event_seq),
    }
    for name, arr in decision_arrays.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
        if arr.shape[0] != metadata.n_rows:
            raise ValueError(f"{name} length must equal metadata.n_rows")
        if arr.dtype.kind not in "iu":
            raise ValueError(f"{name} must have integer dtype")

    if not isinstance(arrays, Mapping):
        raise ValueError("arrays must be a mapping")
    missing = [name for name in _LINEAR_SIGNAL_ARRAY_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"missing linear signal arrays: {missing}")
    signal_arrays: dict[str, np.ndarray] = {}
    dtype: np.dtype | None = None
    for name in _LINEAR_SIGNAL_ARRAY_FIELDS:
        arr = np.asarray(arrays[name])
        if arr.ndim != 1 or arr.shape[0] != metadata.n_rows:
            raise ValueError(f"{name} must be rank-1 with length {metadata.n_rows}")
        arr_dtype = np.dtype(arr.dtype)
        if arr_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{name} must have float32 or float64 dtype")
        if dtype is None:
            dtype = arr_dtype
        elif arr_dtype != dtype:
            raise ValueError("all arrays must have the same dtype")
        signal_arrays[name] = arr

    previous_event_index: int | None = None
    previous_local_ts_us: int | None = None
    for start in range(0, metadata.n_rows, validate_chunk_rows):
        end = min(start + validate_chunk_rows, metadata.n_rows)
        event_idx = np.asarray(decision_arrays["decision_event_index"][start:end], dtype=np.int64)
        local_ts = np.asarray(decision_arrays["decision_local_ts_us"][start:end], dtype=np.int64)
        if (event_idx < 0).any():
            raise ValueError("decision_event_index must be nonnegative")
        if (local_ts <= 0).any():
            raise ValueError("decision_local_ts_us must be positive")
        event_seq = np.asarray(decision_arrays["decision_event_seq"][start:end], dtype=np.int64)
        if (event_seq < 0).any():
            raise ValueError("decision_event_seq must be nonnegative")
        if start == 0 and metadata.start_event_index != int(event_idx[0]):
            raise ValueError("metadata.start_event_index must equal first decision_event_index")
        if previous_event_index is not None and int(event_idx[0]) <= previous_event_index:
            raise ValueError("decision_event_index must be strictly increasing")
        if previous_local_ts_us is not None and int(local_ts[0]) <= previous_local_ts_us:
            raise ValueError("decision_local_ts_us must be strictly increasing")
        if event_idx.shape[0] > 1 and (np.diff(event_idx) <= 0).any():
            raise ValueError("decision_event_index must be strictly increasing")
        if local_ts.shape[0] > 1 and (np.diff(local_ts) <= 0).any():
            raise ValueError("decision_local_ts_us must be strictly increasing")
        previous_event_index = int(event_idx[-1])
        previous_local_ts_us = int(local_ts[-1])

        chunk = {name: np.asarray(signal_arrays[name][start:end]) for name in _LINEAR_SIGNAL_ARRAY_FIELDS}
        for name, arr in chunk.items():
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")
        for name in ("p_no_move", "p_move", "p_up_move", "p_down_move"):
            arr = chunk[name]
            if ((arr < 0.0) | (arr > 1.0)).any():
                raise ValueError(f"{name} must be in [0, 1]")
        if not np.allclose(
            chunk["p_no_move"] + chunk["p_move"],
            1.0,
            rtol=0.0,
            atol=_LINEAR_SIGNAL_CONSISTENCY_EPS,
        ):
            raise ValueError("p_no_move + p_move must be approximately 1")
        if not np.allclose(
            chunk["p_up_move"] + chunk["p_down_move"],
            chunk["p_move"],
            rtol=0.0,
            atol=_LINEAR_SIGNAL_CONSISTENCY_EPS,
        ):
            raise ValueError("p_up_move + p_down_move must be approximately p_move")
        if (np.abs(chunk["signed_move_prob"]) > 1.0 + _LINEAR_SIGNAL_CONSISTENCY_EPS).any():
            raise ValueError("signed_move_prob must have abs <= 1 + tolerance")
        for name in ("expected_up_bps", "expected_down_bps", "expected_abs_move_bps", "predicted_vol_bps", "confidence"):
            if (chunk[name] < 0.0).any():
                raise ValueError(f"{name} must be nonnegative")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    payload = {name: signal_arrays[name] for name in _LINEAR_SIGNAL_ARRAY_FIELDS}
    payload["schema"] = np.array(LINEAR_SIGNAL_ARTIFACT_SCHEMA)
    payload["metadata_json"] = np.array(json.dumps(metadata.as_dict(), sort_keys=True))
    payload["decision_event_index"] = decision_arrays["decision_event_index"]
    payload["decision_local_ts_us"] = decision_arrays["decision_local_ts_us"]
    payload["decision_event_seq"] = decision_arrays["decision_event_seq"]
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)


def load_linear_signal_artifact_npz(path: str | Path, *, mmap_mode: str | None = None) -> LinearSignalArtifact:
    """Load a canonical aligned linear signal artifact."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, mmap_mode=mmap_mode, allow_pickle=False) as data:
        keys = set(data.files)
        if "schema" not in keys:
            raise ValueError("linear signal NPZ missing schema")
        schema = str(np.asarray(data["schema"]).item())
        if schema != LINEAR_SIGNAL_ARTIFACT_SCHEMA:
            raise ValueError("linear signal NPZ schema mismatch")
        required = set(_LINEAR_SIGNAL_ARRAY_FIELDS) | {"metadata_json", "decision_event_index", "decision_local_ts_us", "decision_event_seq"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"linear signal NPZ missing required arrays: {missing}")
        metadata_raw = json.loads(str(np.asarray(data["metadata_json"]).item()))
        metadata = LinearSignalArtifactMetadata.from_dict(metadata_raw)
        arrays = LinearSignalArrays(**{name: np.array(data[name], copy=True) for name in _LINEAR_SIGNAL_ARRAY_FIELDS})
        decision_event_index = np.array(data["decision_event_index"], copy=True)
        decision_local_ts_us = np.array(data["decision_local_ts_us"], copy=True)
        decision_event_seq = np.array(data["decision_event_seq"], copy=True)
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=decision_event_index,
        decision_local_ts_us=decision_local_ts_us,
        decision_event_seq=decision_event_seq,
    )


def linear_signal_artifact_summary(artifact: LinearSignalArtifact, *, path: str | None = None) -> dict[str, object]:
    if not isinstance(artifact, LinearSignalArtifact):
        raise ValueError("artifact must be LinearSignalArtifact")
    return {
        "schema": LINEAR_SIGNAL_ARTIFACT_SCHEMA,
        "path": path,
        "n_rows": artifact.n_rows,
        "dtype": str(artifact.arrays.dtype),
        "fields": list(_LINEAR_SIGNAL_ARRAY_FIELDS),
        "metadata": artifact.metadata.as_dict(),
        "first_decision_event_index": int(artifact.decision_event_index[0]),
        "last_decision_event_index": int(artifact.decision_event_index[-1]),
        "first_decision_local_ts_us": int(artifact.decision_local_ts_us[0]),
        "last_decision_local_ts_us": int(artifact.decision_local_ts_us[-1]),
        "first_decision_event_seq": int(artifact.decision_event_seq[0]),
        "last_decision_event_seq": int(artifact.decision_event_seq[-1]),
    }


def validate_linear_signal_artifact_metadata(
    artifact: LinearSignalArtifact,
    *,
    tape_schema: str,
    exchange: str,
    symbol: str,
    num_events: int,
    num_l2_batches: int,
    num_trades: int,
    start_local_ts_us: int,
    end_local_ts_us: int,
    decision_grid_schema: str,
    decision_grid_hash: str,
    decision_grid_n_rows: int,
    decision_schedule: Mapping[str, object],
    start_event_index: int,
    min_rows: int | None = None,
) -> None:
    if not isinstance(artifact, LinearSignalArtifact):
        raise ValueError("linear signal artifact must be LinearSignalArtifact")
    expected = LinearSignalArtifactMetadata(
        tape_schema=tape_schema,
        exchange=exchange,
        symbol=symbol,
        num_events=num_events,
        num_l2_batches=num_l2_batches,
        num_trades=num_trades,
        start_local_ts_us=start_local_ts_us,
        end_local_ts_us=end_local_ts_us,
        decision_grid_schema=decision_grid_schema,
        decision_grid_hash=decision_grid_hash,
        decision_grid_n_rows=decision_grid_n_rows,
        decision_schedule=dict(decision_schedule),
        start_event_index=start_event_index,
        n_rows=artifact.metadata.n_rows,
    )
    actual = artifact.metadata
    for key, expected_value in expected.as_dict().items():
        if key == "n_rows":
            continue
        actual_value = getattr(actual, key)
        if actual_value != expected_value:
            raise ValueError(
                f"linear signal metadata mismatch for {key}: "
                f"expected={expected_value!r} actual={actual_value!r}"
            )
    if actual.n_rows != artifact.n_rows:
        raise ValueError("linear signal metadata n_rows mismatch")
    if min_rows is not None:
        min_rows = _require_nonnegative_int(min_rows, "min_rows")
        if artifact.n_rows < min_rows:
            raise ValueError("linear signal artifact does not contain enough rows")


def validate_linear_signals_for_decision_grid(
    artifact: LinearSignalArtifact,
    decision_grid: DecisionGrid,
) -> None:
    if not isinstance(artifact, LinearSignalArtifact):
        raise ValueError("artifact must be LinearSignalArtifact")
    if not isinstance(decision_grid, DecisionGrid):
        raise ValueError("decision_grid must be DecisionGrid")
    if artifact.metadata.decision_grid_schema != decision_grid.metadata.schema:
        raise ValueError("linear signal decision_grid_schema mismatch")
    if artifact.metadata.decision_grid_hash != decision_grid.decision_grid_hash:
        raise ValueError("linear signal decision_grid_hash mismatch")
    if artifact.metadata.decision_grid_n_rows != decision_grid.n_rows or artifact.n_rows != decision_grid.n_rows:
        raise ValueError("linear signal row count must equal decision grid rows")
    if not np.array_equal(artifact.decision_event_index, decision_grid.decision_event_index):
        raise ValueError("linear signal decision_event_index does not match decision grid")
    if not np.array_equal(artifact.decision_local_ts_us, decision_grid.decision_local_ts_us):
        raise ValueError("linear signal decision_local_ts_us does not match decision grid")
    if not np.array_equal(artifact.decision_event_seq, decision_grid.decision_event_seq):
        raise ValueError("linear signal decision_event_seq does not match decision grid")


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



def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _coerce_int64_vector(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have integer dtype")
    return np.ascontiguousarray(arr, dtype=np.int64)

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


def _convert_magnitude_array(values: np.ndarray, *, config: LinearSignalConfig, dtype: np.dtype, name: str) -> np.ndarray:
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
    "MAGNITUDE_INPUT_LOG1P_BPS",
    "MAGNITUDE_INPUT_BPS",
    "MAGNITUDE_INPUT_MODES",
    "DIRECTION_PROBA_KEY",
    "NO_MOVE_PROBA_KEY",
    "MAGNITUDE_UP_KEY",
    "MAGNITUDE_DOWN_KEY",
    "LINEAR_SIGNAL_ARTIFACT_SCHEMA",
    "LINEAR_SIGNALS_FILENAME",
    "LinearSignalConfig",
    "LinearSignalArrays",
    "LinearSignalArtifactMetadata",
    "LinearSignalArtifact",
    "magnitude_to_bps",
    "build_gated_linear_signal",
    "prediction_row_to_signal",
    "predictions_to_signal_arrays",
    "linear_signal_at",
    "save_linear_signal_artifact_npz",
    "linear_signal_array_fields",
    "save_linear_signal_artifact_arrays",
    "load_linear_signal_artifact_npz",
    "linear_signal_artifact_summary",
    "validate_linear_signal_artifact_metadata",
    "validate_linear_signals_for_decision_grid",
]
