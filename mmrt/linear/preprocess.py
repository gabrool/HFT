"""Train-only feature preprocessing for storage-backed MMRT linear models.

This module fits feature mean/variance statistics from training feature
matrices and applies a frozen z-score transform for linear models. It does
not read storage, compute features or supervised outcomes, inspect row timing
fields, build splits, train models, or evaluate metrics.

The preprocessor is intentionally shape-preserving: inactive near-constant
features are retained and transformed to zero rather than dropped.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DEFAULT_PREPROCESS_DTYPE = "float32"
ALLOWED_PREPROCESS_DTYPES = ("float32", "float64")
DEFAULT_VARIANCE_FLOOR = 1e-12
DEFAULT_CLIP_Z = 8.0


def _require_positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return out


def _require_output_dtype(value: str) -> str:
    if value not in ALLOWED_PREPROCESS_DTYPES:
        raise ValueError("output_dtype must be one of allowed preprocess dtypes")
    return value


def _coerce_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    cols = tuple(feature_columns)
    if not cols:
        raise ValueError("feature_columns must be non-empty")
    for col in cols:
        if not isinstance(col, str) or col == "":
            raise ValueError("feature_columns must contain non-empty strings")
    if len(set(cols)) != len(cols):
        raise ValueError("feature_columns must be unique")
    return cols


def _coerce_matrix(X: np.ndarray, *, dtype: np.dtype, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    arr = np.array(arr, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _coerce_vector(values: np.ndarray, *, dtype: np.dtype, n_features: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.shape != (n_features,):
        raise ValueError(f"{name} must have shape ({n_features},)")
    arr = np.array(arr, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _coerce_bool_mask(values: np.ndarray, *, n_features: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have bool dtype")
    if arr.shape != (n_features,):
        raise ValueError(f"{name} must have shape ({n_features},)")
    return np.array(arr, dtype=bool, order="C", copy=True)


@dataclass(frozen=True, slots=True)
class LinearPreprocessConfig:
    variance_floor: float = DEFAULT_VARIANCE_FLOOR
    clip_z: float = DEFAULT_CLIP_Z
    output_dtype: str = DEFAULT_PREPROCESS_DTYPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "variance_floor", _require_positive_float(self.variance_floor, "variance_floor"))
        object.__setattr__(self, "clip_z", _require_positive_float(self.clip_z, "clip_z"))
        object.__setattr__(self, "output_dtype", _require_output_dtype(self.output_dtype))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.output_dtype)


@dataclass(slots=True)
class RunningFeatureStats:
    n_rows: int
    n_features: int
    mean: np.ndarray
    m2: np.ndarray
    _centered_scratch: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_rows, int) or self.n_rows < 0:
            raise ValueError("n_rows must be a nonnegative int")
        if not isinstance(self.n_features, int) or self.n_features <= 0:
            raise ValueError("n_features must be a positive int")
        self.mean = _coerce_vector(self.mean, dtype=np.dtype("float64"), n_features=self.n_features, name="mean")
        self.m2 = _coerce_vector(self.m2, dtype=np.dtype("float64"), n_features=self.n_features, name="m2")
        if np.any(self.m2 < 0.0):
            raise ValueError("m2 must be nonnegative")

    @classmethod
    def empty(cls, n_features: int, *, dtype: np.dtype = np.dtype("float64")) -> "RunningFeatureStats":
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be a positive int")
        dd = np.dtype(dtype)
        return cls(
            n_rows=0,
            n_features=n_features,
            mean=np.zeros(n_features, dtype=dd, order="C"),
            m2=np.zeros(n_features, dtype=dd, order="C"),
        )

    def update(self, X: np.ndarray) -> None:
        Xc = _coerce_matrix(X, dtype=np.dtype("float64"), name="X")
        if Xc.shape[1] != self.n_features:
            raise ValueError("X feature count does not match n_features")
        if Xc.shape[0] == 0:
            return
        batch_n = Xc.shape[0]
        batch_mean = Xc.mean(axis=0)
        if self._centered_scratch is None or self._centered_scratch.shape != Xc.shape:
            self._centered_scratch = np.empty_like(Xc, dtype=np.float64)
        np.subtract(Xc, batch_mean, out=self._centered_scratch)
        batch_m2 = np.einsum("ij,ij->j", self._centered_scratch, self._centered_scratch, optimize=True)
        total_n = self.n_rows + batch_n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_n / total_n)
        new_m2 = self.m2 + batch_m2 + delta * delta * (self.n_rows * batch_n / total_n)
        self.n_rows = total_n
        self.mean[...] = new_mean
        self.m2[...] = np.maximum(new_m2, 0.0)

    def variance(self) -> np.ndarray:
        if self.n_rows <= 1:
            return np.zeros(self.n_features, dtype=np.dtype("float64"), order="C")
        return np.array(self.m2 / (self.n_rows - 1), dtype=np.dtype("float64"), order="C", copy=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": int(self.n_rows),
            "n_features": int(self.n_features),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "RunningFeatureStats":
        return cls(
            n_rows=int(d["n_rows"]),
            n_features=int(d["n_features"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            m2=np.asarray(d["m2"], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class LinearPreprocessState:
    feature_columns: tuple[str, ...]
    n_rows_fit: int
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    active_mask: np.ndarray
    config: LinearPreprocessConfig

    def __post_init__(self) -> None:
        cols = _coerce_feature_columns(self.feature_columns)
        object.__setattr__(self, "feature_columns", cols)
        if not isinstance(self.n_rows_fit, int) or self.n_rows_fit <= 0:
            raise ValueError("n_rows_fit must be a positive int")
        if not isinstance(self.config, LinearPreprocessConfig):
            raise ValueError("config must be LinearPreprocessConfig")
        n_features = len(cols)
        mean = _coerce_vector(self.mean, dtype=np.dtype("float64"), n_features=n_features, name="mean")
        variance = _coerce_vector(self.variance, dtype=np.dtype("float64"), n_features=n_features, name="variance")
        scale = _coerce_vector(self.scale, dtype=np.dtype("float64"), n_features=n_features, name="scale")
        active_mask = _coerce_bool_mask(self.active_mask, n_features=n_features, name="active_mask")
        if np.any(variance < 0.0):
            raise ValueError("variance must be nonnegative")
        if np.any(scale <= 0.0):
            raise ValueError("scale must be positive")
        expected_active = variance > self.config.variance_floor
        if not np.array_equal(active_mask, expected_active):
            raise ValueError("active_mask must match variance > variance_floor")
        expected_scale = np.sqrt(np.maximum(variance, self.config.variance_floor))
        if not np.allclose(scale, expected_scale, rtol=1e-12, atol=1e-15):
            raise ValueError("scale must match sqrt(max(variance, variance_floor))")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "active_mask", active_mask)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def active_count(self) -> int:
        return int(np.sum(self.active_mask))

    @property
    def dtype(self) -> np.dtype:
        return self.config.dtype

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_columns": list(self.feature_columns),
            "n_rows_fit": int(self.n_rows_fit),
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
            "scale": self.scale.tolist(),
            "active_mask": self.active_mask.tolist(),
            "config": {
                "variance_floor": self.config.variance_floor,
                "clip_z": self.config.clip_z,
                "output_dtype": self.config.output_dtype,
            },
            "method": "train_only_zscore_shape_preserving",
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "LinearPreprocessState":
        config_dict = d["config"]
        config = LinearPreprocessConfig(
            variance_floor=float(config_dict["variance_floor"]),
            clip_z=float(config_dict["clip_z"]),
            output_dtype=str(config_dict["output_dtype"]),
        )
        return cls(
            feature_columns=tuple(d["feature_columns"]),
            n_rows_fit=int(d["n_rows_fit"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            variance=np.asarray(d["variance"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            active_mask=np.asarray(d["active_mask"], dtype=bool),
            config=config,
        )


class LinearPreprocessor:
    def __init__(self, config: LinearPreprocessConfig | None = None):
        self.config = config if config is not None else LinearPreprocessConfig()
        self._feature_columns: tuple[str, ...] | None = None
        self._stats: RunningFeatureStats | None = None
        self.state: LinearPreprocessState | None = None

    def is_fitted(self) -> bool:
        return self.state is not None

    def partial_fit(self, X: np.ndarray, *, feature_columns: Sequence[str]) -> "LinearPreprocessor":
        if self.state is not None:
            raise ValueError("cannot partial_fit after finalize")
        cols = _coerce_feature_columns(feature_columns)
        Xc = _coerce_matrix(X, dtype=np.dtype("float64"), name="X")
        if Xc.shape[1] != len(cols):
            raise ValueError("X feature count must match feature_columns")
        if self._feature_columns is None:
            self._feature_columns = cols
            self._stats = RunningFeatureStats.empty(len(cols))
        elif cols != self._feature_columns:
            raise ValueError("feature_columns must match prior partial_fit calls")
        self._stats.update(Xc)
        return self

    def finalize(self) -> LinearPreprocessState:
        if self._stats is None or self._feature_columns is None:
            raise ValueError("cannot finalize without partial_fit")
        if self._stats.n_rows <= 0:
            raise ValueError("cannot finalize with zero rows")
        variance = self._stats.variance()
        active_mask = variance > self.config.variance_floor
        scale = np.sqrt(np.maximum(variance, self.config.variance_floor))
        state = LinearPreprocessState(
            feature_columns=self._feature_columns,
            n_rows_fit=self._stats.n_rows,
            mean=self._stats.mean,
            variance=variance,
            scale=scale,
            active_mask=active_mask,
            config=self.config,
        )
        self.state = state
        return state

    def fit(self, X: np.ndarray, *, feature_columns: Sequence[str]) -> LinearPreprocessState:
        if self._stats is not None or self.state is not None:
            raise ValueError("fit requires fresh preprocessor")
        self.partial_fit(X, feature_columns=feature_columns)
        return self.finalize()

    def transform(self, X: np.ndarray, *, feature_columns: Sequence[str] | None = None) -> np.ndarray:
        if self.state is None:
            raise ValueError("preprocessor is not fitted")
        state = self.state
        if feature_columns is not None:
            cols = _coerce_feature_columns(feature_columns)
            if cols != state.feature_columns:
                raise ValueError("feature_columns must match fitted state")
        Xc = _coerce_matrix(X, dtype=np.dtype("float64"), name="X")
        if Xc.shape[1] != state.n_features:
            raise ValueError("X feature count does not match fitted state")
        Z = (Xc - state.mean) / state.scale
        Z[:, ~state.active_mask] = 0.0
        Z = np.clip(Z, -self.config.clip_z, self.config.clip_z)
        return np.array(Z, dtype=self.config.dtype, order="C", copy=True)

    def load_state(self, state: LinearPreprocessState) -> "LinearPreprocessor":
        if not isinstance(state, LinearPreprocessState):
            raise ValueError("state must be LinearPreprocessState")
        self.config = state.config
        self._feature_columns = state.feature_columns
        self._stats = None
        self.state = state
        return self

    def as_dict(self) -> dict[str, object]:
        if self.state is None:
            raise ValueError("preprocessor is not fitted")
        return self.state.as_dict()

    @classmethod
    def from_state(cls, state: LinearPreprocessState) -> "LinearPreprocessor":
        pre = cls(config=state.config)
        return pre.load_state(state)


def fit_preprocessor(
    batches: Sequence[np.ndarray],
    *,
    feature_columns: Sequence[str],
    config: LinearPreprocessConfig | None = None,
) -> LinearPreprocessState:
    pre = LinearPreprocessor(config=config)
    for batch in batches:
        pre.partial_fit(batch, feature_columns=feature_columns)
    return pre.finalize()


def transform_with_state(
    X: np.ndarray,
    state: LinearPreprocessState,
    *,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    return LinearPreprocessor.from_state(state).transform(X, feature_columns=feature_columns)


__all__ = [
    "DEFAULT_PREPROCESS_DTYPE",
    "ALLOWED_PREPROCESS_DTYPES",
    "DEFAULT_VARIANCE_FLOOR",
    "DEFAULT_CLIP_Z",
    "LinearPreprocessConfig",
    "RunningFeatureStats",
    "LinearPreprocessState",
    "LinearPreprocessor",
    "fit_preprocessor",
    "transform_with_state",
]
