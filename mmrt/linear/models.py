"""Linear model heads for storage-backed MMRT training.

This module implements NumPy-only linear heads for direction and magnitude
prediction. Magnitude heads use Huber loss on log1p side-specific magnitude
targets. It consumes already-preprocessed feature matrices and explicit target
arrays. It does not read storage, build features, construct targets,
fit preprocessing, inspect row timing fields, evaluate metrics, or run
training orchestration.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

LINEAR_MODEL_SCHEMA = "mmrt_linear_model"
DEFAULT_MODEL_DTYPE = "float32"
ALLOWED_MODEL_DTYPES = ("float32", "float64")

DEFAULT_LEARNING_RATE = 0.05
DEFAULT_L2 = 1e-4
DEFAULT_MAX_GRAD_NORM = 10.0
DEFAULT_INIT_SCALE = 0.0
DEFAULT_MAGNITUDE_HUBER_DELTA = 1.0

NO_MOVE_HEAD = "no_move"
DIRECTION_HEAD = "direction"
MAGNITUDE_UP_HEAD = "magnitude_up"
MAGNITUDE_DOWN_HEAD = "magnitude_down"
MODEL_HEADS = (NO_MOVE_HEAD, DIRECTION_HEAD, MAGNITUDE_UP_HEAD, MAGNITUDE_DOWN_HEAD)


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite float")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite float")
    return coerced


def _require_positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return coerced


def _require_probability_threshold(value: float, name: str = "threshold") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    out = float(value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return out


def _require_output_dtype(value: str) -> str:
    if value not in ALLOWED_MODEL_DTYPES:
        raise ValueError(f"output_dtype must be one of {ALLOWED_MODEL_DTYPES}")
    return value


def _coerce_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    cols = tuple(feature_columns)
    if not cols:
        raise ValueError("feature_columns must be non-empty")
    seen: set[str] = set()
    for col in cols:
        if not isinstance(col, str) or col == "":
            raise ValueError("feature_columns entries must be non-empty strings")
        if col in seen:
            raise ValueError("feature_columns must not contain duplicates")
        seen.add(col)
    return cols


def _coerce_matrix(X: np.ndarray, *, n_features: int | None = None, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(f"{name} must have {n_features} columns")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite values")
    return out


def _coerce_binary_classes(y: np.ndarray, *, n_rows: int) -> np.ndarray:
    arr = np.asarray(y)
    if arr.shape != (n_rows,):
        raise ValueError("y_direction must have shape (n_rows,)")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("y_direction must contain finite values")
    if not np.logical_or(out == 0.0, out == 1.0).all():
        raise ValueError("y_direction must contain only binary classes 0/1")
    return out


def _coerce_regression_target(y: np.ndarray, *, n_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(y)
    if arr.shape != (n_rows,):
        raise ValueError(f"{name} must have shape (n_rows,)")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite values")
    if (out < 0.0).any():
        raise ValueError(f"{name} must be nonnegative")
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _clip_gradient(vec: np.ndarray, max_norm: float) -> np.ndarray:
    grad = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(grad))
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad.copy()


def _huber_loss_and_grad(residual: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    r = np.asarray(residual, dtype=np.float64)
    d = _require_positive_float(delta, "magnitude_huber_delta")
    abs_r = np.abs(r)
    quadratic = abs_r <= d
    loss = np.where(quadratic, 0.5 * r * r, d * (abs_r - 0.5 * d))
    grad = np.where(quadratic, r, d * np.sign(r))
    return float(np.mean(loss)) if r.size else 0.0, np.ascontiguousarray(grad, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LinearModelConfig:
    learning_rate: float = DEFAULT_LEARNING_RATE
    l2: float = DEFAULT_L2
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    output_dtype: str = DEFAULT_MODEL_DTYPE
    magnitude_huber_delta: float = DEFAULT_MAGNITUDE_HUBER_DELTA

    def __post_init__(self) -> None:
        object.__setattr__(self, "learning_rate", _require_positive_float(self.learning_rate, "learning_rate"))
        object.__setattr__(self, "l2", _require_nonnegative_float(self.l2, "l2"))
        object.__setattr__(self, "max_grad_norm", _require_positive_float(self.max_grad_norm, "max_grad_norm"))
        object.__setattr__(self, "output_dtype", _require_output_dtype(self.output_dtype))
        object.__setattr__(
            self,
            "magnitude_huber_delta",
            _require_positive_float(self.magnitude_huber_delta, "magnitude_huber_delta"),
        )

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.output_dtype)


@dataclass(frozen=True, slots=True)
class LinearHeadState:
    head_name: str
    feature_columns: tuple[str, ...]
    weights: np.ndarray
    intercept: float
    n_updates: int
    n_rows_seen: int
    config: LinearModelConfig

    def __post_init__(self) -> None:
        if self.head_name not in MODEL_HEADS:
            raise ValueError("invalid head_name")
        cols = _coerce_feature_columns(self.feature_columns)
        object.__setattr__(self, "feature_columns", cols)
        w = np.ascontiguousarray(np.asarray(self.weights, dtype=np.float64), dtype=np.float64)
        if w.shape != (len(cols),):
            raise ValueError("weights must have shape (n_features,)")
        if not np.isfinite(w).all():
            raise ValueError("weights must be finite")
        object.__setattr__(self, "weights", w.copy())
        b = float(self.intercept)
        if not np.isfinite(b):
            raise ValueError("intercept must be finite")
        object.__setattr__(self, "intercept", b)
        object.__setattr__(self, "n_updates", _require_nonnegative_int(self.n_updates, "n_updates"))
        object.__setattr__(self, "n_rows_seen", _require_nonnegative_int(self.n_rows_seen, "n_rows_seen"))
        if not isinstance(self.config, LinearModelConfig):
            raise ValueError("config must be a LinearModelConfig")

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def as_dict(self) -> dict[str, object]:
        return {
            "head_name": self.head_name,
            "feature_columns": list(self.feature_columns),
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "n_updates": self.n_updates,
            "n_rows_seen": self.n_rows_seen,
            "config": {
                "learning_rate": self.config.learning_rate,
                "l2": self.config.l2,
                "max_grad_norm": self.config.max_grad_norm,
                "output_dtype": self.config.output_dtype,
                "magnitude_huber_delta": self.config.magnitude_huber_delta,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "LinearHeadState":
        cfg_raw = d["config"]
        if not isinstance(cfg_raw, dict):
            raise ValueError("config must be a mapping")
        cfg = LinearModelConfig(
            learning_rate=cfg_raw["learning_rate"],
            l2=cfg_raw["l2"],
            max_grad_norm=cfg_raw["max_grad_norm"],
            output_dtype=cfg_raw["output_dtype"],
            magnitude_huber_delta=cfg_raw["magnitude_huber_delta"],
        )
        return cls(
            head_name=d["head_name"],
            feature_columns=tuple(d["feature_columns"]),
            weights=np.asarray(d["weights"], dtype=np.float64),
            intercept=d["intercept"],
            n_updates=d["n_updates"],
            n_rows_seen=d["n_rows_seen"],
            config=cfg,
        )


class BaseLinearHead:
    def __init__(self, head_name: str, feature_columns: Sequence[str], config: LinearModelConfig | None = None):
        if head_name not in MODEL_HEADS:
            raise ValueError("invalid head_name")
        self.head_name = head_name
        self.feature_columns = _coerce_feature_columns(feature_columns)
        self.config = config if config is not None else LinearModelConfig()
        n_features = len(self.feature_columns)
        self.weights = np.zeros(n_features, dtype=np.float64)
        if DEFAULT_INIT_SCALE != 0.0:
            self.weights += DEFAULT_INIT_SCALE
        self.intercept = 0.0
        self.n_updates = 0
        self.n_rows_seen = 0

    def is_fitted(self) -> bool:
        return self.n_updates > 0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        scores = np.ascontiguousarray(Xc @ self.weights + self.intercept, dtype=self.config.dtype)
        if not np.isfinite(scores).all():
            raise ValueError("decision function produced non-finite values")
        return scores

    def _apply_gradient(self, grad_w: np.ndarray, grad_b: float, n_rows: int) -> None:
        _require_positive_int(n_rows, "n_rows")
        gw = np.ascontiguousarray(np.asarray(grad_w, dtype=np.float64), dtype=np.float64)
        if gw.shape != self.weights.shape or not np.isfinite(gw).all():
            raise ValueError("grad_w must be finite and match weights shape")
        gb = float(grad_b)
        if not np.isfinite(gb):
            raise ValueError("grad_b must be finite")
        total_w = gw + self.config.l2 * self.weights
        full_grad = np.concatenate([total_w, np.array([gb], dtype=np.float64)])
        clipped = _clip_gradient(full_grad, self.config.max_grad_norm)
        self.weights -= self.config.learning_rate * clipped[:-1]
        self.intercept -= self.config.learning_rate * float(clipped[-1])
        self.n_updates += 1
        self.n_rows_seen += n_rows

    def state(self) -> LinearHeadState:
        return LinearHeadState(
            head_name=self.head_name,
            feature_columns=self.feature_columns,
            weights=self.weights.copy(),
            intercept=self.intercept,
            n_updates=self.n_updates,
            n_rows_seen=self.n_rows_seen,
            config=self.config,
        )

    def load_state(self, state: LinearHeadState) -> "BaseLinearHead":
        if state.head_name != self.head_name:
            raise ValueError("state head_name does not match")
        if state.feature_columns != self.feature_columns:
            raise ValueError("state feature_columns do not match")
        self.weights = state.weights.copy()
        self.intercept = float(state.intercept)
        self.n_updates = int(state.n_updates)
        self.n_rows_seen = int(state.n_rows_seen)
        self.config = state.config
        return self

    def as_dict(self) -> dict[str, object]:
        return self.state().as_dict()


class DirectionLinearHead(BaseLinearHead):
    def __init__(self, feature_columns: Sequence[str], config: LinearModelConfig | None = None):
        super().__init__(DIRECTION_HEAD, feature_columns, config)

    def partial_fit(self, X: np.ndarray, y_direction: np.ndarray) -> "DirectionLinearHead":
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        y = _coerce_binary_classes(y_direction, n_rows=Xc.shape[0])
        n_rows = Xc.shape[0]
        if n_rows == 0:
            return self
        logits = Xc @ self.weights + self.intercept
        p = _sigmoid(logits)
        err = p - y
        grad_w = (Xc.T @ err) / n_rows
        grad_b = float(np.mean(err))
        self._apply_gradient(grad_w, grad_b, n_rows)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        scores = Xc @ self.weights + self.intercept
        p_up = _sigmoid(scores)
        proba = np.column_stack([1.0 - p_up, p_up]).astype(self.config.dtype, copy=False)
        proba = np.ascontiguousarray(proba)
        if not np.isfinite(proba).all():
            raise ValueError("predict_proba produced non-finite values")
        return proba

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        thr = _require_probability_threshold(threshold)
        p_up = self.predict_proba(X)[:, 1]
        return (p_up >= thr).astype(np.int8, copy=False)

    def loss(self, X: np.ndarray, y_direction: np.ndarray) -> float:
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        y = _coerce_binary_classes(y_direction, n_rows=Xc.shape[0])
        if Xc.shape[0] == 0:
            return 0.5 * self.config.l2 * float(np.dot(self.weights, self.weights))
        p = _sigmoid(Xc @ self.weights + self.intercept)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        ce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = 0.5 * self.config.l2 * float(np.dot(self.weights, self.weights))
        return float(ce + reg)


class NoMoveLinearHead(BaseLinearHead):
    def __init__(self, feature_columns: Sequence[str], config: LinearModelConfig | None = None):
        super().__init__(NO_MOVE_HEAD, feature_columns, config)

    def partial_fit(self, X: np.ndarray, y_no_move: np.ndarray) -> "NoMoveLinearHead":
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        y = _coerce_binary_classes(y_no_move, n_rows=Xc.shape[0])
        n_rows = Xc.shape[0]
        if n_rows == 0:
            return self
        logits = Xc @ self.weights + self.intercept
        p = _sigmoid(logits)
        err = p - y
        self._apply_gradient((Xc.T @ err) / n_rows, float(np.mean(err)), n_rows)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(_coerce_matrix(X, n_features=len(self.feature_columns)) @ self.weights + self.intercept)
        proba = np.ascontiguousarray(np.column_stack([1.0 - p, p]).astype(self.config.dtype, copy=False))
        if not np.isfinite(proba).all():
            raise ValueError("predict_proba produced non-finite values")
        return proba

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        thr = _require_probability_threshold(threshold)
        return (self.predict_proba(X)[:, 1] >= thr).astype(np.int8, copy=False)

    def loss(self, X: np.ndarray, y_no_move: np.ndarray) -> float:
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        y = _coerce_binary_classes(y_no_move, n_rows=Xc.shape[0])
        if Xc.shape[0] == 0:
            return 0.5 * self.config.l2 * float(np.dot(self.weights, self.weights))
        p = _sigmoid(Xc @ self.weights + self.intercept)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        ce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = 0.5 * self.config.l2 * float(np.dot(self.weights, self.weights))
        return float(ce + reg)


class MagnitudeLinearHead(BaseLinearHead):
    def __init__(self, head_name: str, feature_columns: Sequence[str], config: LinearModelConfig | None = None):
        if head_name not in (MAGNITUDE_UP_HEAD, MAGNITUDE_DOWN_HEAD):
            raise ValueError("head_name must be a magnitude head")
        super().__init__(head_name, feature_columns, config)

    def partial_fit(self, X: np.ndarray, y_magnitude: np.ndarray) -> "MagnitudeLinearHead":
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        y = _coerce_regression_target(y_magnitude, n_rows=Xc.shape[0], name="y_magnitude")
        n_rows = Xc.shape[0]
        if n_rows == 0:
            return self
        pred = Xc @ self.weights + self.intercept
        residual = pred - y
        _, grad_pred = _huber_loss_and_grad(residual, self.config.magnitude_huber_delta)
        grad_w = (Xc.T @ grad_pred) / n_rows
        grad_b = float(np.mean(grad_pred))
        self._apply_gradient(grad_w, grad_b, n_rows)
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X)

    def predict_nonnegative(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        clipped = np.maximum(raw, 0.0)
        return np.ascontiguousarray(clipped.astype(self.config.dtype, copy=False))

    def loss(self, X: np.ndarray, y_magnitude: np.ndarray) -> float:
        Xc = _coerce_matrix(X, n_features=len(self.feature_columns))
        y = _coerce_regression_target(y_magnitude, n_rows=Xc.shape[0], name="y_magnitude")
        if Xc.shape[0] == 0:
            return 0.5 * self.config.l2 * float(np.dot(self.weights, self.weights))
        pred = Xc @ self.weights + self.intercept
        residual = pred - y
        data_loss, _ = _huber_loss_and_grad(residual, self.config.magnitude_huber_delta)
        reg = 0.5 * self.config.l2 * float(np.dot(self.weights, self.weights))
        return float(data_loss + reg)


@dataclass(slots=True)
class LinearModelBundle:
    no_move: NoMoveLinearHead
    direction: DirectionLinearHead
    magnitude_up: MagnitudeLinearHead
    magnitude_down: MagnitudeLinearHead

    def __post_init__(self) -> None:
        if not isinstance(self.no_move, NoMoveLinearHead):
            raise ValueError("no_move must be NoMoveLinearHead")
        if not isinstance(self.direction, DirectionLinearHead):
            raise ValueError("direction must be DirectionLinearHead")
        if not isinstance(self.magnitude_up, MagnitudeLinearHead):
            raise ValueError("magnitude_up must be MagnitudeLinearHead")
        if not isinstance(self.magnitude_down, MagnitudeLinearHead):
            raise ValueError("magnitude_down must be MagnitudeLinearHead")
        if self.magnitude_up.head_name != MAGNITUDE_UP_HEAD:
            raise ValueError("magnitude_up has wrong head_name")
        if self.magnitude_down.head_name != MAGNITUDE_DOWN_HEAD:
            raise ValueError("magnitude_down has wrong head_name")
        if self.no_move.config != self.direction.config or self.direction.config != self.magnitude_up.config or self.direction.config != self.magnitude_down.config:
            raise ValueError("all heads must share identical config")

    @property
    def feature_columns_by_head(self) -> dict[str, tuple[str, ...]]:
        return {
            NO_MOVE_HEAD: self.no_move.feature_columns,
            DIRECTION_HEAD: self.direction.feature_columns,
            MAGNITUDE_UP_HEAD: self.magnitude_up.feature_columns,
            MAGNITUDE_DOWN_HEAD: self.magnitude_down.feature_columns,
        }

    @property
    def feature_counts_by_head(self) -> dict[str, int]:
        return {
            head: len(cols)
            for head, cols in self.feature_columns_by_head.items()
        }

    def heads_share_feature_columns(self) -> bool:
        return (
            self.no_move.feature_columns
            == self.direction.feature_columns
            == self.magnitude_up.feature_columns
            == self.magnitude_down.feature_columns
        )

    @property
    def n_features(self) -> int:
        if not self.heads_share_feature_columns():
            raise ValueError("n_features is only defined when all heads share feature_columns")
        return len(self.direction.feature_columns)

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self.heads_share_feature_columns():
            raise ValueError(
                "LinearModelBundle.predict(X) requires all heads to share identical feature_columns; "
                "use per-head prediction paths for per-head feature sets"
            )
        return {
            "direction_proba": self.direction.predict_proba(X),
            "direction_pred": self.direction.predict(X),
            "no_move_proba": self.no_move.predict_proba(X),
            "no_move_pred": self.no_move.predict(X),
            "magnitude_up": self.magnitude_up.predict_nonnegative(X),
            "magnitude_down": self.magnitude_down.predict_nonnegative(X),
        }

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": LINEAR_MODEL_SCHEMA,
            "bundle_type": "linear_four_head_gated",
            "feature_columns_by_head": {
                head: list(cols)
                for head, cols in self.feature_columns_by_head.items()
            },
            "feature_counts_by_head": self.feature_counts_by_head,
            "no_move": self.no_move.as_dict(),
            "direction": self.direction.as_dict(),
            "magnitude_up": self.magnitude_up.as_dict(),
            "magnitude_down": self.magnitude_down.as_dict(),
        }
        if self.heads_share_feature_columns():
            payload["shared_feature_columns"] = list(self.direction.feature_columns)
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "LinearModelBundle":
        if d.get("bundle_type") != "linear_four_head_gated":
            raise ValueError("unsupported bundle_type")
        no_move = load_linear_head_state(LinearHeadState.from_dict(d["no_move"]))
        direction = load_linear_head_state(LinearHeadState.from_dict(d["direction"]))
        magnitude_up = load_linear_head_state(LinearHeadState.from_dict(d["magnitude_up"]))
        magnitude_down = load_linear_head_state(LinearHeadState.from_dict(d["magnitude_down"]))
        if not isinstance(no_move, NoMoveLinearHead):
            raise ValueError("no_move state is invalid")
        if not isinstance(direction, DirectionLinearHead):
            raise ValueError("direction state is invalid")
        if not isinstance(magnitude_up, MagnitudeLinearHead) or not isinstance(magnitude_down, MagnitudeLinearHead):
            raise ValueError("magnitude states are invalid")
        return cls(no_move=no_move, direction=direction, magnitude_up=magnitude_up, magnitude_down=magnitude_down)


def make_linear_model_bundle(
    feature_columns_by_head: Mapping[str, Sequence[str]] | Sequence[str],
    config: LinearModelConfig | None = None,
) -> LinearModelBundle:
    cfg = config if config is not None else LinearModelConfig()
    if isinstance(feature_columns_by_head, Mapping):
        if set(feature_columns_by_head.keys()) != set(MODEL_HEADS):
            raise ValueError("feature_columns_by_head keys must exactly match MODEL_HEADS")
        no_move_cols = tuple(feature_columns_by_head[NO_MOVE_HEAD])
        direction_cols = tuple(feature_columns_by_head[DIRECTION_HEAD])
        up_cols = tuple(feature_columns_by_head[MAGNITUDE_UP_HEAD])
        down_cols = tuple(feature_columns_by_head[MAGNITUDE_DOWN_HEAD])
    else:
        no_move_cols = tuple(feature_columns_by_head)
        direction_cols = tuple(feature_columns_by_head)
        up_cols = direction_cols
        down_cols = direction_cols
    return LinearModelBundle(
        no_move=NoMoveLinearHead(no_move_cols, cfg),
        direction=DirectionLinearHead(direction_cols, cfg),
        magnitude_up=MagnitudeLinearHead(MAGNITUDE_UP_HEAD, up_cols, cfg),
        magnitude_down=MagnitudeLinearHead(MAGNITUDE_DOWN_HEAD, down_cols, cfg),
    )


def load_linear_head_state(state: LinearHeadState) -> BaseLinearHead:
    if state.head_name == NO_MOVE_HEAD:
        head: BaseLinearHead = NoMoveLinearHead(state.feature_columns, state.config)
    elif state.head_name == DIRECTION_HEAD:
        head: BaseLinearHead = DirectionLinearHead(state.feature_columns, state.config)
    elif state.head_name in (MAGNITUDE_UP_HEAD, MAGNITUDE_DOWN_HEAD):
        head = MagnitudeLinearHead(state.head_name, state.feature_columns, state.config)
    else:
        raise ValueError("unsupported head_name")
    head.load_state(state)
    return head


def load_linear_model_bundle(d: dict[str, object]) -> LinearModelBundle:
    return LinearModelBundle.from_dict(d)


__all__ = [
    "LINEAR_MODEL_SCHEMA",
    "DEFAULT_MODEL_DTYPE",
    "ALLOWED_MODEL_DTYPES",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_L2",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_INIT_SCALE",
    "DEFAULT_MAGNITUDE_HUBER_DELTA",
    "NO_MOVE_HEAD",
    "DIRECTION_HEAD",
    "MAGNITUDE_UP_HEAD",
    "MAGNITUDE_DOWN_HEAD",
    "MODEL_HEADS",
    "LinearModelConfig",
    "LinearHeadState",
    "BaseLinearHead",
    "NoMoveLinearHead",
    "DirectionLinearHead",
    "MagnitudeLinearHead",
    "LinearModelBundle",
    "make_linear_model_bundle",
    "load_linear_head_state",
    "load_linear_model_bundle",
]
