"""Causal feature transforms for the MMRT feature pipeline.

This module applies per-feature raw transforms and causal EWMA normalization to
raw engine feature vectors. It consumes already-built feature rows ordered on
the microsecond local clock. It does not compute features, build labels, parse
market data, fit global dataset scalers, or write storage artifacts.
"""

from dataclasses import dataclass
import math

import numpy as np

from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_NAMES_HASH,
    FEATURE_SPECS,
    FEATURE_SPECS_HASH,
    TransformKey,
    feature_index,
    feature_spec_by_name,
)

DEFAULT_FAST_HALF_LIFE_US = 30_000_000
DEFAULT_MEDIUM_HALF_LIFE_US = 120_000_000
DEFAULT_SLOW_HALF_LIFE_US = 600_000_000
DEFAULT_MIN_OBS = 20
DEFAULT_VARIANCE_FLOOR = 1e-6
DEFAULT_Z_CLIP = 8.0
DEFAULT_RAW_CLIP = 1_000_000_000.0
DEFAULT_BOUNDED_ABS_CLIP = 10.0
FLOAT_EPS = 1e-12

TRANSFORM_KEYS = tuple(spec.transform_key for spec in FEATURE_SPECS)

_IDENTITY_EWMA_FAST_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.IDENTITY_EWMA_FAST)
_IDENTITY_EWMA_MEDIUM_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.IDENTITY_EWMA_MEDIUM)
_IDENTITY_EWMA_SLOW_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.IDENTITY_EWMA_SLOW)
_IDENTITY_NO_EWMA_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.IDENTITY_NO_EWMA)
_LOG1P_POS_NO_EWMA_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.LOG1P_POS_NO_EWMA)
_LOG1P_POS_EWMA_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.LOG1P_POS_EWMA)
_SIGNED_LOG1P_EWMA_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.SIGNED_LOG1P_EWMA)
_RATIO_BOUNDED_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.RATIO_BOUNDED)
_SIGN_NO_EWMA_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.SIGN_NO_EWMA)
_TIME_LOG1P_NO_EWMA_IDX = tuple(i for i, k in enumerate(TRANSFORM_KEYS) if k == TransformKey.TIME_LOG1P_NO_EWMA)

EWMA_FEATURE_INDICES_DEFAULT = tuple(
    i for i, k in enumerate(TRANSFORM_KEYS)
    if k in {
        TransformKey.IDENTITY_EWMA_FAST,
        TransformKey.IDENTITY_EWMA_MEDIUM,
        TransformKey.IDENTITY_EWMA_SLOW,
        TransformKey.LOG1P_POS_EWMA,
        TransformKey.SIGNED_LOG1P_EWMA,
    }
)
NO_EWMA_FEATURE_INDICES_DEFAULT = tuple(i for i in range(FEATURE_COUNT) if i not in set(EWMA_FEATURE_INDICES_DEFAULT))

assert len(FEATURE_SPECS) == FEATURE_COUNT
assert len(TRANSFORM_KEYS) == FEATURE_COUNT
assert set(EWMA_FEATURE_INDICES_DEFAULT).union(NO_EWMA_FEATURE_INDICES_DEFAULT) == set(range(FEATURE_COUNT))
assert set(EWMA_FEATURE_INDICES_DEFAULT).intersection(NO_EWMA_FEATURE_INDICES_DEFAULT) == set()
assert isinstance(FEATURE_NAMES_HASH, str) and FEATURE_NAMES_HASH
assert isinstance(FEATURE_SPECS_HASH, str) and FEATURE_SPECS_HASH


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be non-negative int")
    return value


def _require_positive_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be finite positive float")
    return out


def _require_output_dtype(value: str) -> str:
    if value not in {"float32", "float64"}:
        raise ValueError("output_dtype must be float32 or float64")
    return value


def _coerce_ts_array(ts_us: np.ndarray, expected_len: int) -> np.ndarray:
    _require_nonnegative_int(expected_len, "expected_len")
    arr = np.asarray(ts_us)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError("ts_us must be 1D with expected length")
    if arr.dtype == np.bool_:
        raise ValueError("ts_us bool dtype not allowed")
    if np.issubdtype(arr.dtype, np.integer):
        out = arr.astype(np.int64, copy=False)
    else:
        arrf = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(arrf)) or not np.all(arrf == np.floor(arrf)):
            raise ValueError("ts_us float entries must be finite integers")
        out = arrf.astype(np.int64)
    if np.any(out < 0) or np.any(np.diff(out) < 0):
        raise ValueError("ts_us must be nonnegative and nondecreasing")
    return np.ascontiguousarray(out)


def _coerce_feature_vector(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (FEATURE_COUNT,):
        raise ValueError("raw must have shape (FEATURE_COUNT,)")
    return np.ascontiguousarray(arr)


def _coerce_feature_matrix(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_COUNT:
        raise ValueError("raw_matrix must have shape (n, FEATURE_COUNT)")
    return np.ascontiguousarray(arr)


def _finite_or_zero(arr: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.array(arr, dtype=np.float64, copy=True)
    mask = ~np.isfinite(out)
    c = int(mask.sum())
    if c:
        out[mask] = 0.0
    return out, c


@dataclass(frozen=True, slots=True)
class TransformConfig:
    fast_half_life_us: int = DEFAULT_FAST_HALF_LIFE_US
    medium_half_life_us: int = DEFAULT_MEDIUM_HALF_LIFE_US
    slow_half_life_us: int = DEFAULT_SLOW_HALF_LIFE_US
    min_obs: int = DEFAULT_MIN_OBS
    variance_floor: float = DEFAULT_VARIANCE_FLOOR
    z_clip: float = DEFAULT_Z_CLIP
    raw_clip: float = DEFAULT_RAW_CLIP
    bounded_abs_clip: float = DEFAULT_BOUNDED_ABS_CLIP
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fast_half_life_us", _require_positive_int(self.fast_half_life_us, "fast_half_life_us"))
        object.__setattr__(self, "medium_half_life_us", _require_positive_int(self.medium_half_life_us, "medium_half_life_us"))
        object.__setattr__(self, "slow_half_life_us", _require_positive_int(self.slow_half_life_us, "slow_half_life_us"))
        object.__setattr__(self, "min_obs", _require_positive_int(self.min_obs, "min_obs"))
        object.__setattr__(self, "variance_floor", _require_positive_float(self.variance_floor, "variance_floor"))
        object.__setattr__(self, "z_clip", _require_positive_float(self.z_clip, "z_clip"))
        object.__setattr__(self, "raw_clip", _require_positive_float(self.raw_clip, "raw_clip"))
        object.__setattr__(self, "bounded_abs_clip", _require_positive_float(self.bounded_abs_clip, "bounded_abs_clip"))
        object.__setattr__(self, "output_dtype", _require_output_dtype(self.output_dtype))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.output_dtype)

    def as_dict(self) -> dict[str, object]:
        return {
            "fast_half_life_us": self.fast_half_life_us,
            "medium_half_life_us": self.medium_half_life_us,
            "slow_half_life_us": self.slow_half_life_us,
            "min_obs": self.min_obs,
            "variance_floor": self.variance_floor,
            "z_clip": self.z_clip,
            "raw_clip": self.raw_clip,
            "bounded_abs_clip": self.bounded_abs_clip,
            "output_dtype": self.output_dtype,
            "feature_names_hash": FEATURE_NAMES_HASH,
            "feature_specs_hash": FEATURE_SPECS_HASH,
        }

@dataclass(slots=True)
class TransformDiagnostics:
    rows_seen: int = 0
    nonfinite_raw_count: int = 0
    raw_clip_count: int = 0
    bounded_clip_count: int = 0
    z_clip_count: int = 0
    warmup_ewma_count: int = 0
    def reset(self) -> None:
        self.rows_seen = 0; self.nonfinite_raw_count = 0; self.raw_clip_count = 0; self.bounded_clip_count = 0; self.z_clip_count = 0; self.warmup_ewma_count = 0
    def as_dict(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.__dict__.items()}

@dataclass(frozen=True, slots=True)
class TransformStateSnapshot:
    rows_seen: int
    last_ts_us: int | None
    mean: np.ndarray
    var: np.ndarray
    count: np.ndarray
    def __post_init__(self) -> None:
        object.__setattr__(self, "rows_seen", _require_nonnegative_int(self.rows_seen, "rows_seen"))
        if self.last_ts_us is not None:
            object.__setattr__(self, "last_ts_us", _require_nonnegative_int(self.last_ts_us, "last_ts_us"))
        mean = np.ascontiguousarray(np.asarray(self.mean, dtype=np.float64))
        var = np.ascontiguousarray(np.asarray(self.var, dtype=np.float64))
        count = np.ascontiguousarray(np.asarray(self.count, dtype=np.int64))
        if mean.shape != (FEATURE_COUNT,) or var.shape != (FEATURE_COUNT,) or count.shape != (FEATURE_COUNT,):
            raise ValueError("snapshot arrays must be shape (FEATURE_COUNT,)")
        if np.any(~np.isfinite(var)) or np.any(var < 0) or np.any(count < 0):
            raise ValueError("invalid snapshot arrays")
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "var", var.copy())
        object.__setattr__(self, "count", count.copy())


def feature_transform_keys() -> tuple[TransformKey, ...]: return TRANSFORM_KEYS

def transform_key_for_feature(name: str) -> TransformKey: return feature_spec_by_name(name).transform_key

def ewma_feature_indices() -> tuple[int, ...]: return EWMA_FEATURE_INDICES_DEFAULT

def no_ewma_feature_indices() -> tuple[int, ...]: return NO_EWMA_FEATURE_INDICES_DEFAULT


def _base_transform_values_with_counts(raw64: np.ndarray, config: TransformConfig) -> tuple[np.ndarray, int, int, int]:
    base, nonfinite = _finite_or_zero(raw64)
    raw_clip_count = 0
    bounded_clip_count = 0
    if _IDENTITY_EWMA_FAST_IDX or _IDENTITY_EWMA_MEDIUM_IDX or _IDENTITY_EWMA_SLOW_IDX or _IDENTITY_NO_EWMA_IDX:
        idx = np.array(_IDENTITY_EWMA_FAST_IDX + _IDENTITY_EWMA_MEDIUM_IDX + _IDENTITY_EWMA_SLOW_IDX + _IDENTITY_NO_EWMA_IDX, dtype=np.int64)
        orig = raw64[idx]
        lo, hi = -config.raw_clip, config.raw_clip
        clipped = np.clip(base[idx], lo, hi)
        raw_clip_count += int(np.sum(np.isfinite(orig) & ((orig < lo) | (orig > hi))))
        base[idx] = clipped
    if _LOG1P_POS_NO_EWMA_IDX:
        idx = np.array(_LOG1P_POS_NO_EWMA_IDX, dtype=np.int64); base[idx] = np.log1p(np.maximum(base[idx], 0.0))
    if _LOG1P_POS_EWMA_IDX:
        idx = np.array(_LOG1P_POS_EWMA_IDX, dtype=np.int64); base[idx] = np.log1p(np.maximum(base[idx], 0.0))
    if _SIGNED_LOG1P_EWMA_IDX:
        idx = np.array(_SIGNED_LOG1P_EWMA_IDX, dtype=np.int64); x = base[idx]; base[idx] = np.sign(x) * np.log1p(np.abs(x))
    if _RATIO_BOUNDED_IDX:
        idx = np.array(_RATIO_BOUNDED_IDX, dtype=np.int64); orig = raw64[idx]; lo, hi = -config.bounded_abs_clip, config.bounded_abs_clip; base[idx] = np.clip(base[idx], lo, hi); bounded_clip_count += int(np.sum(np.isfinite(orig) & ((orig < lo) | (orig > hi))))
    if _SIGN_NO_EWMA_IDX:
        idx = np.array(_SIGN_NO_EWMA_IDX, dtype=np.int64); orig = raw64[idx]; base[idx] = np.clip(base[idx], -1.0, 1.0); bounded_clip_count += int(np.sum(np.isfinite(orig) & ((orig < -1.0) | (orig > 1.0))))
    if _TIME_LOG1P_NO_EWMA_IDX:
        idx = np.array(_TIME_LOG1P_NO_EWMA_IDX, dtype=np.int64); base[idx] = np.log1p(np.maximum(base[idx], 0.0))
    return base, nonfinite, raw_clip_count, bounded_clip_count

def base_transform_values(raw: np.ndarray, config: TransformConfig | None = None) -> np.ndarray:
    cfg = config or TransformConfig()
    raw64 = _coerce_feature_vector(raw)
    base, _, _, _ = _base_transform_values_with_counts(raw64, cfg)
    return base

class CausalFeatureTransformer:
    def __init__(self, config: TransformConfig | None = None, snapshot: TransformStateSnapshot | None = None):
        self.config = config or TransformConfig()
        self.mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self.var = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self.count = np.zeros(FEATURE_COUNT, dtype=np.int64)
        self.rows_seen = 0
        self.last_ts_us = None
        self.diagnostics = TransformDiagnostics()
        self._ewma_indices = np.asarray(EWMA_FEATURE_INDICES_DEFAULT, dtype=np.int64)
        self._no_ewma_indices = np.asarray(NO_EWMA_FEATURE_INDICES_DEFAULT, dtype=np.int64)
        self._half_life_us_by_index = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self._half_life_us_by_index[list(_IDENTITY_EWMA_FAST_IDX)] = self.config.fast_half_life_us
        self._half_life_us_by_index[list(_IDENTITY_EWMA_MEDIUM_IDX)] = self.config.medium_half_life_us
        self._half_life_us_by_index[list(_IDENTITY_EWMA_SLOW_IDX)] = self.config.slow_half_life_us
        self._half_life_us_by_index[list(_LOG1P_POS_EWMA_IDX)] = self.config.medium_half_life_us
        self._half_life_us_by_index[list(_SIGNED_LOG1P_EWMA_IDX)] = self.config.medium_half_life_us
        if snapshot is not None:
            self.load_snapshot(snapshot)

    @property
    def is_initialized(self) -> bool: return self.rows_seen > 0
    def reset(self) -> None:
        self.mean.fill(0.0); self.var.fill(0.0); self.count.fill(0); self.rows_seen = 0; self.last_ts_us = None; self.diagnostics.reset()
    def snapshot(self) -> TransformStateSnapshot:
        return TransformStateSnapshot(self.rows_seen, self.last_ts_us, self.mean.copy(), self.var.copy(), self.count.copy())
    def load_snapshot(self, snapshot: TransformStateSnapshot) -> None:
        s = TransformStateSnapshot(snapshot.rows_seen, snapshot.last_ts_us, snapshot.mean, snapshot.var, snapshot.count)
        self.mean = s.mean.copy(); self.var = s.var.copy(); self.count = s.count.copy(); self.rows_seen = s.rows_seen; self.last_ts_us = s.last_ts_us; self.diagnostics.reset()
    def transform_one(self, ts_us: int, raw: np.ndarray) -> np.ndarray:
        ts = _require_nonnegative_int(ts_us, "ts_us")
        if self.last_ts_us is not None and ts < self.last_ts_us: raise ValueError("decreasing ts_us")
        raw64 = _coerce_feature_vector(raw)
        base, n_nonfinite, rawc, boundc = _base_transform_values_with_counts(raw64, self.config)
        out = base.copy()
        idx = self._ewma_indices
        if idx.size:
            counts = self.count[idx]
            warm = counts < self.config.min_obs
            out[idx[warm]] = 0.0
            self.diagnostics.warmup_ewma_count += int(np.sum(warm))
            ready_idx = idx[~warm]
            if ready_idx.size:
                v = self.var[ready_idx]
                valid = (v > self.config.variance_floor) & (np.sqrt(v) > math.sqrt(self.config.variance_floor))
                bad = ~valid
                if np.any(bad):
                    out[ready_idx[bad]] = 0.0
                good_idx = ready_idx[valid]
                if good_idx.size:
                    z = (base[good_idx] - self.mean[good_idx]) / np.sqrt(self.var[good_idx])
                    zc = np.clip(z, -self.config.z_clip, self.config.z_clip)
                    self.diagnostics.z_clip_count += int(np.sum(z != zc))
                    out[good_idx] = zc
        self._update_ewma_state(ts, base)
        self.rows_seen += 1
        self.last_ts_us = ts
        self.diagnostics.rows_seen += 1
        self.diagnostics.nonfinite_raw_count += n_nonfinite
        self.diagnostics.raw_clip_count += rawc
        self.diagnostics.bounded_clip_count += boundc
        out = out.astype(self.config.dtype, copy=False)
        return np.ascontiguousarray(out)
    def _update_ewma_state(self, ts_us: int, base: np.ndarray) -> None:
        idx = self._ewma_indices
        if idx.size == 0: return
        if self.rows_seen == 0:
            self.mean[idx] = base[idx]; self.var[idx] = 0.0; self.count[idx] = 1; return
        dt = max(0, ts_us - int(self.last_ts_us))
        if dt == 0:
            alpha = np.zeros(idx.shape[0], dtype=np.float64)
        else:
            alpha = 1.0 - np.exp(-math.log(2.0) * float(dt) / self._half_life_us_by_index[idx])
        delta = base[idx] - self.mean[idx]
        new_mean = self.mean[idx] + alpha * delta
        new_var = (1.0 - alpha) * (self.var[idx] + alpha * delta * delta)
        self.mean[idx] = new_mean
        self.var[idx] = np.maximum(new_var, 0.0)
        self.count[idx] += 1
    def transform_many(self, ts_us: np.ndarray, raw_matrix: np.ndarray) -> np.ndarray:
        mat = _coerce_feature_matrix(raw_matrix)
        ts = _coerce_ts_array(ts_us, mat.shape[0])
        out = np.empty((mat.shape[0], FEATURE_COUNT), dtype=self.config.dtype)
        for i in range(mat.shape[0]): out[i] = self.transform_one(int(ts[i]), mat[i])
        return out
    def diagnostics_snapshot(self) -> TransformDiagnostics:
        d = self.diagnostics
        return TransformDiagnostics(d.rows_seen, d.nonfinite_raw_count, d.raw_clip_count, d.bounded_clip_count, d.z_clip_count, d.warmup_ewma_count)

def transform_feature_matrix_causal(ts_us: np.ndarray, raw_matrix: np.ndarray, config: TransformConfig | None = None, initial_snapshot: TransformStateSnapshot | None = None) -> tuple[np.ndarray, TransformStateSnapshot, TransformDiagnostics]:
    transformer = CausalFeatureTransformer(config=config, snapshot=initial_snapshot)
    transformed = transformer.transform_many(ts_us, raw_matrix)
    return transformed, transformer.snapshot(), transformer.diagnostics_snapshot()

__all__ = [
    "DEFAULT_FAST_HALF_LIFE_US", "DEFAULT_MEDIUM_HALF_LIFE_US", "DEFAULT_SLOW_HALF_LIFE_US", "DEFAULT_MIN_OBS",
    "DEFAULT_VARIANCE_FLOOR", "DEFAULT_Z_CLIP", "DEFAULT_RAW_CLIP", "DEFAULT_BOUNDED_ABS_CLIP", "TransformConfig",
    "TransformDiagnostics", "TransformStateSnapshot", "CausalFeatureTransformer", "feature_transform_keys",
    "transform_key_for_feature", "ewma_feature_indices", "no_ewma_feature_indices", "base_transform_values",
    "transform_feature_matrix_causal",
]
