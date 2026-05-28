"""Target construction for storage-backed MMRT linear models.

This module consumes already-matured storage label columns and converts one
fixed-horizon return column into target arrays for linear heads. It does not
parse market data, compute labels from prices, inspect row timing fields,
build splits, fit preprocessing, train models, or evaluate metrics.

The v1 gated linear heads are no_move, direction, magnitude_up, and magnitude_down.
Stored return labels are the source of truth.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa

from mmrt.storage import manifest as mf

DEFAULT_TARGET_HORIZON_US = 1_000_000
DEFAULT_MOVE_DEADBAND_BPS = 0.0
DEFAULT_TARGET_DTYPE = "float32"
ALLOWED_TARGET_DTYPES = ("float32", "float64")

DIRECTION_INVALID_CLASS = -1
DIRECTION_DOWN_CLASS = 0
DIRECTION_UP_CLASS = 1


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative float")
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return out


def _require_output_dtype(value: str) -> str:
    if value not in ALLOWED_TARGET_DTYPES:
        raise ValueError(f"output_dtype must be one of {ALLOWED_TARGET_DTYPES}")
    return value


def _coerce_return_vector(values: np.ndarray, *, dtype: np.dtype, name: str = "return_bps") -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    out = np.asarray(arr, dtype=dtype)
    out = np.ascontiguousarray(out, dtype=dtype)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True, slots=True)
class LinearTargetConfig:
    target_horizon_us: int = DEFAULT_TARGET_HORIZON_US
    move_deadband_bps: float = DEFAULT_MOVE_DEADBAND_BPS
    output_dtype: str = DEFAULT_TARGET_DTYPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_horizon_us", _require_positive_int(self.target_horizon_us, "target_horizon_us"))
        object.__setattr__(
            self,
            "move_deadband_bps",
            _require_nonnegative_finite_float(self.move_deadband_bps, "move_deadband_bps"),
        )
        object.__setattr__(self, "output_dtype", _require_output_dtype(self.output_dtype))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.output_dtype)


@dataclass(frozen=True, slots=True)
class LinearTargetBatch:
    y_return_bps: np.ndarray
    y_no_move: np.ndarray
    y_direction: np.ndarray
    y_magnitude_up: np.ndarray
    y_magnitude_down: np.ndarray
    no_move_mask: np.ndarray
    move_mask: np.ndarray
    up_move_mask: np.ndarray
    down_move_mask: np.ndarray
    target_column: str
    horizon_us: int

    def __post_init__(self) -> None:
        ret = np.asarray(self.y_return_bps)
        if ret.ndim != 1 or ret.dtype.name not in ALLOWED_TARGET_DTYPES or not ret.flags.c_contiguous or not np.isfinite(ret).all():
            raise ValueError("return_bps must be 1D contiguous finite float32/float64")

        y_dir = np.asarray(self.y_direction)
        if y_dir.ndim != 1 or y_dir.dtype != np.int8 or y_dir.shape[0] != ret.shape[0]:
            raise ValueError("y_direction must be 1D int8 with matching length")
        if not np.isin(y_dir, np.array([DIRECTION_INVALID_CLASS, DIRECTION_DOWN_CLASS, DIRECTION_UP_CLASS], dtype=np.int8)).all():
            raise ValueError("y_direction has invalid class values")

        y_no_move = np.asarray(self.y_no_move)
        if y_no_move.ndim != 1 or y_no_move.shape[0] != ret.shape[0]:
            raise ValueError("y_no_move must be 1D with matching length")
        if not np.isin(y_no_move, np.array([0.0, 1.0], dtype=y_no_move.dtype)).all():
            raise ValueError("y_no_move must be binary 0/1")

        no_move_mask = np.asarray(self.no_move_mask)
        move_mask = np.asarray(self.move_mask)
        up_move_mask = np.asarray(self.up_move_mask)
        down_move_mask = np.asarray(self.down_move_mask)
        for arr, name in ((no_move_mask, "no_move_mask"), (move_mask, "move_mask"), (up_move_mask, "up_move_mask"), (down_move_mask, "down_move_mask")):
            if arr.ndim != 1 or arr.dtype != np.bool_ or arr.shape[0] != ret.shape[0]:
                raise ValueError(f"{name} must be 1D bool with matching length")
        if not np.array_equal(no_move_mask, ~move_mask):
            raise ValueError("no_move_mask must equal ~move_mask")
        if not np.array_equal(move_mask, up_move_mask | down_move_mask):
            raise ValueError("move_mask must equal up_move_mask | down_move_mask")
        if np.any(up_move_mask & down_move_mask):
            raise ValueError("up_move_mask and down_move_mask must be disjoint")
        if not np.array_equal(y_dir == DIRECTION_UP_CLASS, up_move_mask):
            raise ValueError("y_direction up class mismatch")
        if not np.array_equal(y_dir == DIRECTION_DOWN_CLASS, down_move_mask):
            raise ValueError("y_direction down class mismatch")

        up = np.asarray(self.y_magnitude_up)
        down = np.asarray(self.y_magnitude_down)
        for arr, name in ((up, "y_magnitude_up"), (down, "y_magnitude_down")):
            if arr.ndim != 1 or arr.shape[0] != ret.shape[0] or arr.dtype != ret.dtype:
                raise ValueError(f"{name} must be 1D matching length and dtype")
            if not np.isfinite(arr).all() or np.any(arr < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")

        if not isinstance(self.target_column, str) or not self.target_column.strip():
            raise ValueError("target_column must be a non-empty string")
        _require_positive_int(self.horizon_us, "horizon_us")

        object.__setattr__(self, "y_return_bps", np.ascontiguousarray(ret, dtype=ret.dtype).copy())
        object.__setattr__(self, "y_no_move", np.ascontiguousarray(y_no_move, dtype=ret.dtype).copy())
        object.__setattr__(self, "y_direction", np.ascontiguousarray(y_dir, dtype=np.int8).copy())
        object.__setattr__(self, "y_magnitude_up", np.ascontiguousarray(up, dtype=ret.dtype).copy())
        object.__setattr__(self, "y_magnitude_down", np.ascontiguousarray(down, dtype=ret.dtype).copy())
        object.__setattr__(self, "no_move_mask", np.ascontiguousarray(no_move_mask, dtype=np.bool_).copy())
        object.__setattr__(self, "move_mask", np.ascontiguousarray(move_mask, dtype=np.bool_).copy())
        object.__setattr__(self, "up_move_mask", np.ascontiguousarray(up_move_mask, dtype=np.bool_).copy())
        object.__setattr__(self, "down_move_mask", np.ascontiguousarray(down_move_mask, dtype=np.bool_).copy())
        object.__setattr__(self, "target_column", self.target_column.strip())

    @property
    def n_rows(self) -> int:
        return int(self.y_return_bps.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return self.y_return_bps.dtype

    @property
    def direction_valid_count(self) -> int:
        return int(self.move_mask.sum())


def target_column_for_horizon(horizon_us: int) -> str:
    h = _require_positive_int(horizon_us, "horizon_us")
    return f"{mf.LABEL_COLUMN_PREFIX}{h}us"


def resolve_target_column(manifest: mf.StorageManifest, config: LinearTargetConfig | None = None) -> str:
    if not isinstance(manifest, mf.StorageManifest):
        raise ValueError("manifest must be StorageManifest")
    cfg = config or LinearTargetConfig()
    h = cfg.target_horizon_us
    if h not in manifest.label_spec.horizons_us:
        raise ValueError("target horizon not found in label_spec")
    col = target_column_for_horizon(h)
    if col not in manifest.label_columns:
        raise ValueError("target column not found in manifest.label_columns")
    if tuple(manifest.label_columns) != mf.label_columns(manifest.label_spec):
        raise ValueError("label column schema drift")
    return col


def target_column_projection(manifest: mf.StorageManifest, config: LinearTargetConfig | None = None) -> tuple[str, ...]:
    return (resolve_target_column(manifest, config),)


def table_to_return_vector(table: pa.Table, target_column: str, *, output_dtype: str = DEFAULT_TARGET_DTYPE) -> np.ndarray:
    if not isinstance(table, pa.Table):
        raise ValueError("table must be pyarrow.Table")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("target_column must be a non-empty string")
    _require_output_dtype(output_dtype)
    if target_column not in table.column_names:
        raise ValueError("target column missing from table")
    col = table[target_column]
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    arr = col.to_numpy(zero_copy_only=False)
    out = np.asarray(arr, dtype=np.dtype(output_dtype))
    out = np.ascontiguousarray(out, dtype=np.dtype(output_dtype))
    if out.ndim != 1:
        raise ValueError("target column must be 1D")
    if not np.isfinite(out).all():
        raise ValueError("target column must be finite")
    return out


def build_linear_targets(return_bps: np.ndarray, *, config: LinearTargetConfig | None = None, target_column: str | None = None) -> LinearTargetBatch:
    cfg = config or LinearTargetConfig()
    ret = _coerce_return_vector(return_bps, dtype=cfg.dtype)
    h = cfg.target_horizon_us
    col = target_column or target_column_for_horizon(h)
    deadband = cfg.move_deadband_bps

    abs_ret = np.abs(ret)
    no_move_mask = abs_ret <= deadband
    move_mask = abs_ret > deadband
    up = ret > deadband
    down = ret < -deadband

    y_direction = np.full(ret.shape[0], DIRECTION_INVALID_CLASS, dtype=np.int8)
    y_direction[up] = DIRECTION_UP_CLASS
    y_direction[down] = DIRECTION_DOWN_CLASS

    y_magnitude_up = np.log1p(np.maximum(ret, 0.0)).astype(cfg.dtype, copy=False)
    y_magnitude_down = np.log1p(np.maximum(-ret, 0.0)).astype(cfg.dtype, copy=False)

    return LinearTargetBatch(
        y_return_bps=ret,
        y_no_move=no_move_mask.astype(cfg.dtype, copy=False),
        y_direction=y_direction,
        y_magnitude_up=y_magnitude_up,
        y_magnitude_down=y_magnitude_down,
        no_move_mask=no_move_mask,
        move_mask=move_mask,
        up_move_mask=up,
        down_move_mask=down,
        target_column=col,
        horizon_us=h,
    )


class LinearTargetBuilder:
    def __init__(self, config: LinearTargetConfig | None = None, *, manifest: mf.StorageManifest | None = None):
        self.config = config or LinearTargetConfig()
        self.target_column: str | None = None
        self.label_spec: dict[str, object] | None = None
        if manifest is not None:
            self.resolve_target_column(manifest)

    def resolve_target_column(self, manifest: mf.StorageManifest) -> str:
        col = resolve_target_column(manifest, self.config)
        self.target_column = col
        self.label_spec = mf.label_spec_to_dict(manifest.label_spec)
        return col

    def column_projection(self, manifest: mf.StorageManifest) -> tuple[str, ...]:
        return (self.resolve_target_column(manifest),)

    def transform_table(self, table: pa.Table, *, manifest: mf.StorageManifest | None = None) -> LinearTargetBatch:
        if manifest is not None or self.target_column is None:
            if manifest is None:
                raise ValueError("manifest is required when target_column is unresolved")
            self.resolve_target_column(manifest)
        ret = table_to_return_vector(table, self.target_column, output_dtype=self.config.output_dtype)
        return build_linear_targets(ret, config=self.config, target_column=self.target_column)

    def transform_numpy(self, return_bps: np.ndarray, *, target_column: str | None = None) -> LinearTargetBatch:
        if target_column is not None:
            if not isinstance(target_column, str) or not target_column.strip():
                raise ValueError("target_column must be a non-empty string")
            if self.target_column is not None and self.target_column != target_column:
                raise ValueError("target_column does not match resolved target")
            self.target_column = target_column
        elif self.target_column is None:
            self.target_column = target_column_for_horizon(self.config.target_horizon_us)
        return build_linear_targets(return_bps, config=self.config, target_column=self.target_column)

    def as_dict(self) -> dict[str, object]:
        return {
            "target_horizon_us": self.config.target_horizon_us,
            "target_column": self.target_column,
            "move_deadband_bps": self.config.move_deadband_bps,
            "output_dtype": self.config.output_dtype,
            "direction_invalid_class": DIRECTION_INVALID_CLASS,
            "direction_down_class": DIRECTION_DOWN_CLASS,
            "direction_up_class": DIRECTION_UP_CLASS,
            "magnitude_formula": "log1p_positive_bps",
            "label_spec": self.label_spec,
        }


def make_target_builder(manifest: mf.StorageManifest, config: LinearTargetConfig | None = None) -> LinearTargetBuilder:
    builder = LinearTargetBuilder(config=config)
    builder.resolve_target_column(manifest)
    return builder


__all__: Sequence[str] = [
    "DEFAULT_TARGET_HORIZON_US",
    "DEFAULT_MOVE_DEADBAND_BPS",
    "DEFAULT_TARGET_DTYPE",
    "ALLOWED_TARGET_DTYPES",
    "DIRECTION_INVALID_CLASS",
    "DIRECTION_DOWN_CLASS",
    "DIRECTION_UP_CLASS",
    "LinearTargetConfig",
    "LinearTargetBatch",
    "LinearTargetBuilder",
    "target_column_for_horizon",
    "resolve_target_column",
    "target_column_projection",
    "table_to_return_vector",
    "build_linear_targets",
    "make_target_builder",
]
