"""Identity feature extraction for storage-backed MMRT linear models.

This module consumes already-materialized feature columns from storage outputs
outputs and converts them to NumPy matrices for linear models. It does not
parse market data, compute features, build rolling windows, apply transforms,
compute targets, train models, or evaluate metrics.

The only supported extractor is identity/projection over stored feature columns.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa

from mmrt.storage import manifest as mf

DEFAULT_EXTRACTOR_DTYPE = "float32"
ALLOWED_EXTRACTOR_DTYPES = ("float32", "float64")


def _coerce_feature_columns(feature_columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if feature_columns is None:
        return None
    out = tuple(feature_columns)
    if not out:
        raise ValueError("feature_columns must be non-empty when provided")
    for name in out:
        if not isinstance(name, str):
            raise ValueError("feature_columns entries must be strings")
        if not name:
            raise ValueError("feature_columns entries must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError("feature_columns must not contain duplicates")
    return out


@dataclass(frozen=True, slots=True)
class LinearFeatureExtractorConfig:
    feature_columns: tuple[str, ...] | None = None
    output_dtype: str = DEFAULT_EXTRACTOR_DTYPE

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_columns", _coerce_feature_columns(self.feature_columns))
        if self.output_dtype not in ALLOWED_EXTRACTOR_DTYPES:
            raise ValueError(f"output_dtype must be one of {ALLOWED_EXTRACTOR_DTYPES}")

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.output_dtype)


@dataclass(frozen=True, slots=True)
class LinearFeatureBatch:
    X: np.ndarray
    feature_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        cols = _coerce_feature_columns(self.feature_columns)
        assert cols is not None
        arr = np.asarray(self.X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D NumPy array")
        if arr.shape[1] != len(cols):
            raise ValueError("X column count must match feature_columns length")
        if arr.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("X dtype must be float32 or float64")
        arr = np.ascontiguousarray(arr).copy()
        if not np.isfinite(arr).all():
            raise ValueError("X must contain only finite values")
        object.__setattr__(self, "feature_columns", cols)
        object.__setattr__(self, "X", arr)

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def resolve_feature_columns(
    manifest: mf.StorageManifest,
    feature_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(manifest, mf.StorageManifest):
        raise ValueError("manifest must be StorageManifest")
    available = tuple(manifest.feature_columns)
    selected = available if feature_columns is None else _coerce_feature_columns(feature_columns)
    assert selected is not None
    if len(set(selected)) != len(selected):
        raise ValueError("feature_columns must not contain duplicates")
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"unknown feature columns: {missing}")
    return selected


def table_to_feature_matrix(
    table: pa.Table,
    feature_columns: Sequence[str],
    *,
    output_dtype: str = DEFAULT_EXTRACTOR_DTYPE,
    require_all_finite: bool = True,
    copy: bool = True,
) -> np.ndarray:
    if not isinstance(table, pa.Table):
        raise ValueError("table must be pyarrow.Table")
    cols = _coerce_feature_columns(feature_columns)
    assert cols is not None
    if output_dtype not in ALLOWED_EXTRACTOR_DTYPES:
        raise ValueError(f"output_dtype must be one of {ALLOWED_EXTRACTOR_DTYPES}")
    if not isinstance(require_all_finite, bool):
        raise ValueError("require_all_finite must be bool")
    if not isinstance(copy, bool):
        raise ValueError("copy must be bool")

    missing = [name for name in cols if name not in table.column_names]
    if missing:
        raise ValueError(f"table missing required feature columns: {missing}")

    dtype = np.dtype(output_dtype)
    if table.num_rows == 0:
        X = np.empty((0, len(cols)), dtype=dtype)
    else:
        matrix_cols = []
        for name in cols:
            col = table[name]
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            np_col = col.to_numpy(zero_copy_only=False)
            matrix_cols.append(np.asarray(np_col, dtype=dtype))
        X = np.column_stack(matrix_cols)
        X = np.ascontiguousarray(X, dtype=dtype)

    if require_all_finite and not np.isfinite(X).all():
        raise ValueError("feature matrix contains non-finite values")
    if copy:
        X = X.copy()
    return X


class IdentityFeatureExtractor:
    def __init__(
        self,
        config: LinearFeatureExtractorConfig | None = None,
        *,
        manifest: mf.StorageManifest | None = None,
    ):
        self.config = config or LinearFeatureExtractorConfig()
        self.feature_columns: tuple[str, ...] | None = None
        self.feature_schema_hash: str | None = None
        if manifest is not None:
            self.resolve_feature_columns(manifest)

    def resolve_feature_columns(self, manifest: mf.StorageManifest) -> tuple[str, ...]:
        selected = resolve_feature_columns(manifest, self.config.feature_columns)
        self.feature_columns = selected
        self.feature_schema_hash = manifest.feature_schema.get("feature_specs_hash")
        return selected

    def column_projection(self, manifest: mf.StorageManifest) -> tuple[str, ...]:
        return self.resolve_feature_columns(manifest)

    def transform_table(self, table: pa.Table, *, manifest: mf.StorageManifest | None = None) -> LinearFeatureBatch:
        if manifest is not None or self.feature_columns is None:
            if manifest is None:
                raise ValueError("manifest is required when feature columns are unresolved")
            selected = self.resolve_feature_columns(manifest)
        else:
            selected = self.feature_columns
        X = table_to_feature_matrix(
            table,
            selected,
            output_dtype=self.config.output_dtype,
            require_all_finite=True,
            copy=False,
        )
        return LinearFeatureBatch(X=X, feature_columns=selected)

    def transform_numpy(self, X: np.ndarray, *, feature_columns: Sequence[str] | None = None) -> LinearFeatureBatch:
        if feature_columns is not None:
            cols = _coerce_feature_columns(feature_columns)
            assert cols is not None
            if self.feature_columns is not None and cols != self.feature_columns:
                raise ValueError("feature_columns do not match resolved extractor columns")
            self.feature_columns = cols
        elif self.feature_columns is None:
            raise ValueError("feature_columns must be provided before first transform_numpy call")
        cols2 = self.feature_columns
        assert cols2 is not None

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be 2D")
        if arr.shape[1] != len(cols2):
            raise ValueError("X column count must match feature columns")
        arr = np.asarray(arr, dtype=self.config.dtype)
        arr = np.ascontiguousarray(arr, dtype=self.config.dtype)
        if not np.isfinite(arr).all():
            raise ValueError("X contains non-finite values")
        return LinearFeatureBatch(X=arr, feature_columns=cols2)

    def as_dict(self) -> dict[str, object]:
        return {
            "extractor": "identity",
            "feature_columns": list(self.feature_columns) if self.feature_columns is not None else None,
            "output_dtype": self.config.output_dtype,
            "feature_schema_hash": self.feature_schema_hash,
        }


def make_identity_extractor(
    manifest: mf.StorageManifest,
    config: LinearFeatureExtractorConfig | None = None,
) -> IdentityFeatureExtractor:
    extractor = IdentityFeatureExtractor(config=config)
    extractor.resolve_feature_columns(manifest)
    return extractor


__all__ = [
    "DEFAULT_EXTRACTOR_DTYPE",
    "ALLOWED_EXTRACTOR_DTYPES",
    "LinearFeatureExtractorConfig",
    "LinearFeatureBatch",
    "IdentityFeatureExtractor",
    "resolve_feature_columns",
    "table_to_feature_matrix",
    "make_identity_extractor",
]
