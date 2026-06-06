"""Storage-backed feature health, drift, and redundancy audit for MMRT.

This module reads existing MMRT storage splits and audits already-materialized
feature columns. It computes train-only feature redundancy/correlation,
split-level feature health, and train-vs-val/test distribution drift. It does
not parse Tardis CSV, compute market features, build labels, create splits,
train models, evaluate predictions, select model features, or mutate storage
manifests.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np
import pyarrow as pa

from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.linear import extractors as ex
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd

FEATURE_AUDIT_REPORT_TYPE = "feature_audit"
DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS = 100_000
DEFAULT_FEATURE_AUDIT_BATCH_SIZE = rd.DEFAULT_BATCH_SIZE
DEFAULT_FEATURE_AUDIT_SUMMARY_FILENAME = "feature_audit_summary.json"
DEFAULT_FEATURE_AUDIT_HEALTH_FILENAME = "feature_health.csv"
DEFAULT_FEATURE_AUDIT_DRIFT_FILENAME = "feature_train_val_drift.csv"
DEFAULT_FEATURE_AUDIT_FAMILY_FILENAME = "feature_family_summary.csv"
DEFAULT_FEATURE_AUDIT_CORR_PAIRS_FILENAME = "feature_corr_top_pairs.csv"
DEFAULT_FEATURE_AUDIT_CLUSTERS_FILENAME = "feature_clusters.csv"
DEFAULT_FEATURE_AUDIT_CLUSTER_SUMMARY_FILENAME = "feature_cluster_summary.json"
DEFAULT_LOW_VARIANCE_STD_THRESHOLD = 1e-8
DEFAULT_HIGH_CORR_THRESHOLD = 0.97
DEFAULT_MIN_CORR_OUTPUT_THRESHOLD = 0.90
DEFAULT_MAX_CORR_PAIRS = 1_000
DEFAULT_DRIFT_MEAN_Z_THRESHOLD = 1.0
DEFAULT_DRIFT_STD_RATIO_LOW = 0.5
DEFAULT_DRIFT_STD_RATIO_HIGH = 2.0
ALLOWED_SPLITS = ("train", "val", "test")
ALLOWED_HEALTH_STATUSES = ("ok", "low_variance")
ALLOWED_DRIFT_STATUSES = ("ok", "distribution_shift", "low_variance_train")
ALLOWED_PAIR_STATUSES = ("moderate_redundancy", "high_redundancy")


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty str")
    return value.strip()


def _require_finite_float(value: float, name: str, *, allow_nan: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError(f"{name} must be a float")
    fv = float(value)
    if math.isnan(fv):
        if allow_nan:
            return fv
        raise ValueError(f"{name} must be finite")
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite")
    return fv


def _role_to_str(role: SplitRole | str) -> str:
    role_value = role.value if isinstance(role, SplitRole) else role
    if role_value not in ALLOWED_SPLITS:
        raise ValueError("split must be one of train/val/test")
    return role_value


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"unsupported JSON type: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class FeatureAuditConfig:
    batch_size: int = DEFAULT_FEATURE_AUDIT_BATCH_SIZE
    validate_dataset_on_open: bool = True
    max_sample_rows_per_split: int = DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS
    feature_columns: tuple[str, ...] | None = None
    extractor_dtype: str = ex.DEFAULT_EXTRACTOR_DTYPE
    low_variance_std_threshold: float = DEFAULT_LOW_VARIANCE_STD_THRESHOLD
    high_corr_threshold: float = DEFAULT_HIGH_CORR_THRESHOLD
    min_corr_output_threshold: float = DEFAULT_MIN_CORR_OUTPUT_THRESHOLD
    max_corr_pairs: int = DEFAULT_MAX_CORR_PAIRS
    drift_mean_z_threshold: float = DEFAULT_DRIFT_MEAN_Z_THRESHOLD
    drift_std_ratio_low: float = DEFAULT_DRIFT_STD_RATIO_LOW
    drift_std_ratio_high: float = DEFAULT_DRIFT_STD_RATIO_HIGH

    def __post_init__(self) -> None:
        _require_positive_int(self.batch_size, "batch_size")
        _require_bool(self.validate_dataset_on_open, "validate_dataset_on_open")
        _require_nonnegative_int(self.max_sample_rows_per_split, "max_sample_rows_per_split")

        if self.feature_columns is not None:
            if not isinstance(self.feature_columns, tuple) or not self.feature_columns:
                raise ValueError("feature_columns must be non-empty tuple[str,...] when provided")
            for col in self.feature_columns:
                _require_non_empty_str(col, "feature_columns")
            if len(set(self.feature_columns)) != len(self.feature_columns):
                raise ValueError("feature_columns must not contain duplicates")

        if self.extractor_dtype not in ex.ALLOWED_EXTRACTOR_DTYPES:
            raise ValueError("invalid extractor_dtype")

        if _require_finite_float(self.low_variance_std_threshold, "low_variance_std_threshold") <= 0:
            raise ValueError("low_variance_std_threshold must be > 0")

        min_corr = _require_finite_float(self.min_corr_output_threshold, "min_corr_output_threshold")
        high_corr = _require_finite_float(self.high_corr_threshold, "high_corr_threshold")
        if not (0 < min_corr <= high_corr < 1):
            raise ValueError("require 0 < min_corr_output_threshold <= high_corr_threshold < 1")

        _require_positive_int(self.max_corr_pairs, "max_corr_pairs")

        if _require_finite_float(self.drift_mean_z_threshold, "drift_mean_z_threshold") <= 0:
            raise ValueError("drift_mean_z_threshold must be > 0")
        drift_low = _require_finite_float(self.drift_std_ratio_low, "drift_std_ratio_low")
        drift_high = _require_finite_float(self.drift_std_ratio_high, "drift_std_ratio_high")
        if not (0 < drift_low < 1 < drift_high):
            raise ValueError("require 0 < drift_std_ratio_low < 1 < drift_std_ratio_high")


@dataclass(frozen=True, slots=True)
class _FeatureMeta:
    column: str
    feature_index: int
    source: str
    owner: str
    family: str
    unit: str
    transform_key: str
    required_book_depth: int


def _feature_meta_from_column(column: str) -> _FeatureMeta:
    col = _require_non_empty_str(column, "column")
    if not col.startswith(mf.FEATURE_COLUMN_PREFIX):
        raise ValueError("feature column must have x_ prefix")
    name = col[len(mf.FEATURE_COLUMN_PREFIX) :]
    spec = specs.feature_spec_by_name(name)
    return _FeatureMeta(
        col,
        spec.index,
        spec.source.value,
        spec.owner.value,
        spec.family.value,
        spec.unit.value,
        spec.transform_key.value,
        spec.required_book_depth,
    )


def _resolve_audit_feature_columns(
    manifest: mf.StorageManifest,
    feature_columns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = ex.resolve_feature_columns(manifest, feature_columns)
    selected_set = set(selected)
    return tuple(column for column in manifest.feature_columns if column in selected_set)


@dataclass(slots=True)
class _StreamingFeatureStats:
    n_rows: int
    n_features: int
    mean: np.ndarray
    m2: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray

    @classmethod
    def empty(cls, n_features: int) -> "_StreamingFeatureStats":
        _require_positive_int(n_features, "n_features")
        return cls(
            0,
            n_features,
            np.zeros(n_features, np.float64),
            np.zeros(n_features, np.float64),
            np.full(n_features, np.inf),
            np.full(n_features, -np.inf),
        )

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_features or not np.isfinite(arr).all():
            raise ValueError("invalid X")
        if arr.shape[0] == 0:
            return

        batch_n = int(arr.shape[0])
        batch_mean = arr.mean(0)
        centered = arr - batch_mean
        batch_m2 = np.sum(centered * centered, 0)

        if self.n_rows == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total_n = self.n_rows + batch_n
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * (batch_n / total_n)
            self.m2 = np.maximum(
                self.m2 + batch_m2 + delta * delta * ((self.n_rows * batch_n) / total_n),
                0.0,
            )

        self.min_value = np.minimum(self.min_value, arr.min(0))
        self.max_value = np.maximum(self.max_value, arr.max(0))
        self.n_rows += batch_n

    def variance(self) -> np.ndarray:
        if self.n_rows <= 1:
            return np.zeros(self.n_features, np.float64)
        return self.m2 / float(self.n_rows - 1)

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance(), 0.0))


@dataclass(slots=True)
class _StreamingTrainCorrelationStats:
    n_rows: int
    n_features: int
    sum_x: np.ndarray
    sum_x2: np.ndarray
    cross_x: np.ndarray

    @classmethod
    def empty(cls, n_features: int) -> "_StreamingTrainCorrelationStats":
        _require_positive_int(n_features, "n_features")
        return cls(
            0,
            n_features,
            np.zeros(n_features, np.float64),
            np.zeros(n_features, np.float64),
            np.zeros((n_features, n_features), np.float64),
        )

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_features or not np.isfinite(arr).all():
            raise ValueError("invalid X")
        self.n_rows += int(arr.shape[0])
        self.sum_x += arr.sum(0)
        self.sum_x2 += (arr * arr).sum(0)
        self.cross_x += arr.T @ arr

    def correlation_matrix(self, *, low_variance_std_threshold: float) -> np.ndarray:
        n_rows = self.n_rows
        corr = np.full((self.n_features, self.n_features), np.nan, dtype=np.float64)
        if n_rows <= 1:
            return corr

        cov = (self.cross_x - np.outer(self.sum_x, self.sum_x) / n_rows) / (n_rows - 1)
        var = (self.sum_x2 - (self.sum_x * self.sum_x) / n_rows) / (n_rows - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, np.nan)

        active = std >= low_variance_std_threshold
        corr[~active, :] = np.nan
        corr[:, ~active] = np.nan
        active_indices = np.where(active)[0]
        corr[active_indices, active_indices] = 1.0
        return corr


@dataclass(frozen=True, slots=True)
class FeatureHealthRecord:
    split: str
    feature: str
    feature_index: int
    source: str
    owner: str
    family: str
    unit: str
    transform_key: str
    required_book_depth: int
    n_rows: int
    n_sample_rows: int
    raw_mean: float
    raw_std: float
    raw_min: float
    raw_max: float
    raw_p01: float
    raw_p50: float
    raw_p99: float
    raw_abs_p99: float
    low_variance: bool
    status: str

    def __post_init__(self) -> None:
        _role_to_str(self.split)
        _require_non_empty_str(self.feature, "feature")
        _require_non_empty_str(self.source, "source")
        _require_non_empty_str(self.owner, "owner")
        _require_non_empty_str(self.family, "family")
        _require_non_empty_str(self.unit, "unit")
        _require_non_empty_str(self.transform_key, "transform_key")
        _require_nonnegative_int(self.feature_index, "feature_index")
        _require_nonnegative_int(self.required_book_depth, "required_book_depth")
        n_rows = _require_nonnegative_int(self.n_rows, "n_rows")
        n_sample_rows = _require_nonnegative_int(self.n_sample_rows, "n_sample_rows")
        if n_rows > 0 and n_sample_rows > n_rows:
            raise ValueError("n_sample_rows must be <= n_rows")

        _require_bool(self.low_variance, "low_variance")
        if self.status not in ALLOWED_HEALTH_STATUSES:
            raise ValueError("status must be in ALLOWED_HEALTH_STATUSES")

        _require_finite_float(self.raw_mean, "raw_mean")
        raw_std = _require_finite_float(self.raw_std, "raw_std")
        if raw_std < 0:
            raise ValueError("raw_std must be >= 0")

        if n_rows > 0:
            _require_finite_float(self.raw_min, "raw_min")
            _require_finite_float(self.raw_max, "raw_max")
        else:
            _require_finite_float(self.raw_min, "raw_min", allow_nan=True)
            _require_finite_float(self.raw_max, "raw_max", allow_nan=True)

        if n_sample_rows > 0:
            _require_finite_float(self.raw_p01, "raw_p01")
            _require_finite_float(self.raw_p50, "raw_p50")
            _require_finite_float(self.raw_p99, "raw_p99")
            _require_finite_float(self.raw_abs_p99, "raw_abs_p99")
        else:
            _require_finite_float(self.raw_p01, "raw_p01", allow_nan=True)
            _require_finite_float(self.raw_p50, "raw_p50", allow_nan=True)
            _require_finite_float(self.raw_p99, "raw_p99", allow_nan=True)
            _require_finite_float(self.raw_abs_p99, "raw_abs_p99", allow_nan=True)

        if self.low_variance and self.status != "low_variance":
            raise ValueError("low_variance=True requires status=low_variance")
        if not self.low_variance and self.status != "ok":
            raise ValueError("low_variance=False requires status=ok")


@dataclass(frozen=True, slots=True)
class FeatureDriftRecord:
    split: str
    feature: str
    feature_index: int
    source: str
    owner: str
    family: str
    train_mean: float
    train_std: float
    split_mean: float
    split_std: float
    mean_shift_train_std: float
    std_ratio: float
    train_p50: float
    split_p50: float
    p50_shift_train_std: float
    status: str

    def __post_init__(self) -> None:
        if self.split not in ("val", "test"):
            raise ValueError("split must be val/test for drift")
        _require_non_empty_str(self.feature, "feature")
        _require_non_empty_str(self.source, "source")
        _require_non_empty_str(self.owner, "owner")
        _require_non_empty_str(self.family, "family")
        _require_nonnegative_int(self.feature_index, "feature_index")
        if self.status not in ALLOWED_DRIFT_STATUSES:
            raise ValueError("status must be in ALLOWED_DRIFT_STATUSES")

        _require_finite_float(self.train_mean, "train_mean")
        train_std = _require_finite_float(self.train_std, "train_std")
        _require_finite_float(self.split_mean, "split_mean")
        split_std = _require_finite_float(self.split_std, "split_std")
        if train_std < 0 or split_std < 0:
            raise ValueError("train_std and split_std must be >= 0")

        _require_finite_float(self.train_p50, "train_p50", allow_nan=True)
        _require_finite_float(self.split_p50, "split_p50", allow_nan=True)

        if self.status == "low_variance_train":
            if not math.isnan(float(self.mean_shift_train_std)):
                raise ValueError("mean_shift_train_std must be NaN when low_variance_train")
            if not math.isnan(float(self.std_ratio)):
                raise ValueError("std_ratio must be NaN when low_variance_train")
            if not math.isnan(float(self.p50_shift_train_std)):
                raise ValueError("p50_shift_train_std must be NaN when low_variance_train")
            return

        _require_finite_float(self.mean_shift_train_std, "mean_shift_train_std")
        std_ratio = _require_finite_float(self.std_ratio, "std_ratio")
        if std_ratio < 0:
            raise ValueError("std_ratio must be >= 0")

        if math.isnan(float(self.train_p50)) or math.isnan(float(self.split_p50)):
            _require_finite_float(self.p50_shift_train_std, "p50_shift_train_std", allow_nan=True)
        else:
            _require_finite_float(self.p50_shift_train_std, "p50_shift_train_std")


@dataclass(frozen=True, slots=True)
class FeatureCorrelationPairRecord:
    feature_a: str
    feature_b: str
    index_a: int
    index_b: int
    source_a: str
    source_b: str
    family_a: str
    family_b: str
    corr: float
    abs_corr: float
    same_source: bool
    same_family: bool
    status: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.feature_a, "feature_a")
        _require_non_empty_str(self.feature_b, "feature_b")
        _require_non_empty_str(self.source_a, "source_a")
        _require_non_empty_str(self.source_b, "source_b")
        _require_non_empty_str(self.family_a, "family_a")
        _require_non_empty_str(self.family_b, "family_b")
        index_a = _require_nonnegative_int(self.index_a, "index_a")
        index_b = _require_nonnegative_int(self.index_b, "index_b")
        if index_a >= index_b:
            raise ValueError("index_a must be < index_b")

        corr = _require_finite_float(self.corr, "corr")
        abs_corr = _require_finite_float(self.abs_corr, "abs_corr")
        if not (0 <= abs_corr <= 1.0000001):
            raise ValueError("abs_corr must be in [0, 1.0000001]")
        if not math.isclose(abs_corr, abs(corr), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("abs_corr must match abs(corr)")

        _require_bool(self.same_source, "same_source")
        _require_bool(self.same_family, "same_family")
        if self.status not in ALLOWED_PAIR_STATUSES:
            raise ValueError("status must be in ALLOWED_PAIR_STATUSES")


@dataclass(frozen=True, slots=True)
class FeatureClusterRecord:
    feature: str
    feature_index: int
    source: str
    family: str
    cluster_id: int
    cluster_size: int
    representative_feature: str
    max_abs_corr_in_cluster: float

    def __post_init__(self) -> None:
        _require_non_empty_str(self.feature, "feature")
        _require_non_empty_str(self.source, "source")
        _require_non_empty_str(self.family, "family")
        rep = _require_non_empty_str(self.representative_feature, "representative_feature")
        _require_nonnegative_int(self.feature_index, "feature_index")
        if isinstance(self.cluster_id, bool) or not isinstance(self.cluster_id, int) or self.cluster_id < -1:
            raise ValueError("cluster_id must be >= -1")
        size = _require_positive_int(self.cluster_size, "cluster_size")

        if self.cluster_id >= 0:
            _require_finite_float(self.max_abs_corr_in_cluster, "max_abs_corr_in_cluster")
        else:
            if not math.isnan(float(self.max_abs_corr_in_cluster)):
                raise ValueError("max_abs_corr_in_cluster must be NaN for singleton")
            if size == 1 and rep != self.feature:
                raise ValueError("singleton representative_feature must equal feature")


@dataclass(frozen=True, slots=True)
class FeatureFamilySummaryRecord:
    split: str
    family: str
    n_features: int
    low_variance_count: int
    mean_raw_std: float
    median_raw_abs_p99: float
    train_high_corr_pair_count: float
    train_max_abs_corr: float
    train_mean_abs_corr: float

    def __post_init__(self) -> None:
        split = _role_to_str(self.split)
        _require_non_empty_str(self.family, "family")
        n_features = _require_positive_int(self.n_features, "n_features")
        lv_count = _require_nonnegative_int(self.low_variance_count, "low_variance_count")
        if lv_count > n_features:
            raise ValueError("low_variance_count must be <= n_features")

        mean_raw_std = _require_finite_float(self.mean_raw_std, "mean_raw_std")
        if mean_raw_std < 0:
            raise ValueError("mean_raw_std must be >= 0")
        _require_finite_float(self.median_raw_abs_p99, "median_raw_abs_p99", allow_nan=True)

        if split == "train":
            train_pair_count = _require_finite_float(self.train_high_corr_pair_count, "train_high_corr_pair_count")
            if train_pair_count < 0:
                raise ValueError("train_high_corr_pair_count must be >= 0")
            _require_finite_float(self.train_max_abs_corr, "train_max_abs_corr", allow_nan=True)
            _require_finite_float(self.train_mean_abs_corr, "train_mean_abs_corr", allow_nan=True)
        else:
            if not math.isnan(float(self.train_high_corr_pair_count)):
                raise ValueError("train_high_corr_pair_count must be NaN outside train")
            if not math.isnan(float(self.train_max_abs_corr)):
                raise ValueError("train_max_abs_corr must be NaN outside train")
            if not math.isnan(float(self.train_mean_abs_corr)):
                raise ValueError("train_mean_abs_corr must be NaN outside train")


@dataclass(frozen=True, slots=True)
class FeatureAuditSplitSummary:
    split: str
    manifest_row_count: int
    scanned_rows: int
    sampled_rows: int
    sample_stride: int | None
    n_features: int
    low_variance_count: int
    drift_count: int

    def __post_init__(self) -> None:
        split = _role_to_str(self.split)
        manifest_row_count = _require_nonnegative_int(self.manifest_row_count, "manifest_row_count")
        scanned_rows = _require_nonnegative_int(self.scanned_rows, "scanned_rows")
        sampled_rows = _require_nonnegative_int(self.sampled_rows, "sampled_rows")
        if sampled_rows > scanned_rows:
            raise ValueError("sampled_rows must be <= scanned_rows")
        if scanned_rows > manifest_row_count:
            raise ValueError("scanned_rows must be <= manifest_row_count")

        if self.sample_stride is not None:
            _require_positive_int(self.sample_stride, "sample_stride")

        n_features = _require_positive_int(self.n_features, "n_features")
        low_variance_count = _require_nonnegative_int(self.low_variance_count, "low_variance_count")
        if low_variance_count > n_features:
            raise ValueError("low_variance_count must be <= n_features")

        drift_count = _require_nonnegative_int(self.drift_count, "drift_count")
        if drift_count > n_features:
            raise ValueError("drift_count must be <= n_features")
        if split == "train" and drift_count != 0:
            raise ValueError("train drift_count must be 0")


@dataclass(frozen=True, slots=True)
class FeatureAuditResult:
    report_type: str
    dataset_root: str
    dataset_id: str
    manifest_hash: str
    feature_schema_hash: str
    config: dict[str, object]
    splits: dict[str, FeatureAuditSplitSummary]
    health_records: tuple[FeatureHealthRecord, ...]
    drift_records: tuple[FeatureDriftRecord, ...]
    correlation_pairs: tuple[FeatureCorrelationPairRecord, ...]
    cluster_records: tuple[FeatureClusterRecord, ...]
    family_records: tuple[FeatureFamilySummaryRecord, ...]
    cluster_summary: dict[str, object]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.report_type != FEATURE_AUDIT_REPORT_TYPE:
            raise ValueError("schema mismatch")
        _require_non_empty_str(self.dataset_root, "dataset_root")
        _require_non_empty_str(self.dataset_id, "dataset_id")
        _require_non_empty_str(self.manifest_hash, "manifest_hash")
        _require_non_empty_str(self.feature_schema_hash, "feature_schema_hash")
        if not isinstance(self.config, dict):
            raise ValueError("config must be dict")
        if not isinstance(self.splits, dict):
            raise ValueError("splits must be dict")
        if "train" not in self.splits or "val" not in self.splits:
            raise ValueError("splits must include train and val")
        if not set(self.splits.keys()).issubset(set(ALLOWED_SPLITS)):
            raise ValueError("split keys must be subset of train/val/test")
        for key, value in self.splits.items():
            if not isinstance(value, FeatureAuditSplitSummary):
                raise ValueError("split values must be FeatureAuditSplitSummary")
            if key != value.split:
                raise ValueError("split key must match FeatureAuditSplitSummary.split")

        if not isinstance(self.health_records, tuple) or not all(
            isinstance(record, FeatureHealthRecord) for record in self.health_records
        ):
            raise ValueError("health_records must be tuple[FeatureHealthRecord, ...]")
        if not isinstance(self.drift_records, tuple) or not all(
            isinstance(record, FeatureDriftRecord) for record in self.drift_records
        ):
            raise ValueError("drift_records must be tuple[FeatureDriftRecord, ...]")
        if not isinstance(self.correlation_pairs, tuple) or not all(
            isinstance(record, FeatureCorrelationPairRecord) for record in self.correlation_pairs
        ):
            raise ValueError("correlation_pairs must be tuple[FeatureCorrelationPairRecord, ...]")
        if not isinstance(self.cluster_records, tuple) or not all(
            isinstance(record, FeatureClusterRecord) for record in self.cluster_records
        ):
            raise ValueError("cluster_records must be tuple[FeatureClusterRecord, ...]")
        if not isinstance(self.family_records, tuple) or not all(
            isinstance(record, FeatureFamilySummaryRecord) for record in self.family_records
        ):
            raise ValueError("family_records must be tuple[FeatureFamilySummaryRecord, ...]")
        if not isinstance(self.cluster_summary, dict):
            raise ValueError("cluster_summary must be dict")
        if not isinstance(self.warnings, tuple) or not all(isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be tuple[str, ...]")

        split_keys = set(self.splits.keys())
        for record in self.health_records:
            if record.split not in split_keys:
                raise ValueError("health record split missing from splits")
        for record in self.drift_records:
            if record.split not in split_keys:
                raise ValueError("drift record split missing from splits")
        for record in self.family_records:
            if record.split not in split_keys:
                raise ValueError("family record split missing from splits")

    def as_dict(self) -> dict[str, object]:
        return {
            "report_type": self.report_type,
            "dataset_root": self.dataset_root,
            "dataset_id": self.dataset_id,
            "manifest_hash": self.manifest_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "config": _json_safe(self.config),
            "splits": {k: asdict(v) for k, v in self.splits.items()},
            "correlation_summary": self.cluster_summary,
            "health_summary": {
                "low_variance_train_count": sum(
                    1 for r in self.health_records if r.split == "train" and r.low_variance
                ),
                "drift_val_count": sum(
                    1 for r in self.drift_records if r.split == "val" and r.status == "distribution_shift"
                ),
                "drift_test_count": sum(
                    1 for r in self.drift_records if r.split == "test" and r.status == "distribution_shift"
                ),
            },
            "warnings": list(self.warnings),
        }


def _scan_split_features(reader, manifest, role, config):
    role_s = _role_to_str(role)
    cols = _resolve_audit_feature_columns(manifest, config.feature_columns)
    metas = [_feature_meta_from_column(c) for c in cols]

    stats = _StreamingFeatureStats.empty(len(cols))
    corr_stats = _StreamingTrainCorrelationStats.empty(len(cols)) if role_s == "train" else None
    extractor = ex.IdentityFeatureExtractor(
        ex.LinearFeatureExtractorConfig(feature_columns=cols, output_dtype=config.extractor_dtype),
        manifest=manifest,
    )

    manifest_row_count = sum(entry.end_row - entry.start_row for entry in reader.split_entries(role))
    if config.max_sample_rows_per_split > 0 and manifest_row_count > 0:
        stride = max(1, math.ceil(manifest_row_count / config.max_sample_rows_per_split))
    else:
        stride = None

    sampled_rows: list[np.ndarray] = []
    row_pos = 0
    for batch in reader.iter_split_batches(role, columns=cols, batch_size=config.batch_size):
        if not isinstance(batch, pa.RecordBatch):
            raise ValueError("reader.iter_split_batches must yield pyarrow.RecordBatch")
        if batch.num_rows == 0:
            continue

        table = pa.Table.from_batches([batch])
        feature_batch = extractor.transform_table(table)
        x = feature_batch.X
        stats.update(x)
        if corr_stats is not None:
            corr_stats.update(x)

        if stride is not None:
            idx = np.arange(x.shape[0], dtype=np.int64) + row_pos
            mask = (idx % stride) == 0
            if np.any(mask):
                sampled_rows.append(x[mask].astype(np.float32, copy=True))
        row_pos += x.shape[0]

    sample = np.vstack(sampled_rows) if sampled_rows else np.empty((0, len(cols)), np.float32)

    health_records: list[FeatureHealthRecord] = []
    std = stats.std()
    for i, meta in enumerate(metas):
        if sample.shape[0] == 0:
            q = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        else:
            q = np.array(
                [
                    np.quantile(sample[:, i], 0.01),
                    np.quantile(sample[:, i], 0.5),
                    np.quantile(sample[:, i], 0.99),
                    np.quantile(np.abs(sample[:, i]), 0.99),
                ],
                dtype=np.float64,
            )

        low_variance = bool(std[i] < config.low_variance_std_threshold)
        status = "low_variance" if low_variance else "ok"
        health_records.append(
            FeatureHealthRecord(
                split=role_s,
                feature=meta.column,
                feature_index=meta.feature_index,
                source=meta.source,
                owner=meta.owner,
                family=meta.family,
                unit=meta.unit,
                transform_key=meta.transform_key,
                required_book_depth=meta.required_book_depth,
                n_rows=stats.n_rows,
                n_sample_rows=sample.shape[0],
                raw_mean=float(stats.mean[i]) if stats.n_rows else 0.0,
                raw_std=float(std[i]) if stats.n_rows else 0.0,
                raw_min=float(stats.min_value[i]) if stats.n_rows else np.nan,
                raw_max=float(stats.max_value[i]) if stats.n_rows else np.nan,
                raw_p01=float(q[0]),
                raw_p50=float(q[1]),
                raw_p99=float(q[2]),
                raw_abs_p99=float(q[3]),
                low_variance=low_variance,
                status=status,
            )
        )

    summary = FeatureAuditSplitSummary(
        split=role_s,
        manifest_row_count=manifest_row_count,
        scanned_rows=stats.n_rows,
        sampled_rows=sample.shape[0],
        sample_stride=stride,
        n_features=len(cols),
        low_variance_count=sum(r.low_variance for r in health_records),
        drift_count=0,
    )

    return summary, health_records, {
        "mean": stats.mean,
        "std": std,
        "p50": np.nanmedian(sample, axis=0) if sample.shape[0] else np.full(len(cols), np.nan),
        "n_rows": stats.n_rows,
        "feature_columns": cols,
        "corr_stats": corr_stats,
    }


def _replace_split_drift_count(summary: FeatureAuditSplitSummary, drift_count: int) -> FeatureAuditSplitSummary:
    return FeatureAuditSplitSummary(
        split=summary.split,
        manifest_row_count=summary.manifest_row_count,
        scanned_rows=summary.scanned_rows,
        sampled_rows=summary.sampled_rows,
        sample_stride=summary.sample_stride,
        n_features=summary.n_features,
        low_variance_count=summary.low_variance_count,
        drift_count=drift_count,
    )


def run_feature_audit(dataset_root: str, *, config: FeatureAuditConfig | None = None) -> FeatureAuditResult:
    dataset_root = _require_non_empty_str(dataset_root, "dataset_root")
    cfg = config or FeatureAuditConfig()

    reader = rd.open_dataset(
        dataset_root,
        validate_on_open=cfg.validate_dataset_on_open,
        batch_size=cfg.batch_size,
    )
    manifest = reader.manifest
    manifest.validate_against_current_code()

    if not reader.split_entries(SplitRole.TRAIN):
        raise ValueError("dataset must contain non-empty train split")
    if not reader.split_entries(SplitRole.VAL):
        raise ValueError("dataset must contain non-empty val split")

    splits: dict[str, FeatureAuditSplitSummary] = {}
    health: list[FeatureHealthRecord] = []
    drift: list[FeatureDriftRecord] = []
    warnings: list[str] = []
    scan: dict[str, dict[str, object]] = {}

    for role in (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST):
        if role is SplitRole.TEST and not reader.split_entries(role):
            warnings.append("missing_test_split")
            continue
        summary, records, stats = _scan_split_features(reader, manifest, role, cfg)
        splits[role.value] = summary
        health.extend(records)
        scan[role.value] = stats

    corr_stats = scan["train"]["corr_stats"]
    assert isinstance(corr_stats, _StreamingTrainCorrelationStats)
    corr = corr_stats.correlation_matrix(low_variance_std_threshold=cfg.low_variance_std_threshold)
    metas = [_feature_meta_from_column(c) for c in scan["train"]["feature_columns"]]

    pairs: list[FeatureCorrelationPairRecord] = []
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            c = corr[i, j]
            if np.isfinite(c) and abs(c) >= cfg.min_corr_output_threshold:
                pairs.append(
                    FeatureCorrelationPairRecord(
                        feature_a=metas[i].column,
                        feature_b=metas[j].column,
                        index_a=metas[i].feature_index,
                        index_b=metas[j].feature_index,
                        source_a=metas[i].source,
                        source_b=metas[j].source,
                        family_a=metas[i].family,
                        family_b=metas[j].family,
                        corr=float(c),
                        abs_corr=float(abs(c)),
                        same_source=metas[i].source == metas[j].source,
                        same_family=metas[i].family == metas[j].family,
                        status="high_redundancy" if abs(c) >= cfg.high_corr_threshold else "moderate_redundancy",
                    )
                )
    pairs.sort(key=lambda r: (-r.abs_corr, r.index_a, r.index_b))
    pairs_tuple = tuple(pairs[: cfg.max_corr_pairs])

    parent = list(range(len(metas)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= cfg.high_corr_threshold:
                union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(len(metas)):
        comps.setdefault(find(i), []).append(i)
    non_singleton = [sorted(v) for v in comps.values() if len(v) > 1]
    non_singleton.sort(key=lambda v: v[0])
    cluster_id_map = {tuple(v): k for k, v in enumerate(non_singleton)}

    clusters: list[FeatureClusterRecord] = []
    for i, meta in enumerate(metas):
        grp = sorted(comps[find(i)])
        if len(grp) == 1:
            clusters.append(
                FeatureClusterRecord(
                    feature=meta.column,
                    feature_index=meta.feature_index,
                    source=meta.source,
                    family=meta.family,
                    cluster_id=-1,
                    cluster_size=1,
                    representative_feature=meta.column,
                    max_abs_corr_in_cluster=float("nan"),
                )
            )
        else:
            rep = grp[0]
            max_corr = max(abs(corr[a, b]) for ai, a in enumerate(grp) for b in grp[ai + 1 :])
            clusters.append(
                FeatureClusterRecord(
                    feature=meta.column,
                    feature_index=meta.feature_index,
                    source=meta.source,
                    family=meta.family,
                    cluster_id=cluster_id_map[tuple(grp)],
                    cluster_size=len(grp),
                    representative_feature=metas[rep].column,
                    max_abs_corr_in_cluster=float(max_corr),
                )
            )

    for split in ("val", "test"):
        if split not in scan:
            continue
        for i, meta in enumerate(metas):
            train_std = float(scan["train"]["std"][i])
            train_mean = float(scan["train"]["mean"][i])
            split_std = float(scan[split]["std"][i])
            split_mean = float(scan[split]["mean"][i])
            train_p50 = float(scan["train"]["p50"][i])
            split_p50 = float(scan[split]["p50"][i])

            if train_std < cfg.low_variance_std_threshold:
                status = "low_variance_train"
                mean_shift = float("nan")
                std_ratio = float("nan")
                p50_shift = float("nan")
            else:
                mean_shift = (split_mean - train_mean) / train_std
                std_ratio = split_std / train_std
                p50_shift = (split_p50 - train_p50) / train_std
                status = "distribution_shift" if (
                    abs(mean_shift) >= cfg.drift_mean_z_threshold
                    or std_ratio <= cfg.drift_std_ratio_low
                    or std_ratio >= cfg.drift_std_ratio_high
                ) else "ok"

            drift.append(
                FeatureDriftRecord(
                    split=split,
                    feature=meta.column,
                    feature_index=meta.feature_index,
                    source=meta.source,
                    owner=meta.owner,
                    family=meta.family,
                    train_mean=train_mean,
                    train_std=train_std,
                    split_mean=split_mean,
                    split_std=split_std,
                    mean_shift_train_std=mean_shift,
                    std_ratio=std_ratio,
                    train_p50=train_p50,
                    split_p50=split_p50,
                    p50_shift_train_std=p50_shift,
                    status=status,
                )
            )

    for split in ("val", "test"):
        if split in splits:
            count = sum(record.split == split and record.status == "distribution_shift" for record in drift)
            splits[split] = _replace_split_drift_count(splits[split], count)

    family: list[FeatureFamilySummaryRecord] = []
    for split in splits:
        split_recs = [record for record in health if record.split == split]
        families = sorted({record.family for record in split_recs})
        for fam in families:
            recs = [record for record in split_recs if record.family == fam]
            family_idxs = [i for i, meta in enumerate(metas) if meta.family == fam]
            if split == "train":
                high_count = 0.0
                max_abs = float("nan")
                mean_abs = float("nan")

                if len(family_idxs) >= 2:
                    vals = [
                        abs(corr[a, b])
                        for ai, a in enumerate(family_idxs)
                        for b in family_idxs[ai + 1 :]
                        if np.isfinite(corr[a, b])
                    ]
                    if vals:
                        high_count = float(sum(v >= cfg.high_corr_threshold for v in vals))
                        max_abs = float(max(vals))
                        mean_abs = float(np.mean(vals))
            else:
                high_count = float("nan")
                max_abs = float("nan")
                mean_abs = float("nan")

            family.append(
                FeatureFamilySummaryRecord(
                    split=split,
                    family=fam,
                    n_features=len(recs),
                    low_variance_count=sum(r.low_variance for r in recs),
                    mean_raw_std=float(np.mean([r.raw_std for r in recs])) if recs else float("nan"),
                    median_raw_abs_p99=float(np.median([r.raw_abs_p99 for r in recs])) if recs else float("nan"),
                    train_high_corr_pair_count=high_count,
                    train_max_abs_corr=max_abs,
                    train_mean_abs_corr=mean_abs,
                )
            )

    if any(r.low_variance for r in health if r.split == "train"):
        warnings.append("low_variance_train")
    if any(r.status == "distribution_shift" for r in drift if r.split == "val"):
        warnings.append("distribution_shift:val")
    if any(r.status == "distribution_shift" for r in drift if r.split == "test"):
        warnings.append("distribution_shift:test")
    if any(r.status == "high_redundancy" for r in pairs_tuple):
        warnings.append("high_correlation_train")

    cluster_summary = {
        "pair_count": len(pairs_tuple),
        "high_redundancy_pair_count": sum(r.status == "high_redundancy" for r in pairs_tuple),
        "moderate_redundancy_pair_count": sum(r.status == "moderate_redundancy" for r in pairs_tuple),
        "cluster_count": len({r.cluster_id for r in clusters if r.cluster_id >= 0}),
        "clustered_feature_count": sum(r.cluster_id >= 0 for r in clusters),
        "max_abs_corr": float(max([r.abs_corr for r in pairs_tuple], default=float("nan"))),
    }

    return FeatureAuditResult(
        report_type=FEATURE_AUDIT_REPORT_TYPE,
        dataset_root=dataset_root,
        dataset_id=manifest.dataset_id,
        manifest_hash=manifest.content_hash(),
        feature_schema_hash=manifest.feature_schema.get("feature_specs_hash", ""),
        config={
            "batch_size": cfg.batch_size,
            "validate_dataset_on_open": cfg.validate_dataset_on_open,
            "max_sample_rows_per_split": cfg.max_sample_rows_per_split,
            "feature_columns": list(cfg.feature_columns) if cfg.feature_columns else None,
            "extractor_dtype": cfg.extractor_dtype,
            "low_variance_std_threshold": cfg.low_variance_std_threshold,
            "high_corr_threshold": cfg.high_corr_threshold,
            "min_corr_output_threshold": cfg.min_corr_output_threshold,
            "max_corr_pairs": cfg.max_corr_pairs,
            "drift_mean_z_threshold": cfg.drift_mean_z_threshold,
            "drift_std_ratio_low": cfg.drift_std_ratio_low,
            "drift_std_ratio_high": cfg.drift_std_ratio_high,
        },
        splits=splits,
        health_records=tuple(health),
        drift_records=tuple(drift),
        correlation_pairs=pairs_tuple,
        cluster_records=tuple(clusters),
        family_records=tuple(family),
        cluster_summary=cluster_summary,
        warnings=tuple(warnings),
    )


def _write_json_atomic(path: Path, payload: object) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return str(path)


def _write_csv_atomic(path: Path, records: tuple[object, ...], record_cls: type) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = list(record_cls.__dataclass_fields__.keys())
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    tmp.replace(path)
    return str(path)


def write_feature_audit_artifacts(
    result: FeatureAuditResult,
    output_dir: str,
    *,
    summary_filename: str = DEFAULT_FEATURE_AUDIT_SUMMARY_FILENAME,
    health_filename: str = DEFAULT_FEATURE_AUDIT_HEALTH_FILENAME,
    drift_filename: str = DEFAULT_FEATURE_AUDIT_DRIFT_FILENAME,
    family_filename: str = DEFAULT_FEATURE_AUDIT_FAMILY_FILENAME,
    corr_pairs_filename: str = DEFAULT_FEATURE_AUDIT_CORR_PAIRS_FILENAME,
    clusters_filename: str = DEFAULT_FEATURE_AUDIT_CLUSTERS_FILENAME,
    cluster_summary_filename: str = DEFAULT_FEATURE_AUDIT_CLUSTER_SUMMARY_FILENAME,
) -> dict[str, str]:
    if not isinstance(result, FeatureAuditResult):
        raise ValueError("result must be FeatureAuditResult")

    out = Path(_require_non_empty_str(output_dir, "output_dir"))
    out.mkdir(parents=True, exist_ok=True)

    if not _require_non_empty_str(summary_filename, "summary_filename").endswith(".json"):
        raise ValueError("summary_filename must end with .json")
    if not _require_non_empty_str(cluster_summary_filename, "cluster_summary_filename").endswith(".json"):
        raise ValueError("cluster_summary_filename must end with .json")
    if not _require_non_empty_str(health_filename, "health_filename").endswith(".csv"):
        raise ValueError("health_filename must end with .csv")
    if not _require_non_empty_str(drift_filename, "drift_filename").endswith(".csv"):
        raise ValueError("drift_filename must end with .csv")
    if not _require_non_empty_str(family_filename, "family_filename").endswith(".csv"):
        raise ValueError("family_filename must end with .csv")
    if not _require_non_empty_str(corr_pairs_filename, "corr_pairs_filename").endswith(".csv"):
        raise ValueError("corr_pairs_filename must end with .csv")
    if not _require_non_empty_str(clusters_filename, "clusters_filename").endswith(".csv"):
        raise ValueError("clusters_filename must end with .csv")

    return {
        "summary_json": _write_json_atomic(out / summary_filename, result.as_dict()),
        "health_csv": _write_csv_atomic(out / health_filename, result.health_records, FeatureHealthRecord),
        "drift_csv": _write_csv_atomic(out / drift_filename, result.drift_records, FeatureDriftRecord),
        "family_csv": _write_csv_atomic(
            out / family_filename,
            result.family_records,
            FeatureFamilySummaryRecord,
        ),
        "corr_pairs_csv": _write_csv_atomic(
            out / corr_pairs_filename,
            result.correlation_pairs,
            FeatureCorrelationPairRecord,
        ),
        "clusters_csv": _write_csv_atomic(out / clusters_filename, result.cluster_records, FeatureClusterRecord),
        "cluster_summary_json": _write_json_atomic(out / cluster_summary_filename, result.cluster_summary),
    }


__all__ = [
    "DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS",
    "DEFAULT_FEATURE_AUDIT_SUMMARY_FILENAME",
    "DEFAULT_FEATURE_AUDIT_HEALTH_FILENAME",
    "DEFAULT_FEATURE_AUDIT_DRIFT_FILENAME",
    "DEFAULT_FEATURE_AUDIT_FAMILY_FILENAME",
    "DEFAULT_FEATURE_AUDIT_CORR_PAIRS_FILENAME",
    "DEFAULT_FEATURE_AUDIT_CLUSTERS_FILENAME",
    "DEFAULT_FEATURE_AUDIT_CLUSTER_SUMMARY_FILENAME",
    "DEFAULT_DRIFT_MEAN_Z_THRESHOLD",
    "DEFAULT_DRIFT_STD_RATIO_HIGH",
    "DEFAULT_DRIFT_STD_RATIO_LOW",
    "DEFAULT_HIGH_CORR_THRESHOLD",
    "DEFAULT_LOW_VARIANCE_STD_THRESHOLD",
    "DEFAULT_MAX_CORR_PAIRS",
    "DEFAULT_MIN_CORR_OUTPUT_THRESHOLD",
    "FeatureAuditConfig",
    "FeatureHealthRecord",
    "FeatureDriftRecord",
    "FeatureFamilySummaryRecord",
    "FeatureCorrelationPairRecord",
    "FeatureClusterRecord",
    "FeatureAuditSplitSummary",
    "FeatureAuditResult",
    "run_feature_audit",
    "write_feature_audit_artifacts",
]
