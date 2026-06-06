"""Storage-backed preprocessing audit for MMRT linear models.

This module reads existing storage splits, fits the linear preprocessor on the
train split only, and audits raw/z-scored/clipped feature behavior on train,
val, and test splits. It does not parse Tardis CSV, compute market features,
build labels, create splits, train models, evaluate predictions, or mutate
storage manifests.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np
import pyarrow as pa

from mmrt.contracts import SplitRole
from mmrt.linear import extractors as ex
from mmrt.linear import preprocess as pp
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd

DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS = 100_000
DEFAULT_PREPROCESS_AUDIT_BATCH_SIZE = rd.DEFAULT_BATCH_SIZE
DEFAULT_PREPROCESS_AUDIT_SUMMARY_FILENAME = "preprocess_audit_summary.json"
DEFAULT_PREPROCESS_AUDIT_FEATURES_FILENAME = "preprocess_audit_features.csv"
PREPROCESS_AUDIT_REPORT_TYPE = "preprocess_audit"

CLIP_REVIEW_RATE = 0.001
CLIP_EXCESSIVE_RATE = 0.01
NEAR_CLIP_FRACTION = 0.80
DRIFT_MEAN_Z_REVIEW = 1.0
DRIFT_STD_RATIO_LOW = 0.5
DRIFT_STD_RATIO_HIGH = 2.0
CLIP_NOT_BINDING_ABS_Z_FRACTION = 0.50

ALLOWED_SPLITS = ("train", "val", "test")
ALLOWED_FEATURE_STATUSES = (
    "ok",
    "inactive",
    "clip_review",
    "clip_excessive",
    "drift_review",
)
ALLOWED_FEATURE_RECOMMENDATIONS = (
    "keep",
    "review_variance_floor",
    "review_clip_z",
    "review_distribution_drift",
    "review_clip_z_and_drift",
)


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
    float_value = float(value)
    if math.isnan(float_value):
        if allow_nan:
            return float_value
        raise ValueError(f"{name} must be finite")
    if not math.isfinite(float_value):
        raise ValueError(f"{name} must be finite")
    return float_value


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
class PreprocessAuditConfig:
    batch_size: int = DEFAULT_PREPROCESS_AUDIT_BATCH_SIZE
    validate_dataset_on_open: bool = True
    max_sample_rows_per_split: int = DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS
    extractor_config: ex.LinearFeatureExtractorConfig = ex.LinearFeatureExtractorConfig()
    preprocess_config: pp.LinearPreprocessConfig = pp.LinearPreprocessConfig()

    def __post_init__(self) -> None:
        _require_positive_int(self.batch_size, "batch_size")
        _require_bool(self.validate_dataset_on_open, "validate_dataset_on_open")
        _require_nonnegative_int(self.max_sample_rows_per_split, "max_sample_rows_per_split")
        if not isinstance(self.extractor_config, ex.LinearFeatureExtractorConfig):
            raise ValueError("extractor_config must be LinearFeatureExtractorConfig")
        if not isinstance(self.preprocess_config, pp.LinearPreprocessConfig):
            raise ValueError("preprocess_config must be LinearPreprocessConfig")


@dataclass(slots=True)
class _StreamingMatrixStats:
    n_rows: int
    n_features: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, n_features: int) -> "_StreamingMatrixStats":
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be a positive int")
        return cls(
            n_rows=0,
            n_features=n_features,
            mean=np.zeros(n_features, dtype=np.float64),
            m2=np.zeros(n_features, dtype=np.float64),
        )

    def update(self, X: np.ndarray) -> None:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("X must be 2D")
        if arr.shape[1] != self.n_features:
            raise ValueError("X feature count mismatch")
        if not np.isfinite(arr).all():
            raise ValueError("X must be finite")
        if arr.shape[0] == 0:
            return

        batch_n = int(arr.shape[0])
        batch_mean = np.mean(arr, axis=0)
        centered = arr - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)

        total_n = self.n_rows + batch_n
        delta = batch_mean - self.mean

        self.mean = self.mean + delta * (batch_n / total_n)
        correction = delta * delta * ((self.n_rows * batch_n) / total_n)
        self.m2 = np.maximum(self.m2 + batch_m2 + correction, 0.0)
        self.n_rows = total_n

    def variance(self) -> np.ndarray:
        if self.n_rows <= 1:
            return np.zeros(self.n_features, dtype=np.float64)
        return self.m2 / float(self.n_rows - 1)

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance(), 0.0))


@dataclass(frozen=True, slots=True)
class PreprocessFeatureRecord:
    split: str
    feature: str
    feature_index: int
    n_rows: int
    n_sample_rows: int
    active: bool
    train_mean: float
    train_variance: float
    train_scale: float
    raw_mean: float
    raw_std: float
    raw_p01: float
    raw_p50: float
    raw_p99: float
    z_pre_mean: float
    z_pre_std: float
    z_pre_p01: float
    z_pre_p50: float
    z_pre_p99: float
    z_pre_abs_p95: float
    z_pre_abs_p99: float
    z_pre_abs_max: float
    z_post_mean: float
    z_post_std: float
    clip_pos_count: int
    clip_neg_count: int
    clip_total_count: int
    clip_pos_rate: float
    clip_neg_rate: float
    clip_total_rate: float
    near_clip_rate: float
    drift_mean_z: float
    drift_std_ratio: float
    status: str
    recommendation: str

    def __post_init__(self) -> None:
        _role_to_str(self.split)
        _require_non_empty_str(self.feature, "feature")
        _require_nonnegative_int(self.feature_index, "feature_index")
        _require_nonnegative_int(self.n_rows, "n_rows")
        _require_nonnegative_int(self.n_sample_rows, "n_sample_rows")
        if self.n_rows > 0 and self.n_sample_rows > self.n_rows:
            raise ValueError("n_sample_rows must be <= n_rows when n_rows > 0")
        _require_bool(self.active, "active")

        for name in ("clip_pos_count", "clip_neg_count", "clip_total_count"):
            _require_nonnegative_int(getattr(self, name), name)
        if self.clip_total_count != self.clip_pos_count + self.clip_neg_count:
            raise ValueError("clip_total_count must equal clip_pos_count + clip_neg_count")

        quantile_fields = {
            "raw_p01",
            "raw_p50",
            "raw_p99",
            "z_pre_p01",
            "z_pre_p50",
            "z_pre_p99",
            "z_pre_abs_p95",
            "z_pre_abs_p99",
            "z_pre_abs_max",
        }
        float_fields = (
            "train_mean", "train_variance", "train_scale", "raw_mean", "raw_std", "raw_p01", "raw_p50", "raw_p99",
            "z_pre_mean", "z_pre_std", "z_pre_p01", "z_pre_p50", "z_pre_p99", "z_pre_abs_p95", "z_pre_abs_p99", "z_pre_abs_max",
            "z_post_mean", "z_post_std", "clip_pos_rate", "clip_neg_rate", "clip_total_rate", "near_clip_rate", "drift_mean_z", "drift_std_ratio",
        )
        for name in float_fields:
            allow_nan = self.n_sample_rows == 0 and name in quantile_fields
            value = _require_finite_float(getattr(self, name), name, allow_nan=allow_nan)
            if not math.isnan(value) and value < 0.0 and name in {
                "train_variance", "train_scale", "raw_std", "z_pre_std", "z_post_std", "clip_pos_rate", "clip_neg_rate", "clip_total_rate", "near_clip_rate",
            }:
                raise ValueError(f"{name} must be >= 0")

        if self.n_rows > 0:
            if not np.isclose(self.clip_pos_rate, self.clip_pos_count / self.n_rows, rtol=1e-8, atol=1e-10):
                raise ValueError("clip_pos_rate mismatch")
            if not np.isclose(self.clip_neg_rate, self.clip_neg_count / self.n_rows, rtol=1e-8, atol=1e-10):
                raise ValueError("clip_neg_rate mismatch")
            if not np.isclose(self.clip_total_rate, self.clip_total_count / self.n_rows, rtol=1e-8, atol=1e-10):
                raise ValueError("clip_total_rate mismatch")

        if self.status not in ALLOWED_FEATURE_STATUSES:
            raise ValueError(f"status must be one of {ALLOWED_FEATURE_STATUSES}")
        if self.recommendation not in ALLOWED_FEATURE_RECOMMENDATIONS:
            raise ValueError(f"recommendation must be one of {ALLOWED_FEATURE_RECOMMENDATIONS}")


@dataclass(frozen=True, slots=True)
class PreprocessSplitSummary:
    split: str
    manifest_row_count: int
    scanned_rows: int
    sampled_rows: int
    sample_stride: int | None
    n_features: int
    active_count: int
    inactive_count: int
    features_clip_review_count: int
    features_clip_excessive_count: int
    features_drift_review_count: int
    features_not_binding_count: int
    max_clip_total_rate: float
    median_clip_total_rate: float
    max_near_clip_rate: float
    max_abs_drift_mean_z: float
    min_drift_std_ratio: float
    max_drift_std_ratio: float

    def __post_init__(self) -> None:
        _role_to_str(self.split)
        _require_nonnegative_int(self.manifest_row_count, "manifest_row_count")
        _require_nonnegative_int(self.scanned_rows, "scanned_rows")
        _require_nonnegative_int(self.sampled_rows, "sampled_rows")
        if self.sampled_rows > self.scanned_rows:
            raise ValueError("sampled_rows must be <= scanned_rows")
        if self.scanned_rows > self.manifest_row_count:
            raise ValueError("scanned_rows must be <= manifest_row_count")
        if self.sample_stride is not None:
            _require_positive_int(self.sample_stride, "sample_stride")
        _require_positive_int(self.n_features, "n_features")
        _require_nonnegative_int(self.active_count, "active_count")
        _require_nonnegative_int(self.inactive_count, "inactive_count")
        if self.active_count + self.inactive_count != self.n_features:
            raise ValueError("active_count + inactive_count must equal n_features")

        for name in (
            "features_clip_review_count",
            "features_clip_excessive_count",
            "features_drift_review_count",
            "features_not_binding_count",
        ):
            value = _require_nonnegative_int(getattr(self, name), name)
            if value > self.n_features:
                raise ValueError(f"{name} must be <= n_features")

        for name in (
            "max_clip_total_rate",
            "median_clip_total_rate",
            "max_near_clip_rate",
            "max_abs_drift_mean_z",
            "min_drift_std_ratio",
            "max_drift_std_ratio",
        ):
            value = _require_finite_float(getattr(self, name), name, allow_nan=False)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class PreprocessAuditResult:
    report_type: str
    dataset_root: str
    dataset_id: str
    manifest_hash: str
    config: dict[str, object]
    preprocess_state: dict[str, object]
    splits: dict[str, PreprocessSplitSummary]
    feature_records: tuple[PreprocessFeatureRecord, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.report_type != PREPROCESS_AUDIT_REPORT_TYPE:
            raise ValueError("schema mismatch")
        _require_non_empty_str(self.dataset_root, "dataset_root")
        _require_non_empty_str(self.dataset_id, "dataset_id")
        _require_non_empty_str(self.manifest_hash, "manifest_hash")
        if not isinstance(self.config, dict):
            raise ValueError("config must be dict")
        if not isinstance(self.preprocess_state, dict):
            raise ValueError("preprocess_state must be dict")
        if not isinstance(self.splits, dict):
            raise ValueError("splits must be dict")
        for key, value in self.splits.items():
            _role_to_str(key)
            if not isinstance(value, PreprocessSplitSummary):
                raise ValueError("splits values must be PreprocessSplitSummary")
            if value.split != key:
                raise ValueError("split summary key must match split")
        if not {"train", "val"}.issubset(set(self.splits.keys())):
            raise ValueError("splits must include train and val")
        if not set(self.splits.keys()).issubset(set(ALLOWED_SPLITS)):
            raise ValueError("splits keys must be subset of train/val/test")
        if not isinstance(self.feature_records, tuple) or not all(isinstance(r, PreprocessFeatureRecord) for r in self.feature_records):
            raise ValueError("feature_records must be tuple[PreprocessFeatureRecord, ...]")
        if not isinstance(self.warnings, tuple) or not all(isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be tuple[str, ...]")
        split_keys = set(self.splits.keys())
        for record in self.feature_records:
            if record.split not in split_keys:
                raise ValueError("feature record split must exist in splits")

    def as_dict(self) -> dict[str, object]:
        state = self.preprocess_state
        active_mask = state["active_mask"]
        summary = {
            "n_rows_fit": state["n_rows_fit"],
            "n_features": len(state["feature_columns"]),
            "active_count": int(sum(active_mask)),
            "inactive_count": int(len(active_mask) - sum(active_mask)),
            "clip_z": state["config"]["clip_z"],
            "variance_floor": state["config"]["variance_floor"],
        }
        return {
            "report_type": self.report_type,
            "dataset_root": self.dataset_root,
            "dataset_id": self.dataset_id,
            "manifest_hash": self.manifest_hash,
            "config": _json_safe(self.config),
            "preprocess_state_summary": _json_safe(summary),
            "splits": {name: asdict(value) for name, value in self.splits.items()},
            "warnings": list(self.warnings),
        }


def _fit_preprocessor_from_train(
    reader: rd.StorageDatasetReader,
    manifest: mf.StorageManifest,
    config: PreprocessAuditConfig,
) -> pp.LinearPreprocessState:
    extractor = ex.IdentityFeatureExtractor(config.extractor_config, manifest=manifest)
    feature_columns = extractor.column_projection(manifest)
    preprocessor = pp.LinearPreprocessor(config.preprocess_config)

    for batch in reader.iter_split_batches(
        SplitRole.TRAIN,
        columns=feature_columns,
        batch_size=config.batch_size,
    ):
        if not isinstance(batch, pa.RecordBatch):
            raise ValueError("reader.iter_split_batches must yield pyarrow.RecordBatch")
        if batch.num_rows == 0:
            continue
        table = pa.Table.from_batches([batch])
        feature_batch = extractor.transform_table(table)
        preprocessor.partial_fit(feature_batch.X, feature_columns=feature_batch.feature_columns)

    return preprocessor.finalize()


def _audit_split(
    reader: rd.StorageDatasetReader,
    manifest: mf.StorageManifest,
    role: SplitRole,
    preprocess_state: pp.LinearPreprocessState,
    config: PreprocessAuditConfig,
) -> tuple[PreprocessSplitSummary, list[PreprocessFeatureRecord]]:
    extractor = ex.IdentityFeatureExtractor(config.extractor_config, manifest=manifest)
    feature_columns = extractor.column_projection(manifest)
    n_features = len(feature_columns)
    clip_z = preprocess_state.config.clip_z

    raw_stats = _StreamingMatrixStats.empty(n_features)
    z_pre_stats = _StreamingMatrixStats.empty(n_features)
    z_post_stats = _StreamingMatrixStats.empty(n_features)

    entries = reader.split_entries(role)
    manifest_row_count = sum(entry.end_row - entry.start_row for entry in entries)

    sample_stride: int | None = None
    if config.max_sample_rows_per_split > 0 and manifest_row_count > 0:
        sample_stride = max(1, math.ceil(manifest_row_count / config.max_sample_rows_per_split))

    clip_pos_counts = np.zeros(n_features, dtype=np.int64)
    clip_neg_counts = np.zeros(n_features, dtype=np.int64)
    near_clip_counts = np.zeros(n_features, dtype=np.int64)
    raw_samples: list[np.ndarray] = []
    z_pre_samples: list[np.ndarray] = []

    scanned_rows = 0
    row_pos = 0
    for batch in reader.iter_split_batches(role, columns=feature_columns, batch_size=config.batch_size):
        if not isinstance(batch, pa.RecordBatch):
            raise ValueError("reader.iter_split_batches must yield pyarrow.RecordBatch")
        if batch.num_rows == 0:
            continue

        table = pa.Table.from_batches([batch])
        feature_batch = extractor.transform_table(table)
        X = np.asarray(feature_batch.X, dtype=np.float64)

        Z_pre = (X - preprocess_state.mean) / preprocess_state.scale
        Z_pre[:, ~preprocess_state.active_mask] = 0.0

        Z_post = np.clip(Z_pre, -clip_z, clip_z)

        raw_stats.update(X)
        z_pre_stats.update(Z_pre)
        z_post_stats.update(Z_post)

        clip_pos_counts += np.sum(Z_pre >= clip_z, axis=0).astype(np.int64)
        clip_neg_counts += np.sum(Z_pre <= -clip_z, axis=0).astype(np.int64)
        near_clip_counts += np.sum(
            (np.abs(Z_pre) >= NEAR_CLIP_FRACTION * clip_z) & (np.abs(Z_pre) < clip_z),
            axis=0,
        ).astype(np.int64)

        if sample_stride is not None:
            idx = np.arange(X.shape[0], dtype=np.int64) + row_pos
            keep = (idx % sample_stride) == 0
            if np.any(keep):
                raw_samples.append(X[keep].astype(np.float32, copy=False))
                z_pre_samples.append(Z_pre[keep].astype(np.float32, copy=False))

        scanned_rows += int(X.shape[0])
        row_pos += int(X.shape[0])

    sampled_raw = np.vstack(raw_samples) if raw_samples else np.empty((0, n_features), dtype=np.float32)
    sampled_z_pre = np.vstack(z_pre_samples) if z_pre_samples else np.empty((0, n_features), dtype=np.float32)

    raw_std = raw_stats.std()
    z_pre_std = z_pre_stats.std()
    z_post_std = z_post_stats.std()

    def sample_quantile(values: np.ndarray, index: int, q: float) -> float:
        if values.shape[0] == 0:
            return float("nan")
        return float(np.quantile(values[:, index], q))

    feature_records: list[PreprocessFeatureRecord] = []
    for i, feature_name in enumerate(feature_columns):
        n_rows = scanned_rows
        clip_total_count = int(clip_pos_counts[i] + clip_neg_counts[i])
        clip_total_rate = float(clip_total_count / n_rows) if n_rows else 0.0
        clip_pos_rate = float(clip_pos_counts[i] / n_rows) if n_rows else 0.0
        clip_neg_rate = float(clip_neg_counts[i] / n_rows) if n_rows else 0.0
        near_clip_rate = float(near_clip_counts[i] / n_rows) if n_rows else 0.0

        drift_mean_z = float(z_pre_stats.mean[i])
        drift_std_ratio = float(z_pre_std[i])

        status = "ok"
        recommendation = "keep"
        active = bool(preprocess_state.active_mask[i])

        if not active:
            status = "inactive"
            recommendation = "review_variance_floor"
        else:
            clip_triggered = False
            if clip_total_rate >= CLIP_EXCESSIVE_RATE:
                status = "clip_excessive"
                recommendation = "review_clip_z"
                clip_triggered = True
            elif clip_total_rate >= CLIP_REVIEW_RATE:
                status = "clip_review"
                recommendation = "review_clip_z"
                clip_triggered = True

            # Z_pre is normalized with train-fitted scale, so std(Z_pre) is the
            # split raw std divided by train scale. This is the split/train std ratio.
            drift_triggered = (
                abs(drift_mean_z) >= DRIFT_MEAN_Z_REVIEW
                or drift_std_ratio <= DRIFT_STD_RATIO_LOW
                or drift_std_ratio >= DRIFT_STD_RATIO_HIGH
            )
            if drift_triggered:
                if clip_triggered:
                    recommendation = "review_clip_z_and_drift"
                else:
                    status = "drift_review"
                    recommendation = "review_distribution_drift"

        feature_records.append(
            PreprocessFeatureRecord(
                split=role.value,
                feature=feature_name,
                feature_index=i,
                n_rows=n_rows,
                n_sample_rows=int(sampled_raw.shape[0]),
                active=active,
                train_mean=float(preprocess_state.mean[i]),
                train_variance=float(preprocess_state.variance[i]),
                train_scale=float(preprocess_state.scale[i]),
                raw_mean=float(raw_stats.mean[i]),
                raw_std=float(raw_std[i]),
                raw_p01=sample_quantile(sampled_raw, i, 0.01),
                raw_p50=sample_quantile(sampled_raw, i, 0.50),
                raw_p99=sample_quantile(sampled_raw, i, 0.99),
                z_pre_mean=float(z_pre_stats.mean[i]),
                z_pre_std=float(z_pre_std[i]),
                z_pre_p01=sample_quantile(sampled_z_pre, i, 0.01),
                z_pre_p50=sample_quantile(sampled_z_pre, i, 0.50),
                z_pre_p99=sample_quantile(sampled_z_pre, i, 0.99),
                z_pre_abs_p95=float(np.quantile(np.abs(sampled_z_pre[:, i]), 0.95)) if sampled_z_pre.shape[0] > 0 else float("nan"),
                z_pre_abs_p99=float(np.quantile(np.abs(sampled_z_pre[:, i]), 0.99)) if sampled_z_pre.shape[0] > 0 else float("nan"),
                z_pre_abs_max=float(np.max(np.abs(sampled_z_pre[:, i]))) if sampled_z_pre.shape[0] > 0 else float("nan"),
                z_post_mean=float(z_post_stats.mean[i]),
                z_post_std=float(z_post_std[i]),
                clip_pos_count=int(clip_pos_counts[i]),
                clip_neg_count=int(clip_neg_counts[i]),
                clip_total_count=clip_total_count,
                clip_pos_rate=clip_pos_rate,
                clip_neg_rate=clip_neg_rate,
                clip_total_rate=clip_total_rate,
                near_clip_rate=near_clip_rate,
                drift_mean_z=drift_mean_z,
                drift_std_ratio=drift_std_ratio,
                status=status,
                recommendation=recommendation,
            )
        )

    split_summary = PreprocessSplitSummary(
        split=role.value,
        manifest_row_count=manifest_row_count,
        scanned_rows=scanned_rows,
        sampled_rows=int(sampled_raw.shape[0]),
        sample_stride=sample_stride,
        n_features=n_features,
        active_count=int(np.sum(preprocess_state.active_mask)),
        inactive_count=int(np.sum(~preprocess_state.active_mask)),
        features_clip_review_count=sum(record.status == "clip_review" for record in feature_records),
        features_clip_excessive_count=sum(record.status == "clip_excessive" for record in feature_records),
        features_drift_review_count=sum(
            record.status == "drift_review"
            or record.recommendation == "review_clip_z_and_drift"
            for record in feature_records
        ),
        features_not_binding_count=sum(
            (record.z_pre_abs_p99 < CLIP_NOT_BINDING_ABS_Z_FRACTION * clip_z)
            if np.isfinite(record.z_pre_abs_p99)
            else False
            for record in feature_records
        ),
        max_clip_total_rate=float(max(record.clip_total_rate for record in feature_records) if feature_records else 0.0),
        median_clip_total_rate=float(np.median([record.clip_total_rate for record in feature_records]) if feature_records else 0.0),
        max_near_clip_rate=float(max(record.near_clip_rate for record in feature_records) if feature_records else 0.0),
        max_abs_drift_mean_z=float(max(abs(record.drift_mean_z) for record in feature_records) if feature_records else 0.0),
        min_drift_std_ratio=float(min(record.drift_std_ratio for record in feature_records) if feature_records else 0.0),
        max_drift_std_ratio=float(max(record.drift_std_ratio for record in feature_records) if feature_records else 0.0),
    )

    return split_summary, feature_records


def run_preprocess_audit(
    dataset_root: str,
    *,
    config: PreprocessAuditConfig | None = None,
) -> PreprocessAuditResult:
    dataset_root = _require_non_empty_str(dataset_root, "dataset_root")
    cfg = config if config is not None else PreprocessAuditConfig()

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

    preprocess_state = _fit_preprocessor_from_train(reader, manifest, cfg)

    splits: dict[str, PreprocessSplitSummary] = {}
    feature_records: list[PreprocessFeatureRecord] = []
    warnings: list[str] = []

    for role in (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST):
        if role is SplitRole.TEST and not reader.split_entries(role):
            warnings.append("missing_test_split")
            continue

        split_summary, split_records = _audit_split(reader, manifest, role, preprocess_state, cfg)
        splits[role.value] = split_summary
        feature_records.extend(split_records)

        if split_summary.features_clip_excessive_count > 0:
            warnings.append(f"clip_excessive:{role.value}")
        elif split_summary.features_clip_review_count > 0:
            warnings.append(f"clip_review:{role.value}")

        if split_summary.features_drift_review_count > 0:
            warnings.append(f"drift_review:{role.value}")

    if int(np.sum(~preprocess_state.active_mask)) > 0:
        warnings.append("inactive_features_present")

    return PreprocessAuditResult(
        report_type=PREPROCESS_AUDIT_REPORT_TYPE,
        dataset_root=dataset_root,
        dataset_id=manifest.dataset_id,
        manifest_hash=manifest.content_hash(),
        config={
            "batch_size": cfg.batch_size,
            "validate_dataset_on_open": cfg.validate_dataset_on_open,
            "max_sample_rows_per_split": cfg.max_sample_rows_per_split,
            "extractor_config": {
                "feature_columns": list(cfg.extractor_config.feature_columns) if cfg.extractor_config.feature_columns else None,
                "output_dtype": cfg.extractor_config.output_dtype,
            },
            "preprocess_config": {
                "variance_floor": cfg.preprocess_config.variance_floor,
                "clip_z": cfg.preprocess_config.clip_z,
                "output_dtype": cfg.preprocess_config.output_dtype,
            },
        },
        preprocess_state=preprocess_state.as_dict(),
        splits=splits,
        feature_records=tuple(feature_records),
        warnings=tuple(warnings),
    )


def write_preprocess_audit_artifacts(
    result: PreprocessAuditResult,
    output_dir: str,
    *,
    summary_filename: str = DEFAULT_PREPROCESS_AUDIT_SUMMARY_FILENAME,
    features_filename: str = DEFAULT_PREPROCESS_AUDIT_FEATURES_FILENAME,
) -> dict[str, str]:
    if not isinstance(result, PreprocessAuditResult):
        raise ValueError("result must be PreprocessAuditResult")

    output_dir = _require_non_empty_str(output_dir, "output_dir")
    summary_filename = _require_non_empty_str(summary_filename, "summary_filename")
    features_filename = _require_non_empty_str(features_filename, "features_filename")
    if not summary_filename.endswith(".json"):
        raise ValueError("summary_filename must end with .json")
    if not features_filename.endswith(".csv"):
        raise ValueError("features_filename must end with .csv")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / summary_filename
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_tmp.write_text(
        json.dumps(result.as_dict(), sort_keys=True, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)

    features_path = out / features_filename
    features_tmp = features_path.with_suffix(features_path.suffix + ".tmp")
    fields = list(PreprocessFeatureRecord.__dataclass_fields__.keys())
    with features_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in result.feature_records:
            writer.writerow(asdict(record))
    features_tmp.replace(features_path)

    return {
        "summary_json": str(summary_path),
        "features_csv": str(features_path),
    }


__all__ = [
    "DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS",
    "DEFAULT_PREPROCESS_AUDIT_SUMMARY_FILENAME",
    "DEFAULT_PREPROCESS_AUDIT_FEATURES_FILENAME",
    "PreprocessAuditConfig",
    "PreprocessFeatureRecord",
    "PreprocessSplitSummary",
    "PreprocessAuditResult",
    "run_preprocess_audit",
    "write_preprocess_audit_artifacts",
]
