"""Storage-backed Feature Lab for externally generated MMRT candidate features.

Feature Lab evaluates candidate feature columns from a Parquet file against an
existing storage dataset and trained linear artifact. It is analysis-only: it
uses train/validation rows, joins candidates by decision_index, writes audit
artifacts, and never parses raw data, rebuilds labels, retrains models, uses the
held-out split, or mutates storage/manifests/specifications.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat

from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.linear import extractors as ex
from mmrt.linear import models as lm
from mmrt.linear import preprocess as pp
from mmrt.linear import targets as tg
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd

FEATURE_LAB_REPORT_TYPE = "feature_lab"

DEFAULT_FEATURE_LAB_BATCH_SIZE = rd.DEFAULT_BATCH_SIZE
DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN = 100_000
DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL = 100_000
DEFAULT_FEATURE_LAB_SEED = 17

DEFAULT_FEATURE_LAB_SUMMARY_FILENAME = "feature_lab_summary.json"
DEFAULT_CANDIDATE_HEALTH_FILENAME = "candidate_health.csv"
DEFAULT_CANDIDATE_EXISTING_CORR_FILENAME = "candidate_existing_correlations.csv"
DEFAULT_CANDIDATE_REDUNDANCY_FILENAME = "candidate_redundancy_summary.csv"
DEFAULT_CANDIDATE_HEAD_METRICS_FILENAME = "candidate_head_metrics.csv"
DEFAULT_CANDIDATE_RECOMMENDATIONS_FILENAME = "candidate_recommendations.csv"


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


def _require_positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return out


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"unsupported JSON type: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class FeatureLabConfig:
    batch_size: int = DEFAULT_FEATURE_LAB_BATCH_SIZE
    validate_dataset_on_open: bool = True
    max_sample_rows_train: int = DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN
    max_sample_rows_val: int = DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL
    seed: int = DEFAULT_FEATURE_LAB_SEED
    variance_floor: float = 1e-12
    z_clip: float = 8.0
    high_redundancy_threshold: float = 0.97
    moderate_redundancy_threshold: float = 0.90
    min_scope_rows: int = 25

    def __post_init__(self) -> None:
        _require_positive_int(self.batch_size, "batch_size")
        _require_bool(self.validate_dataset_on_open, "validate_dataset_on_open")
        _require_nonnegative_int(self.max_sample_rows_train, "max_sample_rows_train")
        _require_nonnegative_int(self.max_sample_rows_val, "max_sample_rows_val")
        _require_nonnegative_int(self.seed, "seed")
        object.__setattr__(self, "variance_floor", _require_positive_finite_float(self.variance_floor, "variance_floor"))
        object.__setattr__(self, "z_clip", _require_positive_finite_float(self.z_clip, "z_clip"))
        high = float(self.high_redundancy_threshold)
        mod = float(self.moderate_redundancy_threshold)
        if not (math.isfinite(mod) and math.isfinite(high) and 0.0 < mod <= high < 1.0):
            raise ValueError("0 < moderate_redundancy_threshold <= high_redundancy_threshold < 1 required")
        object.__setattr__(self, "moderate_redundancy_threshold", mod)
        object.__setattr__(self, "high_redundancy_threshold", high)
        _require_positive_int(self.min_scope_rows, "min_scope_rows")


@dataclass(frozen=True, slots=True)
class CandidateHealthRecord:
    candidate: str
    split: str
    n_storage_rows_sampled: int
    n_joined_rows: int
    n_finite_rows: int
    missing_rate: float
    finite_rate: float
    zero_rate: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    min_value: float
    max_value: float
    status: str


@dataclass(frozen=True, slots=True)
class CandidateExistingCorrelationRecord:
    candidate: str
    existing_feature: str
    existing_feature_index: int
    existing_source: str
    existing_owner: str
    existing_family: str
    existing_transform_key: str
    n_rows: int
    pearson_corr: float
    abs_pearson_corr: float


@dataclass(frozen=True, slots=True)
class CandidateRedundancyRecord:
    candidate: str
    max_abs_existing_corr: float
    most_correlated_existing_feature: str
    most_correlated_existing_family: str
    n_corr_ge_090: int
    n_corr_ge_095: int
    n_corr_ge_097: int
    n_corr_ge_099: int
    status: str


@dataclass(frozen=True, slots=True)
class CandidateHeadMetricRecord:
    candidate: str
    head: str
    scope: str
    n_train_rows: int
    n_val_rows: int
    target_metric_primary: str
    target_train_value: float
    target_val_value: float
    target_val_abs_value: float
    target_same_sign: bool
    residual_metric_primary: str
    residual_train_value: float
    residual_val_value: float
    residual_val_abs_value: float
    residual_same_sign: bool
    max_abs_existing_corr: float
    most_correlated_existing_feature: str
    missing_rate_train: float
    missing_rate_val: float
    finite_rate_train: float
    finite_rate_val: float
    zero_rate_train: float
    zero_rate_val: float
    health_status: str
    redundancy_status: str
    rank_within_head_by_residual: int


@dataclass(frozen=True, slots=True)
class CandidateRecommendationRecord:
    candidate: str
    head: str
    recommendation: str
    reason: str
    residual_rank: int
    target_rank: int
    max_abs_existing_corr: float
    health_status: str
    redundancy_status: str


@dataclass(frozen=True, slots=True)
class FeatureLabResult:
    report_type: str
    dataset_id: str
    manifest_hash: str
    train_result_path: str
    candidate_features_path: str
    n_candidates: int
    train_sample_rows: int
    val_sample_rows: int
    config: dict[str, object]
    health_records: tuple[CandidateHealthRecord, ...]
    existing_correlation_records: tuple[CandidateExistingCorrelationRecord, ...]
    redundancy_records: tuple[CandidateRedundancyRecord, ...]
    head_metric_records: tuple[CandidateHeadMetricRecord, ...]
    recommendation_records: tuple[CandidateRecommendationRecord, ...]
    summary: dict[str, object]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "report_type": self.report_type,
            "dataset_id": self.dataset_id,
            "manifest_hash": self.manifest_hash,
            "train_result_path": self.train_result_path,
            "candidate_features_path": self.candidate_features_path,
            "n_candidates": self.n_candidates,
            "train_sample_rows": self.train_sample_rows,
            "val_sample_rows": self.val_sample_rows,
            "config": _json_safe(self.config),
            "health_records": [asdict(r) for r in self.health_records],
            "existing_correlation_records": [asdict(r) for r in self.existing_correlation_records],
            "redundancy_records": [asdict(r) for r in self.redundancy_records],
            "head_metric_records": [asdict(r) for r in self.head_metric_records],
            "recommendation_records": [asdict(r) for r in self.recommendation_records],
            "summary": _json_safe(self.summary),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _CandidateData:
    candidate_names: tuple[str, ...]
    by_decision_index: dict[int, np.ndarray]


@dataclass(frozen=True, slots=True)
class _CandidateStandardizer:
    candidate_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    active: np.ndarray


def _read_json(path: str) -> dict[str, object]:
    with Path(_require_non_empty_str(path, "train_result_json")).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("train result JSON must contain an object")
    return data


def _require_artifact(artifact: dict[str, object]) -> None:
    required = ("model_bundle_state", "preprocess_state", "config", "selection_summary")
    for key in required:
        if key not in artifact:
            raise ValueError(f"train artifact missing {key}")
    cfg = artifact["config"]
    if not isinstance(cfg, dict) or "resolved_head_features" not in cfg or "target_config" not in cfg:
        raise ValueError("train artifact config missing resolved_head_features or target_config")
    sel = artifact["selection_summary"]
    if not isinstance(sel, dict) or sel.get("selection_split") != "val":
        raise ValueError("selection_summary.selection_split must be val")


def _preprocess_states(artifact: dict[str, object]) -> dict[str, pp.LinearPreprocessState]:
    state = artifact["preprocess_state"]
    if not isinstance(state, dict) or "states_by_head" not in state:
        raise ValueError("preprocess_state.states_by_head missing")
    states_dict = state["states_by_head"]
    if not isinstance(states_dict, dict) or set(states_dict.keys()) != set(lm.MODEL_HEADS):
        raise ValueError("preprocess states must exactly match MODEL_HEADS")
    return {head: pp.LinearPreprocessState.from_dict(states_dict[head]) for head in lm.MODEL_HEADS}


def _target_config(artifact: dict[str, object]) -> tg.LinearTargetConfig:
    cfg = artifact["config"]
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    tc = cfg["target_config"]
    if not isinstance(tc, dict):
        raise ValueError("target_config must be an object")
    return tg.LinearTargetConfig(
        target_horizon_us=int(tc["target_horizon_us"]),
        move_deadband_bps=float(tc["move_deadband_bps"]),
        output_dtype=str(tc["output_dtype"]),
    )


def _head_features(artifact: dict[str, object]) -> dict[str, tuple[str, ...]]:
    cfg = artifact["config"]
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    resolved = cfg["resolved_head_features"]
    if not isinstance(resolved, dict) or "feature_columns_by_head" not in resolved:
        raise ValueError("resolved_head_features.feature_columns_by_head missing")
    by_head = resolved["feature_columns_by_head"]
    if not isinstance(by_head, dict) or set(by_head.keys()) != set(lm.MODEL_HEADS):
        raise ValueError("feature_columns_by_head keys must exactly match MODEL_HEADS")
    return {head: tuple(str(c) for c in by_head[head]) for head in lm.MODEL_HEADS}


def _feature_meta(column: str) -> tuple[int, str, str, str, str]:
    if not column.startswith(mf.FEATURE_COLUMN_PREFIX):
        raise ValueError("existing feature column must use x_ prefix")
    canonical = column[len(mf.FEATURE_COLUMN_PREFIX) :]
    spec = specs.feature_spec_by_name(canonical)
    return spec.index, spec.source.value, spec.owner.value, spec.family.value, spec.transform_key.value


def _is_numeric_arrow_type(dtype: pa.DataType) -> bool:
    return (pat.is_integer(dtype) or pat.is_floating(dtype) or pat.is_decimal(dtype)) and not pat.is_boolean(dtype)


def _read_candidate_parquet(path: str) -> pa.Table:
    p = Path(_require_non_empty_str(path, "candidate_features_path"))
    if p.suffix.lower() != ".parquet":
        raise ValueError("candidate feature file must be Parquet with .parquet suffix")
    table = pq.read_table(p)
    if not isinstance(table, pa.Table):
        raise ValueError("candidate parquet did not produce a pyarrow Table")
    names = tuple(table.column_names)
    if mf.DECISION_INDEX_COLUMN not in names:
        raise ValueError("candidate parquet missing decision_index")
    candidate_names = tuple(n for n in names if n != mf.DECISION_INDEX_COLUMN)
    if not candidate_names:
        raise ValueError("candidate parquet must contain at least one candidate column")
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            raise ValueError("candidate columns must not duplicate each other")
        seen.add(name)
        if not name.startswith("c_"):
            raise ValueError("candidate columns must start with c_")
        if not _is_numeric_arrow_type(table.schema.field(name).type):
            raise ValueError("candidate columns must be numeric")
    di_type = table.schema.field(mf.DECISION_INDEX_COLUMN).type
    if not _is_numeric_arrow_type(di_type):
        raise ValueError("decision_index must be integer-valued")
    di_np = table[mf.DECISION_INDEX_COLUMN].combine_chunks().to_numpy(zero_copy_only=False)
    di_float = np.asarray(di_np, dtype=np.float64)
    if not np.isfinite(di_float).all() or np.any(di_float < 0.0) or not np.all(di_float == np.floor(di_float)):
        raise ValueError("decision_index must be finite non-negative integer-valued")
    di_int = np.asarray(di_float, dtype=np.int64)
    if np.unique(di_int).size != di_int.size:
        raise ValueError("decision_index must be unique")
    return table


def _candidate_data_from_table(table: pa.Table) -> _CandidateData:
    candidate_names = tuple(n for n in table.column_names if n != mf.DECISION_INDEX_COLUMN)
    di = np.asarray(table[mf.DECISION_INDEX_COLUMN].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
    cols = [np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64) for name in candidate_names]
    mat = np.column_stack(cols) if cols else np.empty((di.shape[0], 0), dtype=np.float64)
    return _CandidateData(candidate_names=candidate_names, by_decision_index={int(idx): mat[i].copy() for i, idx in enumerate(di)})


def _validate_candidate_collisions(table: pa.Table, manifest: mf.StorageManifest) -> None:
    storage_cols = set(manifest.required_columns) | set(mf.BASE_ROW_COLUMNS) | set(manifest.x_columns) | set(manifest.y_columns)
    collisions = [name for name in table.column_names if name != mf.DECISION_INDEX_COLUMN and name in storage_cols]
    if collisions:
        raise ValueError(f"candidate names collide with storage columns: {collisions}")


def _split_row_count(reader: rd.StorageDatasetReader, role: SplitRole) -> int:
    entries = reader.split_entries(role)
    if not entries:
        raise ValueError(f"storage dataset must include {role.value} split")
    total = 0
    for entry in entries:
        total += int(entry.row_count) if hasattr(entry, "row_count") else int(entry.end_row) - int(entry.start_row)
    return int(total)


def _sample_positions(n_rows: int, max_sample_rows: int) -> np.ndarray:
    n = _require_nonnegative_int(n_rows, "n_rows")
    max_n = _require_nonnegative_int(max_sample_rows, "max_sample_rows")
    if max_n == 0 or n == 0:
        return np.empty((0,), dtype=np.int64)
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_n, dtype=np.int64))


def _read_sampled_split_table(reader: rd.StorageDatasetReader, role: SplitRole, *, columns: tuple[str, ...], max_sample_rows: int, batch_size: int) -> pa.Table:
    n_rows = _split_row_count(reader, role)
    sample_pos = _sample_positions(n_rows, max_sample_rows)
    cursor = 0
    batch_start = 0
    pieces: list[pa.Table] = []
    empty_schema_table: pa.Table | None = None
    for batch in reader.iter_split_batches(role, columns=columns, batch_size=batch_size):
        if empty_schema_table is None:
            empty_schema_table = pa.Table.from_batches([batch]).slice(0, 0)
        batch_n = batch.num_rows
        batch_end = batch_start + batch_n
        while cursor < sample_pos.size and sample_pos[cursor] < batch_start:
            cursor += 1
        start_cursor = cursor
        while cursor < sample_pos.size and sample_pos[cursor] < batch_end:
            cursor += 1
        if cursor > start_cursor:
            local_indices = sample_pos[start_cursor:cursor] - batch_start
            pieces.append(pa.Table.from_batches([batch]).take(pa.array(local_indices, type=pa.int64())))
        batch_start = batch_end
    if pieces:
        return pa.concat_tables(pieces)
    if empty_schema_table is not None:
        return empty_schema_table
    return pa.table({column: pa.array([]) for column in columns})


def _table_matrix(table: pa.Table, columns: tuple[str, ...]) -> np.ndarray:
    if table.num_rows == 0:
        return np.empty((0, len(columns)), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack([np.asarray(table[c].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64) for c in columns]), dtype=np.float64)


def _candidate_matrix_for_sample(table: pa.Table, candidate_data: _CandidateData) -> tuple[np.ndarray, np.ndarray]:
    if mf.DECISION_INDEX_COLUMN not in table.column_names:
        raise ValueError("sample table missing decision_index")
    di = np.asarray(table[mf.DECISION_INDEX_COLUMN].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
    mat = np.full((di.shape[0], len(candidate_data.candidate_names)), np.nan, dtype=np.float64)
    joined = np.zeros(di.shape[0], dtype=bool)
    for i, idx in enumerate(di):
        values = candidate_data.by_decision_index.get(int(idx))
        if values is not None:
            mat[i, :] = values
            joined[i] = True
    return mat, joined


def _fit_candidate_standardizer(candidate_names: tuple[str, ...], train_matrix: np.ndarray, joined_mask: np.ndarray, variance_floor: float) -> _CandidateStandardizer:
    n = len(candidate_names)
    mean = np.zeros(n, dtype=np.float64)
    std = np.ones(n, dtype=np.float64)
    active = np.zeros(n, dtype=bool)
    for j in range(n):
        mask = joined_mask & np.isfinite(train_matrix[:, j])
        vals = train_matrix[mask, j]
        if vals.size:
            mean[j] = float(np.mean(vals))
        if vals.size > 1:
            std[j] = float(np.std(vals, ddof=1))
            active[j] = bool(std[j] > variance_floor)
        if not active[j]:
            std[j] = 1.0
    return _CandidateStandardizer(candidate_names, mean, std, active)


def _transform_candidates(matrix: np.ndarray, standardizer: _CandidateStandardizer, z_clip: float) -> np.ndarray:
    out = np.full(matrix.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(matrix)
    z = (matrix - standardizer.mean) / standardizer.std
    z[:, ~standardizer.active] = 0.0
    z = np.clip(z, -z_clip, z_clip)
    out[finite] = z[finite]
    return out


def _fit_existing_z(X: np.ndarray, variance_floor: float, z_clip: float) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[1] == 0:
        return X.copy(), np.zeros((0,), dtype=bool)
    Z = np.full(X.shape, np.nan, dtype=np.float64)
    active = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        finite = np.isfinite(X[:, j])
        vals = X[finite, j]
        if vals.size > 1:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            if std > variance_floor:
                active[j] = True
                Z[finite, j] = np.clip((vals - mean) / std, -z_clip, z_clip)
    return Z, active


def _rank_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < arr.shape[0]:
        j = i + 1
        while j < arr.shape[0] and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def _binary_auc(y_binary: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_binary, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    if y.shape != s.shape or y.size < 2:
        return float("nan")
    pos = y == 1.0
    neg = y == 0.0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    ranks = _rank_average(s)
    return float((float(np.sum(ranks[pos])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.size < 2:
        return float("nan")
    ca = aa - float(np.mean(aa))
    cb = bb - float(np.mean(bb))
    denom = math.sqrt(float(np.sum(ca * ca) * np.sum(cb * cb)))
    if denom == 0.0 or not math.isfinite(denom):
        return float("nan")
    out = float(np.sum(ca * cb) / denom)
    return out if math.isfinite(out) else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.size < 2:
        return float("nan")
    return _pearson(_rank_average(aa), _rank_average(bb))


def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    out = float(np.quantile(arr, q))
    return out if math.isfinite(out) else float("nan")


def _same_sign(a: float, b: float) -> bool:
    return bool(math.isfinite(float(a)) and math.isfinite(float(b)) and float(a) != 0.0 and float(b) != 0.0 and math.copysign(1.0, float(a)) == math.copysign(1.0, float(b)))


def _scope_mask(head: str, targets: tg.LinearTargetBatch) -> np.ndarray:
    if head == lm.NO_MOVE_HEAD:
        return np.ones(targets.n_rows, dtype=bool)
    if head == lm.DIRECTION_HEAD:
        return targets.move_mask
    if head == lm.MAGNITUDE_UP_HEAD:
        return targets.up_move_mask
    if head == lm.MAGNITUDE_DOWN_HEAD:
        return targets.down_move_mask
    raise ValueError("unknown head")


def _scope_name(head: str) -> str:
    return {
        lm.NO_MOVE_HEAD: "all_rows",
        lm.DIRECTION_HEAD: "move_mask",
        lm.MAGNITUDE_UP_HEAD: "up_move_mask",
        lm.MAGNITUDE_DOWN_HEAD: "down_move_mask",
    }[head]


def _target_for_metric(head: str, targets: tg.LinearTargetBatch) -> np.ndarray:
    if head == lm.NO_MOVE_HEAD:
        return np.asarray(targets.y_no_move, dtype=np.float64)
    if head == lm.DIRECTION_HEAD:
        return (targets.y_direction == tg.DIRECTION_UP_CLASS).astype(np.float64)
    if head == lm.MAGNITUDE_UP_HEAD:
        return np.asarray(targets.y_magnitude_up, dtype=np.float64)
    if head == lm.MAGNITUDE_DOWN_HEAD:
        return np.asarray(targets.y_magnitude_down, dtype=np.float64)
    raise ValueError("unknown head")


def _head_predictions(table: pa.Table, manifest: mf.StorageManifest, head: str, feature_columns: tuple[str, ...], preprocess_state: pp.LinearPreprocessState, model_head: lm.BaseLinearHead) -> np.ndarray:
    extractor = ex.IdentityFeatureExtractor(ex.LinearFeatureExtractorConfig(feature_columns=feature_columns))
    X_raw = extractor.transform_table(table, manifest=manifest).X
    Xz = pp.LinearPreprocessor.from_state(preprocess_state).transform(X_raw, feature_columns=feature_columns)
    if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD):
        return np.asarray(model_head.predict_proba(Xz)[:, 1], dtype=np.float64)
    if isinstance(model_head, lm.MagnitudeLinearHead):
        return np.asarray(model_head.predict_nonnegative(Xz), dtype=np.float64)
    raise ValueError("magnitude head object invalid")


def _residuals_for_head(head: str, targets: tg.LinearTargetBatch, pred: np.ndarray) -> np.ndarray:
    y = _target_for_metric(head, targets)
    return np.asarray(y - pred, dtype=np.float64)


def _health_records(candidate_names: tuple[str, ...], matrix: np.ndarray, joined: np.ndarray, split: str, n_rows: int, train_low_variance: np.ndarray | None, variance_floor: float) -> list[CandidateHealthRecord]:
    records: list[CandidateHealthRecord] = []
    n_joined = int(np.sum(joined))
    for j, name in enumerate(candidate_names):
        finite_mask = joined & np.isfinite(matrix[:, j])
        vals = matrix[finite_mask, j]
        n_finite = int(vals.size)
        missing_rate = float(1.0 - (n_joined / n_rows)) if n_rows else 0.0
        finite_rate = float(n_finite / n_joined) if n_joined else 0.0
        zero_rate = float(np.mean(vals == 0.0)) if n_finite else float("nan")
        std = float(np.std(vals, ddof=1)) if n_finite > 1 else float("nan")
        if missing_rate >= 0.50 or finite_rate < 0.50:
            status = "bad_health"
        elif missing_rate >= 0.05:
            status = "missing_review"
        elif finite_rate < 0.99:
            status = "nonfinite_review"
        elif split == "train" and (n_finite <= 1 or (math.isfinite(std) and std <= variance_floor)):
            status = "low_variance"
        elif split != "train" and train_low_variance is not None and bool(train_low_variance[j]):
            status = "low_variance"
        else:
            status = "ok"
        records.append(CandidateHealthRecord(
            candidate=name,
            split=split,
            n_storage_rows_sampled=int(n_rows),
            n_joined_rows=n_joined,
            n_finite_rows=n_finite,
            missing_rate=missing_rate,
            finite_rate=finite_rate,
            zero_rate=zero_rate,
            mean=float(np.mean(vals)) if n_finite else float("nan"),
            std=std,
            p01=_quantile_or_nan(vals, 0.01),
            p50=_quantile_or_nan(vals, 0.50),
            p99=_quantile_or_nan(vals, 0.99),
            min_value=float(np.min(vals)) if n_finite else float("nan"),
            max_value=float(np.max(vals)) if n_finite else float("nan"),
            status=status,
        ))
    return records


def _redundancy_records(candidate_names: tuple[str, ...], candidate_z_train: np.ndarray, train_joined: np.ndarray, existing_z: np.ndarray, existing_active: np.ndarray, existing_features: tuple[str, ...], config: FeatureLabConfig) -> tuple[tuple[CandidateExistingCorrelationRecord, ...], tuple[CandidateRedundancyRecord, ...]]:
    corr_records: list[CandidateExistingCorrelationRecord] = []
    summaries: list[CandidateRedundancyRecord] = []
    for j, cand in enumerate(candidate_names):
        abs_vals: list[tuple[float, str, str]] = []
        cand_active = bool(np.isfinite(candidate_z_train[train_joined, j]).any() and np.nanstd(candidate_z_train[train_joined, j]) > 0.0)
        for k, feat in enumerate(existing_features):
            idx, source, owner, family, transform = _feature_meta(feat)
            mask = train_joined & np.isfinite(candidate_z_train[:, j]) & np.isfinite(existing_z[:, k]) & existing_active[k] & cand_active
            n_rows = int(np.sum(mask))
            corr = _pearson(candidate_z_train[mask, j], existing_z[mask, k]) if n_rows >= 2 else float("nan")
            abs_corr = abs(float(corr)) if math.isfinite(float(corr)) else float("nan")
            if math.isfinite(abs_corr):
                abs_vals.append((abs_corr, feat, family))
            corr_records.append(CandidateExistingCorrelationRecord(cand, feat, idx, source, owner, family, transform, n_rows, float(corr), float(abs_corr)))
        if abs_vals:
            max_abs, top_feat, top_family = max(abs_vals, key=lambda x: (x[0], x[1]))
            status = "high_redundancy" if max_abs >= config.high_redundancy_threshold else "moderate_redundancy" if max_abs >= config.moderate_redundancy_threshold else "ok"
            summaries.append(CandidateRedundancyRecord(cand, float(max_abs), top_feat, top_family, sum(v >= 0.90 for v, _, _ in abs_vals), sum(v >= 0.95 for v, _, _ in abs_vals), sum(v >= 0.97 for v, _, _ in abs_vals), sum(v >= 0.99 for v, _, _ in abs_vals), status))
        else:
            summaries.append(CandidateRedundancyRecord(cand, float("nan"), "", "", 0, 0, 0, 0, "unknown"))
    return tuple(corr_records), tuple(summaries)


def _target_assoc(head: str, y: np.ndarray, z: np.ndarray) -> float:
    if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD):
        auc = _binary_auc(y, z)
        return float(auc - 0.5) if math.isfinite(auc) else float("nan")
    return _spearman(z, y)


def _metric_rows(head: str, train_targets: tg.LinearTargetBatch, val_targets: tg.LinearTargetBatch, train_resids: dict[str, np.ndarray], val_resids: dict[str, np.ndarray], train_z: np.ndarray, val_z: np.ndarray, train_joined: np.ndarray, val_joined: np.ndarray, health_by_candidate: dict[str, dict[str, CandidateHealthRecord]], redundancy_by_candidate: dict[str, CandidateRedundancyRecord], config: FeatureLabConfig, candidate_names: tuple[str, ...]) -> tuple[CandidateHeadMetricRecord, ...]:
    out: list[CandidateHeadMetricRecord] = []
    train_scope = _scope_mask(head, train_targets)
    val_scope = _scope_mask(head, val_targets)
    train_y = _target_for_metric(head, train_targets)
    val_y = _target_for_metric(head, val_targets)
    for j, cand in enumerate(candidate_names):
        train_mask = train_scope & train_joined & np.isfinite(train_z[:, j]) & np.isfinite(train_y)
        val_mask = val_scope & val_joined & np.isfinite(val_z[:, j]) & np.isfinite(val_y)
        train_resid = train_resids[head]
        val_resid = val_resids[head]
        train_resid_mask = train_mask & np.isfinite(train_resid)
        val_resid_mask = val_mask & np.isfinite(val_resid)
        target_train = _target_assoc(head, train_y[train_mask], train_z[train_mask, j]) if int(np.sum(train_mask)) >= 2 else float("nan")
        target_val = _target_assoc(head, val_y[val_mask], val_z[val_mask, j]) if int(np.sum(val_mask)) >= 2 else float("nan")
        resid_train = _pearson(train_z[train_resid_mask, j], train_resid[train_resid_mask]) if int(np.sum(train_resid_mask)) >= 2 else float("nan")
        resid_val = _pearson(val_z[val_resid_mask, j], val_resid[val_resid_mask]) if int(np.sum(val_resid_mask)) >= 2 else float("nan")
        red = redundancy_by_candidate[cand]
        ht = health_by_candidate[cand]["train"]
        hv = health_by_candidate[cand]["val"]
        out.append(CandidateHeadMetricRecord(
            candidate=cand,
            head=head,
            scope=_scope_name(head),
            n_train_rows=int(np.sum(train_resid_mask)),
            n_val_rows=int(np.sum(val_resid_mask)),
            target_metric_primary="auc_lift" if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD) else "spearman",
            target_train_value=float(target_train),
            target_val_value=float(target_val),
            target_val_abs_value=abs(float(target_val)) if math.isfinite(float(target_val)) else float("nan"),
            target_same_sign=_same_sign(target_train, target_val),
            residual_metric_primary="residual_pearson",
            residual_train_value=float(resid_train),
            residual_val_value=float(resid_val),
            residual_val_abs_value=abs(float(resid_val)) if math.isfinite(float(resid_val)) else float("nan"),
            residual_same_sign=_same_sign(resid_train, resid_val),
            max_abs_existing_corr=red.max_abs_existing_corr,
            most_correlated_existing_feature=red.most_correlated_existing_feature,
            missing_rate_train=ht.missing_rate,
            missing_rate_val=hv.missing_rate,
            finite_rate_train=ht.finite_rate,
            finite_rate_val=hv.finite_rate,
            zero_rate_train=ht.zero_rate,
            zero_rate_val=hv.zero_rate,
            health_status=ht.status,
            redundancy_status=red.status,
            rank_within_head_by_residual=0,
        ))
    return tuple(out)


def _rank_head_metrics(records: tuple[CandidateHeadMetricRecord, ...]) -> tuple[CandidateHeadMetricRecord, ...]:
    ranked: list[CandidateHeadMetricRecord] = []
    for head in lm.MODEL_HEADS:
        subset = [r for r in records if r.head == head]
        ordered = sorted(subset, key=lambda r: (1 if math.isnan(r.residual_val_abs_value) else 0, -r.residual_val_abs_value if math.isfinite(r.residual_val_abs_value) else 0.0, r.candidate))
        ranks = {id(r): i + 1 for i, r in enumerate(ordered)}
        for r in subset:
            ranked.append(CandidateHeadMetricRecord(**{**asdict(r), "rank_within_head_by_residual": ranks[id(r)]}))
    head_order = {h: i for i, h in enumerate(lm.MODEL_HEADS)}
    return tuple(sorted(ranked, key=lambda r: (head_order[r.head], r.rank_within_head_by_residual, r.candidate)))


def _recommendations(metrics: tuple[CandidateHeadMetricRecord, ...], n_candidates: int, config: FeatureLabConfig) -> tuple[CandidateRecommendationRecord, ...]:
    target_rank: dict[tuple[str, str], int] = {}
    for head in lm.MODEL_HEADS:
        subset = [r for r in metrics if r.head == head]
        ordered = sorted(subset, key=lambda r: (1 if math.isnan(r.target_val_abs_value) else 0, -r.target_val_abs_value if math.isfinite(r.target_val_abs_value) else 0.0, r.candidate))
        for i, r in enumerate(ordered):
            target_rank[(r.candidate, head)] = i + 1
    recs: list[CandidateRecommendationRecord] = []
    promising_cutoff = max(3, math.ceil(0.10 * n_candidates))
    for r in metrics:
        if r.health_status in {"bad_health", "low_variance"}:
            rec, reason = "bad_health", "bad candidate health"
        elif r.n_val_rows < config.min_scope_rows:
            rec, reason = "insufficient_rows", "insufficient finite scoped rows"
        elif r.redundancy_status == "high_redundancy":
            rec, reason = "review_redundant", "high correlation with existing feature"
        elif r.health_status in {"ok", "missing_review", "nonfinite_review"} and r.redundancy_status != "high_redundancy" and math.isfinite(r.residual_val_abs_value) and r.rank_within_head_by_residual <= promising_cutoff and r.residual_same_sign:
            rec, reason = "promising", "top residual association with stable sign"
        else:
            rec, reason = "weak_signal", "weak or unstable residual signal"
        recs.append(CandidateRecommendationRecord(r.candidate, r.head, rec, reason, r.rank_within_head_by_residual, target_rank[(r.candidate, r.head)], r.max_abs_existing_corr, r.health_status, r.redundancy_status))
    return tuple(recs)


def _summary(result_base: dict[str, object], metrics: tuple[CandidateHeadMetricRecord, ...], recs: tuple[CandidateRecommendationRecord, ...], health: tuple[CandidateHealthRecord, ...], redundancy: tuple[CandidateRedundancyRecord, ...], warnings: tuple[str, ...]) -> dict[str, object]:
    rec_by_key = {(r.candidate, r.head): r for r in recs}
    top: dict[str, object] = {}
    for head in lm.MODEL_HEADS:
        rows = sorted([m for m in metrics if m.head == head], key=lambda r: r.rank_within_head_by_residual)[:10]
        top[head] = [
            {
                "candidate": r.candidate,
                "residual_val_abs_value": r.residual_val_abs_value,
                "target_val_abs_value": r.target_val_abs_value,
                "rank_within_head_by_residual": r.rank_within_head_by_residual,
                "health_status": r.health_status,
                "redundancy_status": r.redundancy_status,
                "recommendation": rec_by_key[(r.candidate, r.head)].recommendation,
            }
            for r in rows
        ]
    health_summary: dict[str, int] = {}
    for h in health:
        key = f"{h.split}:{h.status}"
        health_summary[key] = health_summary.get(key, 0) + 1
    redundancy_summary: dict[str, int] = {}
    for r in redundancy:
        redundancy_summary[r.status] = redundancy_summary.get(r.status, 0) + 1
    return {**result_base, "top_candidates_by_head": top, "health_summary": health_summary, "redundancy_summary": redundancy_summary, "warnings": list(warnings)}


def run_feature_lab(dataset_root: str, train_result_json: str, candidate_features_path: str, *, config: FeatureLabConfig | None = None) -> FeatureLabResult:
    _require_non_empty_str(dataset_root, "dataset_root")
    _require_non_empty_str(train_result_json, "train_result_json")
    _require_non_empty_str(candidate_features_path, "candidate_features_path")
    cfg = config or FeatureLabConfig()
    if not isinstance(cfg, FeatureLabConfig):
        raise ValueError("config must be FeatureLabConfig")

    reader = rd.open_dataset(dataset_root, validate_on_open=cfg.validate_dataset_on_open, batch_size=cfg.batch_size)
    manifest = reader.manifest
    manifest.validate_against_current_code()
    _split_row_count(reader, SplitRole.TRAIN)
    _split_row_count(reader, SplitRole.VAL)

    artifact = _read_json(train_result_json)
    manifest_hash = manifest.content_hash()
    if artifact.get("dataset_id") != manifest.dataset_id:
        raise ValueError("train artifact dataset_id does not match storage manifest")
    if artifact.get("manifest_hash") != manifest_hash:
        raise ValueError("train artifact manifest_hash does not match storage manifest")
    _require_artifact(artifact)

    candidate_table = _read_candidate_parquet(candidate_features_path)
    _validate_candidate_collisions(candidate_table, manifest)
    candidate_data = _candidate_data_from_table(candidate_table)

    model_bundle = lm.load_linear_model_bundle(artifact["model_bundle_state"])
    preprocess_states = _preprocess_states(artifact)
    feature_columns_by_head = _head_features(artifact)
    target_config = _target_config(artifact)
    target_builder = tg.LinearTargetBuilder(target_config, manifest=manifest)
    target_column = target_builder.resolve_target_column(manifest)

    all_manifest_features = tuple(manifest.x_columns)
    union_head_features = tuple(dict.fromkeys(col for head in lm.MODEL_HEADS for col in feature_columns_by_head[head]))
    sample_columns = tuple(dict.fromkeys((mf.DECISION_INDEX_COLUMN, *all_manifest_features, *union_head_features, target_column)))
    train_table = _read_sampled_split_table(reader, SplitRole.TRAIN, columns=sample_columns, max_sample_rows=cfg.max_sample_rows_train, batch_size=cfg.batch_size)
    val_table = _read_sampled_split_table(reader, SplitRole.VAL, columns=sample_columns, max_sample_rows=cfg.max_sample_rows_val, batch_size=cfg.batch_size)

    train_cand_raw, train_joined = _candidate_matrix_for_sample(train_table, candidate_data)
    val_cand_raw, val_joined = _candidate_matrix_for_sample(val_table, candidate_data)
    standardizer = _fit_candidate_standardizer(candidate_data.candidate_names, train_cand_raw, train_joined, cfg.variance_floor)
    train_cand_z = _transform_candidates(train_cand_raw, standardizer, cfg.z_clip)
    val_cand_z = _transform_candidates(val_cand_raw, standardizer, cfg.z_clip)

    train_health = _health_records(candidate_data.candidate_names, train_cand_raw, train_joined, "train", train_table.num_rows, None, cfg.variance_floor)
    val_health = _health_records(candidate_data.candidate_names, val_cand_raw, val_joined, "val", val_table.num_rows, ~standardizer.active, cfg.variance_floor)
    health_records = tuple(train_health + val_health)
    health_by_candidate: dict[str, dict[str, CandidateHealthRecord]] = {name: {} for name in candidate_data.candidate_names}
    for rec in health_records:
        health_by_candidate[rec.candidate][rec.split] = rec

    X_existing = _table_matrix(train_table, all_manifest_features)
    existing_z, existing_active = _fit_existing_z(X_existing, cfg.variance_floor, cfg.z_clip)
    corr_records, redundancy_records = _redundancy_records(candidate_data.candidate_names, train_cand_z, train_joined, existing_z, existing_active, all_manifest_features, cfg)
    redundancy_by_candidate = {r.candidate: r for r in redundancy_records}

    train_targets = target_builder.transform_table(train_table, manifest=manifest)
    val_targets = target_builder.transform_table(val_table, manifest=manifest)
    train_resids: dict[str, np.ndarray] = {}
    val_resids: dict[str, np.ndarray] = {}
    for head in lm.MODEL_HEADS:
        cols = feature_columns_by_head[head]
        model_head = getattr(model_bundle, head)
        train_pred = _head_predictions(train_table.select(list(dict.fromkeys((*cols, target_column)))), manifest, head, cols, preprocess_states[head], model_head)
        val_pred = _head_predictions(val_table.select(list(dict.fromkeys((*cols, target_column)))), manifest, head, cols, preprocess_states[head], model_head)
        train_resids[head] = _residuals_for_head(head, train_targets, train_pred)
        val_resids[head] = _residuals_for_head(head, val_targets, val_pred)

    metric_records_raw: list[CandidateHeadMetricRecord] = []
    for head in lm.MODEL_HEADS:
        metric_records_raw.extend(_metric_rows(head, train_targets, val_targets, train_resids, val_resids, train_cand_z, val_cand_z, train_joined, val_joined, health_by_candidate, redundancy_by_candidate, cfg, candidate_data.candidate_names))
    metric_records = _rank_head_metrics(tuple(metric_records_raw))
    recommendation_records = _recommendations(metric_records, len(candidate_data.candidate_names), cfg)

    warnings: tuple[str, ...] = ()
    base_summary = {
        "report_type": FEATURE_LAB_REPORT_TYPE,
        "dataset_id": manifest.dataset_id,
        "manifest_hash": manifest_hash,
        "train_result_path": str(Path(train_result_json)),
        "candidate_features_path": str(Path(candidate_features_path)),
        "n_candidates": len(candidate_data.candidate_names),
        "train_sample_rows": train_table.num_rows,
        "val_sample_rows": val_table.num_rows,
        "config": asdict(cfg),
    }
    summary = _summary(base_summary, metric_records, recommendation_records, health_records, redundancy_records, warnings)
    return FeatureLabResult(
        report_type=FEATURE_LAB_REPORT_TYPE,
        dataset_id=manifest.dataset_id,
        manifest_hash=manifest_hash,
        train_result_path=str(Path(train_result_json)),
        candidate_features_path=str(Path(candidate_features_path)),
        n_candidates=len(candidate_data.candidate_names),
        train_sample_rows=train_table.num_rows,
        val_sample_rows=val_table.num_rows,
        config=asdict(cfg),
        health_records=health_records,
        existing_correlation_records=corr_records,
        redundancy_records=redundancy_records,
        head_metric_records=metric_records,
        recommendation_records=recommendation_records,
        summary=summary,
        warnings=warnings,
    )


def _write_json_atomic(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return str(path)


def _write_csv_atomic(path: Path, records: tuple[object, ...], fields: tuple[str, ...]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fields})
    tmp.replace(path)
    return str(path)


def write_feature_lab_artifacts(result: FeatureLabResult, output_dir: str) -> dict[str, str]:
    if not isinstance(result, FeatureLabResult):
        raise ValueError("result must be FeatureLabResult")
    out = Path(_require_non_empty_str(output_dir, "output_dir"))
    return {
        "summary_json": _write_json_atomic(out / DEFAULT_FEATURE_LAB_SUMMARY_FILENAME, result.summary),
        "candidate_health_csv": _write_csv_atomic(out / DEFAULT_CANDIDATE_HEALTH_FILENAME, result.health_records, ("candidate", "split", "n_storage_rows_sampled", "n_joined_rows", "n_finite_rows", "missing_rate", "finite_rate", "zero_rate", "mean", "std", "p01", "p50", "p99", "min_value", "max_value", "status")),
        "candidate_existing_correlations_csv": _write_csv_atomic(out / DEFAULT_CANDIDATE_EXISTING_CORR_FILENAME, result.existing_correlation_records, ("candidate", "existing_feature", "existing_feature_index", "existing_source", "existing_owner", "existing_family", "existing_transform_key", "n_rows", "pearson_corr", "abs_pearson_corr")),
        "candidate_redundancy_summary_csv": _write_csv_atomic(out / DEFAULT_CANDIDATE_REDUNDANCY_FILENAME, result.redundancy_records, ("candidate", "max_abs_existing_corr", "most_correlated_existing_feature", "most_correlated_existing_family", "n_corr_ge_090", "n_corr_ge_095", "n_corr_ge_097", "n_corr_ge_099", "status")),
        "candidate_head_metrics_csv": _write_csv_atomic(out / DEFAULT_CANDIDATE_HEAD_METRICS_FILENAME, result.head_metric_records, ("candidate", "head", "scope", "n_train_rows", "n_val_rows", "target_metric_primary", "target_train_value", "target_val_value", "target_val_abs_value", "target_same_sign", "residual_metric_primary", "residual_train_value", "residual_val_value", "residual_val_abs_value", "residual_same_sign", "max_abs_existing_corr", "most_correlated_existing_feature", "missing_rate_train", "missing_rate_val", "finite_rate_train", "finite_rate_val", "zero_rate_train", "zero_rate_val", "health_status", "redundancy_status", "rank_within_head_by_residual")),
        "candidate_recommendations_csv": _write_csv_atomic(out / DEFAULT_CANDIDATE_RECOMMENDATIONS_FILENAME, result.recommendation_records, ("candidate", "head", "recommendation", "reason", "residual_rank", "target_rank", "max_abs_existing_corr", "health_status", "redundancy_status")),
    }


__all__ = [
    "FEATURE_LAB_REPORT_TYPE",
    "DEFAULT_FEATURE_LAB_BATCH_SIZE",
    "DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN",
    "DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL",
    "DEFAULT_FEATURE_LAB_SEED",
    "DEFAULT_FEATURE_LAB_SUMMARY_FILENAME",
    "DEFAULT_CANDIDATE_HEALTH_FILENAME",
    "DEFAULT_CANDIDATE_EXISTING_CORR_FILENAME",
    "DEFAULT_CANDIDATE_REDUNDANCY_FILENAME",
    "DEFAULT_CANDIDATE_HEAD_METRICS_FILENAME",
    "DEFAULT_CANDIDATE_RECOMMENDATIONS_FILENAME",
    "FeatureLabConfig",
    "CandidateHealthRecord",
    "CandidateExistingCorrelationRecord",
    "CandidateRedundancyRecord",
    "CandidateHeadMetricRecord",
    "CandidateRecommendationRecord",
    "FeatureLabResult",
    "run_feature_lab",
    "write_feature_lab_artifacts",
]
