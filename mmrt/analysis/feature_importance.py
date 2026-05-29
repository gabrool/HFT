"""Storage-backed feature importance for trained MMRT linear heads.

This module ranks already-materialized storage feature columns for an existing
trained linear artifact. It reads finalized storage rows, model artifacts, and
split definitions only. It does not parse raw Tardis CSV, compute market
features, build storage labels, create splits, fit preprocessing, train models,
select feature subsets, or mutate storage/manifests.
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
from mmrt.linear import models as lm
from mmrt.linear import preprocess as pp
from mmrt.linear import targets as tg
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd

FEATURE_IMPORTANCE_SCHEMA_VERSION = 1

DEFAULT_FEATURE_IMPORTANCE_BATCH_SIZE = rd.DEFAULT_BATCH_SIZE
DEFAULT_FEATURE_IMPORTANCE_MAX_SAMPLE_ROWS = 100_000
DEFAULT_FEATURE_IMPORTANCE_SEED = 17

DEFAULT_FEATURE_IMPORTANCE_SUMMARY_FILENAME = "feature_importance_summary.json"
DEFAULT_FEATURE_IMPORTANCE_BY_HEAD_FILENAME = "feature_importance_by_head.csv"
DEFAULT_FEATURE_IMPORTANCE_FAMILY_SUMMARY_FILENAME = "feature_importance_family_summary.csv"


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
class FeatureImportanceConfig:
    batch_size: int = DEFAULT_FEATURE_IMPORTANCE_BATCH_SIZE
    validate_dataset_on_open: bool = True
    max_sample_rows: int = DEFAULT_FEATURE_IMPORTANCE_MAX_SAMPLE_ROWS
    seed: int = DEFAULT_FEATURE_IMPORTANCE_SEED

    def __post_init__(self) -> None:
        _require_positive_int(self.batch_size, "batch_size")
        _require_bool(self.validate_dataset_on_open, "validate_dataset_on_open")
        _require_nonnegative_int(self.max_sample_rows, "max_sample_rows")
        _require_nonnegative_int(self.seed, "seed")


@dataclass(frozen=True, slots=True)
class FeatureImportanceRecord:
    head: str
    feature: str
    feature_index: int
    source: str
    owner: str
    family: str
    unit: str
    transform_key: str
    required_book_depth: int
    n_eval_rows: int
    primary_metric: str
    primary_mode: str
    base_primary: float
    permuted_primary: float
    primary_importance: float
    guardrail_metric: str
    base_guardrail: float
    permuted_guardrail: float
    guardrail_delta: float
    coefficient: float
    abs_coefficient: float
    coefficient_rank: int
    importance_rank: int


@dataclass(frozen=True, slots=True)
class FeatureImportanceFamilyRecord:
    head: str
    family: str
    source: str
    owner: str
    n_features: int
    sum_positive_importance: float
    mean_positive_importance: float
    max_importance: float
    top_feature: str


@dataclass(frozen=True, slots=True)
class FeatureImportanceResult:
    schema_version: int
    dataset_id: str
    manifest_hash: str
    train_result_path: str
    selection_split: str
    n_sample_rows: int
    seed: int
    records: tuple[FeatureImportanceRecord, ...]
    family_records: tuple[FeatureImportanceFamilyRecord, ...]
    summary: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "manifest_hash": self.manifest_hash,
            "train_result_path": self.train_result_path,
            "selection_split": self.selection_split,
            "n_sample_rows": self.n_sample_rows,
            "seed": self.seed,
            "records": [asdict(r) for r in self.records],
            "family_records": [asdict(r) for r in self.family_records],
            "summary": _json_safe(self.summary),
        }


@dataclass(frozen=True, slots=True)
class _FeatureMeta:
    column: str
    canonical_name: str
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
    canonical_name = col[len(mf.FEATURE_COLUMN_PREFIX) :]
    spec = specs.feature_spec_by_name(canonical_name)
    return _FeatureMeta(
        col,
        canonical_name,
        spec.index,
        spec.source.value,
        spec.owner.value,
        spec.family.value,
        spec.unit.value,
        spec.transform_key.value,
        spec.required_book_depth,
    )


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_z = np.exp(arr[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    i = 0
    while i < arr.shape[0]:
        j = i + 1
        while j < arr.shape[0] and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    ppred = np.asarray(p, dtype=np.float64)
    if yy.shape != ppred.shape:
        raise ValueError("y and p must have matching shapes")
    pos = yy == 1.0
    neg = yy == 0.0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    ranks = _rankdata_average(ppred)
    sum_pos = float(np.sum(ranks[pos]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    ppred = np.asarray(p, dtype=np.float64)
    if yy.size == 0:
        return float("nan")
    return float(np.mean((ppred - yy) ** 2))


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    ppred = np.asarray(pred, dtype=np.float64)
    if yy.size == 0:
        return float("nan")
    return float(np.mean(np.abs(ppred - yy)))


def _spearman(y: np.ndarray, pred: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    ppred = np.asarray(pred, dtype=np.float64)
    if yy.size < 2:
        return float("nan")
    ry = _rankdata_average(yy)
    rp = _rankdata_average(ppred)
    cy = ry - np.mean(ry)
    cp = rp - np.mean(rp)
    denom = math.sqrt(float(np.sum(cy * cy) * np.sum(cp * cp)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(cy * cp) / denom)


def _selection_metric(head: str) -> tuple[str, str, str, str]:
    if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD):
        return "auc", "max", "brier", "classification"
    return "mae", "min", "spearman", "regression"


def _scope_mask(head: str, targets: tg.LinearTargetBatch) -> np.ndarray:
    if head == lm.NO_MOVE_HEAD:
        return np.ones(targets.y_return_bps.shape[0], dtype=bool)
    if head == lm.DIRECTION_HEAD:
        return targets.move_mask
    if head == lm.MAGNITUDE_UP_HEAD:
        return targets.up_move_mask
    if head == lm.MAGNITUDE_DOWN_HEAD:
        return targets.down_move_mask
    raise ValueError("unknown head")


def _target_for_head(head: str, targets: tg.LinearTargetBatch) -> np.ndarray:
    if head == lm.NO_MOVE_HEAD:
        return targets.y_no_move
    if head == lm.DIRECTION_HEAD:
        return targets.y_direction
    if head == lm.MAGNITUDE_UP_HEAD:
        return targets.y_magnitude_up
    if head == lm.MAGNITUDE_DOWN_HEAD:
        return targets.y_magnitude_down
    raise ValueError("unknown head")


def _metric_values(head: str, y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD):
        pred = _stable_sigmoid(score)
        return _binary_auc(y, pred), _brier(y, pred)
    pred = np.maximum(score, 0.0)
    return _mae(y, pred), _spearman(y, pred)


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


def _split_row_count(reader: rd.StorageDatasetReader, role: SplitRole) -> int:
    entries = reader.split_entries(role)
    if not entries:
        raise ValueError("storage dataset must include validation split")
    total = 0
    for entry in entries:
        if hasattr(entry, "row_count"):
            total += int(entry.row_count)
        else:
            total += int(entry.end_row) - int(entry.start_row)
    return int(total)


def _sample_positions(n_rows: int, max_sample_rows: int) -> np.ndarray:
    n = _require_nonnegative_int(n_rows, "n_rows")
    max_n = _require_nonnegative_int(max_sample_rows, "max_sample_rows")
    if max_n == 0 or n == 0:
        return np.empty((0,), dtype=np.int64)
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, max_n, dtype=np.int64))


def _read_sampled_split_table(
    reader: rd.StorageDatasetReader,
    role: SplitRole,
    *,
    columns: tuple[str, ...],
    max_sample_rows: int,
    batch_size: int,
) -> pa.Table:
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
            piece = pa.Table.from_batches([batch]).take(pa.array(local_indices, type=pa.int64()))
            pieces.append(piece)
        batch_start = batch_end

    if pieces:
        return pa.concat_tables(pieces)
    if empty_schema_table is not None:
        return empty_schema_table
    return pa.table({column: pa.array([]) for column in columns})


def _coefficient_ranks(weights: np.ndarray) -> dict[int, int]:
    order = sorted(range(weights.shape[0]), key=lambda i: (-abs(float(weights[i])), i))
    return {idx: rank + 1 for rank, idx in enumerate(order)}


def _importance_ranks(importances: list[float]) -> dict[int, int]:
    def key(i: int) -> tuple[int, float, int]:
        v = importances[i]
        if math.isnan(v):
            return (1, 0.0, i)
        return (0, -v, i)
    order = sorted(range(len(importances)), key=key)
    return {idx: rank + 1 for rank, idx in enumerate(order)}


def _head_object(bundle: lm.LinearModelBundle, head: str) -> lm.BaseLinearHead:
    obj = getattr(bundle, head)
    return obj


def _records_for_head(
    *,
    head: str,
    head_index: int,
    table: pa.Table,
    manifest: mf.StorageManifest,
    feature_columns: tuple[str, ...],
    preprocess_state: pp.LinearPreprocessState,
    model_head: lm.BaseLinearHead,
    target_builder: tg.LinearTargetBuilder,
    seed: int,
) -> tuple[list[FeatureImportanceRecord], dict[str, object]]:
    extractor = ex.IdentityFeatureExtractor(ex.LinearFeatureExtractorConfig(feature_columns=feature_columns))
    X_raw = extractor.transform_table(table, manifest=manifest).X
    Xz = pp.transform_with_state(X_raw, preprocess_state, feature_columns=feature_columns)
    targets = target_builder.transform_table(table, manifest=manifest)
    scope = _scope_mask(head, targets)
    y_all = _target_for_head(head, targets)
    y = y_all[scope]
    n_eval_rows = int(np.sum(scope))
    weights = np.asarray(model_head.weights, dtype=np.float64)
    score = np.ascontiguousarray(Xz @ weights + float(model_head.intercept), dtype=np.float64)
    metric, mode, guardrail, _kind = _selection_metric(head)
    base_primary, base_guardrail = _metric_values(head, y, score[scope])
    perm = np.random.default_rng(seed + head_index).permutation(Xz.shape[0]) if Xz.shape[0] else np.empty((0,), dtype=np.int64)

    coef_ranks = _coefficient_ranks(weights)
    prelim: list[tuple[FeatureImportanceRecord, float]] = []
    importances: list[float] = []
    for j, feature in enumerate(feature_columns):
        permuted_score = score + weights[j] * (Xz[perm, j] - Xz[:, j]) if Xz.shape[0] else score.copy()
        perm_primary, perm_guardrail = _metric_values(head, y, permuted_score[scope])
        if mode == "max":
            primary_importance = base_primary - perm_primary
        else:
            primary_importance = perm_primary - base_primary
        if guardrail == "brier":
            guardrail_delta = perm_guardrail - base_guardrail
        else:
            guardrail_delta = base_guardrail - perm_guardrail
        meta = _feature_meta_from_column(feature)
        rec = FeatureImportanceRecord(
            head=head,
            feature=feature,
            feature_index=meta.feature_index,
            source=meta.source,
            owner=meta.owner,
            family=meta.family,
            unit=meta.unit,
            transform_key=meta.transform_key,
            required_book_depth=meta.required_book_depth,
            n_eval_rows=n_eval_rows,
            primary_metric=metric,
            primary_mode=mode,
            base_primary=float(base_primary),
            permuted_primary=float(perm_primary),
            primary_importance=float(primary_importance),
            guardrail_metric=guardrail,
            base_guardrail=float(base_guardrail),
            permuted_guardrail=float(perm_guardrail),
            guardrail_delta=float(guardrail_delta),
            coefficient=float(weights[j]),
            abs_coefficient=abs(float(weights[j])),
            coefficient_rank=coef_ranks[j],
            importance_rank=0,
        )
        prelim.append((rec, float(primary_importance)))
        importances.append(float(primary_importance))

    imp_ranks = _importance_ranks(importances)
    records = [FeatureImportanceRecord(**{**asdict(rec), "importance_rank": imp_ranks[i]}) for i, (rec, _) in enumerate(prelim)]
    head_summary = {
        "n_features": len(feature_columns),
        "n_eval_rows": n_eval_rows,
        "primary_metric": metric,
        "primary_mode": mode,
        "base_primary": float(base_primary),
        "top_features": [
            {
                "feature": r.feature,
                "primary_importance": r.primary_importance,
                "coefficient": r.coefficient,
                "abs_coefficient": r.abs_coefficient,
                "importance_rank": r.importance_rank,
                "coefficient_rank": r.coefficient_rank,
                "family": r.family,
            }
            for r in sorted(records, key=lambda x: (x.importance_rank, x.feature_index))[:10]
        ],
    }
    return records, head_summary


def _family_records(records: tuple[FeatureImportanceRecord, ...]) -> tuple[FeatureImportanceFamilyRecord, ...]:
    groups: dict[tuple[str, str, str, str], list[FeatureImportanceRecord]] = {}
    for record in records:
        groups.setdefault((record.head, record.family, record.source, record.owner), []).append(record)
    out: list[FeatureImportanceFamilyRecord] = []
    for (head, family, source, owner), recs in groups.items():
        positives = [max(r.primary_importance, 0.0) if not math.isnan(r.primary_importance) else 0.0 for r in recs]
        top = sorted(recs, key=lambda r: (math.isnan(r.primary_importance), -r.primary_importance, r.feature_index))[0]
        max_imp = max((r.primary_importance for r in recs if not math.isnan(r.primary_importance)), default=float("nan"))
        out.append(
            FeatureImportanceFamilyRecord(
                head=head,
                family=family,
                source=source,
                owner=owner,
                n_features=len(recs),
                sum_positive_importance=float(sum(positives)),
                mean_positive_importance=float(sum(positives) / len(positives)) if positives else float("nan"),
                max_importance=float(max_imp),
                top_feature=top.feature,
            )
        )
    head_order = {h: i for i, h in enumerate(lm.MODEL_HEADS)}
    return tuple(sorted(out, key=lambda r: (head_order[r.head], -r.sum_positive_importance, r.family)))


def run_feature_importance(
    dataset_root: str,
    train_result_json: str,
    *,
    config: FeatureImportanceConfig | None = None,
) -> FeatureImportanceResult:
    cfg = config or FeatureImportanceConfig()
    reader = rd.open_dataset(
        dataset_root,
        validate_on_open=cfg.validate_dataset_on_open,
        batch_size=cfg.batch_size,
    )
    manifest = reader.manifest
    manifest.validate_against_current_code()
    artifact = _read_json(train_result_json)
    _require_artifact(artifact)
    if artifact.get("dataset_id") != manifest.dataset_id:
        raise ValueError("train artifact dataset_id does not match storage manifest")
    manifest_hash = manifest.content_hash()
    if artifact.get("manifest_hash") != manifest_hash:
        raise ValueError("train artifact manifest_hash does not match storage manifest")
    if not reader.split_entries(SplitRole.VAL):
        raise ValueError("storage dataset must include validation split")

    model_bundle = lm.load_linear_model_bundle(artifact["model_bundle_state"])
    preprocess_states = _preprocess_states(artifact)
    feature_columns_by_head = _head_features(artifact)
    target_config = _target_config(artifact)
    target_builder = tg.LinearTargetBuilder(target_config, manifest=manifest)
    target_column = target_builder.resolve_target_column(manifest)

    all_feature_columns = tuple(
        dict.fromkeys(
            col
            for head in lm.MODEL_HEADS
            for col in feature_columns_by_head[head]
        )
    )
    sample_columns = tuple(dict.fromkeys((*all_feature_columns, target_column)))
    sample_table = _read_sampled_split_table(
        reader,
        SplitRole.VAL,
        columns=sample_columns,
        max_sample_rows=cfg.max_sample_rows,
        batch_size=cfg.batch_size,
    )
    n_sample_rows = sample_table.num_rows

    records: list[FeatureImportanceRecord] = []
    heads_summary: dict[str, object] = {}
    for head_index, head in enumerate(lm.MODEL_HEADS):
        feature_columns = feature_columns_by_head[head]
        head_columns = tuple(dict.fromkeys((*feature_columns, target_column)))
        head_table = sample_table.select(list(head_columns))
        head_records, head_summary = _records_for_head(
            head=head,
            head_index=head_index,
            table=head_table,
            manifest=manifest,
            feature_columns=feature_columns,
            preprocess_state=preprocess_states[head],
            model_head=_head_object(model_bundle, head),
            target_builder=target_builder,
            seed=cfg.seed,
        )
        records.extend(head_records)
        heads_summary[head] = head_summary

    head_order = {h: i for i, h in enumerate(lm.MODEL_HEADS)}
    records_tuple = tuple(sorted(records, key=lambda r: (head_order[r.head], r.importance_rank, r.feature_index)))
    family_tuple = _family_records(records_tuple)
    summary: dict[str, object] = {
        "schema_version": FEATURE_IMPORTANCE_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "manifest_hash": manifest_hash,
        "train_result_path": str(Path(train_result_json)),
        "selection_split": "val",
        "n_sample_rows": n_sample_rows,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "heads": heads_summary,
    }
    return FeatureImportanceResult(
        schema_version=FEATURE_IMPORTANCE_SCHEMA_VERSION,
        dataset_id=manifest.dataset_id,
        manifest_hash=manifest_hash,
        train_result_path=str(Path(train_result_json)),
        selection_split="val",
        n_sample_rows=n_sample_rows,
        seed=cfg.seed,
        records=records_tuple,
        family_records=family_tuple,
        summary=summary,
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


def write_feature_importance_artifacts(
    result: FeatureImportanceResult,
    output_dir: str,
) -> dict[str, str]:
    if not isinstance(result, FeatureImportanceResult):
        raise ValueError("result must be FeatureImportanceResult")
    out = Path(_require_non_empty_str(output_dir, "output_dir"))
    by_head_fields = (
        "head",
        "feature",
        "feature_index",
        "source",
        "owner",
        "family",
        "unit",
        "transform_key",
        "required_book_depth",
        "n_eval_rows",
        "primary_metric",
        "primary_mode",
        "base_primary",
        "permuted_primary",
        "primary_importance",
        "guardrail_metric",
        "base_guardrail",
        "permuted_guardrail",
        "guardrail_delta",
        "coefficient",
        "abs_coefficient",
        "coefficient_rank",
        "importance_rank",
    )
    family_fields = (
        "head",
        "family",
        "source",
        "owner",
        "n_features",
        "sum_positive_importance",
        "mean_positive_importance",
        "max_importance",
        "top_feature",
    )
    head_order = {h: i for i, h in enumerate(lm.MODEL_HEADS)}
    records = tuple(sorted(result.records, key=lambda r: (head_order[r.head], r.importance_rank, r.feature_index)))
    families = tuple(sorted(result.family_records, key=lambda r: (head_order[r.head], -r.sum_positive_importance, r.family)))
    return {
        "summary_json": _write_json_atomic(out / DEFAULT_FEATURE_IMPORTANCE_SUMMARY_FILENAME, result.summary),
        "by_head_csv": _write_csv_atomic(out / DEFAULT_FEATURE_IMPORTANCE_BY_HEAD_FILENAME, records, by_head_fields),
        "family_summary_csv": _write_csv_atomic(
            out / DEFAULT_FEATURE_IMPORTANCE_FAMILY_SUMMARY_FILENAME,
            families,
            family_fields,
        ),
    }


__all__ = [
    "FEATURE_IMPORTANCE_SCHEMA_VERSION",
    "DEFAULT_FEATURE_IMPORTANCE_BATCH_SIZE",
    "DEFAULT_FEATURE_IMPORTANCE_MAX_SAMPLE_ROWS",
    "DEFAULT_FEATURE_IMPORTANCE_SEED",
    "DEFAULT_FEATURE_IMPORTANCE_SUMMARY_FILENAME",
    "DEFAULT_FEATURE_IMPORTANCE_BY_HEAD_FILENAME",
    "DEFAULT_FEATURE_IMPORTANCE_FAMILY_SUMMARY_FILENAME",
    "FeatureImportanceConfig",
    "FeatureImportanceRecord",
    "FeatureImportanceFamilyRecord",
    "FeatureImportanceResult",
    "run_feature_importance",
    "write_feature_importance_artifacts",
]
