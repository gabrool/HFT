"""Training orchestration for storage-backed MMRT linear models.

This module wires storage split readers to the frozen linear extractor,
target, preprocess, model, evaluation, and diagnostics layers. It does not
build market-data features or labels, create splits, inspect row timing
fields, run hyperparameter search, or modify dataset manifests.
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

import numpy as np
import pyarrow as pa

from mmrt.contracts import SplitRole
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.linear import extractors as ex
from mmrt.linear import head_features as hf
from mmrt.linear import targets as tg
from mmrt.linear import preprocess as pp
from mmrt.linear import models as lm
from mmrt.linear import evaluate as ev
from mmrt.linear import diagnostics as dg

DEFAULT_TRAIN_BATCH_SIZE = 8192
DEFAULT_EPOCHS = 5
DEFAULT_OUTPUT_FILENAME = "linear_train_result.json"
LINEAR_TRAINING_RESULT_SCHEMA = "mmrt_linear_training_result"


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
        raise ValueError(f"{name} must be bool")
    return value


def _require_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_safe(obj: object, name: str = "object") -> object:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist(), name=name)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, name=f"{name}[]") for x in obj]
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"{name} contains non-string dict key")
            out[k] = _json_safe(v, name=f"{name}.{k}")
        return out
    raise ValueError(f"{name} is not JSON-safe")


def _concat_1d(parts: list[np.ndarray], *, dtype: np.dtype, name: str) -> np.ndarray:
    if not parts:
        return np.empty((0,), dtype=dtype)
    normalized: list[np.ndarray] = []
    for part in parts:
        arr = np.asarray(part)
        if arr.ndim != 1:
            raise ValueError(f"{name} parts must be 1D")
        normalized.append(np.asarray(arr, dtype=dtype))
    return np.ascontiguousarray(np.concatenate(normalized, axis=0), dtype=dtype)


def _require_manifest_has_split_roles(manifest: mf.StorageManifest) -> tuple[SplitRole, ...]:
    role_set = {SplitRole(sp.role) for sp in manifest.splits}
    if SplitRole.TRAIN not in role_set:
        raise ValueError("manifest splits must include train")
    if SplitRole.VAL not in role_set:
        raise ValueError("manifest splits must include val")
    ordered: list[SplitRole] = []
    for role in (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST):
        if role in role_set:
            ordered.append(role)
    return tuple(ordered)


def _role_to_str(role: SplitRole | str) -> str:
    return SplitRole(role).value


@dataclass(frozen=True, slots=True)
class LinearTrainConfig:
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    validate_dataset_on_open: bool = True
    extractor_config: ex.LinearFeatureExtractorConfig = ex.LinearFeatureExtractorConfig()
    head_feature_config: hf.HeadFeatureConfig = hf.HeadFeatureConfig()
    target_config: tg.LinearTargetConfig = tg.LinearTargetConfig()
    preprocess_config: pp.LinearPreprocessConfig = pp.LinearPreprocessConfig()
    model_config: lm.LinearModelConfig = lm.LinearModelConfig()
    diagnostics_config: dg.DiagnosticsConfig = dg.DiagnosticsConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", _require_positive_int(self.batch_size, "batch_size"))
        object.__setattr__(self, "epochs", _require_positive_int(self.epochs, "epochs"))
        object.__setattr__(
            self,
            "validate_dataset_on_open",
            _require_bool(self.validate_dataset_on_open, "validate_dataset_on_open"),
        )
        if not isinstance(self.extractor_config, ex.LinearFeatureExtractorConfig):
            raise ValueError("extractor_config must be LinearFeatureExtractorConfig")
        if self.extractor_config.feature_columns is not None:
            raise ValueError("LinearTrainConfig requires head_feature_config for feature selection")
        if not isinstance(self.head_feature_config, hf.HeadFeatureConfig):
            raise ValueError("head_feature_config must be HeadFeatureConfig")
        if not isinstance(self.target_config, tg.LinearTargetConfig):
            raise ValueError("target_config must be LinearTargetConfig")
        if not isinstance(self.preprocess_config, pp.LinearPreprocessConfig):
            raise ValueError("preprocess_config must be LinearPreprocessConfig")
        if not isinstance(self.model_config, lm.LinearModelConfig):
            raise ValueError("model_config must be LinearModelConfig")
        if not isinstance(self.diagnostics_config, dg.DiagnosticsConfig):
            raise ValueError("diagnostics_config must be DiagnosticsConfig")

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validate_dataset_on_open": self.validate_dataset_on_open,
            "extractor_config": {
                "feature_columns": list(self.extractor_config.feature_columns) if self.extractor_config.feature_columns is not None else None,
                "output_dtype": self.extractor_config.output_dtype,
            },
            "head_feature_config": self.head_feature_config.as_dict(),
            "target_config": {
                "target_horizon_us": self.target_config.target_horizon_us,
                "move_deadband_bps": self.target_config.move_deadband_bps,
                "output_dtype": self.target_config.output_dtype,
            },
            "preprocess_config": {
                "variance_floor": self.preprocess_config.variance_floor,
                "clip_z": self.preprocess_config.clip_z,
                "output_dtype": self.preprocess_config.output_dtype,
            },
            "model_config": {
                "learning_rate": self.model_config.learning_rate,
                "l2": self.model_config.l2,
                "max_grad_norm": self.model_config.max_grad_norm,
                "output_dtype": self.model_config.output_dtype,
                "magnitude_huber_delta": self.model_config.magnitude_huber_delta,
            },
            "diagnostics_config": {
                "top_k": self.diagnostics_config.top_k,
                "num_bins": self.diagnostics_config.num_bins,
                "max_rows": self.diagnostics_config.max_rows,
            },
        }


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    role: str
    n_rows: int
    evaluation: dict[str, object]
    diagnostics: dict[str, object]

    def __post_init__(self) -> None:
        role = _role_to_str(self.role)
        if role not in {"train", "val", "test"}:
            raise ValueError("role must be train/val/test")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "n_rows", _require_nonnegative_int(self.n_rows, "n_rows"))
        if not isinstance(self.evaluation, dict):
            raise ValueError("evaluation must be dict")
        if not isinstance(self.diagnostics, dict):
            raise ValueError("diagnostics must be dict")

    def as_dict(self) -> dict[str, object]:
        return _json_safe(
            {
                "role": self.role,
                "n_rows": self.n_rows,
                "evaluation": self.evaluation,
                "diagnostics": self.diagnostics,
            },
            name="split_evaluation",
        )


@dataclass(frozen=True, slots=True)
class LinearTrainResult:
    schema: str
    dataset_id: str
    manifest_hash: str
    config: dict[str, object]
    preprocess_state: dict[str, object]
    model_bundle_state: dict[str, object]
    splits: dict[str, SplitEvaluation]
    selection_summary: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != LINEAR_TRAINING_RESULT_SCHEMA:
            raise ValueError("invalid schema")
        object.__setattr__(self, "dataset_id", _require_non_empty_str(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "manifest_hash", _require_non_empty_str(self.manifest_hash, "manifest_hash"))
        for name in ("config", "preprocess_state", "model_bundle_state"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be dict")
        if not isinstance(self.selection_summary, dict):
            raise ValueError("selection_summary must be dict")
        if not isinstance(self.splits, dict):
            raise ValueError("splits must be dict")
        keys = set(self.splits.keys())
        if not keys.issubset({"train", "val", "test"}):
            raise ValueError("splits keys must be subset of train/val/test")
        if "train" not in keys or "val" not in keys:
            raise ValueError("splits must include train and val")
        for k, v in self.splits.items():
            if not isinstance(v, SplitEvaluation):
                raise ValueError(f"splits[{k!r}] must be SplitEvaluation")
            if v.role != k:
                raise ValueError("split key must match role")

    def as_dict(self) -> dict[str, object]:
        return _json_safe(
            {
                "schema": self.schema,
                "dataset_id": self.dataset_id,
                "manifest_hash": self.manifest_hash,
                "config": self.config,
                "preprocess_state": self.preprocess_state,
                "model_bundle_state": self.model_bundle_state,
                "splits": {k: v.as_dict() for k, v in self.splits.items()},
                "selection_summary": self.selection_summary,
            },
            name="linear_train_result",
        )


def _projection_for_head(
    manifest: mf.StorageManifest,
    feature_columns: tuple[str, ...],
    target_builder: tg.LinearTargetBuilder,
) -> tuple[str, ...]:
    y_cols = target_builder.column_projection(manifest)
    return tuple(feature_columns) + tuple(c for c in y_cols if c not in feature_columns)


def _extractor_for_columns(
    columns: tuple[str, ...],
    config: LinearTrainConfig,
    manifest: mf.StorageManifest,
) -> ex.IdentityFeatureExtractor:
    return ex.IdentityFeatureExtractor(
        ex.LinearFeatureExtractorConfig(
            feature_columns=columns,
            output_dtype=config.extractor_config.output_dtype,
        ),
        manifest=manifest,
    )


def _split_batches(reader: rd.StorageDatasetReader, role: SplitRole | str, columns: Sequence[str], batch_size: int):
    for batch in reader.iter_split_batches(role, columns=tuple(columns), batch_size=batch_size):
        if not isinstance(batch, pa.RecordBatch):
            raise ValueError("reader.iter_split_batches must yield pyarrow.RecordBatch")
        if batch.num_rows == 0:
            continue
        yield pa.Table.from_batches([batch])


def _preprocess_states_as_dict(states_by_head: dict[str, pp.LinearPreprocessState]) -> dict[str, object]:
    if set(states_by_head.keys()) != set(lm.MODEL_HEADS):
        raise ValueError("states_by_head keys must exactly match model heads")
    for v in states_by_head.values():
        if not isinstance(v, pp.LinearPreprocessState):
            raise ValueError("states_by_head values must be LinearPreprocessState")
    return {
        "schema": "mmrt_linear_preprocess",
        "states_by_head": {head: states_by_head[head].as_dict() for head in lm.MODEL_HEADS},
    }


def fit_preprocessors_from_train_split(reader: rd.StorageDatasetReader, *, manifest: mf.StorageManifest, head_features: hf.ResolvedHeadFeatureSets, config: LinearTrainConfig) -> dict[str, pp.LinearPreprocessState]:
    out: dict[str, pp.LinearPreprocessState] = {}
    for head in lm.MODEL_HEADS:
        cols = head_features.columns_for_head(head)
        extractor = _extractor_for_columns(cols, config, manifest)
        pre = pp.LinearPreprocessor(config.preprocess_config)
        for table in _split_batches(reader, SplitRole.TRAIN, cols, config.batch_size):
            batch = extractor.transform_table(table)
            pre.partial_fit(batch.X, feature_columns=batch.feature_columns)
        out[head] = pre.finalize()
    return out


def train_model_bundle_from_train_split(reader: rd.StorageDatasetReader, *, manifest: mf.StorageManifest, head_features: hf.ResolvedHeadFeatureSets, preprocess_states_by_head: dict[str, pp.LinearPreprocessState], config: LinearTrainConfig) -> lm.LinearModelBundle:
    target_builder = tg.LinearTargetBuilder(config.target_config, manifest=manifest)
    bundle = lm.make_linear_model_bundle(head_features.feature_columns_by_head, config.model_config)
    for _ in range(config.epochs):
        for head in lm.MODEL_HEADS:
            cols = head_features.columns_for_head(head)
            extractor = _extractor_for_columns(cols, config, manifest)
            projection = _projection_for_head(manifest, cols, target_builder)
            pre = pp.LinearPreprocessor.from_state(preprocess_states_by_head[head])
            for table in _split_batches(reader, SplitRole.TRAIN, projection, config.batch_size):
                xb = extractor.transform_table(table)
                tb = target_builder.transform_table(table)
                Xz = pre.transform(xb.X, feature_columns=xb.feature_columns)
                if head == lm.NO_MOVE_HEAD:
                    bundle.no_move.partial_fit(Xz, tb.y_no_move)
                elif head == lm.DIRECTION_HEAD:
                    if tb.move_mask.any():
                        bundle.direction.partial_fit(Xz[tb.move_mask], tb.y_direction[tb.move_mask])
                elif head == lm.MAGNITUDE_UP_HEAD:
                    if tb.up_move_mask.any():
                        bundle.magnitude_up.partial_fit(Xz[tb.up_move_mask], tb.y_magnitude_up[tb.up_move_mask])
                elif head == lm.MAGNITUDE_DOWN_HEAD:
                    if tb.down_move_mask.any():
                        bundle.magnitude_down.partial_fit(Xz[tb.down_move_mask], tb.y_magnitude_down[tb.down_move_mask])
                else:
                    raise ValueError("unknown head")
    return bundle


def evaluate_model_on_split(reader: rd.StorageDatasetReader, *, manifest: mf.StorageManifest, role: SplitRole | str, head_features: hf.ResolvedHeadFeatureSets, preprocess_states_by_head: dict[str, pp.LinearPreprocessState], model_bundle: lm.LinearModelBundle, config: LinearTrainConfig) -> SplitEvaluation:
    role_str = _role_to_str(role)
    target_builder = tg.LinearTargetBuilder(config.target_config, manifest=manifest)

    y_return_bps_parts: list[np.ndarray] = []
    y_direction_parts: list[np.ndarray] = []
    no_move_mask_parts: list[np.ndarray] = []
    move_mask_parts: list[np.ndarray] = []
    up_move_mask_parts: list[np.ndarray] = []
    down_move_mask_parts: list[np.ndarray] = []
    y_no_move_parts: list[np.ndarray] = []
    no_move_p_parts: list[np.ndarray] = []
    direction_p_up_parts: list[np.ndarray] = []
    pred_mag_up_parts: list[np.ndarray] = []
    pred_mag_down_parts: list[np.ndarray] = []

    for head, proba_parts, model_head in (
        (lm.NO_MOVE_HEAD, no_move_p_parts, model_bundle.no_move),
        (lm.DIRECTION_HEAD, direction_p_up_parts, model_bundle.direction),
    ):
        cols = head_features.columns_for_head(head)
        extractor = _extractor_for_columns(cols, config, manifest)
        pre = pp.LinearPreprocessor.from_state(preprocess_states_by_head[head])
        projection = _projection_for_head(manifest, cols, target_builder)
        for table in _split_batches(reader, role, projection, config.batch_size):
            xb = extractor.transform_table(table)
            tb = target_builder.transform_table(table)
            Xz = pre.transform(xb.X, feature_columns=xb.feature_columns)
            proba_parts.append(model_head.predict_proba(Xz)[:, 1])
            if head == lm.DIRECTION_HEAD:
                y_direction_parts.append(tb.y_direction)
                move_mask_parts.append(tb.move_mask)
                no_move_mask_parts.append(tb.no_move_mask)
                up_move_mask_parts.append(tb.up_move_mask)
                down_move_mask_parts.append(tb.down_move_mask)
                y_no_move_parts.append(tb.y_no_move)
                y_return_bps_parts.append(tb.y_return_bps)

    for head, pred_parts, model_head in (
        (lm.MAGNITUDE_UP_HEAD, pred_mag_up_parts, model_bundle.magnitude_up),
        (lm.MAGNITUDE_DOWN_HEAD, pred_mag_down_parts, model_bundle.magnitude_down),
    ):
        cols = head_features.columns_for_head(head)
        extractor = _extractor_for_columns(cols, config, manifest)
        pre = pp.LinearPreprocessor.from_state(preprocess_states_by_head[head])
        projection = _projection_for_head(manifest, cols, target_builder)
        for table in _split_batches(reader, role, projection, config.batch_size):
            xb = extractor.transform_table(table)
            tb = target_builder.transform_table(table)
            Xz = pre.transform(xb.X, feature_columns=xb.feature_columns)
            pred_parts.append(model_head.predict_nonnegative(Xz))

    y_direction = _concat_1d(y_direction_parts, dtype=np.dtype(np.int8), name="y_direction")
    y_return_bps = _concat_1d(y_return_bps_parts, dtype=np.dtype(np.float64), name="y_return_bps")
    direction_p_up = _concat_1d(direction_p_up_parts, dtype=np.dtype(np.float64), name="direction_p_up")
    no_move_p = _concat_1d(no_move_p_parts, dtype=np.dtype(np.float64), name="no_move_p")
    y_no_move = _concat_1d(y_no_move_parts, dtype=np.dtype(np.float64), name="y_no_move")
    no_move_mask = _concat_1d(no_move_mask_parts, dtype=np.dtype(bool), name="no_move_mask")
    move_mask = _concat_1d(move_mask_parts, dtype=np.dtype(bool), name="move_mask")
    up_move_mask = _concat_1d(up_move_mask_parts, dtype=np.dtype(bool), name="up_move_mask")
    down_move_mask = _concat_1d(down_move_mask_parts, dtype=np.dtype(bool), name="down_move_mask")
    pred_mag_up = _concat_1d(pred_mag_up_parts, dtype=np.dtype(np.float64), name="pred_mag_up")
    pred_mag_down = _concat_1d(pred_mag_down_parts, dtype=np.dtype(np.float64), name="pred_mag_down")
    gated = ev.derive_gated_signal_predictions(
        p_no_move=no_move_p,
        p_up_given_move=direction_p_up,
        pred_magnitude_up=pred_mag_up,
        pred_magnitude_down=pred_mag_down,
    )

    evaluation = ev.evaluate_linear_predictions(
        y_return_bps=y_return_bps,
        y_no_move=y_no_move,
        y_direction=y_direction,
        no_move_mask=no_move_mask,
        move_mask=move_mask,
        up_move_mask=up_move_mask,
        down_move_mask=down_move_mask,
        p_no_move=no_move_p,
        p_up_given_move=direction_p_up,
        pred_magnitude_up=pred_mag_up,
        pred_magnitude_down=pred_mag_down,
    ).as_dict()

    diagnostics = dg.build_linear_diagnostics_report(
        model_bundle_state=model_bundle.as_dict(),
        preprocess_state=_preprocess_states_as_dict(preprocess_states_by_head),
        evaluation_result=evaluation,
        p_no_move=no_move_p,
        p_move=gated["p_move"],
        p_up_given_move=direction_p_up,
        p_up_effective=gated["p_up_effective"],
        p_down_effective=gated["p_down_effective"],
        magnitude_up=pred_mag_up,
        magnitude_down=pred_mag_down,
        expected_up_bps=gated["expected_up_bps"],
        expected_down_bps=gated["expected_down_bps"],
        expected_signed_edge_bps=gated["expected_signed_edge_bps"],
        expected_abs_move_bps=gated["expected_abs_move_bps"],
        y_no_move=y_no_move,
        y_direction=y_direction,
        move_mask=move_mask,
        config=config.diagnostics_config,
    )

    return SplitEvaluation(role=role_str, n_rows=int(y_direction.shape[0]), evaluation=evaluation, diagnostics=diagnostics)



def _selection_summary_from_splits(splits: dict[str, SplitEvaluation]) -> dict[str, object]:
    if "val" not in splits:
        raise ValueError("selection summary requires val split")
    val_eval = splits["val"].evaluation

    def req(path: tuple[str, ...]) -> float:
        cur: object = val_eval
        prefix = "evaluation"
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                raise ValueError(f"missing required selection metric: {prefix}[{key!r}]")
            cur = cur[key]
            prefix += f"[{key!r}]"
        return float(cur)

    return {
        "selection_split": "val",
        "primary_metrics": {
            "no_move": {"metric": "auc", "value": req(("no_move", "auc")), "mode": "max", "scope": "all_rows"},
            "direction": {"metric": "auc", "value": req(("direction", "auc")), "mode": "max", "scope": "move_mask"},
            "magnitude_up": {"metric": "mae", "value": req(("magnitude_up", "mae")), "mode": "min", "scope": "up_move_mask"},
            "magnitude_down": {"metric": "mae", "value": req(("magnitude_down", "mae")), "mode": "min", "scope": "down_move_mask"},
        },
        "guardrails": {
            "no_move": {"log_loss": req(("no_move", "log_loss")), "brier": req(("no_move", "brier"))},
            "direction": {"log_loss": req(("direction", "log_loss")), "brier": req(("direction", "brier"))},
            "magnitude_up": {"spearman": req(("magnitude_up", "spearman")), "rmse": req(("magnitude_up", "rmse"))},
            "magnitude_down": {"spearman": req(("magnitude_down", "spearman")), "rmse": req(("magnitude_down", "rmse"))},
        },
    }

def train_linear_model(dataset_root: str, *, config: LinearTrainConfig | None = None) -> LinearTrainResult:
    cfg = config or LinearTrainConfig()
    root_str = _require_non_empty_str(dataset_root, "dataset_root")
    reader = rd.open_dataset(root_str, validate_on_open=cfg.validate_dataset_on_open, batch_size=cfg.batch_size)
    manifest = reader.manifest
    manifest.validate_against_current_code()
    roles = _require_manifest_has_split_roles(manifest)

    resolved_head_features = hf.resolve_head_feature_sets(manifest, cfg.head_feature_config)
    preprocess_states_by_head = fit_preprocessors_from_train_split(reader, manifest=manifest, head_features=resolved_head_features, config=cfg)
    model_bundle = train_model_bundle_from_train_split(
        reader,
        manifest=manifest,
        head_features=resolved_head_features,
        preprocess_states_by_head=preprocess_states_by_head,
        config=cfg,
    )

    split_evals: dict[str, SplitEvaluation] = {}
    for role in roles:
        role_str = role.value
        split_evals[role_str] = evaluate_model_on_split(
            reader,
            manifest=manifest,
            role=role,
            head_features=resolved_head_features,
            preprocess_states_by_head=preprocess_states_by_head,
            model_bundle=model_bundle,
            config=cfg,
        )

    selection_summary = _selection_summary_from_splits(split_evals)

    return LinearTrainResult(
        schema=LINEAR_TRAINING_RESULT_SCHEMA,
        dataset_id=manifest.dataset_id,
        manifest_hash=manifest.content_hash(),
        config={**cfg.as_dict(), "resolved_head_features": resolved_head_features.as_dict()},
        preprocess_state=_preprocess_states_as_dict(preprocess_states_by_head),
        model_bundle_state=model_bundle.as_dict(),
        splits=split_evals,
        selection_summary=selection_summary,
    )


def write_linear_train_artifacts(result: LinearTrainResult, output_dir: str, *, filename: str = DEFAULT_OUTPUT_FILENAME) -> dict[str, str]:
    if not isinstance(result, LinearTrainResult):
        raise ValueError("result must be LinearTrainResult")
    out_dir = Path(_require_non_empty_str(output_dir, "output_dir"))
    name = _require_non_empty_str(filename, "filename")
    if not name.endswith(".json"):
        raise ValueError("filename must end with .json")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    tmp = Path(str(target) + ".tmp")
    payload = json.dumps(result.as_dict(), sort_keys=True, indent=2, allow_nan=True) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return {"result_json": str(target)}


__all__ = [
    "DEFAULT_TRAIN_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_OUTPUT_FILENAME",
    "LINEAR_TRAINING_RESULT_SCHEMA",
    "LinearTrainConfig",
    "SplitEvaluation",
    "LinearTrainResult",
    "fit_preprocessors_from_train_split",
    "train_model_bundle_from_train_split",
    "evaluate_model_on_split",
    "train_linear_model",
    "write_linear_train_artifacts",
]
