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
from mmrt.linear import targets as tg
from mmrt.linear import preprocess as pp
from mmrt.linear import models as lm
from mmrt.linear import evaluate as ev
from mmrt.linear import diagnostics as dg

DEFAULT_TRAIN_BATCH_SIZE = 8192
DEFAULT_EPOCHS = 5
DEFAULT_OUTPUT_FILENAME = "linear_train_result.json"
TRAIN_RESULT_SCHEMA_VERSION = 1


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
            "target_config": {
                "target_horizon_us": self.target_config.target_horizon_us,
                "direction_deadband_bps": self.target_config.direction_deadband_bps,
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
    schema_version: int
    dataset_id: str
    manifest_hash: str
    config: dict[str, object]
    preprocess_state: dict[str, object]
    model_bundle_state: dict[str, object]
    splits: dict[str, SplitEvaluation]

    def __post_init__(self) -> None:
        if self.schema_version != TRAIN_RESULT_SCHEMA_VERSION:
            raise ValueError("invalid schema_version")
        object.__setattr__(self, "dataset_id", _require_non_empty_str(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "manifest_hash", _require_non_empty_str(self.manifest_hash, "manifest_hash"))
        for name in ("config", "preprocess_state", "model_bundle_state"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be dict")
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
                "schema_version": self.schema_version,
                "dataset_id": self.dataset_id,
                "manifest_hash": self.manifest_hash,
                "config": self.config,
                "preprocess_state": self.preprocess_state,
                "model_bundle_state": self.model_bundle_state,
                "splits": {k: v.as_dict() for k, v in self.splits.items()},
            },
            name="linear_train_result",
        )


def _column_projection(manifest: mf.StorageManifest, extractor: ex.IdentityFeatureExtractor, target_builder: tg.LinearTargetBuilder) -> tuple[str, ...]:
    x_cols = extractor.column_projection(manifest)
    y_cols = target_builder.column_projection(manifest)
    return tuple(x_cols) + tuple(c for c in y_cols if c not in x_cols)


def _split_batches(reader: rd.StorageDatasetReader, role: SplitRole | str, columns: Sequence[str], batch_size: int):
    for batch in reader.iter_split_batches(role, columns=tuple(columns), batch_size=batch_size):
        if not isinstance(batch, pa.RecordBatch):
            raise ValueError("reader.iter_split_batches must yield pyarrow.RecordBatch")
        if batch.num_rows == 0:
            continue
        yield pa.Table.from_batches([batch])


def fit_preprocessor_from_train_split(reader: rd.StorageDatasetReader, *, manifest: mf.StorageManifest, config: LinearTrainConfig) -> pp.LinearPreprocessState:
    extractor = ex.IdentityFeatureExtractor(config.extractor_config, manifest=manifest)
    x_cols = extractor.column_projection(manifest)
    pre = pp.LinearPreprocessor(config.preprocess_config)
    for table in _split_batches(reader, SplitRole.TRAIN, x_cols, config.batch_size):
        batch = extractor.transform_table(table)
        pre.partial_fit(batch.X, feature_columns=batch.feature_columns)
    return pre.finalize()


def train_model_bundle_from_train_split(reader: rd.StorageDatasetReader, *, manifest: mf.StorageManifest, preprocess_state: pp.LinearPreprocessState, config: LinearTrainConfig) -> lm.LinearModelBundle:
    extractor = ex.IdentityFeatureExtractor(config.extractor_config, manifest=manifest)
    target_builder = tg.LinearTargetBuilder(config.target_config, manifest=manifest)
    projection = _column_projection(manifest, extractor, target_builder)
    pre = pp.LinearPreprocessor.from_state(preprocess_state)
    bundle = lm.make_linear_model_bundle(preprocess_state.feature_columns, config.model_config)
    for _ in range(config.epochs):
        for table in _split_batches(reader, SplitRole.TRAIN, projection, config.batch_size):
            xb = extractor.transform_table(table)
            tb = target_builder.transform_table(table)
            Xz = pre.transform(xb.X, feature_columns=xb.feature_columns)
            if tb.direction_mask.any():
                bundle.direction.partial_fit(Xz[tb.direction_mask], tb.y_direction[tb.direction_mask])
            bundle.magnitude_up.partial_fit(Xz, tb.y_magnitude_up)
            bundle.magnitude_down.partial_fit(Xz, tb.y_magnitude_down)
    return bundle


def evaluate_model_on_split(reader: rd.StorageDatasetReader, *, manifest: mf.StorageManifest, role: SplitRole | str, preprocess_state: pp.LinearPreprocessState, model_bundle: lm.LinearModelBundle, config: LinearTrainConfig) -> SplitEvaluation:
    role_str = _role_to_str(role)
    extractor = ex.IdentityFeatureExtractor(config.extractor_config, manifest=manifest)
    target_builder = tg.LinearTargetBuilder(config.target_config, manifest=manifest)
    projection = _column_projection(manifest, extractor, target_builder)
    pre = pp.LinearPreprocessor.from_state(preprocess_state)

    y_direction_parts: list[np.ndarray] = []
    direction_mask_parts: list[np.ndarray] = []
    direction_p_up_parts: list[np.ndarray] = []
    y_mag_up_parts: list[np.ndarray] = []
    pred_mag_up_parts: list[np.ndarray] = []
    y_mag_down_parts: list[np.ndarray] = []
    pred_mag_down_parts: list[np.ndarray] = []

    for table in _split_batches(reader, role, projection, config.batch_size):
        xb = extractor.transform_table(table)
        tb = target_builder.transform_table(table)
        Xz = pre.transform(xb.X, feature_columns=xb.feature_columns)
        pred = model_bundle.predict(Xz)

        y_direction_parts.append(tb.y_direction)
        direction_mask_parts.append(tb.direction_mask)
        direction_p_up_parts.append(pred["direction_proba"][:, 1])
        y_mag_up_parts.append(tb.y_magnitude_up)
        pred_mag_up_parts.append(pred["magnitude_up"])
        y_mag_down_parts.append(tb.y_magnitude_down)
        pred_mag_down_parts.append(pred["magnitude_down"])

    y_direction = _concat_1d(y_direction_parts, dtype=np.dtype(np.int8), name="y_direction")
    direction_mask = _concat_1d(direction_mask_parts, dtype=np.dtype(bool), name="direction_mask")
    direction_p_up = _concat_1d(direction_p_up_parts, dtype=np.dtype(np.float64), name="direction_p_up")
    y_mag_up = _concat_1d(y_mag_up_parts, dtype=np.dtype(np.float64), name="y_mag_up")
    pred_mag_up = _concat_1d(pred_mag_up_parts, dtype=np.dtype(np.float64), name="pred_mag_up")
    y_mag_down = _concat_1d(y_mag_down_parts, dtype=np.dtype(np.float64), name="y_mag_down")
    pred_mag_down = _concat_1d(pred_mag_down_parts, dtype=np.dtype(np.float64), name="pred_mag_down")

    evaluation = ev.evaluate_linear_predictions(
        y_direction=y_direction,
        direction_mask=direction_mask,
        direction_p_up=direction_p_up,
        y_magnitude_up=y_mag_up,
        pred_magnitude_up=pred_mag_up,
        y_magnitude_down=y_mag_down,
        pred_magnitude_down=pred_mag_down,
    ).as_dict()

    diagnostics = dg.build_linear_diagnostics_report(
        model_bundle_state=model_bundle.as_dict(),
        preprocess_state=preprocess_state.as_dict(),
        evaluation_result=evaluation,
        direction_p_up=direction_p_up,
        magnitude_up=pred_mag_up,
        magnitude_down=pred_mag_down,
        y_direction=y_direction,
        direction_mask=direction_mask,
        config=config.diagnostics_config,
    )

    return SplitEvaluation(role=role_str, n_rows=int(y_direction.shape[0]), evaluation=evaluation, diagnostics=diagnostics)


def train_linear_model(dataset_root: str, *, config: LinearTrainConfig | None = None) -> LinearTrainResult:
    cfg = config or LinearTrainConfig()
    root_str = _require_non_empty_str(dataset_root, "dataset_root")
    reader = rd.open_dataset(root_str, validate_on_open=cfg.validate_dataset_on_open, batch_size=cfg.batch_size)
    manifest = reader.manifest
    manifest.validate_against_current_code()
    roles = _require_manifest_has_split_roles(manifest)

    preprocess_state = fit_preprocessor_from_train_split(reader, manifest=manifest, config=cfg)
    model_bundle = train_model_bundle_from_train_split(
        reader,
        manifest=manifest,
        preprocess_state=preprocess_state,
        config=cfg,
    )

    split_evals: dict[str, SplitEvaluation] = {}
    for role in roles:
        role_str = role.value
        split_evals[role_str] = evaluate_model_on_split(
            reader,
            manifest=manifest,
            role=role,
            preprocess_state=preprocess_state,
            model_bundle=model_bundle,
            config=cfg,
        )

    return LinearTrainResult(
        schema_version=TRAIN_RESULT_SCHEMA_VERSION,
        dataset_id=manifest.dataset_id,
        manifest_hash=manifest.content_hash(),
        config=cfg.as_dict(),
        preprocess_state=preprocess_state.as_dict(),
        model_bundle_state=model_bundle.as_dict(),
        splits=split_evals,
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
    "TRAIN_RESULT_SCHEMA_VERSION",
    "LinearTrainConfig",
    "SplitEvaluation",
    "LinearTrainResult",
    "fit_preprocessor_from_train_split",
    "train_model_bundle_from_train_split",
    "evaluate_model_on_split",
    "train_linear_model",
    "write_linear_train_artifacts",
]
