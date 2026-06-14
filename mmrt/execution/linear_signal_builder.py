"""Build canonical linear signal artifacts from execution-tape feature replay.

This module converts shared decision-feature-pipeline output into aligned
linear signals. Feature rows come exclusively from
:mod:`mmrt.execution.feature_replay`, the same path that produces supervised
training features, and every artifact records the transform identity it was
built with. It does not read labels, storage datasets, RL code, or
adverse-selection components.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Iterator, Mapping

import numpy as np

from mmrt.execution.execution_tape import ExecutionTape
from mmrt.execution.execution_tape_writer import NpyChunkWriter
from mmrt.execution.decision_grid import DecisionGrid, validate_decision_grid_for_execution_tape
from mmrt.execution.feature_replay import (
    DecisionFeatureChunk,
    decision_feature_column_names,
    iter_decision_feature_chunks_for_decision_grid,
)
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    LINEAR_SIGNAL_ARTIFACT_SCHEMA,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    LinearSignalConfig,
    linear_signal_array_fields,
    predictions_to_signal_arrays,
    save_linear_signal_artifact_arrays,
    validate_linear_signal_artifact_metadata,
    validate_linear_signals_for_decision_grid,
)
from mmrt.features.pipeline import FeaturePipelineConfig
from mmrt.features.schedule import decision_schedule_config_from_dict
from mmrt.features.transforms import TransformConfig, transform_config_from_dict
from mmrt.linear import models as lm
from mmrt.linear import preprocess as pp
from mmrt.linear.train import (
    LinearTrainResult,
    linear_model_bundle_from_train_result,
    linear_preprocess_states_from_train_result,
)

_ALLOWED_DTYPES = ("float32", "float64")
ExecutionLinearFeatureChunk = DecisionFeatureChunk


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_output_dtype(value: str) -> str:
    if value not in _ALLOWED_DTYPES:
        raise ValueError(f"output_dtype must be one of {_ALLOWED_DTYPES}")
    return value


def _coerce_feature_names(values: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(values)
    if not names:
        raise ValueError("feature_names must be non-empty")
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("feature_names entries must be non-empty strings")
        if name in seen:
            raise ValueError("feature_names must be unique")
        seen.add(name)
    return names


def _coerce_transform_config_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("transform_config must be a mapping")
    payload = dict(value)
    transform_config_from_dict(payload)
    return payload


def _coerce_decision_schedule_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("decision_schedule must be a mapping")
    payload = dict(value)
    decision_schedule_config_from_dict(payload)
    return payload


def _payloads_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return json.dumps(dict(left), sort_keys=True) == json.dumps(dict(right), sort_keys=True)


def _close_memmap(arr: np.ndarray) -> None:
    mmap = getattr(arr, "_mmap", None)
    if mmap is not None:
        mmap.close()


@dataclass(frozen=True, slots=True)
class ExecutionLinearFeatureDataset:
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    decision_event_seq: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    replay_start_event_index: int
    start_event_index: int
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: dict[str, object]
    transform_config: dict[str, object]

    def __post_init__(self) -> None:
        event_idx = np.ascontiguousarray(np.asarray(self.decision_event_index, dtype=np.int64))
        local_ts = np.ascontiguousarray(np.asarray(self.decision_local_ts_us, dtype=np.int64))
        event_seq = np.ascontiguousarray(np.asarray(self.decision_event_seq, dtype=np.int64))
        features = np.ascontiguousarray(np.asarray(self.features))
        if event_idx.ndim != 1:
            raise ValueError("decision_event_index must be rank-1")
        if local_ts.ndim != 1:
            raise ValueError("decision_local_ts_us must be rank-1")
        if event_seq.ndim != 1:
            raise ValueError("decision_event_seq must be rank-1")
        if features.ndim != 2:
            raise ValueError("features must be rank-2")
        if features.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("features dtype must be float32 or float64")
        if features.shape[0] != event_idx.shape[0] or features.shape[0] != local_ts.shape[0] or features.shape[0] != event_seq.shape[0]:
            raise ValueError("row count must match decision arrays")
        if event_idx.size and (event_idx < 0).any():
            raise ValueError("decision_event_index must be nonnegative")
        if local_ts.size and (local_ts <= 0).any():
            raise ValueError("decision_local_ts_us must be positive")
        if event_seq.size and (event_seq < 0).any():
            raise ValueError("decision_event_seq must be nonnegative")
        if event_idx.size > 1 and (np.diff(event_idx) <= 0).any():
            raise ValueError("decision_event_index must be strictly increasing")
        if local_ts.size > 1 and (np.diff(local_ts) <= 0).any():
            raise ValueError("decision_local_ts_us must be strictly increasing")
        if not np.isfinite(features).all():
            raise ValueError("features must be finite")
        names = _coerce_feature_names(tuple(self.feature_names))
        if len(names) != features.shape[1]:
            raise ValueError("feature_names length must equal features width")
        replay_start = _require_nonnegative_int(self.replay_start_event_index, "replay_start_event_index")
        start = _require_nonnegative_int(self.start_event_index, "start_event_index")
        if event_idx.size:
            if start != int(event_idx[0]):
                raise ValueError("start_event_index must equal first decision_event_index when decisions exist")
            if replay_start > start:
                raise ValueError("replay_start_event_index must be <= start_event_index")
        object.__setattr__(self, "decision_event_index", event_idx)
        object.__setattr__(self, "decision_local_ts_us", local_ts)
        object.__setattr__(self, "decision_event_seq", event_seq)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "replay_start_event_index", replay_start)
        object.__setattr__(self, "start_event_index", start)
        object.__setattr__(self, "decision_grid_schema", _require_nonempty_str(self.decision_grid_schema, "decision_grid_schema"))
        object.__setattr__(self, "decision_grid_hash", _require_nonempty_str(self.decision_grid_hash, "decision_grid_hash"))
        object.__setattr__(self, "decision_grid_n_rows", _require_positive_int(self.decision_grid_n_rows, "decision_grid_n_rows"))
        if self.decision_grid_n_rows < features.shape[0]:
            raise ValueError("decision_grid_n_rows must cover dataset rows")
        object.__setattr__(self, "decision_schedule", _coerce_decision_schedule_payload(self.decision_schedule))
        object.__setattr__(self, "transform_config", _coerce_transform_config_payload(self.transform_config))

    @property
    def num_decisions(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])


def execution_linear_feature_dataset_summary(dataset: ExecutionLinearFeatureDataset) -> dict[str, object]:
    if not isinstance(dataset, ExecutionLinearFeatureDataset):
        raise ValueError("dataset must be ExecutionLinearFeatureDataset")
    return {
        "num_decisions": dataset.num_decisions,
        "num_features": dataset.num_features,
        "feature_names": list(dataset.feature_names),
        "first_decision_event_index": int(dataset.decision_event_index[0]) if dataset.num_decisions else None,
        "last_decision_event_index": int(dataset.decision_event_index[-1]) if dataset.num_decisions else None,
        "first_decision_local_ts_us": int(dataset.decision_local_ts_us[0]) if dataset.num_decisions else None,
        "last_decision_local_ts_us": int(dataset.decision_local_ts_us[-1]) if dataset.num_decisions else None,
        "first_decision_event_seq": int(dataset.decision_event_seq[0]) if dataset.num_decisions else None,
        "last_decision_event_seq": int(dataset.decision_event_seq[-1]) if dataset.num_decisions else None,
        "decision_grid_schema": dataset.decision_grid_schema,
        "decision_grid_hash": dataset.decision_grid_hash,
        "decision_grid_n_rows": dataset.decision_grid_n_rows,
        "decision_schedule": dict(dataset.decision_schedule),
        "replay_start_event_index": dataset.replay_start_event_index,
        "start_event_index": dataset.start_event_index,
        "transform_config": dict(dataset.transform_config),
    }


def execution_linear_feature_names() -> tuple[str, ...]:
    return decision_feature_column_names()


def iter_execution_linear_feature_chunks_for_decision_grid(
    tape: ExecutionTape,
    *,
    decision_grid: DecisionGrid,
    transform_config: TransformConfig | None = None,
    chunk_rows: int = 100_000,
    output_dtype: str = "float32",
) -> Iterator[ExecutionLinearFeatureChunk]:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(decision_grid, DecisionGrid):
        raise ValueError("decision_grid must be DecisionGrid")
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    transform = transform_config if transform_config is not None else TransformConfig()
    pipeline_config = FeaturePipelineConfig(
        schedule=decision_schedule_config_from_dict(decision_grid.decision_schedule),
        transform=transform,
    )
    yield from iter_decision_feature_chunks_for_decision_grid(
        tape,
        decision_grid=decision_grid,
        pipeline_config=pipeline_config,
        chunk_rows=chunk_rows,
        output_dtype=_require_output_dtype(output_dtype),
    )


@dataclass(frozen=True, slots=True)
class LinearSignalBuildResult:
    feature_dataset: ExecutionLinearFeatureDataset
    artifact: LinearSignalArtifact
    predictions: dict[str, np.ndarray]
    prediction_summary: dict[str, object]


def _head_model(bundle: lm.LinearModelBundle, head: str):
    if head == lm.NO_MOVE_HEAD:
        return bundle.no_move
    if head == lm.DIRECTION_HEAD:
        return bundle.direction
    if head == lm.MAGNITUDE_UP_HEAD:
        return bundle.magnitude_up
    if head == lm.MAGNITUDE_DOWN_HEAD:
        return bundle.magnitude_down
    raise ValueError("unknown head")


def transform_config_from_train_result(result: LinearTrainResult) -> TransformConfig:
    if not isinstance(result, LinearTrainResult):
        raise ValueError("result must be LinearTrainResult")
    return transform_config_from_dict(result.transform_config)


def predict_linear_heads_for_execution_features(
    *,
    feature_dataset: ExecutionLinearFeatureDataset,
    model_bundle: lm.LinearModelBundle,
    preprocess_states_by_head: Mapping[str, pp.LinearPreprocessState],
    output_dtype: str = "float32",
) -> dict[str, np.ndarray]:
    if not isinstance(feature_dataset, ExecutionLinearFeatureDataset):
        raise ValueError("feature_dataset must be ExecutionLinearFeatureDataset")
    if not isinstance(model_bundle, lm.LinearModelBundle):
        raise ValueError("model_bundle must be LinearModelBundle")
    dtype = np.dtype(_require_output_dtype(output_dtype))
    if set(preprocess_states_by_head.keys()) != set(lm.MODEL_HEADS):
        raise ValueError("preprocess_states_by_head keys must exactly match MODEL_HEADS")
    name_to_idx = {name: idx for idx, name in enumerate(feature_dataset.feature_names)}
    transformed: dict[str, np.ndarray] = {}
    for head in lm.MODEL_HEADS:
        state = preprocess_states_by_head[head]
        if not isinstance(state, pp.LinearPreprocessState):
            raise ValueError("preprocess_states_by_head values must be LinearPreprocessState")
        model = _head_model(model_bundle, head)
        cols = tuple(model.feature_columns)
        if state.feature_columns != cols:
            raise ValueError(f"preprocess feature_columns differ from model feature_columns for head {head!r}")
        missing = [col for col in cols if col not in name_to_idx]
        if missing:
            raise ValueError(f"missing execution feature columns for head {head!r}: {missing}")
        indices = np.asarray([name_to_idx[col] for col in cols], dtype=np.int64)
        X = np.ascontiguousarray(feature_dataset.features[:, indices], dtype=dtype)
        transformed[head] = pp.LinearPreprocessor.from_state(state).transform(X, feature_columns=cols)
    return {
        NO_MOVE_PROBA_KEY: np.ascontiguousarray(model_bundle.no_move.predict_proba(transformed[lm.NO_MOVE_HEAD]), dtype=dtype),
        DIRECTION_PROBA_KEY: np.ascontiguousarray(model_bundle.direction.predict_proba(transformed[lm.DIRECTION_HEAD]), dtype=dtype),
        MAGNITUDE_UP_KEY: np.ascontiguousarray(model_bundle.magnitude_up.predict_nonnegative(transformed[lm.MAGNITUDE_UP_HEAD]), dtype=dtype),
        MAGNITUDE_DOWN_KEY: np.ascontiguousarray(model_bundle.magnitude_down.predict_nonnegative(transformed[lm.MAGNITUDE_DOWN_HEAD]), dtype=dtype),
    }


def _validate_train_feature_identity(feature_dataset: ExecutionLinearFeatureDataset, linear_train_result: LinearTrainResult) -> None:
    if not _payloads_equal(feature_dataset.transform_config, linear_train_result.transform_config):
        raise ValueError(
            "feature_dataset transform_config does not match linear_train_result transform_config; "
            "build features with transform_config_from_train_result(linear_train_result)"
        )
    if not _payloads_equal(feature_dataset.decision_schedule, linear_train_result.decision_schedule):
        raise ValueError(
            "feature_dataset decision_schedule does not match linear_train_result decision_schedule; "
            "build features from the same decision grid lineage as the linear_train_result"
        )


def build_linear_signal_build_result(
    *,
    tape: ExecutionTape,
    feature_dataset: ExecutionLinearFeatureDataset,
    linear_train_result: LinearTrainResult,
    signal_config: LinearSignalConfig = LinearSignalConfig(),
    output_dtype: str = "float32",
) -> LinearSignalBuildResult:
    if not isinstance(feature_dataset, ExecutionLinearFeatureDataset):
        raise ValueError("feature_dataset must be ExecutionLinearFeatureDataset")
    if feature_dataset.num_decisions <= 0:
        raise ValueError("feature_dataset must contain at least one decision")
    if not isinstance(linear_train_result, LinearTrainResult):
        raise ValueError("linear_train_result must be LinearTrainResult")
    _validate_train_feature_identity(feature_dataset, linear_train_result)
    model_bundle = linear_model_bundle_from_train_result(linear_train_result)
    preprocess_states = linear_preprocess_states_from_train_result(linear_train_result)
    predictions = predict_linear_heads_for_execution_features(
        feature_dataset=feature_dataset,
        model_bundle=model_bundle,
        preprocess_states_by_head=preprocess_states,
        output_dtype=output_dtype,
    )
    arrays = predictions_to_signal_arrays(predictions, config=signal_config, output_dtype=output_dtype)
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_grid_schema=feature_dataset.decision_grid_schema,
        decision_grid_hash=feature_dataset.decision_grid_hash,
        decision_grid_n_rows=feature_dataset.decision_grid_n_rows,
        decision_schedule=dict(feature_dataset.decision_schedule),
        start_event_index=feature_dataset.start_event_index,
        n_rows=arrays.n_rows,
    )
    artifact = LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=feature_dataset.decision_event_index,
        decision_local_ts_us=feature_dataset.decision_local_ts_us,
        decision_event_seq=feature_dataset.decision_event_seq,
    )
    validate_linear_signal_artifact_metadata(
        artifact,
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_grid_schema=feature_dataset.decision_grid_schema,
        decision_grid_hash=feature_dataset.decision_grid_hash,
        decision_grid_n_rows=feature_dataset.decision_grid_n_rows,
        decision_schedule=dict(feature_dataset.decision_schedule),
        start_event_index=feature_dataset.start_event_index,
    )
    return LinearSignalBuildResult(
        feature_dataset=feature_dataset,
        artifact=artifact,
        predictions=predictions,
        prediction_summary=linear_prediction_summary(predictions, artifact),
    )


def build_linear_signal_artifact_from_execution_features(
    *,
    tape: ExecutionTape,
    feature_dataset: ExecutionLinearFeatureDataset,
    linear_train_result: LinearTrainResult,
    signal_config: LinearSignalConfig = LinearSignalConfig(),
    output_dtype: str = "float32",
) -> LinearSignalArtifact:
    return build_linear_signal_build_result(
        tape=tape,
        feature_dataset=feature_dataset,
        linear_train_result=linear_train_result,
        signal_config=signal_config,
        output_dtype=output_dtype,
    ).artifact


def _stats(arr: np.ndarray, *, include_std: bool) -> dict[str, object]:
    values = np.asarray(arr, dtype=np.float64)
    if values.size == 0:
        out: dict[str, object] = {"mean": None, "min": None, "max": None}
        out.update({"std": None} if include_std else {"p01": None, "p50": None, "p99": None})
        return out
    out = {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
    if include_std:
        out["std"] = float(np.std(values))
    else:
        q = np.quantile(values, [0.01, 0.50, 0.99])
        out.update({"p01": float(q[0]), "p50": float(q[1]), "p99": float(q[2])})
    return out


def linear_prediction_summary(predictions: Mapping[str, np.ndarray], signals: LinearSignalArtifact) -> dict[str, object]:
    if not isinstance(signals, LinearSignalArtifact):
        raise ValueError("signals must be LinearSignalArtifact")
    p_no_move = np.asarray(predictions[NO_MOVE_PROBA_KEY])[:, 1]
    return {
        "n_rows": signals.n_rows,
        "p_no_move": _stats(p_no_move, include_std=False),
        "expected_return_bps": _stats(signals.arrays.expected_return_bps, include_std=True) | {k: v for k, v in _stats(signals.arrays.expected_return_bps, include_std=False).items() if k.startswith("p")},
        "expected_abs_move_bps": _stats(signals.arrays.expected_abs_move_bps, include_std=True) | {k: v for k, v in _stats(signals.arrays.expected_abs_move_bps, include_std=False).items() if k.startswith("p")},
        "predicted_vol_bps": _stats(signals.arrays.predicted_vol_bps, include_std=True),
        "confidence": _stats(signals.arrays.confidence, include_std=True),
    }


@dataclass(frozen=True, slots=True)
class LinearSignalDiskBuildResult:
    output_npz: Path
    feature_dataset_summary: dict[str, object]
    linear_signals_summary: dict[str, object]
    alignment_summary: dict[str, object]
    prediction_summary: dict[str, object]


@dataclass(slots=True)
class _StreamStats:
    feature_names: tuple[str, ...]
    replay_start_event_index: int
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: dict[str, object]
    transform_config: dict[str, object]
    num_decisions: int = 0
    first_decision_event_index: int | None = None
    last_decision_event_index: int | None = None
    first_decision_local_ts_us: int | None = None
    last_decision_local_ts_us: int | None = None
    first_decision_event_seq: int | None = None
    last_decision_event_seq: int | None = None

    @property
    def start_event_index(self) -> int:
        if self.first_decision_event_index is None:
            raise ValueError("feature_dataset must contain at least one decision")
        return self.first_decision_event_index

    def update(self, dataset: ExecutionLinearFeatureDataset) -> None:
        if dataset.feature_names != self.feature_names:
            raise ValueError("execution feature names changed during chunk replay")
        if self.last_decision_event_index is not None and int(dataset.decision_event_index[0]) <= self.last_decision_event_index:
            raise ValueError("decision_event_index must be strictly increasing across chunks")
        if self.last_decision_local_ts_us is not None and int(dataset.decision_local_ts_us[0]) <= self.last_decision_local_ts_us:
            raise ValueError("decision_local_ts_us must be strictly increasing across chunks")
        if self.first_decision_event_index is None:
            self.first_decision_event_index = int(dataset.decision_event_index[0])
            self.first_decision_local_ts_us = int(dataset.decision_local_ts_us[0])
            self.first_decision_event_seq = int(dataset.decision_event_seq[0])
        self.last_decision_event_index = int(dataset.decision_event_index[-1])
        self.last_decision_local_ts_us = int(dataset.decision_local_ts_us[-1])
        self.last_decision_event_seq = int(dataset.decision_event_seq[-1])
        self.num_decisions += dataset.num_decisions

    def require_nonempty(self) -> None:
        if self.num_decisions <= 0:
            raise ValueError("feature_dataset must contain at least one decision")

    def feature_dataset_summary(self) -> dict[str, object]:
        self.require_nonempty()
        return {
            "num_decisions": self.num_decisions,
            "num_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "first_decision_event_index": self.first_decision_event_index,
            "last_decision_event_index": self.last_decision_event_index,
            "first_decision_local_ts_us": self.first_decision_local_ts_us,
            "last_decision_local_ts_us": self.last_decision_local_ts_us,
            "first_decision_event_seq": self.first_decision_event_seq,
            "last_decision_event_seq": self.last_decision_event_seq,
            "decision_grid_schema": self.decision_grid_schema,
            "decision_grid_hash": self.decision_grid_hash,
            "decision_grid_n_rows": self.decision_grid_n_rows,
            "decision_schedule": dict(self.decision_schedule),
            "replay_start_event_index": self.replay_start_event_index,
            "start_event_index": self.start_event_index,
            "transform_config": dict(self.transform_config),
        }


@dataclass(frozen=True, slots=True)
class _FinalizedArrays:
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    decision_event_seq: np.ndarray
    arrays: dict[str, np.ndarray]
    row_counts: dict[str, int]

    def close(self) -> None:
        _close_memmap(self.decision_event_index)
        _close_memmap(self.decision_local_ts_us)
        _close_memmap(self.decision_event_seq)
        for arr in self.arrays.values():
            _close_memmap(arr)


@dataclass(slots=True)
class _LinearSignalChunkedWriters:
    chunk_dir: Path
    arrays_dir: Path
    writers: dict[str, NpyChunkWriter]

    @classmethod
    def create(cls, output_npz: Path, *, chunk_rows: int, output_dtype: str) -> "_LinearSignalChunkedWriters":
        chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
        dtype = np.dtype(_require_output_dtype(output_dtype))
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        prefix = output_npz.parent / f".{output_npz.name}"
        chunk_dir = Path(str(prefix) + ".chunks")
        arrays_dir = Path(str(prefix) + ".arrays")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        shutil.rmtree(arrays_dir, ignore_errors=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        writers = {
            "decision_event_index": NpyChunkWriter("decision_event_index", np.int64, (), chunk_rows, chunk_dir),
            "decision_local_ts_us": NpyChunkWriter("decision_local_ts_us", np.int64, (), chunk_rows, chunk_dir),
            "decision_event_seq": NpyChunkWriter("decision_event_seq", np.int64, (), chunk_rows, chunk_dir),
        }
        writers.update({name: NpyChunkWriter(name, dtype, (), chunk_rows, chunk_dir) for name in linear_signal_array_fields()})
        return cls(chunk_dir=chunk_dir, arrays_dir=arrays_dir, writers=writers)

    def append(self, dataset: ExecutionLinearFeatureDataset, signal_arrays) -> None:
        if dataset.num_decisions != int(signal_arrays.n_rows):
            raise ValueError("signal rows must match feature chunk rows")
        self.writers["decision_event_index"].append_many(dataset.decision_event_index)
        self.writers["decision_local_ts_us"].append_many(dataset.decision_local_ts_us)
        self.writers["decision_event_seq"].append_many(dataset.decision_event_seq)
        for name in linear_signal_array_fields():
            self.writers[name].append_many(getattr(signal_arrays, name))

    def finalize(self) -> _FinalizedArrays:
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        row_counts = {name: writer.finalize(self.arrays_dir / f"{name}.npy") for name, writer in self.writers.items()}
        if len(set(row_counts.values())) != 1:
            raise RuntimeError("linear signal chunk row count mismatch")
        return _FinalizedArrays(
            decision_event_index=np.load(self.arrays_dir / "decision_event_index.npy", mmap_mode="r"),
            decision_local_ts_us=np.load(self.arrays_dir / "decision_local_ts_us.npy", mmap_mode="r"),
            decision_event_seq=np.load(self.arrays_dir / "decision_event_seq.npy", mmap_mode="r"),
            arrays={name: np.load(self.arrays_dir / f"{name}.npy", mmap_mode="r") for name in linear_signal_array_fields()},
            row_counts=row_counts,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.chunk_dir, ignore_errors=True)
        shutil.rmtree(self.arrays_dir, ignore_errors=True)


def _feature_dataset_from_chunk(
    chunk: ExecutionLinearFeatureChunk,
    *,
    replay_start_event_index: int,
    decision_grid: DecisionGrid,
    decision_schedule: Mapping[str, object],
    transform_config: Mapping[str, object],
) -> ExecutionLinearFeatureDataset:
    if int(chunk.features.shape[0]) <= 0:
        raise ValueError("feature chunk must contain at least one decision")
    return ExecutionLinearFeatureDataset(
        decision_event_index=chunk.decision_event_index,
        decision_local_ts_us=chunk.decision_local_ts_us,
        decision_event_seq=chunk.decision_event_seq,
        features=chunk.features,
        feature_names=tuple(chunk.feature_names),
        replay_start_event_index=replay_start_event_index,
        start_event_index=int(chunk.decision_event_index[0]),
        decision_grid_schema=decision_grid.metadata.schema,
        decision_grid_hash=decision_grid.decision_grid_hash,
        decision_grid_n_rows=decision_grid.n_rows,
        decision_schedule=dict(decision_schedule),
        transform_config=dict(transform_config),
    )


def _quantiles_disk_backed(arr: np.ndarray, *, chunk_rows: int, temp_path: Path) -> list[float]:
    n = int(arr.shape[0])
    tmp = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float64, shape=(n,))
    try:
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            tmp[start:end] = np.asarray(arr[start:end], dtype=np.float64)
        tmp.sort()
        out: list[float] = []
        for q in (0.01, 0.50, 0.99):
            h = (n - 1) * q
            lo = int(np.floor(h))
            hi = int(np.ceil(h))
            out.append(float(tmp[lo]) if lo == hi else float((1.0 - (h - lo)) * tmp[lo] + (h - lo) * tmp[hi]))
        return out
    finally:
        tmp.flush()
        _close_memmap(tmp)
        temp_path.unlink(missing_ok=True)


def _stats_chunked(arr: np.ndarray, *, include_std: bool, chunk_rows: int, quantile_temp_path: Path | None = None) -> dict[str, object]:
    n = int(arr.shape[0])
    total = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = np.asarray(arr[start:end], dtype=np.float64)
        total += float(np.sum(chunk, dtype=np.float64))
        min_value = min(min_value, float(np.min(chunk)))
        max_value = max(max_value, float(np.max(chunk)))
    mean = float(total / n)
    out = {"mean": mean, "min": min_value, "max": max_value}
    if include_std:
        ss = 0.0
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            centered = np.asarray(arr[start:end], dtype=np.float64) - mean
            ss += float(np.sum(centered * centered, dtype=np.float64))
        out["std"] = float(np.sqrt(max(ss / n, 0.0)))
    else:
        values = _quantiles_disk_backed(arr, chunk_rows=chunk_rows, temp_path=quantile_temp_path) if quantile_temp_path is not None else [float(x) for x in np.quantile(np.asarray(arr, dtype=np.float64), [0.01, 0.50, 0.99])]
        out.update({"p01": values[0], "p50": values[1], "p99": values[2]})
    return out


def linear_signal_array_prediction_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    n_rows: int,
    chunk_rows: int = 100_000,
    temp_prefix: str | Path | None = None,
) -> dict[str, object]:
    if not isinstance(arrays, Mapping):
        raise ValueError("arrays must be a mapping")
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    prefix = None if temp_prefix is None else Path(temp_prefix)

    def qpath(name: str) -> Path | None:
        return None if prefix is None else Path(str(prefix) + f".summary.{name}.npy")

    return {
        "n_rows": int(n_rows),
        "p_no_move": _stats_chunked(arrays["p_no_move"], include_std=False, chunk_rows=chunk_rows, quantile_temp_path=qpath("p_no_move")),
        "expected_return_bps": _stats_chunked(arrays["expected_return_bps"], include_std=True, chunk_rows=chunk_rows) | {k: v for k, v in _stats_chunked(arrays["expected_return_bps"], include_std=False, chunk_rows=chunk_rows, quantile_temp_path=qpath("expected_return_bps")).items() if k.startswith("p")},
        "expected_abs_move_bps": _stats_chunked(arrays["expected_abs_move_bps"], include_std=True, chunk_rows=chunk_rows) | {k: v for k, v in _stats_chunked(arrays["expected_abs_move_bps"], include_std=False, chunk_rows=chunk_rows, quantile_temp_path=qpath("expected_abs_move_bps")).items() if k.startswith("p")},
        "predicted_vol_bps": _stats_chunked(arrays["predicted_vol_bps"], include_std=True, chunk_rows=chunk_rows),
        "confidence": _stats_chunked(arrays["confidence"], include_std=True, chunk_rows=chunk_rows),
    }


def _linear_signal_summary_from_arrays(
    *,
    path: str,
    metadata: LinearSignalArtifactMetadata,
    dtype: np.dtype,
    decision_event_index: np.ndarray,
    decision_local_ts_us: np.ndarray,
    decision_event_seq: np.ndarray,
) -> dict[str, object]:
    return {
        "schema": LINEAR_SIGNAL_ARTIFACT_SCHEMA,
        "path": path,
        "n_rows": metadata.n_rows,
        "dtype": str(dtype),
        "fields": list(linear_signal_array_fields()),
        "metadata": metadata.as_dict(),
        "first_decision_event_index": int(decision_event_index[0]),
        "last_decision_event_index": int(decision_event_index[-1]),
        "first_decision_local_ts_us": int(decision_local_ts_us[0]),
        "last_decision_local_ts_us": int(decision_local_ts_us[-1]),
        "first_decision_event_seq": int(decision_event_seq[0]),
        "last_decision_event_seq": int(decision_event_seq[-1]),
    }


def _validate_stream_identity(decision_schedule: Mapping[str, object], transform_config: Mapping[str, object], result: LinearTrainResult) -> None:
    if not _payloads_equal(transform_config, result.transform_config):
        raise ValueError(
            "feature_dataset transform_config does not match linear_train_result transform_config; "
            "build features with transform_config_from_train_result(linear_train_result)"
        )
    if not _payloads_equal(decision_schedule, result.decision_schedule):
        raise ValueError(
            "feature_dataset decision_schedule does not match linear_train_result decision_schedule; "
            "build features from the same decision grid lineage as the linear_train_result"
        )


def build_linear_signal_artifact_npz_from_execution_feature_chunks(
    *,
    tape: ExecutionTape,
    decision_grid: DecisionGrid,
    output_npz: str | Path,
    linear_train_result: LinearTrainResult,
    chunk_rows: int = 100_000,
    signal_config: LinearSignalConfig = LinearSignalConfig(),
    output_dtype: str = "float32",
    transform_config: TransformConfig | None = None,
    overwrite: bool = False,
) -> LinearSignalDiskBuildResult:
    if not isinstance(linear_train_result, LinearTrainResult):
        raise ValueError("linear_train_result must be LinearTrainResult")
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(decision_grid, DecisionGrid):
        raise ValueError("decision_grid must be DecisionGrid")
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    dtype = np.dtype(_require_output_dtype(output_dtype))
    transform = transform_config if transform_config is not None else TransformConfig()
    output_path = Path(output_npz)
    if output_path.exists() and not overwrite:
        raise FileExistsError(str(output_path))
    _validate_stream_identity(decision_grid.decision_schedule, transform.as_dict(), linear_train_result)
    if getattr(linear_train_result, "decision_grid_hash", None) != decision_grid.decision_grid_hash:
        raise ValueError("linear_train_result decision_grid_hash does not match decision_grid")
    stats = _StreamStats(
        feature_names=execution_linear_feature_names(),
        replay_start_event_index=0,
        decision_grid_schema=decision_grid.metadata.schema,
        decision_grid_hash=decision_grid.decision_grid_hash,
        decision_grid_n_rows=decision_grid.n_rows,
        decision_schedule=decision_grid.decision_schedule,
        transform_config=transform.as_dict(),
    )
    model_bundle = linear_model_bundle_from_train_result(linear_train_result)
    preprocess_states = linear_preprocess_states_from_train_result(linear_train_result)
    writers = _LinearSignalChunkedWriters.create(output_path, chunk_rows=chunk_rows, output_dtype=output_dtype)
    finalized: _FinalizedArrays | None = None
    try:
        for chunk in iter_execution_linear_feature_chunks_for_decision_grid(
            tape,
            decision_grid=decision_grid,
            chunk_rows=chunk_rows,
            output_dtype=output_dtype,
            transform_config=transform,
        ):
            dataset = _feature_dataset_from_chunk(
                chunk,
                replay_start_event_index=stats.replay_start_event_index,
                decision_grid=decision_grid,
                decision_schedule=stats.decision_schedule,
                transform_config=stats.transform_config,
            )
            stats.update(dataset)
            predictions = predict_linear_heads_for_execution_features(
                feature_dataset=dataset,
                model_bundle=model_bundle,
                preprocess_states_by_head=preprocess_states,
                output_dtype=output_dtype,
            )
            writers.append(dataset, predictions_to_signal_arrays(predictions, config=signal_config, output_dtype=output_dtype))
        stats.require_nonempty()
        metadata = LinearSignalArtifactMetadata(
            tape_schema=tape.manifest.schema,
            exchange=tape.manifest.exchange,
            symbol=tape.manifest.symbol,
            num_events=tape.manifest.num_events,
            num_l2_batches=tape.manifest.num_l2_batches,
            num_trades=tape.manifest.num_trades,
            start_local_ts_us=tape.manifest.start_local_ts_us,
            end_local_ts_us=tape.manifest.end_local_ts_us,
            decision_grid_schema=stats.decision_grid_schema,
            decision_grid_hash=stats.decision_grid_hash,
            decision_grid_n_rows=stats.decision_grid_n_rows,
            decision_schedule=dict(stats.decision_schedule),
            start_event_index=stats.start_event_index,
            n_rows=stats.num_decisions,
        )
        finalized = writers.finalize()
        if finalized.row_counts["decision_event_index"] != stats.num_decisions:
            raise RuntimeError("linear signal chunk row count changed during finalize")
        save_linear_signal_artifact_arrays(
            output_path,
            metadata=metadata,
            decision_event_index=finalized.decision_event_index,
            decision_local_ts_us=finalized.decision_local_ts_us,
            decision_event_seq=finalized.decision_event_seq,
            arrays=finalized.arrays,
            overwrite=overwrite,
            validate_chunk_rows=chunk_rows,
        )
        prediction_summary = linear_signal_array_prediction_summary(
            finalized.arrays,
            n_rows=stats.num_decisions,
            chunk_rows=chunk_rows,
            temp_prefix=output_path.parent / f".{output_path.name}",
        )
        return LinearSignalDiskBuildResult(
            output_npz=output_path,
            feature_dataset_summary=stats.feature_dataset_summary(),
            linear_signals_summary=_linear_signal_summary_from_arrays(
                path=str(output_path),
                metadata=metadata,
                dtype=dtype,
                decision_event_index=finalized.decision_event_index,
                decision_local_ts_us=finalized.decision_local_ts_us,
                decision_event_seq=finalized.decision_event_seq,
            ),
            alignment_summary={
                "replay_start_event_index": stats.replay_start_event_index,
                "first_signal_event_index": stats.first_decision_event_index,
                "first_signal_local_ts_us": stats.first_decision_local_ts_us,
                "first_signal_event_seq": stats.first_decision_event_seq,
                "decision_grid_schema": stats.decision_grid_schema,
                "decision_grid_hash": stats.decision_grid_hash,
                "decision_grid_n_rows": stats.decision_grid_n_rows,
                "n_signal_rows": stats.num_decisions,
            },
            prediction_summary=prediction_summary,
        )
    finally:
        if finalized is not None:
            finalized.close()
        writers.cleanup()


__all__ = [
    "ExecutionLinearFeatureDataset",
    "ExecutionLinearFeatureChunk",
    "iter_execution_linear_feature_chunks_for_decision_grid",
    "execution_linear_feature_dataset_summary",
    "execution_linear_feature_names",
    "LinearSignalBuildResult",
    "transform_config_from_train_result",
    "predict_linear_heads_for_execution_features",
    "build_linear_signal_build_result",
    "build_linear_signal_artifact_from_execution_features",
    "linear_prediction_summary",
    "LinearSignalDiskBuildResult",
    "linear_signal_array_prediction_summary",
    "build_linear_signal_artifact_npz_from_execution_feature_chunks",
]
