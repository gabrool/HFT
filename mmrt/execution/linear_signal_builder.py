"""Build canonical linear signal artifacts from execution-tape feature replay.

This module owns execution-tape feature generation and conversion into aligned
linear signals. It consumes execution tapes, trained linear artifacts, and the
shared causal FeatureEngine; it does not read labels, storage datasets, RL code,
or adverse-selection components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from mmrt.execution.execution_tape import (
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    ExecutionTape,
)
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    LinearSignalConfig,
    predictions_to_signal_arrays,
    validate_linear_signal_artifact_metadata,
)
from mmrt.features.book_state import BOOK_DEPTH, BookSnapshotInput
from mmrt.features.engine import FeatureEngine, FeatureEngineConfig
from mmrt.features.specs import FEATURE_NAMES
from mmrt.features.trade_state import TradeInput
from mmrt.linear import models as lm
from mmrt.linear import preprocess as pp
from mmrt.linear.train import (
    LinearTrainResult,
    linear_model_bundle_from_train_result,
    linear_preprocess_states_from_train_result,
)

_ALLOWED_DTYPES = ("float32", "float64")


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


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


@dataclass(frozen=True, slots=True)
class ExecutionLinearFeatureDataset:
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    replay_start_event_index: int
    start_event_index: int
    decision_interval_us: int

    def __post_init__(self) -> None:
        event_idx = np.ascontiguousarray(np.asarray(self.decision_event_index, dtype=np.int64))
        if event_idx.ndim != 1:
            raise ValueError("decision_event_index must be rank-1")
        if event_idx.size and (event_idx < 0).any():
            raise ValueError("decision_event_index must be nonnegative")
        if event_idx.size > 1 and (np.diff(event_idx) <= 0).any():
            raise ValueError("decision_event_index must be strictly increasing")

        local_ts = np.ascontiguousarray(np.asarray(self.decision_local_ts_us, dtype=np.int64))
        if local_ts.ndim != 1:
            raise ValueError("decision_local_ts_us must be rank-1")
        if local_ts.size and (local_ts <= 0).any():
            raise ValueError("decision_local_ts_us must be positive")
        if local_ts.size > 1 and (np.diff(local_ts) <= 0).any():
            raise ValueError("decision_local_ts_us must be strictly increasing")

        features = np.asarray(self.features)
        if features.ndim != 2:
            raise ValueError("features must be rank-2")
        if features.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("features dtype must be float32 or float64")
        features = np.ascontiguousarray(features)
        if not np.isfinite(features).all():
            raise ValueError("features must be finite")
        if features.shape[0] != event_idx.shape[0] or features.shape[0] != local_ts.shape[0]:
            raise ValueError("row count must match decision arrays")
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
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "replay_start_event_index", replay_start)
        object.__setattr__(self, "start_event_index", start)
        object.__setattr__(self, "decision_interval_us", _require_positive_int(self.decision_interval_us, "decision_interval_us"))

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
        "decision_interval_us": dataset.decision_interval_us,
        "replay_start_event_index": dataset.replay_start_event_index,
        "start_event_index": dataset.start_event_index,
    }


def execution_linear_feature_names() -> tuple[str, ...]:
    return tuple(name if name.startswith("x_") else f"x_{name}" for name in FEATURE_NAMES)


def _book_snapshot_input_from_tape_row(tape: ExecutionTape, *, event_row: np.void) -> BookSnapshotInput:
    book_ptr = int(event_row["book_ptr"])
    if book_ptr < 0:
        raise ValueError("event row does not reference a book snapshot")
    tick_size = float(tape.manifest.symbol_spec.tick_size)
    def padded(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        out = np.zeros(BOOK_DEPTH, dtype=dtype)
        src = np.asarray(values, dtype=dtype)[:BOOK_DEPTH]
        out[: src.shape[0]] = src
        return out

    bid_ticks = padded(tape.arrays.book_bid_ticks[book_ptr], dtype=np.dtype("float64"))
    bid_sizes = padded(tape.arrays.book_bid_sizes[book_ptr], dtype=np.dtype("float64"))
    ask_ticks = padded(tape.arrays.book_ask_ticks[book_ptr], dtype=np.dtype("float64"))
    ask_sizes = padded(tape.arrays.book_ask_sizes[book_ptr], dtype=np.dtype("float64"))
    return BookSnapshotInput(
        local_ts_us=int(event_row["local_ts_us"]),
        ts_us=int(event_row["ts_us"]),
        event_seq=int(event_row["event_seq"]),
        bid_px=bid_ticks * tick_size,
        bid_sz=bid_sizes,
        ask_px=ask_ticks * tick_size,
        ask_sz=ask_sizes,
    )


def _trade_input_from_tape_row(tape: ExecutionTape, *, event_row: np.void) -> TradeInput:
    trade_ptr = int(event_row["trade_ptr"])
    if trade_ptr < 0:
        raise ValueError("event row does not reference a trade")
    trade = tape.arrays.trades[trade_ptr]
    price = int(trade["price_tick"]) * float(tape.manifest.symbol_spec.tick_size)
    return TradeInput(
        local_ts_us=int(event_row["local_ts_us"]),
        ts_us=int(event_row["ts_us"]),
        price=float(price),
        amount=float(trade["amount"]),
        side_code=int(trade["side_code"]),
        event_seq=int(event_row["event_seq"]),
    )


def build_execution_linear_feature_dataset(
    tape: ExecutionTape,
    *,
    decision_interval_us: int = 500_000,
    start_event_index: int | None = None,
    max_decisions: int | None = None,
    output_dtype: str = "float32",
) -> ExecutionLinearFeatureDataset:
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    decision_interval_us = _require_positive_int(decision_interval_us, "decision_interval_us")
    dtype = np.dtype(_require_output_dtype(output_dtype))
    events = tape.arrays.events
    start = 0 if start_event_index is None else _require_nonnegative_int(start_event_index, "start_event_index")
    if start >= len(events):
        raise ValueError("start_event_index must be < len(events)")
    if max_decisions is not None:
        max_decisions = _require_positive_int(max_decisions, "max_decisions")

    engine = FeatureEngine(FeatureEngineConfig(decision_stride_us=decision_interval_us))
    decision_event_index: list[int] = []
    decision_local_ts_us: list[int] = []
    feature_rows: list[np.ndarray] = []

    for event_idx in range(start, len(events)):
        event_row = events[event_idx]
        code = int(event_row["event_type_code"])
        if code == EVENT_TYPE_CODE_TRADE:
            engine.on_trade(_trade_input_from_tape_row(tape, event_row=event_row))
        elif code == EVENT_TYPE_CODE_L2_BATCH:
            decision = engine.on_book_snapshot(_book_snapshot_input_from_tape_row(tape, event_row=event_row))
            if decision is not None:
                decision_event_index.append(event_idx)
                decision_local_ts_us.append(int(decision.local_ts_us))
                feature_rows.append(np.asarray(decision.feature_vector, dtype=dtype))
                if max_decisions is not None and len(feature_rows) >= max_decisions:
                    break
        else:
            raise ValueError(f"unsupported execution event_type_code: {code}")

    names = execution_linear_feature_names()
    if feature_rows:
        features = np.ascontiguousarray(np.vstack(feature_rows), dtype=dtype)
    else:
        features = np.empty((0, len(names)), dtype=dtype)
    effective_start_event_index = int(decision_event_index[0]) if decision_event_index else start
    return ExecutionLinearFeatureDataset(
        decision_event_index=np.asarray(decision_event_index, dtype=np.int64),
        decision_local_ts_us=np.asarray(decision_local_ts_us, dtype=np.int64),
        features=features,
        feature_names=names,
        replay_start_event_index=start,
        start_event_index=effective_start_event_index,
        decision_interval_us=decision_interval_us,
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
        decision_interval_us=feature_dataset.decision_interval_us,
        start_event_index=feature_dataset.start_event_index,
        n_rows=arrays.n_rows,
    )
    artifact = LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=feature_dataset.decision_event_index,
        decision_local_ts_us=feature_dataset.decision_local_ts_us,
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
        decision_interval_us=feature_dataset.decision_interval_us,
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
        if include_std:
            out["std"] = None
        else:
            out.update({"p01": None, "p50": None, "p99": None})
        return out
    out = {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
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


__all__ = [
    "ExecutionLinearFeatureDataset",
    "execution_linear_feature_dataset_summary",
    "execution_linear_feature_names",
    "build_execution_linear_feature_dataset",
    "LinearSignalBuildResult",
    "predict_linear_heads_for_execution_features",
    "build_linear_signal_build_result",
    "build_linear_signal_artifact_from_execution_features",
    "linear_prediction_summary",
]
