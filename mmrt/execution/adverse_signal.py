"""Adverse-selection model and aligned signal artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.adverse_selection import AdverseSelectionDataset, AdverseSelectionFeatureDataset
from mmrt.execution.decision_grid import DecisionGrid

ADVERSE_SELECTION_MODEL_SCHEMA = "mmrt_adverse_selection_ridge_grid_v1"
ADVERSE_SELECTION_SIGNALS_SCHEMA = "mmrt_adverse_selection_signals_grid_v1"
ADVERSE_SELECTION_MODEL_FILENAME = "adverse_selection_model.npz"
ADVERSE_SELECTION_SIGNALS_FILENAME = "adverse_selection_signals.npz"


def _names_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    out = tuple(str(v) for v in values)
    if not out or any(not v for v in out):
        raise ValueError(f"{name} must be non-empty strings")
    return out


def _finite_1d(arr: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ValueError(f"{name} must be 1D ndarray")
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _hash64(value: str, name: str) -> str:
    out = str(value)
    if len(out) != 64 or any(ch not in "0123456789abcdef" for ch in out):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return out


def _nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class AdverseSelectionModelArtifact:
    schema: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    config_json: str
    exchange: str
    symbol: str
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema != ADVERSE_SELECTION_MODEL_SCHEMA:
            raise ValueError(f"schema must be {ADVERSE_SELECTION_MODEL_SCHEMA!r}")
        object.__setattr__(self, "feature_names", _names_tuple(self.feature_names, "feature_names"))
        object.__setattr__(self, "target_names", _names_tuple(self.target_names, "target_names"))
        fm = _finite_1d(self.feature_mean, "feature_mean")
        fs = _finite_1d(self.feature_scale, "feature_scale")
        coef = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        intercepts = _finite_1d(self.intercepts, "intercepts")
        if coef.ndim != 2 or not np.isfinite(coef).all():
            raise ValueError("coefficients must be finite rank-2")
        if len(self.feature_names) != fm.shape[0] or len(self.feature_names) != fs.shape[0] or coef.shape[1] != len(self.feature_names):
            raise ValueError("feature dimensions must match")
        if len(self.target_names) != coef.shape[0] or len(self.target_names) != intercepts.shape[0]:
            raise ValueError("target dimensions must match")
        if (fs <= 0.0).any():
            raise ValueError("feature_scale must be > 0")
        if not isinstance(self.config_json, str):
            raise ValueError("config_json must be string")
        object.__setattr__(self, "exchange", _nonempty_str(self.exchange, "exchange"))
        object.__setattr__(self, "symbol", _nonempty_str(self.symbol, "symbol"))
        object.__setattr__(self, "decision_grid_schema", _nonempty_str(self.decision_grid_schema, "decision_grid_schema"))
        object.__setattr__(self, "decision_grid_hash", _hash64(self.decision_grid_hash, "decision_grid_hash"))
        if isinstance(self.decision_grid_n_rows, bool) or int(self.decision_grid_n_rows) <= 0:
            raise ValueError("decision_grid_n_rows must be a positive int")
        object.__setattr__(self, "decision_grid_n_rows", int(self.decision_grid_n_rows))
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")
        object.__setattr__(self, "decision_schedule", dict(self.decision_schedule))
        object.__setattr__(self, "feature_mean", fm)
        object.__setattr__(self, "feature_scale", fs)
        object.__setattr__(self, "coefficients", coef)
        object.__setattr__(self, "intercepts", intercepts)


def save_adverse_selection_model(path: str | Path, artifact: AdverseSelectionModelArtifact, *, overwrite: bool = False) -> None:
    if not isinstance(artifact, AdverseSelectionModelArtifact):
        raise ValueError("artifact must be AdverseSelectionModelArtifact")
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            schema=np.array(artifact.schema),
            feature_names=np.asarray(artifact.feature_names, dtype=object),
            target_names=np.asarray(artifact.target_names, dtype=object),
            feature_mean=artifact.feature_mean,
            feature_scale=artifact.feature_scale,
            coefficients=artifact.coefficients,
            intercepts=artifact.intercepts,
            config_json=np.array(artifact.config_json),
            exchange=np.array(artifact.exchange),
            symbol=np.array(artifact.symbol),
            decision_grid_schema=np.array(artifact.decision_grid_schema),
            decision_grid_hash=np.array(artifact.decision_grid_hash),
            decision_grid_n_rows=np.array(artifact.decision_grid_n_rows, dtype=np.int64),
            decision_schedule=np.array(dict(artifact.decision_schedule), dtype=object),
        )
    tmp.replace(path)


def load_adverse_selection_model(path: str | Path) -> AdverseSelectionModelArtifact:
    with np.load(Path(path), allow_pickle=True) as data:
        return AdverseSelectionModelArtifact(
            schema=str(data["schema"].item()),
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            target_names=tuple(str(x) for x in data["target_names"].tolist()),
            feature_mean=np.asarray(data["feature_mean"]),
            feature_scale=np.asarray(data["feature_scale"]),
            coefficients=np.asarray(data["coefficients"]),
            intercepts=np.asarray(data["intercepts"]),
            config_json=str(data["config_json"].item()),
            exchange=str(data["exchange"].item()),
            symbol=str(data["symbol"].item()),
            decision_grid_schema=str(data["decision_grid_schema"].item()),
            decision_grid_hash=str(data["decision_grid_hash"].item()),
            decision_grid_n_rows=int(data["decision_grid_n_rows"].item()),
            decision_schedule=dict(data["decision_schedule"].item()),
        )


def predict_adverse_selection(artifact: AdverseSelectionModelArtifact, features: np.ndarray, *, output_dtype: str = "float32") -> dict[str, np.ndarray]:
    if not isinstance(artifact, AdverseSelectionModelArtifact):
        raise ValueError("artifact must be AdverseSelectionModelArtifact")
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(artifact.feature_names) or not np.isfinite(X).all():
        raise ValueError("features must be finite rank-2 with matching width")
    raw = (X - artifact.feature_mean) / artifact.feature_scale @ artifact.coefficients.T + artifact.intercepts
    out: dict[str, np.ndarray] = {}
    for i, target in enumerate(artifact.target_names):
        pred = raw[:, i]
        if target.endswith("_filled") or target.endswith("_toxic_fill"):
            pred = np.clip(pred, 0.0, 1.0)
        elif target.endswith("_toxic_cost_bps") or target.endswith("_adverse_bps") or target.endswith("_fill_latency_us"):
            pred = np.maximum(pred, 0.0)
        out[target] = np.ascontiguousarray(pred.astype(output_dtype, copy=False))
    return out


@dataclass(frozen=True, slots=True)
class AdverseSelectionSignalArtifact:
    schema: str
    decision_local_ts_us: np.ndarray
    decision_event_index: np.ndarray
    decision_event_seq: np.ndarray
    target_names: tuple[str, ...]
    predictions: dict[str, np.ndarray]
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema != ADVERSE_SELECTION_SIGNALS_SCHEMA:
            raise ValueError(f"schema must be {ADVERSE_SELECTION_SIGNALS_SCHEMA!r}")
        arrays = [np.ascontiguousarray(a, dtype=np.int64) for a in (self.decision_local_ts_us, self.decision_event_index, self.decision_event_seq)]
        if any(a.ndim != 1 for a in arrays):
            raise ValueError("decision arrays must be 1D")
        n = arrays[0].shape[0]
        if any(a.shape[0] != n for a in arrays):
            raise ValueError("decision arrays must have same length")
        names = _names_tuple(self.target_names, "target_names")
        if not isinstance(self.predictions, Mapping):
            raise ValueError("predictions must be a mapping")
        preds: dict[str, np.ndarray] = {}
        for name in names:
            if name not in self.predictions:
                raise ValueError(f"missing prediction array for target {name!r}")
            arr = np.ascontiguousarray(self.predictions[name], dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != n or not np.isfinite(arr).all():
                raise ValueError("prediction arrays must be finite and aligned")
            if (name.endswith("_filled") or name.endswith("_toxic_fill")) and ((arr < 0.0) | (arr > 1.0)).any():
                raise ValueError("probability predictions must be in [0, 1]")
            if (name.endswith("_toxic_cost_bps") or name.endswith("_adverse_bps") or name.endswith("_fill_latency_us")) and (arr < 0.0).any():
                raise ValueError("cost predictions must be >= 0")
            preds[name] = arr
        object.__setattr__(self, "decision_local_ts_us", arrays[0])
        object.__setattr__(self, "decision_event_index", arrays[1])
        object.__setattr__(self, "decision_event_seq", arrays[2])
        object.__setattr__(self, "target_names", names)
        object.__setattr__(self, "predictions", preds)
        object.__setattr__(self, "decision_grid_schema", _nonempty_str(self.decision_grid_schema, "decision_grid_schema"))
        object.__setattr__(self, "decision_grid_hash", _hash64(self.decision_grid_hash, "decision_grid_hash"))
        if int(self.decision_grid_n_rows) != n:
            raise ValueError("decision_grid_n_rows must match signal rows")
        object.__setattr__(self, "decision_grid_n_rows", int(self.decision_grid_n_rows))
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")
        object.__setattr__(self, "decision_schedule", dict(self.decision_schedule))


def build_adverse_selection_signal_artifact(dataset: AdverseSelectionDataset | AdverseSelectionFeatureDataset, model: AdverseSelectionModelArtifact) -> AdverseSelectionSignalArtifact:
    if not isinstance(dataset, (AdverseSelectionDataset, AdverseSelectionFeatureDataset)):
        raise ValueError("dataset must be AdverseSelectionDataset or AdverseSelectionFeatureDataset")
    if tuple(dataset.feature_names) != tuple(model.feature_names):
        raise ValueError("dataset feature_names must match model feature_names exactly")
    predictions = predict_adverse_selection(model, dataset.features)
    return AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=dataset.decision_local_ts_us.copy(),
        decision_event_index=dataset.decision_event_index.copy(),
        decision_event_seq=dataset.decision_event_seq.copy(),
        target_names=model.target_names,
        predictions=predictions,
        decision_grid_schema=model.decision_grid_schema,
        decision_grid_hash=model.decision_grid_hash,
        decision_grid_n_rows=model.decision_grid_n_rows,
        decision_schedule=model.decision_schedule,
    )


def save_adverse_selection_signals(path: str | Path, artifact: AdverseSelectionSignalArtifact, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"pred_{name}": arr for name, arr in artifact.predictions.items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            schema=np.array(artifact.schema),
            decision_local_ts_us=artifact.decision_local_ts_us,
            decision_event_index=artifact.decision_event_index,
            decision_event_seq=artifact.decision_event_seq,
            target_names=np.asarray(artifact.target_names, dtype=object),
            decision_grid_schema=np.array(artifact.decision_grid_schema),
            decision_grid_hash=np.array(artifact.decision_grid_hash),
            decision_grid_n_rows=np.array(artifact.decision_grid_n_rows, dtype=np.int64),
            decision_schedule=np.array(dict(artifact.decision_schedule), dtype=object),
            **payload,
        )
    tmp.replace(path)



def _is_probability_target(name: str) -> bool:
    return name.endswith("_filled") or name.endswith("_toxic_fill")


def _is_nonnegative_target(name: str) -> bool:
    return name.endswith("_toxic_cost_bps") or name.endswith("_adverse_bps") or name.endswith("_fill_latency_us")


def save_adverse_selection_signals_arrays(
    path: str | Path,
    *,
    decision_local_ts_us: np.ndarray,
    decision_event_index: np.ndarray,
    decision_event_seq: np.ndarray,
    target_names: Sequence[str],
    predictions: Mapping[str, np.ndarray],
    decision_grid_schema: str,
    decision_grid_hash: str,
    decision_grid_n_rows: int,
    decision_schedule: Mapping[str, object],
    overwrite: bool = False,
    validate_chunk_rows: int = 100_000,
) -> None:
    """Save an adverse-selection signal NPZ from array-like inputs without artifact copies."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if isinstance(validate_chunk_rows, bool) or int(validate_chunk_rows) <= 0:
        raise ValueError("validate_chunk_rows must be a positive int")
    validate_chunk_rows = int(validate_chunk_rows)
    names = _names_tuple(target_names, "target_names")
    decision_grid_schema = _nonempty_str(decision_grid_schema, "decision_grid_schema")
    decision_grid_hash = _hash64(decision_grid_hash, "decision_grid_hash")
    if not isinstance(decision_schedule, Mapping):
        raise ValueError("decision_schedule must be a mapping")

    decision_arrays = {
        "decision_local_ts_us": np.asarray(decision_local_ts_us),
        "decision_event_index": np.asarray(decision_event_index),
        "decision_event_seq": np.asarray(decision_event_seq),
    }
    n: int | None = None
    for name, arr in decision_arrays.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
        if not np.can_cast(arr.dtype, np.int64, casting="same_kind") and arr.dtype.kind not in "iu":
            raise ValueError(f"{name} must be int64-compatible")
        if n is None:
            n = int(arr.shape[0])
        elif int(arr.shape[0]) != n:
            raise ValueError("decision arrays must have the same length")
    assert n is not None
    if int(decision_grid_n_rows) != n:
        raise ValueError("decision_grid_n_rows must match signal rows")

    pred_arrays: dict[str, np.ndarray] = {}
    missing = [name for name in names if name not in predictions]
    if missing:
        raise ValueError(f"missing prediction arrays: {missing}")
    for target in names:
        arr = np.asarray(predictions[target])
        if arr.ndim != 1 or int(arr.shape[0]) != n:
            raise ValueError(f"pred_{target} must be rank-1 with length {n}")
        pred_arrays[target] = arr

    for start in range(0, n, validate_chunk_rows):
        end = min(start + validate_chunk_rows, n)
        for name, arr in decision_arrays.items():
            # Conversion validates integer compatibility for the chunk only.
            np.asarray(arr[start:end], dtype=np.int64)
        for target, arr in pred_arrays.items():
            chunk = np.asarray(arr[start:end], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError(f"pred_{target} must be finite")
            if _is_probability_target(target):
                if ((chunk < 0.0) | (chunk > 1.0)).any():
                    raise ValueError(f"pred_{target} must be in [0, 1]")
            elif _is_nonnegative_target(target):
                if (chunk < 0.0).any():
                    raise ValueError(f"pred_{target} must be >= 0")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"pred_{name}": pred_arrays[name] for name in names}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            schema=np.array(ADVERSE_SELECTION_SIGNALS_SCHEMA),
            decision_local_ts_us=decision_arrays["decision_local_ts_us"],
            decision_event_index=decision_arrays["decision_event_index"],
            decision_event_seq=decision_arrays["decision_event_seq"],
            target_names=np.asarray(names, dtype=object),
            decision_grid_schema=np.array(decision_grid_schema),
            decision_grid_hash=np.array(decision_grid_hash),
            decision_grid_n_rows=np.array(int(decision_grid_n_rows), dtype=np.int64),
            decision_schedule=np.array(dict(decision_schedule), dtype=object),
            **payload,
        )
    tmp.replace(path)


def load_adverse_selection_signals(path: str | Path) -> AdverseSelectionSignalArtifact:
    with np.load(Path(path), allow_pickle=True) as data:
        required_keys = (
            "schema",
            "decision_local_ts_us",
            "decision_event_index",
            "decision_event_seq",
            "target_names",
            "decision_grid_schema",
            "decision_grid_hash",
            "decision_grid_n_rows",
            "decision_schedule",
        )
        missing_base = [key for key in required_keys if key not in data.files]
        if missing_base:
            raise ValueError(f"missing required arrays in adverse-selection signals artifact: {missing_base}")
        target_names = tuple(str(x) for x in data["target_names"].tolist())
        predictions: dict[str, np.ndarray] = {}
        missing_predictions: list[str] = []
        for name in target_names:
            key = f"pred_{name}"
            if key not in data.files:
                missing_predictions.append(key)
                continue
            predictions[name] = np.asarray(data[key])
        if missing_predictions:
            raise ValueError(f"missing prediction arrays in adverse-selection signals artifact: {missing_predictions}")
        return AdverseSelectionSignalArtifact(
            schema=str(data["schema"].item()),
            decision_local_ts_us=np.asarray(data["decision_local_ts_us"]),
            decision_event_index=np.asarray(data["decision_event_index"]),
            decision_event_seq=np.asarray(data["decision_event_seq"]),
            target_names=target_names,
            predictions=predictions,
            decision_grid_schema=str(data["decision_grid_schema"].item()),
            decision_grid_hash=str(data["decision_grid_hash"].item()),
            decision_grid_n_rows=int(data["decision_grid_n_rows"].item()),
            decision_schedule=dict(data["decision_schedule"].item()),
        )



def _coerce_grid_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1")
    try:
        return np.ascontiguousarray(arr, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be int64-compatible") from exc


def validate_decision_grid_alignment(
    *,
    left_local_ts_us: np.ndarray,
    left_event_index: np.ndarray,
    left_event_seq: np.ndarray,
    right_local_ts_us: np.ndarray,
    right_event_index: np.ndarray,
    right_event_seq: np.ndarray,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    left_ts = _coerce_grid_array(left_local_ts_us, f"{left_name}_local_ts_us")
    left_idx = _coerce_grid_array(left_event_index, f"{left_name}_event_index")
    left_seq = _coerce_grid_array(left_event_seq, f"{left_name}_event_seq")
    right_ts = _coerce_grid_array(right_local_ts_us, f"{right_name}_local_ts_us")
    right_idx = _coerce_grid_array(right_event_index, f"{right_name}_event_index")
    right_seq = _coerce_grid_array(right_event_seq, f"{right_name}_event_seq")
    left_len = left_ts.shape[0]
    if left_idx.shape[0] != left_len or left_seq.shape[0] != left_len:
        raise ValueError(f"decision grid mismatch between {left_name} and {right_name}: {left_name} arrays have different lengths")
    right_len = right_ts.shape[0]
    if right_idx.shape[0] != right_len or right_seq.shape[0] != right_len:
        raise ValueError(f"decision grid mismatch between {left_name} and {right_name}: {right_name} arrays have different lengths")
    if left_len != right_len:
        raise ValueError(f"decision grid mismatch between {left_name} and {right_name}: length differs ({left_len} != {right_len})")
    mismatches = np.nonzero((left_ts != right_ts) | (left_idx != right_idx) | (left_seq != right_seq))[0]
    if mismatches.size:
        i = int(mismatches[0])
        raise ValueError(
            f"decision grid mismatch between {left_name} and {right_name}: first mismatch at row {i}: "
            f"{left_name}=({int(left_ts[i])}, {int(left_idx[i])}, {int(left_seq[i])}) "
            f"{right_name}=({int(right_ts[i])}, {int(right_idx[i])}, {int(right_seq[i])})"
        )


def validate_adverse_signal_alignment(
    signals: AdverseSelectionSignalArtifact,
    *,
    decision_local_ts_us: np.ndarray,
    decision_event_index: np.ndarray,
    decision_event_seq: np.ndarray,
    right_name: str,
) -> None:
    if not isinstance(signals, AdverseSelectionSignalArtifact):
        raise ValueError("signals must be AdverseSelectionSignalArtifact")
    validate_decision_grid_alignment(
        left_local_ts_us=signals.decision_local_ts_us,
        left_event_index=signals.decision_event_index,
        left_event_seq=signals.decision_event_seq,
        right_local_ts_us=decision_local_ts_us,
        right_event_index=decision_event_index,
        right_event_seq=decision_event_seq,
        left_name="adverse_signals",
        right_name=right_name,
    )


def validate_adverse_signals_for_decision_grid(
    signals: AdverseSelectionSignalArtifact,
    decision_grid: DecisionGrid,
) -> None:
    if not isinstance(signals, AdverseSelectionSignalArtifact):
        raise ValueError("signals must be AdverseSelectionSignalArtifact")
    if not isinstance(decision_grid, DecisionGrid):
        raise ValueError("decision_grid must be DecisionGrid")
    if signals.decision_grid_schema != decision_grid.metadata.schema:
        raise ValueError("adverse signals decision_grid_schema mismatch")
    if signals.decision_grid_hash != decision_grid.decision_grid_hash:
        raise ValueError("adverse signals decision_grid_hash mismatch")
    if signals.decision_grid_n_rows != decision_grid.n_rows:
        raise ValueError("adverse signals decision_grid_n_rows mismatch")
    validate_adverse_signal_alignment(
        signals,
        decision_local_ts_us=decision_grid.decision_local_ts_us,
        decision_event_index=decision_grid.decision_event_index,
        decision_event_seq=decision_grid.decision_event_seq,
        right_name="decision_grid",
    )

def required_adverse_targets_for_executable_edge(candidate_names: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for c in candidate_names:
        names.extend((f"bid_{c}_filled", f"ask_{c}_filled", f"bid_{c}_toxic_cost_bps", f"ask_{c}_toxic_cost_bps"))
    return tuple(names)


def require_adverse_targets_for_executable_edge(available_target_names: Sequence[str], candidate_names: Sequence[str]) -> None:
    required = required_adverse_targets_for_executable_edge(candidate_names)
    missing = sorted(set(required) - set(available_target_names))
    if missing:
        raise ValueError(f"missing adverse-selection targets required for executable edge: {missing}")
