"""Audit execution-tape linear features against trained linear preprocess state."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.linear_signal import load_linear_signal_artifact_npz, linear_signal_artifact_summary, NO_MOVE_PROBA_KEY, DIRECTION_PROBA_KEY, MAGNITUDE_UP_KEY, MAGNITUDE_DOWN_KEY
from mmrt.execution.linear_signal_builder import build_execution_linear_feature_dataset, execution_linear_feature_dataset_summary
from mmrt.linear import models as lm
from mmrt.linear.train import load_linear_train_result, linear_model_bundle_from_train_result, linear_preprocess_states_from_train_result


def _require_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _opt_nonneg(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _opt_pos(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class LinearExecutionFeatureAuditConfig:
    tape_root: str
    linear_train_result_json: str
    linear_signals_npz: str | None = None
    output_json: str | None = None
    mmap_mode: str | None = "r"
    decision_interval_us: int = 500_000
    start_event_index: int | None = None
    max_decisions: int | None = None
    chunk_rows: int = 100_000
    z_thresholds: tuple[float, ...] = (3.0, 5.0, 8.0)
    top_k: int = 25

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_path(self.tape_root, "tape_root"))
        object.__setattr__(self, "linear_train_result_json", _require_path(self.linear_train_result_json, "linear_train_result_json"))
        if self.linear_signals_npz is not None:
            object.__setattr__(self, "linear_signals_npz", _require_path(self.linear_signals_npz, "linear_signals_npz"))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", _require_path(self.output_json, "output_json"))
        if self.mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'")
        if isinstance(self.decision_interval_us, bool) or self.decision_interval_us <= 0:
            raise ValueError("decision_interval_us must be positive")
        object.__setattr__(self, "start_event_index", _opt_nonneg(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "max_decisions", _opt_pos(self.max_decisions, "max_decisions"))
        if isinstance(self.chunk_rows, bool) or self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        thresholds = tuple(float(x) for x in self.z_thresholds)
        if not thresholds or any(x <= 0.0 or not np.isfinite(x) for x in thresholds) or tuple(sorted(thresholds)) != thresholds:
            raise ValueError("z_thresholds must be positive, sorted, and non-empty")
        object.__setattr__(self, "z_thresholds", thresholds)
        if isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("top_k must be positive")


def _stats(values: np.ndarray) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {k: None for k in ("mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max")}
    q = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)), "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "max": float(np.max(x))}


def _head_model(bundle, head: str):
    if head == lm.NO_MOVE_HEAD:
        return bundle.no_move
    if head == lm.DIRECTION_HEAD:
        return bundle.direction
    if head == lm.MAGNITUDE_UP_HEAD:
        return bundle.magnitude_up
    if head == lm.MAGNITUDE_DOWN_HEAD:
        return bundle.magnitude_down
    raise ValueError("unknown head")


def _prediction_for_head(model, head: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = model.decision_function(z).astype(np.float64)
    if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD):
        proba = model.predict_proba(z).astype(np.float64)[:, 1]
    else:
        proba = model.predict_nonnegative(z).astype(np.float64)
    return logits, proba


def _audit_head(*, head: str, features: np.ndarray, feature_names: tuple[str, ...], state, model, thresholds: tuple[float, ...], top_k: int) -> tuple[dict[str, object], list[str], list[str]]:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    missing = [c for c in state.feature_columns if c not in name_to_idx]
    extra = [c for c in feature_names if c not in state.feature_columns]
    if missing:
        return {"feature_columns": list(state.feature_columns), "n_features": len(state.feature_columns), "n_rows": int(features.shape[0]), "warnings": ["feature_schema_mismatch"], "missing_features": missing}, missing, extra
    idx = np.asarray([name_to_idx[c] for c in state.feature_columns], dtype=np.int64)
    X = np.asarray(features[:, idx], dtype=np.float64)
    Z = (X - state.mean) / state.scale
    Z[:, ~state.active_mask] = 0.0
    clip_z = float(state.config.clip_z)
    clipped = np.clip(Z, -clip_z, clip_z)
    logits, pred = _prediction_for_head(model, head, clipped)
    absz = np.abs(Z)
    clip_frac = np.mean(absz > clip_z, axis=0) if X.size else np.zeros(X.shape[1])
    out: dict[str, object] = {
        "feature_columns": list(state.feature_columns),
        "n_features": len(state.feature_columns),
        "n_rows": int(features.shape[0]),
        "raw_feature_stats_by_feature": {name: _stats(X[:, i]) for i, name in enumerate(state.feature_columns)},
        "z_feature_stats_by_feature": {name: _stats(Z[:, i]) for i, name in enumerate(state.feature_columns)},
        "clip_fraction_by_feature": {name: float(clip_frac[i]) for i, name in enumerate(state.feature_columns)},
        "top_abs_z_mean_features": [{"feature": state.feature_columns[i], "abs_z_mean": float(np.mean(absz[:, i])) if absz.size else 0.0} for i in np.argsort(-(np.mean(absz, axis=0) if absz.size else np.zeros(X.shape[1])))[:top_k]],
        "top_clip_fraction_features": [{"feature": state.feature_columns[i], "clip_fraction": float(clip_frac[i])} for i in np.argsort(-clip_frac)[:top_k]],
        "logit_stats": _stats(logits),
        "prediction_stats": _stats(pred),
    }
    for threshold in thresholds:
        out[f"fraction_abs_z_gt_{threshold:g}"] = {name: float(np.mean(absz[:, i] > threshold)) if absz.size else 0.0 for i, name in enumerate(state.feature_columns)}
    return out, missing, extra


def audit_linear_execution_features_from_config(config: LinearExecutionFeatureAuditConfig) -> dict[str, object]:
    if not isinstance(config, LinearExecutionFeatureAuditConfig):
        raise ValueError("config must be LinearExecutionFeatureAuditConfig")
    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode, validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY)
    result = load_linear_train_result(config.linear_train_result_json)
    bundle = linear_model_bundle_from_train_result(result)
    states = linear_preprocess_states_from_train_result(result)
    features = build_execution_linear_feature_dataset(tape, decision_interval_us=config.decision_interval_us, start_event_index=config.start_event_index, max_decisions=config.max_decisions)
    signals_summary = None
    if config.linear_signals_npz is not None and Path(config.linear_signals_npz).exists():
        artifact = load_linear_signal_artifact_npz(config.linear_signals_npz, mmap_mode=config.mmap_mode)
        signals_summary = linear_signal_artifact_summary(artifact, path=config.linear_signals_npz)
    per_head: dict[str, object] = {}
    all_missing: set[str] = set(); all_extra: set[str] = set(); warnings: list[str] = []
    for head in lm.MODEL_HEADS:
        summary, missing, extra = _audit_head(head=head, features=features.features, feature_names=features.feature_names, state=states[head], model=_head_model(bundle, head), thresholds=config.z_thresholds, top_k=config.top_k)
        per_head[head] = summary
        all_missing.update(missing); all_extra.update(extra)
        if missing:
            warnings.append("feature_schema_mismatch")
        for threshold in config.z_thresholds:
            if threshold >= 8.0:
                vals = summary.get(f"fraction_abs_z_gt_{threshold:g}", {})
                if isinstance(vals, Mapping) and any(float(v) > 0.01 for v in vals.values()):
                    warnings.append("high_z_fraction")
        clips = summary.get("clip_fraction_by_feature", {})
        if isinstance(clips, Mapping) and any(float(v) > 0.01 for v in clips.values()):
            warnings.append("high_clip_fraction")
    no_move_stats = per_head.get(lm.NO_MOVE_HEAD, {}).get("prediction_stats", {}) if isinstance(per_head.get(lm.NO_MOVE_HEAD), Mapping) else {}
    if isinstance(no_move_stats, Mapping) and ((no_move_stats.get("mean") is not None and float(no_move_stats["mean"]) > 0.98) or (no_move_stats.get("p01") is not None and float(no_move_stats["p01"]) > 0.95)):
        warnings.append("p_no_move_collapsed")
    manifest = tape.manifest
    payload = {
        "status": "ok" if not warnings else "warning",
        "run_type": "audit_linear_execution_features",
        "config": asdict(config),
        "tape": {"schema": manifest.schema, "exchange": manifest.exchange, "symbol": manifest.symbol, "num_events": manifest.num_events, "num_l2_batches": manifest.num_l2_batches, "num_trades": manifest.num_trades, "start_local_ts_us": manifest.start_local_ts_us, "end_local_ts_us": manifest.end_local_ts_us},
        "linear_train_result": {"schema": result.schema, "dataset_id": result.dataset_id, "manifest_hash": result.manifest_hash, "splits": {k: v.as_dict() for k, v in result.splits.items()}, "selection_summary": result.selection_summary},
        "linear_signals": signals_summary,
        "feature_dataset": execution_linear_feature_dataset_summary(features),
        "per_head": per_head,
        "combined": {"feature_schema_match": not all_missing, "missing_features": sorted(all_missing), "extra_features": sorted(all_extra), "shared_feature_count": len(set(features.feature_names) - all_missing), "warnings": sorted(set(warnings))},
        "warnings": sorted(set(warnings)),
    }
    if config.output_json is not None:
        path = Path(config.output_json); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        json.dumps(payload, allow_nan=False)
    return payload
