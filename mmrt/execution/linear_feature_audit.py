"""Audit execution-tape linear features against trained linear preprocess state."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.linear_signal import load_linear_signal_artifact_npz, linear_signal_artifact_summary
from mmrt.execution.linear_signal_builder import iter_execution_linear_feature_chunks, execution_linear_feature_names
from mmrt.linear import models as lm
from mmrt.linear.train import load_linear_train_result, linear_model_bundle_from_train_result, linear_preprocess_states_from_train_result


def _require_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise ValueError(f"{name} must be non-empty")
    return value.strip()

def _opt_nonneg(value: int | None, name: str) -> int | None:
    if value is None: return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError(f"{name} must be nonnegative")
    return value

def _opt_pos(value: int | None, name: str) -> int | None:
    if value is None: return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0: raise ValueError(f"{name} must be positive")
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
    work_dir: str | None = None
    cleanup_work_dir: bool = True
    quantile_mode: str = "exact_memmap"
    max_quantile_samples: int = 1_000_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_path(self.tape_root, "tape_root"))
        object.__setattr__(self, "linear_train_result_json", _require_path(self.linear_train_result_json, "linear_train_result_json"))
        if self.linear_signals_npz is not None: object.__setattr__(self, "linear_signals_npz", _require_path(self.linear_signals_npz, "linear_signals_npz"))
        if self.output_json is not None: object.__setattr__(self, "output_json", _require_path(self.output_json, "output_json"))
        if self.mmap_mode not in (None, "r"): raise ValueError("mmap_mode must be None or 'r'")
        if isinstance(self.decision_interval_us, bool) or self.decision_interval_us <= 0: raise ValueError("decision_interval_us must be positive")
        object.__setattr__(self, "start_event_index", _opt_nonneg(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "max_decisions", _opt_pos(self.max_decisions, "max_decisions"))
        if isinstance(self.chunk_rows, bool) or self.chunk_rows <= 0: raise ValueError("chunk_rows must be positive")
        thresholds = tuple(float(x) for x in self.z_thresholds)
        if not thresholds or any(x <= 0.0 or not np.isfinite(x) for x in thresholds) or tuple(sorted(thresholds)) != thresholds: raise ValueError("z_thresholds must be positive, sorted, and non-empty")
        object.__setattr__(self, "z_thresholds", thresholds)
        if isinstance(self.top_k, bool) or self.top_k <= 0: raise ValueError("top_k must be positive")
        if self.work_dir is not None: object.__setattr__(self, "work_dir", _require_path(self.work_dir, "work_dir"))
        if not isinstance(self.cleanup_work_dir, bool): raise ValueError("cleanup_work_dir must be bool")
        if self.quantile_mode not in ("exact_memmap", "reservoir"): raise ValueError("quantile_mode must be exact_memmap or reservoir")
        if isinstance(self.max_quantile_samples, bool) or self.max_quantile_samples <= 0: raise ValueError("max_quantile_samples must be positive")


def _empty_stats() -> dict[str, object]: return {k: None for k in ("mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max")}
def _stats(values: np.ndarray) -> dict[str, object]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0: return _empty_stats()
    q = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)), "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "max": float(np.max(x))}

def _head_model(bundle, head: str):
    if head == lm.NO_MOVE_HEAD: return bundle.no_move
    if head == lm.DIRECTION_HEAD: return bundle.direction
    if head == lm.MAGNITUDE_UP_HEAD: return bundle.magnitude_up
    if head == lm.MAGNITUDE_DOWN_HEAD: return bundle.magnitude_down
    raise ValueError("unknown head")

def _prediction_for_head(model, head: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = model.decision_function(z).astype(np.float64)
    if head in (lm.NO_MOVE_HEAD, lm.DIRECTION_HEAD): proba = model.predict_proba(z).astype(np.float64)[:, 1]
    else: proba = model.predict_nonnegative(z).astype(np.float64)
    return logits, proba

def _feature_dataset_summary(n: int, nf: int, names: tuple[str, ...], idx: np.ndarray, ts: np.ndarray, decision_interval_us: int, replay_start: int) -> dict[str, object]:
    return {"num_decisions": n, "num_features": nf, "feature_names": list(names), "first_decision_event_index": int(idx[0]) if n else None, "last_decision_event_index": int(idx[-1]) if n else None, "first_decision_local_ts_us": int(ts[0]) if n else None, "last_decision_local_ts_us": int(ts[-1]) if n else None, "decision_interval_us": decision_interval_us, "replay_start_event_index": replay_start, "start_event_index": int(idx[0]) if n else replay_start}

def audit_linear_execution_features_from_config(config: LinearExecutionFeatureAuditConfig) -> dict[str, object]:
    if not isinstance(config, LinearExecutionFeatureAuditConfig): raise ValueError("config must be LinearExecutionFeatureAuditConfig")
    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode, validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY)
    result = load_linear_train_result(config.linear_train_result_json)
    bundle = linear_model_bundle_from_train_result(result); states = linear_preprocess_states_from_train_result(result)
    work_root = (Path(config.work_dir) if config.work_dir is not None else Path(config.tape_root)) / "linear_execution_feature_audit_work"
    if work_root.exists(): shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    names = execution_linear_feature_names(); nf = len(names)
    idx_parts=[]; ts_parts=[]; n=0
    feature_path = work_root / "features.npy"
    # first stream to chunk files, then concatenate into a single memmap without ever making a float64 matrix
    chunks=[]
    for c in iter_execution_linear_feature_chunks(tape, decision_interval_us=config.decision_interval_us, start_event_index=config.start_event_index, max_decisions=config.max_decisions, chunk_rows=config.chunk_rows):
        path = work_root / f"chunk_{len(chunks):06d}.npy"; np.save(path, c.features); chunks.append(path); idx_parts.append(c.decision_event_index); ts_parts.append(c.decision_local_ts_us); n += c.features.shape[0]
    feats = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(n, nf)); pos=0
    for path in chunks:
        arr=np.load(path, mmap_mode="r"); feats[pos:pos+arr.shape[0]] = arr; pos += arr.shape[0]; path.unlink(missing_ok=True)
    feats.flush(); del feats
    features = np.load(feature_path, mmap_mode="r")
    decision_idx = np.concatenate(idx_parts) if idx_parts else np.asarray([], dtype=np.int64)
    decision_ts = np.concatenate(ts_parts) if ts_parts else np.asarray([], dtype=np.int64)
    signals_summary = None
    if config.linear_signals_npz is not None and Path(config.linear_signals_npz).exists():
        artifact = load_linear_signal_artifact_npz(config.linear_signals_npz, mmap_mode=config.mmap_mode); signals_summary = linear_signal_artifact_summary(artifact, path=config.linear_signals_npz)
    per_head={}; all_missing=set(); all_extra=set(); warnings=[]; name_to_idx={name:i for i,name in enumerate(names)}
    for head in lm.MODEL_HEADS:
        state=states[head]; model=_head_model(bundle, head); missing=[c for c in state.feature_columns if c not in name_to_idx]; extra=[c for c in names if c not in state.feature_columns]
        all_missing.update(missing); all_extra.update(extra)
        if missing:
            per_head[head]={"feature_columns": list(state.feature_columns), "n_features": len(state.feature_columns), "n_rows": n, "warnings": ["feature_schema_mismatch"], "missing_features": missing}; warnings.append("feature_schema_mismatch"); continue
        cols=np.asarray([name_to_idx[c] for c in state.feature_columns], dtype=np.int64); hnf=len(cols)
        raw_stats={}; z_stats={}; clip_frac={}; frac_gt={thr:{} for thr in config.z_thresholds}; absz_sum=np.zeros(hnf); clip_counts=np.zeros(hnf); gt_counts={thr:np.zeros(hnf) for thr in config.z_thresholds}
        for j,col in enumerate(cols):
            raw=np.asarray(features[:, col], dtype=np.float64); z=(raw - state.mean[j])/state.scale[j]
            raw_stats[state.feature_columns[j]]=_stats(raw); z_stats[state.feature_columns[j]]=_stats(z); absz=np.abs(z); absz_sum[j]=float(np.mean(absz)) if n else 0.0; clip_counts[j]=float(np.mean(absz > state.config.clip_z)) if n else 0.0
            for thr in config.z_thresholds: gt_counts[thr][j]=float(np.mean(absz > thr)) if n else 0.0
            del raw, z, absz
        logits_path=work_root/f"{head}_logits.npy"; pred_path=work_root/f"{head}_pred.npy"; logits=np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float64, shape=(n,)); pred=np.lib.format.open_memmap(pred_path, mode="w+", dtype=np.float64, shape=(n,))
        for start in range(0,n,config.chunk_rows):
            end=min(start+config.chunk_rows,n); X=np.asarray(features[start:end][:, cols], dtype=np.float64); Z=(X-state.mean)/state.scale; Z[:, ~state.active_mask]=0.0; C=np.clip(Z, -float(state.config.clip_z), float(state.config.clip_z)); lo,pr=_prediction_for_head(model, head, C); logits[start:end]=lo; pred[start:end]=pr
        logits.flush(); pred.flush()
        for j,name in enumerate(state.feature_columns): clip_frac[name]=float(clip_counts[j])
        summary={"feature_columns": list(state.feature_columns), "n_features": hnf, "n_rows": n, "raw_feature_stats_by_feature": raw_stats, "z_feature_stats_by_feature": z_stats, "clip_fraction_by_feature": clip_frac, "top_abs_z_mean_features": [{"feature": state.feature_columns[i], "abs_z_mean": float(absz_sum[i])} for i in np.argsort(-absz_sum)[:config.top_k]], "top_clip_fraction_features": [{"feature": state.feature_columns[i], "clip_fraction": float(clip_counts[i])} for i in np.argsort(-clip_counts)[:config.top_k]], "logit_stats": _stats(np.asarray(logits)), "prediction_stats": _stats(np.asarray(pred)), "quantile_mode": config.quantile_mode}
        for thr in config.z_thresholds: summary[f"fraction_abs_z_gt_{thr:g}"]={state.feature_columns[i]: float(gt_counts[thr][i]) for i in range(hnf)}
        per_head[head]=summary
        for threshold in config.z_thresholds:
            if threshold >= 8.0 and any(float(v)>0.01 for v in summary[f"fraction_abs_z_gt_{threshold:g}"].values()): warnings.append("high_z_fraction")
        if any(float(v)>0.01 for v in clip_frac.values()): warnings.append("high_clip_fraction")
    no_move_stats = per_head.get(lm.NO_MOVE_HEAD, {}).get("prediction_stats", {}) if isinstance(per_head.get(lm.NO_MOVE_HEAD), Mapping) else {}
    if isinstance(no_move_stats, Mapping) and ((no_move_stats.get("mean") is not None and float(no_move_stats["mean"]) > 0.98) or (no_move_stats.get("p01") is not None and float(no_move_stats["p01"]) > 0.95)): warnings.append("p_no_move_collapsed")
    manifest=tape.manifest; replay_start=0 if config.start_event_index is None else config.start_event_index
    payload={"status": "ok" if not warnings else "warning", "run_type": "audit_linear_execution_features", "config": asdict(config), "tape": {"schema": manifest.schema, "exchange": manifest.exchange, "symbol": manifest.symbol, "num_events": manifest.num_events, "num_l2_batches": manifest.num_l2_batches, "num_trades": manifest.num_trades, "start_local_ts_us": manifest.start_local_ts_us, "end_local_ts_us": manifest.end_local_ts_us}, "linear_train_result": {"schema": result.schema, "dataset_id": result.dataset_id, "manifest_hash": result.manifest_hash, "splits": {k:v.as_dict() for k,v in result.splits.items()}, "selection_summary": result.selection_summary}, "linear_signals": signals_summary, "feature_dataset": _feature_dataset_summary(n,nf,names,decision_idx,decision_ts,config.decision_interval_us,replay_start), "per_head": per_head, "combined": {"feature_schema_match": not all_missing, "missing_features": sorted(all_missing), "extra_features": sorted(all_extra), "shared_feature_count": len(set(names)-all_missing), "warnings": sorted(set(warnings))}, "warnings": sorted(set(warnings)), "resource_mode": {"chunked_features": True, "quantile_mode": config.quantile_mode, "chunk_rows": config.chunk_rows, "work_dir": str(work_root)}}
    if config.output_json is not None:
        path=Path(config.output_json); path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8"); tmp.replace(path)
    else: json.dumps(payload, allow_nan=False)
    if config.cleanup_work_dir: shutil.rmtree(work_root, ignore_errors=True)
    return payload
