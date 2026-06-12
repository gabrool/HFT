"""Build and train lightweight adverse-selection baselines from an execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from mmrt.execution.adverse_selection import (
    AdverseSelectionConfig,
    CounterfactualQuoteConfig,
    DEFAULT_QUOTE_CANDIDATES,
    QuoteCandidateConfig,
    KyleLambdaConfig,
    quote_candidate_configs_from_names,
    VPINConfig,
    build_adverse_selection_dataset_to_disk,
    summarize_disk_adverse_selection_dataset,
    adverse_selection_feature_names,
    adverse_selection_label_names,
)
from mmrt.execution.contracts import LatencyConfig, QueueModelMode
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_MODEL_SCHEMA, AdverseSelectionModelArtifact, save_adverse_selection_model
from mmrt.execution.adverse_selection_dataset import estimate_adverse_dataset_bytes
from mmrt.execution.adverse_selection_fit import fit_adverse_baselines_streaming

__all__ = [
    "AdverseSelectionTrainCLIConfig",
    "run_adverse_selection_training",
    "build_arg_parser",
    "main",
]



def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, name)


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def _require_positive_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_nonnegative_float(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_probability(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_probability_exclusive(value: float, name: str) -> float:
    value = _require_finite_float(value, name)
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return value


def _coerce_queue_mode(value: QueueModelMode | str) -> QueueModelMode:
    if isinstance(value, QueueModelMode):
        return value
    if isinstance(value, str):
        try:
            return QueueModelMode(value)
        except ValueError as exc:
            raise ValueError(f"queue_mode has invalid value {value!r}") from exc
    raise ValueError("queue_mode must be QueueModelMode or str")


def _parse_int_tuple(value: str | Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(part == "" for part in parts):
            raise ValueError(f"{name} must be a comma-separated list of positive ints")
        try:
            values = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of positive ints") from exc
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence of positive ints") from exc
    if not values:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_require_positive_int(item, f"{name}[{i}]") for i, item in enumerate(values))


def _parse_target_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        try:
            parts = [str(part).strip() if isinstance(part, str) else part for part in value]
        except TypeError as exc:
            raise ValueError("target_names must be a sequence of non-empty strings") from exc
    if not parts:
        raise ValueError("target_names must be non-empty")
    return tuple(_require_nonempty_str(part, f"target_names[{i}]") for i, part in enumerate(parts))


def _parse_quote_candidates(value: str | Sequence[QuoteCandidateConfig]) -> tuple[QuoteCandidateConfig, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(part == "" for part in parts):
            raise ValueError("quote_candidates must be a comma-separated non-empty list")
        return quote_candidate_configs_from_names(parts)
    if isinstance(value, (bytes, bytearray)):
        raise ValueError("quote_candidates must be a sequence of QuoteCandidateConfig")
    try:
        candidates = tuple(value)
    except TypeError as exc:
        raise ValueError("quote_candidates must be a sequence of QuoteCandidateConfig") from exc
    if not candidates:
        raise ValueError("quote_candidates must be non-empty")
    seen: set[str] = set()
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, QuoteCandidateConfig):
            raise ValueError(f"quote_candidates[{i}] must be QuoteCandidateConfig")
        if candidate.name in seen:
            raise ValueError(f"duplicate quote candidate name {candidate.name!r}")
        seen.add(candidate.name)
    return candidates

def _resolve_target_names(dataset, requested: tuple[str, ...] | str) -> tuple[str, ...]:
    if requested == "auto":
        return tuple(name for name in dataset.label_names if name.endswith("_filled") or name.endswith("_toxic_cost_bps"))
    return _parse_target_names(requested)


@dataclass(frozen=True, slots=True)
class AdverseSelectionTrainCLIConfig:
    tape_root: str
    output_json: str | None = None
    model_npz: str | None = None
    overwrite: bool = False
    mmap_mode: str | None = "r"
    dataset_root: str | None = None
    work_dir: str | None = None
    chunk_rows: int = 100_000
    keep_dataset: bool = True
    cleanup_work_dir: bool = False
    metrics_mode: str = "approx"
    auc_bins: int = 2000
    exact_auc_max_rows: int = 1_000_000
    progress_interval: int | None = None

    decision_interval_us: int = 500_000
    start_event_index: int | None = None
    max_decisions: int | None = None
    flow_windows_us: tuple[int, ...] | str = (200_000, 500_000, 1_000_000)
    drop_incomplete_horizon: bool = True

    vpin_bucket_volume: float = 50.0
    vpin_num_buckets: int = 50
    vpin_min_completed_buckets: int = 10
    vpin_use_notional_volume: bool = False

    kyle_sample_interval_us: int = 500_000
    kyle_response_horizon_us: int = 1_000_000
    kyle_windows_us: tuple[int, ...] | str = (10_000_000, 30_000_000)
    kyle_min_samples: int = 5
    kyle_use_notional_flow: bool = False

    quote_candidates: tuple[QuoteCandidateConfig, ...] | str = DEFAULT_QUOTE_CANDIDATES
    post_only_gap_ticks: int = 1
    decision_compute_latency_us: int = 50
    order_entry_latency_us: int = 500
    invalid_quote_policy: str = "mask"
    order_qty: float = 0.001
    fill_horizon_us: int = 1_000_000
    adverse_horizon_us: int = 1_000_000
    toxic_threshold_bps: float = 0.0

    queue_mode: QueueModelMode | str = QueueModelMode.CONSERVATIVE
    l2_decrease_weight: float = 0.25
    trade_at_level_weight: float = 0.5
    unknown_level_queue_ahead_qty: float = 1_000_000_000.0
    dedupe_l2_decrease_with_trade_prints: bool = True

    train_fraction: float = 0.7
    ridge_l2: float = 1e-3
    min_train_samples: int = 10
    target_names: tuple[str, ...] | str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_nonempty_str(self.tape_root, "tape_root"))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", _require_nonempty_str(self.output_json, "output_json"))
        if self.model_npz is not None:
            object.__setattr__(self, "model_npz", _require_nonempty_str(self.model_npz, "model_npz"))
        object.__setattr__(self, "overwrite", _require_bool(self.overwrite, "overwrite"))
        if self.mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'")
        if self.dataset_root is not None:
            object.__setattr__(self, "dataset_root", _require_nonempty_str(self.dataset_root, "dataset_root"))
        if self.work_dir is not None:
            object.__setattr__(self, "work_dir", _require_nonempty_str(self.work_dir, "work_dir"))
        object.__setattr__(self, "chunk_rows", _require_positive_int(self.chunk_rows, "chunk_rows"))
        object.__setattr__(self, "keep_dataset", _require_bool(self.keep_dataset, "keep_dataset"))
        object.__setattr__(self, "cleanup_work_dir", _require_bool(self.cleanup_work_dir, "cleanup_work_dir"))
        if self.metrics_mode not in ("approx", "none", "exact"):
            raise ValueError("metrics_mode must be approx, none, or exact")
        object.__setattr__(self, "auc_bins", _require_positive_int(self.auc_bins, "auc_bins"))
        object.__setattr__(self, "exact_auc_max_rows", _require_positive_int(self.exact_auc_max_rows, "exact_auc_max_rows"))
        object.__setattr__(self, "progress_interval", _optional_positive_int(self.progress_interval, "progress_interval"))
        object.__setattr__(self, "decision_interval_us", _require_positive_int(self.decision_interval_us, "decision_interval_us"))
        object.__setattr__(self, "start_event_index", _optional_nonnegative_int(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "max_decisions", _optional_positive_int(self.max_decisions, "max_decisions"))
        object.__setattr__(self, "flow_windows_us", _parse_int_tuple(self.flow_windows_us, "flow_windows_us"))
        object.__setattr__(self, "drop_incomplete_horizon", _require_bool(self.drop_incomplete_horizon, "drop_incomplete_horizon"))
        object.__setattr__(self, "vpin_bucket_volume", _require_positive_float(self.vpin_bucket_volume, "vpin_bucket_volume"))
        object.__setattr__(self, "vpin_num_buckets", _require_positive_int(self.vpin_num_buckets, "vpin_num_buckets"))
        object.__setattr__(self, "vpin_min_completed_buckets", _require_positive_int(self.vpin_min_completed_buckets, "vpin_min_completed_buckets"))
        object.__setattr__(self, "vpin_use_notional_volume", _require_bool(self.vpin_use_notional_volume, "vpin_use_notional_volume"))
        object.__setattr__(self, "kyle_sample_interval_us", _require_positive_int(self.kyle_sample_interval_us, "kyle_sample_interval_us"))
        object.__setattr__(self, "kyle_response_horizon_us", _require_positive_int(self.kyle_response_horizon_us, "kyle_response_horizon_us"))
        object.__setattr__(self, "kyle_windows_us", _parse_int_tuple(self.kyle_windows_us, "kyle_windows_us"))
        object.__setattr__(self, "kyle_min_samples", _require_positive_int(self.kyle_min_samples, "kyle_min_samples"))
        object.__setattr__(self, "kyle_use_notional_flow", _require_bool(self.kyle_use_notional_flow, "kyle_use_notional_flow"))
        object.__setattr__(self, "quote_candidates", _parse_quote_candidates(self.quote_candidates))
        object.__setattr__(self, "post_only_gap_ticks", _require_nonnegative_int(self.post_only_gap_ticks, "post_only_gap_ticks"))
        object.__setattr__(
            self,
            "decision_compute_latency_us",
            _require_nonnegative_int(self.decision_compute_latency_us, "decision_compute_latency_us"),
        )
        object.__setattr__(
            self,
            "order_entry_latency_us",
            _require_nonnegative_int(self.order_entry_latency_us, "order_entry_latency_us"),
        )
        if self.invalid_quote_policy not in ("mask", "error"):
            raise ValueError("invalid_quote_policy must be mask or error")
        object.__setattr__(self, "order_qty", _require_positive_float(self.order_qty, "order_qty"))
        object.__setattr__(self, "fill_horizon_us", _require_positive_int(self.fill_horizon_us, "fill_horizon_us"))
        object.__setattr__(self, "adverse_horizon_us", _require_positive_int(self.adverse_horizon_us, "adverse_horizon_us"))
        object.__setattr__(self, "toxic_threshold_bps", _require_nonnegative_float(self.toxic_threshold_bps, "toxic_threshold_bps"))
        object.__setattr__(self, "queue_mode", _coerce_queue_mode(self.queue_mode))
        object.__setattr__(self, "l2_decrease_weight", _require_probability(self.l2_decrease_weight, "l2_decrease_weight"))
        object.__setattr__(self, "trade_at_level_weight", _require_probability(self.trade_at_level_weight, "trade_at_level_weight"))
        object.__setattr__(self, "unknown_level_queue_ahead_qty", _require_nonnegative_float(self.unknown_level_queue_ahead_qty, "unknown_level_queue_ahead_qty"))
        object.__setattr__(self, "dedupe_l2_decrease_with_trade_prints", _require_bool(self.dedupe_l2_decrease_with_trade_prints, "dedupe_l2_decrease_with_trade_prints"))
        object.__setattr__(self, "train_fraction", _require_probability_exclusive(self.train_fraction, "train_fraction"))
        object.__setattr__(self, "ridge_l2", _require_nonnegative_float(self.ridge_l2, "ridge_l2"))
        object.__setattr__(self, "min_train_samples", _require_positive_int(self.min_train_samples, "min_train_samples"))
        if self.target_names != "auto":
            object.__setattr__(self, "target_names", _parse_target_names(self.target_names))
        if self.vpin_min_completed_buckets > self.vpin_num_buckets:
            raise ValueError("vpin_min_completed_buckets must be <= vpin_num_buckets")


def _build_adverse_selection_config(config: AdverseSelectionTrainCLIConfig) -> AdverseSelectionConfig:
    return AdverseSelectionConfig(
        decision_interval_us=config.decision_interval_us,
        start_event_index=config.start_event_index,
        max_decisions=config.max_decisions,
        flow_windows_us=config.flow_windows_us,
        vpin=VPINConfig(
            bucket_volume=config.vpin_bucket_volume,
            num_buckets=config.vpin_num_buckets,
            min_completed_buckets=config.vpin_min_completed_buckets,
            use_notional_volume=config.vpin_use_notional_volume,
        ),
        kyle=KyleLambdaConfig(
            sample_interval_us=config.kyle_sample_interval_us,
            response_horizon_us=config.kyle_response_horizon_us,
            windows_us=config.kyle_windows_us,
            min_samples=config.kyle_min_samples,
            use_notional_flow=config.kyle_use_notional_flow,
        ),
        quote=CounterfactualQuoteConfig(
            quote_candidates=config.quote_candidates,
            post_only_gap_ticks=config.post_only_gap_ticks,
            latency_config=LatencyConfig(
                decision_compute_latency_us=config.decision_compute_latency_us,
                order_entry_latency_us=config.order_entry_latency_us,
            ),
            invalid_quote_policy=config.invalid_quote_policy,
            order_qty=config.order_qty,
            fill_horizon_us=config.fill_horizon_us,
            adverse_horizon_us=config.adverse_horizon_us,
            toxic_threshold_bps=config.toxic_threshold_bps,
            queue_model=QueueModelConfig(
                mode=config.queue_mode,
                l2_decrease_weight=config.l2_decrease_weight,
                trade_at_level_weight=config.trade_at_level_weight,
                unknown_level_queue_ahead_qty=config.unknown_level_queue_ahead_qty,
                dedupe_l2_decrease_with_trade_prints=config.dedupe_l2_decrease_with_trade_prints,
            ),
        ),
        drop_incomplete_horizon=config.drop_incomplete_horizon,
    )


def _summary_config(config: AdverseSelectionTrainCLIConfig) -> dict[str, object]:
    return {
        "tape_root": config.tape_root,
        "output_json": config.output_json,
        "model_npz": config.model_npz,
        "overwrite": config.overwrite,
        "mmap_mode": config.mmap_mode,
        "dataset_root": config.dataset_root,
        "work_dir": config.work_dir,
        "chunk_rows": config.chunk_rows,
        "keep_dataset": config.keep_dataset,
        "cleanup_work_dir": config.cleanup_work_dir,
        "metrics_mode": config.metrics_mode,
        "auc_bins": config.auc_bins,
        "exact_auc_max_rows": config.exact_auc_max_rows,
        "progress_interval": config.progress_interval,
        "decision_interval_us": config.decision_interval_us,
        "start_event_index": config.start_event_index,
        "max_decisions": config.max_decisions,
        "flow_windows_us": list(config.flow_windows_us),
        "drop_incomplete_horizon": config.drop_incomplete_horizon,
        "vpin_bucket_volume": config.vpin_bucket_volume,
        "vpin_num_buckets": config.vpin_num_buckets,
        "vpin_min_completed_buckets": config.vpin_min_completed_buckets,
        "vpin_use_notional_volume": config.vpin_use_notional_volume,
        "kyle_sample_interval_us": config.kyle_sample_interval_us,
        "kyle_response_horizon_us": config.kyle_response_horizon_us,
        "kyle_windows_us": list(config.kyle_windows_us),
        "kyle_min_samples": config.kyle_min_samples,
        "kyle_use_notional_flow": config.kyle_use_notional_flow,
        "quote_candidates": [c.name for c in config.quote_candidates],
        "post_only_gap_ticks": config.post_only_gap_ticks,
        "decision_compute_latency_us": config.decision_compute_latency_us,
        "order_entry_latency_us": config.order_entry_latency_us,
        "invalid_quote_policy": config.invalid_quote_policy,
        "order_qty": config.order_qty,
        "fill_horizon_us": config.fill_horizon_us,
        "adverse_horizon_us": config.adverse_horizon_us,
        "toxic_threshold_bps": config.toxic_threshold_bps,
        "queue_mode": config.queue_mode.value,
        "l2_decrease_weight": config.l2_decrease_weight,
        "trade_at_level_weight": config.trade_at_level_weight,
        "unknown_level_queue_ahead_qty": config.unknown_level_queue_ahead_qty,
        "dedupe_l2_decrease_with_trade_prints": config.dedupe_l2_decrease_with_trade_prints,
        "train_fraction": config.train_fraction,
        "ridge_l2": config.ridge_l2,
        "min_train_samples": config.min_train_samples,
        "target_names": config.target_names if config.target_names == "auto" else list(config.target_names),
    }


def _default_output_json(tape_root: str) -> Path:
    return Path(tape_root) / "adverse_selection_summary.json"


def _default_model_npz(tape_root: str) -> Path:
    return Path(tape_root) / "adverse_selection_model.npz"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)



@dataclass(frozen=True, slots=True)
class _BaselineFitResult:
    target_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    train_rows: int
    val_rows: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    metrics: dict[str, object]


def _fit_feature_standardization(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize_features(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((X - mean) / scale).astype(np.float64, copy=False)


def _chronological_split(n: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * train_fraction)
    n_train = min(max(n_train, 1), n - 1)
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n, dtype=np.int64)
    return train_idx, val_idx


def _fit_ridge(X_train: np.ndarray, y_train: np.ndarray, ridge_l2: float) -> tuple[np.ndarray, float]:
    n = X_train.shape[0]
    X_aug = np.concatenate([np.ones((n, 1), dtype=np.float64), X_train], axis=1)
    reg = np.eye(X_aug.shape[1], dtype=np.float64) * ridge_l2
    reg[0, 0] = 0.0
    lhs = X_aug.T @ X_aug + reg
    rhs = X_aug.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return beta[1:].astype(np.float64, copy=False), float(beta[0])


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_true) > 0.5
    n_pos = int(np.count_nonzero(y))
    n = len(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_score[j] == sorted_score[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    sum_ranks_pos = float(np.sum(ranks[y]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _target_metrics(y_train: np.ndarray, pred_train: np.ndarray, y_val: np.ndarray, pred_val: np.ndarray, target_name: str) -> dict[str, object]:
    metrics: dict[str, object] = {
        "target_name": target_name,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_rmse": _rmse(y_train, pred_train),
        "val_rmse": _rmse(y_val, pred_val),
        "train_mae": _mae(y_train, pred_train),
        "val_mae": _mae(y_val, pred_val),
        "train_r2": _r2_score(y_train, pred_train),
        "val_r2": _r2_score(y_val, pred_val),
        "label_mean_train": float(np.mean(y_train)),
        "label_mean_val": float(np.mean(y_val)),
        "prediction_mean_val": float(np.mean(pred_val)),
        "skipped": False,
    }
    if target_name.endswith("_filled") or target_name.endswith("_toxic_fill"):
        metrics["train_auc"] = _binary_auc(y_train, pred_train)
        metrics["val_auc"] = _binary_auc(y_val, pred_val)
    return metrics


def _fit_baselines(
    dataset,
    *,
    target_names: tuple[str, ...],
    train_fraction: float,
    ridge_l2: float,
    min_train_samples: int,
) -> _BaselineFitResult:
    n = int(dataset.num_decisions)
    if n < 2:
        targets_metrics = {
            target_name: {
                "target_name": target_name,
                "train_rows": 0,
                "val_rows": 0,
                "skipped": True,
                "skip_reason": "not_enough_decisions",
            }
            for target_name in target_names
        }
        return _BaselineFitResult(
            target_names=(),
            feature_names=tuple(dataset.feature_names),
            train_rows=0,
            val_rows=0,
            feature_mean=np.asarray([], dtype=np.float32),
            feature_scale=np.asarray([], dtype=np.float32),
            coefficients=np.empty((0, dataset.num_features), dtype=np.float64),
            intercepts=np.asarray([], dtype=np.float64),
            metrics={
                "enabled": True,
                "train_fraction": train_fraction,
                "ridge_l2": ridge_l2,
                "min_train_samples": min_train_samples,
                "train_rows_total": 0,
                "val_rows_total": 0,
                "fitted_target_count": 0,
                "requested_target_count": len(target_names),
                "targets": targets_metrics,
                "skipped": True,
                "skip_reason": "not_enough_decisions",
            },
        )
    train_idx, val_idx = _chronological_split(n, train_fraction)
    X = dataset.features.astype(np.float64, copy=False)
    feature_mean, feature_scale = _fit_feature_standardization(X[train_idx])
    X_std = _standardize_features(X, feature_mean, feature_scale)
    label_index = {name: i for i, name in enumerate(dataset.label_names)}
    fitted_names: list[str] = []
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    targets_metrics: dict[str, object] = {}
    for target_name in target_names:
        if target_name not in label_index:
            targets_metrics[target_name] = {
                "target_name": target_name,
                "train_rows": 0,
                "val_rows": 0,
                "skipped": True,
                "skip_reason": "unknown_target",
            }
            continue
        target_idx = label_index[target_name]
        mask = dataset.label_masks[:, target_idx]
        target_train_idx = train_idx[mask[train_idx]]
        target_val_idx = val_idx[mask[val_idx]]
        train_rows = int(len(target_train_idx))
        val_rows = int(len(target_val_idx))
        if train_rows < min_train_samples or val_rows == 0:
            reason = "not_enough_train_rows" if train_rows < min_train_samples else "no_validation_rows"
            targets_metrics[target_name] = {
                "target_name": target_name,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "skipped": True,
                "skip_reason": reason,
            }
            continue
        y_train = dataset.labels[target_train_idx, target_idx].astype(np.float64, copy=False)
        y_val = dataset.labels[target_val_idx, target_idx].astype(np.float64, copy=False)
        coef, intercept = _fit_ridge(X_std[target_train_idx], y_train, ridge_l2)
        pred_train = X_std[target_train_idx] @ coef + intercept
        pred_val = X_std[target_val_idx] @ coef + intercept
        fitted_names.append(target_name)
        coefficients.append(coef)
        intercepts.append(intercept)
        targets_metrics[target_name] = _target_metrics(y_train, pred_train, y_val, pred_val, target_name)
    if not fitted_names:
        metrics: dict[str, object] = {
            "enabled": True,
            "train_fraction": train_fraction,
            "ridge_l2": ridge_l2,
            "min_train_samples": min_train_samples,
            "train_rows_total": int(len(train_idx)),
            "val_rows_total": int(len(val_idx)),
            "fitted_target_count": 0,
            "requested_target_count": len(target_names),
            "targets": targets_metrics,
            "skipped": True,
            "skip_reason": "all_targets_skipped",
        }
        return _BaselineFitResult(
            target_names=(),
            feature_names=tuple(dataset.feature_names),
            train_rows=int(len(train_idx)),
            val_rows=int(len(val_idx)),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            coefficients=np.empty((0, dataset.num_features), dtype=np.float64),
            intercepts=np.asarray([], dtype=np.float64),
            metrics=metrics,
        )
    metrics: dict[str, object] = {
        "enabled": True,
        "train_fraction": train_fraction,
        "ridge_l2": ridge_l2,
        "min_train_samples": min_train_samples,
        "train_rows_total": int(len(train_idx)),
        "val_rows_total": int(len(val_idx)),
        "fitted_target_count": len(fitted_names),
        "requested_target_count": len(target_names),
        "targets": targets_metrics,
    }
    return _BaselineFitResult(
        target_names=tuple(fitted_names),
        feature_names=tuple(dataset.feature_names),
        train_rows=int(len(train_idx)),
        val_rows=int(len(val_idx)),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=np.vstack(coefficients),
        intercepts=np.asarray(intercepts, dtype=np.float64),
        metrics=metrics,
    )


def run_adverse_selection_training(config: AdverseSelectionTrainCLIConfig) -> dict[str, object]:
    if not isinstance(config, AdverseSelectionTrainCLIConfig):
        raise ValueError("config must be AdverseSelectionTrainCLIConfig")
    output_json = Path(config.output_json) if config.output_json is not None else _default_output_json(config.tape_root)
    model_npz = Path(config.model_npz) if config.model_npz is not None else _default_model_npz(config.tape_root)
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output_json already exists: {output_json}")
    if model_npz.exists() and not config.overwrite:
        raise FileExistsError(f"model_npz already exists: {model_npz}")

    tape = load_execution_tape(
        config.tape_root,
        mmap_mode=config.mmap_mode,
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    adverse_config = _build_adverse_selection_config(config)
    dataset_root = Path(config.dataset_root) if config.dataset_root is not None else Path(config.tape_root) / "adverse_selection_dataset"
    work_root = (Path(config.work_dir) if config.work_dir is not None else dataset_root.parent) / "adverse_selection_work"
    feature_count = len(adverse_selection_feature_names(adverse_config))
    label_count = len(adverse_selection_label_names(adverse_config))
    duration_us = max(0, int(tape.manifest.end_local_ts_us) - int(tape.manifest.start_local_ts_us))
    decisions_est = 0 if config.decision_interval_us <= 0 else duration_us // config.decision_interval_us + 1
    disk_estimate = estimate_adverse_dataset_bytes(num_decisions_estimate=decisions_est, num_features=feature_count, num_labels=label_count)
    dataset = build_adverse_selection_dataset_to_disk(
        tape,
        config=adverse_config,
        output_root=dataset_root,
        work_dir=config.work_dir,
        chunk_rows=config.chunk_rows,
        overwrite=config.overwrite,
        cleanup_chunks=True,
        cleanup_work_dir=config.cleanup_work_dir,
        progress_interval=config.progress_interval,
    )
    dataset_summary = summarize_disk_adverse_selection_dataset(dataset, chunk_rows=config.chunk_rows)
    baseline_fit = fit_adverse_baselines_streaming(
        dataset,
        target_names=_resolve_target_names(dataset, config.target_names),
        train_fraction=config.train_fraction,
        ridge_l2=config.ridge_l2,
        min_train_samples=config.min_train_samples,
        chunk_rows=config.chunk_rows,
        metrics_mode=config.metrics_mode,
        auc_bins=config.auc_bins,
        exact_auc_max_rows=config.exact_auc_max_rows,
    )
    baseline_summary = baseline_fit.metrics
    model_written = False
    if baseline_fit.target_names:
        save_adverse_selection_model(
            model_npz,
            AdverseSelectionModelArtifact(
                schema=ADVERSE_SELECTION_MODEL_SCHEMA,
                feature_names=baseline_fit.feature_names,
                target_names=baseline_fit.target_names,
                feature_mean=baseline_fit.feature_mean.astype(np.float32),
                feature_scale=baseline_fit.feature_scale.astype(np.float32),
                coefficients=baseline_fit.coefficients.astype(np.float32),
                intercepts=baseline_fit.intercepts.astype(np.float32),
                config_json=json.dumps(_summary_config(config), sort_keys=True),
                exchange=tape.manifest.exchange,
                symbol=tape.manifest.symbol,
            ),
            overwrite=config.overwrite,
        )
        model_written = True
    status = "ok" if dataset.num_decisions >= 1 and model_written else "warning"
    manifest = tape.manifest
    try:
        from mmrt.execution.adverse_selection_index import load_adverse_selection_index
        idx = load_adverse_selection_index(work_root, mmap_mode="r")
        index_summary = {"valid_l2_count": idx.manifest.valid_l2_count, "trade_flow_count": idx.manifest.trade_flow_count, "kyle_sample_count": idx.manifest.kyle_sample_count}
    except Exception:
        index_summary = {"valid_l2_count": None, "trade_flow_count": None, "kyle_sample_count": None}
    summary = {
        "status": status,
        "run_type": "train_adverse_selection",
        "tape_root": str(Path(config.tape_root)),
        "dataset_root": str(dataset_root),
        "work_dir": str(work_root),
        "output_json": str(output_json),
        "model_npz": str(model_npz) if model_written else None,
        "config": _summary_config(config),
        "tape": {
            "schema": manifest.schema,
            "exchange": manifest.exchange,
            "symbol": manifest.symbol,
            "num_events": manifest.num_events,
            "num_l2_batches": manifest.num_l2_batches,
            "num_trades": manifest.num_trades,
            "start_local_ts_us": manifest.start_local_ts_us,
            "end_local_ts_us": manifest.end_local_ts_us,
            "book_depth": manifest.notes.get("book_depth") if manifest.notes is not None else None,
        },
        "index": index_summary,
        "dataset": dataset_summary,
        "baseline": baseline_summary,
        "resource_mode": {
            "disk_backed_dataset": True,
            "disk_backed_index": True,
            "chunk_rows": config.chunk_rows,
            "metrics_mode": config.metrics_mode,
            "auc_bins": config.auc_bins,
            "exact_auc_max_rows": config.exact_auc_max_rows,
        "progress_interval": config.progress_interval,
            "keep_dataset": config.keep_dataset,
            "keep_work_dir": not config.cleanup_work_dir,
            "cleanup_work_dir": config.cleanup_work_dir,
        },
        "disk_estimate": disk_estimate,
    }
    _write_json_atomic(output_json, summary)
    if not config.keep_dataset:
        import shutil
        shutil.rmtree(dataset_root, ignore_errors=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--model-npz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--dataset-root")
    parser.add_argument("--work-dir")
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--keep-dataset", dest="keep_dataset", action="store_true", default=True)
    parser.add_argument("--delete-dataset-after-success", dest="keep_dataset", action="store_false")
    parser.add_argument("--cleanup-work-dir", dest="cleanup_work_dir", action="store_true", default=False)
    parser.add_argument("--keep-work-dir", dest="cleanup_work_dir", action="store_false")
    parser.add_argument("--metrics-mode", choices=("approx", "none", "exact"), default="approx")
    parser.add_argument("--auc-bins", type=int, default=2000)
    parser.add_argument("--exact-auc-max-rows", type=int, default=1_000_000)
    parser.add_argument("--progress-interval", type=int)
    parser.add_argument("--decision-interval-us", type=int, default=500_000)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--flow-windows-us", default="200000,500000,1000000")
    parser.add_argument("--keep-incomplete-horizon", action="store_true")
    parser.add_argument("--vpin-bucket-volume", type=float, default=50.0)
    parser.add_argument("--vpin-num-buckets", type=int, default=50)
    parser.add_argument("--vpin-min-completed-buckets", type=int, default=10)
    parser.add_argument("--vpin-use-notional-volume", action="store_true")
    parser.add_argument("--kyle-sample-interval-us", type=int, default=500_000)
    parser.add_argument("--kyle-response-horizon-us", type=int, default=1_000_000)
    parser.add_argument("--kyle-windows-us", default="10000000,30000000")
    parser.add_argument("--kyle-min-samples", type=int, default=5)
    parser.add_argument("--kyle-use-notional-flow", action="store_true")
    parser.add_argument("--quote-candidates", default="touch,inside_1,away_1", help="Comma-separated quote candidates: touch, inside_1, away_1.")
    parser.add_argument("--post-only-gap-ticks", type=int, default=1)
    parser.add_argument("--decision-compute-latency-us", type=int, default=50)
    parser.add_argument("--order-entry-latency-us", type=int, default=500)
    parser.add_argument("--invalid-quote-policy", choices=("mask", "error"), default="mask")
    parser.add_argument("--order-qty", type=float, default=0.001)
    parser.add_argument("--fill-horizon-us", type=int, default=1_000_000)
    parser.add_argument("--adverse-horizon-us", type=int, default=1_000_000)
    parser.add_argument("--toxic-threshold-bps", type=float, default=0.0)
    parser.add_argument("--queue-mode", choices=("conservative", "balanced"), default="conservative")
    parser.add_argument("--l2-decrease-weight", type=float, default=0.25)
    parser.add_argument("--trade-at-level-weight", type=float, default=0.5)
    parser.add_argument("--unknown-level-queue-ahead-qty", type=float, default=1_000_000_000.0)
    parser.add_argument(
        "--no-dedupe-l2-decrease-with-trade-prints",
        action="store_true",
        help="Disable de-duplication of L2 visible decreases already explained by same-level trade prints.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--ridge-l2", type=float, default=0.001)
    parser.add_argument("--min-train-samples", type=int, default=10)
    parser.add_argument("--target-names", default="auto")
    return parser


def _config_from_args(args: argparse.Namespace) -> AdverseSelectionTrainCLIConfig:
    return AdverseSelectionTrainCLIConfig(
        tape_root=args.tape_root,
        output_json=args.output_json,
        model_npz=args.model_npz,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
        dataset_root=args.dataset_root,
        work_dir=args.work_dir,
        chunk_rows=args.chunk_rows,
        keep_dataset=args.keep_dataset,
        cleanup_work_dir=args.cleanup_work_dir,
        metrics_mode=args.metrics_mode,
        auc_bins=args.auc_bins,
        exact_auc_max_rows=args.exact_auc_max_rows,
        progress_interval=args.progress_interval,
        decision_interval_us=args.decision_interval_us,
        start_event_index=args.start_event_index,
        max_decisions=args.max_decisions,
        flow_windows_us=args.flow_windows_us,
        drop_incomplete_horizon=not args.keep_incomplete_horizon,
        vpin_bucket_volume=args.vpin_bucket_volume,
        vpin_num_buckets=args.vpin_num_buckets,
        vpin_min_completed_buckets=args.vpin_min_completed_buckets,
        vpin_use_notional_volume=args.vpin_use_notional_volume,
        kyle_sample_interval_us=args.kyle_sample_interval_us,
        kyle_response_horizon_us=args.kyle_response_horizon_us,
        kyle_windows_us=args.kyle_windows_us,
        kyle_min_samples=args.kyle_min_samples,
        kyle_use_notional_flow=args.kyle_use_notional_flow,
        quote_candidates=args.quote_candidates,
        post_only_gap_ticks=args.post_only_gap_ticks,
        decision_compute_latency_us=args.decision_compute_latency_us,
        order_entry_latency_us=args.order_entry_latency_us,
        invalid_quote_policy=args.invalid_quote_policy,
        order_qty=args.order_qty,
        fill_horizon_us=args.fill_horizon_us,
        adverse_horizon_us=args.adverse_horizon_us,
        toxic_threshold_bps=args.toxic_threshold_bps,
        queue_mode=args.queue_mode,
        l2_decrease_weight=args.l2_decrease_weight,
        trade_at_level_weight=args.trade_at_level_weight,
        unknown_level_queue_ahead_qty=args.unknown_level_queue_ahead_qty,
        dedupe_l2_decrease_with_trade_prints=not args.no_dedupe_l2_decrease_with_trade_prints,
        train_fraction=args.train_fraction,
        ridge_l2=args.ridge_l2,
        min_train_samples=args.min_train_samples,
        target_names=args.target_names,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    summary = run_adverse_selection_training(config)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
