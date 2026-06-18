"""Build and train lightweight adverse-selection baselines from an execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

from mmrt.contracts import SplitRole
from mmrt.execution.adverse_selection import (
    AdverseSelectionConfig,
    CounterfactualQuoteConfig,
    DEFAULT_QUOTE_CANDIDATES,
    QuoteCandidateConfig,
    KyleLambdaConfig,
    quote_candidate_configs_from_names,
    VPINConfig,
    build_adverse_selection_dataset_to_disk,
    profile_adverse_selection_label_generation,
    summarize_disk_adverse_selection_dataset,
    adverse_selection_feature_names,
    adverse_selection_label_names,
)
from mmrt.execution.contracts import LatencyConfig, QueueModelMode
from mmrt.execution.decision_grid import load_decision_grid, validate_decision_grid_for_execution_tape
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_MODEL_SCHEMA, AdverseSelectionModelArtifact, save_adverse_selection_model
from mmrt.execution.adverse_selection_dataset import (
    ADVERSE_SPLIT_CONTRACT_SCHEMA,
    adverse_selection_dataset_manifest_with_split_contract,
    estimate_adverse_dataset_bytes,
    write_adverse_selection_dataset_manifest,
)
from mmrt.execution.adverse_selection_fit import fit_adverse_baselines_streaming
from mmrt.storage import manifest as storage_manifest

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
    decision_grid_path: str
    split_source_dataset_root: str | None = None
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
    progress_interval: int | None = 100_000
    label_engine: str = "auto"
    profile_rows: int | None = None
    profile_output_json: str | None = None

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

    ridge_l2: float = 1e-3
    min_train_samples: int = 10
    target_names: tuple[str, ...] | str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_nonempty_str(self.tape_root, "tape_root"))
        object.__setattr__(self, "decision_grid_path", _require_nonempty_str(self.decision_grid_path, "decision_grid_path"))
        if self.split_source_dataset_root is not None:
            object.__setattr__(self, "split_source_dataset_root", _require_nonempty_str(self.split_source_dataset_root, "split_source_dataset_root"))
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
        if self.label_engine not in ("auto", "numba", "scalar"):
            raise ValueError("label_engine must be auto, numba, or scalar")
        if self.profile_output_json is not None:
            object.__setattr__(self, "profile_output_json", _require_nonempty_str(self.profile_output_json, "profile_output_json"))
        object.__setattr__(self, "profile_rows", _optional_positive_int(self.profile_rows, "profile_rows"))
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
        object.__setattr__(self, "ridge_l2", _require_nonnegative_float(self.ridge_l2, "ridge_l2"))
        object.__setattr__(self, "min_train_samples", _require_positive_int(self.min_train_samples, "min_train_samples"))
        if self.target_names != "auto":
            object.__setattr__(self, "target_names", _parse_target_names(self.target_names))
        if self.vpin_min_completed_buckets > self.vpin_num_buckets:
            raise ValueError("vpin_min_completed_buckets must be <= vpin_num_buckets")


def _build_adverse_selection_config(config: AdverseSelectionTrainCLIConfig) -> AdverseSelectionConfig:
    return AdverseSelectionConfig(
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
        "decision_grid_path": config.decision_grid_path,
        "split_source_dataset_root": config.split_source_dataset_root,
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
        "label_engine": config.label_engine,
        "profile_rows": config.profile_rows,
        "profile_output_json": config.profile_output_json,
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


def _log_stage(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _manifest_decision_grid_lineage(manifest: storage_manifest.StorageManifest) -> dict[str, object]:
    notes = manifest.notes or {}
    lineage = notes.get("decision_grid")
    if not isinstance(lineage, Mapping):
        raise ValueError("split source manifest notes must include decision_grid lineage")
    required = ("decision_grid_schema", "decision_grid_hash", "decision_grid_n_rows", "decision_schedule")
    missing = [key for key in required if key not in lineage]
    if missing:
        raise ValueError(f"split source manifest decision_grid lineage missing fields: {missing}")
    schedule = dict(lineage["decision_schedule"])  # type: ignore[arg-type]
    if schedule != manifest.decision_schedule:
        raise ValueError("split source manifest decision_grid schedule must match manifest decision_schedule")
    return {
        "decision_grid_schema": _require_nonempty_str(str(lineage["decision_grid_schema"]), "decision_grid_schema"),
        "decision_grid_hash": _require_nonempty_str(str(lineage["decision_grid_hash"]), "decision_grid_hash"),
        "decision_grid_n_rows": _require_positive_int(int(lineage["decision_grid_n_rows"]), "decision_grid_n_rows"),
        "decision_schedule": schedule,
    }


def _validate_split_source_lineage(manifest: storage_manifest.StorageManifest, decision_grid) -> dict[str, object]:
    lineage = _manifest_decision_grid_lineage(manifest)
    expected = {
        "decision_grid_schema": decision_grid.metadata.schema,
        "decision_grid_hash": decision_grid.decision_grid_hash,
        "decision_grid_n_rows": decision_grid.n_rows,
        "decision_schedule": decision_grid.decision_schedule,
    }
    for key, value in expected.items():
        if lineage[key] != value:
            raise ValueError(f"split source decision grid mismatch for {key}: expected={value!r} actual={lineage[key]!r}")
    return lineage


def _split_range_payload(split: storage_manifest.SplitMetadata) -> dict[str, object]:
    return {
        "segment_key": split.segment_key,
        "start_row": int(split.start_row),
        "end_row": int(split.end_row),
        "row_count": int(split.end_row - split.start_row),
        "start_local_ts_us": int(split.local_time_range.start_us),
        "end_local_ts_us": int(split.local_time_range.end_us),
        "embargo_before_us": int(split.embargo_before_us),
        "embargo_after_us": int(split.embargo_after_us),
    }


def _split_contract_from_source_dataset(split_source_dataset_root: str, decision_grid) -> dict[str, object]:
    root = Path(_require_nonempty_str(split_source_dataset_root, "split_source_dataset_root"))
    manifest_path = root / storage_manifest.DEFAULT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"split source manifest not found: {manifest_path}")
    manifest = storage_manifest.read_manifest_json(manifest_path)
    manifest.validate_against_current_code()
    lineage = _validate_split_source_lineage(manifest, decision_grid)
    entries_by_role = {
        role.value: tuple(split for split in manifest.splits if split.role == role)
        for role in (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)
    }
    missing = [role for role, entries in entries_by_role.items() if not entries]
    if missing:
        raise ValueError(f"split source manifest must include train/val/test splits; missing={missing}")
    ranges = {
        role: [_split_range_payload(split) for split in entries]
        for role, entries in entries_by_role.items()
    }
    source_row_counts = {
        role: int(sum(split.end_row - split.start_row for split in entries))
        for role, entries in entries_by_role.items()
    }
    return {
        "schema": ADVERSE_SPLIT_CONTRACT_SCHEMA,
        "version": 1,
        "split_source_dataset_root": str(root),
        "split_source_dataset_id": manifest.dataset_id,
        "split_source_manifest_hash": manifest.content_hash(),
        "ranges": ranges,
        "source_row_counts": source_row_counts,
        "adverse_dataset_rows_total": 0,
        "adverse_row_counts": {"train": 0, "val": 0, "test": 0, "out_of_split": 0},
        **lineage,
    }


def run_adverse_selection_training(config: AdverseSelectionTrainCLIConfig) -> dict[str, object]:
    if not isinstance(config, AdverseSelectionTrainCLIConfig):
        raise ValueError("config must be AdverseSelectionTrainCLIConfig")
    output_json = Path(config.output_json) if config.output_json is not None else _default_output_json(config.tape_root)
    model_npz = Path(config.model_npz) if config.model_npz is not None else _default_model_npz(config.tape_root)
    profile_mode = config.profile_rows is not None
    if not profile_mode and config.split_source_dataset_root is None:
        raise ValueError("split_source_dataset_root is required for adverse-selection training")
    if not profile_mode and output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output_json already exists: {output_json}")
    if not profile_mode and model_npz.exists() and not config.overwrite:
        raise FileExistsError(f"model_npz already exists: {model_npz}")

    _log_stage("train_adverse_selection stage=tape_load")
    tape = load_execution_tape(
        config.tape_root,
        mmap_mode=config.mmap_mode,
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    _log_stage("train_adverse_selection stage=decision_grid_load")
    decision_grid = load_decision_grid(config.decision_grid_path)
    _log_stage("train_adverse_selection stage=decision_grid_validate")
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    adverse_config = _build_adverse_selection_config(config)
    dataset_root = Path(config.dataset_root) if config.dataset_root is not None else Path(config.tape_root) / "adverse_selection_dataset"
    work_root = (Path(config.work_dir) if config.work_dir is not None else dataset_root.parent) / "adverse_selection_work"
    if profile_mode:
        _log_stage("train_adverse_selection stage=profile_label_generation")
        profile = profile_adverse_selection_label_generation(
            tape,
            config=adverse_config,
            decision_grid=decision_grid,
            profile_rows=int(config.profile_rows),
            work_dir=config.work_dir if config.work_dir is not None else str(dataset_root.parent),
            chunk_rows=config.chunk_rows,
            overwrite=config.overwrite,
            cleanup_chunks=True,
            progress_interval=config.progress_interval,
            label_engine=config.label_engine,
        )
        manifest = tape.manifest
        summary = {
            **profile,
            "tape_root": str(Path(config.tape_root)),
            "decision_grid_path": str(Path(config.decision_grid_path)),
            "work_dir": str(work_root),
            "output_json": None,
            "model_npz": None,
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
            "decision_grid": {
                "schema": decision_grid.metadata.schema,
                "hash": decision_grid.decision_grid_hash,
                "n_rows": decision_grid.n_rows,
                "schedule": decision_grid.decision_schedule,
            },
        }
        if config.profile_output_json is not None:
            _write_json_atomic(Path(config.profile_output_json), summary)
        return summary
    assert config.split_source_dataset_root is not None
    _log_stage("train_adverse_selection stage=split_source_load")
    split_contract = _split_contract_from_source_dataset(config.split_source_dataset_root, decision_grid)
    feature_count = len(adverse_selection_feature_names(adverse_config))
    label_count = len(adverse_selection_label_names(adverse_config))
    disk_estimate = estimate_adverse_dataset_bytes(num_decisions_estimate=decision_grid.n_rows, num_features=feature_count, num_labels=label_count)
    _log_stage("train_adverse_selection stage=dataset_build")
    dataset = build_adverse_selection_dataset_to_disk(
        tape,
        config=adverse_config,
        decision_grid=decision_grid,
        split_contract=split_contract,
        output_root=dataset_root,
        work_dir=config.work_dir,
        chunk_rows=config.chunk_rows,
        overwrite=config.overwrite,
        cleanup_chunks=True,
        cleanup_work_dir=config.cleanup_work_dir,
        progress_interval=config.progress_interval,
        label_engine=config.label_engine,
    )
    _log_stage("train_adverse_selection stage=dataset_summarize")
    dataset_summary = summarize_disk_adverse_selection_dataset(dataset, chunk_rows=config.chunk_rows)
    _log_stage("train_adverse_selection stage=model_fit")
    baseline_fit = fit_adverse_baselines_streaming(
        dataset,
        target_names=_resolve_target_names(dataset, config.target_names),
        split_contract=split_contract,
        ridge_l2=config.ridge_l2,
        min_train_samples=config.min_train_samples,
        chunk_rows=config.chunk_rows,
        metrics_mode=config.metrics_mode,
        auc_bins=config.auc_bins,
        exact_auc_max_rows=config.exact_auc_max_rows,
    )
    baseline_summary = baseline_fit.metrics
    split_contract_with_counts = dict(baseline_summary["split_contract"])  # type: ignore[arg-type]
    dataset_manifest = adverse_selection_dataset_manifest_with_split_contract(dataset.manifest, split_contract_with_counts)
    write_adverse_selection_dataset_manifest(dataset.root, dataset_manifest)
    dataset_summary = {
        **dataset_summary,
        "split_source_dataset_root": split_contract_with_counts["split_source_dataset_root"],
        "split_source_dataset_id": split_contract_with_counts["split_source_dataset_id"],
        "split_source_manifest_hash": split_contract_with_counts["split_source_manifest_hash"],
        "split_contract": split_contract_with_counts,
        "adverse_dataset_rows_total": baseline_summary["adverse_dataset_rows_total"],
        "adverse_train_rows": baseline_summary["adverse_train_rows"],
        "adverse_val_rows": baseline_summary["adverse_val_rows"],
        "adverse_test_rows": baseline_summary["adverse_test_rows"],
        "out_of_split_rows": baseline_summary["out_of_split_rows"],
        "dropped_rows": int(decision_grid.n_rows - dataset.num_rows),
    }
    model_written = False
    if baseline_fit.target_names:
        _log_stage("train_adverse_selection stage=model_write")
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
                decision_grid_schema=decision_grid.metadata.schema,
                decision_grid_hash=decision_grid.decision_grid_hash,
                decision_grid_n_rows=decision_grid.n_rows,
                decision_schedule=decision_grid.decision_schedule,
                split_source_dataset_root=str(split_contract_with_counts["split_source_dataset_root"]),
                split_source_dataset_id=str(split_contract_with_counts["split_source_dataset_id"]),
                split_source_manifest_hash=str(split_contract_with_counts["split_source_manifest_hash"]),
                split_contract=split_contract_with_counts,
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
        "decision_grid_path": str(Path(config.decision_grid_path)),
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
        "decision_grid": {
            "schema": decision_grid.metadata.schema,
            "hash": decision_grid.decision_grid_hash,
            "n_rows": decision_grid.n_rows,
            "schedule": decision_grid.decision_schedule,
        },
        "split_source": {
            "dataset_root": split_contract_with_counts["split_source_dataset_root"],
            "dataset_id": split_contract_with_counts["split_source_dataset_id"],
            "manifest_hash": split_contract_with_counts["split_source_manifest_hash"],
            "split_contract_schema": split_contract_with_counts["schema"],
            "split_contract_version": split_contract_with_counts["version"],
            "ranges": split_contract_with_counts["ranges"],
            "source_row_counts": split_contract_with_counts["source_row_counts"],
            "adverse_row_counts": split_contract_with_counts["adverse_row_counts"],
            "decision_grid_schema": split_contract_with_counts["decision_grid_schema"],
            "decision_grid_hash": split_contract_with_counts["decision_grid_hash"],
            "decision_grid_n_rows": split_contract_with_counts["decision_grid_n_rows"],
            "decision_schedule": split_contract_with_counts["decision_schedule"],
        },
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
            "label_engine": config.label_engine,
            "keep_dataset": config.keep_dataset,
            "keep_work_dir": not config.cleanup_work_dir,
            "cleanup_work_dir": config.cleanup_work_dir,
        },
        "disk_estimate": disk_estimate,
    }
    _log_stage("train_adverse_selection stage=summary_write")
    _write_json_atomic(output_json, summary)
    if not config.keep_dataset:
        import shutil
        shutil.rmtree(dataset_root, ignore_errors=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--decision-grid", dest="decision_grid_path", required=True)
    parser.add_argument("--split-source-dataset-root")
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
    parser.add_argument("--progress-interval", type=int, default=100_000)
    parser.add_argument("--label-engine", choices=("auto", "numba", "scalar"), default="auto")
    parser.add_argument("--profile-rows", type=int)
    parser.add_argument("--profile-output-json")
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
    parser.add_argument("--ridge-l2", type=float, default=0.001)
    parser.add_argument("--min-train-samples", type=int, default=10)
    parser.add_argument("--target-names", default="auto")
    return parser


def _config_from_args(args: argparse.Namespace) -> AdverseSelectionTrainCLIConfig:
    return AdverseSelectionTrainCLIConfig(
        tape_root=args.tape_root,
        decision_grid_path=args.decision_grid_path,
        split_source_dataset_root=args.split_source_dataset_root,
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
        label_engine=args.label_engine,
        profile_rows=args.profile_rows,
        profile_output_json=args.profile_output_json,
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
