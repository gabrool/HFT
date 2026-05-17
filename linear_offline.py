#!/usr/bin/env python3
"""Linear offline entrypoint using CMSSL-compatible eval machinery."""

import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from CMSSL17 import (  # type: ignore
    LOOKBACK, WINDOW_MS, HORIZONS_MS,
    BATCH_SIZE,
    PRIMARY_METRIC, PRIMARY_METRIC_HORIZON_MS, PRIMARY_DIR_BAL_ACC_GUARD,
    LOW_ABS_TRIM_FRACTION, HIGH_ABS_TRIM_FRACTION,
    TARGET_TRANSFORM, TARGET_TASK, LABEL_TRIM_SCHEMA,
    MODEL_OUTPUT_SCHEMA,
    build_dataset_from_split,
    compute_primary_metric,
)
from CMSSL17_offline import (  # type: ignore
    require_supported_pipeline_splits,
    make_single_week_split_from_meta,
    validate_dataset_label_dim,
    validate_contract_meta,
    validate_loaded_label_array,
    compute_signed_raw_stats,
    build_signed_side_trim_masks_from_stats_np,
    compute_dir_class_weights_from_train_labels,
    compute_mag_init_targets_from_train_labels,
    load_stats_cache,
    cache_matches,
    save_stats_cache,
    CPUWindowBatchSource,
    make_train_band_eval_source,
    summarize_metrics,
    print_band_metrics_summary,
    save_band_metrics_jsonl,
    FAST_VAL_MAX_ROWS,
    BAND_DIAG,
    BAND_DIAG_TRAIN,
    BAND_DIAG_TRAIN_MAX_ROWS,
    BAND_DIAG_QUANTILES,
)
from CMSSL17_linear import (  # type: ignore
    LINEAR_CHECKPOINT_SCHEMA,
    LINEAR_MODEL_ARCH_SCHEMA,
    LINEAR_EXTRACTOR_SCHEMA,
    LinearConstantPriorModel,
    build_constant_priors_from_train_labels,
    linear_model_summary,
    build_linear_extractor_from_config,
    LinearPreprocessBundle,
    save_linear_preprocess_bundle,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: int = 0) -> bool:
    return int(os.environ.get(name, str(int(default)))) == 1


def _env_int_list(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default).strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
LINEAR_OUT_DIR = os.environ.get("BYBIT_LINEAR_OUT_DIR", "").strip()
LINEAR_STAGE = os.environ.get("BYBIT_LINEAR_STAGE", "stage1").strip().lower()
LINEAR_DEVICE = os.environ.get("BYBIT_LINEAR_DEVICE", "cpu").strip().lower()
LINEAR_EVAL_BATCH_SIZE = _env_int("BYBIT_LINEAR_BATCH_SIZE", BATCH_SIZE)
LINEAR_RUN_TEST = _env_bool("BYBIT_LINEAR_RUN_TEST", 1)
LINEAR_STAGE2_RUN_PRIOR_EVAL = _env_bool("BYBIT_LINEAR_STAGE2_RUN_PRIOR_EVAL", 1)

LINEAR_PREPROCESS_SCHEMA = "linear_preprocess_stage3_v1"
LINEAR_PREPROCESS_FIT_SPLIT = os.environ.get(
    "BYBIT_LINEAR_PREPROCESS_FIT_SPLIT", "train_sample"
).strip().lower()
LINEAR_PREPROCESS_WINSORIZE = _env_bool("BYBIT_LINEAR_PREPROCESS_WINSORIZE", 1)
LINEAR_PREPROCESS_WINSOR_Q_LO = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_WINSOR_Q_LO", "0.001"))
LINEAR_PREPROCESS_WINSOR_Q_HI = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_WINSOR_Q_HI", "0.999"))
LINEAR_PREPROCESS_STANDARDIZE = _env_bool("BYBIT_LINEAR_PREPROCESS_STANDARDIZE", 1)
LINEAR_PREPROCESS_STD_EPS = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_STD_EPS", "1e-6"))
LINEAR_PREPROCESS_VARIANCE_FILTER = _env_bool("BYBIT_LINEAR_PREPROCESS_VARIANCE_FILTER", 1)
LINEAR_PREPROCESS_MIN_STD = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_MIN_STD", "1e-6"))
LINEAR_PREPROCESS_POST_CLIP_ABS = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_POST_CLIP_ABS", "0.0"))
LINEAR_PREPROCESS_NONFINITE_POLICY = os.environ.get(
    "BYBIT_LINEAR_PREPROCESS_NONFINITE_POLICY", "raise"
).strip().lower()
LINEAR_PREPROCESS_FIT_MAX_ROWS = _env_int("BYBIT_LINEAR_PREPROCESS_FIT_MAX_ROWS", 50000)
LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB = _env_int("BYBIT_LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB", 2048)
LINEAR_PREPROCESS_SHARD_ROWS = _env_int("BYBIT_LINEAR_PREPROCESS_SHARD_ROWS", 50000)
LINEAR_PREPROCESS_MAX_Z_CHUNK_MB = _env_int("BYBIT_LINEAR_PREPROCESS_MAX_Z_CHUNK_MB", 2048)

LINEAR_EXTRACTOR = os.environ.get("BYBIT_LINEAR_EXTRACTOR", "raw_linear").strip().lower()
LINEAR_EXTRACTOR_FIT_MAX_ROWS = _env_int("BYBIT_LINEAR_EXTRACTOR_FIT_MAX_ROWS", 50000)
LINEAR_TRANSFORM_MAX_ROWS_PER_SPLIT = _env_int("BYBIT_LINEAR_TRANSFORM_MAX_ROWS_PER_SPLIT", 0)
LINEAR_EXTRACT_BATCH_ROWS = _env_int("BYBIT_LINEAR_EXTRACT_BATCH_ROWS", 4096)
LINEAR_CHUNKED_TRANSFORMS = _env_bool("BYBIT_LINEAR_CHUNKED_TRANSFORMS", 1)
LINEAR_TRANSFORM_SHARD_ROWS = _env_int("BYBIT_LINEAR_TRANSFORM_SHARD_ROWS", 50000)
LINEAR_MAX_X_CHUNK_MB = _env_int("BYBIT_LINEAR_MAX_X_CHUNK_MB", 2048)
LINEAR_MAX_Z_CHUNK_MB = _env_int("BYBIT_LINEAR_MAX_Z_CHUNK_MB", 2048)
LINEAR_TRANSFORM_SAVE_FORMAT = os.environ.get("BYBIT_LINEAR_TRANSFORM_SAVE_FORMAT", "npz_shards").strip().lower()
LINEAR_SAVE_TRANSFORMS = _env_bool("BYBIT_LINEAR_SAVE_TRANSFORMS", 1)
LINEAR_EXTRACTOR_N_JOBS = _env_int("BYBIT_LINEAR_EXTRACTOR_N_JOBS", 1)
LINEAR_RANDOM_SEED = _env_int("BYBIT_LINEAR_RANDOM_SEED", 17)

RAW_LINEAR_MODE = os.environ.get("BYBIT_RAW_LINEAR_MODE", "lag_bank_stats").strip().lower()
RAW_LINEAR_LAGS = _env_int_list("BYBIT_RAW_LINEAR_LAGS", "1,2,5,10,20,50")
RAW_LINEAR_WINDOWS = _env_int_list("BYBIT_RAW_LINEAR_WINDOWS", "5,10,20,50")
RAW_LINEAR_INCLUDE_STD = _env_bool("BYBIT_RAW_LINEAR_INCLUDE_STD", 1)
RAW_LINEAR_INCLUDE_SLOPE = _env_bool("BYBIT_RAW_LINEAR_INCLUDE_SLOPE", 0)

LINEAR_NUM_KERNELS = _env_int("BYBIT_LINEAR_NUM_KERNELS", 10000)
LINEAR_HYDRA_N_KERNELS = _env_int("BYBIT_LINEAR_HYDRA_N_KERNELS", 8)
LINEAR_HYDRA_N_GROUPS = _env_int("BYBIT_LINEAR_HYDRA_N_GROUPS", 64)

if LINEAR_STAGE == "stage2":
    if LINEAR_TRANSFORM_SHARD_ROWS <= 0:
        raise ValueError(f"BYBIT_LINEAR_TRANSFORM_SHARD_ROWS must be > 0, got {LINEAR_TRANSFORM_SHARD_ROWS}")
    if LINEAR_MAX_X_CHUNK_MB <= 0:
        raise ValueError(f"BYBIT_LINEAR_MAX_X_CHUNK_MB must be > 0, got {LINEAR_MAX_X_CHUNK_MB}")
    if LINEAR_MAX_Z_CHUNK_MB <= 0:
        raise ValueError(f"BYBIT_LINEAR_MAX_Z_CHUNK_MB must be > 0, got {LINEAR_MAX_Z_CHUNK_MB}")
    if LINEAR_TRANSFORM_SAVE_FORMAT != "npz_shards":
        raise ValueError(
            f"BYBIT_LINEAR_TRANSFORM_SAVE_FORMAT must be 'npz_shards', got {LINEAR_TRANSFORM_SAVE_FORMAT!r}"
        )


if LINEAR_STAGE == "stage3":
    if LINEAR_PREPROCESS_FIT_SPLIT not in {"train_sample"}:
        raise ValueError("Stage 3 currently supports BYBIT_LINEAR_PREPROCESS_FIT_SPLIT=train_sample")
    if not (0.0 <= LINEAR_PREPROCESS_WINSOR_Q_LO < LINEAR_PREPROCESS_WINSOR_Q_HI <= 1.0):
        raise ValueError(
            "BYBIT_LINEAR_PREPROCESS_WINSOR_Q_LO/HI must satisfy "
            f"0 <= lo < hi <= 1, got {LINEAR_PREPROCESS_WINSOR_Q_LO}, {LINEAR_PREPROCESS_WINSOR_Q_HI}"
        )
    if LINEAR_PREPROCESS_STD_EPS <= 0.0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_STD_EPS must be > 0, got {LINEAR_PREPROCESS_STD_EPS}")
    if LINEAR_PREPROCESS_MIN_STD < 0.0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_MIN_STD must be >= 0, got {LINEAR_PREPROCESS_MIN_STD}")
    if LINEAR_PREPROCESS_NONFINITE_POLICY not in {"raise", "warn_zero"}:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_NONFINITE_POLICY must be one of: raise, warn_zero")
    if LINEAR_PREPROCESS_FIT_MAX_ROWS <= 0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_FIT_MAX_ROWS must be > 0, got {LINEAR_PREPROCESS_FIT_MAX_ROWS}")
    if LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB <= 0:
        raise ValueError(
            f"BYBIT_LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB must be > 0, got {LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB}"
        )
    if LINEAR_PREPROCESS_SHARD_ROWS <= 0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_SHARD_ROWS must be > 0, got {LINEAR_PREPROCESS_SHARD_ROWS}")
    if LINEAR_PREPROCESS_MAX_Z_CHUNK_MB <= 0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_MAX_Z_CHUNK_MB must be > 0, got {LINEAR_PREPROCESS_MAX_Z_CHUNK_MB}")


def _resolve_device() -> torch.device:
    if LINEAR_STAGE not in {"stage1", "stage2", "stage3"}:
        raise ValueError(f"BYBIT_LINEAR_STAGE must be 'stage1', 'stage2', or 'stage3', got {LINEAR_STAGE!r}")
    if LINEAR_DEVICE not in {"cpu", "cuda", "auto"}:
        raise ValueError("BYBIT_LINEAR_DEVICE must be one of: cpu, cuda, auto")
    if LINEAR_EVAL_BATCH_SIZE <= 0:
        raise ValueError(f"BYBIT_LINEAR_BATCH_SIZE must be > 0, got {LINEAR_EVAL_BATCH_SIZE}")
    if LINEAR_DEVICE == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if LINEAR_DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BYBIT_LINEAR_DEVICE=cuda was requested, but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _validate_dataset_split(ds: Any, split_name: str, feature_dim_total: int) -> None:
    if feature_dim_total != int(ds.feature_dim_total):
        raise ValueError(
            f"Feature dimension mismatch for {split_name}: meta={feature_dim_total}, "
            f"dataset={int(ds.feature_dim_total)}"
        )
    if int(ds.lookback) != int(LOOKBACK):
        raise ValueError(f"LOOKBACK mismatch for {split_name}: config={LOOKBACK}, dataset={int(ds.lookback)}")
    if len(ds.stores) != 1:
        raise ValueError(f"{split_name} split must have exactly one store/week, got {len(ds.stores)}")
    if ds.week_ids.size and not np.all(ds.week_ids == 0):
        raise ValueError(f"{split_name} split week_ids must all be 0 for single-week protocol")
    if len(ds) > 0 and int(ds.row_idx.min()) < int(LOOKBACK - 1):
        raise ValueError(
            f"{split_name} split has rows without full history: "
            f"min_row_idx={int(ds.row_idx.min())}, lookback={LOOKBACK}"
        )


def _make_cache_meta(meta: Dict[str, Any], protocol: str, train_week_keys: list[str], train_split_entries: list[dict]) -> Dict[str, Any]:
    tr_start = int(min(entry["start"] for entry in train_split_entries))
    tr_end = int(max(entry["end"] for entry in train_split_entries))
    return {
        "feature_schema": meta.get("feature_schema"),
        "feature_transform": meta.get("feature_transform"),
        "feature_transform_policy": meta.get("feature_transform_policy"),
        "feature_transform_spec_hash": meta.get("feature_transform_spec_hash"),
        "feature_transform_warmup_rows": int(meta.get("feature_transform_warmup_rows", -1)),
        "feature_dim_core": int(meta.get("feature_dim_core", -1)),
        "feature_dim_total": int(meta.get("feature_dim_total", -1)),
        "feature_names_hash": meta.get("feature_names_hash"),
        "aux_dim": int(meta.get("aux_dim", -1)),
        "aux_transform": meta.get("aux_transform"),
        "label_trim_schema": LABEL_TRIM_SCHEMA,
        "low_abs_trim_fraction": float(LOW_ABS_TRIM_FRACTION),
        "high_abs_trim_fraction": float(HIGH_ABS_TRIM_FRACTION),
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "split_protocol": protocol,
        "train_week_keys": list(train_week_keys),
        "train_ts_start": tr_start,
        "train_ts_end": tr_end,
        "decision_time_basis": meta.get("decision_time_basis"),
        "trade_history_enabled": meta.get("trade_history_enabled"),
        "event_stream_mode": meta.get("event_stream_mode"),
        "target_transform": TARGET_TRANSFORM,
        "label_units": "signed_log_return_bps",
        "target_task": TARGET_TASK,
        "loss_weighting_schema": "dir_mag_signed_nonzero_side_trim_tempered_class_dir_plain_mag_q50_q85_ema_v1",
        "ranking_schema": "tie_aware_average_ranks_v1",
        "band_diag_quantiles": [float(x) for x in BAND_DIAG_QUANTILES],
        "linear_stage": "stage1",
    }


def _print_primary(tag: str, metrics: Dict[str, Any], primary_metric_value: float, primary_metric_label: str) -> None:
    print(
        f"[{tag}] rows={int(metrics.get('n_eval_rows', 0))} "
        f"primary_metric_name={primary_metric_label} value={primary_metric_value:.8g} "
        f"guard_dir_bal_acc={float(metrics.get('primary_dir_bal_acc', float('nan'))):.8g} "
        f"guard_passed={bool(metrics.get('primary_metric_guard_passed', False))}",
        flush=True,
    )



def _build_extractor_config() -> Dict[str, Any]:
    return {
        "extractor": LINEAR_EXTRACTOR,
        "raw_mode": RAW_LINEAR_MODE,
        "raw_lags": [int(x) for x in RAW_LINEAR_LAGS],
        "raw_windows": [int(x) for x in RAW_LINEAR_WINDOWS],
        "raw_include_std": bool(RAW_LINEAR_INCLUDE_STD),
        "raw_include_slope": bool(RAW_LINEAR_INCLUDE_SLOPE),
        "n_kernels": int(LINEAR_NUM_KERNELS),
        "hydra_n_kernels": int(LINEAR_HYDRA_N_KERNELS),
        "n_groups": int(LINEAR_HYDRA_N_GROUPS),
        "n_jobs": int(LINEAR_EXTRACTOR_N_JOBS),
        "random_state": int(LINEAR_RANDOM_SEED),
    }


def _dataset_positions(n_rows: int, max_rows: int) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("Cannot collect windows from empty dataset")
    if max_rows <= 0 or max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    return np.linspace(0, n_rows - 1, int(max_rows), dtype=np.int64)


def estimate_x_window_mb(n_rows: int, lookback: int, feature_dim: int, dtype_bytes: int = 4) -> float:
    return float(n_rows) * float(lookback) * float(feature_dim) * float(dtype_bytes) / (1024.0**2)


def estimate_matrix_mb(n_rows: int, n_cols: int, dtype_bytes: int = 4) -> float:
    return float(n_rows) * float(n_cols) * float(dtype_bytes) / (1024.0**2)


def compute_safe_window_chunk_rows(
    *,
    requested_rows: int,
    lookback: int,
    feature_dim: int,
    max_x_chunk_mb: int,
    hard_cap_rows: int,
) -> int:
    bytes_per_row = int(lookback) * int(feature_dim) * 4
    by_mem = max(1, int((int(max_x_chunk_mb) * 1024 * 1024) // max(1, bytes_per_row)))
    if requested_rows > 0:
        by_mem = min(by_mem, int(requested_rows))
    if hard_cap_rows > 0:
        by_mem = min(by_mem, int(hard_cap_rows))
    return max(1, by_mem)


def compute_safe_transform_chunk_rows(
    *,
    requested_rows: int,
    lookback: int,
    feature_dim: int,
    output_dim: int,
    max_x_chunk_mb: int,
    max_z_chunk_mb: int,
    hard_cap_rows: int,
) -> int:
    if output_dim <= 0:
        raise ValueError(f"output_dim must be > 0, got {output_dim}")

    bytes_per_x_row = int(lookback) * int(feature_dim) * 4
    bytes_per_z_row = int(output_dim) * 4

    by_x_mem = max(
        1,
        int((int(max_x_chunk_mb) * 1024 * 1024) // max(1, bytes_per_x_row)),
    )
    by_z_mem = max(
        1,
        int((int(max_z_chunk_mb) * 1024 * 1024) // max(1, bytes_per_z_row)),
    )

    rows = min(by_x_mem, by_z_mem)

    if requested_rows > 0:
        rows = min(rows, int(requested_rows))
    if hard_cap_rows > 0:
        rows = min(rows, int(hard_cap_rows))

    return max(1, int(rows))


def assert_transform_matches_labels(Z: np.ndarray, y: np.ndarray, split_name: str) -> None:
    if Z.ndim != 2:
        raise ValueError(f"{split_name}: Z must be 2D, got {Z.shape}")
    if y.ndim != 2:
        raise ValueError(f"{split_name}: y must be 2D, got {y.shape}")
    if Z.shape[0] != y.shape[0]:
        raise ValueError(f"{split_name}: Z rows {Z.shape[0]} != y rows {y.shape[0]}")


def collect_windows_for_positions(
    ds: Any,
    positions: np.ndarray,
    *,
    batch_rows: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape={positions.shape}")
    if positions.size <= 0:
        raise ValueError(f"Cannot collect zero rows for split={split_name}")
    batch_rows = max(1, int(batch_rows))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for start in range(0, int(positions.shape[0]), batch_rows):
        batch_pos = positions[start : start + batch_rows]
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for pos in batch_pos:
            x_i, y_i = ds[int(pos)]
            if hasattr(x_i, "detach"):
                x_i = x_i.detach().cpu().numpy()
            if hasattr(y_i, "detach"):
                y_i = y_i.detach().cpu().numpy()
            xs.append(np.asarray(x_i, dtype=np.float32))
            ys.append(np.asarray(y_i, dtype=np.float32))
        x_parts.append(np.stack(xs, axis=0).astype(np.float32, copy=False))
        y_parts.append(np.stack(ys, axis=0).astype(np.float32, copy=False))
    X = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0).astype(np.float32, copy=False)
    print(
        f"[linear-extract-collect] split={split_name} rows={X.shape[0]} "
        f"X_shape={list(X.shape)} y_shape={list(y.shape)}",
        flush=True,
    )
    return X, y


def collect_windows_from_dataset(
    ds: Any,
    *,
    max_rows: int,
    batch_rows: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions = _dataset_positions(len(ds), int(max_rows))
    return collect_windows_for_positions(ds, positions, batch_rows=batch_rows, split_name=split_name)


def collect_fit_windows_from_train(
    ds_train_list: list[Any],
    max_rows: int,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not ds_train_list:
        raise ValueError("No train datasets supplied for extractor fitting")
    if max_rows <= 0:
        max_rows = sum(len(ds) for ds in ds_train_list)
    per_week = int(np.ceil(max_rows / max(1, len(ds_train_list))))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for i, ds in enumerate(ds_train_list):
        rows_i = min(len(ds), per_week)
        X_i, y_i = collect_windows_from_dataset(
            ds,
            max_rows=rows_i,
            batch_rows=batch_rows,
            split_name=f"train_fit_week{i}",
        )
        x_parts.append(X_i)
        y_parts.append(y_i)
    X = np.concatenate(x_parts, axis=0)[:max_rows].astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0)[:max_rows].astype(np.float32, copy=False)
    print(
        f"[linear-extract-collect] split=train_fit rows={X.shape[0]} "
        f"X_shape={list(X.shape)} y_shape={list(y.shape)}",
        flush=True,
    )
    return X, y


ABS_SAMPLE_MAX = 2_000_000


def _empty_streaming_stats() -> Dict[str, Any]:
    return {
        "total_rows": 0,
        "output_dim": None,
        "sum": 0.0,
        "sumsq": 0.0,
        "zero_count": 0,
        "finite_count": 0,
        "total_count": 0,
        "abs_sample_parts": [],
    }


def _update_streaming_stats(stats: Dict[str, Any], Z: np.ndarray) -> None:
    vals = Z.reshape(-1)
    stats["total_rows"] += int(Z.shape[0])
    stats["output_dim"] = int(Z.shape[1])
    stats["total_count"] += int(vals.size)
    stats["finite_count"] += int(np.isfinite(vals).sum())
    stats["sum"] += float(vals.sum(dtype=np.float64))
    stats["sumsq"] += float(np.square(vals, dtype=np.float64).sum())
    stats["zero_count"] += int(np.count_nonzero(vals == 0.0))
    abs_vals = np.abs(vals)
    if abs_vals.size > 0:
        parts = stats["abs_sample_parts"]
        stride = max(1, abs_vals.size // max(1, ABS_SAMPLE_MAX // max(1, len(parts) + 1)))
        parts.append(abs_vals[::stride].astype(np.float32, copy=False))


def _finalize_streaming_summary(stats: Dict[str, Any], *, n_shards: int, chunk_rows: int, positions_rows: int) -> Dict[str, Any]:
    total_count = int(stats["total_count"])
    if total_count <= 0:
        raise ValueError("Cannot summarize empty transform output")
    mean = float(stats["sum"] / total_count)
    var = max(0.0, float(stats["sumsq"] / total_count - mean * mean))
    sample_parts = stats["abs_sample_parts"]
    sample = np.concatenate(sample_parts, axis=0) if sample_parts else np.zeros((0,), dtype=np.float32)
    if sample.size > ABS_SAMPLE_MAX:
        sample = sample[:ABS_SAMPLE_MAX]
    return {
        "shape": [int(stats["total_rows"]), int(stats["output_dim"])],
        "dtype": "float32",
        "finite_frac": float(stats["finite_count"] / total_count),
        "mean": mean,
        "std": float(math.sqrt(var)),
        "abs_p50": float(np.percentile(sample, 50)) if sample.size else float("nan"),
        "abs_p95": float(np.percentile(sample, 95)) if sample.size else float("nan"),
        "abs_p99": float(np.percentile(sample, 99)) if sample.size else float("nan"),
        "zero_frac": float(stats["zero_count"] / total_count),
        "n_shards": int(n_shards),
        "chunk_rows": int(chunk_rows),
        "positions_rows": int(positions_rows),
    }


def transform_dataset_to_npz_shards(
    *,
    extractor: Any,
    ds: Any,
    split_name: str,
    out_dir: Path,
    max_rows: int,
    collect_batch_rows: int,
    transform_shard_rows: int,
    max_x_chunk_mb: int,
    max_z_chunk_mb: int,
    extractor_output_dim: int,
    save_transforms: bool,
) -> Dict[str, Any]:
    positions = _dataset_positions(len(ds), int(max_rows))
    n_total = int(positions.shape[0])
    feature_dim = int(ds.feature_dim_total)
    chunk_rows = compute_safe_transform_chunk_rows(
        requested_rows=n_total,
        lookback=LOOKBACK,
        feature_dim=feature_dim,
        output_dim=extractor_output_dim,
        max_x_chunk_mb=max_x_chunk_mb,
        max_z_chunk_mb=max_z_chunk_mb,
        hard_cap_rows=transform_shard_rows,
    )
    print(
        f"[linear-memory] split={split_name} rows={n_total} "
        f"estimated_full_X_mb={estimate_x_window_mb(n_total, LOOKBACK, feature_dim):.1f} "
        f"estimated_full_Z_mb={estimate_matrix_mb(n_total, extractor_output_dim):.1f} "
        f"chunk_rows={chunk_rows} "
        f"estimated_chunk_X_mb={estimate_x_window_mb(chunk_rows, LOOKBACK, feature_dim):.1f} "
        f"estimated_chunk_Z_mb={estimate_matrix_mb(chunk_rows, extractor_output_dim):.1f} "
        f"output_dim={extractor_output_dim}",
        flush=True,
    )

    shards: list[Dict[str, Any]] = []
    stats = _empty_streaming_stats()
    seconds_total = 0.0
    processed_chunks = 0
    for shard_idx, start in enumerate(range(0, n_total, chunk_rows)):
        processed_chunks += 1
        pos_chunk = positions[start : start + chunk_rows]
        X_chunk, y_chunk = collect_windows_for_positions(
            ds,
            pos_chunk,
            batch_rows=collect_batch_rows,
            split_name=f"{split_name}_shard_{shard_idx:05d}",
        )
        t0 = time.time()
        Z_chunk = extractor.transform(X_chunk).astype(np.float32, copy=False)
        dt = time.time() - t0
        seconds_total += dt
        assert_transform_matches_labels(Z_chunk, y_chunk, f"{split_name}_shard_{shard_idx:05d}")
        if Z_chunk.shape[0] != pos_chunk.shape[0]:
            raise ValueError(
                f"{split_name}_shard_{shard_idx:05d}: Z rows {Z_chunk.shape[0]} != positions rows {pos_chunk.shape[0]}"
            )
        if not np.isfinite(Z_chunk).all():
            raise ValueError(f"Stage 2 extractor produced non-finite values for split={split_name} shard={shard_idx:05d}")
        print(
            f"[linear-extractor-transform] split={split_name} shard={shard_idx:05d} "
            f"rows={Z_chunk.shape[0]} Z_shape={list(Z_chunk.shape)} seconds={dt:.3f}",
            flush=True,
        )
        if save_transforms:
            path = out_dir / f"{split_name}_transform_shard_{shard_idx:05d}.npz"
            np.savez_compressed(
                path,
                Z=Z_chunk,
                y=y_chunk.astype(np.float32, copy=False),
                positions=pos_chunk.astype(np.int64, copy=False),
            )
            print(f"[linear-transform-shard] wrote {path}", flush=True)
            shards.append(
                {
                    "shard": int(shard_idx),
                    "path": str(path),
                    "rows": int(Z_chunk.shape[0]),
                    "z_shape": [int(x) for x in Z_chunk.shape],
                    "y_shape": [int(x) for x in y_chunk.shape],
                    "positions_start": int(pos_chunk[0]),
                    "positions_end": int(pos_chunk[-1]),
                    "seconds": float(dt),
                }
            )
        _update_streaming_stats(stats, Z_chunk)

    summary = _finalize_streaming_summary(stats, n_shards=processed_chunks, chunk_rows=chunk_rows, positions_rows=n_total)
    if summary["finite_frac"] < 1.0:
        raise ValueError(f"Stage 2 extractor produced non-finite values for split={split_name}: {summary}")
    manifest_path = out_dir / f"{split_name}_transform_manifest.json"
    manifest = {
        "split": split_name,
        "format": LINEAR_TRANSFORM_SAVE_FORMAT,
        "save_transforms": bool(save_transforms),
        "n_rows": int(stats["total_rows"]),
        "extractor_output_dim": int(extractor_output_dim),
        "max_z_chunk_mb": int(max_z_chunk_mb),
        "processed_chunks": int(processed_chunks),
        "n_saved_shards": len(shards),
        "n_shards": len(shards),
        "seconds_total": float(seconds_total),
        "summary": summary,
        "shards": shards,
        "manifest_path": str(manifest_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, allow_nan=True, indent=2)
    print(f"[linear-transform-manifest] wrote {manifest_path}", flush=True)
    return manifest


def transform_array_to_npz_shards(
    *,
    extractor: Any,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    out_dir: Path,
    transform_shard_rows: int,
    max_z_chunk_mb: int,
    extractor_output_dim: int,
    save_transforms: bool,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n_total = int(X.shape[0])
    if n_total <= 0:
        raise ValueError(f"Cannot transform empty array split={split_name}")
    if extractor_output_dim <= 0:
        raise ValueError(f"extractor_output_dim must be > 0, got {extractor_output_dim}")
    z_cap_rows = max(
        1,
        int((int(max_z_chunk_mb) * 1024 * 1024) // max(1, int(extractor_output_dim) * 4)),
    )
    chunk_rows = max(1, min(n_total, int(transform_shard_rows), z_cap_rows))
    print(
        f"[linear-memory] split={split_name} rows={n_total} "
        f"estimated_full_Z_mb={estimate_matrix_mb(n_total, extractor_output_dim):.1f} "
        f"chunk_rows={chunk_rows} "
        f"estimated_chunk_Z_mb={estimate_matrix_mb(chunk_rows, extractor_output_dim):.1f} "
        f"output_dim={extractor_output_dim}",
        flush=True,
    )
    shards: list[Dict[str, Any]] = []
    stats = _empty_streaming_stats()
    seconds_total = 0.0
    processed_chunks = 0
    for shard_idx, start in enumerate(range(0, n_total, chunk_rows)):
        processed_chunks += 1
        end = min(n_total, start + chunk_rows)
        y_chunk = y[start:end]
        positions = np.arange(start, end, dtype=np.int64)
        t0 = time.time()
        Z_chunk = extractor.transform(X[start:end]).astype(np.float32, copy=False)
        dt = time.time() - t0
        seconds_total += dt
        assert_transform_matches_labels(Z_chunk, y_chunk, f"{split_name}_shard_{shard_idx:05d}")
        if Z_chunk.shape[0] != positions.shape[0]:
            raise ValueError(
                f"{split_name}_shard_{shard_idx:05d}: Z rows {Z_chunk.shape[0]} != positions rows {positions.shape[0]}"
            )
        if not np.isfinite(Z_chunk).all():
            raise ValueError(f"Stage 2 extractor produced non-finite values for split={split_name} shard={shard_idx:05d}")
        print(
            f"[linear-extractor-transform] split={split_name} shard={shard_idx:05d} "
            f"rows={Z_chunk.shape[0]} Z_shape={list(Z_chunk.shape)} seconds={dt:.3f}",
            flush=True,
        )
        if save_transforms:
            path = out_dir / f"{split_name}_transform_shard_{shard_idx:05d}.npz"
            np.savez_compressed(path, Z=Z_chunk, y=y_chunk.astype(np.float32, copy=False), positions=positions)
            print(f"[linear-transform-shard] wrote {path}", flush=True)
            shards.append(
                {
                    "shard": int(shard_idx),
                    "path": str(path),
                    "rows": int(Z_chunk.shape[0]),
                    "z_shape": [int(x) for x in Z_chunk.shape],
                    "y_shape": [int(x) for x in y_chunk.shape],
                    "positions_start": int(positions[0]),
                    "positions_end": int(positions[-1]),
                    "seconds": float(dt),
                }
            )
        _update_streaming_stats(stats, Z_chunk)

    summary = _finalize_streaming_summary(stats, n_shards=processed_chunks, chunk_rows=chunk_rows, positions_rows=n_total)
    if summary["finite_frac"] < 1.0:
        raise ValueError(f"Stage 2 extractor produced non-finite values for split={split_name}: {summary}")
    manifest_path = out_dir / f"{split_name}_transform_manifest.json"
    manifest = {
        "split": split_name,
        "format": LINEAR_TRANSFORM_SAVE_FORMAT,
        "save_transforms": bool(save_transforms),
        "positions_reference": "train_fit_sample_order",
        "n_rows": int(stats["total_rows"]),
        "extractor_output_dim": int(extractor_output_dim),
        "max_z_chunk_mb": int(max_z_chunk_mb),
        "processed_chunks": int(processed_chunks),
        "n_saved_shards": len(shards),
        "n_shards": len(shards),
        "seconds_total": float(seconds_total),
        "summary": summary,
        "shards": shards,
        "manifest_path": str(manifest_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, allow_nan=True, indent=2)
    print(f"[linear-transform-manifest] wrote {manifest_path}", flush=True)
    return manifest


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def resolve_stage2_dir(linear_out_dir: Path, extractor_name: str) -> Path:
    return Path(linear_out_dir) / "stage2_extractors" / str(extractor_name)


def load_stage2_payload(linear_out_dir: Path, extractor_name: str) -> Dict[str, Any]:
    stage2_dir = resolve_stage2_dir(linear_out_dir, extractor_name)
    path = stage2_dir / "linear_stage2_extractor_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 payload not found for extractor={extractor_name!r}: {path}")
    payload = load_json(path)
    if payload.get("stage") != "stage2":
        raise ValueError(f"Expected stage2 payload at {path}, got stage={payload.get('stage')!r}")
    if "extractor_output_dim" not in payload:
        raise ValueError(f"Stage 2 payload missing extractor_output_dim: {path}")
    print(f"[linear-stage3] loaded stage2 payload {path}", flush=True)
    return payload


def load_manifest_from_payload(payload: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    manifests = payload.get("manifests", {})
    manifest = manifests.get(split)
    if manifest is None:
        return None
    path = Path(str(manifest.get("manifest_path", ""))) if isinstance(manifest, dict) else Path("")
    if path.exists():
        loaded = load_json(path)
        if "manifest_path" not in loaded:
            loaded["manifest_path"] = str(path)
        return loaded
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest payload for split={split}: {manifest!r}")
    return manifest


def _apply_preprocess_nonfinite_policy(Z: np.ndarray, *, policy: str, context: str) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float32)
    if np.isfinite(Z).all():
        return Z
    if policy == "raise":
        raise ValueError(f"{context} contains non-finite values")
    if policy == "warn_zero":
        print(f"[linear-preprocess-warn] {context} contains non-finite values; replacing with zero", flush=True)
        return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported nonfinite_policy {policy!r}")


def _manifest_dim(manifest: Dict[str, Any]) -> int:
    summary = manifest.get("summary", {})
    if "shape" in summary and len(summary["shape"]) >= 2:
        return int(summary["shape"][1])
    if "extractor_output_dim" in manifest:
        return int(manifest["extractor_output_dim"])
    raise ValueError("Manifest missing summary.shape[1] / extractor_output_dim")


def load_sample_rows_from_manifest(
    manifest: Dict[str, Any],
    *,
    max_rows: int,
    max_matrix_mb: int,
    split_name: str,
) -> np.ndarray:
    D = _manifest_dim(manifest)
    n_rows = int(manifest.get("n_rows", 0))
    if n_rows <= 0:
        raise ValueError(f"Cannot sample empty manifest split={split_name}")
    row_cap_by_mem = max(1, int((int(max_matrix_mb) * 1024 * 1024) // max(1, D * 4)))
    n_sample = min(n_rows, int(max_rows), row_cap_by_mem)
    sample_global = np.linspace(0, n_rows - 1, n_sample, dtype=np.int64)
    parts: list[np.ndarray] = []
    cursor = 0
    for shard in manifest.get("shards", []):
        rows = int(shard.get("rows", 0))
        if rows <= 0:
            continue
        shard_start = cursor
        shard_end = cursor + rows
        mask = (sample_global >= shard_start) & (sample_global < shard_end)
        if mask.any():
            local = sample_global[mask] - shard_start
            with np.load(shard["path"]) as arr:
                Z = np.asarray(arr["Z"], dtype=np.float32)
                if Z.ndim != 2 or Z.shape[1] != D:
                    raise ValueError(f"Shard {shard['path']} has invalid Z shape {Z.shape}; expected width {D}")
                parts.append(Z[local])
        cursor = shard_end
    if cursor != n_rows:
        raise ValueError(f"Manifest split={split_name} shard rows {cursor} != n_rows {n_rows}")
    Z_sample = np.concatenate(parts, axis=0) if parts else np.zeros((0, D), dtype=np.float32)
    if Z_sample.shape != (n_sample, D):
        raise ValueError(f"Sample shape mismatch for {split_name}: got {Z_sample.shape}, expected {(n_sample, D)}")
    print(
        f"[linear-preprocess-fit-sample] split={split_name} rows={n_sample} dim={D} "
        f"matrix_mb={estimate_matrix_mb(n_sample, D):.1f}",
        flush=True,
    )
    return Z_sample.astype(np.float32, copy=False)


def fit_linear_preprocessor_from_manifest(
    train_manifest: Dict[str, Any],
    *,
    config: Dict[str, Any],
) -> LinearPreprocessBundle:
    D = _manifest_dim(train_manifest)
    policy = str(config.get("nonfinite_policy", "raise"))
    Z_sample = load_sample_rows_from_manifest(
        train_manifest,
        max_rows=int(config["fit_max_rows"]),
        max_matrix_mb=int(config["fit_max_matrix_mb"]),
        split_name="train_sample",
    )
    Z_sample = _apply_preprocess_nonfinite_policy(Z_sample, policy=policy, context="train_sample quantile sample")
    winsorize = bool(config.get("winsorize", True))
    if winsorize:
        lower = np.quantile(Z_sample, float(config["winsor_q_lo"]), axis=0).astype(np.float32)
        upper = np.quantile(Z_sample, float(config["winsor_q_hi"]), axis=0).astype(np.float32)
    else:
        lower = np.full(D, -np.inf, dtype=np.float32)
        upper = np.full(D, np.inf, dtype=np.float32)
    if winsorize:
        bad = ~np.isfinite(lower) | ~np.isfinite(upper) | (lower > upper)
    else:
        bad = lower > upper
    if bad.any():
        raise ValueError(f"Invalid winsor caps for {int(bad.sum())} features")

    count = 0
    sum_ = np.zeros(D, dtype=np.float64)
    sumsq = np.zeros(D, dtype=np.float64)
    for shard in train_manifest.get("shards", []):
        with np.load(shard["path"]) as arr:
            Z = _apply_preprocess_nonfinite_policy(arr["Z"], policy=policy, context=f"train shard {shard['path']}")
        if Z.ndim != 2 or Z.shape[1] != D:
            raise ValueError(f"Train shard {shard['path']} has invalid Z shape {Z.shape}; expected width {D}")
        Zc = np.clip(Z, lower, upper)
        sum_ += Zc.sum(axis=0, dtype=np.float64)
        sumsq += np.square(Zc, dtype=np.float64).sum(axis=0)
        count += int(Zc.shape[0])
    if count <= 0:
        raise ValueError("Cannot fit preprocessor on empty train_sample manifest")
    mean64 = sum_ / count
    var64 = np.maximum(0.0, sumsq / count - mean64 * mean64)
    std64 = np.sqrt(var64)
    if bool(config.get("standardize", True)):
        mean = mean64.astype(np.float32)
        std = std64.astype(np.float32)
    else:
        mean = np.zeros(D, dtype=np.float32)
        std = np.ones(D, dtype=np.float32)
    if bool(config.get("variance_filter", True)):
        keep_mask = np.isfinite(std) & (std >= float(config.get("min_std", 0.0)))
    else:
        keep_mask = np.ones(D, dtype=bool)
    if not keep_mask.any():
        raise ValueError("Preprocessor variance filter removed all features")
    kept_dim = int(keep_mask.sum())
    fit_summary = {
        "fit_split": "train_sample",
        "original_dim": int(D),
        "kept_dim": kept_dim,
        "removed_dim": int(D - kept_dim),
        "fit_rows_for_quantiles": int(Z_sample.shape[0]),
        "fit_rows_for_mean_std": int(count),
        "winsorize": winsorize,
        "winsor_q_lo": float(config.get("winsor_q_lo", 0.0)),
        "winsor_q_hi": float(config.get("winsor_q_hi", 1.0)),
        "standardize": bool(config.get("standardize", True)),
        "variance_filter": bool(config.get("variance_filter", True)),
        "min_std": float(config.get("min_std", 0.0)),
        "std_eps": float(config.get("std_eps", 1e-6)),
        "std_min": float(np.nanmin(std)),
        "std_p50": float(np.nanpercentile(std, 50)),
        "std_p95": float(np.nanpercentile(std, 95)),
        "std_max": float(np.nanmax(std)),
        "lower_finite_frac": float(np.isfinite(lower).mean()),
        "upper_finite_frac": float(np.isfinite(upper).mean()),
    }
    print(
        f"[linear-preprocess-fit] original_dim={D} kept_dim={kept_dim} removed_dim={D - kept_dim}",
        flush=True,
    )
    return LinearPreprocessBundle(
        schema=str(config.get("schema", LINEAR_PREPROCESS_SCHEMA)),
        config=dict(config),
        original_dim=int(D),
        kept_dim=kept_dim,
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        keep_mask=keep_mask.astype(bool),
        fit_summary=fit_summary,
    )


def preprocess_manifest_to_npz_shards(
    *,
    bundle: LinearPreprocessBundle,
    source_manifest: Dict[str, Any],
    split_name: str,
    out_dir: Path,
    shard_rows: int,
    max_z_chunk_mb: int,
    bundle_path: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    z_cap_rows = max(1, int((int(max_z_chunk_mb) * 1024 * 1024) // max(1, int(bundle.kept_dim) * 4)))
    chunk_rows = max(1, min(int(shard_rows), z_cap_rows))
    shards: list[Dict[str, Any]] = []
    stats = _empty_streaming_stats()
    processed_chunks = 0
    seconds_total = 0.0
    shard_idx = 0
    total_rows = 0
    for source_shard in source_manifest.get("shards", []):
        with np.load(source_shard["path"]) as arr:
            Z_src = np.asarray(arr["Z"], dtype=np.float32)
            y = np.asarray(arr["y"], dtype=np.float32)
            positions = np.asarray(arr["positions"], dtype=np.int64)
        if Z_src.shape[0] != y.shape[0] or Z_src.shape[0] != positions.shape[0]:
            raise ValueError(f"Source shard row mismatch: {source_shard['path']}")
        for start in range(0, Z_src.shape[0], chunk_rows):
            end = min(Z_src.shape[0], start + chunk_rows)
            t0 = time.time()
            Z_out = bundle.transform(Z_src[start:end])
            dt = time.time() - t0
            y_out = y[start:end].astype(np.float32, copy=False)
            pos_out = positions[start:end].astype(np.int64, copy=False)
            if Z_out.shape[0] != y_out.shape[0] or Z_out.shape[0] != pos_out.shape[0]:
                raise ValueError(f"Output shard row mismatch for split={split_name} shard={shard_idx:05d}")
            if not np.isfinite(Z_out).all():
                raise ValueError(f"Stage 3 produced non-finite values for split={split_name} shard={shard_idx:05d}")
            path = out_dir / f"{split_name}_preprocessed_shard_{shard_idx:05d}.npz"
            np.savez_compressed(path, Z=Z_out, y=y_out, positions=pos_out)
            print(
                f"[linear-preprocess-transform] split={split_name} shard={shard_idx:05d} "
                f"rows={Z_out.shape[0]} Z_shape={list(Z_out.shape)} seconds={dt:.3f}",
                flush=True,
            )
            shards.append(
                {
                    "shard": int(shard_idx),
                    "path": str(path),
                    "rows": int(Z_out.shape[0]),
                    "z_shape": [int(x) for x in Z_out.shape],
                    "y_shape": [int(x) for x in y_out.shape],
                    "positions_start": int(pos_out[0]),
                    "positions_end": int(pos_out[-1]),
                    "source_shard_path": str(source_shard["path"]),
                    "source_local_start": int(start),
                    "source_local_end": int(end),
                    "seconds": float(dt),
                }
            )
            _update_streaming_stats(stats, Z_out)
            total_rows += int(Z_out.shape[0])
            seconds_total += dt
            processed_chunks += 1
            shard_idx += 1
    summary = _finalize_streaming_summary(stats, n_shards=len(shards), chunk_rows=chunk_rows, positions_rows=total_rows)
    manifest_path = out_dir / f"{split_name}_preprocessed_manifest.json"
    manifest = {
        "split": split_name,
        "stage": "stage3",
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "source_manifest_path": source_manifest.get("manifest_path"),
        "preprocess_bundle_path": str(bundle_path),
        "n_rows": int(total_rows),
        "original_dim": int(bundle.original_dim),
        "kept_dim": int(bundle.kept_dim),
        "processed_chunks": int(processed_chunks),
        "n_saved_shards": len(shards),
        "n_shards": len(shards),
        "seconds_total": float(seconds_total),
        "max_z_chunk_mb": int(max_z_chunk_mb),
        "summary": summary,
        "shards": shards,
        "manifest_path": str(manifest_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, allow_nan=True, indent=2)
    print(f"[linear-preprocess-manifest] wrote {manifest_path}", flush=True)
    return manifest


def run_stage3_preprocessing(
    *,
    linear_out_dir: Path,
    extractor_name: str,
) -> Dict[str, Any]:
    stage2_dir = resolve_stage2_dir(linear_out_dir, extractor_name)
    stage3_dir = Path(linear_out_dir) / "stage3_preprocess" / extractor_name / "default"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    stage2_payload = load_stage2_payload(linear_out_dir, extractor_name)
    train_manifest = load_manifest_from_payload(stage2_payload, "train_sample")
    val_manifest = load_manifest_from_payload(stage2_payload, "val")
    test_manifest = load_manifest_from_payload(stage2_payload, "test")
    if train_manifest is None:
        raise ValueError("Stage 3 requires a Stage 2 train_sample manifest")
    if val_manifest is None:
        raise ValueError("Stage 3 requires a Stage 2 val manifest")
    preprocess_config = {
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "extractor": extractor_name,
        "fit_split": LINEAR_PREPROCESS_FIT_SPLIT,
        "winsorize": bool(LINEAR_PREPROCESS_WINSORIZE),
        "winsor_q_lo": float(LINEAR_PREPROCESS_WINSOR_Q_LO),
        "winsor_q_hi": float(LINEAR_PREPROCESS_WINSOR_Q_HI),
        "standardize": bool(LINEAR_PREPROCESS_STANDARDIZE),
        "std_eps": float(LINEAR_PREPROCESS_STD_EPS),
        "variance_filter": bool(LINEAR_PREPROCESS_VARIANCE_FILTER),
        "min_std": float(LINEAR_PREPROCESS_MIN_STD),
        "post_clip_abs": float(LINEAR_PREPROCESS_POST_CLIP_ABS),
        "nonfinite_policy": LINEAR_PREPROCESS_NONFINITE_POLICY,
        "fit_max_rows": int(LINEAR_PREPROCESS_FIT_MAX_ROWS),
        "fit_max_matrix_mb": int(LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB),
        "preprocess_shard_rows": int(LINEAR_PREPROCESS_SHARD_ROWS),
        "preprocess_max_z_chunk_mb": int(LINEAR_PREPROCESS_MAX_Z_CHUNK_MB),
    }
    bundle = fit_linear_preprocessor_from_manifest(train_manifest, config=preprocess_config)
    bundle_path = stage3_dir / "linear_preprocess_bundle.npz"
    save_linear_preprocess_bundle(bundle, bundle_path)
    train_pre_manifest = preprocess_manifest_to_npz_shards(
        bundle=bundle,
        source_manifest=train_manifest,
        split_name="train_sample",
        out_dir=stage3_dir,
        shard_rows=LINEAR_PREPROCESS_SHARD_ROWS,
        max_z_chunk_mb=LINEAR_PREPROCESS_MAX_Z_CHUNK_MB,
        bundle_path=bundle_path,
    )
    val_pre_manifest = preprocess_manifest_to_npz_shards(
        bundle=bundle,
        source_manifest=val_manifest,
        split_name="val",
        out_dir=stage3_dir,
        shard_rows=LINEAR_PREPROCESS_SHARD_ROWS,
        max_z_chunk_mb=LINEAR_PREPROCESS_MAX_Z_CHUNK_MB,
        bundle_path=bundle_path,
    )
    test_pre_manifest = None
    if test_manifest is not None:
        test_pre_manifest = preprocess_manifest_to_npz_shards(
            bundle=bundle,
            source_manifest=test_manifest,
            split_name="test",
            out_dir=stage3_dir,
            shard_rows=LINEAR_PREPROCESS_SHARD_ROWS,
            max_z_chunk_mb=LINEAR_PREPROCESS_MAX_Z_CHUNK_MB,
            bundle_path=bundle_path,
        )
    payload = {
        "stage": "stage3",
        "status": "ok",
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "extractor": extractor_name,
        "stage2_dir": str(stage2_dir),
        "stage3_dir": str(stage3_dir),
        "preprocess_config": preprocess_config,
        "preprocess_bundle_path": str(bundle_path),
        "fit_summary": bundle.fit_summary,
        "original_dim": int(bundle.original_dim),
        "kept_dim": int(bundle.kept_dim),
        "train_sample_summary": train_pre_manifest["summary"],
        "val_summary": val_pre_manifest["summary"],
        "test_summary": None if test_pre_manifest is None else test_pre_manifest["summary"],
        "manifests": {
            "train_sample": train_pre_manifest,
            "val": val_pre_manifest,
            "test": test_pre_manifest,
        },
    }
    metrics_path = stage3_dir / "linear_stage3_preprocess_metrics.json"
    copy_path = Path(linear_out_dir) / "linear_stage3_preprocess_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=True, indent=2)
    with copy_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=True, indent=2)
    print(f"[linear-stage3] wrote {metrics_path} and {copy_path}", flush=True)
    return payload


def run_stage2_extraction(
    *,
    linear_out_dir: Path,
    ds_train_list: list[Any],
    ds_val: Any,
    ds_test: Optional[Any],
    has_cmssl_test: bool,
    meta: Dict[str, Any],
    protocol: str,
    train_week_keys: list[str],
    extractor_config: Dict[str, Any],
) -> Dict[str, Any]:
    extractor_name = str(extractor_config["extractor"]).strip().lower()
    stage2_dir = linear_out_dir / "stage2_extractors" / extractor_name
    stage2_dir.mkdir(parents=True, exist_ok=True)

    if not LINEAR_CHUNKED_TRANSFORMS:
        raise ValueError("Stage 2 currently supports only BYBIT_LINEAR_CHUNKED_TRANSFORMS=1")

    extractor = build_linear_extractor_from_config(extractor_config)
    X_fit, y_fit = collect_fit_windows_from_train(
        ds_train_list,
        max_rows=LINEAR_EXTRACTOR_FIT_MAX_ROWS,
        batch_rows=LINEAR_EXTRACT_BATCH_ROWS,
    )
    t0 = time.time()
    extractor.fit(X_fit)
    fit_seconds = time.time() - t0
    print(
        f"[linear-extractor-fit] name={extractor.name} rows={X_fit.shape[0]} seconds={fit_seconds:.3f}",
        flush=True,
    )

    X_probe = X_fit[: min(128, X_fit.shape[0])]
    Z_probe = extractor.transform(X_probe).astype(np.float32, copy=False)
    if Z_probe.ndim != 2:
        raise ValueError(f"Extractor probe produced non-2D output shape {Z_probe.shape}")
    if Z_probe.shape[0] != X_probe.shape[0]:
        raise ValueError(
            f"Extractor probe rows {Z_probe.shape[0]} != X probe rows {X_probe.shape[0]}"
        )
    if not np.isfinite(Z_probe).all():
        raise ValueError("Extractor probe produced non-finite values")
    extractor_output_dim = int(Z_probe.shape[1])
    print(
        f"[linear-extractor-output] name={extractor.name} "
        f"output_dim={extractor_output_dim} "
        f"probe_rows={Z_probe.shape[0]}",
        flush=True,
    )

    train_sample_manifest = transform_array_to_npz_shards(
        extractor=extractor,
        X=X_fit,
        y=y_fit,
        split_name="train_sample",
        out_dir=stage2_dir,
        transform_shard_rows=LINEAR_TRANSFORM_SHARD_ROWS,
        max_z_chunk_mb=LINEAR_MAX_Z_CHUNK_MB,
        extractor_output_dim=extractor_output_dim,
        save_transforms=LINEAR_SAVE_TRANSFORMS,
    )
    val_manifest = transform_dataset_to_npz_shards(
        extractor=extractor,
        ds=ds_val,
        split_name="val",
        out_dir=stage2_dir,
        max_rows=LINEAR_TRANSFORM_MAX_ROWS_PER_SPLIT,
        collect_batch_rows=LINEAR_EXTRACT_BATCH_ROWS,
        transform_shard_rows=LINEAR_TRANSFORM_SHARD_ROWS,
        max_x_chunk_mb=LINEAR_MAX_X_CHUNK_MB,
        max_z_chunk_mb=LINEAR_MAX_Z_CHUNK_MB,
        extractor_output_dim=extractor_output_dim,
        save_transforms=LINEAR_SAVE_TRANSFORMS,
    )

    test_manifest = None
    if has_cmssl_test and ds_test is not None and LINEAR_RUN_TEST:
        test_manifest = transform_dataset_to_npz_shards(
            extractor=extractor,
            ds=ds_test,
            split_name="test",
            out_dir=stage2_dir,
            max_rows=LINEAR_TRANSFORM_MAX_ROWS_PER_SPLIT,
            collect_batch_rows=LINEAR_EXTRACT_BATCH_ROWS,
            transform_shard_rows=LINEAR_TRANSFORM_SHARD_ROWS,
            max_x_chunk_mb=LINEAR_MAX_X_CHUNK_MB,
            max_z_chunk_mb=LINEAR_MAX_Z_CHUNK_MB,
            extractor_output_dim=extractor_output_dim,
            save_transforms=LINEAR_SAVE_TRANSFORMS,
        )

    saved_files: Dict[str, Optional[str]] = {
        "train_sample_manifest": train_sample_manifest.get("manifest_path"),
        "val_manifest": val_manifest.get("manifest_path"),
        "test_manifest": None if test_manifest is None else test_manifest.get("manifest_path"),
        "train_sample_transform_manifest": train_sample_manifest.get("manifest_path"),
        "val_transform_manifest": val_manifest.get("manifest_path"),
        "test_transform_manifest": None if test_manifest is None else test_manifest.get("manifest_path"),
        "extractor_pickle": None,
        "stage2_metrics": None,
        "stage2_metrics_copy": None,
    }

    extractor_pickle_saved = False
    pkl_path = stage2_dir / "extractor.pkl"
    try:
        with pkl_path.open("wb") as f:
            pickle.dump(extractor, f)
        extractor_pickle_saved = True
        saved_files["extractor_pickle"] = str(pkl_path)
    except Exception as exc:
        print(
            f"[linear-extractor-warn] pickle failed; transformed matrices and metadata were still saved: {exc}",
            flush=True,
        )

    stage2_payload: Dict[str, Any] = {
        "stage": "stage2",
        "status": "ok",
        "linear_extractor_schema": LINEAR_EXTRACTOR_SCHEMA,
        "extractor_config": extractor_config,
        "extractor_summary": extractor.summary(),
        "out_root": str(OUT_ROOT),
        "linear_out_dir": str(linear_out_dir),
        "stage2_dir": str(stage2_dir),
        "protocol": protocol,
        "train_week_keys": train_week_keys,
        "feature_dim_total": int(meta["feature_dim_total"]),
        "lookback": LOOKBACK,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "fit_rows": int(X_fit.shape[0]),
        "fit_seconds": float(fit_seconds),
        "chunked_transforms": True,
        "transform_save_format": LINEAR_TRANSFORM_SAVE_FORMAT,
        "transform_shard_rows": int(LINEAR_TRANSFORM_SHARD_ROWS),
        "max_x_chunk_mb": int(LINEAR_MAX_X_CHUNK_MB),
        "max_z_chunk_mb": int(LINEAR_MAX_Z_CHUNK_MB),
        "extractor_output_dim": int(extractor_output_dim),
        "transform_seconds": {
            "train_sample": float(train_sample_manifest["seconds_total"]),
            "val": float(val_manifest["seconds_total"]),
            "test": None if test_manifest is None else float(test_manifest["seconds_total"]),
        },
        "train_sample_summary": train_sample_manifest["summary"],
        "val_summary": val_manifest["summary"],
        "test_summary": None if test_manifest is None else test_manifest["summary"],
        "manifests": {
            "train_sample": train_sample_manifest,
            "val": val_manifest,
            "test": test_manifest,
        },
        "extractor_pickle_saved": bool(extractor_pickle_saved),
        "save_transforms": bool(LINEAR_SAVE_TRANSFORMS),
        "saved_files": saved_files,
    }

    metrics_path = stage2_dir / "linear_stage2_extractor_metrics.json"
    copy_path = linear_out_dir / "linear_stage2_extractor_metrics.json"
    saved_files["stage2_metrics"] = str(metrics_path)
    saved_files["stage2_metrics_copy"] = str(copy_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(stage2_payload, f, allow_nan=True, indent=2)
    with copy_path.open("w", encoding="utf-8") as f:
        json.dump(stage2_payload, f, allow_nan=True, indent=2)
    print(f"[linear-extractor-summary] {json.dumps(extractor.summary(), allow_nan=True)}", flush=True)
    print(f"[linear-stage2] wrote {metrics_path} and {copy_path}", flush=True)
    return stage2_payload


def main() -> None:
    out_root = Path(OUT_ROOT)
    linear_out_dir = Path(LINEAR_OUT_DIR) if LINEAR_OUT_DIR else out_root / "linear_stage1"
    linear_out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    if LINEAR_STAGE == "stage3":
        print(
            f"[linear-config] stage=stage3 extractor={LINEAR_EXTRACTOR} device={device} "
            f"out_root={out_root} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        run_stage3_preprocessing(linear_out_dir=linear_out_dir, extractor_name=LINEAR_EXTRACTOR)
        return
    if LINEAR_STAGE == "stage2":
        print(
            f"[linear-config] stage={LINEAR_STAGE} extractor={LINEAR_EXTRACTOR} device={device} "
            f"batch_size={LINEAR_EVAL_BATCH_SIZE} run_test={int(LINEAR_RUN_TEST)} "
            f"out_root={out_root} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        extractor_config: Optional[Dict[str, Any]] = _build_extractor_config()
        print(f"[linear-extractor-config] {json.dumps(extractor_config, sort_keys=True)}", flush=True)
    else:
        print(
            f"[linear-config] stage={LINEAR_STAGE} device={device} batch_size={LINEAR_EVAL_BATCH_SIZE} "
            f"run_test={int(LINEAR_RUN_TEST)} out_root={out_root} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        extractor_config = None

    meta = json.loads((out_root / "meta.json").read_text())
    validate_contract_meta(meta, "global meta.json")
    validate_dataset_label_dim(meta, "global meta.json")
    split_info = require_supported_pipeline_splits(meta, out_root)
    protocol = split_info["protocol"]
    splits = split_info["splits"]
    cmssl = splits["cmssl"]
    train_week_keys = list(cmssl["train"]["weeks"])
    cmssl_val = cmssl["val"]
    cmssl_test = cmssl.get("test")
    has_cmssl_test = cmssl_test is not None and bool(cmssl_test.get("weeks"))
    print(
        f"[split] protocol={protocol} cmssl.train={','.join(train_week_keys)} "
        f"cmssl.val={cmssl_val.get('weeks')} "
        f"cmssl.test={cmssl_test.get('weeks') if has_cmssl_test else '<missing>'}",
        flush=True,
    )

    train_split_entries = [
        make_single_week_split_from_meta(out_root=out_root, global_meta=meta, week_key=wk)
        for wk in train_week_keys
    ]
    ds_train_list = [build_dataset_from_split(str(out_root), entry) for entry in train_split_entries]
    ds_val = build_dataset_from_split(str(out_root), cmssl_val)
    ds_test = build_dataset_from_split(str(out_root), cmssl_test) if has_cmssl_test else None

    feature_dim_total = int(meta.get("feature_dim_total", 0))
    for i, ds_train in enumerate(ds_train_list):
        _validate_dataset_split(ds_train, f"train[{i}]/{train_week_keys[i]}", feature_dim_total)
    _validate_dataset_split(ds_val, "val", feature_dim_total)
    if ds_test is not None:
        _validate_dataset_split(ds_test, "test", feature_dim_total)

    y_train = np.concatenate([np.asarray(ds.y, dtype=np.float32) for ds in ds_train_list], axis=0)
    validate_loaded_label_array(y_train, "linear train labels")

    cache_path = linear_out_dir / "linear_signed_side_trim_stats_cache.npz"
    cache_meta = _make_cache_meta(meta, protocol, train_week_keys, train_split_entries)
    cached = load_stats_cache(cache_path)
    if cached and cache_matches(cached[1], cache_meta):
        stats = cached[0]
        print(f"[linear-train-stats] loaded_cache={cache_path}", flush=True)
    else:
        stats = compute_signed_raw_stats(y_train)
        save_stats_cache(cache_path, stats, cache_meta)
        print(f"[linear-train-stats] wrote_cache={cache_path}", flush=True)

    dir_pos_w, dir_neg_w = compute_dir_class_weights_from_train_labels(
        y_train,
        pos_lo=stats["pos_lo_raw_bps"],
        pos_hi=stats["pos_hi_raw_bps"],
        neg_lo=stats["neg_lo_abs_bps"],
        neg_hi=stats["neg_hi_abs_bps"],
    )
    mag_pos_init_sqrt, mag_neg_init_sqrt = compute_mag_init_targets_from_train_labels(
        y_train,
        pos_lo=stats["pos_lo_raw_bps"],
        pos_hi=stats["pos_hi_raw_bps"],
        neg_lo=stats["neg_lo_abs_bps"],
        neg_hi=stats["neg_hi_abs_bps"],
        pos_q50=stats["kept_pos_q50_abs_raw_bps"],
        neg_q50=stats["kept_neg_q50_abs_raw_bps"],
    )
    print(f"[linear-train-stats] dir_pos_w={dir_pos_w.tolist()} dir_neg_w={dir_neg_w.tolist()}", flush=True)
    print(
        f"[linear-prior-mag] pos_target_sqrt={mag_pos_init_sqrt.tolist()} "
        f"neg_target_sqrt={mag_neg_init_sqrt.tolist()}",
        flush=True,
    )

    train_keep_pos, train_keep_neg, train_keep_signed = build_signed_side_trim_masks_from_stats_np(y_train, stats)
    prior_info = build_constant_priors_from_train_labels(
        y_train=y_train,
        stats=stats,
        mag_up_sqrt_prior=mag_pos_init_sqrt,
        mag_down_sqrt_prior=mag_neg_init_sqrt,
        keep_pos=train_keep_pos,
        keep_neg=train_keep_neg,
        keep_signed=train_keep_signed,
    )
    print(
        f"[linear-prior] p_up={prior_info['p_up_prior'].tolist()} "
        f"dir_logit={prior_info['dir_logit_prior'].tolist()}",
        flush=True,
    )
    model = LinearConstantPriorModel(
        prior_info["dir_logit_prior"],
        prior_info["mag_up_sqrt_prior"],
        prior_info["mag_down_sqrt_prior"],
    ).to(device)
    model.eval()

    if LINEAR_STAGE == "stage2":
        run_stage2_extraction(
            linear_out_dir=linear_out_dir,
            ds_train_list=ds_train_list,
            ds_val=ds_val,
            ds_test=ds_test,
            has_cmssl_test=has_cmssl_test,
            meta=meta,
            protocol=protocol,
            train_week_keys=train_week_keys,
            extractor_config=extractor_config or _build_extractor_config(),
        )
        if not LINEAR_STAGE2_RUN_PRIOR_EVAL:
            return

    val_full_src = CPUWindowBatchSource(
        ds_val,
        device,
        LINEAR_EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        row_stride=1,
    )
    val_fast_src = val_full_src.make_evenly_spaced_subset(FAST_VAL_MAX_ROWS)

    train_band_metrics: Optional[Dict[str, Any]] = None
    if BAND_DIAG and BAND_DIAG_TRAIN:
        train_eval_row_stride = max(1, int(os.environ.get("BYBIT_LINEAR_TRAIN_EVAL_ROW_STRIDE", "1")))
        train_sources = [
            CPUWindowBatchSource(
                ds,
                device,
                LINEAR_EVAL_BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                row_stride=train_eval_row_stride,
            )
            for ds in ds_train_list
        ]
        train_band_src = make_train_band_eval_source(train_sources, BAND_DIAG_TRAIN_MAX_ROWS)
        train_band_metrics = summarize_metrics(
            model,
            train_band_src,
            device,
            stats,
            amp_enabled=False,
            amp_dtype=torch.float32,
            primary_only=True,
            epoch=0,
            band_diag=True,
            split_name="linear_train_band",
        )
        if "band_metrics" in train_band_metrics:
            print_band_metrics_summary(train_band_metrics["band_metrics"], split_name="linear_train_band", epoch=0)
            save_band_metrics_jsonl(linear_out_dir, train_band_metrics["band_metrics"], epoch=0, split_name="linear_train_band")

    val_fast = summarize_metrics(
        model,
        val_fast_src,
        device,
        stats,
        amp_enabled=False,
        amp_dtype=torch.float32,
        primary_only=True,
        epoch=0,
        band_diag=BAND_DIAG,
        split_name="linear_val_fast",
    )
    primary_metric_value, primary_metric_label = compute_primary_metric(val_fast)
    _print_primary("linear_val_fast", val_fast, primary_metric_value, primary_metric_label)

    val_full = summarize_metrics(
        model,
        val_full_src,
        device,
        stats,
        amp_enabled=False,
        amp_dtype=torch.float32,
        primary_only=False,
        epoch=0,
        band_diag=BAND_DIAG,
        split_name="linear_val_full",
    )
    val_full_primary_value, val_full_primary_label = compute_primary_metric(val_full)
    _print_primary("linear_val_full", val_full, val_full_primary_value, val_full_primary_label)
    if BAND_DIAG and "band_metrics" in val_full:
        print_band_metrics_summary(val_full["band_metrics"], split_name="linear_val_full", epoch=0)
        save_band_metrics_jsonl(linear_out_dir, val_full["band_metrics"], epoch=0, split_name="linear_val_full")

    test_metrics: Optional[Dict[str, Any]] = None
    if LINEAR_RUN_TEST and ds_test is not None:
        test_src = CPUWindowBatchSource(
            ds_test,
            device,
            LINEAR_EVAL_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            row_stride=1,
        )
        test_metrics = summarize_metrics(
            model,
            test_src,
            device,
            stats,
            amp_enabled=False,
            amp_dtype=torch.float32,
            primary_only=False,
            epoch=0,
            band_diag=BAND_DIAG,
            split_name="linear_test",
        )
        test_primary_value, test_primary_label = compute_primary_metric(test_metrics)
        _print_primary("linear_test", test_metrics, test_primary_value, test_primary_label)
        if BAND_DIAG and "band_metrics" in test_metrics:
            print_band_metrics_summary(test_metrics["band_metrics"], split_name="linear_test", epoch=0)
            save_band_metrics_jsonl(linear_out_dir, test_metrics["band_metrics"], epoch=0, split_name="linear_test")

    metrics_payload = {
        "stage": "stage1",
        "status": "ok",
        "out_root": str(out_root),
        "linear_out_dir": str(linear_out_dir),
        "protocol": protocol,
        "train_week_keys": train_week_keys,
        "val_weeks": cmssl_val.get("weeks"),
        "test_weeks": cmssl_test.get("weeks") if has_cmssl_test else None,
        "feature_dim_total": feature_dim_total,
        "lookback": LOOKBACK,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "target_task": TARGET_TASK,
        "target_transform": TARGET_TRANSFORM,
        "label_trim_schema": LABEL_TRIM_SCHEMA,
        "model_output_schema": MODEL_OUTPUT_SCHEMA,
        "linear_checkpoint_schema": LINEAR_CHECKPOINT_SCHEMA,
        "linear_model_arch_schema": LINEAR_MODEL_ARCH_SCHEMA,
        "prior": linear_model_summary(model),
        "primary_metric": {
            "name": primary_metric_label,
            "value": float(primary_metric_value),
            "guard_dir_bal_acc": float(val_fast.get("primary_dir_bal_acc", float("nan"))),
            "guard_passed": bool(val_fast.get("primary_metric_guard_passed", False)),
        },
        "val_fast_metrics": val_fast,
        "val_full_metrics": val_full,
        "test_metrics": test_metrics,
        "train_band_metrics": train_band_metrics,
    }
    metrics_path = linear_out_dir / "linear_stage1_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, allow_nan=True, indent=2)
    print(f"[linear_metrics] wrote {metrics_path}", flush=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "args": {
            "linear_checkpoint_schema": LINEAR_CHECKPOINT_SCHEMA,
            "linear_model_arch_schema": LINEAR_MODEL_ARCH_SCHEMA,
            "model_output_schema": MODEL_OUTPUT_SCHEMA,
            "stage": "stage1",
            "feature_dim_total": feature_dim_total,
            "LOOKBACK": LOOKBACK,
            "WINDOW_MS": WINDOW_MS,
            "HORIZONS_MS": HORIZONS_MS,
            "target_task": TARGET_TASK,
            "target_transform": TARGET_TRANSFORM,
            "label_trim_schema": LABEL_TRIM_SCHEMA,
            "low_abs_trim_fraction": float(LOW_ABS_TRIM_FRACTION),
            "high_abs_trim_fraction": float(HIGH_ABS_TRIM_FRACTION),
            "primary_metric": PRIMARY_METRIC,
            "primary_metric_horizon_ms": PRIMARY_METRIC_HORIZON_MS,
            "primary_dir_bal_acc_guard": PRIMARY_DIR_BAL_ACC_GUARD,
            "split_protocol": protocol,
            "train_week_keys": train_week_keys,
        },
        "prior": linear_model_summary(model),
        "stats": stats,
        "val_fast_metrics": val_fast,
        "val_full_metrics": val_full,
    }
    ckpt_path = linear_out_dir / "linear_stage1_prior.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[linear_ckpt] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"
    main()
