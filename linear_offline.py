#!/usr/bin/env python3
"""Linear offline entrypoint using CMSSL-compatible eval machinery."""

import csv
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

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
    is_metric_improved,
)
from CMSSL17_offline import (  # type: ignore
    require_supported_pipeline_splits,
    make_single_week_split_from_meta,
    validate_dataset_label_dim,
    validate_contract_meta,
    validate_loaded_label_array,
    compute_signed_raw_stats,
    build_signed_side_trim_masks_from_stats_np,
    _binary_auc_np,
    _safe_spearman_np,
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
    LinearSklearnTakerBundle,
    LinearSklearnTorchWrapper,
    save_linear_preprocess_bundle,
    load_linear_preprocess_bundle,
    save_linear_sklearn_bundle,
    load_linear_sklearn_bundle,
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


def _env_float_list(name: str, default: str) -> list[float]:
    raw = os.environ.get(name, default).strip()
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _env_str_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default).strip()
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _env_int_list_allow_empty(name: str, default: str) -> list[int]:
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
LINEAR_DECISION_STRIDE_ROWS = _env_int("BYBIT_LINEAR_DECISION_STRIDE_ROWS", 5)
LINEAR_DECISION_OFFSET_ROWS = _env_int("BYBIT_LINEAR_DECISION_OFFSET_ROWS", 0)
DECISION_ROW_POLICY = "linear_every_n_rows_v1"

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
LINEAR_PREPROCESS_AUDIT = _env_bool("BYBIT_LINEAR_PREPROCESS_AUDIT", 1)
LINEAR_PREPROCESS_AUDIT_TOP_K = _env_int("BYBIT_LINEAR_PREPROCESS_AUDIT_TOP_K", 50)
LINEAR_PREPROCESS_AUDIT_FULL_PER_FEATURE = _env_bool(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_FULL_PER_FEATURE", 0
)
LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES = _env_bool(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES", 0
)
LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE = _env_int(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE", 2_000_000
)

LINEAR_STAGE4_SCHEMA = "linear_target_models_stage4_v1"
LINEAR_STAGE4_PREPROCESS_NAME = os.environ.get("BYBIT_LINEAR_STAGE4_PREPROCESS_NAME", "default").strip()
LINEAR_STAGE4_TRAIN_SPLIT = os.environ.get("BYBIT_LINEAR_STAGE4_TRAIN_SPLIT", "train_full").strip().lower()
LINEAR_STAGE4_PREDICTOR = os.environ.get("BYBIT_LINEAR_STAGE4_PREDICTOR", "sgd_l2_huber").strip().lower()
LINEAR_STAGE4_ALPHA_GRID = os.environ.get("BYBIT_LINEAR_STAGE4_ALPHA_GRID", "1e-6,3e-6,1e-5,3e-5,1e-4").strip()
LINEAR_STAGE4_ALPHA_VALUES = _env_float_list("BYBIT_LINEAR_STAGE4_ALPHA_GRID", "1e-6,3e-6,1e-5,3e-5,1e-4")
LINEAR_STAGE4_PENALTY = os.environ.get("BYBIT_LINEAR_STAGE4_PENALTY", "l2").strip().lower()
LINEAR_STAGE4_L1_RATIO = float(os.environ.get("BYBIT_LINEAR_STAGE4_L1_RATIO", "0.15"))
LINEAR_STAGE4_EPOCHS = _env_int("BYBIT_LINEAR_STAGE4_EPOCHS", 3)
LINEAR_STAGE4_BATCH_ROWS = _env_int("BYBIT_LINEAR_STAGE4_BATCH_ROWS", 8192)
LINEAR_STAGE4_RANDOM_SEED = _env_int("BYBIT_LINEAR_STAGE4_RANDOM_SEED", 17)
LINEAR_STAGE4_DIRECTION_WEIGHTING = os.environ.get("BYBIT_LINEAR_STAGE4_DIRECTION_WEIGHTING", "tempered").strip().lower()
LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING = os.environ.get("BYBIT_LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING", "none").strip().lower()
LINEAR_STAGE4_RUN_TEST = _env_bool("BYBIT_LINEAR_STAGE4_RUN_TEST", 1)
LINEAR_STAGE4_MAG_FLOOR = float(os.environ.get("BYBIT_LINEAR_STAGE4_MAG_FLOOR", "1e-4"))
LINEAR_STAGE4_MAX_VAL_ROWS = _env_int("BYBIT_LINEAR_STAGE4_MAX_VAL_ROWS", 0)
LINEAR_STAGE4_MAX_TEST_ROWS = _env_int("BYBIT_LINEAR_STAGE4_MAX_TEST_ROWS", 0)
LINEAR_STAGE4_SAVE_VAL_PREDICTIONS = _env_bool("BYBIT_LINEAR_STAGE4_SAVE_VAL_PREDICTIONS", 0)
LINEAR_STAGE4_ALLOW_SAMPLE_TRIM_STATS = _env_bool("BYBIT_LINEAR_STAGE4_ALLOW_SAMPLE_TRIM_STATS", 0)

LINEAR_STAGE5_SCHEMA = "linear_comparison_stage5_v1"
LINEAR_STAGE5_EXTRACTORS = os.environ.get(
    "BYBIT_LINEAR_STAGE5_EXTRACTORS",
    "raw_linear,minirocket,multirocket,hydra,multirocket_hydra",
).strip()
LINEAR_STAGE5_EXTRACTOR_VALUES = _env_str_list(
    "BYBIT_LINEAR_STAGE5_EXTRACTORS",
    "raw_linear,minirocket,multirocket,hydra,multirocket_hydra",
)
LINEAR_STAGE5_PREPROCESS_NAME = os.environ.get("BYBIT_LINEAR_STAGE5_PREPROCESS_NAME", "default").strip()
LINEAR_STAGE5_PREDICTOR = os.environ.get("BYBIT_LINEAR_STAGE5_PREDICTOR", "sgd_l2_huber").strip().lower()
LINEAR_STAGE5_STRICT = _env_bool("BYBIT_LINEAR_STAGE5_STRICT", 0)
LINEAR_STAGE5_REEVALUATE = _env_bool("BYBIT_LINEAR_STAGE5_REEVALUATE", 1)
LINEAR_STAGE5_RUN_TEST = _env_bool("BYBIT_LINEAR_STAGE5_RUN_TEST", 1)
LINEAR_STAGE5_BATCH_ROWS = _env_int("BYBIT_LINEAR_STAGE5_BATCH_ROWS", 8192)
LINEAR_STAGE5_MAX_VAL_ROWS = _env_int("BYBIT_LINEAR_STAGE5_MAX_VAL_ROWS", 0)
LINEAR_STAGE5_MAX_TEST_ROWS = _env_int("BYBIT_LINEAR_STAGE5_MAX_TEST_ROWS", 0)
LINEAR_STAGE5_TOP_COEFS = _env_int("BYBIT_LINEAR_STAGE5_TOP_COEFS", 50)
LINEAR_STAGE5_SAVE_PREDICTIONS = _env_bool("BYBIT_LINEAR_STAGE5_SAVE_PREDICTIONS", 0)
LINEAR_STAGE5_PREDICTION_MAX_ROWS = _env_int("BYBIT_LINEAR_STAGE5_PREDICTION_MAX_ROWS", 0)
LINEAR_STAGE5_LABEL_SHIFTS = os.environ.get("BYBIT_LINEAR_STAGE5_LABEL_SHIFTS", "-5,-1,1,5").strip()
LINEAR_STAGE5_LABEL_SHIFT_VALUES = _env_int_list_allow_empty("BYBIT_LINEAR_STAGE5_LABEL_SHIFTS", "-5,-1,1,5")
LINEAR_STAGE5_LABEL_PERMUTATION = _env_bool("BYBIT_LINEAR_STAGE5_LABEL_PERMUTATION", 1)
LINEAR_STAGE5_PERMUTATION_SEED = _env_int("BYBIT_LINEAR_STAGE5_PERMUTATION_SEED", 17)
LINEAR_STAGE5_BASELINE_METRICS_JSON = os.environ.get("BYBIT_LINEAR_STAGE5_BASELINE_METRICS_JSON", "").strip()

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

if LINEAR_DECISION_STRIDE_ROWS <= 0:
    raise ValueError(
        f"BYBIT_LINEAR_DECISION_STRIDE_ROWS must be > 0, got {LINEAR_DECISION_STRIDE_ROWS}"
    )
if LINEAR_DECISION_OFFSET_ROWS < 0:
    raise ValueError(
        f"BYBIT_LINEAR_DECISION_OFFSET_ROWS must be >= 0, got {LINEAR_DECISION_OFFSET_ROWS}"
    )
if LINEAR_DECISION_OFFSET_ROWS >= LINEAR_DECISION_STRIDE_ROWS:
    raise ValueError(
        "BYBIT_LINEAR_DECISION_OFFSET_ROWS must be smaller than "
        f"BYBIT_LINEAR_DECISION_STRIDE_ROWS; got offset={LINEAR_DECISION_OFFSET_ROWS}, "
        f"stride={LINEAR_DECISION_STRIDE_ROWS}"
    )

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
    if LINEAR_PREPROCESS_AUDIT_TOP_K < 0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_AUDIT_TOP_K must be >= 0, got {LINEAR_PREPROCESS_AUDIT_TOP_K}")
    if LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE <= 0:
        raise ValueError(
            "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE must be > 0, "
            f"got {LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE}"
        )

if LINEAR_STAGE == "stage4":
    if LINEAR_STAGE4_TRAIN_SPLIT not in {"train_sample"}:
        raise ValueError("Stage 4 currently supports train_sample only")
    if LINEAR_STAGE4_PREDICTOR not in {"sgd_l2_huber"}:
        raise ValueError("Stage 4 currently supports BYBIT_LINEAR_STAGE4_PREDICTOR=sgd_l2_huber")
    if LINEAR_STAGE4_PENALTY not in {"l2", "elasticnet"}:
        raise ValueError("BYBIT_LINEAR_STAGE4_PENALTY must be one of: l2, elasticnet")
    if not LINEAR_STAGE4_ALPHA_VALUES or any(a <= 0 for a in LINEAR_STAGE4_ALPHA_VALUES):
        raise ValueError("BYBIT_LINEAR_STAGE4_ALPHA_GRID must contain positive alpha values")
    if LINEAR_STAGE4_EPOCHS <= 0:
        raise ValueError(f"BYBIT_LINEAR_STAGE4_EPOCHS must be > 0, got {LINEAR_STAGE4_EPOCHS}")
    if LINEAR_STAGE4_BATCH_ROWS <= 0:
        raise ValueError(f"BYBIT_LINEAR_STAGE4_BATCH_ROWS must be > 0, got {LINEAR_STAGE4_BATCH_ROWS}")
    if LINEAR_STAGE4_DIRECTION_WEIGHTING not in {"none", "balanced", "tempered"}:
        raise ValueError("BYBIT_LINEAR_STAGE4_DIRECTION_WEIGHTING must be one of: none, balanced, tempered")
    if LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING not in {"none"}:
        raise ValueError("Only none for magnitude weighting in first Stage 4 implementation")
    if LINEAR_STAGE4_MAG_FLOOR <= 0:
        raise ValueError(f"BYBIT_LINEAR_STAGE4_MAG_FLOOR must be > 0, got {LINEAR_STAGE4_MAG_FLOOR}")


if LINEAR_STAGE not in {"stage1", "stage2", "stage3", "stage4", "stage5"}:
    raise ValueError(
        "BYBIT_LINEAR_STAGE must be 'stage1', 'stage2', 'stage3', 'stage4', or 'stage5', "
        f"got {LINEAR_STAGE!r}"
    )

if LINEAR_STAGE == "stage5":
    if not LINEAR_STAGE5_EXTRACTOR_VALUES:
        raise ValueError("BYBIT_LINEAR_STAGE5_EXTRACTORS must not be empty")
    if LINEAR_STAGE5_BATCH_ROWS <= 0:
        raise ValueError(f"BYBIT_LINEAR_STAGE5_BATCH_ROWS must be > 0, got {LINEAR_STAGE5_BATCH_ROWS}")
    if LINEAR_STAGE5_TOP_COEFS < 0:
        raise ValueError(f"BYBIT_LINEAR_STAGE5_TOP_COEFS must be >= 0, got {LINEAR_STAGE5_TOP_COEFS}")
    if LINEAR_STAGE5_MAX_VAL_ROWS < 0 or LINEAR_STAGE5_MAX_TEST_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_STAGE5_MAX_VAL_ROWS and BYBIT_LINEAR_STAGE5_MAX_TEST_ROWS must be >= 0")
    if LINEAR_STAGE5_PREDICTION_MAX_ROWS < 0:
        raise ValueError(
            f"BYBIT_LINEAR_STAGE5_PREDICTION_MAX_ROWS must be >= 0, got {LINEAR_STAGE5_PREDICTION_MAX_ROWS}"
        )


def _resolve_device() -> torch.device:
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
        **_decision_metadata(),
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



def _decision_metadata() -> Dict[str, Any]:
    return {
        "decision_stride_rows": int(LINEAR_DECISION_STRIDE_ROWS),
        "decision_offset_rows": int(LINEAR_DECISION_OFFSET_ROWS),
        "decision_row_policy": DECISION_ROW_POLICY,
    }


def _print_decision_row_policy(stage: str) -> None:
    print(
        f"[linear-decision-rows] stage={stage} "
        f"stride_rows={LINEAR_DECISION_STRIDE_ROWS} "
        f"offset_rows={LINEAR_DECISION_OFFSET_ROWS} policy={DECISION_ROW_POLICY}",
        flush=True,
    )


def _validate_manifest_decision_policy(manifest: Dict[str, Any], *, context: str) -> None:
    stride = int(manifest.get("decision_stride_rows", -1))
    offset = int(manifest.get("decision_offset_rows", -1))
    policy = manifest.get("decision_row_policy")
    if stride != int(LINEAR_DECISION_STRIDE_ROWS) or offset != int(LINEAR_DECISION_OFFSET_ROWS):
        raise ValueError(
            f"{context} decision-row mismatch: manifest stride/offset={stride}/{offset}, "
            f"current={LINEAR_DECISION_STRIDE_ROWS}/{LINEAR_DECISION_OFFSET_ROWS}"
        )
    if policy != DECISION_ROW_POLICY:
        raise ValueError(f"{context} unexpected decision_row_policy={policy!r}")


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


def _decision_positions(n_rows: int) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("Cannot collect rows from empty dataset")
    if LINEAR_DECISION_OFFSET_ROWS >= n_rows:
        raise ValueError(
            f"Decision offset {LINEAR_DECISION_OFFSET_ROWS} is >= dataset rows {n_rows}"
        )
    return np.arange(
        LINEAR_DECISION_OFFSET_ROWS,
        n_rows,
        LINEAR_DECISION_STRIDE_ROWS,
        dtype=np.int64,
    )


def _dataset_positions(n_rows: int, max_rows: int) -> np.ndarray:
    base = _decision_positions(n_rows)
    if base.size <= 0:
        raise ValueError(
            f"No decision rows selected from n_rows={n_rows}, "
            f"offset={LINEAR_DECISION_OFFSET_ROWS}, stride={LINEAR_DECISION_STRIDE_ROWS}"
        )
    if max_rows <= 0 or max_rows >= base.size:
        return base
    idx = np.linspace(0, base.size - 1, int(max_rows), dtype=np.int64)
    return base[idx]


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


def collect_labels_from_dataset_positions(
    ds: Any,
    *,
    max_rows: int,
    split_name: str,
) -> np.ndarray:
    positions = _dataset_positions(len(ds), int(max_rows))
    y = np.asarray(ds.y[positions], dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"{split_name}: labels must be 2D, got {y.shape}")
    print(
        f"[linear-label-collect] split={split_name} rows={y.shape[0]} "
        f"stride={LINEAR_DECISION_STRIDE_ROWS} offset={LINEAR_DECISION_OFFSET_ROWS} "
        f"y_shape={list(y.shape)}",
        flush=True,
    )
    return y


def collect_train_labels_from_datasets(
    ds_train_list: list[Any],
    *,
    split_name: str = "train",
) -> np.ndarray:
    if not ds_train_list:
        raise ValueError("No train datasets supplied for label collection")
    parts = []
    for i, ds in enumerate(ds_train_list):
        parts.append(
            collect_labels_from_dataset_positions(
                ds,
                max_rows=0,
                split_name=f"{split_name}_week{i}",
            )
        )
    y = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    validate_loaded_label_array(y, f"linear {split_name} decision-stride labels")
    return y


class DatasetPositionsBatchSource:
    def __init__(
        self,
        ds: Any,
        device: torch.device,
        batch_rows: int,
        *,
        max_rows: int = 0,
        split_name: str = "",
        positions: Optional[np.ndarray] = None,
    ):
        self.ds = ds
        self.device = device
        self.batch_rows = max(1, int(batch_rows))
        if positions is None:
            positions = _dataset_positions(len(ds), int(max_rows))
        self.positions = np.asarray(positions, dtype=np.int64)
        if self.positions.ndim != 1 or self.positions.size <= 0:
            raise ValueError(f"{split_name}: positions must be non-empty 1D, got {self.positions.shape}")
        self.split_name = split_name
        self.n_rows = int(self.positions.shape[0])

    def __len__(self) -> int:
        return int(math.ceil(self.n_rows / self.batch_rows))

    def __iter__(self):
        for start in range(0, self.n_rows, self.batch_rows):
            pos = self.positions[start:start + self.batch_rows]
            X, y = collect_windows_for_positions(
                self.ds,
                pos,
                batch_rows=self.batch_rows,
                split_name=f"{self.split_name}_batch",
            )
            yield (
                torch.as_tensor(X, dtype=torch.float32, device=self.device),
                torch.as_tensor(y, dtype=torch.float32, device=self.device),
            )

    def make_evenly_spaced_subset(self, max_rows: int):
        if max_rows <= 0 or max_rows >= self.n_rows:
            return self
        idx = np.linspace(0, self.n_rows - 1, int(max_rows), dtype=np.int64)
        return DatasetPositionsBatchSource(
            self.ds,
            self.device,
            self.batch_rows,
            positions=self.positions[idx],
            split_name=f"{self.split_name}_subset",
        )


def collect_fit_windows_from_train(
    ds_train_list: list[Any],
    max_rows: int,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not ds_train_list:
        raise ValueError("No train datasets supplied for extractor fitting")
    if max_rows <= 0:
        max_rows = sum(int(_decision_positions(len(ds)).shape[0]) for ds in ds_train_list)
    per_week = int(np.ceil(max_rows / max(1, len(ds_train_list))))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for i, ds in enumerate(ds_train_list):
        rows_i = min(int(_decision_positions(len(ds)).shape[0]), per_week)
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


# Legacy persisted-shard helper. Not used by the default streaming pipeline.
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
        **_decision_metadata(),
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


# Legacy persisted-shard helper. Not used by the default streaming pipeline.
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
        **_decision_metadata(),
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


def resolve_stage3_audit_dir(stage3_dir: Path) -> Path:
    audit_dir = Path(stage3_dir) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir


def summarize_vector(x: np.ndarray, *, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return {
            f"{prefix}_finite_frac": 0.0,
            f"{prefix}_min": float("nan"),
            f"{prefix}_p01": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
        }
    xf = x[finite]
    return {
        f"{prefix}_finite_frac": float(finite.mean()),
        f"{prefix}_min": float(np.min(xf)),
        f"{prefix}_p01": float(np.percentile(xf, 1)),
        f"{prefix}_p05": float(np.percentile(xf, 5)),
        f"{prefix}_p50": float(np.percentile(xf, 50)),
        f"{prefix}_p95": float(np.percentile(xf, 95)),
        f"{prefix}_p99": float(np.percentile(xf, 99)),
        f"{prefix}_max": float(np.max(xf)),
        f"{prefix}_mean": float(np.mean(xf)),
    }


def topk_feature_records(values: np.ndarray, *, k: int, metric_name: str, descending: bool = True) -> list[dict]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if k <= 0 or vals.size == 0:
        return []
    finite_vals = np.where(np.isfinite(vals), vals, -np.inf if descending else np.inf)
    order = np.argsort(-finite_vals if descending else finite_vals)[: min(int(k), vals.size)]
    return [
        {"feature_index": int(i), "metric": metric_name, "value": float(vals[i])}
        for i in order
    ]


def _safe_percentile(x: np.ndarray, q: float) -> float:
    vals = np.asarray(x, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def _sample_array(parts: list[np.ndarray], *, max_values: int) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.float32)
    vals = np.concatenate(parts).astype(np.float32, copy=False)
    if vals.size > max_values:
        vals = vals[:max_values]
    return vals


def _append_abs_sample(parts: list[np.ndarray], values: np.ndarray, *, max_values: int) -> None:
    vals = np.abs(np.asarray(values).reshape(-1))
    if vals.size == 0:
        return
    current_parts = max(1, len(parts) + 1)
    stride = max(1, vals.size // max(1, int(max_values) // current_parts))
    parts.append(vals[::stride].astype(np.float32, copy=False))


def new_preprocess_audit_accumulator(original_dim: int, kept_dim: int) -> Dict[str, Any]:
    return {
        "rows": 0,
        "original_dim": int(original_dim),
        "kept_dim": int(kept_dim),
        "raw_nonfinite_count": 0,
        "raw_total_count": 0,
        "below_lower_counts": np.zeros(original_dim, dtype=np.int64),
        "above_upper_counts": np.zeros(original_dim, dtype=np.int64),
        "raw_abs_sample": [],
        "cap_abs_sample": [],
        "std_abs_sample": [],
        "out_abs_sample": [],
        "std_sum": np.zeros(original_dim, dtype=np.float64),
        "std_sumsq": np.zeros(original_dim, dtype=np.float64),
        "std_count": 0,
        "out_sum": np.zeros(kept_dim, dtype=np.float64),
        "out_sumsq": np.zeros(kept_dim, dtype=np.float64),
        "out_count": 0,
        "std_abs_gt_5": 0,
        "std_abs_gt_10": 0,
        "std_abs_gt_20": 0,
        "out_abs_gt_5": 0,
        "out_abs_gt_10": 0,
        "out_abs_gt_20": 0,
        "post_clip_count": 0,
        "post_clip_total": 0,
    }


def update_preprocess_audit(
    acc: Dict[str, Any],
    *,
    Z_raw: np.ndarray,
    bundle: LinearPreprocessBundle,
    max_sample_values: int,
) -> None:
    Z_raw = np.asarray(Z_raw, dtype=np.float32)
    if Z_raw.ndim != 2 or Z_raw.shape[1] != int(bundle.original_dim):
        raise ValueError(f"Audit raw Z shape {Z_raw.shape} does not match original_dim={bundle.original_dim}")
    N = int(Z_raw.shape[0])
    acc["rows"] += N
    acc["raw_total_count"] += int(Z_raw.size)
    finite = np.isfinite(Z_raw)
    nonfinite = int((~finite).sum())
    acc["raw_nonfinite_count"] += nonfinite
    policy = str(bundle.config.get("nonfinite_policy", "raise"))
    if policy == "raise":
        if nonfinite:
            raise ValueError("nonfinite in audit raw Z")
        Z0 = Z_raw
    elif policy == "warn_zero":
        Z0 = np.where(finite, Z_raw, 0.0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown preprocessing nonfinite policy: {policy!r}")

    lower = np.asarray(bundle.lower, dtype=np.float32)
    upper = np.asarray(bundle.upper, dtype=np.float32)
    below = Z0 < lower
    above = Z0 > upper
    acc["below_lower_counts"] += below.sum(axis=0)
    acc["above_upper_counts"] += above.sum(axis=0)
    Z_cap = np.minimum(np.maximum(Z0, lower), upper)

    std_eps = float(bundle.config.get("std_eps", 1e-6))
    Z_std = (Z_cap - bundle.mean) / np.maximum(bundle.std, std_eps)
    acc["std_sum"] += Z_std.sum(axis=0, dtype=np.float64)
    acc["std_sumsq"] += np.square(Z_std, dtype=np.float64).sum(axis=0)
    acc["std_count"] += N

    abs_std = np.abs(Z_std)
    acc["std_abs_gt_5"] += int((abs_std > 5.0).sum())
    acc["std_abs_gt_10"] += int((abs_std > 10.0).sum())
    acc["std_abs_gt_20"] += int((abs_std > 20.0).sum())

    Z_keep = Z_std[:, bundle.keep_mask]
    post_clip_abs = float(bundle.config.get("post_clip_abs", 0.0))
    if post_clip_abs > 0:
        Z_out = np.clip(Z_keep, -post_clip_abs, post_clip_abs)
        acc["post_clip_count"] += int((Z_keep != Z_out).sum())
    else:
        Z_out = Z_keep
    acc["post_clip_total"] += int(Z_keep.size)

    acc["out_sum"] += Z_out.sum(axis=0, dtype=np.float64)
    acc["out_sumsq"] += np.square(Z_out, dtype=np.float64).sum(axis=0)
    acc["out_count"] += N
    abs_out = np.abs(Z_out)
    acc["out_abs_gt_5"] += int((abs_out > 5.0).sum())
    acc["out_abs_gt_10"] += int((abs_out > 10.0).sum())
    acc["out_abs_gt_20"] += int((abs_out > 20.0).sum())

    _append_abs_sample(acc["raw_abs_sample"], Z0, max_values=max_sample_values)
    _append_abs_sample(acc["cap_abs_sample"], Z_cap, max_values=max_sample_values)
    _append_abs_sample(acc["std_abs_sample"], Z_std, max_values=max_sample_values)
    _append_abs_sample(acc["out_abs_sample"], Z_out, max_values=max_sample_values)


def finalize_preprocess_audit(
    acc: Dict[str, Any],
    *,
    bundle: LinearPreprocessBundle,
    split_name: str,
    top_k: int,
    max_sample_values: int,
) -> Dict[str, Any]:
    rows = int(acc["rows"])
    original_dim = int(acc["original_dim"])
    kept_dim = int(acc["kept_dim"])
    below_frac = acc["below_lower_counts"].astype(np.float64) / max(1, rows)
    above_frac = acc["above_upper_counts"].astype(np.float64) / max(1, rows)
    clip_frac = below_frac + above_frac

    std_count = max(1, int(acc["std_count"]))
    std_mean = acc["std_sum"] / std_count
    std_var = np.maximum(0.0, acc["std_sumsq"] / std_count - std_mean * std_mean)
    std_std = np.sqrt(std_var)

    out_count = max(1, int(acc["out_count"]))
    out_mean = acc["out_sum"] / out_count if kept_dim else np.zeros(0, dtype=np.float64)
    out_var = np.maximum(0.0, acc["out_sumsq"] / out_count - out_mean * out_mean) if kept_dim else np.zeros(0)
    out_std = np.sqrt(out_var)

    raw_abs_sample = _sample_array(acc["raw_abs_sample"], max_values=max_sample_values)
    cap_abs_sample = _sample_array(acc["cap_abs_sample"], max_values=max_sample_values)
    std_abs_sample = _sample_array(acc["std_abs_sample"], max_values=max_sample_values)
    out_abs_sample = _sample_array(acc["out_abs_sample"], max_values=max_sample_values)
    total_original = max(1, rows * original_dim)
    total_kept = max(1, rows * kept_dim)
    raw_nonfinite_frac = float(acc["raw_nonfinite_count"] / max(1, acc["raw_total_count"]))
    removed = ~np.asarray(bundle.keep_mask, dtype=bool)
    removed_idx = np.where(removed)[0]
    removed_sorted = removed_idx[np.argsort(np.asarray(bundle.std, dtype=np.float64)[removed_idx])] if removed_idx.size else []

    summary: Dict[str, Any] = {
        "split": split_name,
        "rows": rows,
        "original_dim": original_dim,
        "kept_dim": kept_dim,
        "winsor_below_frac_p50": _safe_percentile(below_frac, 50),
        "winsor_below_frac_p95": _safe_percentile(below_frac, 95),
        "winsor_below_frac_max": float(np.max(below_frac)) if below_frac.size else float("nan"),
        "winsor_above_frac_p50": _safe_percentile(above_frac, 50),
        "winsor_above_frac_p95": _safe_percentile(above_frac, 95),
        "winsor_above_frac_max": float(np.max(above_frac)) if above_frac.size else float("nan"),
        "winsor_total_clip_frac_mean": float(np.mean(clip_frac)) if clip_frac.size else float("nan"),
        "winsor_total_clip_frac_p50": _safe_percentile(clip_frac, 50),
        "winsor_total_clip_frac_p95": _safe_percentile(clip_frac, 95),
        "winsor_total_clip_frac_p99": _safe_percentile(clip_frac, 99),
        "winsor_total_clip_frac_max": float(np.max(clip_frac)) if clip_frac.size else float("nan"),
        "winsor_features_gt_0p1pct": int((clip_frac > 0.001).sum()),
        "winsor_features_gt_1pct": int((clip_frac > 0.01).sum()),
        "winsor_features_gt_5pct": int((clip_frac > 0.05).sum()),
        "std_mean_abs_p50": _safe_percentile(np.abs(std_mean), 50),
        "std_mean_abs_p95": _safe_percentile(np.abs(std_mean), 95),
        "std_mean_abs_max": float(np.max(np.abs(std_mean))) if std_mean.size else float("nan"),
        "std_std_p05": _safe_percentile(std_std, 5),
        "std_std_p50": _safe_percentile(std_std, 50),
        "std_std_p95": _safe_percentile(std_std, 95),
        "std_std_max": float(np.max(std_std)) if std_std.size else float("nan"),
        "std_abs_gt_5_frac": float(acc["std_abs_gt_5"] / total_original),
        "std_abs_gt_10_frac": float(acc["std_abs_gt_10"] / total_original),
        "std_abs_gt_20_frac": float(acc["std_abs_gt_20"] / total_original),
        "std_abs_p50": _safe_percentile(std_abs_sample, 50),
        "std_abs_p95": _safe_percentile(std_abs_sample, 95),
        "std_abs_p99": _safe_percentile(std_abs_sample, 99),
        "std_abs_p999": _safe_percentile(std_abs_sample, 99.9),
        "std_abs_max_sample": float(np.max(std_abs_sample)) if std_abs_sample.size else float("nan"),
        "variance_original_dim": original_dim,
        "variance_kept_dim": kept_dim,
        "variance_removed_dim": int(original_dim - kept_dim),
        "variance_removed_frac": float((original_dim - kept_dim) / max(1, original_dim)),
        "variance_train_std_min": float(np.nanmin(bundle.std)) if original_dim else float("nan"),
        "variance_train_std_p01": _safe_percentile(bundle.std, 1),
        "variance_train_std_p05": _safe_percentile(bundle.std, 5),
        "variance_train_std_p50": _safe_percentile(bundle.std, 50),
        "variance_train_std_p95": _safe_percentile(bundle.std, 95),
        "variance_train_std_max": float(np.nanmax(bundle.std)) if original_dim else float("nan"),
        "variance_n_std_lt_1e-8": int((np.asarray(bundle.std) < 1e-8).sum()),
        "variance_n_std_lt_1e-6": int((np.asarray(bundle.std) < 1e-6).sum()),
        "variance_n_std_lt_1e-4": int((np.asarray(bundle.std) < 1e-4).sum()),
        "out_mean_abs_p50": _safe_percentile(np.abs(out_mean), 50),
        "out_mean_abs_p95": _safe_percentile(np.abs(out_mean), 95),
        "out_mean_abs_max": float(np.max(np.abs(out_mean))) if out_mean.size else float("nan"),
        "out_std_p05": _safe_percentile(out_std, 5),
        "out_std_p50": _safe_percentile(out_std, 50),
        "out_std_p95": _safe_percentile(out_std, 95),
        "out_std_max": float(np.max(out_std)) if out_std.size else float("nan"),
        "out_abs_p50": _safe_percentile(out_abs_sample, 50),
        "out_abs_p95": _safe_percentile(out_abs_sample, 95),
        "out_abs_p99": _safe_percentile(out_abs_sample, 99),
        "out_abs_p999": _safe_percentile(out_abs_sample, 99.9),
        "out_abs_gt_5_frac": float(acc["out_abs_gt_5"] / total_kept),
        "out_abs_gt_10_frac": float(acc["out_abs_gt_10"] / total_kept),
        "out_abs_gt_20_frac": float(acc["out_abs_gt_20"] / total_kept),
        "post_clip_frac": float(acc["post_clip_count"] / max(1, acc["post_clip_total"])),
        "raw_nonfinite_frac": raw_nonfinite_frac,
        "top_clipped_features": topk_feature_records(clip_frac, k=top_k, metric_name="winsor_total_clip_frac"),
        "top_abs_mean_shift_features": topk_feature_records(np.abs(std_mean), k=top_k, metric_name="std_abs_mean_shift"),
        "top_std_ratio_features": topk_feature_records(np.abs(std_std - 1.0), k=top_k, metric_name="std_abs_std_minus_one"),
        "top_removed_low_variance_features": [
            {"feature_index": int(i), "metric": "removed_low_variance", "value": float(bundle.std[int(i)]), "train_std": float(bundle.std[int(i)])}
            for i in list(removed_sorted)[: int(top_k)]
        ],
        "_per_feature": {
            "clip_frac": clip_frac.astype(np.float32),
            "std_mean": std_mean.astype(np.float32),
            "std_std": std_std.astype(np.float32),
        },
        "_value_samples": {
            "raw_abs_sample": raw_abs_sample,
            "cap_abs_sample": cap_abs_sample,
            "std_abs_sample": std_abs_sample,
            "out_abs_sample": out_abs_sample,
        },
    }
    warnings = []
    if summary["winsor_total_clip_frac_p95"] > 0.01:
        warnings.append("winsor_clip_p95_gt_1pct")
    if summary["winsor_total_clip_frac_max"] > 0.05:
        warnings.append("winsor_clip_max_gt_5pct")
    if summary["std_abs_gt_20_frac"] > 1e-4:
        warnings.append("std_abs_gt_20_frac_high")
    if summary["out_abs_gt_10_frac"] > 1e-3:
        warnings.append("out_abs_gt_10_frac_high")
    if summary["out_abs_gt_20_frac"] > 1e-4:
        warnings.append("out_abs_gt_20_frac_high")
    if split_name in {"train_sample", "train"} and summary["out_mean_abs_p95"] > 0.05:
        warnings.append("train_mean_not_centered")
    if split_name in {"train_sample", "train"} and (summary["out_std_p50"] < 0.8 or summary["out_std_p50"] > 1.2):
        warnings.append("train_std_not_unit_scaled")
    if summary["variance_removed_frac"] > 0.10:
        warnings.append("variance_removed_gt_10pct")
    if raw_nonfinite_frac > 0:
        warnings.append("raw_nonfinite_present")
    summary["warnings"] = warnings
    return summary


def compact_preprocess_audit_summary(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None
    compact = {}
    for k, v in summary.items():
        if k.startswith("_") or isinstance(v, (list, dict)):
            continue
        compact[k] = v
    compact["warnings"] = list(summary.get("warnings", []))
    return compact


def jsonable_preprocess_audit_summary(summary: Dict[str, Any], *, sample_values: bool = False) -> Dict[str, Any]:
    out = {k: v for k, v in summary.items() if not k.startswith("_")}
    if sample_values:
        samples = summary.get("_value_samples", {}) or {}
        out["value_samples"] = {
            key: np.asarray(value).astype(float).tolist()
            for key, value in samples.items()
        }
    return out


def write_preprocess_audit_csv(path: Path, split_summaries: Dict[str, Optional[Dict[str, Any]]]) -> None:
    columns = [
        "split", "rows", "original_dim", "kept_dim", "winsor_total_clip_frac_p50",
        "winsor_total_clip_frac_p95", "winsor_total_clip_frac_max", "std_abs_p99",
        "std_abs_p999", "std_abs_gt_10_frac", "out_abs_p99", "out_abs_p999",
        "out_abs_gt_10_frac", "variance_removed_frac", "raw_nonfinite_frac", "post_clip_frac", "warnings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for split, summary in split_summaries.items():
            if summary is None:
                continue
            row = {c: summary.get(c) for c in columns}
            row["split"] = split
            row["warnings"] = ";".join(summary.get("warnings", []))
            writer.writerow(row)


def write_preprocess_top_features_csv(path: Path, split_summaries: Dict[str, Optional[Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "category", "rank", "feature_index", "value", "extra"])
        writer.writeheader()
        mapping = {
            "top_clipped_features": "winsor_total_clip_frac",
            "top_abs_mean_shift_features": "std_abs_mean_shift",
            "top_std_ratio_features": "std_abs_std_minus_one",
            "top_removed_low_variance_features": "removed_low_variance",
        }
        for split, summary in split_summaries.items():
            if summary is None:
                continue
            for key, category in mapping.items():
                for rank, rec in enumerate(summary.get(key, []) or [], start=1):
                    extra = ""
                    if "train_std" in rec:
                        extra = json.dumps({"train_std": rec.get("train_std")})
                    writer.writerow({
                        "split": split,
                        "category": category,
                        "rank": rank,
                        "feature_index": rec.get("feature_index"),
                        "value": rec.get("value"),
                        "extra": extra,
                    })


def write_preprocess_per_feature_npz(path: Path, split_summaries: Dict[str, Optional[Dict[str, Any]]], bundle: LinearPreprocessBundle) -> None:
    arrays: Dict[str, np.ndarray] = {
        "bundle_std": np.asarray(bundle.std, dtype=np.float32),
        "keep_mask": np.asarray(bundle.keep_mask, dtype=bool),
    }
    for split, summary in split_summaries.items():
        if summary is None:
            continue
        per = summary.get("_per_feature", {}) or {}
        prefix = "train" if split in {"train_sample", "train"} else split
        if "clip_frac" in per:
            arrays[f"{prefix}_clip_frac"] = per["clip_frac"]
        if "std_mean" in per:
            arrays[f"{prefix}_std_mean"] = per["std_mean"]
        if "std_std" in per:
            arrays[f"{prefix}_std_std"] = per["std_std"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# Legacy persisted-shard helper. Not used by the default streaming pipeline.
def preprocess_manifest_to_npz_shards(
    *,
    bundle: LinearPreprocessBundle,
    source_manifest: Dict[str, Any],
    split_name: str,
    out_dir: Path,
    shard_rows: int,
    max_z_chunk_mb: int,
    bundle_path: Path,
    audit_enabled: bool = False,
    audit_top_k: int = 0,
    audit_max_sample_values: int = 2_000_000,
    audit_path: Optional[Path] = None,
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
    audit_acc = new_preprocess_audit_accumulator(bundle.original_dim, bundle.kept_dim) if audit_enabled else None
    for source_shard in source_manifest.get("shards", []):
        with np.load(source_shard["path"]) as arr:
            Z_src = np.asarray(arr["Z"], dtype=np.float32)
            y = np.asarray(arr["y"], dtype=np.float32)
            positions = np.asarray(arr["positions"], dtype=np.int64)
        if Z_src.shape[0] != y.shape[0] or Z_src.shape[0] != positions.shape[0]:
            raise ValueError(f"Source shard row mismatch: {source_shard['path']}")
        for start in range(0, Z_src.shape[0], chunk_rows):
            end = min(Z_src.shape[0], start + chunk_rows)
            Z_chunk = Z_src[start:end]
            if audit_acc is not None:
                update_preprocess_audit(
                    audit_acc,
                    Z_raw=Z_chunk,
                    bundle=bundle,
                    max_sample_values=int(audit_max_sample_values),
                )
            t0 = time.time()
            Z_out = bundle.transform(Z_chunk)
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
    audit_summary = None
    if audit_acc is not None:
        audit_summary = finalize_preprocess_audit(
            audit_acc,
            bundle=bundle,
            split_name=split_name,
            top_k=int(audit_top_k),
            max_sample_values=int(audit_max_sample_values),
        )
    manifest_path = out_dir / f"{split_name}_preprocessed_manifest.json"
    manifest = {
        "split": split_name,
        "stage": "stage3",
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "source_manifest_path": source_manifest.get("manifest_path"),
        "preprocess_bundle_path": str(bundle_path),
        **_decision_metadata(),
        "n_rows": int(total_rows),
        "original_dim": int(bundle.original_dim),
        "kept_dim": int(bundle.kept_dim),
        "processed_chunks": int(processed_chunks),
        "n_saved_shards": len(shards),
        "n_shards": len(shards),
        "seconds_total": float(seconds_total),
        "max_z_chunk_mb": int(max_z_chunk_mb),
        "summary": summary,
        "audit_summary": compact_preprocess_audit_summary(audit_summary),
        "audit_path": None if audit_path is None else str(audit_path),
        "shards": shards,
        "manifest_path": str(manifest_path),
    }
    if audit_summary is not None and audit_path is not None:
        audit_payload = {
            "stage": "stage3",
            "schema": "linear_preprocess_audit_v1",
            "extractor": bundle.config.get("extractor"),
            "split": split_name,
            "preprocess_config": dict(bundle.config),
            "bundle_path": str(bundle_path),
            "summary": jsonable_preprocess_audit_summary(
                audit_summary, sample_values=bool(LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES)
            ),
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit_payload, allow_nan=True, indent=2), encoding="utf-8")
        print(f"[linear-preprocess-audit] wrote {audit_path}", flush=True)
    manifest["_audit_full_summary"] = audit_summary
    with manifest_path.open("w", encoding="utf-8") as f:
        json_manifest = {k: v for k, v in manifest.items() if k != "_audit_full_summary"}
        json.dump(json_manifest, f, allow_nan=True, indent=2)
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
    audit_dir = resolve_stage3_audit_dir(stage3_dir) if LINEAR_PREPROCESS_AUDIT else None
    stage2_payload = load_stage2_payload(linear_out_dir, extractor_name)
    train_manifest = load_manifest_from_payload(stage2_payload, "train_sample")
    val_manifest = load_manifest_from_payload(stage2_payload, "val")
    test_manifest = load_manifest_from_payload(stage2_payload, "test")
    if train_manifest is None:
        raise ValueError("Stage 3 requires a Stage 2 train_sample manifest")
    if val_manifest is None:
        raise ValueError("Stage 3 requires a Stage 2 val manifest")
    _print_decision_row_policy("stage3")
    _validate_manifest_decision_policy(train_manifest, context="stage3 train_sample")
    _validate_manifest_decision_policy(val_manifest, context="stage3 val")
    if test_manifest is not None:
        _validate_manifest_decision_policy(test_manifest, context="stage3 test")
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
        "audit_enabled": bool(LINEAR_PREPROCESS_AUDIT),
        "audit_top_k": int(LINEAR_PREPROCESS_AUDIT_TOP_K),
        "audit_full_per_feature": bool(LINEAR_PREPROCESS_AUDIT_FULL_PER_FEATURE),
        "audit_sample_values": bool(LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES),
        "audit_max_value_sample": int(LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE),
        **_decision_metadata(),
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
        audit_enabled=bool(LINEAR_PREPROCESS_AUDIT),
        audit_top_k=LINEAR_PREPROCESS_AUDIT_TOP_K,
        audit_max_sample_values=LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE,
        audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_train_sample.json",
    )
    val_pre_manifest = preprocess_manifest_to_npz_shards(
        bundle=bundle,
        source_manifest=val_manifest,
        split_name="val",
        out_dir=stage3_dir,
        shard_rows=LINEAR_PREPROCESS_SHARD_ROWS,
        max_z_chunk_mb=LINEAR_PREPROCESS_MAX_Z_CHUNK_MB,
        bundle_path=bundle_path,
        audit_enabled=bool(LINEAR_PREPROCESS_AUDIT),
        audit_top_k=LINEAR_PREPROCESS_AUDIT_TOP_K,
        audit_max_sample_values=LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE,
        audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_val.json",
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
            audit_enabled=bool(LINEAR_PREPROCESS_AUDIT),
            audit_top_k=LINEAR_PREPROCESS_AUDIT_TOP_K,
            audit_max_sample_values=LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE,
            audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_test.json",
        )

    split_audit_summaries = {
        "train_sample": train_pre_manifest.pop("_audit_full_summary", None),
        "val": val_pre_manifest.pop("_audit_full_summary", None),
        "test": None if test_pre_manifest is None else test_pre_manifest.pop("_audit_full_summary", None),
    }
    combined_audit_summary = None
    audit_summary_path = None
    audit_csv_path = None
    audit_top_features_csv_path = None
    if LINEAR_PREPROCESS_AUDIT and audit_dir is not None:
        jsonable_splits = {
            split: None if summary is None else jsonable_preprocess_audit_summary(
                summary, sample_values=bool(LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES)
            )
            for split, summary in split_audit_summaries.items()
        }
        recommendation_hints = [
            "Inspect preprocess_audit_summary.csv for compact split-level winsorization, standardization, variance-filter, nonfinite, and post-clip diagnostics.",
            "Inspect preprocess_audit_top_features.csv for top problematic synthetic feature indices without a full per-feature dump.",
        ]
        combined_audit_summary = {
            "stage": "stage3",
            "schema": "linear_preprocess_audit_v1",
            "extractor": extractor_name,
            "preprocess_name": "default",
            "bundle_path": str(bundle_path),
            "splits": jsonable_splits,
            "recommendation_hints": recommendation_hints,
        }
        audit_summary_path = audit_dir / "preprocess_audit_summary.json"
        audit_csv_path = audit_dir / "preprocess_audit_summary.csv"
        audit_top_features_csv_path = audit_dir / "preprocess_audit_top_features.csv"
        audit_summary_path.write_text(json.dumps(combined_audit_summary, allow_nan=True, indent=2), encoding="utf-8")
        write_preprocess_audit_csv(audit_csv_path, split_audit_summaries)
        write_preprocess_top_features_csv(audit_top_features_csv_path, split_audit_summaries)
        if LINEAR_PREPROCESS_AUDIT_FULL_PER_FEATURE:
            write_preprocess_per_feature_npz(audit_dir / "preprocess_audit_per_feature.npz", split_audit_summaries, bundle)
        print(f"[linear-preprocess-audit] wrote {audit_summary_path}, {audit_csv_path}, {audit_top_features_csv_path}", flush=True)

    payload = {
        "stage": "stage3",
        "status": "ok",
        "schema": LINEAR_PREPROCESS_SCHEMA,
        **_decision_metadata(),
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
        "audit_enabled": bool(LINEAR_PREPROCESS_AUDIT),
        "audit_dir": None if audit_dir is None else str(audit_dir),
        "audit_summary_path": None if audit_summary_path is None else str(audit_summary_path),
        "audit_csv_path": None if audit_csv_path is None else str(audit_csv_path),
        "audit_top_features_csv_path": None if audit_top_features_csv_path is None else str(audit_top_features_csv_path),
        "audit_summary": combined_audit_summary,
        "saved_files": {
            "preprocess_bundle": str(bundle_path),
            "audit_summary": None if audit_summary_path is None else str(audit_summary_path),
            "audit_csv": None if audit_csv_path is None else str(audit_csv_path),
            "audit_top_features_csv": None if audit_top_features_csv_path is None else str(audit_top_features_csv_path),
        },
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



def resolve_stage3_dir(linear_out_dir: Path, extractor_name: str, preprocess_name: str) -> Path:
    return Path(linear_out_dir) / "stage3_preprocess" / str(extractor_name) / str(preprocess_name)


def load_stage3_payload(linear_out_dir: Path, extractor_name: str, preprocess_name: str) -> Dict[str, Any]:
    stage3_dir = resolve_stage3_dir(linear_out_dir, extractor_name, preprocess_name)
    path = stage3_dir / "linear_stage3_preprocess_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 3 payload not found for extractor={extractor_name!r} preprocess={preprocess_name!r}: {path}")
    payload = load_json(path)
    if payload.get("stage") != "stage3":
        raise ValueError(f"Expected stage3 payload at {path}, got stage={payload.get('stage')!r}")
    if "manifests" not in payload:
        raise ValueError(f"Stage 3 payload missing manifests: {path}")
    payload["payload_path"] = str(path)
    print(f"[linear-stage4] loaded stage3 payload {path}", flush=True)
    return payload


# Legacy persisted-shard helper. Not used by the default streaming pipeline.
def load_stage3_manifest(payload: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    manifest = payload.get("manifests", {}).get(split)
    if manifest is None:
        return None
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid Stage 3 manifest payload for split={split}: {manifest!r}")
    path = Path(str(manifest.get("manifest_path", "")))
    if path.exists():
        loaded = load_json(path)
        if "manifest_path" not in loaded:
            loaded["manifest_path"] = str(path)
        return loaded
    return manifest


def load_linear_trim_stats(linear_out_dir: Path) -> Dict[str, np.ndarray]:
    path = Path(linear_out_dir) / "linear_signed_side_trim_stats_cache.npz"
    cached = load_stats_cache(path)
    if not cached:
        raise FileNotFoundError(
            f"Missing linear trim stats cache: {path}. Run stage1 or stage2 first."
        )

    stats, cache_meta = cached

    stride = int(cache_meta.get("decision_stride_rows", -1))
    offset = int(cache_meta.get("decision_offset_rows", -1))
    policy = cache_meta.get("decision_row_policy")

    if stride != int(LINEAR_DECISION_STRIDE_ROWS) or offset != int(LINEAR_DECISION_OFFSET_ROWS):
        raise ValueError(
            f"Trim stats cache decision-row mismatch: cache stride/offset={stride}/{offset}, "
            f"current={LINEAR_DECISION_STRIDE_ROWS}/{LINEAR_DECISION_OFFSET_ROWS}. "
            f"Delete/rebuild {path} by rerunning Stage 1 or Stage 2."
        )

    if policy != DECISION_ROW_POLICY:
        raise ValueError(
            f"Trim stats cache decision_row_policy mismatch: "
            f"cache={policy!r}, current={DECISION_ROW_POLICY!r}. "
            f"Delete/rebuild {path} by rerunning Stage 1 or Stage 2."
        )

    print(f"[linear-stage4] loaded trim stats {path}", flush=True)
    return stats


def compute_sample_trim_stats_from_manifest(train_manifest: Dict[str, Any]) -> Dict[str, np.ndarray]:
    y_parts = []
    for _Z, y_batch in iter_manifest_numpy_batches(train_manifest, batch_rows=LINEAR_STAGE4_BATCH_ROWS):
        y_parts.append(y_batch.astype(np.float32, copy=False))
    if not y_parts:
        raise ValueError("Cannot compute sample trim stats from empty Stage 3 train manifest")
    print("[linear-stage4-warn] computing trim stats from train_sample because BYBIT_LINEAR_STAGE4_ALLOW_SAMPLE_TRIM_STATS=1", flush=True)
    return compute_signed_raw_stats(np.concatenate(y_parts, axis=0))


def manifest_n_rows(manifest: Dict[str, Any]) -> int:
    if "n_rows" in manifest:
        return int(manifest["n_rows"])
    return int(manifest["summary"]["shape"][0])


def manifest_dim(manifest: Dict[str, Any]) -> int:
    if "kept_dim" in manifest:
        return int(manifest["kept_dim"])
    summary = manifest.get("summary", {})
    if "shape" in summary and len(summary["shape"]) >= 2:
        return int(summary["shape"][1])
    raise ValueError("Manifest missing kept_dim / summary.shape[1]")


# Legacy persisted-shard helper. Not used by the default streaming pipeline.
def iter_manifest_numpy_batches(
    manifest: Dict[str, Any],
    *,
    batch_rows: int,
    max_rows: int = 0,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if int(batch_rows) <= 0:
        raise ValueError(f"batch_rows must be > 0, got {batch_rows}")
    remaining = None if int(max_rows) <= 0 else int(max_rows)
    expected_dim = manifest_dim(manifest)
    for shard in manifest.get("shards", []):
        if remaining is not None and remaining <= 0:
            break
        with np.load(shard["path"]) as arr:
            Z = np.asarray(arr["Z"], dtype=np.float32)
            y = np.asarray(arr["y"], dtype=np.float32)
        if Z.ndim != 2 or Z.shape[1] != expected_dim:
            raise ValueError(f"Shard {shard['path']} has invalid Z shape {Z.shape}; expected width {expected_dim}")
        if y.ndim != 2 or y.shape[1] != len(HORIZONS_MS) or y.shape[0] != Z.shape[0]:
            raise ValueError(f"Shard {shard['path']} has invalid y shape {y.shape} for Z shape {Z.shape}")
        n = int(Z.shape[0]) if remaining is None else min(int(Z.shape[0]), int(remaining))
        if n <= 0:
            continue
        Z = Z[:n]
        y = y[:n]
        order = np.arange(n, dtype=np.int64)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(order)
        for start in range(0, n, int(batch_rows)):
            idx = order[start:min(n, start + int(batch_rows))]
            yield Z[idx].astype(np.float32, copy=False), y[idx].astype(np.float32, copy=False)
        if remaining is not None:
            remaining -= n


# Legacy persisted-shard helper. Not used by the default streaming pipeline.
class PreprocessedShardBatchSource:
    def __init__(self, manifest: Dict[str, Any], device: torch.device, batch_rows: int, max_rows: int = 0):
        self.manifest = manifest
        self.target_device = device
        self.device = torch.device("cpu")
        self.batch_size = int(batch_rows)
        self.n_rows = manifest_n_rows(manifest) if int(max_rows) <= 0 else min(manifest_n_rows(manifest), int(max_rows))
        self.effective_rows_nominal = int(self.n_rows)
        self.num_horizons = len(HORIZONS_MS)
        self.feature_shape = (int(self.n_rows), manifest_dim(manifest))
        self.is_shared_feature_view = False
        self.pin_memory = bool(device.type == "cuda")

    def __len__(self) -> int:
        return (int(self.n_rows) + self.batch_size - 1) // self.batch_size

    def iter_epoch(self, epoch: int):
        del epoch
        for Z, y in iter_manifest_numpy_batches(self.manifest, batch_rows=self.batch_size, max_rows=self.n_rows, shuffle=False):
            x_cpu = torch.as_tensor(Z, dtype=torch.float32)
            y_cpu = torch.as_tensor(y, dtype=torch.float32)
            if self.pin_memory:
                x_cpu = x_cpu.pin_memory()
                y_cpu = y_cpu.pin_memory()
            yield x_cpu.to(self.target_device, non_blocking=self.pin_memory), y_cpu.to(self.target_device, non_blocking=self.pin_memory)


def compute_direction_batch_sample_weight(y_binary: np.ndarray, *, mode: str) -> Optional[np.ndarray]:
    mode = str(mode).strip().lower()
    if mode == "none":
        return None
    yb = np.asarray(y_binary, dtype=np.int64).reshape(-1)
    pos = int(np.sum(yb == 1))
    neg = int(np.sum(yb == 0))
    n = pos + neg
    if n <= 0 or pos <= 0 or neg <= 0:
        return np.ones_like(yb, dtype=np.float32)
    pos_frac = pos / n
    neg_frac = neg / n
    if mode == "balanced":
        pos_w = 0.5 / pos_frac
        neg_w = 0.5 / neg_frac
    elif mode == "tempered":
        pos_w = math.sqrt(0.5 / pos_frac)
        neg_w = math.sqrt(0.5 / neg_frac)
    else:
        raise ValueError(f"Unsupported direction weighting mode {mode!r}")
    return np.where(yb == 1, pos_w, neg_w).astype(np.float32)


def compute_global_direction_weights(train_manifest: Dict[str, Any], stats: Dict[str, np.ndarray], mode: str) -> list[tuple[float, float]]:
    mode = str(mode).strip().lower()
    if mode == "none":
        return [(1.0, 1.0) for _ in HORIZONS_MS]
    pos_counts = np.zeros(len(HORIZONS_MS), dtype=np.float64)
    neg_counts = np.zeros(len(HORIZONS_MS), dtype=np.float64)
    for _Z, y_batch in iter_manifest_numpy_batches(train_manifest, batch_rows=LINEAR_STAGE4_BATCH_ROWS):
        keep_pos, keep_neg, keep_signed = build_signed_side_trim_masks_from_stats_np(y_batch, stats)
        for h in range(len(HORIZONS_MS)):
            rows = keep_signed[:, h]
            if rows.any():
                vals = y_batch[rows, h] > 0.0
                pos_counts[h] += float(np.sum(vals))
                neg_counts[h] += float(np.sum(~vals))
    weights: list[tuple[float, float]] = []
    for h in range(len(HORIZONS_MS)):
        pos = float(pos_counts[h])
        neg = float(neg_counts[h])
        n = pos + neg
        if n <= 0 or pos <= 0 or neg <= 0:
            weights.append((1.0, 1.0))
            continue
        pos_frac = pos / n
        neg_frac = neg / n
        if mode == "balanced":
            weights.append((0.5 / neg_frac, 0.5 / pos_frac))
        elif mode == "tempered":
            weights.append((math.sqrt(0.5 / neg_frac), math.sqrt(0.5 / pos_frac)))
        else:
            raise ValueError(f"Unsupported direction weighting mode {mode!r}")
    return weights


def make_direction_model(alpha: float, config: Dict[str, Any]) -> Any:
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(
        loss="log_loss",
        penalty=config["penalty"],
        alpha=float(alpha),
        l1_ratio=float(config["l1_ratio"]),
        fit_intercept=True,
        learning_rate="optimal",
        average=True,
        random_state=int(config["random_state"]),
    )


def make_magnitude_model(alpha: float, config: Dict[str, Any]) -> Any:
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(
        loss="huber",
        penalty=config["penalty"],
        alpha=float(alpha),
        l1_ratio=float(config["l1_ratio"]),
        fit_intercept=True,
        learning_rate="optimal",
        average=True,
        random_state=int(config["random_state"]),
    )


def train_stage4_candidate(
    *,
    train_manifest: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    alpha: float,
    config: Dict[str, Any],
) -> LinearSklearnTakerBundle:
    n_h = len(HORIZONS_MS)
    direction_models = [make_direction_model(alpha, config) for _ in range(n_h)]
    mag_up_models = [make_magnitude_model(alpha, config) for _ in range(n_h)]
    mag_down_models = [make_magnitude_model(alpha, config) for _ in range(n_h)]
    dir_fitted = [False] * n_h
    up_fitted = [False] * n_h
    down_fitted = [False] * n_h
    dir_counts = np.zeros(n_h, dtype=np.int64)
    up_counts = np.zeros(n_h, dtype=np.int64)
    down_counts = np.zeros(n_h, dtype=np.int64)
    dir_weights = compute_global_direction_weights(train_manifest, stats, str(config["direction_weighting"]))
    rng = np.random.default_rng(int(config["random_state"]))

    for epoch in range(int(config["epochs"])):
        print(f"[linear-stage4-train] alpha={alpha} epoch={epoch + 1}/{config['epochs']}", flush=True)
        for Z_batch, y_batch in iter_manifest_numpy_batches(
            train_manifest,
            batch_rows=int(config["batch_rows"]),
            shuffle=True,
            rng=rng,
        ):
            keep_pos, keep_neg, keep_signed = build_signed_side_trim_masks_from_stats_np(y_batch, stats)
            for h in range(n_h):
                rows = keep_signed[:, h]
                if rows.any():
                    y_dir = (y_batch[rows, h] > 0.0).astype(np.int64)
                    X_dir = Z_batch[rows]
                    if str(config["direction_weighting"]) == "none":
                        sample_weight = None
                    else:
                        neg_w, pos_w = dir_weights[h]
                        sample_weight = np.where(y_dir == 1, pos_w, neg_w).astype(np.float32)
                    if not dir_fitted[h]:
                        direction_models[h].partial_fit(X_dir, y_dir, classes=np.array([0, 1], dtype=np.int64), sample_weight=sample_weight)
                        dir_fitted[h] = True
                    else:
                        direction_models[h].partial_fit(X_dir, y_dir, sample_weight=sample_weight)
                    dir_counts[h] += int(y_dir.shape[0])

                rows = keep_pos[:, h]
                if rows.any():
                    y_up = np.sqrt(np.maximum(y_batch[rows, h], 0.0)).astype(np.float32)
                    if not up_fitted[h]:
                        mag_up_models[h].partial_fit(Z_batch[rows], y_up)
                        up_fitted[h] = True
                    else:
                        mag_up_models[h].partial_fit(Z_batch[rows], y_up)
                    up_counts[h] += int(y_up.shape[0])

                rows = keep_neg[:, h]
                if rows.any():
                    y_down = np.sqrt(np.maximum(-y_batch[rows, h], 0.0)).astype(np.float32)
                    if not down_fitted[h]:
                        mag_down_models[h].partial_fit(Z_batch[rows], y_down)
                        down_fitted[h] = True
                    else:
                        mag_down_models[h].partial_fit(Z_batch[rows], y_down)
                    down_counts[h] += int(y_down.shape[0])

    if not all(dir_fitted) or not all(up_fitted) or not all(down_fitted):
        raise ValueError("Insufficient train rows for one or more target/horizon models")
    print(
        f"[linear-stage4-counts] alpha={alpha} dir_rows={dir_counts.tolist()} up_rows={up_counts.tolist()} down_rows={down_counts.tolist()}",
        flush=True,
    )
    fit_summary = {
        "alpha": float(alpha),
        "penalty": str(config["penalty"]),
        "l1_ratio": float(config["l1_ratio"]),
        "epochs": int(config["epochs"]),
        "batch_rows": int(config["batch_rows"]),
        "dir_rows_per_horizon": dir_counts.tolist(),
        "up_rows_per_horizon": up_counts.tolist(),
        "down_rows_per_horizon": down_counts.tolist(),
        "direction_weights_neg_pos": [(float(a), float(b)) for a, b in dir_weights],
    }
    return LinearSklearnTakerBundle(
        schema=str(config["schema"]),
        config=dict(config, alpha=float(alpha)),
        horizons_ms=[int(x) for x in HORIZONS_MS],
        direction_models=direction_models,
        mag_up_models=mag_up_models,
        mag_down_models=mag_down_models,
        mag_floor=float(config["mag_floor"]),
        fit_summary=fit_summary,
    )


def evaluate_stage4_bundle(
    *,
    bundle: LinearSklearnTakerBundle,
    manifest: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    device: torch.device,
    split_name: str,
    max_rows: int = 0,
    batch_rows: Optional[int] = None,
) -> Dict[str, Any]:
    source = PreprocessedShardBatchSource(
        manifest,
        device=device,
        batch_rows=LINEAR_STAGE4_BATCH_ROWS if batch_rows is None else int(batch_rows),
        max_rows=max_rows,
    )
    model = LinearSklearnTorchWrapper(bundle).to(device)
    metrics = summarize_metrics(
        model,
        source,
        device,
        stats,
        amp_enabled=False,
        amp_dtype=torch.float32,
        primary_only=False,
        epoch=0,
        band_diag=BAND_DIAG,
        split_name=split_name,
    )
    primary_value, primary_label = compute_primary_metric(metrics)
    metrics["primary_metric_value"] = float(primary_value)
    metrics["primary_metric_label"] = str(primary_label)
    return metrics


def _jsonable_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): conv(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return {str(k): conv(v) for k, v in metrics.items()}



# ---------------------------------------------------------------------------
# Stage 5 comparison and sanity-check helpers
# ---------------------------------------------------------------------------


def resolve_stage4_dir(
    linear_out_dir: Path,
    extractor_name: str,
    preprocess_name: str,
    predictor: str,
) -> Path:
    return Path(linear_out_dir) / "stage4_models" / str(extractor_name) / str(preprocess_name) / str(predictor)


def load_stage4_payload(
    linear_out_dir: Path,
    extractor_name: str,
    preprocess_name: str,
    predictor: str,
) -> Dict[str, Any]:
    stage4_dir = resolve_stage4_dir(linear_out_dir, extractor_name, preprocess_name, predictor)
    path = stage4_dir / "linear_stage4_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 4 metrics for extractor={extractor_name}: {path}")
    payload = load_json(path)
    if payload.get("stage") != "stage4":
        raise ValueError(f"Expected stage4 payload at {path}, got stage={payload.get('stage')!r}")
    if "best_model_path" not in payload:
        raise ValueError(f"Stage 4 payload missing best_model_path: {path}")
    payload["payload_path"] = str(path)
    return payload


def load_stage4_artifacts_if_available(
    linear_out_dir: Path,
    extractor_name: str,
    preprocess_name: str,
    predictor: str,
    *,
    strict: bool,
) -> Optional[Dict[str, Any]]:
    try:
        return load_stage4_payload(linear_out_dir, extractor_name, preprocess_name, predictor)
    except FileNotFoundError as exc:
        if strict:
            raise
        print(f"[linear-stage5-warn] skipping extractor={extractor_name}: {exc}", flush=True)
        return None


def _metric_float(metrics: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        v = metrics.get(key, default)
        return float(v)
    except Exception:
        return float(default)


def _metric_horizon_value(metrics: Dict[str, Any], key: str, horizon_idx: int, default: float = float("nan")) -> float:
    if key in metrics:
        v = metrics.get(key)
        if isinstance(v, (list, tuple)) and horizon_idx < len(v):
            try:
                return float(v[horizon_idx])
            except Exception:
                return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)
    suffix_key = f"{key}_{int(HORIZONS_MS[horizon_idx])}ms"
    return _metric_float(metrics, suffix_key, default)


def extract_comparison_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out

    for h, horizon in enumerate([int(x) for x in HORIZONS_MS]):
        for prefix in [
            "dir_auc_kept",
            "dir_bal_acc_kept",
            "dir_auc_q50plus",
            "dir_bal_acc_q50plus",
            "edge_spearman_q50plus",
            "edge_bal_q50plus",
            "edge_bal_sign_acc_q50plus",
        ]:
            val = _metric_horizon_value(metrics, prefix, h)
            if math.isfinite(val):
                out[f"{prefix}_{horizon}ms"] = val

    for key in [
        "primary_metric_value",
        "primary_dir_bal_acc",
        "primary_metric_guard_passed",
        "n_eval_rows",
        "n_rows",
    ]:
        if key in metrics:
            try:
                out[key] = float(metrics[key])
            except Exception:
                pass
    return out


def collect_matching_metric_keys(metrics: Dict[str, Any]) -> Dict[str, float]:
    patterns = (
        "dir_auc",
        "dir_bal",
        "edge_spearman",
        "edge_bal",
        "prob_std",
        "spread_pos_neg",
    )
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        if not any(p in str(k) for p in patterns):
            continue
        if isinstance(v, (list, tuple)):
            for h, item in enumerate(v[: len(HORIZONS_MS)]):
                try:
                    out[f"{k}_{int(HORIZONS_MS[h])}ms"] = float(item)
                except Exception:
                    pass
        else:
            try:
                out[str(k)] = float(v)
            except Exception:
                pass
    return out


def _coef_array(model: Any) -> Optional[np.ndarray]:
    if not hasattr(model, "coef_"):
        return None
    coef = np.asarray(model.coef_, dtype=np.float64)
    return coef.reshape(-1)


def _summarize_one_model_coefficients(model: Any, *, task: str, horizon_index: int, top_k: int) -> Dict[str, Any]:
    coef = _coef_array(model)
    intercept = float("nan")
    if hasattr(model, "intercept_"):
        try:
            intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
        except Exception:
            intercept = float("nan")
    row: Dict[str, Any] = {
        "task": task,
        "horizon_index": int(horizon_index),
        "horizon_ms": int(HORIZONS_MS[horizon_index]),
        "intercept": intercept,
    }
    if coef is None or coef.size == 0:
        row.update({
            "coef_l2": float("nan"),
            "coef_l1": float("nan"),
            "coef_abs_max": float("nan"),
            "coef_nonzero_frac": float("nan"),
            "n_coefficients": 0,
            "top_coefficients": [],
        })
        return row
    abs_coef = np.abs(coef)
    row.update({
        "coef_l2": float(np.linalg.norm(coef)),
        "coef_l1": float(abs_coef.sum()),
        "coef_abs_max": float(abs_coef.max()),
        "coef_nonzero_frac": float(np.mean(abs_coef > 1e-12)),
        "n_coefficients": int(coef.size),
    })
    if top_k > 0:
        top_indices = np.argsort(-abs_coef)[: int(top_k)]
        row["top_coefficients"] = [
            {"index": int(i), "coef": float(coef[i]), "abs_coef": float(abs_coef[i])}
            for i in top_indices
        ]
    else:
        row["top_coefficients"] = []
    return row


def summarize_linear_model_coefficients(bundle: Any, *, top_k: int) -> Dict[str, Any]:
    tasks = {
        "direction": getattr(bundle, "direction_models", []),
        "mag_up": getattr(bundle, "mag_up_models", []),
        "mag_down": getattr(bundle, "mag_down_models", []),
    }
    out: Dict[str, Any] = {}
    for task, models in tasks.items():
        out[task] = [
            _summarize_one_model_coefficients(model, task=task, horizon_index=h, top_k=top_k)
            for h, model in enumerate(list(models)[: len(HORIZONS_MS)])
        ]
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x64, -60.0, 60.0)))).astype(np.float32)


def collect_predictions_and_labels_from_manifest(
    *,
    bundle: LinearSklearnTakerBundle,
    manifest: Dict[str, Any],
    max_rows: int,
    batch_rows: int,
) -> Dict[str, np.ndarray]:
    parts: Dict[str, list[np.ndarray]] = {
        "dir_logits": [],
        "p_up": [],
        "mag_up_sqrt": [],
        "mag_down_sqrt": [],
        "mag_up_bps": [],
        "mag_down_bps": [],
        "edge_bps": [],
        "y": [],
        "positions": [],
    }
    seen = 0
    for shard in manifest.get("shards", []):
        if max_rows > 0 and seen >= max_rows:
            break
        with np.load(shard["path"], allow_pickle=False) as arr:
            Z = arr["Z"].astype(np.float32, copy=False)
            y = arr["y"].astype(np.float32, copy=False)
            positions = arr["positions"].astype(np.int64, copy=False)
        if max_rows > 0:
            keep = max(0, min(Z.shape[0], int(max_rows) - seen))
            Z, y, positions = Z[:keep], y[:keep], positions[:keep]
        if Z.shape[0] <= 0:
            continue
        for start in range(0, Z.shape[0], int(batch_rows)):
            end = min(Z.shape[0], start + int(batch_rows))
            pred = bundle.predict_dict_np(Z[start:end])
            dir_logits = np.asarray(pred["dir_logits"], dtype=np.float32)
            mag_up_sqrt = np.asarray(pred["mag_up_sqrt"], dtype=np.float32)
            mag_down_sqrt = np.asarray(pred["mag_down_sqrt"], dtype=np.float32)
            p_up = _sigmoid_np(dir_logits)
            mag_up_bps = (mag_up_sqrt * mag_up_sqrt)
            mag_down_bps = (mag_down_sqrt * mag_down_sqrt)
            edge_bps = p_up * mag_up_bps - (1.0 - p_up) * mag_down_bps
            parts["dir_logits"].append(dir_logits)
            parts["p_up"].append(p_up.astype(np.float32, copy=False))
            parts["mag_up_sqrt"].append(mag_up_sqrt)
            parts["mag_down_sqrt"].append(mag_down_sqrt)
            parts["mag_up_bps"].append(mag_up_bps.astype(np.float32, copy=False))
            parts["mag_down_bps"].append(mag_down_bps.astype(np.float32, copy=False))
            parts["edge_bps"].append(edge_bps.astype(np.float32, copy=False))
            parts["y"].append(y[start:end])
            parts["positions"].append(positions[start:end])
        seen += int(Z.shape[0])
    if not parts["y"]:
        raise ValueError(f"Manifest contains no rows: {manifest.get('manifest_path')}")
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def _array_stats_by_horizon(arr: np.ndarray) -> list[Dict[str, float]]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = []
    for h in range(arr.shape[1]):
        x = arr[:, h]
        x = x[np.isfinite(x)]
        if x.size <= 0:
            out.append({k: float("nan") for k in ["mean", "std", "p01", "p05", "p50", "p95", "p99", "min", "max"]})
            continue
        out.append({
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=0)),
            "p01": float(np.quantile(x, 0.01)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        })
    return out


def _primary_horizon_index() -> int:
    try:
        return [int(x) for x in HORIZONS_MS].index(int(PRIMARY_METRIC_HORIZON_MS))
    except ValueError:
        return len(HORIZONS_MS) - 1


def summarize_edge_buckets(pred_payload: Dict[str, np.ndarray], split_name: str) -> Dict[str, Any]:
    y = np.asarray(pred_payload["y"], dtype=np.float32)
    edge = np.asarray(pred_payload["edge_bps"], dtype=np.float32)
    h = _primary_horizon_index()
    abs_edge = np.abs(edge[:, h])
    if abs_edge.size < 4:
        return {"split": split_name, "horizon_index": h, "buckets": []}
    qs = np.unique(np.quantile(abs_edge.astype(np.float64), [0.0, 0.25, 0.50, 0.75, 1.0]))
    buckets = []
    for i in range(max(0, len(qs) - 1)):
        lo, hi = float(qs[i]), float(qs[i + 1])
        mask = (abs_edge >= lo) & (abs_edge <= hi if i == len(qs) - 2 else abs_edge < hi)
        if not np.any(mask):
            continue
        truth = y[:, h][mask] > 0.0
        pred_up = edge[:, h][mask] >= 0.0
        buckets.append({
            "bucket": int(i),
            "abs_edge_lo": lo,
            "abs_edge_hi": hi,
            "n_rows": int(np.sum(mask)),
            "realized_pos_frac": float(np.mean(truth)),
            "edge_direction_acc": float(np.mean(pred_up == truth)),
        })
    return {"split": split_name, "horizon_index": h, "horizon_ms": int(HORIZONS_MS[h]), "buckets": buckets}


def summarize_prediction_arrays(pred_payload: Dict[str, np.ndarray]) -> Dict[str, Any]:
    y = np.asarray(pred_payload["y"])
    positions = np.asarray(pred_payload["positions"])
    out: Dict[str, Any] = {
        "n_rows": int(y.shape[0]),
        "position_min": int(np.min(positions)) if positions.size else None,
        "position_max": int(np.max(positions)) if positions.size else None,
        "horizons_ms": [int(x) for x in HORIZONS_MS],
    }
    for key in ["dir_logits", "p_up", "mag_up_sqrt", "mag_down_sqrt", "mag_up_bps", "mag_down_bps", "edge_bps", "y"]:
        out[key] = _array_stats_by_horizon(np.asarray(pred_payload[key]))
    edge = np.asarray(pred_payload["edge_bps"])
    p_up = np.asarray(pred_payload["p_up"])
    out["edge_positive_frac_by_horizon"] = [float(np.mean(edge[:, h] > 0.0)) for h in range(edge.shape[1])]
    out["p_up_gt_0p5_frac_by_horizon"] = [float(np.mean(p_up[:, h] > 0.5)) for h in range(p_up.shape[1])]
    out["edge_abs_p95_by_horizon"] = [float(np.quantile(np.abs(edge[:, h]), 0.95)) for h in range(edge.shape[1])]
    out["edge_buckets_primary_horizon"] = summarize_edge_buckets(pred_payload, "prediction_summary")
    return out


def _balanced_acc_bool(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    pos = truth
    neg = ~truth
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    return 0.5 * (float(np.mean(pred[pos] == truth[pos])) + float(np.mean(pred[neg] == truth[neg])))


def compute_array_metrics_with_cmssl_logic(
    *,
    pred: Dict[str, np.ndarray],
    y: np.ndarray,
    stats: Dict[str, np.ndarray],
    split_name: str,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float32)
    keep_pos, keep_neg, keep_signed = build_signed_side_trim_masks_from_stats_np(y, stats)
    out: Dict[str, Any] = {"split_name": split_name, "n_rows": int(y.shape[0])}
    dir_logits = np.asarray(pred["dir_logits"], dtype=np.float32)
    edge_bps = np.asarray(pred["edge_bps"], dtype=np.float32)
    q50 = np.asarray(stats.get("kept_q50_abs_raw_bps", np.full(len(HORIZONS_MS), np.nan)), dtype=np.float32).reshape(-1)
    for h, horizon in enumerate([int(x) for x in HORIZONS_MS]):
        kh = keep_signed[:, h]
        out[f"kept_frac_{horizon}ms"] = float(np.mean(kh)) if kh.size else float("nan")
        if int(np.sum(kh)) >= 2:
            truth = y[:, h][kh] > 0.0
            scores = dir_logits[:, h][kh]
            out[f"dir_auc_kept_{horizon}ms"] = _binary_auc_np(scores, truth)
            out[f"dir_bal_acc_kept_{horizon}ms"] = _balanced_acc_bool(scores >= 0.0, truth)
            out[f"edge_spearman_kept_{horizon}ms"] = _safe_spearman_np(edge_bps[:, h][kh], y[:, h][kh])
        else:
            out[f"dir_auc_kept_{horizon}ms"] = float("nan")
            out[f"dir_bal_acc_kept_{horizon}ms"] = float("nan")
            out[f"edge_spearman_kept_{horizon}ms"] = float("nan")
        q50plus = kh & (np.abs(y[:, h]) >= float(q50[h] if h < q50.shape[0] else np.nan))
        if int(np.sum(q50plus)) >= 2:
            truth_q = y[:, h][q50plus] > 0.0
            out[f"edge_spearman_q50plus_{horizon}ms"] = _safe_spearman_np(edge_bps[:, h][q50plus], y[:, h][q50plus])
            out[f"edge_bal_q50plus_{horizon}ms"] = _balanced_acc_bool(edge_bps[:, h][q50plus] >= 0.0, truth_q)
        else:
            out[f"edge_spearman_q50plus_{horizon}ms"] = float("nan")
            out[f"edge_bal_q50plus_{horizon}ms"] = float("nan")
    ph = _primary_horizon_index()
    out["primary_like_auc"] = out.get(f"dir_auc_kept_{int(HORIZONS_MS[ph])}ms", float("nan"))
    out["primary_like_bal_acc"] = out.get(f"dir_bal_acc_kept_{int(HORIZONS_MS[ph])}ms", float("nan"))
    return out


def _slice_prediction_payload(pred_payload: Dict[str, np.ndarray], mask_or_indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[mask_or_indices] for k, v in pred_payload.items()}


def make_shifted_y(y: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return y.copy()
    y_shift = np.empty_like(y)
    y_shift[:] = np.nan
    if shift > 0:
        y_shift[:-shift] = y[shift:]
    else:
        k = abs(shift)
        y_shift[k:] = y[:-k]
    return y_shift


def run_label_shift_sanity_checks(
    *,
    pred_payload: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
    shifts: list[int],
    split_name: str,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for shift in shifts:
        y_shift = make_shifted_y(np.asarray(pred_payload["y"], dtype=np.float32), int(shift))
        valid = np.isfinite(y_shift).all(axis=1)
        if not np.any(valid):
            checks[f"shift_{int(shift)}"] = {"n_rows": 0, "error": "no valid shifted rows"}
            continue
        checks[f"shift_{int(shift)}"] = compute_array_metrics_with_cmssl_logic(
            pred=_slice_prediction_payload(pred_payload, valid),
            y=y_shift[valid],
            stats=stats,
            split_name=f"{split_name}_shift_{int(shift)}",
        )
    return checks


def run_label_permutation_sanity_check(
    *,
    pred_payload: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
    seed: int,
    split_name: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(pred_payload["y"].shape[0])
    return compute_array_metrics_with_cmssl_logic(
        pred=pred_payload,
        y=np.asarray(pred_payload["y"])[perm],
        stats=stats,
        split_name=f"{split_name}_label_permutation_seed_{int(seed)}",
    )


def save_stage5_prediction_dump(path: Path, pred_payload: Dict[str, np.ndarray], max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = pred_payload["y"].shape[0] if max_rows <= 0 else min(int(max_rows), pred_payload["y"].shape[0])
    np.savez_compressed(path, **{k: np.asarray(v)[:n] for k, v in pred_payload.items()})



def load_stage3_audit_summary_for_stage5(stage3_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    audit_summary = stage3_payload.get("audit_summary")
    if isinstance(audit_summary, dict):
        return audit_summary
    audit_path = stage3_payload.get("audit_summary_path")
    if audit_path:
        path = Path(audit_path)
        if path.exists():
            try:
                loaded = load_json(path)
                return loaded if isinstance(loaded, dict) else None
            except Exception as exc:
                print(f"[linear-stage5-warn] could not load Stage 3 audit summary {path}: {exc}", flush=True)
    return None


def add_stage3_audit_fields_to_comparison_row(row: Dict[str, Any], audit_summary: Optional[Dict[str, Any]]) -> None:
    if not isinstance(audit_summary, dict):
        return
    splits = audit_summary.get("splits", {}) if isinstance(audit_summary.get("splits"), dict) else {}
    train = splits.get("train") if isinstance(splits.get("train"), dict) else (splits.get("train_sample") if isinstance(splits.get("train_sample"), dict) else {})
    val = splits.get("val") if isinstance(splits.get("val"), dict) else {}
    row["preprocess_kept_dim"] = train.get("kept_dim") or val.get("kept_dim") or audit_summary.get("kept_dim")
    row["preprocess_variance_removed_frac"] = train.get("variance_removed_frac", float("nan"))
    row["preprocess_train_clip_p95"] = train.get("winsor_total_clip_frac_p95", float("nan"))
    row["preprocess_val_clip_p95"] = val.get("winsor_total_clip_frac_p95", float("nan"))
    row["preprocess_val_out_abs_p999"] = val.get("out_abs_p999", float("nan"))
    row["preprocess_val_out_abs_gt_10_frac"] = val.get("out_abs_gt_10_frac", float("nan"))
    warnings: list[str] = []
    for split_name, summary in splits.items():
        if isinstance(summary, dict):
            for warning in summary.get("warnings", []) or []:
                warnings.append(f"{split_name}:{warning}")
    row["preprocess_warnings"] = ";".join(warnings)

def build_stage5_comparison_row(
    *,
    extractor_name: str,
    preprocess_name: str,
    predictor: str,
    stage4_payload: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]],
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    best_primary = stage4_payload.get("best_primary_metric", {}) or {}
    row: Dict[str, Any] = {
        "extractor": extractor_name,
        "preprocess_name": preprocess_name,
        "predictor": predictor,
        "best_alpha": stage4_payload.get("best_alpha"),
        "train_split": stage4_payload.get("train_split"),
        "train_rows": stage4_payload.get("train_rows"),
        "original_dim": stage4_payload.get("original_dim"),
        "kept_dim": stage4_payload.get("kept_dim"),
        "best_model_path": stage4_payload.get("best_model_path"),
        "val_primary_metric_name": best_primary.get("name", best_primary.get("label")),
        "val_primary_metric_value": best_primary.get("value"),
        "val_guard_passed": best_primary.get("guard_passed"),
    }
    for k, v in extract_comparison_metrics(val_metrics).items():
        row[f"val_{k}"] = v
    for k, v in collect_matching_metric_keys(val_metrics).items():
        row.setdefault(f"val_{k}", v)
    if test_metrics:
        for k, v in extract_comparison_metrics(test_metrics).items():
            row[f"test_{k}"] = v
        for k, v in collect_matching_metric_keys(test_metrics).items():
            row.setdefault(f"test_{k}", v)
    ph_ms = int(HORIZONS_MS[_primary_horizon_index()])
    shift_checks = diagnostics.get("label_shift_sanity_val", {}) or {}
    for shift_key, metrics in shift_checks.items():
        if isinstance(metrics, dict):
            row[f"val_{shift_key}_primary_like_auc_{ph_ms}ms"] = metrics.get("primary_like_auc", float("nan"))
            row[f"val_{shift_key}_dir_auc_kept_{ph_ms}ms"] = metrics.get(f"dir_auc_kept_{ph_ms}ms", float("nan"))
    perm = diagnostics.get("label_permutation_sanity_val")
    if isinstance(perm, dict):
        row[f"val_perm_dir_auc_kept_{ph_ms}ms"] = perm.get(f"dir_auc_kept_{ph_ms}ms", float("nan"))
        row[f"val_perm_edge_spearman_q50plus_{ph_ms}ms"] = perm.get(f"edge_spearman_q50plus_{ph_ms}ms", float("nan"))
    return row


def write_rows_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _maybe_add_baseline_row(rows: list[Dict[str, Any]], *, strict: bool) -> None:
    if not LINEAR_STAGE5_BASELINE_METRICS_JSON:
        return
    try:
        baseline = load_json(Path(LINEAR_STAGE5_BASELINE_METRICS_JSON))
        metrics = baseline.get("val_metrics", baseline.get("val_full_metrics", baseline))
        row: Dict[str, Any] = {"extractor": "CMSSL_neural_baseline", "preprocess_name": "", "predictor": "SAMBA"}
        for k, v in extract_comparison_metrics(metrics).items():
            row[f"val_{k}"] = v
        for k, v in collect_matching_metric_keys(metrics).items():
            row.setdefault(f"val_{k}", v)
        rows.append(row)
    except Exception as exc:
        if strict:
            raise
        print(f"[linear-stage5-warn] could not parse baseline metrics JSON: {exc}", flush=True)


def run_stage5_comparison(
    *,
    linear_out_dir: Path,
    extractor_names: list[str],
    preprocess_name: str,
    predictor: str,
    device: torch.device,
) -> Dict[str, Any]:
    stage5_dir = Path(linear_out_dir) / "stage5_comparison" / preprocess_name / predictor
    diag_dir = stage5_dir / "stage5_diagnostics"
    stage5_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    stats = load_linear_trim_stats(linear_out_dir)
    _print_decision_row_policy("stage5")

    for extractor_name in extractor_names:
        stage4_payload = load_stage4_artifacts_if_available(
            linear_out_dir,
            extractor_name,
            preprocess_name,
            predictor,
            strict=LINEAR_STAGE5_STRICT,
        )
        if stage4_payload is None:
            continue
        bundle_path = Path(str(stage4_payload["best_model_path"]))
        bundle = load_linear_sklearn_bundle(bundle_path)
        try:
            _validate_manifest_decision_policy(stage4_payload, context=f"stage5 stage4 {extractor_name}")
        except ValueError as exc:
            if LINEAR_STAGE5_STRICT:
                raise
            print(f"[linear-stage5-warn] {exc}; skipping extractor={extractor_name}", flush=True)
            continue
        stage3_payload = load_stage3_payload(linear_out_dir, extractor_name, preprocess_name)
        val_manifest = load_stage3_manifest(stage3_payload, "val")
        test_manifest = load_stage3_manifest(stage3_payload, "test")
        if val_manifest is None:
            raise ValueError(f"Missing val manifest for extractor={extractor_name}")
        try:
            _validate_manifest_decision_policy(val_manifest, context=f"stage5 val {extractor_name}")
            if test_manifest is not None:
                _validate_manifest_decision_policy(test_manifest, context=f"stage5 test {extractor_name}")
        except ValueError as exc:
            if LINEAR_STAGE5_STRICT:
                raise
            print(f"[linear-stage5-warn] {exc}; skipping extractor={extractor_name}", flush=True)
            continue
        if LINEAR_STAGE5_RUN_TEST and test_manifest is None:
            msg = f"Missing test manifest for extractor={extractor_name}"
            if LINEAR_STAGE5_STRICT:
                raise ValueError(msg)
            print(f"[linear-stage5-warn] {msg}; skipping test diagnostics", flush=True)

        if LINEAR_STAGE5_REEVALUATE:
            val_metrics = evaluate_stage4_bundle(
                bundle=bundle,
                manifest=val_manifest,
                stats=stats,
                device=device,
                split_name=f"stage5_val_{extractor_name}",
                max_rows=LINEAR_STAGE5_MAX_VAL_ROWS,
                batch_rows=LINEAR_STAGE5_BATCH_ROWS,
            )
        else:
            val_metrics = stage4_payload.get("val_metrics", {}) or {}

        test_metrics = None
        if LINEAR_STAGE5_RUN_TEST and test_manifest is not None:
            if LINEAR_STAGE5_REEVALUATE:
                test_metrics = evaluate_stage4_bundle(
                    bundle=bundle,
                    manifest=test_manifest,
                    stats=stats,
                    device=device,
                    split_name=f"stage5_test_{extractor_name}",
                    max_rows=LINEAR_STAGE5_MAX_TEST_ROWS,
                    batch_rows=LINEAR_STAGE5_BATCH_ROWS,
                )
            else:
                test_metrics = stage4_payload.get("test_metrics")

        coef_diag = summarize_linear_model_coefficients(bundle, top_k=LINEAR_STAGE5_TOP_COEFS)
        pred_val = collect_predictions_and_labels_from_manifest(
            bundle=bundle,
            manifest=val_manifest,
            max_rows=LINEAR_STAGE5_MAX_VAL_ROWS,
            batch_rows=LINEAR_STAGE5_BATCH_ROWS,
        )
        pred_val_summary = summarize_prediction_arrays(pred_val)
        shift_checks = run_label_shift_sanity_checks(
            pred_payload=pred_val,
            stats=stats,
            shifts=LINEAR_STAGE5_LABEL_SHIFT_VALUES,
            split_name=f"val_{extractor_name}",
        ) if LINEAR_STAGE5_LABEL_SHIFT_VALUES else {}
        perm_check = run_label_permutation_sanity_check(
            pred_payload=pred_val,
            stats=stats,
            seed=LINEAR_STAGE5_PERMUTATION_SEED,
            split_name=f"val_{extractor_name}",
        ) if LINEAR_STAGE5_LABEL_PERMUTATION else None

        pred_test_summary = None
        shift_checks_test = {}
        perm_check_test = None
        if LINEAR_STAGE5_RUN_TEST and test_manifest is not None:
            pred_test = collect_predictions_and_labels_from_manifest(
                bundle=bundle,
                manifest=test_manifest,
                max_rows=LINEAR_STAGE5_MAX_TEST_ROWS,
                batch_rows=LINEAR_STAGE5_BATCH_ROWS,
            )
            pred_test_summary = summarize_prediction_arrays(pred_test)
            shift_checks_test = run_label_shift_sanity_checks(
                pred_payload=pred_test,
                stats=stats,
                shifts=LINEAR_STAGE5_LABEL_SHIFT_VALUES,
                split_name=f"test_{extractor_name}",
            ) if LINEAR_STAGE5_LABEL_SHIFT_VALUES else {}
            perm_check_test = run_label_permutation_sanity_check(
                pred_payload=pred_test,
                stats=stats,
                seed=LINEAR_STAGE5_PERMUTATION_SEED,
                split_name=f"test_{extractor_name}",
            ) if LINEAR_STAGE5_LABEL_PERMUTATION else None
            if LINEAR_STAGE5_SAVE_PREDICTIONS:
                save_stage5_prediction_dump(
                    diag_dir / extractor_name / "test_predictions.npz",
                    pred_test,
                    LINEAR_STAGE5_PREDICTION_MAX_ROWS,
                )
        if LINEAR_STAGE5_SAVE_PREDICTIONS:
            save_stage5_prediction_dump(
                diag_dir / extractor_name / "val_predictions.npz",
                pred_val,
                LINEAR_STAGE5_PREDICTION_MAX_ROWS,
            )

        stage3_audit_summary = load_stage3_audit_summary_for_stage5(stage3_payload)
        diag = {
            "stage4_payload_path": stage4_payload.get("payload_path"),
            "stage3_audit_summary": stage3_audit_summary,
            "best_model_path": str(bundle_path),
            "coefficient_diagnostics": coef_diag,
            "prediction_summary_val": pred_val_summary,
            "prediction_summary_test": pred_test_summary,
            "label_shift_sanity_val": shift_checks,
            "label_permutation_sanity_val": perm_check,
            "label_shift_sanity_test": shift_checks_test,
            "label_permutation_sanity_test": perm_check_test,
            "val_metrics": _jsonable_metrics(val_metrics),
            "test_metrics": _jsonable_metrics(test_metrics),
        }
        diagnostics[extractor_name] = diag
        row = build_stage5_comparison_row(
            extractor_name=extractor_name,
            preprocess_name=preprocess_name,
            predictor=predictor,
            stage4_payload=stage4_payload,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            diagnostics=diag,
        )
        add_stage3_audit_fields_to_comparison_row(row, stage3_audit_summary)
        row["decision_stride_rows"] = int(LINEAR_DECISION_STRIDE_ROWS)
        row["decision_offset_rows"] = int(LINEAR_DECISION_OFFSET_ROWS)
        rows.append(row)
        diag_path = stage5_dir / f"diagnostics_{extractor_name}.json"
        diag_path.write_text(json.dumps(diag, allow_nan=True, indent=2), encoding="utf-8")
        print(f"[linear-stage5] wrote {diag_path}", flush=True)

    _maybe_add_baseline_row(rows, strict=LINEAR_STAGE5_STRICT)
    for row in rows:
        row.setdefault("decision_stride_rows", int(LINEAR_DECISION_STRIDE_ROWS))
        row.setdefault("decision_offset_rows", int(LINEAR_DECISION_OFFSET_ROWS))
    csv_path = stage5_dir / "linear_stage5_comparison.csv"
    json_path = stage5_dir / "linear_stage5_comparison.json"
    copy_csv_path = Path(linear_out_dir) / "linear_stage5_comparison.csv"
    copy_json_path = Path(linear_out_dir) / "linear_stage5_comparison.json"
    write_rows_csv(csv_path, rows)
    write_rows_csv(copy_csv_path, rows)
    payload = {
        "stage": "stage5",
        "status": "ok",
        "schema": LINEAR_STAGE5_SCHEMA,
        **_decision_metadata(),
        "linear_out_dir": str(linear_out_dir),
        "preprocess_name": preprocess_name,
        "predictor": predictor,
        "extractors_requested": extractor_names,
        "extractors_completed": [row["extractor"] for row in rows if row.get("extractor") != "CMSSL_neural_baseline"],
        "strict": bool(LINEAR_STAGE5_STRICT),
        "reevaluate": bool(LINEAR_STAGE5_REEVALUATE),
        "run_test": bool(LINEAR_STAGE5_RUN_TEST),
        "label_shifts": LINEAR_STAGE5_LABEL_SHIFT_VALUES,
        "label_permutation": bool(LINEAR_STAGE5_LABEL_PERMUTATION),
        "comparison_rows": rows,
        "diagnostics": diagnostics,
        "saved_files": {
            "comparison_csv": str(csv_path),
            "comparison_json": str(json_path),
            "comparison_csv_copy": str(copy_csv_path),
            "comparison_json_copy": str(copy_json_path),
        },
    }
    json_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    copy_json_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    print(f"[linear-stage5] wrote {csv_path}, {json_path}, {copy_csv_path}, and {copy_json_path}", flush=True)
    return payload

def run_stage4_training(
    *,
    linear_out_dir: Path,
    extractor_name: str,
    preprocess_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    stage3_payload = load_stage3_payload(linear_out_dir, extractor_name, preprocess_name)
    train_manifest = load_stage3_manifest(stage3_payload, LINEAR_STAGE4_TRAIN_SPLIT)
    val_manifest = load_stage3_manifest(stage3_payload, "val")
    test_manifest = load_stage3_manifest(stage3_payload, "test")
    if train_manifest is None:
        raise ValueError("Stage 4 requires a Stage 3 train_sample manifest")
    if val_manifest is None:
        raise ValueError("Stage 4 requires a Stage 3 val manifest")
    _print_decision_row_policy("stage4")
    _validate_manifest_decision_policy(train_manifest, context="stage4 train")
    _validate_manifest_decision_policy(val_manifest, context="stage4 val")
    if test_manifest is not None:
        _validate_manifest_decision_policy(test_manifest, context="stage4 test")
    try:
        stats = load_linear_trim_stats(linear_out_dir)
        trim_stats_source = str(Path(linear_out_dir) / "linear_signed_side_trim_stats_cache.npz")
    except FileNotFoundError:
        if not LINEAR_STAGE4_ALLOW_SAMPLE_TRIM_STATS:
            raise
        stats = compute_sample_trim_stats_from_manifest(train_manifest)
        trim_stats_source = "train_sample_manifest_escape_hatch"

    stage4_dir = Path(linear_out_dir) / "stage4_models" / extractor_name / preprocess_name / LINEAR_STAGE4_PREDICTOR
    stage4_dir.mkdir(parents=True, exist_ok=True)
    stage4_config = {
        "schema": LINEAR_STAGE4_SCHEMA,
        "extractor": extractor_name,
        "preprocess_name": preprocess_name,
        "predictor": LINEAR_STAGE4_PREDICTOR,
        "alpha_grid": [float(a) for a in LINEAR_STAGE4_ALPHA_VALUES],
        "penalty": LINEAR_STAGE4_PENALTY,
        "l1_ratio": float(LINEAR_STAGE4_L1_RATIO),
        "epochs": int(LINEAR_STAGE4_EPOCHS),
        "batch_rows": int(LINEAR_STAGE4_BATCH_ROWS),
        "random_state": int(LINEAR_STAGE4_RANDOM_SEED),
        "direction_weighting": LINEAR_STAGE4_DIRECTION_WEIGHTING,
        "mag_sample_weighting": LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING,
        "mag_floor": float(LINEAR_STAGE4_MAG_FLOOR),
        **_decision_metadata(),
    }

    candidate_summaries = []
    best_bundle = None
    best_val_metrics = None
    best_score = float("-inf")
    best_alpha = None
    for alpha in LINEAR_STAGE4_ALPHA_VALUES:
        print(f"[linear-stage4-train] alpha={alpha}", flush=True)
        bundle = train_stage4_candidate(train_manifest=train_manifest, stats=stats, alpha=float(alpha), config=stage4_config)
        val_metrics = evaluate_stage4_bundle(
            bundle=bundle,
            manifest=val_manifest,
            stats=stats,
            device=device,
            split_name="val",
            max_rows=LINEAR_STAGE4_MAX_VAL_ROWS,
        )
        primary_value, primary_label = compute_primary_metric(val_metrics)
        guard_passed = bool(val_metrics.get("primary_metric_guard_passed", True)) and math.isfinite(float(primary_value))
        candidate_score = float(primary_value) if guard_passed else float("-inf")
        print(f"[linear-stage4-val] alpha={alpha} primary={primary_value} guard={guard_passed}", flush=True)
        candidate_summaries.append({
            "alpha": float(alpha),
            "primary_metric_label": str(primary_label),
            "primary_metric_value": float(primary_value),
            "guard_passed": bool(guard_passed),
            "val_metrics": _jsonable_metrics(val_metrics),
            "fit_summary": bundle.fit_summary,
        })
        if best_bundle is None or is_metric_improved(candidate_score, best_score, "max"):
            best_bundle = bundle
            best_val_metrics = val_metrics
            best_score = candidate_score
            best_alpha = float(alpha)
            print(f"[linear-stage4-best] alpha={best_alpha} metric={best_score}", flush=True)

    if best_bundle is None or best_val_metrics is None or best_alpha is None:
        raise ValueError("Stage 4 did not produce any trainable candidates")
    bundle_path = stage4_dir / "linear_stage4_best_model.pkl"
    save_linear_sklearn_bundle(best_bundle, bundle_path)

    test_metrics = None
    if LINEAR_STAGE4_RUN_TEST and test_manifest is not None:
        print("[linear-stage4-test] evaluating best bundle", flush=True)
        test_metrics = evaluate_stage4_bundle(
            bundle=best_bundle,
            manifest=test_manifest,
            stats=stats,
            device=device,
            split_name="test",
            max_rows=LINEAR_STAGE4_MAX_TEST_ROWS,
        )

    payload = {
        "stage": "stage4",
        "status": "ok",
        "schema": LINEAR_STAGE4_SCHEMA,
        **_decision_metadata(),
        "stage4_config": stage4_config,
        "extractor": extractor_name,
        "preprocess_name": preprocess_name,
        "stage3_payload_path": stage3_payload.get("payload_path"),
        "trim_stats_cache": trim_stats_source,
        "train_split": LINEAR_STAGE4_TRAIN_SPLIT,
        "train_rows": int(manifest_n_rows(train_manifest)),
        "best_alpha": float(best_alpha),
        "best_model_path": str(bundle_path),
        "best_primary_metric": {
            "label": str(best_val_metrics.get("primary_metric_label", PRIMARY_METRIC)),
            "value": float(best_val_metrics.get("primary_metric_value", best_score)),
            "guard_passed": bool(best_val_metrics.get("primary_metric_guard_passed", True)),
        },
        "candidate_summaries": candidate_summaries,
        "val_metrics": _jsonable_metrics(best_val_metrics),
        "test_metrics": _jsonable_metrics(test_metrics),
        "train_manifest": train_manifest.get("manifest_path"),
        "val_manifest": val_manifest.get("manifest_path"),
        "test_manifest": None if test_manifest is None else test_manifest.get("manifest_path"),
    }
    metrics_path = stage4_dir / "linear_stage4_metrics.json"
    copy_path = Path(linear_out_dir) / "linear_stage4_metrics.json"
    metrics_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    copy_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    print(f"[linear-stage4] wrote {metrics_path} and {copy_path}", flush=True)
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
    _print_decision_row_policy("stage2")

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
        **_decision_metadata(),
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
    if LINEAR_STAGE == "stage5":
        print(
            f"[linear-config] stage=stage5 extractors={LINEAR_STAGE5_EXTRACTOR_VALUES} "
            f"preprocess={LINEAR_STAGE5_PREPROCESS_NAME} predictor={LINEAR_STAGE5_PREDICTOR} "
            f"device={device} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        run_stage5_comparison(
            linear_out_dir=linear_out_dir,
            extractor_names=LINEAR_STAGE5_EXTRACTOR_VALUES,
            preprocess_name=LINEAR_STAGE5_PREPROCESS_NAME,
            predictor=LINEAR_STAGE5_PREDICTOR,
            device=device,
        )
        return
    if LINEAR_STAGE == "stage3":
        print(
            f"[linear-config] stage=stage3 extractor={LINEAR_EXTRACTOR} device={device} "
            f"out_root={out_root} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        run_stage3_preprocessing(linear_out_dir=linear_out_dir, extractor_name=LINEAR_EXTRACTOR)
        return
    if LINEAR_STAGE == "stage4":
        print(
            f"[linear-config] stage=stage4 extractor={LINEAR_EXTRACTOR} "
            f"preprocess={LINEAR_STAGE4_PREPROCESS_NAME} predictor={LINEAR_STAGE4_PREDICTOR} "
            f"device={device} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        run_stage4_training(
            linear_out_dir=linear_out_dir,
            extractor_name=LINEAR_EXTRACTOR,
            preprocess_name=LINEAR_STAGE4_PREPROCESS_NAME,
            device=device,
        )
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

    _print_decision_row_policy(LINEAR_STAGE)
    y_train = collect_train_labels_from_datasets(ds_train_list, split_name="train")

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

    val_full_src = DatasetPositionsBatchSource(
        ds_val,
        device,
        LINEAR_EVAL_BATCH_SIZE,
        split_name="linear_val_full",
    )
    val_fast_src = val_full_src.make_evenly_spaced_subset(FAST_VAL_MAX_ROWS)

    train_band_metrics: Optional[Dict[str, Any]] = None
    if BAND_DIAG and BAND_DIAG_TRAIN:
        train_sources = [
            DatasetPositionsBatchSource(
                ds,
                device,
                LINEAR_EVAL_BATCH_SIZE,
                split_name=f"linear_train_band_week{i}",
            )
            for i, ds in enumerate(ds_train_list)
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
        test_src = DatasetPositionsBatchSource(
            ds_test,
            device,
            LINEAR_EVAL_BATCH_SIZE,
            split_name="linear_test",
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
        **_decision_metadata(),
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
            **_decision_metadata(),
        },
        "prior": linear_model_summary(model),
        "stats": stats,
        "val_fast_metrics": val_fast,
        "val_full_metrics": val_full,
    }
    ckpt_path = linear_out_dir / "linear_stage1_prior.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[linear_ckpt] saved {ckpt_path}", flush=True)


# ---------------------------------------------------------------------------
# Streaming feature pipeline overrides (default path; persisted shards legacy)
# ---------------------------------------------------------------------------


def build_linear_cmssl_splits_from_out_root(*, out_root: Path) -> Dict[str, Any]:
    out_root = Path(out_root)
    meta = json.loads((out_root / "meta.json").read_text())
    validate_contract_meta(meta, "global meta.json")
    validate_dataset_label_dim(meta, "global meta.json")
    split_info = require_supported_pipeline_splits(meta, out_root)
    protocol = split_info["protocol"]
    cmssl = split_info["splits"]["cmssl"]
    train_week_keys = list(cmssl["train"]["weeks"])
    cmssl_val = cmssl["val"]
    cmssl_test = cmssl.get("test")
    has_cmssl_test = cmssl_test is not None and bool(cmssl_test.get("weeks"))
    train_split_entries = [make_single_week_split_from_meta(out_root=out_root, global_meta=meta, week_key=wk) for wk in train_week_keys]
    ds_train_list = [build_dataset_from_split(str(out_root), entry) for entry in train_split_entries]
    ds_val = build_dataset_from_split(str(out_root), cmssl_val)
    ds_test = build_dataset_from_split(str(out_root), cmssl_test) if has_cmssl_test else None
    feature_dim_total = int(meta["feature_dim_total"])
    for i, ds_train in enumerate(ds_train_list):
        _validate_dataset_split(ds_train, f"train[{i}]/{train_week_keys[i]}", feature_dim_total)
    _validate_dataset_split(ds_val, "val", feature_dim_total)
    if ds_test is not None:
        _validate_dataset_split(ds_test, "test", feature_dim_total)
    print(
        f"[split] protocol={protocol} cmssl.train={','.join(train_week_keys)} "
        f"cmssl.val={cmssl_val.get('weeks')} cmssl.test={cmssl_test.get('weeks') if has_cmssl_test else '<missing>'}",
        flush=True,
    )
    return {
        "meta": meta,
        "protocol": protocol,
        "train_week_keys": train_week_keys,
        "train_split_entries": train_split_entries,
        "ds_train_list": ds_train_list,
        "ds_val": ds_val,
        "ds_test": ds_test,
        "has_cmssl_test": has_cmssl_test,
        "feature_dim_total": feature_dim_total,
        "val_split_entry": cmssl_val,
        "test_split_entry": cmssl_test,
    }


def _stage_payload_path(linear_out_dir: Path, extractor_name: str, preprocess_name: str = "default") -> tuple[Path, Path]:
    return (
        resolve_stage2_dir(linear_out_dir, extractor_name) / "linear_stage2_extractor_metrics.json",
        resolve_stage3_dir(linear_out_dir, extractor_name, preprocess_name) / "linear_stage3_preprocess_metrics.json",
    )


def load_stage2_payload(linear_out_dir: Path, extractor_name: str) -> Dict[str, Any]:  # type: ignore[override]
    path, _ = _stage_payload_path(Path(linear_out_dir), extractor_name)
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 payload not found for extractor={extractor_name!r}: {path}")
    payload = load_json(path)
    if payload.get("stage") != "stage2":
        raise ValueError(f"Expected stage2 payload at {path}, got stage={payload.get('stage')!r}")
    if not payload.get("streaming_features", False):
        raise ValueError("Stage 2 payload was created by old persisted-shard pipeline. Rerun Stage 2 with the streaming pipeline.")
    if "extractor_output_dim" not in payload or "extractor_pickle" not in payload:
        raise ValueError(f"Stage 2 payload missing extractor_output_dim/extractor_pickle: {path}")
    payload["payload_path"] = str(path)
    return payload


def load_stage2_extractor_bundle(*, linear_out_dir: Path, extractor_name: str) -> tuple[Any, Dict[str, Any]]:
    payload = load_stage2_payload(linear_out_dir, extractor_name)
    _validate_manifest_decision_policy(payload, context="stage2 extractor")
    pkl_path = Path(str(payload["extractor_pickle"]))
    if not pkl_path.exists():
        raise FileNotFoundError(f"Stage 2 extractor pickle not found: {pkl_path}")
    with pkl_path.open("rb") as f:
        extractor = pickle.load(f)
    return extractor, payload


def load_stage3_payload(linear_out_dir: Path, extractor_name: str, preprocess_name: str) -> Dict[str, Any]:  # type: ignore[override]
    _, path = _stage_payload_path(Path(linear_out_dir), extractor_name, preprocess_name)
    if not path.exists():
        raise FileNotFoundError(f"Stage 3 payload not found for extractor={extractor_name!r} preprocess={preprocess_name!r}: {path}")
    payload = load_json(path)
    if payload.get("stage") != "stage3":
        raise ValueError(f"Expected stage3 payload at {path}, got stage={payload.get('stage')!r}")
    if not payload.get("streaming_features", False):
        raise ValueError("Stage 3 payload was created by old persisted-shard pipeline. Rerun Stage 3 with the streaming pipeline.")
    payload["payload_path"] = str(path)
    return payload


def iter_dataset_window_batches(ds: Any, *, batch_rows: int, max_rows: int = 0, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    positions = _dataset_positions(len(ds), int(max_rows))
    batch_rows = max(1, int(batch_rows))
    for start in range(0, len(positions), batch_rows):
        pos = positions[start:start + batch_rows]
        if shuffle_within_batch:
            rng = np.random.default_rng() if rng is None else rng
            pos = pos[rng.permutation(len(pos))]
        X, y = collect_windows_for_positions(ds, pos, batch_rows=batch_rows, split_name=split_name)
        yield X, y, pos.astype(np.int64, copy=False)


def _train_caps(ds_train_list: list[Any], max_rows: int) -> list[int]:
    counts = [int(_decision_positions(len(ds)).shape[0]) for ds in ds_train_list]
    if int(max_rows) <= 0:
        return counts
    total = sum(counts); cap = min(int(max_rows), total)
    raw = np.asarray(counts, dtype=np.float64) * cap / max(1, total)
    caps = np.floor(raw).astype(np.int64)
    for idx in np.argsort(-(raw - caps))[: cap - int(caps.sum())]:
        caps[idx] += 1
    return [int(min(c, n)) for c, n in zip(caps, counts)]


def iter_train_window_batches(ds_train_list: list[Any], *, batch_rows: int, max_rows: int = 0, split_name: str = "train", shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for week_idx, (ds, cap) in enumerate(zip(ds_train_list, _train_caps(ds_train_list, int(max_rows)))):
        if cap <= 0:
            continue
        for X, y, positions in iter_dataset_window_batches(ds, batch_rows=batch_rows, max_rows=cap, split_name=f"{split_name}_week{week_idx}", shuffle_within_batch=shuffle_within_batch, rng=rng):
            yield X, y, (week_idx, positions)


def iter_extracted_batches_from_dataset(*, extractor: Any, ds: Any, batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for X, y, positions in iter_dataset_window_batches(ds, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        Z = extractor.transform(X).astype(np.float32, copy=False)
        assert_transform_matches_labels(Z, y, split_name)
        if not np.isfinite(Z).all():
            raise ValueError(f"Extractor produced non-finite values for split={split_name}")
        yield Z, y, positions


def iter_extracted_batches_from_train(*, extractor: Any, ds_train_list: list[Any], batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for X, y, pos_info in iter_train_window_batches(ds_train_list, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        Z = extractor.transform(X).astype(np.float32, copy=False)
        assert_transform_matches_labels(Z, y, split_name)
        if not np.isfinite(Z).all():
            raise ValueError(f"Extractor produced non-finite values for split={split_name}")
        yield Z, y, pos_info


def iter_preprocessed_batches_from_dataset(*, extractor: Any, bundle: LinearPreprocessBundle, ds: Any, batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for Z, y, positions in iter_extracted_batches_from_dataset(extractor=extractor, ds=ds, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        yield bundle.transform(Z), y, positions


def iter_preprocessed_batches_from_train(*, extractor: Any, bundle: LinearPreprocessBundle, ds_train_list: list[Any], batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for Z, y, pos_info in iter_extracted_batches_from_train(extractor=extractor, ds_train_list=ds_train_list, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        yield bundle.transform(Z), y, pos_info


def _fit_preprocess_from_stream(extractor: Any, ds_train_list: list[Any], config: Dict[str, Any], train_week_keys: list[str]) -> LinearPreprocessBundle:
    policy = str(config.get("nonfinite_policy", "raise"))
    parts = []
    rows = 0
    for Z, _y, _p in iter_extracted_batches_from_train(extractor=extractor, ds_train_list=ds_train_list, batch_rows=LINEAR_EXTRACT_BATCH_ROWS, max_rows=int(config["fit_max_rows"]), split_name="train_quantile_sample"):
        Z = _apply_preprocess_nonfinite_policy(Z, policy=policy, context="train quantile sample")
        parts.append(Z); rows += Z.shape[0]
        if estimate_matrix_mb(rows, Z.shape[1]) >= int(config["fit_max_matrix_mb"]):
            break
    if not parts:
        raise ValueError("Cannot fit preprocessor on empty streaming train sample")
    sample = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    D = int(sample.shape[1])
    cap_rows = max(1, int((int(config["fit_max_matrix_mb"]) * 1024 * 1024) // max(1, D * 4)))
    if sample.shape[0] > cap_rows:
        sample = sample[np.linspace(0, sample.shape[0] - 1, cap_rows, dtype=np.int64)]
    print(f"[linear-stream] stage=stage3 action=fit_quantile_sample rows={sample.shape[0]}", flush=True)
    winsorize = bool(config.get("winsorize", True))
    lower = np.quantile(sample, float(config["winsor_q_lo"]), axis=0).astype(np.float32) if winsorize else np.full(D, -np.inf, dtype=np.float32)
    upper = np.quantile(sample, float(config["winsor_q_hi"]), axis=0).astype(np.float32) if winsorize else np.full(D, np.inf, dtype=np.float32)
    count = 0; sum_ = np.zeros(D, dtype=np.float64); sumsq = np.zeros(D, dtype=np.float64)
    for Z, _y, _p in iter_extracted_batches_from_train(extractor=extractor, ds_train_list=ds_train_list, batch_rows=LINEAR_EXTRACT_BATCH_ROWS, max_rows=0, split_name="train_mean_std"):
        Z = _apply_preprocess_nonfinite_policy(Z, policy=policy, context="train mean/std")
        Zc = np.clip(Z, lower, upper)
        sum_ += Zc.sum(axis=0, dtype=np.float64); sumsq += np.square(Zc, dtype=np.float64).sum(axis=0); count += Zc.shape[0]
    if count <= 0:
        raise ValueError("Cannot fit preprocessor on empty streaming train split")
    print(f"[linear-stream] stage=stage3 action=fit_mean_std rows={count}", flush=True)
    mean64 = sum_ / count; std64 = np.sqrt(np.maximum(0.0, sumsq / count - mean64 * mean64))
    mean = mean64.astype(np.float32) if bool(config.get("standardize", True)) else np.zeros(D, dtype=np.float32)
    std = std64.astype(np.float32) if bool(config.get("standardize", True)) else np.ones(D, dtype=np.float32)
    keep = (np.isfinite(std) & (std >= float(config.get("min_std", 0.0)))) if bool(config.get("variance_filter", True)) else np.ones(D, dtype=bool)
    if not keep.any():
        raise ValueError("Preprocessor variance filter removed all features")
    fit_summary = {
        "fit_mode": "streaming_full_train_v1", "fit_split": "train", "quantile_fit_rows": int(sample.shape[0]),
        "mean_std_fit_rows": int(count), "train_weeks": list(train_week_keys), "extractor_output_dim": D,
        "original_dim": D, "kept_dim": int(keep.sum()), "removed_dim": int(D - keep.sum()),
        "fit_rows_for_quantiles": int(sample.shape[0]), "fit_rows_for_mean_std": int(count),
        "winsorize": winsorize, "winsor_q_lo": float(config["winsor_q_lo"]), "winsor_q_hi": float(config["winsor_q_hi"]),
        "standardize": bool(config.get("standardize", True)), "variance_filter": bool(config.get("variance_filter", True)),
        "min_std": float(config.get("min_std", 0.0)), "std_eps": float(config.get("std_eps", 1e-6)),
        "std_min": float(np.nanmin(std)), "std_p50": float(np.nanpercentile(std, 50)),
        "std_p95": float(np.nanpercentile(std, 95)), "std_max": float(np.nanmax(std)),
        "lower_finite_frac": float(np.isfinite(lower).mean()), "upper_finite_frac": float(np.isfinite(upper).mean()),
    }
    return LinearPreprocessBundle(str(config.get("schema", LINEAR_PREPROCESS_SCHEMA)), dict(config), D, int(keep.sum()), lower, upper, mean, std, keep.astype(bool), fit_summary)


def _audit_stream_split(extractor: Any, bundle: LinearPreprocessBundle, source: Any, split_name: str, *, is_train: bool, audit_path: Optional[Path]) -> Dict[str, Any]:
    stats = _empty_streaming_stats(); acc = new_preprocess_audit_accumulator(bundle.original_dim, bundle.kept_dim) if LINEAR_PREPROCESS_AUDIT else None
    iterator = iter_extracted_batches_from_train(extractor=extractor, ds_train_list=source, batch_rows=LINEAR_EXTRACT_BATCH_ROWS, max_rows=0, split_name=split_name) if is_train else iter_extracted_batches_from_dataset(extractor=extractor, ds=source, batch_rows=LINEAR_EXTRACT_BATCH_ROWS, max_rows=0, split_name=split_name)
    rows = chunks = 0
    for Z, _y, _p in iterator:
        if acc is not None:
            update_preprocess_audit(acc, Z_raw=Z, bundle=bundle, max_sample_values=LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE)
        Zp = bundle.transform(Z); _update_streaming_stats(stats, Zp); rows += Zp.shape[0]; chunks += 1
    summary = _finalize_streaming_summary(stats, n_shards=chunks, chunk_rows=LINEAR_EXTRACT_BATCH_ROWS, positions_rows=rows)
    audit = finalize_preprocess_audit(acc, bundle=bundle, split_name=split_name, top_k=LINEAR_PREPROCESS_AUDIT_TOP_K, max_sample_values=LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE) if acc is not None else None
    if audit is not None and audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({"stage": "stage3", "schema": "linear_preprocess_audit_v1", "split": split_name, "summary": jsonable_preprocess_audit_summary(audit, sample_values=LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES)}, allow_nan=True, indent=2), encoding="utf-8")
    print(f"[linear-stream] stage=stage3 action=audit split={split_name} rows={rows}", flush=True)
    return {"summary": summary, "audit_summary": compact_preprocess_audit_summary(audit), "_audit_full_summary": audit}


def run_stage2_extraction(*, linear_out_dir: Path, ds_train_list: list[Any], ds_val: Any, ds_test: Optional[Any], has_cmssl_test: bool, meta: Dict[str, Any], protocol: str, train_week_keys: list[str], extractor_config: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
    del ds_val, ds_test, has_cmssl_test
    name = str(extractor_config["extractor"]).strip().lower(); stage2_dir = Path(linear_out_dir) / "stage2_extractors" / name; stage2_dir.mkdir(parents=True, exist_ok=True)
    _print_decision_row_policy("stage2")
    extractor = build_linear_extractor_from_config(extractor_config)
    X_fit, _y_fit = collect_fit_windows_from_train(ds_train_list, max_rows=LINEAR_EXTRACTOR_FIT_MAX_ROWS, batch_rows=LINEAR_EXTRACT_BATCH_ROWS)
    t0 = time.time(); extractor.fit(X_fit); fit_seconds = time.time() - t0
    print(f"[linear-stream] stage=stage2 action=fit_extractor rows={X_fit.shape[0]}", flush=True)
    Zp = extractor.transform(X_fit[:min(128, len(X_fit))]).astype(np.float32, copy=False); D = int(Zp.shape[1])
    pkl_path = stage2_dir / "extractor.pkl"
    with pkl_path.open("wb") as f: pickle.dump(extractor, f)
    payload = {"stage": "stage2", "status": "ok", "streaming_features": True, "persisted_feature_shards": False, **_decision_metadata(),
        "linear_extractor_schema": LINEAR_EXTRACTOR_SCHEMA, "extractor_config": extractor_config, "extractor_summary": extractor.summary(),
        "extractor_pickle": str(pkl_path), "extractor_output_dim": D, "fit_rows": int(X_fit.shape[0]), "fit_split": "train_fit_sample",
        "fit_seconds": float(fit_seconds), "protocol": protocol, "train_week_keys": train_week_keys, "feature_dim_total": int(meta["feature_dim_total"]),
        "lookback": LOOKBACK, "horizons_ms": [int(h) for h in HORIZONS_MS], "manifests": {}, "stage2_dir": str(stage2_dir)}
    metrics_path = stage2_dir / "linear_stage2_extractor_metrics.json"; copy_path = Path(linear_out_dir) / "linear_stage2_extractor_metrics.json"
    metrics_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    print(f"[linear-stage2] wrote {metrics_path} and {copy_path}", flush=True); return payload


def run_stage3_preprocessing(*, linear_out_dir: Path, extractor_name: str, preprocess_name: str = "default") -> Dict[str, Any]:  # type: ignore[override]
    splits = build_linear_cmssl_splits_from_out_root(out_root=Path(OUT_ROOT)); extractor, stage2_payload = load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name=extractor_name)
    stage3_dir = resolve_stage3_dir(linear_out_dir, extractor_name, preprocess_name); stage3_dir.mkdir(parents=True, exist_ok=True); audit_dir = resolve_stage3_audit_dir(stage3_dir) if LINEAR_PREPROCESS_AUDIT else None
    cfg = {"schema": LINEAR_PREPROCESS_SCHEMA, "extractor": extractor_name, "preprocess_name": preprocess_name, "winsorize": LINEAR_PREPROCESS_WINSORIZE, "winsor_q_lo": LINEAR_PREPROCESS_WINSOR_Q_LO, "winsor_q_hi": LINEAR_PREPROCESS_WINSOR_Q_HI, "standardize": LINEAR_PREPROCESS_STANDARDIZE, "std_eps": LINEAR_PREPROCESS_STD_EPS, "variance_filter": LINEAR_PREPROCESS_VARIANCE_FILTER, "min_std": LINEAR_PREPROCESS_MIN_STD, "post_clip_abs": LINEAR_PREPROCESS_POST_CLIP_ABS, "nonfinite_policy": LINEAR_PREPROCESS_NONFINITE_POLICY, "fit_max_rows": LINEAR_PREPROCESS_FIT_MAX_ROWS, "fit_max_matrix_mb": LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB, **_decision_metadata()}
    bundle = _fit_preprocess_from_stream(extractor, splits["ds_train_list"], cfg, splits["train_week_keys"]); bundle_path = stage3_dir / "linear_preprocess_bundle.npz"; save_linear_preprocess_bundle(bundle, bundle_path)
    train_s = _audit_stream_split(extractor, bundle, splits["ds_train_list"], "train", is_train=True, audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_train.json")
    val_s = _audit_stream_split(extractor, bundle, splits["ds_val"], "val", is_train=False, audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_val.json")
    test_s = _audit_stream_split(extractor, bundle, splits["ds_test"], "test", is_train=False, audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_test.json") if splits["has_cmssl_test"] and splits["ds_test"] is not None and LINEAR_RUN_TEST else None
    audits = {"train": train_s.pop("_audit_full_summary", None), "val": val_s.pop("_audit_full_summary", None), "test": None if test_s is None else test_s.pop("_audit_full_summary", None)}
    combined = None; audit_summary_path = audit_csv_path = audit_top_path = None
    if LINEAR_PREPROCESS_AUDIT and audit_dir is not None:
        combined = {"stage": "stage3", "schema": "linear_preprocess_audit_v1", "extractor": extractor_name, "preprocess_name": preprocess_name, "bundle_path": str(bundle_path), "splits": {k: None if v is None else jsonable_preprocess_audit_summary(v, sample_values=LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES) for k, v in audits.items()}}
        audit_summary_path = audit_dir / "preprocess_audit_summary.json"; audit_csv_path = audit_dir / "preprocess_audit_summary.csv"; audit_top_path = audit_dir / "preprocess_audit_top_features.csv"
        audit_summary_path.write_text(json.dumps(combined, allow_nan=True, indent=2), encoding="utf-8"); write_preprocess_audit_csv(audit_csv_path, audits); write_preprocess_top_features_csv(audit_top_path, audits)
    payload = {"stage": "stage3", "status": "ok", "schema": LINEAR_PREPROCESS_SCHEMA, "streaming_features": True, "persisted_preprocessed_shards": False, **_decision_metadata(), "extractor": extractor_name, "stage2_payload_path": stage2_payload.get("payload_path"), "stage3_dir": str(stage3_dir), "preprocess_config": cfg, "preprocess_bundle_path": str(bundle_path), "fit_summary": bundle.fit_summary, "original_dim": int(bundle.original_dim), "kept_dim": int(bundle.kept_dim), "train_summary": train_s["summary"], "val_summary": val_s["summary"], "test_summary": None if test_s is None else test_s["summary"], "audit_enabled": bool(LINEAR_PREPROCESS_AUDIT), "audit_dir": None if audit_dir is None else str(audit_dir), "audit_summary_path": None if audit_summary_path is None else str(audit_summary_path), "audit_csv_path": None if audit_csv_path is None else str(audit_csv_path), "audit_top_features_csv_path": None if audit_top_path is None else str(audit_top_path), "audit_summary": combined, "manifests": {}}
    path = stage3_dir / "linear_stage3_preprocess_metrics.json"; copy = Path(linear_out_dir) / "linear_stage3_preprocess_metrics.json"; path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); return payload


class StreamingPreprocessedBatchSource:
    def __init__(self, *, extractor: Any, bundle: LinearPreprocessBundle, ds: Any, device: torch.device, batch_rows: int, max_rows: int, split_name: str):
        self.extractor = extractor; self.bundle = bundle; self.ds = ds; self.target_device = device; self.device = torch.device("cpu"); self.batch_size = int(batch_rows); self.positions = _dataset_positions(len(ds), int(max_rows)); self.n_rows = int(self.positions.shape[0]); self.effective_rows_nominal = self.n_rows; self.num_horizons = len(HORIZONS_MS); self.feature_shape = (self.n_rows, int(bundle.kept_dim)); self.is_shared_feature_view = False; self.pin_memory = bool(device.type == "cuda"); self.split_name = split_name
    def __len__(self): return (self.n_rows + self.batch_size - 1) // self.batch_size
    def iter_epoch(self, epoch: int):
        del epoch
        for start in range(0, self.n_rows, self.batch_size):
            pos = self.positions[start:start + self.batch_size]
            X, y = collect_windows_for_positions(self.ds, pos, batch_rows=self.batch_size, split_name=self.split_name)
            Z = self.bundle.transform(self.extractor.transform(X).astype(np.float32, copy=False))
            yield torch.as_tensor(Z, dtype=torch.float32, device=self.target_device), torch.as_tensor(y, dtype=torch.float32, device=self.target_device)


def initialize_stage4_candidate_bundle(*, alpha: float, config: Dict[str, Any]) -> LinearSklearnTakerBundle:
    n_h = len(HORIZONS_MS)
    return LinearSklearnTakerBundle(str(config["schema"]), dict(config, alpha=float(alpha)), [int(x) for x in HORIZONS_MS], [make_direction_model(alpha, config) for _ in range(n_h)], [make_magnitude_model(alpha, config) for _ in range(n_h)], [make_magnitude_model(alpha, config) for _ in range(n_h)], float(config["mag_floor"]), {})


def compute_global_direction_weights_from_train_labels_streaming(ds_train_list: list[Any], stats: Dict[str, np.ndarray], mode: str, batch_rows: int) -> list[tuple[float, float]]:
    mode = str(mode).lower()
    if mode == "none": return [(1.0, 1.0) for _ in HORIZONS_MS]
    pos = np.zeros(len(HORIZONS_MS)); neg = np.zeros(len(HORIZONS_MS))
    for ds in ds_train_list:
        positions = _dataset_positions(len(ds), 0)
        for start in range(0, len(positions), int(batch_rows)):
            y = np.asarray(ds.y[positions[start:start + int(batch_rows)]], dtype=np.float32)
            _kp, _kn, keep = build_signed_side_trim_masks_from_stats_np(y, stats)
            for h in range(len(HORIZONS_MS)):
                vals = y[keep[:, h], h] > 0.0; pos[h] += vals.sum(); neg[h] += (~vals).sum()
    out = []
    for p, n in zip(pos, neg):
        total = p + n
        if total <= 0 or p <= 0 or n <= 0: out.append((1.0, 1.0)); continue
        pf = p / total; nf = n / total
        out.append((0.5 / nf, 0.5 / pf) if mode == "balanced" else (math.sqrt(0.5 / nf), math.sqrt(0.5 / pf)))
    return out


def train_stage4_candidates_streaming(*, extractor: Any, preprocess_bundle: LinearPreprocessBundle, ds_train_list: list[Any], stats: Dict[str, np.ndarray], alpha_values: list[float], config: Dict[str, Any]) -> list[LinearSklearnTakerBundle]:
    bundles = [initialize_stage4_candidate_bundle(alpha=float(a), config=config) for a in alpha_values]; n_h = len(HORIZONS_MS)
    states = [{"df": [False]*n_h, "uf": [False]*n_h, "nf": [False]*n_h, "dc": np.zeros(n_h, dtype=np.int64), "uc": np.zeros(n_h, dtype=np.int64), "nc": np.zeros(n_h, dtype=np.int64)} for _ in bundles]
    weights = compute_global_direction_weights_from_train_labels_streaming(ds_train_list, stats, str(config["direction_weighting"]), int(config["batch_rows"]))
    train_rows = sum(int(_decision_positions(len(ds)).shape[0]) for ds in ds_train_list)
    print(f"[linear-stream] stage=stage4 action=train_full rows={train_rows} alpha_count={len(bundles)}", flush=True)
    for epoch in range(int(config["epochs"])):
        rng = np.random.default_rng(int(config["random_state"]) + epoch)
        for Z, y, _p in iter_preprocessed_batches_from_train(extractor=extractor, bundle=preprocess_bundle, ds_train_list=ds_train_list, batch_rows=int(config["batch_rows"]), max_rows=0, split_name="train_full", shuffle_within_batch=True, rng=rng):
            keep_pos, keep_neg, keep_signed = build_signed_side_trim_masks_from_stats_np(y, stats)
            for b, st in zip(bundles, states):
                for h in range(n_h):
                    rows = keep_signed[:, h]
                    if rows.any():
                        yd = (y[rows, h] > 0.0).astype(np.int64); neg_w, pos_w = weights[h]; sw = None if str(config["direction_weighting"]) == "none" else np.where(yd == 1, pos_w, neg_w).astype(np.float32)
                        if not st["df"][h]: b.direction_models[h].partial_fit(Z[rows], yd, classes=np.array([0, 1], dtype=np.int64), sample_weight=sw); st["df"][h] = True
                        else: b.direction_models[h].partial_fit(Z[rows], yd, sample_weight=sw)
                        st["dc"][h] += yd.shape[0]
                    rows = keep_pos[:, h]
                    if rows.any():
                        yu = np.sqrt(np.maximum(y[rows, h], 0.0)).astype(np.float32)
                        (b.mag_up_models[h].partial_fit(Z[rows], yu)); st["uf"][h] = True; st["uc"][h] += yu.shape[0]
                    rows = keep_neg[:, h]
                    if rows.any():
                        yn = np.sqrt(np.maximum(-y[rows, h], 0.0)).astype(np.float32)
                        (b.mag_down_models[h].partial_fit(Z[rows], yn)); st["nf"][h] = True; st["nc"][h] += yn.shape[0]
    for b, st in zip(bundles, states):
        if not all(st["df"]) or not all(st["uf"]) or not all(st["nf"]): raise ValueError("Insufficient train rows for one or more target/horizon models")
        b.fit_summary = {"alpha": float(b.config.get("alpha")), "train_rows": int(train_rows), "dir_rows_per_horizon": st["dc"].tolist(), "up_rows_per_horizon": st["uc"].tolist(), "down_rows_per_horizon": st["nc"].tolist(), "direction_weights_neg_pos": [(float(a), float(b)) for a, b in weights]}
    return bundles


def evaluate_stage4_bundle_streaming(*, bundle: LinearSklearnTakerBundle, extractor: Any, preprocess_bundle: LinearPreprocessBundle, ds: Any, stats: Dict[str, np.ndarray], device: torch.device, split_name: str, max_rows: int = 0, batch_rows: Optional[int] = None) -> Dict[str, Any]:
    source = StreamingPreprocessedBatchSource(extractor=extractor, bundle=preprocess_bundle, ds=ds, device=device, batch_rows=LINEAR_STAGE4_BATCH_ROWS if batch_rows is None else int(batch_rows), max_rows=max_rows, split_name=split_name)
    print(f"[linear-stream] stage=stage4 action=eval split={split_name} rows={source.n_rows}", flush=True)
    metrics = summarize_metrics(LinearSklearnTorchWrapper(bundle).to(device), source, device, stats, amp_enabled=False, amp_dtype=torch.float32, primary_only=False, epoch=0, band_diag=BAND_DIAG, split_name=split_name)
    pv, pl = compute_primary_metric(metrics); metrics["primary_metric_value"] = float(pv); metrics["primary_metric_label"] = str(pl); return metrics


def run_stage4_training(*, linear_out_dir: Path, extractor_name: str, preprocess_name: str, device: torch.device) -> Dict[str, Any]:  # type: ignore[override]
    if LINEAR_STAGE4_TRAIN_SPLIT != "train_full": raise ValueError("Stage 4 now supports only train_full streaming")
    splits = build_linear_cmssl_splits_from_out_root(out_root=Path(OUT_ROOT)); extractor, st2 = load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name=extractor_name); st3 = load_stage3_payload(linear_out_dir, extractor_name, preprocess_name); pb = load_linear_preprocess_bundle(Path(str(st3["preprocess_bundle_path"]))); stats = load_linear_trim_stats(linear_out_dir)
    cfg = {"schema": LINEAR_STAGE4_SCHEMA, "extractor": extractor_name, "preprocess_name": preprocess_name, "predictor": LINEAR_STAGE4_PREDICTOR, "penalty": LINEAR_STAGE4_PENALTY, "l1_ratio": LINEAR_STAGE4_L1_RATIO, "epochs": LINEAR_STAGE4_EPOCHS, "batch_rows": LINEAR_STAGE4_BATCH_ROWS, "random_state": LINEAR_STAGE4_RANDOM_SEED, "direction_weighting": LINEAR_STAGE4_DIRECTION_WEIGHTING, "mag_sample_weighting": LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING, "mag_floor": LINEAR_STAGE4_MAG_FLOOR, **_decision_metadata()}
    cands = train_stage4_candidates_streaming(extractor=extractor, preprocess_bundle=pb, ds_train_list=splits["ds_train_list"], stats=stats, alpha_values=[float(a) for a in LINEAR_STAGE4_ALPHA_VALUES], config=cfg)
    summaries=[]; best=None; best_metrics=None; best_score=float("-inf"); best_alpha=None
    for b in cands:
        vm = evaluate_stage4_bundle_streaming(bundle=b, extractor=extractor, preprocess_bundle=pb, ds=splits["ds_val"], stats=stats, device=device, split_name="val", max_rows=LINEAR_STAGE4_MAX_VAL_ROWS); pv, pl = compute_primary_metric(vm); score = float(pv) if bool(vm.get("primary_metric_guard_passed", True)) and math.isfinite(float(pv)) else float("-inf")
        summaries.append({"alpha": float(b.config.get("alpha")), "primary_metric_label": str(pl), "primary_metric_value": float(pv), "guard_passed": score > float("-inf"), "val_metrics": _jsonable_metrics(vm), "fit_summary": b.fit_summary})
        if best is None or is_metric_improved(score, best_score, "max"): best=b; best_metrics=vm; best_score=score; best_alpha=float(b.config.get("alpha"))
    stage4_dir = Path(linear_out_dir) / "stage4_models" / extractor_name / preprocess_name / LINEAR_STAGE4_PREDICTOR; stage4_dir.mkdir(parents=True, exist_ok=True); model_path = stage4_dir / "linear_stage4_best_model.pkl"; save_linear_sklearn_bundle(best, model_path)
    test_metrics = evaluate_stage4_bundle_streaming(bundle=best, extractor=extractor, preprocess_bundle=pb, ds=splits["ds_test"], stats=stats, device=device, split_name="test", max_rows=LINEAR_STAGE4_MAX_TEST_ROWS) if LINEAR_STAGE4_RUN_TEST and splits["has_cmssl_test"] and splits["ds_test"] is not None else None
    train_rows = sum(int(_decision_positions(len(ds)).shape[0]) for ds in splits["ds_train_list"])
    payload = {"stage": "stage4", "status": "ok", "schema": LINEAR_STAGE4_SCHEMA, "streaming_features": True, **_decision_metadata(), "stage4_config": cfg, "extractor": extractor_name, "preprocess_name": preprocess_name, "stage2_payload_path": st2.get("payload_path"), "stage3_payload_path": st3.get("payload_path"), "preprocess_bundle_path": str(st3["preprocess_bundle_path"]), "train_split": "train_full", "train_rows": int(train_rows), "val_rows": int(_dataset_positions(len(splits["ds_val"]), LINEAR_STAGE4_MAX_VAL_ROWS).shape[0]), "test_rows": None if splits["ds_test"] is None else int(_dataset_positions(len(splits["ds_test"]), LINEAR_STAGE4_MAX_TEST_ROWS).shape[0]), "original_dim": int(pb.original_dim), "kept_dim": int(pb.kept_dim), "best_alpha": float(best_alpha), "best_model_path": str(model_path), "best_primary_metric": {"label": str(best_metrics.get("primary_metric_label", PRIMARY_METRIC)), "value": float(best_metrics.get("primary_metric_value", best_score)), "guard_passed": bool(best_metrics.get("primary_metric_guard_passed", True))}, "candidate_summaries": summaries, "val_metrics": _jsonable_metrics(best_metrics), "test_metrics": _jsonable_metrics(test_metrics)}
    path = stage4_dir / "linear_stage4_metrics.json"; copy = Path(linear_out_dir) / "linear_stage4_metrics.json"; path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); return payload


def collect_predictions_and_labels_streaming(*, model_bundle: LinearSklearnTakerBundle, extractor: Any, preprocess_bundle: LinearPreprocessBundle, ds: Any, max_rows: int, batch_rows: int, split_name: str) -> Dict[str, np.ndarray]:
    parts = {k: [] for k in ["dir_logits", "p_up", "mag_up_sqrt", "mag_down_sqrt", "mag_up_bps", "mag_down_bps", "edge_bps", "y", "positions"]}
    for Z, y, pos in iter_preprocessed_batches_from_dataset(extractor=extractor, bundle=preprocess_bundle, ds=ds, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name):
        pred = model_bundle.predict_dict_np(Z); dl = np.asarray(pred["dir_logits"], dtype=np.float32); up = np.asarray(pred["mag_up_sqrt"], dtype=np.float32); dn = np.asarray(pred["mag_down_sqrt"], dtype=np.float32); p = _sigmoid_np(dl); ub = up * up; db = dn * dn; edge = p * ub - (1.0 - p) * db
        for k, v in [("dir_logits", dl), ("p_up", p), ("mag_up_sqrt", up), ("mag_down_sqrt", dn), ("mag_up_bps", ub), ("mag_down_bps", db), ("edge_bps", edge), ("y", y), ("positions", pos)]: parts[k].append(v.astype(np.int64 if k == "positions" else np.float32, copy=False))
    if not parts["y"]: raise ValueError(f"Streaming split contains no rows: {split_name}")
    print(f"[linear-stream] stage=stage5 action=diagnostics split={split_name} rows={sum(x.shape[0] for x in parts['y'])}", flush=True)
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def run_stage5_comparison(*, linear_out_dir: Path, extractor_names: list[str], preprocess_name: str, predictor: str, device: torch.device) -> Dict[str, Any]:  # type: ignore[override]
    stage5_dir = Path(linear_out_dir) / "stage5_comparison" / preprocess_name / predictor; diag_dir = stage5_dir / "stage5_diagnostics"; stage5_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; diagnostics={}; stats = load_linear_trim_stats(linear_out_dir); splits = build_linear_cmssl_splits_from_out_root(out_root=Path(OUT_ROOT)); _print_decision_row_policy("stage5")
    for extractor_name in extractor_names:
        st4 = load_stage4_artifacts_if_available(linear_out_dir, extractor_name, preprocess_name, predictor, strict=LINEAR_STAGE5_STRICT)
        if st4 is None: continue
        try:
            _validate_manifest_decision_policy(st4, context=f"stage5 stage4 {extractor_name}"); extractor, _st2 = load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name=extractor_name); st3 = load_stage3_payload(linear_out_dir, extractor_name, preprocess_name); _validate_manifest_decision_policy(st3, context=f"stage5 stage3 {extractor_name}")
        except (ValueError, FileNotFoundError) as exc:
            if LINEAR_STAGE5_STRICT: raise
            print(f"[linear-stage5-warn] {exc}; skipping extractor={extractor_name}", flush=True); continue
        model_bundle = load_linear_sklearn_bundle(Path(str(st4["best_model_path"]))); pb = load_linear_preprocess_bundle(Path(str(st3["preprocess_bundle_path"])))
        val_metrics = evaluate_stage4_bundle_streaming(bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=splits["ds_val"], stats=stats, device=device, split_name=f"stage5_val_{extractor_name}", max_rows=LINEAR_STAGE5_MAX_VAL_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS) if LINEAR_STAGE5_REEVALUATE else (st4.get("val_metrics", {}) or {})
        test_metrics = None
        if LINEAR_STAGE5_RUN_TEST and splits["has_cmssl_test"] and splits["ds_test"] is not None:
            test_metrics = evaluate_stage4_bundle_streaming(bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=splits["ds_test"], stats=stats, device=device, split_name=f"stage5_test_{extractor_name}", max_rows=LINEAR_STAGE5_MAX_TEST_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS) if LINEAR_STAGE5_REEVALUATE else st4.get("test_metrics")
        pred_val = collect_predictions_and_labels_streaming(model_bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=splits["ds_val"], max_rows=LINEAR_STAGE5_MAX_VAL_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS, split_name=f"val_{extractor_name}")
        pred_val_summary = summarize_prediction_arrays(pred_val); shift = run_label_shift_sanity_checks(pred_payload=pred_val, stats=stats, shifts=LINEAR_STAGE5_LABEL_SHIFT_VALUES, split_name=f"val_{extractor_name}") if LINEAR_STAGE5_LABEL_SHIFT_VALUES else {}; perm = run_label_permutation_sanity_check(pred_payload=pred_val, stats=stats, seed=LINEAR_STAGE5_PERMUTATION_SEED, split_name=f"val_{extractor_name}") if LINEAR_STAGE5_LABEL_PERMUTATION else None
        pred_test_summary=None; shift_t={}; perm_t=None
        if LINEAR_STAGE5_RUN_TEST and splits["has_cmssl_test"] and splits["ds_test"] is not None:
            pred_test = collect_predictions_and_labels_streaming(model_bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=splits["ds_test"], max_rows=LINEAR_STAGE5_MAX_TEST_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS, split_name=f"test_{extractor_name}"); pred_test_summary = summarize_prediction_arrays(pred_test); shift_t = run_label_shift_sanity_checks(pred_payload=pred_test, stats=stats, shifts=LINEAR_STAGE5_LABEL_SHIFT_VALUES, split_name=f"test_{extractor_name}") if LINEAR_STAGE5_LABEL_SHIFT_VALUES else {}; perm_t = run_label_permutation_sanity_check(pred_payload=pred_test, stats=stats, seed=LINEAR_STAGE5_PERMUTATION_SEED, split_name=f"test_{extractor_name}") if LINEAR_STAGE5_LABEL_PERMUTATION else None
            if LINEAR_STAGE5_SAVE_PREDICTIONS: save_stage5_prediction_dump(diag_dir / extractor_name / "test_predictions.npz", pred_test, LINEAR_STAGE5_PREDICTION_MAX_ROWS)
        if LINEAR_STAGE5_SAVE_PREDICTIONS: save_stage5_prediction_dump(diag_dir / extractor_name / "val_predictions.npz", pred_val, LINEAR_STAGE5_PREDICTION_MAX_ROWS)
        audit = load_stage3_audit_summary_for_stage5(st3); diag = {"stage4_payload_path": st4.get("payload_path"), "stage3_audit_summary": audit, "best_model_path": str(st4["best_model_path"]), "coefficient_diagnostics": summarize_linear_model_coefficients(model_bundle, top_k=LINEAR_STAGE5_TOP_COEFS), "prediction_summary_val": pred_val_summary, "prediction_summary_test": pred_test_summary, "label_shift_sanity_val": shift, "label_permutation_sanity_val": perm, "label_shift_sanity_test": shift_t, "label_permutation_sanity_test": perm_t, "val_metrics": _jsonable_metrics(val_metrics), "test_metrics": _jsonable_metrics(test_metrics)}
        diagnostics[extractor_name]=diag; row = build_stage5_comparison_row(extractor_name=extractor_name, preprocess_name=preprocess_name, predictor=predictor, stage4_payload=st4, val_metrics=val_metrics, test_metrics=test_metrics, diagnostics=diag); add_stage3_audit_fields_to_comparison_row(row, audit); row["decision_stride_rows"] = int(LINEAR_DECISION_STRIDE_ROWS); row["decision_offset_rows"] = int(LINEAR_DECISION_OFFSET_ROWS); rows.append(row)
        ddir = diag_dir / extractor_name; ddir.mkdir(parents=True, exist_ok=True); (ddir / f"diagnostics_{extractor_name}.json").write_text(json.dumps(diag, allow_nan=True, indent=2), encoding="utf-8")
    _maybe_add_baseline_row(rows, strict=LINEAR_STAGE5_STRICT)
    csv_path = stage5_dir / "linear_stage5_comparison.csv"; json_path = stage5_dir / "linear_stage5_comparison.json"; copy_csv = Path(linear_out_dir) / "linear_stage5_comparison.csv"; copy_json = Path(linear_out_dir) / "linear_stage5_comparison.json"; write_rows_csv(csv_path, rows); write_rows_csv(copy_csv, rows)
    payload = {"stage": "stage5", "status": "ok", "schema": LINEAR_STAGE5_SCHEMA, **_decision_metadata(), "linear_out_dir": str(linear_out_dir), "preprocess_name": preprocess_name, "predictor": predictor, "extractors_requested": extractor_names, "extractors_completed": [r["extractor"] for r in rows if r.get("extractor") != "CMSSL_neural_baseline"], "strict": bool(LINEAR_STAGE5_STRICT), "reevaluate": bool(LINEAR_STAGE5_REEVALUATE), "run_test": bool(LINEAR_STAGE5_RUN_TEST), "label_shifts": LINEAR_STAGE5_LABEL_SHIFT_VALUES, "label_permutation": bool(LINEAR_STAGE5_LABEL_PERMUTATION), "comparison_rows": rows, "diagnostics": diagnostics}
    json_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy_json.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); return payload

if __name__ == "__main__":
    if LINEAR_STAGE in {"stage3", "stage4", "stage5"}:
        assert OUT_ROOT or LINEAR_OUT_DIR, "Set BYBIT_OUT_ROOT or BYBIT_LINEAR_OUT_DIR"
    else:
        assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"
    main()
