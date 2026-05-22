#!/usr/bin/env python3
"""Linear offline entrypoint using CMSSL-compatible eval machinery."""

import csv
import gc
import json
import math
import os

if os.environ.get("BYBIT_LINEAR_SUPPRESS_CMSSL_CONFIG_PRINTS", "1").strip() == "1":
    os.environ.setdefault("BYBIT_SUPPRESS_CMSSL_CONFIG_PRINTS", "1")

import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from CMSSL17 import (  # type: ignore
    LOOKBACK, HORIZONS_MS,
    BATCH_SIZE,
    PRIMARY_METRIC, PRIMARY_METRIC_HORIZON_MS,
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
    load_stats_cache,
    cache_matches,
    save_stats_cache,
    summarize_metrics,
    BAND_DIAG,
    BAND_DIAG_QUANTILES,
)
from CMSSL17_linear import (  # type: ignore
    LINEAR_EXTRACTOR_SCHEMA,
    build_linear_extractor_from_config,
    LinearPreprocessBundle,
    LinearSklearnTakerBundle,
    LinearSklearnTorchWrapper,
    save_linear_preprocess_bundle,
    load_linear_preprocess_bundle,
    save_linear_sklearn_bundle,
    load_linear_sklearn_bundle,
    side_cond_log_mag_targets_np,
)



LINEAR_MEMORY_LOG = os.environ.get("BYBIT_LINEAR_MEMORY_LOG", "summary").strip().lower()
if LINEAR_MEMORY_LOG not in {"off", "summary", "dataset", "debug"}:
    raise ValueError("BYBIT_LINEAR_MEMORY_LOG must be one of: off, summary, dataset, debug")


def _should_log_memory(level: str) -> bool:
    mode = globals().get(
        "LINEAR_MEMORY_LOG",
        os.environ.get("BYBIT_LINEAR_MEMORY_LOG", "summary"),
    ).strip().lower()
    if mode == "off":
        return False
    if mode == "summary":
        return level == "summary"
    if mode == "dataset":
        return level in {"summary", "dataset"}
    if mode == "debug":
        return True
    return False


def log_memory(tag: str, *, level: str = "dataset") -> None:
    if not _should_log_memory(level):
        return
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[linear-memory] tag={tag} maxrss_mb={rss_kb / 1024.0:.1f}", flush=True)
    except Exception:
        pass


def force_gc(tag: str = "") -> None:
    gc.collect()
    if bool(getattr(torch.cuda, "is_available", lambda: False)()):
        torch.cuda.empty_cache()
    if tag and _should_log_memory("dataset"):
        print(f"[linear-memory] gc tag={tag}", flush=True)
        log_memory(f"after_gc_{tag}", level="dataset")


def close_dataset(ds: Any, *, name: str = "") -> None:
    try:
        close = getattr(ds, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        print(f"[linear-memory-warn] close failed for {name}: {exc}", flush=True)


def release_dataset(ds: Any, *, name: str = "") -> None:
    close_dataset(ds, name=name)
    del ds
    gc.collect()
    if bool(getattr(torch.cuda, "is_available", lambda: False)()):
        torch.cuda.empty_cache()
    if name and _should_log_memory("dataset"):
        print(f"[linear-memory] released dataset {name}", flush=True)
        log_memory(f"after_release_{name}", level="dataset")

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
LINEAR_DECISION_STRIDE_ROWS = _env_int("BYBIT_LINEAR_DECISION_STRIDE_ROWS", 5)
LINEAR_DECISION_OFFSET_ROWS = _env_int("BYBIT_LINEAR_DECISION_OFFSET_ROWS", 0)
LINEAR_PROGRESS = _env_bool("BYBIT_LINEAR_PROGRESS", 1)
LINEAR_PROGRESS_BACKEND = os.environ.get(
    "BYBIT_LINEAR_PROGRESS_BACKEND", "auto"
).strip().lower()
LINEAR_PROGRESS_EVERY_SEC = float(
    os.environ.get("BYBIT_LINEAR_PROGRESS_EVERY_SEC", "10")
)
DECISION_ROW_POLICY = "linear_every_n_rows_v1"

LINEAR_PREPROCESS_SCHEMA = "linear_preprocess_stage3_v1"
LINEAR_PREPROCESS_FIT_SPLIT = os.environ.get(
    "BYBIT_LINEAR_PREPROCESS_FIT_SPLIT", "train_full"
).strip().lower()
LINEAR_PREPROCESS_WINSORIZE = _env_bool("BYBIT_LINEAR_PREPROCESS_WINSORIZE", 1)
LINEAR_PREPROCESS_WINSOR_Q_LO = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_WINSOR_Q_LO", "0.001"))
LINEAR_PREPROCESS_WINSOR_Q_HI = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_WINSOR_Q_HI", "0.999"))
LINEAR_PREPROCESS_STANDARDIZE = _env_bool("BYBIT_LINEAR_PREPROCESS_STANDARDIZE", 1)
LINEAR_PREPROCESS_STD_EPS = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_STD_EPS", "1e-6"))
LINEAR_PREPROCESS_VARIANCE_FILTER = _env_bool("BYBIT_LINEAR_PREPROCESS_VARIANCE_FILTER", 1)
LINEAR_PREPROCESS_MIN_STD = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_MIN_STD", "1e-4"))
LINEAR_PREPROCESS_POST_CLIP_ABS = float(os.environ.get("BYBIT_LINEAR_PREPROCESS_POST_CLIP_ABS", "10.0"))
LINEAR_PREPROCESS_NONFINITE_POLICY = os.environ.get(
    "BYBIT_LINEAR_PREPROCESS_NONFINITE_POLICY", "raise"
).strip().lower()
LINEAR_PREPROCESS_FIT_MAX_ROWS = _env_int("BYBIT_LINEAR_PREPROCESS_FIT_MAX_ROWS", 50000)
LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB = _env_int("BYBIT_LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB", 2048)
LINEAR_PREPROCESS_AUDIT = _env_bool("BYBIT_LINEAR_PREPROCESS_AUDIT", 1)
LINEAR_PREPROCESS_AUDIT_TOP_K = _env_int("BYBIT_LINEAR_PREPROCESS_AUDIT_TOP_K", 50)
LINEAR_PREPROCESS_AUDIT_FULL_PER_FEATURE = _env_bool(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_FULL_PER_FEATURE", 0
)
LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES = _env_bool(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES", 0
)
LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE = _env_int(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE", 200_000
)
LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS = _env_int(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS", 200_000
)
LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS = _env_int(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS", 200_000
)
LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS = _env_int(
    "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS", 200_000
)

LINEAR_STAGE4_SCHEMA = "linear_target_models_stage4_v1"
LINEAR_STAGE4_PREPROCESS_NAME = os.environ.get("BYBIT_LINEAR_STAGE4_PREPROCESS_NAME", "default").strip()
LINEAR_STAGE4_TRAIN_SPLIT = os.environ.get("BYBIT_LINEAR_STAGE4_TRAIN_SPLIT", "train_full").strip().lower()
LINEAR_STAGE4_PREDICTOR = os.environ.get("BYBIT_LINEAR_STAGE4_PREDICTOR", "sgd_l2_huber").strip().lower()
DEFAULT_LINEAR_STAGE4_ALPHA_GRID = "1e-4,3e-4,1e-3,3e-3,1e-2"
DEFAULT_LINEAR_STAGE4_MAG_ALPHA_GRID = "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2"
LINEAR_STAGE4_ALPHA_GRID = os.environ.get(
    "BYBIT_LINEAR_STAGE4_ALPHA_GRID",
    DEFAULT_LINEAR_STAGE4_ALPHA_GRID,
).strip()
LINEAR_STAGE4_ALPHA_VALUES = _env_float_list(
    "BYBIT_LINEAR_STAGE4_ALPHA_GRID",
    DEFAULT_LINEAR_STAGE4_ALPHA_GRID,
)
LINEAR_STAGE4_MAG_ALPHA_GRID = os.environ.get(
    "BYBIT_LINEAR_STAGE4_MAG_ALPHA_GRID",
    DEFAULT_LINEAR_STAGE4_MAG_ALPHA_GRID,
).strip()
LINEAR_STAGE4_MAG_ALPHA_VALUES = _env_float_list(
    "BYBIT_LINEAR_STAGE4_MAG_ALPHA_GRID",
    DEFAULT_LINEAR_STAGE4_MAG_ALPHA_GRID,
)
LINEAR_STAGE4_MOVE_ALPHA_GRID = os.environ.get(
    "BYBIT_LINEAR_STAGE4_MOVE_ALPHA_GRID",
    LINEAR_STAGE4_ALPHA_GRID,
).strip()
LINEAR_STAGE4_MOVE_ALPHA_VALUES = _env_float_list(
    "BYBIT_LINEAR_STAGE4_MOVE_ALPHA_GRID",
    LINEAR_STAGE4_ALPHA_GRID,
)
LINEAR_STAGE4_MOVE_WEIGHTING = os.environ.get(
    "BYBIT_LINEAR_STAGE4_MOVE_WEIGHTING",
    "balanced",
).strip().lower()
LINEAR_STAGE4_PENALTY = os.environ.get("BYBIT_LINEAR_STAGE4_PENALTY", "l2").strip().lower()
LINEAR_STAGE4_L1_RATIO = float(os.environ.get("BYBIT_LINEAR_STAGE4_L1_RATIO", "0.15"))
LINEAR_STAGE4_EPOCHS = _env_int("BYBIT_LINEAR_STAGE4_EPOCHS", 3)
LINEAR_STAGE4_BATCH_ROWS = _env_int("BYBIT_LINEAR_STAGE4_BATCH_ROWS", 8192)
LINEAR_STAGE4_RANDOM_SEED = _env_int("BYBIT_LINEAR_STAGE4_RANDOM_SEED", 17)
LINEAR_STAGE4_DIRECTION_WEIGHTING = os.environ.get("BYBIT_LINEAR_STAGE4_DIRECTION_WEIGHTING", "tempered").strip().lower()
LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING = os.environ.get("BYBIT_LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING", "none").strip().lower()
LINEAR_STAGE4_RUN_TEST = _env_bool("BYBIT_LINEAR_STAGE4_RUN_TEST", 1)
LINEAR_STAGE4_MAG_FLOOR = float(os.environ.get("BYBIT_LINEAR_STAGE4_MAG_FLOOR", "1e-4"))
LINEAR_STAGE4_MAG_MODE = "side_cond_log"
LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE = os.environ.get("BYBIT_LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE", "train_median_nonzero_side").strip().lower()
LINEAR_STAGE4_MAG_LOG_SCALE_EPS = float(os.environ.get("BYBIT_LINEAR_STAGE4_MAG_LOG_SCALE_EPS", "1e-6"))
LINEAR_STAGE4_MAG_LOG_TARGET_CLIP = float(os.environ.get("BYBIT_LINEAR_STAGE4_MAG_LOG_TARGET_CLIP", "0.0"))
LINEAR_STAGE4_MAG_LOG_PRED_CLIP = float(os.environ.get("BYBIT_LINEAR_STAGE4_MAG_LOG_PRED_CLIP", "20.0"))
LINEAR_STAGE4_MAX_VAL_ROWS = _env_int("BYBIT_LINEAR_STAGE4_MAX_VAL_ROWS", 0)
LINEAR_STAGE4_MAX_TEST_ROWS = _env_int("BYBIT_LINEAR_STAGE4_MAX_TEST_ROWS", 0)
LINEAR_STAGE4_SAVE_VAL_PREDICTIONS = _env_bool("BYBIT_LINEAR_STAGE4_SAVE_VAL_PREDICTIONS", 0)

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
LINEAR_EXTRACT_BATCH_ROWS = _env_int("BYBIT_LINEAR_EXTRACT_BATCH_ROWS", 4096)
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

if LINEAR_PROGRESS_BACKEND not in {"auto", "tqdm", "log", "off"}:
    raise ValueError(
        "BYBIT_LINEAR_PROGRESS_BACKEND must be one of: auto, tqdm, log, off"
    )
if LINEAR_PROGRESS_EVERY_SEC <= 0:
    raise ValueError(
        f"BYBIT_LINEAR_PROGRESS_EVERY_SEC must be > 0, got {LINEAR_PROGRESS_EVERY_SEC}"
    )

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

if LINEAR_STAGE == "stage3":
    if LINEAR_PREPROCESS_FIT_SPLIT not in {"train_full"}:
        raise ValueError("Stage 3 now supports BYBIT_LINEAR_PREPROCESS_FIT_SPLIT=train_full only")
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
    if LINEAR_PREPROCESS_AUDIT_TOP_K < 0:
        raise ValueError(f"BYBIT_LINEAR_PREPROCESS_AUDIT_TOP_K must be >= 0, got {LINEAR_PREPROCESS_AUDIT_TOP_K}")
    if LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE <= 0:
        raise ValueError(
            "BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE must be > 0, "
            f"got {LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE}"
        )
    if LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS must be >= 0")
    if LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS must be >= 0")
    if LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS must be >= 0")

if LINEAR_STAGE == "stage4":
    if LINEAR_STAGE4_TRAIN_SPLIT not in {"train_full"}:
        raise ValueError("Stage 4 now supports only train_full streaming")
    if LINEAR_STAGE4_PREDICTOR not in {"sgd_l2_huber"}:
        raise ValueError("Stage 4 currently supports BYBIT_LINEAR_STAGE4_PREDICTOR=sgd_l2_huber")
    if LINEAR_STAGE4_PENALTY not in {"l2", "elasticnet"}:
        raise ValueError("BYBIT_LINEAR_STAGE4_PENALTY must be one of: l2, elasticnet")
    if not LINEAR_STAGE4_ALPHA_VALUES or any(a <= 0 for a in LINEAR_STAGE4_ALPHA_VALUES):
        raise ValueError("BYBIT_LINEAR_STAGE4_ALPHA_GRID must contain positive alpha values")
    if not LINEAR_STAGE4_MAG_ALPHA_VALUES or any(a <= 0 for a in LINEAR_STAGE4_MAG_ALPHA_VALUES):
        raise ValueError("BYBIT_LINEAR_STAGE4_MAG_ALPHA_GRID must contain positive alpha values")
    if not LINEAR_STAGE4_MOVE_ALPHA_VALUES or any(a <= 0 for a in LINEAR_STAGE4_MOVE_ALPHA_VALUES):
        raise ValueError("BYBIT_LINEAR_STAGE4_MOVE_ALPHA_GRID must contain positive alpha values")
    if LINEAR_STAGE4_MOVE_WEIGHTING not in {"none", "balanced", "tempered"}:
        raise ValueError("BYBIT_LINEAR_STAGE4_MOVE_WEIGHTING must be one of: none, balanced, tempered")
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
    if LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE not in {"train_median_nonzero_side", "train_q75_nonzero_side"}:
        raise ValueError("BYBIT_LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE must be one of: train_median_nonzero_side, train_q75_nonzero_side")
    if LINEAR_STAGE4_MAG_LOG_SCALE_EPS <= 0:
        raise ValueError("BYBIT_LINEAR_STAGE4_MAG_LOG_SCALE_EPS must be > 0")
    if LINEAR_STAGE4_MAG_LOG_TARGET_CLIP < 0:
        raise ValueError("BYBIT_LINEAR_STAGE4_MAG_LOG_TARGET_CLIP must be >= 0")


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


def validate_dataset_label_array_shape(ds: Any, split_name: str) -> None:
    if not hasattr(ds, "y"):
        raise ValueError(f"{split_name}: dataset missing y labels")

    y = np.asarray(ds.y)
    if y.ndim != 2:
        raise ValueError(f"{split_name}: ds.y must be 2D, got shape={y.shape}")

    expected = len(HORIZONS_MS)
    if y.shape[1] != expected:
        raise ValueError(
            f"{split_name}: label dimension mismatch: ds.y.shape[1]={y.shape[1]}, "
            f"expected {expected} horizons"
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


def summarize_linear_trim_stats(stats: Dict[str, np.ndarray]) -> Dict[str, Any]:
    keys = [
        "pos_lo_raw_bps",
        "pos_hi_raw_bps",
        "neg_lo_abs_bps",
        "neg_hi_abs_bps",
        "kept_pos_q50_abs_raw_bps",
        "kept_neg_q50_abs_raw_bps",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in stats:
            out[k] = np.asarray(stats[k]).astype(float).tolist()
    return out


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
        "rocket_channel_filter": int(os.environ.get("BYBIT_ROCKET_CHANNEL_FILTER", "1")),
        "rocket_channel_filter_std_eps": float(os.environ.get("BYBIT_ROCKET_CHANNEL_FILTER_STD_EPS", "1e-7")),
        "rocket_channel_filter_max_const_frac": float(os.environ.get("BYBIT_ROCKET_CHANNEL_FILTER_MAX_CONST_FRAC", "0.995")),
        "rocket_channel_filter_min_p95_std": float(os.environ.get("BYBIT_ROCKET_CHANNEL_FILTER_MIN_P95_STD", "1e-7")),
        "rocket_channel_filter_min_keep_channels": int(os.environ.get("BYBIT_ROCKET_CHANNEL_FILTER_MIN_KEEP_CHANNELS", "16")),
        "rocket_constant_fallback": int(os.environ.get("BYBIT_ROCKET_CONSTANT_FALLBACK", "1")),
        "rocket_constant_fallback_eps": float(os.environ.get("BYBIT_ROCKET_CONSTANT_FALLBACK_EPS", "1e-6")),
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


def decision_row_count(n_rows: int, max_rows: int = 0) -> int:
    n_rows = int(n_rows)
    max_rows = int(max_rows)

    if n_rows <= 0:
        return 0
    if LINEAR_DECISION_OFFSET_ROWS >= n_rows:
        return 0

    base = ((n_rows - 1 - LINEAR_DECISION_OFFSET_ROWS) // LINEAR_DECISION_STRIDE_ROWS) + 1

    if max_rows > 0:
        return min(int(base), max_rows)
    return int(base)


def describe_rows_for_split(obj: Any, *, max_rows: int = 0) -> int:
    return decision_row_count(len(obj), max_rows=max_rows)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def _progress_desc(stage: str, action: str, split: str = "", extra: str = "") -> str:
    parts = [str(stage), str(action)]
    if split:
        parts.append(str(split))
    if extra:
        parts.append(str(extra))
    return " ".join(parts)


def _progress_metadata() -> Dict[str, Any]:
    return {
        "progress_enabled": bool(LINEAR_PROGRESS and LINEAR_PROGRESS_BACKEND != "off"),
        "progress_backend": str(LINEAR_PROGRESS_BACKEND),
    }


def _default_row_getter(item) -> int:
    # Most streaming iterators yield (Z_or_X, y, pos).
    if isinstance(item, tuple) and len(item) >= 2:
        y = item[1]
        if hasattr(y, "shape"):
            return int(y.shape[0])
    return 0


def progress_iter_rows(
    iterable,
    *,
    total_rows: int,
    desc: str,
    row_getter=None,
):
    if row_getter is None:
        row_getter = _default_row_getter

    if (not LINEAR_PROGRESS) or LINEAR_PROGRESS_BACKEND == "off":
        yield from iterable
        return

    total_rows = int(total_rows)
    if total_rows < 0:
        total_rows = 0

    backend = LINEAR_PROGRESS_BACKEND
    use_tqdm = backend in {"auto", "tqdm"}

    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=total_rows if total_rows > 0 else None,
                desc=desc,
                unit="rows",
                dynamic_ncols=True,
                smoothing=0.05,
            )
        except Exception as exc:
            if backend == "tqdm":
                print(
                    f"[linear-progress-warn] tqdm unavailable/failed for {desc}: {exc}; "
                    "falling back to periodic logs",
                    flush=True,
                )
        else:
            try:
                for item in iterable:
                    rows = int(row_getter(item))
                    yield item
                    if rows > 0:
                        pbar.update(rows)
            finally:
                pbar.close()
            return

    start = time.time()
    last = start
    seen = 0

    print(
        f"[linear-progress] start {desc} total_rows={total_rows if total_rows > 0 else 'unknown'}",
        flush=True,
    )

    for item in iterable:
        rows = int(row_getter(item))
        yield item
        seen += max(0, rows)

        now = time.time()
        if now - last >= float(LINEAR_PROGRESS_EVERY_SEC):
            elapsed = now - start
            rate = seen / max(1e-9, elapsed)
            if total_rows > 0:
                remaining = max(0, total_rows - seen)
                eta = remaining / max(1e-9, rate)
                pct = 100.0 * seen / max(1, total_rows)
                print(
                    f"[linear-progress] {desc} rows={seen}/{total_rows} "
                    f"pct={pct:.1f}% rate={rate:.1f} rows/s "
                    f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}",
                    flush=True,
                )
            else:
                print(
                    f"[linear-progress] {desc} rows={seen} "
                    f"rate={rate:.1f} rows/s elapsed={_format_seconds(elapsed)}",
                    flush=True,
                )
            last = now

    elapsed = time.time() - start
    rate = seen / max(1e-9, elapsed)
    print(
        f"[linear-progress] done {desc} rows={seen}"
        + (f"/{total_rows}" if total_rows > 0 else "")
        + f" rate={rate:.1f} rows/s elapsed={_format_seconds(elapsed)}",
        flush=True,
    )


class LinearTimer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        print(f"[linear-timer] start {self.name}", flush=True)
        log_memory(f"{self.name}_start", level="summary")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        status = "error" if exc_type is not None else "done"
        print(
            f"[linear-timer] {status} {self.name} elapsed={_format_seconds(elapsed)}",
            flush=True,
        )
        log_memory(f"{self.name}_{status}", level="summary")
        return False


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

    def iter_epoch(self, epoch: int = 0):
        del epoch
        return iter(self)

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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def resolve_stage2_dir(linear_out_dir: Path, extractor_name: str) -> Path:
    return Path(linear_out_dir) / "stage2_extractors" / str(extractor_name)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

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


def resolve_stage3_dir(linear_out_dir: Path, extractor_name: str, preprocess_name: str) -> Path:
    return Path(linear_out_dir) / "stage3_preprocess" / str(extractor_name) / str(preprocess_name)


def load_linear_trim_stats(linear_out_dir: Path) -> Dict[str, np.ndarray]:
    path = Path(linear_out_dir) / "linear_signed_side_trim_stats_cache.npz"
    cached = load_stats_cache(path)
    if not cached:
        raise FileNotFoundError(
            f"Missing linear trim stats cache: {path}. Run stage1 first."
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


def make_move_model(alpha: float, config: dict):
    return make_direction_model(alpha=alpha, config=config)


def build_binary_sample_weights(y01: np.ndarray, mode: str) -> Optional[np.ndarray]:
    y01 = np.asarray(y01, dtype=np.int64).reshape(-1)
    mode = str(mode).strip().lower()
    if mode == "none":
        return None
    n = int(y01.shape[0])
    n_pos = max(1, int(y01.sum()))
    n_neg = max(1, int(n - n_pos))
    w_pos = 0.5 * n / n_pos
    w_neg = 0.5 * n / n_neg
    if mode == "tempered":
        w_pos = math.sqrt(w_pos)
        w_neg = math.sqrt(w_neg)
    elif mode != "balanced":
        raise ValueError(f"Unsupported binary weighting mode={mode!r}")
    return np.where(y01 == 1, w_pos, w_neg).astype(np.float32)


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


def _stable_sigmoid_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return _stable_sigmoid_np(x)


def _metric_at_primary_horizon(metrics: Dict[str, Any], key: str) -> float:
    vals = metrics.get(key)
    if not isinstance(vals, list):
        return float("nan")
    try:
        idx = list(map(int, HORIZONS_MS)).index(int(PRIMARY_METRIC_HORIZON_MS))
        return float(vals[idx])
    except Exception:
        return float("nan")


def _metric_at_primary_horizon_with_fallback(
    metrics: dict,
    primary_key: str,
    fallback_key: str | None = None,
) -> float:
    v = _metric_at_primary_horizon(metrics, primary_key)
    if np.isfinite(v):
        return float(v)
    if fallback_key is not None:
        return float(_metric_at_primary_horizon(metrics, fallback_key))
    return float(v)


def _mag_primary_metrics_1s(metrics: dict) -> dict[str, float]:
    return {
        "huber": _metric_at_primary_horizon_with_fallback(metrics, "mag_primary_huber", "mean_side_log_huber_cond"),
        "spearman": _metric_at_primary_horizon_with_fallback(metrics, "mag_primary_spearman", "mean_side_spearman_cond"),
        "p50_ratio": _metric_at_primary_horizon_with_fallback(metrics, "mag_primary_p50_ratio", "mean_side_p50_ratio_cond"),
        "p90_ratio": _metric_at_primary_horizon_with_fallback(metrics, "mag_primary_p90_ratio", "mean_side_p90_ratio_cond"),
        "lift": _metric_at_primary_horizon_with_fallback(metrics, "mag_primary_top_bottom_true_mean_lift", "mean_side_top_bottom_true_mean_lift_cond"),
        "zero_pred": _metric_at_primary_horizon_with_fallback(metrics, "zero_row_mean_pred_abs_bps", None),
        "edge_all": _metric_at_primary_horizon_with_fallback(metrics, "edge_spearman_all", None),
        "edge_kept": _metric_at_primary_horizon_with_fallback(metrics, "edge_spearman_kept", None),
    }


def _finite_or(value: float, fallback: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


def _mag_candidate_sort_key(metrics: dict, *, mag_mode: str) -> tuple[float, float, float, float]:
    _ = mag_mode
    m = _mag_primary_metrics_1s(metrics)
    huber = _finite_or(m["huber"], math.inf)
    sp = _finite_or(m["spearman"], -math.inf)
    zero = _finite_or(m["zero_pred"], math.inf)
    edge = _finite_or(m["edge_all"], -math.inf)
    return (-huber, sp, -zero, edge)


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




def _side_decile_calibration(pred_bps: np.ndarray, true_bps: np.ndarray) -> tuple[list[dict], float]:
    pred_bps = np.asarray(pred_bps, dtype=np.float32).reshape(-1)
    true_bps = np.asarray(true_bps, dtype=np.float32).reshape(-1)
    n = int(pred_bps.shape[0])
    if n <= 0:
        return [], float("nan")
    order = np.argsort(pred_bps)
    pred_sorted = pred_bps[order]
    true_sorted = true_bps[order]
    bins = np.array_split(np.arange(n), 10)
    out = []
    bottom_true_mean = top_true_mean = float("nan")
    for i, idx in enumerate(bins):
        if idx.size == 0:
            row = {"decile": int(i), "n": 0, "pred_mean_bps": float("nan"), "true_mean_bps": float("nan"), "true_p50_bps": float("nan"), "true_p90_bps": float("nan")}
        else:
            p = pred_sorted[idx]
            t = true_sorted[idx]
            row = {"decile": int(i), "n": int(idx.size), "pred_mean_bps": float(np.mean(p)), "true_mean_bps": float(np.mean(t)), "true_p50_bps": float(np.percentile(t, 50)), "true_p90_bps": float(np.percentile(t, 90))}
        if i == 0:
            bottom_true_mean = float(row["true_mean_bps"])
        if i == 9:
            top_true_mean = float(row["true_mean_bps"])
        out.append(row)
    lift = float(top_true_mean / max(bottom_true_mean, 1e-12)) if np.isfinite(top_true_mean) and np.isfinite(bottom_true_mean) else float("nan")
    return out, lift

def _safe_nanmean_values(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def add_side_cond_log_magnitude_metrics(metrics: Dict[str, Any], *, y: np.ndarray, pred: Dict[str, np.ndarray], scale_up_bps: np.ndarray, scale_down_bps: np.ndarray) -> None:
    y = np.asarray(y, dtype=np.float32)
    mag_up_bps, mag_down_bps = extract_mag_bps_from_prediction(pred)
    up_log_all = np.asarray(pred["mag_up_log"], dtype=np.float32)
    down_log_all = np.asarray(pred["mag_down_log"], dtype=np.float32)
    for h in range(y.shape[1]):
        up_rows = y[:, h] > 0.0
        down_rows = y[:, h] < 0.0
        zero_rows = y[:, h] == 0.0
        metrics.setdefault("up_n_cond", []).append(int(up_rows.sum()))
        metrics.setdefault("down_n_cond", []).append(int(down_rows.sum()))
        def side_vals(rows, side):
            if int(rows.sum()) < 2:
                return (float('nan'),)*6, [] , float('nan')
            if side=='up':
                true_bps=y[rows,h]; pred_bps=mag_up_bps[rows,h]; true_log=np.log1p(true_bps/max(float(scale_up_bps[h]),1e-12)); pred_log=up_log_all[rows,h]
            else:
                true_bps=-y[rows,h]; pred_bps=mag_down_bps[rows,h]; true_log=np.log1p(true_bps/max(float(scale_down_bps[h]),1e-12)); pred_log=down_log_all[rows,h]
            err=pred_log-true_log
            hub=float(np.mean(np.where(np.abs(err)<=1.0,0.5*err*err,np.abs(err)-0.5)))
            sp=float(_safe_spearman_np(pred_bps,true_bps))
            eps=1e-12
            mr=float(np.mean(pred_bps)/max(float(np.mean(true_bps)),eps))
            p50=float(np.percentile(pred_bps,50)/max(float(np.percentile(true_bps,50)),eps))
            p90=float(np.percentile(pred_bps,90)/max(float(np.percentile(true_bps,90)),eps))
            dec,lift=_side_decile_calibration(pred_bps,true_bps)
            return (hub,sp,mr,p50,p90,float(rows.sum())), dec, lift
        (up_h,up_sp,up_mr,up_p50,up_p90,_), up_dec, up_lift = side_vals(up_rows,'up')
        (dn_h,dn_sp,dn_mr,dn_p50,dn_p90,_), dn_dec, dn_lift = side_vals(down_rows,'down')
        for k,v in [("up_log_huber_cond",up_h),("down_log_huber_cond",dn_h),("up_spearman_cond",up_sp),("down_spearman_cond",dn_sp),("up_mean_ratio_cond",up_mr),("down_mean_ratio_cond",dn_mr),("up_p50_ratio_cond",up_p50),("down_p50_ratio_cond",dn_p50),("up_p90_ratio_cond",up_p90),("down_p90_ratio_cond",dn_p90),("up_top_bottom_true_mean_lift_cond",up_lift),("down_top_bottom_true_mean_lift_cond",dn_lift)]:
            metrics.setdefault(k,[]).append(v)
        metrics.setdefault("up_decile_calibration_cond",[]).append(up_dec)
        metrics.setdefault("down_decile_calibration_cond",[]).append(dn_dec)
        metrics.setdefault("mean_side_log_huber_cond",[]).append(_safe_nanmean_values([up_h,dn_h]))
        metrics.setdefault("mean_side_spearman_cond",[]).append(_safe_nanmean_values([up_sp,dn_sp]))
        metrics.setdefault("mean_side_mean_ratio_cond",[]).append(_safe_nanmean_values([up_mr,dn_mr]))
        metrics.setdefault("mean_side_p50_ratio_cond",[]).append(_safe_nanmean_values([up_p50,dn_p50]))
        metrics.setdefault("mean_side_p90_ratio_cond",[]).append(_safe_nanmean_values([up_p90,dn_p90]))
        metrics.setdefault("mean_side_top_bottom_true_mean_lift_cond",[]).append(_safe_nanmean_values([up_lift,dn_lift]))
        up_inactive = y[:, h] <= 0.0
        down_inactive = y[:, h] >= 0.0
        metrics.setdefault("up_inactive_pred_p90_bps",[]).append(float(np.percentile(mag_up_bps[up_inactive,h],90)) if np.any(up_inactive) else float('nan'))
        metrics.setdefault("down_inactive_pred_p90_bps",[]).append(float(np.percentile(mag_down_bps[down_inactive,h],90)) if np.any(down_inactive) else float('nan'))
        metrics.setdefault("zero_row_up_pred_p90_bps",[]).append(float(np.percentile(mag_up_bps[zero_rows,h],90)) if np.any(zero_rows) else float('nan'))
        metrics.setdefault("zero_row_down_pred_p90_bps",[]).append(float(np.percentile(mag_down_bps[zero_rows,h],90)) if np.any(zero_rows) else float('nan'))
        metrics.setdefault("zero_row_mean_pred_abs_bps",[]).append(float(np.mean(0.5*(mag_up_bps[zero_rows,h]+mag_down_bps[zero_rows,h]))) if np.any(zero_rows) else float('nan'))
        metrics.setdefault("mag_primary_huber", []).append(metrics["mean_side_log_huber_cond"][-1])
        metrics.setdefault("mag_primary_spearman", []).append(metrics["mean_side_spearman_cond"][-1])
        metrics.setdefault("mag_primary_p50_ratio", []).append(metrics["mean_side_p50_ratio_cond"][-1])
        metrics.setdefault("mag_primary_p90_ratio", []).append(metrics["mean_side_p90_ratio_cond"][-1])
        metrics.setdefault("mag_primary_top_bottom_true_mean_lift", []).append(metrics["mean_side_top_bottom_true_mean_lift_cond"][-1])

def build_move_target_from_stats_np(y: np.ndarray, stats: dict) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != len(HORIZONS_MS):
        raise ValueError(f"y must have shape [N, {len(HORIZONS_MS)}], got {y.shape}")
    pos_lo = np.asarray(stats["pos_lo_raw_bps"], dtype=np.float32).reshape(1, -1)
    neg_lo = np.asarray(stats["neg_lo_abs_bps"], dtype=np.float32).reshape(1, -1)
    if pos_lo.shape[1] != len(HORIZONS_MS) or neg_lo.shape[1] != len(HORIZONS_MS):
        raise ValueError("move target stats must have one low threshold per horizon")
    move = ((y > 0.0) & (y >= pos_lo)) | ((y < 0.0) & ((-y) >= neg_lo))
    return move.astype(np.int64, copy=False)


def add_move_head_metrics(metrics: dict, *, y: np.ndarray, pred: dict, stats: dict) -> None:
    move_target = build_move_target_from_stats_np(y, stats)
    _kp, _kn, keep_signed = build_signed_side_trim_masks_from_stats_np(y, stats)
    for h in range(y.shape[1]):
        target = move_target[:, h]
        p = np.asarray(pred["p_move"][:, h], dtype=np.float32)
        eps = 1e-6
        p_clip = np.clip(p, eps, 1.0 - eps)
        bce = -np.mean(target * np.log(p_clip) + (1 - target) * np.log(1 - p_clip))
        auc = _binary_auc_np(target, p)
        pred_label = p >= 0.5
        pos = target == 1; neg = target == 0
        tpr = np.mean(pred_label[pos] == 1) if pos.any() else math.nan
        tnr = np.mean(pred_label[neg] == 0) if neg.any() else math.nan
        bal = 0.5 * (tpr + tnr) if math.isfinite(tpr) and math.isfinite(tnr) else math.nan
        zero_rows = y[:, h] == 0.0; nonmove_rows = target == 0; move_rows = target == 1
        cond_edge = np.asarray(pred["cond_edge_bps"][:, h], dtype=np.float32)
        edge = np.asarray(pred["edge_bps"][:, h], dtype=np.float32)
        true = y[:, h]
        kept = keep_signed[:, h]
        for k,v in {
            "move_auc":auc,"move_bal_acc":bal,"move_bce":float(bce),"move_pos_frac_true":float(np.mean(target)),"move_pos_frac_pred_p50":float(np.mean(pred_label)),
            "move_prob_mean":float(np.mean(p)),"move_prob_p10":float(np.percentile(p,10)),"move_prob_p50":float(np.percentile(p,50)),"move_prob_p90":float(np.percentile(p,90)),
            "move_prob_mean_zero_rows":float(np.mean(p[zero_rows])) if zero_rows.any() else math.nan,
            "move_prob_p50_zero_rows":float(np.percentile(p[zero_rows],50)) if zero_rows.any() else math.nan,
            "move_prob_p90_zero_rows":float(np.percentile(p[zero_rows],90)) if zero_rows.any() else math.nan,
            "move_prob_mean_nonmove_rows":float(np.mean(p[nonmove_rows])) if nonmove_rows.any() else math.nan,
            "move_prob_p90_nonmove_rows":float(np.percentile(p[nonmove_rows],90)) if nonmove_rows.any() else math.nan,
            "move_prob_mean_move_rows":float(np.mean(p[move_rows])) if move_rows.any() else math.nan,
            "move_prob_p10_move_rows":float(np.percentile(p[move_rows],10)) if move_rows.any() else math.nan,
            "move_prob_p50_move_rows":float(np.percentile(p[move_rows],50)) if move_rows.any() else math.nan,
            "cond_edge_spearman_all":_safe_spearman_np(cond_edge,true),"cond_edge_spearman_kept":_safe_spearman_np(cond_edge[kept],true[kept]) if kept.any() else math.nan,
            "edge_spearman_all":_safe_spearman_np(edge,true),"edge_spearman_kept":_safe_spearman_np(edge[kept],true[kept]) if kept.any() else math.nan,
            "edge_bal_sign_acc_q50plus":_balanced_acc_bool(edge[kept] >= 0.0, true[kept] > 0.0) if kept.any() else math.nan,
        }.items():
            metrics.setdefault(k,[]).append(v)


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
        with LinearTimer("stage5"):
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
        with LinearTimer("stage3"):
            run_stage3_preprocessing(linear_out_dir=linear_out_dir, extractor_name=LINEAR_EXTRACTOR)
        return
    if LINEAR_STAGE == "stage4":
        print(
            f"[linear-config] stage=stage4 extractor={LINEAR_EXTRACTOR} "
            f"preprocess={LINEAR_STAGE4_PREPROCESS_NAME} predictor={LINEAR_STAGE4_PREDICTOR} "
            f"alpha_grid={[float(a) for a in LINEAR_STAGE4_ALPHA_VALUES]} "
            f"device={device} linear_out_dir={linear_out_dir}",
            flush=True,
        )
        with LinearTimer("stage4"):
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

    plan = load_linear_split_plan_from_out_root(out_root=out_root)
    meta = plan["meta"]
    protocol = plan["protocol"]
    train_week_keys = list(plan["train_week_keys"])
    train_split_entries = list(plan["train_split_entries"])

    if LINEAR_STAGE == "stage2":
        with LinearTimer("stage2"):
            run_stage2_extraction(
                linear_out_dir=linear_out_dir,
                plan=plan,
                extractor_config=extractor_config or _build_extractor_config(),
            )
        return

    _print_decision_row_policy(LINEAR_STAGE)
    y_train = collect_train_labels_from_plan(plan)

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

    metrics_payload = {
        "stage": "stage1",
        "status": "ok",
        "purpose": "linear_trim_stats_only",
        "out_root": str(out_root),
        "linear_out_dir": str(linear_out_dir),
        **_decision_metadata(),
        "protocol": protocol,
        "train_week_keys": train_week_keys,
        "val_weeks": plan["val_split_entry"].get("weeks"),
        "test_weeks": plan["test_split_entry"].get("weeks") if plan.get("has_cmssl_test") else None,
        "feature_dim_total": int(plan["feature_dim_total"]),
        "lookback": int(LOOKBACK),
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "target_task": TARGET_TASK,
        "target_transform": TARGET_TRANSFORM,
        "label_trim_schema": LABEL_TRIM_SCHEMA,
        "model_output_schema": MODEL_OUTPUT_SCHEMA,
        "trim_stats_cache_path": str(cache_path),
        "train_label_rows": int(y_train.shape[0]),
        "stats_summary": summarize_linear_trim_stats(stats),
    }
    metrics_path = linear_out_dir / "linear_stage1_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, allow_nan=True, indent=2)
    print(f"[linear-stage1] wrote stats metadata {metrics_path}", flush=True)
    return


def load_linear_split_plan_from_out_root(*, out_root: Path) -> Dict[str, Any]:
    out_root = Path(out_root)
    meta = json.loads((out_root / "meta.json").read_text())
    validate_contract_meta(meta, "global meta.json")
    validate_dataset_label_dim(meta, "global meta.json")
    split_info = require_supported_pipeline_splits(meta, out_root)
    protocol = split_info["protocol"]
    cmssl = split_info["splits"]["cmssl"]
    train_week_keys = list(cmssl["train"]["weeks"])
    val_split_entry = cmssl["val"]
    test_split_entry = cmssl.get("test")
    has_cmssl_test = test_split_entry is not None and bool(test_split_entry.get("weeks"))
    train_split_entries = [make_single_week_split_from_meta(out_root=out_root, global_meta=meta, week_key=wk) for wk in train_week_keys]
    print(f"[split] protocol={protocol} cmssl.train={','.join(train_week_keys)} cmssl.val={val_split_entry.get('weeks')} cmssl.test={test_split_entry.get('weeks') if has_cmssl_test else '<missing>'}", flush=True)
    return {"meta": meta, "out_root": out_root, "protocol": protocol, "train_week_keys": train_week_keys, "train_split_entries": train_split_entries, "val_split_entry": val_split_entry, "test_split_entry": test_split_entry, "has_cmssl_test": has_cmssl_test, "feature_dim_total": int(meta["feature_dim_total"])}


def _plan_meta_with_out_root(plan: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(plan["meta"])
    meta["out_root"] = str(plan.get("out_root", OUT_ROOT))
    return meta


def build_single_linear_dataset_from_entry(*, meta: Dict[str, Any], split_entry: Dict[str, Any], split_name: str) -> Any:
    out_root = Path(meta.get("out_root", OUT_ROOT))
    ds = build_dataset_from_split(str(out_root), split_entry)
    feature_dim_total = int(meta["feature_dim_total"])
    _validate_dataset_split(ds, split_name, feature_dim_total)
    validate_dataset_label_array_shape(ds, split_name)
    log_memory(f"after_build_{split_name}", level="dataset")
    return ds


def build_train_week_dataset(*, plan: Dict[str, Any], week_index: int) -> Any:
    return build_single_linear_dataset_from_entry(meta=_plan_meta_with_out_root(plan), split_entry=plan["train_split_entries"][week_index], split_name=f"train_week{week_index}_{plan['train_week_keys'][week_index]}")


def build_val_dataset_from_plan(plan: Dict[str, Any]) -> Any:
    return build_single_linear_dataset_from_entry(meta=_plan_meta_with_out_root(plan), split_entry=plan["val_split_entry"], split_name="val")


def build_test_dataset_from_plan(plan: Dict[str, Any]) -> Optional[Any]:
    if not plan.get("has_cmssl_test"):
        return None
    return build_single_linear_dataset_from_entry(meta=_plan_meta_with_out_root(plan), split_entry=plan["test_split_entry"], split_name="test")


def train_decision_row_count_from_plan(plan: Dict[str, Any], max_rows: int = 0) -> int:
    total = 0
    for i in range(len(plan["train_split_entries"])):
        ds = build_train_week_dataset(plan=plan, week_index=i)
        try:
            total += decision_row_count(len(ds), 0)
        finally:
            close_dataset(ds, name=f"count_train_week{i}")
            del ds
            force_gc(f"count_train_week{i}")
    return min(total, int(max_rows)) if int(max_rows) > 0 else int(total)


def split_decision_row_count_from_plan(plan: Dict[str, Any], split_name: str, max_rows: int = 0) -> int:
    ds = build_val_dataset_from_plan(plan) if split_name == "val" else build_test_dataset_from_plan(plan)
    if ds is None:
        return 0
    try:
        return decision_row_count(len(ds), max_rows=max_rows)
    finally:
        close_dataset(ds, name=f"count_{split_name}")
        del ds
        force_gc(f"count_{split_name}")


def collect_train_labels_from_plan(plan: Dict[str, Any]) -> np.ndarray:
    parts = []
    for i in range(len(plan["train_split_entries"])):
        ds = build_train_week_dataset(plan=plan, week_index=i)
        try:
            parts.append(collect_labels_from_dataset_positions(ds, max_rows=0, split_name=f"train_week{i}"))
        finally:
            close_dataset(ds, name=f"stage1_train_week{i}")
            del ds
            force_gc(f"stage1_train_week{i}")
    if not parts:
        raise ValueError("No train labels collected from split plan")
    y = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    validate_loaded_label_array(y, "linear train decision-stride labels")
    return y

# ---------------------------------------------------------------------------
# Active streaming-only pipeline definitions used by main().
# ---------------------------------------------------------------------------

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


def iter_train_week_window_batches_from_plan(*, plan: Dict[str, Any], batch_rows: int, max_rows: int = 0, split_name: str = "train", shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    n_weeks = len(plan["train_split_entries"])
    remaining = int(max_rows)
    for week_idx in range(n_weeks):
        ds = build_train_week_dataset(plan=plan, week_index=week_idx)
        tag = f"{split_name}_week{week_idx}"
        try:
            if int(max_rows) <= 0:
                rows_i = 0
            else:
                weeks_left = n_weeks - week_idx
                rows_i = int(math.ceil(remaining / max(1, weeks_left)))
                rows_i = min(rows_i, decision_row_count(len(ds), 0))
                remaining -= rows_i
            if int(max_rows) > 0 and rows_i <= 0:
                continue
            for X, y, pos in iter_dataset_window_batches(ds, batch_rows=batch_rows, max_rows=rows_i, split_name=tag, shuffle_within_batch=shuffle_within_batch, rng=rng):
                yield X, y, (week_idx, pos)
        finally:
            close_dataset(ds, name=tag)
            del ds
            force_gc(tag)


def iter_extracted_batches_from_train_plan(*, extractor: Any, plan: Dict[str, Any], batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for X, y, week_pos in iter_train_week_window_batches_from_plan(plan=plan, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        Z = extractor.transform(X).astype(np.float32, copy=False)
        assert_transform_matches_labels(Z, y, split_name)
        if not np.isfinite(Z).all():
            raise ValueError(f"Extractor produced non-finite values for split={split_name}")
        yield Z, y, week_pos


def iter_preprocessed_batches_from_train_plan(*, extractor: Any, bundle: LinearPreprocessBundle, plan: Dict[str, Any], batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for Z, y, week_pos in iter_extracted_batches_from_train_plan(extractor=extractor, plan=plan, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        yield bundle.transform(Z), y, week_pos


def collect_fit_windows_from_train_plan(*, plan: Dict[str, Any], max_rows: int, batch_rows: int, progress_stage: str = "stage2", progress_action: str = "fit_sample") -> tuple[np.ndarray, np.ndarray]:
    parts_x = []
    parts_y = []
    total = train_decision_row_count_from_plan(plan, max_rows=max_rows)
    iterator = iter_train_week_window_batches_from_plan(plan=plan, batch_rows=batch_rows, max_rows=max_rows, split_name="train_fit")
    for X, y, _pos in progress_iter_rows(iterator, total_rows=total, desc=_progress_desc(progress_stage, progress_action, "train")):
        parts_x.append(X.astype(np.float32, copy=False))
        parts_y.append(y.astype(np.float32, copy=False))
    if not parts_x:
        raise ValueError("Cannot fit extractor on empty streaming train sample")
    return np.concatenate(parts_x, axis=0), np.concatenate(parts_y, axis=0)

def iter_dataset_window_batches_with_progress(
    ds: Any,
    *,
    batch_rows: int,
    max_rows: int = 0,
    split_name: str,
    stage: str,
    action: str,
    shuffle_within_batch: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    total = decision_row_count(len(ds), max_rows=max_rows)
    base = iter_dataset_window_batches(
        ds,
        batch_rows=batch_rows,
        max_rows=max_rows,
        split_name=split_name,
        shuffle_within_batch=shuffle_within_batch,
        rng=rng,
    )
    return progress_iter_rows(
        base,
        total_rows=total,
        desc=_progress_desc(stage, action, split_name),
    )


def iter_extracted_batches_from_dataset(*, extractor: Any, ds: Any, batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for X, y, positions in iter_dataset_window_batches(ds, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        Z = extractor.transform(X).astype(np.float32, copy=False)
        assert_transform_matches_labels(Z, y, split_name)
        if not np.isfinite(Z).all():
            raise ValueError(f"Extractor produced non-finite values for split={split_name}")
        yield Z, y, positions


def iter_preprocessed_batches_from_dataset(*, extractor: Any, bundle: LinearPreprocessBundle, ds: Any, batch_rows: int, max_rows: int, split_name: str, shuffle_within_batch: bool = False, rng: Optional[np.random.Generator] = None):
    for Z, y, positions in iter_extracted_batches_from_dataset(extractor=extractor, ds=ds, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name, shuffle_within_batch=shuffle_within_batch, rng=rng):
        yield bundle.transform(Z), y, positions


def fit_linear_preprocessor_streaming_from_plan(*, extractor: Any, plan: Dict[str, Any], config: Dict[str, Any], batch_rows: int) -> LinearPreprocessBundle:
    policy = str(config.get("nonfinite_policy", "raise"))
    parts = []; rows = 0
    quantile_iter = iter_extracted_batches_from_train_plan(extractor=extractor, plan=plan, batch_rows=batch_rows, max_rows=int(config["fit_max_rows"]), split_name="train_quantile_sample")
    for Z, _y, _p in progress_iter_rows(quantile_iter, total_rows=train_decision_row_count_from_plan(plan, max_rows=int(config["fit_max_rows"])), desc="stage3 fit_quantile train"):
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
    q_rows = int(sample.shape[0]); del sample, parts; force_gc("stage3_after_quantile_sample")
    count = 0; sum_ = np.zeros(D, dtype=np.float64); sumsq = np.zeros(D, dtype=np.float64)
    mean_std_iter = iter_extracted_batches_from_train_plan(extractor=extractor, plan=plan, batch_rows=batch_rows, max_rows=0, split_name="train_mean_std")
    for Z, _y, _p in progress_iter_rows(mean_std_iter, total_rows=train_decision_row_count_from_plan(plan, max_rows=0), desc="stage3 fit_mean_std train"):
        Z = _apply_preprocess_nonfinite_policy(Z, policy=policy, context="train mean/std")
        Zc = np.clip(Z, lower, upper)
        sum_ += Zc.sum(axis=0, dtype=np.float64); sumsq += np.square(Zc, dtype=np.float64).sum(axis=0); count += Zc.shape[0]
    if count <= 0:
        raise ValueError("Cannot fit preprocessor on empty streaming train split")
    mean64 = sum_ / count; std64 = np.sqrt(np.maximum(0.0, sumsq / count - mean64 * mean64))
    mean = mean64.astype(np.float32) if bool(config.get("standardize", True)) else np.zeros(D, dtype=np.float32)
    std = std64.astype(np.float32) if bool(config.get("standardize", True)) else np.ones(D, dtype=np.float32)
    keep = (np.isfinite(std) & (std >= float(config.get("min_std", 0.0)))) if bool(config.get("variance_filter", True)) else np.ones(D, dtype=bool)
    if not keep.any():
        raise ValueError("Preprocessor variance filter removed all features")
    fit_summary = {"fit_mode": "streaming_full_train_v1", "fit_split": "train_full", "quantile_fit_rows": q_rows, "mean_std_fit_rows": int(count), "train_weeks": list(plan["train_week_keys"]), "extractor_output_dim": D, "original_dim": D, "kept_dim": int(keep.sum()), "removed_dim": int(D - keep.sum()), "fit_rows_for_quantiles": q_rows, "fit_rows_for_mean_std": int(count), "winsorize": winsorize, "winsor_q_lo": float(config["winsor_q_lo"]), "winsor_q_hi": float(config["winsor_q_hi"]), "standardize": bool(config.get("standardize", True)), "variance_filter": bool(config.get("variance_filter", True)), "min_std": float(config.get("min_std", 0.0)), "std_eps": float(config.get("std_eps", 1e-6)), "std_min": float(np.nanmin(std)), "std_p50": float(np.nanpercentile(std, 50)), "std_p95": float(np.nanpercentile(std, 95)), "std_max": float(np.nanmax(std)), "lower_finite_frac": float(np.isfinite(lower).mean()), "upper_finite_frac": float(np.isfinite(upper).mean())}
    return LinearPreprocessBundle(str(config.get("schema", LINEAR_PREPROCESS_SCHEMA)), dict(config), D, int(keep.sum()), lower, upper, mean, std, keep.astype(bool), fit_summary)


def audit_preprocessing_streaming_train_plan(*, extractor: Any, bundle: LinearPreprocessBundle, plan: Dict[str, Any], split_name: str, audit_path: Optional[Path], max_rows: int = 0) -> Dict[str, Any]:
    stats = _empty_streaming_stats(); acc = new_preprocess_audit_accumulator(bundle.original_dim, bundle.kept_dim) if LINEAR_PREPROCESS_AUDIT else None
    max_rows = int(max_rows)
    iterator = iter_extracted_batches_from_train_plan(extractor=extractor, plan=plan, batch_rows=LINEAR_EXTRACT_BATCH_ROWS, max_rows=max_rows, split_name=split_name)
    rows = chunks = 0
    total = train_decision_row_count_from_plan(plan, max_rows=max_rows)
    for Z, _y, _p in progress_iter_rows(iterator, total_rows=total, desc=_progress_desc("stage3", "audit", split_name)):
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

def _audit_stream_split(extractor: Any, bundle: LinearPreprocessBundle, source: Any, split_name: str, *, audit_path: Optional[Path], max_rows: int = 0) -> Dict[str, Any]:
    stats = _empty_streaming_stats(); acc = new_preprocess_audit_accumulator(bundle.original_dim, bundle.kept_dim) if LINEAR_PREPROCESS_AUDIT else None
    max_rows = int(max_rows)
    iterator = iter_extracted_batches_from_dataset(extractor=extractor, ds=source, batch_rows=LINEAR_EXTRACT_BATCH_ROWS, max_rows=max_rows, split_name=split_name)
    total_rows = describe_rows_for_split(source, max_rows=max_rows)
    rows = chunks = 0
    for Z, _y, _p in progress_iter_rows(iterator, total_rows=total_rows, desc=_progress_desc("stage3", "audit", split_name)):
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


def run_stage2_extraction(*, linear_out_dir: Path, plan: Dict[str, Any], extractor_config: Dict[str, Any], **_unused) -> Dict[str, Any]:  # type: ignore[override]
    name = str(extractor_config["extractor"]).strip().lower(); stage2_dir = Path(linear_out_dir) / "stage2_extractors" / name; stage2_dir.mkdir(parents=True, exist_ok=True)
    _print_decision_row_policy("stage2")
    extractor = build_linear_extractor_from_config(extractor_config)
    X_fit, y_fit = collect_fit_windows_from_train_plan(plan=plan, max_rows=LINEAR_EXTRACTOR_FIT_MAX_ROWS, batch_rows=LINEAR_EXTRACT_BATCH_ROWS)
    try:
        t0 = time.time(); extractor.fit(X_fit); fit_seconds = time.time() - t0
        print(f"[linear-stream] stage=stage2 action=fit_extractor rows={X_fit.shape[0]}", flush=True)
        Zp = extractor.transform(X_fit[:min(128, len(X_fit))]).astype(np.float32, copy=False); D = int(Zp.shape[1])
    finally:
        del y_fit; force_gc("stage2_after_fit_sample")
    pkl_path = stage2_dir / "extractor.pkl"
    with pkl_path.open("wb") as f: pickle.dump(extractor, f)
    fit_rows = int(X_fit.shape[0]); del X_fit; force_gc("stage2_after_save_extractor")
    payload = {"stage": "stage2", "status": "ok", "streaming_features": True, "persisted_feature_shards": False, **_decision_metadata(), **_progress_metadata(), "linear_extractor_schema": LINEAR_EXTRACTOR_SCHEMA, "extractor_config": extractor_config, "extractor_summary": extractor.summary(), "extractor_pickle": str(pkl_path), "extractor_output_dim": D, "fit_rows": fit_rows, "fit_split": "train_fit_sample", "fit_seconds": float(fit_seconds), "protocol": plan["protocol"], "train_week_keys": list(plan["train_week_keys"]), "feature_dim_total": int(plan["feature_dim_total"]), "lookback": LOOKBACK, "horizons_ms": [int(h) for h in HORIZONS_MS], "manifests": {}, "stage2_dir": str(stage2_dir)}
    metrics_path = stage2_dir / "linear_stage2_extractor_metrics.json"; copy_path = Path(linear_out_dir) / "linear_stage2_extractor_metrics.json"
    metrics_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    print(f"[linear-stage2] wrote {metrics_path} and {copy_path}", flush=True); return payload


def run_stage3_preprocessing(*, linear_out_dir: Path, extractor_name: str, preprocess_name: str = "default") -> Dict[str, Any]:  # type: ignore[override]
    if LINEAR_PREPROCESS_FIT_SPLIT != "train_full":
        raise ValueError("Stage 3 now supports BYBIT_LINEAR_PREPROCESS_FIT_SPLIT=train_full only")
    if LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS must be >= 0")
    if LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS must be >= 0")
    if LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS < 0:
        raise ValueError("BYBIT_LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS must be >= 0")
    plan = load_linear_split_plan_from_out_root(out_root=Path(OUT_ROOT))
    extractor, stage2_payload = load_stage2_extractor_bundle(
        linear_out_dir=linear_out_dir, extractor_name=extractor_name
    )
    stage3_dir = resolve_stage3_dir(linear_out_dir, extractor_name, preprocess_name)
    stage3_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = resolve_stage3_audit_dir(stage3_dir) if LINEAR_PREPROCESS_AUDIT else None
    cfg = {
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "extractor": extractor_name,
        "preprocess_name": preprocess_name,
        "fit_split": "train_full",
        "winsorize": LINEAR_PREPROCESS_WINSORIZE,
        "winsor_q_lo": LINEAR_PREPROCESS_WINSOR_Q_LO,
        "winsor_q_hi": LINEAR_PREPROCESS_WINSOR_Q_HI,
        "standardize": LINEAR_PREPROCESS_STANDARDIZE,
        "std_eps": LINEAR_PREPROCESS_STD_EPS,
        "variance_filter": LINEAR_PREPROCESS_VARIANCE_FILTER,
        "min_std": LINEAR_PREPROCESS_MIN_STD,
        "post_clip_abs": LINEAR_PREPROCESS_POST_CLIP_ABS,
        "nonfinite_policy": LINEAR_PREPROCESS_NONFINITE_POLICY,
        "fit_max_rows": LINEAR_PREPROCESS_FIT_MAX_ROWS,
        "fit_max_matrix_mb": LINEAR_PREPROCESS_FIT_MAX_MATRIX_MB,
        "audit_max_value_sample": int(LINEAR_PREPROCESS_AUDIT_MAX_VALUE_SAMPLE),
        "audit_max_train_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS),
        "audit_max_val_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS),
        "audit_max_test_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS),
        **_decision_metadata(),
    }
    bundle = fit_linear_preprocessor_streaming_from_plan(
        extractor=extractor, plan=plan, config=cfg, batch_rows=LINEAR_EXTRACT_BATCH_ROWS
    )
    bundle_path = stage3_dir / "linear_preprocess_bundle.npz"
    save_linear_preprocess_bundle(bundle, bundle_path)

    print(
        f"[linear-stage3-audit] split=train max_rows={LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS}",
        flush=True,
    )
    train_s = audit_preprocessing_streaming_train_plan(
        extractor=extractor,
        bundle=bundle,
        plan=plan,
        split_name="train",
        max_rows=LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS,
        audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_train.json",
    )

    ds_val = build_val_dataset_from_plan(plan)
    try:
        print(
            f"[linear-stage3-audit] split=val max_rows={LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS}",
            flush=True,
        )
        val_s = _audit_stream_split(
            extractor,
            bundle,
            ds_val,
            "val",
            max_rows=LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS,
            audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_val.json",
        )
    finally:
        close_dataset(ds_val, name="stage3_val_audit")
        del ds_val
        force_gc("stage3_val_audit")

    test_s = None
    if plan["has_cmssl_test"] and LINEAR_RUN_TEST:
        ds_test = build_test_dataset_from_plan(plan)
        if ds_test is not None:
            try:
                print(
                    f"[linear-stage3-audit] split=test max_rows={LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS}",
                    flush=True,
                )
                test_s = _audit_stream_split(
                    extractor,
                    bundle,
                    ds_test,
                    "test",
                    max_rows=LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS,
                    audit_path=None if audit_dir is None else audit_dir / "preprocess_audit_test.json",
                )
            finally:
                close_dataset(ds_test, name="stage3_test_audit")
                del ds_test
                force_gc("stage3_test_audit")

    audits = {
        "train": train_s.pop("_audit_full_summary", None),
        "val": val_s.pop("_audit_full_summary", None),
        "test": None if test_s is None else test_s.pop("_audit_full_summary", None),
    }
    combined = None
    audit_summary_path = audit_csv_path = audit_top_path = None
    if LINEAR_PREPROCESS_AUDIT and audit_dir is not None:
        combined = {
            "stage": "stage3",
            "schema": "linear_preprocess_audit_v1",
            "extractor": extractor_name,
            "preprocess_name": preprocess_name,
            "bundle_path": str(bundle_path),
            "audit_max_train_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS),
            "audit_max_val_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS),
            "audit_max_test_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS),
            "splits": {
                k: None
                if v is None
                else jsonable_preprocess_audit_summary(
                    v, sample_values=LINEAR_PREPROCESS_AUDIT_SAMPLE_VALUES
                )
                for k, v in audits.items()
            },
        }
        audit_summary_path = audit_dir / "preprocess_audit_summary.json"
        audit_csv_path = audit_dir / "preprocess_audit_summary.csv"
        audit_top_path = audit_dir / "preprocess_audit_top_features.csv"
        audit_summary_path.write_text(json.dumps(combined, allow_nan=True, indent=2), encoding="utf-8")
        write_preprocess_audit_csv(audit_csv_path, audits)
        write_preprocess_top_features_csv(audit_top_path, audits)

    payload = {
        "stage": "stage3",
        "status": "ok",
        "schema": LINEAR_PREPROCESS_SCHEMA,
        "streaming_features": True,
        "persisted_preprocessed_shards": False,
        **_decision_metadata(),
        **_progress_metadata(),
        "extractor": extractor_name,
        "stage2_payload_path": stage2_payload.get("payload_path"),
        "stage3_dir": str(stage3_dir),
        "preprocess_config": cfg,
        "preprocess_bundle_path": str(bundle_path),
        "fit_summary": bundle.fit_summary,
        "original_dim": int(bundle.original_dim),
        "kept_dim": int(bundle.kept_dim),
        "train_summary": train_s["summary"],
        "val_summary": val_s["summary"],
        "test_summary": None if test_s is None else test_s["summary"],
        "audit_enabled": bool(LINEAR_PREPROCESS_AUDIT),
        "audit_max_train_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_TRAIN_ROWS),
        "audit_max_val_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_VAL_ROWS),
        "audit_max_test_rows": int(LINEAR_PREPROCESS_AUDIT_MAX_TEST_ROWS),
        "audit_dir": None if audit_dir is None else str(audit_dir),
        "audit_summary_path": None if audit_summary_path is None else str(audit_summary_path),
        "audit_csv_path": None if audit_csv_path is None else str(audit_csv_path),
        "audit_top_features_csv_path": None if audit_top_path is None else str(audit_top_path),
        "audit_summary": combined,
        "manifests": {},
    }
    path = stage3_dir / "linear_stage3_preprocess_metrics.json"
    copy = Path(linear_out_dir) / "linear_stage3_preprocess_metrics.json"
    path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    copy.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8")
    return payload


class StreamingPreprocessedBatchSource:
    def __init__(self, *, extractor: Any, bundle: LinearPreprocessBundle, ds: Any, device: torch.device, batch_rows: int, max_rows: int, split_name: str, stage: str = "stage4", action: str = "eval"):
        self.extractor = extractor; self.bundle = bundle; self.ds = ds; self.target_device = device; self.device = torch.device("cpu"); self.batch_size = int(batch_rows); self.max_rows = int(max_rows); self.positions = _dataset_positions(len(ds), self.max_rows); self.n_rows = int(self.positions.shape[0]); self.total_rows = decision_row_count(len(ds), max_rows=self.max_rows); self.effective_rows_nominal = self.n_rows; self.num_horizons = len(HORIZONS_MS); self.feature_shape = (self.n_rows, int(bundle.kept_dim)); self.is_shared_feature_view = False; self.pin_memory = bool(device.type == "cuda"); self.split_name = split_name; self.stage = str(stage); self.action = str(action)
    def __len__(self): return (self.n_rows + self.batch_size - 1) // self.batch_size
    def iter_epoch(self, epoch: int):
        del epoch
        base_iter = iter_preprocessed_batches_from_dataset(extractor=self.extractor, bundle=self.bundle, ds=self.ds, batch_rows=self.batch_size, max_rows=self.max_rows, split_name=self.split_name)
        for Z, y, _pos in progress_iter_rows(base_iter, total_rows=self.total_rows, desc=_progress_desc(self.stage, self.action, self.split_name)):
            yield torch.as_tensor(Z, dtype=torch.float32, device=self.target_device), torch.as_tensor(y, dtype=torch.float32, device=self.target_device)


def initialize_stage4_candidate_bundle(*, alpha: float, config: Dict[str, Any]) -> LinearSklearnTakerBundle:
    n_h = len(HORIZONS_MS)
    return LinearSklearnTakerBundle(str(config["schema"]), dict(config, alpha=float(alpha)), [int(x) for x in HORIZONS_MS], [make_direction_model(alpha, config) for _ in range(n_h)], [make_magnitude_model(alpha, config) for _ in range(n_h)], [make_magnitude_model(alpha, config) for _ in range(n_h)], float(config["mag_floor"]), {}, str(config.get("mag_mode", "side_cond_log")), np.asarray(config.get("mag_up_scale_bps"), dtype=np.float32), np.asarray(config.get("mag_down_scale_bps"), dtype=np.float32))


def _build_bundle_from_parts(*, dir_alpha: float, mag_alpha: float, config: Dict[str, Any], direction_models: list[Any], mag_up_models: list[Any], mag_down_models: list[Any], fit_summary: Dict[str, Any]) -> LinearSklearnTakerBundle:
    return LinearSklearnTakerBundle(str(config["schema"]), dict(config, direction_alpha=float(dir_alpha), mag_alpha=float(mag_alpha), alpha=float(dir_alpha)), [int(x) for x in HORIZONS_MS], direction_models, mag_up_models, mag_down_models, float(config["mag_floor"]), dict(fit_summary), str(config.get("mag_mode", "side_cond_log")), np.asarray(config.get("mag_up_scale_bps"), dtype=np.float32), np.asarray(config.get("mag_down_scale_bps"), dtype=np.float32))


def extract_mag_bps_from_prediction(pred: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "mag_up_bps" in pred and "mag_down_bps" in pred:
        return np.asarray(pred["mag_up_bps"], dtype=np.float32), np.asarray(pred["mag_down_bps"], dtype=np.float32)
    if "mag_up_sqrt" not in pred or "mag_down_sqrt" not in pred:
        raise KeyError("prediction dict missing mag_up_bps/mag_down_bps and legacy mag_up_sqrt/mag_down_sqrt")
    up_sqrt = np.asarray(pred["mag_up_sqrt"], dtype=np.float32)
    down_sqrt = np.asarray(pred["mag_down_sqrt"], dtype=np.float32)
    return up_sqrt * up_sqrt, down_sqrt * down_sqrt


def compute_side_log_mag_scales_from_train_plan(*, plan: Dict[str, Any], source: str, eps: float, batch_rows: int) -> tuple[np.ndarray, np.ndarray]:
    n_h = len(HORIZONS_MS); up_parts = [[] for _ in range(n_h)]; down_parts = [[] for _ in range(n_h)]
    for week_idx in range(len(plan["train_split_entries"])):
        ds = build_train_week_dataset(plan=plan, week_index=week_idx); tag = f"stage4_mag_scales_week{week_idx}"
        try:
            positions = _dataset_positions(len(ds), 0)
            for start in range(0, len(positions), int(batch_rows)):
                y = np.asarray(ds.y[positions[start:start + int(batch_rows)]], dtype=np.float32)
                for h in range(n_h):
                    pos = y[:, h] > 0.0; neg = y[:, h] < 0.0
                    if pos.any(): up_parts[h].append(y[pos, h])
                    if neg.any(): down_parts[h].append(-y[neg, h])
        finally:
            close_dataset(ds, name=tag); del ds; force_gc(tag)
    up_scale = np.zeros(n_h, dtype=np.float32); down_scale = np.zeros(n_h, dtype=np.float32)
    for h in range(n_h):
        uv = np.concatenate(up_parts[h], axis=0) if up_parts[h] else np.zeros(0, dtype=np.float32)
        dv = np.concatenate(down_parts[h], axis=0) if down_parts[h] else np.zeros(0, dtype=np.float32)
        us = np.percentile(uv, 75) if source == "train_q75_nonzero_side" and uv.size else (np.median(uv) if uv.size else eps)
        ds = np.percentile(dv, 75) if source == "train_q75_nonzero_side" and dv.size else (np.median(dv) if dv.size else eps)
        up_scale[h] = max(float(us), float(eps)); down_scale[h] = max(float(ds), float(eps))
    print(f"[linear-mag-scale] mode=side_cond_log source={source} up_scale_bps={up_scale.tolist()} down_scale_bps={down_scale.tolist()}", flush=True)
    return up_scale, down_scale




def compute_global_direction_weights_from_train_labels_plan(*, plan: Dict[str, Any], stats: Dict[str, np.ndarray], mode: str, batch_rows: int) -> list[tuple[float, float]]:
    mode = str(mode).lower()
    if mode == "none": return [(1.0, 1.0) for _ in HORIZONS_MS]
    pos = np.zeros(len(HORIZONS_MS)); neg = np.zeros(len(HORIZONS_MS)); total_rows = train_decision_row_count_from_plan(plan, max_rows=0)
    def label_batches():
        for week_idx in range(len(plan["train_split_entries"])):
            ds = build_train_week_dataset(plan=plan, week_index=week_idx); tag = f"stage4_direction_weights_week{week_idx}"
            try:
                positions = _dataset_positions(len(ds), 0)
                for start in range(0, len(positions), int(batch_rows)):
                    batch_pos = positions[start:start + int(batch_rows)]
                    yield None, np.asarray(ds.y[batch_pos], dtype=np.float32), (week_idx, batch_pos)
            finally:
                close_dataset(ds, name=tag); del ds; force_gc(tag)
    for _x, y, _p in progress_iter_rows(label_batches(), total_rows=total_rows, desc="stage4 direction_weights train"):
        _kp, _kn, keep = build_signed_side_trim_masks_from_stats_np(y, stats)
        for h in range(len(HORIZONS_MS)):
            vals = y[keep[:, h], h] > 0.0; pos[h] += vals.sum(); neg[h] += (~vals).sum()
    out = []
    for p, n in zip(pos, neg):
        total = p + n
        if total <= 0 or p <= 0 or n <= 0: out.append((1.0, 1.0)); continue
        pf = p / total; nf = n / total; out.append((0.5 / nf, 0.5 / pf) if mode == "balanced" else (math.sqrt(0.5 / nf), math.sqrt(0.5 / pf)))
    return out


def train_direction_models_streaming_from_plan(**kwargs: Any) -> list[LinearSklearnTakerBundle]:
    extractor = kwargs["extractor"]; preprocess_bundle = kwargs["preprocess_bundle"]; plan = kwargs["plan"]; stats = kwargs["stats"]
    direction_alpha_values = kwargs["direction_alpha_values"]; config = kwargs["config"]; direction_weights = kwargs["direction_weights"]
    n_h = len(HORIZONS_MS)
    candidates = [{"direction_alpha": float(a), "direction_models": [make_direction_model(alpha=float(a), config=config) for _ in HORIZONS_MS], "df": [False] * n_h, "dc": np.zeros(n_h, dtype=np.int64)} for a in direction_alpha_values]
    train_rows = train_decision_row_count_from_plan(plan, max_rows=0)
    for epoch in range(int(config["epochs"])):
        rng = np.random.default_rng(int(config["random_state"]) + epoch)
        train_iter = iter_preprocessed_batches_from_train_plan(extractor=extractor, bundle=preprocess_bundle, plan=plan, batch_rows=int(config["batch_rows"]), max_rows=0, split_name="train_full", shuffle_within_batch=True, rng=rng)
        for Z, y, _p in progress_iter_rows(train_iter, total_rows=train_rows, desc=f"stage4 direction train epoch {epoch + 1}/{config['epochs']}"):
            _kp, _kn, keep_signed = build_signed_side_trim_masks_from_stats_np(y, stats)
            for c in candidates:
                for h in range(n_h):
                    rows = keep_signed[:, h]
                    if not rows.any(): continue
                    yd = (y[rows, h] > 0.0).astype(np.int64); neg_w, pos_w = direction_weights[h]
                    sw = None if str(config["direction_weighting"]) == "none" else np.where(yd == 1, pos_w, neg_w).astype(np.float32)
                    if not c["df"][h]: c["direction_models"][h].partial_fit(Z[rows], yd, classes=np.array([0, 1], dtype=np.int64), sample_weight=sw); c["df"][h] = True
                    else: c["direction_models"][h].partial_fit(Z[rows], yd, sample_weight=sw)
                    c["dc"][h] += yd.shape[0]
    out = []
    for c in candidates:
        if not all(c["df"]): raise ValueError("Insufficient train rows for one or more direction horizon models")
        out.append({"direction_alpha": float(c["direction_alpha"]), "direction_models": c["direction_models"], "fit_summary": {"direction_alpha": float(c["direction_alpha"]), "train_rows": int(train_rows), "dir_rows_per_horizon": c["dc"].tolist(), "direction_weights_neg_pos": [(float(a), float(b)) for a, b in direction_weights], "direction_training_rows": "kept_signed_rows"}})
    return out

def _train_side_cond_log_magnitude_models_streaming_from_plan(*, extractor: Any, preprocess_bundle: LinearPreprocessBundle, plan: Dict[str, Any], mag_alpha_values: list[float], config: Dict[str, Any]) -> list[dict]:
    n_h = len(HORIZONS_MS)
    cands = [{"mag_alpha": float(a), "mag_up_models": [make_magnitude_model(alpha=float(a), config=config) for _ in HORIZONS_MS], "mag_down_models": [make_magnitude_model(alpha=float(a), config=config) for _ in HORIZONS_MS], "uc": np.zeros(n_h, dtype=np.int64), "dc": np.zeros(n_h, dtype=np.int64)} for a in mag_alpha_values]
    train_rows = train_decision_row_count_from_plan(plan, max_rows=0)
    up_scale = np.asarray(config["mag_up_scale_bps"], dtype=np.float32); down_scale = np.asarray(config["mag_down_scale_bps"], dtype=np.float32)
    for epoch in range(int(config["epochs"])):
        rng = np.random.default_rng(int(config["random_state"]) + epoch)
        train_iter = iter_preprocessed_batches_from_train_plan(extractor=extractor, bundle=preprocess_bundle, plan=plan, batch_rows=int(config["batch_rows"]), max_rows=0, split_name="train_full", shuffle_within_batch=True, rng=rng)
        for Z, y, _p in progress_iter_rows(train_iter, total_rows=train_rows, desc=f"stage4 magnitude train epoch {epoch + 1}/{config['epochs']}"):
            up_log_targets, down_log_targets = side_cond_log_mag_targets_np(y, up_scale_bps=up_scale, down_scale_bps=down_scale, target_clip=float(config.get("mag_log_target_clip", 0.0)))
            for c in cands:
                for h in range(n_h):
                    up_rows = y[:, h] > 0.0; down_rows = y[:, h] < 0.0
                    if np.any(up_rows): c["mag_up_models"][h].partial_fit(Z[up_rows], up_log_targets[up_rows, h]); c["uc"][h] += int(np.sum(up_rows))
                    if np.any(down_rows): c["mag_down_models"][h].partial_fit(Z[down_rows], down_log_targets[down_rows, h]); c["dc"][h] += int(np.sum(down_rows))
    out = []
    for c in cands:
        out.append({"mag_alpha": float(c["mag_alpha"]), "mag_mode": "side_cond_log", "mag_up_models": c["mag_up_models"], "mag_down_models": c["mag_down_models"], "mag_abs_models": [], "fit_summary": {"mag_alpha": float(c["mag_alpha"]), "train_rows": int(train_rows), "up_rows_per_horizon": c["uc"].tolist(), "down_rows_per_horizon": c["dc"].tolist(), "mag_mode": "side_cond_log", "mag_training_rows": "side_active_rows", "mag_up_scale_bps": up_scale.tolist(), "mag_down_scale_bps": down_scale.tolist(), "mag_eval": "side_conditional_rows_only"}})
    return out

def train_magnitude_models_streaming_from_plan(*, extractor: Any, preprocess_bundle: LinearPreprocessBundle, plan: Dict[str, Any], mag_alpha_values: list[float], config: Dict[str, Any]) -> list[dict]:
    mag_mode = str(config.get("mag_mode", "side_cond_log")).strip().lower()
    if mag_mode == "side_cond_log":
        return _train_side_cond_log_magnitude_models_streaming_from_plan(extractor=extractor, preprocess_bundle=preprocess_bundle, plan=plan, mag_alpha_values=mag_alpha_values, config=config)
    raise ValueError(f"Unsupported mag_mode={mag_mode!r}; only side_cond_log is supported")

def train_move_models_streaming_from_plan(*, extractor, preprocess_bundle, plan, stats, move_alpha_values, config) -> list[dict]:
    n_h = len(HORIZONS_MS); train_rows = train_decision_row_count_from_plan(plan, max_rows=0)
    cands = [{"move_alpha": float(alpha), "move_models": [make_move_model(alpha=float(alpha), config=config) for _ in HORIZONS_MS], "fit": [False] * n_h, "rows": np.zeros(n_h, dtype=np.int64), "pos": np.zeros(n_h, dtype=np.int64), "neg": np.zeros(n_h, dtype=np.int64)} for alpha in move_alpha_values]
    for epoch in range(int(config["epochs"])):
        rng = np.random.default_rng(int(config["random_state"]) + 3000 + epoch)
        train_iter = iter_preprocessed_batches_from_train_plan(extractor=extractor, bundle=preprocess_bundle, plan=plan, batch_rows=int(config["batch_rows"]), max_rows=0, split_name="train_full", shuffle_within_batch=True, rng=rng)
        train_iter = progress_iter_rows(train_iter, total_rows=train_rows, desc=f"stage4 move train all_alphas={len(cands)} epoch {epoch + 1}/{config['epochs']}")
        for Z, y, _pos in train_iter:
            y_move = build_move_target_from_stats_np(y, stats)
            for cand in cands:
                for h in range(n_h):
                    target = y_move[:, h].astype(np.int64, copy=False)
                    if epoch == 0:
                        cand["rows"][h] += int(target.shape[0]); cand["pos"][h] += int(target.sum()); cand["neg"][h] += int(target.shape[0] - target.sum())
                    if target.min() == target.max():
                        continue
                    sw = build_binary_sample_weights(target, str(config.get("move_weighting", "balanced")))
                    if not cand["fit"][h]:
                        cand["move_models"][h].partial_fit(Z, target, classes=np.array([0, 1], dtype=np.int64), sample_weight=sw); cand["fit"][h] = True
                    else:
                        cand["move_models"][h].partial_fit(Z, target, sample_weight=sw)
    out = []
    for cand in cands:
        if not all(cand["fit"]):
            raise ValueError(f"Move model failed to fit all horizons for alpha={cand['move_alpha']}; fit={cand['fit']}")
        rows = cand["rows"].astype(np.float64); pos = cand["pos"].astype(np.float64)
        out.append({"move_alpha": float(cand["move_alpha"]), "move_models": cand["move_models"], "fit_summary": {"move_alpha": float(cand["move_alpha"]), "move_training_rows": "all_decision_rows", "move_target_schema": "abs_return_exceeds_side_low_threshold_v1", "move_rows_per_horizon": cand["rows"].astype(int).tolist(), "move_pos_rows_per_horizon": cand["pos"].astype(int).tolist(), "move_neg_rows_per_horizon": cand["neg"].astype(int).tolist(), "move_pos_frac_per_horizon": (np.divide(pos, np.maximum(rows, 1.0)).astype(float).tolist()), "move_weighting": str(config.get("move_weighting", "balanced"))}})
    return out

def make_identity_move_result() -> dict:
    return {"move_alpha": float("nan"), "move_models": None, "fit_summary": {"move_head_enabled": False, "move_alpha": float("nan"), "move_training_rows": "identity_p_move_1", "move_target_schema": "none_identity_fallback"}}

def build_stage4_bundle_from_parts(*, config: Dict[str, Any], horizons_ms: list[int], direction_result: Dict[str, Any], magnitude_result: Dict[str, Any], move_result: Dict[str, Any]) -> LinearSklearnTakerBundle:
    bundle_config = dict(config)
    fit_summary = {**direction_result.get("fit_summary", {}), **magnitude_result.get("fit_summary", {}), **move_result.get("fit_summary", {}), "direction_alpha": float(direction_result["direction_alpha"]), "mag_alpha": float(magnitude_result["mag_alpha"]), "move_alpha": float(move_result["move_alpha"])}
    return LinearSklearnTakerBundle(schema=LINEAR_STAGE4_SCHEMA, config=bundle_config, horizons_ms=[int(h) for h in horizons_ms], direction_models=direction_result["direction_models"], mag_up_models=magnitude_result["mag_up_models"], mag_down_models=magnitude_result["mag_down_models"], mag_floor=float(config["mag_floor"]), fit_summary=fit_summary, mag_mode="side_cond_log", mag_up_scale_bps=np.asarray(config["mag_up_scale_bps"], dtype=np.float32), mag_down_scale_bps=np.asarray(config["mag_down_scale_bps"], dtype=np.float32), move_models=move_result["move_models"])

def evaluate_stage4_bundle_streaming(*, bundle: LinearSklearnTakerBundle, extractor: Any, preprocess_bundle: LinearPreprocessBundle, ds: Any, stats: Dict[str, np.ndarray], device: torch.device, split_name: str, max_rows: int = 0, batch_rows: Optional[int] = None, include_cond_mag_metrics: bool = True) -> Dict[str, Any]:
    eval_stage = "stage5" if str(split_name).startswith("stage5_") else "stage4"
    eval_action = "reeval" if eval_stage == "stage5" else "eval"
    source = StreamingPreprocessedBatchSource(extractor=extractor, bundle=preprocess_bundle, ds=ds, device=device, batch_rows=LINEAR_STAGE4_BATCH_ROWS if batch_rows is None else int(batch_rows), max_rows=max_rows, split_name=split_name, stage=eval_stage, action=eval_action)
    print(f"[linear-stream] stage=stage4 action=eval split={split_name} rows={source.n_rows}", flush=True)
    metrics = summarize_metrics(LinearSklearnTorchWrapper(bundle, cmssl_schema_only=True).to(device), source, device, stats, amp_enabled=False, amp_dtype=torch.float32, primary_only=False, epoch=0, band_diag=BAND_DIAG, split_name=split_name)
    if include_cond_mag_metrics:
        pred_payload = collect_predictions_and_labels_streaming(model_bundle=bundle, extractor=extractor, preprocess_bundle=preprocess_bundle, ds=ds, max_rows=max_rows, batch_rows=LINEAR_STAGE4_BATCH_ROWS if batch_rows is None else int(batch_rows), split_name=f"{split_name}_all_rows_eval", progress_stage=eval_stage, progress_action="cond_mag_eval" if eval_stage == "stage4" else "diagnostics")
        add_side_cond_log_magnitude_metrics(metrics, y=np.asarray(pred_payload["y"], dtype=np.float32), pred=pred_payload, scale_up_bps=np.asarray(bundle.mag_up_scale_bps, dtype=np.float32), scale_down_bps=np.asarray(bundle.mag_down_scale_bps, dtype=np.float32))
        add_move_head_metrics(metrics, y=np.asarray(pred_payload["y"], dtype=np.float32), pred=pred_payload, stats=stats)
    pv, pl = compute_primary_metric(metrics); metrics["primary_metric_value"] = float(pv); metrics["primary_metric_label"] = str(pl); return metrics


def run_stage4_training(*, linear_out_dir: Path, extractor_name: str, preprocess_name: str, device: torch.device) -> Dict[str, Any]:  # type: ignore[override]
    if LINEAR_STAGE4_TRAIN_SPLIT != "train_full": raise ValueError("Stage 4 now supports only train_full streaming")
    plan = load_linear_split_plan_from_out_root(out_root=Path(OUT_ROOT)); extractor, st2 = load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name=extractor_name); st3 = load_stage3_payload(linear_out_dir, extractor_name, preprocess_name); _validate_manifest_decision_policy(st3, context=f"stage4 stage3 {extractor_name}"); pb = load_linear_preprocess_bundle(Path(str(st3["preprocess_bundle_path"]))); stats = load_linear_trim_stats(linear_out_dir)
    mag_mode = str(LINEAR_STAGE4_MAG_MODE).strip().lower()
    up_scale = down_scale = abs_scale = None
    if mag_mode == "side_cond_log":
        up_scale, down_scale = compute_side_log_mag_scales_from_train_plan(plan=plan, source=LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE, eps=LINEAR_STAGE4_MAG_LOG_SCALE_EPS, batch_rows=LINEAR_STAGE4_BATCH_ROWS)
    elif mag_mode == "abs_all_log":
        abs_scale = compute_abs_log_mag_scales_from_train_plan(plan=plan, source=LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE, eps=LINEAR_STAGE4_MAG_LOG_SCALE_EPS, batch_rows=LINEAR_STAGE4_BATCH_ROWS)
    else:
        raise ValueError(f"Unsupported mag_mode={mag_mode!r}; only side_cond_log is supported")
    cfg = {"schema": LINEAR_STAGE4_SCHEMA, "extractor": extractor_name, "preprocess_name": preprocess_name, "predictor": LINEAR_STAGE4_PREDICTOR, "penalty": LINEAR_STAGE4_PENALTY, "l1_ratio": LINEAR_STAGE4_L1_RATIO, "alpha_grid": [float(a) for a in LINEAR_STAGE4_ALPHA_VALUES], "direction_alpha_grid": [float(a) for a in LINEAR_STAGE4_ALPHA_VALUES], "mag_alpha_grid": [float(a) for a in LINEAR_STAGE4_MAG_ALPHA_VALUES], "epochs": LINEAR_STAGE4_EPOCHS, "batch_rows": LINEAR_STAGE4_BATCH_ROWS, "random_state": LINEAR_STAGE4_RANDOM_SEED, "direction_weighting": LINEAR_STAGE4_DIRECTION_WEIGHTING, "mag_sample_weighting": LINEAR_STAGE4_MAG_SAMPLE_WEIGHTING, "mag_floor": LINEAR_STAGE4_MAG_FLOOR, "mag_mode": mag_mode, "mag_log_scale_source": LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE, "mag_log_scale_eps": LINEAR_STAGE4_MAG_LOG_SCALE_EPS, "mag_log_target_clip": LINEAR_STAGE4_MAG_LOG_TARGET_CLIP, "mag_log_pred_clip": LINEAR_STAGE4_MAG_LOG_PRED_CLIP, **_decision_metadata()}
    if mag_mode == "side_cond_log":
        cfg.update({"mag_up_scale_bps": up_scale.astype(float).tolist(), "mag_down_scale_bps": down_scale.astype(float).tolist(), "mag_abs_scale_bps": None, "mag_training_rows": "side_active_rows", "mag_eval": "side_conditional_rows_only", "mag_target_transform": "log1p(side_bps / side_scale_bps)", "mag_inverse_transform": "side_scale_bps * expm1(pred_log)", "mag_prediction_units": "bps", "mag_model_output_units": "log1p_side_bps_scaled"})
    else:
        cfg.update({"mag_up_scale_bps": None, "mag_down_scale_bps": None, "mag_abs_scale_bps": abs_scale.astype(float).tolist(), "mag_training_rows": "all_decision_rows", "mag_eval": "all_rows_abs_return", "mag_target_transform": "log1p(abs_return_bps / abs_scale_bps)", "mag_inverse_transform": "abs_scale_bps * expm1(pred_log)", "mag_prediction_units": "abs_bps", "mag_model_output_units": "log1p_abs_bps_scaled"})
    direction_weights = compute_global_direction_weights_from_train_labels_plan(plan=plan, stats=stats, mode=str(cfg["direction_weighting"]), batch_rows=int(cfg["batch_rows"]))
    direction_results = train_direction_models_streaming_from_plan(extractor=extractor, preprocess_bundle=pb, plan=plan, stats=stats, direction_alpha_values=[float(a) for a in LINEAR_STAGE4_ALPHA_VALUES], config=cfg, direction_weights=direction_weights)
    reference_mag_alpha = 1e-3 if 1e-3 in LINEAR_STAGE4_MAG_ALPHA_VALUES else LINEAR_STAGE4_MAG_ALPHA_VALUES[len(LINEAR_STAGE4_MAG_ALPHA_VALUES)//2]
    reference_mag_result = train_magnitude_models_streaming_from_plan(extractor=extractor, preprocess_bundle=pb, plan=plan, mag_alpha_values=[float(reference_mag_alpha)], config=cfg)[0]
    identity_move_result = make_identity_move_result()
    cands = [build_stage4_bundle_from_parts(config=cfg, horizons_ms=[int(h) for h in HORIZONS_MS], direction_result=dr, magnitude_result=reference_mag_result, move_result=identity_move_result) for dr in direction_results]
    direction_summaries=[]; best=None; best_metrics=None; best_score=float("-inf"); best_alpha=None
    for b in cands:
        ds_val = build_val_dataset_from_plan(plan)
        try: vm = evaluate_stage4_bundle_streaming(bundle=b, extractor=extractor, preprocess_bundle=pb, ds=ds_val, stats=stats, device=device, split_name="val", max_rows=LINEAR_STAGE4_MAX_VAL_ROWS, include_cond_mag_metrics=False)
        finally: close_dataset(ds_val, name="stage4_val_eval"); del ds_val; force_gc("stage4_val_eval")
        pv, pl = compute_primary_metric(vm); score = float(pv) if bool(vm.get("primary_metric_guard_passed", True)) and math.isfinite(float(pv)) else float("-inf")
        guard_passed = score > float("-inf")
        auc_1s = _metric_at_primary_horizon(vm, "dir_auc_kept")
        bal_1s = _metric_at_primary_horizon(vm, "dir_bal_acc_kept")
        bce_1s = _metric_at_primary_horizon(vm, "val_dir_bce_kept")
        edge_sp_1s = _metric_at_primary_horizon(vm, "edge_spearman_kept")
        mag_sp_1s = _metric_at_primary_horizon(vm, "mean_side_spearman_cond")
        mag_ratio_1s = _metric_at_primary_horizon(vm, "mean_side_p90_ratio_cond")
        if not np.isfinite(mag_ratio_1s):
            mag_ratio_1s = _metric_at_primary_horizon(vm, "pred_abs_p90_over_true_abs_p90_kept")
        alpha = float(b.config.get("alpha"))
        print(
            f"[linear-stage4-dir-candidate] dir_alpha={alpha:g} "
            f"auc_1s={auc_1s:.6g} bal_1s={bal_1s:.6g} "
            f"bce_1s={bce_1s:.6g} "
            f"guard_passed={bool(guard_passed)}",
            flush=True,
        )
        direction_summaries.append({"direction_alpha": alpha, "primary_metric_label": str(pl), "primary_metric_value": float(pv), "guard_passed": guard_passed, "auc_1s": auc_1s, "bal_1s": bal_1s, "bce_1s": bce_1s})
        if best is None or is_metric_improved(score, best_score, "max"): best=b; best_metrics=vm; best_score=score; best_alpha=alpha
    if best is None or best_metrics is None: raise ValueError("No Stage 4 candidate models were trained")
    mag_alphas = [float(a) for a in LINEAR_STAGE4_MAG_ALPHA_VALUES]
    remaining_mag_alphas = [a for a in mag_alphas if not math.isclose(float(a), float(reference_mag_alpha), rel_tol=0.0, abs_tol=1e-18)]
    magnitude_results = [reference_mag_result]
    if remaining_mag_alphas:
        magnitude_results.extend(
            train_magnitude_models_streaming_from_plan(
                extractor=extractor,
                preprocess_bundle=pb,
                plan=plan,
                mag_alpha_values=remaining_mag_alphas,
                config=cfg,
            )
        )
    magnitude_results = sorted(magnitude_results, key=lambda r: float(r["mag_alpha"]))
    mag_candidates = magnitude_results
    magnitude_summaries = []
    best_mag_alpha = float(LINEAR_STAGE4_MAG_ALPHA_VALUES[0]); best_mag_score = None; best_mag_bundle = best
    best_direction_result = next(dr for dr in direction_results if float(dr["direction_alpha"]) == float(best_alpha))
    for mb in mag_candidates:
        combo = build_stage4_bundle_from_parts(config=cfg, horizons_ms=[int(h) for h in HORIZONS_MS], direction_result=best_direction_result, magnitude_result=mb, move_result=identity_move_result)
        ds_val = build_val_dataset_from_plan(plan)
        try: mvm = evaluate_stage4_bundle_streaming(bundle=combo, extractor=extractor, preprocess_bundle=pb, ds=ds_val, stats=stats, device=device, split_name="val", max_rows=LINEAR_STAGE4_MAX_VAL_ROWS)
        finally: close_dataset(ds_val, name="stage4_val_mag_eval"); del ds_val; force_gc("stage4_val_mag_eval")
        m1 = _mag_primary_metrics_1s(mvm)
        mag_log_huber_1s = m1["huber"]
        mag_sp_1s = m1["spearman"]
        mag_p50_ratio_1s = m1["p50_ratio"]
        mag_ratio_1s = m1["p90_ratio"]
        mag_lift_1s = m1["lift"]
        zero_pred_1s = m1["zero_pred"]
        edge_sp_1s = m1["edge_all"]
        mag_score = _mag_candidate_sort_key(mvm, mag_mode=mag_mode)
        mag_alpha = float(mb.get("mag_alpha"))
        print(f"[linear-stage4-mag-candidate] mag_mode={mag_mode} mag_alpha={mag_alpha:g} log_huber_1s={mag_log_huber_1s:.6g} sp_1s={mag_sp_1s:.6g} p50_ratio_1s={mag_p50_ratio_1s:.6g} p90_ratio_1s={mag_ratio_1s:.6g} lift_1s={mag_lift_1s:.6g} zero_pred_1s={zero_pred_1s:.6g} edge_sp_1s={edge_sp_1s:.6g} selection_score={mag_score}", flush=True)
        magnitude_summaries.append({"mag_alpha": mag_alpha, "mag_mode": mag_mode, "mag_log_huber_1s": float(mag_log_huber_1s), "mag_sp_1s": float(mag_sp_1s), "mag_p50_ratio_1s": float(mag_p50_ratio_1s), "mag_p90_ratio_1s": float(mag_ratio_1s), "mag_lift_1s": float(mag_lift_1s), "zero_pred_1s": float(zero_pred_1s), "edge_sp_1s": float(edge_sp_1s), "selection_score": [float(x) for x in mag_score]})
        if best_mag_score is None or mag_score > best_mag_score: best_mag_score = mag_score; best_mag_alpha = mag_alpha; best_mag_bundle = combo; best_metrics = mvm
    best = best_mag_bundle
    best.fit_summary = {
        "direction_alpha": float(best_alpha),
        "mag_alpha": float(best_mag_alpha),
        "direction_fit_summary": best_direction_result["fit_summary"],
        "magnitude_fit_summary": next(mr for mr in magnitude_results if float(mr["mag_alpha"]) == float(best_mag_alpha))["fit_summary"],
        "selection": {"direction_primary_metric_label": str(best_metrics.get("primary_metric_label", PRIMARY_METRIC)), "direction_primary_metric_value": float(best_score), "magnitude_selection_score_name": "mag_primary_tuple_1s", "magnitude_selection_score": [float(x) for x in best_mag_score], "magnitude_selection_neg_huber_1s": float(best_mag_score[0]), "magnitude_selection_spearman_1s": float(best_mag_score[1]), "magnitude_selection_neg_zero_pred_1s": float(best_mag_score[2]), "magnitude_selection_edge_spearman_1s": float(best_mag_score[3])},
    }
    best_summary = next((s for s in direction_summaries if float(s.get("direction_alpha", float("nan"))) == float(best_alpha)), {})
    best_val_metrics = _jsonable_metrics(best_metrics)
    best_auc_1s = _metric_at_primary_horizon(best_val_metrics, "dir_auc_kept")
    best_bal_1s = _metric_at_primary_horizon(best_val_metrics, "dir_bal_acc_kept")
    best_bce_1s = _metric_at_primary_horizon(best_val_metrics, "val_dir_bce_kept")
    best_edge_sp_1s = _metric_at_primary_horizon(best_val_metrics, "edge_spearman_kept")
    best_m1 = _mag_primary_metrics_1s(best_val_metrics)
    best_mag_log_huber_1s = best_m1["huber"]
    best_mag_sp_1s = best_m1["spearman"]
    best_mag_p50_ratio_1s = best_m1["p50_ratio"]
    best_mag_ratio_1s = best_m1["p90_ratio"]
    best_mag_lift_1s = best_m1["lift"]
    best_zero_pred_1s = best_m1["zero_pred"]
    best_edge_sp_all_1s = best_m1["edge_all"]
    print(
        f"[linear-stage4-best] mag_mode={mag_mode} dir_alpha={float(best_alpha):g} mag_alpha={float(best_mag_alpha):g} "
        f"auc_1s={best_auc_1s:.6g} bal_1s={best_bal_1s:.6g} "
        f"mag_huber_1s={best_mag_log_huber_1s:.6g} "
        f"mag_sp_1s={best_mag_sp_1s:.6g} "
        f"mag_p50_ratio_1s={best_mag_p50_ratio_1s:.6g} "
        f"mag_p90_ratio_1s={best_mag_ratio_1s:.6g} "
        f"mag_lift_1s={best_mag_lift_1s:.6g} "
        f"zero_pred_1s={best_zero_pred_1s:.6g} "
        f"edge_sp_1s={best_edge_sp_all_1s:.6g}",
        flush=True,
    )
    stage4_dir = Path(linear_out_dir) / "stage4_models" / extractor_name / preprocess_name / LINEAR_STAGE4_PREDICTOR; stage4_dir.mkdir(parents=True, exist_ok=True); model_path = stage4_dir / "linear_stage4_best_model.pkl"; save_linear_sklearn_bundle(best, model_path)
    test_metrics = None
    if LINEAR_STAGE4_RUN_TEST and plan["has_cmssl_test"]:
        ds_test = build_test_dataset_from_plan(plan)
        if ds_test is not None:
            try: test_metrics = evaluate_stage4_bundle_streaming(bundle=best, extractor=extractor, preprocess_bundle=pb, ds=ds_test, stats=stats, device=device, split_name="test", max_rows=LINEAR_STAGE4_MAX_TEST_ROWS)
            finally: close_dataset(ds_test, name="stage4_test_eval"); del ds_test; force_gc("stage4_test_eval")
    train_rows = train_decision_row_count_from_plan(plan, max_rows=0); val_rows = split_decision_row_count_from_plan(plan, "val", LINEAR_STAGE4_MAX_VAL_ROWS); test_rows = split_decision_row_count_from_plan(plan, "test", LINEAR_STAGE4_MAX_TEST_ROWS) if plan["has_cmssl_test"] and LINEAR_STAGE4_RUN_TEST else None
    m1 = _mag_primary_metrics_1s(best_val_metrics)
    payload = {"stage": "stage4", "status": "ok", "schema": LINEAR_STAGE4_SCHEMA, "streaming_features": True, **_decision_metadata(), **_progress_metadata(), "stage4_config": cfg, "extractor": extractor_name, "preprocess_name": preprocess_name, "stage2_payload_path": st2.get("payload_path"), "stage3_payload_path": st3.get("payload_path"), "preprocess_bundle_path": str(st3["preprocess_bundle_path"]), "train_split": "train_full", "train_rows": int(train_rows), "val_rows": int(val_rows), "test_rows": test_rows, "original_dim": int(pb.original_dim), "kept_dim": int(pb.kept_dim), "mag_mode": mag_mode, "mag_target_transform": cfg["mag_target_transform"], "mag_inverse_transform": cfg["mag_inverse_transform"], "mag_training_rows": cfg["mag_training_rows"], "mag_eval": cfg["mag_eval"], "mag_prediction_units": cfg["mag_prediction_units"], "mag_model_output_units": cfg["mag_model_output_units"], "mag_up_scale_bps": cfg.get("mag_up_scale_bps"), "mag_down_scale_bps": cfg.get("mag_down_scale_bps"), "mag_log_scale_source": LINEAR_STAGE4_MAG_LOG_SCALE_SOURCE, "mag_log_scale_eps": LINEAR_STAGE4_MAG_LOG_SCALE_EPS, "mag_log_target_clip": LINEAR_STAGE4_MAG_LOG_TARGET_CLIP, "mag_log_pred_clip": LINEAR_STAGE4_MAG_LOG_PRED_CLIP, "best_alpha": float(best_alpha), "best_direction_alpha": float(best_alpha), "best_mag_alpha": float(best_mag_alpha), "best_model_path": str(model_path), "best_primary_metric": {"label": str(best_metrics.get("primary_metric_label", PRIMARY_METRIC)), "value": float(best_metrics.get("primary_metric_value", best_score)), "guard_passed": bool(best_metrics.get("primary_metric_guard_passed", True))}, "candidate_summaries": direction_summaries, "direction_candidate_summaries": direction_summaries, "magnitude_candidate_summaries": magnitude_summaries, "stage4_summary_metrics": {"primary_horizon_ms": int(PRIMARY_METRIC_HORIZON_MS), "direction": {"auc_kept": best_auc_1s, "bal_acc_kept": best_bal_1s, "bce_kept": best_bce_1s, "pos_frac_pred_q50plus": _metric_at_primary_horizon(best_val_metrics, "dir_pos_frac_pred_q50plus"), "pos_frac_true_q50plus": _metric_at_primary_horizon(best_val_metrics, "dir_pos_frac_true_q50plus")}, "magnitude": {"mode": mag_mode, "log_huber_1s": float(m1["huber"]), "spearman_1s": float(m1["spearman"]), "p50_ratio_1s": float(m1["p50_ratio"]), "p90_ratio_1s": float(m1["p90_ratio"]), "top_bottom_true_mean_lift_1s": float(m1["lift"]), "zero_row_mean_pred_abs_bps_1s": float(m1["zero_pred"])}, "edge": {"spearman_all": _metric_at_primary_horizon(best_val_metrics, "edge_spearman_all"), "spearman_kept": _metric_at_primary_horizon(best_val_metrics, "edge_spearman_kept"), "sign_bal_acc_q50plus": _metric_at_primary_horizon(best_val_metrics, "edge_bal_sign_acc_q50plus")}}, "val_metrics": _jsonable_metrics(best_metrics), "test_metrics": _jsonable_metrics(test_metrics)}
    path = stage4_dir / "linear_stage4_metrics.json"; copy = Path(linear_out_dir) / "linear_stage4_metrics.json"; path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); return payload


def collect_predictions_and_labels_streaming(*, model_bundle: LinearSklearnTakerBundle, extractor: Any, preprocess_bundle: LinearPreprocessBundle, ds: Any, max_rows: int, batch_rows: int, split_name: str, progress_stage: str = "stage4", progress_action: str = "diagnostics") -> Dict[str, np.ndarray]:
    required_keys = ["dir_logits", "p_up", "move_logits", "p_move", "mag_up_sqrt", "mag_down_sqrt", "mag_up_log", "mag_down_log", "mag_up_bps", "mag_down_bps", "cond_edge_bps", "edge_bps", "y", "positions"]
    parts = {k: [] for k in required_keys}
    base_iter = iter_preprocessed_batches_from_dataset(extractor=extractor, bundle=preprocess_bundle, ds=ds, batch_rows=batch_rows, max_rows=max_rows, split_name=split_name)
    for Z, y, pos in progress_iter_rows(base_iter, total_rows=decision_row_count(len(ds), max_rows=max_rows), desc=_progress_desc(progress_stage, progress_action, split_name)):
        pred = model_bundle.predict_dict_np(Z); dl = np.asarray(pred["dir_logits"], dtype=np.float32); p = _sigmoid_np(dl)
        ub, db = extract_mag_bps_from_prediction(pred)
        up = np.sqrt(np.maximum(ub, 0.0)); dn = np.sqrt(np.maximum(db, 0.0))
        pm = np.asarray(pred["p_move"], dtype=np.float32)
        ml = np.asarray(pred["move_logits"], dtype=np.float32)
        cond_edge = p * ub - (1.0 - p) * db
        edge = pm * cond_edge
        vals=[("dir_logits", dl), ("p_up", p), ("move_logits", ml), ("p_move", pm), ("mag_up_sqrt", up), ("mag_down_sqrt", dn), ("mag_up_log", np.asarray(pred["mag_up_log"], dtype=np.float32)), ("mag_down_log", np.asarray(pred["mag_down_log"], dtype=np.float32)), ("mag_up_bps", ub), ("mag_down_bps", db), ("cond_edge_bps", cond_edge), ("edge_bps", edge), ("y", y), ("positions", pos)]
        for k, v in vals: parts[k].append(v.astype(np.int64 if k == "positions" else np.float32, copy=False))
    if not parts["y"]: raise ValueError(f"Streaming split contains no rows: {split_name}")
    missing = [k for k, v in parts.items() if not v]
    if missing:
        raise ValueError(f"collect_predictions_and_labels_streaming missing required prediction arrays for split={split_name!r}: {missing}")
    print(f"[linear-stream] stage={progress_stage} action={progress_action} split={split_name} rows={sum(x.shape[0] for x in parts['y'])}", flush=True)
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def run_stage5_comparison(*, linear_out_dir: Path, extractor_names: list[str], preprocess_name: str, predictor: str, device: torch.device) -> Dict[str, Any]:  # type: ignore[override]
    rows=[]; diagnostics={}; stats = load_linear_trim_stats(linear_out_dir); plan = load_linear_split_plan_from_out_root(out_root=Path(OUT_ROOT)); _print_decision_row_policy("stage5")
    stage5_dir = Path(linear_out_dir) / "stage5_comparison" / preprocess_name / predictor; diag_dir = stage5_dir / "diagnostics"; diag_dir.mkdir(parents=True, exist_ok=True)
    for extractor_name in extractor_names:
        try:
            st4 = load_stage4_payload(linear_out_dir=linear_out_dir, extractor_name=extractor_name, preprocess_name=preprocess_name, predictor=predictor); _validate_manifest_decision_policy(st4, context=f"stage5 stage4 {extractor_name}")
            extractor, st2 = load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name=extractor_name); st3 = load_stage3_payload(linear_out_dir, extractor_name, preprocess_name); _validate_manifest_decision_policy(st3, context=f"stage5 stage3 {extractor_name}")
        except (ValueError, FileNotFoundError) as exc:
            if LINEAR_STAGE5_STRICT: raise
            print(f"[linear-stage5-warn] {exc}; skipping extractor={extractor_name}", flush=True); continue
        model_bundle = load_linear_sklearn_bundle(Path(str(st4["best_model_path"]))); pb = load_linear_preprocess_bundle(Path(str(st3["preprocess_bundle_path"])))
        ds_val = build_val_dataset_from_plan(plan)
        try:
            val_metrics = evaluate_stage4_bundle_streaming(bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=ds_val, stats=stats, device=device, split_name=f"stage5_val_{extractor_name}", max_rows=LINEAR_STAGE5_MAX_VAL_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS) if LINEAR_STAGE5_REEVALUATE else (st4.get("val_metrics", {}) or {})
            pred_val = collect_predictions_and_labels_streaming(model_bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=ds_val, max_rows=LINEAR_STAGE5_MAX_VAL_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS, split_name=f"val_{extractor_name}", progress_stage="stage5", progress_action="diagnostics")
        finally:
            close_dataset(ds_val, name=f"stage5_{extractor_name}_val"); del ds_val; force_gc(f"stage5_{extractor_name}_val")
        pred_val_summary = summarize_prediction_arrays(pred_val); shift = run_label_shift_sanity_checks(pred_payload=pred_val, stats=stats, shifts=LINEAR_STAGE5_LABEL_SHIFT_VALUES, split_name=f"val_{extractor_name}") if LINEAR_STAGE5_LABEL_SHIFT_VALUES else {}; perm = run_label_permutation_sanity_check(pred_payload=pred_val, stats=stats, seed=LINEAR_STAGE5_PERMUTATION_SEED, split_name=f"val_{extractor_name}") if LINEAR_STAGE5_LABEL_PERMUTATION else None
        test_metrics = None; pred_test_summary=None; shift_t={}; perm_t=None
        if LINEAR_STAGE5_RUN_TEST and plan["has_cmssl_test"]:
            ds_test = build_test_dataset_from_plan(plan)
            if ds_test is not None:
                try:
                    test_metrics = evaluate_stage4_bundle_streaming(bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=ds_test, stats=stats, device=device, split_name=f"stage5_test_{extractor_name}", max_rows=LINEAR_STAGE5_MAX_TEST_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS) if LINEAR_STAGE5_REEVALUATE else st4.get("test_metrics")
                    pred_test = collect_predictions_and_labels_streaming(model_bundle=model_bundle, extractor=extractor, preprocess_bundle=pb, ds=ds_test, max_rows=LINEAR_STAGE5_MAX_TEST_ROWS, batch_rows=LINEAR_STAGE5_BATCH_ROWS, split_name=f"test_{extractor_name}", progress_stage="stage5", progress_action="diagnostics"); pred_test_summary = summarize_prediction_arrays(pred_test); shift_t = run_label_shift_sanity_checks(pred_payload=pred_test, stats=stats, shifts=LINEAR_STAGE5_LABEL_SHIFT_VALUES, split_name=f"test_{extractor_name}") if LINEAR_STAGE5_LABEL_SHIFT_VALUES else {}; perm_t = run_label_permutation_sanity_check(pred_payload=pred_test, stats=stats, seed=LINEAR_STAGE5_PERMUTATION_SEED, split_name=f"test_{extractor_name}") if LINEAR_STAGE5_LABEL_PERMUTATION else None
                    if LINEAR_STAGE5_SAVE_PREDICTIONS: save_stage5_prediction_dump(diag_dir / extractor_name / "test_predictions.npz", pred_test, LINEAR_STAGE5_PREDICTION_MAX_ROWS)
                finally:
                    close_dataset(ds_test, name=f"stage5_{extractor_name}_test"); del ds_test; force_gc(f"stage5_{extractor_name}_test")
        if LINEAR_STAGE5_SAVE_PREDICTIONS: save_stage5_prediction_dump(diag_dir / extractor_name / "val_predictions.npz", pred_val, LINEAR_STAGE5_PREDICTION_MAX_ROWS)
        audit = load_stage3_audit_summary_for_stage5(st3); diag = {"stage4_payload_path": st4.get("payload_path"), "stage3_audit_summary": audit, "best_model_path": str(st4["best_model_path"]), "coefficient_diagnostics": summarize_linear_model_coefficients(model_bundle, top_k=LINEAR_STAGE5_TOP_COEFS), "prediction_summary_val": pred_val_summary, "prediction_summary_test": pred_test_summary, "label_shift_sanity_val": shift, "label_permutation_sanity_val": perm, "label_shift_sanity_test": shift_t, "label_permutation_sanity_test": perm_t, "val_metrics": _jsonable_metrics(val_metrics), "test_metrics": _jsonable_metrics(test_metrics)}
        diagnostics[extractor_name]=diag; row = build_stage5_comparison_row(extractor_name=extractor_name, preprocess_name=preprocess_name, predictor=predictor, stage4_payload=st4, val_metrics=val_metrics, test_metrics=test_metrics, diagnostics=diag); add_stage3_audit_fields_to_comparison_row(row, audit); row["decision_stride_rows"] = int(LINEAR_DECISION_STRIDE_ROWS); row["decision_offset_rows"] = int(LINEAR_DECISION_OFFSET_ROWS); rows.append(row)
        ddir = diag_dir / extractor_name; ddir.mkdir(parents=True, exist_ok=True); (ddir / f"diagnostics_{extractor_name}.json").write_text(json.dumps(diag, allow_nan=True, indent=2), encoding="utf-8")
    _maybe_add_baseline_row(rows, strict=LINEAR_STAGE5_STRICT)
    csv_path = stage5_dir / "linear_stage5_comparison.csv"; json_path = stage5_dir / "linear_stage5_comparison.json"; copy_csv = Path(linear_out_dir) / "linear_stage5_comparison.csv"; copy_json = Path(linear_out_dir) / "linear_stage5_comparison.json"; write_rows_csv(csv_path, rows); write_rows_csv(copy_csv, rows)
    payload = {"stage": "stage5", "status": "ok", "schema": LINEAR_STAGE5_SCHEMA, **_decision_metadata(), **_progress_metadata(), "linear_out_dir": str(linear_out_dir), "preprocess_name": preprocess_name, "predictor": predictor, "extractors_requested": extractor_names, "extractors_completed": [r["extractor"] for r in rows if r.get("extractor") != "CMSSL_neural_baseline"], "strict": bool(LINEAR_STAGE5_STRICT), "reevaluate": bool(LINEAR_STAGE5_REEVALUATE), "run_test": bool(LINEAR_STAGE5_RUN_TEST), "label_shifts": LINEAR_STAGE5_LABEL_SHIFT_VALUES, "label_permutation": bool(LINEAR_STAGE5_LABEL_PERMUTATION), "comparison_rows": rows, "diagnostics": diagnostics}
    json_path.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); copy_json.write_text(json.dumps(payload, allow_nan=True, indent=2), encoding="utf-8"); return payload


if __name__ == "__main__":
    assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"
    main()
