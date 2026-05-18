
#!/usr/bin/env python3
"""
CMSSL17_offline.py

Run CMSSL17's model using flat decision rows produced by offline_ingest.py.
This mirrors the training/eval flow in CMSSL17.py but reads dataset splits
from OUT_ROOT/meta.json and week meta files, with dynamic sequence slicing at load time.
"""

import os, sys, math, json, time
from typing import List, Dict, Tuple, Iterable, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch._inductor.config
import torch.nn.functional as F
from tqdm import tqdm


# ---------------- Import from CMSSL17 ----------------
# Configure CUDA allocator only for this entrypoint execution to avoid
# import-time side effects when CMSSL17 is used as a library module.
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from CMSSL17 import (  # type: ignore
    SAMBA, ModelArgs,
    LOOKBACK, WINDOW_MS, AUX_DIM, HORIZONS_MS, NUM_HORIZONS, HORIZON_WEIGHTS,
    BATCH_SIZE, EPOCHS, LR, PATIENCE, CLIP_GRAD,
    DMODEL, MAMBA_LAYERS,
    PRIMARY_METRIC, PRIMARY_METRIC_HORIZON_MS,
    PRIMARY_DIR_BAL_ACC_GUARD,
    LOW_ABS_TRIM_FRACTION, HIGH_ABS_TRIM_FRACTION, TARGET_TRANSFORM, TARGET_TASK, LABEL_TRIM_SCHEMA, CHECKPOINT_SCHEMA, MODEL_ARCH_SCHEMA,
    MODEL_OUTPUT_SCHEMA,
    DIR_LOSS_WEIGHT, MAG_LOSS_WEIGHT, MAG_CORR_LOSS_WEIGHT, EMA_DECAY,
    FEATURE_SCHEMA, FEATURE_TRANSFORM, FEATURE_TRANSFORM_POLICY, FEATURE_TRANSFORM_WARMUP_ROWS, AUX_TRANSFORM, AUX_SCHEMA, FEATURE_AUX_TAIL, feature_transform_spec_hash,
    SINGLE_WEEK_PATIENCE, get_primary_metric_mode, compute_primary_metric, is_metric_improved,
    derive_dir_mag_predictions, derive_mag_pred_sqrt_for_mag_loss,
    SAM,
    build_dataset_from_split,
    FOUR_WEEK_PROTOCOL,
    FIVE_WEEK_PROTOCOL,
    CMSSL_TRAIN_VAL_PROTOCOL,
    CMSSL_TRAIN_VAL_TEST_PROTOCOL,
    SUPPORTED_SPLIT_PROTOCOLS,
)

# ---------------- Config via env ----------------
OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
USE_IN_MEMORY = int(os.environ.get("BYBIT_USE_IN_MEMORY", "0")) == 1
WORKERS_TRAIN = int(os.environ.get("BYBIT_WORKERS", "8"))
WORKERS_VAL   = max(1, min(4, WORKERS_TRAIN // 2))
AMP_ENABLED   = int(os.environ.get("BYBIT_AMP", "1")) == 1
COMPILE_ENABLED = int(os.environ.get("BYBIT_TORCH_COMPILE", "1")) == 1
COMPILE_MODE = os.environ.get("BYBIT_TORCH_COMPILE_MODE", "max-autotune").strip()
LOG_EVERY     = max(1, int(os.environ.get("BYBIT_LOG_EVERY", "100")))
CUDNN_BENCHMARK = int(os.environ.get("BYBIT_CUDNN_BENCHMARK", "1")) == 1
MATMUL_PRECISION = os.environ.get("BYBIT_MATMUL_PRECISION", "high").strip().lower()
LR = float(os.environ.get("BYBIT_LR", "2e-4"))
CLIP_GRAD = float(os.environ.get("BYBIT_CLIP_GRAD", "5.0"))
FINITE_DEBUG = int(os.environ.get("BYBIT_FINITE_DEBUG", "1")) == 1
USE_SAM = int(os.environ.get("BYBIT_USE_SAM", "0")) == 1
SAM_RHO = float(os.environ.get("BYBIT_SAM_RHO", "0.005"))
GATE_WARMUP_STEPS = int(os.environ.get("BYBIT_GATE_WARMUP_STEPS", "18892"))
GATE_LR = float(os.environ.get("BYBIT_GATE_LR", "2e-5"))
MODEL_DIAG = int(os.environ.get("BYBIT_MODEL_DIAG", "1")) == 1
MODEL_DIAG_EVERY = max(1, int(os.environ.get("BYBIT_MODEL_DIAG_EVERY", str(LOG_EVERY))))
MODEL_DIAG_MAX_BATCH = max(1, int(os.environ.get("BYBIT_MODEL_DIAG_MAX_BATCH", "64")))
GRAD_DIAG_P95_MAX_ELEMS = max(
    1024,
    int(os.environ.get("BYBIT_GRAD_DIAG_P95_MAX_ELEMS", "1000000")),
)
BAND_DIAG = int(os.environ.get("BYBIT_BAND_DIAG", "1")) == 1
BAND_DIAG_TRAIN = int(os.environ.get("BYBIT_BAND_DIAG_TRAIN", "1")) == 1
BAND_DIAG_TRAIN_MAX_ROWS = max(1, int(os.environ.get("BYBIT_BAND_DIAG_TRAIN_MAX_ROWS", "200000")))
BAND_DIAG_QUANTILES = np.array([0.00, 0.25, 0.50, 0.65, 0.75, 0.85, 0.925, 0.975, 1.00], dtype=np.float32)
BAND_DIAG_NAMES = ["q00-q25", "q25-q50", "q50-q65", "q65-q75", "q75-q85", "q85-q925", "q925-q975", "q975-q100"]
if os.environ.get("BYBIT_SUPPRESS_CMSSL_CONFIG_PRINTS", "0").strip() != "1":
    print(
        f"[band-diag-config] enabled={int(BAND_DIAG)} train_enabled={int(BAND_DIAG_TRAIN)} "
        f"train_max_rows={BAND_DIAG_TRAIN_MAX_ROWS} bands={','.join(BAND_DIAG_NAMES)}",
        flush=True,
    )
    print(
        f"[model-diag-config] enabled={int(MODEL_DIAG)} "
        f"every={MODEL_DIAG_EVERY} max_batch={MODEL_DIAG_MAX_BATCH} "
        f"grad_p95_max_elems={GRAD_DIAG_P95_MAX_ELEMS}",
        flush=True,
    )

if not math.isfinite(SAM_RHO) or SAM_RHO < 0.0:
    raise ValueError(f"BYBIT_SAM_RHO must be finite and >= 0, got {SAM_RHO}")
if GATE_WARMUP_STEPS < 0:
    raise ValueError(f"BYBIT_GATE_WARMUP_STEPS must be >= 0, got {GATE_WARMUP_STEPS}")
if not math.isfinite(GATE_LR) or GATE_LR <= 0.0:
    raise ValueError(f"BYBIT_GATE_LR must be finite and > 0, got {GATE_LR}")
if os.environ.get("BYBIT_SUPPRESS_CMSSL_CONFIG_PRINTS", "0").strip() != "1":
    print(f"[gate-config] warmup_steps={GATE_WARMUP_STEPS} gate_lr={GATE_LR}", flush=True)

EXPECTED_DECISION_TIME_BASIS = "ob_event_time"
EXPECTED_DECISION_POLICY = "ob_event_time"
SUPPORTED_PROTOCOLS = set(SUPPORTED_SPLIT_PROTOCOLS)
FAST_VAL_MAX_ROWS = 200_000
FULL_VAL_EVERY = 5
TRAIN_ROW_STRIDE = int(os.environ.get("BYBIT_TRAIN_ROW_STRIDE", "5"))
TRAIN_DATA_DEVICE = os.environ.get("BYBIT_TRAIN_DATA_DEVICE", "cpu_pinned").strip().lower()
if TRAIN_DATA_DEVICE not in {"cpu_pinned", "gpu_all"}:
    raise ValueError(
        "BYBIT_TRAIN_DATA_DEVICE must be one of: cpu_pinned, gpu_all; "
        f"got {TRAIN_DATA_DEVICE!r}"
    )
_feature_storage_dtype_raw = os.environ.get("BYBIT_FEATURE_STORAGE_DTYPE", "bf16").strip().lower()
BF16_FEATURE_DEBUG = int(os.environ.get("BYBIT_BF16_FEATURE_DEBUG", "0")) == 1
BF16_FEATURE_DEBUG_MAX_BATCHES = max(1, int(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_MAX_BATCHES", "3")))
BF16_FEATURE_DEBUG_WARN_MAX_ABS_ERR = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_MAX_ABS_ERR", "0.05"))
BF16_FEATURE_DEBUG_WARN_MEAN_ABS_ERR = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_MEAN_ABS_ERR", "0.002"))
BF16_FEATURE_DEBUG_WARN_SIGN_FLIP_FRAC = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_SIGN_FLIP_FRAC", "0.001"))
BF16_FEATURE_DEBUG_WARN_ZERO_FLIP_FRAC = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_ZERO_FLIP_FRAC", "0.001"))

if _feature_storage_dtype_raw in {"fp32", "float32"}:
    FEATURE_STORAGE_DTYPE = torch.float32
    FEATURE_STORAGE_DTYPE_NAME = "fp32"
elif _feature_storage_dtype_raw in {"bf16", "bfloat16"}:
    FEATURE_STORAGE_DTYPE = torch.bfloat16
    FEATURE_STORAGE_DTYPE_NAME = "bf16"
else:
    raise ValueError(
        "BYBIT_FEATURE_STORAGE_DTYPE must be one of: fp32, float32, bf16, bfloat16; "
        f"got {_feature_storage_dtype_raw!r}"
    )
if TRAIN_ROW_STRIDE < 1:
    raise ValueError(f"BYBIT_TRAIN_ROW_STRIDE must be >= 1, got {TRAIN_ROW_STRIDE}")

if os.environ.get("BYBIT_SUPPRESS_CMSSL_CONFIG_PRINTS", "0").strip() != "1":
    print(
        f"[data-config] train_data_device={TRAIN_DATA_DEVICE} "
        f"feature_storage_dtype={FEATURE_STORAGE_DTYPE_NAME}",
        flush=True,
    )
    print(
        f"[bf16-feature-debug-config] enabled={int(BF16_FEATURE_DEBUG)} "
        f"max_batches={BF16_FEATURE_DEBUG_MAX_BATCHES}",
        flush=True,
    )

_BF16_FEATURE_DEBUG_BATCHES_PRINTED = 0
DIR_CLASS_WEIGHT_TEMPER = 0.5
DIR_CLASS_WEIGHT_MIN = 0.75
DIR_CLASS_WEIGHT_MAX = 1.50


def summarize_bf16_feature_error(x_fp32: torch.Tensor, x_stored: torch.Tensor) -> Dict[str, float]:
    """
    Compare original FP32 feature batch to stored dtype roundtrip.
    x_fp32 must be float32.
    x_stored may be bf16/fp32 and is compared after conversion back to fp32.
    """
    a = x_fp32.detach().float()
    b = x_stored.detach().float()

    if a.shape != b.shape:
        raise ValueError(f"BF16 debug shape mismatch: fp32={tuple(a.shape)} stored={tuple(b.shape)}")

    diff = b - a
    abs_diff = diff.abs()
    abs_a = a.abs()
    denom = torch.clamp(abs_a, min=1e-6)
    rel_abs = abs_diff / denom

    finite_a = torch.isfinite(a)
    finite_b = torch.isfinite(b)
    finite_both = finite_a & finite_b

    nonzero_a = abs_a > 0.0
    sign_flip = ((torch.sign(a) != torch.sign(b)) & nonzero_a & finite_both)
    zero_flip = (((a == 0.0) != (b == 0.0)) & finite_both)

    # Per-feature max mean error is useful because a global mean can hide ruined sparse features.
    flat_abs = abs_diff.reshape(-1, abs_diff.shape[-1]) if abs_diff.ndim >= 2 else abs_diff.reshape(-1, 1)
    per_feature_mean_abs = flat_abs.mean(dim=0)

    return {
        "max_abs_err": float(abs_diff.max().cpu()),
        "mean_abs_err": float(abs_diff.mean().cpu()),
        "p99_abs_err": float(torch.quantile(abs_diff.reshape(-1), 0.99).cpu()),
        "max_rel_abs_err": float(rel_abs.max().cpu()),
        "p99_rel_abs_err": float(torch.quantile(rel_abs.reshape(-1), 0.99).cpu()),
        "sign_flip_frac": float(sign_flip.float().mean().cpu()),
        "zero_flip_frac": float(zero_flip.float().mean().cpu()),
        "nonfinite_after_frac": float((~finite_b).float().mean().cpu()),
        "per_feature_mean_abs_err_max": float(per_feature_mean_abs.max().cpu()),
    }


def maybe_print_bf16_feature_debug(x_fp32: torch.Tensor, x_stored: torch.Tensor, *, enabled: bool) -> None:
    global _BF16_FEATURE_DEBUG_BATCHES_PRINTED
    if not enabled or FEATURE_STORAGE_DTYPE_NAME != "bf16":
        return
    if _BF16_FEATURE_DEBUG_BATCHES_PRINTED >= BF16_FEATURE_DEBUG_MAX_BATCHES:
        return

    debug_batch_i = int(_BF16_FEATURE_DEBUG_BATCHES_PRINTED)
    _BF16_FEATURE_DEBUG_BATCHES_PRINTED += 1
    diag = summarize_bf16_feature_error(x_fp32, x_stored)
    print(
        "[bf16-feature-debug] "
        f"batch={debug_batch_i} "
        f"max_abs_err={diag['max_abs_err']:.6g} "
        f"mean_abs_err={diag['mean_abs_err']:.6g} "
        f"p99_abs_err={diag['p99_abs_err']:.6g} "
        f"max_rel_abs_err={diag['max_rel_abs_err']:.6g} "
        f"p99_rel_abs_err={diag['p99_rel_abs_err']:.6g} "
        f"sign_flip_frac={diag['sign_flip_frac']:.6g} "
        f"zero_flip_frac={diag['zero_flip_frac']:.6g} "
        f"nonfinite_after_frac={diag['nonfinite_after_frac']:.6g} "
        f"per_feature_mean_abs_err_max={diag['per_feature_mean_abs_err_max']:.6g}",
        flush=True,
    )
    if (
        diag["max_abs_err"] > BF16_FEATURE_DEBUG_WARN_MAX_ABS_ERR
        or diag["mean_abs_err"] > BF16_FEATURE_DEBUG_WARN_MEAN_ABS_ERR
        or diag["sign_flip_frac"] > BF16_FEATURE_DEBUG_WARN_SIGN_FLIP_FRAC
        or diag["zero_flip_frac"] > BF16_FEATURE_DEBUG_WARN_ZERO_FLIP_FRAC
        or diag["nonfinite_after_frac"] > 0.0
    ):
        print("[bf16-feature-debug-warn] BF16 feature storage may be materially changing features; consider BYBIT_FEATURE_STORAGE_DTYPE=fp32", flush=True)

if __name__ == "__main__":
    assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"

def require_supported_pipeline_splits(meta: dict, out_root: Path) -> dict:
    if "splits" not in meta:
        raise KeyError(
            "meta.json missing required key 'splits'. Run offline_ingest to generate offline dataset metadata."
        )
    splits = meta["splits"]
    if not isinstance(splits, dict):
        raise KeyError("meta['splits'] must be a dict. Rerun offline_ingest.")

    protocol = splits.get("protocol")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"meta['splits']['protocol'] must be one of {sorted(SUPPORTED_PROTOCOLS)}. Rerun offline_ingest."
        )

    if "weeks_in_order" not in meta:
        raise KeyError("meta.json missing required key 'weeks_in_order'. Rerun offline_ingest.")
    weeks_in_order = meta["weeks_in_order"]
    if not isinstance(weeks_in_order, list) or not all(isinstance(w, str) and w for w in weeks_in_order):
        raise KeyError("meta['weeks_in_order'] must be a non-empty list[str]. Rerun offline_ingest.")
    expected_week_counts = {
        FIVE_WEEK_PROTOCOL: {5},
        FOUR_WEEK_PROTOCOL: {4},
        CMSSL_TRAIN_VAL_PROTOCOL: {2, 3},
        CMSSL_TRAIN_VAL_TEST_PROTOCOL: {3, 4},
    }[protocol]
    if len(weeks_in_order) not in expected_week_counts:
        raise KeyError(
            f"meta['weeks_in_order'] has {len(weeks_in_order)} entries but protocol {protocol} requires "
            f"{sorted(expected_week_counts)}. Rerun offline_ingest."
        )

    decision_time_basis = meta.get("decision_time_basis")
    if decision_time_basis != EXPECTED_DECISION_TIME_BASIS:
        raise ValueError(
            "meta.json has incompatible decision_time_basis. "
            f"Expected '{EXPECTED_DECISION_TIME_BASIS}' (event-time decision timestamps); "
            f"got {decision_time_basis!r}. "
            "Rerun offline_ingest to regenerate metadata with event-time decisions enabled."
        )
    if "decision_policy" in meta:
        decision_policy = meta.get("decision_policy")
        if decision_policy != EXPECTED_DECISION_POLICY:
            raise ValueError(
                "meta.json has incompatible decision_policy. "
                f"Expected '{EXPECTED_DECISION_POLICY}' (event-time decision policy); "
                f"got {decision_policy!r}. "
                "Rerun offline_ingest to regenerate metadata with event-time decisions enabled."
            )

    known_weeks = set(weeks_in_order)

    weeks_meta_map = meta.get("weeks_meta")
    if not isinstance(weeks_meta_map, dict) or not weeks_meta_map:
        raise KeyError("meta.json missing required non-empty key 'weeks_meta'. Rerun offline_ingest.")

    def _full_week_range(week_key: str, stage: str) -> Tuple[int, int]:
        rel_path = weeks_meta_map.get(week_key)
        if not isinstance(rel_path, str) or not rel_path:
            raise KeyError(f"meta['weeks_meta'] missing path for week '{week_key}' referenced by {stage}.")
        week_meta = json.loads((out_root / rel_path).read_text())
        decision_range = week_meta.get("decision_ts_range")
        if not isinstance(decision_range, dict) or "min" not in decision_range or "max" not in decision_range:
            raise KeyError(f"Week metadata for {stage} must include decision_ts_range min/max.")
        start = int(decision_range["min"])
        end = int(decision_range["max"]) + 1
        if start >= end:
            raise ValueError(f"Week metadata for {stage} has invalid decision_ts_range: start={start} end={end}.")
        return start, end

    def _normalize_split_entry(stage: str, entry: Any, *, require_range: bool) -> dict:
        if not isinstance(entry, dict):
            raise KeyError(f"meta['splits']['{stage}'] must be a dict. Rerun offline_ingest.")

        week_value = entry.get("week", entry.get("weeks"))
        if isinstance(week_value, str) and week_value:
            weeks = [week_value]
        elif isinstance(week_value, list) and week_value and all(isinstance(w, str) and w for w in week_value):
            weeks = list(week_value)
        else:
            raise KeyError(
                f"meta['splits']['{stage}'] must include non-empty 'week' or 'weeks'. Rerun offline_ingest."
            )

        missing_weeks = sorted(w for w in weeks if w not in known_weeks)
        if missing_weeks:
            raise KeyError(
                f"meta['splits']['{stage}'] references week(s) not present in meta['weeks_in_order']: {missing_weeks}"
            )

        decision_ts_range = entry.get("decision_ts_range")
        if require_range:
            if not isinstance(decision_ts_range, dict) or "start" not in decision_ts_range or "end" not in decision_ts_range:
                raise KeyError(
                    f"meta['splits']['{stage}'] must include decision_ts_range with start/end. Rerun offline_ingest."
                )
            start = int(decision_ts_range["start"])
            end = int(decision_ts_range["end"])
        else:
            explicit_start = entry.get("start")
            explicit_end = entry.get("end")
            if explicit_start is None or explicit_end is None:
                if isinstance(decision_ts_range, dict):
                    explicit_start = decision_ts_range.get("start")
                    explicit_end = decision_ts_range.get("end")
            if explicit_start is not None and explicit_end is not None:
                start = int(explicit_start)
                end = int(explicit_end)
            else:
                start, end = _full_week_range(weeks[0], stage)
        if start >= end:
            raise ValueError(f"meta['splits']['{stage}'] must satisfy start < end. Rerun offline_ingest.")

        return {"weeks": weeks, "start": start, "end": end}

    if protocol in {FOUR_WEEK_PROTOCOL, FIVE_WEEK_PROTOCOL}:
        required_entries = {
            "cmssl.train": ("cmssl", "train", False),
            "cmssl.val": ("cmssl", "val", False),
            "cmssl.test": ("cmssl", "test", False),
            "rl.train": ("rl", "train", True),
            "rl.val": ("rl", "val", True),
            "rl.test": ("rl", "test", True),
            "eval.full": ("eval", "full", False),
        }
    elif protocol == CMSSL_TRAIN_VAL_PROTOCOL:
        required_entries = {
            "cmssl.train": ("cmssl", "train", False),
            "cmssl.val": ("cmssl", "val", False),
        }
    elif protocol == CMSSL_TRAIN_VAL_TEST_PROTOCOL:
        required_entries = {
            "cmssl.train": ("cmssl", "train", False),
            "cmssl.val": ("cmssl", "val", False),
            "cmssl.test": ("cmssl", "test", False),
        }
    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")

    normalized = {"protocol": protocol}
    for section in ("cmssl", "rl", "eval"):
        sec = splits.get(section, {})
        if not isinstance(sec, dict):
            raise KeyError(f"meta['splits']['{section}'] must be a dict. Rerun offline_ingest.")
        normalized[section] = {}

    for label, (section, name, require_range) in required_entries.items():
        normalized[section][name] = _normalize_split_entry(label, splits[section].get(name), require_range=require_range)

    if protocol == FOUR_WEEK_PROTOCOL:
        week1, week2, week3, week4 = weeks_in_order
        if normalized["cmssl"]["train"]["weeks"] != [week1]:
            raise ValueError("meta['splits']['cmssl']['train'] must reference weeks_in_order[0].")
        if normalized["cmssl"]["val"]["weeks"] != [week2]:
            raise ValueError("meta['splits']['cmssl']['val'] must reference weeks_in_order[1].")
        if normalized["cmssl"]["test"]["weeks"] != [week3]:
            raise ValueError("meta['splits']['cmssl']['test'] must reference weeks_in_order[2].")
        if any(normalized["rl"][name]["weeks"] != [week3] for name in ("train", "val", "test")):
            raise ValueError("meta['splits']['rl'] train/val/test must all reference weeks_in_order[2].")
        if normalized["eval"]["full"]["weeks"] != [week4]:
            raise ValueError("meta['splits']['eval']['full'] must reference weeks_in_order[3].")
    elif protocol == FIVE_WEEK_PROTOCOL:
        week1, week2, week3, week4, week5 = weeks_in_order
        if normalized["cmssl"]["train"]["weeks"] != [week1, week2]:
            raise ValueError("meta['splits']['cmssl']['train'] must reference weeks_in_order[0:2].")
        if normalized["cmssl"]["val"]["weeks"] != [week3]:
            raise ValueError("meta['splits']['cmssl']['val'] must reference weeks_in_order[2].")
        if normalized["cmssl"]["test"]["weeks"] != [week4]:
            raise ValueError("meta['splits']['cmssl']['test'] must reference weeks_in_order[3].")
        if any(normalized["rl"][name]["weeks"] != [week4] for name in ("train", "val", "test")):
            raise ValueError("meta['splits']['rl'] train/val/test must all reference weeks_in_order[3].")
        if normalized["eval"]["full"]["weeks"] != [week5]:
            raise ValueError("meta['splits']['eval']['full'] must reference weeks_in_order[4].")
    elif protocol == CMSSL_TRAIN_VAL_PROTOCOL:
        if len(weeks_in_order) == 3:
            week1, week2, week3 = weeks_in_order
            expected_train, expected_val = [week1, week2], [week3]
        else:
            week1, week2 = weeks_in_order
            expected_train, expected_val = [week1], [week2]
        if normalized["cmssl"]["train"]["weeks"] != expected_train:
            raise ValueError("meta['splits']['cmssl']['train'] does not match the CMSSL train+val protocol week layout.")
        if normalized["cmssl"]["val"]["weeks"] != expected_val:
            raise ValueError("meta['splits']['cmssl']['val'] does not match the CMSSL train+val protocol week layout.")
    elif protocol == CMSSL_TRAIN_VAL_TEST_PROTOCOL:
        if len(weeks_in_order) == 4:
            week1, week2, week3, week4 = weeks_in_order
            expected_train, expected_val, expected_test = [week1, week2], [week3], [week4]
        else:
            week1, week2, week3 = weeks_in_order
            expected_train, expected_val, expected_test = [week1], [week2], [week3]
        if normalized["cmssl"]["train"]["weeks"] != expected_train:
            raise ValueError("meta['splits']['cmssl']['train'] does not match the CMSSL train+val+test protocol week layout.")
        if normalized["cmssl"]["val"]["weeks"] != expected_val:
            raise ValueError("meta['splits']['cmssl']['val'] does not match the CMSSL train+val+test protocol week layout.")
        if normalized["cmssl"]["test"]["weeks"] != expected_test:
            raise ValueError("meta['splits']['cmssl']['test'] does not match the CMSSL train+val+test protocol week layout.")

    if protocol in {FOUR_WEEK_PROTOCOL, FIVE_WEEK_PROTOCOL}:
        rl_train = normalized["rl"]["train"]
        rl_val = normalized["rl"]["val"]
        rl_test = normalized["rl"]["test"]
        if not (rl_train["end"] <= rl_val["start"] < rl_val["end"] <= rl_test["start"] < rl_test["end"]):
            raise ValueError(
                "meta['splits']['rl'] train/val/test decision_ts_range must be strictly ordered and non-overlapping."
            )
        eval_full = normalized["eval"]["full"]
        if not eval_full["weeks"]:
            raise ValueError("meta['splits']['eval']['full'] must reference at least one week.")

    return {
        "protocol": protocol,
        "splits": normalized,
        "weeks_in_order": weeks_in_order,
    }


def make_single_week_split_from_meta(
    *,
    out_root: Path,
    global_meta: dict,
    week_key: str,
) -> dict:
    weeks_meta = global_meta.get("weeks_meta", {})
    rel_path = weeks_meta.get(week_key)
    if not isinstance(rel_path, str) or not rel_path:
        raise KeyError(f"weeks_meta is missing path for week '{week_key}'")
    week_meta = json.loads((out_root / rel_path).read_text())
    decision_range = week_meta.get("decision_ts_range")
    if not isinstance(decision_range, dict) or "min" not in decision_range or "max" not in decision_range:
        raise KeyError(f"Week '{week_key}' metadata missing decision_ts_range min/max")
    start = int(decision_range["min"])
    end = int(decision_range["max"]) + 1
    if start >= end:
        raise ValueError(f"Week '{week_key}' has invalid decision_ts_range: start={start}, end={end}")
    return {"weeks": [week_key], "start": start, "end": end}


def _label_dim_error(source: str, observed: Any) -> ValueError:
    return ValueError(
        f"{source} has label_dim={observed!r}, expected {NUM_HORIZONS}. "
        "Old or incompatible offline dataset. Rerun offline_ingest.py with "
        f"FEATURE_SCHEMA={FEATURE_SCHEMA}."
    )


def validate_dataset_label_dim(meta: dict, source: str) -> None:
    label_dim = meta.get("label_dim")
    if label_dim is None:
        raise ValueError(
            f"{source} is missing label_dim metadata. Rebuild the offline data with offline_ingest.py."
        )
    try:
        label_dim_int = int(label_dim)
    except (TypeError, ValueError):
        raise _label_dim_error(source, label_dim)
    if label_dim_int != NUM_HORIZONS:
        raise _label_dim_error(source, label_dim_int)


def validate_loaded_label_array(y: np.ndarray, source: str) -> None:
    if y.ndim != 2:
        raise ValueError(f"{source} must be 2D, got shape={y.shape}")
    if y.shape[1] != NUM_HORIZONS:
        raise _label_dim_error(source, y.shape[1])


def validate_contract_meta(meta: dict, source: str) -> None:
    ok = (
        meta.get("feature_schema") == FEATURE_SCHEMA
        and meta.get("feature_transform") == FEATURE_TRANSFORM
        and meta.get("feature_transform_policy") == FEATURE_TRANSFORM_POLICY
        and meta.get("aux_transform") == AUX_TRANSFORM
        and bool(meta.get("feature_transform_spec_hash"))
        and int(meta.get("feature_transform_warmup_rows", -1)) == int(FEATURE_TRANSFORM_WARMUP_ROWS)
        and meta.get("aux_schema") == AUX_SCHEMA
        and meta.get("target_transform") == TARGET_TRANSFORM
        and meta.get("target_task") == TARGET_TASK
        and meta.get("label_trim_schema") == LABEL_TRIM_SCHEMA
        and list(map(int, meta.get("horizons_ms", []))) == [int(h) for h in HORIZONS_MS]
        and int(meta.get("label_dim", -1)) == NUM_HORIZONS
        and int(meta.get("aux_dim", -1)) == AUX_DIM
        and list(meta.get("aux_names", [])) == list(FEATURE_AUX_TAIL)
        and int(meta.get("feature_dim_total", -1)) == int(meta.get("feature_dim_core", -1)) + AUX_DIM
        and bool(meta.get("feature_names"))
        and int(meta.get("feature_dim_core", -1)) == len(meta.get("feature_names", []))
        and bool(meta.get("feature_names_hash"))
    )
    if not ok:
        raise ValueError(
            "Old or incompatible offline dataset. Expected offline data contract: "
            f"FEATURE_SCHEMA={FEATURE_SCHEMA}, "
            f"FEATURE_TRANSFORM={FEATURE_TRANSFORM}, "
            f"FEATURE_TRANSFORM_POLICY={FEATURE_TRANSFORM_POLICY}, "
            f"FEATURE_TRANSFORM_WARMUP_ROWS={FEATURE_TRANSFORM_WARMUP_ROWS}, "
            f"AUX_SCHEMA={AUX_SCHEMA}, AUX_DIM={AUX_DIM}, "
            f"TARGET_TASK={TARGET_TASK}, TARGET_TRANSFORM={TARGET_TRANSFORM}, "
            f"LABEL_TRIM_SCHEMA={LABEL_TRIM_SCHEMA}, HORIZONS_MS={[int(h) for h in HORIZONS_MS]}, "
            f"label_dim={NUM_HORIZONS}, feature_dim_total=feature_dim_core+{AUX_DIM}."
        )
    expected_spec_hash = feature_transform_spec_hash(list(meta.get("feature_names", [])))
    if str(meta.get("feature_transform_spec_hash")) != expected_spec_hash:
        raise ValueError(
            "Old or incompatible offline dataset transform spec hash. Rerun offline_ingest.py with "
            f"FEATURE_SCHEMA={FEATURE_SCHEMA}."
        )

    stored_ckpt_schema = meta.get("checkpoint_schema_expected")
    if stored_ckpt_schema and stored_ckpt_schema != CHECKPOINT_SCHEMA:
        print(
            "[contract-warning] dataset checkpoint_schema_expected differs from current model "
            f"stored={stored_ckpt_schema!r} current={CHECKPOINT_SCHEMA!r}; "
            "allowed because checkpoint schema is model-side, not offline data-side.",
            flush=True,
        )


def validate_week_matches_global(global_meta: dict, week_meta: dict, source: str) -> None:
    checks = {
        "feature_schema": (week_meta.get("feature_schema"), global_meta.get("feature_schema")),
        "feature_transform": (week_meta.get("feature_transform"), global_meta.get("feature_transform")),
        "feature_transform_policy": (week_meta.get("feature_transform_policy"), global_meta.get("feature_transform_policy")),
        "feature_transform_spec_hash": (week_meta.get("feature_transform_spec_hash"), global_meta.get("feature_transform_spec_hash")),
        "feature_transform_warmup_rows": (int(week_meta.get("feature_transform_warmup_rows", -1)), int(global_meta.get("feature_transform_warmup_rows", -2))),
        "aux_transform": (week_meta.get("aux_transform"), global_meta.get("aux_transform")),
        "aux_schema": (week_meta.get("aux_schema"), global_meta.get("aux_schema")),
        "checkpoint_schema_expected": (
            week_meta.get("checkpoint_schema_expected"),
            global_meta.get("checkpoint_schema_expected"),
        ),
        "target_transform": (week_meta.get("target_transform"), global_meta.get("target_transform")),
        "target_task": (week_meta.get("target_task"), global_meta.get("target_task")),
        "aux_dim": (int(week_meta.get("aux_dim", -1)), int(global_meta.get("aux_dim", -2))),
        "feature_dim_core": (
            int(week_meta.get("feature_dim_core", -1)),
            int(global_meta.get("feature_dim_core", -2)),
        ),
        "feature_dim_total": (
            int(week_meta.get("feature_dim_total", -1)),
            int(global_meta.get("feature_dim_total", -2)),
        ),
        "feature_names_hash": (
            week_meta.get("feature_names_hash"),
            global_meta.get("feature_names_hash"),
        ),
    }

    for field, (week_value, global_value) in checks.items():
        if week_value != global_value:
            raise ValueError(
                "Week metadata does not match global metadata. "
                f"source={source}, field={field}, week_value={week_value!r}, "
                f"global_value={global_value!r}. Rerun offline_ingest.py with "
                f"FEATURE_SCHEMA={FEATURE_SCHEMA}."
            )

    if list(week_meta.get("aux_names", [])) != list(global_meta.get("aux_names", [])):
        raise ValueError(
            "Week metadata aux_names do not match global metadata. "
            f"source={source}. Rerun offline_ingest.py with FEATURE_SCHEMA={FEATURE_SCHEMA}."
        )

    if list(week_meta.get("feature_names", [])) != list(global_meta.get("feature_names", [])):
        raise ValueError(
            "Week metadata feature_names do not match global metadata. "
            f"source={source}. Rerun offline_ingest.py with FEATURE_SCHEMA={FEATURE_SCHEMA}."
        )


# ---------------- Signed-raw preprocessing, cache, and metrics ----------------


def build_signed_side_trim_masks_np(
    y_raw_bps: np.ndarray,
    *,
    pos_lo_raw_bps: np.ndarray,
    pos_hi_raw_bps: np.ndarray,
    neg_lo_abs_bps: np.ndarray,
    neg_hi_abs_bps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build signed nonzero trim masks.

    Positive rows use y > 0 and y itself.
    Negative rows use y < 0 and -y as magnitude.
    Zero rows are excluded from all masks.
    """
    y = np.asarray(y_raw_bps, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != NUM_HORIZONS:
        raise ValueError(f"y must have shape [N, {NUM_HORIZONS}], got {y.shape}")

    pos_lo = np.asarray(pos_lo_raw_bps, dtype=np.float32).reshape(1, -1)
    pos_hi = np.asarray(pos_hi_raw_bps, dtype=np.float32).reshape(1, -1)
    neg_lo = np.asarray(neg_lo_abs_bps, dtype=np.float32).reshape(1, -1)
    neg_hi = np.asarray(neg_hi_abs_bps, dtype=np.float32).reshape(1, -1)

    pos = y > 0.0
    neg = y < 0.0
    neg_mag = (-y).clip(min=0.0)

    keep_pos = pos & (y >= pos_lo) & (y <= pos_hi)
    keep_neg = neg & (neg_mag >= neg_lo) & (neg_mag <= neg_hi)
    keep_signed = keep_pos | keep_neg
    return keep_pos, keep_neg, keep_signed


def build_signed_side_trim_masks_torch(
    y_raw: torch.Tensor,
    *,
    pos_lo_t: torch.Tensor,
    pos_hi_t: torch.Tensor,
    neg_lo_t: torch.Tensor,
    neg_hi_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = y_raw > 0.0
    neg = y_raw < 0.0
    neg_mag = (-y_raw).clamp_min(0.0)

    keep_pos = pos & (y_raw >= pos_lo_t) & (y_raw <= pos_hi_t)
    keep_neg = neg & (neg_mag >= neg_lo_t) & (neg_mag <= neg_hi_t)
    keep_signed = keep_pos | keep_neg
    return keep_pos, keep_neg, keep_signed


def build_signed_side_trim_masks_from_stats_np(
    y_raw_bps: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_signed_side_trim_masks_np(
        y_raw_bps,
        pos_lo_raw_bps=stats["pos_lo_raw_bps"],
        pos_hi_raw_bps=stats["pos_hi_raw_bps"],
        neg_lo_abs_bps=stats["neg_lo_abs_bps"],
        neg_hi_abs_bps=stats["neg_hi_abs_bps"],
    )


def compute_signed_raw_stats(y_train: np.ndarray) -> Dict[str, np.ndarray]:
    y = np.asarray(y_train, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != NUM_HORIZONS:
        raise ValueError(f"y must have shape [N, {NUM_HORIZONS}], got {y.shape}")

    pos_lo = np.zeros(NUM_HORIZONS, dtype=np.float32)
    pos_hi = np.zeros(NUM_HORIZONS, dtype=np.float32)
    neg_lo = np.zeros(NUM_HORIZONS, dtype=np.float32)
    neg_hi = np.zeros(NUM_HORIZONS, dtype=np.float32)

    q50 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    q85 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    pos_q50 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    neg_q50 = np.zeros(NUM_HORIZONS, dtype=np.float32)

    band_edges = np.full((NUM_HORIZONS, len(BAND_DIAG_QUANTILES)), np.nan, dtype=np.float32)
    min_side_rows = 100

    for h in range(NUM_HORIZONS):
        yh = y[:, h].astype(np.float64, copy=False)
        pos_vals = yh[yh > 0.0]
        neg_vals = -yh[yh < 0.0]

        if pos_vals.size < min_side_rows:
            raise ValueError(
                f"Too few positive nonzero train labels for horizon={HORIZONS_MS[h]}ms: "
                f"n={pos_vals.size}, required>={min_side_rows}"
            )
        if neg_vals.size < min_side_rows:
            raise ValueError(
                f"Too few negative nonzero train labels for horizon={HORIZONS_MS[h]}ms: "
                f"n={neg_vals.size}, required>={min_side_rows}"
            )

        pos_lo[h] = np.float32(np.quantile(pos_vals, LOW_ABS_TRIM_FRACTION))
        pos_hi[h] = np.float32(np.quantile(pos_vals, 1.0 - HIGH_ABS_TRIM_FRACTION))
        neg_lo[h] = np.float32(np.quantile(neg_vals, LOW_ABS_TRIM_FRACTION))
        neg_hi[h] = np.float32(np.quantile(neg_vals, 1.0 - HIGH_ABS_TRIM_FRACTION))

    keep_pos, keep_neg, keep_signed = build_signed_side_trim_masks_np(
        y,
        pos_lo_raw_bps=pos_lo,
        pos_hi_raw_bps=pos_hi,
        neg_lo_abs_bps=neg_lo,
        neg_hi_abs_bps=neg_hi,
    )

    abs_y = np.abs(y).astype(np.float32)

    for h in range(NUM_HORIZONS):
        kept_abs = abs_y[keep_signed[:, h], h]
        kept_pos_abs = y[keep_pos[:, h], h]
        kept_neg_abs = -y[keep_neg[:, h], h]

        if kept_abs.size < min_side_rows:
            raise ValueError(f"Too few signed kept labels for horizon={HORIZONS_MS[h]}ms after trim")

        q50[h] = np.float32(np.quantile(kept_abs.astype(np.float64), 0.50))
        q85[h] = np.float32(np.quantile(kept_abs.astype(np.float64), 0.85))
        pos_q50[h] = np.float32(np.quantile(kept_pos_abs.astype(np.float64), 0.50))
        neg_q50[h] = np.float32(np.quantile(kept_neg_abs.astype(np.float64), 0.50))

        edges = np.quantile(kept_abs.astype(np.float64), BAND_DIAG_QUANTILES).astype(np.float32)
        edges[0] = float(np.min(kept_abs))
        edges[-1] = float(np.max(kept_abs))
        band_edges[h] = edges

    zero_frac = np.mean(y == 0.0, axis=0).astype(np.float32)
    pos_kept_count = keep_pos.sum(axis=0).astype(np.int64)
    neg_kept_count = keep_neg.sum(axis=0).astype(np.int64)
    signed_kept_count = keep_signed.sum(axis=0).astype(np.int64)

    return {
        "pos_lo_raw_bps": pos_lo,
        "pos_hi_raw_bps": pos_hi,
        "neg_lo_abs_bps": neg_lo,
        "neg_hi_abs_bps": neg_hi,
        "kept_q50_abs_raw_bps": q50,
        "kept_q85_abs_raw_bps": q85,
        "kept_pos_q50_abs_raw_bps": pos_q50,
        "kept_neg_q50_abs_raw_bps": neg_q50,
        "band_quantiles": BAND_DIAG_QUANTILES.copy(),
        "kept_abs_band_edges_raw_bps": band_edges,
        "zero_frac": zero_frac,
        "pos_kept_count": pos_kept_count,
        "neg_kept_count": neg_kept_count,
        "signed_kept_count": signed_kept_count,
    }


def load_stats_cache(path: Path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as c:
        required = (
            "pos_lo_raw_bps",
            "pos_hi_raw_bps",
            "neg_lo_abs_bps",
            "neg_hi_abs_bps",
            "kept_q50_abs_raw_bps",
            "kept_q85_abs_raw_bps",
            "kept_pos_q50_abs_raw_bps",
            "kept_neg_q50_abs_raw_bps",
            "band_quantiles",
            "kept_abs_band_edges_raw_bps",
            "zero_frac",
            "pos_kept_count",
            "neg_kept_count",
            "signed_kept_count",
        )
        if any(k not in c.files for k in required) or "metadata_json" not in c.files:
            return None
        count_keys = {"pos_kept_count", "neg_kept_count", "signed_kept_count"}
        stats = {
            k: np.asarray(c[k], dtype=np.int64 if k in count_keys else np.float32)
            for k in required
        }
        meta = json.loads(str(c["metadata_json"].item()))
    return stats, meta


def save_stats_cache(path: Path, stats: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    np.savez_compressed(path, **stats, metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_))


def cache_matches(cached_meta: Dict[str, Any], current_meta: Dict[str, Any]) -> bool:
    keys = (
        "feature_schema",
        "feature_transform",
        "feature_transform_policy",
        "feature_transform_spec_hash",
        "feature_transform_warmup_rows",
        "feature_dim_core",
        "feature_dim_total",
        "feature_names_hash",
        "aux_dim",
        "aux_transform",
        "label_trim_schema",
        "low_abs_trim_fraction",
        "high_abs_trim_fraction",
        "horizons_ms",
        "split_protocol",
        "train_week_keys",
        "train_ts_start",
        "train_ts_end",
        "decision_time_basis",
        "trade_history_enabled",
        "event_stream_mode",
        "target_transform",
        "label_units",
        "target_task",
        "loss_weighting_schema",
        "ranking_schema",
    )
    if not all(cached_meta.get(k) == current_meta.get(k) for k in keys):
        return False

    expected_band_q = [float(x) for x in BAND_DIAG_QUANTILES]
    cached_band_q = cached_meta.get("band_diag_quantiles")
    current_band_q = current_meta.get("band_diag_quantiles")

    if current_band_q != expected_band_q:
        return False
    if cached_band_q != expected_band_q:
        return False

    return True


def _safe_pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    x0 = x - x.mean(); y0 = y - y.mean()
    den = np.sqrt((x0*x0).sum() * (y0*y0).sum())
    if den <= 0:
        return float('nan')
    return float((x0*y0).sum()/den)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)
    i = 0
    while i < xs.size:
        j = i + 1
        while j < xs.size and xs[j] == xs[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _safe_spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float('nan')
    rx = _average_ranks(x.astype(np.float64, copy=False))
    ry = _average_ranks(y.astype(np.float64, copy=False))
    return _safe_pearson_np(rx, ry)


def _binary_auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size != y.size or s.size < 2:
        return float("nan")
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    pos_rank_sum = float(np.sum(ranks[y]))
    u = pos_rank_sum - (n_pos * (n_pos - 1) / 2.0)
    return float(u / (n_pos * n_neg))


def _balanced_acc_np(pred_bool: np.ndarray, true_bool: np.ndarray) -> float:
    p = np.asarray(pred_bool, dtype=bool)
    t = np.asarray(true_bool, dtype=bool)
    if p.size != t.size or p.size == 0:
        return float("nan")
    pos = (t == True)
    neg = (t == False)
    if int(np.sum(pos)) == 0 or int(np.sum(neg)) == 0:
        return float("nan")
    tpr = float(np.mean(p[pos] == t[pos]))
    tnr = float(np.mean(p[neg] == t[neg]))
    return 0.5 * (tpr + tnr)


def _safe_quantile_np(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


class GPUWindowBatchSource:
    def __init__(
        self,
        ds,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 12345,
        subset_max_rows: Optional[int] = None,
        row_stride: int = 1,
    ):
        if len(ds.stores) != 1:
            raise ValueError(f"GPUWindowBatchSource requires exactly one week/store, got {len(ds.stores)}")
        if ds.week_ids.size and not np.all(ds.week_ids == 0):
            raise ValueError("GPUWindowBatchSource requires ds.week_ids to contain only zeros for single-week datasets")
        if len(ds) > 0 and int(ds.row_idx.min()) < int(ds.lookback - 1):
            raise ValueError(
                f"GPUWindowBatchSource requires full-history rows only, min row_idx={int(ds.row_idx.min())}, lookback={int(ds.lookback)}"
            )

        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.row_stride = int(row_stride)
        if self.row_stride < 1:
            raise ValueError(f"row_stride must be >= 1, got {self.row_stride}")
        self.subset_max_rows = subset_max_rows
        self.is_shared_feature_view = False
        self.lookback = int(ds.lookback)
        self.num_horizons = int(ds.y.shape[1]) if ds.y.ndim == 2 else NUM_HORIZONS

        features_np = ds.stores[0].contiguous_features()
        features_np = np.ascontiguousarray(features_np, dtype=np.float32)
        features_fp32 = torch.from_numpy(features_np).to(
            device=device,
            dtype=torch.float32,
            non_blocking=False,
        )
        self.bf16_debug_enabled = bool(BF16_FEATURE_DEBUG and FEATURE_STORAGE_DTYPE_NAME == "bf16" and self.shuffle)
        self.features_debug_fp32 = features_fp32 if self.bf16_debug_enabled else None
        self.features = features_fp32.to(dtype=FEATURE_STORAGE_DTYPE)
        if self.features.dtype != FEATURE_STORAGE_DTYPE:
            raise ValueError(f"Expected {FEATURE_STORAGE_DTYPE_NAME} features, got {self.features.dtype}")
        self.row_idx = torch.from_numpy(ds.row_idx.astype(np.int64, copy=False)).to(device=device)
        self.y = torch.from_numpy(ds.y.astype(np.float32, copy=False)).to(device=device)
        self.offsets = torch.arange(self.lookback - 1, -1, -1, device=device, dtype=torch.long)
        if self.y.ndim != 2 or self.y.shape[1] != NUM_HORIZONS:
            raise ValueError(f"Expected y to have shape [N, {NUM_HORIZONS}], got {tuple(self.y.shape)}")

        if subset_max_rows is not None:
            n = min(int(subset_max_rows), int(self.y.shape[0]))
            if n > 0:
                idx = torch.linspace(0, int(self.y.shape[0]) - 1, steps=n, device=device).round().long()
                self.row_idx = self.row_idx[idx]
                self.y = self.y[idx]

        self.n_rows = int(self.y.shape[0])
        self.effective_rows_nominal = (
            int((self.n_rows + self.row_stride - 1) // self.row_stride)
            if self.row_stride > 1
            else int(self.n_rows)
        )
        self.feature_shape = tuple(self.features.shape)
        self.feature_gb = float(self.features.numel() * self.features.element_size()) / (1024 ** 3)
        self.label_index_gb = float(
            self.y.numel() * self.y.element_size() + self.row_idx.numel() * self.row_idx.element_size()
        ) / (1024 ** 3)

    def __len__(self) -> int:
        n = int(self.effective_rows_nominal)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def make_evenly_spaced_subset(self, max_rows: int) -> "GPUWindowBatchSource":
        if int(max_rows) <= 0:
            raise ValueError(f"max_rows must be > 0, got {max_rows}")
        if int(self.n_rows) <= 0:
            raise ValueError("Cannot subset an empty source")
        n = min(int(max_rows), int(self.n_rows))
        idx = torch.linspace(0, self.n_rows - 1, steps=n, device=self.device).round().long()
        child = object.__new__(GPUWindowBatchSource)
        child.features = self.features
        child.features_debug_fp32 = None
        child.bf16_debug_enabled = False
        child.offsets = self.offsets
        child.device = self.device
        child.lookback = self.lookback
        child.batch_size = self.batch_size
        child.shuffle = False
        child.drop_last = False
        child.seed = self.seed
        child.row_stride = 1
        child.num_horizons = self.num_horizons
        child.row_idx = self.row_idx[idx]
        child.y = self.y[idx]
        if child.row_idx.numel() > 0 and int(child.row_idx.min().item()) < int(child.lookback - 1):
            raise ValueError(
                f"Subset violates full-history invariant: min row_idx={int(child.row_idx.min().item())}, lookback={int(child.lookback)}"
            )
        child.n_rows = int(child.y.shape[0])
        child.effective_rows_nominal = int(child.n_rows)
        child.feature_shape = tuple(child.features.shape)
        child.feature_gb = 0.0
        child.label_index_gb = float(
            child.y.numel() * child.y.element_size() + child.row_idx.numel() * child.row_idx.element_size()
        ) / (1024 ** 3)
        child.subset_max_rows = int(max_rows)
        child.is_shared_feature_view = True
        if child.features is not self.features:
            raise RuntimeError("Subset source must share exact feature tensor object")
        return child

    def iter_epoch(self, epoch: int):
        n_total = int(self.n_rows)
        stride = int(self.row_stride)

        if stride <= 1:
            offset = 0
        else:
            offset = int(epoch) % int(stride)
        selected = torch.arange(offset, n_total, stride, device=self.device)

        n = int(selected.numel())
        if self.shuffle:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed + int(epoch))
            order = torch.randperm(n, device=self.device, generator=g)
            perm = selected[order]
        else:
            perm = selected

        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, stop, self.batch_size):
            end = min(start + self.batch_size, stop)
            ids = perm[start:end]
            rows = self.row_idx[ids]
            win_idx = rows[:, None] - self.offsets[None, :]
            x = self.features[win_idx]
            if self.features_debug_fp32 is not None:
                x_fp32 = self.features_debug_fp32[win_idx]
                maybe_print_bf16_feature_debug(x_fp32, x, enabled=self.bf16_debug_enabled)
            y_raw = self.y[ids]
            expected_b = int(end - start)
            if x.shape != (expected_b, self.lookback, self.features.shape[1]):
                raise ValueError(f"Window tensor shape mismatch: got {tuple(x.shape)}")
            if y_raw.ndim != 2 or y_raw.shape[1] != NUM_HORIZONS:
                raise ValueError(f"Label tensor shape mismatch: got {tuple(y_raw.shape)}")
            yield x, y_raw


class CPUWindowBatchSource:
    def __init__(
        self,
        ds,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 12345,
        subset_max_rows: Optional[int] = None,
        row_stride: int = 1,
    ):
        if len(ds.stores) != 1:
            raise ValueError(f"CPUWindowBatchSource requires exactly one week/store, got {len(ds.stores)}")
        if ds.week_ids.size and not np.all(ds.week_ids == 0):
            raise ValueError("CPUWindowBatchSource requires ds.week_ids to contain only zeros for single-week datasets")
        if len(ds) > 0 and int(ds.row_idx.min()) < int(ds.lookback - 1):
            raise ValueError(
                f"CPUWindowBatchSource requires full-history rows only, min row_idx={int(ds.row_idx.min())}, lookback={int(ds.lookback)}"
            )

        self.target_device = device
        self.device = torch.device("cpu")
        self.pin_memory = bool(device.type == "cuda")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.row_stride = int(row_stride)
        if self.row_stride < 1:
            raise ValueError(f"row_stride must be >= 1, got {self.row_stride}")
        self.subset_max_rows = subset_max_rows
        self.is_shared_feature_view = False
        self.lookback = int(ds.lookback)
        self.num_horizons = int(ds.y.shape[1]) if ds.y.ndim == 2 else NUM_HORIZONS

        features_np = ds.stores[0].contiguous_features()
        features_np = np.ascontiguousarray(features_np, dtype=np.float32)

        features_fp32_cpu = torch.from_numpy(features_np).to(dtype=torch.float32)
        self.bf16_debug_enabled = bool(BF16_FEATURE_DEBUG and FEATURE_STORAGE_DTYPE_NAME == "bf16" and self.shuffle)
        self.features_debug_fp32 = features_fp32_cpu if self.bf16_debug_enabled else None
        features_cpu = features_fp32_cpu.to(dtype=FEATURE_STORAGE_DTYPE)
        if self.pin_memory:
            features_cpu = features_cpu.pin_memory()

        self.features = features_cpu
        if self.features.dtype != FEATURE_STORAGE_DTYPE:
            raise ValueError(f"Expected {FEATURE_STORAGE_DTYPE_NAME} features, got {self.features.dtype}")

        row_idx_cpu = torch.from_numpy(ds.row_idx.astype(np.int64, copy=False))
        y_cpu = torch.from_numpy(ds.y.astype(np.float32, copy=False))
        if self.pin_memory:
            row_idx_cpu = row_idx_cpu.pin_memory()
            y_cpu = y_cpu.pin_memory()
        self.row_idx = row_idx_cpu
        self.y = y_cpu
        self.offsets = torch.arange(self.lookback - 1, -1, -1, dtype=torch.long)
        if self.y.ndim != 2 or self.y.shape[1] != NUM_HORIZONS:
            raise ValueError(f"Expected y to have shape [N, {NUM_HORIZONS}], got {tuple(self.y.shape)}")

        if subset_max_rows is not None:
            n = min(int(subset_max_rows), int(self.y.shape[0]))
            if n > 0:
                idx = torch.linspace(0, int(self.y.shape[0]) - 1, steps=n).round().long()
                self.row_idx = self.row_idx[idx]
                self.y = self.y[idx]
                if self.pin_memory:
                    self.row_idx = self.row_idx.pin_memory()
                    self.y = self.y.pin_memory()

        self.n_rows = int(self.y.shape[0])
        self.effective_rows_nominal = (
            int((self.n_rows + self.row_stride - 1) // self.row_stride)
            if self.row_stride > 1
            else int(self.n_rows)
        )
        self.feature_shape = tuple(self.features.shape)
        self.feature_gb = float(self.features.numel() * self.features.element_size()) / (1024 ** 3)
        self.label_index_gb = float(
            self.y.numel() * self.y.element_size() + self.row_idx.numel() * self.row_idx.element_size()
        ) / (1024 ** 3)

    def __len__(self) -> int:
        n = int(self.effective_rows_nominal)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def make_evenly_spaced_subset(self, max_rows: int) -> "CPUWindowBatchSource":
        if int(max_rows) <= 0:
            raise ValueError(f"max_rows must be > 0, got {max_rows}")
        if int(self.n_rows) <= 0:
            raise ValueError("Cannot subset an empty source")

        n = min(int(max_rows), int(self.n_rows))
        idx = torch.linspace(0, self.n_rows - 1, steps=n).round().long()

        child = object.__new__(CPUWindowBatchSource)
        child.features = self.features
        child.features_debug_fp32 = None
        child.bf16_debug_enabled = False
        child.offsets = self.offsets
        child.target_device = self.target_device
        child.device = self.device
        child.pin_memory = self.pin_memory
        child.lookback = self.lookback
        child.batch_size = self.batch_size
        child.shuffle = False
        child.drop_last = False
        child.seed = self.seed
        child.row_stride = 1
        child.num_horizons = self.num_horizons
        child.row_idx = self.row_idx[idx]
        child.y = self.y[idx]
        if child.pin_memory:
            child.row_idx = child.row_idx.pin_memory()
            child.y = child.y.pin_memory()
        if child.row_idx.numel() > 0 and int(child.row_idx.min().item()) < int(child.lookback - 1):
            raise ValueError(
                f"Subset violates full-history invariant: min row_idx={int(child.row_idx.min().item())}, lookback={int(child.lookback)}"
            )
        child.n_rows = int(child.y.shape[0])
        child.effective_rows_nominal = int(child.n_rows)
        child.feature_shape = tuple(child.features.shape)
        child.feature_gb = float(child.features.numel() * child.features.element_size()) / (1024 ** 3)
        child.label_index_gb = float(
            child.y.numel() * child.y.element_size() + child.row_idx.numel() * child.row_idx.element_size()
        ) / (1024 ** 3)
        child.subset_max_rows = int(max_rows)
        child.is_shared_feature_view = True

        if child.features is not self.features:
            raise RuntimeError("Subset source must share exact feature tensor object")

        return child

    def iter_epoch(self, epoch: int):
        n_total = int(self.n_rows)
        stride = int(self.row_stride)

        if stride <= 1:
            offset = 0
        else:
            offset = int(epoch) % int(stride)
        selected = torch.arange(offset, n_total, stride)

        n = int(selected.numel())
        if self.shuffle:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed + int(epoch))
            order = torch.randperm(n, generator=g)
            perm = selected[order]
        else:
            perm = selected

        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, stop, self.batch_size):
            end = min(start + self.batch_size, stop)
            ids = perm[start:end]
            rows = self.row_idx[ids]
            win_idx = rows[:, None] - self.offsets[None, :]
            x_cpu = self.features[win_idx]
            if self.features_debug_fp32 is not None:
                x_fp32_cpu = self.features_debug_fp32[win_idx]
                maybe_print_bf16_feature_debug(x_fp32_cpu, x_cpu, enabled=self.bf16_debug_enabled)
            y_cpu = self.y[ids]
            if self.pin_memory:
                x_cpu = x_cpu.pin_memory()
                y_cpu = y_cpu.pin_memory()
            x = x_cpu.to(self.target_device, non_blocking=self.pin_memory)
            y_raw = y_cpu.to(self.target_device, non_blocking=self.pin_memory)
            expected_b = int(end - start)
            if x.shape != (expected_b, self.lookback, self.features.shape[1]):
                raise ValueError(f"Window tensor shape mismatch: got {tuple(x.shape)}")
            if y_raw.ndim != 2 or y_raw.shape[1] != NUM_HORIZONS:
                raise ValueError(f"Label tensor shape mismatch: got {tuple(y_raw.shape)}")
            yield x, y_raw


class MultiWeekTrainBatchSource:
    def __init__(self, sources: List[Any], seed: int = 12345):
        if not sources:
            raise ValueError("MultiWeekTrainBatchSource requires at least one source")
        self.sources = list(sources)
        self.seed = int(seed)
        self.n_rows = sum(src.n_rows for src in self.sources)
        self.effective_rows_nominal = sum(src.effective_rows_nominal for src in self.sources)
        self.batch_size = self.sources[0].batch_size
        for src in self.sources:
            if src.batch_size != self.batch_size:
                raise ValueError("All train sources must use the same batch size")

    def __len__(self) -> int:
        return sum(len(src) for src in self.sources)

    def iter_epoch(self, epoch: int):
        rng = np.random.default_rng(self.seed + int(epoch))
        source_order = []
        for i, src in enumerate(self.sources):
            source_order.extend([i] * len(src))
        source_order = np.asarray(source_order, dtype=np.int64)
        rng.shuffle(source_order)

        iters = [src.iter_epoch(epoch) for src in self.sources]
        exhausted = [False] * len(self.sources)
        for i in source_order.tolist():
            if exhausted[i]:
                continue
            try:
                yield next(iters[i])
            except StopIteration:
                exhausted[i] = True




class MultiSourceEvalBatchSource:
    def __init__(self, sources: List[Any]):
        if not sources:
            raise ValueError("MultiSourceEvalBatchSource requires at least one source")
        self.sources = list(sources)
        self.n_rows = sum(int(src.n_rows) for src in self.sources)
        self.effective_rows_nominal = sum(int(getattr(src, "effective_rows_nominal", src.n_rows)) for src in self.sources)

    def __len__(self) -> int:
        return sum(len(src) for src in self.sources)

    def iter_epoch(self, epoch: int):
        for src in self.sources:
            yield from src.iter_epoch(epoch)


def make_train_band_eval_source(train_sources: List[Any], max_rows: int) -> Any:
    if not train_sources:
        raise ValueError("No train sources available for band diagnostics")
    max_rows = max(1, int(max_rows))
    if len(train_sources) == 1:
        return train_sources[0].make_evenly_spaced_subset(max_rows)
    total_rows = sum(max(0, int(src.n_rows)) for src in train_sources)
    if total_rows <= 0:
        raise ValueError("Cannot create train band diagnostic source from empty train sources")
    target = min(max_rows, total_rows)
    raw = np.asarray([int(src.n_rows) for src in train_sources], dtype=np.float64) * (float(target) / float(total_rows))
    alloc = np.floor(raw).astype(np.int64)
    for i, src in enumerate(train_sources):
        if int(src.n_rows) > 0 and alloc[i] == 0 and int(alloc.sum()) < target:
            alloc[i] = 1
    remainder = int(target - int(alloc.sum()))
    if remainder > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for i in order.tolist():
            if remainder <= 0:
                break
            room = int(train_sources[i].n_rows) - int(alloc[i])
            if room > 0:
                add = min(room, remainder)
                alloc[i] += add
                remainder -= add
    elif remainder < 0:
        order = np.argsort(raw - np.floor(raw))
        to_remove = -remainder
        for i in order.tolist():
            if to_remove <= 0:
                break
            min_keep = 1 if int(train_sources[i].n_rows) > 0 else 0
            removable = max(0, int(alloc[i]) - min_keep)
            rem = min(removable, to_remove)
            alloc[i] -= rem
            to_remove -= rem
    children = [src.make_evenly_spaced_subset(int(n)) for src, n in zip(train_sources, alloc.tolist()) if int(n) > 0]
    return MultiSourceEvalBatchSource(children)

class LossEmaState:
    def __init__(self, decay: float):
        self.decay = float(decay)
        self.values: Dict[str, torch.Tensor] = {}

    def update(self, name: str, value: torch.Tensor, valid: bool = True) -> None:
        if not valid:
            return
        v = value.detach()
        if not torch.isfinite(v):
            return
        if name not in self.values:
            self.values[name] = v
        else:
            self.values[name] = self.values[name] * self.decay + v * (1.0 - self.decay)

    def denom(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        if name not in self.values:
            return torch.ones((), device=ref.device, dtype=ref.dtype)
        return self.values[name].to(device=ref.device, dtype=ref.dtype).clamp_min(1e-6)


def compute_dir_class_weights_from_train_labels(
    y: np.ndarray,
    *,
    pos_lo: np.ndarray,
    pos_hi: np.ndarray,
    neg_lo: np.ndarray,
    neg_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    keep_pos, keep_neg, _keep_signed = build_signed_side_trim_masks_np(
        y,
        pos_lo_raw_bps=pos_lo,
        pos_hi_raw_bps=pos_hi,
        neg_lo_abs_bps=neg_lo,
        neg_hi_abs_bps=neg_hi,
    )

    pos_count = keep_pos.sum(axis=0).astype(np.float64)
    neg_count = keep_neg.sum(axis=0).astype(np.float64)
    total = pos_count + neg_count

    if np.any(pos_count <= 0.0) or np.any(neg_count <= 0.0):
        raise ValueError(
            f"Direction class weights require positive and negative signed kept rows, "
            f"got pos_count={pos_count.tolist()} neg_count={neg_count.tolist()}"
        )

    pos_w = (total / (2.0 * np.maximum(pos_count, 1.0))) ** DIR_CLASS_WEIGHT_TEMPER
    neg_w = (total / (2.0 * np.maximum(neg_count, 1.0))) ** DIR_CLASS_WEIGHT_TEMPER

    pos_w = np.clip(pos_w, DIR_CLASS_WEIGHT_MIN, DIR_CLASS_WEIGHT_MAX)
    neg_w = np.clip(neg_w, DIR_CLASS_WEIGHT_MIN, DIR_CLASS_WEIGHT_MAX)

    return pos_w.astype(np.float32), neg_w.astype(np.float32)



def compute_mag_init_targets_from_train_labels(
    y: np.ndarray,
    *,
    pos_lo: np.ndarray,
    pos_hi: np.ndarray,
    neg_lo: np.ndarray,
    neg_hi: np.ndarray,
    pos_q50: np.ndarray,
    neg_q50: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    pos_q50 = np.asarray(pos_q50, dtype=np.float32).reshape(-1)
    neg_q50 = np.asarray(neg_q50, dtype=np.float32).reshape(-1)
    if y.ndim != 2 or y.shape[1] != NUM_HORIZONS:
        raise ValueError(f"y must have shape [N, {NUM_HORIZONS}], got {y.shape}")
    keep_pos, keep_neg, _ = build_signed_side_trim_masks_np(
        y,
        pos_lo_raw_bps=pos_lo,
        pos_hi_raw_bps=pos_hi,
        neg_lo_abs_bps=neg_lo,
        neg_hi_abs_bps=neg_hi,
    )
    pos_target_sqrt = np.empty((NUM_HORIZONS,), dtype=np.float32)
    neg_target_sqrt = np.empty((NUM_HORIZONS,), dtype=np.float32)
    for h in range(NUM_HORIZONS):
        pos_vals = np.sqrt(y[keep_pos[:, h], h].astype(np.float64, copy=False))
        neg_vals = np.sqrt((-y[keep_neg[:, h], h]).astype(np.float64, copy=False))
        if int(pos_vals.size) >= 100:
            pos_val = float(np.median(pos_vals))
        else:
            pos_val = float(np.sqrt(max(float(pos_q50[h]), 1e-8)))
        if int(neg_vals.size) >= 100:
            neg_val = float(np.median(neg_vals))
        else:
            neg_val = float(np.sqrt(max(float(neg_q50[h]), 1e-8)))
        pos_target_sqrt[h] = np.float32(np.clip(pos_val, 1e-4, 10.0))
        neg_target_sqrt[h] = np.float32(np.clip(neg_val, 1e-4, 10.0))
    return pos_target_sqrt.astype(np.float32), neg_target_sqrt.astype(np.float32)


def compute_dir_mag_loss(
    pred: Dict[str, torch.Tensor],
    y_raw: torch.Tensor,
    *,
    pos_lo_t: torch.Tensor,
    pos_hi_t: torch.Tensor,
    neg_lo_t: torch.Tensor,
    neg_hi_t: torch.Tensor,
    q50_t: torch.Tensor,
    q85_t: torch.Tensor,
    hwt: torch.Tensor,
    dir_pos_w_t: torch.Tensor,
    dir_neg_w_t: torch.Tensor,
    ema_state: LossEmaState,
    update_ema: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    abs_raw = torch.abs(y_raw)
    keep_pos, keep_neg, keep_signed = build_signed_side_trim_masks_torch(
        y_raw,
        pos_lo_t=pos_lo_t,
        pos_hi_t=pos_hi_t,
        neg_lo_t=neg_lo_t,
        neg_hi_t=neg_hi_t,
    )
    tau_t = torch.clamp(0.10 * (q85_t - q50_t), min=0.05)
    active_w = torch.sigmoid((abs_raw - q50_t) / tau_t)
    strong_w = torch.sigmoid((abs_raw - q85_t) / tau_t)

    dir_logits = pred["dir_logits"]
    dir_target = (y_raw > 0.0).to(dtype=dir_logits.dtype)
    dir_class_w = torch.where(dir_target > 0.5, dir_pos_w_t, dir_neg_w_t)
    dir_w = keep_signed.to(dtype=dir_logits.dtype) * hwt * dir_class_w
    dir_weight_sum = dir_w.sum(dim=0)
    if bool((dir_weight_sum.detach() <= 0.0).any().item()):
        raise ValueError("Direction loss has zero effective weight; check signed nonzero keep masks and training data.")
    dir_bce_raw = F.binary_cross_entropy_with_logits(dir_logits, dir_target, reduction="none")
    dir_loss_per_h = (dir_bce_raw * dir_w).sum(dim=0) / dir_weight_sum.clamp_min(1.0)
    dir_bce = dir_loss_per_h.mean()

    mag_up_target_sqrt = torch.sqrt(torch.clamp(y_raw, min=0.0))
    mag_down_target_sqrt = torch.sqrt(torch.clamp(-y_raw, min=0.0))
    mag_shape_w = torch.clamp(0.50 + 0.50 * active_w + 0.25 * strong_w, min=0.50, max=1.25)
    mag_w = hwt * mag_shape_w
    up_d = F.huber_loss(pred["mag_up_sqrt"], mag_up_target_sqrt, delta=1.0, reduction="none")
    down_d = F.huber_loss(pred["mag_down_sqrt"], mag_down_target_sqrt, delta=1.0, reduction="none")
    up_w = keep_pos.to(dtype=dir_logits.dtype) * mag_w
    down_w = keep_neg.to(dtype=dir_logits.dtype) * mag_w
    up_den = up_w.sum()
    down_den = down_w.sum()
    up_valid = up_den.detach() > 0
    down_valid = down_den.detach() > 0
    mag_side_terms = []
    if bool(up_valid.item()):
        mag_up_huber = (up_d * up_w).sum() / up_den.clamp_min(1e-9)
        mag_side_terms.append(mag_up_huber)
    else:
        mag_up_huber = dir_logits.sum() * 0.0
    if bool(down_valid.item()):
        mag_down_huber = (down_d * down_w).sum() / down_den.clamp_min(1e-9)
        mag_side_terms.append(mag_down_huber)
    else:
        mag_down_huber = dir_logits.sum() * 0.0
    if mag_side_terms:
        mag_huber = torch.stack(mag_side_terms).mean()
    else:
        mag_huber = dir_logits.sum() * 0.0
    mag_den = up_den + down_den
    mag_huber_valid = bool(up_valid.item()) or bool(down_valid.item())

    mag_corr = torch.zeros((), device=y_raw.device, dtype=y_raw.dtype)
    mag_corr_valid = False

    if update_ema:
        ema_state.update("dir_bce", dir_bce, valid=True)
        ema_state.update("mag_huber", mag_huber, valid=mag_huber_valid)
        ema_state.update("mag_corr", mag_corr, valid=mag_corr_valid)
    dir_norm = dir_bce / ema_state.denom("dir_bce", dir_bce)
    mag_norm = mag_huber / ema_state.denom("mag_huber", mag_huber)
    corr_norm = mag_corr / ema_state.denom("mag_corr", mag_corr) if mag_corr_valid else mag_corr
    loss = (
        DIR_LOSS_WEIGHT * dir_norm
        + MAG_LOSS_WEIGHT * mag_norm
        + MAG_CORR_LOSS_WEIGHT * corr_norm
    )
    components = {
        "loss": loss.detach(),
        "dir_bce": dir_bce.detach(),
        "mag_huber": mag_huber.detach(),
        "mag_corr": mag_corr.detach(),
        "dir_norm": dir_norm.detach(),
        "mag_norm": mag_norm.detach(),
        "corr_norm": corr_norm.detach(),
        "dir_weight_sum": dir_weight_sum.sum().detach(),
        "mag_up_huber": mag_up_huber.detach(),
        "mag_down_huber": mag_down_huber.detach(),
        "mag_up_weight_sum": up_den.detach(),
        "mag_down_weight_sum": down_den.detach(),
        "mag_weight_sum": mag_den.detach(),
        "dir_mask_frac": keep_signed.float().mean().detach(),
        "pos_mask_frac": keep_pos.float().mean().detach(),
        "neg_mask_frac": keep_neg.float().mean().detach(),
        "zero_frac_batch": (y_raw == 0.0).float().mean().detach(),
    }
    return loss, components



def _nan_band_metrics(n: int, frac: float, edge_lo: float, edge_hi: float) -> Dict[str, Any]:
    fields = [
        "true_abs_p50", "true_abs_mean", "true_abs_p90", "true_pos_frac", "true_ret_mean_bps", "true_ret_std_bps",
        "dir_auc", "dir_bal_acc", "dir_acc", "dir_bce", "dir_logit_mean", "dir_logit_std", "dir_prob_mean",
        "dir_prob_std", "dir_pos_frac_pred", "dir_pos_frac_true", "pred_abs_p50", "pred_abs_p90",
        "true_abs_std", "pred_abs_std", "pred_abs_std_over_true_abs_std", "mag_spearman_abs",
        "edge_mean", "edge_std", "edge_abs_p50", "edge_abs_p90", "edge_pearson", "edge_spearman",
        "edge_sign_acc", "edge_sign_bal_acc", "edge_pos_frac", "realized_mean_when_edge_pos",
        "realized_mean_when_edge_neg", "realized_spread_edge_pos_minus_neg", "top_edge_abs_q90_threshold",
        "n_top_edge_abs_q90", "top_edge_abs_q90_realized_abs_mean", "top_edge_abs_q90_edge_spearman",
        "top_edge_abs_q90_edge_sign_bal_acc",
    ]
    d: Dict[str, Any] = {"n": int(n), "frac": float(frac), "edge_lo_bps": float(edge_lo), "edge_hi_bps": float(edge_hi)}
    for f in fields:
        d[f] = 0 if f == "n_top_edge_abs_q90" else float("nan")
    d["learnability_tag"] = "too_few" if n < 1000 else "ignore_candidate"
    return d


def summarize_label_band_audit(y_raw: np.ndarray, stats: Dict[str, np.ndarray], *, split_name: str) -> Dict[str, Any]:
    y_raw = np.asarray(y_raw, dtype=np.float32)
    abs_raw = np.abs(y_raw)
    _keep_pos, _keep_neg, keep = build_signed_side_trim_masks_from_stats_np(y_raw, stats)
    edges = np.asarray(stats["kept_abs_band_edges_raw_bps"], dtype=np.float32)
    out: Dict[str, Any] = {"split": split_name, "band_quantiles": [float(x) for x in BAND_DIAG_QUANTILES], "bands": {}}
    for h, horizon_ms in enumerate(HORIZONS_MS):
        h_bands: Dict[str, Any] = {}
        denom = max(1, int(np.sum(keep[:, h])))
        h_diag = {
            "zero_frac": float(np.mean(y_raw[:, h] == 0.0)),
            "nonzero_frac": float(np.mean(y_raw[:, h] != 0.0)),
            "signed_kept_frac": float(np.mean(keep[:, h])),
            "pos_kept_frac": float(np.mean(_keep_pos[:, h])),
            "neg_kept_frac": float(np.mean(_keep_neg[:, h])),
        }
        for b, name in enumerate(BAND_DIAG_NAMES):
            lo, hi = float(edges[h, b]), float(edges[h, b + 1])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                mask = np.zeros(y_raw.shape[0], dtype=bool)
            elif b == len(BAND_DIAG_NAMES) - 1:
                mask = keep[:, h] & (abs_raw[:, h] >= lo) & (abs_raw[:, h] <= hi)
            else:
                mask = keep[:, h] & (abs_raw[:, h] >= lo) & (abs_raw[:, h] < hi)
            n = int(np.sum(mask))
            vals = y_raw[:, h][mask]
            absv = abs_raw[:, h][mask]
            item = {
                "n": n, "frac": float(n / denom), "edge_lo_bps": lo, "edge_hi_bps": hi,
                "true_abs_p50": _safe_quantile_np(absv, 0.50),
                "true_abs_mean": float(np.mean(absv)) if n else float("nan"),
                "true_abs_p90": _safe_quantile_np(absv, 0.90),
                "true_pos_frac": float(np.mean(vals > 0)) if n else float("nan"),
                "true_ret_mean_bps": float(np.mean(vals)) if n else float("nan"),
                "true_ret_std_bps": float(np.std(vals, ddof=0)) if n else float("nan"),
            }
            h_bands[name] = item
            print(
                f"[band-label split={split_name} h={int(horizon_ms)} band={name}] "
                f"n={n} frac={item['frac']:.6f} abs_p50={item['true_abs_p50']:.6g} "
                f"abs_mean={item['true_abs_mean']:.6g} abs_p90={item['true_abs_p90']:.6g} "
                f"pos_frac={item['true_pos_frac']:.6f} ret_mean={item['true_ret_mean_bps']:.6g} "
                f"ret_std={item['true_ret_std_bps']:.6g}",
                flush=True,
            )
        out["bands"][str(int(horizon_ms))] = h_bands
        out.setdefault("zero_diagnostics", {})[str(int(horizon_ms))] = h_diag
    return out


def _stable_sigmoid_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float32)
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def summarize_band_metrics_from_arrays(
    *,
    dir_logits: np.ndarray,
    mag_up_sqrt: np.ndarray,
    mag_down_sqrt: np.ndarray,
    y_raw: np.ndarray,
    stats: Dict[str, np.ndarray],
    split_name: str,
) -> Dict[str, Any]:
    p_up = _stable_sigmoid_np(dir_logits)
    true_up = y_raw > 0
    pred_up = p_up >= 0.5
    mag_up_bps = mag_up_sqrt ** 2
    mag_down_bps = mag_down_sqrt ** 2
    mag_pred_sqrt = p_up * mag_up_sqrt + (1.0 - p_up) * mag_down_sqrt
    pred_abs_bps = mag_pred_sqrt ** 2
    edge_bps = p_up * mag_up_bps - (1.0 - p_up) * mag_down_bps
    edge_sign = edge_bps >= 0
    abs_raw = np.abs(y_raw)
    _keep_pos, _keep_neg, keep = build_signed_side_trim_masks_from_stats_np(y_raw, stats)
    edges = np.asarray(stats["kept_abs_band_edges_raw_bps"], dtype=np.float32)
    out: Dict[str, Any] = {"split": split_name, "band_quantiles": [float(x) for x in BAND_DIAG_QUANTILES], "bands": {}}
    for h, horizon_ms in enumerate(HORIZONS_MS):
        h_bands: Dict[str, Any] = {}
        denom = max(1, int(np.sum(keep[:, h])))
        h_diag = {
            "zero_frac": float(np.mean(y_raw[:, h] == 0.0)),
            "nonzero_frac": float(np.mean(y_raw[:, h] != 0.0)),
            "signed_kept_frac": float(np.mean(keep[:, h])),
            "pos_kept_frac": float(np.mean(_keep_pos[:, h])),
            "neg_kept_frac": float(np.mean(_keep_neg[:, h])),
        }
        for b, name in enumerate(BAND_DIAG_NAMES):
            lo, hi = float(edges[h, b]), float(edges[h, b + 1])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                mask = np.zeros(y_raw.shape[0], dtype=bool)
            elif b == len(BAND_DIAG_NAMES) - 1:
                mask = keep[:, h] & (abs_raw[:, h] >= lo) & (abs_raw[:, h] <= hi)
            else:
                mask = keep[:, h] & (abs_raw[:, h] >= lo) & (abs_raw[:, h] < hi)
            n = int(np.sum(mask))
            frac = float(n / denom)
            if n < 100:
                h_bands[name] = _nan_band_metrics(n, frac, lo, hi)
                continue
            yv = y_raw[:, h][mask]
            absv = abs_raw[:, h][mask]
            prob = p_up[:, h][mask]
            pred = pred_up[:, h][mask]
            truth = true_up[:, h][mask]
            logits = dir_logits[:, h][mask]
            pred_abs = pred_abs_bps[:, h][mask]
            edge = edge_bps[:, h][mask]
            esign = edge_sign[:, h][mask]
            edge_pos = edge >= 0
            edge_neg = ~edge_pos
            yt = truth.astype(np.float32)
            bce = -(yt * np.log(np.clip(prob, 1e-9, 1.0 - 1e-9)) + (1.0 - yt) * np.log(np.clip(1.0 - prob, 1e-9, 1.0 - 1e-9)))
            true_std = float(np.std(absv, ddof=0))
            pred_std = float(np.std(pred_abs, ddof=0))
            edge_abs = np.abs(edge)
            top_thr = _safe_quantile_np(edge_abs, 0.90)
            top_mask = edge_abs >= top_thr if math.isfinite(top_thr) else np.zeros(n, dtype=bool)
            n_top = int(np.sum(top_mask))
            edge_spearman = _safe_spearman_np(edge, yv)
            edge_sign_bal = _balanced_acc_np(esign, truth)
            dir_auc = _binary_auc_np(prob, truth)
            dir_prob_std = float(np.std(prob, ddof=0))
            if n < 1000:
                tag = "too_few"
            elif math.isfinite(edge_spearman) and edge_spearman >= 0.02 and edge_sign_bal >= 0.505:
                tag = "focus_candidate"
            elif math.isfinite(dir_auc) and dir_auc >= 0.505 and dir_prob_std >= 0.01:
                tag = "accept_candidate"
            else:
                tag = "ignore_candidate"
            h_bands[name] = {
                "n": n, "frac": frac, "edge_lo_bps": lo, "edge_hi_bps": hi,
                "true_abs_p50": _safe_quantile_np(absv, 0.50), "true_abs_mean": float(np.mean(absv)), "true_abs_p90": _safe_quantile_np(absv, 0.90),
                "true_pos_frac": float(np.mean(truth)), "true_ret_mean_bps": float(np.mean(yv)), "true_ret_std_bps": float(np.std(yv, ddof=0)),
                "dir_auc": dir_auc, "dir_bal_acc": _balanced_acc_np(pred, truth), "dir_acc": float(np.mean(pred == truth)), "dir_bce": float(np.mean(bce)),
                "dir_logit_mean": float(np.mean(logits)), "dir_logit_std": float(np.std(logits, ddof=0)), "dir_prob_mean": float(np.mean(prob)), "dir_prob_std": dir_prob_std,
                "dir_pos_frac_pred": float(np.mean(pred)), "dir_pos_frac_true": float(np.mean(truth)),
                "pred_abs_p50": _safe_quantile_np(pred_abs, 0.50), "pred_abs_p90": _safe_quantile_np(pred_abs, 0.90),
                "pred_abs_std": pred_std, "true_abs_std": true_std, "pred_abs_std_over_true_abs_std": pred_std / true_std if true_std > 0 else float("nan"),
                "mag_spearman_abs": _safe_spearman_np(pred_abs, absv),
                "edge_mean": float(np.mean(edge)), "edge_std": float(np.std(edge, ddof=0)), "edge_abs_p50": _safe_quantile_np(edge_abs, 0.50), "edge_abs_p90": _safe_quantile_np(edge_abs, 0.90),
                "edge_pearson": _safe_pearson_np(edge, yv), "edge_spearman": edge_spearman, "edge_sign_acc": float(np.mean(esign == truth)), "edge_sign_bal_acc": edge_sign_bal,
                "edge_pos_frac": float(np.mean(edge_pos)),
                "realized_mean_when_edge_pos": float(np.mean(yv[edge_pos])) if np.any(edge_pos) else float("nan"),
                "realized_mean_when_edge_neg": float(np.mean(yv[edge_neg])) if np.any(edge_neg) else float("nan"),
                "realized_spread_edge_pos_minus_neg": (float(np.mean(yv[edge_pos]) - np.mean(yv[edge_neg])) if np.any(edge_pos) and np.any(edge_neg) else float("nan")),
                "top_edge_abs_q90_threshold": top_thr, "n_top_edge_abs_q90": n_top,
                "top_edge_abs_q90_realized_abs_mean": float(np.mean(absv[top_mask])) if n_top else float("nan"),
                "top_edge_abs_q90_edge_spearman": _safe_spearman_np(edge[top_mask], yv[top_mask]) if n_top >= 2 else float("nan"),
                "top_edge_abs_q90_edge_sign_bal_acc": _balanced_acc_np(esign[top_mask], truth[top_mask]) if n_top >= 2 else float("nan"),
                "learnability_tag": tag,
            }
        out["bands"][str(int(horizon_ms))] = h_bands
        out.setdefault("zero_diagnostics", {})[str(int(horizon_ms))] = h_diag
    return out


def _fmt_band_float(x: Any) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "nan"
    return f"{xf:.6g}" if math.isfinite(xf) else "nan"


def print_band_metrics_summary(metrics: Dict[str, Any], *, split_name: str, epoch: int) -> None:
    tag = f"band-{split_name}"
    for horizon_ms, bands in metrics.get("bands", {}).items():
        for band_name, m in bands.items():
            print(
                f"[{tag} epoch={epoch + 1} h={horizon_ms} band={band_name}] "
                f"n={int(m.get('n', 0))} frac={_fmt_band_float(m.get('frac'))} "
                f"abs_p50={_fmt_band_float(m.get('true_abs_p50'))} pos_frac={_fmt_band_float(m.get('true_pos_frac'))} "
                f"dir_auc={_fmt_band_float(m.get('dir_auc'))} dir_bal={_fmt_band_float(m.get('dir_bal_acc'))} "
                f"prob_std={_fmt_band_float(m.get('dir_prob_std'))} edge_sp={_fmt_band_float(m.get('edge_spearman'))} "
                f"edge_bal={_fmt_band_float(m.get('edge_sign_bal_acc'))} edge_abs_p90={_fmt_band_float(m.get('edge_abs_p90'))} "
                f"pred_abs_std_ratio={_fmt_band_float(m.get('pred_abs_std_over_true_abs_std'))} "
                f"spread_pos_neg={_fmt_band_float(m.get('realized_spread_edge_pos_minus_neg'))} "
                f"tag={m.get('learnability_tag', 'unknown')}",
                flush=True,
            )


def save_band_metrics_jsonl(out_root: Path, metrics: Dict[str, Any], *, epoch: int, split_name: str) -> None:
    path = out_root / "cmssl17_band_metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": int(epoch + 1), "split": split_name, "metrics": metrics}, sort_keys=True, allow_nan=True) + "\n")

def summarize_metrics(
    model,
    source,
    device,
    stats,
    amp_enabled,
    amp_dtype,
    primary_only=False,
    epoch: int = 0,
    band_diag: bool = False,
    split_name: str = "eval",
):
    was_training = model.training
    model.eval()
    try:
        dir_parts, up_parts, down_parts, y_parts = [], [], [], []
        with torch.inference_mode():
            for x, y_raw in source.iter_epoch(epoch):
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    pred = model(x)
                if not isinstance(pred, dict) or set(pred.keys()) != {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}:
                    raise ValueError(f"Model output must be dict with fixed schema, got keys={list(pred.keys()) if isinstance(pred, dict) else type(pred)}")
                for key in pred:
                    if pred[key].shape != y_raw.shape:
                        raise ValueError(f"Shape mismatch for {key}: pred={tuple(pred[key].shape)} y_raw={tuple(y_raw.shape)}")
                d = derive_dir_mag_predictions(pred)
                dir_parts.append(pred["dir_logits"].detach().float().cpu().numpy())
                up_parts.append(pred["mag_up_sqrt"].detach().float().cpu().numpy())
                down_parts.append(pred["mag_down_sqrt"].detach().float().cpu().numpy())
                y_parts.append(y_raw.detach().float().cpu().numpy())
        n_eval_rows = int(sum(a.shape[0] for a in y_parts))
        out = {"primary_horizon_ms": int(PRIMARY_METRIC_HORIZON_MS), "n_eval_rows": n_eval_rows}
        if not y_parts:
            out["edge_spearman_q50plus"] = [float("nan")] * NUM_HORIZONS
            out["dir_auc_kept"] = [float("nan")] * NUM_HORIZONS
            out["dir_bal_acc_kept"] = [float("nan")] * NUM_HORIZONS
            out["dir_auc_q50plus"] = [float("nan")] * NUM_HORIZONS
            out["dir_bal_acc_q50plus"] = [float("nan")] * NUM_HORIZONS
            out["primary_dir_bal_acc"] = float("nan")
            out["primary_metric_guard_passed"] = False
        else:
            dir_logits = np.concatenate(dir_parts, axis=0)
            p_up = _stable_sigmoid_np(dir_logits)
            mag_up_sqrt = np.concatenate(up_parts, axis=0)
            mag_down_sqrt = np.concatenate(down_parts, axis=0)
            y_raw = np.concatenate(y_parts, axis=0)
            mag_up_bps = mag_up_sqrt ** 2
            mag_down_bps = mag_down_sqrt ** 2
            mag_pred_sqrt = p_up * mag_up_sqrt + (1.0 - p_up) * mag_down_sqrt
            edge_bps = p_up * mag_up_bps - (1.0 - p_up) * mag_down_bps
            abs_raw = np.abs(y_raw)
            keep_pos, keep_neg, keep = build_signed_side_trim_masks_from_stats_np(y_raw, stats)
            true_up = y_raw > 0
            pred_up = p_up >= 0.5
            if primary_only:
                out["edge_spearman_q50plus"] = [float("nan")] * NUM_HORIZONS
                out["dir_auc_kept"] = [float("nan")] * NUM_HORIZONS
                out["dir_bal_acc_kept"] = [float("nan")] * NUM_HORIZONS
                out["dir_auc_q50plus"] = [float("nan")] * NUM_HORIZONS
                out["dir_bal_acc_q50plus"] = [float("nan")] * NUM_HORIZONS
                h = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)
                kh = keep[:, h]
                q50plus = keep[:, h] & (abs_raw[:, h] >= float(stats['kept_q50_abs_raw_bps'][h]))
                out["dir_auc_kept"][h] = _binary_auc_np(p_up[:, h][kh], true_up[:, h][kh])
                out["dir_bal_acc_kept"][h] = _balanced_acc_np(pred_up[:, h][kh], true_up[:, h][kh])
                out["edge_spearman_q50plus"][h] = _safe_spearman_np(edge_bps[:, h][q50plus], y_raw[:, h][q50plus]) if int(np.sum(q50plus)) >= 2 else float("nan")
                out["dir_auc_q50plus"][h] = _binary_auc_np(p_up[:, h][q50plus], true_up[:, h][q50plus])
                out["dir_bal_acc_q50plus"][h] = _balanced_acc_np(pred_up[:, h][q50plus], true_up[:, h][q50plus]) if int(np.sum(q50plus)) >= 2 else float("nan")
                if PRIMARY_METRIC.startswith("dir_auc_kept"):
                    primary_guard_series = out["dir_bal_acc_kept"]
                elif PRIMARY_METRIC.startswith("dir_auc_q50plus"):
                    primary_guard_series = out["dir_bal_acc_q50plus"]
                elif PRIMARY_METRIC.startswith("edge_spearman_q50plus"):
                    primary_guard_series = out["dir_bal_acc_q50plus"]
                else:
                    raise ValueError(f"Unsupported PRIMARY_METRIC={PRIMARY_METRIC!r}")
                out["primary_dir_bal_acc"] = float(primary_guard_series[h])
                out["primary_metric_guard_passed"] = bool(math.isfinite(out["primary_dir_bal_acc"]) and out["primary_dir_bal_acc"] >= PRIMARY_DIR_BAL_ACC_GUARD)
                if band_diag:
                    out["band_metrics"] = summarize_band_metrics_from_arrays(
                        dir_logits=dir_logits,
                        mag_up_sqrt=mag_up_sqrt,
                        mag_down_sqrt=mag_down_sqrt,
                        y_raw=y_raw,
                        stats=stats,
                        split_name=split_name,
                    )
            else:
                keys = [
                    "dir_auc_q50plus", "dir_auc_q85plus", "dir_acc_q50plus", "dir_acc_q85plus", "dir_bal_acc_q50plus",
                    "dir_bal_acc_q85plus", "dir_pos_frac_pred_q50plus", "dir_pos_frac_true_q50plus", "dir_pos_frac_pred_q85plus",
                    "dir_pos_frac_true_q85plus", "dir_auc_kept", "dir_bal_acc_kept", "mag_spearman_abs_q50plus",
                    "mag_pearson_abs_q50plus", "mag_spearman_abs_q85plus", "mag_pearson_abs_q85plus", "true_abs_bps_p50_kept",
                    "true_abs_bps_p90_kept", "pred_abs_bps_p50_kept", "pred_abs_bps_p90_kept", "true_abs_bps_std_kept",
                    "pred_abs_bps_std_kept", "pred_abs_std_over_true_abs_std_kept", "pred_abs_p90_over_true_abs_p90_kept",
                    "mag_up_huber_pos_kept", "mag_down_huber_neg_kept", "mag_up_pred_bps_p50_pos_kept", "mag_down_pred_bps_p50_neg_kept",
                    "edge_pearson_all", "edge_spearman_all", "edge_pearson_kept", "edge_spearman_kept", "edge_pearson_q50plus",
                    "edge_spearman_q50plus", "edge_pearson_q85plus", "edge_spearman_q85plus", "edge_sign_acc_q50plus",
                    "edge_bal_sign_acc_q50plus", "edge_sign_acc_q85plus", "edge_bal_sign_acc_q85plus", "edge_pos_frac_q50plus",
                    "edge_pos_frac_q85plus", "edge_mean_kept", "edge_std_kept", "edge_abs_p50_kept", "edge_abs_p90_kept",
                    "val_dir_bce_kept", "val_mag_huber_kept",
                ]
                for k in keys:
                    out[k] = []
                true_abs_bps = abs_raw
                pred_abs_bps = mag_pred_sqrt ** 2
                for h in range(NUM_HORIZONS):
                    kh = keep[:, h]
                    q50plus = kh & (abs_raw[:, h] >= float(stats['kept_q50_abs_raw_bps'][h]))
                    q85plus = kh & (abs_raw[:, h] >= float(stats['kept_q85_abs_raw_bps'][h]))
                    out["dir_auc_q50plus"].append(_binary_auc_np(p_up[:, h][q50plus], true_up[:, h][q50plus]))
                    out["dir_auc_q85plus"].append(_binary_auc_np(p_up[:, h][q85plus], true_up[:, h][q85plus]))
                    out["dir_acc_q50plus"].append(float(np.mean(pred_up[:, h][q50plus] == true_up[:, h][q50plus])) if np.any(q50plus) else float("nan"))
                    out["dir_acc_q85plus"].append(float(np.mean(pred_up[:, h][q85plus] == true_up[:, h][q85plus])) if np.any(q85plus) else float("nan"))
                    out["dir_bal_acc_q50plus"].append(_balanced_acc_np(pred_up[:, h][q50plus], true_up[:, h][q50plus]))
                    out["dir_bal_acc_q85plus"].append(_balanced_acc_np(pred_up[:, h][q85plus], true_up[:, h][q85plus]))
                    out["dir_pos_frac_pred_q50plus"].append(float(np.mean(pred_up[:, h][q50plus])) if np.any(q50plus) else float("nan"))
                    out["dir_pos_frac_true_q50plus"].append(float(np.mean(true_up[:, h][q50plus])) if np.any(q50plus) else float("nan"))
                    out["dir_pos_frac_pred_q85plus"].append(float(np.mean(pred_up[:, h][q85plus])) if np.any(q85plus) else float("nan"))
                    out["dir_pos_frac_true_q85plus"].append(float(np.mean(true_up[:, h][q85plus])) if np.any(q85plus) else float("nan"))
                    out["dir_auc_kept"].append(_binary_auc_np(p_up[:, h][kh], true_up[:, h][kh]))
                    out["dir_bal_acc_kept"].append(_balanced_acc_np(pred_up[:, h][kh], true_up[:, h][kh]))
                    out["mag_spearman_abs_q50plus"].append(_safe_spearman_np(pred_abs_bps[:, h][q50plus], true_abs_bps[:, h][q50plus]))
                    out["mag_pearson_abs_q50plus"].append(_safe_pearson_np(pred_abs_bps[:, h][q50plus], true_abs_bps[:, h][q50plus]))
                    out["mag_spearman_abs_q85plus"].append(_safe_spearman_np(pred_abs_bps[:, h][q85plus], true_abs_bps[:, h][q85plus]))
                    out["mag_pearson_abs_q85plus"].append(_safe_pearson_np(pred_abs_bps[:, h][q85plus], true_abs_bps[:, h][q85plus]))
                    out["true_abs_bps_p50_kept"].append(_safe_quantile_np(true_abs_bps[:, h][kh], 0.50))
                    out["true_abs_bps_p90_kept"].append(_safe_quantile_np(true_abs_bps[:, h][kh], 0.90))
                    out["pred_abs_bps_p50_kept"].append(_safe_quantile_np(pred_abs_bps[:, h][kh], 0.50))
                    out["pred_abs_bps_p90_kept"].append(_safe_quantile_np(pred_abs_bps[:, h][kh], 0.90))
                    true_std = float(np.std(true_abs_bps[:, h][kh], ddof=0)) if np.any(kh) else float("nan")
                    pred_std = float(np.std(pred_abs_bps[:, h][kh], ddof=0)) if np.any(kh) else float("nan")
                    out["true_abs_bps_std_kept"].append(true_std)
                    out["pred_abs_bps_std_kept"].append(pred_std)
                    out["pred_abs_std_over_true_abs_std_kept"].append(pred_std / true_std if math.isfinite(true_std) and true_std > 0 else float("nan"))
                    true_p90 = out["true_abs_bps_p90_kept"][-1]
                    pred_p90 = out["pred_abs_bps_p90_kept"][-1]
                    out["pred_abs_p90_over_true_abs_p90_kept"].append(pred_p90 / true_p90 if math.isfinite(true_p90) and true_p90 > 0 else float("nan"))
                    pos_kept = kh & (y_raw[:, h] > 0)
                    neg_kept = kh & (y_raw[:, h] < 0)
                    out["mag_up_huber_pos_kept"].append(float(np.mean(np.where((mag_up_sqrt[:, h][pos_kept] - np.sqrt(abs_raw[:, h][pos_kept])) ** 2 <= 1.0, 0.5 * (mag_up_sqrt[:, h][pos_kept] - np.sqrt(abs_raw[:, h][pos_kept])) ** 2, np.abs(mag_up_sqrt[:, h][pos_kept] - np.sqrt(abs_raw[:, h][pos_kept])) - 0.5))) if np.any(pos_kept) else float("nan"))
                    out["mag_down_huber_neg_kept"].append(float(np.mean(np.where((mag_down_sqrt[:, h][neg_kept] - np.sqrt(abs_raw[:, h][neg_kept])) ** 2 <= 1.0, 0.5 * (mag_down_sqrt[:, h][neg_kept] - np.sqrt(abs_raw[:, h][neg_kept])) ** 2, np.abs(mag_down_sqrt[:, h][neg_kept] - np.sqrt(abs_raw[:, h][neg_kept])) - 0.5))) if np.any(neg_kept) else float("nan"))
                    out["mag_up_pred_bps_p50_pos_kept"].append(_safe_quantile_np(mag_up_bps[:, h][pos_kept], 0.50))
                    out["mag_down_pred_bps_p50_neg_kept"].append(_safe_quantile_np(mag_down_bps[:, h][neg_kept], 0.50))
                    out["edge_pearson_all"].append(_safe_pearson_np(edge_bps[:, h], y_raw[:, h]))
                    out["edge_spearman_all"].append(_safe_spearman_np(edge_bps[:, h], y_raw[:, h]))
                    out["edge_pearson_kept"].append(_safe_pearson_np(edge_bps[:, h][kh], y_raw[:, h][kh]))
                    out["edge_spearman_kept"].append(_safe_spearman_np(edge_bps[:, h][kh], y_raw[:, h][kh]))
                    out["edge_pearson_q50plus"].append(_safe_pearson_np(edge_bps[:, h][q50plus], y_raw[:, h][q50plus]))
                    out["edge_spearman_q50plus"].append(_safe_spearman_np(edge_bps[:, h][q50plus], y_raw[:, h][q50plus]))
                    out["edge_pearson_q85plus"].append(_safe_pearson_np(edge_bps[:, h][q85plus], y_raw[:, h][q85plus]))
                    out["edge_spearman_q85plus"].append(_safe_spearman_np(edge_bps[:, h][q85plus], y_raw[:, h][q85plus]))
                    edge_sign = edge_bps[:, h] >= 0
                    out["edge_sign_acc_q50plus"].append(float(np.mean(edge_sign[q50plus] == true_up[:, h][q50plus])) if np.any(q50plus) else float("nan"))
                    out["edge_bal_sign_acc_q50plus"].append(_balanced_acc_np(edge_sign[q50plus], true_up[:, h][q50plus]))
                    out["edge_sign_acc_q85plus"].append(float(np.mean(edge_sign[q85plus] == true_up[:, h][q85plus])) if np.any(q85plus) else float("nan"))
                    out["edge_bal_sign_acc_q85plus"].append(_balanced_acc_np(edge_sign[q85plus], true_up[:, h][q85plus]))
                    out["edge_pos_frac_q50plus"].append(float(np.mean(edge_sign[q50plus])) if np.any(q50plus) else float("nan"))
                    out["edge_pos_frac_q85plus"].append(float(np.mean(edge_sign[q85plus])) if np.any(q85plus) else float("nan"))
                    out["edge_mean_kept"].append(float(np.mean(edge_bps[:, h][kh])) if np.any(kh) else float("nan"))
                    out["edge_std_kept"].append(float(np.std(edge_bps[:, h][kh], ddof=0)) if np.any(kh) else float("nan"))
                    out["edge_abs_p50_kept"].append(_safe_quantile_np(np.abs(edge_bps[:, h][kh]), 0.50))
                    out["edge_abs_p90_kept"].append(_safe_quantile_np(np.abs(edge_bps[:, h][kh]), 0.90))
                    if np.any(kh):
                        yt = true_up[:, h][kh].astype(np.float32)
                        prob = p_up[:, h][kh]
                        bce = -(yt * np.log(np.clip(prob, 1e-9, 1.0)) + (1.0 - yt) * np.log(np.clip(1.0 - prob, 1e-9, 1.0)))
                        out["val_dir_bce_kept"].append(float(np.mean(bce)))
                        up_h = kh & (y_raw[:, h] > 0)
                        dn_h = kh & (y_raw[:, h] < 0)
                        errs = []
                        if np.any(up_h):
                            errs.append(mag_up_sqrt[:, h][up_h] - np.sqrt(abs_raw[:, h][up_h]))
                        if np.any(dn_h):
                            errs.append(mag_down_sqrt[:, h][dn_h] - np.sqrt(abs_raw[:, h][dn_h]))
                        if errs:
                            d = np.abs(np.concatenate(errs))
                            hub = np.where(d <= 1.0, 0.5 * d * d, d - 0.5)
                            out["val_mag_huber_kept"].append(float(np.mean(hub)))
                        else:
                            out["val_mag_huber_kept"].append(float("nan"))
                    else:
                        out["val_dir_bce_kept"].append(float("nan"))
                        out["val_mag_huber_kept"].append(float("nan"))
                primary_idx = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)
                if PRIMARY_METRIC.startswith("dir_auc_kept"):
                    primary_guard_series = out["dir_bal_acc_kept"]
                elif PRIMARY_METRIC.startswith("dir_auc_q50plus"):
                    primary_guard_series = out["dir_bal_acc_q50plus"]
                elif PRIMARY_METRIC.startswith("edge_spearman_q50plus"):
                    primary_guard_series = out["dir_bal_acc_q50plus"]
                else:
                    raise ValueError(f"Unsupported PRIMARY_METRIC={PRIMARY_METRIC!r}")
                out["primary_dir_bal_acc"] = float(primary_guard_series[primary_idx])
                out["primary_metric_guard_passed"] = bool(math.isfinite(out["primary_dir_bal_acc"]) and out["primary_dir_bal_acc"] >= PRIMARY_DIR_BAL_ACC_GUARD)
                if band_diag:
                    out["band_metrics"] = summarize_band_metrics_from_arrays(
                        dir_logits=dir_logits,
                        mag_up_sqrt=mag_up_sqrt,
                        mag_down_sqrt=mag_down_sqrt,
                        y_raw=y_raw,
                        stats=stats,
                        split_name=split_name,
                    )
        return out
    finally:
        model.train(was_training)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def get_model_state_dict(model: torch.nn.Module) -> dict:
    return unwrap_model(model).state_dict()


def _tensor_float_list(t: torch.Tensor) -> List[float]:
    return [float(v) for v in t.detach().float().cpu().tolist()]


def model_param_groups(model: torch.nn.Module) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    prefixes = [
        ("extractor_depatch", "depatch_proj_encoder.depatch."),
        ("extractor_patch_embed", "depatch_proj_encoder.output_linear."),
        ("extractor_ci", "depatch_proj_encoder.ci_encoder."),
        ("extractor_gate", "depatch_proj_encoder.feature_gate."),
        ("extractor_post_gate", "depatch_proj_encoder.post_gate_proj."),
        ("extractor_mixed", "depatch_proj_encoder.mixed_encoder."),
        ("extractor_final_proj", "depatch_proj_encoder.final_mixer."),
        ("extractor_output_norm", "depatch_proj_encoder.output_norm."),
        ("mamba", "mamba."),
        ("task_decoders", "dir_token_decoder."),
        ("task_decoders", "mag_token_decoder."),
        ("heads_pools", "dir_pool."),
        ("heads_pools", "mag_pool."),
        ("heads_pools", "dir_head."),
        ("heads_pools", "mag_up_head."),
        ("heads_pools", "mag_down_head."),
    ]
    groups = {k: [] for k in [
        "extractor_depatch", "extractor_patch_embed", "extractor_ci", "extractor_gate",
        "extractor_post_gate", "extractor_mixed", "extractor_final_proj", "extractor_output_norm", "mamba",
        "task_decoders", "heads_pools", "other",
    ]}
    for name, p in unwrap_model(model).named_parameters():
        matched = False
        for group, prefix in prefixes:
            if name.startswith(prefix):
                groups[group].append((name, p))
                matched = True
                break
        if not matched:
            groups["other"].append((name, p))
    return groups


def summarize_param_groups(model: torch.nn.Module) -> Dict[str, dict]:
    out = {}
    for group, params in model_param_groups(model).items():
        count = sum(p.numel() for _, p in params)
        req_count = sum(p.numel() for _, p in params if p.requires_grad)
        sq = 0.0
        abs_sum = 0.0
        for _, p in params:
            pf = p.detach().float()
            sq += float((pf * pf).sum().item())
            abs_sum += float(pf.abs().sum().item())
        out[group] = {
            "param_count": int(count),
            "param_norm": math.sqrt(sq),
            "param_abs_mean": abs_sum / max(1, count),
            "requires_grad_count": int(req_count),
        }
    return out


def _bounded_abs_sample_for_quantile(
    chunks: List[torch.Tensor],
    *,
    max_elems: int,
) -> Optional[torch.Tensor]:
    """Return a bounded deterministic sample of flattened abs-value chunks for quantile diagnostics.

    This is only for diagnostics. It avoids concatenating all gradients in large parameter groups.
    Sampling is deterministic stride sampling, not random, so logs are reproducible.
    """
    if not chunks:
        return None

    flat_chunks = [c.detach().reshape(-1) for c in chunks if c is not None and c.numel() > 0]
    if not flat_chunks:
        return None

    total = sum(int(c.numel()) for c in flat_chunks)
    if total <= max_elems:
        return torch.cat(flat_chunks)

    max_elems = int(max(1, max_elems))
    samples = []

    remaining_budget = max_elems
    remaining_total = total

    for c in flat_chunks:
        n = int(c.numel())
        if n <= 0 or remaining_budget <= 0:
            remaining_total -= n
            continue

        # Allocate a proportional deterministic sample budget to this chunk.
        take = int(round(remaining_budget * (n / max(1, remaining_total))))
        take = max(1, min(take, n, remaining_budget))

        if take >= n:
            samples.append(c)
        else:
            stride = max(1, n // take)
            samples.append(c[::stride][:take])

        remaining_budget -= take
        remaining_total -= n

    if not samples:
        return None

    return torch.cat(samples)




def build_optimizer_param_groups(model: torch.nn.Module, *, base_lr: float, gate_lr: float) -> List[dict]:
    base = unwrap_model(model)
    gate_params = []
    main_params = []
    trainable_params = []
    gate_name_part = "depatch_proj_encoder.feature_gate."
    for name, p in base.named_parameters():
        if not p.requires_grad:
            continue
        trainable_params.append(p)
        if gate_name_part in name:
            gate_params.append(p)
        else:
            main_params.append(p)
    if len(gate_params) <= 0:
        if os.environ.get("BYBIT_SUPPRESS_CMSSL_CONFIG_PRINTS", "0").strip() != "1":
            print("[gate-config] no feature_gate parameters found; gate_lr disabled", flush=True)
    if len(main_params) <= 0:
        raise ValueError("No trainable main parameters found for optimizer group.")
    gate_ids = {id(p) for p in gate_params}
    main_ids = {id(p) for p in main_params}
    if gate_ids & main_ids:
        raise ValueError("Optimizer parameter groups overlap.")
    all_group_ids = list(main_ids) + list(gate_ids)
    trainable_ids = [id(p) for p in trainable_params]
    if len(all_group_ids) != len(set(all_group_ids)) or set(all_group_ids) != set(trainable_ids):
        raise ValueError("Optimizer parameter groups must include every trainable parameter exactly once.")
    groups = [
        {
            "params": main_params,
            "lr": base_lr,
            "weight_decay": 1e-3,
            "name": "main",
        },
    ]
    if gate_params:
        groups.append(
            {
                "params": gate_params,
                "lr": gate_lr,
                "weight_decay": 0.0,
                "name": "gate",
            }
        )
    return groups


def summarize_grad_groups(model: torch.nn.Module, lr: float) -> Dict[str, dict]:
    out = {}
    for group, params in model_param_groups(model).items():
        param_count = sum(p.numel() for _, p in params)
        grad_param_count = 0
        grad_none_param_count = 0
        grad_sq = 0.0
        param_sq = 0.0
        grad_abs_sum = 0.0
        grad_numel = 0
        grad_abs_chunks = []
        for _, p in params:
            pf = p.detach().float()
            param_sq += float((pf * pf).sum().item())
            if p.grad is None:
                grad_none_param_count += p.numel()
                continue
            gf = p.grad.detach().float()
            grad_param_count += p.numel()
            grad_sq += float((gf * gf).sum().item())
            ga = gf.abs().reshape(-1)
            grad_abs_sum += float(ga.sum().item())
            grad_numel += ga.numel()
            grad_abs_chunks.append(ga)
        grad_norm = math.sqrt(grad_sq)
        param_norm = math.sqrt(param_sq)
        q_in = _bounded_abs_sample_for_quantile(
            grad_abs_chunks,
            max_elems=GRAD_DIAG_P95_MAX_ELEMS,
        )
        if q_in is not None and q_in.numel() > 0:
            grad_abs_p95 = float(torch.quantile(q_in, 0.95).cpu())
        else:
            grad_abs_p95 = 0.0
        out[group] = {
            "param_count": int(param_count),
            "grad_param_count": int(grad_param_count),
            "grad_none_param_count": int(grad_none_param_count),
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "lr_grad_over_param": float(lr) * grad_norm / (param_norm + 1e-12),
            "grad_abs_mean": grad_abs_sum / max(1, grad_numel),
            "grad_abs_p95": grad_abs_p95,
            "grad_abs_p95_sampled": bool(
                sum(int(c.numel()) for c in grad_abs_chunks) > GRAD_DIAG_P95_MAX_ELEMS
            ),
            "grad_abs_p95_elems": int(0 if q_in is None else q_in.numel()),
        }
    return out


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _pearson_small(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    am = af - af.mean()
    bm = bf - bf.mean()
    denom = torch.sqrt((am * am).mean() * (bm * bm).mean())
    if float(denom.cpu()) <= 1e-12:
        return float("nan")
    return float(((am * bm).mean() / denom).cpu())


@torch.no_grad()
def run_model_diagnostics(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_raw: torch.Tensor,
    *,
    epoch: int,
    batch_i: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    base = unwrap_model(model)
    was_training = base.training
    base.eval()
    x_diag = x[:MODEL_DIAG_MAX_BATCH].detach()
    y_diag = y_raw[:MODEL_DIAG_MAX_BATCH].detach()
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred_diag, diag = base.forward_with_diagnostics(x_diag)
    if was_training:
        base.train()

    y_sign = (y_diag.float() > 0).float()
    y_abs = y_diag.float().abs()
    probs = torch.sigmoid(pred_diag["dir_logits"].float())
    dir_corr = []
    prob_std = []
    for h in range(NUM_HORIZONS):
        dir_corr.append(_pearson_small(pred_diag["dir_logits"][:, h], y_sign[:, h]))
        prob_std.append(float(probs[:, h].std(unbiased=False).cpu()))
    diag["target_prediction"] = {
        "y_sign_pos_frac_by_horizon": _tensor_float_list(y_sign.mean(dim=0)),
        "y_abs_mean_by_horizon": _tensor_float_list(y_abs.mean(dim=0)),
        "dir_logit_pearson_by_horizon": dir_corr,
        "dir_prob_std_by_horizon": prob_std,
    }
    diag["epoch"] = int(epoch + 1)
    diag["batch"] = int(batch_i)
    return diag


def _fmt_ratio(grad_diag: Optional[Dict[str, dict]], key: str) -> str:
    if not grad_diag or key not in grad_diag:
        return "nan"
    return f"{grad_diag[key]['lr_grad_over_param']:.3e}"


def print_model_diagnostics(epoch: int, batch_i: int, diag: dict, grad_diag: Optional[Dict[str, dict]]) -> None:
    ext = diag["extractor"]
    ea = ext["activations"]
    ma = diag["activations"]
    ratios = ext.get("ratios", {})
    mratios = diag.get("ratios", {})
    gate = ext["gate"]
    dep = ext["depatch"]
    res = ext["residual_scalars"]
    pred = diag["prediction"]
    tp = diag["target_prediction"]
    print(
        f"[model-act] epoch={epoch+1} batch={batch_i} input_rms={ea['x_input']['rms']:.6g} "
        f"depatch_rms={ea['depatch_out']['rms']:.6g} ci_rms={ea['ci_out']['rms']:.6g} "
        f"gate_rms={ea['post_gate']['rms']:.6g} proj_rms={ea['post_proj']['rms']:.6g} "
        f"mixed_rms={ea['post_mixed']['rms']:.6g} mamba_rms={ma['mamba_tokens']['rms']:.6g} "
        f"fused_rms={ma['mamba_fused']['rms']:.6g}",
        flush=True,
    )
    print(
        f"[model-ratio] epoch={epoch+1} batch={batch_i} gate_over_ci={ratios.get('gate_over_ci_rms', float('nan')):.6g} "
        f"proj_over_flat={ratios.get('proj_over_flat_rms', float('nan')):.6g} "
        f"mixed_over_proj={ratios.get('mixed_over_proj_rms', float('nan')):.6g} "
        f"fused_over_tokens={mratios.get('fused_over_tokens_rms', float('nan')):.6g}",
        flush=True,
    )
    print(
        f"[model-gate] epoch={epoch+1} batch={batch_i} mean={gate['gate_mean']:.6g} std={gate['gate_std']:.6g} "
        f"p05={gate['gate_p05']:.6g} p50={gate['gate_p50']:.6g} p95={gate['gate_p95']:.6g} "
        f"frac_lt_0p5={gate['gate_frac_lt_0p5']:.6g} frac_gt_0p95={gate['gate_frac_gt_0p95']:.6g} "
        f"alpha={gate.get('alpha', float('nan')):.6g} prior_std={gate['prior_std']:.6g} dyn_std={gate['dyn_std']:.6g}",
        flush=True,
    )
    print(
        f"[model-offset] epoch={epoch+1} batch={batch_i} dx_std={dep['offset_dx_std']:.6g} "
        f"dx_abs_p95={dep['offset_dx_abs_p95']:.6g} span_mean={dep['span_samples_mean']:.6g} "
        f"span_p05={dep['span_samples_p05']:.6g} span_p95={dep['span_samples_p95']:.6g} "
        f"left_clip_frac={dep['bound_left_clip_frac']:.6g} right_clip_frac={dep['bound_right_clip_frac']:.6g}",
        flush=True,
    )
    print(
        f"[model-res] epoch={epoch+1} batch={batch_i} ci_a_mean={res['ci_res_a_mean']:.6g} "
        f"ci_a_absmax={res['ci_res_a_absmax']:.6g} mixed_a_mean={res['mixed_res_a_mean']:.6g} "
        f"mixed_a_absmax={res['mixed_res_a_absmax']:.6g}",
        flush=True,
    )
    corr = [float(v) for v in tp["dir_logit_pearson_by_horizon"]]
    print(
        f"[model-pred] epoch={epoch+1} batch={batch_i} dir_logit_std={pred['dir_logit_std']:.6g} "
        f"dir_prob_mean={pred['dir_prob_mean']:.6g} dir_prob_std={pred['dir_prob_std']:.6g} "
        f"mag_up_mean={pred['mag_up_mean']:.6g} mag_down_mean={pred['mag_down_mean']:.6g} "
        f"mag_up_raw_p50={pred.get('mag_up_raw_p50', float('nan')):.6g} "
        f"mag_down_raw_p50={pred.get('mag_down_raw_p50', float('nan')):.6g} "
        f"mag_up_floor_frac={pred.get('mag_up_floor_frac', float('nan')):.6g} "
        f"mag_down_floor_frac={pred.get('mag_down_floor_frac', float('nan')):.6g} "
        f"corr_dir_y={corr}",
        flush=True,
    )
    print(
        f"[model-grad] epoch={epoch+1} batch={batch_i} depatch={_fmt_ratio(grad_diag, 'extractor_depatch')} "
        f"patch={_fmt_ratio(grad_diag, 'extractor_patch_embed')} ci={_fmt_ratio(grad_diag, 'extractor_ci')} "
        f"gate={_fmt_ratio(grad_diag, 'extractor_gate')} post_gate={_fmt_ratio(grad_diag, 'extractor_post_gate')} "
        f"mixed={_fmt_ratio(grad_diag, 'extractor_mixed')} mamba={_fmt_ratio(grad_diag, 'mamba')} "
        f"decoders={_fmt_ratio(grad_diag, 'task_decoders')} heads={_fmt_ratio(grad_diag, 'heads_pools')}",
        flush=True,
    )



def _finite_stats(t: torch.Tensor) -> str:
    td = t.detach()
    finite = torch.isfinite(td)
    bad = int((~finite).sum().item())
    total = int(td.numel())
    if finite.any():
        vals = td[finite].float()
        return (
            f"shape={tuple(td.shape)} dtype={td.dtype} device={td.device} "
            f"bad={bad}/{total} "
            f"min={float(vals.min().item()):.6g} "
            f"max={float(vals.max().item()):.6g} "
            f"mean={float(vals.mean().item()):.6g} "
            f"absmax={float(vals.abs().max().item()):.6g}"
        )
    return (
        f"shape={tuple(td.shape)} dtype={td.dtype} device={td.device} "
        f"bad={bad}/{total} min=nan max=nan mean=nan absmax=nan"
    )


def _check_tensor_finite(name: str, t: torch.Tensor, *, epoch: int, batch: int, stage: str) -> None:
    if not FINITE_DEBUG:
        return
    if not torch.isfinite(t).all():
        raise FloatingPointError(
            f"[nonfinite-tensor] epoch={epoch+1} batch={batch} stage={stage} "
            f"name={name} {_finite_stats(t)}"
        )


def _check_pred_finite(pred: Dict[str, torch.Tensor], *, epoch: int, batch: int, stage: str) -> None:
    if not FINITE_DEBUG:
        return
    expected = {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
    if not isinstance(pred, dict) or set(pred.keys()) != expected:
        raise ValueError(
            f"[bad-pred] epoch={epoch+1} batch={batch} stage={stage} "
            f"type={type(pred)} keys={list(pred.keys()) if isinstance(pred, dict) else None}"
        )
    for k, v in pred.items():
        _check_tensor_finite(f"pred.{k}", v, epoch=epoch, batch=batch, stage=stage)


def _check_grads_finite(model: torch.nn.Module, *, epoch: int, batch: int, stage: str) -> float:
    if not FINITE_DEBUG:
        return float("nan")
    sq_sum = 0.0
    for name, p in unwrap_model(model).named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            raise FloatingPointError(
                f"[nonfinite-grad] epoch={epoch+1} batch={batch} stage={stage} "
                f"name={name} {_finite_stats(g)}"
            )
        gf = g.float()
        sq_sum += float((gf * gf).sum().item())
    return math.sqrt(sq_sum)


def _check_params_finite(model: torch.nn.Module, *, epoch: int, batch: int, stage: str) -> None:
    if not FINITE_DEBUG:
        return
    for name, p in unwrap_model(model).named_parameters():
        pd = p.detach()
        if not torch.isfinite(pd).all():
            raise FloatingPointError(
                f"[nonfinite-param] epoch={epoch+1} batch={batch} stage={stage} "
                f"name={name} {_finite_stats(pd)}"
            )


# ---------------- Train/Eval ----------------
def train_from_offline():
    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        try: torch.set_float32_matmul_precision(MATMUL_PRECISION)
        except Exception: pass
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amp_enabled = AMP_ENABLED and device.type=='cuda'
    amp_dtype = torch.bfloat16
    print(f"[config] BYBIT_TRAIN_ROW_STRIDE={TRAIN_ROW_STRIDE}", flush=True)
    out_root = Path(OUT_ROOT)
    meta = json.loads((out_root / "meta.json").read_text())
    validate_dataset_label_dim(meta, f"global metadata {out_root / 'meta.json'}")
    validate_contract_meta(meta, f"global metadata {out_root / 'meta.json'}")
    for rel_path in meta.get("weeks_meta", {}).values():
        wk_path = out_root / rel_path
        wk_meta = json.loads(wk_path.read_text())
        source = f"week metadata {rel_path}"
        validate_dataset_label_dim(wk_meta, source)
        validate_contract_meta(wk_meta, source)
        validate_week_matches_global(meta, wk_meta, source)
    print(
        f"[metadata-contract] schema={FEATURE_SCHEMA} "
        f"feature_transform={meta.get('feature_transform')} "
        f"feature_dim_core={meta.get('feature_dim_core')} "
        f"aux_dim={meta.get('aux_dim')} "
        f"feature_dim_total={meta.get('feature_dim_total')} "
        f"feature_names_hash={meta.get('feature_names_hash')}",
        flush=True,
    )
    trade_history_enabled = meta.get('trade_history_enabled')
    event_stream_mode = meta.get('event_stream_mode')
    split_info = require_supported_pipeline_splits(meta, out_root)
    protocol = split_info["protocol"]
    splits = split_info['splits']

    cmssl_train = splits['cmssl']['train']
    cmssl_val = splits['cmssl']['val']
    cmssl_test = splits["cmssl"].get("test")
    has_cmssl_test = isinstance(cmssl_test, dict)
    train_week_keys = list(cmssl_train["weeks"])
    if has_cmssl_test:
        print(
            f"[split] protocol={protocol} cmssl.train={','.join(train_week_keys)} "
            f"cmssl.val={cmssl_val['weeks']} cmssl.test={cmssl_test['weeks']}",
            flush=True,
        )
    else:
        print(
            f"[split] protocol={protocol} cmssl.train={','.join(train_week_keys)} "
            f"cmssl.val={cmssl_val['weeks']} cmssl.test=<missing>",
            flush=True,
        )
    train_split_entries = [
        make_single_week_split_from_meta(out_root=out_root, global_meta=meta, week_key=wk)
        for wk in train_week_keys
    ]
    ds_train_list = [build_dataset_from_split(str(out_root), entry) for entry in train_split_entries]
    ds_val = build_dataset_from_split(str(out_root), cmssl_val)
    ds_test = build_dataset_from_split(str(out_root), cmssl_test) if has_cmssl_test else None
    F_total = int(meta.get("feature_dim_total", 0))
    for i, ds_train in enumerate(ds_train_list):
        if F_total != int(ds_train.feature_dim_total):
            raise ValueError(f"Feature dimension mismatch: meta={F_total}, train_dataset={int(ds_train.feature_dim_total)}")
        if int(ds_train.lookback) != int(LOOKBACK):
            raise ValueError(f"LOOKBACK mismatch: config={LOOKBACK}, train_dataset={int(ds_train.lookback)}")
    split_items = [
        *[(f"train[{i}]/{train_week_keys[i]}", ds_train_i) for i, ds_train_i in enumerate(ds_train_list)],
        ("val", ds_val),
    ]
    if has_cmssl_test:
        split_items.append(("test", ds_test))
    for split_name, ds in split_items:
        if len(ds.stores) != 1:
            raise ValueError(f"{split_name} split must have exactly one store/week, got {len(ds.stores)}")
        if ds.week_ids.size and not np.all(ds.week_ids == 0):
            raise ValueError(f"{split_name} split week_ids must all be 0 for single-week protocol")
        if len(ds) > 0 and int(ds.row_idx.min()) < int(LOOKBACK - 1):
            raise ValueError(
                f"{split_name} split has rows without full history: min_row_idx={int(ds.row_idx.min())}, lookback={LOOKBACK}"
            )

    tr_start = int(min(entry["start"] for entry in train_split_entries))
    tr_end = int(max(entry["end"] for entry in train_split_entries))
    va_start,va_end=int(cmssl_val['start']),int(cmssl_val['end'])
    if has_cmssl_test:
        te_start,te_end=int(cmssl_test['start']),int(cmssl_test['end'])
    else:
        te_start,te_end=None,None

    cache_path=out_root/'signed_side_trim_stats_cache.npz'
    cache_meta={
        'feature_schema': meta.get('feature_schema'),
        'feature_transform': meta.get('feature_transform'),
        'feature_transform_policy': meta.get('feature_transform_policy'),
        'feature_transform_spec_hash': meta.get('feature_transform_spec_hash'),
        'feature_transform_warmup_rows': int(meta.get('feature_transform_warmup_rows', -1)),
        'feature_dim_core': int(meta.get('feature_dim_core', -1)),
        'feature_dim_total': int(meta.get('feature_dim_total', -1)),
        'feature_names_hash': meta.get('feature_names_hash'),
        'aux_dim': int(meta.get('aux_dim', -1)),
        'aux_transform': meta.get('aux_transform'),
        'label_trim_schema': LABEL_TRIM_SCHEMA,
        'low_abs_trim_fraction': float(LOW_ABS_TRIM_FRACTION),
        'high_abs_trim_fraction': float(HIGH_ABS_TRIM_FRACTION),
        'horizons_ms':[int(h) for h in HORIZONS_MS], 'split_protocol': protocol, 'train_week_keys': list(train_week_keys),
        'train_ts_start': int(tr_start), 'train_ts_end': int(tr_end), 'decision_time_basis': EXPECTED_DECISION_TIME_BASIS,
        'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
        'target_transform': TARGET_TRANSFORM, 'label_units': 'signed_log_return_bps', 'target_task': TARGET_TASK,
        'loss_weighting_schema': 'dir_mag_signed_nonzero_side_trim_tempered_class_dir_plain_mag_q50_q85_ema_v1',
        'ranking_schema': 'tie_aware_average_ranks_v1',
        'band_diag_quantiles': [float(x) for x in BAND_DIAG_QUANTILES],
    }
    cached=load_stats_cache(cache_path); stats=None
    if cached and cache_matches(cached[1], cache_meta): stats=cached[0]
    if stats is None:
        y_train=np.concatenate([np.asarray(ds.y, dtype=np.float32) for ds in ds_train_list], axis=0)
        stats=compute_signed_raw_stats(y_train)
        save_stats_cache(cache_path,stats,cache_meta)
    else:
        y_train = np.concatenate([np.asarray(ds.y, dtype=np.float32) for ds in ds_train_list], axis=0)
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
    print(f"[train_stats] dir_pos_w={dir_pos_w.tolist()} dir_neg_w={dir_neg_w.tolist()}")
    train_label_band_audit = summarize_label_band_audit(y_train, stats, split_name="train") if BAND_DIAG else None

    TrainSourceCls = CPUWindowBatchSource if TRAIN_DATA_DEVICE == "cpu_pinned" else GPUWindowBatchSource
    train_sources = [
        TrainSourceCls(
            ds,
            device,
            BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            row_stride=TRAIN_ROW_STRIDE,
        )
        for ds in ds_train_list
    ]
    if len(train_sources) == 1:
        train_src = train_sources[0]
    else:
        train_src = MultiWeekTrainBatchSource(train_sources)
    val_full_src = CPUWindowBatchSource(
        ds_val,
        device,
        BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        row_stride=1,
    )
    val_fast_src = val_full_src.make_evenly_spaced_subset(FAST_VAL_MAX_ROWS)
    train_band_src = None
    if BAND_DIAG and BAND_DIAG_TRAIN:
        train_band_src = make_train_band_eval_source(train_sources, BAND_DIAG_TRAIN_MAX_ROWS)
        print(
            f"[band-train-data] rows={train_band_src.n_rows} max_rows={BAND_DIAG_TRAIN_MAX_ROWS} weeks={len(train_sources)}",
            flush=True,
        )
    train_feature_gb_name = "feature_gb_cpu" if TRAIN_DATA_DEVICE == "cpu_pinned" else "feature_gb_gpu"
    for i, src in enumerate(train_sources):
        print(
            f"[train_data] mode={TRAIN_DATA_DEVICE} train_week[{i}]={train_week_keys[i]} rows={src.n_rows} "
            f"train_row_stride={src.row_stride} effective_rows_nominal={src.effective_rows_nominal} "
            f"feature_shape={src.feature_shape} feature_dtype={FEATURE_STORAGE_DTYPE_NAME} "
            f"{train_feature_gb_name}={src.feature_gb:.3f} label_index_gb={src.label_index_gb:.3f} "
            f"pin_memory={getattr(src, 'pin_memory', False)}",
            flush=True,
        )
    if len(train_sources) > 1:
        print(
            f"[multi_train] weeks={len(train_sources)} rows={train_src.n_rows} "
            f"effective_rows_nominal={train_src.effective_rows_nominal} batches={len(train_src)}",
            flush=True,
        )
    print(
        f"[cpu_val_data] val_full rows={val_full_src.n_rows} row_stride={val_full_src.row_stride} "
        f"feature_shape={val_full_src.feature_shape} feature_dtype={FEATURE_STORAGE_DTYPE_NAME} "
        f"feature_gb_cpu={val_full_src.feature_gb:.3f} label_index_gb_cpu={val_full_src.label_index_gb:.3f} "
        f"pin_memory={val_full_src.pin_memory}"
    )
    print(
        f"[cpu_val_data] val_fast rows={val_fast_src.n_rows} subset={FAST_VAL_MAX_ROWS} "
        f"shared_features={val_fast_src.is_shared_feature_view} "
        f"label_index_gb_cpu={val_fast_src.label_index_gb:.3f} pin_memory={val_fast_src.pin_memory}"
    )

    args = ModelArgs(DMODEL, MAMBA_LAYERS, F_total, LOOKBACK)
    model = SAMBA(args).to(device)
    mag_init_info = model.initialize_magnitude_head_bias(
        mag_pos_init_sqrt,
        mag_neg_init_sqrt,
    )
    print(
        f"[mag-init] pos_target_sqrt={mag_init_info['pos_target_sqrt']} "
        f"neg_target_sqrt={mag_init_info['neg_target_sqrt']} "
        f"pos_bias={mag_init_info['pos_bias']} neg_bias={mag_init_info['neg_bias']}",
        flush=True,
    )
    param_groups = build_optimizer_param_groups(model, base_lr=LR, gate_lr=GATE_LR)
    gate_param_count = sum(p.numel() for group in param_groups if group.get("name") == "gate" for p in group["params"])
    param_summary = summarize_param_groups(model)
    param_keys = [
        "extractor_depatch",
        "extractor_patch_embed",
        "extractor_ci",
        "extractor_gate",
        "extractor_post_gate",
        "extractor_mixed",
        "extractor_final_proj",
        "extractor_output_norm",
        "mamba",
        "task_decoders",
        "heads_pools",
        "other",
    ]
    total_params = sum(v["param_count"] for v in param_summary.values())
    print(
        "[param-groups] "
        + " ".join(f"{k}={param_summary[k]['param_count']}" for k in param_keys)
        + f" total={total_params}",
        flush=True,
    )
    diag_path = out_root / "model_diagnostics.jsonl"

    if COMPILE_ENABLED and hasattr(torch, "compile"):
        model = torch.compile(model, mode=COMPILE_MODE, dynamic=False)
        print(f"[compile] enabled full-model compile with {COMPILE_MODE} (dynamic=False)", flush=True)

    if USE_SAM:
        opt = SAM(param_groups, torch.optim.AdamW, rho=SAM_RHO)
        optimizer_name = "SAM(AdamW)"
    else:
        opt = torch.optim.AdamW(param_groups)
        optimizer_name = "AdamW"
    print(
        f"[optim-config] optimizer={optimizer_name} use_sam={int(USE_SAM)} "
        f"lr={LR} gate_lr={GATE_LR} clip_grad={CLIP_GRAD} sam_rho={SAM_RHO} "
        f"weight_decay_main=1e-3 weight_decay_gate=0 "
        f"params={sum(p.numel() for p in unwrap_model(model).parameters())} gate_params={gate_param_count}",
        flush=True,
    )
    primary_metric_mode=get_primary_metric_mode()
    best=-float('inf') if primary_metric_mode=='max' else float('inf')
    no_imp = 0
    early_stop_patience = SINGLE_WEEK_PATIENCE if len(train_week_keys) <= 1 else PATIENCE

    pos_lo_t=torch.tensor(stats['pos_lo_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    pos_hi_t=torch.tensor(stats['pos_hi_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    neg_lo_t=torch.tensor(stats['neg_lo_abs_bps'],device=device,dtype=torch.float32).view(1,-1)
    neg_hi_t=torch.tensor(stats['neg_hi_abs_bps'],device=device,dtype=torch.float32).view(1,-1)
    q50_t=torch.tensor(stats['kept_q50_abs_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    q85_t=torch.tensor(stats['kept_q85_abs_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    hwt=torch.tensor(HORIZON_WEIGHTS,device=device,dtype=torch.float32).view(1,-1)
    dir_pos_w_t = torch.tensor(dir_pos_w, device=device, dtype=torch.float32).view(1, NUM_HORIZONS)
    dir_neg_w_t = torch.tensor(dir_neg_w, device=device, dtype=torch.float32).view(1, NUM_HORIZONS)
    ema_state = LossEmaState(EMA_DECAY)

    for epoch in range(EPOCHS):
        epoch_t0 = time.perf_counter()
        train_t0 = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        dir_bce_sum = torch.zeros((), device=device)
        mag_huber_sum = torch.zeros((), device=device)
        mag_up_huber_sum = torch.zeros((), device=device)
        mag_down_huber_sum = torch.zeros((), device=device)
        mag_corr_sum = torch.zeros((), device=device)
        dir_norm_sum = torch.zeros((), device=device)
        mag_norm_sum = torch.zeros((), device=device)
        corr_norm_sum = torch.zeros((), device=device)
        dir_mask_frac_sum = torch.zeros((), device=device)
        pos_mask_frac_sum = torch.zeros((), device=device)
        neg_mask_frac_sum = torch.zeros((), device=device)
        zero_frac_batch_sum = torch.zeros((), device=device)
        n_batches = 0
        first_batch_checked = False
        for batch_i, (x, y_raw) in enumerate(tqdm(train_src.iter_epoch(epoch), total=len(train_src), desc=f"Ep{epoch+1}/{EPOCHS}")):
            diag_due = MODEL_DIAG and (batch_i == 0 or ((batch_i + 1) % MODEL_DIAG_EVERY == 0))
            opt.zero_grad(set_to_none=True)
            _check_tensor_finite("x", x, epoch=epoch, batch=batch_i, stage="input")
            _check_tensor_finite("y_raw", y_raw, epoch=epoch, batch=batch_i, stage="input")
            _check_params_finite(model, epoch=epoch, batch=batch_i, stage="batch_start")
            global_step = epoch * len(train_src) + batch_i
            if GATE_WARMUP_STEPS <= 0:
                gate_alpha = 1.0
            else:
                gate_alpha = min(1.0, float(global_step) / float(GATE_WARMUP_STEPS))
            unwrap_model(model).set_gate_alpha(gate_alpha)

            # First forward/backward: compute the raw gradient used to choose SAM's adversarial perturbation.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x)
                _check_pred_finite(pred, epoch=epoch, batch=batch_i, stage="forward1")
                if not first_batch_checked:
                    assert isinstance(pred, dict)
                    assert set(pred.keys()) == {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
                    for key in pred:
                        assert pred[key].shape == y_raw.shape
                    first_batch_checked = True
                loss, comps = compute_dir_mag_loss(
                    pred,
                    y_raw,
                    pos_lo_t=pos_lo_t,
                    pos_hi_t=pos_hi_t,
                    neg_lo_t=neg_lo_t,
                    neg_hi_t=neg_hi_t,
                    q50_t=q50_t,
                    q85_t=q85_t,
                    hwt=hwt,
                    dir_pos_w_t=dir_pos_w_t,
                    dir_neg_w_t=dir_neg_w_t,
                    ema_state=ema_state,
                    update_ema=True,
                )
                _check_tensor_finite("loss", loss, epoch=epoch, batch=batch_i, stage="loss1")
                for k, v in comps.items():
                    _check_tensor_finite(f"comps.{k}", v, epoch=epoch, batch=batch_i, stage="loss1")

            loss.backward()
            grad_norm1 = _check_grads_finite(model, epoch=epoch, batch=batch_i, stage="backward1")

            if USE_SAM:
                opt.first_step(zero_grad=True)
                _check_params_finite(model, epoch=epoch, batch=batch_i, stage="after_sam_first_step")

                # Second forward/backward at perturbed weights.
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    pred2 = model(x)
                    _check_pred_finite(pred2, epoch=epoch, batch=batch_i, stage="forward2_sam_perturbed")
                    loss2, comps2 = compute_dir_mag_loss(
                        pred2,
                        y_raw,
                        pos_lo_t=pos_lo_t,
                        pos_hi_t=pos_hi_t,
                        neg_lo_t=neg_lo_t,
                        neg_hi_t=neg_hi_t,
                        q50_t=q50_t,
                        q85_t=q85_t,
                        hwt=hwt,
                        dir_pos_w_t=dir_pos_w_t,
                        dir_neg_w_t=dir_neg_w_t,
                        ema_state=ema_state,
                        update_ema=False,
                    )
                    _check_tensor_finite("loss2", loss2, epoch=epoch, batch=batch_i, stage="loss2")
                    for k, v in comps2.items():
                        _check_tensor_finite(f"comps2.{k}", v, epoch=epoch, batch=batch_i, stage="loss2")

                loss2.backward()
                grad_norm2 = _check_grads_finite(model, epoch=epoch, batch=batch_i, stage="backward2")

                # Clip only before the actual optimizer update.
                grad_diag = summarize_grad_groups(model, LR) if diag_due else None
                grad_norm_clip_return = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CLIP_GRAD,
                    error_if_nonfinite=True,
                )
                _check_grads_finite(model, epoch=epoch, batch=batch_i, stage="after_clip")

                opt.second_step(zero_grad=True)
                _check_params_finite(model, epoch=epoch, batch=batch_i, stage="after_optimizer_step")

                if diag_due:
                    activation_diag = run_model_diagnostics(
                        model, x, y_raw, epoch=epoch, batch_i=batch_i, device=device,
                        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    )
                    print_model_diagnostics(epoch, batch_i, activation_diag, grad_diag)
                    append_jsonl(diag_path, {
                        "epoch": epoch + 1,
                        "batch": batch_i,
                        "optimizer": optimizer_name,
                        "loss1": float(loss.detach().float().item()),
                        "loss2": float(loss2.detach().float().item()),
                        "grad_global_norm1": float(grad_norm1),
                        "grad_global_norm2": float(grad_norm2),
                        "clip_return": float(grad_norm_clip_return.detach().float().item()),
                        "activation_diag": activation_diag,
                        "gradient_diag": grad_diag,
                        "residual_scalar_diag": activation_diag["extractor"]["residual_scalars"],
                        "prediction_diag": activation_diag["prediction"],
                    })

                if FINITE_DEBUG and ((batch_i + 1) % LOG_EVERY == 0 or batch_i == 0):
                    print(
                        f"[debug-train] epoch={epoch+1} batch={batch_i} optimizer=SAM "
                        f"loss1={float(loss.detach().float().item()):.6g} "
                        f"loss2={float(loss2.detach().float().item()):.6g} "
                        f"grad_norm1={grad_norm1:.6g} "
                        f"grad_norm2={grad_norm2:.6g} "
                        f"clip_return={float(grad_norm_clip_return.detach().float().item()):.6g}",
                        flush=True,
                    )
            else:
                grad_diag = summarize_grad_groups(model, LR) if diag_due else None
                grad_norm_clip_return = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CLIP_GRAD,
                    error_if_nonfinite=True,
                )

                _check_grads_finite(model, epoch=epoch, batch=batch_i, stage="after_clip")

                opt.step()
                opt.zero_grad(set_to_none=True)

                _check_params_finite(model, epoch=epoch, batch=batch_i, stage="after_optimizer_step")

                if diag_due:
                    activation_diag = run_model_diagnostics(
                        model, x, y_raw, epoch=epoch, batch_i=batch_i, device=device,
                        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    )
                    print_model_diagnostics(epoch, batch_i, activation_diag, grad_diag)
                    append_jsonl(diag_path, {
                        "epoch": epoch + 1,
                        "batch": batch_i,
                        "optimizer": optimizer_name,
                        "loss1": float(loss.detach().float().item()),
                        "loss2": None,
                        "grad_global_norm1": float(grad_norm1),
                        "grad_global_norm2": None,
                        "clip_return": float(grad_norm_clip_return.detach().float().item()),
                        "activation_diag": activation_diag,
                        "gradient_diag": grad_diag,
                        "residual_scalar_diag": activation_diag["extractor"]["residual_scalars"],
                        "prediction_diag": activation_diag["prediction"],
                    })

                if FINITE_DEBUG and ((batch_i + 1) % LOG_EVERY == 0 or batch_i == 0):
                    print(
                        f"[debug-train] epoch={epoch+1} batch={batch_i} optimizer=AdamW "
                        f"loss1={float(loss.detach().float().item()):.6g} "
                        f"grad_norm1={grad_norm1:.6g} "
                        f"clip_return={float(grad_norm_clip_return.detach().float().item()):.6g}",
                        flush=True,
                    )

            loss_sum += loss.detach()
            dir_bce_sum += comps["dir_bce"]
            mag_huber_sum += comps["mag_huber"]
            mag_up_huber_sum += comps["mag_up_huber"]
            mag_down_huber_sum += comps["mag_down_huber"]
            mag_corr_sum += comps["mag_corr"]
            dir_norm_sum += comps["dir_norm"]
            mag_norm_sum += comps["mag_norm"]
            corr_norm_sum += comps["corr_norm"]
            dir_mask_frac_sum += comps["dir_mask_frac"]
            pos_mask_frac_sum += comps["pos_mask_frac"]
            neg_mask_frac_sum += comps["neg_mask_frac"]
            zero_frac_batch_sum += comps["zero_frac_batch"]
            n_batches += 1

        train_sec = time.perf_counter() - train_t0
        train_loss = (loss_sum / max(1, n_batches)).item()
        train_dir_bce = (dir_bce_sum / max(1, n_batches)).item()
        train_mag_huber = (mag_huber_sum / max(1, n_batches)).item()
        train_mag_up_huber = (mag_up_huber_sum / max(1, n_batches)).item()
        train_mag_down_huber = (mag_down_huber_sum / max(1, n_batches)).item()
        train_mag_corr = (mag_corr_sum / max(1, n_batches)).item()
        train_dir_norm = (dir_norm_sum / max(1, n_batches)).item()
        train_mag_norm = (mag_norm_sum / max(1, n_batches)).item()
        train_corr_norm = (corr_norm_sum / max(1, n_batches)).item()
        train_dir_mask_frac = (dir_mask_frac_sum / max(1, n_batches)).item()
        train_pos_mask_frac = (pos_mask_frac_sum / max(1, n_batches)).item()
        train_neg_mask_frac = (neg_mask_frac_sum / max(1, n_batches)).item()
        train_zero_frac_batch = (zero_frac_batch_sum / max(1, n_batches)).item()
        print(f"[train] loss={train_loss:.6f} dir_bce={train_dir_bce:.6f} mag_huber={train_mag_huber:.6f} mag_up_huber={train_mag_up_huber:.6f} mag_down_huber={train_mag_down_huber:.6f} mag_corr={train_mag_corr:.6f} dir_norm={train_dir_norm:.6f} mag_norm={train_mag_norm:.6f} corr_norm={train_corr_norm:.6f} dir_mask_frac={train_dir_mask_frac:.6f} pos_mask_frac={train_pos_mask_frac:.6f} neg_mask_frac={train_neg_mask_frac:.6f} zero_frac_batch={train_zero_frac_batch:.6f}")
        if BAND_DIAG and BAND_DIAG_TRAIN and train_band_src is not None:
            train_band = summarize_metrics(
                model, train_band_src, device, stats, amp_enabled, amp_dtype,
                primary_only=False, epoch=epoch, band_diag=True, split_name="train",
            )
            print_band_metrics_summary(train_band["band_metrics"], split_name="train", epoch=epoch)
            save_band_metrics_jsonl(out_root, train_band["band_metrics"], epoch=epoch, split_name="train")

        val_fast_t0 = time.perf_counter()
        val_fast=summarize_metrics(model, val_fast_src, device, stats, amp_enabled, amp_dtype, primary_only=True, epoch=epoch, band_diag=BAND_DIAG, split_name="val_fast")
        val_fast_sec = time.perf_counter() - val_fast_t0
        primary_metric_value, primary_metric_label = compute_primary_metric(val_fast)
        print(
            f"[val_fast] primary_metric_name={primary_metric_label} "
            f"primary_horizon_ms={val_fast.get('primary_horizon_ms', PRIMARY_METRIC_HORIZON_MS)} "
            f"value={primary_metric_value:.6f} "
            f"guard_dir_bal_acc={val_fast.get('primary_dir_bal_acc', float('nan')):.6f} "
            f"guard_passed={val_fast.get('primary_metric_guard_passed', False)} "
            f"rows={val_fast.get('n_eval_rows', 0)}"
        )
        if BAND_DIAG:
            print_band_metrics_summary(val_fast["band_metrics"], split_name="val", epoch=epoch)
            save_band_metrics_jsonl(out_root, val_fast["band_metrics"], epoch=epoch, split_name="val_fast")
        full_val_sec = 0.0
        improved = math.isfinite(primary_metric_value) and is_metric_improved(primary_metric_value,best,primary_metric_mode)
        if improved:
            best=float(primary_metric_value); no_imp=0
            full_t0 = time.perf_counter()
            full=summarize_metrics(model, val_full_src, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch=epoch, band_diag=BAND_DIAG, split_name="val_full")
            full_val_sec += time.perf_counter() - full_t0
            full_value, full_label = compute_primary_metric(full)
            print(
                f"[val_full] epoch={epoch + 1} rows={full.get('n_eval_rows', 0)} "
                f"primary_metric_name={full_label} value={full_value:.6f} "
                f"guard_dir_bal_acc={full.get('primary_dir_bal_acc', float('nan')):.6f} "
                f"guard_passed={full.get('primary_metric_guard_passed', False)}"
            )
            if BAND_DIAG:
                print_band_metrics_summary(full["band_metrics"], split_name="val_full", epoch=epoch)
                save_band_metrics_jsonl(out_root, full["band_metrics"], epoch=epoch, split_name="val_full")
            ckpt={
                'epoch': epoch,
                'state_dict': get_model_state_dict(model),
                'args': {
                    'DMODEL':DMODEL, 'MAMBA_LAYERS':MAMBA_LAYERS, 'feat_dim':F_total, 'LOOKBACK':LOOKBACK,
                    'WINDOW_MS': WINDOW_MS, 'HORIZONS_MS': HORIZONS_MS, 'checkpoint_schema': CHECKPOINT_SCHEMA,
                    'model_arch_schema': MODEL_ARCH_SCHEMA,
                    'model_output_schema': MODEL_OUTPUT_SCHEMA,
                    'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
                    'decision_time_basis': meta.get('decision_time_basis'), 'decision_stride_policy':'every_ob_event',
                    'label_delta_ms':0, 'label_units':'signed_log_return_bps',
                    'target_task': TARGET_TASK,
                    'target_transform': TARGET_TRANSFORM,
                    'label_trim_schema': LABEL_TRIM_SCHEMA,
                    'loss_weighting_schema': 'dir_mag_signed_nonzero_side_trim_tempered_class_dir_plain_mag_q50_q85_ema_v1',
                    'low_abs_trim_fraction': float(LOW_ABS_TRIM_FRACTION),
                    'high_abs_trim_fraction': float(HIGH_ABS_TRIM_FRACTION),
                    'primary_metric': PRIMARY_METRIC,
                    'primary_metric_horizon_ms': PRIMARY_METRIC_HORIZON_MS,
                    'primary_dir_bal_acc_guard': PRIMARY_DIR_BAL_ACC_GUARD,
                    'feature_storage_dtype': FEATURE_STORAGE_DTYPE_NAME,
                    'split_protocol': protocol,
                    'train_week_keys': list(train_week_keys),
                    'train_row_stride': int(TRAIN_ROW_STRIDE),
                },
                'model_output_schema': MODEL_OUTPUT_SCHEMA,
                'best_primary_metric': best,
                'selection_metric_source': f'fast_val_{FAST_VAL_MAX_ROWS}_primary',
                'full_val_ran_on_improvement': True,
                'fast_val_primary_metric': float(primary_metric_value),
                'fast_val_metrics': val_fast,
                'full_val_metrics': full,
            }
            out_ckpt=out_root/'cmssl17_offline_best.pt'; torch.save(ckpt,out_ckpt); print(f"[ckpt] saved best to {out_ckpt}")
        else:
            no_imp += 1
            if (epoch + 1) % FULL_VAL_EVERY == 0:
                full_t0 = time.perf_counter()
                full_periodic = summarize_metrics(model, val_full_src, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch=epoch, band_diag=BAND_DIAG, split_name="val_full")
                full_val_sec += time.perf_counter() - full_t0
                full_value, full_label = compute_primary_metric(full_periodic)
                print(
                    f"[val_full] epoch={epoch + 1} rows={full_periodic.get('n_eval_rows', 0)} "
                    f"primary_metric_name={full_label} value={full_value:.6f} "
                    f"guard_dir_bal_acc={full_periodic.get('primary_dir_bal_acc', float('nan')):.6f} "
                    f"guard_passed={full_periodic.get('primary_metric_guard_passed', False)}"
                )
                if BAND_DIAG:
                    print_band_metrics_summary(full_periodic["band_metrics"], split_name="val_full", epoch=epoch)
                    save_band_metrics_jsonl(out_root, full_periodic["band_metrics"], epoch=epoch, split_name="val_full")
        total_sec = time.perf_counter() - epoch_t0
        print(f"[epoch_time] train_sec={train_sec:.3f} val_fast_sec={val_fast_sec:.3f} full_val_sec={full_val_sec:.3f} total_sec={total_sec:.3f}")

        if not improved and no_imp >= early_stop_patience:
            print('Early stopping triggered.')
            break

    best_path = out_root / 'cmssl17_offline_best.pt'
    if not best_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint was saved. No epoch satisfied {PRIMARY_METRIC} with "
            f"dir_bal_acc_q50plus guard >= {PRIMARY_DIR_BAL_ACC_GUARD}."
        )
    ckpt = torch.load(best_path, map_location=device)
    state = ckpt.get('state_dict')
    if state is None:
        raise KeyError(f"Checkpoint {best_path} missing 'state_dict'")
    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Checkpoint schema mismatch: got {ckpt_args.get('checkpoint_schema')}, expected {CHECKPOINT_SCHEMA}")
    if ckpt_args.get("model_arch_schema") != MODEL_ARCH_SCHEMA:
        raise ValueError(f"Model architecture schema mismatch: got {ckpt_args.get('model_arch_schema')}, expected {MODEL_ARCH_SCHEMA}")
    if ckpt_args.get("model_output_schema") != MODEL_OUTPUT_SCHEMA:
        raise ValueError(f"Model output schema mismatch: got {ckpt_args.get('model_output_schema')}, expected {MODEL_OUTPUT_SCHEMA}")
    unwrap_model(model).load_state_dict(state, strict=True)
    model.eval()

    del train_src
    if "train_sources" in locals():
        del train_sources
    if "ds_train_list" in locals():
        del ds_train_list
    if "val_fast_src" in locals():
        del val_fast_src
    if "val_full_src" in locals():
        del val_full_src
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if has_cmssl_test:
        test_full_src = CPUWindowBatchSource(
            ds_test,
            device,
            BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            row_stride=1,
        )
        print(
            f"[cpu_test_data] test_full rows={test_full_src.n_rows} row_stride={test_full_src.row_stride} "
            f"feature_shape={test_full_src.feature_shape} feature_dtype={FEATURE_STORAGE_DTYPE_NAME} "
            f"feature_gb_cpu={test_full_src.feature_gb:.3f} label_index_gb_cpu={test_full_src.label_index_gb:.3f} "
            f"pin_memory={test_full_src.pin_memory}"
        )

        test=summarize_metrics(model, test_full_src, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch=0, band_diag=BAND_DIAG, split_name="test")
        test_value, test_label = compute_primary_metric(test)
        print(
            f"[test] rows={test.get('n_eval_rows', 0)} primary_metric_name={test_label} value={test_value:.6f} "
            f"guard_dir_bal_acc={test.get('primary_dir_bal_acc', float('nan')):.6f} "
            f"guard_passed={test.get('primary_metric_guard_passed', False)}"
        )
        if BAND_DIAG:
            best_epoch_or_0 = int(ckpt.get('epoch', 0)) if isinstance(ckpt, dict) else 0
            print_band_metrics_summary(test["band_metrics"], split_name="test", epoch=best_epoch_or_0)
            save_band_metrics_jsonl(out_root, test["band_metrics"], epoch=best_epoch_or_0, split_name="test")
    else:
        print(f"[test] skipped: cmssl.test split not present for protocol={protocol}", flush=True)
    print('[done] Training complete.')


# ---------------- Entry ----------------
if __name__ == "__main__":
    train_from_offline()
