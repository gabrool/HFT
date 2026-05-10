
#!/usr/bin/env python3
"""
CM_offline.py

Run CM.py using flat decision rows produced by data_ingest.py.
Windows are materialized dynamically from flat core/aux/y/ts chunks.
"""

import os, sys, math, json
from typing import List, Dict, Tuple, Iterable, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from offline_tokens import (
    read_json,
    load_global_meta,
    ChunkRef,
)

# ---------------- Import from CMSSL17 ----------------
# Configure CUDA allocator only for this entrypoint execution to avoid
# import-time side effects when CMSSL17 is used as a library module.
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from CM import (  # type: ignore
    # model + args
    SAMBA, ModelArgs,
    # core hypers
    LOOKBACK, AUX_DIM, HORIZONS_MS, NUM_HORIZONS, HORIZON_WEIGHTS,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
    # schedules
    DIR_MASK_TAIL_FRACTION,
    DMODEL, MAMBA_LAYERS,
    PRIMARY_METRIC_HORIZON_MS,
    # utils
    binary_auc_from_logits,
    SINGLE_WEEK_PATIENCE, get_primary_metric_mode, compute_primary_metric, is_metric_improved,
    # optimizer
    SAM,
)

# ---------------- Config via env ----------------
OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
USE_IN_MEMORY = int(os.environ.get("BYBIT_USE_IN_MEMORY", "0")) == 1
WORKERS_TRAIN = int(os.environ.get("BYBIT_WORKERS", "8"))
WORKERS_VAL   = max(1, min(4, WORKERS_TRAIN // 2))
AMP_ENABLED   = int(os.environ.get("BYBIT_AMP", "1")) == 1
COMPILE_ENABLED = int(os.environ.get("BYBIT_TORCH_COMPILE", "1")) == 1
COMPILE_MODE = os.environ.get("BYBIT_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs").strip()
LOG_EVERY     = max(1, int(os.environ.get("BYBIT_LOG_EVERY", "100")))
CUDNN_BENCHMARK = int(os.environ.get("BYBIT_CUDNN_BENCHMARK", "1")) == 1
MATMUL_PRECISION = os.environ.get("BYBIT_MATMUL_PRECISION", "high").strip().lower()
EXPECTED_DECISION_TIME_BASIS = "ob_event_time"
EXPECTED_DECISION_POLICY = "ob_event_time"

TRAIN_ROW_STRIDE = int(os.environ.get("BYBIT_TRAIN_ROW_STRIDE", "5"))
if TRAIN_ROW_STRIDE < 1:
    raise ValueError(f"BYBIT_TRAIN_ROW_STRIDE must be >= 1, got {TRAIN_ROW_STRIDE}")
TRAIN_DATA_DEVICE = os.environ.get("BYBIT_TRAIN_DATA_DEVICE", "cpu_pinned").strip().lower()
if TRAIN_DATA_DEVICE not in {"cpu", "cpu_pinned", "cuda"}:
    raise ValueError(f"BYBIT_TRAIN_DATA_DEVICE must be one of cpu, cpu_pinned, cuda; got {TRAIN_DATA_DEVICE!r}")
_feature_storage_dtype_raw = os.environ.get("BYBIT_FEATURE_STORAGE_DTYPE", "bf16").strip().lower()
if _feature_storage_dtype_raw in {"fp32", "float32"}:
    FEATURE_STORAGE_DTYPE = torch.float32
    FEATURE_STORAGE_DTYPE_NAME = "fp32"
elif _feature_storage_dtype_raw in {"bf16", "bfloat16"}:
    FEATURE_STORAGE_DTYPE = torch.bfloat16
    FEATURE_STORAGE_DTYPE_NAME = "bf16"
else:
    raise ValueError("BYBIT_FEATURE_STORAGE_DTYPE must be one of: fp32, float32, bf16, bfloat16")
BF16_FEATURE_DEBUG = int(os.environ.get("BYBIT_BF16_FEATURE_DEBUG", "0")) == 1
BF16_FEATURE_DEBUG_MAX_BATCHES = max(1, int(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_MAX_BATCHES", "3")))
BF16_FEATURE_DEBUG_WARN_MAX_ABS_ERR = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_MAX_ABS_ERR", "0.05"))
BF16_FEATURE_DEBUG_WARN_MEAN_ABS_ERR = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_MEAN_ABS_ERR", "0.002"))
BF16_FEATURE_DEBUG_WARN_SIGN_FLIP_FRAC = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_SIGN_FLIP_FRAC", "0.001"))
BF16_FEATURE_DEBUG_WARN_ZERO_FLIP_FRAC = float(os.environ.get("BYBIT_BF16_FEATURE_DEBUG_WARN_ZERO_FLIP_FRAC", "0.001"))
USE_SAM = int(os.environ.get("BYBIT_USE_SAM", "1")) == 1
SAM_RHO = float(os.environ.get("BYBIT_SAM_RHO", "0.01"))

_BF16_FEATURE_DEBUG_BATCHES_PRINTED = 0

def summarize_bf16_feature_error(x_fp32: torch.Tensor, x_stored: torch.Tensor) -> Dict[str, float]:
    a = x_fp32.detach().float()
    b = x_stored.detach().float()
    if a.shape != b.shape:
        raise ValueError(f"BF16 debug shape mismatch: fp32={tuple(a.shape)} stored={tuple(b.shape)}")
    diff = b - a
    abs_diff = diff.abs()
    abs_a = a.abs()
    rel_abs = abs_diff / torch.clamp(abs_a, min=1e-6)
    finite_b = torch.isfinite(b)
    finite_both = torch.isfinite(a) & finite_b
    nonzero_a = abs_a > 0.0
    sign_flip = ((torch.sign(a) != torch.sign(b)) & nonzero_a & finite_both)
    zero_flip = (((a == 0.0) != (b == 0.0)) & finite_both)
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
    if not enabled or FEATURE_STORAGE_DTYPE_NAME != "bf16" or _BF16_FEATURE_DEBUG_BATCHES_PRINTED >= BF16_FEATURE_DEBUG_MAX_BATCHES:
        return
    batch_i = int(_BF16_FEATURE_DEBUG_BATCHES_PRINTED)
    _BF16_FEATURE_DEBUG_BATCHES_PRINTED += 1
    diag = summarize_bf16_feature_error(x_fp32, x_stored)
    print(
        "[bf16-feature-debug] "
        f"batch={batch_i} max_abs_err={diag['max_abs_err']:.6g} mean_abs_err={diag['mean_abs_err']:.6g} "
        f"p99_abs_err={diag['p99_abs_err']:.6g} max_rel_abs_err={diag['max_rel_abs_err']:.6g} "
        f"p99_rel_abs_err={diag['p99_rel_abs_err']:.6g} sign_flip_frac={diag['sign_flip_frac']:.6g} "
        f"zero_flip_frac={diag['zero_flip_frac']:.6g} nonfinite_after_frac={diag['nonfinite_after_frac']:.6g} "
        f"per_feature_mean_abs_err_max={diag['per_feature_mean_abs_err_max']:.6g}",
        flush=True,
    )
    if (diag["max_abs_err"] > BF16_FEATURE_DEBUG_WARN_MAX_ABS_ERR or diag["mean_abs_err"] > BF16_FEATURE_DEBUG_WARN_MEAN_ABS_ERR or diag["sign_flip_frac"] > BF16_FEATURE_DEBUG_WARN_SIGN_FLIP_FRAC or diag["zero_flip_frac"] > BF16_FEATURE_DEBUG_WARN_ZERO_FLIP_FRAC or diag["nonfinite_after_frac"] > 0.0):
        print("[bf16-feature-debug-warn] BF16 feature storage may be materially changing features; consider BYBIT_FEATURE_STORAGE_DTYPE=fp32", flush=True)


def require_phase1_five_week_pipeline_splits(meta: dict, out_root: Path) -> dict:
    if "pipeline_splits" not in meta:
        raise KeyError("meta.json missing required key 'pipeline_splits'. Rerun data_ingest.py.")
    splits = meta["pipeline_splits"]
    if not isinstance(splits, dict):
        raise KeyError("meta['pipeline_splits'] must be a dict. Rerun data_ingest.py.")
    if splits.get("protocol") != "five_week_cmssl2w_val_test_rl_eval_v1":
        raise ValueError("meta['pipeline_splits']['protocol'] must be 'five_week_cmssl2w_val_test_rl_eval_v1'.")

    weeks_in_order = meta.get("weeks_in_order")
    if not isinstance(weeks_in_order, list) or len(weeks_in_order) != 5 or not all(isinstance(w, str) and w for w in weeks_in_order):
        raise KeyError("meta['weeks_in_order'] must be a list[str] with exactly 5 entries.")

    decision_time_basis = meta.get("decision_time_basis")
    if decision_time_basis != EXPECTED_DECISION_TIME_BASIS:
        raise ValueError(f"Expected decision_time_basis={EXPECTED_DECISION_TIME_BASIS!r}, got {decision_time_basis!r}.")
    if meta.get("decision_policy", EXPECTED_DECISION_POLICY) != EXPECTED_DECISION_POLICY:
        raise ValueError(f"Expected decision_policy={EXPECTED_DECISION_POLICY!r}, got {meta.get('decision_policy')!r}.")

    weeks_meta_map = meta.get("weeks_meta")
    if not isinstance(weeks_meta_map, dict) or not weeks_meta_map:
        raise KeyError("meta.json missing required non-empty key 'weeks_meta'.")

    known_weeks = set(weeks_in_order)

    def _weeks(section: str, name: str | None = None) -> List[str]:
        obj = splits.get(section)
        if name is not None:
            if not isinstance(obj, dict):
                raise KeyError(f"meta['pipeline_splits']['{section}'] must be a dict")
            obj = obj.get(name)
        elif isinstance(obj, dict) and "weeks" in obj:
            obj = obj.get("weeks")
        if not isinstance(obj, list) or not obj or not all(isinstance(w, str) and w for w in obj):
            label = f"{section}.{name}" if name else section
            raise KeyError(f"meta['pipeline_splits']['{label}'] must be a non-empty list of week keys")
        missing = sorted(w for w in obj if w not in known_weeks)
        if missing:
            raise KeyError(f"pipeline_splits references unknown week key(s): {missing}")
        return list(obj)

    def _full_week_range(week_key: str, stage: str) -> Tuple[int, int]:
        rel_path = weeks_meta_map.get(week_key)
        if not isinstance(rel_path, str) or not rel_path:
            raise KeyError(f"meta['weeks_meta'] missing path for week '{week_key}' referenced by {stage}.")
        week_meta = read_json(out_root / rel_path)
        decision_range = week_meta.get("decision_ts_range")
        if not isinstance(decision_range, dict) or "min" not in decision_range or "max" not in decision_range:
            raise KeyError(f"Week metadata for {stage} must include decision_ts_range min/max.")
        start = int(decision_range["min"])
        end = int(decision_range["max"]) + 1
        if start >= end:
            raise ValueError(f"Week metadata for {stage} has invalid decision_ts_range: start={start} end={end}.")
        return start, end

    def _entry(stage: str, weeks: List[str]) -> dict:
        starts_ends = [_full_week_range(w, stage) for w in weeks]
        return {"weeks": weeks, "start": min(se[0] for se in starts_ends), "end": max(se[1] for se in starts_ends)}

    week1, week2, week3, week4, week5 = weeks_in_order
    train_weeks = _weeks("cmssl", "train")
    val_weeks = _weeks("cmssl", "val")
    test_weeks = _weeks("cmssl", "test")
    rl_weeks = _weeks("rl", "train")
    eval_weeks = _weeks("eval")
    if train_weeks != [week1, week2] or val_weeks != [week3] or test_weeks != [week4] or rl_weeks != [week4] or eval_weeks != [week5]:
        raise ValueError("Phase 1 splits must be week1+week2=train, week3=val, week4=test/RL, week5=eval.")

    normalized = {
        "protocol": splits["protocol"],
        "cmssl": {
            "train": _entry("cmssl.train", train_weeks),
            "val": _entry("cmssl.val", val_weeks),
            "test": _entry("cmssl.test", test_weeks),
        },
        "rl": {"train": _entry("rl.train", rl_weeks)},
        "eval": {"full": _entry("eval", eval_weeks)},
    }
    return {"splits": normalized, "weeks_in_order": weeks_in_order}

def _label_dim_error(source: str, observed: Any) -> ValueError:
    return ValueError(
        f"{source} has label_dim={observed!r}, but CM_offline.py now requires "
        f"label_dim={NUM_HORIZONS}. Old offline datasets with 2 * NUM_HORIZONS labels are no longer supported; "
        "rebuild the offline data with offline_ingest.py."
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


def build_chunk_refs_by_ts(meta_week_path: Path, start: int, end: int) -> List[ChunkRef]:
    """
    Build ChunkRefs for rows whose timestamps satisfy start <= ts < end.

    The function performs contiguous slicing per chunk via searchsorted on each
    chunk's ts file and avoids materializing full boolean masks / index lists.
    """
    if end < start:
        raise ValueError(f"Invalid ts range: start={start} must be <= end={end}")

    wmeta = read_json(meta_week_path)
    validate_dataset_label_dim(wmeta, f"week metadata {meta_week_path}")
    week_dir = meta_week_path.parent
    refs: List[ChunkRef] = []

    for idx, ch in enumerate(wmeta.get("chunks", [])):
        files = ch.get("files", {})
        ts_rel = files.get("ts")
        if not ts_rel:
            raise KeyError(
                f"Chunk {idx} in {meta_week_path} is missing files['ts']; cannot slice by timestamp"
            )

        ts_arr = np.load(week_dir / ts_rel, mmap_mode="r")
        if ts_arr.ndim != 1:
            raise ValueError(
                f"Expected 1D ts array in chunk {idx} ({week_dir / ts_rel}), got shape={ts_arr.shape}"
            )

        # Safety check: searchsorted semantics require non-decreasing input.
        if ts_arr.size > 1 and not np.all(ts_arr[1:] >= ts_arr[:-1]):
            raise ValueError(
                f"Timestamp file is not non-decreasing for chunk {idx}: {week_dir / ts_rel}"
            )

        l = int(np.searchsorted(ts_arr, start, side="left"))
        r = int(np.searchsorted(ts_arr, end, side="left"))

        if r > l:
            refs.append(ChunkRef(
                week_dir=week_dir,
                core_file=week_dir / files["core"],
                aux_file=week_dir / files["aux"],
                y_file=week_dir / files["y"],
                n=r - l,
                offset=l,
            ))

    return refs

# ---------------- Dynamic flat-row datasets ----------------
VALIDATE_DYNAMIC_WINDOWS = int(os.environ.get("BYBIT_VALIDATE_DYNAMIC_WINDOWS", "1")) == 1


class FlatWindowDataset(Dataset):
    """Materialize [LOOKBACK, F] windows dynamically from flat decision rows."""

    def __init__(self, chunk_refs: List[ChunkRef], feature_dim_total: int, *, split: str):
        self.refs = list(chunk_refs)
        self.F = int(feature_dim_total)
        self.F_core = self.F - AUX_DIM
        self.split = str(split)
        if self.F_core <= 0:
            raise ValueError(f"feature_dim_total ({self.F}) must exceed AUX_DIM ({AUX_DIM})")
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        ts_parts: List[np.ndarray] = []
        self.segments: List[Tuple[str, int, int]] = []
        cursor = 0
        current_source: Optional[str] = None
        current_start = 0
        for ref in self.refs:
            Xc = np.load(ref.core_file, mmap_mode="r")
            Xa = np.load(ref.aux_file, mmap_mode="r")
            Y = np.load(ref.y_file, mmap_mode="r")
            validate_loaded_label_array(Y, f"label file {ref.y_file}")
            l = int(ref.offset); r = l + int(ref.n)
            core = np.asarray(Xc[l:r], dtype=np.float32)
            aux = np.asarray(Xa[l:r], dtype=np.float32)
            if core.ndim != 2 or aux.ndim != 2:
                raise ValueError(f"Flat core/aux chunks must be 2D: {ref.core_file}, {ref.aux_file}")
            source = str(ref.week_dir)
            if current_source is None:
                current_source = source
                current_start = cursor
            elif source != current_source:
                self.segments.append((current_source, current_start, cursor))
                current_source = source
                current_start = cursor
            xs.append(np.concatenate([core, aux], axis=-1).astype(np.float32, copy=False))
            ys.append(np.asarray(Y[l:r], dtype=np.float32))
            ts_path = ref.week_dir / Path(str(ref.y_file.name).replace('y_', 'ts_'))
            if ts_path.exists():
                ts_parts.append(np.asarray(np.load(ts_path, mmap_mode="r")[l:r], dtype=np.int64))
            cursor += int(r - l)
        if current_source is not None:
            self.segments.append((current_source, current_start, cursor))
        if xs:
            self.features_fp32 = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
            self.y = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
            self.ts = np.concatenate(ts_parts, axis=0).astype(np.int64, copy=False) if ts_parts else np.arange(self.y.shape[0], dtype=np.int64)
        else:
            self.features_fp32 = np.empty((0, self.F), dtype=np.float32)
            self.y = np.empty((0, NUM_HORIZONS), dtype=np.float32)
            self.ts = np.empty((0,), dtype=np.int64)
        if self.features_fp32.shape[1] != self.F:
            raise ValueError(f"Feature dim mismatch: loaded {self.features_fp32.shape[1]} expected {self.F}")
        valid_parts = []
        for source, start, end in self.segments:
            first = int(start + LOOKBACK - 1)
            if end > first:
                valid_parts.append(np.arange(first, end, dtype=np.int64))
        self.valid_rows = np.concatenate(valid_parts, axis=0) if valid_parts else np.empty((0,), dtype=np.int64)
        self.total = int(self.valid_rows.shape[0])
        self.features = torch.from_numpy(self.features_fp32).to(dtype=FEATURE_STORAGE_DTYPE)
        self.labels = torch.from_numpy(self.y)
        if TRAIN_DATA_DEVICE == "cpu_pinned" and torch.cuda.is_available():
            self.features = self.features.pin_memory()
            self.labels = self.labels.pin_memory()
        elif TRAIN_DATA_DEVICE == "cuda" and torch.cuda.is_available():
            self.features = self.features.cuda(non_blocking=True)
            self.labels = self.labels.cuda(non_blocking=True)
        feature_gb = (self.features.numel() * self.features.element_size()) / 1e9
        label_gb = (self.labels.numel() * self.labels.element_size()) / 1e9
        print(
            f"[flat-data] split={self.split} rows={self.features_fp32.shape[0]} sources={len(self.segments)} "
            f"valid_window_rows={self.total} row_stride={(TRAIN_ROW_STRIDE if split=='train' else 1)} "
            f"effective_rows_nominal={(max(0, (self.total + (TRAIN_ROW_STRIDE if split=='train' else 1) - 1) // (TRAIN_ROW_STRIDE if split=='train' else 1)))} "
            f"feature_shape={tuple(self.features.shape)} feature_dtype={FEATURE_STORAGE_DTYPE_NAME} "
            f"feature_gb={feature_gb:.3f} label_gb={label_gb:.3f} pin_memory={TRAIN_DATA_DEVICE == 'cpu_pinned'}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.total

    def flat_row_for_dataset_index(self, idx: int) -> int:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        return int(self.valid_rows[int(idx)])

    def __getitem__(self, idx: int):
        r = self.flat_row_for_dataset_index(int(idx))
        lo = r - LOOKBACK + 1
        hi = r + 1
        if lo < 0:
            raise IndexError("Dynamic window would use negative flat row index")
        x_fp32 = torch.from_numpy(self.features_fp32[lo:hi])
        x_stored = self.features[lo:hi]
        maybe_print_bf16_feature_debug(x_fp32, x_stored, enabled=BF16_FEATURE_DEBUG and FEATURE_STORAGE_DTYPE_NAME == "bf16")
        return x_stored, self.labels[r]

    def iter_epoch(self, epoch: int = 0):
        stride = TRAIN_ROW_STRIDE if self.split == "train" else 1
        offset = int(epoch) % int(stride)
        indices = np.arange(offset, len(self), stride, dtype=np.int64)
        if self.split == "train":
            rng = np.random.default_rng(1234 + int(epoch))
            rng.shuffle(indices)
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            if batch_indices.size == 0:
                continue
            xs, ys = zip(*(self[int(i)] for i in batch_indices))
            yield torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0)


def validate_dynamic_window_alignment(source: FlatWindowDataset, n_checks: int = 16) -> None:
    if len(source) <= 0:
        print(f"[window-check] split={source.split} checks=0 passed=1 (empty)", flush=True)
        return
    rng = np.random.default_rng(12345)
    checks = min(int(n_checks), len(source))
    picks = rng.choice(len(source), size=checks, replace=False) if len(source) > checks else np.arange(len(source))
    for idx in picks:
        r = source.flat_row_for_dataset_index(int(idx))
        lo = r - LOOKBACK + 1
        x, y = source[int(idx)]
        if lo < 0 or x.shape != (LOOKBACK, source.F):
            raise RuntimeError(f"Bad dynamic window shape/alignment idx={idx} r={r} lo={lo} shape={tuple(x.shape)}")
        if not torch.allclose(x[-1].float(), torch.from_numpy(source.features_fp32[r]).float(), atol=0.0, rtol=0.0):
            raise RuntimeError(f"Window last row != flat row r for split={source.split} idx={idx} r={r}")
        if not torch.allclose(y.float(), source.labels[r].float(), atol=0.0, rtol=0.0):
            raise RuntimeError(f"Window label != y[r] for split={source.split} idx={idx} r={r}")
        if not any(seg_start <= lo and r < seg_end for _name, seg_start, seg_end in source.segments):
            raise RuntimeError(f"Dynamic window crosses source/week boundary for split={source.split} idx={idx} lo={lo} r={r}")
    print(f"[window-check] split={source.split} checks={checks} passed=1", flush=True)


def load_split_in_memory_ts(split_week_paths: List[Path], start: int, end: int) -> Tuple[np.ndarray, np.ndarray, int]:
    refs: List[ChunkRef] = []
    feat_dim = 0
    for wp in split_week_paths:
        wm = read_json(wp)
        feat_dim = int(wm.get("feature_dim_total", feat_dim))
        refs.extend(build_chunk_refs_by_ts(wp, start, end))
    ds = FlatWindowDataset(refs, feat_dim, split="in_memory")
    X = np.stack([ds[i][0].float().numpy() for i in range(len(ds))], axis=0) if len(ds) else np.empty((0, LOOKBACK, feat_dim), np.float32)
    y = np.stack([ds[i][1].numpy() for i in range(len(ds))], axis=0) if len(ds) else np.empty((0, NUM_HORIZONS), np.float32)
    return X.astype(np.float32, copy=False), y.astype(np.float32, copy=False), int(feat_dim)

# ---------------- Directional-noise filter quantiles from TRAIN set ----------------
def compute_dir_mask_quantiles_from_ytrain(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Label-space noise trimming: keep only mid-quantile return magnitudes per direction/horizon;
    # this is unrelated to model token dropout objectives.
    y_ret = y_train.astype(np.float32, copy=False)
    def _compute_trim_bounds(arr: np.ndarray) -> Tuple[float, float]:
        if arr.size == 0:
            return float("inf"), float("-inf")
        try:
            lo = float(np.quantile(arr, DIR_MASK_TAIL_FRACTION, method="linear"))
            hi = float(np.quantile(arr, 1.0 - DIR_MASK_TAIL_FRACTION, method="linear"))
        except TypeError:
            lo = float(np.quantile(arr, DIR_MASK_TAIL_FRACTION, interpolation="linear"))
            hi = float(np.quantile(arr, 1.0 - DIR_MASK_TAIL_FRACTION, interpolation="linear"))
        return lo, hi

    pos_lo_list = []
    pos_hi_list = []
    neg_lo_list = []
    neg_hi_list = []
    print("[directional-noise-filter quantiles]")
    for idx, horizon in enumerate(HORIZONS_MS):
        horizon_returns = y_ret[:, idx]
        pos_returns = horizon_returns[horizon_returns > 0]
        neg_returns = horizon_returns[horizon_returns < 0]
        pos_lo, pos_hi = _compute_trim_bounds(pos_returns)
        neg_lo, neg_hi = _compute_trim_bounds((-neg_returns))
        pos_lo_list.append(pos_lo); pos_hi_list.append(pos_hi)
        neg_lo_list.append(neg_lo); neg_hi_list.append(neg_hi)
        print(f"  {horizon}ms → pos:[{pos_lo:.3e}, {pos_hi:.3e}]  neg|mag:[{neg_lo:.3e}, {neg_hi:.3e}] (tail {DIR_MASK_TAIL_FRACTION:.2%})")

    pos_lo_arr = np.array(pos_lo_list, dtype=np.float32)
    pos_hi_arr = np.array(pos_hi_list, dtype=np.float32)
    neg_lo_arr = np.array(neg_lo_list, dtype=np.float32)
    neg_hi_arr = np.array(neg_hi_list, dtype=np.float32)

    pos_mask = y_ret > 0
    neg_mask = y_ret < 0
    neg_mag = -y_ret
    keep_mask = (
        (pos_mask & (y_ret >= pos_lo_arr) & (y_ret <= pos_hi_arr))
        | (neg_mask & (neg_mag >= neg_lo_arr) & (neg_mag <= neg_hi_arr))
    )
    kept_per_h = keep_mask.mean(axis=0)
    per_horizon_line = " | ".join(
        f"{horizon}ms={float(kept):.2%}" for horizon, kept in zip(HORIZONS_MS, kept_per_h)
    )
    print(f"[dir-mask] kept per horizon: {per_horizon_line}")

    main_idx = NUM_HORIZONS - 1
    main_kept = float(keep_mask[:, main_idx].mean())
    main_removed = 1.0 - main_kept
    print(
        f"[dir-mask] main horizon {HORIZONS_MS[main_idx]}ms kept={main_kept:.2%}, removed={main_removed:.2%}"
    )

    none_kept = float((~keep_mask.any(axis=1)).mean())
    all_kept = float((keep_mask.all(axis=1)).mean())
    print(f"[dir-mask] row sanity: none_kept={none_kept:.2%}, all_kept={all_kept:.2%}")

    return (
        pos_lo_arr,
        pos_hi_arr,
        neg_lo_arr,
        neg_hi_arr,
    )


def load_quantile_cache(path: Path) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            bounds = {
                "pos_lo": np.asarray(cached["pos_lo"], dtype=np.float32),
                "pos_hi": np.asarray(cached["pos_hi"], dtype=np.float32),
                "neg_lo": np.asarray(cached["neg_lo"], dtype=np.float32),
                "neg_hi": np.asarray(cached["neg_hi"], dtype=np.float32),
            }
            meta_json = str(cached["metadata_json"].item())
        metadata = json.loads(meta_json)
    except Exception as exc:
        print(f"[dir-mask-cache] invalid cache at {path}: {exc}")
        return None
    return bounds, metadata


def save_quantile_cache(path: Path, bounds: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pos_lo=np.asarray(bounds["pos_lo"], dtype=np.float32),
        pos_hi=np.asarray(bounds["pos_hi"], dtype=np.float32),
        neg_lo=np.asarray(bounds["neg_lo"], dtype=np.float32),
        neg_hi=np.asarray(bounds["neg_hi"], dtype=np.float32),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )


def quantile_cache_matches(cached_meta: Dict[str, Any], current_meta: Dict[str, Any]) -> bool:
    required_keys = (
        "tail_fraction",
        "horizons_ms",
        "train_week_keys",
        "train_ts_start",
        "train_ts_end",
        "decision_time_basis",
        "trade_history_enabled",
        "event_stream_mode",
    )
    return all(cached_meta.get(k) == current_meta.get(k) for k in required_keys)

def build_directional_noise_filter_mask_np(y_ret: np.ndarray, stats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(stats, dict):
        pos_lo = np.asarray(stats["pos_lo"], dtype=np.float32)
        pos_hi = np.asarray(stats["pos_hi"], dtype=np.float32)
        neg_lo = np.asarray(stats["neg_lo"], dtype=np.float32)
        neg_hi = np.asarray(stats["neg_hi"], dtype=np.float32)
    else:
        pos_lo, pos_hi, neg_lo, neg_hi = [np.asarray(x, dtype=np.float32) for x in stats]
    y = np.asarray(y_ret, dtype=np.float32)
    pos = y > 0
    neg = y < 0
    mag_neg = -y
    keep_pos = pos & (y >= pos_lo.reshape(1, -1)) & (y <= pos_hi.reshape(1, -1))
    keep_neg = neg & (mag_neg >= neg_lo.reshape(1, -1)) & (mag_neg <= neg_hi.reshape(1, -1))
    return keep_pos | keep_neg

def log_band_label_audit_np(split_name: str, y_ret: np.ndarray, keep: np.ndarray) -> None:
    y = np.asarray(y_ret, dtype=np.float32)
    keep = np.asarray(keep, dtype=bool)
    abs_y = np.abs(y)
    for h, horizon_ms in enumerate(HORIZONS_MS):
        signed = y[:, h] != 0.0
        kept = keep[:, h] & signed
        zero_frac = float(np.mean(~signed)) if y.shape[0] else float("nan")
        signed_kept_frac = float(np.mean(kept)) if y.shape[0] else float("nan")
        pos_kept_frac = float(np.mean(y[kept, h] > 0)) if np.any(kept) else float("nan")
        neg_kept_frac = float(np.mean(y[kept, h] < 0)) if np.any(kept) else float("nan")
        pos_frac = float(np.mean(y[signed, h] > 0)) if np.any(signed) else float("nan")
        print(
            f"[label-audit split={split_name} h={int(horizon_ms)}] zero_frac={zero_frac:.6f} "
            f"signed_kept_frac={signed_kept_frac:.6f} pos_kept_frac={pos_kept_frac:.6f} "
            f"neg_kept_frac={neg_kept_frac:.6f} pos_frac={pos_frac:.6f}",
            flush=True,
        )
        vals_abs = abs_y[kept, h]
        vals = y[kept, h]
        if vals_abs.size == 0:
            continue
        try:
            qs = np.quantile(vals_abs, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], method="linear")
        except TypeError:
            qs = np.quantile(vals_abs, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], interpolation="linear")
        bands = ("low", "mid", "high")
        denom = max(1, int(vals_abs.size))
        for bi, name in enumerate(bands):
            lo = float(qs[bi])
            hi = float(qs[bi + 1])
            if bi == len(bands) - 1:
                mask = kept & (abs_y[:, h] >= lo) & (abs_y[:, h] <= hi)
            else:
                mask = kept & (abs_y[:, h] >= lo) & (abs_y[:, h] < hi)
            n = int(np.sum(mask))
            vv = y[:, h][mask]
            aa = abs_y[:, h][mask]
            frac = float(n / denom)
            abs_p50 = float(np.quantile(aa, 0.50)) if n else float("nan")
            abs_mean = float(np.mean(aa)) if n else float("nan")
            abs_p90 = float(np.quantile(aa, 0.90)) if n else float("nan")
            band_pos_frac = float(np.mean(vv > 0)) if n else float("nan")
            ret_mean = float(np.mean(vv)) if n else float("nan")
            ret_std = float(np.std(vv, ddof=0)) if n else float("nan")
            print(
                f"[band-label split={split_name} h={int(horizon_ms)} band={name}] "
                f"n={n} frac={frac:.6f} abs_p50={abs_p50:.6g} abs_mean={abs_mean:.6g} "
                f"abs_p90={abs_p90:.6g} pos_frac={band_pos_frac:.6f} ret_mean={ret_mean:.6g} ret_std={ret_std:.6g}",
                flush=True,
            )

def make_build_directional_noise_filter_mask_torch(pos_lo, pos_hi, neg_lo, neg_hi):
    # Build a label-space noise filter mask (mid-quantile magnitude keeper), not an SSL token mask.
    pos_lo_t = torch.from_numpy(pos_lo)
    pos_hi_t = torch.from_numpy(pos_hi)
    neg_lo_t = torch.from_numpy(neg_lo)
    neg_hi_t = torch.from_numpy(neg_hi)

    def build_directional_noise_filter_mask(y_ret: torch.Tensor) -> torch.Tensor:
        pos = y_ret > 0
        neg = y_ret < 0
        lo_pos = pos_lo_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        hi_pos = pos_hi_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        lo_neg = neg_lo_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        hi_neg = neg_hi_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        mag_neg = (-y_ret).clamp_min(0.0)
        keep_pos = pos & (y_ret >= lo_pos) & (y_ret <= hi_pos)
        keep_neg = neg & (mag_neg >= lo_neg) & (mag_neg <= hi_neg)
        return keep_pos | keep_neg
    return build_directional_noise_filter_mask

def compute_directional_loss_fn(build_directional_noise_filter_mask_fn, horizon_weights: torch.Tensor):
    def compute_directional_loss(logits: torch.Tensor, y_ret: torch.Tensor) -> torch.Tensor:
        noise_filter_mask = build_directional_noise_filter_mask_fn(y_ret)
        if not noise_filter_mask.any():
            return torch.tensor(0.0, device=logits.device)
        y_dir = (y_ret > 0).float()
        losses = []
        weights = []
        for h_idx in range(NUM_HORIZONS):
            noise_filter_mask_h = noise_filter_mask[:, h_idx]
            if noise_filter_mask_h.any():
                loss_h = F.binary_cross_entropy_with_logits(
                    logits[noise_filter_mask_h, h_idx], y_dir[noise_filter_mask_h, h_idx], reduction='mean'
                )
                losses.append(loss_h)
                weights.append(horizon_weights[h_idx])
        if not losses:
            return torch.tensor(0.0, device=logits.device)
        loss_stack = torch.stack(losses)
        weight_stack = torch.stack(weights)
        return (loss_stack * weight_stack).sum() / weight_stack.sum()
    return compute_directional_loss


def get_model_state_dict_for_ckpt(model: torch.nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()

# ---------------- Train/Eval ----------------
def train_from_offline():
    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(MATMUL_PRECISION)
        except Exception as exc:
            print(f"[warn] failed to set float32 matmul precision to '{MATMUL_PRECISION}': {exc}")
    print(f"[startup] cudnn_benchmark={CUDNN_BENCHMARK} matmul_precision={MATMUL_PRECISION}")
    print(f"[startup] log_every={LOG_EVERY}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = AMP_ENABLED and device.type == "cuda"
    amp_dtype = torch.bfloat16
    print(f"[amp] enabled={amp_enabled} dtype=bf16")
    out_root = Path(OUT_ROOT)
    meta = load_global_meta(out_root)
    if meta.get("storage_format") != "flat_decision_rows_v1":
        raise ValueError("This Phase 1B CM_offline.py requires flat_decision_rows_v1. Rerun data_ingest.py.")
    validate_dataset_label_dim(meta, f"global metadata {out_root / 'meta.json'}")
    meta_aux_dim = int(meta.get("aux_dim", -1))
    meta_f_total = int(meta.get("feature_dim_total", -1))
    meta_f_core = int(meta.get("feature_dim_core", -1))
    if meta_aux_dim != AUX_DIM or meta_f_total != meta_f_core + AUX_DIM:
        raise ValueError(f"Invalid flat feature dimensions in meta.json: F={meta_f_total} core={meta_f_core} aux={meta_aux_dim}, expected aux={AUX_DIM}")
    trade_history_enabled = meta.get("trade_history_enabled")
    event_stream_mode = meta.get("event_stream_mode")
    print(f"[meta] trade_history_enabled={trade_history_enabled!r}")
    if "event_stream_mode" in meta:
        print(f"[meta] event_stream_mode={event_stream_mode!r}")
    splits = require_phase1_five_week_pipeline_splits(meta, out_root)

    pca_info = meta.get("pca", {}) or {}
    if pca_info:
        applied = bool(pca_info.get("applied", False))
        summary_parts = [f"applied={applied}"]
        var_kept = pca_info.get("var_kept")
        if isinstance(var_kept, (int, float)):
            summary_parts.append(f"var_kept={float(var_kept):.4f}")
        elif var_kept is not None:
            summary_parts.append(f"var_kept={var_kept}")
        k = pca_info.get("k")
        if k is not None:
            try:
                summary_parts.append(f"k={int(k)}")
            except (TypeError, ValueError):
                summary_parts.append(f"k={k}")
        model_path = pca_info.get("model_path")
        if model_path:
            summary_parts.append(f"model={model_path}")
        print(f"[pca-meta] {' '.join(summary_parts)}")
        if not applied:
            print("[warn] PCA metadata indicates the dataset was not reduced; training will use original feature dimensionality.")

    weeks_order = splits["weeks_in_order"]
    weeks_meta_map = meta.get("weeks_meta", {})

    key_to_meta: Dict[str, Path] = {}
    if weeks_meta_map and weeks_order:
        key_to_meta = {
            wk: out_root / weeks_meta_map[wk]
            for wk in weeks_order
            if wk in weeks_meta_map
        }

    if not key_to_meta:
        raise KeyError("meta must include non-empty 'weeks_in_order' and 'weeks_meta' for split week-key mapping")

    def keys_to_paths(keys: List[str], split_name: str) -> List[Path]:
        missing = [k for k in keys if k not in key_to_meta]
        if missing:
            raise KeyError(f"Split '{split_name}' references unknown week key(s): {missing}")
        return [key_to_meta[k] for k in keys]

    cmssl_train = splits["splits"]["cmssl"]["train"]
    cmssl_val = splits["splits"]["cmssl"]["val"]
    cmssl_test = splits["splits"]["cmssl"]["test"]
    eval_full = splits["splits"]["eval"]["full"]
    rl_train = splits["splits"]["rl"]["train"]

    train_week_keys = cmssl_train["weeks"]
    tr_weeks = keys_to_paths(train_week_keys, "cmssl.train")
    va_weeks = keys_to_paths(cmssl_val["weeks"], "cmssl.val")
    te_weeks = keys_to_paths(cmssl_test["weeks"], "cmssl.test")
    eval_weeks = keys_to_paths(eval_full["weeks"], "eval.full")
    rl_train_weeks = keys_to_paths(rl_train["weeks"], "rl.train")

    if not (tr_weeks and va_weeks and te_weeks):
        raise ValueError("CMSSL split metadata must resolve to at least one week for train/val/test")

    week1, week2, week3, week4, week5 = weeks_order
    print(
        "[cmssl weeks] "
        f"train=week1+week2({week1},{week2}) val=week3({week3}) test=week4({week4}) eval=week5({week5}) "
        f"| train_keys={train_week_keys} val_keys={cmssl_val['weeks']} test_keys={cmssl_test['weeks']}"
    )

    early_stop_patience = SINGLE_WEEK_PATIENCE if len(tr_weeks) <= 1 else PATIENCE
    if early_stop_patience != PATIENCE:
        print(f"[early-stop] using short patience={early_stop_patience} for single-week training")


    # feature/label dim sanity
    feat_dim_total = None
    resolved_split_week_paths = []
    seen_week_meta_paths: set[str] = set()
    for week_group in (
        tr_weeks, va_weeks, te_weeks, eval_weeks, rl_train_weeks
    ):
        for wp in week_group:
            wp_key = str(wp)
            if wp_key not in seen_week_meta_paths:
                seen_week_meta_paths.add(wp_key)
                resolved_split_week_paths.append(wp)

    for wp in resolved_split_week_paths:
        week_meta = read_json(wp)
        if week_meta.get("storage_format") != "flat_decision_rows_v1":
            raise ValueError("This Phase 1B CM_offline.py requires flat_decision_rows_v1. Rerun data_ingest.py.")
        validate_dataset_label_dim(week_meta, f"week metadata {wp}")
        if int(week_meta.get("aux_dim", -1)) != AUX_DIM:
            raise ValueError(f"Week metadata aux_dim mismatch in {wp}: {week_meta.get('aux_dim')} != {AUX_DIM}")
        if week_meta.get("trade_history_enabled") != trade_history_enabled:
            raise ValueError(
                "trade_history_enabled mismatch between global metadata and week metadata: "
                f"global={trade_history_enabled!r}, week={week_meta.get('trade_history_enabled')!r}, week_meta={wp}"
            )
        if "event_stream_mode" in meta:
            if week_meta.get("event_stream_mode") != event_stream_mode:
                raise ValueError(
                    "event_stream_mode mismatch between global metadata and week metadata: "
                    f"global={event_stream_mode!r}, week={week_meta.get('event_stream_mode')!r}, week_meta={wp}"
                )
        elif "event_stream_mode" in week_meta:
            raise ValueError(
                "event_stream_mode present in week metadata but missing from global metadata: "
                f"week={week_meta.get('event_stream_mode')!r}, week_meta={wp}"
            )
        fm = int(week_meta["feature_dim_total"])
        if feat_dim_total is None:
            feat_dim_total = fm
        elif feat_dim_total != fm:
            raise ValueError(f"Feature dim mismatch: saw {feat_dim_total} then {fm}")
    F_total = int(feat_dim_total or 0)

    # ---- build datasets or fully load ----
    tr_start, tr_end = int(cmssl_train["start"]), int(cmssl_train["end"])
    va_start, va_end = int(cmssl_val["start"]), int(cmssl_val["end"])
    te_start, te_end = int(cmssl_test["start"]), int(cmssl_test["end"])

    quantile_cache_path = out_root / "dir_mask_quantiles_cache.npz"
    current_meta = {
        "tail_fraction": float(DIR_MASK_TAIL_FRACTION),
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "train_week_keys": list(train_week_keys),
        "train_ts_start": int(tr_start),
        "train_ts_end": int(tr_end),
        "decision_time_basis": EXPECTED_DECISION_TIME_BASIS,
        "trade_history_enabled": trade_history_enabled,
        "event_stream_mode": event_stream_mode,
    }
    cached_quantiles = load_quantile_cache(quantile_cache_path)
    cached_bounds = None
    if cached_quantiles is None:
        print(f"[dir-mask-cache] miss path={quantile_cache_path} (quantile prepass required)")
    else:
        cached_bounds, cached_meta = cached_quantiles
        if quantile_cache_matches(cached_meta, current_meta):
            print(f"[dir-mask-cache] hit path={quantile_cache_path} (quantile prepass skipped)")
        else:
            cached_bounds = None
            print(
                "[dir-mask-cache] event-time identity mismatch "
                f"path={quantile_cache_path} (quantile prepass required)"
            )

    if USE_IN_MEMORY:
        X_tr, y_tr, feat_dim1 = load_split_in_memory_ts(tr_weeks, tr_start, tr_end)
        X_va, y_va, feat_dim2 = load_split_in_memory_ts(va_weeks, va_start, va_end)
        X_te, y_te, feat_dim3 = load_split_in_memory_ts(te_weeks, te_start, te_end)
        assert feat_dim1 == feat_dim2 == feat_dim3 == F_total, "feat dim mismatch"
        print(
            f"[cmssl split-ts] train=[{tr_start},{tr_end}) N={len(y_tr)} "
            f"val=[{va_start},{va_end}) N={len(y_va)} test=[{te_start},{te_end}) N={len(y_te)}"
        )

        # Build in-RAM datasets
        ds_train = HFTDataset(X_tr, y_tr)
        ds_val   = HFTDataset(X_va, y_va)
        ds_test  = HFTDataset(X_te, y_te)
        print(
            f"[offline-data] train windows={len(ds_train)}, "
            f"val windows={len(ds_val)}, test windows={len(ds_test)}"
        )
        if VALIDATE_DYNAMIC_WINDOWS:
            validate_dynamic_window_alignment(ds_train)
            validate_dynamic_window_alignment(ds_val)
            validate_dynamic_window_alignment(ds_test)
        # we still need y_tr to build directional mask quantiles unless cache hit
        y_train_for_quant = None if cached_bounds is not None else y_tr

    else:
        def refs_for_weeks_timerange(weeks: List[Path], start: int, end: int) -> List[ChunkRef]:
            refs: List[ChunkRef] = []
            for wp in weeks:
                refs.extend(build_chunk_refs_by_ts(wp, start, end))
            return refs

        tr_refs = refs_for_weeks_timerange(tr_weeks, tr_start, tr_end)
        va_refs = refs_for_weeks_timerange(va_weeks, va_start, va_end)
        te_refs = refs_for_weeks_timerange(te_weeks, te_start, te_end)
        print(
            f"[cmssl split-ts] train=[{tr_start},{tr_end}) N={sum(r.n for r in tr_refs)} "
            f"val=[{va_start},{va_end}) N={sum(r.n for r in va_refs)} "
            f"test=[{te_start},{te_end}) N={sum(r.n for r in te_refs)}"
        )

        ds_train = FlatWindowDataset(tr_refs, F_total, split="train")
        ds_val   = FlatWindowDataset(va_refs, F_total, split="val")
        ds_test  = FlatWindowDataset(te_refs, F_total, split="test")
        print(
            f"[offline-data] train windows={len(ds_train)}, "
            f"val windows={len(ds_val)}, test windows={len(ds_test)}"
        )
        if VALIDATE_DYNAMIC_WINDOWS:
            validate_dynamic_window_alignment(ds_train)
            validate_dynamic_window_alignment(ds_val)
            validate_dynamic_window_alignment(ds_test)

        # Build y_train_for_quant without loading features into RAM unless cache hit.
        if cached_bounds is not None:
            y_train_for_quant = None
        elif len(ds_train) == 0:
            y_train_for_quant = np.empty((0, NUM_HORIZONS), dtype=np.float32)
        else:
            y_train_for_quant = ds_train.y[ds_train.valid_rows].astype(np.float32, copy=False)


    # ---------------- directional-noise filter quantiles & loss closure ----------------
    if cached_bounds is not None:
        pos_lo = cached_bounds["pos_lo"]
        pos_hi = cached_bounds["pos_hi"]
        neg_lo = cached_bounds["neg_lo"]
        neg_hi = cached_bounds["neg_hi"]
    else:
        pos_lo, pos_hi, neg_lo, neg_hi = compute_dir_mask_quantiles_from_ytrain(y_train_for_quant)
        save_quantile_cache(
            quantile_cache_path,
            {
                "pos_lo": pos_lo,
                "pos_hi": pos_hi,
                "neg_lo": neg_lo,
                "neg_hi": neg_hi,
            },
            current_meta,
        )
        print(f"[dir-mask-cache] saved path={quantile_cache_path}")
    build_directional_noise_filter_mask = make_build_directional_noise_filter_mask_torch(pos_lo, pos_hi, neg_lo, neg_hi)
    if y_train_for_quant is not None:
        train_keep_np = build_directional_noise_filter_mask_np(y_train_for_quant, (pos_lo, pos_hi, neg_lo, neg_hi))
        log_band_label_audit_np("train", y_train_for_quant, train_keep_np)
    horizon_weights = torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32, device=device)
    horizon_weights_cpu = horizon_weights.detach().cpu().to(torch.float64)
    horizon_weights_np = horizon_weights_cpu.numpy()
    compute_directional_loss = compute_directional_loss_fn(build_directional_noise_filter_mask, horizon_weights)

    def format_metric(values: Iterable[float], fmt: str) -> str:
        formatted = []
        for horizon, value in zip(HORIZONS_MS, values):
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                formatted.append(f"{horizon}ms:nan")
            else:
                formatted.append(f"{horizon}ms:{fmt.format(val)}")
        return '[' + ', '.join(formatted) + ']'

    # ---------------- DataLoaders ----------------
    dl_train = DataLoader(
        ds_train,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS_TRAIN,
        pin_memory=True,
        prefetch_factor=8 if WORKERS_TRAIN > 0 else None,
        persistent_workers=(WORKERS_TRAIN > 0),
    )
    dl_val = DataLoader(
        ds_val,
        BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, WORKERS_VAL),
        pin_memory=True,
        persistent_workers=(max(1, WORKERS_VAL) > 0),
    )
    dl_test = DataLoader(
        ds_test,
        BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, WORKERS_VAL),
        pin_memory=True,
        persistent_workers=(max(1, WORKERS_VAL) > 0),
    )

    # ---------------- Model ----------------
    args = ModelArgs(DMODEL, MAMBA_LAYERS, F_total, LOOKBACK)
    model = SAMBA(args).to(device)
    if COMPILE_ENABLED:
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=COMPILE_MODE)
                print(f"[compile] enabled mode={COMPILE_MODE}")
            except Exception as exc:
                print(f"[warn] torch.compile failed ({exc}); continuing in eager mode")
        else:
            print("[warn] BYBIT_TORCH_COMPILE=1 but torch.compile is unavailable; continuing in eager mode")
    else:
        print("[compile] enabled=False")
    primary_metric_mode = get_primary_metric_mode()
    if USE_SAM:
        opt = SAM(model.parameters(), torch.optim.AdamW, lr=LR, weight_decay=1e-3, rho=SAM_RHO)
        optimizer_name = "SAM(AdamW)"
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        optimizer_name = "AdamW"
    print(f"[optim-config] optimizer={optimizer_name} use_sam={int(USE_SAM)} lr={LR} weight_decay=1e-3 sam_rho={SAM_RHO}")
    print(f"[data-config] train_row_stride={TRAIN_ROW_STRIDE} train_data_device={TRAIN_DATA_DEVICE} feature_storage_dtype={FEATURE_STORAGE_DTYPE_NAME}")
    torch.cuda.empty_cache()

    # ---------------- Epoch loop ----------------
    best = -float('inf') if primary_metric_mode == "max" else float('inf')
    no_imp = 0
    primary_horizon_idx = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)

    def summarize_directional_metrics(dl: DataLoader, *, primary_only: bool) -> dict:
        model.eval()
        logits_all = [[] for _ in range(NUM_HORIZONS)]
        ypos_all = [[] for _ in range(NUM_HORIZONS)]
        logits_masked = [[] for _ in range(NUM_HORIZONS)]
        ypos_masked = [[] for _ in range(NUM_HORIZONS)]
        bce_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
        bce_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
        bce_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
        bce_masked_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
        acc_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
        total = np.zeros(NUM_HORIZONS, dtype=np.float64)
        acc_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
        masked_total = np.zeros(NUM_HORIZONS, dtype=np.float64)

        with torch.no_grad():
            for x, y in dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_ret = y
                y_dir = (y_ret > 0).float()
                noise_filter_mask = build_directional_noise_filter_mask(y_ret)

                # Keep validation/test directional metrics in fp32 to avoid bf16-induced
                # logit quantization ties in AUC and logit summary statistics.
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=False):
                    dir_logits = model(x)
                    bce_elem = F.binary_cross_entropy_with_logits(dir_logits, y_dir, reduction='none')

                dir_logits_metrics = dir_logits.detach().float()
                bce_elem_fp32 = bce_elem.detach().float()
                y_dir_metrics = y_dir.detach()
                noise_filter_mask_metrics = noise_filter_mask.detach()
                pred_class = (dir_logits_metrics > 0).to(torch.int32)
                true_class = y_dir_metrics.to(torch.int32)

                horizon_indices = [primary_horizon_idx] if primary_only else range(NUM_HORIZONS)
                for h_idx in horizon_indices:
                    logits_h_all = dir_logits_metrics[:, h_idx]
                    targets_h_all = y_dir_metrics[:, h_idx]
                    bce_sum[h_idx] += bce_elem_fp32[:, h_idx].sum().item()
                    bce_count[h_idx] += targets_h_all.numel()
                    acc_sum[h_idx] += (pred_class[:, h_idx] == true_class[:, h_idx]).sum().item()
                    total[h_idx] += targets_h_all.numel()
                    logits_all[h_idx].append(logits_h_all.detach().cpu())
                    ypos_all[h_idx].append(true_class[:, h_idx].detach().cpu())

                    noise_filter_mask_h = noise_filter_mask_metrics[:, h_idx]
                    if noise_filter_mask_h.any():
                        logits_h = dir_logits_metrics[noise_filter_mask_h, h_idx]
                        targets_h = y_dir_metrics[noise_filter_mask_h, h_idx]
                        bce_masked_sum[h_idx] += bce_elem_fp32[noise_filter_mask_h, h_idx].sum().item()
                        bce_masked_count[h_idx] += noise_filter_mask_h.sum().item()
                        acc_masked_sum[h_idx] += ((logits_h > 0).to(torch.int32) == targets_h.to(torch.int32)).sum().item()
                        masked_total[h_idx] += noise_filter_mask_h.sum().item()
                        logits_masked[h_idx].append(logits_h.detach().cpu())
                        ypos_masked[h_idx].append(targets_h.to(torch.int32).detach().cpu())

        bce = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        bce_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        acc = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        acc_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        auc = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        auc_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        pos_rate_all = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        logit_mean_all = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        logit_std_all = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        pos_rate_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        logit_mean_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        logit_std_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        prob_std_masked = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
        dir_bal_acc_kept = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)

        for h_idx in ([primary_horizon_idx] if primary_only else range(NUM_HORIZONS)):
            if bce_count[h_idx] > 0:
                bce[h_idx] = bce_sum[h_idx] / bce_count[h_idx]
                acc[h_idx] = acc_sum[h_idx] / max(total[h_idx], 1.0)
            if bce_masked_count[h_idx] > 0:
                bce_masked[h_idx] = bce_masked_sum[h_idx] / bce_masked_count[h_idx]
                acc_masked[h_idx] = acc_masked_sum[h_idx] / max(masked_total[h_idx], 1.0)
            if logits_all[h_idx]:
                logits_cat = torch.cat(logits_all[h_idx], dim=0).view(-1)
                ypos_cat = torch.cat(ypos_all[h_idx], dim=0).view(-1)
                auc[h_idx] = binary_auc_from_logits(logits_cat, ypos_cat)
                pos_rate_all[h_idx] = float(ypos_cat.float().mean().item())
                logit_mean_all[h_idx] = float(logits_cat.mean().item())
                logit_std_all[h_idx] = float(logits_cat.std(unbiased=False).item())
            if logits_masked[h_idx]:
                logits_cat = torch.cat(logits_masked[h_idx], dim=0).view(-1)
                ypos_cat = torch.cat(ypos_masked[h_idx], dim=0).view(-1)
                auc_masked[h_idx] = binary_auc_from_logits(logits_cat, ypos_cat)
                pos_rate_masked[h_idx] = float(ypos_cat.float().mean().item())
                logit_mean_masked[h_idx] = float(logits_cat.mean().item())
                logit_std_masked[h_idx] = float(logits_cat.std(unbiased=False).item())
                prob_cat = torch.sigmoid(logits_cat)
                prob_std_masked[h_idx] = float(prob_cat.std(unbiased=False).item())
                pred_up = prob_cat >= 0.5
                truth = ypos_cat.bool()
                pos_m = truth
                neg_m = ~truth
                tpr = float((pred_up[pos_m] == truth[pos_m]).float().mean().item()) if pos_m.any() else float("nan")
                tnr = float((pred_up[neg_m] == truth[neg_m]).float().mean().item()) if neg_m.any() else float("nan")
                if math.isfinite(tpr) and math.isfinite(tnr):
                    dir_bal_acc_kept[h_idx] = 0.5 * (tpr + tnr)

        primary_metric_value, primary_metric_label = compute_primary_metric({"dir_auc_kept": auc_masked, "dir_bal_acc_kept": dir_bal_acc_kept})
        return {
            "val_bce_unmasked": bce,
            "val_bce_masked": bce_masked,
            "val_acc": acc,
            "val_acc_masked": acc_masked,
            "val_auc": auc,
            "val_auc_masked": auc_masked,
            "val_pos_rate_all": pos_rate_all,
            "val_logit_mean_all": logit_mean_all,
            "val_logit_std_all": logit_std_all,
            "val_pos_rate_masked": pos_rate_masked,
            "val_logit_mean_masked": logit_mean_masked,
            "val_logit_std_masked": logit_std_masked,
            "dir_auc_kept": auc_masked,
            "dir_bal_acc_kept": dir_bal_acc_kept,
            "prob_std_kept": prob_std_masked,
            "primary_metric_value": float(primary_metric_value),
            "primary_metric_label": primary_metric_label,
            "primary_masked_bce": float(bce_masked[primary_horizon_idx]),
            "primary_masked_auc": float(auc_masked[primary_horizon_idx]),
            "primary_masked_acc": float(acc_masked[primary_horizon_idx]),
        }

    def run_validation(*, full_metrics: bool) -> dict:
        return summarize_directional_metrics(dl_val, primary_only=not full_metrics)

    def make_train_loader_for_epoch(epoch: int) -> DataLoader:
        n_train = len(ds_train)
        offset = int(epoch) % int(TRAIN_ROW_STRIDE)
        indices = np.arange(offset, n_train, TRAIN_ROW_STRIDE, dtype=np.int64)
        print(f"[train-stride] epoch={epoch} stride={TRAIN_ROW_STRIDE} offset={offset} selected={len(indices)}/{n_train}")
        return DataLoader(
            Subset(ds_train, indices.tolist()),
            BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=WORKERS_TRAIN,
            pin_memory=(TRAIN_DATA_DEVICE == "cpu_pinned"),
            prefetch_factor=8 if WORKERS_TRAIN > 0 else None,
            persistent_workers=(WORKERS_TRAIN > 0),
        )

    for epoch in range(EPOCHS):
        early_stop_triggered = False
        model.train()
        dl_train_epoch = make_train_loader_for_epoch(epoch)
        pbar = tqdm(dl_train_epoch, desc=f"Ep{epoch+1}/{EPOCHS}")
        num_train_batches = len(dl_train_epoch)
        running_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        running_bce_t = torch.zeros((), device=device, dtype=torch.float32)
        n_batches = 0

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_ret = y

            if USE_SAM:
                opt.base_optimizer.zero_grad(set_to_none=True)
            else:
                opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                dir_logits = model(x)
                bce_loss = compute_directional_loss(dir_logits, y_ret)
                loss = bce_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {float(loss.detach().float().cpu())}")

            running_bce_t += bce_loss.detach().float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000)
            if USE_SAM:
                opt.first_step(zero_grad=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    dir_logits2 = model(x)
                    bce_loss2 = compute_directional_loss(dir_logits2, y_ret)
                    loss2 = bce_loss2
                if not torch.isfinite(loss2):
                    raise RuntimeError(f"Non-finite training loss in SAM pass #2: {float(loss2.detach().float().cpu())}")
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000)
                opt.second_step(zero_grad=True)
            else:
                opt.step()

            running_loss_t += loss.detach().float()
            n_batches += 1
            should_log_batch = ((batch_idx + 1) % LOG_EVERY == 0) or ((batch_idx + 1) == num_train_batches)
            if should_log_batch:
                denom = float(max(1, n_batches))
                running_loss = float(running_loss_t.detach().cpu())
                running_bce = float(running_bce_t.detach().cpu())
                pbar.set_postfix(loss=f"{(running_loss / denom):.4f}", bce=f"{(running_bce / denom):.4f}")

        epoch_train_loss = float(running_loss_t.detach().cpu()) / float(max(1, n_batches))
        epoch_train_bce = float(running_bce_t.detach().cpu()) / float(max(1, n_batches))
        print(f"[train] loss={epoch_train_loss:.4f} bce={epoch_train_bce:.4f}")

        # ---------------- Validation ----------------
        fast_val = run_validation(full_metrics=False)
        primary_metric_value = float(fast_val["primary_metric_value"])
        primary_metric_label = str(fast_val["primary_metric_label"])

        if math.isfinite(primary_metric_value):
            print(
                f"[val-fast] primary_metric({primary_metric_label})={primary_metric_value:.6f} "
                f"[masked_bce_{PRIMARY_METRIC_HORIZON_MS}ms={float(fast_val['primary_masked_bce']):.6f}, "
                f"masked_auc_{PRIMARY_METRIC_HORIZON_MS}ms={float(fast_val['primary_masked_auc']):.6f}, "
                f"masked_acc_{PRIMARY_METRIC_HORIZON_MS}ms={float(fast_val['primary_masked_acc']):.3%}]"
            )
            if is_metric_improved(primary_metric_value, best, primary_metric_mode):
                best = float(primary_metric_value)
                no_imp = 0
                full_val = run_validation(full_metrics=True)
                print(
                    f"[val] BCE(all)={format_metric(full_val['val_bce_unmasked'], '{:.5f}')}  "
                    f"BCE(mask)={format_metric(full_val['val_bce_masked'], '{:.5f}')}  "
                    f"Acc(all)={format_metric(full_val['val_acc'], '{:.3%}')}  "
                    f"Acc(mask)={format_metric(full_val['val_acc_masked'], '{:.3%}')}  "
                    f"AUC(all)={format_metric(full_val['val_auc'], '{:.3f}')}  "
                    f"AUC(mask)={format_metric(full_val['val_auc_masked'], '{:.3f}')}")
                print(
                    f"[val_diag] pos_rate(all)={format_metric(full_val['val_pos_rate_all'], '{:.3%}')}  "
                    f"logit_mean(all)={format_metric(full_val['val_logit_mean_all'], '{:.3f}')}  "
                    f"logit_std(all)={format_metric(full_val['val_logit_std_all'], '{:.3f}')}  "
                    f"pos_rate(mask)={format_metric(full_val['val_pos_rate_masked'], '{:.3%}')}  "
                    f"logit_mean(mask)={format_metric(full_val['val_logit_mean_masked'], '{:.3f}')}  "
                    f"logit_std(mask)={format_metric(full_val['val_logit_std_masked'], '{:.3f}')}  "
                    f"prob_std(kept)={format_metric(full_val['prob_std_kept'], '{:.6f}')}")
                print(
                    f"[val] primary_metric({primary_metric_label})={primary_metric_value:.6f} "
                    f"[masked_bce_{PRIMARY_METRIC_HORIZON_MS}ms={float(full_val['primary_masked_bce']):.6f}, "
                    f"masked_auc_{PRIMARY_METRIC_HORIZON_MS}ms={float(full_val['primary_masked_auc']):.6f}]"
                )
                ckpt = {
                    "epoch": epoch,
                    "state_dict": get_model_state_dict_for_ckpt(model),
                    "args": {
                        "DMODEL": DMODEL, "MAMBA_LAYERS": MAMBA_LAYERS,
                        "feat_dim": F_total, "LOOKBACK": LOOKBACK,
                        "HORIZONS_MS": HORIZONS_MS,
                        "checkpoint_schema": "phase1-direction-only-v1",
                        "trade_history_enabled": trade_history_enabled,
                        "event_stream_mode": event_stream_mode,
                        "decision_time_basis": meta.get("decision_time_basis"),
                    },
                    "best_primary_metric": best,
                }
                out_ckpt = out_root / "cm_offline_phase1_best.pt"
                torch.save(ckpt, out_ckpt)
                print(f"[ckpt] saved best to {out_ckpt}")
            else:
                no_imp += 1
                print(f"no improve {no_imp}/{early_stop_patience}")
                if no_imp >= early_stop_patience:
                    print("Early stopping triggered.")
                    early_stop_triggered = True
        else:
            print(f"[val-fast] primary_metric({primary_metric_label})=nan (skipping early stop)")

        if early_stop_triggered:
            break

        # (Optional) early stop on long stagnation
        # if no_imp > 50: break

    # ---------------- Test Evaluation ----------------
    test_metrics = summarize_directional_metrics(dl_test, primary_only=False)
    print(
        f"[test] BCE(all)={format_metric(test_metrics['val_bce_unmasked'], '{:.4e}')}  "
        f"Acc(all)={format_metric(test_metrics['val_acc'], '{:.4f}')}  "
        f"AUC(all)={format_metric(test_metrics['val_auc'], '{:.4f}')}")
    print(
        f"  BCE(mask)={format_metric(test_metrics['val_bce_masked'], '{:.4e}')}  "
        f"Acc(mask)={format_metric(test_metrics['val_acc_masked'], '{:.4f}')}  "
        f"AUC(mask)={format_metric(test_metrics['val_auc_masked'], '{:.4f}')}")
    print(
        f"[test_diag] pos_rate(all)={format_metric(test_metrics['val_pos_rate_all'], '{:.3%}')}  "
        f"logit_mean(all)={format_metric(test_metrics['val_logit_mean_all'], '{:.3f}')}  "
        f"logit_std(all)={format_metric(test_metrics['val_logit_std_all'], '{:.3f}')}  "
        f"pos_rate(mask)={format_metric(test_metrics['val_pos_rate_masked'], '{:.3%}')}  "
        f"logit_mean(mask)={format_metric(test_metrics['val_logit_mean_masked'], '{:.3f}')}  "
        f"logit_std(mask)={format_metric(test_metrics['val_logit_std_masked'], '{:.3f}')}  "
        f"prob_std(kept)={format_metric(test_metrics['prob_std_kept'], '{:.6f}')}")

    print("[done] Training complete.")

# ---------------- Lightweight HFTDataset (when loading into RAM) ----------------
class HFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        x_fp32 = torch.from_numpy(X.astype(np.float32, copy=False))
        x_stored = x_fp32.to(dtype=FEATURE_STORAGE_DTYPE)
        maybe_print_bf16_feature_debug(x_fp32[:1], x_stored[:1], enabled=BF16_FEATURE_DEBUG and FEATURE_STORAGE_DTYPE_NAME == "bf16" and len(x_fp32) > 0)
        if TRAIN_DATA_DEVICE == "cpu_pinned" and torch.cuda.is_available():
            x_stored = x_stored.pin_memory()
        elif TRAIN_DATA_DEVICE == "cuda" and torch.cuda.is_available():
            x_stored = x_stored.cuda(non_blocking=True)
        self.X = x_stored
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))
        if TRAIN_DATA_DEVICE == "cpu_pinned" and torch.cuda.is_available():
            self.y = self.y.pin_memory()
        elif TRAIN_DATA_DEVICE == "cuda" and torch.cuda.is_available():
            self.y = self.y.cuda(non_blocking=True)
    def __len__(self): return int(self.y.shape[0])
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _validate_flat_global_meta(meta: dict, out_root: Path) -> None:
    if meta.get("storage_format") != "flat_decision_rows_v1":
        raise ValueError("This Phase 1B CM_offline.py requires flat_decision_rows_v1. Rerun data_ingest.py.")
    validate_dataset_label_dim(meta, f"global metadata {out_root / 'meta.json'}")
    aux_dim = int(meta.get("aux_dim", -1))
    f_total = int(meta.get("feature_dim_total", -1))
    f_core = int(meta.get("feature_dim_core", -1))
    if aux_dim != AUX_DIM or f_total != f_core + AUX_DIM:
        raise ValueError(f"Invalid flat feature dimensions: F={f_total} core={f_core} aux={aux_dim}")


def build_sources_for_debug(out_root_str: str) -> Dict[str, Any]:
    out_root = Path(out_root_str)
    meta = load_global_meta(out_root)
    _validate_flat_global_meta(meta, out_root)
    splits = require_phase1_five_week_pipeline_splits(meta, out_root)
    weeks_order = splits["weeks_in_order"]
    weeks_meta_map = meta.get("weeks_meta", {})
    key_to_meta = {wk: out_root / weeks_meta_map[wk] for wk in weeks_order if wk in weeks_meta_map}
    def paths(keys: List[str]) -> List[Path]:
        return [key_to_meta[k] for k in keys]
    def refs_for(weeks: List[Path], start: int, end: int) -> List[ChunkRef]:
        refs: List[ChunkRef] = []
        for wp in weeks:
            wm = read_json(wp)
            if wm.get("storage_format") != "flat_decision_rows_v1":
                raise ValueError("This Phase 1B CM_offline.py requires flat_decision_rows_v1. Rerun data_ingest.py.")
            refs.extend(build_chunk_refs_by_ts(wp, start, end))
        return refs
    f_total = int(meta["feature_dim_total"])
    cmssl = splits["splits"]["cmssl"]
    train = cmssl["train"]; val = cmssl["val"]; test = cmssl["test"]
    sources = {
        "train": FlatWindowDataset(refs_for(paths(train["weeks"]), int(train["start"]), int(train["end"])), f_total, split="train"),
        "val": FlatWindowDataset(refs_for(paths(val["weeks"]), int(val["start"]), int(val["end"])), f_total, split="val"),
        "test": FlatWindowDataset(refs_for(paths(test["weeks"]), int(test["start"]), int(test["end"])), f_total, split="test"),
        "feature_dim_total": f_total,
    }
    if VALIDATE_DYNAMIC_WINDOWS:
        validate_dynamic_window_alignment(sources["train"])
        validate_dynamic_window_alignment(sources["val"])
        validate_dynamic_window_alignment(sources["test"])
    return sources

# ---------------- Entry ----------------
if __name__ == "__main__":
    if "--dry-run-data-check" in sys.argv:
        if not OUT_ROOT:
            raise SystemExit("BYBIT_OUT_ROOT must be set for --dry-run-data-check")
        build_sources_for_debug(OUT_ROOT)
        print("[dry-run-data-check] passed")
    else:
        train_from_offline()
