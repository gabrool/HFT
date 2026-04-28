
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
    LOW_ABS_TRIM_FRACTION, HIGH_ABS_TRIM_FRACTION, TARGET_TRANSFORM, TARGET_TASK, CHECKPOINT_SCHEMA,
    MODEL_OUTPUT_SCHEMA,
    DIR_LOSS_WEIGHT, MAG_LOSS_WEIGHT, MAG_CORR_LOSS_WEIGHT, EMA_DECAY,
    FEATURE_SCHEMA, AUX_SCHEMA, FEATURE_AUX_TAIL,
    SINGLE_WEEK_PATIENCE, get_primary_metric_mode, compute_primary_metric, is_metric_improved,
    derive_dir_mag_predictions, derive_mag_pred_sqrt_for_mag_loss,
    SAM,
    build_dataset_from_split,
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
EXPECTED_DECISION_TIME_BASIS = "ob_event_time"
EXPECTED_DECISION_POLICY = "ob_event_time"
FAST_VAL_MAX_ROWS = 200_000
FULL_VAL_EVERY = 5
TRAIN_ROW_STRIDE = int(os.environ.get("BYBIT_TRAIN_ROW_STRIDE", "10"))
if TRAIN_ROW_STRIDE < 1:
    raise ValueError(f"BYBIT_TRAIN_ROW_STRIDE must be >= 1, got {TRAIN_ROW_STRIDE}")

assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"

def require_four_week_pipeline_splits(meta: dict, out_root: Path) -> dict:
    if "splits" not in meta:
        raise KeyError(
            "meta.json missing required key 'splits'. Run offline_ingest to generate offline dataset metadata."
        )
    splits = meta["splits"]
    if not isinstance(splits, dict):
        raise KeyError("meta['splits'] must be a dict. Rerun offline_ingest.")

    if "weeks_in_order" not in meta:
        raise KeyError("meta.json missing required key 'weeks_in_order'. Rerun offline_ingest.")
    weeks_in_order = meta["weeks_in_order"]
    if not isinstance(weeks_in_order, list) or len(weeks_in_order) != 4 or not all(isinstance(w, str) and w for w in weeks_in_order):
        raise KeyError("meta['weeks_in_order'] must be a list[str] with exactly 4 entries. Rerun offline_ingest.")

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

    if splits.get("protocol") != "four_week_cmssl_val_test_rl_eval_v2":
        raise ValueError(
            "meta['splits']['protocol'] must be 'four_week_cmssl_val_test_rl_eval_v2'. Rerun offline_ingest."
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
            if not isinstance(decision_ts_range, dict):
                raise KeyError(
                    f"meta['splits']['{stage}'] must include decision_ts_range with start/end. Rerun offline_ingest."
                )
            if "start" not in decision_ts_range or "end" not in decision_ts_range:
                raise KeyError(
                    f"meta['splits']['{stage}']['decision_ts_range'] must include start/end. Rerun offline_ingest."
                )
            try:
                start = int(decision_ts_range["start"])
                end = int(decision_ts_range["end"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"meta['splits']['{stage}']['decision_ts_range'] start/end must be integers. Rerun offline_ingest."
                )
            if start >= end:
                raise ValueError(
                    f"meta['splits']['{stage}']['decision_ts_range'] must satisfy start < end. Rerun offline_ingest."
                )
        else:
            explicit_start = entry.get("start")
            explicit_end = entry.get("end")
            if explicit_start is None or explicit_end is None:
                if isinstance(decision_ts_range, dict):
                    explicit_start = decision_ts_range.get("start")
                    explicit_end = decision_ts_range.get("end")
            if explicit_start is not None and explicit_end is not None:
                try:
                    start = int(explicit_start)
                    end = int(explicit_end)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"meta['splits']['{stage}'] explicit start/end must be integers. Rerun offline_ingest."
                    )
            else:
                start, end = _full_week_range(weeks[0], stage)
            if start >= end:
                raise ValueError(
                    f"meta['splits']['{stage}'] must satisfy start < end. Rerun offline_ingest."
                )

        return {"weeks": weeks, "start": start, "end": end}

    required_entries = {
        "cmssl.train": ("cmssl", "train", False),
        "cmssl.val": ("cmssl", "val", False),
        "cmssl.test": ("cmssl", "test", False),
        "rl.train": ("rl", "train", True),
        "rl.val": ("rl", "val", True),
        "rl.test": ("rl", "test", True),
        "eval.full": ("eval", "full", False),
    }

    normalized = {"protocol": splits["protocol"]}
    for section in ("cmssl", "rl", "eval"):
        sec = splits.get(section)
        if not isinstance(sec, dict):
            raise KeyError(f"meta['splits']['{section}'] must be a dict. Rerun offline_ingest.")
        normalized[section] = {}

    for label, (section, name, require_range) in required_entries.items():
        normalized[section][name] = _normalize_split_entry(label, splits[section].get(name), require_range=require_range)

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
        "splits": normalized,
        "weeks_in_order": weeks_in_order,
    }


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
        and meta.get("aux_schema") == AUX_SCHEMA
        and meta.get("checkpoint_schema_expected") == CHECKPOINT_SCHEMA
        and meta.get("target_transform") == TARGET_TRANSFORM
        and meta.get("target_task") == TARGET_TASK
        and int(meta.get("label_dim", -1)) == NUM_HORIZONS
        and int(meta.get("aux_dim", -1)) == AUX_DIM
        and list(meta.get("aux_names", [])) == list(FEATURE_AUX_TAIL)
        and int(meta.get("feature_dim_total", -1)) == int(meta.get("feature_dim_core", -1)) + AUX_DIM
        and bool(meta.get("feature_names_pre_pca"))
        and int(meta.get("feature_dim_core_pre_pca", -1)) == len(meta.get("feature_names_pre_pca", []))
        and bool(meta.get("feature_names_hash"))
        and bool(meta.get("pca", {}).get("applied", False))
        and int(meta.get("pca", {}).get("k", -1)) == int(meta.get("feature_dim_core", -1))
    )
    if not ok:
        raise ValueError(
            "Old or incompatible offline dataset. Rerun offline_ingest.py with "
            f"FEATURE_SCHEMA={FEATURE_SCHEMA}. "
            f"Expected TARGET_TASK={TARGET_TASK}, TARGET_TRANSFORM={TARGET_TRANSFORM}, CHECKPOINT_SCHEMA={CHECKPOINT_SCHEMA}."
        )


def validate_week_matches_global(global_meta: dict, week_meta: dict, source: str) -> None:
    checks = {
        "feature_schema": (week_meta.get("feature_schema"), global_meta.get("feature_schema")),
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
        "feature_dim_core_pre_pca": (
            int(week_meta.get("feature_dim_core_pre_pca", -1)),
            int(global_meta.get("feature_dim_core_pre_pca", -2)),
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

    if list(week_meta.get("feature_names_pre_pca", [])) != list(global_meta.get("feature_names_pre_pca", [])):
        raise ValueError(
            "Week metadata feature_names_pre_pca do not match global metadata. "
            f"source={source}. Rerun offline_ingest.py with FEATURE_SCHEMA={FEATURE_SCHEMA}."
        )

    week_pca = week_meta.get("pca", {}) or {}
    global_pca = global_meta.get("pca", {}) or {}
    if bool(week_pca.get("applied", False)) != bool(global_pca.get("applied", False)):
        raise ValueError(
            "Week metadata PCA applied flag does not match global metadata. "
            f"source={source}. Rerun offline_ingest.py with FEATURE_SCHEMA={FEATURE_SCHEMA}."
        )
    if int(week_pca.get("k", -1)) != int(global_pca.get("k", -2)):
        raise ValueError(
            "Week metadata PCA k does not match global metadata. "
            f"source={source}. Rerun offline_ingest.py with FEATURE_SCHEMA={FEATURE_SCHEMA}."
        )


# ---------------- Signed-raw preprocessing, cache, and metrics ----------------


def build_abs_trim_mask(y_raw_bps: np.ndarray, abs_lo_raw_bps: np.ndarray, abs_hi_raw_bps: np.ndarray) -> np.ndarray:
    abs_y = np.abs(y_raw_bps)
    lo = abs_lo_raw_bps.reshape(1, -1)
    hi = abs_hi_raw_bps.reshape(1, -1)
    return (abs_y >= lo) & (abs_y <= hi)


def compute_signed_raw_stats(y_train: np.ndarray) -> Dict[str, np.ndarray]:
    abs_y = np.abs(y_train)
    abs_lo = np.quantile(abs_y, LOW_ABS_TRIM_FRACTION, axis=0).astype(np.float32)
    abs_hi = np.quantile(abs_y, 1.0 - HIGH_ABS_TRIM_FRACTION, axis=0).astype(np.float32)
    keep = build_abs_trim_mask(y_train, abs_lo, abs_hi)
    q50 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    q85 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    for h in range(NUM_HORIZONS):
        kept_abs = abs_y[keep[:, h], h]
        if kept_abs.size:
            q50[h] = float(np.quantile(kept_abs, 0.50))
            q85[h] = float(np.quantile(kept_abs, 0.85))
    return {
        'abs_lo_raw_bps': abs_lo,
        'abs_hi_raw_bps': abs_hi,
        'kept_q50_abs_raw_bps': q50,
        'kept_q85_abs_raw_bps': q85,
    }


def load_stats_cache(path: Path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as c:
        stats = {k: np.asarray(c[k], dtype=np.float32) for k in ('abs_lo_raw_bps','abs_hi_raw_bps','kept_q50_abs_raw_bps','kept_q85_abs_raw_bps')}
        meta = json.loads(str(c['metadata_json'].item()))
    return stats, meta


def save_stats_cache(path: Path, stats: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    np.savez_compressed(path, **stats, metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_))


def cache_matches(cached_meta: Dict[str, Any], current_meta: Dict[str, Any]) -> bool:
    keys = ('low_abs_trim_fraction','high_abs_trim_fraction','horizons_ms','train_week_keys','train_ts_start','train_ts_end','decision_time_basis','trade_history_enabled','event_stream_mode','target_transform','label_units','target_task','loss_weighting_schema','ranking_schema')
    return all(cached_meta.get(k)==current_meta.get(k) for k in keys)


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
        self.features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32, non_blocking=False)
        if self.features.dtype != torch.float32:
            raise ValueError(f"Expected float32 features, got {self.features.dtype}")
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

        if stride == 1:
            selected = torch.arange(n_total, device=self.device)
        else:
            g_offset = torch.Generator(device=self.device)
            g_offset.manual_seed(self.seed + 1_000_003 + int(epoch))
            offset = int(torch.randint(
                low=0,
                high=stride,
                size=(1,),
                device=self.device,
                generator=g_offset,
            ).item())
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
        features_cpu = torch.from_numpy(features_np)
        if self.pin_memory:
            features_cpu = features_cpu.pin_memory()
        self.features = features_cpu
        if self.features.dtype != torch.float32:
            raise ValueError(f"Expected float32 features, got {self.features.dtype}")

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

        if stride == 1:
            selected = torch.arange(n_total)
        else:
            g_offset = torch.Generator(device="cpu")
            g_offset.manual_seed(self.seed + 1_000_003 + int(epoch))
            offset = int(torch.randint(
                low=0,
                high=stride,
                size=(1,),
                generator=g_offset,
            ).item())
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
    abs_lo: np.ndarray,
    abs_hi: np.ndarray,
    q50: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    abs_y = np.abs(y)
    keep = (abs_y >= abs_lo.reshape(1, -1)) & (abs_y <= abs_hi.reshape(1, -1))
    active = keep & (abs_y >= q50.reshape(1, -1))
    pos_w = np.ones((NUM_HORIZONS,), dtype=np.float32)
    neg_w = np.ones((NUM_HORIZONS,), dtype=np.float32)
    for h in range(NUM_HORIZONS):
        m = active[:, h]
        if int(m.sum()) < 100:
            continue
        pos = float(np.sum(y[m, h] > 0))
        neg = float(np.sum(y[m, h] < 0))
        total = pos + neg
        if total <= 0:
            continue
        pos_frac = max(pos / total, 1e-6)
        neg_frac = max(neg / total, 1e-6)
        pos_w[h] = float(np.clip(0.5 / pos_frac, 0.75, 1.25))
        neg_w[h] = float(np.clip(0.5 / neg_frac, 0.75, 1.25))
    return pos_w, neg_w


def compute_dir_mag_loss(
    pred: Dict[str, torch.Tensor],
    y_raw: torch.Tensor,
    *,
    abs_lo_t: torch.Tensor,
    abs_hi_t: torch.Tensor,
    q50_t: torch.Tensor,
    q85_t: torch.Tensor,
    hwt: torch.Tensor,
    dir_pos_w_t: torch.Tensor,
    dir_neg_w_t: torch.Tensor,
    ema_state: LossEmaState,
    update_ema: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    abs_raw = torch.abs(y_raw)
    keep = (abs_raw >= abs_lo_t) & (abs_raw <= abs_hi_t)
    tau_t = torch.clamp(0.10 * (q85_t - q50_t), min=0.05)
    active_w = torch.sigmoid((abs_raw - q50_t) / tau_t)
    strong_w = torch.sigmoid((abs_raw - q85_t) / tau_t)

    dir_logits = pred["dir_logits"]
    dir_target = (y_raw > 0).to(dtype=dir_logits.dtype)
    dir_class_w = torch.where(dir_target > 0.5, dir_pos_w_t, dir_neg_w_t)
    dir_w = keep.to(dtype=dir_logits.dtype) * hwt * dir_class_w
    dir_w = dir_w * torch.clamp(0.25 + 0.75 * active_w + 0.25 * strong_w, 0.25, 1.25)
    dir_weight_sum = dir_w.sum()
    if float(dir_weight_sum.detach().item()) <= 0.0:
        raise ValueError("Direction loss has zero effective weight; check keep masks and training data.")
    dir_raw = F.binary_cross_entropy_with_logits(dir_logits, dir_target, reduction="none")
    dir_bce = (dir_raw * dir_w).sum() / dir_weight_sum.clamp_min(1e-9)

    mag_target = torch.sqrt(abs_raw.clamp_min(0.0))
    up_mask = keep & (y_raw > 0)
    down_mask = keep & (y_raw < 0)
    mag_w_base = keep.to(dtype=dir_logits.dtype) * hwt
    mag_w_base = mag_w_base * torch.clamp(0.50 + 0.50 * active_w + 0.25 * strong_w, 0.50, 1.25)
    up_d = F.huber_loss(pred["mag_up_sqrt"], mag_target, delta=1.0, reduction="none")
    down_d = F.huber_loss(pred["mag_down_sqrt"], mag_target, delta=1.0, reduction="none")
    up_w = mag_w_base * up_mask.to(dtype=dir_logits.dtype)
    down_w = mag_w_base * down_mask.to(dtype=dir_logits.dtype)
    mag_num = (up_d * up_w).sum() + (down_d * down_w).sum()
    mag_den = up_w.sum() + down_w.sum()
    mag_huber = mag_num / mag_den.clamp_min(1e-9) if float(mag_den.detach().item()) > 0.0 else dir_logits.sum() * 0.0

    mag_pred_sqrt = derive_mag_pred_sqrt_for_mag_loss(pred)
    corr_terms = []
    for h in range(NUM_HORIZONS):
        mask = keep[:, h] & (abs_raw[:, h] >= q50_t[0, h])
        if int(mask.sum().item()) >= 2:
            px = mag_pred_sqrt[:, h][mask]
            ty = mag_target[:, h][mask]
            px = px - px.mean()
            ty = ty - ty.mean()
            den = torch.sqrt((px * px).sum() * (ty * ty).sum()).clamp_min(1e-9)
            corr = (px * ty).sum() / den
            corr_terms.append(1.0 - corr)
    if corr_terms:
        mag_corr = torch.stack(corr_terms).mean()
        mag_corr_valid = True
    else:
        mag_corr = dir_logits.sum() * 0.0
        mag_corr_valid = False

    if update_ema:
        ema_state.update("dir_bce", dir_bce, valid=True)
        ema_state.update("mag_huber", mag_huber, valid=float(mag_den.detach().item()) > 0.0)
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
        "dir_weight_sum": dir_weight_sum.detach(),
        "mag_weight_sum": mag_den.detach(),
    }
    return loss, components


def summarize_metrics(model, source, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch: int = 0):
    model.eval()
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
        out["dir_bal_acc_q50plus"] = [float("nan")] * NUM_HORIZONS
        out["primary_metric_guard_passed"] = False
        return out
    dir_logits = np.concatenate(dir_parts, axis=0)
    p_up = 1.0 / (1.0 + np.exp(-dir_logits))
    mag_up_sqrt = np.concatenate(up_parts, axis=0)
    mag_down_sqrt = np.concatenate(down_parts, axis=0)
    y_raw = np.concatenate(y_parts, axis=0)
    mag_up_bps = mag_up_sqrt ** 2
    mag_down_bps = mag_down_sqrt ** 2
    mag_pred_sqrt = p_up * mag_up_sqrt + (1.0 - p_up) * mag_down_sqrt
    edge_bps = p_up * mag_up_bps - (1.0 - p_up) * mag_down_bps
    abs_raw = np.abs(y_raw)
    keep = build_abs_trim_mask(y_raw, stats['abs_lo_raw_bps'], stats['abs_hi_raw_bps'])
    true_up = y_raw > 0
    pred_up = p_up >= 0.5
    if primary_only:
        out["edge_spearman_q50plus"] = [float("nan")] * NUM_HORIZONS
        out["dir_bal_acc_q50plus"] = [float("nan")] * NUM_HORIZONS
        h = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)
        q50plus = keep[:, h] & (abs_raw[:, h] >= float(stats['kept_q50_abs_raw_bps'][h]))
        out["edge_spearman_q50plus"][h] = _safe_spearman_np(edge_bps[:, h][q50plus], y_raw[:, h][q50plus]) if int(np.sum(q50plus)) >= 2 else float("nan")
        out["dir_bal_acc_q50plus"][h] = _balanced_acc_np(pred_up[:, h][q50plus], true_up[:, h][q50plus]) if int(np.sum(q50plus)) >= 2 else float("nan")
        out["primary_dir_bal_acc"] = out["dir_bal_acc_q50plus"][h]
        out["primary_metric_guard_passed"] = bool(math.isfinite(out["primary_dir_bal_acc"]) and out["primary_dir_bal_acc"] >= PRIMARY_DIR_BAL_ACC_GUARD)
        return out

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
    out["primary_dir_bal_acc"] = float(out["dir_bal_acc_q50plus"][primary_idx])
    out["primary_metric_guard_passed"] = bool(math.isfinite(out["primary_dir_bal_acc"]) and out["primary_dir_bal_acc"] >= PRIMARY_DIR_BAL_ACC_GUARD)
    return out


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def get_model_state_dict(model: torch.nn.Module) -> dict:
    return unwrap_model(model).state_dict()


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
        f"feature_dim_core={meta.get('feature_dim_core')} "
        f"feature_dim_total={meta.get('feature_dim_total')} "
        f"feature_names_hash={meta.get('feature_names_hash')}",
        flush=True,
    )
    trade_history_enabled = meta.get('trade_history_enabled')
    event_stream_mode = meta.get('event_stream_mode')
    splits = require_four_week_pipeline_splits(meta, out_root)

    cmssl_train = splits['splits']['cmssl']['train']
    cmssl_val = splits['splits']['cmssl']['val']
    cmssl_test = splits['splits']['cmssl']['test']

    ds_train = build_dataset_from_split(str(out_root), cmssl_train)
    ds_val = build_dataset_from_split(str(out_root), cmssl_val)
    ds_test = build_dataset_from_split(str(out_root), cmssl_test)
    F_total = int(meta.get("feature_dim_total", 0))
    if F_total != int(ds_train.feature_dim_total):
        raise ValueError(f"Feature dimension mismatch: meta={F_total}, train_dataset={int(ds_train.feature_dim_total)}")
    if int(ds_train.lookback) != int(LOOKBACK):
        raise ValueError(f"LOOKBACK mismatch: config={LOOKBACK}, train_dataset={int(ds_train.lookback)}")
    for split_name, ds in (("train", ds_train), ("val", ds_val), ("test", ds_test)):
        if len(ds.stores) != 1:
            raise ValueError(f"{split_name} split must have exactly one store/week, got {len(ds.stores)}")
        if ds.week_ids.size and not np.all(ds.week_ids == 0):
            raise ValueError(f"{split_name} split week_ids must all be 0 for single-week protocol")
        if len(ds) > 0 and int(ds.row_idx.min()) < int(LOOKBACK - 1):
            raise ValueError(
                f"{split_name} split has rows without full history: min_row_idx={int(ds.row_idx.min())}, lookback={LOOKBACK}"
            )

    tr_start,tr_end=int(cmssl_train['start']),int(cmssl_train['end'])
    va_start,va_end=int(cmssl_val['start']),int(cmssl_val['end'])
    te_start,te_end=int(cmssl_test['start']),int(cmssl_test['end'])

    cache_path=out_root/'signed_raw_stats_cache.npz'
    cache_meta={
        'low_abs_trim_fraction': float(LOW_ABS_TRIM_FRACTION),
        'high_abs_trim_fraction': float(HIGH_ABS_TRIM_FRACTION),
        'horizons_ms':[int(h) for h in HORIZONS_MS], 'train_week_keys': list(cmssl_train['weeks']),
        'train_ts_start': int(tr_start), 'train_ts_end': int(tr_end), 'decision_time_basis': EXPECTED_DECISION_TIME_BASIS,
        'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
        'target_transform': TARGET_TRANSFORM, 'label_units': 'signed_log_return_bps', 'target_task': TARGET_TASK,
        'loss_weighting_schema': 'dir_mag_smooth_q50_q85_class_bal_v1',
        'ranking_schema': 'tie_aware_average_ranks_v1'
    }
    cached=load_stats_cache(cache_path); stats=None
    if cached and cache_matches(cached[1], cache_meta): stats=cached[0]
    if stats is None:
        y_train=np.asarray(ds_train.y, dtype=np.float32)
        stats=compute_signed_raw_stats(y_train)
        save_stats_cache(cache_path,stats,cache_meta)
    else:
        y_train = np.asarray(ds_train.y, dtype=np.float32)
    dir_pos_w, dir_neg_w = compute_dir_class_weights_from_train_labels(
        y_train,
        abs_lo=stats["abs_lo_raw_bps"],
        abs_hi=stats["abs_hi_raw_bps"],
        q50=stats["kept_q50_abs_raw_bps"],
    )
    print(f"[train_stats] dir_pos_w={dir_pos_w.tolist()} dir_neg_w={dir_neg_w.tolist()}")

    train_src = GPUWindowBatchSource(
        ds_train,
        device,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        row_stride=TRAIN_ROW_STRIDE,
    )
    val_full_src = CPUWindowBatchSource(
        ds_val,
        device,
        BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        row_stride=1,
    )
    val_fast_src = val_full_src.make_evenly_spaced_subset(FAST_VAL_MAX_ROWS)
    print(
        f"[gpu_data] train rows={train_src.n_rows} train_row_stride={train_src.row_stride} "
        f"effective_rows_nominal={train_src.effective_rows_nominal} "
        f"feature_shape={train_src.feature_shape} "
        f"feature_gb={train_src.feature_gb:.3f} label_index_gb={train_src.label_index_gb:.3f}"
    )
    print(
        f"[cpu_val_data] val_full rows={val_full_src.n_rows} row_stride={val_full_src.row_stride} "
        f"feature_shape={val_full_src.feature_shape} "
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

    if COMPILE_ENABLED and hasattr(torch, "compile"):
        model = torch.compile(model, mode=COMPILE_MODE, dynamic=False)
        print(f"[compile] enabled full-model compile with {COMPILE_MODE} (dynamic=False)", flush=True)
        
    opt=SAM(model.parameters(), torch.optim.AdamW, lr=LR, weight_decay=1e-3, rho=0.01)
    primary_metric_mode=get_primary_metric_mode()
    best=-float('inf') if primary_metric_mode=='max' else float('inf')
    no_imp = 0
    early_stop_patience = SINGLE_WEEK_PATIENCE if len(cmssl_train['weeks']) <= 1 else PATIENCE

    abs_lo_t=torch.tensor(stats['abs_lo_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    abs_hi_t=torch.tensor(stats['abs_hi_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
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
        mag_corr_sum = torch.zeros((), device=device)
        dir_norm_sum = torch.zeros((), device=device)
        mag_norm_sum = torch.zeros((), device=device)
        corr_norm_sum = torch.zeros((), device=device)
        n_batches = 0
        first_batch_checked = False
        for x, y_raw in tqdm(train_src.iter_epoch(epoch), total=len(train_src), desc=f"Ep{epoch+1}/{EPOCHS}"):
            opt.base_optimizer.zero_grad(set_to_none=True)
            
            # First forward/backward: compute the raw gradient used to choose SAM's adversarial perturbation.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x)
                if not first_batch_checked:
                    assert isinstance(pred, dict)
                    assert set(pred.keys()) == {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
                    for key in pred:
                        assert pred[key].shape == y_raw.shape
                    first_batch_checked = True
                loss, comps = compute_dir_mag_loss(
                    pred,
                    y_raw,
                    abs_lo_t=abs_lo_t,
                    abs_hi_t=abs_hi_t,
                    q50_t=q50_t,
                    q85_t=q85_t,
                    hwt=hwt,
                    dir_pos_w_t=dir_pos_w_t,
                    dir_neg_w_t=dir_neg_w_t,
                    ema_state=ema_state,
                    update_ema=True,
                )
            
            loss.backward()
            
            # Do NOT clip here. SAM first_step must see the raw gradient direction.
            opt.first_step(zero_grad=True)
            
            # Second forward/backward at perturbed weights.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred2 = model(x)
                loss2, comps2 = compute_dir_mag_loss(
                    pred2,
                    y_raw,
                    abs_lo_t=abs_lo_t,
                    abs_hi_t=abs_hi_t,
                    q50_t=q50_t,
                    q85_t=q85_t,
                    hwt=hwt,
                    dir_pos_w_t=dir_pos_w_t,
                    dir_neg_w_t=dir_neg_w_t,
                    ema_state=ema_state,
                    update_ema=False,
                )
            
            loss2.backward()
            
            # Clip only before the actual optimizer update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.second_step(zero_grad=True)

            loss_sum += loss.detach()
            dir_bce_sum += comps["dir_bce"]
            mag_huber_sum += comps["mag_huber"]
            mag_corr_sum += comps["mag_corr"]
            dir_norm_sum += comps["dir_norm"]
            mag_norm_sum += comps["mag_norm"]
            corr_norm_sum += comps["corr_norm"]
            n_batches += 1

        train_sec = time.perf_counter() - train_t0
        train_loss = (loss_sum / max(1, n_batches)).item()
        train_dir_bce = (dir_bce_sum / max(1, n_batches)).item()
        train_mag_huber = (mag_huber_sum / max(1, n_batches)).item()
        train_mag_corr = (mag_corr_sum / max(1, n_batches)).item()
        train_dir_norm = (dir_norm_sum / max(1, n_batches)).item()
        train_mag_norm = (mag_norm_sum / max(1, n_batches)).item()
        train_corr_norm = (corr_norm_sum / max(1, n_batches)).item()
        print(f"[train] loss={train_loss:.6f} dir_bce={train_dir_bce:.6f} mag_huber={train_mag_huber:.6f} mag_corr={train_mag_corr:.6f} dir_norm={train_dir_norm:.6f} mag_norm={train_mag_norm:.6f} corr_norm={train_corr_norm:.6f}")

        val_fast_t0 = time.perf_counter()
        val_fast=summarize_metrics(model, val_fast_src, device, stats, amp_enabled, amp_dtype, primary_only=True, epoch=epoch)
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
        full_val_sec = 0.0
        improved = math.isfinite(primary_metric_value) and is_metric_improved(primary_metric_value,best,primary_metric_mode)
        if improved:
            best=float(primary_metric_value); no_imp=0
            full_t0 = time.perf_counter()
            full=summarize_metrics(model, val_full_src, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch=epoch)
            full_val_sec += time.perf_counter() - full_t0
            print(f"[val_dir] dir_auc_q50plus={full['dir_auc_q50plus']} dir_bal_acc_q50plus={full['dir_bal_acc_q50plus']} dir_pos_frac_pred_q50plus={full['dir_pos_frac_pred_q50plus']} dir_pos_frac_true_q50plus={full['dir_pos_frac_true_q50plus']}")
            print(f"[val_mag] mag_spearman_abs_q50plus={full['mag_spearman_abs_q50plus']} pred_abs_bps_p50_kept={full['pred_abs_bps_p50_kept']} true_abs_bps_p50_kept={full['true_abs_bps_p50_kept']} pred_abs_bps_p90_kept={full['pred_abs_bps_p90_kept']} true_abs_bps_p90_kept={full['true_abs_bps_p90_kept']} pred_abs_std_over_true_abs_std_kept={full['pred_abs_std_over_true_abs_std_kept']}")
            print(f"[val_edge] edge_spearman_q50plus={full['edge_spearman_q50plus']} edge_bal_sign_acc_q50plus={full['edge_bal_sign_acc_q50plus']} edge_mean_kept={full['edge_mean_kept']} edge_std_kept={full['edge_std_kept']} edge_abs_p90_kept={full['edge_abs_p90_kept']}")
            ckpt={
                'epoch': epoch,
                'state_dict': get_model_state_dict(model),
                'args': {
                    'DMODEL':DMODEL, 'MAMBA_LAYERS':MAMBA_LAYERS, 'feat_dim':F_total, 'LOOKBACK':LOOKBACK,
                    'WINDOW_MS': WINDOW_MS, 'HORIZONS_MS': HORIZONS_MS, 'checkpoint_schema': CHECKPOINT_SCHEMA,
                    'model_output_schema': MODEL_OUTPUT_SCHEMA,
                    'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
                    'decision_time_basis': meta.get('decision_time_basis'), 'decision_stride_policy':'every_ob_event',
                    'label_delta_ms':0, 'label_units':'signed_log_return_bps',
                    'target_task': TARGET_TASK,
                    'target_transform': TARGET_TRANSFORM,
                    'low_abs_trim_fraction': float(LOW_ABS_TRIM_FRACTION),
                    'high_abs_trim_fraction': float(HIGH_ABS_TRIM_FRACTION),
                    'primary_metric': PRIMARY_METRIC,
                    'primary_metric_horizon_ms': PRIMARY_METRIC_HORIZON_MS,
                    'primary_dir_bal_acc_guard': PRIMARY_DIR_BAL_ACC_GUARD,
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
                full_periodic = summarize_metrics(model, val_full_src, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch=epoch)
                full_val_sec += time.perf_counter() - full_t0
                print(f"[val_dir] dir_auc_q50plus={full_periodic['dir_auc_q50plus']} dir_bal_acc_q50plus={full_periodic['dir_bal_acc_q50plus']} dir_pos_frac_pred_q50plus={full_periodic['dir_pos_frac_pred_q50plus']} dir_pos_frac_true_q50plus={full_periodic['dir_pos_frac_true_q50plus']}")
                print(f"[val_mag] mag_spearman_abs_q50plus={full_periodic['mag_spearman_abs_q50plus']} pred_abs_bps_p50_kept={full_periodic['pred_abs_bps_p50_kept']} true_abs_bps_p50_kept={full_periodic['true_abs_bps_p50_kept']} pred_abs_bps_p90_kept={full_periodic['pred_abs_bps_p90_kept']} true_abs_bps_p90_kept={full_periodic['true_abs_bps_p90_kept']} pred_abs_std_over_true_abs_std_kept={full_periodic['pred_abs_std_over_true_abs_std_kept']}")
                print(f"[val_edge] edge_spearman_q50plus={full_periodic['edge_spearman_q50plus']} edge_bal_sign_acc_q50plus={full_periodic['edge_bal_sign_acc_q50plus']} edge_mean_kept={full_periodic['edge_mean_kept']} edge_std_kept={full_periodic['edge_std_kept']} edge_abs_p90_kept={full_periodic['edge_abs_p90_kept']}")
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
    if ckpt_args.get("model_output_schema") != MODEL_OUTPUT_SCHEMA:
        raise ValueError(f"Model output schema mismatch: got {ckpt_args.get('model_output_schema')}, expected {MODEL_OUTPUT_SCHEMA}")
    unwrap_model(model).load_state_dict(state, strict=True)
    model.eval()

    del train_src
    del val_fast_src
    del val_full_src
    if device.type == 'cuda':
        torch.cuda.empty_cache()

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
        f"feature_shape={test_full_src.feature_shape} "
        f"feature_gb_cpu={test_full_src.feature_gb:.3f} label_index_gb_cpu={test_full_src.label_index_gb:.3f} "
        f"pin_memory={test_full_src.pin_memory}"
    )

    test=summarize_metrics(model, test_full_src, device, stats, amp_enabled, amp_dtype, primary_only=False, epoch=0)
    print(f"[test_dir] dir_auc_q50plus={test['dir_auc_q50plus']} dir_bal_acc_q50plus={test['dir_bal_acc_q50plus']} dir_pos_frac_pred_q50plus={test['dir_pos_frac_pred_q50plus']} dir_pos_frac_true_q50plus={test['dir_pos_frac_true_q50plus']}")
    print(f"[test_mag] mag_spearman_abs_q50plus={test['mag_spearman_abs_q50plus']} pred_abs_std_over_true_abs_std_kept={test['pred_abs_std_over_true_abs_std_kept']} pred_abs_bps_p50_kept={test['pred_abs_bps_p50_kept']} true_abs_bps_p50_kept={test['true_abs_bps_p50_kept']}")
    print(f"[test_edge] edge_spearman_q50plus={test['edge_spearman_q50plus']} edge_bal_sign_acc_q50plus={test['edge_bal_sign_acc_q50plus']} edge_mean_kept={test['edge_mean_kept']} edge_std_kept={test['edge_std_kept']}")
    print('[done] Training complete.')


# ---------------- Entry ----------------
if __name__ == "__main__":
    train_from_offline()
