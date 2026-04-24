
#!/usr/bin/env python3
"""
CMSSL17_offline.py

Run CMSSL17's model using flat decision rows produced by offline_ingest.py.
This mirrors the training/eval flow in CMSSL17.py but reads dataset splits
from OUT_ROOT/meta.json and week meta files, with dynamic sequence slicing at load time.
"""

import os, sys, math, json
from typing import List, Dict, Tuple, Iterable, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch._inductor.config
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
    DMODEL, MAMBA_LAYERS,
    PRIMARY_METRIC, PRIMARY_METRIC_HORIZON_MS,
    LOW_ABS_TRIM_FRACTION, HIGH_ABS_TRIM_FRACTION, TARGET_TRANSFORM, TARGET_TASK, CHECKPOINT_SCHEMA,
    SINGLE_WEEK_PATIENCE, get_primary_metric_mode, compute_primary_metric, is_metric_improved,
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
        f"{source} has label_dim={observed!r}, but CMSSL17_offline.py now requires "
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


# ---------------- Signed-raw preprocessing, cache, and metrics ----------------
def signed_sqrt_transform(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def build_abs_trim_mask(y_raw_bps: np.ndarray, abs_lo_raw_bps: np.ndarray, abs_hi_raw_bps: np.ndarray) -> np.ndarray:
    abs_y = np.abs(y_raw_bps)
    lo = abs_lo_raw_bps.reshape(1, -1)
    hi = abs_hi_raw_bps.reshape(1, -1)
    return (abs_y >= lo) & (abs_y <= hi)


def build_raw_loss_weights(y_raw_bps: np.ndarray, kept_q50_abs_raw_bps: np.ndarray, kept_q85_abs_raw_bps: np.ndarray) -> np.ndarray:
    abs_raw = np.abs(y_raw_bps)
    q50 = kept_q50_abs_raw_bps.reshape(1, -1)
    q85 = kept_q85_abs_raw_bps.reshape(1, -1)
    tau = np.maximum(0.10 * (q85 - q50), 0.05)
    w = 0.50 + 0.50 * (1.0 / (1.0 + np.exp(-(abs_raw - q50) / tau))) + 0.25 * (1.0 / (1.0 + np.exp(-(abs_raw - q85) / tau)))
    return np.clip(w, 0.50, 1.25).astype(np.float32, copy=False)


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
    keys = ('low_abs_trim_fraction','high_abs_trim_fraction','horizons_ms','train_week_keys','train_ts_start','train_ts_end','decision_time_basis','trade_history_enabled','event_stream_mode','target_transform','label_units','target_task','loss_weighting_schema','spearman_ranking_schema')
    return all(cached_meta.get(k)==current_meta.get(k) for k in keys)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float('nan')
    rx = _average_ranks(x.astype(np.float64, copy=False))
    ry = _average_ranks(y.astype(np.float64, copy=False))
    return _pearson(rx, ry)


def inverse_signed_sqrt_transform_to_bps(z: np.ndarray) -> np.ndarray:
    return np.sign(z) * (np.abs(z) ** 2)


def summarize_metrics(model, dl, device, stats, amp_enabled, amp_dtype, primary_only=False):
    model.eval()
    pred_parts=[]; y_parts=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred=model(x)
            pred_parts.append(pred.detach().float().cpu().numpy())
            y_parts.append(y.numpy())
    if not y_parts:
        out={"spearman_kept_q50plus":[float('nan')]*NUM_HORIZONS}
        return out
    pred=np.concatenate(pred_parts,0); y_raw=np.concatenate(y_parts,0)
    keep=build_abs_trim_mask(y_raw, stats['abs_lo_raw_bps'], stats['abs_hi_raw_bps'])
    y_t=signed_sqrt_transform(y_raw)
    pred_raw_bps = inverse_signed_sqrt_transform_to_bps(pred)
    out={
        'kept_fraction':[], 'raw_q50plus_fraction_true':[], 'huber_kept':[], 'mae_kept_transformed':[],
        'pearson_all':[], 'spearman_all':[], 'pearson_kept_q50plus':[], 'spearman_kept_q50plus':[],
        'sign_acc_kept_q50plus':[], 'sign_acc_pred_mag_ge_1p0bps':[],
        'true_raw_abs_p50_kept':[], 'true_raw_abs_p90_kept':[],
        'pred_raw_abs_p50_bps':[], 'pred_raw_abs_p90_bps':[],
        'true_near_zero_frac_0p5bps':[], 'true_near_zero_frac_1p0bps':[],
        'pred_near_zero_frac_0p5bps':[], 'pred_near_zero_frac_1p0bps':[],
        'true_pos_kept_frac':[], 'true_neg_kept_frac':[],
        'pred_zero_frac_1p0bps':[], 'pred_pos_frac_1p0bps':[], 'pred_neg_frac_1p0bps':[],
        'balanced_sign_acc_kept_q50plus':[],
        'true_mean_bps_all':[], 'pred_mean_bps_all':[], 'true_std_bps_all':[], 'pred_std_bps_all':[],
        'true_mean_bps_kept':[], 'pred_mean_bps_kept':[], 'true_std_bps_kept':[], 'pred_std_bps_kept':[],
        'bin_frac':[], 'bin_sign_acc':[], 'bin_pred_abs_p90_bps':[], 'bin_spearman':[],
    }
    for h in range(NUM_HORIZONS):
        kh=keep[:,h]; ph=pred[:,h]; yh=y_t[:,h]; raw=y_raw[:,h]; ph_raw=pred_raw_bps[:,h]
        q50=float(stats['kept_q50_abs_raw_bps'][h]); q85=float(stats['kept_q85_abs_raw_bps'][h])
        q50plus = kh & (np.abs(raw) >= q50)

        out['kept_fraction'].append(float(kh.mean()))
        out['raw_q50plus_fraction_true'].append(float(q50plus.mean()))
        out['true_mean_bps_all'].append(float(np.mean(raw)))
        out['pred_mean_bps_all'].append(float(np.mean(ph_raw)))
        out['true_std_bps_all'].append(float(np.std(raw, ddof=0)))
        out['pred_std_bps_all'].append(float(np.std(ph_raw, ddof=0)))
        if kh.any():
            d=np.abs(ph[kh]-yh[kh]); hub=np.where(d<=1.0,0.5*d*d,d-0.5)
            out['huber_kept'].append(float(hub.mean())); out['mae_kept_transformed'].append(float(d.mean()))
            abs_raw_k = np.abs(raw[kh]); abs_pred_k = np.abs(ph_raw[kh])
            out['true_raw_abs_p50_kept'].append(float(np.quantile(abs_raw_k,0.50)))
            out['true_raw_abs_p90_kept'].append(float(np.quantile(abs_raw_k,0.90)))
            out['pred_raw_abs_p50_bps'].append(float(np.quantile(abs_pred_k,0.50)))
            out['pred_raw_abs_p90_bps'].append(float(np.quantile(abs_pred_k,0.90)))
            out['true_near_zero_frac_0p5bps'].append(float((abs_raw_k < 0.5).mean()))
            out['true_near_zero_frac_1p0bps'].append(float((abs_raw_k < 1.0).mean()))
            out['pred_near_zero_frac_0p5bps'].append(float((abs_pred_k < 0.5).mean()))
            out['pred_near_zero_frac_1p0bps'].append(float((abs_pred_k < 1.0).mean()))
            raw_k = raw[kh]; pred_k = ph_raw[kh]
            out['true_pos_kept_frac'].append(float((raw_k > 0).mean()))
            out['true_neg_kept_frac'].append(float((raw_k < 0).mean()))
            out['pred_zero_frac_1p0bps'].append(float((np.abs(pred_k) < 1.0).mean()))
            out['pred_pos_frac_1p0bps'].append(float((pred_k >= 1.0).mean()))
            out['pred_neg_frac_1p0bps'].append(float((pred_k <= -1.0).mean()))
            out['true_mean_bps_kept'].append(float(np.mean(raw_k)))
            out['pred_mean_bps_kept'].append(float(np.mean(pred_k)))
            out['true_std_bps_kept'].append(float(np.std(raw_k, ddof=0)))
            out['pred_std_bps_kept'].append(float(np.std(pred_k, ddof=0)))
            kept_abs = np.abs(raw_k)
            bin_masks=[kept_abs < q50, (kept_abs >= q50) & (kept_abs <= q85), kept_abs > q85]
            n_k=float(raw_k.size)
            bin_frac=[]; bin_sign_acc=[]; bin_pred_abs_p90_bps=[]; bin_spearman=[]
            for bm in bin_masks:
                if bm.any():
                    raw_bin = raw_k[bm]; pred_bin = pred_k[bm]
                    bin_frac.append(float(bm.sum()/n_k))
                    bin_sign_acc.append(float((np.sign(pred_bin)==np.sign(raw_bin)).mean()))
                    bin_pred_abs_p90_bps.append(float(np.quantile(np.abs(pred_bin),0.90)))
                    bin_spearman.append(_spearman(pred_bin, raw_bin) if raw_bin.size>=2 else float('nan'))
                else:
                    bin_frac.append(0.0); bin_sign_acc.append(float('nan')); bin_pred_abs_p90_bps.append(float('nan')); bin_spearman.append(float('nan'))
            out['bin_frac'].append(bin_frac)
            out['bin_sign_acc'].append(bin_sign_acc)
            out['bin_pred_abs_p90_bps'].append(bin_pred_abs_p90_bps)
            out['bin_spearman'].append(bin_spearman)
        else:
            out['huber_kept'].append(float('nan')); out['mae_kept_transformed'].append(float('nan'))
            out['true_raw_abs_p50_kept'].append(float('nan')); out['true_raw_abs_p90_kept'].append(float('nan'))
            out['pred_raw_abs_p50_bps'].append(float('nan')); out['pred_raw_abs_p90_bps'].append(float('nan'))
            out['true_near_zero_frac_0p5bps'].append(float('nan')); out['true_near_zero_frac_1p0bps'].append(float('nan'))
            out['pred_near_zero_frac_0p5bps'].append(float('nan')); out['pred_near_zero_frac_1p0bps'].append(float('nan'))
            out['true_pos_kept_frac'].append(float('nan')); out['true_neg_kept_frac'].append(float('nan'))
            out['pred_zero_frac_1p0bps'].append(float('nan')); out['pred_pos_frac_1p0bps'].append(float('nan')); out['pred_neg_frac_1p0bps'].append(float('nan'))
            out['true_mean_bps_kept'].append(float('nan')); out['pred_mean_bps_kept'].append(float('nan'))
            out['true_std_bps_kept'].append(float('nan')); out['pred_std_bps_kept'].append(float('nan'))
            out['bin_frac'].append([float('nan')]*3); out['bin_sign_acc'].append([float('nan')]*3); out['bin_pred_abs_p90_bps'].append([float('nan')]*3); out['bin_spearman'].append([float('nan')]*3)

        out['pearson_all'].append(_pearson(ph,yh)); out['spearman_all'].append(_spearman(ph,yh))
        out['pearson_kept_q50plus'].append(_pearson(ph[q50plus], yh[q50plus]) if q50plus.sum()>1 else float('nan'))
        out['spearman_kept_q50plus'].append(_spearman(ph[q50plus], yh[q50plus]) if q50plus.sum()>1 else float('nan'))
        out['sign_acc_kept_q50plus'].append(float((np.sign(ph_raw[q50plus])==np.sign(raw[q50plus])).mean()) if q50plus.any() else float('nan'))
        pred_mag_1p0 = np.abs(ph_raw) >= 1.0
        out['sign_acc_pred_mag_ge_1p0bps'].append(float((np.sign(ph_raw[pred_mag_1p0])==np.sign(raw[pred_mag_1p0])).mean()) if pred_mag_1p0.any() else float('nan'))
        pos_true = q50plus & (raw > 0)
        neg_true = q50plus & (raw < 0)
        if pos_true.any() and neg_true.any():
            acc_pos = float((np.sign(ph_raw[pos_true]) == np.sign(raw[pos_true])).mean())
            acc_neg = float((np.sign(ph_raw[neg_true]) == np.sign(raw[neg_true])).mean())
            out['balanced_sign_acc_kept_q50plus'].append(0.5 * (acc_pos + acc_neg))
        else:
            out['balanced_sign_acc_kept_q50plus'].append(float('nan'))

    if primary_only:
        return {'spearman_kept_q50plus': out['spearman_kept_q50plus']}
    return out


def get_model_state_dict_for_ckpt(model: torch.nn.Module) -> dict:
    return model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()


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
    out_root = Path(OUT_ROOT)
    meta = json.loads((out_root / "meta.json").read_text())
    validate_dataset_label_dim(meta, f"global metadata {out_root / 'meta.json'}")
    trade_history_enabled = meta.get('trade_history_enabled')
    event_stream_mode = meta.get('event_stream_mode')
    splits = require_four_week_pipeline_splits(meta, out_root)

    weeks_order = splits['weeks_in_order']
    cmssl_train = splits['splits']['cmssl']['train']
    cmssl_val = splits['splits']['cmssl']['val']
    cmssl_test = splits['splits']['cmssl']['test']

    ds_train = build_dataset_from_split(str(out_root), cmssl_train)
    ds_val = build_dataset_from_split(str(out_root), cmssl_val)
    ds_test = build_dataset_from_split(str(out_root), cmssl_test)
    F_total = int(meta.get("feature_dim_total", 0))

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
        'loss_weighting_schema': 'smooth_two_sigmoid_q50_q85_v1',
        'spearman_ranking_schema': 'tie_aware_average_ranks_v1'
    }
    cached=load_stats_cache(cache_path); stats=None
    if cached and cache_matches(cached[1], cache_meta): stats=cached[0]
    if stats is None:
        dl_pre=DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=False,drop_last=False,num_workers=WORKERS_TRAIN,pin_memory=True)
        y_parts=[yb.numpy() for _,yb in dl_pre]
        y_train=np.concatenate(y_parts,0) if y_parts else np.empty((0,NUM_HORIZONS),np.float32)
        stats=compute_signed_raw_stats(y_train)
        save_stats_cache(cache_path,stats,cache_meta)

    dl_train=DataLoader(ds_train,BATCH_SIZE,shuffle=True,drop_last=True,num_workers=WORKERS_TRAIN,pin_memory=True,prefetch_factor=8 if WORKERS_TRAIN>0 else None,persistent_workers=(WORKERS_TRAIN>0))
    dl_val=DataLoader(ds_val,BATCH_SIZE,shuffle=False,num_workers=max(1,WORKERS_VAL),pin_memory=True,persistent_workers=(max(1,WORKERS_VAL)>0))
    dl_test=DataLoader(ds_test,BATCH_SIZE,shuffle=False,num_workers=max(1,WORKERS_VAL),pin_memory=True,persistent_workers=(max(1,WORKERS_VAL)>0))

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

    for epoch in range(EPOCHS):
        model.train(); running={'loss':0.0,'huber':0.0,'corr':0.0}; n_batches=0
        for x,y in tqdm(dl_train, desc=f"Ep{epoch+1}/{EPOCHS}"):
            x=x.to(device, non_blocking=True); y_raw=y.to(device, non_blocking=True)
            def compute_loss(pred, y_raw):
                keep=(torch.abs(y_raw)>=abs_lo_t)&(torch.abs(y_raw)<=abs_hi_t)
                y_t=torch.sign(y_raw)*torch.sqrt(torch.abs(y_raw))
                if not keep.any():
                    z=pred.sum()*0.0
                    return z,z,z
                abs_raw=torch.abs(y_raw)
                tau_t=torch.clamp(0.10 * (q85_t - q50_t), min=0.05)
                w=0.50 + 0.50 * torch.sigmoid((abs_raw - q50_t) / tau_t) + 0.25 * torch.sigmoid((abs_raw - q85_t) / tau_t)
                w=torch.clamp(w, 0.50, 1.25)
                d=F.huber_loss(pred, y_t, delta=1.0, reduction='none')
                wm=(w*keep.float()*hwt)
                hub=(d*wm).sum()/wm.sum().clamp_min(1e-9)
                corrs=[]
                for h in range(NUM_HORIZONS):
                    mask=keep[:,h] & (abs_raw[:,h] >= q50_t[0,h])
                    if mask.sum()>=2:
                        px=pred[:,h][mask]; ty=y_t[:,h][mask]
                        px=px-px.mean(); ty=ty-ty.mean()
                        den=torch.sqrt((px*px).sum()*(ty*ty).sum()).clamp_min(1e-9)
                        corr=(px*ty).sum()/den
                        corrs.append(1.0-corr)
                corr_pen=torch.stack(corrs).mean() if corrs else pred.sum()*0.0
                return hub+0.10*corr_pen, hub, corr_pen

            opt.base_optimizer.zero_grad(set_to_none=True)
            
            # First forward/backward: compute the raw gradient used to choose SAM's adversarial perturbation.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x)
                loss, hub, corr = compute_loss(pred, y_raw)
            
            loss.backward()
            
            # Do NOT clip here. SAM first_step must see the raw gradient direction.
            opt.first_step(zero_grad=True)
            
            # Second forward/backward at perturbed weights.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred2 = model(x)
                loss2, _, _ = compute_loss(pred2, y_raw)
            
            loss2.backward()
            
            # Clip only before the actual optimizer update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000)
            opt.second_step(zero_grad=True)
            
            running['loss'] += float(loss.detach().cpu())
            running['huber'] += float(hub.detach().cpu())
            running['corr'] += float(corr.detach().cpu())
            n_batches += 1
            
        print(f"[train] loss={running['loss']/max(1,n_batches):.6f} huber={running['huber']/max(1,n_batches):.6f} corr_penalty={running['corr']/max(1,n_batches):.6f}")

        val_fast=summarize_metrics(model, dl_val, device, stats, amp_enabled, amp_dtype, primary_only=True)
        primary_metric_value, primary_metric_label = compute_primary_metric(val_fast)
        print(f"[val-fast] primary_metric({primary_metric_label})={primary_metric_value:.6f}")
        if math.isfinite(primary_metric_value) and is_metric_improved(primary_metric_value,best,primary_metric_mode):
            best=float(primary_metric_value); no_imp=0
            full=summarize_metrics(model, dl_val, device, stats, amp_enabled, amp_dtype, primary_only=False)
            print(f"[val] kept_fraction={full['kept_fraction']} raw_q50plus_fraction_true={full['raw_q50plus_fraction_true']}")
            print(f"[val_reg] huber_kept={full['huber_kept']} pearson_all={full['pearson_all']} spearman_all={full['spearman_all']} pearson_kept_q50plus={full['pearson_kept_q50plus']} spearman_kept_q50plus={full['spearman_kept_q50plus']}")
            print(f"[val_zero] pred_abs_p50_bps={full['pred_raw_abs_p50_bps']} pred_abs_p90_bps={full['pred_raw_abs_p90_bps']} near_zero_0p5={full['pred_near_zero_frac_0p5bps']} near_zero_1p0={full['pred_near_zero_frac_1p0bps']}")
            print(f"[val_cls] true_pos={full['true_pos_kept_frac']} true_neg={full['true_neg_kept_frac']} pred_zero={full['pred_zero_frac_1p0bps']} pred_pos={full['pred_pos_frac_1p0bps']} pred_neg={full['pred_neg_frac_1p0bps']} bal_sign_acc={full['balanced_sign_acc_kept_q50plus']} pred_mag_sign_acc={full['sign_acc_pred_mag_ge_1p0bps']}")
            print(f"[val_bins] frac={full['bin_frac']} sign_acc={full['bin_sign_acc']} pred_abs_p90_bps={full['bin_pred_abs_p90_bps']} spearman={full['bin_spearman']}")
            print(f"[val_mean] true_mean_bps_all={full['true_mean_bps_all']} pred_mean_bps_all={full['pred_mean_bps_all']} true_mean_bps_kept={full['true_mean_bps_kept']} pred_mean_bps_kept={full['pred_mean_bps_kept']}")
            print(f"[val_std] true_std_bps_all={full['true_std_bps_all']} pred_std_bps_all={full['pred_std_bps_all']} true_std_bps_kept={full['true_std_bps_kept']} pred_std_bps_kept={full['pred_std_bps_kept']}")
            ckpt={
                'epoch': epoch,
                'state_dict': get_model_state_dict_for_ckpt(model),
                'args': {
                    'DMODEL':DMODEL, 'MAMBA_LAYERS':MAMBA_LAYERS, 'feat_dim':F_total, 'LOOKBACK':LOOKBACK,
                    'WINDOW_MS': WINDOW_MS, 'HORIZONS_MS': HORIZONS_MS, 'checkpoint_schema': CHECKPOINT_SCHEMA,
                    'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
                    'decision_time_basis': meta.get('decision_time_basis'), 'decision_stride_policy':'every_ob_event',
                    'label_delta_ms':0, 'label_units':'signed_log_return_bps',
                    'target_task': TARGET_TASK,
                    'target_transform': TARGET_TRANSFORM,
                    'low_abs_trim_fraction': float(LOW_ABS_TRIM_FRACTION),
                    'high_abs_trim_fraction': float(HIGH_ABS_TRIM_FRACTION),
                },
                'best_primary_metric': best,
            }
            out_ckpt=out_root/'cmssl17_offline_best.pt'; torch.save(ckpt,out_ckpt); print(f"[ckpt] saved best to {out_ckpt}")
        else:
            no_imp += 1
            if no_imp >= early_stop_patience:
                print('Early stopping triggered.')
                break

    test=summarize_metrics(model, dl_test, device, stats, amp_enabled, amp_dtype, primary_only=False)
    print(f"[test] kept_fraction={test['kept_fraction']} raw_q50plus_fraction_true={test['raw_q50plus_fraction_true']}")
    print(f"[test_reg] huber_kept={test['huber_kept']} pearson_all={test['pearson_all']} spearman_all={test['spearman_all']} pearson_kept_q50plus={test['pearson_kept_q50plus']} spearman_kept_q50plus={test['spearman_kept_q50plus']}")
    print(f"[test_zero] pred_abs_p50_bps={test['pred_raw_abs_p50_bps']} pred_abs_p90_bps={test['pred_raw_abs_p90_bps']} near_zero_0p5={test['pred_near_zero_frac_0p5bps']} near_zero_1p0={test['pred_near_zero_frac_1p0bps']}")
    print(f"[test_cls] true_pos={test['true_pos_kept_frac']} true_neg={test['true_neg_kept_frac']} pred_zero={test['pred_zero_frac_1p0bps']} pred_pos={test['pred_pos_frac_1p0bps']} pred_neg={test['pred_neg_frac_1p0bps']} bal_sign_acc={test['balanced_sign_acc_kept_q50plus']} pred_mag_sign_acc={test['sign_acc_pred_mag_ge_1p0bps']}")
    print(f"[test_bins] frac={test['bin_frac']} sign_acc={test['bin_sign_acc']} pred_abs_p90_bps={test['bin_pred_abs_p90_bps']} spearman={test['bin_spearman']}")
    print(f"[test_mean] true_mean_bps_all={test['true_mean_bps_all']} pred_mean_bps_all={test['pred_mean_bps_all']} true_mean_bps_kept={test['true_mean_bps_kept']} pred_mean_bps_kept={test['pred_mean_bps_kept']}")
    print(f"[test_std] true_std_bps_all={test['true_std_bps_all']} pred_std_bps_all={test['pred_std_bps_all']} true_std_bps_kept={test['true_std_bps_kept']} pred_std_bps_kept={test['pred_std_bps_kept']}")
    print('[done] Training complete.')


# ---------------- Entry ----------------
if __name__ == "__main__":
    train_from_offline()
