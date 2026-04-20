
#!/usr/bin/env python3
"""
CMSSL17_offline.py

Run CMSSL17's model *using prebuilt tokens* produced by offline_ingest.py.
This mirrors the training/eval flow in CMSSL17.py but reads dataset splits
from OUT_ROOT/meta.json and week meta files, avoiding any online feature building.
"""

import os, sys, math, json
from typing import List, Dict, Tuple, Iterable, Optional, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

from CMSSL17 import (  # type: ignore
    SAMBA, ModelArgs,
    LOOKBACK, WINDOW_MS, AUX_DIM, HORIZONS_MS, NUM_HORIZONS, HORIZON_WEIGHTS,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
    DMODEL, MAMBA_LAYERS,
    PRIMARY_METRIC, PRIMARY_METRIC_HORIZON_MS,
    FEE_HURDLE_BPS, ABS_TRIM_TAIL_FRACTION, TARGET_TRANSFORM, CHECKPOINT_SCHEMA,
    SINGLE_WEEK_PATIENCE, get_primary_metric_mode, compute_primary_metric, is_metric_improved,
    SAM,
)

# ---------------- Config via env ----------------
OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
USE_IN_MEMORY = int(os.environ.get("BYBIT_USE_IN_MEMORY", "0")) == 1
WORKERS_TRAIN = int(os.environ.get("BYBIT_WORKERS", "8"))
WORKERS_VAL   = max(1, min(4, WORKERS_TRAIN // 2))
AMP_ENABLED   = int(os.environ.get("BYBIT_AMP", "1")) == 1
COMPILE_ENABLED = int(os.environ.get("BYBIT_TORCH_COMPILE", "1")) == 1
COMPILE_MODE = os.environ.get("BYBIT_TORCH_COMPILE_MODE", "default").strip()
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
        week_meta = read_json(out_root / rel_path)
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

# ---------------- Dataset (streaming from .npy chunks) ----------------
class NpyChunksDataset(Dataset):
    def __init__(self, chunk_refs: List[ChunkRef], feature_dim_total: int):
        """
        chunk_refs: list of chunks in chronological order (kept as given)
        feature_dim_total: F (including AUX_DIM)
        """
        self.refs = list(chunk_refs)
        self.F = int(feature_dim_total)
        self.F_core = self.F - AUX_DIM
        if self.F_core <= 0:
            raise ValueError(f"feature_dim_total ({self.F}) must exceed AUX_DIM ({AUX_DIM})")

        # prefix sums for O(log N) lookup
        self.starts = []
        total = 0
        for r in self.refs:
            self.starts.append(total)
            total += r.n
        self.total = total

        # small cache of currently loaded memory-mapped arrays per process
        self._cache: Dict[Tuple[str, int], Tuple[np.memmap, np.memmap, np.memmap]] = {}
        self._lru: List[Tuple[str, int]] = []
        self._cap = 8  # keep up to 8 chunks mapped

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        # binary search on starts
        lo, hi = 0, len(self.starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.starts[mid]
            next_start = self.starts[mid + 1] if mid + 1 < len(self.starts) else self.total
            if start <= idx < next_start:
                return mid, idx - start
            elif idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def _load_chunk(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = self.refs[i]
        key = (str(ref.week_dir), i)
        if key in self._cache:
            # move to end of LRU
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            return self._cache[key]
        # mmap lazy
        Xc = np.load(ref.core_file, mmap_mode='r')
        Xa = np.load(ref.aux_file,  mmap_mode='r')
        Y  = np.load(ref.y_file,    mmap_mode='r')
        validate_loaded_label_array(Y, f"label file {ref.y_file}")
        self._cache[key] = (Xc, Xa, Y)
        self._lru.append(key)
        if len(self._lru) > self._cap:
            evict_key = self._lru.pop(0)
            try:
                del self._cache[evict_key]
            except KeyError:
                pass
        return Xc, Xa, Y

    def __getitem__(self, idx: int):
        ci, offset_in_dataset = self._locate(idx)
        ref = self.refs[ci]
        Xc, Xa, Y = self._load_chunk(ci)

        idx_in_file = ref.offset + offset_in_dataset

        core = np.asarray(Xc[idx_in_file], dtype=np.float32)
        aux  = np.asarray(Xa[idx_in_file], dtype=np.float32)
        x = np.concatenate([core, aux], axis=-1)
        if not x.flags.writeable:
            x = x.copy()
        y = np.asarray(Y[idx_in_file], dtype=np.float32)
        if not y.flags.writeable:
            y = y.copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def load_split_in_memory_ts(split_week_paths: List[Path], start: int, end: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load rows in start <= ts < end across weeks into RAM. Returns X [N, L, F], y [N, H], F."""
    if end < start:
        raise ValueError(f"Invalid ts range: start={start} must be <= end={end}")

    Xs, Ys = [], []
    feat_dim = None
    for wp in split_week_paths:
        wmeta = read_json(wp)
        validate_dataset_label_dim(wmeta, f"week metadata {wp}")
        F_total = int(wmeta["feature_dim_total"])
        if feat_dim is None:
            feat_dim = F_total
        elif feat_dim != F_total:
            raise ValueError(f"Feature dim mismatch between weeks: {feat_dim} vs {F_total}")

        week_dir = wp.parent
        for idx, ch in enumerate(wmeta.get("chunks", [])):
            files = ch.get("files", {})
            ts_rel = files.get("ts")
            if not ts_rel:
                raise KeyError(
                    f"Chunk {idx} in {wp} is missing files['ts']; cannot slice by timestamp"
                )

            ts_arr = np.load(week_dir / ts_rel, mmap_mode="r")
            if ts_arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D ts array in chunk {idx} ({week_dir / ts_rel}), got shape={ts_arr.shape}"
                )

            # Safety check: searchsorted semantics require non-decreasing input.
            if ts_arr.size > 1 and not np.all(ts_arr[1:] >= ts_arr[:-1]):
                raise ValueError(
                    f"Timestamp file is not non-decreasing for chunk {idx} in {wp}: "
                    f"{week_dir / ts_rel}; ts must be non-decreasing for range slicing"
                )

            l = int(np.searchsorted(ts_arr, start, side="left"))
            r = int(np.searchsorted(ts_arr, end, side="left"))
            if r <= l:
                continue

            Xc = np.load(week_dir / files["core"])
            Xa = np.load(week_dir / files["aux"])
            Y = np.load(week_dir / files["y"])
            validate_loaded_label_array(Y, f"label file {week_dir / files['y']}")
            Xs.append(np.concatenate([Xc[l:r], Xa[l:r]], axis=-1))
            Ys.append(Y[l:r])

    if not Xs:
        return (
            np.empty((0, LOOKBACK, feat_dim or 0), np.float32),
            np.empty((0, NUM_HORIZONS), np.float32),
            (feat_dim or 0),
        )
    X = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(Ys, axis=0).astype(np.float32, copy=False)
    return X, y, int(feat_dim)

# ---------------- Signed-excess preprocessing, cache, and metrics ----------------
def raw_returns_to_excess_bps(y_raw_bps: np.ndarray, fee_hurdle_bps: float) -> np.ndarray:
    mag = np.maximum(np.abs(y_raw_bps) - float(fee_hurdle_bps), 0.0)
    return np.sign(y_raw_bps) * mag


def signed_sqrt_transform(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def build_abs_trim_mask(y_raw_bps: np.ndarray, abs_lo_raw_bps: np.ndarray, abs_hi_raw_bps: np.ndarray) -> np.ndarray:
    abs_y = np.abs(y_raw_bps)
    lo = abs_lo_raw_bps.reshape(1, -1)
    hi = abs_hi_raw_bps.reshape(1, -1)
    return (abs_y >= lo) & (abs_y <= hi)


def build_loss_weights(y_excess_bps: np.ndarray, active_q50_excess: np.ndarray, active_q85_excess: np.ndarray) -> np.ndarray:
    w = np.full_like(y_excess_bps, 0.25, dtype=np.float32)
    abs_ex = np.abs(y_excess_bps)
    active = abs_ex > 0
    q50 = active_q50_excess.reshape(1, -1)
    q85 = active_q85_excess.reshape(1, -1)
    w[active & (abs_ex <= q50)] = 1.00
    w[active & (abs_ex > q50) & (abs_ex <= q85)] = 1.25
    w[active & (abs_ex > q85)] = 1.00
    return w


def compute_signed_excess_stats(y_train: np.ndarray) -> Dict[str, np.ndarray]:
    abs_y = np.abs(y_train)
    abs_lo = np.quantile(abs_y, ABS_TRIM_TAIL_FRACTION, axis=0).astype(np.float32)
    abs_hi = np.quantile(abs_y, 1.0 - ABS_TRIM_TAIL_FRACTION, axis=0).astype(np.float32)
    keep = build_abs_trim_mask(y_train, abs_lo, abs_hi)
    y_ex = raw_returns_to_excess_bps(y_train, FEE_HURDLE_BPS)
    q50 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    q85 = np.zeros(NUM_HORIZONS, dtype=np.float32)
    for h in range(NUM_HORIZONS):
        a = np.abs(y_ex[:, h])
        act = a[(a > 0) & keep[:, h]]
        if act.size:
            q50[h] = float(np.quantile(act, 0.50))
            q85[h] = float(np.quantile(act, 0.85))
    return {
        'abs_lo_raw_bps': abs_lo,
        'abs_hi_raw_bps': abs_hi,
        'active_q50_excess': q50,
        'active_q85_excess': q85,
    }


def load_stats_cache(path: Path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as c:
        stats = {k: np.asarray(c[k], dtype=np.float32) for k in ('abs_lo_raw_bps','abs_hi_raw_bps','active_q50_excess','active_q85_excess')}
        meta = json.loads(str(c['metadata_json'].item()))
    return stats, meta


def save_stats_cache(path: Path, stats: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    np.savez_compressed(path, **stats, metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_))


def cache_matches(cached_meta: Dict[str, Any], current_meta: Dict[str, Any]) -> bool:
    keys = ('abs_trim_tail_fraction','fee_hurdle_bps','horizons_ms','train_week_keys','train_ts_start','train_ts_end','decision_time_basis','trade_history_enabled','event_stream_mode','target_transform','label_units')
    return all(cached_meta.get(k)==current_meta.get(k) for k in keys)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    x0 = x - x.mean(); y0 = y - y.mean()
    den = np.sqrt((x0*x0).sum() * (y0*y0).sum())
    if den <= 0:
        return float('nan')
    return float((x0*y0).sum()/den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(np.float64), ry.astype(np.float64))


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
        out={"spearman_active":[float('nan')]*NUM_HORIZONS}
        return out
    pred=np.concatenate(pred_parts,0); y_raw=np.concatenate(y_parts,0)
    keep=build_abs_trim_mask(y_raw, stats['abs_lo_raw_bps'], stats['abs_hi_raw_bps'])
    y_ex=raw_returns_to_excess_bps(y_raw, FEE_HURDLE_BPS)
    y_t=signed_sqrt_transform(y_ex)
    out={
        'kept_fraction':[], 'active_fraction_true':[], 'huber_kept':[], 'mae_kept_transformed':[],
        'pearson_all':[], 'pearson_active':[], 'spearman_all':[], 'spearman_active':[],
        'sign_acc_active_true':[], 'sign_acc_pred_active_1p0bps':[],
        'true_excess_abs_p50_top10':[], 'true_excess_abs_p90_top10':[],
        'pred_excess_abs_p50_bps':[], 'pred_excess_abs_p90_bps':[], 'pred_near_zero_frac_0p5bps':[], 'pred_near_zero_frac_1p0bps':[],
        'true_zero_frac':[], 'true_pos_kept_frac':[], 'true_neg_kept_frac':[],
        'pred_zero_frac_1p0bps':[], 'pred_pos_frac_1p0bps':[], 'pred_neg_frac_1p0bps':[],
        'balanced_sign_acc_active_true':[],
        'bin_frac':[], 'bin_sign_acc':[], 'bin_pred_abs_p90_bps':[], 'bin_spearman':[],
    }
    for h in range(NUM_HORIZONS):
        kh=keep[:,h]; ph=pred[:,h]; yh=y_t[:,h]; ex=y_ex[:,h]
        ph_ex_bps = inverse_signed_sqrt_transform_to_bps(ph)
        out['kept_fraction'].append(float(kh.mean()))
        active=np.abs(ex)>0
        out['active_fraction_true'].append(float(active.mean()))
        if kh.any():
            d=np.abs(ph[kh]-yh[kh]); hub=np.where(d<=1.0,0.5*d*d,d-0.5)
            out['huber_kept'].append(float(hub.mean())); out['mae_kept_transformed'].append(float(d.mean()))
        else:
            out['huber_kept'].append(float('nan')); out['mae_kept_transformed'].append(float('nan'))
        out['pearson_all'].append(_pearson(ph,yh)); out['spearman_all'].append(_spearman(ph,yh))
        out['pearson_active'].append(_pearson(ph[active], yh[active]) if active.sum()>1 else float('nan'))
        out['spearman_active'].append(_spearman(ph[active], yh[active]) if active.sum()>1 else float('nan'))
        out['sign_acc_active_true'].append(float((np.sign(ph[active])==np.sign(ex[active])).mean()) if active.any() else float('nan'))
        pred_active_1p0 = np.abs(ph_ex_bps) >= 1.0
        out['sign_acc_pred_active_1p0bps'].append(float((np.sign(ph_ex_bps[pred_active_1p0])==np.sign(ex[pred_active_1p0])).mean()) if pred_active_1p0.any() else float('nan'))

        if kh.any():
            ex_k = ex[kh]
            ph_ex_bps_k = ph_ex_bps[kh]
            abs_ph_ex_k = np.abs(ph_ex_bps_k)
            out['pred_excess_abs_p50_bps'].append(float(np.quantile(abs_ph_ex_k,0.50)))
            out['pred_excess_abs_p90_bps'].append(float(np.quantile(abs_ph_ex_k,0.90)))
            out['pred_near_zero_frac_0p5bps'].append(float((abs_ph_ex_k < 0.5).mean()))
            out['pred_near_zero_frac_1p0bps'].append(float((abs_ph_ex_k < 1.0).mean()))

            out['true_zero_frac'].append(float((np.abs(ex_k) == 0).mean()))
            out['true_pos_kept_frac'].append(float((ex_k > 0).mean()))
            out['true_neg_kept_frac'].append(float((ex_k < 0).mean()))
            out['pred_zero_frac_1p0bps'].append(float((np.abs(ph_ex_bps_k) < 1.0).mean()))
            out['pred_pos_frac_1p0bps'].append(float((ph_ex_bps_k >= 1.0).mean()))
            out['pred_neg_frac_1p0bps'].append(float((ph_ex_bps_k <= -1.0).mean()))

            pos_true = ex_k > 0
            neg_true = ex_k < 0
            if pos_true.any() and neg_true.any():
                acc_pos = float((np.sign(ph_ex_bps_k[pos_true]) == np.sign(ex_k[pos_true])).mean())
                acc_neg = float((np.sign(ph_ex_bps_k[neg_true]) == np.sign(ex_k[neg_true])).mean())
                out['balanced_sign_acc_active_true'].append(0.5 * (acc_pos + acc_neg))
            else:
                out['balanced_sign_acc_active_true'].append(float('nan'))

            q50=float(stats['active_q50_excess'][h]); q85=float(stats['active_q85_excess'][h])
            abs_ex_k = np.abs(ex_k)
            bin_masks = [
                (abs_ex_k == 0),
                ((abs_ex_k > 0) & (abs_ex_k <= q50)),
                ((abs_ex_k > q50) & (abs_ex_k <= q85)),
                (abs_ex_k > q85),
            ]
            n_k = float(ex_k.size)
            bin_frac = []
            bin_sign_acc = []
            bin_pred_abs_p90_bps = []
            bin_spearman = []
            for j, bm in enumerate(bin_masks):
                if bm.any():
                    ex_bin = ex_k[bm]
                    ph_bin = ph_ex_bps_k[bm]
                    bin_frac.append(float(bm.sum() / n_k))
                    bin_pred_abs_p90_bps.append(float(np.quantile(np.abs(ph_bin), 0.90)))
                    if j == 0:
                        bin_sign_acc.append(float('nan'))
                        bin_spearman.append(float('nan'))
                    else:
                        bin_sign_acc.append(float((np.sign(ph_bin) == np.sign(ex_bin)).mean()))
                        bin_spearman.append(_spearman(ph_bin, ex_bin) if ex_bin.size >= 2 else float('nan'))
                else:
                    bin_frac.append(0.0)
                    bin_sign_acc.append(float('nan'))
                    bin_pred_abs_p90_bps.append(float('nan'))
                    bin_spearman.append(float('nan'))
            out['bin_frac'].append(bin_frac)
            out['bin_sign_acc'].append(bin_sign_acc)
            out['bin_pred_abs_p90_bps'].append(bin_pred_abs_p90_bps)
            out['bin_spearman'].append(bin_spearman)
        else:
            out['pred_excess_abs_p50_bps'].append(float('nan'))
            out['pred_excess_abs_p90_bps'].append(float('nan'))
            out['pred_near_zero_frac_0p5bps'].append(float('nan'))
            out['pred_near_zero_frac_1p0bps'].append(float('nan'))
            out['true_zero_frac'].append(float('nan'))
            out['true_pos_kept_frac'].append(float('nan'))
            out['true_neg_kept_frac'].append(float('nan'))
            out['pred_zero_frac_1p0bps'].append(float('nan'))
            out['pred_pos_frac_1p0bps'].append(float('nan'))
            out['pred_neg_frac_1p0bps'].append(float('nan'))
            out['balanced_sign_acc_active_true'].append(float('nan'))
            out['bin_frac'].append([float('nan')]*4)
            out['bin_sign_acc'].append([float('nan')]*4)
            out['bin_pred_abs_p90_bps'].append([float('nan')]*4)
            out['bin_spearman'].append([float('nan')]*4)

        score=np.abs(ph); order=np.argsort(-score)
        n=len(order)
        k10=max(1,int(np.ceil(n*0.10))); idx10=order[:k10]; ta=np.abs(ex[idx10])
        out['true_excess_abs_p50_top10'].append(float(np.quantile(ta,0.50))); out['true_excess_abs_p90_top10'].append(float(np.quantile(ta,0.90)))
    if primary_only:
        return {'spearman_active': out['spearman_active']}
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
    meta = load_global_meta(out_root)
    validate_dataset_label_dim(meta, f"global metadata {out_root / 'meta.json'}")
    trade_history_enabled = meta.get('trade_history_enabled')
    event_stream_mode = meta.get('event_stream_mode')
    splits = require_four_week_pipeline_splits(meta, out_root)

    weeks_order = splits['weeks_in_order']
    key_to_meta = {wk: out_root / meta['weeks_meta'][wk] for wk in weeks_order if wk in meta.get('weeks_meta',{})}
    cmssl_train = splits['splits']['cmssl']['train']; cmssl_val = splits['splits']['cmssl']['val']; cmssl_test = splits['splits']['cmssl']['test']
    tr_weeks=[key_to_meta[k] for k in cmssl_train['weeks']]; va_weeks=[key_to_meta[k] for k in cmssl_val['weeks']]; te_weeks=[key_to_meta[k] for k in cmssl_test['weeks']]

    feat_dim_total=None
    for wp in tr_weeks+va_weeks+te_weeks:
        wm=read_json(wp); validate_dataset_label_dim(wm, f"week metadata {wp}")
        fm=int(wm['feature_dim_total']); feat_dim_total=fm if feat_dim_total is None else feat_dim_total
    F_total=int(feat_dim_total or 0)

    tr_start,tr_end=int(cmssl_train['start']),int(cmssl_train['end'])
    va_start,va_end=int(cmssl_val['start']),int(cmssl_val['end'])
    te_start,te_end=int(cmssl_test['start']),int(cmssl_test['end'])

    def refs(weeks,start,end):
        out=[]
        for wp in weeks: out.extend(build_chunk_refs_by_ts(wp,start,end))
        return out
    tr_refs,va_refs,te_refs = refs(tr_weeks,tr_start,tr_end),refs(va_weeks,va_start,va_end),refs(te_weeks,te_start,te_end)
    ds_train,ds_val,ds_test = NpyChunksDataset(tr_refs,F_total),NpyChunksDataset(va_refs,F_total),NpyChunksDataset(te_refs,F_total)

    cache_path=out_root/'signed_excess_stats_cache.npz'
    cache_meta={
        'abs_trim_tail_fraction': float(ABS_TRIM_TAIL_FRACTION), 'fee_hurdle_bps': float(FEE_HURDLE_BPS),
        'horizons_ms':[int(h) for h in HORIZONS_MS], 'train_week_keys': list(cmssl_train['weeks']),
        'train_ts_start': int(tr_start), 'train_ts_end': int(tr_end), 'decision_time_basis': EXPECTED_DECISION_TIME_BASIS,
        'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
        'target_transform': TARGET_TRANSFORM, 'label_units': 'signed_log_return_bps'
    }
    cached=load_stats_cache(cache_path); stats=None
    if cached and cache_matches(cached[1], cache_meta): stats=cached[0]
    if stats is None:
        dl_pre=DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=False,drop_last=False,num_workers=WORKERS_TRAIN,pin_memory=True)
        y_parts=[yb.numpy() for _,yb in dl_pre]
        y_train=np.concatenate(y_parts,0) if y_parts else np.empty((0,NUM_HORIZONS),np.float32)
        stats=compute_signed_excess_stats(y_train)
        save_stats_cache(cache_path,stats,cache_meta)

    dl_train=DataLoader(ds_train,BATCH_SIZE,shuffle=True,drop_last=True,num_workers=WORKERS_TRAIN,pin_memory=True,prefetch_factor=8 if WORKERS_TRAIN>0 else None,persistent_workers=(WORKERS_TRAIN>0))
    dl_val=DataLoader(ds_val,BATCH_SIZE,shuffle=False,num_workers=max(1,WORKERS_VAL),pin_memory=True,persistent_workers=(max(1,WORKERS_VAL)>0))
    dl_test=DataLoader(ds_test,BATCH_SIZE,shuffle=False,num_workers=max(1,WORKERS_VAL),pin_memory=True,persistent_workers=(max(1,WORKERS_VAL)>0))

    args=ModelArgs(DMODEL,MAMBA_LAYERS,F_total,LOOKBACK)
    model=SAMBA(args).to(device)
    if COMPILE_ENABLED and hasattr(torch,'compile'):
        try: model=torch.compile(model, mode=COMPILE_MODE)
        except Exception: pass
    opt=SAM(model.parameters(), torch.optim.AdamW, lr=LR, weight_decay=1e-3, rho=0.01)
    primary_metric_mode=get_primary_metric_mode()
    best=-float('inf') if primary_metric_mode=='max' else float('inf')
    no_imp=0; early_stop_patience=SINGLE_WEEK_PATIENCE if len(tr_weeks)<=1 else PATIENCE

    abs_lo_t=torch.tensor(stats['abs_lo_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    abs_hi_t=torch.tensor(stats['abs_hi_raw_bps'],device=device,dtype=torch.float32).view(1,-1)
    q50_t=torch.tensor(stats['active_q50_excess'],device=device,dtype=torch.float32).view(1,-1)
    q85_t=torch.tensor(stats['active_q85_excess'],device=device,dtype=torch.float32).view(1,-1)
    hwt=torch.tensor(HORIZON_WEIGHTS,device=device,dtype=torch.float32).view(1,-1)

    for epoch in range(EPOCHS):
        model.train(); running={'loss':0.0,'huber':0.0,'corr':0.0}; n_batches=0
        for x,y in tqdm(dl_train, desc=f"Ep{epoch+1}/{EPOCHS}"):
            x=x.to(device, non_blocking=True); y_raw=y.to(device, non_blocking=True)
            def compute_loss(pred, y_raw):
                keep=(torch.abs(y_raw)>=abs_lo_t)&(torch.abs(y_raw)<=abs_hi_t)
                y_ex=torch.sign(y_raw)*torch.clamp(torch.abs(y_raw)-FEE_HURDLE_BPS, min=0.0)
                y_t=torch.sign(y_ex)*torch.sqrt(torch.abs(y_ex))
                if not keep.any():
                    z=pred.sum()*0.0
                    return z,z,z
                w=torch.full_like(y_ex,0.25)
                abs_ex=torch.abs(y_ex); active=abs_ex>0
                w = torch.where(active & (abs_ex<=q50_t), torch.ones_like(w), w)
                w = torch.where(active & (abs_ex>q50_t) & (abs_ex<=q85_t), torch.full_like(w,1.25), w)
                w = torch.where(active & (abs_ex>q85_t), torch.ones_like(w), w)
                d=F.huber_loss(pred, y_t, delta=1.0, reduction='none')
                wm=(w*keep.float()*hwt)
                hub=(d*wm).sum()/wm.sum().clamp_min(1e-9)
                corrs=[]
                for h in range(NUM_HORIZONS):
                    mask=keep[:,h] & (torch.abs(y_ex[:,h])>0)
                    if mask.sum()>=2:
                        px=pred[:,h][mask]; ty=y_t[:,h][mask]
                        px=px-px.mean(); ty=ty-ty.mean()
                        den=torch.sqrt((px*px).sum()*(ty*ty).sum()).clamp_min(1e-9)
                        corr=(px*ty).sum()/den
                        corrs.append(1.0-corr)
                corr_pen=torch.stack(corrs).mean() if corrs else pred.sum()*0.0
                return hub+0.10*corr_pen, hub, corr_pen

            opt.base_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred=model(x); loss,hub,corr=compute_loss(pred,y_raw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000); opt.first_step(zero_grad=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred2=model(x); loss2,_,_=compute_loss(pred2,y_raw)
            loss2.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000); opt.second_step(zero_grad=True)
            running['loss']+=float(loss.detach().cpu()); running['huber']+=float(hub.detach().cpu()); running['corr']+=float(corr.detach().cpu()); n_batches+=1
        print(f"[train] loss={running['loss']/max(1,n_batches):.6f} huber={running['huber']/max(1,n_batches):.6f} corr_penalty={running['corr']/max(1,n_batches):.6f}")

        val_fast=summarize_metrics(model, dl_val, device, stats, amp_enabled, amp_dtype, primary_only=True)
        primary_metric_value, primary_metric_label = compute_primary_metric(val_fast)
        print(f"[val-fast] primary_metric({primary_metric_label})={primary_metric_value:.6f}")
        if math.isfinite(primary_metric_value) and is_metric_improved(primary_metric_value,best,primary_metric_mode):
            best=float(primary_metric_value); no_imp=0
            full=summarize_metrics(model, dl_val, device, stats, amp_enabled, amp_dtype, primary_only=False)
            print(f"[val] kept_fraction={full['kept_fraction']} active_fraction={full['active_fraction_true']}")
            print(f"[val_reg] huber_kept={full['huber_kept']} pearson_all={full['pearson_all']} spearman_all={full['spearman_all']} pearson_active={full['pearson_active']} spearman_active={full['spearman_active']}")
            print(f"[val_zero] pred_abs_p50_bps={full['pred_excess_abs_p50_bps']} pred_abs_p90_bps={full['pred_excess_abs_p90_bps']} near_zero_0p5={full['pred_near_zero_frac_0p5bps']} near_zero_1p0={full['pred_near_zero_frac_1p0bps']}")
            print(f"[val_cls] true_zero={full['true_zero_frac']} true_pos={full['true_pos_kept_frac']} true_neg={full['true_neg_kept_frac']} pred_zero={full['pred_zero_frac_1p0bps']} pred_pos={full['pred_pos_frac_1p0bps']} pred_neg={full['pred_neg_frac_1p0bps']} bal_sign_acc={full['balanced_sign_acc_active_true']} pred_active_sign_acc={full['sign_acc_pred_active_1p0bps']}")
            print(f"[val_bins] frac={full['bin_frac']} sign_acc={full['bin_sign_acc']} pred_abs_p90_bps={full['bin_pred_abs_p90_bps']} spearman={full['bin_spearman']}")
            ckpt={
                'epoch': epoch,
                'state_dict': get_model_state_dict_for_ckpt(model),
                'args': {
                    'DMODEL':DMODEL, 'MAMBA_LAYERS':MAMBA_LAYERS, 'feat_dim':F_total, 'LOOKBACK':LOOKBACK,
                    'WINDOW_MS': WINDOW_MS, 'HORIZONS_MS': HORIZONS_MS, 'checkpoint_schema': CHECKPOINT_SCHEMA,
                    'trade_history_enabled': trade_history_enabled, 'event_stream_mode': event_stream_mode,
                    'decision_time_basis': meta.get('decision_time_basis'), 'decision_stride_policy':'every_ob_event',
                    'label_delta_ms':0, 'label_units':'signed_log_return_bps',
                    'target_task':'signed_excess_return_training_from_raw_bps_labels',
                    'fee_hurdle_bps': float(FEE_HURDLE_BPS), 'target_transform': TARGET_TRANSFORM,
                    'abs_trim_tail_fraction': float(ABS_TRIM_TAIL_FRACTION),
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
    print(f"[test] kept_fraction={test['kept_fraction']} active_fraction={test['active_fraction_true']}")
    print(f"[test_reg] huber_kept={test['huber_kept']} pearson_all={test['pearson_all']} spearman_all={test['spearman_all']} pearson_active={test['pearson_active']} spearman_active={test['spearman_active']}")
    print(f"[test_zero] pred_abs_p50_bps={test['pred_excess_abs_p50_bps']} pred_abs_p90_bps={test['pred_excess_abs_p90_bps']} near_zero_0p5={test['pred_near_zero_frac_0p5bps']} near_zero_1p0={test['pred_near_zero_frac_1p0bps']}")
    print(f"[test_cls] true_zero={test['true_zero_frac']} true_pos={test['true_pos_kept_frac']} true_neg={test['true_neg_kept_frac']} pred_zero={test['pred_zero_frac_1p0bps']} pred_pos={test['pred_pos_frac_1p0bps']} pred_neg={test['pred_neg_frac_1p0bps']} bal_sign_acc={test['balanced_sign_acc_active_true']} pred_active_sign_acc={test['sign_acc_pred_active_1p0bps']}")
    print(f"[test_bins] frac={test['bin_frac']} sign_acc={test['bin_sign_acc']} pred_abs_p90_bps={test['bin_pred_abs_p90_bps']} spearman={test['bin_spearman']}")
    print('[done] Training complete.')


# ---------------- Lightweight HFTDataset (when loading into RAM) ----------------
class HFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
    def __len__(self): return int(self.y.shape[0])
    def __getitem__(self, idx): 
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# ---------------- Entry ----------------
if __name__ == "__main__":
    train_from_offline()
