
#!/usr/bin/env python3
"""
CMSSL17_offline.py

Run CMSSL17's model *using prebuilt tokens* produced by offline_ingest.py.
This mirrors the training/eval flow in CMSSL17.py but reads dataset splits
from OUT_ROOT/meta.json and week meta files, avoiding any online feature building.
"""

import os, sys, math, json, csv
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

def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(x[finite]))


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.std(x[finite]))


def _safe_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.median(x[finite]))


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    return float(np.quantile(x[finite], q))


def _safe_frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask.astype(np.float64)))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
        return float("nan")
    return float(num / den)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    finite_idx = np.where(np.isfinite(x))[0]
    if finite_idx.size == 0:
        return out
    xf = x[finite_idx]
    order = np.argsort(xf, kind="mergesort")
    ranks = np.empty_like(xf, dtype=np.float64)
    sorted_vals = xf[order]
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    out[finite_idx] = ranks
    return out


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    xf = x[finite]
    yf = y[finite]
    xstd = float(np.std(xf))
    ystd = float(np.std(yf))
    if xstd <= 0.0 or ystd <= 0.0:
        return float("nan")
    return float(np.mean((xf - float(np.mean(xf))) * (yf - float(np.mean(yf)))) / (xstd * ystd))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = _rankdata_average_ties(x)
    yr = _rankdata_average_ties(y)
    return _safe_pearson(xr, yr)


def _topk_mask_from_score(score: np.ndarray, frac: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n = score.size
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    if frac >= 1.0:
        out[:] = True
        return out
    if frac <= 0.0:
        return out
    k = max(1, int(math.floor(frac * n)))
    order = np.argsort(-score, kind="mergesort")
    out[order[:k]] = True
    return out


def _quantile_bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return np.array([0.0, 1e-12], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(xf, qs)
    edges = np.unique(edges.astype(np.float64))
    if edges.size >= 2:
        return edges
    v = float(xf[0])
    eps = max(1e-12, abs(v) * 1e-12)
    return np.array([v - eps, v + eps], dtype=np.float64)


def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edges.size < 2:
        raise ValueError("edges must contain at least 2 values")
    right = np.searchsorted(edges, x, side="right") - 1
    return np.clip(right, 0, edges.size - 2).astype(np.int64)


def _binary_auc_from_score(score: np.ndarray, target01: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target01 = np.asarray(target01).reshape(-1)
    finite = np.isfinite(score) & np.isfinite(target01)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    s = score[finite]
    t = target01[finite].astype(np.int32)
    n_pos = int(np.sum(t == 1))
    n_neg = int(np.sum(t == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average_ties(s)
    sum_ranks_pos = float(np.sum(ranks[t == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_logit_diagnostics_for_horizon(
    logits_h: np.ndarray,
    y_ret_h: np.ndarray,
    mask_h: np.ndarray,
    pos_lo: float,
    pos_hi: float,
    neg_lo: float,
    neg_hi: float,
    horizon_ms: int,
    split_name: str,
) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    logit = np.asarray(logits_h, dtype=np.float64).reshape(-1)
    r = np.asarray(y_ret_h, dtype=np.float64).reshape(-1)
    provided_mask = np.asarray(mask_h, dtype=bool).reshape(-1)
    if not (logit.size == r.size == provided_mask.size):
        raise ValueError(f"Shape mismatch in diagnostics for split={split_name} horizon={horizon_ms}")

    a = np.abs(r)
    abs_logit = np.abs(logit)
    p_up = 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))
    conf = np.abs(p_up - 0.5) * 2.0
    pred_sign = np.where(logit >= 0.0, 1.0, -1.0)
    realized_sign = np.where(r > 0.0, 1.0, np.where(r < 0.0, -1.0, 0.0))
    sign_aligned_return = pred_sign * r

    is_zero = (r == 0.0)
    is_pos = (r > 0.0)
    is_neg = (r < 0.0)
    is_pos_low = is_pos & (r < float(pos_lo))
    is_pos_kept = is_pos & (r >= float(pos_lo)) & (r <= float(pos_hi))
    is_pos_high = is_pos & (r > float(pos_hi))
    is_neg_low = is_neg & (a < float(neg_lo))
    is_neg_kept = is_neg & (a >= float(neg_lo)) & (a <= float(neg_hi))
    is_neg_high = is_neg & (a > float(neg_hi))
    is_masked = is_pos_kept | is_neg_kept
    is_low_tail = is_pos_low | is_neg_low
    is_high_tail = is_pos_high | is_neg_high
    is_nonzero_unmasked = (~is_masked) & (~is_zero)
    is_dead = is_zero | is_low_tail

    if not np.array_equal(is_masked, provided_mask):
        raise ValueError(f"Mask mismatch in diagnostics for split={split_name} horizon={horizon_ms}")

    partition_stack = np.stack(
        [is_zero, is_pos_low, is_pos_kept, is_pos_high, is_neg_low, is_neg_kept, is_neg_high], axis=0
    )
    if not np.all(np.sum(partition_stack.astype(np.int32), axis=0) == 1):
        raise ValueError(f"Partition coverage mismatch in diagnostics for split={split_name} horizon={horizon_ms}")

    def _part_stats(part_mask: np.ndarray) -> dict:
        return {
            "mean_logit": _safe_mean(logit[part_mask]),
            "std_logit": _safe_std(logit[part_mask]),
            "mean_abs_logit": _safe_mean(abs_logit[part_mask]),
            "median_abs_logit": _safe_median(abs_logit[part_mask]),
            "p75_abs_logit": _safe_quantile(abs_logit[part_mask], 0.75),
            "p90_abs_logit": _safe_quantile(abs_logit[part_mask], 0.90),
            "mean_conf": _safe_mean(conf[part_mask]),
        }

    def _sign_stats(part_mask: np.ndarray) -> dict:
        nz = part_mask & (r != 0.0)
        return {
            "pearson_logit_signed_return": _safe_pearson(logit[part_mask], r[part_mask]),
            "spearman_logit_signed_return": _safe_spearman(logit[part_mask], r[part_mask]),
            "mean_sign_aligned_return": _safe_mean(sign_aligned_return[part_mask]),
            "mean_sign_aligned_return_nonzero": _safe_mean(sign_aligned_return[nz]),
            "sign_accuracy_nonzero": _safe_mean((pred_sign[nz] == realized_sign[nz]).astype(np.float64)),
        }

    def _mag_stats(part_mask: np.ndarray) -> dict:
        return {
            "pearson_abs_logit_abs_return": _safe_pearson(abs_logit[part_mask], a[part_mask]),
            "spearman_abs_logit_abs_return": _safe_spearman(abs_logit[part_mask], a[part_mask]),
            "pearson_conf_abs_return": _safe_pearson(conf[part_mask], a[part_mask]),
            "spearman_conf_abs_return": _safe_spearman(conf[part_mask], a[part_mask]),
        }

    summary: dict = {
        "split_name": split_name,
        "horizon_ms": int(horizon_ms),
        "n_all": int(logit.size),
        "frac_zero": _safe_frac(is_zero),
        "frac_masked": _safe_frac(is_masked),
        "frac_low_tail": _safe_frac(is_low_tail),
        "frac_high_tail": _safe_frac(is_high_tail),
        "frac_nonzero_unmasked": _safe_frac(is_nonzero_unmasked),
        "frac_dead": _safe_frac(is_dead),
        "frac_pos": _safe_frac(is_pos),
        "frac_neg": _safe_frac(is_neg),
    }

    y01 = (r > 0.0).astype(np.float64)
    pred01 = (logit >= 0.0).astype(np.float64)
    summary["auc_all"] = _binary_auc_from_score(logit, y01)
    summary["auc_masked"] = _binary_auc_from_score(logit[is_masked], y01[is_masked])
    summary["acc_all"] = _safe_mean((pred01 == y01).astype(np.float64))
    summary["acc_masked"] = _safe_mean((pred01[is_masked] == y01[is_masked]).astype(np.float64))
    bce_elem_all = np.maximum(logit, 0.0) - logit * y01 + np.log1p(np.exp(-np.abs(logit)))
    summary["bce_all"] = _safe_mean(bce_elem_all)
    summary["bce_masked"] = _safe_mean(bce_elem_all[is_masked])
    summary["pos_rate_all"] = _safe_mean(y01)
    summary["pos_rate_masked"] = _safe_mean(y01[is_masked])

    partitions = {
        "all": np.ones_like(is_zero, dtype=bool),
        "masked": is_masked,
        "zero": is_zero,
        "low_tail": is_low_tail,
        "high_tail": is_high_tail,
        "dead": is_dead,
        "nonzero_unmasked": is_nonzero_unmasked,
    }
    for name, pmask in partitions.items():
        for k, v in _part_stats(pmask).items():
            summary[f"{k}_{name}"] = v

    summary["masked_vs_zero_mean_abs_logit_ratio"] = _safe_ratio(summary["mean_abs_logit_masked"], summary["mean_abs_logit_zero"])
    summary["masked_vs_low_tail_mean_abs_logit_ratio"] = _safe_ratio(summary["mean_abs_logit_masked"], summary["mean_abs_logit_low_tail"])
    summary["masked_vs_dead_mean_abs_logit_ratio"] = _safe_ratio(summary["mean_abs_logit_masked"], summary["mean_abs_logit_dead"])
    summary["high_tail_vs_low_tail_mean_abs_logit_ratio"] = _safe_ratio(summary["mean_abs_logit_high_tail"], summary["mean_abs_logit_low_tail"])

    for name, pmask in {
        "all": partitions["all"],
        "masked": partitions["masked"],
        "nonzero_unmasked": partitions["nonzero_unmasked"],
        "low_tail": partitions["low_tail"],
        "high_tail": partitions["high_tail"],
    }.items():
        for k, v in _sign_stats(pmask).items():
            summary[f"{k}_{name}"] = v

    for name, pmask in {
        "all": partitions["all"],
        "masked": partitions["masked"],
        "nonzero_unmasked": partitions["nonzero_unmasked"],
        "dead": partitions["dead"],
        "low_tail": partitions["low_tail"],
        "high_tail": partitions["high_tail"],
    }.items():
        for k, v in _mag_stats(pmask).items():
            summary[f"{k}_{name}"] = v

    def _auc_between(score: np.ndarray, a_mask: np.ndarray, b_mask: np.ndarray) -> float:
        use = a_mask | b_mask
        if np.count_nonzero(a_mask) == 0 or np.count_nonzero(b_mask) == 0:
            return float("nan")
        t = np.where(a_mask[use], 1.0, 0.0)
        return _binary_auc_from_score(score[use], t)

    summary["auc_abs_logit_masked_vs_zero"] = _auc_between(abs_logit, is_masked, is_zero)
    summary["auc_abs_logit_masked_vs_low_tail"] = _auc_between(abs_logit, is_masked, is_low_tail)
    summary["auc_abs_logit_masked_vs_dead"] = _auc_between(abs_logit, is_masked, is_dead)
    summary["auc_abs_logit_high_tail_vs_dead"] = _auc_between(abs_logit, is_high_tail, is_dead)

    signed_rows: List[dict] = []
    signed_edges = _quantile_bin_edges(logit, n_bins=10)
    signed_bins = _assign_bins(logit, signed_edges)
    for bi in range(signed_edges.size - 1):
        bm = (signed_bins == bi)
        nz = bm & (r != 0.0)
        signed_rows.append({
            "split_name": split_name,
            "horizon_ms": int(horizon_ms),
            "bin_kind": "signed_logit_decile",
            "bin_index": int(bi),
            "bin_left": float(signed_edges[bi]),
            "bin_right": float(signed_edges[bi + 1]),
            "n": int(np.count_nonzero(bm)),
            "frac_of_split": _safe_frac(bm),
            "mean_logit": _safe_mean(logit[bm]),
            "mean_abs_logit": _safe_mean(abs_logit[bm]),
            "mean_signed_return": _safe_mean(r[bm]),
            "mean_abs_return": _safe_mean(a[bm]),
            "pos_rate": _safe_mean((r[bm] > 0.0).astype(np.float64)),
            "masked_frac": _safe_frac(is_masked[bm]),
            "zero_frac": _safe_frac(is_zero[bm]),
            "low_tail_frac": _safe_frac(is_low_tail[bm]),
            "high_tail_frac": _safe_frac(is_high_tail[bm]),
            "sign_accuracy_nonzero": _safe_mean((pred_sign[nz] == realized_sign[nz]).astype(np.float64)),
            "mean_sign_aligned_return": _safe_mean(sign_aligned_return[bm]),
        })

    abs_rows: List[dict] = []
    abs_edges = _quantile_bin_edges(abs_logit, n_bins=10)
    abs_bins = _assign_bins(abs_logit, abs_edges)
    for bi in range(abs_edges.size - 1):
        bm = (abs_bins == bi)
        nz = bm & (r != 0.0)
        abs_rows.append({
            "split_name": split_name,
            "horizon_ms": int(horizon_ms),
            "bin_kind": "abs_logit_decile",
            "bin_index": int(bi),
            "bin_left": float(abs_edges[bi]),
            "bin_right": float(abs_edges[bi + 1]),
            "n": int(np.count_nonzero(bm)),
            "frac_of_split": _safe_frac(bm),
            "mean_abs_logit": _safe_mean(abs_logit[bm]),
            "mean_signed_return": _safe_mean(r[bm]),
            "mean_abs_return": _safe_mean(a[bm]),
            "masked_frac": _safe_frac(is_masked[bm]),
            "zero_frac": _safe_frac(is_zero[bm]),
            "low_tail_frac": _safe_frac(is_low_tail[bm]),
            "high_tail_frac": _safe_frac(is_high_tail[bm]),
            "dead_frac": _safe_frac(is_dead[bm]),
            "sign_accuracy_nonzero": _safe_mean((pred_sign[nz] == realized_sign[nz]).astype(np.float64)),
            "mean_sign_aligned_return": _safe_mean(sign_aligned_return[bm]),
            "mean_conf": _safe_mean(conf[bm]),
        })

    topk_rows: List[dict] = []
    for frac in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]:
        km = _topk_mask_from_score(abs_logit, frac=frac)
        nz = km & (r != 0.0)
        threshold = _safe_quantile(abs_logit[km], 0.0)
        topk_rows.append({
            "split_name": split_name,
            "horizon_ms": int(horizon_ms),
            "frac_selected": float(frac),
            "n": int(np.count_nonzero(km)),
            "threshold_abs_logit_min": threshold,
            "mean_abs_logit": _safe_mean(abs_logit[km]),
            "mean_signed_return": _safe_mean(r[km]),
            "mean_abs_return": _safe_mean(a[km]),
            "masked_frac": _safe_frac(is_masked[km]),
            "zero_frac": _safe_frac(is_zero[km]),
            "low_tail_frac": _safe_frac(is_low_tail[km]),
            "high_tail_frac": _safe_frac(is_high_tail[km]),
            "dead_frac": _safe_frac(is_dead[km]),
            "sign_accuracy_nonzero": _safe_mean((pred_sign[nz] == realized_sign[nz]).astype(np.float64)),
            "mean_sign_aligned_return": _safe_mean(sign_aligned_return[km]),
            "mean_conf": _safe_mean(conf[km]),
        })

    return summary, signed_rows, abs_rows, topk_rows


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
    validate_dataset_label_dim(meta, f"global metadata {out_root / 'meta.json'}")
    trade_history_enabled = meta.get("trade_history_enabled")
    event_stream_mode = meta.get("event_stream_mode")
    print(f"[meta] trade_history_enabled={trade_history_enabled!r}")
    if "event_stream_mode" in meta:
        print(f"[meta] event_stream_mode={event_stream_mode!r}")
    splits = require_four_week_pipeline_splits(meta, out_root)

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
    rl_val = splits["splits"]["rl"]["val"]
    rl_test = splits["splits"]["rl"]["test"]

    train_week_keys = cmssl_train["weeks"]
    tr_weeks = keys_to_paths(train_week_keys, "cmssl.train")
    va_weeks = keys_to_paths(cmssl_val["weeks"], "cmssl.val")
    te_weeks = keys_to_paths(cmssl_test["weeks"], "cmssl.test")
    eval_weeks = keys_to_paths(eval_full["weeks"], "eval.full")
    rl_train_weeks = keys_to_paths(rl_train["weeks"], "rl.train")
    rl_val_weeks = keys_to_paths(rl_val["weeks"], "rl.val")
    rl_test_weeks = keys_to_paths(rl_test["weeks"], "rl.test")

    if not (tr_weeks and va_weeks and te_weeks):
        raise ValueError("CMSSL split metadata must resolve to at least one week for train/val/test")

    week1, week2, week3, week4 = weeks_order
    print(
        "[cmssl weeks] "
        f"train=week1({week1}) val=week2({week2}) test=week3({week3}) eval_full=week4({week4}) "
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
        tr_weeks, va_weeks, te_weeks, eval_weeks, rl_train_weeks, rl_val_weeks, rl_test_weeks
    ):
        for wp in week_group:
            wp_key = str(wp)
            if wp_key not in seen_week_meta_paths:
                seen_week_meta_paths.add(wp_key)
                resolved_split_week_paths.append(wp)

    for wp in resolved_split_week_paths:
        week_meta = read_json(wp)
        validate_dataset_label_dim(week_meta, f"week metadata {wp}")
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
    eval_start, eval_end = int(eval_full["start"]), int(eval_full["end"])

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
        X_ev, y_ev, feat_dim4 = load_split_in_memory_ts(eval_weeks, eval_start, eval_end)
        assert feat_dim1 == feat_dim2 == feat_dim3 == feat_dim4 == F_total, "feat dim mismatch"
        print(
            f"[cmssl split-ts] train=[{tr_start},{tr_end}) N={len(y_tr)} "
            f"val=[{va_start},{va_end}) N={len(y_va)} test=[{te_start},{te_end}) N={len(y_te)} "
            f"eval_full=[{eval_start},{eval_end}) N={len(y_ev)}"
        )

        # Build in-RAM datasets
        ds_train = HFTDataset(X_tr, y_tr)
        ds_val   = HFTDataset(X_va, y_va)
        ds_test  = HFTDataset(X_te, y_te)
        ds_eval  = HFTDataset(X_ev, y_ev)
        print(
            f"[offline-data] train N={len(ds_train)}, "
            f"val N={len(ds_val)}, test N={len(ds_test)}, eval_full N={len(ds_eval)}"
        )
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
        ev_refs = refs_for_weeks_timerange(eval_weeks, eval_start, eval_end)
        print(
            f"[cmssl split-ts] train=[{tr_start},{tr_end}) N={sum(r.n for r in tr_refs)} "
            f"val=[{va_start},{va_end}) N={sum(r.n for r in va_refs)} "
            f"test=[{te_start},{te_end}) N={sum(r.n for r in te_refs)} "
            f"eval_full=[{eval_start},{eval_end}) N={sum(r.n for r in ev_refs)}"
        )

        ds_train = NpyChunksDataset(tr_refs, F_total)
        ds_val   = NpyChunksDataset(va_refs, F_total)
        ds_test  = NpyChunksDataset(te_refs, F_total)
        ds_eval  = NpyChunksDataset(ev_refs, F_total)
        print(
            f"[offline-data] train N={len(ds_train)}, "
            f"val N={len(ds_val)}, test N={len(ds_test)}, eval_full N={len(ds_eval)}"
        )

        # Build y_train_for_quant without loading features into RAM unless cache hit.
        if cached_bounds is not None:
            y_train_for_quant = None
        elif len(ds_train) == 0:
            y_train_for_quant = np.empty((0, NUM_HORIZONS), dtype=np.float32)
        else:
            dl_prepass = DataLoader(
                ds_train,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                num_workers=WORKERS_TRAIN,
                pin_memory=True,
            )
            y_parts: List[np.ndarray] = []
            with torch.no_grad():
                for _, y_batch in tqdm(dl_prepass, desc="[prepass y_train quantiles]"):
                    y_parts.append(y_batch.numpy())
            y_train_for_quant = (
                np.concatenate(y_parts, axis=0)
                if y_parts
                else np.empty((0, NUM_HORIZONS), dtype=np.float32)
            )


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
    dl_eval = DataLoader(
        ds_eval,
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
    opt = SAM(model.parameters(), torch.optim.AdamW, lr=LR, weight_decay=1e-3, rho=0.01)
    torch.cuda.empty_cache()

    # ---------------- Epoch loop ----------------
    best = -float('inf') if primary_metric_mode == "max" else float('inf')
    no_imp = 0
    primary_horizon_idx = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)

    def summarize_directional_metrics(dl: DataLoader, *, primary_only: bool, split_name: str) -> dict:
        model.eval()
        logits_all = [[] for _ in range(NUM_HORIZONS)]
        ypos_all = [[] for _ in range(NUM_HORIZONS)]
        logits_masked = [[] for _ in range(NUM_HORIZONS)]
        ypos_masked = [[] for _ in range(NUM_HORIZONS)]
        yret_all = [[] for _ in range(NUM_HORIZONS)]
        mask_all = [[] for _ in range(NUM_HORIZONS)]
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
                    y_ret_h_all = y_ret[:, h_idx]
                    noise_filter_mask_h = noise_filter_mask_metrics[:, h_idx]
                    bce_sum[h_idx] += bce_elem_fp32[:, h_idx].sum().item()
                    bce_count[h_idx] += targets_h_all.numel()
                    acc_sum[h_idx] += (pred_class[:, h_idx] == true_class[:, h_idx]).sum().item()
                    total[h_idx] += targets_h_all.numel()
                    logits_all[h_idx].append(logits_h_all.detach().cpu())
                    ypos_all[h_idx].append(true_class[:, h_idx].detach().cpu())
                    yret_all[h_idx].append(y_ret_h_all.detach().cpu())
                    mask_all[h_idx].append(noise_filter_mask_h.detach().cpu())

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
        logit_diag_summary_per_h: List[dict] = []
        logit_diag_signed_bins_rows: List[dict] = []
        logit_diag_abs_bins_rows: List[dict] = []
        logit_diag_topk_rows: List[dict] = []

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
            if logits_all[h_idx]:
                logits_h_np = torch.cat(logits_all[h_idx], dim=0).view(-1).numpy()
                y_ret_h_np = torch.cat(yret_all[h_idx], dim=0).view(-1).numpy()
                mask_h_np = torch.cat(mask_all[h_idx], dim=0).view(-1).numpy().astype(bool)
                summary_h, signed_rows_h, abs_rows_h, topk_rows_h = build_logit_diagnostics_for_horizon(
                    logits_h=logits_h_np,
                    y_ret_h=y_ret_h_np,
                    mask_h=mask_h_np,
                    pos_lo=float(pos_lo[h_idx]),
                    pos_hi=float(pos_hi[h_idx]),
                    neg_lo=float(neg_lo[h_idx]),
                    neg_hi=float(neg_hi[h_idx]),
                    horizon_ms=int(HORIZONS_MS[h_idx]),
                    split_name=split_name,
                )
                logit_diag_summary_per_h.append(summary_h)
                logit_diag_signed_bins_rows.extend(signed_rows_h)
                logit_diag_abs_bins_rows.extend(abs_rows_h)
                logit_diag_topk_rows.extend(topk_rows_h)

        primary_metric_value, primary_metric_label = compute_primary_metric(auc_masked)
        return {
            "bce_unmasked": bce,
            "bce_masked": bce_masked,
            "acc_unmasked": acc,
            "acc_masked": acc_masked,
            "auc_unmasked": auc,
            "auc_masked": auc_masked,
            "pos_rate_unmasked": pos_rate_all,
            "logit_mean_unmasked": logit_mean_all,
            "logit_std_unmasked": logit_std_all,
            "pos_rate_masked": pos_rate_masked,
            "logit_mean_masked": logit_mean_masked,
            "logit_std_masked": logit_std_masked,
            "primary_metric_value": float(primary_metric_value),
            "primary_metric_label": primary_metric_label,
            "primary_masked_bce": float(bce_masked[primary_horizon_idx]),
            "primary_masked_auc": float(auc_masked[primary_horizon_idx]),
            "primary_masked_acc": float(acc_masked[primary_horizon_idx]),
            "logit_diag_summary_per_h": logit_diag_summary_per_h,
            "logit_diag_signed_bins_rows": logit_diag_signed_bins_rows,
            "logit_diag_abs_bins_rows": logit_diag_abs_bins_rows,
            "logit_diag_topk_rows": logit_diag_topk_rows,
        }

    def run_validation(*, full_metrics: bool) -> dict:
        return summarize_directional_metrics(dl_val, primary_only=not full_metrics, split_name="val")

    def _write_csv_rows(path: Path, rows: List[dict]) -> None:
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _save_logit_diag_bundle(split: str, metrics: dict) -> None:
        summary_path = out_root / f"cmssl_logit_diagnostics_{split}.json"
        signed_path = out_root / f"cmssl_logit_signed_logit_bins_{split}.csv"
        abs_path = out_root / f"cmssl_logit_abs_logit_bins_{split}.csv"
        topk_path = out_root / f"cmssl_logit_topk_abs_logit_{split}.csv"
        payload = {
            "split": split,
            "horizons_ms": [int(h) for h in HORIZONS_MS],
            "tail_fraction": float(DIR_MASK_TAIL_FRACTION),
            "summary_per_horizon": metrics["logit_diag_summary_per_h"],
        }
        with summary_path.open("w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        _write_csv_rows(signed_path, metrics["logit_diag_signed_bins_rows"])
        _write_csv_rows(abs_path, metrics["logit_diag_abs_bins_rows"])
        _write_csv_rows(topk_path, metrics["logit_diag_topk_rows"])
        print(
            f"[logit_diag_saved][{split}] json={summary_path} signed_bins={signed_path} "
            f"abs_bins={abs_path} topk={topk_path}"
        )

    def _print_logit_diag_compact(split: str, metrics: dict) -> None:
        top10_by_h = {
            int(row["horizon_ms"]): row
            for row in metrics["logit_diag_topk_rows"]
            if abs(float(row.get("frac_selected", -1.0)) - 0.10) < 1e-12
        }
        for row in metrics["logit_diag_summary_per_h"]:
            h = int(row["horizon_ms"])
            top10 = top10_by_h.get(h, {})
            print(
                f"[logit_diag][{split}][{h}ms] "
                f"horizon_ms={h} "
                f"frac_masked={row.get('frac_masked', float('nan')):.4f} "
                f"frac_zero={row.get('frac_zero', float('nan')):.4f} "
                f"frac_low_tail={row.get('frac_low_tail', float('nan')):.4f} "
                f"frac_high_tail={row.get('frac_high_tail', float('nan')):.4f} "
                f"mean_abs_logit_masked={row.get('mean_abs_logit_masked', float('nan')):.6f} "
                f"mean_abs_logit_zero={row.get('mean_abs_logit_zero', float('nan')):.6f} "
                f"mean_abs_logit_low_tail={row.get('mean_abs_logit_low_tail', float('nan')):.6f} "
                f"mean_abs_logit_high_tail={row.get('mean_abs_logit_high_tail', float('nan')):.6f} "
                f"auc_abs_logit_masked_vs_dead={row.get('auc_abs_logit_masked_vs_dead', float('nan')):.6f} "
                f"pearson_logit_signed_return_all={row.get('pearson_logit_signed_return_all', float('nan')):.6f} "
                f"spearman_logit_signed_return_all={row.get('spearman_logit_signed_return_all', float('nan')):.6f} "
                f"pearson_abs_logit_abs_return_all={row.get('pearson_abs_logit_abs_return_all', float('nan')):.6f} "
                f"mean_sign_aligned_return_top10pct={float(top10.get('mean_sign_aligned_return', float('nan'))):.6f} "
                f"masked_frac_top10pct={float(top10.get('masked_frac', float('nan'))):.4f} "
                f"zero_frac_top10pct={float(top10.get('zero_frac', float('nan'))):.4f}"
            )

    for epoch in range(EPOCHS):
        early_stop_triggered = False
        model.train()
        pbar = tqdm(dl_train, desc=f"Ep{epoch+1}/{EPOCHS}")
        num_train_batches = len(dl_train)
        running_loss_t = torch.zeros((), device=device, dtype=torch.float32)
        running_bce_t = torch.zeros((), device=device, dtype=torch.float32)
        n_batches = 0

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_ret = y

            opt.base_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                dir_logits = model(x)
                bce_loss = compute_directional_loss(dir_logits, y_ret)
                loss = bce_loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss in SAM pass #1: {float(loss.detach().float().cpu())}"
                )

            running_bce_t += bce_loss.detach().float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000)
            opt.first_step(zero_grad=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                dir_logits2 = model(x)
                bce_loss2 = compute_directional_loss(dir_logits2, y_ret)
                loss2 = bce_loss2

            if not torch.isfinite(loss2):
                raise RuntimeError(
                    f"Non-finite training loss in SAM pass #2: {float(loss2.detach().float().cpu())}"
                )

            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000)
            opt.second_step(zero_grad=True)

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
                    f"[val] BCE(all)={format_metric(full_val['bce_unmasked'], '{:.5f}')}  "
                    f"BCE(mask)={format_metric(full_val['bce_masked'], '{:.5f}')}  "
                    f"Acc(all)={format_metric(full_val['acc_unmasked'], '{:.3%}')}  "
                    f"Acc(mask)={format_metric(full_val['acc_masked'], '{:.3%}')}  "
                    f"AUC(all)={format_metric(full_val['auc_unmasked'], '{:.3f}')}  "
                    f"AUC(mask)={format_metric(full_val['auc_masked'], '{:.3f}')}")
                print(
                    f"[val_diag] pos_rate(all)={format_metric(full_val['pos_rate_unmasked'], '{:.3%}')}  "
                    f"logit_mean(all)={format_metric(full_val['logit_mean_unmasked'], '{:.3f}')}  "
                    f"logit_std(all)={format_metric(full_val['logit_std_unmasked'], '{:.3f}')}  "
                    f"pos_rate(mask)={format_metric(full_val['pos_rate_masked'], '{:.3%}')}  "
                    f"logit_mean(mask)={format_metric(full_val['logit_mean_masked'], '{:.3f}')}  "
                    f"logit_std(mask)={format_metric(full_val['logit_std_masked'], '{:.3f}')}")
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
                        "checkpoint_schema": "cmssl17-direction-only-v1",
                        "trade_history_enabled": trade_history_enabled,
                        "event_stream_mode": event_stream_mode,
                        "decision_time_basis": meta.get("decision_time_basis"),
                    },
                    "best_primary_metric": best,
                }
                out_ckpt = out_root / "cmssl17_offline_best.pt"
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

    # ---------------- Final Split Evaluations ----------------
    val_metrics = summarize_directional_metrics(dl_val, primary_only=False, split_name="val")
    test_metrics = summarize_directional_metrics(dl_test, primary_only=False, split_name="test")
    eval_metrics = summarize_directional_metrics(dl_eval, primary_only=False, split_name="eval_full")

    print(
        f"[val] BCE(all)={format_metric(val_metrics['bce_unmasked'], '{:.4e}')}  "
        f"Acc(all)={format_metric(val_metrics['acc_unmasked'], '{:.4f}')}  "
        f"AUC(all)={format_metric(val_metrics['auc_unmasked'], '{:.4f}')}")
    print(
        f"  BCE(mask)={format_metric(val_metrics['bce_masked'], '{:.4e}')}  "
        f"Acc(mask)={format_metric(val_metrics['acc_masked'], '{:.4f}')}  "
        f"AUC(mask)={format_metric(val_metrics['auc_masked'], '{:.4f}')}")
    print(
        f"[val_diag] pos_rate(all)={format_metric(val_metrics['pos_rate_unmasked'], '{:.3%}')}  "
        f"logit_mean(all)={format_metric(val_metrics['logit_mean_unmasked'], '{:.3f}')}  "
        f"logit_std(all)={format_metric(val_metrics['logit_std_unmasked'], '{:.3f}')}  "
        f"pos_rate(mask)={format_metric(val_metrics['pos_rate_masked'], '{:.3%}')}  "
        f"logit_mean(mask)={format_metric(val_metrics['logit_mean_masked'], '{:.3f}')}  "
        f"logit_std(mask)={format_metric(val_metrics['logit_std_masked'], '{:.3f}')}")

    print(
        f"[test] BCE(all)={format_metric(test_metrics['bce_unmasked'], '{:.4e}')}  "
        f"Acc(all)={format_metric(test_metrics['acc_unmasked'], '{:.4f}')}  "
        f"AUC(all)={format_metric(test_metrics['auc_unmasked'], '{:.4f}')}")
    print(
        f"  BCE(mask)={format_metric(test_metrics['bce_masked'], '{:.4e}')}  "
        f"Acc(mask)={format_metric(test_metrics['acc_masked'], '{:.4f}')}  "
        f"AUC(mask)={format_metric(test_metrics['auc_masked'], '{:.4f}')}")
    print(
        f"[test_diag] pos_rate(all)={format_metric(test_metrics['pos_rate_unmasked'], '{:.3%}')}  "
        f"logit_mean(all)={format_metric(test_metrics['logit_mean_unmasked'], '{:.3f}')}  "
        f"logit_std(all)={format_metric(test_metrics['logit_std_unmasked'], '{:.3f}')}  "
        f"pos_rate(mask)={format_metric(test_metrics['pos_rate_masked'], '{:.3%}')}  "
        f"logit_mean(mask)={format_metric(test_metrics['logit_mean_masked'], '{:.3f}')}  "
        f"logit_std(mask)={format_metric(test_metrics['logit_std_masked'], '{:.3f}')}")

    print(
        f"[eval_full] BCE(all)={format_metric(eval_metrics['bce_unmasked'], '{:.4e}')}  "
        f"Acc(all)={format_metric(eval_metrics['acc_unmasked'], '{:.4f}')}  "
        f"AUC(all)={format_metric(eval_metrics['auc_unmasked'], '{:.4f}')}")
    print(
        f"  BCE(mask)={format_metric(eval_metrics['bce_masked'], '{:.4e}')}  "
        f"Acc(mask)={format_metric(eval_metrics['acc_masked'], '{:.4f}')}  "
        f"AUC(mask)={format_metric(eval_metrics['auc_masked'], '{:.4f}')}")
    print(
        f"[eval_full_diag] pos_rate(all)={format_metric(eval_metrics['pos_rate_unmasked'], '{:.3%}')}  "
        f"logit_mean(all)={format_metric(eval_metrics['logit_mean_unmasked'], '{:.3f}')}  "
        f"logit_std(all)={format_metric(eval_metrics['logit_std_unmasked'], '{:.3f}')}  "
        f"pos_rate(mask)={format_metric(eval_metrics['pos_rate_masked'], '{:.3%}')}  "
        f"logit_mean(mask)={format_metric(eval_metrics['logit_mean_masked'], '{:.3f}')}  "
        f"logit_std(mask)={format_metric(eval_metrics['logit_std_masked'], '{:.3f}')}")

    _print_logit_diag_compact("val", val_metrics)
    _print_logit_diag_compact("test", test_metrics)
    _print_logit_diag_compact("eval_full", eval_metrics)

    _save_logit_diag_bundle("val", val_metrics)
    _save_logit_diag_bundle("test", test_metrics)
    _save_logit_diag_bundle("eval_full", eval_metrics)

    print("[done] Training complete.")

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
