#!/usr/bin/env python3
"""Standalone feature audit for offline_ingest.py flat outputs.

The audit intentionally uses only the CMSSL training weeks and samples label rows
systematically before looking up their matching feature rows.  It writes compact
CSV/JSON reports intended to guide low-risk feature pruning without running
training ablations.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
AUDIT_OUT_DIR = os.environ.get("BYBIT_AUDIT_OUT_DIR", "").strip()
AUDIT_SEED = int(os.environ.get("BYBIT_AUDIT_SEED", "17"))
AUDIT_MAX_ROWS_TOTAL = int(os.environ.get("BYBIT_AUDIT_MAX_ROWS_TOTAL", "1000000"))
AUDIT_MAX_ROWS_PER_WEEK = int(os.environ.get("BYBIT_AUDIT_MAX_ROWS_PER_WEEK", "600000"))
AUDIT_MI_MAX_ROWS = int(os.environ.get("BYBIT_AUDIT_MI_MAX_ROWS", "200000"))
AUDIT_USE_AUX = int(os.environ.get("BYBIT_AUDIT_USE_AUX", "0")) == 1
AUDIT_CORR_METHODS = os.environ.get("BYBIT_AUDIT_CORR_METHODS", "pearson,spearman")
AUDIT_HIGH_CORR = float(os.environ.get("BYBIT_AUDIT_HIGH_CORR", "0.95"))
AUDIT_MED_CORR = float(os.environ.get("BYBIT_AUDIT_MED_CORR", "0.90"))
AUDIT_MIN_ABS_LABEL_EPS = float(os.environ.get("BYBIT_AUDIT_MIN_ABS_LABEL_EPS", "0.0"))
AUDIT_TOP_CORR_PAIRS = int(os.environ.get("BYBIT_AUDIT_TOP_CORR_PAIRS", "5000"))
AUDIT_TOP_REMOVAL_CANDIDATES = int(os.environ.get("BYBIT_AUDIT_TOP_REMOVAL_CANDIDATES", "200"))

EPS = 1e-12

SEMANTIC_PRIORITY = {
    "calendar": 0.80,
    "notional_context": 0.90,
    "top_book": 0.90,
    "spread_gap": 0.85,
    "returns": 0.95,
    "micro_vamp": 0.85,
    "depth": 0.80,
    "depth_imbalance": 0.90,
    "obi": 0.90,
    "rolling_obi": 0.75,
    "ofi_raw": 0.70,
    "ofi_normalized": 0.90,
    "ofi_pressure": 0.95,
    "trade_count": 0.75,
    "trade_activity": 0.75,
    "trade_imbalance": 0.90,
    "trade_toxicity": 0.90,
    "book_dynamics": 0.80,
    "signed_notional_flow": 0.90,
    "cvd": 0.90,
    "absorption": 0.85,
    "volatility_regime": 0.85,
    "spread_depth_regime": 0.80,
    "event_density_aux": 0.50,
    "aux": 0.50,
    "unknown": 0.50,
}

FAST_FAMILIES = {
    "top_book",
    "book_dynamics",
    "spread_gap",
    "returns",
    "micro_vamp",
    "ofi_raw",
    "ofi_normalized",
    "ofi_pressure",
    "trade_activity",
    "trade_imbalance",
    "trade_toxicity",
    "signed_notional_flow",
    "absorption",
}
SLOW_FAMILIES = {
    "calendar",
    "notional_context",
    "depth",
    "depth_imbalance",
    "rolling_obi",
    "cvd",
    "volatility_regime",
    "spread_depth_regime",
    "event_density_aux",
    "aux",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def resolve_chunk_path(out_root: Path, week_key: str, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    cand1 = out_root / p
    if cand1.exists():
        return cand1
    return out_root / week_key / p


def iter_chunk_file_refs(chunks: list, kind: str) -> List[dict]:
    if kind == "feature":
        required = ("features", "ts")
    elif kind == "label":
        required = ("row_idx", "label_ts", "y")
    else:
        raise ValueError(f"Unsupported chunk kind={kind!r}")

    refs: List[dict] = []
    for idx, entry in enumerate(chunks):
        if isinstance(entry, str):
            # Older metadata fallback: keep the path under a generic key and let callers fail
            # clearly if this legacy shape is insufficient for the requested kind.
            refs.append({"chunk": idx, "n": None, "files": {kind: entry}, "raw": entry})
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid {kind} chunk entry {idx}: expected dict or string, got {type(entry).__name__}")
        files = entry.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"Invalid {kind} chunk entry {idx}: missing files dict")
        missing = [key for key in required if not isinstance(files.get(key), str) or not files.get(key)]
        if missing:
            raise ValueError(f"Invalid {kind} chunk entry {idx}: missing files {missing}")
        rec = dict(entry)
        rec["files"] = dict(files)
        refs.append(rec)
    return refs


def systematic_positions(n_total: int, n_sample: int, seed: int) -> np.ndarray:
    n_total = int(n_total)
    n_sample = int(max(0, n_sample))
    if n_total <= 0 or n_sample <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_sample >= n_total:
        return np.arange(n_total, dtype=np.int64)
    step = n_total / n_sample
    rng = np.random.default_rng(seed)
    offset = rng.uniform(0.0, step)
    pos = np.floor(offset + np.arange(n_sample) * step).astype(np.int64)
    return np.unique(np.clip(pos, 0, n_total - 1))


def resolve_week_meta_path(out_root: Path, weeks_meta: dict, week_key: str) -> Path:
    rel = weeks_meta.get(week_key)
    if not rel:
        raise KeyError(f"meta['weeks_meta'] missing path for train week {week_key!r}")
    p = Path(rel)
    return p if p.is_absolute() else out_root / p


def proportional_week_targets(labels_by_week: Dict[str, int], max_total: int, max_per_week: int) -> Dict[str, int]:
    total_labels = sum(max(0, int(v)) for v in labels_by_week.values())
    if total_labels <= 0:
        return {wk: 0 for wk in labels_by_week}
    raw = {wk: min(int(max_per_week), int(math.floor(max_total * (n / total_labels)))) for wk, n in labels_by_week.items()}
    for wk, n in labels_by_week.items():
        if n > 0 and raw[wk] <= 0:
            raw[wk] = 1
    # Distribute any leftover to weeks that can still accept rows.
    while sum(raw.values()) < int(max_total):
        eligible = [wk for wk, n in labels_by_week.items() if raw[wk] < min(int(max_per_week), int(n))]
        if not eligible:
            break
        for wk in sorted(eligible, key=lambda k: labels_by_week[k], reverse=True):
            if sum(raw.values()) >= int(max_total):
                break
            raw[wk] += 1
    return {wk: min(int(labels_by_week[wk]), int(v)) for wk, v in raw.items()}


def positions_to_chunk_offsets(positions: np.ndarray, chunks: List[dict], start_key: str, end_key: str) -> Dict[int, np.ndarray]:
    out: Dict[int, List[int]] = defaultdict(list)
    if positions.size == 0:
        return {}
    starts = np.array([int(c.get(start_key, 0)) for c in chunks], dtype=np.int64)
    ends = np.array([int(c.get(end_key, int(c.get(start_key, 0)) + int(c.get("n", 0)))) for c in chunks], dtype=np.int64)
    chunk_ids = np.searchsorted(ends, positions, side="right")
    for p, cid in zip(positions, chunk_ids):
        if cid < 0 or cid >= len(chunks) or not (starts[cid] <= p < ends[cid]):
            raise IndexError(f"Position {int(p)} did not map to a valid chunk")
        out[int(cid)].append(int(p - starts[cid]))
    return {cid: np.asarray(vals, dtype=np.int64) for cid, vals in out.items()}


def build_feature_chunk_index(feature_chunks: List[dict], out_root: Path, week_key: str) -> List[dict]:
    refs = iter_chunk_file_refs(feature_chunks, "feature")
    idx: List[dict] = []
    cursor = 0
    for rec in refs:
        start = int(rec.get("row_start", cursor))
        n = int(rec.get("n", 0))
        end = int(rec.get("row_end", start + n))
        files = rec["files"]
        idx.append({
            "start_row": start,
            "end_row": end,
            "n": end - start,
            "features_path": resolve_chunk_path(out_root, week_key, files["features"]),
            "ts_path": resolve_chunk_path(out_root, week_key, files["ts"]),
        })
        cursor = end
    idx.sort(key=lambda x: int(x["start_row"]))
    return idx


def load_labels_for_positions(out_root: Path, week_key: str, label_chunks: List[dict], positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = iter_chunk_file_refs(label_chunks, "label")
    offsets = positions_to_chunk_offsets(positions, refs, "label_start", "label_end")
    row_idx_parts: List[np.ndarray] = []
    label_ts_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for cid in sorted(offsets):
        rec = refs[cid]
        files = rec["files"]
        off = offsets[cid]
        row_idx_arr = np.load(resolve_chunk_path(out_root, week_key, files["row_idx"]), mmap_mode="r")
        label_ts_arr = np.load(resolve_chunk_path(out_root, week_key, files["label_ts"]), mmap_mode="r")
        y_arr = np.load(resolve_chunk_path(out_root, week_key, files["y"]), mmap_mode="r")
        row_idx_parts.append(np.asarray(row_idx_arr[off], dtype=np.int64))
        label_ts_parts.append(np.asarray(label_ts_arr[off], dtype=np.int64))
        y_parts.append(np.asarray(y_arr[off], dtype=np.float32))
    if not row_idx_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    return np.concatenate(row_idx_parts), np.concatenate(label_ts_parts), np.concatenate(y_parts, axis=0)


def load_features_by_row_idx(out_root: Path, week_key: str, feature_chunks: List[dict], row_idx: np.ndarray, audit_indices: np.ndarray) -> np.ndarray:
    index = build_feature_chunk_index(feature_chunks, out_root, week_key)
    if row_idx.size == 0:
        return np.empty((0, len(audit_indices)), dtype=np.float32)
    starts = np.array([int(c["start_row"]) for c in index], dtype=np.int64)
    ends = np.array([int(c["end_row"]) for c in index], dtype=np.int64)
    order = np.argsort(row_idx, kind="mergesort")
    sorted_rows = row_idx[order]
    out = np.empty((row_idx.shape[0], len(audit_indices)), dtype=np.float32)
    chunk_ids = np.searchsorted(ends, sorted_rows, side="right")
    for cid in np.unique(chunk_ids):
        if cid < 0 or cid >= len(index):
            raise IndexError(f"Feature row_idx maps outside chunks: chunk_id={cid}")
        mask = chunk_ids == cid
        rows = sorted_rows[mask]
        if np.any(rows < starts[cid]) or np.any(rows >= ends[cid]):
            raise IndexError(f"Feature row_idx outside chunk bounds for week={week_key} chunk={cid}")
        local = rows - starts[cid]
        arr = np.load(index[cid]["features_path"], mmap_mode="r")
        vals = np.asarray(arr[local][:, audit_indices], dtype=np.float32)
        out[order[mask]] = vals
    return out


def choose_finite_label_positions(out_root: Path, week_key: str, label_chunks: List[dict], candidate_positions: np.ndarray, label_dim: int) -> np.ndarray:
    """Keep candidates whose labels are finite across all horizons, without loading all labels."""
    refs = iter_chunk_file_refs(label_chunks, "label")
    offsets = positions_to_chunk_offsets(candidate_positions, refs, "label_start", "label_end")
    keep_parts: List[np.ndarray] = []
    for cid in sorted(offsets):
        rec = refs[cid]
        off = offsets[cid]
        y = np.load(resolve_chunk_path(out_root, week_key, rec["files"]["y"]), mmap_mode="r")
        yy = np.asarray(y[off], dtype=np.float32)
        if yy.ndim != 2 or yy.shape[1] != int(label_dim):
            raise ValueError(f"Label shape mismatch for week={week_key} chunk={cid}: got {yy.shape}, expected H={label_dim}")
        keep_parts.append(np.all(np.isfinite(yy), axis=1))
    if not keep_parts:
        return candidate_positions[:0]
    keep = np.concatenate(keep_parts)
    return candidate_positions[keep]


def load_sampled_dataset(out_root: Path, global_meta: dict, audit_indices: np.ndarray, warnings_out: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    splits = global_meta.get("splits", {})
    try:
        train_weeks = list(splits["cmssl"]["train"]["weeks"])
    except Exception as exc:
        raise KeyError("meta.json missing required splits['cmssl']['train']['weeks']; refusing to use non-train weeks") from exc
    if not train_weeks:
        raise ValueError("CMSSL train split has no weeks; refusing to audit")

    weeks_meta = dict(global_meta["weeks_meta"])
    week_metas: Dict[str, dict] = {}
    labels_by_week: Dict[str, int] = {}
    for wk in train_weeks:
        wm = read_json(resolve_week_meta_path(out_root, weeks_meta, wk))
        week_metas[wk] = wm
        labels_by_week[wk] = int(wm.get("labels_total", sum(int(c.get("n", 0)) for c in wm.get("label_chunks", []))))
    targets = proportional_week_targets(labels_by_week, AUDIT_MAX_ROWS_TOTAL, AUDIT_MAX_ROWS_PER_WEEK)

    X_parts: List[np.ndarray] = []
    Y_parts: List[np.ndarray] = []
    row_parts: List[np.ndarray] = []
    ts_parts: List[np.ndarray] = []
    week_id_parts: List[np.ndarray] = []
    week_key_parts: List[np.ndarray] = []
    sample_rows_by_week: Dict[str, int] = {}
    label_dim = int(global_meta["label_dim"])

    for widx, wk in enumerate(train_weeks):
        wm = week_metas[wk]
        labels_total = int(labels_by_week[wk])
        target = int(targets.get(wk, 0))
        # Oversample candidate positions modestly to compensate for any nonfinite labels.
        cand_n = min(labels_total, max(target, int(math.ceil(target * 1.10)) + 128))
        candidates = systematic_positions(labels_total, cand_n, AUDIT_SEED + widx * 1009)
        positions = choose_finite_label_positions(out_root, wk, wm.get("label_chunks", []), candidates, label_dim)[:target]
        if positions.size < target:
            msg = f"week {wk} finite-label sample smaller than requested: {positions.size} < {target}"
            warnings_out.append(msg)
        row_idx, label_ts, y = load_labels_for_positions(out_root, wk, wm.get("label_chunks", []), positions)
        if y.size and not np.all(np.isfinite(y)):
            finite = np.all(np.isfinite(y), axis=1)
            row_idx, label_ts, y = row_idx[finite], label_ts[finite], y[finite]
        x = load_features_by_row_idx(out_root, wk, wm.get("feature_chunks", []), row_idx, audit_indices)
        X_parts.append(x)
        Y_parts.append(y)
        row_parts.append(row_idx)
        ts_parts.append(label_ts)
        week_id_parts.append(np.full(row_idx.shape[0], widx, dtype=np.int32))
        week_key_parts.append(np.asarray([wk] * row_idx.shape[0], dtype=object))
        sample_rows_by_week[wk] = int(row_idx.shape[0])
        print(f"[audit-sample] week={wk} labels_total={labels_total} sampled={row_idx.shape[0]}", flush=True)

    if not X_parts:
        raise RuntimeError("No sampled rows loaded")
    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    Y = np.concatenate(Y_parts, axis=0).astype(np.float32, copy=False)
    return (
        X,
        Y,
        np.concatenate(row_parts).astype(np.int64, copy=False),
        np.concatenate(ts_parts).astype(np.int64, copy=False),
        np.concatenate(week_id_parts).astype(np.int32, copy=False),
        np.concatenate(week_key_parts),
        sample_rows_by_week,
    )


def side_specific_keep_mask(y: np.ndarray, low_frac: float, high_frac: float) -> np.ndarray:
    finite = np.isfinite(y)
    pos = finite & (y > AUDIT_MIN_ABS_LABEL_EPS)
    neg = finite & (y < -AUDIT_MIN_ABS_LABEL_EPS)
    keep = np.zeros_like(finite, dtype=bool)
    for side_mask in (pos, neg):
        vals = np.abs(y[side_mask])
        if len(vals) == 0:
            continue
        lo = np.quantile(vals, low_frac)
        hi = np.quantile(vals, 1.0 - high_frac)
        side_indices = np.where(side_mask)[0]
        keep_side = (vals >= lo) & (vals <= hi)
        keep[side_indices[keep_side]] = True
    return keep


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask].astype(np.float64, copy=False)
    yy = y[mask].astype(np.float64, copy=False)
    sx = float(np.std(xx))
    sy = float(np.std(yy))
    if sx <= EPS or sy <= EPS:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def rank_1d(a: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata  # type: ignore

        return rankdata(a, method="average").astype(np.float64, copy=False)
    except Exception:
        return pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return safe_corr(rank_1d(x[mask]), rank_1d(y[mask]))


def auc_score_binary(y01: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(y01)
    y = y01[mask].astype(np.int8, copy=False)
    s = scores[mask]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rank_1d(s)
    sum_pos = float(np.sum(ranks[y == 1]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balanced_accuracy_at_threshold(y01: np.ndarray, x: np.ndarray, thr: float) -> float:
    pred = x > thr
    pos = y01 == 1
    neg = y01 == 0
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    tpr = float(np.mean(pred[pos]))
    tnr = float(np.mean(~pred[neg]))
    return 0.5 * (tpr + tnr)


def best_balanced_accuracy(y01: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y01)
    y = y01[mask]
    xx = x[mask]
    if xx.size < 10 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(xx, qs))
    best = -1.0
    best_thr = float("nan")
    for thr in thresholds:
        ba = balanced_accuracy_at_threshold(y, xx, float(thr))
        ba_flip = balanced_accuracy_at_threshold(y, -xx, float(-thr))
        if np.isfinite(ba) and ba > best:
            best, best_thr = float(ba), float(thr)
        if np.isfinite(ba_flip) and ba_flip > best:
            best, best_thr = float(ba_flip), float(thr)
    return float(best), float(best_thr)



def sign_balanced_accuracy(y01: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y01)
    y = y01[mask].astype(np.int8, copy=False)
    xx = x[mask]
    if xx.size < 3 or len(np.unique(y)) < 2:
        return float("nan")

    ba_pos = balanced_accuracy_at_threshold(y, xx, 0.0)
    ba_neg = balanced_accuracy_at_threshold(y, -xx, 0.0)

    vals = [v for v in (ba_pos, ba_neg) if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def compute_health(X: np.ndarray, feature_names: List[str], week_id: np.ndarray, n_weeks: int) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        finite = np.isfinite(col)
        finite_vals = col[finite]
        if finite_vals.size:
            qs = np.quantile(finite_vals.astype(np.float64), [0.01, 0.05, 0.50, 0.95, 0.99])
            mean = float(np.mean(finite_vals))
            std = float(np.std(finite_vals))
            min_v = float(np.min(finite_vals))
            max_v = float(np.max(finite_vals))
            abs_max = float(np.max(np.abs(finite_vals)))
            zero_frac = float(np.mean(finite_vals == 0.0))
            near_zero_frac = float(np.mean(np.abs(finite_vals) < 1e-6))
        else:
            qs = [float("nan")] * 5
            mean = std = min_v = max_v = abs_max = zero_frac = near_zero_frac = float("nan")
        std_by_week = []
        mean_by_week = []
        for wid in range(n_weeks):
            vals = col[(week_id == wid) & finite]
            std_by_week.append(float(np.std(vals)) if vals.size else float("nan"))
            mean_by_week.append(float(np.mean(vals)) if vals.size else float("nan"))
        std_arr = np.asarray([v for v in std_by_week if np.isfinite(v)], dtype=np.float64)
        week_std_cv = float(np.std(std_arr) / (np.mean(std_arr) + EPS)) if std_arr.size else float("nan")
        rows.append({
            "feature": name,
            "feature_index": j,
            "mean": mean,
            "std": std,
            "p01": float(qs[0]),
            "p05": float(qs[1]),
            "p50": float(qs[2]),
            "p95": float(qs[3]),
            "p99": float(qs[4]),
            "min": min_v,
            "max": max_v,
            "abs_max": abs_max,
            "zero_frac": zero_frac,
            "near_zero_frac_abs_lt_1e-6": near_zero_frac,
            "finite_frac": float(np.mean(finite)) if col.size else float("nan"),
            "nan_count": int(np.isnan(col).sum()),
            "inf_count": int(np.isinf(col).sum()),
            "std_by_week": json.dumps(std_by_week),
            "mean_by_week": json.dumps(mean_by_week),
            "week_std_cv": week_std_cv,
        })
    return pd.DataFrame(rows)


def deterministic_subsample(mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size <= max_rows:
        return idx
    pos = systematic_positions(idx.size, max_rows, seed)
    return idx[pos]


def compute_mi_optional(X: np.ndarray, y: np.ndarray, is_classif: bool, seed: int, warnings_out: List[str]) -> np.ndarray:
    try:
        if is_classif:
            from sklearn.feature_selection import mutual_info_classif  # type: ignore

            return mutual_info_classif(X, y, discrete_features=False, random_state=seed)
        from sklearn.feature_selection import mutual_info_regression  # type: ignore

        return mutual_info_regression(X, y, discrete_features=False, random_state=seed)
    except Exception as exc:
        msg = f"sklearn mutual information skipped: {type(exc).__name__}: {exc}"
        if msg not in warnings_out:
            warnings_out.append(msg)
        return np.full(X.shape[1], np.nan, dtype=np.float64)


def compute_target_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    horizons_ms: List[int],
    masks_by_horizon: Dict[int, Dict[str, np.ndarray]],
    warnings_out: List[str],
) -> pd.DataFrame:
    rows: List[dict] = []
    for h, horizon in enumerate(horizons_ms):
        y_h = Y[:, h]
        # MI is only required on kept mask; copy to all mask rows as NaN except kept.
        kept_idx = deterministic_subsample(masks_by_horizon[h]["kept"], AUDIT_MI_MAX_ROWS, AUDIT_SEED + h * 10007)
        mi_dir = np.full(X.shape[1], np.nan)
        mi_abs = np.full(X.shape[1], np.nan)
        if kept_idx.size >= 20 and len(np.unique((y_h[kept_idx] > 0).astype(np.int8))) == 2:
            mi_dir = compute_mi_optional(X[kept_idx], (y_h[kept_idx] > 0).astype(np.int8), True, AUDIT_SEED + h, warnings_out)
            mi_abs = compute_mi_optional(X[kept_idx], np.abs(y_h[kept_idx]), False, AUDIT_SEED + h, warnings_out)
        for mask_type, mask in masks_by_horizon[h].items():
            idx = np.where(mask)[0]
            yy = y_h[idx]
            direction = (yy > 0).astype(np.int8)
            abs_y = np.abs(yy)
            for j, name in enumerate(feature_names):
                xx = X[idx, j]
                pearson_signed = safe_corr(xx, yy)
                spearman_signed = safe_spearman(xx, yy)
                pearson_abs = safe_corr(xx, abs_y)
                spearman_abs = safe_spearman(xx, abs_y)
                auc_raw = auc_score_binary(direction, xx)
                if np.isfinite(auc_raw):
                    auc_best = max(float(auc_raw), 1.0 - float(auc_raw))
                    auc_sign = 1 if auc_raw >= 0.5 else -1
                else:
                    auc_best = float("nan")
                    auc_sign = 0
                bal_sign = sign_balanced_accuracy(direction, xx)
                bal_best, bal_thr = best_balanced_accuracy(direction, xx)
                rows.append({
                    "feature": name,
                    "feature_index": j,
                    "horizon_ms": int(horizon),
                    "mask_type": mask_type,
                    "n_rows": int(idx.size),
                    "pearson_signed_return": pearson_signed,
                    "spearman_signed_return": spearman_signed,
                    "pearson_abs_return": pearson_abs,
                    "spearman_abs_return": spearman_abs,
                    "single_feature_auc_direction": auc_best,
                    "single_feature_auc_direction_sign": auc_sign,
                    "single_feature_bal_acc_sign": bal_sign,
                    "single_feature_bal_acc_best_threshold": bal_best,
                    "single_feature_bal_acc_best_threshold_value": bal_thr,
                    "mi_direction": float(mi_dir[j]) if mask_type == "kept" else float("nan"),
                    "mi_abs_return": float(mi_abs[j]) if mask_type == "kept" else float("nan"),
                })
    return pd.DataFrame(rows)


def corrcoef_pairwise_complete(X: np.ndarray) -> np.ndarray:
    X64 = X.astype(np.float64, copy=False)
    good_rows = np.all(np.isfinite(X64), axis=1)
    if int(good_rows.sum()) < 3:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    return np.corrcoef(X64[good_rows], rowvar=False)


def spearman_corr_matrix(X: np.ndarray) -> np.ndarray:
    good_rows = np.all(np.isfinite(X), axis=1)
    if int(good_rows.sum()) < 3:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    Xg = X[good_rows]
    ranks = np.empty_like(Xg, dtype=np.float32)
    for j in range(Xg.shape[1]):
        ranks[:, j] = rank_1d(Xg[:, j]).astype(np.float32)
    return np.corrcoef(ranks.astype(np.float64, copy=False), rowvar=False)


def top_corr_pairs(pearson: np.ndarray, spearman: np.ndarray, feature_names: List[str], limit: int) -> pd.DataFrame:
    rows = []
    F = len(feature_names)
    for i in range(F):
        for j in range(i + 1, F):
            p = float(pearson[i, j]) if pearson.size else float("nan")
            s = float(spearman[i, j]) if spearman.size else float("nan")
            m = np.nanmax([abs(p), abs(s)])
            rows.append((m, {"feature_i": feature_names[i], "feature_j": feature_names[j], "i": i, "j": j, "pearson": p, "spearman": s, "abs_pearson": abs(p) if np.isfinite(p) else np.nan, "abs_spearman": abs(s) if np.isfinite(s) else np.nan, "max_abs_corr": float(m)}))
    rows.sort(key=lambda x: (-np.nan_to_num(x[0], nan=-1.0), x[1]["feature_i"], x[1]["feature_j"]))
    return pd.DataFrame([r for _, r in rows[:limit]])


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _has_token(n: str, token: str) -> bool:
    return re.search(rf"(^|_){re.escape(token)}($|_)", n) is not None


def _starts_with_any(n: str, prefixes: Sequence[str]) -> bool:
    return any(n.startswith(p) for p in prefixes)


def _contains_any(n: str, parts: Sequence[str]) -> bool:
    return any(p in n for p in parts)


def _matches_any(n: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, n) is not None for p in patterns)


def parse_feature_name(name: str) -> dict:
    n = name.lower()
    family = "unknown"

    # 1. Calendar/context.
    if (
        _has_token(n, "utc_hour")
        or _has_token(n, "utc_dow")
        or _has_token(n, "hour")
        or _has_token(n, "dow")
        or _has_token(n, "minute")
        or _has_token(n, "calendar")
        or _has_token(n, "tod")
        or n in {
            "utc_hour_sin",
            "utc_hour_cos",
            "utc_dow_sin",
            "utc_dow_cos",
            "is_weekend",
        }
    ):
        family = "calendar"

    # 2. Volatility / volatility asymmetry.
    elif (
        "down_up_vol_imbalance" in n
        or "volatility" in n
        or "vol_regime" in n
        or ("realized_vol" in n and "ofi" not in n)
        or "regime_volume" in n
        or "regime_flow" in n
        or "rv_" in n
    ):
        family = "volatility_regime"

    # 3. CVD before notional/USD.
    elif (
        "cvd" in n
        or "cumulative_volume_delta" in n
    ):
        family = "cvd"

    # 4. Trade toxicity / trade imbalance / trade activity before generic notional.
    elif "trade_toxicity" in n or "toxicity" in n:
        family = "trade_toxicity"

    elif (
        "trade_imbalance" in n
        or ("trade" in n and "imb" in n)
        or ("buy" in n and "sell" in n and "trade" in n)
        or "zero_tick_fraction" in n
        or "tick_sign_imbalance" in n
        or "last_tick_sign" in n
    ):
        family = "trade_imbalance"

    elif (
        "time_since_trade" in n
        or "time_since_last_buy_trade" in n
        or "time_since_last_sell_trade" in n
        or "trade_count" in n
        or ("trade" in n and "count" in n)
        or "trade_rate" in n
        or "last_trade_side_sign" in n
        or "last_is_zero_tick" in n
        or "last_trade_notional" in n
        or "top5_trade_notional" in n
    ):
        family = "trade_activity"

    # 5. Signed flow.
    elif (
        "signed_notional_flow" in n
        or ("signed" in n and "notional" in n and "flow" in n)
        or "max_signed_trade_notional" in n
        or "signed_trade_premium" in n
        or ("signed" in n and "flow" in n)
    ):
        family = "signed_notional_flow"

    # 6. OFI variants.
    elif "ofi_pressure" in n or ("pressure" in n and "ofi" in n):
        family = "ofi_pressure"

    elif (
        "ofi_norm" in n
        or "normalized_ofi" in n
        or ("ofi" in n and "norm" in n)
    ):
        family = "ofi_normalized"

    elif "ofi" in n:
        family = "ofi_raw"

    # 7. Absorption / depletion / no-price-move.
    elif (
        "absorption" in n
        or "no_price_move" in n
        or "depletion" in n
        or "l1_depletion" in n
        or "depletion_over_depth" in n
        or "flow_without_price" in n
    ):
        family = "absorption"

    # 8. Returns before top_book, because mid_ret/micro_ret contain mid/micro.
    elif (
        "return" in n
        or "ret_bps" in n
        or "_ret_" in n
        or n.startswith("ret_")
        or "logret" in n
        or "mid_ret_bps" in n
        or "micro_ret_bps" in n
        or "mid_slope_bps_per_sec" in n
        or "mid_range_bps" in n
    ):
        family = "returns"

    # 9. Micro/VAMP.
    elif "micro" in n or "vamp" in n:
        family = "micro_vamp"

    # 10. Depth imbalance before generic depth.
    elif (
        "depth_imbalance" in n
        or ("depth" in n and "imb" in n)
    ):
        family = "depth_imbalance"

    # 11. OBI / rolling OBI.
    elif (
        "rolling_obi" in n
        or ("obi" in n and _contains_any(n, ("roll", "ema", "mean", "slope", "persist")))
    ):
        family = "rolling_obi"

    elif re.search(r"(^|_)obi(_|$)", n) or "order_book_imbalance" in n:
        family = "obi"

    # 12. Spread / gap.
    elif (
        "spread_depth" in n
        or ("spread" in n and "depth" in n and "regime" in n)
    ):
        family = "spread_depth_regime"

    elif (
        ("spread" in n and "spread_change_count" not in n)
        or n in {"gap_a_bps", "gap_b_bps"}
        or re.search(r"(^|_)gap_[ab]_bps($|_)", n) is not None
    ):
        family = "spread_gap"

    # 13. Book dynamics / top book.
    elif (
        "ob_update_rate" in n
        or "book_update_rate" in n
    ):
        family = "event_density_aux"

    elif (
        "bid_price_change_rate" in n
        or "ask_price_change_rate" in n
        or "price_change_rate" in n
        or "spread_change_count" in n
        or "time_since_mid_change" in n
    ):
        family = "book_dynamics"

    elif (
        n in {"bsz1", "asz1", "bid1", "ask1"}
        or "vwap_vs_mid" in n
        or _starts_with_any(n, ("bsz", "asz"))
        or _contains_any(n, ("best_bid", "best_ask", "top_book", "bbo"))
    ):
        family = "top_book"

    # 14. Notional context after trade/CVD/signed-flow.
    elif (
        (_contains_any(n, ("notional", "usd", "quote_qty")))
        and "flow" not in n
        and "cvd" not in n
        and "trade" not in n
    ):
        family = "notional_context"

    # 15. Generic depth.
    elif "depth" in n:
        family = "depth"

    # 16. Event density / aux.
    elif "event_density" in n or "event_rate" in n:
        family = "event_density_aux"

    elif n.startswith("aux") or "_aux" in n:
        family = "aux"

    timescales_ms: List[int] = []
    for m in re.finditer(r"_(\d+)_minus_(\d+)ms\b", n):
        timescales_ms.extend([int(m.group(1)), int(m.group(2))])
    for m in re.finditer(r"(\d+)ms\b", n):
        timescales_ms.append(int(m.group(1)))
    for m in re.finditer(r"(?:^|_)(\d+)(?:s|sec)\b", n):
        timescales_ms.append(int(m.group(1)) * 1000)
    dedup = []
    for t in timescales_ms:
        if t not in dedup:
            dedup.append(t)
    timescales_ms = dedup
    timescale = max(timescales_ms) if timescales_ms else None

    level = None
    m = re.search(r"(?:level|lvl|l)(\d+)", n)
    if m:
        level = int(m.group(1))
    band_bps = None
    m = re.search(r"(\d+)bps", n)
    if m:
        band_bps = int(m.group(1))
    side = "bid" if "bid" in n else ("ask" if "ask" in n else ("buy" if "buy" in n else ("sell" if "sell" in n else "")))
    is_relative = any(k in n for k in ("_minus_", "_over_", "_imbalance", "_slope", "_accel", "ratio"))
    return {"family": family, "timescale_ms": timescale, "timescales_ms": json.dumps(timescales_ms), "level": level, "band_bps": band_bps, "side": side, "is_relative": is_relative}

def compute_scores(target_df: pd.DataFrame, health_df: pd.DataFrame, feature_names: List[str], week_id: np.ndarray, X: np.ndarray, Y: np.ndarray, horizons_ms: List[int], masks_by_horizon: Dict[int, Dict[str, np.ndarray]]) -> pd.DataFrame:
    kept = target_df[target_df["mask_type"] == "kept"].copy()
    score_rows = []
    health_by_feature = health_df.set_index("feature").to_dict("index")
    for j, name in enumerate(feature_names):
        ft = kept[kept["feature"] == name]
        target_score = 0.0
        max_abs_spear_signed = 0.0
        max_abs_spear_abs = 0.0
        max_auc_edge = 0.0
        best_auc = np.nan
        best_abs_spear = np.nan
        for _, r in ft.iterrows():
            ss = abs(float(r.get("spearman_signed_return", np.nan))) if np.isfinite(r.get("spearman_signed_return", np.nan)) else 0.0
            sa = abs(float(r.get("spearman_abs_return", np.nan))) if np.isfinite(r.get("spearman_abs_return", np.nan)) else 0.0
            auc = float(r.get("single_feature_auc_direction", np.nan))
            auc_edge = abs(auc - 0.5) if np.isfinite(auc) else 0.0
            score = 0.50 * ss + 0.30 * auc_edge * 2.0 + 0.20 * sa
            target_score = max(target_score, float(score))
            max_abs_spear_signed = max(max_abs_spear_signed, ss)
            max_abs_spear_abs = max(max_abs_spear_abs, sa)
            max_auc_edge = max(max_auc_edge, auc_edge)
            best_auc = np.nanmax([best_auc, auc])
            best_abs_spear = np.nanmax([best_abs_spear, sa])

        # Stability: sign consistency of per-week signed Spearman for the best horizon-ish aggregate.
        week_metrics = []
        for h in range(len(horizons_ms)):
            mask = masks_by_horizon[h]["kept"]
            for wid in np.unique(week_id):
                m = mask & (week_id == wid)
                if int(m.sum()) >= 20:
                    week_metrics.append(safe_spearman(X[m, j], Y[m, h]))
        finite_metrics = np.asarray([v for v in week_metrics if np.isfinite(v)], dtype=np.float64)
        if finite_metrics.size:
            dominant = 1.0 if np.nanmean(finite_metrics) >= 0 else -1.0
            sign_consistency = float(np.mean(np.sign(finite_metrics + EPS) == dominant))
            week_metric_std = float(np.std(finite_metrics))
            stability_score = float(sign_consistency * (1.0 / (1.0 + week_metric_std)))
            mean_week_metric = float(np.mean(finite_metrics))
        else:
            sign_consistency = 0.0
            week_metric_std = float("nan")
            stability_score = 0.0
            mean_week_metric = float("nan")

        hrow = health_by_feature[name]
        health_score = 1.0
        if float(hrow["finite_frac"]) < 0.999:
            health_score -= 0.4
        if not np.isfinite(float(hrow["std"])) or float(hrow["std"]) < 1e-4:
            health_score -= 0.35
        if np.isfinite(float(hrow["zero_frac"])) and float(hrow["zero_frac"]) > 0.99:
            health_score -= 0.25
        health_score = float(np.clip(health_score, 0.0, 1.0))
        parsed = parse_feature_name(name)
        semantic_priority = SEMANTIC_PRIORITY.get(parsed["family"], 0.50)
        keep_score = 0.55 * target_score + 0.25 * stability_score + 0.10 * semantic_priority + 0.10 * health_score
        score_rows.append({
            "feature": name,
            "feature_index": j,
            **parsed,
            "target_score": target_score,
            "stability_score": stability_score,
            "semantic_priority": semantic_priority,
            "health_score": health_score,
            "keep_score": keep_score,
            "max_abs_spearman_signed_kept": max_abs_spear_signed,
            "max_abs_spearman_abs_kept": max_abs_spear_abs,
            "max_auc_edge_kept": max_auc_edge,
            "best_horizon_auc": best_auc,
            "best_horizon_abs_spearman": best_abs_spear,
            "sign_consistency": sign_consistency,
            "week_metric_std": week_metric_std,
            "mean_week_metric": mean_week_metric,
        })
    return pd.DataFrame(score_rows)


def pair_decision(i: int, j: int, p: float, s: float, scores: pd.DataFrame) -> dict:
    ri = scores.iloc[i]
    rj = scores.iloc[j]
    si = float(ri["keep_score"])
    sj = float(rj["keep_score"])
    reason = "keep_score_gap"
    confidence = abs(si - sj)
    if si >= sj + 0.05:
        winner, loser = ri["feature"], rj["feature"]
    elif sj >= si + 0.05:
        winner, loser = rj["feature"], ri["feature"]
    else:
        # Tie-breaking heuristics.
        fi, fj = str(ri["family"]), str(rj["family"])
        ti = ri.get("timescale_ms")
        tj = rj.get("timescale_ms")
        tsi = 0 if pd.isna(ti) else int(ti)
        tsj = 0 if pd.isna(tj) else int(tj)
        if fi == fj and abs(float(ri["target_score"]) - float(rj["target_score"])) < 0.01:
            if bool(ri["is_relative"]) != bool(rj["is_relative"]):
                winner, loser = (rj["feature"], ri["feature"]) if bool(ri["is_relative"]) else (ri["feature"], rj["feature"])
                reason = "same_family_similar_target_prefer_base"
            elif fi in SLOW_FAMILIES and tsi != tsj:
                winner, loser = (ri["feature"], rj["feature"]) if tsi > tsj else (rj["feature"], ri["feature"])
                reason = "slow_family_prefer_longer_timescale"
            elif fi in FAST_FAMILIES and tsi != tsj:
                winner, loser = (ri["feature"], rj["feature"]) if tsi < tsj else (rj["feature"], ri["feature"])
                reason = "fast_family_prefer_shorter_timescale"
            else:
                winner = loser = "needs_ablation"
                reason = "scores_close"
        else:
            winner = loser = "needs_ablation"
            reason = "scores_close"
    return {"feature_i": ri["feature"], "feature_j": rj["feature"], "corr_pearson": p, "corr_spearman": s, "winner": winner, "loser": loser, "confidence": confidence, "reason": reason}


def build_clusters_and_pairs(pearson: np.ndarray, spearman: np.ndarray, scores: pd.DataFrame, high: float) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Dict[str, int], Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, str]]:
    F = scores.shape[0]
    uf = UnionFind(F)
    pair_rows = []
    better_feature: Dict[str, str] = {str(f): "" for f in scores["feature"]}
    max_corr_with_better: Dict[str, float] = {str(f): np.nan for f in scores["feature"]}
    for i in range(F):
        for j in range(i + 1, F):
            p = float(pearson[i, j])
            s = float(spearman[i, j])
            m = float(np.nanmax([abs(p), abs(s)]))
            if np.isfinite(m) and m >= high:
                uf.union(i, j)
                dec = pair_decision(i, j, p, s, scores)
                pair_rows.append(dec)
                if dec["winner"] not in ("needs_ablation", "") and dec["loser"] not in ("needs_ablation", ""):
                    loser = str(dec["loser"])
                    if not np.isfinite(max_corr_with_better.get(loser, np.nan)) or m > float(max_corr_with_better[loser]):
                        max_corr_with_better[loser] = m
                        better_feature[loser] = str(dec["winner"])

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(F):
        groups[uf.find(i)].append(i)
    cluster_id_by_feature: Dict[str, int] = {}
    cluster_representative_by_feature: Dict[str, str] = {}
    cluster_nonrep_reason_by_feature: Dict[str, str] = {}
    cluster_rows = []
    summary_clusters = []
    cid = 0
    for _, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0])):
        if len(members) < 2:
            for i in members:
                feature = str(scores.iloc[i]["feature"])
                cluster_id_by_feature[feature] = -1
                cluster_representative_by_feature[feature] = ""
                cluster_nonrep_reason_by_feature[feature] = ""
            continue
        sub_scores = scores.iloc[members]
        rep_idx = int(sub_scores["keep_score"].astype(float).idxmax())
        rep_feature = str(scores.loc[rep_idx, "feature"])
        max_pair_corr = 0.0
        for a_pos, i in enumerate(members):
            for j in members[a_pos + 1:]:
                max_pair_corr = max(max_pair_corr, float(np.nanmax([abs(pearson[i, j]), abs(spearman[i, j])])) )
        feature_list = [str(scores.iloc[i]["feature"]) for i in members]
        for i in members:
            r = scores.iloc[i]
            feature = str(r["feature"])
            cluster_id_by_feature[feature] = cid
            cluster_representative_by_feature[feature] = rep_feature
            is_rep = feature == rep_feature
            cluster_nonrep_reason_by_feature[feature] = "" if is_rep else "high_corr_cluster_nonrepresentative"
            if is_rep:
                rec_action = "keep"
                rec_reason = "cluster_representative"
            elif better_feature.get(feature):
                rec_action = "remove_low_risk"
                rec_reason = "high_corr_with_clear_better_feature"
            else:
                rec_action = "remove_if_budget_tight"
                rec_reason = "high_corr_cluster_nonrepresentative_scores_close"
            cluster_rows.append({
                "cluster_id": cid,
                "cluster_size": len(members),
                "feature": feature,
                "is_representative": bool(is_rep),
                "representative_feature": rep_feature,
                "family": r["family"],
                "timescale_ms": r["timescale_ms"],
                "best_horizon_auc": r["best_horizon_auc"],
                "best_horizon_abs_spearman": r["best_horizon_abs_spearman"],
                "stability_score": r["stability_score"],
                "recommended_action": rec_action,
                "reason": rec_reason,
            })
        summary_clusters.append({"cluster_id": cid, "size": len(members), "representative": rep_feature, "features": feature_list, "max_pair_corr": max_pair_corr})
        cid += 1
    summary = {"n_clusters": cid, "clusters": summary_clusters}
    return (
        pd.DataFrame(cluster_rows),
        pd.DataFrame(pair_rows),
        summary,
        cluster_id_by_feature,
        better_feature,
        max_corr_with_better,
        cluster_representative_by_feature,
        cluster_nonrep_reason_by_feature,
    )


def generate_removal_candidates(
    scores: pd.DataFrame,
    health_df: pd.DataFrame,
    cluster_id_by_feature: Dict[str, int],
    better_feature: Dict[str, str],
    max_corr_better: Dict[str, float],
    scores_by_feature: Dict[str, dict],
    cluster_representative_by_feature: Dict[str, str],
    cluster_nonrep_reason_by_feature: Dict[str, str],
) -> pd.DataFrame:
    health = health_df.set_index("feature").to_dict("index")
    rows = []
    for _, r in scores.iterrows():
        feature = str(r["feature"])
        h = health[feature]
        health_bad = float(h["finite_frac"]) < 0.999 or float(h["std"]) < 1e-4 or float(h["abs_max"]) == 0.0
        low_target = float(r["max_abs_spearman_signed_kept"]) < 0.005 and float(r["max_auc_edge_kept"]) < 0.005 and float(r["max_abs_spearman_abs_kept"]) < 0.005
        unstable = float(r["sign_consistency"]) < 0.67 or (np.isfinite(float(r["week_metric_std"])) and float(r["week_metric_std"]) > 2 * abs(float(r["mean_week_metric"])) + 1e-6)
        bf = better_feature.get(feature, "")
        mc = max_corr_better.get(feature, np.nan)
        rep_target = scores_by_feature.get(bf, {}).get("target_score", np.nan) if bf else np.nan
        redundant_low_risk = bool(bf) and np.isfinite(mc) and mc >= AUDIT_HIGH_CORR and (scores_by_feature[bf]["keep_score"] - float(r["keep_score"])) >= 0.05 and float(r["target_score"]) <= float(rep_target) * 1.05
        cluster_id = int(cluster_id_by_feature.get(feature, -1))
        cluster_rep = cluster_representative_by_feature.get(feature, "")
        is_cluster_nonrep = cluster_id != -1 and bool(cluster_rep) and cluster_rep != feature
        no_unique_cluster_role = cluster_id == -1
        if health_bad:
            action = "remove_low_risk"
            reason = "health_bad"
        elif redundant_low_risk:
            action = "remove_low_risk"
            reason = f"redundant_with_better_feature:{bf}"
        elif is_cluster_nonrep and not bf:
            action = "remove_if_budget_tight"
            reason = f"high_corr_cluster_nonrepresentative_scores_close:{cluster_rep}"
        elif low_target and unstable and no_unique_cluster_role:
            action = "remove_low_risk"
            reason = "low_target_unstable_no_unique_cluster_role"
        elif low_target and not bf:
            action = "needs_ablation"
            reason = "low_univariate_target_not_redundant"
        elif bf and np.isfinite(mc) and mc >= AUDIT_MED_CORR:
            action = "remove_if_budget_tight"
            reason = f"medium_high_corr_with:{bf}"
        else:
            action = "keep"
            reason = "not_low_risk_removal"
        rows.append({
            "feature": feature,
            "family": r["family"],
            "reason": reason,
            "recommended_action": action,
            "keep_score": r["keep_score"],
            "target_score": r["target_score"],
            "stability_score": r["stability_score"],
            "max_corr_with_better_feature": mc,
            "better_feature": bf,
            "cluster_id": cluster_id,
            "cluster_representative_feature": cluster_rep,
            "cluster_nonrep_reason": cluster_nonrep_reason_by_feature.get(feature, ""),
            "timescale_ms": r["timescale_ms"],
            "is_relative": r["is_relative"],
            "health_bad": health_bad,
            "low_target_score": low_target,
            "unstable": unstable,
        })
    df = pd.DataFrame(rows)
    order = {"remove_low_risk": 0, "remove_if_budget_tight": 1, "needs_ablation": 2, "keep": 3}
    df["_ord"] = df["recommended_action"].map(order).fillna(9)
    return df.sort_values(["_ord", "keep_score", "target_score"], ascending=[True, True, True]).drop(columns=["_ord"])


def component_guess(name: str) -> str:
    if "_minus_" in name:
        return json.dumps(name.split("_minus_", 1))
    if "_over_" in name:
        return json.dumps(name.split("_over_", 1))
    lower = name.lower()
    if "slope" in lower or "accel" in lower:
        return "temporal_derivative"
    if "ratio" in lower or "imbalance" in lower:
        return "relative_ratio_or_imbalance"
    return ""


def find_component_candidates(component: str, feature_names: Sequence[str], limit: int = 5) -> List[str]:
    c = component.lower()
    out = []
    for f in feature_names:
        fl = f.lower()
        if fl == c or fl.endswith(c) or c.endswith(fl) or c in fl:
            out.append(f)
            if len(out) >= limit:
                break
    return out


def _best_reported_corr_with_candidates(feature: str, candidates: Sequence[str], corr_pairs: pd.DataFrame) -> Tuple[float, str]:
    if corr_pairs.empty or not candidates:
        return np.nan, ""
    candidate_set = set(candidates)
    sub = corr_pairs[
        ((corr_pairs["feature_i"] == feature) & corr_pairs["feature_j"].isin(candidate_set))
        | ((corr_pairs["feature_j"] == feature) & corr_pairs["feature_i"].isin(candidate_set))
    ]
    if sub.empty:
        return np.nan, ""
    best = sub.sort_values("max_abs_corr", ascending=False).iloc[0]
    other = str(best["feature_j"] if best["feature_i"] == feature else best["feature_i"])
    return float(best["max_abs_corr"]), other


def relative_report(scores: pd.DataFrame, corr_pairs: pd.DataFrame, removal_df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    action_by_feature = removal_df.set_index("feature").to_dict("index")
    feature_set = set(feature_names)
    rows = []
    for _, r in scores[scores["is_relative"] == True].iterrows():  # noqa: E712
        feature = str(r["feature"])
        sub = corr_pairs[(corr_pairs["feature_i"] == feature) | (corr_pairs["feature_j"] == feature)] if not corr_pairs.empty else pd.DataFrame()
        if sub.empty:
            mc = np.nan
            mf = ""
        else:
            best = sub.sort_values("max_abs_corr", ascending=False).iloc[0]
            mc = float(best["max_abs_corr"])
            mf = str(best["feature_j"] if best["feature_i"] == feature else best["feature_i"])

        component_candidates: List[str] = []
        if "_minus_" in feature or "_over_" in feature:
            sep = "_minus_" if "_minus_" in feature else "_over_"
            left_guess, right_guess = feature.split(sep, 1)
            if right_guess not in feature_set:
                right_guess = re.sub(r"_(\d+)ms$", "", right_guess)
            for component in (left_guess, right_guess):
                if component in feature_set:
                    candidates = [component]
                else:
                    candidates = find_component_candidates(component, feature_names)
                for candidate in candidates:
                    if candidate != feature and candidate not in component_candidates:
                        component_candidates.append(candidate)
        component_mc, component_mf = _best_reported_corr_with_candidates(feature, component_candidates, corr_pairs)

        act = action_by_feature.get(feature, {})
        rows.append({
            "feature": feature,
            "family": r["family"],
            "timescale_ms": r["timescale_ms"],
            "is_relative": r["is_relative"],
            "components_guess": component_guess(feature),
            "max_corr_with_any_reported_feature": mc,
            "most_correlated_reported_feature": mf,
            "component_candidates_json": json.dumps(component_candidates),
            "max_corr_with_component_candidate": component_mc,
            "most_correlated_component_candidate": component_mf,
            "target_score": r["target_score"],
            "keep_score": r["keep_score"],
            "recommended_action": act.get("recommended_action", ""),
            "reason": act.get("reason", ""),
        })
    return pd.DataFrame(rows)


def family_summary(scores: pd.DataFrame, target_df: pd.DataFrame, corr_pairs: pd.DataFrame, removal_df: pd.DataFrame, horizons_ms: List[int]) -> pd.DataFrame:
    action = removal_df.set_index("feature")["recommended_action"].to_dict()
    feature_to_family = scores.set_index("feature")["family"].to_dict()
    target_df2 = target_df.copy()
    target_df2["family"] = target_df2["feature"].map(feature_to_family)
    kept = target_df2[target_df2["mask_type"] == "kept"]
    rows = []
    for fam, grp in scores.groupby("family", dropna=False):
        features = set(str(x) for x in grp["feature"])
        within = corr_pairs[corr_pairs["feature_i"].isin(features) & corr_pairs["feature_j"].isin(features)] if not corr_pairs.empty else pd.DataFrame()
        mean_corr = float(within["max_abs_corr"].mean()) if not within.empty else np.nan
        max_corr = float(within["max_abs_corr"].max()) if not within.empty else np.nan
        row = {
            "family": fam,
            "n_features": int(len(grp)),
            "n_relative": int(grp["is_relative"].sum()),
            "mean_abs_corr_within_family": mean_corr,
            "max_abs_corr_within_family": max_corr,
            "best_auc_200": np.nan,
            "best_auc_500": np.nan,
            "best_auc_1000": np.nan,
            "n_low_risk_remove": int(sum(action.get(f) == "remove_low_risk" for f in features)),
            "n_remove_if_budget_tight": int(sum(action.get(f) == "remove_if_budget_tight" for f in features)),
            "n_keep": int(sum(action.get(f) == "keep" for f in features)),
            "n_needs_ablation": int(sum(action.get(f) == "needs_ablation" for f in features)),
        }
        for h in (200, 500, 1000):
            if h in horizons_ms:
                vals = kept[(kept["family"] == fam) & (kept["horizon_ms"] == h)]["single_feature_auc_direction"]
                row[f"best_auc_{h}"] = float(vals.max()) if len(vals) else np.nan
            else:
                row[f"best_auc_{h}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("family")



def _self_test_parsers() -> None:
    assert parse_feature_name("down_up_vol_imbalance_1000ms")["family"] == "volatility_regime"
    assert parse_feature_name("utc_dow_sin")["family"] == "calendar"

    assert parse_feature_name("micro_l1_minus_micro_l10_bps")["family"] == "micro_vamp"
    assert parse_feature_name("micro_minus_mid_bps")["family"] == "micro_vamp"
    assert parse_feature_name("micro_ret_bps_500ms")["family"] == "returns"

    assert parse_feature_name("mid_ret_bps_1000ms")["family"] == "returns"

    assert parse_feature_name("cvd_change_usd_500ms")["family"] == "cvd"
    assert parse_feature_name("cvd_slope_usd_per_sec_1000ms")["family"] == "cvd"
    assert parse_feature_name("cvd_minus_ema_usd_3000ms")["family"] == "cvd"

    assert parse_feature_name("trade_imbalance_notional_1000ms")["family"] == "trade_imbalance"
    assert parse_feature_name("trade_toxicity_notional_1000ms")["family"] == "trade_toxicity"
    assert parse_feature_name("time_since_trade_ms")["family"] == "trade_activity"

    assert parse_feature_name("gap_a_bps")["family"] == "spread_gap"
    assert parse_feature_name("gap_b_bps")["family"] == "spread_gap"

    assert parse_feature_name("bsz1")["family"] == "top_book"
    assert parse_feature_name("asz1")["family"] == "top_book"

    assert parse_feature_name("ob_update_rate_500ms")["family"] == "event_density_aux"
    assert parse_feature_name("bid_l1_depletion_200ms")["family"] == "absorption"
    assert parse_feature_name("ask_l1_depletion_over_depth_200ms")["family"] == "absorption"
    assert parse_feature_name("bid_price_change_rate_200ms")["family"] == "book_dynamics"
    assert parse_feature_name("spread_change_count_200ms")["family"] == "book_dynamics"

    p = parse_feature_name("ofi_l3_accel_200_minus_500ms")
    assert p["timescale_ms"] == 500
    assert json.loads(p["timescales_ms"]) == [200, 500]

    p = parse_feature_name("micro_ret_bps_1000ms")
    assert p["timescale_ms"] == 1000
    assert json.loads(p["timescales_ms"]) == [1000]


def maybe_check_current_feature_parser_coverage(feature_names: Sequence[str], warnings_out: List[str]) -> None:
    parsed = [parse_feature_name(f) for f in feature_names]
    unknown = [f for f, p in zip(feature_names, parsed) if p["family"] == "unknown"]
    if unknown:
        warnings_out.append(f"unknown parser features sample: {unknown[:20]}")

def validate_meta(global_meta: dict) -> Tuple[int, int, int, List[str], List[str], List[int]]:
    feature_dim_core = int(global_meta["feature_dim_core"])
    feature_dim_total = int(global_meta["feature_dim_total"])
    aux_dim = int(global_meta["aux_dim"])
    feature_names = list(global_meta["feature_names"])
    aux_names = list(global_meta.get("aux_names", []))
    horizons_ms = [int(x) for x in global_meta["horizons_ms"]]
    if feature_dim_core != len(feature_names):
        raise AssertionError(f"feature_dim_core={feature_dim_core} != len(feature_names)={len(feature_names)}")
    if feature_dim_total != feature_dim_core + aux_dim:
        raise AssertionError("feature_dim_total must equal feature_dim_core + aux_dim")
    if len(horizons_ms) != int(global_meta["label_dim"]):
        raise AssertionError("len(horizons_ms) must equal label_dim")
    if aux_dim != len(aux_names):
        raise AssertionError(f"aux_dim={aux_dim} != len(aux_names)={len(aux_names)}")
    if not isinstance(global_meta.get("weeks_meta"), dict) or not global_meta.get("weeks_meta"):
        raise KeyError("meta.json missing non-empty weeks_meta")
    return feature_dim_core, feature_dim_total, aux_dim, feature_names, aux_names, horizons_ms


def main() -> None:
    if not OUT_ROOT:
        raise SystemExit("Set BYBIT_OUT_ROOT to the root created by offline_ingest.py")
    out_root = Path(OUT_ROOT).expanduser().resolve()
    if not out_root.exists():
        raise SystemExit(f"BYBIT_OUT_ROOT does not exist: {out_root}")
    out_dir = Path(AUDIT_OUT_DIR).expanduser().resolve() if AUDIT_OUT_DIR else out_root / "feature_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings_out: List[str] = []

    print(f"[audit-config] out_root={out_root} out_dir={out_dir} seed={AUDIT_SEED} max_rows_total={AUDIT_MAX_ROWS_TOTAL} max_rows_per_week={AUDIT_MAX_ROWS_PER_WEEK} mi_max_rows={AUDIT_MI_MAX_ROWS} use_aux={int(AUDIT_USE_AUX)} high_corr={AUDIT_HIGH_CORR}", flush=True)
    meta_path = out_root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing meta.json: {meta_path}")
    global_meta = read_json(meta_path)
    feature_dim_core, feature_dim_total, aux_dim, feature_names, aux_names, horizons_ms = validate_meta(global_meta)
    _self_test_parsers()
    train_weeks = list(global_meta["splits"]["cmssl"]["train"]["weeks"])
    if AUDIT_USE_AUX:
        audit_feature_names = feature_names + aux_names
        audit_feature_indices = np.arange(feature_dim_total, dtype=np.int64)
    else:
        audit_feature_names = feature_names
        audit_feature_indices = np.arange(feature_dim_core, dtype=np.int64)
    maybe_check_current_feature_parser_coverage(audit_feature_names, warnings_out)
    print(f"[audit-meta] schema={global_meta.get('feature_schema', '')} transform={global_meta.get('feature_transform', '')} feature_dim_core={feature_dim_core} feature_dim_total={feature_dim_total} train_weeks={train_weeks}", flush=True)

    X, Y, row_idx, label_ts, week_id, week_key, sample_rows_by_week = load_sampled_dataset(out_root, global_meta, audit_feature_indices, warnings_out)
    print(f"[audit-load] X shape={X.shape} Y shape={Y.shape}", flush=True)

    low_frac = float(global_meta.get("low_abs_trim_fraction", 0.0))
    high_frac = float(global_meta.get("high_abs_trim_fraction", 0.0))
    masks_by_horizon: Dict[int, Dict[str, np.ndarray]] = {}
    for h in range(Y.shape[1]):
        y_h = Y[:, h]
        finite = np.isfinite(y_h)
        nonzero = finite & (np.abs(y_h) > AUDIT_MIN_ABS_LABEL_EPS)
        kept = side_specific_keep_mask(y_h, low_frac, high_frac)
        masks_by_horizon[h] = {"all_finite": finite, "nonzero": nonzero, "kept": kept}

    health_df = compute_health(X, audit_feature_names, week_id, len(train_weeks))
    health_path = out_dir / "feature_health.csv"
    health_df.to_csv(health_path, index=False)
    print(f"[audit-health] wrote {health_path}", flush=True)
    if int((health_df["finite_frac"] < 0.999).sum()) > 0:
        warnings_out.append(f"features with finite_frac < 0.999: {int((health_df['finite_frac'] < 0.999).sum())}")

    target_df = compute_target_metrics(X, Y, audit_feature_names, horizons_ms, masks_by_horizon, warnings_out)
    target_path = out_dir / "feature_target_metrics.csv"
    target_df.to_csv(target_path, index=False)
    print(f"[audit-target] wrote {target_path}", flush=True)

    methods = {m.strip().lower() for m in AUDIT_CORR_METHODS.split(",") if m.strip()}
    pearson = corrcoef_pairwise_complete(X) if "pearson" in methods else np.eye(len(audit_feature_names))
    spearman = spearman_corr_matrix(X) if "spearman" in methods else np.eye(len(audit_feature_names))
    corr_pairs_df = top_corr_pairs(pearson, spearman, audit_feature_names, AUDIT_TOP_CORR_PAIRS)
    corr_path = out_dir / "feature_corr_top_pairs.csv"
    corr_pairs_df.to_csv(corr_path, index=False)
    print(f"[audit-corr] wrote {corr_path}", flush=True)

    scores_df = compute_scores(target_df, health_df, audit_feature_names, week_id, X, Y, horizons_ms, masks_by_horizon)
    parser_rows = []
    for _, r in scores_df.iterrows():
        parser_rows.append({
            "feature": r["feature"],
            "family": r["family"],
            "timescale_ms": r["timescale_ms"],
            "timescales_ms": r["timescales_ms"],
            "level": r["level"],
            "band_bps": r["band_bps"],
            "side": r["side"],
            "is_relative": r["is_relative"],
        })

    parser_df = pd.DataFrame(parser_rows)
    parser_path = out_dir / "feature_parser_coverage.csv"
    parser_df.to_csv(parser_path, index=False)
    print(f"[audit-parser] wrote {parser_path}", flush=True)

    (
        cluster_df,
        pair_df,
        cluster_summary,
        cluster_id_by_feature,
        better_feature,
        max_corr_better,
        cluster_representative_by_feature,
        cluster_nonrep_reason_by_feature,
    ) = build_clusters_and_pairs(pearson, spearman, scores_df, AUDIT_HIGH_CORR)
    clusters_path = out_dir / "feature_clusters.csv"
    cluster_df.to_csv(clusters_path, index=False)
    write_json(out_dir / "feature_cluster_summary.json", cluster_summary)
    pair_path = out_dir / "feature_pair_comparisons.csv"
    pair_df.to_csv(pair_path, index=False)
    print(f"[audit-clusters] clusters={cluster_summary['n_clusters']} wrote {clusters_path}", flush=True)

    scores_by_feature = scores_df.set_index("feature").to_dict("index")
    removal_df = generate_removal_candidates(
        scores_df,
        health_df,
        cluster_id_by_feature,
        better_feature,
        max_corr_better,
        scores_by_feature,
        cluster_representative_by_feature,
        cluster_nonrep_reason_by_feature,
    )
    removal_path = out_dir / "removal_candidates_low_risk.csv"
    removal_df.head(AUDIT_TOP_REMOVAL_CANDIDATES).to_csv(removal_path, index=False)

    rel_df = relative_report(scores_df, corr_pairs_df, removal_df, audit_feature_names)
    rel_df.to_csv(out_dir / "relative_feature_report.csv", index=False)
    fam_df = family_summary(scores_df, target_df, corr_pairs_df, removal_df, horizons_ms)
    fam_df.to_csv(out_dir / "feature_family_summary.csv", index=False)

    unknown_count = int((scores_df["family"] == "unknown").sum())
    unknown_frac = unknown_count / max(1, len(scores_df))

    if unknown_count:
        warnings_out.append(f"unknown feature families: {unknown_count}")

    if unknown_frac > 0.10:
        warnings_out.append(
            f"WARNING: unknown family fraction too high: {unknown_count}/{len(scores_df)} = {unknown_frac:.3%}; parser coverage should be improved before using recommendations"
        )

    print(f"[audit-parser] unknown={unknown_count} unknown_frac={unknown_frac:.3%}", flush=True)

    n_low = int((removal_df["recommended_action"] == "remove_low_risk").sum())
    n_budget = int((removal_df["recommended_action"] == "remove_if_budget_tight").sum())
    n_ablate = int((removal_df["recommended_action"] == "needs_ablation").sum())
    print(f"[audit-removal] low_risk={n_low} budget_tight={n_budget} needs_ablation={n_ablate}", flush=True)

    n_high_corr_cluster_nonrepresentatives = int(sum(1 for v in cluster_nonrep_reason_by_feature.values() if v))
    nonrep_features = [f for f, v in cluster_nonrep_reason_by_feature.items() if v]
    n_high_corr_cluster_nonrep_remove_if_budget_tight = int(
        removal_df[
            removal_df["feature"].isin(nonrep_features)
            & (removal_df["recommended_action"] == "remove_if_budget_tight")
        ].shape[0]
    )

    summary = {
        "out_root": str(out_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema": global_meta.get("feature_schema"),
        "feature_transform": global_meta.get("feature_transform"),
        "feature_dim_core": feature_dim_core,
        "feature_dim_total": feature_dim_total,
        "audit_use_aux": bool(AUDIT_USE_AUX),
        "train_weeks": train_weeks,
        "sample_rows_total": int(X.shape[0]),
        "sample_rows_by_week": sample_rows_by_week,
        "horizons_ms": horizons_ms,
        "corr_threshold_high": AUDIT_HIGH_CORR,
        "corr_threshold_medium": AUDIT_MED_CORR,
        "n_features": int(len(audit_feature_names)),
        "n_clusters_high_corr": int(cluster_summary["n_clusters"]),
        "n_high_corr_cluster_nonrepresentatives": n_high_corr_cluster_nonrepresentatives,
        "n_high_corr_cluster_nonrep_remove_if_budget_tight": n_high_corr_cluster_nonrep_remove_if_budget_tight,
        "n_parser_unknown_family": unknown_count,
        "parser_unknown_fraction": unknown_frac,
        "n_relative_features": int(scores_df["is_relative"].sum()),
        "n_remove_low_risk": n_low,
        "n_remove_if_budget_tight": n_budget,
        "n_needs_ablation": n_ablate,
        "top_removal_candidates": removal_df[removal_df["recommended_action"] == "remove_low_risk"].head(25).to_dict("records"),
        "known_limitations": [
            "single-feature metrics do not capture nonlinear interactions",
            "mutual information is approximate and sample-dependent",
            "relative feature component matching is heuristic",
            "recommendations are pruning candidates, not final proof",
        ],
        "warnings": sorted(set(warnings_out)),
    }
    write_json(out_dir / "feature_audit_summary.json", summary)
    print(f"[audit-done] out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
