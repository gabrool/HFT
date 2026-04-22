#!/usr/bin/env python3
"""
Decision-time ingest (memory-safe):
- Store one flat decision row per non-trade OB decision timestamp.
- Materialize training windows dynamically at train time from flat rows.
- Use a RAM budget to auto-size chunked writes (avoid huge in-RAM lists).

Input layout support:
- OB: YYYY-MM-DD_BTCUSDT_...ob...*.zip.
- TH: BTCUSDTYYYY-MM-DD.csv.gz, with tolerant handling for .csv / .csv.gzip.

Downstream ingest contract:
- pair_weeks() supports two grouping branches controlled by BYBIT_USE_TRADES:
  - BYBIT_USE_TRADES=1: aligned OB/TH daily files are grouped into consecutive
    7-day blocks.
  - BYBIT_USE_TRADES=0: OB-only daily files are grouped into consecutive 7-day
    blocks and each week emits th_paths=[].
- pair_weeks() emits canonical week keys: DD-MM-YYYY-to-DD-MM-YYYY.
- pair_weeks() and all ingest entry points operate on WeekPair tuples:
  (week_key, ob_paths: List[str], th_paths: List[str]).
- Event streaming is chained per week; daily files are processed in day order
  and timestamp monotonicity is enforced across day boundaries.

Environment variables (read via os.environ.get in this module):
  BYBIT_OB_DIR=/home/gabrool/Documents/OB
  BYBIT_TH_DIR=/home/gabrool/Documents/TH
  BYBIT_OUT_ROOT=/media/gabrool/Expansion/Gabriel/bybit_offline_dt
  BYBIT_WEEKS=""                    # optional comma/space-separated week keys
  BYBIT_PCA_VAR=0.99
  BYBIT_PCA_MAX_ROWS=200000
  BYBIT_PCA_BATCH=4096
  BYBIT_PCA_MODEL=pca_model.npz
  BYBIT_PCA_USE_EXISTING=0
  BYBIT_RAM_BUDGET_MB=512            # memory budget for one chunk
  BYBIT_CHUNK_SIZE=0                 # default auto-size from RAM budget; set a positive integer to force a fixed chunk size

Shared constants from CMSSL17:
  LOOKBACK (and related model/data constants) are defined in CMSSL17.py.
  If these values are intentionally changed, update them in CMSSL17.py.
  Decision timestamps are the actual OB event timestamps (event-time).
"""

import os, sys, csv, json, re, time, logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple, Iterable, Dict, Optional, Any
from collections import deque, defaultdict
import itertools
import numpy as np
from datetime import date, datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ---------------- config ----------------
OB_DIR      = os.environ.get("BYBIT_OB_DIR",   "/home/gabrool/Documents/OB")
TH_DIR      = os.environ.get("BYBIT_TH_DIR",   "/home/gabrool/Documents/TH")
OUT_ROOT    = os.environ.get("BYBIT_OUT_ROOT", "/media/gabrool/Expansion/Gabriel/bybit_offline_dt")

# Week selection: use discovered week pairs; optionally restrict with BYBIT_WEEKS.
RAW_BYBIT_WEEKS = os.environ.get("BYBIT_WEEKS", "")

# Optional PCA dimensionality reduction on the core features
PCA_VAR_TARGET      = float(os.environ.get("BYBIT_PCA_VAR", "0.99"))
PCA_MAX_SAMPLE_ROWS = int(os.environ.get("BYBIT_PCA_MAX_ROWS", "200000"))
PCA_BATCH_SIZE      = int(os.environ.get("BYBIT_PCA_BATCH", "4096"))
PCA_MODEL_FILENAME  = os.environ.get("BYBIT_PCA_MODEL", "pca_model.npz")
PCA_USE_EXISTING    = int(os.environ.get("BYBIT_PCA_USE_EXISTING", "0"))

# Memory & chunking
RAM_BUDGET  = int(os.environ.get("BYBIT_RAM_BUDGET_MB", "512"))
CHUNK_SIZE  = int(os.environ.get("BYBIT_CHUNK_SIZE", "0"))  # 0 = auto-size from RAM budget; >0 = explicit fixed override
FLUSH_WORKERS = int(os.environ.get("BYBIT_FLUSH_WORKERS", "4"))
DECISION_POLICY = "ob_event_time"




OB_TP_SNAPSHOT = 1
OB_TP_DELTA = 2
TRADE_SIDE_BUY = 1
TRADE_SIDE_SELL = -1
TRADE_SIDE_UNKNOWN = 0
TRADE_TICK_PLUS = 1
TRADE_TICK_MINUS = -1
TRADE_TICK_ZERO = 0


def _compact_ob_type_code(tp_raw: Any) -> int:
    tp_norm = str(tp_raw or "delta").strip().lower()
    return OB_TP_SNAPSHOT if tp_norm == "snapshot" else OB_TP_DELTA


def _compact_trade_side_code(side_raw: Any) -> int:
    side_norm = str(side_raw or "").strip().lower()
    if side_norm == "buy":
        return TRADE_SIDE_BUY
    if side_norm == "sell":
        return TRADE_SIDE_SELL
    return TRADE_SIDE_UNKNOWN


def _compact_tick_dir_code(tick_raw: Any) -> int:
    norm = str(tick_raw or "").strip().lower()
    cleaned = norm.replace("-", "").replace("_", "").replace(" ", "")
    if "plus" in cleaned or cleaned in {"plustick", "uptick", "up", "buy", "bid", "+", "1"}:
        return TRADE_TICK_PLUS
    if "minus" in cleaned or cleaned in {"minustick", "downtick", "down", "sell", "ask", "-", "-1"}:
        return TRADE_TICK_MINUS
    if "zero" in cleaned or cleaned in {"zerotick", "flat", "unchanged", "0"}:
        return TRADE_TICK_ZERO
    try:
        val = float(tick_raw)
    except (TypeError, ValueError):
        return TRADE_TICK_ZERO
    if val > 0:
        return TRADE_TICK_PLUS
    if val < 0:
        return TRADE_TICK_MINUS
    return TRADE_TICK_ZERO


def _compact_is_rpi_code(rpi_raw: Any) -> int:
    if rpi_raw is None:
        return 0
    if isinstance(rpi_raw, str):
        rpi_norm = rpi_raw.strip().lower()
        if rpi_norm in {"1", "true", "t", "yes", "y", "on"}:
            return 1
        if rpi_norm in {"0", "false", "f", "no", "n", "off", ""}:
            return 0
    try:
        return 1 if float(rpi_raw) != 0.0 else 0
    except (TypeError, ValueError):
        return 0


def _compact_book_levels(levels: Any) -> Tuple[Tuple[float, float], ...]:
    if not isinstance(levels, list):
        return tuple()
    out = []
    for lvl in levels:
        if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
            continue
        try:
            out.append((float(lvl[0]), float(lvl[1])))
        except (TypeError, ValueError):
            continue
    return tuple(out)

def _env_bool_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    v = str(raw).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return 1
    if v in {"0", "false", "f", "no", "n", "off"}:
        return 0
    return int(v)


# Quality/repair env config (parsed once at import time)
BYBIT_DAY_CLIP = int(os.environ.get("BYBIT_DAY_CLIP", "1"))
BYBIT_TS_BACKSTEP_CLAMP_MS = int(os.environ.get("BYBIT_TS_BACKSTEP_CLAMP_MS", "5000"))
BYBIT_STRICT_DATA = _env_bool_int("BYBIT_STRICT_DATA", 0)
ALLOW_DUPLICATE_OB_TS = _env_bool_int("BYBIT_ALLOW_DUPLICATE_OB_TS", 0)
USE_TRADES = _env_bool_int("BYBIT_USE_TRADES", 1)
BYBIT_BAD_EXAMPLES_N = int(os.environ.get("BYBIT_BAD_EXAMPLES_N", "25"))
BYBIT_BAD_FRAC_ABORT = float(os.environ.get("BYBIT_BAD_FRAC_ABORT", "0.005"))
BYBIT_BAD_ABS_ABORT = int(os.environ.get("BYBIT_BAD_ABS_ABORT", "50000"))
ONE_DAY = timedelta(days=1)


def canonical_mode_fields() -> Dict[str, object]:
    trade_history_enabled = bool(USE_TRADES)
    return {
        "trade_history_enabled": trade_history_enabled,
        "event_stream_mode": "ob_th_merged" if trade_history_enabled else "ob_only",
    }


def quality_env_config() -> Dict[str, object]:
    """Serializable quality/repair env knobs for reports/metadata."""
    return {
        "day_clip": int(BYBIT_DAY_CLIP),
        "ts_backstep_clamp_ms": int(BYBIT_TS_BACKSTEP_CLAMP_MS),
        "strict_data": int(BYBIT_STRICT_DATA),
        "bad_examples_n": int(BYBIT_BAD_EXAMPLES_N),
        "bad_frac_abort": float(BYBIT_BAD_FRAC_ABORT),
        "bad_abs_abort": int(BYBIT_BAD_ABS_ABORT),
    }


@dataclass
class DayQuality:
    day: str
    ob_path: str
    th_path: str
    counters: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "ob": {},
            "th": {},
            "merge": {},
            "chain": {},
        }
    )
    raw_ts_min: Optional[int] = None
    raw_ts_max: Optional[int] = None
    out_ts_min: Optional[int] = None
    out_ts_max: Optional[int] = None
    examples: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    abort_flags: Dict[str, bool] = field(default_factory=dict)

    def increment_counter(self, namespace: str, key: str, amount: int = 1) -> None:
        ns = self.counters.setdefault(namespace, {})
        ns[key] = int(ns.get(key, 0) + amount)

    def update_raw_ts(self, ts_ms: int) -> None:
        ts = int(ts_ms)
        self.raw_ts_min = ts if self.raw_ts_min is None else min(self.raw_ts_min, ts)
        self.raw_ts_max = ts if self.raw_ts_max is None else max(self.raw_ts_max, ts)

    def update_output_ts(self, ts_ms: int) -> None:
        ts = int(ts_ms)
        self.out_ts_min = ts if self.out_ts_min is None else min(self.out_ts_min, ts)
        self.out_ts_max = ts if self.out_ts_max is None else max(self.out_ts_max, ts)

    def append_example(self, category: str, payload: Dict[str, object]) -> None:
        bucket = self.examples.setdefault(category, [])
        if len(bucket) < BYBIT_BAD_EXAMPLES_N:
            bucket.append(dict(payload))

    def set_abort_flag(self, flag: str, value: bool = True) -> None:
        self.abort_flags[flag] = bool(value)

    def to_dict(self) -> Dict[str, object]:
        return {
            "day": self.day,
            "ob_path": self.ob_path,
            "th_path": self.th_path,
            "counters": {ns: dict(vals) for ns, vals in self.counters.items()},
            "raw_ts": {"min": self.raw_ts_min, "max": self.raw_ts_max},
            "output_ts": {"min": self.out_ts_min, "max": self.out_ts_max},
            "examples": {k: list(v) for k, v in self.examples.items()},
            "abort_flags": dict(self.abort_flags),
        }


@dataclass
class WeekQuality:
    week_key: str
    days: List[DayQuality] = field(default_factory=list)
    totals: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "ob": {},
            "th": {},
            "merge": {},
            "chain": {},
        }
    )
    tainted: bool = False
    notes: List[str] = field(default_factory=list)

    def add_day(self, day_quality: DayQuality) -> None:
        self.days.append(day_quality)

    def increment_total(self, namespace: str, key: str, amount: int = 1) -> None:
        ns = self.totals.setdefault(namespace, {})
        ns[key] = int(ns.get(key, 0) + amount)

    def append_note(self, note: str) -> None:
        self.notes.append(str(note))

    def recompute_totals(self) -> None:
        self.totals = {"ob": {}, "th": {}, "merge": {}, "chain": {}}
        for day in self.days:
            for namespace, values in day.counters.items():
                ns = self.totals.setdefault(namespace, {})
                for key, value in values.items():
                    ns[key] = int(ns.get(key, 0) + int(value))
            if any(day.abort_flags.values()):
                self.tainted = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "week_key": self.week_key,
            "days": [d.to_dict() for d in self.days],
            "totals": {ns: dict(vals) for ns, vals in self.totals.items()},
            "tainted": bool(self.tainted),
            "notes": list(self.notes),
        }


def _day_bad_abs_and_total(day_quality: DayQuality) -> Tuple[int, int]:
    # Retained for event-time ingest quality gating in iter_weekly_event_stream().
    """Compute corruption and input totals for a day quality record."""
    bad_abs = 0
    for namespace in ("ob", "th", "merge", "chain"):
        for key, value in day_quality.counters.get(namespace, {}).items():
            key_l = str(key).lower()
            if "drop" in key_l or "error" in key_l or "bad" in key_l:
                bad_abs += int(value)

    total_ob = int(day_quality.counters.get("ob", {}).get("total", 0))
    total_th = int(day_quality.counters.get("th", {}).get("total", 0))
    total = total_ob + total_th
    return int(bad_abs), int(total)


# import your training utilities
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from CMSSL17 import (
    FeatureEngine,
    LabelBuilder,
    HORIZONS_MS,
    NUM_HORIZONS,
    LOOKBACK,
    AUX_DIM,
    _open_text,
    timestamp_to_ms_half_even,
    CHECKPOINT_SCHEMA,
)  # keep shared model/data constants only; ingest helpers are local below
# LOOKBACK is a shared model constant from CMSSL17 (single source of truth).

GRACE_MS = max(int(h) for h in HORIZONS_MS)
EVENT_QUEUE_MAXSIZE = 4096
# Weekly chaining guard for multi-file weeks.
WEEK_CHAIN_TS_TOLERANCE_MS = int(BYBIT_TS_BACKSTEP_CLAMP_MS)

# fast json if available
try:
    import orjson as _fastjson
    def fast_json_loads(s: str): return _fastjson.loads(s)
except Exception:
    import json as _fastjson
    def fast_json_loads(s: str): return _fastjson.loads(s)

# --------------- utils ------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)


def merge_event_time(ob_iter, tr_iter, dq_day: Optional[DayQuality] = None, strict: bool = True, B: int = 0):
    """Merge compact OB/trade events by timestamp/sequence with a monotonicity guard."""
    ob_item = next(ob_iter, None)
    tr_item = next(tr_iter, None)
    last_ts = -1
    while ob_item or tr_item:
        ob_ts = ob_item[1] if ob_item is not None else None
        tr_ts = tr_item[1] if tr_item is not None else None
        if ob_item is not None and (tr_item is None or ob_ts <= tr_ts):
            event = ob_item
            ob_item = next(ob_iter, None)
        else:
            # OB wins exact timestamp ties so decision-time book state updates
            # before same-ms trade-derived features are consumed.
            event = tr_item
            tr_item = next(tr_iter, None)
        etype = event[0]
        ts = int(event[1])
        if ts + B < last_ts:
            backstep_ms = int(last_ts - ts)
            if strict:
                raise ValueError("Non-monotonic timestamps in event stream")

            if backstep_ms <= BYBIT_TS_BACKSTEP_CLAMP_MS:
                if dq_day is not None:
                    dq_day.increment_counter("merge", "merge_clamped_backstep")
                    dq_day.append_example(
                        "merge_backstep",
                        {"kind": "clamp", "s": etype[0], "d": backstep_ms, "in": int(ts), "out": int(last_ts)},
                    )
                ts = last_ts
                event = (event[0], int(last_ts), *event[2:])
            else:
                if dq_day is not None:
                    dq_day.increment_counter("merge", "merge_dropped_big_backstep")
                    dq_day.append_example(
                        "merge_backstep",
                        {"kind": "drop", "s": etype[0], "d": backstep_ms, "in": int(ts), "last": int(last_ts)},
                    )
                continue
        last_ts = ts
        if dq_day is not None:
            dq_day.update_output_ts(last_ts)
        yield event



FEATURE_AUX_TAIL = [
    "log_dt_ms",
    "is_trade",
    "log_events_100ms",
    "log_events_500ms",
    "log_events_1000ms",
    "log_events_3000ms",
    "log_events_7500ms",
]


@dataclass
class FeatureFlushJob:
    week_key: str
    chunk_id: int
    row_start: int
    row_end: int
    row_count: int
    out_dir: str
    features_file: str
    ts_file: str
    features: np.ndarray
    ts: np.ndarray


@dataclass
class LabelFlushJob:
    week_key: str
    chunk_id: int
    label_start: int
    label_end: int
    label_count: int
    out_dir: str
    row_idx_file: str
    label_ts_file: str
    y_file: str
    row_idx: np.ndarray
    label_ts: np.ndarray
    y: np.ndarray


class FlatFeatureWriter:
    def __init__(self, out_dir: str, feature_dim: int, ram_budget_mb: int, chunk_size_override: int = 0, start_chunk_id: int = 0, week_key: str = "", flush_callback: Optional[Callable[[object], None]] = None):
        self.out_dir = out_dir
        self.week_key = str(week_key)
        self.feature_dim = int(feature_dim)
        self.flush_callback = flush_callback
        bytes_per_row = (self.feature_dim * 4) + 8
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            auto_n = max(256, int((ram_budget_mb * 1024 * 1024) // max(1, bytes_per_row)))
            safety_cap = max(256, int((2 * 1024 * 1024 * 1024) // max(1, bytes_per_row)))
            self.N = min(auto_n, safety_cap)

        self.features = np.empty((self.N, self.feature_dim), dtype=np.float32)
        self.ts = np.empty((self.N,), dtype=np.int64)
        self.i = 0
        self.cid = int(start_chunk_id)
        self.rows_total = 0
        self.chunks_meta: List[Dict[str, Any]] = []

    def append_row(self, ts_decision_ms: int, row: np.ndarray) -> int:
        if self.i >= self.N:
            self.flush()
        if row.shape[0] != self.feature_dim:
            raise ValueError(f"Feature row dim mismatch: {row.shape[0]} != {self.feature_dim}")
        self.features[self.i] = row
        self.ts[self.i] = int(ts_decision_ms)
        row_idx = self.rows_total + self.i
        self.i += 1
        return int(row_idx)

    def overwrite_latest_row(self, ts_decision_ms: int, row: np.ndarray) -> int:
        if self.i <= 0:
            raise RuntimeError("Cannot overwrite latest feature row in an empty open chunk")
        if row.shape[0] != self.feature_dim:
            raise ValueError(f"Feature row dim mismatch: {row.shape[0]} != {self.feature_dim}")
        idx = self.i - 1
        self.features[idx] = row
        self.ts[idx] = int(ts_decision_ms)
        return int(self.rows_total + idx)

    def _build_flush_job(self) -> Optional[FeatureFlushJob]:
        if self.i == 0:
            return None
        chunk_id = int(self.cid)
        row_count = int(self.i)
        row_start = int(self.rows_total)
        row_end = int(row_start + row_count)
        job = FeatureFlushJob(
            week_key=self.week_key,
            chunk_id=chunk_id,
            row_start=row_start,
            row_end=row_end,
            row_count=row_count,
            out_dir=self.out_dir,
            features_file=f"features_{chunk_id:03d}.npy",
            ts_file=f"ts_{chunk_id:03d}.npy",
            features=self.features,
            ts=self.ts,
        )
        self.chunks_meta.append({
            "chunk": chunk_id,
            "row_start": row_start,
            "row_end": row_end,
            "n": row_count,
            "files": {"features": job.features_file, "ts": job.ts_file},
        })
        self.rows_total = row_end
        self.cid += 1
        self.i = 0
        self.features = np.empty((self.N, self.feature_dim), dtype=np.float32)
        self.ts = np.empty((self.N,), dtype=np.int64)
        return job

    def flush(self) -> None:
        job = self._build_flush_job()
        if job is None:
            return
        if self.flush_callback is None:
            _persist_flush_job(job)
        else:
            self.flush_callback(job)


class FlatLabelWriter:
    def __init__(self, out_dir: str, ram_budget_mb: int, chunk_size_override: int = 0, start_chunk_id: int = 0, week_key: str = "", flush_callback: Optional[Callable[[object], None]] = None):
        self.out_dir = out_dir
        self.week_key = str(week_key)
        self.flush_callback = flush_callback
        bytes_per_row = (8 + 8 + (NUM_HORIZONS * 4))
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            auto_n = max(256, int((ram_budget_mb * 1024 * 1024) // max(1, bytes_per_row)))
            safety_cap = max(256, int((2 * 1024 * 1024 * 1024) // max(1, bytes_per_row)))
            self.N = min(auto_n, safety_cap)

        self.row_idx = np.empty((self.N,), dtype=np.int64)
        self.label_ts = np.empty((self.N,), dtype=np.int64)
        self.y = np.empty((self.N, NUM_HORIZONS), dtype=np.float32)
        self.i = 0
        self.cid = int(start_chunk_id)
        self.labels_total = 0
        self.chunks_meta: List[Dict[str, Any]] = []

    def append_label(self, row_idx: int, label_ts: int, y: np.ndarray) -> None:
        if self.i >= self.N:
            self.flush()
        self.row_idx[self.i] = int(row_idx)
        self.label_ts[self.i] = int(label_ts)
        self.y[self.i] = y
        self.i += 1

    def _build_flush_job(self) -> Optional[LabelFlushJob]:
        if self.i == 0:
            return None
        chunk_id = int(self.cid)
        label_count = int(self.i)
        label_start = int(self.labels_total)
        label_end = int(label_start + label_count)
        job = LabelFlushJob(
            week_key=self.week_key,
            chunk_id=chunk_id,
            label_start=label_start,
            label_end=label_end,
            label_count=label_count,
            out_dir=self.out_dir,
            row_idx_file=f"row_idx_{chunk_id:03d}.npy",
            label_ts_file=f"label_ts_{chunk_id:03d}.npy",
            y_file=f"y_{chunk_id:03d}.npy",
            row_idx=self.row_idx,
            label_ts=self.label_ts,
            y=self.y,
        )
        self.chunks_meta.append({
            "chunk": chunk_id,
            "label_start": label_start,
            "label_end": label_end,
            "n": label_count,
            "files": {"row_idx": job.row_idx_file, "label_ts": job.label_ts_file, "y": job.y_file},
        })
        self.labels_total = label_end
        self.cid += 1
        self.i = 0
        self.row_idx = np.empty((self.N,), dtype=np.int64)
        self.label_ts = np.empty((self.N,), dtype=np.int64)
        self.y = np.empty((self.N, NUM_HORIZONS), dtype=np.float32)
        return job

    def flush(self) -> None:
        job = self._build_flush_job()
        if job is None:
            return
        if self.flush_callback is None:
            _persist_flush_job(job)
        else:
            self.flush_callback(job)


_SENTINEL_FLUSH_JOB = object()
_FLUSH_QUEUE_MAXSIZE = max(8, 2 * FLUSH_WORKERS)


def _persist_flush_job(job: object) -> None:
    if isinstance(job, FeatureFlushJob):
        np.save(os.path.join(job.out_dir, job.features_file), job.features[: job.row_count])
        np.save(os.path.join(job.out_dir, job.ts_file), job.ts[: job.row_count])
        return
    if isinstance(job, LabelFlushJob):
        np.save(os.path.join(job.out_dir, job.row_idx_file), job.row_idx[: job.label_count])
        np.save(os.path.join(job.out_dir, job.label_ts_file), job.label_ts[: job.label_count])
        np.save(os.path.join(job.out_dir, job.y_file), job.y[: job.label_count])
        return
    raise TypeError(f"Unsupported flush job type: {type(job)!r}")


class FlatWeekRouter:
    def __init__(self, out_root: str, feature_dim: int, ram_budget_mb: int, chunk_size_override: int, week_index: List[Tuple[str, int, int]], pca_meta: Optional[dict] = None):
        self.out_root = out_root
        self.feature_dim = int(feature_dim)
        self.ram_budget_mb = int(ram_budget_mb)
        self.chunk_size_override = int(chunk_size_override)
        self.week_index = list(week_index)
        self.week_bounds: Dict[str, Tuple[int, int]] = {wk: (start, end) for wk, start, end in self.week_index}
        self.feature_writers: Dict[str, FlatFeatureWriter] = {}
        self.label_writers: Dict[str, FlatLabelWriter] = {}
        self.closed_feature_writers: Dict[str, List[FlatFeatureWriter]] = defaultdict(list)
        self.closed_label_writers: Dict[str, List[FlatLabelWriter]] = defaultdict(list)
        self.next_feature_chunk_id: Dict[str, int] = defaultdict(int)
        self.next_label_chunk_id: Dict[str, int] = defaultdict(int)
        self.week_rows_total: Dict[str, int] = defaultdict(int)
        self.week_labels_total: Dict[str, int] = defaultdict(int)
        self.week_decision_span: Dict[str, List[int]] = {}
        self.chunk_size_used: int = 0
        self.week_metas: Dict[str, dict] = {}
        self.pca_meta = dict(pca_meta) if pca_meta is not None else {}
        self.flush_queue: "queue.Queue[object]" = queue.Queue(maxsize=_FLUSH_QUEUE_MAXSIZE)
        self.writer_exception: Optional[BaseException] = None
        self._writer_exception_lock = threading.Lock()
        self.writer_threads: List[threading.Thread] = []
        worker_count = max(1, int(FLUSH_WORKERS))
        for idx in range(worker_count):
            t = threading.Thread(target=self._writer_loop, name=f"offline-ingest-flat-writer-{idx}", daemon=True)
            t.start()
            self.writer_threads.append(t)

    def _check_writer_exception(self) -> None:
        if self.writer_exception is not None:
            raise RuntimeError("Asynchronous chunk writer failed") from self.writer_exception

    def _writer_loop(self) -> None:
        while True:
            if self.writer_exception is not None:
                return
            try:
                job = self.flush_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job is _SENTINEL_FLUSH_JOB:
                    return
                _persist_flush_job(job)
            except BaseException as exc:
                with self._writer_exception_lock:
                    if self.writer_exception is None:
                        self.writer_exception = exc
                return
            finally:
                self.flush_queue.task_done()

    def _enqueue_flush_job(self, job: object) -> None:
        while True:
            self._check_writer_exception()
            try:
                self.flush_queue.put(job, timeout=0.5)
                self._check_writer_exception()
                return
            except queue.Full:
                continue

    def _ensure_feature_writer(self, week_key: str) -> FlatFeatureWriter:
        writer = self.feature_writers.get(week_key)
        if writer is not None:
            return writer
        if week_key in self.week_metas:
            raise RuntimeError(f"Week '{week_key}' is already finalized; refusing to reopen writer.")
        week_dir = os.path.join(self.out_root, week_key)
        ensure_dir(week_dir)
        writer = FlatFeatureWriter(
            week_dir,
            self.feature_dim,
            self.ram_budget_mb,
            self.chunk_size_override,
            start_chunk_id=int(self.next_feature_chunk_id.get(week_key, 0)),
            week_key=week_key,
            flush_callback=self._enqueue_flush_job,
        )
        self.feature_writers[week_key] = writer
        if not self.chunk_size_used:
            self.chunk_size_used = int(writer.N)
        return writer

    def _ensure_label_writer(self, week_key: str) -> FlatLabelWriter:
        writer = self.label_writers.get(week_key)
        if writer is not None:
            return writer
        if week_key in self.week_metas:
            raise RuntimeError(f"Week '{week_key}' is already finalized; refusing to reopen writer.")
        week_dir = os.path.join(self.out_root, week_key)
        ensure_dir(week_dir)
        writer = FlatLabelWriter(
            week_dir,
            self.ram_budget_mb,
            self.chunk_size_override,
            start_chunk_id=int(self.next_label_chunk_id.get(week_key, 0)),
            week_key=week_key,
            flush_callback=self._enqueue_flush_job,
        )
        self.label_writers[week_key] = writer
        return writer

    def _find_week_key(self, ts_ms: int) -> str:
        for wk, start_ms, end_ms in self.week_index:
            if start_ms <= ts_ms < end_ms:
                return wk
        if self.week_index:
            last_wk, _last_start, last_end = self.week_index[-1]
            if ts_ms >= last_end and ts_ms < last_end + GRACE_MS:
                return last_wk
        raise ValueError(f"No week found for decision timestamp {ts_ms}")

    def append_feature_row(self, ts_decision_ms: int, row: np.ndarray) -> Tuple[str, int]:
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_feature_writer(wk)
        row_idx = writer.append_row(int(ts_decision_ms), row)
        self.week_rows_total[wk] = max(self.week_rows_total[wk], int(row_idx) + 1)
        if wk not in self.week_decision_span:
            self.week_decision_span[wk] = [int(ts_decision_ms), int(ts_decision_ms)]
        else:
            span = self.week_decision_span[wk]
            span[0] = min(span[0], int(ts_decision_ms))
            span[1] = max(span[1], int(ts_decision_ms))
        return wk, int(row_idx)

    def overwrite_latest_feature_row(self, ts_decision_ms: int, row: np.ndarray) -> Tuple[str, int]:
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_feature_writer(wk)
        row_idx = writer.overwrite_latest_row(int(ts_decision_ms), row)
        if wk not in self.week_decision_span:
            self.week_decision_span[wk] = [int(ts_decision_ms), int(ts_decision_ms)]
        else:
            span = self.week_decision_span[wk]
            span[0] = min(span[0], int(ts_decision_ms))
            span[1] = max(span[1], int(ts_decision_ms))
        return wk, int(row_idx)

    def add_label(self, week_key: str, row_idx: int, label_ts: int, label: np.ndarray) -> None:
        self._check_writer_exception()
        writer = self._ensure_label_writer(week_key)
        writer.append_label(int(row_idx), int(label_ts), label.astype(np.float32, copy=False))
        self.week_labels_total[week_key] = int(self.week_labels_total.get(week_key, 0) + 1)

    def _close_week_writers(self, week_key: str) -> None:
        f_writer = self.feature_writers.pop(week_key, None)
        if f_writer is not None:
            f_writer.flush()
            self.next_feature_chunk_id[week_key] = int(f_writer.cid)
            self.closed_feature_writers[week_key].append(f_writer)
        l_writer = self.label_writers.pop(week_key, None)
        if l_writer is not None:
            l_writer.flush()
            self.next_label_chunk_id[week_key] = int(l_writer.cid)
            self.closed_label_writers[week_key].append(l_writer)

    def _build_week_meta(self, week_key: str, feature_writers: List[FlatFeatureWriter], label_writers: List[FlatLabelWriter]) -> dict:
        span = self.week_decision_span.pop(week_key, None)
        meta_path = os.path.join(self.out_root, week_key, "meta_week.json")
        feature_chunks = []
        for writer in feature_writers:
            feature_chunks.extend(dict(entry) for entry in writer.chunks_meta)
        feature_chunks.sort(key=lambda entry: int(entry["chunk"]))

        label_chunks = []
        for writer in label_writers:
            label_chunks.extend(dict(entry) for entry in writer.chunks_meta)
        label_chunks.sort(key=lambda entry: int(entry["chunk"]))

        rows_total = int(sum(int(entry.get("n", 0)) for entry in feature_chunks))
        labels_total = int(sum(int(entry.get("n", 0)) for entry in label_chunks))

        meta = {
            "week": week_key,
            "decision_policy": DECISION_POLICY,
            "decision_time_basis": "ob_event_time",
            "window_ms": 60_000,
            "decision_stride_policy": "every_ob_event",
            "label_delta_ms": 0,
            "label_units": "signed_log_return_bps",
            "target_task": "horizon_specific_signed_raw_bps_targets",
            "target_transform": "signed_sqrt_raw_bps",
            "low_abs_trim_fraction": 0.02,
            "high_abs_trim_fraction": 0.02,
            "checkpoint_schema_expected": CHECKPOINT_SCHEMA,
            **canonical_mode_fields(),
            "lookback": int(LOOKBACK),
            "feature_dim_total": int(self.feature_dim),
            "feature_dim_core": int(self.feature_dim - AUX_DIM),
            "aux_dim": int(AUX_DIM),
            "aux_tail": list(FEATURE_AUX_TAIL),
            "label_dim": int(NUM_HORIZONS),
            "horizons_ms": [int(h) for h in HORIZONS_MS],
            "rows_total": rows_total,
            "labels_total": labels_total,
            "feature_chunks": feature_chunks,
            "label_chunks": label_chunks,
            "meta_path": os.path.join(week_key, "meta_week.json"),
        }
        if span:
            meta["decision_ts_range"] = {"min": int(span[0]), "max": int(span[1])}
        if self.pca_meta:
            meta["pca"] = dict(self.pca_meta)
        else:
            meta["pca"] = {"applied": False, "var_kept": float(PCA_VAR_TARGET), "k": 0, "model_path": None}

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.week_metas[week_key] = meta
        print(f"[write] week={week_key} feature_chunks={len(feature_chunks)} rows={rows_total} labels={labels_total}", flush=True)
        return meta

    def _finalize_closed_weeks(self) -> None:
        week_keys = sorted(set(self.closed_feature_writers.keys()) | set(self.closed_label_writers.keys()))
        for wk in week_keys:
            f_writers = self.closed_feature_writers.pop(wk, [])
            l_writers = self.closed_label_writers.pop(wk, [])
            self._build_week_meta(wk, f_writers, l_writers)

    def close_old_writers(self, watermark_ms: int) -> None:
        to_close = []
        for wk in list(self.feature_writers.keys()):
            _start_ms, end_ms = self.week_bounds[wk]
            if end_ms + GRACE_MS < watermark_ms:
                to_close.append(wk)
        for wk in to_close:
            self._close_week_writers(wk)

    def flush_all(self) -> None:
        for wk in sorted(set(self.feature_writers.keys()) | set(self.label_writers.keys())):
            self._close_week_writers(wk)
        self._check_writer_exception()
        for _ in self.writer_threads:
            self.flush_queue.put(_SENTINEL_FLUSH_JOB)
        for t in self.writer_threads:
            t.join()
        self._check_writer_exception()
        self._finalize_closed_weeks()
        for wk in list(self.week_decision_span.keys()):
            self.week_decision_span.pop(wk, None)

# --------------- dataset-wide processing ---------------
def _compute_dataset_span(pairs: List[WeekPair]):
    if not pairs:
        return None, None
    starts = []
    ends = []
    for wk, _ob_path, _th_path in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(wk)
        starts.append(start_dt)
        ends.append(end_dt)
    return min(starts), max(ends)


def _dt_to_epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_week_index(pairs: List[WeekPair]):
    index = []
    for wk, _ob_path, _th_path in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(wk)
        start_ms = _dt_to_epoch_ms(start_dt)
        end_ms = _dt_to_epoch_ms(end_dt + ONE_DAY)
        index.append((wk, start_ms, end_ms))
    index.sort(key=lambda x: x[1])
    return index


def _print_coarse_timing_totals(prefix: str, totals: Dict[str, float]) -> None:
    ordered = [
        ("wall_s", "wall"),
        ("queue_wait_s", "queue_wait"),
        ("event_proc_s", "event_proc"),
        ("router_housekeeping_s", "router_housekeeping"),
    ]
    parts = []
    for key, label in ordered:
        if key in totals:
            parts.append(f"{label}={float(totals[key]):.6f}s")
    if not parts:
        return
    print(f"{prefix} {' '.join(parts)}", flush=True)




def _iter_week_merged_events(
    week_key: str,
    ob_paths: List[str],
    th_paths: List[str],
    week_quality: Optional[WeekQuality] = None,
):
    """Yield compact ingest tuples for a full week in timestamp order."""
    ob_list = list(ob_paths)
    th_list = list(th_paths)

    def _daily_path_day(path: str, side: str) -> date:
        name = os.path.basename(path)
        pattern = OB_DAILY_RE if side == "OB" else TH_DAILY_RE
        m = pattern.match(name)
        if not m:
            raise ValueError(
                f"Could not parse daily date for {side} file '{name}' in week={week_key}"
            )
        return _parse_ymd_date(m.group("d"))

    def _assert_daily_side_sorted(paths: List[str], side: str):
        prev_day: Optional[date] = None
        prev_name: Optional[str] = None
        for path in paths:
            day = _daily_path_day(path, side)
            if prev_day is not None and day <= prev_day:
                raise ValueError(
                    f"Daily file list is not sorted ascending by day: week={week_key} side={side} "
                    f"prev={prev_name}({prev_day.isoformat()}) curr={os.path.basename(path)}({day.isoformat()})"
                )
            prev_day = day
            prev_name = os.path.basename(path)

    _assert_daily_side_sorted(ob_list, "OB")
    if th_list:
        _assert_daily_side_sorted(th_list, "TH")

        for ob_p, th_p in zip(ob_list, th_list):
            ob_day = _daily_path_day(ob_p, "OB")
            th_day = _daily_path_day(th_p, "TH")
            if ob_day != th_day:
                raise ValueError(
                    "Daily OB/TH day mismatch: "
                    f"week_key={week_key} "
                    f"ob={os.path.basename(ob_p)}({ob_day.isoformat()}) "
                    f"th={os.path.basename(th_p)}({th_day.isoformat()})"
                )

        if len(ob_list) != len(th_list):
            raise ValueError(
                "Mismatched OB/TH file counts within week block: "
                f"ob={len(ob_list)} th={len(th_list)}"
            )

    strict_mode = bool(BYBIT_STRICT_DATA)
    assert ONE_DAY.total_seconds() > 0, "ONE_DAY must be positive and non-zero"
    last_ts_global: Optional[int] = None
    prev_ob_name: Optional[str] = None
    prev_th_name: Optional[str] = None

    if not th_list:
        for ob_path in ob_list:
            ob_name = os.path.basename(ob_path)
            day = _daily_path_day(ob_path, "OB")
            day_start_ms = _dt_to_epoch_ms(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
            day_end_ms = _dt_to_epoch_ms(datetime.combine(day + ONE_DAY, datetime.min.time(), tzinfo=timezone.utc))
            dq_day = DayQuality(
                day=day.isoformat(),
                ob_path=ob_path,
                th_path="",
            )
            aborted_for_corruption = False
            ob_iter = safe_ob_iter(ob_path, day_start_ms, day_end_ms, dq_day)
            for event in ob_iter:
                ts = int(event[1])
                if (
                    last_ts_global is not None
                    and ts + WEEK_CHAIN_TS_TOLERANCE_MS < last_ts_global
                ):
                    prev_pair = (
                        f"{prev_ob_name} | {prev_th_name}"
                        if prev_ob_name is not None and prev_th_name is not None
                        else (f"{prev_ob_name} | <ob-only>" if prev_ob_name is not None else "<week-start>")
                    )
                    backstep_ms = int(last_ts_global - ts)
                    if strict_mode:
                        raise ValueError(
                            "Non-monotonic timestamps while chaining daily files within week: "
                            f"week={week_key} "
                            f"prev_day_files={prev_pair} "
                            f"curr_day_files={ob_name} | <ob-only> "
                            f"prev_ts={last_ts_global} curr_ts={ts} "
                            f"tolerance_ms={WEEK_CHAIN_TS_TOLERANCE_MS}"
                        )

                    if backstep_ms <= WEEK_CHAIN_TS_TOLERANCE_MS:
                        dq_day.increment_counter("chain", "chain_clamped_backstep")
                        dq_day.append_example(
                            "chain_backstep",
                            {
                                "a": "clamp",
                                "p": prev_pair,
                                "c": f"{ob_name} | <ob-only>",
                                "prev_ts": int(last_ts_global),
                                "curr_ts": int(ts),
                            },
                        )
                        event = (event[0], int(last_ts_global), *event[2:])
                        ts = int(last_ts_global)
                    else:
                        dq_day.increment_counter("chain", "chain_dropped_big_backstep")
                        dq_day.append_example(
                            "chain_backstep",
                            {
                                "a": "drop",
                                "p": prev_pair,
                                "c": f"{ob_name} | <ob-only>",
                                "prev_ts": int(last_ts_global),
                                "curr_ts": int(ts),
                            },
                        )
                        continue

                last_ts_global = ts
                prev_ob_name = ob_name
                prev_th_name = None
                yield event

                bad_abs, total = _day_bad_abs_and_total(dq_day)
                bad_frac = float(bad_abs) / float(max(1, total))
                if bad_abs >= BYBIT_BAD_ABS_ABORT or bad_frac >= BYBIT_BAD_FRAC_ABORT:
                    aborted_for_corruption = True
                    dq_day.set_abort_flag("aborted_due_to_corruption", True)
                    if week_quality is not None:
                        week_quality.tainted = True
                        week_quality.append_note(
                            "[warn] corruption abort day="
                            f"{dq_day.day} week={week_key} bad_abs={bad_abs} total={total} "
                            f"bad_frac={bad_frac:.6f} thresholds(abs={BYBIT_BAD_ABS_ABORT}, frac={BYBIT_BAD_FRAC_ABORT})"
                        )
                    break

            bad_abs, total = _day_bad_abs_and_total(dq_day)
            bad_frac = float(bad_abs) / float(max(1, total))
            if (not aborted_for_corruption) and (
                bad_abs >= BYBIT_BAD_ABS_ABORT or bad_frac >= BYBIT_BAD_FRAC_ABORT
            ):
                aborted_for_corruption = True
                dq_day.set_abort_flag("aborted_due_to_corruption", True)
                if week_quality is not None:
                    week_quality.tainted = True
                    week_quality.append_note(
                        "[warn] corruption abort day="
                        f"{dq_day.day} week={week_key} bad_abs={bad_abs} total={total} "
                        f"bad_frac={bad_frac:.6f} thresholds(abs={BYBIT_BAD_ABS_ABORT}, frac={BYBIT_BAD_FRAC_ABORT})"
                    )
            dq_day.increment_counter("merge", "bad_abs", bad_abs)
            dq_day.increment_counter("merge", "total", total)
            if week_quality is not None:
                week_quality.add_day(dq_day)
            if aborted_for_corruption:
                continue
        return

    for ob_path, th_path in zip(ob_list, th_list):
        ob_name = os.path.basename(ob_path)
        th_name = os.path.basename(th_path)
        day = _daily_path_day(ob_path, "OB")
        day_start_ms = _dt_to_epoch_ms(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end_ms = _dt_to_epoch_ms(datetime.combine(day + ONE_DAY, datetime.min.time(), tzinfo=timezone.utc))
        dq_day = DayQuality(
            day=day.isoformat(),
            ob_path=ob_path,
            th_path=th_path,
        )
        aborted_for_corruption = False
        ob_iter = safe_ob_iter(ob_path, day_start_ms, day_end_ms, dq_day)
        th_iter = safe_th_iter(th_path, day_start_ms, day_end_ms, dq_day)
        for event in merge_event_time(ob_iter, th_iter, dq_day=dq_day, strict=strict_mode, B=0):
            ts = int(event[1])
            if (
                last_ts_global is not None
                and ts + WEEK_CHAIN_TS_TOLERANCE_MS < last_ts_global
            ):
                prev_pair = (
                    f"{prev_ob_name} | {prev_th_name}"
                    if prev_ob_name is not None and prev_th_name is not None
                    else "<week-start>"
                )
                backstep_ms = int(last_ts_global - ts)
                if strict_mode:
                    raise ValueError(
                        "Non-monotonic timestamps while chaining daily files within week: "
                        f"week={week_key} "
                        f"prev_day_files={prev_pair} "
                        f"curr_day_files={ob_name} | {th_name} "
                        f"prev_ts={last_ts_global} curr_ts={ts} "
                        f"tolerance_ms={WEEK_CHAIN_TS_TOLERANCE_MS}"
                    )

                if backstep_ms <= WEEK_CHAIN_TS_TOLERANCE_MS:
                    dq_day.increment_counter("chain", "chain_clamped_backstep")
                    dq_day.append_example(
                        "chain_backstep",
                        {
                            "a": "clamp",
                            "p": prev_pair,
                            "c": f"{ob_name} | {th_name}",
                            "prev_ts": int(last_ts_global),
                            "curr_ts": int(ts),
                        },
                    )
                    event = (event[0], int(last_ts_global), *event[2:])
                    ts = int(last_ts_global)
                else:
                    dq_day.increment_counter("chain", "chain_dropped_big_backstep")
                    dq_day.append_example(
                        "chain_backstep",
                        {
                            "a": "drop",
                            "p": prev_pair,
                            "c": f"{ob_name} | {th_name}",
                            "prev_ts": int(last_ts_global),
                            "curr_ts": int(ts),
                        },
                    )
                    continue

            last_ts_global = ts
            prev_ob_name = ob_name
            prev_th_name = th_name
            yield event

            bad_abs, total = _day_bad_abs_and_total(dq_day)
            bad_frac = float(bad_abs) / float(max(1, total))
            if bad_abs >= BYBIT_BAD_ABS_ABORT or bad_frac >= BYBIT_BAD_FRAC_ABORT:
                aborted_for_corruption = True
                dq_day.set_abort_flag("aborted_due_to_corruption", True)
                if week_quality is not None:
                    week_quality.tainted = True
                    week_quality.append_note(
                        "[warn] corruption abort day="
                        f"{dq_day.day} week={week_key} bad_abs={bad_abs} total={total} "
                        f"bad_frac={bad_frac:.6f} thresholds(abs={BYBIT_BAD_ABS_ABORT}, frac={BYBIT_BAD_FRAC_ABORT})"
                    )
                break

        bad_abs, total = _day_bad_abs_and_total(dq_day)
        bad_frac = float(bad_abs) / float(max(1, total))
        if (not aborted_for_corruption) and (
            bad_abs >= BYBIT_BAD_ABS_ABORT or bad_frac >= BYBIT_BAD_FRAC_ABORT
        ):
            aborted_for_corruption = True
            dq_day.set_abort_flag("aborted_due_to_corruption", True)
            if week_quality is not None:
                week_quality.tainted = True
                week_quality.append_note(
                    "[warn] corruption abort day="
                    f"{dq_day.day} week={week_key} bad_abs={bad_abs} total={total} "
                    f"bad_frac={bad_frac:.6f} thresholds(abs={BYBIT_BAD_ABS_ABORT}, frac={BYBIT_BAD_FRAC_ABORT})"
                )
        dq_day.increment_counter("merge", "bad_abs", bad_abs)
        dq_day.increment_counter("merge", "total", total)
        if week_quality is not None:
            week_quality.add_day(dq_day)
        if aborted_for_corruption:
            continue

class EventFeeder:
    def __init__(
        self,
        pairs: List[WeekPair],
        maxsize: int = EVENT_QUEUE_MAXSIZE,
        collect_quality: bool = True,
    ):
        self.pairs = list(pairs)
        self.queue: "queue.Queue[Tuple[str, Optional[str], Optional[object]]]" = queue.Queue(maxsize=maxsize)
        self._last_first_ts: Optional[int] = None
        self.collect_quality = bool(collect_quality)
        self.week_qualities: Dict[str, WeekQuality] = {}
        self.quality_by_week: Dict[str, Dict[str, object]] = {}

    def _put(self, item: Tuple[str, Optional[str], Optional[object]]):
        while True:
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                kind, wk, _payload = item
                print(f"[feeder] queue full while sending kind={kind!r} week={wk!r}", flush=True)

    def run(self):
        try:
            for wk, ob_paths, th_paths in self.pairs:
                week_quality: Optional[WeekQuality] = None
                if self.collect_quality:
                    week_quality = WeekQuality(week_key=wk)
                    self.week_qualities[wk] = week_quality
                merged = _iter_week_merged_events(wk, ob_paths, th_paths, week_quality=week_quality)

                first_event = next(merged, None)
                if first_event is None:
                    if week_quality is not None:
                        week_quality.recompute_totals()
                        self.quality_by_week[wk] = week_quality.to_dict()
                    self._put(("first", wk, None))
                    self._put(("eof", wk, None))
                    continue

                ts_first = _event_ts(first_event)
                if self._last_first_ts is not None and ts_first < self._last_first_ts:
                    raise ValueError(
                        "Non-monotonic timestamps across weeks: "
                        f"week {wk} starts at {ts_first} < last seen {self._last_first_ts}"
                    )
                self._last_first_ts = ts_first

                # Forward the compact tuple unchanged so both PCA and main ingest
                # can use FeatureEngine.on_fast_event(...).
                self._put(("first", wk, first_event))
                for event in merged:
                    self._put(("evt", wk, event))
                if week_quality is not None:
                    week_quality.recompute_totals()
                    self.quality_by_week[wk] = week_quality.to_dict()
                self._put(("eof", wk, None))

            self._put(("eof", None, None))
        except Exception as exc:
            self._put(("eof", None, exc))


def _stream_core_features(pairs: List[WeekPair]):
    """Stream OB decision-candidate core feature vectors (z-scored) for PCA fitting."""
    if not pairs:
        return

    fe = FeatureEngine()
    sample_count = 0
    last_log = time.monotonic()
    last_wk = None
    stream_started = time.monotonic()
    queue_wait_s = 0.0
    event_proc_s = 0.0

    feeder = EventFeeder(pairs, collect_quality=False)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    q = feeder.queue

    last_global_ts: Optional[int] = None
    try:
        while True:
            t_q = time.monotonic()
            kind, wk, payload = q.get()
            queue_wait_s += time.monotonic() - t_q

            if kind == "first":
                if wk is None:
                    raise RuntimeError("Received 'first' marker without a week key")
                if payload is None:
                    continue
                event = payload
                print(f"[pca-week] {wk}", flush=True)
            elif kind == "evt":
                event = payload
                last_wk = wk
            elif kind == "eof":
                if isinstance(payload, Exception):
                    raise payload
                if wk is None:
                    break
                continue
            else:
                print(f"[pca ] ignoring feeder message kind={kind!r} week={wk}", flush=True)
                continue

            if event is None:
                continue

            t_evt = time.monotonic()
            try:
                ts_ms, feat_z, _mid, _is_trade, _dt_ms = fe.on_fast_event(event)
            except Exception as exc:
                event_repr = repr(event)
                if len(event_repr) > 500:
                    event_repr = event_repr[:500] + "..."
                print(
                    f"[pca-error] week={wk} kind={kind} event={event_repr} exc={exc!r}",
                    flush=True,
                )
                raise
            event_proc_s += time.monotonic() - t_evt
            if last_global_ts is not None and ts_ms < last_global_ts:
                raise ValueError(
                    "Non-monotonic timestamps across weeks during PCA stream: "
                    f"week {wk} event {ts_ms} < last {last_global_ts}"
                )
            last_global_ts = int(ts_ms)
            if _is_trade:
                continue
            sample_count += 1
            now = time.monotonic()
            if now - last_log >= 300:
                print(f"[pca-sample] rows={sample_count} last_wk={last_wk}", flush=True)
                last_log = now
            yield np.asarray(feat_z, dtype=np.float32)
    finally:
        producer_thread.join(timeout=2.0)
        if producer_thread.is_alive():
            print("[pca ] producer thread still alive during shutdown; skipping blocking join", flush=True)
        _print_coarse_timing_totals(
            "[pca-time]",
            {
                "wall_s": time.monotonic() - stream_started,
                "queue_wait_s": queue_wait_s,
                "event_proc_s": event_proc_s,
            },
        )
        fe.print_timer_totals(prefix="[pca-timers]")


def _select_pca_components(sample_rows: np.ndarray, target_var: float) -> int:
    if sample_rows.ndim != 2 or sample_rows.size == 0:
        return 0
    n_samples, n_features = sample_rows.shape
    if n_samples == 0 or n_features == 0:
        return 0
    target = float(max(0.0, min(1.0, target_var)))
    if target <= 0.0:
        return 0

    mean_vec = np.mean(sample_rows, axis=0, keepdims=True)
    centered = sample_rows - mean_vec
    try:
        _u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return min(n_samples, n_features)
    denom = max(1, n_samples - 1)
    explained = (s ** 2) / denom
    total = float(np.sum(explained))
    if not np.isfinite(total) or total <= 0.0:
        return min(n_samples, n_features)
    ratios = np.cumsum(explained / total)
    k_idx = int(np.searchsorted(ratios, target, side="left"))
    k = max(1, min(n_features, n_samples, k_idx + 1))
    return k


def maybe_fit_pca_model(
    pairs: List[WeekPair],
    out_root: str,
    train_weeks: List[str],
    target_var: float,
    sample_limit: int,
    batch_size: int,
    model_filename: str,
    use_existing: int,
):
    """Fit (or reuse) a PCA model using the training subset of week-keyed daily paths.

    Each pair is ``(week_key, ob_paths, th_paths)`` where ``ob_paths`` is an
    ordered per-day file-path list for the week and ``th_paths`` is either the
    aligned per-day trade-history list (trade-enabled mode) or ``[]`` in
    OB-only mode.
    """
    meta = {
        "applied": False,
        "var_kept": float(target_var),
        "k": 0,
        "model_path": None,
    }
    last_log = time.monotonic()
    batches = 0

    if target_var <= 0.0:
        return meta

    if int(use_existing) == 1:
        model_path = os.path.join(out_root, model_filename)
        try:
            with np.load(model_path) as data:
                components = data["components"]
                k = int(components.shape[0]) if components.size else 0
                if k <= 0:
                    raise ValueError("PCA model has no components")
        except Exception as exc:
            print(f"[pca  ] Failed to reuse PCA model '{model_path}': {exc}; disabling PCA")
            return meta

        meta.update({
            "applied": True,
            "k": k,
            "model_path": model_filename,
        })
        print(f"[pca  ] Reusing existing PCA model '{model_path}' (k={k})")
        return meta

    try:
        from sklearn.decomposition import IncrementalPCA  # type: ignore
    except Exception as exc:
        print(f"[pca  ] sklearn unavailable ({exc}); skipping PCA fit")
        return meta

    train_set = set(train_weeks)
    train_pairs = [p for p in pairs if p[0] in train_set]
    if not train_pairs:
        print("[pca  ] No training weeks available; skipping PCA fit")
        return meta

    sample_limit = max(1, int(sample_limit))
    batch_size = int(batch_size)

    sample_rows: List[np.ndarray] = []
    sample_array: Optional[np.ndarray] = None
    pad_rows: Optional[np.ndarray] = None
    ipca = None
    fitted_rows = 0
    total_rows = 0
    pending: List[np.ndarray] = []
    n_components = 0

    def flush_pending(force: bool = False):
        nonlocal pending, fitted_rows, batches, last_log
        if ipca is None or not pending:
            return
        need = max(1, ipca.n_components)
        thresh = max(need, batch_size) if batch_size > 0 else need
        if not force and len(pending) < thresh:
            return
        arr = np.asarray(pending, dtype=np.float32)
        actual_rows = arr.shape[0]
        if actual_rows < need:
            source = pad_rows if pad_rows is not None and pad_rows.size else sample_array
            if source is not None and source.shape[0] >= need:
                pad_needed = need - actual_rows
                arr = np.vstack([arr, source[:pad_needed]])
        ipca.partial_fit(arr)
        fitted_rows += actual_rows
        batches += 1
        if time.monotonic() - last_log >= 300:
            print(f"[pca-fit] fitted={fitted_rows} batches={batches}", flush=True)
            last_log = time.monotonic()
        pending = []

    def ensure_ipca(force: bool = False):
        nonlocal ipca, sample_array, pad_rows, n_components, fitted_rows, last_log
        if ipca is not None:
            return
        if not sample_rows:
            return
        if not force and len(sample_rows) < sample_limit:
            return
        sample_array = np.asarray(sample_rows, dtype=np.float32)
        n_components = _select_pca_components(sample_array, target_var)
        if n_components <= 0:
            return
        ipca = IncrementalPCA(
            n_components=n_components,
            batch_size=None if batch_size <= 0 else max(batch_size, n_components),
        )
        ipca.partial_fit(sample_array)
        print(f"[pca-init] n_components={n_components} sample_rows={sample_array.shape[0]}", flush=True)
        last_log = time.monotonic()
        fitted_rows += sample_array.shape[0]
        pad_rows = sample_array[:n_components].copy()
        sample_rows.clear()

    for feat in _stream_core_features(train_pairs):
        total_rows += 1
        vec = np.asarray(feat, dtype=np.float32)
        if ipca is None:
            sample_rows.append(vec)
            ensure_ipca()
            continue
        pending.append(vec)
        flush_pending()

    ensure_ipca(force=True)

    if ipca is None:
        print("[pca  ] Unable to initialise PCA (insufficient data); skipping")
        return meta

    flush_pending(force=True)

    model_path = os.path.join(out_root, model_filename)
    ensure_dir(os.path.dirname(model_path))
    np.savez(
        model_path,
        mean=ipca.mean_.astype(np.float32, copy=False),
        components=ipca.components_.astype(np.float32, copy=False),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32, copy=False),
    )

    meta.update(
        {
            "applied": True,
            "k": int(ipca.n_components),
            "model_path": model_filename,
            "rows_fitted": int(fitted_rows),
            "rows_total": int(total_rows),
            "sample_rows": int(sample_array.shape[0] if sample_array is not None else 0),
        }
    )

    print(
        f"[pca  ] applied target={target_var:.4f} k={meta['k']} "
        f"sample={meta.get('sample_rows', 0)} fitted={meta.get('rows_fitted', 0)}"
    )

    return meta


def _summarise_pca_meta(meta: Optional[dict]) -> dict:
    base = {
        "applied": False,
        "var_kept": float(PCA_VAR_TARGET),
        "k": 0,
        "model_path": None,
    }
    if not meta:
        return base
    applied = bool(meta.get("applied", False))
    base.update(
        {
            "applied": applied,
            "var_kept": float(meta.get("var_kept", base["var_kept"])),
            "k": int(meta.get("k", 0) if applied else 0),
            "model_path": meta.get("model_path") if applied else None,
        }
    )
    return base


def process_all(
    pairs: List[WeekPair],
    out_root: str,
    pca_meta: dict,
):
    """Run ingest across week pairs with ordered daily OB paths and mode-dependent TH paths (which may be empty in OB-only mode)."""
    ensure_dir(out_root)

    pca_summary = _summarise_pca_meta(pca_meta)
    pca_mean: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_var_ratio: Optional[np.ndarray] = None

    if pca_summary["applied"]:
        model_path = pca_summary.get("model_path")
        full_model_path = os.path.join(out_root, model_path) if model_path else ""
        try:
            with np.load(full_model_path) as data:
                pca_mean = data["mean"].astype(np.float32)
                pca_components = data["components"].astype(np.float32)
                if "explained_variance_ratio" in data:
                    pca_var_ratio = data["explained_variance_ratio"].astype(np.float32)
        except Exception as exc:
            print(f"[pca  ] Failed to load PCA model '{full_model_path}': {exc}; disabling PCA")
            pca_mean = None
            pca_components = None
            pca_var_ratio = None
            pca_summary = _summarise_pca_meta({"applied": False, "var_kept": pca_summary.get("var_kept", PCA_VAR_TARGET)})

    fe = FeatureEngine()
    labeler = LabelBuilder(delta_ms=0, horizons_ms=HORIZONS_MS)

    pending_decisions: deque[Tuple[str, int, int]] = deque()
    last_decision_ts_ms: Optional[int] = None

    F = None
    router: FlatWeekRouter = None  # type: ignore
    total_feature_rows = 0
    total_labels = 0

    ds_start, ds_end = _compute_dataset_span(pairs)
    start_iso = ds_start.date().isoformat() if ds_start else None
    end_iso = ds_end.date().isoformat() if ds_end else None

    week_index = _build_week_index(pairs)

    print(f"[start] ingest weeks={len(pairs)} L={LOOKBACK} budget={RAM_BUDGET}MB")
    last_log = time.monotonic()
    ingest_started = time.monotonic()
    queue_wait_s = 0.0
    event_proc_s = 0.0
    router_housekeeping_s = 0.0

    feeder = EventFeeder(pairs)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    q = feeder.queue

    week_total = len(pairs)
    week_counter = 0

    while True:
        t_q = time.monotonic()
        kind, wk, payload = q.get()
        queue_wait_s += time.monotonic() - t_q

        if kind == "first":
            if wk is None:
                raise RuntimeError("Received 'first' marker without a week key")
            week_counter += 1
            print(f"[week ] {week_counter}/{week_total} {wk}")
            if payload is None:
                print(f"[skip ] {wk} yielded no events")
                continue
            ts_first = _event_ts(payload)
            if last_decision_ts_ms is not None and ts_first < last_decision_ts_ms:
                raise ValueError(
                    "Non-monotonic event timestamps across weeks relative to prior decision time: "
                    f"week {wk} starts at {ts_first} < last_decision_ts_ms {last_decision_ts_ms}"
                )
            event = payload
        elif kind == "evt":
            event = payload
        elif kind == "eof":
            if isinstance(payload, Exception):
                raise payload
            if wk is None:
                break
            continue
        else:
            print(f"[ingest] ignoring feeder message kind={kind!r} week={wk}", flush=True)
            continue

        if event is None:
            continue

        t_evt = time.monotonic()
        ts_ms, feat_z, mid, is_trade, dt_ms = fe.on_fast_event(event)
        event_proc_s += time.monotonic() - t_evt

        if not is_trade:
            feat_core = feat_z
            if pca_components is not None and pca_mean is not None:
                if np.asarray(feat_z).shape[-1] != pca_mean.shape[0]:
                    raise ValueError(
                        f"PCA mean/components dimension {pca_mean.shape[0]} does not match "
                        f"feature dimension {np.asarray(feat_z).shape[-1]}"
                    )
                centered = np.asarray(feat_z, dtype=np.float32, copy=False) - pca_mean
                feat_core = np.dot(centered, pca_components.T).astype(np.float32, copy=False)

            is_duplicate_decision_ts = (last_decision_ts_ms is not None and int(ts_ms) == last_decision_ts_ms)
            if last_decision_ts_ms is not None and int(ts_ms) < last_decision_ts_ms:
                raise RuntimeError(
                    f"Non-monotone decision timestamp: decision_ts_ms={int(ts_ms)} < last_decision_ts_ms={last_decision_ts_ms}"
                )
            if is_duplicate_decision_ts and not ALLOW_DUPLICATE_OB_TS:
                raise RuntimeError(
                    f"Non-monotone decision timestamp: decision_ts_ms={int(ts_ms)} <= last_decision_ts_ms={last_decision_ts_ms}"
                )

            dt_tick = 1 if last_decision_ts_ms is None else int(ts_ms - last_decision_ts_ms)
            tok = build_token(fe, feat_core, is_trade, dt_tick)

            if F is None:
                F = tok.shape[0]
                router = FlatWeekRouter(
                    out_root,
                    F,
                    RAM_BUDGET,
                    CHUNK_SIZE,
                    week_index,
                    pca_meta=pca_summary,
                )

            if router is None:
                raise RuntimeError("Router not initialised")

            if is_duplicate_decision_ts:
                if not pending_decisions:
                    raise RuntimeError("Duplicate OB timestamp cannot update state because no pending decision exists")
                prev_week_key, _prev_row_idx, _prev_ts = pending_decisions[-1]
                week_key, row_idx = router.overwrite_latest_feature_row(int(ts_ms), tok)
                if week_key != prev_week_key:
                    raise RuntimeError("Duplicate timestamp mapped to a different week during overwrite")
                pending_decisions[-1] = (week_key, row_idx, int(ts_ms))
            else:
                week_key, row_idx = router.append_feature_row(int(ts_ms), tok)
                pending_decisions.append((week_key, row_idx, int(ts_ms)))
                labeler.on_decision(int(ts_ms))
                total_feature_rows += 1

            matured = labeler.on_event(int(ts_ms), float(mid))
            last_decision_ts_ms = int(ts_ms)

            if matured is None:
                raise RuntimeError("Matured labels were not produced for OB event")
            for yy in matured:
                if not pending_decisions:
                    raise RuntimeError("Matured label available but no pending decisions to pair")
                lbl_week, lbl_row_idx, lbl_ts = pending_decisions.popleft()
                router.add_label(lbl_week, lbl_row_idx, lbl_ts, yy.astype(np.float32, copy=False))
                total_labels += 1

        t_router = time.monotonic()
        if router is not None:
            router.close_old_writers(int(ts_ms))
        router_housekeeping_s += time.monotonic() - t_router

        if time.monotonic() - last_log >= 300:
            print(
                f"[tok  ] rows={total_feature_rows} labels={total_labels} weeks={week_counter}/{week_total} "
                f"chunkN={router.chunk_size_used if router else 0}",
                flush=True,
            )
            last_log = time.monotonic()

    producer_thread.join()

    if router is not None:
        router.flush_all()

    feature_dim_total = None if F is None else int(F)
    feature_dim_core = None if F is None else int(F - AUX_DIM)
    label_dim = int(NUM_HORIZONS)
    week_meta_records = {} if router is None else dict(router.week_metas)
    week_quality_records = dict(feeder.quality_by_week)
    weeks_in_order = [wk for wk, _ob, _th in pairs]
    week_row_counts = {wk: int(0 if router is None else router.week_rows_total.get(wk, 0)) for wk in weeks_in_order}
    week_label_counts = {wk: int(0 if router is None else router.week_labels_total.get(wk, 0)) for wk in weeks_in_order}
    total_feature_rows_from_weeks = sum(int(week_meta.get("rows_total", 0)) for week_meta in week_meta_records.values())
    total_labels_from_weeks = sum(int(week_meta.get("labels_total", 0)) for week_meta in week_meta_records.values())
    if int(total_feature_rows) != int(total_feature_rows_from_weeks):
        raise ValueError(
            "Inconsistent dataset totals: total_feature_rows "
            f"{int(total_feature_rows)} != sum(weeks_meta.rows_total) {int(total_feature_rows_from_weeks)}"
        )
    if int(total_labels) != int(total_labels_from_weeks):
        raise ValueError(
            "Inconsistent dataset totals: total_labels "
            f"{int(total_labels)} != sum(weeks_meta.labels_total) {int(total_labels_from_weeks)}"
        )
    weeks_meta_paths = {wk: week_meta_records[wk].get("meta_path", os.path.join(wk, "meta_week.json")) for wk in week_meta_records.keys()}

    quality_week_totals: Dict[str, Dict[str, int]] = {"ob": {}, "th": {}, "merge": {}, "chain": {}}
    quality_week_tainted = 0
    quality_day_count = 0
    quality_day_tainted = 0
    for wk in weeks_in_order:
        week_quality = week_quality_records.get(wk)
        if not week_quality:
            continue
        if bool(week_quality.get("tainted", False)):
            quality_week_tainted += 1
        days = list(week_quality.get("days", []))
        quality_day_count += len(days)
        quality_day_tainted += sum(1 for day in days if any(day.get("abort_flags", {}).values()))
        for namespace, values in week_quality.get("totals", {}).items():
            ns_totals = quality_week_totals.setdefault(namespace, {})
            for key, value in values.items():
                ns_totals[key] = int(ns_totals.get(key, 0) + int(value))

    data_quality_dataset = {
        "quality_config": quality_env_config(),
        "weeks": {wk: week_quality_records[wk] for wk in weeks_in_order if wk in week_quality_records},
        "totals": quality_week_totals,
        "flags": {
            "tainted": bool(quality_week_tainted > 0),
            "tainted_week_count": int(quality_week_tainted),
            "week_count": int(len(week_quality_records)),
            "day_count": int(quality_day_count),
            "tainted_day_count": int(quality_day_tainted),
        },
    }

    for wk in weeks_in_order:
        if wk not in week_quality_records:
            continue
        week_quality_path = os.path.join(out_root, wk, "data_quality.json")
        with open(week_quality_path, "w") as f:
            json.dump(week_quality_records[wk], f, indent=2)

    with open(os.path.join(out_root, "_data_quality.json"), "w") as f:
        json.dump(data_quality_dataset, f, indent=2)

    meta = {
        "dataset_start": start_iso,
        "dataset_end": end_iso,
        "weeks_in_order": weeks_in_order,
        "decision_policy": DECISION_POLICY,
        "decision_time_basis": "ob_event_time",
        "window_ms": 60_000,
        "decision_stride_policy": "every_ob_event",
        "label_delta_ms": 0,
        "label_units": "signed_log_return_bps",
        "target_task": "horizon_specific_signed_raw_bps_targets",
        "target_transform": "signed_sqrt_raw_bps",
        "low_abs_trim_fraction": 0.02,
        "high_abs_trim_fraction": 0.02,
        "checkpoint_schema_expected": CHECKPOINT_SCHEMA,
        **canonical_mode_fields(),
        "storage_format": "flat_decision_rows_v1",
        "lookback": int(LOOKBACK),
        "feature_dim_total": feature_dim_total,
        "feature_dim_core": feature_dim_core,
        "aux_dim": int(AUX_DIM),
        "aux_tail": list(FEATURE_AUX_TAIL),
        "dtype": "float32",
        "ram_budget_mb": int(RAM_BUDGET),
        "chunk_size_used": 0 if (router is None or router.chunk_size_used == 0) else int(router.chunk_size_used),
        "label_dim": label_dim,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "total_feature_rows": int(total_feature_rows),
        "total_labels": int(total_labels),
        "week_row_counts": week_row_counts,
        "week_label_counts": week_label_counts,
        "weeks_meta": weeks_meta_paths,
        "data_quality_path": "_data_quality.json",
    }
    meta["pca"] = dict(pca_summary)
    if pca_var_ratio is not None:
        meta["pca"]["explained_variance_ratio"] = [float(x) for x in pca_var_ratio]
    meta["splits"] = build_four_week_pipeline_splits(weeks_in_order, week_meta_records)
    if week_meta_records:
        expected_mode = canonical_mode_fields()
        for wk in weeks_in_order:
            week_meta = week_meta_records.get(wk)
            if not week_meta:
                continue
            for field, expected in expected_mode.items():
                observed = week_meta.get(field)
                if observed != expected:
                    raise ValueError(
                        f"Inconsistent ingest mode in week '{wk}': {field}={observed!r} (expected {expected!r})"
                    )

    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[done ] dataset weeks={len(pairs)} total_rows={total_feature_rows} total_labels={total_labels} "
        f"L={LOOKBACK} F={feature_dim_total or 0} chunkN={meta['chunk_size_used']}"
    )
    print(
        f"[pca  ] summary applied={pca_summary['applied']} var_kept={pca_summary['var_kept']:.4f} "
        f"k={pca_summary['k']} model={pca_summary['model_path']}"
    )
    _print_coarse_timing_totals(
        "[ingest-time]",
        {
            "wall_s": time.monotonic() - ingest_started,
            "queue_wait_s": queue_wait_s,
            "event_proc_s": event_proc_s,
            "router_housekeeping_s": router_housekeeping_s,
        },
    )
    fe.print_timer_totals(prefix="[timers]")

# --------------- driver ----------------
def main():
    ensure_dir(OUT_ROOT)
    mode_fields = canonical_mode_fields()
    trade_history_enabled = bool(mode_fields["trade_history_enabled"])
    print(
        f"[ingest mode] trade_history_enabled={str(trade_history_enabled).lower()} "
        f"event_stream_mode={mode_fields['event_stream_mode']}"
    )
    pairs = pair_weeks(OB_DIR, TH_DIR)

    if not pairs:
        if trade_history_enabled:
            print(f"No week pairs found under OB_DIR={OB_DIR} and TH_DIR={TH_DIR}")
        else:
            print(f"No week pairs found under OB_DIR={OB_DIR}")
        return

    requested_weeks = _parse_requested_weeks(RAW_BYBIT_WEEKS)

    if requested_weeks:
        week_lookup = {wk for wk, _ob, _th in pairs}
        missing = [wk for wk in requested_weeks if wk not in week_lookup]
        if missing:
            raise ValueError(
                f"Requested BYBIT_WEEKS not found in available data: {', '.join(missing)}"
            )

        seen = set()
        duplicate_weeks = []
        duplicate_seen = set()
        for wk in requested_weeks:
            if wk in seen:
                if wk not in duplicate_seen:
                    duplicate_weeks.append(wk)
                    duplicate_seen.add(wk)
            else:
                seen.add(wk)

        if duplicate_weeks:
            raise ValueError(
                "BYBIT_WEEKS contains duplicate week keys; duplicates are not allowed: "
                + ", ".join(duplicate_weeks)
            )

        requested_set = set(requested_weeks)
        pairs = [pair for pair in pairs if pair[0] in requested_set]

    pairs = _sort_pairs_by_end(pairs)
    if len(pairs) != 4:
        raise ValueError(
            f"Need exactly 4 distinct consecutive weeks of data after BYBIT_WEEKS filtering; found {len(pairs)}."
        )

    _assert_week_order(pairs)
    _assert_weeks_consecutive(pairs)

    chosen_weeks = [wk for wk, _ob, _th in pairs]

    print(f"[plan ] weeks={len(pairs)} "
          f"RAM={RAM_BUDGET}MB chunk_size={CHUNK_SIZE if CHUNK_SIZE>0 else 'auto'}")
    print(f"[weeks] {', '.join(chosen_weeks)}")

    print(f"[paths] OB_DIR={OB_DIR}")
    if trade_history_enabled:
        print(f"[paths] TH_DIR={TH_DIR}")
    print(f"[out  ] OUT_ROOT={OUT_ROOT}")


    selected_weeks = [wk for wk, _ob, _th in pairs]
    week1, week2, week3, week4 = selected_weeks
    print(
        f"[split] protocol=four_week_cmssl_val_test_rl_eval_v2 cmssl.train={week1} cmssl.val={week2} cmssl.test={week3} rl={week3} eval={week4}"
    )
    pca_fit_meta = maybe_fit_pca_model(
        pairs,
        OUT_ROOT,
        [week1],
        PCA_VAR_TARGET,
        PCA_MAX_SAMPLE_ROWS,
        PCA_BATCH_SIZE,
        PCA_MODEL_FILENAME,
        PCA_USE_EXISTING,
    )

    process_all(pairs, OUT_ROOT, pca_fit_meta)

if __name__ == "__main__":
    main()
