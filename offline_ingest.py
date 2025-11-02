#!/usr/bin/env python3
"""
Decision-time ingest (memory-safe):
- Snapshot ONE [LOOKBACK, F] sequence at each decision time.
- Use a RAM budget to auto-size chunked writes (avoid huge in-RAM lists).

Env (defaults are SSD-friendly):
  BYBIT_OB_DIR=/home/gabrool/Documents/OB
  BYBIT_TH_DIR=/home/gabrool/Documents/TH
  BYBIT_OUT_ROOT=/media/gabrool/Expansion/Gabriel/bybit_offline_dt
  BYBIT_MAX_WEEKS=0
  BYBIT_WORKERS=1
  BYBIT_LOOKBACK=1024
  BYBIT_RAM_BUDGET_MB=512          # memory budget for one chunk
  BYBIT_CHUNK_SIZE=0               # 0 = auto from budget; else fixed size
"""

import os, sys, csv, json, re
import queue
import threading
from typing import List, Tuple, Iterable, Dict, Optional
from collections import deque, defaultdict
from decimal import Decimal, ROUND_HALF_EVEN
import itertools
import numpy as np
from datetime import datetime, timezone, timedelta

# ---------------- config ----------------
OB_DIR      = os.environ.get("BYBIT_OB_DIR",   "/home/gabrool/Documents/OB")
TH_DIR      = os.environ.get("BYBIT_TH_DIR",   "/home/gabrool/Documents/TH")
OUT_ROOT    = os.environ.get("BYBIT_OUT_ROOT", "/media/gabrool/Expansion/Gabriel/bybit_offline_dt")

# Week selection: anchor on a known last-week end date, keep the last K weeks
KEEP_WEEKS    = int(os.environ.get("BYBIT_KEEP_WEEKS", "24"))

# Optional PCA dimensionality reduction on the core features
PCA_VAR_TARGET      = float(os.environ.get("BYBIT_PCA_VAR", "0"))
PCA_MAX_SAMPLE_ROWS = int(os.environ.get("BYBIT_PCA_MAX_ROWS", "200000"))
PCA_BATCH_SIZE      = int(os.environ.get("BYBIT_PCA_BATCH", "4096"))
PCA_MODEL_FILENAME  = os.environ.get("BYBIT_PCA_MODEL", "pca_model.npz")

# Parallelism / sequence geometry
WORKERS     = int(os.environ.get("BYBIT_WORKERS", "8"))
# Memory & chunking
RAM_BUDGET  = int(os.environ.get("BYBIT_RAM_BUDGET_MB", "512"))
CHUNK_SIZE  = int(os.environ.get("BYBIT_CHUNK_SIZE", "0"))


# import your training utilities
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from CMSSL17 import (
    FeatureEngine,
    LabelBuilder,
    merge_event_time,
    build_sequence_from_tokens,
    HORIZONS_MS,
    NUM_HORIZONS,
    LOOKBACK,
    AUX_DIM,
    BybitRawIter,
)  # reuse exactly

GRACE_MS = max(int(h) for h in HORIZONS_MS)
EVENT_QUEUE_MAXSIZE = 4096

# fast json if available
try:
    import orjson as _fastjson
    def fast_json_loads(s: str): return _fastjson.loads(s)
except Exception:
    import json as _fastjson
    def fast_json_loads(s: str): return _fastjson.loads(s)

# --------------- utils ------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def list_glob(dir_path: str, pattern: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(dir_path, pattern)))

def _normalise_ob_prefix(base: str) -> str:
    if base.startswith("BTCUSDT_OB_"):
        return base
    if base.startswith("BTCUSDT_TH_"):
        return "BTCUSDT_OB_" + base[len("BTCUSDT_TH_"):]
    return base

def _week_key(path: str, prefix: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r'\.(?:zip|gz|jsonl|csv)$', '', base)
    return base.replace(prefix, "")

def _parse_week_key_any(base: str):
    wk = re.sub(r'^(BTCUSDT_(?:OB|TH)_)', '', base)
    wk = re.sub(r'\.(?:zip|gz|jsonl|csv)$', '', wk)
    m = re.match(r"(\d{2}-\d{2}-\d{4})-to-(\d{2}-\d{2}-\d{4})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%d-%m-%Y")
        e = datetime.strptime(m.group(2), "%d-%m-%Y")
        return s, e, wk
    m = re.match(r"(\d{4}-\d{2}-\d{2})-to-(\d{4}-\d{2}-\d{2})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%Y-%m-%d")
        e = datetime.strptime(m.group(2), "%Y-%m-%d")
        return s, e, wk
    raise ValueError(f"Unrecognized week key: {base}")

def _parse_week_from_pair(ob_path: str, th_path: str):
    ob_base = os.path.basename(ob_path)
    th_base = os.path.basename(th_path)
    ob_key = _normalise_ob_prefix(ob_base)
    th_key = _normalise_ob_prefix(th_base)
    try:
        start_ob, end_ob, wk = _parse_week_key_any(ob_key)
    except ValueError as exc:
        raise ValueError(f"Failed to parse week range from OB file '{ob_base}': {exc}") from exc
    try:
        start_th, end_th, _ = _parse_week_key_any(th_key)
    except ValueError as exc:
        raise ValueError(f"Failed to parse week range from TH file '{th_base}': {exc}") from exc
    if (start_ob, end_ob) != (start_th, end_th):
        raise ValueError(
            "Mismatch between OB/TH week ranges: "
            f"OB='{ob_base}' ({start_ob.date()}→{end_ob.date()}) vs "
            f"TH='{th_base}' ({start_th.date()}→{end_th.date()})"
        )
    return start_ob, end_ob, wk

def pair_weeks(ob_dir: str, th_dir: str) -> List[Tuple[str, str, str]]:
    ob_files = list_glob(ob_dir, "BTCUSDT_OB_*.jsonl")
    th_files = list_glob(th_dir, "BTCUSDT_TH_*.csv")

    ob_map = { _week_key(p, "BTCUSDT_OB_"): p for p in ob_files }
    th_map = { _week_key(p, "BTCUSDT_TH_"): p for p in th_files }

    common = sorted(set(ob_map) & set(th_map))
    if not common:
        return []

    missing_ob = sorted(set(th_map) - set(ob_map))
    missing_th = sorted(set(ob_map) - set(th_map))
    if missing_ob:
        print(f"Warning: missing OB for weeks: {missing_ob}")
    if missing_th:
        print(f"Warning: missing TH for weeks: {missing_th}")

    rows = []
    for wk_key in common:
        ob_path = ob_map[wk_key]
        th_path = th_map[wk_key]
        start_dt, end_dt, wk = _parse_week_from_pair(ob_path, th_path)
        rows.append((end_dt, start_dt, wk, ob_path, th_path))

    rows.sort()
    return [(wk, ob_p, th_p) for (_, _, wk, ob_p, th_p) in rows]

def _slice_last_weeks_pairs(pairs: List[Tuple[str, str, str]], last_end_iso: str, k: int):
    if not pairs:
        return []

    target_end = datetime.strptime(last_end_iso, "%Y-%m-%d")
    rows = []
    for wk, ob_p, th_p in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(_normalise_ob_prefix(f"BTCUSDT_OB_{wk}"))
        rows.append((end_dt, start_dt, wk, ob_p, th_p))
    rows.sort()

    try:
        idx = max(i for i,(e,_,_,_,_) in enumerate(rows) if e <= target_end)
    except ValueError as exc:
        raise ValueError(
            f"No week ending on/before {last_end_iso} found. "
            "Check BYBIT_LAST_WEEK_END or available data."
        ) from exc

    lo = max(0, idx - (k - 1))
    sel = rows[lo:idx+1]
    return [(wk, ob_p, th_p) for (_, _, wk, ob_p, th_p) in sel]

def _assert_week_order(pairs: List[Tuple[str, str, str]]):
    if not pairs:
        return

    parsed = []
    for wk, ob_p, th_p in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(_normalise_ob_prefix(f"BTCUSDT_OB_{wk}"))
        parsed.append((start_dt, end_dt, ob_p, th_p, wk))

    for idx in range(1, len(parsed)):
        _prev_start, prev_end, prev_ob, prev_th, _prev_wk = parsed[idx - 1]
        _curr_start, curr_end, curr_ob, curr_th, _curr_wk = parsed[idx]
        if curr_end <= prev_end:
            raise ValueError(
                "Week files must be strictly increasing by end date: "
                f"'{os.path.basename(curr_ob)}'/'{os.path.basename(curr_th)}' (end={curr_end.date()}) "
                f"not after '{os.path.basename(prev_ob)}'/'{os.path.basename(prev_th)}' (end={prev_end.date()})"
            )

_DEC_THOUSAND = Decimal("1000")


def classify_week_splits(pairs: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """Mirror CMSSL17_offline.choose_splits for week-key tuples."""
    weeks = [wk for wk, _ob, _th in pairs]
    if len(weeks) >= 24:
        weeks = weeks[-24:]
        tr, va, te = weeks[:18], weeks[18:21], weeks[21:24]
    else:
        n = len(weeks)
        if n == 0:
            return [], [], []
        n_tr = max(1, int(round(n * 0.75)))
        n_rest = n - n_tr
        n_va = max(1, int(round(n_rest / 2)))
        n_te = max(1, n - n_tr - n_va)
        tr = weeks[:n_tr]
        va = weeks[n_tr:n_tr + n_va]
        te = weeks[n_tr + n_va:n_tr + n_va + n_te]
    return tr, va, te


def _event_ts(event) -> int:
    """Extract the first integer-like timestamp from an event tuple."""
    if event is None:
        raise ValueError("Expected an event tuple, got None")

    for idx in (0, 1):
        if len(event) <= idx:
            continue
        candidate = event[idx]
        try:
            ts = int(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, float) and not candidate.is_integer():
            continue
        if isinstance(candidate, np.floating) and not candidate.is_integer():
            continue
        return ts

    raise ValueError(
        "Event does not expose an integer timestamp at positions 0 or 1: "
        f"{event!r}"
    )


def _trade_iter_precise(tr_iter: Iterable[Tuple[int, int, dict]]):
    for _ts, seq, row in tr_iter:
        ts_decimal = (Decimal(row["timestamp"]) * _DEC_THOUSAND).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
        yield int(ts_decimal), seq, row

def build_token(fe: FeatureEngine, feat_z, is_trade: bool, dt_ms: float) -> np.ndarray:
    # exact tail order: [log_dt_ms, is_trade, events_100ms]
    events_100ms = fe.event_density_100ms()
    aux_tail = np.array(
        [np.log1p(float(dt_ms)), float(is_trade), events_100ms],
        dtype=np.float32,
    )
    return np.concatenate(
        [np.asarray(feat_z, dtype=np.float32), aux_tail], axis=0
    ).astype(np.float32, copy=False)

# ---------- chunk writer (preallocated) ----------
class ChunkWriter:
    def __init__(self, out_dir: str, lookback: int, feature_dim: int,
                 ram_budget_mb: int, chunk_size_override: int = 0):
        self.out_dir = out_dir
        self.L = int(lookback)
        self.F = int(feature_dim)
        self.F_core = self.F - AUX_DIM
        assert self.F_core > 0, "feature_dim must be > AUX_DIM"
        self.core_dtype = np.float32

        # compute chunk size (keep as you already had it)
        bytes_per_seq = (
            (self.L * self.F_core * 4)
            + (self.L * AUX_DIM * 4)
            + (2 * NUM_HORIZONS * 4)
        )
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            self.N = max(256, int((ram_budget_mb * 1024 * 1024) // bytes_per_seq))
        self.N = min(self.N, 4096)

        # preallocate separate buffers
        self.X_core = np.empty((self.N, self.L, self.F_core), dtype=np.float32)  # cast on flush
        self.X_aux  = np.empty((self.N, self.L, AUX_DIM),     dtype=np.float32)  # keep fp32
        self.Y      = np.empty((self.N, 2 * NUM_HORIZONS), dtype=np.float32)
        self.i = 0
        self.cid = 0
        self.chunks_meta = []

    def add(self, seq: np.ndarray, y: np.ndarray):
        core = seq[:, :self.F_core]
        aux  = seq[:, self.F_core:]
        self.X_core[self.i] = core
        self.X_aux[self.i]  = aux
        self.Y[self.i]      = y
        self.i += 1
        if self.i >= self.N:
            self.flush()

    def flush(self):
        if self.i == 0: return
        x_core_path = os.path.join(self.out_dir, f"Xcore_{self.cid:03d}.npy")
        x_aux_path  = os.path.join(self.out_dir, f"Xaux_{self.cid:03d}.npy")
        y_path      = os.path.join(self.out_dir, f"y_{self.cid:03d}.npy")

        # optional: warn if core would overflow fp16
        if self.core_dtype == np.float16:
            maxabs = float(np.max(np.abs(self.X_core[:self.i])))
            if maxabs > np.finfo(np.float16).max:
                print(f"[warn] core max {maxabs:.1f} exceeds fp16 range; consider BYBIT_SAVE_DTYPE=bf16", flush=True)

        np.save(x_core_path, self.X_core[:self.i].astype(self.core_dtype, copy=False))
        np.save(x_aux_path,  self.X_aux[:self.i])                 # fp32
        np.save(y_path,      self.Y[:self.i])                     # fp32

        self.chunks_meta.append({
            "chunk": int(self.cid),
            "n": int(self.i),
            "files": {"core": os.path.basename(x_core_path),
                      "aux":  os.path.basename(x_aux_path),
                      "y":    os.path.basename(y_path)}
        })
        self.cid += 1
        self.i = 0


class WeekWriterRouter:
    def __init__(
        self,
        out_root: str,
        lookback: int,
        feature_dim: int,
        ram_budget_mb: int,
        chunk_size_override: int,
        week_index: List[Tuple[str, int, int]],
        pca_meta: Optional[dict] = None,
    ):
        self.out_root = out_root
        self.lookback = int(lookback)
        self.feature_dim = int(feature_dim)
        self.ram_budget_mb = int(ram_budget_mb)
        self.chunk_size_override = int(chunk_size_override)
        self.week_index = list(week_index)
        self.week_bounds: Dict[str, Tuple[int, int]] = {
            wk: (start, end) for wk, start, end in self.week_index
        }
        self.writers: Dict[str, ChunkWriter] = {}
        self.week_counts: Dict[str, int] = defaultdict(int)
        self.week_decision_span: Dict[str, List[int]] = {}
        self.chunk_size_used: int = 0
        self.week_metas: Dict[str, dict] = {}
        self.pca_meta = dict(pca_meta) if pca_meta is not None else {}

    def _ensure_writer(self, week_key: str) -> ChunkWriter:
        if week_key in self.writers:
            return self.writers[week_key]
        week_dir = os.path.join(self.out_root, week_key)
        ensure_dir(week_dir)
        writer = ChunkWriter(
            week_dir,
            self.lookback,
            self.feature_dim,
            self.ram_budget_mb,
            self.chunk_size_override,
        )
        self.writers[week_key] = writer
        if not self.chunk_size_used:
            self.chunk_size_used = int(writer.N)
        return writer

    def _find_week_key(self, ts_ms: int) -> str:
        for wk, start_ms, end_ms in self.week_index:
            if start_ms <= ts_ms < end_ms:
                return wk
        raise ValueError(f"No week found for decision timestamp {ts_ms}")

    def add(self, ts_decision_ms: int, seq: np.ndarray, label: np.ndarray):
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_writer(wk)
        writer.add(seq, label)
        self.week_counts[wk] += 1
        if wk not in self.week_decision_span:
            self.week_decision_span[wk] = [ts_decision_ms, ts_decision_ms]
        else:
            span = self.week_decision_span[wk]
            span[0] = min(span[0], ts_decision_ms)
            span[1] = max(span[1], ts_decision_ms)

    def _finalize_week(self, week_key: str):
        writer = self.writers.pop(week_key, None)
        span = self.week_decision_span.pop(week_key, None)
        total_sequences = int(self.week_counts.get(week_key, 0))
        if writer is None:
            # Week already finalised or produced no data.
            return
        writer.flush()
        meta_path = os.path.join(self.out_root, week_key, "meta_week.json")
        chunks_meta = [
            {
                "chunk": int(entry["chunk"]),
                "n": int(entry["n"]),
                "files": dict(entry["files"]),
            }
            for entry in writer.chunks_meta
        ]
        rows_total = int(sum(entry["n"] for entry in chunks_meta))
        meta = {
            "week": week_key,
            "lookback": self.lookback,
            "feature_dim_total": self.feature_dim,
            "feature_dim_core": self.feature_dim - AUX_DIM,
            "label_dim": int(2 * NUM_HORIZONS),
            "horizons_ms": [int(h) for h in HORIZONS_MS],
            "chunk_size_used": int(writer.N),
            "chunks": chunks_meta,
            "chunk_count": int(len(chunks_meta)),
            "rows_total": rows_total,
            "total_sequences": total_sequences,
            "meta_path": os.path.join(week_key, "meta_week.json"),
        }
        if span:
            meta["decision_ts_range"] = {
                "min": int(span[0]),
                "max": int(span[1]),
            }
        if self.pca_meta:
            meta["pca"] = dict(self.pca_meta)
        else:
            meta["pca"] = {
                "applied": False,
                "var_kept": float(PCA_VAR_TARGET),
                "k": 0,
                "model_path": None,
            }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.week_metas[week_key] = meta

    def close_old_writers(self, watermark_ms: int):
        to_close = []
        for wk, writer in list(self.writers.items()):
            _start_ms, end_ms = self.week_bounds[wk]
            if end_ms + GRACE_MS < watermark_ms:
                to_close.append(wk)
        for wk in to_close:
            self._finalize_week(wk)

    def flush_all(self):
        for wk in list(self.writers.keys()):
            self._finalize_week(wk)
        # If any metadata spans remain (e.g. weeks with no chunks), clear them.
        for wk in list(self.week_decision_span.keys()):
            self._finalize_week(wk)
# --------------- dataset-wide processing ---------------
def _compute_dataset_span(pairs: List[Tuple[str, str, str]]):
    if not pairs:
        return None, None
    starts = []
    ends = []
    for wk, _ob_path, _th_path in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(
            _normalise_ob_prefix(f"BTCUSDT_OB_{wk}")
        )
        starts.append(start_dt)
        ends.append(end_dt)
    return min(starts), max(ends)


def _dt_to_epoch_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _build_week_index(pairs: List[Tuple[str, str, str]]):
    index = []
    for wk, _ob_path, _th_path in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(
            _normalise_ob_prefix(f"BTCUSDT_OB_{wk}")
        )
        start_ms = _dt_to_epoch_ms(start_dt)
        end_ms = _dt_to_epoch_ms(end_dt + timedelta(days=1))
        index.append((wk, start_ms, end_ms))
    index.sort(key=lambda x: x[1])
    return index


class EventFeeder:
    def __init__(
        self,
        pairs: List[Tuple[str, str, str]],
        maxsize: int = EVENT_QUEUE_MAXSIZE,
    ):
        self.pairs = list(pairs)
        self.queue: "queue.Queue[Tuple[str, Optional[str], Optional[object]]]" = queue.Queue(maxsize=maxsize)
        self._last_first_ts: Optional[int] = None

    def _put(self, item: Tuple[str, Optional[str], Optional[object]]):
        self.queue.put(item)

    def run(self):
        try:
            for wk, ob_path, th_path in self.pairs:
                raw = BybitRawIter(ob_path, th_path)
                merged = merge_event_time(
                    raw.ob_iter(), _trade_iter_precise(raw.trade_iter()), B=0
                )

                first_event = next(merged, None)
                if first_event is None:
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

                self._put(("first", wk, first_event))
                for event in merged:
                    self._put(("evt", wk, event))
                self._put(("eof", wk, None))

            self._put(("eof", None, None))
        except Exception as exc:
            self._put(("eof", None, exc))


def _stream_core_features(pairs: List[Tuple[str, str, str]]):
    """Yield core feature vectors (z-scored) for the given week pairs."""
    if not pairs:
        return

    fe = FeatureEngine()

    feeder = EventFeeder(pairs)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    q = feeder.queue

    last_global_ts: Optional[int] = None
    try:
        while True:
            kind, wk, payload = q.get()

            if kind == "first":
                if wk is None:
                    raise RuntimeError("Received 'first' marker without a week key")
                if payload is None:
                    continue
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
                raise RuntimeError(f"Unknown feeder message kind: {kind}")

            if event is None:
                continue

            ts_ms, feat_z, _mid, _is_trade, _dt_ms = fe.on_event(event)
            if last_global_ts is not None and ts_ms < last_global_ts:
                raise ValueError(
                    "Non-monotonic timestamps across weeks during PCA stream: "
                    f"week {wk} event {ts_ms} < last {last_global_ts}"
                )
            last_global_ts = int(ts_ms)
            yield np.asarray(feat_z, dtype=np.float32)
    finally:
        producer_thread.join()


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
    pairs: List[Tuple[str, str, str]],
    out_root: str,
    train_weeks: List[str],
    target_var: float,
    sample_limit: int,
    batch_size: int,
    model_filename: str,
):
    meta = {
        "applied": False,
        "var_kept": float(target_var),
        "k": 0,
        "model_path": None,
    }

    if target_var <= 0.0:
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
        nonlocal pending, fitted_rows
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
        pending = []

    def ensure_ipca(force: bool = False):
        nonlocal ipca, sample_array, pad_rows, n_components, fitted_rows
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
    pairs: List[Tuple[str, str, str]],
    out_root: str,
    pca_meta: dict,
    split_info: Optional[Dict[str, List[str]]] = None,
):
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
            pca_summary = _summarise_pca_meta({
                "applied": False,
                "var_kept": pca_summary.get("var_kept", PCA_VAR_TARGET),
            })

    fe = FeatureEngine()
    labeler = LabelBuilder(delta_ms=5, horizons_ms=HORIZONS_MS)

    tokens_buf: deque = deque(maxlen=LOOKBACK)
    pending_seqs: deque = deque()

    F = None
    router: WeekWriterRouter = None  # type: ignore
    total_sequences = 0

    ds_start, ds_end = _compute_dataset_span(pairs)
    start_iso = ds_start.date().isoformat() if ds_start else None
    end_iso = ds_end.date().isoformat() if ds_end else None

    week_index = _build_week_index(pairs)

    print(
        f"[start] ingest weeks={len(pairs)} L={LOOKBACK} budget={RAM_BUDGET}MB"
    )

    last_global_ts: Optional[int] = None

    feeder = EventFeeder(pairs)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    q = feeder.queue

    week_total = len(pairs)
    week_counter = 0

    while True:
        kind, wk, payload = q.get()

        if kind == "first":
            if wk is None:
                raise RuntimeError("Received 'first' marker without a week key")
            week_counter += 1
            print(f"[week ] {week_counter}/{week_total} {wk}")
            if payload is None:
                print(f"[skip ] {wk} yielded no events")
                continue
            ts_first = _event_ts(payload)
            if last_global_ts is not None and ts_first < last_global_ts:
                raise ValueError(
                    "Non-monotonic timestamps across weeks: "
                    f"week {wk} starts at {ts_first} < last seen {last_global_ts}"
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
            raise RuntimeError(f"Unknown feeder message kind: {kind}")

        if event is None:
            continue

        ts_ms, feat_z, mid, is_trade, dt_ms = fe.on_event(event)
        feat_core = feat_z
        if pca_components is not None and pca_mean is not None:
            if np.asarray(feat_z).shape[-1] != pca_mean.shape[0]:
                raise ValueError(
                    f"PCA mean/components dimension {pca_mean.shape[0]} does not match "
                    f"feature dimension {np.asarray(feat_z).shape[-1]}"
                )
            centered = np.asarray(feat_z, dtype=np.float32, copy=False) - pca_mean
            feat_core = np.dot(centered, pca_components.T).astype(np.float32, copy=False)
        tok = build_token(fe, feat_core, is_trade, dt_ms)
        if F is None:
            F = tok.shape[0]
            router = WeekWriterRouter(
                out_root,
                LOOKBACK,
                F,
                RAM_BUDGET,
                CHUNK_SIZE,
                week_index,
                pca_meta=pca_summary,
            )
        tokens_buf.append(tok)

        seq = build_sequence_from_tokens(tokens_buf, LOOKBACK)
        ts_decision = int(ts_ms)
        pending_seqs.append((ts_decision, seq.astype(np.float32, copy=False)))
        labeler.on_decision(ts_decision)

        matured = labeler.on_event(int(ts_ms), float(mid))
        for yy in matured:
            if not pending_seqs:
                raise RuntimeError(
                    "Matured label available but no pending sequences to pair"
                )
            if router is None:
                raise RuntimeError("Router not initialised before label maturity")
            ts_ready, seq_ready = pending_seqs.popleft()
            router.add(ts_ready, seq_ready, yy.astype(np.float32, copy=False))
            total_sequences += 1

        last_global_ts = int(ts_ms)

        if router is not None:
            router.close_old_writers(int(ts_ms))

    producer_thread.join()

    if router is not None:
        router.flush_all()

    feature_dim_total = None if F is None else int(F)
    feature_dim_core = None if F is None else int(F - AUX_DIM)
    label_dim = int(2 * NUM_HORIZONS)
    week_meta_records = {} if router is None else dict(router.week_metas)
    weeks_in_order = [wk for wk, _ob, _th in pairs]
    week_counts = {
        wk: int(0 if router is None else router.week_counts.get(wk, 0))
        for wk in weeks_in_order
    }
    total_chunks = sum(
        int(week_meta.get("chunk_count", len(week_meta.get("chunks", []))))
        for week_meta in week_meta_records.values()
    )
    rows_via_week_metas = sum(
        int(week_meta.get("rows_total", week_meta.get("total_sequences", 0)))
        for week_meta in week_meta_records.values()
    )
    weeks_meta_paths = {
        wk: week_meta_records[wk].get("meta_path", os.path.join(wk, "meta_week.json"))
        for wk in week_meta_records.keys()
    }

    meta = {
        "dataset_start": start_iso,
        "dataset_end": end_iso,
        "weeks": weeks_in_order,
        "lookback": int(LOOKBACK),
        "feature_dim_total": feature_dim_total,
        "feature_dim_core": feature_dim_core,
        "aux_tail": ["log_dt_ms", "is_trade", "events_100ms"],
        "dtype": "float32",
        "ram_budget_mb": int(RAM_BUDGET),
        "chunk_size_used": 0 if (router is None or router.chunk_size_used == 0) else int(router.chunk_size_used),
        "aux_dim": int(AUX_DIM),
        "label_dim": label_dim,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "core_dtype": "float32",
        "total_sequences": int(total_sequences),
        "week_counts": week_counts,
        "total_chunks": int(total_chunks),
        "rows_total_from_weeks": int(rows_via_week_metas),
        "weeks_meta": weeks_meta_paths,
    }
    meta["pca"] = dict(pca_summary)
    if pca_var_ratio is not None:
        meta["pca"]["explained_variance_ratio"] = [float(x) for x in pca_var_ratio]
    if split_info:
        meta["splits"] = {
            key: list(vals) for key, vals in split_info.items()
        }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    chunk_summary = 0 if router is None else sum(router.week_counts.values())

    print(
        f"[done ] dataset weeks={len(pairs)} total_seqs={total_sequences} "
        f"L={LOOKBACK} F={feature_dim_total or 0} chunkN={meta['chunk_size_used']} "
        f"routed={chunk_summary}"
    )
    print(
        f"[pca  ] summary applied={pca_summary['applied']} "
        f"var_kept={pca_summary['var_kept']:.4f} k={pca_summary['k']} "
        f"model={pca_summary['model_path']}"
    )

# --------------- driver ----------------
def main():
    ensure_dir(OUT_ROOT)
    pairs = pair_weeks(OB_DIR, TH_DIR)
    
    if not pairs:
        print(f"No week pairs found under OB_DIR={OB_DIR} and TH_DIR={TH_DIR}")
        return
    
    raw_last_week_end = os.environ.get("BYBIT_LAST_WEEK_END", "")
    explicit_last_week_end = raw_last_week_end.strip()
    normalized_last_week_end = explicit_last_week_end.lower()

    if normalized_last_week_end in ("", "latest", "auto"):
        max_end_dt = max(
            _parse_week_key_any(_normalise_ob_prefix(f"BTCUSDT_OB_{wk}"))[1]
            for wk, _ob, _th in pairs
        )
        last_week_end = max_end_dt.date().isoformat()
    else:
        last_week_end = explicit_last_week_end

    pairs = _slice_last_weeks_pairs(pairs, last_week_end, KEEP_WEEKS)

    _assert_week_order(pairs)

    print(f"[plan ] weeks={len(pairs)} workers={WORKERS} "
          f"RAM={RAM_BUDGET}MB chunk_size={CHUNK_SIZE if CHUNK_SIZE>0 else 'auto'}")

    print(f"[paths] OB_DIR={OB_DIR}")
    print(f"[paths] TH_DIR={TH_DIR}")
    print(f"[out  ] OUT_ROOT={OUT_ROOT}")

    if WORKERS > 1:
        print("[warn ] WORKERS>1 requested but process_all runs sequentially; using 1 worker")

    train_weeks, val_weeks, test_weeks = classify_week_splits(pairs)
    split_info = {
        "train": train_weeks,
        "val": val_weeks,
        "test": test_weeks,
    }
    print(
        f"[split] train={len(train_weeks)} val={len(val_weeks)} test={len(test_weeks)}"
    )
    pca_fit_meta = maybe_fit_pca_model(
        pairs,
        OUT_ROOT,
        train_weeks,
        PCA_VAR_TARGET,
        PCA_MAX_SAMPLE_ROWS,
        PCA_BATCH_SIZE,
        PCA_MODEL_FILENAME,
    )

    process_all(pairs, OUT_ROOT, pca_fit_meta, split_info=split_info)

if __name__ == "__main__":
    main()
