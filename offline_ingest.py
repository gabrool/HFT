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
from typing import List, Tuple
from collections import deque
import numpy as np
from datetime import datetime

# ---------------- config ----------------
OB_DIR      = os.environ.get("BYBIT_OB_DIR",   "/home/gabrool/Documents/OB")
TH_DIR      = os.environ.get("BYBIT_TH_DIR",   "/home/gabrool/Documents/TH")
OUT_ROOT    = os.environ.get("BYBIT_OUT_ROOT", "/media/gabrool/Expansion/Gabriel/bybit_offline_dt")

# Week selection: anchor on a known last-week end date, keep the last K weeks
LAST_WEEK_END = os.environ.get("BYBIT_LAST_WEEK_END", "2025-08-27")  # ISO date (YYYY-MM-DD)
KEEP_WEEKS    = int(os.environ.get("BYBIT_KEEP_WEEKS", "24"))

# Parallelism / sequence geometry
WORKERS     = int(os.environ.get("BYBIT_WORKERS", "8"))
LOOKBACK    = int(os.environ.get("BYBIT_LOOKBACK", "1024"))

# Memory & chunking
RAM_BUDGET  = int(os.environ.get("BYBIT_RAM_BUDGET_MB", "512"))
CHUNK_SIZE  = int(os.environ.get("BYBIT_CHUNK_SIZE", "0"))

AUX_DIM     = 3


# import your training utilities
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from CMSSL16 import (
    FeatureEngine,
    LabelBuilder,
    merge_event_time,
    build_sequence_from_tokens,
)  # reuse exactly

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

def ob_iter_plain(jsonl_path: str):
    seq = 0
    with open(jsonl_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line: continue
            obj = fast_json_loads(line)
            ts = int(obj.get("ts", obj.get("cts", 0)))
            seq += 1
            yield ts, seq, obj

def th_iter_plain(csv_path: str):
    seq = 0
    with open(csv_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq += 1
            ts = int(float(row["timestamp"]) * 1000)
            row["seq"] = seq
            yield ts, seq, row

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
        bytes_per_seq = (self.L * self.F_core * 4) + (self.L * AUX_DIM * 4) + (2 * 4)
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            self.N = max(256, int((ram_budget_mb * 1024 * 1024) // bytes_per_seq))
        self.N = min(self.N, 4096)

        # preallocate separate buffers
        self.X_core = np.empty((self.N, self.L, self.F_core), dtype=np.float32)  # cast on flush
        self.X_aux  = np.empty((self.N, self.L, AUX_DIM),     dtype=np.float32)  # keep fp32
        self.Y      = np.empty((self.N, 2), dtype=np.float32)
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

# --------------- core per-week ---------------
def process_week(wk: str, ob_path: str, th_path: str, out_root: str):
    out_dir = os.path.join(out_root, wk)
    ensure_dir(out_dir)

    fe = FeatureEngine()
    labeler = LabelBuilder(delta_ms=5, horizon_ms=1000)

    tokens_buf: deque = deque(maxlen=LOOKBACK)  # rolling tokens
    pending_seqs: deque = deque()              # one seq per decision (FIFO)

    F = None
    cw = None  # ChunkWriter will be created once F is known

    print(f"[start] {wk} L={LOOKBACK} budget={RAM_BUDGET}MB")

    merged = merge_event_time(ob_iter_plain(ob_path), th_iter_plain(th_path), B=0)

    for e in merged:
        ts_ms, feat_z, mid, is_trade, dt_ms = fe.on_event(e)
        tok = build_token(fe, feat_z, is_trade, dt_ms)
        if F is None:
            F = tok.shape[0]
            ensure_dir(out_dir)
            cw = ChunkWriter(out_dir, LOOKBACK, F, RAM_BUDGET, CHUNK_SIZE)
        tokens_buf.append(tok)

        seq = build_sequence_from_tokens(tokens_buf, LOOKBACK)
        pending_seqs.append(seq.astype(np.float32, copy=False))
        labeler.on_decision(int(ts_ms))

        matured = labeler.on_event(int(ts_ms), float(mid))
        for yy in matured:
            if not pending_seqs:
                raise RuntimeError("Matured label available but no pending sequences to pair")
            cw.add(pending_seqs.popleft(), yy.astype(np.float32, copy=False))

    # flush remainder
    if cw is not None:
        cw.flush()

    # meta
    chunks = [] if cw is None else cw.chunks_meta
    feature_dim = 0 if F is None else int(F)
    meta = {
        "week": wk,
        "lookback": int(LOOKBACK),
        "feature_dim": feature_dim,
        "aux_tail": ["log_dt_ms", "is_trade", "events_100ms"],
        "chunks": chunks,
        "dtype": "float32",
        "ram_budget_mb": int(RAM_BUDGET),
        "chunk_size_used": 0 if cw is None else int(cw.N),
        "feature_core": 0 if F is None else int(F - AUX_DIM),
        "aux_dim": int(AUX_DIM),
        "core_dtype": "float32"
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_seqs = sum(c["n"] for c in chunks)
    print(f"[done ] {wk} chunks={len(chunks)} total_seqs={total_seqs} "
          f"L={LOOKBACK} F={feature_dim} chunkN={meta['chunk_size_used']}")

# --------------- driver ----------------
def main():
    ensure_dir(OUT_ROOT)
    pairs = pair_weeks(OB_DIR, TH_DIR)
    
    if not pairs:
        print(f"No week pairs found under OB_DIR={OB_DIR} and TH_DIR={TH_DIR}")
        return
    
    pairs = _slice_last_weeks_pairs(pairs, LAST_WEEK_END, KEEP_WEEKS)

    _assert_week_order(pairs)

    print(f"[plan ] weeks={len(pairs)} workers={WORKERS} "
          f"RAM={RAM_BUDGET}MB chunk_size={CHUNK_SIZE if CHUNK_SIZE>0 else 'auto'}")

    print(f"[paths] OB_DIR={OB_DIR}")
    print(f"[paths] TH_DIR={TH_DIR}")
    print(f"[out  ] OUT_ROOT={OUT_ROOT}")
    print(f"[plan ] weeks={len(pairs)} workers={WORKERS} "
          f"RAM={RAM_BUDGET}MB chunk_size={CHUNK_SIZE if CHUNK_SIZE>0 else 'auto'}")

    if WORKERS <= 1:
        for i, (wk, ob_p, th_p) in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] {wk}")
            try:
                process_week(wk, ob_p, th_p, OUT_ROOT)
            except Exception as e:
                print(f"[error] {wk}: {e}")
    else:
        # parallel-by-week (be careful on SATA/external SSDs)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            from tqdm import tqdm
        except Exception:
            class tqdm:
                def __init__(self, total=0, desc="", dynamic_ncols=True): self.n=0; self.total=total; print(f"{desc}: 0/{total}")
                def update(self, k): self.n+=k; print(f"progress: {self.n}/{self.total}")
                def close(self): pass
        with ProcessPoolExecutor(max_workers=WORKERS) as ex, tqdm(total=len(pairs), desc="Weeks done", dynamic_ncols=True) as pbar:
            futs = {ex.submit(process_week, wk, ob_p, th_p, OUT_ROOT): wk for (wk, ob_p, th_p) in pairs}
            for fut in as_completed(futs):
                wk = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[error] {wk}: {e}")
                pbar.update(1)

if __name__ == "__main__":
    main()
