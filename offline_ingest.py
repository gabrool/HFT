#!/usr/bin/env python3
"""
Decision-time ingest (memory-safe):
- Snapshot ONE [LOOKBACK, F] sequence at each decision time.
- Use a RAM budget to auto-size chunked writes (avoid huge in-RAM lists).
- Minimal logging; optional heartbeat file.

Env (defaults are SSD-friendly):
  BYBIT_OB_DIR=/home/gabrool/Documents/OB
  BYBIT_TH_DIR=/home/gabrool/Documents/TH
  BYBIT_OUT_ROOT=/media/gabrool/Expansion/Gabriel/bybit_offline_dt
  BYBIT_MAX_WEEKS=0
  BYBIT_WORKERS=1
  BYBIT_LOOKBACK=1024
  BYBIT_DECISION_INTERVAL_MS=500   # start practical; lower later if desired
  BYBIT_RAM_BUDGET_MB=512          # memory budget for one chunk
  BYBIT_CHUNK_SIZE=0               # 0 = auto from budget; else fixed size
  BYBIT_HEARTBEAT_SEC=0
"""

import os, sys, csv, json, time, re
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

# Hybrid gating
DECISION_DT = int(os.environ.get("BYBIT_DECISION_INTERVAL_MS", "250"))  # backstop
MIN_GAP_MS  = int(os.environ.get("BYBIT_MIN_GAP_MS",          "250"))   # minimum spacing
REQUIRE_EVENT_FOR_EARLY = bool(int(os.environ.get("BYBIT_REQUIRE_EVENT", "1")))  # 1 = early fires require event

# Memory & chunking
RAM_BUDGET  = int(os.environ.get("BYBIT_RAM_BUDGET_MB", "512"))
CHUNK_SIZE  = int(os.environ.get("BYBIT_CHUNK_SIZE", "0"))
HEARTBEAT   = float(os.environ.get("BYBIT_HEARTBEAT_SEC", "0"))

AUX_DIM     = 3


# import your training utilities
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from CMSSL16 import FeatureEngine, LabelBuilder, merge_event_time  # reuse exactly

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

def week_key_from_stem(stem: str, prefix: str) -> str:
    return stem[len(prefix):] if stem.startswith(prefix) else stem

def pair_weeks(ob_dir: str, th_dir: str) -> List[Tuple[str, str, str]]:
    ob_files = list_glob(ob_dir, "BTCUSDT_OB_*.jsonl")
    th_files = list_glob(th_dir, "BTCUSDT_TH_*.csv")
    ob_map = { week_key_from_stem(os.path.splitext(os.path.basename(p))[0], "BTCUSDT_OB_"): p for p in ob_files }
    th_map = { week_key_from_stem(os.path.splitext(os.path.basename(p))[0], "BTCUSDT_TH_"): p for p in th_files }
    common = sorted(set(ob_map) & set(th_map))
    return [(wk, ob_map[wk], th_map[wk]) for wk in common]

def _parse_week_key(wk: str):
    # Supports "DD-MM-YYYY-to-DD-MM-YYYY" and "YYYY-MM-DD-to-YYYY-MM-DD"
    m = re.match(r"(\d{2}-\d{2}-\d{4})-to-(\d{2}-\d{2}-\d{4})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%d-%m-%Y")
        e = datetime.strptime(m.group(2), "%d-%m-%Y")
        return s, e
    m = re.match(r"(\d{4}-\d{2}-\d{2})-to-(\d{4}-\d{2}-\d{2})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%Y-%m-%d")
        e = datetime.strptime(m.group(2), "%Y-%m-%d")
        return s, e
    raise ValueError(f"Unrecognized week key format: {wk}")

def _slice_last_weeks(pairs, last_end_iso: str, k: int):
    target_end = datetime.strptime(last_end_iso, "%Y-%m-%d")
    rows = []
    for wk, ob_p, th_p in pairs:
        s, e = _parse_week_key(wk)
        rows.append((e, s, wk, ob_p, th_p))
    rows.sort()  # by end date

    # find the index of the row whose end date == target_end (or the nearest <= if exact not present)
    idx = max(i for i,(e,_,_,_,_) in enumerate(rows) if e <= target_end)
    lo = max(0, idx - (k - 1))
    sel = rows[lo:idx+1]
    return [(wk, ob_p, th_p) for (_,_,wk,ob_p,th_p) in sel]

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
    # exact tail order: [dt_ms, is_trade, events_100ms]
    events_100ms = fe.event_density_100ms()
    return np.concatenate([
        np.asarray(feat_z, dtype=np.float32),
        np.array([dt_ms, float(is_trade), events_100ms], dtype=np.float32)
    ], axis=0).astype(np.float32)

def pad_left_repeat(x: np.ndarray, L: int) -> np.ndarray:
    k, F = x.shape
    if k >= L: return x[-L:]
    if k == 0: return np.zeros((L, F), dtype=np.float32)
    pad = np.repeat(x[:1], L - k, axis=0)
    return np.concatenate([pad, x], axis=0)

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
    progress_path = os.path.join(out_dir, "progress.txt")

    fe = FeatureEngine()
    labeler = LabelBuilder(delta_ms=5, horizon_ms=1000)

    tokens_buf: deque = deque(maxlen=LOOKBACK)  # rolling tokens
    pending_seqs: deque = deque()              # one seq per decision (FIFO)
    MAX_PENDING = max(1000, int(1000 / max(1, DECISION_DT)) * 5)  # guardrail

    F = None
    cw = None  # ChunkWriter will be created once F is known

    next_hb = time.time() + HEARTBEAT if HEARTBEAT > 0 else float("inf")
    last_decision_ts = -10**15

    print(f"[start] {wk} gate={DECISION_DT}ms L={LOOKBACK} budget={RAM_BUDGET}MB")

    merged = merge_event_time(ob_iter_plain(ob_path), th_iter_plain(th_path), B=0)

    for e in merged:
        ts_ms, feat_z, mid, is_trade, dt_ms = fe.on_event(e)
        tok = build_token(fe, feat_z, is_trade, dt_ms)
        if F is None:
            F = tok.shape[0]
            ensure_dir(out_dir)
            cw = ChunkWriter(out_dir, LOOKBACK, F, RAM_BUDGET, CHUNK_SIZE)
        tokens_buf.append(tok)

        # --- hybrid gating (event-aware with 250 ms backstop) ---
        # Fire if:
        #  (a) a trade happened OR mid changed, AND at least MIN_GAP_MS since last decision
        #  (b) OR the backstop (DECISION_DT) elapsed since last decision
        trigger_event = bool(is_trade) or (getattr(process_week, "_last_mid", None) != mid)
        enough_gap = (ts_ms - last_decision_ts) >= MIN_GAP_MS
        backstop = (ts_ms - last_decision_ts) >= DECISION_DT

        should_decide = False
        if REQUIRE_EVENT_FOR_EARLY:
            if (trigger_event and enough_gap) or backstop:
                should_decide = True
        else:
            # allow time-only early decision if enough_gap
            if enough_gap and (trigger_event or backstop):
                should_decide = True

        if should_decide:
            seq = pad_left_repeat(np.vstack(tokens_buf), LOOKBACK)
            pending_seqs.append(seq.astype(np.float32))
            labeler.on_decision(int(ts_ms))
            last_decision_ts = ts_ms

        setattr(process_week, "_last_mid", mid)

        # matured labels -> pair with oldest decision
        matured = labeler.on_event(int(ts_ms), float(mid))
        if matured:
            for yy in matured:
                if not pending_seqs:
                    continue  # shouldn't happen
                cw.add(pending_seqs.popleft(), np.asarray(yy, dtype=np.float32))

        # guardrail: avoid unbounded pending growth (e.g., if DECISION_DT=0)
        if len(pending_seqs) > MAX_PENDING:
            # drop oldest to prevent RAM blowup; also record a warning
            dropped = len(pending_seqs) - MAX_PENDING
            for _ in range(dropped):
                pending_seqs.popleft()
            try:
                with open(os.path.join(out_dir, "warnings.txt"), "a") as w:
                    w.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} dropped {dropped} pending seqs\n")
            except Exception:
                pass

        # heartbeat to file only (no console spam)
        if time.time() >= next_hb:
            try:
                with open(progress_path, "w") as hb:
                    hb.write(f"chunks={0 if cw is None else cw.cid} "
                             f"in_chunk={0 if cw is None else cw.i} "
                             f"pending={len(pending_seqs)} "
                             f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception:
                pass
            next_hb = time.time() + HEARTBEAT

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
        "aux_tail": ["dt_ms", "is_trade", "events_100ms"],
        "chunks": chunks,
        "dtype": "float32",
        "decision_interval_ms": int(DECISION_DT),
        "ram_budget_mb": int(RAM_BUDGET),
        "chunk_size_used": 0 if cw is None else int(cw.N),
        "feature_core": int(F - AUX_DIM),
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
    
    pairs = _slice_last_weeks(pairs, LAST_WEEK_END, KEEP_WEEKS)

    print(f"[plan ] weeks={len(pairs)} workers={WORKERS} gate={DECISION_DT}ms "
        f"RAM={RAM_BUDGET}MB chunk_size={CHUNK_SIZE if CHUNK_SIZE>0 else 'auto'}")

    print(f"[paths] OB_DIR={OB_DIR}")
    print(f"[paths] TH_DIR={TH_DIR}")
    print(f"[out  ] OUT_ROOT={OUT_ROOT}")
    print(f"[plan ] weeks={len(pairs)} workers={WORKERS} gate={DECISION_DT}ms "
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
