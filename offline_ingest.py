#!/usr/bin/env python3
"""
Decision-time ingest (memory-safe):
- Snapshot ONE [LOOKBACK, F] sequence at each decision time.
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


class TokenRingBuffer:
    def __init__(self, lookback: int, feature_dim: int):
        self.lookback = int(lookback)
        self.feature_dim = int(feature_dim)
        self.tokens = np.empty((self.lookback, self.feature_dim), dtype=np.float32)
        self.cursor = 0
        self.count = 0

    def append(self, token: np.ndarray) -> None:
        self.tokens[self.cursor] = token
        self.cursor = (self.cursor + 1) % self.lookback
        self.count = min(self.count + 1, self.lookback)

    def overwrite_latest(self, token: np.ndarray) -> None:
        if self.count <= 0:
            raise RuntimeError("Cannot overwrite latest token in an empty ring buffer")
        latest_idx = (self.cursor - 1) % self.lookback
        self.tokens[latest_idx] = token

    def snapshot(self, ts_decision_ms: int) -> "TokenBufferSnapshot":
        if self.count <= 0:
            raise RuntimeError("Cannot snapshot an empty token ring buffer")
        return TokenBufferSnapshot(
            ts_decision_ms=int(ts_decision_ms),
            source=self,
            cursor=int(self.cursor),
            count=int(self.count),
        )


@dataclass
class TokenBufferSnapshot:
    ts_decision_ms: int
    source: TokenRingBuffer
    cursor: int
    count: int

    def refresh(self, ts_decision_ms: int) -> None:
        self.ts_decision_ms = int(ts_decision_ms)
        self.cursor = int(self.source.cursor)
        self.count = int(self.source.count)

    @property
    def lookback(self) -> int:
        return self.source.lookback

    @property
    def feature_dim(self) -> int:
        return self.source.feature_dim


def _parse_requested_weeks(raw: str) -> List[str]:
    items = [wk.strip() for wk in re.split(r"[\s,]+", raw) if wk.strip()]
    # Preserve potential duplicates in the env var for explicit validation later
    return items

_EXT_PRIORITY = {
    ".zip": 0,
    ".gz": 1,
    ".jsonl": 2,
    ".csv": 3,
}

# Daily OB names must start with YYYY-MM-DD_BTCUSDT_, include "ob" in the
# stem (to avoid unrelated BTCUSDT zips), and end in .zip.
OB_DAILY_RE = re.compile(
    r"^(?P<d>\d{4}-\d{2}-\d{2})_BTCUSDT_.*ob.*\.zip$",
    re.IGNORECASE,
)
TH_DAILY_RE = re.compile(
    r"^BTCUSDT(?P<d>\d{4}-\d{2}-\d{2})\.csv(\.(gz|gzip))?$",
    re.IGNORECASE,
)


def _choose_preferred_daily_file(day: date, candidates: List[str], side: str) -> str:
    def _ext_rank(path: str) -> int:
        lower_path = str(path).lower()
        if side.upper() == "TH":
            if lower_path.endswith(".csv.gz"):
                return _EXT_PRIORITY[".gz"]
            if lower_path.endswith(".csv.gzip"):
                return _EXT_PRIORITY[".gz"] + 1
            if lower_path.endswith(".csv"):
                return _EXT_PRIORITY[".csv"]
            return 4

        if side.upper() == "OB":
            if lower_path.endswith(".data.zip"):
                return _EXT_PRIORITY[".zip"]
            p = Path(path)
            return _EXT_PRIORITY.get(p.suffix.lower(), 4)

        p = Path(path)
        return _EXT_PRIORITY.get(p.suffix.lower(), 4)

    def _sort_key(path: str):
        p = Path(path)
        return (_ext_rank(path), p.name, str(p))

    chosen = min(candidates, key=_sort_key)
    if len(candidates) > 1:
        alternatives = sorted([p for p in candidates if p != chosen], key=_sort_key)
        print(
            f"Warning: duplicate {side} files for day '{day.isoformat()}'; "
            f"chosen='{chosen}', alternatives={alternatives}"
        )
    return chosen


def _build_ob_daily_map(ob_dir: str) -> Dict[date, str]:
    groups: Dict[date, List[str]] = defaultdict(list)
    for p in Path(ob_dir).iterdir():
        if not p.is_file():
            continue
        m = OB_DAILY_RE.match(p.name)
        if not m:
            continue
        day = _parse_ymd_date(m.group("d"))
        groups[day].append(str(p))

    return {
        day: _choose_preferred_daily_file(day, candidates, "OB")
        for day, candidates in groups.items()
    }


def _build_th_daily_map(th_dir: str) -> Dict[date, str]:
    groups: Dict[date, List[str]] = defaultdict(list)
    for p in Path(th_dir).iterdir():
        if not p.is_file():
            continue
        m = TH_DAILY_RE.match(p.name)
        if not m:
            continue
        day = _parse_ymd_date(m.group("d"))
        groups[day].append(str(p))

    return {
        day: _choose_preferred_daily_file(day, candidates, "TH")
        for day, candidates in groups.items()
    }

def extract_week_key_from_name(name: str) -> str:
    m = re.search(r"\d{2}-\d{2}-\d{4}-to-\d{2}-\d{2}-\d{4}", name)
    if m:
        return m.group(0)
    raise ValueError(f"Could not extract week key from file name: {name}")


def _parse_ymd_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _week_key_from_dates(d0: date, d6: date) -> str:
    return f"{d0.strftime('%d-%m-%Y')}-to-{d6.strftime('%d-%m-%Y')}"


def _group_common_days_into_weeks(common_days: List[date], *, strict: bool = True) -> List[List[date]]:
    """
    Partition sorted common days into non-overlapping 7-day blocks.

    Args:
        common_days: Sorted candidate dates for week grouping.
        strict: If True, raise on any day-to-day gap inside a 7-day block.
            If False, skip invalid blocks and continue.

    Returns:
        A list of valid week blocks (each block has exactly 7 dates).
    """
    groups: List[List[date]] = []
    assert ONE_DAY.total_seconds() > 0, "ONE_DAY must be positive and non-zero"
    total_days = len(common_days)
    usable_days = (total_days // 7) * 7

    if usable_days < total_days:
        trailing = common_days[usable_days:]
        print(
            "Warning: ignoring trailing partial week "
            f"({len(trailing)} day(s)): {[d.isoformat() for d in trailing]}"
        )

    for start_idx in range(0, usable_days, 7):
        block = common_days[start_idx:start_idx + 7]
        gap_idx = None
        for i in range(1, len(block)):
            expected = block[i - 1] + ONE_DAY
            if block[i] != expected:
                gap_idx = i
                break

        if gap_idx is not None:
            prev_day = block[gap_idx - 1]
            curr_day = block[gap_idx]
            expected_day = prev_day + ONE_DAY
            msg = (
                "Non-consecutive days inside 7-day block: "
                f"block_idx={start_idx // 7}, "
                f"span={block[0].isoformat()}..{block[-1].isoformat()}, "
                f"expected={expected_day.isoformat()}, got={curr_day.isoformat()}, "
                f"full_block={[d.isoformat() for d in block]}"
            )
            if strict:
                raise ValueError(msg)
            print(f"Warning: {msg}; skipping block")
            continue

        groups.append(block)

    return groups

def _parse_week_key_any(wk: str):
    m = re.fullmatch(r"(\d{2}-\d{2}-\d{4})-to-(\d{2}-\d{2}-\d{4})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%d-%m-%Y")
        e = datetime.strptime(m.group(2), "%d-%m-%Y")
        return s, e, wk
    raise ValueError(
        "Unrecognized week key format. Expected 'DD-MM-YYYY-to-DD-MM-YYYY', "
        f"got: {wk!r}"
    )

WeekPaths = List[str]
WeekPair = Tuple[str, WeekPaths, WeekPaths]


def pair_weeks(ob_dir: str, th_dir: str) -> List[WeekPair]:
    """
    Discover daily inputs and emit 7-day week groups.

    In trade-enabled mode (BYBIT_USE_TRADES=1), this enforces exact OB/TH daily
    parity before grouping and emits aligned `ob_paths`/`th_paths` lists.
    In OB-only mode (BYBIT_USE_TRADES=0), this groups only OB daily files and
    emits `th_paths=[]` for each returned week.

    Returns:
        List of (week_key, ob_paths, th_paths), ordered by block end date ascending.
        `ob_paths` is an ordered 7-element file-path list (one per day in each
        week block). `th_paths` is an ordered 7-element list in trade-enabled
        mode, or an empty list in OB-only mode.
    """
    ob_by_day = _build_ob_daily_map(ob_dir)

    if not ob_by_day:
        raise ValueError(
            "No OB daily files found. Expected filenames like "
            "'2024-01-15_BTCUSDT_orderbook.ob.zip' (YYYY-MM-DD_BTCUSDT_*ob*.zip)."
        )

    if USE_TRADES:
        th_by_day = _build_th_daily_map(th_dir)
        if not th_by_day:
            raise ValueError(
                "No TH daily files found. Expected filenames like "
                "'BTCUSDT2024-01-15.csv.gz' (BTCUSDTYYYY-MM-DD.csv[.gz|.gzip])."
            )

        missing_th_days = sorted(set(ob_by_day) - set(th_by_day))
        missing_ob_days = sorted(set(th_by_day) - set(ob_by_day))

        def _format_missing_days(days: List[date]) -> str:
            days_fmt = [d.strftime("%Y-%m-%d") for d in days]
            if len(days_fmt) <= 10:
                return f"count={len(days_fmt)}, full={days_fmt}"
            sample = days_fmt[:5] + ["..."] + days_fmt[-5:]
            return f"count={len(days_fmt)}, sample={sample}"

        if missing_th_days or missing_ob_days:
            raise ValueError(
                "Daily ingest requires exact OB/TH date parity before week grouping. "
                f"Missing TH days (present in OB): {_format_missing_days(missing_th_days)}. "
                f"Missing OB days (present in TH): {_format_missing_days(missing_ob_days)}."
            )

        common_days = sorted(set(ob_by_day) & set(th_by_day))
        if not common_days:
            return []
    else:
        th_by_day = {}
        common_days = sorted(ob_by_day)
        if not common_days:
            return []

    week_blocks = _group_common_days_into_weeks(common_days, strict=bool(BYBIT_STRICT_DATA))
    rows = []
    for block in week_blocks:
        week_key = _week_key_from_dates(block[0], block[-1])
        ob_paths = [ob_by_day[d] for d in block]
        th_paths = [th_by_day[d] for d in block] if USE_TRADES else []
        rows.append((block[-1], block[0], week_key, ob_paths, th_paths))

    rows.sort()
    return [(wk, ob_p, th_p) for (_end, _start, wk, ob_p, th_p) in rows]


def _week_path_label(paths: List[str]) -> str:
    if not paths:
        return "[]"
    return f"{os.path.basename(paths[0])} ... {os.path.basename(paths[-1])} ({len(paths)} files)"


def _assert_week_order(pairs: List[WeekPair]):
    if not pairs:
        return

    parsed = []
    for wk, ob_p, th_p in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(wk)
        parsed.append((start_dt, end_dt, ob_p, th_p, wk))

    for idx in range(1, len(parsed)):
        _prev_start, prev_end, prev_ob, prev_th, _prev_wk = parsed[idx - 1]
        _curr_start, curr_end, curr_ob, curr_th, _curr_wk = parsed[idx]
        if curr_end <= prev_end:
            raise ValueError(
                "Week files must be strictly increasing by end date: "
                f"'{_week_path_label(curr_ob)}'/'{_week_path_label(curr_th)}' (end={curr_end.date()}) "
                f"not after '{_week_path_label(prev_ob)}'/'{_week_path_label(prev_th)}' (end={prev_end.date()})"
            )


def _assert_weeks_consecutive(pairs: List[WeekPair]):
    if len(pairs) < 2:
        return

    parsed = []
    for wk, _ob_p, _th_p in pairs:
        start_dt, end_dt, _ = _parse_week_key_any(wk)
        parsed.append((start_dt, end_dt, wk))

    parsed.sort(key=lambda row: row[1])
    assert ONE_DAY.total_seconds() > 0, "ONE_DAY must be positive and non-zero"
    for idx in range(1, len(parsed)):
        prev_start, prev_end, prev_wk = parsed[idx - 1]
        next_start, next_end, next_wk = parsed[idx]
        expected_next_start = prev_end.date() + ONE_DAY
        if next_start.date() != expected_next_start:
            relation = "gap" if next_start.date() > expected_next_start else "overlap"
            raise ValueError(
                f"Weeks must be consecutive with no gaps/overlaps; detected {relation} between "
                f"'{prev_wk}' ({prev_start.date()}–{prev_end.date()}) and "
                f"'{next_wk}' ({next_start.date()}–{next_end.date()})."
            )



def build_four_week_pipeline_splits(
    weeks_in_order: List[str],
    week_meta_records: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    if len(weeks_in_order) != 4:
        raise ValueError(
            f"build_four_week_pipeline_splits requires exactly 4 weeks; got {len(weeks_in_order)}."
        )

    def _decision_range(week_key: str) -> Tuple[int, int]:
        wk_meta = week_meta_records.get(week_key)
        if not wk_meta or "decision_ts_range" not in wk_meta:
            raise ValueError(
                f"Missing decision_ts_range for week '{week_key}'; cannot derive four-week split boundaries."
            )
        decision_range = wk_meta["decision_ts_range"]
        start = int(decision_range["min"])
        end_inclusive = int(decision_range["max"])
        end_exclusive = end_inclusive + 1
        if end_exclusive <= start:
            raise ValueError(
                f"Week '{week_key}' decision_ts_range invalid: min={start} max={end_inclusive}"
            )
        return start, end_exclusive

    week1, week2, week3, week4 = weeks_in_order
    week1_start, week1_end_exclusive = _decision_range(week1)
    week2_start, week2_end_exclusive = _decision_range(week2)
    week3_start, week3_end_exclusive = _decision_range(week3)
    week4_start, week4_end_exclusive = _decision_range(week4)

    week3_40 = week3_start + ((week3_end_exclusive - week3_start) * 4) // 10
    week3_70 = week3_start + ((week3_end_exclusive - week3_start) * 7) // 10

    return {
        "protocol": "four_week_cmssl_val_test_rl_eval_v2",
        "cmssl": {
            # All emitted ranges are half-open: [start, end).
            "train": {"weeks": [week1], "start": week1_start, "end": week1_end_exclusive},
            "val": {"weeks": [week2], "start": week2_start, "end": week2_end_exclusive},
            "test": {"weeks": [week3], "start": week3_start, "end": week3_end_exclusive},
        },
        "rl": {
            "week": week3,
            # All emitted ranges are half-open: [start, end).
            "train": {"week": week3, "decision_ts_range": {"start": week3_start, "end": week3_40}},
            "val": {"week": week3, "decision_ts_range": {"start": week3_40, "end": week3_70}},
            "test": {"week": week3, "decision_ts_range": {"start": week3_70, "end": week3_end_exclusive}},
        },
        "eval": {
            "week": week4,
            # All emitted ranges are half-open: [start, end).
            "full": {"weeks": [week4], "start": week4_start, "end": week4_end_exclusive},
        },
    }


def _sort_pairs_by_end(pairs: List[WeekPair]) -> List[WeekPair]:
    rows = []
    for wk, ob_p, th_p in pairs:
        _start_dt, end_dt, _ = _parse_week_key_any(wk)
        rows.append((end_dt, wk, ob_p, th_p))
    rows.sort()
    return [(wk, ob_p, th_p) for _end, wk, ob_p, th_p in rows]


def _event_ts(event) -> int:
    """Extract the timestamp from a compact ingest event tuple."""
    if event is None:
        raise ValueError("Expected an event tuple, got None")
    if not isinstance(event, tuple) or len(event) < 2:
        raise ValueError(f"Expected compact event tuple, got: {event!r}")
    return int(event[1])


def _trade_iter_precise(tr_iter: Iterable[Tuple[int, int, dict]]):
    for ts_ms, seq, row in tr_iter:
        t_raw = row.get("timestamp", "")
        try:
            ts_ms_precise = timestamp_to_ms_half_even(t_raw)
        except ValueError:
            logger.warning(
                "Falling back to coarse trade timestamp for seq=%s raw_timestamp=%r",
                seq,
                t_raw,
            )
            # Safe fallback for missing/unparseable timestamp values.
            yield int(ts_ms), seq, row
            continue

        yield ts_ms_precise, seq, row


def _try_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_ob_iter(ob_path, day_start_ms, day_end_ms, dq_day):
    total = 0
    emitted = 0
    last_ts_out: Optional[int] = None
    day_clip_enabled = bool(BYBIT_DAY_CLIP)

    with _open_text(ob_path) as f:
        for line_no, line in enumerate(f, start=1):
            total += 1
            dq_day.increment_counter("ob", "total")
            if not line or not line.strip():
                dq_day.increment_counter("ob", "blank_line")
                continue
            try:
                obj = fast_json_loads(line)
            except Exception:
                dq_day.increment_counter("ob", "bad_json")
                dq_day.append_example("ob_bad_json", {"line_no": line_no, "line": line[:256]})
                continue

            ts_raw = obj.get("ts")
            if ts_raw is None:
                ts_raw = obj.get("cts")
            if ts_raw is None or (isinstance(ts_raw, str) and not ts_raw.strip()):
                dq_day.increment_counter("ob", "missing_ts")
                dq_day.append_example("ob_missing_ts", {"line_no": line_no, "payload": obj})
                continue
            try:
                ts_ms = timestamp_to_ms_half_even(ts_raw)
            except Exception:
                dq_day.increment_counter("ob", "bad_ts")
                dq_day.append_example("ob_bad_ts", {"line_no": line_no, "ts_raw": ts_raw, "payload": obj})
                continue

            dq_day.update_raw_ts(ts_ms)

            if day_clip_enabled and (ts_ms < day_start_ms or ts_ms >= day_end_ms):
                dq_day.increment_counter("ob", "clipped_day")
                continue

            if last_ts_out is not None and ts_ms < last_ts_out:
                backstep_ms = last_ts_out - ts_ms
                if backstep_ms <= BYBIT_TS_BACKSTEP_CLAMP_MS:
                    dq_day.increment_counter("ob", "clamped_backstep")
                    dq_day.append_example(
                        "ob_clamped_backstep",
                        {"line_no": line_no, "backstep_ms": backstep_ms, "ts_in": ts_ms, "ts_out": last_ts_out},
                    )
                    ts_ms = last_ts_out
                else:
                    dq_day.increment_counter("ob", "dropped_backstep")
                    dq_day.append_example(
                        "ob_dropped_backstep",
                        {"line_no": line_no, "backstep_ms": backstep_ms, "ts_in": ts_ms, "last_ts_out": last_ts_out},
                    )
                    continue

            data = obj.get("data")
            if not isinstance(data, dict):
                dq_day.increment_counter("ob", "missing_data")
                continue

            seq = _try_int(data.get("seq"), 0)
            tp_code = _compact_ob_type_code(obj.get("type") or data.get("type") or obj.get("DataType"))
            bids = _compact_book_levels(data.get("b"))
            asks = _compact_book_levels(data.get("a"))

            last_ts_out = int(ts_ms)
            emitted += 1
            dq_day.increment_counter("ob", "emitted")
            dq_day.update_output_ts(last_ts_out)
            yield ("ob", last_ts_out, seq, tp_code, bids, asks)

    dq_day.increment_counter("ob", "total_seen", total)
    dq_day.increment_counter("ob", "total_emitted", emitted)

def safe_th_iter(th_path, day_start_ms, day_end_ms, dq_day):
    total = 0
    emitted = 0
    last_ts_out: Optional[int] = None
    day_clip_enabled = bool(BYBIT_DAY_CLIP)

    with _open_text(th_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            dq_day.increment_counter("th", "missing_header")
            dq_day.increment_counter("th", "total_seen", total)
            dq_day.increment_counter("th", "total_emitted", emitted)
            return
        header_map = {str(name).strip(): idx for idx, name in enumerate(header)}

        def get_col(row, *names):
            for name in names:
                idx = header_map.get(name)
                if idx is not None and idx < len(row):
                    return row[idx]
            return None

        for seq, row in enumerate(reader, start=1):
            total += 1
            dq_day.increment_counter("th", "total")
            t_raw = get_col(row, "timestamp", "ts", "T")
            if t_raw is None or (isinstance(t_raw, str) and not t_raw.strip()):
                dq_day.increment_counter("th", "missing_ts")
                dq_day.append_example("th_missing_ts", {"seq": seq, "row": row[:16]})
                continue
            try:
                ts_ms = timestamp_to_ms_half_even(t_raw)
            except Exception:
                dq_day.increment_counter("th", "bad_ts")
                dq_day.append_example("th_bad_ts", {"seq": seq, "ts_raw": t_raw, "row": row[:16]})
                continue

            dq_day.update_raw_ts(ts_ms)

            if day_clip_enabled and (ts_ms < day_start_ms or ts_ms >= day_end_ms):
                dq_day.increment_counter("th", "clipped_day")
                continue

            if last_ts_out is not None and ts_ms < last_ts_out:
                backstep_ms = last_ts_out - ts_ms
                if backstep_ms <= BYBIT_TS_BACKSTEP_CLAMP_MS:
                    dq_day.increment_counter("th", "clamped_backstep")
                    dq_day.append_example(
                        "th_clamped_backstep",
                        {"seq": seq, "backstep_ms": backstep_ms, "ts_in": ts_ms, "ts_out": last_ts_out},
                    )
                    ts_ms = last_ts_out
                else:
                    dq_day.increment_counter("th", "dropped_backstep")
                    dq_day.append_example(
                        "th_dropped_backstep",
                        {"seq": seq, "backstep_ms": backstep_ms, "ts_in": ts_ms, "last_ts_out": last_ts_out},
                    )
                    continue

            try:
                price = float(get_col(row, "price"))
                size = float(get_col(row, "size"))
            except (TypeError, ValueError):
                dq_day.increment_counter("th", "bad_pxsz")
                dq_day.append_example("th_bad_pxsz", {"seq": seq, "row": row[:16]})
                continue

            side_code = _compact_trade_side_code(get_col(row, "side", "S"))
            tick_dir_code = _compact_tick_dir_code(get_col(row, "tickDirection", "tick_direction"))
            is_rpi = _compact_is_rpi_code(get_col(row, "RPI", "rpi"))

            last_ts_out = int(ts_ms)
            emitted += 1
            dq_day.increment_counter("th", "emitted")
            dq_day.update_output_ts(last_ts_out)
            yield ("trade", last_ts_out, seq, price, size, side_code, tick_dir_code, is_rpi)

    dq_day.increment_counter("th", "total_seen", total)
    dq_day.increment_counter("th", "total_emitted", emitted)

def build_token(fe: FeatureEngine, feat_z, is_trade: bool, dt_ms: float) -> np.ndarray:
    # exact tail order:
    # [log_dt_ms, is_trade, log_events_100ms, log_events_500ms, log_events_1000ms, log_events_3000ms, log_events_7500ms]
    aux_tail = np.array(
        [
            np.log1p(float(dt_ms)),
            float(is_trade),
            np.log1p(fe.event_density_100ms()),
            np.log1p(fe.event_density_500ms()),
            np.log1p(fe.event_density_1000ms()),
            np.log1p(fe.event_density_3000ms()),
            np.log1p(fe.event_density_7500ms()),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [np.asarray(feat_z, dtype=np.float32), aux_tail], axis=0
    ).astype(np.float32, copy=False)

# ---------- chunk writer (preallocated) ----------
@dataclass
class FlushJob:
    week_key: str
    chunk_id: int
    row_count: int
    out_dir: str
    x_core_file: str
    x_aux_file: str
    y_file: str
    ts_file: str
    x_core: np.ndarray
    x_aux: np.ndarray
    y: np.ndarray
    ts: np.ndarray
    core_dtype: Any


class ChunkWriter:
    def __init__(
        self,
        out_dir: str,
        lookback: int,
        feature_dim: int,
        ram_budget_mb: int,
        chunk_size_override: int = 0,
        start_chunk_id: int = 0,
        week_key: str = "",
        flush_callback: Optional[Callable[[FlushJob], None]] = None,
    ):
        self.out_dir = out_dir
        self.week_key = str(week_key)
        self.L = int(lookback)
        self.F = int(feature_dim)
        self.F_core = self.F - AUX_DIM
        assert self.F_core > 0, "feature_dim must be > AUX_DIM"
        self.core_dtype = np.float32
        self.flush_callback = flush_callback

        total_bytes_per_seq = (
            (self.L * self.F_core * 4)
            + (self.L * AUX_DIM * 4)
            + (NUM_HORIZONS * 4)
            + 8
        )
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            auto_n = max(256, int((ram_budget_mb * 1024 * 1024) // total_bytes_per_seq))
            safety_cap = max(256, int((2 * 1024 * 1024 * 1024) // max(1, total_bytes_per_seq)))
            self.N = min(auto_n, safety_cap)

        self.X_core: np.ndarray
        self.X_aux: np.ndarray
        self.Y: np.ndarray
        self.TS: np.ndarray
        self._alloc_buffers()
        self.i = 0
        self.cid = int(start_chunk_id)
        self.chunks_meta = []

    def _alloc_buffers(self) -> None:
        self.X_core = np.empty((self.N, self.L, self.F_core), dtype=np.float32)
        self.X_aux = np.empty((self.N, self.L, AUX_DIM), dtype=np.float32)
        self.Y = np.empty((self.N, NUM_HORIZONS), dtype=np.float32)
        self.TS = np.empty((self.N,), dtype=np.int64)

    def add_from_token_buffer(self, ts_decision_ms: int, token_buffer: TokenBufferSnapshot, y: np.ndarray):
        if token_buffer.feature_dim != self.F:
            raise ValueError(
                f"Token buffer feature_dim={token_buffer.feature_dim} does not match writer feature_dim={self.F}"
            )
        if token_buffer.lookback != self.L:
            raise ValueError(
                f"Token buffer lookback={token_buffer.lookback} does not match writer lookback={self.L}"
            )
        if token_buffer.count <= 0:
            raise RuntimeError("Cannot add sequence from empty token buffer snapshot")

        row_core = self.X_core[self.i]
        row_aux = self.X_aux[self.i]
        pad_n = self.L - token_buffer.count
        if pad_n > 0:
            earliest_idx = (token_buffer.cursor - token_buffer.count) % self.L
            earliest = token_buffer.source.tokens[earliest_idx]
            row_core[:pad_n] = earliest[:self.F_core]
            row_aux[:pad_n, :] = 0.0

        dest_start = pad_n
        src_start = (token_buffer.cursor - token_buffer.count) % self.L
        first_block = min(token_buffer.count, self.L - src_start)
        second_block = token_buffer.count - first_block

        src = token_buffer.source.tokens
        row_core[dest_start : dest_start + first_block] = src[src_start : src_start + first_block, : self.F_core]
        row_aux[dest_start : dest_start + first_block] = src[src_start : src_start + first_block, self.F_core :]
        if second_block > 0:
            mid = dest_start + first_block
            row_core[mid : mid + second_block] = src[:second_block, : self.F_core]
            row_aux[mid : mid + second_block] = src[:second_block, self.F_core :]

        self.Y[self.i] = y
        self.TS[self.i] = ts_decision_ms
        self.i += 1
        if self.i >= self.N:
            self.flush()

    def _build_flush_job(self) -> Optional[FlushJob]:
        if self.i == 0:
            return None
        chunk_id = int(self.cid)
        row_count = int(self.i)
        job = FlushJob(
            week_key=self.week_key,
            chunk_id=chunk_id,
            row_count=row_count,
            out_dir=self.out_dir,
            x_core_file=f"Xcore_{chunk_id:03d}.npy",
            x_aux_file=f"Xaux_{chunk_id:03d}.npy",
            y_file=f"y_{chunk_id:03d}.npy",
            ts_file=f"ts_{chunk_id:03d}.npy",
            x_core=self.X_core,
            x_aux=self.X_aux,
            y=self.Y,
            ts=self.TS,
            core_dtype=self.core_dtype,
        )
        self.chunks_meta.append({
            "chunk": chunk_id,
            "n": row_count,
            "files": {
                "core": job.x_core_file,
                "aux": job.x_aux_file,
                "y": job.y_file,
                "ts": job.ts_file,
            },
        })
        self.cid += 1
        self.i = 0
        self._alloc_buffers()
        return job

    def flush(self):
        job = self._build_flush_job()
        if job is None:
            return
        if self.flush_callback is None:
            _persist_flush_job(job)
        else:
            self.flush_callback(job)


_SENTINEL_FLUSH_JOB = object()
_FLUSH_QUEUE_MAXSIZE = 4


def _persist_flush_job(job: FlushJob) -> None:
    x_core_path = os.path.join(job.out_dir, job.x_core_file)
    x_aux_path = os.path.join(job.out_dir, job.x_aux_file)
    y_path = os.path.join(job.out_dir, job.y_file)
    ts_path = os.path.join(job.out_dir, job.ts_file)

    if job.core_dtype == np.float16:
        maxabs = float(np.max(np.abs(job.x_core[: job.row_count])))
        if maxabs > np.finfo(np.float16).max:
            print(f"[warn] core max {maxabs:.1f} exceeds fp16 range; consider BYBIT_SAVE_DTYPE=bf16", flush=True)

    np.save(x_core_path, job.x_core[: job.row_count].astype(job.core_dtype, copy=False))
    np.save(x_aux_path, job.x_aux[: job.row_count])
    np.save(y_path, job.y[: job.row_count])
    np.save(ts_path, job.ts[: job.row_count])


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
        self.closed_writers: Dict[str, List[ChunkWriter]] = defaultdict(list)
        self.next_chunk_id: Dict[str, int] = defaultdict(int)
        self.week_counts: Dict[str, int] = defaultdict(int)
        self.week_decision_span: Dict[str, List[int]] = {}
        self.chunk_size_used: int = 0
        self.week_metas: Dict[str, dict] = {}
        self.pca_meta = dict(pca_meta) if pca_meta is not None else {}
        self.flush_queue: "queue.Queue[object]" = queue.Queue(maxsize=_FLUSH_QUEUE_MAXSIZE)
        self.writer_exception: Optional[BaseException] = None
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="offline-ingest-chunk-writer",
            daemon=True,
        )
        self.writer_thread.start()

    def _check_writer_exception(self) -> None:
        if self.writer_exception is not None:
            raise RuntimeError("Asynchronous chunk writer failed") from self.writer_exception

    def _writer_loop(self) -> None:
        try:
            while True:
                job = self.flush_queue.get()
                try:
                    if job is _SENTINEL_FLUSH_JOB:
                        return
                    _persist_flush_job(job)
                finally:
                    self.flush_queue.task_done()
        except BaseException as exc:
            self.writer_exception = exc
            while True:
                try:
                    pending = self.flush_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.flush_queue.task_done()
                    if pending is _SENTINEL_FLUSH_JOB:
                        break

    def _enqueue_flush_job(self, job: FlushJob) -> None:
        self.next_chunk_id[job.week_key] = max(
            int(self.next_chunk_id.get(job.week_key, 0)),
            int(job.chunk_id) + 1,
        )
        while True:
            self._check_writer_exception()
            try:
                self.flush_queue.put(job, timeout=0.5)
                self._check_writer_exception()
                return
            except queue.Full:
                continue

    def _ensure_writer(self, week_key: str) -> ChunkWriter:
        if week_key in self.writers:
            return self.writers[week_key]
        if week_key in self.week_metas:
            raise RuntimeError(
                f"Week '{week_key}' is already finalized; refusing to reopen writer."
            )
        week_dir = os.path.join(self.out_root, week_key)
        ensure_dir(week_dir)
        start_chunk_id = int(self.next_chunk_id.get(week_key, 0))
        writer = ChunkWriter(
            week_dir,
            self.lookback,
            self.feature_dim,
            self.ram_budget_mb,
            self.chunk_size_override,
            start_chunk_id=start_chunk_id,
            week_key=week_key,
            flush_callback=self._enqueue_flush_job,
        )
        self.writers[week_key] = writer
        if not self.chunk_size_used:
            self.chunk_size_used = int(writer.N)
        return writer

    def _find_week_key(self, ts_ms: int) -> str:
        # First, normal exact matching: ts in [start_ms, end_ms)
        for wk, start_ms, end_ms in self.week_index:
            if start_ms <= ts_ms < end_ms:
                return wk

        # If nothing matched, allow a small grace window on the *last* week.
        # This covers tiny spillovers like a few ms after midnight of the "to" date,
        # or horizon-related edges, without creating overlaps.
        if self.week_index:
            last_wk, last_start, last_end = self.week_index[-1]
            if ts_ms >= last_end and ts_ms < last_end + GRACE_MS:
                return last_wk

        # If we're here, this really is outside any reasonable week boundary.
        raise ValueError(f"No week found for decision timestamp {ts_ms}")

    def add_from_token_buffer(self, ts_decision_ms: int, token_buffer: TokenBufferSnapshot, label: np.ndarray):
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_writer(wk)
        writer.add_from_token_buffer(ts_decision_ms, token_buffer, label)
        self.week_counts[wk] += 1
        if wk not in self.week_decision_span:
            self.week_decision_span[wk] = [ts_decision_ms, ts_decision_ms]
        else:
            span = self.week_decision_span[wk]
            span[0] = min(span[0], ts_decision_ms)
            span[1] = max(span[1], ts_decision_ms)

    def _close_writer(self, week_key: str):
        writer = self.writers.pop(week_key, None)
        if writer is None:
            return
        writer.flush()
        self.next_chunk_id[week_key] = int(writer.cid)
        self.closed_writers[week_key].append(writer)

    def _build_week_meta(self, week_key: str, writers: List[ChunkWriter]) -> dict:
        span = self.week_decision_span.pop(week_key, None)
        total_sequences = int(self.week_counts.get(week_key, 0))
        meta_path = os.path.join(self.out_root, week_key, "meta_week.json")
        chunks_meta = []
        for writer in writers:
            chunks_meta.extend(
                {
                    "chunk": int(entry["chunk"]),
                    "n": int(entry["n"]),
                    "files": dict(entry["files"]),
                }
                for entry in writer.chunks_meta
            )
        chunks_meta.sort(key=lambda entry: int(entry["chunk"]))
        seen_chunk_ids = set()
        for entry in chunks_meta:
            chunk_id = int(entry["chunk"])
            if chunk_id in seen_chunk_ids:
                raise RuntimeError(
                    f"Duplicate chunk id {chunk_id} detected while finalizing week '{week_key}'."
                )
            seen_chunk_ids.add(chunk_id)
        for entry in chunks_meta:
            ts_file = entry.get("files", {}).get("ts")
            if not ts_file:
                raise ValueError(
                    f"Chunk {entry.get('chunk')} in week '{week_key}' missing ts file metadata."
                )
            ts_path = os.path.join(self.out_root, week_key, ts_file)
            if not os.path.exists(ts_path):
                raise FileNotFoundError(
                    f"Chunk {entry.get('chunk')} in week '{week_key}' missing ts file '{ts_file}'."
                )
        rows_total = int(sum(entry["n"] for entry in chunks_meta))
        chunk_size_used = int(writers[0].N) if writers else 0
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
            "checkpoint_schema_expected": "cmssl17-signed-raw-v1",
            **canonical_mode_fields(),
            "lookback": self.lookback,
            "feature_dim_total": self.feature_dim,
            "feature_dim_core": self.feature_dim - AUX_DIM,
            "label_dim": int(NUM_HORIZONS),
            "horizons_ms": [int(h) for h in HORIZONS_MS],
            "chunk_size_used": chunk_size_used,
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
        print(f"[write] week={week_key} chunks={len(chunks_meta)} rows={rows_total}", flush=True)
        return meta

    def _finalize_closed_weeks(self):
        for week_key, writers in list(self.closed_writers.items()):
            if not writers:
                del self.closed_writers[week_key]
                continue
            self._build_week_meta(week_key, writers)
            del self.closed_writers[week_key]

    def close_old_writers(self, watermark_ms: int):
        to_close = []
        for wk, writer in list(self.writers.items()):
            _start_ms, end_ms = self.week_bounds[wk]
            if end_ms + GRACE_MS < watermark_ms:
                to_close.append(wk)
        for wk in to_close:
            self._close_writer(wk)

    def flush_all(self):
        for wk in list(self.writers.keys()):
            self._close_writer(wk)
        self._check_writer_exception()
        self.flush_queue.put(_SENTINEL_FLUSH_JOB)
        self.writer_thread.join()
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


def _weekday_only_week_pair(week_pair: WeekPair) -> WeekPair:
    wk, ob_paths, th_paths = week_pair
    ob_weekday: List[str] = []
    th_weekday: List[str] = []

    for idx, ob_path in enumerate(ob_paths):
        ob_name = Path(ob_path).name
        m_ob = OB_DAILY_RE.match(ob_name)
        if not m_ob:
            raise ValueError(f"Unable to parse OB day from path for PCA weekday filter: {ob_path}")
        day = _parse_ymd_date(m_ob.group("d"))
        if day.weekday() >= 5:
            continue
        ob_weekday.append(ob_path)
        if th_paths:
            if idx >= len(th_paths):
                raise ValueError(
                    "OB/TH day alignment mismatch while building weekday-only PCA subset "
                    f"for week '{wk}': ob_days={len(ob_paths)} th_days={len(th_paths)}"
                )
            th_weekday.append(th_paths[idx])

    return wk, ob_weekday, th_weekday


def _count_stream_core_feature_rows(pairs: List[WeekPair]) -> int:
    rows = 0
    for _feat in _stream_core_features(pairs):
        rows += 1
    return rows


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

    wk, _ob_paths, _th_paths = train_pairs[0]
    train_weekday_pair = _weekday_only_week_pair(train_pairs[0])
    weekday_pairs = [train_weekday_pair]

    total_rows = _count_stream_core_feature_rows(weekday_pairs)
    if total_rows <= 0:
        print(f"[pca  ] No weekday PCA rows available in week '{wk}'; skipping PCA fit")
        return meta

    block_size = min(sample_limit, total_rows)
    start_idx = max(0, (total_rows - block_size) // 2)

    sample_rows: List[np.ndarray] = []
    end_idx = start_idx + block_size
    for idx, feat in enumerate(_stream_core_features(weekday_pairs)):
        if idx < start_idx:
            continue
        if idx >= end_idx:
            break
        sample_rows.append(np.asarray(feat, dtype=np.float32))

    if not sample_rows:
        print("[pca  ] Unable to collect PCA sample rows; skipping")
        return meta

    sample_array = np.asarray(sample_rows, dtype=np.float32)
    n_components = _select_pca_components(sample_array, target_var)
    if n_components <= 0:
        print("[pca  ] Unable to initialise PCA (insufficient data); skipping")
        return meta

    ipca = IncrementalPCA(
        n_components=n_components,
        batch_size=None if batch_size <= 0 else max(batch_size, n_components),
    )
    ipca.partial_fit(sample_array)
    batches += 1
    fitted_rows = int(sample_array.shape[0])
    if time.monotonic() - last_log >= 300:
        print(f"[pca-fit] fitted={fitted_rows} batches={batches}", flush=True)
        last_log = time.monotonic()
    print(
        f"[pca-init] n_components={n_components} sample_rows={sample_array.shape[0]} "
        f"start_idx={start_idx} total_weekday_rows={total_rows}",
        flush=True,
    )

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
            "rows_total": int(block_size),
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
            pca_summary = _summarise_pca_meta({
                "applied": False,
                "var_kept": pca_summary.get("var_kept", PCA_VAR_TARGET),
            })

    fe = FeatureEngine()
    # Decision timestamps are OB event timestamps (event-time).
    # Entry references use decision_ts directly (no additional delay).
    labeler = LabelBuilder(delta_ms=0, horizons_ms=HORIZONS_MS)

    token_buffer: Optional[TokenRingBuffer] = None
    pending_decisions: deque[TokenBufferSnapshot] = deque()
    last_decision_ts_ms: Optional[int] = None

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

            is_duplicate_decision_ts = (
                last_decision_ts_ms is not None and int(ts_ms) == last_decision_ts_ms
            )
            if last_decision_ts_ms is not None and int(ts_ms) < last_decision_ts_ms:
                raise RuntimeError(
                    f"Non-monotone decision timestamp: decision_ts_ms={int(ts_ms)} "
                    f"< last_decision_ts_ms={last_decision_ts_ms}"
                )
            if is_duplicate_decision_ts and not ALLOW_DUPLICATE_OB_TS:
                raise RuntimeError(
                    f"Non-monotone decision timestamp: decision_ts_ms={int(ts_ms)} "
                    f"<= last_decision_ts_ms={last_decision_ts_ms}"
                )

            dt_tick = 1 if last_decision_ts_ms is None else int(ts_ms - last_decision_ts_ms)
            tok = build_token(fe, feat_core, is_trade, dt_tick)
            if F is None:
                F = tok.shape[0]
                token_buffer = TokenRingBuffer(LOOKBACK, F)
                router = WeekWriterRouter(
                    out_root,
                    LOOKBACK,
                    F,
                    RAM_BUDGET,
                    CHUNK_SIZE,
                    week_index,
                    pca_meta=pca_summary,
                )
            if token_buffer is None:
                raise RuntimeError("Token ring buffer was not initialised")
            token_buffer.append(tok)
            if is_duplicate_decision_ts:
                if not pending_decisions:
                    raise RuntimeError(
                        "Duplicate OB timestamp cannot update state because no pending decision exists"
                    )
                pending_decisions[-1] = token_buffer.snapshot(int(ts_ms))
            else:
                pending_decisions.append(token_buffer.snapshot(int(ts_ms)))
                labeler.on_decision(int(ts_ms))
            matured = labeler.on_event(int(ts_ms), float(mid))
            last_decision_ts_ms = int(ts_ms)

            if matured is None:
                raise RuntimeError("Matured labels were not produced for OB event")
            for yy in matured:
                if not pending_decisions:
                    raise RuntimeError(
                        "Matured label available but no pending sequences to pair"
                    )
                if router is None:
                    raise RuntimeError("Router not initialised before label maturity")
                snapshot = pending_decisions.popleft()
                router.add_from_token_buffer(
                    snapshot.ts_decision_ms,
                    snapshot,
                    yy.astype(np.float32, copy=False),
                )
                total_sequences += 1

        t_router = time.monotonic()
        if router is not None:
            router.close_old_writers(int(ts_ms))
        router_housekeeping_s += time.monotonic() - t_router
        
        if time.monotonic() - last_log >= 300:
            print(f"[tok  ] seq={total_sequences} weeks={week_counter}/{week_total} "
                  f"chunkN={router.chunk_size_used if router else 0}", flush=True)
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
    chunk_files = []
    for wk, week_meta in week_meta_records.items():
        for entry in week_meta.get("chunks", []):
            files = dict(entry.get("files", {}))
            if "ts" not in files:
                raise ValueError(
                    f"Missing ts file entry for week={wk} chunk={entry.get('chunk')}"
                )
            chunk_files.append({
                "week": wk,
                "chunk": int(entry.get("chunk", 0)),
                "n": int(entry.get("n", 0)),
                "files": files,
            })


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

    # Dataset metadata contract: `weeks_in_order` is the only supported key for
    # week ordering in OUT_ROOT/meta.json.
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
        "checkpoint_schema_expected": "cmssl17-signed-raw-v1",
        **canonical_mode_fields(),
        "lookback": int(LOOKBACK),
        "feature_dim_total": feature_dim_total,
        "feature_dim_core": feature_dim_core,
        "aux_tail": ["log_dt_ms", "is_trade", "log_events_100ms", "log_events_500ms", "log_events_1000ms", "log_events_3000ms", "log_events_7500ms"],
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
        "chunks": chunk_files,
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
                        f"Inconsistent ingest mode in week '{wk}': {field}={observed!r} "
                        f"(expected {expected!r})"
                    )

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
