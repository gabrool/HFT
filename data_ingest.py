#!/usr/bin/env python3
"""
Decision-time ingest (memory-safe):
- Store one flat feature row per OB decision timestamp.
- Training materializes [LOOKBACK, F] windows dynamically from flat rows.
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

Output storage format:
  flat_decision_rows_v1 chunks: core_*.npy / aux_*.npy / y_*.npy / ts_*.npy

Shared constants from CM:
  LOOKBACK (and related model/data constants) are defined in CM.py.
  If these values are intentionally changed, update them in CM.py.
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
PCA_MAX_ROWS        = int(os.environ.get("BYBIT_PCA_MAX_ROWS", "200000"))
if PCA_MAX_ROWS <= 0:
    raise ValueError(f"BYBIT_PCA_MAX_ROWS must be > 0, got {PCA_MAX_ROWS}")
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
from CM import (
    FeatureEngine,
    LabelBuilder,
    HORIZONS_MS,
    NUM_HORIZONS,
    LOOKBACK,
    WINDOW_MS,
    AUX_DIM,
    _open_text,
    timestamp_to_ms_half_even,
)  # keep shared model/data constants only; ingest helpers are local below
# LOOKBACK is a shared model constant from CM (single source of truth).

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



def build_pipeline_splits(week_keys: List[str]) -> Dict[str, Any]:
    if len(week_keys) != 5:
        raise ValueError(
            f"Phase 1 requires exactly 5 weeks, got {len(week_keys)}: {week_keys}"
        )

    return {
        "protocol": "five_week_cmssl2w_val_test_rl_eval_v1",
        "cmssl": {
            "train": [week_keys[0], week_keys[1]],
            "val": [week_keys[2]],
            "test": [week_keys[3]],
        },
        "rl": {
            "train": [week_keys[3]],
        },
        "eval": {
            "weeks": [week_keys[4]],
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

def build_aux_tail(fe: FeatureEngine, *, is_trade: bool, dt_ms: float) -> np.ndarray:
    # exact Phase 1 tail order: [log_dt_ms, is_trade, log_events_100ms, log_events_200ms, log_events_500ms]
    return np.asarray([
        np.log1p(float(dt_ms)),
        float(is_trade),
        np.log1p(fe.event_density_100ms()),
        np.log1p(fe.event_density_200ms()),
        np.log1p(fe.event_density_500ms()),
    ], dtype=np.float32)

# ---------- flat-row chunk writer ----------
@dataclass
class FlatFlushJob:
    week_key: str
    chunk_id: int
    row_start: int
    row_count: int
    out_dir: str
    core_file: str
    aux_file: str
    y_file: str
    ts_file: str
    core: np.ndarray
    aux: np.ndarray
    y: np.ndarray
    ts: np.ndarray


class FlatWeekWriter:
    """Chunked flat decision-row writer.

    Rows are appended at OB decision time and labels are filled later by the
    matured-label FIFO.  Chunks are persisted only once every row in the chunk
    has a label, so written core/aux/y/ts rows remain exactly aligned.
    """

    def __init__(
        self,
        out_dir: str,
        feature_dim_core: int,
        pre_pca_dim: int,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        aux_dim: int,
        ram_budget_mb: int,
        chunk_size_override: int = 0,
        start_chunk_id: int = 0,
        week_key: str = "",
        flush_callback: Optional[Callable[[FlatFlushJob], None]] = None,
    ):
        self.out_dir = out_dir
        self.week_key = str(week_key)
        self.feature_dim_core = int(feature_dim_core)
        self.pre_pca_dim = int(pre_pca_dim)
        self.aux_dim = int(aux_dim)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float32)
        self.pca_components = np.asarray(pca_components, dtype=np.float32)
        if self.aux_dim != AUX_DIM:
            raise ValueError(f"aux_dim={self.aux_dim} must equal AUX_DIM={AUX_DIM}")
        if self.pca_mean.shape != (self.pre_pca_dim,):
            raise ValueError("PCA mean/pre-PCA dimension mismatch")
        if self.pca_components.shape != (self.feature_dim_core, self.pre_pca_dim):
            raise ValueError("PCA components dimension mismatch")
        self.flush_callback = flush_callback
        bytes_per_row = (4 * self.pre_pca_dim) + (4 * self.feature_dim_core) + (4 * self.aux_dim) + (4 * NUM_HORIZONS) + 8
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            auto_n = max(256, int((ram_budget_mb * 1024 * 1024) // max(1, bytes_per_row)))
            safety_cap = max(256, int((2 * 1024 * 1024 * 1024) // max(1, bytes_per_row)))
            self.N = min(auto_n, safety_cap)
        self.cid = int(start_chunk_id)
        self.rows_total = 0
        self.open = self._new_chunk(self.cid, self.rows_total)
        self.pending_chunks: List[dict] = []
        self.chunks_meta: List[Dict[str, Any]] = []

    def _new_chunk(self, chunk_id: int, row_start: int) -> dict:
        return {
            "chunk_id": int(chunk_id),
            "row_start": int(row_start),
            "row_count": 0,
            "labels_set": 0,
            "core": np.empty((self.N, self.feature_dim_core), dtype=np.float32),
            "aux": np.empty((self.N, self.aux_dim), dtype=np.float32),
            "y": np.full((self.N, NUM_HORIZONS), np.nan, dtype=np.float32),
            "ts": np.empty((self.N,), dtype=np.int64),
        }

    def append_row_pre_pca(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> int:
        if self.open["row_count"] >= self.N:
            self.pending_chunks.append(self.open)
            self.cid += 1
            self.open = self._new_chunk(self.cid, self.rows_total)
            self._flush_ready_prefix()
        core_pre = np.asarray(core_pre_pca, dtype=np.float32)
        aux = np.asarray(aux_tail, dtype=np.float32)
        if core_pre.shape != (self.pre_pca_dim,):
            raise ValueError(f"Core pre-PCA dim mismatch: {core_pre.shape} != {(self.pre_pca_dim,)}")
        if aux.shape != (self.aux_dim,):
            raise ValueError(f"Aux dim mismatch: {aux.shape} != {(self.aux_dim,)}")
        if not np.all(np.isfinite(core_pre)) or not np.all(np.isfinite(aux)):
            raise ValueError("Non-finite flat-row features")
        centered = core_pre - self.pca_mean
        core = np.dot(centered, self.pca_components.T).astype(np.float32, copy=False)
        i = int(self.open["row_count"])
        self.open["core"][i] = core
        self.open["aux"][i] = aux
        self.open["ts"][i] = int(ts_decision_ms)
        row_idx = int(self.rows_total)
        self.open["row_count"] += 1
        self.rows_total += 1
        return row_idx

    def overwrite_latest_row(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> int:
        if self.open["row_count"] <= 0:
            raise RuntimeError("Cannot overwrite latest row after chunk rollover")
        row_idx = int(self.rows_total - 1)
        local = int(self.open["row_count"] - 1)
        core_pre = np.asarray(core_pre_pca, dtype=np.float32)
        aux = np.asarray(aux_tail, dtype=np.float32)
        centered = core_pre - self.pca_mean
        self.open["core"][local] = np.dot(centered, self.pca_components.T).astype(np.float32, copy=False)
        self.open["aux"][local] = aux
        self.open["ts"][local] = int(ts_decision_ms)
        return row_idx

    def set_label(self, row_idx: int, y: np.ndarray) -> None:
        yy = np.asarray(y, dtype=np.float32)
        if yy.shape != (NUM_HORIZONS,):
            raise ValueError(f"Label dim mismatch: {yy.shape} != {(NUM_HORIZONS,)}")
        for ch in self.pending_chunks + [self.open]:
            start = int(ch["row_start"]); end = start + int(ch["row_count"])
            if start <= int(row_idx) < end:
                local = int(row_idx) - start
                if not np.isnan(ch["y"][local]).all():
                    raise RuntimeError(f"Duplicate label for row_idx={row_idx}")
                ch["y"][local] = yy
                ch["labels_set"] += 1
                self._flush_ready_prefix()
                return
        raise KeyError(f"row_idx={row_idx} not found in writer for week={self.week_key}")

    def _build_job(self, ch: dict, row_count: Optional[int] = None) -> Optional[FlatFlushJob]:
        n = int(ch["row_count"] if row_count is None else row_count)
        if n <= 0:
            return None
        if np.isnan(ch["y"][:n]).any():
            raise RuntimeError("Attempted to flush unlabeled flat rows")
        chunk_id = int(ch["chunk_id"])
        core_file = f"core_{chunk_id:06d}.npy"
        aux_file = f"aux_{chunk_id:06d}.npy"
        y_file = f"y_{chunk_id:06d}.npy"
        ts_file = f"ts_{chunk_id:06d}.npy"
        self.chunks_meta.append({
            "chunk_id": chunk_id,
            "chunk": chunk_id,
            "rows": n,
            "n": n,
            "row_start": int(ch["row_start"]),
            "row_end": int(ch["row_start"] + n),
            "files": {"core": core_file, "aux": aux_file, "y": y_file, "ts": ts_file},
        })
        return FlatFlushJob(
            week_key=self.week_key, chunk_id=chunk_id, row_start=int(ch["row_start"]), row_count=n,
            out_dir=self.out_dir, core_file=core_file, aux_file=aux_file, y_file=y_file, ts_file=ts_file,
            core=ch["core"][:n].copy(), aux=ch["aux"][:n].copy(), y=ch["y"][:n].copy(), ts=ch["ts"][:n].copy(),
        )

    def _emit(self, job: FlatFlushJob) -> None:
        if self.flush_callback is None:
            _persist_flat_flush_job(job)
        else:
            self.flush_callback(job)

    def _flush_ready_prefix(self) -> None:
        while self.pending_chunks and int(self.pending_chunks[0]["labels_set"]) == int(self.pending_chunks[0]["row_count"]):
            ch = self.pending_chunks.pop(0)
            job = self._build_job(ch)
            if job is not None:
                self._emit(job)

    def flush_labeled_tail(self) -> None:
        self._flush_ready_prefix()
        # At dataset end, labels mature FIFO.  Persist any labeled prefix and
        # drop the final unlabeled suffix (possibly spanning multiple chunks).
        for ch in list(self.pending_chunks) + [self.open]:
            n_labeled = int(ch["labels_set"])
            if n_labeled > 0:
                if np.isnan(ch["y"][:n_labeled]).any():
                    raise RuntimeError("Non-prefix label maturation detected")
                job = self._build_job(ch, n_labeled)
                if job is not None:
                    self._emit(job)
            dropped = int(ch["row_count"] - n_labeled)
            if dropped > 0:
                print(f"[warn] dropping {dropped} unlabeled tail rows week={self.week_key}", flush=True)
        self.pending_chunks.clear()
        self.open = self._new_chunk(self.cid + 1, self.rows_total)


def _persist_flat_flush_job(job: FlatFlushJob) -> None:
    ensure_dir(job.out_dir)
    np.save(os.path.join(job.out_dir, job.core_file), job.core[: job.row_count].astype(np.float32, copy=False))
    np.save(os.path.join(job.out_dir, job.aux_file), job.aux[: job.row_count].astype(np.float32, copy=False))
    np.save(os.path.join(job.out_dir, job.y_file), job.y[: job.row_count].astype(np.float32, copy=False))
    np.save(os.path.join(job.out_dir, job.ts_file), job.ts[: job.row_count].astype(np.int64, copy=False))


_SENTINEL_FLUSH_JOB = object()
_FLUSH_QUEUE_MAXSIZE = 4


class FlatWeekRouter:
    def __init__(
        self,
        out_root: str,
        feature_dim_core: int,
        pre_pca_dim: int,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        aux_dim: int,
        ram_budget_mb: int,
        chunk_size_override: int,
        week_index: List[Tuple[str, int, int]],
        pca_meta: Optional[dict] = None,
    ):
        self.out_root = out_root
        self.feature_dim_core = int(feature_dim_core)
        self.pre_pca_dim = int(pre_pca_dim)
        self.feature_dim_total = int(feature_dim_core + aux_dim)
        self.aux_dim = int(aux_dim)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float32)
        self.pca_components = np.asarray(pca_components, dtype=np.float32)
        self.ram_budget_mb = int(ram_budget_mb)
        self.chunk_size_override = int(chunk_size_override)
        self.week_index = list(week_index)
        self.week_bounds = {wk: (start, end) for wk, start, end in self.week_index}
        self.writers: Dict[str, FlatWeekWriter] = {}
        self.closed_writers: Dict[str, List[FlatWeekWriter]] = defaultdict(list)
        self.next_chunk_id: Dict[str, int] = defaultdict(int)
        self.week_counts: Dict[str, int] = defaultdict(int)
        self.week_label_counts: Dict[str, int] = defaultdict(int)
        self.week_decision_span: Dict[str, List[int]] = {}
        self.chunk_size_used = 0
        self.week_metas: Dict[str, dict] = {}
        self.pca_meta = dict(pca_meta) if pca_meta is not None else {}
        self.flush_queue: "queue.Queue[object]" = queue.Queue(maxsize=_FLUSH_QUEUE_MAXSIZE)
        self.writer_exception: Optional[BaseException] = None
        self.writer_thread = threading.Thread(target=self._writer_loop, name="flat-row-writer", daemon=True)
        self.writer_thread.start()

    def _check_writer_exception(self) -> None:
        if self.writer_exception is not None:
            raise RuntimeError("Asynchronous flat-row writer failed") from self.writer_exception

    def _writer_loop(self) -> None:
        try:
            while True:
                job = self.flush_queue.get()
                try:
                    if job is _SENTINEL_FLUSH_JOB:
                        return
                    _persist_flat_flush_job(job)  # type: ignore[arg-type]
                finally:
                    self.flush_queue.task_done()
        except BaseException as exc:
            self.writer_exception = exc

    def _enqueue_flush_job(self, job: FlatFlushJob) -> None:
        while True:
            self._check_writer_exception()
            try:
                self.flush_queue.put(job, timeout=0.5)
                return
            except queue.Full:
                continue

    def _find_week_key(self, ts_ms: int) -> str:
        for wk, start_ms, end_ms in self.week_index:
            if start_ms <= ts_ms < end_ms:
                return wk
        if self.week_index:
            last_wk, _last_start, last_end = self.week_index[-1]
            if ts_ms >= last_end and ts_ms < last_end + GRACE_MS:
                return last_wk
        raise ValueError(f"No week found for decision timestamp {ts_ms}")

    def _ensure_writer(self, week_key: str) -> FlatWeekWriter:
        if week_key in self.writers:
            return self.writers[week_key]
        if week_key in self.week_metas:
            raise RuntimeError(f"Week '{week_key}' is finalized; refusing to reopen writer")
        week_dir = os.path.join(self.out_root, week_key)
        ensure_dir(week_dir)
        writer = FlatWeekWriter(
            week_dir, self.feature_dim_core, self.pre_pca_dim, self.pca_mean, self.pca_components,
            self.aux_dim, self.ram_budget_mb, self.chunk_size_override,
            start_chunk_id=int(self.next_chunk_id.get(week_key, 0)), week_key=week_key,
            flush_callback=self._enqueue_flush_job,
        )
        self.writers[week_key] = writer
        if not self.chunk_size_used:
            self.chunk_size_used = int(writer.N)
        return writer

    def append_feature_row(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> Tuple[str, int]:
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_writer(wk)
        row_idx = writer.append_row_pre_pca(ts_decision_ms, core_pre_pca, aux_tail)
        self.week_counts[wk] += 1
        span = self.week_decision_span.setdefault(wk, [int(ts_decision_ms), int(ts_decision_ms)])
        span[0] = min(span[0], int(ts_decision_ms)); span[1] = max(span[1], int(ts_decision_ms))
        return wk, int(row_idx)

    def overwrite_latest_feature_row(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> Tuple[str, int]:
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_writer(wk)
        row_idx = writer.overwrite_latest_row(ts_decision_ms, core_pre_pca, aux_tail)
        return wk, int(row_idx)

    def set_label(self, week_key: str, row_idx: int, y: np.ndarray) -> None:
        self._check_writer_exception()
        writer = self.writers.get(week_key)
        if writer is None:
            for w in self.closed_writers.get(week_key, []):
                writer = w
                break
        if writer is None:
            raise KeyError(f"No writer for label week={week_key} row_idx={row_idx}")
        writer.set_label(row_idx, y)
        self.week_label_counts[week_key] += 1

    def _close_writer(self, week_key: str) -> None:
        writer = self.writers.pop(week_key, None)
        if writer is None:
            return
        writer.flush_labeled_tail()
        self.next_chunk_id[week_key] = int(writer.cid + 1)
        self.closed_writers[week_key].append(writer)

    def close_old_writers(self, watermark_ms: int) -> None:
        for wk in list(self.writers.keys()):
            _start_ms, end_ms = self.week_bounds[wk]
            if end_ms + GRACE_MS < watermark_ms:
                self._close_writer(wk)
        self._finalize_closed_weeks()

    def flush_all(self) -> None:
        for wk in list(self.writers.keys()):
            self._close_writer(wk)
        self.flush_queue.join()
        self.flush_queue.put(_SENTINEL_FLUSH_JOB)
        self.writer_thread.join()
        self._check_writer_exception()
        self._finalize_closed_weeks()

    def _finalize_closed_weeks(self) -> None:
        for week_key, writers in list(self.closed_writers.items()):
            if not writers:
                del self.closed_writers[week_key]
                continue
            self._build_week_meta(week_key, writers)
            del self.closed_writers[week_key]

    def _build_week_meta(self, week_key: str, writers: List[FlatWeekWriter]) -> dict:
        span = self.week_decision_span.pop(week_key, None)
        meta_path = os.path.join(self.out_root, week_key, "meta_week.json")
        chunks_meta: List[dict] = []
        for writer in writers:
            chunks_meta.extend(dict(entry) for entry in writer.chunks_meta)
        chunks_meta.sort(key=lambda entry: int(entry["chunk_id"]))
        rows_total = int(sum(int(entry["rows"]) for entry in chunks_meta))
        labels_total = rows_total
        meta = {
            "week": week_key,
            "storage_format": "flat_decision_rows_v1",
            "decision_policy": DECISION_POLICY,
            "decision_time_basis": "ob_event_time",
            "decision_stride_policy": "every_ob_event",
            **canonical_mode_fields(),
            "lookback": int(LOOKBACK),
            "window_ms": int(WINDOW_MS),
            "feature_dim_total": int(self.feature_dim_total),
            "feature_dim_core": int(self.feature_dim_core),
            "feature_dim_core_pre_pca": int(self.pre_pca_dim),
            "aux_dim": int(self.aux_dim),
            "aux_names": ["log_dt_ms", "is_trade", "log_events_100ms", "log_events_200ms", "log_events_500ms"],
            "label_dim": int(NUM_HORIZONS),
            "horizons_ms": [int(h) for h in HORIZONS_MS],
            "chunk_size_used": int(writers[0].N) if writers else 0,
            "chunks": chunks_meta,
            "chunk_count": int(len(chunks_meta)),
            "rows_total": rows_total,
            "labels_total": labels_total,
            "meta_path": os.path.join(week_key, "meta_week.json"),
            "pca": dict(self.pca_meta),
        }
        if span:
            meta["decision_ts_range"] = {"min": int(span[0]), "max": int(span[1])}
        write_json_atomic_with_backup(Path(meta_path), meta)
        self.week_metas[week_key] = meta
        print(f"[write] week={week_key} chunks={len(chunks_meta)} rows={rows_total}", flush=True)


def write_json_atomic_with_backup(path: Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(str(path.parent))
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    backup = path.with_name(f"{path.name}.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    if path.exists():
        backup.write_bytes(path.read_bytes())
    os.replace(tmp, path)

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
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def _put(self, item: Tuple[str, Optional[str], Optional[object]]):
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        try:
            for wk, ob_paths, th_paths in self.pairs:
                if self.stop_event.is_set():
                    break
                week_quality: Optional[WeekQuality] = None
                if self.collect_quality:
                    week_quality = WeekQuality(week_key=wk)
                    self.week_qualities[wk] = week_quality
                merged = _iter_week_merged_events(wk, ob_paths, th_paths, week_quality=week_quality)

                if self.stop_event.is_set():
                    break
                first_event = next(merged, None)
                if first_event is None:
                    if week_quality is not None:
                        week_quality.recompute_totals()
                        self.quality_by_week[wk] = week_quality.to_dict()
                    if not self._put(("first", wk, None)):
                        return
                    if not self._put(("eof", wk, None)):
                        return
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
                if not self._put(("first", wk, first_event)):
                    return
                for event in merged:
                    if self.stop_event.is_set():
                        return
                    if not self._put(("evt", wk, event)):
                        return
                if week_quality is not None:
                    week_quality.recompute_totals()
                    self.quality_by_week[wk] = week_quality.to_dict()
                if not self._put(("eof", wk, None)):
                    return

            if not self.stop_event.is_set():
                self._put(("eof", None, None))
        except Exception as exc:
            if not self.stop_event.is_set():
                try:
                    self.queue.put(("eof", None, exc), timeout=1.0)
                except queue.Full:
                    pass


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
    event_count = 0
    decision_count = 0
    non_decision_count = 0
    last_heartbeat = time.monotonic()
    last_event_ts = None
    last_kind = None

    feeder = EventFeeder(pairs, collect_quality=False)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    q = feeder.queue

    last_global_ts: Optional[int] = None
    try:
        while True:
            t_q = time.monotonic()
            try:
                kind, wk, payload = q.get(timeout=60.0)
            except queue.Empty:
                now = time.monotonic()
                queue_wait_s += now - t_q
                print(
                    f"[pca-heartbeat] waiting_for_events elapsed={now - stream_started:.1f}s "
                    f"events={event_count} decisions={decision_count} non_decisions={non_decision_count} "
                    f"sample_rows={sample_count} producer_alive={producer_thread.is_alive()} qsize={q.qsize()}",
                    flush=True,
                )
                continue
            queue_wait_s += time.monotonic() - t_q

            if kind == "first":
                if wk is None:
                    raise RuntimeError("Received 'first' marker without a week key")
                if payload is None:
                    continue
                event = payload
                last_wk = wk
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

            event_count += 1
            last_kind = kind
            t_evt = time.monotonic()
            ts_ms, feat_z, _dt_ms, _is_decision, _mid = fe.on_fast_event(event)
            event_proc_s += time.monotonic() - t_evt
            last_event_ts = int(ts_ms)
            if last_global_ts is not None and ts_ms < last_global_ts:
                raise ValueError(
                    "Non-monotonic timestamps across weeks during PCA stream: "
                    f"week {wk} event {ts_ms} < last {last_global_ts}"
                )
            last_global_ts = int(ts_ms)
            if not _is_decision:
                non_decision_count += 1
                if np.asarray(feat_z).shape[0] != 0:
                    raise RuntimeError("Non-decision fast path returned non-empty feature vector during PCA sampling")
                now = time.monotonic()
                if now - last_heartbeat >= 60.0:
                    print(
                        f"[pca-heartbeat] elapsed={now - stream_started:.1f}s "
                        f"events={event_count} decisions={decision_count} non_decisions={non_decision_count} "
                        f"sample_rows={sample_count} wk={wk} ts={last_event_ts} kind={last_kind} qsize={q.qsize()}",
                        flush=True,
                    )
                    last_heartbeat = now
                continue
            decision_count += 1
            sample_count += 1
            now = time.monotonic()
            if now - last_log >= 60:
                print(f"[pca-sample] rows={sample_count} last_wk={last_wk}", flush=True)
                last_log = now
            if now - last_heartbeat >= 60.0:
                print(
                    f"[pca-heartbeat] elapsed={now - stream_started:.1f}s "
                    f"events={event_count} decisions={decision_count} non_decisions={non_decision_count} "
                    f"sample_rows={sample_count} wk={wk} ts={last_event_ts} kind={last_kind} qsize={q.qsize()}",
                    flush=True,
                )
                last_heartbeat = now
            yield np.asarray(feat_z, dtype=np.float32)
    finally:
        feeder.stop()
        producer_thread.join(timeout=5.0)
        if producer_thread.is_alive():
            print(
                "[pca] feeder thread did not exit after cancellation; continuing after early PCA stop",
                flush=True,
            )
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
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as exc:
        print(f"[pca  ] sklearn unavailable ({exc}); skipping PCA fit")
        return meta

    train_set = set(train_weeks)
    train_pairs = [p for p in pairs if p[0] in train_set]
    if not train_pairs:
        print("[pca  ] No training weeks available; skipping PCA fit")
        return meta

    sample_limit = int(sample_limit)
    if sample_limit <= 0:
        raise ValueError(f"BYBIT_PCA_MAX_ROWS must be > 0, got {sample_limit}")

    sample_parts: List[np.ndarray] = []
    sample_rows_collected = 0
    total_rows = 0
    for feat in _stream_core_features(train_pairs):
        vec = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        total_rows += int(vec.shape[0])
        remaining = sample_limit - sample_rows_collected
        if remaining <= 0:
            break
        take = min(remaining, int(vec.shape[0]))
        sample_parts.append(vec[:take])
        sample_rows_collected += take
        if sample_rows_collected >= sample_limit:
            print(
                f"[pca] reached sample cap rows={sample_rows_collected} "
                f"max_rows={sample_limit}; stopping feeder",
                flush=True,
            )
            break

    print(f"[pca] sample_rows={sample_rows_collected} max_rows={sample_limit}", flush=True)
    if sample_rows_collected <= 0:
        print("[pca  ] Unable to initialise PCA (insufficient data); skipping")
        return meta

    sample_array = np.concatenate(sample_parts, axis=0).astype(np.float32, copy=False)
    n_components = _select_pca_components(sample_array, target_var)
    if n_components <= 0:
        print("[pca  ] Unable to select PCA components; skipping")
        return meta

    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(sample_array)
    fitted_rows = int(sample_array.shape[0])

    model_path = os.path.join(out_root, model_filename)
    ensure_dir(os.path.dirname(model_path))
    np.savez(
        model_path,
        mean=pca.mean_.astype(np.float32, copy=False),
        components=pca.components_.astype(np.float32, copy=False),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32, copy=False),
    )

    meta.update(
        {
            "applied": True,
            "k": int(n_components),
            "model_path": model_filename,
            "rows_fitted": int(fitted_rows),
            "rows_total": int(total_rows),
            "sample_rows": int(sample_array.shape[0]),
            "max_rows": int(sample_limit),
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

    pending_decisions: deque[Tuple[str, int, int]] = deque()
    last_decision_ts_ms: Optional[int] = None

    pre_pca_dim = int(pca_mean.shape[0]) if pca_mean is not None else 0
    pca_k = int(pca_components.shape[0]) if pca_components is not None else 0
    F = None if pca_k <= 0 else int(pca_k + AUX_DIM)
    router: Optional[FlatWeekRouter] = None
    total_feature_rows = 0
    total_labels = 0

    ds_start, ds_end = _compute_dataset_span(pairs)
    start_iso = ds_start.date().isoformat() if ds_start else None
    end_iso = ds_end.date().isoformat() if ds_end else None

    week_index = _build_week_index(pairs)

    print(
        f"[start] ingest weeks={len(pairs)} storage=flat_decision_rows_v1 L={LOOKBACK} budget={RAM_BUDGET}MB"
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
        ts_ms, feat_z, dt_ms, is_decision, mid = fe.on_fast_event(event)
        event_proc_s += time.monotonic() - t_evt

        if not is_decision:
            if np.asarray(feat_z).shape[0] != 0:
                raise RuntimeError("Non-decision fast path returned non-empty feature vector")
            continue

        core_pre_pca = np.asarray(feat_z, dtype=np.float32, copy=False)
        if pca_components is None or pca_mean is None:
            raise RuntimeError("Phase 1B flat ingest requires PCA model arrays")
        if core_pre_pca.shape[-1] != pca_mean.shape[0]:
            raise ValueError(
                f"PCA mean/components dimension {pca_mean.shape[0]} does not match feature dimension {core_pre_pca.shape[-1]}"
            )

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
        aux_tail = build_aux_tail(fe, is_trade=False, dt_ms=dt_tick)
        if F is None:
            pre_pca_dim = int(pca_mean.shape[0])
            pca_k = int(pca_components.shape[0])
            F = int(pca_k + AUX_DIM)
        if router is None:
            print(f"[first-row] feature_dim_core={int(pca_k)} aux_dim={AUX_DIM} feature_dim_total={int(F)}", flush=True)
            router = FlatWeekRouter(
                out_root, int(pca_k), int(pre_pca_dim), pca_mean, pca_components, AUX_DIM,
                RAM_BUDGET, CHUNK_SIZE, week_index, pca_meta=pca_summary,
            )

        if is_duplicate_decision_ts:
            if not pending_decisions:
                raise RuntimeError("Duplicate OB timestamp cannot update state because no pending decision exists")
            prev_week_key, _prev_row_idx, _prev_ts = pending_decisions[-1]
            week_key, row_idx = router.overwrite_latest_feature_row(int(ts_ms), core_pre_pca, aux_tail)
            if week_key != prev_week_key:
                raise RuntimeError("Duplicate timestamp mapped to a different week during overwrite")
            pending_decisions[-1] = (week_key, row_idx, int(ts_ms))
        else:
            week_key, row_idx = router.append_feature_row(int(ts_ms), core_pre_pca, aux_tail)
            pending_decisions.append((week_key, row_idx, int(ts_ms)))
            labeler.on_decision(int(ts_ms))
            total_feature_rows += 1

        matured = labeler.on_event(int(ts_ms), float(mid))
        last_decision_ts_ms = int(ts_ms)
        if matured is None:
            raise RuntimeError("Matured labels were not produced for OB event")
        for yy in matured:
            if not pending_decisions:
                raise RuntimeError("Matured label available but no pending flat rows to pair")
            label_week, label_row_idx, _label_ts = pending_decisions.popleft()
            router.set_label(label_week, label_row_idx, yy.astype(np.float32, copy=False))
            total_labels += 1

        t_router = time.monotonic()
        if router is not None:
            router.close_old_writers(int(ts_ms))
        router_housekeeping_s += time.monotonic() - t_router
        
        if time.monotonic() - last_log >= 300:
            print(f"[tok  ] rows={total_feature_rows} labels={total_labels} weeks={week_counter}/{week_total} "
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
        "storage_format": "flat_decision_rows_v1",
        "decision_policy": DECISION_POLICY,
        "decision_time_basis": "ob_event_time",
        "decision_stride_policy": "every_ob_event",
        "label_delta_ms": 0,
        **canonical_mode_fields(),
        "lookback": int(LOOKBACK),
        "window_ms": int(WINDOW_MS),
        "feature_dim_total": feature_dim_total,
        "feature_dim_core": feature_dim_core,
        "feature_dim_core_pre_pca": int(pre_pca_dim),
        "aux_names": ["log_dt_ms", "is_trade", "log_events_100ms", "log_events_200ms", "log_events_500ms"],
        "dtype": "float32",
        "ram_budget_mb": int(RAM_BUDGET),
        "chunk_size_used": 0 if (router is None or router.chunk_size_used == 0) else int(router.chunk_size_used),
        "aux_dim": int(AUX_DIM),
        "label_dim": label_dim,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "core_dtype": "float32",
        "total_feature_rows": int(rows_via_week_metas),
        "total_labels": int(rows_via_week_metas),
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
    meta["split_protocol"] = "five_week_cmssl2w_val_test_rl_eval_v1"
    meta["pipeline_splits"] = build_pipeline_splits(weeks_in_order)
    meta["label_units"] = "signed_log_return_bps"
    meta["target_transform"] = "signed_log_return_bps"
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

    write_json_atomic_with_backup(Path(out_root) / "meta.json", meta)

    chunk_summary = 0 if router is None else sum(router.week_counts.values())

    print(
        f"[done ] dataset weeks={len(pairs)} total_rows={total_feature_rows} total_labels={total_labels} "
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
    if len(pairs) != 5:
        raise ValueError(
            f"Phase 1 requires exactly 5 distinct consecutive weeks of data after BYBIT_WEEKS filtering; found {len(pairs)}."
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
    week1, week2, week3, week4, week5 = selected_weeks
    print(
        f"[split] protocol=five_week_cmssl2w_val_test_rl_eval_v1 cmssl.train={week1},{week2} cmssl.val={week3} cmssl.test={week4} rl={week4} eval={week5}"
    )
    pca_fit_meta = maybe_fit_pca_model(
        pairs,
        OUT_ROOT,
        [week1, week2],
        PCA_VAR_TARGET,
        PCA_MAX_ROWS,
        PCA_BATCH_SIZE,
        PCA_MODEL_FILENAME,
        PCA_USE_EXISTING,
    )

    process_all(pairs, OUT_ROOT, pca_fit_meta)

if __name__ == "__main__":
    main()
