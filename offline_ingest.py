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
  BYBIT_PCA_SELECT_MODE=max_components
  BYBIT_PCA_MAX_COMPONENTS=506
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

import os, sys, csv, json, re, time, logging, hashlib, math
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
PCA_SELECT_MODE     = os.environ.get("BYBIT_PCA_SELECT_MODE", "max_components").strip().lower()
PCA_MAX_COMPONENTS  = int(os.environ.get("BYBIT_PCA_MAX_COMPONENTS", "506"))
PCA_BATCH_SIZE      = int(os.environ.get("BYBIT_PCA_BATCH", "4096"))
PCA_MODEL_FILENAME  = os.environ.get("BYBIT_PCA_MODEL", "pca_model.npz")
PCA_USE_EXISTING    = int(os.environ.get("BYBIT_PCA_USE_EXISTING", "0"))
PCA_WEEKDAY_ROWS_PER_DAY = 20_000
PCA_SAMPLE_START_UTC_HOUR = 14
PCA_SAMPLE_START_UTC_MINUTE = 30
if PCA_SELECT_MODE != "max_components":
    raise ValueError("BYBIT_PCA_SELECT_MODE must be 'max_components' for the 1s maker contract")

# Memory & chunking
RAM_BUDGET  = int(os.environ.get("BYBIT_RAM_BUDGET_MB", "512"))
CHUNK_SIZE  = int(os.environ.get("BYBIT_CHUNK_SIZE", "0"))  # 0 = auto-size from RAM budget; >0 = explicit fixed override
FLUSH_WORKERS = int(os.environ.get("BYBIT_FLUSH_WORKERS", "4"))
DECISION_POLICY = "ob_event_time"
FOUR_WEEK_PROTOCOL = "four_week_cmssl_val_test_rl_eval_v2"
FIVE_WEEK_PROTOCOL = "five_week_cmssl2w_val_test_rl_eval_v1"




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
APPEND_MISSING_WEEKS = _env_bool_int("BYBIT_APPEND_MISSING_WEEKS", 0)
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
    WINDOW_MS,
    AUX_DIM,
    FEATURE_SCHEMA,
    AUX_SCHEMA,
    FEATURE_AUX_TAIL,
    _open_text,
    timestamp_to_ms_half_even,
    CHECKPOINT_SCHEMA,
    TARGET_TRANSFORM,
    TARGET_TASK,
    LOW_ABS_TRIM_FRACTION,
    HIGH_ABS_TRIM_FRACTION,
)  # keep shared model/data constants only; ingest helpers are local below
# LOOKBACK is a shared model constant from CMSSL17 (single source of truth).
FINAL_FEATURE_DIM = 512
PCA_CORE_COMPONENTS = FINAL_FEATURE_DIM - AUX_DIM
PCA_MAX_COMPONENTS = int(os.environ.get("BYBIT_PCA_MAX_COMPONENTS", str(PCA_CORE_COMPONENTS)))
if PCA_MAX_COMPONENTS != PCA_CORE_COMPONENTS:
    raise ValueError(f"BYBIT_PCA_MAX_COMPONENTS must equal PCA_CORE_COMPONENTS={PCA_CORE_COMPONENTS} for final feature_dim_total=512")

GRACE_MS = max(int(h) for h in HORIZONS_MS)
EVENT_QUEUE_MAXSIZE = int(os.environ.get("BYBIT_EVENT_QUEUE_MAXSIZE", "4096"))
EVENT_QUEUE_FULL_LOG_SEC = float(os.environ.get("BYBIT_EVENT_QUEUE_FULL_LOG_SEC", "60"))
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


WeekPair = Tuple[str, List[str], List[str]]

OB_DAILY_RE = re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2})_BTCUSDT_.*ob.*\.zip$", re.IGNORECASE)
TH_DAILY_RE = re.compile(r"^BTCUSDT(?P<d>\d{4}-\d{2}-\d{2})\.csv(?:\.gz|\.gzip)?$", re.IGNORECASE)
WEEK_KEY_RE = re.compile(r"^(\d{2}-\d{2}-\d{4})-to-(\d{2}-\d{2}-\d{4})$")


def _ob_ext_rank(path: str) -> int:
    lower = str(path).lower()
    if lower.endswith(".data.zip"):
        return 0
    if lower.endswith(".zip"):
        return 1
    return 2


def _choose_preferred_daily_ob_path(candidates: List[str]) -> str:
    def _key(path: str) -> Tuple[int, str, str]:
        p = Path(path)
        return (_ob_ext_rank(path), p.name, str(p))

    return min(candidates, key=_key)


def _parse_ymd_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daily_path_day(path: str, side: str, *, week_key: str = "") -> date:
    name = os.path.basename(path)
    pattern = OB_DAILY_RE if side == "OB" else TH_DAILY_RE
    m = pattern.match(name)
    if not m:
        wk_suffix = f" in week={week_key}" if week_key else ""
        raise ValueError(f"Could not parse daily date for {side} file '{name}'{wk_suffix}")
    return _parse_ymd_date(m.group("d"))


def _parse_week_key_any(week_key: str) -> Tuple[datetime, datetime, str]:
    m = WEEK_KEY_RE.fullmatch(str(week_key).strip())
    if not m:
        raise ValueError(f"Unrecognized week key: {week_key}")
    start_dt = datetime.strptime(m.group(1), "%d-%m-%Y").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(m.group(2), "%d-%m-%Y").replace(tzinfo=timezone.utc)
    if end_dt < start_dt:
        raise ValueError(f"Week key end before start: {week_key}")
    return start_dt, end_dt, week_key


def _format_week_key(start_day: date, end_day: date) -> str:
    return f"{start_day.strftime('%d-%m-%Y')}-to-{end_day.strftime('%d-%m-%Y')}"


def _parse_requested_weeks(raw: str) -> List[str]:
    return [wk.strip() for wk in re.split(r"[\s,]+", str(raw or "")) if wk.strip()]


def _sort_pairs_by_end(pairs: List[WeekPair]) -> List[WeekPair]:
    return sorted(pairs, key=lambda p: _parse_week_key_any(p[0])[1])


def _assert_week_order(pairs: List[WeekPair]) -> None:
    if not pairs:
        return
    prev_end: Optional[datetime] = None
    prev_wk: Optional[str] = None
    for wk, _ob_paths, _th_paths in pairs:
        _start, end, _ = _parse_week_key_any(wk)
        if prev_end is not None and end <= prev_end:
            raise ValueError(f"Weeks not strictly ordered by end date: prev={prev_wk} curr={wk}")
        prev_end = end
        prev_wk = wk


def _assert_weeks_consecutive(pairs: List[WeekPair]) -> None:
    if not pairs:
        return
    ordered = _sort_pairs_by_end(list(pairs))
    for idx in range(1, len(ordered)):
        prev_wk = ordered[idx - 1][0]
        curr_wk = ordered[idx][0]
        _prev_start, prev_end, _ = _parse_week_key_any(prev_wk)
        curr_start, _curr_end, _ = _parse_week_key_any(curr_wk)
        expected = (prev_end + ONE_DAY).date()
        if curr_start.date() != expected:
            raise ValueError(
                "Weeks are not consecutive: "
                f"prev={prev_wk} (ends {prev_end.date().isoformat()}) "
                f"curr={curr_wk} (starts {curr_start.date().isoformat()})"
            )


def _build_ob_daily_map(ob_dir: str) -> Dict[date, str]:
    by_day: Dict[date, str] = {}
    for path in sorted(Path(ob_dir).iterdir()):
        if not path.is_file():
            continue
        m = OB_DAILY_RE.match(path.name)
        if not m:
            continue
        day = _parse_ymd_date(m.group("d"))
        cand = str(path)
        cur = by_day.get(day)
        if cur is None:
            by_day[day] = cand
            continue
        chosen = _choose_preferred_daily_ob_path([cur, cand])
        by_day[day] = chosen
        print(
            f"[warn] duplicate daily OB for {day.isoformat()}: "
            f"{Path(cur).name} vs {path.name}; using {Path(chosen).name}",
            flush=True,
        )
    return by_day


def _build_th_daily_map(th_dir: str) -> Dict[date, str]:
    by_day: Dict[date, str] = {}
    for path in sorted(Path(th_dir).iterdir()):
        if not path.is_file():
            continue
        m = TH_DAILY_RE.match(path.name)
        if not m:
            continue
        day = _parse_ymd_date(m.group("d"))
        cur = by_day.get(day)
        cand = str(path)
        if cur is None:
            by_day[day] = cand
            continue
        keep = min([cur, cand], key=lambda p: (0 if str(p).lower().endswith(".csv.gz") else 1, Path(p).name, str(p)))
        by_day[day] = keep
        print(
            f"[warn] duplicate daily TH for {day.isoformat()}: "
            f"{Path(cur).name} vs {path.name}; using {Path(keep).name}",
            flush=True,
        )
    return by_day


def pair_weeks(ob_dir: str, th_dir: str) -> List[WeekPair]:
    ob_by_day = _build_ob_daily_map(ob_dir)
    if not ob_by_day:
        return []

    if not USE_TRADES:
        days = sorted(ob_by_day.keys())
        pairs: List[WeekPair] = []
        for i in range(0, len(days), 7):
            block = days[i : i + 7]
            if len(block) < 7:
                continue
            if any((block[j] - block[j - 1]).days != 1 for j in range(1, len(block))):
                continue
            week_key = _format_week_key(block[0], block[-1])
            ob_paths = [ob_by_day[d] for d in block]
            pairs.append((week_key, ob_paths, []))
        return _sort_pairs_by_end(pairs)

    th_by_day = _build_th_daily_map(th_dir)
    common_days = sorted(set(ob_by_day.keys()) & set(th_by_day.keys()))
    pairs = []
    for i in range(0, len(common_days), 7):
        block = common_days[i : i + 7]
        if len(block) < 7:
            continue
        if any((block[j] - block[j - 1]).days != 1 for j in range(1, len(block))):
            continue
        ob_paths = [ob_by_day[d] for d in block]
        th_paths = [th_by_day[d] for d in block]
        if len(ob_paths) != len(th_paths):
            continue
        week_key = _format_week_key(block[0], block[-1])
        pairs.append((week_key, ob_paths, th_paths))
    return _sort_pairs_by_end(pairs)


def _try_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _trade_iter_precise(rows: Iterable[Tuple[int, int, dict]]) -> Iterable[Tuple[int, int, dict]]:
    for ts_raw, line_no, row in rows:
        t_raw = row.get("timestamp")
        if t_raw is None:
            t_raw = row.get("ts")
        if t_raw is None:
            t_raw = row.get("T")
        if t_raw is None:
            yield int(ts_raw), int(line_no), row
            continue
        try:
            ts_ms = timestamp_to_ms_half_even(t_raw)
        except Exception:
            yield int(ts_raw), int(line_no), row
            continue
        yield int(ts_ms), int(line_no), row


def safe_ob_iter(ob_path: str, day_start_ms: int, day_end_ms: int, dq_day: Optional[DayQuality] = None):
    last_ts: Optional[int] = None
    with _open_text(ob_path) as f:
        for line_no, line in enumerate(f, start=1):
            if dq_day is not None:
                dq_day.increment_counter("ob", "total")
            if not line:
                continue
            try:
                raw = fast_json_loads(line)
            except Exception:
                if dq_day is not None:
                    dq_day.increment_counter("ob", "bad_json")
                    dq_day.append_example("ob_bad_json", {"line": int(line_no), "sample": line[:200].rstrip()})
                continue
            if not isinstance(raw, dict):
                if dq_day is not None:
                    dq_day.increment_counter("ob", "bad_shape")
                    dq_day.append_example("ob_bad_shape", {"line": int(line_no)})
                continue
            data = raw.get("data", raw)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                data = data[0]
            if not isinstance(data, dict):
                if dq_day is not None:
                    dq_day.increment_counter("ob", "bad_payload")
                continue

            ts_raw = raw.get("ts")
            if ts_raw is None:
                ts_raw = raw.get("cts")
            if ts_raw is None:
                ts_raw = data.get("ts")
            if ts_raw is None:
                ts_raw = data.get("cts")
            ts_ms = _try_int(ts_raw)
            if ts_ms is None:
                if dq_day is not None:
                    dq_day.increment_counter("ob", "bad_ts")
                    dq_day.append_example("ob_bad_ts", {"line": int(line_no), "ts": ts_raw})
                continue

            if dq_day is not None:
                dq_day.update_raw_ts(ts_ms)
            if BYBIT_DAY_CLIP and not (int(day_start_ms) <= ts_ms < int(day_end_ms)):
                if dq_day is not None:
                    dq_day.increment_counter("ob", "day_clipped")
                continue

            out_ts = int(ts_ms)
            if last_ts is not None and out_ts < last_ts:
                backstep = int(last_ts - out_ts)
                if BYBIT_STRICT_DATA:
                    raise ValueError(
                        f"Non-decreasing OB timestamps violated in {ob_path}: prev={last_ts} curr={out_ts}"
                    )
                if backstep <= BYBIT_TS_BACKSTEP_CLAMP_MS:
                    out_ts = int(last_ts)
                    if dq_day is not None:
                        dq_day.increment_counter("ob", "clamped_backstep")
                else:
                    if dq_day is not None:
                        dq_day.increment_counter("ob", "dropped_big_backstep")
                    continue
            last_ts = out_ts
            if dq_day is not None:
                dq_day.update_output_ts(out_ts)

            tp_raw = raw.get("type")
            if tp_raw is None:
                tp_raw = data.get("type")
            if tp_raw is None:
                tp_raw = raw.get("DataType")
            tp_code = _compact_ob_type_code(tp_raw)
            bids = _compact_book_levels(data.get("b", []))
            asks = _compact_book_levels(data.get("a", []))
            seq = _try_int(raw.get("seq"), 0)
            yield ("ob", int(out_ts), int(seq or 0), int(tp_code), bids, asks)


def safe_th_iter(th_path: str, day_start_ms: int, day_end_ms: int, dq_day: Optional[DayQuality] = None):
    last_ts: Optional[int] = None
    with _open_text(th_path) as f:
        reader = csv.DictReader(f)
        base_rows = (
            ((_try_int((row or {}).get("ts"), 0) or 0), line_no, row or {})
            for line_no, row in enumerate(reader, start=2)
        )
        for ts_ms_raw, line_no, row in _trade_iter_precise(base_rows):
            if dq_day is not None:
                dq_day.increment_counter("th", "total")
            ts_ms = int(ts_ms_raw)
            if dq_day is not None:
                dq_day.update_raw_ts(ts_ms)
            if BYBIT_DAY_CLIP and not (int(day_start_ms) <= ts_ms < int(day_end_ms)):
                if dq_day is not None:
                    dq_day.increment_counter("th", "day_clipped")
                continue

            out_ts = int(ts_ms)
            if last_ts is not None and out_ts < last_ts:
                backstep = int(last_ts - out_ts)
                if BYBIT_STRICT_DATA:
                    raise ValueError(
                        f"Non-decreasing TH timestamps violated in {th_path}: prev={last_ts} curr={out_ts}"
                    )
                if backstep <= BYBIT_TS_BACKSTEP_CLAMP_MS:
                    out_ts = int(last_ts)
                    if dq_day is not None:
                        dq_day.increment_counter("th", "clamped_backstep")
                else:
                    if dq_day is not None:
                        dq_day.increment_counter("th", "dropped_big_backstep")
                    continue

            price_raw = row.get("price")
            size_raw = row.get("size")
            side_raw = row.get("side")
            try:
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                if dq_day is not None:
                    dq_day.increment_counter("th", "bad_price_or_size")
                    dq_day.append_example(
                        "th_bad_price_or_size",
                        {"line": int(line_no), "price": price_raw, "size": size_raw},
                    )
                continue
            seq = _try_int(row.get("seq"), 0) or 0
            side_code = _compact_trade_side_code(side_raw)
            tick_dir_code = _compact_tick_dir_code(row.get("tickDirection"))
            is_rpi = _compact_is_rpi_code(row.get("RPI", row.get("rpi")))

            last_ts = out_ts
            if dq_day is not None:
                dq_day.update_output_ts(out_ts)
            yield ("trade", int(out_ts), int(seq), float(price), float(size), int(side_code), int(tick_dir_code), int(is_rpi))


def _event_ts(event: Any) -> int:
    return int(event[1])


def build_aux_tail(fe: FeatureEngine, dt_ms: float) -> np.ndarray:
    return np.asarray(
        [
            np.log1p(float(dt_ms)),
            np.log1p(fe.event_density(100)),
            np.log1p(fe.event_density(200)),
            np.log1p(fe.event_density(500)),
            np.log1p(fe.event_density(1_000)),
            np.log1p(fe.event_density(3_000)),
        ],
        dtype=np.float32,
    )


def build_token(fe: FeatureEngine, feat_z, dt_ms: float) -> np.ndarray:
    core = np.asarray(feat_z, dtype=np.float32)
    aux = build_aux_tail(fe, dt_ms)
    return np.concatenate([core, aux], axis=0).astype(np.float32, copy=False)


def build_pipeline_splits(
    weeks_in_order: List[str],
    week_metas: Dict[str, dict],
    protocol: str,
) -> dict:
    if protocol == FOUR_WEEK_PROTOCOL:
        if len(weeks_in_order) != 4:
            raise ValueError(f"Expected exactly 4 weeks for {FOUR_WEEK_PROTOCOL}; got {len(weeks_in_order)}")
        week1, week2, week3, week4 = weeks_in_order
        cmssl_train_weeks = [week1]
        cmssl_val_week = week2
        cmssl_test_week = week3
        rl_week = week3
        eval_week = week4
    elif protocol == FIVE_WEEK_PROTOCOL:
        if len(weeks_in_order) != 5:
            raise ValueError(f"Expected exactly 5 weeks for {FIVE_WEEK_PROTOCOL}; got {len(weeks_in_order)}")
        week1, week2, week3, week4, week5 = weeks_in_order
        cmssl_train_weeks = [week1, week2]
        cmssl_val_week = week3
        cmssl_test_week = week4
        rl_week = week4
        eval_week = week5
    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")

    def _range_for(week_key: str) -> Tuple[int, int]:
        week_meta = week_metas.get(week_key)
        if not isinstance(week_meta, dict):
            raise KeyError(f"Missing week metadata for split week '{week_key}'")
        decision_range = week_meta.get("decision_ts_range")
        if not isinstance(decision_range, dict):
            raise KeyError(f"Week '{week_key}' missing decision_ts_range in metadata")
        if "min" not in decision_range or "max" not in decision_range:
            raise KeyError(f"Week '{week_key}' decision_ts_range must contain min/max")
        start = int(decision_range["min"])
        end_exclusive = int(decision_range["max"]) + 1
        if start >= end_exclusive:
            raise ValueError(f"Invalid decision_ts_range for week '{week_key}': {start}..{end_exclusive}")
        return start, end_exclusive

    train_start = _range_for(cmssl_train_weeks[0])[0]
    train_end = _range_for(cmssl_train_weeks[-1])[1]
    vals, vale = _range_for(cmssl_val_week)
    tests, teste = _range_for(cmssl_test_week)
    evals, evale = _range_for(eval_week)

    span_rl = max(1, teste - tests)
    rl_40 = tests + int(np.floor(0.4 * span_rl))
    rl_70 = tests + int(np.floor(0.7 * span_rl))
    rl_40 = min(max(rl_40, tests + 1), teste - 2)
    rl_70 = min(max(rl_70, rl_40 + 1), teste - 1)

    return {
        "protocol": protocol,
        "cmssl": {
            "train": {
                "weeks": list(cmssl_train_weeks),
                "decision_ts_range": {"start": int(train_start), "end": int(train_end)},
                "start": int(train_start),
                "end": int(train_end),
            },
            "val": {
                "week": cmssl_val_week,
                "decision_ts_range": {"start": int(vals), "end": int(vale)},
                "start": int(vals),
                "end": int(vale),
            },
            "test": {
                "week": cmssl_test_week,
                "decision_ts_range": {"start": int(tests), "end": int(teste)},
                "start": int(tests),
                "end": int(teste),
            },
        },
        "rl": {
            "train": {"week": rl_week, "decision_ts_range": {"start": int(tests), "end": int(rl_40)}},
            "val": {"week": rl_week, "decision_ts_range": {"start": int(rl_40), "end": int(rl_70)}},
            "test": {"week": rl_week, "decision_ts_range": {"start": int(rl_70), "end": int(teste)}},
        },
        "eval": {
            "full": {
                "week": eval_week,
                "decision_ts_range": {"start": int(evals), "end": int(evale)},
                "start": int(evals),
                "end": int(evale),
            },
        },
    }


def iter_weekly_event_stream(pairs: List[WeekPair], collect_quality: bool = True):
    feeder = EventFeeder(pairs, collect_quality=collect_quality)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    try:
        while True:
            kind, wk, payload = feeder.queue.get()
            if kind == "eof" and wk is None:
                if isinstance(payload, Exception):
                    raise payload
                break
            yield kind, wk, payload
    finally:
        producer_thread.join(timeout=2.0)


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
    pre_pca_core: np.ndarray
    aux: np.ndarray
    ts: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    final_feature_dim: int
    aux_dim: int


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
    def __init__(self, out_dir: str, final_feature_dim: int, pre_pca_dim: int, pca_mean: np.ndarray, pca_components: np.ndarray, aux_dim: int, ram_budget_mb: int, chunk_size_override: int = 0, start_chunk_id: int = 0, week_key: str = "", flush_callback: Optional[Callable[[object], None]] = None):
        self.out_dir = out_dir
        self.week_key = str(week_key)
        self.final_feature_dim = int(final_feature_dim)
        self.pre_pca_dim = int(pre_pca_dim)
        self.aux_dim = int(aux_dim)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float32, copy=False)
        self.pca_components = np.asarray(pca_components, dtype=np.float32, copy=False)
        if self.pca_mean.ndim != 1:
            raise ValueError("pca_mean must be 1D")
        if self.pca_components.ndim != 2:
            raise ValueError("pca_components must be 2D")
        if self.pca_components.shape[1] != self.pca_mean.shape[0] or self.pca_mean.shape[0] != self.pre_pca_dim:
            raise ValueError("PCA dimension mismatch")
        if self.final_feature_dim != self.pca_components.shape[0] + self.aux_dim:
            raise ValueError("final_feature_dim mismatch")
        if self.aux_dim != AUX_DIM:
            raise ValueError("aux_dim must equal AUX_DIM")
        self.flush_callback = flush_callback
        pca_k = int(self.pca_components.shape[0])
        bytes_per_row = (4 * self.pre_pca_dim) + (4 * self.aux_dim) + 8 + (4 * self.final_feature_dim) + (4 * pca_k)
        if chunk_size_override > 0:
            self.N = int(chunk_size_override)
        else:
            auto_n = max(256, int((ram_budget_mb * 1024 * 1024) // max(1, bytes_per_row)))
            safety_cap = max(256, int((2 * 1024 * 1024 * 1024) // max(1, bytes_per_row)))
            self.N = min(auto_n, safety_cap)

        self.pre_pca_core = np.empty((self.N, self.pre_pca_dim), dtype=np.float32)
        self.aux = np.empty((self.N, self.aux_dim), dtype=np.float32)
        self.ts = np.empty((self.N,), dtype=np.int64)
        self.i = 0
        self.cid = int(start_chunk_id)
        self.rows_total = 0
        self.chunks_meta: List[Dict[str, Any]] = []

    def append_row_pre_pca(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> int:
        if self.i >= self.N:
            self.flush()
        core = np.asarray(core_pre_pca, dtype=np.float32, copy=False)
        aux = np.asarray(aux_tail, dtype=np.float32, copy=False)
        if core.shape != (self.pre_pca_dim,):
            raise ValueError(f"Core pre-PCA dim mismatch: {core.shape} != {(self.pre_pca_dim,)}")
        if aux.shape != (self.aux_dim,):
            raise ValueError(f"Aux dim mismatch: {aux.shape} != {(self.aux_dim,)}")
        if not np.all(np.isfinite(core)) or not np.all(np.isfinite(aux)):
            raise ValueError("Non-finite input features")
        self.pre_pca_core[self.i] = core
        self.aux[self.i] = aux
        self.ts[self.i] = int(ts_decision_ms)
        row_idx = self.rows_total + self.i
        self.i += 1
        return int(row_idx)

    def overwrite_latest_row(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> int:
        if self.i <= 0:
            raise RuntimeError("Cannot overwrite latest feature row in an empty open chunk")
        core = np.asarray(core_pre_pca, dtype=np.float32, copy=False)
        aux = np.asarray(aux_tail, dtype=np.float32, copy=False)
        if core.shape != (self.pre_pca_dim,):
            raise ValueError(f"Core pre-PCA dim mismatch: {core.shape} != {(self.pre_pca_dim,)}")
        if aux.shape != (self.aux_dim,):
            raise ValueError(f"Aux dim mismatch: {aux.shape} != {(self.aux_dim,)}")
        if not np.all(np.isfinite(core)) or not np.all(np.isfinite(aux)):
            raise ValueError("Non-finite input features")
        idx = self.i - 1
        self.pre_pca_core[idx] = core
        self.aux[idx] = aux
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
            pre_pca_core=self.pre_pca_core,
            aux=self.aux,
            ts=self.ts,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            final_feature_dim=self.final_feature_dim,
            aux_dim=self.aux_dim,
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
        self.pre_pca_core = np.empty((self.N, self.pre_pca_dim), dtype=np.float32)
        self.aux = np.empty((self.N, self.aux_dim), dtype=np.float32)
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
_FLUSH_PERF = {"pca_project_s": 0.0, "feature_flush_s": 0.0}


def _persist_flush_job(job: object) -> None:
    if isinstance(job, FeatureFlushJob):
        t0 = time.perf_counter()
        n = int(job.row_count)
        X = job.pre_pca_core[:n].astype(np.float32, copy=False)
        aux = job.aux[:n].astype(np.float32, copy=False)
        t1 = time.perf_counter()
        centered = X - job.pca_mean[None, :]
        projected = centered @ job.pca_components.T
        _FLUSH_PERF["pca_project_s"] += float(time.perf_counter() - t1)
        out = np.empty((n, int(job.final_feature_dim)), dtype=np.float32)
        pca_k = int(job.pca_components.shape[0])
        out[:, :pca_k] = projected.astype(np.float32, copy=False)
        out[:, pca_k:] = aux
        if out.shape != (n, int(job.final_feature_dim)):
            raise ValueError(f"Projected shape mismatch: {out.shape} != {(n, int(job.final_feature_dim))}")
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite projected feature output")
        np.save(os.path.join(job.out_dir, job.features_file), out)
        np.save(os.path.join(job.out_dir, job.ts_file), job.ts[: job.row_count])
        _FLUSH_PERF["feature_flush_s"] += float(time.perf_counter() - t0)
        return
    if isinstance(job, LabelFlushJob):
        np.save(os.path.join(job.out_dir, job.row_idx_file), job.row_idx[: job.label_count])
        np.save(os.path.join(job.out_dir, job.label_ts_file), job.label_ts[: job.label_count])
        np.save(os.path.join(job.out_dir, job.y_file), job.y[: job.label_count])
        return
    raise TypeError(f"Unsupported flush job type: {type(job)!r}")


class FlatWeekRouter:
    def __init__(self, out_root: str, final_feature_dim: int, pre_pca_dim: int, pca_mean: np.ndarray, pca_components: np.ndarray, aux_dim: int, ram_budget_mb: int, chunk_size_override: int, week_index: List[Tuple[str, int, int]], pca_meta: Optional[dict] = None):
        self.out_root = out_root
        self.feature_dim = int(final_feature_dim)
        self.pre_pca_dim = int(pre_pca_dim)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float32, copy=False)
        self.pca_components = np.asarray(pca_components, dtype=np.float32, copy=False)
        self.aux_dim = int(aux_dim)
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
            self.pre_pca_dim,
            self.pca_mean,
            self.pca_components,
            self.aux_dim,
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

    def append_feature_row(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> Tuple[str, int]:
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_feature_writer(wk)
        row_idx = writer.append_row_pre_pca(int(ts_decision_ms), core_pre_pca, aux_tail)
        self.week_rows_total[wk] = max(self.week_rows_total[wk], int(row_idx) + 1)
        if wk not in self.week_decision_span:
            self.week_decision_span[wk] = [int(ts_decision_ms), int(ts_decision_ms)]
        else:
            span = self.week_decision_span[wk]
            span[0] = min(span[0], int(ts_decision_ms))
            span[1] = max(span[1], int(ts_decision_ms))
        return wk, int(row_idx)

    def overwrite_latest_feature_row(self, ts_decision_ms: int, core_pre_pca: np.ndarray, aux_tail: np.ndarray) -> Tuple[str, int]:
        self._check_writer_exception()
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_feature_writer(wk)
        row_idx = writer.overwrite_latest_row(int(ts_decision_ms), core_pre_pca, aux_tail)
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
            "window_ms": int(WINDOW_MS),
            "decision_stride_policy": "every_ob_event",
            "label_delta_ms": 0,
            # Labels remain signed raw log-return bps; direction/conditional magnitude targets are derived downstream at train time.
            "label_units": "signed_log_return_bps",
            "feature_schema": FEATURE_SCHEMA,
            "aux_schema": AUX_SCHEMA,
            "target_task": TARGET_TASK,
            "target_transform": TARGET_TRANSFORM,
            "low_abs_trim_fraction": float(LOW_ABS_TRIM_FRACTION),
            "high_abs_trim_fraction": float(HIGH_ABS_TRIM_FRACTION),
            "checkpoint_schema_expected": CHECKPOINT_SCHEMA,
            **canonical_mode_fields(),
            "lookback": int(LOOKBACK),
            "feature_dim_total": int(self.feature_dim),
            "feature_dim_core": int(self.feature_dim - AUX_DIM),
            "feature_dim_core_pre_pca": int(self.pca_meta.get("feature_dim_core_pre_pca", 0)),
            "feature_names_pre_pca": list(self.pca_meta.get("feature_names_pre_pca", [])),
            "feature_names_hash": str(self.pca_meta.get("feature_names_hash", "")),
            "aux_dim": int(AUX_DIM),
            "aux_names": list(FEATURE_AUX_TAIL),
            "label_dim": int(NUM_HORIZONS),  # one signed-return label per configured horizon
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
            raise ValueError("Missing required PCA metadata")

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

        if not to_close:
            return

        for wk in to_close:
            self._close_week_writers(wk)

        # _close_week_writers() flushes writers, which enqueues chunk write jobs.
        # meta_week.json must only be built after those queued writes have completed,
        # otherwise metadata could reference files still being written.
        self._check_writer_exception()
        self.flush_queue.join()
        self._check_writer_exception()

        # Build the exact same meta_week.json files that flush_all() would build,
        # but as soon as completed weeks are safely closed. This makes completed
        # weeks reusable after a later crash.
        self._finalize_closed_weeks()

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
        ("pca_project_s", "pca_project"),
        ("feature_flush_s", "feature_flush"),
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

    def _assert_daily_side_sorted(paths: List[str], side: str):
        prev_day: Optional[date] = None
        prev_name: Optional[str] = None
        for path in paths:
            day = _daily_path_day(path, side, week_key=week_key)
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
            ob_day = _daily_path_day(ob_p, "OB", week_key=week_key)
            th_day = _daily_path_day(th_p, "TH", week_key=week_key)
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
            day = _daily_path_day(ob_path, "OB", week_key=week_key)
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
        day = _daily_path_day(ob_path, "OB", week_key=week_key)
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
        self._queue_full_count = 0
        self._queue_full_last_log = 0.0

    def _put(self, item: Tuple[str, Optional[str], Optional[object]]):
        while True:
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                self._queue_full_count += 1
                if EVENT_QUEUE_FULL_LOG_SEC > 0:
                    now = time.monotonic()
                    if now - self._queue_full_last_log >= EVENT_QUEUE_FULL_LOG_SEC:
                        kind, wk, _payload = item
                        print(
                            f"[feeder] queue full blocked_count={self._queue_full_count} "
                            f"kind={kind!r} week={wk!r} qsize={self.queue.qsize()}",
                            flush=True,
                        )
                        self._queue_full_last_log = now

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


def collect_weekday_fixed_utc_pca_sample(
    train_pairs: List[WeekPair],
    *,
    rows_per_day: int = PCA_WEEKDAY_ROWS_PER_DAY,
    start_hour_utc: int = PCA_SAMPLE_START_UTC_HOUR,
    start_minute_utc: int = PCA_SAMPLE_START_UTC_MINUTE,
) -> np.ndarray:
    if not train_pairs:
        raise ValueError("No train pairs provided for PCA sampling")
    if rows_per_day < 1:
        raise ValueError(f"PCA weekday rows_per_day must be >=1, got {rows_per_day}")

    day_entries: List[Tuple[str, date, str, str]] = []
    for week_key, ob_paths, th_paths in train_pairs:
        if th_paths and len(th_paths) != len(ob_paths):
            raise ValueError(f"PCA sampling week {week_key}: OB/TH day count mismatch")
        for idx, ob_path in enumerate(ob_paths):
            day = _daily_path_day(ob_path, "OB", week_key=week_key)
            if day.weekday() > 4:
                continue
            th_path = th_paths[idx] if th_paths else ""
            if th_path:
                th_day = _daily_path_day(th_path, "TH", week_key=week_key)
                if th_day != day:
                    raise ValueError(f"PCA sampling week {week_key}: OB/TH day mismatch for {day.isoformat()}")
            day_entries.append((week_key, day, ob_path, th_path))

    if not day_entries:
        raise ValueError("No Monday-Friday train days available for PCA sampling")

    day_entries.sort(key=lambda x: (x[1], x[0], x[2]))
    unique_weekdays = len(day_entries)
    expected_rows = int(unique_weekdays * rows_per_day)
    print(
        f"[pca-sample] mode=weekday_fixed_utc_block train_weeks={len(train_pairs)} "
        f"weekdays={unique_weekdays} rows_per_day={rows_per_day} "
        f"start_utc={start_hour_utc:02d}:{start_minute_utc:02d}:00 total_rows={expected_rows}",
        flush=True,
    )

    fe = FeatureEngine()
    out_rows: List[np.ndarray] = []
    stream_started = time.monotonic()
    for week_key, day, ob_path, th_path in day_entries:
        start_dt = datetime.combine(
            day,
            datetime.min.time().replace(hour=start_hour_utc, minute=start_minute_utc),
            tzinfo=timezone.utc,
        )
        start_ts_ms = _dt_to_epoch_ms(start_dt)
        day_start_ms = _dt_to_epoch_ms(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end_ms = _dt_to_epoch_ms(datetime.combine(day + ONE_DAY, datetime.min.time(), tzinfo=timezone.utc))
        ob_iter = safe_ob_iter(ob_path, day_start_ms, day_end_ms, dq_day=None)
        if th_path:
            th_iter = safe_th_iter(th_path, day_start_ms, day_end_ms, dq_day=None)
            merged = merge_event_time(ob_iter, th_iter, dq_day=None, strict=bool(BYBIT_STRICT_DATA), B=0)
        else:
            merged = ob_iter

        day_rows: List[np.ndarray] = []
        first_ts: Optional[int] = None
        last_ts: Optional[int] = None
        for event in merged:
            ts_ms, feat_z, _mid, is_trade, _dt_ms = fe.on_fast_event(event)
            if is_trade:
                continue
            ts_i = int(ts_ms)
            if ts_i < start_ts_ms:
                continue
            if first_ts is None:
                first_ts = ts_i
            last_ts = ts_i
            day_rows.append(np.asarray(feat_z, dtype=np.float32))
            if len(day_rows) >= rows_per_day:
                break
        if len(day_rows) != rows_per_day:
            raise ValueError(
                f"PCA weekday sample for {day.isoformat()} produced {len(day_rows)} rows, expected {rows_per_day}. "
                "Do not silently fall back to first-row PCA sampling."
            )
        out_rows.extend(day_rows)
        print(
            f"[pca-sample-day] day={day.isoformat()} week={week_key} rows={len(day_rows)} "
            f"first_ts={first_ts} last_ts={last_ts}",
            flush=True,
        )

    sample_array = np.asarray(out_rows, dtype=np.float32)
    if int(sample_array.shape[0]) != expected_rows:
        raise ValueError(f"PCA weekday sample rows mismatch: got {sample_array.shape[0]} expected {expected_rows}")
    _print_coarse_timing_totals(
        "[pca-time]",
        {"wall_s": time.monotonic() - stream_started},
    )
    return sample_array


def _fit_pca_svd_from_sample(
    sample_rows: np.ndarray,
    *,
    select_mode: str,
    target_var: float,
    max_components: int,
) -> Dict[str, np.ndarray]:
    if sample_rows.ndim != 2 or sample_rows.size == 0:
        raise ValueError("PCA sample_rows must be a non-empty 2D array")

    n_samples, n_features = sample_rows.shape
    if n_samples < 2:
        raise ValueError(f"PCA requires at least 2 sample rows, got {n_samples}")
    if n_features <= 0:
        raise ValueError("PCA sample has zero features")

    x = np.asarray(sample_rows, dtype=np.float64)
    mean = np.mean(x, axis=0)
    centered = x - mean.reshape(1, -1)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)

    explained_variance = (s * s) / max(1, n_samples - 1)
    total_variance = float(np.sum(explained_variance))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("PCA sample has non-positive total variance")

    explained_ratio_all = explained_variance / total_variance
    cum = np.cumsum(explained_ratio_all)

    max_possible_components = min(n_samples, n_features)
    mode = str(select_mode).strip().lower()
    if mode != "max_components":
        raise ValueError("BYBIT_PCA_SELECT_MODE must be 'max_components' for the 1s maker contract")
    if int(n_features) < int(PCA_CORE_COMPONENTS):
        raise ValueError(
            f"Raw core feature dimension {int(n_features)} is smaller than required PCA_CORE_COMPONENTS={PCA_CORE_COMPONENTS}; "
            "cannot produce final feature_dim_total=512."
        )
    if int(max_components) != int(PCA_CORE_COMPONENTS):
        raise ValueError(f"PCA max_components must equal PCA_CORE_COMPONENTS={PCA_CORE_COMPONENTS}")
    if int(PCA_CORE_COMPONENTS) > int(max_possible_components):
        raise ValueError(
            f"PCA_CORE_COMPONENTS={PCA_CORE_COMPONENTS} exceeds max_possible_components={max_possible_components}"
        )
    k = int(PCA_CORE_COMPONENTS)
    retained_var = float(cum[k - 1])
    print(
        f"[pca-select] mode=max_components required_components={PCA_CORE_COMPONENTS} k={k} retained_var={retained_var:.6f}",
        flush=True,
    )

    if int(vt[:k].shape[0]) != int(PCA_CORE_COMPONENTS):
        raise ValueError("PCA component row count mismatch after SVD")
    if int(mean.shape[0]) != int(n_features):
        raise ValueError("PCA mean dimension mismatch after SVD")

    return {
        "mean": mean.astype(np.float32, copy=False),
        "components": vt[:k].astype(np.float32, copy=False),
        "explained_variance_ratio": explained_ratio_all[:k].astype(np.float32, copy=False),
        "k": np.array(k, dtype=np.int64),
        "total_explained_variance_ratio": np.array(retained_var, dtype=np.float32),
        "max_possible_components": np.array(max_possible_components, dtype=np.int64),
    }


def maybe_fit_pca_model(
    pairs: List[WeekPair],
    out_root: str,
    train_weeks: List[str],
    target_var: float,
    select_mode: str,
    max_components: int,
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
    feature_names_pre_pca = list(FeatureEngine().feature_names())
    names_hash = hashlib.sha256(json.dumps(feature_names_pre_pca).encode()).hexdigest()[:12]
    print(
        f"[pca-feature-schema] schema={FEATURE_SCHEMA} raw_dim={len(feature_names_pre_pca)} names_hash={names_hash}",
        flush=True,
    )
    meta = {
        "applied": False,
        "var_kept": float(target_var),
        "pca_select_mode": str(select_mode),
        "pca_max_components": int(max_components),
        "k": 0,
        "model_path": None,
        "feature_names_pre_pca": feature_names_pre_pca,
        "feature_dim_core_pre_pca": int(len(feature_names_pre_pca)),
        "feature_names_hash": names_hash,
    }
    if target_var <= 0.0:
        raise ValueError(f"PCA is required for FEATURE_SCHEMA={FEATURE_SCHEMA}")

    if int(use_existing) == 1:
        model_path = os.path.join(out_root, model_filename)
        try:
            with np.load(model_path) as data:
                components = data["components"]
                k = int(components.shape[0]) if components.size else 0
                if k <= 0:
                    raise ValueError("PCA model has no components")
                if "metadata_json" not in data:
                    raise ValueError("PCA model missing metadata_json")
                model_meta = json.loads(str(data["metadata_json"].item()))
                if model_meta.get("feature_schema") != FEATURE_SCHEMA:
                    raise ValueError("PCA feature schema mismatch")
                if model_meta.get("aux_schema") != AUX_SCHEMA:
                    raise ValueError("PCA aux schema mismatch")
                if int(model_meta.get("feature_dim_core_pre_pca", -1)) != len(feature_names_pre_pca):
                    raise ValueError("PCA pre-PCA feature dim mismatch")
                if list(model_meta.get("feature_names_pre_pca", [])) != feature_names_pre_pca:
                    raise ValueError("PCA feature names mismatch")
                if str(model_meta.get("feature_names_hash", "")) != names_hash:
                    raise ValueError("PCA feature names hash mismatch")
                if str(model_meta.get("pca_select_mode", "")).strip().lower() != str(select_mode).strip().lower():
                    raise ValueError("PCA select mode mismatch")
                if str(model_meta.get("pca_sample_mode", "")) != "weekday_fixed_utc_block":
                    raise ValueError("PCA sample mode mismatch")
                if int(model_meta.get("aux_dim", -1)) != int(AUX_DIM):
                    raise ValueError("PCA aux_dim mismatch")
                if int(model_meta.get("pca_k", -1)) != int(PCA_CORE_COMPONENTS):
                    raise ValueError("PCA pca_k mismatch")
                if int(model_meta.get("final_feature_dim", -1)) != int(FINAL_FEATURE_DIM):
                    raise ValueError("PCA final_feature_dim mismatch")
                if int(model_meta.get("pca_core_components_required", -1)) != int(PCA_CORE_COMPONENTS):
                    raise ValueError("PCA required core component mismatch")
                if int(k) != int(PCA_CORE_COMPONENTS):
                    raise ValueError("PCA component count mismatch")
        except Exception as exc:
            raise ValueError(f"Failed to load required PCA model '{model_path}': {exc}") from exc

        meta.update({
            "applied": True,
            "k": k,
            "pca_k": k,
            "final_feature_dim": int(FINAL_FEATURE_DIM),
            "pca_core_components_required": int(PCA_CORE_COMPONENTS),
            "model_path": model_filename,
            "pca_sample_mode": "weekday_fixed_utc_block",
        })
        print(f"[pca  ] Reusing existing PCA model '{model_path}' (k={k})")
        return meta

    train_set = set(train_weeks)
    train_pairs = [p for p in pairs if p[0] in train_set]
    if not train_pairs:
        raise ValueError(f"PCA is required for FEATURE_SCHEMA={FEATURE_SCHEMA}")

    _ = int(batch_size)
    sample_array = collect_weekday_fixed_utc_pca_sample(train_pairs)
    print(
        f"[pca-fit] method=deterministic_weekday_fixed_utc rows={sample_array.shape[0]} raw_dim={sample_array.shape[1]}",
        flush=True,
    )
    pca_fit = _fit_pca_svd_from_sample(
        sample_array,
        select_mode=select_mode,
        target_var=target_var,
        max_components=max_components,
    )
    mean = pca_fit["mean"]
    components = pca_fit["components"]
    explained_variance_ratio = pca_fit["explained_variance_ratio"]
    k = int(pca_fit["k"])
    total_kept = float(pca_fit["total_explained_variance_ratio"])
    print(f"[pca-init] n_components={k} sample_rows={sample_array.shape[0]} kept_var={total_kept:.6f}", flush=True)

    model_path = os.path.join(out_root, model_filename)
    ensure_dir(os.path.dirname(model_path))
    np.savez(
        model_path,
        mean=mean,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        metadata_json=np.array(
            json.dumps(
                {
                    "feature_schema": FEATURE_SCHEMA,
                    "aux_schema": AUX_SCHEMA,
                    "aux_dim": int(AUX_DIM),
                    "feature_dim_core_pre_pca": len(feature_names_pre_pca),
                    "feature_names_pre_pca": feature_names_pre_pca,
                    "feature_names_hash": names_hash,
                    "created_by": "offline_ingest.py",
                    "stage": "stage4_v6_fast_trade_obnorm",
                    "pca_fit_method": "deterministic_weekday_fixed_utc_svd",
                    "pca_select_mode": str(select_mode),
                    "pca_max_components": int(max_components),
                    "k": int(k),
                    "pca_k": int(k),
                    "final_feature_dim": int(FINAL_FEATURE_DIM),
                    "pca_core_components_required": int(PCA_CORE_COMPONENTS),
                    "pca_sample_mode": "weekday_fixed_utc_block",
                    "pca_sample_start_utc": f"{PCA_SAMPLE_START_UTC_HOUR:02d}:{PCA_SAMPLE_START_UTC_MINUTE:02d}:00",
                    "pca_weekday_rows_per_day": int(PCA_WEEKDAY_ROWS_PER_DAY),
                    "pca_sample_weekdays": int(sample_array.shape[0] // PCA_WEEKDAY_ROWS_PER_DAY),
                    "pca_sample_rows": int(sample_array.shape[0]),
                    "pca_target_var": float(target_var),
                    "pca_total_explained_variance_ratio": float(total_kept),
                },
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
    )

    meta.update(
        {
            "applied": True,
            "k": int(k),
            "pca_k": int(k),
            "final_feature_dim": int(FINAL_FEATURE_DIM),
            "pca_core_components_required": int(PCA_CORE_COMPONENTS),
            "model_path": model_filename,
            "rows_fitted": int(sample_array.shape[0]),
            "sample_rows": int(sample_array.shape[0]),
            "fit_method": "deterministic_weekday_fixed_utc_svd",
            "pca_select_mode": str(select_mode),
            "pca_max_components": int(max_components),
            "pca_target_var": float(target_var),
            "pca_total_explained_variance_ratio": float(total_kept),
            "total_explained_variance_ratio": float(total_kept),
            "pca_sample_mode": "weekday_fixed_utc_block",
            "pca_sample_start_utc": f"{PCA_SAMPLE_START_UTC_HOUR:02d}:{PCA_SAMPLE_START_UTC_MINUTE:02d}:00",
            "pca_weekday_rows_per_day": int(PCA_WEEKDAY_ROWS_PER_DAY),
            "pca_sample_weekdays": int(sample_array.shape[0] // PCA_WEEKDAY_ROWS_PER_DAY),
        }
    )

    print(
        f"[pca  ] applied target={target_var:.4f} k={meta['k']} "
        f"sample={meta.get('sample_rows', 0)} fitted={meta.get('rows_fitted', 0)} "
        f"method={meta.get('fit_method')}"
    )

    return meta


def _summarise_pca_meta(meta: Optional[dict]) -> dict:
    base = {
        "applied": False,
        "var_kept": float(PCA_VAR_TARGET),
        "pca_select_mode": str(PCA_SELECT_MODE),
        "pca_max_components": int(PCA_MAX_COMPONENTS),
        "k": 0,
        "pca_k": 0,
        "final_feature_dim": int(FINAL_FEATURE_DIM),
        "pca_core_components_required": int(PCA_CORE_COMPONENTS),
        "model_path": None,
        "feature_names_pre_pca": [],
        "feature_dim_core_pre_pca": 0,
        "feature_names_hash": "",
        "pca_sample_mode": "weekday_fixed_utc_block",
        "pca_sample_start_utc": f"{PCA_SAMPLE_START_UTC_HOUR:02d}:{PCA_SAMPLE_START_UTC_MINUTE:02d}:00",
        "pca_weekday_rows_per_day": int(PCA_WEEKDAY_ROWS_PER_DAY),
    }
    if not meta:
        return base
    applied = bool(meta.get("applied", False))
    base.update(
        {
            "applied": applied,
            "var_kept": float(meta.get("var_kept", base["var_kept"])),
            "pca_select_mode": str(meta.get("pca_select_mode", base["pca_select_mode"])),
            "pca_max_components": int(meta.get("pca_max_components", base["pca_max_components"])),
            "k": int(meta.get("k", 0) if applied else 0),
            "pca_k": int(meta.get("k", 0) if applied else 0),
            "final_feature_dim": int(meta.get("final_feature_dim", FINAL_FEATURE_DIM)),
            "pca_core_components_required": int(meta.get("pca_core_components_required", PCA_CORE_COMPONENTS)),
            "model_path": meta.get("model_path") if applied else None,
            "feature_names_pre_pca": list(meta.get("feature_names_pre_pca", [])),
            "feature_dim_core_pre_pca": int(meta.get("feature_dim_core_pre_pca", 0)),
            "feature_names_hash": str(meta.get("feature_names_hash", "")),
            "pca_sample_mode": str(meta.get("pca_sample_mode", base["pca_sample_mode"])),
            "pca_sample_start_utc": str(meta.get("pca_sample_start_utc", base["pca_sample_start_utc"])),
            "pca_weekday_rows_per_day": int(meta.get("pca_weekday_rows_per_day", base["pca_weekday_rows_per_day"])),
            "pca_sample_rows": int(meta.get("sample_rows", meta.get("pca_sample_rows", 0))),
            "pca_sample_weekdays": int(meta.get("pca_sample_weekdays", 0)),
            "pca_target_var": float(meta.get("pca_target_var", PCA_VAR_TARGET)),
            "pca_total_explained_variance_ratio": float(meta.get("pca_total_explained_variance_ratio", meta.get("total_explained_variance_ratio", 0.0))),
        }
    )
    return base


def _resolve_chunk_path(out_root: Path, week_key: str, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    cand1 = out_root / p
    if cand1.exists():
        return cand1
    return out_root / week_key / p


def _iter_chunk_file_refs(chunks: list, *, kind: str) -> List[str]:
    out: List[str] = []
    if kind == "feature":
        required = ("features", "ts")
    elif kind == "label":
        required = ("row_idx", "label_ts", "y")
    else:
        raise ValueError(f"Unsupported chunk kind: {kind}")

    for idx, entry in enumerate(chunks):
        if isinstance(entry, str):
            if not entry:
                raise ValueError(f"Empty {kind} chunk path at index {idx}")
            out.append(entry)
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid {kind} chunk entry at index {idx}: expected dict, got {type(entry).__name__}"
            )
        files = entry.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"Invalid {kind} chunk entry at index {idx}: missing files dict")
        for key in required:
            path = files.get(key)
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid {kind} chunk entry at index {idx}: missing files.{key}")
            out.append(path)
    return out


def _assert_append_build_dir_safe(out_root: Path, week_key: str) -> None:
    week_dir = out_root / week_key
    meta_path = week_dir / "meta_week.json"
    if not week_dir.exists():
        return
    if meta_path.exists():
        raise ValueError(
            f"Refusing to build week {week_key}: week directory already has meta_week.json. "
            "This should have been reused or rejected, not rebuilt."
        )
    entries = [p for p in week_dir.iterdir() if p.name not in {".DS_Store"}]
    if entries:
        sample = ", ".join(p.name for p in entries[:10])
        raise ValueError(
            f"Refusing to build week {week_key}: directory exists but has no meta_week.json and is not empty. "
            f"This may be a partial previous run. Move/delete it manually before append. Entries: {sample}"
        )


def load_reusable_week_meta(
    out_root: str,
    week_key: str,
    *,
    expected: dict,
) -> Optional[dict]:
    root = Path(out_root)
    week_meta_path = root / week_key / "meta_week.json"
    if not week_meta_path.exists():
        return None

    try:
        meta = json.loads(week_meta_path.read_text())
    except Exception as exc:
        raise ValueError(f"[reuse] week={week_key} invalid meta_week.json: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"[reuse] week={week_key} meta_week.json must decode to dict")

    expected_meta_path = os.path.join(week_key, "meta_week.json")
    observed_meta_path = meta.get("meta_path")
    if observed_meta_path is not None and observed_meta_path != expected_meta_path:
        raise ValueError(
            f"[reuse] week={week_key} incompatible meta_path={observed_meta_path!r} expected={expected_meta_path!r}"
        )

    feature_chunks = meta.get("feature_chunks")
    label_chunks = meta.get("label_chunks")
    if not isinstance(feature_chunks, list) or not feature_chunks:
        raise ValueError(f"[reuse] week={week_key} missing non-empty feature_chunks")
    if not isinstance(label_chunks, list) or not label_chunks:
        raise ValueError(f"[reuse] week={week_key} missing non-empty label_chunks")

    for rel_path in _iter_chunk_file_refs(feature_chunks, kind="feature"):
        if not _resolve_chunk_path(root, week_key, rel_path).exists():
            raise ValueError(f"[reuse] week={week_key} missing feature chunk file: {rel_path}")
    for rel_path in _iter_chunk_file_refs(label_chunks, kind="label"):
        if not _resolve_chunk_path(root, week_key, rel_path).exists():
            raise ValueError(f"[reuse] week={week_key} missing label chunk file: {rel_path}")

    if int(meta.get("rows_total", 0)) <= 0:
        raise ValueError(f"[reuse] week={week_key} rows_total must be > 0")
    if int(meta.get("labels_total", 0)) <= 0:
        raise ValueError(f"[reuse] week={week_key} labels_total must be > 0")

    decision_range = meta.get("decision_ts_range")
    if not isinstance(decision_range, dict):
        raise ValueError(f"[reuse] week={week_key} missing decision_ts_range")
    if "min" not in decision_range or "max" not in decision_range:
        raise ValueError(f"[reuse] week={week_key} decision_ts_range must include min/max")
    min_ts = int(decision_range["min"])
    max_ts = int(decision_range["max"])
    if min_ts > max_ts:
        raise ValueError(f"[reuse] week={week_key} invalid decision_ts_range min={min_ts} max={max_ts}")

    for key in (
        "feature_schema",
        "aux_schema",
        "target_task",
        "target_transform",
        "checkpoint_schema_expected",
        "lookback",
        "window_ms",
        "label_dim",
        "aux_dim",
        "feature_dim_core",
        "feature_dim_total",
        "feature_names_hash",
        "trade_history_enabled",
        "event_stream_mode",
    ):
        if meta.get(key) != expected.get(key):
            raise ValueError(
                f"[reuse] week={week_key} incompatible field {key}: {meta.get(key)!r} != {expected.get(key)!r}"
            )
    if int(meta.get("feature_dim_core", -1)) != int(PCA_CORE_COMPONENTS):
        raise ValueError(f"[reuse] week={week_key} feature_dim_core mismatch")
    if int(meta.get("feature_dim_total", -1)) != int(FINAL_FEATURE_DIM):
        raise ValueError(f"[reuse] week={week_key} feature_dim_total mismatch")
    if int(meta.get("feature_dim_total", -1)) != int(meta.get("feature_dim_core", -1)) + AUX_DIM:
        raise ValueError(f"[reuse] week={week_key} feature_dim_total must equal feature_dim_core + AUX_DIM")

    pca = meta.get("pca")
    if not isinstance(pca, dict) or not bool(pca.get("applied", False)):
        raise ValueError(f"[reuse] week={week_key} requires pca.applied=true")
    if int(pca.get("final_feature_dim", -1)) != int(FINAL_FEATURE_DIM):
        raise ValueError(f"[reuse] week={week_key} PCA final_feature_dim mismatch")
    if int(pca.get("pca_core_components_required", -1)) != int(PCA_CORE_COMPONENTS):
        raise ValueError(f"[reuse] week={week_key} PCA required component mismatch")
    if int(pca.get("k", -1)) != int(expected["pca_k"]):
        raise ValueError(
            f"[reuse] week={week_key} PCA k mismatch: {int(pca.get('k', -1))} != {int(expected['pca_k'])}"
        )
    if str(pca.get("pca_select_mode", "")).strip().lower() != str(expected.get("pca_select_mode", "")).strip().lower():
        raise ValueError(f"[reuse] week={week_key} PCA select_mode mismatch")
    if str(pca.get("pca_sample_mode", "")) != "weekday_fixed_utc_block":
        raise ValueError(f"[reuse] week={week_key} PCA sample_mode must be weekday_fixed_utc_block")
    if str(meta.get("feature_names_hash", "")) != str(expected.get("feature_names_hash", "")):
        raise ValueError(f"[reuse] week={week_key} feature_names_hash mismatch")
    if int(meta.get("aux_dim", -1)) != int(AUX_DIM):
        raise ValueError(f"[reuse] week={week_key} aux_dim mismatch")
    print(f"[reuse] week={week_key} rows={int(meta.get('rows_total', 0))} labels={int(meta.get('labels_total', 0))}", flush=True)
    return meta


def build_global_meta_from_week_metas(
    *,
    pairs: List[WeekPair],
    week_metas: Dict[str, dict],
    pca_summary: dict,
    pca_var_ratio: Optional[np.ndarray],
    protocol: str,
    router: Optional[FlatWeekRouter],
    week_quality_records: Dict[str, dict],
) -> dict:
    ds_start, ds_end = _compute_dataset_span(pairs)
    start_iso = ds_start.date().isoformat() if ds_start else None
    end_iso = ds_end.date().isoformat() if ds_end else None
    weeks_in_order = [wk for wk, _ob, _th in pairs]
    week_row_counts = {wk: int(week_metas[wk].get("rows_total", 0)) for wk in weeks_in_order}
    week_label_counts = {wk: int(week_metas[wk].get("labels_total", 0)) for wk in weeks_in_order}
    total_feature_rows = int(sum(week_row_counts.values()))
    total_labels = int(sum(week_label_counts.values()))
    feature_dim_core = int(PCA_CORE_COMPONENTS)
    feature_dim_total = int(FINAL_FEATURE_DIM)
    label_dim = int(NUM_HORIZONS)
    weeks_meta_paths = {wk: week_metas[wk].get("meta_path", os.path.join(wk, "meta_week.json")) for wk in weeks_in_order}

    meta = {
        "dataset_start": start_iso,
        "dataset_end": end_iso,
        "weeks_in_order": weeks_in_order,
        "decision_policy": DECISION_POLICY,
        "decision_time_basis": "ob_event_time",
        "window_ms": int(WINDOW_MS),
        "decision_stride_policy": "every_ob_event",
        "label_delta_ms": 0,
        "label_units": "signed_log_return_bps",
        "feature_schema": FEATURE_SCHEMA,
        "aux_schema": AUX_SCHEMA,
        "target_task": TARGET_TASK,
        "target_transform": TARGET_TRANSFORM,
        "low_abs_trim_fraction": float(LOW_ABS_TRIM_FRACTION),
        "high_abs_trim_fraction": float(HIGH_ABS_TRIM_FRACTION),
        "checkpoint_schema_expected": CHECKPOINT_SCHEMA,
        **canonical_mode_fields(),
        "storage_format": "flat_decision_rows_v1",
        "lookback": int(LOOKBACK),
        "feature_dim_total": feature_dim_total,
        "feature_dim_core": feature_dim_core,
        "feature_dim_core_pre_pca": int(pca_summary["feature_dim_core_pre_pca"]),
        "feature_names_pre_pca": list(pca_summary["feature_names_pre_pca"]),
        "feature_names_hash": str(pca_summary["feature_names_hash"]),
        "aux_dim": int(AUX_DIM),
        "aux_names": list(FEATURE_AUX_TAIL),
        "dtype": "float32",
        "ram_budget_mb": int(RAM_BUDGET),
        "chunk_size_used": 0 if (router is None or router.chunk_size_used == 0) else int(router.chunk_size_used),
        "label_dim": label_dim,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "total_feature_rows": total_feature_rows,
        "total_labels": total_labels,
        "week_row_counts": week_row_counts,
        "week_label_counts": week_label_counts,
        "weeks_meta": weeks_meta_paths,
        "data_quality_path": "_data_quality.json",
    }
    meta["pca"] = dict(pca_summary)
    if pca_var_ratio is not None:
        meta["pca"]["explained_variance_ratio"] = [float(x) for x in pca_var_ratio]
    meta["splits"] = build_pipeline_splits(weeks_in_order, week_metas, protocol)
    return meta


def write_json_atomic_with_backup(path: Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(str(path.parent))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    backup = path.with_name(f"{path.name}.bak.{ts}")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        backup.write_bytes(path.read_bytes())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def _mature_with_next_week_context(
    *,
    pair: WeekPair,
    week_key_to_mature: str,
    week_decision_ts_max: int,
    fe: FeatureEngine,
    labeler: LabelBuilder,
    pending_decisions: deque[Tuple[str, int, int]],
    router: FlatWeekRouter,
) -> Tuple[int, Optional[int], int]:
    cutoff_ts = int(week_decision_ts_max + GRACE_MS + 1)
    context_events = 0
    context_last_ts: Optional[int] = None
    matured_labels = 0
    for event in _iter_week_merged_events(pair[0], pair[1], pair[2], week_quality=None):
        ts_ms, feat_z, mid, is_trade, _dt_ms = fe.on_fast_event(event)
        if is_trade:
            continue
        if int(ts_ms) > cutoff_ts:
            break
        context_events += 1
        context_last_ts = int(ts_ms)
        matured = labeler.on_event(int(ts_ms), float(mid))
        if matured is None:
            raise RuntimeError("Matured labels were not produced for OB context event")
        for yy in matured:
            if not pending_decisions:
                raise RuntimeError("Context maturation produced labels but pending queue was empty")
            lbl_week, lbl_row_idx, lbl_ts = pending_decisions.popleft()
            router.add_label(lbl_week, lbl_row_idx, lbl_ts, yy.astype(np.float32, copy=False))
            matured_labels += 1
        if not pending_decisions:
            break
        if pending_decisions[0][0] != week_key_to_mature:
            break
    return context_events, context_last_ts, matured_labels


def process_all(
    pairs: List[WeekPair],
    out_root: str,
    pca_meta: dict,
    *,
    protocol: str,
    append_missing_weeks: bool = False,
):
    """Run ingest across week pairs with ordered daily OB paths and mode-dependent TH paths (which may be empty in OB-only mode)."""
    ensure_dir(out_root)

    pca_summary = _summarise_pca_meta(pca_meta)
    if not pca_summary["applied"]:
        raise ValueError(f"PCA is required for FEATURE_SCHEMA={FEATURE_SCHEMA}")
    feature_names_pre_pca = list(FeatureEngine().feature_names())
    names_hash = hashlib.sha256(json.dumps(feature_names_pre_pca).encode()).hexdigest()[:12]
    pca_mean: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_var_ratio: Optional[np.ndarray] = None

    model_path = pca_summary.get("model_path")
    full_model_path = os.path.join(out_root, model_path) if model_path else ""
    try:
        with np.load(full_model_path) as data:
            pca_mean = data["mean"].astype(np.float32)
            pca_components = data["components"].astype(np.float32)
            if "explained_variance_ratio" in data:
                pca_var_ratio = data["explained_variance_ratio"].astype(np.float32)
            if "metadata_json" not in data:
                raise ValueError("missing metadata_json")
            model_meta = json.loads(str(data["metadata_json"].item()))
    except Exception as exc:
        raise ValueError(f"Failed to load required PCA model '{full_model_path}': {exc}") from exc

    if model_meta.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("PCA feature_schema mismatch")
    if model_meta.get("aux_schema") != AUX_SCHEMA:
        raise ValueError("PCA aux_schema mismatch")
    if int(model_meta.get("feature_dim_core_pre_pca", -1)) != len(feature_names_pre_pca):
        raise ValueError("PCA feature_dim_core_pre_pca mismatch")
    if list(model_meta.get("feature_names_pre_pca", [])) != feature_names_pre_pca:
        raise ValueError("PCA feature_names_pre_pca mismatch")
    if str(model_meta.get("feature_names_hash", "")) != names_hash:
        raise ValueError("PCA feature_names_hash mismatch")
    if str(model_meta.get("pca_select_mode", "")).strip().lower() != str(PCA_SELECT_MODE).strip().lower():
        raise ValueError("PCA pca_select_mode mismatch")
    if str(model_meta.get("pca_sample_mode", "")) != "weekday_fixed_utc_block":
        raise ValueError("PCA pca_sample_mode mismatch")
    if int(model_meta.get("aux_dim", -1)) != int(AUX_DIM):
        raise ValueError("PCA aux_dim mismatch")
    if int(model_meta.get("pca_k", -1)) != int(PCA_CORE_COMPONENTS):
        raise ValueError("PCA pca_k mismatch")
    if int(model_meta.get("final_feature_dim", -1)) != int(FINAL_FEATURE_DIM):
        raise ValueError("PCA final_feature_dim mismatch")
    if int(model_meta.get("pca_core_components_required", -1)) != int(PCA_CORE_COMPONENTS):
        raise ValueError("PCA required core component mismatch")
    if int(pca_components.shape[0]) != int(PCA_CORE_COMPONENTS):
        raise ValueError(
            f"PCA components mismatch: got {int(pca_components.shape[0])} expected {int(PCA_CORE_COMPONENTS)}"
        )
    pca_summary["feature_names_pre_pca"] = feature_names_pre_pca
    pca_summary["feature_dim_core_pre_pca"] = len(feature_names_pre_pca)
    pca_summary["feature_names_hash"] = names_hash
    for req in ("feature_names_pre_pca", "feature_dim_core_pre_pca", "feature_names_hash"):
        if req not in pca_summary or not pca_summary[req]:
            raise ValueError(f"Missing required PCA metadata field: {req}")
    if pca_mean is None or pca_components is None:
        raise ValueError("Missing PCA tensors")
    pre_pca_dim = int(pca_mean.shape[0])
    pca_k = int(pca_components.shape[0])
    if int(pca_mean.shape[0]) != int(len(feature_names_pre_pca)):
        raise ValueError("PCA mean raw core dimension mismatch")
    if int(pca_components.shape[1]) != int(pca_mean.shape[0]):
        raise ValueError("PCA components raw core dimension mismatch")
    final_feature_dim = int(pca_k + AUX_DIM)
    if int(pca_k) != int(PCA_CORE_COMPONENTS):
        raise ValueError(f"PCA core dimension mismatch: got {pca_k}, expected {PCA_CORE_COMPONENTS}")
    if int(final_feature_dim) != int(FINAL_FEATURE_DIM):
        raise ValueError(f"Expected final_feature_dim={FINAL_FEATURE_DIM}, got {final_feature_dim}")
    weeks_in_order = [wk for wk, _ob, _th in pairs]
    if append_missing_weeks and protocol != FIVE_WEEK_PROTOCOL:
        raise ValueError("append_missing_weeks=True requires five-week protocol")
    expected_reuse_fields = {
        "feature_schema": FEATURE_SCHEMA,
        "aux_schema": AUX_SCHEMA,
        "target_task": TARGET_TASK,
        "target_transform": TARGET_TRANSFORM,
        "checkpoint_schema_expected": CHECKPOINT_SCHEMA,
        "lookback": int(LOOKBACK),
        "window_ms": int(WINDOW_MS),
        "label_dim": int(NUM_HORIZONS),
        "aux_dim": int(AUX_DIM),
        "feature_dim_core": int(PCA_CORE_COMPONENTS),
        "feature_dim_total": int(FINAL_FEATURE_DIM),
        "feature_names_hash": str(names_hash),
        "trade_history_enabled": canonical_mode_fields()["trade_history_enabled"],
        "event_stream_mode": canonical_mode_fields()["event_stream_mode"],
        "pca_k": int(pca_summary["k"]),
        "pca_select_mode": str(pca_summary.get("pca_select_mode", PCA_SELECT_MODE)),
    }
    week_meta_records: Dict[str, dict] = {}
    pairs_to_build: List[WeekPair] = []
    reused_week_metas: Dict[str, dict] = {}
    for pair in pairs:
        wk = pair[0]
        if append_missing_weeks:
            reusable = load_reusable_week_meta(out_root, wk, expected=expected_reuse_fields)
            if reusable is not None:
                week_meta_records[wk] = reusable
                reused_week_metas[wk] = reusable
                continue
            _assert_append_build_dir_safe(Path(out_root), wk)
        pairs_to_build.append(pair)
    build_week_keys = {wk for wk, _ob, _th in pairs_to_build}

    fe = FeatureEngine()
    labeler = LabelBuilder(delta_ms=0, horizons_ms=HORIZONS_MS)

    pending_decisions: deque[Tuple[str, int, int]] = deque()
    last_decision_ts_ms: Optional[int] = None
    built_week_last_decision_ts: Dict[str, int] = {}

    F = final_feature_dim
    router: Optional[FlatWeekRouter] = None
    total_feature_rows = 0
    total_labels = 0
    week_index = _build_week_index(pairs_to_build)

    print(f"[append] requested_weeks={len(pairs)} reused={len(reused_week_metas)} build={len(pairs_to_build)}", flush=True)
    for wk in reused_week_metas:
        print(f"[reuse] week={wk}", flush=True)
    for wk, _ob, _th in pairs_to_build:
        print(f"[build] week={wk}", flush=True)
    print(
        f"[start] ingest weeks={len(pairs)} build={len(pairs_to_build)} reuse={len(week_meta_records)} "
        f"L={LOOKBACK} budget={RAM_BUDGET}MB"
    )
    last_log = time.monotonic()
    next_log = last_log + 300.0
    ingest_started = time.monotonic()
    events_seen = 0

    feeder = EventFeeder(pairs_to_build)
    producer_thread = threading.Thread(target=feeder.run, daemon=True)
    producer_thread.start()
    q = feeder.queue

    week_total = len(pairs_to_build)
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

        ts_ms, feat_z, mid, is_trade, dt_ms = fe.on_fast_event(event)
        events_seen += 1
        if is_trade and np.asarray(feat_z).shape[0] != 0:
            raise RuntimeError("Trade fast path returned a non-empty feature vector")

        if not is_trade:
            core_pre_pca = np.asarray(feat_z, dtype=np.float32, copy=False)
            assert len(feature_names_pre_pca) == core_pre_pca.shape[0]
            if core_pre_pca.shape[-1] != pre_pca_dim:
                raise ValueError(
                    f"PCA pre-core dimension {pre_pca_dim} does not match feature dimension {core_pre_pca.shape[-1]}"
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
            aux_tail = build_aux_tail(fe, dt_tick)

            if router is None:
                print(f"[first-token] feature_dim_core={int(pca_k)} aux_dim={AUX_DIM} feature_dim_total={int(F)}", flush=True)
                router = FlatWeekRouter(
                    out_root,
                    F,
                    pre_pca_dim,
                    pca_mean,
                    pca_components,
                    AUX_DIM,
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
                week_key, row_idx = router.overwrite_latest_feature_row(int(ts_ms), core_pre_pca, aux_tail)
                if week_key != prev_week_key:
                    raise RuntimeError("Duplicate timestamp mapped to a different week during overwrite")
                pending_decisions[-1] = (week_key, row_idx, int(ts_ms))
            else:
                week_key, row_idx = router.append_feature_row(int(ts_ms), core_pre_pca, aux_tail)
                pending_decisions.append((week_key, row_idx, int(ts_ms)))
                labeler.on_decision(int(ts_ms))
                total_feature_rows += 1
            built_week_last_decision_ts[week_key] = int(ts_ms)

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

            if router is not None:
                router.close_old_writers(int(ts_ms))

        now = time.monotonic()
        if now >= next_log:
            elapsed = max(1e-9, now - ingest_started)
            print(
                f"[tok  ] rows={total_feature_rows} labels={total_labels} weeks={week_counter}/{week_total} "
                f"chunkN={router.chunk_size_used if router else 0} rows_per_sec={total_feature_rows/elapsed:.2f} "
                f"queue_full={feeder._queue_full_count}",
                flush=True,
            )
            last_log = now
            next_log = now + 300.0

    producer_thread.join()

    if append_missing_weeks and router is not None:
        for idx, pair in enumerate(pairs):
            wk = pair[0]
            if wk not in build_week_keys:
                continue
            next_pair = pairs[idx + 1] if (idx + 1) < len(pairs) else None
            if next_pair is None or next_pair[0] not in reused_week_metas:
                continue
            if wk not in built_week_last_decision_ts:
                continue
            if not pending_decisions:
                continue
            if pending_decisions[0][0] != wk:
                continue
            print(
                f"[append-context] week={wk} next={next_pair[0]} mode=label_maturation_only max_ms={GRACE_MS}",
                flush=True,
            )
            context_events, context_last_ts, matured_labels = _mature_with_next_week_context(
                pair=next_pair,
                week_key_to_mature=wk,
                week_decision_ts_max=int(built_week_last_decision_ts[wk]),
                fe=fe,
                labeler=labeler,
                pending_decisions=pending_decisions,
                router=router,
            )
            total_labels += int(matured_labels)
            print(
                f"[append-context] done week={wk} context_events={context_events} "
                f"context_last_ts={context_last_ts} matured_labels={matured_labels}",
                flush=True,
            )

    if router is not None:
        router.flush_all()

    feature_dim_total = int(F)
    feature_dim_core = int(PCA_CORE_COMPONENTS)
    label_dim = int(NUM_HORIZONS)
    if router is not None:
        week_meta_records.update(dict(router.week_metas))
    week_quality_records = dict(feeder.quality_by_week)
    if set(week_meta_records.keys()) != set(weeks_in_order):
        missing = [wk for wk in weeks_in_order if wk not in week_meta_records]
        raise ValueError(f"Missing week metadata for requested weeks: {missing}")
    built_feature_rows_from_weeks = sum(
        int(week_meta_records[wk].get("rows_total", 0)) for wk, _ob, _th in pairs_to_build if wk in week_meta_records
    )
    built_labels_from_weeks = sum(
        int(week_meta_records[wk].get("labels_total", 0)) for wk, _ob, _th in pairs_to_build if wk in week_meta_records
    )
    if int(total_feature_rows) != int(built_feature_rows_from_weeks):
        raise ValueError(
            "Inconsistent built totals: total_feature_rows "
            f"{int(total_feature_rows)} != built weeks_meta.rows_total {int(built_feature_rows_from_weeks)}"
        )
    if int(total_labels) != int(built_labels_from_weeks):
        raise ValueError(
            "Inconsistent built totals: total_labels "
            f"{int(total_labels)} != built weeks_meta.labels_total {int(built_labels_from_weeks)}"
        )
    weeks_meta_paths = {wk: week_meta_records[wk].get("meta_path", os.path.join(wk, "meta_week.json")) for wk in weeks_in_order}

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
        week_quality_path = os.path.join(out_root, wk, "data_quality.json")
        if wk in week_quality_records:
            with open(week_quality_path, "w") as f:
                json.dump(week_quality_records[wk], f, indent=2)
            continue
        if append_missing_weeks and os.path.exists(week_quality_path):
            try:
                data_quality_dataset["weeks"][wk] = json.loads(Path(week_quality_path).read_text())
            except Exception:
                pass

    with open(os.path.join(out_root, "_data_quality.json"), "w") as f:
        json.dump(data_quality_dataset, f, indent=2)
    meta = build_global_meta_from_week_metas(
        pairs=pairs,
        week_metas=week_meta_records,
        pca_summary=pca_summary,
        pca_var_ratio=pca_var_ratio,
        protocol=protocol,
        router=router,
        week_quality_records=week_quality_records,
    )
    if week_meta_records:
        expected_mode = canonical_mode_fields()
        for wk in weeks_in_order:
            week_meta = week_meta_records.get(wk)
            if not week_meta:
                continue
            for field, expected in (
                ("feature_schema", FEATURE_SCHEMA),
                ("aux_schema", AUX_SCHEMA),
                ("feature_dim_core", feature_dim_core),
                ("feature_dim_total", feature_dim_total),
                ("aux_names", list(FEATURE_AUX_TAIL)),
                ("feature_names_hash", str(pca_summary["feature_names_hash"])),
            ):
                if week_meta.get(field) != expected:
                    raise ValueError(
                        f"Week/global metadata mismatch in week '{wk}': {field}={week_meta.get(field)!r} expected={expected!r}"
                    )
            for field, expected in expected_mode.items():
                observed = week_meta.get(field)
                if observed != expected:
                    raise ValueError(
                        f"Inconsistent ingest mode in week '{wk}': {field}={observed!r} (expected {expected!r})"
                    )

    write_json_atomic_with_backup(Path(out_root) / "meta.json", meta)

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
            "pca_project_s": float(_FLUSH_PERF.get("pca_project_s", 0.0)),
            "feature_flush_s": float(_FLUSH_PERF.get("feature_flush_s", 0.0)),
        },
    )
    elapsed_total = max(1e-9, time.monotonic() - ingest_started)
    print(f"[ingest] rows_per_sec={total_feature_rows/elapsed_total:.2f} queue_full_count={feeder._queue_full_count}", flush=True)

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
    if len(pairs) not in (4, 5):
        raise ValueError(
            f"Need exactly 4 or 5 distinct consecutive weeks of data after BYBIT_WEEKS filtering; found {len(pairs)}."
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
    protocol = FIVE_WEEK_PROTOCOL if len(selected_weeks) == 5 else FOUR_WEEK_PROTOCOL
    if APPEND_MISSING_WEEKS:
        if protocol != FIVE_WEEK_PROTOCOL:
            raise ValueError("BYBIT_APPEND_MISSING_WEEKS=1 requires exactly 5 weeks")
        if not PCA_USE_EXISTING:
            raise ValueError("BYBIT_APPEND_MISSING_WEEKS=1 requires BYBIT_PCA_USE_EXISTING=1")

    if protocol == FOUR_WEEK_PROTOCOL:
        week1, week2, week3, week4 = selected_weeks
        print(
            f"[split] protocol={FOUR_WEEK_PROTOCOL} cmssl.train={week1} cmssl.val={week2} cmssl.test={week3} rl={week3} eval={week4}"
        )
        train_week_keys = [week1]
    else:
        week1, week2, week3, week4, week5 = selected_weeks
        print(
            f"[split] protocol={FIVE_WEEK_PROTOCOL} cmssl.train={week1},{week2} cmssl.val={week3} cmssl.test={week4} rl={week4} eval={week5}"
        )
        print(f"[append] enabled={str(bool(APPEND_MISSING_WEEKS)).lower()}")
        train_week_keys = [week1, week2]

    pca_fit_meta = maybe_fit_pca_model(
        pairs,
        OUT_ROOT,
        train_week_keys,
        PCA_VAR_TARGET,
        PCA_SELECT_MODE,
        PCA_MAX_COMPONENTS,
        PCA_BATCH_SIZE,
        PCA_MODEL_FILENAME,
        PCA_USE_EXISTING,
    )

    process_all(
        pairs,
        OUT_ROOT,
        pca_fit_meta,
        protocol=protocol,
        append_missing_weeks=bool(APPEND_MISSING_WEEKS),
    )

if __name__ == "__main__":
    main()
