#!/usr/bin/env python3
"""
Decision-time ingest (memory-safe):
- Snapshot ONE [LOOKBACK, F] sequence at each decision time.
- Use a RAM budget to auto-size chunked writes (avoid huge in-RAM lists).

Input layout support:
- OB: YYYY-MM-DD_BTCUSDT_...ob...*.zip.
- TH: BTCUSDTYYYY-MM-DD.csv.gz, with tolerant handling for .csv / .csv.gzip.

Downstream ingest contract:
- pair_weeks() groups aligned daily OB/TH files into consecutive 7-day blocks
  and emits canonical week keys: DD-MM-YYYY-to-DD-MM-YYYY.
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
  BYBIT_CHUNK_SIZE=4096              # 0 = auto from budget; else fixed size

Shared constants from CMSSL17:
  LOOKBACK (and related model/data constants) are defined in CMSSL17.py.
  If these values are intentionally changed, update them in CMSSL17.py.
  The decision-time grid contract is centralized in CMSSL17.py.
"""

import os, sys, csv, json, re, time, logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Optional
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
CHUNK_SIZE  = int(os.environ.get("BYBIT_CHUNK_SIZE", "4096"))
DECISION_POLICY = "ob_only_grid_quantized"


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
BYBIT_BAD_EXAMPLES_N = int(os.environ.get("BYBIT_BAD_EXAMPLES_N", "25"))
BYBIT_BAD_FRAC_ABORT = float(os.environ.get("BYBIT_BAD_FRAC_ABORT", "0.005"))
BYBIT_BAD_ABS_ABORT = int(os.environ.get("BYBIT_BAD_ABS_ABORT", "50000"))


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
# import your training utilities
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from CMSSL17 import (
    FeatureEngine,
    LabelBuilder,
    quantize_ts_ms,
    HORIZONS_MS,
    NUM_HORIZONS,
    LOOKBACK,
    AUX_DIM,
    _open_text,
    TIME_GRID_STEP_MS,
    TIME_GRID_GUARD_MS,
    timestamp_to_ms_half_even,
)  # keep shared model/data constants only; ingest helpers are local below
# LOOKBACK is a shared model constant from CMSSL17 (single source of truth).

DECISION_NOMINAL_STEP_MS = int(TIME_GRID_STEP_MS)
DECISION_GUARD_MS = int(TIME_GRID_GUARD_MS)

GRACE_MS = max(int(h) for h in HORIZONS_MS)
EVENT_QUEUE_MAXSIZE = 4096
# Weekly chaining guard for multi-file weeks.
WEEK_CHAIN_TS_TOLERANCE_MS = int(BYBIT_TS_BACKSTEP_CLAMP_MS)
DAY_CLIP_DELTA = timedelta(days=BYBIT_DAY_CLIP)

# fast json if available
try:
    import orjson as _fastjson
    def fast_json_loads(s: str): return _fastjson.loads(s)
except Exception:
    import json as _fastjson
    def fast_json_loads(s: str): return _fastjson.loads(s)

# --------------- utils ------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)


def merge_event_time(ob_iter, tr_iter, B: int = 0):
    """Merge OB/trade events by timestamp/sequence with a monotonicity guard."""
    ob_item = next(ob_iter, None)
    tr_item = next(tr_iter, None)
    last_ts = -1
    while ob_item or tr_item:
        if ob_item and (tr_item is None or ob_item[0] < tr_item[0]):
            ts, seq, data = ob_item
            ob_item = next(ob_iter, None)
            etype = "ob"
        else:
            # Prefer the trade when timestamps tie to preserve causal ordering.
            ts, seq, data = tr_item
            tr_item = next(tr_iter, None)
            etype = "trade"
        if ts + B < last_ts:
            raise ValueError("Non-monotonic timestamps in event stream")
        last_ts = ts
        yield etype, ts, seq, data


def build_sequence_from_tokens(tokens: deque, lookback: int) -> np.ndarray:
    """Local sequence builder used by ingest; intentionally not imported from CMSSL17."""
    """
    Build a fixed-length [L, F] sequence from a deque of tokens (each 1D np.array of size F).
    - If len(tokens) >= L: trim older (deque already keeps last L if maxlen=L).
    - If len(tokens) <  L: left-pad by repeating the earliest token.
      Important: set aux Δt for pads to 0 so padding doesn't distort time/CPC.
    """
    assert len(tokens) >= 1
    if len(tokens) >= lookback:
        return np.stack(list(tokens), axis=0)

    pad_n = lookback - len(tokens)
    first = tokens[0].copy()
    # Last channels are [log_dt_ms, is_trade, log_events_100ms, log_events_250ms, log_events_500ms].
    first[-AUX_DIM:] = 0.0
    pad_block = np.repeat(first[None, :], pad_n, axis=0)
    arr = np.stack(list(tokens), axis=0)
    return np.concatenate([pad_block, arr], axis=0)


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
        common_days: Sorted list of dates known to exist in both OB/TH maps.
        strict: If True, raise on any day-to-day gap inside a 7-day block.
            If False, skip invalid blocks and continue.

    Returns:
        A list of valid week blocks (each block has exactly 7 dates).
    """
    groups: List[List[date]] = []
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
            expected = block[i - 1] + DAY_CLIP_DELTA
            if block[i] != expected:
                gap_idx = i
                break

        if gap_idx is not None:
            prev_day = block[gap_idx - 1]
            curr_day = block[gap_idx]
            expected_day = prev_day + DAY_CLIP_DELTA
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
    Discover aligned OB/TH daily inputs and emit 7-day week groups.

    Returns:
        List of (week_key, ob_paths, th_paths), ordered by block end date ascending.
        `ob_paths`/`th_paths` are ordered 7-element file-path lists (one per
        day in each week block). Day parity is strict: OB/TH must have exact
        matching daily coverage before grouping.
    """
    ob_by_day = _build_ob_daily_map(ob_dir)
    th_by_day = _build_th_daily_map(th_dir)

    if not ob_by_day:
        raise ValueError(
            "No OB daily files found. Expected filenames like "
            "'2024-01-15_BTCUSDT_orderbook.ob.zip' (YYYY-MM-DD_BTCUSDT_*ob*.zip)."
        )
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

    week_blocks = _group_common_days_into_weeks(common_days, strict=bool(BYBIT_STRICT_DATA))
    rows = []
    for block in week_blocks:
        week_key = _week_key_from_dates(block[0], block[-1])
        ob_paths = [ob_by_day[d] for d in block]
        th_paths = [th_by_day[d] for d in block]
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
    for idx in range(1, len(parsed)):
        prev_start, prev_end, prev_wk = parsed[idx - 1]
        next_start, next_end, next_wk = parsed[idx]
        expected_next_start = prev_end.date() + DAY_CLIP_DELTA
        if next_start.date() != expected_next_start:
            relation = "gap" if next_start.date() > expected_next_start else "overlap"
            raise ValueError(
                f"Weeks must be consecutive with no gaps/overlaps; detected {relation} between "
                f"'{prev_wk}' ({prev_start.date()}–{prev_end.date()}) and "
                f"'{next_wk}' ({next_start.date()}–{next_end.date()})."
            )



def classify_week_splits(pairs: List[WeekPair]) -> Tuple[List[str], List[str], List[str]]:
    """
    Apply the N-week split policy for train/val/test assignment.

    Policy:
      - n >= 2 weeks are required.
      - Weeks are assumed already ordered/consecutive (validated in main()).
      - All earlier weeks are TRAIN.
      - The final week is the holdout week for both VAL and TEST.
      - VAL/TEST half/half is enforced downstream using timestamps.
    """
    weeks = [wk for wk, _ob, _th in pairs]
    n = len(weeks)

    if n < 2:
        raise ValueError(
            f"classify_week_splits requires at least two weeks; got {n}."
        )

    train_weeks = weeks[:-1]
    val_weeks = [weeks[-1]]
    test_weeks = [weeks[-1]]
    return train_weeks, val_weeks, test_weeks


def _sort_pairs_by_end(pairs: List[WeekPair]) -> List[WeekPair]:
    rows = []
    for wk, ob_p, th_p in pairs:
        _start_dt, end_dt, _ = _parse_week_key_any(wk)
        rows.append((end_dt, wk, ob_p, th_p))
    rows.sort()
    return [(wk, ob_p, th_p) for _end, wk, ob_p, th_p in rows]


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
            if isinstance(data, dict):
                seq = _try_int(data.get("seq"), 0)
            else:
                dq_day.increment_counter("ob", "missing_data")
                seq = 0

            last_ts_out = int(ts_ms)
            emitted += 1
            dq_day.increment_counter("ob", "emitted")
            dq_day.update_output_ts(last_ts_out)
            yield last_ts_out, seq, obj

    dq_day.increment_counter("ob", "total_seen", total)
    dq_day.increment_counter("ob", "total_emitted", emitted)


def safe_th_iter(th_path, day_start_ms, day_end_ms, dq_day):
    total = 0
    emitted = 0
    last_ts_out: Optional[int] = None
    day_clip_enabled = bool(BYBIT_DAY_CLIP)

    with _open_text(th_path) as f:
        reader = csv.DictReader(f)
        for seq, row in enumerate(reader, start=1):
            total += 1
            dq_day.increment_counter("th", "total")
            t_raw = row.get("timestamp")
            if t_raw is None or (isinstance(t_raw, str) and not t_raw.strip()):
                dq_day.increment_counter("th", "missing_ts")
                dq_day.append_example("th_missing_ts", {"seq": seq, "row": row})
                continue
            try:
                ts_ms = timestamp_to_ms_half_even(t_raw)
            except Exception:
                dq_day.increment_counter("th", "bad_ts")
                dq_day.append_example("th_bad_ts", {"seq": seq, "ts_raw": t_raw, "row": row})
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

            row["seq"] = seq
            last_ts_out = int(ts_ms)
            emitted += 1
            dq_day.increment_counter("th", "emitted")
            dq_day.update_output_ts(last_ts_out)
            yield last_ts_out, seq, row

    dq_day.increment_counter("th", "total_seen", total)
    dq_day.increment_counter("th", "total_emitted", emitted)

def build_token(fe: FeatureEngine, feat_z, is_trade: bool, dt_ms: float) -> np.ndarray:
    # exact tail order: [log_dt_ms, is_trade, log_events_100ms, log_events_250ms, log_events_500ms]
    aux_tail = np.array(
        [
            np.log1p(float(dt_ms)),
            float(is_trade),
            np.log1p(fe.event_density_100ms()),
            np.log1p(fe.event_density_250ms()),
            np.log1p(fe.event_density_500ms()),
        ],
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
        self.TS     = np.empty((self.N,), dtype=np.int64)
        self.i = 0
        self.cid = 0
        self.chunks_meta = []

    def add(self, ts_decision_ms: int, seq: np.ndarray, y: np.ndarray):
        core = seq[:, :self.F_core]
        aux  = seq[:, self.F_core:]
        self.X_core[self.i] = core
        self.X_aux[self.i]  = aux
        self.Y[self.i]      = y
        self.TS[self.i]     = ts_decision_ms
        self.i += 1
        if self.i >= self.N:
            self.flush()

    def flush(self):
        if self.i == 0: return
        x_core_path = os.path.join(self.out_dir, f"Xcore_{self.cid:03d}.npy")
        x_aux_path  = os.path.join(self.out_dir, f"Xaux_{self.cid:03d}.npy")
        y_path      = os.path.join(self.out_dir, f"y_{self.cid:03d}.npy")
        ts_path     = os.path.join(self.out_dir, f"ts_{self.cid:03d}.npy")

        # optional: warn if core would overflow fp16
        if self.core_dtype == np.float16:
            maxabs = float(np.max(np.abs(self.X_core[:self.i])))
            if maxabs > np.finfo(np.float16).max:
                print(f"[warn] core max {maxabs:.1f} exceeds fp16 range; consider BYBIT_SAVE_DTYPE=bf16", flush=True)

        np.save(x_core_path, self.X_core[:self.i].astype(self.core_dtype, copy=False))
        np.save(x_aux_path,  self.X_aux[:self.i])                 # fp32
        np.save(y_path,      self.Y[:self.i])                     # fp32
        np.save(ts_path,     self.TS[:self.i])                    # int64

        self.chunks_meta.append({
            "chunk": int(self.cid),
            "n": int(self.i),
            "files": {"core": os.path.basename(x_core_path),
                      "aux":  os.path.basename(x_aux_path),
                      "y":    os.path.basename(y_path),
                      "ts":   os.path.basename(ts_path)}
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


    def add(self, ts_decision_ms: int, seq: np.ndarray, label: np.ndarray):
        wk = self._find_week_key(ts_decision_ms)
        writer = self._ensure_writer(wk)
        writer.add(ts_decision_ms, seq, label)
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
        meta = {
            "week": week_key,
            "decision_policy": DECISION_POLICY,
            "decision_nominal_step_ms": int(DECISION_NOMINAL_STEP_MS),
            "decision_guard_ms": int(DECISION_GUARD_MS),
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
        print(f"[write] week={week_key} chunks={len(chunks_meta)} rows={rows_total}", flush=True)

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
        end_ms = _dt_to_epoch_ms(end_dt + DAY_CLIP_DELTA)
        index.append((wk, start_ms, end_ms))
    index.sort(key=lambda x: x[1])
    return index




def _iter_week_merged_events(
    week_key: str,
    ob_paths: List[str],
    th_paths: List[str],
):
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

    last_ts_global: Optional[int] = None
    prev_ob_name: Optional[str] = None
    prev_th_name: Optional[str] = None

    for ob_path, th_path in zip(ob_list, th_list):
        ob_name = os.path.basename(ob_path)
        th_name = os.path.basename(th_path)
        day = _daily_path_day(ob_path, "OB")
        day_start_ms = _dt_to_epoch_ms(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc))
        day_end_ms = _dt_to_epoch_ms(datetime.combine(day + DAY_CLIP_DELTA, datetime.min.time(), tzinfo=timezone.utc))
        dq_day = DayQuality(
            day=day.isoformat(),
            ob_path=ob_path,
            th_path=th_path,
        )
        ob_iter = safe_ob_iter(ob_path, day_start_ms, day_end_ms, dq_day)
        th_iter = safe_th_iter(th_path, day_start_ms, day_end_ms, dq_day)
        for event in merge_event_time(ob_iter, th_iter, B=0):
            _etype, ts, _seq, _data = event
            if (
                last_ts_global is not None
                and ts + WEEK_CHAIN_TS_TOLERANCE_MS < last_ts_global
            ):
                prev_pair = (
                    f"{prev_ob_name} | {prev_th_name}"
                    if prev_ob_name is not None and prev_th_name is not None
                    else "<week-start>"
                )
                raise ValueError(
                    "Non-monotonic timestamps while chaining daily files within week: "
                    f"week={week_key} "
                    f"prev_day_files={prev_pair} "
                    f"curr_day_files={ob_name} | {th_name} "
                    f"prev_ts={last_ts_global} curr_ts={ts} "
                    f"tolerance_ms={WEEK_CHAIN_TS_TOLERANCE_MS}"
                )

            last_ts_global = ts
            prev_ob_name = ob_name
            prev_th_name = th_name
            yield event

class EventFeeder:
    def __init__(
        self,
        pairs: List[WeekPair],
        maxsize: int = EVENT_QUEUE_MAXSIZE,
    ):
        self.pairs = list(pairs)
        self.queue: "queue.Queue[Tuple[str, Optional[str], Optional[object]]]" = queue.Queue(maxsize=maxsize)
        self._last_first_ts: Optional[int] = None

    def _put(self, item: Tuple[str, Optional[str], Optional[object]]):
        self.queue.put(item)

    def run(self):
        try:
            for wk, ob_paths, th_paths in self.pairs:
                merged = _iter_week_merged_events(wk, ob_paths, th_paths)

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


def _stream_core_features(pairs: List[WeekPair]):
    """Stream OB decision-candidate core feature vectors (z-scored) for PCA fitting."""
    if not pairs:
        return

    fe = FeatureEngine()
    sample_count = 0
    last_log = time.monotonic()
    last_wk = None

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
            if _is_trade:
                continue
            sample_count += 1
            now = time.monotonic()
            if now - last_log >= 300:
                print(f"[pca-sample] rows={sample_count} last_wk={last_wk}", flush=True)
                last_log = now
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

    Each pair is ``(week_key, ob_paths, th_paths)`` where ``ob_paths`` and
    ``th_paths`` are ordered lists of per-day file paths for that week.
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
    split_info: Optional[Dict[str, List[str]]] = None,
):
    """Run ingest across week pairs composed of ordered daily OB/TH file lists."""
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
    # Label timing is aligned to the same quantized 100ms grid as decisions.
    # Entry references use decision_ts directly (no sub-grid delay).
    labeler = LabelBuilder(delta_ms=0, horizons_ms=HORIZONS_MS)

    tokens_buf: deque = deque(maxlen=LOOKBACK)
    pending_seqs: deque = deque()
    last_grid_ts: Optional[int] = None
    last_tick_dt_ms: Optional[int] = None

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

            grid_ts = quantize_ts_ms(int(ts_ms), DECISION_NOMINAL_STEP_MS, DECISION_GUARD_MS)
            matured = None

            is_new_tick = last_grid_ts is None or grid_ts > last_grid_ts
            is_collision = (last_grid_ts is not None and grid_ts == last_grid_ts)

            if is_new_tick:
                dt_tick = DECISION_NOMINAL_STEP_MS if last_grid_ts is None else int(grid_ts - last_grid_ts)
                tok = build_token(fe, feat_core, is_trade, dt_tick)
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
                pending_seqs.append((grid_ts, seq.astype(np.float32, copy=False)))
                # Decision frontier must advance only on a new decision-time tick.
                labeler.on_decision(grid_ts)
                # register decision at tick first, then advance/update tick price.
                matured = labeler.on_event(grid_ts, float(mid))
                last_grid_ts = grid_ts
                last_tick_dt_ms = int(dt_tick)
            elif is_collision:
                if not pending_seqs:
                    raise RuntimeError("Grid collision observed but no pending sequence to overwrite")
                if last_tick_dt_ms is None:
                    raise RuntimeError("Grid collision observed without last_tick_dt_ms state")
                tok = build_token(fe, feat_core, is_trade, int(last_tick_dt_ms))
                if not tokens_buf:
                    raise RuntimeError("Grid collision observed but token buffer is empty")
                tokens_buf[-1] = tok

                seq = build_sequence_from_tokens(tokens_buf, LOOKBACK)
                # Collision updates the current decision token/sequence in place.
                pending_seqs[-1] = (grid_ts, seq.astype(np.float32, copy=False))
                matured = labeler.on_event(grid_ts, float(mid))
            else:
                raise RuntimeError(
                    f"Non-monotone grid timestamp: grid_ts={grid_ts} < last_grid_ts={last_grid_ts}"
                )

            if matured is None:
                raise RuntimeError("Matured labels were not produced for OB event")
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
        
        if time.monotonic() - last_log >= 300:
            print(f"[tok  ] seq={total_sequences} weeks={week_counter}/{week_total} "
                  f"chunkN={router.chunk_size_used if router else 0}", flush=True)
            last_log = time.monotonic()


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

    split_ranges = None
    if split_info and len(weeks_in_order) >= 2:
        holdout_week = weeks_in_order[-1]
        train_weeks = weeks_in_order[:-1]

        train_week_mins = []
        train_week_maxs = []
        for wk in train_weeks:
            wk_meta = week_meta_records.get(wk)
            if not wk_meta or "decision_ts_range" not in wk_meta:
                raise ValueError(
                    f"Missing decision_ts_range for week '{wk}'; cannot derive train split range."
                )
            decision_range = wk_meta["decision_ts_range"]
            train_week_mins.append(int(decision_range["min"]))
            train_week_maxs.append(int(decision_range["max"]))

        holdout_meta = week_meta_records.get(holdout_week)
        if not holdout_meta or "decision_ts_range" not in holdout_meta:
            raise ValueError(
                f"Missing decision_ts_range for week '{holdout_week}'; cannot derive val/test split ranges."
            )

        holdout_range = holdout_meta["decision_ts_range"]
        holdout_min = int(holdout_range["min"])
        holdout_max = int(holdout_range["max"])
        if holdout_max <= holdout_min:
            raise ValueError(
                f"Week '{holdout_week}' decision_ts_range invalid: min={holdout_min} max={holdout_max}"
            )

        midpoint = holdout_min + (holdout_max - holdout_min) // 2
        split_ranges = {
            "train_week": train_weeks[-1],
            "holdout_week": holdout_week,
            "train_ts_range": {
                "min": min(train_week_mins),
                "max": max(train_week_maxs),
            },
            "val_ts_range": {"min": holdout_min, "max": midpoint},
            "test_ts_range": {"min": midpoint, "max": holdout_max},
        }

    # Dataset metadata contract: `weeks_in_order` is the only supported key for
    # week ordering in OUT_ROOT/meta.json.
    meta = {
        "dataset_start": start_iso,
        "dataset_end": end_iso,
        "weeks_in_order": weeks_in_order,
        "decision_policy": DECISION_POLICY,
        "decision_nominal_step_ms": int(DECISION_NOMINAL_STEP_MS),
        "time_grid": {
            "step_ms": int(DECISION_NOMINAL_STEP_MS),
            "guard_ms": int(DECISION_GUARD_MS),
            "mode": "nearest",
        },
        "lookback": int(LOOKBACK),
        "feature_dim_total": feature_dim_total,
        "feature_dim_core": feature_dim_core,
        "aux_tail": ["log_dt_ms", "is_trade", "log_events_100ms", "log_events_250ms", "log_events_500ms"],
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
        "quality_config": quality_env_config(),
    }
    meta["pca"] = dict(pca_summary)
    if pca_var_ratio is not None:
        meta["pca"]["explained_variance_ratio"] = [float(x) for x in pca_var_ratio]
    if split_info:
        meta["splits"] = {key: list(vals) for key, vals in split_info.items()}
        if split_ranges:
            meta["splits"].update(split_ranges)
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
    # Week selection contract:
    #   1) start from all discovered OB/TH week pairs on disk
    #   2) optionally filter explicitly via BYBIT_WEEKS
    # No implicit "last K weeks" or anchor-date selection is applied.
    pairs = pair_weeks(OB_DIR, TH_DIR)

    if not pairs:
        print(f"No week pairs found under OB_DIR={OB_DIR} and TH_DIR={TH_DIR}")
        return

    requested_weeks = _parse_requested_weeks(RAW_BYBIT_WEEKS)

    if requested_weeks:
        week_lookup = {wk for wk, _ob, _th in pairs}
        missing = [wk for wk in requested_weeks if wk not in week_lookup]
        if missing:
            raise ValueError(
                f"Requested BYBIT_WEEKS not found in available data: {', '.join(missing)}"
            )
        requested_unique = list(dict.fromkeys(requested_weeks))
        if len(requested_unique) < 2:
            raise ValueError(
                f"BYBIT_WEEKS must include at least two distinct weeks; got {len(requested_unique)}."
            )
        requested_set = set(requested_unique)
        pairs = [pair for pair in pairs if pair[0] in requested_set]

    pairs = _sort_pairs_by_end(pairs)
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least two weeks of data after selection; found {len(pairs)}."
        )

    _assert_week_order(pairs)
    _assert_weeks_consecutive(pairs)

    chosen_weeks = [wk for wk, _ob, _th in pairs]

    print(f"[plan ] weeks={len(pairs)} "
          f"RAM={RAM_BUDGET}MB chunk_size={CHUNK_SIZE if CHUNK_SIZE>0 else 'auto'}")
    print(f"[weeks] {', '.join(chosen_weeks)}")

    print(f"[paths] OB_DIR={OB_DIR}")
    print(f"[paths] TH_DIR={TH_DIR}")
    print(f"[out  ] OUT_ROOT={OUT_ROOT}")


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
        PCA_USE_EXISTING,
    )

    process_all(pairs, OUT_ROOT, pca_fit_meta, split_info=split_info)

if __name__ == "__main__":
    main()
