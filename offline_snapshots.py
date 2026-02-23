#!/usr/bin/env python3
"""
Rebuild 100ms order-book snapshots from Bybit OB delta streams.

Outputs canonical snapshots.npz files compatible with RL_exec.load_raw_snapshots.

Env defaults:
  BYBIT_OB_DIR=/home/gabrool/Documents/OB
  BYBIT_OUT_ROOT=/media/gabrool/Expansion/Gabriel/bybit_offline_dt
  BYBIT_WEEKS=""  # optional comma-separated week keys
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from CMSSL17 import FeatureEngine, _open_text


# fast json if available
_orjson_spec = importlib.util.find_spec("orjson")
if _orjson_spec is not None:
    import orjson as _fastjson

    def fast_json_loads(s: str):
        return _fastjson.loads(s)
else:
    def fast_json_loads(s: str):
        return json.loads(s)


OB_DIR = os.environ.get("BYBIT_OB_DIR", "/home/gabrool/Documents/OB")
OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "/media/gabrool/Expansion/Gabriel/bybit_offline_dt")
RAW_BYBIT_WEEKS = os.environ.get("BYBIT_WEEKS", "")


def _parse_requested_weeks(raw: str) -> List[str]:
    items = [wk.strip() for wk in re.split(r"[\s,]+", raw) if wk.strip()]
    return items


def extract_week_key_from_name(name: str) -> str:
    m = re.search(r"\d{2}-\d{2}-\d{4}-to-\d{2}-\d{2}-\d{4}", name)
    if m:
        return m.group(0)
    m = re.search(r"\d{4}-\d{2}-\d{2}-to-\d{4}-\d{2}-\d{2}", name)
    if m:
        return m.group(0)
    raise ValueError(f"Could not extract week key from file name: {name}")


def _parse_week_key_any(base: str) -> Tuple[datetime, datetime, str]:
    wk = re.sub(r"^(BTCUSDT_(?:OB|TH)_)", "", base)
    wk = re.sub(r"\.(?:zip|gz|jsonl|csv)$", "", wk)
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


def list_ob_files(ob_dir: str) -> List[str]:
    candidates = sorted(Path(ob_dir).glob("BTCUSDT_OB_*"))
    out = []
    for path in candidates:
        if path.is_file():
            out.append(str(path))
    return out


def iter_ob_events(ob_path: str):
    with _open_text(ob_path) as f:
        for line in f:
            if not line:
                continue
            obj = fast_json_loads(line)
            yield obj


@dataclass
class SnapshotSeries:
    ts: List[int]
    best_bid: List[float]
    best_ask: List[float]
    best_bid_size: List[float]
    best_ask_size: List[float]

    def append(self, ts_ms: int, bid: float, ask: float, bid_size: float, ask_size: float) -> None:
        # Invariant: every per-row series must be appended in lockstep.
        self.ts.append(int(ts_ms))
        self.best_bid.append(float(bid))
        self.best_ask.append(float(ask))
        self.best_bid_size.append(float(bid_size))
        self.best_ask_size.append(float(ask_size))

    def to_npz(self, path: Path) -> None:
        # Invariant: all per-row arrays must remain equal length.
        n_rows = len(self.ts)
        if not (
            n_rows == len(self.best_bid) == len(self.best_ask) == len(self.best_bid_size) == len(self.best_ask_size)
        ):
            raise ValueError("SnapshotSeries arrays have mismatched lengths")
        snapshots = np.column_stack([self.best_bid, self.best_ask]).astype(np.float32)
        np.savez_compressed(
            path,
            ts=np.asarray(self.ts, dtype=np.int64),
            snapshots=snapshots,
        )


def build_snapshots_from_ob(ob_path: str) -> SnapshotSeries:
    fe = FeatureEngine()
    series = SnapshotSeries(ts=[], best_bid=[], best_ask=[], best_bid_size=[], best_ask_size=[])

    next_sample_ts: Optional[int] = None
    last_bid: Optional[float] = None
    last_ask: Optional[float] = None
    last_bid_size: Optional[float] = None
    last_ask_size: Optional[float] = None

    for raw in iter_ob_events(ob_path):
        etype, ts_ms, payload = fe._parse_event(raw)
        if etype != "ob":
            continue

        if (
            last_bid is not None
            and last_ask is not None
            and last_bid_size is not None
            and last_ask_size is not None
            and next_sample_ts is not None
        ):
            while next_sample_ts < ts_ms:
                series.append(next_sample_ts, last_bid, last_ask, last_bid_size, last_ask_size)
                next_sample_ts += 100

        fe._update_book_from_ob(payload)
        bid, ask, bid_size, ask_size = fe._book_best()
        if bid <= 0.0 or ask <= 0.0:
            continue

        if next_sample_ts is None:
            next_sample_ts = int(ts_ms)

        while next_sample_ts <= ts_ms:
            series.append(next_sample_ts, bid, ask, bid_size, ask_size)
            next_sample_ts += 100

        last_bid = bid
        last_ask = ask
        last_bid_size = bid_size
        last_ask_size = ask_size

    return series


def resolve_weeks(ob_files: Iterable[str], requested: Optional[List[str]]) -> List[Tuple[str, str]]:
    requested_set = None
    if requested:
        requested_set = {extract_week_key_from_name(wk) for wk in requested}

    rows = []
    for path in ob_files:
        wk_key = extract_week_key_from_name(os.path.basename(path))
        if requested_set and wk_key not in requested_set:
            continue
        _start_dt, end_dt, _wk = _parse_week_key_any(wk_key)
        rows.append((end_dt, wk_key, path))
    rows.sort()
    return [(wk, path) for _end, wk, path in rows]


def write_week_snapshots(out_root: str, week_key: str, series: SnapshotSeries, *, overwrite: bool) -> Path:
    week_dir = Path(out_root) / week_key
    week_dir.mkdir(parents=True, exist_ok=True)
    out_path = week_dir / "snapshots.npz"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Snapshot file exists: {out_path} (use --overwrite)")
    series.to_npz(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ob-dir", default=OB_DIR, help="Directory containing BTCUSDT_OB_* files")
    parser.add_argument("--out-root", default=OUT_ROOT, help="Output root matching offline_ingest")
    parser.add_argument("--weeks", default=RAW_BYBIT_WEEKS, help="Comma-separated week keys")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing snapshot files")
    args = parser.parse_args()

    requested = _parse_requested_weeks(args.weeks) if args.weeks else None
    ob_files = list_ob_files(args.ob_dir)
    if not ob_files:
        raise FileNotFoundError(f"No OB files found in {args.ob_dir}")

    weeks = resolve_weeks(ob_files, requested)
    if not weeks:
        raise ValueError("No matching weeks found for requested filter.")

    for wk, ob_path in weeks:
        print(f"[snapshots] week={wk} ob={os.path.basename(ob_path)}")
        series = build_snapshots_from_ob(ob_path)
        if not series.ts:
            print(f"  [skip] no snapshots produced for {wk}")
            continue
        out_path = write_week_snapshots(args.out_root, wk, series, overwrite=args.overwrite)
        print(f"  [write] {out_path} rows={len(series.ts):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
