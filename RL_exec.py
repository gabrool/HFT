import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from CMSSL17 import SAMBA, ModelArgs, DMODEL, MAMBA_LAYERS, LOOKBACK
from offline_tokens import iter_week_chunks, load_global_meta

RAW_SNAPSHOT_PATHS = [
    Path(p)
    for p in os.environ.get("RAW_SNAPSHOT_PATHS", "").split(",")
    if p
]
RAW_SNAPSHOT_EXPECTED_STEP_MS = 100
RAW_SNAPSHOT_TOLERANCE_MS = 20
RAW_SNAPSHOT_MAX_IRREGULAR_FRAC = 0.05
SHORT_VOL_WINDOW = 50
LONG_VOL_WINDOW = 200


def load_cmssl(out_root: str, ckpt_path: str, device: str = "cuda"):
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    feat_dim = int(meta["feature_dim_total"])  # includes AUX_DIM already

    args = ModelArgs(DMODEL, MAMBA_LAYERS, feat_dim, LOOKBACK)
    model = SAMBA(args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def cmssl_predict(model, x_core, x_aux, meta, device: str = "cuda"):
    # x_core: [B, L, F_core]  x_aux: [B, L, AUX_DIM]
    x_core = torch.as_tensor(x_core, device=device)
    x_aux = torch.as_tensor(x_aux, device=device)
    x = torch.cat([x_core, x_aux], dim=-1)
    mask_idx = torch.empty((x.shape[0], 0), dtype=torch.long, device=device)
    ret_pred, vol_pred, dir_logits, *_ = model(x, mask_ratio=0.0, mask_idx=mask_idx)
    horizons = meta.get("horizons_ms", [])
    expected_h = len(horizons)
    assert expected_h > 0, "meta['horizons_ms'] must be non-empty"
    assert ret_pred.shape[-1] == expected_h, (
        f"ret_pred shape {ret_pred.shape} does not match horizons {expected_h}"
    )
    assert vol_pred.shape[-1] == expected_h, (
        f"vol_pred shape {vol_pred.shape} does not match horizons {expected_h}"
    )
    assert dir_logits.shape[-1] == expected_h, (
        f"dir_logits shape {dir_logits.shape} does not match horizons {expected_h}"
    )
    return ret_pred, vol_pred, dir_logits


def iter_chunk_batches(out_root: str):
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    for week, week_meta, week_dir in iter_week_chunks(out_root, meta=meta):
        for entry in week_meta.get("chunks", []):
            files = entry.get("files", {})
            x_core = np.load(week_dir / files["core"])
            x_aux = np.load(week_dir / files["aux"])
            y = np.load(week_dir / files["y"])
            ts_path = files.get("ts")
            ts = np.load(week_dir / ts_path) if ts_path else None
            yield week, int(entry.get("chunk", 0)), ts, x_core, x_aux, y


def _decision_ts_bounds(week_key: str, week_meta: dict) -> tuple[int, int]:
    ts_range = week_meta.get("decision_ts_range")
    assert ts_range, f"week {week_key} missing decision_ts_range in meta_week.json"
    ts_min = int(ts_range["min"])
    ts_max = int(ts_range["max"])
    assert ts_min < ts_max, f"week {week_key} has invalid decision_ts_range: {ts_range}"
    return ts_min, ts_max


def get_cmssl_splits(out_root: str) -> dict:
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    weeks = list(meta.get("weeks", []))
    assert len(weeks) == 2, f"expected exactly 2 weeks, found {len(weeks)}"

    week_meta_map = {wk: wmeta for wk, wmeta, _ in iter_week_chunks(out_root, meta=meta)}
    assert len(week_meta_map) == 2, f"expected two week metas, found {len(week_meta_map)}"
    week1_key, week2_key = weeks
    assert week1_key in week_meta_map and week2_key in week_meta_map, (
        f"week keys {weeks} do not match week metas {list(week_meta_map.keys())}"
    )

    week1_min, week1_max = _decision_ts_bounds(week1_key, week_meta_map[week1_key])
    week2_min, week2_max = _decision_ts_bounds(week2_key, week_meta_map[week2_key])

    week2_span = week2_max - week2_min
    expected_week_ms = 7 * 24 * 60 * 60 * 1000
    expected_half_ms = expected_week_ms / 2.0
    tolerance_ms = 60 * 60 * 1000
    assert abs(week2_span - expected_week_ms) <= tolerance_ms, (
        f"week2 span {week2_span:.0f}ms not ~7 days"
    )

    week2_half = week2_span / 2.0
    assert abs(week2_half - expected_half_ms) <= tolerance_ms, (
        f"week2 half span {week2_half:.0f}ms not ~3.5 days"
    )

    week2_mid = int(week2_min + week2_half)
    return {
        "train": {"week": week1_key, "start": week1_min, "end": week1_max},
        "val": {"week": week2_key, "start": week2_min, "end": week2_mid},
        "test": {"week": week2_key, "start": week2_mid, "end": week2_max},
    }


def build_two_week_time_splits(out_root: str) -> dict:
    return get_cmssl_splits(out_root)


def sigma_from_vol(log_vol: np.ndarray) -> np.ndarray:
    """Recover volatility from log-vol predictions."""
    return np.exp(log_vol)


def spread_bps_from_vol_pred(vol_pred: np.ndarray, spread_mult: float = 1.0) -> np.ndarray:
    """
    Convert model vol predictions into a spread size in basis points.

    vol_pred is trained against y_logvol (log volatility), so we recover
    sigma by exponentiating the log-vol and then scale to bps.
    If the model ever switches to predicting log-variance, use
    sigma = exp(0.5 * logvar) instead.
    """
    sigma_bps = 1e4 * sigma_from_vol(vol_pred)
    return spread_mult * sigma_bps


def alpha_from_probs(p_up: np.ndarray, sigma_bps: np.ndarray) -> np.ndarray:
    """Convert directional probabilities into a signed alpha in bps."""
    return (p_up - 0.5) * 2.0 * sigma_bps


def half_spread(mid: float, spread_bps: float) -> float:
    return mid * spread_bps * 1e-4 / 2.0


def skew(mid: float, alpha_bps: float) -> float:
    return mid * alpha_bps * 1e-4


def load_split_arrays(out_root: str, split: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_core_list: List[np.ndarray] = []
    x_aux_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    for week, _chunk, ts, x_core, x_aux, y in iter_chunk_batches(out_root):
        if week != split["week"]:
            continue
        if ts is None:
            raise ValueError("Chunk timestamps missing; cannot filter by split range.")
        mask = (ts >= split["start"]) & (ts < split["end"])
        if not np.any(mask):
            continue
        x_core_list.append(x_core[mask])
        x_aux_list.append(x_aux[mask])
        y_list.append(y[mask])
        ts_list.append(ts[mask])
    if not x_core_list:
        raise ValueError(f"No data found for split {split}")
    x_core_all = np.concatenate(x_core_list, axis=0)
    x_aux_all = np.concatenate(x_aux_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    ts_all = np.concatenate(ts_list, axis=0)
    order = np.argsort(ts_all)
    return x_core_all[order], x_aux_all[order], y_all[order], ts_all[order]


def resolve_test_split(out_root: str, meta: dict) -> Dict[str, int]:
    splits = meta.get("splits", {})
    test_range = splits.get("test_ts_range")
    holdout_week = splits.get("holdout_week")
    if test_range and holdout_week:
        return {
            "week": holdout_week,
            "start": int(test_range["min"]),
            "end": int(test_range["max"]),
        }
    return get_cmssl_splits(out_root)["test"]


def _build_windowed_inputs(
    x_core: np.ndarray,
    x_aux: np.ndarray,
    ts: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_core.ndim == 3:
        if ts.shape[0] != x_core.shape[0]:
            raise ValueError("Timestamp length does not match windowed inputs.")
        return x_core, x_aux, ts
    if x_core.ndim != 2:
        raise ValueError(f"Expected x_core to be 2D or 3D, got {x_core.ndim}D")
    if ts.ndim != 1 or ts.shape[0] != x_core.shape[0]:
        raise ValueError("Raw token timestamps must be 1D and match x_core rows.")
    n, feat_dim = x_core.shape
    if n < lookback:
        return (
            np.empty((0, lookback, feat_dim), dtype=x_core.dtype),
            np.empty((0, lookback, x_aux.shape[1]), dtype=x_aux.dtype),
            np.empty((0,), dtype=ts.dtype),
        )
    window_count = n - lookback + 1
    x_core_win = np.empty((window_count, lookback, feat_dim), dtype=x_core.dtype)
    x_aux_win = np.empty((window_count, lookback, x_aux.shape[1]), dtype=x_aux.dtype)
    for i in range(lookback - 1, n):
        start = i - lookback + 1
        x_core_win[start] = x_core[start:i + 1]
        x_aux_win[start] = x_aux[start:i + 1]
    return x_core_win, x_aux_win, ts[lookback - 1:]


def load_test_windowed_inputs(
    out_root: str,
    meta: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = resolve_test_split(out_root, meta)
    x_core, x_aux, _y, ts = load_split_arrays(out_root, split)
    return _build_windowed_inputs(x_core, x_aux, ts, lookback=LOOKBACK)


def run_cmssl_test_window_inference(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    batch_size: int = 256,
) -> Dict[str, Any]:
    model, meta = load_cmssl(out_root, ckpt_path, device=device)
    x_core, x_aux, ts = load_test_windowed_inputs(out_root, meta)
    cmssl_out = run_cmssl_inference(model, meta, x_core, x_aux, batch_size=batch_size, device=device)
    horizons = meta.get("horizons_ms", [])
    output: Dict[str, Dict[int, np.ndarray]] = {
        "horizons_ms": horizons,
        "ret_pred": {},
        "vol_pred": {},
        "dir_logits": {},
    }
    for idx, ts_val in enumerate(ts):
        ts_key = int(ts_val)
        output["ret_pred"][ts_key] = cmssl_out["ret_pred"][idx]
        output["vol_pred"][ts_key] = cmssl_out["vol_pred"][idx]
        output["dir_logits"][ts_key] = cmssl_out["dir_logits"][idx]
    return output


def run_cmssl_inference(
    model,
    meta: dict,
    x_core: np.ndarray,
    x_aux: np.ndarray,
    batch_size: int = 256,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    ret_preds: List[np.ndarray] = []
    vol_preds: List[np.ndarray] = []
    dir_logits_list: List[np.ndarray] = []
    n = x_core.shape[0]
    for i in range(0, n, batch_size):
        xc = x_core[i:i + batch_size]
        xa = x_aux[i:i + batch_size]
        ret_pred, vol_pred, dir_logits = cmssl_predict(model, xc, xa, meta, device=device)
        ret_preds.append(ret_pred.detach().cpu().numpy())
        vol_preds.append(vol_pred.detach().cpu().numpy())
        dir_logits_list.append(dir_logits.detach().cpu().numpy())
    return {
        "ret_pred": np.concatenate(ret_preds, axis=0),
        "vol_pred": np.concatenate(vol_preds, axis=0),
        "dir_logits": np.concatenate(dir_logits_list, axis=0),
    }


def _find_week_dir(out_root: Path, week_key: str) -> Path:
    meta = load_global_meta(out_root)
    for wk, _wmeta, wk_dir in iter_week_chunks(out_root, meta=meta):
        if wk == week_key:
            return wk_dir
    raise ValueError(f"Unable to locate week directory for {week_key}")


def _load_snapshot_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot path not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in {".csv", ".gz", ".bz2"}:
        df = pd.read_csv(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        if {"ts", "best_bid", "best_ask"}.issubset(data.files):
            df = pd.DataFrame(
                {
                    "ts": data["ts"],
                    "best_bid": data["best_bid"],
                    "best_ask": data["best_ask"],
                }
            )
        elif {"timestamps", "best_bid", "best_ask"}.issubset(data.files):
            df = pd.DataFrame(
                {
                    "ts": data["timestamps"],
                    "best_bid": data["best_bid"],
                    "best_ask": data["best_ask"],
                }
            )
        elif {"ts", "snapshots"}.issubset(data.files):
            snaps = data["snapshots"]
            if snaps.ndim != 2 or snaps.shape[1] < 2:
                raise ValueError(f"Unsupported snapshots array shape in {path}: {snaps.shape}")
            df = pd.DataFrame(
                {
                    "ts": data["ts"],
                    "best_bid": snaps[:, 0],
                    "best_ask": snaps[:, 1],
                }
            )
        else:
            raise ValueError(f"Unsupported npz layout in {path}")
    elif path.suffix == ".npy":
        arr = np.load(path)
        if arr.dtype.names:
            if {"ts", "best_bid", "best_ask"}.issubset(arr.dtype.names):
                df = pd.DataFrame(
                    {
                        "ts": arr["ts"],
                        "best_bid": arr["best_bid"],
                        "best_ask": arr["best_ask"],
                    }
                )
            elif {"ts", "snapshot"}.issubset(arr.dtype.names):
                snaps = arr["snapshot"]
                df = pd.DataFrame(
                    {
                        "ts": arr["ts"],
                        "best_bid": snaps[:, 0],
                        "best_ask": snaps[:, 1],
                    }
                )
            elif {"ts", "snapshots"}.issubset(arr.dtype.names):
                snaps = arr["snapshots"]
                df = pd.DataFrame(
                    {
                        "ts": arr["ts"],
                        "best_bid": snaps[:, 0],
                        "best_ask": snaps[:, 1],
                    }
                )
            else:
                raise ValueError(f"Unsupported structured snapshot dtype in {path}")
        else:
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"Unsupported raw snapshot array shape in {path}: {arr.shape}")
            df = pd.DataFrame(
                {
                    "ts": arr[:, 0],
                    "best_bid": arr[:, 1],
                    "best_ask": arr[:, 2],
                }
            )
    else:
        raise ValueError(f"Unsupported snapshot file type: {path.suffix}")
    missing = {"ts", "best_bid", "best_ask"} - set(df.columns)
    if missing:
        raise ValueError(f"Snapshot data missing columns {missing} in {path}")
    return df[["ts", "best_bid", "best_ask"]]


def _ensure_sorted_near_regular(ts: np.ndarray) -> None:
    if ts.ndim != 1:
        raise ValueError("Snapshot timestamps must be 1D.")
    if len(ts) < 2:
        return
    if np.any(np.diff(ts) < 0):
        raise ValueError("Snapshot timestamps must be sorted in ascending order.")
    deltas = np.diff(ts)
    median_step = float(np.median(deltas))
    if abs(median_step - RAW_SNAPSHOT_EXPECTED_STEP_MS) > RAW_SNAPSHOT_TOLERANCE_MS:
        raise ValueError(
            "Snapshot cadence not close to 100ms. "
            f"Median step {median_step:.2f}ms."
        )
    irregular_frac = np.mean(np.abs(deltas - RAW_SNAPSHOT_EXPECTED_STEP_MS) > RAW_SNAPSHOT_TOLERANCE_MS)
    if irregular_frac > RAW_SNAPSHOT_MAX_IRREGULAR_FRAC:
        raise ValueError(
            "Snapshot timestamps are too irregular. "
            f"{irregular_frac:.2%} exceed tolerance."
        )


def load_raw_snapshot_features(out_root: str) -> pd.DataFrame:
    if not RAW_SNAPSHOT_PATHS:
        raise ValueError("RAW_SNAPSHOT_PATHS is empty. Set RAW_SNAPSHOT_PATHS or env var.")
    frames = [_load_snapshot_frame(path) for path in RAW_SNAPSHOT_PATHS]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ts", "best_bid", "best_ask"]).copy()
    df["ts"] = df["ts"].astype(np.int64)
    df = df.sort_values("ts").reset_index(drop=True)
    _ensure_sorted_near_regular(df["ts"].to_numpy())
    split = get_cmssl_splits(out_root)["test"]
    df = df[(df["ts"] >= split["start"]) & (df["ts"] < split["end"])].copy()
    _ensure_sorted_near_regular(df["ts"].to_numpy())
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df["spread_bps"] = (df["best_ask"] - df["best_bid"]) / df["mid"] * 1e4
    df["mid_ret_1"] = np.log(df["mid"]).diff()
    df["vol_short"] = df["mid_ret_1"].rolling(SHORT_VOL_WINDOW, min_periods=1).std()
    df["vol_long"] = df["mid_ret_1"].rolling(LONG_VOL_WINDOW, min_periods=1).std()
    return df


def load_raw_snapshots(out_root: str, week_key: str) -> Tuple[np.ndarray, np.ndarray]:
    week_dir = _find_week_dir(Path(out_root), week_key)
    candidates = [
        week_dir / "raw_snapshots.npz",
        week_dir / "snapshots.npz",
        week_dir / "raw_snapshots.npy",
        week_dir / "snapshots.npy",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No raw snapshot file found in {week_dir}. Expected one of: "
            f"{', '.join(p.name for p in candidates)}"
        )

    if path.suffix == ".npz":
        data = np.load(path)
        if "ts" in data and "snapshots" in data:
            return data["ts"], data["snapshots"]
        if "timestamps" in data and "X" in data:
            return data["timestamps"], data["X"]
        raise ValueError(f"Unsupported npz layout in {path}")

    arr = np.load(path)
    if arr.dtype.names:
        if "ts" in arr.dtype.names and "snapshot" in arr.dtype.names:
            return arr["ts"], arr["snapshot"]
        if "ts" in arr.dtype.names and "snapshots" in arr.dtype.names:
            return arr["ts"], arr["snapshots"]
    ts_path = path.with_name("snapshots_ts.npy")
    if ts_path.exists():
        return np.load(ts_path), arr
    raise ValueError(f"Unsupported raw snapshot layout in {path}")


def _format_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _format_duration_ms(duration_ms: int) -> str:
    seconds = duration_ms / 1000.0
    minutes = seconds / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    return f"{duration_ms}ms (~{days:.2f}d)"


def _ensure_monotonic(ts: np.ndarray, label: str) -> None:
    if ts.size and np.any(np.diff(ts) < 0):
        raise ValueError(f"{label} timestamps must be monotonically non-decreasing.")


def report_pretrain_diagnostics(out_root: str, splits: Dict[str, Dict[str, int]]) -> None:
    test_split = splits["test"]
    start_ms = int(test_split["start"])
    end_ms = int(test_split["end"])
    duration_ms = end_ms - start_ms
    print(
        "[cmssl split:test]",
        f"week={test_split['week']}",
        f"start={_format_ts(start_ms)}",
        f"end={_format_ts(end_ms)}",
        f"duration={_format_duration_ms(duration_ms)}",
    )
    expected_week_ms = 7 * 24 * 60 * 60 * 1000
    expected_half_ms = int(expected_week_ms / 2.0)
    tolerance_ms = 60 * 60 * 1000
    assert abs(duration_ms - expected_half_ms) <= tolerance_ms, (
        f"Test split duration {duration_ms}ms not ~3.5 days."
    )

    snapshot_ts, _snapshots = load_raw_snapshots(out_root, test_split["week"])
    snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
    snapshot_ts = np.sort(snapshot_ts)
    filtered = snapshot_ts[(snapshot_ts >= start_ms) & (snapshot_ts < end_ms)]
    if filtered.size == 0:
        raise ValueError("No raw snapshots found inside the CMSSL test split range.")
    _ensure_monotonic(filtered, "Raw snapshot (filtered)")
    print(
        "[raw snapshots:test]",
        f"count={filtered.size}",
        f"start={_format_ts(int(filtered[0]))}",
        f"end={_format_ts(int(filtered[-1]))}",
    )


def align_snapshots_to_decisions(
    decision_ts: np.ndarray,
    snapshot_ts: np.ndarray,
    snapshots: np.ndarray,
    label: Optional[str] = None,
    *,
    tolerance_ms: int = 50,
    match_rate_target: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D")
    if decision_ts.ndim != 1:
        raise ValueError("decision_ts must be 1D")
    if decision_ts.size and np.any(np.diff(decision_ts) < 0):
        raise ValueError("decision_ts must be monotonically non-decreasing")
    order = np.argsort(snapshot_ts)
    snapshot_ts = snapshot_ts[order]
    snapshots = snapshots[order]
    if snapshot_ts.size and np.any(np.diff(snapshot_ts) < 0):
        raise ValueError("snapshot_ts must be monotonically non-decreasing after sorting")
    if snapshot_ts.size == 0:
        matched = np.zeros(decision_ts.shape[0], dtype=bool)
        aligned = np.full((decision_ts.shape[0], snapshots.shape[1]), np.nan, dtype=np.float32)
        match_rate = 0.0
        raise ValueError(
            "Snapshot alignment failed; no snapshots available to match decisions. "
            f"match_rate={match_rate:.6f} target={match_rate_target:.6f}"
        )
    insert_idx = np.searchsorted(snapshot_ts, decision_ts, side="left")
    right_valid = insert_idx < snapshot_ts.size
    left_valid = insert_idx > 0
    left_idx = np.clip(insert_idx - 1, 0, snapshot_ts.size - 1)
    right_idx = np.clip(insert_idx, 0, snapshot_ts.size - 1)
    exact = right_valid & (snapshot_ts[right_idx] == decision_ts)
    left_delta = np.full(decision_ts.shape, np.inf, dtype=np.float64)
    right_delta = np.full(decision_ts.shape, np.inf, dtype=np.float64)
    left_delta[left_valid] = np.abs(decision_ts[left_valid] - snapshot_ts[left_idx[left_valid]])
    right_delta[right_valid] = np.abs(snapshot_ts[right_idx[right_valid]] - decision_ts[right_valid])
    nearest_idx = np.where(left_delta <= right_delta, left_idx, right_idx)
    nearest_delta = np.minimum(left_delta, right_delta)
    nearest_idx = np.where(exact, right_idx, nearest_idx)
    nearest_delta = np.where(exact, 0, nearest_delta)
    matched = nearest_delta <= tolerance_ms
    aligned = np.full((decision_ts.shape[0], snapshots.shape[1]), np.nan, dtype=np.float32)
    if np.any(matched):
        aligned[matched] = snapshots[nearest_idx[matched]].astype(np.float32)
    match_rate = float(np.mean(matched)) if matched.size else 0.0
    exact_rate = float(np.mean(exact)) if exact.size else 0.0
    matched_decision_ts = decision_ts[matched]
    matched_snapshot_ts = snapshot_ts[nearest_idx[matched]] if np.any(matched) else np.array([], dtype=np.int64)

    def _median_dt(ts: np.ndarray) -> float:
        if ts.size < 2:
            return float("nan")
        return float(np.median(np.diff(ts)))

    decision_median_dt = _median_dt(matched_decision_ts)
    snapshot_median_dt = _median_dt(matched_snapshot_ts)
    if matched_decision_ts.size:
        decision_first = int(matched_decision_ts[0])
        decision_last = int(matched_decision_ts[-1])
        snapshot_first = int(matched_snapshot_ts[0])
        snapshot_last = int(matched_snapshot_ts[-1])
    else:
        decision_first = decision_last = snapshot_first = snapshot_last = None
    if label:
        print(
            "[snapshot alignment]",
            f"split={label}",
            f"match_rate={match_rate:.6f}",
            f"exact_rate={exact_rate:.6f}",
            f"tolerance_ms={tolerance_ms}",
            f"decision_first={_format_ts(decision_first) if decision_first is not None else 'n/a'}",
            f"decision_last={_format_ts(decision_last) if decision_last is not None else 'n/a'}",
            f"snapshot_first={_format_ts(snapshot_first) if snapshot_first is not None else 'n/a'}",
            f"snapshot_last={_format_ts(snapshot_last) if snapshot_last is not None else 'n/a'}",
            f"decision_median_dt_ms={decision_median_dt:.2f}",
            f"snapshot_median_dt_ms={snapshot_median_dt:.2f}",
        )
    if match_rate < match_rate_target:
        mismatch_idx = np.flatnonzero(~matched)
        sample_count = min(5, mismatch_idx.size)
        samples = [int(decision_ts[i]) for i in mismatch_idx[:sample_count]]
        raise ValueError(
            "Snapshot alignment match rate below target; "
            f"match_rate={match_rate:.6f} target={match_rate_target:.6f} "
            f"matched={matched_decision_ts.size} total={decision_ts.size} "
            f"tolerance_ms={tolerance_ms} "
            f"decision_first={decision_first} decision_last={decision_last} "
            f"snapshot_first={snapshot_first} snapshot_last={snapshot_last} "
            f"decision_median_dt_ms={decision_median_dt:.2f} "
            f"snapshot_median_dt_ms={snapshot_median_dt:.2f} "
            f"samples={samples}"
        )
    return aligned, matched


def _resolve_horizon_indices(meta: dict, targets: Iterable[int]) -> Dict[int, int]:
    horizons = [int(h) for h in meta.get("horizons_ms", [])]
    if not horizons:
        raise ValueError("meta['horizons_ms'] must be non-empty")
    index_map = {h: idx for idx, h in enumerate(horizons)}
    missing = [h for h in targets if h not in index_map]
    if missing:
        raise ValueError(f"Requested horizons not in meta: {missing}")
    return {h: index_map[h] for h in targets}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def join_features(
    decision_ts: np.ndarray,
    y: np.ndarray,
    cmssl_out: Dict[str, np.ndarray],
    snapshots: np.ndarray,
    snapshot_mask: np.ndarray,
    meta: dict,
) -> Dict[str, np.ndarray]:
    ret_pred = cmssl_out["ret_pred"]
    vol_pred = cmssl_out["vol_pred"]
    dir_logits = cmssl_out["dir_logits"]
    p_up = _sigmoid(dir_logits)
    horizons = [int(h) for h in meta.get("horizons_ms", [])]
    if not horizons:
        raise ValueError("meta['horizons_ms'] must be non-empty")
    sorted_horizons = sorted(set(horizons))
    short_h = sorted_horizons[0]
    long_h = sorted_horizons[-1]
    mid_h = sorted_horizons[len(sorted_horizons) // 2]
    horizon_idx = _resolve_horizon_indices(meta, targets=[short_h, mid_h, long_h])
    idx_short = horizon_idx[short_h]
    idx_mid = horizon_idx[mid_h]
    idx_long = horizon_idx[long_h]
    conf = np.abs(p_up - 0.5) * 2.0
    align_all = np.logical_or(
        np.all(p_up >= 0.5, axis=1),
        np.all(p_up <= 0.5, axis=1),
    ).astype(np.float32)
    diff_short_long = p_up[:, idx_short] - p_up[:, idx_long]
    diff_mid_long = p_up[:, idx_mid] - p_up[:, idx_long]
    conf_long = conf[:, idx_long]
    conf_min = np.min(conf, axis=1)
    sigma = sigma_from_vol(vol_pred)
    sigma_bps = 1e4 * sigma
    spread_bps = spread_bps_from_vol_pred(vol_pred[:, idx_short])
    alpha_bps = alpha_from_probs(p_up[:, idx_long], sigma_bps[:, idx_long])

    features = np.concatenate(
        [
            ret_pred,
            vol_pred,
            dir_logits,
            p_up,
            align_all[:, None],
            diff_short_long[:, None],
            diff_mid_long[:, None],
            conf_long[:, None],
            conf_min[:, None],
            snapshots,
        ],
        axis=-1,
    )
    return {
        "ts": decision_ts,
        "features": features.astype(np.float32),
        "y": y.astype(np.float32),
        "spread_bps": spread_bps.astype(np.float32),
        "alpha_bps": alpha_bps.astype(np.float32),
        "snapshot_mask": snapshot_mask.astype(np.bool_),
        "snapshots": snapshots.astype(np.float32),
    }


def build_joined_split(
    out_root: str,
    split: Dict[str, int],
    model,
    meta: dict,
    device: str,
    batch_size: int = 256,
    split_label: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    x_core, x_aux, y, ts = load_split_arrays(out_root, split)
    cmssl_out = run_cmssl_inference(model, meta, x_core, x_aux, batch_size=batch_size, device=device)
    snapshot_ts, snapshots = load_raw_snapshots(out_root, split["week"])
    aligned_snapshots, snapshot_mask = align_snapshots_to_decisions(
        ts,
        snapshot_ts,
        snapshots,
        label=split_label,
    )
    return join_features(ts, y, cmssl_out, aligned_snapshots, snapshot_mask, meta)


def chronological_split(
    data: Dict[str, np.ndarray],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, Dict[str, np.ndarray]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(data["ts"])
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    def _slice(idx: slice) -> Dict[str, np.ndarray]:
        return {key: value[idx] for key, value in data.items()}

    return {
        "train": _slice(idx_train),
        "val": _slice(idx_val),
        "test": _slice(idx_test),
        "bounds": {
            "train": {"start": 0, "end": n_train},
            "val": {"start": n_train, "end": n_train + n_val},
            "test": {"start": n_train + n_val, "end": n},
        },
    }


def persist_split_bounds(out_root: str, bounds: Dict[str, Dict[str, int]], total: int) -> Path:
    out_root = Path(out_root)
    payload = {
        "total": total,
        "bounds": bounds,
    }
    path = out_root / "rl_exec_split_bounds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


@dataclass
class TradingBatch:
    features: np.ndarray
    returns: np.ndarray
    spread_bps: np.ndarray


@dataclass
class MarketMakingBatch:
    features: np.ndarray
    spread_bps: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    alpha_bps: Optional[np.ndarray] = None
    snapshot_mask: Optional[np.ndarray] = None


class MarketMakingEnv:
    def __init__(
        self,
        batch: MarketMakingBatch,
        *,
        maker_rebate_bps: float = 0.0,
        inventory_penalty: float = 0.0,
        max_inventory: Optional[float] = None,
        fill_size: float = 1.0,
        fill_tolerance: float = 1e-6,
    ):
        self.features = batch.features
        self.spread_bps = batch.spread_bps
        self.best_bid = batch.best_bid
        self.best_ask = batch.best_ask
        self.alpha_bps = batch.alpha_bps if batch.alpha_bps is not None else np.zeros_like(self.spread_bps)
        self.snapshot_mask = (
            batch.snapshot_mask if batch.snapshot_mask is not None else np.ones_like(self.best_bid, dtype=bool)
        )
        self.maker_rebate_bps = maker_rebate_bps
        self.inventory_penalty = inventory_penalty
        self.max_inventory = max_inventory
        self.fill_size = fill_size
        self.fill_tolerance = fill_tolerance

        self.n = len(self.spread_bps)
        self.idx = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.total_reward = 0.0
        self.prev_equity = 0.0

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.total_reward = 0.0
        mid = self._mid_price(self.idx)
        self.prev_equity = self.cash + self.inventory * mid
        return self._build_observation(self.idx)

    def _mid_price(self, idx: int) -> float:
        return float((self.best_bid[idx] + self.best_ask[idx]) / 2.0)

    def _build_observation(self, idx: int) -> np.ndarray:
        mid = self._mid_price(idx)
        spread = float(self.best_ask[idx] - self.best_bid[idx])
        extra = np.array([self.inventory, self.cash, mid, spread], dtype=np.float32)
        return np.concatenate([self.features[idx].astype(np.float32), extra], axis=0)

    def _parse_action(self, action: Any) -> Tuple[float, float]:
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            return float(action[0]), float(action[1])
        if np.isscalar(action):
            return float(action), float(action)
        raise ValueError("Action must be a scalar or (bid_delta_bps, ask_delta_bps).")

    def _baseline_quotes(self, idx: int) -> Tuple[float, float, float]:
        mid = self._mid_price(idx)
        spread_bps = float(self.spread_bps[idx])
        alpha_bps = float(self.alpha_bps[idx])
        baseline_half_spread = half_spread(mid, spread_bps)
        baseline_skew = skew(mid, alpha_bps)
        bid = mid - baseline_half_spread + baseline_skew
        ask = mid + baseline_half_spread + baseline_skew
        return bid, ask, mid

    def _apply_deltas(self, bid: float, ask: float, mid: float, action: Any) -> Tuple[float, float]:
        bid_delta_bps, ask_delta_bps = self._parse_action(action)
        bid += mid * bid_delta_bps * 1e-4
        ask += mid * ask_delta_bps * 1e-4
        return bid, ask

    def _enforce_passive(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        bid = min(bid, best_bid)
        ask = max(ask, best_ask)
        if bid >= ask:
            bid = best_bid
            ask = best_ask
        return bid, ask

    def _apply_fills(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        if not bool(self.snapshot_mask[idx]):
            return 0.0, 0.0
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        buy_fill = 0.0
        sell_fill = 0.0
        if bid >= best_bid - self.fill_tolerance:
            buy_fill = self.fill_size
            self.cash -= bid * buy_fill
            self.inventory += buy_fill
        if ask <= best_ask + self.fill_tolerance:
            sell_fill = self.fill_size
            self.cash += ask * sell_fill
            self.inventory -= sell_fill
        return buy_fill, sell_fill

    def _compute_penalty(self) -> float:
        penalty = self.inventory_penalty * abs(self.inventory)
        if self.max_inventory is not None and abs(self.inventory) > self.max_inventory:
            penalty += self.inventory_penalty * (abs(self.inventory) - self.max_inventory)
        return penalty

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        bid, ask, mid = self._baseline_quotes(self.idx)
        bid, ask = self._apply_deltas(bid, ask, mid, action)
        bid, ask = self._enforce_passive(bid, ask, self.idx)
        buy_fill, sell_fill = self._apply_fills(bid, ask, self.idx)

        equity = self.cash + self.inventory * mid
        delta_equity = equity - self.prev_equity
        rebate_notional = buy_fill * bid + sell_fill * ask
        rebate = rebate_notional * self.maker_rebate_bps * 1e-4
        penalty = self._compute_penalty()
        reward = delta_equity + rebate - penalty

        self.prev_equity = equity
        self.total_reward += reward
        self.idx += 1
        done = self.idx >= self.n
        next_obs = self._build_observation(self.idx - 1 if done else self.idx)
        info = {
            "reward": float(reward),
            "total_reward": float(self.total_reward),
            "cash": float(self.cash),
            "inventory": float(self.inventory),
            "equity": float(equity),
            "delta_equity": float(delta_equity),
            "rebate": float(rebate),
            "penalty": float(penalty),
            "bid": float(bid),
            "ask": float(ask),
            "buy_fill": float(buy_fill),
            "sell_fill": float(sell_fill),
        }
        return next_obs, float(reward), done, info


class TradingEnv:
    def __init__(self, batch: TradingBatch):
        self.features = batch.features
        self.returns = batch.returns
        self.spread_bps = batch.spread_bps
        self.n = len(self.returns)
        self.idx = 0
        self.position = 0
        self.total_reward = 0.0

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.position = 0
        self.total_reward = 0.0
        return self.features[self.idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        action_map = {0: -1, 1: 0, 2: 1}
        action_sign = action_map.get(action, 0)
        ret = float(self.returns[self.idx])
        spread_cost = float(self.spread_bps[self.idx]) * 1e-4
        turnover = abs(action_sign - self.position)
        reward = action_sign * ret - turnover * spread_cost
        self.total_reward += reward
        self.position = action_sign
        self.idx += 1
        done = self.idx >= self.n
        next_obs = self.features[self.idx - 1] if done else self.features[self.idx]
        info = {
            "reward": reward,
            "total_reward": self.total_reward,
        }
        return next_obs, reward, done, info


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        policy_hidden: Iterable[int] = (128, 128),
        value_hidden: Iterable[int] = (128, 128),
        action_dim: int = 3,
    ):
        super().__init__()
        self.policy_net = MLP(input_dim, policy_hidden, action_dim)
        self.value_net = MLP(input_dim, value_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)
        return logits, value


class MarketPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (128, 128)):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        policy_hidden: Iterable[int] = (128, 128),
        value_hidden: Iterable[int] = (128, 128),
        action_dim: int = 2,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.policy_net = MLP(input_dim, policy_hidden, action_dim)
        self.value_net = MLP(input_dim, value_hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)
        log_std = torch.clamp(self.log_std, min=-6.0, max=2.0)
        return mean, log_std, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    policy_hidden: Tuple[int, ...] = (128, 128)
    value_hidden: Tuple[int, ...] = (128, 128)
    val_every: int = 1
    max_drawdown_guard: Optional[float] = None


def collect_rollout(env: TradingEnv, model: PolicyValueNet, device: str) -> Dict[str, torch.Tensor]:
    obs_list = []
    action_list = []
    logp_list = []
    value_list = []
    reward_list = []
    done_list = []

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device)
        logits, value = model(obs_t.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, done, _info = env.step(int(action.item()))

        obs_list.append(obs_t)
        action_list.append(action)
        logp_list.append(logp)
        value_list.append(value.squeeze(0))
        reward_list.append(torch.tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(done, dtype=torch.float32, device=device))
        obs = next_obs

    return {
        "obs": torch.stack(obs_list),
        "actions": torch.stack(action_list),
        "logp": torch.stack(logp_list),
        "values": torch.stack(value_list),
        "rewards": torch.stack(reward_list),
        "dones": torch.stack(done_list),
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: PPOConfig,
    device: str,
):
    obs = rollout["obs"].to(device)
    actions = rollout["actions"].to(device)
    old_logp = rollout["logp"].detach().to(device)
    values = rollout["values"].detach().to(device)
    rewards = rollout["rewards"].to(device)
    dones = rollout["dones"].to(device)

    advantages, returns = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = torch.arange(n)
    for _ in range(config.update_epochs):
        perm = indices[torch.randperm(n)]
        for start in range(0, n, config.batch_size):
            mb_idx = perm[start:start + config.batch_size]
            logits, value = model(obs[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[mb_idx])
            ratio = torch.exp(logp - old_logp[mb_idx])
            clip_adv = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[mb_idx]
            policy_loss = -(torch.min(ratio * advantages[mb_idx], clip_adv)).mean()
            value_loss = nn.functional.mse_loss(value, returns[mb_idx])
            entropy_loss = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def collect_market_rollout(
    env: MarketMakingEnv,
    model: MarketPolicyValueNet,
    device: str,
    delta_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    obs_list = []
    action_list = []
    logp_list = []
    value_list = []
    reward_list = []
    done_list = []

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device)
        mean, log_std, value = model(obs_t.unsqueeze(0))
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        scaled_action = (action.squeeze(0) * delta_scale).cpu().numpy()
        next_obs, reward, done, _info = env.step(scaled_action)

        obs_list.append(obs_t)
        action_list.append(action.squeeze(0))
        logp_list.append(logp.squeeze(0))
        value_list.append(value.squeeze(0))
        reward_list.append(torch.tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(done, dtype=torch.float32, device=device))
        obs = next_obs

    return {
        "obs": torch.stack(obs_list),
        "actions": torch.stack(action_list),
        "logp": torch.stack(logp_list),
        "values": torch.stack(value_list),
        "rewards": torch.stack(reward_list),
        "dones": torch.stack(done_list),
    }


def ppo_update_market(
    model: MarketPolicyValueNet,
    optimizer: optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: PPOConfig,
    device: str,
):
    obs = rollout["obs"].to(device)
    actions = rollout["actions"].to(device)
    old_logp = rollout["logp"].detach().to(device)
    values = rollout["values"].detach().to(device)
    rewards = rollout["rewards"].to(device)
    dones = rollout["dones"].to(device)

    advantages, returns = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = torch.arange(n)
    for _ in range(config.update_epochs):
        perm = indices[torch.randperm(n)]
        for start in range(0, n, config.batch_size):
            mb_idx = perm[start:start + config.batch_size]
            mean, log_std, value = model(obs[mb_idx])
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(actions[mb_idx]).sum(dim=-1)
            ratio = torch.exp(logp - old_logp[mb_idx])
            clip_adv = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[mb_idx]
            policy_loss = -(torch.min(ratio * advantages[mb_idx], clip_adv)).mean()
            value_loss = nn.functional.mse_loss(value, returns[mb_idx])
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_sharpe(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(mean / std * np.sqrt(returns.size))


def compute_max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(drawdown.max(initial=0.0))


def train_ppo(
    train_env: TradingEnv,
    val_env: TradingEnv,
    input_dim: int,
    device: str = "cuda",
    epochs: int = 10,
    config: Optional[PPOConfig] = None,
    ckpt_path: Optional[Path] = None,
) -> Tuple[PolicyValueNet, Dict[str, float]]:
    config = config or PPOConfig()
    model = PolicyValueNet(
        input_dim,
        policy_hidden=config.policy_hidden,
        value_hidden=config.value_hidden,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_report: Dict[str, float] = {"sharpe": -np.inf}

    for epoch in range(epochs):
        rollout = collect_rollout(train_env, model, device)
        ppo_update(model, optimizer, rollout, config, device)
        if (epoch + 1) % config.val_every == 0:
            report = evaluate_policy(val_env, model, device=device)
            sharpe = report["sharpe"]
            drawdown = report["max_drawdown"]
            guard = config.max_drawdown_guard
            if sharpe > best_report.get("sharpe", -np.inf) and (guard is None or drawdown <= guard):
                best_report = report
                if ckpt_path:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "config": config.__dict__,
                            "val_report": report,
                        },
                        ckpt_path,
                    )
    return model, best_report


def evaluate_policy(env: TradingEnv, model: PolicyValueNet, device: str = "cuda") -> Dict[str, float]:
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device)
        logits, _value = model(obs_t.unsqueeze(0))
        action = torch.argmax(logits, dim=-1)
        obs, reward, done, _info = env.step(int(action.item()))
        rewards.append(reward)
    rewards_arr = np.array(rewards, dtype=np.float32)
    return {
        "total_reward": float(rewards_arr.sum()),
        "mean_reward": float(rewards_arr.mean()) if rewards_arr.size else 0.0,
        "sharpe": compute_sharpe(rewards_arr),
        "max_drawdown": compute_max_drawdown(rewards_arr),
    }


def evaluate_market_policy(
    env: MarketMakingEnv,
    policy: MarketPolicyNet,
    device: str = "cuda",
    delta_scale: float = 1.0,
) -> Dict[str, Any]:
    def _policy_fn(obs: np.ndarray) -> Tuple[float, float]:
        obs_t = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            deltas = policy(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        return float(deltas[0] * delta_scale), float(deltas[1] * delta_scale)

    return evaluate_market_making(env, _policy_fn)


def train_market_ppo(
    train_env: MarketMakingEnv,
    val_env: MarketMakingEnv,
    input_dim: int,
    device: str = "cuda",
    epochs: int = 10,
    config: Optional[PPOConfig] = None,
    ckpt_path: Optional[Path] = None,
    delta_scale: float = 1.0,
) -> Tuple[MarketPolicyValueNet, Dict[str, Any]]:
    config = config or PPOConfig()
    model = MarketPolicyValueNet(
        input_dim,
        policy_hidden=config.policy_hidden,
        value_hidden=config.value_hidden,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_report: Dict[str, Any] = {"sharpe": -np.inf}

    for epoch in range(epochs):
        rollout = collect_market_rollout(train_env, model, device, delta_scale=delta_scale)
        ppo_update_market(model, optimizer, rollout, config, device)
        if (epoch + 1) % config.val_every == 0:
            eval_policy = MarketPolicyNet(input_dim, hidden_dims=config.policy_hidden).to(device)
            eval_policy.load_state_dict(model.policy_net.state_dict(), strict=True)
            report = evaluate_market_policy(val_env, eval_policy, device=device, delta_scale=delta_scale)
            sharpe = report["sharpe"]
            drawdown = report["max_drawdown"]
            guard = config.max_drawdown_guard
            if sharpe > best_report.get("sharpe", -np.inf) and (guard is None or drawdown <= guard):
                best_report = report
                if ckpt_path:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "mean_state_dict": model.policy_net.state_dict(),
                            "value_state_dict": model.value_net.state_dict(),
                            "log_std": model.log_std.detach().cpu().numpy(),
                            "hidden_dims": tuple(config.policy_hidden),
                            "config": config.__dict__,
                            "val_report": report,
                        },
                        ckpt_path,
                    )
    return model, best_report


def report_cmssl_metrics(y_true: np.ndarray, cmssl_out: Dict[str, np.ndarray]) -> Dict[str, float]:
    num_h = y_true.shape[1] // 2
    y_ret = y_true[:, :num_h]
    y_vol = y_true[:, num_h:]
    ret_pred = cmssl_out["ret_pred"]
    vol_pred = cmssl_out["vol_pred"]
    ret_mae = float(np.mean(np.abs(ret_pred - y_ret)))
    vol_mae = float(np.mean(np.abs(vol_pred - y_vol)))
    return {
        "ret_mae": ret_mae,
        "vol_mae": vol_mae,
    }


def _fill_forward(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = arr.copy()
    isnan = np.isnan(out)
    if not np.any(isnan):
        return out
    idx = np.where(~isnan, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    if np.isnan(out[0]):
        first_valid = np.flatnonzero(~isnan)
        if first_valid.size:
            out[: first_valid[0]] = out[first_valid[0]]
    return out


def build_market_batch(split: Dict[str, np.ndarray]) -> MarketMakingBatch:
    snapshots = split.get("snapshots")
    if snapshots is None:
        raise ValueError("snapshots missing from split data; ensure join_features stores snapshots.")
    if snapshots.ndim != 2 or snapshots.shape[1] < 2:
        raise ValueError(f"Expected snapshots with >=2 columns, got {snapshots.shape}")
    best_bid = snapshots[:, 0].astype(np.float32)
    best_ask = snapshots[:, 1].astype(np.float32)
    best_bid = _fill_forward(best_bid)
    best_ask = _fill_forward(best_ask)
    return MarketMakingBatch(
        features=split["features"],
        spread_bps=split["spread_bps"],
        best_bid=best_bid,
        best_ask=best_ask,
        alpha_bps=split.get("alpha_bps"),
        snapshot_mask=split.get("snapshot_mask"),
    )


def _inventory_distribution(inventory: np.ndarray) -> Dict[str, float]:
    if inventory.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "min": float(np.min(inventory)),
        "max": float(np.max(inventory)),
        "mean": float(np.mean(inventory)),
        "std": float(np.std(inventory)),
        "p05": float(np.quantile(inventory, 0.05)),
        "p50": float(np.quantile(inventory, 0.50)),
        "p95": float(np.quantile(inventory, 0.95)),
    }


def evaluate_market_making(
    env: MarketMakingEnv,
    policy_fn,
) -> Dict[str, Any]:
    obs = env.reset()
    equity_curve: List[float] = []
    inventory_curve: List[float] = []
    turnover = 0.0
    fill_count = 0
    fill_opps = 0
    initial_equity = env.prev_equity

    done = False
    while not done:
        action = policy_fn(obs)
        obs, _reward, done, info = env.step(action)
        equity_curve.append(info["equity"])
        inventory_curve.append(info["inventory"])
        turnover += abs(info["buy_fill"]) + abs(info["sell_fill"])
        fill_count += int(info["buy_fill"] > 0.0) + int(info["sell_fill"] > 0.0)
        if env.snapshot_mask[env.idx - 1]:
            fill_opps += 2

    equity_arr = np.array(equity_curve, dtype=np.float32)
    returns = np.diff(np.concatenate([[initial_equity], equity_arr]))
    sharpe = compute_sharpe(returns)
    max_drawdown = compute_max_drawdown(returns)
    fill_rate = float(fill_count / fill_opps) if fill_opps > 0 else 0.0
    inventory_arr = np.array(inventory_curve, dtype=np.float32)

    return {
        "equity_curve": equity_arr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": float(turnover),
        "fill_rate": fill_rate,
        "inventory_distribution": _inventory_distribution(inventory_arr),
    }


def _format_mm_summary(label: str, metrics: Dict[str, Any]) -> str:
    inv = metrics["inventory_distribution"]
    return (
        f"{label}: sharpe={metrics['sharpe']:.4f} "
        f"max_dd={metrics['max_drawdown']:.4f} "
        f"turnover={metrics['turnover']:.4f} "
        f"fill_rate={metrics['fill_rate']:.4f} "
        f"inv[min={inv['min']:.2f}, p50={inv['p50']:.2f}, max={inv['max']:.2f}]"
    )


def load_market_policy(
    input_dim: int,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
) -> Tuple[Optional[MarketPolicyNet], Optional[np.ndarray]]:
    if not ckpt_path:
        return None, None
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Market policy checkpoint not found: {ckpt_path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "mean_state_dict" in ckpt:
        state = ckpt["mean_state_dict"]
        log_std = ckpt.get("log_std")
    else:
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        log_std = None
    hidden_dims = ckpt.get("hidden_dims") if isinstance(ckpt, dict) else None
    if hidden_dims is None:
        hidden_dims = tuple(
            int(x) for x in os.environ.get("BYBIT_MM_POLICY_HIDDEN", "128,128").split(",")
        )
    model = MarketPolicyNet(input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, None if log_std is None else np.asarray(log_std, dtype=np.float32)


def run_pipeline(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    ppo_epochs: int = 10,
) -> Dict[str, Any]:
    meta = load_global_meta(Path(out_root))
    splits = build_two_week_time_splits(out_root)

    report_pretrain_diagnostics(out_root, splits)

    model, _meta = load_cmssl(out_root, ckpt_path, device=device)

    joined_test = build_joined_split(
        out_root,
        splits["test"],
        model,
        meta,
        device,
        split_label="test",
    )

    num_h = len(meta.get("horizons_ms", []))
    cmssl_report = report_cmssl_metrics(
        joined_test["y"],
        {
            "ret_pred": joined_test["features"][:, :num_h],
            "vol_pred": joined_test["features"][:, num_h:2 * num_h],
            "dir_logits": joined_test["features"][:, 2 * num_h:3 * num_h],
        },
    )

    splits_rl = chronological_split(joined_test, ratios=(0.6, 0.2, 0.2))
    persist_split_bounds(out_root, splits_rl["bounds"], total=len(joined_test["ts"]))

    def _to_env(split: Dict[str, np.ndarray]) -> TradingEnv:
        returns = split["y"][:, 0]
        batch = TradingBatch(
            features=split["features"],
            returns=returns,
            spread_bps=split["spread_bps"],
        )
        return TradingEnv(batch)

    train_env = _to_env(splits_rl["train"])
    val_env = _to_env(splits_rl["val"])
    test_env = _to_env(splits_rl["test"])

    input_dim = train_env.features.shape[-1]
    ppo_config = PPOConfig(
        lr=float(os.environ.get("BYBIT_PPO_LR", "3e-4")),
        update_epochs=int(os.environ.get("BYBIT_PPO_UPDATE_EPOCHS", "4")),
        batch_size=int(os.environ.get("BYBIT_PPO_BATCH_SIZE", "256")),
        clip_ratio=float(os.environ.get("BYBIT_PPO_CLIP_RATIO", "0.2")),
        gamma=float(os.environ.get("BYBIT_PPO_GAMMA", "0.99")),
        gae_lambda=float(os.environ.get("BYBIT_PPO_GAE_LAMBDA", "0.95")),
        entropy_coef=float(os.environ.get("BYBIT_PPO_ENTROPY_COEF", "0.01")),
        value_coef=float(os.environ.get("BYBIT_PPO_VALUE_COEF", "0.5")),
        policy_hidden=tuple(int(x) for x in os.environ.get("BYBIT_PPO_POLICY_HIDDEN", "128,128").split(",")),
        value_hidden=tuple(int(x) for x in os.environ.get("BYBIT_PPO_VALUE_HIDDEN", "128,128").split(",")),
        val_every=int(os.environ.get("BYBIT_PPO_VAL_EVERY", "1")),
        max_drawdown_guard=float(os.environ.get("BYBIT_PPO_MAX_DRAWDOWN", "nan")),
    )
    if np.isnan(ppo_config.max_drawdown_guard):
        ppo_config.max_drawdown_guard = None
    best_ckpt = Path(os.environ.get("BYBIT_PPO_BEST_CKPT", Path(out_root) / "ppo_best.pt"))
    ppo_model, best_val = train_ppo(
        train_env,
        val_env,
        input_dim,
        device=device,
        epochs=ppo_epochs,
        config=ppo_config,
        ckpt_path=best_ckpt,
    )

    val_report = evaluate_policy(val_env, ppo_model, device=device)
    test_report = evaluate_policy(test_env, ppo_model, device=device)

    mm_train_batch = build_market_batch(splits_rl["train"])
    mm_val_batch = build_market_batch(splits_rl["val"])
    mm_test_batch = build_market_batch(splits_rl["test"])
    mm_obs_dim = mm_train_batch.features.shape[-1] + 4
    maker_rebate_bps = float(os.environ.get("BYBIT_MM_MAKER_REBATE_BPS", "0.0"))
    inventory_penalty = float(os.environ.get("BYBIT_MM_INVENTORY_PENALTY", "0.0"))
    max_inventory_str = os.environ.get("BYBIT_MM_MAX_INVENTORY", "").strip()
    max_inventory = float(max_inventory_str) if max_inventory_str else None
    fill_size = float(os.environ.get("BYBIT_MM_FILL_SIZE", "1.0"))
    fill_tolerance = float(os.environ.get("BYBIT_MM_FILL_TOLERANCE", "1e-6"))
    delta_scale = float(os.environ.get("BYBIT_MM_DELTA_SCALE", "1.0"))

    mm_train_env = MarketMakingEnv(
        mm_train_batch,
        maker_rebate_bps=maker_rebate_bps,
        inventory_penalty=inventory_penalty,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
    )
    mm_val_env = MarketMakingEnv(
        mm_val_batch,
        maker_rebate_bps=maker_rebate_bps,
        inventory_penalty=inventory_penalty,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
    )
    mm_test_env = MarketMakingEnv(
        mm_test_batch,
        maker_rebate_bps=maker_rebate_bps,
        inventory_penalty=inventory_penalty,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
    )

    mm_ppo_config = PPOConfig(
        lr=float(os.environ.get("BYBIT_MM_PPO_LR", "3e-4")),
        update_epochs=int(os.environ.get("BYBIT_MM_PPO_UPDATE_EPOCHS", "4")),
        batch_size=int(os.environ.get("BYBIT_MM_PPO_BATCH_SIZE", "256")),
        clip_ratio=float(os.environ.get("BYBIT_MM_PPO_CLIP_RATIO", "0.2")),
        gamma=float(os.environ.get("BYBIT_MM_PPO_GAMMA", "0.99")),
        gae_lambda=float(os.environ.get("BYBIT_MM_PPO_GAE_LAMBDA", "0.95")),
        entropy_coef=float(os.environ.get("BYBIT_MM_PPO_ENTROPY_COEF", "0.01")),
        value_coef=float(os.environ.get("BYBIT_MM_PPO_VALUE_COEF", "0.5")),
        policy_hidden=tuple(int(x) for x in os.environ.get("BYBIT_MM_PPO_POLICY_HIDDEN", "128,128").split(",")),
        value_hidden=tuple(int(x) for x in os.environ.get("BYBIT_MM_PPO_VALUE_HIDDEN", "128,128").split(",")),
        val_every=int(os.environ.get("BYBIT_MM_PPO_VAL_EVERY", "1")),
        max_drawdown_guard=float(os.environ.get("BYBIT_MM_PPO_MAX_DRAWDOWN", "nan")),
    )
    if np.isnan(mm_ppo_config.max_drawdown_guard):
        mm_ppo_config.max_drawdown_guard = None
    mm_best_ckpt = Path(os.environ.get("BYBIT_MM_PPO_BEST_CKPT", Path(out_root) / "mm_ppo_best.pt"))
    train_market_ppo(
        mm_train_env,
        mm_val_env,
        mm_obs_dim,
        device=device,
        epochs=int(os.environ.get("BYBIT_MM_PPO_EPOCHS", str(ppo_epochs))),
        config=mm_ppo_config,
        ckpt_path=mm_best_ckpt,
        delta_scale=delta_scale,
    )

    baseline_env = MarketMakingEnv(
        mm_test_batch,
        maker_rebate_bps=maker_rebate_bps,
        inventory_penalty=inventory_penalty,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
    )
    baseline_metrics = evaluate_market_making(baseline_env, lambda _obs: (0.0, 0.0))

    mm_policy_path = os.environ.get("BYBIT_MM_RL_CKPT", "").strip() or str(mm_best_ckpt)
    mm_policy, _log_std = load_market_policy(mm_obs_dim, device=device, ckpt_path=mm_policy_path or None)
    if mm_policy is None:
        print("[mm eval] no BYBIT_MM_RL_CKPT provided; using baseline deltas for RL run.")
        rl_policy_fn = lambda _obs: (0.0, 0.0)
        rl_policy_loaded = False
    else:
        rl_policy_loaded = True

        def rl_policy_fn(obs: np.ndarray) -> Tuple[float, float]:
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                deltas = mm_policy(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            return float(deltas[0] * delta_scale), float(deltas[1] * delta_scale)

    rl_env = MarketMakingEnv(
        mm_test_batch,
        maker_rebate_bps=maker_rebate_bps,
        inventory_penalty=inventory_penalty,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
    )
    rl_metrics = evaluate_market_making(rl_env, rl_policy_fn)

    print("[mm eval]", _format_mm_summary("baseline", baseline_metrics))
    print("[mm eval]", _format_mm_summary("baseline+rl", rl_metrics))

    return {
        "cmssl_test": cmssl_report,
        "ppo_val": val_report,
        "ppo_val_best": best_val,
        "ppo_test": test_report,
        "mm_baseline": baseline_metrics,
        "mm_rl": rl_metrics,
        "mm_rl_policy_loaded": {"loaded": rl_policy_loaded},
    }


if __name__ == "__main__":
    out_root = os.environ.get("BYBIT_OUT_ROOT", "").strip()
    ckpt_path = os.environ.get("BYBIT_CMSSL_CKPT", "").strip()
    device = os.environ.get("BYBIT_DEVICE", "cuda")
    ppo_epochs = int(os.environ.get("BYBIT_PPO_EPOCHS", "10"))

    if not out_root or not ckpt_path:
        raise SystemExit("Set BYBIT_OUT_ROOT and BYBIT_CMSSL_CKPT before running.")

    report = run_pipeline(out_root, ckpt_path, device=device, ppo_epochs=ppo_epochs)
    print("[cmssl test]", report["cmssl_test"])
    print("[ppo val]", report["ppo_val"])
    print("[ppo val best]", report["ppo_val_best"])
    print("[ppo test]", report["ppo_test"])
