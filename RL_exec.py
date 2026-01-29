import json
import os
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


def spread_bps_from_vol_pred(vol_pred, spread_mult: float = 1.0):
    """
    Convert model vol predictions into a spread size in basis points.

    vol_pred is trained against y_logvol (log volatility), so we recover
    sigma by exponentiating the log-vol and then scale to bps.
    If the model ever switches to predicting log-variance, use
    sigma = exp(0.5 * logvar) instead.
    """
    sigma = np.exp(vol_pred)
    sigma_bps = 1e4 * sigma
    return spread_mult * sigma_bps


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


def align_snapshots_to_decisions(
    decision_ts: np.ndarray,
    snapshot_ts: np.ndarray,
    snapshots: np.ndarray,
    tolerance_ms: int = 50,
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
    insert_idx = np.searchsorted(snapshot_ts, decision_ts, side="left")
    right_idx = np.clip(insert_idx, 0, len(snapshot_ts) - 1)
    left_idx = np.clip(insert_idx - 1, 0, len(snapshot_ts) - 1)
    left_diff = np.abs(snapshot_ts[left_idx] - decision_ts)
    right_diff = np.abs(snapshot_ts[right_idx] - decision_ts)
    choose_right = right_diff < left_diff
    idx = np.where(choose_right, right_idx, left_idx)
    aligned = snapshots[idx]
    delta = snapshot_ts[idx] - decision_ts
    abs_delta = np.abs(delta)
    mask = abs_delta <= tolerance_ms
    match_rate = float(np.mean(mask)) if mask.size else 0.0
    if not np.all(mask):
        mismatch_delta = delta[~mask]
        mismatch_abs = abs_delta[~mask]
        sample_count = min(5, mismatch_delta.size)
        sample_indices = np.flatnonzero(~mask)[:sample_count]
        samples = [
            (int(decision_ts[i]), int(snapshot_ts[idx[i]]), int(delta[i]))
            for i in sample_indices
        ]
        print(
            "[snapshot alignment] mismatches:",
            f"count={mismatch_delta.size}",
            f"match_rate={match_rate:.6f}",
            f"min_dt={mismatch_delta.min():.1f}ms",
            f"max_dt={mismatch_delta.max():.1f}ms",
            f"median_dt={float(np.median(mismatch_delta)):.1f}ms",
            f"min_abs_dt={float(mismatch_abs.min()):.1f}ms",
            f"max_abs_dt={float(mismatch_abs.max()):.1f}ms",
            f"median_abs_dt={float(np.median(mismatch_abs)):.1f}ms",
            f"samples={samples}",
        )
    assert match_rate >= 0.995, (
        f"Snapshot match rate {match_rate:.6f} below 0.995 "
        f"(tolerance={tolerance_ms}ms)."
    )
    aligned = aligned.astype(np.float32)
    aligned[~mask] = np.nan
    return aligned, mask


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
    horizon_idx = _resolve_horizon_indices(meta, targets=[250, 500, 1000])
    idx_250 = horizon_idx[250]
    idx_500 = horizon_idx[500]
    idx_1000 = horizon_idx[1000]
    conf = np.abs(p_up - 0.5) * 2.0
    align_all = np.logical_or(
        np.all(p_up >= 0.5, axis=1),
        np.all(p_up <= 0.5, axis=1),
    ).astype(np.float32)
    diff_250_1000 = p_up[:, idx_250] - p_up[:, idx_1000]
    diff_500_1000 = p_up[:, idx_500] - p_up[:, idx_1000]
    conf_1000 = conf[:, idx_1000]
    conf_min = np.min(conf, axis=1)
    spread_bps = spread_bps_from_vol_pred(vol_pred[:, 0])

    features = np.concatenate(
        [
            ret_pred,
            vol_pred,
            dir_logits,
            p_up,
            align_all[:, None],
            diff_250_1000[:, None],
            diff_500_1000[:, None],
            conf_1000[:, None],
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
        "snapshot_mask": snapshot_mask.astype(np.bool_),
    }


def build_joined_split(
    out_root: str,
    split: Dict[str, int],
    model,
    meta: dict,
    device: str,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    x_core, x_aux, y, ts = load_split_arrays(out_root, split)
    cmssl_out = run_cmssl_inference(model, meta, x_core, x_aux, batch_size=batch_size, device=device)
    snapshot_ts, snapshots = load_raw_snapshots(out_root, split["week"])
    aligned_snapshots, snapshot_mask = align_snapshots_to_decisions(ts, snapshot_ts, snapshots)
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


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 256


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
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_ppo(env: TradingEnv, input_dim: int, device: str = "cuda", epochs: int = 10) -> PolicyValueNet:
    model = PolicyValueNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig()

    for _ in range(epochs):
        rollout = collect_rollout(env, model, device)
        ppo_update(model, optimizer, rollout, config, device)
    return model


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
    }


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


def run_pipeline(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    ppo_epochs: int = 10,
) -> Dict[str, Dict[str, float]]:
    meta = load_global_meta(Path(out_root))
    splits = build_two_week_time_splits(out_root)

    model, _meta = load_cmssl(out_root, ckpt_path, device=device)

    joined_train = build_joined_split(out_root, splits["train"], model, meta, device)
    joined_val = build_joined_split(out_root, splits["val"], model, meta, device)
    joined_test = build_joined_split(out_root, splits["test"], model, meta, device)

    num_h = len(meta.get("horizons_ms", []))
    cmssl_report = report_cmssl_metrics(
        joined_test["y"],
        {
            "ret_pred": joined_test["features"][:, :num_h],
            "vol_pred": joined_test["features"][:, num_h:2 * num_h],
            "dir_logits": joined_test["features"][:, 2 * num_h:3 * num_h],
        },
    )

    joined = {
        key: np.concatenate([joined_train[key], joined_val[key], joined_test[key]], axis=0)
        for key in joined_train.keys()
    }
    order = np.argsort(joined["ts"])
    joined = {key: value[order] for key, value in joined.items()}

    splits_rl = chronological_split(joined, ratios=(0.6, 0.2, 0.2))
    persist_split_bounds(out_root, splits_rl["bounds"], total=len(joined["ts"]))

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
    ppo_model = train_ppo(train_env, input_dim, device=device, epochs=ppo_epochs)

    val_report = evaluate_policy(val_env, ppo_model, device=device)
    test_report = evaluate_policy(test_env, ppo_model, device=device)

    return {
        "cmssl_test": cmssl_report,
        "ppo_val": val_report,
        "ppo_test": test_report,
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
    print("[ppo test]", report["ppo_test"])
