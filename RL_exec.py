import json
import os
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from CMSSL17 import SAMBA, ModelArgs, DMODEL, MAMBA_LAYERS, LOOKBACK, FeatureEngine, _open_text
from offline_tokens import iter_week_chunks, load_global_meta

RAW_SNAPSHOT_PATHS = [
    Path(p)
    for p in os.environ.get("RAW_SNAPSHOT_PATHS", "").split(",")
    if p
]
RAW_SNAPSHOT_EXPECTED_STEP_MS = 100
RAW_SNAPSHOT_TOLERANCE_MS = 20
RAW_SNAPSHOT_MAX_IRREGULAR_FRAC = 0.05
RAW_SNAPSHOT_FEATURE_COLUMNS = [
    "best_bid",
    "best_ask",
    "mid",
    "spread_bps",
    "mid_ret_1",
    "vol_short",
    "vol_long",
]
RAW_OB_DIR = os.environ.get("BYBIT_OB_DIR", "").strip()
ALLOW_TS_RECONSTRUCT = os.environ.get("BYBIT_ALLOW_TS_RECONSTRUCT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
FEATURE_EXTRA_DIM = 5
SHORT_VOL_WINDOW = 50
LONG_VOL_WINDOW = 200
DEFAULT_MM_HORIZONS_MS = [250, 500, 1000]
DEFAULT_MM_VOL_HORIZON_MS = 1000
DEFAULT_MM_S_MIN_BPS = 0.0
DEFAULT_MM_K_SIGMA = 0.5
DEFAULT_MM_K_INV = 0.0
DEFAULT_MM_K_ALPHA = 1.0
DEFAULT_MM_SPREAD_FLOOR_BPS = 0.0
DEFAULT_MM_SPREAD_CAP_BPS = 10_000.0
DEFAULT_MM_INV_REF_NOTIONAL = 1.0
DEFAULT_MM_P250_WEIGHT = 0.0
DEFAULT_MM_P500_WEIGHT = 0.0
DEFAULT_MM_P1000_WEIGHT = 1.0
# PPO training epochs environment variable (used across entrypoint/config helpers).
PPO_EPOCHS_ENV = "BYBIT_MM_PPO_EPOCHS"
# Scaling factors for observation features.
# Notional and cash scales are denominated in quote currency.
# Time-since-fill scale is in environment steps (1 step per snapshot).
DEFAULT_MM_NOTIONAL_SCALE = 1e4
DEFAULT_MM_CASH_SCALE = 1e4
DEFAULT_MM_TIME_SINCE_FILL_SCALE = 1000.0
DEFAULT_MM_TAKER_FEE_BPS = 1.7
DEFAULT_MM_TAKER_THRESHOLD = 0.25
SNAPSHOT_ALIGN_BOUNDS_TOLERANCE_MS = int(
    os.environ.get("SNAPSHOT_ALIGN_BOUNDS_TOLERANCE_MS", "3000")
)


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


def sigma_from_vol(log_vol: np.ndarray) -> np.ndarray:
    """Recover volatility from log-vol predictions."""
    return np.exp(log_vol)


def alpha_from_probs(p_up: np.ndarray, sigma_bps: np.ndarray) -> np.ndarray:
    """Convert directional probabilities into a signed alpha in bps."""
    return (p_up - 0.5) * 2.0 * sigma_bps


def bps_to_px(mid: float, bps: float) -> float:
    return mid * bps * 1e-4


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_int_list(name: str, default: List[int]) -> List[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [int(item) for item in raw.split(",") if item.strip()]


def _resolve_ppo_epochs(default: int) -> int:
    return _env_int(PPO_EPOCHS_ENV, default)


@dataclass(frozen=True)
class BaselineQuoteConfig:
    s_min_bps: float
    k_sigma: float
    k_inv: float
    k_alpha: float
    spread_floor_bps: float
    spread_cap_bps: float
    inv_ref_notional: float
    vol_horizon_ms: int
    horizons_ms: List[int]
    p250_weight: float
    p500_weight: float
    p1000_weight: float


def load_baseline_quote_config() -> BaselineQuoteConfig:
    return BaselineQuoteConfig(
        s_min_bps=_env_float("BYBIT_MM_S_MIN_BPS", DEFAULT_MM_S_MIN_BPS),
        k_sigma=_env_float("BYBIT_MM_K_SIGMA", DEFAULT_MM_K_SIGMA),
        k_inv=_env_float("BYBIT_MM_K_INV", DEFAULT_MM_K_INV),
        k_alpha=_env_float("BYBIT_MM_K_ALPHA", DEFAULT_MM_K_ALPHA),
        spread_floor_bps=_env_float("BYBIT_MM_SPREAD_FLOOR_BPS", DEFAULT_MM_SPREAD_FLOOR_BPS),
        spread_cap_bps=_env_float("BYBIT_MM_SPREAD_CAP_BPS", DEFAULT_MM_SPREAD_CAP_BPS),
        inv_ref_notional=_env_float("BYBIT_MM_INV_REF_NOTIONAL", DEFAULT_MM_INV_REF_NOTIONAL),
        vol_horizon_ms=_env_int("BYBIT_MM_VOL_HORIZON_MS", DEFAULT_MM_VOL_HORIZON_MS),
        horizons_ms=_env_int_list("BYBIT_MM_HORIZONS_MS", DEFAULT_MM_HORIZONS_MS),
        p250_weight=_env_float("BYBIT_MM_P250_WEIGHT", DEFAULT_MM_P250_WEIGHT),
        p500_weight=_env_float("BYBIT_MM_P500_WEIGHT", DEFAULT_MM_P500_WEIGHT),
        p1000_weight=_env_float("BYBIT_MM_P1000_WEIGHT", DEFAULT_MM_P1000_WEIGHT),
    )


def _infer_num_horizons(feature_dim: int) -> int:
    base_dim = feature_dim - len(RAW_SNAPSHOT_FEATURE_COLUMNS) - FEATURE_EXTRA_DIM
    if base_dim <= 0 or base_dim % 4 != 0:
        raise ValueError(
            "Feature dimension does not align with expected horizon layout: "
            f"feature_dim={feature_dim} base_dim={base_dim}"
        )
    return base_dim // 4


def _normalize_horizons(num_h: int, horizons: List[int]) -> List[int]:
    if len(horizons) == num_h:
        return list(horizons)
    if len(horizons) > num_h:
        return list(horizons[:num_h])
    return list(range(num_h))


def _resolve_horizon_index(target_ms: int, horizons: List[int], fallback_idx: int) -> int:
    if target_ms in horizons:
        return horizons.index(target_ms)
    return min(fallback_idx, len(horizons) - 1)


def _load_ts_from_npy(path: Path, expected_len: int, label: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        print(
            "[ts reconstruct]",
            f"skip={label}",
            f"path={path}",
            f"reason=shape_mismatch",
            f"expected_len={expected_len}",
            f"got_shape={arr.shape}",
        )
        return None
    return arr.astype(np.int64, copy=False)


def _load_chunk_ts_alternate(
    week_dir: Path,
    chunk_idx: int,
    expected_len: int,
) -> Optional[np.ndarray]:
    candidates = [
        week_dir / f"ts_{chunk_idx:03d}.npy",
        week_dir / f"ts_{chunk_idx}.npy",
    ]
    for path in candidates:
        ts = _load_ts_from_npy(path, expected_len, label="candidate")
        if ts is not None:
            return ts
    pattern_hits = sorted(week_dir.glob(f"*{chunk_idx:03d}*ts*.npy"))
    for path in pattern_hits:
        ts = _load_ts_from_npy(path, expected_len, label="pattern")
        if ts is not None:
            return ts
    fallback_hits = sorted(week_dir.glob("*_ts.npy"))
    if len(fallback_hits) == 1:
        return _load_ts_from_npy(fallback_hits[0], expected_len, label="fallback")
    for path in fallback_hits:
        ts = _load_ts_from_npy(path, expected_len, label="fallback")
        if ts is not None:
            return ts
    return None


def _ensure_ts_alignment(
    ts: np.ndarray,
    *,
    label: str,
    expected_len: int,
    snapshot_ts: Optional[np.ndarray] = None,
    chunk_offset: Optional[int] = None,
) -> None:
    if ts.ndim != 1 or ts.shape[0] != expected_len:
        raise ValueError(
            f"{label} timestamps length mismatch: expected {expected_len}, got {ts.shape}"
        )
    _ensure_monotonic(ts, label)
    if snapshot_ts is not None and chunk_offset is not None:
        end = chunk_offset + expected_len
        if end > snapshot_ts.shape[0]:
            raise ValueError(
                f"{label} snapshot alignment overflow: chunk_offset={chunk_offset} "
                f"expected_len={expected_len} snapshot_len={snapshot_ts.shape[0]}"
            )
        expected_slice = snapshot_ts[chunk_offset:end]
        if not np.array_equal(ts, expected_slice):
            delta = np.abs(ts - expected_slice)
            max_delta = int(delta.max()) if delta.size else 0
            raise ValueError(
                f"{label} reconstructed ts misaligned with snapshot timeline. "
                f"chunk_offset={chunk_offset} max_delta_ms={max_delta}"
            )
        print(
            "[ts reconstruct]",
            f"label={label}",
            f"len={expected_len}",
            f"start={_format_ts(int(ts[0])) if ts.size else 'n/a'}",
            f"end={_format_ts(int(ts[-1])) if ts.size else 'n/a'}",
            "alignment=ok",
        )


def _load_week_snapshot_ts(out_root: Path, week_key: str, week_meta: dict) -> np.ndarray:
    try:
        snapshot_ts, _raw_snapshots = load_raw_snapshots(out_root, week_key)
        snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
        return np.sort(snapshot_ts)
    except (FileNotFoundError, ValueError) as exc:
        if RAW_SNAPSHOT_PATHS:
            ts_range = week_meta.get("decision_ts_range")
            if not ts_range:
                raise ValueError(
                    f"Missing decision_ts_range for week {week_key}; "
                    "cannot filter RAW_SNAPSHOT_PATHS for timestamp reconstruction."
                ) from exc
            split = {
                "week": week_key,
                "start": int(ts_range["min"]),
                "end": int(ts_range["max"]),
            }
            snapshot_df = load_raw_snapshot_features(str(out_root), split=split, meta=None)
            snapshot_ts = snapshot_df["ts"].to_numpy(dtype=np.int64)
            return np.sort(snapshot_ts)
        raise


def _reconstruct_chunk_ts_from_snapshots(
    snapshot_ts: np.ndarray,
    *,
    chunk_offset: int,
    expected_len: int,
    label: str,
) -> np.ndarray:
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D for reconstruction.")
    if chunk_offset < 0:
        raise ValueError(f"chunk_offset must be >= 0, got {chunk_offset}")
    end = chunk_offset + expected_len
    if end > snapshot_ts.shape[0]:
        raise ValueError(
            f"Insufficient snapshot timestamps for reconstruction: "
            f"chunk_offset={chunk_offset} expected_len={expected_len} "
            f"snapshot_len={snapshot_ts.shape[0]}"
        )
    ts = snapshot_ts[chunk_offset:end].astype(np.int64, copy=False)
    _ensure_ts_alignment(
        ts,
        label=label,
        expected_len=expected_len,
        snapshot_ts=snapshot_ts,
        chunk_offset=chunk_offset,
    )
    return ts


def load_split_arrays(out_root: str, split: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CMSSL tensors for a split, requiring per-chunk decision timestamps by default.

    Decision timestamps are the authoritative alignment source. Snapshot timelines must
    be generated from the same raw delta stream as the decisions if reconstruction is enabled.
    """
    x_core_list: List[np.ndarray] = []
    x_aux_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    out_root_path = Path(out_root)
    meta = load_global_meta(out_root_path)
    week_meta_map = {wk: (wmeta, wk_dir) for wk, wmeta, wk_dir in iter_week_chunks(out_root_path, meta=meta)}
    snapshot_ts_cache: Dict[str, np.ndarray] = {}
    chunk_offsets: Dict[str, int] = {}
    for week, chunk_idx, ts, x_core, x_aux, y in iter_chunk_batches(out_root):
        if week != split["week"]:
            continue
        n_rows = x_core.shape[0]
        chunk_offset = chunk_offsets.get(week, 0)
        if ts is None:
            if not ALLOW_TS_RECONSTRUCT:
                raise ValueError(
                    f"Missing decision timestamps for {week}/chunk{chunk_idx:03d}. "
                    "Ensure meta_week.json includes ts_*.npy entries, or set "
                    "BYBIT_ALLOW_TS_RECONSTRUCT=true to enable snapshot-based recovery."
                )
            week_meta, week_dir = week_meta_map.get(week, (None, None))
            if week_dir is None or week_meta is None:
                raise ValueError(f"Missing week meta for {week}; cannot recover timestamps.")
            ts = _load_chunk_ts_alternate(week_dir, chunk_idx, n_rows)
            if ts is None:
                snapshot_ts = snapshot_ts_cache.get(week)
                if snapshot_ts is None:
                    snapshot_ts = _load_week_snapshot_ts(out_root_path, week, week_meta)
                    snapshot_ts_cache[week] = snapshot_ts
                    expected_total = int(sum(ch.get("n", 0) for ch in week_meta.get("chunks", [])))
                    if expected_total and snapshot_ts.shape[0] < expected_total:
                        raise ValueError(
                            f"Snapshot timeline shorter than token count for week {week}. "
                            f"snapshots={snapshot_ts.shape[0]} tokens={expected_total}"
                        )
                ts = _reconstruct_chunk_ts_from_snapshots(
                    snapshot_ts,
                    chunk_offset=chunk_offset,
                    expected_len=n_rows,
                    label=f"{week}/chunk{chunk_idx:03d}",
                )
            _ensure_ts_alignment(
                ts,
                label=f"{week}/chunk{chunk_idx:03d}",
                expected_len=n_rows,
            )
        mask = (ts >= split["start"]) & (ts < split["end"])
        if not np.any(mask):
            chunk_offsets[week] = chunk_offset + n_rows
            continue
        x_core_list.append(x_core[mask])
        x_aux_list.append(x_aux[mask])
        y_list.append(y[mask])
        ts_list.append(ts[mask])
        chunk_offsets[week] = chunk_offset + n_rows
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


def _resolve_ob_path(week_key: str, ob_dir: str) -> Path:
    if not ob_dir:
        raise FileNotFoundError(
            "BYBIT_OB_DIR not set; cannot reconstruct snapshots from delta stream."
        )
    prefix = "BTCUSDT_OB_"
    if week_key.startswith(prefix):
        pattern = f"{week_key}*"
    else:
        pattern = f"{prefix}{week_key}*"
    candidates = sorted(Path(ob_dir).glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No OB delta stream found for week {week_key} in {ob_dir} "
            f"matching {pattern}"
        )
    return candidates[0]


def _iter_ob_events(ob_path: Path) -> Iterable[Dict[str, Any]]:
    with _open_text(str(ob_path)) as f:
        for line in f:
            if not line:
                continue
            yield json.loads(line)


def _reconstruct_snapshots_from_ob_stream(ob_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    fe = FeatureEngine()
    ts_list: List[int] = []
    bids: List[float] = []
    asks: List[float] = []

    next_sample_ts: Optional[int] = None
    last_bid: Optional[float] = None
    last_ask: Optional[float] = None

    for raw in _iter_ob_events(ob_path):
        etype, ts_ms, payload = fe._parse_event(raw)
        if etype != "ob":
            continue

        if last_bid is not None and last_ask is not None and next_sample_ts is not None:
            while next_sample_ts < ts_ms:
                ts_list.append(int(next_sample_ts))
                bids.append(float(last_bid))
                asks.append(float(last_ask))
                next_sample_ts += RAW_SNAPSHOT_EXPECTED_STEP_MS

        fe._update_book_from_ob(payload)
        bid, ask, _bsz, _asz = fe._book_best()
        if bid <= 0.0 or ask <= 0.0:
            continue

        if next_sample_ts is None:
            next_sample_ts = int(ts_ms)

        while next_sample_ts <= ts_ms:
            ts_list.append(int(next_sample_ts))
            bids.append(float(bid))
            asks.append(float(ask))
            next_sample_ts += RAW_SNAPSHOT_EXPECTED_STEP_MS

        last_bid = bid
        last_ask = ask

    snapshot_ts = np.asarray(ts_list, dtype=np.int64)
    snapshots = np.column_stack([bids, asks]).astype(np.float32)
    return snapshot_ts, snapshots


def _cache_reconstructed_snapshots(week_dir: Path, snapshot_ts: np.ndarray, snapshots: np.ndarray) -> Path:
    out_path = week_dir / "snapshots.npz"
    np.savez_compressed(out_path, ts=snapshot_ts, snapshots=snapshots)
    return out_path


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


def load_raw_snapshot_features(
    out_root: str,
    *,
    split: Optional[Dict[str, int]] = None,
    meta: Optional[dict] = None,
) -> pd.DataFrame:
    if not RAW_SNAPSHOT_PATHS:
        raise ValueError("RAW_SNAPSHOT_PATHS is empty. Set RAW_SNAPSHOT_PATHS or env var.")
    if split is None:
        if meta is None:
            meta = load_global_meta(Path(out_root))
        split = resolve_test_split(out_root, meta)
    frames = [_load_snapshot_frame(path) for path in RAW_SNAPSHOT_PATHS]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["ts", "best_bid", "best_ask"]).copy()
    df["ts"] = df["ts"].astype(np.int64)
    df = df.sort_values("ts").reset_index(drop=True)
    _ensure_sorted_near_regular(df["ts"].to_numpy())
    df = df[(df["ts"] >= split["start"]) & (df["ts"] < split["end"])].copy()
    _ensure_sorted_near_regular(df["ts"].to_numpy())
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df["spread_bps"] = (df["best_ask"] - df["best_bid"]) / df["mid"] * 1e4
    df["mid_ret_1"] = np.log(df["mid"]).diff()
    df["vol_short"] = df["mid_ret_1"].rolling(SHORT_VOL_WINDOW, min_periods=1).std()
    df["vol_long"] = df["mid_ret_1"].rolling(LONG_VOL_WINDOW, min_periods=1).std()
    return df


def _compute_snapshot_feature_matrix(
    snapshot_ts: np.ndarray,
    snapshots: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
    snapshots = np.asarray(snapshots)
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D.")
    if snapshots.ndim != 2 or snapshots.shape[1] < 2:
        raise ValueError(f"Snapshots must be 2D with >=2 columns, got {snapshots.shape}.")
    order = np.argsort(snapshot_ts)
    snapshot_ts = snapshot_ts[order]
    snapshots = snapshots[order]
    best_bid = snapshots[:, 0].astype(np.float64)
    best_ask = snapshots[:, 1].astype(np.float64)
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4
    mid_ret_1 = np.log(mid)
    mid_ret_1 = np.concatenate([[np.nan], np.diff(mid_ret_1)])
    mid_ret_series = pd.Series(mid_ret_1)
    vol_short = mid_ret_series.rolling(SHORT_VOL_WINDOW, min_periods=1).std().to_numpy()
    vol_long = mid_ret_series.rolling(LONG_VOL_WINDOW, min_periods=1).std().to_numpy()
    features = np.column_stack(
        [best_bid, best_ask, mid, spread_bps, mid_ret_1, vol_short, vol_long]
    )
    return snapshot_ts, features


def load_raw_snapshots(out_root: str, week_key: str) -> Tuple[np.ndarray, np.ndarray]:
    week_dir = _find_week_dir(Path(out_root), week_key)
    candidates = [
        week_dir / "snapshots.npz",
        week_dir / "raw_snapshots.npz",
        week_dir / "snapshots.npy",
        week_dir / "raw_snapshots.npy",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        ob_path = _resolve_ob_path(week_key, RAW_OB_DIR)
        snapshot_ts, snapshots = _reconstruct_snapshots_from_ob_stream(ob_path)
        if snapshot_ts.size == 0:
            raise ValueError(f"No snapshots reconstructed from {ob_path}")
        _cache_reconstructed_snapshots(week_dir, snapshot_ts, snapshots)
        return snapshot_ts, snapshots

    if path.suffix == ".npz":
        data = np.load(path)
        if "ts" in data and "snapshots" in data:
            if path.name != "snapshots.npz":
                warnings.warn(
                    f"Non-canonical snapshot filename {path.name}; "
                    "prefer snapshots.npz with ts/snapshots fields.",
                    UserWarning,
                )
            return data["ts"], data["snapshots"]
        if {"ts", "best_bid", "best_ask"}.issubset(data.files):
            warnings.warn(
                f"Non-canonical snapshot fields in {path.name}; "
                "expected ts/snapshots.",
                UserWarning,
            )
            snapshots = np.column_stack([data["best_bid"], data["best_ask"]])
            return data["ts"], snapshots
        if {"timestamps", "best_bid", "best_ask"}.issubset(data.files):
            warnings.warn(
                f"Non-canonical snapshot fields in {path.name}; "
                "expected ts/snapshots.",
                UserWarning,
            )
            snapshots = np.column_stack([data["best_bid"], data["best_ask"]])
            return data["timestamps"], snapshots
        if "timestamps" in data and "X" in data:
            warnings.warn(
                f"Non-canonical snapshot fields in {path.name}; "
                "expected ts/snapshots.",
                UserWarning,
            )
            return data["timestamps"], data["X"]
        raise ValueError(f"Unsupported npz layout in {path}")

    arr = np.load(path)
    if arr.dtype.names:
        if "ts" in arr.dtype.names and "snapshot" in arr.dtype.names:
            warnings.warn(
                f"Non-canonical snapshot layout in {path.name}; "
                "expected ts/snapshots arrays.",
                UserWarning,
            )
            return arr["ts"], arr["snapshot"]
        if "ts" in arr.dtype.names and "snapshots" in arr.dtype.names:
            warnings.warn(
                f"Non-canonical snapshot layout in {path.name}; "
                "expected ts/snapshots in an npz.",
                UserWarning,
            )
            return arr["ts"], arr["snapshots"]
    ts_path = path.with_name("snapshots_ts.npy")
    if ts_path.exists():
        warnings.warn(
            f"Non-canonical snapshot layout using {path.name} + snapshots_ts.npy; "
            "expected snapshots.npz with ts/snapshots.",
            UserWarning,
        )
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


def report_pretrain_diagnostics(out_root: str, meta: dict) -> None:
    test_split = resolve_test_split(out_root, meta)
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

    if RAW_SNAPSHOT_PATHS:
        snapshot_df = load_raw_snapshot_features(out_root, split=test_split, meta=meta)
        snapshot_ts = snapshot_df["ts"].to_numpy(dtype=np.int64)
        filtered = snapshot_ts
    else:
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
) -> np.ndarray:
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
        match_rate = 0.0
        raise ValueError(
            "Snapshot alignment failed; no snapshots available to match decisions. "
            f"match_rate={match_rate:.6f}"
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
    match_rate = float(np.mean(matched)) if matched.size else 0.0
    exact_rate = float(np.mean(exact)) if exact.size else 0.0
    matched_decision_ts = decision_ts[matched]
    matched_snapshot_ts = snapshot_ts[nearest_idx[matched]] if np.any(matched) else np.array([], dtype=np.int64)
    if matched.size and not np.all(matched):
        mismatch_idx = np.flatnonzero(~matched)
        sample_count = min(5, mismatch_idx.size)
        samples = [int(decision_ts[i]) for i in mismatch_idx[:sample_count]]
        raise ValueError(
            "Snapshot alignment failed; decisions outside tolerance. "
            f"unmatched={mismatch_idx.size} total={decision_ts.size} "
            f"match_rate={match_rate:.6f} tolerance_ms={tolerance_ms} "
            f"samples={samples}"
        )
    aligned = snapshots[nearest_idx].astype(np.float32) if decision_ts.size else snapshots[:0].astype(np.float32)

    def _median_dt(ts: np.ndarray) -> float:
        if ts.size < 2:
            return float("nan")
        return float(np.median(np.diff(ts)))

    decision_median_dt = _median_dt(matched_decision_ts)
    snapshot_median_dt = _median_dt(matched_snapshot_ts)
    snapshot_bound_first = int(snapshot_ts[0]) if snapshot_ts.size else None
    snapshot_bound_last = int(snapshot_ts[-1]) if snapshot_ts.size else None
    if matched_decision_ts.size:
        decision_first = int(matched_decision_ts[0])
        decision_last = int(matched_decision_ts[-1])
        snapshot_first = int(matched_snapshot_ts[0])
        snapshot_last = int(matched_snapshot_ts[-1])
    else:
        decision_first = decision_last = snapshot_first = snapshot_last = None
    if matched_decision_ts.size:
        assert snapshot_bound_first is not None and snapshot_bound_last is not None
        assert (
            abs(decision_first - snapshot_bound_first) <= SNAPSHOT_ALIGN_BOUNDS_TOLERANCE_MS
        ), (
            "First matched decision timestamp is too far from snapshot start; "
            f"decision_first={decision_first} snapshot_bound_first={snapshot_bound_first} "
            f"delta_ms={abs(decision_first - snapshot_bound_first)} "
            f"tolerance_ms={SNAPSHOT_ALIGN_BOUNDS_TOLERANCE_MS}"
        )
        assert (
            abs(decision_last - snapshot_bound_last) <= SNAPSHOT_ALIGN_BOUNDS_TOLERANCE_MS
        ), (
            "Last matched decision timestamp is too far from snapshot end; "
            f"decision_last={decision_last} snapshot_bound_last={snapshot_bound_last} "
            f"delta_ms={abs(decision_last - snapshot_bound_last)} "
            f"tolerance_ms={SNAPSHOT_ALIGN_BOUNDS_TOLERANCE_MS}"
        )
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
            f"snapshot_bound_first={_format_ts(snapshot_bound_first) if snapshot_bound_first is not None else 'n/a'}",
            f"snapshot_bound_last={_format_ts(snapshot_bound_last) if snapshot_bound_last is not None else 'n/a'}",
            f"decision_median_dt_ms={decision_median_dt:.2f}",
            f"snapshot_median_dt_ms={snapshot_median_dt:.2f}",
        )
    return aligned


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
    meta: dict,
) -> Dict[str, np.ndarray]:
    ret_pred = cmssl_out["ret_pred"]
    vol_pred = cmssl_out["vol_pred"]
    dir_logits = cmssl_out["dir_logits"]
    snapshot_mask = cmssl_out.get("snapshot_mask")
    if snapshot_mask is not None:
        assert np.all(snapshot_mask), "snapshot_mask contains unmatched decisions; alignment should be exact."
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
    snapshot_spread_col = RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")
    spread_bps = snapshots[:, snapshot_spread_col]  # use aligned snapshot spread
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
    output = {
        "ts": decision_ts,
        "features": features.astype(np.float32),
        "y": y.astype(np.float32),
        "spread_bps": spread_bps.astype(np.float32),
        "alpha_bps": alpha_bps.astype(np.float32),
        "snapshots": snapshots.astype(np.float32),
    }
    if snapshot_mask is not None:
        output["snapshot_mask"] = snapshot_mask.astype(bool)
    return output


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
    if RAW_SNAPSHOT_PATHS:
        test_split = resolve_test_split(out_root, meta)
        if split != test_split:
            raise ValueError(
                "RAW_SNAPSHOT_PATHS only supports CMSSL test split alignment. "
                f"Requested split={split} test_split={test_split}"
            )
        snapshot_df = load_raw_snapshot_features(out_root, split=split, meta=meta)
        snapshot_ts = snapshot_df["ts"].to_numpy(dtype=np.int64)
        snapshots = snapshot_df[RAW_SNAPSHOT_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    else:
        snapshot_ts, raw_snapshots = load_raw_snapshots(out_root, split["week"])
        snapshot_ts, snapshots = _compute_snapshot_feature_matrix(snapshot_ts, raw_snapshots)
    aligned_snapshots = align_snapshots_to_decisions(
        ts,
        snapshot_ts,
        snapshots,
        label=split_label,
    )
    return join_features(ts, y, cmssl_out, aligned_snapshots, meta)


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
class MarketMakingBatch:
    features: np.ndarray
    spread_bps: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    alpha_bps: Optional[np.ndarray] = None


class MarketMakingEnv:
    def __init__(
        self,
        batch: MarketMakingBatch,
        *,
        maker_rebate_bps: float = 0.0,
        taker_fee_bps: float = DEFAULT_MM_TAKER_FEE_BPS,
        allow_taker: bool = True,
        taker_threshold: float = DEFAULT_MM_TAKER_THRESHOLD,
        inventory_penalty: float = 0.0,
        inv_soft: float = 1.0,
        lambda_inv: float = 0.0,
        lambda_turn: float = 0.0,
        max_inventory: Optional[float] = None,
        fill_size: float = 1.0,
        fill_tolerance: float = 1e-6,
        delta_bps_limit: Optional[float] = None,
    ):
        self.features = batch.features
        self.spread_bps = batch.spread_bps
        self.best_bid = batch.best_bid
        self.best_ask = batch.best_ask
        self.alpha_bps = batch.alpha_bps if batch.alpha_bps is not None else np.zeros_like(self.spread_bps)
        self.maker_rebate_bps = maker_rebate_bps
        self.taker_fee_bps = taker_fee_bps
        self.allow_taker = allow_taker
        self.taker_threshold = taker_threshold
        self.inventory_penalty = inventory_penalty
        self.inv_soft = inv_soft
        self.lambda_inv = lambda_inv
        self.lambda_turn = lambda_turn
        self.max_inventory = max_inventory
        self.fill_size = fill_size
        self.fill_tolerance = fill_tolerance
        self.delta_bps_limit = delta_bps_limit
        self.notional_scale = _env_float("BYBIT_MM_NOTIONAL_SCALE", DEFAULT_MM_NOTIONAL_SCALE)
        self.cash_scale = _env_float("BYBIT_MM_CASH_SCALE", DEFAULT_MM_CASH_SCALE)
        self.time_since_fill_scale = _env_float(
            "BYBIT_MM_TIME_SINCE_FILL_SCALE",
            DEFAULT_MM_TIME_SINCE_FILL_SCALE,
        )
        self._baseline_cfg = load_baseline_quote_config()
        self._num_h = _infer_num_horizons(self.features.shape[-1])
        self._horizons_ms = _normalize_horizons(self._num_h, self._baseline_cfg.horizons_ms)
        self._vol_horizon_idx = _resolve_horizon_index(
            self._baseline_cfg.vol_horizon_ms,
            self._horizons_ms,
            fallback_idx=min(self._num_h - 1, 2),
        )
        self._p250_idx = _resolve_horizon_index(250, self._horizons_ms, fallback_idx=0)
        self._p500_idx = _resolve_horizon_index(500, self._horizons_ms, fallback_idx=min(1, self._num_h - 1))
        self._p1000_idx = _resolve_horizon_index(1000, self._horizons_ms, fallback_idx=min(2, self._num_h - 1))

        self.n = len(self.spread_bps)
        self.idx = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.total_reward = 0.0
        self.prev_equity = 0.0
        self.time_since_last_fill = 0.0
        self._obs_count = 0
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_m2: Optional[np.ndarray] = None
        self._obs_continuous_mask: Optional[np.ndarray] = None
        self._episode_obs_count = 0
        self._episode_obs_mean: Optional[np.ndarray] = None
        self._episode_obs_m2: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.total_reward = 0.0
        self.time_since_last_fill = 0.0
        mid = self._mid_price(self.idx)
        self.prev_equity = self.cash + self.inventory * mid
        self._episode_obs_count = 0
        self._episode_obs_mean = None
        self._episode_obs_m2 = None
        return self._build_observation(self.idx)

    def _mid_price(self, idx: int) -> float:
        return float((self.best_bid[idx] + self.best_ask[idx]) / 2.0)

    def _build_observation(self, idx: int) -> np.ndarray:
        cash_scaled = self.cash / self.cash_scale if self.cash_scale else 0.0
        time_since_last_fill_scaled = (
            self.time_since_last_fill / self.time_since_fill_scale if self.time_since_fill_scale else 0.0
        )
        extra = np.array(
            [
                self.inventory,
                cash_scaled,
                time_since_last_fill_scaled,
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([self.features[idx].astype(np.float32), extra], axis=0)
        return self._normalize_observation(obs)

    def _continuous_mask(self, obs_dim: int) -> np.ndarray:
        mask = np.ones(obs_dim, dtype=bool)
        prob_start = 3 * self._num_h
        prob_end = 4 * self._num_h
        if prob_end <= obs_dim:
            mask[prob_start:prob_end] = False
        return mask

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        if self._obs_mean is None or self._obs_m2 is None:
            self._obs_mean = np.zeros_like(obs, dtype=np.float64)
            self._obs_m2 = np.zeros_like(obs, dtype=np.float64)
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_m2 += delta * delta2

    def _update_episode_obs_stats(self, obs: np.ndarray) -> None:
        if self._episode_obs_mean is None or self._episode_obs_m2 is None:
            self._episode_obs_mean = np.zeros_like(obs, dtype=np.float64)
            self._episode_obs_m2 = np.zeros_like(obs, dtype=np.float64)
        self._episode_obs_count += 1
        delta = obs - self._episode_obs_mean
        self._episode_obs_mean += delta / self._episode_obs_count
        delta2 = obs - self._episode_obs_mean
        self._episode_obs_m2 += delta * delta2

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_continuous_mask is None:
            self._obs_continuous_mask = self._continuous_mask(obs.shape[0])
        normalized = obs.copy()
        if self._obs_count >= 2 and self._obs_mean is not None and self._obs_m2 is not None:
            var = self._obs_m2 / max(self._obs_count - 1, 1)
            std = np.sqrt(np.maximum(var, 1e-6))
            mask = self._obs_continuous_mask
            normalized[mask] = (obs[mask] - self._obs_mean[mask]) / std[mask]
        self._update_obs_stats(obs)
        self._update_episode_obs_stats(obs)
        return normalized

    def _parse_action(self, action: Any) -> Tuple[float, float, float]:
        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) == 3:
                return float(action[0]), float(action[1]), float(action[2])
            if len(action) == 2:
                return float(action[0]), float(action[1]), 0.0
        if np.isscalar(action):
            return float(action), float(action), 0.0
        raise ValueError("Action must be a scalar or (bid_delta_bps, ask_delta_bps[, taker_signal]).")

    def _feature_slice(self, idx: int, start: int, end: int) -> np.ndarray:
        return self.features[idx, start:end]

    def _baseline_quotes(self, idx: int) -> Tuple[float, float, float]:
        cfg = self._baseline_cfg
        mid = self._mid_price(idx)
        ret_pred = self._feature_slice(idx, 0, self._num_h)
        vol_pred = self._feature_slice(idx, self._num_h, 2 * self._num_h)
        p_up = self._feature_slice(idx, 3 * self._num_h, 4 * self._num_h)
        sigma_bps = 1e4 * float(sigma_from_vol(vol_pred[self._vol_horizon_idx]))
        ret_forecast_bps = 1e4 * float(ret_pred[self._vol_horizon_idx])
        p250 = float(p_up[self._p250_idx])
        p500 = float(p_up[self._p500_idx])
        p1000 = float(p_up[self._p1000_idx])
        weight_sum = cfg.p250_weight + cfg.p500_weight + cfg.p1000_weight
        if weight_sum > 0.0:
            p_weighted = (
                cfg.p250_weight * p250 + cfg.p500_weight * p500 + cfg.p1000_weight * p1000
            ) / weight_sum
        else:
            p_weighted = 0.5
        # Blend the explicit return forecast with probability-weighted sigma to steer skew.
        alpha = (p_weighted - 0.5) * 2.0 * sigma_bps + ret_forecast_bps
        s_min_bps = cfg.s_min_bps
        snapshot_offset = 4 * self._num_h + FEATURE_EXTRA_DIM
        spread_idx = snapshot_offset + RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")
        if spread_idx < self.features.shape[1]:
            observed_spread_bps = float(self.features[idx, spread_idx])
            if np.isfinite(observed_spread_bps) and observed_spread_bps > 0.0:
                # Anchor the minimum spread to the observed half-spread when snapshots are available.
                s_min_bps = max(s_min_bps, 0.5 * observed_spread_bps)
        half_spread_bps = s_min_bps + cfg.k_sigma * sigma_bps
        half_spread_bps = float(np.clip(half_spread_bps, cfg.spread_floor_bps, cfg.spread_cap_bps))
        inv_ref = cfg.inv_ref_notional if cfg.inv_ref_notional > 0.0 else 1.0
        inv_notional = self.inventory * mid
        skew_bps = cfg.k_inv * (inv_notional / inv_ref) - cfg.k_alpha * alpha
        half_spread_px = bps_to_px(mid, half_spread_bps)
        skew_px = bps_to_px(mid, skew_bps)
        # Plan: spread = s_min + k_sigma*sigma (floored/capped), skew = k_inv*inv - k_alpha*alpha.
        bid = mid - half_spread_px - skew_px
        ask = mid + half_spread_px - skew_px
        return bid, ask, mid

    def _apply_deltas(
        self, bid: float, ask: float, mid: float, action: Any
    ) -> Tuple[float, float, float, float]:
        bid_delta_bps, ask_delta_bps, _taker_signal = self._parse_action(action)
        if self.delta_bps_limit is not None:
            bid_delta_bps = float(np.clip(bid_delta_bps, -self.delta_bps_limit, self.delta_bps_limit))
            ask_delta_bps = float(np.clip(ask_delta_bps, -self.delta_bps_limit, self.delta_bps_limit))
        bid += mid * bid_delta_bps * 1e-4
        ask += mid * ask_delta_bps * 1e-4
        return bid, ask, bid_delta_bps, ask_delta_bps

    def _enforce_passive(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        mid = 0.5 * (best_bid + best_ask)
        eps = max(1e-8, mid * 1e-6)
        bid = min(bid, best_ask - eps)
        ask = max(ask, best_bid + eps)
        if bid >= ask:
            bid = best_bid
            ask = best_ask
        return bid, ask

    def _apply_fills(self, bid: float, ask: float, idx: int) -> Tuple[float, float]:
        touch_epsilon = 1e-9
        best_bid_next = float(self.best_bid[idx])
        best_ask_next = float(self.best_ask[idx])
        best_bid_prev = float(self.best_bid[idx - 1]) if idx > 0 else best_bid_next
        best_ask_prev = float(self.best_ask[idx - 1]) if idx > 0 else best_ask_next
        buy_fill = 0.0
        sell_fill = 0.0
        # Evaluate fills against the next snapshot's opposite side.
        if best_ask_next <= bid + self.fill_tolerance:
            buy_fill = self.fill_size
            self.cash -= bid * buy_fill
            self.inventory += buy_fill
        if best_bid_next >= ask - self.fill_tolerance:
            sell_fill = self.fill_size
            self.cash += ask * sell_fill
            self.inventory -= sell_fill
        # Heuristic: if we're at the touch and the next best moves away, we got hit.
        touch_tolerance = max(self.fill_tolerance, touch_epsilon)
        if buy_fill == 0.0 and abs(bid - best_bid_prev) <= touch_tolerance:
            if best_bid_next < best_bid_prev - touch_epsilon:
                buy_fill = self.fill_size
                self.cash -= bid * buy_fill
                self.inventory += buy_fill
        if sell_fill == 0.0 and abs(ask - best_ask_prev) <= touch_tolerance:
            if best_ask_next > best_ask_prev + touch_epsilon:
                sell_fill = self.fill_size
                self.cash += ask * sell_fill
                self.inventory -= sell_fill
        return buy_fill, sell_fill

    def _apply_taker(self, idx: int, taker_signal: float) -> Tuple[float, float]:
        if not self.allow_taker:
            return 0.0, 0.0
        if abs(taker_signal) < self.taker_threshold:
            return 0.0, 0.0
        best_bid = float(self.best_bid[idx])
        best_ask = float(self.best_ask[idx])
        buy_fill = 0.0
        sell_fill = 0.0
        if taker_signal > 0.0:
            buy_fill = self.fill_size
            self.cash -= best_ask * buy_fill
            self.inventory += buy_fill
        elif taker_signal < 0.0:
            sell_fill = self.fill_size
            self.cash += best_bid * sell_fill
            self.inventory -= sell_fill
        return buy_fill, sell_fill

    def _compute_penalty(self, mid: float) -> float:
        # Penalize inventory exposure in notional terms.
        inventory_notional = self.inventory * mid
        penalty = self.inventory_penalty * abs(inventory_notional)
        if self.max_inventory is not None and abs(inventory_notional) > self.max_inventory:
            penalty += self.inventory_penalty * (abs(inventory_notional) - self.max_inventory)
        return penalty

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        # Execution convention: both maker and taker fills are priced using the next snapshot
        # (next_idx). We quote on self.idx, then advance state after applying fills at next_idx.
        next_idx = self.idx + 1
        if next_idx >= self.n:
            mid = self._mid_price(self.idx)
            equity = self.cash + self.inventory * mid
            info = {
                "reward": 0.0,
                "total_reward": float(self.total_reward),
                "cash": float(self.cash),
                "inventory": float(self.inventory),
                "equity": float(equity),
                "delta_equity": 0.0,
                "rebate": 0.0,
                "penalty": 0.0,
                "inv_penalty": 0.0,
                "turnover_penalty": 0.0,
                "mid": float(mid),
                "bid": 0.0,
                "ask": 0.0,
                "maker_buy": 0.0,
                "maker_sell": 0.0,
                "taker_buy": 0.0,
                "taker_sell": 0.0,
                "taker_fee": 0.0,
            }
            return self._build_observation(self.idx), 0.0, True, info
        bid, ask, mid = self._baseline_quotes(self.idx)
        bid, ask, bid_delta_bps, ask_delta_bps = self._apply_deltas(bid, ask, mid, action)
        bid, ask = self._enforce_passive(bid, ask, self.idx)
        inv_prev = self.inventory
        _, _, taker_signal = self._parse_action(action)
        maker_buy, maker_sell = self._apply_fills(bid, ask, next_idx)
        taker_buy, taker_sell = self._apply_taker(next_idx, taker_signal)
        inv_new = self.inventory
        inv_change = inv_new - inv_prev
        if maker_buy > 0.0 or maker_sell > 0.0 or taker_buy > 0.0 or taker_sell > 0.0:
            self.time_since_last_fill = 0.0
        else:
            self.time_since_last_fill += 1.0

        mid_next = self._mid_price(next_idx)
        maker_rebate_notional = maker_buy * bid + maker_sell * ask
        rebate = maker_rebate_notional * self.maker_rebate_bps * 1e-4
        taker_notional = taker_buy * float(self.best_ask[next_idx]) + taker_sell * float(self.best_bid[next_idx])
        taker_fee = taker_notional * self.taker_fee_bps * 1e-4
        self.cash += rebate - taker_fee
        equity = self.cash + self.inventory * mid_next
        delta_equity = equity - self.prev_equity
        penalty = self._compute_penalty(mid_next)
        inv_notional = inv_new * mid_next
        excess = max(0.0, abs(inv_notional) - self.inv_soft)
        inv_penalty = (
            self.lambda_inv * (excess / self.inv_soft) ** 2 if self.inv_soft > 0.0 else 0.0
        )
        turnover_notional = maker_rebate_notional + taker_notional
        turnover_penalty = self.lambda_turn * turnover_notional
        reward = delta_equity - penalty - inv_penalty - turnover_penalty

        self.prev_equity = equity
        self.total_reward += reward
        self.idx = next_idx
        done = self.idx >= self.n - 1
        next_obs = self._build_observation(self.idx)
        info = {
            "reward": float(reward),
            "total_reward": float(self.total_reward),
            "cash": float(self.cash),
            "inventory": float(self.inventory),
            "equity": float(equity),
            "delta_equity": float(delta_equity),
            "rebate": float(rebate),
            "taker_fee": float(taker_fee),
            "penalty": float(penalty),
            "inv_penalty": float(inv_penalty),
            "turnover_penalty": float(turnover_penalty),
            "mid": float(mid),
            "bid": float(bid),
            "ask": float(ask),
            "bid_delta_bps": float(bid_delta_bps),
            "ask_delta_bps": float(ask_delta_bps),
            "inv_change": float(inv_change),
            "maker_buy": float(maker_buy),
            "maker_sell": float(maker_sell),
            "taker_buy": float(taker_buy),
            "taker_sell": float(taker_sell),
        }
        return next_obs, float(reward), done, info


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


class MarketPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (128, 128), action_dim: int = 3):
        super().__init__()
        # MarketPolicyNet wraps its MLP under .net for compatibility with checkpoints.
        self.net = MLP(input_dim, hidden_dims, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketPolicyValueNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        policy_hidden: Iterable[int] = (128, 128),
        value_hidden: Iterable[int] = (128, 128),
        action_dim: int = 3,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.policy_net = MarketPolicyNet(input_dim, hidden_dims=policy_hidden, action_dim=action_dim)
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


def collect_market_rollout(
    env: MarketMakingEnv,
    model: MarketPolicyValueNet,
    device: str,
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
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
        action_np = action.squeeze(0).cpu().numpy()
        if action_np.shape[0] >= 3:
            scaled_action = np.array(
                [action_np[0] * delta_scale, action_np[1] * delta_scale, action_np[2] * taker_scale],
                dtype=np.float32,
            )
        else:
            scaled_action = np.array([action_np[0] * delta_scale, action_np[1] * delta_scale], dtype=np.float32)
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

def _steps_per_year_from_snapshot_ms(step_ms: float) -> float:
    if step_ms <= 0:
        return 0.0
    steps_per_second = 1000.0 / step_ms
    return steps_per_second * 60.0 * 60.0 * 24.0 * 365.0


def compute_sharpe(returns: np.ndarray, steps_per_year: float) -> float:
    """Compute annualized Sharpe for per-step percentage returns."""
    if returns.size == 0 or steps_per_year <= 0:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1) if returns.size > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(mean / std * np.sqrt(steps_per_year))


def compute_max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(drawdown.max(initial=0.0))


def evaluate_market_policy(
    env: MarketMakingEnv,
    policy: MarketPolicyNet,
    device: str = "cuda",
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
) -> Dict[str, Any]:
    def _policy_fn(obs: np.ndarray) -> Tuple[float, float, float]:
        obs_t = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            deltas = policy(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        if deltas.shape[0] >= 3:
            return (
                float(deltas[0] * delta_scale),
                float(deltas[1] * delta_scale),
                float(deltas[2] * taker_scale),
            )
        return float(deltas[0] * delta_scale), float(deltas[1] * delta_scale), 0.0

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
    taker_scale: float = 1.0,
) -> Tuple[MarketPolicyValueNet, Dict[str, Any]]:
    config = config or PPOConfig()
    model = MarketPolicyValueNet(
        input_dim,
        policy_hidden=config.policy_hidden,
        value_hidden=config.value_hidden,
        action_dim=int(os.environ.get("BYBIT_MM_ACTION_DIM", "3")),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_report: Dict[str, Any] = {"sharpe": -np.inf}

    for epoch in range(epochs):
        rollout = collect_market_rollout(
            train_env,
            model,
            device,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
        )
        ppo_update_market(model, optimizer, rollout, config, device)
        if (epoch + 1) % config.val_every == 0:
            report = evaluate_market_policy(
                val_env,
                model.policy_net,
                device=device,
                delta_scale=delta_scale,
                taker_scale=taker_scale,
            )
            sharpe = report["sharpe"]
            drawdown = report["max_drawdown"]
            guard = config.max_drawdown_guard
            if sharpe > best_report.get("sharpe", -np.inf) and (guard is None or drawdown <= guard):
                best_report = report
                if ckpt_path:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "policy_state_dict": model.policy_net.state_dict(),
                            "value_state_dict": model.value_net.state_dict(),
                            "hidden_dims": tuple(config.policy_hidden),
                            "action_dim": model.log_std.shape[0],
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
    turnover_qty = 0.0
    turnover_notional = 0.0
    taker_notional = 0.0
    taker_fee_total = 0.0
    maker_fill_count = 0
    maker_opps = 0
    taker_steps = 0
    steps = 0
    initial_equity = env.prev_equity

    done = False
    while not done:
        action = policy_fn(obs)
        obs, _reward, done, info = env.step(action)
        equity_curve.append(info["equity"])
        inventory_curve.append(info["inventory"])
        steps += 1
        step_qty = abs(info["maker_buy"]) + abs(info["maker_sell"]) + abs(info["taker_buy"]) + abs(info["taker_sell"])
        step_mid = float(info.get("mid", env._mid_price(env.idx - 1)))
        step_taker_qty = abs(info["taker_buy"]) + abs(info["taker_sell"])
        turnover_qty += step_qty
        step_notional = step_qty * step_mid
        turnover_notional += step_notional
        taker_notional += step_taker_qty * step_mid
        taker_fee_total += float(info.get("taker_fee", 0.0))
        maker_fill_count += int(info["maker_buy"] > 0.0) + int(info["maker_sell"] > 0.0)
        maker_opps += 2
        taker_steps += int(info["taker_buy"] > 0.0 or info["taker_sell"] > 0.0)

    equity_arr = np.array(equity_curve, dtype=np.float32)
    # Per-snapshot percentage returns; annualization uses the snapshot cadence.
    prev_equity = np.concatenate([[initial_equity], equity_arr[:-1]])
    returns = np.divide(
        equity_arr,
        prev_equity,
        out=np.zeros_like(equity_arr),
        where=prev_equity != 0,
    ) - 1.0
    step_ms = _env_float("BYBIT_MM_SNAPSHOT_STEP_MS", RAW_SNAPSHOT_EXPECTED_STEP_MS)
    steps_per_year = _steps_per_year_from_snapshot_ms(step_ms)
    sharpe = compute_sharpe(returns, steps_per_year)
    max_drawdown = compute_max_drawdown(returns)
    maker_fill_rate = float(maker_fill_count / maker_opps) if maker_opps > 0 else 0.0
    taker_usage_frequency = float(taker_steps / steps) if steps > 0 else 0.0
    taker_volume_share = float(taker_notional / turnover_notional) if turnover_notional > 0 else 0.0
    fee_drag = float(taker_fee_total / turnover_notional) if turnover_notional > 0 else 0.0
    inventory_arr = np.array(inventory_curve, dtype=np.float32)

    return {
        "equity_curve": equity_arr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover_qty": float(turnover_qty),
        "turnover_notional": float(turnover_notional),
        "maker_fill_rate": maker_fill_rate,
        "taker_usage_frequency": taker_usage_frequency,
        "taker_volume_share": taker_volume_share,
        "fee_drag": fee_drag,
        "inventory_distribution": _inventory_distribution(inventory_arr),
    }


def _format_mm_summary(label: str, metrics: Dict[str, Any]) -> str:
    inv = metrics["inventory_distribution"]
    return (
        f"{label}: sharpe={metrics['sharpe']:.4f} "
        f"max_dd={metrics['max_drawdown']:.4f} "
        f"turnover_notional={metrics['turnover_notional']:.4f} "
        f"turnover_qty={metrics['turnover_qty']:.4f} "
        f"maker_fill_rate={metrics['maker_fill_rate']:.4f} "
        f"taker_usage_freq={metrics['taker_usage_frequency']:.4f} "
        f"taker_volume_share={metrics['taker_volume_share']:.4f} "
        f"fee_drag={metrics['fee_drag']:.4f} "
        f"inv[min={inv['min']:.2f}, p50={inv['p50']:.2f}, max={inv['max']:.2f}]"
    )


def load_market_policy(
    input_dim: int,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
) -> Optional[MarketPolicyNet]:
    """Load a deterministic market policy (mean-action inference only)."""
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Market policy checkpoint not found: {ckpt_path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        state = ckpt["policy_state_dict"]
    else:
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hidden_dims = ckpt.get("hidden_dims") if isinstance(ckpt, dict) else None
    if hidden_dims is None:
        hidden_dims = tuple(
            int(x) for x in os.environ.get("BYBIT_MM_POLICY_HIDDEN", "128,128").split(",")
        )
    if isinstance(ckpt, dict) and "action_dim" in ckpt:
        action_dim = int(ckpt["action_dim"])
    else:
        action_dim = int(os.environ.get("BYBIT_MM_ACTION_DIM", "3"))
    model = MarketPolicyNet(input_dim, hidden_dims=hidden_dims, action_dim=action_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_pipeline(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    ppo_epochs: int = 10,
) -> Dict[str, Any]:
    meta = load_global_meta(Path(out_root))
    test_split = resolve_test_split(out_root, meta)

    report_pretrain_diagnostics(out_root, meta)

    model, _meta = load_cmssl(out_root, ckpt_path, device=device)

    joined_test = build_joined_split(
        out_root,
        test_split,
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

    mm_train_batch = build_market_batch(splits_rl["train"])
    mm_val_batch = build_market_batch(splits_rl["val"])
    mm_test_batch = build_market_batch(splits_rl["test"])
    mm_obs_dim = mm_train_batch.features.shape[-1] + 3
    maker_rebate_bps = float(os.environ.get("BYBIT_MM_MAKER_REBATE_BPS", "0.0"))
    inventory_penalty = float(os.environ.get("BYBIT_MM_INVENTORY_PENALTY", "0.0"))
    # Inventory/turnover penalties applied inside MarketMakingEnv.step().
    inv_soft = float(os.environ.get("BYBIT_MM_INV_SOFT", "1.0"))
    lambda_inv = float(os.environ.get("BYBIT_MM_LAMBDA_INV", "0.0"))
    lambda_turn = float(os.environ.get("BYBIT_MM_LAMBDA_TURN", "0.0"))
    max_inventory_str = os.environ.get("BYBIT_MM_MAX_INVENTORY", "").strip()
    max_inventory = float(max_inventory_str) if max_inventory_str else None
    fill_size = float(os.environ.get("BYBIT_MM_FILL_SIZE", "1.0"))
    fill_tolerance = float(os.environ.get("BYBIT_MM_FILL_TOLERANCE", "1e-6"))
    delta_scale = float(os.environ.get("BYBIT_MM_DELTA_SCALE", "1.0"))
    taker_scale = float(os.environ.get("BYBIT_MM_TAKER_SCALE", "1.0"))
    allow_taker = os.environ.get("BYBIT_MM_ALLOW_TAKER", "true").strip().lower() in {"1", "true", "yes", "y"}
    taker_fee_bps = float(os.environ.get("BYBIT_MM_TAKER_FEE_BPS", str(DEFAULT_MM_TAKER_FEE_BPS)))
    taker_threshold = float(os.environ.get("BYBIT_MM_TAKER_THRESHOLD", str(DEFAULT_MM_TAKER_THRESHOLD)))
    delta_bps_limit_str = os.environ.get("BYBIT_MM_DELTA_BPS_LIMIT", "").strip()
    delta_bps_limit = float(delta_bps_limit_str) if delta_bps_limit_str else None

    mm_train_env = MarketMakingEnv(
        mm_train_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=allow_taker,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft=inv_soft,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
        delta_bps_limit=delta_bps_limit,
    )
    mm_val_env = MarketMakingEnv(
        mm_val_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=allow_taker,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft=inv_soft,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
        delta_bps_limit=delta_bps_limit,
    )
    mm_test_env = MarketMakingEnv(
        mm_test_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=allow_taker,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft=inv_soft,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
        delta_bps_limit=delta_bps_limit,
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
        epochs=_resolve_ppo_epochs(ppo_epochs),
        config=mm_ppo_config,
        ckpt_path=mm_best_ckpt,
        delta_scale=delta_scale,
        taker_scale=taker_scale,
    )

    baseline_env = MarketMakingEnv(
        mm_test_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=False,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft=inv_soft,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory=max_inventory,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
        delta_bps_limit=delta_bps_limit,
    )
    baseline_metrics = evaluate_market_making(baseline_env, lambda _obs: (0.0, 0.0, 0.0))

    mm_policy_path = os.environ.get("BYBIT_MM_RL_CKPT", "").strip() or str(mm_best_ckpt)
    mm_policy = load_market_policy(mm_obs_dim, device=device, ckpt_path=mm_policy_path or None)
    if mm_policy is None:
        print("[mm eval] no BYBIT_MM_RL_CKPT provided; using baseline deltas for RL run.")
        rl_policy_fn = lambda _obs: (0.0, 0.0, 0.0)
        rl_policy_loaded = False
    else:
        rl_policy_loaded = True

        def rl_policy_fn(obs: np.ndarray) -> Tuple[float, float, float]:
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                deltas = mm_policy(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            if deltas.shape[0] >= 3:
                return (
                    float(deltas[0] * delta_scale),
                    float(deltas[1] * delta_scale),
                    float(deltas[2] * taker_scale),
                )
            return float(deltas[0] * delta_scale), float(deltas[1] * delta_scale), 0.0

    rl_metrics = evaluate_market_making(mm_test_env, rl_policy_fn)

    print("[mm eval]", _format_mm_summary("baseline", baseline_metrics))
    print("[mm eval]", _format_mm_summary("baseline+rl", rl_metrics))

    return {
        "cmssl_test": cmssl_report,
        "mm_baseline": baseline_metrics,
        "mm_rl": rl_metrics,
        "mm_rl_policy_loaded": {"loaded": rl_policy_loaded},
    }


if __name__ == "__main__":
    out_root = os.environ.get("BYBIT_OUT_ROOT", "").strip()
    ckpt_path = os.environ.get("BYBIT_CMSSL_CKPT", "").strip()
    device = os.environ.get("BYBIT_DEVICE", "cuda")
    ppo_epochs = _resolve_ppo_epochs(10)

    if not out_root or not ckpt_path:
        raise SystemExit("Set BYBIT_OUT_ROOT and BYBIT_CMSSL_CKPT before running.")

    print(
        "[rl exec config]",
        json.dumps(
            {
                "out_root": out_root,
                "ckpt_path": ckpt_path,
                "device": device,
                "ppo_epochs": ppo_epochs,
                "ppo_epochs_env": PPO_EPOCHS_ENV,
            },
            sort_keys=True,
        ),
    )
    report = run_pipeline(out_root, ckpt_path, device=device, ppo_epochs=ppo_epochs)
    print("[cmssl test]", report["cmssl_test"])
    print("[mm baseline]", report["mm_baseline"])
    print("[mm rl]", report["mm_rl"])
