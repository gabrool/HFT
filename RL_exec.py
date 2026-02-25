import json
import os
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from CMSSL17 import SAMBA, ModelArgs, DMODEL, MAMBA_LAYERS, LOOKBACK
from offline_tokens import iter_week_chunks, load_global_meta

RAW_SNAPSHOT_EXPECTED_STEP_MS = 100
RAW_SNAPSHOT_FEATURE_COLUMNS = [
    "best_bid",
    "best_ask",
    "best_bid_size",
    "best_ask_size",
    "imbalance",
    "mid",
    "spread_bps",
    "mid_ret_1",
    "vol_short",
    "vol_long",
]
FEATURE_EXTRA_DIM = 5
ENV_OBS_EXTRA_STATE_DIM = 14
SHORT_VOL_WINDOW = 50
LONG_VOL_WINDOW = 200
# CMSSL market-making horizon contract is fixed: exactly [250, 500, 1000] ms.
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
# Scaling factors for market-making observation extra-state features.
# Inventory notional and cash scales are denominated in quote currency.
# Time-since-fill scale is in environment steps (1 step per snapshot).
DEFAULT_MM_INVENTORY_NOTIONAL_SCALE = 1e4
DEFAULT_MM_CASH_SCALE = 1e4
DEFAULT_MM_TIME_SINCE_FILL_SCALE = 1000.0
DEFAULT_MM_FILL_NOTIONAL_SCALE = 1e4
DEFAULT_MM_PNL_NOTIONAL_SCALE = 1e4
DEFAULT_MM_MARKOUT_NOTIONAL_SCALE = 1e4
DEFAULT_MM_FILL_EMA_WINDOW_STEPS = 3
DEFAULT_MM_INITIAL_CASH = 1_000_000.0
DEFAULT_MM_TAKER_FEE_BPS = 1.7
DEFAULT_MM_TAKER_THRESHOLD = 0.25
# Inventory risk thresholds are denominated in quote notional (USD).

def require(condition: bool, msg: str, exc_type: type[Exception] = ValueError) -> None:
    """Raise a typed exception when a runtime precondition fails."""
    if not condition:
        raise exc_type(msg)


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
    require(expected_h > 0, "meta['horizons_ms'] must be non-empty")
    require(ret_pred.shape[-1] == expected_h, (
        f"ret_pred shape {ret_pred.shape} does not match horizons {expected_h}"
    ))
    require(vol_pred.shape[-1] == expected_h, (
        f"vol_pred shape {vol_pred.shape} does not match horizons {expected_h}"
    ))
    require(dir_logits.shape[-1] == expected_h, (
        f"dir_logits shape {dir_logits.shape} does not match horizons {expected_h}"
    ))
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


def get_cmssl_splits(out_root: str) -> dict:
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    splits = meta.get("splits", {})

    missing = [
        key for key in ("train", "holdout_week", "train_ts_range", "val_ts_range", "test_ts_range")
        if key not in splits
    ]
    require(not missing, (
        "meta.json missing split ranges — rerun offline_ingest to generate canonical splits"
    ))
    require(isinstance(splits["train"], list), (
        "meta.json missing split ranges — rerun offline_ingest to generate canonical splits"
    ))

    train_ts_range = splits["train_ts_range"]
    val_ts_range = splits["val_ts_range"]
    test_ts_range = splits["test_ts_range"]

    return {
        "train": {
            "weeks": splits["train"],
            "start": int(train_ts_range["min"]),
            "end": int(train_ts_range["max"]),
        },
        "val": {
            "weeks": [splits["holdout_week"]],
            "start": int(val_ts_range["min"]),
            "end": int(val_ts_range["max"]),
        },
        "test": {
            "weeks": [splits["holdout_week"]],
            "start": int(test_ts_range["min"]),
            "end": int(test_ts_range["max"]),
        },
    }


def sigma_from_vol(log_vol: np.ndarray) -> np.ndarray:
    """Recover volatility from log-vol predictions."""
    return np.exp(log_vol)


def bps_to_px(mid: float, bps: float) -> float:
    return mid * bps * 1e-4


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int_list(name: str, default: List[int]) -> List[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [int(item) for item in raw.split(",") if item.strip()]


def _resolve_ppo_epochs(default: int) -> int:
    return _env_int(PPO_EPOCHS_ENV, default)


def _set_seed_from_env(env_name: str = "BYBIT_SEED") -> Optional[int]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    seed = int(raw)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("[seed]", f"{env_name}={seed}")
    return seed


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
        # CMSSL MM contract is fixed to [250, 500, 1000]; any deviation is a hard error.
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


def _joined_feature_layout(num_horizons: int, snapshot_dim: int) -> Dict[str, slice]:
    """Schema for join_features() tensor layout (excluding env extra state).

    Layout order:
      [ret(h), vol(h), logits(h), p_up(h), align/conf-delta/conf metrics(5), snapshots(snapshot_dim)]
    """
    offset = 0
    layout = {
        "ret": slice(offset, offset + num_horizons),
    }
    offset += num_horizons
    layout["vol"] = slice(offset, offset + num_horizons)
    offset += num_horizons
    layout["dir_logits"] = slice(offset, offset + num_horizons)
    offset += num_horizons
    layout["p_up"] = slice(offset, offset + num_horizons)
    offset += num_horizons
    layout["align_all"] = slice(offset, offset + 1)
    offset += 1
    layout["diff_short_long"] = slice(offset, offset + 1)
    offset += 1
    layout["diff_mid_long"] = slice(offset, offset + 1)
    offset += 1
    layout["conf_long"] = slice(offset, offset + 1)
    offset += 1
    layout["conf_min"] = slice(offset, offset + 1)
    offset += 1
    layout["snapshots"] = slice(offset, offset + snapshot_dim)
    return layout


def _normalize_horizons(
    num_h: int,
    horizons: List[int],
) -> List[int]:
    if len(horizons) == num_h:
        return list(horizons)
    raise ValueError(
        "Horizon count mismatch between model outputs and configured horizons: "
        f"num_model_horizons={num_h} configured_horizons={horizons} configured_count={len(horizons)}. "
        "Set BYBIT_MM_HORIZONS_MS to exactly match model horizons; mismatches are hard errors."
    )


def _validate_fixed_cmssl_horizons(horizons: List[int]) -> List[int]:
    expected_horizons = [250, 500, 1000]
    unique_sorted = sorted(set(horizons))
    if unique_sorted != expected_horizons:
        raise ValueError(
            "CMSSL horizon contract violation: configured horizons must be exactly "
            f"{expected_horizons}, got configured_horizons={horizons} "
            f"(unique_sorted={unique_sorted})."
        )
    return expected_horizons


def _resolve_horizon_index(
    target_ms: int,
    horizons: List[int],
    *,
    label: str,
) -> int:
    if target_ms in horizons:
        return horizons.index(target_ms)
    raise ValueError(
        "Required horizon is not available in configured horizon mapping: "
        f"label={label} target_ms={target_ms} configured_horizons={horizons}."
    )


def _split_weeks(split: Dict[str, Any]) -> list[str]:
    weeks = split.get("weeks")
    if weeks:
        return list(weeks)
    if "week" in split:
        return [split["week"]]
    raise KeyError("split must contain 'week' or 'weeks'")


def load_split_arrays(out_root: str, split: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CMSSL tensors for a split.

    Args:
        out_root: Output root containing CMSSL chunk artifacts.
        split: Split config with either ``week`` (single week key) or ``weeks``
            (list of week keys), plus ``start``/``end`` timestamp bounds.
    """
    weeks = _split_weeks(split)
    x_core_list: List[np.ndarray] = []
    x_aux_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    for week, chunk_idx, ts, x_core, x_aux, y in iter_chunk_batches(out_root):
        if week not in weeks:
            continue
        n_rows = x_core.shape[0]
        if ts is None:
            raise ValueError(
                f"Missing decision timestamps for {week}/chunk{chunk_idx:03d}. "
                "Ensure meta_week.json includes ts_*.npy entries."
            )
        if ts.ndim != 1 or ts.shape[0] != n_rows:
            raise ValueError(
                f"{week}/chunk{chunk_idx:03d} timestamps length mismatch: "
                f"expected {n_rows}, got {ts.shape}"
            )
        _ensure_monotonic(ts, f"{week}/chunk{chunk_idx:03d}")
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


def resolve_test_split(out_root: str, meta: dict) -> Dict[str, Any]:
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
    """Run CMSSL inference over test windowed inputs for offline diagnostics."""
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
    """Run CMSSL inference for batched inputs; empty batches are valid."""
    ret_preds: List[np.ndarray] = []
    vol_preds: List[np.ndarray] = []
    dir_logits_list: List[np.ndarray] = []
    num_h = len(meta["horizons_ms"])
    n = x_core.shape[0]
    if n == 0:
        empty = np.empty((0, num_h), dtype=np.float32)
        return {
            "ret_pred": empty.copy(),
            "vol_pred": empty.copy(),
            "dir_logits": empty.copy(),
        }
    for i in range(0, n, batch_size):
        xc = x_core[i:i + batch_size]
        xa = x_aux[i:i + batch_size]
        ret_pred, vol_pred, dir_logits = cmssl_predict(model, xc, xa, meta, device=device)
        ret_preds.append(ret_pred.detach().cpu().numpy())
        vol_preds.append(vol_pred.detach().cpu().numpy())
        dir_logits_list.append(dir_logits.detach().cpu().numpy())
    return {
        "ret_pred": np.concatenate(ret_preds, axis=0).astype(np.float32, copy=False),
        "vol_pred": np.concatenate(vol_preds, axis=0).astype(np.float32, copy=False),
        "dir_logits": np.concatenate(dir_logits_list, axis=0).astype(np.float32, copy=False),
    }


def _find_week_dir(out_root: Path, week_key: str) -> Path:
    meta = load_global_meta(out_root)
    for wk, _wmeta, wk_dir in iter_week_chunks(out_root, meta=meta):
        if wk == week_key:
            return wk_dir
    raise ValueError(f"Unable to locate week directory for {week_key}")


def _ffill_1d(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if out.size == 0:
        return out
    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        out[:] = 0.0
        return out
    idx = np.where(finite_mask, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    out[~np.isfinite(out)] = 0.0
    return out


def _rolling_std_ignore_nan(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(x)
    x_finite = np.where(finite, x, 0.0)
    x_finite_sq = x_finite * x_finite

    csum = np.cumsum(x_finite)
    csum2 = np.cumsum(x_finite_sq)
    count_csum = np.cumsum(finite.astype(np.int64))

    count = count_csum.copy()
    sums = csum.copy()
    sumsq = csum2.copy()
    if window < n:
        count[window:] -= count_csum[:-window]
        sums[window:] -= csum[:-window]
        sumsq[window:] -= csum2[:-window]

    out = np.full(n, np.nan, dtype=np.float64)
    valid = count > 1
    var_num = sumsq[valid] - (sums[valid] * sums[valid]) / count[valid]
    out[valid] = np.sqrt(np.maximum(var_num / (count[valid] - 1), 0.0))
    return out


def _sanitize_snapshot_features(arr: np.ndarray) -> np.ndarray:
    target_cols = ["best_bid_size", "best_ask_size", "imbalance", "mid_ret_1", "vol_short", "vol_long", "spread_bps"]
    arr = np.asarray(arr).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D snapshot feature array, got shape={arr.shape}.")
    col_idx = {
        name: RAW_SNAPSHOT_FEATURE_COLUMNS.index(name)
        for name in target_cols
        if name in RAW_SNAPSHOT_FEATURE_COLUMNS and RAW_SNAPSHOT_FEATURE_COLUMNS.index(name) < arr.shape[1]
    }
    for idx in col_idx.values():
        col = arr[:, idx].astype(np.float64, copy=False)
        col[~np.isfinite(col)] = np.nan
        col = _ffill_1d(col)
        col[~np.isfinite(col)] = 0.0
        col = col.astype(arr.dtype, copy=False)
        arr[:, idx] = col
    return arr


def _compute_snapshot_feature_matrix(
    snapshot_ts: np.ndarray,
    snapshots: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
    snapshots = np.asarray(snapshots)
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D.")
    if snapshots.ndim != 2 or snapshots.shape[1] != 4:
        raise ValueError("Snapshots must be [N,4] with bid/ask and sizes. Rebuild snapshots.")
    order = np.argsort(snapshot_ts)
    snapshot_ts = snapshot_ts[order]
    snapshots = snapshots[order]
    best_bid = snapshots[:, 0].astype(np.float64)
    best_ask = snapshots[:, 1].astype(np.float64)
    best_bid_size = np.maximum(snapshots[:, 2].astype(np.float64), 0.0)
    best_ask_size = np.maximum(snapshots[:, 3].astype(np.float64), 0.0)
    mid = (best_bid + best_ask) / 2.0
    eps = 1e-9
    imbalance = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size + eps)
    spread_bps = (best_ask - best_bid) / mid * 1e4
    mid_ret_1 = np.log(mid)
    mid_ret_1 = np.concatenate([[np.nan], np.diff(mid_ret_1)])
    vol_short = _rolling_std_ignore_nan(mid_ret_1, SHORT_VOL_WINDOW)
    vol_long = _rolling_std_ignore_nan(mid_ret_1, LONG_VOL_WINDOW)
    features = np.column_stack(
        [best_bid, best_ask, best_bid_size, best_ask_size, imbalance, mid, spread_bps, mid_ret_1, vol_short, vol_long]
    )
    return snapshot_ts, _sanitize_snapshot_features(features)


def load_raw_snapshots(out_root: str, week_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load canonical raw snapshots only (no alternate ingestion fallbacks)."""
    week_dir = _find_week_dir(Path(out_root), week_key)
    canonical_path = week_dir / "snapshots.npz"
    if not canonical_path.exists():
        raise FileNotFoundError("Run offline_snapshots.py first.")

    data = np.load(canonical_path)
    if not {"ts", "snapshots"}.issubset(data.files):
        raise ValueError(
            f"Expected canonical snapshots at {out_root}/{week_key}/snapshots.npz "
            "with fields ts and snapshots. Generate them with offline_snapshots.py."
        )
    snapshots = data["snapshots"]
    if snapshots.ndim != 2 or snapshots.shape[1] != 4:
        raise ValueError(
            f"{canonical_path} has snapshots shape {snapshots.shape}; expected [N,4] (bid, ask, bid_size, ask_size). "
            "Re-run offline_snapshots (1).py to regenerate."
        )
    return data["ts"], snapshots


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
    split_weeks = _split_weeks(test_split)
    if not split_weeks:
        raise ValueError("Test split contains no weeks.")
    split_weeks_label = ",".join(split_weeks)
    start_ms = int(test_split["start"])
    end_ms = int(test_split["end"])
    duration_ms = end_ms - start_ms
    print(
        "[cmssl split:test]",
        f"weeks={split_weeks_label}",
        f"start={_format_ts(start_ms)}",
        f"end={_format_ts(end_ms)}",
        f"duration={_format_duration_ms(duration_ms)}",
    )
    expected_week_ms = 7 * 24 * 60 * 60 * 1000
    expected_half_ms = int(expected_week_ms / 2.0)
    tolerance_ms = 60 * 60 * 1000
    require(abs(duration_ms - expected_half_ms) <= tolerance_ms, (
        f"Test split duration {duration_ms}ms not ~3.5 days."
    ))

    canonical_snapshot_ts_parts: List[np.ndarray] = []
    for week in split_weeks:
        week_snapshot_ts, _snapshots = load_raw_snapshots(out_root, week)
        canonical_snapshot_ts_parts.append(np.asarray(week_snapshot_ts, dtype=np.int64))
    canonical_snapshot_ts = np.concatenate(canonical_snapshot_ts_parts, axis=0)
    canonical_snapshot_ts = np.asarray(canonical_snapshot_ts, dtype=np.int64)
    canonical_snapshot_ts = np.sort(canonical_snapshot_ts)
    filtered = canonical_snapshot_ts[(canonical_snapshot_ts >= start_ms) & (canonical_snapshot_ts < end_ms)]
    if filtered.size == 0:
        raise ValueError("No canonical raw snapshots found inside the CMSSL test split range.")
    _ensure_monotonic(filtered, "Raw snapshot (filtered)")
    print(
        "[raw snapshots:test]",
        f"count={filtered.size}",
        f"start={_format_ts(int(filtered[0]))}",
        f"end={_format_ts(int(filtered[-1]))}",
    )


def align_snapshots_to_decisions(
    snapshot_ts: np.ndarray,
    decision_ts: np.ndarray,
) -> np.ndarray:
    """Return exact snapshot indices for each decision timestamp."""
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D")
    if decision_ts.ndim != 1:
        raise ValueError("decision_ts must be 1D")
    if snapshot_ts.size and np.any(np.diff(snapshot_ts) <= 0):
        raise ValueError("snapshot_ts must be strictly increasing (np.diff(snapshot_ts) > 0)")
    if decision_ts.size and np.any(np.diff(decision_ts) < 0):
        raise ValueError("decision_ts must be monotonically non-decreasing (np.diff(decision_ts) >= 0)")

    index = {int(t): i for i, t in enumerate(snapshot_ts)}
    aligned_idx = np.empty(decision_ts.shape[0], dtype=np.int64)
    missing: List[int] = []
    for i, ts_i in enumerate(decision_ts):
        idx = index.get(int(ts_i))
        if idx is None:
            missing.append(int(ts_i))
            aligned_idx[i] = -1
        else:
            aligned_idx[i] = int(idx)

    if missing:
        sample_count = min(5, len(missing))
        raise ValueError(
            "Snapshot alignment failed; exact timestamp matches missing. "
            f"missing={len(missing)} total={decision_ts.size} "
            f"samples={missing[:sample_count]}. "
            "Run offline_snapshots.py and ensure decision_policy=ob_only; "
            "timestamps must land on snapshot grid"
        )
    return aligned_idx


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
    """Join model outputs and snapshot state into a single feature tensor.

    Per-row layout (excluding environment-only extra state):
      1) ret_pred[h]
      2) vol_pred[h]
      3) dir_logits[h]
      4) p_up[h]
      5) align/conf deltas and confidence metrics:
         - align_all
         - diff_short_long
         - diff_mid_long
         - conf_long
         - conf_min
      6) snapshot features from RAW_SNAPSHOT_FEATURE_COLUMNS
    """
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
    layout = _joined_feature_layout(ret_pred.shape[1], snapshots.shape[1])
    snapshot_spread_col = RAW_SNAPSHOT_FEATURE_COLUMNS.index("spread_bps")
    spread_bps = snapshots[:, snapshot_spread_col]  # use aligned snapshot spread
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
    expected_feature_dim = layout["snapshots"].stop
    if features.shape[1] != expected_feature_dim:
        raise ValueError(
            "join_features layout mismatch: "
            f"features_shape={features.shape} expected_dim={expected_feature_dim}"
        )
    if not np.all(np.isfinite(features)):
        bad_rows = np.where(~np.isfinite(features).all(axis=1))[0]
        sample_rows = bad_rows[:5].tolist()
        raise ValueError(
            "join_features produced non-finite values in feature tensor. "
            f"bad_row_count={len(bad_rows)} sample_rows={sample_rows} features_shape={features.shape}"
        )
    output = {
        "ts": decision_ts,
        "features": features.astype(np.float32),
        "y": y.astype(np.float32),
        "spread_bps": spread_bps.astype(np.float32),
        "snapshots": snapshots.astype(np.float32),
    }
    return output


def build_joined_split(
    out_root: str,
    split: Dict[str, Any],
    model,
    meta: dict,
    device: str,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    week_outputs: List[Dict[str, np.ndarray]] = []
    for wk in _split_weeks(split):
        wk_split = {"week": wk, "start": split["start"], "end": split["end"]}
        try:
            x_core, x_aux, y, ts = load_split_arrays(out_root, wk_split)
        except ValueError as exc:
            if str(exc).startswith("No data found for split"):
                continue
            raise

        if ts.shape[0] == 0:
            continue

        cmssl_out = run_cmssl_inference(
            model,
            meta,
            x_core,
            x_aux,
            batch_size=batch_size,
            device=device,
        )

        # Canonical snapshot flow: load_raw_snapshots(...) ->
        # _compute_snapshot_feature_matrix(...).
        week_snapshot_ts, week_raw_snapshots = load_raw_snapshots(out_root, wk)
        snapshot_ts, snapshots = _compute_snapshot_feature_matrix(
            np.asarray(week_snapshot_ts, dtype=np.int64),
            np.asarray(week_raw_snapshots),
        )
        snapshot_ts = np.asarray(snapshot_ts, dtype=np.int64)
        snapshots = np.asarray(snapshots, dtype=np.float32)

        window_start = int(split["start"])
        window_end = int(split["end"])
        effective_mask = (snapshot_ts >= window_start) & (snapshot_ts < window_end)
        if np.any(effective_mask):
            snapshot_ts = snapshot_ts[effective_mask]
            snapshots = snapshots[effective_mask]

        # Perform exact-match decision/snapshot alignment week-by-week so split
        # boundaries follow the authoritative `weeks` ordering without cross-week
        # ambiguity at week edges.
        snap_idx = align_snapshots_to_decisions(snapshot_ts, ts)
        assert snapshot_ts[snap_idx].shape == ts.shape
        assert np.all(snapshot_ts[snap_idx] == ts)
        aligned_snapshots = snapshots[snap_idx]
        week_outputs.append(join_features(ts, y, cmssl_out, aligned_snapshots, meta))

    if not week_outputs:
        raise ValueError(f"No data found for split {split}")

    out = {
        "ts": np.concatenate([wk["ts"] for wk in week_outputs], axis=0),
        "features": np.concatenate([wk["features"] for wk in week_outputs], axis=0),
        "y": np.concatenate([wk["y"] for wk in week_outputs], axis=0),
        "spread_bps": np.concatenate([wk["spread_bps"] for wk in week_outputs], axis=0),
        "snapshots": np.vstack([wk["snapshots"] for wk in week_outputs]),
    }

    expected_rows = out["ts"].shape[0]
    for key, value in out.items():
        if value.shape[0] != expected_rows:
            raise ValueError(
                "build_joined_split row-count mismatch after weekly concatenation: "
                f"ts_rows={expected_rows} {key}_rows={value.shape[0]}"
            )

    ts_all = out["ts"]
    ts_diff = np.diff(ts_all)
    bad_idx = np.where(ts_diff <= 0)[0]
    if bad_idx.size > 0:
        first_bad = int(bad_idx[0])
        raise ValueError(
            "build_joined_split requires strictly increasing concatenated timestamps "
            "(weeks order is preserved; outputs are not resorted). "
            f"first_bad_index={first_bad} ts_prev={int(ts_all[first_bad])} "
            f"ts_next={int(ts_all[first_bad + 1])} diff={int(ts_diff[first_bad])}"
        )

    return out


def chronological_split(
    data: Dict[str, np.ndarray],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, Dict[str, np.ndarray]]:
    require(abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1.0; got {ratios}")
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
    decision_ts: Optional[np.ndarray] = None


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
        inv_soft_notional: float,
        lambda_inv: float = 0.0,
        lambda_turn: float = 0.0,
        max_inventory_notional: float,
        fill_size: float = 1.0,
        fill_tolerance: float = 1e-6,
        delta_bps_limit: float = 0.0,
        initial_cash: Optional[float] = None,
        obs_norm_state: Optional[dict] = None,
        freeze_obs_norm: bool = False,
    ):
        self.features = batch.features
        self.spread_bps = batch.spread_bps
        self.best_bid = batch.best_bid
        self.best_ask = batch.best_ask
        self.decision_ts = batch.decision_ts
        self.maker_rebate_bps = maker_rebate_bps
        self.taker_fee_bps = taker_fee_bps
        self.allow_taker = allow_taker
        self.taker_threshold = taker_threshold
        self.inventory_penalty = inventory_penalty
        if inv_soft_notional <= 0.0:
            raise ValueError("inv_soft_notional must be > 0 in quote notional (USD).")
        if max_inventory_notional <= 0.0:
            raise ValueError("max_inventory_notional must be > 0 in quote notional (USD).")
        if max_inventory_notional < inv_soft_notional:
            raise ValueError("max_inventory_notional must be >= inv_soft_notional.")
        self.inv_soft_notional = inv_soft_notional
        self.lambda_inv = lambda_inv
        self.lambda_turn = lambda_turn
        self.stack_inventory_penalties = _env_bool("BYBIT_MM_STACK_INVENTORY_PENALTIES", False)
        self.max_inventory_notional = max_inventory_notional
        self.fill_size = fill_size
        self.fill_tolerance = fill_tolerance
        self.delta_bps_limit = float(delta_bps_limit)
        if not np.isfinite(self.delta_bps_limit) or self.delta_bps_limit <= 0.0:
            raise ValueError(
                "delta_bps_limit must be finite and > 0 in basis points (bps)."
            )
        self.initial_cash = (
            float(initial_cash)
            if initial_cash is not None
            else _env_float("BYBIT_MM_INITIAL_CASH", DEFAULT_MM_INITIAL_CASH)
        )
        inventory_notional_scale_raw = os.environ.get(
            "BYBIT_MM_INVENTORY_NOTIONAL_SCALE",
            "",
        ).strip()
        if not inventory_notional_scale_raw:
            raise ValueError(
                "Missing required env var BYBIT_MM_INVENTORY_NOTIONAL_SCALE "
                "(quote notional, USD)."
            )
        try:
            self.inventory_notional_scale = float(inventory_notional_scale_raw)
        except ValueError as exc:
            raise ValueError(
                "BYBIT_MM_INVENTORY_NOTIONAL_SCALE must be a finite float "
                "in quote notional (USD)."
            ) from exc
        if not np.isfinite(self.inventory_notional_scale) or self.inventory_notional_scale <= 0.0:
            raise ValueError(
                "BYBIT_MM_INVENTORY_NOTIONAL_SCALE must be finite and > 0 "
                "in quote notional (USD)."
            )
        self.cash_scale = _env_float("BYBIT_MM_CASH_SCALE", DEFAULT_MM_CASH_SCALE)
        self.time_since_fill_scale = _env_float(
            "BYBIT_MM_TIME_SINCE_FILL_SCALE",
            DEFAULT_MM_TIME_SINCE_FILL_SCALE,
        )
        self.fill_notional_scale = _env_float(
            "BYBIT_MM_FILL_NOTIONAL_SCALE",
            DEFAULT_MM_FILL_NOTIONAL_SCALE,
        )
        self.pnl_notional_scale = _env_float(
            "BYBIT_MM_PNL_NOTIONAL_SCALE",
            DEFAULT_MM_PNL_NOTIONAL_SCALE,
        )
        self.markout_notional_scale = _env_float(
            "BYBIT_MM_MARKOUT_NOTIONAL_SCALE",
            DEFAULT_MM_MARKOUT_NOTIONAL_SCALE,
        )
        self.fill_ema_window_steps = max(
            1,
            _env_int("BYBIT_MM_FILL_EMA_WINDOW_STEPS", DEFAULT_MM_FILL_EMA_WINDOW_STEPS),
        )
        self.fill_ema_alpha = 2.0 / (float(self.fill_ema_window_steps) + 1.0)
        self._baseline_cfg = load_baseline_quote_config()
        self._num_h = _infer_num_horizons(self.features.shape[-1])
        self._horizons_ms = _normalize_horizons(
            self._num_h,
            self._baseline_cfg.horizons_ms,
        )
        self._horizons_ms = _validate_fixed_cmssl_horizons(self._horizons_ms)
        self._vol_horizon_idx = _resolve_horizon_index(
            self._baseline_cfg.vol_horizon_ms,
            self._horizons_ms,
            label="vol",
        )
        self._p250_idx = _resolve_horizon_index(
            250,
            self._horizons_ms,
            label="p250",
        )
        self._p500_idx = _resolve_horizon_index(
            500,
            self._horizons_ms,
            label="p500",
        )
        self._p1000_idx = _resolve_horizon_index(
            1000,
            self._horizons_ms,
            label="p1000",
        )
        print(
            "[mm horizons]",
            f"resolved_horizons_ms={self._horizons_ms}",
            f"vol_ms={self._baseline_cfg.vol_horizon_ms}",
            f"vol_idx={self._vol_horizon_idx}",
            f"p250_idx={self._p250_idx}",
            f"p500_idx={self._p500_idx}",
            f"p1000_idx={self._p1000_idx}",
        )
        self._feature_layout = _joined_feature_layout(self._num_h, len(RAW_SNAPSHOT_FEATURE_COLUMNS))
        self._validate_feature_layout()

        self.n = len(self.spread_bps)
        self.idx = 0
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.total_reward = 0.0
        self.prev_equity = self.initial_cash
        self.time_since_last_fill = self._initial_time_since_last_fill()
        self.avg_entry_price = 0.0
        self.last_maker_buy_notional = 0.0
        self.last_maker_sell_notional = 0.0
        self.last_taker_buy_notional = 0.0
        self.last_taker_sell_notional = 0.0
        self.last_net_fill_notional = 0.0
        self.last_gross_fill_notional = 0.0
        self.ema_net_fill_notional = 0.0
        self.ema_gross_fill_notional = 0.0
        self.ema_maker_buy_markout = 0.0
        self.ema_maker_sell_markout = 0.0
        self._obs_count = 0
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_m2: Optional[np.ndarray] = None
        self._obs_continuous_mask: Optional[np.ndarray] = None
        self.freeze_obs_norm = bool(freeze_obs_norm)
        if obs_norm_state is not None:
            self.set_obs_norm_state(obs_norm_state, freeze=freeze_obs_norm)

    def reset(self, start_idx: int = 0) -> np.ndarray:
        max_start = max(0, self.n - 2)
        if start_idx < 0 or start_idx > max_start:
            raise ValueError(
                f"start_idx out of bounds: start_idx={start_idx} valid=[0, {max_start}]"
            )
        self.idx = int(start_idx)
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.total_reward = 0.0
        # Episode startup semantics: no prior fill is represented by a large sentinel
        # so the feature is distinct from "just filled" (0.0).
        self.time_since_last_fill = self._initial_time_since_last_fill()
        self.avg_entry_price = 0.0
        self.last_maker_buy_notional = 0.0
        self.last_maker_sell_notional = 0.0
        self.last_taker_buy_notional = 0.0
        self.last_taker_sell_notional = 0.0
        self.last_net_fill_notional = 0.0
        self.last_gross_fill_notional = 0.0
        self.ema_net_fill_notional = 0.0
        self.ema_gross_fill_notional = 0.0
        self.ema_maker_buy_markout = 0.0
        self.ema_maker_sell_markout = 0.0
        mid = self._mid_price(self.idx)
        self.prev_equity = self.cash + self.inventory * mid
        return self._build_observation(self.idx)

    def _mid_price(self, idx: int) -> float:
        return float((self.best_bid[idx] + self.best_ask[idx]) / 2.0)

    def _initial_time_since_last_fill(self) -> float:
        # Prefer a startup value near 1.0 after scaling. This signals "no fill yet"
        # at episode start while keeping 0.0 reserved for a fresh fill event.
        if self.time_since_fill_scale > 0.0:
            return float(self.time_since_fill_scale)
        return 1.0

    def _build_observation(self, idx: int) -> np.ndarray:
        mid = self._mid_price(idx)
        inventory_notional_scaled = (
            (self.inventory * mid) / self.inventory_notional_scale if self.inventory_notional_scale else 0.0
        )
        cash_scaled = self.cash / self.cash_scale if self.cash_scale else 0.0
        time_since_last_fill_scaled = (
            self.time_since_last_fill / self.time_since_fill_scale if self.time_since_fill_scale else 0.0
        )
        unrealized_pnl_notional = (
            self.inventory * (mid - self.avg_entry_price) if self.inventory != 0.0 else 0.0
        )
        unrealized_pnl_scaled = (
            unrealized_pnl_notional / self.pnl_notional_scale if self.pnl_notional_scale else 0.0
        )
        # Fill-notional `last_*` fields capture the last non-zero fill aggregates.
        # At reset, `time_since_last_fill` starts at a sentinel for "no prior fill"
        # (scaled value ~1.0). A real fill sets it to 0.0. On no-fill steps, it is
        # incremented by (decision_ts[next_idx] - decision_ts[idx]) / RAW_SNAPSHOT_EXPECTED_STEP_MS,
        # i.e., accumulated in RAW_SNAPSHOT_EXPECTED_STEP_MS-equivalent units rather
        # than fixed "1 snapshot == 1 step" units. Under jitter this keeps intent
        # explicit: ~100ms gaps contribute ~1.0, ~300ms gaps contribute ~3.0.
        # `last_*` values persist on no-fill steps.
        extra = np.array(
            [
                inventory_notional_scaled,
                cash_scaled,
                time_since_last_fill_scaled,
                self.last_maker_buy_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_maker_sell_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_taker_buy_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_taker_sell_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_net_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.last_gross_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.ema_net_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                self.ema_gross_fill_notional / self.fill_notional_scale if self.fill_notional_scale else 0.0,
                unrealized_pnl_scaled,
                self.ema_maker_buy_markout / self.markout_notional_scale if self.markout_notional_scale else 0.0,
                self.ema_maker_sell_markout / self.markout_notional_scale if self.markout_notional_scale else 0.0,
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([self.features[idx].astype(np.float32), extra], axis=0)
        return self._normalize_observation(obs)

    def _validate_feature_layout(self) -> None:
        expected_feature_dim = self._feature_layout["snapshots"].stop
        actual_feature_dim = int(self.features.shape[-1])
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                "MarketMakingEnv feature layout drift detected: "
                f"actual_feature_dim={actual_feature_dim} expected_feature_dim={expected_feature_dim} "
                f"num_horizons={self._num_h} snapshot_dim={len(RAW_SNAPSHOT_FEATURE_COLUMNS)}"
            )

    def get_observation_scaling_config(self) -> Dict[str, float]:
        return {
            "inventory_notional_scale": float(self.inventory_notional_scale),
            "cash_scale": float(self.cash_scale),
            "time_since_fill_scale": float(self.time_since_fill_scale),
            "fill_notional_scale": float(self.fill_notional_scale),
            "pnl_notional_scale": float(self.pnl_notional_scale),
            "markout_notional_scale": float(self.markout_notional_scale),
            "fill_ema_window_steps": int(self.fill_ema_window_steps),
        }

    def _continuous_mask(self, obs_dim: int) -> np.ndarray:
        expected_obs_dim = self._feature_layout["snapshots"].stop + ENV_OBS_EXTRA_STATE_DIM
        if obs_dim != expected_obs_dim:
            raise ValueError(
                "Observation dimension mismatch for normalization mask: "
                f"obs_dim={obs_dim} expected_obs_dim={expected_obs_dim}"
            )
        mask = np.ones(obs_dim, dtype=bool)
        bounded_feature_keys = (
            "p_up",
            "align_all",
            "conf_long",
            "conf_min",
        )
        for key in bounded_feature_keys:
            feature_slice = self._feature_layout[key]
            mask[feature_slice] = False
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

    def get_obs_norm_state(self) -> Dict[str, Any]:
        return {
            "count": int(self._obs_count),
            "mean": self._obs_mean.astype(np.float64).tolist() if self._obs_mean is not None else None,
            "m2": self._obs_m2.astype(np.float64).tolist() if self._obs_m2 is not None else None,
            "continuous_mask": (
                self._obs_continuous_mask.astype(bool).tolist()
                if self._obs_continuous_mask is not None
                else None
            ),
        }

    def set_obs_norm_state(self, state: Dict[str, Any], freeze: bool = True) -> None:
        if not isinstance(state, dict):
            raise ValueError("obs normalization state must be a dictionary.")
        count = int(state.get("count", 0))
        mean_raw = state.get("mean")
        m2_raw = state.get("m2")
        mask_raw = state.get("continuous_mask")
        mean = None if mean_raw is None else np.asarray(mean_raw, dtype=np.float64)
        m2 = None if m2_raw is None else np.asarray(m2_raw, dtype=np.float64)
        mask = None if mask_raw is None else np.asarray(mask_raw, dtype=bool)
        if count < 0:
            raise ValueError(f"obs normalization count must be non-negative, got {count}")
        if (mean is None) ^ (m2 is None):
            raise ValueError("obs normalization state must provide both mean and m2 or neither.")
        if mean is not None and mean.shape != m2.shape:
            raise ValueError(
                f"obs normalization mean/m2 shape mismatch: mean={mean.shape} m2={m2.shape}"
            )
        if mask is not None and mean is not None and mask.shape != mean.shape:
            raise ValueError(
                "obs normalization continuous_mask shape mismatch: "
                f"mask={mask.shape} mean={mean.shape}"
            )
        if count == 0:
            mean = None
            m2 = None
        self._obs_count = count
        self._obs_mean = mean
        self._obs_m2 = m2
        self._obs_continuous_mask = mask
        self.freeze_obs_norm = bool(freeze)

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_continuous_mask is None:
            self._obs_continuous_mask = self._continuous_mask(obs.shape[0])
        normalized = obs.copy()
        if self._obs_count >= 2 and self._obs_mean is not None and self._obs_m2 is not None:
            var = self._obs_m2 / max(self._obs_count - 1, 1)
            std = np.sqrt(np.maximum(var, 1e-6))
            mask = self._obs_continuous_mask
            normalized[mask] = (obs[mask] - self._obs_mean[mask]) / std[mask]
        if not self.freeze_obs_norm:
            self._update_obs_stats(obs)
        return normalized

    def _parse_action(self, action: Any) -> Tuple[float, float, float]:
        """Parse an action into (bid_delta_bps, ask_delta_bps, taker_signal).

        Accepted action formats:
        - Scalar: applies the same delta to bid and ask, with no taker signal.
        - Length-2 sequence: interpreted as (bid_delta_bps, ask_delta_bps), taker=0.
        - Length-3 sequence: interpreted as (bid_delta_bps, ask_delta_bps, taker_signal).
        """
        bid_delta_bps: float
        ask_delta_bps: float
        taker_signal: float

        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) == 3:
                bid_delta_bps = float(action[0])
                ask_delta_bps = float(action[1])
                taker_signal = float(action[2])
            elif len(action) == 2:
                bid_delta_bps = float(action[0])
                ask_delta_bps = float(action[1])
                taker_signal = 0.0
            else:
                raise ValueError(
                    "Action sequence must be length 2 or 3: "
                    "(bid_delta_bps, ask_delta_bps[, taker_signal])."
                )
        elif np.isscalar(action):
            bid_delta_bps = float(action)
            ask_delta_bps = float(action)
            taker_signal = 0.0
        else:
            raise ValueError(
                "Action must be a scalar or (bid_delta_bps, ask_delta_bps[, taker_signal])."
            )

        if not np.all(np.isfinite([bid_delta_bps, ask_delta_bps, taker_signal])):
            raise ValueError(
                "Action components must be finite: "
                f"bid_delta_bps={bid_delta_bps}, ask_delta_bps={ask_delta_bps}, "
                f"taker_signal={taker_signal}"
            )
        return bid_delta_bps, ask_delta_bps, taker_signal

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
        snapshot_offset = self._feature_layout["snapshots"].start
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
        self, bid: float, ask: float, mid: float, bid_delta_bps: float, ask_delta_bps: float
    ) -> Tuple[float, float, float, float]:
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


    def _next_avg_entry_price(
        self,
        prev_inv: float,
        prev_avg: float,
        side: int,
        qty: float,
        price: float,
    ) -> Tuple[float, float]:
        if qty <= 0.0:
            return prev_avg, prev_inv
        signed_qty = float(side) * qty
        new_inv = prev_inv + signed_qty
        if prev_inv == 0.0 or np.sign(prev_inv) == np.sign(signed_qty):
            base_qty = abs(prev_inv)
            total_qty = base_qty + qty
            next_avg = (prev_avg * base_qty + price * qty) / total_qty if total_qty > 0.0 else 0.0
            return float(next_avg), float(new_inv)
        # Trade is against existing position.
        if abs(signed_qty) < abs(prev_inv):
            return float(prev_avg), float(new_inv)
        if abs(signed_qty) == abs(prev_inv):
            return 0.0, 0.0
        return float(price), float(new_inv)

    def _ema_update(self, prev: float, value: float) -> float:
        return (1.0 - self.fill_ema_alpha) * prev + self.fill_ema_alpha * value

    def _compute_penalty(self, mid: float) -> float:
        # Industry convention: apply linear inventory penalty only for breaching
        # an explicit hard inventory cap, measured in quote notional (USD).
        inv_notional = abs(self.inventory * mid)
        penalty = 0.0
        if inv_notional > self.max_inventory_notional:
            penalty += self.inventory_penalty * (inv_notional - self.max_inventory_notional)
        return penalty

    def _combine_inventory_penalties(self, linear_penalty: float, quadratic_penalty: float) -> float:
        # Both terms penalize inventory risk. Default to non-stacking behavior to avoid
        # double-charging the same exposure unless explicitly enabled.
        if self.stack_inventory_penalties:
            return linear_penalty + quadratic_penalty
        return max(linear_penalty, quadratic_penalty)

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
                "inventory_notional": float(abs(self.inventory * mid)),
                "equity": float(equity),
                "delta_equity": 0.0,
                "rebate": 0.0,
                "penalty": 0.0,
                "inv_penalty": 0.0,
                "inventory_excess_notional": 0.0,
                "turnover_penalty": 0.0,
                "mid": float(mid),
                "bid": 0.0,
                "ask": 0.0,
                "maker_buy": 0.0,
                "maker_sell": 0.0,
                "taker_buy": 0.0,
                "taker_sell": 0.0,
                "taker_fee": 0.0,
                "maker_buy_notional": float(self.last_maker_buy_notional),
                "maker_sell_notional": float(self.last_maker_sell_notional),
                "taker_buy_notional": float(self.last_taker_buy_notional),
                "taker_sell_notional": float(self.last_taker_sell_notional),
                "net_fill_notional": float(self.last_net_fill_notional),
                "gross_fill_notional": float(self.last_gross_fill_notional),
                "ema_net_fill_notional": float(self.ema_net_fill_notional),
                "ema_gross_fill_notional": float(self.ema_gross_fill_notional),
                "avg_entry_price": float(self.avg_entry_price),
                "unrealized_pnl_notional": float(self.inventory * (mid - self.avg_entry_price) if self.inventory != 0.0 else 0.0),
                "maker_buy_markout": 0.0,
                "maker_sell_markout": 0.0,
                "ema_maker_buy_markout": float(self.ema_maker_buy_markout),
                "ema_maker_sell_markout": float(self.ema_maker_sell_markout),
            }
            return self._build_observation(self.idx), 0.0, True, info
        bid_delta_bps, ask_delta_bps, taker_signal = self._parse_action(action)
        bid, ask, mid = self._baseline_quotes(self.idx)
        bid, ask, bid_delta_bps, ask_delta_bps = self._apply_deltas(
            bid, ask, mid, bid_delta_bps, ask_delta_bps
        )
        bid, ask = self._enforce_passive(bid, ask, self.idx)
        inv_prev = self.inventory
        maker_buy, maker_sell = self._apply_fills(bid, ask, next_idx)
        taker_buy, taker_sell = self._apply_taker(next_idx, taker_signal)
        best_ask_next = float(self.best_ask[next_idx])
        best_bid_next = float(self.best_bid[next_idx])
        avg_tracker = float(self.avg_entry_price)
        inv_tracker = float(inv_prev)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, 1, maker_buy, bid)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, -1, maker_sell, ask)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, 1, taker_buy, best_ask_next)
        avg_tracker, inv_tracker = self._next_avg_entry_price(inv_tracker, avg_tracker, -1, taker_sell, best_bid_next)
        self.avg_entry_price = avg_tracker if inv_tracker != 0.0 else 0.0
        inv_new = self.inventory
        inv_change = inv_new - inv_prev
        had_fill = maker_buy > 0.0 or maker_sell > 0.0 or taker_buy > 0.0 or taker_sell > 0.0
        if had_fill:
            self.time_since_last_fill = 0.0
        else:
            dt_ms = 1
            if self.decision_ts is not None:
                dt_ms = int(self.decision_ts[next_idx]) - int(self.decision_ts[self.idx])
            dt_ms = max(1, dt_ms)
            self.time_since_last_fill += float(dt_ms) / float(RAW_SNAPSHOT_EXPECTED_STEP_MS)

        mid_next = self._mid_price(next_idx)
        maker_rebate_notional = maker_buy * bid + maker_sell * ask
        rebate = maker_rebate_notional * self.maker_rebate_bps * 1e-4
        taker_notional = taker_buy * best_ask_next + taker_sell * best_bid_next
        taker_fee = taker_notional * self.taker_fee_bps * 1e-4
        self.cash += rebate - taker_fee

        maker_buy_notional = maker_buy * bid
        maker_sell_notional = maker_sell * ask
        taker_buy_notional = taker_buy * best_ask_next
        taker_sell_notional = taker_sell * best_bid_next
        buy_notional_total = maker_buy_notional + taker_buy_notional
        sell_notional_total = maker_sell_notional + taker_sell_notional
        net_fill_notional = buy_notional_total - sell_notional_total
        gross_fill_notional = buy_notional_total + sell_notional_total
        maker_buy_markout = (mid_next - bid) * maker_buy if maker_buy > 0.0 else 0.0
        maker_sell_markout = (ask - mid_next) * maker_sell if maker_sell > 0.0 else 0.0

        if maker_buy > 0.0:
            self.last_maker_buy_notional = maker_buy_notional
        if maker_sell > 0.0:
            self.last_maker_sell_notional = maker_sell_notional
        if taker_buy > 0.0:
            self.last_taker_buy_notional = taker_buy_notional
        if taker_sell > 0.0:
            self.last_taker_sell_notional = taker_sell_notional
        # Channel-specific last_* tracks the last non-zero event for that channel,
        # while net/gross track the last step with any fill.
        if had_fill:
            self.last_net_fill_notional = net_fill_notional
            self.last_gross_fill_notional = gross_fill_notional
        self.ema_net_fill_notional = self._ema_update(self.ema_net_fill_notional, net_fill_notional)
        self.ema_gross_fill_notional = self._ema_update(self.ema_gross_fill_notional, gross_fill_notional)
        # EMA of conditional markout given maker fill.
        # Interpretable as adverse-selection quality (not maker-fill activity intensity).
        if maker_buy > 0.0:
            self.ema_maker_buy_markout = self._ema_update(self.ema_maker_buy_markout, maker_buy_markout)
        if maker_sell > 0.0:
            self.ema_maker_sell_markout = self._ema_update(self.ema_maker_sell_markout, maker_sell_markout)
        equity = self.cash + self.inventory * mid_next
        delta_equity = equity - self.prev_equity
        penalty = self._compute_penalty(mid_next)
        # Quadratic regularizer uses quote notional inventory with a dead-zone.
        inv_notional = abs(inv_new * mid_next)
        excess_notional = max(0.0, inv_notional - self.inv_soft_notional)
        inv_penalty = (
            self.lambda_inv * (excess_notional / self.inv_soft_notional) ** 2
            if self.inv_soft_notional > 0.0
            else 0.0
        )
        inventory_penalty_total = self._combine_inventory_penalties(penalty, inv_penalty)
        turnover_notional = maker_rebate_notional + taker_notional
        turnover_penalty = self.lambda_turn * turnover_notional
        reward = delta_equity - inventory_penalty_total - turnover_penalty

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
            "inventory_notional": float(inv_notional),
            "equity": float(equity),
            "delta_equity": float(delta_equity),
            "rebate": float(rebate),
            "taker_fee": float(taker_fee),
            "penalty": float(penalty),
            "inv_penalty": float(inv_penalty),
            "inventory_excess_notional": float(excess_notional),
            "inventory_penalty_total": float(inventory_penalty_total),
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
            "maker_buy_notional": float(maker_buy_notional),
            "maker_sell_notional": float(maker_sell_notional),
            "taker_buy_notional": float(taker_buy_notional),
            "taker_sell_notional": float(taker_sell_notional),
            "net_fill_notional": float(net_fill_notional),
            "gross_fill_notional": float(gross_fill_notional),
            "ema_net_fill_notional": float(self.ema_net_fill_notional),
            "ema_gross_fill_notional": float(self.ema_gross_fill_notional),
            "avg_entry_price": float(self.avg_entry_price),
            "unrealized_pnl_notional": float(self.inventory * (mid_next - self.avg_entry_price) if self.inventory != 0.0 else 0.0),
            "maker_buy_markout": float(maker_buy_markout),
            "maker_sell_markout": float(maker_sell_markout),
            "ema_maker_buy_markout": float(self.ema_maker_buy_markout),
            "ema_maker_sell_markout": float(self.ema_maker_sell_markout),
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
    rollout_horizon: int = 2048
    rollouts_per_epoch: int = 1
    randomize_rollout_start: bool = True


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminals: torch.Tensor,
    gamma: float,
    lam: float,
):
    # `terminals` only marks true environment terminations. Rollout truncation
    # (e.g., horizon cutoff) still bootstraps from `next_values`.
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - terminals[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_market_rollout(
    env: MarketMakingEnv,
    model: MarketPolicyValueNet,
    device: str,
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
    horizon: int = 2048,
    rollouts_per_epoch: int = 1,
    randomize_start: bool = True,
) -> Dict[str, torch.Tensor]:
    # Canonical action space for rollout + PPO is the *env action space*.
    # The policy predicts normalized parameters, then we apply a fixed affine
    # transform (scale-only) to mean/std so sampling, env stepping, and stored
    # actions/log-probs all refer to the exact same (scaled) action tensor.
    obs_list = []
    action_list = []
    logp_list = []
    value_list = []
    next_value_list = []
    reward_list = []
    terminated_list = []
    truncated_list = []
    done_list = []

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if rollouts_per_epoch <= 0:
        raise ValueError(f"rollouts_per_epoch must be positive, got {rollouts_per_epoch}")

    max_start = max(0, env.n - 2)
    for _ in range(rollouts_per_epoch):
        start_idx = int(np.random.randint(0, max_start + 1)) if randomize_start else 0
        obs = env.reset(start_idx=start_idx)
        done = False
        steps = 0
        while not done and steps < horizon:
            obs_cpu = torch.from_numpy(obs).float()
            obs_t = obs_cpu.to(device)
            mean, log_std, value = model(obs_t.unsqueeze(0))
            std = log_std.exp()

            action_scale = torch.ones_like(mean)
            action_scale[..., :2] = delta_scale
            if action_scale.shape[-1] >= 3:
                action_scale[..., 2] = taker_scale

            mean_env = mean * action_scale
            std_env = std * action_scale
            dist_env = torch.distributions.Normal(mean_env, std_env)
            action_env = dist_env.sample()
            logp_env = dist_env.log_prob(action_env).sum(dim=-1)

            next_obs, reward, env_done, _info = env.step(action_env.squeeze(0).cpu().numpy())
            steps += 1
            terminated = bool(env_done)
            # Truncation means the rollout horizon ended; it is not a true
            # environment terminal state and should continue to bootstrap.
            truncated = (not terminated) and (steps >= horizon)
            done = terminated or truncated

            if terminated:
                next_value = torch.tensor(0.0, dtype=torch.float32)
            else:
                with torch.no_grad():
                    next_obs_t = torch.from_numpy(next_obs).float().to(device)
                    _next_mean, _next_log_std, next_value_t = model(next_obs_t.unsqueeze(0))
                    next_value = next_value_t.squeeze(0).detach().cpu()

            obs_list.append(obs_cpu)
            action_list.append(action_env.squeeze(0).detach().cpu())
            logp_list.append(logp_env.squeeze(0).detach().cpu())
            value_list.append(value.squeeze(0).detach().cpu())
            next_value_list.append(next_value)
            reward_list.append(torch.tensor(reward, dtype=torch.float32))
            terminated_list.append(torch.tensor(float(terminated), dtype=torch.float32))
            truncated_list.append(torch.tensor(float(truncated), dtype=torch.float32))
            done_list.append(torch.tensor(float(done), dtype=torch.float32))
            obs = next_obs

    return {
        "obs": torch.stack(obs_list),
        "actions": torch.stack(action_list),
        "logp": torch.stack(logp_list),
        "values": torch.stack(value_list),
        "next_values": torch.stack(next_value_list),
        "rewards": torch.stack(reward_list),
        "terminated": torch.stack(terminated_list),
        "truncated": torch.stack(truncated_list),
        "dones": torch.stack(done_list),
    }


def ppo_update_market(
    model: MarketPolicyValueNet,
    optimizer: optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: PPOConfig,
    device: str,
    delta_scale: float = 1.0,
    taker_scale: float = 1.0,
):
    # PPO loss is computed in the same env action space used during rollout.
    # Stored rollout["actions"]/rollout["logp"] are scaled env actions/log-probs,
    # so we rebuild the Normal distribution in env scale before ratio/log-prob.
    obs = rollout["obs"]
    actions = rollout["actions"]
    old_logp = rollout["logp"].detach()
    values = rollout["values"].detach()
    next_values = rollout["next_values"].detach()
    rewards = rollout["rewards"]
    terminals = rollout["terminated"]

    advantages, returns = compute_gae(rewards, values, next_values, terminals, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = torch.arange(n)
    for _ in range(config.update_epochs):
        perm = indices[torch.randperm(n)]
        for start in range(0, n, config.batch_size):
            mb_idx = perm[start:start + config.batch_size]
            mb_obs = obs[mb_idx].to(device)
            mb_actions = actions[mb_idx].to(device)
            mb_old_logp = old_logp[mb_idx].to(device)
            mb_advantages = advantages[mb_idx].to(device)
            mb_returns = returns[mb_idx].to(device)

            mean, log_std, value = model(mb_obs)
            std = log_std.exp()

            action_scale = torch.ones_like(mean)
            action_scale[..., :2] = delta_scale
            if action_scale.shape[-1] >= 3:
                action_scale[..., 2] = taker_scale

            mean_env = mean * action_scale
            std_env = std * action_scale
            dist_env = torch.distributions.Normal(mean_env, std_env)
            logp = dist_env.log_prob(mb_actions).sum(dim=-1)
            ratio = torch.exp(logp - mb_old_logp)
            clip_adv = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * mb_advantages
            policy_loss = -(torch.min(ratio * mb_advantages, clip_adv)).mean()
            value_loss = nn.functional.mse_loss(value, mb_returns)
            entropy_loss = dist_env.entropy().sum(dim=-1).mean()
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


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute max drawdown as the largest peak-to-trough equity decline."""
    if equity_curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = np.divide(
        peak - equity_curve,
        peak,
        out=np.zeros_like(equity_curve),
        where=peak != 0,
    )
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
            horizon=config.rollout_horizon,
            rollouts_per_epoch=config.rollouts_per_epoch,
            randomize_start=config.randomize_rollout_start,
        )
        ppo_update_market(
            model,
            optimizer,
            rollout,
            config,
            device,
            delta_scale=delta_scale,
            taker_scale=taker_scale,
        )
        if (epoch + 1) % config.val_every == 0:
            # Keep validation normalization aligned with training normalization at
            # checkpoint-selection time; otherwise validation Sharpe is not
            # comparable across epochs.
            val_env.set_obs_norm_state(train_env.get_obs_norm_state(), freeze=True)
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
                    val_report = {
                        k: v for k, v in report.items() if k not in {"equity_curve"}
                    }  # Prevent oversized checkpoints from embedding full curves.
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "policy_state_dict": model.policy_net.state_dict(),
                            "value_state_dict": model.value_net.state_dict(),
                            "hidden_dims": tuple(config.policy_hidden),
                            "action_dim": model.log_std.shape[0],
                            "config": config.__dict__,
                            "val_report": val_report,
                            "obs_norm_state": train_env.get_obs_norm_state(),
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
    decision_ts = split.get("ts")
    if decision_ts is not None:
        decision_ts = np.asarray(decision_ts, dtype=np.int64)
        if decision_ts.ndim != 1:
            raise ValueError("split['ts'] must be a 1D array.")
        if decision_ts.shape[0] != split["features"].shape[0]:
            raise ValueError(
                "split['ts'] length mismatch: "
                f"expected {split['features'].shape[0]}, got {decision_ts.shape[0]}"
            )
        _ensure_monotonic(decision_ts, "split")
    return MarketMakingBatch(
        features=split["features"],
        spread_bps=split["spread_bps"],
        best_bid=best_bid,
        best_ask=best_ask,
        decision_ts=decision_ts,
    )


def _resolve_eval_step_ms(env: MarketMakingEnv, steps: int) -> Dict[str, Any]:
    fallback_step_ms = _env_float("BYBIT_MM_SNAPSHOT_STEP_MS", RAW_SNAPSHOT_EXPECTED_STEP_MS)
    decision_ts = env.decision_ts
    if decision_ts is None or decision_ts.size < 2 or steps <= 0:
        return {
            "step_ms": float(fallback_step_ms),
            "source": "env_var_fallback",
            "diff_count": 0,
        }

    eval_count = min(int(steps) + 1, int(decision_ts.size))
    diffs = np.diff(decision_ts[:eval_count])
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return {
            "step_ms": float(fallback_step_ms),
            "source": "env_var_fallback",
            "diff_count": 0,
        }

    return {
        "step_ms": float(np.median(positive_diffs)),
        "source": "decision_ts_median_diff",
        "diff_count": int(positive_diffs.size),
    }


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
        maker_buy = abs(float(info["maker_buy"]))
        maker_sell = abs(float(info["maker_sell"]))
        taker_buy = abs(float(info["taker_buy"]))
        taker_sell = abs(float(info["taker_sell"]))
        step_qty = maker_buy + maker_sell + taker_buy + taker_sell
        turnover_qty += step_qty
        maker_notional = maker_buy * float(info.get("bid", 0.0)) + maker_sell * float(info.get("ask", 0.0))
        step_taker_notional = taker_buy * float(env.best_ask[env.idx]) + taker_sell * float(env.best_bid[env.idx])
        step_notional = maker_notional + step_taker_notional
        turnover_notional += step_notional
        taker_notional += step_taker_notional
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
    cadence = _resolve_eval_step_ms(env, steps)
    step_ms = float(cadence["step_ms"])
    steps_per_year = _steps_per_year_from_snapshot_ms(step_ms)
    sharpe = compute_sharpe(returns, steps_per_year)
    max_drawdown = compute_max_drawdown(equity_arr)
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
        "cadence": {
            "step_ms": step_ms,
            "steps_per_year": float(steps_per_year),
            "source": cadence["source"],
            "diff_count": cadence["diff_count"],
        },
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
    require_checkpoint: bool = False,
) -> Optional[MarketPolicyNet]:
    """Load a deterministic market policy (mean-action inference only)."""
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    if not path.exists():
        if require_checkpoint:
            raise FileNotFoundError(f"Market policy checkpoint not found: {ckpt_path}")
        warnings.warn(
            f"Market policy checkpoint not found: {ckpt_path}. Falling back to baseline policy.",
            RuntimeWarning,
        )
        return None
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
    maker_rebate_bps = float(os.environ.get("BYBIT_MM_MAKER_REBATE_BPS", "0.0"))
    inventory_penalty = float(os.environ.get("BYBIT_MM_INVENTORY_PENALTY", "0.0"))
    # Inventory/turnover penalties applied inside MarketMakingEnv.step().
    # Required inventory risk knobs (quote notional, USD):
    #   BYBIT_MM_INV_SOFT_NOTIONAL
    #   BYBIT_MM_MAX_INV_NOTIONAL
    #   BYBIT_MM_INVENTORY_NOTIONAL_SCALE
    # Required delta control knob (basis points, bps):
    #   BYBIT_MM_DELTA_BPS_LIMIT
    # Migration: BYBIT_MM_NOTIONAL_SCALE removed; set BYBIT_MM_INVENTORY_NOTIONAL_SCALE explicitly.
    # Units are quote notional (USD), not base units.
    inv_soft_notional_str = os.environ.get("BYBIT_MM_INV_SOFT_NOTIONAL", "").strip()
    if not inv_soft_notional_str:
        raise ValueError(
            "Missing required env var BYBIT_MM_INV_SOFT_NOTIONAL (quote notional, USD)."
        )
    inv_soft_notional = float(inv_soft_notional_str)
    lambda_inv = float(os.environ.get("BYBIT_MM_LAMBDA_INV", "0.0"))
    lambda_turn = float(os.environ.get("BYBIT_MM_LAMBDA_TURN", "0.0"))
    # Hard inventory cap in quote notional (USD).
    max_inventory_notional_str = os.environ.get("BYBIT_MM_MAX_INV_NOTIONAL", "").strip()
    if not max_inventory_notional_str:
        raise ValueError(
            "Missing required env var BYBIT_MM_MAX_INV_NOTIONAL (quote notional, USD)."
        )
    max_inventory_notional = float(max_inventory_notional_str)
    fill_size = float(os.environ.get("BYBIT_MM_FILL_SIZE", "1.0"))
    fill_tolerance = float(os.environ.get("BYBIT_MM_FILL_TOLERANCE", "1e-6"))
    delta_scale = float(os.environ.get("BYBIT_MM_DELTA_SCALE", "1.0"))
    taker_scale = float(os.environ.get("BYBIT_MM_TAKER_SCALE", "1.0"))
    allow_taker = os.environ.get("BYBIT_MM_ALLOW_TAKER", "true").strip().lower() in {"1", "true", "yes", "y"}
    taker_fee_bps = float(os.environ.get("BYBIT_MM_TAKER_FEE_BPS", str(DEFAULT_MM_TAKER_FEE_BPS)))
    taker_threshold = float(os.environ.get("BYBIT_MM_TAKER_THRESHOLD", str(DEFAULT_MM_TAKER_THRESHOLD)))
    delta_bps_limit_str = os.environ.get("BYBIT_MM_DELTA_BPS_LIMIT", "").strip()
    if not delta_bps_limit_str:
        raise ValueError(
            "Missing required env var BYBIT_MM_DELTA_BPS_LIMIT (basis points, bps)."
        )
    try:
        delta_bps_limit = float(delta_bps_limit_str)
    except ValueError as exc:
        raise ValueError(
            "BYBIT_MM_DELTA_BPS_LIMIT must be a finite float in basis points (bps)."
        ) from exc
    if not np.isfinite(delta_bps_limit) or delta_bps_limit <= 0.0:
        raise ValueError(
            "BYBIT_MM_DELTA_BPS_LIMIT must be finite and > 0 in basis points (bps)."
        )
    if inv_soft_notional <= 0.0:
        raise ValueError("BYBIT_MM_INV_SOFT_NOTIONAL must be > 0 (quote notional, USD).")
    if max_inventory_notional <= 0.0:
        raise ValueError("BYBIT_MM_MAX_INV_NOTIONAL must be > 0 (quote notional, USD).")
    if max_inventory_notional < inv_soft_notional:
        raise ValueError("BYBIT_MM_MAX_INV_NOTIONAL must be >= BYBIT_MM_INV_SOFT_NOTIONAL.")
    if lambda_inv > 0.0 and inv_soft_notional > 0.0:
        print(
            "[mm config warning]",
            "Inventory penalties now use USD notional units; retune",
            "BYBIT_MM_INVENTORY_PENALTY/BYBIT_MM_LAMBDA_INV if needed.",
            f"inv_soft_notional={inv_soft_notional}",
            f"lambda_inv={lambda_inv}",
        )

    mm_train_env = MarketMakingEnv(
        mm_train_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=allow_taker,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft_notional=inv_soft_notional,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory_notional=max_inventory_notional,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
        delta_bps_limit=delta_bps_limit,
    )
    mm_obs = mm_train_env.reset()
    mm_obs_dim = mm_obs.shape[0]
    print("[mm obs scaling]", json.dumps(mm_train_env.get_observation_scaling_config(), sort_keys=True))
    mm_val_env = MarketMakingEnv(
        mm_val_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=allow_taker,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft_notional=inv_soft_notional,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory_notional=max_inventory_notional,
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
        inv_soft_notional=inv_soft_notional,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory_notional=max_inventory_notional,
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
        rollout_horizon=int(os.environ.get("BYBIT_MM_PPO_ROLLOUT_HORIZON", "2048")),
        rollouts_per_epoch=int(os.environ.get("BYBIT_MM_PPO_ROLLOUTS_PER_EPOCH", "1")),
        randomize_rollout_start=os.environ.get("BYBIT_MM_PPO_RANDOMIZE_START", "true").strip().lower()
        in {"1", "true", "yes", "y", "on"},
    )
    if np.isnan(mm_ppo_config.max_drawdown_guard):
        mm_ppo_config.max_drawdown_guard = None
    mm_best_ckpt = Path(os.environ.get("BYBIT_MM_PPO_BEST_CKPT", Path(out_root) / "mm_ppo_best.pt"))
    require_rl_ckpt = _env_bool("BYBIT_MM_REQUIRE_RL_CKPT", False)
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
    train_obs_norm_state = mm_train_env.get_obs_norm_state()
    mm_val_env.set_obs_norm_state(train_obs_norm_state, freeze=True)
    mm_test_env.set_obs_norm_state(train_obs_norm_state, freeze=True)

    baseline_env = MarketMakingEnv(
        mm_test_batch,
        maker_rebate_bps=maker_rebate_bps,
        taker_fee_bps=taker_fee_bps,
        allow_taker=False,
        taker_threshold=taker_threshold,
        inventory_penalty=inventory_penalty,
        inv_soft_notional=inv_soft_notional,
        lambda_inv=lambda_inv,
        lambda_turn=lambda_turn,
        max_inventory_notional=max_inventory_notional,
        fill_size=fill_size,
        fill_tolerance=fill_tolerance,
        delta_bps_limit=delta_bps_limit,
        obs_norm_state=train_obs_norm_state,
        freeze_obs_norm=True,
    )
    baseline_metrics = evaluate_market_making(baseline_env, lambda _obs: (0.0, 0.0, 0.0))

    mm_policy_path = os.environ.get("BYBIT_MM_RL_CKPT", "").strip() or str(mm_best_ckpt)
    resolved_mm_policy_path = str(Path(mm_policy_path).expanduser().resolve()) if mm_policy_path else None

    if resolved_mm_policy_path is None:
        mm_policy = None
        rl_policy_reason = "no path provided"
    elif not Path(resolved_mm_policy_path).exists():
        missing_msg = (
            f"[mm eval] no checkpoint saved/found at {resolved_mm_policy_path}; "
            "using baseline deltas for RL run."
        )
        if require_rl_ckpt:
            raise FileNotFoundError(missing_msg)
        warnings.warn(missing_msg, RuntimeWarning)
        mm_policy = None
        rl_policy_reason = "missing checkpoint"
    else:
        mm_policy = load_market_policy(
            mm_obs_dim,
            device=device,
            ckpt_path=resolved_mm_policy_path,
            require_checkpoint=require_rl_ckpt,
        )
        rl_policy_reason = "loaded" if mm_policy is not None else "missing checkpoint"

    if mm_policy is None:
        if rl_policy_reason == "no path provided":
            print("[mm eval] no policy path provided; using baseline deltas for RL run.")
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
        "mm_obs_scaling": mm_train_env.get_observation_scaling_config(),
        "mm_baseline": baseline_metrics,
        "mm_rl": rl_metrics,
        "mm_rl_policy_loaded": {
            "loaded": rl_policy_loaded,
            "reason": rl_policy_reason,
            "path": resolved_mm_policy_path,
            "require_checkpoint": require_rl_ckpt,
        },
    }


if __name__ == "__main__":
    out_root = os.environ.get("BYBIT_OUT_ROOT", "").strip()
    ckpt_path = os.environ.get("BYBIT_CMSSL_CKPT", "").strip()
    device = os.environ.get("BYBIT_DEVICE", "cuda")
    ppo_epochs = _resolve_ppo_epochs(10)
    run_cmssl_test_window = os.environ.get("BYBIT_RUN_CMSSL_TEST_WINDOW", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    if not out_root or not ckpt_path:
        raise SystemExit("Set BYBIT_OUT_ROOT and BYBIT_CMSSL_CKPT before running.")

    _set_seed_from_env()

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
    print("[mm obs scaling]", report["mm_obs_scaling"])
    print("[mm baseline]", report["mm_baseline"])
    print("[mm rl]", report["mm_rl"])
    if run_cmssl_test_window:
        print("[cmssl test window] running windowed inference for diagnostics.")
        test_window_report = run_cmssl_test_window_inference(out_root, ckpt_path, device=device)
        print("[cmssl test window] completed", json.dumps({"horizons_ms": test_window_report["horizons_ms"]}))
