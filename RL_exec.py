import torch
import numpy as np
from pathlib import Path
from CMSSL17 import SAMBA, ModelArgs, DMODEL, MAMBA_LAYERS, LOOKBACK
from offline_tokens import iter_week_chunks, load_global_meta

def load_cmssl(out_root: str, ckpt_path: str, device="cuda"):
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
def cmssl_predict(model, x_core, x_aux, meta, device="cuda"):
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
            ts = np.load(week_dir / files["ts"])
            yield week, int(entry.get("chunk", 0)), ts, x_core, x_aux, y


def _decision_ts_bounds(week_key: str, week_meta: dict) -> tuple[int, int]:
    ts_range = week_meta.get("decision_ts_range")
    assert ts_range, f"week {week_key} missing decision_ts_range in meta_week.json"
    ts_min = int(ts_range["min"])
    ts_max = int(ts_range["max"])
    assert ts_min < ts_max, f"week {week_key} has invalid decision_ts_range: {ts_range}"
    return ts_min, ts_max


def build_two_week_time_splits(out_root: str) -> dict:
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
    week2_half = week2_span / 2.0
    expected_half_ms = 3.5 * 24 * 60 * 60 * 1000
    tolerance_ms = 60 * 60 * 1000
    assert abs(week2_half - expected_half_ms) <= tolerance_ms, (
        f"week2 half span {week2_half:.0f}ms not ~3.5 days"
    )

    week2_mid = int(week2_min + week2_half)
    return {
        "train": {"week": week1_key, "start": week1_min, "end": week1_max},
        "val": {"week": week2_key, "start": week2_min, "end": week2_mid},
        "test": {"week": week2_key, "start": week2_mid, "end": week2_max},
    }


def spread_bps_from_vol_pred(vol_pred, spread_mult=1.0):
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
