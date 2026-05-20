from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _extract_linear_coef(model: Any) -> np.ndarray:
    for candidate in (model, getattr(model, "estimator", None), getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "coef_"):
            return np.asarray(getattr(candidate, "coef_"), dtype=np.float64).reshape(-1)
    raise ValueError(f"Unable to extract linear coefficients from model type={type(model).__name__}")


def _build_raw_linear_extracted_names(base_names: list[str], blocks: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for block in blocks:
        for bi, base in enumerate(base_names):
            rows.append(
                {
                    "raw_extracted_index": idx,
                    "base_feature_index": bi,
                    "base_feature_name": base,
                    "block_name": block,
                    "extracted_feature_name": f"{base}:{block}",
                }
            )
            idx += 1
    return rows


def _base_names_from_meta(meta: dict[str, Any]) -> list[str]:
    if isinstance(meta.get("feature_names"), list):
        return list(meta.get("feature_names") or []) + list(meta.get("aux_feature_names") or [])
    if isinstance(meta.get("raw_feature_names"), list):
        return list(meta.get("raw_feature_names") or [])
    out = list(meta.get("core_feature_names") or []) + list(meta.get("aux_feature_names") or [])
    if out:
        return out
    raise ValueError("Unable to resolve base names from meta.json")


def _parse_keep_indices(pb) -> np.ndarray:
    km = np.asarray(pb.keep_mask)
    return np.flatnonzero(km) if km.dtype == bool else np.asarray(km, dtype=np.int64).reshape(-1)


def build_flat_importance_df(*, extracted_rows: list[dict], kept_indices: np.ndarray, dir_coefs: list[np.ndarray], mag_up_coefs: list[np.ndarray], mag_down_coefs: list[np.ndarray]) -> pd.DataFrame:
    flat = []
    for j in range(len(kept_indices)):
        src = extracted_rows[int(kept_indices[j])]
        da = np.abs([dir_coefs[0][j], dir_coefs[1][j], dir_coefs[2][j]])
        ua = np.abs([mag_up_coefs[0][j], mag_up_coefs[1][j], mag_up_coefs[2][j]])
        na = np.abs([mag_down_coefs[0][j], mag_down_coefs[1][j], mag_down_coefs[2][j]])
        allv = np.concatenate([da, ua, na])
        flat.append({
            "model_coef_index": j,
            **src,
            "dir_coef_1000ms": float(dir_coefs[2][j]),
            "dir_abs_coef_200ms": float(da[0]), "dir_abs_coef_500ms": float(da[1]), "dir_abs_coef_1000ms": float(da[2]),
            "mag_up_abs_coef_200ms": float(ua[0]), "mag_up_abs_coef_500ms": float(ua[1]), "mag_up_abs_coef_1000ms": float(ua[2]),
            "mag_down_abs_coef_200ms": float(na[0]), "mag_down_abs_coef_500ms": float(na[1]), "mag_down_abs_coef_1000ms": float(na[2]),
            "dir_abs_coef_max": float(np.max(da)),
            "dir_abs_coef_mean": float(np.mean(da)),
            "mag_up_abs_coef_max": float(np.max(ua)),
            "mag_up_abs_coef_mean": float(np.mean(ua)),
            "mag_down_abs_coef_max": float(np.max(na)),
            "mag_down_abs_coef_mean": float(np.mean(na)),
            "mag_abs_coef_max": float(np.max(np.concatenate([ua, na]))),
            "mag_abs_coef_mean": float(np.mean(np.concatenate([ua, na]))),
            "all_abs_coef_max": float(np.max(allv)),
            "all_abs_coef_mean": float(np.mean(allv)),
        })
    return pd.DataFrame(flat)


def aggregate_importance_by_base(flat_df: pd.DataFrame, base_names: list[str], blocks: list[str]) -> pd.DataFrame:
    rows = []
    for name, g in flat_df.groupby("base_feature_name"):
        d = g[["dir_abs_coef_200ms", "dir_abs_coef_500ms", "dir_abs_coef_1000ms"]].to_numpy().reshape(-1)
        u = g[["mag_up_abs_coef_200ms", "mag_up_abs_coef_500ms", "mag_up_abs_coef_1000ms", "mag_down_abs_coef_200ms", "mag_down_abs_coef_500ms", "mag_down_abs_coef_1000ms"]].to_numpy().reshape(-1)
        rows.append({
            "base_feature_name": name,
            "n_kept_columns": int(len(g)),
            "n_total_columns": int(len(blocks)),
            "base_feature_index": int(base_names.index(name)),
            "dir_importance_l2": float(np.sqrt(np.sum(d ** 2))),
            "mag_importance_l2": float(np.sqrt(np.sum(u ** 2))),
            "all_importance_l2": float(np.sqrt(np.sum(d ** 2) + np.sum(u ** 2))),
            "dir_importance_1000ms_l2": float(np.sqrt(np.sum(g["dir_abs_coef_1000ms"].to_numpy() ** 2))),
            "mag_importance_1000ms_l2": float(np.sqrt(np.sum(g[["mag_up_abs_coef_1000ms", "mag_down_abs_coef_1000ms"]].to_numpy() ** 2))),
        })
    out = pd.DataFrame(rows)
    out["all_importance_1000ms_l2"] = np.sqrt(out["dir_importance_1000ms_l2"] ** 2 + out["mag_importance_1000ms_l2"] ** 2)
    for c in ("dir", "mag", "all"):
        denom = float(out[f"{c}_importance_l2"].sum()) or 1.0
        out[f"{c}_importance_l2_share"] = out[f"{c}_importance_l2"] / denom
        denom_1s = float(out[f"{c}_importance_1000ms_l2"].sum()) or 1.0
        out[f"{c}_importance_1000ms_l2_share"] = out[f"{c}_importance_1000ms_l2"] / denom_1s
    return out


def aggregate_importance_by_block(flat_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in flat_df.groupby("block_name"):
        d = g[["dir_abs_coef_200ms", "dir_abs_coef_500ms", "dir_abs_coef_1000ms"]].to_numpy().reshape(-1)
        u = g[["mag_up_abs_coef_200ms", "mag_up_abs_coef_500ms", "mag_up_abs_coef_1000ms", "mag_down_abs_coef_200ms", "mag_down_abs_coef_500ms", "mag_down_abs_coef_1000ms"]].to_numpy().reshape(-1)
        out.append({"block_name": name, "n_kept_columns": int(len(g)), "dir_importance_l2": float(np.sqrt(np.sum(d**2))), "mag_importance_l2": float(np.sqrt(np.sum(u**2))), "all_importance_l2": float(np.sqrt(np.sum(d**2) + np.sum(u**2)))})
    out_df = pd.DataFrame(out)
    for c in ("dir", "mag", "all"):
        denom = float(out_df[f"{c}_importance_l2"].sum()) or 1.0
        out_df[f"{c}_importance_l2_share"] = out_df[f"{c}_importance_l2"] / denom
    return out_df


def add_low_importance_flags(base_df: pd.DataFrame, *, low_share: float, low_dir_share: float, low_mag_share: float, coef_eps: float, flat_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    out["low_importance_candidate"] = (out["all_importance_l2_share"] <= low_share) & (out["dir_importance_l2_share"] <= low_dir_share) & (out["mag_importance_l2_share"] <= low_mag_share)
    absmax = flat_df.groupby("base_feature_name")["all_abs_coef_max"].max()
    out["zero_or_near_zero_all_heads"] = out["base_feature_name"].map(absmax <= coef_eps).fillna(False)
    return out


def select_ablation_groups(*, base_df: pd.DataFrame, low_df: pd.DataFrame, top_n: int, low_n: int, groups_spec: str, all_base: bool) -> list[str]:
    if all_base:
        return list(base_df["base_feature_name"])
    selected: list[str] = []
    for token in [t.strip() for t in groups_spec.split(",") if t.strip()]:
        if token == "top_all":
            selected.extend(base_df.sort_values("all_importance_l2", ascending=False).head(top_n)["base_feature_name"])
        elif token == "top_direction":
            selected.extend(base_df.sort_values("dir_importance_l2", ascending=False).head(top_n)["base_feature_name"])
        elif token == "top_magnitude":
            selected.extend(base_df.sort_values("mag_importance_l2", ascending=False).head(top_n)["base_feature_name"])
        elif token == "low_importance":
            selected.extend(low_df.sort_values("all_importance_l2", ascending=True).head(low_n)["base_feature_name"])
        else:
            raise ValueError(f"Unknown BYBIT_LINEAR_IMPORTANCE_ABLATION_GROUPS token: {token}")
    out = []
    for x in selected:
        if x not in out:
            out.append(x)
    return out


def get_group_columns(flat_rows: list[dict], group_key: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in flat_rows:
        out.setdefault(str(r[group_key]), []).append(int(r["model_coef_index"]))
    return out


def _group_columns(flat_rows: list[dict], feature_name: str) -> dict[str, list[int]]:
    return get_group_columns(flat_rows, feature_name)


def _direction_kept_mask_1s(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y[:, 2]) & (y[:, 2] != 0.0)


def _huber_np(err: np.ndarray, delta: float = 1.0) -> float:
    ae = np.abs(err)
    return float(np.where(ae <= delta, 0.5 * ae * ae, delta * (ae - 0.5 * delta)).mean())


def _safe_nanmean_pair(a: float, b: float) -> float:
    vals = np.asarray([a, b], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else math.nan


def _get_mag_scales(bundle: Any, st4: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    up = getattr(bundle, "mag_up_scale_bps", None)
    dn = getattr(bundle, "mag_down_scale_bps", None)
    if up is None:
        up = st4.get("stage4_config", {}).get("mag_up_scale_bps")
    if dn is None:
        dn = st4.get("stage4_config", {}).get("mag_down_scale_bps")
    if up is None or dn is None:
        raise ValueError("Unable to find mag_up_scale_bps / mag_down_scale_bps in model bundle or stage4_config")
    up = np.asarray(up, dtype=np.float64)
    dn = np.asarray(dn, dtype=np.float64)
    if up.shape[0] < 3 or dn.shape[0] < 3:
        raise ValueError(f"Bad magnitude scale shapes: up={up.shape} down={dn.shape}")
    return up, dn


def compute_ablation_metrics(*, y: np.ndarray, pred: dict[str, np.ndarray], mag_up_scale_bps: np.ndarray, mag_down_scale_bps: np.ndarray) -> dict[str, float]:
    import linear_offline
    h = 2
    scores = pred["dir_logits"][:, h]
    kept = _direction_kept_mask_1s(y)
    truth_kept = y[kept, h] > 0.0
    scores_kept = scores[kept]
    p_kept = 1.0 / (1.0 + np.exp(-scores_kept))
    auc = linear_offline._binary_auc_np(scores_kept, truth_kept) if scores_kept.size else math.nan
    bal = linear_offline._balanced_acc_bool(scores_kept >= 0.0, truth_kept) if scores_kept.size else math.nan
    bce = float(-np.mean(truth_kept * np.log(np.clip(p_kept, 1e-8, 1.0)) + (1.0 - truth_kept) * np.log(np.clip(1.0 - p_kept, 1e-8, 1.0)))) if scores_kept.size else math.nan
    up_rows = y[:, h] > 0.0
    dn_rows = y[:, h] < 0.0
    up_scale = float(mag_up_scale_bps[h]); dn_scale = float(mag_down_scale_bps[h])
    up_h = _huber_np(pred["mag_up_log"][up_rows, h] - np.log1p(y[up_rows, h] / up_scale)) if np.any(up_rows) else math.nan
    dn_h = _huber_np(pred["mag_down_log"][dn_rows, h] - np.log1p((-y[dn_rows, h]) / dn_scale)) if np.any(dn_rows) else math.nan
    side_h = _safe_nanmean_pair(up_h, dn_h)
    up_sp = linear_offline._safe_spearman_np(pred["mag_up_bps"][up_rows, h], y[up_rows, h]) if np.any(up_rows) else math.nan
    dn_sp = linear_offline._safe_spearman_np(pred["mag_down_bps"][dn_rows, h], -y[dn_rows, h]) if np.any(dn_rows) else math.nan
    side_sp = _safe_nanmean_pair(up_sp, dn_sp)
    p_all = 1.0 / (1.0 + np.exp(-scores))
    edge = p_all * pred["mag_up_bps"][:, h] - (1.0 - p_all) * pred["mag_down_bps"][:, h]
    return {
        "dir_auc_kept_1000ms": float(auc), "dir_bal_acc_kept_1000ms": float(bal), "dir_bce_kept_1000ms": float(bce),
        "mean_side_log_huber_cond_1000ms": float(side_h), "mean_side_spearman_cond_1000ms": float(side_sp),
        "edge_spearman_all_1000ms": float(linear_offline._safe_spearman_np(edge, y[:, h])),
        "edge_spearman_kept_1000ms": float(linear_offline._safe_spearman_np(edge[kept], y[kept, h])) if np.any(kept) else math.nan,
    }


def main() -> None:
    from CMSSL17_linear import load_linear_preprocess_bundle, load_linear_sklearn_bundle
    import linear_offline
    linear_out_dir = Path(os.environ["BYBIT_LINEAR_OUT_DIR"]); out_root = Path(os.environ["BYBIT_OUT_ROOT"]); out_dir = linear_out_dir / "feature_importance"; out_dir.mkdir(parents=True, exist_ok=True)
    st4 = json.loads((linear_out_dir / "linear_stage4_metrics.json").read_text()); st2 = json.loads((linear_out_dir / "linear_stage2_extractor_metrics.json").read_text()); st3 = json.loads((linear_out_dir / "linear_stage3_preprocess_metrics.json").read_text()); meta = json.loads((out_root / "meta.json").read_text())
    if str(st4.get("extractor") or st2.get("extractor_config", {}).get("name") or "").strip().lower() != "raw_linear":
        raise ValueError("linear_feature_importance.py currently supports raw_linear only")
    bundle = load_linear_sklearn_bundle(Path(str(st4["best_model_path"]))); pb = load_linear_preprocess_bundle(Path(str(st3["preprocess_bundle_path"])))
    kept_indices = _parse_keep_indices(pb); base_names = _base_names_from_meta(meta); blocks = list((st2.get("extractor_summary", {}) or {}).get("blocks") or [])

    if len(base_names) != int(st2["feature_dim_total"]):
        raise ValueError(f"base name count mismatch: got {len(base_names)} expected feature_dim_total={st2['feature_dim_total']}")
    extracted_rows = _build_raw_linear_extracted_names(base_names, blocks)
    if len(extracted_rows) != int(st2["extractor_output_dim"]):
        raise ValueError(f"extractor output dim mismatch: names={len(extracted_rows)} expected={st2['extractor_output_dim']}")
    if kept_indices.size != int(pb.kept_dim):
        raise ValueError(f"kept_indices size mismatch: got {kept_indices.size} expected kept_dim={pb.kept_dim}")
    if kept_indices.size and int(np.max(kept_indices)) >= len(extracted_rows):
        raise ValueError(f"kept index out of range: max={int(np.max(kept_indices))} extracted_dim={len(extracted_rows)}")

    dir_coefs = [_extract_linear_coef(m) for m in bundle.direction_models]
    up_coefs = [_extract_linear_coef(m) for m in bundle.mag_up_models]
    dn_coefs = [_extract_linear_coef(m) for m in bundle.mag_down_models]
    for name, coef in [
        *[(f"direction_{i}", c) for i, c in enumerate(dir_coefs)],
        *[(f"mag_up_{i}", c) for i, c in enumerate(up_coefs)],
        *[(f"mag_down_{i}", c) for i, c in enumerate(dn_coefs)],
    ]:
        if coef.shape[0] != int(pb.kept_dim):
            raise ValueError(f"Coefficient dim mismatch for {name}: got {coef.shape[0]} expected kept_dim={pb.kept_dim}")

    flat_df = build_flat_importance_df(extracted_rows=extracted_rows, kept_indices=kept_indices, dir_coefs=dir_coefs, mag_up_coefs=up_coefs, mag_down_coefs=dn_coefs)
    base_df = aggregate_importance_by_base(flat_df, base_names, blocks)
    by_base_block = flat_df.groupby(["base_feature_name", "block_name"]).agg(dirv=("dir_abs_coef_1000ms", "max"), magv=("mag_abs_coef_max", "max"), allv=("all_abs_coef_max", "max")).reset_index()
    for metric, col in [("dirv", "top_block_by_direction_importance"), ("magv", "top_block_by_magnitude_importance"), ("allv", "top_block_by_all_importance")]:
        top_map = by_base_block.sort_values(metric, ascending=False).drop_duplicates("base_feature_name").set_index("base_feature_name")["block_name"]
        base_df[col] = base_df["base_feature_name"].map(top_map)
    base_df = add_low_importance_flags(base_df, low_share=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_LOW_SHARE", "0.0005")), low_dir_share=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_LOW_DIR_SHARE", "0.0005")), low_mag_share=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_LOW_MAG_SHARE", "0.0005")), coef_eps=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_COEF_EPS", "1e-10")), flat_df=flat_df)
    block_df = aggregate_importance_by_block(flat_df)
    low_df = base_df[base_df["low_importance_candidate"] | base_df["zero_or_near_zero_all_heads"]].copy()
    ab_rows = []
    if int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION", "1")) == 1:
        plan = linear_offline.load_linear_split_plan_from_out_root(out_root=out_root); extractor_obj, _ = linear_offline.load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name="raw_linear"); ds = linear_offline.build_val_dataset_from_plan(plan)
        max_rows = int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_MAX_ROWS", "200000")); z_parts = []; y_parts = []
        try:
            for Zb, yb, _ in linear_offline.iter_preprocessed_batches_from_dataset(extractor=extractor_obj, bundle=pb, ds=ds, batch_rows=linear_offline.LINEAR_STAGE5_BATCH_ROWS, max_rows=max_rows, split_name="importance_ablation"):
                z_parts.append(Zb); y_parts.append(yb)
        finally:
            linear_offline.close_dataset(ds, name="importance_ablation")
        Z = np.concatenate(z_parts, axis=0); y = np.concatenate(y_parts, axis=0)
        base_pred = bundle.predict_dict_np(Z)
        mag_up_scale_bps, mag_down_scale_bps = _get_mag_scales(bundle, st4)
        base_metrics = compute_ablation_metrics(y=y, pred=base_pred, mag_up_scale_bps=mag_up_scale_bps, mag_down_scale_bps=mag_down_scale_bps)
        groups = get_group_columns(flat_df.to_dict("records"), "base_feature_name")
        names = select_ablation_groups(base_df=base_df, low_df=low_df, top_n=int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_TOP_N", "25")), low_n=int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_LOW_N", "50")), groups_spec=os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_GROUPS", "low_importance,top_direction,top_magnitude,top_all"), all_base=int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_ALL_BASE", "0")) == 1)
        for name in names:
            cols = np.asarray(groups.get(name, []), dtype=np.int64)
            if cols.size == 0:
                continue
            Za = Z.copy(); Za[:, cols] = 0.0
            m = compute_ablation_metrics(y=y, pred=bundle.predict_dict_np(Za), mag_up_scale_bps=mag_up_scale_bps, mag_down_scale_bps=mag_down_scale_bps)
            row = {"group_type": "base_feature", "group_name": name, "n_columns_zeroed": int(cols.size)}
            for k in base_metrics:
                row[f"baseline_{k}"] = base_metrics[k]; row[f"ablated_{k}"] = m[k]; row[f"delta_{k}"] = float(m[k] - base_metrics[k])
            ab_rows.append(row)
    flat_path = out_dir / "linear_importance_flat.csv"; base_path = out_dir / "linear_importance_by_base_feature.csv"; block_path = out_dir / "linear_importance_by_block.csv"; low_path = out_dir / "linear_importance_low_candidates.csv"; ab_path = out_dir / "linear_importance_ablation.csv"
    flat_df.to_csv(flat_path, index=False); base_df.sort_values("all_importance_l2", ascending=False).to_csv(base_path, index=False); block_df.sort_values("all_importance_l2", ascending=False).to_csv(block_path, index=False); low_df.to_csv(low_path, index=False); pd.DataFrame(ab_rows).to_csv(ab_path, index=False)
    ablation_enabled = bool(int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION", "1")) == 1)
    summary = {
        "schema": "linear_feature_importance_v1",
        "linear_out_dir": str(linear_out_dir),
        "out_root": str(out_root),
        "extractor": "raw_linear",
        "kept_dim": int(pb.kept_dim),
        "original_dim": int(pb.original_dim),
        "n_base_features": int(len(base_names)),
        "n_blocks": int(len(blocks)),
        "top_base_features_direction_1000ms": list(base_df.sort_values("dir_importance_1000ms_l2", ascending=False).head(10)["base_feature_name"]),
        "top_base_features_magnitude_1000ms": list(base_df.sort_values("mag_importance_1000ms_l2", ascending=False).head(10)["base_feature_name"]),
        "top_base_features_all_1000ms": list(base_df.sort_values("all_importance_1000ms_l2", ascending=False).head(10)["base_feature_name"]),
        "top_base_features_direction_all_horizons": list(base_df.sort_values("dir_importance_l2", ascending=False).head(10)["base_feature_name"]),
        "top_base_features_magnitude_all_horizons": list(base_df.sort_values("mag_importance_l2", ascending=False).head(10)["base_feature_name"]),
        "top_base_features_all_horizons": list(base_df.sort_values("all_importance_l2", ascending=False).head(10)["base_feature_name"]),
        "top_blocks_all": list(block_df.sort_values("all_importance_l2", ascending=False).head(10)["block_name"]),
        "top_blocks_direction": list(block_df.sort_values("dir_importance_l2", ascending=False).head(10)["block_name"]),
        "top_blocks_magnitude": list(block_df.sort_values("mag_importance_l2", ascending=False).head(10)["block_name"]),
        "n_low_importance_candidates": int(low_df.shape[0]),
        "ablation_enabled": ablation_enabled,
        "ablation_rows": int(max_rows if ablation_enabled else 0),
        "ablation_n_groups": int(len(ab_rows)),
        "paths": {
            "flat": str(flat_path),
            "base": str(base_path),
            "block": str(block_path),
            "low": str(low_path),
            "ablation": str(ab_path),
            "summary": str(out_dir / "linear_importance_summary.json"),
        },
    }
    (out_dir / "linear_importance_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(f"[linear-importance] loaded model={st4['best_model_path']} kept_dim={pb.kept_dim} extractor=raw_linear", flush=True)
    print(f"[linear-importance] wrote flat={flat_path} base={base_path} block={block_path} low={low_path}", flush=True)
    print("[linear-importance-top] dir_1000ms=" + ",".join(summary["top_base_features_direction_1000ms"][:5]), flush=True)
    print("[linear-importance-top] mag_1000ms=" + ",".join(summary["top_base_features_magnitude_1000ms"][:5]), flush=True)
    print(f"[linear-importance-low] candidates={summary['n_low_importance_candidates']}", flush=True)
    print(f"[linear-importance-ablation] rows={summary['ablation_rows']} groups={summary['ablation_n_groups']} wrote={ab_path}", flush=True)


if __name__ == "__main__":
    main()
