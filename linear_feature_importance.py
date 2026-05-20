from __future__ import annotations
import json, math, os
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from CMSSL17_linear import load_linear_preprocess_bundle, load_linear_sklearn_bundle
import linear_offline


def _extract_linear_coef(model: Any) -> np.ndarray:
    for candidate in (model, getattr(model, "estimator", None), getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "coef_"):
            return np.asarray(getattr(candidate, "coef_"), dtype=np.float64).reshape(-1)
    raise ValueError(f"Unable to extract linear coefficients from model type={type(model).__name__}")


def _build_raw_linear_extracted_names(base_names: List[str], blocks: List[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for block in blocks:
        for bi, base in enumerate(base_names):
            rows.append({
                "raw_extracted_index": idx,
                "base_feature_index": bi,
                "base_feature_name": base,
                "block_name": block,
                "extracted_feature_name": f"{base}:{block}",
            })
            idx += 1
    return rows


def _base_names_from_meta(meta: Dict[str, Any]) -> List[str]:
    if isinstance(meta.get("feature_names"), list):
        core = list(meta.get("feature_names") or [])
        aux = list(meta.get("aux_feature_names") or [])
        return core + aux
    if isinstance(meta.get("raw_feature_names"), list):
        return list(meta.get("raw_feature_names") or [])
    core = list(meta.get("core_feature_names") or [])
    aux = list(meta.get("aux_feature_names") or [])
    out = core + aux
    if out:
        return out
    raise ValueError("Unable to resolve base names from meta.json")


def _parse_keep_indices(pb) -> np.ndarray:
    km = np.asarray(pb.keep_mask)
    if km.dtype == bool:
        return np.flatnonzero(km)
    return np.asarray(km, dtype=np.int64).reshape(-1)


def _compute_ablation_metrics(y: np.ndarray, dir_logits: np.ndarray, mag_up_bps: np.ndarray, mag_down_bps: np.ndarray) -> dict[str, float]:
    h = [int(x) for x in linear_offline.HORIZONS_MS].index(1000)
    truth = (y[:, h] > 0.0)
    scores = dir_logits[:, h]
    p = 1.0 / (1.0 + np.exp(-scores))
    edge = p * mag_up_bps[:, h] - (1.0 - p) * mag_down_bps[:, h]
    auc = linear_offline._binary_auc_np(scores, truth)
    bal = linear_offline._balanced_acc_bool(scores >= 0.0, truth)
    bce = float(-np.mean(truth * np.log(np.clip(p, 1e-8, 1.0)) + (1.0 - truth) * np.log(np.clip(1.0 - p, 1e-8, 1.0))))
    true_mag = np.abs(y[:, h])
    pred_mag = np.where(scores >= 0.0, mag_up_bps[:, h], mag_down_bps[:, h])
    log_huber = float(np.mean(np.log1p(np.abs(pred_mag - true_mag))))
    sp_mag = linear_offline._safe_spearman_np(pred_mag, true_mag)
    sp_edge = linear_offline._safe_spearman_np(edge, y[:, h])
    return {
        "dir_auc_1000ms": float(auc), "dir_bal_acc_1000ms": float(bal), "dir_bce_1000ms": float(bce),
        "mag_log_huber_1000ms": float(log_huber), "mag_spearman_1000ms": float(sp_mag),
        "edge_spearman_all_1000ms": float(sp_edge), "edge_spearman_kept_1000ms": float(sp_edge),
    }


def _group_columns(flat_rows: List[dict[str, Any]], feature_name: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in flat_rows:
        out.setdefault(str(r[feature_name]), []).append(int(r["model_coef_index"]))
    return out


def main() -> None:
    linear_out_dir = Path(os.environ["BYBIT_LINEAR_OUT_DIR"])
    out_root = Path(os.environ["BYBIT_OUT_ROOT"])
    out_dir = linear_out_dir / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    st4 = json.loads((linear_out_dir / "linear_stage4_metrics.json").read_text())
    st2 = json.loads((linear_out_dir / "linear_stage2_extractor_metrics.json").read_text())
    st3 = json.loads((linear_out_dir / "linear_stage3_preprocess_metrics.json").read_text())
    meta = json.loads((out_root / "meta.json").read_text())

    extractor = str(st4.get("extractor") or st2.get("extractor_config", {}).get("name") or "").strip().lower()
    if extractor != "raw_linear":
        raise ValueError("linear_feature_importance.py currently supports raw_linear only")

    model_path = Path(str(st4["best_model_path"]))
    bundle = load_linear_sklearn_bundle(model_path)
    pb = load_linear_preprocess_bundle(Path(str(st3["preprocess_bundle_path"])))
    kept_indices = _parse_keep_indices(pb)
    if kept_indices.size != int(pb.kept_dim):
        raise ValueError(f"kept_indices size mismatch: {kept_indices.size} vs kept_dim={pb.kept_dim}")

    base_names = _base_names_from_meta(meta)
    if len(base_names) != int(st2["feature_dim_total"]):
        raise ValueError("base name count mismatch")
    blocks = list((st2.get("extractor_summary", {}) or {}).get("blocks") or [])
    extracted_rows = _build_raw_linear_extracted_names(base_names, blocks)
    if len(extracted_rows) != int(st2["extractor_output_dim"]):
        raise ValueError("extractor output dim mismatch")

    dir_coefs = [_extract_linear_coef(m) for m in bundle.direction_models]
    up_coefs = [_extract_linear_coef(m) for m in bundle.mag_up_models]
    dn_coefs = [_extract_linear_coef(m) for m in bundle.mag_down_models]
    for c in dir_coefs + up_coefs + dn_coefs:
        if c.shape[0] != int(pb.kept_dim):
            raise ValueError(f"Coefficient dim mismatch: coef={c.shape}, kept_dim={pb.kept_dim}")

    flat = []
    for j in range(int(pb.kept_dim)):
        src = extracted_rows[int(kept_indices[j])]
        da = np.abs([dir_coefs[0][j], dir_coefs[1][j], dir_coefs[2][j]])
        ua = np.abs([up_coefs[0][j], up_coefs[1][j], up_coefs[2][j]])
        na = np.abs([dn_coefs[0][j], dn_coefs[1][j], dn_coefs[2][j]])
        allv = np.concatenate([da, ua, na])
        row = {"model_coef_index": j, **src,
               "dir_coef_1000ms": float(dir_coefs[2][j]), "dir_abs_coef_200ms": float(da[0]), "dir_abs_coef_500ms": float(da[1]), "dir_abs_coef_1000ms": float(da[2]),
               "dir_abs_coef_max": float(np.max(da)), "dir_abs_coef_mean": float(np.mean(da)),
               "mag_up_abs_coef_200ms": float(ua[0]), "mag_up_abs_coef_500ms": float(ua[1]), "mag_up_abs_coef_1000ms": float(ua[2]), "mag_up_abs_coef_max": float(np.max(ua)), "mag_up_abs_coef_mean": float(np.mean(ua)),
               "mag_down_abs_coef_200ms": float(na[0]), "mag_down_abs_coef_500ms": float(na[1]), "mag_down_abs_coef_1000ms": float(na[2]), "mag_down_abs_coef_max": float(np.max(na)), "mag_down_abs_coef_mean": float(np.mean(na)),
               "mag_abs_coef_max": float(np.max(np.concatenate([ua, na]))), "mag_abs_coef_mean": float(np.mean(np.concatenate([ua, na]))),
               "all_abs_coef_max": float(np.max(allv)), "all_abs_coef_mean": float(np.mean(allv))}
        flat.append(row)

    import pandas as pd
    flat_df = pd.DataFrame(flat)
    def agg_by(df, key):
        rows=[]
        for name, g in df.groupby(key):
            d=np.concatenate([g[["dir_abs_coef_200ms","dir_abs_coef_500ms","dir_abs_coef_1000ms"]].to_numpy().reshape(-1)])
            u=np.concatenate([g[["mag_up_abs_coef_200ms","mag_up_abs_coef_500ms","mag_up_abs_coef_1000ms","mag_down_abs_coef_200ms","mag_down_abs_coef_500ms","mag_down_abs_coef_1000ms"]].to_numpy().reshape(-1)])
            a=np.concatenate([d,u])
            rows.append({key:name,"n_kept_columns":int(len(g)),"dir_importance_l2":float(np.sqrt(np.sum(d**2))),"mag_importance_l2":float(np.sqrt(np.sum(u**2))),"all_importance_l2":float(np.sqrt(np.sum(a**2))),"dir_importance_max":float(np.max(d)),"dir_importance_mean":float(np.mean(d)),"mag_importance_max":float(np.max(u)),"mag_importance_mean":float(np.mean(u)),"all_importance_max":float(np.max(a)),"all_importance_mean":float(np.mean(a))})
        out=pd.DataFrame(rows)
        for c in ["dir","mag","all"]:
            s=float(out[f"{c}_importance_l2"].sum()) or 1.0
            out[f"{c}_importance_l2_share"]=out[f"{c}_importance_l2"]/s
        return out
    base_df=agg_by(flat_df,"base_feature_name")
    bidx=flat_df[["base_feature_name","base_feature_index"]].drop_duplicates(); base_df=base_df.merge(bidx,on="base_feature_name",how="left")
    total_cols={name:len(blocks) for name in base_names}; base_df["n_total_columns"]=base_df["base_feature_name"].map(total_cols).astype(int)
    by_base_block=flat_df.groupby(["base_feature_name","block_name"]).agg(dirv=("dir_abs_coef_max","max"),magv=("mag_abs_coef_max","max"),allv=("all_abs_coef_max","max")).reset_index()
    top_dir=by_base_block.sort_values("dirv",ascending=False).drop_duplicates("base_feature_name").set_index("base_feature_name")["block_name"]
    top_mag=by_base_block.sort_values("magv",ascending=False).drop_duplicates("base_feature_name").set_index("base_feature_name")["block_name"]
    top_all=by_base_block.sort_values("allv",ascending=False).drop_duplicates("base_feature_name").set_index("base_feature_name")["block_name"]
    base_df["top_block_by_direction_importance"]=base_df["base_feature_name"].map(top_dir)
    base_df["top_block_by_magnitude_importance"]=base_df["base_feature_name"].map(top_mag)
    base_df["top_block_by_all_importance"]=base_df["base_feature_name"].map(top_all)

    low=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_LOW_SHARE","0.0005")); lowd=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_LOW_DIR_SHARE","0.0005")); lowm=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_LOW_MAG_SHARE","0.0005")); eps=float(os.getenv("BYBIT_LINEAR_IMPORTANCE_COEF_EPS","1e-10"))
    base_df["low_importance_candidate"]=(base_df["all_importance_l2_share"]<=low)&(base_df["dir_importance_l2_share"]<=lowd)&(base_df["mag_importance_l2_share"]<=lowm)
    absmax=flat_df.groupby("base_feature_name")["all_abs_coef_max"].max()
    base_df["zero_or_near_zero_all_heads"]=base_df["base_feature_name"].map(absmax<=eps).fillna(False)

    block_df=agg_by(flat_df,"block_name")

    low_df=base_df[base_df["low_importance_candidate"]|base_df["zero_or_near_zero_all_heads"]].copy()

    # ablation (bounded, no retrain)
    ab_rows=[]
    if int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION","1"))==1:
        plan=linear_offline.load_linear_split_plan_from_out_root(out_root=out_root)
        extractor_obj,_=linear_offline.load_stage2_extractor_bundle(linear_out_dir=linear_out_dir, extractor_name="raw_linear")
        ds=linear_offline.build_val_dataset_from_plan(plan)
        max_rows=int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_MAX_ROWS","200000"))
        try:
            parts=linear_offline.collect_predictions_and_labels_streaming(model_bundle=bundle,extractor=extractor_obj,preprocess_bundle=pb,ds=ds,max_rows=max_rows,batch_rows=linear_offline.LINEAR_STAGE5_BATCH_ROWS,split_name="importance_ablation",progress_stage="stage5",progress_action="diagnostics")
            # recreate Z
            z_parts=[]; y_parts=[]
            it=linear_offline.iter_preprocessed_batches_from_dataset(extractor=extractor_obj,bundle=pb,ds=ds,batch_rows=linear_offline.LINEAR_STAGE5_BATCH_ROWS,max_rows=max_rows,split_name="importance_ablation_z")
            for Z,y,_ in it: z_parts.append(Z); y_parts.append(y)
            Z=np.concatenate(z_parts,axis=0); y=np.concatenate(y_parts,axis=0)
        finally:
            linear_offline.close_dataset(ds,name="importance_ablation")
        base_metrics=_compute_ablation_metrics(y,parts["dir_logits"],parts["mag_up_bps"],parts["mag_down_bps"])
        groups=_group_columns(flat,"base_feature_name")
        top_n=int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_TOP_N","25")); low_n=int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_LOW_N","50"))
        cand=[]
        cand += list(base_df.sort_values("all_importance_l2",ascending=False).head(top_n)["base_feature_name"])
        cand += list(base_df.sort_values("dir_importance_l2",ascending=False).head(top_n)["base_feature_name"])
        cand += list(base_df.sort_values("mag_importance_l2",ascending=False).head(top_n)["base_feature_name"])
        cand += list(low_df.sort_values("all_importance_l2",ascending=True).head(low_n)["base_feature_name"])
        if int(os.getenv("BYBIT_LINEAR_IMPORTANCE_ABLATION_ALL_BASE","0"))==1:
            cand=list(base_df["base_feature_name"])
        dedup=[]
        for x in cand:
            if x not in dedup: dedup.append(x)
        for name in dedup:
            cols=np.asarray(groups.get(name,[]),dtype=np.int64)
            if cols.size==0: continue
            Za=Z.copy(); Za[:,cols]=0.0
            pred=bundle.predict_dict_np(Za)
            m=_compute_ablation_metrics(y,pred["dir_logits"],pred["mag_up_bps"],pred["mag_down_bps"])
            ab_rows.append({"group_type":"base_feature","group_name":name,"n_columns_zeroed":int(cols.size),
                **{f"baseline_{k}":v for k,v in base_metrics.items()},**{f"ablated_{k}":v for k,v in m.items()},**{f"delta_{k}":float(m[k]-base_metrics[k]) for k in base_metrics}})

    flat_path=out_dir/"linear_importance_flat.csv"; base_path=out_dir/"linear_importance_by_base_feature.csv"; block_path=out_dir/"linear_importance_by_block.csv"; low_path=out_dir/"linear_importance_low_candidates.csv"; ab_path=out_dir/"linear_importance_ablation.csv"
    flat_df.to_csv(flat_path,index=False); base_df.sort_values("all_importance_l2",ascending=False).to_csv(base_path,index=False); block_df.sort_values("all_importance_l2",ascending=False).to_csv(block_path,index=False); low_df.to_csv(low_path,index=False); __import__('pandas').DataFrame(ab_rows).to_csv(ab_path,index=False)
    summary={"schema":"linear_feature_importance_v1","linear_out_dir":str(linear_out_dir),"out_root":str(out_root),"extractor":"raw_linear","kept_dim":int(pb.kept_dim),"original_dim":int(pb.original_dim),"n_base_features":int(len(base_names)),"n_blocks":int(len(blocks)),"top_base_features_direction_1s":list(base_df.sort_values('dir_importance_l2',ascending=False).head(10)['base_feature_name']),"top_base_features_magnitude_1s":list(base_df.sort_values('mag_importance_l2',ascending=False).head(10)['base_feature_name']),"top_base_features_all":list(base_df.sort_values('all_importance_l2',ascending=False).head(10)['base_feature_name']),"top_blocks_all":list(block_df.sort_values('all_importance_l2',ascending=False).head(10)['block_name']),"n_low_importance_candidates":int(low_df.shape[0]),"ablation_enabled":bool(int(os.getenv('BYBIT_LINEAR_IMPORTANCE_ABLATION','1'))==1),"ablation_rows":int(os.getenv('BYBIT_LINEAR_IMPORTANCE_ABLATION_MAX_ROWS','200000')),"ablation_n_groups":int(len(ab_rows)),"paths":{"flat":str(flat_path),"base":str(base_path),"block":str(block_path),"low":str(low_path),"ablation":str(ab_path)}}
    (out_dir/"linear_importance_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True))

    print(f"[linear-importance] loaded model={model_path} kept_dim={pb.kept_dim} extractor=raw_linear", flush=True)
    print(f"[linear-importance] wrote flat={flat_path} base={base_path} block={block_path} low={low_path}", flush=True)
    print("[linear-importance-top] dir_1s=" + ",".join(summary["top_base_features_direction_1s"][:5]), flush=True)
    print("[linear-importance-top] mag_1s=" + ",".join(summary["top_base_features_magnitude_1s"][:5]), flush=True)
    print(f"[linear-importance-low] candidates={summary['n_low_importance_candidates']}", flush=True)
    print(f"[linear-importance-ablation] rows={summary['ablation_rows']} groups={summary['ablation_n_groups']} wrote={ab_path}", flush=True)

if __name__ == "__main__":
    main()
