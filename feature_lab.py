#!/usr/bin/env python3
"""Lean feature lab for candidate feature evaluation.

Expression example:
BYBIT_OUT_ROOT="/home/gabrool/Documents/Quant/tokens/btc_20260222_20260328_v8_pruned172" \
python -u feature_lab.py \
  --mode expr \
  --candidate-name "ofi_l5_minus_l10_500ms" \
  --expr "ofi_l5_sum_over_depth_500ms - ofi_l10_sum_over_depth_500ms"

Plugin example:
BYBIT_OUT_ROOT="/home/gabrool/Documents/Quant/tokens/btc_20260222_20260328_v8_pruned172" \
python -u feature_lab.py \
  --mode plugin \
  --candidate-module feature_candidates \
  --candidate-class ExampleCandidate
"""
from __future__ import annotations
import argparse, csv, importlib, json, os
from pathlib import Path
import numpy as np
from feature_audit import (
    systematic_positions, proportional_week_targets, resolve_week_meta_path,
    read_json, load_labels_for_positions, load_features_by_row_idx, parse_feature_name,
)

SEED=int(os.environ.get("BYBIT_FEATURE_LAB_SEED","17")); MAX_TOTAL=int(os.environ.get("BYBIT_FEATURE_LAB_MAX_ROWS_TOTAL","1000000")); MAX_PER_WEEK=int(os.environ.get("BYBIT_FEATURE_LAB_MAX_ROWS_PER_WEEK","600000")); USE_AUX=os.environ.get("BYBIT_FEATURE_LAB_USE_AUX","0")=="1"; MI_MAX=int(os.environ.get("BYBIT_FEATURE_LAB_MI_MAX_ROWS","200000")); HIGH=float(os.environ.get("BYBIT_FEATURE_LAB_HIGH_CORR","0.95")); MED=float(os.environ.get("BYBIT_FEATURE_LAB_MED_CORR","0.90")); TOP=int(os.environ.get("BYBIT_FEATURE_LAB_TOP_CORR","50"))

class SafeNpNamespace:
    log1p=np.log1p; abs=np.abs; sign=np.sign; clip=np.clip; maximum=np.maximum; minimum=np.minimum

def _safe_pearson_np(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3:return np.nan
    xv=x[m]-x[m].mean(); yv=y[m]-y[m].mean(); d=np.sqrt((xv*xv).sum()*(yv*yv).sum())
    return float((xv*yv).sum()/d) if d>0 else np.nan

def _safe_spearman_np(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return _safe_pearson_np(rx.astype(float),ry.astype(float))

def _binary_auc_np(y01,s):
    m=np.isfinite(s); y=y01[m].astype(int); s=s[m]
    pos=(y==1).sum(); neg=(y==0).sum()
    if pos==0 or neg==0:return np.nan
    ord=np.argsort(s); r=np.empty_like(ord); r[ord]=np.arange(len(s))+1
    return float((r[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def _bal_acc_best_threshold_np(y01,s):
    m=np.isfinite(s); y=y01[m].astype(int); s=s[m]
    if len(s)<3:return (np.nan,0,np.nan)
    best=(0.0,0,np.nan)
    for t in np.unique(np.quantile(s,np.linspace(0.05,0.95,19))):
        p=(s>=t).astype(int); tp=((p==1)&(y==1)).sum(); tn=((p==0)&(y==0)).sum(); pp=(y==1).sum(); nn=(y==0).sum()
        if pp==0 or nn==0: continue
        ba=0.5*(tp/pp+tn/nn); sign=1
        if 1-ba>ba: ba=1-ba; sign=-1
        if ba>best[0]: best=(float(ba),sign,float(t))
    return best

def _mutual_info_optional(x,y,is_cls):
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        f = mutual_info_classif if is_cls else mutual_info_regression
        return float(f(x.reshape(-1,1), y, discrete_features=False, random_state=SEED)[0])
    except Exception:
        return np.nan

def eval_expr(expr,X,names):
    safe_globals={"__builtins__":{},"np":SafeNpNamespace}
    safe_locals={n:X[:,i] for i,n in enumerate(names)}
    try: out=eval(expr,safe_globals,safe_locals)
    except NameError as e: raise ValueError(f"Unknown feature or name in expression: {e}")
    except Exception as e: raise ValueError(f"Invalid expression: {e}")
    return np.asarray(out,dtype=np.float32)

def evaluate_candidate_array(candidate_name,candidate,X,y,feature_names,week_keys):
    if candidate.shape!=(X.shape[0],): raise ValueError("candidate shape mismatch")
    finite=np.isfinite(candidate)
    if finite.mean()<1.0: raise ValueError("candidate has nonfinite values")
    if float(np.std(candidate))<1e-8: raise ValueError("candidate nearly constant")
    health={"candidate":candidate_name,"n_rows":int(len(candidate)),"finite_frac":1.0,"nan_count":0,"inf_count":0,"mean":float(candidate.mean()),"std":float(candidate.std()),"p01":float(np.quantile(candidate,0.01)),"p05":float(np.quantile(candidate,0.05)),"p50":float(np.quantile(candidate,0.5)),"p95":float(np.quantile(candidate,0.95)),"p99":float(np.quantile(candidate,0.99)),"min":float(candidate.min()),"max":float(candidate.max()),"abs_max":float(np.abs(candidate).max()),"zero_frac":float((candidate==0).mean()),"near_zero_frac_abs_lt_1e-6":float((np.abs(candidate)<1e-6).mean())}
    wstds=[]; mean_by_week={}; std_by_week={}
    for wk in sorted(set(week_keys)):
        m=np.array([w==wk for w in week_keys]); mean_by_week[wk]=float(candidate[m].mean()); std_by_week[wk]=float(candidate[m].std()); wstds.append(std_by_week[wk])
    health["week_std_cv"]=float(np.std(wstds)/(np.mean(wstds)+1e-12))
    corr=[]
    for i,n in enumerate(feature_names):
        p=_safe_pearson_np(candidate,X[:,i]); s=_safe_spearman_np(candidate,X[:,i]); m=np.nanmax([abs(p),abs(s)])
        corr.append((m,{"candidate":candidate_name,"existing_feature":n,"existing_feature_index":i,"pearson":p,"spearman":s,"abs_pearson":abs(p),"abs_spearman":abs(s),"max_abs_corr":m,**{k:v for k,v in parse_feature_name(n).items() if k in ("family","timescale_ms")}}))
    corr_rows=[r[1] for r in sorted(corr,key=lambda z:z[0],reverse=True)[:TOP]]
    horizons=[200,500,1000]; target_rows=[]; dec=[]
    for h_i,h in enumerate(horizons):
        ys=y[:,h_i]; direction=(ys>0).astype(int); absr=np.abs(ys)
        masks={"all_finite":np.isfinite(ys),"nonzero":np.isfinite(ys)&(candidate!=0),"kept":np.isfinite(ys)&(absr>0)}
        for mk,m in masks.items():
            xx=candidate[m]; yy=ys[m]; dd=direction[m]; aa=absr[m]
            auc=_binary_auc_np(dd,xx); bal,sign,thr=_bal_acc_best_threshold_np(dd,xx)
            target_rows.append({"candidate":candidate_name,"horizon_ms":h,"mask_type":mk,"pearson_signed_return":_safe_pearson_np(xx,yy),"spearman_signed_return":_safe_spearman_np(xx,yy),"pearson_abs_return":_safe_pearson_np(xx,aa),"spearman_abs_return":_safe_spearman_np(xx,aa),"single_feature_auc_direction":max(auc,1-auc) if np.isfinite(auc) else np.nan,"single_feature_auc_direction_sign":1 if (np.isfinite(auc) and auc>=0.5) else (-1 if np.isfinite(auc) else 0),"single_feature_bal_acc_sign":sign,"single_feature_bal_acc_best_threshold":bal,"single_feature_bal_acc_best_threshold_value":thr,"mi_direction":_mutual_info_optional(xx[:MI_MAX],dd[:MI_MAX],True),"mi_abs_return":_mutual_info_optional(xx[:MI_MAX],aa[:MI_MAX],False)})
        q=np.quantile(candidate,np.linspace(0,1,11)); bins=np.digitize(candidate,q[1:-1],right=True)
        for d in range(10):
            m=(bins==d)&np.isfinite(ys)
            if not m.any(): continue
            dec.append({"candidate":candidate_name,"horizon_ms":h,"decile":d,"n_rows":int(m.sum()),"candidate_min":float(candidate[m].min()),"candidate_max":float(candidate[m].max()),"mean_signed_return":float(ys[m].mean()),"mean_abs_return":float(np.abs(ys[m]).mean()),"up_frac":float((ys[m]>0).mean())})
    max_corr=float(corr_rows[0]["max_abs_corr"]) if corr_rows else np.nan
    best_auc=max((r["single_feature_auc_direction"] for r in target_rows if r["mask_type"]=="kept"), default=np.nan)
    best_abs=max((abs(r["spearman_abs_return"]) for r in target_rows if r["mask_type"]=="kept"), default=np.nan)
    decision,reason="needs_ablation","plausible"
    if health["std"]<1e-8 or max_corr>=HIGH and best_auc<=0.55: decision,reason="reject","weak or redundant"
    elif max_corr<MED and (best_auc>0.56 or best_abs>0.03): decision,reason="promote_candidate","useful signal and low redundancy"
    rel={"candidate":candidate_name,"max_corr_with_existing":max_corr,"most_correlated_existing_feature":corr_rows[0]["existing_feature"] if corr_rows else "","high_corr_duplicate":bool(max_corr>=HIGH),"medium_corr_related":bool(MED<=max_corr<HIGH),"best_kept_auc_200ms":max((r["single_feature_auc_direction"] for r in target_rows if r["horizon_ms"]==200 and r["mask_type"]=="kept"),default=np.nan),"best_kept_auc_500ms":max((r["single_feature_auc_direction"] for r in target_rows if r["horizon_ms"]==500 and r["mask_type"]=="kept"),default=np.nan),"best_kept_auc_1000ms":max((r["single_feature_auc_direction"] for r in target_rows if r["horizon_ms"]==1000 and r["mask_type"]=="kept"),default=np.nan),"best_kept_bal_acc_1000ms":max((r["single_feature_bal_acc_best_threshold"] for r in target_rows if r["horizon_ms"]==1000 and r["mask_type"]=="kept"),default=np.nan),"best_abs_return_spearman":best_abs,"best_mi_direction":max((r["mi_direction"] for r in target_rows),default=np.nan),"best_mi_abs_return":max((r["mi_abs_return"] for r in target_rows),default=np.nan),"finite_frac":1.0,"std":health["std"],"week_std_cv":health["week_std_cv"],"decision":decision,"reason":reason}
    summary={"schema":"feature_lab_v1","candidate":candidate_name,"n_rows":int(len(candidate)),"feature_dim":int(X.shape[1]),"use_aux":USE_AUX,"health":{"finite_frac":1.0,"std":health["std"],"week_std_cv":health["week_std_cv"],"mean_by_week":mean_by_week,"std_by_week":std_by_week},"best_direction":{"best_kept_auc":best_auc},"best_magnitude":{"best_abs_return_spearman":best_abs},"correlation":{"max_abs_corr":max_corr},"decision":decision,"reason":reason}
    return health,target_rows,corr_rows,rel,summary,dec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",required=True,choices=["expr","plugin","event"]); ap.add_argument("--candidate-name",default=""); ap.add_argument("--expr",default=""); ap.add_argument("--candidate-module",default="feature_candidates"); ap.add_argument("--candidate-class",default=""); ap.add_argument("--out-dir",default="")
    a=ap.parse_args()
    if a.mode=="event": raise NotImplementedError("event mode is intentionally deferred; use expr/plugin mode first")
    out_root=Path(os.environ.get("BYBIT_OUT_ROOT","").strip()); meta=read_json(out_root/"meta.json")
    names=[str(x) for x in meta["feature_names"]]; idx=np.arange(len(names)) if USE_AUX else np.array([i for i,n in enumerate(names) if "aux" not in n],dtype=np.int64); names=[names[i] for i in idx]
    labels_by_week={wk:int(read_json(resolve_week_meta_path(out_root,meta["weeks_meta"],wk)).get("n_labels",0)) for wk in meta.get("train_weeks",[])}
    targets=proportional_week_targets(labels_by_week,MAX_TOTAL,MAX_PER_WEEK)
    Xs=[]; Ys=[]; wks=[]
    for wi,wk in enumerate(meta.get("train_weeks",[])):
        n=targets.get(wk,0); 
        if n<=0: continue
        wm=read_json(resolve_week_meta_path(out_root,meta["weeks_meta"],wk)); pos=systematic_positions(int(wm.get("n_labels",0)),n,SEED+wi)
        row_idx,_,y=load_labels_for_positions(out_root,wk,wm["label_chunks"],pos); x=load_features_by_row_idx(out_root,wk,wm["feature_chunks"],row_idx,idx)
        Xs.append(x); Ys.append(y.astype(np.float32)); wks.extend([wk]*len(x))
    X=np.concatenate(Xs,0); y=np.concatenate(Ys,0)
    if a.mode=="expr":
        cname=a.candidate_name or "expr_candidate"; cand=eval_expr(a.expr,X,names)
    else:
        mod=importlib.import_module(a.candidate_module); cls=getattr(mod,a.candidate_class); obj=cls(); cname=getattr(obj,"name",a.candidate_class); cand=np.asarray(obj.compute(X,names),dtype=np.float32)
    assert X.shape[0]==y.shape[0]==cand.shape[0]
    health,target,corr,rel,summary,dec=evaluate_candidate_array(cname,cand,X,y,names,wks)
    out=Path(a.out_dir) if a.out_dir else Path(os.environ.get("BYBIT_FEATURE_LAB_OUT_DIR", str(out_root/"feature_lab")))/cname; out.mkdir(parents=True,exist_ok=True)
    def wcsv(path,rows):
        if not rows:return
        with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    wcsv(out/"candidate_health.csv",[health]); wcsv(out/"candidate_target_metrics.csv",target); wcsv(out/"candidate_corr_top_pairs.csv",corr); wcsv(out/"candidate_relative_report.csv",[rel]); wcsv(out/"candidate_decile_report.csv",dec)
    summary.update({"mode":a.mode,"expr":a.expr if a.mode=="expr" else None,"out_root":str(out_root)})
    (out/"feature_lab_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    (out/"feature_lab.log").write_text(f"candidate={cname} n_rows={X.shape[0]}\n")

if __name__=="__main__":
    main()
