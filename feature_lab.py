#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, csv, importlib, json, os
from pathlib import Path
import numpy as np
from feature_audit import (
    systematic_positions, proportional_week_targets, resolve_week_meta_path,
    read_json, load_labels_for_positions, load_features_by_row_idx, parse_feature_name,
)

SEED=int(os.environ.get("BYBIT_FEATURE_LAB_SEED","17")); MAX_TOTAL=int(os.environ.get("BYBIT_FEATURE_LAB_MAX_ROWS_TOTAL","1000000")); MAX_PER_WEEK=int(os.environ.get("BYBIT_FEATURE_LAB_MAX_ROWS_PER_WEEK","600000")); USE_AUX=os.environ.get("BYBIT_FEATURE_LAB_USE_AUX","0")=="1"; MI_MAX=int(os.environ.get("BYBIT_FEATURE_LAB_MI_MAX_ROWS","200000")); HIGH=float(os.environ.get("BYBIT_FEATURE_LAB_HIGH_CORR","0.95")); MED=float(os.environ.get("BYBIT_FEATURE_LAB_MED_CORR","0.90")); TOP=int(os.environ.get("BYBIT_FEATURE_LAB_TOP_CORR","50")); MIN_ABS_LABEL_EPS=float(os.environ.get("BYBIT_FEATURE_LAB_MIN_ABS_LABEL_EPS","0.0"))
DEFAULT_LOW_ABS_TRIM_FRACTION = 0.02
DEFAULT_HIGH_ABS_TRIM_FRACTION = 0.02

ALLOWED_NP_FUNCS={"log1p":np.log1p,"abs":np.abs,"sign":np.sign,"clip":np.clip,"maximum":np.maximum,"minimum":np.minimum}
HEALTH_FIELDS=["candidate","n_rows","finite_frac","nan_count","inf_count","mean","std","p01","p05","p50","p95","p99","min","max","abs_max","zero_frac","near_zero_frac_abs_lt_1e-6","week_std_cv"]
TARGET_FIELDS=["candidate","horizon_ms","mask_type","pearson_signed_return","spearman_signed_return","pearson_abs_return","spearman_abs_return","single_feature_auc_direction","single_feature_auc_direction_sign","single_feature_bal_acc_sign","single_feature_bal_acc_best_threshold","single_feature_bal_acc_best_threshold_value","mi_direction","mi_abs_return"]
CORR_FIELDS=["candidate","existing_feature","existing_feature_index","pearson","spearman","abs_pearson","abs_spearman","max_abs_corr","existing_family","existing_timescale_ms","existing_best_kept_auc","existing_best_abs_return_spearman","existing_target_score"]
RELATIVE_FIELDS=["candidate","max_corr_with_existing","most_correlated_existing_feature","most_correlated_existing_target_score","candidate_target_score","candidate_minus_existing_target_score","high_corr_duplicate","medium_corr_related","best_kept_auc_200ms","best_kept_auc_500ms","best_kept_auc_1000ms","best_kept_bal_acc_1000ms","best_abs_return_spearman","best_abs_return_spearman_200ms","best_abs_return_spearman_500ms","best_abs_return_spearman_1000ms","best_mi_direction","best_mi_abs_return","finite_frac","std","week_std_cv","decision","reason"]
DECILE_FIELDS=["candidate","horizon_ms","decile","n_rows","candidate_min","candidate_max","mean_signed_return","mean_abs_return","up_frac"]

class SafeNpNamespace:
    log1p=np.log1p; abs=np.abs; sign=np.sign; clip=np.clip; maximum=np.maximum; minimum=np.minimum

def resolve_trim_fractions(meta: dict) -> tuple[float, float]:
    low = (
        meta.get("low_abs_trim_fraction")
        or meta.get("LOW_ABS_TRIM_FRACTION")
        or meta.get("label_low_abs_trim_fraction")
        or DEFAULT_LOW_ABS_TRIM_FRACTION
    )
    high = (
        meta.get("high_abs_trim_fraction")
        or meta.get("HIGH_ABS_TRIM_FRACTION")
        or meta.get("label_high_abs_trim_fraction")
        or DEFAULT_HIGH_ABS_TRIM_FRACTION
    )
    return float(low), float(high)

def resolve_train_weeks(meta: dict) -> list[str]:
    try:
        weeks = meta["splits"]["cmssl"]["train"]["weeks"]
        if weeks:
            return [str(w) for w in weeks]
    except Exception:
        pass
    weeks = meta.get("train_weeks")
    if weeks:
        return [str(w) for w in weeks]
    raise ValueError("Could not resolve CMSSL train weeks from meta.json")

def resolve_week_label_count(week_meta: dict) -> int:
    for key in ("labels_total", "n_labels", "label_rows", "num_labels"):
        if key in week_meta:
            return int(week_meta[key])
    chunks = week_meta.get("label_chunks", [])
    if chunks:
        total = 0
        for c in chunks:
            if "n" in c:
                total += int(c["n"])
            elif "rows" in c:
                total += int(c["rows"])
            elif "num_rows" in c:
                total += int(c["num_rows"])
        if total > 0:
            return total
    raise ValueError("Could not resolve label count from week metadata")

def parse_corr_methods_from_env() -> set[str]:
    methods={x.strip().lower() for x in os.environ.get("BYBIT_FEATURE_LAB_CORR_METHODS","pearson,spearman").split(",") if x.strip()}
    valid={"pearson","spearman"}
    if not methods or not methods <= valid:
        raise ValueError(f"Invalid BYBIT_FEATURE_LAB_CORR_METHODS={methods}")
    return methods

def _safe_pearson_np(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3:return np.nan
    xv=x[m]-x[m].mean(); yv=y[m]-y[m].mean(); d=np.sqrt((xv*xv).sum()*(yv*yv).sum())
    return float((xv*yv).sum()/d) if d>0 else np.nan

def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x); order=np.argsort(x,kind="mergesort"); ranks=np.empty(len(x),dtype=np.float64); sx=x[order]; i=0
    while i<len(x):
        j=i+1
        while j<len(x) and sx[j]==sx[i]: j+=1
        ranks[order[i:j]]=0.5*(i+j-1)+1.0
        i=j
    return ranks

def _safe_spearman_np(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3:return np.nan
    return _safe_pearson_np(_rankdata_average(x[m]),_rankdata_average(y[m]))

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

def deterministic_subsample_indices(n:int,max_n:int,seed:int)->np.ndarray:
    if n<=max_n:return np.arange(n,dtype=np.int64)
    rng=np.random.default_rng(seed)
    return np.sort(rng.choice(n,size=max_n,replace=False))

def _mutual_info_optional(x,y,is_cls):
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        f = mutual_info_classif if is_cls else mutual_info_regression
        return float(f(x.reshape(-1,1), y, discrete_features=False, random_state=SEED)[0])
    except Exception:
        return np.nan

def side_specific_keep_mask(y_h: np.ndarray, low_frac: float, high_frac: float) -> np.ndarray:
    y=np.asarray(y_h,dtype=np.float64); finite=np.isfinite(y); keep=np.zeros_like(finite,dtype=bool)
    for side_sign in (1.0,-1.0):
        side=finite&(np.sign(y)==side_sign); abs_side=np.abs(y[side])
        if abs_side.size==0: continue
        lo=np.quantile(abs_side,low_frac) if low_frac>0 else -np.inf
        hi=np.quantile(abs_side,1.0-high_frac) if high_frac>0 else np.inf
        idx=np.where(side)[0]; keep[idx]=(abs_side>=lo)&(abs_side<=hi)
    return keep

def validate_expr_ast(expr: str, feature_names: set[str]) -> ast.Expression:
    tree=ast.parse(expr,mode="eval")
    allowed=(ast.Expression,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.UnaryOp,ast.UAdd,ast.USub,ast.Name,ast.Load,ast.Constant,ast.Call,ast.Attribute)
    for n in ast.walk(tree):
        if not isinstance(n,allowed): raise ValueError(f"Disallowed syntax: {type(n).__name__}")
        if isinstance(n,ast.Name) and n.id not in feature_names|{"np"}: raise ValueError(f"Unknown name: {n.id}")
        if isinstance(n,ast.Attribute):
            if not (isinstance(n.value,ast.Name) and n.value.id=="np" and n.attr in ALLOWED_NP_FUNCS):
                raise ValueError("Only np.<allowed_func> attribute access is permitted")
        if isinstance(n,ast.Call):
            if not (isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="np" and n.func.attr in ALLOWED_NP_FUNCS):
                raise ValueError("Only np.<allowed_func>(...) calls are allowed")
    return tree

def eval_expr(expr,X,names):
    tree=validate_expr_ast(expr,set(names))
    safe_locals={n:X[:,i] for i,n in enumerate(names)}
    code=compile(tree,"<feature_lab_expr>","eval")
    out=eval(code,{"__builtins__":{},"np":SafeNpNamespace},safe_locals)
    return np.asarray(out,dtype=np.float32)

def select_feature_names_and_idx(meta:dict,use_aux:bool,X_cols:int|None=None):
    all_names=list(meta.get("feature_names",[])); core_dim=int(meta.get("feature_dim_core",len(all_names))); total_dim=int(meta.get("feature_dim_total",len(all_names)))
    aux_names=list(meta.get("aux_names") or meta.get("feature_aux_tail") or [])
    if len(all_names)<total_dim and aux_names:
        names=all_names+aux_names
    else:
        names=all_names[:total_dim]
    if len(names)<total_dim: raise ValueError("Insufficient feature names for feature_dim_total")
    if use_aux: idx=np.arange(total_dim,dtype=np.int64); out_names=names[:total_dim]
    else: idx=np.arange(core_dim,dtype=np.int64); out_names=names[:core_dim]
    if len(out_names)!=len(idx): raise ValueError("name/index length mismatch")
    if X_cols is not None and X_cols!=len(out_names): raise ValueError(f"Loaded feature dim {X_cols} != expected {len(out_names)}")
    return out_names, idx

def compute_existing_feature_target_scores(X,y,feature_names,low_abs_trim_fraction,high_abs_trim_fraction):
    out={}
    for i,n in enumerate(feature_names):
        feat=X[:,i]; best_auc=np.nan; best_abs=np.nan
        for h_i,_h in enumerate([200,500,1000]):
            ys=y[:,h_i]; kept=side_specific_keep_mask(ys,low_abs_trim_fraction,high_abs_trim_fraction)&np.isfinite(feat)
            if not kept.any(): continue
            xx=feat[kept]; yy=ys[kept]; dd=(yy>0).astype(int); aa=np.abs(yy)
            auc=_binary_auc_np(dd,xx); auc=max(auc,1-auc) if np.isfinite(auc) else np.nan
            abs_sp=abs(_safe_spearman_np(xx,aa))
            best_auc=np.nanmax([best_auc,auc]); best_abs=np.nanmax([best_abs,abs_sp])
        score=max(abs(best_auc-0.5) if np.isfinite(best_auc) else 0.0,0.5*abs(best_abs) if np.isfinite(best_abs) else 0.0)
        out[n]={"existing_best_kept_auc":best_auc,"existing_best_abs_return_spearman":best_abs,"existing_target_score":float(score)}
    return out

def candidate_decile_bins(candidate: np.ndarray, n_deciles: int = 10) -> np.ndarray:
    q = np.quantile(candidate, np.linspace(0.0, 1.0, n_deciles + 1))
    if np.unique(q).size < n_deciles + 1:
        order = np.argsort(candidate, kind="mergesort")
        bins = np.empty(len(candidate), dtype=np.int64)
        splits = np.array_split(order, n_deciles)
        for d, idx in enumerate(splits):
            bins[idx] = d
        return bins
    return np.digitize(candidate, q[1:-1], right=True).astype(np.int64)

def evaluate_candidate_array(candidate_name,candidate,X,y,feature_names,week_keys,*,low_abs_trim_fraction: float = DEFAULT_LOW_ABS_TRIM_FRACTION,high_abs_trim_fraction: float = DEFAULT_HIGH_ABS_TRIM_FRACTION):
    CORR_METHODS=parse_corr_methods_from_env()
    if candidate.shape!=(X.shape[0],): raise ValueError("candidate shape mismatch")
    finite=np.isfinite(candidate)
    if finite.mean()<1.0: raise ValueError("candidate has nonfinite values")
    if float(np.std(candidate))<1e-8: raise ValueError("candidate nearly constant")
    health={"candidate":candidate_name,"n_rows":int(len(candidate)),"finite_frac":float(np.isfinite(candidate).mean()),"nan_count":int(np.isnan(candidate).sum()),"inf_count":int(np.isinf(candidate).sum()),"mean":float(np.mean(candidate)),"std":float(np.std(candidate)),"p01":float(np.quantile(candidate, 0.01)),"p05":float(np.quantile(candidate, 0.05)),"p50":float(np.quantile(candidate, 0.50)),"p95":float(np.quantile(candidate, 0.95)),"p99":float(np.quantile(candidate, 0.99)),"min":float(np.min(candidate)),"max":float(np.max(candidate)),"abs_max":float(np.max(np.abs(candidate))),"zero_frac":float((candidate == 0).mean()),"near_zero_frac_abs_lt_1e-6":float((np.abs(candidate) < 1e-6).mean()),"week_std_cv":0.0}
    wstds=[]; mean_by_week={}; std_by_week={}
    for wk in sorted(set(week_keys)):
        m=np.array([w==wk for w in week_keys]); mean_by_week[wk]=float(candidate[m].mean()); std_by_week[wk]=float(candidate[m].std()); wstds.append(std_by_week[wk])
    health["week_std_cv"]=float(np.std(wstds)/(np.mean(wstds)+1e-12))
    target_lookup=compute_existing_feature_target_scores(X,y,feature_names,low_abs_trim_fraction,high_abs_trim_fraction)
    corr=[]
    for i,n in enumerate(feature_names):
        p=_safe_pearson_np(candidate,X[:,i]) if "pearson" in CORR_METHODS else np.nan
        s=_safe_spearman_np(candidate,X[:,i]) if "spearman" in CORR_METHODS else np.nan
        vals=[abs(p) if np.isfinite(p) else np.nan,abs(s) if np.isfinite(s) else np.nan]; m=np.nanmax(vals)
        parsed=parse_feature_name(n)
        corr.append((m,{"candidate":candidate_name,"existing_feature":n,"existing_feature_index":i,"pearson":p,"spearman":s,"abs_pearson":abs(p) if np.isfinite(p) else np.nan,"abs_spearman":abs(s) if np.isfinite(s) else np.nan,"max_abs_corr":m,"existing_family":parsed.get("family","unknown"),"existing_timescale_ms":parsed.get("timescale_ms",np.nan),**target_lookup.get(n,{})}))
    corr_rows=[r[1] for r in sorted(corr,key=lambda z:z[0],reverse=True)[:TOP]]
    horizons=[200,500,1000]; target_rows=[]; dec=[]
    per_h_auc={}; per_h_abs={}; per_h_bal={}
    bins = candidate_decile_bins(candidate, 10)
    for h_i,h in enumerate(horizons):
        ys=y[:,h_i]; direction=(ys>0).astype(int); absr=np.abs(ys)
        finite=np.isfinite(ys)&np.isfinite(candidate)
        nonzero=finite&(np.abs(ys)>MIN_ABS_LABEL_EPS)
        kept=side_specific_keep_mask(ys,low_abs_trim_fraction,high_abs_trim_fraction)&np.isfinite(candidate)
        masks={"all_finite":finite,"nonzero":nonzero,"kept":kept}
        mi_dir_kept=np.nan; mi_abs_kept=np.nan
        if kept.any():
            xk=candidate[kept]; yk=ys[kept]; dk=(yk>0).astype(np.int64); ak=np.abs(yk); sub=deterministic_subsample_indices(len(xk),MI_MAX,SEED+h_i)
            mi_dir_kept=_mutual_info_optional(xk[sub],dk[sub],True); mi_abs_kept=_mutual_info_optional(xk[sub],ak[sub],False)
        for mk,m in masks.items():
            xx=candidate[m]; yy=ys[m]; dd=direction[m]; aa=absr[m]
            auc=_binary_auc_np(dd,xx); bal,sign,thr=_bal_acc_best_threshold_np(dd,xx)
            target_rows.append({"candidate":candidate_name,"horizon_ms":h,"mask_type":mk,"pearson_signed_return":_safe_pearson_np(xx,yy),"spearman_signed_return":_safe_spearman_np(xx,yy),"pearson_abs_return":_safe_pearson_np(xx,aa),"spearman_abs_return":_safe_spearman_np(xx,aa),"single_feature_auc_direction":max(auc,1-auc) if np.isfinite(auc) else np.nan,"single_feature_auc_direction_sign":1 if (np.isfinite(auc) and auc>=0.5) else (-1 if np.isfinite(auc) else 0),"single_feature_bal_acc_sign":sign,"single_feature_bal_acc_best_threshold":bal,"single_feature_bal_acc_best_threshold_value":thr,"mi_direction":mi_dir_kept if mk=="kept" else np.nan,"mi_abs_return":mi_abs_kept if mk=="kept" else np.nan})
        for d in range(10):
            m = (bins == d) & np.isfinite(ys)
            dec.append({"candidate": candidate_name, "horizon_ms": h, "decile": d, "n_rows": int(m.sum()), "candidate_min": float(np.min(candidate[m])) if m.any() else np.nan, "candidate_max": float(np.max(candidate[m])) if m.any() else np.nan, "mean_signed_return": float(np.mean(ys[m])) if m.any() else np.nan, "mean_abs_return": float(np.mean(np.abs(ys[m]))) if m.any() else np.nan, "up_frac": float(np.mean(ys[m] > 0)) if m.any() else np.nan})
        per_h_auc[h]=max((r["single_feature_auc_direction"] for r in target_rows if r["horizon_ms"]==h and r["mask_type"]=="kept"),default=np.nan)
        per_h_abs[h]=max((abs(r["spearman_abs_return"]) for r in target_rows if r["horizon_ms"]==h and r["mask_type"]=="kept"),default=np.nan)
        per_h_bal[h]=max((r["single_feature_bal_acc_best_threshold"] for r in target_rows if r["horizon_ms"]==h and r["mask_type"]=="kept"),default=np.nan)
    max_corr=float(corr_rows[0]["max_abs_corr"]) if corr_rows else np.nan
    best_auc=max(per_h_auc.values()) if per_h_auc else np.nan; best_abs=max(per_h_abs.values()) if per_h_abs else np.nan
    candidate_target_score=max(abs(best_auc-0.5) if np.isfinite(best_auc) else 0.0,0.5*abs(best_abs) if np.isfinite(best_abs) else 0.0)
    existing_score=(corr_rows[0].get("existing_target_score") if corr_rows else np.nan)
    decision,reason="needs_ablation","plausible"
    if health["finite_frac"]<1.0: decision,reason="reject","nonfinite_candidate"
    elif health["std"]<1e-8: decision,reason="reject","near_constant_candidate"
    elif max_corr>=HIGH:
        if candidate_target_score <= (existing_score*1.03 if np.isfinite(existing_score) else np.inf): decision,reason="reject","high_corr_duplicate_without_target_improvement"
        else: decision,reason="needs_ablation","high_corr_but_target_improved"
    elif best_auc<=0.55 and best_abs<=0.03: decision,reason="reject","weak_direction_and_magnitude"
    elif max_corr<MED and (best_auc>0.56 or best_abs>0.03): decision,reason="promote_candidate","useful_signal_and_low_redundancy"
    best_mi_direction=max((r["mi_direction"] for r in target_rows if r["mask_type"]=="kept" and np.isfinite(r["mi_direction"])),default=np.nan)
    best_mi_abs_return=max((r["mi_abs_return"] for r in target_rows if r["mask_type"]=="kept" and np.isfinite(r["mi_abs_return"])),default=np.nan)
    rel={"candidate":candidate_name,"max_corr_with_existing":max_corr,"most_correlated_existing_feature":corr_rows[0]["existing_feature"] if corr_rows else "","most_correlated_existing_target_score":existing_score,"candidate_target_score":candidate_target_score,"candidate_minus_existing_target_score":candidate_target_score-existing_score if np.isfinite(existing_score) else np.nan,"high_corr_duplicate":bool(max_corr >= HIGH),"medium_corr_related":bool(MED <= max_corr < HIGH),"best_kept_auc_200ms":per_h_auc.get(200, np.nan),"best_kept_auc_500ms":per_h_auc.get(500, np.nan),"best_kept_auc_1000ms":per_h_auc.get(1000, np.nan),"best_kept_bal_acc_1000ms":per_h_bal.get(1000, np.nan),"best_abs_return_spearman":best_abs,"best_abs_return_spearman_200ms":per_h_abs.get(200, np.nan),"best_abs_return_spearman_500ms":per_h_abs.get(500, np.nan),"best_abs_return_spearman_1000ms":per_h_abs.get(1000, np.nan),"best_mi_direction":best_mi_direction,"best_mi_abs_return":best_mi_abs_return,"finite_frac":health["finite_frac"],"std":health["std"],"week_std_cv":health["week_std_cv"],"decision":decision,"reason":reason}
    summary={"schema":"feature_lab_v1","candidate":candidate_name,"n_rows":int(len(candidate)),"feature_dim":int(X.shape[1]),"use_aux":USE_AUX,"health":{"finite_frac":1.0,"std":health["std"],"week_std_cv":health["week_std_cv"],"mean_by_week":mean_by_week,"std_by_week":std_by_week},"best_direction":{"best_kept_auc":best_auc,"best_kept_auc_200ms":per_h_auc.get(200,np.nan),"best_kept_auc_500ms":per_h_auc.get(500,np.nan),"best_kept_auc_1000ms":per_h_auc.get(1000,np.nan),"best_kept_bal_acc_1000ms":per_h_bal.get(1000,np.nan)},"best_magnitude":{"best_abs_return_spearman":best_abs,"best_abs_return_spearman_200ms":per_h_abs.get(200,np.nan),"best_abs_return_spearman_500ms":per_h_abs.get(500,np.nan),"best_abs_return_spearman_1000ms":per_h_abs.get(1000,np.nan),"best_mi_abs_return":best_mi_abs_return},"correlation":{"max_abs_corr":max_corr},"decision":decision,"reason":reason}
    return health,target_rows,corr_rows,rel,summary,dec

def wcsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",required=True,choices=["expr","plugin","event"]); ap.add_argument("--candidate-name",default=""); ap.add_argument("--expr",default=""); ap.add_argument("--candidate-module",default="feature_candidates"); ap.add_argument("--candidate-class",default=""); ap.add_argument("--out-dir",default="")
    a=ap.parse_args()
    if a.mode=="event": raise NotImplementedError("event mode is intentionally deferred; use expr/plugin mode first")
    out_root=Path(os.environ.get("BYBIT_OUT_ROOT","").strip()); meta=read_json(out_root/"meta.json")
    low_trim, high_trim = resolve_trim_fractions(meta)
    names,idx=select_feature_names_and_idx(meta,USE_AUX)
    train_weeks = resolve_train_weeks(meta)
    labels_by_week = {}
    week_metas = {}
    for wk in train_weeks:
        wm = read_json(resolve_week_meta_path(out_root, meta["weeks_meta"], wk))
        week_metas[wk] = wm
        labels_by_week[wk] = resolve_week_label_count(wm)
    targets=proportional_week_targets(labels_by_week,MAX_TOTAL,MAX_PER_WEEK)
    Xs=[]; Ys=[]; wks=[]
    for wi,wk in enumerate(train_weeks):
        n=targets.get(wk,0)
        if n<=0: continue
        wm=week_metas[wk]
        n_labels = resolve_week_label_count(wm)
        pos=systematic_positions(n_labels,n,SEED+wi)
        row_idx,_,y=load_labels_for_positions(out_root,wk,wm["label_chunks"],pos); x=load_features_by_row_idx(out_root,wk,wm["feature_chunks"],row_idx,idx)
        Xs.append(x); Ys.append(y.astype(np.float32)); wks.extend([wk]*len(x))
    if not Xs:
        raise RuntimeError("No feature_lab rows sampled; check train week metadata and sampling config")
    X=np.concatenate(Xs,0); y=np.concatenate(Ys,0); names,_=select_feature_names_and_idx(meta,USE_AUX,X.shape[1])
    if a.mode=="expr": cname=a.candidate_name or "expr_candidate"; cand=eval_expr(a.expr,X,names)
    else:
        mod=importlib.import_module(a.candidate_module); cls=getattr(mod,a.candidate_class); obj=cls(); cname=getattr(obj,"name",a.candidate_class); cand=np.asarray(obj.compute(X,names),dtype=np.float32)
    health,target,corr,rel,summary,dec=evaluate_candidate_array(cname,cand,X,y,names,wks,low_abs_trim_fraction=low_trim,high_abs_trim_fraction=high_trim)
    out=Path(a.out_dir) if a.out_dir else Path(os.environ.get("BYBIT_FEATURE_LAB_OUT_DIR", str(out_root/"feature_lab")))/cname; out.mkdir(parents=True,exist_ok=True)
    wcsv(out/"candidate_health.csv",[health],HEALTH_FIELDS); wcsv(out/"candidate_target_metrics.csv",target,TARGET_FIELDS); wcsv(out/"candidate_corr_top_pairs.csv",corr,CORR_FIELDS); wcsv(out/"candidate_relative_report.csv",[rel],RELATIVE_FIELDS); wcsv(out/"candidate_decile_report.csv",dec,DECILE_FIELDS)
    summary.update({"mode":a.mode,"expr":a.expr if a.mode=="expr" else None,"out_root":str(out_root)})
    (out/"feature_lab_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    (out/"feature_lab.log").write_text(f"candidate={cname} n_rows={X.shape[0]}\n")

if __name__=="__main__": main()
