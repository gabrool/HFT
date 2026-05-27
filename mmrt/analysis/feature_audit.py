"""Storage-backed feature health, drift, and redundancy audit for MMRT.

This module reads existing MMRT storage splits and audits already-materialized
feature columns. It computes train-only feature redundancy/correlation,
split-level feature health, and train-vs-val/test distribution drift. It does
not parse Tardis CSV, compute market features, build labels, create splits,
train models, evaluate predictions, select model features, or mutate storage
manifests.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np
import pyarrow as pa

from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.linear import extractors as ex
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd

FEATURE_AUDIT_SCHEMA_VERSION = 1
DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS = 100_000
DEFAULT_FEATURE_AUDIT_BATCH_SIZE = rd.DEFAULT_BATCH_SIZE
DEFAULT_FEATURE_AUDIT_SUMMARY_FILENAME = "feature_audit_summary.json"
DEFAULT_FEATURE_AUDIT_HEALTH_FILENAME = "feature_health.csv"
DEFAULT_FEATURE_AUDIT_DRIFT_FILENAME = "feature_train_val_drift.csv"
DEFAULT_FEATURE_AUDIT_FAMILY_FILENAME = "feature_family_summary.csv"
DEFAULT_FEATURE_AUDIT_CORR_PAIRS_FILENAME = "feature_corr_top_pairs.csv"
DEFAULT_FEATURE_AUDIT_CLUSTERS_FILENAME = "feature_clusters.csv"
DEFAULT_FEATURE_AUDIT_CLUSTER_SUMMARY_FILENAME = "feature_cluster_summary.json"
DEFAULT_LOW_VARIANCE_STD_THRESHOLD = 1e-8
DEFAULT_HIGH_CORR_THRESHOLD = 0.97
DEFAULT_MIN_CORR_OUTPUT_THRESHOLD = 0.90
DEFAULT_MAX_CORR_PAIRS = 1_000
DEFAULT_DRIFT_MEAN_Z_THRESHOLD = 1.0
DEFAULT_DRIFT_STD_RATIO_LOW = 0.5
DEFAULT_DRIFT_STD_RATIO_HIGH = 2.0
ALLOWED_SPLITS = ("train", "val", "test")
ALLOWED_HEALTH_STATUSES = ("ok", "low_variance")
ALLOWED_DRIFT_STATUSES = ("ok", "distribution_shift", "low_variance_train")
ALLOWED_PAIR_STATUSES = ("moderate_redundancy", "high_redundancy")

def _require_positive_int(value:int,name:str)->int:
    if isinstance(value,bool) or not isinstance(value,int) or value<=0: raise ValueError(f"{name} must be a positive int")
    return value

def _require_nonnegative_int(value:int,name:str)->int:
    if isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError(f"{name} must be a nonnegative int")
    return value

def _require_bool(value:bool,name:str)->bool:
    if not isinstance(value,bool): raise ValueError(f"{name} must be a bool")
    return value

def _require_non_empty_str(value:str,name:str)->str:
    if not isinstance(value,str) or not value.strip(): raise ValueError(f"{name} must be a non-empty str")
    return value.strip()

def _require_finite_float(value: float, name: str, *, allow_nan: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)): raise ValueError(f"{name} must be a float")
    fv=float(value)
    if math.isnan(fv):
        if allow_nan: return fv
        raise ValueError(f"{name} must be finite")
    if not math.isfinite(fv): raise ValueError(f"{name} must be finite")
    return fv

def _role_to_str(role: SplitRole | str) -> str:
    rv=role.value if isinstance(role,SplitRole) else role
    if rv not in ALLOWED_SPLITS: raise ValueError("split must be one of train/val/test")
    return rv

def _json_safe(value: object) -> object:
    if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    raise ValueError(f"unsupported JSON type: {type(value)!r}")

@dataclass(frozen=True, slots=True)
class FeatureAuditConfig:
    batch_size: int = DEFAULT_FEATURE_AUDIT_BATCH_SIZE
    validate_dataset_on_open: bool = True
    max_sample_rows_per_split: int = DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS
    feature_columns: tuple[str, ...] | None = None
    extractor_dtype: str = ex.DEFAULT_EXTRACTOR_DTYPE
    low_variance_std_threshold: float = DEFAULT_LOW_VARIANCE_STD_THRESHOLD
    high_corr_threshold: float = DEFAULT_HIGH_CORR_THRESHOLD
    min_corr_output_threshold: float = DEFAULT_MIN_CORR_OUTPUT_THRESHOLD
    max_corr_pairs: int = DEFAULT_MAX_CORR_PAIRS
    drift_mean_z_threshold: float = DEFAULT_DRIFT_MEAN_Z_THRESHOLD
    drift_std_ratio_low: float = DEFAULT_DRIFT_STD_RATIO_LOW
    drift_std_ratio_high: float = DEFAULT_DRIFT_STD_RATIO_HIGH
    def __post_init__(self)->None:
        _require_positive_int(self.batch_size,"batch_size"); _require_bool(self.validate_dataset_on_open,"validate_dataset_on_open"); _require_nonnegative_int(self.max_sample_rows_per_split,"max_sample_rows_per_split")
        if self.feature_columns is not None:
            if not isinstance(self.feature_columns,tuple) or not self.feature_columns: raise ValueError("feature_columns must be non-empty tuple[str,...] when provided")
            for c in self.feature_columns: _require_non_empty_str(c,"feature_columns")
            if len(set(self.feature_columns))!=len(self.feature_columns): raise ValueError("feature_columns must not contain duplicates")
        if self.extractor_dtype not in ex.ALLOWED_EXTRACTOR_DTYPES: raise ValueError("invalid extractor_dtype")
        if _require_finite_float(self.low_variance_std_threshold,"low_variance_std_threshold")<=0: raise ValueError("low_variance_std_threshold must be > 0")
        mn=_require_finite_float(self.min_corr_output_threshold,"min_corr_output_threshold"); hi=_require_finite_float(self.high_corr_threshold,"high_corr_threshold")
        if not (0<mn<=hi<1): raise ValueError("require 0 < min_corr_output_threshold <= high_corr_threshold < 1")
        _require_positive_int(self.max_corr_pairs,"max_corr_pairs")
        if _require_finite_float(self.drift_mean_z_threshold,"drift_mean_z_threshold")<=0: raise ValueError("drift_mean_z_threshold must be > 0")
        low=_require_finite_float(self.drift_std_ratio_low,"drift_std_ratio_low"); high=_require_finite_float(self.drift_std_ratio_high,"drift_std_ratio_high")
        if not (0<low<1<high): raise ValueError("require 0 < drift_std_ratio_low < 1 < drift_std_ratio_high")

@dataclass(frozen=True, slots=True)
class _FeatureMeta:
    column:str; canonical_name:str; feature_index:int; source:str; owner:str; family:str; unit:str; transform_key:str; required_book_depth:int

def _feature_meta_from_column(column:str)->_FeatureMeta:
    col=_require_non_empty_str(column,"column")
    if not col.startswith(mf.FEATURE_COLUMN_PREFIX): raise ValueError("feature column must have x_ prefix")
    cname=col[len(mf.FEATURE_COLUMN_PREFIX):]
    spec=specs.feature_spec_by_name(cname)
    return _FeatureMeta(col,cname,spec.index,spec.source.value,spec.owner.value,spec.family.value,spec.unit.value,spec.transform_key.value,spec.required_book_depth)

@dataclass(slots=True)
class _StreamingFeatureStats:
    n_rows:int; n_features:int; mean:np.ndarray; m2:np.ndarray; min_value:np.ndarray; max_value:np.ndarray
    @classmethod
    def empty(cls,n_features:int)->"_StreamingFeatureStats":
        _require_positive_int(n_features,"n_features")
        return cls(0,n_features,np.zeros(n_features,np.float64),np.zeros(n_features,np.float64),np.full(n_features,np.inf),np.full(n_features,-np.inf))
    def update(self,X:np.ndarray)->None:
        arr=np.asarray(X,dtype=np.float64)
        if arr.ndim!=2 or arr.shape[1]!=self.n_features or not np.isfinite(arr).all(): raise ValueError("invalid X")
        if arr.shape[0]==0:return
        bn=int(arr.shape[0]); bm=arr.mean(0); centered=arr-bm; bm2=np.sum(centered*centered,0)
        if self.n_rows==0:
            self.mean=bm; self.m2=bm2
        else:
            tn=self.n_rows+bn; delta=bm-self.mean
            self.mean=self.mean+delta*(bn/tn)
            self.m2=np.maximum(self.m2+bm2+delta*delta*((self.n_rows*bn)/tn),0.0)
        self.min_value=np.minimum(self.min_value,arr.min(0)); self.max_value=np.maximum(self.max_value,arr.max(0)); self.n_rows+=bn
    def variance(self)->np.ndarray:
        if self.n_rows<=1: return np.zeros(self.n_features,np.float64)
        return self.m2/float(self.n_rows-1)
    def std(self)->np.ndarray: return np.sqrt(np.maximum(self.variance(),0.0))

@dataclass(slots=True)
class _StreamingTrainCorrelationStats:
    n_rows:int;n_features:int;sum_x:np.ndarray;sum_x2:np.ndarray;cross_x:np.ndarray
    @classmethod
    def empty(cls,n_features:int)->"_StreamingTrainCorrelationStats":
        return cls(0,n_features,np.zeros(n_features,np.float64),np.zeros(n_features,np.float64),np.zeros((n_features,n_features),np.float64))
    def update(self,X:np.ndarray)->None:
        arr=np.asarray(X,dtype=np.float64)
        if arr.ndim!=2 or arr.shape[1]!=self.n_features or not np.isfinite(arr).all(): raise ValueError("invalid X")
        self.n_rows += int(arr.shape[0]); self.sum_x += arr.sum(0); self.sum_x2 += (arr*arr).sum(0); self.cross_x += arr.T @ arr
    def correlation_matrix(self,*,low_variance_std_threshold:float)->np.ndarray:
        n=self.n_rows
        corr=np.full((self.n_features,self.n_features),np.nan,dtype=np.float64)
        if n<=1: return corr
        cov=(self.cross_x - np.outer(self.sum_x,self.sum_x)/n)/(n-1)
        var=(self.sum_x2 - (self.sum_x*self.sum_x)/n)/(n-1)
        std=np.sqrt(np.maximum(var,0.0)); denom=np.outer(std,std)
        with np.errstate(divide='ignore',invalid='ignore'): corr=np.where(denom>0,cov/denom,np.nan)
        active=std>=low_variance_std_threshold
        corr[~active,:]=np.nan; corr[:,~active]=np.nan
        idx=np.where(active)[0]; corr[idx,idx]=1.0
        return corr

# dataclasses simplified validations
@dataclass(frozen=True, slots=True)
class FeatureHealthRecord: split:str; feature:str; canonical_name:str; feature_index:int; source:str; owner:str; family:str; unit:str; transform_key:str; required_book_depth:int; n_rows:int; n_sample_rows:int; raw_mean:float; raw_std:float; raw_min:float; raw_max:float; raw_p01:float; raw_p50:float; raw_p99:float; raw_abs_p99:float; low_variance:bool; status:str
@dataclass(frozen=True, slots=True)
class FeatureDriftRecord: split:str; feature:str; canonical_name:str; feature_index:int; source:str; owner:str; family:str; train_mean:float; train_std:float; split_mean:float; split_std:float; mean_shift_train_std:float; std_ratio:float; train_p50:float; split_p50:float; p50_shift_train_std:float; status:str
@dataclass(frozen=True, slots=True)
class FeatureCorrelationPairRecord: feature_a:str; feature_b:str; canonical_a:str; canonical_b:str; index_a:int; index_b:int; source_a:str; source_b:str; family_a:str; family_b:str; corr:float; abs_corr:float; same_source:bool; same_family:bool; status:str
@dataclass(frozen=True, slots=True)
class FeatureClusterRecord: feature:str; canonical_name:str; feature_index:int; source:str; family:str; cluster_id:int; cluster_size:int; representative_feature:str; max_abs_corr_in_cluster:float
@dataclass(frozen=True, slots=True)
class FeatureFamilySummaryRecord: split:str; family:str; n_features:int; low_variance_count:int; mean_raw_std:float; median_raw_abs_p99:float; train_high_corr_pair_count:float; train_max_abs_corr:float; train_mean_abs_corr:float
@dataclass(frozen=True, slots=True)
class FeatureAuditSplitSummary: split:str; manifest_row_count:int; scanned_rows:int; sampled_rows:int; sample_stride:int|None; n_features:int; low_variance_count:int; drift_count:int
@dataclass(frozen=True, slots=True)
class FeatureAuditResult:
    schema_version:int; dataset_root:str; dataset_id:str; manifest_hash:str; feature_schema_hash:str; config:dict[str,object]; splits:dict[str,FeatureAuditSplitSummary]; health_records:tuple[FeatureHealthRecord,...]; drift_records:tuple[FeatureDriftRecord,...]; correlation_pairs:tuple[FeatureCorrelationPairRecord,...]; cluster_records:tuple[FeatureClusterRecord,...]; family_records:tuple[FeatureFamilySummaryRecord,...]; cluster_summary:dict[str,object]; warnings:tuple[str,...]
    def as_dict(self)->dict[str,object]:
        return {"schema_version":self.schema_version,"dataset_root":self.dataset_root,"dataset_id":self.dataset_id,"manifest_hash":self.manifest_hash,"feature_schema_hash":self.feature_schema_hash,"config":_json_safe(self.config),"splits":{k:asdict(v) for k,v in self.splits.items()},"correlation_summary":self.cluster_summary,"health_summary":{"low_variance_train_count":sum(1 for r in self.health_records if r.split=="train" and r.low_variance),"drift_val_count":sum(1 for r in self.drift_records if r.split=="val" and r.status=="distribution_shift"),"drift_test_count":sum(1 for r in self.drift_records if r.split=="test" and r.status=="distribution_shift")},"warnings":list(self.warnings)}

def _scan_split_features(reader,manifest,role,config):
    role_s=_role_to_str(role); cols=ex.resolve_feature_columns(manifest,config.feature_columns); metas=[_feature_meta_from_column(c) for c in cols]
    stats=_StreamingFeatureStats.empty(len(cols)); corr_stats=_StreamingTrainCorrelationStats.empty(len(cols)) if role_s=="train" else None
    extractor=ex.IdentityFeatureExtractor(ex.LinearFeatureExtractorConfig(feature_columns=cols,output_dtype=config.extractor_dtype),manifest=manifest)
    mcount=sum(e.end_row-e.start_row for e in reader.split_entries(role)); stride=max(1,math.ceil(mcount/config.max_sample_rows_per_split)) if config.max_sample_rows_per_split>0 and mcount>0 else None
    srows=[]; row_pos=0
    for b in reader.iter_split_batches(role,columns=cols,batch_size=config.batch_size):
        fb=extractor.transform_table(b); X=fb.X; stats.update(X)
        if corr_stats is not None: corr_stats.update(X)
        if stride is not None and X.shape[0]>0:
            idx=np.arange(X.shape[0],dtype=np.int64)+row_pos
            mask=(idx%stride)==0
            if np.any(mask): srows.append(X[mask].astype(np.float32,copy=True))
        row_pos += X.shape[0]
    sample=np.vstack(srows) if srows else np.empty((0,len(cols)),np.float32)
    recs=[]; std=stats.std()
    for i,m in enumerate(metas):
        q=np.array([np.nan,np.nan,np.nan,np.nan]) if sample.shape[0]==0 else np.array([np.quantile(sample[:,i],0.01),np.quantile(sample[:,i],0.5),np.quantile(sample[:,i],0.99),np.quantile(np.abs(sample[:,i]),0.99)],dtype=np.float64)
        lv=bool(std[i]<config.low_variance_std_threshold); st="low_variance" if lv else "ok"
        recs.append(FeatureHealthRecord(role_s,m.column,m.canonical_name,m.feature_index,m.source,m.owner,m.family,m.unit,m.transform_key,m.required_book_depth,stats.n_rows,sample.shape[0],float(stats.mean[i]) if stats.n_rows else 0.0,float(std[i]) if stats.n_rows else 0.0,float(stats.min_value[i]) if stats.n_rows else np.nan,float(stats.max_value[i]) if stats.n_rows else np.nan,float(q[0]),float(q[1]),float(q[2]),float(q[3]),lv,st))
    summary=FeatureAuditSplitSummary(role_s,mcount,stats.n_rows,sample.shape[0],stride,len(cols),sum(r.low_variance for r in recs),0)
    return summary,recs,{"mean":stats.mean,"std":std,"p50":(np.nanmedian(sample,axis=0) if sample.shape[0] else np.full(len(cols),np.nan)),"n_rows":stats.n_rows,"feature_columns":cols,"corr_stats":corr_stats}

def run_feature_audit(dataset_root:str,*,config:FeatureAuditConfig|None=None)->FeatureAuditResult:
    dataset_root=_require_non_empty_str(dataset_root,"dataset_root"); cfg=config or FeatureAuditConfig(); reader=rd.open_dataset(dataset_root,validate_on_open=cfg.validate_dataset_on_open,batch_size=cfg.batch_size); manifest=reader.manifest; manifest.validate_against_current_code()
    if not reader.split_entries(SplitRole.TRAIN): raise ValueError("dataset must contain non-empty train split")
    if not reader.split_entries(SplitRole.VAL): raise ValueError("dataset must contain non-empty val split")
    splits={}; health=[]; drift=[]; warnings=[]; scan={}
    for role in (SplitRole.TRAIN,SplitRole.VAL,SplitRole.TEST):
        if role is SplitRole.TEST and not reader.split_entries(role): warnings.append("missing_test_split"); continue
        s,re,st=_scan_split_features(reader,manifest,role,cfg); splits[role.value]=s; health.extend(re); scan[role.value]=st
    corr=scan["train"]["corr_stats"].correlation_matrix(low_variance_std_threshold=cfg.low_variance_std_threshold)
    metas=[_feature_meta_from_column(c) for c in scan["train"]["feature_columns"]]
    pairs=[]
    for i in range(len(metas)):
        for j in range(i+1,len(metas)):
            c=corr[i,j]
            if np.isfinite(c) and abs(c)>=cfg.min_corr_output_threshold:
                pairs.append(FeatureCorrelationPairRecord(metas[i].column,metas[j].column,metas[i].canonical_name,metas[j].canonical_name,metas[i].feature_index,metas[j].feature_index,metas[i].source,metas[j].source,metas[i].family,metas[j].family,float(c),float(abs(c)),metas[i].source==metas[j].source,metas[i].family==metas[j].family,"high_redundancy" if abs(c)>=cfg.high_corr_threshold else "moderate_redundancy"))
    pairs.sort(key=lambda r:(-r.abs_corr,r.index_a,r.index_b)); pairs=tuple(pairs[:cfg.max_corr_pairs])
    parent=list(range(len(metas)))
    def f(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def u(a,b): ra,rb=f(a),f(b); parent[rb]=ra if ra!=rb else ra
    for i in range(len(metas)):
        for j in range(i+1,len(metas)):
            if np.isfinite(corr[i,j]) and abs(corr[i,j])>=cfg.high_corr_threshold: u(i,j)
    comps={}
    for i in range(len(metas)): comps.setdefault(f(i),[]).append(i)
    nons=[sorted(v) for v in comps.values() if len(v)>1]; nons.sort(key=lambda v:v[0]); cid_map={tuple(v):k for k,v in enumerate(nons)}
    clusters=[]
    for i,m in enumerate(metas):
        grp=sorted(comps[f(i)])
        if len(grp)==1: clusters.append(FeatureClusterRecord(m.column,m.canonical_name,m.feature_index,m.source,m.family,-1,1,m.column,float("nan")))
        else:
            rep=grp[0]; maxc=max(abs(corr[a,b]) for ai,a in enumerate(grp) for b in grp[ai+1:]); clusters.append(FeatureClusterRecord(m.column,m.canonical_name,m.feature_index,m.source,m.family,cid_map[tuple(grp)],len(grp),metas[rep].column,float(maxc)))
    for split in ("val","test"):
        if split not in scan: continue
        for i,m in enumerate(metas):
            tstd=float(scan["train"]["std"][i]); tmean=float(scan["train"]["mean"][i]); sstd=float(scan[split]["std"][i]); smean=float(scan[split]["mean"][i]); tp50=float(scan["train"]["p50"][i]); sp50=float(scan[split]["p50"][i])
            if tstd<cfg.low_variance_std_threshold: status="low_variance_train"; ms=sr=ps=float("nan")
            else:
                ms=(smean-tmean)/tstd; sr=sstd/tstd; ps=(sp50-tp50)/tstd; status="distribution_shift" if (abs(ms)>=cfg.drift_mean_z_threshold or sr<=cfg.drift_std_ratio_low or sr>=cfg.drift_std_ratio_high) else "ok"
            drift.append(FeatureDriftRecord(split,m.column,m.canonical_name,m.feature_index,m.source,m.owner,m.family,tmean,tstd,smean,sstd,ms,sr,tp50,sp50,ps,status))
    family=[]
    for split in splits:
        split_recs=[r for r in health if r.split==split]
        fams=sorted({r.family for r in split_recs})
        for fam in fams:
            rr=[r for r in split_recs if r.family==fam]; idx=[i for i,m in enumerate(metas) if m.family==fam]
            hc=mc=ma=float('nan')
            if split=="train" and len(idx)>=2:
                vals=[abs(corr[a,b]) for ai,a in enumerate(idx) for b in idx[ai+1:] if np.isfinite(corr[a,b])];
                if vals: hc=float(sum(v>=cfg.high_corr_threshold for v in vals)); mc=float(np.mean(vals)); ma=float(max(vals))
            family.append(FeatureFamilySummaryRecord(split,fam,len(rr),sum(r.low_variance for r in rr),float(np.mean([r.raw_std for r in rr])) if rr else float('nan'),float(np.median([r.raw_abs_p99 for r in rr])) if rr else float('nan'),hc,ma,mc))
    if any(r.low_variance for r in health if r.split=="train"): warnings.append("low_variance_train")
    if any(r.status=="distribution_shift" for r in drift if r.split=="val"): warnings.append("distribution_shift:val")
    if any(r.status=="distribution_shift" for r in drift if r.split=="test"): warnings.append("distribution_shift:test")
    if any(r.status=="high_redundancy" for r in pairs): warnings.append("high_correlation_train")
    csum={"pair_count":len(pairs),"high_redundancy_pair_count":sum(r.status=="high_redundancy" for r in pairs),"moderate_redundancy_pair_count":sum(r.status=="moderate_redundancy" for r in pairs),"cluster_count":len({r.cluster_id for r in clusters if r.cluster_id>=0}),"clustered_feature_count":sum(r.cluster_id>=0 for r in clusters),"max_abs_corr":float(max([r.abs_corr for r in pairs],default=float('nan')))}
    return FeatureAuditResult(FEATURE_AUDIT_SCHEMA_VERSION,dataset_root,manifest.dataset_id,manifest.content_hash(),manifest.feature_schema.get("feature_specs_hash",""),{"batch_size":cfg.batch_size,"validate_dataset_on_open":cfg.validate_dataset_on_open,"max_sample_rows_per_split":cfg.max_sample_rows_per_split,"feature_columns":list(cfg.feature_columns) if cfg.feature_columns else None,"extractor_dtype":cfg.extractor_dtype},splits,tuple(health),tuple(drift),pairs,tuple(clusters),tuple(family),csum,tuple(warnings))

def write_feature_audit_artifacts(result:FeatureAuditResult,output_dir:str,*,summary_filename:str=DEFAULT_FEATURE_AUDIT_SUMMARY_FILENAME,health_filename:str=DEFAULT_FEATURE_AUDIT_HEALTH_FILENAME,drift_filename:str=DEFAULT_FEATURE_AUDIT_DRIFT_FILENAME,family_filename:str=DEFAULT_FEATURE_AUDIT_FAMILY_FILENAME,corr_pairs_filename:str=DEFAULT_FEATURE_AUDIT_CORR_PAIRS_FILENAME,clusters_filename:str=DEFAULT_FEATURE_AUDIT_CLUSTERS_FILENAME,cluster_summary_filename:str=DEFAULT_FEATURE_AUDIT_CLUSTER_SUMMARY_FILENAME)->dict[str,str]:
    out=Path(_require_non_empty_str(output_dir,"output_dir")); out.mkdir(parents=True,exist_ok=True)
    for n,s in ((summary_filename,'.json'),(cluster_summary_filename,'.json'),(health_filename,'.csv'),(drift_filename,'.csv'),(family_filename,'.csv'),(corr_pairs_filename,'.csv'),(clusters_filename,'.csv')):
        if not _require_non_empty_str(n,n).endswith(s): raise ValueError(f"{n} must end with {s}")
    def wj(name,obj): p=out/name; t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=True)+"\n",encoding='utf-8'); t.replace(p); return str(p)
    def wc(name,recs,cls): p=out/name; t=p.with_suffix(p.suffix+'.tmp'); f=list(cls.__dataclass_fields__.keys());
    
    def _wc(name,recs,cls):
        p=out/name; t=p.with_suffix(p.suffix+'.tmp')
        with t.open('w',newline='',encoding='utf-8') as h:
            w=csv.DictWriter(h,fieldnames=list(cls.__dataclass_fields__.keys())); w.writeheader(); [w.writerow(asdict(r)) for r in recs]
        t.replace(p); return str(p)
    return {"summary_json":wj(summary_filename,result.as_dict()),"health_csv":_wc(health_filename,result.health_records,FeatureHealthRecord),"drift_csv":_wc(drift_filename,result.drift_records,FeatureDriftRecord),"family_csv":_wc(family_filename,result.family_records,FeatureFamilySummaryRecord),"corr_pairs_csv":_wc(corr_pairs_filename,result.correlation_pairs,FeatureCorrelationPairRecord),"clusters_csv":_wc(clusters_filename,result.cluster_records,FeatureClusterRecord),"cluster_summary_json":wj(cluster_summary_filename,result.cluster_summary)}

__all__=["DEFAULT_FEATURE_AUDIT_MAX_SAMPLE_ROWS","DEFAULT_FEATURE_AUDIT_SUMMARY_FILENAME","DEFAULT_FEATURE_AUDIT_HEALTH_FILENAME","DEFAULT_FEATURE_AUDIT_DRIFT_FILENAME","DEFAULT_FEATURE_AUDIT_FAMILY_FILENAME","DEFAULT_FEATURE_AUDIT_CORR_PAIRS_FILENAME","DEFAULT_FEATURE_AUDIT_CLUSTERS_FILENAME","DEFAULT_FEATURE_AUDIT_CLUSTER_SUMMARY_FILENAME","FeatureAuditConfig","FeatureHealthRecord","FeatureDriftRecord","FeatureFamilySummaryRecord","FeatureCorrelationPairRecord","FeatureClusterRecord","FeatureAuditSplitSummary","FeatureAuditResult","run_feature_audit","write_feature_audit_artifacts"]
