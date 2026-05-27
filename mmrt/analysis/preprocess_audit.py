"""Storage-backed preprocessing audit for MMRT linear models.

This module reads existing storage splits, fits the linear preprocessor on the
train split only, and audits raw/z-scored/clipped feature behavior on train,
val, and test splits. It does not parse Tardis CSV, compute market features,
build labels, create splits, train models, evaluate predictions, or mutate
storage manifests.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math

import numpy as np
import pyarrow as pa

from mmrt.contracts import SplitRole
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.linear import extractors as ex
from mmrt.linear import preprocess as pp

DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS = 100_000
DEFAULT_PREPROCESS_AUDIT_BATCH_SIZE = rd.DEFAULT_BATCH_SIZE
DEFAULT_PREPROCESS_AUDIT_SUMMARY_FILENAME = "preprocess_audit_summary.json"
DEFAULT_PREPROCESS_AUDIT_FEATURES_FILENAME = "preprocess_audit_features.csv"
PREPROCESS_AUDIT_SCHEMA_VERSION = 1
CLIP_REVIEW_RATE = 0.001
CLIP_EXCESSIVE_RATE = 0.01
NEAR_CLIP_FRACTION = 0.80
DRIFT_MEAN_Z_REVIEW = 1.0
DRIFT_STD_RATIO_LOW = 0.5
DRIFT_STD_RATIO_HIGH = 2.0
CLIP_NOT_BINDING_ABS_Z_FRACTION = 0.50

@dataclass(frozen=True, slots=True)
class PreprocessAuditConfig:
    batch_size: int = DEFAULT_PREPROCESS_AUDIT_BATCH_SIZE
    validate_dataset_on_open: bool = True
    max_sample_rows_per_split: int = DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS
    extractor_config: ex.LinearFeatureExtractorConfig = ex.LinearFeatureExtractorConfig()
    preprocess_config: pp.LinearPreprocessConfig = pp.LinearPreprocessConfig()
    def __post_init__(self):
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0: raise ValueError
        if not isinstance(self.validate_dataset_on_open, bool): raise ValueError
        if isinstance(self.max_sample_rows_per_split, bool) or not isinstance(self.max_sample_rows_per_split, int) or self.max_sample_rows_per_split < 0: raise ValueError
        if not isinstance(self.extractor_config, ex.LinearFeatureExtractorConfig): raise ValueError
        if not isinstance(self.preprocess_config, pp.LinearPreprocessConfig): raise ValueError

@dataclass(slots=True)
class _StreamingMatrixStats:
    n_rows:int; n_features:int; mean:np.ndarray; m2:np.ndarray
    @classmethod
    def empty(cls,n_features:int): return cls(0,n_features,np.zeros(n_features,np.float64),np.zeros(n_features,np.float64))
    def update(self,X):
        X=np.asarray(X,np.float64)
        if X.ndim!=2 or X.shape[1]!=self.n_features: raise ValueError
        if X.shape[0]==0: return
        if not np.isfinite(X).all(): raise ValueError
        n=X.shape[0]; m=X.mean(0); c=X-m; m2=np.sum(c*c,0); total=self.n_rows+n; d=m-self.mean
        self.mean=self.mean + d*(n/total); self.m2=np.maximum(self.m2+m2+d*d*(self.n_rows*n/total),0.0); self.n_rows=total
    def variance(self): return np.zeros(self.n_features,np.float64) if self.n_rows<=1 else self.m2/(self.n_rows-1)
    def std(self): return np.sqrt(np.maximum(self.variance(),0.0))

@dataclass(frozen=True, slots=True)
class PreprocessFeatureRecord:
    split:str; feature:str; feature_index:int; n_rows:int; n_sample_rows:int; active:bool
    train_mean:float; train_variance:float; train_scale:float
    raw_mean:float; raw_std:float; raw_p01:float; raw_p50:float; raw_p99:float
    z_pre_mean:float; z_pre_std:float; z_pre_p01:float; z_pre_p50:float; z_pre_p99:float; z_pre_abs_p95:float; z_pre_abs_p99:float; z_pre_abs_max:float
    z_post_mean:float; z_post_std:float
    clip_pos_count:int; clip_neg_count:int; clip_total_count:int; clip_pos_rate:float; clip_neg_rate:float; clip_total_rate:float; near_clip_rate:float
    drift_mean_z:float; drift_std_ratio:float; status:str; recommendation:str

@dataclass(frozen=True, slots=True)
class PreprocessSplitSummary:
    split:str; manifest_row_count:int; scanned_rows:int; sampled_rows:int; sample_stride:int|None; n_features:int; active_count:int; inactive_count:int
    features_clip_review_count:int; features_clip_excessive_count:int; features_drift_review_count:int; features_not_binding_count:int
    max_clip_total_rate:float; median_clip_total_rate:float; max_near_clip_rate:float; max_abs_drift_mean_z:float; min_drift_std_ratio:float; max_drift_std_ratio:float

@dataclass(frozen=True, slots=True)
class PreprocessAuditResult:
    schema_version:int; dataset_root:str; dataset_id:str; manifest_hash:str; config:dict[str,object]; preprocess_state:dict[str,object]; splits:dict[str,PreprocessSplitSummary]; feature_records:tuple[PreprocessFeatureRecord,...]; warnings:tuple[str,...]
    def as_dict(self):
        st=self.preprocess_state
        return {"schema_version":self.schema_version,"dataset_root":self.dataset_root,"dataset_id":self.dataset_id,"manifest_hash":self.manifest_hash,"config":self.config,
        "preprocess_state_summary":{"n_rows_fit":st["n_rows_fit"],"n_features":len(st["feature_columns"]),"active_count":int(sum(st["active_mask"])),"inactive_count":int(len(st["active_mask"])-sum(st["active_mask"])),"clip_z":st["config"]["clip_z"],"variance_floor":st["config"]["variance_floor"]},
        "splits":{k:asdict(v) for k,v in self.splits.items()},"warnings":list(self.warnings)}

def _fit_preprocessor_from_train(reader, manifest, config):
    e=ex.IdentityFeatureExtractor(config.extractor_config,manifest=manifest); cols=e.column_projection(manifest); p=pp.LinearPreprocessor(config.preprocess_config)
    for b in reader.iter_split_batches(SplitRole.TRAIN,columns=cols,batch_size=config.batch_size): p.partial_fit(e.transform_table(pa.Table.from_batches([b])).X,feature_columns=cols)
    return p.finalize()

def _audit_split(reader, manifest, role, preprocess_state, config):
    e=ex.IdentityFeatureExtractor(config.extractor_config,manifest=manifest); cols=e.column_projection(manifest); n=len(cols); clip_z=preprocess_state.config.clip_z
    raw=_StreamingMatrixStats.empty(n); pre=_StreamingMatrixStats.empty(n); post=_StreamingMatrixStats.empty(n)
    entries=reader.split_entries(role); mrows=sum(sp.end_row-sp.start_row for sp in entries); stride=None if config.max_sample_rows_per_split==0 or mrows==0 else max(1,math.ceil(mrows/config.max_sample_rows_per_split))
    pos=np.zeros(n,np.int64);neg=np.zeros(n,np.int64);near=np.zeros(n,np.int64); samples_x=[];samples_z=[];scanned=0;row_pos=0
    for b in reader.iter_split_batches(role,columns=cols,batch_size=config.batch_size):
        X=np.asarray(e.transform_table(pa.Table.from_batches([b])).X,np.float64); Z=(X-preprocess_state.mean)/preprocess_state.scale; Z[:,~preprocess_state.active_mask]=0.0; C=np.clip(Z,-clip_z,clip_z)
        raw.update(X); pre.update(Z); post.update(C); pos += np.sum(Z>=clip_z,0).astype(np.int64); neg += np.sum(Z<=-clip_z,0).astype(np.int64); near += np.sum((np.abs(Z)>=NEAR_CLIP_FRACTION*clip_z)&(np.abs(Z)<clip_z),0).astype(np.int64)
        if stride is not None:
            idx=np.arange(X.shape[0])+row_pos; keep=(idx%stride)==0
            if np.any(keep): samples_x.append(X[keep].astype(np.float32)); samples_z.append(Z[keep].astype(np.float32))
        row_pos += X.shape[0]; scanned += X.shape[0]
    sx=np.vstack(samples_x) if samples_x else np.empty((0,n),np.float32); sz=np.vstack(samples_z) if samples_z else np.empty((0,n),np.float32)
    recs=[]; prs=pre.std(); psts=post.std(); rstd=raw.std()
    for i,f in enumerate(cols):
        nrows=scanned; ctot=int(pos[i]+neg[i]); crate=ctot/nrows if nrows else 0.0; dmean=float(pre.mean[i]); dr=float(prs[i]); drift=(abs(dmean)>=DRIFT_MEAN_Z_REVIEW) or (dr<=DRIFT_STD_RATIO_LOW) or (dr>=DRIFT_STD_RATIO_HIGH)
        status,reco=("inactive","review_variance_floor") if not preprocess_state.active_mask[i] else ("ok","keep")
        clip=False
        if crate>=CLIP_EXCESSIVE_RATE: status,reco,clip=("clip_excessive","review_clip_z",True)
        elif crate>=CLIP_REVIEW_RATE: status,reco,clip=("clip_review","review_clip_z",True)
        if drift:
            if status=="inactive": status,reco="inactive_and_drift_review","review_distribution_drift"
            elif clip: reco="review_clip_z_and_drift"
            else: status,reco="drift_review","review_distribution_drift"
        q=lambda a,p: float(np.quantile(a[:,i],p)) if a.shape[0]>0 else float('nan')
        recs.append(PreprocessFeatureRecord(role.value,f,i,nrows,sx.shape[0],bool(preprocess_state.active_mask[i]),float(preprocess_state.mean[i]),float(preprocess_state.variance[i]),float(preprocess_state.scale[i]),float(raw.mean[i]),float(rstd[i]),q(sx,0.01),q(sx,0.5),q(sx,0.99),float(pre.mean[i]),float(prs[i]),q(sz,0.01),q(sz,0.5),q(sz,0.99),float(np.quantile(np.abs(sz[:,i]),0.95)) if sz.shape[0]>0 else float('nan'),float(np.quantile(np.abs(sz[:,i]),0.99)) if sz.shape[0]>0 else float('nan'),float(np.max(np.abs(sz[:,i]))) if sz.shape[0]>0 else float('nan'),float(post.mean[i]),float(psts[i]),int(pos[i]),int(neg[i]),ctot,float(pos[i]/nrows if nrows else 0.0),float(neg[i]/nrows if nrows else 0.0),float(crate),float(near[i]/nrows if nrows else 0.0),dmean,dr,status,reco))
    s=PreprocessSplitSummary(role.value,mrows,scanned,sx.shape[0],stride,n,int(np.sum(preprocess_state.active_mask)),int(np.sum(~preprocess_state.active_mask)),sum(r.status=="clip_review" for r in recs),sum(r.status=="clip_excessive" for r in recs),sum("drift_review" in r.status for r in recs),sum((r.z_pre_abs_p99<CLIP_NOT_BINDING_ABS_Z_FRACTION*clip_z) if np.isfinite(r.z_pre_abs_p99) else False for r in recs),float(max(r.clip_total_rate for r in recs) if recs else 0.0),float(np.median([r.clip_total_rate for r in recs]) if recs else 0.0),float(max(r.near_clip_rate for r in recs) if recs else 0.0),float(max(abs(r.drift_mean_z) for r in recs) if recs else 0.0),float(min(r.drift_std_ratio for r in recs) if recs else 0.0),float(max(r.drift_std_ratio for r in recs) if recs else 0.0))
    return s,recs

def run_preprocess_audit(dataset_root:str,*,config:PreprocessAuditConfig|None=None)->PreprocessAuditResult:
    if not isinstance(dataset_root,str) or not dataset_root.strip(): raise ValueError
    cfg=config or PreprocessAuditConfig(); reader=rd.open_dataset(dataset_root,validate_on_open=cfg.validate_dataset_on_open,batch_size=cfg.batch_size); manifest=reader.manifest; manifest.validate_against_current_code()
    if not reader.split_entries(SplitRole.TRAIN): raise ValueError
    if not reader.split_entries(SplitRole.VAL): raise ValueError
    st=_fit_preprocessor_from_train(reader,manifest,cfg); splits={}; records=[]; warnings=[]
    for role in (SplitRole.TRAIN,SplitRole.VAL,SplitRole.TEST):
        if role==SplitRole.TEST and not reader.split_entries(role): warnings.append("missing_test_split"); continue
        ss,rr=_audit_split(reader,manifest,role,st,cfg); splits[role.value]=ss; records.extend(rr)
        if ss.features_clip_excessive_count>0: warnings.append(f"clip_excessive:{role.value}")
        if ss.features_clip_review_count>0: warnings.append(f"clip_review:{role.value}")
        if ss.features_drift_review_count>0: warnings.append(f"drift_review:{role.value}")
    if int(np.sum(~st.active_mask))>0: warnings.append("inactive_features_present")
    return PreprocessAuditResult(PREPROCESS_AUDIT_SCHEMA_VERSION,dataset_root,manifest.dataset_id,manifest.content_hash(),{"batch_size":cfg.batch_size,"validate_dataset_on_open":cfg.validate_dataset_on_open,"max_sample_rows_per_split":cfg.max_sample_rows_per_split,"extractor_config":{"feature_columns":list(cfg.extractor_config.feature_columns) if cfg.extractor_config.feature_columns else None,"output_dtype":cfg.extractor_config.output_dtype},"preprocess_config":{"variance_floor":cfg.preprocess_config.variance_floor,"clip_z":cfg.preprocess_config.clip_z,"output_dtype":cfg.preprocess_config.output_dtype}},st.as_dict(),splits,tuple(records),tuple(warnings))

def write_preprocess_audit_artifacts(result:PreprocessAuditResult,output_dir:str,*,summary_filename:str=DEFAULT_PREPROCESS_AUDIT_SUMMARY_FILENAME,features_filename:str=DEFAULT_PREPROCESS_AUDIT_FEATURES_FILENAME)->dict[str,str]:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); s=out/summary_filename; c=out/features_filename
    t=s.with_suffix(s.suffix+".tmp")
    t.write_text(json.dumps(result.as_dict(),sort_keys=True,indent=2,allow_nan=True)+"\n",encoding="utf-8")
    t.replace(s)
    fields=list(PreprocessFeatureRecord.__dataclass_fields__.keys()); tc=c.with_suffix(c.suffix+".tmp")
    with tc.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow(asdict(r)) for r in result.feature_records]
    tc.replace(c); return {"summary_json":str(s),"features_csv":str(c)}

__all__=["DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS","DEFAULT_PREPROCESS_AUDIT_SUMMARY_FILENAME","DEFAULT_PREPROCESS_AUDIT_FEATURES_FILENAME","PreprocessAuditConfig","PreprocessFeatureRecord","PreprocessSplitSummary","PreprocessAuditResult","run_preprocess_audit","write_preprocess_audit_artifacts"]
