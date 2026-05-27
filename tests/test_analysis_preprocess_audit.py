import numpy as np
from pathlib import Path
import pytest
from mmrt.analysis import preprocess_audit as pa
from mmrt.contracts import SplitRole
from mmrt.storage import writer as wr, splits as sp
from mmrt.features import specs

def _write_ds(root:Path, train_rows=20, val_rows=20, test_rows=0, shift=0.0, clip=False, constant=False):
    w=wr.DecisionRowWriter(wr.WriterConfig(dataset_id='d',created_at_utc='2026',dataset_root=str(root),chunk_rows=16))
    n=train_rows+val_rows+test_rows
    for i in range(n):
        ts=1_000_000+i*1000; x0=(0.1*i if i<train_rows else 0.1*i+shift)
        if clip and i>=train_rows: x0=1000.0
        feats=[x0]+[float((i+j)%3) for j in range(specs.FEATURE_COUNT-1)]
        if constant: feats[1]=1.0
        w.append_values(decision_index=i+1,ts_us=ts,local_ts_us=ts,event_seq=i,raw_mid=100.0,label_entry_ts_us=ts,label_values=(1.0,1.0,1.0),feature_values=tuple(feats))
    w.finalize()
    windows=[sp.SplitWindow(SplitRole.TRAIN,1_000_000,1_000_000+train_rows*1000),sp.SplitWindow(SplitRole.VAL,1_000_000+train_rows*1000,1_000_000+(train_rows+val_rows)*1000)]
    if test_rows: windows.append(sp.SplitWindow(SplitRole.TEST,1_000_000+(train_rows+val_rows)*1000,1_000_000+n*1000+1))
    sp.build_and_write_splits(str(root),sp.SplitConfig(windows=tuple(windows),purge_before_us=0,purge_after_us=0,embargo_before_us=0,embargo_after_us=0,min_rows_per_split=1,allow_empty_roles=False,validate_dataset_on_open=True),replace_existing=True)

def test_train_only_fit_no_leakage(tmp_path:Path):
    r=tmp_path/'a'; _write_ds(r,shift=100.0)
    out=pa.run_preprocess_audit(str(r))
    assert out.as_dict()['preprocess_state_summary']['n_rows_fit']==20
    t0=[x for x in out.feature_records if x.split=='train' and x.feature_index==0][0]
    v0=[x for x in out.feature_records if x.split=='val' and x.feature_index==0][0]
    assert abs(t0.drift_mean_z)<1e-6
    assert abs(v0.drift_mean_z)>1.0

def test_clip_detection(tmp_path:Path):
    r=tmp_path/'b'; _write_ds(r,clip=True)
    out=pa.run_preprocess_audit(str(r))
    v0=[x for x in out.feature_records if x.split=='val' and x.feature_index==0][0]
    assert v0.clip_total_rate>0
    assert v0.status in {'clip_review','clip_excessive','drift_review'}

def test_inactive_detection(tmp_path:Path):
    r=tmp_path/'c'; _write_ds(r,constant=True)
    out=pa.run_preprocess_audit(str(r))
    rec=[x for x in out.feature_records if x.split=='train' and x.feature_index==1][0]
    assert rec.active is False
    assert rec.recommendation in {'review_variance_floor','review_distribution_drift'}

def test_sampling_deterministic(tmp_path:Path):
    r=tmp_path/'d'; _write_ds(r,train_rows=100,val_rows=100)
    cfg=pa.PreprocessAuditConfig(max_sample_rows_per_split=10)
    a=pa.run_preprocess_audit(str(r),config=cfg); b=pa.run_preprocess_audit(str(r),config=cfg)
    assert a.splits['train'].sample_stride==b.splits['train'].sample_stride
    assert a.as_dict()==b.as_dict()

def test_counts_full_split_not_sampled(tmp_path:Path):
    r=tmp_path/'e'; _write_ds(r,train_rows=100,val_rows=100,clip=True)
    cfg=pa.PreprocessAuditConfig(max_sample_rows_per_split=1)
    out=pa.run_preprocess_audit(str(r),config=cfg)
    v0=[x for x in out.feature_records if x.split=='val' and x.feature_index==0][0]
    assert v0.clip_total_count==100
