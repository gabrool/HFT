from pathlib import Path
import numpy as np
import pytest
from mmrt.analysis import feature_audit as fa
from mmrt.contracts import SplitRole
from mmrt.features import specs
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr
from mmrt.storage import manifest as mf

def _write_feature_audit_ds(root: Path, train_rows: int = 100, val_rows: int = 100, test_rows: int = 0, correlated: bool = False, low_variance: bool = False, val_shift: float = 0.0) -> None:
    writer = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(root), chunk_rows=32))
    n=train_rows+val_rows+test_rows
    for i in range(n):
        ts=1_000_000+i*1_000
        x0=float(i if i<train_rows else i+val_shift)
        x1=x0*2.0 if correlated else float((i*7)%11)
        x2=1.0 if low_variance else float((i*3)%5)
        feats=[x0,x1,x2]+[float((i+j)%13) for j in range(specs.FEATURE_COUNT-3)]
        writer.append_values(decision_index=i+1,ts_us=ts,local_ts_us=ts,event_seq=i,raw_mid=100.0,label_entry_ts_us=ts,label_values=(1.0,1.0,1.0),feature_values=tuple(feats))
    writer.finalize()
    windows=[sp.SplitWindow(SplitRole.TRAIN,1_000_000,1_000_000+train_rows*1_000),sp.SplitWindow(SplitRole.VAL,1_000_000+train_rows*1_000,1_000_000+(train_rows+val_rows)*1_000)]
    if test_rows: windows.append(sp.SplitWindow(SplitRole.TEST,1_000_000+(train_rows+val_rows)*1_000,1_000_000+n*1_000+1))
    sp.build_and_write_splits(str(root),sp.SplitConfig(windows=tuple(windows),purge_before_us=0,purge_after_us=0,embargo_before_us=0,embargo_after_us=0,min_rows_per_split=1,allow_empty_roles=False,validate_dataset_on_open=True),replace_existing=True)

def test_feature_audit_runs_and_returns_expected_records(tmp_path: Path):
    root=tmp_path/'ds'; _write_feature_audit_ds(root)
    out=fa.run_feature_audit(str(root))
    assert 'train' in out.splits and 'val' in out.splits
    assert len(out.health_records)==specs.FEATURE_COUNT*2
    assert 'feature_records' not in out.as_dict()
    assert 'missing_test_split' in out.warnings

def test_train_only_correlation_detection(tmp_path: Path):
    root=tmp_path/'corr'; _write_feature_audit_ds(root,correlated=True)
    out=fa.run_feature_audit(str(root))
    pair=[p for p in out.correlation_pairs if p.index_a==0 and p.index_b==1][0]
    assert pair.abs_corr >= out.config['high_corr_threshold'] if 'high_corr_threshold' in out.config else pair.abs_corr>0.97
    c0=[c for c in out.cluster_records if c.feature_index==0][0]; c1=[c for c in out.cluster_records if c.feature_index==1][0]
    assert c0.cluster_id>=0 and c0.cluster_id==c1.cluster_id

def test_low_variance_detection(tmp_path: Path):
    root=tmp_path/'lv'; _write_feature_audit_ds(root,low_variance=True)
    out=fa.run_feature_audit(str(root))
    r=[x for x in out.health_records if x.split=='train' and x.feature_index==2][0]
    assert r.status=='low_variance' and r.low_variance
    assert 'low_variance_train' in out.warnings

def test_full_split_stats_not_sample_only(tmp_path: Path):
    root=tmp_path/'sample'; _write_feature_audit_ds(root)
    out=fa.run_feature_audit(str(root),config=fa.FeatureAuditConfig(max_sample_rows_per_split=1))
    r=[x for x in out.health_records if x.split=='train' and x.feature_index==0][0]
    assert np.isclose(r.raw_mean,np.mean(np.arange(100)))

def test_feature_subset(tmp_path: Path):
    root=tmp_path/'subset'; _write_feature_audit_ds(root)
    cols=mf.feature_columns()[:2]
    out=fa.run_feature_audit(str(root),config=fa.FeatureAuditConfig(feature_columns=cols))
    assert {r.feature for r in out.health_records}==set(cols)

def test_artifact_filename_validation(tmp_path: Path):
    root=tmp_path/'art'; _write_feature_audit_ds(root)
    out=fa.run_feature_audit(str(root))
    with pytest.raises(ValueError): fa.write_feature_audit_artifacts(out,str(tmp_path),summary_filename='x.csv')
