import json
import numpy as np
import pytest

from feature_lab import eval_expr, evaluate_candidate_array


def _mk():
    rng=np.random.default_rng(0)
    X=rng.normal(size=(800,4)).astype(np.float32)
    y=np.zeros((800,3),dtype=np.float32)
    for h in range(3): y[:,h]=0.4*X[:,0]+0.1*rng.normal(size=800)
    names=["f0","f1","f2","f3"]
    weeks=["w1"]*400+["w2"]*400
    return X,y,names,weeks


def test_expr_candidate_shape_and_finite():
    X,_,names,_=_mk()
    c=eval_expr("f0 - f1",X,names)
    assert c.shape==(X.shape[0],)
    assert np.isfinite(c).all()


def test_expr_rejects_unknown_feature():
    X,_,names,_=_mk()
    with pytest.raises(ValueError):
        eval_expr("missing + f0",X,names)


def test_expr_rejects_unsafe_code():
    X,_,names,_=_mk()
    with pytest.raises(ValueError):
        eval_expr("__import__('os').system('echo x')",X,names)


def test_corr_detects_duplicate_candidate():
    X,y,names,weeks=_mk()
    h,_,corr,rel,_,_=evaluate_candidate_array("dup",X[:,0],X,y,names,weeks)
    assert h["finite_frac"]==1.0
    assert corr[0]["max_abs_corr"]>0.999
    assert rel["max_corr_with_existing"]>0.999


def test_target_metrics_detect_directional_signal():
    X,y,names,weeks=_mk()
    _,target,_,_,_,_=evaluate_candidate_array("sig",X[:,0],X,y,names,weeks)
    kept=[r for r in target if r["horizon_ms"]==1000 and r["mask_type"]=="kept"][0]
    assert kept["single_feature_auc_direction"]>0.6


def test_decile_report_has_10_deciles_per_horizon():
    X,y,names,weeks=_mk()
    _,_,_,_,_,dec=evaluate_candidate_array("sig",X[:,0],X,y,names,weeks)
    for h in [200,500,1000]:
        d={r["decile"] for r in dec if r["horizon_ms"]==h}
        assert d==set(range(10))


def test_summary_is_compact_no_large_arrays():
    X,y,names,weeks=_mk()
    *_,summary,_=evaluate_candidate_array("sig",X[:,0],X,y,names,weeks)
    txt=json.dumps(summary)
    assert "candidate_values" not in txt
    assert len(txt)<20000
