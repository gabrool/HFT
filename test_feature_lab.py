import json, os
import numpy as np
import pytest

from feature_lab import eval_expr, evaluate_candidate_array, side_specific_keep_mask, select_feature_names_and_idx


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


def test_expr_ast_safety():
    X,_,names,_=_mk()
    assert np.isfinite(eval_expr("np.log1p(np.abs(f0))",X,names)).all()
    for expr in ["__import__('os').system('echo x')","np.asarray(f0)","f0.__class__","(lambda x: x)(f0)","f0[0]"]:
        with pytest.raises(ValueError):
            eval_expr(expr,X,names)


def test_mask_semantics():
    y=np.array([0.0,1e-8,0.1,-0.2,0.4,-0.6],dtype=np.float64)
    cand=np.array([0,0,1,1,1,1],dtype=np.float64)
    finite=np.isfinite(y)&np.isfinite(cand)
    nonzero=finite&(np.abs(y)>1e-3)
    assert nonzero.tolist()==[False,False,True,True,True,True]
    kept=side_specific_keep_mask(y,0.25,0.25)
    assert kept.sum()< (np.abs(y)>0).sum()


def test_mi_only_on_kept_and_corr_methods(monkeypatch):
    monkeypatch.setenv("BYBIT_FEATURE_LAB_CORR_METHODS","pearson")
    X,y,names,weeks=_mk()
    _,target,corr,_,_,_=evaluate_candidate_array("sig",X[:,0],X,y,names,weeks)
    non_kept=[r for r in target if r["mask_type"]!="kept"]
    assert all(np.isnan(r["mi_direction"]) for r in non_kept)
    assert np.isnan(corr[0]["spearman"])
    assert np.isfinite(corr[0]["pearson"])


def test_aux_selection():
    meta={"feature_names":["a","b"],"feature_dim_core":2,"feature_dim_total":4,"aux_names":["aux_dt","aux_event"]}
    names0,idx0=select_feature_names_and_idx(meta,False)
    names1,idx1=select_feature_names_and_idx(meta,True)
    assert names0==["a","b"] and idx0.tolist()==[0,1]
    assert names1==["a","b","aux_dt","aux_event"] and idx1.tolist()==[0,1,2,3]


def test_corr_detects_duplicate_candidate_and_existing_scores():
    X,y,names,weeks=_mk()
    h,_,corr,rel,_,_=evaluate_candidate_array("dup",X[:,0],X,y,names,weeks)
    assert h["finite_frac"]==1.0
    assert corr[0]["max_abs_corr"]>0.999
    assert "existing_best_kept_auc" in corr[0]
    assert "existing_best_abs_return_spearman" in corr[0]
    assert "existing_target_score" in corr[0]
    assert "most_correlated_existing_target_score" in rel


def test_summary_is_compact_no_large_arrays():
    X,y,names,weeks=_mk()
    *_,summary,_=evaluate_candidate_array("sig",X[:,0],X,y,names,weeks)
    txt=json.dumps(summary)
    assert "candidate_values" not in txt
    assert len(txt)<20000
