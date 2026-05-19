import json
import numpy as np
import pytest

from feature_lab import (
    eval_expr,
    evaluate_candidate_array,
    side_specific_keep_mask,
    select_feature_names_and_idx,
    resolve_train_weeks,
    resolve_week_label_count,
)


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


def test_side_specific_keep_mask_respects_min_abs_label_eps():
    y = np.array([0.0, 1e-8, -1e-8, 0.1, -0.2, 0.4, -0.6], dtype=np.float64)

    kept_no_eps = side_specific_keep_mask(y, 0.0, 0.0, min_abs_label_eps=0.0)
    kept_eps = side_specific_keep_mask(y, 0.0, 0.0, min_abs_label_eps=1e-3)

    assert kept_no_eps.sum() == 6
    assert kept_eps.sum() == 4
    assert not kept_eps[1]
    assert not kept_eps[2]


def test_binary_auc_uses_average_ranks_for_ties():
    from feature_lab import _binary_auc_np

    y = np.array([0, 1, 0, 1], dtype=np.int64)
    s = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    assert abs(_binary_auc_np(y, s) - 0.5) < 1e-12


def test_resolve_trim_fractions_preserves_explicit_zero():
    from feature_lab import resolve_trim_fractions

    low, high = resolve_trim_fractions({
        "low_abs_trim_fraction": 0.0,
        "high_abs_trim_fraction": 0.0,
    })

    assert low == 0.0
    assert high == 0.0


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


def test_decile_report_has_10_deciles_per_horizon():
    X, y, names, weeks = _mk()
    _, _, _, _, _, dec = evaluate_candidate_array("sig", X[:, 0], X, y, names, weeks)
    for h in [200, 500, 1000]:
        d = {r["decile"] for r in dec if r["horizon_ms"] == h}
        assert d == set(range(10))
    assert len(dec) == 30


def test_health_report_has_required_distribution_fields():
    X, y, names, weeks = _mk()
    health, *_ = evaluate_candidate_array("sig", X[:, 0], X, y, names, weeks)
    for k in ["p01", "p05", "p50", "p95", "p99", "min", "max", "abs_max", "zero_frac", "near_zero_frac_abs_lt_1e-6"]:
        assert k in health
        assert np.isfinite(health[k])


def test_relative_report_has_required_decision_context():
    X, y, names, weeks = _mk()
    *_, rel, summary, dec = evaluate_candidate_array("sig", X[:, 0], X, y, names, weeks)
    for k in ["high_corr_duplicate", "medium_corr_related", "best_kept_auc_200ms", "best_kept_auc_500ms", "best_kept_auc_1000ms", "best_kept_bal_acc_1000ms", "best_abs_return_spearman", "best_mi_direction", "best_mi_abs_return", "finite_frac", "std", "week_std_cv"]:
        assert k in rel


def test_resolve_train_weeks_current_split_schema():
    meta = {"splits": {"cmssl": {"train": {"weeks": ["w1", "w2"]}}}}
    assert resolve_train_weeks(meta) == ["w1", "w2"]


def test_resolve_train_weeks_legacy_schema():
    meta = {"train_weeks": ["w1"]}
    assert resolve_train_weeks(meta) == ["w1"]


def test_resolve_week_label_count_labels_total():
    assert resolve_week_label_count({"labels_total": 123}) == 123


def test_resolve_week_label_count_chunks():
    assert resolve_week_label_count({"label_chunks": [{"n": 5}, {"n": 7}]}) == 12
