import numpy as np
from feature_lab import build_move_target_from_sample,evaluate_candidate_array

def assert_rows_match_fields(rows, fields):
    allowed = set(fields)
    for row in rows:
        extra = set(row) - allowed
        assert not extra, f"extra fields: {extra}"

def test_move_target_and_reports():
    X=np.array([[0.0,0],[0.1,0],[0.9,0],[-0.9,0],[0.0,0]],dtype=np.float32)
    y=np.array([[0,0,0],[0.01,0.01,0.01],[1,1,1],[-1,-1,-1],[0,0,0]],dtype=np.float32)
    weeks=['w']*5
    mt=build_move_target_from_sample(y,0.2,0.0)
    assert mt.shape==y.shape and mt[2,2]==1 and mt[0,2]==0
    health,target,corr,rel,summary,dec=evaluate_candidate_array('c',X[:,0],X,y,['f0','f1'],weeks)
    assert any('single_feature_auc_move' in r for r in target)
    assert 'best_move' in summary
    assert 'best_move_auc_1000ms' in rel and 'best_mi_move' in rel
    assert {'move_frac','nonmove_frac','zero_frac'}.issubset(dec[0].keys())
    assert rel['reason']!='weak_direction_and_magnitude'

def test_feature_lab_csv_writing_accepts_move_fields(tmp_path):
    import feature_lab

    X = np.array([[0.0,0.0],[0.1,0.0],[0.9,0.0],[-0.9,0.0],[0.0,0.0],[0.8,0.0],[-0.8,0.0],[0.2,0.0]], dtype=np.float32)
    y = np.array([[0.0,0.0,0.0],[0.01,0.01,0.01],[1.0,1.0,1.0],[-1.0,-1.0,-1.0],[0.0,0.0,0.0],[0.8,0.8,0.8],[-0.8,-0.8,-0.8],[0.02,0.02,0.02]], dtype=np.float32)
    weeks = ["w"] * len(X)

    health, target, corr, rel, summary, dec = feature_lab.evaluate_candidate_array("candidate", X[:, 0], X, y, ["f0", "f1"], weeks)
    assert_rows_match_fields([health], feature_lab.HEALTH_FIELDS)
    assert_rows_match_fields(target, feature_lab.TARGET_FIELDS)
    assert_rows_match_fields(corr, feature_lab.CORR_FIELDS)
    assert_rows_match_fields([rel], feature_lab.RELATIVE_FIELDS)
    assert_rows_match_fields(dec, feature_lab.DECILE_FIELDS)

    feature_lab.wcsv(tmp_path / "candidate_health.csv", [health], feature_lab.HEALTH_FIELDS)
    feature_lab.wcsv(tmp_path / "candidate_target_metrics.csv", target, feature_lab.TARGET_FIELDS)
    feature_lab.wcsv(tmp_path / "candidate_corr_top_pairs.csv", corr, feature_lab.CORR_FIELDS)
    feature_lab.wcsv(tmp_path / "candidate_relative_report.csv", [rel], feature_lab.RELATIVE_FIELDS)
    feature_lab.wcsv(tmp_path / "candidate_decile_report.csv", dec, feature_lab.DECILE_FIELDS)
    assert (tmp_path / "candidate_corr_top_pairs.csv").exists()

def test_existing_feature_move_score_does_not_require_kept_rows(monkeypatch):
    import feature_lab

    X = np.array([[0.1],[0.2],[0.9],[1.0],[0.0],[0.1],[0.8],[0.9]], dtype=np.float32)
    y = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[1.0,1.0,1.0],[1.1,1.1,1.1],[0.0,0.0,0.0],[0.0,0.0,0.0],[1.2,1.2,1.2],[1.3,1.3,1.3]], dtype=np.float32)

    monkeypatch.setattr(feature_lab, "side_specific_keep_mask", lambda *args, **kwargs: np.zeros(y.shape[0], dtype=bool))
    scores = feature_lab.compute_existing_feature_target_scores(X, y, ["f0"], low_abs_trim_fraction=0.2, high_abs_trim_fraction=0.2)
    assert "f0" in scores
    assert np.isfinite(scores["f0"]["existing_best_move_auc"])
    assert scores["f0"]["existing_best_move_auc"] > 0.5
    assert np.isnan(scores["f0"]["existing_best_kept_auc"])
