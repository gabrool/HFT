import numpy as np
from feature_lab import build_move_target_from_sample,evaluate_candidate_array

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
