import numpy as np
import pandas as pd
import linear_feature_importance as lfi

def _coefs(v): return [np.array(x,float) for x in v]

def test_move_columns_and_aggregations():
    rows=lfi._build_raw_linear_extracted_names(['a','b'],['blk'])
    flat=lfi.build_flat_importance_df(extracted_rows=rows,kept_indices=np.array([0,1]),dir_coefs=_coefs([[1,0],[1,0],[1,0]]),mag_up_coefs=_coefs([[0,0],[0,0],[0,0]]),mag_down_coefs=_coefs([[0,0],[0,0],[0,0]]),move_coefs=_coefs([[0,2],[0,2],[0,2]]))
    assert 'move_abs_coef_1000ms' in flat.columns
    base=lfi.aggregate_importance_by_base(flat,['a','b'],['blk'])
    block=lfi.aggregate_importance_by_block(flat)
    assert 'move_importance_l2' in base.columns and 'move_importance_l2_share' in base.columns
    assert 'move_importance_l2' in block.columns and 'move_importance_l2_share' in block.columns

def test_low_flags_and_top_move():
    base=pd.DataFrame([{'base_feature_name':'a','all_importance_l2_share':1e-4,'dir_importance_l2_share':1e-4,'mag_importance_l2_share':1e-4,'move_importance_l2_share':1e-4,'move_importance_l2':2.0,'all_importance_l2':2.0,'dir_importance_l2':0.1,'mag_importance_l2':0.1},{'base_feature_name':'b','all_importance_l2_share':1e-4,'dir_importance_l2_share':1e-4,'mag_importance_l2_share':1e-4,'move_importance_l2_share':1e-2,'move_importance_l2':0.1,'all_importance_l2':1.0,'dir_importance_l2':1.0,'mag_importance_l2':1.0}])
    flat=pd.DataFrame([{'base_feature_name':'a','all_abs_coef_max':0.0},{'base_feature_name':'b','all_abs_coef_max':1.0}])
    out=lfi.add_low_importance_flags(base,low_share=5e-4,low_dir_share=5e-4,low_mag_share=5e-4,low_move_share=5e-4,coef_eps=1e-10,flat_df=flat)
    assert out['low_importance_candidate'].tolist()==[True,False]
    got=lfi.select_ablation_groups(base_df=base,low_df=base,top_n=1,low_n=1,groups_spec='top_move',all_base=False)
    assert got==['a']

def test_compute_ablation_metrics_move_and_gated():
    y=np.array([[1,1,1],[-1,-1,-1],[0,0,0],[2,2,2]],float)
    pred={'dir_logits':np.array([[1,1,1],[-1,-1,-1],[0,0,0],[2,2,2]],float),'mag_up_bps':np.ones((4,3)),'mag_down_bps':np.ones((4,3)),'mag_up_log':np.zeros((4,3)),'mag_down_log':np.zeros((4,3)),'p_move':np.array([[0.9,0.9,0.9],[0.9,0.9,0.9],[0.1,0.1,0.1],[0.9,0.9,0.9]])}
    stats={'pos_lo_raw_bps':np.array([0.5,0.5,0.5]),'neg_lo_abs_bps':np.array([0.5,0.5,0.5])}
    m=lfi.compute_ablation_metrics(y=y,pred=pred,mag_up_scale_bps=np.ones(3),mag_down_scale_bps=np.ones(3),signed_raw_stats=stats)
    for k in [
        "move_auc_1000ms",
        "move_bal_acc_1000ms",
        "move_bce_1000ms",
        "move_pos_frac_true_1000ms",
        "p_move_mean_zero_rows_1000ms",
        "p_move_mean_nonmove_rows_1000ms",
        "p_move_mean_move_rows_1000ms",
        "cond_edge_spearman_all_1000ms",
        "cond_edge_spearman_kept_1000ms",
        "gated_edge_spearman_all_1000ms",
        "gated_edge_spearman_kept_1000ms",
        "edge_spearman_all_1000ms",
        "edge_spearman_kept_1000ms",
    ]:
        assert k in m

def test_resolve_extractor_name_accepts_extractor_config_extractor():
    st4 = {}
    st2 = {"extractor_config": {"extractor": "raw_linear"}}
    assert lfi._resolve_extractor_name(st4, st2) == "raw_linear"

def test_resolve_extractor_name_accepts_extractor_config_name():
    st4 = {}
    st2 = {"extractor_config": {"name": "raw_linear"}}
    assert lfi._resolve_extractor_name(st4, st2) == "raw_linear"
