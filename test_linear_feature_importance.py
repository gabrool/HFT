import numpy as np
import pandas as pd
import pytest

import linear_feature_importance as lfi


def test_raw_linear_feature_name_mapping():
    rows = lfi._build_raw_linear_extracted_names(["a", "b"], ["last", "delta_lag_1", "mean_w_5"])
    assert [r["extracted_feature_name"] for r in rows] == ["a:last", "b:last", "a:delta_lag_1", "b:delta_lag_1", "a:mean_w_5", "b:mean_w_5"]


def _coefs(vals):
    return [np.array(v, dtype=float) for v in vals]


def test_flat_mapping_with_kept_indices():
    rows = lfi._build_raw_linear_extracted_names(["a", "b"], ["last", "delta_lag_1", "mean_w_5"])
    flat = lfi.build_flat_importance_df(extracted_rows=rows, kept_indices=np.array([0, 2, 5]), dir_coefs=_coefs([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), mag_up_coefs=_coefs([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), mag_down_coefs=_coefs([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert flat["extracted_feature_name"].tolist() == ["a:last", "a:delta_lag_1", "b:mean_w_5"]


def test_aggregation_helpers_values():
    rows = lfi._build_raw_linear_extracted_names(["a", "b"], ["last", "delta_lag_1", "mean_w_5"])
    flat = lfi.build_flat_importance_df(extracted_rows=rows, kept_indices=np.array([0, 2, 5]), dir_coefs=_coefs([[1, 0, 0], [1, 0, 0], [1, 0, 3]]), mag_up_coefs=_coefs([[0, 2, 0], [0, 0, 0], [0, 0, 0]]), mag_down_coefs=_coefs([[0, 0, 0], [0, 0, 4], [0, 0, 0]]))
    base = lfi.aggregate_importance_by_base(flat, ["a", "b"], ["last", "delta_lag_1", "mean_w_5"])
    block = lfi.aggregate_importance_by_block(flat)
    assert np.isclose(base["all_importance_l2_share"].sum(), 1.0)
    a = base.set_index("base_feature_name").loc["a"]
    assert np.isclose(a["dir_importance_1000ms_l2"], 1.0)
    assert np.isclose(a["mag_importance_1000ms_l2"], 4.0)
    assert set(block.columns) >= {"dir_importance_l2", "mag_importance_l2", "all_importance_l2"}


def test_low_candidate_helper_flags():
    base_df = pd.DataFrame([
        {"base_feature_name": "x", "all_importance_l2_share": 1e-4, "dir_importance_l2_share": 1e-4, "mag_importance_l2_share": 1e-4},
        {"base_feature_name": "y", "all_importance_l2_share": 1e-2, "dir_importance_l2_share": 1e-4, "mag_importance_l2_share": 1e-4},
    ])
    flat_df = pd.DataFrame([{"base_feature_name": "x", "all_abs_coef_max": 0.0}, {"base_feature_name": "y", "all_abs_coef_max": 1.0}])
    out = lfi.add_low_importance_flags(base_df, low_share=5e-4, low_dir_share=5e-4, low_mag_share=5e-4, coef_eps=1e-10, flat_df=flat_df)
    assert out["low_importance_candidate"].tolist() == [True, False]
    assert out["zero_or_near_zero_all_heads"].tolist() == [True, False]


def test_select_ablation_groups_and_unknown_token():
    base = pd.DataFrame([{"base_feature_name": "a", "all_importance_l2": 3, "dir_importance_l2": 2, "mag_importance_l2": 1}, {"base_feature_name": "b", "all_importance_l2": 2, "dir_importance_l2": 3, "mag_importance_l2": 2}])
    low = pd.DataFrame([{"base_feature_name": "c", "all_importance_l2": 0.1}])
    got = lfi.select_ablation_groups(base_df=base, low_df=low, top_n=1, low_n=1, groups_spec="low_importance,top_all", all_base=False)
    assert got == ["c", "a"]
    with pytest.raises(ValueError):
        lfi.select_ablation_groups(base_df=base, low_df=low, top_n=1, low_n=1, groups_spec="bad", all_base=False)


def test_group_column_selection():
    groups = lfi.get_group_columns([{"model_coef_index": 0, "base_feature_name": "a"}, {"model_coef_index": 1, "base_feature_name": "b"}, {"model_coef_index": 2, "base_feature_name": "a"}], "base_feature_name")
    assert groups["a"] == [0, 2]
    assert groups["b"] == [1]


def test_ablation_metric_semantics():
    y = np.array([[1, 1, 1], [-2, -2, -2], [0, 0, 0], [3, 3, 3], [-4, -4, -4]], dtype=float)
    pred = {
        "dir_logits": np.array([[1, 1, 1], [-1, -1, -1], [0.2, 0.2, 0.2], [2, 2, 2], [-2, -2, -2]], dtype=float),
        "mag_up_bps": np.array([[1, 1, 1], [2, 2, 2], [9, 9, 9], [3, 3, 3], [4, 4, 4]], dtype=float),
        "mag_down_bps": np.array([[1, 1, 1], [2, 2, 2], [9, 9, 9], [3, 3, 3], [4, 4, 4]], dtype=float),
        "mag_up_log": np.log1p(np.array([[1, 1, 1], [2, 2, 2], [9, 9, 9], [3, 3, 3], [4, 4, 4]], dtype=float)),
        "mag_down_log": np.log1p(np.array([[1, 1, 1], [2, 2, 2], [9, 9, 9], [3, 3, 3], [4, 4, 4]], dtype=float)),
    }
    m = lfi.compute_ablation_metrics(y=y, pred=pred, mag_up_scale_bps=np.array([1, 1, 1], dtype=float), mag_down_scale_bps=np.array([1, 1, 1], dtype=float))
    assert "dir_auc_kept_1000ms" in m
    assert "mean_side_log_huber_cond_1000ms" in m
    assert m["edge_spearman_kept_1000ms"] != m["edge_spearman_all_1000ms"]
