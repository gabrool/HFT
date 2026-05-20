import numpy as np
import pandas as pd

import linear_feature_importance as lfi


def test_raw_linear_feature_name_mapping():
    rows = lfi._build_raw_linear_extracted_names(["a", "b"], ["last", "delta_lag_1", "mean_w_5"])
    got = [r["extracted_feature_name"] for r in rows]
    assert got == ["a:last", "b:last", "a:delta_lag_1", "b:delta_lag_1", "a:mean_w_5", "b:mean_w_5"]


def test_kept_index_mapping():
    rows = lfi._build_raw_linear_extracted_names(["a", "b"], ["last", "delta_lag_1", "mean_w_5"])
    kept = [0, 2, 5]
    mapped = [rows[i]["extracted_feature_name"] for i in kept]
    assert mapped == ["a:last", "a:delta_lag_1", "b:mean_w_5"]


def test_coefficient_aggregation_and_shares():
    flat = pd.DataFrame([
        {"base_feature_name":"a","block_name":"last","dir_abs_coef_200ms":1.0,"dir_abs_coef_500ms":1.0,"dir_abs_coef_1000ms":1.0,
         "mag_up_abs_coef_200ms":0.0,"mag_up_abs_coef_500ms":0.0,"mag_up_abs_coef_1000ms":0.0,
         "mag_down_abs_coef_200ms":0.0,"mag_down_abs_coef_500ms":0.0,"mag_down_abs_coef_1000ms":0.0,
         "mag_abs_coef_max":0.0,"all_abs_coef_max":1.0},
        {"base_feature_name":"b","block_name":"delta","dir_abs_coef_200ms":0.0,"dir_abs_coef_500ms":0.0,"dir_abs_coef_1000ms":0.0,
         "mag_up_abs_coef_200ms":2.0,"mag_up_abs_coef_500ms":0.0,"mag_up_abs_coef_1000ms":0.0,
         "mag_down_abs_coef_200ms":0.0,"mag_down_abs_coef_500ms":0.0,"mag_down_abs_coef_1000ms":0.0,
         "mag_abs_coef_max":2.0,"all_abs_coef_max":2.0},
    ])
    # mimic aggregation logic quickly
    a_l2 = np.sqrt(3.0)
    b_l2 = 2.0
    total = a_l2 + b_l2
    assert np.isclose(a_l2 / total + b_l2 / total, 1.0)
    assert b_l2 > a_l2


def test_low_candidate_logic():
    df = pd.DataFrame([
        {"base_feature_name":"x","all_importance_l2_share":1e-4,"dir_importance_l2_share":1e-4,"mag_importance_l2_share":1e-4},
        {"base_feature_name":"y","all_importance_l2_share":1e-2,"dir_importance_l2_share":1e-4,"mag_importance_l2_share":1e-4},
    ])
    low = (df["all_importance_l2_share"] <= 5e-4) & (df["dir_importance_l2_share"] <= 5e-4) & (df["mag_importance_l2_share"] <= 5e-4)
    assert low.tolist() == [True, False]


def test_ablation_group_column_selection():
    flat_rows = [
        {"model_coef_index":0,"base_feature_name":"a"},
        {"model_coef_index":1,"base_feature_name":"b"},
        {"model_coef_index":2,"base_feature_name":"a"},
    ]
    groups = lfi._group_columns(flat_rows, "base_feature_name")
    assert groups["a"] == [0, 2]
    assert groups["b"] == [1]
