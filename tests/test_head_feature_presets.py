import inspect

import pytest

from mmrt.linear import head_feature_presets as hp
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm


def test_available_presets_are_stable():
    assert hp.ALL_FEATURES_PRESET == "all"
    assert hp.CORR_PRUNED152_HEAD_SUBSET_V1 == "corr_pruned152_head_subset_v1"
    assert hp.CORR_PRUNED152_HEAD_SUBSET_V2 == "corr_pruned152_head_subset_v2"
    assert hp.CORR_PRUNED152_HEAD_SUBSET_V3 == "corr_pruned152_head_subset_v3"
    assert hp.CORR_PRUNED152_HEAD_SUBSET_V4 == "corr_pruned152_head_subset_v4"
    assert hp.CORR_PRUNED152_HEAD_SUBSET_V5 == "corr_pruned152_head_subset_v5"
    assert hp.AVAILABLE_HEAD_FEATURE_PRESETS == (
        "all",
        "corr_pruned152_head_subset_v1",
        "corr_pruned152_head_subset_v2",
        "corr_pruned152_head_subset_v3",
        "corr_pruned152_head_subset_v4",
        "corr_pruned152_head_subset_v5",
    )


def test_preset_module_has_no_runtime_data_dependencies():
    src = inspect.getsource(hp)
    forbidden = [
        "import pan" + "das",
        "from pan" + "das",
        "import po" + "lars",
        "from po" + "lars",
        "import sk" + "learn",
        "from sk" + "learn",
        "import to" + "rch",
        "from to" + "rch",
        "pyarrow",
        "open_dataset",
        "read_csv",
        "read_parquet",
        "train_linear_model",
        "fit_preprocessors",
    ]
    for token in forbidden:
        assert token not in src


def test_all_preset_returns_default_head_feature_config():
    cfg = hp.head_feature_config_for_preset("all")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert cfg.feature_columns_by_head is None


def test_corr_pruned152_preset_counts_and_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v1")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert set(cfg.feature_columns_by_head) == set(lm.MODEL_HEADS)
    assert len(cfg.feature_columns_by_head[lm.DIRECTION_HEAD]) == 40
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 40
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]) == 30
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 40
    assert hp.preset_feature_counts("corr_pruned152_head_subset_v1") == {
        lm.DIRECTION_HEAD: 40,
        lm.NO_MOVE_HEAD: 40,
        lm.MAGNITUDE_UP_HEAD: 30,
        lm.MAGNITUDE_DOWN_HEAD: 40,
    }


def test_corr_pruned152_v2_preset_counts_and_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v2")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert set(cfg.feature_columns_by_head) == set(lm.MODEL_HEADS)
    assert len(cfg.feature_columns_by_head[lm.DIRECTION_HEAD]) == 34
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 39
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]) == 19
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 9


def test_corr_pruned152_v3_preset_counts_and_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v3")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert set(cfg.feature_columns_by_head) == set(lm.MODEL_HEADS)
    assert len(cfg.feature_columns_by_head[lm.DIRECTION_HEAD]) == 25
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 38
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]) == 15
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 6


def test_corr_pruned152_v4_preset_counts_and_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v4")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert set(cfg.feature_columns_by_head) == set(lm.MODEL_HEADS)
    assert len(cfg.feature_columns_by_head[lm.DIRECTION_HEAD]) == 23
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 38
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]) == 14
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 6


def test_corr_pruned152_v5_preset_counts_and_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v5")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert set(cfg.feature_columns_by_head) == set(lm.MODEL_HEADS)
    assert len(cfg.feature_columns_by_head[lm.DIRECTION_HEAD]) == 20
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 35
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]) == 13
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 6


@pytest.mark.parametrize("preset", [
    "corr_pruned152_head_subset_v1",
    "corr_pruned152_head_subset_v2",
    "corr_pruned152_head_subset_v3",
    "corr_pruned152_head_subset_v4",
    "corr_pruned152_head_subset_v5",
])
def test_corr_pruned152_presets_have_no_duplicate_columns(preset):
    cfg = hp.head_feature_config_for_preset(preset)
    for cols in cfg.feature_columns_by_head.values():
        assert len(cols) == len(set(cols))


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        hp.head_feature_config_for_preset("__missing__")
    with pytest.raises(ValueError):
        hp.preset_feature_counts("__missing__")


def test_corr_pruned152_anchor_features_present():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v1")
    assert "x_depth_imbalance_within_1bps" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_trade_imbalance_notional_500000us" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_log_events_1000000us" in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_spread_z_500000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_time_since_mid_change_us" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v2_anchor_features_present():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v2")
    assert "x_depth_imbalance_within_1bps" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_trade_imbalance_notional_500000us" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_log_events_1000000us" in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_trade_count_per_second_1000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_bid_depth_notional_5bps" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_harmful_features_excluded_from_specific_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v1")
    assert "x_ask_l1_notional_usd" not in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_obi_l1" not in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_spread_z_1000000us" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_cvd_change_usd_1000000us" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v2_excludes_bad_features():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v2")
    assert "x_ask_l1_add_rate_over_depth_200000us" not in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_max_abs_return_bps_500000us" not in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_spread_state_transition_rate_3000000us" not in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_asz1" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_obi_l1" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_bid_l1_notional_usd" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v3_anchor_features_present():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v3")
    assert "x_depth_imbalance_within_1bps" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_trade_imbalance_notional_500000us" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_log_events_1000000us" in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_trade_count_per_second_1000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_bid_depth_notional_5bps" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_max_trade_silence_gap_3000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v3_excludes_negative_and_negligible_features():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v3")

    direction = cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_ask_l1_add_rate_over_depth_1000000us" not in direction
    assert "x_micro_l10_minus_mid_bps" not in direction
    assert "x_asz1" not in direction
    assert "x_bid_l1_depletion_200000us" not in direction

    no_move = cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_absorption_ask_1000000us" not in no_move

    mag_up = cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_bid_l1_depletion_500000us" not in mag_up
    assert "x_bid_l1_add_rate_over_depth_200000us" not in mag_up
    assert "x_total_depth_notional_5bps" not in mag_up
    assert "x_ofi_l3_over_depth_5bps" not in mag_up

    mag_down = cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_mid_range_bps_500000us" not in mag_down
    assert "x_regime_volume_ewma_500000us" not in mag_down
    assert "x_vamp_l10_minus_mid_bps" not in mag_down


def test_corr_pruned152_v4_anchor_features_present():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v4")
    assert "x_depth_imbalance_within_1bps" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_trade_imbalance_notional_500000us" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_log_events_1000000us" in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_trade_count_per_second_1000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_bid_depth_notional_5bps" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_max_trade_silence_gap_3000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v4_excludes_final_negligible_features():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v4")

    direction = cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_bid_depth_notional_5bps" not in direction
    assert "x_bid_depth_centroid_bps_25bps" not in direction

    mag_up = cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_ofi_l1_pressure_over_depth_5bps_200000us" not in mag_up

    # These heads are intentionally unchanged from v3.
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 38
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 6


def test_corr_pruned152_v4_preserves_no_move_and_magnitude_down_from_v3():
    v3 = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v3")
    v4 = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v4")
    assert v4.feature_columns_by_head[lm.NO_MOVE_HEAD] == v3.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert v4.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD] == v3.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v5_anchor_features_present():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v5")
    assert "x_depth_imbalance_within_1bps" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_trade_imbalance_notional_500000us" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_log_events_1000000us" in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_trade_count_per_second_1000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_bid_depth_notional_5bps" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_max_trade_silence_gap_3000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_corr_pruned152_v5_excludes_remaining_negligible_features():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v5")

    direction = cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_down_up_vol_imbalance_3000000us" not in direction
    assert "x_ask_l1_depletion_over_depth_500000us" not in direction
    assert "x_last_trade_side_sign" not in direction

    no_move = cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_ask_l1_depletion_1000000us" not in no_move
    assert "x_top5_trade_notional_sum_usd_200000us" not in no_move
    assert "x_regime_volume_ewma_500000us" not in no_move

    mag_up = cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_spread_z_500000us" not in mag_up

    # Magnitude-down is intentionally unchanged from v4.
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 6


def test_corr_pruned152_v5_preserves_magnitude_down_from_v4():
    v4 = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v4")
    v5 = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v5")
    assert v5.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD] == v4.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
