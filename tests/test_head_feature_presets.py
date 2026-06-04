import inspect

import pytest

from mmrt.linear import head_feature_presets as hp
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm


def test_available_presets_are_stable():
    assert hp.ALL_FEATURES_PRESET == "all"
    assert hp.FEATURE_SUBSET_PRESET == "feature_subset"
    assert hp.AVAILABLE_HEAD_FEATURE_PRESETS == (
        "all",
        "feature_subset",
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


def test_feature_subset_preset_counts_and_heads():
    cfg = hp.head_feature_config_for_preset("feature_subset")
    assert isinstance(cfg, hf.HeadFeatureConfig)
    assert set(cfg.feature_columns_by_head) == set(lm.MODEL_HEADS)
    assert len(cfg.feature_columns_by_head[lm.DIRECTION_HEAD]) == 20
    assert len(cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]) == 30
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]) == 13
    assert len(cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]) == 6
    assert hp.preset_feature_counts("feature_subset") == {
        lm.DIRECTION_HEAD: 20,
        lm.NO_MOVE_HEAD: 30,
        lm.MAGNITUDE_UP_HEAD: 13,
        lm.MAGNITUDE_DOWN_HEAD: 6,
    }


def test_feature_subset_has_no_duplicate_columns():
    cfg = hp.head_feature_config_for_preset("feature_subset")
    for cols in cfg.feature_columns_by_head.values():
        assert len(cols) == len(set(cols))


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        hp.head_feature_config_for_preset("__missing__")
    with pytest.raises(ValueError):
        hp.preset_feature_counts("__missing__")


def test_feature_subset_anchor_features_present():
    cfg = hp.head_feature_config_for_preset("feature_subset")
    assert "x_depth_imbalance_within_1bps" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_trade_imbalance_notional_500000us" in cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_log_events_1000000us" in cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_trade_count_per_second_1000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_bid_depth_notional_5bps" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_max_trade_silence_gap_3000000us" in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]


def test_feature_subset_no_move_keeps_core_features():
    cfg = hp.head_feature_config_for_preset("feature_subset")
    no_move = cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]

    assert "x_ask_l1_notional_usd" in no_move
    assert "x_bid_l1_notional_usd" in no_move
    assert "x_total_depth_notional_5bps" in no_move
    assert "x_spread_state_transition_rate_3000000us" in no_move
    assert "x_log_events_1000000us" in no_move
    assert "x_touch_flicker_score_3000000us" in no_move
    assert "x_bid_price_change_rate_1000000us" in no_move
    assert "x_bid_l1_depletion_1000000us" in no_move
    assert "x_trade_sign_entropy_3000000us" in no_move


def test_feature_subset_excludes_removed_features():
    cfg = hp.head_feature_config_for_preset("feature_subset")

    direction = cfg.feature_columns_by_head[lm.DIRECTION_HEAD]
    assert "x_down_up_vol_imbalance_3000000us" not in direction
    assert "x_ask_l1_depletion_over_depth_500000us" not in direction
    assert "x_last_trade_side_sign" not in direction
    assert "x_bid_depth_notional_5bps" not in direction
    assert "x_bid_depth_centroid_bps_25bps" not in direction

    no_move = cfg.feature_columns_by_head[lm.NO_MOVE_HEAD]
    assert "x_ask_l1_depletion_1000000us" not in no_move
    assert "x_top5_trade_notional_sum_usd_200000us" not in no_move
    assert "x_top5_trade_notional_sum_usd_500000us" not in no_move
    assert "x_regime_volume_ewma_500000us" not in no_move
    assert "x_bid_depth_centroid_bps_25bps" not in no_move
    assert "x_bsz1" not in no_move
    assert "x_ask_depth_centroid_bps_25bps" not in no_move
    assert "x_asz1" not in no_move

    mag_up = cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_spread_z_500000us" not in mag_up
    assert "x_ofi_l1_pressure_over_depth_5bps_200000us" not in mag_up

    mag_down = cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert len(mag_down) == 6
