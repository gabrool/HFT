import inspect

import pytest

from mmrt.linear import head_feature_presets as hp
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm


def test_available_presets_are_stable():
    assert hp.ALL_FEATURES_PRESET == "all"
    assert hp.CORR_PRUNED152_HEAD_SUBSET_V1 == "corr_pruned152_head_subset_v1"
    assert hp.AVAILABLE_HEAD_FEATURE_PRESETS == (
        "all",
        "corr_pruned152_head_subset_v1",
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


def test_corr_pruned152_preset_has_no_duplicate_columns():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v1")
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


def test_harmful_features_excluded_from_specific_heads():
    cfg = hp.head_feature_config_for_preset("corr_pruned152_head_subset_v1")
    assert "x_ask_l1_notional_usd" not in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_obi_l1" not in cfg.feature_columns_by_head[lm.MAGNITUDE_UP_HEAD]
    assert "x_spread_z_1000000us" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
    assert "x_cvd_change_usd_1000000us" not in cfg.feature_columns_by_head[lm.MAGNITUDE_DOWN_HEAD]
