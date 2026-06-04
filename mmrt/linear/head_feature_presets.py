"""Named per-head feature presets for MMRT linear models.

This module defines static feature-column subsets used by storage-backed
linear training. It does not read storage, compute features, build labels,
fit preprocessing, train models, evaluate metrics, or parse raw Tardis data.
"""

from typing import Sequence

from mmrt.linear import head_features as hf
from mmrt.linear import models as lm

ALL_FEATURES_PRESET = "all"
FEATURE_SUBSET_PRESET = "feature_subset"

AVAILABLE_HEAD_FEATURE_PRESETS = (
    ALL_FEATURES_PRESET,
    FEATURE_SUBSET_PRESET,
)

DIRECTION_FEATURE_SUBSET = (
    "x_depth_imbalance_within_1bps",
    "x_depth_imbalance_5bps_mean_1000000us",
    "x_ask_l1_notional_usd",
    "x_bid_l1_notional_usd",
    "x_trade_side_quote_response_asymmetry_500000us",
    "x_trade_imbalance_notional_500000us",
    "x_obi_l1",
    "x_ofi_l10_sum_over_depth_1000000us",
    "x_near_touch_depth_drop_asymmetry",
    "x_max_signed_trade_notional_usd_1000000us",
    "x_ofi_l1_pressure_over_realized_vol_1000000us",
    "x_cvd_change_usd_1000000us",
    "x_best_ask_size_age_us",
    "x_absorption_bid_1000000us",
    "x_depth_imbalance_5bps_slope_3000000us",
    "x_absorption_ask_1000000us",
    "x_best_bid_size_age_us",
    "x_time_since_last_buy_trade_us",
    "x_micro_minus_mid_bps",
    "x_time_since_last_sell_trade_us",
)

NO_MOVE_FEATURE_SUBSET = (
    "x_ask_l1_notional_usd",
    "x_bid_l1_notional_usd",
    "x_total_depth_notional_5bps",
    "x_spread_state_transition_rate_3000000us",
    "x_log_events_500000us",
    "x_touch_flicker_score_3000000us",
    "x_log_events_1000000us",
    "x_l1_churn_over_depth_1000000us",
    "x_time_since_mid_change_us",
    "x_log_events_3000000us",
    "x_ask_depth_notional_5bps",
    "x_bid_depth_notional_5bps",
    "x_ob_update_rate_500000us",
    "x_log_events_200000us",
    "x_ask_l1_depletion_over_depth_1000000us",
    "x_trade_count_per_second_1000000us",
    "x_trade_count_per_second_200000us",
    "x_time_since_last_buy_trade_us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_time_since_last_sell_trade_us",
    "x_ob_update_rate_200000us",
    "x_bid_l1_rem_rate_over_depth_200000us",
    "x_zero_tick_fraction_1000000us",
    "x_microprice_realized_vol_1000000us",
    "x_same_side_trade_cluster_notional_1000000us",
    "x_trade_count_per_second_500000us",
    "x_microprice_zero_cross_rate_1000000us",
    "x_bid_l1_depletion_1000000us",
    "x_trade_sign_entropy_3000000us",
    "x_bid_price_change_rate_1000000us",
)

MAGNITUDE_UP_FEATURE_SUBSET = (
    "x_trade_count_per_second_1000000us",
    "x_bid_depth_notional_5bps",
    "x_trade_sign_entropy_3000000us",
    "x_log_events_1000000us",
    "x_ask_depth_within_1bps",
    "x_ob_update_rate_500000us",
    "x_ask_l1_depletion_500000us",
    "x_ask_depth_notional_5bps",
    "x_depth_imbalance_5bps_slope_1000000us",
    "x_micro_l10_minus_mid_bps",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_ofi_l10_sum_over_depth_1000000us",
)

MAGNITUDE_DOWN_FEATURE_SUBSET = (
    "x_bid_depth_notional_5bps",
    "x_touch_flicker_score_3000000us",
    "x_max_trade_silence_gap_3000000us",
    "x_ob_update_rate_500000us",
    "x_microprice_zero_cross_rate_1000000us",
    "x_mid_slope_bps_per_sec_1000000us",
)

FEATURE_SUBSET_COLUMNS_BY_HEAD = {
    lm.DIRECTION_HEAD: DIRECTION_FEATURE_SUBSET,
    lm.NO_MOVE_HEAD: NO_MOVE_FEATURE_SUBSET,
    lm.MAGNITUDE_UP_HEAD: MAGNITUDE_UP_FEATURE_SUBSET,
    lm.MAGNITUDE_DOWN_HEAD: MAGNITUDE_DOWN_FEATURE_SUBSET,
}


def _require_preset_name(name: str) -> str:
    if name not in AVAILABLE_HEAD_FEATURE_PRESETS:
        raise ValueError(
            f"unknown head feature preset: {name!r}; "
            f"expected one of {AVAILABLE_HEAD_FEATURE_PRESETS}"
        )
    return name


def _assert_unique_columns(cols: Sequence[str]) -> None:
    assert len(cols) == len(set(cols))


assert len(DIRECTION_FEATURE_SUBSET) == 20
assert len(NO_MOVE_FEATURE_SUBSET) == 30
assert len(MAGNITUDE_UP_FEATURE_SUBSET) == 13
assert len(MAGNITUDE_DOWN_FEATURE_SUBSET) == 6

for _cols in (
    DIRECTION_FEATURE_SUBSET,
    NO_MOVE_FEATURE_SUBSET,
    MAGNITUDE_UP_FEATURE_SUBSET,
    MAGNITUDE_DOWN_FEATURE_SUBSET,
):
    _assert_unique_columns(_cols)

assert set(FEATURE_SUBSET_COLUMNS_BY_HEAD) == set(lm.MODEL_HEADS)
assert "x_asz1" not in NO_MOVE_FEATURE_SUBSET
assert "x_top5_trade_notional_sum_usd_500000us" not in NO_MOVE_FEATURE_SUBSET
assert "x_bid_depth_centroid_bps_25bps" not in NO_MOVE_FEATURE_SUBSET
assert "x_bsz1" not in NO_MOVE_FEATURE_SUBSET
assert "x_ask_depth_centroid_bps_25bps" not in NO_MOVE_FEATURE_SUBSET


def head_feature_config_for_preset(name: str) -> hf.HeadFeatureConfig:
    preset = _require_preset_name(name)
    if preset == ALL_FEATURES_PRESET:
        return hf.HeadFeatureConfig()
    if preset == FEATURE_SUBSET_PRESET:
        return hf.HeadFeatureConfig(
            feature_columns_by_head=FEATURE_SUBSET_COLUMNS_BY_HEAD
        )
    raise AssertionError("unreachable")


def preset_feature_counts(name: str) -> dict[str, int]:
    cfg = head_feature_config_for_preset(name)
    if cfg.feature_columns_by_head is None:
        return {}
    return {head: len(cols) for head, cols in cfg.feature_columns_by_head.items()}


__all__ = [
    "ALL_FEATURES_PRESET",
    "FEATURE_SUBSET_PRESET",
    "AVAILABLE_HEAD_FEATURE_PRESETS",
    "DIRECTION_FEATURE_SUBSET",
    "NO_MOVE_FEATURE_SUBSET",
    "MAGNITUDE_UP_FEATURE_SUBSET",
    "MAGNITUDE_DOWN_FEATURE_SUBSET",
    "FEATURE_SUBSET_COLUMNS_BY_HEAD",
    "head_feature_config_for_preset",
    "preset_feature_counts",
]
