"""Named per-head feature presets for MMRT linear models.

This module defines static feature-column subsets used by storage-backed
linear training. It does not read storage, compute features, build labels,
fit preprocessing, train models, evaluate metrics, or parse raw Tardis data.
"""

from typing import Sequence

from mmrt.linear import head_features as hf
from mmrt.linear import models as lm

ALL_FEATURES_PRESET = "all"
CORR_PRUNED152_HEAD_SUBSET_V1 = "corr_pruned152_head_subset_v1"
CORR_PRUNED152_HEAD_SUBSET_V2 = "corr_pruned152_head_subset_v2"
CORR_PRUNED152_HEAD_SUBSET_V3 = "corr_pruned152_head_subset_v3"
CORR_PRUNED152_HEAD_SUBSET_V4 = "corr_pruned152_head_subset_v4"

AVAILABLE_HEAD_FEATURE_PRESETS = (
    ALL_FEATURES_PRESET,
    CORR_PRUNED152_HEAD_SUBSET_V1,
    CORR_PRUNED152_HEAD_SUBSET_V2,
    CORR_PRUNED152_HEAD_SUBSET_V3,
    CORR_PRUNED152_HEAD_SUBSET_V4,
)

DIRECTION_CORR_PRUNED152_V1 = (
    "x_depth_imbalance_within_1bps",
    "x_depth_imbalance_5bps_mean_1000000us",
    "x_bid_l1_notional_usd",
    "x_ask_l1_notional_usd",
    "x_trade_imbalance_notional_500000us",
    "x_obi_l1",
    "x_max_signed_trade_notional_usd_1000000us",
    "x_trade_side_quote_response_asymmetry_500000us",
    "x_ofi_l1_pressure_over_realized_vol_1000000us",
    "x_near_touch_depth_drop_asymmetry",
    "x_cvd_change_usd_1000000us",
    "x_best_ask_size_age_us",
    "x_ofi_l3_accel_200000us_minus_500000us",
    "x_last_trade_side_sign",
    "x_time_since_last_sell_trade_us",
    "x_asz1",
    "x_micro_l10_minus_mid_bps",
    "x_bid_depth_notional_5bps",
    "x_down_up_vol_imbalance_3000000us",
    "x_micro_minus_mid_bps",
    "x_ask_depth_notional_5bps",
    "x_ask_l1_add_rate_over_depth_1000000us",
    "x_ask_l1_add_rate_over_depth_200000us",
    "x_ofi_l10_sum_over_depth_1000000us",
    "x_ask_l1_depletion_over_depth_500000us",
    "x_time_since_last_buy_trade_us",
    "x_signed_trade_count_imbalance_500000us",
    "x_ask_l1_add_rate_over_depth_500000us",
    "x_spread_z_3000000us",
    "x_bid_l1_depletion_1000000us",
    "x_depth_imbalance_5bps_slope_500000us",
    "x_bid_l1_depletion_200000us",
    "x_depth_imbalance_5bps_slope_3000000us",
    "x_absorption_bid_1000000us",
    "x_absorption_ask_1000000us",
    "x_best_bid_size_age_us",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_max_signed_trade_notional_usd_500000us",
    "x_obi_l3_mean_500000us",
    "x_bid_depth_centroid_bps_25bps",
)

NO_MOVE_CORR_PRUNED152_V1 = (
    "x_ask_l1_notional_usd",
    "x_bid_l1_notional_usd",
    "x_total_depth_notional_5bps",
    "x_spread_state_transition_rate_3000000us",
    "x_time_since_mid_change_us",
    "x_l1_churn_over_depth_1000000us",
    "x_touch_flicker_score_3000000us",
    "x_log_events_1000000us",
    "x_bid_depth_notional_5bps",
    "x_ask_depth_notional_5bps",
    "x_log_events_500000us",
    "x_ob_update_rate_500000us",
    "x_log_events_3000000us",
    "x_trade_count_per_second_1000000us",
    "x_trade_count_per_second_200000us",
    "x_log_events_200000us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_ask_l1_depletion_over_depth_1000000us",
    "x_ob_update_rate_200000us",
    "x_bid_l1_depletion_1000000us",
    "x_trade_count_per_second_500000us",
    "x_ask_l1_depletion_1000000us",
    "x_time_since_last_sell_trade_us",
    "x_time_since_last_buy_trade_us",
    "x_microprice_zero_cross_rate_1000000us",
    "x_top5_trade_notional_sum_usd_200000us",
    "x_microprice_realized_vol_1000000us",
    "x_bid_l1_rem_rate_over_depth_200000us",
    "x_asz1",
    "x_ask_depth_centroid_bps_25bps",
    "x_bsz1",
    "x_zero_tick_fraction_1000000us",
    "x_bid_depth_centroid_bps_25bps",
    "x_regime_volume_ewma_500000us",
    "x_same_side_trade_cluster_notional_1000000us",
    "x_top5_trade_notional_sum_usd_500000us",
    "x_bid_price_change_rate_1000000us",
    "x_absorption_ask_1000000us",
    "x_trade_sign_entropy_3000000us",
    "x_max_abs_return_bps_500000us",
)

MAGNITUDE_UP_CORR_PRUNED152_V1 = (
    "x_ask_depth_notional_5bps",
    "x_bid_depth_notional_5bps",
    "x_spread_z_500000us",
    "x_ask_depth_within_1bps",
    "x_ofi_l5_sum_over_depth_500000us",
    "x_micro_l10_minus_mid_bps",
    "x_depth_imbalance_5bps_slope_1000000us",
    "x_ofi_l1_pressure_over_depth_5bps_200000us",
    "x_ofi_l5_sum_over_depth_200000us",
    "x_ofi_l3_over_depth_5bps",
    "x_mid_range_bps_500000us",
    "x_ofi_l10_sum_over_depth_1000000us",
    "x_bid_price_change_rate_500000us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_log_events_1000000us",
    "x_spread_state_transition_rate_3000000us",
    "x_ask_l1_depletion_500000us",
    "x_bid_price_change_rate_1000000us",
    "x_same_side_trade_cluster_notional_1000000us",
    "x_ob_update_rate_500000us",
    "x_total_depth_notional_5bps",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_bid_l1_depletion_500000us",
    "x_log_events_500000us",
    "x_bid_l1_add_rate_over_depth_200000us",
    "x_trade_count_per_second_1000000us",
    "x_trade_sign_entropy_3000000us",
    "x_ofi_l1_pressure_over_realized_vol_200000us",
    "x_absorption_ask_500000us",
    "x_absorption_bid_200000us",
)

MAGNITUDE_DOWN_CORR_PRUNED152_V1 = (
    "x_time_since_mid_change_us",
    "x_bid_l1_notional_usd",
    "x_asz1",
    "x_bid_depth_notional_5bps",
    "x_bsz1",
    "x_bid_depth_centroid_bps_25bps",
    "x_depth_imbalance_5bps_mean_1000000us",
    "x_bid_price_change_rate_200000us",
    "x_ask_depth_within_1bps",
    "x_depth_imbalance_within_1bps",
    "x_opposite_side_replenishment_after_depletion_200000us",
    "x_trade_count_per_second_200000us",
    "x_trade_count_per_second_500000us",
    "x_total_depth_notional_5bps",
    "x_obi_l1",
    "x_ask_l1_notional_usd",
    "x_last_trade_side_sign",
    "x_ob_update_rate_500000us",
    "x_mid_slope_bps_per_sec_1000000us",
    "x_microprice_zero_cross_rate_1000000us",
    "x_bid_price_change_rate_500000us",
    "x_max_trade_silence_gap_3000000us",
    "x_ask_l1_depletion_over_depth_1000000us",
    "x_trade_count_per_second_1000000us",
    "x_consecutive_buy_trade_count",
    "x_down_up_vol_imbalance_3000000us",
    "x_obi_l3_mean_500000us",
    "x_mid_slope_bps_per_sec_500000us",
    "x_bid_price_change_rate_1000000us",
    "x_vamp_l10_minus_mid_bps",
    "x_touch_flicker_score_3000000us",
    "x_depth_imbalance_5bps_slope_500000us",
    "x_depth_5bps_z_3000000us",
    "x_spread_state_transition_rate_3000000us",
    "x_mid_range_bps_500000us",
    "x_log_events_100000us",
    "x_regime_volume_ewma_500000us",
    "x_signed_trade_premium_bps_volume_weighted_500000us",
    "x_l1_churn_over_depth_1000000us",
    "x_bid_depth_within_1bps",
)

CORR_PRUNED152_HEAD_SUBSET_V1_COLUMNS_BY_HEAD = {
    lm.DIRECTION_HEAD: DIRECTION_CORR_PRUNED152_V1,
    lm.NO_MOVE_HEAD: NO_MOVE_CORR_PRUNED152_V1,
    lm.MAGNITUDE_UP_HEAD: MAGNITUDE_UP_CORR_PRUNED152_V1,
    lm.MAGNITUDE_DOWN_HEAD: MAGNITUDE_DOWN_CORR_PRUNED152_V1,
}


DIRECTION_CORR_PRUNED152_V2 = (
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
    "x_ask_l1_depletion_over_depth_500000us",
    "x_micro_minus_mid_bps",
    "x_time_since_last_sell_trade_us",
    "x_last_trade_side_sign",
    "x_down_up_vol_imbalance_3000000us",
    "x_bid_depth_notional_5bps",
    "x_ask_l1_add_rate_over_depth_1000000us",
    "x_bid_depth_centroid_bps_25bps",
    "x_asz1",
    "x_depth_imbalance_5bps_slope_500000us",
    "x_ask_depth_notional_5bps",
    "x_micro_l10_minus_mid_bps",
    "x_ofi_l3_accel_200000us_minus_500000us",
    "x_signed_trade_count_imbalance_500000us",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_bid_l1_depletion_200000us",
)

NO_MOVE_CORR_PRUNED152_V2 = (
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
    "x_bsz1",
    "x_asz1",
    "x_ask_depth_centroid_bps_25bps",
    "x_top5_trade_notional_sum_usd_500000us",
    "x_bid_price_change_rate_1000000us",
    "x_bid_depth_centroid_bps_25bps",
    "x_ask_l1_depletion_1000000us",
    "x_top5_trade_notional_sum_usd_200000us",
    "x_regime_volume_ewma_500000us",
    "x_absorption_ask_1000000us",
)

MAGNITUDE_UP_CORR_PRUNED152_V2 = (
    "x_trade_count_per_second_1000000us",
    "x_bid_depth_notional_5bps",
    "x_trade_sign_entropy_3000000us",
    "x_log_events_1000000us",
    "x_ask_depth_within_1bps",
    "x_ob_update_rate_500000us",
    "x_ask_l1_depletion_500000us",
    "x_ask_depth_notional_5bps",
    "x_depth_imbalance_5bps_slope_1000000us",
    "x_spread_z_500000us",
    "x_micro_l10_minus_mid_bps",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_ofi_l10_sum_over_depth_1000000us",
    "x_total_depth_notional_5bps",
    "x_ofi_l1_pressure_over_depth_5bps_200000us",
    "x_bid_l1_depletion_500000us",
    "x_bid_l1_add_rate_over_depth_200000us",
    "x_ofi_l3_over_depth_5bps",
)

MAGNITUDE_DOWN_CORR_PRUNED152_V2 = (
    "x_bid_depth_notional_5bps",
    "x_touch_flicker_score_3000000us",
    "x_vamp_l10_minus_mid_bps",
    "x_max_trade_silence_gap_3000000us",
    "x_ob_update_rate_500000us",
    "x_mid_range_bps_500000us",
    "x_microprice_zero_cross_rate_1000000us",
    "x_mid_slope_bps_per_sec_1000000us",
    "x_regime_volume_ewma_500000us",
)

CORR_PRUNED152_HEAD_SUBSET_V2_COLUMNS_BY_HEAD = {
    lm.DIRECTION_HEAD: DIRECTION_CORR_PRUNED152_V2,
    lm.NO_MOVE_HEAD: NO_MOVE_CORR_PRUNED152_V2,
    lm.MAGNITUDE_UP_HEAD: MAGNITUDE_UP_CORR_PRUNED152_V2,
    lm.MAGNITUDE_DOWN_HEAD: MAGNITUDE_DOWN_CORR_PRUNED152_V2,
}


DIRECTION_CORR_PRUNED152_V3 = (
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
    "x_ask_l1_depletion_over_depth_500000us",
    "x_micro_minus_mid_bps",
    "x_time_since_last_sell_trade_us",
    "x_last_trade_side_sign",
    "x_down_up_vol_imbalance_3000000us",
    "x_bid_depth_notional_5bps",
    "x_bid_depth_centroid_bps_25bps",
)

NO_MOVE_CORR_PRUNED152_V3 = (
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
    "x_bsz1",
    "x_asz1",
    "x_ask_depth_centroid_bps_25bps",
    "x_top5_trade_notional_sum_usd_500000us",
    "x_bid_price_change_rate_1000000us",
    "x_bid_depth_centroid_bps_25bps",
    "x_ask_l1_depletion_1000000us",
    "x_top5_trade_notional_sum_usd_200000us",
    "x_regime_volume_ewma_500000us",
)

MAGNITUDE_UP_CORR_PRUNED152_V3 = (
    "x_trade_count_per_second_1000000us",
    "x_bid_depth_notional_5bps",
    "x_trade_sign_entropy_3000000us",
    "x_log_events_1000000us",
    "x_ask_depth_within_1bps",
    "x_ob_update_rate_500000us",
    "x_ask_l1_depletion_500000us",
    "x_ask_depth_notional_5bps",
    "x_depth_imbalance_5bps_slope_1000000us",
    "x_spread_z_500000us",
    "x_micro_l10_minus_mid_bps",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_ofi_l10_sum_over_depth_1000000us",
    "x_ofi_l1_pressure_over_depth_5bps_200000us",
)

MAGNITUDE_DOWN_CORR_PRUNED152_V3 = (
    "x_bid_depth_notional_5bps",
    "x_touch_flicker_score_3000000us",
    "x_max_trade_silence_gap_3000000us",
    "x_ob_update_rate_500000us",
    "x_microprice_zero_cross_rate_1000000us",
    "x_mid_slope_bps_per_sec_1000000us",
)

CORR_PRUNED152_HEAD_SUBSET_V3_COLUMNS_BY_HEAD = {
    lm.DIRECTION_HEAD: DIRECTION_CORR_PRUNED152_V3,
    lm.NO_MOVE_HEAD: NO_MOVE_CORR_PRUNED152_V3,
    lm.MAGNITUDE_UP_HEAD: MAGNITUDE_UP_CORR_PRUNED152_V3,
    lm.MAGNITUDE_DOWN_HEAD: MAGNITUDE_DOWN_CORR_PRUNED152_V3,
}

DIRECTION_CORR_PRUNED152_V4 = (
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
    "x_ask_l1_depletion_over_depth_500000us",
    "x_micro_minus_mid_bps",
    "x_time_since_last_sell_trade_us",
    "x_last_trade_side_sign",
    "x_down_up_vol_imbalance_3000000us",
)

NO_MOVE_CORR_PRUNED152_V4 = NO_MOVE_CORR_PRUNED152_V3

MAGNITUDE_UP_CORR_PRUNED152_V4 = (
    "x_trade_count_per_second_1000000us",
    "x_bid_depth_notional_5bps",
    "x_trade_sign_entropy_3000000us",
    "x_log_events_1000000us",
    "x_ask_depth_within_1bps",
    "x_ob_update_rate_500000us",
    "x_ask_l1_depletion_500000us",
    "x_ask_depth_notional_5bps",
    "x_depth_imbalance_5bps_slope_1000000us",
    "x_spread_z_500000us",
    "x_micro_l10_minus_mid_bps",
    "x_ask_l1_depletion_over_depth_200000us",
    "x_bid_l1_depletion_over_depth_1000000us",
    "x_ofi_l10_sum_over_depth_1000000us",
)

MAGNITUDE_DOWN_CORR_PRUNED152_V4 = MAGNITUDE_DOWN_CORR_PRUNED152_V3

CORR_PRUNED152_HEAD_SUBSET_V4_COLUMNS_BY_HEAD = {
    lm.DIRECTION_HEAD: DIRECTION_CORR_PRUNED152_V4,
    lm.NO_MOVE_HEAD: NO_MOVE_CORR_PRUNED152_V4,
    lm.MAGNITUDE_UP_HEAD: MAGNITUDE_UP_CORR_PRUNED152_V4,
    lm.MAGNITUDE_DOWN_HEAD: MAGNITUDE_DOWN_CORR_PRUNED152_V4,
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


assert len(DIRECTION_CORR_PRUNED152_V1) == 40
assert len(NO_MOVE_CORR_PRUNED152_V1) == 40
assert len(MAGNITUDE_UP_CORR_PRUNED152_V1) == 30
assert len(MAGNITUDE_DOWN_CORR_PRUNED152_V1) == 40
assert len(DIRECTION_CORR_PRUNED152_V2) == 34
assert len(NO_MOVE_CORR_PRUNED152_V2) == 39
assert len(MAGNITUDE_UP_CORR_PRUNED152_V2) == 19
assert len(MAGNITUDE_DOWN_CORR_PRUNED152_V2) == 9
assert len(DIRECTION_CORR_PRUNED152_V3) == 25
assert len(NO_MOVE_CORR_PRUNED152_V3) == 38
assert len(MAGNITUDE_UP_CORR_PRUNED152_V3) == 15
assert len(MAGNITUDE_DOWN_CORR_PRUNED152_V3) == 6
assert len(DIRECTION_CORR_PRUNED152_V4) == 23
assert len(NO_MOVE_CORR_PRUNED152_V4) == 38
assert len(MAGNITUDE_UP_CORR_PRUNED152_V4) == 14
assert len(MAGNITUDE_DOWN_CORR_PRUNED152_V4) == 6

for _cols in (
    DIRECTION_CORR_PRUNED152_V1,
    NO_MOVE_CORR_PRUNED152_V1,
    MAGNITUDE_UP_CORR_PRUNED152_V1,
    MAGNITUDE_DOWN_CORR_PRUNED152_V1,
    DIRECTION_CORR_PRUNED152_V2,
    NO_MOVE_CORR_PRUNED152_V2,
    MAGNITUDE_UP_CORR_PRUNED152_V2,
    MAGNITUDE_DOWN_CORR_PRUNED152_V2,
    DIRECTION_CORR_PRUNED152_V3,
    NO_MOVE_CORR_PRUNED152_V3,
    MAGNITUDE_UP_CORR_PRUNED152_V3,
    MAGNITUDE_DOWN_CORR_PRUNED152_V3,
    DIRECTION_CORR_PRUNED152_V4,
    NO_MOVE_CORR_PRUNED152_V4,
    MAGNITUDE_UP_CORR_PRUNED152_V4,
    MAGNITUDE_DOWN_CORR_PRUNED152_V4,
):
    _assert_unique_columns(_cols)

assert set(CORR_PRUNED152_HEAD_SUBSET_V1_COLUMNS_BY_HEAD) == set(lm.MODEL_HEADS)
assert set(CORR_PRUNED152_HEAD_SUBSET_V2_COLUMNS_BY_HEAD) == set(lm.MODEL_HEADS)
assert set(CORR_PRUNED152_HEAD_SUBSET_V3_COLUMNS_BY_HEAD) == set(lm.MODEL_HEADS)
assert set(CORR_PRUNED152_HEAD_SUBSET_V4_COLUMNS_BY_HEAD) == set(lm.MODEL_HEADS)


def head_feature_config_for_preset(name: str) -> hf.HeadFeatureConfig:
    preset = _require_preset_name(name)
    if preset == ALL_FEATURES_PRESET:
        return hf.HeadFeatureConfig()
    if preset == CORR_PRUNED152_HEAD_SUBSET_V1:
        return hf.HeadFeatureConfig(
            feature_columns_by_head=CORR_PRUNED152_HEAD_SUBSET_V1_COLUMNS_BY_HEAD
        )
    if preset == CORR_PRUNED152_HEAD_SUBSET_V2:
        return hf.HeadFeatureConfig(
            feature_columns_by_head=CORR_PRUNED152_HEAD_SUBSET_V2_COLUMNS_BY_HEAD
        )
    if preset == CORR_PRUNED152_HEAD_SUBSET_V3:
        return hf.HeadFeatureConfig(
            feature_columns_by_head=CORR_PRUNED152_HEAD_SUBSET_V3_COLUMNS_BY_HEAD
        )
    if preset == CORR_PRUNED152_HEAD_SUBSET_V4:
        return hf.HeadFeatureConfig(
            feature_columns_by_head=CORR_PRUNED152_HEAD_SUBSET_V4_COLUMNS_BY_HEAD
        )
    raise AssertionError("unreachable")


def preset_feature_counts(name: str) -> dict[str, int]:
    cfg = head_feature_config_for_preset(name)
    if cfg.feature_columns_by_head is None:
        return {}
    return {head: len(cols) for head, cols in cfg.feature_columns_by_head.items()}


__all__ = [
    "ALL_FEATURES_PRESET",
    "CORR_PRUNED152_HEAD_SUBSET_V1",
    "CORR_PRUNED152_HEAD_SUBSET_V2",
    "CORR_PRUNED152_HEAD_SUBSET_V3",
    "CORR_PRUNED152_HEAD_SUBSET_V4",
    "AVAILABLE_HEAD_FEATURE_PRESETS",
    "DIRECTION_CORR_PRUNED152_V1",
    "NO_MOVE_CORR_PRUNED152_V1",
    "MAGNITUDE_UP_CORR_PRUNED152_V1",
    "MAGNITUDE_DOWN_CORR_PRUNED152_V1",
    "DIRECTION_CORR_PRUNED152_V2",
    "NO_MOVE_CORR_PRUNED152_V2",
    "MAGNITUDE_UP_CORR_PRUNED152_V2",
    "MAGNITUDE_DOWN_CORR_PRUNED152_V2",
    "DIRECTION_CORR_PRUNED152_V3",
    "NO_MOVE_CORR_PRUNED152_V3",
    "MAGNITUDE_UP_CORR_PRUNED152_V3",
    "MAGNITUDE_DOWN_CORR_PRUNED152_V3",
    "DIRECTION_CORR_PRUNED152_V4",
    "NO_MOVE_CORR_PRUNED152_V4",
    "MAGNITUDE_UP_CORR_PRUNED152_V4",
    "MAGNITUDE_DOWN_CORR_PRUNED152_V4",
    "CORR_PRUNED152_HEAD_SUBSET_V1_COLUMNS_BY_HEAD",
    "CORR_PRUNED152_HEAD_SUBSET_V2_COLUMNS_BY_HEAD",
    "CORR_PRUNED152_HEAD_SUBSET_V3_COLUMNS_BY_HEAD",
    "CORR_PRUNED152_HEAD_SUBSET_V4_COLUMNS_BY_HEAD",
    "head_feature_config_for_preset",
    "preset_feature_counts",
]
