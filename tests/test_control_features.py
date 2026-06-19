import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.control_features import ControlFeatureTracker, depth_shape_features


def test_depth_shape_features_sums_first_five_levels():
    bid_sizes = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 99.0]], dtype=np.float32)
    ask_sizes = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 99.0]], dtype=np.float32)

    features = depth_shape_features(
        book_ptr=0,
        book_bid_sizes=bid_sizes,
        book_ask_sizes=ask_sizes,
    )

    assert features["depth_bid_qty_5"] == pytest.approx(15.0)
    assert features["depth_ask_qty_5"] == pytest.approx(20.0)
    assert features["depth_imbalance_5"] == pytest.approx(-5.0 / 35.0)


def test_control_tracker_trade_flow_signs_ratios_and_pruning():
    tracker = ControlFeatureTracker()
    tracker.record_trade(local_ts_us=100, side=AggressorSide.BUY, qty=2.0)
    tracker.record_trade(local_ts_us=150, side=AggressorSide.SELL, qty=1.0)

    features = tracker.snapshot(200)
    assert features["flow_signed_qty_200ms"] == pytest.approx(1.0)
    assert features["flow_abs_qty_200ms"] == pytest.approx(3.0)
    assert features["flow_trade_count_200ms"] == pytest.approx(2.0)
    assert features["flow_imbalance_ratio_200ms"] == pytest.approx(1.0 / 3.0)
    assert -1.0 <= features["flow_imbalance_ratio_200ms"] <= 1.0

    pruned = tracker.snapshot(400_000)
    assert pruned["flow_signed_qty_200ms"] == pytest.approx(0.0)
    assert pruned["flow_abs_qty_200ms"] == pytest.approx(0.0)
    assert pruned["flow_trade_count_200ms"] == pytest.approx(0.0)


def test_control_tracker_touch_ratios_same_tick_and_top_moves():
    tracker = ControlFeatureTracker()
    tracker.record_l2_top(local_ts_us=100, best_bid_tick=100, best_bid_size=10.0, best_ask_tick=102, best_ask_size=10.0)
    tracker.record_l2_top(local_ts_us=200, best_bid_tick=100, best_bid_size=6.0, best_ask_tick=102, best_ask_size=12.0)
    tracker.record_l2_top(local_ts_us=300, best_bid_tick=99, best_bid_size=5.0, best_ask_tick=103, best_ask_size=9.0)
    tracker.record_l2_top(local_ts_us=400, best_bid_tick=100, best_bid_size=7.0, best_ask_tick=102, best_ask_size=8.0)

    features = tracker.snapshot(500)

    assert features["bid_touch_depletion_ratio_1000ms"] == pytest.approx(10.0 / 17.0)
    assert features["bid_touch_replenishment_ratio_1000ms"] == pytest.approx(7.0 / 17.0)
    assert features["ask_touch_depletion_ratio_1000ms"] == pytest.approx(12.0 / 22.0)
    assert features["ask_touch_replenishment_ratio_1000ms"] == pytest.approx(10.0 / 22.0)


def test_control_tracker_touch_ratios_zero_without_activity():
    tracker = ControlFeatureTracker()
    tracker.record_l2_top(local_ts_us=100, best_bid_tick=100, best_bid_size=10.0, best_ask_tick=102, best_ask_size=10.0)

    features = tracker.snapshot(200)

    assert features["bid_touch_depletion_ratio_1000ms"] == 0.0
    assert features["bid_touch_replenishment_ratio_1000ms"] == 0.0
    assert features["ask_touch_depletion_ratio_1000ms"] == 0.0
    assert features["ask_touch_replenishment_ratio_1000ms"] == 0.0


def test_control_tracker_recent_mid_return_zero_when_unavailable_and_correct_when_available():
    tracker = ControlFeatureTracker()
    assert tracker.snapshot(100)["recent_mid_return_bps_200ms"] == 0.0

    tracker.record_l2_top(local_ts_us=100, best_bid_tick=100, best_bid_size=1.0, best_ask_tick=102, best_ask_size=1.0)
    tracker.record_l2_top(local_ts_us=800_000, best_bid_tick=100, best_bid_size=1.0, best_ask_tick=102, best_ask_size=1.0)
    tracker.record_l2_top(local_ts_us=1_000_000, best_bid_tick=101, best_bid_size=1.0, best_ask_tick=103, best_ask_size=1.0)

    features = tracker.snapshot(1_000_100)

    assert features["recent_mid_return_bps_200ms"] == pytest.approx((102.0 - 101.0) / 102.0 * 10_000.0)
    assert features["recent_mid_return_bps_1000ms"] == pytest.approx((102.0 - 101.0) / 102.0 * 10_000.0)
