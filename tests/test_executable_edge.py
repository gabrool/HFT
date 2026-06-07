import pytest

from mmrt.execution.contracts import LinearSignal, OrderSide
from mmrt.execution.executable_edge import compute_side_executable_edge


def _signal(ret=2.0):
    return LinearSignal(
        p_no_move=0.5,
        p_move=0.5,
        p_up_move=0.3,
        p_down_move=0.2,
        signed_move_prob=0.1,
        expected_up_bps=3.0,
        expected_down_bps=1.0,
        expected_return_bps=ret,
        expected_abs_move_bps=2.0,
        predicted_vol_bps=2.0,
        confidence=1.0,
    )


def test_executable_edge_bid_ask_alpha_and_costs():
    preds = {
        "bid_touch_filled": 0.5,
        "bid_touch_toxic_cost_bps": 1.0,
        "ask_touch_filled": 0.5,
        "ask_touch_toxic_cost_bps": 1.0,
    }
    bid = compute_side_executable_edge(candidate_name="touch", side=OrderSide.BUY, mid_tick=100.0, price_tick=99, linear_signal=_signal(2.0), adverse_predictions=preds)
    ask = compute_side_executable_edge(candidate_name="touch", side=OrderSide.SELL, mid_tick=100.0, price_tick=101, linear_signal=_signal(2.0), adverse_predictions=preds)
    assert bid.alpha_bps == pytest.approx(2.0)
    assert ask.alpha_bps == pytest.approx(-2.0)
    assert bid.maker_rebate_bps > 0.0
    assert bid.adverse_cost_bps_uncond == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_side_executable_edge(candidate_name="away_1", side=OrderSide.BUY, mid_tick=100.0, price_tick=99, linear_signal=_signal(), adverse_predictions=preds)

from pathlib import Path


def test_executable_edge_spread_capture_is_signed():
    preds = {
        "bid_touch_filled": 1.0,
        "bid_touch_toxic_cost_bps": 0.0,
        "ask_touch_filled": 1.0,
        "ask_touch_toxic_cost_bps": 0.0,
    }
    bid_below_mid = compute_side_executable_edge(candidate_name="touch", side=OrderSide.BUY, mid_tick=100.0, price_tick=99, linear_signal=_signal(0.0), adverse_predictions=preds)
    bid_above_mid = compute_side_executable_edge(candidate_name="touch", side=OrderSide.BUY, mid_tick=100.0, price_tick=101, linear_signal=_signal(0.0), adverse_predictions=preds)
    ask_above_mid = compute_side_executable_edge(candidate_name="touch", side=OrderSide.SELL, mid_tick=100.0, price_tick=101, linear_signal=_signal(0.0), adverse_predictions=preds)
    ask_below_mid = compute_side_executable_edge(candidate_name="touch", side=OrderSide.SELL, mid_tick=100.0, price_tick=99, linear_signal=_signal(0.0), adverse_predictions=preds)
    assert bid_below_mid.spread_capture_bps == pytest.approx(100.0)
    assert bid_above_mid.spread_capture_bps == pytest.approx(-100.0)
    assert ask_above_mid.spread_capture_bps == pytest.approx(100.0)
    assert ask_below_mid.spread_capture_bps == pytest.approx(-100.0)


def test_executable_edge_does_not_floor_spread_capture():
    source = Path("mmrt/execution/executable_edge.py").read_text(encoding="utf-8")
    assert "max(mid_tick - price_tick, 0.0)" not in source
    assert "max(price_tick - mid_tick, 0.0)" not in source
