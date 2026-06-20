from dataclasses import fields
from pathlib import Path

import pytest

from mmrt.execution.contracts import LinearSignal, OrderSide
from mmrt.execution.executable_edge import (
    ExecutableEdgeConfig,
    SideExecutableEdge,
    compute_side_executable_edge,
)


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
    assert not hasattr(bid, "adverse_cost_bps_cond")
    assert not hasattr(bid, "edge_cond_fill_bps")
    with pytest.raises(ValueError):
        compute_side_executable_edge(candidate_name="away_1", side=OrderSide.BUY, mid_tick=100.0, price_tick=99, linear_signal=_signal(), adverse_predictions=preds)


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


def test_executable_edge_removed_conditional_fill_fields_are_absent():
    config_fields = {field.name for field in fields(ExecutableEdgeConfig)}
    edge_fields = {field.name for field in fields(SideExecutableEdge)}

    assert "probability_epsilon" not in config_fields
    assert "adverse_cost_bps_cond" not in edge_fields
    assert "edge_cond_fill_bps" not in edge_fields


def test_executable_edge_zero_fill_probability_keeps_attempt_edge_only():
    preds = {
        "bid_touch_filled": 0.0,
        "bid_touch_toxic_cost_bps": 1.25,
    }

    edge = compute_side_executable_edge(
        candidate_name="touch",
        side=OrderSide.BUY,
        mid_tick=100.0,
        price_tick=99,
        linear_signal=_signal(2.0),
        adverse_predictions=preds,
        config=ExecutableEdgeConfig(latency_buffer_bps=0.75, inventory_skew_bps_per_unit=0.5),
        inventory_qty=2.0,
    )

    assert edge.fill_prob == pytest.approx(0.0)
    assert edge.edge_attempt_bps == pytest.approx(-1.25 - 0.75 - 1.0)
    assert not edge.quote_allowed
    assert not hasattr(edge, "edge_cond_fill_bps")
