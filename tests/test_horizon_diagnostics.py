import pytest

from mmrt.execution.contracts import Fill, FillReason, OrderSide
from mmrt.execution.horizon_diagnostics import (
    HorizonDiagnosticsAccumulator,
    HorizonDiagnosticsConfig,
    quote_mode_from_bools,
)


def test_horizon_diagnostics_aggregates_future_equity_and_fill_markouts():
    acc = HorizonDiagnosticsAccumulator(
        decision_local_ts_us=[100, 250_100, 500_100, 1_000_100],
        mid_prices=[100.0, 101.0, 102.0, 103.0],
        tick_size=1.0,
        contract_size=1.0,
        linear_expected_return_bps=[1.0, -1.0, 0.0, 0.0],
        linear_confidence=[0.8, 0.7, 0.1, 0.1],
        config=HorizonDiagnosticsConfig(horizons_us=(250_000, 500_000, 1_000_000, 2_000_000)),
    )
    acc.start_episode()
    first = acc.add_decision_record(
        step_index=0,
        decision_row=0,
        next_decision_row=1,
        previous_equity=0.0,
        current_equity=0.9,
        immediate_reward=0.9,
        cash_after_step=-100.1,
        inventory_after_step=1.0,
        effective_quote_mode="bid_only",
        requested_quote_mode="bid_only",
        quote_bid_enabled=True,
        quote_ask_enabled=False,
        fill_count=2,
        buy_fill_qty=1.0,
        sell_fill_qty=1.0,
    )
    acc.add_fill_record(
        Fill(
            order_id=1,
            side=OrderSide.BUY,
            local_ts_us=100,
            event_seq=0,
            price_tick=100,
            qty=1.0,
            fee=0.1,
            reason=FillReason.TRADE_THROUGH,
        ),
        decision_record=first,
    )
    acc.add_fill_record(
        Fill(
            order_id=2,
            side=OrderSide.SELL,
            local_ts_us=100,
            event_seq=1,
            price_tick=100,
            qty=1.0,
            fee=-0.05,
            reason=FillReason.QUEUE_DEPLETION,
        ),
        decision_record=first,
    )
    acc.add_decision_record(
        step_index=1,
        decision_row=1,
        next_decision_row=2,
        previous_equity=0.9,
        current_equity=1.9,
        immediate_reward=1.0,
        cash_after_step=-100.1,
        inventory_after_step=1.0,
        effective_quote_mode="no_quote",
        requested_quote_mode="no_quote",
        quote_bid_enabled=False,
        quote_ask_enabled=False,
        fill_count=0,
        buy_fill_qty=0.0,
        sell_fill_qty=0.0,
    )
    acc.add_decision_record(
        step_index=2,
        decision_row=2,
        next_decision_row=3,
        previous_equity=1.9,
        current_equity=3.0,
        immediate_reward=1.1,
        cash_after_step=-100.0,
        inventory_after_step=1.0,
        effective_quote_mode="two_sided",
        requested_quote_mode="two_sided",
        quote_bid_enabled=True,
        quote_ask_enabled=True,
        fill_count=0,
        buy_fill_qty=0.0,
        sell_fill_qty=0.0,
    )

    payload = acc.as_dict()
    h250 = payload["decision_level"]["by_horizon"]["250000"]["all"]
    assert h250["count"] == 3
    assert h250["available_count"] == 3
    assert h250["actual_path_equity_delta_mean"] == pytest.approx(1.0)
    assert h250["carry_mark_equity_delta_mean"] == pytest.approx(1.0)

    h1s_bid = payload["decision_level"]["by_horizon"]["1000000"]["bid_only"]
    assert h1s_bid["available_count"] == 1
    assert h1s_bid["actual_path_equity_delta_mean"] == pytest.approx(3.0)
    assert h1s_bid["carry_mark_equity_delta_mean"] == pytest.approx(2.9)
    assert h1s_bid["carry_mark_increment_after_step_mean"] == pytest.approx(2.0)
    assert h1s_bid["future_mid_return_bps_mean"] == pytest.approx(300.0)

    h2s = payload["decision_level"]["by_horizon"]["2000000"]["all"]
    assert h2s["available_count"] == 0
    assert h2s["unavailable_count"] == 3

    fill_250 = payload["fill_markouts"]["by_horizon"]["250000"]
    assert fill_250["all"]["fill_count"] == 2
    assert fill_250["side:buy"]["gross_markout_bps_mean"] == pytest.approx(100.0)
    assert fill_250["side:buy"]["net_markout_bps_mean"] == pytest.approx(90.0)
    assert fill_250["side:sell"]["gross_markout_bps_mean"] == pytest.approx(-100.0)
    assert fill_250["side:sell"]["net_markout_bps_mean"] == pytest.approx(-95.0)
    assert fill_250["reason:trade_through"]["fill_count"] == 1
    assert fill_250["reason:queue_depletion"]["fill_count"] == 1

    alignment = payload["signal_alignment"]["signal_alignment_by_horizon"]["250000"]
    assert alignment["count"] == 3
    assert alignment["mean_predicted_return_bps"] == pytest.approx(0.0)
    buckets = payload["signal_alignment"]["action_by_signal_bucket"]["buckets"]
    assert buckets["positive"]["bid_enabled_rate"] == pytest.approx(1.0)
    assert buckets["negative"]["no_quote_rate"] == pytest.approx(1.0)


def test_horizon_config_validation_and_quote_mode_names():
    with pytest.raises(ValueError, match="sorted"):
        HorizonDiagnosticsConfig(horizons_us=(500_000, 250_000))
    with pytest.raises(ValueError, match="unique"):
        HorizonDiagnosticsConfig(horizons_us=(250_000, 250_000))
    with pytest.raises(ValueError, match="empty"):
        HorizonDiagnosticsConfig(horizons_us=())

    assert quote_mode_from_bools(False, False) == "no_quote"
    assert quote_mode_from_bools(True, False) == "bid_only"
    assert quote_mode_from_bools(False, True) == "ask_only"
    assert quote_mode_from_bools(True, True) == "two_sided"
