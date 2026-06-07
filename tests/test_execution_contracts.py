import math
from decimal import Decimal

import pytest

from mmrt.contracts import AggressorSide, BookSide, TardisDataType
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.execution.contracts import (
    ActionSpec,
    ActiveOrder,
    BookLevelSnapshot,
    BookTop,
    DecisionRef,
    ExecutionEventRef,
    ExecutionEventType,
    ExecutionStepResult,
    ExecutionTapeFormat,
    ExecutionTapeManifest,
    Fill,
    FillReason,
    L2Update,
    L2UpdateBatch,
    LinearSignal,
    OrderSide,
    OrderStatus,
    PositionState,
    QuoteIntent,
    RewardComponents,
    SymbolSpec,
)
from mmrt.time_key import EventKey


def _rules(**overrides):
    kwargs = {
        "exchange": "binance-futures", "symbol": "BTCUSDT", "mode": SymbolRuleMode.CURRENT_RULES_REPLAY,
        "base_asset": "BTC", "quote_asset": "USDT", "margin_asset": "USDT",
        "contract_type": "PERPETUAL", "status": "TRADING",
        "tick_size": Decimal("0.1"), "min_price": Decimal("0.1"), "max_price": Decimal("1000000"),
        "step_size": Decimal("0.001"), "min_qty": Decimal("0.001"), "max_qty": Decimal("100"),
        "min_notional": Decimal("5"), "allowed_order_types": ("LIMIT",), "allowed_time_in_force": ("GTC", "GTX"),
    }
    kwargs.update(overrides)
    return ExchangeSymbolRules(**kwargs)

def _spec(**overrides):
    kwargs = {
        "exchange": "binance-futures",
        "symbol": "BTCUSDT",
        "tick_size": 0.1,
        "step_size": 0.001,
        "min_qty": 0.001,
        "max_qty": 100.0,
        "min_notional": 5.0,
    }
    kwargs.update(overrides)
    return SymbolSpec(**kwargs)


def _update(**overrides):
    kwargs = {
        "local_ts_us": 100,
        "ts_us": 90,
        "side": BookSide.BID,
        "price_tick": 1000,
        "amount": 1.0,
        "is_snapshot": False,
    }
    kwargs.update(overrides)
    return L2Update(**kwargs)


def test_execution_contracts_importable():
    import mmrt.execution.contracts as c

    assert c.ExecutionTapeFormat.L2_TRADES_ARRAYS.value == "l2_trades_arrays"


def test_symbol_spec_conversions_and_validation():
    spec = _spec()

    assert spec.price_to_tick(100.0) == 1000
    assert spec.tick_to_price(1000) == pytest.approx(100.0)
    assert spec.qty_to_steps_floor(0.0019) == 1
    assert spec.steps_to_qty(2) == pytest.approx(0.002)
    assert spec.round_qty_down(0.0019) == pytest.approx(0.001)
    assert spec.notional(0.1, 1000) == pytest.approx(10.0)
    assert spec.is_valid_notional(0.1, 1000)
    assert spec.is_valid_qty(0.001)
    assert not spec.is_valid_qty(0.0019)

    with pytest.raises(ValueError):
        _spec(tick_size=0)
    with pytest.raises(ValueError):
        _spec(step_size=0)
    with pytest.raises(ValueError):
        _spec(min_qty=2.0, max_qty=1.0)


def test_l2_update_batch_atomicity():
    update = _update()
    batch = L2UpdateBatch(
        local_ts_us=100,
        min_ts_us=90,
        max_ts_us=95,
        updates=(update, _update(ts_us=95, is_snapshot=True)),
        is_snapshot_batch=True,
        batch_seq=0,
    )
    assert batch.local_ts_us == 100

    with pytest.raises(ValueError):
        L2UpdateBatch(100, 90, 95, (_update(local_ts_us=101),), False, 0)
    with pytest.raises(ValueError):
        L2UpdateBatch(100, 90, 95, (), False, 0)
    with pytest.raises(ValueError):
        L2UpdateBatch(100, 90, 95, (_update(is_snapshot=True),), False, 0)
    with pytest.raises(ValueError):
        L2UpdateBatch(100, 90, 95, (_update(ts_us=89),), False, 0)
    with pytest.raises(ValueError):
        L2UpdateBatch(100, 90, 95, (_update(ts_us=96),), False, 0)


def test_book_top_properties_and_cross_validation():
    top = BookTop(100, 1000, 1002, 1.5, 2.5)

    assert top.spread_ticks == 2
    assert top.mid_tick_x2 == 2002

    with pytest.raises(ValueError):
        BookTop(100, 1000, 1000, 1.0, 1.0)
    with pytest.raises(ValueError):
        BookTop(100, 1001, 1000, 1.0, 1.0)


def test_book_level_snapshot_ordering_and_lengths():
    snapshot = BookLevelSnapshot(100, (1000, 999), (1.0, 2.0), (1002, 1003), (1.5, 2.5))
    assert snapshot.bid_ticks == (1000, 999)

    with pytest.raises(ValueError):
        BookLevelSnapshot(100, (999, 1000), (1.0, 2.0), (1002,), (1.0,))
    with pytest.raises(ValueError):
        BookLevelSnapshot(100, (1000,), (1.0,), (1003, 1002), (1.0, 2.0))
    with pytest.raises(ValueError):
        BookLevelSnapshot(100, (1000,), (1.0,), (1000,), (1.0,))
    with pytest.raises(ValueError):
        BookLevelSnapshot(100, (1000,), (1.0, 2.0), (1002,), (1.0,))


def test_execution_event_ref_pointer_requirements():
    assert ExecutionEventRef(0, 100, ExecutionEventType.L2_BATCH, book_ptr=0).book_ptr == 0
    assert ExecutionEventRef(1, 101, "trade", trade_ptr=2).trade_ptr == 2
    assert ExecutionEventRef(2, 102, ExecutionEventType.DECISION, decision_ptr=3).decision_ptr == 3

    with pytest.raises(ValueError):
        ExecutionEventRef(0, 100, ExecutionEventType.L2_BATCH)
    with pytest.raises(ValueError):
        ExecutionEventRef(1, 101, ExecutionEventType.TRADE)
    with pytest.raises(ValueError):
        ExecutionEventRef(2, 102, ExecutionEventType.DECISION)
    with pytest.raises(ValueError):
        ExecutionEventRef(0, 100, ExecutionEventType.L2_BATCH, book_ptr=0, decision_ptr=1)
    with pytest.raises(ValueError):
        ExecutionEventRef(1, 101, ExecutionEventType.TRADE, trade_ptr=2, decision_ptr=1)
    with pytest.raises(ValueError):
        ExecutionEventRef(2, 102, ExecutionEventType.DECISION, decision_ptr=3, book_ptr=1)
    with pytest.raises(ValueError):
        ExecutionEventRef(2, 102, ExecutionEventType.DECISION, decision_ptr=3, trade_ptr=1)


def test_decision_ref_sequence_window_validation():
    ref = DecisionRef(0, 100, 5, 5, 1)
    assert ref.event_seq_end_next == 5

    with pytest.raises(ValueError):
        DecisionRef(0, 100, 5, 4, 1)


def test_linear_signal_bounds_and_negative_expected_return():
    signal = LinearSignal(
        p_no_move=0.2,
        p_move=0.8,
        p_up_move=0.3,
        p_down_move=0.5,
        signed_move_prob=-0.2,
        expected_up_bps=1.0,
        expected_down_bps=2.0,
        expected_return_bps=-1.0,
        expected_abs_move_bps=3.0,
        predicted_vol_bps=1.5,
        confidence=0.2,
    )
    assert signal.expected_return_bps == pytest.approx(-1.0)

    with pytest.raises(ValueError):
        LinearSignal(0.2, 1.1, 0.3, 0.5, -0.2, 1.0, 2.0, -1.0, 3.0, 1.5, 0.2)
    with pytest.raises(ValueError):
        LinearSignal(0.2, 0.8, 0.3, 0.5, -0.2, -1.0, 2.0, -1.0, 3.0, 1.5, 0.2)
    with pytest.raises(ValueError):
        LinearSignal(0.2, 0.9, 0.3, 0.5, -0.2, 1.0, 2.0, -1.0, 3.0, 1.5, 0.2)


def test_action_spec_validation():
    spec = ActionSpec(max_distance_ticks=10, max_order_qty=0.02)
    assert spec.max_distance_ticks == 10

    assert not hasattr(spec, "default_order_qty")

    with pytest.raises(ValueError):
        ActionSpec(max_distance_ticks=0)

    with pytest.raises(ValueError):
        ActionSpec(max_order_qty=0.0)

    with pytest.raises(ValueError):
        ActionSpec(allow_bid=False, allow_ask=False)


def test_action_spec_has_no_quote_geometry_defaults():
    spec = ActionSpec()
    assert not hasattr(spec, "default_order_qty")


def test_quote_intent_validation():
    quote = QuoteIntent(True, True, bid_price_tick=1000, ask_price_tick=1002, bid_qty=0.01, ask_qty=0.02)
    assert quote.bid_enabled and quote.ask_enabled

    one_sided = QuoteIntent(True, False, bid_price_tick=1000, bid_qty=0.01)
    assert one_sided.ask_price_tick == 0

    with pytest.raises(ValueError):
        QuoteIntent(True, False, bid_price_tick=1000, bid_qty=0.0)
    with pytest.raises(ValueError):
        QuoteIntent(True, True, bid_price_tick=1002, ask_price_tick=1001, bid_qty=0.01, ask_qty=0.02)


def test_active_order_and_fill_validation_and_properties():
    order = ActiveOrder(
        order_id=1,
        side=OrderSide.BUY,
        price_tick=1000,
        qty=0.02,
        remaining_qty=0.005,
        queue_ahead_qty=1.0,
        status=OrderStatus.PARTIALLY_FILLED,
        created_local_ts_us=100,
        created_event_seq=0,
        last_update_local_ts_us=110,
        last_update_event_seq=0,
    )
    assert order.filled_qty == pytest.approx(0.015)
    assert order.is_live
    assert not ActiveOrder(1, OrderSide.SELL, 1002, 0.02, 0.0, 0.0, OrderStatus.FILLED, 100, 110, 0, 0).is_live

    fill = Fill(1, "buy", 120, 0, 1000, 0.01, -0.001, FillReason.TRADE_AT_LEVEL)
    assert fill.fee == pytest.approx(-0.001)

    with pytest.raises(ValueError):
        ActiveOrder(1, OrderSide.BUY, 1000, 0.02, 0.03, 0.0, OrderStatus.ACTIVE, 100, 110, 0, 0)
    with pytest.raises(ValueError):
        ActiveOrder(1, OrderSide.BUY, 1000, 0.02, 0.01, 0.0, OrderStatus.FILLED, 100, 110, 0, 0)
    with pytest.raises(ValueError):
        ActiveOrder(1, OrderSide.BUY, 1000, 0.02, 0.02, 0.0, OrderStatus.PARTIALLY_FILLED, 100, 110, 0, 0)
    with pytest.raises(ValueError):
        ActiveOrder(1, OrderSide.BUY, 1000, 0.02, 0.0, 0.0, OrderStatus.ACTIVE, 100, 110, 0, 0)


def test_active_order_rejects_pending_cancel_keys():
    with pytest.raises(ValueError, match="ACTIVE order cannot have pending cancel"):
        ActiveOrder(
            order_id=1,
            side=OrderSide.BUY,
            price_tick=1000,
            qty=0.01,
            remaining_qty=0.01,
            queue_ahead_qty=0.0,
            status=OrderStatus.ACTIVE,
            created_local_ts_us=100,
            created_event_seq=0,
            last_update_local_ts_us=200,
            last_update_event_seq=0,
            cancel_requested_local_ts_us=200,
            cancel_requested_event_seq=0,
            cancel_effective_local_ts_us=300,
            cancel_effective_event_seq=0,
        )


def test_pending_cancel_order_accepts_cancel_keys():
    order = ActiveOrder(
        order_id=1,
        side=OrderSide.BUY,
        price_tick=1000,
        qty=0.01,
        remaining_qty=0.01,
        queue_ahead_qty=0.0,
        status=OrderStatus.PENDING_CANCEL,
        created_local_ts_us=100,
        created_event_seq=0,
        last_update_local_ts_us=200,
        last_update_event_seq=0,
        cancel_requested_local_ts_us=200,
        cancel_requested_event_seq=0,
        cancel_effective_local_ts_us=300,
        cancel_effective_event_seq=0,
    )

    assert order.is_live
    assert order.cancel_requested_key == EventKey(200, 0)
    assert order.cancel_effective_key == EventKey(300, 0)


def test_partially_filled_order_rejects_pending_cancel_keys():
    with pytest.raises(ValueError, match="PARTIALLY_FILLED order cannot have pending cancel"):
        ActiveOrder(
            order_id=1,
            side=OrderSide.BUY,
            price_tick=1000,
            qty=0.02,
            remaining_qty=0.01,
            queue_ahead_qty=0.0,
            status=OrderStatus.PARTIALLY_FILLED,
            created_local_ts_us=100,
            created_event_seq=0,
            last_update_local_ts_us=200,
            last_update_event_seq=0,
            cancel_requested_local_ts_us=200,
            cancel_requested_event_seq=0,
            cancel_effective_local_ts_us=300,
            cancel_effective_event_seq=0,
        )


def test_position_state_mark_to_market_requires_positive_mid():
    pos = PositionState(cash=10.0, inventory_qty=0.5)
    assert pos.mark_to_market(100.0) == pytest.approx(60.0)

    with pytest.raises(ValueError):
        pos.mark_to_market(0.0)

    with pytest.raises(ValueError):
        pos.mark_to_market(-1.0)


def test_reward_components_total_and_penalties():
    reward = RewardComponents(
        raw_equity_delta=10.0,
        inventory_penalty=1.0,
        drawdown_penalty=2.0,
        turnover_penalty=0.5,
        cancel_penalty=0.25,
        terminal_penalty=0.75,
    )
    assert reward.total_reward == pytest.approx(5.5)

    with pytest.raises(ValueError):
        RewardComponents(1.0, inventory_penalty=-0.1)


def test_execution_step_result_validation():
    fill = Fill(1, OrderSide.BUY, 120, 0, 1000, 0.01, -0.001, FillReason.TRADE_AT_LEVEL)
    result = ExecutionStepResult(
        reward=RewardComponents(1.0),
        position=PositionState(),
        fills=(fill,),
        done=False,
        truncated=False,
        info={"equity": 1.0},
    )
    assert result.fills == (fill,)
    assert result.info == {"equity": 1.0}

    with pytest.raises(ValueError):
        ExecutionStepResult(RewardComponents(1.0), PositionState(), fills=(object(),), done=False, truncated=False)

    with pytest.raises(ValueError):
        ExecutionStepResult(RewardComponents(1.0), PositionState(), fills=(), done=0, truncated=False)

    with pytest.raises(ValueError):
        ExecutionStepResult(
            RewardComponents(1.0),
            PositionState(),
            fills=(),
            done=False,
            truncated=False,
            info={"x": float("nan")},
        )


def test_execution_step_result_info_accepts_json_safe_scalars():
    result = ExecutionStepResult(
        reward=RewardComponents(raw_equity_delta=1.0),
        position=PositionState(),
        fills=(),
        done=False,
        truncated=False,
        info={
            "float_value": 1.5,
            "int_value": 2,
            "bool_value": True,
            "str_value": "reason",
            "none_value": None,
        },
    )

    assert result.info["float_value"] == pytest.approx(1.5)
    assert result.info["int_value"] == 2
    assert result.info["bool_value"] is True
    assert result.info["str_value"] == "reason"
    assert result.info["none_value"] is None


def test_execution_step_result_info_rejects_non_scalar_or_nonfinite_values():
    base = dict(
        reward=RewardComponents(raw_equity_delta=0.0),
        position=PositionState(),
        fills=(),
        done=False,
        truncated=False,
    )

    with pytest.raises(ValueError):
        ExecutionStepResult(**base, info={"bad": float("nan")})

    with pytest.raises(ValueError):
        ExecutionStepResult(**base, info={"bad": float("inf")})

    with pytest.raises(ValueError):
        ExecutionStepResult(**base, info={"bad": ["nested"]})

    with pytest.raises(ValueError):
        ExecutionStepResult(**base, info={1: "bad_key"})


def test_execution_tape_manifest_required_metadata():
    spec = _spec()
    manifest = ExecutionTapeManifest(
        schema="1",
        tape_format=ExecutionTapeFormat.L2_TRADES_ARRAYS,
        exchange="binance-futures",
        symbol="BTCUSDT",
        symbol_spec=spec,
        symbol_rules=_rules(),
        source_data_types=(TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES),
        array_names=("events", "books", "trades"),
        num_events=10,
        num_l2_batches=5,
        num_trades=3,
        num_decisions=2,
        start_local_ts_us=100,
        end_local_ts_us=200,
    )
    assert manifest.num_events == 10

    kwargs = {
        "schema": "1",
        "tape_format": ExecutionTapeFormat.L2_TRADES_ARRAYS,
        "exchange": "binance-futures",
        "symbol": "BTCUSDT",
        "symbol_spec": spec,
        "symbol_rules": _rules(),
        "source_data_types": (TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES),
        "array_names": ("events",),
        "num_events": 2,
        "num_l2_batches": 1,
        "num_trades": 1,
        "num_decisions": 0,
        "start_local_ts_us": 100,
        "end_local_ts_us": 200,
    }
    with pytest.raises(ValueError):
        ExecutionTapeManifest(**{**kwargs, "array_names": "events"})
    with pytest.raises(ValueError):
        ExecutionTapeManifest(**{**kwargs, "source_data_types": (TardisDataType.TRADES,)})
    with pytest.raises(ValueError):
        ExecutionTapeManifest(**{**kwargs, "source_data_types": (TardisDataType.INCREMENTAL_BOOK_L2,)})
    with pytest.raises(ValueError):
        ExecutionTapeManifest(
            **{
                **kwargs,
                "source_data_types": (
                    TardisDataType.INCREMENTAL_BOOK_L2,
                    TardisDataType.TRADES,
                    TardisDataType.TRADES,
                ),
            }
        )
    with pytest.raises(ValueError):
        ExecutionTapeManifest(**{**kwargs, "array_names": ("events", "events")})
    with pytest.raises(ValueError):
        ExecutionTapeManifest(**{**kwargs, "end_local_ts_us": 100})
    with pytest.raises(ValueError):
        ExecutionTapeManifest(**{**kwargs, "symbol_spec": _spec(symbol="ETHUSDT")})


def test_trade_print_allows_unknown_aggressor_side():
    from mmrt.execution.contracts import TradePrint

    trade = TradePrint(100, 90, AggressorSide.UNKNOWN, 1000, 0.1)
    assert trade.side == AggressorSide.UNKNOWN
    assert math.isfinite(trade.amount)
