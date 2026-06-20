from pathlib import Path

import numpy as np
import pytest

from mmrt.execution.contracts import (
    ActiveOrder,
    BookTop,
    Fill,
    FillReason,
    LinearSignal,
    OrderSide,
    OrderStatus,
    PositionState,
    SymbolSpec,
)
from mmrt.execution.linear_signal import build_gated_linear_signal
from mmrt.execution.obs_schema import (
    DEFAULT_OBSERVATION_FIELDS,
    DEFAULT_ADVERSE_CANDIDATE_NAMES,
    CONTROL_GROUP,
    CONTROL_FIELDS,
    MARKET_FIELDS,
    LINEAR_SIGNAL_FIELDS,
    POSITION_FIELDS,
    ORDERS_FIELDS,
    FILLS_FIELDS,
    TIME_FIELDS,
    ObservationSchema,
    default_observation_schema,
    executable_edge_fields,
    execution_observation_schema,
    is_removed_observation_field,
    observation_field_groups,
    validate_observation_vector,
)
from mmrt.execution.obs_builder import (
    ObservationBuilder,
    ObservationBuilderConfig,
    ObservationContext,
    ObservationInput,
    build_observation,
)


def _spec() -> SymbolSpec:
    return SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=5.0,
    )


def _top(local_ts_us: int = 1_000_000) -> BookTop:
    return BookTop(
        local_ts_us=local_ts_us,
        best_bid_tick=1000,
        best_ask_tick=1002,
        best_bid_size=2.0,
        best_ask_size=1.0,
    )


def _signal() -> LinearSignal:
    return build_gated_linear_signal(
        p_no_move=0.2,
        p_up=0.7,
        magnitude_up=np.log1p(10.0),
        magnitude_down=np.log1p(5.0),
    )


def _order(
    *,
    order_id: int,
    side: OrderSide,
    price_tick: int,
    qty: float = 1.0,
    remaining_qty: float = 1.0,
    queue_ahead_qty: float = 2.0,
    created_local_ts_us: int = 900_000,
    status: OrderStatus = OrderStatus.ACTIVE,
) -> ActiveOrder:
    return ActiveOrder(
        order_id=order_id,
        side=side,
        price_tick=price_tick,
        qty=qty,
        remaining_qty=remaining_qty,
        queue_ahead_qty=queue_ahead_qty,
        status=status,
        created_local_ts_us=created_local_ts_us,
        created_event_seq=0,
        last_update_local_ts_us=created_local_ts_us,
        last_update_event_seq=0,
    )


def _fill(
    *,
    side: OrderSide,
    price_tick: int = 1001,
    qty: float = 0.5,
    fee: float = 0.01,
    local_ts_us: int = 990_000,
) -> Fill:
    return Fill(
        order_id=1,
        side=side,
        local_ts_us=local_ts_us,
        event_seq=0,
        price_tick=price_tick,
        qty=qty,
        fee=fee,
        reason=FillReason.TRADE_AT_LEVEL,
        queue_ahead_before=1.0,
        queue_ahead_after=0.0,
    )


def test_default_schema_fields_and_groups():
    schema = default_observation_schema()

    assert schema.dim == 73
    assert schema.field_names == DEFAULT_OBSERVATION_FIELDS
    assert schema.index("spread_ticks") == 0
    assert schema.index(CONTROL_FIELDS[0]) == len(MARKET_FIELDS)
    assert schema.index("linear_p_no_move") == len(MARKET_FIELDS) + len(CONTROL_FIELDS)
    assert len(schema.field_names) == len(set(schema.field_names))
    assert schema.has_field("linear_p_no_move")
    assert schema.has_field("linear_p_move")
    assert schema.has_field("linear_p_up_move")
    assert schema.has_field("linear_p_down_move")
    assert schema.has_field("linear_signed_move_prob")
    assert schema.has_field("linear_expected_up_bps")
    assert schema.has_field("linear_expected_down_bps")
    assert schema.has_field("linear_expected_return_bps")
    assert schema.has_field("linear_expected_abs_move_bps")
    assert schema.has_field("linear_predicted_vol_bps")
    assert schema.has_field("linear_confidence")
    assert not schema.has_field("linear_" + "p_up")
    assert not schema.has_field("linear_" + "mag_up_bps")
    assert not schema.has_field("linear_" + "mag_down_bps")
    retired_inventory_pnl_name = "_".join(("unrealized", "inventory", "pnl"))
    assert schema.has_field("inventory_abs_notional")
    assert schema.has_field("inventory_order_units")
    assert not schema.has_field("inventory_notional_bps")
    assert not schema.has_field(retired_inventory_pnl_name)
    assert not schema.has_field("missing")

    groups = observation_field_groups()
    assert groups["market"] == MARKET_FIELDS
    assert groups[CONTROL_GROUP] == CONTROL_FIELDS
    assert groups["linear_signal"] == LINEAR_SIGNAL_FIELDS
    assert groups["position"] == POSITION_FIELDS
    assert groups["orders"] == ORDERS_FIELDS
    assert groups["fills"] == FILLS_FIELDS
    assert groups["time"] == TIME_FIELDS


def test_observation_schema_validation():
    with pytest.raises(ValueError):
        ObservationSchema(field_names=())

    with pytest.raises(ValueError):
        ObservationSchema(field_names=("x", "x"))

    with pytest.raises(ValueError):
        ObservationSchema(field_names=("x", ""), dtype="float32")

    with pytest.raises(ValueError):
        ObservationSchema(dtype="int64")

    with pytest.raises(ValueError, match="inventory_notional_bps was removed"):
        ObservationSchema(field_names=("cash", "inventory_notional_bps"))

    with pytest.raises(ValueError, match="conditional-fill edge is numerically unstable"):
        ObservationSchema(field_names=("edge_bid_touch_cond_fill_bps",))

    assert is_removed_observation_field("inventory_notional_bps")
    assert is_removed_observation_field("edge_ask_inside_1_cond_fill_bps")
    assert not is_removed_observation_field("edge_ask_inside_1_attempt_bps")


def test_observation_schema_round_trip_dict():
    schema = ObservationSchema(field_names=("a", "b"), dtype="float64")
    restored = ObservationSchema.from_dict(schema.as_dict())

    assert restored == schema
    assert restored.empty().shape == (2,)
    assert restored.empty().dtype == np.float64


def test_validate_observation_vector():
    schema = ObservationSchema(field_names=("a", "b"), dtype="float32")
    obs = np.zeros(2, dtype=np.float32)

    assert validate_observation_vector(obs, schema=schema) is obs

    with pytest.raises(ValueError):
        validate_observation_vector(np.zeros(3, dtype=np.float32), schema=schema)

    with pytest.raises(ValueError):
        validate_observation_vector(np.zeros(2, dtype=np.float64), schema=schema)

    bad = np.array([0.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError):
        validate_observation_vector(bad, schema=schema)


def test_build_market_neutral_observation():
    schema = default_observation_schema()
    builder = ObservationBuilder(schema=schema)
    inputs = ObservationInput(
        symbol_spec=_spec(),
        book_top=_top(),
        bid_depth=10,
        ask_depth=12,
        linear_signal=_signal(),
    )

    obs = builder.build(inputs)

    assert obs.shape == (schema.dim,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    assert obs[schema.index("spread_ticks")] == pytest.approx(2.0)
    assert obs[schema.index("mid_price")] == pytest.approx(100.1)
    assert obs[schema.index("bid_depth_count")] == pytest.approx(10.0)
    assert obs[schema.index("ask_depth_count")] == pytest.approx(12.0)

    assert obs[schema.index("spread_bps")] == pytest.approx(0.2 / 100.1 * 10_000.0)
    assert obs[schema.index("top_imbalance")] == pytest.approx(1.0 / 3.0)
    microprice = (100.0 * 1.0 + 100.2 * 2.0) / 3.0
    assert obs[schema.index("microprice_bps")] == pytest.approx((microprice - 100.1) / 100.1 * 10_000.0)

    assert obs[schema.index("linear_p_no_move")] == pytest.approx(0.2)
    assert obs[schema.index("linear_p_move")] == pytest.approx(0.8)
    assert obs[schema.index("linear_expected_abs_move_bps")] == pytest.approx(6.8)


def test_build_observation_uses_linear_signal():
    schema = default_observation_schema()
    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
        ),
        schema=schema,
    )

    assert obs[schema.index("linear_p_no_move")] == pytest.approx(0.2)
    assert obs[schema.index("linear_p_move")] == pytest.approx(0.8)
    assert obs[schema.index("linear_p_up_move")] == pytest.approx(0.56)
    assert obs[schema.index("linear_p_down_move")] == pytest.approx(0.24)
    assert obs[schema.index("linear_signed_move_prob")] == pytest.approx(0.32)
    assert obs[schema.index("linear_expected_up_bps")] == pytest.approx(5.6)
    assert obs[schema.index("linear_expected_down_bps")] == pytest.approx(1.2)
    assert obs[schema.index("linear_expected_return_bps")] == pytest.approx(4.4)
    assert obs[schema.index("linear_expected_abs_move_bps")] == pytest.approx(6.8)
    assert obs[schema.index("linear_predicted_vol_bps")] == pytest.approx(np.sqrt(62.0 - 4.4 * 4.4))
    assert obs[schema.index("linear_confidence")] == pytest.approx(0.32)


def test_observation_builder_fills_control_feature_map_and_ignores_unknown_names():
    schema = default_observation_schema()
    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            control_features={
                "depth_bid_qty_5": 7.0,
                "flow_imbalance_ratio_200ms": -0.25,
                "unknown_control_feature": 123.0,
            },
        ),
        schema=schema,
    )

    assert obs[schema.index("depth_bid_qty_5")] == pytest.approx(7.0)
    assert obs[schema.index("flow_imbalance_ratio_200ms")] == pytest.approx(-0.25)
    assert "unknown_control_feature" not in schema.field_names


def test_build_observation_position_fields():
    schema = default_observation_schema()
    position = PositionState(cash=10.0, inventory_qty=-2.0, fees_paid=0.25)

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=BookTop(
                local_ts_us=1_000_000,
                best_bid_tick=999,
                best_ask_tick=1001,
                best_bid_size=2.0,
                best_ask_size=1.0,
            ),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            position=position,
        ),
        schema=schema,
    )

    mid = 100.0
    inventory_notional = -2.0 * mid
    equity = 10.0 + inventory_notional

    assert obs[schema.index("cash")] == pytest.approx(10.0)
    assert obs[schema.index("inventory_qty")] == pytest.approx(-2.0)
    assert obs[schema.index("inventory_notional")] == pytest.approx(inventory_notional)
    assert obs[schema.index("inventory_order_units")] == pytest.approx(-2.0 / 0.003)
    assert obs[schema.index("equity")] == pytest.approx(equity)
    assert obs[schema.index("inventory_abs_notional")] == pytest.approx(abs(inventory_notional))
    assert obs[schema.index("fees_paid")] == pytest.approx(0.25)


def test_inventory_order_units_uses_configured_reference_qty():
    schema = default_observation_schema()
    builder = ObservationBuilder(
        schema=schema,
        config=ObservationBuilderConfig(inventory_qty_reference=0.003),
    )
    obs = builder.build(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            position=PositionState(inventory_qty=0.006),
        )
    )

    assert obs[schema.index("inventory_order_units")] == pytest.approx(2.0)


def test_near_zero_equity_does_not_create_inventory_bps_observation():
    schema = default_observation_schema()
    position = PositionState(cash=-0.6006, inventory_qty=0.006)

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            position=position,
        ),
        schema=schema,
    )

    assert "inventory_notional_bps" not in schema.field_names
    assert np.isfinite(obs).all()
    assert abs(obs[schema.index("equity")]) < 1e-9
    assert obs[schema.index("inventory_order_units")] == pytest.approx(2.0)


def test_build_observation_live_order_fields():
    schema = default_observation_schema()
    bid = _order(order_id=1, side=OrderSide.BUY, price_tick=999, qty=1.0, remaining_qty=0.8, queue_ahead_qty=3.0)
    ask = _order(order_id=2, side=OrderSide.SELL, price_tick=1004, qty=2.0, remaining_qty=1.5, queue_ahead_qty=4.0)

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(local_ts_us=1_000_000),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            live_orders=(bid, ask),
        ),
        schema=schema,
    )

    assert obs[schema.index("has_live_bid")] == 1.0
    assert obs[schema.index("has_live_ask")] == 1.0
    assert obs[schema.index("bid_distance_ticks")] == pytest.approx(1.0)
    assert obs[schema.index("ask_distance_ticks")] == pytest.approx(2.0)
    assert obs[schema.index("bid_distance_bps")] == pytest.approx(0.1 / 100.1 * 10_000.0)
    assert obs[schema.index("ask_distance_bps")] == pytest.approx(0.2 / 100.1 * 10_000.0)
    assert obs[schema.index("bid_qty")] == pytest.approx(1.0)
    assert obs[schema.index("ask_qty")] == pytest.approx(2.0)
    assert obs[schema.index("bid_remaining_qty")] == pytest.approx(0.8)
    assert obs[schema.index("ask_remaining_qty")] == pytest.approx(1.5)
    assert obs[schema.index("bid_queue_ahead_qty")] == pytest.approx(3.0)
    assert obs[schema.index("ask_queue_ahead_qty")] == pytest.approx(4.0)
    assert obs[schema.index("bid_age_ms")] == pytest.approx(100.0)


def test_non_live_orders_ignored():
    schema = default_observation_schema()
    cancelled = _order(
        order_id=1,
        side=OrderSide.BUY,
        price_tick=999,
        status=OrderStatus.CANCELLED,
    )

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            live_orders=(cancelled,),
        ),
        schema=schema,
    )

    assert obs[schema.index("has_live_bid")] == 0.0
    assert obs[schema.index("bid_qty")] == 0.0


def test_duplicate_live_side_rejected():
    bid1 = _order(order_id=1, side=OrderSide.BUY, price_tick=999)
    bid2 = _order(order_id=2, side=OrderSide.BUY, price_tick=998)

    with pytest.raises(ValueError, match="at most one live order per side"):
        build_observation(
            ObservationInput(
                symbol_spec=_spec(),
                book_top=_top(),
                bid_depth=1,
                ask_depth=1,
                linear_signal=_signal(),
                live_orders=(bid1, bid2),
            )
        )


def test_build_observation_fill_fields():
    schema = default_observation_schema()
    buy = _fill(side=OrderSide.BUY, qty=0.5, price_tick=1000, fee=0.01, local_ts_us=990_000)
    sell = _fill(side=OrderSide.SELL, qty=0.25, price_tick=1002, fee=-0.02, local_ts_us=995_000)

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(local_ts_us=1_000_000),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            recent_fills=(buy, sell),
        ),
        schema=schema,
    )

    assert obs[schema.index("last_fill_side")] == -1.0
    assert obs[schema.index("last_fill_qty")] == pytest.approx(0.25)
    assert obs[schema.index("last_fill_fee")] == pytest.approx(-0.02)
    assert obs[schema.index("last_fill_age_ms")] == pytest.approx(5.0)
    assert obs[schema.index("step_fill_count")] == pytest.approx(2.0)
    assert obs[schema.index("step_buy_fill_qty")] == pytest.approx(0.5)
    assert obs[schema.index("step_sell_fill_qty")] == pytest.approx(0.25)

    expected_notional = 100.0 * 0.5 + 100.2 * 0.25
    assert obs[schema.index("step_fill_notional")] == pytest.approx(expected_notional)
    assert obs[schema.index("last_fill_notional")] == pytest.approx(100.2 * 0.25)


def test_future_fill_rejected():
    future = _fill(side=OrderSide.BUY, local_ts_us=2_000_000)

    with pytest.raises(ValueError):
        build_observation(
            ObservationInput(
                symbol_spec=_spec(),
                book_top=_top(local_ts_us=1_000_000),
                bid_depth=1,
                ask_depth=1,
                linear_signal=_signal(),
                recent_fills=(future,),
            )
        )


def test_time_context_fields():
    schema = default_observation_schema()
    context = ObservationContext(
        current_local_ts_us=2_000_000,
        episode_start_local_ts_us=1_000_000,
        current_event_index=5,
        total_events=11,
        previous_event_local_ts_us=1_990_000,
    )

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(local_ts_us=2_000_000),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            context=context,
        ),
        schema=schema,
    )

    assert obs[schema.index("local_time_since_start_s")] == pytest.approx(1.0)
    assert "event_progress" not in schema.field_names
    assert obs[schema.index("time_since_last_event_ms")] == pytest.approx(10.0)


def test_custom_schema_subset():
    schema = ObservationSchema(field_names=("spread_ticks", "linear_p_up_move", "cash"), dtype="float32")

    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            position=PositionState(cash=123.0),
        ),
        schema=schema,
    )

    assert obs.shape == (3,)
    assert obs[0] == pytest.approx(2.0)
    assert obs[1] == pytest.approx(0.56)
    assert obs[2] == pytest.approx(123.0)


def test_out_buffer_reused_and_zeroed():
    schema = default_observation_schema()
    builder = ObservationBuilder(schema=schema)
    out = np.full(schema.dim, 999.0, dtype=np.float32)

    result = builder.build(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
        ),
        out=out,
    )

    assert result is out
    assert np.isfinite(out).all()
    assert out[schema.index("cash")] == 0.0

    bad_out = np.zeros(schema.dim, dtype=np.float64)
    with pytest.raises(ValueError):
        builder.build(
            ObservationInput(
                symbol_spec=_spec(),
                book_top=_top(),
                bid_depth=1,
                ask_depth=1,
                linear_signal=_signal(),
            ),
            out=bad_out,
        )


def test_max_abs_observation_clipping():
    schema = ObservationSchema(field_names=("cash",), dtype="float32")
    builder = ObservationBuilder(
        schema=schema,
        config=ObservationBuilderConfig(max_abs_observation=10.0),
    )

    obs = builder.build(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            position=PositionState(cash=123.0),
        )
    )

    assert obs[0] == pytest.approx(10.0)


def test_invalid_observation_inputs_rejected():
    with pytest.raises(ValueError):
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=-1,
            ask_depth=1,
            linear_signal=_signal(),
        )

    with pytest.raises(TypeError):
        ObservationInput(symbol_spec=_spec(), book_top=_top(), bid_depth=1, ask_depth=1)

    with pytest.raises(ValueError):
        ObservationInput(symbol_spec=_spec(), book_top=_top(), bid_depth=1, ask_depth=1, linear_signal=None)

    with pytest.raises(ValueError):
        ObservationContext(
            current_local_ts_us=1_000_000,
            episode_start_local_ts_us=2_000_000,
        )

    with pytest.raises(ValueError, match="inventory_qty_reference"):
        ObservationBuilderConfig(inventory_qty_reference=0.0)

    with pytest.raises(ValueError):
        ObservationBuilderConfig(max_abs_observation=-1.0)


def test_obs_modules_have_no_forbidden_imports():
    schema_source = Path("mmrt/execution/obs_schema.py").read_text(encoding="utf-8")
    builder_source = Path("mmrt/execution/obs_builder.py").read_text(encoding="utf-8")

    for source in (schema_source, builder_source):
        assert "import torch" not in source
        assert "import pandas" not in source
        assert "import polars" not in source
        assert "import sklearn" not in source
        assert "import pyarrow" not in source
        assert "mmrt.linear.models" not in source
        assert "mmrt.storage" not in source
        assert "mmrt.rl" not in source

def test_execution_observation_schema_includes_adverse_and_edge_fields():
    schema = execution_observation_schema(include_adverse_selection=True, include_executable_edge=True)
    assert len(default_observation_schema().field_names) == len(DEFAULT_OBSERVATION_FIELDS)
    assert schema.dim == 109
    assert schema.has_field("adverse_bid_touch_fill_prob")
    assert schema.has_field("edge_bid_touch_attempt_bps")
    assert not any(name.endswith("_cond_fill_bps") for name in schema.field_names)

    edge_fields = executable_edge_fields()
    assert not any(name.endswith("_cond_fill_bps") for name in edge_fields)
    for candidate in DEFAULT_ADVERSE_CANDIDATE_NAMES:
        for side in ("bid", "ask"):
            prefix = f"edge_{side}_{candidate}"
            assert f"{prefix}_attempt_bps" in edge_fields
            assert f"{prefix}_allowed" in edge_fields
            assert f"{prefix}_valid" in edge_fields
            assert f"{prefix}_cond_fill_bps" not in edge_fields


def test_observation_builder_fills_adverse_and_edge_feature_maps():
    schema = execution_observation_schema(include_adverse_selection=True, include_executable_edge=True)
    obs = build_observation(
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            adverse_features={"adverse_bid_touch_fill_prob": 0.7},
            executable_edge_features={"edge_bid_touch_attempt_bps": 1.25},
        ),
        schema=schema,
    )
    assert obs[schema.index("adverse_bid_touch_fill_prob")] == pytest.approx(0.7)
    assert obs[schema.index("edge_bid_touch_attempt_bps")] == pytest.approx(1.25)


def test_observation_builder_rejects_nonfinite_adverse_feature():
    with pytest.raises(ValueError, match="adverse_features"):
        ObservationInput(
            symbol_spec=_spec(),
            book_top=_top(),
            bid_depth=1,
            ask_depth=1,
            linear_signal=_signal(),
            adverse_features={"adverse_bid_touch_fill_prob": float("nan")},
        )
