import math
from pathlib import Path

import pytest

from mmrt.execution.contracts import ActionSpec, BookTop, QuoteIntent, SymbolSpec
from mmrt.execution.quote_geometry import (
    QuoteAction,
    QuoteGeometryConfig,
    QuoteGeometryResult,
    continuous_action_to_quote,
    raw_bid_price_to_tick,
    raw_ask_price_to_tick,
    raw_size_to_qty,
)


def _spec() -> SymbolSpec:
    return SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=0.0,
    )


def _top() -> BookTop:
    return BookTop(
        local_ts_us=100,
        best_bid_tick=1000,
        best_ask_tick=1002,
        best_bid_size=1.0,
        best_ask_size=1.0,
    )


def _action(**kwargs) -> QuoteAction:
    base = dict(
        bid_enabled=True,
        ask_enabled=True,
        bid_price_raw=0.0,
        ask_price_raw=0.0,
        bid_size_raw=0.0,
        ask_size_raw=0.0,
    )
    base.update(kwargs)
    return QuoteAction(**base)


def test_raw_price_to_tick_is_bounded_and_monotonic():
    top = _top()
    spec = ActionSpec(max_distance_ticks=20)
    config = QuoteGeometryConfig()
    bids = [raw_bid_price_to_tick(x, book_top=top, action_spec=spec, config=config) for x in [-10.0, -1.0, 0.0, 1.0, 10.0]]
    asks = [raw_ask_price_to_tick(x, book_top=top, action_spec=spec, config=config) for x in [-10.0, -1.0, 0.0, 1.0, 10.0]]
    assert bids == sorted(bids)
    assert asks == sorted(asks, reverse=True)
    assert all(v < top.best_ask_tick for v in bids)
    assert all(v > top.best_bid_tick for v in asks)


def test_raw_size_to_qty_uses_step_grid_and_bounds():
    spec = _spec()
    qty = raw_size_to_qty(
        0.0,
        symbol_spec=spec,
        max_order_qty=0.01,
        default_order_qty=0.001,
    )

    assert qty >= spec.min_qty
    assert qty <= 0.01
    assert spec.is_valid_qty(qty)
    assert raw_size_to_qty(0.0, symbol_spec=spec, max_order_qty=0.0005, default_order_qty=0.0005) == 0.0


def test_continuous_action_builds_two_sided_passive_quote():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(max_distance_ticks=20, max_order_qty=0.01),
    )

    quote = result.quote
    assert isinstance(result, QuoteGeometryResult)
    assert quote.bid_enabled
    assert quote.ask_enabled
    assert quote.bid_price_tick < quote.ask_price_tick
    assert quote.bid_price_tick < _top().best_ask_tick
    assert quote.ask_price_tick > _top().best_bid_tick
    assert quote.bid_qty > 0
    assert quote.ask_qty > 0


def test_quote_geometry_result_uses_offset_fields_only():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(max_distance_ticks=20, max_order_qty=0.01),
    )

    assert hasattr(result, "bid_offset_ticks")
    assert hasattr(result, "ask_offset_ticks")
    assert not hasattr(result, "bid_distance_ticks")
    assert not hasattr(result, "ask_distance_ticks")


def test_action_enable_flags_disable_sides():
    result = continuous_action_to_quote(
        action=_action(bid_enabled=False, ask_enabled=False),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(),
    )

    assert result.quote == QuoteIntent(False, False)
    assert result.bid_disabled_reason == "disabled_by_action"
    assert result.ask_disabled_reason == "disabled_by_action"


def test_action_spec_side_permissions_disable_quotes():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(allow_bid=False, allow_ask=True),
    )

    assert not result.quote.bid_enabled
    assert result.quote.ask_enabled
    assert result.bid_disabled_reason == "disabled_by_action_spec"

    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(allow_bid=True, allow_ask=False),
    )

    assert result.quote.bid_enabled
    assert not result.quote.ask_enabled
    assert result.ask_disabled_reason == "disabled_by_action_spec"


def test_missing_book_top_disables_both_sides():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=None,
        symbol_spec=_spec(),
        action_spec=ActionSpec(),
    )

    assert result.quote == QuoteIntent(False, False)
    assert result.bid_disabled_reason == "missing_book_top"
    assert result.ask_disabled_reason == "missing_book_top"


def test_min_notional_disables_side():
    spec = SymbolSpec(
        exchange="binance-futures",
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        min_qty=0.001,
        max_qty=100.0,
        min_notional=1_000_000.0,
    )

    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=spec,
        action_spec=ActionSpec(max_order_qty=0.01),
    )

    assert not result.quote.bid_enabled
    assert not result.quote.ask_enabled
    assert result.bid_disabled_reason == "notional_below_min"
    assert result.ask_disabled_reason == "notional_below_min"


def test_inventory_limit_disables_risk_increasing_side():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(max_order_qty=0.01),
        config=QuoteGeometryConfig(max_inventory_abs_qty=1.0),
        inventory_qty=1.0,
    )

    assert not result.quote.bid_enabled
    assert result.quote.ask_enabled
    assert result.bid_disabled_reason == "inventory_limit"

    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(max_order_qty=0.01),
        config=QuoteGeometryConfig(max_inventory_abs_qty=1.0),
        inventory_qty=-1.0,
    )

    assert result.quote.bid_enabled
    assert not result.quote.ask_enabled
    assert result.ask_disabled_reason == "inventory_limit"


def test_position_notional_limit_disables_projected_breach():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(max_order_qty=1.0),
        config=QuoteGeometryConfig(max_position_notional=0.01),
        inventory_qty=0.0,
    )

    assert not result.quote.bid_enabled
    assert not result.quote.ask_enabled
    assert result.bid_disabled_reason == "position_notional_limit"
    assert result.ask_disabled_reason == "position_notional_limit"


def test_quantity_below_minimum_disables_side():
    result = continuous_action_to_quote(
        action=_action(),
        book_top=_top(),
        symbol_spec=_spec(),
        action_spec=ActionSpec(max_order_qty=0.0005),
    )

    assert not result.quote.bid_enabled
    assert not result.quote.ask_enabled
    assert result.bid_disabled_reason == "qty_below_min"
    assert result.ask_disabled_reason == "qty_below_min"


def test_config_validation():
    with pytest.raises(ValueError):
        QuoteGeometryConfig(post_only_gap_ticks=0)

    with pytest.raises(ValueError):
        QuoteGeometryConfig(default_order_qty=0.0)

    with pytest.raises(ValueError):
        QuoteGeometryConfig(max_inventory_abs_qty=0.0)


def test_action_validation_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        QuoteAction(
            bid_enabled=math.nan,
            ask_enabled=True,
            bid_price_raw=0.0,
            ask_price_raw=0.0,
            bid_size_raw=0.0,
            ask_size_raw=0.0,
        )

    with pytest.raises(ValueError):
        QuoteAction(
            bid_enabled=math.inf,
            ask_enabled=True,
            bid_price_raw=0.0,
            ask_price_raw=0.0,
            bid_size_raw=0.0,
            ask_size_raw=0.0,
        )


def test_quote_geometry_has_no_forbidden_imports():
    source = Path("mmrt/execution/quote_geometry.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import numpy" not in source
    assert "import sklearn" not in source
