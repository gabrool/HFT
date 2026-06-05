from pathlib import Path

import pytest

from mmrt.execution.contracts import (
    BookTop,
    Fill,
    FillReason,
    OrderSide,
    PositionState,
    RewardComponents,
    SymbolSpec,
)
from mmrt.execution.reward import (
    RewardConfig,
    RewardStepResult,
    apply_fill_to_position,
    apply_fills_to_position,
    compute_reward_components,
    compute_reward_step,
    fill_notional,
    fills_turnover_notional,
    mark_price_from_book_top,
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


def _top(*, bid: int = 1000, ask: int = 1002) -> BookTop:
    return BookTop(
        local_ts_us=100,
        best_bid_tick=bid,
        best_ask_tick=ask,
        best_bid_size=1.0,
        best_ask_size=1.0,
    )


def _fill(
    *,
    side: OrderSide = OrderSide.BUY,
    price_tick: int = 1000,
    qty: float = 1.0,
    fee: float = 0.0,
) -> Fill:
    return Fill(
        order_id=1,
        side=side,
        local_ts_us=200,
        price_tick=price_tick,
        qty=qty,
        fee=fee,
        reason=FillReason.TRADE_AT_LEVEL,
        queue_ahead_before=0.0,
        queue_ahead_after=0.0,
    )


def test_reward_config_validation():
    assert RewardConfig().inventory_penalty_bps == 0.0

    with pytest.raises(ValueError):
        RewardConfig(inventory_penalty_bps=-1.0)

    with pytest.raises(ValueError):
        RewardConfig(turnover_penalty_bps=-1.0)

    with pytest.raises(ValueError):
        RewardConfig(cancel_penalty=-1.0)

    with pytest.raises(ValueError):
        RewardConfig(drawdown_penalty_rate=-1.0)

    with pytest.raises(ValueError):
        RewardConfig(terminal_inventory_penalty_bps=-1.0)

    with pytest.raises(ValueError):
        RewardConfig(inventory_penalty_bps=float("nan"))

    with pytest.raises(ValueError):
        RewardConfig(turnover_penalty_bps=float("inf"))


def test_mark_price_from_book_top():
    spec = _spec()
    mark = mark_price_from_book_top(_top(bid=1000, ask=1002), spec)

    assert mark == pytest.approx(100.1)


def test_fill_notional():
    spec = _spec()
    fill = _fill(price_tick=1000, qty=0.5)

    assert fill_notional(fill, spec) == pytest.approx(100.0 * 0.5 * spec.contract_size)


def test_fills_turnover_notional():
    spec = _spec()
    fills = [
        _fill(price_tick=1000, qty=0.5),
        _fill(side=OrderSide.SELL, price_tick=1002, qty=0.25),
    ]

    expected = 100.0 * 0.5 * spec.contract_size + 100.2 * 0.25 * spec.contract_size
    assert fills_turnover_notional(fills, symbol_spec=spec) == pytest.approx(expected)
    assert fills_turnover_notional([], symbol_spec=spec) == 0.0


def test_apply_buy_fill_to_position():
    spec = _spec()
    position = PositionState()
    fill = _fill(side=OrderSide.BUY, price_tick=1000, qty=0.5, fee=0.01)

    out = apply_fill_to_position(position, fill, symbol_spec=spec)

    assert out.inventory_qty == pytest.approx(0.5)
    assert out.cash == pytest.approx(-(100.0 * 0.5) - 0.01)
    assert out.fees_paid == pytest.approx(0.01)
    assert out.realized_pnl == position.realized_pnl
    assert position == PositionState()


def test_apply_sell_fill_to_position():
    spec = _spec()
    position = PositionState()
    fill = _fill(side=OrderSide.SELL, price_tick=1002, qty=0.25, fee=0.02)

    out = apply_fill_to_position(position, fill, symbol_spec=spec)

    assert out.inventory_qty == pytest.approx(-0.25)
    assert out.cash == pytest.approx(100.2 * 0.25 - 0.02)
    assert out.fees_paid == pytest.approx(0.02)


def test_negative_fee_rebate_increases_cash():
    spec = _spec()
    fill = _fill(side=OrderSide.BUY, price_tick=1000, qty=1.0, fee=-0.05)

    out = apply_fill_to_position(PositionState(), fill, symbol_spec=spec)

    assert out.cash == pytest.approx(-100.0 + 0.05)
    assert out.fees_paid == pytest.approx(-0.05)


def test_apply_fills_to_position_multiple():
    spec = _spec()
    fills = [
        _fill(side=OrderSide.BUY, price_tick=1000, qty=1.0, fee=0.01),
        _fill(side=OrderSide.SELL, price_tick=1002, qty=0.4, fee=0.02),
    ]

    out = apply_fills_to_position(PositionState(), fills, symbol_spec=spec)

    expected_cash = -100.0 * 1.0 - 0.01 + 100.2 * 0.4 - 0.02
    assert out.inventory_qty == pytest.approx(0.6)
    assert out.cash == pytest.approx(expected_cash)
    assert out.fees_paid == pytest.approx(0.03)


def test_compute_reward_step_raw_equity_delta_from_mark_move():
    spec = _spec()
    previous_position = PositionState(cash=0.0, inventory_qty=1.0)

    result = compute_reward_step(
        previous_position=previous_position,
        fills=[],
        previous_book_top=_top(bid=1000, ask=1002),
        current_book_top=_top(bid=1010, ask=1012),
        symbol_spec=spec,
    )

    assert isinstance(result, RewardStepResult)
    assert result.previous_equity == pytest.approx(100.1)
    assert result.current_equity == pytest.approx(101.1)
    assert result.reward.raw_equity_delta == pytest.approx(1.0)
    assert result.reward.total_reward == pytest.approx(1.0)


def test_compute_reward_step_includes_fill_and_fee():
    spec = _spec()
    fill = _fill(side=OrderSide.BUY, price_tick=1000, qty=1.0, fee=0.01)

    result = compute_reward_step(
        previous_position=PositionState(),
        fills=[fill],
        previous_book_top=_top(bid=1000, ask=1002),
        current_book_top=_top(bid=1000, ask=1002),
        symbol_spec=spec,
    )

    assert result.current_equity == pytest.approx(0.09)
    assert result.reward.raw_equity_delta == pytest.approx(0.09)


def test_inventory_penalty():
    spec = _spec()
    components = compute_reward_components(
        previous_equity=0.0,
        current_equity=0.0,
        current_position=PositionState(inventory_qty=2.0),
        current_mark_price=100.0,
        symbol_spec=spec,
        config=RewardConfig(inventory_penalty_bps=10.0),
    )

    expected = 2.0 * 100.0 * spec.contract_size * 10.0 / 10_000.0
    assert components.inventory_penalty == pytest.approx(expected)
    assert components.total_reward == pytest.approx(-expected)


def test_turnover_penalty():
    components = compute_reward_components(
        previous_equity=0.0,
        current_equity=0.0,
        current_position=PositionState(),
        current_mark_price=100.0,
        symbol_spec=_spec(),
        turnover_notional=1_000.0,
        config=RewardConfig(turnover_penalty_bps=5.0),
    )

    assert components.turnover_penalty == pytest.approx(1_000.0 * 5.0 / 10_000.0)


def test_cancel_penalty():
    components = compute_reward_components(
        previous_equity=0.0,
        current_equity=0.0,
        current_position=PositionState(),
        current_mark_price=100.0,
        symbol_spec=_spec(),
        cancel_count=3,
        config=RewardConfig(cancel_penalty=0.25),
    )

    assert components.cancel_penalty == pytest.approx(0.75)


def test_drawdown_penalty_incremental_only():
    components = compute_reward_components(
        previous_equity=95.0,
        current_equity=90.0,
        current_position=PositionState(),
        current_mark_price=100.0,
        symbol_spec=_spec(),
        peak_equity=100.0,
        config=RewardConfig(drawdown_penalty_rate=0.2),
    )

    assert components.drawdown_penalty == pytest.approx(1.0)


def test_drawdown_penalty_zero_when_drawdown_improves():
    components = compute_reward_components(
        previous_equity=90.0,
        current_equity=95.0,
        current_position=PositionState(),
        current_mark_price=100.0,
        symbol_spec=_spec(),
        peak_equity=100.0,
        config=RewardConfig(drawdown_penalty_rate=0.2),
    )

    assert components.drawdown_penalty == 0.0


def test_terminal_inventory_penalty():
    spec = _spec()
    components = compute_reward_components(
        previous_equity=0.0,
        current_equity=0.0,
        current_position=PositionState(inventory_qty=-2.0),
        current_mark_price=100.0,
        symbol_spec=spec,
        terminal=True,
        config=RewardConfig(terminal_inventory_penalty_bps=25.0),
    )

    expected = 2.0 * 100.0 * spec.contract_size * 25.0 / 10_000.0
    assert components.terminal_penalty == pytest.approx(expected)

    non_terminal = compute_reward_components(
        previous_equity=0.0,
        current_equity=0.0,
        current_position=PositionState(inventory_qty=-2.0),
        current_mark_price=100.0,
        symbol_spec=spec,
        terminal=False,
        config=RewardConfig(terminal_inventory_penalty_bps=25.0),
    )
    assert non_terminal.terminal_penalty == 0.0


def test_reward_components_total_reward_combines_penalties():
    components = compute_reward_components(
        previous_equity=10.0,
        current_equity=12.0,
        current_position=PositionState(inventory_qty=1.0),
        current_mark_price=100.0,
        symbol_spec=_spec(),
        turnover_notional=100.0,
        cancel_count=2,
        peak_equity=15.0,
        terminal=True,
        config=RewardConfig(
            inventory_penalty_bps=10.0,
            turnover_penalty_bps=5.0,
            cancel_penalty=0.1,
            drawdown_penalty_rate=0.5,
            terminal_inventory_penalty_bps=20.0,
        ),
    )

    assert isinstance(components, RewardComponents)
    expected_total = (
        components.raw_equity_delta
        - components.inventory_penalty
        - components.drawdown_penalty
        - components.turnover_penalty
        - components.cancel_penalty
        - components.terminal_penalty
    )
    assert components.total_reward == pytest.approx(expected_total)


def test_reward_step_returns_updated_peak_equity():
    result = compute_reward_step(
        previous_position=PositionState(cash=0.0, inventory_qty=1.0),
        fills=[],
        previous_book_top=_top(bid=1000, ask=1002),
        current_book_top=_top(bid=1010, ask=1012),
        symbol_spec=_spec(),
        peak_equity=100.5,
    )

    assert result.peak_equity == pytest.approx(max(result.previous_equity, result.current_equity, 100.5))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        compute_reward_components(
            previous_equity=0.0,
            current_equity=0.0,
            current_position=PositionState(),
            current_mark_price=0.0,
            symbol_spec=_spec(),
        )

    with pytest.raises(ValueError):
        compute_reward_components(
            previous_equity=0.0,
            current_equity=0.0,
            current_position=PositionState(),
            current_mark_price=100.0,
            symbol_spec=_spec(),
            cancel_count=-1,
        )

    with pytest.raises(ValueError):
        apply_fills_to_position(PositionState(), ["not a fill"], symbol_spec=_spec())


def test_reward_has_no_forbidden_imports():
    source = Path("mmrt/execution/reward.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import numpy" not in source
    assert "import sklearn" not in source
    assert "import pyarrow" not in source
