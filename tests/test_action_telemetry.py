import json

import numpy as np
import pytest
import torch

from mmrt.execution.contracts import (
    ExecutionStepResult,
    Fill,
    FillReason,
    OrderSide,
    PositionState,
    RewardComponents,
)
from mmrt.rl.action_telemetry import (
    ACTION_TELEMETRY_SCHEMA,
    ASK_ONLY,
    BID_ONLY,
    NO_QUOTE,
    TWO_SIDED,
    ActionTelemetryAccumulator,
    quote_mode_from_bools,
    scalar_stats,
)


def _fill(side: OrderSide, reason: FillReason, *, qty: float = 0.5, fee: float = 0.1) -> Fill:
    return Fill(
        order_id=1,
        side=side,
        local_ts_us=100,
        event_seq=1,
        price_tick=1000,
        qty=qty,
        fee=fee,
        reason=reason,
    )


def _step(
    *,
    bid_enabled: bool,
    ask_enabled: bool,
    raw_equity_delta: float,
    fills=(),
    bid_reason: str = "",
    ask_reason: str = "",
    cancel_count: int = 0,
    post_only_reject_count: int = 0,
    turnover_notional: float = 0.0,
) -> ExecutionStepResult:
    return ExecutionStepResult(
        reward=RewardComponents(raw_equity_delta=raw_equity_delta),
        position=PositionState(),
        fills=tuple(fills),
        done=False,
        truncated=False,
        info={
            "quote_bid_enabled": bid_enabled,
            "quote_ask_enabled": ask_enabled,
            "quote_bid_qty": 1.0 if bid_enabled else 0.0,
            "quote_ask_qty": 2.0 if ask_enabled else 0.0,
            "quote_bid_offset_ticks": 1,
            "quote_ask_offset_ticks": 2,
            "quote_bid_disabled_reason": bid_reason,
            "quote_ask_disabled_reason": ask_reason,
            "cancel_count": cancel_count,
            "post_only_reject_count": post_only_reject_count,
            "turnover_notional": turnover_notional,
        },
    )


def test_quote_mode_helpers_and_scalar_stats_are_json_safe():
    assert quote_mode_from_bools(False, False) == NO_QUOTE
    assert quote_mode_from_bools(True, False) == BID_ONLY
    assert quote_mode_from_bools(False, True) == ASK_ONLY
    assert quote_mode_from_bools(True, True) == TWO_SIDED

    stats = scalar_stats([1.0, 2.0, 3.0])
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["min"] == 1.0
    assert stats["p50"] == 2.0
    json.dumps(stats)


def test_requested_action_counts_rates_and_cancel_guards():
    acc = ActionTelemetryAccumulator()
    actions = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    modes = acc.update_requested_actions(actions)
    payload = acc.as_dict()
    requested = payload["requested_actions"]

    assert modes.tolist() == [0, 1, 2, 3]
    assert requested["requested_bid_enabled_count"] == 2
    assert requested["requested_ask_enabled_count"] == 2
    assert requested["requested_no_quote_count"] == 1
    assert requested["requested_bid_only_count"] == 1
    assert requested["requested_ask_only_count"] == 1
    assert requested["requested_two_sided_count"] == 1
    assert requested["requested_bid_cancel_guard_count"] == 2
    assert requested["requested_ask_cancel_guard_count"] == 2
    assert requested["requested_bid_enabled_rate"] == 0.5
    assert requested["requested_ask_enabled_rate"] == 0.5


def test_effective_quote_reasons_and_outcomes_by_mode_are_aggregated():
    acc = ActionTelemetryAccumulator()
    acc.update_execution_step(
        _step(
            bid_enabled=False,
            ask_enabled=False,
            raw_equity_delta=-1.0,
            bid_reason="disabled_by_action",
            ask_reason="qty_below_min",
        )
    )
    acc.update_execution_step(
        _step(
            bid_enabled=True,
            ask_enabled=False,
            raw_equity_delta=2.0,
            fills=(_fill(OrderSide.BUY, FillReason.TRADE_THROUGH, qty=0.25),),
            ask_reason="disabled_by_action",
            cancel_count=1,
            turnover_notional=25.0,
        )
    )
    acc.update_execution_step(
        _step(
            bid_enabled=False,
            ask_enabled=True,
            raw_equity_delta=3.0,
            fills=(_fill(OrderSide.SELL, FillReason.QUEUE_DEPLETION, qty=0.5),),
            bid_reason="inventory_limit",
            post_only_reject_count=1,
            turnover_notional=50.0,
        )
    )

    payload = acc.as_dict()
    effective = payload["effective_quotes"]
    outcomes = payload["outcomes_by_effective_quote_mode"]

    assert payload["schema"] == ACTION_TELEMETRY_SCHEMA
    assert payload["sample_count"] == 3
    assert effective["quote_no_quote_count"] == 1
    assert effective["quote_bid_only_count"] == 1
    assert effective["quote_ask_only_count"] == 1
    assert effective["bid_disabled_reason_counts"] == {
        "disabled_by_action": 1,
        "inventory_limit": 1,
    }
    assert effective["ask_disabled_reason_counts"] == {
        "qty_below_min": 1,
        "disabled_by_action": 1,
    }
    assert effective["quote_bid_qty"]["count"] == 1
    assert effective["quote_ask_offset_ticks"]["count"] == 1

    bid = outcomes[BID_ONLY]
    ask = outcomes[ASK_ONLY]
    assert bid["step_count"] == 1
    assert bid["reward_sum"] == 2.0
    assert bid["fill_count"] == 1
    assert bid["fill_step_rate"] == 1.0
    assert bid["buy_fill_count"] == 1
    assert bid["net_qty"] == 0.25
    assert bid["fill_reason_counts"]["trade_through"] == 1
    assert bid["cancel_count"] == 1
    assert ask["sell_fill_count"] == 1
    assert ask["net_qty"] == -0.5
    assert ask["post_only_reject_count"] == 1
    assert ask["turnover_notional"] == 50.0
    json.dumps(payload)


def test_training_by_effective_quote_mode_advantage_stats():
    acc = ActionTelemetryAccumulator()
    acc.update_training_by_effective_quote_mode(
        np.array([0, 1, 1, 3]),
        advantages=np.array([1.0, 2.0, 4.0, -1.0]),
        returns=np.array([1.5, 2.5, 4.5, -0.5]),
        values=np.array([0.5, 0.5, 0.5, 0.5]),
        rewards=np.array([0.1, 0.2, 0.4, -0.1]),
    )

    training = acc.as_dict()["training_by_effective_quote_mode"]
    assert training[NO_QUOTE]["count"] == 1
    assert training[BID_ONLY]["count"] == 2
    assert training[BID_ONLY]["advantage_mean"] == 3.0
    assert training[TWO_SIDED]["reward_mean"] == -0.1
    json.dumps(training)


def test_training_by_effective_quote_mode_cpu_tensor_valid_mask_filters_rows():
    acc = ActionTelemetryAccumulator()
    acc.update_training_by_effective_quote_mode(
        torch.tensor([0, 1, 1, 3], dtype=torch.int64),
        advantages=torch.tensor([1.0, 2.0, 4.0, -1.0]),
        returns=torch.tensor([1.5, 2.5, 4.5, -0.5]),
        values=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        rewards=torch.tensor([0.1, 0.2, 0.4, -0.1]),
        valid_mask=torch.tensor([True, False, True, False]),
    )

    training = acc.as_dict()["training_by_effective_quote_mode"]
    assert training[NO_QUOTE]["count"] == 1
    assert training[BID_ONLY]["count"] == 1
    assert training[BID_ONLY]["advantage_mean"] == 4.0
    assert training[BID_ONLY]["reward_mean"] == pytest.approx(0.4)
    assert training[TWO_SIDED]["count"] == 0
    assert training[TWO_SIDED]["reward_mean"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_training_by_effective_quote_mode_cuda_tensor_valid_mask_filters_rows():
    device = torch.device("cuda:0")
    acc = ActionTelemetryAccumulator()
    acc.update_training_by_effective_quote_mode(
        torch.tensor([0, 1, 1, 3], dtype=torch.int64, device=device),
        advantages=torch.tensor([1.0, 2.0, 4.0, -1.0], device=device),
        returns=torch.tensor([1.5, 2.5, 4.5, -0.5], device=device),
        values=torch.tensor([0.5, 0.5, 0.5, 0.5], device=device),
        rewards=torch.tensor([0.1, 0.2, 0.4, -0.1], device=device),
        valid_mask=torch.tensor([True, False, True, False], device=device),
    )

    training = acc.as_dict()["training_by_effective_quote_mode"]
    assert training[NO_QUOTE]["count"] == 1
    assert training[BID_ONLY]["count"] == 1
    assert training[BID_ONLY]["advantage_mean"] == 4.0
    assert training[BID_ONLY]["reward_mean"] == pytest.approx(0.4)
    assert training[TWO_SIDED]["count"] == 0
    assert training[TWO_SIDED]["reward_mean"] is None


def test_training_by_effective_quote_mode_numpy_valid_mask_filters_rows():
    acc = ActionTelemetryAccumulator()
    acc.update_training_by_effective_quote_mode(
        np.array([0, 1, 1, 3], dtype=np.int64),
        advantages=np.array([1.0, 2.0, 4.0, -1.0], dtype=np.float32),
        returns=np.array([1.5, 2.5, 4.5, -0.5], dtype=np.float32),
        values=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        rewards=np.array([0.1, 0.2, 0.4, -0.1], dtype=np.float32),
        valid_mask=np.array([True, False, True, False], dtype=np.bool_),
    )

    training = acc.as_dict()["training_by_effective_quote_mode"]
    assert training[NO_QUOTE]["count"] == 1
    assert training[BID_ONLY]["count"] == 1
    assert training[BID_ONLY]["advantage_mean"] == 4.0
    assert training[BID_ONLY]["reward_mean"] == pytest.approx(0.4)
    assert training[TWO_SIDED]["count"] == 0
    assert training[TWO_SIDED]["reward_mean"] is None


def test_training_by_effective_quote_mode_rejects_wrong_size_valid_mask():
    acc = ActionTelemetryAccumulator()
    with pytest.raises(ValueError, match="valid_mask must have the same flattened size"):
        acc.update_training_by_effective_quote_mode(
            np.array([0, 1, 1, 3], dtype=np.int64),
            advantages=np.array([1.0, 2.0, 4.0, -1.0], dtype=np.float32),
            returns=np.array([1.5, 2.5, 4.5, -0.5], dtype=np.float32),
            values=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            rewards=np.array([0.1, 0.2, 0.4, -0.1], dtype=np.float32),
            valid_mask=np.array([True, False, True], dtype=np.bool_),
        )
