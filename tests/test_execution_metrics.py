import pytest

from mmrt.execution.contracts import ExecutionStepResult, PositionState, RewardComponents
from mmrt.execution.metrics import ExecutionMetricAccumulator


def test_metric_accumulator_summarizes_order_and_queue_diagnostics():
    info = {
        "events_processed": 3,
        "cancel_count": 1,
        "post_only_reject_count": 2,
        "activated_order_count": 3,
        "effective_cancel_count": 4,
        "pending_cancel_request_count": 5,
        "queue_trade_advance_qty": 0.1,
        "queue_l2_advance_qty": 0.2,
        "queue_advanced_qty": 0.3,
        "queue_fillable_qty": 0.4,
        "trade_at_level_fill_count": 1,
        "trade_through_fill_count": 2,
        "queue_depletion_fill_count": 3,
        "l2_trade_dedupe_qty": 0.5,
        "l2_raw_decrease_qty": 0.6,
        "l2_effective_decrease_qty": 0.1,
    }
    step = ExecutionStepResult(
        reward=RewardComponents(raw_equity_delta=0.0),
        position=PositionState(),
        fills=(),
        done=False,
        truncated=False,
        info=info,
    )

    acc = ExecutionMetricAccumulator()
    acc.update(step)
    summary = acc.as_dict()

    assert summary["orders"]["post_only_reject_count_total"] == 2
    assert summary["orders"]["activated_order_count_total"] == 3
    assert summary["orders"]["effective_cancel_count_total"] == 4
    assert summary["orders"]["pending_cancel_request_count_total"] == 5
    assert summary["queue"]["trade_advance_qty_total"] == pytest.approx(0.1)
    assert summary["queue"]["l2_advance_qty_total"] == pytest.approx(0.2)
    assert summary["queue"]["advanced_qty_total"] == pytest.approx(0.3)
    assert summary["queue"]["fillable_qty_total"] == pytest.approx(0.4)
    assert summary["queue"]["trade_at_level_fill_count"] == 1
    assert summary["queue"]["trade_through_fill_count"] == 2
    assert summary["queue"]["queue_depletion_fill_count"] == 3
    assert summary["queue"]["l2_trade_dedupe_qty_total"] == pytest.approx(0.5)
    assert summary["queue"]["l2_raw_decrease_qty_total"] == pytest.approx(0.6)
    assert summary["queue"]["l2_effective_decrease_qty_total"] == pytest.approx(0.1)
