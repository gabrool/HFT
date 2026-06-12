"""Parity tests: closed-form conservative fills vs the event-replay simulator."""

import numpy as np
import pytest

from mmrt.execution.adverse_selection import (
    CounterfactualQuoteConfig,
    _build_conservative_fill_index,
    _conservative_fill_one_side,
    _counterfactual_fill_one_side,
)
from mmrt.execution.contracts import LatencyConfig, OrderSide
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH
from mmrt.execution.queue_model import QueueModelConfig
from mmrt.time_key import EventKey, MAX_EVENT_SEQ
from tests.test_execution_feature_replay import make_tape


def _decision_rows(tape):
    events = tape.arrays.events
    out = []
    latest_ptr = -1
    for idx in range(len(events)):
        row = events[idx]
        if int(row["event_type_code"]) == EVENT_TYPE_CODE_L2_BATCH and int(row["book_ptr"]) >= 0:
            l2 = tape.arrays.l2_events[int(row["book_ptr"])]
            if int(l2["best_bid_tick"]) > 0 and int(l2["best_ask_tick"]) > int(l2["best_bid_tick"]):
                latest_ptr = int(row["book_ptr"])
        if latest_ptr >= 0:
            out.append((idx, latest_ptr, int(row["local_ts_us"])))
    return out


@pytest.mark.parametrize("latency", [LatencyConfig(), LatencyConfig(decision_compute_latency_us=0, order_entry_latency_us=0, cancel_latency_us=0)])
@pytest.mark.parametrize("weight", [0.5, 1.0, 0.0])
def test_conservative_kernel_matches_event_replay(latency, weight):
    tape = make_tape(n_l2=240, l2_step_us=20_000)
    index = _build_conservative_fill_index(tape)
    config = CounterfactualQuoteConfig(
        queue_model=QueueModelConfig(trade_at_level_weight=weight),
        latency_config=latency,
    )
    events_ts = np.asarray(tape.arrays.events["local_ts_us"], dtype=np.int64)
    rows = _decision_rows(tape)
    rng = np.random.default_rng(11)
    checked = 0
    for idx, book_ptr, ts in rows[:: max(len(rows) // 60, 1)]:
        l2 = tape.arrays.l2_events[book_ptr]
        best_bid = int(l2["best_bid_tick"])
        best_ask = int(l2["best_ask_tick"])
        decision_key = EventKey(ts, MAX_EVENT_SEQ)
        deadline = EventKey(ts + config.fill_horizon_us, MAX_EVENT_SEQ)
        end_event_index = int(np.searchsorted(events_ts, deadline.local_ts_us, side="right"))
        for side, price_tick in (
            (OrderSide.BUY, best_bid),
            (OrderSide.SELL, best_ask),
            (OrderSide.BUY, best_bid + 1),
            (OrderSide.SELL, best_ask - 1),
            (OrderSide.BUY, best_bid - 1),
            (OrderSide.SELL, best_ask + 1),
            (OrderSide.BUY, best_bid - int(rng.integers(2, 6))),
            (OrderSide.SELL, best_ask + int(rng.integers(2, 6))),
        ):
            if price_tick <= 0 or (side == OrderSide.BUY and price_tick > best_ask - config.post_only_gap_ticks):
                continue
            if side == OrderSide.SELL and price_tick < best_bid + config.post_only_gap_ticks:
                continue
            slow = _counterfactual_fill_one_side(
                tape, start_event_index=idx, start_book_ptr=book_ptr, decision_key=decision_key,
                side=side, price_tick=price_tick, qty=config.order_qty, fill_deadline_key=deadline,
                end_event_index=end_event_index, config=config,
            )
            fast = _conservative_fill_one_side(
                index, tape, start_event_index=idx, start_book_ptr=book_ptr, decision_key=decision_key,
                side=side, price_tick=price_tick, end_event_index=end_event_index, config=config,
            )
            assert fast.filled == slow.filled, (idx, side, price_tick)
            if slow.filled:
                assert fast.fill_local_ts_us == slow.fill_local_ts_us, (idx, side, price_tick)
                assert fast.fill_price_tick == slow.fill_price_tick
                assert fast.fill_latency_us == slow.fill_latency_us
                assert fast.fill_reason == slow.fill_reason
            checked += 1
    assert checked > 100
