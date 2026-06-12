from decimal import Decimal

import numpy as np
import pytest

from mmrt.contracts import AggressorSide
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
from mmrt.execution.event_merge import merge_execution_events
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, build_execution_tape
from mmrt.execution.feature_replay import (
    decision_feature_column_names,
    iter_decision_feature_chunks,
    iter_tape_feature_steps,
)
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.features.pipeline import DecisionFeaturePipeline, FeaturePipelineConfig
from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode
from mmrt.storage import manifest as mf


def _rules():
    return ExchangeSymbolRules(
        exchange="binance-futures", symbol="BTCUSDT", mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC", quote_asset="USDT", margin_asset="USDT", contract_type="PERPETUAL", status="TRADING",
        tick_size=Decimal("0.1"), min_price=Decimal("0.1"), max_price=Decimal("1000000"),
        step_size=Decimal("0.001"), min_qty=Decimal("0.001"), max_qty=Decimal("100"), min_notional=Decimal("5"),
        allowed_order_types=("LIMIT",), allowed_time_in_force=("GTC", "GTX"),
    )


def _spec():
    return SymbolSpec("binance-futures", "BTCUSDT", 0.1, 0.001, 0.001, 100.0, 5.0)


def _l2(seq: int, local_ts_us: int, *, bid: int, ask: int, bid_size: float = 1.0) -> ReconstructedL2Event:
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us,
        max_ts_us=local_ts_us,
        num_updates=2,
        is_snapshot_batch=(seq == 0),
        book_top=BookTop(local_ts_us, bid, ask, bid_size, 1.2),
        bid_depth=3,
        ask_depth=3,
        book_snapshot=BookLevelSnapshot(
            local_ts_us,
            (bid, bid - 1, bid - 2), (bid_size, 2.0, 3.0),
            (ask, ask + 1, ask + 2), (1.2, 2.2, 3.2),
        ),
    )


def _one_sided_l2(seq: int, local_ts_us: int, *, bid: int) -> ReconstructedL2Event:
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us,
        max_ts_us=local_ts_us,
        num_updates=1,
        is_snapshot_batch=False,
        book_top=None,
        bid_depth=1,
        ask_depth=0,
        book_snapshot=BookLevelSnapshot(local_ts_us, (bid,), (1.0,), (), ()),
    )


def make_tape(*, n_l2: int = 120, l2_step_us: int = 100_000, base_ts_us: int = 1_000_000, one_sided_at: int | None = None):
    l2 = []
    for i in range(n_l2):
        ts = base_ts_us + i * l2_step_us
        if one_sided_at is not None and i == one_sided_at:
            l2.append(_one_sided_l2(i, ts, bid=1000))
            continue
        drift = (i // 7) - (i // 11)
        bid = 1000 + drift
        ask = bid + 2 + (i % 2)
        l2.append(_l2(i, ts, bid=bid, ask=ask, bid_size=1.0 + 0.5 * (i % 3)))
    trades = []
    for k in range(max(n_l2 // 3, 1)):
        ts = base_ts_us + 50_000 + k * 3 * l2_step_us
        side = AggressorSide.BUY if k % 2 == 0 else AggressorSide.SELL
        trades.append(TradePrint(ts, ts, side, 1001, 0.02, source_row=k))
    merged = merge_execution_events(l2, trades).events
    return build_execution_tape(
        symbol_spec=_spec(), symbol_rules=_rules(), l2_events=l2, trades=trades,
        merged_events=merged, book_depth=3, created_at_utc="2026-01-01T00:00:00Z",
    )


_SCHED_500MS = DecisionScheduleConfig(min_decision_interval_us=500_000, max_decision_interval_us=500_000)


def test_decision_feature_column_names_match_storage_feature_columns():
    assert decision_feature_column_names() == mf.feature_columns()


def test_iter_tape_feature_steps_yields_valid_l2_steps_with_decisions():
    tape = make_tape(n_l2=60)
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(schedule=_SCHED_500MS))
    steps = list(iter_tape_feature_steps(tape, pipeline=pipeline))
    assert steps
    events = tape.arrays.events
    for step in steps:
        row = events[step.event_index]
        assert int(row["event_type_code"]) == EVENT_TYPE_CODE_L2_BATCH
        assert int(row["local_ts_us"]) == step.local_ts_us
        assert step.mid > 0.0
    decisions = [s.decision for s in steps if s.decision is not None]
    assert len(decisions) > 1
    decision_ts = [d.local_ts_us for d in decisions]
    assert decision_ts == sorted(decision_ts)
    gaps = np.diff(np.asarray(decision_ts))
    assert (gaps >= 500_000).all()


def test_iter_tape_feature_steps_skips_one_sided_books():
    skip_index = 30
    tape = make_tape(n_l2=60, one_sided_at=skip_index)
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(schedule=_SCHED_500MS))
    steps = list(iter_tape_feature_steps(tape, pipeline=pipeline))
    skipped_event_indices = {
        int(event["event_seq"])
        for event in tape.arrays.events
        if int(event["event_type_code"]) == EVENT_TYPE_CODE_L2_BATCH
        and int(event["book_ptr"]) >= 0
        and int(tape.arrays.l2_events[int(event["book_ptr"])]["best_ask_tick"]) <= 0
    }
    assert skipped_event_indices
    assert not skipped_event_indices.intersection({s.event_index for s in steps})


def test_iter_tape_feature_steps_respects_max_events():
    tape = make_tape(n_l2=60)
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(schedule=_SCHED_500MS))
    steps = list(iter_tape_feature_steps(tape, pipeline=pipeline, max_events=10))
    assert all(s.event_index < 10 for s in steps)


def test_iter_tape_feature_steps_validates_inputs():
    tape = make_tape(n_l2=10)
    pipeline = DecisionFeaturePipeline()
    with pytest.raises(ValueError, match="start_event_index"):
        list(iter_tape_feature_steps(tape, pipeline=pipeline, start_event_index=10_000))
    with pytest.raises(ValueError, match="pipeline"):
        list(iter_tape_feature_steps(tape, pipeline=None))  # type: ignore[arg-type]


def test_iter_decision_feature_chunks_chunking_and_alignment():
    tape = make_tape(n_l2=120)
    config = FeaturePipelineConfig(schedule=_SCHED_500MS)
    chunks = list(iter_decision_feature_chunks(tape, pipeline_config=config, chunk_rows=3))
    assert chunks
    assert all(c.features.shape[0] <= 3 for c in chunks)
    assert all(c.feature_names == decision_feature_column_names() for c in chunks)
    event_index = np.concatenate([c.decision_event_index for c in chunks])
    assert (np.diff(event_index) > 0).all()
    features = np.vstack([c.features for c in chunks])
    assert np.isfinite(features).all()

    single = list(iter_decision_feature_chunks(tape, pipeline_config=config, chunk_rows=100_000))
    assert len(single) == 1
    np.testing.assert_array_equal(np.vstack([c.features for c in single]), features)


def test_iter_decision_feature_chunks_max_decisions():
    tape = make_tape(n_l2=120)
    config = FeaturePipelineConfig(schedule=_SCHED_500MS)
    limited = list(iter_decision_feature_chunks(tape, pipeline_config=config, max_decisions=2))
    total = sum(c.features.shape[0] for c in limited)
    assert total == 2


def test_replay_is_deterministic_across_runs():
    tape = make_tape(n_l2=120)
    config = FeaturePipelineConfig(schedule=_SCHED_500MS)
    a = np.vstack([c.features for c in iter_decision_feature_chunks(tape, pipeline_config=config)])
    b = np.vstack([c.features for c in iter_decision_feature_chunks(tape, pipeline_config=config)])
    np.testing.assert_array_equal(a, b)
