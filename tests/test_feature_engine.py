import numpy as np

from mmrt.features.engine import EventHistory, FeatureEngine, L2_EVENT_CODE, TRADE_EVENT_CODE
from mmrt.features.book_state import BOOK_DEPTH, BookSnapshotInput
from mmrt.features.trade_state import BUY_SIDE_CODE, SELL_SIDE_CODE, TradeInput
from mmrt.features.specs import feature_spec_by_name


def test_event_history_window_view_matches_count_in_window_wrapped():
    h = EventHistory(capacity=3)
    for t in (100, 200, 300, 400, 500):
        h.append(t, L2_EVENT_CODE)
    view = h.window_view(now_us=500, windows_us=(250,))
    assert view.count(250) == 3
    assert view.count(250) == h.count_in_window(500, 250)


def _snapshot(local_ts_us: int, mid: float) -> BookSnapshotInput:
    half_spread = 0.5
    bid = mid - half_spread
    ask = mid + half_spread
    bid_px = bid - np.arange(BOOK_DEPTH, dtype=np.float64) * 0.01
    ask_px = ask + np.arange(BOOK_DEPTH, dtype=np.float64) * 0.01
    bid_sz = np.full(BOOK_DEPTH, 10.0, dtype=np.float64)
    ask_sz = np.full(BOOK_DEPTH, 10.0, dtype=np.float64)
    return BookSnapshotInput(local_ts_us, local_ts_us, bid_px, bid_sz, ask_px, ask_sz)


def test_feature_engine_feature_vector_stable_with_window_cache():
    engine = FeatureEngine()
    engine.on_book_snapshot(_snapshot(1_000_000, 100.0))
    engine.on_trade(TradeInput(1_100_000, 1_100_000, 100.0, 2.0, BUY_SIDE_CODE))
    engine.on_trade(TradeInput(1_200_000, 1_200_000, 100.0, 1.0, SELL_SIDE_CODE))
    snapshot = _snapshot(1_500_000, 100.2)
    engine.observe_book_snapshot(snapshot)
    decision = engine.force_decision(local_ts_us=snapshot.local_ts_us, ts_us=snapshot.ts_us, event_seq=snapshot.event_seq)
    assert decision is not None
    fv = decision.feature_vector
    assert np.all(np.isfinite(fv))
    count_idx = feature_spec_by_name("trade_count_per_second_500000us").index
    imbalance_idx = feature_spec_by_name("trade_imbalance_notional_500000us").index
    events_idx = feature_spec_by_name("log_events_500000us").index
    assert fv[count_idx] == 4.0
    assert fv[imbalance_idx] == (200.0 - 100.0) / (200.0 + 100.0)
    assert fv[events_idx] == np.log1p(4.0)
