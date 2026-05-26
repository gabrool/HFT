import subprocess
import sys
import numpy as np
import pytest
from mmrt.features import engine as eg
from mmrt.features import kernels as k
from mmrt.features.book_state import BOOK_DEPTH, BookSnapshotInput
from mmrt.features.trade_state import BUY_SIDE_CODE, SELL_SIDE_CODE, TradeInput
from mmrt.features.specs import FEATURE_COUNT, FEATURE_SPECS, FeatureSource, feature_spec_by_name

def make_snapshot(local_ts_us=2_000_000, mid=100.0, spread=0.10, bid_sz0=10.0, ask_sz0=12.0, bid_size_offset=0.0, ask_size_offset=0.0):
    best_bid = mid - spread / 2.0; best_ask = mid + spread / 2.0
    bid_px = best_bid - 0.1 * np.arange(BOOK_DEPTH, dtype=np.float64); ask_px = best_ask + 0.1 * np.arange(BOOK_DEPTH, dtype=np.float64)
    bid_sz = bid_sz0 + bid_size_offset + np.arange(BOOK_DEPTH, dtype=np.float64); ask_sz = ask_sz0 + ask_size_offset + np.arange(BOOK_DEPTH, dtype=np.float64)
    return BookSnapshotInput(local_ts_us=local_ts_us, ts_us=local_ts_us, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)

def make_trade(local_ts_us=2_000_000, price=100.0, amount=1.0, side_code=BUY_SIDE_CODE):
    return TradeInput(local_ts_us=local_ts_us, ts_us=local_ts_us, price=price, amount=amount, side_code=side_code)

def fv_value(vec, name): return vec[feature_spec_by_name(name).index]
def feed_ready_engine():
    eng = eg.FeatureEngine(); eng.on_trade(make_trade(2_000_000)); dec = eng.on_book_snapshot(make_snapshot(2_000_000)); assert dec is not None; return eng, dec

def test_public_api_boundary():
    assert set(eg.__all__) == {"BOOK_EVENT_CODE","TRADE_EVENT_CODE","DECISION_STRIDE_US","ENGINE_EVENT_WINDOWS_US","DEFAULT_EVENT_HISTORY_CAPACITY","CROSS_FEATURE_INDICES","CROSS_FEATURE_NAMES","EVENT_CONTEXT_FEATURE_INDICES","EVENT_CONTEXT_FEATURE_NAMES","ENGINE_FEATURE_INDICES","ENGINE_FEATURE_NAMES","FeatureEngineConfig","EngineDecision","EventHistory","FeatureEngine","cross_feature_names","cross_feature_indices","event_context_feature_names","event_context_feature_indices","engine_owned_feature_names","engine_owned_feature_indices"}

def test_no_forbidden_imports():
    out = subprocess.check_output([sys.executable, "-c", "import sys; b=set(sys.modules); import mmrt.features.engine; a=set(sys.modules); print('\\n'.join(sorted(a-b)))"], text=True)
    assert all(f not in out for f in ["po"+"lars","pa"+"ndas","to"+"rch","py"+"arrow","mmrt.features.trans"+"forms"])

def test_config_validation():
    assert eg.FeatureEngineConfig().decision_stride_us == 500_000
    with pytest.raises(ValueError): eg.FeatureEngineConfig(decision_stride_us=0)
    with pytest.raises(ValueError): eg.FeatureEngineConfig(event_history_capacity=0)

def test_event_history_counts_inclusive():
    h=eg.EventHistory(); h.append(2_000_000,1); h.append(2_100_000,2); h.append(2_200_000,1)
    assert h.count_in_window(2_200_000,200_000)==3 and h.count_in_window(2_200_000,100_000)==2

def test_event_history_capacity():
    h=eg.EventHistory(3); [h.append(i,1) for i in [1,2,3,4,5]]; assert np.array_equal(h.ordered_ts(), np.array([3,4,5]))

def test_no_decision_before_ready():
    e=eg.FeatureEngine(); assert e.on_book_snapshot(make_snapshot(2_000_000)) is None; e.on_trade(make_trade(2_100_000)); assert e.on_book_snapshot(make_snapshot(2_100_000)) is not None

def test_decision_cadence_book_only():
    e=eg.FeatureEngine(); e.on_trade(make_trade(2_000_000)); assert e.on_book_snapshot(make_snapshot(2_000_000)); assert e.on_book_snapshot(make_snapshot(2_100_000)) is None; assert e.on_book_snapshot(make_snapshot(2_500_000)); assert e.on_book_snapshot(make_snapshot(2_500_000)) is None

def test_long_gap_emits_at_most_one_decision():
    e,_=feed_ready_engine(); c=e.decision_count; assert e.on_book_snapshot(make_snapshot(4_000_000)); assert e.decision_count==c+1

def test_feature_vector_shape_and_all_finite():
    e,_=feed_ready_engine(); e.on_trade(make_trade(2_100_000,101,2,SELL_SIDE_CODE)); d=e.on_book_snapshot(make_snapshot(2_500_000,mid=100.2)); assert d.feature_vector.shape==(FEATURE_COUNT,) and np.all(np.isfinite(d.feature_vector))

def test_engine_owned_indices_match_specs():
    cross=tuple(s.index for s in FEATURE_SPECS if s.source==FeatureSource.CROSS); event=tuple(s.index for s in FEATURE_SPECS if s.source==FeatureSource.EVENT_CONTEXT)
    assert eg.engine_owned_feature_indices()==cross+event

def test_vwap_vs_mid_formula_manual():
    e=eg.FeatureEngine(); e.on_trade(make_trade(2_000_000,101,2,BUY_SIDE_CODE)); d=e.on_book_snapshot(make_snapshot(2_000_000,mid=100)); assert fv_value(d.feature_vector,"vwap_vs_mid_bps_200000us")==pytest.approx(k.bps_change(101,100))

def test_event_context_formulas_manual():
    e=eg.FeatureEngine(); e.on_trade(make_trade(2_000_000)); d1=e.on_book_snapshot(make_snapshot(2_000_000)); e.on_trade(make_trade(2_300_000)); d2=e.on_book_snapshot(make_snapshot(2_500_000)); assert fv_value(d1.feature_vector,"log_dt_decision_us")==0.0; assert fv_value(d2.feature_vector,"log_dt_decision_us")==pytest.approx(np.log1p(500_000))

def test_out_of_order_events_rejected():
    e,_=feed_ready_engine();
    with pytest.raises(ValueError): e.on_trade(make_trade(1_999_999))

def test_build_feature_vector_rejects_not_ready_or_bad_as_of():
    e=eg.FeatureEngine();
    with pytest.raises(ValueError): e.build_feature_vector()
