import subprocess
import sys
import numpy as np
import pytest
from mmrt.features import book_state as bs
from mmrt.features import kernels as k
from mmrt.features.specs import FEATURE_COUNT, FEATURE_SPECS, FeatureSource, feature_spec_by_name

def make_snapshot(local_ts_us=1_000_000, mid=100.0, spread=0.10, bid_sz0=10.0, ask_sz0=12.0, bid_size_offset=0.0, ask_size_offset=0.0):
    best_bid = mid - spread / 2.0; best_ask = mid + spread / 2.0
    bid_px = best_bid - 0.1 * np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    ask_px = best_ask + 0.1 * np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    bid_sz = bid_sz0 + bid_size_offset + np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    ask_sz = ask_sz0 + ask_size_offset + np.arange(bs.BOOK_DEPTH, dtype=np.float64)
    return bs.BookSnapshotInput(local_ts_us=local_ts_us, ts_us=local_ts_us, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)

def fv_value(vec,name): return vec[feature_spec_by_name(name).index]

def apply_dynamic_sequence(st):
    mids=[100.0,100.20,100.10,100.40,100.15,100.60,100.45,100.80,100.55,100.95,100.70,101.10,100.90,101.25,101.05]
    spreads=[0.10,0.12,0.09,0.15,0.11,0.14,0.08,0.16,0.10,0.13,0.09,0.15,0.12,0.14,0.10]
    for i,(m,s) in enumerate(zip(mids,spreads)):
        st.apply_snapshot(make_snapshot(local_ts_us=1_000_000+i*250_000,mid=m,spread=s,bid_sz0=10+((i*3)%7),ask_sz0=12+((i*5)%9),bid_size_offset=float(i%4),ask_size_offset=float((i+1)%4)))

def test_public_api_boundary():
    exp={"BOOK_DEPTH","MAX_EMITTED_DEPTH","BID_SIDE_CODE","ASK_SIDE_CODE","BOOK_WINDOWS_US","DEFAULT_HISTORY_CAPACITY","BOOK_FEATURE_INDICES","BOOK_FEATURE_NAMES","BookSnapshotInput","BookSummary","BookHistory","BookState","book_owned_feature_names","book_owned_feature_indices"}
    assert set(bs.__all__)==exp

def test_book_owned_indices_match_specs_after_reclassification():
    expected=tuple(i for i,s in enumerate(FEATURE_SPECS) if s.source==FeatureSource.BOOK)
    assert bs.book_owned_feature_indices()==expected
    names=bs.book_owned_feature_names()
    for n in ("time_since_trade_us","vwap_vs_mid_bps_200000us","vwap_vs_mid_bps_500000us","regime_volume_ewma_500000us","regime_volume_ewma_3000000us"):
        assert n not in names

def test_sparse_window_micro_ret_uses_right_asof():
    st=bs.BookState(); st.apply_snapshot(make_snapshot(local_ts_us=1_000_000,mid=100.0)); st.apply_snapshot(make_snapshot(local_ts_us=1_250_000,mid=101.0)); st.apply_snapshot(make_snapshot(local_ts_us=1_500_000,mid=102.0))
    out=np.full(FEATURE_COUNT,-1.0); st.fill_book_features(out)
    micro_now=st.history.asof_value("microprice",1_500_000); micro_past=st.history.asof_value("microprice",1_300_000)
    expected=float(k.bps_change(micro_now,micro_past))
    got=fv_value(out,"micro_ret_bps_200000us")
    assert got==pytest.approx(expected)
    assert got!=0.0

def test_feature_vector_shape_and_non_book_indices_untouched():
    st=bs.BookState(); apply_dynamic_sequence(st)
    out=np.full(FEATURE_COUNT,-123.0); st.fill_book_features(out)
    b=set(bs.book_owned_feature_indices())
    assert np.all(np.isfinite(out[list(b)]))
    for i in range(FEATURE_COUNT):
        if i not in b: assert out[i]==-123.0

def test_return_distribution_features():
    st=bs.BookState(); apply_dynamic_sequence(st)
    out=np.full(FEATURE_COUNT,0.0); st.fill_book_features(out)
    assert fv_value(out,"max_abs_return_bps_500000us")>0
    assert -1<=fv_value(out,"down_up_vol_imbalance_500000us")<=1

def test_no_invalid_placeholders_remaining():
    src=open("mmrt/features/book_state.py","r",encoding="utf-8").read().replace(" ","")
    for bad in ['setf("vwap_vs_mid_bps_200000us",0.0)','setf("vwap_vs_mid_bps_500000us",0.0)','setf("down_up_vol_imbalance_500000us",0.0)','setf("down_up_vol_imbalance_1000000us",0.0)','setf("down_up_vol_imbalance_3000000us",0.0)']:
        assert bad.replace(" ","") not in src

def test_liquidity_void_distance_from_mid():
    st=bs.BookState(); snap=make_snapshot()
    snap.bid_sz[:20]=np.array([1,2,3,4,5,6,7,8,9,10,100,1,1,1,1,1,1,1,1,1],dtype=np.float64)
    st.apply_snapshot(snap)
    out=np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    val=fv_value(out,"bid_liquidity_void_bps")
    assert val>=0

def test_no_forbidden_imports():
    code="import sys;before=set(sys.modules);import mmrt.features.book_state as b;after=set(sys.modules)-before;print('\\n'.join(sorted(after)))"
    out=subprocess.check_output([sys.executable,"-c",code],text=True)
    for bad in ["po"+"lars","pan"+"das","to"+"rch","py"+"arrow","mmrt.data.tardis_csv","mmrt.data.event_merge","mmrt.data.quality","CMSSL17","offline_ingest"]:
        assert bad not in out
