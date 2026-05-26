import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import book_state as bs
from mmrt.features.specs import FEATURE_COUNT, FEATURE_SPECS, FeatureSource, feature_spec_by_name


def make_snapshot(local_ts_us=1_000_000, mid=100.0, spread=0.10, bid_sz0=10.0, ask_sz0=12.0, bid_size_offset=0.0, ask_size_offset=0.0):
    best_bid = mid - spread / 2
    best_ask = mid + spread / 2
    bid_px = best_bid - 0.1 * np.arange(25)
    ask_px = best_ask + 0.1 * np.arange(25)
    bid_sz = bid_sz0 + bid_size_offset + np.arange(25)
    ask_sz = ask_sz0 + ask_size_offset + np.arange(25)
    return bs.BookSnapshotInput(local_ts_us=local_ts_us, ts_us=local_ts_us, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)

def fv_value(vec, name): return vec[feature_spec_by_name(name).index]

def test_public_api_boundary():
    exp={"BOOK_DEPTH","MAX_EMITTED_DEPTH","BID_SIDE_CODE","ASK_SIDE_CODE","BOOK_WINDOWS_US","DEFAULT_HISTORY_CAPACITY","BOOK_FEATURE_INDICES","BOOK_FEATURE_NAMES","BookSnapshotInput","BookSummary","BookHistory","BookState","book_owned_feature_names","book_owned_feature_indices"}
    assert set(bs.__all__)==exp

def test_no_forbidden_imports():
    code="import sys;before=set(sys.modules);import mmrt.features.book_state as b;after=set(sys.modules)-before;print('\\n'.join(sorted(after)))"
    out=subprocess.check_output([sys.executable,"-c",code],text=True)
    for bad in ["po"+"lars","pan"+"das","to"+"rch","py"+"arrow","mmrt.data.tardis_csv","mmrt.data.event_merge","mmrt.data.quality","CM"+"SSL17","offline_"+"ingest"]:
        assert bad not in out

def test_snapshot_input_validation():
    make_snapshot()
    with pytest.raises(ValueError): make_snapshot().__class__(1,1,np.ones(24),np.ones(25),np.ones(25),np.ones(25))
    with pytest.raises(ValueError): make_snapshot(bid_sz0=-1)
    bad=make_snapshot(); a=bad.ask_px.copy(); a[0]=0; 
    with pytest.raises(ValueError): bs.BookSnapshotInput(1,1,bad.bid_px,bad.bid_sz,a,bad.ask_sz)

def test_apply_and_features_smoke():
    st=bs.BookState()
    for i in range(15): st.apply_snapshot(make_snapshot(local_ts_us=1_000_000+i*250_000,mid=100+i*0.1,spread=0.1+0.01*(i%3),bid_sz0=10+i,ask_sz0=12-i*0.3))
    out=np.full(FEATURE_COUNT,-123.0); st.fill_book_features(out)
    assert np.all(np.isfinite(out[np.array(bs.book_owned_feature_indices())]))
    assert fv_value(out,"micro_ret_bps_200000us")!=0
    assert fv_value(out,"ofi_l1_sum_over_depth_200000us")!=0
    assert fv_value(out,"ob_update_rate_500000us")>0

def test_monotonic_and_reset_and_capacity():
    st=bs.BookState(history_capacity=3)
    st.apply_snapshot(make_snapshot(local_ts_us=1000)); st.apply_snapshot(make_snapshot(local_ts_us=1000))
    with pytest.raises(ValueError): st.apply_snapshot(make_snapshot(local_ts_us=999))
    for i in range(3): st.apply_snapshot(make_snapshot(local_ts_us=2000+i))
    assert st.history.size==3
    st.reset(); assert not st.has_book(); assert st.history.size==0

def test_ofi_and_l1_add_rem_specific():
    st=bs.BookState(); st.apply_snapshot(make_snapshot(bid_sz0=10,ask_sz0=12))
    st.apply_snapshot(make_snapshot(local_ts_us=1_200_000,bid_sz0=15,ask_sz0=10))
    out=np.zeros(FEATURE_COUNT); st.fill_book_features(out)
    assert fv_value(out,"ofi_l1")==pytest.approx(7.0)
    assert fv_value(out,"bid_l1_depletion_200000us")==0
    assert fv_value(out,"ask_l1_depletion_200000us")>0

def test_non_book_untouched_and_trade_placeholders():
    st=bs.BookState(); st.apply_snapshot(make_snapshot())
    out=np.full(FEATURE_COUNT,-7.0); st.fill_book_features(out)
    book=set(bs.book_owned_feature_indices())
    for i in range(FEATURE_COUNT):
        if i not in book: assert out[i]==-7.0
    assert fv_value(out,"time_since_trade_us")==0.0
