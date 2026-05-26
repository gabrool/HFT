import importlib
import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import book_state as bs
from mmrt.features.specs import FEATURE_COUNT, FeatureSource, FEATURE_SPECS, feature_spec_by_name


def make_snapshot(local_ts_us=1_000_000, bid0=100.0, ask0=100.1, bid_sz0=10.0, ask_sz0=12.0):
    bid_px = np.array([100.0 - 0.1 * i for i in range(bs.BOOK_DEPTH)], dtype=np.float64)
    ask_px = np.array([100.1 + 0.1 * i for i in range(bs.BOOK_DEPTH)], dtype=np.float64)
    bid_sz = np.array([10.0 + i for i in range(bs.BOOK_DEPTH)], dtype=np.float64)
    ask_sz = np.array([12.0 + i for i in range(bs.BOOK_DEPTH)], dtype=np.float64)
    bid_px[0], ask_px[0], bid_sz[0], ask_sz[0] = bid0, ask0, bid_sz0, ask_sz0
    return bs.BookSnapshotInput(local_ts_us=local_ts_us, ts_us=local_ts_us, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)


def test_public_api_boundary():
    expected={"BOOK_DEPTH","MAX_EMITTED_DEPTH","BID_SIDE_CODE","ASK_SIDE_CODE","BOOK_WINDOWS_US","DEFAULT_HISTORY_CAPACITY","BOOK_FEATURE_INDICES","BOOK_FEATURE_NAMES","BookSnapshotInput","BookSummary","BookHistory","BookState","book_owned_feature_names","book_owned_feature_indices"}
    assert set(bs.__all__)==expected


def test_no_forbidden_imports():
    code="import sys; b=set(sys.modules); import mmrt.features.book_state; a=set(sys.modules); print('\\n'.join(sorted(a-b)))"
    p=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True)
    delta=p.stdout
    for bad in ["po"+"lars","pan"+"das","to"+"rch","py"+"arrow","mmrt.data.tardis_csv","mmrt.data.event_merge","mmrt.data.quality","CM"+"SSL17","offline_"+"ingest"]:
        assert bad not in delta


def test_snapshot_input_validation():
    s=make_snapshot(); assert s.bid_px.shape==(25,)
    bp=s.bid_px.copy(); bp[1]=bp[0]+1
    with pytest.raises(ValueError): make_snapshot().__class__(1,1,bp,s.bid_sz,s.ask_px,s.ask_sz)
    ap=s.ask_px.copy(); ap[1]=ap[0]-1
    with pytest.raises(ValueError): make_snapshot().__class__(1,1,s.bid_px,s.bid_sz,ap,s.ask_sz)


def test_book_feature_vector_shape_and_only_book_indices_written():
    st=bs.BookState(); st.apply_snapshot(make_snapshot()); out=np.full(FEATURE_COUNT,-123.0)
    st.fill_book_features(out)
    book=set(bs.BOOK_FEATURE_INDICES)
    assert all(np.isfinite(out[i]) for i in book)
    assert all(out[i]==-123.0 for i in range(FEATURE_COUNT) if i not in book)


def test_time_since_trade_placeholder_zero():
    st=bs.BookState(); st.apply_snapshot(make_snapshot()); v=st.book_feature_vector()
    assert v[feature_spec_by_name("time_since_trade_us").index]==0.0
