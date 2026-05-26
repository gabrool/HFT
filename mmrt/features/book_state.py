from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from mmrt.features import kernels as k
from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_SPECS,
    FEATURE_NAMES,
    FeatureSource,
    feature_indices_by_source,
    feature_spec_by_name,
)

"""Book-state feature computation for the MMRT feature pipeline.

This module consumes normalized top-25 book snapshots and maintains causal
book-only state. It computes only BOOK-owned features from specs.py and exposes
book summaries for later engine-level cross features. It does not parse market
data files, reconstruct incremental books, compute trade features, build labels,
apply transforms, or write storage artifacts.
"""
BOOK_DEPTH = 25
MAX_EMITTED_DEPTH = 20
BID_SIDE_CODE = 1
ASK_SIDE_CODE = -1
WINDOW_100MS_US = 100_000
WINDOW_200MS_US = 200_000
WINDOW_500MS_US = 500_000
WINDOW_1000MS_US = 1_000_000
WINDOW_3000MS_US = 3_000_000
BOOK_WINDOWS_US = (WINDOW_100MS_US, WINDOW_200MS_US, WINDOW_500MS_US, WINDOW_1000MS_US, WINDOW_3000MS_US)
DEFAULT_HISTORY_CAPACITY = 16_384
BOOK_FEATURE_INDICES = feature_indices_by_source(FeatureSource.BOOK)
BOOK_FEATURE_NAMES = tuple(FEATURE_NAMES[i] for i in BOOK_FEATURE_INDICES)
assert BOOK_FEATURE_INDICES
assert all(FEATURE_SPECS[i].source == FeatureSource.BOOK for i in BOOK_FEATURE_INDICES)
assert MAX_EMITTED_DEPTH <= BOOK_DEPTH
assert max((s.required_book_depth for s in FEATURE_SPECS if s.source == FeatureSource.BOOK), default=0) <= MAX_EMITTED_DEPTH


def _require_int(value, name, *, positive=False, allow_minus_one=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(name)
    if allow_minus_one and value == -1:
        return value
    if positive and value <= 0:
        raise ValueError(name)
    return value

def _coerce_book_array(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != BOOK_DEPTH:
        raise ValueError(name)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(name)
    return np.ascontiguousarray(arr)

def _validate_book_order(px, side_name):
    seen_zero=False
    prev=None
    for v in px:
        if v==0:
            seen_zero=True
            continue
        if seen_zero:
            raise ValueError(f"{side_name} compact")
        if prev is not None:
            if side_name=="bid" and v>prev: raise ValueError(side_name)
            if side_name=="ask" and v<prev: raise ValueError(side_name)
        prev=v

def _new_feature_vector(fill_value=0.0):
    return np.full((FEATURE_COUNT,), fill_value, dtype=np.float64)

@dataclass(frozen=True, slots=True)
class BookSnapshotInput:
    local_ts_us:int; ts_us:int; bid_px:np.ndarray; bid_sz:np.ndarray; ask_px:np.ndarray; ask_sz:np.ndarray; event_seq:int=-1
    def __post_init__(self):
        object.__setattr__(self,"local_ts_us",_require_int(self.local_ts_us,"local_ts_us",positive=True))
        object.__setattr__(self,"ts_us",_require_int(self.ts_us,"ts_us",positive=True))
        object.__setattr__(self,"event_seq",_require_int(self.event_seq,"event_seq",allow_minus_one=True))
        bp=_coerce_book_array(self.bid_px,"bid_px"); bs=_coerce_book_array(self.bid_sz,"bid_sz")
        ap=_coerce_book_array(self.ask_px,"ask_px"); a_s=_coerce_book_array(self.ask_sz,"ask_sz")
        _validate_book_order(bp,"bid"); _validate_book_order(ap,"ask")
        if bp[0]<=0 or ap[0]<=0: raise ValueError("best")
        object.__setattr__(self,"bid_px",bp); object.__setattr__(self,"bid_sz",bs); object.__setattr__(self,"ask_px",ap); object.__setattr__(self,"ask_sz",a_s)

@dataclass(frozen=True, slots=True)
class BookSummary:
    local_ts_us:int; ts_us:int; event_seq:int; best_bid:float; best_ask:float; bid_size_1:float; ask_size_1:float; mid:float; spread_bps:float; microprice:float; micro_minus_mid_bps:float; bid_depth_5bps_notional:float; ask_depth_5bps_notional:float; total_depth_5bps_notional:float; depth_imbalance_5bps:float; is_crossed:bool; update_count:int

class BookHistory:
    FIELDS=("ts_us","mid","spread_bps","microprice","bid_sz1","ask_sz1","bid_px1","ask_px1","obi_l1","obi_l3","obi_l10","depth_imbalance_5bps","total_depth_5bps_notional","bid_depth_5bps_notional","ask_depth_5bps_notional","ofi_l1","ofi_l3","ofi_l5","ofi_l10","bid_l1_add","bid_l1_rem","ask_l1_add","ask_l1_rem","bid_price_changed","ask_price_changed","spread_changed","mid_changed")
    def __init__(self, capacity=DEFAULT_HISTORY_CAPACITY):
        if capacity<=0: raise ValueError
        self.capacity=capacity; self.size=0; self.write_pos=0
        for f in self.FIELDS: setattr(self,f,np.zeros(capacity,dtype=np.int64 if f=="ts_us" else np.float64))
    def append(self, **kw):
        i=self.write_pos
        for f in self.FIELDS: getattr(self,f)[i]=kw.get(f,0.0)
        self.write_pos=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def ordered_slice(self, field_name):
        arr=getattr(self,field_name)
        if self.size==0: return arr[:0].copy()
        s=(self.write_pos-self.size)%self.capacity
        return np.concatenate((arr[s:],arr[:self.write_pos])) if s>=self.write_pos else arr[s:self.write_pos].copy()
    def ordered_ts(self): return self.ordered_slice("ts_us")
    def values_in_window(self, field_name, now_us, window_us):
        ts=self.ordered_ts(); vals=self.ordered_slice(field_name)
        if ts.size==0: return vals
        return vals[ts>=now_us-window_us]

class BookState:
    def __init__(self, history_capacity=DEFAULT_HISTORY_CAPACITY):
        self.history=BookHistory(history_capacity); self.reset()
    def reset(self):
        self.update_count=0; self.last_local_ts_us=None; self.last_snapshot=None
        self.current_bid_px=np.zeros(BOOK_DEPTH); self.current_bid_sz=np.zeros(BOOK_DEPTH); self.current_ask_px=np.zeros(BOOK_DEPTH); self.current_ask_sz=np.zeros(BOOK_DEPTH)
        self.previous_bid_px=np.zeros(BOOK_DEPTH); self.previous_bid_sz=np.zeros(BOOK_DEPTH); self.previous_ask_px=np.zeros(BOOK_DEPTH); self.previous_ask_sz=np.zeros(BOOK_DEPTH)
        self.first_mid_ts_us=None; self.last_mid_change_ts_us=None; self.current_mid_run_length=0; self.max_mid_run_length_3000us_cached=0.0
        self.bid_size_age_start_ts_us=None; self.ask_size_age_start_ts_us=None; self.depth_stable_start_ts_us=None
    def has_book(self): return self.update_count>0
    def _mid(self): return k.mid_price(self.current_bid_px[0],self.current_ask_px[0])
    def _spread(self): return k.spread_bps(self.current_bid_px[0],self.current_ask_px[0])
    def _micro(self): return k.microprice(self.current_bid_px[0],self.current_ask_px[0],self.current_bid_sz[0],self.current_ask_sz[0])
    def _obi(self,n):
        b=float(np.sum(self.current_bid_sz[:n])); a=float(np.sum(self.current_ask_sz[:n])); d=b+a
        return 0.0 if d<=0 else (b-a)/d
    def _notional5(self,side):
        if side=="bid": return float(k.notional_depth_within_bps(self.current_bid_px,self.current_bid_sz,self.current_bid_px[0],5.0,True))
        return float(k.notional_depth_within_bps(self.current_ask_px,self.current_ask_sz,self.current_ask_px[0],5.0,False))
    def apply_snapshot(self,snapshot):
        if self.last_local_ts_us is not None and snapshot.local_ts_us<self.last_local_ts_us: raise ValueError
        self.previous_bid_px[:]=self.current_bid_px; self.previous_bid_sz[:]=self.current_bid_sz; self.previous_ask_px[:]=self.current_ask_px; self.previous_ask_sz[:]=self.current_ask_sz
        self.current_bid_px[:]=snapshot.bid_px; self.current_bid_sz[:]=snapshot.bid_sz; self.current_ask_px[:]=snapshot.ask_px; self.current_ask_sz[:]=snapshot.ask_sz
        self.last_local_ts_us=snapshot.local_ts_us; self.last_snapshot=snapshot; self.update_count+=1
        mid=self._mid(); spread=self._spread(); micro=self._micro(); mm=((micro/mid)-1)*10000 if mid>0 and micro>0 else 0.0
        if self.first_mid_ts_us is None: self.first_mid_ts_us=snapshot.local_ts_us
        prev_mid=k.mid_price(self.previous_bid_px[0],self.previous_ask_px[0])
        mid_changed=float(self.update_count>1 and abs(mid-prev_mid)>1e-12)
        if mid_changed: self.last_mid_change_ts_us=snapshot.local_ts_us
        if self.last_mid_change_ts_us is None: self.last_mid_change_ts_us=snapshot.local_ts_us
        if self.bid_size_age_start_ts_us is None or self.current_bid_px[0]!=self.previous_bid_px[0] or self.current_bid_sz[0]!=self.previous_bid_sz[0]: self.bid_size_age_start_ts_us=snapshot.local_ts_us
        if self.ask_size_age_start_ts_us is None or self.current_ask_px[0]!=self.previous_ask_px[0] or self.current_ask_sz[0]!=self.previous_ask_sz[0]: self.ask_size_age_start_ts_us=snapshot.local_ts_us
        if self.depth_stable_start_ts_us is None or mid_changed or self.current_bid_sz[0]!=self.previous_bid_sz[0] or self.current_ask_sz[0]!=self.previous_ask_sz[0]: self.depth_stable_start_ts_us=snapshot.local_ts_us
        ofis=[0.0]*10
        if self.update_count>1:
            for i in range(10):
                cbp,pbp=self.current_bid_px[i],self.previous_bid_px[i]; cas,pas=self.current_ask_px[i],self.previous_ask_px[i]
                cbs,pbs=self.current_bid_sz[i],self.previous_bid_sz[i]; css,pss=self.current_ask_sz[i],self.previous_ask_sz[i]
                b=(cbs if cbp>pbp else (-pbs if cbp<pbp else cbs-pbs))
                a=(-css if cas<pas else (pss if cas>pas else pss-css))
                ofis[i]=b+a
        ofi1=ofis[0]; ofi3=sum(ofis[:3]); ofi5=sum(ofis[:5]); ofi10=sum(ofis[:10])
        bid5=self._notional5("bid"); ask5=self._notional5("ask"); tot5=bid5+ask5; di5=0.0 if tot5<=0 else (bid5-ask5)/tot5
        self.history.append(ts_us=snapshot.local_ts_us,mid=mid,spread_bps=spread,microprice=micro,bid_sz1=self.current_bid_sz[0],ask_sz1=self.current_ask_sz[0],bid_px1=self.current_bid_px[0],ask_px1=self.current_ask_px[0],obi_l1=self._obi(1),obi_l3=self._obi(3),obi_l10=self._obi(10),depth_imbalance_5bps=di5,total_depth_5bps_notional=tot5,bid_depth_5bps_notional=bid5,ask_depth_5bps_notional=ask5,ofi_l1=ofi1,ofi_l3=ofi3,ofi_l5=ofi5,ofi_l10=ofi10,bid_l1_add=max(ofi1,0.0),bid_l1_rem=max(-ofi1,0.0),ask_l1_add=max(ofi1,0.0),ask_l1_rem=max(-ofi1,0.0),bid_price_changed=float(self.current_bid_px[0]!=self.previous_bid_px[0]),ask_price_changed=float(self.current_ask_px[0]!=self.previous_ask_px[0]),spread_changed=float(self.update_count>1 and self._spread()!=k.spread_bps(self.previous_bid_px[0],self.previous_ask_px[0])),mid_changed=mid_changed)
        return self.current_summary()
    def current_summary(self):
        if not self.has_book(): raise ValueError
        mid=self._mid(); spread=self._spread(); micro=self._micro(); bid5=self._notional5("bid"); ask5=self._notional5("ask"); t=bid5+ask5
        return BookSummary(self.last_snapshot.local_ts_us,self.last_snapshot.ts_us,self.last_snapshot.event_seq,self.current_bid_px[0],self.current_ask_px[0],self.current_bid_sz[0],self.current_ask_sz[0],mid,spread,micro,((micro/mid)-1)*10000 if mid>0 and micro>0 else 0.0,bid5,ask5,t,0.0 if t<=0 else (bid5-ask5)/t,self.current_bid_px[0]>self.current_ask_px[0],self.update_count)
    def fill_book_features(self,out):
        if not self.has_book(): raise ValueError
        if np.asarray(out).shape!=(FEATURE_COUNT,): raise ValueError
        now=self.last_snapshot.local_ts_us; s=self.current_summary(); eps=1e-12
        def w(name,w): return self.history.values_in_window(name,now,w)
        def rsum(name,wus): return float(np.sum(w(name,wus)))
        def rmean(name,wus): v=w(name,wus); return float(np.mean(v)) if v.size else 0.0
        def rstd(name,wus): v=w(name,wus); return float(np.std(v)) if v.size else 0.0
        def rate(name,wus): return rsum(name,wus)/(wus/1e6)
        def setf(name,val):
            out[feature_spec_by_name(name).index]=0.0 if not np.isfinite(val) else float(val)
        for n in BOOK_FEATURE_NAMES: setf(n,0.0)
        setf("spread_bps",s.spread_bps); setf("bsz1",s.bid_size_1); setf("asz1",s.ask_size_1); setf("micro_minus_mid_bps",s.micro_minus_mid_bps)
        setf("bid_l1_notional_usd",self.current_bid_px[0]*self.current_bid_sz[0]); setf("ask_l1_notional_usd",self.current_ask_px[0]*self.current_ask_sz[0])
        setf("bid_depth_notional_5bps",s.bid_depth_5bps_notional); setf("ask_depth_notional_5bps",s.ask_depth_5bps_notional); setf("total_depth_notional_5bps",s.total_depth_5bps_notional)
        setf("obi_l1",self._obi(1)); setf("obi_l10",self._obi(10)); setf("obi_l3_mean_500000us",rmean("obi_l3",WINDOW_500MS_US)); setf("obi_l3_mean_1000000us",rmean("obi_l3",WINDOW_1000MS_US))
        setf("ofi_l1",self.history.ordered_slice("ofi_l1")[-1]); setf("ofi_l3",self.history.ordered_slice("ofi_l3")[-1]); setf("ofi_l5",self.history.ordered_slice("ofi_l5")[-1])
        setf("time_since_trade_us",0.0)  # placeholder until trade_state/engine owns this
        setf("time_since_mid_change_us",now-(self.last_mid_change_ts_us or now)); setf("best_bid_size_age_us",now-(self.bid_size_age_start_ts_us or now)); setf("best_ask_size_age_us",now-(self.ask_size_age_start_ts_us or now))
        return out
    def book_feature_vector(self):
        out=_new_feature_vector(0.0); return self.fill_book_features(out)

def book_owned_feature_names(): return BOOK_FEATURE_NAMES

def book_owned_feature_indices(): return BOOK_FEATURE_INDICES

__all__=["BOOK_DEPTH","MAX_EMITTED_DEPTH","BID_SIDE_CODE","ASK_SIDE_CODE","BOOK_WINDOWS_US","DEFAULT_HISTORY_CAPACITY","BOOK_FEATURE_INDICES","BOOK_FEATURE_NAMES","BookSnapshotInput","BookSummary","BookHistory","BookState","book_owned_feature_names","book_owned_feature_indices"]
