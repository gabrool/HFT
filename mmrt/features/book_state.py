"""Book-state feature computation for the MMRT feature pipeline.

This module consumes normalized top-25 book snapshots and maintains causal
book-only state. It computes only BOOK-owned features from specs.py and exposes
book summaries for later engine-level cross features. It does not parse market
data files, reconstruct incremental books, compute trade features, build labels,
apply transforms, or write storage artifacts.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from mmrt.features import kernels as k
from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSource,
    feature_indices_by_source,
    feature_spec_by_name,
)

BOOK_DEPTH = 25
MAX_EMITTED_DEPTH = 20
BID_SIDE_CODE = 1
ASK_SIDE_CODE = -1

WINDOW_200MS_US = 200_000
WINDOW_500MS_US = 500_000
WINDOW_1000MS_US = 1_000_000
WINDOW_3000MS_US = 3_000_000

BOOK_WINDOWS_US = (
    WINDOW_200MS_US,
    WINDOW_500MS_US,
    WINDOW_1000MS_US,
    WINDOW_3000MS_US,
)

DEFAULT_HISTORY_CAPACITY = 16_384
FLOAT_EPS = 1e-12

BOOK_FEATURE_INDICES = feature_indices_by_source(FeatureSource.BOOK)
BOOK_FEATURE_NAMES = tuple(FEATURE_NAMES[i] for i in BOOK_FEATURE_INDICES)
BOOK_FEATURE_NAME_SET = frozenset(BOOK_FEATURE_NAMES)
ACTIVE_BOOK_FEATURES = {
    "mid_slope_bps_per_sec_1000000us",
    "time_since_mid_change_us",
    "bid_l1_notional_usd",
    "ask_l1_notional_usd",
    "total_depth_notional_5bps",
    "obi_l1",
    "ofi_l10_sum_over_depth_1000000us",
    "micro_l10_minus_mid_bps",
    "ask_depth_within_1bps",
    "depth_imbalance_within_1bps",
    "ask_l1_depletion_over_depth_200000us",
    "ask_l1_depletion_500000us",
    "bid_price_change_rate_1000000us",
    "bid_l1_depletion_1000000us",
    "bid_l1_depletion_over_depth_1000000us",
    "ask_l1_depletion_over_depth_1000000us",
    "ob_update_rate_200000us",
    "ob_update_rate_500000us",
    "bid_l1_rem_rate_over_depth_200000us",
    "depth_imbalance_5bps_slope_1000000us",
    "depth_imbalance_5bps_slope_3000000us",
    "microprice_zero_cross_rate_1000000us",
    "l1_churn_over_depth_1000000us",
    "touch_flicker_score_3000000us",
    "spread_state_transition_rate_3000000us",
    "microprice_realized_vol_1000000us",
    "best_bid_size_age_us",
    "best_ask_size_age_us",
    "near_touch_depth_drop_asymmetry",
}
assert BOOK_FEATURE_NAME_SET == ACTIVE_BOOK_FEATURES
assert BOOK_FEATURE_INDICES
assert all(FEATURE_SPECS[i].source == FeatureSource.BOOK for i in BOOK_FEATURE_INDICES)
assert MAX_EMITTED_DEPTH <= BOOK_DEPTH
assert max((s.required_book_depth for s in FEATURE_SPECS if s.source == FeatureSource.BOOK), default=0) <= MAX_EMITTED_DEPTH
assert len(BOOK_FEATURE_NAMES) == len(set(BOOK_FEATURE_NAMES))

def _require_int(value: int, name: str, *, positive: bool = False, allow_minus_one: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(name)
    if allow_minus_one and value == -1:
        return value
    if positive and value <= 0:
        raise ValueError(name)
    if not positive and value < 0:
        raise ValueError(name)
    return value

def _require_positive_capacity(capacity: int) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity")
    return capacity

def _coerce_book_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != BOOK_DEPTH or np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(name)
    return np.ascontiguousarray(arr)

def _validate_book_order(px: np.ndarray, side_name: str) -> None:
    seen_zero = False
    prev = None
    for p in px:
        if p == 0.0:
            seen_zero = True
            continue
        if seen_zero:
            raise ValueError(side_name)
        if prev is not None:
            if side_name == "bid" and p > prev:
                raise ValueError(side_name)
            if side_name == "ask" and p < prev:
                raise ValueError(side_name)
        prev = p


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0

def _safe_bps_change(new_value: float, old_value: float) -> float:
    return _finite(k.bps_change(new_value, old_value))

@dataclass(frozen=True, slots=True)
class BookSnapshotInput:
    local_ts_us: int
    ts_us: int
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    event_seq: int = -1
    def __post_init__(self) -> None:
        object.__setattr__(self, "local_ts_us", _require_int(self.local_ts_us, "local_ts_us", positive=True))
        object.__setattr__(self, "ts_us", _require_int(self.ts_us, "ts_us", positive=True))
        object.__setattr__(self, "event_seq", _require_int(self.event_seq, "event_seq", allow_minus_one=True))
        bp = _coerce_book_array(self.bid_px, "bid_px"); bs = _coerce_book_array(self.bid_sz, "bid_sz")
        ap = _coerce_book_array(self.ask_px, "ask_px"); az = _coerce_book_array(self.ask_sz, "ask_sz")
        _validate_book_order(bp, "bid"); _validate_book_order(ap, "ask")
        if bp[0] <= 0 or ap[0] <= 0: raise ValueError("best")
        if bp[0] >= ap[0]: raise ValueError("best bid must be < best ask")
        object.__setattr__(self, "bid_px", bp); object.__setattr__(self, "bid_sz", bs)
        object.__setattr__(self, "ask_px", ap); object.__setattr__(self, "ask_sz", az)

@dataclass(frozen=True, slots=True)
class BookSummary:
    local_ts_us:int; ts_us:int; event_seq:int; best_bid:float; best_ask:float; bid_size_1:float; ask_size_1:float; mid:float; spread_bps:float; microprice:float; micro_minus_mid_bps:float
    bid_depth_5bps_size:float; ask_depth_5bps_size:float; bid_depth_5bps_notional:float; ask_depth_5bps_notional:float; total_depth_5bps_size:float; total_depth_5bps_notional:float; depth_imbalance_5bps:float; is_crossed:bool; update_count:int

class BookHistory:
    FIELDS = (
        "ts_us",
        "mid",
        "microprice",
        "micro_minus_mid_bps",
        "depth_imbalance_5bps",
        "total_depth_1bps_size",
        "ofi_l1",
        "ofi_l10",
        "bid_l1_add",
        "bid_l1_rem",
        "ask_l1_add",
        "ask_l1_rem",
        "bid_price_changed",
        "ask_price_changed",
        "spread_changed",
    )
    def __init__(self, capacity: int = DEFAULT_HISTORY_CAPACITY):
        self.capacity = _require_positive_capacity(capacity); self.size = 0; self.write_pos = 0
        self._arrays = {f: np.zeros(self.capacity, dtype=np.int64 if f=="ts_us" else np.float64) for f in self.FIELDS}
    def append(self, **kwargs):
        got = set(kwargs)
        expected = set(self.FIELDS)
        missing = expected - got
        extra = got - expected
        if missing or extra:
            raise KeyError(f"book history fields mismatch missing={sorted(missing)} extra={sorted(extra)}")

        i = self.write_pos
        for f, a in self._arrays.items():
            a[i] = kwargs[f]
        self.write_pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def ordered_slice(self, field_name:str)->np.ndarray:
        arr=self._arrays[field_name]
        if self.size==0: return arr[:0].copy()
        s=(self.write_pos-self.size)%self.capacity
        return np.concatenate((arr[s:],arr[:self.write_pos])).copy() if s>=self.write_pos else arr[s:self.write_pos].copy()
    def ordered_ts(self)->np.ndarray: return self.ordered_slice("ts_us")
    def values_in_window(self, field_name:str, now_us:int, window_us:int)->np.ndarray:
        ts=self.ordered_ts(); vals=self.ordered_slice(field_name)
        return vals[ts>=now_us-window_us] if ts.size else vals
    def ts_in_window(self, now_us:int, window_us:int)->np.ndarray:
        ts=self.ordered_ts(); return ts[ts>=now_us-window_us] if ts.size else ts

class BookState:
    def __init__(self, history_capacity:int=DEFAULT_HISTORY_CAPACITY): self._history_capacity=_require_positive_capacity(history_capacity); self.history=BookHistory(self._history_capacity); self.reset()
    def reset(self):
        self.update_count=0; self.last_local_ts_us=None; self.last_snapshot=None; self.history=BookHistory(self._history_capacity)
        z=np.zeros(BOOK_DEPTH,dtype=np.float64)
        self.current_bid_px=z.copy(); self.current_bid_sz=z.copy(); self.current_ask_px=z.copy(); self.current_ask_sz=z.copy()
        self.previous_bid_px=z.copy(); self.previous_bid_sz=z.copy(); self.previous_ask_px=z.copy(); self.previous_ask_sz=z.copy()
        self.last_mid_change_ts_us=None; self.bid_size_age_start_ts_us=None; self.ask_size_age_start_ts_us=None
    def has_book(self)->bool: return self.update_count>0
    def _mid(self)->float: return _finite(k.mid_price(self.current_bid_px[0],self.current_ask_px[0]))
    def _spread_bps(self)->float: return _finite(k.spread_bps(self.current_bid_px[0],self.current_ask_px[0]))
    def _microprice_l1(self)->float: return _finite(k.microprice(self.current_bid_px[0],self.current_ask_px[0],self.current_bid_sz[0],self.current_ask_sz[0]))
    def _micro_minus_mid_bps(self)->float: return _safe_bps_change(self._microprice_l1(), self._mid())
    def _sum_size(self, side:str, n:int)->float: return float(np.sum((self.current_bid_sz if side=="bid" else self.current_ask_sz)[:n]))
    def _obi(self,n:int)->float: b=self._sum_size("bid",n); a=self._sum_size("ask",n); d=b+a; return 0.0 if d<=FLOAT_EPS else (b-a)/d
    def _depth_size_within_bps(self, side:str, bps:float)->float: code=BID_SIDE_CODE if side=="bid" else ASK_SIDE_CODE; return _finite(k.depth_within_bps(self.current_bid_px if side=="bid" else self.current_ask_px, self.current_bid_sz if side=="bid" else self.current_ask_sz, self._mid(), code, float(bps)))
    def _depth_notional_within_bps(self, side:str, bps:float)->float: code=BID_SIDE_CODE if side=="bid" else ASK_SIDE_CODE; return _finite(k.notional_depth_within_bps(self.current_bid_px if side=="bid" else self.current_ask_px, self.current_bid_sz if side=="bid" else self.current_ask_sz, self._mid(), code, float(bps)))
    def _depth_imbalance_within_bps(self,bps:float)->float: b=self._depth_size_within_bps("bid",bps); a=self._depth_size_within_bps("ask",bps); d=a+b; return 0.0 if d<=FLOAT_EPS else (b-a)/d
    def _micro_depth(self,n:int)->float:
        num=float(np.sum(self.current_ask_px[:n]*self.current_bid_sz[:n]+self.current_bid_px[:n]*self.current_ask_sz[:n])); den=float(np.sum(self.current_bid_sz[:n]+self.current_ask_sz[:n])); return 0.0 if den<=FLOAT_EPS else num/den
    def _minus_mid_bps(self,value:float)->float: return _safe_bps_change(value,self._mid())
    def _ofi_by_level(self)->np.ndarray:
        if self.update_count<=1: return np.zeros(10,dtype=np.float64)
        out=np.zeros(10,dtype=np.float64)
        for i in range(10):
            cbp,pbp=self.current_bid_px[i],self.previous_bid_px[i]; cas,pas=self.current_ask_px[i],self.previous_ask_px[i]
            cbs,pbs=self.current_bid_sz[i],self.previous_bid_sz[i]; css,pss=self.current_ask_sz[i],self.previous_ask_sz[i]
            b = cbs if cbp>pbp else (-pbs if cbp<pbp else cbs-pbs)
            a = -css if cas<pas else (pss if cas>pas else pss-css)
            out[i]=b+a
        return out
    def _l1_add_rem(self)->tuple[float,float,float,float]:
        if self.update_count<=1: return (0.0,0.0,0.0,0.0)
        bid_add=bid_rem=ask_add=ask_rem=0.0
        if self.current_bid_px[0]>self.previous_bid_px[0]: bid_add=self.current_bid_sz[0]
        elif self.current_bid_px[0]<self.previous_bid_px[0]: bid_rem=self.previous_bid_sz[0]
        else:
            d=self.current_bid_sz[0]-self.previous_bid_sz[0]; bid_add=max(d,0.0); bid_rem=max(-d,0.0)
        if self.current_ask_px[0]<self.previous_ask_px[0]: ask_add=self.current_ask_sz[0]
        elif self.current_ask_px[0]>self.previous_ask_px[0]: ask_rem=self.previous_ask_sz[0]
        else:
            d=self.current_ask_sz[0]-self.previous_ask_sz[0]; ask_add=max(d,0.0); ask_rem=max(-d,0.0)
        return bid_add,bid_rem,ask_add,ask_rem
    def apply_snapshot(self, snapshot: BookSnapshotInput) -> BookSummary:
        if not isinstance(snapshot, BookSnapshotInput):
            raise TypeError("snapshot")
        if self.last_local_ts_us is not None and snapshot.local_ts_us < self.last_local_ts_us:
            raise ValueError("local_ts_us")
        self.previous_bid_px[:] = self.current_bid_px
        self.previous_bid_sz[:] = self.current_bid_sz
        self.previous_ask_px[:] = self.current_ask_px
        self.previous_ask_sz[:] = self.current_ask_sz
        self.current_bid_px[:] = snapshot.bid_px
        self.current_bid_sz[:] = snapshot.bid_sz
        self.current_ask_px[:] = snapshot.ask_px
        self.current_ask_sz[:] = snapshot.ask_sz
        self.last_snapshot = snapshot
        self.last_local_ts_us = snapshot.local_ts_us
        self.update_count += 1

        now = snapshot.local_ts_us
        mid = self._mid()
        spread = self._spread_bps()
        micro = self._microprice_l1()
        micro_minus_mid = self._micro_minus_mid_bps()
        ofi = self._ofi_by_level()
        ofi_l1 = float(ofi[0])
        ofi_l10 = float(np.sum(ofi[:10]))
        bid_add, bid_rem, ask_add, ask_rem = self._l1_add_rem()
        first = self.update_count == 1
        bid_price_changed = 0.0 if first else float(self.current_bid_px[0] != self.previous_bid_px[0])
        ask_price_changed = 0.0 if first else float(self.current_ask_px[0] != self.previous_ask_px[0])
        prev_spread = spread if first else _finite(k.spread_bps(self.previous_bid_px[0], self.previous_ask_px[0]))
        spread_changed = 0.0 if first else float(abs(spread - prev_spread) > FLOAT_EPS)
        prev_mid = mid if first else _finite(k.mid_price(self.previous_bid_px[0], self.previous_ask_px[0]))
        mid_changed = 0.0 if first else float(abs(mid - prev_mid) > FLOAT_EPS)

        if self.last_mid_change_ts_us is None or mid_changed > 0:
            self.last_mid_change_ts_us = now
        if (
            self.bid_size_age_start_ts_us is None
            or self.current_bid_px[0] != self.previous_bid_px[0]
            or self.current_bid_sz[0] != self.previous_bid_sz[0]
        ):
            self.bid_size_age_start_ts_us = now
        if (
            self.ask_size_age_start_ts_us is None
            or self.current_ask_px[0] != self.previous_ask_px[0]
            or self.current_ask_sz[0] != self.previous_ask_sz[0]
        ):
            self.ask_size_age_start_ts_us = now
        total_depth_1bps = self._depth_size_within_bps("bid", 1.0) + self._depth_size_within_bps("ask", 1.0)
        bid_depth_5bps_notional = self._depth_notional_within_bps("bid", 5.0)
        ask_depth_5bps_notional = self._depth_notional_within_bps("ask", 5.0)
        total_depth_5bps_notional = bid_depth_5bps_notional + ask_depth_5bps_notional
        depth_imbalance_5bps = (
            0.0
            if total_depth_5bps_notional <= FLOAT_EPS
            else (bid_depth_5bps_notional - ask_depth_5bps_notional) / total_depth_5bps_notional
        )
        self.history.append(
            ts_us=now,
            mid=mid,
            microprice=micro,
            micro_minus_mid_bps=micro_minus_mid,
            depth_imbalance_5bps=depth_imbalance_5bps,
            total_depth_1bps_size=total_depth_1bps,
            ofi_l1=ofi_l1,
            ofi_l10=ofi_l10,
            bid_l1_add=bid_add,
            bid_l1_rem=bid_rem,
            ask_l1_add=ask_add,
            ask_l1_rem=ask_rem,
            bid_price_changed=bid_price_changed,
            ask_price_changed=ask_price_changed,
            spread_changed=spread_changed,
        )
        return self.current_summary()
    def current_summary(self)->BookSummary:
        if not self.has_book(): raise ValueError("no book")
        mid=self._mid(); b5s=self._depth_size_within_bps("bid",5.0); a5s=self._depth_size_within_bps("ask",5.0); b5n=self._depth_notional_within_bps("bid",5.0); a5n=self._depth_notional_within_bps("ask",5.0); t5s=b5s+a5s; t5n=b5n+a5n
        return BookSummary(self.last_snapshot.local_ts_us,self.last_snapshot.ts_us,self.last_snapshot.event_seq,self.current_bid_px[0],self.current_ask_px[0],self.current_bid_sz[0],self.current_ask_sz[0],mid,self._spread_bps(),self._microprice_l1(),self._micro_minus_mid_bps(),b5s,a5s,b5n,a5n,t5s,t5n,0.0 if t5n<=FLOAT_EPS else (b5n-a5n)/t5n,self.current_bid_px[0]>=self.current_ask_px[0],self.update_count)
    def _window_values(self, field_name:str, window_us:int)->np.ndarray: return self.history.values_in_window(field_name,self.last_local_ts_us,window_us)
    def _window_ts(self, window_us:int)->np.ndarray: return self.history.ts_in_window(self.last_local_ts_us,window_us)
    def _rolling_sum(self, field_name:str, window_us:int)->float: return float(np.sum(self._window_values(field_name,window_us)))
    def _rolling_mean(self, field_name:str, window_us:int)->float: v=self._window_values(field_name,window_us); return float(np.mean(v)) if v.size else 0.0
    def _rolling_slope_per_sec(self, field_name:str, window_us:int)->float:
        v=self._window_values(field_name,window_us); t=self._window_ts(window_us)
        if v.size<2: return 0.0
        i=np.where(np.isfinite(v))[0]
        if i.size<2: return 0.0
        dt=(t[i[-1]]-t[i[0]])/1e6
        return 0.0 if dt<=FLOAT_EPS else float((v[i[-1]]-v[i[0]])/dt)
    def _rolling_mid_slope_bps_per_sec(self, window_us:int)->float:
        v=self._window_values("mid",window_us); t=self._window_ts(window_us); p=np.where(np.isfinite(v)&(v>0))[0]
        if p.size<2: return 0.0
        dt=(t[p[-1]]-t[p[0]])/1e6
        return 0.0 if dt<=FLOAT_EPS else _safe_bps_change(v[p[-1]],v[p[0]])/dt
    def _rolling_update_rate(self, window_us:int)->float: return float(self._window_ts(window_us).size)/(window_us/1e6)
    def _rolling_count(self, field_name:str, window_us:int)->float: return float(self._window_values(field_name,window_us).size)
    def _rolling_realized_vol_bps(self, field_name:str, window_us:int)->float:
        v=self._window_values(field_name,window_us); v=v[np.isfinite(v)&(v>0)]
        if v.size<2: return 0.0
        r=np.array([_safe_bps_change(v[i],v[i-1]) for i in range(1,v.size)],dtype=np.float64)
        return float(np.std(r)) if r.size else 0.0
    def _zero_cross_rate(self, window_us:int)->float:
        v=self._window_values("micro_minus_mid_bps",window_us); s=np.sign(v); s=s[s!=0]
        if s.size<2: return 0.0
        return float(np.sum(s[1:]!=s[:-1]))/(window_us/1e6)
    def fill_book_features(self, out: np.ndarray) -> np.ndarray:
        if not self.has_book():
            raise ValueError
        arr = np.asarray(out)
        if arr.ndim != 1 or arr.shape[0] != FEATURE_COUNT:
            raise ValueError
        assigned = set()
        now = self.last_local_ts_us
        s = self.current_summary()
        depth = max(s.total_depth_5bps_size, FLOAT_EPS)
        bid_depth_1bps = max(self._depth_size_within_bps("bid", 1.0), FLOAT_EPS)
        ask_depth_1bps = max(self._depth_size_within_bps("ask", 1.0), FLOAT_EPS)

        def setf(name, v):
            if name not in BOOK_FEATURE_NAME_SET:
                raise KeyError(name)
            arr[feature_spec_by_name(name).index] = _finite(v)
            assigned.add(name)

        setf("mid_slope_bps_per_sec_1000000us", self._rolling_mid_slope_bps_per_sec(WINDOW_1000MS_US))
        setf("time_since_mid_change_us", now - self.last_mid_change_ts_us)
        setf("bid_l1_notional_usd", self.current_bid_px[0] * self.current_bid_sz[0])
        setf("ask_l1_notional_usd", self.current_ask_px[0] * self.current_ask_sz[0])
        setf("total_depth_notional_5bps", s.total_depth_5bps_notional)
        setf("obi_l1", self._obi(1))
        setf("ofi_l10_sum_over_depth_1000000us", self._rolling_sum("ofi_l10", WINDOW_1000MS_US) / depth)
        setf("micro_l10_minus_mid_bps", self._minus_mid_bps(self._micro_depth(10)))
        setf("ask_depth_within_1bps", ask_depth_1bps)
        setf("depth_imbalance_within_1bps", self._depth_imbalance_within_bps(1.0))
        setf("ask_l1_depletion_over_depth_200000us", self._rolling_sum("ask_l1_rem", WINDOW_200MS_US) / ask_depth_1bps)
        setf("ask_l1_depletion_500000us", self._rolling_sum("ask_l1_rem", WINDOW_500MS_US))
        setf("bid_price_change_rate_1000000us", self._rolling_sum("bid_price_changed", WINDOW_1000MS_US))
        setf("bid_l1_depletion_1000000us", self._rolling_sum("bid_l1_rem", WINDOW_1000MS_US))
        setf("bid_l1_depletion_over_depth_1000000us", self._rolling_sum("bid_l1_rem", WINDOW_1000MS_US) / bid_depth_1bps)
        setf("ask_l1_depletion_over_depth_1000000us", self._rolling_sum("ask_l1_rem", WINDOW_1000MS_US) / ask_depth_1bps)
        setf("ob_update_rate_200000us", self._rolling_update_rate(WINDOW_200MS_US))
        setf("ob_update_rate_500000us", self._rolling_update_rate(WINDOW_500MS_US))
        setf("bid_l1_rem_rate_over_depth_200000us", self._rolling_sum("bid_l1_rem", WINDOW_200MS_US) / (0.2 * bid_depth_1bps))
        setf("depth_imbalance_5bps_slope_1000000us", self._rolling_slope_per_sec("depth_imbalance_5bps", WINDOW_1000MS_US))
        setf("depth_imbalance_5bps_slope_3000000us", self._rolling_slope_per_sec("depth_imbalance_5bps", WINDOW_3000MS_US))
        setf("microprice_zero_cross_rate_1000000us", self._zero_cross_rate(WINDOW_1000MS_US))
        setf("l1_churn_over_depth_1000000us", (self._rolling_sum("bid_l1_add", WINDOW_1000MS_US) + self._rolling_sum("bid_l1_rem", WINDOW_1000MS_US) + self._rolling_sum("ask_l1_add", WINDOW_1000MS_US) + self._rolling_sum("ask_l1_rem", WINDOW_1000MS_US)) / max(self._rolling_mean("total_depth_1bps_size", WINDOW_1000MS_US), FLOAT_EPS))
        setf("touch_flicker_score_3000000us", (self._rolling_sum("bid_price_changed", WINDOW_3000MS_US) + self._rolling_sum("ask_price_changed", WINDOW_3000MS_US)) / max(self._rolling_count("mid", WINDOW_3000MS_US), 1.0))
        setf("spread_state_transition_rate_3000000us", self._rolling_sum("spread_changed", WINDOW_3000MS_US) / 3.0)
        setf("microprice_realized_vol_1000000us", self._rolling_realized_vol_bps("microprice", WINDOW_1000MS_US))
        setf("best_bid_size_age_us", now - self.bid_size_age_start_ts_us)
        setf("best_ask_size_age_us", now - self.ask_size_age_start_ts_us)
        bd = self._rolling_sum("bid_l1_rem", WINDOW_200MS_US)
        ad = self._rolling_sum("ask_l1_rem", WINDOW_200MS_US)
        d = bd + ad
        setf("near_touch_depth_drop_asymmetry", 0.0 if d <= FLOAT_EPS else (ad - bd) / d)
        missing = BOOK_FEATURE_NAME_SET - assigned
        extra = assigned - BOOK_FEATURE_NAME_SET
        if missing or extra:
            raise RuntimeError(f"incomplete book feature assignment missing={sorted(missing)} extra={sorted(extra)}")
        return out

def book_owned_feature_names() -> tuple[str, ...]: return BOOK_FEATURE_NAMES

def book_owned_feature_indices() -> tuple[int, ...]: return BOOK_FEATURE_INDICES

__all__ = ["BOOK_DEPTH","MAX_EMITTED_DEPTH","BID_SIDE_CODE","ASK_SIDE_CODE","BOOK_WINDOWS_US","DEFAULT_HISTORY_CAPACITY","BOOK_FEATURE_INDICES","BOOK_FEATURE_NAMES","ACTIVE_BOOK_FEATURES","BookSnapshotInput","BookSummary","BookHistory","BookState","book_owned_feature_names","book_owned_feature_indices"]
