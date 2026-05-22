from __future__ import annotations
import math
from collections import deque
import numpy as np
EPS=1e-9; RATIO_CLIP=100.0; AGE_CLIP_MS=60000.0


def _finite_float(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(float(x), -1e9, 1e9))

class RollingWindow:
    def __init__(self, window_ms:int): self.w=window_ms; self.d=deque()
    def add(self,ts,v): self.d.append((int(ts),float(v))); self.expire(ts)
    def expire(self,ts):
        cut=int(ts)-self.w
        while self.d and self.d[0][0]<cut: self.d.popleft()
    def sum(self): return float(sum(v for _,v in self.d))
    def abs_sum(self): return float(sum(abs(v) for _,v in self.d))
    def count(self): return len(self.d)
    def values_array(self): return np.asarray([v for _,v in self.d],dtype=np.float64)
    def mean(self): a=self.values_array(); return float(a.mean()) if a.size else 0.0
    def std(self): a=self.values_array(); return float(a.std()) if a.size else 0.0
    def max(self): a=self.values_array(); return float(a.max()) if a.size else 0.0
    def min(self): a=self.values_array(); return float(a.min()) if a.size else 0.0

class RollingInterarrival:
    def __init__(self): self.last=None; self.gaps=deque()
    def on_event(self,ts):
        ts=int(ts)
        if self.last is not None: self.gaps.append((ts, ts-self.last))
        self.last=ts
    def _arr(self,w,ts):
        cut=ts-w
        while self.gaps and self.gaps[0][0]<cut: self.gaps.popleft()
        return np.asarray([g for _,g in self.gaps],dtype=np.float64)
    def cv(self,w,ts):
        a=self._arr(w,ts)
        return float(a.std()/max(a.mean(),EPS)) if a.size else 0.0
    def max_gap(self,w,ts): a=self._arr(w,ts); return float(a.max()) if a.size else 0.0
    def min_gap(self,w,ts): a=self._arr(w,ts); return float(a.min()) if a.size else 0.0

class EWMARate:
    def __init__(self,half_life_ms): self.h=float(half_life_ms); self.s=0.0; self.t=None
    def update(self,ts,impulse=1.0):
        self.value(ts); self.s += impulse
    def value(self,ts):
        ts=int(ts)
        if self.t is None: self.t=ts; return self.s
        dt=max(0,ts-self.t); self.t=ts
        self.s*=math.exp(-math.log(2.0)*dt/max(self.h,1.0)); return self.s

class MovementMicrostructureCandidatePack:
    name="movement_microstructure_v1"
    def __init__(self): self.reset()
    def feature_names(self): return list(self.metadata().keys())
    def metadata(self):
        names = [
"l1_churn_notional_200ms","l1_churn_notional_500ms","l1_churn_notional_1000ms","l1_churn_over_depth_200ms","l1_churn_over_depth_500ms","l1_churn_over_depth_1000ms","bid_l1_cancel_to_add_ratio_200ms","ask_l1_cancel_to_add_ratio_200ms","bid_l1_cancel_to_add_ratio_500ms","ask_l1_cancel_to_add_ratio_500ms","same_side_replenishment_after_depletion_200ms","opposite_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_200ms","sell_trade_depth_recovery_ratio_200ms","buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms","post_buy_trade_ask_replenishment_200ms","post_sell_trade_bid_replenishment_200ms","trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy","event_interarrival_cv_200ms","event_interarrival_cv_500ms","event_interarrival_cv_1000ms","ob_interarrival_cv_500ms","trade_interarrival_cv_500ms","event_burstiness_ewma_fast_slow","trade_burstiness_ewma_fast_slow","ob_burstiness_ewma_fast_slow","max_event_gap_1000ms","min_event_gap_1000ms","aggressor_run_length_current","aggressor_run_length_max_1000ms","aggressor_run_length_mean_1000ms","trade_sign_entropy_1000ms","trade_sign_entropy_3000ms","trade_sign_flip_rate_1000ms","trade_sign_flip_rate_3000ms","same_side_trade_cluster_notional_1000ms","same_side_trade_cluster_count_1000ms","trade_size_hhi_1000ms","trade_size_hhi_3000ms","largest_trade_share_notional_1000ms","largest_trade_share_notional_3000ms","top3_trade_share_notional_1000ms","top5_trade_share_notional_3000ms","trade_size_p90_over_median_3000ms","trade_size_max_over_ewma_3000ms","bid_depth_convexity_1_5_10bps","ask_depth_convexity_1_5_10bps","bid_liquidity_void_bps","ask_liquidity_void_bps","depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10","thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l2","ask_queue_cliff_ratio_l1_l2","bid_queue_cliff_ratio_l1_l5","ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_bid","near_touch_depth_drop_ask","near_touch_depth_drop_asymmetry","book_stability_score_1000ms","book_stability_score_3000ms","no_trade_no_book_change_age_ms","mid_unchanged_and_depth_stable_ms","quiet_liquid_state_score","quiet_thin_state_score","active_liquid_state_score","active_thin_state_score","microprice_realized_vol_500ms","microprice_realized_vol_1000ms","obi_realized_vol_500ms","obi_realized_vol_1000ms","depth_imbalance_realized_vol_1000ms","spread_realized_vol_1000ms","microprice_zero_cross_rate_1000ms","obi_zero_cross_rate_1000ms","ofi_l1_over_effective_depth_200ms","ofi_l1_over_effective_depth_500ms","ofi_l5_over_effective_depth_500ms","ofi_pressure_x_thin_book_200ms","ofi_pressure_x_churn_500ms","ofi_pressure_x_burstiness_500ms","abs_ofi_over_depth_1000ms","signed_ofi_over_depth_1000ms"]
        return {n:{"candidate_family":n.split("_")[0],"candidate_kind":"event_derived","candidate_horizon_ms":(200 if "200ms" in n else 500 if "500ms" in n else 1000 if "1000ms" in n else 3000 if "3000ms" in n else None),"uses_book_state":True,"uses_trade_state":("trade" in n or "aggressor" in n),"expected_target":"all"} for n in names}
    def reset(self):
        self.ts=0; self.bids={}; self.asks={}; self.prev_bid_l1=0.0; self.prev_ask_l1=0.0; self.last_trade_ts=0; self.last_l1_change=0; self.last_mid_change=0
        self.churn={w:RollingWindow(w) for w in (200,500,1000)}; self.bid_add={w:RollingWindow(w) for w in (200,500,1000)}; self.bid_cancel={w:RollingWindow(w) for w in (200,500,1000)}; self.ask_add={w:RollingWindow(w) for w in (200,500,1000)}; self.ask_cancel={w:RollingWindow(w) for w in (200,500,1000)}
        self.ofi1={w:RollingWindow(w) for w in (200,500,1000)}; self.ofi5={w:RollingWindow(w) for w in (500,)}
        self.event_i=RollingInterarrival(); self.ob_i=RollingInterarrival(); self.tr_i=RollingInterarrival(); self.ef=EWMARate(200); self.es=EWMARate(1000); self.tf=EWMARate(200); self.tsr=EWMARate(1000); self.of=EWMARate(200); self.os=EWMARate(1000)
        self.mid_hist=deque(); self.micro_hist=deque(); self.obi_hist=deque(); self.dim_hist=deque(); self.spread_hist=deque(); self.trade_sizes=deque(); self.signs=deque(); self.cur_sign=0; self.cur_run=0
    def on_event(self,event):
        k=event[0]; ts=int(event[1]); self.ts=ts; self.event_i.on_event(ts); self.ef.update(ts,1.0); self.es.update(ts,1.0)
        if k=="trade": self._on_trade(event)
        elif k=="ob": self._on_ob(event)
    def _on_trade(self,e):
        _,ts,seq,price,size,side,*_=e; n=float(price)*float(size); self.last_trade_ts=ts; self.tr_i.on_event(ts); self.tf.update(ts,1.0); self.tsr.update(ts,1.0); self.trade_sizes.append((ts,n))
        s=int(side) if int(side) in (-1,1) else 0
        if s!=0:
            prev=self.cur_sign; self.cur_sign=s; self.cur_run = self.cur_run+1 if s==prev else 1; self.signs.append((ts,s,self.cur_run,n))
    def _on_ob(self,e):
        _,ts,seq,tp,bids,asks=e; self.ob_i.on_event(ts); self.of.update(ts,1.0); self.os.update(ts,1.0)
        if int(tp)==1: self.bids={float(p):float(sz) for p,sz in bids}; self.asks={float(p):float(sz) for p,sz in asks}
        else:
            for p,sz in bids:
                p=float(p); sz=float(sz); self.bids.pop(p,None) if sz<=0 else self.bids.__setitem__(p,sz)
            for p,sz in asks:
                p=float(p); sz=float(sz); self.asks.pop(p,None) if sz<=0 else self.asks.__setitem__(p,sz)
        bb=max(self.bids) if self.bids else 0.0; ba=min(self.asks) if self.asks else 0.0
        mid=(bb+ba)/2.0 if bb>0 and ba>0 else 0.0; spread=(ba-bb)/max(mid,EPS)*1e4 if mid>0 else 0.0
        b1=bb*self.bids.get(bb,0.0); a1=ba*self.asks.get(ba,0.0)
        bd,ad=b1-self.prev_bid_l1,a1-self.prev_ask_l1; self.prev_bid_l1=b1; self.prev_ask_l1=a1
        if abs(bd)+abs(ad)>0: self.last_l1_change=ts
        for w in (200,500,1000):
            self.churn[w].add(ts,abs(bd)+abs(ad)); self.bid_add[w].add(ts,max(bd,0)); self.bid_cancel[w].add(ts,max(-bd,0)); self.ask_add[w].add(ts,max(ad,0)); self.ask_cancel[w].add(ts,max(-ad,0)); self.ofi1[w].add(ts,max(bd,0)-max(-bd,0)-max(ad,0)+max(-ad,0))
        self.ofi5[500].add(ts,self.ofi1[500].d[-1][1])
        if len(self.mid_hist)==0 or abs(mid-self.mid_hist[-1][1])>0: self.last_mid_change=ts
        self.mid_hist.append((ts,mid)); self.micro_hist.append((ts,0.0)); self.obi_hist.append((ts,(b1-a1)/max(b1+a1,EPS))); self.dim_hist.append((ts,0.0)); self.spread_hist.append((ts,spread))
        for dq in (self.mid_hist,self.micro_hist,self.obi_hist,self.dim_hist,self.spread_hist,self.trade_sizes,self.signs):
            while dq and dq[0][0]<ts-3500: dq.popleft()
    def emit(self):
        o={n:0.0 for n in self.feature_names()}; ts=self.ts
        d5=max(self.prev_bid_l1+self.prev_ask_l1,EPS)
        for w in (200,500,1000):
            c=self.churn[w].sum(); o[f"l1_churn_notional_{w}ms"]=c; o[f"l1_churn_over_depth_{w}ms"]=c/d5
        o["bid_l1_cancel_to_add_ratio_200ms"]=min(RATIO_CLIP,self.bid_cancel[200].sum()/max(self.bid_add[200].sum(),EPS));o["ask_l1_cancel_to_add_ratio_200ms"]=min(RATIO_CLIP,self.ask_cancel[200].sum()/max(self.ask_add[200].sum(),EPS));o["bid_l1_cancel_to_add_ratio_500ms"]=min(RATIO_CLIP,self.bid_cancel[500].sum()/max(self.bid_add[500].sum(),EPS));o["ask_l1_cancel_to_add_ratio_500ms"]=min(RATIO_CLIP,self.ask_cancel[500].sum()/max(self.ask_add[500].sum(),EPS))
        o["event_interarrival_cv_200ms"]=self.event_i.cv(200,ts); o["event_interarrival_cv_500ms"]=self.event_i.cv(500,ts); o["event_interarrival_cv_1000ms"]=self.event_i.cv(1000,ts); o["ob_interarrival_cv_500ms"]=self.ob_i.cv(500,ts); o["trade_interarrival_cv_500ms"]=self.tr_i.cv(500,ts); o["max_event_gap_1000ms"]=self.event_i.max_gap(1000,ts); o["min_event_gap_1000ms"]=self.event_i.min_gap(1000,ts)
        o["event_burstiness_ewma_fast_slow"]=self.ef.value(ts)/max(self.es.value(ts),EPS); o["trade_burstiness_ewma_fast_slow"]=self.tf.value(ts)/max(self.tsr.value(ts),EPS); o["ob_burstiness_ewma_fast_slow"]=self.of.value(ts)/max(self.os.value(ts),EPS)
        o["aggressor_run_length_current"]=self.cur_run
        o["no_trade_no_book_change_age_ms"]=min(ts-max(self.last_trade_ts,self.last_l1_change),AGE_CLIP_MS); o["mid_unchanged_and_depth_stable_ms"]=min(ts-self.last_mid_change,AGE_CLIP_MS)
        o["ofi_l1_over_effective_depth_200ms"]=self.ofi1[200].sum()/d5; o["ofi_l1_over_effective_depth_500ms"]=self.ofi1[500].sum()/d5; o["ofi_l5_over_effective_depth_500ms"]=self.ofi5[500].sum()/d5
        o["abs_ofi_over_depth_1000ms"]=abs(self.ofi1[1000].sum())/d5; o["signed_ofi_over_depth_1000ms"]=self.ofi1[1000].sum()/d5
        return {k:_finite_float(v) for k,v in o.items()}
