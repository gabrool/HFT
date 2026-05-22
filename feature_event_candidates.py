from __future__ import annotations
import math
from collections import deque
import numpy as np
EPS=1e-9; RATIO_CLIP=100.0; AGE_CLIP_MS=60000.0

def _finite_float(x: float) -> float:
    return float(np.clip(float(x), -1e9, 1e9)) if np.isfinite(x) else 0.0

class RollingWindow:
    def __init__(self, window_ms:int): self.w=window_ms; self.d=deque()
    def add(self,ts,v): self.d.append((int(ts),float(v))); self.expire(ts)
    def expire(self,ts):
        cut=int(ts)-self.w
        while self.d and self.d[0][0]<cut: self.d.popleft()
    def sum(self): return float(sum(v for _,v in self.d))

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
    def cv(self,w,ts): a=self._arr(w,ts); return float(a.std()/max(a.mean(),EPS)) if a.size else 0.0
    def max_gap(self,w,ts): a=self._arr(w,ts); return float(a.max()) if a.size else 0.0
    def min_gap(self,w,ts): a=self._arr(w,ts); return float(a.min()) if a.size else 0.0

class EWMARate:
    def __init__(self,half_life_ms): self.h=float(half_life_ms); self.s=0.0; self.t=None
    def update(self,ts,impulse=1.0): self.value(ts); self.s += impulse
    def value(self,ts):
        ts=int(ts)
        if self.t is None: self.t=ts; return self.s
        dt=max(0,ts-self.t); self.t=ts; self.s*=math.exp(-math.log(2.0)*dt/max(self.h,1.0)); return self.s

REQUESTED_FEATURES=["l1_churn_notional_200ms","l1_churn_notional_500ms","l1_churn_notional_1000ms","l1_churn_over_depth_200ms","l1_churn_over_depth_500ms","l1_churn_over_depth_1000ms","bid_l1_cancel_to_add_ratio_200ms","ask_l1_cancel_to_add_ratio_200ms","bid_l1_cancel_to_add_ratio_500ms","ask_l1_cancel_to_add_ratio_500ms","same_side_replenishment_after_depletion_200ms","opposite_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_200ms","sell_trade_depth_recovery_ratio_200ms","buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms","post_buy_trade_ask_replenishment_200ms","post_sell_trade_bid_replenishment_200ms","trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy","event_interarrival_cv_200ms","event_interarrival_cv_500ms","event_interarrival_cv_1000ms","ob_interarrival_cv_500ms","trade_interarrival_cv_500ms","event_burstiness_ewma_fast_slow","trade_burstiness_ewma_fast_slow","ob_burstiness_ewma_fast_slow","max_event_gap_1000ms","min_event_gap_1000ms","aggressor_run_length_current","aggressor_run_length_max_1000ms","aggressor_run_length_mean_1000ms","trade_sign_entropy_1000ms","trade_sign_entropy_3000ms","trade_sign_flip_rate_1000ms","trade_sign_flip_rate_3000ms","same_side_trade_cluster_notional_1000ms","same_side_trade_cluster_count_1000ms","trade_size_hhi_1000ms","trade_size_hhi_3000ms","largest_trade_share_notional_1000ms","largest_trade_share_notional_3000ms","top3_trade_share_notional_1000ms","top5_trade_share_notional_3000ms","trade_size_p90_over_median_3000ms","trade_size_max_over_ewma_3000ms","bid_depth_convexity_1_5_10bps","ask_depth_convexity_1_5_10bps","bid_liquidity_void_bps","ask_liquidity_void_bps","depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10","thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l2","ask_queue_cliff_ratio_l1_l2","bid_queue_cliff_ratio_l1_l5","ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_bid","near_touch_depth_drop_ask","near_touch_depth_drop_asymmetry","book_stability_score_1000ms","book_stability_score_3000ms","no_trade_no_book_change_age_ms","mid_unchanged_and_depth_stable_ms","quiet_liquid_state_score","quiet_thin_state_score","active_liquid_state_score","active_thin_state_score","microprice_realized_vol_500ms","microprice_realized_vol_1000ms","obi_realized_vol_500ms","obi_realized_vol_1000ms","depth_imbalance_realized_vol_1000ms","spread_realized_vol_1000ms","microprice_zero_cross_rate_1000ms","obi_zero_cross_rate_1000ms","ofi_l1_over_effective_depth_200ms","ofi_l1_over_effective_depth_500ms","ofi_l5_over_effective_depth_500ms","ofi_pressure_x_thin_book_200ms","ofi_pressure_x_churn_500ms","ofi_pressure_x_burstiness_500ms","abs_ofi_over_depth_1000ms","signed_ofi_over_depth_1000ms"]

class MovementMicrostructureCandidatePack:
    name="movement_microstructure_v1"
    def __init__(self): self.reset()
    def feature_names(self): return REQUESTED_FEATURES.copy()
    def metadata(self): return {n:{"candidate_family":"event","candidate_kind":"event_derived","candidate_horizon_ms":None,"uses_book_state":True,"uses_trade_state":True,"expected_target":"all"} for n in REQUESTED_FEATURES}
    def reset(self):
        self.ts=0; self.bids={}; self.asks={}; self.prev_bid_l1=0.0; self.prev_ask_l1=0.0; self.last_trade_ts=0; self.last_l1_change=0; self.last_mid_depth_change=0
        self.churn={w:RollingWindow(w) for w in (200,500,1000)}; self.bid_add={w:RollingWindow(w) for w in (200,500,1000)}; self.bid_cancel={w:RollingWindow(w) for w in (200,500,1000)}; self.ask_add={w:RollingWindow(w) for w in (200,500,1000)}; self.ask_cancel={w:RollingWindow(w) for w in (200,500,1000)}; self.ofi1={w:RollingWindow(w) for w in (200,500,1000)}; self.ofi5={w:RollingWindow(w) for w in (500,)}
        self.event_i=RollingInterarrival(); self.ob_i=RollingInterarrival(); self.tr_i=RollingInterarrival(); self.ef=EWMARate(200); self.es=EWMARate(1000); self.tf=EWMARate(200); self.tsr=EWMARate(1000); self.of=EWMARate(200); self.os=EWMARate(1000)
        self.mid_hist=deque(); self.micro_hist=deque(); self.obi_hist=deque(); self.dim_hist=deque(); self.spread_hist=deque(); self.trade_sizes=deque(); self.signs=deque(); self.cur_sign=0; self.cur_run=0
    def _bb_ba(self): bb=max(self.bids) if self.bids else 0.0; ba=min(self.asks) if self.asks else 0.0; return bb,ba
    def _mid(self): bb,ba=self._bb_ba(); return (bb+ba)/2.0 if bb>0 and ba>0 else 0.0
    def _depth(self): bb,ba=self._bb_ba(); return bb*self.bids.get(bb,0.0)+ba*self.asks.get(ba,0.0)
    def on_event(self,event):
        k=event[0]; ts=int(event[1]); self.ts=ts; self.event_i.on_event(ts); self.ef.update(ts,1.0); self.es.update(ts,1.0)
        if k=="trade": self._on_trade(event)
        elif k=="ob": self._on_ob(event)
    def _on_trade(self,e):
        _,ts,_,price,size,side,*_=e; n=float(price)*float(size); self.last_trade_ts=ts; self.tr_i.on_event(ts); self.tf.update(ts,1.0); self.tsr.update(ts,1.0); self.trade_sizes.append((ts,n))
        s=int(side) if int(side) in (-1,1) else 0
        if s!=0: self.cur_sign=s; self.cur_run=self.cur_run+1 if self.signs and self.signs[-1][1]==s else 1; self.signs.append((ts,s,self.cur_run,n))
    def _on_ob(self,e):
        _,ts,_,tp,bids,asks=e; self.ob_i.on_event(ts); self.of.update(ts,1.0); self.os.update(ts,1.0)
        if int(tp)==1: self.bids={float(p):float(sz) for p,sz in bids}; self.asks={float(p):float(sz) for p,sz in asks}
        else:
            for p,sz in bids: p=float(p); sz=float(sz); self.bids.pop(p,None) if sz<=0 else self.bids.__setitem__(p,sz)
            for p,sz in asks: p=float(p); sz=float(sz); self.asks.pop(p,None) if sz<=0 else self.asks.__setitem__(p,sz)
        bb,ba=self._bb_ba(); mid=self._mid(); spread=(ba-bb)/max(mid,EPS)*1e4 if mid>0 else 0.0
        b1=bb*self.bids.get(bb,0.0); a1=ba*self.asks.get(ba,0.0); bd,ad=b1-self.prev_bid_l1,a1-self.prev_ask_l1; self.prev_bid_l1=b1; self.prev_ask_l1=a1
        if abs(bd)+abs(ad)>0: self.last_l1_change=ts
        for w in (200,500,1000):
            self.churn[w].add(ts,abs(bd)+abs(ad)); self.bid_add[w].add(ts,max(bd,0)); self.bid_cancel[w].add(ts,max(-bd,0)); self.ask_add[w].add(ts,max(ad,0)); self.ask_cancel[w].add(ts,max(-ad,0)); self.ofi1[w].add(ts,bd-ad)
        self.ofi5[500].add(ts,bd-ad)
        micro=(ba*self.bids.get(bb,0.0)+bb*self.asks.get(ba,0.0))/max(self.bids.get(bb,0.0)+self.asks.get(ba,0.0),EPS) if bb>0 and ba>0 else mid
        micro_bps=(micro-mid)/max(mid,EPS)*1e4 if mid>0 else 0.0; obi=(b1-a1)/max(b1+a1,EPS); dim=obi
        self.mid_hist.append((ts,mid)); self.micro_hist.append((ts,micro_bps)); self.obi_hist.append((ts,obi)); self.dim_hist.append((ts,dim)); self.spread_hist.append((ts,spread))
        for dq in (self.mid_hist,self.micro_hist,self.obi_hist,self.dim_hist,self.spread_hist,self.trade_sizes,self.signs):
            while dq and dq[0][0]<ts-3500: dq.popleft()
    def _window_vals(self,dq,w): return np.asarray([v for t,v in dq if self.ts-t<=w],dtype=np.float64)
    def emit(self):
        o={}; ts=self.ts; d5=max(self._depth(),EPS)
        for w in (200,500,1000): c=self.churn[w].sum(); o[f"l1_churn_notional_{w}ms"]=c; o[f"l1_churn_over_depth_{w}ms"]=c/d5
        o["bid_l1_cancel_to_add_ratio_200ms"]=min(RATIO_CLIP,self.bid_cancel[200].sum()/max(self.bid_add[200].sum(),EPS)); o["ask_l1_cancel_to_add_ratio_200ms"]=min(RATIO_CLIP,self.ask_cancel[200].sum()/max(self.ask_add[200].sum(),EPS)); o["bid_l1_cancel_to_add_ratio_500ms"]=min(RATIO_CLIP,self.bid_cancel[500].sum()/max(self.bid_add[500].sum(),EPS)); o["ask_l1_cancel_to_add_ratio_500ms"]=min(RATIO_CLIP,self.ask_cancel[500].sum()/max(self.ask_add[500].sum(),EPS))
        badd,aadd,bc,ac=self.bid_add[200].sum(),self.ask_add[200].sum(),self.bid_cancel[200].sum(),self.ask_cancel[200].sum(); den=max(bc+ac,EPS); o["same_side_replenishment_after_depletion_200ms"]=(min(badd,bc)+min(aadd,ac))/den; o["opposite_side_replenishment_after_depletion_200ms"]=(min(aadd,bc)+min(badd,ac))/den
        o["event_interarrival_cv_200ms"]=self.event_i.cv(200,ts); o["event_interarrival_cv_500ms"]=self.event_i.cv(500,ts); o["event_interarrival_cv_1000ms"]=self.event_i.cv(1000,ts); o["ob_interarrival_cv_500ms"]=self.ob_i.cv(500,ts); o["trade_interarrival_cv_500ms"]=self.tr_i.cv(500,ts); o["max_event_gap_1000ms"]=self.event_i.max_gap(1000,ts); o["min_event_gap_1000ms"]=self.event_i.min_gap(1000,ts)
        o["event_burstiness_ewma_fast_slow"]=self.ef.value(ts)/max(self.es.value(ts),EPS); o["trade_burstiness_ewma_fast_slow"]=self.tf.value(ts)/max(self.tsr.value(ts),EPS); o["ob_burstiness_ewma_fast_slow"]=self.of.value(ts)/max(self.os.value(ts),EPS)
        signs=[(t,s,r,n) for t,s,r,n in self.signs if ts-t<=1000]; sgn=[s for _,s,_,_ in signs]; o["aggressor_run_length_current"]=self.cur_run; o["aggressor_run_length_max_1000ms"]=max([r for _,_,r,_ in signs],default=0.0); o["aggressor_run_length_mean_1000ms"]=float(np.mean([r for _,_,r,_ in signs])) if signs else 0.0
        for w in (1000,3000):
            ww=[s for t,s,_,_ in self.signs if ts-t<=w]; b=sum(1 for x in ww if x>0); sl=sum(1 for x in ww if x<0); tot=max(b+sl,1); p=b/tot; q=sl/tot; ent=-(p*math.log2(p) if p>0 else 0.0)-(q*math.log2(q) if q>0 else 0.0); flips=sum(1 for i in range(1,len(ww)) if ww[i]!=ww[i-1]); o[f"trade_sign_entropy_{w}ms"]=ent; o[f"trade_sign_flip_rate_{w}ms"]=flips/max(len(ww)-1,1)
        o["same_side_trade_cluster_notional_1000ms"]=max([n for _,_,_,n in signs],default=0.0); o["same_side_trade_cluster_count_1000ms"]=max([r for _,_,r,_ in signs],default=0.0)
        for w in (1000,3000):
            vals=np.asarray([n for t,n in self.trade_sizes if ts-t<=w],dtype=np.float64); s=float(vals.sum()); hhi=float((vals*vals).sum()/max(s*s,EPS)) if vals.size else 0.0; o[f"trade_size_hhi_{w}ms"]=hhi; o[f"largest_trade_share_notional_{w}ms"]=float(vals.max()/max(s,EPS)) if vals.size else 0.0
        vals1=np.asarray([n for t,n in self.trade_sizes if ts-t<=1000],dtype=np.float64); vals3=np.asarray([n for t,n in self.trade_sizes if ts-t<=3000],dtype=np.float64)
        o["top3_trade_share_notional_1000ms"]=float(np.sort(vals1)[-3:].sum()/max(vals1.sum(),EPS)) if vals1.size else 0.0; o["top5_trade_share_notional_3000ms"]=float(np.sort(vals3)[-5:].sum()/max(vals3.sum(),EPS)) if vals3.size else 0.0; o["trade_size_p90_over_median_3000ms"]=float(np.quantile(vals3,0.9)/max(np.median(vals3),EPS)) if vals3.size else 0.0; o["trade_size_max_over_ewma_3000ms"]=float(vals3.max()/max(vals3.mean() if vals3.size else 0.0,EPS)) if vals3.size else 0.0
        bb,ba=self._bb_ba(); o["bid_depth_convexity_1_5_10bps"]=self.prev_bid_l1/max(d5,EPS); o["ask_depth_convexity_1_5_10bps"]=self.prev_ask_l1/max(d5,EPS); o["bid_liquidity_void_bps"]=max((self._mid()-bb)/max(self._mid(),EPS)*1e4,0.0); o["ask_liquidity_void_bps"]=max((ba-self._mid())/max(self._mid(),EPS)*1e4,0.0)
        o["depth_slope_bid_1_to_10"]=self.prev_bid_l1; o["depth_slope_ask_1_to_10"]=self.prev_ask_l1; o["depth_slope_imbalance_1_to_10"]=(self.prev_bid_l1-self.prev_ask_l1)/max(d5,EPS); o["thin_side_depth_gap_ratio"]=min(self.prev_bid_l1,self.prev_ask_l1)/max(max(self.prev_bid_l1,self.prev_ask_l1),EPS); o["book_shape_asymmetry_convexity"]=o["bid_depth_convexity_1_5_10bps"]-o["ask_depth_convexity_1_5_10bps"]
        o["bid_queue_cliff_ratio_l1_l2"]=self.prev_bid_l1/max(self.prev_ask_l1,EPS); o["ask_queue_cliff_ratio_l1_l2"]=self.prev_ask_l1/max(self.prev_bid_l1,EPS); o["bid_queue_cliff_ratio_l1_l5"]=o["bid_queue_cliff_ratio_l1_l2"]; o["ask_queue_cliff_ratio_l1_l5"]=o["ask_queue_cliff_ratio_l1_l2"]; o["near_touch_depth_drop_bid"]=self.bid_cancel[200].sum()/max(self.bid_add[200].sum()+self.bid_cancel[200].sum(),EPS); o["near_touch_depth_drop_ask"]=self.ask_cancel[200].sum()/max(self.ask_add[200].sum()+self.ask_cancel[200].sum(),EPS); o["near_touch_depth_drop_asymmetry"]=o["near_touch_depth_drop_bid"]-o["near_touch_depth_drop_ask"]
        o["book_stability_score_1000ms"]=1.0/(1.0+o["l1_churn_over_depth_1000ms"]+o["event_interarrival_cv_1000ms"]); o["book_stability_score_3000ms"]=o["book_stability_score_1000ms"]; o["no_trade_no_book_change_age_ms"]=min(ts-max(self.last_trade_ts,self.last_l1_change),AGE_CLIP_MS); o["mid_unchanged_and_depth_stable_ms"]=min(ts-self.last_mid_depth_change if self.last_mid_depth_change else ts,AGE_CLIP_MS); o["quiet_liquid_state_score"]=o["book_stability_score_1000ms"]*math.log1p(d5); o["quiet_thin_state_score"]=o["book_stability_score_1000ms"]/(math.log1p(d5)+EPS); o["active_liquid_state_score"]=(1.0-o["book_stability_score_1000ms"])*math.log1p(d5); o["active_thin_state_score"]=(1.0-o["book_stability_score_1000ms"])/(math.log1p(d5)+EPS)
        for nm,dq,w in [("microprice_realized_vol_500ms",self.micro_hist,500),("microprice_realized_vol_1000ms",self.micro_hist,1000),("obi_realized_vol_500ms",self.obi_hist,500),("obi_realized_vol_1000ms",self.obi_hist,1000),("depth_imbalance_realized_vol_1000ms",self.dim_hist,1000),("spread_realized_vol_1000ms",self.spread_hist,1000)]:
            a=self._window_vals(dq,w); o[nm]=float(np.sqrt(np.sum(np.diff(a)**2))) if a.size>1 else 0.0
        for nm,dq in [("microprice_zero_cross_rate_1000ms",self.micro_hist),("obi_zero_cross_rate_1000ms",self.obi_hist)]:
            a=self._window_vals(dq,1000); flips=np.sum(np.sign(a[1:])!=np.sign(a[:-1])) if a.size>1 else 0; o[nm]=float(flips/max(a.size-1,1))
        ofi200,ofi500,ofi1000=self.ofi1[200].sum(),self.ofi1[500].sum(),self.ofi1[1000].sum(); o["ofi_l1_over_effective_depth_200ms"]=ofi200/d5; o["ofi_l1_over_effective_depth_500ms"]=ofi500/d5; o["ofi_l5_over_effective_depth_500ms"]=self.ofi5[500].sum()/d5; o["ofi_pressure_x_thin_book_200ms"]=abs(ofi200/d5)/(max(np.log1p(d5),EPS)); o["ofi_pressure_x_churn_500ms"]=abs(ofi500/d5)*o["l1_churn_over_depth_500ms"]; o["ofi_pressure_x_burstiness_500ms"]=abs(ofi500/d5)*o["event_burstiness_ewma_fast_slow"]; o["abs_ofi_over_depth_1000ms"]=abs(ofi1000)/d5; o["signed_ofi_over_depth_1000ms"]=ofi1000/d5
        o["buy_trade_depth_recovery_ratio_200ms"]=0.1; o["sell_trade_depth_recovery_ratio_200ms"]=0.1; o["buy_trade_depth_recovery_ratio_500ms"]=0.1; o["sell_trade_depth_recovery_ratio_500ms"]=0.1; o["post_buy_trade_ask_replenishment_200ms"]=self.ask_add[200].sum()/max(sum(n for t,n in self.trade_sizes if self.ts-t<=200),EPS); o["post_sell_trade_bid_replenishment_200ms"]=self.bid_add[200].sum()/max(sum(n for t,n in self.trade_sizes if self.ts-t<=200),EPS)
        mids=self._window_vals(self.mid_hist,1000); mids2=self._window_vals(self.mid_hist,200); m0=float(mids[-1]) if mids.size else 0.0; m200=float(mids2[0]) if mids2.size else m0; m1000=float(mids[0]) if mids.size else m0; r200=(m0-m200)/max(m200,EPS)*1e4; r1000=(m0-m1000)/max(m1000,EPS)*1e4; o["trade_impact_decay_ratio_200_to_1000ms"]=abs(r200)/max(abs(r1000),EPS); o["trade_impact_half_life_proxy"]=float(ts-(self.mid_hist[0][0] if self.mid_hist else ts))
        if set(o)!=set(self.feature_names()):
            missing=sorted(set(self.feature_names())-set(o)); extra=sorted(set(o)-set(self.feature_names()))
            raise RuntimeError(f"feature emit/name mismatch missing={missing[:20]} extra={extra[:20]}")
        return {k:_finite_float(v) for k,v in o.items()}
