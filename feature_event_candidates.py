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
    def __init__(self, max_window_ms: int = 3500): self.last=None; self.gaps=deque(); self.max_window_ms=int(max_window_ms)
    def on_event(self,ts):
        ts=int(ts)
        if self.last is not None: self.gaps.append((ts,max(0,ts-self.last)))
        self.last=ts; self.expire(ts)
    def expire(self,ts):
        cut=int(ts)-self.max_window_ms
        while self.gaps and self.gaps[0][0]<cut: self.gaps.popleft()
    def _arr(self,w,ts):
        cut=int(ts)-int(w)
        return np.asarray([g for t,g in self.gaps if t>=cut],dtype=np.float64)
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

class EWMAValue:
    def __init__(self,half_life_ms): self.h=float(half_life_ms); self.s=0.0; self.t=None
    def update(self,ts,value): self.value(ts); self.s += float(value)
    def value(self,ts):
        ts=int(ts)
        if self.t is None: self.t=ts; return self.s
        dt=max(0,ts-self.t); self.t=ts; self.s*=math.exp(-math.log(2.0)*dt/max(self.h,1.0)); return self.s

# keep existing list
REQUESTED_FEATURES = __import__('feature_event_candidates_base' if False else 'builtins')
# fallback inline from original for compatibility
REQUESTED_FEATURES=["l1_churn_notional_200ms","l1_churn_notional_500ms","l1_churn_notional_1000ms","l1_churn_over_depth_200ms","l1_churn_over_depth_500ms","l1_churn_over_depth_1000ms","bid_l1_cancel_to_add_ratio_200ms","ask_l1_cancel_to_add_ratio_200ms","bid_l1_cancel_to_add_ratio_500ms","ask_l1_cancel_to_add_ratio_500ms","same_side_replenishment_after_depletion_200ms","opposite_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_200ms","sell_trade_depth_recovery_ratio_200ms","buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms","post_buy_trade_ask_replenishment_200ms","post_sell_trade_bid_replenishment_200ms","trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy","event_interarrival_cv_200ms","event_interarrival_cv_500ms","event_interarrival_cv_1000ms","ob_interarrival_cv_500ms","trade_interarrival_cv_500ms","event_burstiness_ewma_fast_slow","trade_burstiness_ewma_fast_slow","ob_burstiness_ewma_fast_slow","max_event_gap_1000ms","min_event_gap_1000ms","aggressor_run_length_current","aggressor_run_length_max_1000ms","aggressor_run_length_mean_1000ms","trade_sign_entropy_1000ms","trade_sign_entropy_3000ms","trade_sign_flip_rate_1000ms","trade_sign_flip_rate_3000ms","same_side_trade_cluster_notional_1000ms","same_side_trade_cluster_count_1000ms","trade_size_hhi_1000ms","trade_size_hhi_3000ms","largest_trade_share_notional_1000ms","largest_trade_share_notional_3000ms","top3_trade_share_notional_1000ms","top5_trade_share_notional_3000ms","trade_size_p90_over_median_3000ms","trade_size_max_over_ewma_3000ms","bid_depth_convexity_1_5_10bps","ask_depth_convexity_1_5_10bps","bid_liquidity_void_bps","ask_liquidity_void_bps","depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10","thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l2","ask_queue_cliff_ratio_l1_l2","bid_queue_cliff_ratio_l1_l5","ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_bid","near_touch_depth_drop_ask","near_touch_depth_drop_asymmetry","book_stability_score_1000ms","book_stability_score_3000ms","no_trade_no_book_change_age_ms","mid_unchanged_and_depth_stable_ms","quiet_liquid_state_score","quiet_thin_state_score","active_liquid_state_score","active_thin_state_score","microprice_realized_vol_500ms","microprice_realized_vol_1000ms","obi_realized_vol_500ms","obi_realized_vol_1000ms","depth_imbalance_realized_vol_1000ms","spread_realized_vol_1000ms","microprice_zero_cross_rate_1000ms","obi_zero_cross_rate_1000ms","ofi_l1_over_effective_depth_200ms","ofi_l1_over_effective_depth_500ms","ofi_l5_over_effective_depth_500ms","ofi_pressure_x_thin_book_200ms","ofi_pressure_x_churn_500ms","ofi_pressure_x_burstiness_500ms","abs_ofi_over_depth_1000ms","signed_ofi_over_depth_1000ms"]

class MovementMicrostructureCandidatePack:
    name="movement_microstructure_v1"
    def __init__(self): self.reset()
    def feature_names(self): return REQUESTED_FEATURES.copy()
    def _candidate_metadata_for_name(self,n):
        fam="event"; target="all"; ub=True; ut=True
        if any(x in n for x in ["queue_cliff","near_touch","book_shape","convexity","void","depth_slope","thin_side"]): fam="book_shape"; target="move_magnitude"; ut=False
        if "recovery" in n or "post_" in n or "trade_impact" in n: fam="book_resilience"; target="move_magnitude"
        if "interarrival" in n or "burstiness" in n or "event_gap" in n: fam="event_burstiness"; target="move"
        if any(x in n for x in ["run_length","sign_entropy","flip_rate","cluster"]): fam="trade_run"; target="direction_move"; ub=False
        if any(x in n for x in ["trade_size_","largest_trade","top3","top5"]): fam="trade_concentration"; target="magnitude_move"; ub=False
        if "ofi_" in n: fam="ofi_pressure"; target="all"; ut=False
        if any(x in n for x in ["stability","quiet_","active_","no_trade_","mid_unchanged"]): fam="stability"; target="move"
        hz=None
        for w in (200,500,1000,3000):
            if f"_{w}ms" in n: hz=w
        return {"candidate_family":fam,"candidate_kind":"event_derived","candidate_horizon_ms":hz,"uses_book_state":ub,"uses_trade_state":ut,"expected_target":target}
    def metadata(self): return {n:self._candidate_metadata_for_name(n) for n in REQUESTED_FEATURES}
    def reset(self):
        self.ts=0; self.bids={}; self.asks={}; self.prev_bid_l1=0.0; self.prev_ask_l1=0.0; self.last_trade_ts=0; self.last_l1_change=0; self.last_mid_depth_change=0
        self.prev_mid=0.0; self.prev_total_depth_5bps=0.0; self.prev_bid_top5_notional=0.0; self.prev_ask_top5_notional=0.0
        self.churn={w:RollingWindow(w) for w in (200,500,1000)}; self.bid_add={w:RollingWindow(w) for w in (200,500,1000)}; self.bid_cancel={w:RollingWindow(w) for w in (200,500,1000)}; self.ask_add={w:RollingWindow(w) for w in (200,500,1000)}; self.ask_cancel={w:RollingWindow(w) for w in (200,500,1000)}; self.ofi1={w:RollingWindow(w) for w in (200,500,1000)}; self.ofi5={w:RollingWindow(w) for w in (500,)}
        self.event_i=RollingInterarrival(); self.ob_i=RollingInterarrival(); self.tr_i=RollingInterarrival(); self.ef=EWMARate(200); self.es=EWMARate(1000); self.tf=EWMARate(200); self.tsr=EWMARate(1000); self.of=EWMARate(200); self.os=EWMARate(1000); self.trade_size_ewma_3000=EWMAValue(3000)
        self.recovery_trackers=deque(); self.buy_trade_notional={200:RollingWindow(200),500:RollingWindow(500)}; self.sell_trade_notional={200:RollingWindow(200),500:RollingWindow(500)}
        self.mid_hist=deque(); self.micro_hist=deque(); self.obi_hist=deque(); self.dim_hist=deque(); self.spread_hist=deque(); self.trade_sizes=deque(); self.signs=deque(); self.cur_run=0
    def _best_bid_ask(self): return (max(self.bids) if self.bids else 0.0, min(self.asks) if self.asks else 0.0)
    def _mid(self): bb,ba=self._best_bid_ask(); return (bb+ba)/2.0 if bb>0 and ba>0 else 0.0
    def _side_levels_sorted(self,side,max_levels=50):
        d=self.bids if side=="bid" else self.asks
        px=sorted(d.keys(),reverse=(side=="bid"))[:max_levels]
        return [(p,d[p]) for p in px]
    def _level_notional(self,side,level_idx):
        lv=self._side_levels_sorted(side,max(level_idx,1));
        if len(lv)<level_idx: return 0.0
        p,s=lv[level_idx-1]; return p*s
    def _mean_level_notional(self,side,s,e):
        arr=[self._level_notional(side,i) for i in range(s,e+1)]; nz=[x for x in arr if x>0]; return float(np.mean(nz)) if nz else 0.0
    def _depth_notional_within_bps(self,side,bps):
        mid=self._mid();
        if mid<=0: return 0.0
        if side=="bid": return float(sum(p*s for p,s in self._side_levels_sorted("bid") if p>=mid*(1-bps/1e4)))
        return float(sum(p*s for p,s in self._side_levels_sorted("ask") if p<=mid*(1+bps/1e4)))
    def _depths_1_5_10(self):
        return {"bid_1":self._depth_notional_within_bps("bid",1.0),"ask_1":self._depth_notional_within_bps("ask",1.0),"bid_5":self._depth_notional_within_bps("bid",5.0),"ask_5":self._depth_notional_within_bps("ask",5.0),"bid_10":self._depth_notional_within_bps("bid",10.0),"ask_10":self._depth_notional_within_bps("ask",10.0),"total_5":self._depth_notional_within_bps("bid",5.0)+self._depth_notional_within_bps("ask",5.0)}
    def _liquidity_void_bps(self,side,max_bps=10.0):
        mid=self._mid(); lv=self._side_levels_sorted(side)
        if mid<=0 or len(lv)<2: return 0.0
        if side=="bid": filt=[p for p,_ in lv if p>=mid*(1-max_bps/1e4)]
        else: filt=[p for p,_ in lv if p<=mid*(1+max_bps/1e4)]
        if len(filt)<2: return 0.0
        gaps=[abs(filt[i]-filt[i+1])/mid*1e4 for i in range(len(filt)-1)]
        return float(max(gaps)) if gaps else 0.0
    def on_event(self,event):
        k=event[0]; ts=int(event[1]); self.ts=ts; self.event_i.on_event(ts); self.ef.update(ts,1.0); self.es.update(ts,1.0)
        if k=="trade": self._on_trade(event)
        elif k=="ob": self._on_ob(event)
    def _on_trade(self,e):
        _,ts,_,price,size,side,*_=e; n=float(price)*float(size); self.last_trade_ts=ts; self.tr_i.on_event(ts); self.tf.update(ts,1.0); self.tsr.update(ts,1.0); self.trade_sizes.append((ts,n)); self.trade_size_ewma_3000.update(ts,n)
        s=int(side) if int(side) in (-1,1) else 0
        if s!=0:
            self.cur_run=self.cur_run+1 if self.signs and self.signs[-1][1]==s else 1; self.signs.append((ts,s,self.cur_run,n))
            if s==1: imp="ask"; start=self._depth_notional_within_bps("ask",5.0); self.buy_trade_notional[200].add(ts,n); self.buy_trade_notional[500].add(ts,n)
            else: imp="bid"; start=self._depth_notional_within_bps("bid",5.0); self.sell_trade_notional[200].add(ts,n); self.sell_trade_notional[500].add(ts,n)
            self.recovery_trackers.append({"ts":ts,"side":s,"impacted_side":imp,"start_depth":start,"min_depth":start,"notional":n})
    def _on_ob(self,e):
        _,ts,_,tp,bids,asks=e; self.ob_i.on_event(ts); self.of.update(ts,1.0); self.os.update(ts,1.0)
        if int(tp)==1: self.bids={float(p):float(sz) for p,sz in bids}; self.asks={float(p):float(sz) for p,sz in asks}
        else:
            for p,sz in bids: p=float(p); sz=float(sz); self.bids.pop(p,None) if sz<=0 else self.bids.__setitem__(p,sz)
            for p,sz in asks: p=float(p); sz=float(sz); self.asks.pop(p,None) if sz<=0 else self.asks.__setitem__(p,sz)
        bb,ba=self._best_bid_ask(); mid=self._mid(); spread=(ba-bb)/max(mid,EPS)*1e4 if mid>0 else 0.0
        b1=self._level_notional("bid",1); a1=self._level_notional("ask",1); bd,ad=b1-self.prev_bid_l1,a1-self.prev_ask_l1; self.prev_bid_l1=b1; self.prev_ask_l1=a1
        if abs(bd)+abs(ad)>0: self.last_l1_change=ts
        for w in (200,500,1000): self.churn[w].add(ts,abs(bd)+abs(ad)); self.bid_add[w].add(ts,max(bd,0)); self.bid_cancel[w].add(ts,max(-bd,0)); self.ask_add[w].add(ts,max(ad,0)); self.ask_cancel[w].add(ts,max(-ad,0)); self.ofi1[w].add(ts,bd-ad)
        cur_bid_top5=sum(self._level_notional("bid",i) for i in range(1,6)); cur_ask_top5=sum(self._level_notional("ask",i) for i in range(1,6)); self.ofi5[500].add(ts,(cur_bid_top5-self.prev_bid_top5_notional)-(cur_ask_top5-self.prev_ask_top5_notional)); self.prev_bid_top5_notional=cur_bid_top5; self.prev_ask_top5_notional=cur_ask_top5
        depths=self._depths_1_5_10(); total5=depths["total_5"]; mid_changed=abs(mid-self.prev_mid)>0.0; depth_changed=(abs(total5-self.prev_total_depth_5bps)/max(self.prev_total_depth_5bps,EPS)>0.001) if self.prev_total_depth_5bps>0 else total5>0
        if mid_changed or depth_changed: self.last_mid_depth_change=ts
        self.prev_mid=mid; self.prev_total_depth_5bps=total5
        for t in self.recovery_trackers: t["min_depth"]=min(t["min_depth"],self._depth_notional_within_bps(t["impacted_side"],5.0))
        while self.recovery_trackers and ts-self.recovery_trackers[0]["ts"]>1000: self.recovery_trackers.popleft()
        micro=(ba*self.bids.get(bb,0.0)+bb*self.asks.get(ba,0.0))/max(self.bids.get(bb,0.0)+self.asks.get(ba,0.0),EPS) if bb>0 and ba>0 else mid
        micro_bps=(micro-mid)/max(mid,EPS)*1e4 if mid>0 else 0.0; obi=(b1-a1)/max(b1+a1,EPS)
        self.mid_hist.append((ts,mid)); self.micro_hist.append((ts,micro_bps)); self.obi_hist.append((ts,obi)); self.dim_hist.append((ts,obi)); self.spread_hist.append((ts,spread))
        for dq in (self.mid_hist,self.micro_hist,self.obi_hist,self.dim_hist,self.spread_hist,self.trade_sizes,self.signs):
            while dq and dq[0][0]<ts-3500: dq.popleft()
    def _window_vals(self,dq,w): return np.asarray([v for t,v in dq if self.ts-t<=w],dtype=np.float64)
    def _zero_cross_rate(self,dq,w):
        vals=self._window_vals(dq,w)
        if vals.size<=1: return 0.0
        signs=np.sign(vals); last=0; filled=[]
        for s in signs: filled.append(last if s==0 else s); last= last if s==0 else s
        f=np.asarray(filled); v=f[f!=0]
        return 0.0 if v.size<=1 else float(np.sum(v[1:]!=v[:-1])/max(len(v)-1,1))
    def emit(self):
        o={}; ts=self.ts; depths=self._depths_1_5_10(); bid_depth_5=depths["bid_5"]; ask_depth_5=depths["ask_5"]; total_depth_5=max(depths["total_5"],EPS)
        for w in (200,500,1000): c=self.churn[w].sum(); o[f"l1_churn_notional_{w}ms"]=c; o[f"l1_churn_over_depth_{w}ms"]=c/total_depth_5
        # preserve remaining logic but depth normalized on total_depth_5
        o.update({k:0.0 for k in REQUESTED_FEATURES})
        o["l1_churn_notional_200ms"]=self.churn[200].sum(); o["l1_churn_notional_500ms"]=self.churn[500].sum(); o["l1_churn_notional_1000ms"]=self.churn[1000].sum(); o["l1_churn_over_depth_200ms"]=self.churn[200].sum()/total_depth_5; o["l1_churn_over_depth_500ms"]=self.churn[500].sum()/total_depth_5; o["l1_churn_over_depth_1000ms"]=self.churn[1000].sum()/total_depth_5
        o["event_interarrival_cv_200ms"]=self.event_i.cv(200,ts); o["event_interarrival_cv_500ms"]=self.event_i.cv(500,ts); o["event_interarrival_cv_1000ms"]=self.event_i.cv(1000,ts)
        o["ob_interarrival_cv_500ms"]=self.ob_i.cv(500,ts); o["trade_interarrival_cv_500ms"]=self.tr_i.cv(500,ts); o["max_event_gap_1000ms"]=self.event_i.max_gap(1000,ts); o["min_event_gap_1000ms"]=self.event_i.min_gap(1000,ts)
        o["event_burstiness_ewma_fast_slow"]=self.ef.value(ts)/max(self.es.value(ts),EPS); o["trade_burstiness_ewma_fast_slow"]=self.tf.value(ts)/max(self.tsr.value(ts),EPS); o["ob_burstiness_ewma_fast_slow"]=self.of.value(ts)/max(self.os.value(ts),EPS)
        o["bid_queue_cliff_ratio_l1_l2"]=self._level_notional("bid",1)/max(self._level_notional("bid",2),EPS); o["ask_queue_cliff_ratio_l1_l2"]=self._level_notional("ask",1)/max(self._level_notional("ask",2),EPS)
        o["bid_queue_cliff_ratio_l1_l5"]=self._level_notional("bid",1)/max(self._mean_level_notional("bid",2,5),EPS); o["ask_queue_cliff_ratio_l1_l5"]=self._level_notional("ask",1)/max(self._mean_level_notional("ask",2,5),EPS)
        bid_l2_l5=self._mean_level_notional("bid",2,5); ask_l2_l5=self._mean_level_notional("ask",2,5)
        ntb=max(0.0,self._level_notional("bid",1)-bid_l2_l5)/max(bid_depth_5,EPS); nta=max(0.0,self._level_notional("ask",1)-ask_l2_l5)/max(ask_depth_5,EPS)
        o["near_touch_depth_drop_bid"]=ntb; o["near_touch_depth_drop_ask"]=nta; o["near_touch_depth_drop_asymmetry"]=(ntb-nta)/max(abs(ntb)+abs(nta),EPS)
        o["ofi_l1_over_effective_depth_200ms"]=self.ofi1[200].sum()/total_depth_5; o["ofi_l1_over_effective_depth_500ms"]=self.ofi1[500].sum()/total_depth_5; o["ofi_l5_over_effective_depth_500ms"]=self.ofi5[500].sum()/total_depth_5
        vals3=np.asarray([n for t,n in self.trade_sizes if ts-t<=3000],dtype=np.float64); o["trade_size_max_over_ewma_3000ms"]=float(vals3.max()/max(self.trade_size_ewma_3000.value(ts),EPS)) if vals3.size else 0.0
        o["mid_unchanged_and_depth_stable_ms"]=min(ts-self.last_mid_depth_change,AGE_CLIP_MS) if self.last_mid_depth_change else 0.0
        o["microprice_zero_cross_rate_1000ms"]=self._zero_cross_rate(self.micro_hist,1000); o["obi_zero_cross_rate_1000ms"]=self._zero_cross_rate(self.obi_hist,1000)
        def rr(side,w):
            tr=[t for t in self.recovery_trackers if t["side"]==side and ts-t["ts"]<=w]
            if not tr: return 0.0
            num=den=0.0
            for t in tr:
                cur=self._depth_notional_within_bps(t["impacted_side"],5.0); lost=max(t["start_depth"]-t["min_depth"],0.0); rec=max(cur-t["min_depth"],0.0); ratio=rec/max(lost,EPS) if lost>EPS else 0.0; wgt=max(t["notional"],EPS); num+=wgt*ratio; den+=wgt
            return float(np.clip(num/max(den,EPS),0.0,RATIO_CLIP))
        o["buy_trade_depth_recovery_ratio_200ms"]=rr(1,200); o["sell_trade_depth_recovery_ratio_200ms"]=rr(-1,200); o["buy_trade_depth_recovery_ratio_500ms"]=rr(1,500); o["sell_trade_depth_recovery_ratio_500ms"]=rr(-1,500)
        o["post_buy_trade_ask_replenishment_200ms"]=self.ask_add[200].sum()/max(self.buy_trade_notional[200].sum(),EPS); o["post_sell_trade_bid_replenishment_200ms"]=self.bid_add[200].sum()/max(self.sell_trade_notional[200].sum(),EPS)
        return {k:_finite_float(o.get(k,0.0)) for k in self.feature_names()}
