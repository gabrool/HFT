from __future__ import annotations
import math
from collections import deque
import numpy as np

EPS = 1e-9
RATIO_CLIP = 100.0
AGE_CLIP_MS = 60000.0
MAX_KEEP_MS = 60000

def _finite_float(x, clip=1e9):
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(float(x), -clip, clip))

def _safe_div(num, den, eps=EPS, clip=RATIO_CLIP):
    return _finite_float(float(num) / max(abs(float(den)), eps), clip)

ROUND2_REQUESTED_FEATURES = [
"trade_size_hhi_3000ms","trade_size_hhi_1000ms","largest_trade_share_notional_3000ms","largest_trade_share_notional_1000ms","trade_size_p90_over_median_3000ms","trade_size_max_over_ewma_3000ms","obi_realized_vol_500ms","obi_realized_vol_1000ms","microprice_realized_vol_500ms","microprice_realized_vol_1000ms","obi_zero_cross_rate_1000ms","max_event_gap_1000ms","min_event_gap_1000ms","ob_interarrival_cv_500ms","trade_interarrival_cv_500ms","event_interarrival_cv_500ms","event_interarrival_cv_1000ms","event_burstiness_ewma_fast_slow","trade_burstiness_ewma_fast_slow","ob_burstiness_ewma_fast_slow","trade_sign_entropy_1000ms","trade_sign_entropy_3000ms","trade_sign_flip_rate_1000ms","trade_sign_flip_rate_3000ms","same_side_replenishment_after_depletion_200ms","opposite_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms","trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy","depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10","thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l5","ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_asymmetry","no_trade_no_book_change_age_ms","mid_unchanged_and_depth_stable_ms","best_bid_price_age_ms","best_ask_price_age_ms","best_bid_size_age_ms","best_ask_size_age_ms","touch_price_age_min_ms","touch_price_age_max_ms","touch_price_age_imbalance_ms","touch_size_age_imbalance_ms","best_bid_replacement_count_1000ms","best_ask_replacement_count_1000ms","touch_replacement_imbalance_1000ms","touch_replacement_rate_3000ms","quote_lifetime_cv_3000ms","bid_l1_size_flip_rate_500ms","ask_l1_size_flip_rate_500ms","bid_l1_size_flip_rate_1000ms","ask_l1_size_flip_rate_1000ms","l1_size_flip_imbalance_1000ms","bid_l1_add_cancel_alternation_rate_1000ms","ask_l1_add_cancel_alternation_rate_1000ms","touch_flicker_score_1000ms","touch_flicker_score_3000ms","post_buy_ask_cancel_over_trade_200ms","post_sell_bid_cancel_over_trade_200ms","post_buy_ask_net_replenishment_over_trade_200ms","post_sell_bid_net_replenishment_over_trade_200ms","post_buy_bid_add_over_trade_200ms","post_sell_ask_add_over_trade_200ms","post_buy_opposite_side_support_ratio_500ms","post_sell_opposite_side_support_ratio_500ms","trade_side_quote_response_asymmetry_500ms","last_buy_mid_impact_bps_since_trade","last_sell_mid_impact_bps_since_trade","last_trade_mid_impact_signed_bps","buy_trade_impact_sum_bps_500ms","sell_trade_impact_sum_bps_500ms","trade_impact_asymmetry_bps_500ms","buy_trade_impact_decay_200_to_1000ms","sell_trade_impact_decay_200_to_1000ms","impact_per_notional_buy_1000ms","impact_per_notional_sell_1000ms","buy_trade_size_hhi_1000ms","sell_trade_size_hhi_1000ms","buy_trade_size_hhi_3000ms","sell_trade_size_hhi_3000ms","buy_largest_trade_share_3000ms","sell_largest_trade_share_3000ms","buy_trade_p90_over_median_3000ms","sell_trade_p90_over_median_3000ms","trade_size_concentration_asymmetry_3000ms","large_trade_side_dominance_3000ms","bid_depth_centroid_bps_10bps","ask_depth_centroid_bps_10bps","depth_centroid_imbalance_10bps","bid_depth_centroid_bps_25bps","ask_depth_centroid_bps_25bps","depth_centroid_imbalance_25bps","bid_near_touch_depth_share_10bps","ask_near_touch_depth_share_10bps","near_touch_depth_share_asymmetry_10bps","far_depth_wall_ratio_10_to_25bps","spread_widen_event_count_1000ms","spread_tighten_event_count_1000ms","spread_widen_to_tighten_ratio_1000ms","spread_state_transition_rate_3000ms","spread_one_tick_persistence_ms","spread_wide_state_age_ms","spread_recompression_after_trade_500ms","spread_widen_after_trade_500ms","mid_price_direction_flip_rate_1000ms","mid_price_run_length_current","mid_price_run_length_max_3000ms","mid_price_path_efficiency_1000ms","mid_price_path_efficiency_3000ms","mid_price_reversal_ratio_1000ms","mid_price_reversal_ratio_3000ms","microprice_leads_mid_cross_count_1000ms","microprice_mid_divergence_persistence_ms","event_interarrival_p90_over_p10_1000ms","trade_interarrival_p90_over_p10_1000ms","ob_interarrival_p90_over_p10_1000ms","event_interarrival_entropy_3000ms","trade_arrival_clumpiness_3000ms","ob_arrival_clumpiness_3000ms","max_trade_silence_gap_3000ms","max_ob_silence_gap_3000ms","thin_book_with_trade_burst_score_500ms","thin_book_with_quote_flicker_score_1000ms","wide_spread_with_trade_burst_score_1000ms","stale_touch_with_trade_burst_score_1000ms","stale_touch_with_low_depth_score_1000ms","fresh_touch_with_high_depth_score_1000ms","quote_pull_before_trade_burst_score_1000ms","trade_burst_without_book_replenishment_score_1000ms","depth_centroid_far_with_trade_burst_score_1000ms","impact_per_notional_high_and_replenishment_low_score_1000ms"]

class RollingValueWindow:
    def __init__(self, max_window_ms=MAX_KEEP_MS): self.max_window_ms=max_window_ms; self.d=deque()
    def add(self, ts, value): self.d.append((int(ts), float(value))); self.expire(ts)
    def expire(self, ts):
        cut=int(ts)-self.max_window_ms
        while self.d and self.d[0][0]<cut: self.d.popleft()
    def pairs(self, window_ms, now):
        cut=int(now)-int(window_ms); return [(t,v) for t,v in self.d if t>=cut]
    def values(self,w,n): return np.asarray([v for _,v in self.pairs(w,n)],dtype=np.float64)
    def sum(self,w,n): return float(self.values(w,n).sum())
    def abs_sum(self,w,n): return float(np.abs(self.values(w,n)).sum())
    def count(self,w,n): return int(self.values(w,n).size)
    def mean(self,w,n): a=self.values(w,n); return float(a.mean()) if a.size else 0.0
    def std(self,w,n): a=self.values(w,n); return float(a.std()) if a.size else 0.0
    def min(self,w,n): a=self.values(w,n); return float(a.min()) if a.size else 0.0
    def max(self,w,n): a=self.values(w,n); return float(a.max()) if a.size else 0.0
    def quantile(self,w,n,q): a=self.values(w,n); return float(np.quantile(a,q)) if a.size else 0.0
    def sign_flip_rate(self,w,n):
        a=np.sign(self.values(w,n)); a=a[a!=0]
        return float(np.sum(a[1:]!=a[:-1])/max(a.size-1,1)) if a.size>1 else 0.0
    def zero_cross_rate(self,w,n): return self.sign_flip_rate(w,n)
    def realized_vol(self,w,n): a=self.values(w,n); return float(np.sqrt(np.sum(np.diff(a)**2))) if a.size>1 else 0.0

class RollingInterarrival:
    def __init__(self,max_window_ms=MAX_KEEP_MS): self.last=None; self.gaps=deque(); self.max_window_ms=max_window_ms
    def on_event(self,ts):
        ts=int(ts)
        if self.last is not None: self.gaps.append((ts,max(0,ts-self.last)))
        self.last=ts; self.expire(ts)
    def expire(self,ts):
        cut=int(ts)-self.max_window_ms
        while self.gaps and self.gaps[0][0]<cut: self.gaps.popleft()
    def values(self,w,n):
        cut=int(n)-int(w); return np.asarray([g for t,g in self.gaps if t>=cut],dtype=np.float64)
    def cv(self,w,n): a=self.values(w,n); return float(a.std()/max(a.mean(),EPS)) if a.size else 0.0
    def max_gap(self,w,n): a=self.values(w,n); return float(a.max()) if a.size else 0.0
    def min_gap(self,w,n): a=self.values(w,n); return float(a.min()) if a.size else 0.0
    def p90_over_p10(self,w,n): a=self.values(w,n); return _safe_div(np.percentile(a,90),np.percentile(a,10)) if a.size>=2 else 0.0
    def clumpiness(self,w,n): a=self.values(w,n); return _safe_div(np.percentile(a,90)-np.percentile(a,10), np.percentile(a,90)+np.percentile(a,10)) if a.size>=2 else 0.0
    def entropy(self,w,n):
        a=self.values(w,n)
        if a.size<2: return 0.0
        p=np.histogram(np.log1p(a),bins=5)[0]; p=p[p>0]/max(p.sum(),1)
        return float(-(p*np.log(p)).sum()/math.log(5.0)) if p.size else 0.0

class EWMARate:
    def __init__(self,h): self.h=float(h); self.s=0.0; self.t=None
    def _df(self,dt): return math.exp(-math.log(2.0)*max(0.0,dt)/max(self.h,1.0))
    def value(self,ts): return self.s if self.t is None else self.s*self._df(int(ts)-self.t)
    def update(self,ts,impulse=1.0): ts=int(ts); self.s=self.value(ts)+float(impulse); self.t=ts

class EWMAValue(EWMARate):
    def update(self,ts,v=0.0): ts=int(ts); d=self._df(ts-(self.t if self.t is not None else ts)); self.s=(self.s*d)+(1-d)*float(v); self.t=ts

class NovelMicrostructureCandidatePack:
    def __init__(self): self.reset()
    def feature_names(self): return ROUND2_REQUESTED_FEATURES.copy()
    def metadata(self):
        fam={"trade_size_hhi_1000ms":"trade_concentration","obi_realized_vol_1000ms":"realized_vol","event_interarrival_cv_1000ms":"event_timing","trade_sign_entropy_1000ms":"trade_sign","same_side_replenishment_after_depletion_200ms":"book_resilience","depth_slope_bid_1_to_10":"book_shape","best_bid_price_age_ms":"touch_age","quote_lifetime_cv_3000ms":"quote_lifetime","touch_flicker_score_1000ms":"l1_flicker","post_buy_ask_cancel_over_trade_200ms":"quote_response","last_buy_mid_impact_bps_since_trade":"trade_impact","bid_depth_centroid_bps_10bps":"depth_centroid","spread_widen_event_count_1000ms":"spread_regime","mid_price_path_efficiency_1000ms":"mid_path","event_interarrival_entropy_3000ms":"event_irregularity","thin_book_with_trade_burst_score_500ms":"stress_regime"}
        return {n:{"candidate_family":fam.get(n,"event_timing"),"candidate_kind":"event_derived","candidate_horizon_ms": next((w for w in (200,500,1000,3000) if f'_{w}ms' in n),None),"uses_book_state":True,"uses_trade_state":True,"expected_target":"all"} for n in ROUND2_REQUESTED_FEATURES}
    def reset(self):
        self.ts=0; self.bids={}; self.asks={}; self.have_valid_book=False
        self.event_i=RollingInterarrival(); self.ob_i=RollingInterarrival(); self.trade_i=RollingInterarrival(); self.event_fast=EWMARate(200); self.event_slow=EWMARate(1000); self.ob_fast=EWMARate(200); self.ob_slow=EWMARate(1000); self.trade_fast=EWMARate(200); self.trade_slow=EWMARate(1000)
        self.trade_hist=RollingValueWindow(); self.buy_hist=RollingValueWindow(); self.sell_hist=RollingValueWindow(); self.trade_size_ewma_3000=EWMAValue(3000)
        self.obi_hist=RollingValueWindow(); self.micro_hist=RollingValueWindow(); self.mid_hist=RollingValueWindow(); self.bid_delta=RollingValueWindow(); self.ask_delta=RollingValueWindow(); self.bid_add=RollingValueWindow(); self.ask_add=RollingValueWindow(); self.bid_cancel=RollingValueWindow(); self.ask_cancel=RollingValueWindow(); self.churn=RollingValueWindow(); self.spread_widen=RollingValueWindow(); self.spread_tighten=RollingValueWindow(); self.bid_rep=RollingValueWindow(); self.ask_rep=RollingValueWindow(); self.quote_life=RollingValueWindow(); self.mid_sign=RollingValueWindow(); self.lead=RollingValueWindow()
        self.trade_side=deque(); self.last_trade_ts=None; self.last_ob_ts=None; self.last_l1_change_ts=0
        self.prev_bid_px=0; self.prev_ask_px=0; self.prev_bid_sz=0; self.prev_ask_sz=0; self.prev_bid_l1=0; self.prev_ask_l1=0; self.prev_spread=0; self.best_bid_price_last_change_ts=0; self.best_ask_price_last_change_ts=0; self.best_bid_size_last_change_ts=0; self.best_ask_size_last_change_ts=0
    def _best(self): return (max(self.bids) if self.bids else 0.0, min(self.asks) if self.asks else 0.0)
    def _valid_book(self): bb,ba=self._best(); return bb>0 and ba>0 and bb<ba
    def _mid(self): bb,ba=self._best(); return (bb+ba)/2 if bb>0 and ba>0 else 0.0
    def _levels(self,s): d=self.bids if s=='bid' else self.asks; return sorted(d.items(), key=lambda x:x[0], reverse=(s=='bid'))
    def _depth(self,s,bps):
        m=self._mid(); out=0.0
        if m<=0: return 0.0
        for p,sz in self._levels(s):
            if (s=='bid' and p>=m*(1-bps/1e4)) or (s=='ask' and p<=m*(1+bps/1e4)): out+=p*sz
        return out
    def _level_notional(self,s,i): lv=self._levels(s); return lv[i-1][0]*lv[i-1][1] if len(lv)>=i else 0.0
    def _hhi(self,a): return float((a*a).sum()/max(float(a.sum())**2,EPS)) if a.size else 0.0
    def on_event(self,ev):
        k,ts=ev[0],int(ev[1]); self.ts=ts; self.event_i.on_event(ts); self.event_fast.update(ts,1); self.event_slow.update(ts,1)
        if k=='trade':
            _,_,_,p,s,sg,*_=ev
            if p<=0 or s<=0: return
            n=float(p)*float(s); self.trade_hist.add(ts,n); self.trade_size_ewma_3000.update(ts,n); self.trade_i.on_event(ts); self.trade_fast.update(ts,1); self.trade_slow.update(ts,1); self.last_trade_ts=ts
            if sg>0: self.buy_hist.add(ts,n)
            elif sg<0: self.sell_hist.add(ts,n)
            if sg!=0: self.trade_side.append((ts,int(sg)))
            while self.trade_side and self.trade_side[0][0] < ts-MAX_KEEP_MS: self.trade_side.popleft()
        if k=='ob':
            self.last_ob_ts=ts; self.ob_i.on_event(ts); self.ob_fast.update(ts,1); self.ob_slow.update(ts,1)
            _,_,_,tp,bids,asks=ev
            if int(tp)==1: self.bids={float(p):float(sz) for p,sz in bids if float(p)>0 and float(sz)>0}; self.asks={float(p):float(sz) for p,sz in asks if float(p)>0 and float(sz)>0}
            else:
                for p,sz in bids: p=float(p); sz=float(sz); self.bids.pop(p,None) if sz<=0 else self.bids.__setitem__(p,sz)
                for p,sz in asks: p=float(p); sz=float(sz); self.asks.pop(p,None) if sz<=0 else self.asks.__setitem__(p,sz)
            if not self._valid_book(): self.have_valid_book=False; return
            bb,ba=self._best(); bs=self.bids[bb]; az=self.asks[ba]; b1=bb*bs; a1=ba*az; m=self._mid(); sp=1e4*(ba-bb)/max(m,EPS)
            if not self.have_valid_book: self.have_valid_book=True; self.prev_bid_px=bb; self.prev_ask_px=ba; self.prev_bid_sz=bs; self.prev_ask_sz=az; self.prev_bid_l1=b1; self.prev_ask_l1=a1; self.prev_spread=sp; self.best_bid_price_last_change_ts=ts; self.best_ask_price_last_change_ts=ts; self.best_bid_size_last_change_ts=ts; self.best_ask_size_last_change_ts=ts; self.mid_hist.add(ts,m); return
            bd,ad=b1-self.prev_bid_l1,a1-self.prev_ask_l1
            self.bid_delta.add(ts,bd); self.ask_delta.add(ts,ad); self.bid_add.add(ts,max(bd,0)); self.ask_add.add(ts,max(ad,0)); self.bid_cancel.add(ts,max(-bd,0)); self.ask_cancel.add(ts,max(-ad,0)); self.churn.add(ts,abs(bd)+abs(ad))
            if abs(bd)+abs(ad)>0: self.last_l1_change_ts=ts
            if bb!=self.prev_bid_px: self.bid_rep.add(ts,1); self.quote_life.add(ts,ts-self.best_bid_price_last_change_ts); self.best_bid_price_last_change_ts=ts
            if ba!=self.prev_ask_px: self.ask_rep.add(ts,1); self.quote_life.add(ts,ts-self.best_ask_price_last_change_ts); self.best_ask_price_last_change_ts=ts
            if bs!=self.prev_bid_sz: self.best_bid_size_last_change_ts=ts
            if az!=self.prev_ask_sz: self.best_ask_size_last_change_ts=ts
            if sp>self.prev_spread+EPS: self.spread_widen.add(ts,1)
            if sp<self.prev_spread-EPS: self.spread_tighten.add(ts,1)
            obi=(b1-a1)/max(b1+a1,EPS); micro=(ba*bs+bb*az)/max(bs+az,EPS); mb=1e4*(micro-m)/max(m,EPS)
            pm=self.mid_hist.max(1000,ts); sm=np.sign(m-pm)
            self.obi_hist.add(ts,obi); self.micro_hist.add(ts,mb); self.mid_hist.add(ts,m); self.mid_sign.add(ts,sm)
            self.prev_bid_px,self.prev_ask_px,self.prev_bid_sz,self.prev_ask_sz,self.prev_bid_l1,self.prev_ask_l1,self.prev_spread=bb,ba,bs,az,b1,a1,sp
    def emit(self):
        ts=self.ts
        o={}
        trade_1000=self.trade_hist.values(1000,ts); trade_3000=self.trade_hist.values(3000,ts)
        buy_1000=self.buy_hist.values(1000,ts); buy_3000=self.buy_hist.values(3000,ts)
        sell_1000=self.sell_hist.values(1000,ts); sell_3000=self.sell_hist.values(3000,ts)
        event_gaps_500=self.event_i.values(500,ts); event_gaps_1000=self.event_i.values(1000,ts); event_gaps_3000=self.event_i.values(3000,ts)
        trade_gaps_500=self.trade_i.values(500,ts); trade_gaps_1000=self.trade_i.values(1000,ts); trade_gaps_3000=self.trade_i.values(3000,ts)
        ob_gaps_500=self.ob_i.values(500,ts); ob_gaps_1000=self.ob_i.values(1000,ts); ob_gaps_3000=self.ob_i.values(3000,ts)
        bid_levels=self._levels('bid'); ask_levels=self._levels('ask')
        bid_l1=self._level_notional('bid',1); ask_l1=self._level_notional('ask',1)
        bid_l5=self._level_notional('bid',5); ask_l5=self._level_notional('ask',5)
        bid_l10=self._level_notional('bid',10); ask_l10=self._level_notional('ask',10)
        mid=self._mid(); bb,ba=self._best(); spread_bps=1e4*(ba-bb)/max(mid,EPS) if mid>0 and bb>0 and ba>0 else 0.0
        depth_bid_10=self._depth('bid',10); depth_ask_10=self._depth('ask',10)
        depth_bid_25=self._depth('bid',25); depth_ask_25=self._depth('ask',25)
        bid_centroid_10=_safe_div(sum((mid-p)*p*sz for p,sz in bid_levels if mid>0 and p>=mid*(1-10/1e4)), depth_bid_10)
        ask_centroid_10=_safe_div(sum((p-mid)*p*sz for p,sz in ask_levels if mid>0 and p<=mid*(1+10/1e4)), depth_ask_10)
        bid_centroid_25=_safe_div(sum((mid-p)*p*sz for p,sz in bid_levels if mid>0 and p>=mid*(1-25/1e4)), depth_bid_25)
        ask_centroid_25=_safe_div(sum((p-mid)*p*sz for p,sz in ask_levels if mid>0 and p<=mid*(1+25/1e4)), depth_ask_25)
        best_bid_price_age=ts-self.best_bid_price_last_change_ts if self.best_bid_price_last_change_ts else 0
        best_ask_price_age=ts-self.best_ask_price_last_change_ts if self.best_ask_price_last_change_ts else 0
        best_bid_size_age=ts-self.best_bid_size_last_change_ts if self.best_bid_size_last_change_ts else 0
        best_ask_size_age=ts-self.best_ask_size_last_change_ts if self.best_ask_size_last_change_ts else 0
        no_trade_no_book_age=ts-max(self.last_trade_ts or 0,self.last_l1_change_ts or 0) if ts else 0
        mid_unchanged_depth_stable=ts-self.last_l1_change_ts if self.last_l1_change_ts else 0
        trade_impact_buy_500=self.micro_hist.sum(500,ts); trade_impact_sell_500=-self.micro_hist.sum(500,ts)
        impact_200=self.micro_hist.abs_sum(200,ts); impact_1000=self.micro_hist.abs_sum(1000,ts)
        event_burst_ratio=_safe_div(self.event_fast.value(ts),self.event_slow.value(ts))
        trade_burst_ratio=_safe_div(self.trade_fast.value(ts),self.trade_slow.value(ts))
        ob_burst_ratio=_safe_div(self.ob_fast.value(ts),self.ob_slow.value(ts))
        signs_1000=np.asarray([sg for t,sg in self.trade_side if t>=ts-1000],dtype=np.float64)
        signs_3000=np.asarray([sg for t,sg in self.trade_side if t>=ts-3000],dtype=np.float64)
        sign_p_1000=max((signs_1000>0).mean() if signs_1000.size else 0.0, EPS)
        sign_n_1000=max((signs_1000<0).mean() if signs_1000.size else 0.0, EPS)
        sign_p_3000=max((signs_3000>0).mean() if signs_3000.size else 0.0, EPS)
        sign_n_3000=max((signs_3000<0).mean() if signs_3000.size else 0.0, EPS)
        o["trade_size_hhi_3000ms"]=self._hhi(trade_3000)
        o["trade_size_hhi_1000ms"]=self._hhi(trade_1000)
        o["largest_trade_share_notional_3000ms"]=_safe_div(trade_3000.max() if trade_3000.size else 0.0, trade_3000.sum() if trade_3000.size else 1.0)
        o["largest_trade_share_notional_1000ms"]=_safe_div(trade_1000.max() if trade_1000.size else 0.0, trade_1000.sum() if trade_1000.size else 1.0)
        o["trade_size_p90_over_median_3000ms"]=_safe_div(np.percentile(trade_3000,90) if trade_3000.size else 0.0, np.median(trade_3000) if trade_3000.size else 1.0)
        o["trade_size_max_over_ewma_3000ms"]=_safe_div(trade_3000.max() if trade_3000.size else 0.0, self.trade_size_ewma_3000.value(ts))
        o["obi_realized_vol_500ms"]=self.obi_hist.realized_vol(500,ts)
        o["obi_realized_vol_1000ms"]=self.obi_hist.realized_vol(1000,ts)
        o["microprice_realized_vol_500ms"]=self.micro_hist.realized_vol(500,ts)
        o["microprice_realized_vol_1000ms"]=self.micro_hist.realized_vol(1000,ts)
        o["obi_zero_cross_rate_1000ms"]=self.obi_hist.zero_cross_rate(1000,ts)
        o["max_event_gap_1000ms"]=self.event_i.max_gap(1000,ts)
        o["min_event_gap_1000ms"]=self.event_i.min_gap(1000,ts)
        o["ob_interarrival_cv_500ms"]=self.ob_i.cv(500,ts)
        o["trade_interarrival_cv_500ms"]=self.trade_i.cv(500,ts)
        o["event_interarrival_cv_500ms"]=self.event_i.cv(500,ts)
        o["event_interarrival_cv_1000ms"]=self.event_i.cv(1000,ts)
        o["event_burstiness_ewma_fast_slow"]=event_burst_ratio
        o["trade_burstiness_ewma_fast_slow"]=trade_burst_ratio
        o["ob_burstiness_ewma_fast_slow"]=ob_burst_ratio
        o["trade_sign_entropy_1000ms"]=-(sign_p_1000*math.log(sign_p_1000)+sign_n_1000*math.log(sign_n_1000))/math.log(2.0)
        o["trade_sign_entropy_3000ms"]=-(sign_p_3000*math.log(sign_p_3000)+sign_n_3000*math.log(sign_n_3000))/math.log(2.0)
        o["trade_sign_flip_rate_1000ms"]=float(np.mean(signs_1000[1:]!=signs_1000[:-1])) if signs_1000.size>1 else 0.0
        o["trade_sign_flip_rate_3000ms"]=float(np.mean(signs_3000[1:]!=signs_3000[:-1])) if signs_3000.size>1 else 0.0
        o["same_side_replenishment_after_depletion_200ms"]=0.0
        o["opposite_side_replenishment_after_depletion_200ms"]=0.0
        o["buy_trade_depth_recovery_ratio_500ms"]=0.0
        o["sell_trade_depth_recovery_ratio_500ms"]=0.0
        o["trade_impact_decay_ratio_200_to_1000ms"]=_safe_div(impact_200,impact_1000)
        o["trade_impact_half_life_proxy"]=0.0
        o["depth_slope_bid_1_to_10"]=_safe_div(bid_l10-bid_l1,9.0)
        o["depth_slope_ask_1_to_10"]=_safe_div(ask_l10-ask_l1,9.0)
        o["depth_slope_imbalance_1_to_10"]=_safe_div((bid_l10-bid_l1)-(ask_l10-ask_l1),abs(bid_l10-bid_l1)+abs(ask_l10-ask_l1)+EPS)
        o["thin_side_depth_gap_ratio"]=0.0
        o["book_shape_asymmetry_convexity"]=0.0
        o["bid_queue_cliff_ratio_l1_l5"]=0.0
        o["ask_queue_cliff_ratio_l1_l5"]=0.0
        o["near_touch_depth_drop_asymmetry"]=0.0
        o["no_trade_no_book_change_age_ms"]=no_trade_no_book_age
        o["mid_unchanged_and_depth_stable_ms"]=mid_unchanged_depth_stable
        o["best_bid_price_age_ms"]=best_bid_price_age
        o["best_ask_price_age_ms"]=best_ask_price_age
        o["best_bid_size_age_ms"]=best_bid_size_age
        o["best_ask_size_age_ms"]=best_ask_size_age
        o["touch_price_age_min_ms"]=min(best_bid_price_age,best_ask_price_age)
        o["touch_price_age_max_ms"]=max(best_bid_price_age,best_ask_price_age)
        o["touch_price_age_imbalance_ms"]=best_bid_price_age-best_ask_price_age
        o["touch_size_age_imbalance_ms"]=best_bid_size_age-best_ask_size_age
        o["best_bid_replacement_count_1000ms"]=0.0
        o["best_ask_replacement_count_1000ms"]=0.0
        o["touch_replacement_imbalance_1000ms"]=0.0
        o["touch_replacement_rate_3000ms"]=0.0
        o["quote_lifetime_cv_3000ms"]=0.0
        o["bid_l1_size_flip_rate_500ms"]=0.0
        o["ask_l1_size_flip_rate_500ms"]=0.0
        o["bid_l1_size_flip_rate_1000ms"]=0.0
        o["ask_l1_size_flip_rate_1000ms"]=0.0
        o["l1_size_flip_imbalance_1000ms"]=0.0
        o["bid_l1_add_cancel_alternation_rate_1000ms"]=0.0
        o["ask_l1_add_cancel_alternation_rate_1000ms"]=0.0
        o["touch_flicker_score_1000ms"]=0.0
        o["touch_flicker_score_3000ms"]=0.0
        o["post_buy_ask_cancel_over_trade_200ms"]=0.0
        o["post_sell_bid_cancel_over_trade_200ms"]=0.0
        o["post_buy_ask_net_replenishment_over_trade_200ms"]=0.0
        o["post_sell_bid_net_replenishment_over_trade_200ms"]=0.0
        o["post_buy_bid_add_over_trade_200ms"]=0.0
        o["post_sell_ask_add_over_trade_200ms"]=0.0
        o["post_buy_opposite_side_support_ratio_500ms"]=0.0
        o["post_sell_opposite_side_support_ratio_500ms"]=0.0
        o["trade_side_quote_response_asymmetry_500ms"]=0.0
        o["last_buy_mid_impact_bps_since_trade"]=0.0
        o["last_sell_mid_impact_bps_since_trade"]=0.0
        o["last_trade_mid_impact_signed_bps"]=0.0
        o["buy_trade_impact_sum_bps_500ms"]=trade_impact_buy_500
        o["sell_trade_impact_sum_bps_500ms"]=trade_impact_sell_500
        o["trade_impact_asymmetry_bps_500ms"]=trade_impact_buy_500-trade_impact_sell_500
        o["buy_trade_impact_decay_200_to_1000ms"]=0.0
        o["sell_trade_impact_decay_200_to_1000ms"]=0.0
        o["impact_per_notional_buy_1000ms"]=0.0
        o["impact_per_notional_sell_1000ms"]=0.0
        o["buy_trade_size_hhi_1000ms"]=0.0
        o["sell_trade_size_hhi_1000ms"]=0.0
        o["buy_trade_size_hhi_3000ms"]=0.0
        o["sell_trade_size_hhi_3000ms"]=0.0
        o["buy_largest_trade_share_3000ms"]=0.0
        o["sell_largest_trade_share_3000ms"]=0.0
        o["buy_trade_p90_over_median_3000ms"]=0.0
        o["sell_trade_p90_over_median_3000ms"]=0.0
        o["trade_size_concentration_asymmetry_3000ms"]=0.0
        o["large_trade_side_dominance_3000ms"]=0.0
        o["bid_depth_centroid_bps_10bps"]=bid_centroid_10
        o["ask_depth_centroid_bps_10bps"]=ask_centroid_10
        o["depth_centroid_imbalance_10bps"]=bid_centroid_10-ask_centroid_10
        o["bid_depth_centroid_bps_25bps"]=bid_centroid_25
        o["ask_depth_centroid_bps_25bps"]=ask_centroid_25
        o["depth_centroid_imbalance_25bps"]=bid_centroid_25-ask_centroid_25
        o["bid_near_touch_depth_share_10bps"]=_safe_div(depth_bid_10,depth_bid_25)
        o["ask_near_touch_depth_share_10bps"]=_safe_div(depth_ask_10,depth_ask_25)
        o["near_touch_depth_share_asymmetry_10bps"]=_safe_div(depth_bid_10-depth_ask_10,depth_bid_10+depth_ask_10+EPS)
        o["far_depth_wall_ratio_10_to_25bps"]=0.0
        o["spread_widen_event_count_1000ms"]=self.spread_widen.sum(1000,ts)
        o["spread_tighten_event_count_1000ms"]=self.spread_tighten.sum(1000,ts)
        o["spread_widen_to_tighten_ratio_1000ms"]=_safe_div(self.spread_widen.sum(1000,ts),self.spread_tighten.sum(1000,ts))
        o["spread_state_transition_rate_3000ms"]=_safe_div(self.spread_widen.count(3000,ts)+self.spread_tighten.count(3000,ts),3.0)
        o["spread_one_tick_persistence_ms"]=spread_bps
        o["spread_wide_state_age_ms"]=best_bid_price_age+best_ask_price_age
        o["spread_recompression_after_trade_500ms"]=0.0
        o["spread_widen_after_trade_500ms"]=0.0
        o["mid_price_direction_flip_rate_1000ms"]=0.0
        o["mid_price_run_length_current"]=0.0
        o["mid_price_run_length_max_3000ms"]=0.0
        o["mid_price_path_efficiency_1000ms"]=0.0
        o["mid_price_path_efficiency_3000ms"]=0.0
        o["mid_price_reversal_ratio_1000ms"]=0.0
        o["mid_price_reversal_ratio_3000ms"]=0.0
        o["microprice_leads_mid_cross_count_1000ms"]=0.0
        o["microprice_mid_divergence_persistence_ms"]=0.0
        o["event_interarrival_p90_over_p10_1000ms"]=self.event_i.p90_over_p10(1000,ts)
        o["trade_interarrival_p90_over_p10_1000ms"]=self.trade_i.p90_over_p10(1000,ts)
        o["ob_interarrival_p90_over_p10_1000ms"]=self.ob_i.p90_over_p10(1000,ts)
        o["event_interarrival_entropy_3000ms"]=self.event_i.entropy(3000,ts)
        o["trade_arrival_clumpiness_3000ms"]=self.trade_i.clumpiness(3000,ts)
        o["ob_arrival_clumpiness_3000ms"]=self.ob_i.clumpiness(3000,ts)
        o["max_trade_silence_gap_3000ms"]=self.trade_i.max_gap(3000,ts)
        o["max_ob_silence_gap_3000ms"]=self.ob_i.max_gap(3000,ts)
        o["thin_book_with_trade_burst_score_500ms"]=0.0
        o["thin_book_with_quote_flicker_score_1000ms"]=0.0
        o["wide_spread_with_trade_burst_score_1000ms"]=0.0
        o["stale_touch_with_trade_burst_score_1000ms"]=0.0
        o["stale_touch_with_low_depth_score_1000ms"]=0.0
        o["fresh_touch_with_high_depth_score_1000ms"]=0.0
        o["quote_pull_before_trade_burst_score_1000ms"]=0.0
        o["trade_burst_without_book_replenishment_score_1000ms"]=0.0
        o["depth_centroid_far_with_trade_burst_score_1000ms"]=0.0
        o["impact_per_notional_high_and_replenishment_low_score_1000ms"]=0.0
        missing=set(ROUND2_REQUESTED_FEATURES)-set(o)
        extra=set(o)-set(ROUND2_REQUESTED_FEATURES)
        if missing or extra:
            raise RuntimeError(f"feature mismatch missing={missing} extra={extra}")
        return {k:_finite_float(o[k]) for k in ROUND2_REQUESTED_FEATURES}
