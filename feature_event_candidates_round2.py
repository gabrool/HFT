from __future__ import annotations
import math
import re
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

def _signed_safe_div(num, den, eps=EPS, clip=RATIO_CLIP):
    out = float(num) / max(abs(float(den)), eps)
    return _finite_float(out, clip)

def _trade_sign_entropy(signs):
    signs = np.asarray(signs, dtype=np.float64)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    buy = int(np.sum(signs > 0))
    sell = int(np.sum(signs < 0))
    total = buy + sell
    if total < 2 or buy == 0 or sell == 0:
        return 0.0
    p_buy = buy / total
    p_sell = sell / total
    return _finite_float(-(p_buy * math.log(p_buy) + p_sell * math.log(p_sell)) / math.log(2.0))

ROUND2_REQUESTED_FEATURES = [
"trade_size_hhi_3000ms","trade_size_hhi_1000ms","largest_trade_share_notional_3000ms","largest_trade_share_notional_1000ms","trade_size_p90_over_median_3000ms","trade_size_max_over_ewma_3000ms","obi_realized_vol_500ms","obi_realized_vol_1000ms","microprice_realized_vol_500ms","microprice_realized_vol_1000ms","obi_zero_cross_rate_1000ms","max_event_gap_1000ms","min_event_gap_1000ms","ob_interarrival_cv_500ms","trade_interarrival_cv_500ms","event_interarrival_cv_500ms","event_interarrival_cv_1000ms","event_burstiness_ewma_fast_slow","trade_burstiness_ewma_fast_slow","ob_burstiness_ewma_fast_slow","trade_sign_entropy_1000ms","trade_sign_entropy_3000ms","trade_sign_flip_rate_1000ms","trade_sign_flip_rate_3000ms","same_side_replenishment_after_depletion_200ms","opposite_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms","trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy","depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10","thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l5","ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_asymmetry","no_trade_no_book_change_age_ms","mid_unchanged_and_depth_stable_ms","best_bid_price_age_ms","best_ask_price_age_ms","best_bid_size_age_ms","best_ask_size_age_ms","touch_price_age_min_ms","touch_price_age_max_ms","touch_price_age_imbalance_ms","touch_size_age_imbalance_ms","best_bid_replacement_count_1000ms","best_ask_replacement_count_1000ms","touch_replacement_imbalance_1000ms","touch_replacement_rate_3000ms","quote_lifetime_cv_3000ms","bid_l1_size_flip_rate_500ms","ask_l1_size_flip_rate_500ms","bid_l1_size_flip_rate_1000ms","ask_l1_size_flip_rate_1000ms","l1_size_flip_imbalance_1000ms","bid_l1_add_cancel_alternation_rate_1000ms","ask_l1_add_cancel_alternation_rate_1000ms","touch_flicker_score_1000ms","touch_flicker_score_3000ms","post_buy_ask_cancel_over_trade_200ms","post_sell_bid_cancel_over_trade_200ms","post_buy_ask_net_replenishment_over_trade_200ms","post_sell_bid_net_replenishment_over_trade_200ms","post_buy_bid_add_over_trade_200ms","post_sell_ask_add_over_trade_200ms","post_buy_opposite_side_support_ratio_500ms","post_sell_opposite_side_support_ratio_500ms","trade_side_quote_response_asymmetry_500ms","last_buy_mid_impact_bps_since_trade","last_sell_mid_impact_bps_since_trade","last_trade_mid_impact_signed_bps","buy_trade_impact_sum_bps_500ms","sell_trade_impact_sum_bps_500ms","trade_impact_asymmetry_bps_500ms","buy_trade_impact_decay_200_to_1000ms","sell_trade_impact_decay_200_to_1000ms","impact_per_notional_buy_1000ms","impact_per_notional_sell_1000ms","buy_trade_size_hhi_1000ms","sell_trade_size_hhi_1000ms","buy_trade_size_hhi_3000ms","sell_trade_size_hhi_3000ms","buy_largest_trade_share_3000ms","sell_largest_trade_share_3000ms","buy_trade_p90_over_median_3000ms","sell_trade_p90_over_median_3000ms","trade_size_concentration_asymmetry_3000ms","large_trade_side_dominance_3000ms","bid_depth_centroid_bps_10bps","ask_depth_centroid_bps_10bps","depth_centroid_imbalance_10bps","bid_depth_centroid_bps_25bps","ask_depth_centroid_bps_25bps","depth_centroid_imbalance_25bps","bid_near_touch_depth_share_10bps","ask_near_touch_depth_share_10bps","near_touch_depth_share_asymmetry_10bps","far_depth_wall_ratio_10_to_25bps","spread_widen_event_count_1000ms","spread_tighten_event_count_1000ms","spread_widen_to_tighten_ratio_1000ms","spread_state_transition_rate_3000ms","spread_one_tick_persistence_ms","spread_wide_state_age_ms","spread_recompression_after_trade_500ms","spread_widen_after_trade_500ms","mid_price_direction_flip_rate_1000ms","mid_price_run_length_current","mid_price_run_length_max_3000ms","mid_price_path_efficiency_1000ms","mid_price_path_efficiency_3000ms","mid_price_reversal_ratio_1000ms","mid_price_reversal_ratio_3000ms","microprice_leads_mid_cross_count_1000ms","microprice_mid_divergence_persistence_ms","event_interarrival_p90_over_p10_1000ms","trade_interarrival_p90_over_p10_1000ms","ob_interarrival_p90_over_p10_1000ms","event_interarrival_entropy_3000ms","trade_arrival_clumpiness_3000ms","ob_arrival_clumpiness_3000ms","max_trade_silence_gap_3000ms","max_ob_silence_gap_3000ms","thin_book_with_trade_burst_score_500ms","thin_book_with_quote_flicker_score_1000ms","wide_spread_with_trade_burst_score_1000ms","stale_touch_with_trade_burst_score_1000ms","stale_touch_with_low_depth_score_1000ms","fresh_touch_with_high_depth_score_1000ms","quote_pull_before_trade_burst_score_1000ms","trade_burst_without_book_replenishment_score_1000ms","depth_centroid_far_with_trade_burst_score_1000ms","impact_per_notional_high_and_replenishment_low_score_1000ms"]

FAMILY_BY_FEATURE = {
    "trade_size_hhi_3000ms": "trade_concentration",
    "trade_size_hhi_1000ms": "trade_concentration",
    "largest_trade_share_notional_3000ms": "trade_concentration",
    "largest_trade_share_notional_1000ms": "trade_concentration",
    "trade_size_p90_over_median_3000ms": "trade_concentration",
    "trade_size_max_over_ewma_3000ms": "trade_concentration",
    "obi_realized_vol_500ms": "realized_vol",
    "obi_realized_vol_1000ms": "realized_vol",
    "microprice_realized_vol_500ms": "realized_vol",
    "microprice_realized_vol_1000ms": "realized_vol",
    "obi_zero_cross_rate_1000ms": "realized_vol",
    "max_event_gap_1000ms": "event_timing",
    "min_event_gap_1000ms": "event_timing",
    "ob_interarrival_cv_500ms": "event_timing",
    "trade_interarrival_cv_500ms": "event_timing",
    "event_interarrival_cv_500ms": "event_timing",
    "event_interarrival_cv_1000ms": "event_timing",
    "event_burstiness_ewma_fast_slow": "event_timing",
    "trade_burstiness_ewma_fast_slow": "event_timing",
    "ob_burstiness_ewma_fast_slow": "event_timing",
    "trade_sign_entropy_1000ms": "trade_sign",
    "trade_sign_entropy_3000ms": "trade_sign",
    "trade_sign_flip_rate_1000ms": "trade_sign",
    "trade_sign_flip_rate_3000ms": "trade_sign",
    "same_side_replenishment_after_depletion_200ms": "book_resilience",
    "opposite_side_replenishment_after_depletion_200ms": "book_resilience",
    "buy_trade_depth_recovery_ratio_500ms": "book_resilience",
    "sell_trade_depth_recovery_ratio_500ms": "book_resilience",
    "trade_impact_decay_ratio_200_to_1000ms": "trade_impact",
    "trade_impact_half_life_proxy": "trade_impact",
    "depth_slope_bid_1_to_10": "book_shape",
    "depth_slope_ask_1_to_10": "book_shape",
    "depth_slope_imbalance_1_to_10": "book_shape",
    "thin_side_depth_gap_ratio": "book_shape",
    "book_shape_asymmetry_convexity": "book_shape",
    "bid_queue_cliff_ratio_l1_l5": "book_shape",
    "ask_queue_cliff_ratio_l1_l5": "book_shape",
    "near_touch_depth_drop_asymmetry": "book_shape",
    "no_trade_no_book_change_age_ms": "quiet_state",
    "mid_unchanged_and_depth_stable_ms": "quiet_state",
    "best_bid_price_age_ms": "touch_age",
    "best_ask_price_age_ms": "touch_age",
    "best_bid_size_age_ms": "touch_age",
    "best_ask_size_age_ms": "touch_age",
    "touch_price_age_min_ms": "touch_age",
    "touch_price_age_max_ms": "touch_age",
    "touch_price_age_imbalance_ms": "touch_age",
    "touch_size_age_imbalance_ms": "touch_age",
    "best_bid_replacement_count_1000ms": "quote_lifetime",
    "best_ask_replacement_count_1000ms": "quote_lifetime",
    "touch_replacement_imbalance_1000ms": "quote_lifetime",
    "touch_replacement_rate_3000ms": "quote_lifetime",
    "quote_lifetime_cv_3000ms": "quote_lifetime",
    "bid_l1_size_flip_rate_500ms": "l1_flicker",
    "ask_l1_size_flip_rate_500ms": "l1_flicker",
    "bid_l1_size_flip_rate_1000ms": "l1_flicker",
    "ask_l1_size_flip_rate_1000ms": "l1_flicker",
    "l1_size_flip_imbalance_1000ms": "l1_flicker",
    "bid_l1_add_cancel_alternation_rate_1000ms": "l1_flicker",
    "ask_l1_add_cancel_alternation_rate_1000ms": "l1_flicker",
    "touch_flicker_score_1000ms": "l1_flicker",
    "touch_flicker_score_3000ms": "l1_flicker",
    "post_buy_ask_cancel_over_trade_200ms": "quote_response",
    "post_sell_bid_cancel_over_trade_200ms": "quote_response",
    "post_buy_ask_net_replenishment_over_trade_200ms": "quote_response",
    "post_sell_bid_net_replenishment_over_trade_200ms": "quote_response",
    "post_buy_bid_add_over_trade_200ms": "quote_response",
    "post_sell_ask_add_over_trade_200ms": "quote_response",
    "post_buy_opposite_side_support_ratio_500ms": "quote_response",
    "post_sell_opposite_side_support_ratio_500ms": "quote_response",
    "trade_side_quote_response_asymmetry_500ms": "quote_response",
    "last_buy_mid_impact_bps_since_trade": "trade_impact",
    "last_sell_mid_impact_bps_since_trade": "trade_impact",
    "last_trade_mid_impact_signed_bps": "trade_impact",
    "buy_trade_impact_sum_bps_500ms": "trade_impact",
    "sell_trade_impact_sum_bps_500ms": "trade_impact",
    "trade_impact_asymmetry_bps_500ms": "trade_impact",
    "buy_trade_impact_decay_200_to_1000ms": "trade_impact",
    "sell_trade_impact_decay_200_to_1000ms": "trade_impact",
    "impact_per_notional_buy_1000ms": "trade_impact",
    "impact_per_notional_sell_1000ms": "trade_impact",
    "buy_trade_size_hhi_1000ms": "trade_concentration",
    "sell_trade_size_hhi_1000ms": "trade_concentration",
    "buy_trade_size_hhi_3000ms": "trade_concentration",
    "sell_trade_size_hhi_3000ms": "trade_concentration",
    "buy_largest_trade_share_3000ms": "trade_concentration",
    "sell_largest_trade_share_3000ms": "trade_concentration",
    "buy_trade_p90_over_median_3000ms": "trade_concentration",
    "sell_trade_p90_over_median_3000ms": "trade_concentration",
    "trade_size_concentration_asymmetry_3000ms": "trade_concentration",
    "large_trade_side_dominance_3000ms": "trade_concentration",
    "bid_depth_centroid_bps_10bps": "depth_centroid",
    "ask_depth_centroid_bps_10bps": "depth_centroid",
    "depth_centroid_imbalance_10bps": "depth_centroid",
    "bid_depth_centroid_bps_25bps": "depth_centroid",
    "ask_depth_centroid_bps_25bps": "depth_centroid",
    "depth_centroid_imbalance_25bps": "depth_centroid",
    "bid_near_touch_depth_share_10bps": "depth_centroid",
    "ask_near_touch_depth_share_10bps": "depth_centroid",
    "near_touch_depth_share_asymmetry_10bps": "depth_centroid",
    "far_depth_wall_ratio_10_to_25bps": "depth_centroid",
    "spread_widen_event_count_1000ms": "spread_regime",
    "spread_tighten_event_count_1000ms": "spread_regime",
    "spread_widen_to_tighten_ratio_1000ms": "spread_regime",
    "spread_state_transition_rate_3000ms": "spread_regime",
    "spread_one_tick_persistence_ms": "spread_regime",
    "spread_wide_state_age_ms": "spread_regime",
    "spread_recompression_after_trade_500ms": "spread_regime",
    "spread_widen_after_trade_500ms": "spread_regime",
    "mid_price_direction_flip_rate_1000ms": "mid_path",
    "mid_price_run_length_current": "mid_path",
    "mid_price_run_length_max_3000ms": "mid_path",
    "mid_price_path_efficiency_1000ms": "mid_path",
    "mid_price_path_efficiency_3000ms": "mid_path",
    "mid_price_reversal_ratio_1000ms": "mid_path",
    "mid_price_reversal_ratio_3000ms": "mid_path",
    "microprice_leads_mid_cross_count_1000ms": "mid_path",
    "microprice_mid_divergence_persistence_ms": "mid_path",
    "event_interarrival_p90_over_p10_1000ms": "event_timing",
    "trade_interarrival_p90_over_p10_1000ms": "event_timing",
    "ob_interarrival_p90_over_p10_1000ms": "event_timing",
    "event_interarrival_entropy_3000ms": "event_irregularity",
    "trade_arrival_clumpiness_3000ms": "event_irregularity",
    "ob_arrival_clumpiness_3000ms": "event_irregularity",
    "max_trade_silence_gap_3000ms": "event_timing",
    "max_ob_silence_gap_3000ms": "event_timing",
    "thin_book_with_trade_burst_score_500ms": "stress_regime",
    "thin_book_with_quote_flicker_score_1000ms": "stress_regime",
    "wide_spread_with_trade_burst_score_1000ms": "spread_regime",
    "stale_touch_with_trade_burst_score_1000ms": "stress_regime",
    "stale_touch_with_low_depth_score_1000ms": "stress_regime",
    "fresh_touch_with_high_depth_score_1000ms": "stress_regime",
    "quote_pull_before_trade_burst_score_1000ms": "stress_regime",
    "trade_burst_without_book_replenishment_score_1000ms": "stress_regime",
    "depth_centroid_far_with_trade_burst_score_1000ms": "stress_regime",
    "impact_per_notional_high_and_replenishment_low_score_1000ms": "stress_regime",
}

assert set(FAMILY_BY_FEATURE) == set(ROUND2_REQUESTED_FEATURES)

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
    def zero_cross_rate(self,w,n):
        a=self.values(w,n)
        if a.size<2: return 0.0
        carried=[]; last=0
        for v in a:
            s=int(np.sign(v))
            if s==0:
                if last!=0: carried.append(last)
            else:
                carried.append(s); last=s
        if len(carried)<2: return 0.0
        arr=np.asarray(carried,dtype=np.int8)
        return float(np.sum(arr[1:]!=arr[:-1])/max(arr.size-1,1))
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

class EWMAValue:
    def __init__(self, half_life_ms):
        self.h = float(half_life_ms)
        self.s = 0.0
        self.t = None

    def _decay_factor(self, dt_ms):
        return math.exp(-math.log(2.0) * max(0.0, float(dt_ms)) / max(self.h, 1.0))

    def value(self, ts):
        ts = int(ts)
        if self.t is None:
            self.t = ts
            return self.s
        dt = max(0, ts - self.t)
        self.s *= self._decay_factor(dt)
        self.t = ts
        return self.s

    def update(self, ts, value):
        self.value(ts)
        self.s += float(value)

assert len(ROUND2_REQUESTED_FEATURES) == 136
assert len(set(ROUND2_REQUESTED_FEATURES)) == 136

INVALID_BOOK_NEUTRAL_FEATURES = (
    "depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10",
    "thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l5",
    "ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_asymmetry",
    "bid_depth_centroid_bps_10bps","ask_depth_centroid_bps_10bps","depth_centroid_imbalance_10bps",
    "bid_depth_centroid_bps_25bps","ask_depth_centroid_bps_25bps","depth_centroid_imbalance_25bps",
    "bid_near_touch_depth_share_10bps","ask_near_touch_depth_share_10bps","near_touch_depth_share_asymmetry_10bps",
    "far_depth_wall_ratio_10_to_25bps",
    "best_bid_price_age_ms","best_ask_price_age_ms","best_bid_size_age_ms","best_ask_size_age_ms",
    "touch_price_age_min_ms","touch_price_age_max_ms","touch_price_age_imbalance_ms","touch_size_age_imbalance_ms",
    "mid_unchanged_and_depth_stable_ms","spread_one_tick_persistence_ms","spread_wide_state_age_ms",
    "buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms",
    "last_buy_mid_impact_bps_since_trade","last_sell_mid_impact_bps_since_trade","last_trade_mid_impact_signed_bps",
    "buy_trade_impact_sum_bps_500ms","sell_trade_impact_sum_bps_500ms","trade_impact_asymmetry_bps_500ms",
    "buy_trade_impact_decay_200_to_1000ms","sell_trade_impact_decay_200_to_1000ms",
    "impact_per_notional_buy_1000ms","impact_per_notional_sell_1000ms",
    "trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy",
    "thin_book_with_trade_burst_score_500ms","thin_book_with_quote_flicker_score_1000ms",
    "wide_spread_with_trade_burst_score_1000ms","stale_touch_with_trade_burst_score_1000ms",
    "stale_touch_with_low_depth_score_1000ms","fresh_touch_with_high_depth_score_1000ms",
    "quote_pull_before_trade_burst_score_1000ms","trade_burst_without_book_replenishment_score_1000ms",
    "depth_centroid_far_with_trade_burst_score_1000ms",
    "impact_per_notional_high_and_replenishment_low_score_1000ms",
)

class NovelMicrostructureCandidatePack:
    name = "novel_microstructure_round2_v1"
    def __init__(self): self.reset()
    def feature_names(self): return ROUND2_REQUESTED_FEATURES.copy()
    def metadata(self):
        expected_target_by_family = {
            "trade_concentration": "move",
            "realized_vol": "magnitude",
            "event_timing": "move",
            "trade_sign": "direction",
            "book_resilience": "move",
            "book_shape": "magnitude",
            "touch_age": "move",
            "quote_lifetime": "move",
            "l1_flicker": "move",
            "quote_response": "move",
            "trade_impact": "direction",
            "depth_centroid": "magnitude",
            "spread_regime": "move",
            "mid_path": "direction",
            "quiet_state": "move",
            "event_irregularity": "move",
            "stress_regime": "move",
        }

        def _candidate_horizon_ms(feature_name):
            match = re.search(r"_(200|500|1000|3000)ms$", feature_name)
            return int(match.group(1)) if match else None

        trade_only_features = {
            "trade_size_hhi_3000ms","trade_size_hhi_1000ms","largest_trade_share_notional_3000ms","largest_trade_share_notional_1000ms","trade_size_p90_over_median_3000ms","trade_size_max_over_ewma_3000ms","trade_sign_entropy_1000ms","trade_sign_entropy_3000ms","trade_sign_flip_rate_1000ms","trade_sign_flip_rate_3000ms","buy_trade_size_hhi_1000ms","sell_trade_size_hhi_1000ms","buy_trade_size_hhi_3000ms","sell_trade_size_hhi_3000ms","buy_largest_trade_share_3000ms","sell_largest_trade_share_3000ms","buy_trade_p90_over_median_3000ms","sell_trade_p90_over_median_3000ms","trade_size_concentration_asymmetry_3000ms","large_trade_side_dominance_3000ms","trade_interarrival_cv_500ms","trade_burstiness_ewma_fast_slow","trade_interarrival_p90_over_p10_1000ms","trade_arrival_clumpiness_3000ms","max_trade_silence_gap_3000ms"
        }
        event_wide_timing_features = {"max_event_gap_1000ms","min_event_gap_1000ms","event_interarrival_cv_500ms","event_interarrival_cv_1000ms","event_burstiness_ewma_fast_slow","event_interarrival_p90_over_p10_1000ms","event_interarrival_entropy_3000ms"}
        both_features = {"same_side_replenishment_after_depletion_200ms","opposite_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms","trade_impact_decay_ratio_200_to_1000ms","trade_impact_half_life_proxy","no_trade_no_book_change_age_ms","post_buy_ask_cancel_over_trade_200ms","post_sell_bid_cancel_over_trade_200ms","post_buy_ask_net_replenishment_over_trade_200ms","post_sell_bid_net_replenishment_over_trade_200ms","post_buy_bid_add_over_trade_200ms","post_sell_ask_add_over_trade_200ms","post_buy_opposite_side_support_ratio_500ms","post_sell_opposite_side_support_ratio_500ms","trade_side_quote_response_asymmetry_500ms","last_buy_mid_impact_bps_since_trade","last_sell_mid_impact_bps_since_trade","last_trade_mid_impact_signed_bps","buy_trade_impact_sum_bps_500ms","sell_trade_impact_sum_bps_500ms","trade_impact_asymmetry_bps_500ms","buy_trade_impact_decay_200_to_1000ms","sell_trade_impact_decay_200_to_1000ms","impact_per_notional_buy_1000ms","impact_per_notional_sell_1000ms","spread_recompression_after_trade_500ms","spread_widen_after_trade_500ms","thin_book_with_trade_burst_score_500ms","thin_book_with_quote_flicker_score_1000ms","wide_spread_with_trade_burst_score_1000ms","stale_touch_with_trade_burst_score_1000ms","stale_touch_with_low_depth_score_1000ms","fresh_touch_with_high_depth_score_1000ms","quote_pull_before_trade_burst_score_1000ms","trade_burst_without_book_replenishment_score_1000ms","depth_centroid_far_with_trade_burst_score_1000ms","impact_per_notional_high_and_replenishment_low_score_1000ms"}
        out = {}
        for feature_name in ROUND2_REQUESTED_FEATURES:
            if feature_name not in FAMILY_BY_FEATURE:
                raise KeyError(f"Unknown feature family for {feature_name}")
            family = FAMILY_BY_FEATURE[feature_name]
            if family not in expected_target_by_family:
                raise KeyError(f"Unknown expected target family mapping for {family}")
            if feature_name in trade_only_features:
                uses_book_state = False; uses_trade_state = True
            elif feature_name in event_wide_timing_features or feature_name in both_features:
                uses_book_state = True; uses_trade_state = True
            else:
                uses_book_state = True; uses_trade_state = False
            out[feature_name] = {
                "candidate_family": family,
                "candidate_kind": "event_derived",
                "candidate_horizon_ms": _candidate_horizon_ms(feature_name),
                "uses_book_state": uses_book_state,
                "uses_trade_state": uses_trade_state,
                "expected_target": expected_target_by_family[family],
            }
        return out
    def reset(self):
        self.ts=0; self.bids={}; self.asks={}; self.have_valid_book=False; self.book_valid=False; self.last_valid_book_ts=None; self.has_valid_book=False
        self.event_i=RollingInterarrival(); self.ob_i=RollingInterarrival(); self.trade_i=RollingInterarrival(); self.event_fast=EWMARate(200); self.event_slow=EWMARate(1000); self.ob_fast=EWMARate(200); self.ob_slow=EWMARate(1000); self.trade_fast=EWMARate(200); self.trade_slow=EWMARate(1000)
        self.trade_hist=RollingValueWindow(); self.buy_hist=RollingValueWindow(); self.sell_hist=RollingValueWindow(); self.trade_size_ewma_3000=EWMAValue(3000)
        self.obi_hist=RollingValueWindow(); self.micro_hist=RollingValueWindow(); self.mid_hist=RollingValueWindow(); self.bid_delta=RollingValueWindow(); self.ask_delta=RollingValueWindow(); self.bid_add=RollingValueWindow(); self.ask_add=RollingValueWindow(); self.bid_cancel=RollingValueWindow(); self.ask_cancel=RollingValueWindow(); self.churn=RollingValueWindow(); self.spread_widen=RollingValueWindow(); self.spread_tighten=RollingValueWindow(); self.bid_rep=RollingValueWindow(); self.ask_rep=RollingValueWindow(); self.quote_life=RollingValueWindow(); self.mid_sign=RollingValueWindow(); self.lead=RollingValueWindow()
        self.trade_side=deque(); self.last_trade_ts=None; self.last_ob_ts=None; self.last_l1_change_ts=None
        self.trade_records=deque(); self.last_buy_trade=None; self.last_sell_trade=None; self.last_nonzero_trade=None
        self.depletion_trackers=deque()
        self.post_buy_ask_cancel_200=RollingValueWindow(); self.post_sell_bid_cancel_200=RollingValueWindow()
        self.post_buy_ask_add_200=RollingValueWindow(); self.post_sell_bid_add_200=RollingValueWindow()
        self.post_buy_bid_add_200=RollingValueWindow(); self.post_sell_ask_add_200=RollingValueWindow()
        self.post_buy_ask_net_200=RollingValueWindow(); self.post_sell_bid_net_200=RollingValueWindow()
        self.post_buy_bid_add_500=RollingValueWindow(); self.post_sell_ask_add_500=RollingValueWindow()
        self.bid_delta_sign_hist=RollingValueWindow(); self.ask_delta_sign_hist=RollingValueWindow()
        self.bid_alt_events=RollingValueWindow(); self.ask_alt_events=RollingValueWindow()
        self.prev_bid_delta_sign=0; self.prev_ask_delta_sign=0
        self.prev_mid=0.0; self.prev_micro_bps=0.0; self.prev_nonzero_micro_sign=0; self.mid_delta_sign_hist=RollingValueWindow(); self.current_mid_run_sign=0; self.current_mid_run_len=0; self.mid_run_len_hist=RollingValueWindow(); self.micro_lead_events=RollingValueWindow(); self.divergence_start_ts=None
        self.min_positive_spread_bps=None; self.in_one_tick_spread=False; self.one_tick_spread_entry_ts=None; self.in_wide_spread=False; self.wide_spread_entry_ts=None
        self.depth_ewma_10bps=EWMAValue(5000); self.last_mid_or_depth_stable_break_ts=None; self.prev_stable_mid=0.0; self.prev_stable_total_depth_5bps=0.0
        self.prev_bid_px=0; self.prev_ask_px=0; self.prev_bid_sz=0; self.prev_ask_sz=0; self.prev_bid_l1=0; self.prev_ask_l1=0; self.prev_spread=0; self.best_bid_price_last_change_ts=None; self.best_ask_price_last_change_ts=None; self.best_bid_size_last_change_ts=None; self.best_ask_size_last_change_ts=None
    def _best(self): return (max(self.bids) if self.bids else 0.0, min(self.asks) if self.asks else 0.0)
    def _best_bid_ask(self):
        bid_px=max(self.bids) if self.bids else 0.0; ask_px=min(self.asks) if self.asks else 0.0
        bid_sz=self.bids.get(bid_px,0.0) if bid_px>0 else 0.0; ask_sz=self.asks.get(ask_px,0.0) if ask_px>0 else 0.0
        return bid_px,bid_sz,ask_px,ask_sz
    def _update_book_validity(self,ts):
        bid_px,bid_sz,ask_px,ask_sz=self._best_bid_ask(); valid=(bid_px>0.0 and ask_px>0.0 and bid_sz>0.0 and ask_sz>0.0 and bid_px<ask_px)
        self.book_valid=bool(valid)
        if self.book_valid: self.last_valid_book_ts=int(ts)
        return self.book_valid
    def _valid_book(self): return self.book_valid
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
    def _safe_asym(self,a,b): return _safe_div(float(a)-float(b), abs(float(a))+abs(float(b))+EPS)
    def _depth_recovery_ratio(self, side, window_ms, now, current_depth_5):
        current_depth_5=float(current_depth_5) if current_depth_5>0 else 0.0
        num=0.0; den=0.0
        for r in self.trade_records:
            if now-r["ts"]>window_ms: continue
            if side=="buy":
                if r["side"]<=0: continue
                depth0=float(r.get("ask_depth_5_at_trade",0.0))
            else:
                if r["side"]>=0: continue
                depth0=float(r.get("bid_depth_5_at_trade",0.0))
            notional=float(r.get("notional",0.0))
            if depth0<=0 or notional<=0: continue
            ratio=current_depth_5/max(depth0,EPS)
            num += ratio*notional; den += notional
        if den<=0: return 0.0
        return _finite_float(num/den,RATIO_CLIP)
    def _depth_centroid(self, side, bps):
        m=self._mid()
        if m<=0: return 0.0
        num=0.0; den=0.0
        for p,sz in self._levels(side):
            if (side=='bid' and p>=m*(1-bps/1e4)) or (side=='ask' and p<=m*(1+bps/1e4)):
                d=1e4*abs(p-m)/m; n=p*sz
                num += d*n; den += n
        return _safe_div(num, den)
    def _trade_impacts(self, window_ms, now, mid_now):
        buy_impacts=[]; sell_impacts=[]
        buy_abs_notional=[]; sell_abs_notional=[]
        for r in self.trade_records:
            if now-r["ts"]>window_ms or r["mid_at_trade"]<=0 or r["side"]==0:
                continue
            if r["side"]>0:
                imp=1e4*(mid_now-r["mid_at_trade"])/max(r["mid_at_trade"],EPS)
                buy_impacts.append(imp); buy_abs_notional.append((abs(imp), r["notional"]))
            else:
                imp=1e4*(r["mid_at_trade"]-mid_now)/max(r["mid_at_trade"],EPS)
                sell_impacts.append(imp); sell_abs_notional.append((abs(imp), r["notional"]))
        return buy_impacts, sell_impacts, buy_abs_notional, sell_abs_notional

    def _current_book_metrics(self, ts):
        bb,bs,ba,az=self._best_bid_ask()
        if not self.book_valid:
            return {"mid":0.0,"spread_bps":0.0,"bb":0.0,"ba":0.0,"bs":0.0,"az":0.0,"b1":0.0,"a1":0.0,"depth_bid_10":0.0,"depth_ask_10":0.0,"depth_bid_25":0.0,"depth_ask_25":0.0,"bid_centroid_10":0.0,"ask_centroid_10":0.0,"bid_centroid_25":0.0,"ask_centroid_25":0.0,"obi":0.0,"micro_bps":0.0,"total_depth_5":0.0}
        m=(bb+ba)/2.0; sp=1e4*(ba-bb)/max(m,EPS); b1=bb*bs; a1=ba*az
        depth_bid_10=self._depth('bid',10); depth_ask_10=self._depth('ask',10); depth_bid_25=self._depth('bid',25); depth_ask_25=self._depth('ask',25)
        total_depth_5=self._depth('bid',5)+self._depth('ask',5)
        obi=(b1-a1)/max(b1+a1,EPS); micro=(ba*bs+bb*az)/max(bs+az,EPS); mb=1e4*(micro-m)/max(m,EPS)
        return {"mid":m,"spread_bps":sp,"bb":bb,"ba":ba,"bs":bs,"az":az,"b1":b1,"a1":a1,"depth_bid_10":depth_bid_10,"depth_ask_10":depth_ask_10,"depth_bid_25":depth_bid_25,"depth_ask_25":depth_ask_25,"bid_centroid_10":self._depth_centroid("bid",10),"ask_centroid_10":self._depth_centroid("ask",10),"bid_centroid_25":self._depth_centroid("bid",25),"ask_centroid_25":self._depth_centroid("ask",25),"obi":obi,"micro_bps":mb,"total_depth_5":total_depth_5}

    def on_event(self,ev):
        k,ts=ev[0],int(ev[1]); self.ts=ts; self.event_i.on_event(ts); self.event_fast.update(ts,1); self.event_slow.update(ts,1)
        if k=='trade':
            _,_,_,p,s,sg,*_=ev
            if p<=0 or s<=0: return
            n=float(p)*float(s); self.trade_hist.add(ts,n); self.trade_size_ewma_3000.update(ts,n); self.trade_i.on_event(ts); self.trade_fast.update(ts,1); self.trade_slow.update(ts,1); self.last_trade_ts=ts
            if sg>0: self.buy_hist.add(ts,n)
            elif sg<0: self.sell_hist.add(ts,n)
            has_valid_book=self._valid_book()
            mid_at_trade=self._mid() if has_valid_book else float(p)
            bid_depth_5_at_trade=self._depth('bid',5) if has_valid_book else 0.0
            ask_depth_5_at_trade=self._depth('ask',5) if has_valid_book else 0.0
            rec={"ts":ts,"price":float(p),"size":float(s),"notional":n,"side":int(sg),"mid_at_trade":mid_at_trade,"bid_depth_5_at_trade":bid_depth_5_at_trade,"ask_depth_5_at_trade":ask_depth_5_at_trade}
            if sg!=0 and mid_at_trade>0:
                self.trade_records.append(rec)
                if sg>0: self.last_buy_trade=rec
                elif sg<0: self.last_sell_trade=rec
                self.last_nonzero_trade=rec
            while self.trade_records and self.trade_records[0]["ts"] < ts-MAX_KEEP_MS: self.trade_records.popleft()
            if sg!=0: self.trade_side.append((ts,int(sg)))
            while self.trade_side and self.trade_side[0][0] < ts-MAX_KEEP_MS: self.trade_side.popleft()
        if k=='ob':
            self.last_ob_ts=ts; self.ob_i.on_event(ts); self.ob_fast.update(ts,1); self.ob_slow.update(ts,1)
            _,_,_,tp,bids,asks=ev
            if int(tp)==1: self.bids={float(p):float(sz) for p,sz in bids if float(p)>0 and float(sz)>0}; self.asks={float(p):float(sz) for p,sz in asks if float(p)>0 and float(sz)>0}
            else:
                for p,sz in bids:
                    try: p=float(p); sz=float(sz)
                    except (TypeError, ValueError): continue
                    if p<=0.0: self.bids.pop(p,None); continue
                    self.bids.pop(p,None) if sz<=0 else self.bids.__setitem__(p,sz)
                for p,sz in asks:
                    try: p=float(p); sz=float(sz)
                    except (TypeError, ValueError): continue
                    if p<=0.0: self.asks.pop(p,None); continue
                    self.asks.pop(p,None) if sz<=0 else self.asks.__setitem__(p,sz)
            if not self._update_book_validity(ts):
                return
            bm=self._current_book_metrics(ts); bb,ba,bs,az,b1,a1,m,sp=bm["bb"],bm["ba"],bm["bs"],bm["az"],bm["b1"],bm["a1"],bm["mid"],bm["spread_bps"]
            if not self.has_valid_book:
                self.has_valid_book=True; self.have_valid_book=True; self.prev_bid_px=bb; self.prev_ask_px=ba; self.prev_bid_sz=bs; self.prev_ask_sz=az; self.prev_bid_l1=b1; self.prev_ask_l1=a1; self.prev_spread=sp; self.best_bid_price_last_change_ts=ts; self.best_ask_price_last_change_ts=ts; self.best_bid_size_last_change_ts=ts; self.best_ask_size_last_change_ts=ts; self.last_l1_change_ts=ts; self.last_mid_or_depth_stable_break_ts=ts; self.prev_stable_mid=m; self.prev_stable_total_depth_5bps=bm["total_depth_5"]; self.mid_hist.add(ts,m); self.obi_hist.add(ts,bm["obi"]); self.micro_hist.add(ts,bm["micro_bps"]); self.depth_ewma_10bps.update(ts,bm["depth_bid_10"]+bm["depth_ask_10"]); self.prev_mid=m; self.prev_micro_bps=bm["micro_bps"]; self.prev_nonzero_micro_sign=int(np.sign(bm["micro_bps"])) if bm["micro_bps"]!=0 else 0; return
            bd,ad=b1-self.prev_bid_l1,a1-self.prev_ask_l1
            self.bid_delta.add(ts,bd); self.ask_delta.add(ts,ad); self.bid_add.add(ts,max(bd,0)); self.ask_add.add(ts,max(ad,0)); self.bid_cancel.add(ts,max(-bd,0)); self.ask_cancel.add(ts,max(-ad,0)); self.churn.add(ts,abs(bd)+abs(ad))
            bid_add=max(bd,0); ask_add=max(ad,0); bid_cancel=max(-bd,0); ask_cancel=max(-ad,0)
            for tr in self.depletion_trackers:
                if tr["ts"]<ts:
                    if tr["side"]=="bid": tr["same_recovered"] += bid_add; tr["opp_recovered"] += ask_add
                    else: tr["same_recovered"] += ask_add; tr["opp_recovered"] += bid_add
            if bid_cancel>0: self.depletion_trackers.append({"ts":ts,"side":"bid","amount":bid_cancel,"same_recovered":0.0,"opp_recovered":0.0})
            if ask_cancel>0: self.depletion_trackers.append({"ts":ts,"side":"ask","amount":ask_cancel,"same_recovered":0.0,"opp_recovered":0.0})
            while self.depletion_trackers and self.depletion_trackers[0]["ts"] < ts-500: self.depletion_trackers.popleft()
            if self.last_buy_trade and ts-self.last_buy_trade["ts"]<=200:
                self.post_buy_ask_cancel_200.add(ts,ask_cancel); self.post_buy_ask_add_200.add(ts,ask_add); self.post_buy_ask_net_200.add(ts,ask_add-ask_cancel); self.post_buy_bid_add_200.add(ts,bid_add)
            if self.last_sell_trade and ts-self.last_sell_trade["ts"]<=200:
                self.post_sell_bid_cancel_200.add(ts,bid_cancel); self.post_sell_bid_add_200.add(ts,bid_add); self.post_sell_bid_net_200.add(ts,bid_add-bid_cancel); self.post_sell_ask_add_200.add(ts,ask_add)
            if self.last_buy_trade and ts-self.last_buy_trade["ts"]<=500: self.post_buy_bid_add_500.add(ts,bid_add)
            if self.last_sell_trade and ts-self.last_sell_trade["ts"]<=500: self.post_sell_ask_add_500.add(ts,ask_add)
            sbd=int(np.sign(bd)); sad=int(np.sign(ad))
            if sbd!=0:
                self.bid_delta_sign_hist.add(ts,sbd)
                if self.prev_bid_delta_sign!=0 and sbd!=self.prev_bid_delta_sign: self.bid_alt_events.add(ts,1)
                self.prev_bid_delta_sign=sbd
            if sad!=0:
                self.ask_delta_sign_hist.add(ts,sad)
                if self.prev_ask_delta_sign!=0 and sad!=self.prev_ask_delta_sign: self.ask_alt_events.add(ts,1)
                self.prev_ask_delta_sign=sad
            bid_price_changed=(bb!=self.prev_bid_px); ask_price_changed=(ba!=self.prev_ask_px)
            bid_size_changed=bid_price_changed or (bs!=self.prev_bid_sz); ask_size_changed=ask_price_changed or (az!=self.prev_ask_sz)
            l1_notional_changed=(abs(bd)+abs(ad)>0)
            if l1_notional_changed or bid_price_changed or ask_price_changed or bid_size_changed or ask_size_changed: self.last_l1_change_ts=ts
            if bid_price_changed: self.bid_rep.add(ts,1); self.quote_life.add(ts,ts-self.best_bid_price_last_change_ts); self.best_bid_price_last_change_ts=ts
            if ask_price_changed: self.ask_rep.add(ts,1); self.quote_life.add(ts,ts-self.best_ask_price_last_change_ts); self.best_ask_price_last_change_ts=ts
            if bid_size_changed: self.best_bid_size_last_change_ts=ts
            if ask_size_changed: self.best_ask_size_last_change_ts=ts
            if sp>self.prev_spread+EPS: self.spread_widen.add(ts,1)
            if sp<self.prev_spread-EPS: self.spread_tighten.add(ts,1)
            obi=bm["obi"]; mb=bm["micro_bps"]
            pm=self.mid_hist.max(1000,ts); sm=np.sign(m-pm)
            self.obi_hist.add(ts,obi); self.micro_hist.add(ts,mb); self.mid_hist.add(ts,m); self.mid_sign.add(ts,sm)
            prev_nonzero_micro_sign_for_lead=self.prev_nonzero_micro_sign
            mid_delta=m-self.prev_mid if self.prev_mid>0 else 0.0; ms=int(np.sign(mid_delta))
            if ms!=0:
                self.mid_delta_sign_hist.add(ts,ms)
                if ms==self.current_mid_run_sign: self.current_mid_run_len += 1
                else: self.current_mid_run_sign=ms; self.current_mid_run_len=1
                self.mid_run_len_hist.add(ts,self.current_mid_run_len)
                if prev_nonzero_micro_sign_for_lead==ms: self.micro_lead_events.add(ts,1)
            current_micro_sign=int(np.sign(mb))
            if current_micro_sign!=0: self.prev_nonzero_micro_sign=current_micro_sign
            micro_sign=int(np.sign(mb)); div=(micro_sign!=0 and self.current_mid_run_sign!=0 and micro_sign!=self.current_mid_run_sign)
            if div and self.divergence_start_ts is None: self.divergence_start_ts=ts
            if not div: self.divergence_start_ts=None
            total_depth_10=bm["depth_bid_10"]+bm["depth_ask_10"]; self.depth_ewma_10bps.update(ts,total_depth_10)
            total_depth_5=bm["total_depth_5"]
            mid_changed=abs(m-self.prev_stable_mid)>1e-12
            depth_changed=abs(total_depth_5-self.prev_stable_total_depth_5bps)/max(abs(self.prev_stable_total_depth_5bps),EPS)>0.01
            if mid_changed or depth_changed: self.last_mid_or_depth_stable_break_ts=ts; self.prev_stable_mid=m; self.prev_stable_total_depth_5bps=total_depth_5
            if sp>0:
                if self.min_positive_spread_bps is None: self.min_positive_spread_bps=sp
                else: self.min_positive_spread_bps=min(self.min_positive_spread_bps,sp)
                one_tick_threshold=self.min_positive_spread_bps*1.05; wide_threshold=self.min_positive_spread_bps*1.50
                now_one_tick=sp<=one_tick_threshold; now_wide=sp>=wide_threshold
                if now_one_tick and not self.in_one_tick_spread: self.one_tick_spread_entry_ts=ts
                self.in_one_tick_spread=now_one_tick
                if not now_one_tick: self.one_tick_spread_entry_ts=None
                if now_wide and not self.in_wide_spread: self.wide_spread_entry_ts=ts
                self.in_wide_spread=now_wide
                if not now_wide: self.wide_spread_entry_ts=None
            self.prev_mid=m; self.prev_micro_bps=mb
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
        self._update_book_validity(ts); bm=self._current_book_metrics(ts)
        if not self.book_valid:
            bid_levels=[]; ask_levels=[]
            bid_l1=ask_l1=0.0; bid_l2=ask_l2=0.0; bid_l5=ask_l5=0.0; bid_l10=ask_l10=0.0
            bid_depth_1=ask_depth_1=0.0; depth_bid_5=depth_ask_5=0.0; depth_bid_10=depth_ask_10=0.0; depth_bid_25=depth_ask_25=0.0
            bid_centroid_10=ask_centroid_10=0.0; bid_centroid_25=ask_centroid_25=0.0
            mid=0.0; bb=ba=0.0; spread_bps=0.0
        else:
            bid_levels=self._levels('bid'); ask_levels=self._levels('ask')
            bid_l1=bid_levels[0][0]*bid_levels[0][1] if len(bid_levels)>=1 else 0.0; ask_l1=ask_levels[0][0]*ask_levels[0][1] if len(ask_levels)>=1 else 0.0
            bid_l2=bid_levels[1][0]*bid_levels[1][1] if len(bid_levels)>=2 else 0.0; ask_l2=ask_levels[1][0]*ask_levels[1][1] if len(ask_levels)>=2 else 0.0
            bid_l5=bid_levels[4][0]*bid_levels[4][1] if len(bid_levels)>=5 else 0.0; ask_l5=ask_levels[4][0]*ask_levels[4][1] if len(ask_levels)>=5 else 0.0
            bid_l10=bid_levels[9][0]*bid_levels[9][1] if len(bid_levels)>=10 else 0.0; ask_l10=ask_levels[9][0]*ask_levels[9][1] if len(ask_levels)>=10 else 0.0
            bid_depth_1=self._depth('bid',1); ask_depth_1=self._depth('ask',1)
            depth_bid_5=self._depth('bid',5); depth_ask_5=self._depth('ask',5)
            mid=bm["mid"]; bb=bm["bb"]; ba=bm["ba"]; spread_bps=bm["spread_bps"]
            depth_bid_10=bm["depth_bid_10"]; depth_ask_10=bm["depth_ask_10"]; depth_bid_25=bm["depth_bid_25"]; depth_ask_25=bm["depth_ask_25"]
            bid_centroid_10=bm["bid_centroid_10"]; ask_centroid_10=bm["ask_centroid_10"]; bid_centroid_25=bm["bid_centroid_25"]; ask_centroid_25=bm["ask_centroid_25"]
        best_bid_price_age=min(max(ts-self.best_bid_price_last_change_ts,0.0),AGE_CLIP_MS) if self.best_bid_price_last_change_ts is not None else 0.0
        best_ask_price_age=min(max(ts-self.best_ask_price_last_change_ts,0.0),AGE_CLIP_MS) if self.best_ask_price_last_change_ts is not None else 0.0
        best_bid_size_age=min(max(ts-self.best_bid_size_last_change_ts,0.0),AGE_CLIP_MS) if self.best_bid_size_last_change_ts is not None else 0.0
        best_ask_size_age=min(max(ts-self.best_ask_size_last_change_ts,0.0),AGE_CLIP_MS) if self.best_ask_size_last_change_ts is not None else 0.0
        trade_age=AGE_CLIP_MS if self.last_trade_ts is None else (ts-self.last_trade_ts); book_age=AGE_CLIP_MS if self.last_l1_change_ts is None else (ts-self.last_l1_change_ts); no_trade_no_book_age=min(max(min(trade_age,book_age),0.0),AGE_CLIP_MS)
        mid_unchanged_depth_stable=0.0 if self.last_mid_or_depth_stable_break_ts is None else min(max(ts-self.last_mid_or_depth_stable_break_ts,0.0),AGE_CLIP_MS)
        if self.book_valid and mid>0:
            bi500,si500,bw500,sw500=self._trade_impacts(500,ts,mid); bi200,si200,bw200,sw200=self._trade_impacts(200,ts,mid); bi1000,si1000,bw1000,sw1000=self._trade_impacts(1000,ts,mid)
        else:
            bi500=si500=bi200=si200=bi1000=si1000=[]; bw500=sw500=bw200=sw200=bw1000=sw1000=[]
        trade_impact_buy_500=float(sum(bi500)); trade_impact_sell_500=float(sum(si500))
        impact_200_num=sum(v*n for v,n in (bw200+sw200)); impact_200_den=sum(n for _,n in (bw200+sw200)); impact_200=_safe_div(impact_200_num,impact_200_den,clip=1e9)
        impact_1000_num=sum(v*n for v,n in (bw1000+sw1000)); impact_1000_den=sum(n for _,n in (bw1000+sw1000)); impact_1000=_safe_div(impact_1000_num,impact_1000_den,clip=1e9)
        event_burst_ratio=_safe_div(self.event_fast.value(ts),self.event_slow.value(ts))
        trade_burst_ratio=_safe_div(self.trade_fast.value(ts),self.trade_slow.value(ts))
        ob_burst_ratio=_safe_div(self.ob_fast.value(ts),self.ob_slow.value(ts))
        signs_1000=np.asarray([sg for t,sg in self.trade_side if t>=ts-1000],dtype=np.float64)
        signs_3000=np.asarray([sg for t,sg in self.trade_side if t>=ts-3000],dtype=np.float64)
        buy_notional_200=self.buy_hist.sum(200,ts); sell_notional_200=self.sell_hist.sum(200,ts); buy_notional_500=self.buy_hist.sum(500,ts); sell_notional_500=self.sell_hist.sum(500,ts); buy_notional_1000=self.buy_hist.sum(1000,ts); sell_notional_1000=self.sell_hist.sum(1000,ts)
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
        o["trade_sign_entropy_1000ms"]=_trade_sign_entropy(signs_1000)
        o["trade_sign_entropy_3000ms"]=_trade_sign_entropy(signs_3000)
        o["trade_sign_flip_rate_1000ms"]=float(np.mean(signs_1000[1:]!=signs_1000[:-1])) if signs_1000.size>1 else 0.0
        o["trade_sign_flip_rate_3000ms"]=float(np.mean(signs_3000[1:]!=signs_3000[:-1])) if signs_3000.size>1 else 0.0
        trk=[t for t in self.depletion_trackers if ts-t["ts"]<=200]; o["same_side_replenishment_after_depletion_200ms"]=_safe_div(sum(min(t["same_recovered"],t["amount"]) for t in trk),sum(t["amount"] for t in trk))
        o["opposite_side_replenishment_after_depletion_200ms"]=_safe_div(sum(min(t["opp_recovered"],t["amount"]) for t in trk),sum(t["amount"] for t in trk))
        if self.book_valid:
            o["buy_trade_depth_recovery_ratio_500ms"]=self._depth_recovery_ratio("buy",500,ts,depth_ask_5)
            o["sell_trade_depth_recovery_ratio_500ms"]=self._depth_recovery_ratio("sell",500,ts,depth_bid_5)
        else:
            o["buy_trade_depth_recovery_ratio_500ms"]=0.0
            o["sell_trade_depth_recovery_ratio_500ms"]=0.0
        o["trade_impact_decay_ratio_200_to_1000ms"]=_safe_div(impact_200,impact_1000)
        o["trade_impact_half_life_proxy"]=float(np.clip(_safe_div(math.log(max(impact_200,EPS)/max(impact_1000,EPS)),math.log(5.0),clip=10.0),-10,10)) if impact_200>EPS and impact_1000>EPS else 0.0
        o["depth_slope_bid_1_to_10"]=_safe_div(depth_bid_10-bid_depth_1,depth_bid_10)
        o["depth_slope_ask_1_to_10"]=_safe_div(depth_ask_10-ask_depth_1,depth_ask_10)
        o["depth_slope_imbalance_1_to_10"]=self._safe_asym(o["depth_slope_bid_1_to_10"],o["depth_slope_ask_1_to_10"])
        o["thin_side_depth_gap_ratio"]=_safe_div(abs(depth_bid_10-depth_ask_10),depth_bid_10+depth_ask_10)
        bid_convexity=_signed_safe_div(depth_bid_10-2.0*depth_bid_5+bid_depth_1,depth_bid_10,clip=1e9); ask_convexity=_signed_safe_div(depth_ask_10-2.0*depth_ask_5+ask_depth_1,depth_ask_10,clip=1e9); o["book_shape_asymmetry_convexity"]=self._safe_asym(bid_convexity,ask_convexity)
        o["bid_queue_cliff_ratio_l1_l5"]=_safe_div(bid_l1,bid_l5)
        o["ask_queue_cliff_ratio_l1_l5"]=_safe_div(ask_l1,ask_l5)
        bid_drop=_safe_div(max(0.0,bid_l1-bid_l2),bid_l1,clip=1e9); ask_drop=_safe_div(max(0.0,ask_l1-ask_l2),ask_l1,clip=1e9); o["near_touch_depth_drop_asymmetry"]=self._safe_asym(bid_drop,ask_drop)
        o["no_trade_no_book_change_age_ms"]=no_trade_no_book_age
        o["mid_unchanged_and_depth_stable_ms"]=mid_unchanged_depth_stable
        o["best_bid_price_age_ms"]=best_bid_price_age
        o["best_ask_price_age_ms"]=best_ask_price_age
        o["best_bid_size_age_ms"]=best_bid_size_age
        o["best_ask_size_age_ms"]=best_ask_size_age
        o["touch_price_age_min_ms"]=min(best_bid_price_age,best_ask_price_age)
        o["touch_price_age_max_ms"]=max(best_bid_price_age,best_ask_price_age)
        o["touch_price_age_imbalance_ms"]=self._safe_asym(bid_price_age:=best_bid_price_age,ask_price_age:=best_ask_price_age)
        o["touch_size_age_imbalance_ms"]=self._safe_asym(bid_size_age:=best_bid_size_age,ask_size_age:=best_ask_size_age)
        o["best_bid_replacement_count_1000ms"]=self.bid_rep.sum(1000,ts)
        o["best_ask_replacement_count_1000ms"]=self.ask_rep.sum(1000,ts)
        o["touch_replacement_imbalance_1000ms"]=self._safe_asym(self.bid_rep.sum(1000,ts),self.ask_rep.sum(1000,ts))
        o["touch_replacement_rate_3000ms"]=_safe_div(self.bid_rep.sum(3000,ts)+self.ask_rep.sum(3000,ts),3.0)
        o["quote_lifetime_cv_3000ms"]=_safe_div(self.quote_life.std(3000,ts),self.quote_life.mean(3000,ts))
        o["bid_l1_size_flip_rate_500ms"]=self.bid_delta_sign_hist.sign_flip_rate(500,ts)
        o["ask_l1_size_flip_rate_500ms"]=self.ask_delta_sign_hist.sign_flip_rate(500,ts)
        o["bid_l1_size_flip_rate_1000ms"]=self.bid_delta_sign_hist.sign_flip_rate(1000,ts)
        o["ask_l1_size_flip_rate_1000ms"]=self.ask_delta_sign_hist.sign_flip_rate(1000,ts)
        o["l1_size_flip_imbalance_1000ms"]=self._safe_asym(o["bid_l1_size_flip_rate_1000ms"],o["ask_l1_size_flip_rate_1000ms"])
        o["bid_l1_add_cancel_alternation_rate_1000ms"]=_safe_div(self.bid_alt_events.count(1000,ts),max(self.bid_delta_sign_hist.count(1000,ts)-1,1))
        o["ask_l1_add_cancel_alternation_rate_1000ms"]=_safe_div(self.ask_alt_events.count(1000,ts),max(self.ask_delta_sign_hist.count(1000,ts)-1,1))
        o["touch_flicker_score_1000ms"]=0.5*(o["bid_l1_add_cancel_alternation_rate_1000ms"]+o["ask_l1_add_cancel_alternation_rate_1000ms"])*_safe_div(self.churn.sum(1000,ts),depth_bid_10+depth_ask_10)
        o["touch_flicker_score_3000ms"]=0.5*(_safe_div(self.bid_alt_events.count(3000,ts),max(self.bid_delta_sign_hist.count(3000,ts)-1,1))+_safe_div(self.ask_alt_events.count(3000,ts),max(self.ask_delta_sign_hist.count(3000,ts)-1,1)))*_safe_div(self.churn.sum(3000,ts),depth_bid_10+depth_ask_10)
        o["post_buy_ask_cancel_over_trade_200ms"]=_safe_div(self.post_buy_ask_cancel_200.sum(200,ts),buy_notional_200)
        o["post_sell_bid_cancel_over_trade_200ms"]=_safe_div(self.post_sell_bid_cancel_200.sum(200,ts),sell_notional_200)
        o["post_buy_ask_net_replenishment_over_trade_200ms"]=_safe_div(self.post_buy_ask_net_200.sum(200,ts),buy_notional_200)
        o["post_sell_bid_net_replenishment_over_trade_200ms"]=_safe_div(self.post_sell_bid_net_200.sum(200,ts),sell_notional_200)
        o["post_buy_bid_add_over_trade_200ms"]=_safe_div(self.post_buy_bid_add_200.sum(200,ts),buy_notional_200)
        o["post_sell_ask_add_over_trade_200ms"]=_safe_div(self.post_sell_ask_add_200.sum(200,ts),sell_notional_200)
        o["post_buy_opposite_side_support_ratio_500ms"]=_safe_div(self.post_buy_bid_add_500.sum(500,ts),buy_notional_500)
        o["post_sell_opposite_side_support_ratio_500ms"]=_safe_div(self.post_sell_ask_add_500.sum(500,ts),sell_notional_500)
        o["trade_side_quote_response_asymmetry_500ms"]=self._safe_asym(o["post_buy_opposite_side_support_ratio_500ms"],o["post_sell_opposite_side_support_ratio_500ms"])
        o["last_buy_mid_impact_bps_since_trade"]=(1e4*(mid-self.last_buy_trade["mid_at_trade"])/max(self.last_buy_trade["mid_at_trade"],EPS)) if self.last_buy_trade and mid>0 else 0.0
        o["last_sell_mid_impact_bps_since_trade"]=(1e4*(self.last_sell_trade["mid_at_trade"]-mid)/max(self.last_sell_trade["mid_at_trade"],EPS)) if self.last_sell_trade and mid>0 else 0.0
        o["last_trade_mid_impact_signed_bps"]=((1e4*(mid-self.last_nonzero_trade["mid_at_trade"])/max(self.last_nonzero_trade["mid_at_trade"],EPS))*np.sign(self.last_nonzero_trade["side"])) if self.last_nonzero_trade and mid>0 else 0.0
        o["buy_trade_impact_sum_bps_500ms"]=_finite_float(trade_impact_buy_500,RATIO_CLIP)
        o["sell_trade_impact_sum_bps_500ms"]=_finite_float(trade_impact_sell_500,RATIO_CLIP)
        o["trade_impact_asymmetry_bps_500ms"]=self._safe_asym(trade_impact_buy_500,trade_impact_sell_500)
        o["buy_trade_impact_decay_200_to_1000ms"]=_safe_div(abs(sum(bi200)),abs(sum(bi1000)))
        o["sell_trade_impact_decay_200_to_1000ms"]=_safe_div(abs(sum(si200)),abs(sum(si1000)))
        o["impact_per_notional_buy_1000ms"]=_signed_safe_div(sum(bi1000),buy_notional_1000)
        o["impact_per_notional_sell_1000ms"]=_signed_safe_div(sum(si1000),sell_notional_1000)
        o["buy_trade_size_hhi_1000ms"]=self._hhi(buy_1000)
        o["sell_trade_size_hhi_1000ms"]=self._hhi(sell_1000)
        o["buy_trade_size_hhi_3000ms"]=self._hhi(buy_3000)
        o["sell_trade_size_hhi_3000ms"]=self._hhi(sell_3000)
        buy_largest_notional_3000=float(buy_3000.max()) if buy_3000.size else 0.0
        sell_largest_notional_3000=float(sell_3000.max()) if sell_3000.size else 0.0
        o["buy_largest_trade_share_3000ms"]=_safe_div(buy_largest_notional_3000,float(buy_3000.sum()) if buy_3000.size else 1.0)
        o["sell_largest_trade_share_3000ms"]=_safe_div(sell_largest_notional_3000,float(sell_3000.sum()) if sell_3000.size else 1.0)
        o["buy_trade_p90_over_median_3000ms"]=_safe_div(np.percentile(buy_3000,90) if buy_3000.size else 0.0,np.median(buy_3000) if buy_3000.size else 1.0)
        o["sell_trade_p90_over_median_3000ms"]=_safe_div(np.percentile(sell_3000,90) if sell_3000.size else 0.0,np.median(sell_3000) if sell_3000.size else 1.0)
        o["trade_size_concentration_asymmetry_3000ms"]=self._safe_asym(o["buy_trade_size_hhi_3000ms"],o["sell_trade_size_hhi_3000ms"])
        o["large_trade_side_dominance_3000ms"]=_safe_div(buy_largest_notional_3000-sell_largest_notional_3000,buy_largest_notional_3000+sell_largest_notional_3000)
        o["bid_depth_centroid_bps_10bps"]=bid_centroid_10
        o["ask_depth_centroid_bps_10bps"]=ask_centroid_10
        o["depth_centroid_imbalance_10bps"]=self._safe_asym(bid_centroid_10,ask_centroid_10)
        o["bid_depth_centroid_bps_25bps"]=bid_centroid_25
        o["ask_depth_centroid_bps_25bps"]=ask_centroid_25
        o["depth_centroid_imbalance_25bps"]=self._safe_asym(bid_centroid_25,ask_centroid_25)
        o["bid_near_touch_depth_share_10bps"]=_safe_div(bid_depth_1,depth_bid_10)
        o["ask_near_touch_depth_share_10bps"]=_safe_div(ask_depth_1,depth_ask_10)
        o["near_touch_depth_share_asymmetry_10bps"]=self._safe_asym(o["bid_near_touch_depth_share_10bps"],o["ask_near_touch_depth_share_10bps"])
        o["far_depth_wall_ratio_10_to_25bps"]=_safe_div((depth_bid_25-depth_bid_10)+(depth_ask_25-depth_ask_10),depth_bid_10+depth_ask_10)
        o["spread_widen_event_count_1000ms"]=self.spread_widen.sum(1000,ts)
        o["spread_tighten_event_count_1000ms"]=self.spread_tighten.sum(1000,ts)
        o["spread_widen_to_tighten_ratio_1000ms"]=_safe_div(self.spread_widen.sum(1000,ts),self.spread_tighten.sum(1000,ts))
        o["spread_state_transition_rate_3000ms"]=_safe_div(self.spread_widen.count(3000,ts)+self.spread_tighten.count(3000,ts),3.0)
        o["spread_one_tick_persistence_ms"]=min(max(ts-self.one_tick_spread_entry_ts,0.0),AGE_CLIP_MS) if self.in_one_tick_spread and self.one_tick_spread_entry_ts is not None else 0.0
        o["spread_wide_state_age_ms"]=min(max(ts-self.wide_spread_entry_ts,0.0),AGE_CLIP_MS) if self.in_wide_spread and self.wide_spread_entry_ts is not None else 0.0
        trade_count_500=self.trade_hist.count(500,ts)
        if trade_count_500<=0:
            o["spread_recompression_after_trade_500ms"]=0.0
            o["spread_widen_after_trade_500ms"]=0.0
        else:
            o["spread_recompression_after_trade_500ms"]=_finite_float(self.spread_tighten.sum(500,ts)/trade_count_500,RATIO_CLIP)
            o["spread_widen_after_trade_500ms"]=_finite_float(self.spread_widen.sum(500,ts)/trade_count_500,RATIO_CLIP)
        o["mid_price_direction_flip_rate_1000ms"]=self.mid_delta_sign_hist.sign_flip_rate(1000,ts)
        o["mid_price_run_length_current"]=float(self.current_mid_run_len)
        o["mid_price_run_length_max_3000ms"]=self.mid_run_len_hist.max(3000,ts)
        mid_vals_1000=self.mid_hist.values(1000,ts)
        mid_vals_3000=self.mid_hist.values(3000,ts)
        if mid_vals_1000.size<2:
            o["mid_price_path_efficiency_1000ms"]=0.0; o["mid_price_reversal_ratio_1000ms"]=0.0
        else:
            pe1000=_safe_div(abs(mid_vals_1000[-1]-mid_vals_1000[0]),max(np.abs(np.diff(mid_vals_1000)).sum(),EPS))
            o["mid_price_path_efficiency_1000ms"]=_finite_float(pe1000,RATIO_CLIP); o["mid_price_reversal_ratio_1000ms"]=_finite_float(1.0-pe1000,RATIO_CLIP)
        if mid_vals_3000.size<2:
            o["mid_price_path_efficiency_3000ms"]=0.0; o["mid_price_reversal_ratio_3000ms"]=0.0
        else:
            pe3000=_safe_div(abs(mid_vals_3000[-1]-mid_vals_3000[0]),max(np.abs(np.diff(mid_vals_3000)).sum(),EPS))
            o["mid_price_path_efficiency_3000ms"]=_finite_float(pe3000,RATIO_CLIP); o["mid_price_reversal_ratio_3000ms"]=_finite_float(1.0-pe3000,RATIO_CLIP)
        o["microprice_leads_mid_cross_count_1000ms"]=self.micro_lead_events.sum(1000,ts)
        o["microprice_mid_divergence_persistence_ms"]=(0.0 if self.divergence_start_ts is None else min(max(ts-self.divergence_start_ts,0.0),AGE_CLIP_MS))
        o["event_interarrival_p90_over_p10_1000ms"]=self.event_i.p90_over_p10(1000,ts)
        o["trade_interarrival_p90_over_p10_1000ms"]=self.trade_i.p90_over_p10(1000,ts)
        o["ob_interarrival_p90_over_p10_1000ms"]=self.ob_i.p90_over_p10(1000,ts)
        o["event_interarrival_entropy_3000ms"]=self.event_i.entropy(3000,ts)
        o["trade_arrival_clumpiness_3000ms"]=self.trade_i.clumpiness(3000,ts)
        o["ob_arrival_clumpiness_3000ms"]=self.ob_i.clumpiness(3000,ts)
        o["max_trade_silence_gap_3000ms"]=self.trade_i.max_gap(3000,ts)
        o["max_ob_silence_gap_3000ms"]=self.ob_i.max_gap(3000,ts)
        o["thin_book_with_trade_burst_score_500ms"]=_safe_div(max(self.depth_ewma_10bps.value(ts)-(depth_bid_10+depth_ask_10),0.0),self.depth_ewma_10bps.value(ts))*trade_burst_ratio
        o["thin_book_with_quote_flicker_score_1000ms"]=_safe_div(max(self.depth_ewma_10bps.value(ts)-(depth_bid_10+depth_ask_10),0.0),self.depth_ewma_10bps.value(ts))*o["touch_flicker_score_1000ms"]
        o["wide_spread_with_trade_burst_score_1000ms"]=(1.0 if self.in_wide_spread else 0.0)*trade_burst_ratio
        o["stale_touch_with_trade_burst_score_1000ms"]=(math.log1p(o["touch_price_age_min_ms"])/math.log1p(AGE_CLIP_MS))*trade_burst_ratio
        o["stale_touch_with_low_depth_score_1000ms"]=(math.log1p(o["touch_price_age_min_ms"])/math.log1p(AGE_CLIP_MS))*_safe_div(max(self.depth_ewma_10bps.value(ts)-(depth_bid_10+depth_ask_10),0.0),self.depth_ewma_10bps.value(ts))
        o["fresh_touch_with_high_depth_score_1000ms"]=(1.0/(1.0+o["touch_price_age_min_ms"]/1000.0))*_safe_div(max((depth_bid_10+depth_ask_10)-self.depth_ewma_10bps.value(ts),0.0),self.depth_ewma_10bps.value(ts))
        o["quote_pull_before_trade_burst_score_1000ms"]=_safe_div(self.bid_cancel.sum(1000,ts)+self.ask_cancel.sum(1000,ts),depth_bid_10+depth_ask_10)*trade_burst_ratio
        repl=_safe_div(self.bid_add.sum(1000,ts)+self.ask_add.sum(1000,ts),depth_bid_10+depth_ask_10,clip=1e9); o["trade_burst_without_book_replenishment_score_1000ms"]=_safe_div(trade_burst_ratio,1.0+repl,clip=1e9)
        o["depth_centroid_far_with_trade_burst_score_1000ms"]=o["far_depth_wall_ratio_10_to_25bps"]*trade_burst_ratio
        impact_hi=max(abs(o["impact_per_notional_buy_1000ms"]),abs(o["impact_per_notional_sell_1000ms"])); o["impact_per_notional_high_and_replenishment_low_score_1000ms"]=_safe_div(impact_hi,1.0+repl,clip=1e9)
        if not self.book_valid:
            for k in INVALID_BOOK_NEUTRAL_FEATURES:
                o[k]=0.0
        missing=set(ROUND2_REQUESTED_FEATURES)-set(o)
        extra=set(o)-set(ROUND2_REQUESTED_FEATURES)
        if missing or extra:
            raise RuntimeError(f"feature mismatch missing={missing} extra={extra}")
        return {k:_finite_float(o[k]) for k in ROUND2_REQUESTED_FEATURES}
