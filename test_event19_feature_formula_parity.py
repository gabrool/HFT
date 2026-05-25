import math
import numpy as np
import pytest

from CMSSL17 import FeatureEngine, ROUND2_PRODUCTION_EVENT_FEATURES, FEATURE_SCHEMA, FEATURE_TRANSFORM, CHECKPOINT_SCHEMA, build_feature_transform_specs, AUX_DIM
from feature_event_candidates_round2 import NovelMicrostructureCandidatePack


def make_engine_identity():
    eng = FeatureEngine()
    eng._transform_features = lambda raw, dt_ms: np.asarray(raw, dtype=np.float32)
    return eng


def feed_both(events):
    eng = make_engine_identity()
    ref = NovelMicrostructureCandidatePack()
    out = []
    for ev in events:
        cmssl_out = eng.on_fast_event(ev)
        ref.on_event(ev)
        if ev[0] == 'ob':
            assert cmssl_out.is_decision
            out.append((dict(zip(eng.feature_names(), cmssl_out.features)), ref.emit()))
        else:
            assert not cmssl_out.is_decision
    return out


def test_event19_schema_and_feature_count():
    names = FeatureEngine().feature_names()
    assert len(names) == 172
    assert len(set(names)) == len(names)
    assert len(ROUND2_PRODUCTION_EVENT_FEATURES) == 19


def test_event19_transform_specs_cover_all_features():
    names = FeatureEngine().feature_names(); specs = build_feature_transform_specs(names)
    assert len(specs) == len(names)
    smap = {s.name: s for s in specs}
    for n in ROUND2_PRODUCTION_EVENT_FEATURES: assert n in smap

RICH_VALID_EVENTS = [
    ("ob",0,1,1,[(100.00,10),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,9),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("trade",50,2,100.02,2.0,1,1,0),
    ("ob",100,3,1,[(100.00,6),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,9),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("ob",150,4,1,[(100.00,13),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,9),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("ob",200,5,1,[(100.00,13),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,13),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("trade",250,6,100.00,3.0,-1,-1,0),("ob",300,7,1,[(100.00,13),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,6),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("ob",350,8,1,[(100.00,13),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,14),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("ob",400,9,1,[(100.00,15),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,14),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
    ("trade",500,10,100.02,1.5,1,1,0),("trade",900,11,100.00,4.0,-1,-1,0),
    ("ob",1000,12,1,[(100.01,9),(100.00,8),(99.99,5),(99.97,4),(99.95,3)],[(100.03,11),(100.04,7),(100.05,5),(100.07,4),(100.09,3)]),
    ("ob",1200,13,1,[(100.02,10),(100.01,8),(100.00,5),(99.98,4),(99.96,3)],[(100.04,9),(100.05,7),(100.06,5),(100.08,4),(100.10,3)]),
    ("ob",1400,14,1,[(100.01,12),(100.00,8),(99.99,5),(99.97,4),(99.95,3)],[(100.03,8),(100.04,7),(100.05,5),(100.07,4),(100.09,3)]),
    ("trade",2500,15,100.03,5.0,1,1,0),
    ("ob",2600,16,1,[(100.01,5),(100.00,8),(99.99,5),(99.97,4),(99.95,3)],[(100.04,16),(100.05,7),(100.06,5),(100.08,4),(100.10,3)]),
    ("ob",2800,17,1,[(100.00,14),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.03,7),(100.04,7),(100.05,5),(100.07,4),(100.09,3)]),
    ("ob",3200,18,1,[(100.02,9),(100.01,8),(100.00,5),(99.98,4),(99.96,3)],[(100.05,13),(100.06,7),(100.07,5),(100.09,4),(100.11,3)]),
]

def test_event19_rich_formula_parity_and_nonzero():
    cmssl, ref = feed_both(RICH_VALID_EVENTS)[-1]
    for n in ROUND2_PRODUCTION_EVENT_FEATURES:
        assert float(cmssl[n]) == pytest.approx(float(ref[n]), rel=1e-5, abs=1e-5), n
    must=["touch_flicker_score_3000ms","spread_state_transition_rate_3000ms","max_trade_silence_gap_3000ms","ask_depth_centroid_bps_25bps","bid_depth_centroid_bps_25bps","microprice_realized_vol_1000ms","buy_trade_p90_over_median_3000ms","sell_trade_p90_over_median_3000ms","ob_arrival_clumpiness_3000ms","trade_sign_entropy_3000ms","mid_price_run_length_max_3000ms","best_ask_size_age_ms","opposite_side_replenishment_after_depletion_200ms","same_side_replenishment_after_depletion_200ms","trade_side_quote_response_asymmetry_500ms","near_touch_depth_drop_asymmetry"]
    for n in must: assert abs(float(cmssl[n])) > 1e-12, n

def test_offline_ingest_uses_dynamic_round2_dim():
    import offline_ingest
    names = FeatureEngine().feature_names()
    assert offline_ingest.RAW_FEATURE_DIM_CORE == len(names) == 172
    assert offline_ingest.RAW_FEATURE_DIM_TOTAL == len(names) + AUX_DIM
    assert "pruned172" in offline_ingest.FEATURE_SCHEMA
    assert "round2xformv3" in offline_ingest.FEATURE_TRANSFORM
    assert "pruned172" in offline_ingest.CHECKPOINT_SCHEMA

def _last(events): return feed_both(events)[-1]

def test_event19_depth_centroid_25bps_hand_formula():
    ev=[('ob',0,1,1,[(100.0,10),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,9),(100.03,7),(100.04,5),(100.06,4),(100.08,3)])]
    cm,rf=_last(ev); mid=100.01
    def c(levels):
        n=sum((1e4*abs(p-mid)/mid)*(p*s) for p,s in levels); d=sum(p*s for p,s in levels); return n/d
    assert cm['bid_depth_centroid_bps_25bps']==pytest.approx(c(ev[0][4]),rel=1e-6,abs=1e-6)
    assert cm['ask_depth_centroid_bps_25bps']==pytest.approx(c(ev[0][5]),rel=1e-6,abs=1e-6)
    assert cm['bid_depth_centroid_bps_25bps']==pytest.approx(rf['bid_depth_centroid_bps_25bps'],rel=1e-6,abs=1e-6)

def test_event19_trade_sign_entropy_3000ms_hand_formula():
    ev=[('ob',0,1,1,[(100,10)],[(100.02,10)]),('trade',10,2,100.02,1,1,1,0),('trade',20,3,100.02,1,1,1,0),('trade',30,4,100.0,1,-1,-1,0),('ob',40,5,1,[(100,10)],[(100.02,10)])]
    cm,rf=_last(ev); pb,ps=2/3,1/3; e=-(pb*math.log(pb)+ps*math.log(ps))/math.log(2)
    assert cm['trade_sign_entropy_3000ms']==pytest.approx(e,rel=1e-6,abs=1e-6)
    assert cm['trade_sign_entropy_3000ms']==pytest.approx(rf['trade_sign_entropy_3000ms'],rel=1e-6,abs=1e-6)

def test_event19_side_trade_p90_over_median_3000ms_hand_formula():
    ev=[('ob',0,1,1,[(100,10)],[(100.02,10)]),('trade',10,2,100,1,1,1,0),('trade',20,3,100,2,1,1,0),('trade',30,4,100,3,1,1,0),('trade',40,5,100,1,-1,-1,0),('trade',50,6,100,4,-1,-1,0),('ob',60,7,1,[(100,10)],[(100.02,10)])]
    cm,rf=_last(ev)
    b=np.array([100,200,300.],dtype=float); s=np.array([100,400.],dtype=float)
    assert cm['buy_trade_p90_over_median_3000ms']==pytest.approx(np.percentile(b,90)/np.median(b),rel=1e-6,abs=1e-6)
    assert cm['sell_trade_p90_over_median_3000ms']==pytest.approx(np.percentile(s,90)/np.median(s),rel=1e-6,abs=1e-6)
    assert cm['sell_trade_p90_over_median_3000ms']==pytest.approx(rf['sell_trade_p90_over_median_3000ms'],rel=1e-6,abs=1e-6)

def test_event19_interarrival_max_gap_and_ob_clumpiness():
    ev=[('ob',0,1,1,[(100,10)],[(100.02,10)]),('trade',100,2,100.02,1,1,1,0),('ob',100,3,1,[(100,10)],[(100.02,10)]),('ob',300,4,1,[(100,10)],[(100.02,10)]),('trade',700,5,100.0,1,-1,-1,0),('ob',1000,6,1,[(100,10)],[(100.02,10)]),('trade',2500,7,100.02,1,1,1,0),('ob',3200,8,1,[(100,10)],[(100.02,10)])]
    cm,rf=_last(ev)
    assert cm['max_trade_silence_gap_3000ms']==pytest.approx(1800.0,abs=1e-6)
    assert cm['ob_arrival_clumpiness_3000ms']==pytest.approx(rf['ob_arrival_clumpiness_3000ms'],rel=1e-6,abs=1e-6)

def test_event19_best_size_age_only_resets_on_size_change():
    ev=[('ob',0,1,1,[(100,10)],[(100.02,9)]),('ob',100,2,1,[(100.01,10)],[(100.03,9)]),('ob',300,3,1,[(100.01,12)],[(100.03,9)]),('ob',500,4,1,[(100.01,12)],[(100.03,9)])]
    cm,rf=_last(ev)
    assert cm['best_bid_size_age_ms']==pytest.approx(200.0,abs=1e-6)
    assert cm['best_ask_size_age_ms']==pytest.approx(500.0,abs=1e-6)
    assert cm['best_ask_size_age_ms']==pytest.approx(rf['best_ask_size_age_ms'],rel=1e-6,abs=1e-6)

def test_event19_trade_side_quote_response_asymmetry():
    ev=[('ob',0,1,1,[(100,10)],[(100.02,10)]),('trade',50,2,100.02,2,1,1,0),('ob',100,3,1,[(100,12)],[(100.02,10)]),('trade',200,4,100.0,2,-1,-1,0),('ob',250,5,1,[(100,12)],[(100.02,12)]),('ob',300,6,1,[(100,12)],[(100.02,12)])]
    cm,rf=_last(ev)
    assert np.isfinite(cm['trade_side_quote_response_asymmetry_500ms'])
    assert cm['trade_side_quote_response_asymmetry_500ms']==pytest.approx(rf['trade_side_quote_response_asymmetry_500ms'],rel=1e-6,abs=1e-6)

def test_event19_trade_impact_mid_fallback_before_book():
    ev=[('trade',0,1,100.0,1,1,1,0),('ob',100,2,1,[(100,10)],[(100.02,10)])]
    cm,rf=_last(ev)
    assert np.isfinite(cm['trade_impact_half_life_proxy'])
    assert cm['trade_impact_half_life_proxy']==pytest.approx(rf['trade_impact_half_life_proxy'],rel=1e-6,abs=1e-6)

def test_event19_near_touch_depth_drop_asymmetry_hand_formula():
    ev=[("ob",0,1,1,[(100.00,20.0),(99.99,5.0),(99.98,4.0)],[(100.02,10.0),(100.03,9.5),(100.04,4.0)])]
    cm,rf=_last(ev); eps=1e-12
    bid_l1,bid_l2=100.00*20.0,99.99*5.0; ask_l1,ask_l2=100.02*10.0,100.03*9.5
    bid_drop=max(0.0,bid_l1-bid_l2)/max(bid_l1,eps); ask_drop=max(0.0,ask_l1-ask_l2)/max(ask_l1,eps)
    exp=(bid_drop-ask_drop)/max(abs(bid_drop)+abs(ask_drop),eps)
    assert exp > 0 and np.isfinite(cm["near_touch_depth_drop_asymmetry"])
    assert cm["near_touch_depth_drop_asymmetry"]==pytest.approx(exp,rel=1e-6,abs=1e-6)
    assert cm["near_touch_depth_drop_asymmetry"]==pytest.approx(rf["near_touch_depth_drop_asymmetry"],rel=1e-6,abs=1e-6)

def test_event19_mid_unchanged_and_depth_stable_age_hand_formula():
    ev=[("ob",0,1,1,[(100.00,10.0),(99.99,8.0)],[(100.02,10.0),(100.03,8.0)]),("ob",100,2,1,[(100.00,10.02),(99.99,8.0)],[(100.02,10.0),(100.03,8.0)]),("ob",200,3,1,[(100.00,10.03),(99.99,8.0)],[(100.02,10.0),(100.03,8.0)]),("ob",300,4,1,[(100.00,12.0),(99.99,8.0)],[(100.02,10.0),(100.03,8.0)]),("ob",500,5,1,[(100.00,12.0),(99.99,8.0)],[(100.02,10.0),(100.03,8.0)])]
    cm,rf=_last(ev)
    assert np.isfinite(cm["mid_unchanged_and_depth_stable_ms"])
    assert cm["mid_unchanged_and_depth_stable_ms"]==pytest.approx(200.0,rel=1e-6,abs=1e-6)
    assert cm["mid_unchanged_and_depth_stable_ms"]==pytest.approx(rf["mid_unchanged_and_depth_stable_ms"],rel=1e-6,abs=1e-6)

def test_event19_depletion_replenishment_ratios_hand_formula():
    ev=[("ob",0,1,1,[(100.00,10.0)],[(100.02,10.0)]),("ob",100,2,1,[(100.00,6.0)],[(100.02,10.0)]),("ob",150,3,1,[(100.00,8.0)],[(100.02,10.0)]),("ob",180,4,1,[(100.00,8.0)],[(100.02,11.0)]),("ob",250,5,1,[(100.00,8.0)],[(100.02,11.0)])]
    cm,rf=_last(ev); same=200.0/400.0; opp=100.02/400.0
    assert cm["same_side_replenishment_after_depletion_200ms"]==pytest.approx(same,rel=1e-6,abs=1e-6)
    assert cm["opposite_side_replenishment_after_depletion_200ms"]==pytest.approx(opp,rel=1e-6,abs=1e-6)
    assert cm["same_side_replenishment_after_depletion_200ms"]==pytest.approx(rf["same_side_replenishment_after_depletion_200ms"],rel=1e-6,abs=1e-6)
    assert cm["opposite_side_replenishment_after_depletion_200ms"]==pytest.approx(rf["opposite_side_replenishment_after_depletion_200ms"],rel=1e-6,abs=1e-6)

def test_event19_trade_side_quote_response_asymmetry_hand_formula():
    ev=[("ob",0,1,1,[(100.00,10.0)],[(100.02,10.0)]),("trade",50,2,100.02,2.0,1,1,0),("ob",100,3,1,[(100.00,12.0)],[(100.02,10.0)]),("trade",200,4,100.00,4.0,-1,-1,0),("ob",250,5,1,[(100.00,12.0)],[(100.02,11.0)]),("ob",300,6,1,[(100.00,12.0)],[(100.02,11.0)])]
    cm,rf=_last(ev); eps=1e-12
    pb=200.0/200.04; ps=100.02/400.0; exp=(pb-ps)/(abs(pb)+abs(ps)+eps)
    assert np.isfinite(cm["trade_side_quote_response_asymmetry_500ms"])
    assert cm["trade_side_quote_response_asymmetry_500ms"]==pytest.approx(exp,rel=1e-6,abs=1e-6)
    assert cm["trade_side_quote_response_asymmetry_500ms"]==pytest.approx(rf["trade_side_quote_response_asymmetry_500ms"],rel=1e-6,abs=1e-6)

def test_event19_trade_impact_half_life_proxy_hand_formula():
    ev=[("ob",0,1,1,[(99.99,10.0)],[(100.01,10.0)]),("trade",100,2,100.01,2.0,1,1,0),("ob",600,3,1,[(99.99,11.0)],[(100.01,9.0)]),("trade",700,4,99.99,1.0,-1,-1,0),("ob",800,5,1,[(100.04,10.0)],[(100.06,10.0)])]
    cm,rf=_last(ev)
    mid_now=100.05; mid_t=100.0; n_buy=200.02; n_sell=99.99
    bi=1e4*(mid_now-mid_t)/mid_t; si=1e4*(mid_t-mid_now)/mid_t
    i200=abs(si); i1000=(abs(bi)*n_buy+abs(si)*n_sell)/(n_buy+n_sell)
    exp=math.log(i200/i1000)/math.log(5.0)
    assert np.isfinite(exp) and np.isfinite(cm["trade_impact_half_life_proxy"])
    assert cm["trade_impact_half_life_proxy"]==pytest.approx(exp,rel=1e-6,abs=1e-6)
    assert cm["trade_impact_half_life_proxy"]==pytest.approx(rf["trade_impact_half_life_proxy"],rel=1e-6,abs=1e-6)

def test_event19_3000ms_windows_do_not_use_stale_values():
    ev=[("ob",0,1,1,[(100.00,10.0)],[(100.02,10.0)]),("trade",100,2,100.02,2.0,1,1,0),("ob",200,3,1,[(100.01,10.0)],[(100.03,10.0)]),("trade",300,4,100.01,3.0,-1,-1,0),("ob",500,5,1,[(100.00,9.0)],[(100.02,11.0)]),("ob",5000,6,1,[(100.00,9.0)],[(100.02,11.0)])]
    cm,rf=_last(ev)
    for n in ROUND2_PRODUCTION_EVENT_FEATURES:
        assert float(cm[n]) == pytest.approx(float(rf[n]), rel=1e-5, abs=1e-5), n
    assert cm["buy_trade_p90_over_median_3000ms"]==pytest.approx(0.0,abs=1e-6)
    assert cm["sell_trade_p90_over_median_3000ms"]==pytest.approx(0.0,abs=1e-6)
    assert cm["touch_flicker_score_3000ms"]==pytest.approx(0.0,abs=1e-6)
    assert cm["max_trade_silence_gap_3000ms"]==pytest.approx(0.0,abs=1e-6)

def test_event19_deterministic_random_snapshot_parity():
    rng=np.random.default_rng(17); t=0; mid=100.0
    bid,ask=99.99,100.01
    bs=np.array([10,9,8,7,6.],dtype=float); ass=np.array([10,9,8,7,6.],dtype=float)
    events=[("ob",t,1,1,[(bid-0.01*i,float(bs[i])) for i in range(5)],[(ask+0.01*i,float(ass[i])) for i in range(5)])]
    seq=2
    for _ in range(160):
        t += int(rng.integers(10,80))
        if rng.random() < 0.6:
            mid += float(rng.choice([-0.01,0.0,0.01]))
            spr = float(rng.choice([0.02,0.03]))
            bid,ask=mid-spr/2,mid+spr/2
            bs=np.maximum(1.0,bs+rng.normal(0,0.8,5))
            ass=np.maximum(1.0,ass+rng.normal(0,0.8,5))
            events.append(("ob",t,seq,1,[(round(bid-0.01*i,2),float(bs[i])) for i in range(5)],[(round(ask+0.01*i,2),float(ass[i])) for i in range(5)]))
        else:
            side=1 if rng.random()<0.5 else -1
            px=round(ask,2) if side>0 else round(bid,2)
            sz=float(rng.uniform(0.2,5.0))
            events.append(("trade",t,seq,px,sz,side,side,0))
        seq += 1
    for cm,rf in feed_both(events):
        for n in ROUND2_PRODUCTION_EVENT_FEATURES:
            assert float(cm[n]) == pytest.approx(float(rf[n]), rel=1e-5, abs=1e-5), n
