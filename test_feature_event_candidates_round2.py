import inspect, math
import numpy as np
from feature_event_candidates_round2 import FAMILY_BY_FEATURE, ROUND2_REQUESTED_FEATURES, NovelMicrostructureCandidatePack

def _feed(p, evs):
    for e in evs: p.on_event(e)

def test_exact_feature_list_and_metadata_keys():
    p=NovelMicrostructureCandidatePack()
    assert len(ROUND2_REQUESTED_FEATURES)==136
    assert len(set(ROUND2_REQUESTED_FEATURES))==136
    assert p.feature_names()==ROUND2_REQUESTED_FEATURES
    assert set(p.metadata())==set(ROUND2_REQUESTED_FEATURES)
    assert set(FAMILY_BY_FEATURE)==set(ROUND2_REQUESTED_FEATURES)

def test_source_guards_and_explicit_assignments():
    src=''.join(inspect.getsource(NovelMicrostructureCandidatePack.emit).split())
    for bad in ['{n:0.0forninROUND2_REQUESTED_FEATURES}','fork,vinconst.items()','forninROUND2_REQUESTED_FEATURES:o[n]=o[n]','setdefault(','.get(k,0.0)','.get(name,0.0)']:
        assert bad not in src
    must=["same_side_replenishment_after_depletion_200ms","post_buy_ask_cancel_over_trade_200ms","mid_price_path_efficiency_1000ms","thin_book_with_trade_burst_score_500ms"]
    for n in must:
        assert f'o["{n}"]=0.0' not in src and f"o['{n}']=0.0" not in src
    src2=inspect.getsource(NovelMicrostructureCandidatePack.emit)
    for n in ROUND2_REQUESTED_FEATURES: assert f'o["{n}"]' in src2 or f"o['{n}']" in src2

def test_formula_hhi_sign_queue_centroid_near_touch_and_churn():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(99.99,10),(99.98,8),(99.97,5),(99.96,4),(99.95,3)],[(100.01,10),(100.02,8),(100.03,5),(100.04,4),(100.05,3)]),("trade",100,2,100,1,1,1,0),("trade",200,3,100,2,1,1,0),("trade",300,4,100,3,-1,-1,0)])
    o=p.emit(); n=np.asarray([100.0,200.0,300.0])
    assert math.isclose(o['trade_size_hhi_1000ms'], float((n*n).sum()/n.sum()**2), rel_tol=1e-12)
    assert math.isclose(o['largest_trade_share_notional_1000ms'],0.5, rel_tol=1e-12)
    assert math.isclose(o['buy_trade_size_hhi_1000ms'],(100**2+200**2)/300**2, rel_tol=1e-12)
    assert math.isclose(o['sell_trade_size_hhi_1000ms'],1.0, rel_tol=1e-12)
    ent=-(2/3*math.log(2/3)+1/3*math.log(1/3))/math.log(2); assert math.isclose(o['trade_sign_entropy_1000ms'],ent, rel_tol=1e-12)
    assert math.isclose(o['trade_sign_flip_rate_1000ms'],0.5, rel_tol=1e-12)
    assert math.isclose(o['bid_queue_cliff_ratio_l1_l5'],(99.99*10)/(99.95*3), rel_tol=1e-12)
    assert math.isclose(o['ask_queue_cliff_ratio_l1_l5'],(100.01*10)/(100.05*3), rel_tol=1e-12)
    assert o['bid_near_touch_depth_share_10bps']>0 and o['ask_near_touch_depth_share_10bps']>0

def test_rich_nonzero_coverage():
    p=NovelMicrostructureCandidatePack(); ev=[('ob',0,1,1,[(99.99,10),(99.98,8),(99.97,5),(99.95,4),(99.90,3),(99.80,2)],[(100.01,9),(100.02,7),(100.03,5),(100.05,4),(100.10,3),(100.20,2)])]
    ev += [('trade',50,2,100.01,2,1,1,0),('ob',80,3,2,[],[(100.01,5)]),('ob',120,4,2,[],[(100.01,12)]),('ob',150,5,2,[(99.99,13)],[]),('trade',200,6,99.99,3,-1,-1,0),('ob',230,7,2,[(99.99,6)],[]),('ob',270,8,2,[(99.99,14)],[]),('ob',310,9,2,[],[(100.01,14)]),('ob',360,10,2,[(100.00,10)],[(100.01,14)]),('ob',420,11,2,[(100.00,10)],[(100.02,10)]),('trade',430,12,100.02,1,1,1,0),('trade',460,13,100.00,5,-1,-1,0),('trade',530,14,100.01,8,1,1,0),('ob',680,15,2,[(100.00,4)],[(100.02,16)])]
    _feed(p,ev); o=p.emit()
    assert set(o)==set(ROUND2_REQUESTED_FEATURES)
    assert np.isfinite(np.asarray(list(o.values()),dtype=float)).all()
    assert sum(abs(float(v))>1e-12 for v in o.values())>=80
