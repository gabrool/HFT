import inspect, math
import numpy as np
import pandas as pd
from feature_event_candidates_round2 import FAMILY_BY_FEATURE, ROUND2_REQUESTED_FEATURES, NovelMicrostructureCandidatePack, EWMAValue, RATIO_CLIP, RollingValueWindow

def _feed(p, evs):
    for e in evs:
        p.on_event(e)

def test_exact_feature_list_and_metadata_keys():
    p=NovelMicrostructureCandidatePack()
    assert getattr(NovelMicrostructureCandidatePack, "name") == "novel_microstructure_round2_v1"
    assert len(ROUND2_REQUESTED_FEATURES)==136
    assert len(set(ROUND2_REQUESTED_FEATURES))==136
    assert p.feature_names()==ROUND2_REQUESTED_FEATURES
    assert set(p.metadata())==set(ROUND2_REQUESTED_FEATURES)
    assert set(FAMILY_BY_FEATURE)==set(ROUND2_REQUESTED_FEATURES)

def test_source_guards_and_explicit_assignments():
    src=inspect.getsource(NovelMicrostructureCandidatePack.emit)
    compact=''.join(src.split())
    for bad in ['o.update({k:0.0','dict.fromkeys(ROUND2_REQUESTED_FEATURES','forkinROUND2_REQUESTED_FEATURES:o[k]=0.0','.get(k,0.0)']:
        assert bad not in compact
    for n in ROUND2_REQUESTED_FEATURES:
        assert f'o["{n}"]' in src or f"o['{n}']" in src



def test_emit_does_not_use_book_helpers_after_invalid_branch_setup():
    src=inspect.getsource(NovelMicrostructureCandidatePack.emit)
    marker='o["trade_size_hhi_3000ms"]'
    assert marker in src
    formula_src=src[src.index(marker):]
    forbidden=["self._depth(","self._level_notional(","self._levels(","self._depth_centroid("]
    for token in forbidden:
        assert token not in formula_src, token

def test_round2_interarrival_queries_are_non_mutating():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("trade",70,2,100.01,1,1,1,0),("ob",210,3,2,[(100.0,11)],[]),("trade",430,4,100.01,2,-1,-1,0),("ob",900,5,2,[],[(100.02,11)])])
    now=p.ts
    first=p.event_i.p90_over_p10(1000,now)
    _=p.event_i.p90_over_p10(200,now)
    _=p.event_i.cv(500,now)
    _=p.event_i.max_gap(1000,now)
    second=p.event_i.p90_over_p10(1000,now)
    assert first==second

def test_round2_no_future_leakage_prefix_emits_identical_values():
    prefix=[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("trade",100,2,100.02,2,1,1,0),("ob",150,3,2,[(100.0,9)],[(100.02,11)])]
    future=[("trade",500,4,100.0,3,-1,-1,0),("ob",700,5,2,[(99.99,12)],[(100.03,8)])]
    p1=NovelMicrostructureCandidatePack(); _feed(p1,prefix); out1=p1.emit()
    p2=NovelMicrostructureCandidatePack(); _feed(p2,prefix); out2_before=p2.emit()
    assert out1==out2_before
    _feed(p2,future); _=p2.emit()

def test_round2_crossed_book_emits_finite_neutral_book_features():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[
        ("ob",0,1,1,[(100.00,10),(99.99,8),(99.98,5)],[(100.02,10),(100.03,8),(100.04,5)]),
        ("trade",20,2,100.02,2.0,1,1,0),
        ("trade",40,3,100.00,2.0,-1,-1,0),
    ])
    prev_l1=p.last_l1_change_ts
    prev_one=p.one_tick_spread_entry_ts
    prev_wide=p.wide_spread_entry_ts
    p.on_event(("ob",100,4,1,[(100.03,10)],[(100.01,10)]))
    o=p.emit()
    assert p.book_valid is False
    assert set(o)==set(ROUND2_REQUESTED_FEATURES)
    assert np.isfinite(np.asarray(list(o.values()),dtype=float)).all()
    neutral_book_features=[
        "depth_slope_bid_1_to_10","depth_slope_ask_1_to_10","depth_slope_imbalance_1_to_10",
        "thin_side_depth_gap_ratio","book_shape_asymmetry_convexity","bid_queue_cliff_ratio_l1_l5",
        "ask_queue_cliff_ratio_l1_l5","near_touch_depth_drop_asymmetry","bid_depth_centroid_bps_10bps",
        "ask_depth_centroid_bps_10bps","depth_centroid_imbalance_10bps","bid_depth_centroid_bps_25bps",
        "ask_depth_centroid_bps_25bps","depth_centroid_imbalance_25bps","bid_near_touch_depth_share_10bps",
        "ask_near_touch_depth_share_10bps","near_touch_depth_share_asymmetry_10bps","far_depth_wall_ratio_10_to_25bps",
        "spread_one_tick_persistence_ms","spread_wide_state_age_ms",
        "buy_trade_depth_recovery_ratio_500ms","sell_trade_depth_recovery_ratio_500ms",
    ]
    for k in neutral_book_features:
        assert np.isclose(float(o[k]),0.0,atol=1e-12),(k,o[k])
    assert p.last_l1_change_ts == prev_l1
    assert p.one_tick_spread_entry_ts == prev_one
    assert p.wide_spread_entry_ts == prev_wide



def test_round2_invalid_book_emit_does_not_use_crossed_dictionaries():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)])])
    p.on_event(("ob",100,2,1,[(100.03,10)],[(100.01,10)]))
    o=p.emit()
    assert p.book_valid is False
    assert np.isclose(o["bid_queue_cliff_ratio_l1_l5"],0.0,atol=1e-12)
    assert np.isclose(o["ask_queue_cliff_ratio_l1_l5"],0.0,atol=1e-12)
    assert np.isclose(o["depth_slope_bid_1_to_10"],0.0,atol=1e-12)
    assert np.isclose(o["depth_slope_ask_1_to_10"],0.0,atol=1e-12)
    assert np.isclose(o["bid_depth_centroid_bps_10bps"],0.0,atol=1e-12)
    assert np.isclose(o["ask_depth_centroid_bps_10bps"],0.0,atol=1e-12)

def test_round2_touch_age_and_spread_state_formulas():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.00,10)],[(100.02,10)]),("ob",100,2,2,[],[]),("ob",300,3,2,[(100.00,12)],[]),("trade",450,4,100.01,1,1,1,0),("ob",600,5,1,[(99.98,12)],[(100.04,10)]),("ob",900,6,2,[],[])])
    o=p.emit()
    assert np.isclose(o["best_bid_price_age_ms"],300,atol=1e-9)
    assert np.isclose(o["best_ask_price_age_ms"],300,atol=1e-9)
    assert np.isclose(o["best_bid_size_age_ms"],600,atol=1e-9)
    assert np.isclose(o["touch_price_age_min_ms"],300,atol=1e-9)
    assert np.isclose(o["touch_price_age_max_ms"],300,atol=1e-9)
    assert np.isclose(o["touch_price_age_imbalance_ms"],0.0,atol=1e-9)
    expected_size_imb=(600-900)/(600+900)
    assert np.isclose(o["touch_size_age_imbalance_ms"],expected_size_imb,atol=1e-9)
    assert -1.0 <= o["touch_price_age_imbalance_ms"] <= 1.0
    assert -1.0 <= o["touch_size_age_imbalance_ms"] <= 1.0
    assert np.isclose(o["spread_wide_state_age_ms"],300,atol=1e-9)
    assert np.isclose(o["spread_one_tick_persistence_ms"],0.0,atol=1e-9)
    assert np.isclose(o["no_trade_no_book_change_age_ms"],300,atol=1e-9)
    assert np.isclose(o["mid_unchanged_and_depth_stable_ms"],600,atol=1e-9)

def test_formula_hhi_sign_queue_centroid_and_near_touch():
    p=NovelMicrostructureCandidatePack()
    bids=[(99.99,10),(99.98,8),(99.97,5),(99.96,4),(99.95,3)]
    asks=[(100.01,10),(100.02,8),(100.03,5),(100.04,4),(100.05,3)]
    _feed(p,[("ob",0,1,1,bids,asks),("trade",100,2,100,1,1,1,0),("trade",200,3,100,2,1,1,0),("trade",300,4,100,3,-1,-1,0)])
    o=p.emit(); n=np.asarray([100.0,200.0,300.0])
    assert math.isclose(o['trade_size_hhi_1000ms'], float((n*n).sum()/n.sum()**2), rel_tol=1e-12)
    assert math.isclose(o['largest_trade_share_notional_1000ms'],0.5, rel_tol=1e-12)
    p_buy=2.0/3.0; p_sell=1.0/3.0
    expected_entropy=-(p_buy*math.log(p_buy)+p_sell*math.log(p_sell))/math.log(2.0)
    assert math.isclose(o["trade_sign_entropy_1000ms"], expected_entropy, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(o["trade_sign_flip_rate_1000ms"],0.5,rel_tol=1e-12,abs_tol=1e-12)
    expected_bid_l1=99.99*10; expected_bid_l5=99.95*3
    expected_ask_l1=100.01*10; expected_ask_l5=100.05*3
    assert math.isclose(o["bid_queue_cliff_ratio_l1_l5"],expected_bid_l1/expected_bid_l5,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(o["ask_queue_cliff_ratio_l1_l5"],expected_ask_l1/expected_ask_l5,rel_tol=1e-12,abs_tol=1e-12)
    mid=100.0
    bid_not=[px*sz for px,sz in bids if px>=mid*(1-10/1e4)]
    bid_dist=[1e4*abs(px-mid)/mid for px,_ in bids if px>=mid*(1-10/1e4)]
    ask_not=[px*sz for px,sz in asks if px<=mid*(1+10/1e4)]
    ask_dist=[1e4*abs(px-mid)/mid for px,_ in asks if px<=mid*(1+10/1e4)]
    expected_bid_centroid=sum(d*n for d,n in zip(bid_dist,bid_not))/sum(bid_not)
    expected_ask_centroid=sum(d*n for d,n in zip(ask_dist,ask_not))/sum(ask_not)
    assert math.isclose(o["bid_depth_centroid_bps_10bps"],expected_bid_centroid,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(o["ask_depth_centroid_bps_10bps"],expected_ask_centroid,rel_tol=1e-12,abs_tol=1e-12)
    bid_depth_1=sum(px*sz for px,sz in bids if px>=mid*(1-1/1e4)); bid_depth_10=sum(bid_not)
    ask_depth_1=sum(px*sz for px,sz in asks if px<=mid*(1+1/1e4)); ask_depth_10=sum(ask_not)
    assert math.isclose(o["bid_near_touch_depth_share_10bps"],bid_depth_1/bid_depth_10,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(o["ask_near_touch_depth_share_10bps"],ask_depth_1/ask_depth_10,rel_tol=1e-12,abs_tol=1e-12)

def test_rich_nonzero_coverage():
    p=NovelMicrostructureCandidatePack(); ev=[('ob',0,1,1,[(99.99,10),(99.98,8),(99.97,5),(99.95,4),(99.90,3),(99.80,2)],[(100.01,9),(100.02,7),(100.03,5),(100.05,4),(100.10,3),(100.20,2)])]
    ev += [('trade',50,2,100.01,2,1,1,0),('ob',80,3,2,[],[(100.01,5)]),('ob',120,4,2,[],[(100.01,12)]),('ob',150,5,2,[(99.99,13)],[]),('trade',200,6,99.99,3,-1,-1,0),('ob',230,7,2,[(99.99,6)],[]),('ob',270,8,2,[(99.99,14)],[]),('ob',310,9,2,[],[(100.01,14)]),('ob',360,10,2,[(100.00,10)],[(100.01,14)]),('ob',420,11,2,[(100.00,10)],[(100.02,10)]),('trade',430,12,100.02,1,1,1,0),('trade',460,13,100.00,5,-1,-1,0),('trade',530,14,100.01,8,1,1,0),('ob',680,15,2,[(100.00,4)],[(100.02,16)])]
    _feed(p,ev); o=p.emit()
    assert set(o)==set(ROUND2_REQUESTED_FEATURES)
    assert np.isfinite(np.asarray(list(o.values()),dtype=float)).all()
    assert sum(abs(float(v))>1e-12 for v in o.values())>=80

def test_round2_ewma_value_decay_add_first_update():
    e=EWMAValue(3000)
    e.update(100,5.0)
    assert math.isclose(e.value(100),5.0,rel_tol=1e-12)
    e.update(3100,1.0)
    assert e.value(3100)>3.4 and e.value(3100)<3.6

def test_round2_trade_impact_invalid_current_book_neutral():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("trade",100,2,100.01,1,1,1,0),("ob",200,3,1,[(100.03,10)],[(100.01,10)])])
    o=p.emit()
    assert o["last_buy_mid_impact_bps_since_trade"]==0.0
    assert o["buy_trade_impact_sum_bps_500ms"]==0.0

def test_round2_source_guards_new_formulas():
    src=''.join(inspect.getsource(NovelMicrostructureCandidatePack.emit).split())
    for bad in ['min(depth_bid_10,depth_ask_10)','trade_impact_buy_500-trade_impact_sell_500','spread_bps*trade_burst_ratio','*(1.0-_safe_div']:
        assert bad not in src

def test_round2_spread_after_trade_features_are_zero_without_recent_trades():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("ob",100,2,1,[(100.0,10)],[(100.03,10)]),("ob",200,3,1,[(100.0,10)],[(100.01,10)])])
    o=p.emit()
    assert o["spread_widen_event_count_1000ms"]>0 or o["spread_tighten_event_count_1000ms"]>0
    assert o["spread_recompression_after_trade_500ms"]==0.0
    assert o["spread_widen_after_trade_500ms"]==0.0
    assert np.isfinite(np.asarray(list(o.values()),dtype=float)).all()
    p2=NovelMicrostructureCandidatePack()
    _feed(p2,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("trade",100,2,100.01,1,1,1,0),("ob",200,3,1,[(100.0,10)],[(100.03,10)]),("ob",300,4,1,[(100.0,10)],[(100.01,10)])])
    o2=p2.emit()
    assert np.isfinite(o2["spread_recompression_after_trade_500ms"]) and abs(o2["spread_recompression_after_trade_500ms"])<100
    assert np.isfinite(o2["spread_widen_after_trade_500ms"]) and abs(o2["spread_widen_after_trade_500ms"])<100

def test_round2_trade_mid_at_trade_falls_back_to_trade_price_before_book():
    p=NovelMicrostructureCandidatePack()
    p.on_event(("trade",100,1,100.25,2.0,1,1,0))
    assert p.last_buy_trade is not None
    assert p.last_buy_trade["mid_at_trade"]==100.25
    p.on_event(("ob",200,2,1,[(100.30,10)],[(100.40,10)]))
    o=p.emit()
    assert np.isfinite(o["last_buy_mid_impact_bps_since_trade"]) and o["last_buy_mid_impact_bps_since_trade"]!=0.0
    assert np.isfinite(o["last_trade_mid_impact_signed_bps"]) and o["last_trade_mid_impact_signed_bps"]!=0.0

def test_round2_trade_mid_at_trade_falls_back_during_crossed_book():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("ob",100,2,1,[(100.03,10)],[(100.01,10)]),("trade",120,3,100.00,1.0,-1,-1,0)])
    assert p.last_sell_trade is not None
    assert p.last_sell_trade["mid_at_trade"]==100.00
    p.on_event(("ob",200,4,1,[(99.98,10)],[(100.00,10)]))
    o=p.emit()
    assert np.isfinite(o["last_sell_mid_impact_bps_since_trade"]) and o["last_sell_mid_impact_bps_since_trade"]!=0.0

def test_round2_incremental_ob_ignores_nonpositive_prices():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.00,10)],[(100.02,10)]),("ob",100,2,2,[(-1.0,5.0)],[(0.0,5.0)])])
    o=p.emit()
    assert 0.0 not in p.asks and -1.0 not in p.bids
    assert p.book_valid is True
    assert o["bid_queue_cliff_ratio_l1_l5"]!=0.0 or o["spread_state_transition_rate_3000ms"]>=0.0
    assert np.isfinite(np.asarray(list(o.values()),dtype=float)).all()

def test_round2_mid_path_reversal_neutral_with_insufficient_history():
    p=NovelMicrostructureCandidatePack(); p.on_event(("ob",0,1,1,[(100.0,10)],[(100.02,10)])); o=p.emit()
    assert o["mid_price_path_efficiency_1000ms"]==0.0 and o["mid_price_path_efficiency_3000ms"]==0.0
    assert o["mid_price_reversal_ratio_1000ms"]==0.0 and o["mid_price_reversal_ratio_3000ms"]==0.0
    p2=NovelMicrostructureCandidatePack()
    _feed(p2,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("ob",100,2,1,[(100.01,10)],[(100.03,10)]),("ob",200,3,1,[(99.99,10)],[(100.01,10)]),("ob",300,4,1,[(100.00,10)],[(100.02,10)])])
    o2=p2.emit()
    assert 0.0<=o2["mid_price_path_efficiency_1000ms"]<=1e9
    assert np.isfinite(o2["mid_price_reversal_ratio_1000ms"])
    assert abs(o2["mid_price_reversal_ratio_1000ms"]-(1.0-o2["mid_price_path_efficiency_1000ms"]))<1e-9

def test_round2_metadata_usage_flags_are_feature_specific():
    m=NovelMicrostructureCandidatePack().metadata()
    assert set(m)==set(ROUND2_REQUESTED_FEATURES)
    assert m["obi_realized_vol_500ms"]["uses_book_state"] is True and m["obi_realized_vol_500ms"]["uses_trade_state"] is False
    assert m["trade_size_hhi_1000ms"]["uses_book_state"] is False and m["trade_size_hhi_1000ms"]["uses_trade_state"] is True
    assert m["ob_interarrival_cv_500ms"]["uses_book_state"] is True and m["ob_interarrival_cv_500ms"]["uses_trade_state"] is False
    assert m["trade_interarrival_cv_500ms"]["uses_book_state"] is False and m["trade_interarrival_cv_500ms"]["uses_trade_state"] is True
    assert m["event_interarrival_cv_1000ms"]["uses_book_state"] is True and m["event_interarrival_cv_1000ms"]["uses_trade_state"] is True
    assert m["last_buy_mid_impact_bps_since_trade"]["uses_book_state"] is True and m["last_buy_mid_impact_bps_since_trade"]["uses_trade_state"] is True
    assert m["bid_depth_centroid_bps_10bps"]["uses_book_state"] is True and m["bid_depth_centroid_bps_10bps"]["uses_trade_state"] is False
    for v in m.values():
        for k in ["candidate_family","candidate_kind","candidate_horizon_ms","uses_book_state","uses_trade_state","expected_target"]:
            assert k in v

def test_round2_batch_writer_two_candidates(tmp_path):
    from feature_lab import evaluate_candidate_batch
    n=200; rng=np.random.default_rng(0)
    X=rng.normal(size=(n,3)).astype(np.float32); y=rng.normal(size=(n,3)).astype(np.float32); week=["w"]*n
    cand_move=np.linspace(-1,1,n); cand_mag=np.abs(np.linspace(-1,1,n)); cand_constant=np.zeros(n)
    candidates={"cand_move":cand_move,"cand_mag":cand_mag,"cand_constant":cand_constant}
    md={k:{"candidate_family":"test","candidate_kind":"event_derived","candidate_horizon_ms":1000,"uses_book_state":True,"uses_trade_state":True,"expected_target":"move"} for k in candidates}
    evaluate_candidate_batch(candidates,X,y,["f1","f2","f3"],week,low_abs_trim_fraction=0.02,high_abs_trim_fraction=0.02,out_dir=tmp_path,candidate_metadata=md)
    out=tmp_path/"feature_lab_batch_relative_report.csv"
    assert out.exists()
    df=pd.read_csv(out)
    assert "cand_constant" in set(df["candidate"])
    row=df[df["candidate"]=="cand_constant"].iloc[0]
    assert row["decision"]=="reject"
    assert ("evaluation_error" in str(row.get("reason",""))) or ("constant" in str(row.get("reason","")).lower()) or ("variance" in str(row.get("reason","")).lower())
    assert "cand_move" in set(df["candidate"]) and "cand_mag" in set(df["candidate"])

def test_round2_emit_output_finite_after_bad_inputs():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("ob",50,2,2,[(-1.0,5.0)],[(0.0,5.0)]),("ob",100,3,1,[(100.03,10)],[(100.01,10)]),("trade",150,4,100.01,1.0,0,0,0),("trade",200,5,100.02,1.0,1,1,0),("ob",250,6,1,[(100.01,10)],[(100.03,10)])])
    o=p.emit()
    assert set(o)==set(ROUND2_REQUESTED_FEATURES)
    assert np.isfinite(np.asarray(list(o.values()),dtype=float)).all()
    assert p.book_valid is True

def test_round2_crossed_book_state_recovers_after_valid_update():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("ob",100,2,1,[(100.03,10)],[(100.01,10)])]); o1=p.emit()
    assert p.book_valid is False and o1["bid_queue_cliff_ratio_l1_l5"]==0.0
    p.on_event(("ob",200,3,1,[(100.0,11)],[(100.02,9)])); o2=p.emit()
    assert p.book_valid is True
    assert np.isfinite(o2["bid_queue_cliff_ratio_l1_l5"])


def test_round2_zero_cross_rate_carries_exact_zeros():
    w=RollingValueWindow()
    w.add(0,1.0); w.add(100,0.0); w.add(200,-1.0)
    assert math.isclose(w.zero_cross_rate(1000,200),0.5,rel_tol=1e-12)
    w=RollingValueWindow()
    w.add(0,0.0); w.add(100,1.0); w.add(200,0.0); w.add(300,-1.0)
    assert math.isclose(w.zero_cross_rate(1000,300),0.5,rel_tol=1e-12)

def test_round2_trade_sign_entropy_one_sided_exact_zero():
    p=NovelMicrostructureCandidatePack()
    _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("trade",100,2,100.02,1.0,1,1,0),("trade",200,3,100.02,2.0,1,1,0)])
    o=p.emit()
    assert o["trade_sign_entropy_1000ms"]==0.0

def test_round2_large_trade_side_dominance_uses_largest_notional_not_share():
    p=NovelMicrostructureCandidatePack(); _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,10)]),("trade",100,2,100.02,1,1,1,0),("trade",110,3,100.02,1,-1,-1,0),("trade",120,4,100.02,1,-1,-1,0),("trade",130,5,100.02,1,-1,-1,0),("trade",140,6,100.02,1,-1,-1,0)])
    o=p.emit()
    assert math.isclose(o["buy_largest_trade_share_3000ms"],1.0,rel_tol=1e-12)
    assert math.isclose(o["sell_largest_trade_share_3000ms"],0.25,rel_tol=1e-12)
    assert math.isclose(o["large_trade_side_dominance_3000ms"],0.0,abs_tol=1e-12)

def test_round2_depth_recovery_ratio_current_over_trade_depth():
    def run(a0,a1,side):
        p=NovelMicrostructureCandidatePack(); _feed(p,[("ob",0,1,1,[(100.0,10)],[(100.02,a0)])])
        _feed(p,[("trade",100,2,100.02 if side>0 else 100.0,1.0,side,side,0),("ob",200,3,1,[(100.0,a1)],[(100.02,a1)])])
        return p.emit()
    assert math.isclose(run(10,10,1)["buy_trade_depth_recovery_ratio_500ms"],1.0,rel_tol=1e-12)
    assert math.isclose(run(10,20,1)["buy_trade_depth_recovery_ratio_500ms"],2.0,rel_tol=1e-12)
    assert math.isclose(run(10,5,1)["buy_trade_depth_recovery_ratio_500ms"],0.5,rel_tol=1e-12)
    assert math.isclose(run(10,10,-1)["sell_trade_depth_recovery_ratio_500ms"],1.0,rel_tol=1e-12)

def test_round2_trade_impact_sum_outputs_clip_to_ratio_clip():
    p = NovelMicrostructureCandidatePack()
    _feed(p, [
        ("ob", 0, 1, 1, [(99.99, 10.0)], [(100.01, 10.0)]),
        ("trade", 100, 2, 100.01, 1.0, 1, 1, 0),
        ("ob", 200, 3, 1, [(199.99, 10.0)], [(200.01, 10.0)]),
    ])
    o = p.emit()
    assert o["buy_trade_impact_sum_bps_500ms"] == RATIO_CLIP
    assert np.isfinite(o["buy_trade_impact_sum_bps_500ms"])

def test_round2_sell_trade_impact_sum_outputs_clip_to_ratio_clip():
    p = NovelMicrostructureCandidatePack()
    _feed(p, [
        ("ob", 0, 1, 1, [(99.99, 10.0)], [(100.01, 10.0)]),
        ("trade", 100, 2, 99.99, 1.0, -1, -1, 0),
        ("ob", 200, 3, 1, [(0.99, 10.0)], [(1.01, 10.0)]),
    ])
    o = p.emit()
    assert o["sell_trade_impact_sum_bps_500ms"] == RATIO_CLIP
    assert np.isfinite(o["sell_trade_impact_sum_bps_500ms"])

def test_round2_trade_impact_sum_outputs_clip_negative_to_ratio_clip():
    p = NovelMicrostructureCandidatePack()
    _feed(p, [
        ("ob", 0, 1, 1, [(99.99, 10.0)], [(100.01, 10.0)]),
        ("trade", 100, 2, 100.01, 1.0, 1, 1, 0),
        ("ob", 200, 3, 1, [(0.99, 10.0)], [(1.01, 10.0)]),
    ])
    o = p.emit()
    assert o["buy_trade_impact_sum_bps_500ms"] == -RATIO_CLIP
    assert np.isfinite(o["buy_trade_impact_sum_bps_500ms"])

def test_round2_sell_trade_impact_sum_outputs_clip_negative_to_ratio_clip():
    p = NovelMicrostructureCandidatePack()
    _feed(p, [
        ("ob", 0, 1, 1, [(99.99, 10.0)], [(100.01, 10.0)]),
        ("trade", 100, 2, 99.99, 1.0, -1, -1, 0),
        ("ob", 200, 3, 1, [(199.99, 10.0)], [(200.01, 10.0)]),
    ])
    o = p.emit()
    assert o["sell_trade_impact_sum_bps_500ms"] == -RATIO_CLIP
    assert np.isfinite(o["sell_trade_impact_sum_bps_500ms"])

def test_round2_trade_impact_sum_outputs_preserve_normal_values():
    p = NovelMicrostructureCandidatePack()
    _feed(p, [
        ("ob", 0, 1, 1, [(99.99, 10.0)], [(100.01, 10.0)]),
        ("trade", 100, 2, 100.01, 1.0, 1, 1, 0),
        ("ob", 200, 3, 1, [(100.19, 10.0)], [(100.21, 10.0)]),
    ])
    o = p.emit()
    assert math.isclose(o["buy_trade_impact_sum_bps_500ms"], 20.0, rel_tol=1e-9, abs_tol=1e-9)
