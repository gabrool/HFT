import csv
import numpy as np
from pathlib import Path
from feature_event_candidates import MovementMicrostructureCandidatePack, REQUESTED_FEATURES, RollingInterarrival
from feature_lab import evaluate_candidate_batch, _assert_event_pack_filled

def test_pack_emits_all_requested_features():
    p=MovementMicrostructureCandidatePack(); p.reset()
    events=[
        ("ob",1000,0,1,[(100.000,8.0),(99.999,6.0),(99.998,5.0),(99.997,4.0),(99.940,3.0)],[(100.001,8.0),(100.002,6.0),(100.003,5.0),(100.004,4.0),(100.060,3.0)]),
        ("ob",1080,1,2,[(100.000,5.0)],[(100.001,10.0)]),
        ("trade",1120,2,100.001,1.2,1,0,0),
        ("ob",1160,3,2,[],[(100.001,3.0)]),
        ("ob",1190,4,2,[],[(100.001,9.0)]),
        ("trade",1210,5,100.000,0.8,-1,0,0),
        ("trade",1240,6,100.001,1.6,1,0,0),
        ("trade",1270,7,100.001,1.1,1,0,0),
        ("ob",1310,8,2,[(100.000,9.0)],[(100.001,5.0)]),
        ("ob",1360,9,2,[(100.000,4.0)],[(100.001,11.0)]),
        ("ob",1400,10,2,[(100.000,0.0),(99.999,7.0)],[(100.001,12.0)]),
    ]
    for e in events: p.on_event(e)
    names=p.feature_names(); assert set(names)==set(REQUESTED_FEATURES); assert len(names)==len(set(names))
    promoted={"top5_trade_share_notional_3000ms","depth_imbalance_realized_vol_1000ms","microprice_zero_cross_rate_1000ms","l1_churn_over_depth_1000ms","same_side_trade_cluster_notional_1000ms","ofi_pressure_x_churn_500ms","bid_liquidity_void_bps","ask_liquidity_void_bps","post_buy_trade_ask_replenishment_200ms","post_sell_trade_bid_replenishment_200ms"}
    assert len(REQUESTED_FEATURES)==77
    assert len(names)==77
    assert not (promoted & set(names))

    out=p.emit(); assert set(out)==set(names)
    representatives=["l1_churn_notional_200ms","l1_churn_over_depth_500ms","bid_l1_cancel_to_add_ratio_200ms","same_side_replenishment_after_depletion_200ms","buy_trade_depth_recovery_ratio_500ms","trade_impact_decay_ratio_200_to_1000ms","event_interarrival_cv_1000ms","trade_burstiness_ewma_fast_slow","aggressor_run_length_current","trade_sign_entropy_1000ms","trade_sign_flip_rate_1000ms","trade_size_hhi_1000ms","largest_trade_share_notional_1000ms","bid_depth_convexity_1_5_10bps","depth_slope_bid_1_to_10","bid_queue_cliff_ratio_l1_l2","near_touch_depth_drop_bid","book_stability_score_1000ms","quiet_liquid_state_score","microprice_realized_vol_1000ms","obi_realized_vol_1000ms","spread_realized_vol_1000ms","ofi_l1_over_effective_depth_200ms","abs_ofi_over_depth_1000ms"]
    for k in representatives: assert np.isfinite(out[k]) and out[k]!=0.0, k
    vals=np.array(list(out.values()),dtype=np.float64); assert np.isfinite(vals).all(); assert sum(abs(v)>1e-12 for v in vals) >= int(0.50*len(vals))

def test_batch_writer_two_candidates(tmp_path: Path):
    n=200; X=np.random.randn(n,3).astype(np.float32); y=np.random.randn(n,3).astype(np.float32); week=["w"]*n
    c={"cand_move":np.linspace(-1,1,n,dtype=np.float32),"cand_mag":np.abs(np.linspace(-1,1,n,dtype=np.float32)),"cand_constant":np.zeros(n,dtype=np.float32)}
    evaluate_candidate_batch(c,X,y,["f1","f2","f3"],week,low_abs_trim_fraction=0.02,high_abs_trim_fraction=0.02,out_dir=tmp_path)
    rows=list(csv.DictReader((tmp_path/"feature_lab_batch_relative_report.csv").open())); rr={r["candidate"]:r for r in rows}
    assert rr["cand_constant"]["decision"]=="reject"; assert "evaluation_error" in rr["cand_constant"]["reason"]

def test_assert_event_pack_filled_raises():
    try: _assert_event_pack_filled({1:[0],2:[1]}, {1}, "wk", 1); assert False
    except RuntimeError: assert True

def test_interarrival_windows_are_non_mutating():
    ri = RollingInterarrival()
    for ts in [0, 100, 300, 700, 1100]: ri.on_event(ts)
    a = ri.cv(1000, 1100); _ = ri.cv(200, 1100); _ = ri.cv(500, 1100); b = ri.cv(1000, 1100)
    assert a == b

def test_no_hardcoded_recovery_constants_in_source():
    text = Path("feature_event_candidates.py").read_text()
    assert 'buy_trade_depth_recovery_ratio_200ms"]=0.1' not in text
    assert 'sell_trade_depth_recovery_ratio_200ms"]=0.1' not in text

def test_recovery_ratios_are_not_hardcoded_constants():
    p=MovementMicrostructureCandidatePack()
    seq1=[("ob",1000,0,1,[(100.00,5.0),(99.99,5.0),(99.98,5.0)],[(100.01,5.0),(100.02,5.0),(100.03,5.0)]),("trade",1010,1,100.01,3.0,1,0,0),("ob",1020,2,2,[],[(100.01,1.0)]),("ob",1100,3,2,[],[(100.01,5.0)])]
    for e in seq1: p.on_event(e)
    out1=p.emit(); p.reset()
    seq2=[("ob",1000,0,1,[(100.00,5.0),(99.99,5.0),(99.98,5.0)],[(100.01,5.0),(100.02,5.0),(100.03,5.0)]),("trade",1010,1,100.01,3.0,1,0,0),("ob",1020,2,2,[],[(100.01,1.0)])]
    for e in seq2: p.on_event(e)
    out2=p.emit()
    assert out1["buy_trade_depth_recovery_ratio_500ms"] > out2["buy_trade_depth_recovery_ratio_500ms"]

def test_emit_has_no_default_zero_fallback_source_guard():
    text = Path("feature_event_candidates.py").read_text().replace(" ", "")
    assert "o.update({k:0.0for k in REQUESTED_FEATURES})" not in text
    assert "o.get(k,0.0)" not in text

def test_all_requested_features_are_explicitly_assigned_in_emit_source():
    text = Path("feature_event_candidates.py").read_text()
    emit_text = text.split("def emit", 1)[1]
    missing = []
    for name in REQUESTED_FEATURES:
        if f'o["{name}"]' not in emit_text and f"o['{name}']" not in emit_text:
            missing.append(name)
    assert not missing

def test_metadata_has_specific_families():
    md = MovementMicrostructureCandidatePack().metadata()
    assert md["l1_churn_notional_200ms"]["candidate_family"] == "queue_churn"
    assert md["buy_trade_depth_recovery_ratio_500ms"]["candidate_family"] == "book_resilience"
    assert md["trade_size_hhi_1000ms"]["candidate_family"] == "trade_concentration"
    assert md["bid_queue_cliff_ratio_l1_l2"]["candidate_family"] == "queue_cliff"
    assert md["signed_ofi_over_depth_1000ms"]["candidate_family"] == "ofi_pressure"

def test_queue_cliff_uses_same_side_levels():
    p=MovementMicrostructureCandidatePack(); p.on_event(("ob",1000,0,1,[(100,10),(99,5),(98,4),(97,3),(96,2)],[(101,8),(102,4),(103,3),(104,2),(105,1)]))
    o=p.emit(); assert np.isclose(o["bid_queue_cliff_ratio_l1_l2"],(100*10)/(99*5)); assert np.isclose(o["ask_queue_cliff_ratio_l1_l2"],(101*8)/(102*4))
