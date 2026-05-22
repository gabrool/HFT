import csv
import numpy as np
from pathlib import Path
from feature_event_candidates import MovementMicrostructureCandidatePack, REQUESTED_FEATURES
from feature_lab import evaluate_candidate_batch, _assert_event_pack_filled

def test_pack_emits_all_requested_features():
    p=MovementMicrostructureCandidatePack(); p.reset()
    events=[
        ("ob",1000,0,1,[(100-i,1.0+0.1*i) for i in range(10)],[(101+i,1.2+0.1*i) for i in range(10)]),
        ("ob",1100,1,2,[(100,0.7),(99,1.4)],[(101,1.5)]),
        ("trade",1200,2,101,0.5,1,0,0),("ob",1300,3,2,[(100,0.9)],[(101,0.8)]),
        ("trade",1450,4,100,0.4,-1,0,0),("trade",1600,5,101,0.6,1,0,0),("trade",1750,6,101,0.7,1,0,0),
        ("ob",1900,7,2,[(100,1.4)],[(101,0.6)]),("ob",2200,8,2,[(100,0.6)],[(101,1.4)]),
    ]
    for e in events: p.on_event(e)
    names=p.feature_names(); assert set(names)==set(REQUESTED_FEATURES); assert len(names)==len(set(names))
    out=p.emit(); assert set(out)==set(names)
    vals=np.array(list(out.values()),dtype=np.float64); assert np.isfinite(vals).all()
    for k in ["l1_churn_notional_200ms","l1_churn_over_depth_500ms","event_interarrival_cv_1000ms","trade_burstiness_ewma_fast_slow","aggressor_run_length_current","trade_size_hhi_1000ms","bid_depth_convexity_1_5_10bps","bid_queue_cliff_ratio_l1_l2","book_stability_score_1000ms","microprice_realized_vol_1000ms","obi_realized_vol_1000ms","ofi_pressure_x_churn_500ms"]:
        assert np.isfinite(out[k]); assert out[k]!=0.0
    assert sum(abs(v)>1e-12 for v in vals) >= int(0.5*len(vals))

def test_batch_writer_two_candidates(tmp_path: Path):
    n=200; X=np.random.randn(n,3).astype(np.float32); y=np.random.randn(n,3).astype(np.float32); week=["w"]*n
    c={"cand_move":np.linspace(-1,1,n,dtype=np.float32),"cand_mag":np.abs(np.linspace(-1,1,n,dtype=np.float32)),"cand_constant":np.zeros(n,dtype=np.float32)}
    evaluate_candidate_batch(c,X,y,["f1","f2","f3"],week,low_abs_trim_fraction=0.02,high_abs_trim_fraction=0.02,out_dir=tmp_path)
    rows=list(csv.DictReader((tmp_path/"feature_lab_batch_relative_report.csv").open()))
    rr={r["candidate"]:r for r in rows}
    assert rr["cand_constant"]["decision"]=="reject"
    assert "evaluation_error" in rr["cand_constant"]["reason"]

def test_assert_event_pack_filled_raises():
    try:
        _assert_event_pack_filled({1:[0],2:[1]}, {1}, "wk", 1)
        assert False
    except RuntimeError:
        assert True
