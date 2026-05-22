import csv
import numpy as np
from pathlib import Path
from feature_event_candidates import MovementMicrostructureCandidatePack, REQUESTED_FEATURES, RollingInterarrival
from feature_lab import evaluate_candidate_batch, _assert_event_pack_filled

def test_pack_emits_all_requested_features():
    p=MovementMicrostructureCandidatePack(); p.reset()
    events=[("ob",1000,0,1,[(100-i,1.0+0.1*i) for i in range(10)],[(101+i,1.2+0.1*i) for i in range(10)]),("ob",1100,1,2,[(100,0.7),(99,1.4)],[(101,1.5)]),("trade",1200,2,101,0.5,1,0,0),("ob",1300,3,2,[(100,0.9)],[(101,0.8)]),("trade",1450,4,100,0.4,-1,0,0),("trade",1600,5,101,0.6,1,0,0),("trade",1750,6,101,0.7,1,0,0),("ob",1900,7,2,[(100,1.4)],[(101,0.6)]),("ob",2200,8,2,[(100,0.6)],[(101,1.4)])]
    for e in events: p.on_event(e)
    names=p.feature_names(); assert set(names)==set(REQUESTED_FEATURES); assert len(names)==len(set(names))
    out=p.emit(); assert set(out)==set(names)
    vals=np.array(list(out.values()),dtype=np.float64); assert np.isfinite(vals).all(); assert sum(abs(v)>1e-12 for v in vals) >= int(0.2*len(vals))

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
    seq1=[("ob",1000,0,1,[(100,5.0),(99,5.0)],[(101,5.0),(102,5.0)]),("trade",1010,1,101,3.0,1,0,0),("ob",1020,2,2,[],[(101,1.0)]),("ob",1100,3,2,[],[(101,5.0)])]
    for e in seq1: p.on_event(e)
    out1=p.emit(); p.reset()
    seq2=[("ob",1000,0,1,[(100,5.0),(99,5.0)],[(101,5.0),(102,5.0)]),("trade",1010,1,101,3.0,1,0,0),("ob",1020,2,2,[],[(101,1.0)])]
    for e in seq2: p.on_event(e)
    out2=p.emit()
    assert out1["buy_trade_depth_recovery_ratio_500ms"] != out2["buy_trade_depth_recovery_ratio_500ms"]

def test_queue_cliff_uses_same_side_levels():
    p=MovementMicrostructureCandidatePack(); p.on_event(("ob",1000,0,1,[(100,10),(99,5),(98,4),(97,3),(96,2)],[(101,8),(102,4),(103,3),(104,2),(105,1)]))
    o=p.emit(); assert np.isclose(o["bid_queue_cliff_ratio_l1_l2"],(100*10)/(99*5)); assert np.isclose(o["ask_queue_cliff_ratio_l1_l2"],(101*8)/(102*4))
