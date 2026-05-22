import numpy as np
from pathlib import Path
from feature_event_candidates import MovementMicrostructureCandidatePack
from feature_lab import evaluate_candidate_batch

def test_pack_emits_all_requested_features():
    p=MovementMicrostructureCandidatePack(); p.reset()
    events=[("ob",1,0,1,[(100,1.0),(99,1.0)],[(101,1.2),(102,1.0)]),("trade",2,1,101,0.5,1,0,0),("ob",3,2,2,[(100,0.8)],[(101,1.0)]),("trade",4,3,100,0.3,-1,0,0),("ob",5,4,2,[(100,1.1)],[(101,0.7)])]
    for e in events: p.on_event(e)
    names=p.feature_names(); assert len(names)==len(set(names))
    out=p.emit(); assert set(out)==set(names)
    assert np.isfinite(np.array(list(out.values()),dtype=np.float64)).all()

def test_batch_writer_two_candidates(tmp_path: Path):
    n=200
    X=np.random.randn(n,3).astype(np.float32); y=np.random.randn(n,3).astype(np.float32)
    week=["w"]*n
    c={"cand_move":np.linspace(-1,1,n,dtype=np.float32),"cand_mag":np.abs(np.linspace(-1,1,n,dtype=np.float32))}
    evaluate_candidate_batch(c,X,y,["f1","f2","f3"],week,low_abs_trim_fraction=0.02,high_abs_trim_fraction=0.02,out_dir=tmp_path)
    assert (tmp_path/"feature_lab_batch_relative_report.csv").exists()
    assert (tmp_path/"feature_lab_batch_summary.csv").exists()
    assert (tmp_path/"feature_lab_promote_candidates.json").exists()
