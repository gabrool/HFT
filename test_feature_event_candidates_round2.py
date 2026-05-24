from pathlib import Path
import numpy as np
from feature_event_candidates_round2 import ROUND2_REQUESTED_FEATURES, NovelMicrostructureCandidatePack

def test_exact_feature_list():
    p=NovelMicrostructureCandidatePack()
    assert len(ROUND2_REQUESTED_FEATURES)==136
    assert len(set(ROUND2_REQUESTED_FEATURES))==136
    assert p.feature_names()==ROUND2_REQUESTED_FEATURES
    assert set(p.metadata())==set(ROUND2_REQUESTED_FEATURES)

def test_emit_keys_and_finite():
    p=NovelMicrostructureCandidatePack()
    events=[('ob',0,1,1,[(100,1),(99,1),(98,1),(97,1),(96,1)],[(101,1),(102,1),(103,1),(104,1),(105,1)]),('trade',10,2,100.5,1,1,1,0),('ob',20,3,0,[(100,2)],[(101,0.5)]),('trade',30,4,100.5,2,-1,-1,0)]
    for e in events: p.on_event(e)
    out=p.emit()
    assert set(out)==set(ROUND2_REQUESTED_FEATURES)
    assert np.isfinite(np.asarray(list(out.values()),dtype=float)).all()

