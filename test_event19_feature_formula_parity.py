import numpy as np
from CMSSL17 import FeatureEngine, ROUND2_PRODUCTION_EVENT_FEATURES

def test_event19_presence_and_count():
    names = FeatureEngine().feature_names()
    for n in ROUND2_PRODUCTION_EVENT_FEATURES:
        assert names.count(n) == 1
    assert len(set(names)) == len(names)
    assert len(ROUND2_PRODUCTION_EVENT_FEATURES) == 19

def test_ob_trade_smoke_shape():
    eng = FeatureEngine()
    eng._transform_features = lambda raw, dt_ms: np.asarray(raw, dtype=np.float32)
    evs = [
        ('ob',0,0,1,[(100.0,10.0),(99.99,8.0)],[(100.02,9.0),(100.03,7.0)]),
        ('trade',100,0,100.02,1.0,1,1,0),
        ('ob',200,0,2,[(100.0,12.0)],[(100.02,8.0)]),
    ]
    out0 = eng.on_fast_event(evs[0])
    assert out0.is_decision and out0.features.shape == (len(eng.feature_names()),)
    out1 = eng.on_fast_event(evs[1])
    assert (not out1.is_decision) and out1.features.shape == (0,)
    out2 = eng.on_fast_event(evs[2])
    assert out2.is_decision and np.all(np.isfinite(out2.features))
