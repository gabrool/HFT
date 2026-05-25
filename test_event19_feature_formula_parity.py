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
            assert cmssl_out.features.shape == (0,)
    return out


def test_event19_schema_and_feature_count():
    names = FeatureEngine().feature_names()
    assert len(names) == 172
    assert len(set(names)) == len(names)
    assert len(ROUND2_PRODUCTION_EVENT_FEATURES) == 19
    for n in ROUND2_PRODUCTION_EVENT_FEATURES:
        assert names.count(n) == 1
    assert 'v13' in FEATURE_SCHEMA and 'pruned172' in FEATURE_SCHEMA
    assert 'round2xformv3' in FEATURE_TRANSFORM
    assert 'pruned172' in CHECKPOINT_SCHEMA


def test_event19_transform_specs_cover_all_features():
    names = FeatureEngine().feature_names()
    specs = build_feature_transform_specs(names)
    assert len(specs) == len(names)
    smap = {s.name: s for s in specs}
    for n in ROUND2_PRODUCTION_EVENT_FEATURES:
        assert n in smap


def test_event19_rich_formula_parity_and_nonzero():
    events = [
        ('ob',0,1,1,[(100.00,10),(99.99,8),(99.98,5),(99.96,4),(99.94,3)],[(100.02,9),(100.03,7),(100.04,5),(100.06,4),(100.08,3)]),
        ('trade',50,2,100.02,2.0,1,1,0),('ob',100,3,2,[(100.00,6)],[]),('ob',150,4,2,[(100.00,13)],[]),('ob',200,5,2,[],[(100.02,13)]),
        ('trade',250,6,100.00,3.0,-1,-1,0),('ob',300,7,2,[],[(100.02,6)]),('ob',350,8,2,[],[(100.02,14)]),('ob',400,9,2,[(100.00,15)],[]),
        ('trade',500,10,100.02,1.5,1,1,0),('trade',900,11,100.00,4.0,-1,-1,0),
        ('ob',1000,12,2,[(100.01,9)],[(100.03,11)]),('ob',1200,13,2,[(100.02,10)],[(100.04,9)]),('ob',1400,14,2,[(100.01,12)],[(100.03,8)]),
        ('trade',2500,15,100.03,5.0,1,1,0),('ob',2600,16,2,[(100.01,5)],[(100.04,16)]),('ob',2800,17,2,[(100.00,14)],[(100.03,7)]),('ob',3200,18,2,[(100.02,9)],[(100.05,13)]),
    ]
    cmssl, ref = feed_both(events)[-1]
    for n in ROUND2_PRODUCTION_EVENT_FEATURES:
        assert float(cmssl[n]) == pytest.approx(float(ref[n]), rel=1e-5, abs=1e-5), n


def test_offline_ingest_uses_dynamic_round2_dim():
    import offline_ingest
    names = FeatureEngine().feature_names()
    assert offline_ingest.RAW_FEATURE_NAMES == names
    assert offline_ingest.RAW_FEATURE_DIM_CORE == len(names) == 172
    assert offline_ingest.RAW_FEATURE_DIM_TOTAL == len(names) + AUX_DIM
