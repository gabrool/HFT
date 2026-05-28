from pathlib import Path

import pytest

from mmrt.features import specs
from mmrt.storage import writer as wr
from mmrt.linear import head_features as hf
from mmrt.linear import models as lm


def _manifest(tmp_path: Path):
    root = tmp_path / 'ds'
    writer = wr.DecisionRowWriter(wr.WriterConfig(dataset_id='d1', created_at_utc='2026-01-01T00:00:00Z', dataset_root=str(root), chunk_rows=2))
    fv = tuple(float(i) for i in range(specs.FEATURE_COUNT))
    writer.append_values(1,1,1,1,100.0,1,(0.0,0.0,0.0),fv)
    return writer.finalize()


def test_default_all_features_each_head(tmp_path: Path):
    m = _manifest(tmp_path)
    r = hf.resolve_head_feature_sets(m)
    for h in lm.MODEL_HEADS:
        assert r.columns_for_head(h) == tuple(m.feature_columns)


def test_unknown_head_and_feature_and_duplicates(tmp_path: Path):
    m = _manifest(tmp_path)
    c0 = m.feature_columns[0]
    with pytest.raises(ValueError):
        hf.HeadFeatureConfig({'bad': (c0,)})
    with pytest.raises(ValueError):
        hf.resolve_head_feature_sets(m, hf.HeadFeatureConfig({lm.DIRECTION_HEAD: ('missing',)}))
    with pytest.raises(ValueError):
        hf.HeadFeatureConfig({lm.DIRECTION_HEAD: (c0, c0)})


def test_missing_defaults_overlap_and_manifest_order(tmp_path: Path):
    m = _manifest(tmp_path)
    cols = tuple(m.feature_columns)
    cfg = hf.HeadFeatureConfig({lm.DIRECTION_HEAD: (cols[3], cols[1]), lm.MAGNITUDE_UP_HEAD: (cols[1], cols[2])})
    r = hf.resolve_head_feature_sets(m, cfg)
    assert r.columns_for_head(lm.DIRECTION_HEAD) == (cols[1], cols[3])
    assert r.columns_for_head(lm.MAGNITUDE_DOWN_HEAD) == cols
    assert set(r.feature_columns_by_head.keys()) == set(lm.MODEL_HEADS)
    d = r.as_dict()
    assert 'feature_counts_by_head' in d and 'feature_schema_hash' in d
