import math
import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import transforms as tr
from mmrt.features import specs
from mmrt.features.specs import FEATURE_COUNT, TransformKey, feature_index, feature_spec_by_name


def cfg(**kwargs):
    params = dict(fast_half_life_us=100, medium_half_life_us=100, slow_half_life_us=100, min_obs=2, variance_floor=1e-12, z_clip=8.0, raw_clip=1_000_000.0, bounded_abs_clip=10.0, output_dtype="float64")
    params.update(kwargs)
    return tr.TransformConfig(**params)


def raw_zero(): return np.zeros(FEATURE_COUNT, dtype=np.float64)

def set_feature(vec, name, value): vec[feature_index(name)] = value; return vec

def val(vec, name): return vec[feature_index(name)]


def test_public_api_boundary():
    expected = {"DEFAULT_FAST_HALF_LIFE_US","DEFAULT_MEDIUM_HALF_LIFE_US","DEFAULT_SLOW_HALF_LIFE_US","DEFAULT_MIN_OBS","DEFAULT_VARIANCE_FLOOR","DEFAULT_Z_CLIP","DEFAULT_RAW_CLIP","DEFAULT_BOUNDED_ABS_CLIP","TransformConfig","TransformDiagnostics","TransformStateSnapshot","CausalFeatureTransformer","feature_transform_keys","transform_key_for_feature","ewma_feature_indices","no_ewma_feature_indices","base_transform_values","transform_feature_matrix_causal"}
    assert set(tr.__all__) == expected
    for name in tr.__all__:
        assert not name.startswith("_")
        low = name.lower()
        for s in ["bybit","cmssl","aux","pca","label","target","future","storage","fit"]:
            assert s not in low


def test_no_forbidden_imports():
    code = "import sys; before=set(sys.modules.keys()); import mmrt.features.transforms as t; after=set(sys.modules.keys()); print('\\n'.join(sorted(after-before)))"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    delta = proc.stdout
    forbidden = ["pan" + "das","po" + "lars","to" + "rch","py" + "arrow","num" + "ba","mmrt.features.engine","mmrt.features." + "la" + "bels","mmrt.data.tardis_csv","mmrt.data.event_merge","mmrt.data.quality","mmrt.storage","mmrt.linear","CM" + "SSL17","offline_" + "ingest"]
    for f in forbidden:
        assert f not in delta


def test_config_validation_and_as_dict():
    c = tr.TransformConfig()
    assert c.dtype == np.dtype("float32")
    with pytest.raises(ValueError): tr.TransformConfig(fast_half_life_us=0)
    with pytest.raises(ValueError): tr.TransformConfig(min_obs=0)
    with pytest.raises(ValueError): tr.TransformConfig(variance_floor=0)
    with pytest.raises(ValueError): tr.TransformConfig(z_clip=0)
    with pytest.raises(ValueError): tr.TransformConfig(raw_clip=0)
    with pytest.raises(ValueError): tr.TransformConfig(bounded_abs_clip=0)
    with pytest.raises(ValueError): tr.TransformConfig(output_dtype="float16")
    with pytest.raises(ValueError): tr.TransformConfig(fast_half_life_us=True)
    d = c.as_dict()
    assert d["feature_names_hash"] and d["feature_specs_hash"]


def test_transform_key_helpers_match_specs():
    assert tr.feature_transform_keys() == tuple(spec.transform_key for spec in specs.FEATURE_SPECS)
    assert tr.transform_key_for_feature("spread_bps") == TransformKey.IDENTITY_EWMA_FAST
    assert tr.transform_key_for_feature("time_since_trade_us") == TransformKey.TIME_LOG1P_NO_EWMA
    assert tr.transform_key_for_feature("last_trade_side_sign") == TransformKey.SIGN_NO_EWMA
    ew = set(tr.ewma_feature_indices()); ne = set(tr.no_ewma_feature_indices())
    assert ew | ne == set(range(FEATURE_COUNT))
    assert not (ew & ne)


def test_base_transform_formulas_representative_features():
    x = raw_zero()
    set_feature(x, "spread_bps", 12.5)
    set_feature(x, "log_dt_decision_us", 999.0)
    set_feature(x, "time_since_trade_us", 999.0)
    set_feature(x, "signed_notional_flow_usd_200000us", -99.0)
    set_feature(x, "trade_count_per_second_500000us", 9.0)
    set_feature(x, "last_trade_side_sign", 2.5)
    set_feature(x, "trade_side_quote_response_asymmetry_500000us", 12.0)
    b = tr.base_transform_values(x, cfg())
    assert val(b, "spread_bps") == 12.5
    assert val(b, "log_dt_decision_us") == pytest.approx(math.log1p(999.0))
    assert val(b, "time_since_trade_us") == pytest.approx(math.log1p(999.0))
    assert val(b, "signed_notional_flow_usd_200000us") == pytest.approx(-math.log1p(99.0))
    assert val(b, "trade_count_per_second_500000us") == pytest.approx(math.log1p(9.0))
    assert val(b, "last_trade_side_sign") == 1.0
    assert val(b, "trade_side_quote_response_asymmetry_500000us") == 10.0


def test_base_transform_nonfinite_inputs_become_zero():
    x = raw_zero(); x0 = x.copy()
    for n, v in [("spread_bps", np.nan), ("log_dt_decision_us", np.inf), ("signed_notional_flow_usd_200000us", -np.inf)]: set_feature(x, n, v)
    b = tr.base_transform_values(x, cfg())
    assert np.isfinite(b).all()
    assert val(b, "spread_bps") == 0 and val(b, "log_dt_decision_us") == 0 and val(b, "signed_notional_flow_usd_200000us") == 0
    assert np.array_equal(x, x0)

# remaining tests condensed for coverage

def test_causality_and_clipping():
    t = tr.CausalFeatureTransformer(cfg())
    i = feature_index("spread_bps")
    r = raw_zero(); r[i]=10; o1=t.transform_one(1000,r)
    r = raw_zero(); r[i]=12; o2=t.transform_one(1100,r)
    r = raw_zero(); r[i]=12; o3=t.transform_one(1200,r)
    assert o1[i] == 0 and o2[i] == 0 and o3[i] == pytest.approx(1.0)
    t2 = tr.CausalFeatureTransformer(cfg())
    t2.transform_one(1000,set_feature(raw_zero(),"spread_bps",10)); t2.transform_one(1100,set_feature(raw_zero(),"spread_bps",12)); o=t2.transform_one(1200,set_feature(raw_zero(),"spread_bps",100))
    assert o[i] == 8.0


def test_many_snapshot_diagnostics_and_validation():
    c = cfg()
    mat = np.zeros((10, FEATURE_COUNT), dtype=float)
    ts = np.arange(1000,1010)
    mat[:, feature_index("spread_bps")] = np.linspace(1,10,10)
    full, snap, d = tr.transform_feature_matrix_causal(ts, mat, c)
    p1, s1, _ = tr.transform_feature_matrix_causal(ts[:4], mat[:4], c)
    p2, s2, _ = tr.transform_feature_matrix_causal(ts[4:], mat[4:], c, initial_snapshot=s1)
    assert np.allclose(full, np.vstack([p1,p2]))
    assert isinstance(d.as_dict(), dict)
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal(np.array([1,0]), mat[:2], c)
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal(np.array([1.5]), mat[:1], c)
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal(np.array([True]), mat[:1], c)


def test_output_dtype_and_no_global_fit_api():
    t = tr.CausalFeatureTransformer()
    o = t.transform_one(1, raw_zero())
    assert o.dtype == np.float32 and o.shape == (FEATURE_COUNT,) and np.isfinite(o).all()
    for n in ["fit", "fit_" + "transform", "Standard" + "Scaler", "GlobalStandardizer"]:
        assert not hasattr(tr, n)


def test_all_feature_transform_keys_supported():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FEATURE_COUNT)
    b = tr.base_transform_values(x, cfg())
    assert np.isfinite(b).all()
    o = tr.CausalFeatureTransformer(cfg()).transform_one(1, x)
    assert np.isfinite(o).all()
    used = {s.transform_key for s in specs.FEATURE_SPECS}
    assert used.issubset(set(TransformKey))
