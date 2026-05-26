import math
import inspect
import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import transforms as tr
from mmrt.features import specs
from mmrt.features.specs import FEATURE_COUNT, TransformKey, feature_index


def cfg(**kwargs):
    params = dict(fast_half_life_us=100, medium_half_life_us=100, slow_half_life_us=100, min_obs=2, variance_floor=1e-12, z_clip=8.0, raw_clip=1_000_000.0, bounded_abs_clip=10.0, output_dtype="float64")
    params.update(kwargs)
    return tr.TransformConfig(**params)


def raw_zero(): return np.zeros(FEATURE_COUNT, dtype=np.float64)

def set_feature(vec, name, value): vec[feature_index(name)] = value; return vec

def val(vec, name): return vec[feature_index(name)]


def test_public_api_boundary():
    expected = {"DEFAULT_FAST_HALF_LIFE_US", "DEFAULT_MEDIUM_HALF_LIFE_US", "DEFAULT_SLOW_HALF_LIFE_US", "DEFAULT_MIN_OBS", "DEFAULT_VARIANCE_FLOOR", "DEFAULT_Z_CLIP", "DEFAULT_RAW_CLIP", "DEFAULT_BOUNDED_ABS_CLIP", "TransformConfig", "TransformDiagnostics", "TransformStateSnapshot", "CausalFeatureTransformer", "feature_transform_keys", "transform_key_for_feature", "ewma_feature_indices", "no_ewma_feature_indices", "base_transform_values", "transform_feature_matrix_causal_local"}
    assert set(tr.__all__) == expected
    for name in tr.__all__:
        assert not name.startswith("_")


def test_public_api_uses_local_clock_names():
    assert "transform_one_local" in dir(tr.CausalFeatureTransformer)
    assert "transform_many_local" in dir(tr.CausalFeatureTransformer)
    assert "transform_one" not in dir(tr.CausalFeatureTransformer)
    assert "transform_many" not in dir(tr.CausalFeatureTransformer)
    assert "transform_feature_matrix_causal_local" in tr.__all__
    assert "transform_feature_matrix_causal" not in tr.__all__
    t = tr.CausalFeatureTransformer(cfg())
    assert hasattr(t, "last_local_ts_us")
    assert not hasattr(t, "last_" + "ts_us")


def test_transform_snapshot_uses_local_ts_field():
    mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
    var = np.zeros(FEATURE_COUNT, dtype=np.float64)
    count = np.zeros(FEATURE_COUNT, dtype=np.int64)
    snap = tr.TransformStateSnapshot(
        rows_seen=0,
        last_local_ts_us=None,
        mean=mean,
        var=var,
        count=count,
    )
    assert snap.last_local_ts_us is None
    assert not hasattr(snap, "last_" + "ts_us")


def test_no_ambiguous_transform_api_surface():
    src = inspect.getsource(tr)
    old_defs = (
        "def transform_" + "one(",
        "def transform_" + "many(",
        "def transform_feature_matrix_" + "causal(",
    )
    for needle in old_defs:
        assert needle not in src
    assert "last_" + "ts_us" not in src
    assert '"transform_feature_matrix_' + 'causal"' not in src


def test_no_forbidden_imports():
    code = r'''
import sys
before = set(sys.modules)
import mmrt.features.transforms  # noqa: F401
after = set(sys.modules) - before
forbidden = (
    "pan" + "das",
    "po" + "lars",
    "to" + "rch",
    "py" + "arrow",
    "num" + "ba",
    "mmrt.features.engine",
    "mmrt.features." + "la" + "bels",
    "mmrt.data.tardis_csv",
    "mmrt.data.event_merge",
    "mmrt.data.quality",
    "mmrt.storage",
    "mmrt.linear",
    "CM" + "SSL17",
    "offline_" + "ingest",
)
bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit(repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)


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
    x = raw_zero()
    set_feature(x, "spread_bps", np.nan)
    set_feature(x, "log_dt_decision_us", np.inf)
    set_feature(x, "signed_notional_flow_usd_200000us", -np.inf)
    x_before = x.copy()

    b = tr.base_transform_values(x, cfg())

    assert np.isfinite(b).all()
    assert val(b, "spread_bps") == 0.0
    assert val(b, "log_dt_decision_us") == 0.0
    assert val(b, "signed_notional_flow_usd_200000us") == 0.0
    assert np.array_equal(x, x_before, equal_nan=True)


def test_transform_one_local_pre_update_causality():
    t = tr.CausalFeatureTransformer(cfg(min_obs=2, variance_floor=1e-12, output_dtype="float64"))
    i = feature_index("spread_bps")
    o1 = t.transform_one_local(1000, set_feature(raw_zero(), "spread_bps", 10.0))
    o2 = t.transform_one_local(1100, set_feature(raw_zero(), "spread_bps", 12.0))
    o3 = t.transform_one_local(1200, set_feature(raw_zero(), "spread_bps", 12.0))
    assert o1[i] == 0.0
    assert o2[i] == 0.0
    assert o3[i] == pytest.approx(1.0)


def test_transform_updates_after_output_not_before():
    t = tr.CausalFeatureTransformer(cfg())
    i = feature_index("spread_bps")
    t.transform_one_local(1000, set_feature(raw_zero(), "spread_bps", 10.0))
    t.transform_one_local(1100, set_feature(raw_zero(), "spread_bps", 12.0))
    o3 = t.transform_one_local(1200, set_feature(raw_zero(), "spread_bps", 100.0))
    assert o3[i] == 8.0


def test_no_ewma_features_output_immediately():
    o = tr.CausalFeatureTransformer(cfg()).transform_one_local(
        1000,
        set_feature(
            set_feature(
                set_feature(raw_zero(), "last_trade_side_sign", 2.5),
                "log_dt_decision_us",
                999.0,
            ),
            "time_since_trade_us",
            999.0,
        ),
    )
    assert val(o, "last_trade_side_sign") == 1.0
    assert val(o, "log_dt_decision_us") == pytest.approx(math.log1p(999.0))
    assert val(o, "time_since_trade_us") == pytest.approx(math.log1p(999.0))


def test_equal_timestamps_allowed_and_do_not_move_ewma():
    t = tr.CausalFeatureTransformer(cfg(min_obs=2))
    i = feature_index("spread_bps")
    t.transform_one_local(1000, set_feature(raw_zero(), "spread_bps", 10.0))
    t.transform_one_local(1000, set_feature(raw_zero(), "spread_bps", 20.0))
    o3 = t.transform_one_local(1100, set_feature(raw_zero(), "spread_bps", 20.0))
    assert o3[i] == 0.0
    assert t.mean[i] == pytest.approx(15.0)


def test_decreasing_local_timestamp_rejected():
    t = tr.CausalFeatureTransformer(cfg())
    t.transform_one_local(1000, raw_zero())
    with pytest.raises(ValueError):
        t.transform_one_local(999, raw_zero())


def test_transform_many_matches_transform_one_loop():
    mat = np.zeros((5, FEATURE_COUNT), dtype=np.float64)
    set_feature(mat[0], "spread_bps", 10.0); set_feature(mat[1], "spread_bps", 12.0); set_feature(mat[2], "spread_bps", 14.0); set_feature(mat[3], "spread_bps", 16.0); set_feature(mat[4], "spread_bps", 18.0)
    for i, s in enumerate([-1.0, 1.0, -1.0, 1.0, 0.0]):
        set_feature(mat[i], "last_trade_side_sign", s)
    ts = np.array([1000, 1100, 1200, 1300, 1400], dtype=np.int64)
    a = tr.CausalFeatureTransformer(cfg()).transform_many_local(ts, mat)
    t = tr.CausalFeatureTransformer(cfg())
    b = np.vstack([t.transform_one_local(int(ts[i]), mat[i]) for i in range(mat.shape[0])])
    assert np.allclose(a, b)


def test_chunked_snapshot_matches_full_sequence():
    mat = np.zeros((10, FEATURE_COUNT), dtype=np.float64)
    ts = np.arange(1000, 2000, 100, dtype=np.int64)
    mat[:, feature_index("spread_bps")] = np.linspace(1.0, 10.0, 10)
    mat[:, feature_index("signed_notional_flow_usd_200000us")] = np.linspace(-50.0, 50.0, 10)
    mat[:, feature_index("last_trade_side_sign")] = np.array([1 if i % 2 else -1 for i in range(10)], dtype=np.float64)

    full_out, full_snap, _ = tr.transform_feature_matrix_causal_local(ts, mat, cfg())
    p1, snap1, _ = tr.transform_feature_matrix_causal_local(ts[:4], mat[:4], cfg())
    p2, snap2, _ = tr.transform_feature_matrix_causal_local(ts[4:], mat[4:], cfg(), initial_snapshot=snap1)

    assert np.allclose(full_out, np.vstack([p1, p2]))
    assert snap2.rows_seen == full_snap.rows_seen
    assert snap2.last_local_ts_us == full_snap.last_local_ts_us
    assert np.allclose(snap2.mean, full_snap.mean)
    assert np.allclose(snap2.var, full_snap.var)
    assert np.array_equal(snap2.count, full_snap.count)

    t = tr.CausalFeatureTransformer(cfg(), snapshot=snap1)
    old = t.mean[0]
    snap1.mean[0] += 123.0
    assert t.mean[0] == old


def test_snapshot_load_and_reset():
    t1 = tr.CausalFeatureTransformer(cfg())
    for j, v in enumerate([10.0, 12.0, 14.0]):
        t1.transform_one_local(1000 + j * 100, set_feature(raw_zero(), "spread_bps", v))
    snap = t1.snapshot()
    t2 = tr.CausalFeatureTransformer(cfg(), snapshot=snap)
    r4 = set_feature(raw_zero(), "spread_bps", 16.0)
    o1 = t1.transform_one_local(1300, r4)
    o2 = t2.transform_one_local(1300, r4)
    assert np.allclose(o1, o2)
    t2.reset()
    assert t2.rows_seen == 0
    assert t2.last_local_ts_us is None
    assert np.allclose(t2.mean, 0.0)
    assert np.allclose(t2.var, 0.0)
    assert np.array_equal(t2.count, np.zeros(FEATURE_COUNT, dtype=np.int64))
    assert t2.diagnostics.rows_seen == 0


def test_snapshot_rejects_nonfinite_mean():
    mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
    var = np.zeros(FEATURE_COUNT, dtype=np.float64)
    count = np.zeros(FEATURE_COUNT, dtype=np.int64)
    mean[0] = np.nan
    with pytest.raises(ValueError):
        tr.TransformStateSnapshot(0, None, mean, var, count)

    mean[0] = np.inf
    with pytest.raises(ValueError):
        tr.TransformStateSnapshot(0, None, mean, var, count)


def test_snapshot_rejects_invalid_var_and_count():
    mean = np.zeros(FEATURE_COUNT, dtype=np.float64)
    var = np.zeros(FEATURE_COUNT, dtype=np.float64)
    count = np.zeros(FEATURE_COUNT, dtype=np.int64)

    bad_var = var.copy()
    bad_var[0] = np.nan
    with pytest.raises(ValueError):
        tr.TransformStateSnapshot(0, None, mean, bad_var, count)

    bad_var = var.copy()
    bad_var[0] = -1.0
    with pytest.raises(ValueError):
        tr.TransformStateSnapshot(0, None, mean, bad_var, count)

    bad_count = count.copy()
    bad_count[0] = -1
    with pytest.raises(ValueError):
        tr.TransformStateSnapshot(0, None, mean, var, bad_count)


def test_transformer_constructor_does_not_accept_initial_snapshot_alias():
    with pytest.raises(TypeError):
        tr.CausalFeatureTransformer(cfg(), initial_snapshot=None)


def test_diagnostics_counts():
    c = cfg(min_obs=2, raw_clip=10.0, bounded_abs_clip=2.0, z_clip=1.0)
    t = tr.CausalFeatureTransformer(c)
    rows = []
    r1 = raw_zero()
    set_feature(r1, "spread_bps", 0.0)
    set_feature(r1, "signed_notional_flow_usd_200000us", 0.0)
    set_feature(r1, "trade_side_quote_response_asymmetry_500000us", 0.0)
    set_feature(r1, "last_trade_side_sign", 0.0)
    rows.append((1000, r1))
    rows.append((1100, set_feature(raw_zero(), "spread_bps", 2.0)))
    r3 = raw_zero()
    set_feature(r3, "spread_bps", 100.0)
    set_feature(r3, "micro_ret_bps_200000us", 100.0)
    set_feature(r3, "trade_side_quote_response_asymmetry_500000us", 3.0)
    set_feature(r3, "last_trade_side_sign", 3.0)
    set_feature(r3, "trade_count_per_second_500000us", np.inf)
    rows.append((1200, r3))
    for ts, row in rows:
        t.transform_one_local(ts, row)
    d = t.diagnostics_snapshot()
    assert d.rows_seen == len(rows)
    assert d.nonfinite_raw_count > 0
    assert d.raw_clip_count > 0
    assert d.bounded_clip_count > 0
    assert d.z_clip_count > 0
    assert d.warmup_ewma_count > 0
    assert d.as_dict() == {
        "rows_seen": d.rows_seen,
        "nonfinite_raw_count": d.nonfinite_raw_count,
        "raw_clip_count": d.raw_clip_count,
        "bounded_clip_count": d.bounded_clip_count,
        "z_clip_count": d.z_clip_count,
        "warmup_ewma_count": d.warmup_ewma_count,
    }


def test_output_dtype_and_shape():
    t = tr.CausalFeatureTransformer()
    o = t.transform_one_local(1, raw_zero())
    assert o.dtype == np.float32
    assert o.shape == (FEATURE_COUNT,)
    assert np.isfinite(o).all()
    many = t.transform_many_local(np.array([2, 3], dtype=np.int64), np.vstack([raw_zero(), raw_zero()]))
    assert many.shape == (2, FEATURE_COUNT)
    assert np.isfinite(many).all()
    o64 = tr.CausalFeatureTransformer(cfg(output_dtype="float64")).transform_one_local(1, raw_zero())
    assert o64.dtype == np.float64


def test_transform_feature_matrix_validates_inputs():
    mat = np.zeros((2, FEATURE_COUNT), dtype=np.float64)
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([1, 2]), np.zeros(FEATURE_COUNT), cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([1, 2]), np.zeros((2, FEATURE_COUNT - 1)), cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([1]), mat, cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([-1, 2]), mat, cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([1.5, 2.0]), mat, cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([True, False]), mat, cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([2, 1]), mat, cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array(["1", "2"], dtype=object), mat, cfg())
    with pytest.raises(ValueError): tr.transform_feature_matrix_causal_local(np.array([1 + 0j, 2 + 0j]), mat, cfg())
    out, _, _ = tr.transform_feature_matrix_causal_local(np.array([1.0, 2.0]), mat, cfg())
    assert out.shape == (2, FEATURE_COUNT)


def test_input_arrays_not_mutated():
    x = raw_zero()
    set_feature(x, "spread_bps", np.nan)
    set_feature(x, "signed_notional_flow_usd_200000us", 123.0)
    x_before = x.copy()
    tr.CausalFeatureTransformer(cfg()).transform_one_local(1000, x)
    assert np.array_equal(x, x_before, equal_nan=True)

    mat = np.vstack([raw_zero(), raw_zero()])
    set_feature(mat[0], "spread_bps", np.nan)
    set_feature(mat[1], "spread_bps", 5.0)
    ts = np.array([1000.0, 1100.0], dtype=np.float64)
    mat_before = mat.copy()
    ts_before = ts.copy()
    tr.transform_feature_matrix_causal_local(ts, mat, cfg())
    assert np.array_equal(mat, mat_before, equal_nan=True)
    assert np.array_equal(ts, ts_before)


def test_no_labels_targets_or_storage_residue():
    public = " ".join(tr.__all__).lower()
    for bad in ["label", "target", "future", "storage", "fit", "pca", "aux", "bybit", "cmssl"]:
        assert bad not in public


def test_all_feature_transform_keys_supported():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FEATURE_COUNT)
    b = tr.base_transform_values(x, cfg())
    assert np.isfinite(b).all()
    o = tr.CausalFeatureTransformer(cfg()).transform_one_local(1, x)
    assert np.isfinite(o).all()
    supported = {
      TransformKey.IDENTITY_EWMA_FAST,
      TransformKey.IDENTITY_EWMA_MEDIUM,
      TransformKey.IDENTITY_EWMA_SLOW,
      TransformKey.IDENTITY_NO_EWMA,
      TransformKey.LOG1P_POS_NO_EWMA,
      TransformKey.LOG1P_POS_EWMA,
      TransformKey.SIGNED_LOG1P_EWMA,
      TransformKey.RATIO_BOUNDED,
      TransformKey.SIGN_NO_EWMA,
      TransformKey.TIME_LOG1P_NO_EWMA,
    }
    for spec in specs.FEATURE_SPECS:
        assert spec.transform_key in supported


def test_no_global_fit_api():
    for n in ["fit", "fit_" + "transform", "Standard" + "Scaler", "GlobalStandardizer", "P" + "CA"]:
        assert not hasattr(tr, n)
