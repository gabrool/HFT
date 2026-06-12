import math
import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import kernels as k


def test_public_api_boundary():
    assert hasattr(k, "__all__")
    assert all(not name.startswith("_") for name in k.__all__)
    assert "np" not in k.__all__
    assert "njit" not in k.__all__
    assert "_maybe_njit" not in k.__all__
    forbidden = ("feature", "label", "target", "future", "decision", "cmssl", "bybit")
    for name in k.__all__:
        low = name.lower()
        assert not any(tok in low for tok in forbidden)


def test_no_forbidden_imports():
    code = r'''
import sys

before = set(sys.modules)
import mmrt.features.kernels  # noqa: F401
after = set(sys.modules) - before

forbidden = (
    "po" + "lars",
    "pan" + "das",
    "tor" + "ch",
    "pya" + "rrow",
    "CM" + "SSL17",
    "offline_" + "ingest",
)

bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit("forbidden imports loaded by kernels: " + repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_validation_helpers():
    arr = k.require_1d_float_array([1, 2, 3], "x")
    assert arr.dtype == np.float64
    assert arr.ndim == 1
    assert arr.flags.c_contiguous
    with pytest.raises(ValueError):
        k.require_1d_float_array(np.array([[1.0]]), "x")
    with pytest.raises(ValueError):
        k.require_1d_float_array(np.array([1.0, np.nan]), "x")

    a, b = k.require_same_shape_1d([1, 2], [3, 4], "a", "b")
    assert a.shape == b.shape
    with pytest.raises(ValueError):
        k.require_same_shape_1d([1, 2], [3], "a", "b")

    with pytest.raises(TypeError):
        k.require_positive_int(True, "v")
    with pytest.raises(TypeError):
        k.require_nonnegative_int(False, "v")
    with pytest.raises(ValueError):
        k.require_finite_float(float("nan"), "v")
    with pytest.raises(ValueError):
        k.require_finite_float(float("inf"), "v")
    with pytest.raises(TypeError):
        k.require_finite_float(True, "v")


def test_safe_divide_and_signed_log1p_and_clip():
    assert k.safe_divide(4.0, 2.0) == 2.0
    assert k.safe_divide(1.0, 0.0, default=-1.0) == -1.0
    assert k.signed_log1p(9.0) == pytest.approx(math.log1p(9.0))
    assert k.signed_log1p(-9.0) == pytest.approx(-math.log1p(9.0))
    assert k.clip_scalar(5.0, 0.0, 3.0) == 3.0
    assert k.clip_scalar(-1.0, 0.0, 3.0) == 0.0
    assert k.clip_scalar(2.0, 0.0, 3.0) == 2.0
    with pytest.raises(ValueError):
        k.clip_scalar(1.0, 2.0, 1.0)


def test_bps_mid_spread_microprice():
    assert k.mid_price(100.0, 102.0) == 101.0
    assert k.spread_bps(100.0, 102.0) == pytest.approx((2.0 / 101.0) * 10_000.0)
    assert k.spread_bps(102.0, 100.0) < 0.0
    assert k.microprice(100.0, 102.0, 10.0, 30.0) == pytest.approx((102.0 * 10.0 + 100.0 * 30.0) / 40.0)
    assert k.microprice(100.0, 102.0, 0.0, 0.0) == 0.0


def test_imbalance():
    assert k.imbalance(3.0, 1.0) == pytest.approx(0.5)
    assert k.imbalance(1.0, 3.0) == pytest.approx(-0.5)
    assert k.imbalance(0.0, 0.0) == 0.0


def test_depth_sums():
    bid_px = np.array([100.0, 99.0, 98.0])
    bid_sz = np.array([1.0, 2.0, 3.0])
    assert k.sum_first_n(bid_sz, 2) == 3.0
    assert k.notional_sum_first_n(bid_px, bid_sz, 2) == pytest.approx(100.0 * 1.0 + 99.0 * 2.0)
    assert k.sum_first_n(bid_sz, 10) == 6.0


def test_depth_within_bps_and_notional():
    mid = 100.0
    bid_px = np.array([99.99, 99.0])
    bid_sz = np.array([2.0, 3.0])
    ask_px = np.array([100.01, 101.0])
    ask_sz = np.array([4.0, 5.0])
    bps = 2.0

    assert k.depth_within_bps(bid_px, bid_sz, mid, 1, bps) == pytest.approx(2.0)
    assert k.depth_within_bps(ask_px, ask_sz, mid, -1, bps) == pytest.approx(4.0)
    assert k.notional_depth_within_bps(bid_px, bid_sz, mid, 1, bps) == pytest.approx(99.99 * 2.0)
    assert k.notional_depth_within_bps(ask_px, ask_sz, mid, -1, bps) == pytest.approx(100.01 * 4.0)
    assert k.depth_within_bps(bid_px, bid_sz, mid, 0, bps) == 0.0
    assert k.depth_within_bps(bid_px, bid_sz, mid, 1, -1.0) == 0.0


def test_depth_centroid_bps():
    mid = 100.0
    bid_px = np.array([99.99, 99.90])
    bid_sz = np.array([1.0, 3.0])
    expected = (1.0 * 1.0 + 10.0 * 3.0) / 4.0
    assert k.depth_centroid_bps(bid_px, bid_sz, mid, 1, 20.0) == pytest.approx(expected)
    assert k.depth_centroid_bps(bid_px, bid_sz, mid, 1, 0.5) == 0.0


def test_liquidity_void_bps():
    mid = 100.0
    bid_px = np.array([99.99, 99.9])
    bid_sz = np.array([0.1, 10.0])
    assert k.liquidity_void_bps(bid_px, bid_sz, mid, 1, 1.0) == pytest.approx(10.0)
    assert k.liquidity_void_bps(bid_px, bid_sz, mid, 1, 20.0) == 0.0


def test_rolling_prune_left_index():
    ts = np.array([10, 20, 30, 40], dtype=np.int64)
    assert k.rolling_prune_left_index(ts, 0, 4, 40, 15) == 2
    assert k.rolling_prune_left_index(ts, 1, 3, 40, 15) == 2
    with pytest.raises(ValueError):
        k.rolling_prune_left_index(ts, 3, 2, 40, 15)


def test_rolling_range_stats():
    values = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    assert k.rolling_sum_range(values, 0, 3) == 6.0
    assert k.rolling_count_range(1, 4) == 3
    assert k.rolling_mean_range(values, 0, 5) == pytest.approx(2.75)
    assert k.rolling_std_range(values, 0, 3) == pytest.approx(math.sqrt(2.0 / 3.0))
    assert k.rolling_max_abs_range(np.array([-1.0, 2.0, -5.0]), 0, 3) == 5.0


def test_ewma():
    assert k.ewma_update(10.0, 12.0, 0.5) == 11.0
    assert k.ewma_update(float("nan"), 12.0, 0.5) == 12.0
    assert k.ewma_update(10.0, float("nan"), 0.5) == 10.0
    assert k.ewma_alpha_from_dt(0, 10) == 0.0
    assert k.ewma_alpha_from_dt(10, 10) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        k.ewma_update(10.0, 12.0, -0.1)
    with pytest.raises(ValueError):
        k.ewma_update(10.0, 12.0, 1.1)


def test_asof_right_no_future_leakage():
    ts = np.array([100, 200, 300], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0])
    assert k.asof_index_right(ts, 50) == -1
    assert k.asof_index_right(ts, 100) == 0
    assert k.asof_index_right(ts, 250) == 1
    assert k.asof_index_right(ts, 300) == 2
    assert k.asof_value_right(ts, values, 250) == 2.0
    assert k.asof_value_right(ts, values, 50, default=-1.0) == -1.0


def test_clip_array_and_finite_or_zero():
    inp = np.array([-10.0, 0.0, 10.0, np.nan, np.inf])
    out = k.clip_array_inplace(inp, -2.0, 2.0)
    assert np.allclose(out, np.array([-2.0, 0.0, 2.0, 0.0, 0.0]))
    assert np.isnan(inp[3])
    fz = k.finite_or_zero_array(inp)
    assert np.allclose(fz, np.array([-10.0, 0.0, 10.0, 0.0, 0.0]))


def test_numba_flag_is_bool():
    assert isinstance(k.NUMBA_AVAILABLE, bool)


def test_no_future_leakage_concepts_in_source_names():
    forbidden = ("future", "lookahead", "peek", "target", "label")
    for name in k.__all__:
        low = name.lower()
        assert not any(tok in low for tok in forbidden)
