"""Numeric kernels for the MMRT feature pipeline.

This module contains small NumPy/Numba-compatible primitives used by future
book, trade, transform, and label builders. It does not parse market rows,
compute named feature vectors, apply feature specs, construct labels, read
files, or import the data layer.
"""

import math
from typing import Callable

import numpy as np

try:
    from numba import njit
except Exception:
    njit = None


FLOAT_EPS = 1e-12
BPS_SCALE = 10_000.0
US_PER_SECOND = 1_000_000.0

NUMBA_AVAILABLE = njit is not None


def _maybe_njit(func: Callable) -> Callable:
    if njit is None:
        return func
    return njit(cache=True, fastmath=False)(func)


def require_1d_float_array(arr: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(out)


def require_same_shape_1d(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> tuple[np.ndarray, np.ndarray]:
    aa = require_1d_float_array(a, a_name)
    bb = require_1d_float_array(b, b_name)
    if aa.shape != bb.shape:
        raise ValueError(f"{a_name} and {b_name} must have same shape")
    return aa, bb


def require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


@_maybe_njit
def _safe_divide_impl(num: float, den: float, default: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= FLOAT_EPS:
        return default
    out = num / den
    return out if math.isfinite(out) else default


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    n = require_finite_float(num, "num")
    d = require_finite_float(den, "den")
    de = require_finite_float(default, "default")
    return float(_safe_divide_impl(n, d, de))


@_maybe_njit
def _signed_log1p_impl(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x >= 0.0:
        return math.log1p(x)
    return -math.log1p(-x)


def signed_log1p(x: float) -> float:
    return float(_signed_log1p_impl(require_finite_float(x, "x")))


@_maybe_njit
def _clip_scalar_impl(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return 0.0
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clip_scalar(x: float, lo: float, hi: float) -> float:
    xv = require_finite_float(x, "x")
    lv = require_finite_float(lo, "lo")
    hv = require_finite_float(hi, "hi")
    if lv > hv:
        raise ValueError("lo must be <= hi")
    return float(_clip_scalar_impl(xv, lv, hv))


@_maybe_njit
def _bps_change_impl(new_value: float, old_value: float) -> float:
    if not math.isfinite(new_value) or not math.isfinite(old_value) or abs(old_value) <= FLOAT_EPS:
        return 0.0
    return (new_value / old_value - 1.0) * BPS_SCALE


def bps_change(new_value: float, old_value: float) -> float:
    return float(_bps_change_impl(require_finite_float(new_value, "new_value"), require_finite_float(old_value, "old_value")))


@_maybe_njit
def _mid_price_impl(best_bid: float, best_ask: float) -> float:
    if best_bid <= 0.0 or best_ask <= 0.0 or not math.isfinite(best_bid) or not math.isfinite(best_ask):
        return 0.0
    return 0.5 * (best_bid + best_ask)


def mid_price(best_bid: float, best_ask: float) -> float:
    return float(_mid_price_impl(require_finite_float(best_bid, "best_bid"), require_finite_float(best_ask, "best_ask")))


@_maybe_njit
def _spread_bps_impl(best_bid: float, best_ask: float) -> float:
    mid = _mid_price_impl(best_bid, best_ask)
    if mid <= 0.0:
        return 0.0
    return (best_ask - best_bid) / mid * BPS_SCALE


def spread_bps(best_bid: float, best_ask: float) -> float:
    return float(_spread_bps_impl(require_finite_float(best_bid, "best_bid"), require_finite_float(best_ask, "best_ask")))


@_maybe_njit
def _microprice_impl(best_bid: float, best_ask: float, bid_size: float, ask_size: float) -> float:
    den = bid_size + ask_size
    if best_bid <= 0.0 or best_ask <= 0.0 or den <= FLOAT_EPS:
        return 0.0
    if not (math.isfinite(best_bid) and math.isfinite(best_ask) and math.isfinite(bid_size) and math.isfinite(ask_size)):
        return 0.0
    return (best_ask * bid_size + best_bid * ask_size) / den


def microprice(best_bid: float, best_ask: float, bid_size: float, ask_size: float) -> float:
    return float(_microprice_impl(
        require_finite_float(best_bid, "best_bid"),
        require_finite_float(best_ask, "best_ask"),
        require_finite_float(bid_size, "bid_size"),
        require_finite_float(ask_size, "ask_size"),
    ))


@_maybe_njit
def _imbalance_impl(bid_value: float, ask_value: float) -> float:
    den = bid_value + ask_value
    if not math.isfinite(bid_value) or not math.isfinite(ask_value) or abs(den) <= FLOAT_EPS:
        return 0.0
    return (bid_value - ask_value) / den


def imbalance(bid_value: float, ask_value: float) -> float:
    return float(_imbalance_impl(require_finite_float(bid_value, "bid_value"), require_finite_float(ask_value, "ask_value")))


@_maybe_njit
def _sum_first_n_impl(values: np.ndarray, n: int) -> float:
    total = 0.0
    limit = n
    if limit > values.shape[0]:
        limit = values.shape[0]
    for i in range(limit):
        v = values[i]
        if math.isfinite(v):
            total += v
    return total


def sum_first_n(values: np.ndarray, n: int) -> float:
    vals = require_1d_float_array(values, "values")
    nval = require_nonnegative_int(n, "n")
    return float(_sum_first_n_impl(vals, nval))


@_maybe_njit
def _notional_sum_first_n_impl(px: np.ndarray, sz: np.ndarray, n: int) -> float:
    total = 0.0
    limit = n
    if limit > px.shape[0]:
        limit = px.shape[0]
    for i in range(limit):
        p = px[i]
        s = sz[i]
        if math.isfinite(p) and math.isfinite(s) and p > 0.0 and s > 0.0:
            total += p * s
    return total


def notional_sum_first_n(px: np.ndarray, sz: np.ndarray, n: int) -> float:
    pxa, sza = require_same_shape_1d(px, sz, "px", "sz")
    nval = require_nonnegative_int(n, "n")
    return float(_notional_sum_first_n_impl(pxa, sza, nval))


@_maybe_njit
def _depth_within_bps_impl(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, bps: float) -> float:
    if mid <= 0.0 or bps < 0.0 or side_code == 0:
        return 0.0
    threshold = mid * (1.0 - bps / BPS_SCALE) if side_code > 0 else mid * (1.0 + bps / BPS_SCALE)
    total = 0.0
    for i in range(px.shape[0]):
        p = px[i]
        s = sz[i]
        if not (math.isfinite(p) and math.isfinite(s) and p > 0.0 and s > 0.0):
            continue
        if side_code > 0:
            if p >= threshold:
                total += s
        else:
            if p <= threshold:
                total += s
    return total


def depth_within_bps(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, bps: float) -> float:
    pxa, sza = require_same_shape_1d(px, sz, "px", "sz")
    m = require_finite_float(mid, "mid")
    b = require_finite_float(bps, "bps")
    sc = int(side_code)
    return float(_depth_within_bps_impl(pxa, sza, m, sc, b))


@_maybe_njit
def _notional_depth_within_bps_impl(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, bps: float) -> float:
    if mid <= 0.0 or bps < 0.0 or side_code == 0:
        return 0.0
    threshold = mid * (1.0 - bps / BPS_SCALE) if side_code > 0 else mid * (1.0 + bps / BPS_SCALE)
    total = 0.0
    for i in range(px.shape[0]):
        p = px[i]
        s = sz[i]
        if not (math.isfinite(p) and math.isfinite(s) and p > 0.0 and s > 0.0):
            continue
        if side_code > 0:
            if p >= threshold:
                total += p * s
        else:
            if p <= threshold:
                total += p * s
    return total


def notional_depth_within_bps(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, bps: float) -> float:
    pxa, sza = require_same_shape_1d(px, sz, "px", "sz")
    m = require_finite_float(mid, "mid")
    b = require_finite_float(bps, "bps")
    sc = int(side_code)
    return float(_notional_depth_within_bps_impl(pxa, sza, m, sc, b))


@_maybe_njit
def _depth_centroid_bps_impl(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, max_bps: float) -> float:
    if mid <= 0.0 or max_bps < 0.0 or side_code == 0:
        return 0.0
    wsum = 0.0
    dsum = 0.0
    for i in range(px.shape[0]):
        p = px[i]
        s = sz[i]
        if not (math.isfinite(p) and math.isfinite(s) and p > 0.0 and s > 0.0):
            continue
        if side_code > 0:
            dist = (mid - p) / mid * BPS_SCALE
        else:
            dist = (p - mid) / mid * BPS_SCALE
        if dist >= 0.0 and dist <= max_bps:
            wsum += s
            dsum += dist * s
    if wsum <= FLOAT_EPS:
        return 0.0
    return dsum / wsum


def depth_centroid_bps(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, max_bps: float) -> float:
    pxa, sza = require_same_shape_1d(px, sz, "px", "sz")
    m = require_finite_float(mid, "mid")
    mb = require_finite_float(max_bps, "max_bps")
    sc = int(side_code)
    return float(_depth_centroid_bps_impl(pxa, sza, m, sc, mb))


@_maybe_njit
def _liquidity_void_bps_impl(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, min_size: float) -> float:
    if mid <= 0.0 or min_size < 0.0 or side_code == 0:
        return 0.0
    for i in range(px.shape[0]):
        p = px[i]
        s = sz[i]
        if not (math.isfinite(p) and math.isfinite(s) and p > 0.0 and s >= min_size):
            continue
        if side_code > 0:
            dist = (mid - p) / mid * BPS_SCALE
        else:
            dist = (p - mid) / mid * BPS_SCALE
        if dist >= 0.0:
            return dist
    return 0.0


def liquidity_void_bps(px: np.ndarray, sz: np.ndarray, mid: float, side_code: int, min_size: float) -> float:
    pxa, sza = require_same_shape_1d(px, sz, "px", "sz")
    m = require_finite_float(mid, "mid")
    ms = require_finite_float(min_size, "min_size")
    sc = int(side_code)
    return float(_liquidity_void_bps_impl(pxa, sza, m, sc, ms))


@_maybe_njit
def _rolling_prune_left_index_impl(ts_us: np.ndarray, start: int, end: int, now_us: int, window_us: int) -> int:
    cutoff = now_us - window_us
    idx = start
    while idx < end and ts_us[idx] < cutoff:
        idx += 1
    return idx


def rolling_prune_left_index(ts_us: np.ndarray, start: int, end: int, now_us: int, window_us: int) -> int:
    ts = np.ascontiguousarray(np.asarray(ts_us, dtype=np.int64))
    if ts.ndim != 1:
        raise ValueError("ts_us must be 1D")
    s = require_nonnegative_int(start, "start")
    e = require_nonnegative_int(end, "end")
    n = require_positive_int(now_us, "now_us")
    w = require_positive_int(window_us, "window_us")
    if not (0 <= s <= e <= ts.shape[0]):
        raise ValueError("bounds must satisfy 0 <= start <= end <= len(ts_us)")
    return int(_rolling_prune_left_index_impl(ts, s, e, n, w))


@_maybe_njit
def _rolling_sum_range_impl(values: np.ndarray, start: int, end: int) -> float:
    total = 0.0
    for i in range(start, end):
        v = values[i]
        if math.isfinite(v):
            total += v
    return total


def rolling_sum_range(values: np.ndarray, start: int, end: int) -> float:
    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if vals.ndim != 1:
        raise ValueError("values must be a 1D array")
    s = require_nonnegative_int(start, "start")
    e = require_nonnegative_int(end, "end")
    if not (0 <= s <= e <= vals.shape[0]):
        raise ValueError("bounds must satisfy 0 <= start <= end <= len(values)")
    return float(_rolling_sum_range_impl(vals, s, e))


@_maybe_njit
def _rolling_count_range_impl(start: int, end: int) -> int:
    return end - start if end >= start else 0


def rolling_count_range(start: int, end: int) -> int:
    s = require_nonnegative_int(start, "start")
    e = require_nonnegative_int(end, "end")
    return int(_rolling_count_range_impl(s, e))


@_maybe_njit
def _rolling_mean_range_impl(values: np.ndarray, start: int, end: int) -> float:
    total = 0.0
    count = 0
    for i in range(start, end):
        v = values[i]
        if math.isfinite(v):
            total += v
            count += 1
    if count == 0:
        return 0.0
    return total / count


def rolling_mean_range(values: np.ndarray, start: int, end: int) -> float:
    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if vals.ndim != 1:
        raise ValueError("values must be a 1D array")
    s = require_nonnegative_int(start, "start")
    e = require_nonnegative_int(end, "end")
    if not (0 <= s <= e <= vals.shape[0]):
        raise ValueError("bounds must satisfy 0 <= start <= end <= len(values)")
    return float(_rolling_mean_range_impl(vals, s, e))


@_maybe_njit
def _rolling_std_range_impl(values: np.ndarray, start: int, end: int) -> float:
    total = 0.0
    count = 0
    for i in range(start, end):
        v = values[i]
        if math.isfinite(v):
            total += v
            count += 1
    if count < 2:
        return 0.0
    mean = total / count
    ss = 0.0
    for i in range(start, end):
        v = values[i]
        if math.isfinite(v):
            d = v - mean
            ss += d * d
    return math.sqrt(ss / count)


def rolling_std_range(values: np.ndarray, start: int, end: int) -> float:
    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if vals.ndim != 1:
        raise ValueError("values must be a 1D array")
    s = require_nonnegative_int(start, "start")
    e = require_nonnegative_int(end, "end")
    if not (0 <= s <= e <= vals.shape[0]):
        raise ValueError("bounds must satisfy 0 <= start <= end <= len(values)")
    return float(_rolling_std_range_impl(vals, s, e))


@_maybe_njit
def _rolling_max_abs_range_impl(values: np.ndarray, start: int, end: int) -> float:
    out = 0.0
    has = False
    for i in range(start, end):
        v = values[i]
        if math.isfinite(v):
            av = abs(v)
            if (not has) or av > out:
                out = av
                has = True
    return out if has else 0.0


def rolling_max_abs_range(values: np.ndarray, start: int, end: int) -> float:
    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if vals.ndim != 1:
        raise ValueError("values must be a 1D array")
    s = require_nonnegative_int(start, "start")
    e = require_nonnegative_int(end, "end")
    if not (0 <= s <= e <= vals.shape[0]):
        raise ValueError("bounds must satisfy 0 <= start <= end <= len(values)")
    return float(_rolling_max_abs_range_impl(vals, s, e))


@_maybe_njit
def _ewma_update_impl(prev: float, value: float, alpha: float) -> float:
    if not math.isfinite(value):
        return prev
    if not math.isfinite(prev):
        return value
    return prev + alpha * (value - prev)


def ewma_update(prev: float, value: float, alpha: float) -> float:
    p = float(prev)
    v = float(value)
    a = require_finite_float(alpha, "alpha")
    if a < 0.0 or a > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return float(_ewma_update_impl(p, v, a))


@_maybe_njit
def _ewma_alpha_from_dt_impl(dt_us: int, half_life_us: int) -> float:
    if dt_us <= 0 or half_life_us <= 0:
        return 0.0
    return 1.0 - math.exp(-math.log(2.0) * float(dt_us) / float(half_life_us))


def ewma_alpha_from_dt(dt_us: int, half_life_us: int) -> float:
    dt = require_nonnegative_int(dt_us, "dt_us")
    hl = require_positive_int(half_life_us, "half_life_us")
    return float(_ewma_alpha_from_dt_impl(dt, hl))


@_maybe_njit
def _asof_index_right_impl(ts_us: np.ndarray, query_ts_us: int) -> int:
    lo = 0
    hi = ts_us.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_us[mid] <= query_ts_us:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def asof_index_right(ts_us: np.ndarray, query_ts_us: int) -> int:
    ts = np.ascontiguousarray(np.asarray(ts_us, dtype=np.int64))
    if ts.ndim != 1:
        raise ValueError("ts_us must be 1D")
    q = require_positive_int(query_ts_us, "query_ts_us")
    return int(_asof_index_right_impl(ts, q))


@_maybe_njit
def _asof_value_right_impl(ts_us: np.ndarray, values: np.ndarray, query_ts_us: int, default: float) -> float:
    idx = _asof_index_right_impl(ts_us, query_ts_us)
    if idx < 0:
        return default
    value = values[idx]
    return value if math.isfinite(value) else default


def asof_value_right(ts_us: np.ndarray, values: np.ndarray, query_ts_us: int, default: float = 0.0) -> float:
    ts = np.ascontiguousarray(np.asarray(ts_us, dtype=np.int64))
    if ts.ndim != 1:
        raise ValueError("ts_us must be 1D")
    vals = require_1d_float_array(values, "values")
    if ts.shape[0] != vals.shape[0]:
        raise ValueError("ts_us and values must have same shape")
    q = require_positive_int(query_ts_us, "query_ts_us")
    d = require_finite_float(default, "default")
    return float(_asof_value_right_impl(ts, vals, q, d))


@_maybe_njit
def _clip_array_inplace_impl(values: np.ndarray, lo: float, hi: float) -> None:
    for i in range(values.shape[0]):
        v = values[i]
        if not math.isfinite(v):
            values[i] = 0.0
        elif v < lo:
            values[i] = lo
        elif v > hi:
            values[i] = hi


def clip_array_inplace(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lv = require_finite_float(lo, "lo")
    hv = require_finite_float(hi, "hi")
    if lv > hv:
        raise ValueError("lo must be <= hi")
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64)).copy()
    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")
    _clip_array_inplace_impl(arr, lv, hv)
    return arr


@_maybe_njit
def _finite_or_zero_array_impl(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=np.float64)
    for i in range(values.shape[0]):
        v = values[i]
        out[i] = v if math.isfinite(v) else 0.0
    return out


def finite_or_zero_array(values: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError("values must be a 1D array")
    return _finite_or_zero_array_impl(arr)


__all__ = [
    "FLOAT_EPS",
    "BPS_SCALE",
    "US_PER_SECOND",
    "NUMBA_AVAILABLE",
    "require_1d_float_array",
    "require_same_shape_1d",
    "require_positive_int",
    "require_nonnegative_int",
    "require_finite_float",
    "safe_divide",
    "signed_log1p",
    "clip_scalar",
    "bps_change",
    "mid_price",
    "spread_bps",
    "microprice",
    "imbalance",
    "sum_first_n",
    "notional_sum_first_n",
    "depth_within_bps",
    "notional_depth_within_bps",
    "depth_centroid_bps",
    "liquidity_void_bps",
    "rolling_prune_left_index",
    "rolling_sum_range",
    "rolling_count_range",
    "rolling_mean_range",
    "rolling_std_range",
    "rolling_max_abs_range",
    "ewma_update",
    "ewma_alpha_from_dt",
    "asof_index_right",
    "asof_value_right",
    "clip_array_inplace",
    "finite_or_zero_array",
]
