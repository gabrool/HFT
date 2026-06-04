import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import trade_state as ts
from mmrt.features.specs import (
    FEATURE_COUNT,
    FEATURE_SPECS,
    FeatureSource,
    feature_spec_by_name,
)


def make_trade(local_ts_us=1_000_000, price=100.0, amount=1.0, side_code=ts.BUY_SIDE_CODE):
    return ts.TradeInput(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us,
        price=price,
        amount=amount,
        side_code=side_code,
    )


def fv_value(vec, name):
    return vec[feature_spec_by_name(name).index]


def apply_trade_sequence(st):
    rows = [
        (1_000_000, 100.00, 1.0, +1),
        (1_080_000, 100.10, 2.0, +1),
        (1_210_000, 100.05, 1.5, -1),
        (1_420_000, 100.20, 3.0, +1),
        (1_760_000, 100.20, 0.5, 0),
        (2_030_000, 99.95, 2.5, -1),
        (2_400_000, 100.30, 4.0, +1),
        (2_900_000, 100.10, 1.0, -1),
        (3_250_000, 100.50, 5.0, +1),
        (3_700_000, 100.40, 0.8, -1),
        (4_100_000, 100.60, 6.0, +1),
    ]
    for row in rows:
        st.apply_trade(make_trade(*row))


def test_public_api_boundary():
    expected = {
        "BUY_SIDE_CODE",
        "SELL_SIDE_CODE",
        "UNKNOWN_SIDE_CODE",
        "TRADE_WINDOWS_US",
        "DEFAULT_HISTORY_CAPACITY",
        "TRADE_FEATURE_INDICES",
        "TRADE_FEATURE_NAMES",
        "TradeInput",
        "TradeSummary",
        "TradeHistory",
        "TradeState",
        "trade_owned_feature_names",
        "trade_owned_feature_indices",
        "ACTIVE_TRADE_FEATURES",
    }
    assert set(ts.__all__) == expected
    assert all(not name.startswith("_") for name in ts.__all__)
    forbidden = ("target", "future", "lookahead", "peek", "bybit", "cmssl", "aux")
    for name in ts.__all__:
        lowered = name.lower()
        assert not any(tok in lowered for tok in forbidden)


def test_trade_history_fields_are_active_minimal():
    assert ts.TradeHistory.FIELDS == (
        "ts_us",
        "notional",
        "signed_notional",
        "side_code",
        "tick_sign",
        "buy_notional",
        "sell_notional",
    )


def test_no_inactive_trade_feature_computation_remains():
    import inspect

    src = inspect.getsource(ts)
    forbidden = [
        "_window_trade_stats",
        "_cvd_change",
        "_cvd_ema",
        "_p90_over_median",
        "top5",
        "p90",
        "premium",
        "toxicity",
        "cvd_notional",
        "consecutive_buy_trade_count",
        "consecutive_sell_trade_count",
        "signed_trade_count_imbalance",
        "tick_sign_imbalance",
        "top5_trade_share",
    ]
    for token in forbidden:
        assert token not in src


def test_no_forbidden_imports():
    code = r'''
import sys
before = set(sys.modules)
import mmrt.features.trade_state  # noqa: F401
after = set(sys.modules) - before
forbidden = (
    "po" + "lars",
    "pan" + "das",
    "to" + "rch",
    "py" + "arrow",
    "mmrt.data.tardis_csv",
    "mmrt.data.event_merge",
    "mmrt.data.quality",
    "CM" + "SSL17",
    "offline_" + "ingest",
)
bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit(repr(bad))
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_trade_input_validation_and_canonicalization():
    _ = make_trade()
    with pytest.raises(ValueError):
        make_trade(local_ts_us=0)
    with pytest.raises(ValueError):
        ts.TradeInput(local_ts_us=1, ts_us=0, price=100.0, amount=1.0, side_code=1)
    with pytest.raises(ValueError):
        make_trade(price=0)
    with pytest.raises(ValueError):
        make_trade(amount=0)
    with pytest.raises(ValueError):
        make_trade(price=float("nan"))
    with pytest.raises(ValueError):
        make_trade(amount=float("nan"))
    with pytest.raises(ValueError):
        make_trade(side_code=2)
    with pytest.raises(TypeError):
        make_trade(side_code=True)
    ts.TradeInput(local_ts_us=1, ts_us=1, price=1.0, amount=1.0, side_code=1, event_seq=-1)
    ts.TradeInput(local_ts_us=1, ts_us=1, price=1.0, amount=1.0, side_code=1, event_seq=0)
    with pytest.raises(ValueError):
        ts.TradeInput(local_ts_us=1, ts_us=1, price=1.0, amount=1.0, side_code=1, event_seq=-2)

    t = ts.TradeInput(local_ts_us=1, ts_us=1, price=np.float64(100.0), amount=np.float64(2.0), side_code=1)
    assert isinstance(t.price, float)
    assert isinstance(t.amount, float)
    assert isinstance(t.local_ts_us, int)
    assert isinstance(t.ts_us, int)
    assert isinstance(t.side_code, int)


def test_apply_trade_summary():
    st = ts.TradeState()
    sm = st.apply_trade(make_trade(price=100.0, amount=1.0, side_code=+1))
    assert sm.notional == 100.0
    assert sm.side_code == +1
    assert sm.tick_sign == 0
    assert sm.trade_count == 1
    assert sm.buy_trade_count == 1
    assert sm.sell_trade_count == 0
    assert st.has_trades() is True


def test_monotonic_local_time():
    st = ts.TradeState()
    st.apply_trade(make_trade(local_ts_us=1_000_000))
    st.apply_trade(make_trade(local_ts_us=1_000_000))
    with pytest.raises(ValueError):
        st.apply_trade(make_trade(local_ts_us=999_999))


def test_reset_and_capacity():
    st = ts.TradeState(history_capacity=3)
    for i in range(5):
        st.apply_trade(make_trade(local_ts_us=1_000_000 + i))
    assert st.history.size == 3
    assert np.array_equal(st.history.ordered_ts(), np.array([1_000_002, 1_000_003, 1_000_004]))
    st.reset()
    assert st.has_trades() is False
    with pytest.raises(ValueError):
        st.fill_trade_features(np.zeros(FEATURE_COUNT))
    with pytest.raises(ValueError):
        st.current_summary()


def test_trade_owned_indices_match_specs():
    expected = tuple(i for i, spec in enumerate(FEATURE_SPECS) if spec.source == FeatureSource.TRADE)
    assert ts.trade_owned_feature_indices() == expected
    assert ts.trade_owned_feature_names() == tuple(FEATURE_SPECS[i].name for i in expected)
    assert "vwap_vs_mid_bps_200000us" not in ts.trade_owned_feature_names()
    assert "vwap_vs_mid_bps_500000us" not in ts.trade_owned_feature_names()
    assert "absorption_bid_200000us" not in ts.trade_owned_feature_names()
    assert "ofi_l1_pressure_over_depth_5bps_200000us" not in ts.trade_owned_feature_names()
    assert "spread_bps" not in ts.trade_owned_feature_names()
    assert "micro_ret_bps_200000us" not in ts.trade_owned_feature_names()
    assert "bid_depth_notional_5bps" not in ts.trade_owned_feature_names()


def test_feature_vector_shape_and_non_trade_indices_untouched():
    st = ts.TradeState()
    out = np.full(FEATURE_COUNT, -123.0)
    apply_trade_sequence(st)
    st.fill_trade_features(out)
    trade_idx = set(ts.trade_owned_feature_indices())
    for i in range(FEATURE_COUNT):
        if i in trade_idx:
            assert np.isfinite(out[i])
        else:
            assert out[i] == -123.0


def test_all_trade_features_assigned_and_dynamic_nonzero():
    st = ts.TradeState()
    apply_trade_sequence(st)
    v = st.trade_feature_vector()
    for i in ts.trade_owned_feature_indices():
        assert np.isfinite(v[i])
    assert fv_value(v, "trade_count_per_second_200000us") > 0
    assert fv_value(v, "trade_imbalance_notional_500000us") != 0
    assert fv_value(v, "trade_count_per_second_500000us") > 0
    assert fv_value(v, "max_signed_trade_notional_usd_1000000us") != 0
    assert fv_value(v, "same_side_trade_cluster_notional_1000000us") > 0
    assert fv_value(v, "trade_sign_entropy_3000000us") > 0


def test_window_trade_stats_manual_200ms():
    st = ts.TradeState()
    st.apply_trade(make_trade(1_000_000, 100, 1, +1))
    st.apply_trade(make_trade(1_100_000, 101, 2, -1))
    st.apply_trade(make_trade(1_300_000, 102, 1, +1))
    v = st.trade_feature_vector()
    assert fv_value(v, "trade_count_per_second_200000us") == pytest.approx(10.0)


def test_tick_sign_features():
    st = ts.TradeState()
    for i, p in enumerate([100, 101, 101, 100]):
        st.apply_trade(make_trade(1_000_000 + i, p, 1, +1))
    v = st.trade_feature_vector()
    assert fv_value(v, "zero_tick_fraction_1000000us") > 0


def test_time_since_features_with_as_of():
    st = ts.TradeState()
    st.apply_trade(make_trade(1_000_000, 100, 1, +1))
    out = np.zeros(FEATURE_COUNT)
    st.fill_trade_features(out, as_of_local_ts_us=1_250_000)
    assert fv_value(out, "time_since_last_buy_trade_us") == 250_000
    assert fv_value(out, "time_since_last_sell_trade_us") == 0
    with pytest.raises(ValueError):
        st.fill_trade_features(out, as_of_local_ts_us=999_999)


def test_trade_history_asof_nonpositive_query_returns_default():
    h = ts.TradeHistory()
    assert h.asof_value("notional", 0, default=-7.0) == -7.0
    assert h.asof_value("notional", -1, default=-7.0) == -7.0

    h.append(
        ts_us=1_000_000,
        notional=100.0,
        signed_notional=100.0,
        side_code=ts.BUY_SIDE_CODE,
        tick_sign=0,
        buy_notional=100.0,
        sell_notional=0.0,
    )

    assert h.asof_value("notional", 0, default=-7.0) == -7.0
    assert h.asof_value("notional", -1, default=-7.0) == -7.0
    assert h.asof_value("notional", 999_999, default=-7.0) == -7.0
    assert h.asof_value("notional", 1_000_000, default=-7.0) == 100.0

def test_fill_trade_features_handles_early_asof_windows_without_crashing():
    st = ts.TradeState()
    st.apply_trade(make_trade(local_ts_us=1_000_000, price=100.0, amount=2.0, side_code=ts.BUY_SIDE_CODE))

    out = np.full(FEATURE_COUNT, np.nan, dtype=np.float64)
    st.fill_trade_features(out, as_of_local_ts_us=1_000_000)

    trade_idx = ts.trade_owned_feature_indices()
    assert np.all(np.isfinite(out[list(trade_idx)]))
    assert fv_value(out, "max_signed_trade_notional_usd_1000000us") == pytest.approx(100.0 * 2.0)


def test_max_signed_trade_notional_preserves_largest_abs_sign():
    st = ts.TradeState()
    vals = [(10, +1), (50, -1), (30, +1), (20, -1), (40, +1), (60, -1)]
    for i, (n, s) in enumerate(vals):
        st.apply_trade(make_trade(1_000_000 + i * 100_000, 10.0, n / 10.0, s))
    v = st.trade_feature_vector()
    assert fv_value(v, "max_signed_trade_notional_usd_1000000us") == pytest.approx(-60)


def test_trade_sign_entropy_is_bounded_and_nonzero():
    st = ts.TradeState()
    rows = [(10, +1), (20, +1), (40, +1), (15, -1), (30, -1), (60, -1), (5, 0)]
    for i, (n, s) in enumerate(rows):
        st.apply_trade(make_trade(1_000_000 + i * 300_000, 10.0, n / 10.0, s))
    v = st.trade_feature_vector()
    ent = fv_value(v, "trade_sign_entropy_3000000us")
    assert 0 <= ent <= 1
    assert ent > 0


def test_same_side_cluster_and_silence_gap():
    st = ts.TradeState()
    rows = [
        (2_100_000, 10, +1),
        (2_300_000, 20, +1),
        (2_500_000, 5, -1),
        (2_600_000, 7, -1),
        (2_700_000, 8, -1),
        (2_850_000, 1, 0),
        (2_950_000, 4, +1),
    ]
    for t, n, s in rows:
        st.apply_trade(make_trade(t, 10.0, n / 10.0, s))
    v = st.trade_feature_vector()
    assert fv_value(v, "same_side_trade_cluster_notional_1000000us") == pytest.approx(30.0)
    window_ts = np.array([t for t, _, _ in rows])
    assert fv_value(v, "max_trade_silence_gap_3000000us") == pytest.approx(float(np.max(np.diff(window_ts))))


def test_no_cross_book_or_event_context_features_written():
    st = ts.TradeState()
    out = np.full(FEATURE_COUNT, -123.0)
    apply_trade_sequence(st)
    st.fill_trade_features(out)
    for spec in FEATURE_SPECS:
        if spec.source != FeatureSource.TRADE:
            assert out[spec.index] == -123.0


def test_apply_trade_rejects_non_trade_input():
    with pytest.raises(TypeError):
        ts.TradeState().apply_trade(object())


def test_no_future_leakage_public_names():
    forbidden = ("future", "lookahead", "peek", "target")
    public_callables = [
        name for name in dir(ts)
        if not name.startswith("_") and callable(getattr(ts, name))
    ]
    for name in tuple(ts.__all__) + tuple(public_callables):
        lowered = name.lower()
        assert not any(tok in lowered for tok in forbidden)


def test_as_of_does_not_create_future_lookahead():
    st = ts.TradeState()
    st.apply_trade(make_trade(1_000_000, 100, 1, +1))
    st.apply_trade(make_trade(1_500_000, 100, 1, -1))
    st.apply_trade(make_trade(2_000_000, 100, 1, +1))
    now_v = st.trade_feature_vector()
    asof_v = st.trade_feature_vector(as_of_local_ts_us=2_500_000)
    assert fv_value(now_v, "time_since_last_buy_trade_us") == 0.0
    assert fv_value(asof_v, "time_since_last_buy_trade_us") == 500_000
    # Windows are inclusive [now - window_us, now], so the trade at 2_000_000 is included.
    assert fv_value(asof_v, "trade_count_per_second_500000us") == pytest.approx(2.0)
    with pytest.raises(ValueError):
        st.trade_feature_vector(as_of_local_ts_us=1_900_000)


def test_no_invalid_placeholders():
    st = ts.TradeState()
    apply_trade_sequence(st)
    v = st.trade_feature_vector()
    for i in ts.trade_owned_feature_indices():
        assert np.isfinite(v[i])
    assert fv_value(v, "trade_imbalance_notional_500000us") != 0
    assert fv_value(v, "trade_count_per_second_500000us") > 0
    assert fv_value(v, "max_signed_trade_notional_usd_1000000us") != 0
    assert fv_value(v, "same_side_trade_cluster_notional_1000000us") > 0
    assert fv_value(v, "trade_sign_entropy_3000000us") > 0
