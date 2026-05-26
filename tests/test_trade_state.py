import subprocess
import sys

import numpy as np
import pytest

from mmrt.features import trade_state as ts
from mmrt.features.specs import FEATURE_COUNT, FEATURE_SPECS, FeatureSource, feature_spec_by_name


def make_trade(local_ts_us=1_000_000, price=100.0, amount=1.0, side_code=ts.BUY_SIDE_CODE):
    return ts.TradeInput(local_ts_us=local_ts_us, ts_us=local_ts_us, price=price, amount=amount, side_code=side_code)


def fv_value(vec, name):
    return vec[feature_spec_by_name(name).index]


def apply_trade_sequence(st):
    rows = [
        (1_000_000, 100.00, 1.0, +1), (1_080_000, 100.10, 2.0, +1), (1_210_000, 100.05, 1.5, -1),
        (1_420_000, 100.20, 3.0, +1), (1_760_000, 100.20, 0.5, 0), (2_030_000, 99.95, 2.5, -1),
        (2_400_000, 100.30, 4.0, +1), (2_900_000, 100.10, 1.0, -1), (3_250_000, 100.50, 5.0, +1),
        (3_700_000, 100.40, 0.8, -1), (4_100_000, 100.60, 6.0, +1),
    ]
    for row in rows:
        st.apply_trade(make_trade(*row))


def test_public_api_boundary():
    expected = {"BUY_SIDE_CODE","SELL_SIDE_CODE","UNKNOWN_SIDE_CODE","TRADE_WINDOWS_US","DEFAULT_HISTORY_CAPACITY","TRADE_FEATURE_INDICES","TRADE_FEATURE_NAMES","TradeInput","TradeSummary","TradeHistory","TradeState","trade_owned_feature_names","trade_owned_feature_indices"}
    assert set(ts.__all__) == expected
    assert all(not n.startswith("_") for n in ts.__all__)


def test_no_forbidden_imports():
    code = "import mmrt.features.trade_state"
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert cp.returncode == 0


def test_trade_input_validation():
    _ = make_trade()
    with pytest.raises(ValueError):
        make_trade(local_ts_us=0)
    with pytest.raises(ValueError):
        ts.TradeInput(local_ts_us=1, ts_us=0, price=1, amount=1, side_code=1)
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
    ts.TradeInput(local_ts_us=1, ts_us=1, price=1, amount=1, side_code=1, event_seq=-1)
    ts.TradeInput(local_ts_us=1, ts_us=1, price=1, amount=1, side_code=1, event_seq=0)
    with pytest.raises(ValueError):
        ts.TradeInput(local_ts_us=1, ts_us=1, price=1, amount=1, side_code=1, event_seq=-2)


def test_apply_trade_summary():
    st = ts.TradeState()
    sm = st.apply_trade(make_trade())
    assert sm.notional == 100.0
    assert sm.side_code == 1 and sm.tick_sign == 0 and sm.cvd_notional == 100.0
    assert sm.trade_count == 1 and sm.buy_trade_count == 1 and sm.sell_trade_count == 0
    assert sm.consecutive_buy_trade_count == 1


def test_monotonic_local_time():
    st = ts.TradeState(); st.apply_trade(make_trade(local_ts_us=1_000_000)); st.apply_trade(make_trade(local_ts_us=1_000_000))
    with pytest.raises(ValueError): st.apply_trade(make_trade(local_ts_us=999_999))


def test_reset_and_capacity():
    st = ts.TradeState(history_capacity=3)
    for i in range(5): st.apply_trade(make_trade(local_ts_us=1_000_000+i))
    assert st.history.size == 3
    assert np.array_equal(st.history.ordered_ts(), np.array([1_000_002,1_000_003,1_000_004]))
    st.reset()
    with pytest.raises(ValueError): st.fill_trade_features(np.zeros(FEATURE_COUNT))
    with pytest.raises(ValueError): st.current_summary()


def test_trade_owned_indices_match_specs():
    expected = tuple(i for i,s in enumerate(FEATURE_SPECS) if s.source==FeatureSource.TRADE)
    assert ts.trade_owned_feature_indices() == expected


def test_feature_vector_shape_and_non_trade_indices_untouched():
    st = ts.TradeState(); apply_trade_sequence(st)
    out = np.full(FEATURE_COUNT, -123.0)
    st.fill_trade_features(out)
    trade_idx = set(ts.trade_owned_feature_indices())
    for i in range(FEATURE_COUNT):
        if i in trade_idx: assert np.isfinite(out[i])
        else: assert out[i] == -123.0

def test_window_trade_stats_manual_200ms():
    st=ts.TradeState()
    st.apply_trade(make_trade(1_000_000,100,1,+1))
    st.apply_trade(make_trade(1_100_000,101,2,-1))
    st.apply_trade(make_trade(1_300_000,102,1,+1))
    v=st.trade_feature_vector()
    assert fv_value(v,"signed_notional_flow_usd_200000us") == pytest.approx(-100.0)
    assert fv_value(v,"signed_trade_count_imbalance_200000us") == pytest.approx(0.0)
    assert fv_value(v,"trade_toxicity_notional_200000us") == pytest.approx(abs(102-202)/304)
    assert fv_value(v,"trade_count_per_second_200000us") == pytest.approx(10.0)


def test_tick_sign_features():
    st=ts.TradeState()
    for i,p in enumerate([100,101,101,100]): st.apply_trade(make_trade(1_000_000+i,p,1,+1))
    v=st.trade_feature_vector()
    assert fv_value(v,"last_tick_sign") == -1
    assert fv_value(v,"zero_tick_fraction_1000000us") > 0


def test_time_since_features_with_as_of():
    st=ts.TradeState(); st.apply_trade(make_trade(1_000_000,100,1,+1))
    out=np.zeros(FEATURE_COUNT)
    st.fill_trade_features(out,as_of_local_ts_us=1_250_000)
    assert fv_value(out,"time_since_trade_us") == 250_000
    assert fv_value(out,"time_since_last_buy_trade_us") == 250_000
    assert fv_value(out,"time_since_last_sell_trade_us") == 0
    with pytest.raises(ValueError): st.fill_trade_features(out,as_of_local_ts_us=999_999)


def test_consecutive_counts():
    st=ts.TradeState()
    st.apply_trade(make_trade(1,100,1,+1)); assert st.consecutive_buy_trade_count==1
    st.apply_trade(make_trade(2,100,1,+1)); assert st.consecutive_buy_trade_count==2
    st.apply_trade(make_trade(3,100,1,-1)); assert st.consecutive_sell_trade_count==1 and st.consecutive_buy_trade_count==0
    st.apply_trade(make_trade(4,100,1,-1)); assert st.consecutive_sell_trade_count==2
    st.apply_trade(make_trade(5,100,1,0)); assert st.consecutive_sell_trade_count==0 and st.consecutive_buy_trade_count==0


def test_apply_trade_rejects_non_trade_input():
    with pytest.raises(TypeError): ts.TradeState().apply_trade(object())
