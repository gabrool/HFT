import math
import inspect
import subprocess
import sys

import numpy as np
import pytest

from mmrt.contracts import AsOfPolicy, LabelResult, LabelSpec, PriceReference
from mmrt.features import labels as lb


def spec(horizons=(200_000, 500_000, 1_000_000), entry_delay=1_000):
    return LabelSpec(
        horizons_us=tuple(horizons),
        entry_delay_us=entry_delay,
        price_reference=PriceReference.MID,
        asof_policy=AsOfPolicy.LAST_OBSERVATION,
    )


def test_public_api_boundary():
    expected = {
        "DEFAULT_PRICE_HISTORY_CAPACITY", "PriceObservation", "PendingLabel", "PriceHistory",
        "LabelBuilder", "build_labels_from_price_event_arrays", "label_value_names", "label_ready_local_ts_us", "label_entry_local_ts_us",
    }
    assert set(lb.__all__) == expected
    for name in lb.__all__:
        assert not name.startswith("_")
        low = name.lower()
        for bad in ("bybit", "cmssl", "aux", "transform", "storage", "feature_dim", "grace", "ms"):
            assert bad not in low


def test_public_api_uses_local_clock_names():
    assert "observe_price_local" in dir(lb.LabelBuilder)
    assert "on_decision_local" in dir(lb.LabelBuilder)
    assert "label_now_local" in dir(lb.LabelBuilder)
    assert "on_price_and_decision_local" in dir(lb.LabelBuilder)
    assert "observe_price" not in dir(lb.LabelBuilder)
    assert "on_decision" not in dir(lb.LabelBuilder)
    assert "label_now" not in dir(lb.LabelBuilder)
    assert "on_price_and_decision" not in dir(lb.LabelBuilder)
    assert "build_labels_from_price_event_arrays" in lb.__all__
    assert "label_ready_local_ts_us" in lb.__all__
    assert "label_entry_local_ts_us" in lb.__all__
    assert "build_labels_from_price_arrays" not in lb.__all__
    assert "label_ready_ts_us" not in lb.__all__
    assert "label_entry_ts_us" not in lb.__all__


def test_price_observation_uses_local_ts_field():
    obs = lb.PriceObservation(local_ts_us=1_000_000, event_seq=0, price=100.0)
    assert obs.local_ts_us == 1_000_000
    assert not hasattr(obs, "ts_us")


def test_pending_label_uses_local_ts_fields():
    pending = lb.PendingLabel(
        decision_key=lb.EventKey(1_000_000, 0),
        entry_local_ts_us=1_001_000,
        ready_local_ts_us=1_201_000,
        horizons_us=(200_000,),
    )
    assert pending.decision_key.local_ts_us == 1_000_000
    assert pending.entry_local_ts_us == 1_001_000
    assert pending.ready_local_ts_us == 1_201_000
    assert not hasattr(pending, "decision_ts_us")
    assert not hasattr(pending, "entry_ts_us")
    assert not hasattr(pending, "ready_ts_us")


def test_no_forbidden_imports():
    script = """
import sys
mods_before = set(sys.modules)
import mmrt.features.labels  # noqa: F401
mods_after = set(sys.modules)
new = mods_after - mods_before
forbidden = [
    'pan'+'das', 'po'+'lars', 'to'+'rch', 'py'+'arrow',
    'mmrt.data.tardis_csv', 'mmrt.data.event_merge', 'mmrt.data.quality',
    'mmrt.features.'+'engine', 'mmrt.features.'+'trans'+'forms', 'CM'+'SSL17', 'offline_'+'ingest'
]
for f in forbidden:
    assert f not in new, f
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_rejects_unsupported_spec_modes():
    lb.LabelBuilder(spec())
    for p in PriceReference:
        if p != PriceReference.MID:
            with pytest.raises(ValueError):
                lb.LabelBuilder(LabelSpec(horizons_us=(1,), entry_delay_us=0, price_reference=p, asof_policy=AsOfPolicy.LAST_OBSERVATION))
    with pytest.raises(ValueError):
        lb.LabelBuilder(LabelSpec(horizons_us=(1,), entry_delay_us=0, price_reference=PriceReference.MID, asof_policy="not_supported"))


def test_price_observation_validation():
    obs = lb.PriceObservation(local_ts_us=1, event_seq=0, price=np.float64(10.0))
    assert isinstance(obs.price, float)
    for bad in (-1.0, 0.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            lb.PriceObservation(local_ts_us=1, event_seq=0, price=bad)
    with pytest.raises(ValueError):
        lb.PriceObservation(local_ts_us=-1, event_seq=0, price=1.0)
    with pytest.raises(ValueError):
        lb.PriceObservation(local_ts_us=True, event_seq=0, price=1.0)


def test_price_history_append_asof_and_duplicate_event_key_rejected():
    ph = lb.PriceHistory()
    ph.append(lb.PriceObservation(1000, 0, 100.0))
    ph.append(lb.PriceObservation(2000, 0, 101.0))
    assert ph.asof_price(lb.EventKey(999, 0)) is None
    assert ph.asof_price(lb.EventKey(1000, 0)) == 100.0
    assert ph.asof_price(lb.EventKey(1500, 0)) == 100.0
    assert ph.asof_price(lb.EventKey(2000, 0)) == 101.0
    size = ph.size
    with pytest.raises(ValueError):
        ph.append(lb.PriceObservation(2000, 0, 102.0))
    assert ph.size == size
    with pytest.raises(ValueError):
        ph.append(lb.PriceObservation(1999, 0, 99.0))
    ts, seq, px = ph.active_arrays()
    assert np.array_equal(ts, np.array([1000, 2000], dtype=np.int64))
    assert np.array_equal(seq, np.array([0, 0], dtype=np.int64))
    assert np.array_equal(px, np.array([100.0, 101.0], dtype=np.float64))


def test_price_history_capacity_and_compaction():
    ph = lb.PriceHistory(capacity=3)
    for i in range(1, 6):
        ph.append(lb.PriceObservation(i, 0, float(i)))
    ts, seq, px = ph.active_arrays()
    assert np.array_equal(ts, np.array([3, 4, 5]))
    assert np.array_equal(px, np.array([3.0, 4.0, 5.0]))
    assert ph.asof_price(lb.EventKey(2, 0)) is None
    ph.reset()
    assert ph.size == 0


def test_pending_label_validation():
    p = lb.PendingLabel(lb.EventKey(1, 0), 2, 3, (5, 2, 2))
    assert p.horizons_us == (2, 5)
    with pytest.raises(ValueError):
        lb.PendingLabel(lb.EventKey(2, 0), 1, 3, (1,))
    with pytest.raises(ValueError):
        lb.PendingLabel(lb.EventKey(1, 0), 2, 1, (1,))
    with pytest.raises(ValueError):
        lb.PendingLabel(lb.EventKey(1, 0), 2, 3, ())
    with pytest.raises(ValueError):
        lb.PendingLabel("bad", 0, 1, (1,))


def test_label_entry_ready_helpers_and_names():
    s = spec()
    assert lb.label_entry_local_ts_us(1_000_000, s) == 1_001_000
    assert lb.label_ready_local_ts_us(1_000_000, s) == 2_001_000
    assert lb.label_value_names(s) == ("ret_bps_200000us", "ret_bps_500000us", "ret_bps_1000000us")


def test_streaming_label_maturation_with_entry_delay():
    s = spec(horizons=(200_000, 500_000), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    assert b.observe_price_local(1_000_000, 0, 100.0) == []
    assert b.observe_price_local(1_001_000, 0, 101.0) == []
    assert b.observe_price_local(1_201_000, 0, 102.0) == []
    out = b.observe_price_local(1_501_000, 0, 104.0)
    out = b.observe_price_local(1_501_001, 0, 104.0)
    assert len(out) == 1
    r = out[0]
    assert r.decision_ts_us == 1_000_000
    assert r.entry_ts_us == 1_001_000
    assert r.values_bps == pytest.approx((10_000.0 * math.log(102 / 101), 10_000.0 * math.log(104 / 101)))
    assert b.pending_count == 0


def test_last_observation_policy_inclusive_boundaries():
    s = spec(horizons=(200_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_999, 0, 100.0)
    b.observe_price_local(1_001_000, 0, 101.0)
    b.observe_price_local(1_201_000, 0, 103.0)
    out = b.observe_price_local(1_201_001, 0, 103.0)
    assert len(out) == 1
    assert out[0].values_bps[0] == pytest.approx(10_000.0 * math.log(103 / 101))


def test_last_observation_policy_uses_previous_when_no_exact_timestamp():
    s = spec(horizons=(200_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_900, 0, 100.0)
    b.observe_price_local(1_001_500, 0, 101.0)
    b.observe_price_local(1_200_000, 0, 102.0)
    out = b.observe_price_local(1_250_000, 0, 103.0)
    assert len(out) == 1
    assert out[0].values_bps[0] == pytest.approx(10_000.0 * math.log(102 / 100))


def test_decision_without_entry_price_stays_pending_until_entry_asof_exists():
    s = spec(horizons=(200_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    assert b.observe_price_local(1_500_000, 0, 100.0) == []
    assert b.observe_price_local(1_700_000, 0, 101.0) == []
    assert b.pending_count == 1


def test_multiple_pending_decisions_mature_fifo():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.on_decision_local(1_100_000, 0)
    out = []
    out.extend(b.observe_price_local(1_000_000, 0, 100.0))
    out.extend(b.observe_price_local(1_100_000, 0, 101.0))
    out.extend(b.observe_price_local(1_200_000, 0, 102.0))
    out.extend(b.observe_price_local(1_200_001, 0, 102.0))
    assert [x.decision_ts_us for x in out] == [1_000_000, 1_100_000]
    assert b.pending_count == 0


def test_same_timestamp_distinct_decision_event_sequences_allowed():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)

    b.on_decision_local(1_000_000, 0)
    b.on_decision_local(1_000_000, 1)

    b.observe_price_local(1_000_000, 0, 100.0)
    b.observe_price_local(1_000_000, 1, 101.0)
    b.observe_price_local(1_100_000, 2, 105.0)
    out = b.observe_price_local(1_100_001, 3, 105.0)

    assert len(out) == 2
    assert [(r.decision_ts_us, r.decision_event_seq) for r in out] == [
        (1_000_000, 0),
        (1_000_000, 1),
    ]


def test_duplicate_decision_event_key_rejected():
    b = lb.LabelBuilder(spec(horizons=(100_000,), entry_delay=0))

    b.on_decision_local(1_000_000, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        b.on_decision_local(1_000_000, 0)


def test_on_decision_local_rejects_decreasing_decision_local_timestamps():
    b = lb.LabelBuilder(spec())
    b.on_decision_local(1_000_000, 0)
    with pytest.raises(ValueError):
        b.on_decision_local(999_999, 0)


def test_on_decision_rejects_decreasing_after_pending_matures():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_000, 0, 100.0)
    b.observe_price_local(1_100_000, 0, 101.0)
    out = b.observe_price_local(1_100_001, 0, 101.0)
    assert len(out) == 1
    assert b.pending_count == 0
    with pytest.raises(ValueError):
        b.on_decision_local(999_999, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        b.on_decision_local(1_000_000, 0)
    b.on_decision_local(1_000_001, 0)


def test_label_now_does_not_mutate_pending():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.observe_price_local(1_000_000, 0, 100.0)
    b.observe_price_local(1_100_000, 0, 105.0)
    b.observe_price_local(1_100_001, 0, 105.0)
    before = b.pending_count
    r = b.label_now_local(1_000_000, 0)
    assert isinstance(r, LabelResult)
    assert b.pending_count == before


def test_on_price_and_decision_helper():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    assert b.on_price_and_decision_local(1_000_000, 0, 100.0, is_decision=True) == []
    assert b.on_price_and_decision_local(1_100_000, 0, 102.0, is_decision=False) == []
    out = b.on_price_and_decision_local(1_100_001, 0, 102.0, is_decision=False)
    assert len(out) == 1
    assert out[0].decision_ts_us == 1_000_000


def test_label_result_fields_remain_contract_generic_but_values_are_local():
    s = spec(horizons=(100_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    decision_local_ts_us = 1_000_000
    b.on_decision_local(decision_local_ts_us, 0)
    b.observe_price_local(1_000_000, 0, 100.0)
    b.observe_price_local(1_001_000, 0, 101.0)
    b.observe_price_local(1_101_000, 0, 102.0)
    out = b.observe_price_local(1_101_001, 0, 102.0)
    assert len(out) == 1
    result = out[0]
    assert result.decision_ts_us == decision_local_ts_us
    assert result.entry_ts_us == decision_local_ts_us + s.entry_delay_us


def test_batch_labels_match_streaming():
    s = spec(horizons=(100_000, 200_000), entry_delay=0)
    pts = np.array([1_000_000, 1_100_000, 1_200_000, 1_300_000])
    pvals = np.array([100.0, 101.0, 102.0, 103.0])
    dec = np.array([1_000_000, 1_100_000])
    labels, mask = lb.build_labels_from_price_event_arrays(dec, np.arange(len(dec)), pts, np.arange(len(pts)), pvals, s)
    assert mask.all()

    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.on_decision_local(1_100_000, 0)
    out = []
    for t, p in zip(pts, pvals):
        out.extend(b.observe_price_local(int(t), 0, float(p)))
    out.extend(b.finalize_at_eof())
    stream = np.array([x.values_bps for x in out], dtype=np.float64)
    assert np.allclose(labels, stream)


def test_batch_invalid_rows_for_insufficient_future_context():
    s = spec(horizons=(200_000,), entry_delay=0)
    pts = np.array([1_000_000, 1_100_000])
    pvals = np.array([100.0, 101.0])
    dec = np.array([1_000_000, 1_100_000])
    labels, mask = lb.build_labels_from_price_event_arrays(dec, np.arange(len(dec)), pts, np.arange(len(pts)), pvals, s)
    assert mask.tolist() == [False, False]
    assert np.isnan(labels).all()


def test_batch_uses_event_sequence_for_equal_price_timestamps():
    s = spec(horizons=(100_000,), entry_delay=0)
    pts = np.array([1_000_000, 1_000_000, 1_100_000])
    pvals = np.array([100.0, 101.0, 102.0])
    dec = np.array([1_000_000])
    labels, mask = lb.build_labels_from_price_event_arrays(dec, np.arange(len(dec)), pts, np.arange(len(pts)), pvals, s)
    assert mask[0]
    assert labels[0, 0] == pytest.approx(10_000.0 * math.log(102 / 100))


def test_batch_validates_sorted_inputs():
    s = spec(horizons=(1,), entry_delay=0)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([2, 1]), np.array([0, 1]), np.array([1, 2]), np.array([0, 1]), np.array([1.0, 2.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1, 2]), np.array([0, 1]), np.array([1, 1]), np.array([2, 1]), np.array([1.0, 2.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1, 2]), np.array([0, 1]), np.array([1, 2]), np.array([0, 1]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1]), np.array([0]), np.array([1]), np.array([0]), np.array([0.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([[1]]), np.array([0]), np.array([1]), np.array([0]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([-1]), np.array([0]), np.array([1]), np.array([0]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1]), np.array([-1]), np.array([1]), np.array([0]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1.5]), np.array([0]), np.array([1, 2]), np.array([0, 1]), np.array([1.0, 2.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1]), np.array([0]), np.array([1]), np.array([1.5]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([math.nan]), np.array([0]), np.array([1]), np.array([0]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1]), np.array([0]), np.array([math.inf]), np.array([0]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([True]), np.array([0]), np.array([1]), np.array([0]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_event_arrays(np.array([1]), np.array([True]), np.array([1]), np.array([0]), np.array([1.0]), s)


def test_batch_accepts_integer_valued_float_timestamps():
    s = spec(horizons=(100_000,), entry_delay=0)
    labels, mask = lb.build_labels_from_price_event_arrays(
        np.array([1_000_000.0]),
        np.array([0.0]),
        np.array([1_000_000.0, 1_100_000.0]),
        np.array([0.0, 1.0]),
        np.array([100.0, 101.0]),
        s,
    )
    assert mask.tolist() == [True]
    assert labels[0, 0] == pytest.approx(10_000.0 * math.log(101.0 / 100.0))

def test_no_feature_engine_transform_storage_imports_or_public_residue():
    for name in lb.__all__:
        low = name.lower()
        for bad in ("engine", "transform", "storage"):
            assert bad not in low


def test_no_future_leakage_in_streaming_features_sense():
    s = spec(horizons=(200_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_000, 0, 100.0)
    assert b.observe_price_local(1_199_999, 0, 101.0) == []
    assert b.pending_count == 1
    assert b.observe_price_local(1_200_000, 0, 102.0) == []
    assert len(b.observe_price_local(1_200_001, 0, 102.0)) == 1


def test_label_result_contract():
    s = spec(horizons=(100_000, 200_000), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_000, 0, 100.0)
    b.observe_price_local(1_200_000, 0, 102.0)
    out = b.observe_price_local(1_200_001, 0, 102.0)
    assert isinstance(out[0], LabelResult)
    assert out[0].horizons_us == tuple(sorted(out[0].horizons_us))
    assert len(out[0].values_bps) == len(out[0].horizons_us)


def test_no_old_bybit_ms_grace_residue():
    for name in lb.__all__:
        low = name.lower()
        for bad in ("ms", "grace", "bybit", "cmssl", "aux"):
            assert bad not in low


def test_no_ambiguous_label_api_surface():
    source = inspect.getsource(lb)
    forbidden = (
        "def observe_" + "price(",
        "def on_" + "decision(",
        "def label_" + "now(",
        "def on_price_and_" + "decision(",
        "def build_labels_from_" + "price_arrays(",
        "def label_ready_" + "ts_us(",
        "def label_entry_" + "ts_us(",
    )
    for needle in forbidden:
        assert needle not in source
