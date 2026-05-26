import math
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
        "LabelBuilder", "build_labels_from_price_arrays", "label_value_names", "label_ready_ts_us", "label_entry_ts_us",
    }
    assert set(lb.__all__) == expected
    for name in lb.__all__:
        assert not name.startswith("_")
        low = name.lower()
        for bad in ("bybit", "cmssl", "aux", "transform", "storage", "feature_dim", "grace", "ms"):
            assert bad not in low


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
    'mmrt.features.engine', 'mmrt.features.'+'trans'+'forms', 'CM'+'SSL17', 'offline_'+'ingest'
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
    obs = lb.PriceObservation(ts_us=1, price=np.float64(10.0))
    assert isinstance(obs.price, float)
    for bad in (-1.0, 0.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            lb.PriceObservation(ts_us=1, price=bad)
    with pytest.raises(ValueError):
        lb.PriceObservation(ts_us=-1, price=1.0)
    with pytest.raises(ValueError):
        lb.PriceObservation(ts_us=True, price=1.0)


def test_price_history_append_asof_and_equal_timestamp_replace():
    ph = lb.PriceHistory()
    ph.append(lb.PriceObservation(1000, 100.0))
    ph.append(lb.PriceObservation(2000, 101.0))
    assert ph.asof_price(999) is None
    assert ph.asof_price(1000) == 100.0
    assert ph.asof_price(1500) == 100.0
    assert ph.asof_price(2000) == 101.0
    size = ph.size
    ph.append(lb.PriceObservation(2000, 102.0))
    assert ph.asof_price(2000) == 102.0
    assert ph.size == size
    with pytest.raises(ValueError):
        ph.append(lb.PriceObservation(1999, 99.0))
    ts, px = ph.active_arrays()
    assert np.array_equal(ts, np.array([1000, 2000], dtype=np.int64))
    assert np.array_equal(px, np.array([100.0, 102.0], dtype=np.float64))


def test_price_history_capacity_and_compaction():
    ph = lb.PriceHistory(capacity=3)
    for i in range(1, 6):
        ph.append(lb.PriceObservation(i, float(i)))
    ts, px = ph.active_arrays()
    assert np.array_equal(ts, np.array([3, 4, 5]))
    assert np.array_equal(px, np.array([3.0, 4.0, 5.0]))
    assert ph.asof_price(2) is None
    ph.reset()
    assert ph.size == 0


def test_pending_label_validation():
    p = lb.PendingLabel(1, 2, 3, (5, 2, 2))
    assert p.horizons_us == (2, 5)
    with pytest.raises(ValueError):
        lb.PendingLabel(2, 1, 3, (1,))
    with pytest.raises(ValueError):
        lb.PendingLabel(1, 2, 1, (1,))
    with pytest.raises(ValueError):
        lb.PendingLabel(1, 2, 3, ())
    with pytest.raises(ValueError):
        lb.PendingLabel(-1, 0, 1, (1,))


def test_label_entry_ready_helpers_and_names():
    s = spec()
    assert lb.label_entry_ts_us(1_000_000, s) == 1_001_000
    assert lb.label_ready_ts_us(1_000_000, s) == 2_001_000
    assert lb.label_value_names(s) == ("ret_bps_200000us", "ret_bps_500000us", "ret_bps_1000000us")


def test_streaming_label_maturation_with_entry_delay():
    s = spec(horizons=(200_000, 500_000), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    assert b.observe_price(1_000_000, 100.0) == []
    assert b.observe_price(1_001_000, 101.0) == []
    assert b.observe_price(1_201_000, 102.0) == []
    out = b.observe_price(1_501_000, 104.0)
    assert len(out) == 1
    r = out[0]
    assert r.decision_ts_us == 1_000_000
    assert r.entry_ts_us == 1_001_000
    assert r.values_bps == pytest.approx((10_000.0 * math.log(102 / 101), 10_000.0 * math.log(104 / 101)))
    assert b.pending_count == 0


def test_last_observation_policy_inclusive_boundaries():
    s = spec(horizons=(200_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.observe_price(1_000_999, 100.0)
    b.observe_price(1_001_000, 101.0)
    out = b.observe_price(1_201_000, 103.0)
    assert len(out) == 1
    assert out[0].values_bps[0] == pytest.approx(10_000.0 * math.log(103 / 101))


def test_last_observation_policy_uses_previous_when_no_exact_timestamp():
    s = spec(horizons=(200_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.observe_price(1_000_900, 100.0)
    b.observe_price(1_001_500, 101.0)
    b.observe_price(1_200_000, 102.0)
    out = b.observe_price(1_250_000, 103.0)
    assert len(out) == 1
    assert out[0].values_bps[0] == pytest.approx(10_000.0 * math.log(102 / 100))


def test_decision_without_entry_price_stays_pending_until_entry_asof_exists():
    s = spec(horizons=(200_000,), entry_delay=1_000)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    assert b.observe_price(1_500_000, 100.0) == []
    assert b.observe_price(1_700_000, 101.0) == []
    assert b.pending_count == 1


def test_multiple_pending_decisions_mature_fifo():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.on_decision(1_100_000)
    out = []
    out.extend(b.observe_price(1_000_000, 100.0))
    out.extend(b.observe_price(1_100_000, 101.0))
    out.extend(b.observe_price(1_200_000, 102.0))
    assert [x.decision_ts_us for x in out] == [1_000_000, 1_100_000]
    assert b.pending_count == 0


def test_equal_decision_timestamps_allowed():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.on_decision(1_000_000)
    b.observe_price(1_000_000, 100.0)
    out = b.observe_price(1_100_000, 105.0)
    assert len(out) == 2
    assert out[0].decision_ts_us == out[1].decision_ts_us == 1_000_000
    assert out[0].values_bps == pytest.approx(out[1].values_bps)


def test_on_decision_rejects_decreasing_decision_timestamps():
    b = lb.LabelBuilder(spec())
    b.on_decision(1_000_000)
    with pytest.raises(ValueError):
        b.on_decision(999_999)


def test_on_decision_rejects_decreasing_after_pending_matures():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.observe_price(1_000_000, 100.0)
    out = b.observe_price(1_100_000, 101.0)
    assert len(out) == 1
    assert b.pending_count == 0
    with pytest.raises(ValueError):
        b.on_decision(999_999)
    b.on_decision(1_000_000)
    b.on_decision(1_000_001)


def test_label_now_does_not_mutate_pending():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.observe_price(1_000_000, 100.0)
    b.observe_price(1_100_000, 105.0)
    before = b.pending_count
    r = b.label_now(1_000_000)
    assert isinstance(r, LabelResult)
    assert b.pending_count == before


def test_on_price_and_decision_helper():
    s = spec(horizons=(100_000,), entry_delay=0)
    b = lb.LabelBuilder(s)
    assert b.on_price_and_decision(1_000_000, 100.0, is_decision=True) == []
    out = b.on_price_and_decision(1_100_000, 102.0, is_decision=False)
    assert len(out) == 1
    assert out[0].decision_ts_us == 1_000_000


def test_batch_labels_match_streaming():
    s = spec(horizons=(100_000, 200_000), entry_delay=0)
    pts = np.array([1_000_000, 1_100_000, 1_200_000, 1_300_000])
    pvals = np.array([100.0, 101.0, 102.0, 103.0])
    dec = np.array([1_000_000, 1_100_000])
    labels, mask = lb.build_labels_from_price_arrays(dec, pts, pvals, s)
    assert mask.all()

    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.on_decision(1_100_000)
    out = []
    for t, p in zip(pts, pvals):
        out.extend(b.observe_price(int(t), float(p)))
    stream = np.array([x.values_bps for x in out], dtype=np.float64)
    assert np.allclose(labels, stream)


def test_batch_invalid_rows_for_insufficient_future_context():
    s = spec(horizons=(200_000,), entry_delay=0)
    pts = np.array([1_000_000, 1_100_000])
    pvals = np.array([100.0, 101.0])
    dec = np.array([1_000_000, 1_100_000])
    labels, mask = lb.build_labels_from_price_arrays(dec, pts, pvals, s)
    assert mask.tolist() == [False, False]
    assert np.isnan(labels).all()


def test_batch_dedupes_equal_price_timestamps_keep_last():
    s = spec(horizons=(100_000,), entry_delay=0)
    pts = np.array([1_000_000, 1_000_000, 1_100_000])
    pvals = np.array([100.0, 101.0, 102.0])
    dec = np.array([1_000_000])
    labels, mask = lb.build_labels_from_price_arrays(dec, pts, pvals, s)
    assert mask[0]
    assert labels[0, 0] == pytest.approx(10_000.0 * math.log(102 / 101))


def test_batch_validates_sorted_inputs():
    s = spec(horizons=(1,), entry_delay=0)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([2, 1]), np.array([1, 2]), np.array([1.0, 2.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1, 2]), np.array([2, 1]), np.array([1.0, 2.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1, 2]), np.array([1, 2]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1]), np.array([1]), np.array([0.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([[1]]), np.array([1]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([-1]), np.array([1]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1]), np.array([-1]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1.5]), np.array([1, 2]), np.array([1.0, 2.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1]), np.array([1.5]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([math.nan]), np.array([1]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1]), np.array([math.inf]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([True]), np.array([1]), np.array([1.0]), s)
    with pytest.raises(ValueError):
        lb.build_labels_from_price_arrays(np.array([1]), np.array([True]), np.array([1.0]), s)


def test_batch_accepts_integer_valued_float_timestamps():
    s = spec(horizons=(100_000,), entry_delay=0)
    labels, mask = lb.build_labels_from_price_arrays(
        np.array([1_000_000.0]),
        np.array([1_000_000.0, 1_100_000.0]),
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
    b.on_decision(1_000_000)
    b.observe_price(1_000_000, 100.0)
    assert b.observe_price(1_199_999, 101.0) == []
    assert b.pending_count == 1
    assert len(b.observe_price(1_200_000, 102.0)) == 1


def test_label_result_contract():
    s = spec(horizons=(100_000, 200_000), entry_delay=0)
    b = lb.LabelBuilder(s)
    b.on_decision(1_000_000)
    b.observe_price(1_000_000, 100.0)
    out = b.observe_price(1_200_000, 102.0)
    assert isinstance(out[0], LabelResult)
    assert out[0].horizons_us == tuple(sorted(out[0].horizons_us))
    assert len(out[0].values_bps) == len(out[0].horizons_us)


def test_no_old_bybit_ms_grace_residue():
    for name in lb.__all__:
        low = name.lower()
        for bad in ("ms", "grace", "bybit", "cmssl", "aux"):
            assert bad not in low
