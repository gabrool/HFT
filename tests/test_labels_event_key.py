import math
import numpy as np
import pytest

from mmrt.contracts import LabelSpec
from mmrt.features.labels import (
    LabelBuilder,
    PriceHistory,
    PriceObservation,
    build_labels_from_price_event_arrays,
)
from mmrt.time_key import EventKey


def test_asof_same_timestamp_uses_event_sequence_no_leak():
    h = PriceHistory()
    h.append(PriceObservation(1_000_000, 0, 100.0))
    h.append(PriceObservation(1_000_000, 1, 101.0))
    assert h.asof_price(EventKey(1_000_000, 0)) == 100.0
    assert h.asof_price(EventKey(1_000_000, 1)) == 101.0


def test_entry_delay_zero_uses_decision_event_key():
    spec = LabelSpec(horizons_us=(100,), entry_delay_us=0)
    b = LabelBuilder(spec)
    b.observe_price_local(1_000_000, 0, 100.0)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_000, 1, 200.0)
    assert b.observe_price_local(1_000_100, 2, 110.0) == []
    out = b.observe_price_local(1_000_101, 3, 110.0)
    assert len(out) == 1
    assert out[0].values_bps[0] == pytest.approx(10_000.0 * math.log(110.0 / 100.0))


def test_entry_delay_positive_can_use_future_timestamp_last_event():
    spec = LabelSpec(horizons_us=(100,), entry_delay_us=100)
    b = LabelBuilder(spec)
    b.observe_price_local(1_000_000, 0, 100.0)
    b.on_decision_local(1_000_000, 0)
    b.observe_price_local(1_000_100, 1, 101.0)
    b.observe_price_local(1_000_100, 2, 102.0)
    assert b.observe_price_local(1_000_200, 3, 104.0) == []
    out = b.observe_price_local(1_000_201, 4, 104.0)
    assert out[0].values_bps[0] == pytest.approx(10_000.0 * math.log(104.0 / 102.0))


def test_duplicate_event_key_rejected():
    h = PriceHistory()
    h.append(PriceObservation(1_000_000, 0, 100.0))
    with pytest.raises(ValueError):
        h.append(PriceObservation(1_000_000, 0, 101.0))


def test_vectorized_matches_incremental_builder():
    spec = LabelSpec(horizons_us=(100, 200), entry_delay_us=0)
    dts = np.array([1_000_000, 1_000_100], dtype=np.int64)
    dseq = np.array([0, 2], dtype=np.int64)
    pts = np.array([1_000_000, 1_000_000, 1_000_100, 1_000_200, 1_000_300], dtype=np.int64)
    pseq = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    vals = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    labels, valid = build_labels_from_price_event_arrays(dts, dseq, pts, pseq, vals, spec)
    assert valid.tolist() == [True, True]

    b = LabelBuilder(spec)
    b.on_decision_local(1_000_000, 0)
    b.on_decision_local(1_000_100, 2)
    out = []
    for t, q, p in zip(pts, pseq, vals, strict=True):
        out.extend(b.observe_price_local(int(t), int(q), float(p)))
    out.extend(b.finalize_at_eof())
    assert np.allclose(labels[valid], np.array([r.values_bps for r in out]))


def test_same_timestamp_distinct_decision_event_sequences_allowed():
    spec = LabelSpec(horizons_us=(100_000,), entry_delay_us=0)
    b = LabelBuilder(spec)

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
    b = LabelBuilder(LabelSpec(horizons_us=(100_000,), entry_delay_us=0))

    b.on_decision_local(1_000_000, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        b.on_decision_local(1_000_000, 0)
