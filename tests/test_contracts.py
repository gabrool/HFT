import math

import pytest

from mmrt.contracts import (
    AggressorSide,
    AsOfPolicy,
    BookSide,
    DecisionReason,
    LabelResult,
    LabelSpec,
    PriceReference,
    SplitRole,
    StorageFormat,
    TardisDataType,
    TimeRangeUS,
    TimeUnit,
)


def test_current_tardis_data_types_only():
    assert tuple(TardisDataType) == (
        TardisDataType.INCREMENTAL_BOOK_L2,
        TardisDataType.BOOK_SNAPSHOT_25,
        TardisDataType.TRADES,
    )


def test_current_label_enums_only():
    assert tuple(DecisionReason) == (DecisionReason.BOOK_STRIDE,)
    assert tuple(PriceReference) == (PriceReference.MID,)
    assert tuple(AsOfPolicy) == (AsOfPolicy.LAST_OBSERVATION,)


def test_current_shared_enums():
    assert tuple(TimeUnit) == (TimeUnit.MICROSECOND,)
    assert tuple(BookSide) == (BookSide.BID, BookSide.ASK)
    assert tuple(AggressorSide) == (AggressorSide.BUY, AggressorSide.SELL, AggressorSide.UNKNOWN)
    assert tuple(SplitRole) == (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)
    assert tuple(StorageFormat) == (StorageFormat.FLAT_DECISION_ROWS_US,)


def test_time_range_us_validation_and_contains():
    r = TimeRangeUS(100, 200)
    assert r.contains(100)
    assert r.contains(199)
    assert not r.contains(200)
    with pytest.raises(ValueError):
        TimeRangeUS(0, 1)
    with pytest.raises(ValueError):
        TimeRangeUS(2, 1)
    with pytest.raises(ValueError):
        r.contains(0)


def test_label_spec_sorts_horizons_and_context():
    spec = LabelSpec((1_000, 200), entry_delay_us=50)
    assert spec.horizons_us == (200, 1_000)
    assert spec.label_context_us == 1_050
    assert spec.price_reference == PriceReference.MID
    assert spec.asof_policy == AsOfPolicy.LAST_OBSERVATION


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(horizons_us=(), entry_delay_us=0),
        dict(horizons_us=(1, 1), entry_delay_us=0),
        dict(horizons_us=(1,), entry_delay_us=-1),
        dict(horizons_us=(1,), entry_delay_us=0, price_reference="bad"),
        dict(horizons_us=(1,), entry_delay_us=0, asof_policy="bad"),
    ],
)
def test_label_spec_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        LabelSpec(**kwargs)


def test_label_result_validation():
    r = LabelResult(decision_ts_us=100, decision_event_seq=0, entry_ts_us=101, horizons_us=(10, 20), values_bps=(1.5, -2.0))
    assert r.values_bps == (1.5, -2.0)

    with pytest.raises(ValueError):
        LabelResult(100, 0, 99, (10,), (0.0,))
    with pytest.raises(ValueError):
        LabelResult(100, 0, 100, (), ())
    with pytest.raises(ValueError):
        LabelResult(100, 0, 100, (20, 10), (0.0, 0.0))
    with pytest.raises(ValueError):
        LabelResult(100, 0, 100, (10, 10), (0.0, 0.0))
    with pytest.raises(ValueError):
        LabelResult(100, 0, 100, (10,), (0.0, 1.0))
    with pytest.raises(ValueError):
        LabelResult(100, 0, 100, (10,), (math.inf,))
