from datetime import datetime, timezone
from decimal import Decimal

import pytest

from mmrt.time_utils import (
    TARDIS_TIMESTAMP_UNIT,
    US_PER_MS,
    US_PER_SECOND,
    add_us,
    datetime_to_us,
    duration_label_us,
    elapsed_us,
    exchange_order_key,
    first_non_monotonic_index,
    is_non_decreasing,
    label_name_for_horizon_us,
    ms_to_us,
    parse_iso8601_utc_to_us,
    parse_optional_tardis_ts_us,
    parse_tardis_ts_us,
    require_int_us,
    require_nonnegative_duration_us,
    require_positive_duration_us,
    seconds_to_us,
    sub_us,
    tardis_order_key,
    us_to_iso8601_utc,
    validate_non_decreasing_us,
)


def test_constants():
    assert US_PER_MS == 1000
    assert US_PER_SECOND == 1_000_000
    assert TARDIS_TIMESTAMP_UNIT == "us"


def test_require_helpers_reject_bool():
    with pytest.raises(ValueError):
        require_int_us(True)
    with pytest.raises(ValueError):
        require_nonnegative_duration_us(False)
    with pytest.raises(ValueError):
        require_positive_duration_us(True)


def test_second_ms_conversions():
    assert seconds_to_us(1) == 1_000_000
    assert seconds_to_us("0.5") == 500_000
    assert seconds_to_us(Decimal("0.000001")) == 1
    assert ms_to_us(1) == 1000
    assert ms_to_us("0.5") == 500
    with pytest.raises(ValueError):
        seconds_to_us(float("nan"))
    with pytest.raises(ValueError):
        seconds_to_us(True)


def test_parse_tardis_ts_us():
    expected = 1_599_868_800_280_000
    assert parse_tardis_ts_us(expected) == expected
    assert parse_tardis_ts_us(str(expected)) == expected
    assert parse_tardis_ts_us(f"{expected}.0") == expected
    with pytest.raises(ValueError):
        parse_tardis_ts_us(f"{expected}.1")
    with pytest.raises(ValueError):
        parse_tardis_ts_us(float(expected))
    with pytest.raises(ValueError):
        parse_tardis_ts_us("")
    with pytest.raises(ValueError):
        parse_tardis_ts_us(True)

    assert parse_optional_tardis_ts_us(None) is None
    assert parse_optional_tardis_ts_us("") is None
    assert parse_optional_tardis_ts_us(str(expected)) == expected


def test_datetime_iso_conversion():
    assert parse_iso8601_utc_to_us("1970-01-01T00:00:01Z") == 1_000_000
    assert parse_iso8601_utc_to_us("1970-01-01T00:00:01.000250+00:00") == 1_000_250
    with pytest.raises(ValueError):
        parse_iso8601_utc_to_us("1970-01-01T00:00:01")

    assert us_to_iso8601_utc(1_000_000) == "1970-01-01T00:00:01.000000Z"
    assert datetime_to_us(datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)) == 1_000_000
    with pytest.raises(ValueError):
        datetime_to_us(datetime(1970, 1, 1, 0, 0, 1))


def test_duration_labels():
    assert duration_label_us(200_000) == "200ms"
    assert duration_label_us(500_000) == "500ms"
    assert duration_label_us(1_000_000) == "1s"
    assert duration_label_us(1_000) == "1ms"
    assert duration_label_us(250) == "250us"
    assert duration_label_us(60_000_000) == "1min"
    with pytest.raises(ValueError):
        duration_label_us(0)


def test_label_name_helper():
    assert label_name_for_horizon_us("ret_bps", 1_000_000) == "ret_bps_1s"
    assert label_name_for_horizon_us("ret_bps", 1_000_000, entry_delay_us=1_000) == "ret_bps_1s_delay_1ms"
    assert label_name_for_horizon_us("ret_bps", 1_000_000, entry_delay_us=0) == "ret_bps_1s_delay_0us"
    with pytest.raises(ValueError):
        label_name_for_horizon_us("Ret", 1_000_000)
    with pytest.raises(ValueError):
        label_name_for_horizon_us("ret-bps", 1_000_000)


def test_ordering_helpers():
    assert tardis_order_key(100, 110, 7) == (110, 100, 7)
    assert exchange_order_key(100, 110, 7) == (100, 110, 7)
    with pytest.raises(ValueError):
        tardis_order_key(0, 110, 7)
    with pytest.raises(ValueError):
        exchange_order_key(100, 0, 7)
    with pytest.raises(ValueError):
        tardis_order_key(100, 110, True)


def test_monotonicity_helpers():
    assert is_non_decreasing([1, 1, 2]) is True
    assert first_non_monotonic_index([1, 1, 2], allow_equal=True) is None
    assert first_non_monotonic_index([1, 1, 2], allow_equal=False) == 1
    assert first_non_monotonic_index([1, 3, 2], allow_equal=True) == 2
    validate_non_decreasing_us([1, 2, 2], allow_equal=True)
    with pytest.raises(ValueError):
        validate_non_decreasing_us([1, 2, 2], allow_equal=False)


def test_arithmetic_helpers():
    assert add_us(100, 50) == 150
    assert sub_us(100, 50) == 50
    with pytest.raises(ValueError):
        sub_us(100, 100)
    assert elapsed_us(100, 150) == 50
    with pytest.raises(ValueError):
        elapsed_us(150, 100)
