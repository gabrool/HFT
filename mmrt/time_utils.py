"""Microsecond-native timestamp and duration utilities for the MMRT pipeline. This module is IO-free and does not parse market-data rows; it only converts, validates, formats, and compares timestamp/duration values."""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import math
import re
from typing import Iterable, Sequence

US_PER_MS = 1_000
US_PER_SECOND = 1_000_000
US_PER_MINUTE = 60 * US_PER_SECOND
US_PER_HOUR = 60 * US_PER_MINUTE
US_PER_DAY = 24 * US_PER_HOUR

TARDIS_TIMESTAMP_UNIT = "us"

UNIX_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _reject_bool(value, name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be bool")


def _decimal_from_number(value, name: str) -> Decimal:
    _reject_bool(value, name)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Decimal(str(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise ValueError(f"{name} must not be empty")
        try:
            parsed = Decimal(stripped)
        except InvalidOperation as exc:
            raise ValueError(f"{name} must be a valid number") from exc
        if not parsed.is_finite():
            raise ValueError(f"{name} must be finite")
        return parsed
    raise ValueError(f"{name} must be int, float, str, or Decimal")


def _round_half_even_to_int(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_HALF_EVEN))


def require_int_us(value: int, name: str = "timestamp_us", *, allow_zero: bool = False) -> int:
    _reject_bool(value, name)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be int")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    return value


def require_nonnegative_duration_us(value: int, name: str = "duration_us") -> int:
    _reject_bool(value, name)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def require_positive_duration_us(value: int, name: str = "duration_us") -> int:
    _reject_bool(value, name)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def seconds_to_us(value) -> int:
    dec = _decimal_from_number(value, "seconds")
    return _round_half_even_to_int(dec * US_PER_SECOND)


def ms_to_us(value) -> int:
    dec = _decimal_from_number(value, "milliseconds")
    return _round_half_even_to_int(dec * US_PER_MS)


def us_to_seconds(value_us: int) -> float:
    value_us = require_int_us(value_us, "value_us", allow_zero=True)
    return value_us / US_PER_SECOND


def us_to_ms(value_us: int) -> float:
    value_us = require_int_us(value_us, "value_us", allow_zero=True)
    return value_us / US_PER_MS


def us_to_datetime_utc(ts_us: int) -> datetime:
    ts_us = require_int_us(ts_us, "ts_us")
    return datetime.fromtimestamp(ts_us / US_PER_SECOND, tz=timezone.utc)


def datetime_to_us(dt: datetime) -> int:
    if not isinstance(dt, datetime):
        raise ValueError("dt must be datetime")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("dt must be timezone-aware")
    delta = dt.astimezone(timezone.utc) - UNIX_EPOCH_UTC
    return delta.days * US_PER_DAY + delta.seconds * US_PER_SECOND + delta.microseconds


def parse_tardis_ts_us(value) -> int:
    _reject_bool(value, "tardis_ts_us")
    if isinstance(value, float):
        raise ValueError("tardis_ts_us float values are not allowed")

    if isinstance(value, int):
        return require_int_us(value, "tardis_ts_us")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise ValueError("tardis_ts_us must not be empty")
        try:
            dec = Decimal(stripped)
        except InvalidOperation as exc:
            raise ValueError("tardis_ts_us must be a valid integer") from exc
    elif isinstance(value, Decimal):
        dec = value
    else:
        raise ValueError("tardis_ts_us must be int, str, or Decimal")

    if not dec.is_finite():
        raise ValueError("tardis_ts_us must be finite")
    if dec != dec.to_integral_value():
        raise ValueError("tardis_ts_us must be integral")
    return require_int_us(int(dec), "tardis_ts_us")


def parse_optional_tardis_ts_us(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return parse_tardis_ts_us(value)


def parse_iso8601_utc_to_us(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("value must be str")
    stripped = value.strip()
    if stripped == "":
        raise ValueError("value must not be empty")
    normalized = stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("invalid ISO-8601 datetime") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("ISO-8601 datetime must be timezone-aware")
    return datetime_to_us(dt.astimezone(timezone.utc))


def us_to_iso8601_utc(ts_us: int) -> str:
    dt = us_to_datetime_utc(ts_us)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def duration_label_us(duration_us: int) -> str:
    duration_us = require_positive_duration_us(duration_us)
    if duration_us % US_PER_DAY == 0:
        return f"{duration_us // US_PER_DAY}d"
    if duration_us % US_PER_HOUR == 0:
        return f"{duration_us // US_PER_HOUR}h"
    if duration_us % US_PER_MINUTE == 0:
        return f"{duration_us // US_PER_MINUTE}min"
    if duration_us % US_PER_SECOND == 0:
        return f"{duration_us // US_PER_SECOND}s"
    if duration_us % US_PER_MS == 0:
        return f"{duration_us // US_PER_MS}ms"
    return f"{duration_us}us"


def label_name_for_horizon_us(prefix: str, horizon_us: int, *, entry_delay_us: int | None = None) -> str:
    if not isinstance(prefix, str) or _PREFIX_RE.fullmatch(prefix) is None:
        raise ValueError("prefix must match ^[a-z][a-z0-9_]*$")

    horizon_us = require_positive_duration_us(horizon_us, "horizon_us")
    if entry_delay_us is None:
        return f"{prefix}_{duration_label_us(horizon_us)}"

    entry_delay_us = require_nonnegative_duration_us(entry_delay_us, "entry_delay_us")
    delay_label = "0us" if entry_delay_us == 0 else duration_label_us(entry_delay_us)
    return f"{prefix}_{duration_label_us(horizon_us)}_delay_{delay_label}"


def tardis_order_key(ts_us: int, local_ts_us: int, source_row: int) -> tuple[int, int, int]:
    ts_us = require_int_us(ts_us, "ts_us")
    local_ts_us = require_int_us(local_ts_us, "local_ts_us")
    _reject_bool(source_row, "source_row")
    if not isinstance(source_row, int) or source_row < 0:
        raise ValueError("source_row must be a nonnegative int")
    return (local_ts_us, ts_us, source_row)


def exchange_order_key(ts_us: int, local_ts_us: int, source_row: int) -> tuple[int, int, int]:
    ts_us = require_int_us(ts_us, "ts_us")
    local_ts_us = require_int_us(local_ts_us, "local_ts_us")
    _reject_bool(source_row, "source_row")
    if not isinstance(source_row, int) or source_row < 0:
        raise ValueError("source_row must be a nonnegative int")
    return (ts_us, local_ts_us, source_row)


def is_non_decreasing(values: Iterable[int]) -> bool:
    prev = None
    for idx, value in enumerate(values):
        current = require_int_us(value, f"values[{idx}]", allow_zero=True)
        if prev is not None and current < prev:
            return False
        prev = current
    return True


def first_non_monotonic_index(values: Sequence[int], *, allow_equal: bool = True) -> int | None:
    if len(values) < 2:
        for idx, value in enumerate(values):
            require_int_us(value, f"values[{idx}]", allow_zero=True)
        return None

    prev = require_int_us(values[0], "values[0]", allow_zero=True)
    for idx in range(1, len(values)):
        current = require_int_us(values[idx], f"values[{idx}]", allow_zero=True)
        if allow_equal:
            if current < prev:
                return idx
        else:
            if current <= prev:
                return idx
        prev = current
    return None


def validate_non_decreasing_us(values: Sequence[int], name: str = "timestamps_us", *, allow_equal: bool = True) -> None:
    idx = first_non_monotonic_index(values, allow_equal=allow_equal)
    if idx is None:
        return
    prev = values[idx - 1]
    curr = values[idx]
    raise ValueError(f"{name} is non-monotonic at index {idx}: prev={prev}, current={curr}")


def add_us(ts_us: int, delta_us: int) -> int:
    ts_us = require_int_us(ts_us, "ts_us")
    delta_us = require_nonnegative_duration_us(delta_us, "delta_us")
    result = ts_us + delta_us
    if result <= 0:
        raise ValueError("result must be > 0")
    return result


def sub_us(ts_us: int, delta_us: int) -> int:
    ts_us = require_int_us(ts_us, "ts_us")
    delta_us = require_nonnegative_duration_us(delta_us, "delta_us")
    result = ts_us - delta_us
    if result <= 0:
        raise ValueError("result must be > 0")
    return result


def elapsed_us(start_us: int, end_us: int) -> int:
    start_us = require_int_us(start_us, "start_us")
    end_us = require_int_us(end_us, "end_us")
    if end_us < start_us:
        raise ValueError("end_us must be >= start_us")
    return end_us - start_us


__all__ = [
    "US_PER_MS",
    "US_PER_SECOND",
    "US_PER_MINUTE",
    "US_PER_HOUR",
    "US_PER_DAY",
    "TARDIS_TIMESTAMP_UNIT",
    "UNIX_EPOCH_UTC",
    "require_int_us",
    "require_nonnegative_duration_us",
    "require_positive_duration_us",
    "seconds_to_us",
    "ms_to_us",
    "us_to_seconds",
    "us_to_ms",
    "us_to_datetime_utc",
    "datetime_to_us",
    "parse_tardis_ts_us",
    "parse_optional_tardis_ts_us",
    "parse_iso8601_utc_to_us",
    "us_to_iso8601_utc",
    "duration_label_us",
    "label_name_for_horizon_us",
    "tardis_order_key",
    "exchange_order_key",
    "is_non_decreasing",
    "first_non_monotonic_index",
    "validate_non_decreasing_us",
    "add_us",
    "sub_us",
    "elapsed_us",
]
