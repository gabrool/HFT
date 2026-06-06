"""Causal event keys for local-clock market data ordering."""

from __future__ import annotations

from dataclasses import dataclass

MAX_EVENT_SEQ = 2**63 - 1


def _require_int(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if positive:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    elif value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


@dataclass(frozen=True, slots=True, order=True)
class EventKey:
    local_ts_us: int
    event_seq: int

    def __post_init__(self) -> None:
        _require_int(self.local_ts_us, "local_ts_us", positive=True)
        seq = _require_int(self.event_seq, "event_seq")
        if seq > MAX_EVENT_SEQ:
            raise ValueError("event_seq must be <= MAX_EVENT_SEQ")


def key_at_or_after_timestamp(local_ts_us: int) -> EventKey:
    return EventKey(local_ts_us, MAX_EVENT_SEQ)


def key_before_or_at_decision(local_ts_us: int, event_seq: int) -> EventKey:
    return EventKey(local_ts_us, event_seq)


__all__ = [
    "EventKey",
    "MAX_EVENT_SEQ",
    "key_at_or_after_timestamp",
    "key_before_or_at_decision",
]
