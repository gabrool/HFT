"""Event-triggered decision scheduling for the MMRT feature pipeline.

This module owns the single definition of when decisions fire on a causal
market event stream: wake conditions (trades, top-of-book changes), a minimum
re-decision interval (throttle), and a maximum interval (heartbeat). The
feature engine drives it during replay, so supervised dataset ingest, linear
signal building, and execution replay all share one decision grid by
construction.

Decisions can only fire on book events: trades and book changes arm the
trigger, and the decision is emitted at the next book event that satisfies
the throttle. It is IO-free and knows nothing about features, labels, tapes,
or models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite float")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return out


DEFAULT_MIN_DECISION_INTERVAL_US = 100_000
DEFAULT_MAX_DECISION_INTERVAL_US = 500_000
DEFAULT_L1_SIZE_CHANGE_FRACTION = 0.5

_SCHEDULE_PAYLOAD_FIELDS = (
    "min_decision_interval_us",
    "max_decision_interval_us",
    "wake_on_trade",
    "wake_on_top_of_book",
    "l1_size_change_fraction",
)


@dataclass(frozen=True, slots=True)
class DecisionScheduleConfig:
    min_decision_interval_us: int = DEFAULT_MIN_DECISION_INTERVAL_US
    max_decision_interval_us: int = DEFAULT_MAX_DECISION_INTERVAL_US
    wake_on_trade: bool = True
    wake_on_top_of_book: bool = True
    l1_size_change_fraction: float = DEFAULT_L1_SIZE_CHANGE_FRACTION

    def __post_init__(self) -> None:
        _require_positive_int(self.min_decision_interval_us, "min_decision_interval_us")
        _require_positive_int(self.max_decision_interval_us, "max_decision_interval_us")
        if self.max_decision_interval_us < self.min_decision_interval_us:
            raise ValueError("max_decision_interval_us must be >= min_decision_interval_us")
        _require_bool(self.wake_on_trade, "wake_on_trade")
        _require_bool(self.wake_on_top_of_book, "wake_on_top_of_book")
        object.__setattr__(
            self,
            "l1_size_change_fraction",
            _require_positive_float(self.l1_size_change_fraction, "l1_size_change_fraction"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "min_decision_interval_us": self.min_decision_interval_us,
            "max_decision_interval_us": self.max_decision_interval_us,
            "wake_on_trade": self.wake_on_trade,
            "wake_on_top_of_book": self.wake_on_top_of_book,
            "l1_size_change_fraction": self.l1_size_change_fraction,
        }


def decision_schedule_config_from_dict(payload: Mapping[str, object]) -> DecisionScheduleConfig:
    """Rebuild a DecisionScheduleConfig from a stored identity payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("decision schedule payload must be a mapping")
    missing = [k for k in _SCHEDULE_PAYLOAD_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"decision schedule payload missing fields: {missing}")
    extra = sorted(set(payload) - set(_SCHEDULE_PAYLOAD_FIELDS))
    if extra:
        raise ValueError(f"decision schedule payload has unknown fields: {extra}")
    return DecisionScheduleConfig(
        min_decision_interval_us=int(payload["min_decision_interval_us"]),
        max_decision_interval_us=int(payload["max_decision_interval_us"]),
        wake_on_trade=bool(payload["wake_on_trade"]),
        wake_on_top_of_book=bool(payload["wake_on_top_of_book"]),
        l1_size_change_fraction=float(payload["l1_size_change_fraction"]),
    )


class DecisionSchedule:
    """Stateful trigger evaluated on the causal event stream.

    ``observe_trade``/``observe_book`` arm the trigger from wake conditions;
    ``should_fire`` answers whether a decision fires at the current book
    event; ``mark_decision`` resets the throttle window. The first eligible
    book event always fires.
    """

    __slots__ = (
        "config",
        "_last_decision_local_ts_us",
        "_armed",
        "_prev_best_bid",
        "_prev_best_ask",
        "_prev_bid_l1_size",
        "_prev_ask_l1_size",
    )

    def __init__(self, config: DecisionScheduleConfig | None = None) -> None:
        if config is not None and not isinstance(config, DecisionScheduleConfig):
            raise ValueError("config must be DecisionScheduleConfig")
        self.config = config or DecisionScheduleConfig()
        self.reset()

    def reset(self) -> None:
        self._last_decision_local_ts_us: int | None = None
        self._armed = False
        self._prev_best_bid: float | None = None
        self._prev_best_ask: float | None = None
        self._prev_bid_l1_size: float | None = None
        self._prev_ask_l1_size: float | None = None

    @property
    def last_decision_local_ts_us(self) -> int | None:
        return self._last_decision_local_ts_us

    @property
    def is_armed(self) -> bool:
        return self._armed

    def observe_trade(self, local_ts_us: int) -> None:
        _require_positive_int(local_ts_us, "local_ts_us")
        if self.config.wake_on_trade:
            self._armed = True

    def observe_book(
        self,
        local_ts_us: int,
        *,
        best_bid: float,
        best_ask: float,
        bid_l1_size: float,
        ask_l1_size: float,
    ) -> None:
        _require_positive_int(local_ts_us, "local_ts_us")
        changed = (
            self._prev_best_bid is None
            or best_bid != self._prev_best_bid
            or best_ask != self._prev_best_ask
            or self._l1_size_changed(bid_l1_size, self._prev_bid_l1_size)
            or self._l1_size_changed(ask_l1_size, self._prev_ask_l1_size)
        )
        self._prev_best_bid = best_bid
        self._prev_best_ask = best_ask
        self._prev_bid_l1_size = bid_l1_size
        self._prev_ask_l1_size = ask_l1_size
        if changed and self.config.wake_on_top_of_book:
            self._armed = True

    def _l1_size_changed(self, size: float, prev: float | None) -> bool:
        if prev is None:
            return True
        reference = max(abs(prev), 1e-12)
        return abs(size - prev) / reference >= self.config.l1_size_change_fraction

    def should_fire(self, local_ts_us: int) -> bool:
        _require_positive_int(local_ts_us, "local_ts_us")
        last = self._last_decision_local_ts_us
        if last is None:
            return True
        elapsed = local_ts_us - last
        if elapsed < self.config.min_decision_interval_us:
            return False
        if elapsed >= self.config.max_decision_interval_us:
            return True
        return self._armed

    def mark_decision(self, local_ts_us: int) -> None:
        _require_positive_int(local_ts_us, "local_ts_us")
        if self._last_decision_local_ts_us is not None and local_ts_us <= self._last_decision_local_ts_us:
            raise ValueError("decision local_ts_us must be strictly increasing")
        self._last_decision_local_ts_us = local_ts_us
        self._armed = False


__all__ = [
    "DEFAULT_MIN_DECISION_INTERVAL_US",
    "DEFAULT_MAX_DECISION_INTERVAL_US",
    "DEFAULT_L1_SIZE_CHANGE_FRACTION",
    "DecisionScheduleConfig",
    "DecisionSchedule",
    "decision_schedule_config_from_dict",
]
