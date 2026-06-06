"""Observation schema definitions for execution-state feature vectors.

This module intentionally owns only field names, ordering, schema validation,
name/index lookup, and empty vector allocation. It does not import execution
state contracts or any environment, fill, reward, tape, model, or RL code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


MARKET_GROUP = "market"
LINEAR_SIGNAL_GROUP = "linear_signal"
POSITION_GROUP = "position"
ORDERS_GROUP = "orders"
FILLS_GROUP = "fills"
TIME_GROUP = "time"

DEFAULT_OBSERVATION_DTYPE = "float32"
ALLOWED_OBSERVATION_DTYPES = ("float32", "float64")

MARKET_FIELDS = (
    "spread_ticks",
    "spread_bps",
    "mid_price",
    "microprice_bps",
    "top_imbalance",
    "bid_depth_count",
    "ask_depth_count",
    "bid_top_size",
    "ask_top_size",
)

LINEAR_SIGNAL_FIELDS = (
    "linear_p_no_move",
    "linear_p_move",
    "linear_p_up_move",
    "linear_p_down_move",
    "linear_signed_move_prob",
    "linear_expected_up_bps",
    "linear_expected_down_bps",
    "linear_expected_return_bps",
    "linear_expected_abs_move_bps",
    "linear_predicted_vol_bps",
    "linear_confidence",
)

POSITION_FIELDS = (
    "cash",
    "inventory_qty",
    "inventory_notional",
    "inventory_notional_bps",
    "equity",
    "inventory_abs_notional",
    "fees_paid",
)

ORDERS_FIELDS = (
    "has_live_bid",
    "has_live_ask",
    "bid_distance_ticks",
    "ask_distance_ticks",
    "bid_distance_bps",
    "ask_distance_bps",
    "bid_qty",
    "ask_qty",
    "bid_remaining_qty",
    "ask_remaining_qty",
    "bid_queue_ahead_qty",
    "ask_queue_ahead_qty",
    "bid_age_ms",
    "ask_age_ms",
)

FILLS_FIELDS = (
    "last_fill_side",
    "last_fill_qty",
    "last_fill_notional",
    "last_fill_fee",
    "last_fill_age_ms",
    "step_fill_count",
    "step_buy_fill_qty",
    "step_sell_fill_qty",
    "step_fill_notional",
)

TIME_FIELDS = (
    "local_time_since_start_s",
    "event_progress",
    "time_since_last_event_ms",
)

DEFAULT_OBSERVATION_FIELDS = (
    *MARKET_FIELDS,
    *LINEAR_SIGNAL_FIELDS,
    *POSITION_FIELDS,
    *ORDERS_FIELDS,
    *FILLS_FIELDS,
    *TIME_FIELDS,
)


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    field_names: tuple[str, ...] = DEFAULT_OBSERVATION_FIELDS
    dtype: str = DEFAULT_OBSERVATION_DTYPE

    def __post_init__(self) -> None:
        names = _field_names_tuple(self.field_names)
        if not names:
            raise ValueError("field_names must be non-empty")
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate observation field name: {name!r}")
            seen.add(name)
        if self.dtype not in ALLOWED_OBSERVATION_DTYPES:
            raise ValueError(f"dtype must be one of {ALLOWED_OBSERVATION_DTYPES}")
        object.__setattr__(self, "field_names", names)

    @property
    def dim(self) -> int:
        return len(self.field_names)

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def index(self, name: str) -> int:
        return self.field_names.index(name)

    def has_field(self, name: str) -> bool:
        return name in self.field_names

    def names(self) -> tuple[str, ...]:
        return self.field_names

    def empty(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=self.np_dtype)

    def as_dict(self) -> dict[str, object]:
        return {"field_names": list(self.field_names), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ObservationSchema":
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        field_names = payload.get("field_names", DEFAULT_OBSERVATION_FIELDS)
        dtype = payload.get("dtype", DEFAULT_OBSERVATION_DTYPE)
        if not isinstance(dtype, str):
            raise ValueError("dtype must be a string")
        return cls(field_names=_field_names_tuple(field_names), dtype=dtype)


def _field_names_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("field_names must be a sequence of non-empty strings, not a single string/bytes value")
    if not isinstance(values, Sequence):
        raise ValueError("field_names must be a sequence of non-empty strings")
    names = tuple(values)
    for i, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise ValueError(f"field_names[{i}] must be a non-empty string")
    return names


def default_observation_schema(dtype: str = DEFAULT_OBSERVATION_DTYPE) -> ObservationSchema:
    return ObservationSchema(dtype=dtype)


def observation_field_groups() -> dict[str, tuple[str, ...]]:
    return {
        MARKET_GROUP: MARKET_FIELDS,
        LINEAR_SIGNAL_GROUP: LINEAR_SIGNAL_FIELDS,
        POSITION_GROUP: POSITION_FIELDS,
        ORDERS_GROUP: ORDERS_FIELDS,
        FILLS_GROUP: FILLS_FIELDS,
        TIME_GROUP: TIME_FIELDS,
    }


def validate_observation_vector(
    obs: np.ndarray,
    *,
    schema: ObservationSchema = ObservationSchema(),
) -> np.ndarray:
    if not isinstance(obs, np.ndarray):
        raise ValueError("obs must be a NumPy array")
    if obs.shape != (schema.dim,):
        raise ValueError(f"obs shape must be ({schema.dim},)")
    if np.dtype(obs.dtype) != schema.np_dtype:
        raise ValueError(f"obs dtype must be {schema.np_dtype}")
    if not obs.flags.c_contiguous:
        raise ValueError("obs must be C-contiguous")
    if not np.isfinite(obs).all():
        raise ValueError("obs must contain only finite values")
    return obs


__all__ = [
    "MARKET_GROUP",
    "LINEAR_SIGNAL_GROUP",
    "POSITION_GROUP",
    "ORDERS_GROUP",
    "FILLS_GROUP",
    "TIME_GROUP",
    "DEFAULT_OBSERVATION_DTYPE",
    "ALLOWED_OBSERVATION_DTYPES",
    "MARKET_FIELDS",
    "LINEAR_SIGNAL_FIELDS",
    "POSITION_FIELDS",
    "ORDERS_FIELDS",
    "FILLS_FIELDS",
    "TIME_FIELDS",
    "DEFAULT_OBSERVATION_FIELDS",
    "ObservationSchema",
    "default_observation_schema",
    "observation_field_groups",
    "validate_observation_vector",
]
