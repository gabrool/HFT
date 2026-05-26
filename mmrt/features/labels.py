"""Latency-aware label construction for the MMRT feature pipeline.

This module builds fixed-horizon return labels from causal book-mid price
observations. It consumes already-normalized microsecond timestamps and prices.
It does not parse market data, compute features, apply transforms, or write
storage artifacts.
"""

from dataclasses import dataclass
import bisect
import math

import numpy as np

from mmrt.contracts import AsOfPolicy, LabelResult, LabelSpec, PriceReference

DEFAULT_PRICE_HISTORY_CAPACITY = 1_000_000
COMPACT_MIN_START = 4096
COMPACT_FRACTION = 0.5
FLOAT_EPS = 1e-12


def _require_int_us(value: int, name: str, *, positive: bool = False, allow_zero: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if positive:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    elif allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    return value


def _require_positive_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite float > 0")
    return out


def _coerce_label_spec(spec: LabelSpec) -> LabelSpec:
    if not isinstance(spec, LabelSpec):
        raise ValueError("spec must be LabelSpec")
    if spec.price_reference != PriceReference.MID:
        raise ValueError("only PriceReference.MID is supported")
    if spec.asof_policy != AsOfPolicy.LAST_OBSERVATION:
        raise ValueError("only AsOfPolicy.LAST_OBSERVATION is supported")
    return spec


def _safe_log_return_bps(exit_price: float, entry_price: float) -> float:
    if not (math.isfinite(exit_price) and math.isfinite(entry_price)):
        raise ValueError("entry/exit price must be finite")
    if exit_price <= FLOAT_EPS or entry_price <= FLOAT_EPS:
        raise ValueError("entry/exit price must be > 0")
    return 10_000.0 * math.log(exit_price / entry_price)


@dataclass(frozen=True, slots=True)
class PriceObservation:
    ts_us: int
    price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts_us", _require_int_us(self.ts_us, "ts_us", allow_zero=True))
        object.__setattr__(self, "price", _require_positive_float(self.price, "price"))


@dataclass(frozen=True, slots=True)
class PendingLabel:
    decision_ts_us: int
    entry_ts_us: int
    ready_ts_us: int
    horizons_us: tuple[int, ...]

    def __post_init__(self) -> None:
        decision = _require_int_us(self.decision_ts_us, "decision_ts_us", allow_zero=True)
        entry = _require_int_us(self.entry_ts_us, "entry_ts_us", allow_zero=True)
        ready = _require_int_us(self.ready_ts_us, "ready_ts_us", allow_zero=True)
        if entry < decision:
            raise ValueError("entry_ts_us must be >= decision_ts_us")
        if ready < entry:
            raise ValueError("ready_ts_us must be >= entry_ts_us")
        horizons = tuple(sorted({
            _require_int_us(h, f"horizons_us[{idx}]", positive=True) for idx, h in enumerate(self.horizons_us)
        }))
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        object.__setattr__(self, "decision_ts_us", decision)
        object.__setattr__(self, "entry_ts_us", entry)
        object.__setattr__(self, "ready_ts_us", ready)
        object.__setattr__(self, "horizons_us", horizons)


class PriceHistory:
    def __init__(self, capacity: int = DEFAULT_PRICE_HISTORY_CAPACITY):
        self.capacity = _require_int_us(capacity, "capacity", positive=True)
        self._ts: list[int] = []
        self._price: list[float] = []
        self._start = 0

    @property
    def size(self) -> int:
        return len(self._ts) - self._start

    @property
    def latest_ts_us(self) -> int | None:
        return self._ts[-1] if self.size > 0 else None

    @property
    def latest_price(self) -> float | None:
        return self._price[-1] if self.size > 0 else None

    def append(self, obs: PriceObservation) -> None:
        if not isinstance(obs, PriceObservation):
            raise ValueError("obs must be PriceObservation")
        if self.size == 0:
            self._ts.append(obs.ts_us)
            self._price.append(obs.price)
        else:
            last_ts = self._ts[-1]
            if obs.ts_us < last_ts:
                raise ValueError("price timestamp must be nondecreasing")
            if obs.ts_us == last_ts:
                self._price[-1] = obs.price
            else:
                self._ts.append(obs.ts_us)
                self._price.append(obs.price)
        if self.size > self.capacity:
            self._start = len(self._ts) - self.capacity
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        if self._start >= COMPACT_MIN_START and self._start >= len(self._ts) * COMPACT_FRACTION:
            self._ts = self._ts[self._start :]
            self._price = self._price[self._start :]
            self._start = 0

    def asof_price(self, ts_us: int) -> float | None:
        _require_int_us(ts_us, "ts_us", allow_zero=True)
        idx = bisect.bisect_right(self._ts, ts_us, lo=self._start) - 1
        if idx < self._start:
            return None
        return self._price[idx]

    def latest_reaches(self, ts_us: int) -> bool:
        latest = self.latest_ts_us
        return latest is not None and latest >= ts_us

    def active_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._ts[self._start :], dtype=np.int64).copy(),
            np.asarray(self._price[self._start :], dtype=np.float64).copy(),
        )

    def reset(self) -> None:
        self._ts.clear()
        self._price.clear()
        self._start = 0


class LabelBuilder:
    def __init__(self, spec: LabelSpec, *, price_history_capacity: int = DEFAULT_PRICE_HISTORY_CAPACITY):
        self.spec = _coerce_label_spec(spec)
        self.price_history = PriceHistory(price_history_capacity)
        self.pending: list[PendingLabel] = []
        self._pending_start = 0
        self._last_decision_ts_us: int | None = None

    @property
    def pending_count(self) -> int:
        return len(self.pending) - self._pending_start

    @property
    def latest_price_ts_us(self) -> int | None:
        return self.price_history.latest_ts_us

    @property
    def label_context_us(self) -> int:
        return self.spec.label_context_us

    def reset(self) -> None:
        self.price_history.reset()
        self.pending.clear()
        self._pending_start = 0
        self._last_decision_ts_us = None

    def observe_price(self, ts_us: int, price: float) -> list[LabelResult]:
        self.price_history.append(PriceObservation(ts_us, price))
        return self.mature_ready()

    def on_decision(self, decision_ts_us: int) -> None:
        decision_ts_us = _require_int_us(decision_ts_us, "decision_ts_us", allow_zero=True)
        if self._last_decision_ts_us is not None and decision_ts_us < self._last_decision_ts_us:
            raise ValueError("decision_ts_us must be nondecreasing")
        self._last_decision_ts_us = decision_ts_us
        entry_ts_us = decision_ts_us + self.spec.entry_delay_us
        ready_ts_us = entry_ts_us + max(self.spec.horizons_us)
        self.pending.append(
            PendingLabel(
                decision_ts_us=decision_ts_us,
                entry_ts_us=entry_ts_us,
                ready_ts_us=ready_ts_us,
                horizons_us=self.spec.horizons_us,
            )
        )

    def mature_ready(self) -> list[LabelResult]:
        out: list[LabelResult] = []
        while self._pending_start < len(self.pending):
            pend = self.pending[self._pending_start]
            if not self.price_history.latest_reaches(pend.ready_ts_us):
                break
            entry_price = self.price_history.asof_price(pend.entry_ts_us)
            if entry_price is None:
                break
            values: list[float] = []
            complete = True
            for horizon in pend.horizons_us:
                exit_ts = pend.entry_ts_us + horizon
                exit_price = self.price_history.asof_price(exit_ts)
                if exit_price is None:
                    complete = False
                    break
                values.append(_safe_log_return_bps(exit_price, entry_price))
            if not complete:
                break
            out.append(
                LabelResult(
                    decision_ts_us=pend.decision_ts_us,
                    entry_ts_us=pend.entry_ts_us,
                    horizons_us=self.spec.horizons_us,
                    values_bps=tuple(values),
                )
            )
            self._pending_start += 1
        if self._pending_start >= COMPACT_MIN_START and self._pending_start >= len(self.pending) * COMPACT_FRACTION:
            self.pending = self.pending[self._pending_start :]
            self._pending_start = 0
        return out

    def label_now(self, decision_ts_us: int) -> LabelResult | None:
        decision_ts_us = _require_int_us(decision_ts_us, "decision_ts_us", allow_zero=True)
        entry_ts_us = decision_ts_us + self.spec.entry_delay_us
        ready_ts_us = entry_ts_us + max(self.spec.horizons_us)
        if not self.price_history.latest_reaches(ready_ts_us):
            return None
        entry_price = self.price_history.asof_price(entry_ts_us)
        if entry_price is None:
            return None
        values: list[float] = []
        for horizon in self.spec.horizons_us:
            exit_price = self.price_history.asof_price(entry_ts_us + horizon)
            if exit_price is None:
                return None
            values.append(_safe_log_return_bps(exit_price, entry_price))
        return LabelResult(decision_ts_us=decision_ts_us, entry_ts_us=entry_ts_us, horizons_us=self.spec.horizons_us, values_bps=tuple(values))

    def on_price_and_decision(self, ts_us: int, price: float, *, is_decision: bool) -> list[LabelResult]:
        out = self.observe_price(ts_us, price)
        if is_decision:
            self.on_decision(ts_us)
            out.extend(self.mature_ready())
        return out


def _dedupe_equal_timestamps_keep_last(ts: np.ndarray, price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ts.size == 0:
        return ts.astype(np.int64, copy=False), price.astype(np.float64, copy=False)
    keep = np.ones(ts.shape[0], dtype=bool)
    keep[:-1] = ts[:-1] != ts[1:]
    return ts[keep].astype(np.int64, copy=False), price[keep].astype(np.float64, copy=False)



def _coerce_timestamp_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.dtype == np.dtype(bool) or arr.dtype.kind == "b":
        raise ValueError(f"{name} must not be bool")
    if arr.dtype.kind in ("i", "u"):
        out = arr.astype(np.int64, copy=False)
    elif arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ValueError(f"{name} must be integer-valued")
        out = arr.astype(np.int64, copy=False)
    else:
        raise ValueError(f"{name} must be numeric integer timestamps")
    if out.size and np.any(out < 0):
        raise ValueError(f"{name} must be nonnegative")
    if out.size and np.any(np.diff(out) < 0):
        raise ValueError(f"{name} must be nondecreasing")
    return out

def build_labels_from_price_arrays(
    decision_ts_us: np.ndarray,
    price_ts_us: np.ndarray,
    price_values: np.ndarray,
    spec: LabelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    spec = _coerce_label_spec(spec)
    dec = _coerce_timestamp_array(decision_ts_us, "decision_ts_us")
    pts = _coerce_timestamp_array(price_ts_us, "price_ts_us")
    pval = np.asarray(price_values)
    if pval.ndim != 1:
        raise ValueError("price_values must be 1D")
    if pts.shape[0] != pval.shape[0]:
        raise ValueError("price_ts_us and price_values length mismatch")
    if pval.size and (not np.all(np.isfinite(pval)) or np.any(pval <= 0.0)):
        raise ValueError("price_values must be finite and > 0")

    pval = pval.astype(np.float64, copy=False)

    pts, pval = _dedupe_equal_timestamps_keep_last(pts, pval)
    n = dec.shape[0]
    m = len(spec.horizons_us)
    labels = np.full((n, m), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n == 0 or pts.size == 0:
        return labels, valid

    horizons = np.asarray(spec.horizons_us, dtype=np.int64)
    entry_ts = dec + int(spec.entry_delay_us)
    exit_ts = entry_ts[:, None] + horizons[None, :]

    entry_idx = np.searchsorted(pts, entry_ts, side="right") - 1
    exit_idx = np.searchsorted(pts, exit_ts, side="right") - 1
    mature = exit_ts[:, -1] <= pts[-1]
    row_valid = (entry_idx >= 0) & np.all(exit_idx >= 0, axis=1) & mature
    if not np.any(row_valid):
        return labels, valid

    idx_rows = np.where(row_valid)[0]
    ep = pval[entry_idx[idx_rows]]
    xp = pval[exit_idx[idx_rows, :]]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = 10_000.0 * np.log(xp / ep[:, None])
    finite_rows = np.isfinite(vals).all(axis=1) & np.isfinite(ep) & np.all(ep[:, None] > 0.0, axis=1) & np.all(xp > 0.0, axis=1)
    good_rows = idx_rows[finite_rows]
    labels[good_rows, :] = vals[finite_rows, :]
    valid[good_rows] = True
    return labels, valid


def label_value_names(spec: LabelSpec) -> tuple[str, ...]:
    spec = _coerce_label_spec(spec)
    return tuple(f"ret_bps_{h}us" for h in spec.horizons_us)


def label_ready_ts_us(decision_ts_us: int, spec: LabelSpec) -> int:
    decision_ts_us = _require_int_us(decision_ts_us, "decision_ts_us", allow_zero=True)
    spec = _coerce_label_spec(spec)
    return decision_ts_us + spec.entry_delay_us + max(spec.horizons_us)


def label_entry_ts_us(decision_ts_us: int, spec: LabelSpec) -> int:
    decision_ts_us = _require_int_us(decision_ts_us, "decision_ts_us", allow_zero=True)
    spec = _coerce_label_spec(spec)
    return decision_ts_us + spec.entry_delay_us


__all__ = [
    "DEFAULT_PRICE_HISTORY_CAPACITY",
    "PriceObservation",
    "PendingLabel",
    "PriceHistory",
    "LabelBuilder",
    "build_labels_from_price_arrays",
    "label_value_names",
    "label_ready_ts_us",
    "label_entry_ts_us",
]
