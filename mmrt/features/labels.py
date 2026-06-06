"""Latency-aware label construction for the MMRT feature pipeline.

This module builds fixed-horizon return labels from causal book-mid price
observations. All timestamps accepted by this module are local/causal
microsecond timestamps (`local_ts_us`), not exchange event timestamps (`ts_us`).
The shared LabelResult contract still uses generic `decision_ts_us` and
`entry_ts_us` field names, but values produced here are local-clock values.

This module does not parse market data, compute features, apply transforms,
split rows, or write storage artifacts.
"""

from dataclasses import dataclass
import bisect
import math

import numpy as np

from mmrt.contracts import AsOfPolicy, LabelResult, LabelSpec, PriceReference
from mmrt.time_key import EventKey, MAX_EVENT_SEQ, key_at_or_after_timestamp

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
    local_ts_us: int
    event_seq: int
    price: float

    @property
    def key(self) -> EventKey:
        return EventKey(self.local_ts_us, self.event_seq)

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_ts_us", _require_int_us(self.local_ts_us, "local_ts_us", positive=True))
        object.__setattr__(self, "event_seq", _require_int_us(self.event_seq, "event_seq", allow_zero=True))
        object.__setattr__(self, "price", _require_positive_float(self.price, "price"))


@dataclass(frozen=True, slots=True)
class PendingLabel:
    decision_key: EventKey
    entry_local_ts_us: int
    ready_local_ts_us: int
    horizons_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.decision_key, EventKey):
            raise ValueError("decision_key must be EventKey")
        decision = self.decision_key.local_ts_us
        entry = _require_int_us(self.entry_local_ts_us, "entry_local_ts_us", positive=True)
        ready = _require_int_us(self.ready_local_ts_us, "ready_local_ts_us", positive=True)
        if entry < decision:
            raise ValueError("entry_local_ts_us must be >= decision_local_ts_us")
        if ready < entry:
            raise ValueError("ready_local_ts_us must be >= entry_local_ts_us")
        horizons = tuple(sorted({
            _require_int_us(h, f"horizons_us[{idx}]", positive=True) for idx, h in enumerate(self.horizons_us)
        }))
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        object.__setattr__(self, "entry_local_ts_us", entry)
        object.__setattr__(self, "ready_local_ts_us", ready)
        object.__setattr__(self, "horizons_us", horizons)


class PriceHistory:
    def __init__(self, capacity: int = DEFAULT_PRICE_HISTORY_CAPACITY):
        self.capacity = _require_int_us(capacity, "capacity", positive=True)
        self._keys: list[EventKey] = []
        self._price: list[float] = []
        self._start = 0

    @property
    def size(self) -> int:
        return len(self._keys) - self._start

    @property
    def latest_local_ts_us(self) -> int | None:
        return self._keys[-1].local_ts_us if self.size > 0 else None

    @property
    def latest_price(self) -> float | None:
        return self._price[-1] if self.size > 0 else None

    def append(self, obs: PriceObservation) -> None:
        if not isinstance(obs, PriceObservation):
            raise ValueError("obs must be PriceObservation")
        if self.size == 0:
            self._keys.append(obs.key)
            self._price.append(obs.price)
        else:
            last_key = self._keys[-1]
            if obs.key < last_key:
                raise ValueError("price EventKey must be nondecreasing")
            if obs.key == last_key:
                raise ValueError("duplicate price EventKey")
            self._keys.append(obs.key)
            self._price.append(obs.price)
        if self.size > self.capacity:
            self._start = len(self._keys) - self.capacity
        self._maybe_compact()

    def _maybe_compact(self) -> None:
        if self._start >= COMPACT_MIN_START and self._start >= len(self._keys) * COMPACT_FRACTION:
            self._keys = self._keys[self._start :]
            self._price = self._price[self._start :]
            self._start = 0

    def asof_price(self, key: EventKey) -> float | None:
        if not isinstance(key, EventKey):
            raise ValueError("key must be EventKey")
        idx = bisect.bisect_right(self._keys, key, lo=self._start) - 1
        if idx < self._start:
            return None
        return self._price[idx]

    def latest_reaches(self, key: EventKey) -> bool:
        if not isinstance(key, EventKey):
            raise ValueError("key must be EventKey")
        return self.size > 0 and self._keys[-1] >= key

    def active_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = self._keys[self._start :]
        return (
            np.asarray([k.local_ts_us for k in keys], dtype=np.int64),
            np.asarray([k.event_seq for k in keys], dtype=np.int64),
            np.asarray(self._price[self._start :], dtype=np.float64).copy(),
        )

    def reset(self) -> None:
        self._keys.clear()
        self._price.clear()
        self._start = 0


class LabelBuilder:
    def __init__(self, spec: LabelSpec, *, price_history_capacity: int = DEFAULT_PRICE_HISTORY_CAPACITY):
        self.spec = _coerce_label_spec(spec)
        self.price_history = PriceHistory(price_history_capacity)
        self.pending: list[PendingLabel] = []
        self._pending_start = 0
        self._last_decision_key: EventKey | None = None

    @property
    def pending_count(self) -> int:
        return len(self.pending) - self._pending_start

    @property
    def latest_price_local_ts_us(self) -> int | None:
        return self.price_history.latest_local_ts_us

    @property
    def label_context_us(self) -> int:
        return self.spec.label_context_us

    def reset(self) -> None:
        self.price_history.reset()
        self.pending.clear()
        self._pending_start = 0
        self._last_decision_key = None

    def observe_price_local(self, local_ts_us: int, event_seq: int, price: float) -> list[LabelResult]:
        self.price_history.append(PriceObservation(local_ts_us, event_seq, price))
        return self.mature_ready()

    def on_decision_local(self, decision_local_ts_us: int, event_seq: int) -> None:
        decision_key = EventKey(decision_local_ts_us, event_seq)
        if self._last_decision_key is not None and decision_key < self._last_decision_key:
            raise ValueError("decision EventKey must be nondecreasing")
        self._last_decision_key = decision_key
        entry_local_ts_us = decision_local_ts_us + self.spec.entry_delay_us
        ready_local_ts_us = entry_local_ts_us + max(self.spec.horizons_us)
        self.pending.append(
            PendingLabel(
                decision_key=decision_key,
                entry_local_ts_us=entry_local_ts_us,
                ready_local_ts_us=ready_local_ts_us,
                horizons_us=self.spec.horizons_us,
            )
        )

    def _mature_ready(self, *, require_timestamp_advanced: bool) -> list[LabelResult]:
        out: list[LabelResult] = []
        while self._pending_start < len(self.pending):
            pend = self.pending[self._pending_start]
            latest_ts = self.price_history.latest_local_ts_us
            if latest_ts is None:
                break
            if require_timestamp_advanced:
                if latest_ts <= pend.ready_local_ts_us:
                    break
            elif latest_ts < pend.ready_local_ts_us:
                break
            entry_price = self.price_history.asof_price(pend.decision_key if self.spec.entry_delay_us == 0 else key_at_or_after_timestamp(pend.entry_local_ts_us))
            if entry_price is None:
                break
            values: list[float] = []
            complete = True
            for horizon in pend.horizons_us:
                exit_local_ts_us = pend.entry_local_ts_us + horizon
                exit_price = self.price_history.asof_price(key_at_or_after_timestamp(exit_local_ts_us))
                if exit_price is None:
                    complete = False
                    break
                values.append(_safe_log_return_bps(exit_price, entry_price))
            if not complete:
                break
            out.append(
                # LabelResult uses generic ts field names; values here are local-clock timestamps.
                LabelResult(
                    decision_ts_us=pend.decision_key.local_ts_us,
                    decision_event_seq=pend.decision_key.event_seq,
                    entry_ts_us=pend.entry_local_ts_us,
                    horizons_us=self.spec.horizons_us,
                    values_bps=tuple(values),
                )
            )
            self._pending_start += 1
        if self._pending_start >= COMPACT_MIN_START and self._pending_start >= len(self.pending) * COMPACT_FRACTION:
            self.pending = self.pending[self._pending_start :]
            self._pending_start = 0
        return out

    def mature_ready(self) -> list[LabelResult]:
        return self._mature_ready(require_timestamp_advanced=True)

    def finalize_at_eof(self) -> list[LabelResult]:
        return self._mature_ready(require_timestamp_advanced=False)

    def label_now_local(self, decision_local_ts_us: int, event_seq: int) -> LabelResult | None:
        decision_key = EventKey(decision_local_ts_us, event_seq)
        entry_local_ts_us = decision_local_ts_us + self.spec.entry_delay_us
        ready_local_ts_us = entry_local_ts_us + max(self.spec.horizons_us)
        if self.price_history.latest_local_ts_us is None or self.price_history.latest_local_ts_us <= ready_local_ts_us:
            return None
        entry_price = self.price_history.asof_price(decision_key if self.spec.entry_delay_us == 0 else key_at_or_after_timestamp(entry_local_ts_us))
        if entry_price is None:
            return None
        values: list[float] = []
        for horizon in self.spec.horizons_us:
            exit_price = self.price_history.asof_price(key_at_or_after_timestamp(entry_local_ts_us + horizon))
            if exit_price is None:
                return None
            values.append(_safe_log_return_bps(exit_price, entry_price))
        return LabelResult(
            decision_ts_us=decision_local_ts_us,
            decision_event_seq=event_seq,
            entry_ts_us=entry_local_ts_us,
            horizons_us=self.spec.horizons_us,
            values_bps=tuple(values),
        )

    def on_price_and_decision_local(self, local_ts_us: int, event_seq: int, price: float, *, is_decision: bool) -> list[LabelResult]:
        out = self.observe_price_local(local_ts_us, event_seq, price)
        if is_decision:
            self.on_decision_local(local_ts_us, event_seq)
            out.extend(self.mature_ready())
        return out


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
    return out

def _coerce_event_seq_array(values: np.ndarray, name: str) -> np.ndarray:
    return _coerce_timestamp_array(values, name)


def _validate_key_order(local_ts: np.ndarray, event_seq: np.ndarray, name: str) -> None:
    if local_ts.shape != event_seq.shape:
        raise ValueError(f"{name} timestamp/event_seq length mismatch")
    if local_ts.size < 2:
        return
    ts_decrease = local_ts[1:] < local_ts[:-1]
    seq_decrease = (local_ts[1:] == local_ts[:-1]) & (event_seq[1:] <= event_seq[:-1])
    if np.any(ts_decrease | seq_decrease):
        raise ValueError(f"{name} keys must be strictly increasing by (local_ts_us, event_seq)")


def _composite_keys(local_ts: np.ndarray, event_seq: np.ndarray) -> list[int]:
    scale = MAX_EVENT_SEQ + 1
    return [int(ts) * scale + int(seq) for ts, seq in zip(local_ts, event_seq, strict=True)]


def build_labels_from_price_event_arrays(
    decision_local_ts_us: np.ndarray,
    decision_event_seq: np.ndarray,
    price_local_ts_us: np.ndarray,
    price_event_seq: np.ndarray,
    price_values: np.ndarray,
    spec: LabelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    spec = _coerce_label_spec(spec)
    dec = _coerce_timestamp_array(decision_local_ts_us, "decision_local_ts_us")
    dseq = _coerce_event_seq_array(decision_event_seq, "decision_event_seq")
    pts = _coerce_timestamp_array(price_local_ts_us, "price_local_ts_us")
    pseq = _coerce_event_seq_array(price_event_seq, "price_event_seq")
    _validate_key_order(dec, dseq, "decision")
    _validate_key_order(pts, pseq, "price")
    pval = np.asarray(price_values)
    if pval.ndim != 1:
        raise ValueError("price_values must be 1D")
    if pts.shape[0] != pval.shape[0]:
        raise ValueError("price_local_ts_us and price_values length mismatch")
    if pval.size and (not np.all(np.isfinite(pval)) or np.any(pval <= 0.0)):
        raise ValueError("price_values must be finite and > 0")

    pval = pval.astype(np.float64, copy=False)
    n = dec.shape[0]
    m = len(spec.horizons_us)
    labels = np.full((n, m), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    if n == 0 or pts.size == 0:
        return labels, valid

    price_keys = _composite_keys(pts, pseq)
    scale = MAX_EVENT_SEQ + 1
    for i in range(n):
        entry_ts = int(dec[i]) + int(spec.entry_delay_us)
        if spec.entry_delay_us == 0:
            entry_key = int(dec[i]) * scale + int(dseq[i])
        else:
            entry_key = entry_ts * scale + MAX_EVENT_SEQ
        entry_idx = bisect.bisect_right(price_keys, entry_key) - 1
        if entry_idx < 0:
            continue
        entry_price = pval[entry_idx]
        vals: list[float] = []
        ok = True
        for horizon in spec.horizons_us:
            exit_key = (entry_ts + int(horizon)) * scale + MAX_EVENT_SEQ
            exit_idx = bisect.bisect_right(price_keys, exit_key) - 1
            if exit_idx < 0 or int(pts[-1]) < entry_ts + int(horizon):
                ok = False
                break
            vals.append(_safe_log_return_bps(float(pval[exit_idx]), float(entry_price)))
        if ok:
            labels[i, :] = vals
            valid[i] = True
    return labels, valid


def label_value_names(spec: LabelSpec) -> tuple[str, ...]:
    spec = _coerce_label_spec(spec)
    return tuple(f"ret_bps_{h}us" for h in spec.horizons_us)


def label_ready_local_ts_us(decision_local_ts_us: int, spec: LabelSpec) -> int:
    decision_local_ts_us = _require_int_us(decision_local_ts_us, "decision_local_ts_us", allow_zero=True)
    spec = _coerce_label_spec(spec)
    return decision_local_ts_us + spec.entry_delay_us + max(spec.horizons_us)


def label_entry_local_ts_us(decision_local_ts_us: int, spec: LabelSpec) -> int:
    decision_local_ts_us = _require_int_us(decision_local_ts_us, "decision_local_ts_us", allow_zero=True)
    spec = _coerce_label_spec(spec)
    return decision_local_ts_us + spec.entry_delay_us


__all__ = [
    "DEFAULT_PRICE_HISTORY_CAPACITY",
    "PriceObservation",
    "PendingLabel",
    "PriceHistory",
    "LabelBuilder",
    "build_labels_from_price_event_arrays",
    "label_value_names",
    "label_ready_local_ts_us",
    "label_entry_local_ts_us",
]
