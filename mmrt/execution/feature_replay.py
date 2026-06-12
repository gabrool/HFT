"""Single execution-tape replay into the decision feature pipeline.

This module is the only bridge from :class:`ExecutionTape` events into
:class:`DecisionFeaturePipeline`. Supervised dataset ingest, linear signal
building, and feature audits all consume it, so training and serving features
come from one code path by construction.

It does not build labels, fit or apply models, or write artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from mmrt.execution.execution_tape import (
    EVENT_TYPE_CODE_L2_BATCH,
    EVENT_TYPE_CODE_TRADE,
    ExecutionTape,
)
from mmrt.features.book_state import BOOK_DEPTH, BookSnapshotInput
from mmrt.features.pipeline import (
    DecisionFeaturePipeline,
    FeaturePipelineConfig,
    TransformedDecision,
)
from mmrt.features.specs import FEATURE_NAMES
from mmrt.features.trade_state import TradeInput

_ALLOWED_DTYPES = ("float32", "float64")


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _require_output_dtype(value: str) -> np.dtype:
    if value not in _ALLOWED_DTYPES:
        raise ValueError(f"output_dtype must be one of {_ALLOWED_DTYPES}")
    return np.dtype(value)


def decision_feature_column_names() -> tuple[str, ...]:
    """Feature column names of pipeline output, matching storage feature columns."""
    return tuple(f"x_{name}" for name in FEATURE_NAMES)


def book_snapshot_input_from_tape_row(tape: ExecutionTape, *, event_row: np.void) -> BookSnapshotInput:
    book_ptr = int(event_row["book_ptr"])
    if book_ptr < 0:
        raise ValueError("event row does not reference a book snapshot")
    tick_size = float(tape.manifest.symbol_spec.tick_size)

    def padded(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        out = np.zeros(BOOK_DEPTH, dtype=dtype)
        src = np.asarray(values, dtype=dtype)[:BOOK_DEPTH]
        out[: src.shape[0]] = src
        return out

    bid_ticks = padded(tape.arrays.book_bid_ticks[book_ptr], dtype=np.dtype("float64"))
    bid_sizes = padded(tape.arrays.book_bid_sizes[book_ptr], dtype=np.dtype("float64"))
    ask_ticks = padded(tape.arrays.book_ask_ticks[book_ptr], dtype=np.dtype("float64"))
    ask_sizes = padded(tape.arrays.book_ask_sizes[book_ptr], dtype=np.dtype("float64"))
    return BookSnapshotInput(
        local_ts_us=int(event_row["local_ts_us"]),
        ts_us=int(event_row["ts_us"]),
        event_seq=int(event_row["event_seq"]),
        bid_px=bid_ticks * tick_size,
        bid_sz=bid_sizes,
        ask_px=ask_ticks * tick_size,
        ask_sz=ask_sizes,
    )


def trade_input_from_tape_row(tape: ExecutionTape, *, event_row: np.void) -> TradeInput:
    trade_ptr = int(event_row["trade_ptr"])
    if trade_ptr < 0:
        raise ValueError("event row does not reference a trade")
    trade = tape.arrays.trades[trade_ptr]
    price = int(trade["price_tick"]) * float(tape.manifest.symbol_spec.tick_size)
    return TradeInput(
        local_ts_us=int(event_row["local_ts_us"]),
        ts_us=int(event_row["ts_us"]),
        price=float(price),
        amount=float(trade["amount"]),
        side_code=int(trade["side_code"]),
        event_seq=int(event_row["event_seq"]),
    )


def _l2_event_is_two_sided(tape: ExecutionTape, book_ptr: int) -> bool:
    row = tape.arrays.l2_events[book_ptr]
    best_bid_tick = int(row["best_bid_tick"])
    best_ask_tick = int(row["best_ask_tick"])
    return best_bid_tick > 0 and best_ask_tick > best_bid_tick


@dataclass(frozen=True, slots=True)
class TapeFeatureStep:
    """One valid two-sided L2 batch event replayed through the pipeline.

    ``mid`` is the pipeline book mid after applying this event; ``decision``
    is the transformed decision feature row when this event crossed the
    decision stride, else ``None``.
    """

    event_index: int
    local_ts_us: int
    ts_us: int
    event_seq: int
    mid: float
    decision: TransformedDecision | None


_REPLAY_CHUNK_EVENTS = 131_072


def iter_tape_feature_steps(
    tape: ExecutionTape,
    *,
    pipeline: DecisionFeaturePipeline,
    start_event_index: int = 0,
    max_events: int | None = None,
) -> Iterator[TapeFeatureStep]:
    """Replay tape events through the pipeline, yielding valid L2 steps.

    Trades are fed to the pipeline but not yielded. L2 batch events with a
    one-sided or empty book are skipped entirely (no pipeline update), the
    same validity rule the execution environment applies.

    Event columns are decoded in fixed-size chunks and book inputs are built
    through the trusted constructor: tape rows were validated at build time,
    so per-event re-validation is skipped here.
    """
    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(pipeline, DecisionFeaturePipeline):
        raise ValueError("pipeline must be DecisionFeaturePipeline")
    start = _require_nonnegative_int(start_event_index, "start_event_index")
    events = tape.arrays.events
    if start >= len(events):
        raise ValueError("start_event_index must be < len(tape.arrays.events)")
    if max_events is not None:
        max_events = _require_positive_int(max_events, "max_events")
    end = len(events) if max_events is None else min(len(events), start + max_events)

    tick_size = float(tape.manifest.symbol_spec.tick_size)
    l2_events = tape.arrays.l2_events
    best_bid_ticks = np.ascontiguousarray(l2_events["best_bid_tick"])
    best_ask_ticks = np.ascontiguousarray(l2_events["best_ask_tick"])
    book_bid_ticks = tape.arrays.book_bid_ticks
    book_bid_sizes = tape.arrays.book_bid_sizes
    book_ask_ticks = tape.arrays.book_ask_ticks
    book_ask_sizes = tape.arrays.book_ask_sizes
    book_depth = int(book_bid_ticks.shape[1])
    if book_depth > BOOK_DEPTH:
        raise ValueError("tape book depth exceeds feature BOOK_DEPTH")
    trades = tape.arrays.trades
    trade_price = (np.asarray(trades["price_tick"], dtype=np.float64) * tick_size).tolist()
    trade_amount = np.asarray(trades["amount"], dtype=np.float64).tolist()
    trade_side = np.asarray(trades["side_code"], dtype=np.int64).tolist()

    # Reused book input buffers: apply_snapshot copies them immediately and
    # nothing reads the snapshot object afterwards, so per-event allocation
    # and re-validation are avoided.
    bid_px = np.zeros(BOOK_DEPTH, dtype=np.float64)
    bid_sz = np.zeros(BOOK_DEPTH, dtype=np.float64)
    ask_px = np.zeros(BOOK_DEPTH, dtype=np.float64)
    ask_sz = np.zeros(BOOK_DEPTH, dtype=np.float64)

    on_trade = pipeline.on_trade
    on_book = pipeline.on_book_snapshot
    for chunk_start in range(start, end, _REPLAY_CHUNK_EVENTS):
        chunk_end = min(chunk_start + _REPLAY_CHUNK_EVENTS, end)
        chunk = events[chunk_start:chunk_end]
        local_ts = chunk["local_ts_us"].tolist()
        ts = chunk["ts_us"].tolist()
        seq = chunk["event_seq"].tolist()
        codes = chunk["event_type_code"].tolist()
        book_ptrs = chunk["book_ptr"].tolist()
        trade_ptrs = chunk["trade_ptr"].tolist()
        for i in range(chunk_end - chunk_start):
            code = codes[i]
            if code == EVENT_TYPE_CODE_TRADE:
                trade_ptr = trade_ptrs[i]
                if trade_ptr < 0:
                    raise ValueError("event row does not reference a trade")
                on_trade(TradeInput(
                    local_ts_us=local_ts[i],
                    ts_us=ts[i],
                    price=trade_price[trade_ptr],
                    amount=trade_amount[trade_ptr],
                    side_code=trade_side[trade_ptr],
                    event_seq=seq[i],
                ))
                continue
            if code != EVENT_TYPE_CODE_L2_BATCH:
                raise ValueError(f"unsupported execution event_type_code: {code}")
            book_ptr = book_ptrs[i]
            if book_ptr < 0:
                continue
            bid_tick = int(best_bid_ticks[book_ptr])
            if bid_tick <= 0 or int(best_ask_ticks[book_ptr]) <= bid_tick:
                continue
            np.multiply(book_bid_ticks[book_ptr], tick_size, out=bid_px[:book_depth])
            np.multiply(book_ask_ticks[book_ptr], tick_size, out=ask_px[:book_depth])
            bid_sz[:book_depth] = book_bid_sizes[book_ptr]
            ask_sz[:book_depth] = book_ask_sizes[book_ptr]
            decision = on_book(BookSnapshotInput.from_trusted_arrays(
                local_ts_us=local_ts[i],
                ts_us=ts[i],
                event_seq=seq[i],
                bid_px=bid_px,
                bid_sz=bid_sz,
                ask_px=ask_px,
                ask_sz=ask_sz,
            ))
            yield TapeFeatureStep(
                event_index=chunk_start + i,
                local_ts_us=local_ts[i],
                ts_us=ts[i],
                event_seq=seq[i],
                mid=pipeline.current_mid(),
                decision=decision,
            )


@dataclass(frozen=True, slots=True)
class DecisionFeatureChunk:
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]


def iter_decision_feature_chunks(
    tape: ExecutionTape,
    *,
    pipeline_config: FeaturePipelineConfig,
    start_event_index: int | None = None,
    max_decisions: int | None = None,
    chunk_rows: int = 100_000,
    output_dtype: str = "float32",
) -> Iterator[DecisionFeatureChunk]:
    """Yield chunked transformed decision feature rows aligned to tape events."""
    if not isinstance(pipeline_config, FeaturePipelineConfig):
        raise ValueError("pipeline_config must be FeaturePipelineConfig")
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    dtype = _require_output_dtype(output_dtype)
    if max_decisions is not None:
        max_decisions = _require_positive_int(max_decisions, "max_decisions")
    start = 0 if start_event_index is None else _require_nonnegative_int(start_event_index, "start_event_index")

    pipeline = DecisionFeaturePipeline(pipeline_config)
    names = decision_feature_column_names()
    idx_buf: list[int] = []
    ts_buf: list[int] = []
    feat_buf: list[np.ndarray] = []
    emitted = 0
    for step in iter_tape_feature_steps(tape, pipeline=pipeline, start_event_index=start):
        if step.decision is None:
            continue
        idx_buf.append(step.event_index)
        ts_buf.append(step.decision.local_ts_us)
        feat_buf.append(np.asarray(step.decision.feature_values, dtype=dtype))
        emitted += 1
        if len(feat_buf) >= chunk_rows:
            yield DecisionFeatureChunk(
                np.asarray(idx_buf, dtype=np.int64),
                np.asarray(ts_buf, dtype=np.int64),
                np.ascontiguousarray(np.vstack(feat_buf), dtype=dtype),
                names,
            )
            idx_buf.clear(); ts_buf.clear(); feat_buf.clear()
        if max_decisions is not None and emitted >= max_decisions:
            break
    if feat_buf:
        yield DecisionFeatureChunk(
            np.asarray(idx_buf, dtype=np.int64),
            np.asarray(ts_buf, dtype=np.int64),
            np.ascontiguousarray(np.vstack(feat_buf), dtype=dtype),
            names,
        )


__all__ = [
    "decision_feature_column_names",
    "book_snapshot_input_from_tape_row",
    "trade_input_from_tape_row",
    "TapeFeatureStep",
    "iter_tape_feature_steps",
    "DecisionFeatureChunk",
    "iter_decision_feature_chunks",
]
