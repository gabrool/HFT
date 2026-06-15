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
from mmrt.execution.decision_grid import DecisionGrid, DecisionGridValidationMode, validate_decision_grid_for_execution_tape
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

    ``mid`` is the pipeline book mid after applying this event. ``decision``
    is populated only when this event is the current decision-grid row.
    """

    event_index: int
    local_ts_us: int
    ts_us: int
    event_seq: int
    mid: float
    decision: TransformedDecision | None


_REPLAY_CHUNK_EVENTS = 131_072


@dataclass(frozen=True, slots=True)
class DecisionFeatureChunk:
    decision_event_index: np.ndarray
    decision_local_ts_us: np.ndarray
    decision_event_seq: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]


def iter_tape_feature_steps_for_decision_grid(
    tape: ExecutionTape,
    *,
    decision_grid: DecisionGrid,
    pipeline: DecisionFeaturePipeline,
) -> Iterator[TapeFeatureStep]:
    """Replay tape events and emit decisions only at explicit grid rows."""

    if not isinstance(tape, ExecutionTape):
        raise ValueError("tape must be ExecutionTape")
    if not isinstance(decision_grid, DecisionGrid):
        raise ValueError("decision_grid must be DecisionGrid")
    if not isinstance(pipeline, DecisionFeaturePipeline):
        raise ValueError("pipeline must be DecisionFeaturePipeline")
    validate_decision_grid_for_execution_tape(decision_grid, tape, mode=DecisionGridValidationMode.SHAPE_ONLY)

    events = tape.arrays.events
    tick_size = float(tape.manifest.symbol_spec.tick_size)
    l2_events = tape.arrays.l2_events
    best_bid_ticks = l2_events["best_bid_tick"]
    best_ask_ticks = l2_events["best_ask_tick"]
    book_bid_ticks = tape.arrays.book_bid_ticks
    book_bid_sizes = tape.arrays.book_bid_sizes
    book_ask_ticks = tape.arrays.book_ask_ticks
    book_ask_sizes = tape.arrays.book_ask_sizes
    book_depth = int(book_bid_ticks.shape[1])
    if book_depth > BOOK_DEPTH:
        raise ValueError("tape book depth exceeds feature BOOK_DEPTH")
    trades = tape.arrays.trades
    trade_price_tick = trades["price_tick"]
    trade_amount = trades["amount"]
    trade_side = trades["side_code"]

    bid_px = np.zeros(BOOK_DEPTH, dtype=np.float64)
    bid_sz = np.zeros(BOOK_DEPTH, dtype=np.float64)
    ask_px = np.zeros(BOOK_DEPTH, dtype=np.float64)
    ask_sz = np.zeros(BOOK_DEPTH, dtype=np.float64)

    grid_pos = 0
    next_grid_event_index = int(decision_grid.decision_event_index[grid_pos])
    last_grid_event_index = int(decision_grid.decision_event_index[-1])
    on_trade = pipeline.on_trade
    on_book = pipeline.on_book_snapshot_without_decision
    for chunk_start in range(0, last_grid_event_index + 1, _REPLAY_CHUNK_EVENTS):
        chunk_end = min(chunk_start + _REPLAY_CHUNK_EVENTS, last_grid_event_index + 1)
        chunk = events[chunk_start:chunk_end]
        local_ts = chunk["local_ts_us"]
        ts = chunk["ts_us"]
        seq = chunk["event_seq"]
        codes = chunk["event_type_code"]
        book_ptrs = chunk["book_ptr"]
        trade_ptrs = chunk["trade_ptr"]
        for i in range(chunk_end - chunk_start):
            event_index = chunk_start + i
            if event_index > next_grid_event_index:
                raise RuntimeError("decision grid row was not reached during tape replay")
            code = int(codes[i])
            if code == EVENT_TYPE_CODE_TRADE:
                trade_ptr = int(trade_ptrs[i])
                if trade_ptr < 0:
                    raise ValueError("event row does not reference a trade")
                on_trade(TradeInput(
                    local_ts_us=int(local_ts[i]),
                    ts_us=int(ts[i]),
                    price=float(int(trade_price_tick[trade_ptr]) * tick_size),
                    amount=float(trade_amount[trade_ptr]),
                    side_code=int(trade_side[trade_ptr]),
                    event_seq=int(seq[i]),
                ))
                continue
            if code != EVENT_TYPE_CODE_L2_BATCH:
                raise ValueError(f"unsupported execution event_type_code: {code}")
            book_ptr = int(book_ptrs[i])
            if book_ptr < 0:
                continue
            bid_tick = int(best_bid_ticks[book_ptr])
            if bid_tick <= 0 or int(best_ask_ticks[book_ptr]) <= bid_tick:
                continue
            np.multiply(book_bid_ticks[book_ptr], tick_size, out=bid_px[:book_depth])
            np.multiply(book_ask_ticks[book_ptr], tick_size, out=ask_px[:book_depth])
            bid_sz[:book_depth] = book_bid_sizes[book_ptr]
            ask_sz[:book_depth] = book_ask_sizes[book_ptr]
            snapshot = BookSnapshotInput.from_trusted_arrays(
                local_ts_us=int(local_ts[i]),
                ts_us=int(ts[i]),
                event_seq=int(seq[i]),
                bid_px=bid_px,
                bid_sz=bid_sz,
                ask_px=ask_px,
                ask_sz=ask_sz,
            )
            on_book(snapshot)
            decision = None
            if event_index == next_grid_event_index:
                if int(decision_grid.book_ptr[grid_pos]) != book_ptr:
                    raise ValueError("decision grid book_ptr mismatch during feature replay")
                decision = pipeline.force_decision(local_ts_us=int(local_ts[i]), ts_us=int(ts[i]), event_seq=int(seq[i]))
                if int(decision.event_seq) != int(decision_grid.decision_event_seq[grid_pos]):
                    raise ValueError("decision grid event_seq mismatch during feature replay")
                grid_pos += 1
                if grid_pos < decision_grid.n_rows:
                    next_grid_event_index = int(decision_grid.decision_event_index[grid_pos])
            yield TapeFeatureStep(
                event_index=event_index,
                local_ts_us=int(local_ts[i]),
                ts_us=int(ts[i]),
                event_seq=int(seq[i]),
                mid=pipeline.current_mid(),
                decision=decision,
            )
    if grid_pos != decision_grid.n_rows:
        raise RuntimeError("not all decision grid rows were replayed")


def iter_decision_feature_chunks_for_decision_grid(
    tape: ExecutionTape,
    *,
    decision_grid: DecisionGrid,
    pipeline_config: FeaturePipelineConfig,
    chunk_rows: int = 100_000,
    output_dtype: str = "float32",
) -> Iterator[DecisionFeatureChunk]:
    """Yield transformed decision features exactly aligned to decision_grid."""

    if not isinstance(pipeline_config, FeaturePipelineConfig):
        raise ValueError("pipeline_config must be FeaturePipelineConfig")
    chunk_rows = _require_positive_int(chunk_rows, "chunk_rows")
    dtype = _require_output_dtype(output_dtype)
    pipeline = DecisionFeaturePipeline(pipeline_config)
    names = decision_feature_column_names()
    idx_buf = np.empty(chunk_rows, dtype=np.int64)
    ts_buf = np.empty(chunk_rows, dtype=np.int64)
    seq_buf = np.empty(chunk_rows, dtype=np.int64)
    feat_buf = np.empty((chunk_rows, len(names)), dtype=dtype)
    used = 0
    for step in iter_tape_feature_steps_for_decision_grid(tape, decision_grid=decision_grid, pipeline=pipeline):
        if step.decision is None:
            continue
        idx_buf[used] = step.event_index
        ts_buf[used] = step.decision.local_ts_us
        seq_buf[used] = step.decision.event_seq
        feat_buf[used, :] = np.asarray(step.decision.feature_values, dtype=dtype)
        used += 1
        if used >= chunk_rows:
            yield DecisionFeatureChunk(
                idx_buf.copy(),
                ts_buf.copy(),
                seq_buf.copy(),
                feat_buf.copy(),
                names,
            )
            used = 0
    if used:
        yield DecisionFeatureChunk(
            idx_buf[:used].copy(),
            ts_buf[:used].copy(),
            seq_buf[:used].copy(),
            feat_buf[:used].copy(),
            names,
        )


__all__ = [
    "decision_feature_column_names",
    "book_snapshot_input_from_tape_row",
    "trade_input_from_tape_row",
    "TapeFeatureStep",
    "iter_tape_feature_steps_for_decision_grid",
    "DecisionFeatureChunk",
    "iter_decision_feature_chunks_for_decision_grid",
]
