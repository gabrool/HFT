from __future__ import annotations

from typing import Mapping

import numpy as np

from mmrt.execution.decision_grid import (
    DECISION_GRID_ARRAY_ORDER,
    DECISION_GRID_SCHEMA,
    DecisionGrid,
    decision_grid_metadata_from_tape,
)
from mmrt.execution.adverse_selection_dataset import ADVERSE_SPLIT_CONTRACT_SCHEMA
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, EVENT_TYPE_CODE_TRADE, ExecutionTape
from mmrt.features.schedule import (
    DECISION_REASON_FIRST_VALID_BOOK,
    DECISION_REASON_FLAG_FIRST_VALID_BOOK,
    DECISION_REASON_FLAG_HEARTBEAT,
    DECISION_REASON_HEARTBEAT,
    DecisionScheduleConfig,
)

GRID_HASH = "1" * 64
MANIFEST_HASH = "2" * 64


def grid_lineage_fields(
    *,
    n_rows: int = 1,
    schedule: Mapping[str, object] | None = None,
    grid_hash: str = GRID_HASH,
) -> dict[str, object]:
    return {
        "decision_grid_schema": DECISION_GRID_SCHEMA,
        "decision_grid_hash": grid_hash,
        "decision_grid_n_rows": int(n_rows),
        "decision_schedule": dict(schedule or DecisionScheduleConfig().as_dict()),
    }


def grid_identity_fields(
    *,
    n_rows: int = 1,
    grid_hash: str = GRID_HASH,
) -> dict[str, object]:
    fields = grid_lineage_fields(n_rows=n_rows, grid_hash=grid_hash)
    fields.pop("decision_schedule")
    return fields


def grid_lineage_notes(
    *,
    n_rows: int = 1,
    schedule: Mapping[str, object] | None = None,
    grid_hash: str = GRID_HASH,
) -> dict[str, object]:
    return {"decision_grid": grid_lineage_fields(n_rows=n_rows, schedule=schedule, grid_hash=grid_hash)}


def adverse_split_contract_fields(
    *,
    n_rows: int = 3,
    schedule: Mapping[str, object] | None = None,
    grid_hash: str = GRID_HASH,
    root: str = "/tmp/split_source",
    dataset_id: str = "split-source",
    manifest_hash: str = MANIFEST_HASH,
    ranges: Mapping[str, list[Mapping[str, object]]] | None = None,
) -> dict[str, object]:
    if ranges is None:
        ranges = {
            "train": [{"segment_key": "seg_000", "start_row": 0, "end_row": 1, "row_count": 1, "start_local_ts_us": 1, "end_local_ts_us": 2, "embargo_before_us": 0, "embargo_after_us": 0}],
            "val": [{"segment_key": "seg_000", "start_row": 1, "end_row": 2, "row_count": 1, "start_local_ts_us": 2, "end_local_ts_us": 3, "embargo_before_us": 0, "embargo_after_us": 0}],
            "test": [{"segment_key": "seg_000", "start_row": 2, "end_row": 3, "row_count": 1, "start_local_ts_us": 3, "end_local_ts_us": 4, "embargo_before_us": 0, "embargo_after_us": 0}],
        }
    source_row_counts = {
        role: int(sum(int(entry["row_count"]) for entry in entries))
        for role, entries in ranges.items()
    }
    contract = {
        "schema": ADVERSE_SPLIT_CONTRACT_SCHEMA,
        "version": 1,
        "split_source_dataset_root": root,
        "split_source_dataset_id": dataset_id,
        "split_source_manifest_hash": manifest_hash,
        "ranges": {role: [dict(entry) for entry in entries] for role, entries in ranges.items()},
        "source_row_counts": source_row_counts,
        "adverse_dataset_rows_total": 0,
        "adverse_row_counts": {"train": 0, "val": 0, "test": 0, "out_of_split": 0},
        **grid_lineage_fields(n_rows=n_rows, schedule=schedule, grid_hash=grid_hash),
    }
    return {
        "split_source_dataset_root": root,
        "split_source_dataset_id": dataset_id,
        "split_source_manifest_hash": manifest_hash,
        "split_contract": contract,
    }


def adverse_split_contract_for_grid(grid: DecisionGrid, *, root: str = "/tmp/split_source") -> dict[str, object]:
    ts = [int(x) for x in grid.decision_local_ts_us]
    first = ts[0]
    ranges = {
        "train": [{"segment_key": "seg_000", "start_row": 0, "end_row": 1, "row_count": 1, "start_local_ts_us": first, "end_local_ts_us": first + 1, "embargo_before_us": 0, "embargo_after_us": 0}],
        "val": [{"segment_key": "seg_000", "start_row": 1, "end_row": 2, "row_count": 1, "start_local_ts_us": first + 1, "end_local_ts_us": first + 2, "embargo_before_us": 0, "embargo_after_us": 0}],
        "test": [{"segment_key": "seg_000", "start_row": 2, "end_row": 3, "row_count": 1, "start_local_ts_us": first + 2, "end_local_ts_us": first + 3, "embargo_before_us": 0, "embargo_after_us": 0}],
    }
    return adverse_split_contract_fields(
        n_rows=grid.n_rows,
        schedule=grid.decision_schedule,
        grid_hash=grid.decision_grid_hash,
        root=root,
        ranges=ranges,
    )


def decision_grid_for_tape(
    tape: ExecutionTape,
    *,
    max_rows: int | None = None,
    schedule_config: DecisionScheduleConfig | None = None,
) -> DecisionGrid:
    events = tape.arrays.events
    l2_events = tape.arrays.l2_events
    decision_indices: list[int] = []
    for event_index, event in enumerate(events):
        if int(event["event_type_code"]) != EVENT_TYPE_CODE_L2_BATCH:
            continue
        book_ptr = int(event["book_ptr"])
        if book_ptr < 0 or book_ptr >= len(l2_events):
            continue
        book = l2_events[book_ptr]
        if int(book["best_bid_tick"]) <= 0 or int(book["best_ask_tick"]) <= int(book["best_bid_tick"]):
            continue
        decision_indices.append(event_index)
        if max_rows is not None and len(decision_indices) >= max_rows:
            break
    if not decision_indices:
        raise ValueError("test tape has no valid L2 decisions")

    event_idx = np.asarray(decision_indices, dtype=np.int64)
    ts = np.asarray([int(events[i]["local_ts_us"]) for i in decision_indices], dtype=np.int64)
    seq = np.asarray([int(events[i]["event_seq"]) for i in decision_indices], dtype=np.int64)
    book_ptr = np.asarray([int(events[i]["book_ptr"]) for i in decision_indices], dtype=np.int64)
    reason_code = np.full(len(event_idx), DECISION_REASON_HEARTBEAT, dtype=np.int16)
    reason_flags = np.full(len(event_idx), DECISION_REASON_FLAG_HEARTBEAT, dtype=np.int16)
    reason_code[0] = DECISION_REASON_FIRST_VALID_BOOK
    reason_flags[0] = DECISION_REASON_FLAG_FIRST_VALID_BOOK

    elapsed = np.zeros(len(event_idx), dtype=np.int64)
    events_since = np.zeros(len(event_idx), dtype=np.int64)
    l2_since = np.zeros(len(event_idx), dtype=np.int64)
    trade_since = np.zeros(len(event_idx), dtype=np.int64)
    for row in range(1, len(event_idx)):
        prev = int(event_idx[row - 1])
        cur = int(event_idx[row])
        elapsed[row] = int(ts[row] - ts[row - 1])
        events_since[row] = cur - prev
        window = events[prev + 1 : cur + 1]
        l2_since[row] = int(np.count_nonzero(window["event_type_code"] == EVENT_TYPE_CODE_L2_BATCH))
        trade_since[row] = int(np.count_nonzero(window["event_type_code"] == EVENT_TYPE_CODE_TRADE))

    arrays = {
        "decision_event_index": event_idx,
        "decision_local_ts_us": ts,
        "decision_event_seq": seq,
        "book_ptr": book_ptr,
        "reason_code": reason_code,
        "reason_flags": reason_flags,
        "elapsed_since_prev_decision_us": elapsed,
        "events_since_prev_decision": events_since,
        "l2_events_since_prev_decision": l2_since,
        "trade_events_since_prev_decision": trade_since,
    }
    metadata = decision_grid_metadata_from_tape(
        tape,
        schedule_config=schedule_config or DecisionScheduleConfig(),
        arrays={name: arrays[name] for name in DECISION_GRID_ARRAY_ORDER},
        created_at_utc="2026-01-01T00:00:00Z",
    )
    return DecisionGrid(metadata=metadata, **arrays)
