from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from mmrt.config import default_config
from mmrt.contracts import SplitRole, TimeRangeUS
from mmrt.execution.decision_grid import (
    DECISION_GRID_ARRAY_ORDER,
    DECISION_GRID_SCHEMA,
    DecisionGrid,
    decision_grid_metadata_from_tape,
)
from mmrt.execution.split_contract import EXECUTION_SPLIT_CONTRACT_SCHEMA
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, EVENT_TYPE_CODE_TRADE, ExecutionTape
from mmrt.features.schedule import (
    DECISION_REASON_FIRST_VALID_BOOK,
    DECISION_REASON_FLAG_FIRST_VALID_BOOK,
    DECISION_REASON_FLAG_HEARTBEAT,
    DECISION_REASON_HEARTBEAT,
    DecisionScheduleConfig,
    decision_schedule_config_from_dict,
)
from mmrt.storage.manifest import (
    DEFAULT_MANIFEST_FILENAME,
    SplitMetadata,
    StorageSegment,
    make_manifest,
    write_manifest_json,
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


def write_split_source_manifest(
    root: str | Path,
    grid: DecisionGrid,
    *,
    dataset_id: str = "split-source",
    split_spans: Mapping[str, tuple[int, int]] | None = None,
) -> Path:
    """Write a current storage manifest with train/val/test decision-row splits."""

    target = Path(root)
    n_rows = int(grid.n_rows)
    if n_rows < 3:
        raise ValueError("test split source requires at least three decision rows")
    if split_spans is None:
        if n_rows >= 6:
            split_spans = {
                "train": (0, n_rows - 4),
                "val": (n_rows - 4, n_rows - 2),
                "test": (n_rows - 2, n_rows),
            }
        else:
            split_spans = {
                "train": (0, n_rows - 2),
                "val": (n_rows - 2, n_rows - 1),
                "test": (n_rows - 1, n_rows),
            }

    first_ts = int(grid.decision_local_ts_us[0])
    last_ts = int(grid.decision_local_ts_us[-1])
    segment = StorageSegment(
        segment_key="seg_000",
        parquet_path="segments/seg_000.parquet",
        row_count=n_rows,
        label_count=n_rows,
        time_range=TimeRangeUS(first_ts, last_ts + 1),
        local_time_range=TimeRangeUS(first_ts, last_ts + 1),
        first_row_idx=0,
        last_row_idx=n_rows - 1,
    )
    splits = []
    for role in ("train", "val", "test"):
        start, end = split_spans[role]
        splits.append(
            SplitMetadata(
                role=SplitRole(role),
                segment_key="seg_000",
                start_row=int(start),
                end_row=int(end),
                local_time_range=TimeRangeUS(
                    int(grid.decision_local_ts_us[int(start)]),
                    int(grid.decision_local_ts_us[int(end) - 1]) + 1,
                ),
            )
        )

    cfg = default_config()
    cfg = replace(
        cfg,
        decision=replace(
            cfg.decision,
            schedule=decision_schedule_config_from_dict(grid.decision_schedule),
        ),
    )
    manifest = make_manifest(
        dataset_id=dataset_id,
        created_at_utc="2026-01-01T00:00:00Z",
        segments=(segment,),
        config=cfg,
        splits=tuple(splits),
        notes=grid_lineage_notes(
            n_rows=grid.n_rows,
            schedule=grid.decision_schedule,
            grid_hash=grid.decision_grid_hash,
        ),
    )
    write_manifest_json(manifest, target / DEFAULT_MANIFEST_FILENAME)
    return target


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
    contract_n_rows = int(n_rows)
    if ranges is None:
        def _ranges(role: str, preferred_start: int) -> list[dict[str, object]]:
            if preferred_start >= contract_n_rows:
                return []
            start = preferred_start
            end = start + 1
            start_ts = start + 1
            return [{
                "role": role,
                "segment_key": "seg_000",
                "start_decision_row": start,
                "end_decision_row": end,
                "row_count": end - start,
                "start_local_ts_us": start_ts,
                "end_local_ts_us": start_ts + 1,
                "embargo_before_us": 0,
                "embargo_after_us": 0,
            }]

        ranges = {
            "train": _ranges("train", 0),
            "val": _ranges("val", 1),
            "test": _ranges("test", 2),
        }
    row_counts_by_split = {
        role: int(sum(int(entry["row_count"]) for entry in entries))
        for role, entries in ranges.items()
    }
    contract = {
        "schema": EXECUTION_SPLIT_CONTRACT_SCHEMA,
        "version": 1,
        "split_source_dataset_root": root,
        "split_source_dataset_id": dataset_id,
        "split_source_manifest_hash": manifest_hash,
        "ranges_by_split": {role: [dict(entry) for entry in entries] for role, entries in ranges.items()},
        "row_counts_by_split": row_counts_by_split,
        "adverse_dataset_rows_total": 0,
        "adverse_row_counts": {"train": 0, "val": 0, "test": 0, "out_of_split": 0},
        **grid_lineage_fields(n_rows=contract_n_rows, schedule=schedule, grid_hash=grid_hash),
    }
    return {
        "split_source_dataset_root": root,
        "split_source_dataset_id": dataset_id,
        "split_source_manifest_hash": manifest_hash,
        "split_contract": contract,
    }


def adverse_split_contract_for_grid(grid: DecisionGrid, *, root: str = "/tmp/split_source") -> dict[str, object]:
    ts = [int(x) for x in grid.decision_local_ts_us]
    n_rows = int(grid.n_rows)

    def _ranges(role: str, preferred_start: int) -> list[dict[str, object]]:
        if preferred_start >= n_rows:
            return []
        start = preferred_start
        end = start + 1
        start_ts = ts[start]
        return [{
            "role": role,
            "segment_key": "seg_000",
            "start_decision_row": start,
            "end_decision_row": end,
            "row_count": end - start,
            "start_local_ts_us": start_ts,
            "end_local_ts_us": start_ts + 1,
            "embargo_before_us": 0,
            "embargo_after_us": 0,
        }]

    ranges = {
        "train": _ranges("train", 0),
        "val": _ranges("val", 1),
        "test": _ranges("test", 2),
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
