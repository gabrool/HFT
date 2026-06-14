"""Shared validation helpers for CLI execution runs with linear signals."""

from __future__ import annotations

from mmrt.execution.execution_tape import ExecutionTape
from mmrt.execution.decision_grid import DecisionGrid, validate_decision_grid_for_execution_tape
from mmrt.execution.linear_signal import (
    LinearSignalArtifact,
    LinearSignalStart,
    validate_linear_signal_artifact_metadata,
    validate_linear_signal_start_event_index,
    validate_linear_signals_for_decision_grid,
)


def validate_linear_signals_for_execution_tape(
    *,
    linear_signals: LinearSignalArtifact,
    tape: ExecutionTape,
    decision_grid: DecisionGrid,
    requested_start_event_index: int | None,
    min_rows: int | None,
) -> LinearSignalStart:
    """Validate artifact identity separately from the requested run start row.

    The decision grid is the source of truth for decision timing. Linear signal
    metadata must copy that grid lineage exactly.
    """

    validate_decision_grid_for_execution_tape(decision_grid, tape)
    validate_linear_signal_artifact_metadata(
        linear_signals,
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_grid_schema=decision_grid.metadata.schema,
        decision_grid_hash=decision_grid.decision_grid_hash,
        decision_grid_n_rows=decision_grid.n_rows,
        decision_schedule=decision_grid.decision_schedule,
        start_event_index=linear_signals.metadata.start_event_index,
        min_rows=None,
    )
    validate_linear_signals_for_decision_grid(linear_signals, decision_grid)
    return validate_linear_signal_start_event_index(
        linear_signals,
        start_event_index=requested_start_event_index,
        min_rows=min_rows,
    )


__all__ = ["validate_linear_signals_for_execution_tape"]
