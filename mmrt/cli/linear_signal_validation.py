"""Shared validation helpers for CLI execution runs with linear signals."""

from __future__ import annotations

from mmrt.execution.execution_tape import ExecutionTape
from mmrt.execution.linear_signal import (
    LinearSignalArtifact,
    LinearSignalStart,
    validate_linear_signal_artifact_metadata,
    validate_linear_signal_start_event_index,
)


def validate_linear_signals_for_execution_tape(
    *,
    linear_signals: LinearSignalArtifact,
    tape: ExecutionTape,
    requested_start_event_index: int | None,
    min_rows: int | None,
) -> LinearSignalStart:
    """Validate artifact identity separately from the requested run start row.

    The artifact's own decision schedule is the source of truth for decision
    timing; metadata construction re-parses it so only schedules reproducible
    by the current code are accepted.
    """

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
        decision_schedule=linear_signals.metadata.decision_schedule,
        start_event_index=linear_signals.metadata.start_event_index,
        min_rows=None,
    )
    return validate_linear_signal_start_event_index(
        linear_signals,
        start_event_index=requested_start_event_index,
        min_rows=min_rows,
    )


__all__ = ["validate_linear_signals_for_execution_tape"]
