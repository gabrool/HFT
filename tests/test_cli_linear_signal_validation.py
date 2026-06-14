import pytest

from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from tests.test_audit_execution_sim import _linear_artifact_for_tape, _l2, _tape
from tests.grid_helpers import decision_grid_for_tape


def test_validate_linear_signals_for_execution_tape_accepts_later_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    grid = decision_grid_for_tape(tape)
    linear_signals = _linear_artifact_for_tape(tape, decision_grid=grid)

    decision_grid_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_grid=grid,
        requested_start_event_index=1,
        min_rows=2,
    )

    assert linear_signals.metadata.start_event_index == 0
    assert decision_grid_start.event_index == 1
    assert decision_grid_start.decision_grid_row_index == 1
    assert decision_grid_start.rows_available == 2


def test_validate_linear_signals_for_execution_tape_rejects_non_grid_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    grid = decision_grid_for_tape(tape, max_rows=2)
    linear_signals = _linear_artifact_for_tape(tape, decision_grid=grid)

    with pytest.raises(ValueError, match="decision grid row"):
        validate_linear_signals_for_execution_tape(
            linear_signals=linear_signals,
            tape=tape,
            decision_grid=grid,
            requested_start_event_index=2,
            min_rows=None,
        )


def test_validate_linear_signals_for_execution_tape_checks_remaining_rows_after_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    grid = decision_grid_for_tape(tape)
    linear_signals = _linear_artifact_for_tape(tape, decision_grid=grid)

    with pytest.raises(ValueError, match="not contain enough rows"):
        validate_linear_signals_for_execution_tape(
            linear_signals=linear_signals,
            tape=tape,
            decision_grid=grid,
            requested_start_event_index=2,
            min_rows=2,
        )
