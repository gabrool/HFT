import pytest

from mmrt.cli.linear_signal_validation import validate_linear_signals_for_execution_tape
from tests.test_audit_execution_sim import _linear_artifact_for_tape, _l2, _tape


def test_validate_linear_signals_for_execution_tape_accepts_later_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    linear_signals = _linear_artifact_for_tape(tape, n_rows=3, decision_interval_us=50)

    linear_start = validate_linear_signals_for_execution_tape(
        linear_signals=linear_signals,
        tape=tape,
        decision_interval_us=50,
        requested_start_event_index=1,
        min_rows=2,
    )

    assert linear_signals.metadata.start_event_index == 0
    assert linear_start.event_index == 1
    assert linear_start.row_index == 1
    assert linear_start.rows_available == 2


def test_validate_linear_signals_for_execution_tape_rejects_non_grid_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    linear_signals = _linear_artifact_for_tape(tape, n_rows=2, decision_interval_us=50)

    with pytest.raises(ValueError, match="decision_event_index"):
        validate_linear_signals_for_execution_tape(
            linear_signals=linear_signals,
            tape=tape,
            decision_interval_us=50,
            requested_start_event_index=2,
            min_rows=None,
        )


def test_validate_linear_signals_for_execution_tape_checks_remaining_rows_after_start():
    tape = _tape(
        [_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=200), _l2(seq=2, local_ts_us=300)],
        [],
    )
    linear_signals = _linear_artifact_for_tape(tape, n_rows=3, decision_interval_us=50)

    with pytest.raises(ValueError, match="not contain enough rows"):
        validate_linear_signals_for_execution_tape(
            linear_signals=linear_signals,
            tape=tape,
            decision_interval_us=50,
            requested_start_event_index=2,
            min_rows=2,
        )
