from pathlib import Path

import numpy as np
import pytest

from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.execution.contracts import LinearSignal
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    LINEAR_SIGNAL_ARTIFACT_SCHEMA,
    LINEAR_SIGNALS_FILENAME,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_INPUT_BPS,
    MAGNITUDE_INPUT_LOG1P_BPS,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArrays,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    LinearSignalConfig,
    build_gated_linear_signal,
    linear_signal_artifact_summary,
    linear_signal_at,
    load_linear_signal_artifact_npz,
    magnitude_to_bps,
    prediction_row_to_signal,
    predictions_to_signal_arrays,
    save_linear_signal_artifact_npz,
    validate_linear_signal_artifact_metadata,
    validate_linear_signal_start_event_index,
)


def _fixed_schedule_payload(stride_us: int) -> dict:
    return DecisionScheduleConfig(min_decision_interval_us=stride_us, max_decision_interval_us=stride_us).as_dict()


def _prediction_dict():
    return {
        NO_MOVE_PROBA_KEY: np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        DIRECTION_PROBA_KEY: np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32),
        MAGNITUDE_UP_KEY: np.log1p(np.array([10.0, 4.0], dtype=np.float32)),
        MAGNITUDE_DOWN_KEY: np.log1p(np.array([5.0, 8.0], dtype=np.float32)),
    }


def _artifact(n_rows: int) -> LinearSignalArtifact:
    prediction = {
        NO_MOVE_PROBA_KEY: np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (n_rows, 1)),
        DIRECTION_PROBA_KEY: np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (n_rows, 1)),
        MAGNITUDE_UP_KEY: np.full(n_rows, np.log1p(10.0), dtype=np.float32),
        MAGNITUDE_DOWN_KEY: np.full(n_rows, np.log1p(5.0), dtype=np.float32),
    }
    arrays = predictions_to_signal_arrays(prediction)
    metadata = LinearSignalArtifactMetadata(
        tape_schema="tape",
        exchange="X",
        symbol="BTC-USD",
        num_events=5,
        num_l2_batches=4,
        num_trades=1,
        start_local_ts_us=100,
        end_local_ts_us=200,
        decision_schedule=_fixed_schedule_payload(50),
        start_event_index=0,
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.arange(n_rows, dtype=np.int64),
        decision_local_ts_us=np.arange(100, 100 + n_rows * 50, 50, dtype=np.int64),
    )


def _linear_artifact_with_decision_event_index(indices: list[int]) -> LinearSignalArtifact:
    n_rows = len(indices)
    prediction = {
        NO_MOVE_PROBA_KEY: np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (n_rows, 1)),
        DIRECTION_PROBA_KEY: np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (n_rows, 1)),
        MAGNITUDE_UP_KEY: np.full(n_rows, np.log1p(10.0), dtype=np.float32),
        MAGNITUDE_DOWN_KEY: np.full(n_rows, np.log1p(5.0), dtype=np.float32),
    }
    arrays = predictions_to_signal_arrays(prediction)
    metadata = LinearSignalArtifactMetadata(
        tape_schema="tape",
        exchange="X",
        symbol="BTC-USD",
        num_events=max(indices) + 1,
        num_l2_batches=n_rows,
        num_trades=0,
        start_local_ts_us=100,
        end_local_ts_us=100 + 100 * (n_rows - 1),
        decision_schedule=_fixed_schedule_payload(100),
        start_event_index=indices[0],
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.asarray(indices, dtype=np.int64),
        decision_local_ts_us=np.arange(100, 100 + n_rows * 100, 100, dtype=np.int64),
    )


def test_linear_signal_config_validation():
    assert LinearSignalConfig().magnitude_input == MAGNITUDE_INPUT_LOG1P_BPS
    assert LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_BPS).magnitude_input == MAGNITUDE_INPUT_BPS
    with pytest.raises(ValueError):
        LinearSignalConfig(magnitude_input="bad")
    with pytest.raises(ValueError):
        LinearSignalConfig(probability_epsilon=-1.0)
    with pytest.raises(ValueError):
        LinearSignalConfig(probability_epsilon=0.5)
    with pytest.raises(ValueError):
        LinearSignalConfig(probability_epsilon=float("nan"))


def test_magnitude_to_bps_from_log1p():
    assert magnitude_to_bps(np.log1p(10.0), config=LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_LOG1P_BPS)) == pytest.approx(10.0)


def test_magnitude_to_bps_from_bps():
    assert magnitude_to_bps(10.0, config=LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_BPS)) == pytest.approx(10.0)


def test_magnitude_rejects_negative_or_nonfinite():
    with pytest.raises(ValueError):
        magnitude_to_bps(-1.0)
    with pytest.raises(ValueError):
        magnitude_to_bps(float("inf"))


def test_no_move_gating_scalar():
    signal = build_gated_linear_signal(
        p_no_move=1.0,
        p_up=1.0,
        magnitude_up=np.log1p(100.0),
        magnitude_down=np.log1p(100.0),
    )
    assert signal.p_move == 0.0
    assert signal.p_up_move == 0.0
    assert signal.p_down_move == 0.0
    assert signal.signed_move_prob == 0.0
    assert signal.expected_up_bps == 0.0
    assert signal.expected_down_bps == 0.0
    assert signal.expected_return_bps == 0.0
    assert signal.expected_abs_move_bps == 0.0
    assert signal.predicted_vol_bps == 0.0
    assert signal.confidence == 0.0


def test_normal_formula_scalar():
    signal = build_gated_linear_signal(
        p_no_move=0.2,
        p_up=0.7,
        magnitude_up=np.log1p(10.0),
        magnitude_down=np.log1p(5.0),
    )
    assert isinstance(signal, LinearSignal)
    assert signal.p_move == pytest.approx(0.8)
    assert signal.p_up_move == pytest.approx(0.56)
    assert signal.p_down_move == pytest.approx(0.24)
    assert signal.signed_move_prob == pytest.approx(0.32)
    assert signal.expected_up_bps == pytest.approx(5.6)
    assert signal.expected_down_bps == pytest.approx(1.2)
    assert signal.expected_return_bps == pytest.approx(4.4)
    assert signal.expected_abs_move_bps == pytest.approx(6.8)
    assert signal.predicted_vol_bps == pytest.approx(np.sqrt(62.0 - 4.4 * 4.4))
    assert signal.confidence == pytest.approx(0.32)


def test_prediction_row_to_signal():
    signal = prediction_row_to_signal(_prediction_dict(), 0)
    assert signal.p_no_move == pytest.approx(0.2)
    assert signal.p_move == pytest.approx(0.8)
    assert signal.p_up_move == pytest.approx(0.56)
    assert signal.expected_up_bps == pytest.approx(5.6)

    signal = prediction_row_to_signal(_prediction_dict(), 1)
    assert signal.p_no_move == pytest.approx(0.9)
    assert signal.p_move == pytest.approx(0.1)
    assert signal.p_up_move == pytest.approx(0.04)
    assert signal.p_down_move == pytest.approx(0.06)


def test_predictions_to_signal_arrays():
    arrays = predictions_to_signal_arrays(_prediction_dict())
    assert isinstance(arrays, LinearSignalArrays)
    assert arrays.n_rows == 2
    assert arrays.dtype == np.dtype("float32")
    np.testing.assert_allclose(arrays.p_no_move, [0.2, 0.9], rtol=1e-6)
    np.testing.assert_allclose(arrays.p_move, [0.8, 0.1], rtol=1e-6)
    np.testing.assert_allclose(arrays.p_up_move, [0.56, 0.04], rtol=1e-6)
    np.testing.assert_allclose(arrays.p_down_move, [0.24, 0.06], rtol=1e-6)
    np.testing.assert_allclose(arrays.signed_move_prob, [0.32, -0.02], rtol=1e-6)
    np.testing.assert_allclose(arrays.expected_up_bps, [5.6, 0.16], rtol=1e-6)
    np.testing.assert_allclose(arrays.expected_down_bps, [1.2, 0.48], rtol=1e-6)
    np.testing.assert_allclose(arrays.expected_return_bps, [4.4, -0.32], rtol=1e-6)
    np.testing.assert_allclose(arrays.expected_abs_move_bps, [6.8, 0.64], rtol=1e-6)
    np.testing.assert_allclose(arrays.confidence, [0.32, 0.02], rtol=1e-6)


def test_no_raw_fields_on_arrays():
    arrays = predictions_to_signal_arrays(_prediction_dict())
    assert not hasattr(arrays, "p_up")
    assert not hasattr(arrays, "mag_up_bps")
    assert not hasattr(arrays, "mag_down_bps")


def test_predictions_to_signal_arrays_output_dtype_float64():
    arrays = predictions_to_signal_arrays(_prediction_dict(), output_dtype="float64")
    assert arrays.dtype == np.dtype("float64")
    assert arrays.p_no_move.dtype == np.float64
    with pytest.raises(ValueError):
        predictions_to_signal_arrays(_prediction_dict(), output_dtype="int64")


def test_linear_signal_at():
    arrays = predictions_to_signal_arrays(_prediction_dict())
    signal = linear_signal_at(arrays, 0)
    assert isinstance(signal, LinearSignal)
    assert signal.p_no_move == pytest.approx(arrays.p_no_move[0])
    assert signal.expected_abs_move_bps == pytest.approx(arrays.expected_abs_move_bps[0])
    with pytest.raises(ValueError):
        linear_signal_at(arrays, -1)
    with pytest.raises(ValueError):
        linear_signal_at(arrays, arrays.n_rows)


def test_probability_tiny_spillover_clipped():
    signal = build_gated_linear_signal(
        p_no_move=-1e-7,
        p_up=1.0 + 1e-7,
        magnitude_up=0.0,
        magnitude_down=0.0,
        config=LinearSignalConfig(probability_epsilon=1e-6),
    )
    assert signal.p_no_move == 0.0
    assert signal.p_move == 1.0
    assert signal.p_up_move == 1.0


def test_probability_large_out_of_range_rejected():
    cfg = LinearSignalConfig(probability_epsilon=1e-6)
    with pytest.raises(ValueError):
        build_gated_linear_signal(p_no_move=-1e-3, p_up=0.5, magnitude_up=0.0, magnitude_down=0.0, config=cfg)
    prediction = _prediction_dict()
    prediction[NO_MOVE_PROBA_KEY] = np.array([[0.0, 1.01]], dtype=np.float32)
    with pytest.raises(ValueError):
        predictions_to_signal_arrays(prediction, config=cfg)


def test_invalid_prediction_shapes_rejected():
    prediction = _prediction_dict()
    bad = dict(prediction)
    bad[NO_MOVE_PROBA_KEY] = np.array([0.2, 0.9], dtype=np.float32)
    with pytest.raises(ValueError):
        predictions_to_signal_arrays(bad)
    bad = dict(prediction)
    bad[DIRECTION_PROBA_KEY] = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        predictions_to_signal_arrays(bad)
    bad = dict(prediction)
    bad[MAGNITUDE_UP_KEY] = np.ones((3,), dtype=np.float32)
    with pytest.raises(ValueError):
        predictions_to_signal_arrays(bad)


def test_missing_prediction_keys_rejected():
    prediction = _prediction_dict()
    prediction.pop(NO_MOVE_PROBA_KEY)
    with pytest.raises(ValueError):
        predictions_to_signal_arrays(prediction)
    with pytest.raises(ValueError):
        prediction_row_to_signal(prediction, 0)


def test_bps_magnitude_input_mode():
    prediction = {
        NO_MOVE_PROBA_KEY: np.array([[0.8, 0.2]], dtype=np.float32),
        DIRECTION_PROBA_KEY: np.array([[0.3, 0.7]], dtype=np.float32),
        MAGNITUDE_UP_KEY: np.array([10.0], dtype=np.float32),
        MAGNITUDE_DOWN_KEY: np.array([5.0], dtype=np.float32),
    }
    arrays = predictions_to_signal_arrays(prediction, config=LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_BPS))
    assert arrays.expected_up_bps[0] == pytest.approx(5.6)
    assert arrays.expected_down_bps[0] == pytest.approx(1.2)


def test_linear_signal_arrays_validation():
    valid = predictions_to_signal_arrays(_prediction_dict())
    kwargs = {name: getattr(valid, name) for name in linear_signal_artifact_summary(_artifact(2))["fields"]}
    bad = dict(kwargs)
    bad["p_move"] = valid.p_move[:-1]
    with pytest.raises(ValueError):
        LinearSignalArrays(**bad)
    bad = dict(kwargs)
    bad["p_no_move"] = np.array([1.2, 1.2], dtype=np.float32)
    with pytest.raises(ValueError):
        LinearSignalArrays(**bad)


def test_npz_round_trip(tmp_path):
    artifact = _artifact(2)
    path = tmp_path / LINEAR_SIGNALS_FILENAME
    save_linear_signal_artifact_npz(path, artifact)
    loaded = load_linear_signal_artifact_npz(path)
    assert loaded.metadata == artifact.metadata
    np.testing.assert_array_equal(loaded.decision_event_index, artifact.decision_event_index)
    np.testing.assert_array_equal(loaded.decision_local_ts_us, artifact.decision_local_ts_us)
    np.testing.assert_allclose(loaded.arrays.expected_abs_move_bps, artifact.arrays.expected_abs_move_bps)


def test_metadata_free_old_schema_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    arrays = predictions_to_signal_arrays(_prediction_dict())
    payload = {name: getattr(arrays, name) for name in linear_signal_artifact_summary(_artifact(2))["fields"]}
    payload["schema"] = np.array("mmrt_execution_linear_signals_no_move_gated")
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        load_linear_signal_artifact_npz(path)


def test_validate_linear_signal_artifact_metadata_rejects_mismatch():
    artifact = _artifact(2)
    validate_linear_signal_artifact_metadata(
        artifact,
        tape_schema="tape",
        exchange="X",
        symbol="BTC-USD",
        num_events=5,
        num_l2_batches=4,
        num_trades=1,
        start_local_ts_us=100,
        end_local_ts_us=200,
        decision_schedule=_fixed_schedule_payload(50),
        start_event_index=0,
        min_rows=2,
    )
    with pytest.raises(ValueError, match="linear signal metadata mismatch"):
        validate_linear_signal_artifact_metadata(
            artifact,
            tape_schema="tape",
            exchange="X",
            symbol="ETH-USD",
            num_events=5,
            num_l2_batches=4,
            num_trades=1,
            start_local_ts_us=100,
            end_local_ts_us=200,
            decision_schedule=_fixed_schedule_payload(50),
            start_event_index=0,
        )
    with pytest.raises(ValueError, match="linear signal metadata mismatch"):
        validate_linear_signal_artifact_metadata(
            artifact,
            tape_schema="tape",
            exchange="X",
            symbol="BTC-USD",
            num_events=5,
            num_l2_batches=4,
            num_trades=1,
            start_local_ts_us=100,
            end_local_ts_us=200,
            decision_schedule=_fixed_schedule_payload(100),
            start_event_index=0,
        )


def test_linear_signal_has_no_forbidden_imports():
    source = Path("mmrt/execution/linear_signal.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import polars" not in source
    assert "import sklearn" not in source
    assert "import pyarrow" not in source
    assert "mmrt.linear.models" not in source
    assert "mmrt.linear.targets" not in source
    assert "mmrt.storage" not in source
    assert "neutral_linear_signal" not in source


def test_linear_signal_has_no_metadata_free_artifact_helpers():
    source = Path("mmrt/execution/linear_signal.py").read_text(encoding="utf-8")
    assert "_save_linear_signal_arrays_npz" not in source
    assert "save_linear_signal_arrays_npz" not in source
    assert "load_linear_signal_arrays_npz" not in source


def test_validate_linear_signal_start_event_index_accepts_default_and_later_rows():
    artifact = _linear_artifact_with_decision_event_index([10, 20, 30, 40])

    default = validate_linear_signal_start_event_index(artifact)
    assert default.event_index == 10
    assert default.row_index == 0
    assert default.rows_available == 4

    later = validate_linear_signal_start_event_index(artifact, start_event_index=30)
    assert later.event_index == 30
    assert later.row_index == 2
    assert later.rows_available == 2


def test_validate_linear_signal_start_event_index_rejects_non_grid_start():
    artifact = _linear_artifact_with_decision_event_index([10, 20, 30])
    with pytest.raises(ValueError, match="decision_event_index"):
        validate_linear_signal_start_event_index(artifact, start_event_index=25)


def test_validate_linear_signal_start_event_index_checks_remaining_rows():
    artifact = _linear_artifact_with_decision_event_index([10, 20, 30])
    with pytest.raises(ValueError, match="not contain enough rows"):
        validate_linear_signal_start_event_index(
            artifact,
            start_event_index=30,
            min_rows=2,
        )
