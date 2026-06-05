from pathlib import Path

import numpy as np
import pytest

from mmrt.execution.contracts import LinearSignal
from mmrt.execution.linear_signal import (
    DIRECTION_PROBA_KEY,
    MAGNITUDE_DOWN_KEY,
    MAGNITUDE_INPUT_BPS,
    MAGNITUDE_INPUT_LOG1P_BPS,
    MAGNITUDE_UP_KEY,
    NO_MOVE_PROBA_KEY,
    LinearSignalArrays,
    LinearSignalConfig,
    expected_return_bps,
    linear_signal_at,
    magnitude_to_bps,
    make_linear_signal,
    neutral_linear_signal,
    prediction_row_to_signal,
    predictions_to_signal_arrays,
    signal_confidence,
)


def _prediction_dict():
    return {
        NO_MOVE_PROBA_KEY: np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        DIRECTION_PROBA_KEY: np.array(
            [
                [0.3, 0.7],
                [0.6, 0.4],
            ],
            dtype=np.float32,
        ),
        MAGNITUDE_UP_KEY: np.log1p(np.array([10.0, 4.0], dtype=np.float32)),
        MAGNITUDE_DOWN_KEY: np.log1p(np.array([5.0, 8.0], dtype=np.float32)),
    }


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
    cfg = LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_LOG1P_BPS)
    assert magnitude_to_bps(np.log1p(10.0), config=cfg) == pytest.approx(10.0)


def test_magnitude_to_bps_from_bps():
    cfg = LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_BPS)
    assert magnitude_to_bps(10.0, config=cfg) == pytest.approx(10.0)


def test_magnitude_rejects_negative_or_nonfinite():
    with pytest.raises(ValueError):
        magnitude_to_bps(-1.0)
    with pytest.raises(ValueError):
        magnitude_to_bps(float("inf"))


def test_expected_return_bps_uses_no_move_probability():
    value = expected_return_bps(
        p_no_move=0.25,
        p_up=0.8,
        mag_up_bps=10.0,
        mag_down_bps=5.0,
    )

    expected = 0.75 * (0.8 * 10.0 - 0.2 * 5.0)
    assert value == pytest.approx(expected)


def test_signal_confidence():
    confidence = signal_confidence(p_no_move=0.25, p_up=0.8)
    assert confidence == pytest.approx(0.75 * abs(2.0 * 0.8 - 1.0))

    assert signal_confidence(p_no_move=1.0, p_up=1.0) == 0.0
    assert signal_confidence(p_no_move=0.0, p_up=0.5) == 0.0


def test_make_linear_signal_from_log_magnitudes():
    signal = make_linear_signal(
        p_no_move=0.2,
        p_up=0.7,
        magnitude_up=np.log1p(10.0),
        magnitude_down=np.log1p(5.0),
    )

    assert isinstance(signal, LinearSignal)
    assert signal.p_no_move == pytest.approx(0.2)
    assert signal.p_up == pytest.approx(0.7)
    assert signal.mag_up_bps == pytest.approx(10.0)
    assert signal.mag_down_bps == pytest.approx(5.0)
    assert signal.expected_return_bps == pytest.approx(0.8 * (0.7 * 10.0 - 0.3 * 5.0))
    assert signal.confidence == pytest.approx(0.8 * abs(2.0 * 0.7 - 1.0))


def test_neutral_linear_signal():
    signal = neutral_linear_signal()

    assert signal == LinearSignal(
        p_no_move=1.0,
        p_up=0.5,
        mag_up_bps=0.0,
        mag_down_bps=0.0,
        expected_return_bps=0.0,
        confidence=0.0,
    )


def test_prediction_row_to_signal():
    prediction = _prediction_dict()
    signal = prediction_row_to_signal(prediction, 0)

    assert signal.p_no_move == pytest.approx(0.2)
    assert signal.p_up == pytest.approx(0.7)
    assert signal.mag_up_bps == pytest.approx(10.0)
    assert signal.mag_down_bps == pytest.approx(5.0)

    signal = prediction_row_to_signal(prediction, 1)
    assert signal.p_no_move == pytest.approx(0.9)
    assert signal.p_up == pytest.approx(0.4)
    assert signal.mag_up_bps == pytest.approx(4.0)
    assert signal.mag_down_bps == pytest.approx(8.0)


def test_predictions_to_signal_arrays():
    arrays = predictions_to_signal_arrays(_prediction_dict())

    assert isinstance(arrays, LinearSignalArrays)
    assert arrays.n_rows == 2
    assert arrays.dtype == np.dtype("float32")
    assert arrays.p_no_move.tolist() == pytest.approx([0.2, 0.9])
    assert arrays.p_up.tolist() == pytest.approx([0.7, 0.4])
    assert arrays.mag_up_bps.tolist() == pytest.approx([10.0, 4.0])
    assert arrays.mag_down_bps.tolist() == pytest.approx([5.0, 8.0])
    assert arrays.expected_return_bps.tolist() == pytest.approx(
        [
            0.8 * (0.7 * 10.0 - 0.3 * 5.0),
            0.1 * (0.4 * 4.0 - 0.6 * 8.0),
        ]
    )


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
    assert signal.expected_return_bps == pytest.approx(arrays.expected_return_bps[0])

    with pytest.raises(ValueError):
        linear_signal_at(arrays, -1)

    with pytest.raises(ValueError):
        linear_signal_at(arrays, arrays.n_rows)


def test_probability_tiny_spillover_clipped():
    cfg = LinearSignalConfig(probability_epsilon=1e-6)
    signal = make_linear_signal(
        p_no_move=-1e-7,
        p_up=1.0 + 1e-7,
        magnitude_up=0.0,
        magnitude_down=0.0,
        config=cfg,
    )

    assert signal.p_no_move == 0.0
    assert signal.p_up == 1.0


def test_probability_large_out_of_range_rejected():
    cfg = LinearSignalConfig(probability_epsilon=1e-6)

    with pytest.raises(ValueError):
        make_linear_signal(
            p_no_move=-1e-3,
            p_up=0.5,
            magnitude_up=0.0,
            magnitude_down=0.0,
            config=cfg,
        )

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
    cfg = LinearSignalConfig(magnitude_input=MAGNITUDE_INPUT_BPS)

    arrays = predictions_to_signal_arrays(prediction, config=cfg)

    assert arrays.mag_up_bps[0] == pytest.approx(10.0)
    assert arrays.mag_down_bps[0] == pytest.approx(5.0)


def test_linear_signal_arrays_validation():
    valid = predictions_to_signal_arrays(_prediction_dict())

    with pytest.raises(ValueError):
        LinearSignalArrays(
            p_no_move=valid.p_no_move,
            p_up=valid.p_up[:-1],
            mag_up_bps=valid.mag_up_bps,
            mag_down_bps=valid.mag_down_bps,
            expected_return_bps=valid.expected_return_bps,
            confidence=valid.confidence,
        )

    with pytest.raises(ValueError):
        LinearSignalArrays(
            p_no_move=np.array([1.2], dtype=np.float32),
            p_up=np.array([0.5], dtype=np.float32),
            mag_up_bps=np.array([0.0], dtype=np.float32),
            mag_down_bps=np.array([0.0], dtype=np.float32),
            expected_return_bps=np.array([0.0], dtype=np.float32),
            confidence=np.array([0.0], dtype=np.float32),
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
