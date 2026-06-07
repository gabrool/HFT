import pytest
import numpy as np

from mmrt.execution.adverse_runtime import (
    AdverseRuntimeConfig,
    adverse_predictions_for_row,
    build_adverse_observation_features,
    build_executable_edge_observation_features,
)
from mmrt.execution.adverse_signal import ADVERSE_SELECTION_SIGNALS_SCHEMA, AdverseSelectionSignalArtifact
from mmrt.execution.contracts import LinearSignal


def _signal():
    return LinearSignal(0.5, 0.5, 0.3, 0.2, 0.1, 3.0, 1.0, 2.0, 4.0, 1.0, 0.1)


def _preds():
    return {
        "bid_touch_filled": 0.5,
        "ask_touch_filled": 0.4,
        "bid_touch_toxic_cost_bps": 1.0,
        "ask_touch_toxic_cost_bps": 2.0,
    }


def test_adverse_predictions_for_row_returns_scalar_dict():
    signals = AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=np.array([100], dtype=np.int64),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_event_seq=np.array([2**31 - 1], dtype=np.int64),
        target_names=tuple(_preds()),
        predictions={k: np.array([v], dtype=np.float32) for k, v in _preds().items()},
    )
    assert adverse_predictions_for_row(signals, 0)["bid_touch_filled"] == pytest.approx(0.5)


def test_adverse_and_edge_feature_maps():
    adverse = build_adverse_observation_features(predictions=_preds(), candidate_names=("touch", "inside_1"))
    assert adverse["adverse_bid_touch_valid"] == 1.0
    assert adverse["adverse_bid_inside_1_valid"] == 0.0
    edge = build_executable_edge_observation_features(
        predictions=_preds(),
        candidate_names=("touch",),
        best_bid_tick=99,
        best_ask_tick=101,
        linear_signal=_signal(),
        inventory_qty=0.0,
        config=AdverseRuntimeConfig(candidate_names=("touch",)),
    )
    assert edge["edge_bid_touch_valid"] == 1.0
    assert "edge_bid_touch_attempt_bps" in edge


def test_missing_edge_targets_raises():
    with pytest.raises(ValueError, match="missing adverse-selection predictions"):
        build_executable_edge_observation_features(
            predictions={"bid_touch_filled": 0.5},
            candidate_names=("touch",),
            best_bid_tick=99,
            best_ask_tick=101,
            linear_signal=_signal(),
            inventory_qty=0.0,
            config=AdverseRuntimeConfig(candidate_names=("touch",)),
        )
