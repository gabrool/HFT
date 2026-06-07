import numpy as np
import pytest

from mmrt.execution.adverse_signal import (
    ADVERSE_SELECTION_MODEL_SCHEMA,
    AdverseSelectionModelArtifact,
    predict_adverse_selection,
    save_adverse_selection_model,
    load_adverse_selection_model,
    require_adverse_targets_for_executable_edge,
)


def _artifact():
    return AdverseSelectionModelArtifact(
        schema=ADVERSE_SELECTION_MODEL_SCHEMA,
        feature_names=("x",),
        target_names=("bid_touch_filled", "bid_touch_toxic_cost_bps"),
        feature_mean=np.array([0.0]),
        feature_scale=np.array([1.0]),
        coefficients=np.array([[2.0], [-3.0]]),
        intercepts=np.array([0.5, -1.0]),
        config_json="{}",
        exchange="x",
        symbol="y",
    )


def test_adverse_model_roundtrip_and_clipping(tmp_path):
    artifact = _artifact()
    path = tmp_path / "model.npz"
    save_adverse_selection_model(path, artifact)
    loaded = load_adverse_selection_model(path)
    pred = predict_adverse_selection(loaded, np.array([[10.0], [1.0]], dtype=np.float32))
    assert pred["bid_touch_filled"].tolist() == [1.0, 1.0]
    assert pred["bid_touch_toxic_cost_bps"].tolist() == [0.0, 0.0]


def test_adverse_model_rejects_old_schema_and_missing_edge_targets():
    with pytest.raises(ValueError):
        AdverseSelectionModelArtifact(
            schema="mmrt_adverse_selection_ridge",
            feature_names=("x",),
            target_names=("bid_touch_filled",),
            feature_mean=np.array([0.0]),
            feature_scale=np.array([1.0]),
            coefficients=np.array([[1.0]]),
            intercepts=np.array([0.0]),
            config_json="{}",
            exchange="x",
            symbol="y",
        )
    with pytest.raises(ValueError, match="missing adverse-selection targets"):
        require_adverse_targets_for_executable_edge(("bid_touch_filled",), ("touch",))
