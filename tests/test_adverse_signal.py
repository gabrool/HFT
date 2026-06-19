import numpy as np
import pytest

from mmrt.execution.adverse_signal import (
    ADVERSE_SELECTION_MODEL_SCHEMA,
    ADVERSE_SELECTION_SIGNALS_SCHEMA,
    AdverseSelectionModelArtifact,
    AdverseSelectionSignalArtifact,
    predict_adverse_selection,
    save_adverse_selection_model,
    load_adverse_selection_model,
    require_adverse_targets_for_executable_edge,
)
from tests.grid_helpers import adverse_split_contract_fields, grid_lineage_fields


def _grid_kwargs(n_rows: int = 1):
    return grid_lineage_fields(n_rows=n_rows)


def _model_lineage_kwargs(n_rows: int = 1):
    return {
        **grid_lineage_fields(n_rows=n_rows),
        **adverse_split_contract_fields(n_rows=n_rows),
    }


def _label_config(queue_mode: str = "conservative") -> dict[str, object]:
    return {
        "queue_mode": queue_mode,
        "l2_decrease_weight": 0.25,
        "trade_at_level_weight": 0.5,
        "dedupe_l2_decrease_with_trade_prints": True,
        "unknown_level_queue_ahead_qty": 1_000_000_000.0,
        "order_entry_latency_us": 500,
        "decision_compute_latency_us": 50,
        "post_only_gap_ticks": 1,
        "order_qty": 0.001,
        "fill_horizon_us": 1_000_000,
        "adverse_horizon_us": 1_000_000,
    }


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
        **_model_lineage_kwargs(),
    )


def test_adverse_model_roundtrip_and_clipping(tmp_path):
    artifact = _artifact()
    path = tmp_path / "model.npz"
    save_adverse_selection_model(path, artifact)
    loaded = load_adverse_selection_model(path)
    pred = predict_adverse_selection(loaded, np.array([[10.0], [1.0]], dtype=np.float32))
    assert pred["bid_touch_filled"].tolist() == [1.0, 1.0]
    assert pred["bid_touch_toxic_cost_bps"].tolist() == [0.0, 0.0]


def test_adverse_model_requires_current_schema_and_edge_targets():
    with pytest.raises(ValueError):
        AdverseSelectionModelArtifact(
            schema="not_current",
            feature_names=("x",),
            target_names=("bid_touch_filled",),
            feature_mean=np.array([0.0]),
            feature_scale=np.array([1.0]),
            coefficients=np.array([[1.0]]),
            intercepts=np.array([0.0]),
            config_json="{}",
            exchange="x",
            symbol="y",
            **_model_lineage_kwargs(),
        )
    with pytest.raises(ValueError, match="missing adverse-selection targets"):
        require_adverse_targets_for_executable_edge(("bid_touch_filled",), ("touch",))

def test_adverse_signal_artifact_rejects_missing_prediction_key():
    with pytest.raises(ValueError, match="missing prediction array"):
        AdverseSelectionSignalArtifact(
            schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
            decision_local_ts_us=np.array([100], dtype=np.int64),
            decision_event_index=np.array([0], dtype=np.int64),
            decision_event_seq=np.array([2**31 - 1], dtype=np.int64),
            target_names=("bid_touch_filled",),
            predictions={},
            adverse_label_config=_label_config(),
            **_grid_kwargs(),
        )


def test_adverse_signal_artifact_rejects_non_mapping_predictions():
    with pytest.raises(ValueError, match="predictions must be a mapping"):
        AdverseSelectionSignalArtifact(
            schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
            decision_local_ts_us=np.array([100], dtype=np.int64),
            decision_event_index=np.array([0], dtype=np.int64),
            decision_event_seq=np.array([0], dtype=np.int64),
            target_names=("bid_touch_filled",),
            predictions=[],  # type: ignore[arg-type]
            adverse_label_config=_label_config(),
            **_grid_kwargs(),
        )


def test_adverse_signal_artifact_accepts_same_local_ts_with_increasing_event_seq():
    signals = AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        decision_local_ts_us=np.array([100, 100], dtype=np.int64),
        decision_event_index=np.array([0, 1], dtype=np.int64),
        decision_event_seq=np.array([7, 8], dtype=np.int64),
        target_names=("bid_touch_filled",),
        predictions={"bid_touch_filled": np.array([0.25, 0.75], dtype=np.float32)},
        adverse_label_config=_label_config(),
        **_grid_kwargs(n_rows=2),
    )
    np.testing.assert_array_equal(signals.decision_local_ts_us, np.array([100, 100], dtype=np.int64))


@pytest.mark.parametrize("event_seq", ([7, 7], [8, 7]))
def test_adverse_signal_artifact_rejects_same_local_ts_without_increasing_event_seq(event_seq):
    with pytest.raises(ValueError, match="decision event key must be strictly increasing"):
        AdverseSelectionSignalArtifact(
            schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
            decision_local_ts_us=np.array([100, 100], dtype=np.int64),
            decision_event_index=np.array([0, 1], dtype=np.int64),
            decision_event_seq=np.asarray(event_seq, dtype=np.int64),
            target_names=("bid_touch_filled",),
            predictions={"bid_touch_filled": np.array([0.25, 0.75], dtype=np.float32)},
            adverse_label_config=_label_config(),
            **_grid_kwargs(n_rows=2),
        )


from mmrt.execution.adverse_signal import load_adverse_selection_signals


def test_load_adverse_selection_signals_rejects_missing_prediction_array(tmp_path):
    path = tmp_path / "bad_signals.npz"
    np.savez(
        path,
        schema=np.array(ADVERSE_SELECTION_SIGNALS_SCHEMA),
        decision_local_ts_us=np.array([100], dtype=np.int64),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_event_seq=np.array([0], dtype=np.int64),
        target_names=np.asarray(["bid_touch_filled"], dtype=object),
        decision_grid_schema=np.array(_grid_kwargs()["decision_grid_schema"]),
        decision_grid_hash=np.array(_grid_kwargs()["decision_grid_hash"]),
        decision_grid_n_rows=np.array(1, dtype=np.int64),
        decision_schedule=np.array("{}"),
        adverse_label_config=np.array(_label_config(), dtype=object),
    )
    with pytest.raises(ValueError, match="missing prediction arrays"):
        load_adverse_selection_signals(path)


def test_load_adverse_selection_signals_rejects_missing_base_arrays(tmp_path):
    path = tmp_path / "bad_signals.npz"
    np.savez(path, schema=np.array(ADVERSE_SELECTION_SIGNALS_SCHEMA))
    with pytest.raises(ValueError, match="missing required arrays"):
        load_adverse_selection_signals(path)

from mmrt.execution.adverse_selection import AdverseSelectionConfig, AdverseSelectionFeatureDataset
from mmrt.execution.adverse_signal import build_adverse_selection_signal_artifact, validate_decision_grid_alignment


def test_build_adverse_selection_signal_artifact_accepts_feature_dataset():
    model = _artifact()
    dataset = AdverseSelectionFeatureDataset(
        decision_local_ts_us=np.array([100], dtype=np.int64),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_event_seq=np.array([2**31 - 1], dtype=np.int64),
        feature_names=("x",),
        features=np.array([[1.0]], dtype=np.float32),
        config=AdverseSelectionConfig(),
        **_grid_kwargs(),
    )
    signals = build_adverse_selection_signal_artifact(dataset, model)
    assert signals.decision_local_ts_us.tolist() == [100]


def test_validate_decision_grid_alignment_rejects_mismatch():
    grid = dict(
        left_local_ts_us=np.array([100], dtype=np.int64),
        left_event_index=np.array([0], dtype=np.int64),
        left_event_seq=np.array([1], dtype=np.int64),
        right_local_ts_us=np.array([101], dtype=np.int64),
        right_event_index=np.array([0], dtype=np.int64),
        right_event_seq=np.array([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="first mismatch"):
        validate_decision_grid_alignment(**grid, left_name="adverse_signals", right_name="linear_signals")

from mmrt.execution.adverse_signal import save_adverse_selection_signals, save_adverse_selection_signals_arrays


def test_save_adverse_selection_signals_arrays_matches_existing_artifact_writer_tiny(tmp_path):
    kwargs = dict(
        decision_local_ts_us=np.array([1, 2], dtype=np.int64),
        decision_event_index=np.array([0, 1], dtype=np.int64),
        decision_event_seq=np.array([10, 11], dtype=np.int64),
        target_names=("bid_touch_filled", "bid_touch_toxic_cost_bps"),
        predictions={
            "bid_touch_filled": np.array([0.25, 0.75], dtype=np.float32),
            "bid_touch_toxic_cost_bps": np.array([1.0, 2.0], dtype=np.float32),
        },
    )
    artifact = AdverseSelectionSignalArtifact(
        schema=ADVERSE_SELECTION_SIGNALS_SCHEMA,
        adverse_label_config=_label_config(),
        **kwargs,
        **_grid_kwargs(n_rows=2),
    )
    old_path = tmp_path / "old.npz"
    new_path = tmp_path / "new.npz"
    save_adverse_selection_signals(old_path, artifact)
    save_adverse_selection_signals_arrays(new_path, **kwargs, adverse_label_config=_label_config(), **_grid_kwargs(n_rows=2), validate_chunk_rows=1)
    old = load_adverse_selection_signals(old_path)
    new = load_adverse_selection_signals(new_path)
    np.testing.assert_array_equal(new.decision_local_ts_us, old.decision_local_ts_us)
    assert new.target_names == old.target_names
    for name in old.target_names:
        np.testing.assert_array_equal(new.predictions[name], old.predictions[name])


def test_save_adverse_selection_signals_arrays_uses_event_key_order(tmp_path):
    kwargs = dict(
        decision_local_ts_us=np.array([100, 100], dtype=np.int64),
        decision_event_index=np.array([0, 1], dtype=np.int64),
        decision_event_seq=np.array([7, 8], dtype=np.int64),
        target_names=("bid_touch_filled",),
        predictions={"bid_touch_filled": np.array([0.25, 0.75], dtype=np.float32)},
    )
    path = tmp_path / "same_ts.npz"
    save_adverse_selection_signals_arrays(path, **kwargs, adverse_label_config=_label_config(), **_grid_kwargs(n_rows=2), validate_chunk_rows=1)
    loaded = load_adverse_selection_signals(path)
    np.testing.assert_array_equal(loaded.decision_local_ts_us, np.array([100, 100], dtype=np.int64))
    bad = dict(kwargs)
    bad["decision_event_seq"] = np.array([8, 7], dtype=np.int64)
    with pytest.raises(ValueError, match="decision event key must be strictly increasing"):
        save_adverse_selection_signals_arrays(tmp_path / "bad_same_ts.npz", **bad, adverse_label_config=_label_config(), **_grid_kwargs(n_rows=2), validate_chunk_rows=1)


def test_save_adverse_selection_signals_arrays_validates_probability_bounds_chunked(tmp_path):
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        save_adverse_selection_signals_arrays(
            tmp_path / "bad.npz",
            decision_local_ts_us=np.array([1, 2], dtype=np.int64),
            decision_event_index=np.array([0, 1], dtype=np.int64),
            decision_event_seq=np.array([0, 0], dtype=np.int64),
            target_names=("bid_touch_filled",),
            predictions={"bid_touch_filled": np.array([0.5, 1.5], dtype=np.float32)},
            adverse_label_config=_label_config(),
            **_grid_kwargs(n_rows=2),
            validate_chunk_rows=1,
        )
