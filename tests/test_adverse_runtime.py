from pathlib import Path

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
    adverse = build_adverse_observation_features(predictions=_preds(), config=AdverseRuntimeConfig(candidate_names=("touch", "inside_1")))
    assert adverse["adverse_bid_touch_valid"] == 1.0
    assert adverse["adverse_bid_inside_1_valid"] == 0.0
    edge = build_executable_edge_observation_features(
        predictions=_preds(),
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
            best_bid_tick=99,
            best_ask_tick=101,
            linear_signal=_signal(),
            inventory_qty=0.0,
            config=AdverseRuntimeConfig(candidate_names=("touch",)),
        )


def test_adverse_runtime_config_rejects_non_integer_post_only_gap():
    with pytest.raises(ValueError, match="post_only_gap_ticks"):
        AdverseRuntimeConfig(post_only_gap_ticks=1.7)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="post_only_gap_ticks"):
        AdverseRuntimeConfig(post_only_gap_ticks=True)  # type: ignore[arg-type]


def test_adverse_runtime_config_rejects_duplicate_or_malformed_candidates():
    with pytest.raises(ValueError, match="duplicate quote candidate"):
        AdverseRuntimeConfig(candidate_names=("touch", "touch"))

    with pytest.raises(ValueError, match="malformed quote candidate"):
        AdverseRuntimeConfig(candidate_names=("inside_x",))


def test_adverse_runtime_config_stores_parsed_candidate_configs():
    cfg = AdverseRuntimeConfig(candidate_names=("touch", "inside_1"))
    assert tuple(c.name for c in cfg.candidate_configs) == ("touch", "inside_1")
    assert cfg.candidate_names == ("touch", "inside_1")


def test_adverse_runtime_helpers_do_not_reparse_candidates_per_call():
    source = Path("mmrt/execution/adverse_runtime.py").read_text()
    adverse_body = source.split("def build_adverse_observation_features", 1)[1].split("def build_executable_edge_observation_features", 1)[0]
    edge_body = source.split("def build_executable_edge_observation_features", 1)[1].split("__all__", 1)[0]
    assert "quote_candidate_configs_from_names" not in adverse_body
    assert "quote_candidate_configs_from_names" not in edge_body


def test_edge_candidate_validity_respects_runtime_post_only_gap():
    preds = {
        "bid_inside_1_filled": 0.5,
        "ask_inside_1_filled": 0.5,
        "bid_inside_1_toxic_cost_bps": 0.0,
        "ask_inside_1_toxic_cost_bps": 0.0,
    }

    gap1 = build_executable_edge_observation_features(
        predictions=preds,
        best_bid_tick=1000,
        best_ask_tick=1002,
        linear_signal=_signal(),
        inventory_qty=0.0,
        config=AdverseRuntimeConfig(candidate_names=("inside_1",), post_only_gap_ticks=1),
    )
    assert gap1["edge_bid_inside_1_valid"] == 1.0
    assert gap1["edge_ask_inside_1_valid"] == 1.0

    gap2 = build_executable_edge_observation_features(
        predictions=preds,
        best_bid_tick=1000,
        best_ask_tick=1002,
        linear_signal=_signal(),
        inventory_qty=0.0,
        config=AdverseRuntimeConfig(candidate_names=("inside_1",), post_only_gap_ticks=2),
    )
    assert gap2["edge_bid_inside_1_valid"] == 0.0
    assert gap2["edge_ask_inside_1_valid"] == 0.0
