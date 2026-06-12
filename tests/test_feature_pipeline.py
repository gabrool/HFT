import numpy as np
import pytest

from mmrt.features.book_state import BOOK_DEPTH, BookSnapshotInput
from mmrt.features.engine import FeatureEngine, FeatureEngineConfig
from mmrt.features.pipeline import (
    DecisionFeaturePipeline,
    FeaturePipelineConfig,
    TransformedDecision,
)
from mmrt.features.specs import FEATURE_COUNT, FEATURE_NAMES_HASH, FEATURE_SPECS_HASH
from mmrt.features.trade_state import TradeInput
from mmrt.features.transforms import (
    CausalFeatureTransformer,
    TransformConfig,
    transform_config_from_dict,
)


def _snapshot(local_ts_us: int, *, mid: float = 100.0, event_seq: int = -1) -> BookSnapshotInput:
    bid_px = np.array([mid - 0.1 - 0.1 * i for i in range(BOOK_DEPTH)])
    ask_px = np.array([mid + 0.1 + 0.1 * i for i in range(BOOK_DEPTH)])
    sizes = np.full(BOOK_DEPTH, 1.5)
    return BookSnapshotInput(
        local_ts_us=local_ts_us,
        ts_us=local_ts_us,
        event_seq=event_seq,
        bid_px=bid_px,
        bid_sz=sizes,
        ask_px=ask_px,
        ask_sz=sizes,
    )


def _trade(local_ts_us: int, *, side_code: int = 1) -> TradeInput:
    return TradeInput(local_ts_us=local_ts_us, ts_us=local_ts_us, price=100.0, amount=0.25, side_code=side_code, event_seq=-1)


def _drive(consumer_on_trade, consumer_on_snapshot, *, n_decisions: int, stride_us: int):
    """Feed a deterministic event stream and collect decision outputs."""
    out = []
    ts = 1_000_000
    consumer_on_trade(_trade(ts))
    for i in range(n_decisions):
        ts += stride_us
        consumer_on_trade(_trade(ts - 1, side_code=1 if i % 2 == 0 else -1))
        decision = consumer_on_snapshot(_snapshot(ts, mid=100.0 + 0.01 * (i % 5), event_seq=i))
        if decision is not None:
            out.append(decision)
    return out


def test_pipeline_config_validation():
    with pytest.raises(ValueError):
        FeaturePipelineConfig(decision_stride_us=0)
    with pytest.raises(ValueError):
        FeaturePipelineConfig(transform=None)  # type: ignore[arg-type]
    cfg = FeaturePipelineConfig()
    identity = cfg.transform_identity()
    assert identity["feature_names_hash"] == FEATURE_NAMES_HASH
    assert identity["feature_specs_hash"] == FEATURE_SPECS_HASH


def test_pipeline_output_equals_engine_plus_transformer():
    stride = 500_000
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(decision_stride_us=stride))
    engine = FeatureEngine(FeatureEngineConfig(decision_stride_us=stride))
    transformer = CausalFeatureTransformer(TransformConfig())

    manual: list[np.ndarray] = []

    def manual_on_snapshot(snapshot):
        decision = engine.on_book_snapshot(snapshot)
        if decision is None:
            return None
        manual.append(transformer.transform_one_local(decision.local_ts_us, decision.feature_vector))
        return decision

    pipeline_decisions = _drive(pipeline.on_trade, pipeline.on_book_snapshot, n_decisions=30, stride_us=stride)
    _drive(engine.on_trade, manual_on_snapshot, n_decisions=30, stride_us=stride)

    assert len(pipeline_decisions) == len(manual) > 20
    for got, expected in zip(pipeline_decisions, manual):
        assert isinstance(got, TransformedDecision)
        assert got.feature_values.shape == (FEATURE_COUNT,)
        np.testing.assert_array_equal(got.feature_values, expected)


def test_pipeline_output_is_transformed_not_raw_engine_output():
    stride = 500_000
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(decision_stride_us=stride))
    raw_engine = FeatureEngine(FeatureEngineConfig(decision_stride_us=stride))
    raw_vectors: list[np.ndarray] = []

    def raw_on_snapshot(snapshot):
        decision = raw_engine.on_book_snapshot(snapshot)
        if decision is not None:
            raw_vectors.append(decision.feature_vector)
        return decision

    decisions = _drive(pipeline.on_trade, pipeline.on_book_snapshot, n_decisions=30, stride_us=stride)
    _drive(raw_engine.on_trade, raw_on_snapshot, n_decisions=30, stride_us=stride)

    assert len(decisions) == len(raw_vectors)
    diffs = [
        not np.allclose(d.feature_values.astype(np.float64), raw)
        for d, raw in zip(decisions, raw_vectors)
    ]
    assert all(diffs)


def test_pipeline_reset_restarts_transform_state():
    stride = 500_000
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(decision_stride_us=stride))
    first = _drive(pipeline.on_trade, pipeline.on_book_snapshot, n_decisions=25, stride_us=stride)
    pipeline.reset()
    second = _drive(pipeline.on_trade, pipeline.on_book_snapshot, n_decisions=25, stride_us=stride)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.feature_values, b.feature_values)


def test_pipeline_transform_diagnostics_count_decisions():
    stride = 500_000
    pipeline = DecisionFeaturePipeline(FeaturePipelineConfig(decision_stride_us=stride))
    decisions = _drive(pipeline.on_trade, pipeline.on_book_snapshot, n_decisions=25, stride_us=stride)
    diag = pipeline.transform_diagnostics_snapshot()
    assert diag.rows_seen == len(decisions) > 0


def test_transform_identity_round_trips_through_parser():
    cfg = FeaturePipelineConfig()
    parsed = transform_config_from_dict(cfg.transform_identity())
    assert parsed == cfg.transform


def test_transform_config_from_dict_rejects_hash_drift():
    payload = dict(TransformConfig().as_dict())
    payload["feature_names_hash"] = "0" * 12
    with pytest.raises(ValueError, match="feature_names_hash"):
        transform_config_from_dict(payload)
    payload = dict(TransformConfig().as_dict())
    payload.pop("z_clip")
    with pytest.raises(ValueError, match="missing fields"):
        transform_config_from_dict(payload)
