"""Single causal decision-feature pipeline for the MMRT feature layer.

This module composes the :class:`FeatureEngine` with the mandatory
:class:`CausalFeatureTransformer` stage so every consumer (supervised dataset
ingest, execution signal building, feature audits) produces identical
transformed decision features from the same event stream. Raw engine vectors
are an internal intermediate and are never part of pipeline output.

This module does not parse market data, read execution tapes, build labels,
or write storage artifacts.
"""

from dataclasses import dataclass

import numpy as np

from mmrt.features.engine import (
    DECISION_STRIDE_US,
    DEFAULT_EVENT_HISTORY_CAPACITY,
    FeatureEngine,
    FeatureEngineConfig,
)
from mmrt.features.book_state import BookSnapshotInput
from mmrt.features.trade_state import TradeInput
from mmrt.features.transforms import (
    CausalFeatureTransformer,
    TransformConfig,
    TransformDiagnostics,
)
from mmrt.features.specs import FEATURE_COUNT


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


@dataclass(frozen=True, slots=True)
class FeaturePipelineConfig:
    decision_stride_us: int = DECISION_STRIDE_US
    transform: TransformConfig = TransformConfig()
    event_history_capacity: int = DEFAULT_EVENT_HISTORY_CAPACITY

    def __post_init__(self) -> None:
        _require_positive_int(self.decision_stride_us, "decision_stride_us")
        if not isinstance(self.transform, TransformConfig):
            raise ValueError("transform must be TransformConfig")
        _require_positive_int(self.event_history_capacity, "event_history_capacity")

    def transform_identity(self) -> dict[str, object]:
        """Stable transform identity payload recorded in downstream artifacts."""
        return dict(self.transform.as_dict())


@dataclass(frozen=True, slots=True)
class TransformedDecision:
    decision_index: int
    local_ts_us: int
    ts_us: int
    event_seq: int
    raw_mid: float
    feature_values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.feature_values)
        if arr.ndim != 1 or arr.shape[0] != FEATURE_COUNT:
            raise ValueError("feature_values must have shape (FEATURE_COUNT,)")
        if not np.isfinite(arr).all():
            raise ValueError("feature_values must be finite")
        object.__setattr__(self, "feature_values", np.ascontiguousarray(arr))


class DecisionFeaturePipeline:
    """Causal event stream -> transformed decision feature rows.

    The transform stage is not optional: decision feature output of this
    pipeline is always :class:`CausalFeatureTransformer` output, matching the
    feature columns persisted by supervised dataset ingest.
    """

    def __init__(self, config: FeaturePipelineConfig | None = None) -> None:
        self.config = config or FeaturePipelineConfig()
        self.engine = FeatureEngine(
            FeatureEngineConfig(
                decision_stride_us=self.config.decision_stride_us,
                event_history_capacity=self.config.event_history_capacity,
            )
        )
        self.transformer = CausalFeatureTransformer(self.config.transform)

    def reset(self) -> None:
        self.engine.reset()
        self.transformer.reset()

    def is_ready(self) -> bool:
        return self.engine.is_ready()

    def current_mid(self) -> float:
        return float(self.engine.book_state.current_summary().mid)

    def on_trade(self, trade: TradeInput) -> None:
        self.engine.on_trade(trade)

    def on_book_snapshot(self, snapshot: BookSnapshotInput) -> TransformedDecision | None:
        decision = self.engine.on_book_snapshot(snapshot)
        if decision is None:
            return None
        transformed = self.transformer.transform_one_local(
            decision.local_ts_us, decision.feature_vector
        )
        return TransformedDecision(
            decision_index=decision.decision_index,
            local_ts_us=decision.local_ts_us,
            ts_us=decision.ts_us,
            event_seq=decision.event_seq,
            raw_mid=decision.raw_mid,
            feature_values=transformed,
        )

    def transform_diagnostics_snapshot(self) -> TransformDiagnostics:
        return self.transformer.diagnostics_snapshot()


__all__ = [
    "FeaturePipelineConfig",
    "TransformedDecision",
    "DecisionFeaturePipeline",
]
