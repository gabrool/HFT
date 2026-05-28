"""Per-head feature-set resolution for MMRT linear models.

This module defines and validates the mapping from model head names to stored
feature columns. It does not read storage, compute features, construct targets,
fit preprocessing, train models, evaluate metrics, or parse raw Tardis data.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from mmrt.linear import models as lm
from mmrt.storage import manifest as mf

DEFAULT_HEADS = lm.MODEL_HEADS


def _require_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _coerce_head_name(value: str) -> str:
    head = _require_non_empty_str(value, "head")
    if head not in lm.MODEL_HEADS:
        raise ValueError(f"unknown head: {head}")
    return head


def _coerce_feature_columns(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    cols = tuple(values)
    if not cols:
        raise ValueError(f"{name} must be non-empty")
    seen: set[str] = set()
    out: list[str] = []
    for i, col in enumerate(cols):
        c = _require_non_empty_str(col, f"{name}[{i}]")
        if c in seen:
            raise ValueError(f"{name} must not contain duplicates")
        seen.add(c)
        out.append(c)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class HeadFeatureConfig:
    feature_columns_by_head: Mapping[str, Sequence[str]] | None = None

    def __post_init__(self) -> None:
        if self.feature_columns_by_head is None:
            return
        if not isinstance(self.feature_columns_by_head, Mapping):
            raise ValueError("feature_columns_by_head must be a mapping or None")
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_head, raw_cols in self.feature_columns_by_head.items():
            head = _coerce_head_name(raw_head)
            normalized[head] = _coerce_feature_columns(raw_cols, name=f"feature_columns_by_head[{head!r}]")
        object.__setattr__(self, "feature_columns_by_head", normalized)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_columns_by_head": {
                head: list(cols)
                for head, cols in self.feature_columns_by_head.items()
            } if self.feature_columns_by_head is not None else None
        }


@dataclass(frozen=True, slots=True)
class ResolvedHeadFeatureSets:
    feature_columns_by_head: dict[str, tuple[str, ...]]
    feature_schema_hash: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.feature_columns_by_head, dict):
            raise ValueError("feature_columns_by_head must be dict")
        if set(self.feature_columns_by_head.keys()) != set(lm.MODEL_HEADS):
            raise ValueError("feature_columns_by_head keys must exactly match model heads")
        normalized: dict[str, tuple[str, ...]] = {}
        for head in lm.MODEL_HEADS:
            normalized[head] = _coerce_feature_columns(self.feature_columns_by_head[head], name=f"feature_columns_by_head[{head!r}]")
        object.__setattr__(self, "feature_columns_by_head", normalized)

    @property
    def heads(self) -> tuple[str, ...]:
        return lm.MODEL_HEADS

    def columns_for_head(self, head_name: str) -> tuple[str, ...]:
        return self.feature_columns_by_head[_coerce_head_name(head_name)]

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_schema_hash": self.feature_schema_hash,
            "feature_columns_by_head": {
                head: list(self.feature_columns_by_head[head])
                for head in lm.MODEL_HEADS
            },
            "feature_counts_by_head": {
                head: len(self.feature_columns_by_head[head])
                for head in lm.MODEL_HEADS
            },
        }


def resolve_head_feature_sets(
    manifest: mf.StorageManifest,
    config: HeadFeatureConfig | None = None,
) -> ResolvedHeadFeatureSets:
    if not isinstance(manifest, mf.StorageManifest):
        raise ValueError("manifest must be StorageManifest")
    available = tuple(manifest.feature_columns)
    cfg = config if config is not None else HeadFeatureConfig()
    if not isinstance(cfg, HeadFeatureConfig):
        raise ValueError("config must be HeadFeatureConfig or None")
    overrides = cfg.feature_columns_by_head or {}
    resolved: dict[str, tuple[str, ...]] = {}
    available_set = set(available)
    for head in lm.MODEL_HEADS:
        requested = tuple(overrides.get(head, available))
        requested_set = set(requested)
        missing = [c for c in requested if c not in available_set]
        if missing:
            raise ValueError(f"unknown feature columns for {head}: {missing}")
        cols = tuple(col for col in available if col in requested_set)
        if not cols:
            raise ValueError(f"resolved feature columns for {head} must be non-empty")
        resolved[head] = cols
    return ResolvedHeadFeatureSets(
        feature_columns_by_head=resolved,
        feature_schema_hash=manifest.feature_schema.get("feature_specs_hash"),
    )


__all__ = [
    "HeadFeatureConfig",
    "ResolvedHeadFeatureSets",
    "resolve_head_feature_sets",
]
