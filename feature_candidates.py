"""Candidate feature plugins for ``feature_lab.py``.

This module contains optional plugin-style candidates computed only from the
existing sampled feature matrix and feature names.
"""

from __future__ import annotations

import numpy as np


class CandidateFromExisting:
    """Base interface for candidate features derived from existing features."""

    name = "candidate_from_existing"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _name_to_idx(feature_names: list[str]) -> dict[str, int]:
    return {str(n): i for i, n in enumerate(feature_names)}


def _first_existing(names: dict[str, int], candidates: list[str]) -> int:
    for c in candidates:
        if c in names:
            return names[c]
    raise KeyError(f"None of candidate feature names found: {candidates}")


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return num / np.maximum(np.abs(den), eps)


class ExampleCandidate(CandidateFromExisting):
    name = "example_signed_sqrt_first_feature"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be 2D with at least one feature column")
        return np.sign(X[:, 0]) * np.sqrt(np.abs(X[:, 0]))


class AbsMicroDislocation(CandidateFromExisting):
    name = "abs_micro_minus_mid_bps"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        names = _name_to_idx(feature_names)
        i = _first_existing(names, ["micro_minus_mid_bps"])
        return np.abs(X[:, i]).astype(np.float32)


class FragilityAbsMicroOverDepth(CandidateFromExisting):
    name = "fragility_abs_micro_over_depth"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        names = _name_to_idx(feature_names)
        micro = X[:, _first_existing(names, ["micro_minus_mid_bps"])]
        depth = X[:, _first_existing(names, ["total_depth_notional_5bps", "depth_notional_5bps", "depth_5bps"])]
        return _safe_div(np.abs(micro), depth).astype(np.float32)


class AbsVwapDislocation200msOverDepth(CandidateFromExisting):
    name = "fragility_abs_vwap_200_over_depth"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        names = _name_to_idx(feature_names)
        vwap = X[:, _first_existing(names, ["vwap_vs_mid_bps_200ms"])]
        depth = X[:, _first_existing(names, ["total_depth_notional_5bps", "depth_notional_5bps", "depth_5bps"])]
        return _safe_div(np.abs(vwap), depth).astype(np.float32)


class SpreadAndDepthQuietness(CandidateFromExisting):
    name = "spread_depth_quietness"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        names = _name_to_idx(feature_names)
        spread = np.abs(X[:, _first_existing(names, ["spread_bps", "spread"])]).astype(np.float32)
        depth = X[:, _first_existing(names, ["total_depth_notional_5bps", "depth_notional_5bps", "depth_5bps"])]
        return _safe_div(depth, 1.0 + spread).astype(np.float32)


class ShortHorizonAbsReturnPressure(CandidateFromExisting):
    name = "short_horizon_abs_return_pressure"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        names = _name_to_idx(feature_names)
        candidates = ["abs_return_bps_200ms", "max_abs_return_bps_200ms", "vwap_vs_mid_bps_200ms", "micro_minus_mid_bps"]
        vals = [np.abs(X[:, names[c]]) for c in candidates if c in names]
        if not vals:
            raise KeyError(f"None of candidate pressure names found: {candidates}")
        return np.max(np.stack(vals, axis=1), axis=1).astype(np.float32)
