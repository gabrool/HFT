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


class ExampleCandidate(CandidateFromExisting):
    """Simple example: signed sqrt of the first column."""

    name = "example_signed_sqrt_first_feature"

    def compute(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be 2D with at least one feature column")
        return np.sign(X[:, 0]) * np.sqrt(np.abs(X[:, 0]))
