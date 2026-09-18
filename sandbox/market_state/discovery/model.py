"""Frozen protocol boundary for unsupervised market-state discovery.

Discovery models consume only the audited MARKET_STATE_FEATURES_V1 matrix.
Outcome/trade fields remain prohibited upstream by state_vector.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StateDiscoveryModel(Protocol):
    """Minimal contract shared by GMM and later temporal alternatives."""

    def fit(self, X: np.ndarray) -> "StateDiscoveryModel": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


def validate_state_matrix(X: np.ndarray, *, expected_features: int = 14) -> np.ndarray:
    """Require a finite, two-dimensional frozen-feature matrix."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("state discovery matrix must be two-dimensional")
    if values.shape[1] != expected_features:
        raise ValueError(f"state discovery requires exactly {expected_features} features")
    if values.shape[0] < 2:
        raise ValueError("state discovery requires at least two observations")
    if not np.isfinite(values).all():
        raise ValueError("state discovery matrix must contain only finite values")
    return values
