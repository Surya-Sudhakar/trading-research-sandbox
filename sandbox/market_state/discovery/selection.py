"""Outcome-blind candidate-state diagnostics for the GMM baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gmm import GaussianMixtureStateModel, GMMDiagnostics
from .model import validate_state_matrix


DEFAULT_COMPONENT_COUNTS = tuple(range(2, 9))


@dataclass(frozen=True, slots=True)
class GMMCandidateResult:
    diagnostics: GMMDiagnostics


def evaluate_gmm_candidates(
    X: np.ndarray,
    component_counts=DEFAULT_COMPONENT_COUNTS,
) -> tuple[GMMCandidateResult, ...]:
    """Fit candidate counts without using any trading/future outcome metric."""
    values = validate_state_matrix(X)
    counts = tuple(int(n) for n in component_counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("component_counts must be non-empty and unique")
    if any(n < 2 or n > 10 for n in counts):
        raise ValueError("candidate component counts must be between 2 and 10")
    results = []
    for n in counts:
        model = GaussianMixtureStateModel(n).fit(values)
        results.append(GMMCandidateResult(model.diagnostics(values)))
    return tuple(results)
