"""Outcome-blind stability diagnostics for candidate GMM state counts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .gmm import GaussianMixtureStateModel
from .model import validate_state_matrix


DEFAULT_STABILITY_SEEDS = (7, 19, 42, 73, 101)


@dataclass(frozen=True, slots=True)
class GMMStabilityDiagnostics:
    n_components: int
    seed_ari_mean: float
    seed_ari_min: float
    time_slice_ari_mean: float
    time_slice_ari_min: float
    time_slices: int


def evaluate_gmm_stability(
    X: np.ndarray,
    n_components: int,
    *,
    seeds=DEFAULT_STABILITY_SEEDS,
    time_slices: int = 4,
) -> GMMStabilityDiagnostics:
    """Measure label-invariant seed and chronological-slice stability.

    No outcomes or future returns are accepted. Each chronological slice is fit
    independently and compared on the same full discovery feature matrix.
    """
    values = validate_state_matrix(X)
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must contain at least two unique values")
    if time_slices < 2:
        raise ValueError("time_slices must be at least 2")
    if len(values) < n_components * time_slices:
        raise ValueError("insufficient observations for requested time slices")

    reference = GaussianMixtureStateModel(
        n_components, random_state=seed_values[0]
    ).fit(values)
    reference_labels = reference.predict(values)

    seed_scores = []
    for seed in seed_values[1:]:
        labels = GaussianMixtureStateModel(
            n_components, random_state=seed
        ).fit(values).predict(values)
        seed_scores.append(float(adjusted_rand_score(reference_labels, labels)))

    slice_scores = []
    for indices in np.array_split(np.arange(len(values)), time_slices):
        slice_values = values[indices]
        if len(slice_values) < n_components:
            raise ValueError("each time slice must contain at least n_components observations")
        labels = GaussianMixtureStateModel(
            n_components, random_state=seed_values[0]
        ).fit(slice_values).predict(values)
        slice_scores.append(float(adjusted_rand_score(reference_labels, labels)))

    return GMMStabilityDiagnostics(
        n_components=n_components,
        seed_ari_mean=float(np.mean(seed_scores)),
        seed_ari_min=float(np.min(seed_scores)),
        time_slice_ari_mean=float(np.mean(slice_scores)),
        time_slice_ari_min=float(np.min(slice_scores)),
        time_slices=time_slices,
    )
