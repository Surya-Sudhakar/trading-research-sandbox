"""Outcome-blind Gaussian-mixture baseline for frozen market-state vectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from .model import validate_state_matrix


@dataclass(frozen=True, slots=True)
class GMMDiagnostics:
    n_components: int
    bic: float
    aic: float
    converged: bool
    n_iter: int
    min_occupancy_fraction: float


class GaussianMixtureStateModel:
    """Deterministic full-covariance GMM baseline.

    Input is expected to be the already-fitted discovery scaler output.
    """

    def __init__(
        self,
        n_components: int,
        *,
        random_state: int = 42,
        n_init: int = 5,
        reg_covar: float = 1e-6,
        max_iter: int = 300,
    ) -> None:
        if not 2 <= n_components <= 10:
            raise ValueError("n_components must be between 2 and 10")
        self.n_components = n_components
        self._model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            n_init=n_init,
            reg_covar=reg_covar,
            max_iter=max_iter,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> "GaussianMixtureStateModel":
        values = validate_state_matrix(X)
        if len(values) < self.n_components:
            raise ValueError("observations must be >= n_components")
        self._model.fit(values)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        return self._model.predict(validate_state_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        return self._model.predict_proba(validate_state_matrix(X))

    def diagnostics(self, X: np.ndarray) -> GMMDiagnostics:
        self._require_fit()
        values = validate_state_matrix(X)
        labels = self._model.predict(values)
        counts = np.bincount(labels, minlength=self.n_components)
        return GMMDiagnostics(
            n_components=self.n_components,
            bic=float(self._model.bic(values)),
            aic=float(self._model.aic(values)),
            converged=bool(self._model.converged_),
            n_iter=int(self._model.n_iter_),
            min_occupancy_fraction=float(counts.min() / len(values)),
        )

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("GMM state model has not been fitted")
