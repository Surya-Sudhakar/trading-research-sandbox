"""Unsupervised market-state discovery models."""

from .model import StateDiscoveryModel, validate_state_matrix
from .scaler import DiscoveryStandardScaler, DiscoveryScalerMetadata
from .gmm import GaussianMixtureStateModel, GMMDiagnostics
from .selection import GMMCandidateResult, evaluate_gmm_candidates
from .stability import GMMStabilityDiagnostics, evaluate_gmm_stability

__all__ = [
    "StateDiscoveryModel", "validate_state_matrix",
    "DiscoveryStandardScaler", "DiscoveryScalerMetadata",
    "GaussianMixtureStateModel", "GMMDiagnostics",
    "GMMCandidateResult", "evaluate_gmm_candidates",
    "GMMStabilityDiagnostics", "evaluate_gmm_stability",
]
