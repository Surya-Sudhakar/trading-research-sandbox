import numpy as np
import pytest

from sandbox.market_state.discovery import (
    GaussianMixtureStateModel,
    evaluate_gmm_candidates,
)


def _clusters(seed=7):
    rng = np.random.default_rng(seed)
    a = rng.normal(-2.0, 0.25, size=(40, 14))
    b = rng.normal(2.0, 0.25, size=(40, 14))
    return np.vstack([a, b])


def test_gmm_is_deterministic_for_same_data_and_config():
    X = _clusters()
    a = GaussianMixtureStateModel(2).fit(X)
    b = GaussianMixtureStateModel(2).fit(X)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


def test_gmm_probabilities_and_diagnostics_are_well_formed():
    X = _clusters()
    model = GaussianMixtureStateModel(2).fit(X)
    p = model.predict_proba(X)
    assert p.shape == (80, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    d = model.diagnostics(X)
    assert d.converged
    assert 0 < d.min_occupancy_fraction <= 0.5
    assert np.isfinite([d.bic, d.aic]).all()


def test_gmm_requires_fit_and_valid_component_count():
    with pytest.raises(RuntimeError, match="not been fitted"):
        GaussianMixtureStateModel(2).predict(_clusters())
    with pytest.raises(ValueError, match="between 2 and 10"):
        GaussianMixtureStateModel(1)


def test_candidate_evaluation_is_outcome_blind_and_order_preserving():
    X = _clusters()
    results = evaluate_gmm_candidates(X, (2, 3))
    assert tuple(r.diagnostics.n_components for r in results) == (2, 3)
    assert all(np.isfinite(r.diagnostics.bic) for r in results)


def test_candidate_counts_must_be_unique_and_bounded():
    X = _clusters()
    with pytest.raises(ValueError, match="unique"):
        evaluate_gmm_candidates(X, (2, 2))
    with pytest.raises(ValueError, match="between 2 and 10"):
        evaluate_gmm_candidates(X, (2, 11))
