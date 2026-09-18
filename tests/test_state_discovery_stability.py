import numpy as np
import pytest

from sandbox.market_state.discovery import evaluate_gmm_stability


def _stable_clusters(seed=11):
    rng = np.random.default_rng(seed)
    blocks = []
    for center in (-4.0, 0.0, 4.0):
        # Interleave regimes so every chronological slice contains all states.
        blocks.append(rng.normal(center, 0.20, size=(120, 14)))
    stacked = np.stack(blocks, axis=1).reshape(-1, 14)
    return stacked


def test_stability_is_label_invariant_and_high_for_clear_states():
    result = evaluate_gmm_stability(
        _stable_clusters(), 3, seeds=(7, 19, 42), time_slices=4
    )
    assert result.n_components == 3
    assert result.seed_ari_mean > 0.99
    assert result.seed_ari_min > 0.99
    assert result.time_slice_ari_mean > 0.95
    assert result.time_slice_ari_min > 0.90
    assert result.time_slices == 4


def test_stability_is_deterministic():
    X = _stable_clusters()
    a = evaluate_gmm_stability(X, 3, seeds=(7, 19, 42), time_slices=3)
    b = evaluate_gmm_stability(X, 3, seeds=(7, 19, 42), time_slices=3)
    assert a == b


def test_stability_requires_unique_seeds_and_valid_slices():
    X = _stable_clusters()
    with pytest.raises(ValueError, match="two unique"):
        evaluate_gmm_stability(X, 3, seeds=(7, 7))
    with pytest.raises(ValueError, match="at least 2"):
        evaluate_gmm_stability(X, 3, time_slices=1)


def test_stability_rejects_too_little_data_for_slices():
    X = np.arange(56, dtype=float).reshape(4, 14)
    with pytest.raises(ValueError, match="insufficient observations"):
        evaluate_gmm_stability(X, 3, time_slices=2)
