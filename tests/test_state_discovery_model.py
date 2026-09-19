import numpy as np
import pytest

from sandbox.market_state.discovery import StateDiscoveryModel, validate_state_matrix


def test_validate_state_matrix_accepts_exact_frozen_width():
    X = np.zeros((3, 14))
    checked = validate_state_matrix(X)
    assert checked.shape == (3, 14)
    assert checked.dtype == float


@pytest.mark.parametrize("X", [
    np.zeros((3, 13)),
    np.zeros((3, 15)),
])
def test_validate_state_matrix_rejects_non_v1_width(X):
    with pytest.raises(ValueError, match="exactly 14"):
        validate_state_matrix(X)


def test_validate_state_matrix_rejects_nonfinite_and_wrong_rank():
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_state_matrix(np.zeros(14))
    X = np.zeros((3, 14)); X[1, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_state_matrix(X)


def test_protocol_is_runtime_checkable():
    class Dummy:
        def fit(self, X): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
        def predict_proba(self, X): return np.ones((len(X), 1))
        def save(self, path): pass
    assert isinstance(Dummy(), StateDiscoveryModel)
