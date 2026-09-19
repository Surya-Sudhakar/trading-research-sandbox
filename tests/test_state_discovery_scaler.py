from datetime import datetime, timezone

import numpy as np
import pytest

from sandbox.market_state.discovery import DiscoveryStandardScaler
from sandbox.market_state.universal.market_state_v1 import (
    MARKET_STATE_FEATURE_NAMES_V1,
    MARKET_STATE_FEATURE_SCHEMA_VERSION,
)


def _X():
    return np.vstack([np.arange(14, dtype=float), np.arange(14, dtype=float) + 2])


def test_scaler_fits_only_discovery_and_records_frozen_identity():
    s = DiscoveryStandardScaler().fit(
        _X(),
        [datetime(2016, 1, 1, tzinfo=timezone.utc), datetime(2022, 12, 31, tzinfo=timezone.utc)],
    )
    assert s.metadata.schema_version == MARKET_STATE_FEATURE_SCHEMA_VERSION
    assert s.metadata.feature_names == MARKET_STATE_FEATURE_NAMES_V1
    assert s.metadata.observation_count == 2
    np.testing.assert_allclose(s.transform(_X()).mean(axis=0), 0.0)


@pytest.mark.parametrize("stamp", [
    datetime(2015, 12, 31, tzinfo=timezone.utc),
    datetime(2023, 1, 1, tzinfo=timezone.utc),
])
def test_scaler_rejects_fit_outside_discovery(stamp):
    with pytest.raises(ValueError, match="2016-2022"):
        DiscoveryStandardScaler().fit(_X(), [stamp, stamp])


def test_scaler_rejects_naive_timestamp_and_misalignment():
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        DiscoveryStandardScaler().fit(_X(), [datetime(2020, 1, 1)] * 2)
    with pytest.raises(ValueError, match="one-to-one"):
        DiscoveryStandardScaler().fit(_X(), [datetime(2020, 1, 1, tzinfo=timezone.utc)])


def test_transform_requires_fit_but_does_not_refit():
    s = DiscoveryStandardScaler()
    with pytest.raises(RuntimeError, match="not been fitted"):
        s.transform(_X())
    s.fit(_X(), [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 2)
    before = s.metadata
    shifted = _X() + 1000
    s.transform(shifted)
    assert s.metadata == before
