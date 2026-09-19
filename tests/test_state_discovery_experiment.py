from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sandbox.market_state.discovery.experiment import analyze_discovery_frame
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1


def _frame(rows=160):
    rng = np.random.default_rng(17)
    timestamps = pd.date_range("2016-01-01", periods=rows, freq="15min", tz="UTC")
    data = {"timestamp": timestamps}
    for i, name in enumerate(MARKET_STATE_FEATURE_NAMES_V1):
        # Two clear states, with all 14 frozen fields populated.
        state = np.arange(rows) % 2
        data[name] = state * 5.0 + rng.normal(0, 0.1, rows) + i * 0.01
    return pd.DataFrame(data)


def test_experiment_uses_exact_frozen_vector_and_is_outcome_blind():
    frame = _frame()
    frame["win"] = np.arange(len(frame)) % 2
    result = analyze_discovery_frame(frame, component_counts=(2,))
    assert result["outcome_blind"] is True
    assert tuple(result["feature_names"]) == MARKET_STATE_FEATURE_NAMES_V1
    assert result["observation_count"] == len(frame)
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["n_components"] == 2
    assert np.isfinite(candidate["bic"])
    assert 0.0 <= candidate["mean_max_posterior"] <= 1.0
    assert 0.0 <= candidate["p10_max_posterior"] <= 1.0


def test_experiment_filters_outside_discovery_and_incomplete_rows():
    frame = _frame(80)
    outside = frame.iloc[[0]].copy()
    outside["timestamp"] = datetime(2023, 1, 1, tzinfo=timezone.utc)
    frame.loc[1, MARKET_STATE_FEATURE_NAMES_V1[0]] = np.nan
    combined = pd.concat([frame, outside], ignore_index=True)
    result = analyze_discovery_frame(combined, component_counts=(2,))
    assert result["observation_count"] == 79


def test_experiment_rejects_missing_frozen_feature():
    frame = _frame().drop(columns=[MARKET_STATE_FEATURE_NAMES_V1[-1]])
    with pytest.raises(ValueError, match="missing discovery columns"):
        analyze_discovery_frame(frame, component_counts=(2,))
