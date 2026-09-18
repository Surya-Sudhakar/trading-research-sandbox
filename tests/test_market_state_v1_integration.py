from dataclasses import replace
import pytest

from sandbox.market_state.universal.market_state_v1 import (
    MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_CONTEXT_V1,
)
from sandbox.market_state.universal.state_vector import (
    project_mapping_v1, project_market_state_v1, vector_ready_v1,
)
from tests.test_universal_features import bars, compute


def _signature(row):
    return tuple(getattr(row, name) for name in MARKET_STATE_FEATURE_NAMES_V1)


def test_full_14_feature_vector_is_prefix_invariant():
    frame = bars(1200)
    full = compute(frame)
    prefix = compute(frame.iloc[:900])
    assert len(prefix) == 900
    for left, right in zip(full[:900], prefix):
        assert _signature(left) == _signature(right)
        assert tuple(getattr(left, x) for x in MARKET_STATE_CONTEXT_V1) == tuple(
            getattr(right, x) for x in MARKET_STATE_CONTEXT_V1)


def test_future_price_mutation_cannot_change_prior_14_feature_vector():
    frame = bars(1200)
    baseline = compute(frame)
    cut = 900
    mutated = frame.copy()
    mutated.loc[cut:, "open"] *= 1.7
    mutated.loc[cut:, "close"] *= 1.7
    mutated.loc[cut:, "high"] = mutated.loc[cut:, ["open", "close"]].max(axis=1) * 1.001
    mutated.loc[cut:, "low"] = mutated.loc[cut:, ["open", "close"]].min(axis=1) * 0.999
    changed = compute(mutated)
    for i in range(cut):
        assert _signature(baseline[i]) == _signature(changed[i])


def test_state_readiness_is_separate_from_legacy_universal_readiness():
    rows = compute(bars(1200))
    assert rows[113].feature_ready
    assert not vector_ready_v1(rows[113])
    ready = [r for r in rows if vector_ready_v1(r)]
    assert ready
    projected = project_market_state_v1(ready[0])
    assert projected is not None
    assert len(projected.values) == 14
    assert projected.values == tuple(float(getattr(ready[0], x)) for x in MARKET_STATE_FEATURE_NAMES_V1)


def test_state_vector_rejects_outcome_and_future_columns():
    row = next(r for r in compute(bars(1200)) if vector_ready_v1(r))
    mapping = row.model_dump()
    assert len(project_mapping_v1(mapping)) == 14
    for forbidden in ("win", "loss", "result", "tp", "sl", "mfe", "mae",
                      "future_return", "future_high", "future_low", "trade_profitability"):
        contaminated = dict(mapping)
        contaminated[forbidden] = 1
        with pytest.raises(ValueError, match="forbidden"):
            project_mapping_v1(contaminated)


def test_state_projection_is_exactly_the_frozen_14_not_context_or_legacy_fields():
    row = next(r for r in compute(bars(1200)) if vector_ready_v1(r))
    vector = project_market_state_v1(row)
    assert vector is not None and len(vector.values) == 14
    assert len(vector.context) == len(MARKET_STATE_CONTEXT_V1)
    assert "atr_14" not in MARKET_STATE_FEATURE_NAMES_V1
    assert "completed_h1_direction" not in MARKET_STATE_FEATURE_NAMES_V1
    assert "completed_h3_direction" not in MARKET_STATE_FEATURE_NAMES_V1
