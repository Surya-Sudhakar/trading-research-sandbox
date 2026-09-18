from dataclasses import FrozenInstanceError
import pytest

from sandbox.market_state.universal.market_state_v1 import (
    FORBIDDEN_STATE_DISCOVERY_FIELDS,
    MARKET_STATE_CONTEXT_V1,
    MARKET_STATE_FEATURE_NAMES_V1,
    MARKET_STATE_FEATURE_SCHEMA_VERSION,
    MARKET_STATE_FEATURES_V1,
)


EXPECTED = (
    "er_4", "er_16", "rv_16", "atr_change_1", "atr_change_8",
    "slope_atr_4", "slope_atr_16", "range_pos_16",
    "h1_er_8", "h1_atr_change_4", "h1_slope_atr_8",
    "h3_er_8", "h3_atr_change_4", "h3_slope_atr_8",
)


def test_market_state_v1_schema_is_exact_and_frozen():
    assert MARKET_STATE_FEATURE_SCHEMA_VERSION == "MARKET_STATE_FEATURES_V1"
    assert MARKET_STATE_FEATURE_NAMES_V1 == EXPECTED
    assert len(MARKET_STATE_FEATURES_V1) == 14
    assert len(set(MARKET_STATE_FEATURE_NAMES_V1)) == 14
    assert {x.source_timeframe for x in MARKET_STATE_FEATURES_V1} == {"M15", "H1", "H3"}
    assert all(x.lookback > 0 and x.family and x.description for x in MARKET_STATE_FEATURES_V1)
    with pytest.raises(FrozenInstanceError):
        MARKET_STATE_FEATURES_V1[0].name = "changed"


def test_time_context_is_metadata_not_gmm_core():
    assert not set(MARKET_STATE_CONTEXT_V1) & set(MARKET_STATE_FEATURE_NAMES_V1)
    assert MARKET_STATE_CONTEXT_V1 == (
        "analysis_hour", "weekday", "london_active", "new_york_active",
        "london_new_york_overlap",
    )


def test_outcome_leakage_boundary_is_explicit():
    required = {"win", "loss", "tp", "sl", "mfe", "mae", "future_return"}
    assert required <= FORBIDDEN_STATE_DISCOVERY_FIELDS
    assert not FORBIDDEN_STATE_DISCOVERY_FIELDS & set(MARKET_STATE_FEATURE_NAMES_V1)
