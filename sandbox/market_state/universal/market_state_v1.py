"""Frozen specification for the first unsupervised market-state experiment.

This module contains *definitions only*.  It deliberately does not fit, scale,
cluster, or inspect trading outcomes.  V1 changes require a new schema version.
"""
from __future__ import annotations

from dataclasses import dataclass


MARKET_STATE_FEATURE_SCHEMA_VERSION = "MARKET_STATE_FEATURES_V1"


@dataclass(frozen=True, slots=True)
class MarketStateFeatureSpec:
    name: str
    source_timeframe: str
    family: str
    lookback: int
    description: str


MARKET_STATE_FEATURES_V1: tuple[MarketStateFeatureSpec, ...] = (
    MarketStateFeatureSpec("er_4", "M15", "efficiency", 4,
                           "Efficiency ratio over 4 completed M15 bars; unsigned path efficiency."),
    MarketStateFeatureSpec("er_16", "M15", "efficiency", 16,
                           "Efficiency ratio over 16 completed M15 bars; unsigned path efficiency."),
    MarketStateFeatureSpec("rv_16", "M15", "volatility", 16,
                           "Square root of summed squared log returns over 16 completed M15 bars; not annualized."),
    MarketStateFeatureSpec("atr_change_1", "M15", "volatility_transition", 15,
                           "Wilder ATR14[t] / ATR14[t-1] - 1."),
    MarketStateFeatureSpec("atr_change_8", "M15", "volatility_transition", 22,
                           "Wilder ATR14[t] / ATR14[t-8] - 1."),
    MarketStateFeatureSpec("slope_atr_4", "M15", "trend", 15,
                           "OLS close slope over 4 completed M15 bars divided by current Wilder ATR14."),
    MarketStateFeatureSpec("slope_atr_16", "M15", "trend", 16,
                           "OLS close slope over 16 completed M15 bars divided by current Wilder ATR14."),
    MarketStateFeatureSpec("range_pos_16", "M15", "structure", 17,
                           "Current close position versus prior 16-bar high/low range; current bar excluded from range."),
    MarketStateFeatureSpec("h1_er_8", "H1", "efficiency", 8,
                           "Efficiency ratio over 8 strictly completed H1 bars."),
    MarketStateFeatureSpec("h1_atr_change_4", "H1", "volatility_transition", 18,
                           "Completed-H1 Wilder ATR14[t] / ATR14[t-4] - 1."),
    MarketStateFeatureSpec("h1_slope_atr_8", "H1", "trend", 15,
                           "OLS close slope over 8 strictly completed H1 bars divided by completed-H1 ATR14."),
    MarketStateFeatureSpec("h3_er_8", "H3", "efficiency", 8,
                           "Efficiency ratio over 8 strictly completed H3 bars."),
    MarketStateFeatureSpec("h3_atr_change_4", "H3", "volatility_transition", 18,
                           "Completed-H3 Wilder ATR14[t] / ATR14[t-4] - 1."),
    MarketStateFeatureSpec("h3_slope_atr_8", "H3", "trend", 15,
                           "OLS close slope over 8 strictly completed H3 bars divided by completed-H3 ATR14."),
)

MARKET_STATE_FEATURE_NAMES_V1: tuple[str, ...] = tuple(x.name for x in MARKET_STATE_FEATURES_V1)

# Context is attached to rows but is not part of the first GMM numeric vector.
MARKET_STATE_CONTEXT_V1: tuple[str, ...] = (
    "analysis_hour",
    "weekday",
    "london_active",
    "new_york_active",
    "london_new_york_overlap",
)

# Explicit leakage boundary.  These names are documentation + testable policy;
# downstream dataset builders must reject outcome columns from clustering input.
FORBIDDEN_STATE_DISCOVERY_FIELDS: frozenset[str] = frozenset({
    "win", "loss", "result", "tp", "sl", "mfe", "mae",
    "future_return", "future_high", "future_low", "trade_profitability",
})
