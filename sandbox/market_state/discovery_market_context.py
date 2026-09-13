"""Causal market predictors assembled from existing batch feature primitives."""
from dataclasses import dataclass, replace
from datetime import datetime
import math

import pandas as pd

from .block_aggregation import OHLCBar
from .models import FeatureConfiguration
from .normalization import safe_ratio
from .swings import (
    PivotLabel, StructureBreakKind, confirmed_structure_breaks,
    confirmed_swings, label_confirmed_swings, structure_counts,
)
from .touch_transition import LevelTransitionSemantics
from .universal import UniversalFeatureEngine


@dataclass(frozen=True, slots=True)
class LevelMeasurement:
    close_distance_price: float
    close_distance_atr: float | None
    range_distance_price: float
    range_distance_atr: float | None
    penetration_above_price: float
    penetration_above_atr: float | None
    penetration_below_price: float
    penetration_below_atr: float | None


def measure_level(bar: OHLCBar, level_price: float,
                  atr: float | None = None) -> LevelMeasurement:
    """Measure a completed OHLC bar against any finite numeric level."""
    if not isinstance(bar, OHLCBar):
        raise TypeError("bar must be OHLCBar")
    level = float(level_price)
    if not math.isfinite(level):
        raise ValueError("level_price must be finite")
    if atr is not None and (not math.isfinite(float(atr)) or float(atr) <= 0):
        raise ValueError("atr must be positive and finite when supplied")
    close_distance = float(bar.close) - level
    range_distance = (float(bar.low) - level if bar.low > level else
                      float(bar.high) - level if bar.high < level else 0.0)
    above = max(0.0, float(bar.high) - level)
    below = max(0.0, level - float(bar.low))
    divisor = float(atr) if atr is not None else None
    normalized = lambda value: safe_ratio(value, divisor) if divisor is not None else None
    return LevelMeasurement(
        close_distance, normalized(close_distance),
        range_distance, normalized(range_distance),
        above, normalized(above), below, normalized(below),
    )


@dataclass(frozen=True, slots=True)
class DiscoveryMarketContext:
    decision_time_utc: datetime
    atr: float | None
    atr_percentile: float | None
    range_percentile: float | None
    prior_high_12: float | None
    prior_low_12: float | None
    breakout_high_12: bool | None
    breakout_low_12: bool | None
    higher_high_count: int
    lower_high_count: int
    higher_low_count: int
    lower_low_count: int
    latest_high_label: PivotLabel | None
    latest_high_price: float | None
    latest_high_distance_price: float | None
    latest_high_distance_atr: float | None
    latest_high_bars_since_confirmation: int | None
    latest_low_label: PivotLabel | None
    latest_low_price: float | None
    latest_low_distance_price: float | None
    latest_low_distance_atr: float | None
    latest_low_bars_since_confirmation: int | None
    structure_break: StructureBreakKind | None
    level_transition: LevelTransitionSemantics | None = None


def with_level_transition(context: DiscoveryMarketContext,
                          semantics: LevelTransitionSemantics | None) -> DiscoveryMarketContext:
    if not isinstance(context, DiscoveryMarketContext):
        raise TypeError("context must be DiscoveryMarketContext")
    if semantics is not None and not isinstance(semantics, LevelTransitionSemantics):
        raise TypeError("semantics must be LevelTransitionSemantics")
    return replace(context, level_transition=semantics)


def build_discovery_market_contexts(
    bars, base_minutes: int = 15, *, swing_left_bars: int = 2,
    swing_right_bars: int = 2, swing_count_window: int = 8,
) -> dict[datetime, DiscoveryMarketContext]:
    """Build causal contexts with caller-configurable confirmed-swing parameters."""
    materialized = tuple(bars)
    if not materialized:
        return {}
    if any(not isinstance(bar, OHLCBar) for bar in materialized):
        raise TypeError("bars must contain OHLCBar")
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    for name, value in (
        ("swing_left_bars", swing_left_bars),
        ("swing_right_bars", swing_right_bars),
        ("swing_count_window", swing_count_window),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    frame = pd.DataFrame({
        "timestamp_utc": [bar.open_time_utc for bar in materialized],
        "open": [bar.open for bar in materialized], "high": [bar.high for bar in materialized],
        "low": [bar.low for bar in materialized], "close": [bar.close for bar in materialized],
    })
    configuration = FeatureConfiguration(
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        swing_count_window=swing_count_window,
    )
    rows = UniversalFeatureEngine(f"M{base_minutes}").compute(frame, symbol="DISCOVERY")
    highs = [bar.high for bar in materialized]
    lows = [bar.low for bar in materialized]
    closes = [bar.close for bar in materialized]
    swings = confirmed_swings(highs, lows, configuration.swing_left_bars,
                               configuration.swing_right_bars)
    labeled = label_confirmed_swings(swings)
    labels = {(item.swing.kind, item.swing.index): item.label for item in labeled}
    breaks = {event.confirmation_index: event.kind for event in confirmed_structure_breaks(closes, swings)}
    by_confirmation = {}
    for swing in swings:
        by_confirmation.setdefault(swing.confirmation_index, []).append(swing)
    latest_swing = {"HIGH": None, "LOW": None}
    recent = {"HIGH": [], "LOW": []}
    counts = {"higher_high_count": 0, "lower_high_count": 0,
              "higher_low_count": 0, "lower_low_count": 0}
    result = {}
    for index, (bar, row) in enumerate(zip(materialized, rows)):
        changed = False
        for swing in by_confirmation.get(index, ()):
            latest_swing[swing.kind] = swing
            recent[swing.kind].append(swing)
            recent[swing.kind] = recent[swing.kind][-configuration.swing_count_window:]
            changed = True
        if changed:
            counts = structure_counts(recent["HIGH"] + recent["LOW"],
                                      configuration.swing_count_window)
        def latest(kind):
            swing = latest_swing[kind]
            if swing is None:
                return (None,) * 5
            distance = float(bar.close) - swing.price
            return (labels[(kind, swing.index)], swing.price, distance,
                    safe_ratio(distance, row.atr_14) if row.atr_14 is not None else None,
                    index - swing.confirmation_index)
        high = latest("HIGH")
        low = latest("LOW")
        result[row.timestamp] = DiscoveryMarketContext(
            row.timestamp, row.atr_14, row.atr_percentile_100, row.range_percentile_100,
            row.prior_high_12, row.prior_low_12, row.breakout_high_12, row.breakout_low_12,
            **counts,
            latest_high_label=high[0], latest_high_price=high[1],
            latest_high_distance_price=high[2], latest_high_distance_atr=high[3],
            latest_high_bars_since_confirmation=high[4], latest_low_label=low[0],
            latest_low_price=low[1], latest_low_distance_price=low[2],
            latest_low_distance_atr=low[3], latest_low_bars_since_confirmation=low[4],
            structure_break=breaks.get(index),
        )
    return result
