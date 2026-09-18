"""Causal, dimensionless market-state measurements for completed bars."""
from __future__ import annotations
import math


def efficiency_ratio(closes, index: int, window: int):
    """Absolute displacement divided by path length across N completed intervals."""
    if window < 1:
        raise ValueError("window must be positive")
    if index < window:
        return None
    path = sum(abs(closes[j] - closes[j-1]) for j in range(index-window+1, index+1))
    if path == 0:
        return 0.0
    value = abs(closes[index] - closes[index-window]) / path
    return value if math.isfinite(value) else None


def realized_volatility(closes, index: int, window: int):
    """Square root of summed squared log returns; deliberately not annualized."""
    if window < 1:
        raise ValueError("window must be positive")
    if index < window:
        return None
    value = math.sqrt(sum(math.log(closes[j] / closes[j-1]) ** 2
                          for j in range(index-window+1, index+1)))
    return value if math.isfinite(value) else None


def atr_change(atr, index: int, lag: int):
    if lag < 1:
        raise ValueError("lag must be positive")
    if index < lag or atr[index] is None or atr[index-lag] is None or atr[index-lag] <= 0:
        return None
    value = atr[index] / atr[index-lag] - 1.0
    return value if math.isfinite(value) else None


def normalized_regression_slope(closes, index: int, window: int, current_atr):
    """OLS close slope per bar divided by current ATR."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if index + 1 < window or current_atr is None or current_atr <= 0:
        return None
    y = closes[index-window+1:index+1]
    x_mean = (window - 1) / 2.0
    y_mean = sum(y) / window
    denominator = sum((x-x_mean) ** 2 for x in range(window))
    slope = sum((x-x_mean) * (y[x]-y_mean) for x in range(window)) / denominator
    value = slope / current_atr
    return value if math.isfinite(value) else None


def prior_range_position(highs, lows, close, index: int, window: int):
    """Position of current close in prior range; current high/low are excluded."""
    if window < 1:
        raise ValueError("window must be positive")
    if index < window:
        return None
    high = max(highs[index-window:index])
    low = min(lows[index-window:index])
    span = high - low
    if span == 0:
        return 0.5
    value = (close - low) / span
    return value if math.isfinite(value) else None
