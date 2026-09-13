"""Project ATR semantics in a linear-time batch implementation."""
import math
from sandbox.market_state.volatility import true_ranges, causal_percentile


def atr_series(highs, lows, closes, period=14):
    if period < 1:
        raise ValueError("period must be positive")
    trs = true_ranges(highs, lows, closes)
    result = [None] * len(closes)
    if len(trs) < period:
        return result
    value = sum(trs[:period]) / period
    for i in range(period, len(closes)):
        if i > period:
            value = (value * (period-1) + trs[i-1]) / period
        result[i] = value if math.isfinite(value) and value > 0 else None
    return result


def percentile_series(values, window=100):
    """Inclusive weak rank (0..100); require a full window of valid values."""
    if window < 1:
        raise ValueError("window must be positive")
    result = []
    for i, current in enumerate(values):
        history = values[max(0, i-window+1):i+1]
        ready = len(history) == window and all(x is not None for x in history)
        result.append(causal_percentile(current, history) if ready else None)
    return result

