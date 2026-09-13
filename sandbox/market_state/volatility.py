from __future__ import annotations
import math
import statistics


def true_ranges(highs, lows, closes) -> list[float]:
    return [max(float(highs[i])-float(lows[i]), abs(float(highs[i])-float(closes[i-1])), abs(float(lows[i])-float(closes[i-1]))) for i in range(1, len(closes))]


def wilder_atr(highs, lows, closes, period: int) -> float | None:
    values = true_ranges(highs, lows, closes)
    if len(values) < period:
        return None
    atr = sum(values[:period]) / period
    for value in values[period:]:
        atr = (atr * (period-1) + value) / period
    return atr if math.isfinite(atr) and atr > 0 else None


def trailing_atr_series(highs, lows, closes, period: int) -> list[float | None]:
    return [wilder_atr(highs[:i], lows[:i], closes[:i], period) for i in range(1, len(closes)+1)]


def causal_percentile(current: float, historical: list[float]) -> float | None:
    values = [float(x) for x in historical if x is not None and math.isfinite(float(x))]
    if not values:
        return None
    # Inclusive weak rank: observations <= current, including the current observation.
    return 100.0 * sum(x <= current for x in values) / len(values)


def mean_median(values: list[float]) -> tuple[float | None, float | None]:
    return (statistics.fmean(values), statistics.median(values)) if values else (None, None)
