from __future__ import annotations
import math
import re
import numpy as np


def timeframe_minutes(timeframe: str) -> int:
    match = re.fullmatch(r"([MHD])(\d+)", timeframe)
    if not match:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return int(match.group(2)) * {"M": 1, "H": 60, "D": 1440}[match.group(1)]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def regression_slope(values) -> float | None:
    if len(values) < 2:
        return None
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all():
        return None
    x = np.arange(len(y), dtype=float)
    return float(np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2))


def directional_efficiency(values) -> float | None:
    if len(values) < 2:
        return None
    denominator = float(np.abs(np.diff(np.asarray(values, dtype=float))).sum())
    return safe_ratio(abs(float(values[-1]) - float(values[0])), denominator)
