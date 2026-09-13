from __future__ import annotations
from .normalization import regression_slope, safe_ratio


def change_measurements(closes, horizon: int, atr: float | None) -> dict[str, float | bool | None]:
    if len(closes) < 2*horizon + 1:
        return {}
    values = [float(x) for x in closes[-(2*horizon+1):]]
    preceding = values[horizon] - values[0]
    recent = values[-1] - values[horizon]
    preceding_slope = regression_slope(values[:horizon+1])
    recent_slope = regression_slope(values[horizon:])
    return {
        "recent_displacement_price": recent,
        "preceding_displacement_price": preceding,
        "recent_to_preceding_displacement_ratio": safe_ratio(recent, preceding),
        "recent_minus_preceding_displacement_atr": safe_ratio(recent-preceding, atr) if atr else None,
        "recent_regression_slope": recent_slope,
        "preceding_regression_slope": preceding_slope,
        "recent_to_preceding_slope_ratio": safe_ratio(recent_slope, preceding_slope) if recent_slope is not None and preceding_slope is not None else None,
        "displacement_sign_changed": (recent > 0) != (preceding > 0) if recent != 0 and preceding != 0 else False,
    }
