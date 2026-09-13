from __future__ import annotations
from .normalization import directional_efficiency, regression_slope, safe_ratio


def horizon_measurements(closes, horizon: int, atr: float | None) -> dict[str, float | None]:
    if len(closes) < horizon + 1:
        return {}
    values = [float(x) for x in closes[-(horizon+1):]]
    displacement = values[-1] - values[0]
    slope = regression_slope(values)
    return {
        "displacement_price": displacement,
        "absolute_displacement_price": abs(displacement),
        "displacement_pct": safe_ratio(displacement, values[0]),
        "displacement_atr": safe_ratio(displacement, atr) if atr else None,
        "regression_slope_price_per_bar": slope,
        "regression_slope_atr": safe_ratio(slope, atr) if slope is not None and atr else None,
        "directional_efficiency": directional_efficiency(values),
    }
