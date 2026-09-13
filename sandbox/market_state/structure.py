from __future__ import annotations
from .normalization import safe_ratio


def range_measurements(highs, lows, closes, horizon: int, atr: float | None) -> dict[str, float | int | None]:
    if len(closes) < horizon:
        return {}
    hs=[float(x) for x in highs[-horizon:]];ls=[float(x) for x in lows[-horizon:]];close=float(closes[-1])
    high=max(hs);low=min(ls);width=high-low
    high_index=max(i for i,x in enumerate(hs) if x==high);low_index=max(i for i,x in enumerate(ls) if x==low)
    return {"rolling_high":high,"rolling_low":low,"rolling_range_price":width,
        "rolling_range_atr":safe_ratio(width,atr) if atr else None,"close_position_in_range":safe_ratio(close-low,width),
        "distance_from_rolling_high_price":high-close,"distance_from_rolling_low_price":close-low,
        "distance_from_rolling_high_atr":safe_ratio(high-close,atr) if atr else None,
        "distance_from_rolling_low_atr":safe_ratio(close-low,atr) if atr else None,
        "bars_since_rolling_high":horizon-1-high_index,"bars_since_rolling_low":horizon-1-low_index}
