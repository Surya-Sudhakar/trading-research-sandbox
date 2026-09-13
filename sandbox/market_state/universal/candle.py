from sandbox.market_state.normalization import safe_ratio


def geometry(open_: float, high: float, low: float, close: float) -> dict:
    span = high - low
    body = abs(close - open_)
    return dict(candle_range=span, body_size=body, body_ratio=safe_ratio(body, span),
                upper_wick=high-max(open_, close), lower_wick=min(open_, close)-low,
                body_midpoint=(open_+close)/2, close_location=safe_ratio(close-low, span),
                candle_direction=int(close > open_)-int(close < open_))

