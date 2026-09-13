def prior_structure(highs, lows, close, index, lookback=12):
    if index < lookback:
        return dict(prior_high_12=None, prior_low_12=None,
                    breakout_high_12=None, breakout_low_12=None)
    high = max(highs[index-lookback:index])
    low = min(lows[index-lookback:index])
    return dict(prior_high_12=high, prior_low_12=low,
                breakout_high_12=bool(close > high), breakout_low_12=bool(close < low))

