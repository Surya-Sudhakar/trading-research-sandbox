from sandbox.market_state.normalization import safe_ratio


def displacement(closes, index, horizon, atr):
    return (safe_ratio(closes[index]-closes[index-horizon], atr)
            if index >= horizon and atr is not None else None)

