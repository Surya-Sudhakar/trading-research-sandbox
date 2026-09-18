"""Causal completed-timeframe reconstruction and market-state context."""
from datetime import timedelta
from types import SimpleNamespace
from sandbox.partition.service import PartitionService
from sandbox.market_state.normalization import timeframe_minutes
from .market_metrics import atr_change, efficiency_ratio, normalized_regression_slope
from .volatility import atr_series


def completed_bars(frame, decision_timeframe, context_timeframes):
    """Return only fully reconstructed higher-timeframe bars from the existing partition service."""
    request = SimpleNamespace(metadata=SimpleNamespace(required_timeframes=context_timeframes))
    source = frame.copy()
    for name in ("tick_volume", "spread", "real_volume"):
        source[name] = 0  # Price reconstruction does not consume these fields.
    return PartitionService._strategy_timeframe_frames(source, request, decision_timeframe)


def completed_context(frame, decision_timeframe, context_timeframes):
    frames = completed_bars(frame, decision_timeframe, context_timeframes)
    result = {}
    for tf, bars in frames.items():
        duration = timedelta(minutes=int(timeframe_minutes(tf)))
        result[tf] = [
            (r.timestamp_utc + duration, int(r.close > r.open)-int(r.close < r.open))
            for r in bars.itertuples(index=False)
        ]
    return result


def completed_state_context(frame, decision_timeframe, context_timeframes=("H1", "H3")):
    """Return causal state measurements keyed by the source bar close timestamp."""
    frames = completed_bars(frame, decision_timeframe, context_timeframes)
    result = {}
    for tf, bars in frames.items():
        duration = timedelta(minutes=int(timeframe_minutes(tf)))
        closes = bars["close"].astype(float).tolist()
        highs = bars["high"].astype(float).tolist()
        lows = bars["low"].astype(float).tolist()
        atr = atr_series(highs, lows, closes)
        values = []
        for i, r in enumerate(bars.itertuples(index=False)):
            source_close = r.timestamp_utc + duration
            values.append((source_close, {
                "er_8": efficiency_ratio(closes, i, 8),
                "atr_change_4": atr_change(atr, i, 4),
                "slope_atr_8": normalized_regression_slope(closes, i, 8, atr[i]),
            }))
        result[tf] = values
    return result
