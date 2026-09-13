"""Use the existing partition-local UTC reconstruction without strategy execution."""
from types import SimpleNamespace
import pandas as pd
from sandbox.partition.service import PartitionService
from sandbox.market_state.normalization import timeframe_minutes


def completed_context(frame, decision_timeframe, context_timeframes):
    # Only the required_timeframes metadata is consumed by this existing utility.
    request = SimpleNamespace(metadata=SimpleNamespace(required_timeframes=context_timeframes))
    source = frame.copy()
    for name in ("tick_volume", "spread", "real_volume"):
        source[name] = 0  # Unused by price features; do not invent market prices.
    frames = PartitionService._strategy_timeframe_frames(source, request, decision_timeframe)
    result = {}
    for tf, bars in frames.items():
        duration = pd.Timedelta(minutes=timeframe_minutes(tf))
        result[tf] = [
            (r.timestamp_utc + duration, int(r.close > r.open)-int(r.close < r.open))
            for r in bars.itertuples(index=False)
        ]
    return result

