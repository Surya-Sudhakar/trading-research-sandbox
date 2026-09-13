"""Historical raw-OHLC contexts with one aggregation pass per frame."""
from datetime import datetime, timedelta, timezone

from .block_aggregation import OHLCBar, aggregate_completed_blocks
from .multitimeframe_context import (
    ContextFrame, DEFAULT_CONTEXT_FRAMES, CompletedFrameContext,
    MultiTimeframeSnapshot, expected_completed_window,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def iter_historical_multitimeframe_snapshots(bars, frames=DEFAULT_CONTEXT_FRAMES,
                                            base_minutes=15):
    """Yield chronological snapshots using exact completed-window lookups.

    Validation and precomputation occur when iteration starts. Missing windows
    remain None; precomputed future candles cannot substitute for expected ones.
    """
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    frames = tuple(frames)
    if any(not isinstance(frame, ContextFrame) for frame in frames):
        raise ValueError("frames must contain ContextFrame instances")
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("duplicate frame_id")
    materialized = tuple(bars)
    step = timedelta(minutes=base_minutes)
    previous = None
    for bar in materialized:
        if not isinstance(bar, OHLCBar):
            raise ValueError("bars must contain OHLCBar instances")
        stamp = bar.open_time_utc
        if previous is not None and stamp <= previous:
            raise ValueError("bar timestamps must be strictly increasing")
        if (stamp - _EPOCH) % step:
            raise ValueError("bar timestamp is not aligned to base timeframe")
        previous = stamp
    if not materialized:
        return
    final_decision_time = materialized[-1].open_time_utc + step
    indexes = []
    for frame in frames:
        candles = aggregate_completed_blocks(materialized, final_decision_time,
                                             frame.config, base_minutes)
        indexes.append({(c.block_start_utc, c.block_end_utc): c for c in candles})
    for bar in materialized:
        decision = bar.open_time_utc + step
        contexts = []
        for frame, index in zip(frames, indexes):
            start, end = expected_completed_window(decision, frame.config)
            contexts.append(CompletedFrameContext(frame.frame_id, start, end,
                                                   index.get((start, end))))
        yield MultiTimeframeSnapshot(decision, bar, tuple(contexts))
