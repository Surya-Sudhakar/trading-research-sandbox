"""Synchronize exact completed raw OHLC windows without stale fallback."""
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import re

from .session_clock import BlockConfig, NEW_YORK_C1_C8, resolve_block
from .block_aggregation import OHLCBar, AnchoredBlockCandle, aggregate_completed_blocks

UTC = timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
NEW_YORK_H1 = BlockConfig("America/New_York", time(), 60, 24, tuple(f"H1_{h:02d}" for h in range(24)))
NEW_YORK_H4 = BlockConfig("America/New_York", time(), 240, 6, tuple(f"H4_{h:02d}" for h in range(0, 24, 4)))
NEW_YORK_D1 = BlockConfig("America/New_York", time(), 1440, 1, ("D1",))


@dataclass(frozen=True)
class ContextFrame:
    frame_id: str
    config: BlockConfig

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or re.fullmatch(r"[A-Z0-9_]+", self.frame_id) is None:
            raise ValueError("invalid frame_id")
        if not isinstance(self.config, BlockConfig):
            raise ValueError("config must be BlockConfig")


DEFAULT_CONTEXT_FRAMES = (
    ContextFrame("NY_D1", NEW_YORK_D1), ContextFrame("NY_H4", NEW_YORK_H4),
    ContextFrame("NY_H1", NEW_YORK_H1), ContextFrame("NY_H3", NEW_YORK_C1_C8),
)


@dataclass(frozen=True)
class CompletedFrameContext:
    frame_id: str
    expected_block_start_utc: datetime
    expected_block_end_utc: datetime
    candle: AnchoredBlockCandle | None


@dataclass(frozen=True)
class MultiTimeframeSnapshot:
    decision_time_utc: datetime
    base_bar: OHLCBar | None
    frames: tuple[CompletedFrameContext, ...]


def expected_completed_window(decision, config):
    # Move through actual UTC instants, not fixed-duration local days. This also
    # skips nonexistent wall-clock blocks and preserves repeated-hour blocks.
    probe = decision
    while True:
        block = resolve_block(probe, config)
        start = block.local_block_start.astimezone(UTC)
        end = block.local_block_end.astimezone(UTC)
        if start < end <= decision:
            return start, end
        previous_probe = start - timedelta(microseconds=1)
        if previous_probe >= probe:
            raise ValueError("configuration does not yield a preceding UTC block")
        probe = previous_probe


def build_multitimeframe_snapshot(bars, decision_time_utc,
                                 frames=DEFAULT_CONTEXT_FRAMES,
                                 base_minutes=15) -> MultiTimeframeSnapshot:
    if not isinstance(decision_time_utc, datetime) or decision_time_utc.tzinfo is None or decision_time_utc.utcoffset() != timedelta(0):
        raise ValueError("decision timestamp must be timezone-aware UTC")
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
    latest_open = _EPOCH + ((decision_time_utc - _EPOCH) // step - 1) * step
    base_bar = next((bar for bar in materialized if bar.open_time_utc == latest_open), None)
    results = []
    for frame in frames:
        start, end = expected_completed_window(decision_time_utc, frame.config)
        candles = aggregate_completed_blocks(materialized, decision_time_utc, frame.config, base_minutes)
        candle = next((c for c in candles if c.block_start_utc == start and c.block_end_utc == end), None)
        results.append(CompletedFrameContext(frame.frame_id, start, end, candle))
    return MultiTimeframeSnapshot(decision_time_utc, base_bar, tuple(results))
