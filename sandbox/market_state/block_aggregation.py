"""Complete anchored OHLC blocks from ordered UTC open-timestamped bars."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import isfinite

from .session_clock import BlockConfig, NEW_YORK_C1_C8, resolve_block

UTC = timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _require_utc(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")


@dataclass(frozen=True)
class OHLCBar:
    """Base bar; timeframe alignment is validated by the aggregation function."""
    open_time_utc: datetime
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self):
        _require_utc(self.open_time_utc)
        prices = (self.open, self.high, self.low, self.close)
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not isfinite(p) or p <= 0 for p in prices):
            raise ValueError("prices must be positive finite numbers")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close) or self.high < self.low:
            raise ValueError("invalid OHLC geometry")


@dataclass(frozen=True)
class AnchoredBlockCandle:
    local_trading_date: date
    block_label: str
    block_index: int
    block_start_utc: datetime
    block_end_utc: datetime
    local_block_start: datetime
    local_block_end: datetime
    open: float
    high: float
    low: float
    close: float
    base_bar_count: int
    expected_base_bar_count: int


def aggregate_completed_blocks(bars, decision_time_utc: datetime,
                               config: BlockConfig = NEW_YORK_C1_C8,
                               base_minutes: int = 15) -> tuple[AnchoredBlockCandle, ...]:
    """Emit only exact complete grids at/before the decision instant.

    Base timestamps align to a UTC epoch-based minute grid. Input ordering and
    alignment are validated even for future bars, whose prices are never used.
    Expected grids follow actual UTC boundaries, including DST-short/long blocks.
    """
    _require_utc(decision_time_utc)
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    step = timedelta(minutes=base_minutes)
    previous = None
    groups = {}
    for bar in bars:
        if not isinstance(bar, OHLCBar):
            raise ValueError("bars must be OHLCBar instances")
        stamp = bar.open_time_utc
        _require_utc(stamp)
        if previous is not None and stamp <= previous:
            raise ValueError("bar timestamps must be strictly increasing")
        previous = stamp
        if (stamp - _EPOCH) % step:
            raise ValueError("bar timestamp is not aligned to base timeframe")
        if stamp + step > decision_time_utc:
            continue
        resolution = resolve_block(stamp, config)
        start = resolution.local_block_start.astimezone(UTC)
        end = resolution.local_block_end.astimezone(UTC)
        elapsed = end - start
        if elapsed <= timedelta(0) or elapsed % step:
            raise ValueError("actual UTC block duration must be positive and divisible by base_minutes")
        if end > decision_time_utc:
            continue
        key = (start, end)
        if key not in groups:
            groups[key] = (resolution, [])
        groups[key][1].append(bar)

    completed = []
    for (start, end), (resolution, members) in sorted(groups.items()):
        expected = (end - start) // step
        if len(members) != expected:
            continue
        if any(bar.open_time_utc != start + i * step for i, bar in enumerate(members)):
            continue
        completed.append(AnchoredBlockCandle(
            local_trading_date=resolution.local_trading_date,
            block_label=resolution.block_label, block_index=resolution.block_index,
            block_start_utc=start, block_end_utc=end,
            local_block_start=resolution.local_block_start,
            local_block_end=resolution.local_block_end,
            open=members[0].open, high=max(b.high for b in members),
            low=min(b.low for b in members), close=members[-1].close,
            base_bar_count=len(members), expected_base_bar_count=expected,
        ))
    return tuple(completed)
