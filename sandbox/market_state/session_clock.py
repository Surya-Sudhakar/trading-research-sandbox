"""Local wall-clock blocks, independent of candles and trading logic.

Blocks use [start, end) and zero-based indices. Trading date is the local
calendar date on which the cycle's anchor falls (midnight for NY C1-C8).
Boundaries are nominal local wall times: a skipped DST boundary may be an
imaginary local time; an ambiguous boundary uses fold=0. Membership is always
by wall time, so both occurrences of a repeated time receive the same block.
Do not interpret boundary subtraction as elapsed UTC duration.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class BlockConfig:
    timezone: str
    anchor: time
    block_minutes: int
    number_of_blocks: int
    labels: tuple[str, ...]

    def __post_init__(self):
        ZoneInfo(self.timezone)
        if not isinstance(self.anchor, time) or self.anchor.tzinfo is not None:
            raise ValueError("anchor must be a naive local time")
        if self.anchor.fold != 0:
            raise ValueError("anchor must use fold=0")
        if type(self.block_minutes) is not int or self.block_minutes <= 0:
            raise ValueError("block_minutes must be a positive integer")
        if type(self.number_of_blocks) is not int or self.number_of_blocks <= 0:
            raise ValueError("number_of_blocks must be a positive integer")
        if isinstance(self.labels, str):
            raise ValueError("labels must be a sequence of labels")
        labels = tuple(self.labels)
        if len(labels) != self.number_of_blocks:
            raise ValueError("label count must match number_of_blocks")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        if self.block_minutes * self.number_of_blocks != 1440:
            raise ValueError("blocks must cover exactly 24 local wall-clock hours")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class BlockResolution:
    utc_timestamp: datetime
    local_timestamp: datetime
    local_trading_date: date
    block_label: str
    block_index: int
    local_block_start: datetime
    local_block_end: datetime


NEW_YORK_C1_C8 = BlockConfig(
    timezone="America/New_York", anchor=time(0), block_minutes=180,
    number_of_blocks=8, labels=tuple(f"C{i}" for i in range(1, 9)),
)


def resolve_block(utc_timestamp: datetime, config: BlockConfig = NEW_YORK_C1_C8) -> BlockResolution:
    """Resolve an aware UTC instant against a repeating local-day configuration."""
    if not isinstance(utc_timestamp, datetime) or utc_timestamp.tzinfo is None or utc_timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    if utc_timestamp.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    zone = ZoneInfo(config.timezone)
    local = utc_timestamp.astimezone(zone)
    wall = local.replace(tzinfo=None, fold=0)
    anchor = datetime.combine(local.date(), config.anchor)
    if wall < anchor:
        anchor -= timedelta(days=1)
    duration = timedelta(minutes=config.block_minutes)
    index = (wall - anchor) // duration
    start = anchor + index * duration
    end = start + duration
    return BlockResolution(
        utc_timestamp=utc_timestamp, local_timestamp=local,
        local_trading_date=anchor.date(), block_label=config.labels[index],
        block_index=index, local_block_start=start.replace(tzinfo=zone),
        local_block_end=end.replace(tzinfo=zone),
    )
