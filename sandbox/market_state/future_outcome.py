"""Direction-neutral fixed-horizon measurements from completed historical bars."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
import re

from .block_aggregation import OHLCBar

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DEFAULT_FORWARD_HORIZONS = (1, 4, 8, 16, 32)


@dataclass(frozen=True)
class OutcomeAnchor:
    anchor_id: str
    anchor_time_utc: datetime
    reference_price: float

    def __post_init__(self):
        if not isinstance(self.anchor_id, str) or re.fullmatch(r"[A-Z0-9_]+", self.anchor_id) is None:
            raise ValueError("invalid anchor_id")
        stamp = self.anchor_time_utc
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("anchor time must be timezone-aware UTC")
        value = self.reference_price
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("reference_price must be finite numeric")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError("reference_price must be finite numeric") from exc
        if not isfinite(value):
            raise ValueError("reference_price must be finite numeric")
        object.__setattr__(self, "reference_price", value)


@dataclass(frozen=True)
class ForwardOutcome:
    anchor_id: str
    anchor_time_utc: datetime
    reference_price: float
    horizon_bars: int
    base_minutes: int
    window_start_utc: datetime
    window_end_utc: datetime
    first_bar_open_time_utc: datetime
    last_bar_open_time_utc: datetime
    first_open: float
    end_close: float
    highest_high: float
    lowest_low: float
    close_change: float
    close_return_fraction: float | None
    max_upward_excursion: float
    max_downward_excursion: float
    highest_high_bar_open_time_utc: datetime
    lowest_low_bar_open_time_utc: datetime
    bars_to_highest_high: int
    bars_to_lowest_low: int


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _prepare(anchor, bars, base_minutes):
    if not isinstance(anchor, OutcomeAnchor):
        raise TypeError("anchor must be OutcomeAnchor")
    _positive_integer(base_minutes, "base_minutes")
    step = timedelta(minutes=base_minutes)
    if (anchor.anchor_time_utc - _EPOCH) % step:
        raise ValueError("anchor must align to base timeframe")
    materialized = tuple(bars)
    lookup = {}
    previous = None
    for bar in materialized:
        if not isinstance(bar, OHLCBar):
            raise TypeError("bars must contain OHLCBar")
        stamp = bar.open_time_utc
        if previous is not None and stamp <= previous:
            raise ValueError("bars must be strictly increasing")
        if (stamp - _EPOCH) % step:
            raise ValueError("bar must align to base timeframe")
        previous = stamp
        lookup[stamp] = bar
    return lookup, step


def _measure(anchor, lookup, step, horizon, base_minutes):
    members = []
    for index in range(horizon):
        stamp = anchor.anchor_time_utc + index * step
        if stamp not in lookup:
            raise ValueError(f"missing required future bar: {stamp.isoformat()}")
        members.append(lookup[stamp])
    high_index = max(range(horizon), key=lambda i: members[i].high)
    low_index = min(range(horizon), key=lambda i: members[i].low)
    highest, lowest = members[high_index].high, members[low_index].low
    change = members[-1].close - anchor.reference_price
    return ForwardOutcome(
        anchor.anchor_id, anchor.anchor_time_utc, anchor.reference_price,
        horizon, base_minutes, anchor.anchor_time_utc, anchor.anchor_time_utc + horizon * step,
        members[0].open_time_utc, members[-1].open_time_utc,
        members[0].open, members[-1].close, highest, lowest, change,
        change / anchor.reference_price if anchor.reference_price != 0 else None,
        max(0.0, highest - anchor.reference_price), max(0.0, anchor.reference_price - lowest),
        members[high_index].open_time_utc, members[low_index].open_time_utc, high_index, low_index)


def measure_forward_outcome(anchor: OutcomeAnchor, bars, horizon_bars: int,
                            base_minutes: int = 15) -> ForwardOutcome:
    _positive_integer(horizon_bars, "horizon_bars")
    lookup, step = _prepare(anchor, bars, base_minutes)
    return _measure(anchor, lookup, step, horizon_bars, base_minutes)


def measure_forward_outcomes(anchor: OutcomeAnchor, bars, horizons=DEFAULT_FORWARD_HORIZONS,
                             base_minutes: int = 15) -> tuple[ForwardOutcome, ...]:
    horizons = tuple(horizons)
    for horizon in horizons:
        _positive_integer(horizon, "horizon")
    if len(set(horizons)) != len(horizons):
        raise ValueError("duplicate horizons")
    lookup, step = _prepare(anchor, bars, base_minutes)
    return tuple(_measure(anchor, lookup, step, horizon, base_minutes) for horizon in horizons)
