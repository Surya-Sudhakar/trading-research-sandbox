"""Objective level/bar facts. Callers supply completed base OHLC bars."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum

from .block_aggregation import OHLCBar
from .geometry_lifecycle import HistoricalGeometryInstance
from .price_geometry import ProjectedLevel

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class LevelPricePosition(StrEnum):
    ABOVE = "ABOVE"
    AT = "AT"
    BELOW = "BELOW"


@dataclass(frozen=True)
class LevelBarObservation:
    geometry_family_id: str
    source_local_trading_date: date
    source_block_label: str
    source_block_index: int
    source_block_start_utc: datetime
    source_block_end_utc: datetime
    geometry_available_at_utc: datetime
    level_id: str
    coordinate: float
    level_price: float
    bar_open_time_utc: datetime
    bar_close_time_utc: datetime
    open: float
    high: float
    low: float
    close: float
    open_position: LevelPricePosition
    close_position: LevelPricePosition
    range_touched_level: bool
    body_touched_level: bool
    traded_above_level: bool
    traded_below_level: bool
    straddled_level: bool
    body_crossed_level: bool
    wick_only_touch: bool


def _validate(instance, bar, base_minutes):
    if not isinstance(instance, HistoricalGeometryInstance):
        raise TypeError("instance must be HistoricalGeometryInstance")
    if not isinstance(bar, OHLCBar):
        raise TypeError("bar must be OHLCBar")
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    if bar.open_time_utc < instance.available_at_utc:
        raise ValueError("bar opened before geometry availability")
    step = timedelta(minutes=base_minutes)
    if (bar.open_time_utc - _EPOCH) % step:
        raise ValueError("bar timestamp must align to base timeframe")
    return step


def _position(value, price):
    return LevelPricePosition.ABOVE if value > price else (
        LevelPricePosition.BELOW if value < price else LevelPricePosition.AT)


def observe_geometry_level(instance: HistoricalGeometryInstance, level_id: str,
                           bar: OHLCBar, base_minutes: int = 15) -> LevelBarObservation:
    step = _validate(instance, bar, base_minutes)
    matches = [level for level in instance.projection.levels if level.level_id == level_id]
    if len(matches) != 1:
        raise ValueError("level_id must identify exactly one projected level")
    level = matches[0]
    if not isinstance(level, ProjectedLevel):
        raise TypeError("level must be ProjectedLevel")
    price = level.price
    touched = bar.low <= price <= bar.high
    body_touched = min(bar.open, bar.close) <= price <= max(bar.open, bar.close)
    above, below = bar.high > price, bar.low < price
    crossed = (bar.open > price and bar.close < price) or (bar.open < price and bar.close > price)
    return LevelBarObservation(
        instance.geometry_family_id, instance.source_local_trading_date,
        instance.source_block_label, instance.source_block_index,
        instance.source_block_start_utc, instance.source_block_end_utc,
        instance.available_at_utc, level.level_id, level.coordinate, price,
        bar.open_time_utc, bar.open_time_utc + step, bar.open, bar.high, bar.low, bar.close,
        _position(bar.open, price), _position(bar.close, price), touched, body_touched,
        above, below, above and below, crossed, touched and not body_touched)


def observe_geometry_levels(instance: HistoricalGeometryInstance, bar: OHLCBar,
                            base_minutes: int = 15) -> tuple[LevelBarObservation, ...]:
    _validate(instance, bar, base_minutes)
    return tuple(observe_geometry_level(instance, level.level_id, bar, base_minutes)
                 for level in instance.projection.levels)
