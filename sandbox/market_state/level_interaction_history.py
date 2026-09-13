"""Continuous completed-bar observation histories; touched bars are not events."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from .block_aggregation import OHLCBar
from .geometry_lifecycle import HistoricalGeometryInstance
from .level_interaction import LevelBarObservation, observe_geometry_level

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class HistoricalLevelInteractionHistory:
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
    decision_time_utc: datetime
    base_minutes: int
    observation_start_utc: datetime
    observation_end_utc: datetime
    observed_bar_count: int
    touched_bar_count: int
    touched_bar_indices: tuple[int, ...]
    touched_in_observed_window: bool
    remained_untouched_in_observed_window: bool
    first_touch_bar_index: int | None
    first_touch_bar_open_time_utc: datetime | None
    bars_before_first_touch: int | None
    minutes_from_availability_to_first_touch_bar: float | None
    last_touch_bar_index: int | None
    last_touch_bar_open_time_utc: datetime | None
    observations: tuple[LevelBarObservation, ...]

    def __post_init__(self):
        object.__setattr__(self, "touched_bar_indices", tuple(self.touched_bar_indices))
        object.__setattr__(self, "observations", tuple(self.observations))


def _prepare(instance, bars, decision, base_minutes):
    if not isinstance(instance, HistoricalGeometryInstance):
        raise TypeError("instance must be HistoricalGeometryInstance")
    if not isinstance(decision, datetime) or decision.tzinfo is None or decision.utcoffset() != timedelta(0):
        raise ValueError("decision must be timezone-aware UTC")
    if decision < instance.available_at_utc:
        raise ValueError("decision before geometry availability")
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    step = timedelta(minutes=base_minutes)
    if (instance.available_at_utc - _EPOCH) % step:
        raise ValueError("geometry availability must align to base timeframe")
    materialized = tuple(bars)
    previous = None
    lookup = {}
    for bar in materialized:
        if not isinstance(bar, OHLCBar):
            raise TypeError("bars must contain OHLCBar")
        stamp = bar.open_time_utc
        if previous is not None and stamp <= previous:
            raise ValueError("bar timestamps must be strictly increasing")
        previous = stamp
        if (stamp - _EPOCH) % step:
            raise ValueError("bar timestamp must align to base timeframe")
        lookup[stamp] = bar
    count = (decision - instance.available_at_utc) // step
    expected = []
    for i in range(count):
        stamp = instance.available_at_utc + i * step
        if stamp not in lookup:
            raise ValueError(f"missing required completed bar: {stamp.isoformat()}")
        expected.append(lookup[stamp])
    return tuple(expected)


def _history(instance, level_id, expected, decision, base_minutes):
    levels = [level for level in instance.projection.levels if level.level_id == level_id]
    if len(levels) != 1:
        raise ValueError("level_id must identify exactly one projected level")
    level = levels[0]
    observations = tuple(observe_geometry_level(instance, level_id, bar, base_minutes) for bar in expected)
    touched = tuple(i for i, observation in enumerate(observations) if observation.range_touched_level)
    first, last = (touched[0], touched[-1]) if touched else (None, None)
    first_open = observations[first].bar_open_time_utc if first is not None else None
    last_open = observations[last].bar_open_time_utc if last is not None else None
    available = instance.available_at_utc
    return HistoricalLevelInteractionHistory(
        instance.geometry_family_id, instance.source_local_trading_date,
        instance.source_block_label, instance.source_block_index,
        instance.source_block_start_utc, instance.source_block_end_utc, available,
        level.level_id, level.coordinate, level.price, decision, base_minutes,
        available, observations[-1].bar_close_time_utc if observations else available,
        len(observations), len(touched), touched, bool(touched), bool(observations) and not touched,
        first, first_open, first,
        (first_open - available).total_seconds() / 60 if first_open is not None else None,
        last, last_open, observations)


def build_level_interaction_history(instance: HistoricalGeometryInstance, level_id: str,
                                    bars, decision_time_utc, base_minutes: int = 15) -> HistoricalLevelInteractionHistory:
    expected = _prepare(instance, bars, decision_time_utc, base_minutes)
    return _history(instance, level_id, expected, decision_time_utc, base_minutes)


def build_geometry_interaction_histories(instance: HistoricalGeometryInstance, bars,
                                         decision_time_utc, base_minutes: int = 15) -> tuple[HistoricalLevelInteractionHistory, ...]:
    expected = _prepare(instance, bars, decision_time_utc, base_minutes)
    return tuple(_history(instance, level.level_id, expected, decision_time_utc, base_minutes)
                 for level in instance.projection.levels)
