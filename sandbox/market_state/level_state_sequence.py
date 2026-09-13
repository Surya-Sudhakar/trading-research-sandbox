"""Objective range states and maximal runs of completed observations."""
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from .level_interaction_history import HistoricalLevelInteractionHistory
from .level_interaction import LevelBarObservation


class LevelRangeState(StrEnum):
    ABOVE = "ABOVE"
    TOUCH = "TOUCH"
    BELOW = "BELOW"


@dataclass(frozen=True)
class LevelStateRun:
    run_index: int
    state: LevelRangeState
    first_observation_index: int
    last_observation_index: int
    first_bar_open_time_utc: datetime
    last_bar_open_time_utc: datetime
    run_end_utc: datetime
    bar_count: int
    observations: tuple[LevelBarObservation, ...]

    def __post_init__(self):
        object.__setattr__(self, "observations", tuple(self.observations))


@dataclass(frozen=True)
class LevelStateSequence:
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
    observed_bar_count: int
    states: tuple[LevelRangeState, ...]
    runs: tuple[LevelStateRun, ...]
    run_count: int
    transition_count: int

    def __post_init__(self):
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "runs", tuple(self.runs))


_IDENTITY = (
    "geometry_family_id", "source_local_trading_date", "source_block_label",
    "source_block_index", "source_block_start_utc", "source_block_end_utc",
    "geometry_available_at_utc", "level_id", "coordinate", "level_price",
)


def _state(observation):
    if observation.range_touched_level is True:
        return LevelRangeState.TOUCH
    if observation.low > observation.level_price:
        return LevelRangeState.ABOVE
    if observation.high < observation.level_price:
        return LevelRangeState.BELOW
    raise ValueError("observation cannot be classified into a level range state")


def build_level_state_sequence(history: HistoricalLevelInteractionHistory) -> LevelStateSequence:
    if not isinstance(history, HistoricalLevelInteractionHistory):
        raise TypeError("history must be HistoricalLevelInteractionHistory")
    observations = history.observations
    if len(observations) != history.observed_bar_count:
        raise ValueError("inconsistent observed_bar_count")
    previous = None
    for observation in observations:
        if not isinstance(observation, LevelBarObservation):
            raise ValueError("invalid observation type")
        if any(getattr(observation, field) != getattr(history, field) for field in _IDENTITY):
            raise ValueError("inconsistent observation source or level identity")
        stamp = observation.bar_open_time_utc
        if previous is not None and stamp <= previous:
            raise ValueError("observations must be strictly chronological")
        previous = stamp
    touched = tuple(i for i, o in enumerate(observations) if o.range_touched_level is True)
    if touched != history.touched_bar_indices or len(touched) != history.touched_bar_count:
        raise ValueError("inconsistent touched indices or count")
    states = tuple(_state(o) for o in observations)
    runs = []
    first = 0
    while first < len(states):
        end = first + 1
        while end < len(states) and states[end] == states[first]:
            end += 1
        members = observations[first:end]
        runs.append(LevelStateRun(len(runs), states[first], first, end - 1,
            members[0].bar_open_time_utc, members[-1].bar_open_time_utc,
            members[-1].bar_close_time_utc, len(members), members))
        first = end
    return LevelStateSequence(
        **{field: getattr(history, field) for field in _IDENTITY},
        decision_time_utc=history.decision_time_utc, base_minutes=history.base_minutes,
        observed_bar_count=history.observed_bar_count, states=states, runs=tuple(runs),
        run_count=len(runs), transition_count=max(0, len(runs) - 1))
