"""Maximal contiguous runs of already-observed touched bars."""
from dataclasses import dataclass
from datetime import date, datetime

from .level_interaction_history import HistoricalLevelInteractionHistory
from .level_interaction import LevelBarObservation


@dataclass(frozen=True)
class TouchEpisode:
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
    episode_index: int
    first_observation_index: int
    last_observation_index: int
    first_bar_open_time_utc: datetime
    last_bar_open_time_utc: datetime
    episode_end_utc: datetime
    touched_bar_count: int
    bars_from_availability_to_episode: int
    minutes_from_availability_to_episode: float
    observations: tuple[LevelBarObservation, ...]

    def __post_init__(self):
        object.__setattr__(self, "observations", tuple(self.observations))


def build_touch_episodes(history: HistoricalLevelInteractionHistory) -> tuple[TouchEpisode, ...]:
    """Validate summary consistency, then group original observations without copying."""
    if not isinstance(history, HistoricalLevelInteractionHistory):
        raise TypeError("history must be HistoricalLevelInteractionHistory")
    observations = history.observations
    if any(not isinstance(o, LevelBarObservation) for o in observations):
        raise ValueError("history contains invalid observations")
    touched = tuple(i for i, o in enumerate(observations) if o.range_touched_level is True)
    if (len(observations) != history.observed_bar_count
            or touched != history.touched_bar_indices
            or len(touched) != history.touched_bar_count
            or history.touched_in_observed_window != bool(touched)
            or history.remained_untouched_in_observed_window != (len(observations) > 0 and not touched)):
        raise ValueError("inconsistent history touch summary")
    episodes = []
    cursor = 0
    while cursor < len(touched):
        first = last = touched[cursor]
        cursor += 1
        while cursor < len(touched) and touched[cursor] == last + 1:
            last = touched[cursor]
            cursor += 1
        members = observations[first:last + 1]
        first_open = members[0].bar_open_time_utc
        episodes.append(TouchEpisode(
            history.geometry_family_id, history.source_local_trading_date,
            history.source_block_label, history.source_block_index,
            history.source_block_start_utc, history.source_block_end_utc,
            history.geometry_available_at_utc, history.level_id, history.coordinate,
            history.level_price, len(episodes), first, last, first_open,
            members[-1].bar_open_time_utc, members[-1].bar_close_time_utc,
            len(members), first,
            (first_open - history.geometry_available_at_utc).total_seconds() / 60,
            members))
    return tuple(episodes)
