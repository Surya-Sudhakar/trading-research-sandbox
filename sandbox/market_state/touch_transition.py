"""Mechanical TOUCH-centered patterns from validated maximal state runs."""
from dataclasses import dataclass
from datetime import date, datetime

from .level_state_sequence import LevelRangeState, LevelStateRun, LevelStateSequence
from .level_interaction import LevelBarObservation


@dataclass(frozen=True)
class TouchTransitionPattern:
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
    pattern_index: int
    touch_run_index: int
    touch_first_observation_index: int
    touch_last_observation_index: int
    touch_bar_count: int
    touch_first_bar_open_time_utc: datetime
    touch_last_bar_open_time_utc: datetime
    touch_end_utc: datetime
    previous_state: LevelRangeState | None
    previous_run_index: int | None
    previous_bar_count: int | None
    next_state: LevelRangeState | None
    next_run_index: int | None
    next_bar_count: int | None
    has_previous_side: bool
    has_next_side: bool
    complete_side_to_side_transition: bool
    same_side_before_after: bool | None
    opposite_side_before_after: bool | None
    pattern_code: str
    previous_run: LevelStateRun | None
    touch_run: LevelStateRun
    next_run: LevelStateRun | None


_IDENTITY = (
    "geometry_family_id", "source_local_trading_date", "source_block_label",
    "source_block_index", "source_block_start_utc", "source_block_end_utc",
    "geometry_available_at_utc", "level_id", "coordinate", "level_price",
)


def _validate(sequence):
    if not isinstance(sequence, LevelStateSequence):
        raise TypeError("sequence must be LevelStateSequence")
    if (len(sequence.states) != sequence.observed_bar_count
            or len(sequence.runs) != sequence.run_count
            or sequence.transition_count != max(0, sequence.run_count - 1)):
        raise ValueError("inconsistent sequence counts")
    next_index = 0
    previous_state = None
    for index, run in enumerate(sequence.runs):
        if not isinstance(run, LevelStateRun) or not isinstance(run.state, LevelRangeState):
            raise ValueError("invalid state run")
        if (run.run_index != index or run.first_observation_index != next_index
                or run.first_observation_index > run.last_observation_index
                or run.last_observation_index >= sequence.observed_bar_count
                or run.bar_count != run.last_observation_index - run.first_observation_index + 1
                or len(run.observations) != run.bar_count):
            raise ValueError("inconsistent run coverage or counts")
        if previous_state == run.state:
            raise ValueError("consecutive runs must have different states")
        if any(state != run.state for state in sequence.states[run.first_observation_index:run.last_observation_index + 1]):
            raise ValueError("run state does not match sequence states")
        for observation in run.observations:
            if not isinstance(observation, LevelBarObservation):
                raise ValueError("invalid observation type")
            if any(getattr(observation, field) != getattr(sequence, field) for field in _IDENTITY):
                raise ValueError("observation identity mismatch")
        next_index = run.last_observation_index + 1
        previous_state = run.state
    if next_index != sequence.observed_bar_count:
        raise ValueError("runs do not cover all observations")


def build_touch_transition_patterns(sequence: LevelStateSequence) -> tuple[TouchTransitionPattern, ...]:
    _validate(sequence)
    patterns = []
    for index, run in enumerate(sequence.runs):
        if run.state != LevelRangeState.TOUCH:
            continue
        previous = sequence.runs[index - 1] if index else None
        following = sequence.runs[index + 1] if index + 1 < len(sequence.runs) else None
        complete = previous is not None and following is not None
        if previous is None and following is None:
            code = "TOUCH_ONLY"
        else:
            before = previous.state.value if previous is not None else "START"
            after = following.state.value if following is not None else "END"
            code = f"{before}_TOUCH_{after}"
        patterns.append(TouchTransitionPattern(
            **{field: getattr(sequence, field) for field in _IDENTITY},
            pattern_index=len(patterns), touch_run_index=run.run_index,
            touch_first_observation_index=run.first_observation_index,
            touch_last_observation_index=run.last_observation_index,
            touch_bar_count=run.bar_count, touch_first_bar_open_time_utc=run.first_bar_open_time_utc,
            touch_last_bar_open_time_utc=run.last_bar_open_time_utc, touch_end_utc=run.run_end_utc,
            previous_state=previous.state if previous is not None else None,
            previous_run_index=previous.run_index if previous is not None else None,
            previous_bar_count=previous.bar_count if previous is not None else None,
            next_state=following.state if following is not None else None,
            next_run_index=following.run_index if following is not None else None,
            next_bar_count=following.bar_count if following is not None else None,
            has_previous_side=previous is not None, has_next_side=following is not None,
            complete_side_to_side_transition=complete,
            same_side_before_after=previous.state == following.state if complete else None,
            opposite_side_before_after=previous.state != following.state if complete else None,
            pattern_code=code, previous_run=previous, touch_run=run, next_run=following))
    return tuple(patterns)
