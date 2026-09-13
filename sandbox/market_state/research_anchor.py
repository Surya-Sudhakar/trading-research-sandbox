"""Causal geometry evidence anchors; future measurements are delegated."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

from .touch_transition import TouchTransitionPattern
from .level_interaction_history import HistoricalLevelInteractionHistory
from .level_interaction import LevelBarObservation
from .future_outcome import OutcomeAnchor, ForwardOutcome, DEFAULT_FORWARD_HORIZONS, measure_forward_outcomes


class ResearchAnchorKind(StrEnum):
    TOUCH_TRANSITION = "TOUCH_TRANSITION"
    UNTOUCHED_CHECKPOINT = "UNTOUCHED_CHECKPOINT"


def _positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError("expected positive integer")


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("expected timezone-aware UTC timestamp")


@dataclass(frozen=True)
class GeometryResearchAnchor:
    kind: ResearchAnchorKind
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
    evidence_end_utc: datetime
    base_minutes: int
    pattern_index: int | None
    pattern_code: str | None
    checkpoint_bars: int | None
    outcome_anchor: OutcomeAnchor

    def __post_init__(self):
        if not isinstance(self.kind, ResearchAnchorKind):
            raise ValueError("invalid research anchor kind")
        _positive(self.base_minutes)
        _utc(self.evidence_end_utc)
        _utc(self.geometry_available_at_utc)
        if self.evidence_end_utc < self.geometry_available_at_utc:
            raise ValueError("evidence precedes geometry availability")
        if not isinstance(self.outcome_anchor, OutcomeAnchor):
            raise ValueError("outcome_anchor must be OutcomeAnchor")
        if self.outcome_anchor.anchor_time_utc != self.evidence_end_utc:
            raise ValueError("outcome anchor must equal evidence end")
        if self.kind is ResearchAnchorKind.TOUCH_TRANSITION:
            if type(self.pattern_index) is not int or self.pattern_index < 0:
                raise ValueError("invalid pattern_index")
            if not isinstance(self.pattern_code, str) or not self.pattern_code.strip() or self.checkpoint_bars is not None:
                raise ValueError("invalid touch source fields")
        else:
            if self.pattern_index is not None or self.pattern_code is not None:
                raise ValueError("invalid checkpoint source fields")
            _positive(self.checkpoint_bars)


_IDENTITY = ("geometry_family_id", "source_local_trading_date", "source_block_label",
    "source_block_index", "source_block_start_utc", "source_block_end_utc",
    "geometry_available_at_utc", "level_id", "coordinate", "level_price")


def _identity(source):
    return {field: getattr(source, field) for field in _IDENTITY}


def build_touch_transition_research_anchor(pattern: TouchTransitionPattern) -> GeometryResearchAnchor | None:
    if not isinstance(pattern, TouchTransitionPattern):
        raise TypeError("pattern must be TouchTransitionPattern")
    if not pattern.has_next_side or pattern.next_run is None or not pattern.next_run.observations:
        return None
    observation = pattern.next_run.observations[0]
    if not isinstance(observation, LevelBarObservation):
        raise ValueError("invalid confirmation observation")
    if _identity(observation) != _identity(pattern):
        raise ValueError("confirmation identity mismatch")
    _utc(observation.bar_open_time_utc)
    _utc(observation.bar_close_time_utc)
    duration = observation.bar_close_time_utc - observation.bar_open_time_utc
    minute = timedelta(minutes=1)
    if duration <= timedelta(0) or duration % minute:
        raise ValueError("confirmation duration must be positive whole minutes")
    end = observation.bar_close_time_utc
    return GeometryResearchAnchor(
        kind=ResearchAnchorKind.TOUCH_TRANSITION, **_identity(pattern), evidence_end_utc=end,
        base_minutes=duration // minute, pattern_index=pattern.pattern_index,
        pattern_code=pattern.pattern_code, checkpoint_bars=None,
        outcome_anchor=OutcomeAnchor(f"TOUCH_TRANSITION_{pattern.pattern_index}", end, observation.close))


def _validate_history(history):
    if not isinstance(history, HistoricalLevelInteractionHistory):
        raise TypeError("history must be HistoricalLevelInteractionHistory")
    if len(history.observations) != history.observed_bar_count:
        raise ValueError("inconsistent observation count")
    _positive(history.base_minutes)
    previous = None
    for observation in history.observations:
        if not isinstance(observation, LevelBarObservation) or _identity(observation) != _identity(history):
            raise ValueError("inconsistent observation identity")
        _utc(observation.bar_open_time_utc)
        _utc(observation.bar_close_time_utc)
        if previous is not None and observation.bar_open_time_utc <= previous:
            raise ValueError("observations must be strictly chronological")
        previous = observation.bar_open_time_utc


def _checkpoint(history, count):
    if history.observed_bar_count < count:
        return None
    prefix = history.observations[:count]
    step = timedelta(minutes=history.base_minutes)
    for i, observation in enumerate(prefix):
        expected = history.geometry_available_at_utc + i * step
        if observation.bar_open_time_utc != expected or observation.bar_close_time_utc != expected + step:
            raise ValueError("discontinuous checkpoint timing")
    if any(o.range_touched_level is True for o in prefix):
        return None
    last = prefix[-1]
    return GeometryResearchAnchor(
        kind=ResearchAnchorKind.UNTOUCHED_CHECKPOINT, **_identity(history),
        evidence_end_utc=last.bar_close_time_utc, base_minutes=history.base_minutes,
        pattern_index=None, pattern_code=None, checkpoint_bars=count,
        outcome_anchor=OutcomeAnchor(f"UNTOUCHED_CHECKPOINT_{count}", last.bar_close_time_utc, last.close))


def build_untouched_checkpoint_research_anchor(history: HistoricalLevelInteractionHistory,
                                               checkpoint_bars: int) -> GeometryResearchAnchor | None:
    _positive(checkpoint_bars)
    _validate_history(history)
    return _checkpoint(history, checkpoint_bars)


DEFAULT_UNTOUCHED_CHECKPOINTS = (4, 8, 16, 32)


def build_untouched_checkpoint_research_anchors(history, checkpoints=DEFAULT_UNTOUCHED_CHECKPOINTS) -> tuple[GeometryResearchAnchor, ...]:
    checkpoints = tuple(checkpoints)
    for count in checkpoints:
        _positive(count)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("duplicate checkpoints")
    _validate_history(history)
    return tuple(anchor for count in checkpoints if (anchor := _checkpoint(history, count)) is not None)


def measure_research_anchor_outcomes(research_anchor: GeometryResearchAnchor, bars,
                                     horizons=DEFAULT_FORWARD_HORIZONS) -> tuple[ForwardOutcome, ...]:
    if not isinstance(research_anchor, GeometryResearchAnchor):
        raise TypeError("research_anchor must be GeometryResearchAnchor")
    return measure_forward_outcomes(research_anchor.outcome_anchor, bars, horizons, research_anchor.base_minutes)
