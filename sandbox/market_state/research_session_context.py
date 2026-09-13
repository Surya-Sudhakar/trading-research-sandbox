"""Causal relevant-session joins with no active or stale summary fallback."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .research_anchor import GeometryResearchAnchor
from .session_summary import (
    SessionConfig, SessionWindow, CompletedSessionSummary, DEFAULT_SESSIONS,
    session_window_for_date, aggregate_completed_sessions,
)


@dataclass(frozen=True)
class ResearchSessionSlot:
    session_id: str
    target_local_session_date: date
    target_start_utc: datetime
    target_end_utc: datetime
    active_at_anchor: bool
    summary: CompletedSessionSummary | None

    def __post_init__(self):
        if type(self.active_at_anchor) is not bool:
            raise ValueError("active_at_anchor must be bool")
        if self.summary is not None:
            if not isinstance(self.summary, CompletedSessionSummary):
                raise ValueError("invalid session summary")
            if (self.active_at_anchor or self.summary.session_id != self.session_id
                    or self.summary.local_session_date != self.target_local_session_date
                    or self.summary.start_utc != self.target_start_utc
                    or self.summary.end_utc != self.target_end_utc):
                raise ValueError("inconsistent slot summary")


@dataclass(frozen=True)
class ResearchAnchorSessionContext:
    anchor: GeometryResearchAnchor
    sessions: tuple[ResearchSessionSlot, ...]
    available_session_ids: tuple[str, ...]
    active_session_ids: tuple[str, ...]
    missing_completed_session_ids: tuple[str, ...]
    all_session_summaries_available: bool

    def __post_init__(self):
        if not isinstance(self.anchor, GeometryResearchAnchor):
            raise TypeError("anchor must be GeometryResearchAnchor")
        slots = tuple(self.sessions)
        if any(not isinstance(s, ResearchSessionSlot) for s in slots):
            raise ValueError("sessions must contain ResearchSessionSlot")
        if len({s.session_id for s in slots}) != len(slots):
            raise ValueError("duplicate session IDs")
        for slot in slots:
            active = slot.target_start_utc <= self.anchor.evidence_end_utc < slot.target_end_utc
            if slot.active_at_anchor != active or (not active and slot.target_end_utc > self.anchor.evidence_end_utc):
                raise ValueError("slot timing disagrees with anchor")
        expected = _summaries(slots)
        actual = (tuple(self.available_session_ids), tuple(self.active_session_ids),
                  tuple(self.missing_completed_session_ids), self.all_session_summaries_available)
        if actual != expected:
            raise ValueError("inconsistent session summary tuples")
        object.__setattr__(self, "sessions", slots)
        for field, value in zip(("available_session_ids", "active_session_ids", "missing_completed_session_ids"), expected[:3]):
            object.__setattr__(self, field, value)


def _summaries(slots):
    return (tuple(s.session_id for s in slots if s.summary is not None),
            tuple(s.session_id for s in slots if s.active_at_anchor),
            tuple(s.session_id for s in slots if not s.active_at_anchor and s.summary is None),
            all(s.summary is not None for s in slots))


def _relevant(config: SessionConfig, decision: datetime) -> SessionWindow:
    day = decision.astimezone(ZoneInfo(config.timezone)).date()
    window = session_window_for_date(config, day)
    # Today's window is relevant once it starts. Otherwise yesterday's window
    # is either still active (overnight) or the latest completed recurrence.
    if window.start_utc > decision:
        window = session_window_for_date(config, day - timedelta(days=1))
    return window


def join_research_anchors_sessions(anchors, bars, sessions=DEFAULT_SESSIONS,
                                   base_minutes: int = 15) -> tuple[ResearchAnchorSessionContext, ...]:
    if type(base_minutes) is not int or base_minutes <= 0:
        raise ValueError("base_minutes must be a positive integer")
    anchors, sessions = tuple(anchors), tuple(sessions)
    for anchor in anchors:
        if not isinstance(anchor, GeometryResearchAnchor):
            raise TypeError("anchors must contain GeometryResearchAnchor")
        if anchor.base_minutes != base_minutes:
            raise ValueError("anchor base_minutes mismatch")
    if any(not isinstance(s, SessionConfig) for s in sessions):
        raise TypeError("sessions must contain SessionConfig")
    if len({s.session_id for s in sessions}) != len(sessions):
        raise ValueError("duplicate session IDs")
    if not anchors:
        return ()
    summaries = aggregate_completed_sessions(bars, max(a.evidence_end_utc for a in anchors), sessions, base_minutes)
    lookup = {(s.session_id, s.local_session_date): s for s in summaries}
    results = []
    for anchor in anchors:
        slots = []
        for config in sessions:
            window = _relevant(config, anchor.evidence_end_utc)
            active = window.start_utc <= anchor.evidence_end_utc < window.end_utc
            summary = None if active else lookup.get((config.session_id, window.local_session_date))
            slots.append(ResearchSessionSlot(config.session_id, window.local_session_date,
                window.start_utc, window.end_utc, active, summary))
        slots = tuple(slots)
        results.append(ResearchAnchorSessionContext(anchor, slots, *_summaries(slots)))
    return tuple(results)


def join_research_anchor_sessions(anchor: GeometryResearchAnchor, bars,
                                  sessions=DEFAULT_SESSIONS) -> ResearchAnchorSessionContext:
    if not isinstance(anchor, GeometryResearchAnchor):
        raise TypeError("anchor must be GeometryResearchAnchor")
    return join_research_anchors_sessions((anchor,), bars, sessions, anchor.base_minutes)[0]
