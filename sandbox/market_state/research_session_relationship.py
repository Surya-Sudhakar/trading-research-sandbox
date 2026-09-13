"""Compare only the exact completed summaries in an existing causal context."""
from dataclasses import dataclass
import re
from .research_session_context import ResearchAnchorSessionContext
from .session_summary import CompletedSessionSummary
from .session_relationship import CompletedSessionRelationship, compare_completed_sessions


@dataclass(frozen=True)
class SessionRelationshipSpec:
    relationship_id: str
    reference_session_id: str
    observed_session_id: str

    def __post_init__(self):
        for value in (self.relationship_id, self.reference_session_id, self.observed_session_id):
            if not isinstance(value, str) or re.fullmatch(r"[A-Z0-9_]+", value) is None:
                raise ValueError("invalid machine identifier")
        if self.reference_session_id == self.observed_session_id:
            raise ValueError("reference and observed sessions must differ")


TOKYO_TO_LONDON = SessionRelationshipSpec("TOKYO_TO_LONDON", "TOKYO", "LONDON")
LONDON_TO_NEW_YORK = SessionRelationshipSpec("LONDON_TO_NEW_YORK", "LONDON", "NEW_YORK")
DEFAULT_SESSION_RELATIONSHIPS = (TOKYO_TO_LONDON, LONDON_TO_NEW_YORK)


@dataclass(frozen=True)
class ResearchSessionRelationshipSlot:
    relationship_id: str
    reference_session_id: str
    observed_session_id: str
    reference_summary: CompletedSessionSummary | None
    observed_summary: CompletedSessionSummary | None
    relationship: CompletedSessionRelationship | None
    available: bool
    unavailable_session_ids: tuple[str, ...]

    def __post_init__(self):
        SessionRelationshipSpec(self.relationship_id, self.reference_session_id, self.observed_session_id)
        missing = []
        for side in ("reference", "observed"):
            summary = getattr(self, side + "_summary")
            session_id = getattr(self, side + "_session_id")
            if summary is None:
                missing.append(session_id)
            elif not isinstance(summary, CompletedSessionSummary) or summary.session_id != session_id:
                raise ValueError("summary type or ID mismatch")
        if type(self.available) is not bool or self.available != (not missing) or tuple(self.unavailable_session_ids) != tuple(missing):
            raise ValueError("inconsistent availability")
        if missing:
            if self.relationship is not None:
                raise ValueError("unavailable relationship must be None")
        else:
            if not isinstance(self.relationship, CompletedSessionRelationship):
                raise ValueError("available slot requires CompletedSessionRelationship")
            for side in ("reference", "observed"):
                summary = getattr(self, side + "_summary")
                for field in ("session_id", "local_session_date", "start_utc", "end_utc"):
                    if getattr(self.relationship, side + "_" + field) != getattr(summary, field):
                        raise ValueError("relationship source metadata mismatch")
        object.__setattr__(self, "unavailable_session_ids", tuple(missing))


@dataclass(frozen=True)
class ResearchAnchorSessionRelationshipContext:
    session_context: ResearchAnchorSessionContext
    relationships: tuple[ResearchSessionRelationshipSlot, ...]
    available_relationship_ids: tuple[str, ...]
    unavailable_relationship_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.session_context, ResearchAnchorSessionContext):
            raise TypeError("session_context must be ResearchAnchorSessionContext")
        slots = tuple(self.relationships)
        if any(not isinstance(s, ResearchSessionRelationshipSlot) for s in slots):
            raise ValueError("invalid relationship slot")
        if len({s.relationship_id for s in slots}) != len(slots):
            raise ValueError("duplicate relationship IDs")
        lookup = {s.session_id: s for s in self.session_context.sessions}
        for slot in slots:
            for side in ("reference", "observed"):
                session_id = getattr(slot, side + "_session_id")
                if session_id not in lookup or getattr(slot, side + "_summary") is not lookup[session_id].summary:
                    raise ValueError("slot must use original context summaries")
        available = tuple(s.relationship_id for s in slots if s.available)
        unavailable = tuple(s.relationship_id for s in slots if not s.available)
        if tuple(self.available_relationship_ids) != available or tuple(self.unavailable_relationship_ids) != unavailable:
            raise ValueError("inconsistent relationship ID summaries")
        object.__setattr__(self, "relationships", slots)
        object.__setattr__(self, "available_relationship_ids", available)
        object.__setattr__(self, "unavailable_relationship_ids", unavailable)


def build_research_session_relationships(session_context: ResearchAnchorSessionContext,
                                         relationships=DEFAULT_SESSION_RELATIONSHIPS) -> ResearchAnchorSessionRelationshipContext:
    if not isinstance(session_context, ResearchAnchorSessionContext):
        raise TypeError("session_context must be ResearchAnchorSessionContext")
    specs = tuple(relationships)
    if any(not isinstance(s, SessionRelationshipSpec) for s in specs):
        raise TypeError("relationships must contain SessionRelationshipSpec")
    if len({s.relationship_id for s in specs}) != len(specs):
        raise ValueError("duplicate relationship IDs")
    lookup = {s.session_id: s for s in session_context.sessions}
    for spec in specs:
        if spec.reference_session_id not in lookup or spec.observed_session_id not in lookup:
            raise ValueError("unknown session ID")
    slots = []
    for spec in specs:
        reference = lookup[spec.reference_session_id].summary
        observed = lookup[spec.observed_session_id].summary
        missing = tuple(name for name, summary in ((spec.reference_session_id, reference),
            (spec.observed_session_id, observed)) if summary is None)
        relationship = None if missing else compare_completed_sessions(reference, observed)
        slots.append(ResearchSessionRelationshipSlot(spec.relationship_id, spec.reference_session_id,
            spec.observed_session_id, reference, observed, relationship, not missing, missing))
    return ResearchAnchorSessionRelationshipContext(session_context, tuple(slots),
        tuple(s.relationship_id for s in slots if s.available),
        tuple(s.relationship_id for s in slots if not s.available))
