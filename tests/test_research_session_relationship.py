from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import pytest
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor
from sandbox.market_state.session_summary import CompletedSessionSummary, SessionDirection
from sandbox.market_state.research_session_context import ResearchSessionSlot, ResearchAnchorSessionContext
from sandbox.market_state.session_relationship import compare_completed_sessions
from sandbox.market_state import research_session_relationship as module

START = datetime(2024, 1, 15, tzinfo=timezone.utc)


def context(early=False, missing=(), ids=("TOKYO", "LONDON", "NEW_YORK")):
    decision = START + timedelta(hours=15, minutes=45) if early else START + timedelta(hours=23)
    anchor = GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION, "CUSTOM", START.date(), "C1", 0,
        START, START + timedelta(hours=3), START + timedelta(hours=3), "LEVEL", 0, 10,
        decision, 15, 0, "ABOVE_TOUCH_ABOVE", None, OutcomeAnchor("EVIDENCE", decision, 11))
    slots = []
    for i, (name, start_hour, end_hour) in enumerate(zip(ids, (0, 8, 13), (9, 17, 22))):
        start, end = START + timedelta(hours=start_hour), START + timedelta(hours=end_hour)
        active = start <= decision < end
        summary = None if active or name in missing else CompletedSessionSummary(name, START.date(),
            start, end, start, end, 10, 12+i, 8, 11, 4+i, SessionDirection.BULLISH, 36, 36)
        slots.append(ResearchSessionSlot(name, START.date(), start, end, active, summary))
    return ResearchAnchorSessionContext(anchor, tuple(slots),
        tuple(s.session_id for s in slots if s.summary is not None),
        tuple(s.session_id for s in slots if s.active_at_anchor),
        tuple(s.session_id for s in slots if not s.active_at_anchor and s.summary is None),
        all(s.summary is not None for s in slots))


def test_1045_unavailable_no_comparison(monkeypatch):
    def forbidden(*args):
        raise AssertionError("unavailable summaries must not be compared")
    monkeypatch.setattr(module, "compare_completed_sessions", forbidden)
    result = module.build_research_session_relationships(context(early=True))
    assert result.available_relationship_ids == ()
    assert result.unavailable_relationship_ids == ("TOKYO_TO_LONDON", "LONDON_TO_NEW_YORK")
    assert [s.unavailable_session_ids for s in result.relationships] == [("LONDON",), ("LONDON", "NEW_YORK")]
    assert all(not s.available and s.relationship is None for s in result.relationships)


def test_completed_delegation_original_objects(monkeypatch):
    source = context()
    calls = []
    def tracked(a, b):
        calls.append((a, b))
        return compare_completed_sessions(a, b)
    monkeypatch.setattr(module, "compare_completed_sessions", tracked)
    result = module.build_research_session_relationships(source)
    assert result.session_context is source
    assert len(calls) == 2
    assert result.available_relationship_ids == ("TOKYO_TO_LONDON", "LONDON_TO_NEW_YORK")
    assert result.unavailable_relationship_ids == ()
    for i, slot in enumerate(result.relationships):
        assert slot.reference_summary is source.sessions[i].summary
        assert slot.observed_summary is source.sessions[i+1].summary
        assert slot.relationship == compare_completed_sessions(slot.reference_summary, slot.observed_summary)
        assert slot.available is True
        assert slot.unavailable_session_ids == ()


@pytest.mark.parametrize("missing", [("TOKYO",), ("LONDON",), ("TOKYO", "LONDON")])
def test_missing_completed(missing):
    slot, = module.build_research_session_relationships(context(missing=missing), (module.TOKYO_TO_LONDON,)).relationships
    assert slot.unavailable_session_ids == missing
    assert slot.relationship is None
    assert slot.available is False


def test_custom_order_once():
    source = context(ids=("ASIA_CUSTOM", "EUROPE_CUSTOM", "US_CUSTOM"))
    specs = (module.SessionRelationshipSpec("EUROPE_TO_US", "EUROPE_CUSTOM", "US_CUSTOM"),
        module.SessionRelationshipSpec("ASIA_TO_EUROPE", "ASIA_CUSTOM", "EUROPE_CUSTOM"))
    class Once:
        calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from specs
    iterable = Once()
    result = module.build_research_session_relationships(source, iterable)
    assert iterable.calls == 1
    assert result.available_relationship_ids == ("EUROPE_TO_US", "ASIA_TO_EUROPE")


@pytest.mark.parametrize("a,b", [("UNKNOWN", "LONDON"), ("TOKYO", "UNKNOWN")])
def test_unknown(a, b):
    with pytest.raises(ValueError, match="unknown"):
        module.build_research_session_relationships(context(), (module.SessionRelationshipSpec("REL", a, b),))


def test_invalid_inputs():
    with pytest.raises(TypeError):
        module.build_research_session_relationships(object())
    with pytest.raises(TypeError):
        module.build_research_session_relationships(context(), (object(),))
    with pytest.raises(ValueError):
        module.build_research_session_relationships(context(), (module.TOKYO_TO_LONDON,) * 2)
    with pytest.raises(ValueError):
        module.SessionRelationshipSpec("REL", "A", "A")


@pytest.mark.parametrize("field", ["relationship_id", "reference_session_id", "observed_session_id"])
@pytest.mark.parametrize("value", ["", "lower", "A-B", "A B", "A\n", 123])
def test_invalid_ids(field, value):
    data = dict(relationship_id="REL", reference_session_id="A", observed_session_id="B")
    with pytest.raises(ValueError):
        module.SessionRelationshipSpec(**(data | {field: value}))


@pytest.mark.parametrize("changes", [{"available": False}, {"unavailable_session_ids": ("TOKYO",)},
    {"relationship": None}, {"reference_summary": None}, {"observed_summary": object()}])
def test_invalid_available_slot(changes):
    slot = module.build_research_session_relationships(context()).relationships[0]
    with pytest.raises(ValueError):
        replace(slot, **changes)


@pytest.mark.parametrize("field,value", [("reference_session_id", "OTHER"), ("observed_session_id", "OTHER"),
    ("reference_local_session_date", (START+timedelta(days=1)).date()),
    ("observed_local_session_date", (START+timedelta(days=1)).date()),
    ("reference_start_utc", START-timedelta(hours=1)), ("reference_end_utc", START),
    ("observed_start_utc", START), ("observed_end_utc", START)])
def test_mismatched_relationship_metadata(field, value):
    slot = module.build_research_session_relationships(context()).relationships[0]
    with pytest.raises(ValueError):
        replace(slot, relationship=replace(slot.relationship, **{field: value}))


def test_invalid_unavailable_slot():
    slot = module.build_research_session_relationships(context(early=True)).relationships[0]
    complete = module.build_research_session_relationships(context()).relationships[0]
    for changes in ({"available": True}, {"unavailable_session_ids": ()}, {"relationship": complete.relationship}):
        with pytest.raises(ValueError):
            replace(slot, **changes)


def test_parent_consistency_immutable():
    result = module.build_research_session_relationships(context())
    for changes in ({"session_context": object()}, {"relationships": (object(),)},
        {"relationships": (result.relationships[0],)*2}, {"available_relationship_ids": ()},
        {"unavailable_relationship_ids": ("TOKYO_TO_LONDON",)}):
        with pytest.raises((TypeError, ValueError)):
            replace(result, **changes)
    copied = replace(result.relationships[0], reference_summary=replace(result.relationships[0].reference_summary))
    with pytest.raises(ValueError, match="original"):
        replace(result, relationships=(copied, result.relationships[1]))
    with pytest.raises(FrozenInstanceError):
        result.relationships = ()
    with pytest.raises(FrozenInstanceError):
        result.relationships[0].available = False
