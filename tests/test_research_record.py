from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor, ForwardOutcome
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.multitimeframe_context import MultiTimeframeSnapshot, CompletedFrameContext
from sandbox.market_state.research_multitimeframe_context import ResearchAnchorMultiTimeframeContext
from sandbox.market_state.research_session_context import ResearchSessionSlot, ResearchAnchorSessionContext
from sandbox.market_state.research_session_relationship import ResearchSessionRelationshipSlot, ResearchAnchorSessionRelationshipContext
from sandbox.market_state.research_record import ResearchRecord, assemble_research_record

TIME = datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def evidence(checkpoint=False):
    start = TIME - timedelta(hours=8)
    anchor = GeometryResearchAnchor(
        ResearchAnchorKind.UNTOUCHED_CHECKPOINT if checkpoint else ResearchAnchorKind.TOUCH_TRANSITION,
        "CUSTOM", start.date(), "C1", 0, start, start + timedelta(hours=3), start + timedelta(hours=3),
        "LEVEL", 0, 10, TIME, 15, None if checkpoint else 0,
        None if checkpoint else "ABOVE_TOUCH_ABOVE", 20 if checkpoint else None,
        OutcomeAnchor("EVIDENCE", TIME, 11))
    base = OHLCBar(TIME - STEP, 10, 12, 9, 11)
    frame = CompletedFrameContext("CUSTOM_FRAME", TIME - timedelta(hours=2), TIME - timedelta(hours=1), None)
    mtf = ResearchAnchorMultiTimeframeContext(anchor, MultiTimeframeSnapshot(TIME, base, (frame,)), ("CUSTOM_FRAME",), False)
    active = ResearchSessionSlot("ACTIVE", TIME.date(), TIME - timedelta(hours=1), TIME + timedelta(hours=1), True, None)
    missing = ResearchSessionSlot("MISSING", TIME.date(), TIME - timedelta(hours=3), TIME - timedelta(hours=2), False, None)
    sessions = ResearchAnchorSessionContext(anchor, (active, missing), (), ("ACTIVE",), ("MISSING",), False)
    relation = ResearchSessionRelationshipSlot("REL", "ACTIVE", "MISSING", None, None, None, False, ("ACTIVE", "MISSING"))
    context = ResearchAnchorSessionRelationshipContext(sessions, (relation,), (), ("REL",))
    return mtf, context


def outcome(horizon):
    return ForwardOutcome("EVIDENCE", TIME, 11, horizon, 15, TIME, TIME + horizon * STEP,
        TIME, TIME + (horizon - 1) * STEP, 11, 12, 13, 10, 1, 1/11, 2, 1, TIME, TIME, 0, 0)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_assembly_identity_missingness_and_causal_windows(checkpoint):
    mtf, sessions = evidence(checkpoint)
    outcomes = (outcome(1), outcome(4))
    record = assemble_research_record(mtf, sessions, outcomes)
    assert record.anchor is mtf.anchor
    assert record.multitimeframe_context is mtf
    assert record.session_relationship_context is sessions
    assert all(a is b for a, b in zip(record.outcomes, outcomes))
    assert record.multitimeframe_context.snapshot.frames[0].candle is None
    assert record.session_relationship_context.session_context.active_session_ids == ("ACTIVE",)
    assert record.session_relationship_context.session_context.missing_completed_session_ids == ("MISSING",)
    assert record.session_relationship_context.relationships[0].relationship is None
    assert record.outcomes[0].window_end_utc == TIME + STEP
    assert record.outcomes[1].window_end_utc == TIME + timedelta(hours=1)
    assert record.outcomes[1].last_bar_open_time_utc == TIME + 3 * STEP
    with pytest.raises(ValueError):
        assemble_research_record(mtf, sessions, (replace(outcomes[1], first_bar_open_time_utc=TIME + STEP),))


def test_generator_once_and_order():
    mtf, sessions = evidence()
    values = (outcome(4), outcome(1))
    class Once:
        calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from values
    source = Once()
    result = assemble_research_record(mtf, sessions, source)
    assert source.calls == 1
    assert [o.horizon_bars for o in result.outcomes] == [4, 1]
    assert result.outcomes[0] is values[0]
    assert result.outcomes[1] is values[1]


@pytest.mark.parametrize("values", [(), (object(),), (outcome(1), outcome(1))])
def test_bad_outcome_collections(values):
    with pytest.raises((TypeError, ValueError)):
        assemble_research_record(*evidence(), values)


def test_bad_context_and_anchor_types():
    mtf, sessions = evidence()
    with pytest.raises(TypeError):
        assemble_research_record(object(), sessions, (outcome(1),))
    with pytest.raises(TypeError):
        assemble_research_record(mtf, object(), (outcome(1),))
    with pytest.raises(TypeError):
        ResearchRecord(object(), mtf, sessions, (outcome(1),))


def test_anchor_value_equality_and_mismatches():
    mtf, sessions = evidence()
    equivalent = replace(mtf.anchor)
    assert equivalent is not mtf.anchor
    session_copy = replace(sessions, session_context=replace(sessions.session_context, anchor=equivalent))
    assert assemble_research_record(mtf, session_copy, (outcome(1),)).anchor == equivalent
    changed = replace(mtf.anchor, level_id="OTHER")
    with pytest.raises(ValueError, match="anchor mismatch"):
        ResearchRecord(changed, mtf, sessions, (outcome(1),))
    bad_sessions = replace(sessions, session_context=replace(sessions.session_context, anchor=changed))
    with pytest.raises(ValueError, match="anchor mismatch"):
        assemble_research_record(mtf, bad_sessions, (outcome(1),))


@pytest.mark.parametrize("field,value", [("anchor_id", "OTHER"), ("anchor_time_utc", TIME + STEP),
    ("reference_price", 99), ("base_minutes", 30), ("window_start_utc", TIME + STEP),
    ("first_bar_open_time_utc", TIME + STEP), ("window_end_utc", TIME + STEP),
    ("last_bar_open_time_utc", TIME + STEP)])
def test_outcome_consistency(field, value):
    with pytest.raises(ValueError):
        assemble_research_record(*evidence(), (replace(outcome(4), **{field: value}),))


@pytest.mark.parametrize("field", ["horizon_bars", "base_minutes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_positive_integer_fields(field, value):
    with pytest.raises(ValueError):
        assemble_research_record(*evidence(), (replace(outcome(4), **{field: value}),))


def test_immutable_and_replaced_evidence_no_recalculation():
    original = assemble_research_record(*evidence(), (outcome(1),))
    with pytest.raises(FrozenInstanceError):
        original.outcomes = ()
    changed_outcome = replace(original.outcomes[0], highest_high=99)
    changed = replace(original, outcomes=(changed_outcome,))
    assert changed != original
    assert changed.outcomes[0] is changed_outcome
    assert original.outcomes[0].highest_high == 13
    assert changed.multitimeframe_context is original.multitimeframe_context
