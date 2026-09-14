from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone, date

import pytest
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor, ForwardOutcome
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.multitimeframe_context import MultiTimeframeSnapshot, CompletedFrameContext
from sandbox.market_state.research_multitimeframe_context import ResearchAnchorMultiTimeframeContext
from sandbox.market_state.session_summary import CompletedSessionSummary, SessionDirection
from sandbox.market_state.research_session_context import ResearchSessionSlot, ResearchAnchorSessionContext
from sandbox.market_state.research_session_relationship import SessionRelationshipSpec, build_research_session_relationships
from sandbox.market_state.research_record import assemble_research_record
from sandbox.market_state.discovery_projection import project_research_record, values_by_id
from sandbox.market_state import discovery_context_projection as projection

TIME = datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)
SHAPE = ("direction", "range_fraction", "body_to_range", "close_location")
REL = ("observed_open_position", "observed_close_position", "traded_above_reference_high", "traded_below_reference_low",
    "closed_above_reference_high", "closed_below_reference_low", "high_excursion_returned_inside_by_close",
    "low_excursion_returned_inside_by_close", "stayed_inside_reference_range", "traded_both_sides_of_reference_range",
    "range_ratio", "same_direction", "overlaps_in_time", "overlap_minutes")


def record(scale=1, flat=False, missing=False):
    o,h,l,c = (100,100,100,100) if flat else (100,120,80,110)
    o,h,l,c = (v*scale for v in (o,h,l,c))
    start = TIME-timedelta(hours=8)
    anchor = GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION, "CUSTOM", start.date(), "C1", 0,
        start, start+timedelta(hours=3), start+timedelta(hours=3), "LEVEL", -3.3, 90*scale, TIME, 15,
        0, "ABOVE_TOUCH_BELOW", None, OutcomeAnchor("EVIDENCE", TIME, c))
    candle = AnchoredBlockCandle(start.date(), "C3", 2, start, start+timedelta(hours=3), start,
        start+timedelta(hours=3), o,h,l,c,12,12)
    frames = (CompletedFrameContext("Z_CUSTOM", candle.block_start_utc,candle.block_end_utc, None if missing else candle),
              CompletedFrameContext("NY_H3", candle.block_start_utc,candle.block_end_utc,None))
    mtf = ResearchAnchorMultiTimeframeContext(anchor,MultiTimeframeSnapshot(TIME,OHLCBar(TIME-STEP,o,h,l,c),frames),
        tuple(f.frame_id for f in frames if f.candle is None),False)
    slots=[]
    for name,active,absent in (("Z_CUSTOM",False,missing),("A_CUSTOM",False,False),("ACTIVE",True,True),("MISSING",False,True)):
        end=TIME+STEP if active else TIME-STEP
        summary=None if absent else CompletedSessionSummary(name,start.date(),start,end,start,end,o,h,l,c,h-l,
            SessionDirection.FLAT if flat else SessionDirection.BULLISH,32,32)
        slots.append(ResearchSessionSlot(name,start.date(),start,end,active,summary))
    sessions=ResearchAnchorSessionContext(anchor,tuple(slots),tuple(s.session_id for s in slots if s.summary),
        ("ACTIVE",),tuple(s.session_id for s in slots if not s.active_at_anchor and s.summary is None),False)
    relations=build_research_session_relationships(sessions,(
        SessionRelationshipSpec("Z_PAIR","Z_CUSTOM","A_CUSTOM"),SessionRelationshipSpec("A_PAIR","ACTIVE","MISSING")))
    outcome=ForwardOutcome("EVIDENCE",TIME,c,1,15,TIME,TIME+STEP,TIME,TIME,c,c,h,l,0,0,0,0,TIME,TIME,0,0)
    return assemble_research_record(mtf,relations,(outcome,))


def test_core_preserved_and_called_once(monkeypatch):
    source=record()
    core=project_research_record(source)
    calls=[]
    def tracked(r):
        calls.append(r)
        return core
    monkeypatch.setattr(projection,"project_research_record",tracked)
    row=projection.project_research_record_with_context(source)
    assert calls==[source]
    assert row.record is source
    assert row.identity is core.identity
    assert row.targets is core.targets
    assert row.predictors[:len(core.predictors)]==core.predictors
    assert len(core.predictors)==8


def test_exact_order_and_metrics():
    source=record()
    row=projection.project_research_record_with_context(source)
    ids=[v.field_id for v in row.predictors[8:]]
    expected=["x.m15."+m for m in SHAPE]
    for name in ("z_custom","ny_h3"):
        expected += [f"x.mtf.{name}."+m for m in ("available","block_label","block_index",*SHAPE)]
    for name in ("z_custom","a_custom","active","missing"):
        expected += [f"x.session.{name}."+m for m in ("available","active","missing_completed",*SHAPE)]
    for name in ("z_pair","a_pair"):
        expected += [f"x.relationship.{name}."+m for m in ("available",*REL)]
    assert ids[:len(expected)]==expected
    assert ids[len(expected):] == [
        "x.market.m15.available", *("x.market.m15." + name for name, _ in projection._MARKET_FIELDS),
        *("x.market.m15.structure." + name for name in projection._BREAK_FLAGS),
        "x.level_transition.sweep", "x.level_transition.reclaim",
        "x.daily.previous.available", "x.daily.previous.open", "x.daily.previous.high",
        "x.daily.previous.low", "x.daily.previous.close", "x.daily.previous.range",
        "x.daily.previous.midpoint",
        *(f"x.daily.previous.level.{level}.{metric}"
          for level in projection._DAILY_LEVELS for metric in projection._LEVEL_MEASUREMENTS),
    ]
    values=values_by_id(row.predictors)
    for prefix in ("x.m15.","x.mtf.z_custom.","x.session.z_custom."):
        assert values[prefix+"direction"]=="BULLISH"
        assert values[prefix+"range_fraction"]==pytest.approx(40/110)
        assert values[prefix+"body_to_range"]==.25
        assert values[prefix+"close_location"]==.75
    assert values["x.mtf.z_custom.block_label"]=="C3"
    assert values["x.mtf.z_custom.block_index"]==2
    assert values["x.session.z_custom.available"] is True
    assert values["x.session.z_custom.active"] is False
    assert values["x.session.z_custom.missing_completed"] is False
    relation=source.session_relationship_context.relationships[0].relationship
    for metric in REL:
        expected=getattr(relation,metric)
        if metric in ("observed_open_position","observed_close_position"):
            expected=expected.value
            assert type(values["x.relationship.z_pair."+metric]) is str
        assert values["x.relationship.z_pair."+metric]==expected


def test_stable_missingness():
    present=projection.project_research_record_with_context(record())
    missing=projection.project_research_record_with_context(record(missing=True))
    assert [v.field_id for v in present.predictors]==[v.field_id for v in missing.predictors]
    values=values_by_id(missing.predictors)
    for name in ("z_custom","ny_h3"):
        assert values[f"x.mtf.{name}.available"] is False
        assert all(values[f"x.mtf.{name}."+m] is None for m in ("block_label","block_index",*SHAPE))
    for name,active in (("active",True),("missing",False)):
        assert values[f"x.session.{name}.available"] is False
        assert values[f"x.session.{name}.active"] is active
        assert values[f"x.session.{name}.missing_completed"] is (not active)
        assert all(values[f"x.session.{name}."+m] is None for m in SHAPE)
    for name in ("z_pair","a_pair"):
        assert values[f"x.relationship.{name}.available"] is False
        assert all(values[f"x.relationship.{name}."+m] is None for m in REL)


def test_zero_range():
    values=values_by_id(projection.project_research_record_with_context(record(flat=True)).predictors)
    for prefix in ("x.m15.","x.mtf.z_custom.","x.session.z_custom."):
        assert values[prefix+"direction"]=="FLAT"
        assert values[prefix+"range_fraction"]==0
        assert values[prefix+"body_to_range"] is None
        assert values[prefix+"close_location"] is None


def test_future_leakage_and_raw_price_scale_invariance():
    source=record()
    changed=replace(source,outcomes=(replace(source.outcomes[0],end_close=999,highest_high=1000,lowest_low=1,
        close_change=889,close_return_fraction=8,max_upward_excursion=890,max_downward_excursion=109),))
    first=projection.project_research_record_with_context(source)
    second=projection.project_research_record_with_context(changed)
    assert first.identity==second.identity
    assert first.predictors==second.predictors
    assert first.targets!=second.targets
    scaled=projection.project_research_record_with_context(record(scale=2))
    assert first.predictors==scaled.predictors
    assert first.identity!=scaled.identity
    legacy_count = 4 + 2*7 + 4*7 + 2*15
    for value in first.predictors[8:8+legacy_count]:
        assert not isinstance(value.value,(date,datetime))
        assert value.field_id.split(".")[-1] not in ("open","high","low","close","range","level_price","reference_price")


@pytest.mark.parametrize("identifier", ["", "lower", "A-B", "A B", "A\n", 123])
def test_invalid_machine_id(identifier):
    with pytest.raises(ValueError):
        projection._segment(identifier)


def test_invalid_context_id_through_builder():
    source=record()
    frame=replace(source.multitimeframe_context.snapshot.frames[0],frame_id="BAD-ID")
    snapshot=replace(source.multitimeframe_context.snapshot,frames=(frame,))
    mtf=replace(source.multitimeframe_context,snapshot=snapshot,missing_frame_ids=(),all_frames_available=True)
    with pytest.raises(ValueError):
        projection.project_research_record_with_context(replace(source,multitimeframe_context=mtf))


def test_zero_reference_and_bearish_shape():
    class Candle:
        open=110
        high=120
        low=80
        close=100
    assert projection._shape(Candle(),0)==("BEARISH",None,.25,.5)
    assert projection._shape(Candle(),-100)==("BEARISH",.4,.25,.5)


def test_invalid_record_and_immutable():
    with pytest.raises(TypeError):
        projection.project_research_record_with_context(object())
    row=projection.project_research_record_with_context(record())
    with pytest.raises(FrozenInstanceError):
        row.predictors=()
