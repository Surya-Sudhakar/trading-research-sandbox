from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor, ForwardOutcome
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.multitimeframe_context import MultiTimeframeSnapshot
from sandbox.market_state.research_multitimeframe_context import ResearchAnchorMultiTimeframeContext
from sandbox.market_state.research_session_context import ResearchAnchorSessionContext
from sandbox.market_state.research_session_relationship import ResearchAnchorSessionRelationshipContext
from sandbox.market_state.research_record import assemble_research_record
from sandbox.market_state.discovery_projection import DiscoveryRow, DiscoveryValue
from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
from sandbox.research.outcome_distribution import summarize_outcome, summarize_outcomes

TIME = datetime(2024, 1, 15, 10, 45, tzinfo=timezone.utc)
FIELD = "y.h4.close_change"


def record():
    step=timedelta(minutes=15)
    start=TIME-timedelta(hours=8)
    anchor=GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION,"CUSTOM",start.date(),"C1",0,start,
        start+timedelta(hours=3),start+timedelta(hours=3),"LEVEL",0,10,TIME,15,0,"ABOVE_TOUCH_BELOW",None,
        OutcomeAnchor("EVIDENCE",TIME,11))
    mtf=ResearchAnchorMultiTimeframeContext(anchor,MultiTimeframeSnapshot(TIME,OHLCBar(TIME-step,10,12,9,11),()),(),True)
    sessions=ResearchAnchorSessionContext(anchor,(),(),(),(),True)
    context=ResearchAnchorSessionRelationshipContext(sessions,(),(),())
    outcome=ForwardOutcome("EVIDENCE",TIME,11,1,15,TIME,TIME+step,TIME,TIME,11,12,13,10,1,1/11,2,1,TIME,TIME,0,0)
    return assemble_research_record(mtf,context,(outcome,))


def dataset(values):
    source=record()
    return assemble_discovery_dataset(tuple(DiscoveryRow(source,(DiscoveryValue("meta.value",1),),
        (DiscoveryValue("x.value",1),),(DiscoveryValue(FIELD,v),DiscoveryValue("y.h1.end_close",2))) for v in values))


def test_mandatory_example():
    result=summarize_outcome(dataset((-2,-1,0,3,10)),FIELD)
    assert (result.row_count,result.observed_count,result.missing_count)==(5,5,0)
    assert (result.minimum,result.q25,result.median,result.q75,result.maximum,result.mean)==(-2,-1,0,3,10,2)
    assert (result.negative_count,result.zero_count,result.positive_count)==(2,1,2)
    assert (result.negative_fraction,result.zero_fraction,result.positive_fraction)==(.4,.2,.4)
    assert all(type(getattr(result,name)) is float for name in ("minimum","q25","median","q75","maximum","mean"))


def test_interpolation():
    result=summarize_outcome(dataset((0,10)),FIELD)
    assert (result.q25,result.median,result.q75)==(2.5,5.0,7.5)


@pytest.mark.parametrize("value", [-3,0,2.5])
def test_single(value):
    r=summarize_outcome(dataset((value,)),FIELD)
    assert (r.minimum,r.q25,r.median,r.q75,r.maximum,r.mean)==(value,)*6
    assert (r.negative_fraction,r.zero_fraction,r.positive_fraction)==(float(value<0),float(value==0),float(value>0))


def test_all_none():
    r=summarize_outcome(dataset((None,None)),FIELD)
    assert (r.row_count,r.observed_count,r.missing_count)==(2,0,2)
    assert (r.minimum,r.q25,r.median,r.q75,r.maximum,r.mean)==(None,)*6
    assert (r.negative_count,r.zero_count,r.positive_count)==(0,0,0)
    assert (r.negative_fraction,r.zero_fraction,r.positive_fraction)==(None,)*3


def test_mixed_missing():
    r=summarize_outcome(dataset((None,-1,None,0,3)),FIELD)
    assert (r.row_count,r.observed_count,r.missing_count)==(5,3,2)
    assert (r.minimum,r.median,r.maximum)==(-1,0,3)
    assert (r.negative_fraction,r.zero_fraction,r.positive_fraction)==(1/3,)*3


@pytest.mark.parametrize("values,counts", [((-2,-1.5),(2,0,0)),((1,2.5),(0,0,2)),((0,-0.0),(0,2,0))])
def test_sign_only(values,counts):
    r=summarize_outcome(dataset(values),FIELD)
    assert (r.negative_count,r.zero_count,r.positive_count)==counts


@pytest.mark.parametrize("value", [True,"1",TIME.date(),TIME,float("nan"),float("inf"),-float("inf")])
def test_invalid_values(value):
    with pytest.raises(ValueError):
        summarize_outcome(dataset((value,)),FIELD)


@pytest.mark.parametrize("field", ["x.value","meta.value"])
def test_target_barrier(field):
    with pytest.raises(ValueError):
        summarize_outcome(dataset((1,)),field)
    with pytest.raises(ValueError):
        summarize_outcomes(dataset((1,)),(field,))


def test_unknown_empty_and_types():
    for data in (dataset((1,)),assemble_discovery_dataset(())):
        with pytest.raises(KeyError):
            summarize_outcome(data,"y.unknown")
    assert summarize_outcomes(assemble_discovery_dataset(()))==()
    with pytest.raises(TypeError):
        summarize_outcome(object(),FIELD)
    with pytest.raises(TypeError):
        summarize_outcome(dataset((1,)),123)
    with pytest.raises(TypeError):
        summarize_outcomes(object())


def test_order_generator_and_no_mutation():
    data=dataset((1,None,3))
    rows=data.rows
    assert [r.field_id for r in summarize_outcomes(data)]==list(data.target_field_ids)
    class Once:
        calls=0
        def __iter__(self):
            self.calls+=1
            assert self.calls==1
            yield from data.target_field_ids[::-1]
    source=Once()
    assert [r.field_id for r in summarize_outcomes(data,source)]==list(data.target_field_ids[::-1])
    assert source.calls==1
    assert data.rows is rows
    assert data.rows[1].targets[0].value is None
    with pytest.raises(ValueError):
        summarize_outcomes(data,(FIELD,FIELD))
    with pytest.raises(TypeError):
        summarize_outcomes(data,(123,))


@pytest.mark.parametrize("changes", [{"row_count":99},{"observed_count":-1},{"missing_count":1},
    {"negative_count":1},{"zero_count":-1},{"positive_fraction":0.5},{"minimum":None},
    {"mean":float("inf")},{"row_count":True}])
def test_inconsistent_distribution(changes):
    r=summarize_outcome(dataset((1,2)),FIELD)
    with pytest.raises(ValueError):
        replace(r,**changes)


def test_empty_stats_and_immutability():
    r=summarize_outcome(dataset((None,)),FIELD)
    for changes in ({"mean":0.0},{"positive_fraction":0.0}):
        with pytest.raises(ValueError):
            replace(r,**changes)
    with pytest.raises(FrozenInstanceError):
        r.mean=1.0
