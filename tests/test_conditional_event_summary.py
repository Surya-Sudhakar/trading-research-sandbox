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
from sandbox.research import conditional_event_summary as module

TIME=datetime(2024,1,15,10,45,tzinfo=timezone.utc)
H4="x.mtf.ny_h4.direction"
H1="x.mtf.ny_h1.direction"
LONDON="x.session.london.active"
FIELDS=("y.h4.close_change","y.h1.end_close")


def record():
    step=timedelta(minutes=15); start=TIME-timedelta(hours=8)
    a=GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION,"CUSTOM",start.date(),"C1",0,start,
        start+timedelta(hours=3),start+timedelta(hours=3),"LEVEL",0,10,TIME,15,0,"ABOVE_TOUCH_BELOW",None,OutcomeAnchor("EVIDENCE",TIME,11))
    mtf=ResearchAnchorMultiTimeframeContext(a,MultiTimeframeSnapshot(TIME,OHLCBar(TIME-step,10,12,9,11),()),(),True)
    sc=ResearchAnchorSessionContext(a,(),(),(),(),True)
    rel=ResearchAnchorSessionRelationshipContext(sc,(),(),())
    o=ForwardOutcome("EVIDENCE",TIME,11,1,15,TIME,TIME+step,TIME,TIME,11,12,13,10,1,1/11,2,1,TIME,TIME,0,0)
    return assemble_research_record(mtf,rel,(o,))


def dataset(values,events=None):
    source=record(); rows=[]
    events=events or ["ABOVE_TOUCH_BELOW"]*len(values)
    for i,(conditions,event) in enumerate(zip(values,events)):
        kind="UNTOUCHED_CHECKPOINT" if type(event) is int else "TOUCH_TRANSITION"
        predictors=(DiscoveryValue("x.anchor_kind",kind),DiscoveryValue("x.event.pattern_code",None if type(event) is int else event),
                    DiscoveryValue("x.event.checkpoint_bars",event if type(event) is int else None))
        predictors+=tuple(DiscoveryValue(k,v) for k,v in zip((H4,H1,LONDON),conditions))
        rows.append(DiscoveryRow(source,(DiscoveryValue("meta.id",i),),predictors,
                    (DiscoveryValue(FIELDS[0],i),DiscoveryValue(FIELDS[1],10+i))))
    return assemble_discovery_dataset(rows)


def test_single_and_wrapper():
    data=dataset([("BEARISH",None,True),("BULLISH",None,False),("BEARISH",None,True),(None,None,False)])
    result=module.summarize_conditioned_event_cohorts(data,(H4,))
    assert [r.condition_values for r in result]==[(None,),("BEARISH",),("BULLISH",)]
    assert [r.row_indices for r in result]==[(3,),(0,2),(1,)]
    assert module.summarize_single_conditioned_event_cohorts(data,H4)==result
    assert [r.condition_values for r in module.summarize_conditioned_event_cohorts(data,(LONDON,))]==[(False,),(True,)]


def test_two_conditions():
    data=dataset([("BEARISH",None,True),("BULLISH",None,False),("BEARISH",None,False),("BEARISH",None,True),(None,None,False)])
    result=module.summarize_conditioned_event_cohorts(data,(H4,LONDON))
    assert [r.condition_values for r in result]==[(None,False),("BEARISH",False),("BEARISH",True),("BULLISH",False)]
    assert result[2].row_indices==(0,3)


def test_three_conditions_and_event_identity():
    data=dataset([("BEARISH","BEARISH",True),("BEARISH","BULLISH",False),("BEARISH","BEARISH",True),
                  ("BULLISH","BULLISH",False),(None,"BEARISH",False)],
                 ["ABOVE_TOUCH_BELOW"]*3+["ABOVE_TOUCH_ABOVE","ABOVE_TOUCH_BELOW"])
    result=module.summarize_conditioned_event_cohorts(data,(H4,H1,LONDON))
    assert result[0].event_cohort_id=="TOUCH_TRANSITION__ABOVE_TOUCH_ABOVE"
    assert [r.condition_values for r in result[1:]]==[(None,"BEARISH",False),("BEARISH","BEARISH",True),("BEARISH","BULLISH",False)]
    assert [r.row_indices for r in result[1:]]==[(4,),(0,2),(1,)]
    assert all(r.anchor_kind=="TOUCH_TRANSITION" and r.checkpoint_bars is None for r in result)


def test_typed_categories_multi_and_order():
    data=dataset([(v,"S",False) for v in (None,False,0,"0",True,1,-1)])
    expected=[None,False,True,-1,0,1,"0"]
    for fields in ((H4,),(H4,H1)):
        result=module.summarize_conditioned_event_cohorts(data,fields)
        assert [(type(r.condition_values[0]),r.condition_values[0]) for r in result]==[(type(v),v) for v in expected]
    data=dataset([("BEARISH","BULLISH",False)])
    a=module.summarize_conditioned_event_cohorts(data,(H4,H1))[0]
    b=module.summarize_conditioned_event_cohorts(data,(H1,H4))[0]
    assert a.condition_values==b.condition_values[::-1]
    assert b.condition_field_ids==(H1,H4)


def test_mixed_events_duplicates_and_no_cartesian():
    data=dataset([("A","A",False),("B","B",False),("A","A",False)],[8,8,"PATTERN"])
    data=assemble_discovery_dataset(data.rows+(data.rows[0],)*2)
    result=module.summarize_conditioned_event_cohorts(data,(H4,H1))
    assert len(result)==3
    assert result[1].checkpoint_bars==8 and result[1].pattern_code is None
    assert result[1].row_indices==(0,3,4)


@pytest.mark.parametrize("fields", [(),(H4,H4),(123,),("meta.id",),(FIELDS[0],),("x.anchor_kind",),("x.event.pattern_code",),("x.event.checkpoint_bars",)])
def test_invalid_conditions(fields):
    with pytest.raises((ValueError,TypeError)):
        module.summarize_conditioned_event_cohorts(dataset([("A","B",True)]),fields)


def test_unknown_and_invalid_dataset():
    with pytest.raises(KeyError):
        module.summarize_conditioned_event_cohorts(dataset([("A","B",True)]),("x.unknown",))
    with pytest.raises(TypeError):
        module.summarize_conditioned_event_cohorts(object(),(H4,))


@pytest.mark.parametrize("value", [1.5,TIME.date(),TIME])
def test_invalid_column_values(value):
    with pytest.raises(ValueError):
        module.summarize_conditioned_event_cohorts(dataset([(value,None,False)]),(H4,))


def test_delegation_generators_and_original_rows(monkeypatch):
    data=dataset([("A","B",True),("A","B",True),(None,"B",False)])
    events,reads,summaries=[],[],[]
    original_event=module.summarize_event_cohorts; original_column=module.column_values; original_summary=module.summarize_outcomes
    def event(*args,**kwargs):
        events.append(kwargs)
        return original_event(*args,**kwargs)
    def column(data,field):
        reads.append(field)
        return original_column(data,field)
    def summary(subset,fields):
        summaries.append(subset)
        return original_summary(subset,fields)
    monkeypatch.setattr(module,"summarize_event_cohorts",event)
    monkeypatch.setattr(module,"column_values",column)
    monkeypatch.setattr(module,"summarize_outcomes",summary)
    class Once:
        def __init__(self,values): self.values,self.calls=values,0
        def __iter__(self):
            self.calls+=1
            assert self.calls==1
            yield from self.values
    conditions,targets=Once((H4,H1)),Once(FIELDS[::-1])
    result=module.summarize_conditioned_event_cohorts(data,conditions,targets)
    assert conditions.calls==targets.calls==1
    assert events==[{"field_ids":()}]
    assert reads==[H4,H1]
    assert len(summaries)==len(result)==2
    assert summaries[0].rows[0] is data.rows[2]
    assert summaries[1].rows[0] is data.rows[0]
    assert summaries[1].rows[1] is data.rows[1]
    assert all(tuple(d.field_id for d in r.outcome_distributions)==FIELDS[::-1] for r in result)


def test_targets_and_empty(monkeypatch):
    data=dataset([("A","B",True)])
    result=module.summarize_conditioned_event_cohorts(data,(H4,))[0]
    assert tuple(d.field_id for d in result.outcome_distributions)==FIELDS
    assert len(module.summarize_conditioned_event_cohorts(data,(H4,),(FIELDS[0],))[0].outcome_distributions)==1
    for fields in ((FIELDS[0],)*2,(123,),(H4,)):
        with pytest.raises((ValueError,TypeError)):
            module.summarize_conditioned_event_cohorts(data,(H4,),fields)
    def forbidden(*args,**kwargs): raise AssertionError("empty must not delegate")
    for name in ("summarize_event_cohorts","column_values","summarize_outcomes"):
        monkeypatch.setattr(module,name,forbidden)
    assert module.summarize_conditioned_event_cohorts(assemble_discovery_dataset(()),())==()


def test_leakage_unrelated_and_reordering():
    data=dataset([("B","B",True),("A","A",False),("B","B",False)])
    original=module.summarize_conditioned_event_cohorts(data,(H4,H1))
    changed=assemble_discovery_dataset(tuple(replace(r,targets=tuple(replace(v,value=999) for v in r.targets)) for r in data.rows))
    result=module.summarize_conditioned_event_cohorts(changed,(H4,H1))
    for a,b in zip(original,result):
        assert replace(a,outcome_distributions=b.outcome_distributions)==b
    assert original[0].outcome_distributions!=result[0].outcome_distributions
    unrelated=assemble_discovery_dataset(tuple(replace(r,predictors=tuple(replace(v,value=not v.value) if v.field_id==LONDON else v for v in r.predictors)) for r in data.rows))
    assert module.summarize_conditioned_event_cohorts(unrelated,(H4,H1))==original
    reordered=module.summarize_conditioned_event_cohorts(assemble_discovery_dataset(data.rows[::-1]),(H4,H1))
    for a,b in zip(original,reordered):
        assert (a.event_cohort_id,a.condition_values,a.row_count,a.outcome_distributions)==(b.event_cohort_id,b.condition_values,b.row_count,b.outcome_distributions)


@pytest.mark.parametrize("changes", [{"event_cohort_id":""},{"anchor_kind":""},{"condition_field_ids":[]},
    {"condition_field_ids":()},{"condition_field_ids":("x.anchor_kind",)},{"condition_field_ids":(H4,H4)},
    {"condition_values":()},{"condition_values":(1.5,)},{"condition_values":(object(),)},
    {"row_indices":(True,)},{"row_indices":(-1,)},{"row_indices":(1,0)},{"row_count":True},
    {"row_count":0},{"outcome_distributions":[]},{"outcome_distributions":(object(),)}])
def test_dataclass_validation(changes):
    result=module.summarize_conditioned_event_cohorts(dataset([("A","B",True)]),(H4,))[0]
    with pytest.raises(ValueError): replace(result,**changes)


def test_distribution_validation_and_immutable():
    result=module.summarize_conditioned_event_cohorts(dataset([("A","B",True)]),(H4,))[0]
    other=module.summarize_conditioned_event_cohorts(dataset([("A","B",True)]*2),(H4,))[0]
    with pytest.raises(ValueError): replace(result,outcome_distributions=other.outcome_distributions)
    with pytest.raises(ValueError): replace(result,outcome_distributions=(result.outcome_distributions[0],)*2)
    with pytest.raises(FrozenInstanceError): result.row_count=99
