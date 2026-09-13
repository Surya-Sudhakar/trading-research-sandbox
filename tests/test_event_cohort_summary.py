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
from sandbox.research import event_cohort_summary as module

TIME=datetime(2024,1,15,10,45,tzinfo=timezone.utc)
FIELDS=("y.h4.close_change","y.h1.end_close")


def record():
    step=timedelta(minutes=15)
    start=TIME-timedelta(hours=8)
    a=GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION,"CUSTOM",start.date(),"C1",0,start,
        start+timedelta(hours=3),start+timedelta(hours=3),"LEVEL",0,10,TIME,15,0,"ABOVE_TOUCH_BELOW",None,OutcomeAnchor("EVIDENCE",TIME,11))
    mtf=ResearchAnchorMultiTimeframeContext(a,MultiTimeframeSnapshot(TIME,OHLCBar(TIME-step,10,12,9,11),()),(),True)
    sc=ResearchAnchorSessionContext(a,(),(),(),(),True)
    rel=ResearchAnchorSessionRelationshipContext(sc,(),(),())
    o=ForwardOutcome("EVIDENCE",TIME,11,1,15,TIME,TIME+step,TIME,TIME,11,12,13,10,1,1/11,2,1,TIME,TIME,0,0)
    return assemble_research_record(mtf,rel,(o,))


def dataset(events):
    source=record()
    return assemble_discovery_dataset(tuple(DiscoveryRow(source,(),tuple(DiscoveryValue(k,v) for k,v in zip(
        ("x.anchor_kind","x.event.pattern_code","x.event.checkpoint_bars"),event)),
        (DiscoveryValue(FIELDS[0],i),DiscoveryValue(FIELDS[1],10+i))) for i,event in enumerate(events)))


def touch(code):
    return ("TOUCH_TRANSITION",code,None)


def untouched(n):
    return ("UNTOUCHED_CHECKPOINT",None,n)


def test_mandatory_touch_example():
    events=[touch(s) for s in ("ABOVE_TOUCH_BELOW","ABOVE_TOUCH_ABOVE","ABOVE_TOUCH_BELOW","BELOW_TOUCH_ABOVE","ABOVE_TOUCH_ABOVE")]
    results=module.summarize_event_cohorts(dataset(events))
    assert [r.cohort_id for r in results]==["TOUCH_TRANSITION__"+s for s in ("ABOVE_TOUCH_ABOVE","ABOVE_TOUCH_BELOW","BELOW_TOUCH_ABOVE")]
    assert [r.row_indices for r in results]==[(1,4),(0,2),(3,)]
    assert [r.row_count for r in results]==[2,2,1]
    assert all(r.anchor_kind=="TOUCH_TRANSITION" and r.checkpoint_bars is None for r in results)
    for r in results:
        assert [d.field_id for d in r.outcome_distributions]==list(FIELDS)
        assert all(d.row_count==r.row_count for d in r.outcome_distributions)


def test_mandatory_checkpoint_example():
    results=module.summarize_event_cohorts(dataset([untouched(n) for n in (8,4,8,32)]))
    assert [r.cohort_id for r in results]==["UNTOUCHED_CHECKPOINT__32","UNTOUCHED_CHECKPOINT__4","UNTOUCHED_CHECKPOINT__8"]
    assert [r.row_indices for r in results]==[(3,),(1,),(0,2)]
    assert [r.checkpoint_bars for r in results]==[32,4,8]
    assert all(r.pattern_code is None for r in results)


def test_mixed_duplicates_and_target_membership_barrier():
    data=dataset([touch("A"),untouched(8),touch("A")])
    data=assemble_discovery_dataset(data.rows+(data.rows[0],))
    original=module.summarize_event_cohorts(data)
    changed=assemble_discovery_dataset(tuple(replace(r,targets=tuple(replace(v,value=999) for v in r.targets)) for r in data.rows))
    revised=module.summarize_event_cohorts(changed)
    for a,b in zip(original,revised):
        assert (a.cohort_id,a.anchor_kind,a.pattern_code,a.checkpoint_bars,a.row_indices,a.row_count)==(b.cohort_id,b.anchor_kind,b.pattern_code,b.checkpoint_bars,b.row_indices,b.row_count)
    assert original[0].row_indices==(0,2,3)
    assert original[0].outcome_distributions!=revised[0].outcome_distributions


def test_reordering():
    data=dataset([touch("B"),touch("A"),touch("B"),untouched(4)])
    reordered=assemble_discovery_dataset(tuple(data.rows[i] for i in (3,1,0,2)))
    first,second=module.summarize_event_cohorts(data),module.summarize_event_cohorts(reordered)
    assert [r.cohort_id for r in first]==[r.cohort_id for r in second]
    for a,b in zip(first,second):
        assert a.row_count==b.row_count
        assert a.outcome_distributions==b.outcome_distributions


@pytest.mark.parametrize("event", [("TOUCH_TRANSITION",None,None),("TOUCH_TRANSITION","",None),("TOUCH_TRANSITION","A",4),
    ("UNTOUCHED_CHECKPOINT","A",4),untouched(None),untouched(0),untouched(-1),untouched(True),untouched(4.0),("UNKNOWN",None,None)])
def test_invalid_events(event):
    with pytest.raises(ValueError):
        module.summarize_event_cohorts(dataset([event]))


def test_delegation_once_original_rows_column_reads_and_generator(monkeypatch):
    data=dataset([touch("A"),untouched(4),touch("A")])
    original_summary=module.summarize_outcomes
    original_columns=module.column_values
    calls,reads=[],[]
    def tracked(subset,fields):
        calls.append(subset)
        return original_summary(subset,fields)
    def columns(data,field):
        reads.append(field)
        return original_columns(data,field)
    monkeypatch.setattr(module,"summarize_outcomes",tracked)
    monkeypatch.setattr(module,"column_values",columns)
    class Once:
        calls=0
        def __iter__(self):
            self.calls+=1
            assert self.calls==1
            yield from FIELDS[::-1]
    fields=Once()
    result=module.summarize_event_cohorts(data,fields)
    assert fields.calls==1
    assert len(calls)==2
    assert reads==["x.anchor_kind","x.event.pattern_code","x.event.checkpoint_bars"]
    assert calls[0].rows[0] is data.rows[0]
    assert calls[0].rows[1] is data.rows[2]
    assert calls[1].rows[0] is data.rows[1]
    assert all(tuple(d.field_id for d in r.outcome_distributions)==FIELDS[::-1] for r in result)
    assert len(module.summarize_event_cohorts(data,(FIELDS[0],))[0].outcome_distributions)==1


def test_empty_no_summary(monkeypatch):
    def forbidden(*args):
        raise AssertionError("must not summarize")
    monkeypatch.setattr(module,"summarize_outcomes",forbidden)
    assert module.summarize_event_cohorts(assemble_discovery_dataset(()))==()


def test_fields_and_input_validation(monkeypatch):
    data=dataset([touch("A")])
    with pytest.raises(ValueError):
        module.summarize_event_cohorts(data,(FIELDS[0],)*2)
    with pytest.raises(TypeError):
        module.summarize_event_cohorts(data,(123,))
    with pytest.raises(ValueError):
        module.summarize_event_cohorts(data,("x.anchor_kind",))
    with pytest.raises(KeyError):
        module.summarize_event_cohorts(data,("y.unknown",))
    with pytest.raises(TypeError):
        module.summarize_event_cohorts(object())
    missing=assemble_discovery_dataset((replace(data.rows[0],predictors=data.rows[0].predictors[1:]),))
    with pytest.raises(KeyError):
        module.summarize_event_cohorts(missing)
    monkeypatch.setattr(module,"field_role",lambda *args:"identity")
    with pytest.raises(ValueError):
        module.summarize_event_cohorts(data)


@pytest.mark.parametrize("changes", [{"cohort_id":"OTHER"},{"row_indices":(True,)},{"row_indices":(-1,)},
    {"row_indices":[0]},{"row_indices":(1,0)},{"row_indices":(0,0)},{"row_count":0},{"row_count":2},
    {"outcome_distributions":(object(),)}])
def test_summary_consistency(changes):
    result,=module.summarize_event_cohorts(dataset([touch("A")]))
    with pytest.raises(ValueError):
        replace(result,**changes)


def test_distribution_counts_duplicates_and_immutable():
    single,=module.summarize_event_cohorts(dataset([touch("A")]))
    double,=module.summarize_event_cohorts(dataset([touch("A"),touch("A")]))
    with pytest.raises(ValueError):
        replace(single,outcome_distributions=double.outcome_distributions)
    with pytest.raises(ValueError):
        replace(single,outcome_distributions=(single.outcome_distributions[0],)*2)
    with pytest.raises(FrozenInstanceError):
        single.row_count=99
