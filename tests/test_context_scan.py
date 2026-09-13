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
from sandbox.research import context_scan as module

H4='x.mtf.ny_h4.direction'
H1='x.mtf.ny_h1.direction'
NUMBER='x.context.ratio'
TARGETS=('y.h4.close_change','y.h1.end_close')


def dataset(values, events=None):
    now=datetime(2024,1,15,10,45,tzinfo=timezone.utc)
    step=timedelta(minutes=15); start=now-timedelta(hours=8)
    anchor=GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION,'CUSTOM',start.date(),'C1',0,start,
        start+timedelta(hours=3),start+timedelta(hours=3),'LEVEL',0,10,now,15,0,'ABOVE_TOUCH_BELOW',None,OutcomeAnchor('EVIDENCE',now,11))
    mtf=ResearchAnchorMultiTimeframeContext(anchor,MultiTimeframeSnapshot(now,OHLCBar(now-step,10,12,9,11),()),(),True)
    sessions=ResearchAnchorSessionContext(anchor,(),(),(),(),True)
    relationships=ResearchAnchorSessionRelationshipContext(sessions,(),(),())
    outcome=ForwardOutcome('EVIDENCE',now,11,1,15,now,now+step,now,now,11,12,13,10,1,1/11,2,1,now,now,0,0)
    record=assemble_research_record(mtf,relationships,(outcome,))
    rows=[]
    for i,(values,event) in enumerate(zip(values,events or ['ABOVE_TOUCH_BELOW']*len(values))):
        checkpoint=type(event) is int
        predictors=(DiscoveryValue('x.anchor_kind','UNTOUCHED_CHECKPOINT' if checkpoint else 'TOUCH_TRANSITION'),
            DiscoveryValue('x.event.pattern_code',None if checkpoint else event),
            DiscoveryValue('x.event.checkpoint_bars',event if checkpoint else None))
        predictors+=tuple(DiscoveryValue(k,v) for k,v in zip((H4,H1,NUMBER),values))
        rows.append(DiscoveryRow(record,(DiscoveryValue('meta.id',i),),predictors,
            (DiscoveryValue(TARGETS[0],i),DiscoveryValue(TARGETS[1],10-i))))
    return assemble_discovery_dataset(rows)


def categorical():
    return dataset([(v,'A',i) for i,v in enumerate(('BEARISH','BEARISH','BULLISH','BULLISH',None))])


def numeric():
    return dataset([('A','B',v) for v in (None,.1,.25,.49,.5,.74,.75,1.)])


def cat_scan(data=None):
    return module.scan_single_contexts(categorical() if data is None else data,(H4,),field_ids=(TARGETS[0],))


def num_scan():
    return module.scan_single_contexts(numeric(),numeric_condition_specs=((NUMBER,(.25,.5,.75)),),field_ids=(TARGETS[0],))


def test_categorical_complements():
    results=cat_scan()
    assert [r.categorical_value for r in results]==[None,'BEARISH','BULLISH']
    assert [r.subgroup_row_indices for r in results]==[(4,),(0,1),(2,3)]
    assert [r.complement_row_indices for r in results]==[(0,1,2,3),(2,3,4),(0,1,4)]
    for r in results:
        assert not set(r.subgroup_row_indices)&set(r.complement_row_indices)
        assert set(r.subgroup_row_indices+r.complement_row_indices)==set(range(5))
        assert r.context_kind=='CATEGORICAL' and r.numeric_cutpoints is None
        assert r.numeric_missing is r.numeric_bin_index is r.numeric_lower_bound is r.numeric_upper_bound is None
        assert r.numeric_lower_inclusive is r.numeric_upper_inclusive is None
        assert r.comparison.left_row_indices==r.subgroup_row_indices
        assert r.comparison.right_row_indices==r.complement_row_indices


def test_numeric_complements_and_identity():
    result=num_scan()
    assert [r.subgroup_row_indices for r in result]==[(0,),(1,),(2,3),(4,5),(6,7)]
    assert [r.numeric_bin_index for r in result]==[None,0,1,2,3]
    assert [r.numeric_missing for r in result]==[True,False,False,False,False]
    assert [(r.numeric_lower_bound,r.numeric_upper_bound,r.numeric_lower_inclusive,r.numeric_upper_inclusive) for r in result]==[
        (None,None,None,None),(None,.25,None,False),(.25,.5,True,False),(.5,.75,True,False),(.75,None,True,None)]
    for r in result:
        assert r.numeric_cutpoints==(.25,.5,.75) and r.categorical_value is None
        assert r.complement_row_indices==tuple(i for i in range(8) if i not in r.subgroup_row_indices)
        assert r.context_kind=='NUMERIC'


def test_multiple_events_and_single_subgroup_skip():
    events=['ABOVE_TOUCH_BELOW','ABOVE_TOUCH_ABOVE',8]*2+['ONLY']
    data=dataset([('A','B',0)]*3+[('B','B',1)]*3+[('A','B',0)],events)
    result=cat_scan(data)
    assert len(result)==6
    assert [r.event_cohort_id for r in result]==sorted(r.event_cohort_id for r in result)
    for r in result:
        indices=r.subgroup_row_indices+r.complement_row_indices
        assert len(indices)==2
        assert len({events[i] for i in indices})==1
        assert r.pattern_code is None if r.checkpoint_bars else r.pattern_code in events
    assert module.scan_single_contexts(data,(H1,))==()
    assert module.scan_single_contexts(data)==()
    assert module.scan_single_contexts(data,(H4,),field_ids=())==()


def test_order_specs_and_typed_values():
    data=dataset([(v,str(i%2),i) for i,v in enumerate((None,False,0,'0',True,1))])
    result=module.scan_single_contexts(data,(H1,H4),((NUMBER,(2,)),(NUMBER,(3,))),TARGETS[::-1])
    starts=[(r.context_kind,r.condition_field_id,r.numeric_cutpoints) for r in result]
    assert starts==[('CATEGORICAL',H1,None)]*4+[('CATEGORICAL',H4,None)]*12+[
        ('NUMERIC',NUMBER,(2.,))]*4+[('NUMERIC',NUMBER,(3.,))]*4
    assert [r.target_field_id for r in result]==list(TARGETS[::-1])*12
    values=[r.categorical_value for r in result[4:16:2]]
    assert [(type(v),v) for v in values]==[(type(v),v) for v in (None,False,True,0,1,'0')]
    assert module.scan_single_contexts(data,(H1,),field_ids=None)[0].target_field_id==TARGETS[0]


class Once:
    def __init__(self,values): self.values,self.calls=values,0
    def __iter__(self):
        self.calls+=1
        assert self.calls==1
        yield from self.values


def test_delegation_once_and_authoritative_subgroup_order(monkeypatch):
    data=dataset([('A','B',0),('B','B',1),('A','B',2)])
    cats,nums,comparisons=[],[],[]
    cat=module.summarize_single_conditioned_event_cohorts
    num=module.summarize_numeric_conditioned_event_cohorts
    compare=module.compare_cohort_rows
    def categorical(data,field,**kwargs):
        cats.append((field,kwargs))
        return cat(data,field,**kwargs)[::-1]
    def numerical(data,field,cuts,**kwargs):
        nums.append((field,cuts,kwargs))
        return num(data,field,cuts,**kwargs)
    def comparison(data,left,right,field):
        comparisons.append((left,right,field))
        return compare(data,left,right,field)
    monkeypatch.setattr(module,'summarize_single_conditioned_event_cohorts',categorical)
    monkeypatch.setattr(module,'summarize_numeric_conditioned_event_cohorts',numerical)
    monkeypatch.setattr(module,'compare_cohort_rows',comparison)
    cuts=Once((1,)); fields=Once(TARGETS[::-1]); categories=Once((H4,)); specs=Once(((NUMBER,cuts),))
    results=module.scan_single_contexts(data,categories,specs,fields)
    assert cuts.calls==fields.calls==categories.calls==specs.calls==1
    assert cats==[(H4,{'field_ids':()})]
    assert nums==[(NUMBER,(1,),{'field_ids':()})]
    assert len(comparisons)==len(results)==8
    assert results[0].subgroup_row_indices==(1,)
    assert results[0].complement_row_indices==(0,2)
    assert comparisons==[(r.subgroup_row_indices,r.complement_row_indices,r.target_field_id) for r in results]


def test_empty_no_consumption(monkeypatch):
    def fail(*args,**kwargs): raise AssertionError('unexpected call')
    class Never:
        __iter__=fail
    for name in ('summarize_single_conditioned_event_cohorts','summarize_numeric_conditioned_event_cohorts','compare_cohort_rows'):
        monkeypatch.setattr(module,name,fail)
    assert module.scan_single_contexts(assemble_discovery_dataset(()),Never(),Never(),Never())==()
    with pytest.raises(TypeError): module.scan_single_contexts(object())


@pytest.mark.parametrize('kwargs,error', [
    ({'categorical_field_ids':(H4,H4)},ValueError),({'categorical_field_ids':(1,)},TypeError),
    ({'categorical_field_ids':('meta.id',)},ValueError),({'categorical_field_ids':(TARGETS[0],)},ValueError),
    ({'categorical_field_ids':('x.anchor_kind',)},ValueError),({'categorical_field_ids':('x.unknown',)},KeyError),
    ({'numeric_condition_specs':((NUMBER,(1,0)),)},ValueError),
    ({'numeric_condition_specs':((NUMBER,(True,)),)},ValueError),
    ({'numeric_condition_specs':((NUMBER,()),)},ValueError),
    ({'field_ids':(TARGETS[0],)*2},ValueError),({'field_ids':(1,)},TypeError),
    ({'field_ids':(H4,)},ValueError),({'field_ids':('y.unknown',)},KeyError)])
def test_input_validation(kwargs,error):
    with pytest.raises(error): module.scan_single_contexts(categorical(),**kwargs)


def test_target_mutation_and_unrelated_stability():
    data=categorical(); original=cat_scan(data)
    changed=assemble_discovery_dataset(tuple(replace(r,targets=(replace(r.targets[0],value=100-i*10),r.targets[1])) for i,r in enumerate(data.rows)))
    result=cat_scan(changed)
    assert [(r.subgroup_row_indices,r.complement_row_indices) for r in result]==[(r.subgroup_row_indices,r.complement_row_indices) for r in original]
    assert all(a.comparison!=b.comparison for a,b in zip(result,original))
    unrelated=assemble_discovery_dataset(tuple(replace(r,identity=(replace(r.identity[0],value='OTHER'),),
        targets=(r.targets[0],replace(r.targets[1],value=-999)),
        predictors=tuple(replace(v,value='CHANGED') if v.field_id==H1 else v for v in r.predictors)) for r in data.rows))
    assert cat_scan(unrelated)==original
    assert cat_scan(data)==original


@pytest.mark.parametrize('changes', [
    {'context_kind':'OTHER'},{'context_kind':1},{'condition_field_id':''},{'event_cohort_id':' '},
    {'anchor_kind':None},{'target_field_id':''},{'subgroup_row_indices':[]},{'subgroup_row_indices':()},
    {'subgroup_row_indices':(True,)},{'subgroup_row_indices':(-1,)},{'subgroup_row_indices':(4,4)},
    {'complement_row_indices':[0,1,2,3]},{'complement_row_indices':()},
    {'complement_row_indices':(3,2,1,0)},{'complement_row_indices':(0,1,2,4)},
    {'comparison':object()},{'target_field_id':TARGETS[1]},{'subgroup_row_indices':(5,)},
    {'complement_row_indices':(0,1,2)},{'categorical_value':.5},
    {'numeric_cutpoints':(.5,)},{'numeric_missing':False},{'numeric_bin_index':0},
    {'numeric_lower_bound':0.},{'numeric_upper_bound':1.},{'numeric_lower_inclusive':True},
    {'numeric_upper_inclusive':False}])
def test_categorical_dataclass_validation(changes):
    with pytest.raises(ValueError): replace(cat_scan()[0],**changes)


@pytest.mark.parametrize('changes', [
    {'numeric_cutpoints':None},{'numeric_cutpoints':[]},{'numeric_cutpoints':()},
    {'numeric_cutpoints':(1,)},{'numeric_cutpoints':(float('inf'),)},
    {'numeric_cutpoints':(.5,.25)},{'numeric_cutpoints':(.25,.25)},
    {'numeric_cutpoints':(.1,.5,.75)},{'numeric_missing':None},{'numeric_missing':1},
    {'categorical_value':False},{'numeric_bin_index':True},{'numeric_bin_index':-1},
    {'numeric_bin_index':10},{'numeric_lower_bound':None},{'numeric_upper_bound':.1},
    {'numeric_lower_inclusive':False},{'numeric_upper_inclusive':True}])
def test_numeric_dataclass_validation(changes):
    with pytest.raises(ValueError): replace(num_scan()[2],**changes)


def test_missing_numeric_and_frozen():
    missing=num_scan()[0]
    with pytest.raises(ValueError): replace(missing,numeric_bin_index=0)
    with pytest.raises(ValueError): replace(missing,numeric_lower_bound=.25)
    with pytest.raises(FrozenInstanceError): missing.numeric_missing=False
    result=cat_scan()[0]
    with pytest.raises(FrozenInstanceError): result.context_kind='NUMERIC'


def test_duplicate_rows_and_skips_do_not_compare(monkeypatch):
    data=categorical()
    data=assemble_discovery_dataset(data.rows+(data.rows[0],))
    result=cat_scan(data)
    assert result[1].subgroup_row_indices==(0,1,5)
    def fail(*args,**kwargs): raise AssertionError('empty complement')
    monkeypatch.setattr(module,'compare_cohort_rows',fail)
    assert module.scan_single_contexts(data,(H1,))==()
    assert module.scan_single_contexts(data,numeric_condition_specs=((NUMBER,(100,)),))==()
