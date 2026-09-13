from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from types import SimpleNamespace
import pytest
from test_conditional_event_summary import dataset as categorical_dataset, H4, H1, FIELDS
from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
from sandbox.research import numeric_condition_summary as module


def dataset(values, events=None):
    return categorical_dataset([(v, "UNRELATED", False) for v in values], events)


def run(data, cuts=(.25, .5, .75), fields=None):
    return module.summarize_numeric_conditioned_event_cohorts(data, H4, cuts, fields)


class Once:
    def __init__(self, values): self.values, self.calls = values, 0
    def __iter__(self):
        self.calls += 1
        assert self.calls == 1
        yield from self.values


def test_boundaries_metadata_and_statistics():
    results = run(dataset([None, .1, .25, .49, .5, .74, .75, 1]))
    assert [r.row_indices for r in results] == [(0,), (1,), (2,3), (4,5), (6,7)]
    assert [(r.missing, r.bin_index, r.lower_bound, r.upper_bound, r.lower_inclusive, r.upper_inclusive) for r in results] == [
        (True,None,None,None,None,None), (False,0,None,.25,None,False),
        (False,1,.25,.5,True,False), (False,2,.5,.75,True,False), (False,3,.75,None,True,None)]
    assert [r.row_count for r in results] == [1,1,2,2,2]
    assert [r.outcome_distributions[0].mean for r in results] == [0,1,2.5,4.5,6.5]
    assert all(tuple(d.field_id for d in r.outcome_distributions) == FIELDS for r in results)


def test_negative_sparse_and_duplicate_rows():
    data = dataset([-2,-1,0,1,2], [8,8,8,8,8])
    data = assemble_discovery_dataset(data.rows + (data.rows[0],))
    result = run(data, (-1,0,1))
    assert [r.row_indices for r in result] == [(0,5),(1,),(2,),(3,4)]
    assert all(r.checkpoint_bars == 8 and r.pattern_code is None for r in result)
    assert len(run(dataset([5,6]), (0,1,2))) == 1
    assert run(dataset([None]), (0,))[0].missing


@pytest.mark.parametrize("cuts", [(),(.5,.25),(.25,.25),(True,),('1',),(None,),
    (float('nan'),),(float('inf'),),(-float('inf'),),(date.today(),),(object(),),(10**400,)])
def test_invalid_cuts(cuts):
    with pytest.raises(ValueError): run(dataset([1]), cuts)


@pytest.mark.parametrize("value", [True,False,'1',date.today(),datetime.now(),object(),float('nan'),float('inf'),-float('inf'),10**400])
def test_invalid_values(value, monkeypatch):
    monkeypatch.setattr(module, 'column_values', lambda *args: (value,))
    with pytest.raises(ValueError): run(dataset([1]))


@pytest.mark.parametrize("field, error", [(123,TypeError),('meta.id',ValueError),(FIELDS[0],ValueError),
    ('x.anchor_kind',ValueError),('x.event.pattern_code',ValueError),('x.event.checkpoint_bars',ValueError),('x.unknown',KeyError)])
def test_invalid_field(field, error):
    with pytest.raises(error): module.summarize_numeric_conditioned_event_cohorts(dataset([1]),field,(0,))


def test_empty_short_circuit_and_invalid_dataset(monkeypatch):
    def fail(*args, **kwargs): raise AssertionError('unexpected consumption')
    class Never:
        __iter__ = fail
    for name in ('column_values','summarize_event_cohorts','summarize_outcomes','field_role'):
        monkeypatch.setattr(module,name,fail)
    assert module.summarize_numeric_conditioned_event_cohorts(assemble_discovery_dataset(()),None,Never(),Never()) == ()
    with pytest.raises(TypeError): run(object())


def test_delegation_and_authoritative_indices(monkeypatch):
    data = dataset([.1,.6,.2,None])
    reads, events, subsets = [], [], []
    column, summary = module.column_values, module.summarize_outcomes
    def read(data, field):
        reads.append(field)
        return column(data,field)
    def event(data, **kwargs):
        events.append(kwargs)
        return (SimpleNamespace(cohort_id='Z',anchor_kind='CUSTOM',pattern_code='P',checkpoint_bars=None,row_indices=(0,2)),
                SimpleNamespace(cohort_id='A',anchor_kind='CUSTOM',pattern_code=None,checkpoint_bars=7,row_indices=(3,)))
    def summarize(subset, fields):
        subsets.append(subset)
        return summary(subset,fields)
    monkeypatch.setattr(module,'column_values',read)
    monkeypatch.setattr(module,'summarize_event_cohorts',event)
    monkeypatch.setattr(module,'summarize_outcomes',summarize)
    cuts, fields = Once((.5,)), Once(FIELDS[::-1])
    result = run(data,cuts,fields)
    assert cuts.calls == fields.calls == 1
    assert reads == [H4] and events == [{'field_ids':()}]
    assert len(subsets) == len(result) == 2
    assert [r.event_cohort_id for r in result] == ['A','Z']
    assert [r.row_indices for r in result] == [(3,),(0,2)]
    assert subsets[0].rows[0] is data.rows[3]
    assert all(a is data.rows[i] for a,i in zip(subsets[1].rows,(0,2)))
    assert tuple(d.field_id for d in result[0].outcome_distributions) == FIELDS[::-1]


@pytest.mark.parametrize('fields,error', [((FIELDS[0],)*2,ValueError),((1,),TypeError),((H4,),ValueError),(('y.unknown',),KeyError)])
def test_target_contract(fields,error):
    with pytest.raises(error): run(dataset([1]), fields=fields)


def test_leakage_stability_and_cut_changes():
    data = dataset([.1,.6,.2,.8])
    original = run(data,(.5,))
    changed = assemble_discovery_dataset(tuple(replace(r,targets=tuple(replace(v,value=999) for v in r.targets)) for r in data.rows))
    result = run(changed,(.5,))
    assert [r.row_indices for r in original] == [r.row_indices for r in result]
    assert original[0].outcome_distributions != result[0].outcome_distributions
    unrelated = assemble_discovery_dataset(tuple(replace(r,predictors=tuple(replace(v,value='NEW') if v.field_id==H1 else v for v in r.predictors)) for r in data.rows))
    assert run(unrelated,(.5,)) == original
    assert [r.row_indices for r in run(data,(.15,))] != [r.row_indices for r in original]
    reordered = run(assemble_discovery_dataset(data.rows[::-1]),(.5,))
    assert [replace(r,row_indices=q.row_indices) for r,q in zip(original,reordered)] == list(reordered)
    assert run(data,(.5,)) == original


@pytest.mark.parametrize('changes', [
    {'event_cohort_id':''},{'anchor_kind':1},{'condition_field_id':' '},{'missing':1},
    {'row_indices':[]},{'row_indices':(True,)},{'row_indices':(-1,)},{'row_indices':(1,0)},
    {'row_indices':(0,0)},{'row_count':True},{'row_count':0},{'row_count':2},
    {'outcome_distributions':[]},{'outcome_distributions':(object(),)},
    {'bin_index':True},{'bin_index':-1},{'bin_index':None},{'bin_index':1},
    {'lower_bound':0.0},{'upper_bound':None},{'upper_bound':1},{'upper_bound':float('inf')},
    {'lower_inclusive':False},{'upper_inclusive':True},{'upper_inclusive':0},{'missing':True}])
def test_dataclass_invalid(changes):
    result = run(dataset([.1]))[0]
    with pytest.raises(ValueError): replace(result,**changes)


def test_other_intervals_distributions_and_frozen():
    missing, first, middle, last = run(dataset([None,-1,.5,2]),(0,1))
    for changes in ({'bin_index':0},{'upper_bound':1.0},{'lower_inclusive':True}):
        with pytest.raises(ValueError): replace(missing,**changes)
    for changes in ({'lower_bound':2.0},{'lower_inclusive':False},{'bin_index':0}):
        with pytest.raises(ValueError): replace(middle,**changes)
    for changes in ({'upper_inclusive':False},{'bin_index':0}):
        with pytest.raises(ValueError): replace(last,**changes)
    with pytest.raises(ValueError): replace(first,outcome_distributions=(first.outcome_distributions[0],)*2)
    other = run(dataset([.1,.1]))[0]
    with pytest.raises(ValueError): replace(first,outcome_distributions=other.outcome_distributions)
    with pytest.raises(FrozenInstanceError): first.row_count=8
