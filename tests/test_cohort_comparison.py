from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
import pytest
from test_numeric_condition_summary import dataset, Once, H4, FIELDS
from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
from sandbox.research import cohort_comparison as module
from sandbox.research.numeric_condition_summary import summarize_numeric_conditioned_event_cohorts


def data_for(values):
    data=dataset(range(len(values)))
    return assemble_discovery_dataset(tuple(replace(row,targets=(replace(row.targets[0],value=value),row.targets[1])) for row,value in zip(data.rows,values)))


def compare(data,left=(0,1,2),right=(3,4,5)):
    return module.compare_cohort_rows(data,left,right,FIELDS[0])


def test_simple_and_symmetry():
    data=data_for([1,2,3,2,3,4]); r=compare(data)
    assert (r.left_observed_count,r.right_observed_count,r.pair_count)==(3,3,9)
    assert (r.left_greater_count,r.tie_count,r.left_less_count)==(1,2,6)
    assert (r.left_mean,r.right_mean,r.mean_difference)==(2.,3.,-1.)
    assert (r.left_median,r.right_median,r.median_difference)==(2.,3.,-1.)
    assert r.probability_of_superiority==pytest.approx(2/9)
    assert r.cliffs_delta==pytest.approx(-5/9)
    s=compare(data,(3,4,5),(0,1,2))
    assert s.mean_difference==-r.mean_difference and s.median_difference==-r.median_difference
    assert s.probability_of_superiority==pytest.approx(1-r.probability_of_superiority)
    assert s.cliffs_delta==pytest.approx(-r.cliffs_delta)
    assert (s.pair_count,s.tie_count,s.left_greater_count,s.left_less_count)==(9,2,6,1)
    assert compare(data)==r


def test_missing_identical_and_zero_observations():
    r=compare(data_for([None,1,3,2,None,4]))
    assert (r.left_row_count,r.right_row_count)==(3,3)
    assert (r.left_observed_count,r.right_observed_count,r.left_missing_count,r.right_missing_count)==(2,2,1,1)
    assert (r.pair_count,r.left_greater_count,r.tie_count,r.left_less_count)==(4,1,0,3)
    r=compare(data_for([1,2,3,1,2,3]))
    assert (r.mean_difference,r.median_difference,r.probability_of_superiority,r.cliffs_delta)==(0.,0.,.5,0.)
    for values in ([None]*6,[None]*3+[1,2,3],[1,2,3]+[None]*3):
        r=compare(data_for(values))
        assert (r.pair_count,r.left_greater_count,r.tie_count,r.left_less_count)==(0,0,0,0)
        assert r.probability_of_superiority is r.cliffs_delta is r.mean_difference is r.median_difference is None
        assert (r.left_mean is None)==(values[0] is None)
        assert (r.right_median is None)==(values[3] is None)


@pytest.mark.parametrize('left,right', [((),(1,)),((0,),()),((True,),(1,)),((-1,),(1,)),((0,0),(1,)),
    ((1,0),(2,)),((0,),(6,)),((0,),(0,)),((0.,),(1,)),((0,),('1',)),((0,2),(1,2))])
def test_indices_rejected(left,right):
    with pytest.raises(ValueError): compare(data_for(range(6)),left,right)


@pytest.mark.parametrize('field,error', [(1,TypeError),('meta.id',ValueError),(H4,ValueError),('y.unknown',KeyError)])
def test_fields(field,error):
    with pytest.raises(error): module.compare_cohort_rows(data_for([1,2]),(0,),(1,),field)


@pytest.mark.parametrize('value', [True,False,'1',date.today(),datetime.now(),object(),float('nan'),float('inf'),-float('inf'),10**400])
def test_invalid_target(value,monkeypatch):
    monkeypatch.setattr(module,'column_values',lambda *args:(value,1))
    with pytest.raises(ValueError): compare(data_for([1,2]),(0,),(1,))


def test_once_raw_selected_values_and_bisect(monkeypatch):
    data=data_for([1,2,3,2,3,4]); reads=[]; calls=[]
    original=module.column_values; low=module.bisect_left; high=module.bisect_right
    def read(data,field): reads.append(field); return original(data,field)
    def lower(values,value): calls.append('low'); return low(values,value)
    def upper(values,value): calls.append('high'); return high(values,value)
    monkeypatch.setattr(module,'column_values',read)
    monkeypatch.setattr(module,'bisect_left',lower); monkeypatch.setattr(module,'bisect_right',upper)
    left,right=Once((0,1,2)),Once((3,4,5))
    r=compare(data,left,right)
    assert left.calls==right.calls==1 and reads==[FIELDS[0]]
    assert len(calls)==2*r.left_observed_count
    monkeypatch.setattr(module,'column_values',lambda *args:(1,2,object()))
    assert compare(data_for([1,2,3]),(0,),(1,)).pair_count==1
    with pytest.raises(TypeError): compare(object())


def test_reorder_and_unrelated_mutations():
    data=data_for([1,2,3,2,3,4]); original=compare(data)
    reordered=compare(assemble_discovery_dataset(data.rows[::-1]),(3,4,5),(0,1,2))
    assert replace(reordered,left_row_indices=(0,1,2),right_row_indices=(3,4,5))==original
    rows=tuple(replace(r,identity=tuple(replace(v,value=999) for v in r.identity),
        predictors=tuple(replace(v,value=-999) if v.field_id==H4 else v for v in r.predictors),
        targets=(r.targets[0],replace(r.targets[1],value=999))) for r in data.rows)
    assert compare(assemble_discovery_dataset(rows))==original
    assert compare(data_for([10,20,30,2,3,4]))!=original


def test_integration_raw_targets():
    data=data_for([10,20,1,2])
    bins=summarize_numeric_conditioned_event_cohorts(data,H4,(2,),field_ids=())
    assert len(bins)==2 and bins[0].outcome_distributions==()
    result=module.compare_cohort_rows(data,bins[0].row_indices,bins[1].row_indices,FIELDS[0])
    assert result.field_id==FIELDS[0]
    assert (result.left_row_count,result.right_row_count)==(2,2)
    assert (result.left_mean,result.right_mean)==(15.,1.5)
    assert (result.pair_count,result.left_greater_count,result.cliffs_delta)==(4,4,1.)


@pytest.mark.parametrize('changes', [
    {'field_id':''},{'left_row_indices':[0,1,2]},{'right_row_indices':(True,4,5)},
    {'left_row_indices':(-1,1,2)},{'left_row_indices':(0,0,2)},{'left_row_indices':(2,1,0)},
    {'right_row_indices':(2,4,5)},{'left_row_count':True},{'right_row_count':0},{'left_row_count':4},
    {'left_observed_count':True},{'left_missing_count':-1},{'right_missing_count':1},
    {'left_mean':None},{'right_median':2},{'left_mean':float('inf')},{'mean_difference':None},
    {'median_difference':float('nan')},{'mean_difference':0.},{'pair_count':True},{'pair_count':8},
    {'left_greater_count':-1},{'tie_count':True},{'left_less_count':0},
    {'probability_of_superiority':None},{'probability_of_superiority':2.},
    {'probability_of_superiority':True},{'probability_of_superiority':.5},
    {'cliffs_delta':None},{'cliffs_delta':-2.},{'cliffs_delta':0.}])
def test_dataclass_validation(changes):
    result=compare(data_for([1,2,3,2,3,4]))
    with pytest.raises(ValueError): replace(result,**changes)


def test_empty_stats_validation_and_frozen():
    r=compare(data_for([None]*6))
    for changes in ({'left_mean':0.},{'right_median':0.},{'mean_difference':0.},
                    {'median_difference':0.},{'probability_of_superiority':.5},{'cliffs_delta':0.}):
        with pytest.raises(ValueError): replace(r,**changes)
    with pytest.raises(FrozenInstanceError): r.pair_count=1
