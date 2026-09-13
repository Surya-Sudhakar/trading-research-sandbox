from dataclasses import FrozenInstanceError, replace
import pytest
from test_context_scan import dataset, H4, H1, NUMBER, TARGETS, Once
from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
from sandbox.research import context_interaction_scan as module


def example():
    return dataset([('BEARISH','BEARISH',.5),('BEARISH','BEARISH',.6),('BEARISH','BULLISH',.8),
                    ('BULLISH','BEARISH',.7),('BULLISH','BULLISH',.1),('BULLISH','BULLISH',.3)])


def scan(data=None, **kwargs):
    return module.scan_two_context_interactions(example() if data is None else data,
        categorical_field_ids=(H4,H1),field_ids=(TARGETS[0],),**kwargs)


def test_categorical_three_baselines():
    results=scan(); r=results[0]
    assert len(results)==4
    assert r.left_condition.categorical_value==r.right_condition.categorical_value=='BEARISH'
    assert r.left_row_indices==(0,1,2) and r.right_row_indices==(0,1,3)
    assert r.joint_row_indices==(0,1) and r.event_complement_row_indices==(2,3,4,5)
    assert r.left_without_right_row_indices==(2,) and r.right_without_left_row_indices==(3,)
    for comparison, right in ((r.joint_vs_event_complement,(2,3,4,5)),
        (r.joint_vs_left_without_right,(2,)),(r.joint_vs_right_without_left,(3,))):
        assert comparison.left_row_indices==(0,1)
        assert comparison.right_row_indices==right
        assert comparison.field_id==TARGETS[0]
        assert comparison.left_mean==.5
    assert r.joint_vs_event_complement.right_mean==3.5


def test_categorical_numeric_and_numeric_numeric():
    data=dataset([(True,10,.5),(True,30,.6),(False,60,.7),(True,90,.9)])
    result=module.scan_two_context_interactions(data,(H4,),((NUMBER,(.5,.75)),),field_ids=(TARGETS[0],))
    r=next(r for r in result if r.left_condition.categorical_value is True and r.right_condition.numeric_bin_index==1)
    assert r.joint_row_indices==(0,1)
    assert r.event_complement_row_indices==(2,3)
    assert r.left_without_right_row_indices==(3,) and r.right_without_left_row_indices==(2,)
    assert r.right_condition.numeric_cutpoints==(.5,.75)
    result=module.scan_two_context_interactions(data,numeric_condition_specs=((NUMBER,(.5,.75)),(H1,(25,50,75))),field_ids=(TARGETS[0],))
    r=next(r for r in result if r.left_condition.numeric_bin_index==1 and r.right_condition.numeric_bin_index==1)
    assert r.joint_row_indices==(1,)
    assert r.left_without_right_row_indices==(0,2)
    assert r.right_without_left_row_indices==() and r.joint_vs_right_without_left is None
    assert r.right_condition.numeric_cutpoints==(25.,50.,75.)


def test_same_field_exclusion_and_source_order():
    data=example()
    assert module.scan_two_context_interactions(data,numeric_condition_specs=((NUMBER,(.2,)),(NUMBER,(.7,))))==()
    result=module.scan_two_context_interactions(data,(H1,H4),((NUMBER,(.2,)),(NUMBER,(.7,))),TARGETS[::-1])
    keys=[(r.left_condition.condition_field_id,r.right_condition.condition_field_id,r.right_condition.numeric_cutpoints) for r in result]
    order=list(dict.fromkeys(keys))
    assert order==[(H1,H4,None),(H1,NUMBER,(.2,)),(H1,NUMBER,(.7,)),(H4,NUMBER,(.2,)),(H4,NUMBER,(.7,))]
    assert all(r.left_condition.condition_field_id!=r.right_condition.condition_field_id for r in result)
    assert [r.target_field_id for r in result]==list(TARGETS[::-1])*(len(result)//2)


def test_events_and_duplicate_objects():
    events=['ABOVE_TOUCH_BELOW','ABOVE_TOUCH_ABOVE',8]*2
    data=dataset([('A','A',0)]*3+[('B','B',1)]*3,events)
    data=assemble_discovery_dataset(data.rows+(data.rows[0],))
    result=scan(data)
    assert len(result)==6
    assert [r.event_cohort_id for r in result]==sorted(r.event_cohort_id for r in result)
    for r in result:
        all_rows=r.joint_row_indices+r.event_complement_row_indices
        assert len({(events+events[:1])[i] for i in all_rows})==1
    assert next(r for r in result if 0 in r.joint_row_indices).joint_row_indices==(0,6)


def test_skips_and_optional_baselines():
    assert scan(dataset([('A','B',0)]*3))==()
    result=scan(dataset([('A','A',0),('B','B',0)]))
    assert len(result)==2
    assert all(r.joint_vs_left_without_right is r.joint_vs_right_without_left is None for r in result)
    result=scan(dataset([('A','A',0),('A','B',0),('B','B',0)]))
    assert any(r.joint_vs_left_without_right is None and r.joint_vs_right_without_left is not None for r in result)
    assert any(r.joint_vs_right_without_left is None and r.joint_vs_left_without_right is not None for r in result)


def test_delegation_materialization(monkeypatch):
    cats,nums,calls=[],[],[]
    cat,num,compare=module.summarize_single_conditioned_event_cohorts,module.summarize_numeric_conditioned_event_cohorts,module.compare_cohort_rows
    def categorical(data,field,**kw):
        cats.append((field,kw))
        return cat(data,field,**kw)[::-1]
    def numeric(data,field,cuts,**kw):
        nums.append((field,cuts,kw))
        return num(data,field,cuts,**kw)
    def comparison(data,left,right,field):
        calls.append((left,right,field))
        return compare(data,left,right,field)
    monkeypatch.setattr(module,'summarize_single_conditioned_event_cohorts',categorical)
    monkeypatch.setattr(module,'summarize_numeric_conditioned_event_cohorts',numeric)
    monkeypatch.setattr(module,'compare_cohort_rows',comparison)
    fields,categories,cuts=Once(TARGETS[::-1]),Once((H4,H1)),Once((.5,))
    specs=Once(((NUMBER,cuts),))
    result=module.scan_two_context_interactions(example(),categories,specs,fields)
    assert fields.calls==categories.calls==cuts.calls==specs.calls==1
    assert cats==[(H4,{'field_ids':()}),(H1,{'field_ids':()})]
    assert nums==[(NUMBER,(.5,),{'field_ids':()})]
    assert result[0].left_condition.categorical_value=='BULLISH'
    expected=[]
    for r in result:
        for baseline in (r.event_complement_row_indices,r.left_without_right_row_indices,r.right_without_left_row_indices):
            if baseline: expected.append((r.joint_row_indices,baseline,r.target_field_id))
    assert calls==expected


@pytest.mark.parametrize('mismatch', ['universe','metadata','within_source'])
def test_upstream_partition_mismatch(monkeypatch,mismatch):
    original=module.summarize_single_conditioned_event_cohorts
    def conditioning(data,field,**kwargs):
        groups=original(data,field,**kwargs)
        if field==H1:
            if mismatch=='universe': return groups[:1]
            if mismatch=='metadata': return tuple(replace(g,anchor_kind='OTHER') for g in groups)
            return (replace(groups[0],pattern_code='OTHER'),)+groups[1:]
        return groups
    monkeypatch.setattr(module,'summarize_single_conditioned_event_cohorts',conditioning)
    with pytest.raises(ValueError): scan()


def test_empty_and_no_sources(monkeypatch):
    class Never:
        def __iter__(self): raise AssertionError('consumed')
    def fail(*args,**kwargs): raise AssertionError('delegated')
    for name in ('summarize_single_conditioned_event_cohorts','summarize_numeric_conditioned_event_cohorts','compare_cohort_rows'):
        monkeypatch.setattr(module,name,fail)
    assert module.scan_two_context_interactions(assemble_discovery_dataset(()),Never(),Never(),Never())==()
    assert module.scan_two_context_interactions(example())==()
    with pytest.raises(TypeError): module.scan_two_context_interactions(object())


@pytest.mark.parametrize('kwargs,error', [({'categorical_field_ids':(H4,H4)},ValueError),
    ({'categorical_field_ids':(1,)},TypeError),({'categorical_field_ids':('x.unknown',)},KeyError),
    ({'categorical_field_ids':('meta.id',)},ValueError),({'categorical_field_ids':('x.anchor_kind',)},ValueError),
    ({'field_ids':(TARGETS[0],)*2},ValueError),({'field_ids':(1,)},TypeError),
    ({'field_ids':(H4,)},ValueError),({'field_ids':('y.unknown',)},KeyError),
    ({'numeric_condition_specs':((NUMBER,(.7,.2)),)},ValueError)])
def test_bad_inputs(kwargs,error):
    with pytest.raises(error): module.scan_two_context_interactions(example(),**kwargs)


def test_leakage_and_selected_predictor():
    data=example(); original=scan(data)
    changed=assemble_discovery_dataset(tuple(replace(r,targets=(replace(r.targets[0],value=100-i*20),r.targets[1])) for i,r in enumerate(data.rows)))
    result=scan(changed)
    assert [r.joint_row_indices for r in result]==[r.joint_row_indices for r in original]
    assert all(a.joint_vs_event_complement!=b.joint_vs_event_complement for a,b in zip(result,original))
    unrelated=assemble_discovery_dataset(tuple(replace(r,identity=(replace(r.identity[0],value='OTHER'),),
        targets=(r.targets[0],replace(r.targets[1],value=999)),
        predictors=tuple(replace(v,value=-99) if v.field_id==NUMBER else v for v in r.predictors)) for r in data.rows))
    assert scan(unrelated)==original
    selected=assemble_discovery_dataset(tuple(replace(r,predictors=tuple(replace(v,value='ALL') if v.field_id==H1 else v for v in r.predictors)) for r in data.rows))
    assert [r.joint_row_indices for r in scan(selected)]!=[r.joint_row_indices for r in original]


@pytest.mark.parametrize('changes', [{'context_kind':'UNKNOWN'},{'condition_field_id':''},{'categorical_value':.5},
    {'numeric_cutpoints':(.5,)},{'numeric_missing':False},{'numeric_bin_index':0},
    {'numeric_lower_bound':0.},{'numeric_upper_bound':1.},{'numeric_lower_inclusive':True},{'numeric_upper_inclusive':False}])
def test_categorical_condition_validation(changes):
    with pytest.raises(ValueError): replace(scan()[0].left_condition,**changes)


@pytest.mark.parametrize('changes', [{'numeric_cutpoints':()},{'numeric_cutpoints':[.5]},
    {'numeric_cutpoints':(1,)},{'numeric_cutpoints':(float('nan'),)},
    {'numeric_cutpoints':(.75,.5)},{'numeric_cutpoints':(.5,.5)},
    {'numeric_missing':1},{'numeric_bin_index':True},{'numeric_bin_index':9},
    {'categorical_value':False},{'numeric_lower_bound':.3},{'numeric_upper_inclusive':True}])
def test_numeric_condition_validation(changes):
    c=module.ContextCondition('NUMERIC',NUMBER,None,(.5,.75),False,1,.5,.75,True,False)
    with pytest.raises(ValueError): replace(c,**changes)


@pytest.mark.parametrize('changes', [{'left_condition':None},{'right_condition':object()},{'event_cohort_id':''},
    {'anchor_kind':None},{'pattern_code':1},{'checkpoint_bars':True},{'target_field_id':''},
    {'left_row_indices':[0,1,2]},{'right_row_indices':(True,1,3)},
    {'joint_row_indices':()},{'joint_row_indices':(-1,)},{'joint_row_indices':(1,0)},
    {'joint_row_indices':(0,0)},{'joint_row_indices':(0,)},{'event_complement_row_indices':()},
    {'event_complement_row_indices':(0,2,3,4,5)},{'event_complement_row_indices':(4,5)},
    {'left_without_right_row_indices':()},{'right_without_left_row_indices':(2,)},
    {'joint_vs_event_complement':None},{'joint_vs_left_without_right':None},
    {'joint_vs_right_without_left':None},{'target_field_id':TARGETS[1]}])
def test_result_validation(changes):
    with pytest.raises(ValueError): replace(scan()[0],**changes)


def test_condition_identity_optional_and_frozen():
    r=scan()[0]
    with pytest.raises(ValueError): replace(r,right_condition=r.left_condition)
    with pytest.raises(ValueError): replace(r,joint_vs_event_complement=r.joint_vs_left_without_right)
    redundant=scan(dataset([('A','A',0),('B','B',1)]))[0]
    with pytest.raises(ValueError): replace(redundant,joint_vs_left_without_right=redundant.joint_vs_event_complement)
    with pytest.raises(FrozenInstanceError): r.target_field_id='y.other'
    with pytest.raises(FrozenInstanceError): r.left_condition.categorical_value='OTHER'
    missing=module.ContextCondition('NUMERIC',NUMBER,None,(.5,),True,None,None,None,None,None)
    with pytest.raises(ValueError): replace(missing,numeric_bin_index=0)
