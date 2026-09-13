from dataclasses import FrozenInstanceError, replace
import pytest
from test_context_scan import Once, H4, H1, NUMBER, TARGETS
from test_context_interaction_scan import example, scan
from sandbox.research.context_scan import SingleContextScanResult, scan_single_contexts
from sandbox.research.context_interaction_scan import ContextCondition, TwoContextScanResult, scan_two_context_interactions
from sandbox.research.cohort_comparison import NumericCohortComparison
from sandbox.research import context_candidate_rank as module


def comparison(left,right,delta,field=TARGETS[0],missing_left=0,missing_right=0):
    nl,nr=len(left)-missing_left,len(right)-missing_right
    pairs=nl*nr
    greater=round((1+delta)*pairs/2) if pairs else 0
    less=pairs-greater
    lm=0. if nl else None; rm=0. if nr else None
    difference=0. if pairs else None
    return NumericCohortComparison(field,left,right,len(left),len(right),nl,nr,missing_left,missing_right,
        lm,rm,difference,lm,rm,difference,pairs,greater,0,less,
        greater/pairs if pairs else None,(greater-less)/pairs if pairs else None)


def single(delta=.2,n=20,m=20,event='EVENT',target=TARGETS[0],missing_left=0):
    left=tuple(range(n)); right=tuple(range(n,n+m))
    return SingleContextScanResult('CATEGORICAL',H4,None,event,'TOUCH_TRANSITION','PATTERN',None,
        left,right,target,comparison(left,right,delta,target,missing_left), 'A',*(None,)*6)


def interaction(deltas=(.8,.3,.55), redundant=False,event='EVENT',target=TARGETS[0]):
    joint=tuple(range(20)); left_only=() if redundant else tuple(range(20,40))
    right_only=tuple(range(40,60)); rest=tuple(range(20,80))
    a=ContextCondition('CATEGORICAL',H4,'A',None,*(None,)*6)
    b=ContextCondition('CATEGORICAL',H1,'B',None,*(None,)*6)
    return TwoContextScanResult(a,b,event,'TOUCH_TRANSITION','PATTERN',None,
        joint+left_only,joint+right_only,joint,rest,left_only,right_only,target,
        comparison(joint,rest,deltas[0],target),
        comparison(joint,left_only,deltas[1],target) if left_only else None,
        comparison(joint,right_only,deltas[2],target))


def test_single_abs_priority_and_stability():
    sources=(single(.2),single(-.7),single(.4))
    result=module.rank_context_candidates(sources)
    assert [c.priority_score for c in result]==[.7,.4,.2]
    assert [c.source_order for c in result]==[1,2,0]
    assert result[0].single_result is sources[1]
    assert result[0].single_result.comparison.cliffs_delta==-.7
    assert module.rank_context_candidates(sources)==result
    assert tuple(s.comparison.cliffs_delta for s in sources)==(.2,-.7,.4)


def test_interaction_minimum_and_incremental_control():
    source=interaction()
    r=module.rank_context_candidates(interaction_results=(source,))[0]
    assert r.priority_score==.3
    assert r.minimum_comparison_pair_count==400
    assert r.minimum_comparison_observed_per_side==20
    assert r.interaction_result is source
    redundant=interaction(redundant=True)
    assert module.rank_context_candidates(interaction_results=(redundant,))==()
    assert module.rank_context_candidates(interaction_results=(redundant,),require_incremental_baselines=False)[0].priority_score==.55
    zero=interaction((.8,0.,.55))
    assert module.rank_context_candidates(interaction_results=(zero,))[0].priority_score==0.
    assert module.rank_context_candidates(interaction_results=(zero,),minimum_abs_cliffs_delta=.01)==()


@pytest.mark.parametrize('controls', [{'minimum_rows_per_side':21},{'minimum_observed_per_side':21},
    {'minimum_pair_count':401},{'minimum_abs_cliffs_delta':.21}])
def test_single_filters(controls):
    assert module.rank_context_candidates((single(.2),),**controls)==()


@pytest.mark.parametrize('controls', [{'minimum_rows_per_side':21},{'minimum_observed_per_side':21},
    {'minimum_pair_count':401},{'minimum_abs_cliffs_delta':.31}])
def test_interaction_each_comparison_filter(controls):
    assert module.rank_context_candidates(interaction_results=(interaction(),),**controls)==()


def test_filter_boundaries_missing_and_topk():
    s=single(.2)
    assert len(module.rank_context_candidates((s,),minimum_rows_per_side=20,minimum_observed_per_side=20,
        minimum_pair_count=400,minimum_abs_cliffs_delta=.2))==1
    assert module.rank_context_candidates((single(.2,missing_left=20),))==()
    partial=single(.2,missing_left=10)
    assert len(module.rank_context_candidates((partial,),minimum_rows_per_side=20,minimum_observed_per_side=10))==1
    assert module.rank_context_candidates((partial,),minimum_observed_per_side=11)==()
    assert [c.priority_score for c in module.rank_context_candidates((single(.2),single(-.7),single(.4)),top_k=2)]==[.7,.4]
    source=interaction()
    absent=comparison(source.joint_row_indices,source.left_without_right_row_indices,0.,missing_right=20)
    source=replace(source,joint_vs_left_without_right=absent)
    for required in (True,False):
        assert module.rank_context_candidates(interaction_results=(source,),require_incremental_baselines=required)==()


def test_tie_order_all_keys():
    # Pair counts precede minimum observed counts; then identity, kind, and stream position.
    sources=(single(.5,n=10,m=40,event='Z'),single(.5,n=20,m=20,event='Z'),
             single(.5,n=20,m=30,event='Z'),single(.5,event='B'),
             single(.5,event='A',target=TARGETS[0]),single(.5,event='A',target=TARGETS[1]),
             single(.5,event='A',target=TARGETS[1]))
    result=module.rank_context_candidates(sources)
    assert [c.source_order for c in result]==[2,5,6,4,3,1,0]
    pair=interaction((.5,.5,.5),event='A',target=TARGETS[1])
    result=module.rank_context_candidates(sources,(pair,))
    assert [c.source_order for c in result]==[2,7,5,6,4,3,1,0]
    assert result[1].candidate_kind=='INTERACTION'


def test_source_order_materialization_and_scale_neutrality():
    sources=(single(.2),single(-.7)); pairs=(interaction(),)
    singles,interactions=Once(sources),Once(pairs)
    result=module.rank_context_candidates(singles,interactions)
    assert singles.calls==interactions.calls==1
    assert [c.source_order for c in result]==[1,2,0]
    changed=replace(sources[0],comparison=replace(sources[0].comparison,left_mean=1000.,right_mean=-1000.,
        mean_difference=2000.,left_median=900.,right_median=-900.,median_difference=1800.))
    after=module.rank_context_candidates((changed,sources[1]),pairs)
    assert [(c.source_order,c.priority_score) for c in after]==[(c.source_order,c.priority_score) for c in result]
    assert {c.candidate_kind for c in after}=={'SINGLE','INTERACTION'}
    assert sources[0].comparison.mean_difference==0.


@pytest.mark.parametrize('name,value', [(name,value) for name in
    ('minimum_rows_per_side','minimum_observed_per_side','minimum_pair_count') for value in (True,0,-1,1.,None)])
def test_bad_count_controls(name,value):
    with pytest.raises(ValueError): module.rank_context_candidates(**{name:value})


@pytest.mark.parametrize('value',[True,None,'0',float('nan'),float('inf'),-float('inf'),-.1,1.1,10**400])
def test_bad_delta_control(value):
    with pytest.raises(ValueError): module.rank_context_candidates(minimum_abs_cliffs_delta=value)


@pytest.mark.parametrize('kwargs',[{'require_incremental_baselines':1},{'require_incremental_baselines':None},
    {'top_k':True},{'top_k':0},{'top_k':-1},{'top_k':1.}])
def test_other_controls(kwargs):
    with pytest.raises(ValueError): module.rank_context_candidates(**kwargs)


def test_empty_invalid_sources_and_integer_delta():
    assert module.rank_context_candidates()==()
    assert len(module.rank_context_candidates((single(1.),),minimum_abs_cliffs_delta=1))==1
    with pytest.raises(TypeError): module.rank_context_candidates((object(),))
    with pytest.raises(TypeError): module.rank_context_candidates(interaction_results=(single(),))
    # Filtering does not renumber later source identities.
    result=module.rank_context_candidates((single(.2,missing_left=20),single(.7)))
    assert result[0].source_order==1


@pytest.mark.parametrize('changes',[{'candidate_kind':'OTHER'},{'target_field_id':'y.other'},
    {'event_cohort_id':''},{'priority_score':True},{'priority_score':float('nan')},
    {'priority_score':1.1},{'priority_score':.9},{'minimum_comparison_pair_count':True},
    {'minimum_comparison_pair_count':0},{'minimum_comparison_pair_count':399},
    {'minimum_comparison_observed_per_side':0},{'minimum_comparison_observed_per_side':19},
    {'source_order':True},{'source_order':-1},{'single_result':None},
    {'single_result':object()},{'interaction_result':object()}])
def test_ranked_dataclass_validation(changes):
    result=module.rank_context_candidates((single(),))[0]
    with pytest.raises(ValueError): replace(result,**changes)


def test_interaction_dataclass_and_frozen():
    candidate=module.rank_context_candidates(interaction_results=(interaction(),))[0]
    with pytest.raises(ValueError): replace(candidate,single_result=single())
    with pytest.raises(ValueError): replace(candidate,interaction_result=None)
    with pytest.raises(ValueError): replace(candidate,candidate_kind='SINGLE')
    with pytest.raises(FrozenInstanceError): candidate.priority_score=.9


def test_vertical_slice():
    data=example()
    singles=scan_single_contexts(data,(H4,H1),((NUMBER,(.25,.5,.75)),),field_ids=(TARGETS[0],))
    interactions=scan_two_context_interactions(data,(H4,H1),((NUMBER,(.25,.5,.75)),),field_ids=(TARGETS[0],))
    result=module.rank_context_candidates(singles,interactions)
    assert singles and interactions and result
    assert {r.candidate_kind for r in result}=={'SINGLE','INTERACTION'}
    assert [r.priority_score for r in result]==sorted((r.priority_score for r in result),reverse=True)
    assert all(r.target_field_id==TARGETS[0] for r in result)
    raw=scan(data)[0]
    assert raw.joint_row_indices==(0,1)
    assert raw.joint_vs_event_complement.left_mean==.5
    assert raw.joint_vs_event_complement.right_mean==3.5
    for ranked in result:
        source=ranked.single_result or ranked.interaction_result
        if ranked.candidate_kind=='SINGLE':
            assert any(source is s for s in singles)
            assert ranked.priority_score==abs(source.comparison.cliffs_delta)
        else:
            assert any(source is s for s in interactions)
            assert ranked.priority_score==min(abs(c.cliffs_delta) for c in (
                source.joint_vs_event_complement,source.joint_vs_left_without_right,source.joint_vs_right_without_left))
