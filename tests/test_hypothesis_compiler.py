from dataclasses import FrozenInstanceError, replace
import pytest
from test_ai_researcher import packet, proposal_data, FEATURES
from sandbox.research.ai_researcher import parse_hypothesis_proposal
from sandbox.research import hypothesis_compiler as module
from sandbox.strategies.dsl.conditions import AndCondition, ComparisonCondition, ComparisonOperator
from sandbox.strategies.dsl.execution import MarketEntry, FixedPipsStop, RiskMultipleTarget


def compiled():
    p=packet(); proposal=parse_hypothesis_proposal(proposal_data(p),p)
    return module.compile_hypothesis(proposal,p)


def test_complete_compilation_and_fingerprints():
    c=compiled(); spec=c.strategy_spec
    assert (spec.family_id,spec.strategy_id,spec.variant_id)==('MACHINE_DISCOVERED','ML_CONTEXT_001','V1')
    assert len(spec.rules)==1
    rule=spec.rules[0]
    assert type(rule.condition) is AndCondition
    first,second=rule.condition.children
    assert first.reference.namespace=='feature' and first.reference.key==FEATURES[0]
    assert first.reference.timeframe is first.reference.context is None
    assert first.operator==ComparisonOperator.EQ and first.value=='BULLISH'
    assert second.reference.key==FEATURES[1] and second.operator==ComparisonOperator.LT and second.value==30.
    assert type(rule.entry) is MarketEntry and rule.entry.side=='LONG'
    assert type(rule.stop) is FixedPipsStop and rule.stop.pips==10.
    assert type(rule.target) is RiskMultipleTarget and rule.target.multiple==2.
    assert c.strategy_spec_fingerprint==module._digest(spec.model_dump(mode='json'))
    assert c.fingerprint==module._digest(module._compiled_payload(c))
    assert compiled()==c


def test_single_and_multi_rule_order():
    p=packet(); data=proposal_data(p)
    data['rules'][0]['conditions']=data['rules'][0]['conditions'][:1]
    data['rules'][0]['rule_id']='Z_FIRST'
    second={**data['rules'][0],'rule_id':'A_SECOND','side':'SHORT'}
    data['rules'].append(second)
    c=module.compile_hypothesis(parse_hypothesis_proposal(data,p),p)
    assert [r.rule_id for r in c.strategy_spec.rules]==['Z_FIRST','A_SECOND']
    assert all(type(r.condition) is ComparisonCondition for r in c.strategy_spec.rules)
    assert [r.entry.side for r in c.strategy_spec.rules]==['LONG','SHORT']


@pytest.mark.parametrize('operator',['EQ','NE','GT','GTE','LT','LTE','IN','NOT_IN'])
def test_all_operator_mappings(operator):
    p=packet(); data=proposal_data(p)
    data['rules'][0]['conditions']=data['rules'][0]['conditions'][:1]
    value=['BULLISH','BEARISH'] if operator in ('IN','NOT_IN') else 'BULLISH'
    data['rules'][0]['conditions'][0].update(operator=operator,value=value)
    proposal=parse_hypothesis_proposal(data,p)
    c=module.compile_hypothesis(proposal,p)
    assert c.strategy_spec.rules[0].condition.operator.value==operator
    assert c.strategy_spec.rules[0].condition.value==value


def test_condition_order_changes_identity():
    p=packet(); data=proposal_data(p)
    original=module.compile_hypothesis(parse_hypothesis_proposal(data,p),p)
    data['rules'][0]['conditions'].reverse()
    changed=module.compile_hypothesis(parse_hypothesis_proposal(data,p),p)
    assert changed.strategy_spec.rules[0].condition.children[0].reference.key==FEATURES[1]
    assert changed.strategy_spec_fingerprint!=original.strategy_spec_fingerprint
    assert changed.fingerprint!=original.fingerprint


def test_compiler_revalidates_membership_status_and_types():
    p=packet()
    for field in ('x.unknown','meta.id','y.future'):
        proposal=parse_hypothesis_proposal(proposal_data(p),p)
        object.__setattr__(proposal.rules[0].conditions[0],'field_id',field)
        with pytest.raises(ValueError): module.compile_hypothesis(proposal,p)
    proposal=parse_hypothesis_proposal(proposal_data(p),p)
    object.__setattr__(proposal,'status','VALIDATED')
    with pytest.raises(ValueError): module.compile_hypothesis(proposal,p)
    with pytest.raises(TypeError): module.compile_hypothesis(proposal_data(p),p)
    with pytest.raises(TypeError): module.compile_hypothesis(proposal,object())


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'hypothesis_fingerprint':'bad'},
    {'research_packet_fingerprint':'bad'},{'hypothesis_id':'lower'},{'target_field_id':'x.other'},
    {'strategy_spec':object()},{'strategy_spec_fingerprint':'0'*64},{'fingerprint':'0'*64}])
def test_compiled_validation(changes):
    with pytest.raises((ValueError,TypeError)): replace(compiled(),**changes)


def test_nested_dsl_mutation_rejected_and_frozen():
    p=packet(); data=proposal_data(p)
    data['rules'][0]['conditions'][0].update(operator='IN',value=['BULLISH'])
    c=module.compile_hypothesis(parse_hypothesis_proposal(data,p),p)
    c.strategy_spec.rules[0].condition.children[0].value.append(float('nan'))
    with pytest.raises(ValueError): replace(c)
    with pytest.raises(FrozenInstanceError): c.hypothesis_id='OTHER'
