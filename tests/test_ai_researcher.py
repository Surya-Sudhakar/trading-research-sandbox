from dataclasses import asdict, FrozenInstanceError, replace
from copy import deepcopy
import json
import pytest
from test_ml_evaluation import evaluate
from sandbox.research.ml_clue_extraction import MLFeatureClue, MLClueReport
from sandbox.research import ai_researcher as module

FEATURES=('x.h4.direction','x.atr_percentile_100')


def evidence():
    evaluation=evaluate()
    clues=(MLFeatureClue(FEATURES[0],'CATEGORICAL',0,3,30.,.75,.75),
           MLFeatureClue(FEATURES[1],'NUMERIC',1,1,10.,.25,.25))
    state=dict(version='ML_CLUE_REPORT_V1',model_fingerprint=evaluation.model_fingerprint,model_sha256='b'*64,
        training_matrix_fingerprint='c'*64,preprocessor_fingerprint='d'*64,evaluation_fingerprint=evaluation.fingerprint,
        target_field_id=evaluation.target_field_id,feature_count=2,total_split_count=4,total_gain=40.,feature_clues=clues,
        gain_order=FEATURES,split_order=FEATURES)
    payload={**state,'feature_clues':[asdict(c) for c in clues]}
    return MLClueReport(**state,fingerprint=module._digest(payload)),evaluation


def packet():
    return module.build_research_packet(*evidence())


def proposal_data(p=None):
    p=packet() if p is None else p
    return dict(version='HYPOTHESIS_PROPOSAL_V1',status='UNVERIFIED_DISCOVERY_HYPOTHESIS',
        research_packet_fingerprint=p.fingerprint,hypothesis_id='ML_HYP_001',
        statement='Test the specified conjunction on Discovery.',rationale='Proposed threshold requires testing.',
        target_field_id=p.target_field_id,family_id='MACHINE_DISCOVERED',strategy_id='ML_CONTEXT_001',variant_id='V1',
        evidence_feature_ids=list(FEATURES),rules=[dict(rule_id='RULE_1',conditions=[
            dict(field_id=FEATURES[0],operator='EQ',value='BULLISH'),
            dict(field_id=FEATURES[1],operator='LT',value=30.)],condition_logic='ALL',side='LONG',stop_pips=10.,target_r_multiple=2.)])


def test_packet_exact_evidence_and_no_raw_data():
    clues,evaluation=evidence(); p=module.build_research_packet(clues,evaluation)
    assert p.feature_ids==p.allowed_condition_feature_ids==FEATURES
    assert p.feature_kinds==('CATEGORICAL','NUMERIC')
    assert p.feature_gain_fractions==p.feature_split_fractions==(.75,.25)
    assert p.gain_order==p.split_order==FEATURES
    assert p.evaluation_summary==tuple((name,getattr(evaluation,name)) for name in module.SUMMARY_FIELDS)
    assert p.clue_report_fingerprint==clues.fingerprint
    assert p.evaluation_fingerprint==evaluation.fingerprint and p.model_fingerprint==evaluation.model_fingerprint
    assert (p.permitted_entry_kinds,p.permitted_stop_kinds,p.permitted_target_kinds)==(('market',),('fixed_pips',),('r_multiple',))
    assert p.researcher_instructions==module.RESEARCHER_INSTRUCTIONS
    assert not {'model_text','x_rows','y_values','predictions'}.intersection(asdict(p))
    assert module.build_research_packet(clues,evaluation)==p
    assert p.fingerprint==module._digest(module._state(p))


def test_parse_and_immutable_json_round_trip():
    p=packet(); data=proposal_data(p); original=deepcopy(data)
    proposal=module.parse_hypothesis_proposal(data,p)
    assert data==original
    assert proposal.status==module.HYPOTHESIS_STATUS
    assert proposal.rules[0].side=='LONG' and proposal.rules[0].conditions[1].value==30.
    assert proposal.family_id=='MACHINE_DISCOVERED' and proposal.variant_id=='V1'
    assert module.parse_hypothesis_proposal(json.loads(json.dumps(asdict(proposal))),p)==proposal
    assert module.parse_hypothesis_proposal(proposal,p)==proposal
    with pytest.raises(FrozenInstanceError): proposal.status='VALIDATED'
    with pytest.raises(FrozenInstanceError): proposal.rules[0].side='SHORT'
    with pytest.raises(FrozenInstanceError): proposal.rules[0].conditions[0].value='OTHER'
    with pytest.raises(FrozenInstanceError): p.target_field_id='y.other'


@pytest.mark.parametrize('field',['y.future_return','meta.id','x.not_in_packet'])
def test_forbidden_features(field):
    p=packet(); data=proposal_data(p)
    data['rules'][0]['conditions'][0]['field_id']=field
    data['evidence_feature_ids'][0]=field
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)


def test_omitted_evidence_feature():
    p=packet(); data=proposal_data(p); data['evidence_feature_ids']=list(FEATURES[:1])
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)


@pytest.mark.parametrize('key,value',[('side','BUY'),('side','SELL'),('side',None),('condition_logic','ANY'),
    ('condition_logic','OR'),('stop_pips',0.),('stop_pips',-1.),('stop_pips',float('nan')),('stop_pips',float('inf')),
    ('target_r_multiple',0.),('target_r_multiple',-1.),('target_r_multiple',float('nan')),('target_r_multiple',float('inf')),
    ('stop_pips',True),('stop_pips',10),('rule_id','bad-id')])
def test_execution_negatives(key,value):
    p=packet(); data=proposal_data(p); data['rules'][0][key]=value
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)


@pytest.mark.parametrize('operator,value',[('EQ',{}),('EQ',set()),('EQ',object()),('EQ',float('nan')),
    ('EQ',float('inf')),('EQ',(1,2)),('IN',()),('IN','BULLISH'),('IN',(object(),)),
    ('IN',((1,),)),('NOT_IN',()),('OR',1)])
def test_condition_negatives(operator,value):
    with pytest.raises(ValueError): module.HypothesisCondition(FEATURES[0],operator,value)


@pytest.mark.parametrize('operator',['EQ','NE','GT','GTE','LT','LTE','IN','NOT_IN'])
def test_operator_contract(operator):
    p=packet(); data=proposal_data(p)
    data['rules'][0]['conditions'][0].update(operator=operator,value=['A',False,0,0.,None] if operator in ('IN','NOT_IN') else 'A')
    proposal=module.parse_hypothesis_proposal(data,p)
    assert proposal.rules[0].conditions[0].operator==operator
    if operator in ('IN','NOT_IN'): assert type(proposal.rules[0].conditions[0].value) is tuple


def test_no_executable_payloads_or_extra_keys():
    p=packet()
    for payload in ('close > open and eval(...)',lambda:None,[],None):
        with pytest.raises(TypeError): module.parse_hypothesis_proposal(payload,p)
    for level in ('proposal','rule','condition'):
        data=proposal_data(p)
        target=data if level=='proposal' else data['rules'][0] if level=='rule' else data['rules'][0]['conditions'][0]
        target['python_source']='print(1)'
        with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)
    data=proposal_data(p); del data['rules'][0]['side']
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)


@pytest.mark.parametrize('key,value',[('status','VALIDATED'),('version','OTHER'),('hypothesis_id','lower'),
    ('family_id','bad-id'),('strategy_id',''),('variant_id','bad id'),('statement',' '),('rationale',''),
    ('rules',[]),('evidence_feature_ids',[]),('fingerprint',None),('fingerprint','0'*64),
    ('research_packet_fingerprint','0'*64),('target_field_id','y.other')])
def test_proposal_negatives(key,value):
    p=packet(); data=proposal_data(p); data[key]=value
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)


def test_duplicate_rules_features_and_revalidation():
    p=packet(); data=proposal_data(p); data['rules']*=2
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)
    data=proposal_data(p); data['evidence_feature_ids']*=2
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(data,p)
    proposal=module.parse_hypothesis_proposal(proposal_data(p),p)
    object.__setattr__(proposal.rules[0].conditions[0],'value',{})
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(proposal,p)


def test_packet_binding_changes_and_mismatch():
    clues,evaluation=evidence(); p=module.build_research_packet(clues,evaluation)
    proposal=module.parse_hypothesis_proposal(proposal_data(p),p)
    state=asdict(clues); state['model_sha256']='e'*64; state.pop('fingerprint')
    updated=replace(clues,model_sha256='e'*64,fingerprint=module._digest(state))
    q=module.build_research_packet(updated,evaluation)
    assert p.fingerprint!=q.fingerprint
    with pytest.raises(ValueError): module.parse_hypothesis_proposal(proposal,q)
    state=asdict(evaluation); state.pop('fingerprint'); state['r_squared']=-2.
    other=type(evaluation)(**state,fingerprint=module._digest(state))
    with pytest.raises(ValueError): module.build_research_packet(clues,other)
    for c,e in ((object(),evaluation),(clues,object())):
        with pytest.raises(TypeError): module.build_research_packet(c,e)
    with pytest.raises(TypeError): module.parse_hypothesis_proposal(proposal,object())


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'feature_ids':()}, {'feature_kinds':('NUMERIC',)},
    {'feature_gain_fractions':(float('nan'),.25)},{'gain_order':FEATURES[:1]},
    {'allowed_condition_feature_ids':FEATURES[:1]},{'evaluation_summary':()},
    {'permitted_entry_kinds':('limit',)},{'researcher_instructions':'OTHER'},{'fingerprint':'0'*64}])
def test_packet_self_validation(changes):
    with pytest.raises(ValueError): replace(packet(),**changes)


def test_typed_proposal_fingerprint():
    p=packet(); hashes=[]
    for value in (False,0,0.,'0'):
        data=proposal_data(p); data['rules'][0]['conditions'][0]['value']=value
        hashes.append(module.parse_hypothesis_proposal(data,p).fingerprint)
    assert len(set(hashes))==4
