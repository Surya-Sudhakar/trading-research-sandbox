from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone, timedelta
import pytest
from test_ai_researcher import packet, proposal_data, FEATURES
from test_discovery_dataset import record
from sandbox.market_state.discovery_dataset import DiscoveryDataset, assemble_discovery_dataset
from sandbox.market_state.discovery_projection import DiscoveryRow, DiscoveryValue
from sandbox.research.ai_researcher import parse_hypothesis_proposal
from sandbox.research.hypothesis_compiler import compile_hypothesis, CompiledHypothesis
from sandbox.research import compiled_strategy_execution as module
from sandbox.strategies.dsl.conditions import OrCondition, NotCondition, ValueReference
from sandbox.strategies.dsl.execution import LimitEntry, ReferenceStop, ReferenceTarget
from sandbox.execution.models import Direction, EntryType


def dataset(values=(('BULLISH',20.),),reference=1.1):
    source=record()
    # Only the causal anchor reference is consumed by this executor.
    object.__setattr__(source.anchor.outcome_anchor,'reference_price',reference)
    return assemble_discovery_dataset(tuple(DiscoveryRow(source,(),
        tuple(DiscoveryValue(f,v) for f,v in zip(FEATURES,values)),
        (DiscoveryValue('y.unused',999.),)) for values in values))


def artifacts(operator=None,value=None,field=1,side='LONG',two_rules=False):
    p=packet(); data=proposal_data(p)
    data['rules'][0]['side']=side
    if operator is not None:
        data['rules'][0]['conditions']=[dict(field_id=FEATURES[field],operator=operator,value=value)]
    if two_rules:
        data['rules'][0]['rule_id']='Z_FIRST'
        data['rules'].append({**deepcopy(data['rules'][0]),'rule_id':'A_SECOND'})
    return compile_hypothesis(parse_hypothesis_proposal(data,p),p),p


def execute(data=None,parts=None,**kwargs):
    c,p=artifacts() if parts is None else parts
    return module.build_compiled_strategy_intents(c,p,dataset() if data is None else data,
        **dict(dataset_id='DISCOVERY_EVENTS',symbol='EURUSD',pip_size=.0001,**kwargs))


def test_vertical_slice_geometry_identity_and_order():
    source=dataset((('BULLISH',20.),('BULLISH',30.),('BEARISH',20.),('BULLISH',None)))
    c,p=artifacts(); batch=execute(source,(c,p))
    assert batch.source_row_count==4 and batch.matched_row_count==batch.intent_count==1
    assert batch.matched_source_row_indices==(0,)
    i=batch.intents[0]
    assert i.strategy_id==c.strategy_spec.strategy_id and i.experiment_id==c.hypothesis_id
    assert i.dataset_id=='DISCOVERY_EVENTS' and i.symbol=='EURUSD'
    assert i.signal_timestamp==source.rows[0].record.anchor.evidence_end_utc
    assert i.requested_entry_type==EntryType.MARKET_CLOSE and i.requested_entry_price==1.1
    assert i.direction==Direction.LONG
    assert i.stop_loss==pytest.approx(1.099) and i.take_profit==pytest.approx(1.102)
    assert i.metadata['geometry_family_id']==source.rows[0].record.anchor.geometry_family_id
    assert i.metadata['geometry_level_id']==source.rows[0].record.anchor.level_id
    assert set(i.metadata)==module._METADATA_KEYS
    assert execute(source,(c,p))==batch
    assert batch.fingerprint==module._digest(module._payload(batch))


def test_short_geometry_multiple_rules_and_duplicate_events():
    parts=artifacts(side='SHORT',two_rules=True); source=dataset()
    source=assemble_discovery_dataset((source.rows[0],source.rows[0]))
    result=execute(source,parts)
    assert result.matched_source_row_indices==(0,1) and result.intent_count==4
    assert [i.metadata['rule_id'] for i in result.intents]==['Z_FIRST','A_SECOND']*2
    assert [i.metadata['source_row_index'] for i in result.intents]==[0,0,1,1]
    assert len({i.trade_intent_id for i in result.intents})==4
    for i in result.intents:
        assert i.direction==Direction.SHORT
        assert i.stop_loss==pytest.approx(1.101) and i.take_profit==pytest.approx(1.098)


@pytest.mark.parametrize('operator,value,expected',[
    ('LT',30.,(0,)),('LTE',30.,(0,1)),('GT',30.,(2,)),('GTE',30.,(1,2)),
    ('EQ',20.,(0,)),('NE',20.,(1,2,3)),('EQ',None,(3,)),('NE',None,(0,1,2)),
    ('IN',[20.,None],(0,3)),('NOT_IN',[20.,None],(1,2))])
def test_numeric_operators(operator,value,expected):
    result=execute(dataset(tuple(('BULLISH',v) for v in (20,30.,40.,None))),artifacts(operator,value))
    assert result.matched_source_row_indices==expected


@pytest.mark.parametrize('operator,value,expected',[
    ('EQ',False,(0,)),('NE',False,(1,2,3,4)),('IN',[False,'False'],(0,3)),
    ('NOT_IN',[False,'False'],(1,2,4)),('EQ',None,(4,))])
def test_typed_categorical(operator,value,expected):
    result=execute(dataset(tuple((v,20.) for v in (False,0,0.,'False',None))),artifacts(operator,value,field=0))
    assert result.matched_source_row_indices==expected


def test_numeric_equality_and_empty_result():
    assert execute(dataset((('BULLISH',2),)),artifacts('EQ',2.)).matched_source_row_indices==(0,)
    result=execute(dataset((('BEARISH',20.),)))
    assert result.matched_row_count==result.intent_count==0 and result.intents==()


def test_target_guard_and_mutation(monkeypatch):
    source=dataset(); expected=execute(source)
    changed=assemble_discovery_dataset(tuple(replace(r,targets=tuple(replace(v,value=-1e200) for v in r.targets)) for r in source.rows))
    assert execute(changed)==expected
    original=DiscoveryRow.__getattribute__; dataset_get=DiscoveryDataset.__getattribute__
    def guarded(self,name):
        if name=='targets': raise AssertionError('read targets')
        return original(self,name)
    def dataset_guard(self,name):
        if name=='target_field_ids': raise AssertionError('read target schema')
        return dataset_get(self,name)
    monkeypatch.setattr(DiscoveryRow,'__getattribute__',guarded)
    monkeypatch.setattr(DiscoveryDataset,'__getattribute__',dataset_guard)
    assert execute(source)==expected


def alter_compiled(compiled,**rule_changes):
    spec=compiled.strategy_spec.model_copy(update={'rules':(compiled.strategy_spec.rules[0].model_copy(update=rule_changes),)})
    state={name:getattr(compiled,name) for name in compiled.__dataclass_fields__ if name!='fingerprint'}
    state.update(strategy_spec=spec,strategy_spec_fingerprint=module._digest(spec.model_dump(mode='json')))
    return CompiledHypothesis(**state,fingerprint=module._digest({**state,'strategy_spec':spec.model_dump(mode='json')}))


@pytest.mark.parametrize('kind',['or','not','entry','stop','target','unknown','namespace'])
def test_unsupported_dsl(kind):
    c,p=artifacts(); rule=c.strategy_spec.rules[0]; condition=rule.condition.children[0]
    changes={
        'or':{'condition':OrCondition(children=(condition,))},
        'not':{'condition':NotCondition(children=(condition,))},
        'entry':{'entry':LimitEntry(side='LONG',price_reference=ValueReference(namespace='feature',key=FEATURES[0]))},
        'stop':{'stop':ReferenceStop(reference=ValueReference(namespace='feature',key=FEATURES[0]))},
        'target':{'target':ReferenceTarget(reference=ValueReference(namespace='feature',key=FEATURES[0]))},
        'unknown':{'condition':condition.model_copy(update={'reference':ValueReference(namespace='feature',key='x.unknown')})},
        'namespace':{'condition':condition.model_copy(update={'reference':ValueReference(namespace='event',key=FEATURES[0])})}}
    with pytest.raises(ValueError): execute(parts=(alter_compiled(c,**changes[kind]),p))


@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf'),-float('inf'),10**400])
def test_invalid_pip_size(value):
    c,p=artifacts()
    with pytest.raises(ValueError): module.build_compiled_strategy_intents(c,p,dataset(),dataset_id='D',symbol='S',pip_size=value)


@pytest.mark.parametrize('value',[True,'20',float('nan'),float('inf'),-float('inf')])
def test_invalid_numeric_values(value):
    with pytest.raises(ValueError): execute(dataset((('BULLISH',value),)))


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_reference(value):
    with pytest.raises(ValueError): execute(dataset(reference=value))


def test_nonpositive_geometry_and_precision_loss():
    with pytest.raises(ValueError): execute(dataset(reference=.00001))
    with pytest.raises(ValueError): execute(dataset(reference=1e100))


@pytest.mark.parametrize('timestamp',[datetime(2024,1,1),datetime(2024,1,1,tzinfo=timezone(timedelta(hours=1))),None])
def test_invalid_anchor_timestamp(timestamp):
    source=dataset(); object.__setattr__(source.rows[0].record.anchor,'evidence_end_utc',timestamp)
    with pytest.raises(ValueError): execute(source)


def test_missing_feature_and_duplicate_predictors():
    source=dataset(); row=source.rows[0]
    missing=assemble_discovery_dataset((replace(row,predictors=row.predictors[:1]),))
    with pytest.raises(ValueError): execute(missing)
    object.__setattr__(row,'predictors',row.predictors+(row.predictors[0],))
    with pytest.raises(ValueError): execute(source)


def test_types_and_binding():
    c,p=artifacts(); source=dataset()
    for args in ((object(),p,source),(c,object(),source),(c,p,object())):
        with pytest.raises(TypeError): module.build_compiled_strategy_intents(*args,dataset_id='D',symbol='S',pip_size=.01)
    for name,value in (('research_packet_fingerprint','0'*64),('target_field_id','y.other')):
        state={k:getattr(c,k) for k in c.__dataclass_fields__ if k!='fingerprint'}; state[name]=value
        changed=CompiledHypothesis(**state,fingerprint=module._digest({**state,'strategy_spec':c.strategy_spec.model_dump(mode='json')}))
        with pytest.raises(ValueError): execute(source,(changed,p))
    for dataset_id,symbol in (('', 'S'),('D',' ')):
        with pytest.raises(ValueError): module.build_compiled_strategy_intents(c,p,source,dataset_id=dataset_id,symbol=symbol,pip_size=.01)


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'fingerprint':'0'*64},{'strategy_spec_fingerprint':'bad'},
    {'source_row_count':True},{'matched_row_count':True},{'intent_count':2}, {'source_row_count':0},
    {'matched_source_row_indices':[]},{'matched_source_row_indices':(True,)},{'matched_source_row_indices':(-1,)},
    {'matched_source_row_indices':(1,)},{'intents':[]},{'strategy_id':'OTHER'},{'hypothesis_id':'OTHER'},
    {'dataset_id':'OTHER'},{'symbol':'OTHER'},{'pip_size':True},{'pip_size':0.}])
def test_batch_validation(changes):
    with pytest.raises(ValueError): replace(execute(),**changes)


def test_intent_tampering_and_frozen():
    b=execute(); i=b.intents[0]
    for changed in (i.model_copy(update={'stop_loss':0.}),i.model_copy(update={'trade_intent_id':'OTHER'}),
        i.model_copy(update={'metadata':{**i.metadata,'future_value':999}})):
        with pytest.raises(ValueError): replace(b,intents=(changed,))
    with pytest.raises(FrozenInstanceError): b.intent_count=0
