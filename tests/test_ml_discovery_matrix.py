from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
import hashlib
import json
from math import copysign
import re
import pytest
from test_context_scan import dataset, H4, H1, NUMBER, TARGETS, Once
from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
from sandbox.market_state.discovery_projection import DiscoveryValue
from sandbox.research import ml_discovery_matrix as module

Spec=module.MLFeatureSpec
BLOCK='x.geometry.source_block_index'
COORD='x.geometry.coordinate'


def data():
    source=dataset([('BEARISH',True,.63),('BULLISH',False,.28),(None,True,.41),('FLAT',None,1),('OTHER',True,None)])
    targets=(1,None,-2,None,5)
    return assemble_discovery_dataset(tuple(replace(r,
        predictors=r.predictors+(DiscoveryValue(BLOCK,i),DiscoveryValue(COORD,-3.3)),
        targets=(replace(r.targets[0],value=targets[i]),r.targets[1])) for i,r in enumerate(source.rows)))


def specs():
    return (Spec(H4,'CATEGORICAL'),Spec(NUMBER,'NUMERIC'),Spec(H1,'CATEGORICAL'),Spec(BLOCK,'CATEGORICAL'))


def build(source=None,features=None,rows=None):
    return module.build_ml_discovery_matrix(data() if source is None else source,
        specs() if features is None else features,TARGETS[0],rows)


def test_types_missingness_and_caller_semantics():
    m=build()
    assert m.version=='ML_DISCOVERY_MATRIX_V1'
    assert m.source_row_indices==(0,1,2,3,4)
    assert m.feature_specs==specs()
    assert m.x_rows==(('BEARISH',.63,True,0),('BULLISH',.28,False,1),(None,.41,True,2),('FLAT',1.,None,3),('OTHER',None,True,4))
    assert type(m.x_rows[0][0]) is str
    assert type(m.x_rows[3][1]) is float
    assert type(m.x_rows[0][2]) is bool
    assert type(m.x_rows[0][3]) is int
    assert m.y_values==(1.,None,-2.,None,5.)
    assert all(v is None or type(v) is float for v in m.y_values)
    assert m.target_observed==(True,False,True,False,True)
    assert m.observed_target_row_indices==(0,2,4)
    numeric=build(features=(Spec(BLOCK,'NUMERIC'),))
    assert numeric.x_rows==((0.,),(1.,),(2.,),(3.,),(4.,))
    assert all(type(row[0]) is float for row in numeric.x_rows)
    coordinates=build(features=(Spec(COORD,'CATEGORICAL'),))
    assert coordinates.x_rows==((-3.3,),)*5
    assert type(coordinates.x_rows[0][0]) is float


def test_determinism_feature_order_and_independent_fingerprint():
    m=build(); assert build()==m
    reversed_matrix=build(features=specs()[::-1])
    assert reversed_matrix.feature_specs==m.feature_specs[::-1]
    assert reversed_matrix.x_rows==tuple(row[::-1] for row in m.x_rows)
    assert reversed_matrix.fingerprint!=m.fingerprint
    payload={
        'version':m.version,'source_row_indices':m.source_row_indices,
        'feature_specs':[{'field_id':s.field_id,'feature_kind':s.feature_kind} for s in m.feature_specs],
        'x_rows':tuple(tuple(['missing' if v is None else type(v).__name__,v] if s.feature_kind=='CATEGORICAL' else v
                            for s,v in zip(m.feature_specs,row)) for row in m.x_rows),
        'target_field_id':m.target_field_id,'y_values':m.y_values,'target_observed':m.target_observed,
        'observed_target_row_indices':m.observed_target_row_indices}
    expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
    assert m.fingerprint==expected
    assert re.fullmatch('[0-9a-f]{64}',m.fingerprint)
    assert module.MLDiscoveryMatrix(**{name:getattr(m,name) for name in m.__dataclass_fields__})==m


def test_leakage_isolation():
    source=data(); original=build(source)
    meta=assemble_discovery_dataset(tuple(replace(r,identity=tuple(replace(v,value='ALL_META_CHANGED') for v in r.identity)) for r in source.rows))
    assert build(meta)==original
    other_x=assemble_discovery_dataset(tuple(replace(r,predictors=tuple(replace(v,value=999) if v.field_id==COORD else v for v in r.predictors)) for r in source.rows))
    assert build(other_x)==original
    other_y=assemble_discovery_dataset(tuple(replace(r,targets=(r.targets[0],replace(r.targets[1],value=999))) for r in source.rows))
    assert build(other_y)==original
    selected_y=assemble_discovery_dataset(tuple(replace(r,targets=(replace(r.targets[0],value=999),r.targets[1])) for r in source.rows))
    changed=build(selected_y)
    assert changed.feature_specs==original.feature_specs and changed.x_rows==original.x_rows
    assert changed.y_values!=original.y_values and changed.target_observed!=original.target_observed
    assert changed.fingerprint!=original.fingerprint
    selected_x=assemble_discovery_dataset(tuple(replace(r,predictors=tuple(replace(v,value='NEW') if v.field_id==H4 else v for v in r.predictors)) for r in source.rows))
    changed=build(selected_x)
    assert changed.x_rows!=original.x_rows and changed.y_values==original.y_values
    assert changed.fingerprint!=original.fingerprint


def test_subset_scope_and_original_observed_indices():
    source=data(); m=build(source,rows=(1,3,4))
    assert m.source_row_indices==(1,3,4)
    assert m.x_rows==tuple(build(source).x_rows[i] for i in (1,3,4))
    assert m.y_values==(None,None,5.) and m.observed_target_row_indices==(4,)
    changed=assemble_discovery_dataset(tuple(replace(r,
        predictors=tuple(replace(v,value=float('nan')) for v in r.predictors),
        targets=tuple(replace(v,value=float('inf')) for v in r.targets)) if i in (0,2) else r for i,r in enumerate(source.rows)))
    assert build(changed,rows=(1,3,4))==m
    assert build(rows=(0,1,2,3)).y_values==(1.,None,-2.,None)
    assert build(rows=(0,1,2,3)).observed_target_row_indices==(0,2)
    missing=build(rows=(1,3))
    assert missing.target_observed==(False,False) and missing.observed_target_row_indices==()
    assert len(missing.x_rows)==2


def test_materialization_columns_and_authoritative_roles(monkeypatch):
    read,roles=[],[]
    column,role=module.column_values,module.field_role
    def get_column(data,field):
        read.append(field); return column(data,field)
    def get_role(data,field):
        roles.append(field); return role(data,field)
    monkeypatch.setattr(module,'column_values',get_column)
    monkeypatch.setattr(module,'field_role',get_role)
    features,rows=Once(specs()),Once((1,3,4))
    m=build(features=features,rows=rows)
    assert features.calls==rows.calls==1
    expected=[s.field_id for s in specs()]+[TARGETS[0]]
    assert read==expected and roles==expected
    assert m.source_row_indices==(1,3,4)
    monkeypatch.setattr(module,'field_role',lambda data,field:'target')
    with pytest.raises(ValueError): build()


@pytest.mark.parametrize('value',[False,0,0.,'0',None])
def test_categorical_type_tags(value):
    source=dataset([(value,'A',1)])
    m=build(source,features=(Spec(H4,'CATEGORICAL'),))
    assert type(m.x_rows[0][0]) is type(value)
    fingerprints={build(dataset([(v,'A',1)]),features=(Spec(H4,'CATEGORICAL'),)).fingerprint for v in (False,0,0.,'0',None)}
    assert len(fingerprints)==5


def test_negative_zero_and_numeric_equivalence():
    source=dataset([(-0.,'A',-0.)])
    source=assemble_discovery_dataset((replace(source.rows[0],targets=(replace(source.rows[0].targets[0],value=-0.),source.rows[0].targets[1])),))
    m=build(source,features=(Spec(H4,'CATEGORICAL'),Spec(NUMBER,'NUMERIC')))
    assert all(copysign(1,v)==1 for v in m.x_rows[0]+m.y_values)
    positive=dataset([(0.,'A',0)])
    assert build(positive,features=m.feature_specs)==m
    assert build(dataset([('A','B',1)]),features=(Spec(NUMBER,'NUMERIC'),))==build(dataset([('A','B',1.)]),features=(Spec(NUMBER,'NUMERIC'),))
    assert build(dataset([('é雪','B',1)]),features=(Spec(H4,'CATEGORICAL'),)).x_rows==(('é雪',),)


@pytest.mark.parametrize('field,kind',[('', 'NUMERIC'),('meta.id','NUMERIC'),(TARGETS[0],'CATEGORICAL'),
    (1,'NUMERIC'),(H4,'numeric'),(H4,'OTHER'),(H4,None),(H4,True)])
def test_invalid_feature_spec(field,kind):
    with pytest.raises(ValueError): Spec(field,kind)


@pytest.mark.parametrize('features',[(),(object(),),(Spec(H4,'CATEGORICAL'),)*2])
def test_invalid_features(features):
    with pytest.raises(ValueError): build(features=features)


@pytest.mark.parametrize('field,error',[(H4,ValueError),('meta.id',ValueError),('y.unknown',KeyError),(1,TypeError),((TARGETS[0],TARGETS[1]),TypeError)])
def test_invalid_target(field,error):
    with pytest.raises(error): module.build_ml_discovery_matrix(data(),specs(),field)


def test_unknown_feature_and_invalid_dataset():
    with pytest.raises(KeyError): build(features=(Spec('x.unknown','NUMERIC'),))
    with pytest.raises(TypeError): build(object())
    with pytest.raises(ValueError): build(assemble_discovery_dataset(()))


@pytest.mark.parametrize('rows',[(),(True,),(-1,),(5,),(1,0),(0,0),(0.,),('0',)])
def test_invalid_row_selection(rows):
    with pytest.raises(ValueError): build(rows=rows)


@pytest.mark.parametrize('kind,value',[
    (kind,value) for kind in ('NUMERIC','TARGET') for value in
    (True,'1',date(2024,1,1),datetime(2024,1,1),float('nan'),float('inf'),-float('inf'),object(),10**400)
]+[('CATEGORICAL',value) for value in (date(2024,1,1),datetime(2024,1,1),float('nan'),float('inf'),-float('inf'),object())])
def test_invalid_selected_scalars(kind,value,monkeypatch):
    column=module.column_values
    selected=TARGETS[0] if kind=='TARGET' else H4
    def read(data,field):
        return (value,)*len(data.rows) if field==selected else column(data,field)
    monkeypatch.setattr(module,'column_values',read)
    features=(Spec(H4,'CATEGORICAL' if kind=='TARGET' else kind),)
    with pytest.raises(ValueError): build(features=features)


@pytest.mark.parametrize('changes',[
    {'version':'OTHER'},{'source_row_indices':[]},{'source_row_indices':()},
    {'source_row_indices':(True,1,2,3,4)},{'source_row_indices':(-1,1,2,3,4)},
    {'source_row_indices':(1,0,2,3,4)},{'source_row_indices':(0,0,2,3,4)},
    {'feature_specs':[]},{'feature_specs':()},{'feature_specs':(object(),)},
    {'feature_specs':(Spec(H4,'CATEGORICAL'),)*4},
    {'x_rows':[]},{'x_rows':()},{'x_rows':(('A',),)*5},{'x_rows':(['A',1.,True,0],)*5},
    {'target_field_id':H4},{'target_field_id':''},{'y_values':[]},{'y_values':(1.,)},
    {'y_values':(1,None,-2.,None,5.)},{'y_values':(True,None,-2.,None,5.)},
    {'y_values':('1',None,-2.,None,5.)},{'y_values':(float('inf'),None,-2.,None,5.)},
    {'target_observed':[]},{'target_observed':(True,)},{'target_observed':(1,False,True,False,True)},
    {'target_observed':(False,False,True,False,True)},
    {'observed_target_row_indices':[]},{'observed_target_row_indices':(0,2)},
    {'observed_target_row_indices':(False,2,4)},{'observed_target_row_indices':(4,2,0)}])
def test_matrix_structural_validation(changes,monkeypatch):
    m=build()
    # Isolate structural checks so a stale hash cannot hide a missing invariant.
    monkeypatch.setattr(module,'_fingerprint',lambda *args:m.fingerprint)
    with pytest.raises(ValueError): replace(m,**changes)


@pytest.mark.parametrize('kind,value',[
    ('NUMERIC',v) for v in (1,True,'1',float('nan'),float('inf'),date(2024,1,1),object())
]+[('CATEGORICAL',v) for v in (date(2024,1,1),datetime(2024,1,1),float('nan'),-float('inf'),object())])
def test_stored_cell_validation(kind,value,monkeypatch):
    m=build(features=(Spec(NUMBER,kind),))
    monkeypatch.setattr(module,'_fingerprint',lambda *args:m.fingerprint)
    with pytest.raises(ValueError): replace(m,x_rows=((value,),)+m.x_rows[1:])


@pytest.mark.parametrize('fingerprint',[None,1,'','a'*63,'a'*65,'A'*64,'g'*64,'0'*64])
def test_invalid_fingerprint(fingerprint):
    with pytest.raises(ValueError): replace(build(),fingerprint=fingerprint)


def test_content_fingerprint_and_frozen():
    m=build()
    for changes in ({'x_rows':(('NEW',.63,True,0),)+m.x_rows[1:]},
                    {'y_values':(2.,None,-2.,None,5.)},
                    {'target_field_id':TARGETS[1]},
                    {'source_row_indices':(5,6,7,8,9),'observed_target_row_indices':(5,7,9)}):
        with pytest.raises(ValueError): replace(m,**changes)
    with pytest.raises(FrozenInstanceError): m.fingerprint='0'*64
    with pytest.raises(FrozenInstanceError): m.feature_specs[0].feature_kind='NUMERIC'
    with pytest.raises(TypeError): m.x_rows[0][0]='OTHER'
