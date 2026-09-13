from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
import hashlib
import json
from math import copysign
import pytest
from sandbox.research.ml_discovery_matrix import MLDiscoveryMatrix, MLFeatureSpec
from sandbox.research import ml_preprocessing as module

NUM='x.body.ratio'
CAT='x.h4.direction'
TARGET='y.h4.close_change'
SCHEMA=(MLFeatureSpec(NUM,'NUMERIC'),MLFeatureSpec(CAT,'CATEGORICAL'))


def matrix(rows=((1.25,'BEARISH'),(None,'BULLISH'),(-4.,None)), y=None, indices=None, specs=SCHEMA):
    rows=tuple(rows)
    indices=tuple(range(len(rows))) if indices is None else tuple(indices)
    y=tuple(float(i) for i in range(len(rows))) if y is None else tuple(y)
    mask=tuple(v is not None for v in y)
    state=dict(version='ML_DISCOVERY_MATRIX_V1',source_row_indices=indices,feature_specs=specs,x_rows=rows,
        target_field_id=TARGET,y_values=y,target_observed=mask,
        observed_target_row_indices=tuple(i for i,observed in zip(indices,mask) if observed))
    payload=dict(state)
    payload['feature_specs']=[{'field_id':s.field_id,'feature_kind':s.feature_kind} for s in specs]
    payload['x_rows']=[[['missing' if v is None else type(v).__name__,v] if s.feature_kind=='CATEGORICAL' else v
        for s,v in zip(specs,row)] for row in rows]
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return MLDiscoveryMatrix(**state,fingerprint=fingerprint)


def fitted():
    return module.fit_ml_preprocessor(matrix())


def encoded():
    m=matrix(y=(1.,None,-2.),indices=(4,7,10))
    return module.transform_ml_matrix(m,module.fit_ml_preprocessor(m))


def test_fit_transform_unknown_and_numeric_missing():
    discovery=matrix(); pre=module.fit_ml_preprocessor(discovery)
    assert pre.version=='ML_PREPROCESSOR_V1'
    assert pre.feature_specs==SCHEMA
    assert pre.fitted_from_matrix_fingerprint==discovery.fingerprint
    assert pre.missing_category_code==0 and pre.unknown_category_code==1
    assert pre.categorical_encodings==(module.CategoricalFeatureEncoding(CAT,('BEARISH','BULLISH')),)
    transformed=module.transform_ml_matrix(discovery,pre)
    assert transformed.x_rows==((1.25,2),(None,3),(-4.,0))
    assert transformed.categorical_feature_indices==(1,)
    assert transformed.source_row_indices==discovery.source_row_indices
    assert transformed.y_values==discovery.y_values
    assert transformed.target_observed==discovery.target_observed
    assert transformed.observed_target_row_indices==discovery.observed_target_row_indices
    assert transformed.source_matrix_fingerprint==discovery.fingerprint
    assert transformed.preprocessor_fingerprint==pre.fingerprint
    validation=matrix(((1.,'BEARISH'),(None,'SIDEWAYS'),(-4.,None)))
    result=module.transform_ml_matrix(validation,pre)
    assert result.x_rows==((1.,2),(None,1),(-4.,0))
    assert pre==module.fit_ml_preprocessor(discovery)
    assert module.transform_ml_matrix(discovery,pre)==transformed


def test_typed_order_zero_and_row_reordering():
    values=('0',0.,0,False,True,-1,2.5,-2.5,'雪','A')
    m=matrix(tuple((1.,v) for v in values))
    pre=module.fit_ml_preprocessor(m)
    expected=(False,True,-1,0,-2.5,0.,2.5,'0','A','雪')
    assert [(type(v),v) for v in pre.categorical_encodings[0].categories]==[(type(v),v) for v in expected]
    output=module.transform_ml_matrix(m,pre)
    codes=[r[1] for r in output.x_rows]
    assert len(set(codes))==10 and min(codes)==2
    assert len(set(codes[:4]))==4
    reversed_pre=module.fit_ml_preprocessor(matrix(m.x_rows[::-1]))
    assert reversed_pre.categorical_encodings==pre.categorical_encodings
    assert reversed_pre.feature_specs==pre.feature_specs
    assert reversed_pre.fitted_from_matrix_fingerprint!=pre.fitted_from_matrix_fingerprint
    negative=module.CategoricalFeatureEncoding(CAT,(-0.,))
    assert copysign(1,negative.categories[0])==1
    float_only=module.fit_ml_preprocessor(matrix(((1.,0.),)))
    other=module.transform_ml_matrix(matrix(((1.,False),(1.,0),(1.,'0'))),float_only)
    assert [r[1] for r in other.x_rows]==[1,1,1]


def test_all_missing_categories_and_numeric_only():
    m=matrix(((1.,None),(None,None)))
    pre=module.fit_ml_preprocessor(m)
    assert pre.categorical_encodings[0].categories==()
    assert module.transform_ml_matrix(m,pre).x_rows==((1.,0),(None,0))
    assert module.transform_ml_matrix(matrix(((2.,'NEW'),)),pre).x_rows==((2.,1),)
    numeric=matrix(((1.25,),(None,),(-4.,)),specs=(SCHEMA[0],))
    p=module.fit_ml_preprocessor(numeric)
    assert p.categorical_encodings==()
    e=module.transform_ml_matrix(numeric,p)
    assert e.categorical_feature_indices==() and e.x_rows==numeric.x_rows


def test_fit_does_not_read_target_attributes(monkeypatch):
    m=matrix(); original=MLDiscoveryMatrix.__getattribute__
    def guarded(self,name):
        if name in ('y_values','target_observed','observed_target_row_indices','target_field_id'):
            raise AssertionError('fit read target content')
        return original(self,name)
    monkeypatch.setattr(MLDiscoveryMatrix,'__getattribute__',guarded)
    assert module.fit_ml_preprocessor(m).categorical_encodings[0].categories==('BEARISH','BULLISH')


def test_leakage_feature_state_and_trace():
    m=matrix(); p=module.fit_ml_preprocessor(m)
    target_changed=matrix(m.x_rows,y=(None,100.,200.))
    q=module.fit_ml_preprocessor(target_changed)
    assert q.feature_specs==p.feature_specs and q.categorical_encodings==p.categorical_encodings
    assert (q.missing_category_code,q.unknown_category_code)==(p.missing_category_code,p.unknown_category_code)
    assert q.fitted_from_matrix_fingerprint==target_changed.fingerprint
    assert q.fingerprint!=p.fingerprint
    numeric_changed=matrix(((99.,'BEARISH'),(None,'BULLISH'),(-4.,None)))
    assert module.fit_ml_preprocessor(numeric_changed).categorical_encodings==p.categorical_encodings
    e=module.transform_ml_matrix(numeric_changed,p)
    original=module.transform_ml_matrix(m,p)
    assert [r[1] for r in e.x_rows]==[r[1] for r in original.x_rows]
    assert e.x_rows[0][0]==99. and e.y_values==original.y_values
    assert module.fit_ml_preprocessor(matrix(((1.,'NEW'),))).categorical_encodings!=p.categorical_encodings


def test_upstream_unselected_information():
    from test_ml_discovery_matrix import data, build, COORD
    from sandbox.market_state.discovery_dataset import assemble_discovery_dataset
    source=data(); m=build(source)
    changed=assemble_discovery_dataset(tuple(replace(r,identity=tuple(replace(v,value='CHANGED') for v in r.identity),
        predictors=tuple(replace(v,value=999) if v.field_id==COORD else v for v in r.predictors),
        targets=(r.targets[0],replace(r.targets[1],value=999))) for r in source.rows))
    n=build(changed)
    assert m==n
    p=module.fit_ml_preprocessor(m); q=module.fit_ml_preprocessor(n)
    assert p==q
    assert module.transform_ml_matrix(m,p)==module.transform_ml_matrix(n,q)


def test_schema_mismatch_and_types():
    pre=fitted()
    for m in (matrix(((1.,),),specs=(SCHEMA[0],)),
              matrix((('A',1.),),specs=SCHEMA[::-1]),
              matrix(((1.,1.),),specs=(SCHEMA[0],MLFeatureSpec(CAT,'NUMERIC')))):
        with pytest.raises(ValueError): module.transform_ml_matrix(m,pre)
    with pytest.raises(TypeError): module.fit_ml_preprocessor(object())
    with pytest.raises(TypeError): module.transform_ml_matrix(object(),pre)
    with pytest.raises(TypeError): module.transform_ml_matrix(matrix(),object())


@pytest.mark.parametrize('categories', [[],(None,),(1,1),(False,False),(0.,-0.),('B','A'),(0,False),
    (date(2024,1,1),),(datetime(2024,1,1),),(object(),),(float('nan'),),(float('inf'),),(-float('inf'),)])
def test_invalid_categories(categories):
    with pytest.raises(ValueError): module.CategoricalFeatureEncoding(CAT,categories)


@pytest.mark.parametrize('field',['','meta.id','y.target',None])
def test_invalid_category_field(field):
    with pytest.raises(ValueError): module.CategoricalFeatureEncoding(field,())


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'fitted_from_matrix_fingerprint':'bad'},
    {'fitted_from_matrix_fingerprint':'0'*64},{'feature_specs':()},{'feature_specs':[]},
    {'categorical_encodings':()},{'categorical_encodings':[]},
    {'categorical_encodings':(module.CategoricalFeatureEncoding(NUM,()),)},
    {'categorical_encodings':(module.CategoricalFeatureEncoding(CAT,()),)*2},
    {'missing_category_code':False},{'missing_category_code':1},{'unknown_category_code':True},
    {'unknown_category_code':2},{'fingerprint':'A'*64},{'fingerprint':'0'*64}])
def test_preprocessor_validation(changes):
    with pytest.raises(ValueError): replace(fitted(),**changes)


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'source_matrix_fingerprint':'bad'},
    {'source_matrix_fingerprint':'0'*64},{'preprocessor_fingerprint':'0'*64},
    {'source_row_indices':[]},{'source_row_indices':()},{'source_row_indices':(True,7,10)},
    {'source_row_indices':(-1,7,10)},{'source_row_indices':(7,4,10)},
    {'feature_specs':()},{'categorical_feature_indices':(True,)},{'categorical_feature_indices':()},
    {'categorical_feature_indices':(0,)},{'x_rows':[]},{'x_rows':((1.,2),)},
    {'x_rows':((1.,),)*3},{'x_rows':([1.,2],)*3},
    {'x_rows':((True,2),(None,3),(-4.,0))},{'x_rows':((1,2),(None,3),(-4.,0))},
    {'x_rows':((float('nan'),2),(None,3),(-4.,0))},
    {'x_rows':((1.,True),(None,3),(-4.,0))},{'x_rows':((1.,None),(None,3),(-4.,0))},
    {'x_rows':((1.,-1),(None,3),(-4.,0))},{'x_rows':((1.,2.),(None,3),(-4.,0))},
    {'target_field_id':'x.other'},{'y_values':(1,None,-2.)},{'y_values':(True,None,-2.)},
    {'y_values':(float('inf'),None,-2.)},{'target_observed':(True,)},
    {'target_observed':(1,False,True)},{'target_observed':(False,False,True)},
    {'observed_target_row_indices':(0,2)},{'observed_target_row_indices':(True,10)},
    {'fingerprint':'bad'},{'fingerprint':'0'*64}])
def test_encoded_validation(changes):
    with pytest.raises(ValueError): replace(encoded(),**changes)


def test_fingerprints_independent_and_immutable():
    p=fitted(); e=encoded()
    payload={name:getattr(p,name) for name in p.__dataclass_fields__ if name!='fingerprint'}
    payload['feature_specs']=[{'field_id':s.field_id,'feature_kind':s.feature_kind} for s in p.feature_specs]
    payload['categorical_encodings']=[{'field_id':v.field_id,'categories':[[type(c).__name__,c] for c in v.categories]} for v in p.categorical_encodings]
    assert p.fingerprint==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    payload={name:getattr(e,name) for name in e.__dataclass_fields__ if name!='fingerprint'}
    payload['feature_specs']=[{'field_id':s.field_id,'feature_kind':s.feature_kind} for s in e.feature_specs]
    assert e.fingerprint==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with pytest.raises(FrozenInstanceError): p.missing_category_code=8
    with pytest.raises(FrozenInstanceError): p.categorical_encodings[0].categories=()
    with pytest.raises(FrozenInstanceError): e.x_rows=()
