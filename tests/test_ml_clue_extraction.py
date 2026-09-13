from dataclasses import asdict, FrozenInstanceError, replace
import hashlib
import json
from math import copysign, isfinite
import pytest
from test_ml_lightgbm import matrix
from sandbox.research.ml_discovery_matrix import MLFeatureSpec
from sandbox.research.ml_preprocessing import fit_ml_preprocessor, transform_ml_matrix
from sandbox.research.ml_lightgbm import train_lightgbm_clue_model, predict_lightgbm_clue_model, LightGBMClueConfig
from sandbox.research.ml_evaluation import evaluate_prediction_batch, MLPredictionEvaluation
from sandbox.research import ml_clue_extraction as module


def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@pytest.fixture(scope='module')
def artifacts():
    specs=tuple(MLFeatureSpec(f'x.{name}','NUMERIC') for name in ('a','b','c'))
    raw=matrix(tuple((float(i%10),float(i%3),1.) for i in range(30)),
        y=tuple(float(i%10)*3 for i in range(30)),specs=specs)
    pre=fit_ml_preprocessor(raw); encoded=transform_ml_matrix(raw,pre)
    model=train_lightgbm_clue_model(encoded,LightGBMClueConfig(num_boost_round=4,min_data_in_leaf=2))
    prediction=predict_lightgbm_clue_model(model,encoded)
    evaluation=evaluate_prediction_batch(encoded,prediction)
    return model,evaluation


def fake(monkeypatch,splits=(10,5,0),gains=(100.,50.,0.),names=('x.a','x.b','x.c'),count=3):
    calls=[]
    class Booster:
        def __init__(self,**kwargs): calls.append(('init',kwargs))
        def num_feature(self): return count
        def feature_name(self): return names
        def current_iteration(self): return 9
        def feature_importance(self,**kwargs):
            calls.append(('importance',kwargs))
            return splits if kwargs['importance_type']=='split' else gains
    monkeypatch.setattr(module.lightgbm,'Booster',Booster)
    return calls


def report(artifacts,monkeypatch,**kwargs):
    fake(monkeypatch,**kwargs)
    return module.extract_ml_clues(*artifacts)


def test_exact_mapping(artifacts,monkeypatch):
    calls=fake(monkeypatch)
    result=module.extract_ml_clues(*artifacts)
    assert result.feature_count==3 and result.total_split_count==15 and result.total_gain==150.
    assert [c.field_id for c in result.feature_clues]==['x.a','x.b','x.c']
    assert [c.feature_index for c in result.feature_clues]==[0,1,2]
    assert [c.split_count for c in result.feature_clues]==[10,5,0]
    assert [c.gain for c in result.feature_clues]==[100.,50.,0.]
    assert [c.gain_fraction for c in result.feature_clues]==pytest.approx([2/3,1/3,0])
    assert [c.split_fraction for c in result.feature_clues]==pytest.approx([2/3,1/3,0])
    assert result.gain_order==result.split_order==('x.a','x.b','x.c')
    assert calls==[('init',{'model_str':artifacts[0].model_text}),
        ('importance',{'importance_type':'split','iteration':9}),('importance',{'importance_type':'gain','iteration':9})]
    assert all(type(c.split_count) is int and type(c.gain) is float for c in result.feature_clues)


def test_ties_and_secondary_order(artifacts,monkeypatch):
    r=report(artifacts,monkeypatch,splits=(1,3,2),gains=(10.,10.,20.))
    assert r.gain_order==('x.c','x.b','x.a')
    assert r.split_order==('x.b','x.c','x.a')
    r=report(artifacts,monkeypatch,splits=(2,2,2),gains=(5.,5.,5.))
    assert r.gain_order==r.split_order==('x.a','x.b','x.c')
    r=report(artifacts,monkeypatch,splits=(2,2,2),gains=(1.,3.,2.))
    assert r.split_order==('x.b','x.c','x.a')
    assert tuple(c.field_id for c in r.feature_clues)==('x.a','x.b','x.c')


def test_zero_evidence(artifacts,monkeypatch):
    r=report(artifacts,monkeypatch,splits=(0,0,0),gains=(-0.,0.,0.))
    assert r.total_gain==0. and r.total_split_count==0 and len(r.feature_clues)==3
    for c in r.feature_clues:
        assert c.gain_fraction==c.split_fraction==c.gain==0.
        assert copysign(1,c.gain)==copysign(1,c.gain_fraction)==1.
    assert r.gain_order==r.split_order==('x.a','x.b','x.c')


def changed_evaluation(evaluation,**changes):
    state=asdict(evaluation); state.pop('fingerprint'); state.update(changes)
    return MLPredictionEvaluation(**state,fingerprint=digest(state))


@pytest.mark.parametrize('changes',[{'model_fingerprint':'b'*64},{'target_field_id':'y.other'}])
def test_mismatched_evaluation(artifacts,changes):
    model,evaluation=artifacts
    with pytest.raises(ValueError): module.extract_ml_clues(model,changed_evaluation(evaluation,**changes))


def test_other_matrix_and_no_metric_filtering(artifacts,monkeypatch):
    model,evaluation=artifacts
    altered=changed_evaluation(evaluation,encoded_matrix_fingerprint='b'*64,r_squared=-1000.,
        mean_absolute_error=10000.,root_mean_squared_error=10000.,mean_squared_error=100000000.)
    fake(monkeypatch)
    original=MLPredictionEvaluation.__getattribute__
    def guarded(self,name):
        if name not in ('model_fingerprint','target_field_id','fingerprint','__class__'):
            raise AssertionError('extraction inspected evaluation metrics')
        return original(self,name)
    monkeypatch.setattr(MLPredictionEvaluation,'__getattribute__',guarded)
    r=module.extract_ml_clues(model,altered)
    assert r.evaluation_fingerprint==altered.fingerprint


@pytest.mark.parametrize('kwargs',[{'count':2},{'names':('x.b','x.a','x.c')},{'splits':(1,2)},
    {'gains':(1.,2.)},{'splits':(-1,0,0)},{'splits':(True,0,0)},{'splits':(1.5,0,0)},
    {'splits':(float('nan'),0,0)},{'gains':(-1.,0.,0.)},{'gains':(float('nan'),0.,0.)},
    {'gains':(float('inf'),0.,0.)},{'gains':(1e308,1e308,0.)}])
def test_invalid_booster_evidence(artifacts,monkeypatch,kwargs):
    fake(monkeypatch,**kwargs)
    with pytest.raises(ValueError): module.extract_ml_clues(*artifacts)


def test_no_feature_names_and_types(artifacts,monkeypatch):
    class Booster:
        def __init__(self,**kwargs): pass
        def num_feature(self): return 3
        def current_iteration(self): return 1
        def feature_importance(self,**kwargs): return (0,0,0)
    monkeypatch.setattr(module.lightgbm,'Booster',Booster)
    assert module.extract_ml_clues(*artifacts).feature_count==3
    with pytest.raises(TypeError): module.extract_ml_clues(object(),artifacts[1])
    with pytest.raises(TypeError): module.extract_ml_clues(artifacts[0],object())


@pytest.mark.parametrize('changes',[{'field_id':'y.other'},{'feature_kind':'OTHER'},{'feature_index':True},
    {'feature_index':-1},{'split_count':True},{'split_count':-1},{'gain':1},{'gain':float('nan')},
    {'gain':float('inf')},{'gain':-1.},{'gain_fraction':1.1},{'gain_fraction':True},{'split_fraction':-.1}])
def test_feature_validation(changes):
    c=module.MLFeatureClue('x.a','NUMERIC',0,1,1.,1.,1.)
    with pytest.raises(ValueError): replace(c,**changes)


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'model_fingerprint':'bad'},{'model_sha256':'A'*64},
    {'training_matrix_fingerprint':'bad'},{'preprocessor_fingerprint':'bad'},{'evaluation_fingerprint':'bad'},
    {'target_field_id':'x.other'},{'feature_count':True},{'feature_count':0},{'feature_count':4},
    {'total_split_count':True},{'total_split_count':-1},{'total_split_count':14},
    {'total_gain':150},{'total_gain':-1.},{'total_gain':float('inf')},{'total_gain':149.},
    {'feature_clues':[]},{'feature_clues':()},{'feature_clues':(object(),)*3},
    {'gain_order':['x.a','x.b','x.c']},{'gain_order':('x.a','x.a','x.c')},
    {'gain_order':('x.c','x.b','x.a')},{'split_order':('x.c','x.b','x.a')},
    {'fingerprint':'0'*64}])
def test_report_validation(artifacts,monkeypatch,changes):
    r=report(artifacts,monkeypatch)
    with pytest.raises(ValueError): replace(r,**changes)


def test_nested_consistency_fingerprint_and_frozen(artifacts,monkeypatch):
    r=report(artifacts,monkeypatch)
    for altered in (replace(r.feature_clues[0],feature_index=1),
                    replace(r.feature_clues[0],field_id='x.b'),
                    replace(r.feature_clues[0],gain_fraction=.5),
                    replace(r.feature_clues[0],split_fraction=.5)):
        with pytest.raises(ValueError): replace(r,feature_clues=(altered,)+r.feature_clues[1:])
    state=asdict(r); state.pop('fingerprint')
    assert r.fingerprint==digest(state)
    with pytest.raises(ValueError): replace(r,evaluation_fingerprint='b'*64)
    with pytest.raises(FrozenInstanceError): r.total_gain=0.
    with pytest.raises(FrozenInstanceError): r.feature_clues[0].gain=0.


def test_real_integration(artifacts):
    model,evaluation=artifacts
    r=module.extract_ml_clues(model,evaluation)
    assert r.model_fingerprint==model.fingerprint
    assert r.model_sha256==model.model_sha256
    assert r.training_matrix_fingerprint==model.training_matrix_fingerprint
    assert r.preprocessor_fingerprint==model.preprocessor_fingerprint
    assert r.evaluation_fingerprint==evaluation.fingerprint
    assert r.target_field_id==model.target_field_id
    assert r.feature_count==len(model.feature_specs)
    assert tuple(c.field_id for c in r.feature_clues)==tuple(s.field_id for s in model.feature_specs)
    assert all(isfinite(c.gain) and isfinite(c.gain_fraction) and isfinite(c.split_fraction) for c in r.feature_clues)
    assert module.extract_ml_clues(model,evaluation)==r
