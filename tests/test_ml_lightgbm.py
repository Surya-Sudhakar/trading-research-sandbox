from dataclasses import FrozenInstanceError, replace, asdict
import hashlib
import json
from math import isfinite
import numpy as np
import pytest
from test_ml_preprocessing import matrix, SCHEMA, TARGET
from sandbox.research import ml_preprocessing as prep
from sandbox.research import ml_lightgbm as module


def discovery():
    rows=tuple((None if i%13==0 else float(i%30),None if i%17==0 else ('A' if i%2==0 else 'B')) for i in range(120))
    y=tuple(None if i%11==0 else 3.*(i%30)+(10. if i%2 else -10.) for i in range(120))
    return matrix(rows,y,indices=tuple(100+3*i for i in range(120)))


@pytest.fixture(scope='module')
def artifacts():
    raw=discovery(); pre=prep.fit_ml_preprocessor(raw); encoded=prep.transform_ml_matrix(raw,pre)
    config=module.LightGBMClueConfig(num_boost_round=12,min_data_in_leaf=3,num_leaves=7,learning_rate=.1)
    model=module.train_lightgbm_clue_model(encoded,config)
    return raw,pre,encoded,config,model


def test_train_predict_vertical_slice_and_unknown(artifacts):
    raw,pre,encoded,config,model=artifacts
    assert model.version=='LIGHTGBM_CLUE_MODEL_V1'
    assert model.lightgbm_version==module.lightgbm.__version__
    assert model.training_row_indices==raw.observed_target_row_indices
    assert model.training_matrix_fingerprint==encoded.fingerprint
    assert model.preprocessor_fingerprint==pre.fingerprint
    assert model.feature_specs==SCHEMA and model.categorical_feature_indices==(1,)
    assert model.target_field_id==TARGET and model.config==config
    assert model.model_sha256==hashlib.sha256(model.model_text.encode()).hexdigest()
    assert isinstance(model.model_text,str) and len(model.model_text)>0
    result=module.predict_lightgbm_clue_model(model,encoded)
    assert result.version=='LIGHTGBM_PREDICTION_BATCH_V1'
    assert result.source_row_indices==raw.source_row_indices
    assert len(result.predictions)==120
    assert all(type(v) is float and isfinite(v) for v in result.predictions)
    assert result.model_fingerprint==model.fingerprint
    assert result.encoded_matrix_fingerprint==encoded.fingerprint
    assert result.target_field_id==TARGET
    assert max(result.predictions)>min(result.predictions)
    validation=matrix(((2.,'UNSEEN'),(None,None),(25.,'A')),y=(None,None,None),indices=(1000,1003,1007))
    transformed=prep.transform_ml_matrix(validation,pre)
    assert transformed.x_rows[0][1]==1 and transformed.x_rows[1]==(None,0)
    predicted=module.predict_lightgbm_clue_model(model,transformed)
    assert predicted.source_row_indices==(1000,1003,1007)
    assert len(predicted.predictions)==3 and all(isfinite(v) for v in predicted.predictions)
    assert pre==prep.fit_ml_preprocessor(raw)


def test_deterministic_retraining_and_seed_provenance(artifacts):
    raw,pre,encoded,config,model=artifacts
    again=module.train_lightgbm_clue_model(encoded,config)
    assert again.training_row_indices==model.training_row_indices
    assert again.model_sha256==model.model_sha256 and again.fingerprint==model.fingerprint
    assert module.predict_lightgbm_clue_model(again,encoded).predictions==pytest.approx(module.predict_lightgbm_clue_model(model,encoded).predictions,abs=1e-12,rel=1e-12)
    seeded=module.train_lightgbm_clue_model(encoded,replace(config,seed=42))
    assert seeded.config.seed==42 and seeded.fingerprint!=model.fingerprint


def test_prediction_target_independence_and_training_dependence(artifacts):
    raw,pre,encoded,config,model=artifacts
    changed=matrix(raw.x_rows,y=tuple(None if i%2 else 1000.+i for i in range(120)),indices=raw.source_row_indices)
    transformed=prep.transform_ml_matrix(changed,pre)
    assert transformed.target_observed!=encoded.target_observed
    assert transformed.observed_target_row_indices!=encoded.observed_target_row_indices
    assert module.predict_lightgbm_clue_model(model,transformed).predictions==module.predict_lightgbm_clue_model(model,encoded).predictions
    changed_y=matrix(raw.x_rows,y=tuple(None if v is None else 1000.-10*v for v in raw.y_values),indices=raw.source_row_indices)
    other=module.train_lightgbm_clue_model(prep.transform_ml_matrix(changed_y,pre),config)
    assert other.model_sha256!=model.model_sha256


def test_exact_training_inputs_parameters_and_no_validation(artifacts,monkeypatch):
    raw,pre,encoded,config,model=artifacts
    dataset_fn,train_fn=module.lightgbm.Dataset,module.lightgbm.train
    captured={}
    def dataset_spy(x,**kwargs):
        captured['x']=x; captured['dataset']=kwargs
        return dataset_fn(x,**kwargs)
    def train_spy(params,training,**kwargs):
        captured['params']=params; captured['train']=kwargs
        return train_fn(params,training,**kwargs)
    monkeypatch.setattr(module.lightgbm,'Dataset',dataset_spy)
    monkeypatch.setattr(module.lightgbm,'train',train_spy)
    trained=module.train_lightgbm_clue_model(encoded,config)
    positions=[i for i,observed in enumerate(encoded.target_observed) if observed]
    assert captured['x'].dtype==np.float64 and captured['x'].shape==(len(positions),2)
    for row,i in zip(captured['x'],positions):
        if encoded.x_rows[i][0] is None: assert np.isnan(row[0])
        else: assert row[0]==encoded.x_rows[i][0]
        assert row[1]==encoded.x_rows[i][1]
    assert captured['dataset']['label'].tolist()==[encoded.y_values[i] for i in positions]
    assert captured['dataset']['label'].dtype==np.float64
    assert captured['dataset']['categorical_feature']==[1]
    assert captured['dataset']['feature_name']==[s.field_id for s in SCHEMA]
    assert captured['dataset']['free_raw_data'] is False
    assert captured['train']=={'num_boost_round':config.num_boost_round}
    assert captured['params']==dict(objective='regression',metric='l2',learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,min_data_in_leaf=config.min_data_in_leaf,verbosity=-1,
        feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,seed=config.seed,
        feature_fraction_seed=config.seed,bagging_seed=config.seed,data_random_seed=config.seed,
        deterministic=True,force_col_wise=True,num_threads=1,feature_pre_filter=False)
    assert trained.training_row_indices==encoded.observed_target_row_indices


def test_missing_targets_excluded_and_predicted():
    raw=matrix(((1.,'A'),(999.,'A'),(3.,'A'),(-999.,'A')),y=(1.,None,3.,None),indices=(4,7,10,13))
    pre=prep.fit_ml_preprocessor(raw); encoded=prep.transform_ml_matrix(raw,pre)
    config=module.LightGBMClueConfig(num_boost_round=2,min_data_in_leaf=1)
    model=module.train_lightgbm_clue_model(encoded,config)
    assert model.training_row_indices==(4,10)
    assert len(module.predict_lightgbm_clue_model(model,encoded).predictions)==4
    changed=matrix(((1.,'A'),(-9000.,'A'),(3.,'A'),(9000.,'A')),y=raw.y_values,indices=raw.source_row_indices)
    again=module.train_lightgbm_clue_model(prep.transform_ml_matrix(changed,pre),config)
    assert again.model_sha256==model.model_sha256
    for y in ((None,None,None,None),(1.,None,None,None)):
        insufficient=prep.transform_ml_matrix(matrix(raw.x_rows,y=y,indices=raw.source_row_indices),pre)
        with pytest.raises(ValueError): module.train_lightgbm_clue_model(insufficient,config)


def test_prediction_schema_checks(artifacts):
    raw,pre,encoded,config,model=artifacts
    reordered=matrix(tuple(row[::-1] for row in raw.x_rows),raw.y_values,raw.source_row_indices,specs=SCHEMA[::-1])
    other=prep.transform_ml_matrix(reordered,prep.fit_ml_preprocessor(reordered))
    with pytest.raises(ValueError): module.predict_lightgbm_clue_model(model,other)
    changed=matrix(((1.,'OTHER'),))
    different_pre=prep.transform_ml_matrix(changed,prep.fit_ml_preprocessor(changed))
    with pytest.raises(ValueError): module.predict_lightgbm_clue_model(model,different_pre)
    state=prep._state(encoded); state['target_field_id']='y.other'
    different_target=prep.EncodedMLMatrix(**state,fingerprint=prep._encoded_digest(state))
    with pytest.raises(ValueError): module.predict_lightgbm_clue_model(model,different_target)
    with pytest.raises(TypeError): module.predict_lightgbm_clue_model(object(),encoded)
    with pytest.raises(TypeError): module.predict_lightgbm_clue_model(model,object())
    with pytest.raises(TypeError): module.train_lightgbm_clue_model(object())
    with pytest.raises(TypeError): module.train_lightgbm_clue_model(encoded,object())


def test_prediction_reconstruction_and_iteration(artifacts,monkeypatch):
    raw,pre,encoded,config,model=artifacts
    captured={}
    class Booster:
        def __init__(self,**kwargs): captured['init']=kwargs
        def current_iteration(self): return 7
        def predict(self,x,**kwargs):
            captured['x']=x; captured['kwargs']=kwargs
            return np.zeros(len(x),dtype=np.float64)
    monkeypatch.setattr(module.lightgbm,'Booster',Booster)
    result=module.predict_lightgbm_clue_model(model,encoded)
    assert captured['init']=={'model_str':model.model_text}
    assert captured['kwargs']['num_iteration']==7
    assert captured['x'].shape==(len(encoded.x_rows),len(SCHEMA))
    assert captured['x'].dtype==np.float64
    assert result.predictions==(0.,)*len(encoded.x_rows)


@pytest.mark.parametrize('name,value',[(name,value) for name in ('num_boost_round','num_leaves','min_data_in_leaf','seed') for value in (True,-1,1.,None)]
    +[('num_boost_round',0),('num_leaves',1),('min_data_in_leaf',0),('learning_rate',0.),('learning_rate',-.1),
      ('learning_rate',1),('learning_rate',True),('learning_rate',float('nan')),('learning_rate',float('inf'))])
def test_config_validation(name,value):
    with pytest.raises(ValueError): module.LightGBMClueConfig(**{name:value})


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'lightgbm_version':''},{'training_matrix_fingerprint':'bad'},
    {'preprocessor_fingerprint':'A'*64},{'target_field_id':'x.other'},{'feature_specs':()},
    {'categorical_feature_indices':()},{'categorical_feature_indices':(True,)},
    {'training_row_indices':[]},{'training_row_indices':()},{'training_row_indices':(True,)},
    {'training_row_indices':(-1,)},{'training_row_indices':(2,1)},{'training_row_indices':(1,1)},
    {'config':object()},{'model_text':''},{'model_text':'different'},{'model_sha256':'0'*64},
    {'fingerprint':'0'*64}])
def test_model_validation(changes,artifacts):
    with pytest.raises(ValueError): replace(artifacts[4],**changes)


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'model_fingerprint':'bad'},{'encoded_matrix_fingerprint':'A'*64},
    {'source_row_indices':[]},{'source_row_indices':()},{'source_row_indices':(True,)},
    {'source_row_indices':(-1,)},{'source_row_indices':(2,1)},{'target_field_id':'x.other'},
    {'predictions':[]},{'predictions':(1.,)},{'fingerprint':'0'*64}])
def test_prediction_validation(changes,artifacts):
    result=module.predict_lightgbm_clue_model(artifacts[4],artifacts[2])
    with pytest.raises(ValueError): replace(result,**changes)


@pytest.mark.parametrize('value',[True,1,None,'1',float('nan'),float('inf'),-float('inf')])
def test_invalid_prediction_scalar(value,artifacts):
    result=module.predict_lightgbm_clue_model(artifacts[4],artifacts[2])
    with pytest.raises(ValueError): replace(result,predictions=(value,)+result.predictions[1:])


def test_hash_payloads_and_frozen(artifacts):
    raw,pre,encoded,config,model=artifacts
    state=prep._state(model)
    state.pop('model_text')
    state['config']=asdict(config)
    state['feature_specs']=[{'field_id':s.field_id,'feature_kind':s.feature_kind} for s in model.feature_specs]
    assert model.fingerprint==hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    predictions=module.predict_lightgbm_clue_model(model,encoded)
    payload=prep._state(predictions)
    assert predictions.fingerprint==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with pytest.raises(FrozenInstanceError): model.model_text='other'
    with pytest.raises(FrozenInstanceError): config.seed=1
    with pytest.raises(FrozenInstanceError): predictions.predictions=()
