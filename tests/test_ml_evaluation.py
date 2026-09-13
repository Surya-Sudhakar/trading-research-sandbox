from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
from math import copysign, isfinite, sqrt
import pytest
from test_ml_preprocessing import matrix as raw_matrix
from sandbox.research.ml_preprocessing import fit_ml_preprocessor, transform_ml_matrix, EncodedMLMatrix
from sandbox.research.ml_lightgbm import (LightGBMPredictionBatch, LightGBMClueConfig,
    train_lightgbm_clue_model, predict_lightgbm_clue_model)
from sandbox.research import ml_evaluation as module


def digest(state):
    return hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def batch(encoded,values,**changes):
    state=dict(version='LIGHTGBM_PREDICTION_BATCH_V1',model_fingerprint='a'*64,
        encoded_matrix_fingerprint=encoded.fingerprint,source_row_indices=encoded.source_row_indices,
        target_field_id=encoded.target_field_id,predictions=tuple(float(v) for v in values))
    state.update(changes)
    return LightGBMPredictionBatch(**state,fingerprint=digest(state))


def inputs(actual=(1.,2.,3.,4.),predicted=(1.,2.,3.,4.)):
    raw=raw_matrix(tuple((float(i),'A') for i in range(len(actual))),
        y=tuple(None if v is None else 0.0 if v == 0 else float(v) for v in actual),indices=tuple(4+3*i for i in range(len(actual))))
    encoded=transform_ml_matrix(raw,fit_ml_preprocessor(raw))
    return encoded,batch(encoded,predicted)


def evaluate(actual=(1.,2.,3.,4.),predicted=(1.,2.,3.,4.)):
    return module.evaluate_prediction_batch(*inputs(actual,predicted))


def test_perfect_and_provenance():
    matrix,predictions=inputs(); r=module.evaluate_prediction_batch(matrix,predictions)
    assert r.version=='ML_PREDICTION_EVALUATION_V1'
    assert r.model_fingerprint==predictions.model_fingerprint
    assert r.encoded_matrix_fingerprint==matrix.fingerprint
    assert r.prediction_batch_fingerprint==predictions.fingerprint
    assert r.target_field_id==matrix.target_field_id
    assert (r.source_row_count,r.observed_target_count,r.missing_target_count)==(4,4,0)
    assert r.observed_source_row_indices==(4,7,10,13)
    assert r.actual_mean==r.actual_median==r.prediction_mean==r.prediction_median==2.5
    for name in ('mean_absolute_error','median_absolute_error','mean_squared_error','root_mean_squared_error','residual_mean','residual_median'):
        assert getattr(r,name)==0.
        assert type(getattr(r,name)) is float and copysign(1.,getattr(r,name))==1.
    assert r.r_squared==r.pearson_correlation==r.spearman_rank_correlation==1.
    assert (r.actual_minimum,r.actual_maximum,r.prediction_minimum,r.prediction_maximum)==(1.,4.,1.,4.)


def test_residual_convention_and_errors():
    r=evaluate((1,2,3,4),(2,0,6,4))
    # Residuals 1,-2,3,0; absolute errors 1,2,3,0; squares 1,4,9,0.
    assert r.residual_mean==.5 and r.residual_median==.5
    assert r.mean_absolute_error==1.5 and r.median_absolute_error==1.5
    assert r.mean_squared_error==3.5 and r.root_mean_squared_error==sqrt(3.5)
    assert r.prediction_mean==3. and r.prediction_median==3.
    assert r.r_squared==pytest.approx(1-14/5)
    assert r.pearson_correlation==pytest.approx(6/sqrt(100))


def test_constant_actual_prediction_and_negative_r_squared():
    r=evaluate((5,5,5,5),(1,2,3,4))
    assert r.r_squared is r.pearson_correlation is r.spearman_rank_correlation is None
    assert r.mean_absolute_error==2.5 and r.mean_squared_error==7.5
    r=evaluate((1,2,3,4),(2,2,2,2))
    assert r.pearson_correlation is r.spearman_rank_correlation is None
    assert r.r_squared==pytest.approx(-.2)
    r=evaluate((1,2,3,4),(100,100,100,100))
    assert r.r_squared < -1
    r=evaluate((5,5,5,5),(5,5,5,5))
    assert r.mean_absolute_error==r.mean_squared_error==0.
    assert r.r_squared is None


def test_tied_average_ranks_and_reverse():
    # A ranks: 1,2.5,2.5,4; P ranks: 2,1,3.5,3.5.
    # centered dot=2.25; both sums of squares=4.5.
    r=evaluate((10,20,20,40),(20,10,40,40))
    assert r.spearman_rank_correlation==pytest.approx(.5)
    assert module._average_ranks((10.,20.,20.,40.))==(1.,2.5,2.5,4.)
    assert module._average_ranks((20.,10.,40.,40.))==(2.,1.,3.5,3.5)
    reversed_result=evaluate((1,2,3,4,5),(5,4,3,2,1))
    assert reversed_result.pearson_correlation==pytest.approx(-1.)
    assert reversed_result.spearman_rank_correlation==pytest.approx(-1.)
    assert module._average_ranks((3.,3.,3.))==(2.,2.,2.)


def test_missing_targets_exclude_predictions_from_all_metrics():
    r=evaluate((1,None,3,None,5),(1,9999,3,-9999,5))
    assert (r.source_row_count,r.observed_target_count,r.missing_target_count)==(5,3,2)
    assert r.observed_source_row_indices==(4,10,16)
    perfect=evaluate((1,3,5),(1,3,5))
    for name in module._REQUIRED_METRICS+module._OPTIONAL_METRICS:
        assert getattr(r,name)==getattr(perfect,name)
    assert r.prediction_minimum==1. and r.prediction_maximum==5.


def test_no_x_or_model_access(monkeypatch):
    encoded,predictions=inputs()
    original=EncodedMLMatrix.__getattribute__
    def guarded(self,name):
        if name in ('x_rows','feature_specs','categorical_feature_indices'):
            raise AssertionError('evaluator read predictor content')
        return original(self,name)
    monkeypatch.setattr(EncodedMLMatrix,'__getattribute__',guarded)
    assert module.evaluate_prediction_batch(encoded,predictions).mean_squared_error==0.


@pytest.mark.parametrize('change',[{'encoded_matrix_fingerprint':'b'*64},
    {'source_row_indices':(5,8,11,14)},{'target_field_id':'y.other'}])
def test_artifact_mismatch(change):
    encoded,predictions=inputs()
    mismatched=batch(encoded,predictions.predictions,**change)
    with pytest.raises(ValueError): module.evaluate_prediction_batch(encoded,mismatched)


@pytest.mark.parametrize('actual',[(None,None),(1,None)])
def test_insufficient_observations(actual):
    with pytest.raises(ValueError): evaluate(actual,(1,2))


def test_types():
    encoded,predictions=inputs()
    with pytest.raises(TypeError): module.evaluate_prediction_batch(object(),predictions)
    with pytest.raises(TypeError): module.evaluate_prediction_batch(encoded,object())


def test_determinism_fingerprint_and_no_partition_claim():
    encoded,predictions=inputs()
    a=module.evaluate_prediction_batch(encoded,predictions)
    b=module.evaluate_prediction_batch(encoded,predictions)
    assert a==b and a.fingerprint==b.fingerprint
    payload={f.name:getattr(a,f.name) for f in fields(a) if f.name!='fingerprint'}
    assert a.fingerprint==digest(payload)
    forbidden={'partition_role','validation_partition_id','final_partition_id','out_of_sample','validated','phase',
        'edge','alpha','profitable','good','bad','strong','weak','validation_pass','final_pass','buy','sell','win_rate','trade','strategy'}
    assert not forbidden.intersection(payload)
    with pytest.raises(FrozenInstanceError): a.mean_squared_error=1.


@pytest.mark.parametrize('changes',[{'version':'OTHER'},{'model_fingerprint':'bad'},
    {'encoded_matrix_fingerprint':'A'*64},{'prediction_batch_fingerprint':'g'*64},
    {'target_field_id':'x.other'},{'target_field_id':''},{'source_row_count':True},
    {'observed_target_count':True},{'missing_target_count':False},{'source_row_count':0},
    {'observed_target_count':1},{'missing_target_count':-1},{'source_row_count':5},
    {'observed_source_row_indices':[]},{'observed_source_row_indices':(4,7)},
    {'observed_source_row_indices':(True,7,10,13)},{'observed_source_row_indices':(-1,7,10,13)},
    {'observed_source_row_indices':(7,4,10,13)},{'observed_source_row_indices':(4,4,10,13)},
    {'actual_mean':1},{'prediction_mean':True},{'actual_median':None},
    {'residual_mean':float('nan')},{'prediction_median':float('inf')},
    {'residual_median':-0.},{'mean_absolute_error':-1.},{'median_absolute_error':-1.},
    {'mean_squared_error':-1.},{'root_mean_squared_error':-1.},{'r_squared':float('nan')},
    {'r_squared':1},{'pearson_correlation':1.01},{'spearman_rank_correlation':-1.01},
    {'actual_minimum':5.},{'prediction_maximum':0.}])
def test_structural_validation(changes,monkeypatch):
    r=evaluate()
    # Keep the hash check from masking a missing structural check.
    monkeypatch.setattr(module,'_digest',lambda payload:r.fingerprint)
    with pytest.raises(ValueError): replace(r,**changes)


@pytest.mark.parametrize('fingerprint',[None,'','a'*63,'A'*64,'z'*64,'0'*64])
def test_fingerprint_validation(fingerprint):
    with pytest.raises(ValueError): replace(evaluate(),fingerprint=fingerprint)


def test_content_change_requires_new_fingerprint():
    r=evaluate()
    with pytest.raises(ValueError): replace(r,mean_absolute_error=1.)
    with pytest.raises(ValueError): replace(r,model_fingerprint='b'*64)


def test_normalized_zero_and_large_arithmetic():
    r=evaluate((-0.,0.),(-0.,0.))
    for name in module._REQUIRED_METRICS:
        assert copysign(1.,getattr(r,name))==1.
    with pytest.raises(ValueError): evaluate((1e308,-1e308),(-1e308,1e308))


def test_full_api_integration():
    raw=raw_matrix(tuple((float(i%10),'A' if i%2 else 'B') for i in range(30)),
        y=tuple(None if i%7==0 else float(i%10)*2. for i in range(30)),indices=tuple(10+2*i for i in range(30)))
    pre=fit_ml_preprocessor(raw)
    encoded=transform_ml_matrix(raw,pre)
    model=train_lightgbm_clue_model(encoded,LightGBMClueConfig(num_boost_round=4,min_data_in_leaf=2))
    predictions=predict_lightgbm_clue_model(model,encoded)
    result=module.evaluate_prediction_batch(encoded,predictions)
    assert result.model_fingerprint==model.fingerprint
    assert result.encoded_matrix_fingerprint==encoded.fingerprint
    assert result.prediction_batch_fingerprint==predictions.fingerprint
    assert result.target_field_id==encoded.target_field_id
    assert result.source_row_count==30
    assert result.observed_target_count==25 and result.missing_target_count==5
    assert result.observed_source_row_indices==encoded.observed_target_row_indices
    for name in module._REQUIRED_METRICS+module._OPTIONAL_METRICS:
        value=getattr(result,name)
        assert value is None or isfinite(value)
