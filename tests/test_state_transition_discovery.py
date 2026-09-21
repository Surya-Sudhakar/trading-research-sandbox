"""All data are synthetic. Frozen parameters may be loaded but never fitted."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import ast
import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from sandbox.market_state.discovery import transition_discovery as t
from sandbox.market_state.discovery import validation as v
from sandbox.market_state.discovery.validation_contract import contract_json
from sandbox.provenance import sha256_file


def assigned(labels, *, times=None, segments=None, probabilities=None):
    labels=np.asarray(labels)
    n=len(labels)
    if times is None: times=pd.date_range("2023-02-01",periods=n,freq="15min",tz="UTC")
    if segments is None: segments=np.zeros(n,dtype=int)
    if probabilities is None: probabilities=np.eye(3)[labels]*.7+.1
    return t.causal_predictors(times,segments,probabilities,labels)


@pytest.fixture(scope="module")
def frozen():
    with v.no_fitting() as guard: contract,model,checksum=v.load_frozen_inputs()
    assert guard["fit_calls"]==0
    return contract,model,checksum


class StubModel:
    def __init__(self, metadata): self.metadata=deepcopy(metadata)
    def predict(self,x,**kwargs): return x[:,0].astype(int)%3
    def predict_proba(self,x,**kwargs): return np.eye(3)[self.predict(x,**kwargs)]*.7+.1


@pytest.fixture
def features(frozen):
    labels=np.tile([0,0,0,1,1,2,2,2,2,0],5).astype(float)
    f=pd.DataFrame({"timestamp":pd.date_range("2023-02-01T00:15Z",periods=len(labels),freq="15min"),
        **{name:labels for name in t.FEATURES},"continuity_segment_id":np.repeat([0,1],25)})
    return f,StubModel(frozen[1].metadata),frozen[0]


def test_exact_causal_fields_and_safe_zero_entropy():
    x,blocks=assigned([0,0,1,1,1,2],probabilities=np.eye(3)[[0,0,1,1,1,2]])
    assert list(x.state_age_bars)==[1,2,1,2,3,1]
    assert list(x.previous_state)==[None,"S0","S0","S1","S1","S1"]
    np.testing.assert_allclose(x.bars_since_transition,[np.nan,np.nan,0,1,2,0],equal_nan=True)
    np.testing.assert_array_equal(x.posterior_entropy,0)
    np.testing.assert_array_equal(x.posterior_margin,1)
    assert x.loc[2,"delta_p_s0"]==-1 and x.loc[2,"delta_p_s1"]==1
    assert x.loc[0,["delta_p_s0","delta_max_posterior","delta_posterior_entropy"]].isna().all()
    assert set(x)=={"timestamp","state_t","p_s0","p_s1","p_s2","max_posterior","posterior_margin",
        "posterior_entropy","state_age_bars","previous_state","bars_since_transition","delta_p_s0",
        "delta_p_s1","delta_p_s2","delta_max_posterior","delta_posterior_entropy"}
    assert "transition_within_h" not in x


def test_entropy_margin_and_delta_values():
    p=np.array([[.7,.2,.1],[.5,.3,.2],[.1,.8,.1]])
    x,_=assigned([0,0,1],probabilities=p)
    np.testing.assert_allclose(x.posterior_entropy,-np.sum(p*np.log(p),axis=1))
    np.testing.assert_allclose(x.posterior_margin,[.5,.2,.7])
    assert x.loc[1,"delta_max_posterior"]==pytest.approx(-.2)
    assert x.loc[1,"delta_posterior_entropy"]==pytest.approx(x.posterior_entropy.iloc[1]-x.posterior_entropy.iloc[0])


def test_future_changes_cannot_change_past_predictors():
    labels=[0,0,1,1,2,2,0,1]
    full,_=assigned(labels)
    prefix,_=assigned(labels[:5])
    changed,_=assigned(labels[:5]+[1,0,2])
    pd.testing.assert_frame_equal(full.iloc[:5],prefix)
    pd.testing.assert_frame_equal(full.iloc[:5],changed.iloc[:5])


@pytest.mark.parametrize("kind",["gap","weekend","segment"])
def test_disconnections_reset_history_and_invalidate_outcomes(kind):
    times=pd.date_range("2023-02-01",periods=20,freq="15min",tz="UTC")
    segments=np.zeros(20,dtype=int)
    if kind=="segment": segments[5:]=1
    else: times=times[:5].append(times[5:]+(timedelta(days=2) if kind=="weekend" else timedelta(minutes=15)))
    x,blocks=assigned([0,1,1,1,1]+[1]*15,times=times,segments=segments)
    assert x.loc[5,"state_age_bars"]==1
    assert x.loc[5,["previous_state","bars_since_transition","delta_max_posterior"]].isna().all()
    labels=t.transition_outcomes(x,blocks)
    assert labels["1"].loc[4,"exclusion"]=="gap_or_segment_boundary"
    assert np.isnan(labels["8"].loc[0,"transition_within_h"])
    assert labels["8"].loc[0,"exclusion"]=="gap_or_segment_boundary"


def test_any_change_then_return_is_transition_first_destination_not_endpoint():
    x,blocks=assigned([0,1,0,2,0]+[0]*15)
    original=x.copy(deep=True)
    labels=t.transition_outcomes(x,blocks)
    assert labels["2"].loc[0,"transition_within_h"]==1
    assert labels["2"].loc[0,"first_destination"]=="S1"
    assert labels["4"].loc[0,"first_destination"]=="S1"
    assert labels["1"].loc[4,"transition_within_h"]==0
    assert labels["1"].loc[4,"first_destination"] is None
    pd.testing.assert_frame_equal(x,original)
    assert tuple(int(h) for h in labels)==(1,2,4,8,12)


def test_early_change_does_not_rescue_incomplete_horizon():
    x,blocks=assigned([0,1,1])
    labels=t.transition_outcomes(x,blocks)
    assert labels["1"].loc[0,"transition_within_h"]==1
    assert np.isnan(labels["4"].loc[0,"transition_within_h"])
    assert labels["4"].loc[0,"exclusion"]=="insufficient_future_observations"


def test_period_end_no_2025_future_observation_required():
    times=pd.date_range("2024-12-31T23:00Z",periods=4,freq="15min")
    x,blocks=assigned([0,0,1,1],times=times)
    labels=t.transition_outcomes(x,blocks)
    assert labels["1"].loc[2,"transition_within_h"]==0
    assert labels["1"].loc[3,"exclusion"]=="period_end_boundary"
    assert labels["4"].exclusion.eq("period_end_boundary").all()


@pytest.mark.parametrize("time",["2022-12-31T23:45Z","2025-01-01T00:00Z"])
def test_out_of_period_assigned_rows_rejected(time):
    with pytest.raises(ValueError,match="2023-2024"): assigned([0],times=[pd.Timestamp(time)])


@pytest.mark.parametrize("kind",["probability","sum","label","nan","naive","duplicate","misaligned","null_segment"])
def test_bad_frozen_outputs_rejected(kind):
    times=pd.date_range("2023-01-01",periods=2,freq="15min",tz="UTC")
    segments=[0,0];p=np.array([[.8,.1,.1],[.1,.8,.1]]);labels=[0,1]
    if kind=="probability":p[0,0]=-1
    if kind=="sum":p[0,0]=.2
    if kind=="label":labels=[1,1]
    if kind=="nan":p[0,0]=np.nan
    if kind=="naive":times=times.tz_localize(None)
    if kind=="duplicate":times=[times[0],times[0]]
    if kind=="misaligned":times=times+timedelta(seconds=1)
    if kind=="null_segment":segments=[0,None]
    with pytest.raises(ValueError): assigned(labels,times=times,segments=segments,probabilities=p)


def test_cluster_bootstrap_matches_explicit_whole_block_resampling():
    x=np.array([1.,2.,4.,6.,3.,7.,8.,9.])
    y=np.array([0,0,1,1,0,1,0,1]);blocks=np.repeat([0,1],4)
    weights=np.array([[2,0],[1,1],[0,2]],dtype=float)
    result=t.bootstrap_intervals(x,y,blocks,np.ones(8,dtype=bool),weights)
    differences=[];effects=[]
    for counts in weights.astype(int):
        rows=np.concatenate([np.tile(np.flatnonzero(blocks==i),count) for i,count in enumerate(counts)])
        effect=t._effect(x[rows][y[rows]==1].tolist(),x[rows][y[rows]==0].tolist())
        differences.append(effect["difference_in_means"]);effects.append(effect["cohens_d"])
    np.testing.assert_allclose(result["difference_in_means"],np.quantile(differences,[.025,.975]))
    np.testing.assert_allclose(result["cohens_d"],np.quantile(effects,[.025,.975]))
    assert result["defined_effect_replicates"]==3


def test_single_block_ci_and_zero_variance_effect_are_undefined():
    result=t.bootstrap_intervals(np.ones(4),np.array([0,1,0,1]),np.zeros(4,dtype=int),np.ones(4,dtype=bool),np.ones((10,1)))
    assert result["cohens_d"] is None and result["difference_in_means"] is None
    result=t.bootstrap_intervals(np.ones(4),np.array([0,1,0,1]),np.array([0,0,1,1]),np.ones(4,dtype=bool),np.ones((10,2)))
    assert result["cohens_d"] is None
    assert result["difference_in_means"]==[0.,0.]


def test_comparison_counts_effect_orientation_base_rates_and_missingness():
    x,blocks=assigned([0,0,1,1,2,2,0,0,0,1,1,2,2,2,0,0],segments=[0]*8+[1]*8)
    outcomes=t.transition_outcomes(x,blocks)
    result=t.summarize(x,blocks,outcomes)
    table=outcomes["1"]
    for state in ("ALL",*t.STATES):
        mask=np.ones(len(x),dtype=bool) if state=="ALL" else x.state_t.eq(state).to_numpy()
        valid=mask & table.transition_within_h.notna().to_numpy()
        group=result["1"]["groups"][state]
        assert group["n_stay"]+group["n_transition"]==sum(valid)
        assert group["transition_probability"]==pytest.approx(table.transition_within_h[valid].mean())
        assert sum(z["count"] for z in group["first_destination"].values())==group["n_transition"]
        for name,c in group["comparisons"].items():
            assert c["stay"]["n"]+c["missing_stay"]==group["n_stay"]
            assert c["transition"]["n"]+c["missing_transition"]==group["n_transition"]
            if c["stay"]["n"] and c["transition"]["n"]:
                assert c["difference_in_means"]==pytest.approx(c["transition"]["mean"]-c["stay"]["mean"])
    assert result==t.summarize(x,blocks,outcomes)


def test_full_analysis_deterministic_sorted_and_unchanged(features):
    f,model,contract=features
    before=f.copy(deep=True)
    result=t.analyze(f,model,contract)
    assert result==t.analyze(f.sample(frac=1,random_state=7),model,contract)
    pd.testing.assert_frame_equal(f,before)
    assert result["fit_call_counts"]=={"total":0,"scaler":0,"gmm":0}
    assert result["eligible_observations"]==50
    assert result["continuous_observation_blocks"]==2
    assert set(result["horizons"])=={"1","2","4","8","12"}
    assert result["specification"]["pass_fail_threshold"] is None
    assert result["specification"]["bootstrap"]["replicates"]==2000
    assert len(t.report_markdown(result).splitlines())>140


def test_missing_feature_row_breaks_history_and_future(features):
    f,model,contract=features
    f.loc[10,t.FEATURES[-1]]=np.nan
    result=t.analyze(f,model,contract)
    assert result["eligible_observations"]==49
    assert result["incomplete_feature_rows_excluded"]==1
    assert result["continuous_observation_blocks"]==3
    assert result["horizons"]["1"]["groups"]["ALL"]["exclusions"]["gap_or_segment_boundary"]==2


@pytest.mark.parametrize("kind",["schema","order","states","configuration","input_order","future_field"])
def test_frozen_identity_and_leakage_enforced(features,kind):
    f,model,contract=features
    if kind=="schema": model.metadata["feature_schema_version"]="OTHER"
    if kind=="order": model.metadata["feature_names"].reverse()
    if kind=="states": model.metadata["state_ids"].reverse()
    if kind=="configuration": model.metadata["gmm"]["configuration"]["random_state"]=7
    if kind=="input_order": f=f.loc[:,["timestamp",*reversed(t.FEATURES),"continuity_segment_id"]]
    if kind=="future_field":f["future_return"]=0
    with pytest.raises(ValueError):t.analyze(f,model,contract)


@pytest.mark.parametrize("cls",[StandardScaler,GaussianMixture,v.DiscoveryStandardScaler,v.GaussianMixtureStateModel])
def test_fit_blocked_during_assignment(features,cls):
    f,model,contract=features
    model.predict=lambda *args,**kwargs:cls.fit(None,None)
    with pytest.raises(RuntimeError,match="prohibited"):t.analyze(f,model,contract)


def test_frozen_loading_and_real_model_prediction_on_synthetic_only(frozen,features):
    f,_,contract=features
    result=t.analyze(f,frozen[1],contract)
    assert result["fit_call_counts"]["total"]==0
    assert sum(result["state_occupancy"].values())==len(f)
    assert frozen[1].metadata["state_ids"]==["S0","S1","S2"]


def test_period_filter_before_analysis(features):
    f,model,contract=features
    extra=f.iloc[[0,1]].copy();extra["timestamp"]=pd.to_datetime(["2022-01-01T00:00Z","2025-01-01T00:00Z"])
    extra[list(t.FEATURES)]=float("inf")
    assert t.analyze(pd.concat([f,extra],ignore_index=True),model,contract)==t.analyze(f,model,contract)


def test_no_future_return_or_classifier_engine_dependency():
    tree=ast.parse(inspect.getsource(t))
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(any(s in (name or "") for s in ("strategy","execution","lightgbm","hawkes","classifier","version_0")) for name in imports)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ("fit","fit_predict","fit_transform") for n in ast.walk(tree))


def test_frozen_checksums_and_contract_verified_before_input_access(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise ValueError("SHA256 mismatch")
    def never(*args,**kwargs):pytest.fail("data read before preflight")
    monkeypatch.setattr(v,"load_frozen_inputs",fail)
    monkeypatch.setattr(v,"read_validation_inputs",never)
    with pytest.raises(ValueError,match="SHA256"):t.run(None,tmp_path/"result.json")
    assert not list(tmp_path.iterdir())


def test_attempt_reservation_blocks_repeat_before_data_access(tmp_path,monkeypatch):
    output=tmp_path/"result.json";output.with_suffix(".attempt.json").write_text("reserved")
    monkeypatch.setattr(v,"load_frozen_inputs",lambda *a:pytest.fail("repeat must stop"))
    with pytest.raises(FileExistsError):t.run(None,output)


def test_input_failure_cannot_publish_success(tmp_path,monkeypatch):
    monkeypatch.setattr(v,"read_validation_inputs",lambda *a:(_ for _ in ()).throw(ValueError("bad source")))
    output=tmp_path/"result.json"
    with pytest.raises(ValueError,match="bad source"):t.run(None,output)
    assert not output.exists()
    manifest=json.loads(output.with_suffix(".attempt.json").read_text())
    assert manifest["status"]=="ATTEMPT_RESERVED_NOT_COMPLETION"
    assert manifest["frozen_files_before"][str(v.DEFAULT_MODEL)]==v.EXPECTED_MODEL_SHA256


def synthetic_reader(features):
    f=features.copy()
    prices=pd.DataFrame({"timestamp_utc":f.timestamp-timedelta(minutes=15),"continuity_segment_id":f.continuity_segment_id})
    return f,prices,{"provider":"DUKASCOPY","synthetic":True}


def test_success_publication_and_after_checks_without_real_data(tmp_path,features,monkeypatch):
    f,_,_=features
    monkeypatch.setattr(v,"read_validation_inputs",lambda *args:synthetic_reader(f))
    output=tmp_path/"result.json"
    result,checksum=t.run(None,output)
    assert sha256_file(output)==checksum
    assert json.loads(output.read_text())==result
    assert result["frozen_files_before"]==result["frozen_files_after"]
    assert result["contract_checksum_before"]==result["contract_checksum_after"]
    assert result["fit_call_counts"]=={"total":0,"scaler":0,"gmm":0}
    assert output.with_suffix(".md").read_text(encoding="utf-8")==t.report_markdown(result)
    manifest=json.loads(output.with_suffix(".attempt.json").read_text())
    assert manifest["specification"]==result["specification"]
    assert manifest["model_metadata"]["feature_names"]==list(t.FEATURES)
    with pytest.raises(FileExistsError):t.run(None,output)


def test_changed_protected_artifact_prevents_publication(tmp_path,features,monkeypatch):
    f,_,_=features
    monkeypatch.setattr(v,"read_validation_inputs",lambda *args:synthetic_reader(f))
    snapshots=iter([{"protected":"original"},{"protected":"changed"}])
    monkeypatch.setattr(t,"frozen_snapshot",lambda:next(snapshots))
    output=tmp_path/"result.json"
    with pytest.raises(ValueError,match="protected artifact changed"):t.run(None,output)
    assert not output.exists() and not output.with_suffix(".md").exists()


def test_empty_transition_group_keeps_all_horizons_no_fabricated_effects():
    x,blocks=assigned([0]*30,segments=[0]*15+[1]*15)
    result=t.summarize(x,blocks,t.transition_outcomes(x,blocks))
    assert len(result)==5
    for h in result.values():
        for name,c in h["groups"]["ALL"]["comparisons"].items():
            assert c["transition"]["n"]==0
            assert c["transition"]["mean"] is None
            assert c["cohens_d"] is None
            assert c["bootstrap_95_ci"]["cohens_d"] is None
        assert h["groups"]["S1"]["transition_probability"] is None
        assert h["groups"]["S0"]["transition_probability"]==0


def test_sample_summary_and_effect_definition():
    s=t._stats([1.,2.,4.])
    assert s["n"]==3 and s["median"]==2
    assert s["mean"]==pytest.approx(7/3)
    assert s["standard_deviation"]==pytest.approx(np.std([1,2,4],ddof=1))
    assert t._stats([2])["standard_deviation"] is None
    assert t._effect([3.,4.],[1.,2.])["cohens_d"]==pytest.approx(2/np.sqrt(.5))


def test_caught_fit_attempt_still_rejects_analysis(features):
    f,model,contract=features
    original=model.predict
    def attempted(*args,**kw):
        try: StandardScaler().fit(np.zeros((3,14)))
        except RuntimeError: pass
        return original(*args,**kw)
    model.predict=attempted
    with pytest.raises(RuntimeError,match="fit attempt detected"):t.analyze(f,model,contract)
