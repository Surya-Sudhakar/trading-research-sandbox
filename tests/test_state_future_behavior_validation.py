"""Synthetic validation tests. No historical validation observations are opened."""
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sandbox.market_state.discovery import validation as v
from sandbox.market_state.discovery.validation_contract import contract_json
from sandbox.market_state.discovery.frozen_model import FrozenMarketStateModel
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1 as NAMES
from sandbox.partition.service import PartitionService
from sandbox.partition.models import PartitionRole
from sandbox.research.errors import ResearchError
from sandbox.provenance import sha256_file


@pytest.fixture(scope="module")
def frozen():
    # Loading the already persisted parameters is allowed; never fit even synthetic models.
    with v.no_fitting() as calls:
        contract, model, checksum = v.load_frozen_inputs()
    assert calls["fit_calls"] == 0
    return contract, model, checksum


class FixedLabels:
    def __init__(self, metadata):
        self.metadata = deepcopy(metadata)
    def predict(self, x, **kwargs):
        assert tuple(kwargs["feature_names"]) == NAMES
        return x[:, 0].astype(int) % 3
    def predict_proba(self, x, **kwargs):
        return np.eye(3)[self.predict(x, **kwargs)] * .7 + .1


@pytest.fixture
def inputs(frozen):
    times = pd.date_range("2023-06-01", periods=40, freq="15min", tz="UTC")
    close = 100 + np.arange(len(times)) * .1
    prices = pd.DataFrame(dict(timestamp_utc=times, open=close, high=close+.2,
        low=close-.2, close=close, continuity_segment_id="SEG"))
    features = pd.DataFrame({"timestamp": times+v.STEP,
        **{name: (np.arange(len(times)) % 3).astype(float) for name in NAMES},
        "continuity_segment_id":"SEG", "provider":"DUKASCOPY", "symbol":"EURUSD", "decision_timeframe":"M15"})
    return features, prices, FixedLabels(frozen[1].metadata), frozen[0]


def analyze(inputs):
    return v.analyze_validation(*inputs)


def test_frozen_model_loading_and_predictions_never_fit(frozen):
    contract, model, checksum = frozen
    x = np.random.default_rng(7).normal(size=(9,14))
    with v.no_fitting() as guard:
        _, second, _ = v.load_frozen_inputs()
        np.testing.assert_array_equal(model.predict(x), second.predict(x))
        np.testing.assert_array_equal(model.predict_proba(x), second.predict_proba(x))
    assert guard["fit_calls"] == 0
    assert model.metadata["feature_names"] == list(NAMES)
    assert model.metadata["state_ids"] == ["S0","S1","S2"]
    assert model.metadata["gmm"]["configuration"]["random_state"] == 42
    assert len(NAMES) == 14
    assert checksum == sha256_file(v.DEFAULT_CONTRACT)


@pytest.mark.parametrize("cls", [StandardScaler, GaussianMixture, v.DiscoveryStandardScaler, v.GaussianMixtureStateModel])
def test_fit_is_blocked(cls):
    with v.no_fitting() as guard:
        with pytest.raises(RuntimeError, match="prohibited"):
            cls.fit(None, None)
    assert guard["fit_calls"] == 1


@pytest.mark.parametrize("kind", ["corrupt","truncated"])
def test_model_checksum_enforced_before_loading(tmp_path, kind):
    raw = v.DEFAULT_MODEL.read_bytes()
    path = tmp_path / "model.json"
    path.write_bytes(raw[:20] if kind == "truncated" else raw+b" ")
    with pytest.raises(ValueError, match="SHA256"):
        v.load_frozen_inputs(model_path=path)


def test_contract_corruption_rejected_before_model_load(tmp_path):
    contract = json.loads(contract_json())
    contract["model_configuration"]["random_state"] = 7
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        v.load_frozen_inputs(path, tmp_path/"missing-model")


@pytest.mark.parametrize("bad", ["order","missing","schema","states","configuration"])
def test_frozen_schema_order_and_state_configuration_enforced(inputs, bad):
    f,p,m,c = inputs
    if bad == "order":
        f = f.loc[:, ["timestamp", *reversed(NAMES), *[x for x in f if x not in NAMES and x != "timestamp"]]]
    if bad == "missing": f=f.drop(columns=NAMES[-1])
    if bad == "schema": m.metadata["feature_schema_version"]="OTHER"
    if bad == "states": m.metadata["state_ids"].reverse()
    if bad == "configuration": m.metadata["gmm"]["configuration"]["n_components"]=4
    with pytest.raises(ValueError): v.analyze_validation(f,p,m,c)


def test_deterministic_chronological_sorting_inputs_unchanged(inputs):
    f,p,m,c=inputs
    before_f,before_p=f.copy(deep=True),p.copy(deep=True)
    first=analyze(inputs)
    second=v.analyze_validation(f.sample(frac=1,random_state=3),p.sample(frac=1,random_state=7),m,c)
    assert first == second == analyze(inputs)
    pd.testing.assert_frame_equal(f,before_f)
    pd.testing.assert_frame_equal(p,before_p)
    assert first["fit_call_count"] == 0
    assert first["state_occupancy"]["S0"]["count"] == 14
    assert first["state_occupancy"]["S1"]["count"] == 13
    assert first["state_ids"] == ["S0","S1","S2"]
    assert sum(x["fraction"] for x in first["state_occupancy"].values()) == pytest.approx(1)
    assert set(first["horizon_counts"]) == {"1","2","4","8","12"}
    assert first["posterior_confidence"]["overall"]["mean"] == pytest.approx(.8)


def test_only_validation_rows_participate_even_if_outside_prices_invalid(inputs):
    f,p,m,c=inputs
    extra_f=pd.concat([f.iloc[:1],f.iloc[:1]],ignore_index=True)
    extra_f["timestamp"]=pd.to_datetime(["2022-12-31T23:45Z","2025-01-01T00:00Z"])
    extra_f.loc[:,list(NAMES)]=float("inf")
    extra_p=pd.concat([p.iloc[:1],p.iloc[:1]],ignore_index=True)
    extra_p["timestamp_utc"]=pd.to_datetime(["2022-12-31T23:45Z","2025-01-01T00:00Z"])
    extra_p["close"]=-1
    result=v.analyze_validation(pd.concat([f,extra_f],ignore_index=True),pd.concat([p,extra_p],ignore_index=True),m,c)
    assert result == analyze(inputs)
    assert result["final_holdout_observations_used"] == 0


def test_end_boundary_excludes_all_windows_entering_2025(inputs):
    f,p,m,c=inputs
    p["timestamp_utc"]=pd.date_range("2024-12-31T21:00Z",periods=40,freq="15min")
    f["timestamp"]=p.timestamp_utc+v.STEP
    result=analyze(inputs)
    assert result["eligible_validation_observations"] == 11
    for h in v.FUTURE_HORIZONS:
        counts=result["horizon_counts"][str(h)]
        assert counts["observations"] == max(11-h,0)
        assert counts["exclusions"]["validation_end_boundary"] == min(h,11)
        assert counts["exclusions"]["insufficient_future_bars"] == 0


@pytest.mark.parametrize("boundary", ["gap","segment"])
def test_gaps_and_segment_boundaries_invalidate_outcomes_and_transitions(inputs, boundary):
    f,p,m,c=inputs
    if boundary == "gap":
        p=p.drop(index=20); f=f.drop(index=20)
    else:
        p.loc[20:,"continuity_segment_id"]="NEXT"
        f.loc[20:,"continuity_segment_id"]="NEXT"
    result=v.analyze_validation(f,p,m,c)
    assert result["horizon_counts"]["1"]["exclusions"]["gap_or_segment_boundary"] == 1
    assert result["horizon_counts"]["12"]["exclusions"]["gap_or_segment_boundary"] == 12
    counts=np.array(result["transitions"]["counts"])
    assert counts.sum() == len(f)-2
    probs=np.array(result["transitions"]["probabilities"])
    np.testing.assert_allclose(probs.sum(axis=1),1)


def test_formulas_match_discovery_including_unclamped_excursions(inputs):
    f,p,m,c=inputs
    f=f.iloc[:1]
    p.loc[1:,"open"]=110
    p.loc[1:,"close"]=111
    p.loc[1:,"high"]=112
    p.loc[1:,"low"]=109
    result=v.analyze_validation(f,p,m,c)
    for h in v.FUTURE_HORIZONS:
        metrics=result["states"]["S0"][str(h)]["continuous"]
        assert metrics["signed_return"]["mean"] == pytest.approx(111/100-1)
        assert metrics["absolute_return"]["mean"] == pytest.approx(abs(111/100-1))
        assert metrics["realized_volatility"]["mean"] == pytest.approx(abs(np.log(111/100)))
        assert metrics["mfe_up"]["mean"] == 12
        assert metrics["mae_down"]["mean"] == -9
        assert result["states"]["S0"][str(h)]["direction"]["UP"]["count"] == 1


@pytest.mark.parametrize("means, expected", [((3,2,1),5),((2,2,1),0),((2,1,3),0),((None,1,2),0)])
def test_literal_h1_h2_all_horizons_no_thresholds(inputs, means, expected):
    result=analyze(inputs)
    summaries=result["states"]
    for h in v.FUTURE_HORIZONS:
        for state,mean in zip(v.STATES,means):
            for metric in ("absolute_return","realized_volatility"):
                summaries[state][str(h)]["continuous"][metric]["mean"]=mean
    hypotheses=v.evaluate_hypotheses(summaries,result["comparisons"])
    for h in ("H1_MAGNITUDE","H2_VOLATILITY"):
        assert hypotheses[h]["replicated_horizons"] == expected
        assert hypotheses[h]["total_horizons"] == 5
    h3=hypotheses["H3_DIRECTION"]
    assert "NO_BINARY" in h3["assessment"]
    assert len(h3["horizons"]) == 5
    assert all(len(x)==3 for x in h3["horizons"].values())
    assert "threshold" not in json.dumps(h3)
    assert "criterion_satisfied" not in json.dumps(h3)


def test_yearly_occupancy_and_horizon_accounting(inputs):
    f,p,m,c=inputs
    p.loc[20:,"timestamp_utc"] += timedelta(days=366)
    f["timestamp"]=p.timestamp_utc+v.STEP
    result=analyze(inputs)
    for year in ("2023","2024"):
        assert sum(x["count"] for x in result["yearly_occupancy"][year].values())==20
    for h in result["horizon_counts"].values():
        assert h["observations"]+sum(h["exclusions"].values())==40
    for state in v.STATES:
        for h in result["states"][state].values():
            assert h["observation_count"]+sum(h["excluded_counts"].values())==result["state_occupancy"][state]["count"]


@pytest.mark.parametrize("column", ["win","loss","result","pnl","profit","tp","sl","mfe","mae","future_return","future_direction","y.target"])
def test_forbidden_columns_rejected(inputs,column):
    f,p,m,c=inputs
    f[column]=0
    with pytest.raises(ValueError): analyze(inputs)


@pytest.mark.parametrize("bad", ["naive","duplicate","misaligned","null_segment"])
def test_timestamp_and_continuity_validation(inputs,bad):
    f,p,m,c=inputs
    if bad=="naive": f["timestamp"]=f.timestamp.dt.tz_localize(None)
    if bad=="duplicate": f.loc[1,"timestamp"]=f.loc[0,"timestamp"]
    if bad=="misaligned": f.loc[1,"timestamp"]+=timedelta(seconds=1)
    if bad=="null_segment": f.loc[1,"continuity_segment_id"]=None
    with pytest.raises(ValueError): analyze(inputs)


def test_real_frozen_model_used_on_synthetic_validation_only(inputs,frozen):
    f,p,_,c=inputs
    result=v.analyze_validation(f,p,frozen[1],c)
    assert result["fit_call_count"] == 0
    assert result["assigned_observations"] == len(f)
    labels=frozen[1].predict(f.loc[:,NAMES].to_numpy())
    assert [result["state_occupancy"][s]["count"] for s in v.STATES] == list(np.bincount(labels,minlength=3))


def test_failed_fit_during_prediction_prevents_result(inputs):
    f,p,m,c=inputs
    m.predict=lambda *a,**kw: StandardScaler().fit(np.zeros((3,14)))
    with pytest.raises(RuntimeError,match="prohibited"): analyze(inputs)


def test_create_only_publication_and_repeat_refusal(tmp_path):
    output=tmp_path/"result.json"
    checksum=v._publish_json(output,{"synthetic":True})
    assert sha256_file(output)==checksum
    with pytest.raises(FileExistsError): v._publish_json(output,{"synthetic":False})
    with pytest.raises(FileExistsError): v.run_validation(None,output=output)
    assert sha256_file(output)==checksum


def test_reserved_attempt_prevents_data_read(tmp_path,frozen,monkeypatch):
    output=tmp_path/"result.json"
    output.with_suffix(".attempt.json").write_text("reserved")
    def forbidden(*args,**kwargs): pytest.fail("must not read observations on retry")
    monkeypatch.setattr(v,"read_validation_inputs",forbidden)
    with pytest.raises(FileExistsError): v.run_validation(None,output=output)
    assert not output.exists()


def test_no_strategy_dependencies():
    import ast
    tree=ast.parse(__import__('inspect').getsource(v))
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(x and any(t in x for t in ("strategy","execution","version_0","backtest")) for x in imports)


def test_bounded_partition_read_and_protected_roles(tmp_path):
    from test_stage6_partitions import source_fixture
    _,_,service,_,_,_,discovery,validation,final,_,_=source_fixture()
    start=pd.Timestamp("2022-12-31T23:10Z"); end=pd.Timestamp("2022-12-31T23:20Z")
    result=service.access(discovery.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ",start_timestamp=start,end_timestamp=end)
    assert len(result)==10
    assert result.timestamp_utc.ge(start).all() and result.timestamp_utc.lt(end).all()
    assert "bounded read" in service.access_history(discovery.partition_id)[-1]["reason"]
    for partition in (validation,final):
        with pytest.raises(ResearchError,match="DENIED"):
            service.access(partition.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ",start_timestamp=start,end_timestamp=end)
    for bounds in ({"start_timestamp":start},{"start_timestamp":end,"end_timestamp":start},{"start_timestamp":"2023-01-01","end_timestamp":"2024-01-01"}):
        with pytest.raises(ResearchError): service.access(discovery.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ",**bounds)


def test_predicate_filtered_lineage_checked_reads(tmp_path,inputs,monkeypatch):
    f,p,_,contract=inputs
    # Entirely synthetic files deliberately contain out-of-window rows.
    extra_f=f.iloc[:1].copy(); extra_f["timestamp"]=pd.Timestamp("2025-01-01T00:00Z")
    f=pd.concat([f,extra_f],ignore_index=True); f["source_dataset_id"]="SYNTHETIC"
    extra_p=p.iloc[:1].copy(); extra_p["timestamp_utc"]=pd.Timestamp("2025-01-01T00:00Z")
    p=pd.concat([p,extra_p],ignore_index=True)
    feature_path=tmp_path/"features.parquet"; price_path=tmp_path/"prices.parquet"
    f.to_parquet(feature_path,index=False); p.to_parquet(price_path,index=False)
    metadata=dict(provider="DUKASCOPY",source_timeframe="M15",symbol="EURUSD",source_dataset_id="SYNTHETIC",
        source_partition_fingerprint="fingerprint",source_dataset_checksum="source-checksum",
        transformation_history=[dict(step="AUTHORIZED_DISCOVERY",partition_id="SYNTHETIC-PART")])
    metadata_path=tmp_path/"metadata.json"; metadata_path.write_text(json.dumps(metadata))
    # Only this IO-helper fixture has synthetic artifact bindings; actual contract is untouched.
    binding=deepcopy(contract)
    binding["feature_artifact"].update(sha256=sha256_file(feature_path),metadata_sha256=sha256_file(metadata_path))
    manifest=SimpleNamespace(role=PartitionRole.DISCOVERY,source_dataset_id="SYNTHETIC",timeframe="M15",symbol="EURUSD",
        partition_fingerprint="fingerprint",source_dataset_checksum="source-checksum",path=price_path,
        partition_checksum=sha256_file(price_path),partition_id="SYNTHETIC-PART")
    observed=[]
    original=pd.read_parquet
    def read(path,**kw):
        assert kw["filters"]
        observed.append((path,kw))
        return original(path,**kw)
    monkeypatch.setattr(pd,"read_parquet",read)
    def access(partition,context,operation,**kw):
        assert partition==manifest.partition_id
        assert kw["start_timestamp"]==v.START and kw["end_timestamp"]==v.END-v.STEP
        return pd.read_parquet(price_path,filters=[("timestamp_utc",">=",kw["start_timestamp"]),("timestamp_utc","<",kw["end_timestamp"])])
    service=SimpleNamespace(get_partition=lambda _:manifest,access=access)
    features,prices,provenance=v.read_validation_inputs(service,feature_path,binding)
    assert len(features)==40 and len(prices)==40
    assert len(observed)==2
    assert features.timestamp.lt(v.END).all() and prices.timestamp_utc.lt(v.END-v.STEP).all()
    assert provenance["provider"]=="DUKASCOPY"
    manifest.role=PartitionRole.FINAL_TEST
    with pytest.raises(ValueError,match="protected role"): v.read_validation_inputs(service,feature_path,binding)
    assert len(observed)==2
    manifest.role=PartitionRole.DISCOVERY
    metadata["provider"]="IC_MARKETS"
    metadata_path.write_text(json.dumps(metadata))
    binding["feature_artifact"]["metadata_sha256"]=sha256_file(metadata_path)
    with pytest.raises(ValueError,match="Dukascopy"): v.read_validation_inputs(service,feature_path,binding)


def test_failed_input_read_leaves_no_completed_result(tmp_path,monkeypatch):
    output=tmp_path/"result.json"
    def fail(*args): raise ValueError("synthetic input corruption")
    monkeypatch.setattr(v,"read_validation_inputs",fail)
    with pytest.raises(ValueError,match="corruption"): v.run_validation(None,output=output)
    assert not output.exists()
    manifest=json.loads(output.with_suffix(".attempt.json").read_text())
    assert manifest["status"]=="ATTEMPT_RESERVED_NOT_COMPLETION"
    assert manifest["model_metadata"]["versions"]
    assert manifest["model_sha256"]==v.EXPECTED_MODEL_SHA256


def test_synthetic_end_to_end_orchestration_publishes_deterministically(tmp_path,inputs,monkeypatch):
    output=tmp_path/"result.json"
    feature_path=tmp_path/"synthetic-feature-binding"
    original_hash=v.sha256_file
    def artifact_hash(path):
        if Path(path)==feature_path: return inputs[3]["feature_artifact"]["sha256"]
        if Path(path)==feature_path.with_name("metadata.json"): return inputs[3]["feature_artifact"]["metadata_sha256"]
        return original_hash(path)
    monkeypatch.setattr(v,"sha256_file",artifact_hash)
    monkeypatch.setattr(v,"read_validation_inputs",lambda *args:(inputs[0],inputs[1],{"provider":"DUKASCOPY","synthetic":True}))
    result,checksum=v.run_validation(None,feature_path=feature_path,output=output)
    assert sha256_file(output)==checksum
    assert json.loads(output.read_text())==result
    assert result["fit_call_count"]==0
    assert result["frozen_model_sha256"]==v.EXPECTED_MODEL_SHA256
    content=deepcopy(result); checksum_content=content.pop("result_content_sha256")
    assert hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()==checksum_content
    with pytest.raises(FileExistsError): v.run_validation(None,output=output)


def test_no_favorable_horizon_selection(inputs):
    result=analyze(inputs)
    for h in v.FUTURE_HORIZONS:
        for state,mean in zip(v.STATES,(3,2,1)):
            for metric in ("absolute_return","realized_volatility"):
                result["states"][state][str(h)]["continuous"][metric]["mean"]=mean
    result["states"]["S0"]["12"]["continuous"]["absolute_return"]["mean"]=0
    result["states"]["S0"]["12"]["continuous"]["realized_volatility"]["mean"]=0
    hypotheses=v.evaluate_hypotheses(result["states"],result["comparisons"])
    assert hypotheses["H1_MAGNITUDE"]["replication"]=="4 / 5"
    assert hypotheses["H2_VOLATILITY"]["replication"]=="4 / 5"
    assert hypotheses["H1_MAGNITUDE"]["horizons"]["12"]["criterion_satisfied"]=="NO"
