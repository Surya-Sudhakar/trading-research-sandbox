from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sandbox.market_state.discovery.gmm import GaussianMixtureStateModel
from sandbox.market_state.discovery.scaler import DiscoveryStandardScaler, DISCOVERY_START_UTC, DISCOVERY_END_UTC
from sandbox.market_state.discovery.model import StateDiscoveryModel
from sandbox.market_state.discovery.frozen_model import FrozenMarketStateModel, discovery_input_checksum, read_discovery_features
from sandbox.market_state.discovery import parameter_artifact as io
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(17)
    x = np.vstack([rng.normal(center,.2,(35,14)) for center in (-4,0,4)])
    stamps = tuple(datetime(2018,1,1,tzinfo=timezone.utc)+i*timedelta(minutes=15) for i in range(len(x)))
    scaler = DiscoveryStandardScaler().fit(x, stamps)
    model = GaussianMixtureStateModel(3,random_state=42).fit(scaler.transform(x))
    provenance = {
        "feature_artifact_sha256":"a"*64,"discovery_input_sha256":discovery_input_checksum(x,stamps),
        "discovery_artifact_sha256":"b"*64,"discovery_start":DISCOVERY_START_UTC.isoformat(),
        "discovery_end_exclusive":DISCOVERY_END_UTC.isoformat(),"discovery_observation_count":len(x),
        "first_observation":stamps[0].isoformat(),"last_observation":stamps[-1].isoformat(),
    }
    return x,stamps,scaler,model,provenance


def saved(tmp_path, fitted):
    x,t,s,m,p = fitted
    bundle = FrozenMarketStateModel(s,m,p)
    path = tmp_path/"model.json"
    checksum = bundle.save(path)
    kwargs = dict(artifact_sha256=checksum,feature_artifact_sha256=p["feature_artifact_sha256"],discovery_input_sha256=p["discovery_input_sha256"])
    return path,kwargs


def test_bundle_roundtrip_exact_and_no_refit(tmp_path,fitted,monkeypatch):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    x,_,s,m,_ = fitted
    before_z,before_labels,before_probs = s.transform(x),m.predict(s.transform(x)),m.predict_proba(s.transform(x))
    path,kwargs = saved(tmp_path,fitted)
    def fail(*args,**kwargs):
        pytest.fail("load must never fit")
    for cls in (GaussianMixtureStateModel,DiscoveryStandardScaler,GaussianMixture,StandardScaler):
        monkeypatch.setattr(cls,"fit",fail)
    loaded = FrozenMarketStateModel.load(path,**kwargs)
    np.testing.assert_array_equal(loaded.transform(x),before_z)
    np.testing.assert_array_equal(loaded.predict(x),before_labels)
    np.testing.assert_array_equal(loaded.predict_proba(x),before_probs)
    assert not hasattr(loaded,"fit")
    assert loaded.metadata["state_ids"] == ["S0","S1","S2"]
    assert loaded.metadata["gmm"]["configuration"]["random_state"] == 42
    assert loaded.metadata["gmm"]["configuration"]["n_components"] == 3
    assert not loaded._model._model.means_.flags.writeable
    loaded.metadata["state_ids"].reverse()
    assert loaded.metadata["state_ids"] == ["S0","S1","S2"]


def test_standalone_save_load_conforms_protocol(tmp_path,fitted,monkeypatch):
    x,_,s,m,_ = fitted
    assert isinstance(m,StateDiscoveryModel)
    m.save(tmp_path/"gmm.json")
    s.save(tmp_path/"scaler.json")
    def fail(*args,**kwargs):
        pytest.fail("refit attempted")
    monkeypatch.setattr(GaussianMixtureStateModel,"fit",fail)
    monkeypatch.setattr(DiscoveryStandardScaler,"fit",fail)
    restored_s = DiscoveryStandardScaler.load(tmp_path/"scaler.json")
    restored_m = GaussianMixtureStateModel.load(tmp_path/"gmm.json")
    np.testing.assert_array_equal(restored_s.transform(x),s.transform(x))
    np.testing.assert_array_equal(restored_m.predict(s.transform(x)),m.predict(s.transform(x)))
    assert restored_s.metadata == s.metadata


def test_deterministic_serialization_and_overwrite_refusal(tmp_path,fitted):
    path,kwargs = saved(tmp_path,fitted)
    restored = FrozenMarketStateModel.load(path,**kwargs)
    assert restored.save(path) == kwargs["artifact_sha256"]
    other = tmp_path/"copy.json"
    assert restored.save(other) == kwargs["artifact_sha256"]
    assert other.read_bytes() == path.read_bytes()
    other.write_text("different")
    with pytest.raises(FileExistsError):
        restored.save(other)
    assert other.read_text() == "different"


@pytest.mark.parametrize("field,value",[
    ("artifact_sha256","0"*64),("feature_artifact_sha256","0"*64),("discovery_input_sha256","0"*64),
    ("feature_names",tuple(reversed(MARKET_STATE_FEATURE_NAMES_V1))),("schema_version","OTHER"),
])
def test_expected_identity_mismatch_rejected(tmp_path,fitted,field,value):
    path,kwargs = saved(tmp_path,fitted)
    kwargs[field] = value
    with pytest.raises(ValueError,match="mismatch"):
        FrozenMarketStateModel.load(path,**kwargs)


@pytest.mark.parametrize("kind",["truncated","changed","missing_array","bad_shape","invalid_weights","schema","order","versions","missing_metadata"])
def test_corrupt_or_incomplete_artifact_rejected(tmp_path,fitted,kind):
    path,kwargs = saved(tmp_path,fitted)
    if kind == "truncated":
        path.write_bytes(path.read_bytes()[:50])
    else:
        blob = json.loads(path.read_text())
        state = blob["state"]
        if kind == "changed":
            state["gmm"]["arrays"]["means"][0][0] += 1
        elif kind == "missing_array":
            del state["gmm"]["arrays"]["precisions_cholesky"]
        elif kind == "bad_shape":
            state["scaler"]["mean"] = [1]
        elif kind == "invalid_weights":
            state["gmm"]["arrays"]["weights"] = [-1,1,1]
        elif kind == "schema":
            state["feature_schema_version"] = "OTHER"
        elif kind == "order":
            state["feature_names"].reverse()
        elif kind == "versions":
            state["versions"]["scikit_learn"] = "0"
        else:
            del state["provenance"]["discovery_start"]
        # Also exercise semantic validation after checksums were recomputed.
        if kind != "changed":
            blob["payload_sha256"] = io.digest(state)
        path.write_text(json.dumps(blob))
    kwargs["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        FrozenMarketStateModel.load(path,**kwargs)


def test_wrong_inference_order_rejected(tmp_path,fitted):
    path,kwargs = saved(tmp_path,fitted)
    restored = FrozenMarketStateModel.load(path,**kwargs)
    with pytest.raises(ValueError,match="order"):
        restored.predict(fitted[0],feature_names=tuple(reversed(MARKET_STATE_FEATURE_NAMES_V1)))


def test_checksum_binds_values_and_timestamps(fitted):
    x,t,*_ = fitted
    baseline = discovery_input_checksum(x,t)
    changed = x.copy();changed[0,0] += 1
    assert discovery_input_checksum(changed,t) != baseline
    assert discovery_input_checksum(x,[v+timedelta(minutes=15) for v in t]) != baseline
    with pytest.raises(ValueError,match="Discovery"):
        discovery_input_checksum(x,[DISCOVERY_END_UTC+timedelta(minutes=15*i) for i in range(len(x))])


def test_parquet_read_filters_before_returning_any_2023_plus_rows(tmp_path,monkeypatch):
    stamps = pd.to_datetime(["2015-12-31T23:45Z","2016-01-01T00:00Z","2022-12-31T23:45Z","2023-01-01T00:00Z","2025-01-01T00:00Z"])
    frame = pd.DataFrame({"timestamp":stamps,**{f:np.arange(5,dtype=float) for f in MARKET_STATE_FEATURE_NAMES_V1}})
    path = tmp_path/"features.parquet";frame.to_parquet(path,index=False)
    read = pd.read_parquet
    calls = []
    def checked(*args,**kwargs):
        assert kwargs["filters"] == [("timestamp",">=",DISCOVERY_START_UTC),("timestamp","<",DISCOVERY_END_UTC)]
        result = read(*args,**kwargs)
        assert len(result)==2 and result.timestamp.dt.year.tolist()==[2016,2022]
        calls.append(1)
        return result
    monkeypatch.setattr(pd,"read_parquet",checked)
    result = read_discovery_features(path)
    assert len(result)==2 and calls==[1]


def test_forbidden_schema_rejected_before_parquet_read(tmp_path,monkeypatch):
    frame = pd.DataFrame({"timestamp":pd.to_datetime(["2016-01-01T00:00Z"]),"future_return":[1.]})
    path = tmp_path/"features.parquet";frame.to_parquet(path,index=False)
    monkeypatch.setattr(pd,"read_parquet",lambda *a,**k:pytest.fail("must reject schema before reading rows"))
    with pytest.raises(ValueError,match="forbidden"):
        read_discovery_features(path)


def test_unfitted_save_rejected(tmp_path):
    with pytest.raises(RuntimeError,match="not been fitted"):
        GaussianMixtureStateModel(3).save(tmp_path/"gmm.json")
    with pytest.raises(RuntimeError,match="not been fitted"):
        DiscoveryStandardScaler().save(tmp_path/"scaler.json")
