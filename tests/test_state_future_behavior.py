from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sandbox.market_state.discovery import future_behavior as fb
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1


def prices(count=16, start="2016-01-04T00:00:00Z"):
    close = 100 + np.arange(count, dtype=float)
    return pd.DataFrame(dict(timestamp_utc=pd.date_range(start, periods=count, freq="15min"),
                             open=close, high=close+2, low=close-2, close=close,
                             continuity_segment_id=0, tick_volume=1, spread=0, real_volume=0))


def features(count=90):
    rng = np.random.default_rng(8)
    data = {"timestamp": prices(count).timestamp_utc + fb.STEP, "continuity_segment_id": [0]*count}
    for i, name in enumerate(MARKET_STATE_FEATURE_NAMES_V1):
        data[name] = (np.arange(count) % 3) * 4 + rng.normal(0, .25, count) + i / 10
    return pd.DataFrame(data)


def assigned(data, states=None):
    stamps = tuple(data.timestamp_utc + fb.STEP)
    return fb.AssignedStates(stamps, tuple(states or ["S0"]*len(data)), tuple(data.continuity_segment_id),
                             stamps[0].isoformat(), stamps[-1].isoformat(), True)


def bars(data):
    return tuple(OHLCBar(r.timestamp_utc.to_pydatetime(), r.open, r.high, r.low, r.close) for r in data.itertuples())


def test_frozen_horizons():
    assert fb.FUTURE_HORIZONS == (1, 2, 4, 8, 12)
    assert tuple(h*15 for h in fb.FUTURE_HORIZONS) == (15, 30, 60, 120, 180)


def test_exact_formulas_reference_excludes_anchor_high_and_low():
    data = prices(3)
    data.loc[0, ["high", "low"]] = [200, 1]
    data.loc[1, ["open", "high", "low", "close"]] = [101, 104, 98, 102]
    data.loc[2, ["open", "high", "low", "close"]] = [102, 105, 99, 101]
    b = bars(data)
    outcome = fb.measure_window(b[0], b[1:], (1, 2))[2]
    assert outcome["signed_return"] == pytest.approx(.01)
    assert outcome["absolute_return"] == pytest.approx(.01)
    assert outcome["close_change"] == 1
    assert outcome["mfe_up"] == 5
    assert outcome["mae_down"] == 2
    assert outcome["realized_volatility"] == pytest.approx(np.sqrt(np.log(102/100)**2 + np.log(101/102)**2))
    assert outcome["direction"] == "UP"
    assert not any(name.endswith("_pips") for name in outcome)


@pytest.mark.parametrize("change,direction", [(1,"UP"), (-1,"DOWN"), (0,"FLAT")])
def test_direction_and_absolute_return(change, direction):
    data = prices(2)
    data.loc[1, ["open", "close"]] = 100+change
    data.loc[1, "low"] = 98
    b = bars(data)
    outcome = fb.measure_window(b[0], b[1:], (1,))[1]
    assert outcome["direction"] == direction
    assert outcome["absolute_return"] == pytest.approx(abs(change/100))


def test_unclamped_excursions_do_not_change_existing_engine_semantics():
    from sandbox.market_state.future_outcome import OutcomeAnchor, measure_forward_outcome
    data = prices(2)
    data.loc[1, ["open", "high", "low", "close"]] = [110, 112, 109, 111]
    b = bars(data)
    assert fb.measure_window(b[0], b[1:], (1,))[1]["mae_down"] == -9
    assert measure_forward_outcome(OutcomeAnchor("A", b[1].open_time_utc, 100), b[1:], 1).max_downward_excursion == 0


def test_documented_pip_metadata_only():
    b = bars(prices(2))
    result = fb.measure_window(b[0], b[1:], (1,), pip_metadata={"symbol":"EURUSD", "pip_size":.01, "source":"test symbol metadata"})[1]
    assert result["close_change_pips"] == 100
    with pytest.raises(ValueError, match="pip metadata"):
        fb.measure_window(b[0], b[1:], (1,), pip_metadata={"pip_size":.0001})


@pytest.mark.parametrize("kind", ["missing", "weekend", "segment"])
def test_gaps_break_windows_per_horizon(kind):
    data = prices(16)
    states = assigned(data)
    if kind == "missing":
        data = data.drop(index=2)
    elif kind == "weekend":
        data.loc[2:, "timestamp_utc"] += timedelta(days=2)
    else:
        data.loc[2:, "continuity_segment_id"] = 1
        states = assigned(data)
    result = fb.analyze_assigned_states(replace(states, timestamps=states.timestamps[:1], states=("S0",), segments=(0,)), data)
    assert result["states"]["S0"]["1"]["observation_count"] == 1
    for h in (2, 4, 8, 12):
        assert result["states"]["S0"][str(h)]["observation_count"] == 0
        assert result["states"]["S0"][str(h)]["excluded_counts"]["gap_or_segment_boundary"] == 1


def test_end_of_data_counts_and_summaries():
    data = prices()
    result = fb.analyze_assigned_states(assigned(data), data)
    for h in fb.FUTURE_HORIZONS:
        summary = result["states"]["S0"][str(h)]
        assert summary["observation_count"] == 16-h
        assert summary["excluded_counts"]["end_of_discovery_data"] == h
        assert summary["direction"]["UP"] == {"count":16-h, "fraction":1.}
        assert sum(d["count"] for d in summary["direction"].values()) == 16-h
        assert set(summary["continuous"]["signed_return"]) == {"mean","median","standard_deviation","p10","p25","p75","p90"}
    assert result["states"]["S1"]["1"]["continuous"]["signed_return"]["mean"] is None


def test_future_discovery_boundary_is_excluded_not_borrowed_from_2023():
    data = prices(4, "2022-12-31T23:15:00Z")
    states = assigned(data.iloc[:1])
    result = fb.analyze_assigned_states(states, data)
    assert result["states"]["S0"]["1"]["observation_count"] == 1
    assert result["states"]["S0"]["2"]["observation_count"] == 0
    with pytest.raises(ValueError, match="Discovery"):
        fb.analyze_assigned_states(assigned(data), data)


@pytest.mark.parametrize("field", ["win","loss","winner","loser","profit","pnl","tp","sl","mfe","mae","future_return","future_direction","y.h4.return"])
def test_forbidden_columns_rejected(field):
    data = features()
    data[field] = 1
    with pytest.raises(ValueError, match="forbidden"):
        fb.assign_discovery_states(data)


def test_actual_model_price_mutation_determinism_no_input_mutation_and_discovery_only(monkeypatch):
    f, p = features(), prices(90)
    outside = f.iloc[[0]].copy()
    outside["timestamp"] = pd.Timestamp("2023-01-01T00:00Z")
    outside[MARKET_STATE_FEATURE_NAMES_V1[0]] = 1e20
    f = pd.concat([f, outside], ignore_index=True)
    original_f, original_p = f.copy(deep=True), p.copy(deep=True)
    calls = []
    original_fit = fb.GaussianMixtureStateModel.fit
    def fit(model, matrix):
        assert matrix.shape == (90, 14)
        assert model.n_components == 3
        assert model._model.random_state == 42
        calls.append(matrix.copy())
        return original_fit(model, matrix)
    monkeypatch.setattr(fb.GaussianMixtureStateModel, "fit", fit)
    first = fb.assign_discovery_states(f)
    second = fb.assign_discovery_states(f.sample(frac=1, random_state=2))
    assert first == second
    assert set(first.states) == {"S0","S1","S2"}
    assert all(t.year < 2023 for t in first.timestamps)
    result = fb.analyze_assigned_states(first, p)
    assert result == fb.analyze_assigned_states(second, p.sample(frac=1, random_state=3))
    changed = p.copy()
    changed.loc[1, ["open","high","low","close"]] += 10
    mutated = fb.analyze_assigned_states(first, changed)
    assert first == second  # Outcome stage cannot mutate frozen assignments.
    assert result["states"] != mutated["states"]
    before = fb.measure_window(bars(p)[0], bars(p)[1:13])[1]
    after = fb.measure_window(bars(changed)[0], bars(changed)[1:13])[1]
    assert before["signed_return"] != after["signed_return"]
    pd.testing.assert_frame_equal(f, original_f)
    pd.testing.assert_frame_equal(p, original_p)
    np.testing.assert_array_equal(calls[0], calls[1])


def test_effect_size_and_sample_statistics():
    result = fb._effect([1.,2.,3.], [2.,3.,4.])
    assert result == {"difference_in_means":-1., "difference_in_medians":-1., "cohens_d":-1.}
    assert fb._effect([1.,1.], [2.,2.])["cohens_d"] is None
    assert fb._effect([], [2.])["difference_in_means"] is None
    assert fb._statistics([1.,2.,3.])["standard_deviation"] == 1.


def artifact_fixture(tmp_path):
    from sandbox.provenance import Provenance, sha256_file
    from sandbox.research.registry import ResearchRegistry
    from sandbox.partition.service import PartitionService
    p = prices(16)
    source = tmp_path / "prices.parquet"
    p.to_parquet(source, index=False)
    registry = ResearchRegistry(tmp_path/"catalog.sqlite", project_root=Path.cwd(), results_root=tmp_path/"results")
    provenance = Provenance.create(source,"Synthetic","Test","EURUSD","M15",p.timestamp_utc.iloc[0],p.timestamp_utc.iloc[-1],len(p))
    registry.catalog.add_dataset(provenance)
    service = PartitionService(registry, tmp_path/"partitions", tmp_path/"vault")
    manifest = service.create_partition(provenance.dataset_id,"STATE_TEST","DISCOVERY","EURUSD","M15",p.timestamp_utc.iloc[0],p.timestamp_utc.iloc[-1]+fb.STEP)
    feature = tmp_path/"features.parquet"
    features(16).to_parquet(feature, index=False)
    metadata = dict(parquet_sha256=sha256_file(feature), source_dataset_id=manifest.source_dataset_id,
                    source_partition_fingerprint=manifest.partition_fingerprint, source_dataset_checksum=manifest.source_dataset_checksum,
                    transformation_history=[dict(step="AUTHORIZED_DISCOVERY", partition_id=manifest.partition_id)])
    feature.with_name("metadata.json").write_text(json.dumps(metadata))
    return service, feature, p


def test_real_partition_artifact_deterministic_atomic_and_checksum_verified(tmp_path, monkeypatch):
    service, source, p = artifact_fixture(tmp_path)
    monkeypatch.setattr(fb, "assign_discovery_states", lambda _: assigned(p))
    output = tmp_path/"result.json"
    first = fb.write_future_behavior_artifact(service, source, output)
    encoded = output.read_bytes()
    assert fb.write_future_behavior_artifact(service, source, output) == first
    assert encoded == output.read_bytes()
    assert json.loads(encoded) == first
    assert service.verify_access_ledger()
    assert first["feature_artifact_sha256"] == fb.sha256_file(source)
    assert first["state_model"]["n_components"] == 3
    def fail(*a):
        raise OSError("publish failure")
    monkeypatch.setattr(fb.os, "replace", fail)
    with pytest.raises(OSError, match="publish failure"):
        fb.write_future_behavior_artifact(service, source, output)
    assert output.read_bytes() == encoded


@pytest.mark.parametrize("role", ["VALIDATION", "FINAL_TEST"])
def test_writer_refuses_protected_partition_before_access(tmp_path, monkeypatch, role):
    service, source, p = artifact_fixture(tmp_path)
    manifest = service.get_partition(json.loads(source.with_name("metadata.json").read_text())["transformation_history"][0]["partition_id"])
    from sandbox.partition.models import PartitionRole
    monkeypatch.setattr(service, "get_partition", lambda _: manifest.model_copy(update={"role":PartitionRole(role)}))
    monkeypatch.setattr(service, "access", lambda *a, **k: pytest.fail("protected read"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        fb.write_future_behavior_artifact(service, source, tmp_path/"result.json")


def test_checksum_mismatch_preserves_output(tmp_path, monkeypatch):
    service, source, p = artifact_fixture(tmp_path)
    output = tmp_path/"result.json"
    output.write_text("previous result")
    source.write_bytes(source.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fb.write_future_behavior_artifact(service, source, output)
    assert output.read_text() == "previous result"


def test_integrated_future_mutation_cannot_change_states_or_recompute_features(monkeypatch):
    from sandbox.market_state.universal.engine import UniversalFeatureEngine
    from sandbox.execution.engine import BacktestEngine
    def prohibited(*args, **kwargs):
        pytest.fail("feature recomputation or strategy execution is prohibited")
    monkeypatch.setattr(UniversalFeatureEngine, "compute", prohibited)
    monkeypatch.setattr(BacktestEngine, "run", prohibited)
    f, p = features(60), prices(60)
    states = []
    original = fb.assign_discovery_states
    def capture(data):
        state = original(data)
        states.append(state)
        return state
    monkeypatch.setattr(fb, "assign_discovery_states", capture)
    first = fb.analyze_future_behavior(f, p)
    changed = p.copy()
    changed.loc[1, ["open", "high", "low", "close"]] += 20
    second = fb.analyze_future_behavior(f, changed)
    assert states[0] == states[1]
    assert states[0].states[0] == states[1].states[0]
    assert first["states"] != second["states"]


def test_empty_prices_and_missing_reference_are_counted():
    data = prices(2)
    result = fb.analyze_assigned_states(assigned(data), data.iloc[:0])
    for h in fb.FUTURE_HORIZONS:
        assert result["states"]["S0"][str(h)]["excluded_counts"]["missing_reference"] == 2


@pytest.mark.parametrize("problem", ["duplicate", "naive", "offgrid", "bad_ohlc", "segment_mismatch"])
def test_invalid_prices_rejected(problem):
    data = prices(16)
    state = assigned(data)
    if problem == "duplicate":
        data.loc[1, "timestamp_utc"] = data.timestamp_utc.iloc[0]
    elif problem == "naive":
        data["timestamp_utc"] = data.timestamp_utc.dt.tz_localize(None)
    elif problem == "offgrid":
        data["timestamp_utc"] += timedelta(minutes=1)
    elif problem == "bad_ohlc":
        data.loc[1, "high"] = 1
    else:
        data["continuity_segment_id"] = 1
    with pytest.raises(ValueError):
        fb.analyze_assigned_states(state, data)


def test_input_change_during_analysis_does_not_publish(tmp_path, monkeypatch):
    service, source, p = artifact_fixture(tmp_path)
    output = tmp_path/"result.json"
    output.write_text("previous result")
    monkeypatch.setattr(fb, "assign_discovery_states", lambda _: assigned(p))
    original = fb.analyze_assigned_states
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(source.read_bytes()+b"tampered")
        return result
    monkeypatch.setattr(fb, "analyze_assigned_states", changed)
    with pytest.raises(ValueError, match="input artifact changed"):
        fb.write_future_behavior_artifact(service, source, output)
    assert output.read_text() == "previous result"
