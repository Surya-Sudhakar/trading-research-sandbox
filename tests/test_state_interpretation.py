from datetime import timedelta
from itertools import combinations
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from sandbox.market_state.discovery import interpretation as subject
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1


def frame(rows=8):
    values = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2016-01-01", periods=rows, freq="15min", tz="UTC"),
        **{name: values + i * 100 for i, name in enumerate(MARKET_STATE_FEATURE_NAMES_V1)},
    })


def describe(data=None, labels=None):
    return subject.summarize_state_assignments(
        frame() if data is None else data,
        [0, 0, 1, 1, 1, 0, 2, 2] if labels is None else labels, 3)


def test_exact_frozen_profiles_use_original_unscaled_values():
    result = describe()
    assert len(MARKET_STATE_FEATURE_NAMES_V1) == 14
    for state in result["states"].values():
        assert tuple(state["features"]) == MARKET_STATE_FEATURE_NAMES_V1
    profile = result["states"]["S0"]["features"][MARKET_STATE_FEATURE_NAMES_V1[1]]
    assert profile == {"mean": 102., "median": 101., "p10": 100.2, "p90": 104.2}
    assert sum(s["observation_count"] for s in result["states"].values()) == 8
    assert sum(s["occupancy_fraction"] for s in result["states"].values()) == pytest.approx(1)


def test_run_lengths_and_durations():
    persistence = describe()["states"]["S0"]["persistence"]
    assert persistence == {
        "run_count": 2, "mean_run_length_bars": 1.5, "median_run_length_bars": 1.5,
        "p90_run_length_bars": 1.9, "maximum_run_length_bars": 2,
        "mean_duration_minutes": 22.5, "self_transition_probability": 1 / 3,
    }


def test_transition_counts_and_probabilities():
    transitions = describe()["transitions"]
    assert transitions["state_order"] == ["S0", "S1", "S2"]
    assert transitions["counts"] == [[1, 1, 1], [1, 2, 0], [0, 0, 1]]
    np.testing.assert_allclose(np.asarray(transitions["probabilities"]).sum(axis=1), 1)
    assert transitions["probabilities"][0] == [1/3, 1/3, 1/3]


@pytest.mark.parametrize("boundary", ["gap", "segment", "weekend"])
def test_discontinuities_break_runs_and_do_not_create_transitions(boundary):
    data = frame(4)
    if boundary == "segment":
        data["continuity_segment_id"] = [4, 4, 5, 5]
    else:
        data.loc[2:, "timestamp"] += timedelta(minutes=15) if boundary == "gap" else timedelta(days=2)
    result = describe(data, [0, 0, 0, 0])
    assert result["states"]["S0"]["persistence"]["run_count"] == 2
    assert result["states"]["S0"]["persistence"]["maximum_run_length_bars"] == 2
    assert result["transitions"]["counts"] == [[2, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert result["transitions"]["probabilities"][1] == [0, 0, 0]
    assert result["states"]["S1"]["persistence"]["self_transition_probability"] is None


def test_yearly_occupancy_and_absent_years():
    data = frame(14)
    data["timestamp"] = pd.to_datetime([f"{year}-01-01T00:{minute:02d}:00Z"
                                        for year in range(2016, 2023) for minute in (0, 15)])
    result = describe(data, [0, 1] * 7)
    assert list(result["yearly_occupancy"]) == [str(y) for y in range(2016, 2023)]
    for year in result["yearly_occupancy"].values():
        assert year["observation_count"] == 2
        assert year["states"]["S0"] == {"observation_count": 1, "occupancy_fraction": .5}
        assert year["states"]["S2"] == {"observation_count": 0, "occupancy_fraction": 0}
    absent = describe()["yearly_occupancy"]["2022"]
    assert absent["observation_count"] == 0
    assert absent["states"]["S0"]["occupancy_fraction"] is None


def test_sorting_keeps_labels_aligned_and_normalizes_timezone():
    data = frame()
    labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
    order = [7, 1, 0, 4, 3, 2, 5, 6]
    shuffled = data.iloc[order].copy()
    shuffled["timestamp"] = shuffled.timestamp.dt.tz_convert("America/New_York")
    assert describe(shuffled, labels[order]) == describe(data, labels)


@pytest.mark.parametrize("problem", ["naive", "missing", "duplicate", "numeric"])
def test_invalid_timestamps_rejected(problem):
    data = frame()
    if problem == "naive":
        data["timestamp"] = data.timestamp.dt.tz_localize(None)
    elif problem == "missing":
        data.loc[1, "timestamp"] = pd.NaT
    elif problem == "duplicate":
        data.loc[1, "timestamp"] = data.timestamp.iloc[0]
    else:
        data["timestamp"] = np.arange(len(data))
    with pytest.raises(ValueError, match="timestamp"):
        describe(data)


def test_discovery_boundary_and_incomplete_rows_do_not_bridge_runs():
    data = frame(5)
    data.loc[0, "timestamp"] = pd.Timestamp("2015-12-31T23:45Z")
    data.loc[4, "timestamp"] = pd.Timestamp("2023-01-01T00:00Z")
    data.loc[2, MARKET_STATE_FEATURE_NAMES_V1[0]] = np.nan
    result = describe(data, [0] * 5)
    assert result["states"]["S0"]["observation_count"] == 2
    assert result["states"]["S0"]["persistence"]["run_count"] == 2
    assert np.asarray(result["transitions"]["counts"]).sum() == 0


@pytest.mark.parametrize("column", ["win", "loss", "result", "pnl", "profit", "tp", "sl", "mfe", "mae",
                                    "future_return", "future_direction", " PnL ", "y.h4.close_return_fraction"])
def test_forbidden_columns_rejected_before_projection_or_date_filtering(column):
    data = frame()
    data[column] = None
    with pytest.raises(ValueError, match="forbidden"):
        subject.analyze_state_interpretation(data)


@pytest.mark.parametrize("problem", ["missing_feature", "infinite", "duplicate_column", "missing_segment", "symbol", "timeframe"])
def test_invalid_feature_input_rejected(problem):
    data = frame()
    if problem == "missing_feature":
        data = data.drop(columns=MARKET_STATE_FEATURE_NAMES_V1[-1])
    elif problem == "infinite":
        data.loc[0, MARKET_STATE_FEATURE_NAMES_V1[0]] = np.inf
    elif problem == "duplicate_column":
        data = pd.concat([data, data[[MARKET_STATE_FEATURE_NAMES_V1[0]]]], axis=1)
    elif problem == "missing_segment":
        data["continuity_segment_id"] = None
    else:
        data["symbol" if problem == "symbol" else "decision_timeframe"] = "WRONG"
    with pytest.raises(ValueError):
        subject.analyze_state_interpretation(data)


def test_all_ten_seed_comparisons_and_aggregate_statistics():
    rng = np.random.default_rng(3)
    assignments = {seed: rng.integers(0, 3, 30) for seed in subject.DEFAULT_STABILITY_SEEDS}
    result = subject.pairwise_seed_stability(assignments)
    pairs = list(combinations(subject.DEFAULT_STABILITY_SEEDS, 2))
    assert [(p["seed_a"], p["seed_b"]) for p in result["pairwise_aris"]] == pairs
    expected = [adjusted_rand_score(assignments[a], assignments[b]) for a, b in pairs]
    assert len(expected) == 10
    assert [p["ari"] for p in result["pairwise_aris"]] == expected
    assert result["mean_pairwise_ari"] == np.mean(expected)
    assert result["median_pairwise_ari"] == np.median(expected)
    assert result["minimum_pairwise_ari"] == min(expected)
    assert result["maximum_pairwise_ari"] == max(expected)


def test_real_models_deterministic_artifact_and_exact_fit_inputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    data = frame(120)
    for i, name in enumerate(MARKET_STATE_FEATURE_NAMES_V1):
        data[name] = (np.arange(len(data)) % 5) * 5 + rng.normal(0, .25, len(data)) + i * .1
    data["unrelated_context"] = 999
    original_fit = subject.GaussianMixtureStateModel.fit
    calls = []
    def fit(model, values):
        assert values.shape == (120, 14)
        calls.append(model.n_components)
        return original_fit(model, values)
    monkeypatch.setattr(subject.GaussianMixtureStateModel, "fit", fit)
    source, destination = tmp_path / "features.parquet", tmp_path / "interpretation.json"
    data.to_parquet(source, index=False)
    first = subject.write_interpretation_artifact(source, destination)
    encoded = destination.read_bytes()
    second = subject.write_interpretation_artifact(source, destination)
    assert first == second == json.loads(encoded)
    assert encoded == destination.read_bytes()
    assert calls == [3] * 5 + [4] * 5 + [5] * 5 + [3] * 5 + [4] * 5 + [5] * 5
    assert list(first["models"]) == ["3", "4", "5"]
    assert first["feature_names"] == list(MARKET_STATE_FEATURE_NAMES_V1)
    for count, model in first["models"].items():
        assert model["profile_seed"] == 42
        assert list(model["states"]) == [f"S{i}" for i in range(int(count))]
        assert len(model["seed_stability"]["pairwise_aris"]) == 10
        assert sum(s["observation_count"] for s in model["states"].values()) == 120
        assert sum(s["occupancy_fraction"] for s in model["states"].values()) == pytest.approx(1)


def test_input_change_during_analysis_does_not_publish(tmp_path, monkeypatch):
    source, output = tmp_path / "features.parquet", tmp_path / "result.json"
    frame().to_parquet(source)
    output.write_text("previous completed artifact")
    def changed(_):
        frame(9).to_parquet(source)
        return {"outcome_blind": True}
    monkeypatch.setattr(subject, "analyze_state_interpretation", changed)
    with pytest.raises(ValueError, match="changed during"):
        subject.write_interpretation_artifact(source, output)
    assert output.read_text() == "previous completed artifact"


def test_atomic_publish_failure_preserves_existing_output(tmp_path, monkeypatch):
    source, output = tmp_path / "features.parquet", tmp_path / "result.json"
    frame().to_parquet(source)
    output.write_text("previous completed artifact")
    monkeypatch.setattr(subject, "analyze_state_interpretation", lambda _: {"outcome_blind": True})
    def fail(*args):
        raise OSError("publish failed")
    monkeypatch.setattr(subject.os, "replace", fail)
    with pytest.raises(OSError, match="publish failed"):
        subject.write_interpretation_artifact(source, output)
    assert output.read_text() == "previous completed artifact"
    assert {p.name for p in tmp_path.iterdir()} == {source.name, output.name}


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="no complete"):
        subject.analyze_state_interpretation(frame(0))
