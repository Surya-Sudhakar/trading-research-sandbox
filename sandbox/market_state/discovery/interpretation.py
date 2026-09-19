"""Outcome-blind interpretation of the saved frozen M15 discovery features.

State IDs describe seed 42 only; they do not imply semantic labels or alignment
between component counts. Durations are observed bars * 15, not wall-clock spans.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from sandbox.market_state.universal.market_state_v1 import (
    MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_FEATURE_SCHEMA_VERSION,
)
from sandbox.market_state.universal.state_vector import validate_discovery_columns_v1
from .gmm import GaussianMixtureStateModel
from .scaler import DiscoveryStandardScaler, DISCOVERY_START_UTC, DISCOVERY_END_UTC
from .stability import DEFAULT_STABILITY_SEEDS

COMPONENT_COUNTS = (3, 4, 5)
PROFILE_SEED = 42
DEFAULT_FEATURE_PATH = Path(
    "data/derived/features/EURUSD/M15/"
    "adac311828c7e0bcf58158269069689cf7178d3720805629eed481525eacd19d/features.parquet"
)
DEFAULT_OUTPUT_PATH = Path("results/market_state/state_interpretation_v1.json")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Validate the whole input schema before selecting features or dates.
    validate_discovery_columns_v1(frame.columns)
    forbidden = {"win", "loss", "result", "pnl", "profit", "tp", "sl", "mfe", "mae", "future"}
    for column in frame.columns:
        name = str(column).strip().lower()
        if name.startswith("y.") or forbidden.intersection(re.split(r"[^a-z0-9]+", name)):
            raise ValueError(f"forbidden outcome/future/trading field: {column}")
    if frame.columns.duplicated().any():
        raise ValueError("duplicate input columns")
    missing = {"timestamp", *MARKET_STATE_FEATURE_NAMES_V1} - set(frame.columns)
    if missing:
        raise ValueError("missing frozen discovery columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("no complete frozen discovery rows in 2016-2022")
    stamps = []
    for value in frame["timestamp"]:
        if isinstance(value, (int, float, np.number)):
            raise ValueError("timestamps must be timezone-aware datetimes")
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("timestamps must be nonmissing and timezone-aware")
        stamps.append(stamp.tz_convert("UTC"))
    result = frame.copy()
    result["timestamp"] = pd.DatetimeIndex(stamps)
    result = result.loc[(result.timestamp >= DISCOVERY_START_UTC) &
                        (result.timestamp < DISCOVERY_END_UTC)].sort_values("timestamp", kind="stable")
    if result.timestamp.duplicated().any():
        raise ValueError("duplicate discovery timestamps")
    for column, expected in (("symbol", "EURUSD"), ("decision_timeframe", "M15")):
        if column in result and not result[column].eq(expected).all():
            raise ValueError(f"expected {column}={expected}")
    breaks = result.timestamp.diff().ne(timedelta(minutes=15))
    if "continuity_segment_id" in result:
        segments = result.continuity_segment_id
        if segments.isna().any():
            raise ValueError("missing continuity segment")
        breaks |= segments.ne(segments.shift())
    result["_interpretation_block"] = breaks.cumsum()
    result = result.dropna(subset=list(MARKET_STATE_FEATURE_NAMES_V1)).reset_index(drop=True)
    if result.empty:
        raise ValueError("no complete frozen discovery rows in 2016-2022")
    if not np.isfinite(result.loc[:, MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)).all():
        raise ValueError("non-finite frozen feature values")
    return result


def _summarize(frame: pd.DataFrame, labels: np.ndarray, n_components: int) -> dict:
    values = frame.loc[:, MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)
    connected = (frame.timestamp.diff().eq(timedelta(minutes=15)) &
                 frame._interpretation_block.eq(frame._interpretation_block.shift())).to_numpy()
    transitions = np.zeros((n_components, n_components), dtype=np.int64)
    np.add.at(transitions, (labels[:-1][connected[1:]], labels[1:][connected[1:]]), 1)
    totals = transitions.sum(axis=1)
    probabilities = np.divide(transitions, totals[:, None],
                              out=np.zeros_like(transitions, dtype=float), where=totals[:, None] != 0)
    starts = np.flatnonzero(~connected | np.r_[True, labels[1:] != labels[:-1]])
    lengths = np.diff(np.r_[starts, len(labels)])
    states = {}
    for state in range(n_components):
        selected = values[labels == state]
        runs = lengths[labels[starts] == state]
        profiles = {}
        for index, name in enumerate(MARKET_STATE_FEATURE_NAMES_V1):
            column = selected[:, index]
            profiles[name] = {
                "mean": float(np.mean(column)) if len(column) else None,
                "median": float(np.median(column)) if len(column) else None,
                "p10": float(np.quantile(column, .1)) if len(column) else None,
                "p90": float(np.quantile(column, .9)) if len(column) else None,
            }
        states[f"S{state}"] = {
            "observation_count": len(selected), "occupancy_fraction": len(selected) / len(values),
            "features": profiles,
            "persistence": {
                "run_count": len(runs),
                "mean_run_length_bars": float(np.mean(runs)) if len(runs) else None,
                "median_run_length_bars": float(np.median(runs)) if len(runs) else None,
                "p90_run_length_bars": float(np.quantile(runs, .9)) if len(runs) else None,
                "maximum_run_length_bars": int(np.max(runs)) if len(runs) else None,
                "mean_duration_minutes": float(np.mean(runs) * 15) if len(runs) else None,
                "self_transition_probability": float(probabilities[state, state]) if totals[state] else None,
            },
        }
    yearly = {}
    years = frame.timestamp.dt.year.to_numpy()
    for year in range(2016, 2023):
        subset = labels[years == year]
        counts = np.bincount(subset, minlength=n_components)
        yearly[str(year)] = {
            "observation_count": len(subset),
            "states": {f"S{state}": {"observation_count": int(counts[state]),
                       "occupancy_fraction": float(counts[state] / len(subset)) if len(subset) else None}
                       for state in range(n_components)},
        }
    return {"states": states, "yearly_occupancy": yearly,
            "transitions": {"state_order": list(states), "counts": transitions.tolist(),
                            "probabilities": probabilities.tolist()}}


def summarize_state_assignments(frame: pd.DataFrame, labels, n_components: int) -> dict:
    """Describe aligned assignments, sorting/filtering rows and labels together."""
    assignments = np.asarray(labels)
    if (n_components not in COMPONENT_COUNTS or assignments.shape != (len(frame),)
            or assignments.dtype.kind not in "iu" or np.any(assignments < 0)
            or np.any(assignments >= n_components)):
        raise ValueError("valid aligned integer state assignments for 3, 4 or 5 states required")
    prepared = _prepare_frame(frame.assign(_state_assignment=assignments))
    return _summarize(prepared, prepared._state_assignment.to_numpy(), n_components)


def pairwise_seed_stability(assignments: dict[int, np.ndarray]) -> dict:
    """All ten unordered comparisons; ARI is invariant to component numbering."""
    if set(assignments) != set(DEFAULT_STABILITY_SEEDS):
        raise ValueError("exactly the five fixed stability seeds required")
    arrays = [np.asarray(assignments[seed]) for seed in DEFAULT_STABILITY_SEEDS]
    if any(a.ndim != 1 or not len(a) or a.dtype.kind not in "iu" or a.shape != arrays[0].shape
           for a in arrays):
        raise ValueError("seed assignments must be aligned nonempty integer vectors")
    pairs = [{"seed_a": a, "seed_b": b,
              "ari": float(adjusted_rand_score(assignments[a], assignments[b]))}
             for a, b in combinations(DEFAULT_STABILITY_SEEDS, 2)]
    scores = [pair["ari"] for pair in pairs]
    return {"pairwise_aris": pairs, "mean_pairwise_ari": float(np.mean(scores)),
            "median_pairwise_ari": float(np.median(scores)),
            "minimum_pairwise_ari": min(scores), "maximum_pairwise_ari": max(scores)}


def analyze_state_interpretation(frame: pd.DataFrame) -> dict:
    prepared = _prepare_frame(frame)
    values = prepared.loc[:, MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)
    scaler = DiscoveryStandardScaler().fit(values, prepared.timestamp.to_list())
    scaled = scaler.transform(values)
    models = {}
    for count in COMPONENT_COUNTS:
        assignments, diagnostics = {}, {}
        for seed in DEFAULT_STABILITY_SEEDS:
            model = GaussianMixtureStateModel(count, random_state=seed).fit(scaled)
            assignments[seed] = model.predict(scaled)
            diagnostics[str(seed)] = asdict(model.diagnostics(scaled))
        models[str(count)] = {
            "n_components": count, "profile_seed": PROFILE_SEED,
            **_summarize(prepared, assignments[PROFILE_SEED], count),
            "seed_stability": pairwise_seed_stability(assignments),
            "seed_diagnostics": diagnostics,
        }
    return {
        "experiment": "STATE_INTERPRETATION_V1", "outcome_blind": True,
        "schema_version": MARKET_STATE_FEATURE_SCHEMA_VERSION,
        "feature_names": list(MARKET_STATE_FEATURE_NAMES_V1),
        "discovery_start_inclusive": DISCOVERY_START_UTC.isoformat(),
        "discovery_end_exclusive": DISCOVERY_END_UTC.isoformat(),
        "observation_count": len(prepared),
        "first_timestamp": prepared.timestamp.iloc[0].isoformat(),
        "last_timestamp": prepared.timestamp.iloc[-1].isoformat(),
        "seeds": list(DEFAULT_STABILITY_SEEDS),
        "models": models,
    }


def write_interpretation_artifact(feature_path: Path = DEFAULT_FEATURE_PATH,
                                  output: Path = DEFAULT_OUTPUT_PATH) -> dict:
    """Read existing features only; no engine, outcome, or trading dependency."""
    feature_path, output = Path(feature_path), Path(output)
    if feature_path.resolve() == output.resolve():
        raise ValueError("output must not overwrite the feature artifact")
    with feature_path.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    result = analyze_state_interpretation(pd.read_parquet(feature_path))
    with feature_path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != fingerprint:
            raise ValueError("feature artifact changed during interpretation")
    result["feature_artifact_sha256"] = fingerprint
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(encoded)
        except BaseException:
            temporary.close()
            temporary_path.unlink()
            raise
    try:
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    write_interpretation_artifact(args.features, args.output)


if __name__ == "__main__":
    main()
