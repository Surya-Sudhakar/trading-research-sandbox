"""Discovery-only descriptive outcomes, strictly downstream of frozen-feature states.

The discovery-fitted model is retrospective, not a walk-forward/live estimator.
Feature decision timestamps are closes; price timestamps are M15 bar opens.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.future_outcome import OutcomeAnchor, measure_forward_outcomes
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1, MARKET_STATE_FEATURE_SCHEMA_VERSION
from sandbox.market_state.universal.state_vector import validate_discovery_columns_v1
from sandbox.partition.models import AccessContext, AccessOperation, PartitionRole
from sandbox.provenance import sha256_file
from .gmm import GaussianMixtureStateModel
from .interpretation import DEFAULT_FEATURE_PATH, _prepare_frame
from .scaler import DiscoveryStandardScaler, DISCOVERY_START_UTC, DISCOVERY_END_UTC

FUTURE_HORIZONS = (1, 2, 4, 8, 12)
STEP = timedelta(minutes=15)
STATES = ("S0", "S1", "S2")
METRICS = ("signed_return", "absolute_return", "mfe_up", "mae_down", "realized_volatility", "close_change")
DEFAULT_OUTPUT = Path("results/market_state/future_behavior_v1.json")


def _guard(columns):
    validate_discovery_columns_v1(columns)
    banned = {"win", "loss", "winner", "loser", "profit", "pnl", "tp", "sl", "mfe", "mae", "future", "result"}
    for column in columns:
        name = str(column).strip().lower()
        if name.startswith("y.") or banned.intersection(re.split(r"[^a-z0-9]+", name)):
            raise ValueError(f"forbidden outcome/trading field: {column}")


@dataclass(frozen=True)
class AssignedStates:
    timestamps: tuple
    states: tuple[str, ...]
    segments: tuple | None
    fit_start: str
    fit_last: str
    converged: bool
    feature_rows_in_discovery: int | None = None


def assign_discovery_states(features: pd.DataFrame) -> AssignedStates:
    """No prices or outcomes are accepted by the fitting/assignment stage."""
    _guard(features.columns)
    prepared = _prepare_frame(features)
    if not prepared.timestamp.eq(prepared.timestamp.dt.floor("15min")).all():
        raise ValueError("M15-aligned feature timestamps required")
    values = prepared.loc[:, MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)
    scaler = DiscoveryStandardScaler().fit(values, prepared.timestamp.to_list())
    standardized = scaler.transform(values)
    model = GaussianMixtureStateModel(n_components=3, random_state=42).fit(standardized)
    labels = model.predict(standardized)
    return AssignedStates(tuple(prepared.timestamp), tuple(f"S{int(x)}" for x in labels),
                          tuple(prepared.continuity_segment_id) if "continuity_segment_id" in prepared else None,
                          prepared.timestamp.iloc[0].isoformat(), prepared.timestamp.iloc[-1].isoformat(),
                          model.diagnostics(standardized).converged,
                          int(pd.to_datetime(features.timestamp, utc=True).between(
                              DISCOVERY_START_UTC, DISCOVERY_END_UTC, inclusive="left").sum()))


def _prices(prices):
    _guard(prices.columns)
    required = {"timestamp_utc", "open", "high", "low", "close"}
    if prices.columns.duplicated().any() or not required.issubset(prices.columns):
        raise ValueError("unique timestamp_utc and OHLC columns required")
    if prices.empty:
        return prices.copy(), ()
    stamps = []
    for raw in prices.timestamp_utc:
        if isinstance(raw, (int, float, np.number)):
            raise ValueError("aware price timestamps required")
        stamp = pd.Timestamp(raw)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp != stamp.floor("15min"):
            raise ValueError("aware, nonmissing M15-aligned price timestamps required")
        stamps.append(stamp.tz_convert("UTC"))
    table = prices.copy()
    table["timestamp_utc"] = pd.DatetimeIndex(stamps)
    # Neither reference nor outcome bars outside Discovery are measured.
    table = table.loc[(table.timestamp_utc >= DISCOVERY_START_UTC) &
                      (table.timestamp_utc + STEP < DISCOVERY_END_UTC)].sort_values("timestamp_utc").reset_index(drop=True)
    if table.timestamp_utc.duplicated().any():
        raise ValueError("duplicate price timestamps")
    if "symbol" in table and not table.symbol.eq("EURUSD").all():
        raise ValueError("EURUSD price input required")
    if "continuity_segment_id" in table and table.continuity_segment_id.isna().any():
        raise ValueError("missing price continuity segment")
    bars = tuple(OHLCBar(row.timestamp_utc.to_pydatetime(), row.open, row.high, row.low, row.close)
                 for row in table.itertuples())
    return table, bars


def _pip_size(metadata):
    if metadata is None:
        return None
    value = metadata.get("pip_size")
    if (metadata.get("symbol") != "EURUSD" or not isinstance(metadata.get("source"), str)
            or not metadata["source"].strip() or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
        raise ValueError("pip metadata requires EURUSD, positive pip_size and documented source")
    return float(value)


def measure_window(reference: OHLCBar, future_bars, horizons=FUTURE_HORIZONS, *, pip_metadata=None):
    """Reference bar ends at T; future bars open at T, T+15m, ... ."""
    horizons = tuple(horizons)
    if not horizons or any(h not in FUTURE_HORIZONS for h in horizons):
        raise ValueError("only frozen future horizons supported")
    pip = _pip_size(pip_metadata)
    future_bars = tuple(future_bars)
    anchor = OutcomeAnchor("STATE_CLOSE", reference.open_time_utc + STEP, reference.close)
    if any(bar.open_time_utc != anchor.anchor_time_utc + i * STEP for i, bar in enumerate(future_bars)):
        raise ValueError("future bars must begin at T and be consecutive")
    measured = measure_forward_outcomes(anchor, future_bars, horizons=horizons)
    closes = np.array([reference.close, *(bar.close for bar in future_bars)])
    log_returns = np.log(closes[1:] / closes[:-1])
    result = {}
    for outcome in measured:
        h = outcome.horizon_bars
        signed = outcome.end_close / reference.close - 1
        record = {
            "signed_return": signed, "absolute_return": abs(signed),
            # Legacy excursion fields clamp to zero; use its extrema for the
            # explicitly requested, unclamped signed differences instead.
            "mfe_up": outcome.highest_high - reference.close,
            "mae_down": reference.close - outcome.lowest_low,
            "realized_volatility": float(np.sqrt(np.sum(log_returns[:h] ** 2))),
            "close_change": outcome.close_change,
            "direction": "UP" if outcome.close_change > 0 else "DOWN" if outcome.close_change < 0 else "FLAT",
        }
        if pip is not None:
            record.update({f"{name}_pips": record[name] / pip for name in ("close_change", "mfe_up", "mae_down")})
        result[h] = record
    return result


def _statistics(values):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite calculated outcome")
    return {
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else None,
        **{f"p{q}": float(np.quantile(array, q / 100)) if len(array) else None for q in (10, 25, 75, 90)},
    }


def _effect(left, right):
    if not left or not right:
        return {"difference_in_means": None, "difference_in_medians": None, "cohens_d": None}
    a, b = np.asarray(left), np.asarray(right)
    difference = float(a.mean() - b.mean())
    pooled = (float(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
              if len(a) > 1 and len(b) > 1 else 0.)
    return {"difference_in_means": difference, "difference_in_medians": float(np.median(a)-np.median(b)),
            "cohens_d": difference / math.sqrt(pooled) if pooled > 0 else None}


def analyze_assigned_states(assigned: AssignedStates, prices: pd.DataFrame, *, pip_metadata=None) -> dict:
    """Outcome stage: never fits, transforms features, or changes state labels."""
    if (not assigned.timestamps or len(assigned.timestamps) != len(assigned.states)
            or any(s not in STATES for s in assigned.states)
            or (assigned.segments is not None and len(assigned.segments) != len(assigned.states))):
        raise ValueError("invalid state assignments")
    if any(t.tzinfo is None or not DISCOVERY_START_UTC <= t < DISCOVERY_END_UTC for t in assigned.timestamps):
        raise ValueError("state timestamps must be within Discovery")
    if any(b <= a for a, b in zip(assigned.timestamps, assigned.timestamps[1:])):
        raise ValueError("state timestamps must be strictly increasing")
    pip = _pip_size(pip_metadata)
    table, bars = _prices(prices)
    positions = {bar.open_time_utc + STEP: i for i, bar in enumerate(bars)}
    segments = tuple(table.continuity_segment_id) if "continuity_segment_id" in table else None
    if assigned.segments is not None and segments is None:
        raise ValueError("price continuity IDs required to match feature artifact")
    metric_names = METRICS + (("close_change_pips", "mfe_up_pips", "mae_down_pips") if pip is not None else ())
    buckets = {(state, h): {name: [] for name in metric_names} for state in STATES for h in FUTURE_HORIZONS}
    directions = {key: {d: 0 for d in ("UP", "DOWN", "FLAT")} for key in buckets}
    excluded = {key: {reason: 0 for reason in ("missing_reference", "gap_or_segment_boundary", "end_of_discovery_data")}
                for key in buckets}
    for row, (stamp, state) in enumerate(zip(assigned.timestamps, assigned.states)):
        position = positions.get(stamp)
        if position is None:
            for h in FUTURE_HORIZONS:
                excluded[state, h]["missing_reference"] += 1
            continue
        if assigned.segments is not None and assigned.segments[row] != segments[position]:
            raise ValueError("feature/price continuity identity mismatch")
        available = []
        reason = "end_of_discovery_data"
        for offset in range(1, max(FUTURE_HORIZONS) + 1):
            j = position + offset
            if j >= len(bars):
                break
            if (bars[j].open_time_utc != bars[position].open_time_utc + offset * STEP
                    or (segments is not None and segments[j] != segments[position])):
                reason = "gap_or_segment_boundary"
                break
            available.append(bars[j])
        valid = tuple(h for h in FUTURE_HORIZONS if h <= len(available))
        measured = measure_window(bars[position], available, valid, pip_metadata=pip_metadata) if valid else {}
        for h in FUTURE_HORIZONS:
            if h not in measured:
                excluded[state, h][reason] += 1
                continue
            observation = measured[h]
            for name in metric_names:
                buckets[state, h][name].append(observation[name])
            directions[state, h][observation["direction"]] += 1
    summaries = {}
    for state in STATES:
        summaries[state] = {}
        for h in FUTURE_HORIZONS:
            key = state, h
            count = len(buckets[key]["signed_return"])
            summaries[state][str(h)] = {
                "observation_count": count, "excluded_counts": excluded[key],
                "continuous": {name: _statistics(values) for name, values in buckets[key].items()},
                "direction": {d: {"count": n, "fraction": n / count if count else None}
                              for d, n in directions[key].items()},
            }
    comparisons = {str(h): {f"{a}_minus_{b}": {name: _effect(buckets[a, h][name], buckets[b, h][name])
                   for name in METRICS[:5]} for a, b in combinations(STATES, 2)} for h in FUTURE_HORIZONS}
    return {
        "experiment": "FUTURE_MARKET_BEHAVIOUR_V1", "feature_schema_version": MARKET_STATE_FEATURE_SCHEMA_VERSION,
        "feature_names": list(MARKET_STATE_FEATURE_NAMES_V1),
        "state_model": {"n_components": 3, "random_state": 42, "n_init": 5, "covariance_type": "full",
                        "reg_covar": 1e-6, "max_iter": 300, "converged": assigned.converged},
        "discovery_period": {"start_inclusive": DISCOVERY_START_UTC.isoformat(), "end_exclusive": DISCOVERY_END_UTC.isoformat()},
        "scaler_fit_period": {"first_observation": assigned.fit_start, "last_observation": assigned.fit_last},
        "assigned_observation_count": len(assigned.states),
        "incomplete_feature_rows_excluded": (assigned.feature_rows_in_discovery - len(assigned.states)
                                              if assigned.feature_rows_in_discovery is not None else None),
        "assigned_state_counts": {s: assigned.states.count(s) for s in STATES},
        "horizons": [{"bars": h, "minutes": h * 15} for h in FUTURE_HORIZONS],
        "price_reference": "close of the bar opening at T-15 minutes; feature timestamp T is its decision close",
        "continuity_rule": "exact 15-minute bar-open spacing and unchanged continuity ID; all closes strictly before 2023-01-01 UTC",
        "formulas": {"signed_return": "C(T+H)/C(T)-1", "absolute_return": "abs(signed_return)",
                     "mfe_up": "max(future highs)-C(T), not clamped", "mae_down": "C(T)-min(future lows), not clamped",
                     "realized_volatility": "sqrt(sum(log(C_i/C_(i-1))**2)), i=1..H; unannualized",
                     "flat": "C(T+H) == C(T) exactly", "standard_deviation": "sample, ddof=1",
                     "cohens_d": "(mean_A-mean_B)/pooled sample SD; null if either n<2 or SD=0"},
        "pip_metadata": dict(pip_metadata) if pip_metadata else None,
        "states": summaries, "comparisons": comparisons,
        "interpretation_limit": "retrospective Discovery association; overlapping horizons are dependent; no significance tests or state ranking",
    }


def analyze_future_behavior(features, prices, *, pip_metadata=None):
    return analyze_assigned_states(assign_discovery_states(features), prices, pip_metadata=pip_metadata)


def write_future_behavior_artifact(service, feature_path=DEFAULT_FEATURE_PATH, output=DEFAULT_OUTPUT, *, pip_metadata=None):
    """Bind price access to the saved feature artifact's authorized Discovery lineage."""
    feature_path, output = Path(feature_path), Path(output)
    metadata_path = feature_path.with_name("metadata.json")
    metadata_checksum = sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_checksum = sha256_file(feature_path)
    if metadata["parquet_sha256"] != feature_checksum:
        raise ValueError("feature artifact checksum mismatch")
    authorized = [s for s in metadata["transformation_history"] if s["step"] == "AUTHORIZED_DISCOVERY"]
    if len(authorized) != 1:
        raise ValueError("one authorized Discovery lineage required")
    partition_id = authorized[0]["partition_id"]
    manifest = service.get_partition(partition_id)
    if (manifest is None or manifest.role != PartitionRole.DISCOVERY or manifest.timeframe != "M15"
            or manifest.symbol != "EURUSD" or manifest.source_dataset_id != metadata["source_dataset_id"]
            or manifest.partition_fingerprint != metadata["source_partition_fingerprint"]
            or manifest.source_dataset_checksum != metadata["source_dataset_checksum"]):
        raise ValueError("Discovery M15 feature/price provenance mismatch")
    if output.resolve() in {feature_path.resolve(), metadata_path.resolve(), Path(manifest.path).resolve()}:
        raise ValueError("output cannot overwrite an input artifact")
    # Assign before obtaining price outcomes; no feature engine is called.
    assigned = assign_discovery_states(pd.read_parquet(feature_path))
    prices = service.access(partition_id, AccessContext.DISCOVERY_CONTEXT, AccessOperation.FEATURE_ANALYSIS,
                            actor="state-future-behavior-v1")
    result = analyze_assigned_states(assigned, prices, pip_metadata=pip_metadata)
    check = service.verify_partition(partition_id)
    if (not check["checksum_valid"] or not check["row_count_valid"] or sha256_file(feature_path) != feature_checksum
            or sha256_file(metadata_path) != metadata_checksum
            or service.get_partition(partition_id).partition_fingerprint != manifest.partition_fingerprint):
        raise ValueError("input artifact changed or failed checksum verification")
    result.update(feature_artifact_sha256=feature_checksum, feature_metadata_sha256=metadata_checksum,
                  partition_fingerprint=manifest.partition_fingerprint, price_partition_sha256=manifest.partition_checksum)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    from sandbox.partition.service import PartitionService
    from sandbox.research.registry import ResearchRegistry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.sqlite3"))
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    write_future_behavior_artifact(PartitionService(ResearchRegistry(args.catalog)), args.features, args.output)


if __name__ == "__main__":
    main()
