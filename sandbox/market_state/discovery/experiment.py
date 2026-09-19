"""Run the frozen outcome-blind MARKET_STATE_FEATURES_V1 discovery experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sandbox.market_state.universal.dataset import export_discovery_features
from sandbox.market_state.universal.market_state_v1 import MARKET_STATE_FEATURE_NAMES_V1
from sandbox.partition.service import PartitionService
from sandbox.research.registry import ResearchRegistry

from .gmm import GaussianMixtureStateModel
from .scaler import DiscoveryStandardScaler, DISCOVERY_START_UTC, DISCOVERY_END_UTC
from .selection import DEFAULT_COMPONENT_COUNTS, evaluate_gmm_candidates
from .stability import evaluate_gmm_stability


def analyze_discovery_frame(frame: pd.DataFrame, component_counts=DEFAULT_COMPONENT_COUNTS) -> dict:
    """Analyze only complete frozen V1 rows inside 2016-2022; never inspect outcomes."""
    required = {"timestamp", *MARKET_STATE_FEATURE_NAMES_V1}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing discovery columns: " + ", ".join(missing))

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    mask = (timestamps >= DISCOVERY_START_UTC) & (timestamps < DISCOVERY_END_UTC)
    discovery = frame.loc[mask, ["timestamp", *MARKET_STATE_FEATURE_NAMES_V1]].copy()
    discovery["timestamp"] = pd.to_datetime(discovery["timestamp"], utc=True)
    discovery = discovery.dropna(subset=list(MARKET_STATE_FEATURE_NAMES_V1))
    if discovery.empty:
        raise ValueError("no complete MARKET_STATE_FEATURES_V1 rows in 2016-2022")

    X = discovery.loc[:, MARKET_STATE_FEATURE_NAMES_V1].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("discovery features contain non-finite values")
    stamps = tuple(ts.to_pydatetime() for ts in discovery["timestamp"])
    scaler = DiscoveryStandardScaler().fit(X, stamps)
    Z = scaler.transform(X)

    candidates = evaluate_gmm_candidates(Z, component_counts)
    rows = []
    for candidate in candidates:
        n = candidate.diagnostics.n_components
        stability = evaluate_gmm_stability(Z, n)
        model = GaussianMixtureStateModel(n).fit(Z)
        probabilities = model.predict_proba(Z)
        confidence = np.max(probabilities, axis=1)
        rows.append({
            **asdict(candidate.diagnostics),
            **asdict(stability),
            "mean_max_posterior": float(np.mean(confidence)),
            "p10_max_posterior": float(np.quantile(confidence, 0.10)),
        })

    return {
        "experiment": "MARKET_STATE_GMM_V1",
        "outcome_blind": True,
        "feature_names": list(MARKET_STATE_FEATURE_NAMES_V1),
        "observation_count": len(discovery),
        "first_timestamp": discovery["timestamp"].iloc[0].isoformat(),
        "last_timestamp": discovery["timestamp"].iloc[-1].isoformat(),
        "scaler": {
            "schema_version": scaler.metadata.schema_version,
            "observation_count": scaler.metadata.observation_count,
        },
        "candidates": rows,
    }


def run_partition(partition_id: str, catalog: Path = Path("data/catalog.sqlite3")) -> dict:
    registry = ResearchRegistry(catalog)
    service = PartitionService(registry)
    metadata = export_discovery_features(service, partition_id)
    feature_path = registry.project_root / metadata["artifact_path"]
    result = analyze_discovery_frame(pd.read_parquet(feature_path))
    result["partition_id"] = partition_id
    result["feature_artifact_id"] = metadata["artifact_id"]
    result["feature_artifact_path"] = metadata["artifact_path"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition_id")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("results/market_state/gmm_v1.json"))
    args = parser.parse_args()
    result = run_partition(args.partition_id, args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
