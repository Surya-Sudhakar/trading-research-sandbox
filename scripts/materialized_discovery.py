"""Operator utility for Phase 5A materialized Discovery execution.

Examples:
  python scripts/materialized_discovery.py materialize <prepared_id>
  python scripts/materialized_discovery.py execute <prepared_id> <proposal.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from sandbox.config import Settings
from sandbox.execution.models import ExecutionConfig
from sandbox.partition.service import PartitionService
from sandbox.research.registry import ResearchRegistry
from sandbox.research.materialized_runner import (
    execute_materialized_discovery_proposal,
    execution_view_directory,
    load_materialized_context,
    materialize_prepared_execution_view,
)


def _service():
    settings = Settings.load()
    registry = ResearchRegistry(settings.catalog_path, Path.cwd(), Path("results"))
    return PartitionService(registry)


def _config(service, dataset_id: str, config_path: Path | None) -> ExecutionConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    if "candle_interval_seconds" not in payload:
        timeframe = service.get_partition(dataset_id).timeframe
        interval = {"M1": 60, "M15": 900}.get(timeframe)
        if interval is None:
            raise ValueError(f"unsupported Discovery timeframe: {timeframe}")
        payload["candle_interval_seconds"] = interval
    return ExecutionConfig.model_validate(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 5A materialized Discovery operator utility")
    sub = parser.add_subparsers(dest="command", required=True)

    materialize = sub.add_parser("materialize", help="one-time prepared population -> compact Parquet")
    materialize.add_argument("prepared_id")
    materialize.add_argument("--root", type=Path, default=Path("results/discovery_runner"))

    inspect = sub.add_parser("inspect", help="inspect an existing compact execution view")
    inspect.add_argument("prepared_id")
    inspect.add_argument("--root", type=Path, default=Path("results/discovery_runner"))

    execute = sub.add_parser("execute", help="execute a proposal using only packet + compact Parquet")
    execute.add_argument("prepared_id")
    execute.add_argument("proposal", type=Path)
    execute.add_argument("--root", type=Path, default=Path("results/discovery_runner"))
    execute.add_argument("--pip-size", type=float, default=0.0001)
    execute.add_argument("--config", type=Path)

    args = parser.parse_args(argv)
    service = _service()

    if args.command == "materialize":
        started = perf_counter()
        view = materialize_prepared_execution_view(
            service,
            args.prepared_id,
            root=args.root,
            progress=lambda message: print(message, flush=True),
        )
        elapsed = perf_counter() - started
        size = view.parquet_path.stat().st_size
        print(json.dumps({
            "prepared_id": view.prepared_id,
            "execution_view_fingerprint": view.fingerprint,
            "rows": view.row_count,
            "parquet_path": str(view.parquet_path),
            "parquet_size_mb": round(size / (1024 * 1024), 2),
            "elapsed_seconds": round(elapsed, 3),
        }, indent=2))
        return 0

    document, packet, view = load_materialized_context(args.root, args.prepared_id)
    if args.command == "inspect":
        print(json.dumps({
            "prepared_id": args.prepared_id,
            "packet_fingerprint": packet.fingerprint,
            "dataset_id": view.dataset_id,
            "symbol": view.symbol,
            "timeframe": view.timeframe,
            "rows": view.row_count,
            "feature_ids": list(view.feature_ids),
            "execution_view_fingerprint": view.fingerprint,
            "parquet_path": str(view.parquet_path),
            "parquet_size_mb": round(view.parquet_path.stat().st_size / (1024 * 1024), 2),
            "prepared_chain_dataset": document["chain"]["dataset"],
        }, indent=2))
        return 0

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    config = _config(service, view.dataset_id, args.config)
    started = perf_counter()
    result = execute_materialized_discovery_proposal(
        service,
        args.prepared_id,
        proposal,
        root=args.root,
        pip_size=args.pip_size,
        execution_config=config,
        progress=lambda message: print(message, flush=True),
    )
    elapsed = perf_counter() - started
    print(json.dumps({**result, "elapsed_seconds": round(elapsed, 3)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
