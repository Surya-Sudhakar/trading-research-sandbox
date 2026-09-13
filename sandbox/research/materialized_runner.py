"""Production execution against a pre-materialized Discovery execution view.

The expensive ResearchRecord -> DiscoveryDataset -> Parquet conversion happens
once. Repeated hypotheses load only the small packet/manifest plus the compact
Parquet view, while preserving the existing compiler, TradeIntent and backtester.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from sandbox.execution.artifacts import store_run
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import ExecutionConfig
from sandbox.partition.models import AccessContext, AccessOperation, PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.provenance import sha256_file

from .ai_researcher import AIResearchPacket, parse_hypothesis_proposal
from .canonical import canonical_json, sha256_canonical
from .discovery_runner import (
    SNAPSHOT_VERSION,
    VERSION,
    _prepared_path,
    _staging,
    load_prepared_discovery_fast,
)
from .hypothesis_compiler import compile_hypothesis
from .materialized_execution import (
    MaterializedExecutionView,
    build_materialized_strategy_intents,
    load_execution_view,
    materialize_execution_view,
)


def _packet_from_json(path: Path) -> AIResearchPacket:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "feature_ids", "feature_kinds", "feature_gain_fractions", "feature_split_fractions",
        "gain_order", "split_order", "allowed_condition_feature_ids",
        "permitted_entry_kinds", "permitted_stop_kinds", "permitted_target_kinds",
    ):
        data[key] = tuple(data[key])
    data["evaluation_summary"] = tuple(tuple(item) for item in data["evaluation_summary"])
    return AIResearchPacket(**data)


def _prepared_document(root: Path, prepared_id: str) -> tuple[Path, dict]:
    folder = _prepared_path(root, prepared_id)
    document = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    if document.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError("unsupported prepared snapshot")
    if document.get("prepared_id") != prepared_id:
        raise ValueError("prepared ID mismatch")
    if sha256_canonical({k: v for k, v in document.items() if k != "prepared_id"}) != prepared_id:
        raise ValueError("prepared fingerprint mismatch")
    return folder, document


def execution_view_directory(root: Path, prepared_id: str) -> Path:
    _prepared_path(root, prepared_id)  # strict ID validation
    return Path(root) / "materialized" / prepared_id


def materialize_prepared_execution_view(
    service: PartitionService,
    prepared_id: str,
    *,
    root: Path,
    progress=None,
) -> MaterializedExecutionView:
    """Perform the one-time conversion from the prepared population to Parquet."""
    def report(message):
        if progress is not None:
            progress(message)

    report("[M1/4] Loading and verifying prepared Discovery snapshot (one-time cost)...")
    prepared = load_prepared_discovery_fast(root, prepared_id)
    population = prepared.population
    manifest = service.get_partition(population.partition_id)
    if (manifest.role != PartitionRole.DISCOVERY
            or manifest.partition_fingerprint != population.partition_fingerprint
            or manifest.symbol != prepared.artifacts["symbol"]
            or manifest.timeframe != prepared.artifacts["timeframe"]):
        raise ValueError("prepared partition mismatch")
    integrity = service.verify_partition(population.partition_id)
    if not integrity["checksum_valid"] or not integrity["row_count_valid"]:
        raise ValueError("partition integrity mismatch")

    report(f"[M2/4] Projecting {len(population.dataset.rows):,} rows to compact causal columns...")
    view = materialize_execution_view(
        population.dataset,
        prepared.packet,
        directory=execution_view_directory(root, prepared_id),
        prepared_id=prepared_id,
        dataset_id=population.partition_id,
        symbol=manifest.symbol,
        timeframe=manifest.timeframe,
    )
    report("[M3/4] Verifying materialized Parquet checksum/fingerprint...")
    loaded = load_execution_view(execution_view_directory(root, prepared_id))
    if loaded != view:
        raise ValueError("materialized execution-view reload mismatch")
    report(f"[M4/4] Materialized execution view ready: {view.row_count:,} rows")
    return view


def load_materialized_context(root: Path, prepared_id: str):
    """Load only cheap immutable metadata + packet + compact execution view."""
    folder, document = _prepared_document(root, prepared_id)
    packet = _packet_from_json(folder / "packet.json")
    if packet.fingerprint != document["chain"]["packet"]:
        raise ValueError("stored packet fingerprint mismatch")
    view = load_execution_view(execution_view_directory(root, prepared_id))
    if view.prepared_id != prepared_id or view.packet_fingerprint != packet.fingerprint:
        raise ValueError("execution-view lineage mismatch")
    return document, packet, view


def _persist_execution(service, *, root, document, prepared_id, parsed, compiled,
                       batch, result, execution_config, execution_view_fingerprint, progress=None):
    def report(message):
        if progress is not None:
            progress(message)

    chain = dict(
        document["chain"],
        prepared=prepared_id,
        proposal=parsed.fingerprint,
        compiled=compiled.fingerprint,
        strategy_spec=compiled.strategy_spec_fingerprint,
        intent_batch=batch.fingerprint,
        backtest=result.run_fingerprint,
    )
    run_id = "run-" + result.run_fingerprint[:20]
    destination = Path(root) / "executed" / run_id
    catalog = service.registry.catalog
    report("[E5/6] Persisting/validating execution artifacts...")
    existing = catalog.research_run(run_id)
    if existing:
        if existing["status"] != "COMPLETE" or existing["run_fingerprint"] != result.run_fingerprint:
            raise ValueError("run already exists without matching completion")
        saved = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
        if saved["chain"] != chain or sha256_file(Path(existing["ledger_path"])) != saved["ledger_checksum"]:
            raise ValueError("existing execution provenance mismatch")
        if sha256_file(Path(existing["summary_path"])) != saved["summary_checksum"]:
            raise ValueError("existing summary mismatch")
        return saved
    if destination.exists():
        raise ValueError("unregistered execution artifacts exist; explicit recovery required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    registered = False
    try:
        with _staging(destination.parent) as temporary:
            stage = Path(temporary)
            stored = store_run(result, stage)
            ledger_path = destination / stored.ledger_path.relative_to(stage)
            summary_path = destination / stored.summary_path.relative_to(stage)
            provenance = dict(
                version=VERSION,
                run_id=stored.run_id,
                chain=chain,
                execution_view_fingerprint=execution_view_fingerprint,
                ledger_checksum=stored.ledger_checksum,
                summary_checksum=sha256_file(stored.summary_path),
                intent_count=batch.intent_count,
            )
            (stage / "provenance.json").write_text(canonical_json(provenance), encoding="utf-8")
            (stage / "proposal.json").write_text(canonical_json(asdict(parsed)), encoding="utf-8")
            (stage / "strategy.json").write_text(
                canonical_json(compiled.strategy_spec.model_dump(mode="json")), encoding="utf-8")
            (stage / "intents.json").write_text(
                canonical_json([i.model_dump(mode="json") for i in batch.intents]), encoding="utf-8")
            stage.rename(destination)
        registered = True
        catalog.add_research_run(
            stored.run_id,
            stored.run_fingerprint,
            batch.dataset_id,
            result.execution_engine_version,
            result.ledger_schema_version,
            result.metrics_version,
            execution_config.model_dump(mode="json"),
            str(ledger_path),
            str(summary_path),
        )
        report(f"[E6/6] Complete: {stored.run_id}")
        return provenance
    except BaseException:
        if registered:
            with catalog.connection() as connection:
                connection.execute("UPDATE research_runs SET status='FAILED' WHERE run_id=?", (run_id,))
        raise


def execute_materialized_discovery_proposal(
    service: PartitionService,
    prepared_id: str,
    proposal,
    *,
    root: Path,
    pip_size: float,
    execution_config: ExecutionConfig,
    progress=None,
):
    """Execute a proposal without loading population.bin or rebuilding DiscoveryDataset."""
    def report(message):
        if progress is not None:
            progress(message)

    report("[E1/6] Loading compact execution view and packet metadata...")
    document, packet, view = load_materialized_context(root, prepared_id)
    report("[E2/6] Validating proposal and compiling StrategySpec...")
    parsed = parse_hypothesis_proposal(proposal, packet)
    compiled = compile_hypothesis(parsed, packet)

    manifest = service.get_partition(view.dataset_id)
    # Authorization still happens before any protected market-data execution read.
    frame = service.access(
        view.dataset_id,
        AccessContext.DISCOVERY_CONTEXT,
        AccessOperation.BACKTEST_EXECUTION,
        actor="materialized-discovery-runner",
    )
    if (manifest.role != PartitionRole.DISCOVERY
            or manifest.partition_fingerprint != document["chain"]["partition"]
            or manifest.symbol != view.symbol
            or manifest.timeframe != view.timeframe):
        raise ValueError("materialized partition mismatch")
    integrity = service.verify_partition(view.dataset_id)
    if not integrity["checksum_valid"] or not integrity["row_count_valid"]:
        raise ValueError("partition integrity mismatch")

    interval = {"M1": 60, "M15": 900}[manifest.timeframe]
    execution_config = ExecutionConfig.model_validate(execution_config.model_dump())
    if execution_config.candle_interval_seconds != interval:
        raise ValueError("execution interval must match partition timeframe")

    report(f"[E3/6] Querying {view.row_count:,} materialized rows with DuckDB...")
    batch = build_materialized_strategy_intents(view, compiled, packet, pip_size=pip_size)
    if not batch.intents:
        raise ValueError("proposal produced no TradeIntents; no backtest run registered")
    report(f"[E4/6] Running deterministic backtest on {batch.intent_count:,} TradeIntents...")
    result = BacktestEngine(execution_config).run(batch.intents, frame)
    return _persist_execution(
        service,
        root=root,
        document=document,
        prepared_id=prepared_id,
        parsed=parsed,
        compiled=compiled,
        batch=batch,
        result=result,
        execution_config=execution_config,
        execution_view_fingerprint=view.fingerprint,
        progress=progress,
    )
