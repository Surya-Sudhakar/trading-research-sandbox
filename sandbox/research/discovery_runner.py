"""Two-stage Discovery orchestration. PREPARE stops at the external AI boundary.

Prepared snapshots contain the exact event population, not instructions to
regenerate it. ML evaluation here is in-sample Discovery evidence, not Validation.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
from numbers import Integral
from uuid import uuid4

from sandbox.execution.artifacts import store_run
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import ExecutionConfig
from sandbox.market_state.discovery_dataset import build_discovery_dataset
from sandbox.partition.models import AccessContext, AccessOperation, PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.provenance import sha256_file
from .ai_researcher import build_research_packet, parse_hypothesis_proposal
from .canonical import canonical_json, sha256_canonical
from .compiled_strategy_execution import build_compiled_strategy_intents
from .discovery_record_builder import (
    DiscoveryRecordConfig, DiscoveryRecordPopulation, build_discovery_record_population,
)
from .discovery_runner_artifacts import identity, read_binary_snapshot, write_binary_snapshot
from .hypothesis_compiler import compile_hypothesis
from .ml_discovery_matrix import build_ml_discovery_matrix
from .ml_preprocessing import fit_ml_preprocessor, transform_ml_matrix
from .ml_lightgbm import LightGBMClueConfig, train_lightgbm_clue_model, predict_lightgbm_clue_model
from .ml_evaluation import evaluate_prediction_batch
from .ml_clue_extraction import extract_ml_clues


VERSION = "PRODUCTION_DISCOVERY_RUNNER_V1"
SNAPSHOT_VERSION = "PRODUCTION_DISCOVERY_SNAPSHOT_V2"


@contextmanager
def _staging(parent):
    parent = Path(parent).resolve()
    stage = parent / (".pending-" + uuid4().hex)
    stage.mkdir()  # Inherit workspace ACLs on Windows.
    try:
        yield stage
    finally:
        if stage.exists():
            if stage.resolve().parent != parent or stage.is_symlink():
                raise ValueError("unsafe staging cleanup path")
            shutil.rmtree(stage)


@dataclass(frozen=True)
class PreparedDiscovery:
    prepared_id: str
    artifacts: dict

    @property
    def packet(self):
        return self.artifacts["packet"]

    @property
    def population(self):
        return self.artifacts["population"]


def _chain(a):
    p = a["population"]
    return dict(partition=p.partition_fingerprint,
                population=a["_population_fingerprint"],
                dataset=a["_dataset_fingerprint"],
                **{name: a[name].fingerprint for name in (
                    "matrix", "preprocessor", "encoded", "model", "predictions", "evaluation", "clues", "packet")})


def _dataset_identity(population_fingerprint, dataset):
    return sha256_canonical({
        "version": "DISCOVERY_DATASET_PROJECTION_V1",
        "population": population_fingerprint,
        "identity_field_ids": dataset.identity_field_ids,
        "predictor_field_ids": dataset.predictor_field_ids,
        "target_field_ids": dataset.target_field_ids,
    })


def _verify(a):
    if a["version"] != VERSION:
        raise ValueError("unsupported prepared version")
    p = a["population"]
    if (len(p.records) != len(p.record_segment_ids) or len(p.records) != len(p.dataset.rows)
            or any(row.record != record for row, record in zip(p.dataset.rows, p.records))):
        raise ValueError("DiscoveryDataset/population mismatch")
    matrix = build_ml_discovery_matrix(p.dataset, a["matrix"].feature_specs, a["matrix"].target_field_id)
    if matrix != a["matrix"]:
        raise ValueError("DiscoveryDataset/ML matrix mismatch")
    if fit_ml_preprocessor(matrix) != a["preprocessor"]:
        raise ValueError("preprocessor mismatch")
    if transform_ml_matrix(matrix, a["preprocessor"]) != a["encoded"]:
        raise ValueError("encoded matrix mismatch")
    model = a["model"]
    model.__post_init__()
    if (model.training_matrix_fingerprint != a["encoded"].fingerprint
            or model.preprocessor_fingerprint != a["preprocessor"].fingerprint
            or model.target_field_id != matrix.target_field_id
            or model.feature_specs != matrix.feature_specs):
        raise ValueError("model lineage mismatch")
    if predict_lightgbm_clue_model(model, a["encoded"]) != a["predictions"]:
        raise ValueError("prediction mismatch")
    if evaluate_prediction_batch(a["encoded"], a["predictions"]) != a["evaluation"]:
        raise ValueError("evaluation mismatch")
    if extract_ml_clues(model, a["evaluation"]) != a["clues"]:
        raise ValueError("clue report mismatch")
    if build_research_packet(a["clues"], a["evaluation"]) != a["packet"]:
        raise ValueError("packet mismatch")


def _prepared_path(root, prepared_id):
    if not isinstance(prepared_id, str) or re.fullmatch(r"[a-f0-9]{64}", prepared_id) is None:
        raise ValueError("invalid prepared ID")
    return Path(root) / "prepared" / prepared_id


def _load_prepared_discovery(root, prepared_id, *, deep_verify: bool) -> PreparedDiscovery:
    folder = _prepared_path(root, prepared_id)
    document = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    if document.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError("unsupported prepared snapshot")
    if sha256_canonical({k: v for k, v in document.items() if k != "prepared_id"}) != prepared_id:
        raise ValueError("prepared fingerprint mismatch")
    population_path, analysis_path = folder / "population.bin", folder / "analysis.bin"
    if (sha256_file(population_path) != document["population_blob_sha256"]
            or sha256_file(analysis_path) != document["analysis_blob_sha256"]):
        raise ValueError("prepared snapshot checksum mismatch")
    basis = read_binary_snapshot(population_path)
    dataset = build_discovery_dataset(basis["records"])
    if _dataset_identity(document["chain"]["population"], dataset) != document["chain"]["dataset"]:
        raise ValueError("DiscoveryDataset fingerprint mismatch")
    population = DiscoveryRecordPopulation(dataset=dataset, **basis)
    a = read_binary_snapshot(analysis_path)
    a["population"] = population
    a["_population_fingerprint"] = document["chain"]["population"]
    a["_dataset_fingerprint"] = document["chain"]["dataset"]
    if _chain(a) != document["chain"]:
        raise ValueError("prepared provenance mismatch")
    if deep_verify:
        _verify(a)
    if json.loads((folder / "packet.json").read_text(encoding="utf-8")) != json.loads(canonical_json(asdict(a["packet"]))):
        raise ValueError("stored packet mismatch")
    return PreparedDiscovery(prepared_id, a)


def load_prepared_discovery(root, prepared_id) -> PreparedDiscovery:
    """Deep loader retained for audits/tests: replays deterministic ML verification."""
    return _load_prepared_discovery(root, prepared_id, deep_verify=True)


def load_prepared_discovery_fast(root, prepared_id) -> PreparedDiscovery:
    """Execution loader: verifies immutable snapshot/checksums/provenance without replaying PREPARE ML."""
    return _load_prepared_discovery(root, prepared_id, deep_verify=False)


def _verify_prepared_handle_fast(root, prepared: PreparedDiscovery) -> None:
    """Cheap execution-time integrity check without rebuilding ML artifacts."""
    folder = _prepared_path(root, prepared.prepared_id)
    document = json.loads((folder / "prepared.json").read_text(encoding="utf-8"))
    if document.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError("unsupported prepared snapshot")
    if sha256_canonical({k: v for k, v in document.items() if k != "prepared_id"}) != prepared.prepared_id:
        raise ValueError("prepared fingerprint mismatch")
    population_path, analysis_path = folder / "population.bin", folder / "analysis.bin"
    if (sha256_file(population_path) != document["population_blob_sha256"]
            or sha256_file(analysis_path) != document["analysis_blob_sha256"]):
        raise ValueError("prepared snapshot checksum mismatch")
    if _chain(prepared.artifacts) != document["chain"]:
        raise ValueError("prepared provenance mismatch")
    stored_packet = json.loads((folder / "packet.json").read_text(encoding="utf-8"))
    if stored_packet != json.loads(canonical_json(asdict(prepared.packet))):
        raise ValueError("stored packet mismatch")


def prepare_discovery(service: PartitionService, partition_id: str, *, root: Path,
                      feature_specs, target_field_id: str,
                      record_config: DiscoveryRecordConfig = DiscoveryRecordConfig(),
                      model_config: LightGBMClueConfig = LightGBMClueConfig()) -> PreparedDiscovery:
    population = build_discovery_record_population(service, partition_id, record_config)
    manifest = service.get_partition(partition_id)
    if manifest.partition_fingerprint != population.partition_fingerprint:
        raise ValueError("partition changed during PREPARE")
    matrix = build_ml_discovery_matrix(population.dataset, feature_specs, target_field_id)
    preprocessor = fit_ml_preprocessor(matrix)
    encoded = transform_ml_matrix(matrix, preprocessor)
    model = train_lightgbm_clue_model(encoded, model_config)
    predictions = predict_lightgbm_clue_model(model, encoded)
    evaluation = evaluate_prediction_batch(encoded, predictions)
    clues = extract_ml_clues(model, evaluation)
    packet = build_research_packet(clues, evaluation)
    a = dict(version=VERSION, population=population, symbol=manifest.symbol, timeframe=manifest.timeframe,
             matrix=matrix, preprocessor=preprocessor, encoded=encoded, model=model,
             predictions=predictions, evaluation=evaluation, clues=clues, packet=packet)
    prepared_parent = Path(root) / "prepared"
    prepared_parent.mkdir(parents=True, exist_ok=True)
    # The exact ResearchRecord population is persisted, while its deterministic
    # DiscoveryDataset projection is regenerated and fingerprint-checked on load.
    # This avoids expanding tens of millions of scalar cells into JSON nodes.
    with _staging(prepared_parent) as temporary:
        stage = Path(temporary)
        basis = {name: getattr(population, name) for name in (
            "partition_id", "partition_fingerprint", "source_dataset_id", "config",
            "records", "record_segment_ids")}
        basis["record_segment_ids"] = tuple(int(value) if isinstance(value, Integral) else value
                                              for value in basis["record_segment_ids"])
        write_binary_snapshot(stage / "population.bin", basis)
        population_fingerprint = sha256_file(stage / "population.bin")
        dataset_fingerprint = _dataset_identity(population_fingerprint, population.dataset)
        a["_population_fingerprint"] = population_fingerprint
        a["_dataset_fingerprint"] = dataset_fingerprint
        analysis = {k: v for k, v in a.items() if k not in {
            "population", "_population_fingerprint", "_dataset_fingerprint"}}
        write_binary_snapshot(stage / "analysis.bin", analysis)
        chain = _chain(a)
        document = dict(snapshot_version=SNAPSHOT_VERSION,
                        population_blob_sha256=population_fingerprint,
                        analysis_blob_sha256=sha256_file(stage / "analysis.bin"), chain=chain)
        prepared_id = sha256_canonical(document)
        document["prepared_id"] = prepared_id
        destination = _prepared_path(root, prepared_id)
        if destination.exists():
            return load_prepared_discovery(root, prepared_id)
        (stage / "prepared.json").write_text(canonical_json(document), encoding="utf-8")
        (stage / "packet.json").write_text(canonical_json(asdict(packet)), encoding="utf-8")
        stage.rename(destination)
    return PreparedDiscovery(prepared_id, a)


def execute_discovery_proposal(service: PartitionService, prepared: PreparedDiscovery, packet,
                               proposal, *, root: Path, pip_size: float,
                               execution_config: ExecutionConfig, fast_path: bool = False,
                               progress=None):
    """Execute an externally supplied proposal against the exact stored dataset.

    The final catalog COMPLETE transition is the commit point. Files alone are
    insufficient to identify a completed execution. Interrupted writes remain
    unregistered, and caught registration failures are marked FAILED.
    """
    def report(message):
        if progress is not None:
            progress(message)

    if fast_path:
        report("[1/6] Verifying prepared snapshot checksums/provenance...")
        _verify_prepared_handle_fast(root, prepared)
        exact = prepared
    else:
        exact = load_prepared_discovery(root, prepared.prepared_id)
        try:
            _verify(prepared.artifacts)
            matches = _chain(prepared.artifacts) == _chain(exact.artifacts)
        except (AttributeError, KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError("prepared dataset/artifact mismatch")

    if packet != exact.packet:
        raise ValueError("exact prepared packet required")
    a, population = exact.artifacts, exact.population
    report("[2/6] Validating proposal and compiling StrategySpec...")
    parsed = parse_hypothesis_proposal(proposal, exact.packet)
    compiled = compile_hypothesis(parsed, exact.packet)
    manifest = service.get_partition(population.partition_id)
    # Authorization precedes integrity reads, even if catalog role has changed.
    frame = service.access(population.partition_id, AccessContext.DISCOVERY_CONTEXT,
                           AccessOperation.BACKTEST_EXECUTION, actor="discovery-runner")
    if (manifest.role != PartitionRole.DISCOVERY
            or manifest.partition_fingerprint != population.partition_fingerprint
            or manifest.symbol != a["symbol"] or manifest.timeframe != a["timeframe"]):
        raise ValueError("prepared partition mismatch")
    integrity = service.verify_partition(population.partition_id)
    if not integrity["checksum_valid"] or not integrity["row_count_valid"]:
        raise ValueError("partition integrity mismatch")
    interval = {"M1": 60, "M15": 900}[manifest.timeframe]
    execution_config = ExecutionConfig.model_validate(execution_config.model_dump())
    if execution_config.candle_interval_seconds != interval:
        raise ValueError("execution interval must match partition timeframe")
    report("[3/6] Evaluating hypothesis across Discovery rows and generating TradeIntents...")
    batch = build_compiled_strategy_intents(compiled, exact.packet, population.dataset,
        dataset_id=population.partition_id, symbol=manifest.symbol, pip_size=pip_size)
    if not batch.intents:
        raise ValueError("proposal produced no TradeIntents; no backtest run registered")
    report(f"[4/6] Running deterministic backtest on {batch.intent_count:,} TradeIntents...")
    result = BacktestEngine(execution_config).run(batch.intents, frame)
    chain = dict(_chain(a), prepared=prepared.prepared_id, proposal=parsed.fingerprint,
                 compiled=compiled.fingerprint, strategy_spec=compiled.strategy_spec_fingerprint,
                 intent_batch=batch.fingerprint, backtest=result.run_fingerprint)
    run_id = "run-" + result.run_fingerprint[:20]
    destination = Path(root) / "executed" / run_id
    catalog = service.registry.catalog
    report("[5/6] Persisting/validating execution artifacts...")
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
            provenance = dict(version=VERSION, run_id=stored.run_id, chain=chain,
                              ledger_checksum=stored.ledger_checksum,
                              summary_checksum=sha256_file(stored.summary_path),
                              intent_count=batch.intent_count)
            (stage / "provenance.json").write_text(canonical_json(provenance), encoding="utf-8")
            (stage / "proposal.json").write_text(canonical_json(asdict(parsed)), encoding="utf-8")
            (stage / "strategy.json").write_text(canonical_json(compiled.strategy_spec.model_dump(mode="json")), encoding="utf-8")
            (stage / "intents.json").write_text(canonical_json([i.model_dump(mode="json") for i in batch.intents]), encoding="utf-8")
            stage.rename(destination)
        registered = True
        catalog.add_research_run(stored.run_id, stored.run_fingerprint, population.partition_id,
            result.execution_engine_version, result.ledger_schema_version, result.metrics_version,
            execution_config.model_dump(mode="json"), str(ledger_path), str(summary_path))
        report(f"[6/6] Complete: {stored.run_id}")
        return provenance
    except BaseException:
        # Even a wrapper raising after the catalog commit must not leave success.
        if registered:
            with catalog.connection() as connection:
                connection.execute("UPDATE research_runs SET status='FAILED' WHERE run_id=?", (run_id,))
        raise
