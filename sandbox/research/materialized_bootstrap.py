"""Low-overhead one-time bootstrap for the materialized Discovery execution view.

Unlike the original bootstrap, this path does not reconstruct DiscoveryDataset
or load analysis.bin. It verifies the immutable prepared manifest and
population.bin checksum, unpickles the persisted ResearchRecords once, and
extracts only the compact execution columns needed by the packet.
"""
from __future__ import annotations

from sandbox.partition.models import PartitionRole
from sandbox.provenance import sha256_file

from .discovery_runner_artifacts import read_binary_snapshot
from .materialized_execution import load_execution_view
from .materialized_population import materialize_records_execution_view
from .materialized_runner import (
    _packet_from_json,
    _prepared_document,
    execution_view_directory,
)


def materialize_prepared_execution_view_direct(service, prepared_id: str, *, root, progress=None):
    def report(message):
        if progress is not None:
            progress(message)

    report("[M1/5] Verifying prepared manifest and population checksum...")
    folder, document = _prepared_document(root, prepared_id)
    packet = _packet_from_json(folder / "packet.json")
    if packet.fingerprint != document["chain"]["packet"]:
        raise ValueError("stored packet fingerprint mismatch")
    population_path = folder / "population.bin"
    if sha256_file(population_path) != document["population_blob_sha256"]:
        raise ValueError("prepared population checksum mismatch")

    report("[M2/5] Loading population.bin once (no DiscoveryDataset reconstruction)...")
    basis = read_binary_snapshot(population_path)
    required = {"partition_id", "partition_fingerprint", "records"}
    if not isinstance(basis, dict) or not required.issubset(basis):
        raise ValueError("invalid prepared population snapshot")
    if basis["partition_fingerprint"] != document["chain"]["partition"]:
        raise ValueError("prepared population lineage mismatch")

    manifest = service.get_partition(basis["partition_id"])
    if (manifest.role != PartitionRole.DISCOVERY
            or manifest.partition_fingerprint != basis["partition_fingerprint"]):
        raise ValueError("prepared partition mismatch")
    integrity = service.verify_partition(basis["partition_id"])
    if not integrity["checksum_valid"] or not integrity["row_count_valid"]:
        raise ValueError("partition integrity mismatch")

    records = basis["records"]
    report(f"[M3/5] Extracting compact causal columns directly from {len(records):,} ResearchRecords...")
    view = materialize_records_execution_view(
        records,
        packet,
        directory=execution_view_directory(root, prepared_id),
        prepared_id=prepared_id,
        dataset_id=basis["partition_id"],
        symbol=manifest.symbol,
        timeframe=manifest.timeframe,
    )

    report("[M4/5] Verifying compact Parquet checksum/fingerprint...")
    loaded = load_execution_view(execution_view_directory(root, prepared_id))
    if loaded != view:
        raise ValueError("materialized execution-view reload mismatch")
    report(f"[M5/5] Materialized execution view ready: {view.row_count:,} rows")
    return view
