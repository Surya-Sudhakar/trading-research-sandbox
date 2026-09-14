"""Focused production-runner tests for the Phase 5A materialized execution path."""
from pathlib import Path
from uuid import uuid4

from sandbox.execution.models import ExecutionConfig
from sandbox.research import discovery_runner as runner
from sandbox.research import materialized_runner as materialized
from sandbox.research.ml_discovery_matrix import MLFeatureSpec

from test_discovery_record_builder import market_frame, partition
from test_discovery_runner import external_proposal


FEATURES = (
    MLFeatureSpec("x.anchor_kind", "CATEGORICAL"),
    MLFeatureSpec("x.geometry.coordinate", "NUMERIC"),
)
TARGET = "y.h4.close_return_fraction"
MODEL = runner.LightGBMClueConfig(num_boost_round=3, min_data_in_leaf=1)
EXECUTION = ExecutionConfig(candle_interval_seconds=900)


def _environment(tmp_path):
    root = Path(tmp_path) / uuid4().hex
    service, manifest = partition(root / "partition", market_frame(64, touches=True))
    prepared = runner.prepare_discovery(
        service,
        manifest.partition_id,
        root=root,
        feature_specs=FEATURES,
        target_field_id=TARGET,
        model_config=MODEL,
    )
    return service, prepared, root


def test_materialize_once_then_load_without_population(tmp_path, monkeypatch):
    service, prepared, root = _environment(tmp_path)
    view = materialized.materialize_prepared_execution_view(service, prepared.prepared_id, root=root)
    assert view.row_count == len(prepared.population.dataset.rows)
    assert view.prepared_id == prepared.prepared_id
    assert view.packet_fingerprint == prepared.packet.fingerprint

    def forbidden(*args, **kwargs):
        raise AssertionError("population loader must not run after materialization")

    monkeypatch.setattr(materialized, "load_prepared_discovery_fast", forbidden)
    document, packet, loaded = materialized.load_materialized_context(root, prepared.prepared_id)
    assert document["prepared_id"] == prepared.prepared_id
    assert packet == prepared.packet
    assert loaded == view


def test_materialized_runner_matches_existing_runner(tmp_path):
    service, prepared, root = _environment(tmp_path)
    materialized.materialize_prepared_execution_view(service, prepared.prepared_id, root=root)
    proposal = external_proposal(prepared.packet, "MATERIALIZED_PARITY")

    legacy = runner.execute_discovery_proposal(
        service=service,
        prepared=prepared,
        packet=prepared.packet,
        proposal=proposal,
        root=root,
        pip_size=0.1,
        execution_config=EXECUTION,
    )
    fast = materialized.execute_materialized_discovery_proposal(
        service,
        prepared.prepared_id,
        proposal,
        root=root,
        pip_size=0.1,
        execution_config=EXECUTION,
    )

    # Identical execution semantics intentionally deduplicate to the same immutable
    # run artifact. Because the legacy run was persisted first, the materialized
    # call must return that existing provenance unchanged rather than rewriting it.
    assert fast == legacy


def test_repeated_materialized_execution_is_deterministic(tmp_path):
    service, prepared, root = _environment(tmp_path)
    view = materialized.materialize_prepared_execution_view(service, prepared.prepared_id, root=root)
    proposal = external_proposal(prepared.packet, "MATERIALIZED_REPEAT")
    first = materialized.execute_materialized_discovery_proposal(
        service,
        prepared.prepared_id,
        proposal,
        root=root,
        pip_size=0.1,
        execution_config=EXECUTION,
    )
    second = materialized.execute_materialized_discovery_proposal(
        service,
        prepared.prepared_id,
        proposal,
        root=root,
        pip_size=0.1,
        execution_config=EXECUTION,
    )
    assert first["execution_view_fingerprint"] == view.fingerprint
    assert second == first
