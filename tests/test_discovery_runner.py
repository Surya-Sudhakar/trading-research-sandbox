"""Real component integration on catalog-backed synthetic market partitions."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd
import pytest

from test_discovery_record_builder import market_frame, partition
from sandbox.execution.models import ExecutionConfig
from sandbox.partition.models import PartitionRole
from sandbox.research.errors import ResearchError
from sandbox.research.ml_discovery_matrix import MLFeatureSpec
from sandbox.research import discovery_runner as runner


FEATURES = (MLFeatureSpec("x.anchor_kind", "CATEGORICAL"),
            MLFeatureSpec("x.geometry.coordinate", "NUMERIC"))
TARGET = "y.h4.close_return_fraction"
CONFIG = runner.LightGBMClueConfig(num_boost_round=3, min_data_in_leaf=1)
EXECUTION = ExecutionConfig(candle_interval_seconds=900)


@pytest.fixture(scope="module")
def environment():
    root = (Path("tests/runtime/discovery_runner") / uuid4().hex).resolve()
    service, manifest = partition(root / "partition", market_frame(64, touches=True))
    prepared = runner.prepare_discovery(service, manifest.partition_id, root=root,
        feature_specs=FEATURES, target_field_id=TARGET, model_config=CONFIG)
    return service, prepared, root


def external_proposal(packet, name="EXTERNAL_TEST"):
    # Explicit external-controller test input; production never generates this.
    return dict(version="HYPOTHESIS_PROPOSAL_V1", status="UNVERIFIED_DISCOVERY_HYPOTHESIS",
        research_packet_fingerprint=packet.fingerprint, hypothesis_id=name,
        statement="Synthetic integration proposal", rationale="Test-only controller input",
        target_field_id=packet.target_field_id, family_id="MACHINE_DISCOVERED",
        strategy_id="INTEGRATION", variant_id="ORIGINAL", evidence_feature_ids=["x.anchor_kind"],
        rules=[dict(rule_id="TOUCH", conditions=[dict(field_id="x.anchor_kind", operator="EQ",
             value="TOUCH_TRANSITION")], condition_logic="ALL", side="LONG", stop_pips=10., target_r_multiple=2.)])


def execute(environment, name="EXTERNAL_TEST", **overrides):
    service, prepared, root = environment
    values = dict(service=service, prepared=prepared, packet=prepared.packet,
                  proposal=external_proposal(prepared.packet, name), root=root,
                  pip_size=0.1, execution_config=EXECUTION)
    values.update(overrides)
    return runner.execute_discovery_proposal(**values)


def test_prepare_real_population_ml_packet_and_external_boundary(environment):
    service, prepared, root = environment
    assert len(prepared.population.records) == 21
    assert len(prepared.population.dataset.rows) == 21
    a = prepared.artifacts
    assert a["matrix"].source_row_indices == tuple(range(21))
    assert len(a["predictions"].predictions) == 21
    assert a["evaluation"].observed_target_count == 21
    assert prepared.packet.model_fingerprint == a["model"].fingerprint
    assert json.loads((root / "prepared" / prepared.prepared_id / "packet.json").read_text())["fingerprint"] == prepared.packet.fingerprint
    assert not (root / "executed").exists()
    assert not any("proposal" in key for key in a)
    with service.registry.catalog.connection() as con:
        assert con.execute("SELECT count(*) FROM research_runs").fetchone()[0] == 0


def test_prepare_repeatability_and_restart_without_builder(environment, monkeypatch):
    service, prepared, root = environment
    repeated = runner.prepare_discovery(service, prepared.population.partition_id, root=root,
        feature_specs=FEATURES, target_field_id=TARGET, model_config=CONFIG)
    assert repeated == prepared
    def no_rebuild(*args, **kwargs): pytest.fail("rebuilding population at load/execute")
    monkeypatch.setattr(runner, "build_discovery_record_population", no_rebuild)
    loaded = runner.load_prepared_discovery(root, prepared.prepared_id)
    assert loaded == prepared


def test_execute_uses_compiler_4i_engine_store_and_catalog(environment, monkeypatch):
    service, prepared, root = environment
    calls = {}
    for name in ("compile_hypothesis", "build_compiled_strategy_intents", "store_run"):
        original = getattr(runner, name)
        def spy(*args, _name=name, _original=original, **kwargs):
            value = _original(*args, **kwargs)
            calls[_name] = value
            return value
        monkeypatch.setattr(runner, name, spy)
    original_run = runner.BacktestEngine.run
    def engine(self, intents, candles):
        assert intents is calls["build_compiled_strategy_intents"].intents
        calls["engine"] = original_run(self, intents, candles)
        return calls["engine"]
    monkeypatch.setattr(runner.BacktestEngine, "run", engine)
    result = execute(environment)
    assert len(calls) == 4 and result["intent_count"] == 15
    registered = service.registry.catalog.research_run(result["run_id"])
    assert registered["status"] == "COMPLETE"
    assert Path(registered["ledger_path"]).exists() and Path(registered["summary_path"]).exists()
    assert result["chain"]["backtest"] == registered["run_fingerprint"]
    assert result["chain"]["intent_batch"] == calls["build_compiled_strategy_intents"].fingerprint
    assert result["chain"]["compiled"] == calls["compile_hypothesis"].fingerprint
    assert result["chain"]["prepared"] == prepared.prepared_id
    assert set(result["chain"]) == {"partition", "population", "dataset", "matrix", "preprocessor",
        "encoded", "model", "predictions", "evaluation", "clues", "packet", "prepared", "proposal",
        "compiled", "strategy_spec", "intent_batch", "backtest"}
    assert execute(environment) == result


def test_exact_packet_required(environment):
    with pytest.raises(ValueError, match="packet"):
        execute(environment, packet=object())


def test_external_proposal_required(environment):
    with pytest.raises((ValueError, TypeError)):
        execute(environment, proposal=None)


def test_proposal_packet_mismatch(environment):
    proposal = external_proposal(environment[1].packet)
    proposal["research_packet_fingerprint"] = "a" * 64
    with pytest.raises(ValueError, match="packet"):
        execute(environment, proposal=proposal)


@pytest.mark.parametrize("stage", ["population", "matrix", "preprocessor", "model", "evaluation", "clues", "packet"])
def test_prepared_artifact_mismatch(environment, stage):
    prepared = environment[1]
    corrupt = replace(prepared, artifacts={**prepared.artifacts, stage: None})
    with pytest.raises(ValueError, match="mismatch"):
        execute(environment, prepared=corrupt)


def test_persisted_dataset_tampering_rejected(environment):
    _, prepared, root = environment
    copy = root / uuid4().hex
    shutil.copytree(root / "prepared", copy / "prepared")
    path = copy / "prepared" / prepared.prepared_id / "population.bin"
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum"):
        execute(environment, root=copy)


@pytest.mark.parametrize("role", [PartitionRole.VALIDATION, PartitionRole.FINAL_TEST])
def test_protected_partition_prepare_denied(environment, role, monkeypatch):
    root = environment[2] / uuid4().hex
    service, manifest = partition(root, market_frame(), role)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("protected data read"))
    with pytest.raises(ResearchError, match="ACCESS_DENIED"):
        runner.prepare_discovery(service, manifest.partition_id, root=root,
            feature_specs=FEATURES, target_field_id=TARGET, model_config=CONFIG)
    assert not (root / "prepared").exists()


def test_failed_prepare_not_published(environment, monkeypatch):
    service, prepared, root = environment
    root = root / uuid4().hex
    def fail(*args, **kwargs): raise RuntimeError("training failed")
    monkeypatch.setattr(runner, "train_lightgbm_clue_model", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        runner.prepare_discovery(service, prepared.population.partition_id, root=root,
            feature_specs=FEATURES, target_field_id=TARGET)
    assert not (root / "prepared").exists()


@pytest.mark.parametrize("stage", ["store", "register"])
def test_failed_execution_no_successful_catalog_entry(environment, monkeypatch, stage):
    service, _, root = environment
    with service.registry.catalog.connection() as con:
        before = con.execute("SELECT count(*) FROM research_runs WHERE status='COMPLETE'").fetchone()[0]
    if stage == "store":
        original = runner.store_run
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("store failed")
        monkeypatch.setattr(runner, "store_run", fail)
    else:
        original = service.registry.catalog.add_research_run
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("register failed after commit")
        monkeypatch.setattr(service.registry.catalog, "add_research_run", fail)
    with pytest.raises(RuntimeError, match="failed"):
        execute(environment, name="FAIL_" + stage.upper())
    with service.registry.catalog.connection() as con:
        assert con.execute("SELECT count(*) FROM research_runs WHERE status='COMPLETE'").fetchone()[0] == before


def test_target_cannot_be_predictor(environment):
    service, prepared, root = environment
    with pytest.raises(ValueError):
        runner.prepare_discovery(service, prepared.population.partition_id, root=root / uuid4().hex,
            feature_specs=(MLFeatureSpec(TARGET, "NUMERIC"),), target_field_id=TARGET)


def test_execution_timeframe_mismatch_rejected(environment):
    with pytest.raises(ValueError, match="interval"):
        execute(environment, execution_config=ExecutionConfig(candle_interval_seconds=60))
