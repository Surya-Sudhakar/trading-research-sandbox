from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from sandbox.catalog import Catalog
from sandbox.execution.artifacts import store_run
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, TradeIntent
from sandbox.provenance import Provenance
from sandbox.research.canonical import experiment_fingerprint
from sandbox.research.errors import DatasetVerificationError, DuplicateExperimentError, ImmutableRecordError, InvalidTransitionError, JournalIntegrityError, PreRegistrationError
from sandbox.research.models import ExperimentStatus, HypothesisStatus, VerdictValue
from sandbox.research.registry import ResearchRegistry


def setup_registry(tag=None):
    tag=tag or uuid4().hex; root=Path("tests/runtime/stage3")/tag; root.mkdir(parents=True,exist_ok=True)
    registry=ResearchRegistry(root/"catalog.db",Path.cwd(),root/"results")
    dataset=root/"dataset.parquet"; dataset.write_bytes(b"scientific-dataset")
    provenance=Provenance.create(dataset,"Test","Synthetic","X","M1",pd.Timestamp("2024-01-01",tz="UTC").to_pydatetime(),pd.Timestamp("2024-01-02",tz="UTC").to_pydatetime(),2)
    registry.catalog.add_dataset(provenance)
    registry.create_program("TEST_PROGRAM","Test Program","Isolated synthetic program","Test registry behavior")
    registry.create_question("TEST_Q0","TEST_PROGRAM","Does the synthetic mechanism behave?","Infrastructure verification")
    registry.create_hypothesis("TEST_H001","TEST_PROGRAM","TEST_Q0","A deterministic synthetic outcome is reproducible","Tests the registry","Expected known ledger","Any mismatch")
    registry.set_hypothesis_status("TEST_H001",HypothesisStatus.APPROVED)
    return registry,provenance,dataset


def create_experiment(registry, provenance, **changes):
    values=dict(program_id="TEST_PROGRAM",question_id="TEST_Q0",hypothesis_id="TEST_H001",title="Synthetic truth",description="No real strategy",dataset_ids=[provenance.dataset_id],strategy_spec={"family":"synthetic","parameters":{"x":1}},execution_config=ExecutionConfig().model_dump(mode="json"),evaluation_spec={"expected_net_r":2.0},code_version={"method":"TEST","revision":"abc","working_tree":"CLEAN"})
    values.update(changes); return registry.create_experiment(**values)


def test_approved_experiment_and_hypothesis_are_immutable():
    reg,p,_=setup_registry(); exp=create_experiment(reg,p); reg.approve_experiment(exp.experiment_id)
    with pytest.raises(ImmutableRecordError): reg.edit_experiment(exp.experiment_id,title="post-hoc")
    with pytest.raises(ImmutableRecordError): reg.edit_hypothesis_content("TEST_H001",hypothesis_text="post-hoc")


def test_lifecycle_valid_and_invalid_transitions():
    reg,p,_=setup_registry(); exp=create_experiment(reg,p); reg.approve_experiment(exp.experiment_id)
    assert reg.transition_experiment(exp.experiment_id,ExperimentStatus.RUNNING).status == ExperimentStatus.RUNNING
    assert reg.transition_experiment(exp.experiment_id,ExperimentStatus.COMPLETED).status == ExperimentStatus.COMPLETED
    with pytest.raises(InvalidTransitionError): reg.transition_experiment(exp.experiment_id,ExperimentStatus.DRAFT)


def test_fingerprints_are_canonical_and_material_changes_differ():
    base=dict(question_id="Q",hypothesis_id="H",dataset_ids=["b","a"],strategy_spec={"b":2,"a":1},execution_config={"x":1},evaluation_spec={"minimum":0.5},code_version={"revision":"x"})
    same={**base,"dataset_ids":["a","b"],"strategy_spec":{"a":1,"b":2}}
    assert experiment_fingerprint(**base) == experiment_fingerprint(**same)
    assert experiment_fingerprint(**base) != experiment_fingerprint(**{**base,"strategy_spec":{"a":1,"b":3}})
    assert experiment_fingerprint(**base) != experiment_fingerprint(**{**base,"dataset_ids":["c"]})
    assert experiment_fingerprint(**base) != experiment_fingerprint(**{**base,"execution_config":{"x":2}})


def test_duplicate_detected_but_explicit_replication_allowed():
    reg,p,_=setup_registry(); first=create_experiment(reg,p); reg.approve_experiment(first.experiment_id)
    duplicate=create_experiment(reg,p)
    with pytest.raises(DuplicateExperimentError) as error: reg.approve_experiment(duplicate.experiment_id)
    assert error.value.existing_experiment_id == first.experiment_id
    replication=reg.create_replication(first.experiment_id,"RUN-REPLICA","fingerprint",None)
    assert replication["run_kind"] == "REPLICATION" and replication["experiment_id"] == first.experiment_id


@pytest.mark.parametrize("change, message",[
    ({"hypothesis_id":"MISSING"},"missing hypothesis"),
    ({"dataset_ids":["missing"]},"dataset not registered"),
    ({"evaluation_spec":{}},"missing evaluation criteria"),
    ({"execution_config":{}},"missing execution configuration"),
])
def test_pre_registration_blocks_missing_fields(change,message):
    reg,p,_=setup_registry(); exp=create_experiment(reg,p,**change)
    with pytest.raises(PreRegistrationError,match=message): reg.approve_experiment(exp.experiment_id)


def test_dataset_checksum_gate_passes_then_blocks_modification():
    reg,p,path=setup_registry(); assert reg.verify_dataset(p.dataset_id)["checksum"] == p.checksum
    path.write_bytes(b"modified")
    with pytest.raises(DatasetVerificationError,match="checksum mismatch"): reg.verify_dataset(p.dataset_id)


def test_ancestry_preserves_change_reason_and_full_chain():
    reg,p,_=setup_registry(); parent=create_experiment(reg,p); child=create_experiment(reg,p,parent_experiment_id=parent.experiment_id,change_reason="changed synthetic x",strategy_spec={"family":"synthetic","parameters":{"x":2}}); grand=create_experiment(reg,p,parent_experiment_id=child.experiment_id,change_reason="changed synthetic x again",strategy_spec={"family":"synthetic","parameters":{"x":3}})
    chain=reg.experiment_ancestry(grand.experiment_id)
    assert [x["experiment_id"] for x in chain] == [parent.experiment_id,child.experiment_id,grand.experiment_id]
    assert chain[-1]["change_reason"] == "changed synthetic x again"

