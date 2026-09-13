import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
import pytest
from sandbox.cli import _assert_unprotected_market_path
from sandbox.execution.models import TradeIntent,Direction,EntryType
from sandbox.partition.service import PartitionService
from sandbox.provenance import Provenance
from sandbox.research.errors import ResearchError,DatasetVerificationError
from sandbox.research.registry import ResearchRegistry
from sandbox.statistics.service import StatisticalEvidenceEngine
from sandbox.audit.auditor import ResearchAuditor
from sandbox.audit.models import AuditPhase
from test_stage4_auditor import approved
from test_stage6_partitions import source_fixture,freeze_for_validation,validated_fixture,validation_result_id,final_intent

def test_audit_public_partition_metadata_never_resolves_protected_path():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture()
    assert d.path==v.path==f.path=="[PROTECTED_PARTITION_PATH]" and s.get_partition(f.partition_id).path=="[PROTECTED_PARTITION_PATH]"
    frozen=freeze_for_validation(control,c,p);s.bind_validation(frozen.candidate_id,v.partition_id);context=reg.audit_context(e.experiment_id)
    assert all(x["path"]=="[PROTECTED_PARTITION_PATH]" for x in context["partitions"])

def test_audit_manual_validation_result_and_arbitrary_callback_are_denied():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p);s.bind_validation(frozen.candidate_id,v.partition_id)
    with pytest.raises(ResearchError,match="FORGERY_DENIED"):s.complete_validation(frozen.candidate_id,v.partition_id,True,"fake")
    with pytest.raises(ResearchError,match="callback"):s.sealed_validation_execute(frozen.candidate_id,v.partition_id,lambda frame:{"passed":True})

def test_audit_final_requires_real_validation_and_arbitrary_callback_denied():
    reg,control,s,p,d,v,f,c,e=validated_fixture()
    with pytest.raises(ResearchError,match="verified sealed validation"):s.authorize_final(c.candidate_id,f.partition_id,"invented","attacker")
    s.authorize_final(c.candidate_id,f.partition_id,validation_result_id(reg,c.candidate_id),"auditor")
    with pytest.raises(ResearchError,match="callback prohibited"):s.sealed_final_execute(c.candidate_id,f.partition_id,lambda frame:frame.to_dict())

def test_audit_unauthorized_final_attempt_permanently_blocks_program_reuse():
    reg,control,s,p,d,v,f,c,e=validated_fixture()
    with pytest.raises(ResearchError,match="sealed"):s.access(f.partition_id,"FINAL_TEST_CONTEXT","RAW_CANDLE_READ",candidate_id=c.candidate_id)
    with pytest.raises(ResearchError,match="anywhere in this program"):s.authorize_final(c.candidate_id,f.partition_id,validation_result_id(reg,c.candidate_id),"attacker")

def test_audit_sealed_final_result_exposes_no_artifact_paths():
    reg,control,s,p,d,v,f,c,e=validated_fixture();s.authorize_final(c.candidate_id,f.partition_id,validation_result_id(reg,c.candidate_id),"auditor");result=s.sealed_final_execute(c.candidate_id,f.partition_id,[final_intent(f,e)])
    serialized=json.dumps(result);assert "ledger_path" not in serialized and "summary_path" not in serialized and "parquet" not in serialized and "authorized_criteria" in serialized

def test_audit_stage2_cli_path_guard_blocks_partition_and_protected_source():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture()
    with pytest.raises(ResearchError,match="PROTECTED_DATA_PATH"):_assert_unprotected_market_path(reg.catalog,path)
    with reg.catalog.connection() as con:protected=Path(con.execute("SELECT path FROM data_partitions WHERE partition_id=?",(f.partition_id,)).fetchone()[0])
    with pytest.raises(ResearchError,match="PROTECTED_DATA_PATH"):_assert_unprotected_market_path(reg.catalog,protected)

def test_audit_provenance_catalog_and_seal_tampering_detected():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture()
    with reg.catalog.connection() as con:con.execute("UPDATE datasets SET broker='MetaQuotes Ltd.' WHERE dataset_id=?",(p.dataset_id,))
    with pytest.raises(DatasetVerificationError,match="provenance/catalog mismatch"):reg.verify_dataset(p.dataset_id)
    reg2,control2,s2,p2,path2,raw2,d2,v2,f2,c2,e2=source_fixture()
    with reg2.catalog.connection() as con:row=json.loads(con.execute("SELECT record_json FROM provenance WHERE dataset_id=?",(p2.dataset_id,)).fetchone()[0]);row["server"]="MetaQuotes-Demo";con.execute("UPDATE provenance SET record_json=? WHERE dataset_id=?",(json.dumps(row,sort_keys=True),p2.dataset_id))
    with pytest.raises(DatasetVerificationError,match="seal mismatch"):reg2.verify_dataset(p2.dataset_id)

def test_audit_partition_metadata_empty_and_integrity_attacks_blocked():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture()
    with pytest.raises(ResearchError,match="METADATA_MISMATCH"):s.create_partition(p.dataset_id,"OTHER","DIAGNOSTIC","WRONG","M1","2030-01-01T00:00:00Z","2031-01-01T00:00:00Z")
    with pytest.raises(ResearchError,match="EMPTY_PARTITION"):s.create_partition(p.dataset_id,"OTHER","DIAGNOSTIC","X","M1","2030-01-01T00:00:00Z","2031-01-01T00:00:00Z")
    root=Path("tests/runtime")/("audit-corrupt-"+uuid4().hex);root.mkdir(parents=True);frame=pd.DataFrame({"timestamp_utc":pd.to_datetime(["2020-01-01T00:00:00Z","2020-01-01T00:00:00Z"]),"open":[1,1],"high":[.5,.5],"low":[2,2],"close":[1,1],"tick_volume":[1,1],"spread":[1,1],"real_volume":[0,0]});bad=root/"bad.parquet";frame.to_parquet(bad,index=False);prov=Provenance.create(bad,"Synthetic","Fixture","X","M1",pd.Timestamp("2020-01-01T00:00:00Z").to_pydatetime(),pd.Timestamp("2020-01-01T00:00:00Z").to_pydatetime(),2);reg.catalog.add_dataset(prov)
    with pytest.raises(ResearchError,match="SOURCE_DATA_INTEGRITY_FAILED"):s.create_partition(prov.dataset_id,"OTHER","DIAGNOSTIC","X","M1","2019-01-01T00:00:00Z","2021-01-01T00:00:00Z")
    with pytest.raises(ResearchError,match="UNSAFE_PARTITION_PATH_COMPONENT"):s.create_partition(p.dataset_id,"../escape","DIAGNOSTIC","X","M1","2030-01-01T00:00:00Z","2031-01-01T00:00:00Z")

def test_audit_stage7_direct_protected_modes_require_binding():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();engine=StatisticalEvidenceEngine(reg)
    with pytest.raises(ResearchError,match="candidate and partition"):engine.analyze_run("copied-final","FINAL_TEST")
    with pytest.raises(ResearchError,match="Discovery cannot access"):engine.analyze_run("copied-validation","DISCOVERY",partition_id=v.partition_id)

def test_audit_derived_final_timeframe_and_family_reset_denied():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture()
    with pytest.raises(ResearchError,match="Discovery cannot access"):s.derive_timeframe(f.partition_id,15)
    family=control.get_family("FAMILY-A");control.set_family_status(family.strategy_family_id,__import__("sandbox.control.models",fromlist=["StrategyFamilyStatus"]).StrategyFamilyStatus.EXHAUSTED,"synthetic exhausted")
    with pytest.raises(ResearchError,match="cannot be reset"):control.set_family_status(family.strategy_family_id,__import__("sandbox.control.models",fromlist=["StrategyFamilyStatus"]).StrategyFamilyStatus.ACTIVE)
    with pytest.raises(ResearchError,match="DUPLICATE_STRATEGY_FAMILY"):control.create_family("TEST_PROGRAM","  synthetic FAMILY  ","renamed reset")

@pytest.mark.parametrize("field",["next_candle_close","next_candle_high","future_daily_high","future_atr","future_session"])
def test_audit_explicit_future_field_names_block_without_delay(field):
    strategy={field:True,"temporal_contract":{"confirmation_delay":0,"information_available_at":1,"signal_timestamp":0,"execution_timestamp":0}};*_,exp,auditor=approved(strategy);audit=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    assert "EXPLICIT_LOOKAHEAD" in audit.hard_block_reasons

def test_audit_explicit_future_field_allowed_after_delayed_confirmation():
    strategy={"next_candle_close":True,"temporal_contract":{"confirmation_delay":1,"information_available_at":1,"signal_timestamp":1,"execution_timestamp":2}};*_,exp,auditor=approved(strategy)
    assert "EXPLICIT_LOOKAHEAD" not in auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).hard_block_reasons

def test_audit_tail_deletion_detected_by_journal_and_access_anchors():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();s.access(d.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ")
    with reg.catalog.connection() as con:con.execute("DELETE FROM research_journal WHERE sequence=(SELECT max(sequence) FROM research_journal)")
    with pytest.raises(Exception,match="head/count"):reg.verify_journal_integrity()
    reg2,control2,s2,p2,path2,raw2,d2,v2,f2,c2,e2=source_fixture();s2.access(d2.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ")
    with reg2.catalog.connection() as con:con.execute("DELETE FROM partition_access_ledger WHERE sequence=(SELECT max(sequence) FROM partition_access_ledger)")
    with pytest.raises(Exception,match="head/count"):s2.verify_access_ledger()

def test_audit_clean_sealed_path_has_complete_backward_provenance_and_survives_restart():
    reg,control,s,p,d,v,f,c,e=validated_fixture();vid=validation_result_id(reg,c.candidate_id);s.authorize_final(c.candidate_id,f.partition_id,vid,"auditor");final=s.sealed_final_execute(c.candidate_id,f.partition_id,[final_intent(f,e)]);reloaded=PartitionService(ResearchRegistry(reg.catalog.path,reg.project_root,reg.results_root))
    with reg.catalog.connection() as con:
        final_row=con.execute("SELECT * FROM sealed_final_results WHERE final_result_id=?",(final["final_result_id"],)).fetchone();evidence=con.execute("SELECT * FROM sealed_final_evidence WHERE final_result_id=?",(final["final_result_id"],)).fetchone();sealed_result=json.loads(final_row["result_json"]);assert not con.execute("SELECT 1 FROM research_runs WHERE run_id=?",(sealed_result["run_id"],)).fetchone();candidate=control.get_candidate(c.candidate_id);partition=con.execute("SELECT * FROM data_partitions WHERE partition_id=?",(f.partition_id,)).fetchone();source=reg.verify_dataset(partition["source_dataset_id"])
    assert final_row and evidence and sealed_result["run_fingerprint"] and sealed_result["ledger_checksum"] and candidate.originating_question_id and candidate.originating_hypothesis_id and candidate.strategy_family_id and partition and source["checksum"] and candidate.code_version and candidate.evaluation_spec and candidate.execution_config
    assert reloaded.verify_access_ledger() and reloaded.registry.verify_journal_integrity() and reloaded.contamination(c.candidate_id)

def test_audit_validation_failure_preserved_descendant_returns_to_discovery_without_independence():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p);s.bind_validation(frozen.candidate_id,v.partition_id);loser=TradeIntent(trade_intent_id="validation-loss",strategy_id="synthetic",experiment_id=e.experiment_id,dataset_id=v.partition_id,symbol="X",direction=Direction.LONG,signal_timestamp=pd.Timestamp("2023-01-01T00:00:00Z").to_pydatetime(),requested_entry_type=EntryType.MARKET_NEXT_OPEN,stop_loss=99.5,take_profit=200);failed=s.sealed_validation_execute(frozen.candidate_id,v.partition_id,[loser])["candidate"]
    assert failed.status.value=="REJECTED" and failed.validation_status=="FAILED"
    child=control.new_candidate_version(failed.candidate_id,strategy_spec={"threshold":999});assert child.status.value=="EXPLORATORY" and child.validation_status=="NOT_STARTED"
    with pytest.raises(ResearchError,match="contaminated"):s.bind_validation(control.freeze_candidate(child.candidate_id,[p.dataset_id]).candidate_id,v.partition_id)
    with pytest.raises(ResearchError,match="successful validation"):s.authorize_final(child.candidate_id,f.partition_id,"none","attacker")
