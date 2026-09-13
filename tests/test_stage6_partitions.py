from pathlib import Path
from uuid import uuid4
import pandas as pd
import pytest
from sandbox.audit.auditor import ResearchAuditor
from sandbox.audit.models import AuditPhase
from sandbox.control.models import CandidateStatus
from sandbox.partition.models import AccessContext,AccessOperation,PartitionRole
from sandbox.partition.service import PartitionService
from sandbox.provenance import Provenance,sha256_file
from sandbox.research.errors import ResearchError
from sandbox.execution.models import TradeIntent,Direction,EntryType
from test_stage5_control import candidate_fixture

def source_fixture():
    reg,control,_,q,h,family,items,cand=candidate_fixture(2);root=Path("tests/runtime/stage6")/uuid4().hex;root.mkdir(parents=True,exist_ok=True)
    times=list(pd.date_range("2016-01-01",periods=10,freq="365D",tz="UTC"))+list(pd.date_range("2022-12-31 23:00",periods=181,freq="min",tz="UTC"))+[pd.Timestamp("2026-01-01T00:00:00Z")]
    times=sorted(set(times));n=len(times);frame=pd.DataFrame({"timestamp_utc":times,"open":[100+i/1000 for i in range(n)],"high":[101+i/1000 for i in range(n)],"low":[99+i/1000 for i in range(n)],"close":[100.5+i/1000 for i in range(n)],"tick_volume":[1]*n,"spread":[2]*n,"real_volume":[0]*n})
    path=root/"history.parquet";frame.to_parquet(path,index=False);prov=Provenance.create(path,"Synthetic","Fixture","X","M1",times[0].to_pydatetime(),times[-1].to_pydatetime(),n);reg.catalog.add_dataset(prov)
    service=PartitionService(reg,root/"partitions",root/"vault");raw_hash=sha256_file(path)
    discovery=service.create_partition(prov.dataset_id,"TEST_PROGRAM","DISCOVERY","X","M1","2016-01-01T00:00:00Z","2023-01-01T00:00:00Z")
    validation=service.create_partition(prov.dataset_id,"TEST_PROGRAM","VALIDATION","X","M1","2023-01-01T00:00:00Z","2026-01-01T00:00:00Z")
    final=service.create_partition(prov.dataset_id,"TEST_PROGRAM","FINAL_TEST","X","M1","2026-01-01T00:00:00Z","2027-01-01T00:00:00Z")
    return reg,control,service,prov,path,raw_hash,discovery,validation,final,cand,items[-1]

def freeze_for_validation(control,cand,prov):
    return control.freeze_candidate(cand.candidate_id,[prov.dataset_id])

def test_deterministic_partition_fingerprint_boundaries_roles_and_raw_immutable():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();same=s.create_partition(p.dataset_id,"TEST_PROGRAM","DISCOVERY","X","M1","2016-01-01T00:00:00Z","2023-01-01T00:00:00Z")
    assert same.partition_fingerprint==d.partition_fingerprint and same.partition_checksum==d.partition_checksum
    diagnostic=s.create_partition(p.dataset_id,"TEST_PROGRAM","DIAGNOSTIC","X","M1","2016-01-01T00:00:00Z","2017-01-01T00:00:00Z",True)
    assert diagnostic.partition_fingerprint!=d.partition_fingerprint and sha256_file(path)==raw

def test_overlap_blocked_half_open_allowed_and_diagnostic_explicit():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture()
    with pytest.raises(ResearchError,match="OVERLAP"):s.create_partition(p.dataset_id,"TEST_PROGRAM","VALIDATION","X","M1","2022-12-01T00:00:00Z","2024-01-01T00:00:00Z")
    with pytest.raises(ResearchError,match="OVERLAP"):s.create_partition(p.dataset_id,"TEST_PROGRAM","FINAL_TEST","X","M1","2025-01-01T00:00:00Z","2026-06-01T00:00:00Z")
    assert d.end_timestamp==v.start_timestamp
    diag=s.create_partition(p.dataset_id,"TEST_PROGRAM","DIAGNOSTIC","X","M1","2022-01-01T00:00:00Z","2024-01-01T00:00:00Z",True);assert diag.diagnostic_overlap_declared

def test_discovery_access_policy_and_denials_are_logged():
    *_,s,p,path,raw,d,v,f,c,e=source_fixture();assert len(s.access(d.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ"))==d.row_count
    with pytest.raises(ResearchError,match="ACCESS_DENIED"):s.access(v.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ")
    with pytest.raises(ResearchError,match="ACCESS_DENIED"):s.access(f.partition_id,"DISCOVERY_CONTEXT","EXPORT")
    assert any(x["decision"]=="DENIED" for x in s.access_history(f.partition_id)) and s.verify_access_ledger()

def test_validation_binding_fingerprint_and_contamination_gates():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p)
    with pytest.raises(ResearchError,match="binding"):s.access(v.partition_id,"VALIDATION_CONTEXT","EVALUATION_READ",candidate_id=frozen.candidate_id)
    binding=s.bind_validation(frozen.candidate_id,v.partition_id);frame=s.access(v.partition_id,"VALIDATION_CONTEXT","EVALUATION_READ",candidate_id=frozen.candidate_id);assert len(frame)==v.row_count
    with pytest.raises(ResearchError,match="contaminated"):s.access(v.partition_id,"VALIDATION_CONTEXT","EVALUATION_READ",candidate_id=frozen.candidate_id)
    assert binding.candidate_fingerprint==frozen.candidate_fingerprint

def test_validation_wrong_fingerprints_are_denied():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p);binding=s.bind_validation(frozen.candidate_id,v.partition_id)
    data=frozen.model_dump(mode="json");data["candidate_fingerprint"]="changed"
    with reg.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=? WHERE candidate_id=?",(__import__('json').dumps(data),frozen.candidate_id))
    with pytest.raises(ResearchError,match="fingerprint mismatch"):s.access(v.partition_id,"VALIDATION_CONTEXT","EVALUATION_READ",candidate_id=frozen.candidate_id)

def test_system_integrity_reveals_no_market_values_or_contamination():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();result=s.access(f.partition_id,"SYSTEM_INTEGRITY_CONTEXT","CHECKSUM_VERIFY")
    assert result=={"partition_id":f.partition_id,"exists":True,"checksum_valid":True,"row_count_valid":True,"schema_version":"1.0.0"}
    with pytest.raises(ResearchError,match="prohibits"):s.access(f.partition_id,"SYSTEM_INTEGRITY_CONTEXT","AGGREGATE_READ")
    assert not any(x.contamination_type.value=="SYSTEM_INTEGRITY_ACCESS" for x in s.contamination(c.candidate_id))

def test_derived_timeframes_do_not_cross_partition_boundary():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();m15=s.derive_timeframe(d.partition_id,15);h1=s.derive_timeframe(d.partition_id,60)
    assert all((m15.timestamp_utc>=d.start_timestamp)&(m15.timestamp_utc+pd.Timedelta(15*60*1_000_000_000,unit="ns")<=d.end_timestamp)) if not m15.empty else True
    assert all((h1.timestamp_utc>=d.start_timestamp)&(h1.timestamp_utc+pd.Timedelta(60*60*1_000_000_000,unit="ns")<=d.end_timestamp)) if not h1.empty else True
    assert not any(m15.timestamp_utc>=v.start_timestamp) if not m15.empty else True

def test_warmup_requires_frozen_bound_candidate_and_excludes_trading_metrics():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p);s.bind_validation(frozen.candidate_id,v.partition_id);warm=s.warmup(frozen.candidate_id,d.partition_id,v.partition_id,5)
    assert warm["window"].trading_allowed is False and warm["window"].included_in_evaluation is False and warm["window"].evaluation_start==v.start_timestamp
    assert any(x["requested_operation"]=="WARMUP_READ" for x in s.access_history(d.partition_id))

def validated_fixture():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p);s.bind_validation(frozen.candidate_id,v.partition_id);intent=TradeIntent(trade_intent_id="validation-truth",strategy_id="synthetic",experiment_id=e.experiment_id,dataset_id=v.partition_id,symbol="X",direction=Direction.LONG,signal_timestamp=pd.Timestamp("2023-01-01T00:00:00Z").to_pydatetime(),requested_entry_type=EntryType.MARKET_NEXT_OPEN,stop_loss=98,take_profit=100.5);sealed=s.sealed_validation_execute(frozen.candidate_id,v.partition_id,[intent]);return reg,control,s,p,d,v,f,sealed["candidate"],e

def final_intent(f,e):
    return TradeIntent(trade_intent_id="final-truth",strategy_id="synthetic",experiment_id=e.experiment_id,dataset_id=f.partition_id,symbol="X",direction=Direction.LONG,signal_timestamp=pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime(),requested_entry_type=EntryType.MARKET_NEXT_OPEN,stop_loss=99.5,take_profit=200)

def validation_result_id(reg,candidate_id):
    with reg.catalog.connection() as con:return con.execute("SELECT validation_result_id FROM sealed_validation_results WHERE candidate_id=?",(candidate_id,)).fetchone()[0]

def test_final_vault_authorization_sealed_execution_and_repeat_denial():
    reg,control,s,p,d,v,f,c,e=validated_fixture()
    auth=s.authorize_final(c.candidate_id,f.partition_id,validation_result_id(reg,c.candidate_id),"tester");result=s.sealed_final_execute(c.candidate_id,f.partition_id,[final_intent(f,e)])
    assert result["result"]["passed"] is False and auth.candidate_fingerprint==c.candidate_fingerprint
    with pytest.raises(ResearchError,match="already used"):s.sealed_final_execute(c.candidate_id,f.partition_id,[final_intent(f,e)])
    assert any(x.contamination_type.value=="FINAL_TEST_EXPOSURE" for x in s.contamination(c.candidate_id))

def test_final_requires_validation_authorization_and_unchanged_config():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p)
    with pytest.raises(ResearchError,match="successful validation"):s.authorize_final(frozen.candidate_id,f.partition_id,"none","tester")
    reg2,control2,s2,p2,d2,v2,f2,c2,e2=validated_fixture();auth=s2.authorize_final(c2.candidate_id,f2.partition_id,validation_result_id(reg2,c2.candidate_id),"tester");data=c2.model_dump(mode="json");data["execution_config"]={"changed":True}
    with reg2.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=? WHERE candidate_id=?",(__import__('json').dumps(data),c2.candidate_id))
    with pytest.raises(ResearchError,match="fingerprint mismatch"):s2.sealed_final_execute(c2.candidate_id,f2.partition_id,[])

def test_contamination_lineage_aware_and_unrelated_program_not_automatic():
    reg,control,s,p,d,v,f,c,e=validated_fixture();records=s.contamination(c.candidate_id);assert records and all(x.program_id=="TEST_PROGRAM" for x in records)
    child=control.new_candidate_version(c.candidate_id,strategy_spec={"threshold":999});assert s.contamination(child.candidate_id)
    with reg.catalog.connection() as con:assert not con.execute("SELECT 1 FROM contamination_records WHERE program_id='UNRELATED'").fetchone()

def test_partition_role_immutable_and_access_history_tamper_detected():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();s.access(d.partition_id,"DISCOVERY_CONTEXT","RAW_CANDLE_READ")
    with pytest.raises(ResearchError,match="immutable"):s.change_role(d.partition_id,"FINAL_TEST")
    with reg.catalog.connection() as con:con.execute("UPDATE partition_access_ledger SET reason='tampered' WHERE sequence=1")
    with pytest.raises(ResearchError,match="integrity"):s.verify_access_ledger()

def test_partition_manifest_and_stage3_context_include_stage6_evidence():
    reg,control,s,p,path,raw,d,v,f,c,e=source_fixture();frozen=freeze_for_validation(control,c,p);s.bind_validation(frozen.candidate_id,v.partition_id);context=reg.audit_context(e.experiment_id)
    assert any(x["partition_fingerprint"]==v.partition_fingerprint for x in context["partitions"]);assert context["access_policy_version"]=="1.0.0"
