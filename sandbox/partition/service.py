from __future__ import annotations
import json,sqlite3,re
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd
from sandbox import ACCESS_POLICY_VERSION,PARTITION_POLICY_VERSION,PARTITION_SCHEMA_VERSION
from sandbox.audit.models import DatasetUsageType
from sandbox.control.models import CandidateStatus
from sandbox.control.service import ResearchControl
from sandbox.partition.models import AccessContext,AccessDecision,AccessOperation,AccessRecord,ContaminationRecord,ContaminationType,FinalAuthorization,PartitionBinding,PartitionManifest,PartitionRole,WarmupWindow
from sandbox.provenance import sha256_file
from sandbox.research.canonical import canonical_json,sha256_canonical
from sandbox.research.errors import ResearchError
from sandbox.research.models import now_utc
from sandbox.research.registry import ResearchRegistry
from sandbox.integrity import audit

DDL="""
CREATE TABLE IF NOT EXISTS data_partitions(partition_id TEXT PRIMARY KEY,source_dataset_id TEXT NOT NULL,program_id TEXT NOT NULL,role TEXT NOT NULL,symbol TEXT NOT NULL,timeframe TEXT NOT NULL,start_timestamp TEXT NOT NULL,end_timestamp TEXT NOT NULL,partition_fingerprint TEXT NOT NULL UNIQUE,record_json TEXT NOT NULL,path TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS partition_access_ledger(sequence INTEGER PRIMARY KEY AUTOINCREMENT,access_id TEXT NOT NULL UNIQUE,timestamp TEXT NOT NULL,actor TEXT NOT NULL,program_id TEXT NOT NULL,candidate_id TEXT,experiment_id TEXT,partition_id TEXT NOT NULL,requested_operation TEXT NOT NULL,access_context TEXT NOT NULL,decision TEXT NOT NULL,reason TEXT NOT NULL,rows_returned INTEGER NOT NULL,access_policy_version TEXT NOT NULL,previous_entry_hash TEXT NOT NULL,entry_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS partition_access_ledger_head(singleton INTEGER PRIMARY KEY CHECK(singleton=1),entry_count INTEGER NOT NULL,head_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS contamination_records(contamination_id TEXT PRIMARY KEY,program_id TEXT NOT NULL,candidate_id TEXT,candidate_lineage_id TEXT,partition_id TEXT NOT NULL,contamination_type TEXT NOT NULL,source_event TEXT NOT NULL,first_exposed_at TEXT NOT NULL,reason TEXT NOT NULL,severity TEXT NOT NULL,record_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS partition_bindings(binding_id TEXT PRIMARY KEY,binding_type TEXT NOT NULL,candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,record_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(binding_type,candidate_id,partition_id));
CREATE TABLE IF NOT EXISTS sealed_validation_results(validation_result_id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,result_json TEXT NOT NULL,ledger_checksum TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(candidate_id,partition_id));
CREATE TABLE IF NOT EXISTS validation_execution_claims(candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,status TEXT NOT NULL,claimed_at TEXT NOT NULL,PRIMARY KEY(candidate_id,partition_id));
CREATE TABLE IF NOT EXISTS final_authorizations(authorization_id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,record_json TEXT NOT NULL,authorized_at TEXT NOT NULL,UNIQUE(candidate_id,partition_id));
CREATE TABLE IF NOT EXISTS sealed_final_results(final_result_id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,authorization_id TEXT NOT NULL,result_json TEXT NOT NULL,result_fingerprint TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(candidate_id,partition_id));
CREATE TABLE IF NOT EXISTS sealed_final_evidence(evidence_report_id TEXT PRIMARY KEY,final_result_id TEXT NOT NULL UNIQUE,candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,record_json TEXT NOT NULL,report_fingerprint TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS final_execution_claims(candidate_id TEXT NOT NULL,partition_id TEXT NOT NULL,status TEXT NOT NULL,claimed_at TEXT NOT NULL,PRIMARY KEY(candidate_id,partition_id));
CREATE TABLE IF NOT EXISTS dataset_classifications(dataset_id TEXT NOT NULL,classification TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(dataset_id,classification));
"""

class PartitionService:
    def __init__(self,registry:ResearchRegistry,root:Path|None=None,vault_root:Path|None=None):
        self.registry=registry;self.control=ResearchControl(registry);self.root=root or registry.project_root/"data"/"derived"/"partitions";self.vault_root=vault_root or registry.project_root/"data"/"final_vault"
        with registry.catalog.connection() as con:
            con.executescript(DDL)
            if not con.execute("SELECT 1 FROM partition_access_ledger_head WHERE singleton=1").fetchone():
                row=con.execute("SELECT count(*),coalesce((SELECT entry_hash FROM partition_access_ledger ORDER BY sequence DESC LIMIT 1),?) FROM partition_access_ledger",("0"*64,)).fetchone();con.execute("INSERT INTO partition_access_ledger_head VALUES (1,?,?)",(row[0],row[1]))

    def classify_dataset(self,dataset_id,classification,reason):
        with self.registry.catalog.connection() as con:con.execute("INSERT OR IGNORE INTO dataset_classifications VALUES (?,?,?,?)",(dataset_id,classification,reason,now_utc().isoformat()))

    def _source(self,dataset_id):
        record=self.registry.verify_dataset(dataset_id);path=Path(record["verified_path"]);return record,path

    def create_partition(self,source_dataset_id,program_id,role,symbol,timeframe,start_timestamp,end_timestamp,diagnostic_overlap_declared=False):
        if any(not re.fullmatch(r"[A-Za-z0-9._-]+",str(x)) for x in (program_id,symbol,timeframe)):raise ResearchError("UNSAFE_PARTITION_PATH_COMPONENT")
        role=PartitionRole(role);start=pd.Timestamp(start_timestamp);end=pd.Timestamp(end_timestamp)
        if start.tzinfo is None or end.tzinfo is None:raise ResearchError("partition boundaries must be timezone-aware UTC")
        start=start.tz_convert("UTC");end=end.tz_convert("UTC")
        if start>=end:raise ResearchError("partition start must be before end")
        source,path=self._source(source_dataset_id)
        if source["symbol"]!=symbol or source["timeframe"]!=timeframe:raise ResearchError("PARTITION_METADATA_MISMATCH: symbol/timeframe differs from registered source")
        definition={"source_dataset_checksum":source["checksum"],"source_dataset_id":source_dataset_id,"program_id":program_id,"role":role.value,"symbol":symbol,"timeframe":timeframe,"start":start.isoformat(),"end":end.isoformat(),"partition_policy_version":PARTITION_POLICY_VERSION,"diagnostic_overlap_declared":diagnostic_overlap_declared}
        fingerprint=sha256_canonical(definition);partition_id="PART-"+fingerprint[:24]
        with self.registry.catalog.connection() as con:
            existing=con.execute("SELECT record_json FROM data_partitions WHERE partition_fingerprint=?",(fingerprint,)).fetchone()
            overlaps=con.execute("SELECT record_json FROM data_partitions WHERE program_id=? AND symbol=? AND timeframe=? AND start_timestamp<? AND end_timestamp>?",(program_id,symbol,timeframe,end.isoformat(),start.isoformat())).fetchall()
        if existing:return self._public_manifest(PartitionManifest.model_validate_json(existing[0]))
        if overlaps and not (role==PartitionRole.DIAGNOSTIC and diagnostic_overlap_declared):raise ResearchError("PARTITION_OVERLAP_BLOCKED")
        frame=pd.read_parquet(path)
        if {"provider","source_dataset_id","quality_status","continuity_segment_id"}.issubset(frame):
            from sandbox.market_data.storage import validation_view
            integrity=audit(validation_view(frame))
        else:integrity=audit(frame)
        if integrity.status=="error":raise ResearchError("SOURCE_DATA_INTEGRITY_FAILED: "+",".join(x.code for x in integrity.issues if x.category=="confirmed_data_error"))
        if "continuity_segment_id" not in frame:
            from types import SimpleNamespace
            from sandbox.market_data.gaps import inventory
            metadata=SimpleNamespace(provider=source["broker"],symbol=symbol,timeframe=timeframe)
            _,segments=inventory(frame,metadata)
            frame["continuity_segment_id"]=segments
        frame["timestamp_utc"]=pd.to_datetime(frame.timestamp_utc,utc=True);part=frame[(frame.timestamp_utc>=start)&(frame.timestamp_utc<end)].sort_values("timestamp_utc",kind="mergesort").reset_index(drop=True)
        if part.empty:raise ResearchError("EMPTY_PARTITION_BLOCKED")
        target_root=self.vault_root if role==PartitionRole.FINAL_TEST else self.root
        target=target_root/program_id/role.value/symbol/timeframe/f"{partition_id}.parquet";target.parent.mkdir(parents=True,exist_ok=True);part.to_parquet(target,index=False)
        checksum=sha256_file(target);created=now_utc();manifest_basis={**definition,"partition_checksum":checksum,"row_count":len(part),"partition_schema_version":PARTITION_SCHEMA_VERSION}
        manifest=PartitionManifest(partition_id=partition_id,source_dataset_id=source_dataset_id,program_id=program_id,role=role,symbol=symbol,timeframe=timeframe,start_timestamp=start.to_pydatetime(),end_timestamp=end.to_pydatetime(),row_count=len(part),partition_checksum=checksum,source_dataset_checksum=source["checksum"],created_at=created,partition_fingerprint=fingerprint,manifest_fingerprint=sha256_canonical(manifest_basis),path=str(target),diagnostic_overlap_declared=diagnostic_overlap_declared)
        manifest_path=target.with_suffix(".manifest.json");manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"),indent=2,sort_keys=True),encoding="utf-8")
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO data_partitions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(partition_id,source_dataset_id,program_id,role.value,symbol,timeframe,start.isoformat(),end.isoformat(),fingerprint,manifest.model_dump_json(),str(target),created.isoformat()))
        self.registry.append_journal("PARTITION_CREATED",program_id,f"Partition {partition_id} created",metadata={"role":role.value,"fingerprint":fingerprint,"checksum":checksum})
        return self._public_manifest(manifest)

    def _get_partition(self,partition_id):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM data_partitions WHERE partition_id=?",(partition_id,)).fetchone()
        return PartitionManifest.model_validate_json(row[0]) if row else None

    @staticmethod
    def _public_manifest(manifest):
        return manifest.model_copy(update={"path":"[PROTECTED_PARTITION_PATH]"}) if manifest else None

    def get_partition(self,partition_id):
        """Return non-secret partition metadata. Market paths are never part of the public API."""
        return self._public_manifest(self._get_partition(partition_id))

    def verify_partition(self,partition_id):
        p=self._get_partition(partition_id)
        if not p:raise ResearchError("partition not found")
        path=Path(p.path);actual=sha256_file(path) if path.exists() else None
        return {"partition_id":partition_id,"exists":path.exists(),"checksum_valid":actual==p.partition_checksum,"row_count_valid":len(pd.read_parquet(path,columns=["timestamp_utc"]))==p.row_count if path.exists() else False,"schema_version":p.partition_schema_version}

    def _append_access(self,*,actor,program_id,candidate_id,experiment_id,partition_id,operation,context,decision,reason,rows):
        timestamp=now_utc();access_id="ACC-"+uuid4().hex
        with self.registry.catalog.connection() as con:
            con.execute("BEGIN IMMEDIATE");previous=con.execute("SELECT entry_hash FROM partition_access_ledger ORDER BY sequence DESC LIMIT 1").fetchone();previous_hash=previous[0] if previous else "0"*64
            content={"access_id":access_id,"timestamp":timestamp,"actor":actor,"program_id":program_id,"candidate_id":candidate_id,"experiment_id":experiment_id,"partition_id":partition_id,"requested_operation":AccessOperation(operation).value,"access_context":AccessContext(context).value,"decision":AccessDecision(decision).value,"reason":reason,"rows_returned":rows,"access_policy_version":ACCESS_POLICY_VERSION,"previous_entry_hash":previous_hash};entry_hash=sha256_canonical(content)
            con.execute("INSERT INTO partition_access_ledger(access_id,timestamp,actor,program_id,candidate_id,experiment_id,partition_id,requested_operation,access_context,decision,reason,rows_returned,access_policy_version,previous_entry_hash,entry_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(access_id,timestamp.isoformat(),actor,program_id,candidate_id,experiment_id,partition_id,content["requested_operation"],content["access_context"],content["decision"],reason,rows,ACCESS_POLICY_VERSION,previous_hash,entry_hash))
            con.execute("UPDATE partition_access_ledger_head SET entry_count=entry_count+1,head_hash=? WHERE singleton=1",(entry_hash,))
        event="PARTITION_ACCESS_ALLOWED" if decision==AccessDecision.ALLOWED else "PARTITION_ACCESS_DENIED";self.registry.append_journal(event,program_id,f"{event}: {partition_id}",experiment_id=experiment_id,metadata={"access_id":access_id,"operation":content["requested_operation"],"context":content["access_context"],"reason":reason})
        return AccessRecord(**content,entry_hash=entry_hash)

    def verify_access_ledger(self):
        previous="0"*64
        with self.registry.catalog.connection() as con:rows=con.execute("SELECT * FROM partition_access_ledger ORDER BY sequence").fetchall()
        for row in rows:
            content={"access_id":row["access_id"],"timestamp":datetime.fromisoformat(row["timestamp"]),"actor":row["actor"],"program_id":row["program_id"],"candidate_id":row["candidate_id"],"experiment_id":row["experiment_id"],"partition_id":row["partition_id"],"requested_operation":row["requested_operation"],"access_context":row["access_context"],"decision":row["decision"],"reason":row["reason"],"rows_returned":row["rows_returned"],"access_policy_version":row["access_policy_version"],"previous_entry_hash":row["previous_entry_hash"]}
            if row["previous_entry_hash"]!=previous or sha256_canonical(content)!=row["entry_hash"]:raise ResearchError("partition access ledger integrity failure")
            previous=row["entry_hash"]
        with self.registry.catalog.connection() as con:head=con.execute("SELECT entry_count,head_hash FROM partition_access_ledger_head WHERE singleton=1").fetchone()
        if not head or head[0]!=len(rows) or head[1]!=previous:raise ResearchError("partition access ledger head/count integrity failure")
        return True

    def _contaminate(self,program_id,candidate_id,lineage_id,partition_id,kind,source,reason,severity):
        created=now_utc();record=ContaminationRecord(contamination_id="CONT-"+uuid4().hex,program_id=program_id,candidate_id=candidate_id,candidate_lineage_id=lineage_id,partition_id=partition_id,contamination_type=kind,source_event=source,first_exposed_at=created,reason=reason,severity=severity,created_at=created)
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO contamination_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(record.contamination_id,program_id,candidate_id,lineage_id,partition_id,kind.value,source,created.isoformat(),reason,severity,record.model_dump_json(),created.isoformat()))
        self.registry.append_journal("PARTITION_CONTAMINATED",program_id,f"Partition {partition_id}: {kind.value}",metadata={"candidate_id":candidate_id,"lineage":lineage_id,"severity":severity,"reason":reason})
        return record

    def contamination(self,candidate_id):
        candidate=self.control.get_candidate(candidate_id);lineage=candidate.candidate_base_id if candidate else None
        with self.registry.catalog.connection() as con:rows=con.execute("SELECT record_json FROM contamination_records WHERE candidate_id=? OR candidate_lineage_id=? ORDER BY created_at",(candidate_id,lineage)).fetchall()
        return [ContaminationRecord.model_validate_json(x[0]) for x in rows]

    def _deny(self,p,actor,context,operation,reason,candidate_id=None,experiment_id=None):
        self._append_access(actor=actor,program_id=p.program_id,candidate_id=candidate_id,experiment_id=experiment_id,partition_id=p.partition_id,operation=operation,context=context,decision=AccessDecision.DENIED,reason=reason,rows=0)
        if p.role==PartitionRole.FINAL_TEST:self._contaminate(p.program_id,candidate_id,self.control.get_candidate(candidate_id).candidate_base_id if candidate_id and self.control.get_candidate(candidate_id) else None,p.partition_id,ContaminationType.UNAUTHORIZED_ACCESS_ATTEMPT,"ACCESS_DENIED",reason,"CRITICAL")
        raise ResearchError(reason)

    def access(self,partition_id,context,operation,actor="researcher",candidate_id=None,experiment_id=None):
        p=self._get_partition(partition_id);context=AccessContext(context);operation=AccessOperation(operation)
        if not p:raise ResearchError("partition not found")
        if operation==AccessOperation.METADATA_READ:
            self._append_access(actor=actor,program_id=p.program_id,candidate_id=candidate_id,experiment_id=experiment_id,partition_id=partition_id,operation=operation,context=context,decision=AccessDecision.ALLOWED,reason="sanitized metadata only",rows=0)
            return {"partition_id":p.partition_id,"role":p.role.value,"symbol":p.symbol,"timeframe":p.timeframe,"start_timestamp":p.start_timestamp,"end_timestamp":p.end_timestamp,"row_count":p.row_count,"checksum":p.partition_checksum,"fingerprint":p.partition_fingerprint}
        if context==AccessContext.SYSTEM_INTEGRITY_CONTEXT:
            if operation not in {AccessOperation.INTEGRITY_CHECK,AccessOperation.CHECKSUM_VERIFY}:return self._deny(p,actor,context,operation,"SYSTEM_INTEGRITY_CONTEXT prohibits market statistics and values",candidate_id,experiment_id)
            result=self.verify_partition(partition_id);self._append_access(actor=actor,program_id=p.program_id,candidate_id=candidate_id,experiment_id=experiment_id,partition_id=partition_id,operation=operation,context=context,decision=AccessDecision.ALLOWED,reason="non-revealing integrity result",rows=0);return result
        if context==AccessContext.DISCOVERY_CONTEXT and p.role not in {PartitionRole.DISCOVERY,PartitionRole.DIAGNOSTIC}:return self._deny(p,actor,context,operation,"ACCESS_DENIED: Discovery cannot access Validation or Final Test",candidate_id,experiment_id)
        if p.role==PartitionRole.FINAL_TEST:return self._deny(p,actor,context,operation,"FINAL_TEST_ACCESS_DENIED: sealed execution path required",candidate_id,experiment_id)
        if context==AccessContext.VALIDATION_CONTEXT:
            if p.role!=PartitionRole.VALIDATION:return self._deny(p,actor,context,operation,"VALIDATION_ACCESS_DENIED: wrong partition role",candidate_id,experiment_id)
            binding=self._binding("VALIDATION",candidate_id,partition_id)
            if not binding:return self._deny(p,actor,context,operation,"VALIDATION_ACCESS_DENIED: immutable binding required",candidate_id,experiment_id)
            candidate=self.control.get_candidate(candidate_id)
            if not candidate or candidate.candidate_fingerprint!=binding.candidate_fingerprint:return self._deny(p,actor,context,operation,"VALIDATION_ACCESS_DENIED: candidate fingerprint mismatch",candidate_id,experiment_id)
            if sha256_canonical(candidate.evaluation_spec)!=binding.evaluation_spec_fingerprint:return self._deny(p,actor,context,operation,"VALIDATION_ACCESS_DENIED: evaluation fingerprint mismatch",candidate_id,experiment_id)
            if self._lineage_contaminated(candidate,partition_id):return self._deny(p,actor,context,operation,"VALIDATION_ACCESS_DENIED: partition contaminated for lineage",candidate_id,experiment_id)
        frame=pd.read_parquet(p.path);self._append_access(actor=actor,program_id=p.program_id,candidate_id=candidate_id,experiment_id=experiment_id,partition_id=partition_id,operation=operation,context=context,decision=AccessDecision.ALLOWED,reason="policy requirements satisfied",rows=len(frame))
        if context==AccessContext.DISCOVERY_CONTEXT:self._contaminate(p.program_id,candidate_id,self.control.get_candidate(candidate_id).candidate_base_id if candidate_id and self.control.get_candidate(candidate_id) else None,partition_id,ContaminationType.DISCOVERY_EXPOSURE,operation.value,"discovery values exposed","INFO")
        if context==AccessContext.VALIDATION_CONTEXT and operation in {AccessOperation.EVALUATION_READ,AccessOperation.BACKTEST_EXECUTION,AccessOperation.RAW_CANDLE_READ}:candidate=self.control.get_candidate(candidate_id);self._contaminate(p.program_id,candidate_id,candidate.candidate_base_id,partition_id,ContaminationType.VALIDATION_EXPOSURE,operation.value,"validation values/results exposed","HIGH")
        return frame

    def _lineage_contaminated(self,candidate,partition_id):
        with self.registry.catalog.connection() as con:return bool(con.execute("SELECT 1 FROM contamination_records WHERE program_id=? AND candidate_lineage_id=? AND partition_id=? AND contamination_type IN ('VALIDATION_EXPOSURE','RESULT_DRIVEN_REUSE','FINAL_TEST_EXPOSURE')",(candidate.program_id,candidate.candidate_base_id,partition_id)).fetchone())

    def _final_partition_exposed(self,program_id,partition_id):
        with self.registry.catalog.connection() as con:return bool(con.execute("SELECT 1 FROM contamination_records WHERE program_id=? AND partition_id=? AND contamination_type IN ('FINAL_TEST_EXPOSURE','UNAUTHORIZED_ACCESS_ATTEMPT')",(program_id,partition_id)).fetchone())

    def bind_validation(self,candidate_id,partition_id):
        candidate=self.control.get_candidate(candidate_id);p=self._get_partition(partition_id)
        if not candidate or candidate.status!=CandidateStatus.FROZEN:raise ResearchError("VALIDATION_BIND_DENIED: candidate must be frozen")
        if not p or p.role!=PartitionRole.VALIDATION:raise ResearchError("VALIDATION_BIND_DENIED: Validation partition required")
        if p.source_dataset_id not in candidate.validation_dataset_ids:raise ResearchError("VALIDATION_BIND_DENIED: partition source not designated")
        if self._lineage_contaminated(candidate,partition_id):raise ResearchError("VALIDATION_BIND_DENIED: contaminated lineage")
        gate=self.control.validation_eligibility(candidate_id)
        if not gate["eligible"]:raise ResearchError("VALIDATION_BIND_DENIED: "+"; ".join(gate["reasons"]))
        record=PartitionBinding(binding_id="BIND-"+uuid4().hex,binding_type="VALIDATION",candidate_id=candidate_id,candidate_fingerprint=candidate.candidate_fingerprint,partition_id=partition_id,evaluation_spec_fingerprint=sha256_canonical(candidate.evaluation_spec),execution_config_fingerprint=sha256_canonical(candidate.execution_config),created_at=now_utc())
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO partition_bindings VALUES (?,?,?,?,?,?,?)",(record.binding_id,"VALIDATION",candidate_id,partition_id,record.model_dump_json(),record.status,record.created_at.isoformat()))
        self.registry.append_journal("VALIDATION_BOUND",candidate.program_id,f"{candidate_id} bound to {partition_id}",metadata={"binding_id":record.binding_id});return record

    def _binding(self,binding_type,candidate_id,partition_id):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM partition_bindings WHERE binding_type=? AND candidate_id=? AND partition_id=?",(binding_type,candidate_id,partition_id)).fetchone()
        return PartitionBinding.model_validate_json(row[0]) if row else None

    def complete_validation(self,*args,**kwargs):
        raise ResearchError("VALIDATION_RESULT_FORGERY_DENIED: use sealed_validation_execute; pass/fail cannot be supplied manually")

    def sealed_validation_execute(self,candidate_id,partition_id,intents,actor="sealed-validator",results_root:Path|None=None):
        candidate=self.control.get_candidate(candidate_id);p=self._get_partition(partition_id);binding=self._binding("VALIDATION",candidate_id,partition_id)
        if not candidate or not binding or candidate.status!=CandidateStatus.FROZEN:raise ResearchError("VALIDATION_ACCESS_DENIED: frozen bound candidate required")
        if callable(intents):raise ResearchError("VALIDATION_ACCESS_DENIED: arbitrary execution callback prohibited")
        if candidate.candidate_fingerprint!=binding.candidate_fingerprint or sha256_canonical(candidate.execution_config)!=binding.execution_config_fingerprint or sha256_canonical(candidate.evaluation_spec)!=binding.evaluation_spec_fingerprint:raise ResearchError("VALIDATION_ACCESS_DENIED: frozen fingerprint mismatch")
        with self.registry.catalog.connection() as con:
            if con.execute("SELECT 1 FROM sealed_validation_results WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id)).fetchone():raise ResearchError("VALIDATION_ACCESS_DENIED: validation already executed")
            try:con.execute("INSERT INTO validation_execution_claims VALUES (?,?,?,?)",(candidate_id,partition_id,"RUNNING",now_utc().isoformat()))
            except sqlite3.IntegrityError:raise ResearchError("VALIDATION_ACCESS_DENIED: execution already claimed")
        from sandbox.execution.engine import BacktestEngine
        from sandbox.execution.models import ExecutionConfig,TradeIntent
        from sandbox.execution.artifacts import store_run
        frame=pd.read_parquet(p.path);validated=[x if isinstance(x,TradeIntent) else TradeIntent.model_validate(x) for x in intents]
        if not validated or any(x.dataset_id!=partition_id for x in validated):raise ResearchError("VALIDATION_ACCESS_DENIED: non-empty intents must be bound to the Validation partition_id")
        run=BacktestEngine(ExecutionConfig.model_validate(candidate.execution_config)).run(validated,frame.copy());stored=store_run(run,self.registry.results_root/"validation");self.registry.catalog.add_research_run(stored.run_id,stored.run_fingerprint,partition_id,run.execution_engine_version,run.ledger_schema_version,run.metrics_version,candidate.execution_config,str(stored.ledger_path),str(stored.summary_path));summary=json.loads(stored.summary_path.read_text(encoding="utf-8"));metrics=summary["metrics"];mapping={"minimum_win_rate":"win_rate","minimum_trades_per_year":"trades_per_year","minimum_resolved_trades":"resolved_trades","minimum_net_r":"net_R"};criteria=[]
        for criterion,required in sorted(candidate.evaluation_spec.items()):
            metric=mapping.get(criterion);actual=metrics.get(metric) if metric else None;criteria.append({"criterion":criterion,"required":required,"actual":actual,"result":"INCONCLUSIVE" if actual is None else "CRITERION_MET" if float(actual)>=float(required) else "CRITERION_NOT_MET"})
        passed=bool(criteria) and all(x["result"]=="CRITERION_MET" for x in criteria);result_id="VAL-"+uuid4().hex;result={"validation_result_id":result_id,"passed":passed,"run_id":stored.run_id,"run_fingerprint":stored.run_fingerprint,"ledger_checksum":stored.ledger_checksum,"criteria":criteria};status=CandidateStatus.VALIDATED if passed else CandidateStatus.REJECTED;updated=candidate.model_copy(update={"status":status,"validation_status":"PASSED" if passed else "FAILED"});created=now_utc()
        from sandbox.statistics.service import StatisticalEvidenceEngine
        evidence_engine=StatisticalEvidenceEngine(self.registry);evidence=evidence_engine.analyze_frame(pd.read_parquet(stored.ledger_path),stored.run_id,stored.ledger_checksum,"VALIDATION",candidate_id=candidate_id,partition_id=partition_id,evaluation_spec=candidate.evaluation_spec);evidence_engine.store(evidence);result["evidence_report_id"]=evidence.report_id
        with self.registry.catalog.connection() as con:
            con.execute("INSERT INTO sealed_validation_results VALUES (?,?,?,?,?,?)",(result_id,candidate_id,partition_id,canonical_json(result),stored.ledger_checksum,created.isoformat()));con.execute("UPDATE candidates SET record_json=?,status=? WHERE candidate_id=?",(updated.model_dump_json(),status.value,candidate_id));con.execute("UPDATE partition_bindings SET status='EXPOSED' WHERE binding_id=?",(binding.binding_id,));con.execute("UPDATE validation_execution_claims SET status='COMPLETED' WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id))
        self._append_access(actor=actor,program_id=p.program_id,candidate_id=candidate_id,experiment_id=None,partition_id=partition_id,operation=AccessOperation.BACKTEST_EXECUTION,context=AccessContext.VALIDATION_CONTEXT,decision=AccessDecision.ALLOWED,reason="sealed validation execution",rows=len(frame));self._contaminate(candidate.program_id,candidate_id,candidate.candidate_base_id,partition_id,ContaminationType.VALIDATION_EXPOSURE,"VALIDATION_RESULT",f"validation result {result_id} revealed","HIGH");self.registry.append_journal("VALIDATION_EXPOSED",candidate.program_id,f"Validation result exposed for {candidate_id}",metadata={"partition_id":partition_id,"passed":passed,"result_id":result_id,"evidence_report_id":evidence.report_id});return {"candidate":updated,"result":result}

    @staticmethod
    def _strategy_timeframe_frames(frame,plugin,source_timeframe="M1"):
        """Build complete causal bars inside one already-authorized partition boundary."""
        source=frame.copy();source["timestamp_utc"]=pd.to_datetime(source.timestamp_utc,utc=True);frames={}
        def tf_minutes(value):
            match=re.fullmatch(r"([MH])(\d+)",value)
            if not match:raise ResearchError("REQUIRED_TIMEFRAME_UNAVAILABLE: "+value)
            return int(match.group(2))*(60 if match.group(1)=="H" else 1)
        source_minutes=tf_minutes(source_timeframe)
        if source_timeframe in plugin.metadata.required_timeframes:frames[source_timeframe]=source
        for timeframe in plugin.metadata.required_timeframes:
            if timeframe in frames:continue
            minutes=tf_minutes(timeframe)
            if minutes<source_minutes or minutes%source_minutes:raise ResearchError("REQUIRED_TIMEFRAME_UNAVAILABLE: non-divisible source timeframe")
            expected_rows=minutes//source_minutes;work=source.copy();work["bucket"]=work.timestamp_utc.dt.floor(f"{minutes}min");rows=[]
            for _,g in work.groupby("bucket",sort=True):
                if len(g)!=expected_rows:continue
                rows.append({"timestamp_utc":g.bucket.iloc[0],"open":g.open.iloc[0],"high":g.high.max(),"low":g.low.min(),"close":g.close.iloc[-1],"tick_volume":g.tick_volume.sum(),"spread":g.spread.iloc[-1],"real_volume":g.real_volume.sum()})
            frames[timeframe]=pd.DataFrame(rows,columns=["timestamp_utc","open","high","low","close","tick_volume","spread","real_volume"])
        return frames

    def sealed_validation_strategy_execute(self,candidate_id,partition_id,strategy_registry,actor="sealed-strategy-validator"):
        from sandbox.strategies.runtime import StrategyRuntime,_PROTECTED_PHASE_AUTHORITY
        candidate=self.control.get_candidate(candidate_id);p=self._get_partition(partition_id);binding=self._binding("VALIDATION",candidate_id,partition_id)
        if not candidate or not p or not binding or candidate.status!=CandidateStatus.FROZEN:raise ResearchError("VALIDATION_ACCESS_DENIED: frozen bound candidate required")
        if candidate.candidate_fingerprint!=binding.candidate_fingerprint:raise ResearchError("VALIDATION_ACCESS_DENIED: frozen fingerprint mismatch")
        spec=candidate.strategy_spec;plugin=strategy_registry.resolve(spec.get("strategy_id"),spec.get("strategy_version"));frame=pd.read_parquet(p.path);runtime=StrategyRuntime(strategy_registry);frames=self._strategy_timeframe_frames(frame,plugin,p.timeframe)
        run=runtime.assert_deterministic(plugin.metadata.strategy_id,frames,symbol=p.symbol,dataset_id=partition_id,experiment_id=candidate.originating_hypothesis_id,parameters=spec.get("parameters",{}),phase="VALIDATION",version=plugin.metadata.strategy_version,_phase_authority=_PROTECTED_PHASE_AUTHORITY);runtime.assert_frozen_binding(candidate,run)
        return self.sealed_validation_execute(candidate_id,partition_id,run.intents,actor=actor)

    def authorize_final(self,candidate_id,partition_id,validation_result_id,authorized_by):
        candidate=self.control.get_candidate(candidate_id);p=self._get_partition(partition_id)
        if not candidate or candidate.status!=CandidateStatus.VALIDATED or candidate.validation_status!="PASSED":raise ResearchError("FINAL_TEST_ACCESS_DENIED: successful validation required")
        with self.registry.catalog.connection() as con:validated=con.execute("SELECT 1 FROM sealed_validation_results WHERE validation_result_id=? AND candidate_id=? AND json_extract(result_json,'$.passed')=1",(validation_result_id,candidate_id)).fetchone()
        if not validated:raise ResearchError("FINAL_TEST_ACCESS_DENIED: verified sealed validation result required")
        if not p or p.role!=PartitionRole.FINAL_TEST:raise ResearchError("FINAL_TEST_ACCESS_DENIED: Final partition required")
        if self._lineage_contaminated(candidate,partition_id) or self._final_partition_exposed(candidate.program_id,partition_id):raise ResearchError("FINAL_TEST_ACCESS_DENIED: contaminated or already used anywhere in this program")
        record=FinalAuthorization(authorization_id="AUTH-"+uuid4().hex,candidate_id=candidate_id,candidate_fingerprint=candidate.candidate_fingerprint,partition_id=partition_id,validation_result_id=validation_result_id,evaluation_spec_fingerprint=sha256_canonical(candidate.evaluation_spec),execution_config_fingerprint=sha256_canonical(candidate.execution_config),authorized_by=authorized_by,authorized_at=now_utc())
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO final_authorizations VALUES (?,?,?,?,?)",(record.authorization_id,candidate_id,partition_id,record.model_dump_json(),record.authorized_at.isoformat()))
        self.registry.append_journal("FINAL_TEST_AUTHORIZED",candidate.program_id,f"Final test authorized for {candidate_id}",metadata={"partition_id":partition_id,"authorization_id":record.authorization_id});return record

    def sealed_final_execute(self,candidate_id,partition_id,intents,actor="sealed-runner",results_root:Path|None=None):
        """Execute the installed deterministic engine inside the vault; arbitrary callbacks are prohibited."""
        candidate=self.control.get_candidate(candidate_id);p=self._get_partition(partition_id)
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM final_authorizations WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id)).fetchone();used=con.execute("SELECT 1 FROM sealed_final_results WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id)).fetchone()
        if not row:return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: authorization required",candidate_id,None)
        if used:return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: independent final test already used",candidate_id,None)
        auth=FinalAuthorization.model_validate_json(row[0])
        if candidate.candidate_fingerprint!=auth.candidate_fingerprint or sha256_canonical(candidate.execution_config)!=auth.execution_config_fingerprint or sha256_canonical(candidate.evaluation_spec)!=auth.evaluation_spec_fingerprint:return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: frozen fingerprint mismatch",candidate_id,None)
        if self._lineage_contaminated(candidate,partition_id) or self._final_partition_exposed(candidate.program_id,partition_id):return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: final data exposed",candidate_id,None)
        if callable(intents):return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: arbitrary execution callback prohibited",candidate_id,None)
        claim_created=True
        with self.registry.catalog.connection() as con:
            try:con.execute("INSERT INTO final_execution_claims VALUES (?,?,?,?)",(candidate_id,partition_id,"RUNNING",now_utc().isoformat()))
            except sqlite3.IntegrityError:claim_created=False
        if not claim_created:return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: execution already claimed",candidate_id,None)
        from sandbox.execution.engine import BacktestEngine
        from sandbox.execution.models import ExecutionConfig,TradeIntent
        from sandbox.execution.artifacts import store_run
        frame=pd.read_parquet(p.path);validated_intents=[x if isinstance(x,TradeIntent) else TradeIntent.model_validate(x) for x in intents]
        if not validated_intents or any(x.dataset_id!=partition_id for x in validated_intents):return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: non-empty intents must be bound to the Final partition_id",candidate_id,None)
        run=BacktestEngine(ExecutionConfig.model_validate(candidate.execution_config)).run(validated_intents,frame.copy());stored=store_run(run,self.vault_root/"artifacts")
        summary=json.loads(stored.summary_path.read_text(encoding="utf-8"));metrics=summary["metrics"];mapping={"minimum_win_rate":"win_rate","minimum_trades_per_year":"trades_per_year","minimum_resolved_trades":"resolved_trades","minimum_net_r":"net_R"};criteria=[]
        for criterion,required in sorted(candidate.evaluation_spec.items()):
            metric=mapping.get(criterion);actual=metrics.get(metric) if metric else None;criteria.append({"criterion":criterion,"required":required,"actual":actual,"result":"INCONCLUSIVE" if actual is None else "CRITERION_MET" if float(actual)>=float(required) else "CRITERION_NOT_MET"})
        result={"passed":bool(criteria) and all(x["result"]=="CRITERION_MET" for x in criteria),"run_id":stored.run_id,"run_fingerprint":stored.run_fingerprint,"ledger_checksum":stored.ledger_checksum,"authorized_criteria":criteria};fingerprint=sha256_canonical(result);result_id="FINAL-"+uuid4().hex;created=now_utc()
        from sandbox import STATISTICAL_EVIDENCE_ENGINE_VERSION
        from sandbox.statistics.service import StatisticalEvidenceEngine
        evidence_id="EVID-FINAL-"+uuid4().hex;evidence_basis={"evidence_report_id":evidence_id,"final_result_id":result_id,"candidate_id":candidate_id,"partition_id":partition_id,"run_id":stored.run_id,"ledger_checksum":stored.ledger_checksum,"authorized_criteria":criteria,"evidence_engine_version":STATISTICAL_EVIDENCE_ENGINE_VERSION,"statistical_config_fingerprint":StatisticalEvidenceEngine(self.registry).config_fingerprint};evidence_fingerprint=sha256_canonical(evidence_basis);evidence={**evidence_basis,"report_fingerprint":evidence_fingerprint,"created_at":created.isoformat()};result["evidence_report_id"]=evidence_id;fingerprint=sha256_canonical(result)
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO sealed_final_results VALUES (?,?,?,?,?,?,?)",(result_id,candidate_id,partition_id,auth.authorization_id,canonical_json(result),fingerprint,created.isoformat()));con.execute("INSERT INTO sealed_final_evidence VALUES (?,?,?,?,?,?,?)",(evidence_id,result_id,candidate_id,partition_id,canonical_json(evidence),evidence_fingerprint,created.isoformat()));con.execute("UPDATE final_execution_claims SET status='COMPLETED' WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id))
        self._append_access(actor=actor,program_id=p.program_id,candidate_id=candidate_id,experiment_id=None,partition_id=partition_id,operation=AccessOperation.BACKTEST_EXECUTION,context=AccessContext.FINAL_TEST_CONTEXT,decision=AccessDecision.ALLOWED,reason="sealed authorized final execution",rows=len(frame));self._contaminate(p.program_id,candidate_id,candidate.candidate_base_id,partition_id,ContaminationType.FINAL_TEST_EXPOSURE,"FINAL_TEST_EXECUTED","sealed final result revealed","CRITICAL")
        self.registry.append_journal("FINAL_TEST_EXECUTED",candidate.program_id,f"Final test executed for {candidate_id}",metadata={"partition_id":partition_id,"result_id":result_id,"result_fingerprint":fingerprint});self.registry.append_journal("FINAL_TEST_PASSED" if bool(result.get("passed")) else "FINAL_TEST_FAILED",candidate.program_id,f"Final test outcome recorded for {candidate_id}",metadata={"result_id":result_id});return {"final_result_id":result_id,"result":result,"result_fingerprint":fingerprint}

    def sealed_final_strategy_execute(self,candidate_id,partition_id,strategy_registry,actor="sealed-strategy-final"):
        """Generate signals and execute them without exposing Final market values or the signal stream."""
        from sandbox.strategies.runtime import StrategyRuntime,_PROTECTED_PHASE_AUTHORITY
        candidate=self.control.get_candidate(candidate_id);p=self._get_partition(partition_id)
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM final_authorizations WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id)).fetchone();used=con.execute("SELECT 1 FROM sealed_final_results WHERE candidate_id=? AND partition_id=?",(candidate_id,partition_id)).fetchone()
        if not candidate or not p or not row or used:return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: unused authorization required",candidate_id,None)
        auth=FinalAuthorization.model_validate_json(row[0])
        if candidate.candidate_fingerprint!=auth.candidate_fingerprint:return self._deny(p,actor,AccessContext.FINAL_TEST_CONTEXT,AccessOperation.BACKTEST_EXECUTION,"FINAL_TEST_ACCESS_DENIED: frozen fingerprint mismatch",candidate_id,None)
        spec=candidate.strategy_spec;plugin=strategy_registry.resolve(spec.get("strategy_id"),spec.get("strategy_version"));frame=pd.read_parquet(p.path);runtime=StrategyRuntime(strategy_registry);frames=self._strategy_timeframe_frames(frame,plugin,p.timeframe)
        run=runtime.assert_deterministic(plugin.metadata.strategy_id,frames,symbol=p.symbol,dataset_id=partition_id,experiment_id=candidate.originating_hypothesis_id,parameters=spec.get("parameters",{}),phase="FINAL_TEST",version=plugin.metadata.strategy_version,_phase_authority=_PROTECTED_PHASE_AUTHORITY);runtime.assert_frozen_binding(candidate,run)
        return self.sealed_final_execute(candidate_id,partition_id,run.intents,actor=actor)

    def warmup(self,candidate_id,discovery_partition_id,validation_partition_id,bars,actor="validator"):
        candidate=self.control.get_candidate(candidate_id);discovery=self._get_partition(discovery_partition_id);validation=self._get_partition(validation_partition_id)
        if not candidate or candidate.status not in {CandidateStatus.FROZEN,CandidateStatus.VALIDATING}:raise ResearchError("WARMUP_DENIED: candidate must be frozen")
        if not self._binding("VALIDATION",candidate_id,validation_partition_id):raise ResearchError("WARMUP_DENIED: validation binding required")
        if discovery.role not in {PartitionRole.DISCOVERY,PartitionRole.DIAGNOSTIC}:raise ResearchError("WARMUP_DENIED: pre-validation partition required")
        frame=pd.read_parquet(discovery.path).tail(bars).copy();window=WarmupWindow(partition_id=discovery_partition_id,candidate_id=candidate_id,start_timestamp=pd.to_datetime(frame.timestamp_utc,utc=True).min().to_pydatetime(),end_timestamp=pd.to_datetime(frame.timestamp_utc,utc=True).max().to_pydatetime(),evaluation_start=validation.start_timestamp,rows=len(frame))
        self._append_access(actor=actor,program_id=candidate.program_id,candidate_id=candidate_id,experiment_id=None,partition_id=discovery_partition_id,operation=AccessOperation.WARMUP_READ,context=AccessContext.VALIDATION_CONTEXT,decision=AccessDecision.ALLOWED,reason="warm-up only; trading/evaluation prohibited",rows=len(frame));return {"window":window,"data":frame}

    def derive_timeframe(self,partition_id,minutes,context=AccessContext.DISCOVERY_CONTEXT,candidate_id=None,actor="derived-timeframe"):
        p=self._get_partition(partition_id)
        if not p:raise ResearchError("partition not found")
        source=self.access(partition_id,context,AccessOperation.AGGREGATE_READ,actor=actor,candidate_id=candidate_id);source["timestamp_utc"]=pd.to_datetime(source.timestamp_utc,utc=True);rule=f"{int(minutes)}min";source["bucket"]=source.timestamp_utc.dt.floor(rule)
        groups=[]
        for bucket,g in source.groupby("bucket"):
            expected_end=bucket+pd.Timedelta(int(minutes)*60*1_000_000_000,unit="ns")
            if bucket<p.start_timestamp or expected_end>p.end_timestamp or len(g)!=minutes:continue
            groups.append({"timestamp_utc":bucket,"open":g.open.iloc[0],"high":g.high.max(),"low":g.low.min(),"close":g.close.iloc[-1],"tick_volume":g.tick_volume.sum(),"spread":g.spread.iloc[-1],"real_volume":g.real_volume.sum()})
        return pd.DataFrame(groups)

    def access_history(self,partition_id):
        with self.registry.catalog.connection() as con:return [dict(x) for x in con.execute("SELECT * FROM partition_access_ledger WHERE partition_id=? ORDER BY sequence",(partition_id,))]

    def change_role(self,partition_id,new_role):
        p=self._get_partition(partition_id)
        if not p:raise ResearchError("partition not found")
        raise ResearchError("partition role is immutable; create a new partition/version with preserved history")
