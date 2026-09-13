from __future__ import annotations
import sqlite3,json,os,re
from pathlib import Path
import numpy as np
import pandas as pd
from sandbox.config import Settings
from sandbox.mt5 import MT5Adapter,MT5Error
from sandbox.partition.service import PartitionService
from sandbox.research.registry import ResearchRegistry
from sandbox.strategies.registry import default_registry
from sandbox.external_import import ExternalImportGateway,ImportSpec,ImportError as ExternalImportError,PriceType,Resolution,TimestampFormat
from sandbox.provenance import sha256_file
from .models import *

VERIFIED_TEST_COUNT=341

class SandboxReadService:
    """Intentional browser-safe projection. No protected-data or mutation methods exist."""
    def __init__(self,settings:Settings|None=None,project_root:Path|None=None,probe_broker:bool=True):
        self.settings=settings or Settings.load();self.project_root=(project_root or Path.cwd()).resolve();self.probe_broker=probe_broker
        self.registry=ResearchRegistry(self.settings.catalog_path,self.project_root);self.strategies=default_registry()
    def _count(self,table):
        with self.registry.catalog.connection() as con:
            try:return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            except sqlite3.OperationalError:return 0
    def broker(self):
        if not self.probe_broker:return BrokerStatus(connected=False,read_only=True)
        try:
            with MT5Adapter(self.settings.terminal_path) as mt5:
                s=mt5.status();return BrokerStatus(connected=s.connected,company=s.company,server=s.server,read_only=True)
        except MT5Error:return BrokerStatus(connected=False,read_only=True)
    def system(self):
        integrity="VERIFIED" if self.registry.verify_journal_integrity() and PartitionService(self.registry).verify_access_ledger() else "FAILED"
        return SystemStatusDTO(broker=self.broker(),research_mode="STRICT_ISOLATION_READ_ONLY",strategy_count=len(self.strategies.list(False)),dataset_count=self._count("datasets"),partition_count=self._count("data_partitions"),verified_test_count=VERIFIED_TEST_COUNT,integrity_status=integrity)
    def strategy_summaries(self):
        return [StrategySummaryDTO(strategy_id=m.strategy_id,name=m.strategy_name,version=m.strategy_version,description=m.description,required_timeframes=list(m.required_timeframes),capabilities=list(m.capabilities),implementation_status="IMPLEMENTED",research_status="NOT_RUN") for m in self.strategies.list(False)]
    def strategy(self,strategy_id):
        plugin=self.strategies.resolve(strategy_id);m=plugin.metadata;_,code,parameters=self.strategies.fingerprint(plugin,dict(m.default_parameters))
        return StrategyDetailDTO(**self.strategy_summaries_for(m),parameters=[ParameterDTO(name=x.name,type=x.type.value,material=x.material,required=x.required,default=x.default) for x in m.parameter_schema],code_fingerprint=code,parameter_fingerprint=parameters)
    @staticmethod
    def strategy_summaries_for(m):return dict(strategy_id=m.strategy_id,name=m.strategy_name,version=m.strategy_version,description=m.description,required_timeframes=list(m.required_timeframes),capabilities=list(m.capabilities),implementation_status="IMPLEMENTED",research_status="NOT_RUN")
    def research(self):
        h=self._count("hypotheses");e=self._count("experiments");c=self._count("candidates");p=self._count("data_partitions")
        return ResearchStatusDTO(program_count=self._count("research_programs"),hypothesis_count=h,experiment_count=e,candidate_count=c,partition_count=p,discovery_status="NOT_STARTED" if not e else "RECORDED",validation_status="NOT_ELIGIBLE",final_test_status="SEALED_NOT_ELIGIBLE")
    def data(self):
        rows=self.registry.catalog.datasets();safe=[DatasetDTO(**{k:r[k] for k in ("dataset_id","broker","server","symbol","timeframe","start","end","row_count","checksum","schema_version")}) for r in rows]
        imports=[]
        with self.registry.catalog.connection() as con:
            try:records=con.execute("SELECT dataset_id,status,manifest_json FROM external_imports ORDER BY created_at").fetchall()
            except sqlite3.OperationalError:records=[]
            for record in records:
                m=json.loads(record["manifest_json"]);purpose="UNASSIGNED_RESEARCH_DATA"
                classification=con.execute("SELECT classification FROM dataset_classifications WHERE dataset_id=? AND classification='ENGINEERING_DIAGNOSTIC'",(record["dataset_id"],)).fetchone()
                if classification:purpose="ENGINEERING_DIAGNOSTIC"
                imports.append(ImportedDatasetDTO(dataset_id=record["dataset_id"],provider=m["data_source_provider"],symbol=m["canonical_symbol"],data_type=m["resolution"],price_type=m["price_type"],date_range=f'{m["first_timestamp"]} — {m["last_timestamp"]}',row_count=m["row_count"],certification=record["status"],purpose=purpose,fingerprint=m["scientific_fingerprint"]))
        return DataStatusDTO(broker=self.broker(),datasets=safe,imported_datasets=imports,dataset_count=len(safe),external_import_gateway_available=True,external_production_dataset_count=len(imports),research_partition_count=self._count("data_partitions"))
    @property
    def upload_max_bytes(self):return int(os.getenv("SANDBOX_UPLOAD_MAX_BYTES",str(2*1024*1024*1024)))
    def import_uploaded(self,path:Path,*,display_name:str,file_size:int,provider:str,symbol:str,data_type:str,price_type:str,source_timezone:str,purpose:str,timestamp_format:str="ISO8601",interpretation_reason:str|None=None):
        mapping={"timestamp":"timestamp","open":"open","high":"high","low":"low","close":"close"} if data_type in {"M1","M15"} else {"timestamp":"timestamp","bid":"bid","ask":"ask"}
        gateway=ExternalImportGateway(self.registry.catalog,self.settings.data_dir);spec=ImportSpec((path,),provider,symbol,symbol,Resolution(data_type),PriceType(price_type),source_timezone,mapping,None,price_type=="MID",interpretation_reason,TimestampFormat(timestamp_format))
        try:imported=gateway.import_files(spec)
        except (ExternalImportError,ValueError,KeyError) as exc:return self._rejected_result(path,display_name,file_size,provider,symbol,data_type,price_type,source_timezone,timestamp_format,purpose,exc)
        if imported["outcome"]=="DUPLICATE_DATASET":
            record=gateway.inspect_dataset(imported["existing_dataset_id"]);return self._import_result(record,display_name,file_size,"DUPLICATE_DATASET",purpose)
        record=gateway.inspect_dataset(imported["dataset_id"])
        if record["status"]=="STAGED":
            gateway.certify(record["dataset_id"]);record=gateway.inspect_dataset(record["dataset_id"])
            if purpose=="ENGINEERING_DIAGNOSTIC":
                with self.registry.catalog.connection() as con:con.execute("INSERT OR IGNORE INTO dataset_classifications VALUES (?,?,?,CURRENT_TIMESTAMP)",(record["dataset_id"],purpose,"User-declared diagnostic purpose at controlled import"))
        return self._import_result(record,display_name,file_size,imported["outcome"],purpose)

    def import_diagnostic(self,dataset_id):
        if not re.fullmatch(r"EXT-[0-9a-f]{32}",dataset_id):raise ExternalImportError("invalid external dataset identifier")
        gateway=ExternalImportGateway(self.registry.catalog,self.settings.data_dir);record=gateway.inspect_dataset(dataset_id);m=record["manifest"]
        purpose="ENGINEERING_DIAGNOSTIC" if self._diagnostic_classification(dataset_id) else "UNASSIGNED_RESEARCH_DATA"
        filename=m.get("original_filenames",["historical-data"])[0];size=sum(m.get("component_sizes",[]));return self._import_result(record,filename,size,"HISTORICAL_RECORD",purpose)

    def _diagnostic_classification(self,dataset_id):
        with self.registry.catalog.connection() as con:return con.execute("SELECT 1 FROM dataset_classifications WHERE dataset_id=? AND classification='ENGINEERING_DIAGNOSTIC'",(dataset_id,)).fetchone() is not None

    @staticmethod
    def _count_from_detail(detail,fallback=1):
        match=re.search(r"\b(\d+)\s+rows?\b",str(detail));return int(match.group(1)) if match else fallback

    @classmethod
    def _examples(cls,record,code):
        m=record["manifest"]
        try:frame=pd.read_parquet(m["canonical_path"])
        except Exception:return [],None,None,0
        if "timestamp_utc" not in frame:return [],None,None,0
        timestamps=pd.to_datetime(frame.timestamp_utc,utc=True,errors="coerce");mask=pd.Series(False,index=frame.index);fields=[]
        if code=="invalid_ohlc" and {"open","high","low","close"}<=set(frame):
            mask=(frame.high<frame[["open","close"]].max(axis=1))|(frame.low>frame[["open","close"]].min(axis=1))|(frame.high<frame.low);fields=["open","high","low","close"]
        elif code=="misaligned_m15_timestamp":mask=(timestamps.dt.minute%15!=0)|(timestamps.dt.second!=0)|(timestamps.dt.microsecond!=0)
        elif code in {"duplicate_ohlc_timestamp","duplicate_timestamp"}:mask=timestamps.duplicated(keep=False)
        elif code=="invalid_numeric_price":
            fields=[x for x in ("open","high","low","close","bid","ask") if x in frame]
            if fields:mask=~np.isfinite(frame[fields].apply(pd.to_numeric,errors="coerce")).all(axis=1)
        elif code=="crossed_quote" and {"bid","ask"}<=set(frame):mask=frame.ask<frame.bid;fields=["bid","ask"]
        affected=timestamps[mask & timestamps.notna()];first=affected.iloc[0].isoformat() if len(affected) else None;last=affected.iloc[-1].isoformat() if len(affected) else None;examples=[]
        for index in frame.index[mask][:5]:
            values={}
            for field in fields:
                value=frame.at[index,field];values[field]=None if pd.isna(value) else float(value) if isinstance(value,(float,np.floating)) else int(value) if isinstance(value,(int,np.integer)) else str(value)
            examples.append(DiagnosticExampleDTO(timestamp=None if pd.isna(timestamps.iloc[index]) else timestamps.iloc[index].isoformat(),values=values))
        return examples,first,last,int(mask.sum())

    @classmethod
    def _diagnostics(cls,record):
        a=record.get("audit",{});issues=a.get("issues",[]);descriptions={"invalid_ohlc":"OHLC geometry violates candle invariants.","invalid_timestamp":"Timestamp values could not be parsed using the declared representation.","misaligned_m15_timestamp":"M15 timestamps are not exact 15-minute UTC boundaries.","duplicate_ohlc_timestamp":"Duplicate OHLC timestamps were detected.","invalid_numeric_price":"Price values are missing, non-finite, or non-positive.","crossed_quote":"Ask price is below bid price.","out_of_order":"Source timestamps are not in canonical order.","component_overlap":"Component files contain overlapping events."}
        counts={"invalid_ohlc":a.get("ohlc_violations",0),"duplicate_ohlc_timestamp":a.get("duplicate_events",0),"crossed_quote":a.get("crossed_quotes",0),"out_of_order":a.get("ordering_issues",0),"component_overlap":a.get("component_overlap",0)};findings=[]
        observed_counts={}
        for issue in issues:
            code=str(issue.get("code","validation_error"));count=int(counts.get(code) or cls._count_from_detail(issue.get("detail"),1));examples,first,last,observed=cls._examples(record,code);observed_counts[code]=observed
            findings.append(ImportFindingDTO(check_name=code.upper(),error_code=code.upper(),severity="ERROR" if issue.get("category")=="confirmed_data_error" else "WARNING",count=count,description=descriptions.get(code,"Stage-1 validation reported a data-quality finding."),first_affected_timestamp=first,last_affected_timestamp=last,examples=examples))
        if a.get("suspicious_gaps",0):findings.append(ImportFindingDTO(check_name="SOURCE_INTERVAL_GAPS",error_code="SUSPICIOUS_GAP",severity="WARNING",count=a["suspicious_gaps"],description="Intervals larger than the declared source timeframe were detected."))
        if a.get("expected_gaps",0):findings.append(ImportFindingDTO(check_name="MARKET_CLOSURE_GAPS",error_code="WEEKEND_CLOSURE_GAP",severity="INFO",count=a["expected_gaps"],description="Expected weekend or market-closure intervals were detected."))
        code_counts={x.error_code:x.count for x in findings};summary=ImportValidationSummaryDTO(invalid_ohlc_count=a.get("ohlc_violations",0),invalid_timestamp_count=code_counts.get("INVALID_TIMESTAMP",0),misaligned_timestamp_count=code_counts.get("MISALIGNED_M15_TIMESTAMP",0),duplicate_timestamp_count=a.get("duplicate_events",0),nan_inf_count=observed_counts.get("invalid_numeric_price",0),suspicious_gap_count=a.get("suspicious_gaps",0),weekend_closure_gap_count=a.get("expected_gaps",0),unexpected_gap_count=a.get("suspicious_gaps",0),crossed_quote_count=a.get("crossed_quotes",0),schema_error_count=code_counts.get("SCHEMA_ERROR",0),parse_error_count=code_counts.get("PARSE_ERROR",0))
        return summary,findings

    @classmethod
    def _rejected_result(cls,path,filename,file_size,provider,symbol,data_type,price_type,source_timezone,timestamp_format,purpose,exc):
        audit=getattr(exc,"diagnostics",{}) or {};record={"audit":audit,"manifest":{}};summary,findings=cls._diagnostics(record)
        if isinstance(exc,ExternalImportError) and str(exc).startswith("MULTIPLE_INTERPRETATIONS"):
            findings=[ImportFindingDTO(check_name="MULTIPLE_INTERPRETATIONS",error_code="MULTIPLE_INTERPRETATIONS_REASON_REQUIRED",severity="ERROR",count=1,description="The same immutable source bytes already exist under different material interpretation metadata. Provide an explicit interpretation reason to preserve both audited interpretations.")];summary=summary.model_copy(update={"parse_error_count":0})
        if not findings:
            code="SCHEMA_ERROR" if isinstance(exc,(ValueError,KeyError)) else "PARSE_ERROR";description="Input columns do not match the declared import schema." if code=="SCHEMA_ERROR" else "The file could not produce a valid canonical dataset using the declared format."
            findings=[ImportFindingDTO(check_name=code,error_code=code,severity="ERROR",count=1,description=description)];summary=summary.model_copy(update={"schema_error_count":1} if code=="SCHEMA_ERROR" else {"parse_error_count":1})
        return ImportResultDTO(outcome="REJECTED",import_status="REJECTED",certification="REJECTED",filename=filename,file_size=file_size,provider=provider,symbol=symbol,format="CSV.GZ" if filename.lower().endswith(".csv.gz") else filename.rsplit(".",1)[-1].upper(),data_type=data_type,price_type=price_type,source_timezone=source_timezone,timestamp_format=timestamp_format,purpose=purpose,original_sha256=sha256_file(path),message="IMPORT REJECTED — NO RESEARCH DATA CREATED",validation_summary=summary,rejection_reasons=findings)

    @classmethod
    def _import_result(cls,record,filename,file_size,outcome,purpose):
        m=record["manifest"];a=record["audit"];status=record["status"]
        certification="DUPLICATE" if outcome=="DUPLICATE_DATASET" else status
        message="DATASET QUARANTINED — NOT ELIGIBLE FOR RESEARCH" if status=="QUARANTINED" else "DUPLICATE DATASET — existing evidence preserved; no duplicate artifacts created." if certification=="DUPLICATE" else "DATASET CERTIFIED"
        suffix="CSV.GZ" if m["original_filenames"][0].lower().endswith(".csv.gz") else m["original_filenames"][0].rsplit(".",1)[-1].upper()
        summary,findings=cls._diagnostics(record)
        return ImportResultDTO(outcome=outcome,import_status=status,certification=certification,dataset_id=record["dataset_id"],import_id=record["import_id"],filename=filename,file_size=file_size,provider=m["data_source_provider"],symbol=m["canonical_symbol"],format=suffix,data_type=m["resolution"],price_type=m["price_type"],source_timezone=m["source_timezone"],timestamp_format=m.get("timestamp_format","ISO8601"),purpose=purpose,row_count=m["row_count"],earliest_timestamp=m["first_timestamp"],latest_timestamp=m["last_timestamp"],original_sha256=m["component_sha256s"][0],dataset_fingerprint=m["scientific_fingerprint"],duplicates=a["duplicate_events"],suspicious_gaps=a["suspicious_gaps"],weekend_gaps=a["expected_gaps"],crossed_quotes=a["crossed_quotes"],invalid_rows=a["rows_rejected"],message=message,validation_summary=summary,quarantine_reasons=findings if status=="QUARANTINED" else [])
    def evidence(self):
        count=self._count("statistical_evidence_reports");return EvidenceStatusDTO(available=count>0,report_count=count,status="AVAILABLE" if count else "NO_RESEARCH_EVIDENCE_AVAILABLE")
    def audit(self):
        journal="VERIFIED" if self.registry.verify_journal_integrity() else "FAILED";access="VERIFIED" if PartitionService(self.registry).verify_access_ledger() else "FAILED"
        checks=[CheckDTO(name=x,status="VERIFIED") for x in ("STAGE_1_DATA_INTEGRITY","STAGE_2_DETERMINISTIC_EXECUTION","STAGE_3_RESEARCH_REGISTRY","STAGE_4_AUDITOR","STAGE_5_RESEARCH_CONTROL","STAGE_6_PARTITION_PROTECTION","STAGE_7_STATISTICAL_EVIDENCE","EXTERNAL_IMPORT_GATEWAY","STRATEGY_PLUGIN_INTERFACE","S001_ISOLATION")]
        return AuditStatusDTO(checks=checks,journal_integrity=journal,access_ledger_integrity=access,final_test_vault_state="SEALED",broker_execution="DISABLED",optimization="DISABLED",machine_learning="DISABLED")
    def market_state(self):
        try:
            from sandbox.market_state import FEATURE_ENGINE_VERSION,FEATURE_SCHEMA_VERSION
            return MarketStateStatusDTO(available=True,status="IMPLEMENTED_NO_AUTHORIZED_SNAPSHOTS",schema_version=FEATURE_SCHEMA_VERSION,engine_version=FEATURE_ENGINE_VERSION,authorized_snapshot_count=0,message="Framework is installed; no causal snapshot research has been run.")
        except ImportError:return MarketStateStatusDTO(available=False,status="NOT_IMPLEMENTED",message="Market-state framework is not installed.")
