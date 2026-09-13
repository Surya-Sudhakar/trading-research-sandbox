from __future__ import annotations
from enum import StrEnum
from typing import Any
from pydantic import BaseModel,ConfigDict,Field

class DTO(BaseModel):model_config=ConfigDict(extra="forbid")
class BrokerStatus(DTO):connected:bool;company:str|None=None;server:str|None=None;read_only:bool=True
class SystemStatusDTO(DTO):backend_online:bool=True;source:str="LIVE_BACKEND_STATE";broker:BrokerStatus;research_mode:str;strategy_count:int;dataset_count:int;partition_count:int;verified_test_count:int;integrity_status:str
class ParameterDTO(DTO):name:str;type:str;material:bool;required:bool;default:Any=None
class StrategySummaryDTO(DTO):strategy_id:str;name:str;version:str;description:str;required_timeframes:list[str];capabilities:list[str];implementation_status:str;research_status:str
class StrategyDetailDTO(StrategySummaryDTO):parameters:list[ParameterDTO];code_fingerprint:str;parameter_fingerprint:str
class ResearchStatusDTO(DTO):program_count:int;hypothesis_count:int;experiment_count:int;candidate_count:int;partition_count:int;discovery_status:str;validation_status:str;final_test_status:str
class DatasetDTO(DTO):dataset_id:str;broker:str;server:str;symbol:str;timeframe:str;start:str;end:str;row_count:int;checksum:str;schema_version:str
class ImportedDatasetDTO(DTO):
    dataset_id:str;provider:str;symbol:str;data_type:str;price_type:str;date_range:str;row_count:int;certification:str;purpose:str;fingerprint:str
class DataStatusDTO(DTO):broker:BrokerStatus;datasets:list[DatasetDTO];imported_datasets:list[ImportedDatasetDTO]=Field(default_factory=list);dataset_count:int;external_import_gateway_available:bool;external_production_dataset_count:int;research_partition_count:int
class EvidenceStatusDTO(DTO):available:bool;report_count:int;status:str
class CheckDTO(DTO):name:str;status:str
class AuditStatusDTO(DTO):checks:list[CheckDTO];journal_integrity:str;access_ledger_integrity:str;final_test_vault_state:str;broker_execution:str;optimization:str;machine_learning:str
class MarketStateStatusDTO(DTO):available:bool;status:str;schema_version:str|None=None;engine_version:str|None=None;authorized_snapshot_count:int=0;message:str
class ErrorDTO(DTO):error:str;message:str
class ImportProvider(StrEnum):DUKASCOPY="DUKASCOPY";TRUEFX="TRUEFX";HISTDATA="HISTDATA";OTHER="OTHER"
class ImportPurpose(StrEnum):ENGINEERING_DIAGNOSTIC="ENGINEERING_DIAGNOSTIC";UNASSIGNED_RESEARCH_DATA="UNASSIGNED_RESEARCH_DATA"
class ImportTimestampFormat(StrEnum):ISO8601="ISO8601";UNIX_MILLISECONDS="UNIX_MILLISECONDS"
class DiagnosticExampleDTO(DTO):timestamp:str|None=None;values:dict[str,str|int|float|None]=Field(default_factory=dict)
class ImportFindingDTO(DTO):
    check_name:str;error_code:str;severity:str;count:int;description:str;first_affected_timestamp:str|None=None;last_affected_timestamp:str|None=None;examples:list[DiagnosticExampleDTO]=Field(default_factory=list,max_length=5)
class ImportValidationSummaryDTO(DTO):
    invalid_ohlc_count:int=0;invalid_timestamp_count:int=0;misaligned_timestamp_count:int=0;duplicate_timestamp_count:int=0;nan_inf_count:int=0;suspicious_gap_count:int=0;weekend_closure_gap_count:int=0;unexpected_gap_count:int=0;crossed_quote_count:int=0;schema_error_count:int=0;parse_error_count:int=0
class ImportResultDTO(DTO):
    outcome:str;import_status:str;certification:str;dataset_id:str|None=None;import_id:str|None=None
    filename:str;file_size:int;provider:str;symbol:str;format:str;data_type:str;price_type:str;source_timezone:str;timestamp_format:str;purpose:str
    row_count:int=0;earliest_timestamp:str|None=None;latest_timestamp:str|None=None;original_sha256:str|None=None;dataset_fingerprint:str|None=None
    duplicates:int=0;suspicious_gaps:int=0;weekend_gaps:int=0;crossed_quotes:int=0;invalid_rows:int=0;provenance:str="DECLARED_PROVENANCE";message:str
    validation_summary:ImportValidationSummaryDTO=Field(default_factory=ImportValidationSummaryDTO);quarantine_reasons:list[ImportFindingDTO]=Field(default_factory=list);rejection_reasons:list[ImportFindingDTO]=Field(default_factory=list)
