from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel,ConfigDict
from sandbox import ACCESS_POLICY_VERSION,PARTITION_POLICY_VERSION,PARTITION_SCHEMA_VERSION

class FrozenModel(BaseModel):model_config=ConfigDict(frozen=True,extra="forbid")
class PartitionRole(StrEnum):DISCOVERY="DISCOVERY";VALIDATION="VALIDATION";FINAL_TEST="FINAL_TEST";DIAGNOSTIC="DIAGNOSTIC"
class AccessContext(StrEnum):DISCOVERY_CONTEXT="DISCOVERY_CONTEXT";VALIDATION_CONTEXT="VALIDATION_CONTEXT";FINAL_TEST_CONTEXT="FINAL_TEST_CONTEXT";SYSTEM_INTEGRITY_CONTEXT="SYSTEM_INTEGRITY_CONTEXT"
class AccessOperation(StrEnum):METADATA_READ="METADATA_READ";RAW_CANDLE_READ="RAW_CANDLE_READ";AGGREGATE_READ="AGGREGATE_READ";BACKTEST_EXECUTION="BACKTEST_EXECUTION";INTEGRITY_CHECK="INTEGRITY_CHECK";CHECKSUM_VERIFY="CHECKSUM_VERIFY";EXPORT="EXPORT";FEATURE_ANALYSIS="FEATURE_ANALYSIS";WINNER_LOSER_ANALYSIS="WINNER_LOSER_ANALYSIS";WARMUP_READ="WARMUP_READ";EVALUATION_READ="EVALUATION_READ"
class AccessDecision(StrEnum):ALLOWED="ALLOWED";DENIED="DENIED"
class ContaminationType(StrEnum):DISCOVERY_EXPOSURE="DISCOVERY_EXPOSURE";VALIDATION_EXPOSURE="VALIDATION_EXPOSURE";FINAL_TEST_EXPOSURE="FINAL_TEST_EXPOSURE";MANUAL_INSPECTION="MANUAL_INSPECTION";RESULT_DRIVEN_REUSE="RESULT_DRIVEN_REUSE";UNAUTHORIZED_ACCESS_ATTEMPT="UNAUTHORIZED_ACCESS_ATTEMPT";BUG_RELATED_ACCESS="BUG_RELATED_ACCESS";SYSTEM_INTEGRITY_ACCESS="SYSTEM_INTEGRITY_ACCESS"

class PartitionManifest(FrozenModel):
    partition_id:str;source_dataset_id:str;program_id:str;role:PartitionRole;symbol:str;timeframe:str;start_timestamp:datetime;end_timestamp:datetime;row_count:int;partition_checksum:str;source_dataset_checksum:str;created_at:datetime;partition_schema_version:str=PARTITION_SCHEMA_VERSION;partition_policy_version:str=PARTITION_POLICY_VERSION;partition_fingerprint:str;manifest_fingerprint:str;path:str;diagnostic_overlap_declared:bool=False
class AccessRecord(FrozenModel):
    access_id:str;timestamp:datetime;actor:str;program_id:str;candidate_id:str|None=None;experiment_id:str|None=None;partition_id:str;requested_operation:AccessOperation;access_context:AccessContext;decision:AccessDecision;reason:str;rows_returned:int;access_policy_version:str=ACCESS_POLICY_VERSION;previous_entry_hash:str;entry_hash:str
class ContaminationRecord(FrozenModel):
    contamination_id:str;program_id:str;candidate_id:str|None;candidate_lineage_id:str|None;partition_id:str;contamination_type:ContaminationType;source_event:str;first_exposed_at:datetime;reason:str;severity:str;created_at:datetime
class PartitionBinding(FrozenModel):
    binding_id:str;binding_type:str;candidate_id:str;candidate_fingerprint:str;partition_id:str;evaluation_spec_fingerprint:str;execution_config_fingerprint:str;created_at:datetime;status:str="BOUND"
class FinalAuthorization(FrozenModel):
    authorization_id:str;candidate_id:str;candidate_fingerprint:str;partition_id:str;validation_result_id:str;evaluation_spec_fingerprint:str;execution_config_fingerprint:str;authorized_by:str;authorized_at:datetime;policy_version:str=ACCESS_POLICY_VERSION
class WarmupWindow(FrozenModel):
    partition_id:str;candidate_id:str;start_timestamp:datetime;end_timestamp:datetime;evaluation_start:datetime;rows:int;trading_allowed:bool=False;included_in_evaluation:bool=False
