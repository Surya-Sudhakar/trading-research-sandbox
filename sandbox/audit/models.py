from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config=ConfigDict(frozen=True,extra="forbid")


class AuditPhase(StrEnum): PRE_RUN="PRE_RUN"; POST_RUN="POST_RUN"
class AuditClassification(StrEnum): SAFE="SAFE"; WARNING="WARNING"; HIGH_RISK="HIGH_RISK"; BLOCKED="BLOCKED"
class FindingSeverity(StrEnum): INFO="INFO"; WARNING="WARNING"; HIGH="HIGH"; CRITICAL="CRITICAL"
class FindingCategory(StrEnum):
    LOOKAHEAD="LOOKAHEAD"; DATA_LEAKAGE="DATA_LEAKAGE"; POST_HOC_CHANGE="POST_HOC_CHANGE"; DEGREES_OF_FREEDOM="DEGREES_OF_FREEDOM"; PARAMETER_SEARCH="PARAMETER_SEARCH"; HYPOTHESIS_PROLIFERATION="HYPOTHESIS_PROLIFERATION"; DUPLICATION="DUPLICATION"; ANCESTRY="ANCESTRY"; COMPLEXITY="COMPLEXITY"; FALSIFIABILITY="FALSIFIABILITY"; SAMPLE_USAGE="SAMPLE_USAGE"; EXECUTION="EXECUTION"; DATA_INTEGRITY="DATA_INTEGRITY"; REPRODUCIBILITY="REPRODUCIBILITY"
class DatasetUsageType(StrEnum): UNSEEN="UNSEEN"; DISCOVERY="DISCOVERY"; VALIDATION="VALIDATION"; FINAL_TEST="FINAL_TEST"; DIAGNOSTIC="DIAGNOSTIC"


class AuditFinding(FrozenModel):
    finding_id:str; audit_id:str; experiment_id:str; category:FindingCategory; severity:FindingSeverity; code:str; title:str; explanation:str; evidence:dict[str,Any]; remediation:str; created_at:datetime; auditor_version:str


class ResearchAudit(FrozenModel):
    audit_id:str; experiment_id:str; experiment_fingerprint:str; phase:AuditPhase; auditor_version:str; config_fingerprint:str; audited_at:datetime; overall_classification:AuditClassification; findings:tuple[AuditFinding,...]; risk_score:int; hard_block_reasons:tuple[str,...]; warning_count:int; high_risk_count:int; critical_count:int


class AuditOverride(FrozenModel):
    override_id:str; audit_id:str; experiment_id:str; reason:str; approved_by:str; approved_at:datetime


class DegreesOfFreedomProfile(FrozenModel):
    number_of_strategy_parameters:int=0; number_of_threshold_parameters:int=0; number_of_boolean_filters:int=0; number_of_optional_conditions:int=0; number_of_time_windows:int=0; number_of_entry_variants:int=0; number_of_exit_variants:int=0; number_of_symbols_selectable:int=0; number_of_sessions_selectable:int=0; number_of_prior_experiments_in_family:int=0; number_of_result_driven_descendants:int=0; number_of_rejected_variants:int=0
