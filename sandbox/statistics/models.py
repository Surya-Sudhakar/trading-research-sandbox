from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel,ConfigDict
from sandbox import STATISTICAL_EVIDENCE_ENGINE_VERSION

class ResearchMode(StrEnum):
    DISCOVERY="DISCOVERY";VALIDATION="VALIDATION";FINAL_TEST="FINAL_TEST"

class StatisticalEvidenceReport(BaseModel):
    model_config=ConfigDict(frozen=True,extra="forbid")
    report_id:str;experiment_id:str|None=None;candidate_id:str|None=None;run_id:str;partition_id:str|None=None;research_mode:ResearchMode
    ledger_checksum:str;evidence_engine_version:str=STATISTICAL_EVIDENCE_ENGINE_VERSION;statistical_config_fingerprint:str;created_at:datetime
    sample_profile:dict[str,Any];performance_profile:dict[str,Any];uncertainty_profile:dict[str,Any];temporal_profile:dict[str,Any];symbol_profile:dict[str,Any];direction_profile:dict[str,Any];concentration_profile:dict[str,Any];dependence_profile:dict[str,Any];drawdown_profile:dict[str,Any];sensitivity_profile:dict[str,Any]|None=None;resampling_profile:dict[str,Any]|None=None;warnings:tuple[str,...];evidence_summary:str;report_fingerprint:str
