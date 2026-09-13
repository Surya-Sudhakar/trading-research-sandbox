from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel,ConfigDict,Field


class FrozenModel(BaseModel):model_config=ConfigDict(frozen=True,extra="forbid")
class ResearchMode(StrEnum):DISCOVERY="DISCOVERY";VALIDATION="VALIDATION";FINAL_TEST="FINAL_TEST"
class EvidenceOrigin(StrEnum):THEORY="THEORY";PRIOR_RESEARCH="PRIOR_RESEARCH";DISCOVERY_DATA="DISCOVERY_DATA";VALIDATION_DATA="VALIDATION_DATA";FINAL_TEST_DATA="FINAL_TEST_DATA";BUG_INVESTIGATION="BUG_INVESTIGATION";DATA_QUALITY_INVESTIGATION="DATA_QUALITY_INVESTIGATION";EXECUTION_INVESTIGATION="EXECUTION_INVESTIGATION";EXTERNAL_RESEARCH="EXTERNAL_RESEARCH";UNKNOWN="UNKNOWN"
class DecisionAction(StrEnum):CONTINUE_DISCOVERY="CONTINUE_DISCOVERY";CREATE_CHILD_QUESTION="CREATE_CHILD_QUESTION";CREATE_NEW_HYPOTHESIS="CREATE_NEW_HYPOTHESIS";FREEZE_CANDIDATE="FREEZE_CANDIDATE";ADVANCE_TO_VALIDATION="ADVANCE_TO_VALIDATION";REJECT_CANDIDATE="REJECT_CANDIDATE";CLOSE_HYPOTHESIS="CLOSE_HYPOTHESIS";CLOSE_QUESTION="CLOSE_QUESTION";DEFER="DEFER";COLLECT_MORE_EVIDENCE="COLLECT_MORE_EVIDENCE";RETURN_TO_DISCOVERY="RETURN_TO_DISCOVERY"
class DiversionClassification(StrEnum):ON_PATH="ON_PATH";RELATED_BRANCH="RELATED_BRANCH";EXPLORATORY_BRANCH="EXPLORATORY_BRANCH";POSSIBLE_DIVERSION="POSSIBLE_DIVERSION";UNRELATED="UNRELATED"
class CandidateStatus(StrEnum):EXPLORATORY="EXPLORATORY";PROMISING="PROMISING";FROZEN="FROZEN";VALIDATING="VALIDATING";VALIDATED="VALIDATED";REJECTED="REJECTED";RETIRED="RETIRED"
class StrategyFamilyStatus(StrEnum):PROPOSED="PROPOSED";ACTIVE="ACTIVE";EXHAUSTED="EXHAUSTED";VALIDATING="VALIDATING";VALIDATED="VALIDATED";REJECTED="REJECTED";RETIRED="RETIRED"

class DecisionNode(FrozenModel):
    decision_id:str;question_id:str;description:str;possible_outcomes:dict[str,DecisionAction];created_at:datetime;research_mode:ResearchMode;stopping_conditions:dict[str,Any]={};recorded_outcome:str|None=None;recorded_action:DecisionAction|None=None
class CandidateStrategy(FrozenModel):
    candidate_id:str;candidate_base_id:str;version:int;program_id:str;originating_question_id:str;originating_hypothesis_id:str;experiment_family_id:str;strategy_family_id:str;strategy_spec:dict[str,Any];execution_config:dict[str,Any];evaluation_spec:dict[str,Any];symbols:tuple[str,...];timeframes:tuple[str,...];session_rules:dict[str,Any];code_version:dict[str,Any];discovery_dataset_ids:tuple[str,...];validation_dataset_ids:tuple[str,...]=();discovery_history_summary:dict[str,Any];created_at:datetime;status:CandidateStatus;candidate_fingerprint:str;parent_candidate_id:str|None=None;validation_status:str="NOT_STARTED"
class StrategyFamily(FrozenModel):
    strategy_family_id:str;program_id:str;name:str;rationale:str;created_at:datetime;status:StrategyFamilyStatus
