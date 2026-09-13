from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sandbox import RESEARCH_SCHEMA_VERSION


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProgramStatus(StrEnum):
    ACTIVE="ACTIVE"; PAUSED="PAUSED"; COMPLETED="COMPLETED"; ABANDONED="ABANDONED"


class QuestionStatus(StrEnum):
    OPEN="OPEN"; INVESTIGATING="INVESTIGATING"; ANSWERED="ANSWERED"; REJECTED="REJECTED"; DEFERRED="DEFERRED"; CLOSED="CLOSED"


class HypothesisStatus(StrEnum):
    PROPOSED="PROPOSED"; APPROVED="APPROVED"; TESTING="TESTING"; SUPPORTED="SUPPORTED"; REJECTED="REJECTED"; INCONCLUSIVE="INCONCLUSIVE"; RETIRED="RETIRED"


class ExperimentStatus(StrEnum):
    DRAFT="DRAFT"; APPROVED="APPROVED"; RUNNING="RUNNING"; COMPLETED="COMPLETED"; REVIEWED="REVIEWED"; CLOSED="CLOSED"; FAILED="FAILED"; CANCELLED="CANCELLED"


class VerdictValue(StrEnum):
    PASS="PASS"; FAIL="FAIL"; INCONCLUSIVE="INCONCLUSIVE"; INVALID="INVALID"


class ResearchProgram(FrozenModel):
    program_id: str
    name: str = Field(min_length=1)
    description: str
    primary_objective: str = Field(min_length=1)
    created_at: datetime
    status: ProgramStatus = ProgramStatus.ACTIVE
    schema_version: str = RESEARCH_SCHEMA_VERSION


class ResearchQuestion(FrozenModel):
    question_id: str
    program_id: str
    parent_question_id: str | None = None
    question_text: str = Field(min_length=1)
    rationale: str
    created_at: datetime
    status: QuestionStatus = QuestionStatus.OPEN
    revision: int = 1
    schema_version: str = RESEARCH_SCHEMA_VERSION
    purpose: str | None = None
    relevance_to_parent: str | None = None
    evidence_origins: tuple[str, ...] = ()
    research_mode: str | None = None
    control_enabled: bool = False


class Hypothesis(FrozenModel):
    hypothesis_id: str
    program_id: str
    question_id: str
    hypothesis_text: str = Field(min_length=1)
    rationale: str
    expected_observation: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    created_at: datetime
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_hypothesis_id: str | None = None
    revision: int = 1
    schema_version: str = RESEARCH_SCHEMA_VERSION
    research_mode: str | None = None
    evidence_origins: tuple[str, ...] = ()


class ExperimentSpec(FrozenModel):
    experiment_id: str
    program_id: str
    question_id: str
    hypothesis_id: str
    title: str = Field(min_length=1)
    description: str
    dataset_ids: tuple[str, ...] = Field(min_length=1)
    strategy_spec: dict[str, Any]
    execution_config: dict[str, Any]
    evaluation_spec: dict[str, Any]
    code_version: dict[str, Any]
    created_at: datetime
    approved_at: datetime | None = None
    experiment_fingerprint: str
    schema_version: str = RESEARCH_SCHEMA_VERSION
    status: ExperimentStatus = ExperimentStatus.DRAFT
    parent_experiment_id: str | None = None
    change_reason: str | None = None
    experiment_family_id: str | None = None
    change_reason_type: str | None = None
    change_reason_text: str | None = None
    fields_changed_from_parent: tuple[str, ...] = ()
    research_mode: str | None = None
    motivation_spec: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(FrozenModel):
    result_id: str
    experiment_id: str
    run_id: str
    run_fingerprint: str
    summary_metrics: dict[str, Any]
    ledger_checksum: str
    created_at: datetime
    schema_version: str = RESEARCH_SCHEMA_VERSION


class ExperimentVerdict(FrozenModel):
    verdict_id: str
    experiment_id: str
    result_id: str
    verdict: VerdictValue
    rationale: str = Field(min_length=1)
    decided_at: datetime
    decided_by: str = Field(min_length=1)
    schema_version: str = RESEARCH_SCHEMA_VERSION


class JournalEntry(FrozenModel):
    journal_entry_id: str
    timestamp: datetime
    event_type: str
    program_id: str
    question_id: str | None = None
    hypothesis_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    message: str
    metadata: dict[str, Any]
    previous_entry_hash: str
    entry_hash: str


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
