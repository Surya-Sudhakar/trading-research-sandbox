from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sandbox import EXECUTION_ENGINE_VERSION, RESEARCH_SCHEMA_VERSION
from sandbox.catalog import Catalog
from sandbox.execution.models import ExecutionConfig
from sandbox.provenance import sha256_file
from sandbox.research.canonical import canonical_json, experiment_fingerprint, sha256_canonical
from sandbox.research.errors import DatasetVerificationError, DuplicateExperimentError, ImmutableRecordError, InvalidTransitionError, JournalIntegrityError, PreRegistrationError, ResearchError
from sandbox.research.models import ExperimentResult, ExperimentSpec, ExperimentStatus, ExperimentVerdict, Hypothesis, HypothesisStatus, JournalEntry, ProgramStatus, QuestionStatus, ResearchProgram, ResearchQuestion, VerdictValue, now_utc


RESEARCH_DDL = """
CREATE TABLE IF NOT EXISTS id_sequences (name TEXT PRIMARY KEY, next_value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS research_programs (program_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS program_revisions (revision_id TEXT PRIMARY KEY, program_id TEXT NOT NULL, revision INTEGER NOT NULL, record_json TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(program_id, revision));
CREATE TABLE IF NOT EXISTS research_questions (question_id TEXT PRIMARY KEY, program_id TEXT NOT NULL, parent_question_id TEXT, record_json TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hypotheses (hypothesis_id TEXT PRIMARY KEY, program_id TEXT NOT NULL, question_id TEXT NOT NULL, parent_hypothesis_id TEXT, record_json TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (internal_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL UNIQUE, program_id TEXT NOT NULL, question_id TEXT NOT NULL, hypothesis_id TEXT NOT NULL, parent_experiment_id TEXT, change_reason TEXT, fingerprint TEXT NOT NULL, record_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, approved_at TEXT);
CREATE INDEX IF NOT EXISTS ix_experiments_fingerprint ON experiments(fingerprint);
CREATE TABLE IF NOT EXISTS experiment_runs (link_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, run_id TEXT NOT NULL, run_fingerprint TEXT NOT NULL, experiment_fingerprint TEXT NOT NULL, run_kind TEXT NOT NULL, original_run_id TEXT, created_at TEXT NOT NULL, UNIQUE(experiment_id, run_id));
CREATE TABLE IF NOT EXISTS experiment_results (result_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE, record_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_verdicts (verdict_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, result_id TEXT NOT NULL, record_json TEXT NOT NULL, decided_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS research_journal (sequence INTEGER PRIMARY KEY AUTOINCREMENT, journal_entry_id TEXT NOT NULL UNIQUE, timestamp TEXT NOT NULL, event_type TEXT NOT NULL, program_id TEXT NOT NULL, question_id TEXT, hypothesis_id TEXT, experiment_id TEXT, run_id TEXT, message TEXT NOT NULL, metadata_json TEXT NOT NULL, previous_entry_hash TEXT NOT NULL, entry_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS research_journal_head (singleton INTEGER PRIMARY KEY CHECK(singleton=1),entry_count INTEGER NOT NULL,head_hash TEXT NOT NULL);
"""

TRANSITIONS = {
    ExperimentStatus.DRAFT: {ExperimentStatus.APPROVED, ExperimentStatus.CANCELLED},
    ExperimentStatus.APPROVED: {ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED},
    ExperimentStatus.RUNNING: {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED},
    ExperimentStatus.COMPLETED: {ExperimentStatus.REVIEWED},
    ExperimentStatus.REVIEWED: {ExperimentStatus.CLOSED},
    ExperimentStatus.FAILED: {ExperimentStatus.REVIEWED},
    ExperimentStatus.CANCELLED: set(), ExperimentStatus.CLOSED: set(),
}


class ResearchRegistry:
    def __init__(self, catalog_path: Path, project_root: Path | None = None, results_root: Path | None = None):
        self.catalog = Catalog(catalog_path)
        self.project_root = (project_root or Path.cwd()).resolve()
        self.results_root = results_root or self.project_root / "results"
        self.initialize()

    def initialize(self) -> None:
        self.catalog.initialize()
        with self.catalog.connection() as con:
            con.executescript(RESEARCH_DDL)
            if not con.execute("SELECT 1 FROM research_journal_head WHERE singleton=1").fetchone():
                row=con.execute("SELECT count(*),coalesce((SELECT entry_hash FROM research_journal ORDER BY sequence DESC LIMIT 1),?) FROM research_journal",("0"*64,)).fetchone();con.execute("INSERT INTO research_journal_head VALUES (1,?,?)",(row[0],row[1]))

    def _get_json(self, table: str, key: str, value: str):
        with self.catalog.connection() as con:
            row = con.execute(f"SELECT record_json FROM {table} WHERE {key}=?", (value,)).fetchone()
        return json.loads(row[0]) if row else None

    def _allocate_experiment_id(self) -> str:
        with self.catalog.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("INSERT OR IGNORE INTO id_sequences(name,next_value) VALUES ('experiment',1)")
            number = con.execute("SELECT next_value FROM id_sequences WHERE name='experiment'").fetchone()[0]
            con.execute("UPDATE id_sequences SET next_value=next_value+1 WHERE name='experiment'")
        return f"EXP-{number:06d}"

    def capture_code_version(self) -> dict:
        digest = hashlib.sha256()
        files = sorted((self.project_root / "sandbox").rglob("*.py"))
        for path in files:
            digest.update(path.relative_to(self.project_root).as_posix().encode())
            digest.update(path.read_bytes())
        return {"method": "SOURCE_SHA256", "revision": digest.hexdigest(), "working_tree": "UNKNOWN"}

    def append_journal(self, event_type: str, program_id: str, message: str, *, question_id=None, hypothesis_id=None, experiment_id=None, run_id=None, metadata=None) -> JournalEntry:
        timestamp = now_utc(); entry_id = str(uuid4()); metadata = metadata or {}
        with self.catalog.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            previous = con.execute("SELECT entry_hash FROM research_journal ORDER BY sequence DESC LIMIT 1").fetchone()
            previous_hash = previous[0] if previous else "0" * 64
            content = {"journal_entry_id": entry_id, "timestamp": timestamp, "event_type": event_type, "program_id": program_id, "question_id": question_id, "hypothesis_id": hypothesis_id, "experiment_id": experiment_id, "run_id": run_id, "message": message, "metadata": metadata, "previous_entry_hash": previous_hash}
            entry_hash = sha256_canonical(content)
            con.execute("INSERT INTO research_journal(journal_entry_id,timestamp,event_type,program_id,question_id,hypothesis_id,experiment_id,run_id,message,metadata_json,previous_entry_hash,entry_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (entry_id,timestamp.isoformat(),event_type,program_id,question_id,hypothesis_id,experiment_id,run_id,message,canonical_json(metadata),previous_hash,entry_hash))
            con.execute("UPDATE research_journal_head SET entry_count=entry_count+1,head_hash=? WHERE singleton=1",(entry_hash,))
        return JournalEntry(**content, entry_hash=entry_hash)

    def verify_journal_integrity(self) -> bool:
        previous = "0" * 64
        with self.catalog.connection() as con:
            rows = con.execute("SELECT * FROM research_journal ORDER BY sequence").fetchall()
        for row in rows:
            content = {"journal_entry_id": row["journal_entry_id"], "timestamp": datetime.fromisoformat(row["timestamp"]), "event_type": row["event_type"], "program_id": row["program_id"], "question_id": row["question_id"], "hypothesis_id": row["hypothesis_id"], "experiment_id": row["experiment_id"], "run_id": row["run_id"], "message": row["message"], "metadata": json.loads(row["metadata_json"]), "previous_entry_hash": row["previous_entry_hash"]}
            if row["previous_entry_hash"] != previous or sha256_canonical(content) != row["entry_hash"]:
                raise JournalIntegrityError(f"journal integrity failed at {row['journal_entry_id']}")
            previous = row["entry_hash"]
        with self.catalog.connection() as con:head=con.execute("SELECT entry_count,head_hash FROM research_journal_head WHERE singleton=1").fetchone()
        if not head or head[0]!=len(rows) or head[1]!=previous:raise JournalIntegrityError("journal head/count integrity failure")
        return True

    def create_program(self, program_id: str, name: str, description: str, primary_objective: str) -> ResearchProgram:
        record = ResearchProgram(program_id=program_id,name=name,description=description,primary_objective=primary_objective,created_at=now_utc())
        with self.catalog.connection() as con:
            con.execute("INSERT INTO research_programs VALUES (?,?,?,?)", (program_id,record.model_dump_json(),record.status.value,record.created_at.isoformat()))
            con.execute("INSERT INTO program_revisions VALUES (?,?,?,?,?)", (str(uuid4()),program_id,1,record.model_dump_json(),record.created_at.isoformat()))
        self.append_journal("PROGRAM_CREATED",program_id,f"Program {program_id} created")
        return record

    def get_program(self, program_id: str) -> ResearchProgram | None:
        value=self._get_json("research_programs","program_id",program_id); return ResearchProgram.model_validate(value) if value else None

    def create_question(self, question_id: str, program_id: str, question_text: str, rationale: str, parent_question_id: str | None=None, purpose: str | None=None, relevance_to_parent: str | None=None, evidence_origins=(), research_mode: str | None=None, control_enabled: bool=False) -> ResearchQuestion:
        if not self.get_program(program_id): raise ResearchError(f"program not found: {program_id}")
        if parent_question_id and not self._get_json("research_questions","question_id",parent_question_id): raise ResearchError("parent question not found")
        if parent_question_id and control_enabled:
            missing=[name for name,value in (("purpose",purpose),("relevance_to_parent",relevance_to_parent),("evidence_origins",evidence_origins),("research_mode",research_mode)) if not value]
            if missing: raise ResearchError("QUESTION_CONTROL_REJECTED: missing "+", ".join(missing))
        record=ResearchQuestion(question_id=question_id,program_id=program_id,parent_question_id=parent_question_id,question_text=question_text,rationale=rationale,created_at=now_utc(),purpose=purpose,relevance_to_parent=relevance_to_parent,evidence_origins=tuple(evidence_origins),research_mode=research_mode,control_enabled=control_enabled)
        with self.catalog.connection() as con: con.execute("INSERT INTO research_questions VALUES (?,?,?,?,?,?,?)",(question_id,program_id,parent_question_id,record.model_dump_json(),record.status.value,1,record.created_at.isoformat()))
        self.append_journal("QUESTION_CREATED",program_id,f"Question {question_id} created",question_id=question_id)
        return record

    def get_question(self, question_id: str) -> ResearchQuestion | None:
        value=self._get_json("research_questions","question_id",question_id); return ResearchQuestion.model_validate(value) if value else None

    def create_hypothesis(self, hypothesis_id: str, program_id: str, question_id: str, hypothesis_text: str, rationale: str, expected_observation: str, falsification_condition: str, parent_hypothesis_id: str | None=None, research_mode: str | None=None, evidence_origins=()) -> Hypothesis:
        question=self.get_question(question_id)
        if not question or question.program_id != program_id: raise ResearchError("question not found in program")
        if question.control_enabled and (not research_mode or not evidence_origins):raise ResearchError("controlled hypothesis requires explicit research_mode and evidence_origins")
        record=Hypothesis(hypothesis_id=hypothesis_id,program_id=program_id,question_id=question_id,hypothesis_text=hypothesis_text,rationale=rationale,expected_observation=expected_observation,falsification_condition=falsification_condition,created_at=now_utc(),parent_hypothesis_id=parent_hypothesis_id,research_mode=research_mode,evidence_origins=tuple(evidence_origins))
        with self.catalog.connection() as con: con.execute("INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?)",(hypothesis_id,program_id,question_id,parent_hypothesis_id,record.model_dump_json(),record.status.value,1,record.created_at.isoformat()))
        self.append_journal("HYPOTHESIS_CREATED",program_id,f"Hypothesis {hypothesis_id} created",question_id=question_id,hypothesis_id=hypothesis_id)
        return record

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        value=self._get_json("hypotheses","hypothesis_id",hypothesis_id); return Hypothesis.model_validate(value) if value else None

    def set_hypothesis_status(self, hypothesis_id: str, status: HypothesisStatus) -> Hypothesis:
        record=self.get_hypothesis(hypothesis_id)
        if not record: raise ResearchError("hypothesis not found")
        allowed={HypothesisStatus.PROPOSED:{HypothesisStatus.APPROVED,HypothesisStatus.RETIRED},HypothesisStatus.APPROVED:{HypothesisStatus.TESTING,HypothesisStatus.RETIRED},HypothesisStatus.TESTING:{HypothesisStatus.SUPPORTED,HypothesisStatus.REJECTED,HypothesisStatus.INCONCLUSIVE},HypothesisStatus.SUPPORTED:{HypothesisStatus.RETIRED},HypothesisStatus.REJECTED:{HypothesisStatus.RETIRED},HypothesisStatus.INCONCLUSIVE:{HypothesisStatus.RETIRED},HypothesisStatus.RETIRED:set()}
        if status not in allowed[record.status]: raise InvalidTransitionError(f"invalid hypothesis transition {record.status} -> {status}")
        updated=record.model_copy(update={"status":status})
        with self.catalog.connection() as con: con.execute("UPDATE hypotheses SET record_json=?,status=? WHERE hypothesis_id=?",(updated.model_dump_json(),status.value,hypothesis_id))
        self.append_journal("HYPOTHESIS_"+status.value,record.program_id,f"Hypothesis {hypothesis_id} changed to {status.value}",question_id=record.question_id,hypothesis_id=hypothesis_id)
        return updated

    def edit_hypothesis_content(self, hypothesis_id: str, **changes) -> Hypothesis:
        record=self.get_hypothesis(hypothesis_id)
        if not record: raise ResearchError("hypothesis not found")
        if record.status in {HypothesisStatus.APPROVED,HypothesisStatus.TESTING,HypothesisStatus.SUPPORTED,HypothesisStatus.REJECTED,HypothesisStatus.INCONCLUSIVE,HypothesisStatus.RETIRED}: raise ImmutableRecordError("approved hypothesis scientific content is immutable")
        updated=record.model_copy(update={**changes,"revision":record.revision+1})
        with self.catalog.connection() as con: con.execute("UPDATE hypotheses SET record_json=?,revision=? WHERE hypothesis_id=?",(updated.model_dump_json(),updated.revision,hypothesis_id))
        return updated

    def create_experiment(self, program_id: str, question_id: str, hypothesis_id: str, title: str, description: str, dataset_ids, strategy_spec: dict, execution_config: dict, evaluation_spec: dict, code_version: dict | None=None, parent_experiment_id: str | None=None, change_reason: str | None=None, experiment_id: str | None=None, experiment_family_id: str | None=None, change_reason_type: str | None=None, change_reason_text: str | None=None, fields_changed_from_parent=(), research_mode: str | None=None, motivation_spec: dict | None=None) -> ExperimentSpec:
        experiment_id=experiment_id or self._allocate_experiment_id()
        code_version=code_version or self.capture_code_version()
        fingerprint=experiment_fingerprint(question_id=question_id,hypothesis_id=hypothesis_id,dataset_ids=dataset_ids,strategy_spec=strategy_spec,execution_config=execution_config,evaluation_spec=evaluation_spec,code_version=code_version)
        if not experiment_family_id:
            parent=self.get_experiment(parent_experiment_id) if parent_experiment_id else None
            experiment_family_id=parent.experiment_family_id if parent and parent.experiment_family_id else "FAMILY-"+uuid4().hex[:12].upper()
        record=ExperimentSpec(experiment_id=experiment_id,program_id=program_id,question_id=question_id,hypothesis_id=hypothesis_id,title=title,description=description,dataset_ids=tuple(sorted(dataset_ids)),strategy_spec=strategy_spec,execution_config=execution_config,evaluation_spec=evaluation_spec,code_version=code_version,created_at=now_utc(),experiment_fingerprint=fingerprint,parent_experiment_id=parent_experiment_id,change_reason=change_reason,experiment_family_id=experiment_family_id,change_reason_type=change_reason_type,change_reason_text=change_reason_text or change_reason,fields_changed_from_parent=tuple(sorted(fields_changed_from_parent)),research_mode=research_mode,motivation_spec=motivation_spec or {})
        with self.catalog.connection() as con:
            con.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(str(uuid4()),experiment_id,program_id,question_id,hypothesis_id,parent_experiment_id,change_reason,fingerprint,record.model_dump_json(),record.status.value,record.created_at.isoformat(),None))
        self.append_journal("EXPERIMENT_CREATED",program_id,f"Experiment {experiment_id} created",question_id=question_id,hypothesis_id=hypothesis_id,experiment_id=experiment_id,metadata={"fingerprint":fingerprint})
        return record

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        value=self._get_json("experiments","experiment_id",experiment_id); return ExperimentSpec.model_validate(value) if value else None

    def edit_experiment(self, experiment_id: str, **changes) -> ExperimentSpec:
        record=self.get_experiment(experiment_id)
        if not record: raise ResearchError("experiment not found")
        if record.status != ExperimentStatus.DRAFT: raise ImmutableRecordError("approved experiment specification is immutable")
        relevant={"question_id","hypothesis_id","dataset_ids","strategy_spec","execution_config","evaluation_spec","code_version"}
        values=record.model_dump(); values.update(changes)
        if relevant.intersection(changes):
            values["experiment_fingerprint"]=experiment_fingerprint(question_id=values["question_id"],hypothesis_id=values["hypothesis_id"],dataset_ids=values["dataset_ids"],strategy_spec=values["strategy_spec"],execution_config=values["execution_config"],evaluation_spec=values["evaluation_spec"],code_version=values["code_version"])
        updated=ExperimentSpec.model_validate(values)
        with self.catalog.connection() as con: con.execute("UPDATE experiments SET fingerprint=?,record_json=? WHERE experiment_id=?",(updated.experiment_fingerprint,updated.model_dump_json(),experiment_id))
        return updated

    def verify_dataset(self, dataset_id: str) -> dict:
        with self.catalog.connection() as con: row=con.execute("SELECT * FROM datasets WHERE dataset_id=?",(dataset_id,)).fetchone()
        if not row: raise DatasetVerificationError(f"dataset not registered: {dataset_id}")
        record=dict(row); path=Path(record["path"])
        if not path.is_absolute(): path=self.project_root/path
        if not path.exists(): raise DatasetVerificationError(f"dataset file missing: {dataset_id}")
        actual=sha256_file(path)
        if actual.lower()!=record["checksum"].lower(): raise DatasetVerificationError(f"dataset checksum mismatch: {dataset_id}")
        with self.catalog.connection() as con: provenance=con.execute("SELECT record_json FROM provenance WHERE dataset_id=?",(dataset_id,)).fetchone();seal=con.execute("SELECT provenance_fingerprint FROM dataset_provenance_seals WHERE dataset_id=?",(dataset_id,)).fetchone()
        if not provenance: raise DatasetVerificationError(f"dataset provenance missing: {dataset_id}")
        import hashlib
        if not seal or hashlib.sha256(provenance[0].encode("utf-8")).hexdigest()!=seal[0]:raise DatasetVerificationError(f"dataset provenance seal mismatch: {dataset_id}")
        details=json.loads(provenance[0]);comparisons={"dataset_id","broker","server","symbol","timeframe","start","end","row_count","path","checksum","schema_version","retrieved_at"}
        if any(str(record[k])!=str(details[k]) for k in comparisons):raise DatasetVerificationError(f"dataset provenance/catalog mismatch: {dataset_id}")
        record["verified_path"]=str(path); record["provenance"]=details; return record

    def _approval_errors(self, record: ExperimentSpec) -> list[str]:
        errors=[]
        if not self.get_program(record.program_id): errors.append("missing program")
        if not self.get_question(record.question_id): errors.append("missing research question")
        hypothesis=self.get_hypothesis(record.hypothesis_id)
        if not hypothesis: errors.append("missing hypothesis")
        elif hypothesis.status not in {HypothesisStatus.APPROVED,HypothesisStatus.TESTING}: errors.append("hypothesis is not approved")
        if not record.dataset_ids: errors.append("missing dataset selection")
        if not record.strategy_spec: errors.append("missing strategy specification")
        if not record.execution_config: errors.append("missing execution configuration")
        else:
            try: ExecutionConfig.model_validate(record.execution_config)
            except Exception as exc: errors.append(f"invalid execution configuration: {exc}")
        if not record.evaluation_spec: errors.append("missing evaluation criteria")
        if not record.code_version: errors.append("missing code version")
        question=self.get_question(record.question_id)
        if question and question.control_enabled:
            if not record.research_mode:errors.append("missing research mode")
            required={"question_answered","why_it_matters","evidence_or_theory","learn_from_pass","learn_from_fail","outcome_actions"}
            missing=sorted(required-set(record.motivation_spec))
            if missing:errors.append("missing why-test fields: "+", ".join(missing))
        for dataset_id in record.dataset_ids:
            try: self.verify_dataset(dataset_id)
            except DatasetVerificationError as exc: errors.append(str(exc))
        return errors

    def approve_experiment(self, experiment_id: str) -> ExperimentSpec:
        record=self.get_experiment(experiment_id)
        if not record: raise ResearchError("experiment not found")
        if record.status != ExperimentStatus.DRAFT: raise InvalidTransitionError("only DRAFT experiments can be approved")
        errors=self._approval_errors(record)
        if errors: raise PreRegistrationError("APPROVAL REJECTED: "+"; ".join(errors))
        with self.catalog.connection() as con: duplicate=con.execute("SELECT experiment_id FROM experiments WHERE fingerprint=? AND experiment_id<>? AND status<>'DRAFT' ORDER BY created_at LIMIT 1",(record.experiment_fingerprint,experiment_id)).fetchone()
        if duplicate:
            self.append_journal("EXPERIMENT_REJECTED_AS_DUPLICATE",record.program_id,f"Experiment {experiment_id} duplicates {duplicate[0]}",question_id=record.question_id,hypothesis_id=record.hypothesis_id,experiment_id=experiment_id,metadata={"existing_experiment_id":duplicate[0]})
            raise DuplicateExperimentError(duplicate[0])
        approved_at=now_utc(); updated=record.model_copy(update={"status":ExperimentStatus.APPROVED,"approved_at":approved_at})
        with self.catalog.connection() as con: con.execute("UPDATE experiments SET record_json=?,status=?,approved_at=? WHERE experiment_id=?",(updated.model_dump_json(),updated.status.value,approved_at.isoformat(),experiment_id))
        self.append_journal("EXPERIMENT_APPROVED",record.program_id,f"Experiment {experiment_id} approved",question_id=record.question_id,hypothesis_id=record.hypothesis_id,experiment_id=experiment_id)
        return updated

    def transition_experiment(self, experiment_id: str, target: ExperimentStatus) -> ExperimentSpec:
        record=self.get_experiment(experiment_id)
        if not record: raise ResearchError("experiment not found")
        if target not in TRANSITIONS[record.status]: raise InvalidTransitionError(f"invalid experiment transition {record.status} -> {target}")
        if target == ExperimentStatus.RUNNING:
            for dataset_id in record.dataset_ids: self.verify_dataset(dataset_id)
            try: ExecutionConfig.model_validate(record.execution_config)
            except Exception as exc: raise ResearchError(f"installed execution engine incompatible: {exc}") from exc
            from sandbox.audit.auditor import ResearchAuditor
            ResearchAuditor(self).can_execute(experiment_id)
        updated=record.model_copy(update={"status":target})
        with self.catalog.connection() as con: con.execute("UPDATE experiments SET record_json=?,status=? WHERE experiment_id=?",(updated.model_dump_json(),target.value,experiment_id))
        event={ExperimentStatus.RUNNING:"EXPERIMENT_STARTED",ExperimentStatus.COMPLETED:"EXPERIMENT_COMPLETED"}.get(target,"EXPERIMENT_"+target.value)
        self.append_journal(event,record.program_id,f"Experiment {experiment_id} changed to {target.value}",question_id=record.question_id,hypothesis_id=record.hypothesis_id,experiment_id=experiment_id)
        return updated

    def experiment_ancestry(self, experiment_id: str) -> list[dict]:
        result=[]; current=self.get_experiment(experiment_id); seen=set()
        while current:
            if current.experiment_id in seen: raise ResearchError("experiment ancestry cycle")
            seen.add(current.experiment_id); result.append({"experiment_id":current.experiment_id,"parent_experiment_id":current.parent_experiment_id,"change_reason":current.change_reason,"fingerprint":current.experiment_fingerprint})
            current=self.get_experiment(current.parent_experiment_id) if current.parent_experiment_id else None
        return list(reversed(result))

    def create_replication(self, experiment_id: str, run_id: str, run_fingerprint: str, original_run_id: str | None=None) -> dict:
        record=self.get_experiment(experiment_id)
        if not record or record.status == ExperimentStatus.DRAFT: raise ResearchError("replication requires an approved experiment")
        link_id=str(uuid4()); created=now_utc()
        with self.catalog.connection() as con: con.execute("INSERT INTO experiment_runs VALUES (?,?,?,?,?,?,?,?)",(link_id,experiment_id,run_id,run_fingerprint,record.experiment_fingerprint,"REPLICATION",original_run_id,created.isoformat()))
        self.append_journal("REPLICATION_RUN_CREATED",record.program_id,f"Replication {run_id} linked to {experiment_id}",question_id=record.question_id,hypothesis_id=record.hypothesis_id,experiment_id=experiment_id,run_id=run_id,metadata={"original_run_id":original_run_id})
        return {"link_id":link_id,"experiment_id":experiment_id,"run_id":run_id,"original_run_id":original_run_id,"run_kind":"REPLICATION"}

    def link_stage2_run(self, experiment_id: str, run_id: str, run_kind: str="PRIMARY") -> ExperimentResult:
        experiment=self.get_experiment(experiment_id)
        if not experiment or experiment.status not in {ExperimentStatus.RUNNING,ExperimentStatus.COMPLETED}: raise ResearchError("experiment must be RUNNING or COMPLETED")
        with self.catalog.connection() as con: row=con.execute("SELECT * FROM research_runs WHERE run_id=?",(run_id,)).fetchone()
        if not row: raise ResearchError(f"Stage 2 run not found: {run_id}")
        run=dict(row)
        if run["engine_version"] != EXECUTION_ENGINE_VERSION: raise ResearchError("execution engine version mismatch")
        if json.loads(run["execution_config"]) != ExecutionConfig.model_validate(experiment.execution_config).model_dump(mode="json"): raise ResearchError("execution configuration mismatch")
        if set(run["dataset_id"].split(",")) != set(experiment.dataset_ids): raise ResearchError("run dataset linkage mismatch")
        for dataset_id in experiment.dataset_ids: self.verify_dataset(dataset_id)
        summary_path=Path(run["summary_path"]); ledger_path=Path(run["ledger_path"])
        if not summary_path.is_absolute(): summary_path=self.project_root/summary_path
        if not ledger_path.is_absolute(): ledger_path=self.project_root/ledger_path
        summary=json.loads(summary_path.read_text(encoding="utf-8")); checksum=sha256_file(ledger_path)
        if checksum != summary["ledger_checksum"]: raise ResearchError("Stage 2 ledger checksum mismatch")
        created=now_utc(); link_id=str(uuid4())
        with self.catalog.connection() as con:
            con.execute("INSERT OR IGNORE INTO experiment_runs VALUES (?,?,?,?,?,?,?,?)",(link_id,experiment_id,run_id,run["run_fingerprint"],experiment.experiment_fingerprint,run_kind,None,created.isoformat()))
        result=ExperimentResult(result_id="RES-"+uuid4().hex,experiment_id=experiment_id,run_id=run_id,run_fingerprint=run["run_fingerprint"],summary_metrics=summary["metrics"],ledger_checksum=checksum,created_at=created)
        with self.catalog.connection() as con: con.execute("INSERT INTO experiment_results VALUES (?,?,?,?,?)",(result.result_id,experiment_id,run_id,result.model_dump_json(),created.isoformat()))
        self.append_journal("EXPERIMENT_RESULT_RECORDED",experiment.program_id,f"Result {result.result_id} recorded",question_id=experiment.question_id,hypothesis_id=experiment.hypothesis_id,experiment_id=experiment_id,run_id=run_id,metadata={"ledger_checksum":checksum})
        from sandbox.audit.auditor import ResearchAuditor
        ResearchAuditor(self).audit(experiment_id,__import__("sandbox.audit.models",fromlist=["AuditPhase"]).AuditPhase.POST_RUN)
        return result

    def get_result(self, result_id: str) -> ExperimentResult | None:
        value=self._get_json("experiment_results","result_id",result_id); return ExperimentResult.model_validate(value) if value else None

    def record_verdict(self, experiment_id: str, result_id: str, verdict: VerdictValue, rationale: str, decided_by: str) -> ExperimentVerdict:
        experiment=self.get_experiment(experiment_id); result=self.get_result(result_id)
        if not experiment or not result or result.experiment_id != experiment_id: raise ResearchError("verdict result linkage invalid")
        mismatches=[]; metrics=result.summary_metrics; criteria=experiment.evaluation_spec
        checks={"minimum_win_rate":"win_rate","minimum_trades_per_year":"trades_per_year","minimum_resolved_trades":"resolved_trades","minimum_net_r":"net_R"}
        if verdict==VerdictValue.PASS:
            for criterion,metric in checks.items():
                if criterion in criteria and float(metrics.get(metric,float("-inf")))<float(criteria[criterion]):mismatches.append({"criterion":criterion,"required":criteria[criterion],"actual":metrics.get(metric)})
        if mismatches:
            self.append_journal("VERDICT_CRITERIA_MISMATCH",experiment.program_id,f"PASS verdict rejected for {experiment_id}",question_id=experiment.question_id,hypothesis_id=experiment.hypothesis_id,experiment_id=experiment_id,metadata={"result_id":result_id,"mismatches":mismatches})
            raise ResearchError("VERDICT_CRITERIA_MISMATCH: PASS conflicts with frozen evaluation criteria")
        record=ExperimentVerdict(verdict_id="VER-"+uuid4().hex,experiment_id=experiment_id,result_id=result_id,verdict=verdict,rationale=rationale,decided_at=now_utc(),decided_by=decided_by)
        with self.catalog.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            if con.execute("SELECT 1 FROM experiment_verdicts WHERE result_id=?",(result_id,)).fetchone():raise ImmutableRecordError("result already has an immutable verdict")
            con.execute("INSERT INTO experiment_verdicts VALUES (?,?,?,?,?)",(record.verdict_id,experiment_id,result_id,record.model_dump_json(),record.decided_at.isoformat()))
        self.append_journal("VERDICT_RECORDED",experiment.program_id,f"Verdict {verdict.value} recorded for {experiment_id}",question_id=experiment.question_id,hypothesis_id=experiment.hypothesis_id,experiment_id=experiment_id,metadata={"result_id":result_id,"verdict":verdict.value})
        return record

    def audit_context(self, experiment_id: str) -> dict:
        experiment=self.get_experiment(experiment_id)
        if not experiment: raise ResearchError("experiment not found")
        with self.catalog.connection() as con:
            runs=[dict(x) for x in con.execute("SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY created_at",(experiment_id,))]
            results=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM experiment_results WHERE experiment_id=? ORDER BY created_at",(experiment_id,))]
            verdicts=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM experiment_verdicts WHERE experiment_id=? ORDER BY decided_at",(experiment_id,))]
            previous=[x[0] for x in con.execute("SELECT experiment_id FROM experiments WHERE question_id=? AND experiment_id<>? ORDER BY created_at",(experiment.question_id,experiment_id))]
        datasets=[]
        for dataset_id in experiment.dataset_ids:
            verified=self.verify_dataset(dataset_id); verified.pop("verified_path",None); datasets.append(verified)
        audits=[]; overrides=[]
        try:
            with self.catalog.connection() as con:
                audits=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM research_audits WHERE experiment_id=? ORDER BY audited_at",(experiment_id,))]
                overrides=[dict(x) for x in con.execute("SELECT * FROM audit_overrides WHERE experiment_id=? ORDER BY approved_at",(experiment_id,))]
        except sqlite3.OperationalError: pass
        partitions=[];bindings=[];contamination=[]
        try:
            with self.catalog.connection() as con:
                partitions=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM data_partitions WHERE source_dataset_id IN (%s)"%",".join("?"*len(experiment.dataset_ids)),tuple(experiment.dataset_ids))] if experiment.dataset_ids else []
                candidate_ids=[x[0] for x in con.execute("SELECT candidate_id FROM candidates WHERE experiment_family_id=?",(experiment.experiment_family_id,))]
                if candidate_ids:
                    placeholders=",".join("?"*len(candidate_ids));bindings=[json.loads(x[0]) for x in con.execute(f"SELECT record_json FROM partition_bindings WHERE candidate_id IN ({placeholders})",tuple(candidate_ids))];contamination=[json.loads(x[0]) for x in con.execute(f"SELECT record_json FROM contamination_records WHERE candidate_id IN ({placeholders})",tuple(candidate_ids))]
                    bound_ids=[x["partition_id"] for x in bindings]
                    if bound_ids:
                        bound_placeholders=",".join("?"*len(bound_ids));partitions.extend(json.loads(x[0]) for x in con.execute(f"SELECT record_json FROM data_partitions WHERE partition_id IN ({bound_placeholders})",tuple(bound_ids)))
        except sqlite3.OperationalError:pass
        for item in partitions:item["path"]="[PROTECTED_PARTITION_PATH]"
        evidence=[]
        try:
            with self.catalog.connection() as con:evidence=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM statistical_evidence_reports WHERE experiment_id=? ORDER BY created_at",(experiment_id,))]
        except sqlite3.OperationalError:pass
        return {"program":self.get_program(experiment.program_id).model_dump(mode="json"),"question":self.get_question(experiment.question_id).model_dump(mode="json"),"hypothesis":self.get_hypothesis(experiment.hypothesis_id).model_dump(mode="json"),"experiment":experiment.model_dump(mode="json"),"ancestry":self.experiment_ancestry(experiment_id),"previous_related_experiments":previous,"dataset_provenance":datasets,"execution_configuration":experiment.execution_config,"evaluation_specification":experiment.evaluation_spec,"runs":runs,"results":results,"verdicts":verdicts,"research_audits":audits,"audit_overrides":overrides,"partitions":partitions,"partition_bindings":bindings,"contamination":contamination,"statistical_evidence_reports":evidence,"access_policy_version":__import__("sandbox",fromlist=["ACCESS_POLICY_VERSION"]).ACCESS_POLICY_VERSION}

    def generate_manifest(self, experiment_id: str) -> tuple[Path,Path]:
        context=self.audit_context(experiment_id); folder=self.results_root/"experiments"/experiment_id; folder.mkdir(parents=True,exist_ok=True)
        json_path=folder/"manifest.json"; md_path=folder/"manifest.md"
        serialized=json.dumps(context,indent=2,sort_keys=True,default=str)
        if json_path.exists() and json_path.read_text(encoding="utf-8")!=serialized:
            suffix=sha256_canonical(context)[:12];json_path=folder/f"manifest-{suffix}.json";md_path=folder/f"manifest-{suffix}.md"
        json_path.write_text(serialized,encoding="utf-8")
        e=context["experiment"]; lines=[f"# Experiment {experiment_id}","",f"- Status: {e['status']}",f"- Fingerprint: `{e['experiment_fingerprint']}`",f"- Program: {context['program']['program_id']} — {context['program']['name']}",f"- Question: {context['question']['question_id']} — {context['question']['question_text']}",f"- Hypothesis: {context['hypothesis']['hypothesis_id']} — {context['hypothesis']['hypothesis_text']}",f"- Code version: `{canonical_json(e['code_version'])}`",f"- Execution engine: `{EXECUTION_ENGINE_VERSION}`","", "## Evaluation specification","",f"```json\n{json.dumps(e['evaluation_spec'],indent=2,sort_keys=True)}\n```","","## Runs and results","",f"```json\n{json.dumps({'runs':context['runs'],'results':context['results'],'verdicts':context['verdicts']},indent=2,sort_keys=True)}\n```"]
        md_path.write_text("\n".join(lines)+"\n",encoding="utf-8"); return json_path,md_path

    def timeline(self, program_id: str) -> str:
        program=self.get_program(program_id)
        if not program: raise ResearchError("program not found")
        with self.catalog.connection() as con:
            questions=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM research_questions WHERE program_id=? ORDER BY created_at",(program_id,))]
            hypotheses=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM hypotheses WHERE program_id=? ORDER BY created_at",(program_id,))]
            experiments=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM experiments WHERE program_id=? ORDER BY created_at",(program_id,))]
            verdict_rows=[json.loads(x[0]) for x in con.execute("SELECT v.record_json FROM experiment_verdicts v JOIN experiments e ON e.experiment_id=v.experiment_id WHERE e.program_id=? ORDER BY v.decided_at",(program_id,))]
        verdicts={x["experiment_id"]:x["verdict"] for x in verdict_rows}; lines=[program.name]
        for question in questions:
            lines.append(question["question_id"]+" - "+question["question_text"])
            for hypothesis in [h for h in hypotheses if h["question_id"]==question["question_id"]]:
                lines.append("+-- "+hypothesis["hypothesis_id"]+" - "+hypothesis["hypothesis_text"])
                for experiment in [e for e in experiments if e["hypothesis_id"]==hypothesis["hypothesis_id"]]: lines.append("|   +-- "+experiment["experiment_id"]+" -> "+verdicts.get(experiment["experiment_id"],experiment["status"]))
        return "\n".join(lines)
