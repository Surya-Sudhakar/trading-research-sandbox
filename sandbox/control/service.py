from __future__ import annotations

import json,re,sqlite3
from pathlib import Path
from uuid import uuid4

from sandbox.audit.auditor import ResearchAuditor
from sandbox.audit.models import DatasetUsageType
from sandbox.control.models import CandidateStatus,CandidateStrategy,DecisionAction,DecisionNode,DiversionClassification,EvidenceOrigin,ResearchMode,StrategyFamily,StrategyFamilyStatus
from sandbox.research.canonical import canonical_json,sha256_canonical
from sandbox.research.errors import ImmutableRecordError,ResearchError
from sandbox.research.models import ExperimentStatus,HypothesisStatus,QuestionStatus,now_utc
from sandbox.research.registry import ResearchRegistry


DDL="""
CREATE TABLE IF NOT EXISTS question_status_history(history_id TEXT PRIMARY KEY,question_id TEXT NOT NULL,old_status TEXT,new_status TEXT NOT NULL,reason TEXT,new_evidence TEXT,changed_by TEXT,changed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decision_nodes(decision_id TEXT PRIMARY KEY,question_id TEXT NOT NULL,record_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS strategy_families(strategy_family_id TEXT PRIMARY KEY,program_id TEXT NOT NULL,record_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates(candidate_id TEXT PRIMARY KEY,candidate_base_id TEXT NOT NULL,version INTEGER NOT NULL,program_id TEXT NOT NULL,strategy_family_id TEXT NOT NULL,experiment_family_id TEXT NOT NULL,parent_candidate_id TEXT,record_json TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(candidate_base_id,version));
CREATE TABLE IF NOT EXISTS exploration_events(event_id TEXT PRIMARY KEY,program_id TEXT NOT NULL,question_id TEXT,hypothesis_id TEXT,experiment_id TEXT,candidate_id TEXT,event_type TEXT NOT NULL,dataset_id TEXT,description TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS validation_events(event_id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,event_type TEXT NOT NULL,dataset_ids_json TEXT NOT NULL,details_json TEXT NOT NULL,created_at TEXT NOT NULL);
"""


def _norm(text):return re.sub(r"\s+"," ",text.strip().lower())


class ResearchControl:
    def __init__(self,registry:ResearchRegistry):
        self.registry=registry;self.auditor=ResearchAuditor(registry)
        with registry.catalog.connection() as con:con.executescript(DDL)

    def create_question(self,*,question_id,program_id,question_text,parent_question_id,purpose,relevance_to_parent,evidence_origins,research_mode,rationale=""):
        if not parent_question_id:raise ResearchError("controlled non-root question requires parent_question_id")
        missing=[name for name,value in (("purpose",purpose),("relevance_to_parent",relevance_to_parent),("evidence_origins",evidence_origins),("research_mode",research_mode)) if not value]
        if missing:raise ResearchError("QUESTION_CONTROL_REJECTED: missing "+", ".join(missing))
        with self.registry.catalog.connection() as con:rows=con.execute("SELECT record_json FROM research_questions WHERE program_id=?",(program_id,)).fetchall()
        for row in rows:
            existing=json.loads(row[0])
            if _norm(existing["question_text"])==_norm(question_text):
                if existing["status"] in {"CLOSED","ANSWERED","REJECTED"}:
                    self.registry.append_journal("POSSIBLE_RETURN_TO_CLOSED_QUESTION",program_id,f"Question {question_id} resembles closed {existing['question_id']}",question_id=existing["question_id"])
                    raise ResearchError("POSSIBLE_RETURN_TO_CLOSED_QUESTION: reopen the existing question explicitly")
                raise ResearchError(f"duplicate normalized question: {existing['question_id']}")
        record=self.registry.create_question(question_id,program_id,question_text,rationale,parent_question_id,purpose,relevance_to_parent,[EvidenceOrigin(x).value for x in evidence_origins],ResearchMode(research_mode).value,True)
        self.registry.append_journal("BRANCH_CREATED",program_id,f"Controlled branch {question_id} created",question_id=question_id,metadata={"mode":record.research_mode,"origins":record.evidence_origins})
        return record

    def set_question_status(self,question_id,status:QuestionStatus,reason="",new_evidence="",changed_by="system"):
        record=self.registry.get_question(question_id)
        if not record:raise ResearchError("question not found")
        if record.status==QuestionStatus.CLOSED and status!=QuestionStatus.CLOSED:raise ResearchError("closed question requires explicit reopen")
        updated=record.model_copy(update={"status":status})
        with self.registry.catalog.connection() as con:
            con.execute("UPDATE research_questions SET record_json=?,status=? WHERE question_id=?",(updated.model_dump_json(),status.value,question_id))
            con.execute("INSERT INTO question_status_history VALUES (?,?,?,?,?,?,?,?)",(str(uuid4()),question_id,record.status.value,status.value,reason,new_evidence,changed_by,now_utc().isoformat()))
        event="QUESTION_CLOSED" if status==QuestionStatus.CLOSED else "QUESTION_STATUS_CHANGED";self.registry.append_journal(event,record.program_id,f"Question {question_id}: {record.status.value} -> {status.value}",question_id=question_id,metadata={"reason":reason})
        return updated

    def reopen_question(self,question_id,reopen_reason,new_evidence,reopened_by):
        if not reopen_reason or not new_evidence or not reopened_by:raise ResearchError("reopening requires reason, new evidence, and reopened_by")
        record=self.registry.get_question(question_id)
        if not record or record.status!=QuestionStatus.CLOSED:raise ResearchError("only a CLOSED question may be reopened")
        updated=record.model_copy(update={"status":QuestionStatus.OPEN})
        with self.registry.catalog.connection() as con:
            con.execute("UPDATE research_questions SET record_json=?,status=? WHERE question_id=?",(updated.model_dump_json(),updated.status.value,question_id))
            con.execute("INSERT INTO question_status_history VALUES (?,?,?,?,?,?,?,?)",(str(uuid4()),question_id,"CLOSED","OPEN",reopen_reason,new_evidence,reopened_by,now_utc().isoformat()))
        self.registry.append_journal("QUESTION_REOPENED",record.program_id,f"Question {question_id} reopened",question_id=question_id,metadata={"reopen_reason":reopen_reason,"new_evidence":new_evidence,"reopened_by":reopened_by})
        return updated

    def close_branch(self,question_id,reason,closed_by):
        root=self.registry.get_question(question_id)
        if not root:raise ResearchError("question not found")
        queue=[question_id];closed=[]
        while queue:
            qid=queue.pop();q=self.registry.get_question(qid)
            if q and q.status!=QuestionStatus.CLOSED:self.set_question_status(qid,QuestionStatus.CLOSED,reason,changed_by=closed_by);closed.append(qid)
            with self.registry.catalog.connection() as con:queue.extend(x[0] for x in con.execute("SELECT question_id FROM research_questions WHERE parent_question_id=?",(qid,)))
        self.registry.append_journal("BRANCH_CLOSED",root.program_id,f"Branch {question_id} closed",question_id=question_id,metadata={"questions":closed,"reason":reason})
        return closed

    def create_decision(self,question_id,description,possible_outcomes,research_mode,stopping_conditions=None):
        q=self.registry.get_question(question_id)
        if not q:raise ResearchError("question not found")
        record=DecisionNode(decision_id="DEC-"+uuid4().hex,question_id=question_id,description=description,possible_outcomes={k:DecisionAction(v) for k,v in possible_outcomes.items()},created_at=now_utc(),research_mode=ResearchMode(research_mode),stopping_conditions=stopping_conditions or {})
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO decision_nodes VALUES (?,?,?,?)",(record.decision_id,question_id,record.model_dump_json(),record.created_at.isoformat()))
        self.registry.append_journal("DECISION_CREATED",q.program_id,f"Decision {record.decision_id} created",question_id=question_id,metadata={"outcomes":list(possible_outcomes)})
        return record

    def record_decision(self,decision_id,outcome):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM decision_nodes WHERE decision_id=?",(decision_id,)).fetchone()
        if not row:raise ResearchError("decision not found")
        record=DecisionNode.model_validate_json(row[0])
        if outcome not in record.possible_outcomes:raise ResearchError("outcome was not pre-registered")
        updated=record.model_copy(update={"recorded_outcome":outcome,"recorded_action":record.possible_outcomes[outcome]})
        with self.registry.catalog.connection() as con:con.execute("UPDATE decision_nodes SET record_json=? WHERE decision_id=?",(updated.model_dump_json(),decision_id))
        q=self.registry.get_question(record.question_id);self.registry.append_journal("DECISION_RECORDED",q.program_id,f"Decision {decision_id}: {outcome}",question_id=q.question_id,metadata={"action":updated.recorded_action.value})
        return updated

    def create_family(self,program_id,name,rationale,strategy_family_id=None):
        if not self.registry.get_program(program_id):raise ResearchError("program not found")
        with self.registry.catalog.connection() as con:existing=[StrategyFamily.model_validate_json(x[0]) for x in con.execute("SELECT record_json FROM strategy_families WHERE program_id=?",(program_id,))]
        duplicate=next((x for x in existing if _norm(x.name)==_norm(name)),None)
        if duplicate:raise ResearchError(f"DUPLICATE_STRATEGY_FAMILY: existing {duplicate.strategy_family_id} remains authoritative")
        record=StrategyFamily(strategy_family_id=strategy_family_id or "SFAM-"+uuid4().hex[:12].upper(),program_id=program_id,name=name,rationale=rationale,created_at=now_utc(),status=StrategyFamilyStatus.PROPOSED)
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO strategy_families VALUES (?,?,?,?,?)",(record.strategy_family_id,program_id,record.model_dump_json(),record.status.value,record.created_at.isoformat()))
        self.registry.append_journal("FAMILY_CREATED",program_id,f"Strategy family {record.strategy_family_id} created",metadata={"name":name})
        return record

    def get_family(self,family_id):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM strategy_families WHERE strategy_family_id=?",(family_id,)).fetchone()
        return StrategyFamily.model_validate_json(row[0]) if row else None

    def set_family_status(self,family_id,status:StrategyFamilyStatus,reason=""):
        record=self.get_family(family_id)
        if not record:raise ResearchError("family not found")
        if record.status==StrategyFamilyStatus.EXHAUSTED and status!=StrategyFamilyStatus.EXHAUSTED:raise ResearchError("EXHAUSTED strategy family cannot be reset")
        updated=record.model_copy(update={"status":status})
        with self.registry.catalog.connection() as con:con.execute("UPDATE strategy_families SET record_json=?,status=? WHERE strategy_family_id=?",(updated.model_dump_json(),status.value,family_id))
        event="FAMILY_EXHAUSTED" if status==StrategyFamilyStatus.EXHAUSTED else "FAMILY_STATUS_CHANGED";self.registry.append_journal(event,record.program_id,f"Family {family_id}: {status.value}",metadata={"reason":reason})
        return updated

    def _history(self,experiment_family_id,program_id):
        with self.registry.catalog.connection() as con:
            experiments=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM experiments WHERE program_id=? ORDER BY created_at",(program_id,)) if json.loads(x[0]).get("experiment_family_id")==experiment_family_id]
            questions=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM research_questions WHERE program_id=?",(program_id,))]
            hypotheses=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM hypotheses WHERE program_id=?",(program_id,))]
            events=[dict(x) for x in con.execute("SELECT * FROM exploration_events WHERE program_id=?",(program_id,))]
            usages=[dict(x) for x in con.execute("SELECT * FROM dataset_usage WHERE program_id=?",(program_id,))]
        parameter_paths=self.auditor.parameter_history(experiment_family_id)
        return {"questions_explored":len(questions),"hypotheses_created":len(hypotheses),"experiments_attempted":len(experiments),"parameters_changed":len(parameter_paths),"filters_introduced":sum("filter" in k.lower() for k in parameter_paths),"datasets_inspected":len({x["dataset_id"] for x in usages}),"winner_loser_analyses":sum(x["event_type"]=="WINNER_LOSER_ANALYSIS" for x in events),"result_driven_descendants":sum(x.get("change_reason_type")=="RESULT_DRIVEN" for x in experiments),"abandoned_branches":sum(x["status"] in {"CANCELLED","FAILED"} for x in experiments),"experiment_ids":[x["experiment_id"] for x in experiments]}

    def record_exploration(self,program_id,event_type,description,question_id=None,hypothesis_id=None,experiment_id=None,candidate_id=None,dataset_id=None):
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO exploration_events VALUES (?,?,?,?,?,?,?,?,?,?)",(str(uuid4()),program_id,question_id,hypothesis_id,experiment_id,candidate_id,event_type,dataset_id,description,now_utc().isoformat()))

    def create_candidate(self,program_id,originating_question_id,originating_hypothesis_id,experiment_family_id,strategy_family_id,strategy_spec,execution_config,evaluation_spec,symbols,timeframes,session_rules,code_version,discovery_dataset_ids,candidate_base_id=None,parent_candidate_id=None):
        q=self.registry.get_question(originating_question_id);h=self.registry.get_hypothesis(originating_hypothesis_id);family=self.get_family(strategy_family_id)
        if not q or not h or not family:raise ResearchError("candidate lineage is incomplete")
        base=candidate_base_id or "CAND-"+uuid4().hex[:10].upper()
        with self.registry.catalog.connection() as con:row=con.execute("SELECT max(version) FROM candidates WHERE candidate_base_id=?",(base,)).fetchone()
        version=(row[0] or 0)+1;cid=f"{base}-v{version}";history=self._history(experiment_family_id,program_id)
        scientific={"strategy_spec":strategy_spec,"execution_config":execution_config,"evaluation_spec":evaluation_spec,"symbols":sorted(symbols),"timeframes":sorted(timeframes),"session_rules":session_rules,"code_version":code_version}
        record=CandidateStrategy(candidate_id=cid,candidate_base_id=base,version=version,program_id=program_id,originating_question_id=originating_question_id,originating_hypothesis_id=originating_hypothesis_id,experiment_family_id=experiment_family_id,strategy_family_id=strategy_family_id,strategy_spec=strategy_spec,execution_config=execution_config,evaluation_spec=evaluation_spec,symbols=tuple(sorted(symbols)),timeframes=tuple(sorted(timeframes)),session_rules=session_rules,code_version=code_version,discovery_dataset_ids=tuple(sorted(discovery_dataset_ids)),discovery_history_summary=history,created_at=now_utc(),status=CandidateStatus.EXPLORATORY,candidate_fingerprint=sha256_canonical(scientific),parent_candidate_id=parent_candidate_id)
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)",(cid,base,version,program_id,strategy_family_id,experiment_family_id,parent_candidate_id,record.model_dump_json(),record.status.value,record.created_at.isoformat()))
        self.registry.append_journal("CANDIDATE_CREATED",program_id,f"Candidate {cid} created",question_id=originating_question_id,hypothesis_id=originating_hypothesis_id,metadata={"family":strategy_family_id})
        return record

    def get_candidate(self,candidate_id):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM candidates WHERE candidate_id=?",(candidate_id,)).fetchone()
        return CandidateStrategy.model_validate_json(row[0]) if row else None

    def freeze_candidate(self,candidate_id,validation_dataset_ids):
        record=self.get_candidate(candidate_id)
        if not record or record.status not in {CandidateStatus.EXPLORATORY,CandidateStatus.PROMISING}:raise ResearchError("only exploratory/promising candidate can be frozen")
        updated=record.model_copy(update={"status":CandidateStatus.FROZEN,"validation_dataset_ids":tuple(sorted(validation_dataset_ids)),"discovery_history_summary":self._history(record.experiment_family_id,record.program_id)})
        with self.registry.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=?,status=? WHERE candidate_id=?",(updated.model_dump_json(),updated.status.value,candidate_id))
        self.registry.append_journal("CANDIDATE_FROZEN",record.program_id,f"Candidate {candidate_id} frozen",question_id=record.originating_question_id,hypothesis_id=record.originating_hypothesis_id,metadata={"fingerprint":record.candidate_fingerprint,"validation_datasets":validation_dataset_ids})
        return updated

    def modify_candidate(self,candidate_id,**changes):
        record=self.get_candidate(candidate_id)
        if not record:raise ResearchError("candidate not found")
        if record.status not in {CandidateStatus.EXPLORATORY,CandidateStatus.PROMISING}:raise ImmutableRecordError("scientifically committed candidate is immutable; create a new version")
        values=record.model_dump();values.update(changes);scientific={k:values[k] for k in ("strategy_spec","execution_config","evaluation_spec","symbols","timeframes","session_rules","code_version")};values["candidate_fingerprint"]=sha256_canonical(scientific)
        updated=CandidateStrategy.model_validate(values)
        with self.registry.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=? WHERE candidate_id=?",(updated.model_dump_json(),candidate_id))
        return updated

    def new_candidate_version(self,candidate_id,**changes):
        parent=self.get_candidate(candidate_id)
        if not parent:raise ResearchError("candidate not found")
        values={k:getattr(parent,k) for k in ("program_id","originating_question_id","originating_hypothesis_id","experiment_family_id","strategy_family_id","strategy_spec","execution_config","evaluation_spec","symbols","timeframes","session_rules","code_version","discovery_dataset_ids")};values.update(changes)
        child=self.create_candidate(**values,candidate_base_id=parent.candidate_base_id,parent_candidate_id=parent.candidate_id)
        self.registry.append_journal("CANDIDATE_VERSION_CREATED",parent.program_id,f"Candidate {child.candidate_id} created from {parent.candidate_id}",metadata={"returns_to":"DISCOVERY"})
        self.registry.append_journal("RETURNED_TO_DISCOVERY",parent.program_id,f"{child.candidate_id} returned to discovery",question_id=child.originating_question_id,hypothesis_id=child.originating_hypothesis_id)
        return child

    def validation_eligibility(self,candidate_id):
        record=self.get_candidate(candidate_id);reasons=[]
        if not record:raise ResearchError("candidate not found")
        if record.status!=CandidateStatus.FROZEN:reasons.append("candidate is not frozen")
        if not record.strategy_spec or not record.execution_config or not record.evaluation_spec:reasons.append("frozen specification is incomplete")
        if not record.discovery_history_summary.get("experiment_ids"):reasons.append("discovery lineage is incomplete")
        if not record.validation_dataset_ids:reasons.append("validation dataset not designated")
        with self.registry.catalog.connection() as con:
            for dataset_id in record.validation_dataset_ids:
                try:self.registry.verify_dataset(dataset_id)
                except Exception as exc:reasons.append(str(exc))
                types={x[0] for x in con.execute("SELECT usage_type FROM dataset_usage WHERE dataset_id=? AND program_id=?",(dataset_id,record.program_id))}
                if "DISCOVERY" in types:reasons.append(f"validation dataset was used for discovery: {dataset_id}")
        with self.registry.catalog.connection() as con:audit_count=con.execute("SELECT count(*) FROM research_audits a JOIN experiments e ON e.experiment_id=a.experiment_id WHERE e.program_id=? AND e.fingerprint IN (SELECT fingerprint FROM experiments WHERE json_extract(record_json,'$.experiment_family_id')=?)",(record.program_id,record.experiment_family_id)).fetchone()[0]
        if audit_count==0:reasons.append("Stage 4 audit history missing")
        return {"eligible":not reasons,"reasons":reasons,"candidate_id":candidate_id}

    def request_validation(self,candidate_id):
        gate=self.validation_eligibility(candidate_id)
        if not gate["eligible"]:raise ResearchError("VALIDATION_INELIGIBLE: "+"; ".join(gate["reasons"]))
        record=self.get_candidate(candidate_id);updated=record.model_copy(update={"status":CandidateStatus.VALIDATING,"validation_status":"IN_PROGRESS"})
        with self.registry.catalog.connection() as con:con.execute("UPDATE candidates SET record_json=?,status=? WHERE candidate_id=?",(updated.model_dump_json(),updated.status.value,candidate_id))
        self.registry.append_journal("DISCOVERY_TO_VALIDATION_REQUESTED",record.program_id,f"Validation requested for {candidate_id}",metadata={"datasets":record.validation_dataset_ids})
        return updated

    def validation_failed(self,candidate_id,details):
        record=self.get_candidate(candidate_id)
        if not record or record.status not in {CandidateStatus.FROZEN,CandidateStatus.VALIDATING}:raise ResearchError("candidate is not in validation")
        for dataset_id in record.validation_dataset_ids:self.auditor.record_dataset_usage(dataset_id,record.program_id,candidate_id,DatasetUsageType.VALIDATION,"validation failure observed")
        updated=record.model_copy(update={"status":CandidateStatus.REJECTED,"validation_status":"FAILED"})
        with self.registry.catalog.connection() as con:
            con.execute("UPDATE candidates SET record_json=?,status=? WHERE candidate_id=?",(updated.model_dump_json(),updated.status.value,candidate_id));con.execute("INSERT INTO validation_events VALUES (?,?,?,?,?,?)",(str(uuid4()),candidate_id,"VALIDATION_FAILED",canonical_json(record.validation_dataset_ids),canonical_json(details),now_utc().isoformat()))
        self.registry.append_journal("VALIDATION_FAILED",record.program_id,f"Validation failed for {candidate_id}",metadata={"datasets":record.validation_dataset_ids,"details":details})
        return updated

    def final_test_eligibility(self,candidate_id,final_dataset_ids):
        record=self.get_candidate(candidate_id);reasons=[]
        if not record or record.status!=CandidateStatus.VALIDATED:reasons.append("successful validation required")
        if record and record.validation_status!="PASSED":reasons.append("validation status is not PASSED")
        with self.registry.catalog.connection() as con:
            for dataset_id in final_dataset_ids:
                if con.execute("SELECT 1 FROM dataset_usage WHERE dataset_id=? AND usage_type<>'UNSEEN'",(dataset_id,)).fetchone():reasons.append(f"final-test dataset has prior usage: {dataset_id}")
        return {"eligible":not reasons,"reasons":reasons}

    def diversion(self,question_id,research_mode):
        q=self.registry.get_question(question_id)
        if not q:return DiversionClassification.UNRELATED
        if not q.parent_question_id:return DiversionClassification.ON_PATH
        if q.relevance_to_parent and q.evidence_origins:
            return DiversionClassification.EXPLORATORY_BRANCH if ResearchMode(research_mode)==ResearchMode.DISCOVERY and EvidenceOrigin.DISCOVERY_DATA.value in q.evidence_origins else DiversionClassification.RELATED_BRANCH
        return DiversionClassification.POSSIBLE_DIVERSION if ResearchMode(research_mode)==ResearchMode.DISCOVERY else DiversionClassification.UNRELATED

    def authorize_diversion(self,question_id,research_mode,justification=""):
        classification=self.diversion(question_id,research_mode)
        if classification==DiversionClassification.UNRELATED and ResearchMode(research_mode)!=ResearchMode.DISCOVERY:raise ResearchError("unrelated VALIDATION/FINAL_TEST branch is prohibited")
        if classification in {DiversionClassification.POSSIBLE_DIVERSION,DiversionClassification.UNRELATED} and not justification:raise ResearchError("diversion justification required")
        if classification in {DiversionClassification.POSSIBLE_DIVERSION,DiversionClassification.UNRELATED}:
            q=self.registry.get_question(question_id);program=q.program_id if q else "UNKNOWN";self.registry.append_journal("POSSIBLE_DIVERSION",program,f"Diversion accepted with justification: {question_id}",question_id=question_id if q else None,metadata={"classification":classification.value,"justification":justification})
        return classification

    def burden(self,candidate_id):
        record=self.get_candidate(candidate_id)
        if not record:raise ResearchError("candidate not found")
        history=self._history(record.experiment_family_id,record.program_id)
        with self.registry.catalog.connection() as con:related_rejected=con.execute("SELECT count(*) FROM candidates WHERE strategy_family_id=? AND status='REJECTED'",(record.strategy_family_id,)).fetchone()[0]
        return {"candidate_id":candidate_id,"number_of_questions_explored":history["questions_explored"],"number_of_hypotheses_considered":history["hypotheses_created"],"number_of_experiments":history["experiments_attempted"],"number_of_parameter_variants":history["parameters_changed"],"number_of_result_driven_modifications":history["result_driven_descendants"],"number_of_datasets_inspected":history["datasets_inspected"],"number_of_related_candidates_rejected":related_rejected}

    def candidate_card(self,candidate_id):
        record=self.get_candidate(candidate_id);burden=self.burden(candidate_id)
        with self.registry.catalog.connection() as con:
            risks=[x[0] for x in con.execute("SELECT json_extract(a.record_json,'$.overall_classification') FROM research_audits a JOIN experiments e ON e.experiment_id=a.experiment_id WHERE json_extract(e.record_json,'$.experiment_family_id')=?",(record.experiment_family_id,))]
            try:evidence=con.execute("SELECT record_json FROM statistical_evidence_reports WHERE candidate_id=? ORDER BY created_at DESC LIMIT 1",(candidate_id,)).fetchone()
            except sqlite3.OperationalError:evidence=None
        rank={"SAFE":0,"WARNING":1,"HIGH_RISK":2,"BLOCKED":3};risk=max(risks,key=lambda x:rank.get(x,0)) if risks else "NOT_AUDITED"
        latest=json.loads(evidence[0]) if evidence else None
        return {"candidate":candidate_id,"family":record.strategy_family_id,"mode":"DISCOVERY" if record.status in {CandidateStatus.EXPLORATORY,CandidateStatus.PROMISING} else "VALIDATION" if record.status==CandidateStatus.VALIDATING else record.status.value,"origin":f"{record.originating_question_id} -> {record.originating_hypothesis_id}","discovery_experiments":burden["number_of_experiments"],"result_driven_modifications":burden["number_of_result_driven_modifications"],"datasets_seen":list(record.discovery_dataset_ids),"audit_risk":risk,"status":record.status.value,"validation":record.validation_status,"latest_evidence":None if not latest else {"report_id":latest["report_id"],"sample_size":latest["sample_profile"]["resolved_trades"],"uncertainty":latest["uncertainty_profile"],"temporal_stability":latest["temporal_profile"],"warnings":latest["warnings"]}}

    def status_report(self,program_id):
        with self.registry.catalog.connection() as con:
            questions=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM research_questions WHERE program_id=?",(program_id,))];experiments=con.execute("SELECT count(*) FROM experiments WHERE program_id=?",(program_id,)).fetchone()[0];result_driven=con.execute("SELECT count(*) FROM experiments WHERE program_id=? AND json_extract(record_json,'$.change_reason_type')='RESULT_DRIVEN'",(program_id,)).fetchone()[0];families=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM strategy_families WHERE program_id=?",(program_id,))];candidates=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM candidates WHERE program_id=?",(program_id,))]
        return {"program":program_id,"root":next((x["question_id"] for x in questions if not x.get("parent_question_id")),None),"open_questions":[x["question_id"] for x in questions if x["status"] in {"OPEN","INVESTIGATING"}],"closed_questions":[x["question_id"] for x in questions if x["status"]=="CLOSED"],"discovery_branches":sum(bool(x.get("parent_question_id")) and x.get("research_mode")=="DISCOVERY" for x in questions),"experiments":experiments,"result_driven_experiments":result_driven,"candidates":len(candidates),"frozen_candidates":sum(x["status"]=="FROZEN" for x in candidates),"families":len(families),"validation_status":"IN_PROGRESS" if any(x["status"]=="VALIDATING" for x in candidates) else "NOT_STARTED","current_next_decision":self.what_next(program_id)}

    def what_next(self,program_id):
        with self.registry.catalog.connection() as con:
            decision=con.execute("SELECT d.record_json FROM decision_nodes d JOIN research_questions q ON q.question_id=d.question_id WHERE q.program_id=? AND json_extract(d.record_json,'$.recorded_outcome') IS NULL ORDER BY d.created_at LIMIT 1",(program_id,)).fetchone()
            experiment=con.execute("SELECT experiment_id FROM experiments WHERE program_id=? AND status='APPROVED' ORDER BY created_at LIMIT 1",(program_id,)).fetchone()
            question=con.execute("SELECT question_id FROM research_questions WHERE program_id=? AND status IN ('OPEN','INVESTIGATING') AND parent_question_id IS NOT NULL ORDER BY created_at LIMIT 1",(program_id,)).fetchone()
        if decision:return {"type":"REGISTERED_DECISION","decision":json.loads(decision[0])}
        if experiment:return {"type":"APPROVED_EXPERIMENT","experiment_id":experiment[0]}
        if question:return {"type":"OPEN_QUESTION","question_id":question[0]}
        return {"type":"HUMAN_RESEARCH_DECISION_REQUIRED","message":"NO PRE-REGISTERED NEXT STEP. Human research decision required."}

    def dead_end(self,question_id):
        q=self.registry.get_question(question_id)
        with self.registry.catalog.connection() as con:children=con.execute("SELECT count(*) FROM research_questions WHERE parent_question_id=? AND status IN ('OPEN','INVESTIGATING')",(question_id,)).fetchone()[0];active=con.execute("SELECT count(*) FROM candidates WHERE json_extract(record_json,'$.originating_question_id')=? AND status NOT IN ('REJECTED','RETIRED')",(question_id,)).fetchone()[0];decisions=con.execute("SELECT count(*) FROM decision_nodes WHERE question_id=? AND json_extract(record_json,'$.recorded_outcome') IS NULL",(question_id,)).fetchone()[0]
        return "RESEARCH_BRANCH_EXHAUSTED" if q and q.status in {QuestionStatus.CLOSED,QuestionStatus.REJECTED,QuestionStatus.ANSWERED} and children==active==decisions==0 else "ACTIVE_OR_UNRESOLVED"

    def decision_trace(self,experiment_id):
        exp=self.registry.get_experiment(experiment_id)
        if not exp:raise ResearchError("experiment not found")
        with self.registry.catalog.connection() as con:results=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM experiment_results WHERE experiment_id=?",(experiment_id,))];verdicts=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM experiment_verdicts WHERE experiment_id=?",(experiment_id,))];next_questions=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM research_questions WHERE parent_question_id=? ORDER BY created_at",(exp.question_id,))]
        q=self.registry.get_question(exp.question_id)
        return {"why_this_test_existed":{"question":q.question_text,"purpose":q.purpose,"evidence_origins":q.evidence_origins},"what_it_tested":exp.model_dump(mode="json"),"what_happened":results,"decision_made":verdicts,"what_question_came_next":next_questions}

    def question_tree(self,program_id):
        with self.registry.catalog.connection() as con:rows=[json.loads(x[0]) for x in con.execute("SELECT record_json FROM research_questions WHERE program_id=? ORDER BY created_at",(program_id,))]
        def node(item):return {"question_id":item["question_id"],"text":item["question_text"],"status":item["status"],"mode":item.get("research_mode"),"children":[node(x) for x in rows if x.get("parent_question_id")==item["question_id"]]}
        return [node(x) for x in rows if not x.get("parent_question_id")]

    def branch(self,question_id):
        root=self.registry.get_question(question_id)
        if not root:raise ResearchError("question not found")
        tree=self.question_tree(root.program_id)
        def find(nodes):
            for item in nodes:
                if item["question_id"]==question_id:return item
                found=find(item["children"])
                if found:return found
        return find(tree)

    def candidate_lineage(self,candidate_id):
        result=[];current=self.get_candidate(candidate_id);seen=set()
        while current:
            if current.candidate_id in seen:raise ResearchError("candidate lineage cycle")
            seen.add(current.candidate_id);result.append(current.model_dump(mode="json"));current=self.get_candidate(current.parent_candidate_id) if current.parent_candidate_id else None
        return list(reversed(result))

    def get_decision(self,decision_id):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM decision_nodes WHERE decision_id=?",(decision_id,)).fetchone()
        return DecisionNode.model_validate_json(row[0]) if row else None
