from __future__ import annotations

import json,sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sandbox import EXECUTION_ENGINE_VERSION, RESEARCH_AUDITOR_VERSION
from sandbox.audit.models import AuditClassification, AuditFinding, AuditOverride, AuditPhase, DatasetUsageType, DegreesOfFreedomProfile, FindingCategory, FindingSeverity, ResearchAudit
from sandbox.execution.models import ExecutionConfig
from sandbox.research.canonical import canonical_json, experiment_fingerprint, sha256_canonical
from sandbox.research.errors import DatasetVerificationError, JournalIntegrityError, ResearchError
from sandbox.research.models import ExperimentStatus, HypothesisStatus, now_utc
from sandbox.research.registry import ResearchRegistry


AUDIT_DDL="""
CREATE TABLE IF NOT EXISTS dataset_usage (usage_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,program_id TEXT NOT NULL,experiment_id TEXT NOT NULL,usage_type TEXT NOT NULL,first_accessed_at TEXT NOT NULL,access_reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS research_audits (audit_id TEXT PRIMARY KEY,experiment_id TEXT NOT NULL,phase TEXT NOT NULL,auditor_version TEXT NOT NULL,config_fingerprint TEXT NOT NULL,record_json TEXT NOT NULL,audited_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_overrides (override_id TEXT PRIMARY KEY,audit_id TEXT NOT NULL,experiment_id TEXT NOT NULL,reason TEXT NOT NULL,approved_by TEXT NOT NULL,approved_at TEXT NOT NULL);
"""


def _flatten(value,prefix=""):
    result={}
    if isinstance(value,dict):
        for key in sorted(value): result.update(_flatten(value[key],f"{prefix}.{key}" if prefix else key))
    elif isinstance(value,list):
        for index,item in enumerate(value): result.update(_flatten(item,f"{prefix}[{index}]"))
    else: result[prefix]=value
    return result


def structural_diff(left,right):
    a,b=_flatten(left),_flatten(right); result=[]
    for key in sorted(set(a)|set(b)):
        if a.get(key)!=b.get(key): result.append({"field":key,"from":a.get(key),"to":b.get(key)})
    return result


class ResearchAuditor:
    def __init__(self,registry:ResearchRegistry,config_path:Path|None=None):
        self.registry=registry; self.config_path=config_path or registry.project_root/"config"/"research_audit.json"
        self.config=json.loads(self.config_path.read_text(encoding="utf-8")); self.config_fingerprint=sha256_canonical(self.config)
        with registry.catalog.connection() as con: con.executescript(AUDIT_DDL)

    def record_dataset_usage(self,dataset_id,program_id,experiment_id,usage_type:DatasetUsageType,access_reason):
        with self.registry.catalog.connection() as con:
            con.execute("INSERT INTO dataset_usage VALUES (?,?,?,?,?,?,?)",(str(uuid4()),dataset_id,program_id,experiment_id,usage_type.value,now_utc().isoformat(),access_reason))

    def degrees_of_freedom(self,experiment)->DegreesOfFreedomProfile:
        flat=_flatten(experiment.strategy_spec); keys=[x.lower() for x in flat]
        family=self._family(experiment.experiment_family_id)
        return DegreesOfFreedomProfile(
            number_of_strategy_parameters=len(flat),number_of_threshold_parameters=sum("threshold" in k for k in keys),
            number_of_boolean_filters=sum(isinstance(v,bool) and ("filter" in k or "enabled" in k) for k,v in zip(keys,flat.values())),
            number_of_optional_conditions=sum("optional" in k or "condition" in k for k in keys),number_of_time_windows=sum("window" in k or "time" in k for k in keys),
            number_of_entry_variants=sum("entry" in k and isinstance(v,list) for k,v in flat.items()),number_of_exit_variants=sum("exit" in k and isinstance(v,list) for k,v in flat.items()),
            number_of_symbols_selectable=sum("symbol" in k and isinstance(v,list) for k,v in flat.items()),number_of_sessions_selectable=sum("session" in k and isinstance(v,list) for k,v in flat.items()),
            number_of_prior_experiments_in_family=max(0,len(family)-1),number_of_result_driven_descendants=sum(x.change_reason_type=="RESULT_DRIVEN" for x in family),number_of_rejected_variants=sum(x.status in {ExperimentStatus.FAILED,ExperimentStatus.CANCELLED} for x in family))

    def _family(self,family_id):
        if not family_id:return []
        with self.registry.catalog.connection() as con: rows=con.execute("SELECT record_json FROM experiments ORDER BY created_at").fetchall()
        from sandbox.research.models import ExperimentSpec
        return [ExperimentSpec.model_validate_json(x[0]) for x in rows if ExperimentSpec.model_validate_json(x[0]).experiment_family_id==family_id]

    def _findings(self,experiment,phase):
        raw=[]; hard=[]
        def add(category,severity,code,title,explanation,evidence,remediation,block=False):
            raw.append((category,severity,code,title,explanation,evidence,remediation)); hard.append(code) if block else None
        # Frozen fingerprint and prerequisite integrity.
        recalculated=experiment_fingerprint(question_id=experiment.question_id,hypothesis_id=experiment.hypothesis_id,dataset_ids=experiment.dataset_ids,strategy_spec=experiment.strategy_spec,execution_config=experiment.execution_config,evaluation_spec=experiment.evaluation_spec,code_version=experiment.code_version)
        if recalculated!=experiment.experiment_fingerprint:add(FindingCategory.REPRODUCIBILITY,FindingSeverity.CRITICAL,"APPROVED_SPEC_MUTATION","Approved specification fingerprint mismatch","Stored scientific fields no longer match the approved fingerprint.",{},"Create a new experiment from the legitimate specification.",True)
        if experiment.status==ExperimentStatus.DRAFT:add(FindingCategory.REPRODUCIBILITY,FindingSeverity.CRITICAL,"MISSING_PRE_REGISTRATION","Experiment is not approved","Pre-run execution requires an approved pre-registration.",{},"Complete and approve the experiment.",True)
        if experiment.strategy_spec.get("strategy_id"):
            required={"strategy_id","strategy_version","strategy_code_fingerprint","parameter_fingerprint","signal_specification"};missing=sorted(required-set(experiment.strategy_spec))
            if missing:add(FindingCategory.REPRODUCIBILITY,FindingSeverity.CRITICAL,"INCOMPLETE_STRATEGY_PLUGIN_IDENTITY","Strategy plugin identity is incomplete","Plugin-based experiments must freeze code, parameters, and signal specification before execution.",{"missing":missing},"Register the complete plugin identity in strategy_spec and create a new experiment.",True)
        try: ExecutionConfig.model_validate(experiment.execution_config)
        except Exception as exc:add(FindingCategory.EXECUTION,FindingSeverity.CRITICAL,"INCOMPATIBLE_EXECUTION_CONFIG","Execution configuration is incompatible",str(exc),{},"Use a configuration supported by the installed engine.",True)
        for dataset_id in experiment.dataset_ids:
            try:self.registry.verify_dataset(dataset_id)
            except DatasetVerificationError as exc:add(FindingCategory.DATA_INTEGRITY,FindingSeverity.CRITICAL,"INVALID_DATASET_CHECKSUM","Dataset integrity verification failed",str(exc),{"dataset_id":dataset_id},"Restore verified immutable data or register a new dataset.",True)
        try:self.registry.verify_journal_integrity()
        except JournalIntegrityError as exc:add(FindingCategory.REPRODUCIBILITY,FindingSeverity.CRITICAL,"CORRUPTED_RESEARCH_JOURNAL","Research journal integrity failed",str(exc),{},"Investigate and restore the journal before running.",True)
        # Temporal contract.
        temporal=experiment.strategy_spec.get("temporal_contract",{});flat_strategy=_flatten(experiment.strategy_spec);future_markers=("next_candle","future_","right_side","future_daily_high","future_atr","future_session");explicit_future_fields=sorted(k for k in flat_strategy if any(marker in k.lower() for marker in future_markers));requires=bool(temporal.get("requires_future_bars",False) or explicit_future_fields); delay=float(temporal.get("confirmation_delay",0) or 0)
        info=temporal.get("information_available_at"); signal=temporal.get("signal_timestamp"); execution=temporal.get("execution_timestamp")
        chronology_bad=False
        if requires and delay<=0: chronology_bad=True
        try:
            if info is not None and execution is not None and float(execution)<float(info):chronology_bad=True
            if requires and signal is not None and info is not None and float(signal)<float(info) and (execution is None or float(execution)<float(info)):chronology_bad=True
        except (TypeError,ValueError): add(FindingCategory.LOOKAHEAD,FindingSeverity.WARNING,"TEMPORAL_MANUAL_REVIEW","Temporal metadata requires manual review","Temporal values could not be compared deterministically.",temporal,"Use numeric offsets or ISO-compatible explicit timing.")
        if chronology_bad:add(FindingCategory.LOOKAHEAD,FindingSeverity.CRITICAL,"EXPLICIT_LOOKAHEAD","Future information is used before it is available","The declared confirmation/future-bar dependency occurs after the signal or execution.",{**temporal,"explicit_future_fields":explicit_future_fields},"Delay the signal and execution until all required information is available.",True)
        # Usage leakage.
        claims=bool(experiment.strategy_spec.get("claims_independent_validation",False))
        with self.registry.catalog.connection() as con: usages=[dict(x) for x in con.execute("SELECT * FROM dataset_usage WHERE dataset_id IN (%s) ORDER BY first_accessed_at"%",".join("?"*len(experiment.dataset_ids)),tuple(experiment.dataset_ids))] if experiment.dataset_ids else []
        by_dataset={d:{u["usage_type"] for u in usages if u["dataset_id"]==d} for d in experiment.dataset_ids}
        for dataset_id,types in by_dataset.items():
            if claims and "FINAL_TEST" in types and "DISCOVERY" in types:add(FindingCategory.DATA_LEAKAGE,FindingSeverity.CRITICAL,"FINAL_TEST_CONTAMINATION","Claimed final test was previously used for discovery","Usage history conflicts with the independence claim.",{"dataset_id":dataset_id,"usage_types":sorted(types)},"Use genuinely unseen data and a new experiment.",True)
            elif len(types)>1:add(FindingCategory.DATA_LEAKAGE,FindingSeverity.HIGH,"DATASET_LABEL_CONFLICT","Dataset has conflicting usage labels","The same dataset has multiple research roles.",{"dataset_id":dataset_id,"usage_types":sorted(types)},"Clarify and preserve the non-independent role.")
        # Ancestry.
        if experiment.parent_experiment_id:
            parent=self.registry.get_experiment(experiment.parent_experiment_id); diff=structural_diff(parent.strategy_spec,experiment.strategy_spec) if parent else []
            if not experiment.change_reason_type or not experiment.change_reason_text:add(FindingCategory.ANCESTRY,FindingSeverity.HIGH,"MISSING_CHANGE_REASON","Child experiment lacks structured change rationale","Ancestry cannot distinguish theory, repair, and result-driven adaptation.",{"parent":experiment.parent_experiment_id,"changed_fields":diff},"Record change_reason_type and change_reason_text.")
            if experiment.change_reason_type=="RESULT_DRIVEN":add(FindingCategory.POST_HOC_CHANGE,FindingSeverity.HIGH,"RESULT_DRIVEN_DESCENDANT","Child adapts after observing a prior result","This descendant remains in the same discovery family and is not independent confirmation.",{"parent":experiment.parent_experiment_id,"family":experiment.experiment_family_id,"changed_fields":diff},"Label as discovery and validate later on unseen data.")
        # Family search and near duplicates.
        family=self._family(experiment.experiment_family_id); flat_current=_flatten(experiment.strategy_spec); searches=[]
        for key in sorted(flat_current):
            values=[]; ids=[]
            for member in family:
                flat=_flatten(member.strategy_spec)
                if key in flat and flat[key] not in values:values.append(flat[key]);ids.append(member.experiment_id)
            if len(values)>=self.config["parameter_search_warning_count"]:searches.append({"parameter_name":key,"values_tested":values,"number_of_values":len(values),"experiments":ids})
        if searches:
            maximum=max(x["number_of_values"] for x in searches); sev=FindingSeverity.HIGH if maximum>=self.config["parameter_search_high_risk_count"] else FindingSeverity.WARNING
            add(FindingCategory.PARAMETER_SEARCH,sev,"REPEATED_PARAMETER_SEARCH","Repeated values of the same parameter were tested","Related experiments vary the same structural parameter. No value is recommended.",{"searches":searches},"Treat attempts as one discovery family and use independent validation.")
            self.registry.append_journal("PARAMETER_SEARCH_WARNING",experiment.program_id,f"Parameter search detected for {experiment.experiment_id}",question_id=experiment.question_id,hypothesis_id=experiment.hypothesis_id,experiment_id=experiment.experiment_id)
        candidates=[]
        for member in family:
            if member.experiment_id==experiment.experiment_id:continue
            diff=structural_diff(member.strategy_spec,experiment.strategy_spec)
            if 0<len(diff)<=self.config["near_duplicate_max_changed_fields"]:candidates.append({"experiment_id":member.experiment_id,"changed_fields":diff})
        if candidates:add(FindingCategory.DUPLICATION,FindingSeverity.WARNING,"NEAR_DUPLICATE_EXPERIMENT","Experiment is structurally near-duplicate","Only a small number of fields differ from related experiments.",{"near_duplicates":candidates},"Declare sensitivity-analysis intent and count all attempts.")
        # Hypothesis and falsifiability.
        with self.registry.catalog.connection() as con: hypothesis_count=con.execute("SELECT count(*) FROM hypotheses WHERE question_id=? AND status<>'PROPOSED'",(experiment.question_id,)).fetchone()[0]
        if hypothesis_count>=self.config["hypothesis_warning_count"]:
            sev=FindingSeverity.HIGH if hypothesis_count>=self.config["hypothesis_high_risk_count"] else FindingSeverity.WARNING
            add(FindingCategory.HYPOTHESIS_PROLIFERATION,sev,"HYPOTHESIS_PROLIFERATION","Many hypotheses have been tested under one question","This is a risk signal, not proof of overfitting.",{"tested_hypotheses":hypothesis_count},"Report the full hypothesis count and use independent confirmation.")
        hypothesis=self.registry.get_hypothesis(experiment.hypothesis_id); falsification=(hypothesis.falsification_condition if hypothesis else "").strip().lower()
        if not falsification or falsification in {"unprofitable","strategy is unprofitable","none","n/a"}:add(FindingCategory.FALSIFIABILITY,FindingSeverity.WARNING,"MANUAL_REVIEW_REQUIRED","Falsification condition is weak or absent","Deterministic text checks cannot establish a meaningful falsification rule.",{"falsification_condition":falsification},"State a specific observable condition that would reject the mechanism.")
        # Transparent flexibility/complexity.
        profile=self.degrees_of_freedom(experiment)
        add(FindingCategory.DEGREES_OF_FREEDOM,FindingSeverity.INFO,"DEGREES_OF_FREEDOM_PROFILE","Observable researcher flexibility profile","Counts are descriptive rather than a statistical degrees-of-freedom estimate.",profile.model_dump(),"Preserve the profile with the audit.")
        if profile.number_of_strategy_parameters>=self.config["complexity_parameter_warning"]:add(FindingCategory.COMPLEXITY,FindingSeverity.WARNING,"HIGH_SPEC_COMPLEXITY","Strategy specification has many configurable leaves","High flexibility can be risky relative to limited observations.",profile.model_dump(),"Pre-register components and compare flexibility with sample size.")
        if phase==AuditPhase.POST_RUN:
            with self.registry.catalog.connection() as con: result=con.execute("SELECT record_json FROM experiment_results WHERE experiment_id=? ORDER BY created_at DESC LIMIT 1",(experiment.experiment_id,)).fetchone()
            if result:
                metrics=json.loads(result[0])["summary_metrics"]; n=int(metrics.get("resolved_trades",0)); threshold=self.config["very_small_sample"] if n<self.config["very_small_sample"] else self.config["small_sample"]
                if n<self.config["small_sample"]:
                    code="VERY_SMALL_SAMPLE" if n<self.config["very_small_sample"] else "SMALL_SAMPLE"; sev=FindingSeverity.HIGH if n<self.config["very_small_sample"] and profile.number_of_strategy_parameters/max(n,1)>=self.config["complexity_trade_ratio_warning"] else FindingSeverity.WARNING
                    add(FindingCategory.SAMPLE_USAGE,sev,code,"Limited resolved-trade sample","Sample-size labels do not establish statistical robustness.",{"resolved_trades":n,"strategy_parameters":profile.number_of_strategy_parameters},"Collect independent observations without adapting rules to these results.")
            try:
                with self.registry.catalog.connection() as con:evidence=con.execute("SELECT record_json FROM statistical_evidence_reports WHERE experiment_id=? ORDER BY created_at DESC LIMIT 1",(experiment.experiment_id,)).fetchone()
                if evidence:
                    report=json.loads(evidence[0]);methodological={"PARAMETER_CLIFF","HIGH_DISCOVERY_SEARCH_BURDEN","MULTIPLE_SUBGROUP_INSPECTION","SERIAL_DEPENDENCE","CROSS_SYMBOL_DEPENDENCE_POSSIBLE"};found=sorted(methodological.intersection(report.get("warnings",[])))
                    if found:add(FindingCategory.SAMPLE_USAGE,FindingSeverity.WARNING,"STATISTICAL_ROBUSTNESS_CONTEXT","Statistical evidence includes robustness warnings","Outcome-derived Stage 7 warnings are post-run context only and never improve PRE_RUN scoring.",{"report_id":report["report_id"],"warnings":found},"Preserve these warnings with interpretation and independent validation.")
            except sqlite3.OperationalError:pass
        try:
            with self.registry.catalog.connection() as con:
                access_findings=[dict(x) for x in con.execute("SELECT c.* FROM contamination_records c JOIN candidates ca ON ca.candidate_id=c.candidate_id WHERE ca.experiment_family_id=? AND c.contamination_type IN ('UNAUTHORIZED_ACCESS_ATTEMPT','RESULT_DRIVEN_REUSE','FINAL_TEST_EXPOSURE')",(experiment.experiment_family_id,))]
            for evidence in access_findings:
                severity=FindingSeverity.CRITICAL if evidence["contamination_type"]=="UNAUTHORIZED_ACCESS_ATTEMPT" else FindingSeverity.HIGH
                add(FindingCategory.DATA_LEAKAGE,severity,"STAGE6_CONTAMINATION_EVIDENCE","Partition access contamination is recorded","Stage 6 recorded access evidence relevant to this experiment family.",{"partition_id":evidence["partition_id"],"type":evidence["contamination_type"],"reason":evidence["reason"]},"Use an uncontaminated partition and preserve the lineage record.",severity==FindingSeverity.CRITICAL)
        except Exception as exc:
            if not isinstance(exc,sqlite3.OperationalError):raise
        return raw,hard

    def audit(self,experiment_id:str,phase:AuditPhase)->ResearchAudit:
        experiment=self.registry.get_experiment(experiment_id)
        if not experiment:raise ResearchError("experiment not found")
        raw,hard=self._findings(experiment,phase)
        basis={"experiment_fingerprint":experiment.experiment_fingerprint,"phase":phase.value,"auditor_version":RESEARCH_AUDITOR_VERSION,"config_fingerprint":self.config_fingerprint,"findings":[{"code":x[2],"evidence":x[5]} for x in raw]}
        audit_id="AUD-"+sha256_canonical(basis)[:24]
        with self.registry.catalog.connection() as con:
            existing=con.execute("SELECT record_json FROM research_audits WHERE audit_id=?",(audit_id,)).fetchone()
        if existing:return ResearchAudit.model_validate_json(existing[0])
        timestamp=now_utc(); findings=[]
        for category,severity,code,title,explanation,evidence,remediation in raw:
            fid="FND-"+sha256_canonical({"audit_id":audit_id,"code":code,"evidence":evidence})[:24]
            findings.append(AuditFinding(finding_id=fid,audit_id=audit_id,experiment_id=experiment_id,category=category,severity=severity,code=code,title=title,explanation=explanation,evidence=evidence,remediation=remediation,created_at=timestamp,auditor_version=RESEARCH_AUDITOR_VERSION))
        warning=sum(x.severity==FindingSeverity.WARNING for x in findings); high=sum(x.severity==FindingSeverity.HIGH for x in findings); critical=sum(x.severity==FindingSeverity.CRITICAL for x in findings)
        score=min(100,warning*self.config["risk_points"]["warning"]+high*self.config["risk_points"]["high"])
        classification=AuditClassification.BLOCKED if hard else AuditClassification.HIGH_RISK if high or score>=50 else AuditClassification.WARNING if warning else AuditClassification.SAFE
        record=ResearchAudit(audit_id=audit_id,experiment_id=experiment_id,experiment_fingerprint=experiment.experiment_fingerprint,phase=phase,auditor_version=RESEARCH_AUDITOR_VERSION,config_fingerprint=self.config_fingerprint,audited_at=timestamp,overall_classification=classification,findings=tuple(findings),risk_score=score,hard_block_reasons=tuple(hard),warning_count=warning,high_risk_count=high,critical_count=critical)
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO research_audits VALUES (?,?,?,?,?,?,?)",(audit_id,experiment_id,phase.value,RESEARCH_AUDITOR_VERSION,self.config_fingerprint,record.model_dump_json(),timestamp.isoformat()))
        event="PRE_RUN_AUDIT_COMPLETED" if phase==AuditPhase.PRE_RUN else "POST_RUN_AUDIT_COMPLETED"; self.registry.append_journal(event,experiment.program_id,f"{phase.value} audit {audit_id}: {classification.value}",question_id=experiment.question_id,hypothesis_id=experiment.hypothesis_id,experiment_id=experiment_id,metadata={"audit_id":audit_id,"classification":classification.value,"risk_score":score})
        if classification==AuditClassification.BLOCKED:self.registry.append_journal("EXPERIMENT_BLOCKED_BY_AUDIT",experiment.program_id,f"Experiment {experiment_id} blocked",experiment_id=experiment_id,metadata={"audit_id":audit_id,"reasons":hard})
        return record

    def create_override(self,audit_id,reason,approved_by)->AuditOverride:
        audit=self.get_audit(audit_id)
        if not audit:raise ResearchError("audit not found")
        if audit.overall_classification==AuditClassification.BLOCKED:raise ResearchError("BLOCKED audits cannot be overridden")
        if audit.overall_classification!=AuditClassification.HIGH_RISK:raise ResearchError("override is only valid for HIGH_RISK audits")
        record=AuditOverride(override_id="OVR-"+uuid4().hex,audit_id=audit_id,experiment_id=audit.experiment_id,reason=reason,approved_by=approved_by,approved_at=now_utc())
        with self.registry.catalog.connection() as con:con.execute("INSERT INTO audit_overrides VALUES (?,?,?,?,?,?)",(record.override_id,audit_id,audit.experiment_id,reason,approved_by,record.approved_at.isoformat()))
        experiment=self.registry.get_experiment(audit.experiment_id);self.registry.append_journal("HIGH_RISK_OVERRIDE_CREATED",experiment.program_id,f"Override {record.override_id} created",experiment_id=audit.experiment_id,metadata={"audit_id":audit_id,"reason":reason,"approved_by":approved_by})
        return record

    def get_audit(self,audit_id):
        with self.registry.catalog.connection() as con:row=con.execute("SELECT record_json FROM research_audits WHERE audit_id=?",(audit_id,)).fetchone()
        return ResearchAudit.model_validate_json(row[0]) if row else None

    def audit_history(self,experiment_id):
        with self.registry.catalog.connection() as con:rows=con.execute("SELECT record_json FROM research_audits WHERE experiment_id=? ORDER BY audited_at",(experiment_id,)).fetchall()
        return [ResearchAudit.model_validate_json(x[0]) for x in rows]

    def can_execute(self,experiment_id):
        audit=self.audit(experiment_id,AuditPhase.PRE_RUN)
        if audit.overall_classification==AuditClassification.BLOCKED:raise ResearchError("experiment BLOCKED by research audit")
        if audit.overall_classification==AuditClassification.HIGH_RISK:
            with self.registry.catalog.connection() as con:override=con.execute("SELECT 1 FROM audit_overrides WHERE audit_id=?",(audit.audit_id,)).fetchone()
            if not override:raise ResearchError("HIGH_RISK experiment requires explicit override")
        return audit

    def parameter_history(self,family_id):
        family=self._family(family_id); paths={}
        for member in family:
            for key,value in _flatten(member.strategy_spec).items():paths.setdefault(key,[]).append({"experiment_id":member.experiment_id,"value":value})
        return {key:values for key,values in paths.items() if len({canonical_json(x['value']) for x in values})>1}
