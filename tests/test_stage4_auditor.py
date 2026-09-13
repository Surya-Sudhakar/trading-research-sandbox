import json
from uuid import uuid4

import pytest

from sandbox.audit.auditor import ResearchAuditor, structural_diff
from sandbox.audit.report import write_audit_report
from sandbox.audit.models import AuditClassification, AuditPhase, DatasetUsageType
from sandbox.execution.models import ExecutionConfig
from sandbox.research.errors import ResearchError
from sandbox.research.models import ExperimentStatus, HypothesisStatus
from test_stage3_registry import create_experiment, setup_registry


def approved(strategy=None,**kwargs):
    reg,p,path=setup_registry(); values={}
    if strategy is not None:values["strategy_spec"]=strategy
    values.update(kwargs); exp=create_experiment(reg,p,**values); reg.approve_experiment(exp.experiment_id)
    return reg,p,path,reg.get_experiment(exp.experiment_id),ResearchAuditor(reg)


def test_explicit_future_bar_entry_is_blocked():
    strategy={"temporal_contract":{"requires_future_bars":True,"confirmation_delay":0,"information_available_at":30,"signal_timestamp":0,"execution_timestamp":0}}
    *_,exp,auditor=approved(strategy)
    audit=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    assert audit.overall_classification==AuditClassification.BLOCKED and "EXPLICIT_LOOKAHEAD" in audit.hard_block_reasons


def test_delayed_confirmation_entry_after_information_is_allowed():
    strategy={"temporal_contract":{"requires_future_bars":True,"confirmation_delay":30,"information_available_at":30,"signal_timestamp":30,"execution_timestamp":31}}
    *_,exp,auditor=approved(strategy)
    assert auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).overall_classification==AuditClassification.SAFE


def test_future_swing_confirmation_backdated_is_blocked():
    strategy={"temporal_contract":{"requires_future_bars":True,"confirmation_delay":30,"information_available_at":30,"signal_timestamp":0,"execution_timestamp":0}}
    *_,exp,auditor=approved(strategy)
    assert auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).overall_classification==AuditClassification.BLOCKED


def test_final_test_previously_used_for_discovery_blocks_independence_claim():
    reg,p,_,exp,auditor=approved({"claims_independent_validation":True})
    auditor.record_dataset_usage(p.dataset_id,"TEST_PROGRAM","old",DatasetUsageType.DISCOVERY,"rule development")
    auditor.record_dataset_usage(p.dataset_id,"TEST_PROGRAM",exp.experiment_id,DatasetUsageType.FINAL_TEST,"claimed final test")
    audit=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    assert audit.overall_classification==AuditClassification.BLOCKED and "FINAL_TEST_CONTAMINATION" in audit.hard_block_reasons


def test_discovery_reuse_labeled_discovery_is_not_falsely_blocked():
    reg,p,_,exp,auditor=approved({"family":"synthetic"})
    auditor.record_dataset_usage(p.dataset_id,"TEST_PROGRAM",exp.experiment_id,DatasetUsageType.DISCOVERY,"continued discovery")
    assert auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).overall_classification==AuditClassification.SAFE


def child_sequence(reason_type="RESULT_DRIVEN",count=2):
    reg,p,_=setup_registry(); family="FAMILY-TEST"; parent=create_experiment(reg,p,experiment_family_id=family,strategy_spec={"threshold":1.0});reg.approve_experiment(parent.experiment_id)
    current=parent
    for index in range(1,count):
        current=create_experiment(reg,p,experiment_family_id=family,parent_experiment_id=current.experiment_id,change_reason="changed threshold",change_reason_type=reason_type,change_reason_text="synthetic rationale",fields_changed_from_parent=["strategy_spec.threshold"],strategy_spec={"threshold":1.0+index/10});reg.approve_experiment(current.experiment_id)
    return reg,p,current,ResearchAuditor(reg)


def test_result_driven_child_same_family_and_post_hoc_high_risk():
    reg,_,child,auditor=child_sequence()
    audit=auditor.audit(child.experiment_id,AuditPhase.PRE_RUN)
    assert child.experiment_family_id=="FAMILY-TEST" and any(x.code=="RESULT_DRIVEN_DESCENDANT" for x in audit.findings)
    assert audit.overall_classification==AuditClassification.HIGH_RISK


def test_theory_driven_child_is_distinguishable():
    _,_,child,auditor=child_sequence("THEORY_DRIVEN")
    audit=auditor.audit(child.experiment_id,AuditPhase.PRE_RUN)
    assert not any(x.code=="RESULT_DRIVEN_DESCENDANT" for x in audit.findings)


def test_child_change_reason_is_required():
    reg,p,_=setup_registry(); parent=create_experiment(reg,p);reg.approve_experiment(parent.experiment_id);child=create_experiment(reg,p,parent_experiment_id=parent.experiment_id,strategy_spec={"x":2});reg.approve_experiment(child.experiment_id)
    audit=ResearchAuditor(reg).audit(child.experiment_id,AuditPhase.PRE_RUN)
    assert any(x.code=="MISSING_CHANGE_REASON" for x in audit.findings)


@pytest.mark.parametrize("count,classification",[(3,AuditClassification.WARNING),(10,AuditClassification.HIGH_RISK)])
def test_repeated_parameter_values_raise_configured_risk(count,classification):
    reg,p,last,auditor=child_sequence("THEORY_DRIVEN",count)
    audit=auditor.audit(last.experiment_id,AuditPhase.PRE_RUN)
    assert any(x.code=="REPEATED_PARAMETER_SEARCH" for x in audit.findings)
    assert audit.overall_classification==classification
    assert "best" not in " ".join(x.explanation.lower() for x in audit.findings)


def test_near_duplicate_and_difference_report_are_exact():
    left={"threshold":1.50,"fixed":True};right={"threshold":1.51,"fixed":True}
    assert structural_diff(left,right)==[{"field":"threshold","from":1.5,"to":1.51}]
    reg,p,_=setup_registry();first=create_experiment(reg,p,experiment_family_id="F",strategy_spec=left);reg.approve_experiment(first.experiment_id);second=create_experiment(reg,p,experiment_family_id="F",strategy_spec=right);reg.approve_experiment(second.experiment_id)
    finding=next(x for x in ResearchAuditor(reg).audit(second.experiment_id,AuditPhase.PRE_RUN).findings if x.code=="NEAR_DUPLICATE_EXPERIMENT")
    assert finding.evidence["near_duplicates"][0]["changed_fields"][0]["field"]=="threshold"


@pytest.mark.parametrize("count,classification",[(5,AuditClassification.WARNING),(10,AuditClassification.HIGH_RISK)])
def test_hypothesis_proliferation_thresholds(count,classification):
    reg,p,_=setup_registry()
    for index in range(2,count+1):
        hid=f"H{index}";reg.create_hypothesis(hid,"TEST_PROGRAM","TEST_Q0",f"Synthetic hypothesis {index}","test","observable","mismatch");reg.set_hypothesis_status(hid,HypothesisStatus.APPROVED)
    exp=create_experiment(reg,p);reg.approve_experiment(exp.experiment_id);audit=ResearchAuditor(reg).audit(exp.experiment_id,AuditPhase.PRE_RUN)
    assert audit.overall_classification==classification and any(x.code=="HYPOTHESIS_PROLIFERATION" for x in audit.findings)


def test_simple_strategy_has_low_complexity_profile():
    *_,exp,auditor=approved({"x":1});profile=auditor.degrees_of_freedom(exp)
    assert profile.number_of_strategy_parameters==1
    assert not any(x.code=="HIGH_SPEC_COMPLEXITY" for x in auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).findings)


def test_many_filters_and_thresholds_increase_complexity():
    strategy={"parameters":{f"threshold_{i}":i for i in range(16)},"filters":{f"filter_{i}":True for i in range(5)}}
    *_,exp,auditor=approved(strategy);audit=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    profile=auditor.degrees_of_freedom(exp)
    assert profile.number_of_strategy_parameters==21 and profile.number_of_threshold_parameters==16
    assert any(x.code=="HIGH_SPEC_COMPLEXITY" for x in audit.findings)


def test_pre_run_is_outcome_blind_and_profit_cannot_reduce_risk():
    reg1,p1,_,e1,a1=approved({"x":1},evaluation_spec={"fake_win_rate":0.1,"fake_net_r":-40})
    reg2,p2,_,e2,a2=approved({"x":1},evaluation_spec={"fake_win_rate":0.9,"fake_net_r":100})
    first=a1.audit(e1.experiment_id,AuditPhase.PRE_RUN);second=a2.audit(e2.experiment_id,AuditPhase.PRE_RUN)
    assert first.overall_classification==second.overall_classification and first.risk_score==second.risk_score


def test_safe_and_warning_execute_but_high_requires_override():
    reg,_,_,safe,auditor=approved({"x":1});assert reg.transition_experiment(safe.experiment_id,ExperimentStatus.RUNNING).status==ExperimentStatus.RUNNING
    reg2,p2,child,auditor2=child_sequence("RESULT_DRIVEN");audit=auditor2.audit(child.experiment_id,AuditPhase.PRE_RUN)
    with pytest.raises(ResearchError,match="requires explicit override"):reg2.transition_experiment(child.experiment_id,ExperimentStatus.RUNNING)
    auditor2.create_override(audit.audit_id,"documented discovery continuation","tester")
    assert reg2.transition_experiment(child.experiment_id,ExperimentStatus.RUNNING).status==ExperimentStatus.RUNNING


def test_blocked_cannot_execute_or_be_overridden():
    *_,exp,auditor=approved({"temporal_contract":{"requires_future_bars":True,"confirmation_delay":0}});audit=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    with pytest.raises(ResearchError,match="BLOCKED"):auditor.create_override(audit.audit_id,"no","tester")
    with pytest.raises(ResearchError,match="BLOCKED"):auditor.can_execute(exp.experiment_id)


def test_invalid_dataset_and_corrupt_journal_are_hard_blocks():
    reg,p,path,exp,auditor=approved({"x":1});path.write_bytes(b"tampered")
    assert auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).overall_classification==AuditClassification.BLOCKED
    reg2,_,_,exp2,auditor2=approved({"x":1})
    with reg2.catalog.connection() as con:con.execute("UPDATE research_journal SET message='tampered' WHERE sequence=1")
    assert auditor2.audit(exp2.experiment_id,AuditPhase.PRE_RUN).overall_classification==AuditClassification.BLOCKED


def test_approved_spec_mutation_is_hard_blocked():
    reg,_,_,exp,auditor=approved({"x":1});data=exp.model_dump(mode="json");data["strategy_spec"]={"x":999}
    with reg.catalog.connection() as con:con.execute("UPDATE experiments SET record_json=? WHERE experiment_id=?",(json.dumps(data),exp.experiment_id))
    assert "APPROVED_SPEC_MUTATION" in auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN).hard_block_reasons


def test_identical_history_and_config_reuses_identical_audit():
    *_,exp,auditor=approved({"x":1});first=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN);second=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    assert first==second and first.config_fingerprint==auditor.config_fingerprint and first.auditor_version


def test_json_and_markdown_audit_reports_are_generated():
    reg,_,_,exp,auditor=approved({"x":1});audit=auditor.audit(exp.experiment_id,AuditPhase.PRE_RUN)
    paths=write_audit_report(audit,reg.results_root)
    assert all(path.exists() for path in paths) and audit.audit_id in paths[0].read_text(encoding="utf-8") and "RESEARCH AUDIT" in paths[1].read_text(encoding="utf-8")
