from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from sandbox.audit.auditor import ResearchAuditor
from sandbox.audit.models import AuditPhase,DatasetUsageType
from sandbox.control.models import CandidateStatus,DiversionClassification,EvidenceOrigin,ResearchMode,StrategyFamilyStatus
from sandbox.control.service import ResearchControl
from sandbox.execution.models import ExecutionConfig
from sandbox.provenance import Provenance
from sandbox.research.errors import ImmutableRecordError,PreRegistrationError,ResearchError
from sandbox.research.models import HypothesisStatus,QuestionStatus
from test_stage3_registry import setup_registry


MOTIVATION={"question_answered":"TEST_Q1","why_it_matters":"mechanism test","evidence_or_theory":"synthetic discovery evidence","learn_from_pass":"retain candidate","learn_from_fail":"reject candidate","outcome_actions":{"PASS":"FREEZE_CANDIDATE","FAIL":"CLOSE_HYPOTHESIS"}}


def setup_control():
    reg,p,_=setup_registry();control=ResearchControl(reg)
    q=control.create_question(question_id="TEST_Q1",program_id="TEST_PROGRAM",question_text="Does the synthetic family merit investigation?",parent_question_id="TEST_Q0",purpose="Test navigation",relevance_to_parent="Direct synthetic branch",evidence_origins=[EvidenceOrigin.DISCOVERY_DATA],research_mode=ResearchMode.DISCOVERY,rationale="fixture")
    h=reg.create_hypothesis("DISC_H1","TEST_PROGRAM",q.question_id,"Synthetic discovery hypothesis","fixture","known outcome","mismatch",research_mode=ResearchMode.DISCOVERY.value,evidence_origins=[EvidenceOrigin.DISCOVERY_DATA.value]);reg.set_hypothesis_status(h.hypothesis_id,HypothesisStatus.APPROVED)
    family=control.create_family("TEST_PROGRAM","Synthetic Family","fixture",strategy_family_id="FAMILY-A")
    return reg,control,p,q,h,family


def experiment(reg,p,q,h,index=1,family="EXP-FAMILY-A",reason_type=None,strategy=None,motivation=MOTIVATION):
    exp=reg.create_experiment(program_id="TEST_PROGRAM",question_id=q.question_id,hypothesis_id=h.hypothesis_id,title=f"Synthetic {index}",description="fixture",dataset_ids=[p.dataset_id],strategy_spec=strategy or {"threshold":index},execution_config=ExecutionConfig().model_dump(mode="json"),evaluation_spec={"minimum_win_rate":0.5},code_version={"method":"TEST","revision":"x","working_tree":"CLEAN"},experiment_family_id=family,change_reason_type=reason_type,change_reason_text="synthetic result" if reason_type else None,research_mode=ResearchMode.DISCOVERY.value,motivation_spec=motivation)
    reg.approve_experiment(exp.experiment_id);return reg.get_experiment(exp.experiment_id)


def second_dataset(reg,tag="validation"):
    path=Path("tests/runtime/stage5")/(uuid4().hex+".parquet");path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tag.encode())
    p=Provenance.create(path,"Test","Synthetic",tag,"M1",pd.Timestamp("2024-01-01",tz="UTC").to_pydatetime(),pd.Timestamp("2024-01-02",tz="UTC").to_pydatetime(),1);reg.catalog.add_dataset(p);return p


def candidate_fixture(experiments=2):
    reg,control,p,q,h,family=setup_control();items=[]
    for i in range(1,experiments+1):items.append(experiment(reg,p,q,h,i,reason_type="RESULT_DRIVEN" if i>1 else None))
    ResearchAuditor(reg).audit(items[-1].experiment_id,AuditPhase.PRE_RUN)
    cand=control.create_candidate("TEST_PROGRAM",q.question_id,h.hypothesis_id,"EXP-FAMILY-A",family.strategy_family_id,{"threshold":experiments},ExecutionConfig().model_dump(mode="json"),{"minimum_win_rate":0.5},["X"],["M1"],{}, {"revision":"x"},[p.dataset_id])
    return reg,control,p,q,h,family,items,cand


def test_discovery_freedom_result_driven_and_winner_loser_origin_allowed():
    reg,control,p,q,h,family,items,cand=candidate_fixture(4)
    control.record_exploration("TEST_PROGRAM","WINNER_LOSER_ANALYSIS","synthetic diagnostic",q.question_id,h.hypothesis_id,items[-1].experiment_id,cand.candidate_id,p.dataset_id)
    burden=control.burden(cand.candidate_id)
    assert len(items)==4 and burden["number_of_experiments"]==4 and cand.status==CandidateStatus.EXPLORATORY


@pytest.mark.parametrize("missing",["purpose","relevance_to_parent","evidence_origins","research_mode"])
def test_controlled_child_question_requires_structured_purpose(missing):
    reg,p,_=setup_registry();control=ResearchControl(reg);values=dict(question_id="QX",program_id="TEST_PROGRAM",question_text="New unique question",parent_question_id="TEST_Q0",purpose="p",relevance_to_parent="r",evidence_origins=[EvidenceOrigin.THEORY],research_mode=ResearchMode.DISCOVERY,rationale="")
    values[missing]=[] if missing=="evidence_origins" else None
    with pytest.raises(ResearchError,match="missing"):control.create_question(**values)


def test_experiment_why_gate_requires_pass_fail_learning():
    reg,control,p,q,h,_=setup_control();bad={"question_answered":"Q","why_it_matters":"x","evidence_or_theory":"x"};exp=reg.create_experiment(program_id="TEST_PROGRAM",question_id=q.question_id,hypothesis_id=h.hypothesis_id,title="bad",description="",dataset_ids=[p.dataset_id],strategy_spec={"x":1},execution_config=ExecutionConfig().model_dump(mode="json"),evaluation_spec={"x":1},code_version={"revision":"x"},research_mode="DISCOVERY",motivation_spec=bad)
    with pytest.raises(PreRegistrationError,match="why-test"):reg.approve_experiment(exp.experiment_id)


def test_child_question_parent_and_origins_preserved():
    _,_,_,q,_,_=setup_control();assert q.parent_question_id=="TEST_Q0" and q.research_mode=="DISCOVERY" and q.evidence_origins==("DISCOVERY_DATA",)


def test_diversion_classifications_and_validation_strictness():
    reg,control,_,q,_,_=setup_control();assert control.diversion("TEST_Q0","DISCOVERY")==DiversionClassification.ON_PATH;assert control.diversion(q.question_id,"DISCOVERY")==DiversionClassification.EXPLORATORY_BRANCH
    assert control.diversion("MISSING","VALIDATION")==DiversionClassification.UNRELATED
    with pytest.raises(ResearchError,match="prohibited"):control.authorize_diversion("MISSING","VALIDATION","because")


def test_closed_question_cannot_be_recreated_and_reopen_preserves_history():
    reg,control,_,q,_,_=setup_control();control.set_question_status(q.question_id,QuestionStatus.CLOSED,"answered",changed_by="tester")
    with pytest.raises(ResearchError,match="POSSIBLE_RETURN"):control.create_question(question_id="NEW",program_id="TEST_PROGRAM",question_text=q.question_text,parent_question_id="TEST_Q0",purpose="p",relevance_to_parent="r",evidence_origins=["THEORY"],research_mode="DISCOVERY")
    with pytest.raises(ResearchError,match="requires reason"):control.reopen_question(q.question_id,"","","tester")
    reopened=control.reopen_question(q.question_id,"new independent data","new dataset","tester");assert reopened.status==QuestionStatus.OPEN
    with reg.catalog.connection() as con:history=con.execute("SELECT old_status,new_status,reason FROM question_status_history WHERE question_id=? ORDER BY changed_at",(q.question_id,)).fetchall()
    assert any(x[0]=="CLOSED" and x[1]=="OPEN" for x in history)


def test_decision_mapping_and_stopping_conditions_are_pre_registered():
    _,control,_,q,_,_=setup_control();decision=control.create_decision(q.question_id,"Continue?",{"PASS":"FREEZE_CANDIDATE","FAIL":"CLOSE_HYPOTHESIS"},"DISCOVERY",{"maximum_discovery_attempts":5});recorded=control.record_decision(decision.decision_id,"PASS")
    assert recorded.recorded_action.value=="FREEZE_CANDIDATE" and decision.stopping_conditions["maximum_discovery_attempts"]==5


def test_frozen_candidate_immutable_and_new_version_returns_to_discovery():
    reg,control,_,_,_,_,_,cand=candidate_fixture();validation=second_dataset(reg);frozen=control.freeze_candidate(cand.candidate_id,[validation.dataset_id])
    with pytest.raises(ImmutableRecordError):control.modify_candidate(frozen.candidate_id,strategy_spec={"threshold":99})
    child=control.new_candidate_version(frozen.candidate_id,strategy_spec={"threshold":3})
    assert child.version==2 and child.status==CandidateStatus.EXPLORATORY and child.validation_status=="NOT_STARTED" and child.parent_candidate_id==frozen.candidate_id


def test_validation_gates_unfrozen_lineage_usage_and_eligible_candidate():
    reg,control,p,_,_,_,_,cand=candidate_fixture();validation=second_dataset(reg)
    assert not control.validation_eligibility(cand.candidate_id)["eligible"]
    frozen=control.freeze_candidate(cand.candidate_id,[validation.dataset_id]);assert control.validation_eligibility(frozen.candidate_id)["eligible"]
    reg2,control2,p2,_,_,_,_,cand2=candidate_fixture();validation2=second_dataset(reg2);control2.auditor.record_dataset_usage(validation2.dataset_id,"TEST_PROGRAM","old",DatasetUsageType.DISCOVERY,"inspected");frozen2=control2.freeze_candidate(cand2.candidate_id,[validation2.dataset_id]);gate=control2.validation_eligibility(frozen2.candidate_id)
    assert not gate["eligible"] and "used for discovery" in " ".join(gate["reasons"])


def test_validation_failure_rejects_v1_contaminates_data_and_v2_is_discovery():
    reg,control,_,_,_,_,_,cand=candidate_fixture();validation=second_dataset(reg);frozen=control.freeze_candidate(cand.candidate_id,[validation.dataset_id]);validating=control.request_validation(frozen.candidate_id);failed=control.validation_failed(validating.candidate_id,{"criterion":"failed"});child=control.new_candidate_version(failed.candidate_id,strategy_spec={"threshold":3})
    assert failed.status==CandidateStatus.REJECTED and child.status==CandidateStatus.EXPLORATORY and child.validation_dataset_ids==()
    with reg.catalog.connection() as con:assert con.execute("SELECT 1 FROM dataset_usage WHERE dataset_id=? AND usage_type='VALIDATION'",(validation.dataset_id,)).fetchone()


def test_family_exhaustion_does_not_close_q0_and_new_family_can_exist():
    reg,control,_,_,_,family=setup_control();exhausted=control.set_family_status(family.strategy_family_id,StrategyFamilyStatus.EXHAUSTED,"synthetic branch exhausted");other=control.create_family("TEST_PROGRAM","Family B","later human idea")
    assert exhausted.status==StrategyFamilyStatus.EXHAUSTED and control.get_family(family.strategy_family_id).status==StrategyFamilyStatus.EXHAUSTED
    assert reg.get_question("TEST_Q0").status==QuestionStatus.OPEN and other.strategy_family_id!=family.strategy_family_id


def test_status_what_next_dead_end_and_branch_closure():
    reg,control,_,q,h,_=setup_control();decision=control.create_decision(q.question_id,"Next?",{"STOP":"CLOSE_QUESTION"},"DISCOVERY")
    assert control.what_next("TEST_PROGRAM")["type"]=="REGISTERED_DECISION"
    control.record_decision(decision.decision_id,"STOP");control.close_branch(q.question_id,"done","tester")
    assert control.dead_end(q.question_id)=="RESEARCH_BRANCH_EXHAUSTED"
    status=control.status_report("TEST_PROGRAM");assert q.question_id in status["closed_questions"] and status["current_next_decision"]["type"]=="HUMAN_RESEARCH_DECISION_REQUIRED"


def test_search_burden_card_and_journal_integrity():
    reg,control,p,q,h,family,items,cand=candidate_fixture(3);control.record_exploration("TEST_PROGRAM","DATASET_INSPECTION","looked",q.question_id,h.hypothesis_id,items[-1].experiment_id,cand.candidate_id,p.dataset_id);burden=control.burden(cand.candidate_id);card=control.candidate_card(cand.candidate_id)
    assert burden["number_of_experiments"]==3 and burden["number_of_parameter_variants"]>=1 and burden["number_of_result_driven_modifications"]==2
    assert card["candidate"]==cand.candidate_id and reg.verify_journal_integrity()


def test_final_test_hook_requires_successful_validation_and_unseen_data():
    reg,control,_,_,_,_,_,cand=candidate_fixture();final=second_dataset(reg,"final")
    gate=control.final_test_eligibility(cand.candidate_id,[final.dataset_id]);assert not gate["eligible"] and "successful validation" in " ".join(gate["reasons"])
