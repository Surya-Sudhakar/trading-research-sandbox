import pytest

from sandbox.audit.models import AuditPhase
from sandbox.research.errors import ResearchError
from sandbox.research.models import ExperimentStatus,VerdictValue
from test_stage3_registry import create_experiment,setup_registry
from test_stage3_results import stage2_run


def linked(evaluation=None):
    reg,p,_=setup_registry();exp=create_experiment(reg,p,evaluation_spec=evaluation or {"minimum_win_rate":0.5});reg.approve_experiment(exp.experiment_id);reg.transition_experiment(exp.experiment_id,ExperimentStatus.RUNNING);stored=stage2_run(reg,p);result=reg.link_stage2_run(exp.experiment_id,stored.run_id);return reg,exp,result


def test_post_run_audit_has_sample_warning_and_is_stored():
    reg,exp,result=linked();audits=__import__("sandbox.audit.auditor",fromlist=["ResearchAuditor"]).ResearchAuditor(reg).audit_history(exp.experiment_id)
    post=next(x for x in audits if x.phase==AuditPhase.POST_RUN)
    assert any(x.code=="VERY_SMALL_SAMPLE" for x in post.findings)


def test_verdict_guard_rejects_pass_below_frozen_criterion():
    reg,exp,result=linked({"minimum_win_rate":1.1})
    with pytest.raises(ResearchError,match="VERDICT_CRITERIA_MISMATCH"):reg.record_verdict(exp.experiment_id,result.result_id,VerdictValue.PASS,"wishful","tester")
    assert reg.get_result(result.result_id).summary_metrics["win_rate"]==1.0


def test_criteria_compatible_pass_is_accepted():
    reg,exp,result=linked({"minimum_win_rate":0.5})
    verdict=reg.record_verdict(exp.experiment_id,result.result_id,VerdictValue.PASS,"criterion met","tester")
    assert verdict.verdict==VerdictValue.PASS
