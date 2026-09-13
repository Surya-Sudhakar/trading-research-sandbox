import json

import pandas as pd

from sandbox.execution.artifacts import store_run
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction, EntryType, ExecutionConfig, TradeIntent
from sandbox.research.models import ExperimentStatus, VerdictValue
from test_stage3_registry import create_experiment, setup_registry


def stage2_run(reg,p):
    frame=pd.DataFrame({"timestamp_utc":pd.to_datetime(["2024-01-01 00:00Z","2024-01-01 00:01Z"],utc=True),"open":[99,100],"high":[101,121],"low":[98,99],"close":[100,120],"tick_volume":[1,1],"spread":[0,0],"real_volume":[0,0]})
    item=TradeIntent(trade_intent_id="stage3-link",strategy_id="synthetic",experiment_id="pending",dataset_id=p.dataset_id,symbol="X",direction=Direction.LONG,signal_timestamp=pd.Timestamp("2024-01-01 00:00:30Z").to_pydatetime(),requested_entry_type=EntryType.MARKET_NEXT_OPEN,stop_loss=90,take_profit=120)
    result=BacktestEngine(ExecutionConfig()).run([item],frame); stored=store_run(result,reg.results_root/"stage2")
    reg.catalog.add_research_run(stored.run_id,stored.run_fingerprint,p.dataset_id,result.execution_engine_version,result.ledger_schema_version,result.metrics_version,result.execution_config.model_dump(mode="json"),str(stored.ledger_path),str(stored.summary_path))
    return stored


def test_stage2_result_link_verdict_separation_and_immutability():
    reg,p,_=setup_registry(); exp=create_experiment(reg,p); reg.approve_experiment(exp.experiment_id); reg.transition_experiment(exp.experiment_id,ExperimentStatus.RUNNING)
    stored=stage2_run(reg,p); result=reg.link_stage2_run(exp.experiment_id,stored.run_id)
    before=result.summary_metrics.copy(); verdict=reg.record_verdict(exp.experiment_id,result.result_id,VerdictValue.FAIL,"Synthetic threshold deliberately not met","pytest")
    assert verdict.result_id == result.result_id and reg.get_result(result.result_id).summary_metrics == before
    assert reg.get_result(result.result_id).ledger_checksum == stored.ledger_checksum


def test_manifest_is_deterministic_and_timeline_is_structural():
    reg,p,_=setup_registry(); exp=create_experiment(reg,p); reg.approve_experiment(exp.experiment_id); reg.transition_experiment(exp.experiment_id,ExperimentStatus.RUNNING)
    stored=stage2_run(reg,p); result=reg.link_stage2_run(exp.experiment_id,stored.run_id); reg.record_verdict(exp.experiment_id,result.result_id,VerdictValue.PASS,"Synthetic truth matched","pytest")
    first=reg.generate_manifest(exp.experiment_id); contents=[x.read_bytes() for x in first]; second=reg.generate_manifest(exp.experiment_id)
    assert contents == [x.read_bytes() for x in second]
    timeline=reg.timeline("TEST_PROGRAM"); assert "TEST_Q0" in timeline and "TEST_H001" in timeline and exp.experiment_id in timeline and "PASS" in timeline

