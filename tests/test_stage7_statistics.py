from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from sandbox.statistics.service import StatisticalEvidenceEngine,max_drawdown,wilson
from sandbox.research.registry import ResearchRegistry
from sandbox.provenance import sha256_file
from uuid import uuid4

def ledger(outcomes,years=None,symbols=None,directions=None,bars=None):
    n=len(outcomes);years=years or [2022]*n;symbols=symbols or ["EURUSD"]*n;directions=directions or ["LONG"]*n;rows=[]
    for i,r in enumerate(outcomes):
        entry=pd.Timestamp(f"{years[i]}-01-01",tz="UTC")+pd.Timedelta(int(i%300),unit="D");exit=entry+pd.Timedelta(int(bars[i] if bars else 1),unit="min")
        rows.append({"trade_id":f"T{i}","status":"CLOSED","net_r":float(r),"direction":directions[i],"symbol":symbols[i],"entry_timestamp":entry,"exit_timestamp":exit,"bars_held":bars[i] if bars else 1})
    return pd.DataFrame(rows)

@pytest.fixture
def engine():
    e=StatisticalEvidenceEngine(config_path=Path("config/statistical_evidence.json"));e.config["bootstrap_iterations"]=200;return e

def analyze(e,values,**kw):return e.analyze_frame(ledger(values,**kw),"RUN-X","abc")

def test_basic_metrics_and_drawdown_truth(engine):
    p=analyze(engine,[2,2,-1,-1,-1,2]).performance_profile
    assert p["win_rate"]==.5 and p["average_R"]==.5 and p["total_R"]==3 and p["max_drawdown_R"]==3 and p["longest_losing_streak"]==3

def test_wilson_trusted_reference_and_edges():
    x=wilson(52,100);assert x["lower"]==pytest.approx(.42316577765) and x["upper"]==pytest.approx(.61535448242)
    assert wilson(10,10)["upper"]==1 and wilson(0,10)["lower"]==0 and wilson(0,0)["status"]=="NO_BINARY_TRADES"

def test_tiny_sample_and_52_percent_problem(engine):
    tiny=analyze(engine,[2]*12+[-1]*8);assert "VERY_SMALL_SAMPLE" in tiny.warnings and "WIDE_WIN_RATE_INTERVAL" in tiny.warnings
    a=wilson(52,100);b=wilson(520,1000);assert a["observed_win_rate"]==b["observed_win_rate"]==.52 and a["upper"]-a["lower"]>b["upper"]-b["lower"]

def test_large_multiyear_stable_fixture_has_narrower_uncertainty(engine):
    values=([2]*55+[-1]*45)*4;years=[2020]*100+[2021]*100+[2022]*100+[2023]*100;large=analyze(engine,values,years=years);small=analyze(engine,[2]*12+[-1]*8)
    lc=large.uncertainty_profile["win_rate"];sc=small.uncertainty_profile["win_rate"]
    assert large.sample_profile["sample_label"]=="LARGE" and all(x["total_R"]>0 for x in large.temporal_profile["yearly"]) and lc["upper"]-lc["lower"]<sc["upper"]-sc["lower"]

def test_bootstrap_seed_determinism_and_variation(engine):
    x=np.array([2,-1,2,-1,-1,2,2]);a=engine._bootstrap(x,7);b=engine._bootstrap(x,7);c=engine._bootstrap(x,8)
    assert a==b and (a["lower"],a["upper"])!=(c["lower"],c["upper"]) and a["lower"]<=np.mean(x)<=a["upper"]

def test_zero_variance_and_block_bootstrap(engine):
    a=engine._bootstrap(np.ones(30),4);b=engine._bootstrap(np.ones(30),4,True)
    assert a["lower"]==a["upper"]==1 and b==engine._bootstrap(np.ones(30),4,True) and engine._bootstrap(np.ones(3),4,True)["status"]=="BLOCK_BOOTSTRAP_NOT_INFORMATIVE"

def test_dependence_distinguishes_sequences(engine):
    clustered=analyze(engine,[2]*20+[-1]*20);alternating=analyze(engine,[2,-1]*20)
    assert "SERIAL_DEPENDENCE" in clustered.warnings and clustered.dependence_profile["lag1_outcome_autocorrelation"]>0 and alternating.dependence_profile["lag1_outcome_autocorrelation"]<0

def test_temporal_partial_frequency_and_concentration(engine):
    vals=[2]*30+[-1]*10+[2]*5+[-1]*25;years=[2021]*40+[2022]*30;r=analyze(engine,vals,years=years);y=r.temporal_profile["yearly"];f=r.temporal_profile["trade_frequency"]
    assert len(y)==2 and y[0]["trades"]==40 and y[0]["total_R"]==50 and f["mean_trades_per_year"]==35 and f["median_trades_per_year"]==35
    assert all(x["year_completeness"]=="PARTIAL_YEAR" for x in y) and "PARTIAL_YEAR_ONLY" in r.warnings and "TEMPORAL_CONCENTRATION" in r.warnings and f["minimum_full_year_trades"] is None

def test_tiny_year_groups(engine):
    r=analyze(engine,[2]*10+[-1]*10,years=[2020]*5+[2021]*15);f=r.temporal_profile["trade_frequency"]
    assert f["minimum_trades_per_year"]==5 and "INSUFFICIENT_GROUP_SIZE" in r.warnings

def test_symbol_pooled_concentration(engine):
    r=analyze(engine,[2]*30+[-1]*10+[2]*2+[-1]*18,symbols=["EURUSD"]*40+["GBPUSD"]*20);rows=r.symbol_profile["symbols"]
    assert len(rows)==2 and r.symbol_profile["pooled"]["trades"]==60 and rows[0]["trades"]==40 and "SYMBOL_CONCENTRATION" in r.warnings

def test_direction_and_long_only(engine):
    r=analyze(engine,[2,-1,2,-1],directions=["LONG","LONG","SHORT","SHORT"]);assert len(r.direction_profile["directions"])==2
    long_only=analyze(engine,[2,-1]*10);assert len(long_only.direction_profile["directions"])==1 and "DIRECTION_CONCENTRATION" not in long_only.warnings

def test_drawdown_resampling_and_order(engine):
    d=max_drawdown([2,-1,-1,2,2]);assert d["maximum_drawdown_R"]==2 and d["recovered"]
    r=analyze(engine,[2,-1,-1,2,2,-1]*10);q=r.resampling_profile["iid"]["max_drawdown_percentiles"]
    assert q["50"]<=q["90"]<=q["95"]<=q["99"] and r.drawdown_profile["order_sensitivity"]==analyze(engine,[2,-1,-1,2,2,-1]*10).drawdown_profile["order_sensitivity"]

def test_outlier_concentration(engine):
    r=analyze(engine,[20]+[-1]*9);c=r.concentration_profile
    assert c["top_1_trade_contribution"]==pytest.approx(20/11) and c["total_R_after_removing_best_trade"]==-9 and "OUTLIER_DEPENDENCE" in r.warnings

def test_registered_sensitivity_no_generation_or_best(engine):
    stable=[{"parameter_value":1.4,"expectancy":.4},{"parameter_value":1.5,"expectancy":.42},{"parameter_value":1.6,"expectancy":.39}];x=engine.sensitivity(stable,"threshold")
    assert x["status"]=="BROADLY_STABLE" and x["generated_parameter_values"]==[] and "best" not in json.dumps(x).lower()
    cliff=[{"parameter_value":1.4,"expectancy":.5},{"parameter_value":1.5,"expectancy":.55},{"parameter_value":1.6,"expectancy":-1}];assert engine.sensitivity(cliff,"threshold")["warnings"]==["PARAMETER_CLIFF"]

def test_registered_cost_sensitivity(engine):
    rows=[{"name":"BASELINE","adversity_rank":0,"expectancy":.2},{"name":"SEVERE_ADVERSE","adversity_rank":2,"expectancy":-.1}];x=engine.cost_sensitivity(rows)
    assert x["baseline"]=="BASELINE" and x["status"]=="COST_FRAGILE"
    with pytest.raises(Exception,match="BASELINE"):engine.cost_sensitivity([rows[1]])

def test_search_burden_subgroup_warning_not_rejection(engine):
    burden={"experiments_attempted_in_family":25,"hypotheses_tested":3,"parameter_variants_inspected":4,"result_driven_descendants":2,"subgroup_analyses_performed":8};r=engine.analyze_frame(ledger([2,-1]*50),"R","x",search_burden=burden)
    assert "HIGH_DISCOVERY_SEARCH_BURDEN" in r.warnings and "MULTIPLE_SUBGROUP_INSPECTION" in r.warnings and "invalid" not in r.evidence_summary.lower()

def test_frozen_evaluation_distinguishes_observed_and_bound(engine):
    spec={"minimum_win_rate":.5,"minimum_wilson_lower_bound":.5};before=dict(spec);r=engine.analyze_frame(ledger([2]*52+[-1]*48),"R","x",evaluation_spec=spec);rows={x["criterion"]:x for x in r.uncertainty_profile["evaluation_criteria"]}
    assert rows["minimum_win_rate"]["result"]=="CRITERION_MET" and rows["minimum_wilson_lower_bound"]["result"]=="CRITERION_NOT_MET" and spec==before

def test_edge_cases_undefined_pf_and_fingerprint(engine):
    cols=["trade_id","status","net_r","direction","symbol","entry_timestamp","exit_timestamp"];empty=engine.analyze_frame(pd.DataFrame(columns=cols),"R","x")
    assert empty.sample_profile["resolved_trades"]==0 and empty.performance_profile["profit_factor"]=="UNDEFINED_NO_LOSSES"
    assert analyze(engine,[2]).sample_profile["resolved_trades"]==1 and analyze(engine,[2]*10).performance_profile["profit_factor"]=="UNDEFINED_NO_LOSSES" and analyze(engine,[-1]*10).performance_profile["win_rate"]==0
    f=ledger([2,-1]*20);a=engine.analyze_frame(f,"R","a");b=engine.analyze_frame(f,"R","a");c=engine.analyze_frame(f,"R","b");assert a.report_fingerprint==b.report_fingerprint!=c.report_fingerprint

def test_cross_symbol_overlap(engine):
    f=ledger([2,-1],symbols=["EURUSD","GBPUSD"],bars=[60,60]);f.loc[1,"entry_timestamp"]=f.loc[0,"entry_timestamp"]+pd.Timedelta(1,unit="min");f.loc[1,"exit_timestamp"]=f.loc[1,"entry_timestamp"]+pd.Timedelta(60,unit="min");r=engine.analyze_frame(f,"R","x")
    assert r.symbol_profile["overlap"]["trades_overlapping_another_symbol"]==2 and "CROSS_SYMBOL_DEPENDENCE_POSSIBLE" in r.warnings

def test_nominal_one_to_two_context():
    assert 1/(1+2)==pytest.approx(.3333333333) and np.mean([2,-1])==.5

def test_ledger_checksum_gate_persistence_and_journal():
    tmp_path=Path("tests/runtime")/("stage7-"+uuid4().hex);tmp_path.mkdir(parents=True)
    reg=ResearchRegistry(tmp_path/"catalog.db",tmp_path,tmp_path/"results");reg.catalog.initialize();frame=ledger([2,-1]*10);path=tmp_path/"ledger.parquet";frame.to_parquet(path,index=False);checksum=sha256_file(path);summary=tmp_path/"summary.json";summary.write_text(json.dumps({"ledger_checksum":checksum}),encoding="utf-8")
    reg.catalog.add_research_run("RUN-A","fingerprint","dataset","1.0.0","1.0.0","1.0.0",{},str(path.resolve()),str(summary.resolve()));engine=StatisticalEvidenceEngine(reg,Path("config/statistical_evidence.json"),tmp_path/"evidence");engine.config["bootstrap_iterations"]=30
    report=engine.analyze_run("RUN-A");assert engine.get(report.report_id).report_fingerprint==report.report_fingerprint and reg.verify_journal_integrity()
    frame.iloc[0,frame.columns.get_loc("net_r")]=99;frame.to_parquet(path,index=False)
    with pytest.raises(Exception,match="ledger checksum mismatch"):engine.analyze_run("RUN-A")
