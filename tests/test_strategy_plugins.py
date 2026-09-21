from __future__ import annotations
from dataclasses import dataclass
from datetime import timezone,timedelta
from pathlib import Path
from uuid import uuid4
import random
import pandas as pd
import pytest
from pydantic import ValidationError
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction,EntryType,ExecutionConfig
from sandbox.research.errors import ResearchError
from sandbox.strategies.artifacts import store_signal_ledger
from sandbox.strategies.context import StrategyContext
from sandbox.strategies.examples import SyntheticEveryNthBar
from sandbox.strategies.models import ParameterDefinition,ParameterType,ResearchPhase,StrategyMetadata,StrategySignal,WarmupRequirements
from sandbox.strategies.registry import StrategyRegistry,default_registry
from sandbox.strategies.runtime import StrategyRuntime
from sandbox.partition.service import PartitionService

def frame(n=8,start="2024-01-01"):
    return pd.DataFrame({"timestamp_utc":pd.date_range(start,periods=n,freq="min",tz="UTC"),"open":[10.0]*n,"high":[11.0]*n,"low":[9.0]*n,"close":[10.0]*n,"spread":[0]*n,"tick_volume":[1]*n,"real_volume":[0]*n})
def run(runtime=None,**kw):return (runtime or StrategyRuntime(default_registry())).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="DATA",experiment_id="EXP",**kw)

def test_valid_synthetic_plugin_registers_metadata_and_schema_preserved():
    r=default_registry();m=r.resolve("SYNTHETIC_INTERFACE_TEST").metadata;assert m.strategy_family=="TEST_ONLY" and m.parameter_schema[0].name=="every_n" and m.synthetic_test_plugin
def test_invalid_plugin_rejected():
    with pytest.raises(ResearchError,match="metadata"):StrategyRegistry().register(object())
def test_strategy_and_parameter_fingerprints_deterministic_and_material_change():
    r=default_registry();p=r.resolve("SYNTHETIC_INTERFACE_TEST");a=r.fingerprint(p,{"every_n":2});b=r.fingerprint(p,{"every_n":2});c=r.fingerprint(p,{"every_n":3});assert a==b and a[0]!=c[0] and a[2]!=c[2]
def test_code_timeframe_and_entry_implementation_changes_fingerprint():
    base=SyntheticEveryNthBar();registry=StrategyRegistry();original=registry.fingerprint(base,{"every_n":2})
    class ChangedCode(SyntheticEveryNthBar):
        def evaluate(self,context):return ()
    changed=registry.fingerprint(ChangedCode(),{"every_n":2});assert original[0]!=changed[0] and original[1]!=changed[1]
    class ChangedTimeframe(SyntheticEveryNthBar):metadata=SyntheticEveryNthBar.metadata.model_copy(update={"required_timeframes":("M1","M5")})
    assert original[0]!=registry.fingerprint(ChangedTimeframe(),{"every_n":2})[0]
def test_same_input_same_signals_and_state_reset():
    rt=StrategyRuntime(default_registry());a=rt.assert_deterministic("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="DATA",experiment_id="EXP");b=run(rt);assert a.signal_ledger==b.signal_ledger
def test_signal_ledger_fingerprint_and_artifact():
    root=Path("tests/runtime")/("strategy-signals-"+uuid4().hex);a=run();stored=store_signal_ledger(a.signal_ledger,root,"RUN");assert stored.row_count==len(a.intents) and stored.ledger_fingerprint==a.signal_ledger.ledger_fingerprint
    with pytest.raises(FileExistsError):store_signal_ledger(a.signal_ledger,root,"RUN")
def test_context_cannot_be_forged_mutated_or_expose_paths():
    now=pd.Timestamp("2024-01-01",tz="UTC").to_pydatetime()
    with pytest.raises(ResearchError,match="trusted runtime"):StrategyContext(current_time=now,information_cutoff=now,symbol="X",phase=ResearchPhase.DISCOVERY,parameters={},history={},warmup_state={},analysis_timezone="UTC")
    class Mutator(SyntheticEveryNthBar):
        def evaluate(self,c):c._current_time=now
    reg=StrategyRegistry();reg.register(Mutator());rt=StrategyRuntime(reg)
    with pytest.raises(TypeError,match="read-only"):rt.run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E")
def test_no_next_candle_or_future_dataframe_available():
    seen=[]
    class Probe(SyntheticEveryNthBar):
        def evaluate(self,c):seen.append((c.current_time,len(c.history("SYNTH","M1",999))));return ()
    reg=StrategyRegistry();reg.register(Probe());StrategyRuntime(reg).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame(4)},symbol="SYNTH",dataset_id="D",experiment_id="E")
    assert all(length==i+2 for i,(_,length) in enumerate(seen)) and not hasattr(seen,"dataframe")
def test_future_signal_timestamp_denied():
    class Future(SyntheticEveryNthBar):
        def evaluate(self,c):
            b=c.current_closed_bar("SYNTH","M1");t=c.current_time+timedelta(minutes=1);return [StrategySignal(signal_id="x",setup_id="x",symbol="SYNTH",decision_timestamp=t,information_cutoff=c.information_cutoff,direction="LONG",entry_type="MARKET_CLOSE",reference_price=10,stop_price=9,target_price=12)]
    reg=StrategyRegistry();reg.register(Future())
    with pytest.raises(ResearchError,match="FUTURE_SIGNAL"):StrategyRuntime(reg).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E")
def test_partition_and_stage7_access_are_denied_by_context():
    class Probe(SyntheticEveryNthBar):
        def evaluate(self,c):c.request_partition("FINAL_TEST")
    reg=StrategyRegistry();reg.register(Probe())
    with pytest.raises(ResearchError,match="PROTECTED_PARTITION"):StrategyRuntime(reg).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E",phase="DISCOVERY")
    class Stats(SyntheticEveryNthBar):
        def evaluate(self,c):c.stage7_results()
    reg=StrategyRegistry();reg.register(Stats())
    with pytest.raises(ResearchError,match="PERFORMANCE_ACCESS"):StrategyRuntime(reg).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E")
def test_market_intents_accepted_and_stop_rejected():
    assert run().intents and all(x.requested_entry_type==EntryType.MARKET_NEXT_OPEN for x in run().intents)
    for entry in (EntryType.STOP,):
        class Unsupported(SyntheticEveryNthBar):
            def evaluate(self,c):return [StrategySignal(signal_id="x",setup_id="x",symbol="SYNTH",decision_timestamp=c.current_time,information_cutoff=c.information_cutoff,direction="LONG",entry_type=entry,reference_price=10,stop_price=9,target_price=12)]
        reg=StrategyRegistry();reg.register(Unsupported())
        with pytest.raises(ResearchError,match="UNSUPPORTED_STRATEGY_ENTRY"):StrategyRuntime(reg).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E")
@pytest.mark.parametrize("direction,stop,target",[("LONG",11,12),("SHORT",9,8),("LONG",9,8)])
def test_invalid_geometry_rejected(direction,stop,target):
    with pytest.raises(ValidationError):StrategySignal(signal_id="x",setup_id="s",symbol="X",decision_timestamp=pd.Timestamp("2024-01-01",tz="UTC"),information_cutoff=pd.Timestamp("2024-01-01",tz="UTC"),direction=direction,entry_type="MARKET_CLOSE",reference_price=10,stop_price=stop,target_price=target)
def test_missing_field_and_timeframe_rejected():
    with pytest.raises(ResearchError,match="DATA_FIELD"):run(frames=None) if False else StrategyRuntime(default_registry()).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame().drop(columns="spread")},symbol="SYNTH",dataset_id="D",experiment_id="E")
    class Multi(SyntheticEveryNthBar):metadata=SyntheticEveryNthBar.metadata.model_copy(update={"required_timeframes":("M1","M5")})
    reg=StrategyRegistry();reg.register(Multi())
    with pytest.raises(ResearchError,match="TIMEFRAME_UNAVAILABLE"):StrategyRuntime(reg).run("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E")
def test_multitimeframe_cutoff_and_timezone_deterministic():
    observations=[]
    class Multi(SyntheticEveryNthBar):
        metadata=SyntheticEveryNthBar.metadata.model_copy(update={"required_timeframes":("M1","M5"),"analysis_timezone":"America/New_York"})
        def evaluate(self,c):observations.append((c.current_time_local.utcoffset(),all(x.timestamp_utc<=c.information_cutoff for x in c.history("SYNTH","M5",99))));return ()
    reg=StrategyRegistry();reg.register(Multi());rt=StrategyRuntime(reg);m5=frame(2);rt.run("SYNTHETIC_INTERFACE_TEST",{"M1":frame(4),"M5":m5},symbol="SYNTH",dataset_id="D",experiment_id="E");assert observations and all(x[1] for x in observations) and observations[0][0].total_seconds()==-18000
def test_stage6_timeframe_builder_uses_only_complete_causal_bars():
    class Multi(SyntheticEveryNthBar):metadata=SyntheticEveryNthBar.metadata.model_copy(update={"required_timeframes":("M1","M5")})
    built=PartitionService._strategy_timeframe_frames(frame(7),Multi());assert len(built["M5"])==1 and built["M5"].timestamp_utc.iloc[0]==frame(7).timestamp_utc.iloc[0]
def test_warmup_prohibits_trading_until_ready():
    result=run(parameters={"every_n":1});assert result.intents[0].signal_timestamp==frame().timestamp_utc.iloc[1]+timedelta(minutes=1)
def test_strategy_instances_are_isolated():
    a=StrategyRuntime(default_registry());b=StrategyRuntime(default_registry());assert run(a).signal_ledger==run(b).signal_ledger
def test_hidden_random_nondeterminism_detected():
    class RandomPlugin(SyntheticEveryNthBar):
        def evaluate(self,c):
            if random.random()<.5:return ()
            return super().evaluate(c)
    reg=StrategyRegistry();reg.register(RandomPlugin());rt=StrategyRuntime(reg)
    original=random.random;values=iter([.1]*7+[.9]*20);random.random=lambda:next(values)
    try:
        with pytest.raises(ResearchError,match="NONDETERMINISTIC"):rt.assert_deterministic("SYNTHETIC_INTERFACE_TEST",{"M1":frame()},symbol="SYNTH",dataset_id="D",experiment_id="E")
    finally:random.random=original
def test_frozen_candidate_requires_exact_strategy_fingerprints():
    result=run();candidate=type("Candidate",(),{"strategy_spec":{"strategy_code_fingerprint":result.strategy_code_fingerprint,"parameter_fingerprint":result.parameter_fingerprint}})();assert StrategyRuntime.assert_frozen_binding(candidate,result)
    candidate.strategy_spec["parameter_fingerprint"]="changed"
    with pytest.raises(ResearchError,match="FROZEN_CANDIDATE"):StrategyRuntime.assert_frozen_binding(candidate,result)
def test_signal_to_stage2_ledger_linkage_exact():
    s=run(parameters={"every_n":1});ledger=BacktestEngine(ExecutionConfig()).run(s.intents,frame()).ledger;assert [x.trade_intent_id for x in ledger]==[x.signal_id for x in s.signal_ledger.records]
def test_audit_and_manifest_metadata_complete():
    s=run();assert {"strategy_code_fingerprint","parameter_fingerprint","signal_ledger_fingerprint","information_cutoff_policy"}<=set(s.audit_metadata);assert all(x.metadata["strategy_code_fingerprint"]==s.strategy_code_fingerprint for x in s.intents)
def test_real_strategy_registry_contains_only_explicit_s001():assert [x.strategy_id for x in default_registry().list(include_test=False)]==["S001"]
