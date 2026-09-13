from __future__ import annotations
from datetime import timedelta
import math
import pandas as pd
import pytest
from pydantic import ValidationError
from sandbox.execution.engine import BacktestEngine
from sandbox.execution.models import Direction,EntryType,ExecutionConfig,PositionStatus
from sandbox.partition.service import PartitionService
from sandbox.research.errors import ResearchError
from sandbox.strategies.context import StrategyContext
from sandbox.strategies.models import CandleSnapshot
from sandbox.strategies.registry import StrategyRegistry,default_registry
from sandbox.strategies.runtime import StrategyRuntime
from sandbox.strategies.s001 import S001Strategy,BASELINE_PARAMETERS
from sandbox.strategies.s001.features import classify_trend,impulse_anchors,qualify_momentum,true_ranges,wilder_atr
from sandbox.strategies.s001.models import SetupState,TrendContext
from sandbox.strategies.s001.parameters import FUTURE_DISCOVERY_ALTERNATIVES,PARAMETER_STATUS

def market_rows(direction="LONG",pullback=None):
    bullish=direction=="LONG";h=[]
    for i,t in enumerate(pd.date_range("2023-12-31",periods=15,freq="h",tz="UTC")):
        center=100+i if bullish else 115-i;h.append({"timestamp_utc":t,"open":center,"high":center+1,"low":center-.2,"close":center+(.8 if bullish else -.8),"spread":0})
    m=[]
    for i,t in enumerate(pd.date_range("2024-01-01",periods=15,freq="15min",tz="UTC")):
        if i<7:o=c=100;hi=100.2;lo=99.8
        elif bullish:o=100+(i-7)*.5;c=o+.5;hi=c+.1;lo=o-.1
        else:o=100-(i-7)*.5;c=o-.5;hi=o+.1;lo=c-.1
        m.append({"timestamp_utc":t,"open":o,"high":hi,"low":lo,"close":c,"spread":0})
    if pullback is None:
        pullback=([{"open":103.5,"high":103.3,"low":102.8,"close":103.0},{"open":103.0,"high":103.6,"low":102.7,"close":103.5}] if bullish else [{"open":96.5,"high":97.2,"low":96.7,"close":97.0},{"open":97.0,"high":97.3,"low":96.4,"close":96.5}])
    for values in pullback:
        values={**values,"timestamp_utc":pd.Timestamp("2024-01-01",tz="UTC")+timedelta(minutes=15*len(m)),"spread":0};m.append(values)
    return pd.DataFrame(h),pd.DataFrame(m)
def execute(direction="LONG",pullback=None,parameters=None,h1=None,m15=None):
    if h1 is None or m15 is None:h1,m15=market_rows(direction,pullback)
    plugin=S001Strategy();registry=StrategyRegistry();registry.register(plugin);result=StrategyRuntime(registry).run("S001",{"H1":h1,"M15":m15},symbol="EURUSD",dataset_id="SYNTH",experiment_id="TEST",parameters=parameters or {});return result,plugin
def snapshots(frame):return tuple(CandleSnapshot(timestamp_utc=r.timestamp_utc.to_pydatetime(),values={k:getattr(r,k) for k in ("open","high","low","close","spread")}) for r in frame.itertuples(index=False))

def test_s001_registers_and_identity_is_exact():
    p=default_registry().resolve("S001");m=p.metadata;assert (m.strategy_id,m.strategy_version,m.strategy_family)==("S001","1.0.0","INTRADAY_TREND_MOMENTUM_PULLBACK_CONTINUATION") and m.synthetic_test_plugin is False
def test_required_timeframes_entry_directions_and_warmup():
    m=S001Strategy.metadata;assert m.required_base_timeframe=="M15" and m.required_timeframes==("H1","M15") and m.warmup_requirements.bars_by_timeframe=={"H1":15,"M15":15} and set(m.supported_directions)=={Direction.LONG,Direction.SHORT}
def test_baseline_parameters_and_status_exact():
    assert dict(BASELINE_PARAMETERS)=={"h1_trend_lookback":12,"h1_atr_period":14,"h1_min_displacement_atr":1.5,"m15_momentum_lookback":8,"m15_atr_period":14,"m15_min_displacement_atr":1.25,"minimum_pullback":.25,"maximum_pullback":.6,"maximum_pullback_bars":8,"reward_risk":2.0};assert PARAMETER_STATUS["reward_risk"]=="FIXED_INITIAL_RESEARCH"
def test_parameter_and_code_fingerprints_bind_material_changes():
    r=default_registry();p=r.resolve("S001");base=r.fingerprint(p,dict(BASELINE_PARAMETERS));changed=dict(BASELINE_PARAMETERS);changed["minimum_pullback"]=.33;assert base[0]!=r.fingerprint(p,changed)[0]
    class Changed(S001Strategy):
        def evaluate(self,context):return ()
    assert base[1]!=r.code_fingerprint(Changed())
def test_future_discovery_values_are_documentation_not_defaults():
    assert FUTURE_DISCOVERY_ALTERNATIVES["h1_trend_lookback"]==(8,12,16) and dict(BASELINE_PARAMETERS)["h1_trend_lookback"]==12
def test_insufficient_warmup_emits_nothing():
    h,m=market_rows();result,_=execute(h1=h.iloc[:14],m15=m.iloc[:14]);assert not result.intents

@pytest.mark.parametrize("direction,expected",[("LONG",TrendContext.BULLISH),("SHORT",TrendContext.BEARISH)])
def test_h1_trend_classification(direction,expected):
    h,_=market_rows(direction);x=classify_trend(snapshots(h),12,14,1.5);assert x.context==expected and x.strength>1.5
def test_h1_weak_displacement_is_no_trend():
    h,_=market_rows();h[["open","close"]]=100;assert classify_trend(snapshots(h),12,14,1.5).context==TrendContext.NO_TREND
def test_wilder_atr_uses_previous_close_and_rma():
    h,_=market_rows();bars=snapshots(h);trs=true_ranges(bars);assert len(trs)==14 and wilder_atr(bars,14)==pytest.approx(sum(trs)/14) and S001Strategy.atr_convention.startswith("WILDER_RMA")
def test_zero_atr_safely_disqualifies():
    h,_=market_rows();h[["open","high","low","close"]]=100;assert wilder_atr(snapshots(h),14) is None and classify_trend(snapshots(h),12,14,1).context==TrendContext.NO_TREND

@pytest.mark.parametrize("direction",[Direction.LONG,Direction.SHORT])
def test_m15_momentum_qualifies_in_matching_direction(direction):
    _,m=market_rows(direction.value,[]);x=qualify_momentum(snapshots(m),direction,8,14,1.25);assert x.qualified and x.strength>1.25
@pytest.mark.parametrize("fixture_direction,test_direction",[("LONG",Direction.SHORT),("SHORT",Direction.LONG)])
def test_opposite_momentum_rejected(fixture_direction,test_direction):
    _,m=market_rows(fixture_direction,[]);assert not qualify_momentum(snapshots(m),test_direction,8,14,1.25).qualified
def test_weak_momentum_rejected():
    _,m=market_rows("LONG",[]);m.loc[7:,"close"]=m.loc[7:,"open"];assert not qualify_momentum(snapshots(m),Direction.LONG,8,14,99).qualified

@pytest.mark.parametrize("direction",[Direction.LONG,Direction.SHORT])
def test_ordered_impulse_anchors(direction):
    _,m=market_rows(direction.value,[]);a=impulse_anchors(snapshots(m),direction,8);assert a and a.start_timestamp<a.end_timestamp and a.high>a.low
def test_equal_extreme_tie_break_uses_earliest_eligible():
    _,m=market_rows("LONG",[]);m.loc[7,"low"]=99.5;m.loc[8,"low"]=99.5;a=impulse_anchors(snapshots(m),Direction.LONG,8);assert a.start_timestamp==m.timestamp_utc.iloc[7].to_pydatetime()
def test_invalid_anchor_ordering_and_zero_range_rejected():
    _,m=market_rows("LONG",[]);w=m.iloc[-8:].copy();w[["open","high","low","close"]]=100;assert impulse_anchors(snapshots(w),Direction.LONG,8) is None

@pytest.mark.parametrize("direction",["LONG","SHORT"])
def test_pullback_start_and_baseline_signal(direction):
    result,p=execute(direction);assert len(result.intents)==1 and p.setup_history[-1].pullback_start_timestamp is not None and p.setup_history[-1].pullback_bars==2
@pytest.mark.parametrize("depth,state",[(.20,SetupState.PULLBACK_MONITORING),(.25,SetupState.ARMED),(.60,SetupState.ARMED)])
def test_pullback_depth_boundaries(depth,state):
    impulse_high,impulse_low=104.1,99.9;low=impulse_high-depth*(impulse_high-impulse_low);bars=[{"open":103.5,"high":103.8,"low":low,"close":103.4}];_,p=execute("LONG",bars);setup=p._active[Direction.LONG];assert setup.state==state and setup.pullback_depth==pytest.approx(depth)
def test_depth_above_sixty_invalidates_even_if_close_confirms():
    bars=[{"open":103.5,"high":104.4,"low":101.5,"close":104.2}];result,p=execute("LONG",bars);assert not result.intents and p.setup_history[-1].terminal_reason=="INVALID_DEPTH"
def test_bearish_depth_above_sixty_has_priority():
    bars=[{"open":96.5,"high":98.6,"low":95.5,"close":95.7}];result,p=execute("SHORT",bars);assert not result.intents and p.setup_history[-1].terminal_reason=="INVALID_DEPTH"
def test_same_bar_twenty_five_and_confirmation_allowed():
    bars=[{"open":103.5,"high":104.4,"low":102.8,"close":104.2}];result,p=execute("LONG",bars);assert len(result.intents)==1 and result.intents[0].metadata["same_bar_pullback_confirmation"] is True
def test_bearish_same_bar_qualification_and_confirmation_allowed():
    bars=[{"open":96.5,"high":97.2,"low":95.5,"close":95.7}];result,_=execute("SHORT",bars);assert len(result.intents)==1 and result.intents[0].metadata["same_bar_pullback_confirmation"] is True

def pullback_sequence(confirm_on,confirm=True,direction="LONG"):
    rows=[]
    for i in range(1,confirm_on+1):
        if direction=="LONG":rows.append({"open":103.2,"high":103.4,"low":102.9-i*.01,"close":103.2})
        else:rows.append({"open":96.8,"high":97.1+i*.01,"low":96.6,"close":96.8})
    if confirm:
        if direction=="LONG":rows[-1].update(high=103.7,close=103.5)
        else:rows[-1].update(low=96.3,close=96.5)
    return rows
def test_pullback_bar_eight_can_confirm():
    result,p=execute("LONG",pullback_sequence(8));assert len(result.intents)==1 and result.intents[0].metadata["pullback_duration_bars"]==8
def test_bearish_pullback_bar_eight_can_confirm():
    result,_=execute("SHORT",pullback_sequence(8,direction="SHORT"));assert len(result.intents)==1 and result.intents[0].metadata["pullback_duration_bars"]==8
def test_bar_nine_cannot_confirm_after_expiry():
    rows=pullback_sequence(8,False)+[{"open":103.2,"high":103.8,"low":102.8,"close":103.6}];result,p=execute("LONG",rows);assert not result.intents and any(x.terminal_reason=="EXPIRED_TIME" for x in p.setup_history)

@pytest.mark.parametrize("direction",["LONG","SHORT"])
def test_strict_close_confirmation_and_geometry(direction):
    result,_=execute(direction);intent=result.intents[0];assert intent.requested_entry_type==EntryType.MARKET_CLOSE
    if direction=="LONG":assert intent.stop_loss==pytest.approx(102.7) and intent.take_profit==pytest.approx(intent.requested_entry_price+2*(intent.requested_entry_price-intent.stop_loss))
    else:assert intent.stop_loss==pytest.approx(97.3) and intent.take_profit==pytest.approx(intent.requested_entry_price-2*(intent.stop_loss-intent.requested_entry_price))
@pytest.mark.parametrize("kind",["wick","equal"])
def test_bullish_wick_or_equal_close_does_not_confirm(kind):
    close=103.3 if kind=="equal" else 103.2;bars=[{"open":103.5,"high":103.3,"low":102.8,"close":103.0},{"open":103,"high":103.8,"low":102.7,"close":close}];result,_=execute("LONG",bars);assert not result.intents
def test_bearish_wick_only_does_not_confirm():
    bars=[{"open":96.5,"high":97.2,"low":96.7,"close":97},{"open":97,"high":97.3,"low":96.0,"close":96.8}];result,_=execute("SHORT",bars);assert not result.intents
def test_invalid_stop_geometry_no_signal():
    bars=[{"open":103.5,"high":104.4,"low":104.2,"close":104.2}];result,p=execute("LONG",bars);assert not result.intents
def test_context_loss_at_confirmation_invalidates():
    h,m=market_rows();h.loc[len(h)]={"timestamp_utc":pd.Timestamp("2024-01-01 03:00",tz="UTC"),"open":115,"high":116,"low":49,"close":50,"spread":0};result,p=execute(h1=h,m15=m);assert not result.intents and p.setup_history[-1].terminal_reason=="INVALID_CONTEXT_LOST"
def test_frozen_impulse_anchors_do_not_move_during_pullback():
    h,m=market_rows(pullback=[]);expected=impulse_anchors(snapshots(m),Direction.LONG,8);one=[{"open":103.5,"high":103.3,"low":102.8,"close":103.0}];_,p=execute("LONG",one);active=p._active[Direction.LONG];assert (active.impulse_start_timestamp,active.impulse_end_timestamp,active.impulse_low,active.impulse_high)==(expected.start_timestamp,expected.end_timestamp,expected.low,expected.high)

def test_one_setup_one_intent_active_trade_blocks_all_later_signals():
    h,m=market_rows();extra=m.iloc[-1:].copy()
    for i in range(20):
        row=extra.copy();row["timestamp_utc"]=m.timestamp_utc.iloc[-1]+timedelta(minutes=15*(i+1));m=pd.concat([m,row],ignore_index=True)
    result,p=execute(h1=h,m15=m);assert len(result.intents)==1 and p.has_unresolved_position
def test_position_resolution_callback_is_explicit_and_resets():
    _,p=execute();assert p.has_unresolved_position;p.notify_position_resolved();assert not p.has_unresolved_position;p.reset();assert not p.setup_history
def test_active_setup_not_replaced_by_new_momentum_window():
    h,m=market_rows(pullback=[]);m2=pd.concat([m,pd.DataFrame([{"timestamp_utc":m.timestamp_utc.iloc[-1]+timedelta(minutes=15),"open":104,"high":104.7,"low":103.9,"close":104.6,"spread":0}])],ignore_index=True);result,p=execute(h1=h,m15=m2);assert not result.intents and len(p._active)==1
def test_new_setup_not_created_on_signal_decision():
    result,p=execute();assert len(result.intents)==1 and not p._active and p._last_signal_decision==result.intents[0].signal_timestamp
def test_opposite_entry_is_blocked_while_trade_unresolved():
    result,p=execute();assert p.has_unresolved_position and len(result.intents)==1 and all(x.direction==Direction.LONG for x in result.intents)
def test_deterministic_replay_setup_signal_and_ledger():
    a,pa=execute();b,pb=execute();assert pa.setup_history[-1].setup_id==pb.setup_history[-1].setup_id and a.signal_ledger==b.signal_ledger
def test_setup_signal_and_trade_intent_identities_are_distinct():
    result,p=execute();intent=result.intents[0];assert p.setup_history[-1].setup_id!=intent.trade_intent_id and intent.metadata["setup_id"]==p.setup_history[-1].setup_id
def test_s001_state_isolated_from_synthetic_plugin():
    s001=default_registry().resolve("S001");synthetic=default_registry().resolve("SYNTHETIC_INTERFACE_TEST");s001.reset();synthetic.reset();s001._active_trade=True;assert synthetic._seen==0;synthetic._seen=99;assert s001.has_unresolved_position

def test_runtime_hides_unfinished_h1_and_m15_candles():
    h,m=market_rows();result,_=execute(h1=h,m15=m.iloc[:-1]);assert not result.intents
    h_future=h.copy();h_future.loc[len(h_future)]={"timestamp_utc":pd.Timestamp("2024-01-01 04:00",tz="UTC"),"open":1,"high":2,"low":.5,"close":1,"spread":0};result,_=execute(h1=h_future,m15=m.iloc[:-1]);assert not result.intents
def test_stage6_derives_s001_timeframes_from_declared_m1_source():
    source=pd.DataFrame({"timestamp_utc":pd.date_range("2024-01-01",periods=120,freq="min",tz="UTC"),"open":[100]*120,"high":[101]*120,"low":[99]*120,"close":[100]*120,"tick_volume":[1]*120,"spread":[0]*120,"real_volume":[0]*120});built=PartitionService._strategy_timeframe_frames(source,S001Strategy(),"M1");assert len(built["M15"])==8 and len(built["H1"])==2
def test_s001_discovery_context_has_no_partition_or_performance_api():
    h,m=market_rows();plugin=S001Strategy();registry=StrategyRegistry();registry.register(plugin);result=StrategyRuntime(registry).run("S001",{"H1":h,"M15":m},symbol="EURUSD",dataset_id="ROLE-BOUND",experiment_id="TEST",phase="DISCOVERY");assert all(x.dataset_id=="ROLE-BOUND" for x in result.intents) and not hasattr(plugin,"stage7_results") and not hasattr(plugin,"request_partition")
@pytest.mark.parametrize("phase",["VALIDATION","FINAL_TEST"])
def test_direct_protected_strategy_phase_is_denied(phase):
    h,m=market_rows();registry=StrategyRegistry();registry.register(S001Strategy())
    with pytest.raises(ResearchError,match="Stage 6 sealed path"):StrategyRuntime(registry).run("S001",{"H1":h,"M15":m},symbol="EURUSD",dataset_id="UNAUTHORIZED",experiment_id="TEST",phase=phase)
def test_s001_cannot_construct_context_or_access_protected_services():
    now=pd.Timestamp("2024-01-01",tz="UTC").to_pydatetime()
    with pytest.raises(ResearchError):StrategyContext(current_time=now,information_cutoff=now,symbol="EURUSD",phase="DISCOVERY",parameters={},history={},warmup_state={},analysis_timezone="UTC")
def test_nan_inf_candles_fail_before_strategy_math():
    h,m=market_rows();m.loc[0,"close"]=math.nan
    with pytest.raises(ValidationError):execute(h1=h,m15=m)
def test_research_metadata_is_explanatory_not_filtering():
    result,_=execute();meta=result.intents[0].metadata;required={"h1_trend_strength","m15_momentum_strength","impulse_start","pullback_depth_at_confirmation","confirmation_candle_range","hour","day_of_week"};assert required<=set(meta) and len(result.intents)==1
def test_hour_and_day_metadata_do_not_change_eligibility():
    h,m=market_rows();shift=timedelta(hours=7);h2=h.copy();m2=m.copy();h2["timestamp_utc"]+=shift;m2["timestamp_utc"]+=shift;a,_=execute(h1=h,m15=m);b,_=execute(h1=h2,m15=m2);assert len(a.intents)==len(b.intents)==1 and a.intents[0].metadata["hour"]!=b.intents[0].metadata["hour"]
def test_frozen_parameter_mutation_changes_identity():
    registry=default_registry();plugin=registry.resolve("S001");base=registry.fingerprint(plugin,dict(BASELINE_PARAMETERS));changed=dict(BASELINE_PARAMETERS);changed["maximum_pullback_bars"]=12;assert registry.fingerprint(plugin,changed)[0]!=base[0]
def test_stage2_receives_intent_and_remains_authoritative():
    result,_=execute();_,m=market_rows();ledger=BacktestEngine(ExecutionConfig()).run(result.intents,m).ledger;assert ledger[0].trade_intent_id==result.intents[0].trade_intent_id and ledger[0].status in set(PositionStatus)
def test_stage2_same_bar_policy_unchanged():
    assert ExecutionConfig().same_bar_policy.value=="CONSERVATIVE"
def test_no_broker_execution_api_in_s001():
    assert not any(hasattr(S001Strategy(),x) for x in ("order_send","buy","sell","place_order"))
