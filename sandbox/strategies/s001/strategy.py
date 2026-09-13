from __future__ import annotations
import math
from datetime import timezone
from zoneinfo import ZoneInfo
from sandbox.execution.models import Direction,EntryType
from sandbox.research.canonical import sha256_canonical
from sandbox.strategies.models import StrategyMetadata,StrategySignal,WarmupRequirements
from .features import classify_trend,impulse_anchors,qualify_momentum
from .models import SetupRecord,SetupState,TrendContext
from .parameters import BASELINE_PARAMETERS,PARAMETER_SCHEMA,PARAMETER_STATUS

class S001Strategy:
    metadata=StrategyMetadata(strategy_id="S001",strategy_name="Intraday Trend Momentum Pullback Continuation",strategy_version="1.0.0",strategy_family="INTRADAY_TREND_MOMENTUM_PULLBACK_CONTINUATION",description="S001 v1.0 baseline implementation; implementation validity is not evidence of trading edge.",supported_symbols=("EURUSD",),required_base_timeframe="M15",required_timeframes=("H1","M15"),required_data_fields=("open","high","low","close","spread"),warmup_requirements=WarmupRequirements(bars_by_timeframe={"H1":15,"M15":15}),supported_directions=(Direction.LONG,Direction.SHORT),parameter_schema=PARAMETER_SCHEMA,default_parameters=dict(BASELINE_PARAMETERS),signal_schema="S001_SIGNAL_1.0",plugin_version="1.0",capabilities=("LONG","SHORT","MARKET_ENTRY","MULTI_TIMEFRAME","STRUCTURAL_STOP","FIXED_R_TARGET","STATEFUL"),analysis_timezone="America/New_York",synthetic_test_plugin=False)
    atr_convention="WILDER_RMA_TR_WITH_PREVIOUS_CLOSE_V1"
    entry_contract="MARKET_CLOSE"
    parameter_status=PARAMETER_STATUS
    boundary_tolerance=1e-12
    def reset(self):self._active={};self._history=[];self._active_trade=False;self._last_signal_decision=None
    @property
    def setup_history(self):return tuple(self._history)
    @property
    def has_unresolved_position(self):return self._active_trade
    def notify_position_resolved(self):self._active_trade=False
    def _replace(self,setup,**changes):
        updated=setup.model_copy(update=changes);self._active[setup.direction]=updated
        return updated
    def _terminal(self,setup,state,reason,**changes):
        updated=setup.model_copy(update={"state":state,"terminal_reason":reason,**changes});self._history.append(updated);self._active.pop(setup.direction,None);return updated
    def _setup_id(self,direction,anchors,decision):return "S001-SETUP-"+sha256_canonical({"direction":direction.value,"impulse_start":anchors.start_timestamp,"impulse_end":anchors.end_timestamp,"low":anchors.low,"high":anchors.high,"decision":decision})[:24]
    def evaluate(self,context):
        p=context.registered_parameters;h1=context.history("EURUSD","H1",100000);m15=context.history("EURUSD","M15",100000);trend=classify_trend(h1,p["h1_trend_lookback"],p["h1_atr_period"],p["h1_min_displacement_atr"])
        if len(m15)<2:return ()
        current,previous=m15[-1],m15[-2];emitted=[];terminal_this_decision=False
        for direction,setup in list(self._active.items()):
            if current.timestamp_utc<=setup.impulse_end_timestamp:continue
            started=setup.pullback_start_timestamp is not None
            begins=(float(current.values["low"])<float(previous.values["low"])) if direction==Direction.LONG else (float(current.values["high"])>float(previous.values["high"]))
            if not started and not begins:continue
            count=setup.pullback_bars+1 if started else 1
            extreme=(min(setup.pullback_extreme,float(current.values["low"])) if started else float(current.values["low"])) if direction==Direction.LONG else (max(setup.pullback_extreme,float(current.values["high"])) if started else float(current.values["high"]))
            depth=((setup.impulse_high-extreme)/setup.impulse_range) if direction==Direction.LONG else ((extreme-setup.impulse_low)/setup.impulse_range)
            maximum=max(setup.maximum_pullback_depth,depth);state=SetupState.ARMED if depth+self.boundary_tolerance>=p["minimum_pullback"] else SetupState.PULLBACK_MONITORING
            changes={"pullback_start_timestamp":setup.pullback_start_timestamp or current.timestamp_utc,"pullback_extreme":extreme,"pullback_depth":depth,"maximum_pullback_depth":maximum,"pullback_bars":count,"state":state}
            if not math.isfinite(depth) or depth-p["maximum_pullback"]>self.boundary_tolerance:
                self._terminal(setup,SetupState.INVALID,"INVALID_DEPTH",**{k:v for k,v in changes.items() if k!="state"});terminal_this_decision=True;continue
            armed=depth+self.boundary_tolerance>=p["minimum_pullback"];confirms=(float(current.values["close"])>float(previous.values["high"])) if direction==Direction.LONG else (float(current.values["close"])<float(previous.values["low"]))
            same_bar=setup.state!=SetupState.ARMED and armed and confirms
            if armed and confirms:
                required=TrendContext.BULLISH if direction==Direction.LONG else TrendContext.BEARISH
                if trend.context!=required:self._terminal(setup,SetupState.INVALID,"INVALID_CONTEXT_LOST",**{k:v for k,v in changes.items() if k!="state"});terminal_this_decision=True;continue
                entry=float(current.values["close"]);stop=extreme;risk=entry-stop if direction==Direction.LONG else stop-entry
                if not all(math.isfinite(x) for x in (entry,stop,risk)) or risk<=0:
                    self._terminal(setup,SetupState.INVALID,"INVALID_STOP_GEOMETRY",**{k:v for k,v in changes.items() if k!="state"});terminal_this_decision=True;continue
                target=entry+p["reward_risk"]*risk if direction==Direction.LONG else entry-p["reward_risk"]*risk;signal_id="S001-SIGNAL-"+sha256_canonical({"setup_id":setup.setup_id,"decision":context.current_time,"entry":entry,"stop":stop,"target":target})[:24];local=context.current_time.astimezone(ZoneInfo("America/New_York"))
                metadata={"h1_trend_strength":trend.strength,"h1_net_displacement":trend.net_displacement,"h1_atr":trend.atr,"m15_momentum_strength":setup.m15_momentum_strength,"m15_momentum_displacement":setup.m15_momentum_displacement,"m15_atr":setup.m15_atr,"impulse_start":setup.impulse_start_timestamp.isoformat(),"impulse_end":setup.impulse_end_timestamp.isoformat(),"impulse_range":setup.impulse_range,"impulse_high":setup.impulse_high,"impulse_low":setup.impulse_low,"pullback_depth_at_confirmation":depth,"maximum_pullback_depth":maximum,"pullback_duration_bars":count,"pullback_extreme":extreme,"confirmation_candle_range":float(current.values["high"])-float(current.values["low"]),"confirmation_candle_body_size":abs(float(current.values["close"])-float(current.values["open"])),"same_bar_pullback_confirmation":same_bar,"direction":direction.value,"decision_timestamp_utc":context.current_time.isoformat(),"decision_timestamp_america_new_york":local.isoformat(),"hour":local.hour,"day_of_week":local.weekday(),"atr_convention":self.atr_convention,"parameter_status":self.parameter_status}
                emitted.append(StrategySignal(signal_id=signal_id,setup_id=setup.setup_id,symbol="EURUSD",decision_timestamp=context.current_time,information_cutoff=context.information_cutoff,direction=direction,entry_type=EntryType.MARKET_CLOSE,reference_price=entry,stop_price=stop,target_price=target,metadata=metadata));self._terminal(setup,SetupState.TRADED,"SIGNAL_EMITTED",signal_id=signal_id,same_bar_pullback_confirmation=same_bar,**{k:v for k,v in changes.items() if k!="state"});self._active_trade=True;self._last_signal_decision=context.current_time;terminal_this_decision=True;continue
            if count>=p["maximum_pullback_bars"]:self._terminal(setup,SetupState.EXPIRED,"EXPIRED_TIME",**{k:v for k,v in changes.items() if k!="state"});terminal_this_decision=True
            else:self._replace(setup,**changes)
        if self._active_trade or emitted or terminal_this_decision or self._last_signal_decision==context.current_time:return tuple(emitted)
        direction=Direction.LONG if trend.context==TrendContext.BULLISH else Direction.SHORT if trend.context==TrendContext.BEARISH else None
        if direction and direction not in self._active:
            momentum=qualify_momentum(m15,direction,p["m15_momentum_lookback"],p["m15_atr_period"],p["m15_min_displacement_atr"])
            if momentum.qualified:
                anchors=impulse_anchors(m15,direction,p["m15_momentum_lookback"])
                if anchors and anchors.high>anchors.low:
                    setup=SetupRecord(setup_id=self._setup_id(direction,anchors,context.current_time),direction=direction,state=SetupState.WAITING_FOR_PULLBACK,created_at=context.current_time,impulse_start_timestamp=anchors.start_timestamp,impulse_end_timestamp=anchors.end_timestamp,impulse_low=anchors.low,impulse_high=anchors.high,impulse_range=anchors.high-anchors.low,h1_trend_strength=trend.strength,h1_net_displacement=trend.net_displacement,h1_atr=trend.atr,m15_momentum_strength=momentum.strength,m15_momentum_displacement=momentum.displacement,m15_atr=momentum.atr);self._active[direction]=setup
        return tuple(emitted)
