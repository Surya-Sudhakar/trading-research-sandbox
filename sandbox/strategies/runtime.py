from __future__ import annotations
import re
from datetime import timedelta
import pandas as pd
from sandbox.execution.models import EntryType,TradeIntent
from sandbox.research.canonical import sha256_canonical
from sandbox.research.errors import ResearchError
from .context import StrategyContext,_CONTEXT_AUTHORITY
from .models import CandleSnapshot,ResearchPhase,SignalLedger,SignalLedgerRecord,StrategyRun,StrategySignal
from .registry import StrategyRegistry
_PROTECTED_PHASE_AUTHORITY=object()

class StrategyRuntime:
    ALLOWED_ENTRY_TYPES={EntryType.MARKET_NEXT_OPEN,EntryType.MARKET_CLOSE}
    def __init__(self,registry:StrategyRegistry):self.registry=registry
    def _parameters(self,p,provided):
        values=dict(p.metadata.default_parameters);values.update(provided);self.registry.validate_parameters(p,values);return values
    @staticmethod
    def _duration(timeframe):
        match=re.fullmatch(r"([MHD])(\d+)",timeframe)
        if not match:raise ResearchError("UNSUPPORTED_STRATEGY_TIMEFRAME: "+timeframe)
        factor={"M":1,"H":60,"D":1440}[match.group(1)];return timedelta(minutes=factor*int(match.group(2)))
    def _canonical(self,frames,metadata):
        result={}
        for tf in metadata.required_timeframes:
            if tf not in frames:raise ResearchError("REQUIRED_TIMEFRAME_UNAVAILABLE: "+tf)
            frame=frames[tf].copy();missing=(set(metadata.required_data_fields)|{"timestamp_utc"})-set(frame.columns)
            if missing:raise ResearchError("REQUIRED_DATA_FIELD_UNAVAILABLE: "+",".join(sorted(missing)))
            frame["timestamp_utc"]=pd.to_datetime(frame.timestamp_utc,utc=True)
            if frame.timestamp_utc.duplicated().any():raise ResearchError("DUPLICATE_STRATEGY_TIMESTAMPS")
            result[tf]=frame.sort_values("timestamp_utc",kind="mergesort").reset_index(drop=True)
        return result
    def run(self,strategy_id,frames,*,symbol,dataset_id,experiment_id,parameters=None,phase=ResearchPhase.DISCOVERY,version=None,_phase_authority=None):
        plugin=self.registry.resolve(strategy_id,version);m=plugin.metadata
        if symbol not in m.supported_symbols and "*" not in m.supported_symbols:raise ResearchError("UNSUPPORTED_STRATEGY_SYMBOL")
        phase=ResearchPhase(phase)
        if phase!=ResearchPhase.DISCOVERY and _phase_authority is not _PROTECTED_PHASE_AUTHORITY:raise ResearchError("PROTECTED_STRATEGY_PHASE_DENIED: Stage 6 sealed path required")
        values=self._parameters(plugin,parameters or {});canonical=self._canonical(frames,m);fingerprint,code_fp,param_fp=self.registry.fingerprint(plugin,values)
        base=canonical[m.required_base_timeframe];decisions=list(base.timestamp_utc+self._duration(m.required_base_timeframe));records=[];intents=[];plugin.reset()
        for decision in decisions:
            history={};warm={}
            for tf,frame in canonical.items():
                visible=frame[frame.timestamp_utc+self._duration(tf)<=decision];required=m.warmup_requirements.bars_by_timeframe.get(tf,0);warm[tf]=len(visible)>=required
                history[tf]=tuple(CandleSnapshot(timestamp_utc=row.timestamp_utc.to_pydatetime(),values={field:getattr(row,field) for field in m.required_data_fields}) for row in visible.itertuples(index=False))
            if not all(warm.values()):continue
            context=StrategyContext(current_time=decision.to_pydatetime(),information_cutoff=decision.to_pydatetime(),symbol=symbol,phase=phase,parameters=values,history=history,warmup_state=warm,analysis_timezone=m.analysis_timezone,_authority=_CONTEXT_AUTHORITY)
            emitted=plugin.evaluate(context)
            if emitted is None:raise ResearchError("MALFORMED_STRATEGY_SIGNAL")
            for raw in emitted:
                try:s=raw if isinstance(raw,StrategySignal) else StrategySignal.model_validate(raw)
                except Exception as exc:raise ResearchError("MALFORMED_STRATEGY_SIGNAL") from exc
                if s.decision_timestamp!=context.current_time or s.information_cutoff>context.information_cutoff:raise ResearchError("FUTURE_SIGNAL_TIMESTAMP_DENIED")
                if s.entry_type not in self.ALLOWED_ENTRY_TYPES:raise ResearchError("UNSUPPORTED_STRATEGY_ENTRY_TYPE")
                if s.direction not in m.supported_directions:raise ResearchError("UNSUPPORTED_STRATEGY_DIRECTION")
                sf=sha256_canonical(s.model_dump(mode="json"));records.append(SignalLedgerRecord(signal_id=s.signal_id,setup_id=s.setup_id,strategy_id=m.strategy_id,strategy_version=m.strategy_version,symbol=s.symbol,decision_timestamp=s.decision_timestamp,information_cutoff=s.information_cutoff,direction=s.direction,entry_intent=s.entry_type.value,entry_reference=s.reference_price,stop_intent=s.stop_price,target_intent=s.target_price,parameter_fingerprint=param_fp,strategy_code_fingerprint=code_fp,signal_fingerprint=sf,metadata=s.metadata))
                intents.append(TradeIntent(trade_intent_id=s.signal_id,strategy_id=m.strategy_id,experiment_id=experiment_id,dataset_id=dataset_id,symbol=s.symbol,direction=s.direction,signal_timestamp=s.decision_timestamp,requested_entry_type=s.entry_type,requested_entry_price=s.reference_price if s.entry_type==EntryType.MARKET_CLOSE else None,stop_loss=s.stop_price,take_profit=s.target_price,metadata={**s.metadata,"setup_id":s.setup_id,"strategy_version":m.strategy_version,"information_cutoff":s.information_cutoff.isoformat(),"signal_fingerprint":sf,"parameter_fingerprint":param_fp,"strategy_code_fingerprint":code_fp}))
        ledger=SignalLedger.create(records);audit={"strategy_id":m.strategy_id,"strategy_version":m.strategy_version,"strategy_code_fingerprint":code_fp,"parameter_fingerprint":param_fp,"signal_ledger_fingerprint":ledger.ledger_fingerprint,"information_cutoff_policy":"CAUSAL_LEQ_DECISION_TIME","parameter_count":len(values),"phase":phase.value}
        return StrategyRun(strategy_fingerprint=fingerprint,strategy_code_fingerprint=code_fp,parameter_fingerprint=param_fp,signal_ledger=ledger,intents=tuple(intents),audit_metadata=audit)
    def assert_deterministic(self,*args,**kwargs):
        first=self.run(*args,**kwargs);second=self.run(*args,**kwargs)
        if first.signal_ledger.ledger_fingerprint!=second.signal_ledger.ledger_fingerprint:raise ResearchError("NONDETERMINISTIC_STRATEGY_SIGNAL_STREAM")
        return first
    @staticmethod
    def assert_frozen_binding(candidate,run):
        spec=candidate.strategy_spec
        expected={"strategy_code_fingerprint":run.strategy_code_fingerprint,"parameter_fingerprint":run.parameter_fingerprint}
        if any(spec.get(k)!=v for k,v in expected.items()):raise ResearchError("FROZEN_CANDIDATE_STRATEGY_FINGERPRINT_MISMATCH")
        return True
