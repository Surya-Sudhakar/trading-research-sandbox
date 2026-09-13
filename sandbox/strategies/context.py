from __future__ import annotations
from datetime import datetime,timezone
from types import MappingProxyType
from zoneinfo import ZoneInfo
import pandas as pd
from sandbox.research.errors import ResearchError
from .models import CandleSnapshot,ResearchPhase

class StrategyContext:
    """Causal value API. It contains immutable snapshots truncated at its cutoff, never source frames or paths."""
    __slots__=("_current_time","_cutoff","_symbol","_phase","_parameters","_history","_warmup","_timezone","_sealed")
    def __init__(self,*,current_time:datetime,information_cutoff:datetime,symbol:str,phase:ResearchPhase,parameters:dict,history:dict[str,tuple[CandleSnapshot,...]],warmup_state:dict[str,bool],analysis_timezone:str,_authority=None):
        if _authority is not _CONTEXT_AUTHORITY:raise ResearchError("STRATEGY_CONTEXT_CREATION_DENIED: trusted runtime required")
        object.__setattr__(self,"_sealed",False);object.__setattr__(self,"_current_time",current_time.astimezone(timezone.utc));object.__setattr__(self,"_cutoff",information_cutoff.astimezone(timezone.utc));object.__setattr__(self,"_symbol",symbol);object.__setattr__(self,"_phase",ResearchPhase(phase));object.__setattr__(self,"_parameters",MappingProxyType(dict(parameters)));object.__setattr__(self,"_history",MappingProxyType({k:tuple(v) for k,v in history.items()}));object.__setattr__(self,"_warmup",MappingProxyType(dict(warmup_state)));object.__setattr__(self,"_timezone",ZoneInfo(analysis_timezone));object.__setattr__(self,"_sealed",True)
    def __setattr__(self,n,v):
        if getattr(self,"_sealed",False):raise TypeError("StrategyContext is read-only")
        object.__setattr__(self,n,v)
    @property
    def current_time(self):return self._current_time
    @property
    def information_cutoff(self):return self._cutoff
    @property
    def current_time_local(self):return self._current_time.astimezone(self._timezone)
    @property
    def registered_parameters(self):return self._parameters
    @property
    def warmup_state(self):return self._warmup
    @property
    def research_phase(self):return self._phase
    def history(self,symbol:str,timeframe:str,bars:int)->tuple[CandleSnapshot,...]:
        if symbol!=self._symbol:raise ResearchError("STRATEGY_SYMBOL_ACCESS_DENIED")
        if bars<1:raise ValueError("bars must be positive")
        if timeframe not in self._history:raise ResearchError("STRATEGY_TIMEFRAME_ACCESS_DENIED")
        values=self._history[timeframe]
        if any(x.timestamp_utc>self._cutoff for x in values):raise ResearchError("FUTURE_DATA_ACCESS_DENIED")
        return values[-bars:]
    def current_closed_bar(self,symbol,timeframe):
        values=self.history(symbol,timeframe,1);return values[-1] if values else None
    def previous_closed_bar(self,symbol,timeframe):
        values=self.history(symbol,timeframe,2);return values[-2] if len(values)>1 else None
    def request_partition(self,*_args,**_kwargs):raise ResearchError("PROTECTED_PARTITION_LOOKUP_DENIED")
    def stage7_results(self,*_args,**_kwargs):raise ResearchError("PERFORMANCE_ACCESS_DENIED_DURING_SIGNAL_GENERATION")
_CONTEXT_AUTHORITY=object()
