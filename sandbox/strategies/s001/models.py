from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel,ConfigDict
from sandbox.execution.models import Direction

class FrozenModel(BaseModel):model_config=ConfigDict(frozen=True,extra="forbid",allow_inf_nan=False)
class TrendContext(StrEnum):BULLISH="BULLISH";BEARISH="BEARISH";NO_TREND="NO_TREND"
class SetupState(StrEnum):NO_SETUP="NO_SETUP";WAITING_FOR_PULLBACK="WAITING_FOR_PULLBACK";PULLBACK_MONITORING="PULLBACK_MONITORING";ARMED="ARMED";TRADED="TRADED";INVALID="INVALID";EXPIRED="EXPIRED"
class SetupRecord(FrozenModel):
    setup_id:str;direction:Direction;state:SetupState;created_at:datetime;impulse_start_timestamp:datetime;impulse_end_timestamp:datetime;impulse_low:float;impulse_high:float;impulse_range:float;h1_trend_strength:float;h1_net_displacement:float;h1_atr:float;m15_momentum_strength:float;m15_momentum_displacement:float;m15_atr:float;pullback_start_timestamp:datetime|None=None;pullback_extreme:float|None=None;pullback_depth:float=0.0;maximum_pullback_depth:float=0.0;pullback_bars:int=0;same_bar_pullback_confirmation:bool=False;terminal_reason:str|None=None;signal_id:str|None=None
class TrendMeasurement(FrozenModel):context:TrendContext;net_displacement:float;atr:float|None;strength:float|None
class MomentumMeasurement(FrozenModel):qualified:bool;displacement:float;atr:float|None;strength:float|None
class ImpulseAnchors(FrozenModel):start_timestamp:datetime;end_timestamp:datetime;low:float;high:float
