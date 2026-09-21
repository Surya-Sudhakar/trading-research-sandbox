from __future__ import annotations
from datetime import datetime,timezone
from enum import StrEnum
from typing import Any
from pydantic import BaseModel,ConfigDict,Field,field_validator,model_validator
from sandbox.execution.models import Direction,EntryType,TradeIntent
from sandbox.research.canonical import sha256_canonical

class FrozenModel(BaseModel):model_config=ConfigDict(frozen=True,extra="forbid",allow_inf_nan=False)
class ResearchPhase(StrEnum):DISCOVERY="DISCOVERY";VALIDATION="VALIDATION";FINAL_TEST="FINAL_TEST"
class ParameterType(StrEnum):INTEGER="INTEGER";FLOAT="FLOAT";BOOLEAN="BOOLEAN";STRING="STRING"
class ParameterDefinition(FrozenModel):
    name:str=Field(min_length=1);type:ParameterType;material:bool=True;required:bool=True;default:Any=None
class WarmupRequirements(FrozenModel):bars_by_timeframe:dict[str,int]=Field(default_factory=dict)
class StrategyMetadata(FrozenModel):
    strategy_id:str;strategy_name:str;strategy_version:str;strategy_family:str;description:str;supported_symbols:tuple[str,...];required_base_timeframe:str;required_timeframes:tuple[str,...];required_data_fields:tuple[str,...];warmup_requirements:WarmupRequirements;supported_directions:tuple[Direction,...];parameter_schema:tuple[ParameterDefinition,...];default_parameters:dict[str,Any]=Field(default_factory=dict);signal_schema:str="1";plugin_version:str="1";capabilities:tuple[str,...]=();analysis_timezone:str="UTC";synthetic_test_plugin:bool=False
class CandleSnapshot(FrozenModel):
    timestamp_utc:datetime;values:dict[str,float|int]
    @field_validator("timestamp_utc")
    @classmethod
    def utc(cls,v):
        if v.tzinfo is None:raise ValueError("candle timestamp must be timezone-aware")
        return v.astimezone(timezone.utc)
class StrategySignal(FrozenModel):
    signal_id:str;setup_id:str;symbol:str;decision_timestamp:datetime;information_cutoff:datetime;direction:Direction;entry_type:EntryType;reference_price:float;stop_price:float;target_price:float;expires_at:datetime|None=None;metadata:dict[str,Any]=Field(default_factory=dict)
    @field_validator("decision_timestamp","information_cutoff")
    @classmethod
    def utc(cls,v):
        if v.tzinfo is None:raise ValueError("signal timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)
    @model_validator(mode="after")
    def geometry(self):
        if self.information_cutoff>self.decision_timestamp:raise ValueError("information cutoff cannot be after decision timestamp")
        if self.direction==Direction.LONG and not self.stop_price<self.reference_price<self.target_price:raise ValueError("invalid LONG SL/TP geometry")
        if self.direction==Direction.SHORT and not self.target_price<self.reference_price<self.stop_price:raise ValueError("invalid SHORT SL/TP geometry")
        return self
class SignalLedgerRecord(FrozenModel):
    signal_id:str;setup_id:str;strategy_id:str;strategy_version:str;symbol:str;decision_timestamp:datetime;information_cutoff:datetime;direction:Direction;entry_intent:str;entry_reference:float;stop_intent:float;target_intent:float;parameter_fingerprint:str;strategy_code_fingerprint:str;signal_fingerprint:str;metadata:dict[str,Any]=Field(default_factory=dict)
class SignalLedger(FrozenModel):
    records:tuple[SignalLedgerRecord,...];ledger_fingerprint:str
    @classmethod
    def create(cls,records):
        records=tuple(sorted(records,key=lambda x:(x.decision_timestamp,x.signal_id)));return cls(records=records,ledger_fingerprint=sha256_canonical([x.model_dump(mode="json") for x in records]))
class StrategyRun(FrozenModel):
    strategy_fingerprint:str;strategy_code_fingerprint:str;parameter_fingerprint:str;signal_ledger:SignalLedger;intents:tuple[TradeIntent,...];audit_metadata:dict[str,Any]
