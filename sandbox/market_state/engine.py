from __future__ import annotations

import inspect
import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any
import pandas as pd

from sandbox.research.canonical import sha256_canonical
from sandbox.research.errors import ResearchError
from .calendar import calendar_values,time_block_values
from .models import FEATURE_ENGINE_VERSION,FEATURE_SCHEMA_VERSION,FeatureConfiguration,MarketStateSnapshot,SnapshotPhase
from .momentum import change_measurements
from .normalization import safe_ratio,timeframe_minutes
from .structure import range_measurements
from .swings import confirmed_swings,structure_counts
from .trend import horizon_measurements
from .volatility import causal_percentile,mean_median,trailing_atr_series,true_ranges,wilder_atr


_PROTECTED_MARKET_STATE_AUTHORITY=object()


class MarketStateEngine:
    """Pure causal measurement engine. It has no strategy or outcome API."""
    def __init__(self, configuration: FeatureConfiguration | None = None):
        self.configuration=configuration or FeatureConfiguration()
        self.configuration_fingerprint=sha256_canonical(self.configuration.scientific_dict())
        self.code_fingerprint=self._code_fingerprint()

    def _code_fingerprint(self):
        from . import calendar,momentum,normalization,structure,swings,trend,volatility
        material={m.__name__:inspect.getsource(m) for m in (calendar,momentum,normalization,structure,swings,trend,volatility)}
        material[self.__class__.__qualname__]=inspect.getsource(self.__class__)
        return sha256_canonical({"engine_version":FEATURE_ENGINE_VERSION,"implementation":material})

    @staticmethod
    def _canonical_frame(frame: pd.DataFrame, timeframe: str, cutoff: datetime):
        required={"timestamp_utc","open","high","low","close"};missing=required-set(frame.columns)
        if missing:raise ResearchError("MARKET_STATE_REQUIRED_FIELD_UNAVAILABLE: "+",".join(sorted(missing)))
        work=frame[list(required)].copy();work["timestamp_utc"]=pd.to_datetime(work.timestamp_utc,utc=True)
        if work.timestamp_utc.duplicated().any():raise ResearchError("MARKET_STATE_DUPLICATE_TIMESTAMPS")
        work=work.sort_values("timestamp_utc",kind="mergesort").reset_index(drop=True)
        duration=timedelta(minutes=timeframe_minutes(timeframe));cut=pd.Timestamp(cutoff)
        visible=work[work.timestamp_utc+duration<=cut].copy();flags=[]
        if not visible.empty:
            numbers=visible[["open","high","low","close"]].astype(float)
            if not numbers.map(math.isfinite).all().all():raise ResearchError("MARKET_STATE_NON_FINITE_DATA")
            if ((visible.high<visible.low)|(visible.high<visible[["open","close"]].max(axis=1))|(visible.low>visible[["open","close"]].min(axis=1))).any():flags.append(f"{timeframe}:INVALID_OHLC_GEOMETRY_OBSERVED")
        if len(visible)<len(work):flags.append(f"{timeframe}:UNFINISHED_OR_FUTURE_BARS_EXCLUDED")
        return visible,flags

    def snapshot(self,frames:dict[str,pd.DataFrame],*,symbol:str,decision_timestamp:datetime,
        information_cutoff_timestamp:datetime|None=None,base_timeframe:str="M15",phase:SnapshotPhase=SnapshotPhase.ENTRY_DECISION,
        dataset_id:str|None=None,partition_id:str|None=None,research_phase:str="DISCOVERY",_authority=None):
        decision=self._as_utc(decision_timestamp);cutoff=self._as_utc(information_cutoff_timestamp or decision)
        if cutoff>decision:raise ResearchError("MARKET_STATE_FUTURE_CUTOFF_DENIED")
        if research_phase!="DISCOVERY" and _authority is not _PROTECTED_MARKET_STATE_AUTHORITY:
            raise ResearchError("PROTECTED_MARKET_STATE_ACCESS_DENIED: sealed Stage 6 authority required")
        values:dict[str,Any]={};validity={};missing={};quality=[];visible_fingerprints={}
        values.update(calendar_values(decision,self.configuration.analysis_timezone));values.update(time_block_values(decision,self.configuration.time_blocks_minutes))
        for name in values:validity[name]=True
        for timeframe in self.configuration.requested_timeframes:
            if timeframe not in frames:
                quality.append(f"{timeframe}:TIMEFRAME_UNAVAILABLE");continue
            frame,flags=self._canonical_frame(frames[timeframe],timeframe,cutoff);quality.extend(flags)
            visible_fingerprints[timeframe]=sha256_canonical(frame.to_dict(orient="records"))
            self._measure_timeframe(timeframe,frame,values,validity,missing)
        basis={"symbol":symbol,"decision_timestamp":decision,"information_cutoff_timestamp":cutoff,
            "base_timeframe":base_timeframe,"phase":SnapshotPhase(phase),"schema":FEATURE_SCHEMA_VERSION,
            "code_fingerprint":self.code_fingerprint,"configuration_fingerprint":self.configuration_fingerprint,
            "visible_data":visible_fingerprints,"dataset_id":dataset_id,"partition_id":partition_id}
        snapshot_id="MSS-"+sha256_canonical(basis)
        return MarketStateSnapshot(snapshot_id,symbol,decision,cutoff,base_timeframe,SnapshotPhase(phase),FEATURE_SCHEMA_VERSION,
            self.code_fingerprint,self.configuration_fingerprint,values,validity,missing,tuple(sorted(set(quality))),dataset_id,partition_id)

    @staticmethod
    def _as_utc(value):
        value=pd.Timestamp(value)
        if value.tzinfo is None:raise ValueError("timestamps must be timezone-aware")
        return value.tz_convert("UTC").to_pydatetime()

    @staticmethod
    def _put(prefix,data,values,validity,missing):
        for key,value in data.items():
            name=f"{prefix}.{key}"
            if value is None:
                validity[name]=False;missing[name]="undefined_denominator"
            else:values[name]=value;validity[name]=True

    @staticmethod
    def _unavailable(names,validity,missing,reason="insufficient_history"):
        for name in names:validity[name]=False;missing[name]=reason

    def _measure_timeframe(self,tf,frame,values,validity,missing):
        closes=frame.close.astype(float).tolist();highs=frame.high.astype(float).tolist();lows=frame.low.astype(float).tolist()
        atr=wilder_atr(highs,lows,closes,self.configuration.atr_period)
        atr_name=f"{tf}.volatility.atr"
        if atr is None:self._unavailable((atr_name,),validity,missing)
        else:values[atr_name]=atr;validity[atr_name]=True
        if closes and atr is not None:
            self._put(f"{tf}.volatility",{"atr_pct":safe_ratio(atr,closes[-1])},values,validity,missing)
        else:self._unavailable((f"{tf}.volatility.atr_pct",),validity,missing)
        trs=true_ranges(highs,lows,closes)
        returns=[closes[i]/closes[i-1]-1 for i in range(1,len(closes)) if closes[i-1]!=0]
        for period,label in ((self.configuration.volatility_short_window,"short"),(self.configuration.volatility_long_window,"long")):
            if len(trs)>=period:
                mean,median=mean_median(trs[-period:]);self._put(f"{tf}.volatility",{f"tr_mean_{label}":mean,f"tr_median_{label}":median},values,validity,missing)
            else:self._unavailable((f"{tf}.volatility.tr_mean_{label}",f"{tf}.volatility.tr_median_{label}"),validity,missing)
        short=wilder_atr(highs,lows,closes,self.configuration.volatility_short_window);long=wilder_atr(highs,lows,closes,self.configuration.volatility_long_window)
        self._put(f"{tf}.volatility",{"short_long_atr_ratio":safe_ratio(short,long) if short and long else None},values,validity,missing)
        series=trailing_atr_series(highs,lows,closes,self.configuration.atr_period)
        valid_series=[x for x in series if x is not None]
        history=valid_series[-self.configuration.volatility_percentile_history:]
        percentile=causal_percentile(atr,history) if atr is not None and history else None
        self._put(f"{tf}.volatility",{"atr_percentile":percentile},values,validity,missing)
        for horizon in self.configuration.horizons:
            prefix=f"{tf}.h{horizon}";trend=horizon_measurements(closes,horizon,atr)
            expected=("displacement_price","absolute_displacement_price","displacement_pct","displacement_atr","regression_slope_price_per_bar","regression_slope_atr","directional_efficiency")
            if trend:self._put(prefix,trend,values,validity,missing)
            else:self._unavailable(tuple(f"{prefix}.{x}" for x in expected),validity,missing)
            ranges=range_measurements(highs,lows,closes,horizon,atr)
            if ranges:self._put(prefix,ranges,values,validity,missing)
            else:self._unavailable(tuple(f"{prefix}.{x}" for x in ("rolling_high","rolling_low","rolling_range_price","rolling_range_atr","close_position_in_range","distance_from_rolling_high_price","distance_from_rolling_low_price","distance_from_rolling_high_atr","distance_from_rolling_low_atr","bars_since_rolling_high","bars_since_rolling_low")),validity,missing)
            change=change_measurements(closes,horizon,atr)
            if change:self._put(f"{prefix}.momentum_change",change,values,validity,missing)
            else:self._unavailable((f"{prefix}.momentum_change.recent_displacement_price",),validity,missing)
            if len(returns)>=horizon:
                recent_returns=returns[-horizon:]
                self._put(prefix,{"mean_candle_return":statistics.fmean(recent_returns),"median_candle_return":statistics.median(recent_returns),
                    "realized_close_volatility":statistics.pstdev(recent_returns)},values,validity,missing)
            else:self._unavailable(tuple(f"{prefix}.{x}" for x in ("mean_candle_return","median_candle_return","realized_close_volatility")),validity,missing)
        swings=confirmed_swings(highs,lows,self.configuration.swing_left_bars,self.configuration.swing_right_bars)
        values.update({f"{tf}.structure.{k}":v for k,v in structure_counts(swings,self.configuration.swing_count_window).items()})
        for k in structure_counts(swings,self.configuration.swing_count_window):validity[f"{tf}.structure.{k}"]=True
        for kind in ("HIGH","LOW"):
            candidates=[x for x in swings if x.kind==kind];prefix=f"{tf}.swing.{kind.lower()}"
            if not candidates:self._unavailable((f"{prefix}.price",f"{prefix}.distance_price",f"{prefix}.distance_atr",f"{prefix}.distance_pct",f"{prefix}.bars_since_confirmation",f"{prefix}.current_above"),validity,missing);continue
            swing=candidates[-1];distance=float(closes[-1])-swing.price
            self._put(prefix,{"price":swing.price,"distance_price":distance,"distance_atr":safe_ratio(distance,atr) if atr else None,
                "distance_pct":safe_ratio(distance,swing.price),"bars_since_confirmation":len(closes)-1-swing.confirmation_index,"current_above":float(closes[-1])>swing.price},values,validity,missing)

    def stage7_results(self,*_args,**_kwargs):raise ResearchError("MARKET_STATE_STAGE7_ACCESS_DENIED")
    def request_partition(self,*_args,**_kwargs):raise ResearchError("MARKET_STATE_PARTITION_ACCESS_DENIED")
