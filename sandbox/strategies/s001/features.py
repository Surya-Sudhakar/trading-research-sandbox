from __future__ import annotations
import math
from sandbox.execution.models import Direction
from sandbox.strategies.models import CandleSnapshot
from .models import ImpulseAnchors,MomentumMeasurement,TrendContext,TrendMeasurement

def true_ranges(bars):
    values=[]
    for previous,current in zip(bars,bars[1:]):
        h=float(current.values["high"]);l=float(current.values["low"]);pc=float(previous.values["close"]);tr=max(h-l,abs(h-pc),abs(l-pc))
        if not math.isfinite(tr) or tr<0:return ()
        values.append(tr)
    return tuple(values)
def wilder_atr(bars,period):
    trs=true_ranges(bars)
    if len(trs)<period:return None
    atr=sum(trs[:period])/period
    for tr in trs[period:]:atr=((period-1)*atr+tr)/period
    return atr if math.isfinite(atr) and atr>0 else None
def classify_trend(bars,lookback,period,minimum):
    atr=wilder_atr(bars,period)
    if len(bars)<max(lookback,period+1) or atr is None:return TrendMeasurement(context=TrendContext.NO_TREND,net_displacement=0.0,atr=atr,strength=None)
    window=bars[-lookback:];displacement=float(window[-1].values["close"])-float(window[0].values["open"]);strength=abs(displacement)/atr
    context=TrendContext.BULLISH if displacement>0 and strength>=minimum else TrendContext.BEARISH if displacement<0 and strength>=minimum else TrendContext.NO_TREND
    return TrendMeasurement(context=context,net_displacement=displacement,atr=atr,strength=strength)
def qualify_momentum(bars,direction,lookback,period,minimum):
    atr=wilder_atr(bars,period)
    if len(bars)<max(lookback,period+1) or atr is None:return MomentumMeasurement(qualified=False,displacement=0.0,atr=atr,strength=None)
    window=bars[-lookback:];displacement=float(window[-1].values["close"])-float(window[0].values["open"]);strength=abs(displacement)/atr;correct=displacement>0 if direction==Direction.LONG else displacement<0
    return MomentumMeasurement(qualified=bool(correct and strength>=minimum),displacement=displacement,atr=atr,strength=strength)
def impulse_anchors(bars,direction,lookback):
    if len(bars)<lookback:return None
    window=bars[-lookback:]
    if direction==Direction.LONG:
        lows=[float(x.values["low"]) for x in window]
        for i in sorted(range(len(window)-1),key=lambda j:(lows[j],j)):
            later=window[i+1:]
            if later:
                high=max(float(x.values["high"]) for x in later);j=i+1+next(k for k,x in enumerate(later) if float(x.values["high"])==high)
                if high>lows[i]:return ImpulseAnchors(start_timestamp=window[i].timestamp_utc,end_timestamp=window[j].timestamp_utc,low=lows[i],high=high)
    else:
        highs=[float(x.values["high"]) for x in window]
        for i in sorted(range(len(window)-1),key=lambda j:(-highs[j],j)):
            later=window[i+1:]
            if later:
                low=min(float(x.values["low"]) for x in later);j=i+1+next(k for k,x in enumerate(later) if float(x.values["low"])==low)
                if highs[i]>low:return ImpulseAnchors(start_timestamp=window[i].timestamp_utc,end_timestamp=window[j].timestamp_utc,low=low,high=highs[i])
    return None
