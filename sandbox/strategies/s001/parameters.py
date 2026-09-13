from __future__ import annotations
from types import MappingProxyType
from sandbox.strategies.models import ParameterDefinition,ParameterType

BASELINE_PARAMETERS=MappingProxyType({
    "h1_trend_lookback":12,
    "h1_atr_period":14,
    "h1_min_displacement_atr":1.50,
    "m15_momentum_lookback":8,
    "m15_atr_period":14,
    "m15_min_displacement_atr":1.25,
    "minimum_pullback":0.25,
    "maximum_pullback":0.60,
    "maximum_pullback_bars":8,
    "reward_risk":2.0,
})
PARAMETER_SCHEMA=tuple(ParameterDefinition(name=name,type=ParameterType.INTEGER if isinstance(value,int) else ParameterType.FLOAT,material=True,required=True,default=value) for name,value in BASELINE_PARAMETERS.items())
PARAMETER_STATUS={name:("FIXED_INITIAL_RESEARCH" if name in {"h1_atr_period","m15_atr_period","reward_risk"} else "BASELINE_ASSUMPTION") for name in BASELINE_PARAMETERS}
FUTURE_DISCOVERY_ALTERNATIVES=MappingProxyType({"h1_trend_lookback":(8,12,16),"h1_min_displacement_atr":(1.0,1.5,2.0),"m15_momentum_lookback":(4,8,12),"m15_min_displacement_atr":(0.75,1.25,1.75),"minimum_pullback":(0.20,0.25,0.33),"maximum_pullback":(0.50,0.60,0.67),"maximum_pullback_bars":(4,8,12)})
