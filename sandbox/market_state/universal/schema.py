from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class FeatureRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    timestamp: datetime
    symbol: str
    decision_timeframe: str
    feature_ready: bool
    candle_range: float
    body_size: float
    body_ratio: float | None
    upper_wick: float
    lower_wick: float
    body_midpoint: float
    close_location: float | None
    candle_direction: int
    atr_14: float | None
    atr_percentile_100: float | None
    range_percentile_100: float | None
    displacement_4_atr: float | None
    displacement_8_atr: float | None
    er_4: float | None
    er_16: float | None
    rv_16: float | None
    atr_change_1: float | None
    atr_change_8: float | None
    slope_atr_4: float | None
    slope_atr_16: float | None
    range_pos_16: float | None
    prior_high_12: float | None
    prior_low_12: float | None
    breakout_high_12: bool | None
    breakout_low_12: bool | None
    completed_h1_direction: int | None
    completed_h3_direction: int | None
    analysis_hour: int
    weekday: int
    london_active: bool
    new_york_active: bool
    london_new_york_overlap: bool


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    category: str
    dtype: str
    timeframe: str
    lookback: int
    description: str
    version: int = 1


def definitions(decision_timeframe="M15") -> tuple[FeatureDefinition, ...]:
    specs = [
        ("candle_range", "candle", "float", 1, "high-low"),
        ("body_size", "candle", "float", 1, "abs(close-open)"),
        ("body_ratio", "candle", "float", 1, "body_size/candle_range; null if zero range"),
        ("upper_wick", "candle", "float", 1, "high-max(open,close)"),
        ("lower_wick", "candle", "float", 1, "min(open,close)-low"),
        ("body_midpoint", "candle", "float", 1, "(open+close)/2"),
        ("close_location", "candle", "float", 1, "(close-low)/range; null if zero range"),
        ("candle_direction", "candle", "int", 1, "sign(close-open)"),
        ("atr_14", "volatility", "float", 15, "Wilder ATR; 14 TR mean seed; first bar has no TR; nonpositive ATR null"),
        ("atr_percentile_100", "volatility", "float", 114, "100*count(ATR<=current)/100 over 100 valid trailing ATRs including current"),
        ("range_percentile_100", "volatility", "float", 100, "100*count(range<=current)/100 over 100 trailing bars including current"),
        ("displacement_4_atr", "momentum", "float", 15, "(close[t]-close[t-4])/ATR[t]"),
        ("displacement_8_atr", "momentum", "float", 15, "(close[t]-close[t-8])/ATR[t]"),
        ("er_4", "state_efficiency", "float", 5, "abs(C[t]-C[t-4])/sum(abs(dC)) over 4 intervals"),
        ("er_16", "state_efficiency", "float", 17, "abs(C[t]-C[t-16])/sum(abs(dC)) over 16 intervals"),
        ("rv_16", "state_volatility", "float", 17, "sqrt(sum(log(C[i]/C[i-1])^2)) over 16 intervals; not annualized"),
        ("atr_change_1", "state_volatility_transition", "float", 16, "ATR14[t]/ATR14[t-1]-1"),
        ("atr_change_8", "state_volatility_transition", "float", 23, "ATR14[t]/ATR14[t-8]-1"),
        ("slope_atr_4", "state_trend", "float", 15, "OLS close slope over last 4 closes / ATR14[t]"),
        ("slope_atr_16", "state_trend", "float", 16, "OLS close slope over last 16 closes / ATR14[t]"),
        ("range_pos_16", "state_structure", "float", 17, "(close-prior_low_16)/(prior_high_16-prior_low_16); current excluded"),
        ("prior_high_12", "structure", "float", 13, "max(high[t-12:t]); current excluded"),
        ("prior_low_12", "structure", "float", 13, "min(low[t-12:t]); current excluded"),
        ("breakout_high_12", "structure", "bool", 13, "close>prior_high_12; equality false"),
        ("breakout_low_12", "structure", "bool", 13, "close<prior_low_12; equality false"),
        ("completed_h1_direction", "context", "int", 1, "sign(close-open) of latest complete H1 with end<=decision"),
        ("completed_h3_direction", "context", "int", 1, "sign(close-open) of latest complete H3 with end<=decision"),
        ("analysis_hour", "time", "int", 1, "decision close timestamp hour in configured analysis timezone"),
        ("weekday", "time_context", "int", 1, "UTC decision-close weekday, Monday=0"),
        ("london_active", "time_context", "bool", 1, "decision close is in 08:00<=Europe/London local time<17:00"),
        ("new_york_active", "time_context", "bool", 1, "decision close is in 08:00<=America/New_York local time<17:00"),
        ("london_new_york_overlap", "time_context", "bool", 1, "London-active AND New-York-active at decision close"),
    ]
    return tuple(FeatureDefinition(n, c, d, "H1" if n == "completed_h1_direction" else
                 "H3" if n == "completed_h3_direction" else decision_timeframe, lb, desc)
                 for n, c, d, lb, desc in specs)

