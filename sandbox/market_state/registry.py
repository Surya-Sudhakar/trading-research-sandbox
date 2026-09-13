from __future__ import annotations
from .models import FeatureConfiguration,FeatureDefinition


class FeatureRegistry:
    def __init__(self, configuration: FeatureConfiguration):
        self.configuration=configuration;self._definitions={}
        self._build()
    def _add(self,name,definition,units,timeframe,lookback,deps):
        if name in self._definitions:raise ValueError("duplicate feature: "+name)
        self._definitions[name]=FeatureDefinition(name,definition,units,timeframe,lookback,"after every required input bar has completed","UNAVAILABLE: insufficient_history or undefined_denominator",tuple(deps))
    def _build(self):
        for tf in self.configuration.requested_timeframes:
            volatility=(("atr","Wilder RMA true range","price"),("atr_pct","ATR divided by close","ratio"),("tr_mean_short","short-window true-range mean","price"),("tr_median_short","short-window true-range median","price"),("tr_mean_long","long-window true-range mean","price"),("tr_median_long","long-window true-range median","price"),("short_long_atr_ratio","short ATR divided by long ATR","ratio"),("atr_percentile","inclusive weak rank within trailing causal ATR observations","percent"))
            for key,definition,unit in volatility:self._add(f"{tf}.volatility.{key}",definition,unit,tf,"configured ATR/volatility windows",("atr_period","volatility_short_window","volatility_long_window","volatility_percentile_history"))
            for h in self.configuration.horizons:
                prefix=f"{tf}.h{h}"
                for key,definition,unit in (("displacement_price","end close minus start close","price"),("absolute_displacement_price","absolute end-to-start close movement","price"),("displacement_pct","displacement divided by start close","ratio"),("displacement_atr","displacement divided by current ATR","ATR"),("regression_slope_price_per_bar","OLS close slope","price/bar"),("regression_slope_atr","OLS close slope divided by ATR","ATR/bar"),("directional_efficiency","absolute displacement divided by path length","ratio"),("rolling_high","maximum high","price"),("rolling_low","minimum low","price"),("rolling_range_price","rolling high minus low","price"),("rolling_range_atr","rolling range divided by ATR","ATR"),("close_position_in_range","close location in rolling range","ratio"),("distance_from_rolling_high_price","rolling high minus close","price"),("distance_from_rolling_low_price","close minus rolling low","price"),("distance_from_rolling_high_atr","high distance divided by ATR","ATR"),("distance_from_rolling_low_atr","low distance divided by ATR","ATR"),("bars_since_rolling_high","bars since most recent rolling high","bars"),("bars_since_rolling_low","bars since most recent rolling low","bars"),("mean_candle_return","mean close-to-close return","ratio"),("median_candle_return","median close-to-close return","ratio"),("realized_close_volatility","population standard deviation of close returns","ratio")):
                    self._add(f"{prefix}.{key}",definition,unit,tf,f"{h} bars",("horizons",))
                for key,definition,unit in (("recent_displacement_price","recent-window displacement","price"),("preceding_displacement_price","preceding equal-window displacement","price"),("recent_to_preceding_displacement_ratio","recent divided by preceding displacement","ratio"),("recent_minus_preceding_displacement_atr","displacement change divided by ATR","ATR"),("recent_regression_slope","recent OLS slope","price/bar"),("preceding_regression_slope","preceding OLS slope","price/bar"),("recent_to_preceding_slope_ratio","recent divided by preceding slope","ratio"),("displacement_sign_changed","whether nonzero displacement signs differ","boolean")):
                    self._add(f"{prefix}.momentum_change.{key}",definition,unit,tf,f"{2*h+1} closes",("horizons","atr_period"))
            for key in ("higher_high_count","lower_high_count","higher_low_count","lower_low_count"):
                self._add(f"{tf}.structure.{key}","comparison count across causally confirmed pivots","count",tf,"configured confirmed swings",("swing_left_bars","swing_right_bars","swing_count_window"))
            for kind in ("high","low"):
                for key,unit in (("price","price"),("distance_price","price"),("distance_atr","ATR"),("distance_pct","ratio"),("bars_since_confirmation","bars"),("current_above","boolean")):
                    self._add(f"{tf}.swing.{kind}.{key}",f"latest causally confirmed swing {kind} {key.replace('_',' ')}",unit,tf,"left + pivot + right bars",("swing_left_bars","swing_right_bars"))
        for name,definition,unit in (("calendar.utc_timestamp","UTC decision timestamp","timestamp"),("calendar.new_york_timestamp","IANA localized decision timestamp","timestamp"),("calendar.utc_hour","UTC hour","hour"),("calendar.utc_minute","UTC minute","minute"),("calendar.day_of_week","UTC weekday Monday=0","integer"),("calendar.month","UTC calendar month","integer"),("calendar.new_york_hour","IANA America/New_York local hour","hour"),("calendar.new_york_minute","IANA America/New_York local minute","minute")):
            self._add(name,definition,unit,"CLOCK","none",("analysis_timezone",))
        for size in self.configuration.time_blocks_minutes:
            self._add(f"calendar.block_{size}m_elapsed_minutes","known elapsed minutes within UTC clock block","minutes","CLOCK","none",("time_blocks_minutes",))
            self._add(f"calendar.block_{size}m_elapsed_fraction","known elapsed fraction within UTC clock block","ratio","CLOCK","none",("time_blocks_minutes",))
    def definitions(self):return tuple(self._definitions[k] for k in sorted(self._definitions))
    def get(self,name):return self._definitions[name]
    def validate(self):return bool(self._definitions) and all(x.feature_name for x in self._definitions.values())
