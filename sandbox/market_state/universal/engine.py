from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sandbox.models import Candle
from sandbox.market_state.models import FeatureConfiguration
from sandbox.market_state.normalization import timeframe_minutes
from .candle import geometry
from .volatility import atr_series, percentile_series
from .momentum import displacement
from .structure import prior_structure
from .market_metrics import (
    atr_change, efficiency_ratio, normalized_regression_slope,
    prior_range_position, realized_volatility,
)
from .multitimeframe import completed_context, completed_state_context
from .schema import FeatureRow, definitions
from .continuity import QualityContext


class UniversalFeatureEngine:
    """Compute one immutable row per completed bar, labeled by its UTC close.

    Input timestamps are bar opens. Omit as_of only when the caller certifies
    that every input bar is completed. Histories must belong to one authorized
    research partition; this pure API does not load or authorize datasets.
    """
    def __init__(self, decision_timeframe="M15", analysis_timezone=None):
        minutes = timeframe_minutes(decision_timeframe)
        if minutes < 1 or any(target % minutes for target in (60, 180)):
            raise ValueError("decision timeframe must divide H1 and H3")
        self.decision_timeframe = decision_timeframe
        self.analysis_timezone = analysis_timezone or FeatureConfiguration().analysis_timezone
        self._timezone = ZoneInfo(self.analysis_timezone)
        self._london_timezone = ZoneInfo("Europe/London")
        self._new_york_timezone = ZoneInfo("America/New_York")
        self._duration = timedelta(minutes=int(minutes))
        self._frequency = f"{int(minutes)}min"

    @property
    def definitions(self):
        return definitions(self.decision_timeframe)

    def compute(self, bars: pd.DataFrame | Iterable[Candle], *, symbol: str,
                as_of: datetime | None = None, quality_context: QualityContext | None = None) -> tuple[FeatureRow, ...]:
        if not symbol.strip():
            raise ValueError("symbol is required")
        frame = bars.copy() if isinstance(bars, pd.DataFrame) else pd.DataFrame(
            [b.model_dump() for b in bars])
        if frame.empty:
            return ()
        if quality_context is not None:
            return tuple(row for segment in quality_context.split(frame)
                         for row in self.compute(segment, symbol=symbol, as_of=as_of))
        required = ["timestamp_utc", "open", "high", "low", "close"]
        if set(required)-set(frame):
            raise ValueError("missing timestamp/OHLC columns")
        frame = frame[required].copy()
        times = [pd.Timestamp(t) for t in frame.timestamp_utc]
        if any(pd.isna(t) or t.tzinfo is None for t in times):
            raise ValueError("timestamps must be valid and timezone-aware")
        frame["timestamp_utc"] = pd.to_datetime(times, utc=True)
        if frame.timestamp_utc.duplicated().any() or not frame.timestamp_utc.is_monotonic_increasing:
            raise ValueError("bars must be strictly ordered without duplicates")
        if (frame.timestamp_utc != frame.timestamp_utc.dt.floor(self._frequency)).any():
            raise ValueError("bar opens must align with the decision timeframe UTC grid")
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            if pd.isna(cutoff) or cutoff.tzinfo is None:
                raise ValueError("as_of must be timezone-aware")
            frame = frame[frame.timestamp_utc+self._duration <= cutoff]
        frame = frame.reset_index(drop=True)
        if frame.empty:
            return ()
        prices = frame[required[1:]].astype(float)
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
            raise ValueError("OHLC must be finite and positive")
        o, h, l, c = (prices[k].tolist() for k in required[1:])
        if any(hi < max(op, cl) or lo > min(op, cl) or hi < lo
               for op, hi, lo, cl in zip(o, h, l, c)):
            raise ValueError("invalid OHLC relationship")
        frame[required[1:]] = prices
        contexts = completed_context(frame, self.decision_timeframe, ("H1", "H3"))
        state_contexts = completed_state_context(frame, self.decision_timeframe, ("H1", "H3"))
        positions = dict(H1=0, H3=0)
        current = dict(H1=None, H3=None)
        state_positions = dict(H1=0, H3=0)
        current_state = dict(H1=None, H3=None)
        atr = atr_series(h, l, c)
        atr_rank = percentile_series(atr)
        range_rank = percentile_series([hi-lo for hi, lo in zip(h, l)])
        rows = []
        for i, opened in enumerate(frame.timestamp_utc):
            decision = opened + self._duration
            for tf in contexts:
                while positions[tf] < len(contexts[tf]) and contexts[tf][positions[tf]][0] <= decision:
                    current[tf] = contexts[tf][positions[tf]][1]
                    positions[tf] += 1
                while state_positions[tf] < len(state_contexts[tf]) and state_contexts[tf][state_positions[tf]][0] <= decision:
                    current_state[tf] = state_contexts[tf][state_positions[tf]][1]
                    state_positions[tf] += 1
            values = geometry(o[i], h[i], l[i], c[i])
            values.update(atr_14=atr[i], atr_percentile_100=atr_rank[i],
                          range_percentile_100=range_rank[i],
                          displacement_4_atr=displacement(c, i, 4, atr[i]),
                          displacement_8_atr=displacement(c, i, 8, atr[i]),
                          er_4=efficiency_ratio(c, i, 4),
                          er_16=efficiency_ratio(c, i, 16),
                          rv_16=realized_volatility(c, i, 16),
                          atr_change_1=atr_change(atr, i, 1),
                          atr_change_8=atr_change(atr, i, 8),
                          slope_atr_4=normalized_regression_slope(c, i, 4, atr[i]),
                          slope_atr_16=normalized_regression_slope(c, i, 16, atr[i]),
                          range_pos_16=prior_range_position(h, l, c[i], i, 16),
                          **prior_structure(h, l, c[i], i),
                          completed_h1_direction=current["H1"],
                          completed_h3_direction=current["H3"],
                          h1_er_8=None if current_state["H1"] is None else current_state["H1"]["er_8"],
                          h1_atr_change_4=None if current_state["H1"] is None else current_state["H1"]["atr_change_4"],
                          h1_slope_atr_8=None if current_state["H1"] is None else current_state["H1"]["slope_atr_8"],
                          h3_er_8=None if current_state["H3"] is None else current_state["H3"]["er_8"],
                          h3_atr_change_4=None if current_state["H3"] is None else current_state["H3"]["atr_change_4"],
                          h3_slope_atr_8=None if current_state["H3"] is None else current_state["H3"]["slope_atr_8"],
                          analysis_hour=decision.to_pydatetime().astimezone(self._timezone).hour,
                          weekday=decision.weekday(),
                          london_active=8 <= decision.to_pydatetime().astimezone(self._london_timezone).hour < 17,
                          new_york_active=8 <= decision.to_pydatetime().astimezone(self._new_york_timezone).hour < 17,
                          london_new_york_overlap=(
                              8 <= decision.to_pydatetime().astimezone(self._london_timezone).hour < 17
                              and 8 <= decision.to_pydatetime().astimezone(self._new_york_timezone).hour < 17
                          ))
            # Preserve the established UniversalFeatureEngine readiness contract.
            # The new H1/H3 state-context measurements have a much longer warm-up
            # and are optional on otherwise valid generic feature rows.  Market-state
            # dataset readiness is enforced separately against the selected V1 vector.
            optional_state_context = {
                "h1_er_8", "h1_atr_change_4", "h1_slope_atr_8",
                "h3_er_8", "h3_atr_change_4", "h3_slope_atr_8",
            }
            feature_ready = all(
                value is not None for name, value in values.items()
                if name not in optional_state_context
            )
            rows.append(FeatureRow(timestamp=decision.to_pydatetime(), symbol=symbol,
                                   decision_timeframe=self.decision_timeframe,
                                   feature_ready=feature_ready, **values))
        return tuple(rows)
