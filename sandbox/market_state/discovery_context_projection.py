"""Append scale-safe causal context predictors to the frozen core projection."""
from dataclasses import fields
import re
from .discovery_projection import DiscoveryValue, DiscoveryRow, project_research_record
from .discovery_market_context import LevelMeasurement, measure_level
from .research_record import ResearchRecord


def _segment(identifier):
    if not isinstance(identifier, str) or re.fullmatch(r"[A-Z0-9_]+", identifier) is None:
        raise ValueError("invalid machine context ID")
    return identifier.lower()


def _ratios(opening, low, closing, span, reference):
    return (span / abs(reference) if reference != 0 else None,
            abs(closing - opening) / span if span > 0 else None,
            (closing - low) / span if span > 0 else None)


def _shape(candle, reference):
    direction = "BULLISH" if candle.close > candle.open else ("BEARISH" if candle.close < candle.open else "FLAT")
    return (direction, *_ratios(candle.open, candle.low, candle.close, candle.high - candle.low, reference))


_SHAPE = ("direction", "range_fraction", "body_to_range", "close_location")
_RELATIONSHIP = ("observed_open_position", "observed_close_position",
    "traded_above_reference_high", "traded_below_reference_low", "closed_above_reference_high",
    "closed_below_reference_low", "high_excursion_returned_inside_by_close",
    "low_excursion_returned_inside_by_close", "stayed_inside_reference_range",
    "traded_both_sides_of_reference_range", "range_ratio", "same_direction",
    "overlaps_in_time", "overlap_minutes")
_DAILY_LEVELS = ("open", "high", "low", "close", "midpoint")
_LEVEL_MEASUREMENTS = tuple(field.name for field in fields(LevelMeasurement))
_MARKET_FIELDS = (
    ("volatility.atr", "atr"),
    ("volatility.atr_percentile_100", "atr_percentile"),
    ("volatility.range_percentile_100", "range_percentile"),
    ("structure.prior_high_12", "prior_high_12"),
    ("structure.prior_low_12", "prior_low_12"),
    ("structure.breakout_high_12", "breakout_high_12"),
    ("structure.breakout_low_12", "breakout_low_12"),
    ("structure.higher_high_count", "higher_high_count"),
    ("structure.lower_high_count", "lower_high_count"),
    ("structure.higher_low_count", "higher_low_count"),
    ("structure.lower_low_count", "lower_low_count"),
    ("swing.high.label", "latest_high_label"),
    ("swing.high.price", "latest_high_price"),
    ("swing.high.distance_price", "latest_high_distance_price"),
    ("swing.high.distance_atr", "latest_high_distance_atr"),
    ("swing.high.bars_since_confirmation", "latest_high_bars_since_confirmation"),
    ("swing.low.label", "latest_low_label"),
    ("swing.low.price", "latest_low_price"),
    ("swing.low.distance_price", "latest_low_distance_price"),
    ("swing.low.distance_atr", "latest_low_distance_atr"),
    ("swing.low.bars_since_confirmation", "latest_low_bars_since_confirmation"),
    ("structure.break_kind", "structure_break"),
)
_BREAK_FLAGS = ("bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish")


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def project_research_record_with_context(record: ResearchRecord) -> DiscoveryRow:
    if not isinstance(record, ResearchRecord):
        raise TypeError("record must be ResearchRecord")
    core = project_research_record(record)
    predictors = list(core.predictors)
    reference = record.anchor.outcome_anchor.reference_price
    def append(prefix, names, values):
        predictors.extend(DiscoveryValue(prefix + name, value) for name, value in zip(names, values))
    snapshot = record.multitimeframe_context.snapshot
    append("x.m15.", _SHAPE, _shape(snapshot.base_bar, reference))
    for frame in snapshot.frames:
        candle = frame.candle
        values = (False, None, None, None, None, None, None) if candle is None else (
            True, candle.block_label, candle.block_index, *_shape(candle, reference))
        append(f"x.mtf.{_segment(frame.frame_id)}.", ("available", "block_label", "block_index", *_SHAPE), values)
    for slot in record.session_relationship_context.session_context.sessions:
        summary = slot.summary
        shape = (None,) * 4 if summary is None else (summary.direction.value,
            *_ratios(summary.open, summary.low, summary.close, summary.range, reference))
        append(f"x.session.{_segment(slot.session_id)}.", ("available", "active", "missing_completed", *_SHAPE),
            (summary is not None, slot.active_at_anchor, not slot.active_at_anchor and summary is None, *shape))
    for slot in record.session_relationship_context.relationships:
        values = []
        for name in _RELATIONSHIP:
            value = getattr(slot.relationship, name) if slot.available else None
            if slot.available and name in ("observed_open_position", "observed_close_position"):
                value = value.value
            values.append(value)
        append(f"x.relationship.{_segment(slot.relationship_id)}.", ("available", *_RELATIONSHIP), (slot.available, *values))
    market = record.market_context
    break_value = None if market is None else _enum_value(market.structure_break)
    append("x.market.m15.", ("available", *(name for name, _ in _MARKET_FIELDS),
                              *("structure." + name for name in _BREAK_FLAGS)),
           (market is not None,
            *(None if market is None else _enum_value(getattr(market, attribute))
              for _, attribute in _MARKET_FIELDS),
            *(None if market is None else break_value == name.upper()
              for name in _BREAK_FLAGS)))
    transition = None if market is None else market.level_transition
    append("x.level_transition.", ("sweep", "reclaim"),
           (None if transition is None else _enum_value(transition.sweep),
            None if transition is None else _enum_value(transition.reclaim)))
    daily_slots = [frame for frame in snapshot.frames if frame.frame_id == "NY_D1"]
    daily = daily_slots[0].candle if len(daily_slots) == 1 else None
    daily_values = (False, None, None, None, None, None, None) if daily is None else (
        True, daily.open, daily.high, daily.low, daily.close,
        daily.high - daily.low, (daily.high + daily.low) / 2)
    append("x.daily.previous.", ("available", "open", "high", "low", "close", "range", "midpoint"), daily_values)
    daily_levels = {} if daily is None else {
        "open": daily.open, "high": daily.high, "low": daily.low,
        "close": daily.close, "midpoint": (daily.high + daily.low) / 2,
    }
    atr = None if market is None else market.atr
    for name in _DAILY_LEVELS:
        measurement = None if daily is None else measure_level(snapshot.base_bar, daily_levels[name], atr)
        append(f"x.daily.previous.level.{name}.", _LEVEL_MEASUREMENTS,
               tuple(None if measurement is None else getattr(measurement, metric)
                     for metric in _LEVEL_MEASUREMENTS))
    return DiscoveryRow(record, core.identity, tuple(predictors), core.targets)
