"""Append scale-safe causal context predictors to the frozen core projection."""
import re
from .discovery_projection import DiscoveryValue, DiscoveryRow, project_research_record
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
    return DiscoveryRow(record, core.identity, tuple(predictors), core.targets)
