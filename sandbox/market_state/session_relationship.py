"""Descriptive relationships between two completed session summaries."""
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from .session_summary import CompletedSessionSummary


class RangePosition(StrEnum):
    ABOVE = "ABOVE"
    INSIDE = "INSIDE"
    BELOW = "BELOW"


@dataclass(frozen=True)
class CompletedSessionRelationship:
    reference_session_id: str
    observed_session_id: str
    reference_local_session_date: date
    observed_local_session_date: date
    reference_start_utc: datetime
    reference_end_utc: datetime
    observed_start_utc: datetime
    observed_end_utc: datetime
    observed_open_position: RangePosition
    observed_close_position: RangePosition
    traded_above_reference_high: bool
    traded_below_reference_low: bool
    closed_above_reference_high: bool
    closed_below_reference_low: bool
    high_excursion_returned_inside_by_close: bool
    low_excursion_returned_inside_by_close: bool
    stayed_inside_reference_range: bool
    traded_both_sides_of_reference_range: bool
    reference_range: float
    observed_range: float
    range_ratio: float | None
    same_direction: bool
    overlaps_in_time: bool
    overlap_minutes: float


def _position(value, reference):
    if value > reference.high:
        return RangePosition.ABOVE
    if value < reference.low:
        return RangePosition.BELOW
    return RangePosition.INSIDE


def compare_completed_sessions(reference: CompletedSessionSummary,
                               observed: CompletedSessionSummary) -> CompletedSessionRelationship:
    """Compare OHLC ranges without inferring any intraday event sequence.

    Excursion return flags follow the reference-side threshold only; a close
    beyond the opposite edge still satisfies that threshold.
    """
    if not isinstance(reference, CompletedSessionSummary) or not isinstance(observed, CompletedSessionSummary):
        raise TypeError("both arguments must be CompletedSessionSummary")
    above = observed.high > reference.high
    below = observed.low < reference.low
    overlap_start = max(reference.start_utc, observed.start_utc)
    overlap_end = min(reference.end_utc, observed.end_utc)
    overlaps = overlap_end > overlap_start
    return CompletedSessionRelationship(
        reference_session_id=reference.session_id,
        observed_session_id=observed.session_id,
        reference_local_session_date=reference.local_session_date,
        observed_local_session_date=observed.local_session_date,
        reference_start_utc=reference.start_utc, reference_end_utc=reference.end_utc,
        observed_start_utc=observed.start_utc, observed_end_utc=observed.end_utc,
        observed_open_position=_position(observed.open, reference),
        observed_close_position=_position(observed.close, reference),
        traded_above_reference_high=above, traded_below_reference_low=below,
        closed_above_reference_high=observed.close > reference.high,
        closed_below_reference_low=observed.close < reference.low,
        high_excursion_returned_inside_by_close=above and observed.close <= reference.high,
        low_excursion_returned_inside_by_close=below and observed.close >= reference.low,
        stayed_inside_reference_range=observed.high <= reference.high and observed.low >= reference.low,
        traded_both_sides_of_reference_range=above and below,
        reference_range=reference.range, observed_range=observed.range,
        range_ratio=observed.range / reference.range if reference.range > 0 else None,
        same_direction=reference.direction == observed.direction,
        overlaps_in_time=overlaps,
        overlap_minutes=(overlap_end - overlap_start).total_seconds() / 60 if overlaps else 0.0,
    )
