from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone

import pytest
from sandbox.market_state.session_summary import (
    CompletedSessionSummary, SessionDirection, LONDON_SESSION, NEW_YORK_SESSION,
    TOKYO_SESSION, session_window_for_date,
)
from sandbox.market_state.session_relationship import RangePosition, compare_completed_sessions

UTC = timezone.utc


def summary(session_id="CUSTOM", opening=12, high=20, low=10, close=18, start_hour=8, end_hour=17):
    start = datetime(2024, 1, 15, start_hour, tzinfo=UTC)
    end = datetime(2024, 1, 15, end_hour, tzinfo=UTC)
    direction = SessionDirection.BULLISH if close > opening else (SessionDirection.BEARISH if close < opening else SessionDirection.FLAT)
    return CompletedSessionSummary(session_id, start.date(), start, end, start, end,
        opening, high, low, close, high - low, direction, 36, 36)


@pytest.mark.parametrize("high,low,close,above,below,closed_above,closed_below,high_return,low_return,inside,both", [
    (19, 11, 18, False, False, False, False, False, False, True, False),
    (22, 11, 21, True, False, True, False, False, False, False, False),
    (22, 11, 18, True, False, False, False, True, False, False, False),
    (19, 8, 9, False, True, False, True, False, False, False, False),
    (19, 8, 12, False, True, False, False, False, True, False, False),
    (22, 8, 15, True, True, False, False, True, True, False, True),
    (20, 10, 20, False, False, False, False, False, False, True, False),
    (22, 8, 9, True, True, False, True, True, False, False, True),
    (22, 8, 21, True, True, True, False, False, True, False, True),
    (22, 10, 20, True, False, False, False, True, False, False, False),
    (20, 8, 10, False, True, False, False, False, True, False, False),
])
def test_range_relationships(high, low, close, above, below, closed_above, closed_below, high_return, low_return, inside, both):
    result = compare_completed_sessions(summary(), summary(high=high, low=low, close=close))
    assert (result.traded_above_reference_high, result.traded_below_reference_low,
        result.closed_above_reference_high, result.closed_below_reference_low,
        result.high_excursion_returned_inside_by_close, result.low_excursion_returned_inside_by_close,
        result.stayed_inside_reference_range, result.traded_both_sides_of_reference_range) == (
        above, below, closed_above, closed_below, high_return, low_return, inside, both)


@pytest.mark.parametrize("value,position", [(21, RangePosition.ABOVE), (15, RangePosition.INSIDE),
    (9, RangePosition.BELOW), (20, RangePosition.INSIDE), (10, RangePosition.INSIDE)])
def test_open_and_close_positions(value, position):
    result = compare_completed_sessions(summary(), summary(opening=value, close=value, high=22, low=8))
    assert result.observed_open_position is position
    assert result.observed_close_position is position


def test_range_ratio_and_identity():
    reference = summary("CUSTOM_A")
    observed = summary("CUSTOM_B", high=25, low=5)
    result = compare_completed_sessions(reference, observed)
    assert result.reference_range == 10
    assert result.observed_range == 20
    assert result.range_ratio == 2
    assert result.reference_session_id == "CUSTOM_A"
    assert result.observed_session_id == "CUSTOM_B"
    assert result.reference_local_session_date == reference.local_session_date
    assert result.observed_local_session_date == observed.local_session_date
    assert (result.reference_start_utc, result.reference_end_utc) == (reference.start_utc, reference.end_utc)
    assert (result.observed_start_utc, result.observed_end_utc) == (observed.start_utc, observed.end_utc)


def test_zero_reference_range():
    flat = summary(opening=10, high=10, low=10, close=10)
    assert compare_completed_sessions(flat, summary()).range_ratio is None


@pytest.mark.parametrize("first,second,expected", [(SessionDirection.BULLISH, SessionDirection.BULLISH, True),
    (SessionDirection.BULLISH, SessionDirection.BEARISH, False), (SessionDirection.FLAT, SessionDirection.FLAT, True)])
def test_direction(first, second, expected):
    assert compare_completed_sessions(replace(summary(), direction=first), replace(summary(), direction=second)).same_direction is expected


@pytest.mark.parametrize("start,end,minutes", [(18, 22, 0), (17, 22, 0), (13, 22, 240), (9, 10, 60), (8, 17, 540)])
def test_time_overlap(start, end, minutes):
    result = compare_completed_sessions(summary(), summary(start_hour=start, end_hour=end))
    assert result.overlaps_in_time is (minutes > 0)
    assert result.overlap_minutes == float(minutes)


def test_fractional_overlap_minutes():
    observed = replace(summary(), start_utc=datetime(2024, 1, 15, 16, 59, 30, tzinfo=UTC))
    assert compare_completed_sessions(summary(), observed).overlap_minutes == 0.5


@pytest.mark.parametrize("first,second", [(LONDON_SESSION, NEW_YORK_SESSION), (TOKYO_SESSION, LONDON_SESSION)])
def test_preset_summary_pairs(first, second):
    def preset(config):
        w = session_window_for_date(config, date(2024, 3, 20))
        return replace(summary(config.session_id), local_session_date=w.local_session_date,
            start_utc=w.start_utc, end_utc=w.end_utc, local_start=w.local_start, local_end=w.local_end)
    a, b = preset(first), preset(second)
    result = compare_completed_sessions(a, b)
    assert (result.reference_session_id, result.observed_session_id) == (first.session_id, second.session_id)
    expected = max(0, (min(a.end_utc, b.end_utc) - max(a.start_utc, b.start_utc)).total_seconds() / 60)
    assert result.overlap_minutes == expected


@pytest.mark.parametrize("first,second", [(None, summary()), (summary(), {}), (object(), "summary")])
def test_invalid_types(first, second):
    with pytest.raises(TypeError):
        compare_completed_sessions(first, second)


def test_immutable():
    result = compare_completed_sessions(summary(), summary())
    with pytest.raises(FrozenInstanceError):
        result.range_ratio = 5
