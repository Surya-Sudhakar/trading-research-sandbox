from dataclasses import FrozenInstanceError
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfoNotFoundError

import pytest
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.session_summary import (
    SessionConfig, DEFAULT_SESSIONS, TOKYO_SESSION, LONDON_SESSION, NEW_YORK_SESSION,
    session_window_for_date, aggregate_completed_sessions,
)
UTC = timezone.utc


def bars(start, end, closing=11):
    result = []
    while start < end:
        result.append(OHLCBar(start, 10, 12 + len(result), 8, closing))
        start += timedelta(minutes=15)
    return result


def sample(config=TOKYO_SESSION, day=date(2024, 1, 15), closing=11):
    window = session_window_for_date(config, day)
    return window, bars(window.start_utc, window.end_utc, closing)


@pytest.mark.parametrize("config", DEFAULT_SESSIONS)
def test_presets_complete_ohlc(config):
    window, data = sample(config)
    summary, = aggregate_completed_sessions(iter(data), window.end_utc, (config,))
    assert summary.session_id == config.session_id
    assert summary.local_session_date == window.local_session_date
    assert (summary.start_utc, summary.end_utc) == (window.start_utc, window.end_utc)
    assert (summary.open, summary.high, summary.low, summary.close) == (10, 47, 8, 11)
    assert summary.range == 39
    assert summary.direction == "BULLISH"
    assert summary.base_bar_count == summary.expected_base_bar_count == 36


@pytest.mark.parametrize("closing,direction", [(11, "BULLISH"), (9, "BEARISH"), (10, "FLAT")])
def test_directions(closing, direction):
    window, data = sample(closing=closing)
    summary, = aggregate_completed_sessions(data, window.end_utc, (TOKYO_SESSION,))
    assert summary.direction == direction


def test_completion_boundary():
    w, data = sample()
    assert aggregate_completed_sessions(data, w.end_utc - timedelta(seconds=1), (TOKYO_SESSION,)) == ()
    assert len(aggregate_completed_sessions(data, w.end_utc, (TOKYO_SESSION,))) == 1


@pytest.mark.parametrize("missing", [0, 15, -1])
def test_missing(missing):
    w, data = sample()
    del data[missing]
    assert aggregate_completed_sessions(data, w.end_utc, (TOKYO_SESSION,)) == ()


def test_future_invariance():
    w, data = sample()
    expected = aggregate_completed_sessions(data, w.end_utc)
    future = bars(w.end_utc, w.end_utc + timedelta(hours=24))
    assert aggregate_completed_sessions(data + future, w.end_utc) == expected
    changed = [OHLCBar(b.open_time_utc, 100, 200, 50, 150) for b in future]
    assert aggregate_completed_sessions(data + changed, w.end_utc) == expected


def test_overlap_independent():
    london = session_window_for_date(LONDON_SESSION, date(2024, 1, 15))
    ny = session_window_for_date(NEW_YORK_SESSION, date(2024, 1, 15))
    data = bars(london.start_utc, ny.end_utc)
    # Shared bar contributes the same extreme to both sessions.
    data = [OHLCBar(b.open_time_utc, 10, 999, 8, 11) if b.open_time_utc == ny.start_utc else b for b in data]
    result = aggregate_completed_sessions(data, ny.end_utc, (LONDON_SESSION, NEW_YORK_SESSION))
    assert len(result) == 2
    assert all(s.high == 999 and s.base_bar_count == 36 for s in result)


@pytest.mark.parametrize("config,month,start_hour,end_hour", [
    (LONDON_SESSION, 1, 8, 17), (LONDON_SESSION, 7, 7, 16),
    (NEW_YORK_SESSION, 1, 13, 22), (NEW_YORK_SESSION, 7, 12, 21),
])
def test_winter_summer_conversion(config, month, start_hour, end_hour):
    w = session_window_for_date(config, date(2024, month, 15))
    assert w.start_utc.hour == start_hour
    assert w.end_utc.hour == end_hour
    assert w.local_start.hour == 8
    assert w.local_end.hour == 17


def test_us_uk_mismatched_dst_weeks():
    day = date(2024, 3, 20)
    london = session_window_for_date(LONDON_SESSION, day)
    ny = session_window_for_date(NEW_YORK_SESSION, day)
    assert london.start_utc.hour == 8
    assert ny.start_utc.hour == 12
    assert london.local_start.utcoffset() != ny.local_start.utcoffset()


@pytest.mark.parametrize("start,end", [(time(8), time(17)), (time(22), time(6))])
def test_custom_utc_and_overnight(start, end):
    config = SessionConfig("CUSTOM", "UTC", start, end)
    w, data = sample(config)
    summary, = aggregate_completed_sessions(data, w.end_utc, (config,))
    assert summary.local_session_date == date(2024, 1, 15)
    assert summary.local_end.date() == date(2024, 1, 15) + timedelta(days=end < start)
    assert summary.base_bar_count == len(data)


@pytest.mark.parametrize("day,hours", [(date(2024, 3, 9), 7), (date(2024, 11, 2), 9)])
def test_overnight_dst_actual_duration(day, hours):
    config = SessionConfig("OVERNIGHT", "America/New_York", time(22), time(6))
    w, data = sample(config, day)
    summary, = aggregate_completed_sessions(data, w.end_utc, (config,))
    assert w.end_utc - w.start_utc == timedelta(hours=hours)
    assert summary.expected_base_bar_count == hours * 4


@pytest.mark.parametrize("identifier", ["", "lower", "HAS SPACE", "A\n", 123])
def test_invalid_id(identifier):
    with pytest.raises(ValueError):
        SessionConfig(identifier, "UTC", time(8), time(17))


def test_invalid_zone():
    with pytest.raises(ZoneInfoNotFoundError):
        SessionConfig("TEST", "Invalid/Zone", time(8), time(17))


@pytest.mark.parametrize("start,end", [("08:00", time(17)), (time(8), "17:00"),
    (time(8, tzinfo=UTC), time(17)), (time(8), time(17, tzinfo=UTC)), (time(8), time(8))])
def test_invalid_times(start, end):
    with pytest.raises(ValueError):
        SessionConfig("TEST", "UTC", start, end)


@pytest.mark.parametrize("bad", ["duplicate", "unordered", "misaligned", "type"])
def test_bad_bars(bad):
    w, data = sample()
    if bad == "duplicate":
        data = [data[0], data[0]]
    elif bad == "unordered":
        data.reverse()
    elif bad == "misaligned":
        data = [OHLCBar(w.start_utc + timedelta(seconds=1), 10, 12, 8, 11)]
    else:
        data = [object()]
    with pytest.raises(ValueError):
        aggregate_completed_sessions(data, w.end_utc)


@pytest.mark.parametrize("base", [0, -1, True, 15.0])
def test_invalid_base(base):
    with pytest.raises(ValueError):
        aggregate_completed_sessions([], datetime(2024, 1, 1, tzinfo=UTC), base_minutes=base)


def test_duplicate_sessions():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_completed_sessions([], datetime(2024, 1, 1, tzinfo=UTC), (TOKYO_SESSION,) * 2)


@pytest.mark.parametrize("stamp", [datetime(2024, 1, 1), datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=1)))])
def test_invalid_decision(stamp):
    with pytest.raises(ValueError, match="UTC"):
        aggregate_completed_sessions([], stamp)


def test_immutable():
    w, data = sample()
    summary, = aggregate_completed_sessions(data, w.end_utc, (TOKYO_SESSION,))
    for obj, field, value in [(TOKYO_SESSION, "session_id", "OTHER"), (w, "start_utc", None), (summary, "close", 20)]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)


def test_nondivisible_duration():
    config = SessionConfig("ODD", "UTC", time(8), time(9, 1))
    w, data = sample(config)
    with pytest.raises(ValueError, match="divisible"):
        aggregate_completed_sessions(data, w.end_utc, (config,))


def test_nonexistent_boundary_rejected():
    config = SessionConfig("GAP", "America/New_York", time(2, 30), time(6))
    with pytest.raises(ValueError, match="nonexistent"):
        session_window_for_date(config, date(2024, 3, 10))
