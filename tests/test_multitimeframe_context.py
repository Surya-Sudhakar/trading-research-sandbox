from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.multitimeframe_context import (
    ContextFrame, DEFAULT_CONTEXT_FRAMES, NEW_YORK_H1, build_multitimeframe_snapshot,
)

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


def local(day=15, hour=0, minute=0, month=1):
    return datetime(2024, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


def bars(start=None, end=None):
    start = local(14) if start is None else start
    end = local(16) if end is None else end
    result = []
    while start < end:
        result.append(OHLCBar(start, 10, 12, 9, 11))
        start += timedelta(minutes=15)
    return result


def indexed(snapshot):
    return {frame.frame_id: frame for frame in snapshot.frames}


def test_1015_snapshot():
    snapshot = build_multitimeframe_snapshot(bars(), local(hour=10, minute=15))
    assert snapshot.base_bar.open_time_utc == local(hour=10)
    assert [f.frame_id for f in snapshot.frames] == ["NY_D1", "NY_H4", "NY_H1", "NY_H3"]
    expected = {"NY_D1": (local(14), local(15)), "NY_H4": (local(hour=4), local(hour=8)),
                "NY_H1": (local(hour=9), local(hour=10)), "NY_H3": (local(hour=6), local(hour=9))}
    for frame in snapshot.frames:
        assert (frame.expected_block_start_utc, frame.expected_block_end_utc) == expected[frame.frame_id]
        assert frame.candle is not None
        assert (frame.candle.block_start_utc, frame.candle.block_end_utc) == expected[frame.frame_id]
    assert indexed(snapshot)["NY_H3"].candle.block_label == "C3"


def test_noon_boundary():
    at = indexed(build_multitimeframe_snapshot(bars(), local(hour=12)))
    for name, start in [("NY_H1", 11), ("NY_H4", 8), ("NY_H3", 9)]:
        assert at[name].expected_block_start_utc == local(hour=start)
        assert at[name].expected_block_end_utc == local(hour=12)
        assert at[name].candle is not None
    assert at["NY_H3"].candle.block_label == "C4"
    before = indexed(build_multitimeframe_snapshot(bars(), local(hour=12) - timedelta(seconds=1)))
    assert before["NY_H4"].expected_block_end_utc == local(hour=8)
    assert before["NY_H3"].expected_block_end_utc == local(hour=9)


def test_missing_h4_no_stale_fallback():
    data = [b for b in bars() if b.open_time_utc != local(hour=5)]
    frame = indexed(build_multitimeframe_snapshot(data, local(hour=10, minute=15)))["NY_H4"]
    assert frame.expected_block_start_utc == local(hour=4)
    assert frame.candle is None


@pytest.mark.parametrize("minute,expected_hour,expected_minute", [(15, 10, 0), (7, 9, 45)])
def test_base_expected_and_missing(minute, expected_hour, expected_minute):
    decision = local(hour=10, minute=minute)
    expected = local(hour=expected_hour, minute=expected_minute)
    assert build_multitimeframe_snapshot(bars(), decision).base_bar.open_time_utc == expected
    data = [b for b in bars() if b.open_time_utc != expected]
    assert build_multitimeframe_snapshot(data, decision).base_bar is None


def test_future_changes_and_generator():
    decision = local(hour=10, minute=7)
    past = [b for b in bars() if b.open_time_utc + timedelta(minutes=15) <= decision]
    expected = build_multitimeframe_snapshot(past, decision)
    assert build_multitimeframe_snapshot((b for b in bars()), decision) == expected
    replaced = [b if b.open_time_utc + timedelta(minutes=15) <= decision else
                OHLCBar(b.open_time_utc, 100, 120, 90, 110) for b in bars()]
    assert build_multitimeframe_snapshot(replaced, decision) == expected


@pytest.mark.parametrize("decision", [datetime(2024, 1, 15), datetime(2024, 1, 15, tzinfo=NY)])
def test_bad_decision(decision):
    with pytest.raises(ValueError, match="UTC"):
        build_multitimeframe_snapshot([], decision)


@pytest.mark.parametrize("duplicate", [True, False])
def test_bad_order(duplicate):
    data = bars()
    bad = [data[0], data[0]] if duplicate else data[::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_multitimeframe_snapshot(bad, local(hour=12))


@pytest.mark.parametrize("identifier", ["", "ny_h1", "NY H1", "NY-H1", "NY_H1\n", 123])
def test_bad_frame_id(identifier):
    with pytest.raises(ValueError):
        ContextFrame(identifier, NEW_YORK_H1)


def test_duplicate_frames():
    with pytest.raises(ValueError, match="duplicate"):
        build_multitimeframe_snapshot([], local(), frames=(DEFAULT_CONTEXT_FRAMES[0],) * 2)


def test_immutable():
    snapshot = build_multitimeframe_snapshot(bars(), local(hour=12))
    for obj, field, value in [(snapshot, "base_bar", None), (snapshot.frames[0], "candle", None),
                              (DEFAULT_CONTEXT_FRAMES[0], "frame_id", "OTHER")]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)
    assert isinstance(snapshot.frames, tuple)


@pytest.mark.parametrize("month,day,day_hours,h4_hours,h1_hours", [(3, 10, 23, 3, 1), (11, 3, 25, 5, 2)])
def test_dst_frames(month, day, day_hours, h4_hours, h1_hours):
    data = bars(local(day, month=month), local(day + 1, month=month))
    early = indexed(build_multitimeframe_snapshot(data, local(day, 4, month=month)))
    assert early["NY_H4"].candle.expected_base_bar_count == h4_hours * 4
    assert early["NY_H4"].candle.local_block_start.hour == 0
    assert early["NY_H3"].candle.block_label == "C1"
    daily = indexed(build_multitimeframe_snapshot(data, local(day + 1, month=month)))["NY_D1"]
    assert daily.candle.expected_base_bar_count == day_hours * 4
    decision = local(day, 3 if month == 3 else 2, month=month)
    hourly = indexed(build_multitimeframe_snapshot(data, decision))["NY_H1"]
    assert hourly.candle.expected_base_bar_count == h1_hours * 4
    assert hourly.candle.local_block_start.hour == 1


def test_empty_data_still_returns_expected_windows():
    snapshot = build_multitimeframe_snapshot([], local(hour=10, minute=15))
    assert snapshot.base_bar is None
    assert len(snapshot.frames) == 4
    assert all(f.candle is None for f in snapshot.frames)
