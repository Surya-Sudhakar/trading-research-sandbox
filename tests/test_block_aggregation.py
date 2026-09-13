from dataclasses import FrozenInstanceError
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sandbox.market_state.session_clock import BlockConfig, NEW_YORK_C1_C8
from sandbox.market_state.block_aggregation import OHLCBar, aggregate_completed_blocks

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


def bars_between(start, end, minutes=15):
    bars = []
    while start < end:
        i = len(bars)
        bars.append(OHLCBar(start, 100 + i, 110 + i, 90 + i, 105 + i))
        start += timedelta(minutes=minutes)
    return bars


def c1(month=1, day=15):
    start = datetime(2024, month, day, tzinfo=NY).astimezone(UTC)
    end = datetime(2024, month, day, 3, tzinfo=NY).astimezone(UTC)
    return bars_between(start, end), start, end


@pytest.mark.parametrize("month,day,count,hours", [(1, 15, 12, 3), (7, 15, 12, 3), (3, 10, 8, 2), (11, 3, 16, 4)])
def test_normal_and_dst_complete_blocks(month, day, count, hours):
    bars, start, end = c1(month, day)
    candle, = aggregate_completed_blocks(bars, end)
    assert candle.block_label == "C1"
    assert candle.block_index == 0
    assert candle.base_bar_count == candle.expected_base_bar_count == count
    assert candle.block_start_utc == start
    assert candle.block_end_utc == end
    assert end - start == timedelta(hours=hours)
    assert candle.local_block_start.hour == 0
    assert candle.local_block_end.hour == 3
    assert candle.local_block_start.tzinfo == NY
    assert candle.local_block_end.tzinfo == NY
    assert candle.local_trading_date == datetime(2024, month, day).date()
    assert (candle.open, candle.high, candle.low, candle.close) == (100, 110 + count - 1, 90, 105 + count - 1)
    hours_seen = [b.open_time_utc.astimezone(NY).hour for b in bars]
    if month == 3:
        assert 2 not in hours_seen
    if month == 11:
        assert hours_seen.count(1) == 8
        assert {b.open_time_utc.astimezone(NY).fold for b in bars if b.open_time_utc.astimezone(NY).hour == 1} == {0, 1}


def test_completion_boundary():
    bars, _, end = c1()
    assert aggregate_completed_blocks(bars, end - timedelta(seconds=1)) == ()
    assert len(aggregate_completed_blocks(bars, end)) == 1


@pytest.mark.parametrize("missing", [0, 5, -1])
def test_missing_bar_prevents_emission(missing):
    bars, _, end = c1()
    del bars[missing]
    assert aggregate_completed_blocks(bars, end) == ()


def test_future_bars_do_not_change_completed_output():
    bars, _, end = c1()
    expected = aggregate_completed_blocks(bars, end)
    future = bars_between(end, end + timedelta(hours=3))
    assert aggregate_completed_blocks(bars + future, end) == expected
    changed = [OHLCBar(b.open_time_utc, 1000, 2000, 500, 1500) for b in future]
    assert aggregate_completed_blocks(bars + changed, end) == expected
    assert aggregate_completed_blocks(bars + changed, end + timedelta(minutes=7)) == expected


def test_generic_anchored_config_and_chronological_output():
    config = BlockConfig("UTC", time(6), 120, 12, tuple(f"B{i}" for i in range(12)))
    start = datetime(2024, 1, 2, 4, tzinfo=UTC)
    end = start + timedelta(hours=4)
    result = aggregate_completed_blocks(iter(bars_between(start, end)), end, config)
    assert [c.block_label for c in result] == ["B11", "B0"]
    assert [c.block_start_utc for c in result] == [start, start + timedelta(hours=2)]
    assert result[0].local_trading_date == datetime(2024, 1, 1).date()
    assert all(c.base_bar_count == 8 for c in result)


@pytest.mark.parametrize("stamp", [datetime(2024, 1, 1), datetime(2024, 1, 1, tzinfo=NY)])
def test_invalid_bar_timestamp(stamp):
    with pytest.raises(ValueError, match="UTC"):
        OHLCBar(stamp, 10, 12, 9, 11)


@pytest.mark.parametrize("stamp", [datetime(2024, 1, 1), datetime(2024, 1, 1, tzinfo=NY)])
def test_invalid_decision_timestamp(stamp):
    with pytest.raises(ValueError, match="UTC"):
        aggregate_completed_blocks([], stamp)


@pytest.mark.parametrize("duplicate", [True, False])
def test_order_rejected(duplicate):
    bars, _, end = c1()
    bad = [bars[0], bars[0]] if duplicate else list(reversed(bars))
    with pytest.raises(ValueError, match="strictly increasing"):
        aggregate_completed_blocks(bad, end)


@pytest.mark.parametrize("prices", [(0, 12, 9, 11), (-1, 12, 9, 11), (10, float("inf"), 9, 11),
    (10, 12, float("nan"), 11), (10, 9, 8, 11), (10, 12, 11, 10), (10, 8, 12, 10)])
def test_invalid_ohlc(prices):
    with pytest.raises(ValueError):
        OHLCBar(datetime(2024, 1, 1, tzinfo=UTC), *prices)


@pytest.mark.parametrize("base", [0, -15, True, 1.5])
def test_invalid_base(base):
    with pytest.raises(ValueError):
        aggregate_completed_blocks([], datetime(2024, 1, 1, tzinfo=UTC), base_minutes=base)


def test_duration_not_divisible():
    bars, start, end = c1(3, 10)
    # 45 minutes divides a normal 180-minute block, but not spring C1's 120 minutes.
    aligned = datetime(1970, 1, 1, tzinfo=UTC)
    step = timedelta(minutes=45)
    stamp = start + (-(start - aligned)) % step
    bar = OHLCBar(stamp, 10, 12, 9, 11)
    with pytest.raises(ValueError, match="divisible"):
        aggregate_completed_blocks([bar], end, base_minutes=45)


@pytest.mark.parametrize("offset", [timedelta(minutes=1), timedelta(seconds=1), timedelta(microseconds=1)])
def test_misaligned_timestamp(offset):
    bars, start, end = c1()
    bar = OHLCBar(start + offset, 10, 12, 9, 11)
    with pytest.raises(ValueError, match="aligned"):
        aggregate_completed_blocks([bar], end)


def test_immutable_and_empty_input():
    bars, _, end = c1()
    with pytest.raises(FrozenInstanceError):
        bars[0].open = 999
    candle, = aggregate_completed_blocks(bars, end)
    with pytest.raises(FrozenInstanceError):
        candle.close = 999
    assert aggregate_completed_blocks([], end) == ()
