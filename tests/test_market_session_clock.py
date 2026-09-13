from dataclasses import FrozenInstanceError
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from sandbox.market_state.session_clock import BlockConfig, NEW_YORK_C1_C8, resolve_block

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@pytest.mark.parametrize("month", [1, 7])
@pytest.mark.parametrize("hour,minute,second,label,index", [
    (0, 0, 0, "C1", 0), (2, 59, 59, "C1", 0), (3, 0, 0, "C2", 1),
    (6, 0, 0, "C3", 2), (21, 0, 0, "C8", 7), (23, 59, 59, "C8", 7),
])
def test_local_boundaries(month, hour, minute, second, label, index):
    local = datetime(2024, month, 15, hour, minute, second, tzinfo=NY)
    stamp = local.astimezone(UTC)
    result = resolve_block(stamp)
    assert result.utc_timestamp == stamp
    assert result.local_timestamp == local
    assert result.local_trading_date == local.date()
    assert (result.block_label, result.block_index) == (label, index)
    assert result.local_block_start.hour == index * 3
    assert result.local_block_end.hour == ((index + 1) * 3) % 24
    assert result.local_block_start.tzinfo == NY
    assert result.local_block_end.tzinfo == NY
    assert result.local_block_start <= local < result.local_block_end
    if index == 7:
        assert result.local_block_end.date() == local.date() + timedelta(days=1)


@pytest.mark.parametrize("month,utc_hour,offset", [(1, 8, -5), (7, 7, -4)])
def test_utc_conversion(month, utc_hour, offset):
    result = resolve_block(datetime(2024, month, 15, utc_hour, tzinfo=UTC))
    assert result.local_timestamp.hour == 3
    assert result.local_timestamp.utcoffset() == timedelta(hours=offset)
    assert result.block_label == "C2"


@pytest.mark.parametrize("month,day", [(3, 9), (3, 10), (3, 11), (11, 2), (11, 3), (11, 4)])
@pytest.mark.parametrize("hour", [0, 3, 6, 9, 12, 15, 18, 21])
def test_transition_date_boundaries(month, day, hour):
    local = datetime(2024, month, day, hour, tzinfo=NY)
    result = resolve_block(local.astimezone(UTC))
    assert result.block_label == f"C{hour // 3 + 1}"
    assert result.local_block_start == local
    assert result.local_trading_date == local.date()


@pytest.mark.parametrize("month,day,hour,minute,local_hour,fold,label", [
    (3, 10, 6, 59, 1, 0, "C1"), (3, 10, 7, 0, 3, 0, "C2"),
    (11, 3, 5, 30, 1, 0, "C1"), (11, 3, 6, 30, 1, 1, "C1"),
    (11, 3, 8, 0, 3, 0, "C2"),
])
def test_dst_jump_and_repeated_hour(month, day, hour, minute, local_hour, fold, label):
    result = resolve_block(datetime(2024, month, day, hour, minute, tzinfo=UTC))
    assert result.local_timestamp.hour == local_hour
    assert result.local_timestamp.fold == fold
    assert result.block_label == label


@pytest.mark.parametrize("month,day,elapsed,c1_elapsed", [(3, 10, 23, 2), (11, 3, 25, 4)])
def test_transition_days_not_24_elapsed_hours(month, day, elapsed, c1_elapsed):
    first = resolve_block(datetime(2024, month, day, tzinfo=NY).astimezone(UTC))
    last = resolve_block(datetime(2024, month, day, 21, tzinfo=NY).astimezone(UTC))
    assert last.local_block_end.astimezone(UTC) - first.local_block_start.astimezone(UTC) == timedelta(hours=elapsed)
    assert first.local_block_end.astimezone(UTC) - first.local_block_start.astimezone(UTC) == timedelta(hours=c1_elapsed)


def test_naive_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_block(datetime(2024, 1, 1))


def test_non_utc_rejected():
    with pytest.raises(ValueError, match="UTC"):
        resolve_block(datetime(2024, 1, 1, tzinfo=NY))


def test_invalid_timezone():
    with pytest.raises(ZoneInfoNotFoundError):
        BlockConfig("Not/A_Zone", time(), 180, 8, NEW_YORK_C1_C8.labels)


@pytest.mark.parametrize("changes", [
    {"block_minutes": 0}, {"block_minutes": -1}, {"block_minutes": 180.5},
    {"block_minutes": True}, {"block_minutes": 179}, {"number_of_blocks": 0},
    {"number_of_blocks": 7}, {"labels": ()}, {"labels": ("",) * 8},
    {"labels": (" ",) * 8}, {"labels": ("C1",) * 8}, {"labels": "12345678"},
    {"anchor": time(tzinfo=UTC)}, {"anchor": "00:00"},
])
def test_invalid_configurations(changes):
    kwargs = dict(timezone="America/New_York", anchor=time(), block_minutes=180,
                  number_of_blocks=8, labels=NEW_YORK_C1_C8.labels)
    with pytest.raises(ValueError):
        BlockConfig(**(kwargs | changes))


def test_generic_nonmidnight_anchor():
    config = BlockConfig("UTC", time(6), 720, 2, ("FIRST", "SECOND"))
    result = resolve_block(datetime(2024, 1, 2, 5, 59, 59, tzinfo=UTC), config)
    assert result.local_trading_date == date(2024, 1, 1)
    assert result.block_label == "SECOND"
    assert result.local_block_start == datetime(2024, 1, 1, 18, tzinfo=UTC)
    assert result.local_block_end == datetime(2024, 1, 2, 6, tzinfo=UTC)
    next_cycle = resolve_block(datetime(2024, 1, 2, 6, tzinfo=UTC), config)
    assert next_cycle.block_label == "FIRST"
    assert next_cycle.local_trading_date == date(2024, 1, 2)


def test_immutable_config_and_result():
    labels = ["DAY"]
    config = BlockConfig("UTC", time(), 1440, 1, labels)
    labels[0] = "CHANGED"
    assert config.labels == ("DAY",)
    with pytest.raises(FrozenInstanceError):
        config.block_minutes = 1
    result = resolve_block(datetime(2024, 1, 1, tzinfo=UTC), config)
    with pytest.raises(FrozenInstanceError):
        result.block_label = "CHANGED"
