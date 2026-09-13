from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec
from sandbox.market_state.geometry_lifecycle import build_geometry_instance, build_big_brother_geometry_instance
from sandbox.market_state.level_interaction import observe_geometry_level
from sandbox.market_state import level_interaction_history as history

START = datetime(2024, 1, 15, tzinfo=timezone.utc)
AVAILABLE = START + timedelta(hours=3)
STEP = timedelta(minutes=15)


def candle(index=0):
    start = START + timedelta(hours=3 * index)
    end = start + timedelta(hours=3)
    return AnchoredBlockCandle(start.date(), f"C{index + 1}", index, start, end, start, end, 10, 12, 8, 9, 12, 12)


def instance(index=0):
    return build_geometry_instance(candle(index), "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))


def bars(count=8, touched=(), start=AVAILABLE):
    return [OHLCBar(start + i * STEP, 11, 12, 10 if i in touched else 11, 11) for i in range(count)]


def build(data, decision=None, source=None):
    return history.build_level_interaction_history(instance() if source is None else source,
        "LEVEL", data, AVAILABLE + len(data) * STEP if decision is None else decision)


@pytest.mark.parametrize("first", [0, 4, 10])
def test_immediate_and_delayed_touch(first):
    result = build(bars(first + 1, (first,)))
    assert result.first_touch_bar_index == result.bars_before_first_touch == first
    assert result.minutes_from_availability_to_first_touch_bar == first * 15.0
    assert result.first_touch_bar_open_time_utc == AVAILABLE + first * STEP


def test_multiple_and_contiguous_touched_bars():
    result = build(bars(8, (1, 4, 5)))
    assert result.touched_bar_indices == (1, 4, 5)
    assert result.touched_bar_count == 3
    assert result.touched_in_observed_window is True
    assert result.remained_untouched_in_observed_window is False
    assert result.last_touch_bar_index == 5
    assert result.last_touch_bar_open_time_utc == AVAILABLE + 5 * STEP


@pytest.mark.parametrize("count", [8, 32])
def test_untouched_including_mandatory_32_bar_window(count):
    result = build(bars(count))
    assert result.observed_bar_count == count
    assert result.touched_bar_count == 0
    assert result.touched_in_observed_window is False
    assert result.remained_untouched_in_observed_window is True
    assert result.touched_bar_indices == ()
    for field in ("first_touch_bar_index", "first_touch_bar_open_time_utc", "bars_before_first_touch",
        "minutes_from_availability_to_first_touch_bar", "last_touch_bar_index", "last_touch_bar_open_time_utc"):
        assert getattr(result, field) is None
    assert result.observation_start_utc == AVAILABLE
    assert result.observation_end_utc == AVAILABLE + count * STEP
    if count == 32:
        assert result.decision_time_utc == START + timedelta(hours=11)


def test_zero_observations():
    result = build([], AVAILABLE)
    assert result.observed_bar_count == 0
    assert result.touched_in_observed_window is False
    assert result.remained_untouched_in_observed_window is False
    assert result.observation_start_utc == result.observation_end_utc == AVAILABLE


def test_completed_horizon_and_future_and_past_ignored():
    decision = AVAILABLE + timedelta(hours=1, minutes=7)
    expected = build(bars(4), decision)
    supplied = bars(4, start=AVAILABLE - 4 * STEP) + bars(8)
    assert build(supplied, decision) == expected
    modified = [b if b.open_time_utc < AVAILABLE + 4 * STEP else
                OHLCBar(b.open_time_utc, 9, 12, 8, 11) for b in supplied]
    assert build(modified, decision) == expected
    assert expected.observed_bar_count == 4
    assert expected.observation_end_utc == AVAILABLE + timedelta(hours=1)


@pytest.mark.parametrize("missing", [0, 3, 7])
def test_missing_required_bars(missing):
    data = bars()
    del data[missing]
    with pytest.raises(ValueError, match="missing required"):
        build(data, AVAILABLE + 8 * STEP)


@pytest.mark.parametrize("bad", ["duplicate", "unordered", "misaligned", "type"])
def test_invalid_bars(bad):
    data = bars()
    if bad == "duplicate":
        data = [data[0], data[0]]
    elif bad == "unordered":
        data.reverse()
    elif bad == "misaligned":
        data = [OHLCBar(AVAILABLE + timedelta(seconds=1), 11, 12, 10, 11)]
    else:
        data = [object()]
    with pytest.raises((ValueError, TypeError)):
        build(data, AVAILABLE)


def test_invalid_instance_and_level():
    with pytest.raises(TypeError):
        build([], AVAILABLE, object())
    with pytest.raises(ValueError):
        history.build_level_interaction_history(instance(), "MISSING", [], AVAILABLE)
    source = instance()
    source = replace(source, projection=replace(source.projection, levels=source.projection.levels * 2))
    with pytest.raises(ValueError):
        build([], AVAILABLE, source)


@pytest.mark.parametrize("base", [0, -1, True, 15.0, "15"])
def test_invalid_base(base):
    with pytest.raises(ValueError):
        history.build_level_interaction_history(instance(), "LEVEL", [], AVAILABLE, base)


@pytest.mark.parametrize("decision", [AVAILABLE.replace(tzinfo=None),
    AVAILABLE.astimezone(timezone(timedelta(hours=1))), AVAILABLE - STEP])
def test_invalid_decision(decision):
    with pytest.raises(ValueError):
        build([], decision)


def test_misaligned_availability():
    c = candle()
    shifted = replace(c, block_end_utc=c.block_end_utc + timedelta(minutes=1))
    source = build_geometry_instance(shifted, "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))
    with pytest.raises(ValueError, match="availability must align"):
        build([], source.available_at_utc, source)


def test_generators_once_and_level_order():
    class Once:
        calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from bars()
    data = Once()
    assert build(data, AVAILABLE + 8 * STEP) == build(bars())
    assert data.calls == 1
    data = Once()
    source = build_geometry_instance(candle(), "CUSTOM", (ProjectionLevelSpec("B", 0), ProjectionLevelSpec("A", 1)))
    results = history.build_geometry_interaction_histories(source, data, AVAILABLE + 8 * STEP)
    assert data.calls == 1
    assert [r.level_id for r in results] == ["B", "A"]
    common = build_big_brother_geometry_instance(candle())
    results = history.build_geometry_interaction_histories(common, bars(), AVAILABLE + 8 * STEP)
    assert [r.level_id for r in results] == ["NEG_3_0", "NEG_3_3", "NEG_3_5"]


def test_source_identity_and_raw_delegation(monkeypatch):
    source = instance()
    data = bars(4, (2,))
    calls = []
    def tracked(*args):
        calls.append(args)
        return observe_geometry_level(*args)
    monkeypatch.setattr(history, "observe_geometry_level", tracked)
    result = build(data)
    assert len(calls) == 4
    assert result.observations == tuple(observe_geometry_level(source, "LEVEL", b) for b in data)
    for field in ("geometry_family_id", "source_local_trading_date", "source_block_label", "source_block_index",
        "source_block_start_utc", "source_block_end_utc"):
        assert getattr(result, field) == getattr(source, field)
    assert result.geometry_available_at_utc == source.available_at_utc
    assert (result.level_id, result.coordinate, result.level_price) == ("LEVEL", 0, 10)


def test_same_prices_distinct_sources_and_immutable():
    first = build([], AVAILABLE)
    second_source = instance(1)
    second = build([], second_source.available_at_utc, second_source)
    assert first.level_price == second.level_price
    assert first != second
    with pytest.raises(FrozenInstanceError):
        first.observed_bar_count = 99
