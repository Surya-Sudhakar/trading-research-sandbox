from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec
from sandbox.market_state.geometry_lifecycle import build_geometry_instance, build_big_brother_geometry_instance
from sandbox.market_state.level_interaction import LevelPricePosition, observe_geometry_level, observe_geometry_levels

START = datetime(2024, 1, 15, tzinfo=timezone.utc)


def candle(index=0):
    start = START + timedelta(hours=3 * index)
    end = start + timedelta(hours=3)
    return AnchoredBlockCandle(start.date(), f"C{index + 1}", index, start, end, start, end, 10, 12, 8, 9, 12, 12)


def instance(index=0):
    return build_geometry_instance(candle(index), "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))


def bar(opening=11, high=12, low=8, close=9, stamp=None):
    return OHLCBar(START + timedelta(hours=6) if stamp is None else stamp, opening, high, low, close)


@pytest.mark.parametrize("prices,expected", [
    ((11, 13, 11, 12), (False, False, True, False, False, False, False)),
    ((8, 9, 7, 8), (False, False, False, True, False, False, False)),
    ((11, 12, 10, 11), (True, False, True, False, False, False, True)),
    ((9, 10, 8, 9), (True, False, False, True, False, False, True)),
    ((11, 12, 8, 9), (True, True, True, True, True, True, False)),
    ((9, 12, 8, 11), (True, True, True, True, True, True, False)),
    ((10, 12, 8, 11), (True, True, True, True, True, False, False)),
    ((11, 12, 8, 10), (True, True, True, True, True, False, False)),
    ((10, 10, 10, 10), (True, True, False, False, False, False, False)),
])
def test_objective_facts(prices, expected):
    result = observe_geometry_level(instance(), "LEVEL", bar(*prices))
    assert (result.range_touched_level, result.body_touched_level, result.traded_above_level,
        result.traded_below_level, result.straddled_level, result.body_crossed_level,
        result.wick_only_touch) == expected


@pytest.mark.parametrize("price,position", [(11, LevelPricePosition.ABOVE), (10, LevelPricePosition.AT), (9, LevelPricePosition.BELOW)])
def test_positions(price, position):
    result = observe_geometry_level(instance(), "LEVEL", bar(price, 12, 8, price))
    assert result.open_position is position
    assert result.close_position is position


def test_identity_and_bar_data():
    source = instance()
    base = bar()
    result = observe_geometry_level(source, "LEVEL", base)
    for field in ("geometry_family_id", "source_local_trading_date", "source_block_label", "source_block_index",
        "source_block_start_utc", "source_block_end_utc"):
        assert getattr(result, field) == getattr(source, field)
    assert result.geometry_available_at_utc == source.available_at_utc
    assert (result.level_id, result.coordinate, result.level_price) == ("LEVEL", 0, 10)
    assert result.bar_open_time_utc == base.open_time_utc
    assert result.bar_close_time_utc == base.open_time_utc + timedelta(minutes=15)
    assert (result.open, result.high, result.low, result.close) == (base.open, base.high, base.low, base.close)


def test_unknown_and_ambiguous_level():
    source = instance()
    with pytest.raises(ValueError, match="exactly one"):
        observe_geometry_level(source, "MISSING", bar())
    duplicate = replace(source, projection=replace(source.projection, levels=source.projection.levels * 2))
    with pytest.raises(ValueError, match="exactly one"):
        observe_geometry_level(duplicate, "LEVEL", bar())


@pytest.mark.parametrize("which", ["instance", "bar"])
def test_invalid_types(which):
    with pytest.raises(TypeError):
        observe_geometry_level(object() if which == "instance" else instance(), "LEVEL",
                               object() if which == "bar" else bar())


@pytest.mark.parametrize("base", [0, -1, True, 15.0, "15"])
def test_invalid_base(base):
    with pytest.raises(ValueError):
        observe_geometry_level(instance(), "LEVEL", bar(), base)


def test_misaligned():
    with pytest.raises(ValueError, match="align"):
        observe_geometry_level(instance(), "LEVEL", bar(stamp=START + timedelta(hours=6, seconds=1)))


@pytest.mark.parametrize("delta", [timedelta(microseconds=1), timedelta(minutes=15)])
def test_before_availability(delta):
    source = instance()
    with pytest.raises(ValueError, match="before geometry"):
        observe_geometry_level(source, "LEVEL", bar(stamp=source.available_at_utc - delta))


def test_exact_availability():
    source = instance()
    assert observe_geometry_level(source, "LEVEL", bar(stamp=source.available_at_utc)).bar_open_time_utc == source.available_at_utc


def test_order_and_all_levels_including_untouched():
    source = build_geometry_instance(candle(), "CUSTOM", (ProjectionLevelSpec("B", 2), ProjectionLevelSpec("A", -2)))
    results = observe_geometry_levels(source, bar(20, 22, 19, 21))
    assert [r.level_id for r in results] == ["B", "A"]
    assert not any(r.range_touched_level for r in results)
    common = build_big_brother_geometry_instance(candle())
    assert [r.level_id for r in observe_geometry_levels(common, bar())] == ["NEG_3_0", "NEG_3_3", "NEG_3_5"]


def test_same_prices_distinct_sources():
    first = observe_geometry_level(instance(0), "LEVEL", bar())
    second = observe_geometry_level(instance(1), "LEVEL", bar())
    assert first.level_price == second.level_price
    assert first != second
    assert first.source_block_label != second.source_block_label


def test_immutable():
    result = observe_geometry_level(instance(), "LEVEL", bar())
    with pytest.raises(FrozenInstanceError):
        result.level_price = 99


def test_empty_levels_still_validate_availability():
    source = build_geometry_instance(candle(), "EMPTY", ())
    assert observe_geometry_levels(source, bar()) == ()
    with pytest.raises(ValueError):
        observe_geometry_levels(source, bar(stamp=START))
