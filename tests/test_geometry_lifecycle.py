from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec, build_big_brother_common_geometry
from sandbox.market_state import geometry_lifecycle as lifecycle
from sandbox.market_state.geometry_lifecycle import (
    build_geometry_instance, build_big_brother_geometry_instance,
    build_historical_geometry_instances, geometry_available_at, BIG_BROTHER_GEOMETRY_FAMILY_ID,
)
UTC = timezone.utc
START = datetime(2024, 1, 15, tzinfo=UTC)


def candle(index=0):
    start = START + timedelta(hours=index * 3)
    end = start + timedelta(hours=3)
    return AnchoredBlockCandle(start.date(), f"C{index + 1}", index, start, end, start, end,
                               1.105, 1.11, 1.09, 1.1, 12, 12)


def test_generic_source_and_custom_levels():
    source = candle()
    specs = (ProjectionLevelSpec("PLUS", 1), ProjectionLevelSpec("NEG", -2.25))
    instance = build_geometry_instance(source, "CUSTOM_FAMILY", iter(specs))
    assert instance.geometry_family_id == "CUSTOM_FAMILY"
    for field in ("local_trading_date", "block_label", "block_index", "block_start_utc", "block_end_utc"):
        assert getattr(instance, "source_" + field) == getattr(source, field)
    assert instance.available_at_utc == source.block_end_utc
    assert [l.level_id for l in instance.projection.levels] == ["PLUS", "NEG"]
    assert [l.price for l in instance.projection.levels] == pytest.approx([1.11, 1.09375])


def test_exact_availability_boundary():
    instance = build_big_brother_geometry_instance(candle())
    assert geometry_available_at([instance], instance.available_at_utc - timedelta(microseconds=1)) == ()
    assert geometry_available_at([instance], instance.available_at_utc) == (instance,)


def test_mandatory_leakage_and_no_expiry():
    instances = build_historical_geometry_instances([candle(0), candle(1)])
    assert geometry_available_at(instances, START + timedelta(hours=4)) == instances[:1]
    assert geometry_available_at(instances, START + timedelta(hours=6)) == instances
    assert geometry_available_at(instances, START + timedelta(days=100)) == instances


def test_distinct_sources_identical_prices_and_generator():
    class Once:
        calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from (candle(0), candle(1), candle(2))
    source = Once()
    instances = build_historical_geometry_instances(source)
    assert source.calls == 1
    assert len(instances) == 3
    assert [i.source_block_label for i in instances] == ["C1", "C2", "C3"]
    assert len(set(instances)) == 3
    assert instances[0].projection.levels == instances[1].projection.levels
    assert geometry_available_at(reversed(instances), START + timedelta(days=1)) == instances


def test_availability_order_not_start_order():
    first = replace(candle(), block_end_utc=START + timedelta(hours=9), local_block_end=START + timedelta(hours=9))
    instances = build_historical_geometry_instances([first, candle(1)])
    assert [i.source_block_label for i in instances] == ["C2", "C1"]


@pytest.mark.parametrize("data", [[candle(1), candle(0)], [candle(), candle()]])
def test_bad_source_order(data):
    with pytest.raises(ValueError):
        build_historical_geometry_instances(data)


@pytest.mark.parametrize("builder", [build_big_brother_geometry_instance,
    lambda c: build_geometry_instance(c, "CUSTOM", ())])
def test_invalid_candle(builder):
    with pytest.raises(TypeError):
        builder(object())
    with pytest.raises(TypeError):
        build_historical_geometry_instances([object()])


@pytest.mark.parametrize("family", ["", "lower", "A-B", "A B", "A\n", 123])
def test_invalid_family(family):
    with pytest.raises(ValueError):
        build_geometry_instance(candle(), family, ())


@pytest.mark.parametrize("stamp", [datetime(2024, 1, 15, 3), datetime(2024, 1, 15, 3, tzinfo=timezone(timedelta(hours=1)))])
def test_invalid_availability_and_decision(stamp):
    instance = build_big_brother_geometry_instance(candle())
    with pytest.raises(ValueError):
        replace(instance, available_at_utc=stamp)
    with pytest.raises(ValueError):
        geometry_available_at([instance], stamp)


def test_availability_must_equal_end():
    instance = build_big_brother_geometry_instance(candle())
    for delta in (-1, 1):
        with pytest.raises(ValueError):
            replace(instance, available_at_utc=instance.available_at_utc + timedelta(seconds=delta))


@pytest.mark.parametrize("field,value", [("local_trading_date", (START + timedelta(days=1)).date()),
    ("block_label", "OTHER"), ("block_index", 99), ("block_start_utc", START - timedelta(hours=1)),
    ("block_end_utc", START + timedelta(hours=4))])
def test_inconsistent_projection_metadata(field, value):
    instance = build_big_brother_geometry_instance(candle())
    with pytest.raises(ValueError, match="inconsistent"):
        replace(instance, projection=replace(instance.projection, **{field: value}))


def test_invalid_builder_and_query_types():
    with pytest.raises(TypeError):
        build_historical_geometry_instances([candle()], builder=lambda c: object())
    with pytest.raises(ValueError, match="different source"):
        build_historical_geometry_instances([candle()], builder=lambda c: build_big_brother_geometry_instance(candle(1)))
    with pytest.raises(TypeError):
        geometry_available_at([object()], START)


def test_preset_delegation(monkeypatch):
    assert BIG_BROTHER_GEOMETRY_FAMILY_ID == "BIG_BROTHER_COMMON"
    source = candle()
    expected = build_big_brother_common_geometry(source)
    calls = []
    def tracked(c):
        calls.append(c)
        return build_big_brother_common_geometry(c)
    monkeypatch.setattr(lifecycle, "build_big_brother_common_geometry", tracked)
    instance = build_big_brother_geometry_instance(source)
    assert calls == [source]
    assert instance.projection == expected
    assert instance.geometry_family_id == BIG_BROTHER_GEOMETRY_FAMILY_ID
    assert [l.level_id for l in instance.projection.levels] == ["NEG_3_0", "NEG_3_3", "NEG_3_5"]


def test_immutable_and_empty():
    instance = build_big_brother_geometry_instance(candle())
    with pytest.raises(FrozenInstanceError):
        instance.available_at_utc = START
    assert build_historical_geometry_instances([]) == ()
    assert geometry_available_at([], START) == ()
