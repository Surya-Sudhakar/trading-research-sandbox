from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import AnchoredBlockCandle
from sandbox.market_state import price_geometry as geometry
from sandbox.market_state.price_geometry import (
    ProjectionLevelSpec, ProjectedLevel, BodyDirection, BIG_BROTHER_COMMON_LEVELS,
    projection_price, build_completed_block_body_projection, build_big_brother_common_geometry,
)


def candle(opening=1.1050, closing=1.1000):
    start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    end = start + timedelta(hours=3)
    return AnchoredBlockCandle(start.date(), "CUSTOM_BLOCK", 2, start, end, start, end,
                               opening, max(opening, closing), min(opening, closing), closing, 12, 12)


@pytest.mark.parametrize("coordinate,expected", [(0, 1.1050), (-1, 1.1000)])
def test_anchors(coordinate, expected):
    assert projection_price(1.1050, 1.1000, coordinate) == pytest.approx(expected)


@pytest.mark.parametrize("opening,closing,direction,prices", [
    (1.1050, 1.1000, BodyDirection.BEARISH, [1.0900, 1.0885, 1.0875]),
    (1.1000, 1.1050, BodyDirection.BULLISH, [1.1150, 1.1165, 1.1175]),
    (1.1000, 1.1000, BodyDirection.FLAT, [1.1000, 1.1000, 1.1000]),
])
def test_direction_body_and_prices(opening, closing, direction, prices):
    result = build_big_brother_common_geometry(candle(opening, closing))
    assert result.direction is direction
    assert result.signed_body_move == pytest.approx(closing - opening)
    assert result.body_size == pytest.approx(abs(closing - opening))
    assert result.anchor_zero_price == opening
    assert result.anchor_minus_one_price == closing
    assert len(result.levels) == 3
    assert [level.price for level in result.levels] == pytest.approx(prices)


def test_metadata_preserved():
    source = candle()
    result = build_big_brother_common_geometry(source)
    for field in ("local_trading_date", "block_label", "block_index", "block_start_utc", "block_end_utc", "open", "close"):
        assert getattr(result, field) == getattr(source, field)


def test_custom_order_and_single_materialization():
    specs = (ProjectionLevelSpec("PLUS_1", 1), ProjectionLevelSpec("NEG_2_25", -2.25), ProjectionLevelSpec("PLUS_4_5", 4.5))
    class Once:
        def __init__(self):
            self.calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from specs
    source = Once()
    result = build_completed_block_body_projection(candle(10, 8), source)
    assert source.calls == 1
    assert [level.level_id for level in result.levels] == [s.level_id for s in specs]
    assert [level.coordinate for level in result.levels] == [1, -2.25, 4.5]
    assert [level.price for level in result.levels] == [12, 5.5, 19]


@pytest.mark.parametrize("specs", [
    (ProjectionLevelSpec("A", 1), ProjectionLevelSpec("A", 2)),
    (ProjectionLevelSpec("A", 1), ProjectionLevelSpec("B", 1)),
    (ProjectionLevelSpec("A", 0), ProjectionLevelSpec("B", -0.0)),
])
def test_duplicates(specs):
    with pytest.raises(ValueError, match="duplicate"):
        build_completed_block_body_projection(candle(), specs)


@pytest.mark.parametrize("identifier", ["", "lower", "HAS SPACE", "A-B", "A\n", 123])
def test_invalid_id(identifier):
    with pytest.raises(ValueError):
        ProjectionLevelSpec(identifier, 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "1", None, True, object()])
def test_invalid_coordinate(value):
    with pytest.raises(ValueError):
        ProjectionLevelSpec("A", value)


@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
def test_invalid_projection_input(field, bad):
    inputs = [10, 8, 1]
    inputs[field] = bad
    with pytest.raises(ValueError):
        projection_price(*inputs)


def test_overflow_rejected():
    with pytest.raises(ValueError):
        projection_price(1e308, -1e308, 3)


@pytest.mark.parametrize("price", [float("nan"), float("inf"), True])
def test_projected_price_finite(price):
    with pytest.raises(ValueError):
        ProjectedLevel("A", 1, price)


def test_invalid_candle_and_level_type():
    with pytest.raises(TypeError):
        build_completed_block_body_projection(object(), BIG_BROTHER_COMMON_LEVELS)
    with pytest.raises(TypeError):
        build_completed_block_body_projection(candle(), [object()])


def test_immutable():
    result = build_big_brother_common_geometry(candle())
    for obj, field, value in [(BIG_BROTHER_COMMON_LEVELS[0], "coordinate", 0),
        (result.levels[0], "price", 0), (result, "open", 0)]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)
    assert isinstance(result.levels, tuple)


def test_preset_and_generic_delegation(monkeypatch):
    assert [(s.level_id, s.coordinate) for s in BIG_BROTHER_COMMON_LEVELS] == [
        ("NEG_3_0", -3.0), ("NEG_3_3", -3.3), ("NEG_3_5", -3.5)]
    source = candle()
    expected = build_completed_block_body_projection(source, BIG_BROTHER_COMMON_LEVELS)
    calls = []
    original = geometry.build_completed_block_body_projection
    def tracked(candle, specs):
        calls.append((candle, specs))
        return original(candle, specs)
    monkeypatch.setattr(geometry, "build_completed_block_body_projection", tracked)
    assert build_big_brother_common_geometry(source) == expected
    assert calls == [(source, BIG_BROTHER_COMMON_LEVELS)]
