from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state import future_outcome as outcome
from sandbox.market_state.future_outcome import OutcomeAnchor, measure_forward_outcome, measure_forward_outcomes, DEFAULT_FORWARD_HORIZONS

START = datetime(2024, 1, 15, 10, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def anchor(price=100):
    return OutcomeAnchor("RESEARCH", START, price)


def bars(count=32):
    prices = [(100, 102, 99, 101), (101, 104, 100, 103), (103, 103, 97, 98), (98, 101, 96, 100)]
    return [OHLCBar(START + i * STEP, *prices[i % 4]) for i in range(count)]


def test_mandatory_four_bar_example():
    result = measure_forward_outcome(anchor(), bars(4), 4)
    assert (result.first_open, result.end_close, result.highest_high, result.lowest_low) == (100, 100, 104, 96)
    assert result.close_change == result.close_return_fraction == 0
    assert result.max_upward_excursion == result.max_downward_excursion == 4
    assert (result.bars_to_highest_high, result.bars_to_lowest_low) == (1, 3)
    assert result.highest_high_bar_open_time_utc == START + STEP
    assert result.lowest_low_bar_open_time_utc == START + 3 * STEP
    assert result.window_start_utc == result.first_bar_open_time_utc == START
    assert result.window_end_utc == START + timedelta(hours=1)
    assert result.last_bar_open_time_utc == START + 3 * STEP
    assert (result.anchor_id, result.anchor_time_utc, result.reference_price) == ("RESEARCH", START, 100)


def test_one_bar():
    result = measure_forward_outcome(anchor(), bars(), 1)
    assert (result.first_open, result.end_close, result.highest_high, result.lowest_low) == (100, 101, 102, 99)
    assert result.close_change == 1
    assert result.close_return_fraction == pytest.approx(.01)
    assert result.window_end_utc == START + STEP
    assert result.bars_to_highest_high == result.bars_to_lowest_low == 0


def test_defaults_order_and_ties():
    results = measure_forward_outcomes(anchor(), bars())
    assert DEFAULT_FORWARD_HORIZONS == (1, 4, 8, 16, 32)
    assert tuple(r.horizon_bars for r in results) == DEFAULT_FORWARD_HORIZONS
    for result in results[1:]:
        assert (result.bars_to_highest_high, result.bars_to_lowest_low) == (1, 3)
        assert result.highest_high_bar_open_time_utc == START + STEP
        assert result.lowest_low_bar_open_time_utc == START + 3 * STEP
    assert [r.horizon_bars for r in measure_forward_outcomes(anchor(), bars(), (8, 1, 4))] == [8, 1, 4]


@pytest.mark.parametrize("price,change,up,down", [(110, -10, 0, 14), (90, 10, 14, 0), (100, 0, 4, 4), (0, 100, 104, 0), (-10, 110, 114, 0)])
def test_changes_excursions_and_zero_negative_references(price, change, up, down):
    result = measure_forward_outcome(anchor(price), bars(), 4)
    assert result.close_change == change
    assert result.close_return_fraction == (change / price if price else None)
    assert result.max_upward_excursion == up
    assert result.max_downward_excursion == down
    assert up >= 0 and down >= 0


@pytest.mark.parametrize("horizon", [4, 8])
def test_outside_window_invariance(horizon):
    data = bars()
    expected = measure_forward_outcome(anchor(), data[:horizon], horizon)
    before = OHLCBar(START - STEP, 999, 1000, 998, 999)
    modified = data[:horizon] + [OHLCBar(b.open_time_utc, 999, 1000, 998, 999) for b in data[horizon:]]
    assert measure_forward_outcome(anchor(), [before] + data, horizon) == expected
    assert measure_forward_outcome(anchor(), [before] + modified, horizon) == expected


@pytest.mark.parametrize("missing", [0, 1, 3])
def test_missing_required(missing):
    data = bars(4)
    del data[missing]
    with pytest.raises(ValueError, match="missing required"):
        measure_forward_outcome(anchor(), data, 4)


@pytest.mark.parametrize("bad", ["duplicate", "unordered", "misaligned", "type"])
def test_bad_bars(bad):
    data = bars()
    if bad == "duplicate":
        data = [data[0], data[0]]
    elif bad == "unordered":
        data.reverse()
    elif bad == "misaligned":
        data = [OHLCBar(START + timedelta(seconds=1), 100, 101, 99, 100)]
    else:
        data = [object()]
    with pytest.raises((TypeError, ValueError)):
        measure_forward_outcome(anchor(), data, 1)


def test_invalid_anchor_type():
    with pytest.raises(TypeError):
        measure_forward_outcome(object(), bars(), 1)


@pytest.mark.parametrize("identifier", ["", "lower", "A B", "A-B", "A\n", 123])
def test_invalid_id(identifier):
    with pytest.raises(ValueError):
        OutcomeAnchor(identifier, START, 100)


@pytest.mark.parametrize("stamp", [START.replace(tzinfo=None), START.astimezone(timezone(timedelta(hours=1)))])
def test_invalid_time(stamp):
    with pytest.raises(ValueError):
        OutcomeAnchor("A", stamp, 100)


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -float("inf"), True, "100", None])
def test_invalid_price(price):
    with pytest.raises(ValueError):
        anchor(price)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_horizon_and_base(value):
    with pytest.raises(ValueError):
        measure_forward_outcome(anchor(), bars(), value)
    with pytest.raises(ValueError):
        measure_forward_outcome(anchor(), bars(), 1, value)
    with pytest.raises(ValueError):
        measure_forward_outcomes(anchor(), bars(), (value,))


def test_misaligned_anchor():
    with pytest.raises(ValueError, match="align"):
        measure_forward_outcome(replace(anchor(), anchor_time_utc=START + timedelta(seconds=1)), bars(), 1)


def test_duplicate_horizons():
    with pytest.raises(ValueError, match="duplicate"):
        measure_forward_outcomes(anchor(), bars(), (1, 4, 1))


def test_single_preparation_and_once_iterables(monkeypatch):
    class Once:
        def __init__(self, values):
            self.values = values
            self.calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from self.values
    source = Once(bars())
    horizons = Once((8, 1, 4))
    original = outcome._prepare
    calls = []
    def tracked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(outcome, "_prepare", tracked)
    result = measure_forward_outcomes(anchor(), source, horizons)
    assert source.calls == horizons.calls == 1
    assert len(calls) == 1
    assert [r.horizon_bars for r in result] == [8, 1, 4]


def test_immutable():
    source = anchor()
    result = measure_forward_outcome(source, bars(), 1)
    with pytest.raises(FrozenInstanceError):
        source.reference_price = 99
    with pytest.raises(FrozenInstanceError):
        result.end_close = 99
