from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec
from sandbox.market_state.geometry_lifecycle import build_geometry_instance
from sandbox.market_state.level_interaction_history import build_level_interaction_history
from sandbox.market_state.level_state_sequence import LevelRangeState, build_level_state_sequence

START = datetime(2024, 1, 15, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def history(pattern):
    end = START + timedelta(hours=3)
    candle = AnchoredBlockCandle(START.date(), "C1", 0, START, end, START, end, 10, 12, 8, 9, 12, 12)
    source = build_geometry_instance(candle, "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))
    prices = {"A": (12, 13, 11, 11), "T": (11, 12, 10, 11), "B": (8, 9, 7, 8)}
    bars = [OHLCBar(end + i * STEP, *prices[state]) for i, state in enumerate(pattern)]
    return build_level_interaction_history(source, "LEVEL", bars, end + len(bars) * STEP)


@pytest.mark.parametrize("pattern,runs", [
    ("", ()), ("AAA", (("ABOVE", 0, 2),)), ("BBB", (("BELOW", 0, 2),)),
    ("TTT", (("TOUCH", 0, 2),)), ("AATTBB", (("ABOVE", 0, 1), ("TOUCH", 2, 3), ("BELOW", 4, 5))),
    ("ATA", (("ABOVE", 0, 0), ("TOUCH", 1, 1), ("ABOVE", 2, 2))),
    ("BTB", (("BELOW", 0, 0), ("TOUCH", 1, 1), ("BELOW", 2, 2))),
    ("ATB", (("ABOVE", 0, 0), ("TOUCH", 1, 1), ("BELOW", 2, 2))),
    ("BTA", (("BELOW", 0, 0), ("TOUCH", 1, 1), ("ABOVE", 2, 2))),
    ("ATATB", (("ABOVE", 0, 0), ("TOUCH", 1, 1), ("ABOVE", 2, 2), ("TOUCH", 3, 3), ("BELOW", 4, 4))),
    ("AATTAATBB", (("ABOVE", 0, 1), ("TOUCH", 2, 3), ("ABOVE", 4, 5), ("TOUCH", 6, 6), ("BELOW", 7, 8))),
])
def test_states_runs_and_original_observations(pattern, runs):
    source = history(pattern)
    result = build_level_state_sequence(source)
    mapping = {"A": LevelRangeState.ABOVE, "T": LevelRangeState.TOUCH, "B": LevelRangeState.BELOW}
    assert result.states == tuple(mapping[s] for s in pattern)
    assert [(r.state, r.first_observation_index, r.last_observation_index) for r in result.runs] == list(runs)
    assert result.run_count == len(runs)
    assert result.transition_count == max(0, len(runs) - 1)
    assert [r.run_index for r in result.runs] == list(range(len(runs)))
    for run in result.runs:
        first, last = run.first_observation_index, run.last_observation_index
        assert run.bar_count == last - first + 1
        assert run.first_bar_open_time_utc == source.observations[first].bar_open_time_utc
        assert run.last_bar_open_time_utc == source.observations[last].bar_open_time_utc
        assert run.run_end_utc == source.observations[last].bar_close_time_utc
        for offset, observation in enumerate(run.observations):
            assert observation is source.observations[first + offset]
    if not pattern:
        assert result.states == result.runs == ()


def test_identity_and_window_preserved():
    source = history("ATB")
    result = build_level_state_sequence(source)
    for field in ("geometry_family_id", "source_local_trading_date", "source_block_label", "source_block_index",
        "source_block_start_utc", "source_block_end_utc", "geometry_available_at_utc", "level_id", "coordinate",
        "level_price", "decision_time_utc", "base_minutes", "observed_bar_count"):
        assert getattr(result, field) == getattr(source, field)


def test_invalid_history():
    with pytest.raises(TypeError):
        build_level_state_sequence(object())


@pytest.mark.parametrize("field,value", [("observations", (object(),)), ("observed_bar_count", 99),
    ("touched_bar_indices", (1,)), ("touched_bar_count", 0)])
def test_inconsistent_history(field, value):
    with pytest.raises(ValueError):
        build_level_state_sequence(replace(history("T"), **{field: value}))


@pytest.mark.parametrize("field,value", [
    ("geometry_family_id", "OTHER"), ("source_local_trading_date", (START + timedelta(days=1)).date()),
    ("source_block_label", "C2"), ("source_block_index", 1),
    ("source_block_start_utc", START - STEP), ("source_block_end_utc", START + STEP),
    ("geometry_available_at_utc", START + STEP), ("level_id", "OTHER"), ("coordinate", 1), ("level_price", 99),
])
def test_mismatched_observation_identity(field, value):
    source = history("T")
    observation = replace(source.observations[0], **{field: value})
    with pytest.raises(ValueError, match="identity"):
        build_level_state_sequence(replace(source, observations=(observation,)))


@pytest.mark.parametrize("duplicate", [True, False])
def test_nonchronological(duplicate):
    source = history("AA")
    observations = (source.observations[0],) * 2 if duplicate else source.observations[::-1]
    with pytest.raises(ValueError, match="chronological"):
        build_level_state_sequence(replace(source, observations=observations))


def test_unclassifiable_observation():
    source = history("A")
    observation = replace(source.observations[0], low=9, high=11)
    with pytest.raises(ValueError, match="classified"):
        build_level_state_sequence(replace(source, observations=(observation,)))


def test_touch_flag_takes_priority():
    source = history("T")
    observation = replace(source.observations[0], low=11, high=12)
    result = build_level_state_sequence(replace(source, observations=(observation,)))
    assert result.states == (LevelRangeState.TOUCH,)


def test_immutable():
    result = build_level_state_sequence(history("ATB"))
    with pytest.raises(FrozenInstanceError):
        result.run_count = 99
    with pytest.raises(FrozenInstanceError):
        result.runs[0].bar_count = 99
    assert isinstance(result.states, tuple)
    assert isinstance(result.runs, tuple)
    assert isinstance(result.runs[0].observations, tuple)
