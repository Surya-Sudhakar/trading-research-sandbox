from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec
from sandbox.market_state.geometry_lifecycle import build_geometry_instance
from sandbox.market_state.level_interaction_history import build_level_interaction_history
from sandbox.market_state.level_state_sequence import LevelRangeState, build_level_state_sequence
from sandbox.market_state.touch_transition import (
    ReclaimDirection, SweepDirection, build_touch_transition_patterns,
    classify_level_transition,
)

START = datetime(2024, 1, 15, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def sequence(pattern):
    end = START + timedelta(hours=3)
    candle = AnchoredBlockCandle(START.date(), "C1", 0, START, end, START, end, 10, 12, 8, 9, 12, 12)
    source = build_geometry_instance(candle, "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))
    prices = {"A": (11, 12, 11, 11), "T": (11, 12, 9, 11), "B": (8, 9, 7, 8)}
    bars = [OHLCBar(end + i * STEP, *prices[state]) for i, state in enumerate(pattern)]
    return build_level_state_sequence(build_level_interaction_history(source, "LEVEL", bars, end + len(bars) * STEP))


@pytest.mark.parametrize("pattern,code", [("", None), ("AA", None), ("BB", None),
    ("TT", "TOUCH_ONLY"), ("AT", "ABOVE_TOUCH_END"), ("BT", "BELOW_TOUCH_END"),
    ("TA", "START_TOUCH_ABOVE"), ("TB", "START_TOUCH_BELOW"),
    ("ATA", "ABOVE_TOUCH_ABOVE"), ("ATB", "ABOVE_TOUCH_BELOW"),
    ("BTA", "BELOW_TOUCH_ABOVE"), ("BTB", "BELOW_TOUCH_BELOW")])
def test_all_codes_and_side_comparisons(pattern, code):
    source = sequence(pattern)
    results = build_touch_transition_patterns(source)
    if code is None:
        assert results == ()
        return
    result, = results
    assert result.pattern_code == code
    assert result.pattern_index == 0
    has_previous = pattern[0] != "T"
    has_next = pattern[-1] != "T"
    assert result.has_previous_side is has_previous
    assert result.has_next_side is has_next
    assert result.complete_side_to_side_transition is (has_previous and has_next)
    if has_previous and has_next:
        assert result.same_side_before_after is (pattern[0] == pattern[-1])
        assert result.opposite_side_before_after is (pattern[0] != pattern[-1])
    else:
        assert result.same_side_before_after is None
        assert result.opposite_side_before_after is None
    for prefix, exists in (("previous", has_previous), ("next", has_next)):
        if not exists:
            for suffix in ("state", "run_index", "bar_count", "run"):
                assert getattr(result, f"{prefix}_{suffix}") is None


def test_mandatory_multiple_patterns():
    results = build_touch_transition_patterns(sequence("ATATBTB"))
    assert [p.pattern_code for p in results] == ["ABOVE_TOUCH_ABOVE", "ABOVE_TOUCH_BELOW", "BELOW_TOUCH_BELOW"]
    assert [p.pattern_index for p in results] == [0, 1, 2]
    assert [p.touch_run_index for p in results] == [1, 3, 5]


def test_identity_timing_counts_and_original_runs():
    source = sequence("AATTTBB")
    result, = build_touch_transition_patterns(source)
    assert result.previous_run is source.runs[0]
    assert result.touch_run is source.runs[1]
    assert result.next_run is source.runs[2]
    assert result.touch_run.observations[0] is source.runs[1].observations[0]
    assert (result.touch_first_observation_index, result.touch_last_observation_index, result.touch_bar_count) == (2, 4, 3)
    assert result.touch_first_bar_open_time_utc == START + timedelta(hours=3) + 2 * STEP
    assert result.touch_last_bar_open_time_utc == START + timedelta(hours=3) + 4 * STEP
    assert result.touch_end_utc == START + timedelta(hours=3) + 5 * STEP
    assert (result.previous_run_index, result.next_run_index) == (0, 2)
    assert (result.previous_bar_count, result.next_bar_count) == (2, 2)
    assert result.previous_state is LevelRangeState.ABOVE
    assert result.next_state is LevelRangeState.BELOW
    for field in ("geometry_family_id", "source_local_trading_date", "source_block_label", "source_block_index",
        "source_block_start_utc", "source_block_end_utc", "geometry_available_at_utc", "level_id", "coordinate", "level_price"):
        assert getattr(result, field) == getattr(source, field)


def test_invalid_type():
    with pytest.raises(TypeError):
        build_touch_transition_patterns(object())


@pytest.mark.parametrize("changes", [{"run_count": 99}, {"transition_count": 99}, {"observed_bar_count": 99}, {"states": ()}])
def test_sequence_counts(changes):
    with pytest.raises(ValueError):
        build_touch_transition_patterns(replace(sequence("ATB"), **changes))


@pytest.mark.parametrize("changes", [{"run_index": 99}, {"first_observation_index": 2},
    {"first_observation_index": 0}, {"last_observation_index": 0}, {"bar_count": 99},
    {"observations": ()}, {"observations": (object(),)}, {"state": LevelRangeState.BELOW}])
def test_invalid_run(changes):
    source = sequence("ATB")
    runs = (source.runs[0], replace(source.runs[1], **changes), source.runs[2])
    with pytest.raises(ValueError):
        build_touch_transition_patterns(replace(source, runs=runs))


def test_same_state_neighbors():
    source = sequence("AT")
    changed = replace(source.runs[1], state=LevelRangeState.ABOVE)
    with pytest.raises(ValueError, match="different states"):
        build_touch_transition_patterns(replace(source, states=(LevelRangeState.ABOVE,) * 2, runs=(source.runs[0], changed)))


def test_missing_final_coverage():
    source = sequence("ATB")
    with pytest.raises(ValueError):
        build_touch_transition_patterns(replace(source, runs=source.runs[:2], run_count=2, transition_count=1))


@pytest.mark.parametrize("field,value", [("geometry_family_id", "OTHER"), ("source_block_label", "C2"),
    ("source_block_index", 99), ("source_local_trading_date", (START + timedelta(days=1)).date()),
    ("source_block_start_utc", START - STEP), ("source_block_end_utc", START + STEP),
    ("geometry_available_at_utc", START), ("level_id", "OTHER"), ("coordinate", 99), ("level_price", 99)])
def test_observation_identity_mismatch(field, value):
    source = sequence("T")
    observation = replace(source.runs[0].observations[0], **{field: value})
    run = replace(source.runs[0], observations=(observation,))
    with pytest.raises(ValueError, match="identity"):
        build_touch_transition_patterns(replace(source, runs=(run,)))


def test_immutable():
    pattern, = build_touch_transition_patterns(sequence("ATA"))
    with pytest.raises(FrozenInstanceError):
        pattern.pattern_code = "OTHER"


@pytest.mark.parametrize("states,sweep,reclaim", [
    ("BTB", SweepDirection.ABOVE, ReclaimDirection.BELOW),
    ("ATA", SweepDirection.BELOW, ReclaimDirection.ABOVE),
    ("BTA", None, ReclaimDirection.ABOVE),
    ("ATB", None, ReclaimDirection.BELOW),
])
def test_generic_sweep_and_reclaim_semantics(states, sweep, reclaim):
    pattern, = build_touch_transition_patterns(sequence(states))
    assert classify_level_transition(pattern).sweep is sweep
    assert classify_level_transition(pattern).reclaim is reclaim
