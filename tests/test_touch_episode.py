from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec
from sandbox.market_state.geometry_lifecycle import build_geometry_instance
from sandbox.market_state.level_interaction_history import build_level_interaction_history
from sandbox.market_state.touch_episode import build_touch_episodes

START = datetime(2024, 1, 15, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def history(touched=(), count=8, index=0):
    start = START + timedelta(hours=index * 3)
    end = start + timedelta(hours=3)
    candle = AnchoredBlockCandle(start.date(), f"C{index + 1}", index, start, end, start, end,
                                 10, 12, 8, 9, 12, 12)
    instance = build_geometry_instance(candle, "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))
    bars = [OHLCBar(end + i * STEP, 11, 12, 10 if i in touched else 11, 11) for i in range(count)]
    return build_level_interaction_history(instance, "LEVEL", bars, end + count * STEP)


@pytest.mark.parametrize("count", [0, 8])
def test_no_episodes(count):
    assert build_touch_episodes(history(count=count)) == ()


@pytest.mark.parametrize("touched,count,runs", [
    ((0,), 8, ((0, 0),)), ((1, 2, 3), 8, ((1, 3),)),
    ((1, 4, 5), 8, ((1, 1), (4, 5))), ((0, 2, 4, 6), 8, ((0, 0), (2, 2), (4, 4), (6, 6))),
    (tuple(range(8)), 8, ((0, 7),)), ((1, 2, 4, 5), 8, ((1, 2), (4, 5))),
    ((1, 2, 5, 6, 7, 10), 11, ((1, 2), (5, 7), (10, 10))),
])
def test_maximal_runs_and_timing(touched, count, runs):
    source = history(touched, count)
    episodes = build_touch_episodes(source)
    assert [(e.first_observation_index, e.last_observation_index) for e in episodes] == list(runs)
    assert [e.episode_index for e in episodes] == list(range(len(runs)))
    assert sum(e.touched_bar_count for e in episodes) == len(touched)
    for episode, (first, last) in zip(episodes, runs):
        assert episode.touched_bar_count == last - first + 1
        assert episode.first_bar_open_time_utc == source.geometry_available_at_utc + first * STEP
        assert episode.last_bar_open_time_utc == source.geometry_available_at_utc + last * STEP
        assert episode.episode_end_utc == source.observations[last].bar_close_time_utc
        assert episode.bars_from_availability_to_episode == first
        assert episode.minutes_from_availability_to_episode == first * 15.0
        assert len(episode.observations) == last - first + 1
        for offset, observation in enumerate(episode.observations):
            assert observation is source.observations[first + offset]


def test_delayed_episode_0530():
    episode, = build_touch_episodes(history((10,), 11))
    assert episode.geometry_available_at_utc == START + timedelta(hours=3)
    assert episode.first_bar_open_time_utc == START + timedelta(hours=5, minutes=30)
    assert episode.bars_from_availability_to_episode == 10
    assert episode.minutes_from_availability_to_episode == 150.0


def test_source_and_level_identity():
    source = history((1,))
    episode, = build_touch_episodes(source)
    for field in ("geometry_family_id", "source_local_trading_date", "source_block_label", "source_block_index",
                  "source_block_start_utc", "source_block_end_utc", "geometry_available_at_utc",
                  "level_id", "coordinate", "level_price"):
        assert getattr(episode, field) == getattr(source, field)


@pytest.mark.parametrize("field,value", [("observed_bar_count", 99), ("touched_bar_indices", (2,)),
    ("touched_bar_count", 99), ("touched_in_observed_window", False),
    ("remained_untouched_in_observed_window", True)])
def test_inconsistent_summary(field, value):
    with pytest.raises(ValueError, match="inconsistent"):
        build_touch_episodes(replace(history((1,)), **{field: value}))


@pytest.mark.parametrize("count", [0, 8])
def test_inconsistent_untouched_flags(count):
    source = history(count=count)
    with pytest.raises(ValueError):
        build_touch_episodes(replace(source, touched_in_observed_window=True))
    with pytest.raises(ValueError):
        build_touch_episodes(replace(source, remained_untouched_in_observed_window=not source.remained_untouched_in_observed_window))


def test_invalid_history_type():
    with pytest.raises(TypeError):
        build_touch_episodes(object())


def test_distinct_sources_same_prices():
    first, = build_touch_episodes(history((1,), index=0))
    second, = build_touch_episodes(history((1,), index=1))
    assert first.level_price == second.level_price
    assert first.source_block_label != second.source_block_label
    assert first != second


def test_immutable():
    episode, = build_touch_episodes(history((0,)))
    with pytest.raises(FrozenInstanceError):
        episode.episode_index = 10
    assert isinstance(episode.observations, tuple)
