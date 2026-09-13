from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.price_geometry import ProjectionLevelSpec
from sandbox.market_state.geometry_lifecycle import build_geometry_instance
from sandbox.market_state.level_interaction_history import build_level_interaction_history
from sandbox.market_state.level_state_sequence import build_level_state_sequence
from sandbox.market_state.touch_transition import build_touch_transition_patterns
from sandbox.market_state.future_outcome import OutcomeAnchor, measure_forward_outcomes
from sandbox.market_state import research_anchor as research

START = datetime(2024, 1, 15, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def history(pattern, available=None):
    end = START + timedelta(hours=3) if available is None else available
    start = end - timedelta(hours=3)
    candle = AnchoredBlockCandle(start.date(), "C1", 0, start, end, start, end, 10, 12, 8, 9, 12, 12)
    source = build_geometry_instance(candle, "CUSTOM", (ProjectionLevelSpec("LEVEL", 0),))
    prices = {"A": (11, 12, 11, 11), "T": (11, 12, 10, 11), "B": (8, 9, 7, 8)}
    bars = [OHLCBar(end + i * STEP, *prices[state]) for i, state in enumerate(pattern)]
    return build_level_interaction_history(source, "LEVEL", bars, end + len(bars) * STEP)


def pattern(states="ATB", available=None):
    return build_touch_transition_patterns(build_level_state_sequence(history(states, available)))[0]


@pytest.mark.parametrize("states", ["ATB", "ATA", "BTA", "BTB", "TA", "TB"])
def test_eligible_touch(states):
    source = pattern(states)
    anchor = research.build_touch_transition_research_anchor(source)
    confirmation = source.next_run.observations[0]
    assert anchor.evidence_end_utc == anchor.outcome_anchor.anchor_time_utc == confirmation.bar_close_time_utc
    assert anchor.evidence_end_utc != source.touch_end_utc
    assert anchor.evidence_end_utc != confirmation.bar_open_time_utc
    assert anchor.outcome_anchor.reference_price == confirmation.close != source.level_price
    assert anchor.base_minutes == 15
    assert anchor.pattern_index == source.pattern_index
    assert anchor.pattern_code == source.pattern_code
    assert anchor.checkpoint_bars is None
    assert anchor.outcome_anchor.anchor_id == "TOUCH_TRANSITION_0"
    for field in research._IDENTITY:
        assert getattr(anchor, field) == getattr(source, field)


@pytest.mark.parametrize("states", ["AT", "BT", "T"])
def test_ineligible_touch(states):
    assert research.build_touch_transition_research_anchor(pattern(states)) is None


def test_critical_1045_anchor_and_future_measurement(monkeypatch):
    source = pattern("ATB", START + timedelta(hours=10))
    anchor = research.build_touch_transition_research_anchor(source)
    assert source.touch_end_utc == START + timedelta(hours=10, minutes=30)
    assert anchor.evidence_end_utc == START + timedelta(hours=10, minutes=45)
    before = OHLCBar(source.touch_end_utc, 8, 9, 7, 8)
    future = [OHLCBar(anchor.evidence_end_utc + i * STEP, 8, 10, 6, 9) for i in range(4)]
    expected = measure_forward_outcomes(anchor.outcome_anchor, future, (1, 4), 15)
    original = research.measure_forward_outcomes
    calls = []
    returned = []
    def tracked(*args):
        calls.append(args)
        value = original(*args)
        returned.append(value)
        return value
    monkeypatch.setattr(research, "measure_forward_outcomes", tracked)
    result = research.measure_research_anchor_outcomes(anchor, [before] + future, (1, 4))
    assert result is returned[0]
    assert result == expected
    assert calls[0][0] is anchor.outcome_anchor
    assert calls[0][2:] == ((1, 4), 15)
    assert result[0].first_bar_open_time_utc == anchor.evidence_end_utc
    changed = replace(before, open=100, high=200, low=50, close=150)
    assert research.measure_research_anchor_outcomes(anchor, [changed] + future, (1, 4)) == expected


@pytest.mark.parametrize("count", [4, 8, 16, 32])
def test_untouched_checkpoints(count):
    source = history("A" * 32)
    anchor = research.build_untouched_checkpoint_research_anchor(source, count)
    assert anchor.kind is research.ResearchAnchorKind.UNTOUCHED_CHECKPOINT
    assert anchor.evidence_end_utc == source.observations[count - 1].bar_close_time_utc
    assert anchor.outcome_anchor.reference_price == 11
    assert anchor.level_price == 10
    assert anchor.checkpoint_bars == count
    assert anchor.pattern_index is anchor.pattern_code is None
    assert anchor.outcome_anchor.anchor_id == f"UNTOUCHED_CHECKPOINT_{count}"
    for field in research._IDENTITY:
        assert getattr(anchor, field) == getattr(source, field)


def test_mandatory_later_touch_and_requested_order():
    source = history("A" * 10 + "TA")
    for count, minute in [(4, 60), (8, 120), (10, 150)]:
        anchor = research.build_untouched_checkpoint_research_anchor(source, count)
        assert anchor.evidence_end_utc == START + timedelta(hours=3, minutes=minute)
    assert research.build_untouched_checkpoint_research_anchor(source, 11) is None
    assert research.build_untouched_checkpoint_research_anchor(source, 16) is None
    assert research.DEFAULT_UNTOUCHED_CHECKPOINTS == (4, 8, 16, 32)
    assert [a.checkpoint_bars for a in research.build_untouched_checkpoint_research_anchors(source)] == [4, 8]
    assert [a.checkpoint_bars for a in research.build_untouched_checkpoint_research_anchors(source, iter((8, 4, 11, 16)))] == [8, 4]


def test_touch_exactly_after_checkpoint():
    assert research.build_untouched_checkpoint_research_anchor(history("AAAAT"), 4) is not None
    assert research.build_untouched_checkpoint_research_anchor(history("TAAAA"), 4) is None


@pytest.mark.parametrize("count", [0, -1, True, 4.0, "4"])
def test_invalid_checkpoint(count):
    with pytest.raises(ValueError):
        research.build_untouched_checkpoint_research_anchor(history("AAAA"), count)
    with pytest.raises(ValueError):
        research.build_untouched_checkpoint_research_anchors(history("AAAA"), (count,))


def test_duplicate_checkpoints():
    with pytest.raises(ValueError):
        research.build_untouched_checkpoint_research_anchors(history("AAAA"), (4, 4))


@pytest.mark.parametrize("bad", ["count", "identity", "order", "timing", "base"])
def test_bad_history(bad):
    source = history("AAAA")
    if bad == "count":
        source = replace(source, observed_bar_count=99)
    elif bad == "identity":
        source = replace(source, observations=(replace(source.observations[0], level_id="OTHER"),) + source.observations[1:])
    elif bad == "order":
        source = replace(source, observations=source.observations[::-1])
    elif bad == "timing":
        source = replace(source, observations=source.observations[:-1] + (replace(source.observations[-1], bar_close_time_utc=source.observations[-1].bar_close_time_utc + STEP),))
    else:
        source = replace(source, base_minutes=True)
    with pytest.raises(ValueError):
        research.build_untouched_checkpoint_research_anchor(source, 4)


@pytest.mark.parametrize("changes", [
    {"kind": "TOUCH_TRANSITION"}, {"base_minutes": True}, {"pattern_index": -1},
    {"pattern_index": True}, {"pattern_code": ""}, {"checkpoint_bars": 4},
    {"evidence_end_utc": START}, {"evidence_end_utc": START.replace(tzinfo=None)},
    {"evidence_end_utc": START.astimezone(timezone(timedelta(hours=1)))},
    {"outcome_anchor": object()}, {"outcome_anchor": OutcomeAnchor("OTHER", START, 10)},
])
def test_invalid_anchor_combinations(changes):
    anchor = research.build_touch_transition_research_anchor(pattern())
    with pytest.raises(ValueError):
        replace(anchor, **changes)


def test_checkpoint_combinations_and_immutable():
    anchor = research.build_untouched_checkpoint_research_anchor(history("AAAA"), 4)
    for changes in ({"pattern_index": 0}, {"pattern_code": "OTHER"}, {"checkpoint_bars": True}):
        with pytest.raises(ValueError):
            replace(anchor, **changes)
    with pytest.raises(FrozenInstanceError):
        anchor.level_price = 99


def test_invalid_types():
    with pytest.raises(TypeError):
        research.build_touch_transition_research_anchor(object())
    with pytest.raises(TypeError):
        research.build_untouched_checkpoint_research_anchor(object(), 4)
    with pytest.raises(TypeError):
        research.measure_research_anchor_outcomes(object(), [])


@pytest.mark.parametrize("duration", [timedelta(0), timedelta(seconds=30), timedelta(minutes=-1)])
def test_bad_confirmation_duration(duration):
    source = pattern()
    observation = source.next_run.observations[0]
    changed = replace(observation, bar_close_time_utc=observation.bar_open_time_utc + duration)
    source = replace(source, next_run=replace(source.next_run, observations=(changed,)))
    with pytest.raises(ValueError):
        research.build_touch_transition_research_anchor(source)
