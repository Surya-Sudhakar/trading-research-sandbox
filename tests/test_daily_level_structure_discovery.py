from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from sandbox.market_state.block_aggregation import OHLCBar, AnchoredBlockCandle
from sandbox.market_state.discovery_context_projection import project_research_record_with_context
from sandbox.market_state.discovery_market_context import (
    build_discovery_market_contexts, measure_level,
)
from sandbox.market_state.discovery_projection import values_by_id
from sandbox.market_state.multitimeframe_context import CompletedFrameContext
from sandbox.market_state.swings import (
    ConfirmedSwing, PivotLabel, StructureBreakKind, confirmed_structure_breaks,
    label_confirmed_swings,
)
from sandbox.research.discovery_record_builder import DiscoveryRecordConfig
from test_discovery_context_projection import record


UTC = timezone.utc
STEP = timedelta(minutes=15)


def bars(values):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return tuple(OHLCBar(start + i * STEP, close, high, low, close)
                 for i, (high, low, close) in enumerate(values))


def test_arbitrary_level_distance_penetration_raw_and_atr_normalized():
    bar = OHLCBar(datetime(2024, 1, 1, tzinfo=UTC), 99, 103, 97, 101)
    result = measure_level(bar, 100, 2)
    assert result.close_distance_price == 1
    assert result.close_distance_atr == .5
    assert result.range_distance_price == 0
    assert result.penetration_above_price == 3
    assert result.penetration_above_atr == 1.5
    assert result.penetration_below_price == 3
    assert result.penetration_below_atr == 1.5
    assert measure_level(OHLCBar(bar.open_time_utc, 105, 106, 104, 105), 100).range_distance_price == 4
    with pytest.raises(ValueError):
        measure_level(bar, float("nan"))
    with pytest.raises(ValueError):
        measure_level(bar, 100, 0)


def test_ordered_pivot_labels_and_confirmation_filter():
    swings = (
        ConfirmedSwing("HIGH", 1, 3, 10), ConfirmedSwing("LOW", 2, 4, 5),
        ConfirmedSwing("HIGH", 4, 6, 12), ConfirmedSwing("LOW", 5, 7, 6),
        ConfirmedSwing("HIGH", 7, 9, 11), ConfirmedSwing("LOW", 8, 10, 4),
    )
    assert [item.label for item in label_confirmed_swings(swings)] == [
        None, None, PivotLabel.HH, PivotLabel.HL, PivotLabel.LH, PivotLabel.LL]
    assert len(label_confirmed_swings(swings, as_of_index=6)) == 3


def test_bos_and_choch_are_close_crosses_of_visible_confirmed_swings():
    bos = confirmed_structure_breaks([8, 8, 11], (ConfirmedSwing("HIGH", 0, 1, 10),))
    assert [(event.kind, event.confirmation_index) for event in bos] == [
        (StructureBreakKind.BOS_BULLISH, 2)]
    swings = (
        ConfirmedSwing("HIGH", 0, 0, 10), ConfirmedSwing("LOW", 1, 1, 5),
        ConfirmedSwing("HIGH", 2, 2, 9), ConfirmedSwing("LOW", 3, 3, 4),
    )
    events = confirmed_structure_breaks([8, 8, 8, 8, 10, 3], swings)
    assert [event.kind for event in events] == [
        StructureBreakKind.CHOCH_BULLISH, StructureBreakKind.CHOCH_BEARISH]
    assert [event.confirmation_index for event in events] == [4, 5]


def test_market_context_prefix_equivalence_prevents_swing_lookahead():
    source = bars([
        (11, 9, 10), (13, 10, 12), (15, 11, 14), (13, 10, 11),
        (12, 9, 10), (16, 10, 15), (13, 9, 10), (12, 8, 9),
        (14, 9, 13), (13, 10, 11), (12, 9, 10), (17, 10, 16),
        (14, 9, 10), (13, 8, 9), (15, 9, 14), (14, 10, 11),
    ])
    complete = build_discovery_market_contexts(source)
    for size in range(1, len(source) + 1):
        prefix = build_discovery_market_contexts(source[:size])
        expected = {key: value for key, value in complete.items()
                    if key <= source[size - 1].open_time_utc + STEP}
        assert prefix == expected
    assert complete[source[2].open_time_utc + STEP].latest_high_price is None
    assert complete[source[4].open_time_utc + STEP].latest_high_price == 15


def test_market_context_pivot_strength_is_configurable():
    source = bars([
        (10, 8, 9), (15, 9, 14), (11, 8, 10), (12, 9, 11), (10, 7, 8),
    ])
    default = build_discovery_market_contexts(source)
    faster = build_discovery_market_contexts(source, swing_left_bars=1, swing_right_bars=1)
    decision = source[2].open_time_utc + STEP
    assert default[decision].latest_high_price is None
    assert faster[decision].latest_high_price == 15
    assert DiscoveryRecordConfig(swing_left_bars=1, swing_right_bars=3).swing_right_bars == 3
    with pytest.raises(ValueError):
        DiscoveryRecordConfig(swing_left_bars=0)


def test_previous_completed_new_york_day_and_market_features_are_projected():
    source = record()
    start = source.anchor.evidence_end_utc - timedelta(days=1, hours=5)
    daily = AnchoredBlockCandle(start.date(), "D1", 0, start, start + timedelta(days=1),
                                start, start + timedelta(days=1), 100, 120, 80, 110, 96, 96)
    frame = CompletedFrameContext("NY_D1", daily.block_start_utc, daily.block_end_utc, daily)
    snapshot = replace(source.multitimeframe_context.snapshot,
                       frames=source.multitimeframe_context.snapshot.frames + (frame,))
    mtf = replace(source.multitimeframe_context, snapshot=snapshot,
                  missing_frame_ids=("NY_H3",), all_frames_available=False)
    context = build_discovery_market_contexts((snapshot.base_bar,))[source.anchor.evidence_end_utc]
    source = replace(source, multitimeframe_context=mtf, market_context=context)
    values = values_by_id(project_research_record_with_context(source).predictors)
    assert values["x.daily.previous.open"] == 100
    assert values["x.daily.previous.high"] == 120
    assert values["x.daily.previous.low"] == 80
    assert values["x.daily.previous.close"] == 110
    assert values["x.daily.previous.range"] == 40
    assert values["x.daily.previous.midpoint"] == 100
    assert values["x.daily.previous.level.midpoint.close_distance_price"] == 10
    assert values["x.market.m15.available"] is True
    assert values["x.market.m15.volatility.atr"] is None
    with pytest.raises(FrozenInstanceError):
        context.atr = 1
