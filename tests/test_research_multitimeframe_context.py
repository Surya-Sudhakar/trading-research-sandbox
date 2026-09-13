from dataclasses import FrozenInstanceError, replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.session_clock import BlockConfig
from sandbox.market_state.multitimeframe_context import ContextFrame, DEFAULT_CONTEXT_FRAMES
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor
from sandbox.market_state import research_multitimeframe_context as join

UTC = timezone.utc
NY = ZoneInfo("America/New_York")
STEP = timedelta(minutes=15)


def local(day=15, hour=0, minute=0):
    return datetime(2024, 1, day, hour, minute, tzinfo=NY).astimezone(UTC)


def bars():
    start, end = local(14), local(16)
    result = []
    while start < end:
        result.append(OHLCBar(start, 10, 12, 9, 11))
        start += STEP
    return result


def anchor(hour=10, minute=45, checkpoint=False):
    decision = local(hour=hour, minute=minute)
    return GeometryResearchAnchor(
        ResearchAnchorKind.UNTOUCHED_CHECKPOINT if checkpoint else ResearchAnchorKind.TOUCH_TRANSITION,
        "CUSTOM", local().astimezone(NY).date(), "C1", 0, local(), local(hour=3), local(hour=3),
        "LEVEL", 0, 10, decision, 15, None if checkpoint else 0,
        None if checkpoint else "ABOVE_TOUCH_ABOVE", int((decision-local(hour=3))/STEP) if checkpoint else None,
        OutcomeAnchor("EVIDENCE", decision, 11))


@pytest.mark.parametrize("checkpoint", [False, True])
def test_causal_join_and_completed_defaults(checkpoint):
    source = anchor(checkpoint=checkpoint)
    result = join.join_research_anchor_multitimeframe(source, bars())
    assert result.anchor is source
    assert result.snapshot.decision_time_utc == local(hour=10, minute=45)
    assert result.snapshot.base_bar.open_time_utc == local(hour=10, minute=30)
    assert result.snapshot.base_bar.close == source.outcome_anchor.reference_price
    assert result.missing_frame_ids == ()
    assert result.all_frames_available is True
    expected = {"NY_D1": (local(14), local()), "NY_H4": (local(hour=4), local(hour=8)),
                "NY_H1": (local(hour=9), local(hour=10)), "NY_H3": (local(hour=6), local(hour=9))}
    assert [f.frame_id for f in result.snapshot.frames] == list(expected)
    for frame in result.snapshot.frames:
        assert (frame.expected_block_start_utc, frame.expected_block_end_utc) == expected[frame.frame_id]
        assert frame.expected_block_end_utc <= source.evidence_end_utc
        assert frame.candle.block_end_utc <= source.evidence_end_utc


def test_missing_h4_preserved():
    data = [b for b in bars() if b.open_time_utc != local(hour=5)]
    result = join.join_research_anchor_multitimeframe(anchor(), data)
    h4 = next(f for f in result.snapshot.frames if f.frame_id == "NY_H4")
    assert h4.candle is None
    assert h4.expected_block_start_utc == local(hour=4)
    assert result.snapshot.base_bar is not None
    assert result.missing_frame_ids == ("NY_H4",)
    assert result.all_frames_available is False


def test_missing_base_no_stale_fallback():
    data = [b for b in bars() if b.open_time_utc != local(hour=10, minute=30)]
    with pytest.raises(ValueError, match="exact snapshot"):
        join.join_research_anchor_multitimeframe(anchor(), data)


def test_reference_price_mismatch():
    source = anchor()
    source = replace(source, outcome_anchor=replace(source.outcome_anchor, reference_price=12))
    with pytest.raises(ValueError, match="reference price"):
        join.join_research_anchor_multitimeframe(source, bars())


def test_non_grid_anchor_no_exact_snapshot():
    source = anchor()
    stamp = source.evidence_end_utc + timedelta(seconds=1)
    source = replace(source, evidence_end_utc=stamp, outcome_anchor=replace(source.outcome_anchor, anchor_time_utc=stamp))
    with pytest.raises(ValueError, match="exact snapshot"):
        join.join_research_anchor_multitimeframe(source, bars())


def test_future_invariance():
    source = anchor()
    data = bars()
    expected = join.join_research_anchor_multitimeframe(source, data)
    past = [b for b in data if b.open_time_utc < source.evidence_end_utc]
    assert join.join_research_anchor_multitimeframe(source, past) == expected
    changed = [b if b.open_time_utc < source.evidence_end_utc else OHLCBar(b.open_time_utc, 100, 120, 90, 110) for b in data]
    assert join.join_research_anchor_multitimeframe(source, changed) == expected


def test_order_duplicates_and_shared_snapshot():
    sources = (anchor(), anchor(9, 0), anchor(12, 15), replace(anchor(), level_id="OTHER"))
    result = join.join_research_anchors_multitimeframe(sources, bars())
    assert [r.anchor for r in result] == list(sources)
    assert result[0].snapshot is result[3].snapshot


def test_empty_no_rebuild(monkeypatch):
    def forbidden(*args):
        raise AssertionError("must not build snapshots")
    monkeypatch.setattr(join, "iter_historical_multitimeframe_snapshots", forbidden)
    assert join.join_research_anchors_multitimeframe([], []) == ()


@pytest.mark.parametrize("bad", [0, -1, True, 15.0])
def test_invalid_base(bad):
    with pytest.raises(ValueError):
        join.join_research_anchors_multitimeframe([anchor()], bars(), base_minutes=bad)


def test_invalid_anchor_and_base_mismatch():
    with pytest.raises(TypeError):
        join.join_research_anchor_multitimeframe(object(), [])
    with pytest.raises(TypeError):
        join.join_research_anchors_multitimeframe([object()], [])
    with pytest.raises(ValueError, match="mismatch"):
        join.join_research_anchors_multitimeframe([replace(anchor(), base_minutes=30)], bars())


@pytest.mark.parametrize("frames", [(object(),), (DEFAULT_CONTEXT_FRAMES[0],) * 2])
def test_invalid_frames(frames):
    with pytest.raises((TypeError, ValueError)):
        join.join_research_anchors_multitimeframe([anchor()], bars(), frames)


def test_generators_and_single_historical_call(monkeypatch):
    class Once:
        def __init__(self, values):
            self.values, self.calls = values, 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from self.values
    sources, data, frames = Once([anchor()] * 100), Once(bars()), Once(DEFAULT_CONTEXT_FRAMES)
    original = join.iter_historical_multitimeframe_snapshots
    calls = []
    def tracked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(join, "iter_historical_multitimeframe_snapshots", tracked)
    result = join.join_research_anchors_multitimeframe(sources, data, frames)
    assert len(result) == 100
    assert len(calls) == 1
    assert sources.calls == data.calls == frames.calls == 1
    assert all(r.snapshot is result[0].snapshot for r in result)


def test_custom_frame():
    config = BlockConfig("UTC", time(), 120, 12, tuple(f"B{i}" for i in range(12)))
    result = join.join_research_anchor_multitimeframe(anchor(), bars(), (ContextFrame("CUSTOM", config),))
    assert result.snapshot.frames[0].frame_id == "CUSTOM"
    assert result.all_frames_available is True


def test_dataclass_consistency_and_immutable():
    result = join.join_research_anchor_multitimeframe(anchor(), bars())
    for changes in ({"anchor": object()}, {"snapshot": object()}, {"missing_frame_ids": ("NY_H4",)}, {"all_frames_available": False}):
        with pytest.raises((TypeError, ValueError)):
            replace(result, **changes)
    for changes in ({"decision_time_utc": local(hour=12)}, {"base_bar": None},
        {"base_bar": replace(result.snapshot.base_bar, open_time_utc=local(hour=10, minute=45))}):
        with pytest.raises(ValueError):
            replace(result, snapshot=replace(result.snapshot, **changes))
    with pytest.raises(FrozenInstanceError):
        result.all_frames_available = False
