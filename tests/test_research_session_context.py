from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, time, timezone

import pytest
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.session_summary import SessionConfig, DEFAULT_SESSIONS, LONDON_SESSION, NEW_YORK_SESSION, session_window_for_date
from sandbox.market_state.research_anchor import GeometryResearchAnchor, ResearchAnchorKind
from sandbox.market_state.future_outcome import OutcomeAnchor
from sandbox.market_state import research_session_context as join

UTC = timezone.utc
START = datetime(2024, 1, 15, tzinfo=UTC)
STEP = timedelta(minutes=15)


def anchor(stamp=None):
    stamp = START + timedelta(hours=15, minutes=45) if stamp is None else stamp
    source = stamp - timedelta(days=2)
    return GeometryResearchAnchor(ResearchAnchorKind.TOUCH_TRANSITION, "CUSTOM", source.date(), "C1", 0,
        source, source + timedelta(hours=3), source + timedelta(hours=3), "LEVEL", 0, 10,
        stamp, 15, 0, "ABOVE_TOUCH_ABOVE", None, OutcomeAnchor("EVIDENCE", stamp, 11))


def bars(start=START - timedelta(days=2), end=START + timedelta(days=2)):
    result = []
    while start < end:
        result.append(OHLCBar(start, 10, 12, 9, 11))
        start += STEP
    return result


def test_winter_1045_ny_and_no_stale_fallback():
    result = join.join_research_anchor_sessions(anchor(), bars())
    assert result.available_session_ids == ("TOKYO",)
    assert result.active_session_ids == ("LONDON", "NEW_YORK")
    assert result.missing_completed_session_ids == ()
    assert result.all_session_summaries_available is False
    assert result.sessions[0].summary is not None
    assert all(s.summary is None and s.active_at_anchor for s in result.sessions[1:])


@pytest.mark.parametrize("stamp,active,target_day", [
    (START + timedelta(hours=8), True, START.date()),
    (START + timedelta(hours=17) - timedelta(microseconds=1), True, START.date()),
    (START + timedelta(hours=17), False, START.date()),
    (START + timedelta(hours=7), False, (START - timedelta(days=1)).date()),
])
def test_boundaries_and_before_start(stamp, active, target_day):
    result = join.join_research_anchor_sessions(anchor(stamp), bars(), (LONDON_SESSION,))
    slot, = result.sessions
    assert slot.active_at_anchor is active
    assert slot.target_local_session_date == target_day
    assert (slot.summary is None) is active
    assert result.all_session_summaries_available is (not active)


@pytest.mark.parametrize("hour,active,day_offset", [(2, True, -1), (7, False, -1), (23, True, 0)])
def test_custom_overnight(hour, active, day_offset):
    config = SessionConfig("NIGHT", "UTC", time(22), time(6))
    result = join.join_research_anchor_sessions(anchor(START + timedelta(hours=hour)), bars(), (config,))
    slot, = result.sessions
    assert slot.active_at_anchor is active
    assert slot.target_local_session_date == (START + timedelta(days=day_offset)).date()
    assert (slot.summary is None) is active


def test_missing_completed_no_stale_fallback():
    data = [b for b in bars() if b.open_time_utc != START + timedelta(hours=9)]
    result = join.join_research_anchor_sessions(anchor(START + timedelta(hours=18)), data, (LONDON_SESSION,))
    assert result.sessions[0].target_local_session_date == START.date()
    assert result.sessions[0].summary is None
    assert result.sessions[0].active_at_anchor is False
    assert result.missing_completed_session_ids == ("LONDON",)
    assert result.available_session_ids == ()
    assert result.all_session_summaries_available is False


def test_bulk_future_invariance_and_later_completion():
    early, late = anchor(), anchor(START + timedelta(hours=23))
    data = bars()
    only = join.join_research_anchor_sessions(early, data)
    bulk = join.join_research_anchors_sessions((late, early, early), data)
    assert [r.anchor for r in bulk] == [late, early, early]
    assert bulk[1] == bulk[2] == only
    assert bulk[0].sessions[1].summary is not None
    assert bulk[1].sessions[1].summary is None
    assert bulk[1].sessions[1].active_at_anchor is True
    changed = [b if b.open_time_utc <= early.evidence_end_utc else OHLCBar(b.open_time_utc, 100, 120, 90, 110) for b in data]
    assert join.join_research_anchors_sessions((early, late), changed)[0] == only


def test_generators_order_and_one_aggregation(monkeypatch):
    class Once:
        def __init__(self, values):
            self.values, self.calls = values, 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from self.values
    anchors, data, sessions = Once([anchor()] * 100), Once(bars()), Once(DEFAULT_SESSIONS[::-1])
    original = join.aggregate_completed_sessions
    calls = []
    def tracked(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(join, "aggregate_completed_sessions", tracked)
    result = join.join_research_anchors_sessions(anchors, data, sessions)
    assert len(result) == 100
    assert len(calls) == 1
    assert anchors.calls == data.calls == sessions.calls == 1
    assert [s.session_id for s in result[0].sessions] == ["NEW_YORK", "LONDON", "TOKYO"]
    assert result[0].active_session_ids == ("NEW_YORK", "LONDON")


def test_empty_no_aggregation(monkeypatch):
    def forbidden(*args):
        raise AssertionError("unexpected aggregation")
    monkeypatch.setattr(join, "aggregate_completed_sessions", forbidden)
    assert join.join_research_anchors_sessions([], []) == ()


@pytest.mark.parametrize("value", [0, -1, True, 15.0])
def test_invalid_base(value):
    with pytest.raises(ValueError):
        join.join_research_anchors_sessions([anchor()], [], base_minutes=value)


def test_invalid_inputs():
    with pytest.raises(TypeError):
        join.join_research_anchor_sessions(object(), [])
    with pytest.raises(TypeError):
        join.join_research_anchors_sessions([object()], [])
    with pytest.raises(TypeError):
        join.join_research_anchors_sessions([anchor()], [], (object(),))
    with pytest.raises(ValueError):
        join.join_research_anchors_sessions([anchor()], [], (LONDON_SESSION,) * 2)
    with pytest.raises(ValueError):
        join.join_research_anchors_sessions([replace(anchor(), base_minutes=30)], [])


@pytest.mark.parametrize("month", [1, 7, 3])
@pytest.mark.parametrize("config", [LONDON_SESSION, NEW_YORK_SESSION, SessionConfig("CUSTOM", "Australia/Sydney", time(9), time(16))])
def test_dst_and_generic_boundaries(month, config):
    stamp = datetime(2024, month, 20, 23, tzinfo=UTC)
    result = join.join_research_anchor_sessions(anchor(stamp), [], (config,))
    slot, = result.sessions
    window = session_window_for_date(config, slot.target_local_session_date)
    assert (slot.target_start_utc, slot.target_end_utc) == (window.start_utc, window.end_utc)


def test_slot_and_parent_validation_immutable():
    result = join.join_research_anchor_sessions(anchor(START + timedelta(hours=18)), bars(), (LONDON_SESSION,))
    slot = result.sessions[0]
    for changes in ({"summary": object()}, {"active_at_anchor": True}, {"session_id": "OTHER"},
        {"target_local_session_date": (START - timedelta(days=1)).date()},
        {"target_start_utc": START}, {"target_end_utc": START}):
        with pytest.raises(ValueError):
            replace(slot, **changes)
    for changes in ({"anchor": object()}, {"sessions": (object(),)}, {"sessions": (slot, slot)},
        {"available_session_ids": ()}, {"active_session_ids": ("LONDON",)},
        {"missing_completed_session_ids": ("LONDON",)}, {"all_session_summaries_available": False}):
        with pytest.raises((TypeError, ValueError)):
            replace(result, **changes)
    with pytest.raises(FrozenInstanceError):
        slot.summary = None
    with pytest.raises(FrozenInstanceError):
        result.sessions = ()
