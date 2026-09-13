from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sandbox.market_state.block_aggregation import OHLCBar
from sandbox.market_state.session_clock import BlockConfig
from sandbox.market_state.multitimeframe_context import (
    DEFAULT_CONTEXT_FRAMES, ContextFrame, build_multitimeframe_snapshot,
)
from sandbox.market_state import historical_multitimeframe as historical

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


def local(day=15, hour=0, minute=0, month=1):
    return datetime(2024, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


def bars(start=None, end=None, minutes=15):
    start = local(14) if start is None else start
    end = local(16) if end is None else end
    result = []
    while start < end:
        result.append(OHLCBar(start, 10, 12, 9, 11))
        start += timedelta(minutes=minutes)
    return result


def snapshots(data, **kwargs):
    return tuple(historical.iter_historical_multitimeframe_snapshots(data, **kwargs))


def frames_at(results, decision):
    return {f.frame_id: f for s in results if s.decision_time_utc == decision for f in s.frames}


def test_one_chronological_snapshot_per_bar():
    data = bars()
    result = snapshots(data)
    assert len(result) == len(data)
    assert [s.decision_time_utc for s in result] == sorted(s.decision_time_utc for s in result)
    for bar, snapshot in zip(data, result):
        assert snapshot.base_bar is bar
        assert snapshot.decision_time_utc == bar.open_time_utc + timedelta(minutes=15)
        assert all(f.expected_block_end_utc <= snapshot.decision_time_utc for f in snapshot.frames)
        assert all(f.candle is None or f.candle.block_end_utc <= snapshot.decision_time_utc for f in snapshot.frames)


def test_normal_scalar_equivalence_and_windows():
    data = bars()
    result = snapshots(data)
    selected = {local(), local(hour=10, minute=15), local(hour=12), local(hour=23, minute=45)}
    for snapshot in result:
        if snapshot.decision_time_utc in selected:
            assert snapshot == build_multitimeframe_snapshot(data, snapshot.decision_time_utc)
    at = frames_at(result, local(hour=10, minute=15))
    for name, start, end in [("NY_H1", local(hour=9), local(hour=10)),
        ("NY_H4", local(hour=4), local(hour=8)), ("NY_H3", local(hour=6), local(hour=9)),
        ("NY_D1", local(14), local())]:
        assert (at[name].expected_block_start_utc, at[name].expected_block_end_utc) == (start, end)
        assert at[name].candle is not None
    assert at["NY_H3"].candle.block_label == "C3"
    noon = frames_at(result, local(hour=12))
    assert noon["NY_H4"].candle.block_start_utc == local(hour=8)
    assert noon["NY_H4"].candle.block_end_utc == local(hour=12)
    assert noon["NY_H3"].candle.block_label == "C4"
    assert noon["NY_H3"].candle.block_start_utc == local(hour=9)
    assert noon["NY_H3"].candle.block_end_utc == local(hour=12)


def test_missing_window_then_recovery():
    data = [b for b in bars() if b.open_time_utc != local(hour=5)]
    result = snapshots(data)
    missing = frames_at(result, local(hour=10, minute=15))["NY_H4"]
    assert missing.expected_block_start_utc == local(hour=4)
    assert missing.candle is None
    recovered = frames_at(result, local(hour=12))["NY_H4"]
    assert recovered.candle is not None
    assert recovered.candle.block_start_utc == local(hour=8)


def test_future_replacements_do_not_change_earlier_snapshots():
    data = bars()
    cutoff = local(hour=10)
    modified = [b if b.open_time_utc < cutoff else OHLCBar(b.open_time_utc, 100, 120, 90, 110) for b in data]
    before = [s for s in snapshots(data) if s.decision_time_utc <= cutoff]
    after = [s for s in snapshots(modified) if s.decision_time_utc <= cutoff]
    assert before == after
    assert before == list(snapshots([b for b in data if b.open_time_utc < cutoff]))


def test_generator_consumed_once():
    data = bars()
    class Once:
        def __init__(self):
            self.calls = 0
        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield from data
    source = Once()
    assert snapshots(source) == snapshots(data)
    assert source.calls == 1


def test_empty():
    assert snapshots(iter(())) == ()


@pytest.mark.parametrize("bad", ["duplicate", "unordered", "misaligned", "wrong_type"])
def test_bad_bars(bad):
    data = bars()
    if bad == "duplicate":
        data = [data[0], data[0]]
    elif bad == "unordered":
        data.reverse()
    elif bad == "misaligned":
        data = [OHLCBar(local() + timedelta(seconds=1), 10, 12, 9, 11)]
    else:
        data = [object()]
    with pytest.raises(ValueError):
        snapshots(data)


@pytest.mark.parametrize("base", [0, -1, True, 15.0, "15"])
def test_invalid_base(base):
    with pytest.raises(ValueError):
        snapshots([], base_minutes=base)


@pytest.mark.parametrize("frames", [(DEFAULT_CONTEXT_FRAMES[0],) * 2, (object(),)])
def test_invalid_frames(frames):
    with pytest.raises(ValueError):
        snapshots([], frames=frames)


def test_generic_custom_frame_and_base():
    config = BlockConfig("UTC", time(6), 120, 12, tuple(f"B{i}" for i in range(12)))
    frames = (ContextFrame("CUSTOM", config),)
    data = bars(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC), minutes=30)
    result = snapshots(data, frames=frames, base_minutes=30)
    assert len(result) == len(data)
    assert result[-1] == build_multitimeframe_snapshot(data, result[-1].decision_time_utc, frames, 30)
    assert result[-1].frames[0].candle.base_bar_count == 4


@pytest.mark.parametrize("month,day", [(3, 10), (11, 3)])
def test_dst_scalar_equivalence(month, day):
    data = bars(local(day, month=month), local(day + 1, month=month))
    result = snapshots(data)
    # Include both repeated autumn hours and the spring jump, plus day close.
    selected = list(result[:20]) + [result[-1]]
    for snapshot in selected:
        assert snapshot == build_multitimeframe_snapshot(data, snapshot.decision_time_utc)


def test_aggregation_once_per_frame(monkeypatch):
    data = bars(local(10), local(16))
    original = historical.aggregate_completed_blocks
    calls = []
    def counted(all_bars, decision, config, base_minutes):
        calls.append((len(all_bars), decision, config, base_minutes))
        return original(all_bars, decision, config, base_minutes)
    monkeypatch.setattr(historical, "aggregate_completed_blocks", counted)
    result = snapshots(data)
    assert len(result) == len(data)
    assert len(calls) == len(DEFAULT_CONTEXT_FRAMES)
    assert [c[2] for c in calls] == [f.config for f in DEFAULT_CONTEXT_FRAMES]
    assert all(c[0] == len(data) and c[1] == data[-1].open_time_utc + timedelta(minutes=15) for c in calls)
